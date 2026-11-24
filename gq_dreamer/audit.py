"""Dream audit (diagnostic only): plays the same real states forward in the dream and in the real game with the same
greedy policy and compares the returns per reward stream (memory events, standard reward, curiosity), the event rates,
the decoded frames against the real ones, the map / battle flag / explored grid / event memory read from the dreamed
states and the reward heads' error. The special starts are "pre-event" ones (a new memory bit a few steps later)."""
import math

import numpy as np
import torch

AUDIT_STREAMS = ("event", "standard", "curiosity")   # event: +50 per new memory bit (dreams: sparse head x P(gate))
AUDIT_KINDS = ("uniform", "pre_event")
PRE_EVENT_SHARE = 0.3                                # share of pre-event starts
FRAME_STEPS = (1, 5, 10, 15, 25)                     # decoded-frame error at these rollout steps (those <= horizon)
HEAD_STEPS = 5                                       # reward-head error over the first rollout steps
GROUNDING = ("map", "battle", "grid", "memory")      # labels read from the dreamed states, compared with the game
_KIND_LABELS = {"uniform": "uniform", "pre_event": "pre-event starts"}

# metrics CSV columns (trainer.Trainer): (header, source, key); empty in a round without an audit
AUDIT_COLUMNS = [
    ("auditStarts", "audit", "starts"), ("auditSeconds", "audit", "seconds"),
    ("auditStartsPreEvent", "audit", "starts_pre_event"),
    *[(f"audit_{g}_{side}", "audit", f"{g}_{side}") for g in AUDIT_STREAMS for side in ("dream", "real", "gap")],
    *[(f"audit_{g}_pre_event_{side}", "audit", f"{g}_pre_event_{side}") for g in AUDIT_STREAMS
      for side in ("dream", "real", "gap")],
    ("audit_event_rate_dream", "audit", "event_rate_dream"), ("audit_event_rate_real", "audit", "event_rate_real"),
    ("audit_event_rate_pre_event_dream", "audit", "event_rate_pre_event_dream"),
    ("audit_event_rate_pre_event_real", "audit", "event_rate_pre_event_real"),
    ("audit_frame_err_0", "audit", "frame_err_0"), ("audit_sprite_err_0", "audit", "sprite_err_0"),
    *[(f"audit_frame_err_{s}", "audit", f"frame_err_{s}") for s in FRAME_STEPS],
    *[(f"audit_sprite_err_{s}", "audit", f"sprite_err_{s}") for s in FRAME_STEPS],
    *[(f"audit_{g}_acc", "audit", f"{g}_acc") for g in GROUNDING],
    *[(f"audit_head_err_{g}", "audit", f"head_err_{g}") for g in AUDIT_STREAMS],
]


def audit_enabled(cfg):
    """True when the dream audit runs (cfg.audit on and cfg.audit_every > 0)."""
    return bool(cfg.audit) and int(cfg.audit_every) > 0


class SnapshotPool:
    """Env snapshots of the play just finished, one every `every` agent steps of an episode, labelled when it ends."""

    def __init__(self, every):
        self.every = max(int(every), 0)                 # 0: no snapshots (audit off)
        self.items = []                                 # labelled snapshots of finished episodes
        self._open = {}                                 # env index -> [(step, snapshot)] of its running episode

    def __len__(self):
        return len(self.items)

    def clear(self):
        """Forgets every snapshot (an audit uses the latest play only)."""
        self.items, self._open = [], {}

    def due(self, step):
        """True when the observation of episode step `step` (its row index) gets a snapshot."""
        return self.every > 0 and step > 0 and step % self.every == 0

    def add(self, env, step, snapshot):
        """Adds env `env`'s snapshot of the state whose observation is row `step` of its running episode."""
        self._open.setdefault(env, []).append((int(step), snapshot))

    def drop_open(self):
        """Forgets the snapshots of episodes that were not stored."""
        self._open = {}

    def close_episode(self, env, start_row, episode_id, length, event_rows, event_window):
        """Labels the snapshots of env's stored episode (replay rows start_row.., id episode_id, `length` rows) as
        pre-event (an observation with a new memory bit, rows event_rows, event_window[0]..[1] steps later) or not.
        The snapshot of the episode's final state (its observation is not stored) is dropped."""
        lo, hi = event_window
        events = np.asarray(sorted(event_rows), dtype=np.int64)
        for t, snap in self._open.pop(env, []):
            if t >= length:
                continue
            d = events - t
            pre = bool(((d >= lo) & (d <= hi)).any()) if events.size else False
            self.items.append({"snapshot": snap, "step": t, "row": int(start_row) + t, "episode_id": int(episode_id),
                               "pre_event": pre})

    def usable(self, buffer, context):
        """Labelled snapshots whose replay rows (the start and up to `context` earlier steps) still hold their
        episode."""
        out, n = [], len(buffer)
        for it in self.items:
            rows = [it["row"] % buffer.capacity, (it["row"] - min(it["step"], context)) % buffer.capacity]
            if all(r < n and int(buffer.episode_ids[r]) == it["episode_id"] for r in rows) \
                    and int(buffer.ep_steps[rows[0]]) == it["step"]:
                out.append(it)
        return out


def select_starts(items, n, rng):
    """Picks the starts -> [(index into items, kind)]: round(PRE_EVENT_SHARE x n) pre-event, the rest uniform (also
    filling a missing kind); never a snapshot twice. rng: a numpy Generator."""
    n = min(int(n), len(items))
    taken = np.zeros(len(items), dtype=bool)
    picks = []
    for kind, flag, want in (("pre_event", "pre_event", int(round(PRE_EVENT_SHARE * n))), ("uniform", None, n)):
        cands = [i for i, it in enumerate(items) if not taken[i] and (flag is None or it[flag])]
        k = min(want, n - len(picks), len(cands))
        if k <= 0:
            continue
        for i in rng.choice(np.asarray(cands, dtype=np.int64), size=k, replace=False):
            taken[i] = True
            picks.append((int(i), kind))
    return picks


def frame_steps(horizon):
    """The FRAME_STEPS within the horizon, plus the horizon itself."""
    return sorted({s for s in FRAME_STEPS if s <= horizon} | {int(horizon)})


def audit_metrics(dream, real, kinds, discount):
    """Audit metrics of N starts over H steps.

    dream: "r" [N, H, 3] per-stream rewards (event, standard, curiosity), "events" [N, H] expected events (P(gate)
    x alive: at most 1 per dream, so their sum is the dream's probability of an event),
    "frame_err" / "sprite_err" [N, H + 1] (step 0: the start frame's reconstruction), "<g>_hit" [N, H] for the
    GROUNDING labels (dreamed states vs the real game); real: "r" [N, H, 3], "events" [N, H], "steps" [N] (steps taken
    before the episode ended). Returns per stream the discounted return (no death: every dream step counts; real over
    the steps taken) and the gap, the share of rollouts with a new memory bit, the frame errors, the grounding
    accuracies and the reward heads' error over HEAD_STEPS; overall and for pre-event starts. The dream's event rate
    is the mean of its event probability (sum of the expected events, capped at 1); the game's the share with one."""
    rd = torch.as_tensor(dream["r"]).detach().float().cpu()
    rr = torch.as_tensor(real["r"]).float().cpu()
    ev_d = torch.as_tensor(dream["events"]).detach().float().cpu()
    ev_r = torch.as_tensor(real["events"]).float().cpu()
    steps = torch.as_tensor(np.asarray(real["steps"]), dtype=torch.long)
    kinds = torch.as_tensor(np.asarray(kinds), dtype=torch.long)
    N, H, G = rd.shape
    k = torch.arange(H)
    disc = float(discount) ** k.double()
    taken = (k[None] < steps[:, None]).double()
    ret_d = (disc[None, :, None] * rd.double()).sum(1)                           # [N, G]
    ret_r = ((disc * taken)[..., None] * rr.double()).sum(1)
    any_d = ev_d.sum(1).clamp(max=1.0).double()                                  # P(event) of each dream
    any_r = ((ev_r.double() * taken).sum(1) > 0).double()
    out = {"starts": float(N), "horizon": float(H)}
    for name, sel in (("", torch.ones(N, dtype=torch.bool)), ("_pre_event", kinds == AUDIT_KINDS.index("pre_event"))):
        if name:
            out[f"starts{name}"] = float(sel.sum())
        if not bool(sel.any()):
            continue
        for gi, g in enumerate(AUDIT_STREAMS[:G]):
            d, r = float(ret_d[sel, gi].mean()), float(ret_r[sel, gi].mean())
            out.update({f"{g}{name}_dream": d, f"{g}{name}_real": r, f"{g}{name}_gap": d - r})
        out[f"event_rate{name}_dream"] = float(any_d[sel].mean())
        out[f"event_rate{name}_real"] = float(any_r[sel].mean())
    fe, se = (torch.as_tensor(dream[key]).float().cpu() for key in ("frame_err", "sprite_err"))
    out["frame_err_0"] = float(fe[:, 0].mean())
    out["sprite_err_0"] = float(se[:, 0].nanmean()) if bool((~se[:, 0].isnan()).any()) else float("nan")
    for s in frame_steps(H):
        running = steps >= s
        if not bool(running.any()):
            continue
        out[f"frame_err_{s}"] = float(fe[running, s].mean())
        v = se[running, s]
        out[f"sprite_err_{s}"] = float(v.nanmean()) if bool((~v.isnan()).any()) else float("nan")
    m = taken.float()
    for key in GROUNDING:
        if f"{key}_hit" in dream:
            hit = torch.as_tensor(dream[f"{key}_hit"]).float().cpu()
            out[f"{key}_acc"] = float((hit * m).sum() / m.sum().clamp(min=1.0))
    mh = taken[:, :min(HEAD_STEPS, H)]
    if float(mh.sum()) > 0:
        for gi, g in enumerate(AUDIT_STREAMS[:G]):
            err = (rd[:, :mh.shape[1], gi] - rr[:, :mh.shape[1], gi]).abs().double()
            out[f"head_err_{g}"] = float((err * mh).sum() / mh.sum())
    return out


def audit_lines(m):
    """The audit block of the round summary ([] without an audit)."""
    if not m or not m.get("starts"):
        return []

    def pct(v):
        return "–" if v is None or v != v else f"{100 * v:.0f}%"

    def val(v):
        return "–" if v is None or v != v else f"{v:+.2f}"

    def num(v, d=3):
        return "–" if v is None or v != v else f"{v:.{d}f}"

    H = int(m.get("horizon", FRAME_STEPS[0]))
    lines = [f"DREAM AUDIT · {int(m['starts'])} real states ({int(m.get('starts_pre_event', 0))} pre-event), each "
             f"played {H} steps in the dream and in the game ({m.get('seconds', 0.0):.1f} s)",
             f"  {f'return over {H} steps':<24}{'dream':>8}  {'real':>8}  {'gap':>8}   gap = dream − real "
             f"(+ = the dream is too optimistic)"]
    for g in AUDIT_STREAMS:
        if f"{g}_dream" in m:
            pre = (f"   pre-event {val(m.get(f'{g}_pre_event_dream'))} / {val(m.get(f'{g}_pre_event_real'))}"
                   if f"{g}_pre_event_dream" in m else "")
            lines.append(f"    {g:<22}{val(m[f'{g}_dream']):>8}  {val(m.get(f'{g}_real')):>8}  "
                         f"{val(m[f'{g}_dream'] - m.get(f'{g}_real', float('nan'))):>8}{pre}")
    pre = (f"   pre-event {pct(m.get('event_rate_pre_event_dream'))} / {pct(m.get('event_rate_pre_event_real'))}"
           if "event_rate_pre_event_dream" in m else "")
    lines.append(f"  {'new memory bit':<24}{pct(m.get('event_rate_dream')):>8}  {pct(m.get('event_rate_real')):>8}"
                 f"{'':>10}{pre}")
    errs = " · ".join(f"{s} {num(m.get(f'frame_err_{s}'))} ({num(m.get(f'sprite_err_{s}'))})" for s in frame_steps(H))
    lines.append(f"  {'frame RMSE (sprites)':<24}start {num(m.get('frame_err_0'))} "
                 f"({num(m.get('sprite_err_0'))}) · {errs}")
    lines.append(f"  {'dreamed states':<24}map acc {num(m.get('map_acc'))} · battle acc {num(m.get('battle_acc'))} · "
                 f"grid acc {num(m.get('grid_acc'))} · memory exact {num(m.get('memory_acc'))} · head error "
                 f"({HEAD_STEPS} steps) event {num(m.get('head_err_event'))} / standard "
                 f"{num(m.get('head_err_standard'))} / curiosity {num(m.get('head_err_curiosity'))}")
    return lines


def rmse(a, b, mask=None):
    """Per-frame RMSE of images [N, 3, H, W] (mask [N, H, W]: only those pixels; NaN where a frame has none)."""
    sq = (a.float() - b.float()).square().mean(1)                                  # [N, H, W]
    if mask is None:
        return sq.flatten(1).mean(1).sqrt()
    m = mask.float()
    n = m.flatten(1).sum(1)
    v = (sq * m).flatten(1).sum(1) / n.clamp(min=1.0)
    return torch.where(n > 0, v.sqrt(), torch.full_like(v, math.nan))
