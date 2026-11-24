"""GQ-Dreamer agent for Pokémon Red: world-model training, imagination (actor-critic), acting in the game and play
rounds.

The networks it builds are in model.py; this file runs and trains them.

The agent state is s = [h, z, m]: the TSSM state h, the one-hot latent z and the explicit 144-bit event memory m. The
memory head M (tokenizer.memory_head) reads z in three places:
  (1) on posterior latents of real frames: M(z_t) vs m_t trains the encoder (z must hold the memory) and M;
  (2) on the prior's sample of z_{t+1} (straight-through, M frozen): M vs m_{t+1} trains the prior and the TSSM to
      predict when a bit turns on;
  (3) in dreams it only says WHICH bits turn on; WHETHER an event happens is the event gate's decision.
Dream events: two heads read [h_{t+1}, z_{t+1}, m_t] (the memory before the
transition): the event gate head (Bernoulli: "a memory bit turns on in this transition", like a continue head for the
+50 events) and the sparse reward head (the size of that reward, trained on event transitions only). A dream step pays
standard reward head + sparse reward head x P(gate) x alive (alive: no event yet in this dream, so a dream pays at most
one event); the gate's sample decides whether m changes, and then the bits of M(z_{t+1}) above memory_threshold that are
not in m_t turn on (none: M's most likely unset bit). A bit already in m never pays again, and z forgetting a bit cannot
unset m. Both heads train on the prior's one-step samples (event_dreamlike 1: what a dream sees, real targets), so the
gate learns how often the prior's samples look like an event that did not happen: no rule on a sampled latent pays +50.
Loss (2) trains the prior and the TSSM (and, through the TSSM's z input, the encoder, as the KL does); never M.
Death is not modelled: no continue head, every dream runs its full horizon.

Exploration is the curiosity stream only: a new square pays its map's novelty (env), the explored grid is part of the
encoder input, and in dreams the grid head's "square already visited" gates the predicted curiosity. Cross-episode
novelty: the env scales a square's value by its rarity across the run's episodes; the agent keeps the counts
(novelty_tiles / novelty_maps), pushes them to the envs before each play round and merges their visits after it.
Every env plays the policy and every episode runs its whole step budget.

Actor: "pmpo_sign" = PMPO on the sign of each stream's raw advantage
A = lambda - V, one loss per stream (curiosity x curiosity_scale), an entropy bonus and KL(pi || actor_prior), a slow
EMA copy of the actor; scale-free, so the advantage normalisation only matters for "reinforce" (REINFORCE on ext +
curiosity_scale x cur, each stream "std": centred and divided by its batch std, or "range": divided by max(floor, EMA of
its P95 - P5 return range)).

World model: losses weighted around the dynamics KL, HL-Gauss targets, 6-layer encoder / TSSM 512 x 6, a gradient-share
log and the dream audit. z enters the TSSM, the prior and the heads with its gradient, so besides reconstruction,
grounding, the memory head and kl_rep the encoder is also trained by the dynamics KL, memory loss (2) and the reward /
grid heads (the curiosity head reads detached states unless curiosity_grad)."""
import contextlib
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audit import AUDIT_KINDS, SnapshotPool, audit_enabled, audit_metrics, rmse, select_starts
from .checkpoint import CheckpointMixin
from .config import POKEMON_CONFIG, config_to_dict
from .game import GRID_CENTER_INDEX, GRID_DIM, GRID_INPUT_DIM, MEMORY_DIM, MEMORY_REWARD, REC_BATTLE, REC_UI
from .model import (GROUNDING_GROUPS, TSSM, Actor, Critic, EventGateHead, GraphedDreamStep, GridHead, PriorHead,
                    Tokenizer, TwoHotEncoding, TwoHotHead, class_usage, entropy_categorical, gumbel_like,
                    kl_categorical, memory_bce, memory_metrics, sample_onehot, unimix_logprobs)
from .observations import (IMG_H, IMG_W, SIDE_GROUPS, frame_rgb, grid_input, obs_to_tensors, ram_fields, sample_hide,
                           sprite_pixel_mask, stored_grid, unpack_bits)
from .replay import BatchPrefetcher, ReplayBuffer, with_batch_extras
from .train_utils import (DevicePercentileClipper, LossHarmonizer, PercentileClipper, PhaseTimer, actor_metrics,
                          compile_available, compile_module, eager_modules, explained_variance, frozen, lambda_returns,
                          make_grad_scaler, masked_mean, pmpo_loss, reinforce_loss, resolve_amp_dtype, setup_compile,
                          weighted_moments)

torch.set_float32_matmul_precision("high")

ADV_STD_MIN = 1e-3                                                         # floor of each advantage stream's std


class GQDreamer(CheckpointMixin):
    """GroundQuery-Dreamer: query-fusion tokenizer + TSSM world model with an explicit event memory, actor-critic
    trained in imagination."""
    OBS_KEYS = ("frame", "palette", "grid", "map_id", "ram", "memory")     # what the encoder reads
    CURIOSITY_POS_WEIGHT = 5.0                                             # weight of novelty (> 0) curiosity targets
    # world-model loss terms, each multiplied by its fixed weight (they sum to world_model_loss); round averages
    WM_LOSS_TERMS = ("pixel",) + GROUNDING_GROUPS + ("memory", "memory_prior", "kl_dyn", "kl_rep", "rollout",
                                                      "standard_reward", "event_gate", "sparse_reward", "grid",
                                                      "curiosity")
    HARMONY_GROUPS = ("pixel", "grounding", "memory", "kl", "rollout", "reward", "grid")
    GRAD_SHARE_GROUPS = ("tssm", "encoder")                                # parameter groups of the gradient-share log

    def __init__(self, cfg=POKEMON_CONFIG, device="cuda", envs=None, verbose=True):
        c = self.cfg = cfg
        assert c.seq_len > c.context_length, "training sequences must be longer than the context window"
        assert c.rollout_steps == 0 or c.seq_len >= c.rollout_steps + 2, "the rollout needs rollout_steps + 2 frames"
        self.device = torch.device(device)
        self.envs = envs
        self.verbose = verbose
        self.action_dim = c.action_dim
        self.n_latents, self.classes = c.n_latents, c.classes
        self.latent_dim = c.n_latents * c.classes
        self.tssm_dim = c.tssm_dim
        self.mem_dim = MEMORY_DIM
        self.state_dim = c.tssm_dim + self.latent_dim + self.mem_dim      # s = [h, flat one-hot z, memory bits]

        # run state
        self.total_num_episodes = self.total_num_steps = self.total_num_updates = 0
        self.last_play_stats = {}
        self.last_drift = float("nan")
        self._probe = self._probe_ref = None
        self._prefetcher = None
        self._wm_loss_acc = None                                           # round sums of the weighted loss terms
        self._dream_graph = None
        self._use_dream_graph = c.dream_cuda_graph
        self._act_graphs = {}                                              # n envs -> GraphedActStep
        self._use_act_graph = c.act_cuda_graph
        self._pending_prio = None                                          # surprise priorities of the last update
        self._save_thread = None                                           # background checkpoint writer
        self._resumed = False
        self.last_grad_shares = {}                                         # newest per-term gradient norms / shares
        self._grad_share_ok = True
        self.audit_snapshots = SnapshotPool(c.audit_snapshot_every if audit_enabled(c) else 0)   # audit starts
        # cross-episode novelty: (map, x, y) / map id -> episodes that visited it, all envs, whole run;
        # pushed to the envs before every play round, saved in checkpoints
        self.novelty_tiles, self.novelty_maps = {}, {}
        self._audit_env = None                                             # the audit's own emulator (fallback)
        self.timer = PhaseTimer(self.device)
        # 255-bin targets of the reward / curiosity heads and the critics (HL-Gauss or two-hot)
        self.two_hot = TwoHotEncoding(device=self.device, target=c.value_target, sigma_ratio=c.hl_sigma_ratio)
        if c.actor_objective not in ("pmpo_sign", "reinforce"):
            raise ValueError(f"actor_objective must be \"pmpo_sign\" or \"reinforce\", not {c.actor_objective!r}")
        if c.adv_norm not in ("range", "std"):
            raise ValueError(f"adv_norm must be \"range\" (DreamerV3) or \"std\", not {c.adv_norm!r}")
        self.adv_range = torch.zeros(2, device=self.device)               # "range": EMA of the ext / cur return range
        self.adv_range_init = False                                        # set by the first update (or a checkpoint)

        self._build_networks()
        self._build_optimizers()
        self._compile_networks()
        self._set_file_names()
        num_envs = len(envs) if envs is not None else 1
        self.buffer = ReplayBuffer(
            capacity=c.buffer_size, num_envs=num_envs, frames_per_step=c.frames_per_step,
            recent_sample_prob=c.recent_sample_prob, reward_sample_prob=c.reward_sample_prob,
            reward_threshold=c.reward_threshold, reward_sample_per_event=c.reward_sample_per_event,
            loss_sample_prob=c.loss_sample_prob, loss_priority_alpha=c.loss_priority_alpha)
        self._print_setup()

    # --- Setup ------------------------------------------------------------------------------------------------------
    def _build_networks(self):
        """World model (tokenizer, TSSM, prior, heads) and behaviour (actor, critics, EMA copies)."""
        c, dev, S = self.cfg, self.device, self.state_dim
        self.tokenizer = Tokenizer(c.n_latents, c.classes, c.enc_dim, c.enc_depth, c.enc_heads,
                                   dec_chans=tuple(c.dec_chans), head_hidden=c.head_hidden,
                                   memory_hidden=c.memory_hidden, unimix=c.unimix,
                                   grad_checkpoint=c.enc_grad_checkpoint, enc_patch_squares=c.enc_patch_squares,
                                   enc_pos_inject=c.enc_pos_inject, enc_fast_attn=c.enc_fast_attn,
                                   dec_sac=c.dec_sac, dec_targets_once=c.dec_targets_once,
                                   enc_last_queries=c.enc_last_queries).to(dev)
        # sink_max_pos: training sequences hold positions 0 .. seq_len - 1; acting counts higher
        self.tssm = TSSM(self.latent_dim, c.action_dim, c.tssm_dim, c.tssm_layers, c.tssm_heads, c.tssm_kv_heads,
                         c.tssm_ffn, c.context_length, attn_gate=c.attn_gate, mem_dim=self.mem_dim,
                         sink_max_pos=c.seq_len - 1).to(dev)
        self.prior = PriorHead(c.tssm_dim, c.n_latents, c.classes, hidden=c.mlp_dim, layers=c.prior_layers,
                               unimix=c.unimix).to(dev)
        self.standard_reward_head = TwoHotHead(S, hidden=c.mlp_dim).to(dev)
        # dream events: both read [h_{t+1}, z_{t+1}, m_t] (event_state)
        self.event_gate_head = EventGateHead(S, hidden=c.mlp_dim, init_logit=c.event_gate_init).to(dev)
        self.sparse_reward_head = TwoHotHead(S, hidden=c.mlp_dim).to(dev)
        self.curiosity_head = TwoHotHead(S, hidden=c.mlp_dim, zero_bias_bin=127).to(dev)
        self.grid_head = GridHead(S, hidden=c.mlp_dim).to(dev)            # explored 5x5 grid from s = [h, z, m]

        self.actor = Actor(c.action_dim, S, c.mlp_dim).to(dev)
        self.critic = Critic(S, mlp_dim=c.mlp_dim).to(dev)
        self.curiosity_critic = Critic(S, mlp_dim=c.mlp_dim).to(dev)
        self.critic_target = copy.deepcopy(self.critic)
        self.curiosity_critic_target = copy.deepcopy(self.curiosity_critic)
        self.actor_prior = copy.deepcopy(self.actor)                       # pmpo_sign: slow EMA copy (KL trust region)
        for m in (self.critic_target, self.curiosity_critic_target, self.actor_prior):
            m.requires_grad_(False)
        self.harmony = LossHarmonizer(self.HARMONY_GROUPS).to(dev)         # always built: one checkpoint layout

    def _build_optimizers(self):
        """One AdamW for the whole world model, Adam for actor, critics and loss weights."""
        c = self.cfg
        fused = {"fused": True} if c.sync_free and self.device.type == "cuda" else {}   # steps skipped on the GPU
        self.wm_params = self.optimizer_params("wm_optimizer")
        self.wm_optimizer = torch.optim.AdamW(self.wm_params, lr=c.wm_lr, weight_decay=0.01, **fused)
        clipper = DevicePercentileClipper if c.sync_free else PercentileClipper
        self.wm_clipper = clipper(c.clip_percentile, c.clip_window, c.clip_warmup, c.clip_initial,
                                  **({"device": self.device} if c.sync_free else {}))
        self.actor_optimizer = torch.optim.Adam(self.optimizer_params("actor_optimizer"), lr=c.actor_lr, **fused)
        self.critic_optimizer = torch.optim.Adam(self.optimizer_params("critic_optimizer"), lr=c.critic_lr, **fused)
        self.curiosity_critic_optimizer = torch.optim.Adam(self.optimizer_params("curiosity_critic_optimizer"),
                                                           lr=c.critic_lr, **fused)
        self.harmony_optimizer = torch.optim.Adam(self.optimizer_params("harmony_optimizer"), lr=c.harmony_lr)
        self.amp_dtype = resolve_amp_dtype(self.device, c.mixed_precision)
        self.cache_dtype = self.amp_dtype or torch.float32                 # KV caches of acting / dreams
        self.scalers = {name: make_grad_scaler(self.device, self.amp_dtype == torch.float16)
                        for name in self.OPTIMIZERS}

    def _compile_networks(self):
        """cfg.compile: torch.compile of the encoder's checkpointed layers, the decoder's checkpointed chunks, the MLP
        heads / actor / critics / prior / memory head and the TSSM's teacher-forced blocks. Parameters and state_dict
        keys are unchanged; the hand-captured CUDA graphs (dreams, acting) run the eager modules. The first updates
        compile (minutes; cached on disk afterwards)."""
        self.compiled = False
        if not self.cfg.compile:
            return
        ok, why = compile_available(self.device)
        if not ok:
            print(f"\033[93m    [setup] compile=True but {why}: running eagerly\033[0m")
            return
        setup_compile()
        self.tokenizer.encoder.compile_layers()
        self.tokenizer.compile_decoder()
        for name in ("actor", "actor_prior", "critic", "curiosity_critic", "critic_target", "curiosity_critic_target",
                     "standard_reward_head", "event_gate_head", "sparse_reward_head", "curiosity_head", "grid_head",
                     "prior"):
            compile_module(getattr(self, name).net)
        compile_module(self.tokenizer.memory_head.net)
        for blk in self.tssm.blocks:
            blk.forward_train = torch.compile(blk.forward_train)
        self.compiled = True

    def _print_setup(self):
        """One-line model summary."""
        c = self.cfg

        def count(*mods):
            return sum(p.numel() for m in mods for p in m.parameters()) / 1e6
        tk = self.tokenizer
        heads = count(self.standard_reward_head, self.event_gate_head, self.sparse_reward_head, self.curiosity_head,
                      self.grid_head)
        behaviour = count(self.actor, self.critic, self.curiosity_critic)
        amp = str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "off (fp32)"
        print(f"\033[90m    [setup] GQ-Dreamer Pokémon {self.ARCH_VERSION} \"{c.config_name}\" | encoder "
              f"{count(tk.encoder):.1f}M ({IMG_H}x{IMG_W} -> {tk.encoder.n_image_tokens} image tokens + "
              f"{len(SIDE_GROUPS)} vector tokens incl. memory, {c.enc_dim} x {c.enc_depth}) | decoder "
              f"{count(tk.decoder):.1f}M -> {IMG_H}x{IMG_W} | grounding heads {count(tk.heads):.2f}M | memory head M "
              f"{count(tk.memory_head):.2f}M | TSSM {count(self.tssm):.1f}M ({c.tssm_dim} x {c.tssm_layers}, input "
              f"[z, a, m]) | prior {count(self.prior):.1f}M | heads {heads:.1f}M | actor + critics {behaviour:.1f}M "
              f"| z = {c.n_latents} x {c.classes} one-hot | s = [h({c.tssm_dim}), z({self.latent_dim}), "
              f"m({self.mem_dim})] | batch {c.batch_size} x {c.seq_len}, {c.recon_frames or 'all'} frames decoded per "
              f"update | death not modelled | mixed precision: {amp}\033[0m")
        rollout = f"open-loop rollout {c.rollout_steps} steps" if c.rollout_steps > 0 else "no rollout loss"
        if c.actor_objective == "pmpo_sign":
            adv = (f"actor: PMPO sign, one loss per stream (curiosity x {c.curiosity_scale:g}), alpha {c.pmpo_alpha:g}, "
                   f"entropy {c.pmpo_entropy_scale:g}, KL to the EMA actor x {c.pmpo_prior_beta:g} (rate "
                   f"{c.pmpo_prior_rate:g})")
        elif c.adv_norm == "range":
            adv = (f"actor: REINFORCE, per-stream return range (DreamerV3, floors {c.adv_floor_ext:g} / "
                   f"{c.adv_floor_cur:g}), curiosity x {c.curiosity_scale:g}, entropy {c.reinforce_entropy_scale:g}")
        else:
            adv = (f"actor: REINFORCE, per-stream std, curiosity x {c.curiosity_scale:g}, entropy "
                   f"{c.reinforce_entropy_scale:g}")
        audit = (f"dream audit every {c.audit_every} rounds ({c.audit_starts} starts x {c.audit_horizon} steps)"
                 if audit_enabled(c) else "dream audit off")
        print(f"\033[90m    [setup] losses: KL dyn {c.w_dyn:g} / rep {c.w_rep:g} · standard reward "
              f"{c.w_standard_reward:g} · event gate {c.w_event_gate:g} · sparse reward {c.w_sparse_reward:g} · grid "
              f"{c.w_grid:g} · curiosity {c.w_curiosity:g} · memory {c.w_memory:g} / prior {c.w_memory_prior:g} · "
              f"{rollout} · {c.value_target} targets · {adv} · dream events: sparse head x P(event gate), one per "
              f"dream · {c.dreams_per_sequence} dreams per sequence x {c.dream_horizon} steps · {audit}\033[0m")
        on = [k for k in ("compile", "act_cuda_graph", "dec_sac", "dec_targets_once", "enc_fast_attn",
                          "enc_last_queries", "skip_pad_frames", "sync_free", "h2d_stream", "async_save",
                          "async_dream_pdf")
              if getattr(c, k) and (k != "compile" or self.compiled)]
        print(f"\033[90m    [setup] speed: {', '.join(on) or 'none'}"
              f"{' (the first updates compile: minutes, cached on disk afterwards)' if self.compiled else ''}\033[0m")

    @property
    def config(self):
        """Config as a plain dict (stored in checkpoints)."""
        return {**config_to_dict(self.cfg), "arch_version": self.ARCH_VERSION}

    # --- Batches ----------------------------------------------------------------------------------------------------
    def start_prefetch(self, batch_size=None, seq_len=None):
        """Samples batches on a background thread while the GPU trains (the buffer must not change meanwhile)."""
        self.stop_prefetch()
        cuda = self.device.type == "cuda"
        self._prefetcher = BatchPrefetcher(self.buffer, batch_size or self.cfg.batch_size,
                                           seq_len or self.cfg.seq_len, pin=cuda,
                                           device=self.device if cuda and self.cfg.h2d_stream else None)

    def stop_prefetch(self):
        """Stops the background sampler."""
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None

    def sample_batch(self, batch_size=None, seq_len=None):
        """Replay batch on device, decoded into model inputs and targets; None if the buffer is empty."""
        batch_size, seq_len = batch_size or self.cfg.batch_size, seq_len or self.cfg.seq_len
        pf = self._prefetcher
        use_pf = pf is not None and (pf.batch_size, pf.seq_len) == (batch_size, seq_len)
        item = pf.get() if use_pf else self.buffer.sample(batch_size, seq_len)
        if item is None:
            return None
        if "__gpu__" in item:                                              # copied on the prefetcher's stream
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(item["__event__"])
            b, extra = item["__gpu__"], item["__cpu__"]
            for v in b.values():
                v.record_stream(stream)
        else:
            if "__cpu__" in item:
                extra, cpu = item["__cpu__"], item["__batch__"]
            else:                                                          # sampled here
                cpu, extra = with_batch_extras(item)
            b = {k: v.to(self.device, non_blocking=True) for k, v in cpu.items()}
        return {
            "frame": b["frames"],
            "palette": b["palettes"],
            "blocks": b["blocks"].long(),
            "grid": stored_grid(b["grids"]),                               # explored grid
            "map_id": b["map_ids"].long(),
            "ram": b["rams"],
            "memory": unpack_bits(b["memories"], MEMORY_DIM),
            "actions": F.one_hot(b["actions"].long(), self.action_dim).float(),
            "sparse_rewards": b["sparse_rewards"],
            "standard_rewards": b["standard_rewards"],
            "curiosities": b["curiosities"],
            "continues": b["continues"][..., 0],
            "blackouts": b["blackouts"].float(),
            "valid": b["valid"],
            "index": b["index"],                                           # buffer rows (surprise priorities)
            "real_idx": b["real_idx"], "real_gather": b["real_gather"],    # real frames / position -> real frame
            **extra,                                                       # CPU: n_valid1, index_cpu, valid_cpu
        }

    @staticmethod
    def _flat(batch, keys, idx=None):
        """Fields `keys` of a batch, flattened over [B, T] (optionally indexed)."""
        out = {}
        for k in keys:
            v = batch[k].flatten(0, 1)
            out[k] = v if idx is None else v[idx]
        return out

    @classmethod
    def _flat_obs(cls, batch, idx=None):
        """Encoder inputs + sprite overlay of a batch, flattened over [B, T] (optionally indexed)."""
        return cls._flat(batch, cls.OBS_KEYS + ("blocks",), idx)

    # --- Small helpers ----------------------------------------------------------------------------------------------
    def _amp(self):
        """Autocast context (no-op when mixed precision is off)."""
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.bfloat16,
                              enabled=self.amp_dtype is not None)

    def _optimize(self, loss, name, params, max_norm=None):
        """Backward + optional clipping + step of optimizer `name` through its GradScaler -> pre-clip norm."""
        optimizer, scaler = getattr(self, name), self.scalers[name]
        if self.cfg.sync_free and not scaler.is_enabled() and optimizer.defaults.get("fused"):
            # no host sync: the threshold may be a GPU tensor, and the fused optimizer skips a non-finite step on the
            # GPU (found_inf, as GradScaler drives fused optimizers) -> pre-clip norm as a GPU tensor
            loss.backward()
            norm, optimizer.found_inf = None, None
            if max_norm is not None:
                grads = [p.grad for p in params if p.grad is not None]
                norm = (torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))) if grads
                        else torch.zeros((), device=self.device))
                torch._foreach_mul_(grads, (max_norm / (norm + 1e-6)).clamp(max=1.0))
                optimizer.found_inf = (~torch.isfinite(norm)).float()
            optimizer.step()
            return norm
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(params, max_norm) if max_norm is not None else None
        if scaler.is_enabled():
            scaler.step(optimizer)                                         # skipped on fp16 overflow
            scaler.update()
        elif norm is None or torch.isfinite(norm):
            optimizer.step()
        return norm

    @torch.no_grad()
    def _critic_ema_update(self):
        """Moves the target critics towards the critics."""
        for tgt, src in ((self.critic_target, self.critic), (self.curiosity_critic_target, self.curiosity_critic)):
            for pt, po in zip(tgt.parameters(), src.parameters()):
                pt.lerp_(po, 1.0 - self.cfg.critic_ema_tau)

    def _defer_priorities(self, batch, dyn):
        """Surprise priorities without a host sync: this update's KL is copied to pinned memory asynchronously and
        written into the buffer at the next update (or flush_priorities), when the copy has long finished."""
        self.flush_priorities()
        val = torch.empty(dyn.shape, dtype=torch.float32, pin_memory=True)
        val.copy_(dyn.detach().float(), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._pending_prio = (ev, batch["index_cpu"][:, 1:], batch["valid_cpu"][:, 1:] > 0, val)

    def flush_priorities(self):
        """Writes the pending surprise priorities into the buffer (before it changes: end of every training round)."""
        if self._pending_prio is not None:
            ev, idx, sel, val = self._pending_prio
            ev.synchronize()
            self.buffer.update_priorities(idx[sel], val[sel])
            self._pending_prio = None

    @staticmethod
    def full_state(h, zf, mem):
        """Agent state s = [h, z, m]."""
        return torch.cat([h.float(), zf.float(), mem.float()], dim=-1)

    def split_state(self, s):
        """s [..., S] -> (h, z, m)."""
        d, L = self.tssm_dim, self.latent_dim
        return s[..., :d], s[..., d:d + L], s[..., d + L:]

    def event_state(self, h, zf, mem_prev):
        """Input of the event gate and sparse reward heads: [h_{t+1}, z_{t+1}, m_t], the transition's state with the
        memory BEFORE it (in a dream m_{t+1} depends on the gate's decision)."""
        return self.full_state(h, zf, mem_prev)

    def event_step(self, h, zf, mem, alive, mem_logits=None, generator=None):
        """One imagined transition's event -> (expected event reward [N], gate probability [N],
        m_{t+1} [N, M], alive after the step [N], new bits [N, M]).

        The heads read [h_{t+1}, z_{t+1}, m_t]: reward = sparse reward head x P(gate) x alive, the expected value of
        the transition's event (alive = 1 while the dream has had no event, so a dream pays at most one). The gate's
        Bernoulli sample decides whether m changes; then the bits of M(z_{t+1}) above memory_threshold that are not
        in m_t turn on (none above it: M's most likely unset bit). Set bits are never unset and never pay again (the
        gate head reads m_t, and on real data an event never repeats a set bit). generator: the audit's own one."""
        s_ev = self.event_state(h, zf, mem)
        p = torch.sigmoid(self.event_gate_head(s_ev).float())
        size = self.two_hot.decode(self.sparse_reward_head(s_ev)).squeeze(-1).float().clamp(min=0.0)
        reward = size * p * alive
        u = torch.rand(p.shape, device=p.device, generator=generator)
        fire = (u < p).float() * alive
        if mem_logits is None:
            mem_logits = self.tokenizer.memory_head(zf)
        unset = (mem < 0.5).float()
        pm = torch.sigmoid(mem_logits.float()) * unset
        above = (pm > self.cfg.memory_threshold).float()
        top = F.one_hot(pm.argmax(-1), pm.shape[-1]).float() * unset
        bits = torch.where(above.sum(-1, keepdim=True) > 0, above, top)
        new = bits * fire[:, None]
        return reward, p, torch.maximum(mem, new), alive * (1.0 - fire), new

    def dream_rewards(self, nxt, event):
        """Per-stream rewards of new dream states nxt [..., S] as imagine computes them -> (event rewards (given:
        event_step, sparse head x P(gate)), standard reward (head), curiosity): the predicted novelty (>= 0) gated by
        "square not visited yet" (grid head)."""
        standard = self.two_hot.decode(self.standard_reward_head(nxt)).squeeze(-1)
        cur = self.two_hot.decode(self.curiosity_head(nxt)).squeeze(-1)
        visit_gate = 1.0 - torch.sigmoid(self.grid_head(nxt).float()[..., GRID_CENTER_INDEX])   # visited: 0
        cur = cur.clamp(min=0.0) * visit_gate
        return event, standard, cur

    # --- Encoder ----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_frames(self, obs, chunk=1024):
        """Observations [..., ...] -> (flat one-hot samples [..., D], posterior log-probs [..., L, K]), no gradient."""
        lead = obs["frame"].shape[:-2]
        flat = {k: obs[k].reshape(-1, *obs[k].shape[len(lead):]) for k in self.OBS_KEYS}
        n = flat["frame"].shape[0]
        zs, lps = [], []
        with self._amp():
            for i in range(0, n, chunk):
                z, lp = self.tokenizer.encode({k: v[i:i + chunk] for k, v in flat.items()})
                zs.append(z.float())
                lps.append(lp.float())
        return (torch.cat(zs).view(*lead, self.latent_dim),
                torch.cat(lps).view(*lead, self.n_latents, self.classes))

    @torch.no_grad()
    def _update_drift(self):
        """Every drift_interval updates: share of categoricals whose most likely class changed on fixed probe frames."""
        c = self.cfg
        if c.drift_interval <= 0 or self.total_num_updates % c.drift_interval:
            return
        if self._probe is None:
            n = min(c.drift_frames, len(self.buffer))
            if n < 2:
                return
            p = {k: v.to(self.device) for k, v in self.buffer.sample_frames(n).items()}
            self._probe = {"frame": p["frames"], "palette": p["palettes"],
                           "grid": stored_grid(p["grids"]),
                           "map_id": p["map_ids"].long(), "ram": p["rams"],
                           "memory": unpack_bits(p["memories"], MEMORY_DIM)}
        mode = self.encode_frames(self._probe)[1].argmax(-1)
        if self._probe_ref is not None:
            self.last_drift = (mode != self._probe_ref).float().mean().item()
        self._probe_ref = mode

    # --- World model: losses ----------------------------------------------------------------------------------------
    def _side_drop(self, n, dev):
        """Token-dropout mask [n, N_SIDE_TOKENS] of the augmented re-encodings (cfg.drop_<group>)."""
        p = torch.tensor([getattr(self.cfg, f"drop_{g}") for g in SIDE_GROUPS], device=dev)
        return torch.rand(n, len(SIDE_GROUPS), device=dev) < p

    def _grounding(self, zf, o):
        """Grounding losses (map ... menu) and memory loss (1) of flat latents zf against observations o."""
        c = self.cfg
        g, acc = self.tokenizer.heads.losses(zf, o["map_id"], o["ram"])
        mem_logits = self.tokenizer.memory_head(zf)
        g["memory"] = memory_bce(mem_logits, o["memory"], c.memory_pos_weight).mean()
        return g, acc, mem_logits

    def _recon_branch(self, batch, z, dream):
        """Pixel decoder, grounding heads and memory head (1), all on z only.

        Grounding + memory: every valid frame. Decoder: recon_frames real frames (t >= 1; 0 = every valid frame).
        Augmented part: mask_frames of them re-encoded with MAE masking + token dropout (the memory token is never
        dropped), weighted mask_weight. Dream-like states only give accuracies (no loss)."""
        c, dev = self.cfg, self.device
        T = z.shape[1]
        zero = torch.zeros((), device=dev)
        zflat = z.flatten(0, 1)
        allsel = batch["real_idx"] if "real_idx" in batch else batch["valid"].flatten().nonzero().squeeze(1)
        o_all = self._flat(batch, ("map_id", "ram", "memory"), allsel)
        g, acc, mem_logits = self._grounding(zflat[allsel], o_all)
        mem_acc = memory_metrics(mem_logits, o_all["memory"], torch.ones(allsel.shape[0], device=dev),
                                 c.memory_threshold)
        with torch.no_grad():
            # exact match on frames whose memory has a set bit: an M that predicts "nothing set" scores 0 there
            # (on frames with an empty memory it would score 1 without having learned anything)
            has_set = (o_all["memory"].sum(-1) > 0).float()
            pred = (torch.sigmoid(mem_logits.float()) > c.memory_threshold).float()
            n_set = has_set.sum()
            exact_set = ((pred == o_all["memory"]).all(-1).float() * has_set).sum() / n_set.clamp(min=1.0)
            mem_acc["exact_set"] = exact_set

        valid1 = batch["valid"][:, 1:].flatten()                           # frames t >= 1 (as the heads)
        n_valid = batch["n_valid1"] if "n_valid1" in batch else int(valid1.sum().item())   # CPU count: no sync
        if c.recon_frames <= 0 or c.recon_frames >= n_valid:
            sel1 = valid1.nonzero().squeeze(1)                             # index into [B, T - 1]
        else:
            sel1 = torch.multinomial(valid1, c.recon_frames, replacement=False)
        sel = (sel1 // (T - 1)) * T + sel1 % (T - 1) + 1                   # the same frames, index into [B, T]
        o = self._flat_obs(batch, sel)
        tgt_pix = self.tokenizer.recon_targets(o, c.sprite_weight) if c.dec_targets_once else None   # once
        pix_loss, stats = self.tokenizer.recon_loss(zflat[sel], o, c.sprite_weight, targets=tgt_pix)   # chunked

        n_aug = min(c.mask_frames, sel.shape[0])
        hide = None
        map_notoken = zero
        if n_aug > 0:                                                      # augmented re-encoding (MAE + dropout)
            pick = torch.randperm(sel.shape[0], device=dev)[:n_aug]
            oa = {k: v[pick] for k, v in o.items()}
            hide = sample_hide(oa["blocks"], c.mask_max, c.enc_patch_squares)
            drop = self._side_drop(n_aug, dev)
            za = sample_onehot(self.tokenizer.posterior(oa, hide, drop)).flatten(1)
            pix_a, _ = self.tokenizer.recon_loss(za, oa, c.sprite_weight, targets=None if tgt_pix is None
                                                 else self.tokenizer.subset_targets(tgt_pix, pick))
            g_a, _, _ = self._grounding(za, oa)
            a = c.mask_weight
            pix_loss = (1.0 - a) * pix_loss + a * pix_a
            g = {k: (1.0 - a) * g[k] + a * g_a[k] for k in g}
            with torch.no_grad():
                no_map = drop[:, SIDE_GROUPS.index("map")].float()
                hit = (self.tokenizer.heads.predict(za)[0] == oa["map_id"].long()).float()
                map_notoken = (hit * no_map).sum() / no_map.sum().clamp_min(1.0)
        weights = {"map": c.w_map, "status": c.w_status, "party": c.w_party, "enemy": c.w_enemy, "items": c.w_items,
                   "menu": c.w_menu, "memory": c.w_memory}
        parts = {"pixel": c.w_recon * pix_loss, **{k: weights[k] * g[k] for k in g}}
        with torch.no_grad():
            tgt = dream["tgt"]                                             # dream-like states: accuracy only
            zd = self.split_state(dream["states"])[1]
            _, acc_d = self.tokenizer.heads.losses(zd, tgt["map_id"], tgt["ram"], mask=tgt["mask"])
            mem_d = memory_metrics(self.tokenizer.memory_head(zd), tgt["memory"], tgt["mask"], c.memory_threshold)
            stats.update({"pix_loss": pix_loss.detach(), **{f"{k}_loss": v.detach() for k, v in g.items()},
                          **{k: v for k, v in acc.items()}, "map_acc_notoken": map_notoken,
                          "memory_exact": mem_acc["exact"], "memory_recall": mem_acc["recall"],
                          "memory_false_pos": mem_acc["false_pos"], "memory_exact_set": mem_acc["exact_set"],
                          "dream_map_acc": acc_d["map_acc"], "dream_memory_exact": mem_d["exact"],
                          "dream_memory_recall": mem_d["recall"],
                          "mask_ratio": hide.float().mean() if hide is not None else zero,
                          "grounding_loss": sum(parts[k] for k in GROUNDING_GROUPS).detach(),
                          "recon_loss": sum(parts.values()).detach()})
        return stats, parts

    @staticmethod
    def _targets(batch, rows, t):
        """Targets of the state at observation t: standard reward / sparse (memory-event) reward / curiosity of
        t-1 -> t, grid, map, RAM and memory of t."""
        return {"standard": batch["standard_rewards"][rows, t - 1, 0],
                "sparse": batch["sparse_rewards"][rows, t - 1, 0],
                "curiosity": batch["curiosities"][rows, t - 1, 0], "grid": batch["grid"][rows, t],
                "map_id": batch["map_id"][rows, t], "ram": batch["ram"][rows, t], "memory": batch["memory"][rows, t],
                "mask": batch["valid"][rows, t]}

    def _head_losses(self, s, tgt):
        """Standard reward, curiosity and explored-grid losses on states s [..., S]; pads are ignored."""
        c, m = self.cfg, tgt["mask"]
        standard_loss = self.two_hot.loss(self.standard_reward_head(s), tgt["standard"], m)
        cur = tgt["curiosity"]                                             # novelty of a new square (> 0) or 0
        cur_w = torch.where(cur > 0, torch.full_like(cur, self.CURIOSITY_POS_WEIGHT), torch.ones_like(cur)) * m
        curiosity_loss = self.two_hot.loss(self.curiosity_head(s if c.curiosity_grad else s.detach()), cur, cur_w)
        # 5x5 explored grid around the player (it needs the episode's history, so the head reads s): unexplored cells
        # up-weighted; its centre ("the player's square was already visited") gates the curiosity in dreams.
        explored = tgt["grid"].float()
        e_logit = self.grid_head(s).float()
        gw = (1.0 + (c.grid_zero_weight - 1.0) * (1.0 - explored)) * m[..., None]
        grid_loss = (F.binary_cross_entropy_with_logits(e_logit, explored, reduction="none") * gw).sum() \
            / gw.sum().clamp(min=1.0)
        with torch.no_grad():
            hit = ((e_logit > 0).float() == explored).float()
            denom = m.sum().clamp(min=1.0)
            stats = {"grid_acc": (hit.mean(-1) * m).sum() / denom,
                     "grid_center_acc": (hit[..., GRID_CENTER_INDEX] * m).sum() / denom}
        return {"standard": standard_loss, "curiosity": curiosity_loss, "grid": grid_loss}, stats

    def _event_losses(self, s_ev, tgt):
        """Event gate (BCE: does a memory bit turn on in this transition?) and sparse reward head (the transition's
        sparse reward, on event transitions only) on event states [h_{t+1}, z_{t+1}, m_t]; pads are
        ignored -> (losses, stats: mean P(gate) and the real event rate, recall / false events at P > 0.5, P(gate) on
        event transitions, the sparse head's prediction there, events in the batch)."""
        c, m = self.cfg, tgt["mask"]
        sparse = tgt["sparse"].float()
        ev = (sparse > 0).float()
        logit = self.event_gate_head(s_ev).float()
        w = (1.0 + (c.event_gate_pos_weight - 1.0) * ev) * m
        gate_loss = (F.binary_cross_entropy_with_logits(logit, ev, reduction="none") * w).sum() / w.sum().clamp(min=1.0)
        sp_logits = self.sparse_reward_head(s_ev)
        sparse_loss = self.two_hot.loss(sp_logits, sparse, ev * m)       # 0 while the batch holds no event
        with torch.no_grad():
            p = torch.sigmoid(logit)
            hit = (p > 0.5).float()
            evm, no_ev = ev * m, (1.0 - ev) * m
            n_ev = evm.sum()
            stats = {"event_gate_pred": (p * m).sum() / m.sum().clamp(min=1.0),
                     "event_gate_real": n_ev / m.sum().clamp(min=1.0),
                     "event_gate_recall": (hit * evm).sum() / n_ev.clamp(min=1.0),
                     "event_gate_false": (hit * no_ev).sum() / no_ev.sum().clamp(min=1.0),
                     "event_gate_pos_p": (p * evm).sum() / n_ev.clamp(min=1.0),
                     "sparse_reward_pred": (self.two_hot.decode(sp_logits.detach()).squeeze(-1) * evm).sum()
                     / n_ev.clamp(min=1.0),
                     "events_in_batch": n_ev}
        return {"event_gate": gate_loss, "sparse_reward": sparse_loss}, stats

    def _rollout(self, kv, z, post, batch):
        """Open-loop rollout from a random split point, fed its own prior samples (and the real memory): KL to real
        posteriors + states."""
        c, dev = self.cfg, self.device
        B, T = z.shape[:2]
        K = c.rollout_steps
        mem = batch["memory"]
        rows = torch.arange(B, device=dev).repeat_interleave(c.rollout_splits)
        R = rows.shape[0]
        n_real = batch["valid"].sum(1).long()[rows]
        p_max = torch.clamp(n_real - 1 - K, min=1, max=T - 1 - K)
        p = 1 + torch.minimum((torch.rand(R, device=dev) * p_max).long(), p_max - 1)
        state = self.tssm.open_loop_state(kv, rows, p)                     # real prefix K/V, with gradient
        zin = z[rows, p].detach()
        kls, states, match = [], [], []
        for j in range(K):
            t = p + j
            h = self.tssm.step_append(zin, batch["actions"][rows, t], state, mem[rows, t]).float()
            raw = self.prior(h)
            target = post[rows, t + 1].detach()
            kls.append(kl_categorical(target, unimix_logprobs(raw, self.prior.unimix)))
            with torch.no_grad():
                zs = sample_onehot(unimix_logprobs(raw, c.dream_unimix), straight_through=False)
                match.append((zs.argmax(-1) == target.argmax(-1)).float().mean(-1))
                zin = zs.flatten(1)
            states.append(self.full_state(h, zin, mem[rows, t + 1]))
        t_idx = p[:, None] + 1 + torch.arange(K, device=dev)[None]          # frame predicted at step j
        M = batch["valid"][rows[:, None], t_idx]
        w = c.rollout_decay ** torch.arange(K, device=dev, dtype=torch.float32)
        kl = torch.stack(kls, dim=1)
        loss = (kl.clamp(min=c.free_nats) * M * w).sum() / (M * w).sum().clamp(min=1e-6)
        return {"loss": loss, "states": torch.stack(states, dim=1), "rows": rows, "t": t_idx, "mask": M,
                "match": torch.stack(match, dim=1), "kl": kl.detach()}

    def _memory_prior(self, prior, mem, m1):
        """Memory loss (2): M(prior sample of z_{t+1}) vs m_{t+1}, M frozen, so only the prior and the TSSM learn
        (straight-through sample: hard one-hot forward, gradient into the prior's logits)."""
        c = self.cfg
        with frozen(self.tokenizer.memory_head):
            zp = sample_onehot(prior).flatten(-2)
            logits = self.tokenizer.memory_head(zp)
        m_prev, m_next = mem[:, :-1], mem[:, 1:]
        turned_on = ((m_next > 0.5) & (m_prev < 0.5)).float()
        bit_w = 1.0 + (c.memory_event_weight - 1.0) * turned_on
        loss = masked_mean(memory_bce(logits, m_next, 1.0, bit_w), m1)
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > c.memory_threshold).float()
            ev = turned_on * m1[..., None]
            unset = (m_prev < 0.5).float()
            no_event = (turned_on.sum(-1) == 0).float() * m1
            false_event = ((pred * unset).sum(-1) > 0).float()
            stats = {"prior_event_recall": (pred * ev).sum() / ev.sum().clamp(min=1.0),
                     "prior_events": ev.sum(),
                     "prior_false_event": (false_event * no_event).sum() / no_event.sum().clamp(min=1.0),
                     "prior_memory_exact": masked_mean((pred == m_next).all(-1).float(), m1)}
        return loss, stats

    # --- World model: update ----------------------------------------------------------------------------------------
    def train_world_model(self, batch, compute_metrics=True):
        """One end-to-end world-model update -> (states [B, T-1, S], kv context, start mask, metrics, start info)."""
        c = self.cfg
        B, T = batch["frame"].shape[:2]
        valid = batch["valid"]
        m1 = valid[:, 1:]
        mem = batch["memory"]
        tm = self.timer
        tm.start(compute_metrics)
        zero = torch.zeros((), device=self.device)
        self.wm_optimizer.zero_grad(set_to_none=True)
        measure = (c.grad_share_interval > 0 and self._grad_share_ok
                   and self.total_num_updates % c.grad_share_interval == 0)    # gradient-share log this update
        eager = eager_modules() if measure else contextlib.nullcontext()   # a graph that can be backpropagated per term
        with eager, self._amp():
            # 1. clean posteriors of every frame, straight-through one-hot samples
            flat = self._flat_obs(batch)
            if c.skip_pad_frames and "real_idx" in batch:                 # pads repeat their row's last real frame:
                ri = batch["real_idx"]                                     # its posterior is copied, not recomputed
                post = self.tokenizer.posterior({k: flat[k][ri] for k in self.OBS_KEYS})[batch["real_gather"]]
            else:
                post = self.tokenizer.posterior(flat)
            post = post.view(B, T, self.n_latents, self.classes)
            z = sample_onehot(post).flatten(-2)                            # straight-through: gradient reaches the
            tm.mark("encode")                                              # encoder from every branch that reads z

            # 2. dynamics: TSSM (z_t, a_t, m_t) -> h_{t+1} -> prior of z_{t+1}; balanced KL with free nats; memory (2)
            h, kv = self.tssm.forward_sequence(z[:, :-1], batch["actions"][:, :-1], mem[:, :-1], return_kv=True,
                                               detach_kv=False)
            h = h.float()
            prior_raw = self.prior(h)
            prior = unimix_logprobs(prior_raw, self.prior.unimix)
            post1 = post[:, 1:]
            dyn = kl_categorical(post1.detach(), prior)
            rep = kl_categorical(post1, prior.detach())
            kl_dyn_term = c.w_dyn * masked_mean(dyn.clamp(min=c.free_nats), m1)    # trains prior + TSSM (+ encoder via z)
            kl_rep_term = c.w_rep * masked_mean(rep.clamp(min=c.free_nats), m1)    # trains the encoder
            kl_loss = kl_dyn_term + kl_rep_term
            mem_prior_loss, mem_prior_stats = self._memory_prior(prior, mem, m1)
            tm.mark("TSSM + prior fwd")

            # 3. heads on real states [h_{t+1}, z_{t+1}, m_{t+1}] and on dream-like states (open-loop rollout or
            #    one-step prior samples)
            s_real = self.full_state(h, z[:, 1:], mem[:, 1:])
            rows = torch.arange(B, device=self.device)[:, None]
            t_real = torch.arange(1, T, device=self.device)[None]
            tgt_real = self._targets(batch, rows, t_real)
            head_r, head_stats = self._head_losses(s_real, tgt_real)
            with torch.no_grad():                                          # one-step prior samples, as dreams draw them
                z1 = sample_onehot(unimix_logprobs(prior_raw, c.dream_unimix), straight_through=False).flatten(-2)
            ro = None
            if c.rollout_steps > 0:
                ro = self._rollout(kv, z, post, batch)
                dream = {"states": ro["states"], "tgt": self._targets(batch, ro["rows"][:, None], ro["t"])}
                rollout_loss = ro["loss"]
            else:                                                          # one-step prior samples instead
                dream = {"states": self.full_state(h, z1, mem[:, 1:]), "tgt": tgt_real}
                rollout_loss = zero
            head_o, _ = self._head_losses(dream["states"], dream["tgt"])
            head = {k: (1.0 - c.w_dreamlike) * head_r[k] + c.w_dreamlike * head_o[k] for k in head_r}
            # dream events: gate + sparse reward heads on [h_{t+1}, z_{t+1}, m_t], trained on the prior's
            # one-step samples (what a dream sees: P(gate) learns how often a sample only looks like an event) and, with
            # event_dreamlike < 1, partly on real latents; the real-latent statistics are logged either way
            ev_real_in = self.event_state(h, z[:, 1:], mem[:, :-1])
            if c.event_dreamlike < 1.0:
                ev_r, ev_stats_r = self._event_losses(ev_real_in, tgt_real)
            else:
                with torch.no_grad():
                    ev_r, ev_stats_r = self._event_losses(ev_real_in.detach(), tgt_real)
            ev_o, ev_stats = self._event_losses(self.event_state(h, z1, mem[:, :-1]), tgt_real)
            a_ev = c.event_dreamlike
            head.update({k: (1.0 - a_ev) * ev_r[k] + a_ev * ev_o[k] for k in ev_r})
            tm.mark("rollout + heads fwd")

            # 4. reconstruction (decoder reads z) + grounding and memory (1) (heads read z)
            rec_stats, rec_parts = self._recon_branch(batch, z, dream)
            tm.mark("decode")

        # every world-model loss term times its fixed weight: their sum is the world-model loss
        terms = {**rec_parts, "memory_prior": c.w_memory_prior * mem_prior_loss, "kl_dyn": kl_dyn_term,
                 "kl_rep": kl_rep_term, "rollout": c.w_rollout * rollout_loss,
                 "standard_reward": c.w_standard_reward * head["standard"],
                 "event_gate": c.w_event_gate * head["event_gate"],
                 "sparse_reward": c.w_sparse_reward * head["sparse_reward"], "grid": c.w_grid * head["grid"],
                 "curiosity": c.w_curiosity * head["curiosity"]}
        world_model_loss = sum(terms.values())
        self._accumulate_wm_losses(terms)
        if measure:
            with eager_modules():
                self._measure_grad_shares(terms)                           # diagnostic: one backward per term
        # loss groups of the (optional) harmony weights; the curiosity head (it trains only itself) keeps its weight
        groups = {"pixel": terms["pixel"], "grounding": sum(terms[k] for k in GROUNDING_GROUPS),
                  "memory": terms["memory"] + terms["memory_prior"], "kl": terms["kl_dyn"] + terms["kl_rep"],
                  "rollout": terms["rollout"],
                  "reward": terms["standard_reward"] + terms["event_gate"] + terms["sparse_reward"],
                  "grid": terms["grid"]}
        fixed = terms["curiosity"]
        hw = None
        if c.harmony:
            hw = self.harmony.weights()
            wm_opt_loss = sum(hw[k] * v for k, v in groups.items()) + fixed
        else:
            wm_opt_loss = world_model_loss
        clip = self.wm_clipper.threshold()
        grad_norm = self._optimize(wm_opt_loss, "wm_optimizer", self.wm_params, clip)
        if not c.sync_free:
            grad_norm = float(grad_norm)
        self.wm_clipper.record(grad_norm, clip)
        if c.harmony:
            self.harmony_optimizer.zero_grad(set_to_none=True)
            self.harmony.scale_loss({k: v.detach() for k, v in groups.items()}).backward()
            self.harmony_optimizer.step()
        tm.mark("WM backward + step")
        self._update_drift()

        start_mask = m1.detach()
        with torch.no_grad():                                              # dream-start pools
            ram1 = batch["ram"][:, 1:].long()
            battle_menu = ((ram1[..., REC_BATTLE] > 0) & (ram1[..., REC_UI] > 0)).float() * m1
            start_info = {"surprise": dyn.detach().float() * m1, "focus": battle_menu}
            if c.loss_sample_prob > 0 and "index" in batch:                # masked on the CPU: fixed-size GPU tensors
                if c.sync_free and "index_cpu" in batch and self.device.type == "cuda":
                    self._defer_priorities(batch, dyn)
                else:
                    sel = (m1 > 0).cpu()
                    self.buffer.update_priorities(batch["index"][:, 1:].cpu()[sel], dyn.detach().float().cpu()[sel])

        metrics = {}
        if compute_metrics:
            with torch.no_grad():
                metrics = {"world_model_loss_last": world_model_loss.item(), "kl_dyn": masked_mean(dyn, m1).item(),
                           "kl_loss": kl_loss.item(), "rollout_loss": rollout_loss.item(),
                           "memory_prior_loss": mem_prior_loss.item(),
                           "wm_grad_norm": float(grad_norm), "padded_fraction": 1.0 - valid.mean().item(),
                           "encoder_drift": self.last_drift, "battle_menu_states": float(battle_menu.sum())}
                metrics.update({k: float(v) for k, v in rec_stats.items()})
                metrics.update({k: float(v) for k, v in mem_prior_stats.items()})
                metrics.update({f"{k}_loss": v.item() for k, v in head.items()})
                metrics["reward_loss"] = metrics["standard_loss"]
                metrics.update({k: v.item() for k, v in head_stats.items()})
                metrics.update({k: float(v) for k, v in ev_stats.items()})                     # prior samples
                metrics.update({f"{k}_real": float(ev_stats_r[k]) for k in ("event_gate_pred", "event_gate_recall",
                                                                            "event_gate_false", "event_gate_pos_p")})
                mode_post = post.argmax(-1)
                metrics["post_entropy"] = masked_mean(entropy_categorical(post), valid).item()
                metrics["prior_entropy"] = masked_mean(entropy_categorical(prior), m1).item()
                metrics["latent_usage"] = float(class_usage(z.detach(), self.n_latents, self.classes, valid))
                metrics["pred_match"] = masked_mean((prior.argmax(-1) == mode_post[:, 1:]).float().mean(-1), m1).item()
                metrics["copy_match"] = masked_mean((mode_post[:, :-1] == mode_post[:, 1:]).float().mean(-1), m1).item()
                if ro is not None:
                    k = ro["match"].shape[1] - 1
                    metrics["rollout_match_k"] = masked_mean(ro["match"][:, k], ro["mask"][:, k]).item()
                    metrics["rollout_kl_k"] = masked_mean(ro["kl"][:, k], ro["mask"][:, k]).item()
                clip_threshold, clip_fraction = self.wm_clipper.stats()
                metrics.update({"wm_clip_threshold": clip_threshold, "wm_clip_fraction": clip_fraction})
                metrics.update({f"group_{k}": float(v.detach()) for k, v in groups.items()})
                if hw is not None:
                    metrics.update({f"harmony_{k}": float(v) for k, v in hw.items()})
                    metrics["harmonized_loss"] = float(wm_opt_loss.detach())
                metrics.update(self._wm_loss_means())                     # world_model_loss + wm_* (round means)
                metrics.update(self.last_grad_shares)                     # newest gradient-share measurement
            tm.mark("metrics")
        kv_context = [(k.detach(), v.detach()) for k, v in kv]
        return s_real.detach(), kv_context, start_mask, metrics, start_info

    def _measure_grad_shares(self, terms):
        """Diagnostic (no training effect): the gradient norm of every weighted loss term, computed separately
        (torch.autograd.grad, the graph is kept) on the TSSM parameters and on the encoder parameters, and each term's
        share of the sum of the norms of its group -> self.last_grad_shares ("gnorm_<group>_<term>",
        "gshare_<group>_<term>"). A failure disables the measurement with one message; training goes on."""
        groups = {"tssm": [p for p in self.tssm.parameters() if p.requires_grad],
                  "encoder": [p for p in self.tokenizer.encoder.parameters() if p.requires_grad]}
        params = groups["tssm"] + groups["encoder"]
        n_t = len(groups["tssm"])
        zero = torch.zeros((), device=self.device)
        try:
            norms = {}
            for name in self.WM_LOSS_TERMS:
                loss = terms[name]
                if not (torch.is_tensor(loss) and loss.requires_grad):
                    norms[name] = (zero, zero)
                    continue
                grads = torch.autograd.grad(loss.float(), params, retain_graph=True, allow_unused=True)

                def norm(gs):
                    sq = [g.float().square().sum() for g in gs if g is not None]
                    return torch.stack(sq).sum().sqrt() if sq else zero
                norms[name] = (norm(grads[:n_t]), norm(grads[n_t:]))
                del grads
            vals = torch.stack([torch.stack(v) for v in norms.values()]).tolist()        # one host sync
        except Exception as e:                                             # never let a diagnostic stop training
            print(f"[!] gradient-share log disabled ({type(e).__name__}: {e})")
            self._grad_share_ok = False
            return
        out = {"grad_share_update": float(self.total_num_updates)}
        for gi, group in enumerate(self.GRAD_SHARE_GROUPS):
            total = sum(v[gi] for v in vals)
            for name, v in zip(norms, vals):
                out[f"gnorm_{group}_{name}"] = v[gi]
                out[f"gshare_{group}_{name}"] = v[gi] / total if total > 0 else 0.0
        self.last_grad_shares = out

    @torch.no_grad()
    def _accumulate_wm_losses(self, terms):
        """Adds the weighted loss terms of one update to the round sums (on the device: no host sync)."""
        vals = torch.stack([terms[k].detach().float() for k in self.WM_LOSS_TERMS])
        if self._wm_loss_acc is None:
            self._wm_loss_acc = torch.zeros(len(self.WM_LOSS_TERMS) + 1, device=vals.device)
        self._wm_loss_acc[:-1] += vals
        self._wm_loss_acc[-1] += 1.0

    def _wm_loss_means(self):
        """Round averages of the weighted loss terms ("wm_<term>") and of their sum ("world_model_loss"); resets."""
        if self._wm_loss_acc is None or float(self._wm_loss_acc[-1]) == 0.0:
            return {}
        acc = self._wm_loss_acc.tolist()
        n = acc[-1]
        out = {f"wm_{k}": v / n for k, v in zip(self.WM_LOSS_TERMS, acc[:-1])}
        out["world_model_loss"] = sum(acc[:-1]) / n
        out["wm_updates"] = n
        self._wm_loss_acc.zero_()
        return out

    # --- Imagination ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def context_kv(self, zf, actions, mem):
        """Teacher-forced pass over real latents, actions and memory [B, T, ...] -> (h [B, T-1, d], kv)."""
        with self._amp():
            h, kv = self.tssm.forward_sequence(zf[:, :-1], actions[:, :-1], mem[:, :-1], return_kv=True)
        return h.float(), kv

    @torch.no_grad()
    def dream_step(self, z, action, cache, mem):
        """One eager imagined step -> (h, one-hot sample z, prior log-probs)."""
        h = self.tssm.forward_step(z, action, cache, mem).float()
        logp = unimix_logprobs(self.prior(h), self.cfg.dream_unimix)
        return h, sample_onehot(logp, straight_through=False).flatten(1), logp

    def _dream_graph_for(self, n):
        """CUDA graph of one imagined step for n dreams, or None to run eagerly (CPU, disabled or capture failed)."""
        if not self._use_dream_graph or self.device.type != "cuda":
            return None
        g = self._dream_graph
        if g is not None and g.n == n:
            return g
        self._dream_graph = None
        try:
            g = GraphedDreamStep(self.tssm, self.prior, n, self.cache_dtype, self.amp_dtype, self.device)
        except Exception as e:                                             # never let the graph stop training
            print(f"[!] dream CUDA graph disabled ({type(e).__name__}: {e}); dreams run eagerly.")
            self._use_dream_graph = False
            return None
        self._dream_graph = g
        return g

    def _dream_starts(self, start_mask, start_info, B, T, dev):
        """Start (row, t0) of every dream: uniform, surprising (dream_loss_frac), or first-action branch groups
        (started at battle-menu states when the batch has some, where the choice of action matters most)."""
        c, K, A = self.cfg, self.cfg.dreams_per_sequence, self.action_dim
        N = B * K
        rows = torch.arange(B, device=dev).repeat_interleave(K)
        elig = start_mask.float() if start_mask is not None else torch.ones(B, T, device=dev)
        e = elig[rows]
        counts = e.sum(1, keepdim=True)
        t0 = torch.multinomial(torch.where(counts > 0, e, torch.ones_like(e)), 1).squeeze(1)
        forced = torch.full((N,), -1, dtype=torch.long, device=dev)       # forced first action (-1 = policy)
        focus = torch.zeros(B, T, device=dev)
        G, n_norm = 0, N
        if start_info is not None:
            focus = start_info["focus"].float() * elig
            floor = 1e-6 * elig.flatten() + 1e-12                          # empty pool -> uniform over eligible
            G = min(int(c.branch_groups), (N // 2) // A)
            n_norm = N - G * A
            n_l = min(int(round(c.dream_loss_frac * n_norm)), n_norm)
            if n_l > 0:                                                    # the first n_l dreams: surprising states
                sur = start_info["surprise"].float().clamp_min(0.0) * elig
                idx = torch.multinomial(sur.flatten() + floor, n_l, replacement=True)
                rows[:n_l] = idx // T
                t0[:n_l] = idx % T
            if G > 0:                                                      # each group tries every first action
                idx = torch.multinomial(focus.flatten() + floor, G, replacement=False)
                rows[n_norm:] = (idx // T).repeat_interleave(A)
                t0[n_norm:] = (idx % T).repeat_interleave(A)
                forced[n_norm:] = torch.arange(A, device=dev).repeat(G)
        start_ok = (elig[rows, t0] > 0).float()
        return rows, t0, start_ok, forced, n_norm, G, focus

    @torch.no_grad()
    def _update_adv_range(self, ranges):
        """EMA (cfg.adv_norm_decay) of the two streams' return ranges [2]; the first update sets it (no host sync)."""
        if self.adv_range_init:
            d = self.cfg.adv_norm_decay
            self.adv_range.mul_(d).add_((1.0 - d) * ranges)
        else:
            self.adv_range.copy_(ranges)
            self.adv_range_init = True

    @torch.no_grad()
    def _combine_advantages(self, adv, cur_adv, lam, cur_lam, weights, update_scale=True):
        """Combined actor advantage of the extrinsic and curiosity streams [N, H] -> (combined, info).

        "range" (DreamerV3): each stream's advantage divided by its own return scale max(floor, S), S = EMA of the
        P95 - P5 range of its dream lambda-returns (floors adv_floor_ext / adv_floor_cur), not centred; then ext +
        curiosity_scale x cur. A stream whose returns spread less than its floor keeps its raw size (never scaled up),
        larger ones are brought to about unit size, so a +50 cannot swamp the entropy term and critic noise is not
        amplified. update_scale=False leaves the EMA untouched.
        "std": each stream centred on its own weighted mean and divided by its own weighted std (at least
        ADV_STD_MIN), then ext + curiosity_scale x cur.
        The sum is not normalised again. info: the two scales (range mode: the EMA ranges; std mode: the stds, both
        before the floor), the sum's std, and per stream its contribution's std, this batch's return range (P95 - P5)
        and its share of the sum's variance, cov(A_k, A) / var(A) (the shares add to 1; negative = the stream works
        against the total)."""
        c = self.cfg
        q = torch.tensor([0.05, 0.95], device=adv.device)
        r_ext = torch.quantile(lam.detach().flatten().float(), q)
        r_cur = torch.quantile(cur_lam.detach().flatten().float(), q)
        ranges = torch.stack([r_ext[1] - r_ext[0], r_cur[1] - r_cur[0]])
        if c.adv_norm == "range":
            if update_scale:
                self._update_adv_range(ranges)
            s_ext, s_cur = self.adv_range if self.adv_range_init else ranges
            parts = {"ext": adv.float() / s_ext.clamp(min=c.adv_floor_ext),
                     "cur": c.curiosity_scale * cur_adv.float() / s_cur.clamp(min=c.adv_floor_cur)}
        else:
            m_e, s_ext = weighted_moments(adv, weights)
            m_c, s_cur = weighted_moments(cur_adv, weights)
            parts = {"ext": (adv.float() - m_e) / s_ext.clamp(min=ADV_STD_MIN),
                     "cur": c.curiosity_scale * (cur_adv.float() - m_c) / s_cur.clamp(min=ADV_STD_MIN)}
        combined = parts["ext"] + parts["cur"]
        mean, std = weighted_moments(combined, weights)
        wsum = weights.sum().clamp(min=1e-8)
        info = {"scale_ext": s_ext, "scale_cur": s_cur, "combined_std": std}
        for name, a, rng in (("ext", parts["ext"], ranges[0]), ("cur", parts["cur"], ranges[1])):
            m, sigma = weighted_moments(a, weights)
            cov = (weights * (a - m) * (combined - mean)).sum() / wsum
            info[name] = {"adv_std": sigma, "return_range": rng,
                          "adv_share": cov / std.square().clamp(min=1e-12)}
        return combined, info

    def imagine(self, full_state, kv_context, start_mask=None, start_info=None, real=None, horizon=None,
                compute_metrics=True, pack_dreams=True):
        """Actor-critic update on dreams started from real states -> (metrics, best dream, random dream)."""
        c = self.cfg
        horizon = horizon or c.dream_horizon
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.curiosity_critic_optimizer.zero_grad(set_to_none=True)
        tm = self.timer
        tm.start(compute_metrics, reset=False)

        all_states = full_state.detach()
        B, T = all_states.shape[:2]
        dev = all_states.device
        A = self.action_dim
        rows, t0, start_ok, forced, n_norm, G, focus = self._dream_starts(start_mask, start_info, B, T, dev)
        num_dreams = rows.shape[0]
        start_states = all_states[rows, t0]
        graph = self._dream_graph_for(num_dreams)
        cache = graph.cache if graph is not None else self.tssm.make_cache(num_dreams, dev, self.cache_dtype)
        self.tssm.prefill(cache, kv_context, rows, t0)                     # real context in the cache

        with self._amp():
            # --- roll out the policy in the world model; events: gate x sparse heads, m follows the gate (event_step)
            states = [start_states]
            log_probs, entropies, acts, events, forgets, ev_rewards, ev_mass = [], [], [], [], [], [], []
            pmpo = c.actor_objective == "pmpo_sign"
            use_prior = pmpo and c.pmpo_prior_beta > 0
            prior_kls = []
            curr = start_states
            _, z, mem = self.split_state(start_states)
            alive = torch.ones(num_dreams, device=dev)                     # no event yet in this dream
            probs0 = None
            for i in range(horizon):
                action, logprob, entropy, probs = self.actor(curr, return_probs=True)
                if use_prior:                                              # KL(pi || EMA actor) at this dream state
                    with torch.no_grad():
                        pp = self.actor_prior.probs(curr)
                    prior_kls.append((probs * (probs.clamp_min(1e-8).log() - pp.clamp_min(1e-8).log())).sum(-1))
                if i == 0:
                    probs0 = probs.detach()
                    if G > 0:                                              # branch dreams: given first action
                        a_f = F.one_hot(forced.clamp(min=0), A).float()
                        is_b = forced >= 0
                        action = torch.where(is_b[:, None], a_f, action)
                        logprob = torch.where(is_b, (probs.clamp_min(1e-8).log() * a_f).sum(-1), logprob)
                with torch.no_grad():
                    if graph is not None:
                        h, z, _ = graph(action.detach(), z, mem, unimix=c.dream_unimix)
                    else:
                        h, z, _ = self.dream_step(z, action.detach(), cache, mem)
                    mem_logits = self.tokenizer.memory_head(z)
                    held = mem.sum(-1)
                    forgets.append(((torch.sigmoid(mem_logits) <= c.memory_threshold).float() * mem).sum(-1)
                                   / held.clamp(min=1.0))
                    before = alive
                    ev_r, ev_p, mem, alive, new = self.event_step(h, z, mem, alive, mem_logits)
                    ev_mass.append(before * ev_p)                          # expected events of this step
                    ev_rewards.append(ev_r)
                    events.append(new.sum(-1))
                    curr = torch.nan_to_num(self.full_state(h, z, mem), nan=0.0, posinf=1e4, neginf=-1e4)
                states.append(curr)
                log_probs.append(logprob)
                entropies.append(entropy)
                acts.append(action)
            states = torch.stack(states, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            entropies = torch.stack(entropies, dim=1)
            acts = torch.stack(acts, dim=1)
            events = torch.stack(events, dim=1)                            # [N, H] new memory bits (gate samples)
            ev_rewards = torch.stack(ev_rewards, dim=1)                    # [N, H] sparse head x P(gate) x alive
            ev_mass = torch.stack(ev_mass, dim=1)                          # [N, H] P(gate) x alive
            forgets = torch.stack(forgets, dim=1)

            # --- predicted rewards (event heads + standard reward head) and curiosity; no death: full weights
            with torch.no_grad():
                # curiosity = predicted novelty gated by "square not visited yet" (grid head)
                event_rewards, standard_pred, pred_cur = self.dream_rewards(states[:, 1:], ev_rewards)
                pred_rewards = event_rewards + standard_pred
                weights = torch.ones_like(pred_rewards) * start_ok[:, None]
                cont = torch.full_like(pred_rewards, c.continue_discount)

            # --- returns and advantages
            imagined = states.detach()
            critic_logits = self.critic(imagined)
            cur_critic_logits = self.curiosity_critic(imagined)
            with torch.no_grad():
                values = self.two_hot.decode(self.critic_target(imagined)).squeeze(-1)
                cur_values = self.two_hot.decode(self.curiosity_critic_target(imagined)).squeeze(-1)
                lam = lambda_returns(pred_rewards, values, cont)
                cur_lam = lambda_returns(pred_cur, cur_values, cont)
                adv = lam - values[:, :-1]
                cur_adv = cur_lam - cur_values[:, :-1]
                crit_w = weights
                if G > 0:                                                  # branch step 0: group-mean baseline
                    for a_, l_ in ((adv, lam), (cur_adv, cur_lam)):
                        g_ = l_[n_norm:, 0].view(G, A)
                        a_[n_norm:, 0] = (g_ - g_.mean(1, keepdim=True)).flatten()
                    crit_w = weights.clone()
                    crit_w[n_norm:, 0] = 0.0
                combined, adv_info = self._combine_advantages(adv, cur_adv, lam, cur_lam, weights)
                s_ext, s_cur = adv_info["scale_ext"], adv_info["scale_cur"]
                forced_first = None
                if G > 0:                                                  # branch dreams: their forced first action
                    forced_first = (n_norm, probs0[n_norm:].gather(1, forced[n_norm:, None]).squeeze(1))

            # --- losses: actor (PMPO sign per stream + entropy + KL to the EMA actor, or REINFORCE), critics
            prior_kl = torch.zeros((), device=dev)
            if pmpo:
                # each stream's raw advantage A = lambda - V (branch step 0: group-mean baseline); only its sign counts,
                # so neither the streams' scale nor the advantage normalisation matters
                actor_loss = (pmpo_loss(adv, log_probs, c.pmpo_alpha, weights)
                              + c.curiosity_scale * pmpo_loss(cur_adv, log_probs, c.pmpo_alpha, weights)
                              - c.pmpo_entropy_scale * masked_mean(entropies, weights))
                if use_prior:
                    prior_kl = masked_mean(torch.stack(prior_kls, dim=1), weights)
                    actor_loss = actor_loss + c.pmpo_prior_beta * prior_kl
            else:
                actor_loss, _ = reinforce_loss(combined, log_probs, entropies, weights, c.reinforce_entropy_scale,
                                               self.action_dim, forced_first)
            critic_loss = self.two_hot.loss(critic_logits[:, :-1], lam, crit_w)
            cur_critic_loss = self.two_hot.loss(cur_critic_logits[:, :-1], cur_lam, crit_w)

            # replay value loss: the critic also regresses lambda-returns of real rewards along the real sequence
            use_repval = real is not None and c.w_repval > 0 and T > 1
            if use_repval:
                rv = real["valid"].float()
                with torch.no_grad():
                    real_values = self.two_hot.decode(self.critic_target(all_states)).squeeze(-1)
                    real_rew = real["rewards"][:, :T - 1].float()
                    step_ok = rv[:, 1:]                                    # next frame real
                    real_lam = lambda_returns(real_rew, real_values, torch.full_like(real_rew, c.continue_discount),
                                              valid=step_ok)
                real_logits = self.critic(all_states[:, :-1])
                repval_loss = self.two_hot.loss(real_logits, real_lam, step_ok)
            else:
                repval_loss = torch.zeros((), device=dev)

        self._optimize(critic_loss + c.w_repval * repval_loss, "critic_optimizer", self.critic.parameters(), 1.0)
        self._optimize(cur_critic_loss, "curiosity_critic_optimizer", self.curiosity_critic.parameters(), 1.0)
        self._optimize(actor_loss, "actor_optimizer", self.actor.parameters(), 1.0)
        self._critic_ema_update()
        if c.actor_objective == "pmpo_sign" and c.pmpo_prior_beta > 0:
            with torch.no_grad():                                          # the EMA actor follows the actor slowly
                for pp_, pa_ in zip(self.actor_prior.parameters(), self.actor.parameters()):
                    pp_.lerp_(pa_, c.pmpo_prior_rate)
        tm.mark("dream + actor/critic")

        metrics = {}
        if compute_metrics:
            ev_per_dream = (ev_mass * weights).sum(1)                       # expected events per dream (<= 1)
            metrics = {
                "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "curiosity_critic_loss": cur_critic_loss.item(), "entropies": entropies.mean().item(),
                "advantages": adv.mean().item(), "curiosity_advantages": cur_adv.mean().item(),
                "critic_values": self.two_hot.decode(critic_logits.detach()).mean().item(),
                "target_critic_values": values.mean().item(),
                "curiosity_critic_values": self.two_hot.decode(cur_critic_logits.detach()).mean().item(),
                "advantage_pos_fraction": (combined >= 0).float().mean().item(),
                "curiosity_flip_fraction": ((combined >= 0) != (adv >= 0)).float().mean().item(),
                "dream_mean_curiosity": pred_cur.mean().item(),
                "dream_events": ev_per_dream.mean().item(),                # expected events per dream (gate)
                "dream_event_fraction": ((events * weights).sum(1) > 0).float().mean().item(),   # gate fired
                "dream_event_reward": (event_rewards * weights).sum(1).mean().item(),          # per dream
                "dream_memory_forget": forgets.mean().item(),              # set bits M(z) no longer reports
                "replay_value_loss": repval_loss.item(),
                "replay_critic_values": self.two_hot.decode(real_logits.detach()).mean().item() if use_repval else 0.0,
                "replay_returns": masked_mean(real_lam, step_ok).item() if use_repval else 0.0,
                "adv_scale": float(s_ext),                                 # reward stream: EMA return range (range) / std
                "cur_adv_scale": float(s_cur),                             # curiosity stream: the same
                **actor_metrics(combined, weights),
                "dream_start_battle": focus[rows, t0].mean().item(),
                "adv_combined_std": float(adv_info["combined_std"]),      # the combined advantage's std
                "prior_kl": prior_kl.item(),                               # pmpo_sign: KL(pi || EMA actor)
                "ext_pos_fraction": masked_mean((adv >= 0).float(), weights).item(),     # PMPO accept shares
                "cur_pos_fraction": masked_mean((cur_adv >= 0).float(), weights).item(),
            }
            with torch.no_grad():                                          # per stream: is each critic learning?
                for name, logits, target in (("ext", critic_logits, lam), ("cur", cur_critic_logits, cur_lam)):
                    pred = self.two_hot.decode(logits[:, :-1].detach()).squeeze(-1)
                    metrics[f"{name}_critic_ev"] = explained_variance(pred, target, crit_w)
                    for k in ("adv_std", "return_range", "adv_share"):
                        metrics[f"{name}_{k}"] = float(adv_info[name][k])
            if G > 0:
                c0 = combined[n_norm:, 0].view(G, A)
                pol = probs0[n_norm::A].argmax(-1)                         # policy's choice at each group start
                metrics["branch_agree"] = (c0.argmax(1) == pol).float().mean().item()
                metrics["branch_spread"] = (c0.max(1).values - c0.min(1).values).mean().item()
        if not pack_dreams:
            return metrics, None, None

        def pack(i, metric, label):
            """Dream i as CPU tensors for the PDF."""
            return {"metric": metric, "label": label, "states": states[i].detach().float().cpu(),
                    "rewards": pred_rewards[i].detach().float().cpu(), "events": events[i].detach().float().cpu(),
                    "actions": acts[i].detach().cpu(), "advantages": combined[i].detach().float().cpu()}

        traj = (combined * weights).sum(dim=1)
        if c.sync_free:                                                    # GPU copies; to CPU when the round ends
            def keep(i, metric, label):
                return {"metric": metric, "label": label, "states": states[i].detach().float().clone(),
                        "rewards": pred_rewards[i].detach().float().clone(),
                        "events": events[i].detach().float().clone(), "actions": acts[i].detach().clone(),
                        "advantages": combined[i].detach().float().clone()}
            best_idx = torch.argmax(traj)
            rand_idx = torch.randint(0, traj.shape[0], (1,)).item()        # CPU generator: no sync
            return metrics, keep(best_idx, traj[best_idx].detach().clone(), "MaxAdvantage"), \
                keep(rand_idx, traj[rand_idx].detach().clone(), "Random")
        best_idx = torch.argmax(traj).item()
        rand_idx = torch.randint(0, traj.shape[0], (1,)).item()
        return metrics, pack(best_idx, traj[best_idx].item(), "MaxAdvantage"), \
            pack(rand_idx, traj[rand_idx].item(), "Random")

    # --- Dream audit (diagnostics) ----------------------------------------------------------------------------------
    def audit(self, seed=None):
        """Dream audit: from the last play's snapshots, the same states are played audit_horizon steps in the dream and
        in the real game with the actor's greedy action -> metrics ({} without a usable snapshot). Uses its own seeded
        generators; trains nothing."""
        c, t_start = self.cfg, time.perf_counter()
        if c.audit_horizon <= 0:
            return {}
        if seed is None:
            seed = int(np.random.SeedSequence([c.seed, 0xA4D17, self.total_num_updates]).generate_state(1)[0])
        items = self.audit_snapshots.usable(self.buffer, c.context_length)
        picks = select_starts(items, c.audit_starts, np.random.default_rng(seed))
        if not picks:
            return {}
        starts = [items[i] for i, _ in picks]
        kinds = [AUDIT_KINDS.index(k) for _, k in picks]
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        with torch.no_grad(), eager_modules():
            ctx = self.audit_context([s["row"] for s in starts], [s["step"] for s in starts], gen)
            dream = self.audit_dream(ctx, c.audit_horizon, gen)
            real = self.audit_real(ctx, [s["snapshot"] for s in starts], c.audit_horizon, gen)
            dream.update(self._audit_compare(dream, real, ctx))
        metrics = audit_metrics(dream, real, kinds, c.continue_discount)
        metrics["seconds"] = time.perf_counter() - t_start
        return metrics

    def _audit_sample(self, logp, gen):
        """One-hot samples [..., L*K] of log-probs [..., L, K] (Gumbel-max) with the audit's generator."""
        u = torch.rand(logp.shape, generator=gen, device=logp.device).clamp_(1e-6, 1.0 - 1e-6)
        return sample_onehot(logp.float(), straight_through=False, gumbel=-torch.log(-torch.log(u))).flatten(-2)

    def _audit_encode(self, obs, gen, chunk=1024):
        """Posterior samples [N, L*K] of encoder inputs {key: [N, ...]} with the audit's generator."""
        n = obs["frame"].shape[0]
        zs = []
        for i in range(0, n, chunk):
            with self._amp():
                logp = self.tokenizer.posterior({k: obs[k][i:i + chunk] for k in self.OBS_KEYS})
            zs.append(self._audit_sample(logp, gen))
        return torch.cat(zs)

    def _buffer_obs(self, rows):
        """Encoder inputs (+ sprite overlay) of replay rows [N] on the device."""
        b, dev = self.buffer, self.device
        return {"frame": b.frames[rows].to(dev), "palette": b.palettes[rows].to(dev),
                "grid": stored_grid(b.grids[rows].to(dev)),
                "map_id": b.map_ids[rows].long().to(dev), "ram": b.rams[rows].to(dev),
                "memory": unpack_bits(b.memories[rows].to(dev), MEMORY_DIM), "blocks": b.blocks[rows].long().to(dev)}

    def _audit_action(self, s):
        """The actor's greedy one-hot action for states s."""
        return self.actor(s, greedy=True)[0].float()

    @torch.no_grad()
    def audit_context(self, rows, steps, gen):
        """Real context of N starts (replay rows, episode steps) -> {"kv", "t0" [N], "start" [N, S], "obs0"}: the
        context_length frames before each start encoded and run through the TSSM (with their real memory), the start
        state [h, z, m] and the start frame itself (its reconstruction is the audit's error floor)."""
        c, buf, dev = self.cfg, self.buffer, self.device
        W = c.context_length
        rows = torch.as_tensor(rows, dtype=torch.long)
        N = rows.shape[0]
        n_ctx = torch.as_tensor(steps, dtype=torch.long).clamp(min=1, max=W)          # tokens before the start
        j = torch.arange(W + 1)[None]
        src = (rows[:, None] - n_ctx[:, None] + torch.minimum(j, n_ctx[:, None])) % buf.capacity   # pads: the start
        obs = self._buffer_obs(src.flatten())
        zf = self._audit_encode(obs, gen).view(N, W + 1, self.latent_dim)
        mem = obs["memory"].view(N, W + 1, self.mem_dim)
        actions = F.one_hot(buf.actions[src].long(), self.action_dim).float().to(dev)
        h, kv = self.context_kv(zf, actions, mem)                                      # h[:, j]: after token j
        idx = torch.arange(N, device=dev)
        t0 = (n_ctx - 1).to(dev)
        start = torch.nan_to_num(self.full_state(h[idx, t0], zf[idx, t0 + 1], mem[idx, t0 + 1]), nan=0.0, posinf=1e4,
                                 neginf=-1e4)
        obs0 = self._buffer_obs(rows % buf.capacity)
        return {"kv": kv, "t0": t0, "start": start, "obs0": obs0}

    @torch.no_grad()
    def audit_dream(self, ctx, horizon, gen):
        """The audit's dreams from audit_context with the greedy action and the dream events of imagine (event_step,
        the gate sampled with the audit's generator) -> {"r" [N, H, 3] (event, standard, curiosity, as imagine
        computes them), "events" [N, H] (expected events: P(gate) x alive, at most 1 per dream), "frames"
        [N, H + 1, 3, 72, 80] (decoded, step 0 = the start latent), "map" / "battle" [N, H] read from the dreamed
        latents, "grid" [N, H, 25] (explored channel of the grid head), "memory" [N, H, 144] (the dream's m)}."""
        c, dev, N = self.cfg, self.device, ctx["start"].shape[0]
        cache = self.tssm.make_cache(N, dev, self.cache_dtype)
        self.tssm.prefill(cache, ctx["kv"], torch.arange(N, device=dev), ctx["t0"])
        curr = ctx["start"]
        _, z, mem = self.split_state(curr)
        alive = torch.ones(N, device=dev)
        states, events, ev_rewards = [curr], [], []
        with self._amp():
            for _ in range(horizon):
                h = self.tssm.forward_step(z, self._audit_action(curr), cache, mem).float()
                z = self._audit_sample(unimix_logprobs(self.prior(h), c.dream_unimix), gen)
                before = alive
                ev_r, ev_p, mem, alive, _ = self.event_step(h, z, mem, alive, generator=gen)
                events.append(before * ev_p)
                ev_rewards.append(ev_r)
                curr = torch.nan_to_num(self.full_state(h, z, mem), nan=0.0, posinf=1e4, neginf=-1e4)
                states.append(curr)
            states = torch.stack(states, dim=1)                                    # [N, H + 1, S]
            events = torch.stack(events, dim=1)
            r = torch.stack(self.dream_rewards(states[:, 1:], torch.stack(ev_rewards, dim=1)), dim=-1).float()
            _, zs, ms = self.split_state(states)
            frames = torch.cat([torch.sigmoid(self.tokenizer.decode(zz).float())
                                for zz in zs.flatten(0, 1).split(512)]).view(N, horizon + 1, 3, IMG_H, IMG_W)
            map_id, battle = self.tokenizer.heads.predict(zs[:, 1:])
            grid = self.grid_head(states[:, 1:]).float()[..., :GRID_DIM] > 0
        return {"r": r, "events": events, "frames": frames, "map": map_id, "battle": battle, "grid": grid,
                "memory": ms[:, 1:] > 0.5}

    def _audit_envs(self):
        """Emulators for the real rollouts: the training envs when they can restore snapshots (each start restored
        once, then played in lockstep like play(); play() resets them afterwards), else one own headless emulator
        that plays the starts one after another."""
        if self.envs is not None and hasattr(self.envs, "restore"):
            return self.envs
        if self._audit_env is None:
            from .env import PokemonRedEnv, SequentialPokemonEnvs
            c = self.cfg
            rom = getattr(self.envs, "rom_path", "game/PokemonRed.gb")
            state = getattr(self.envs, "state_path", "game/PokemonRed.Start.state")
            self._audit_env = SequentialPokemonEnvs([PokemonRedEnv(
                rom_path=rom, state_path=state, window="null", verbose=False, frames_per_step=c.frames_per_step,
                episode_steps=c.episode_steps, episode_extension=c.episode_extension,
                cross_episode_novelty=c.cross_episode_novelty, novelty_min_rarity=c.novelty_min_rarity,
                novelty_zero_visits=c.novelty_zero_visits, auto_respawn=c.auto_respawn)])
        return self._audit_env

    @torch.no_grad()
    def audit_real(self, ctx, snapshots, horizon, gen):
        """The real rollouts: in batches of the envs' size, each env restores one start's snapshot and the batch is
        played in lockstep with the greedy action of the acting loop (encoder on the real frames, TSSM step with the
        real memory) until each episode ends -> {"r" [N, H, 3] (event = sparse, standard, curiosity), "events" [N, H]
        (new memory bits), "steps" [N], "obs" (encoder inputs + sprite overlay of every step, [N, H, ...])}."""
        dev, N, H = self.device, len(snapshots), int(horizon)
        envs = self._audit_envs()
        self._push_novelty(envs)                                           # the counts the snapshots were played with
        E = len(envs)
        r = np.zeros((N, H, 3), dtype=np.float32)
        events = np.zeros((N, H), dtype=np.float32)
        steps = np.full(N, H, dtype=np.int64)
        per_step = [[None] * N for _ in range(H)]                          # observation of start i after step k
        for b0 in range(0, N, E):
            sel = list(range(b0, min(b0 + E, N)))
            n = len(sel)
            envs.restore([snapshots[i] for i in sel] + [None] * (E - n))
            idx_t = torch.as_tensor(sel, device=dev)
            cache = self.tssm.make_cache(n, dev, self.cache_dtype)
            self.tssm.prefill(cache, [(k[idx_t], v[idx_t]) for k, v in ctx["kv"]], torch.arange(n, device=dev),
                              ctx["t0"][idx_t])
            s = ctx["start"][idx_t]
            alive, last = [True] * n, [None] * n
            for k in range(H):
                with self._amp():
                    action = self._audit_action(s)
                    _, z, mem = self.split_state(s)
                    h = self.tssm.forward_step(z, action, cache, mem).float()
                act = action.argmax(-1).cpu().numpy()
                results = envs.step(np.concatenate([act, np.zeros(E - n, dtype=act.dtype)]),
                                    active=alive + [False] * (E - n))
                for j, i in enumerate(sel):
                    if not alive[j]:
                        continue
                    o, _, terminated, truncated, info = results[j]
                    r[i, k] = (info.get("sparse_reward", 0.0), info.get("standard_reward", 0.0),
                               info.get("curiosity", 0.0))
                    events[i, k] = round(info.get("sparse_reward", 0.0) / MEMORY_REWARD)   # new memory bits
                    last[j] = o
                    if terminated or truncated:
                        alive[j], steps[i] = False, k + 1
                for j, i in enumerate(sel):
                    per_step[k][i] = last[j]
                if not any(alive):
                    for kk in range(k + 1, H):                             # pad: never compared (steps)
                        for j, i in enumerate(sel):
                            per_step[kk][i] = last[j]
                    break
                obs_t = obs_to_tensors(last, dev)
                s = self.full_state(h, self._audit_encode(obs_t, gen), obs_t["memory"])
        seen = []
        for k in range(H):
            o = obs_to_tensors(per_step[k], dev)
            o["blocks"] = torch.from_numpy(np.stack([x["blocks"] for x in per_step[k]])).long().to(dev)
            seen.append(o)
        obs = {key: torch.stack([o[key] for o in seen], dim=1) for key in seen[0]}
        return {"r": torch.as_tensor(r), "events": torch.as_tensor(events), "steps": steps, "obs": obs}

    @torch.no_grad()
    def _audit_compare(self, dream, real, ctx):
        """Dream vs real frames and labels -> {"frame_err" / "sprite_err" [N, H + 1] (RMSE of the decoded frame vs the
        real one, all pixels / sprite squares; step 0 = the start frame's reconstruction), "map_hit" / "battle_hit" /
        "grid_hit" / "memory_hit" [N, H]}."""
        obs, frames = real["obs"], dream["frames"]
        N, H = frames.shape[0], frames.shape[1] - 1
        ctx_obs = ctx["obs0"]                                              # the start frames
        fr = torch.cat([ctx_obs["frame"][:, None], obs["frame"]], dim=1)
        pal = torch.cat([ctx_obs["palette"][:, None], obs["palette"]], dim=1)
        blocks = torch.cat([ctx_obs["blocks"][:, None], obs["blocks"]], dim=1)
        rgb = frame_rgb(fr.flatten(0, 1), pal.flatten(0, 1)).view(N, H + 1, 3, IMG_H, IMG_W)
        sprite = sprite_pixel_mask(blocks.flatten(0, 1)).view(N, H + 1, IMG_H, IMG_W)
        f, rr, sp = frames.flatten(0, 1), rgb.flatten(0, 1), sprite.flatten(0, 1)
        frame_err = rmse(f, rr).view(N, H + 1)
        sprite_err = rmse(f, rr, sp).view(N, H + 1)
        battle = ram_fields(obs["ram"])["mode"][..., 0] > 0.5
        explored = obs["grid"][..., :GRID_DIM] > 0.5
        return {"frame_err": frame_err, "sprite_err": sprite_err,
                "map_hit": (dream["map"] == obs["map_id"]).float(), "battle_hit": (dream["battle"] == battle).float(),
                "grid_hit": (dream["grid"] == explored).float().mean(-1),
                "memory_hit": (dream["memory"] == (obs["memory"] > 0.5)).all(-1).float()}

    # --- Acting in the game -----------------------------------------------------------------------------------------
    def new_act_context(self, n):
        """Per-env acting state: TSSM cache and h (the static buffers of the acting CUDA graph when it is used)."""
        g = self._act_graph_for(n)
        if g is not None:
            g.reset()
            return {"cache": g.cache, "h": g.h, "graph": g}
        return {"cache": self.tssm.make_cache(n, self.device, self.cache_dtype),
                "h": torch.zeros(n, self.tssm_dim, device=self.device)}

    def _act_graph_for(self, n):
        """CUDA graph of the acting step for n envs, or None to act eagerly (CPU, disabled or capture failed)."""
        if not self._use_act_graph or self.device.type != "cuda":
            return None
        g = self._act_graphs.get(n)
        if g is None:
            try:
                t = time.perf_counter()
                g = GraphedActStep(self, n)
            except Exception as e:                                         # never let the graph stop training
                print(f"[!] acting CUDA graph disabled ({type(e).__name__}: {e}); acting runs eagerly.")
                self._use_act_graph = False
                return None
            self._act_graphs[n] = g
            print(f"\033[90m    [setup] acting: CUDA graph of one acting step captured and checked for {n} envs in "
                  f"{time.perf_counter() - t:.1f}s\033[0m")
        return g

    @staticmethod
    def reset_act_rows(ctx, rows):
        """Clears the acting state of envs that start a new episode."""
        ctx["cache"].reset_rows(rows)
        ctx["h"][rows] = 0.0

    @torch.no_grad()
    def act(self, observations, ctx, greedy=False, force=None):
        """Encode observations, sample actions from [h_t, z_t, m_t], advance the TSSM -> one-hot actions [N, A].
        force: optional int array [N], an action index replaces the policy's action (-1: keep it). The TSSM steps with
        the action actually taken."""
        if force is not None and not (np.asarray(force) >= 0).any():
            force = None
        g = ctx.get("graph")
        if g is not None and not greedy:
            return g(observations, force)
        obs = obs_to_tensors(observations, self.device)
        mem = obs["memory"]
        with self._amp():
            z, _ = self.tokenizer.encode(obs)
            z = z.float()
            action, _, _ = self.actor(self.full_state(ctx["h"], z, mem), greedy=greedy, unimix=self.cfg.play_unimix)
            if force is not None:
                rows = torch.as_tensor(np.flatnonzero(np.asarray(force) >= 0), device=self.device)
                fa = torch.as_tensor(np.asarray(force)[np.asarray(force) >= 0], device=self.device)
                action = action.clone()
                action[rows] = F.one_hot(fa, self.action_dim).to(action.dtype)
            h = self.tssm.forward_step(z, action, ctx["cache"], mem).float()
            if g is not None:
                ctx["h"].copy_(h)                                          # the graph's static h
            else:
                ctx["h"] = h
        return action

    def _push_novelty(self, envs):
        """Gives the envs the cross-episode counts (envs without set_novelty are left as they are)."""
        if hasattr(envs, "set_novelty"):
            envs.set_novelty(self.novelty_tiles, self.novelty_maps)

    def _merge_novelty(self, envs):
        """Adds the squares / maps the envs' episodes visited since the last push to the cross-episode counts."""
        if not hasattr(envs, "pop_novelty"):
            return
        for tiles, maps in envs.pop_novelty():
            for k, v in tiles.items():
                self.novelty_tiles[k] = self.novelty_tiles.get(k, 0) + v
            for k, v in maps.items():
                self.novelty_maps[k] = self.novelty_maps.get(k, 0) + v

    @torch.no_grad()
    def play(self, episodes_per_env=1, progress_every=None, snapshots=False):
        """Every env plays `episodes_per_env` whole episodes with the current policy (sampled actions, play_unimix);
        each episode goes to the buffer when it ends -> (mean reward, mean curiosity). snapshots: every
        audit_snapshot_every steps each env also returns its complete state, kept (labelled pre-event or not) for the
        next dream audit."""
        pool = self.audit_snapshots
        snap_on = bool(snapshots) and pool.every > 0
        if snap_on:
            pool.clear()
        progress_every = self.cfg.play_progress_every if progress_every is None else progress_every
        num_envs = len(self.envs)
        self.buffer.mark_round_start()
        self._push_novelty(self.envs)                                      # counts of the earlier episodes
        episodes_completed = [0] * num_envs
        local_buffers = [[] for _ in range(num_envs)]
        event_rows = [[] for _ in range(num_envs)]                         # rows of the observations with a new bit
        cur = [{"reward": 0.0, "curiosity": 0.0} for _ in range(num_envs)]
        episodes = []                                                      # per-episode summaries
        last_info = [None] * num_envs

        ctx = self.new_act_context(num_envs)
        observations = []
        for i, (obs, info) in enumerate(self.envs.reset()):
            observations.append(obs)
            last_info[i] = info

        t_model = t_env = t_buffer = 0.0
        t_play = time.perf_counter()
        step = 0
        while min(episodes_completed) < episodes_per_env:
            t0 = time.perf_counter()
            action_idxs = torch.argmax(self.act(observations, ctx), dim=-1).cpu().numpy()
            t1 = time.perf_counter()
            t_model += t1 - t0

            active = [episodes_completed[i] < episodes_per_env for i in range(num_envs)]
            if snap_on:                                    # the post-step state = row len + 1 of the episode
                flags = [active[i] and pool.due(len(local_buffers[i]) + 1) for i in range(num_envs)]
                results = self.envs.step(action_idxs, active=active, snapshot=flags)
            else:
                results = self.envs.step(action_idxs, active=active)
            t_env += time.perf_counter() - t1
            step += 1
            envs_to_reset = []
            for i in range(num_envs):
                if not active[i]:
                    continue
                next_obs, reward, terminated, truncated, info = results[i]
                for msg in info.get("logs", ()):
                    if self.verbose or "[EVENT]" in msg:
                        print(f"    [Env {i+1}] {msg}")
                self.total_num_steps += 1
                if "snapshot" in info:
                    pool.add(i, len(local_buffers[i]) + 1, info.pop("snapshot"))
                cur[i]["reward"] += reward
                cur[i]["curiosity"] += info.get("curiosity", 0.0)                 # novelty of new squares
                local_buffers[i].append((observations[i], int(action_idxs[i]), info.get("sparse_reward", 0.0),
                                         info.get("standard_reward", 0.0), info.get("curiosity", 0.0), 1.0,
                                         info.get("blackout", False)))
                if info.get("sparse_reward", 0.0) > 0:
                    event_rows[i].append(len(local_buffers[i]))            # next_obs: the first with the new bit
                observations[i] = next_obs
                last_info[i] = info
                if terminated or truncated:                                # truncation only (budget): not terminal
                    tb = time.perf_counter()
                    self.wait_for_save()                                   # async_save: the buffer file is written
                    start_row, episode_id = self.buffer.index, self.buffer._episode_counter
                    self.buffer.add_episode(local_buffers[i])
                    if snap_on:
                        pool.close_episode(i, start_row, episode_id, len(local_buffers[i]), event_rows[i],
                                           self.cfg.audit_event_window)
                    t_buffer += time.perf_counter() - tb
                    local_buffers[i] = []
                    event_rows[i] = []
                    ep = {"reward": cur[i]["reward"], "curiosity": cur[i]["curiosity"], "steps": info["steps"],
                          "events": info["events"], "badges": info["badges"], "party_level": info["party_level"],
                          "maps": info["maps"], "blackouts": info["blackouts"]}
                    episodes.append(ep)
                    if self.verbose:
                        print(f"    [Env {i+1}] Episode done | Reward: {ep['reward']:.2f} | Curiosity: "
                              f"{ep['curiosity']:.3f} | Steps: {ep['steps']} | Events: {ep['events']} | Maps: "
                              f"{ep['maps']} | Blackouts: {ep['blackouts']}")
                    self.total_num_episodes += 1
                    episodes_completed[i] += 1
                    if episodes_completed[i] < episodes_per_env:
                        envs_to_reset.append(i)
                        cur[i] = {"reward": 0.0, "curiosity": 0.0}
                        self.reset_act_rows(ctx, i)
            tr = time.perf_counter()
            for i in envs_to_reset:
                observations[i], last_info[i] = self.envs.reset_one(i)
            t_env += time.perf_counter() - tr
            if progress_every and step % progress_every == 0:
                running = sum(a for a in active)
                infos = [last_info[i] for i in range(num_envs) if active[i] and last_info[i] is not None]
                ev = max((x["events"] for x in infos), default=0)
                maps = max((x["maps"] for x in infos), default=0)
                print(f"\033[90m    [play] step {step:,} | {running} / {num_envs} envs still playing | best "
                      f"running episode: {ev} events, {maps} maps | {time.perf_counter() - t_play:.0f} s\033[0m")

        for i in range(num_envs):
            if local_buffers[i]:
                self.wait_for_save()
                self.buffer.add_episode(local_buffers[i])
        if snap_on:
            pool.drop_open()
        self._merge_novelty(self.envs)                                     # this round's episodes join the counts

        if episodes:
            n = len(episodes)
            ls = sorted(e["steps"] for e in episodes)
            total_steps = sum(ls)
            blackouts = sum(e["blackouts"] for e in episodes)
            print(f"\033[90m    [collect] {total_steps:,} env steps in {n} episodes | length min {ls[0]} / max "
                  f"{ls[-1]} | events mean {sum(e['events'] for e in episodes) / n:.1f} / best "
                  f"{max(e['events'] for e in episodes)} | blackouts {blackouts}\033[0m")
            self.last_play_stats = {
                "episodes": n, "episode_length": total_steps / n, "steps": total_steps,
                "t_play": time.perf_counter() - t_play, "t_model": t_model, "t_env": t_env, "t_buffer": t_buffer,
                "best_reward": max(e["reward"] for e in episodes),
                "best_curiosity": max(e["curiosity"] for e in episodes),
                "mean_events": sum(e["events"] for e in episodes) / n,
                "best_events": max(e["events"] for e in episodes),
                "best_badges": max(e["badges"] for e in episodes),
                "best_party_level": max(e["party_level"] for e in episodes),
                "mean_party_level": sum(e["party_level"] for e in episodes) / n,
                "mean_maps": sum(e["maps"] for e in episodes) / n,
                "best_maps": max(e["maps"] for e in episodes),
                "blackout_rate": 1000.0 * blackouts / max(total_steps, 1),
                "known_tiles": len(self.novelty_tiles), "known_maps": len(self.novelty_maps),
                "audit_snapshots": len(pool) if snap_on else 0,
                "audit_pre_event_snapshots": sum(it["pre_event"] for it in pool.items) if snap_on else 0,
            }
        avg_score = round(sum(e["reward"] for e in episodes) / len(episodes), 2) if episodes else 0.0
        avg_cur = round(sum(e["curiosity"] for e in episodes) / len(episodes), 4) if episodes else 0.0
        return avg_score, avg_cur


class GraphedActStep:
    """agent.act for n envs (sampled actions) as one CUDA graph: encoder posterior -> z sample -> actor -> action
    sample -> forced actions (if any) -> TSSM step on a static cache (with the real memory m). The eager
    step launches hundreds of small kernels (a few ms on Windows); the graph replays them in one call. The Gumbel noise
    of both samples is drawn outside the graph (same distributions as the eager path: Gumbel-max of the posterior / of
    log pi), so the graph itself is deterministic and is checked against the eager computation at capture."""

    def __init__(self, agent, n, warmup=3):
        from .game import FRAME_BYTES_W, FRAME_H, PALETTE_SIZE, RAM_DIM
        self.agent, self.n = agent, n
        dev = self.device = agent.device
        self.cache = agent.tssm.make_cache(n, dev, agent.cache_dtype)
        self.h = torch.zeros(n, agent.tssm_dim, device=dev)
        spec = {"frame": ((n, FRAME_H, FRAME_BYTES_W), torch.uint8), "palette": ((n, PALETTE_SIZE, 3), torch.uint8),
                "grid": ((n, GRID_INPUT_DIM), torch.float32), "map_id": ((n,), torch.long),
                "ram": ((n, RAM_DIM), torch.uint8),
                "memory": ((n, MEMORY_DIM), torch.float32)}
        self.obs = {k: torch.zeros(sh, dtype=dt, device=dev) for k, (sh, dt) in spec.items()}
        self.staging = {k: torch.zeros(sh, dtype=dt, pin_memory=True) for k, (sh, dt) in spec.items()}
        self.gz = torch.zeros(n, agent.n_latents, agent.classes, device=dev)
        self.ga = torch.zeros(n, agent.action_dim, device=dev)
        self.force_a = torch.zeros(n, agent.action_dim, device=dev)              # one-hot forced action
        self.force_m = torch.zeros(n, 1, dtype=torch.bool, device=dev)           # row forced
        self.force_staging = (torch.zeros(n, agent.action_dim, pin_memory=True),
                              torch.zeros(n, 1, dtype=torch.bool, pin_memory=True))
        self.graph = None
        self._capture(warmup)

    @torch.no_grad()
    def _body(self):
        ag = self.agent
        with eager_modules(), torch.autocast(self.device.type, dtype=ag.amp_dtype or torch.bfloat16,
                                             enabled=ag.amp_dtype is not None, cache_enabled=False):
            mem = self.obs["memory"]
            logp = ag.tokenizer.posterior(self.obs)
            z = torch.zeros_like(logp).scatter_(-1, (logp + self.gz).argmax(-1, keepdim=True), 1.0).flatten(1)
            probs = ag.actor.probs(ag.full_state(self.h, z, mem), unimix=ag.cfg.play_unimix)
            a = torch.zeros_like(probs).scatter_(-1, (probs.clamp_min(1e-12).log() + self.ga).argmax(-1, keepdim=True),
                                                 1.0)
            a = torch.where(self.force_m, self.force_a.to(a.dtype), a)
            self.h.copy_(ag.tssm.forward_step(z, a, self.cache, mem).float())
        return a

    def _noise(self):
        self.gz.copy_(gumbel_like(self.gz))
        self.ga.copy_(gumbel_like(self.ga))

    def _randomize(self):
        g = self.obs
        g["frame"].random_(0, 256)
        g["palette"].random_(0, 256)
        g["grid"].copy_((torch.rand(self.n, GRID_DIM, device=self.device) < 0.5).float())
        g["ram"].zero_()
        g["memory"].copy_((torch.rand_like(g["memory"]) < 0.1).float())
        self.force_m.copy_(torch.rand(self.n, 1, device=self.device) < 0.5)
        self.force_a.copy_(F.one_hot(torch.randint(0, self.agent.action_dim, (self.n,), device=self.device),
                                     self.agent.action_dim).float())
        self._noise()

    def _capture(self, warmup):
        """Warms up, captures the graph and checks it against the eager step."""
        snap, h0 = self.cache.snapshot(), self.h.clone()

        def restore():
            self.cache.restore(snap)
            self.h.copy_(h0)
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._randomize()
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        restore()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self.a_out = self._body()
        restore()
        self._randomize()
        graph.replay()
        a_g, h_g = self.a_out.clone(), self.h.clone()
        restore()
        a_e = self._body()
        err = (self.h - h_g).abs().max().item()
        restore()
        tol = 5e-2 if self.agent.amp_dtype is not None else 1e-3
        if not (bool((a_e == a_g).all()) and err <= tol):
            raise RuntimeError(f"graph replay differs from the eager step (h error {err:.3g}, tolerance {tol})")
        self.graph = graph

    def reset(self):
        """Empties every row (a new play round)."""
        self.cache.reset_rows(slice(None))
        self.h.zero_()

    def __call__(self, observations, force=None):
        """Env observations (+ optional forced action indices [n], -1 = policy) -> one-hot actions [n, A] (a static
        tensor: read it before the next call)."""
        fa, fm = self.force_staging
        fa.zero_()
        fm.zero_()
        if force is not None:
            rows = np.flatnonzero(np.asarray(force) >= 0)
            fa.numpy()[rows, np.asarray(force)[rows]] = 1.0
            fm.numpy()[rows, 0] = True
        self.force_a.copy_(fa, non_blocking=True)
        self.force_m.copy_(fm, non_blocking=True)
        st = self.staging
        st["frame"].numpy()[:] = np.stack([o["frame"] for o in observations])
        st["palette"].numpy()[:] = np.stack([o["palette"] for o in observations])
        st["grid"].numpy()[:] = np.stack([grid_input(o) for o in observations])
        st["map_id"].numpy()[:] = [int(o["map_id"]) for o in observations]
        st["ram"].numpy()[:] = np.stack([np.asarray(o["ram"], dtype=np.uint8) for o in observations])
        st["memory"].numpy()[:] = np.stack([o["memory"] for o in observations])
        for k, v in st.items():
            self.obs[k].copy_(v, non_blocking=True)
        self._noise()
        self.graph.replay()
        return self.a_out
