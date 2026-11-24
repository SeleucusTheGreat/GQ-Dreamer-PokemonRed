"""Training / evaluation loop of GQ-Dreamer for Pokémon Red: rounds of one whole episode per env followed by updates,
CSV metrics, plots, dream PDFs, checkpoints, and the notebook entry point run()."""
import csv
import gc
import glob
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .gq_dreamer import GQDreamer
from .audit import AUDIT_COLUMNS, audit_enabled, audit_lines
from .config import model_config
from .env import create_envs
from .exploration_maps import ExplorationMapJob
from .game import FRAMES_PER_STEP
from .plots import find_comparison_runs, plot_reward_comparison, plot_training
from .visualize import save_dreams_pdf

# ANSI styles of the console output
_B, _DIM, _CYA, _GRN = "\033[1m", "\033[90m", "\033[96m", "\033[92m"
_MAG, _YEL, _RST = "\033[95m", "\033[93m", "\033[0m"


def seed_everything(seed):
    """Seeds python, numpy and torch (cuDNN autotuning stays on)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _dreams_to_cpu(dreams):
    """Packed dreams (GPU tensors with sync_free) -> float metric + CPU tensors."""
    return [{k: (float(v) if k == "metric" else v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
            for d in dreams]


class Trainer:
    """Runs GQ-Dreamer in rounds: the agent does updates_per_round world-model + imagination updates, then every env
    plays one whole episode with the trained policy; one CSV row, console summary and checkpoint per round. Only a
    run that starts with an empty replay buffer plays one episode per env before the first round."""

    # CSV column -> (metrics source, metric key)
    COLUMNS = [
        ("bestReward", "play", "best_reward"), ("meanEvents", "play", "mean_events"),
        ("bestEvents", "play", "best_events"), ("meanMaps", "play", "mean_maps"),
        ("blackoutRate", "play", "blackout_rate"), ("episodeLength", "play", "episode_length"),
        ("knownTiles", "play", "known_tiles"),                 # squares any episode of the run stood on (novelty)
        ("worldModelLoss", "wm", "world_model_loss"),          # round mean; = the wm* columns below + curiosity
        ("wmPixel", "wm", "wm_pixel"), ("wmMap", "wm", "wm_map"), ("wmStatus", "wm", "wm_status"),
        ("wmParty", "wm", "wm_party"), ("wmEnemy", "wm", "wm_enemy"), ("wmItems", "wm", "wm_items"),
        ("wmMenu", "wm", "wm_menu"), ("wmMemory", "wm", "wm_memory"), ("wmMemoryPrior", "wm", "wm_memory_prior"),
        ("wmKlDyn", "wm", "wm_kl_dyn"), ("wmKlRep", "wm", "wm_kl_rep"), ("wmRollout", "wm", "wm_rollout"),
        ("wmStandardReward", "wm", "wm_standard_reward"), ("wmEventGate", "wm", "wm_event_gate"),
        ("wmSparseReward", "wm", "wm_sparse_reward"), ("wmGrid", "wm", "wm_grid"),
        ("reconLoss", "wm", "recon_excess"),                  # pixel BCE minus its entropy floor (0 = perfect)
        ("klLoss", "wm", "kl_dyn"),
        ("rewardLoss", "wm", "reward_loss"),
        ("memoryExact", "wm", "memory_exact"),                # (1) M(posterior z) reproduces all 144 bits
        ("memoryRecall", "wm", "memory_recall"),
        ("memoryFalsePos", "wm", "memory_false_pos"),
        ("memoryExactSet", "wm", "memory_exact_set"),         # exact match on frames with a set bit
        ("priorEventRecall", "wm", "prior_event_recall"),     # (2) M(prior z) catches the bits that turn on
        ("priorFalseEvent", "wm", "prior_false_event"),
        ("eventGatePred", "wm", "event_gate_pred"),           # mean P(event gate) on the batch's transitions (prior's
        ("eventGateReal", "wm", "event_gate_real"),           #   one-step samples, as in dreams) and their real event
                                                              #   rate: calibration (both inflated by the event windows)
        ("eventGateRecall", "wm", "event_gate_recall"),       # real events with P(gate) > 0.5
        ("eventGateFalse", "wm", "event_gate_false"),         # transitions without an event but P(gate) > 0.5
        ("sparseRewardPred", "wm", "sparse_reward_pred"),     # sparse reward head on event transitions (~50)
        ("dreamEvents", "dream", "dream_events"),             # expected events per dream (P(gate) x alive, <= 1)
        ("dreamMemoryForget", "dream", "dream_memory_forget"),
        ("actorLoss", "dream", "actor_loss"),
        ("criticLoss", "dream", "critic_loss"),
        ("entropies", "dream", "entropies"),
        ("priorKl", "dream", "prior_kl"),                     # pmpo_sign: KL(pi || slow EMA actor), the trust region
        ("extPosFraction", "dream", "ext_pos_fraction"),      # share of dream steps with A_ext >= 0 (PMPO accept set)
        ("advantages", "dream", "advantages"),
        ("criticValues", "dream", "critic_values"),
        ("extCriticEV", "dream", "ext_critic_ev"), ("curCriticEV", "dream", "cur_critic_ev"),   # critics learning?
        ("extAdvShare", "dream", "ext_adv_share"), ("curAdvShare", "dream", "cur_adv_share"),   # who drives the actor
        ("bestBadges", "play", "best_badges"),                 # plot shading (most badges in an episode of the round)
    ]                                                          # the dream audit has its own CSV (audit_filename)
    GRAD_SHARE_COLUMNS = [(f"g{'Tssm' if g == 'tssm' else 'Enc'}Share{''.join(w.title() for w in t.split('_'))}",
                           "wm", f"gshare_{g}_{t}")
                          for g in GQDreamer.GRAD_SHARE_GROUPS for t in GQDreamer.WM_LOSS_TERMS]
    EXTRA_COLUMNS = [                                          # written only with csv_extra_columns = True
        ("totalCuriosity", "play", "total_curiosity"),
        ("bestPartyLevel", "play", "best_party_level"), ("meanPartyLevel", "play", "mean_party_level"),
        ("bestMaps", "play", "best_maps"), ("knownMaps", "play", "known_maps"),
        ("reconSpriteLoss", "wm", "recon_sprite_excess"), ("rolloutLoss", "wm", "rollout_loss"),
        ("encoderDrift", "wm", "encoder_drift"), ("curiosityAdvantages", "dream", "curiosity_advantages"),
        ("replayReturns", "dream", "replay_returns"),
        ("secPerUpdate", "time", "per_update"), ("playSeconds", "play", "t_play"), ("trainSeconds", "time", "train"),
        ("reconBranchLoss", "wm", "recon_loss"), ("maskRatio", "wm", "mask_ratio"),
        ("postEntropy", "wm", "post_entropy"), ("priorEntropy", "wm", "prior_entropy"),
        ("latentUsage", "wm", "latent_usage"), ("predMatch", "wm", "pred_match"), ("copyMatch", "wm", "copy_match"),
        ("rolloutMatchK", "wm", "rollout_match_k"), ("rolloutKlK", "wm", "rollout_kl_k"),
        ("mapAcc", "wm", "map_acc"), ("mapAccNoToken", "wm", "map_acc_notoken"), ("battleAcc", "wm", "battle_acc"),
        ("partyHpMae", "wm", "party_hp_mae"), ("partyLevelMae", "wm", "party_level_mae"), ("menuAcc", "wm", "menu_acc"),
        ("groundingLoss", "wm", "grounding_loss"), ("memoryLoss", "wm", "memory_loss"),
        ("memoryPriorLoss", "wm", "memory_prior_loss"), ("priorMemoryExact", "wm", "prior_memory_exact"),
        ("priorEventsInBatch", "wm", "prior_events"),
        ("dreamMapAcc", "wm", "dream_map_acc"), ("dreamMemoryExact", "wm", "dream_memory_exact"),
        ("dreamMemoryRecall", "wm", "dream_memory_recall"), ("dreamEventFraction", "dream", "dream_event_fraction"),
        ("dreamEventReward", "dream", "dream_event_reward"), ("eventGatePosP", "wm", "event_gate_pos_p"),
        ("eventsInBatch", "wm", "events_in_batch"),
        ("eventGatePredReal", "wm", "event_gate_pred_real"), ("eventGateRecallReal", "wm", "event_gate_recall_real"),
        ("eventGateFalseReal", "wm", "event_gate_false_real"), ("eventGatePosPReal", "wm", "event_gate_pos_p_real"),
        ("eventGateLoss", "wm", "event_gate_loss"), ("sparseRewardLoss", "wm", "sparse_reward_loss"),
        ("gridLoss", "wm", "grid_loss"), ("gridAcc", "wm", "grid_acc"), ("gridCenterAcc", "wm", "grid_center_acc"),
        ("curiosityLoss", "wm", "curiosity_loss"),
        ("advScale", "dream", "adv_scale"), ("curAdvScale", "dream", "cur_adv_scale"),
        ("dreamStartBattle", "dream", "dream_start_battle"), ("advAbsMean", "dream", "adv_abs_mean"),
        ("branchAgree", "dream", "branch_agree"), ("branchSpread", "dream", "branch_spread"),
        ("battleMenuStates", "wm", "battle_menu_states"), ("harmonizedLoss", "wm", "harmonized_loss"),
        ("wmGradNorm", "wm", "wm_grad_norm"), ("wmClipThreshold", "wm", "wm_clip_threshold"),
        ("wmClipFraction", "wm", "wm_clip_fraction"), ("paddedFraction", "wm", "padded_fraction"),
        ("curiosityCriticLoss", "dream", "curiosity_critic_loss"),
        ("targetCriticValues", "dream", "target_critic_values"),
        ("curiosityCriticValues", "dream", "curiosity_critic_values"),
        ("advantagePosFraction", "dream", "advantage_pos_fraction"),
        ("curiosityFlipFraction", "dream", "curiosity_flip_fraction"),
        ("dreamMeanCuriosity", "dream", "dream_mean_curiosity"),
        ("replayValueLoss", "dream", "replay_value_loss"), ("replayCriticValues", "dream", "replay_critic_values"),
        ("bestCuriosity", "play", "best_curiosity"), ("batchWaitSeconds", "time", "wait"),
        ("gpuPeakGB", "mem", "gpu_peak_gb"), ("gpuReservedGB", "mem", "gpu_reserved_gb"),
        ("gpuFreeGB", "mem", "gpu_free_gb"),
        ("advCombinedStd", "dream", "adv_combined_std"), ("curPosFraction", "dream", "cur_pos_fraction"),
        ("extAdvStd", "dream", "ext_adv_std"), ("curAdvStd", "dream", "cur_adv_std"),
        ("extReturnRange", "dream", "ext_return_range"), ("curReturnRange", "dream", "cur_return_range"),
        ("auditSnapshots", "play", "audit_snapshots"), ("auditPreEventSnapshots", "play", "audit_pre_event_snapshots"),
        ("gradShareUpdate", "wm", "grad_share_update"),
    ] + GRAD_SHARE_COLUMNS
    BASE_HEADERS = ["envSteps", "gradientSteps", "totalReward"]

    def __init__(self, envs, device=None, config="pokemon", visualize_dreams=False, verbose=True, **overrides):
        """config: a name of MODEL_CONFIGS ("pokemon"; "light" is an alias); overrides: GQDreamerConfig
        fields changed on top of it."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.envs = envs
        self.visualize_dreams = visualize_dreams
        sfx = "" if config == "pokemon" else f"_{config}"

        cfg = model_config(config, action_dim=envs.action_space.n,
                           frames_per_step=getattr(envs, "frames_per_step", FRAMES_PER_STEP), **overrides)

        # run control and outputs: set in GQDreamerConfig (config.py), overridable via run(...) / Trainer(...)
        for name in ("total_rounds", "updates_per_round", "seed", "checkpoint_interval",
                     "max_checkpoints", "num_best_dreams", "num_random_dreams", "csv_extra_columns",
                     "reward_plot_interval", "reward_ma_window", "exploration_map_interval"):
            setattr(self, name, getattr(cfg, name))
        self.csv_filename = f"metrics/training_data{sfx}.csv"
        self.audit_filename = f"diagnostics/dream_audit{sfx}.csv"   # dream audit rows (only rounds that ran one)
        self.reward_plot_filename = f"plots/rewards{sfx}.png"
        self.reward_compare_filename = f"plots/reward_compare{sfx}.png"        # over env steps
        self.dreams_dir = "dreams"
        self.reward_compare_labels = {}                # comparison plot: legend names of other metrics/*.csv files

        seed_everything(self.seed)
        self.agent = GQDreamer(cfg, device=self.device, envs=envs, verbose=verbose)
        self._pdf_pool = None                          # dream PDF worker process (async_dream_pdf)
        self._pdf_async_ok = True
        self._exploration_maps = ExplorationMapJob(latest=f"plots/exploration_map{sfx}.png")

    @property
    def columns(self):
        """Logged (header, source, key) columns."""
        return self.COLUMNS + (self.EXTRA_COLUMNS if self.csv_extra_columns else [])

    @property
    def headers(self):
        """CSV header row."""
        return self.BASE_HEADERS + [c[0] for c in self.columns]

    # --- Training ---------------------------------------------------------------------------------------------------
    def train(self, reset_actor_critic=False):
        """Main loop; reset_actor_critic keeps world model and buffer but restarts actor and critics."""
        agent = self.agent
        agent.load_checkpoint(reset_actor_critic=reset_actor_critic)
        self._prepare_csv()
        self._update_plots()

        try:
            if len(agent.buffer) == 0:                                 # nothing to train on yet: collect first
                print(f"{_DIM}    [setup] replay buffer empty -- every env plays one episode before the first "
                      f"training round{_RST}")
                agent.play(episodes_per_env=1, snapshots=self._audit_due(0))
            for rnd in range(self.total_rounds):
                self._train_and_play_round(rnd)
        finally:
            agent.wait_for_save()
            if self._pdf_pool is not None:
                self._pdf_pool.shutdown(wait=True)
                self._pdf_pool = None
            self._report_exploration_map()
            if self._exploration_maps.busy:
                print(f"{_DIM}    [exploration map] G{self._exploration_maps.update} is still being drawn by its "
                      f"worker process; it will appear in diagnostics/images/{_RST}")
            self._exploration_maps.close(wait=False)

    def _train_and_play_round(self, rnd):
        """One round: updates, dream PDF, play, logging, plots and checkpoint."""
        agent = self.agent
        self._print_round_header(rnd)
        # 1. train: updates_per_round updates on the buffer (the envs are idle); every audit_every rounds the dream
        #    audit (snapshots of the last play: dreams vs the real game)
        wm_metrics, dream_metrics, timing, best_dreams, random_dreams = self._train_round()
        audit = self._audit(rnd) if self._audit_due(rnd) else {}

        if self.visualize_dreams and best_dreams:
            best_dreams.sort(key=lambda x: x["metric"], reverse=True)
            vis = best_dreams[:self.num_best_dreams] + random.sample(
                random_dreams, min(self.num_random_dreams, len(random_dreams)))
            self._check_pdf_worker()
            path = save_dreams_pdf(agent, vis, self.dreams_dir, executor=self._dream_pdf_pool())
            how = " (drawn by a worker process)" if self._pdf_pool is not None else ""
            print(f"{_DIM}    [dreams] saved {len(vis)} dreams to {path}{how}{_RST}")

        # 2. play: every env plays one whole episode with the trained policy; it goes to the buffer when it ends
        avg_score, avg_curiosity = agent.play(episodes_per_env=1, snapshots=self._audit_due(rnd + 1))
        play = dict(agent.last_play_stats, total_curiosity=avg_curiosity)
        mem = self._gpu_memory()
        self._print_summary(avg_score, avg_curiosity, wm_metrics, dream_metrics, play, timing, mem, audit)
        self._append_csv_row(avg_score, {"wm": wm_metrics, "dream": dream_metrics, "play": play, "time": timing,
                                         "mem": mem})
        if audit:
            self._append_audit_row(audit)
        self._report_exploration_map()
        if self.exploration_map_interval > 0 and (rnd + 1) % self.exploration_map_interval == 0:
            self._start_exploration_map()

        if (rnd + 1) % self.reward_plot_interval == 0:
            self._update_plots()
        if (rnd + 1) % self.checkpoint_interval == 0:
            self._save_checkpoint(avg_score)

    def _audit_due(self, rnd):
        """True when round `rnd` (of this session) runs the dream audit (every audit_every rounds: 0, N, 2N, ...); the
        play before it takes the snapshots."""
        c = self.agent.cfg
        return audit_enabled(c) and rnd % int(c.audit_every) == 0

    def _audit(self, rnd):
        """Runs the dream audit after the round's updates (before its play) -> metrics; a failure is reported and
        training goes on."""
        try:
            m = self.agent.audit()
            if not m:
                print(f"{_DIM}    [audit] no usable snapshot from the last play: skipped{_RST}")
            return m
        except Exception as e:                                         # a diagnostic never stops training
            print(f"[!] Warning: dream audit failed in round {rnd + 1}: {type(e).__name__}: {e}")
            return {}

    def _train_round(self):
        """updates_per_round world-model + imagination updates -> (wm metrics, dream metrics, timing, dreams)."""
        agent = self.agent
        self._free_gpu_cache()                                         # drop cached blocks left by play / plots
        best_dreams, random_dreams = [], []
        wm_metrics, dream_metrics = {}, {}
        n_updates, t_wait = 0, 0.0
        t_train = time.perf_counter()
        report = max(int(agent.cfg.train_progress_every), 1)         # console line every n updates
        agent.start_prefetch()
        try:
            for step in range(self.updates_per_round):
                if step % report == 0:
                    el = time.perf_counter() - t_train
                    eta = f" | ~{el / step * (self.updates_per_round - step):.0f} s left" if step else ""
                    print(f"{_DIM}    [train] update {step} / {self.updates_per_round}{eta}{_RST}")
                tw = time.perf_counter()
                batch = agent.sample_batch()
                t_wait += time.perf_counter() - tw
                if batch is None:
                    print("[!] Not enough data in the buffer yet; collecting more.")
                    break
                compute_metrics = step == self.updates_per_round - 1
                m, d, best, rnd = self._update(batch, compute_metrics)   # a CUDA OOM raises with its own traceback
                batch = None                                           # free it before the next batch arrives
                if compute_metrics:
                    wm_metrics, dream_metrics = m, d
                if self.visualize_dreams:
                    best_dreams.append(best)
                    random_dreams.append(rnd)
                agent.total_num_updates += 1
                n_updates += 1
        finally:
            agent.stop_prefetch()                                      # the buffer changes while playing
            agent.flush_priorities()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        best_dreams, random_dreams = _dreams_to_cpu(best_dreams), _dreams_to_cpu(random_dreams)
        t_train = time.perf_counter() - t_train
        timing = {"train": t_train, "per_update": t_train / max(n_updates, 1), "wait": t_wait,
                  "phases": dict(agent.timer.times)}
        return wm_metrics, dream_metrics, timing, best_dreams, random_dreams

    def _update(self, batch, compute_metrics):
        """One world-model + imagination update; its large tensors are freed on return."""
        agent = self.agent
        states, kv_context, start_mask, m, start_info = agent.train_world_model(batch, compute_metrics=compute_metrics)
        # real rewards aligned with the states (state j = observation j + 1)
        real = {"rewards": (batch["sparse_rewards"] + batch["standard_rewards"])[:, 1:, 0],
                "valid": batch["valid"][:, 1:]}
        d, best, rnd = agent.imagine(states, kv_context, start_mask=start_mask, start_info=start_info, real=real,
                                     compute_metrics=compute_metrics, pack_dreams=self.visualize_dreams)
        return m, d, best, rnd

    # --- GPU memory --------------------------------------------------------------------------------------------------
    def _free_gpu_cache(self):
        """Returns unused cached GPU blocks to the driver (limits fragmentation between rounds)."""
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _gpu_memory(self):
        """GPU memory in GB: allocated / reserved now, peak of the round, free on the device (all processes)."""
        if self.device.type != "cuda":
            return {}
        free, total = torch.cuda.mem_get_info(self.device)
        gb = 1024 ** 3
        return {"gpu_allocated_gb": torch.cuda.memory_allocated(self.device) / gb,
                "gpu_reserved_gb": torch.cuda.memory_reserved(self.device) / gb,
                "gpu_peak_gb": torch.cuda.max_memory_allocated(self.device) / gb,
                "gpu_free_gb": free / gb, "gpu_total_gb": total / gb}

    def _save_checkpoint(self, avg_score):
        """Saves a checkpoint (+ buffer) and keeps only the newest max_checkpoints (async_save: written by a background
        thread while the next round goes on; play waits for it before its first replay-buffer write)."""
        agent = self.agent
        agent.buffer.print_diagnostics()
        ckpt_dir = agent.CHECKPOINT_DIR
        path = os.path.join(ckpt_dir, f"{agent.CHECKPOINT_PREFIX}_R{int(avg_score)}_G{agent.total_num_updates}.pt")

        def prune():
            ckpts = sorted(glob.glob(os.path.join(ckpt_dir, agent.CHECKPOINT_GLOB)), key=os.path.getmtime)
            while len(ckpts) > self.max_checkpoints:
                oldest = ckpts.pop(0)
                try:
                    os.remove(oldest)
                except OSError as e:
                    print(f"[!] Warning: could not remove old checkpoint {oldest}: {e}")
        if agent.cfg.async_save:
            agent.save_checkpoint(path, background=True, after=prune)
            print(f"{_DIM}    [ckpt] writing {path} (+ replay buffer) in the background{_RST}")
        else:
            agent.save_checkpoint(path)
            prune()

    def _dream_pdf_pool(self):
        """The dream PDF worker process (async_dream_pdf), or None to draw in this process."""
        if not (self.agent.cfg.async_dream_pdf and self._pdf_async_ok):
            return None
        if self._pdf_pool is None:
            self._pdf_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        return self._pdf_pool

    def _check_pdf_worker(self):
        """Reports a failed PDF of the previous round (and falls back to drawing in this process)."""
        fut = save_dreams_pdf.pending
        if fut is not None and fut.done() and fut.exception() is not None:
            print(f"[!] dream PDF worker failed ({fut.exception()!r}); dream PDFs are drawn in this process from now on")
            self._pdf_pool.shutdown(wait=False)
            self._pdf_pool = None
            self._pdf_async_ok = False
        save_dreams_pdf.pending = None if fut is None or fut.done() else fut

    # --- Metrics CSV and plots --------------------------------------------------------------------------------------
    def _prepare_csv(self):
        """Creates the CSV, archives it for a fresh model, or resumes the step counters from it."""
        agent, path = self.agent, self.csv_filename
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path) and not agent._resumed:                # fresh model: archive the existing CSV
            with open(path, "r") as f:
                has_rows = sum(1 for line in f if line.strip()) > 1
            if has_rows:
                stem, ext = os.path.splitext(path)
                archived = f"{stem}_before_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
                os.replace(path, archived)
                print(f"{_DIM}    [setup] new model: previous CSV moved to {archived}{_RST}")
        apath = self.audit_filename
        if os.path.exists(apath) and not agent._resumed:
            with open(apath, "r") as f:
                has_rows = sum(1 for line in f if line.strip()) > 1
            if has_rows:
                stem, ext = os.path.splitext(apath)
                archived = f"{stem}_before_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
                os.replace(apath, archived)
                print(f"{_DIM}    [setup] new model: previous audit CSV moved to {archived}{_RST}")
        if not os.path.exists(path):
            with open(path, mode="w", newline="") as f:
                csv.writer(f).writerow(self.headers)
            return
        self._upgrade_csv_header()
        try:
            with open(path, "r") as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            if len(lines) > 1:
                last = lines[-1].split(",")
                agent.total_num_steps = int(float(last[0]))
                agent.total_num_updates = int(float(last[1]))
                print(f"{_DIM}    [setup] resuming at {agent.total_num_steps:,} env steps | "
                      f"{agent.total_num_updates:,} grad steps{_RST}")
        except Exception as e:
            print(f"[!] Warning: Could not recover steps from CSV: {e}")

    def _upgrade_csv_header(self):
        """Rewrites a CSV with other columns to the current header (matched by name; original kept as *.bak.csv)."""
        with open(self.csv_filename, "r", newline="") as f:
            rows = list(csv.DictReader(f))
            old = list(rows[0].keys()) if rows else None
        headers = self.headers
        if old is None or old == headers:
            return
        self._move_audit_columns(rows, old)
        backup = os.path.splitext(self.csv_filename)[0] + ".bak.csv"
        if not os.path.exists(backup):
            shutil.copy2(self.csv_filename, backup)
        with open(self.csv_filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"[*] Rewrote {self.csv_filename} to the current {len(headers)} columns (original kept as {backup}).")

    @property
    def audit_headers(self):
        """Header row of the dream audit CSV."""
        return self.BASE_HEADERS[:2] + [c[0] for c in AUDIT_COLUMNS]

    def _upgrade_audit_header(self):
        """Rewrites an audit CSV with other columns to the current header (matched by name; original kept as
        *.bak.csv), so a resumed run never appends rows under the wrong columns."""
        path = self.audit_filename
        if not os.path.exists(path):
            return
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
            old = list(rows[0].keys()) if rows else None
        headers = self.audit_headers
        if old is None or old == headers:
            return
        backup = os.path.splitext(path)[0] + ".bak.csv"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"[*] Rewrote {path} to the current {len(headers)} audit columns (original kept as {backup}).")

    def _append_audit_row(self, audit):
        """Appends one dream audit to its own CSV (created with its header when missing)."""
        path, agent = self.audit_filename, self.agent
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not getattr(self, "_audit_header_checked", False):
            self._upgrade_audit_header()
            self._audit_header_checked = True
        new = not os.path.exists(path)
        row = [agent.total_num_steps, agent.total_num_updates] + [audit.get(key, "") for _, _, key in AUDIT_COLUMNS]
        row = [round(x, 4) if isinstance(x, float) else x for x in row]
        with open(path, mode="a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self.audit_headers)
            w.writerow(row)

    def _move_audit_columns(self, rows, old_headers):
        """Moves audit columns found in the metrics CSV into the audit CSV (skipped when the audit CSV exists)."""
        names = [c[0] for c in AUDIT_COLUMNS]
        if os.path.exists(self.audit_filename) or not any(n in old_headers for n in names):
            return
        keep = [r for r in rows if any((r.get(n) or "").strip() for n in names)]
        if not keep:
            return
        os.makedirs(os.path.dirname(self.audit_filename) or ".", exist_ok=True)
        with open(self.audit_filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.audit_headers, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(keep)
        print(f"[*] Moved the audit columns of {len(keep)} rounds to {self.audit_filename}.")

    def _append_csv_row(self, avg_score, metrics):
        """Appends one round of metrics to the CSV."""
        agent = self.agent
        row = [agent.total_num_steps, agent.total_num_updates, avg_score]
        row += [metrics[src].get(key, "") for _, src, key in self.columns]
        row = [round(x, 4) if isinstance(x, float) else x for x in row]
        with open(self.csv_filename, mode="a", newline="") as f:
            csv.writer(f).writerow(row)

    def _update_plots(self):
        """Refreshes the reward PNGs; plotting failures never stop training. The
        comparison draws every other run's CSV in metrics/ (find_comparison_runs), re-listed at each refresh."""
        try:
            runs = find_comparison_runs(self.csv_filename, self.reward_compare_labels)
        except Exception as e:
            runs = []
            print(f"[!] Warning: could not list the earlier runs' CSVs: {e}")
        for name, fn in (("reward graph", lambda: plot_training(self.csv_filename, self.reward_plot_filename,
                                                                self.reward_ma_window)),
                         ("comparison graph (env steps)", lambda: plot_reward_comparison(
                             self.csv_filename, runs, self.reward_compare_filename, self.reward_ma_window))):
            try:
                fn()
            except Exception as e:
                print(f"[!] Warning: could not update {name}: {e}")

    # --- Exploration maps -------------------------------------------------------------------------------------------
    def _start_exploration_map(self):
        """Copies what the map needs from the replay buffer and draws it in a worker process (exploration_maps.py);
        never stops training."""
        job, g = self._exploration_maps, self.agent.total_num_updates
        try:
            if job.submit(self.agent.buffer, g):
                print(f"{_DIM}    [exploration map] drawing the replay buffer at G{g} in a worker process "
                      f"({job.summary}) -> {job.out_dir.name}/exploration_map_G{g}.png{_RST}")
            else:
                print(f"{_DIM}    [exploration map] G{job.update} is still being drawn: skipping G{g}{_RST}")
        except Exception as e:
            print(f"[!] Warning: could not start the exploration map: {e!r}")

    def _report_exploration_map(self):
        """Prints where a map finished since the last call went (or why it failed)."""
        res = self._exploration_maps.poll()
        if res is None:
            return
        path, text = res
        if path is None:
            print(f"[!] Warning: exploration map G{self._exploration_maps.update} failed: {text}")
        else:
            print(f"{_DIM}    [exploration map] -> {path} (+ {self._exploration_maps.latest}){_RST}")

    # --- Console output ---------------------------------------------------------------------------------------------
    def _print_round_header(self, rnd):
        """Banner with the round number and step counters."""
        bar = "═" * 62
        print(f"\n{_B}{_CYA}{bar}{_RST}")
        print(f"{_B}{_CYA}  ROUND {rnd + 1} / {self.total_rounds}{_RST}{_DIM}   env steps "
              f"{self.agent.total_num_steps:,} | grad steps {self.agent.total_num_updates:,}{_RST}")
        print(f"{_B}{_CYA}{bar}{_RST}")

    def _print_summary(self, avg_score, avg_curiosity, wm, dream, play, timing=None, mem=None, audit=None):
        """Round summary: reward, curiosity and progress, then only the world model KL, the policy entropy and
        the speed (dimmed); every other diagnostic goes to the CSV only. The dream audit block follows in the rounds that run it."""
        print()
        print(f"  {_B}{_GRN}★ REWARD     {avg_score:8.2f}{_RST}{_GRN}   best {play.get('best_reward', 0):7.2f}{_RST}"
              f"      {_B}{_MAG}◆ CURIOSITY  {avg_curiosity:8.2f}{_RST}{_MAG}   best "
              f"{play.get('best_curiosity', 0):7.2f}{_RST}")
        print(f"  {_B}{_YEL}➜ PROGRESS   best episode: {play.get('best_events', 0)} events · "
              f"{play.get('best_badges', 0)} badges · party level {play.get('best_party_level', 0)} · "
              f"{play.get('best_maps', 0)} maps{_RST}{_YEL}   (mean {play.get('mean_events', 0):.1f} events, "
              f"{play.get('mean_maps', 0):.1f} maps){_RST}{_DIM}      avg episode "
              f"{play.get('episode_length', 0):.0f} steps · {play.get('known_tiles', 0):,} squares known · "
              f"blackouts "
              f"{play.get('blackout_rate', 0):.2f} / 1k steps"
              f"{_RST}")

        def row(label, text):
            print(f"{_DIM}    {label:<12}│ {text}{_RST}")

        def f(v, d=3):
            return "–" if v is None or v != v else f"{v:.{d}f}"

        print()
        if wm:                                                         # world model KL: nats and its loss terms
            kl_terms = " + ".join(f"{t} {wm[f'wm_{t}']:.3g}" for t in ("kl_dyn", "kl_rep") if f"wm_{t}" in wm)
            row("KL", f"{f(wm.get('kl_dyn'), 2)} nats" + (f" · loss {kl_terms}" if kl_terms else ""))
        if dream:
            row("actor", f"entropy {f(dream.get('entropies'))}"
                         + (f" · KL to the EMA actor {f(dream.get('prior_kl'), 4)}" if dream.get("prior_kl") else ""))
        if timing:
            ph = timing.get("phases", {})
            enc = (f" (last update: posteriors {ph['encode']:.2f} s, decoder {ph.get('decode', 0.0):.2f} s, "
                   f"backward {ph.get('WM backward + step', 0.0):.2f} s, dreams "
                   f"{ph.get('dream + actor/critic', 0.0):.2f} s)") if "encode" in ph else ""
            row("speed", f"{timing['per_update']:.2f} s/update{enc} · training {timing['train']:.0f} s · play "
                         f"{play.get('t_play', 0):.0f} s (model {play.get('t_model', 0):.0f} s, emulators "
                         f"{play.get('t_env', 0):.0f} s)")
        for line in audit_lines(audit or {}):
            print(f"{_DIM}    {line}{_RST}")

    # --- Evaluation -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, num_episodes=1, recording=False, greedy=False, checkpoint=None, max_steps=None):
        """Every env plays `num_episodes` with the policy of `checkpoint` (None: the newest one in checkpoints/);
        recording=True writes one MP4 per env; max_steps ends an episode early."""
        agent = self.agent
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"no checkpoint at {checkpoint}")
            if os.path.getsize(checkpoint) < 1_000_000:
                raise RuntimeError(f"{checkpoint} is a Git LFS pointer, not the model: run `git lfs pull` first")
        agent.load_checkpoint(path=checkpoint, load_buffer=False)
        print(f"Starting Parallel Evaluation for {num_episodes} episodes per agent...")
        num_envs = len(self.envs)

        writers, frame_size = [], (480, 432)
        if recording:
            import cv2
            os.makedirs("recordings", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writers = [cv2.VideoWriter(os.path.join("recordings", f"eval_agent{i + 1}.mp4"), fourcc, 10, frame_size)
                       for i in range(num_envs)]
            print(f"{_DIM}    [rec] recording {num_envs} agents -> recordings/eval_agent1..{num_envs}.mp4{_RST}")

        def record(i):
            """Writes the current full-resolution frame of env i to its video."""
            if writers:
                import cv2
                frame = cv2.resize(self.envs.rgb_frame(i), frame_size, interpolation=cv2.INTER_NEAREST)
                writers[i].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        done_eps = [0] * num_envs
        rewards = [0.0] * num_envs
        steps = [0] * num_envs
        ctx = agent.new_act_context(num_envs)
        observations = [obs for obs, _ in self.envs.reset()]
        for i in range(num_envs):
            record(i)

        while min(done_eps) < num_episodes:
            action_idxs = torch.argmax(agent.act(observations, ctx, greedy=greedy), dim=-1).cpu().numpy()
            active = [done_eps[i] < num_episodes for i in range(num_envs)]
            results = self.envs.step(action_idxs, active=active)
            to_reset = []
            for i in range(num_envs):
                if not active[i]:
                    continue
                obs, reward, terminated, truncated, info = results[i]
                observations[i] = obs
                record(i)
                rewards[i] += reward
                steps[i] += 1
                for msg in info.get("logs", ()):
                    print(f"[Agent {i+1}] {msg}")
                if steps[i] % 500 == 0:
                    print(f"[Agent {i+1}] Eval Step: {steps[i]}/{info['limit']} | Reward: {rewards[i]:.3f} | "
                          f"Events: {info['events']} | Maps: {info['maps']}")
                if terminated or truncated or (max_steps and steps[i] >= max_steps):
                    print(f"--- [Agent {i+1}] Episode Finished! Reward: {rewards[i]:.3f} | Steps: {steps[i]} | "
                          f"Events: {info['events']} | Blackouts: {info['blackouts']} ---")
                    done_eps[i] += 1
                    if done_eps[i] < num_episodes:
                        to_reset.append(i)
                        rewards[i], steps[i] = 0.0, 0
                        agent.reset_act_rows(ctx, i)
            for i in to_reset:
                observations[i], _ = self.envs.reset_one(i)
                record(i)

        for i, w in enumerate(writers):
            w.release()
            print(f"{_DIM}    [rec] saved recordings/eval_agent{i + 1}.mp4{_RST}")


# --- Notebook entry point -------------------------------------------------------------------------------------------
def run(mode="train", num_envs=10, verbose=False, recording=False, visualize_dreams=True, config="pokemon",
        rom_path="game/PokemonRed.gb", num_eval_envs=1, num_eval_episodes=1, reset_actor_critic=False,
        device=None, checkpoint=None, window=True, max_steps=None, **overrides):
    """Creates the envs, trains or evaluates GQ-Dreamer and closes the envs -> the Trainer.

    mode: "train" (num_envs headless emulators) or "eval" (num_eval_envs SDL2 windows, num_eval_episodes each);
    config: "pokemon" (TSSM 512; "light" is the same); verbose: per-env agent logs; recording: eval writes one MP4
    per env to recordings/; visualize_dreams: dream PDFs each training round in dreams/; reset_actor_critic: keep
    world model + buffer, restart actor / critics; checkpoint: eval only, the model file to play (None: the newest
    in checkpoints/); window: eval in SDL2 windows (False: headless); max_steps: eval episodes end after this many
    steps; overrides: GQDreamerConfig fields (everything else, the episode
    length included, comes from config.py). Ctrl+C / "interrupt kernel" stops cleanly and still returns the
    Trainer."""
    if mode not in ("train", "eval"):
        raise ValueError(f"invalid mode {mode!r}: use 'train' or 'eval'")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    if mode == "eval":
        n, speed = num_eval_envs, int(window)
        window = "SDL2" if window else "null"
        overrides.setdefault("buffer_size", 1000)                      # eval needs no 6 GB replay buffer
    else:
        n, window, speed = num_envs, "null", 0
    cfg = model_config(config, **overrides)                            # episode length: config.py
    envs = create_envs(num_envs=n, rom_path=rom_path, state_dir="game", wind=window, speed=speed,
                       episode_steps=cfg.episode_steps, episode_extension=cfg.episode_extension,
                       cross_episode_novelty=cfg.cross_episode_novelty, novelty_min_rarity=cfg.novelty_min_rarity,
                       novelty_zero_visits=cfg.novelty_zero_visits, auto_respawn=cfg.auto_respawn)
    trainer = None
    try:
        print("Initializing GQ-Dreamer...")
        trainer = Trainer(envs, device=device, config=config, visualize_dreams=visualize_dreams, verbose=verbose,
                          **overrides)
        if mode == "train":
            print("Starting Training Mode...")
            trainer.train(reset_actor_critic=reset_actor_critic)
        else:
            print("Starting Evaluation Mode...")
            trainer.evaluate(num_episodes=num_eval_episodes, recording=recording, checkpoint=checkpoint,
                             max_steps=max_steps)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by the user.")
    finally:
        envs.close()
        print("All environments safely closed.")
    return trainer
