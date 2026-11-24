"""Checkpointing of GQ-Dreamer: save, find and partially load checkpoints (every part whose names and shapes match)."""
import glob
import os
import threading
import time

import torch

from .game import GRID_DIM, GRID_INPUT_DIM


def _cpu_copy(obj):
    """Nested dicts / lists of tensors -> the same structure with CPU copies of the tensors."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):                                          # keeps OrderedDict and state_dict metadata
        out = type(obj)((k, _cpu_copy(v)) for k, v in obj.items())
        if hasattr(obj, "_metadata"):
            out._metadata = obj._metadata
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_copy(v) for v in obj)
    return obj


def _short_list(keys, n=4):
    """First n keys, then a count of the rest."""
    keys = list(keys)
    return ", ".join(keys[:n]) + (f", ... (+{len(keys) - n})" if len(keys) > n else "")


class CheckpointMixin:
    """Save / load of every GQDreamer module, optimizer and run counter."""
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_FAMILY = "gq_pokemon"
    ARCH_VERSION = "pk4"           # architecture id in checkpoint and buffer file names
    MODULES = ("tokenizer", "tssm", "prior", "standard_reward_head", "event_gate_head", "sparse_reward_head",
               "curiosity_head", "grid_head", "actor", "actor_prior", "critic", "curiosity_critic", "critic_target",
               "curiosity_critic_target", "harmony")
    OPTIMIZERS = ("wm_optimizer", "actor_optimizer", "critic_optimizer", "curiosity_critic_optimizer",
                  "harmony_optimizer")
    OPTIMIZER_MODULES = {                                              # parameter order of each optimizer
        "wm_optimizer": ("tokenizer", "tssm", "prior", "standard_reward_head", "event_gate_head", "sparse_reward_head",
                         "curiosity_head", "grid_head"),
        "actor_optimizer": ("actor",),
        "critic_optimizer": ("critic",),
        "curiosity_critic_optimizer": ("curiosity_critic",),
        "harmony_optimizer": ("harmony",),
    }
    ACTOR_CRITIC_PARTS = ("actor", "actor_prior", "critic", "curiosity_critic", "critic_target",
                          "curiosity_critic_target", "actor_optimizer", "critic_optimizer", "curiosity_critic_optimizer")
    ARCH_KEYS = ("arch_version", "action_dim", "n_latents", "classes", "enc_patch_squares", "enc_dim", "enc_depth",
                 "enc_heads", "enc_pos_inject", "dec_chans", "head_hidden", "memory_hidden", "tssm_dim", "tssm_layers",
                 "tssm_heads", "tssm_kv_heads", "tssm_ffn", "prior_layers", "mlp_dim", "attn_gate")
    COMPAT_KEYS = ("frames_per_step", "config_name")                   # a mismatch blocks resuming

    # --- Names ------------------------------------------------------------------------------------------------------
    def _set_file_names(self):
        """Checkpoint / buffer file names of this configuration and control rate. The buffer is shared by every
        configuration (its format does not depend on the model), so "light" can resume on the data of "pokemon"."""
        tag = "" if self.cfg.config_name == "pokemon" else f"_{self.cfg.config_name}"
        self.BUFFER_NAME = f"replay_buffer_{self.CHECKPOINT_FAMILY}_{self.ARCH_VERSION}.buffer"
        self.CHECKPOINT_PREFIX = f"{self.CHECKPOINT_FAMILY}_{self.ARCH_VERSION}{tag}_f{self.cfg.frames_per_step}"
        self.CHECKPOINT_GLOB = f"{self.CHECKPOINT_PREFIX}_R*_G*.pt"

    def optimizer_named_params(self, name):
        """[(qualified name, parameter)] updated by optimizer `name`."""
        return [(f"{m}.{n}", p) for m in self.OPTIMIZER_MODULES[name] for n, p in getattr(self, m).named_parameters()]

    def optimizer_params(self, name):
        """Parameters updated by optimizer `name`."""
        return [p for _, p in self.optimizer_named_params(name)]

    def find_checkpoint(self):
        """Newest checkpoint (highest gradient step) of this configuration, or None."""
        files = glob.glob(os.path.join(self.CHECKPOINT_DIR, self.CHECKPOINT_GLOB))
        if not files:
            return None
        return max(files, key=lambda x: int(x.split("_G")[-1].split(".pt")[0]))

    # --- Save -------------------------------------------------------------------------------------------------------
    def save_checkpoint(self, path, save_buffer=True, background=False, after=None):
        """Writes modules, optimizers and counters to `path` (and the replay buffer next to it). background: the state
        is copied to CPU memory here and a thread writes the files (wait_for_save() joins it; play() does so before
        its first replay-buffer write); after: called once the files are written."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        ckpt = {name: getattr(self, name).state_dict() for name in self.MODULES + self.OPTIMIZERS}
        ckpt.update({
            "wm_clipper": self.wm_clipper.state_dict(),
            "adv_range": self.adv_range.tolist() if self.adv_range_init else None,   # "range" advantages: EMA scales
            "novelty": {"tiles": dict(self.novelty_tiles), "maps": dict(self.novelty_maps)},   # cross-episode counts
            "total_num_episodes": self.total_num_episodes,
            "total_num_steps": self.total_num_steps,
            "total_num_updates": self.total_num_updates,
            "config": self.config,
            "optimizer_param_names": {name: [n for n, _ in self.optimizer_named_params(name)]
                                      for name in self.OPTIMIZERS},
        })
        if not background:
            self._write_checkpoint(ckpt, path, save_buffer, after)
            return None
        self.wait_for_save()
        ckpt = _cpu_copy(ckpt)                                         # snapshot: training goes on meanwhile
        self._save_thread = threading.Thread(target=self._write_checkpoint, args=(ckpt, path, save_buffer, after),
                                             name="checkpoint-writer")
        self._save_thread.start()
        return self._save_thread

    def _write_checkpoint(self, ckpt, path, save_buffer, after):
        """Writes a checkpoint dict (+ the replay buffer); errors are reported, not raised."""
        try:
            torch.save(ckpt, path)
            print(f"\033[90m    [ckpt] saved {path}{' (+ replay buffer)' if save_buffer else ''}\033[0m")
            if save_buffer:
                self.buffer.save(os.path.join(os.path.dirname(path) or ".", self.BUFFER_NAME))
            if after is not None:
                after()
        except Exception as e:
            print(f"[!] Warning: checkpoint {path} not written completely: {type(e).__name__}: {e}")

    def wait_for_save(self):
        """Waits for the background checkpoint writer (before the replay buffer changes)."""
        th = getattr(self, "_save_thread", None)
        if th is not None:
            if th.is_alive():
                t = time.perf_counter()
                th.join()
                print(f"\033[90m    [ckpt] waited {time.perf_counter() - t:.1f}s for the checkpoint writer\033[0m")
            self._save_thread = None

    # --- Load -------------------------------------------------------------------------------------------------------
    @staticmethod
    def _grid_resize(name, key, old, new):
        """True for the grid layers that can be loaded across a change of the grid channels (25 <-> 50), one
        dimension changed, nothing else."""
        if not ((name == "tokenizer" and ".grid_in.0." in f".{key}") or name == "grid_head"):
            return False
        diff = [i for i, (a, b) in enumerate(zip(old, new)) if a != b] if len(old) == len(new) else None
        sizes = {GRID_DIM, 2 * GRID_DIM}
        return diff is not None and len(diff) == 1 and old[diff[0]] in sizes and new[diff[0]] in sizes \
            and new[diff[0]] == GRID_INPUT_DIM

    @staticmethod
    def _load_module_partial(name, module, saved):
        """Loads every tensor whose name and shape match; the rest keeps its fresh init. One exception: the grid layers
        whose explored part is the first 25 entries of a grid dimension -- the grid token's first layer (inputs) and
        the grid head's output layer (outputs) -- keep their saved explored part when the grid changes between
        explored (25) and [explored, dwell] (50): dropped dwell inputs / outputs are cut off, added ones start at zero.
        -> True if all matched."""
        own = module.state_dict()
        fits = {k: v for k, v in saved.items() if k in own and own[k].shape == v.shape}
        grown = {}
        for k, v in saved.items():
            o = own.get(k)
            if o is None or not CheckpointMixin._grid_resize(name, k, tuple(v.shape), tuple(o.shape)):
                continue
            t = o.detach().clone()
            if t.dim() == 2 and t.shape[1] > v.shape[1]:
                t[:, v.shape[1]:] = 0                                  # added inputs: no effect until trained
            common = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, v.shape))
            t[common] = v[common].to(t.dtype)
            grown[k] = t
        reshaped = [k for k, v in saved.items() if k in own and own[k].shape != v.shape and k not in grown]
        new = [k for k in own if k not in saved]
        dropped = [k for k in saved if k not in own]
        module.load_state_dict({**fits, **grown}, strict=False)
        if not (reshaped or new or dropped or grown):
            return True
        msg = f"[load] '{name}': {len(fits) + len(grown)}/{len(own)} tensors restored"
        if grown:
            msg += f" | grid layer resized, explored part kept: {_short_list(list(grown))}"
        if reshaped:
            msg += f" | shape changed -> fresh: {_short_list(reshaped)}"
        if new:
            msg += f" | new -> fresh: {_short_list(new)}"
        if dropped:
            msg += f" | no longer in the model -> dropped: {_short_list(dropped)}"
        print(msg)
        return False

    def _load_optimizer_partial(self, name, ckpt):
        """Restores optimizer moments by parameter name (same shape); others start fresh."""
        opt = getattr(self, name)
        saved = ckpt[name]
        old_idx = [i for g in saved["param_groups"] for i in g["params"]]
        old_state = saved.get("state", {})
        names = ckpt.get("optimizer_param_names", {}).get(name)
        if names is None or len(names) != len(old_idx):
            print(f"[load] '{name}': no usable parameter names -> fresh optimizer state")
            return
        by_name = {n: old_state[i] for n, i in zip(names, old_idx) if i in old_state}
        current = self.optimizer_named_params(name)
        new_state = {}
        for idx, (n, p) in enumerate(current):
            st = by_name.get(n)
            if st is not None and all(tuple(t.shape) == tuple(p.shape)
                                      for t in st.values() if torch.is_tensor(t) and t.dim() > 0):
                new_state[idx] = st
        sd = opt.state_dict()
        sd["state"] = new_state
        opt.load_state_dict(sd)
        if old_state and len(new_state) < len(current):                 # (an optimizer never stepped has none)
            print(f"[load] '{name}': moments restored for {len(new_state)}/{len(current)} parameters "
                  f"(the others start fresh)")

    def load_checkpoint(self, path=None, reset_actor_critic=False, load_buffer=True):
        """Loads the newest (or given) checkpoint part by part; reset_actor_critic keeps only the world model."""
        if path is None:
            path = self.find_checkpoint()
            if path is None:
                print(f"\033[90m    [setup] no {self.CHECKPOINT_PREFIX} checkpoint in {self.CHECKPOINT_DIR}/ -- the "
                      f"model starts from scratch\033[0m")
                if load_buffer:
                    self._load_buffer(self.CHECKPOINT_DIR)
                return
        if not os.path.exists(path):
            print(f"\033[90m    [setup] no checkpoint found at {path} -- starting from scratch\033[0m")
            return
        print(f"\033[90m    [setup] loading checkpoint {path}\033[0m")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)                             # not via GPU
        old_cfg, new_cfg = ckpt.get("config", {}), self.config

        def diff(keys):
            return {k: (old_cfg.get(k), new_cfg.get(k)) for k in keys if old_cfg.get(k) != new_cfg.get(k)}

        def fmt(d):
            return ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in d.items())

        incompatible = diff(self.COMPAT_KEYS)
        if incompatible:
            print(f"[load] {os.path.basename(path)} was trained with {fmt(incompatible)}: not loaded, the model "
                  f"starts from scratch (the replay buffer is still reused if it matches).")
            del ckpt
            if load_buffer:
                self._load_buffer(os.path.dirname(path) or ".")
            return
        changed = diff(self.ARCH_KEYS)
        if changed:
            print(f"[load] architecture changed since {os.path.basename(path)} ({fmt(changed)}): loading every part "
                  f"that still fits, the rest starts fresh.")
        if reset_actor_critic:
            print("[load] RESET_ACTOR_CRITIC: actor, critics and their optimizers start fresh.")

        skip = set(self.ACTOR_CRITIC_PARTS) if reset_actor_critic else set()
        exact = True
        for name in self.MODULES:
            if name in skip:
                continue
            if name not in ckpt:
                print(f"[load] '{name}' not in checkpoint -> fresh init.")
                exact = False
                continue
            exact &= self._load_module_partial(name, getattr(self, name), ckpt[name])
        if reset_actor_critic:
            self.actor_prior.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.curiosity_critic_target.load_state_dict(self.curiosity_critic.state_dict())
        for name in self.OPTIMIZERS:
            if name in skip or name not in ckpt:
                continue
            try:
                self._load_optimizer_partial(name, ckpt)
            except Exception as e:                                     # optimizer state never blocks a resume
                print(f"[load] '{name}' could not be restored -> fresh optimizer state ({e}).")

        self.wm_clipper.load_state_dict(ckpt.get("wm_clipper", {}))
        nov = ckpt.get("novelty") or {}                                # cross-episode counts (may be absent)
        self.novelty_tiles = {tuple(k): int(v) for k, v in nov.get("tiles", {}).items()}
        self.novelty_maps = {int(k): int(v) for k, v in nov.get("maps", {}).items()}
        if self.novelty_tiles:
            print(f"\033[90m    [setup] cross-episode novelty: counts of {len(self.novelty_tiles):,} squares / "
                  f"{len(self.novelty_maps)} maps restored\033[0m")
        if ckpt.get("adv_range") is not None:                          # absent: set by the first update
            self.adv_range.copy_(torch.as_tensor(ckpt["adv_range"], dtype=torch.float32))
            self.adv_range_init = True
        self.total_num_episodes = ckpt.get("total_num_episodes", 0)
        self.total_num_steps = ckpt.get("total_num_steps", 0)
        self.total_num_updates = ckpt.get("total_num_updates", 0)
        self._resumed = True
        if exact:
            print("\033[90m    [setup] checkpoint restored completely\033[0m")
        del ckpt
        if load_buffer:
            self._load_buffer(os.path.dirname(path) or ".")

    def _load_buffer(self, directory):
        """Loads the replay buffer file of this configuration if it exists."""
        path = os.path.join(directory, self.BUFFER_NAME)
        if os.path.exists(path):
            self.buffer.load(path)
