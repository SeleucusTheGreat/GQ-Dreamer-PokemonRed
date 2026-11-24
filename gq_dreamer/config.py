"""GQ-Dreamer (Pokémon Red) hyperparameters: one dataclass whose defaults are the "pokemon" configuration (TSSM 512);
"light" is an alias of the same model."""
from dataclasses import asdict, dataclass, fields, replace

from .game import ACTIONS, FRAMES_PER_STEP, MEMORY_REWARD


@dataclass(frozen=True)
class GQDreamerConfig:
    """All model, loss, replay, behaviour and run-control settings of GQ-Dreamer for Pokémon Red."""
    config_name: str = "pokemon"                 # tags checkpoint / metric file names
    action_dim: int = len(ACTIONS)               # UP, DOWN, LEFT, RIGHT, A, B
    frames_per_step: int = FRAMES_PER_STEP       # control rate (a checkpoint of another rate is never resumed)

    # --- Tokenizer: fusion encoder -> z = n_latents one-hot categoricals x classes ----------------------------------
    n_latents: int = 32                          # 32 latent queries -> 32 categoricals (raise to 48 if the memory
    classes: int = 32                            # exact match saturates below ~100%)
    enc_patch_squares: int = 2                   # image tokens: k x k step squares of the 72x80 frame (2: 25 tokens)
    enc_dim: int = 256
    enc_depth: int = 6                           # encoder layers
    enc_heads: int = 8
    enc_pos_inject: bool = True                  # 2D sin-cos position of the image tokens re-added in every layer
    enc_grad_checkpoint: bool = True             # recompute encoder activations in the backward pass
    enc_last_queries: bool = True                # last encoder layer computes only the latent queries (same result)
    unimix: float = 0.01
    dec_chans: tuple = (128, 96, 64, 32)         # conv decoder: 9x10 -> 18x20 -> 36x40 -> 72x80
    head_hidden: int = 256                       # grounding heads (map, status, party, opponent, items, menu) on z
    memory_hidden: int = 512                     # memory head M on z

    # --- Reconstruction and grounding -------------------------------------------------------------------------------
    recon_frames: int = 2048                     # frames decoded per update (0 = every valid frame)
    mask_frames: int = 512                       # frames re-encoded with MAE masking + token dropout (decoded)
    mask_weight: float = 0.5                     # their share of the pixel / grounding / memory loss
    mask_max: float = 0.9
    sprite_weight: float = 5.0
    drop_grid: float = 0.1                       # side-token dropout per group (augmented frames only)
    drop_map: float = 0.2                        # map / battle flags / position: partly visible on the screen
    drop_status: float = 0.0                     # badges, facing, money: not on the screen
    drop_party: float = 0.0
    drop_enemy: float = 0.2                      # opponent: visible on the battle screen
    drop_items: float = 0.0
    drop_menu: float = 0.0                       # the cursor is not readable at 72x80
    drop_memory: float = 0.0                     # the event memory is never on the screen
    w_recon: float = 1.0
    w_map: float = 10.0
    w_status: float = 10.0
    w_party: float = 10.0
    w_enemy: float = 10.0
    w_items: float = 10.0
    w_menu: float = 10.0

    # --- Event memory: explicit m in s = [h, z, m] and in the TSSM input; memory head M on z ------------------------
    w_memory: float = 10.0                       # (1) M(posterior z_t) vs m_t: trains the encoder (z holds m) and M
    memory_pos_weight: float = 1.0               # (1) weight of set bits (> 1 inflates p: rule (3) needs p calibrated)
    w_memory_prior: float = 1.0                  # (2) M(prior sample of z_{t+1}) vs m_{t+1}, M frozen: trains prior
    memory_event_weight: float = 1.0             # (2) weight of bits that turn on at t+1 (> 1: more dream events)
    memory_threshold: float = 0.5                # (3) dreams: when the event gate fires, the bits of M(z_{t+1}) above
                                                 #     this that are not in m_t turn on (none: M's most likely unset bit)

    # --- Dream events: gate head x sparse reward head on [h_{t+1}, z_{t+1}, m_t] -------------------------------------
    # dream reward of a step = standard reward head + sparse reward head x P(gate): the gate head predicts "a memory bit
    # turns on in this transition" (Bernoulli, like a continue head), the sparse reward head the size of that reward
    # (trained on event transitions only). The gate's sample decides whether m changes; one event per dream at most.
    w_event_gate: float = 1.0                    # BCE of the gate head (on dream-like states: event_dreamlike)
    w_sparse_reward: float = 0.5                 # sparse reward head (event transitions only), as w_standard_reward
    event_dreamlike: float = 1.0                 # share of both event heads' loss on the prior's one-step samples
                                                 #   (what a dream sees): 1 = only there, so P(gate) learns how often a
                                                 #   sample merely looks like an event; on real latents the gate would
                                                 #   learn "z shows a new bit => event" and pay hallucinated +50s
    event_gate_pos_weight: float = 1.0           # weight of event transitions in the gate BCE (> 1 inflates P(gate))
    event_gate_init: float = -8.0                # gate head's initial logit (P ~ 3e-4: no dream event before training)

    # --- Dynamics: TSSM + prior head --------------------------------------------------------------------------------
    tssm_dim: int = 512
    tssm_layers: int = 6                         # TSSM blocks
    tssm_heads: int = 4
    tssm_kv_heads: int = 4
    tssm_ffn: int = 1365                         # SwiGLU hidden 8/3 x 512
    context_length: int = 48                     # attention window (steps)
    attn_gate: str = "elementwise"               # gated attention: "none", "headwise", "elementwise"
    prior_layers: int = 3

    # --- World-model losses -----------------------------------------------------------------------------------------
    w_dyn: float = 1.0                           # balanced KL with free nats (dyn : rep = 10 : 1)
    w_rep: float = 0.1
    free_nats: float = 1.0
    rollout_steps: int = 0                       # open-loop rollout on the model's own prior samples (0 = off)
    rollout_splits: int = 1
    rollout_decay: float = 0.9
    w_rollout: float = 1.0                       # (no effect with rollout_steps = 0)
    w_standard_reward: float = 0.5               # standard reward head (level-ups, heal); events: gate x sparse heads
    w_curiosity: float = 1.0                     # curiosity head (reads detached states: trains only itself)
    w_grid: float = 0.25                         # explored-grid head on s (its centre gates curiosity in dreams)
    grid_zero_weight: float = 5.0                # explored grid: unexplored cells up-weighted
    curiosity_grad: bool = False                 # curiosity head reads detached states
    w_dreamlike: float = 0.5                     # reward / curiosity / grid heads: share on rollout states
    harmony: bool = False                        # learned loss weights (HarmonyDream)
    harmony_lr: float = 1e-3

    # --- Optimisation -----------------------------------------------------------------------------------------------
    wm_lr: float = 2e-4
    clip_percentile: float = 99.0
    clip_window: int = 2000
    clip_warmup: int = 500
    clip_initial: float = 1000.0
    mixed_precision: str = "auto"                # bf16 on RTX 30xx+, fp16 + loss scaling on older GPUs

    # --- Batches and replay -----------------------------------------------------------------------------------------
    batch_size: int = 156                         # sequences per update
    seq_len: int = 64                            # steps per sequence
    buffer_size: int = 3000000                   # ~3.1 KB per transition (~6.2 GB)
    recent_sample_prob: float = 0.2              # windows from the last play round
    reward_sample_prob: float = 0.1              # windows around memory-bit events (before / after pairs): at most
    reward_sample_per_event: float = 0.01        #   min(reward_sample_prob, this x number of event rows in the buffer),
    reward_threshold: float = MEMORY_REWARD      # rows paying a memory bit anchor the event windows
    loss_sample_prob: float = 0.1                # windows centred on surprising frames
    loss_priority_alpha: float = 1.0

    # --- Imagination and behaviour ----------------------------------------------------------------------------------
    mlp_dim: int = 1024
    dream_horizon: int = 20
    dreams_per_sequence: int = 16
    dream_unimix: float = 0.0                    # dreams sample the raw prior
    dream_cuda_graph: bool = True
    dream_loss_frac: float = 0.1                # dreams starting at surprising states
    branch_groups: int = 4                       # start states that try every first action (battle menus if any)
    curiosity_scale: float = 1.0                 # curiosity's weight next to the reward stream (pmpo_sign: weight of the
                                                 #   curiosity stream's own PMPO loss; < 1: rewards have priority)
    continue_discount: float = 0.9985            # discount per step (death is not modelled: no continue head); horizon
                                                 #   ~667 steps
    actor_objective: str = "pmpo_sign"           # "pmpo_sign" (PMPO on the sign of each stream's advantage) or
                                                 #   "reinforce" (DreamerV3 REINFORCE on the combined advantage)
    pmpo_alpha: float = 0.5                      # pmpo_sign: weight of the accept set (A >= 0) against the reject set
    pmpo_entropy_scale: float = 0.05             # pmpo_sign: entropy bonus
    pmpo_prior_beta: float = 0.3                 # pmpo_sign: KL(pi || slow EMA copy of the actor) per dream step: a
    pmpo_prior_rate: float = 0.001               #   trust region (0 = off); the copy moves this far per update
    reinforce_entropy_scale: float = 0.025       # REINFORCE actor: entropy weight
    adv_norm: str = "std"                        # REINFORCE only (pmpo_sign reads each stream's raw sign A = lambda - V):
                                                 # "std": each stream centred and divided by its batch std;
                                                 # "range" (DreamerV3): each stream divided by its own return scale max(floor, EMA of P95 - P5), not centred.
                                                 # Then ext + curiosity_scale x cur
    adv_norm_decay: float = 0.99                 # "range": EMA of each stream's return range (P95 - P5 of the dream lambda-returns)
    adv_floor_ext: float = 1.0                   # "range": reward stream's floor (DreamerV3's L): a return range below it keeps its raw size, never scaled up
    adv_floor_cur: float = 1.0                   # "range": the curiosity stream's own floor, in curiosity units
    value_target: str = "hl_gauss"               # critic / reward-head targets: "hl_gauss" or "twohot"
    hl_sigma_ratio: float = 0.75                 # HL-Gauss: sigma = this x bin width (symlog units)
    critic_ema_tau: float = 0.95
    w_repval: float = 0.3                        # replay value loss
    play_unimix: bool = True                    # 1% uniform mix of the actor while playing (training keeps it)
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4

    # --- Exploration: cross-episode novelty (env) ---------------------------------------------------------------------
    cross_episode_novelty: bool = True           # a new square pays its category x max(min_rarity, 1 / sqrt(1 + n)),
                                                 #   n = earlier episodes that stood on it (the new-map bonus: the map's
                                                 #   n); the counts are kept by the agent and saved in checkpoints
    novelty_min_rarity: float = 0.1              # rarity floor: a square every episode visits still pays 10 % ...
    novelty_zero_visits: int = 1000              # ... until this many earlier episodes visited it: then it pays nothing,
                                                 #   and so does the new-map bonus of such a map (0 = off)
    auto_respawn: bool = True                    # a blackout returns to the town with a Pokémon Center (or Route 4 /
                                                 #   10) the episode entered last for the first time, not to Pallet (the
                                                 #   game moves it only when a nurse heals)

    # --- Speed (each flag computes the same result as with it off) ---------------------------------------------------
    compile: bool = True                         # torch.compile: encoder layers, decoder chunks, MLP heads, TSSM blocks
    act_cuda_graph: bool = True                  # acting step (encoder + actor + TSSM step) as one CUDA graph
    dec_sac: bool = True                         # decoder checkpoint keeps conv outputs, recomputes the rest
    dec_targets_once: bool = True                # pixel targets + entropy floor once per update, outside the checkpoint
    enc_fast_attn: bool = True                   # encoder attention: SDPA on views (same params, no MHA copies)
    skip_pad_frames: bool = True                 # posteriors of real frames only (pads copy their row's last frame)
    sync_free: bool = True                       # fused AdamW / Adam, GPU-side clipping, no host syncs per update
    h2d_stream: bool = True                      # the prefetcher copies batches to the GPU on a side stream
    async_save: bool = True                      # checkpoint + replay buffer written by a background thread
    async_dream_pdf: bool = True                 # dream PDFs rendered by a worker process

    # --- Diagnostics ------------------------------------------------------------------------------------------------
    drift_interval: int = 1000
    drift_frames: int = 256
    grad_share_interval: int = 500               # updates between per-term gradient norms on TSSM / encoder (0 = off)
    audit: bool = True                           # dream audit against the real game (False: no snapshots in play)
    audit_every: int = 5                         # rounds between audits (0 = off); only the play before one snapshots
    audit_starts: int = 50                       # start states per audit (30 % pre-event, the rest uniform)
    audit_horizon: int = 15                      # steps of each dream / real rollout (= dream_horizon)
    audit_snapshot_every: int = 100              # agent steps between the env snapshots taken in play (a PyBoy save
    audit_event_window: tuple = (2, 12)          # a snapshot 2-12 steps before a new memory bit is a "pre-event" start


    # --- Run control and outputs (read by the Trainer) -----------------------------------------------------------
    total_rounds: int = 10000                    # round = one whole episode per env, then updates_per_round updates
    updates_per_round: int = 500
    episode_steps: int = 10000                   # step budget of an episode (~67 min of game time)
    episode_extension: int = 2500                # + steps per new memory bit
    seed: int = 7
    checkpoint_interval: int = 1                 # rounds
    max_checkpoints: int = 10                     # ~1 GB each with the TSSM 1024 (+ optimizer state)
    play_progress_every: int = 1000              # play: console line every n steps
    train_progress_every: int = 50               # train: console line every n updates
    num_best_dreams: int = 3                     # dreams per round in the PDF
    num_random_dreams: int = 15
    csv_extra_columns: bool = False              # also log the diagnostic columns (EXTRA_COLUMNS) in the CSV
    reward_plot_interval: int = 1                # rounds
    exploration_map_interval: int = 3            # rounds between exploration maps of the replay buffer (0 = off): diagnostics/images/exploration_map_G<update>.png + plots/exploration_map.png
    reward_ma_window: int = 10                   # records in the plots' moving average


POKEMON_CONFIG = GQDreamerConfig()
LIGHT_CONFIG = replace(POKEMON_CONFIG, config_name="light", tssm_dim=512, tssm_heads=4, tssm_kv_heads=4,
                       tssm_ffn=1365)

MODEL_CONFIGS = {"pokemon": POKEMON_CONFIG, "light": LIGHT_CONFIG}


# config keys that no longer exist: dropped when a stored config is read
REMOVED_KEYS = ("loop_penalty", "loop_window", "loop_max_squares", "loop_break_after", "loop_break_steps",
                "adv_std_floor", "play_epsilon", "stuck_truncate_steps",
                "memory_gate", "memory_gate_decay", "memory_reward")
ADV_NORM_MODES = ("range", "std")


def _legacy_values(d):
    """Maps renamed config keys and values to the current ones and drops REMOVED_KEYS."""
    d = {k: v for k, v in d.items() if k not in REMOVED_KEYS}
    if "w_reward" in d:
        d.setdefault("w_standard_reward", d.pop("w_reward"))
    if d.get("adv_norm") is True:
        d["adv_norm"] = "range"
    if "adv_norm" in d and d["adv_norm"] not in ADV_NORM_MODES:
        del d["adv_norm"]
    return d


def model_config(name="pokemon", **overrides):
    """Named configuration with optional field overrides."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"unknown model configuration {name!r}; available: {sorted(MODEL_CONFIGS)}")
    return replace(MODEL_CONFIGS[name], **_legacy_values(overrides))


def config_from_dict(d, base=POKEMON_CONFIG):
    """Config from a dict (e.g. a checkpoint's); unknown keys are ignored."""
    names = {f.name for f in fields(GQDreamerConfig)}
    d = {k: tuple(v) if isinstance(v, list) else v for k, v in _legacy_values(d).items() if k in names}
    return replace(base, **d)


def config_to_dict(cfg):
    """Plain dict of a config (stored in checkpoints)."""
    return asdict(cfg)
