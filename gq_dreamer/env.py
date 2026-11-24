"""PyBoy environment for Pokémon Red (CGB, start state after the Pokédex) and its parallel wrappers.

Each step returns the screen reduced to 72 x 80 (majority colour of every 2 x 2 pixel block, 4-bit palette indices
packed two per byte), the RAM record (game.py), the 144-bit event memory and the explored grid. Death is not modelled:
a blackout respawns the player at the last Pokémon Center and the episode goes on; it is only logged (info "blackout").

Exploration: curiosity is the novelty of a new square (its map's category, x MAP_BONUS_MULT more for the first square
of a new map), and the 5 x 5 explored grid around the player (squares visited this episode) is part of every
observation, so the curiosity of a step is a function of what the agent sees. Cross-episode novelty: the square's value
is scaled by its rarity max(min_rarity, 1 / sqrt(1 + n)), n = the number of earlier episodes (all envs) that stood on
it, and the new-map bonus by the map's rarity; from n = zero_visits on the rarity is 0. The counts are kept by the
agent and pushed to every env before a play round (set_novelty); the envs report what their episodes visited
(pop_novelty). An episode ends when its step budget runs out."""
import io
import multiprocessing as mp
import os
import time
import traceback
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

warnings.filterwarnings("ignore", message="Using SDL2 binaries")    # pysdl2-dll banner
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .game import (ACTION_NAMES, ACTIONS, BALL_ITEMS, BLK_VALID_BIT, BLOCK_COLS, BLOCK_ROWS, BROCK_OPPONENT_ID,
                   EVENT_BYTES_READ, EVENT_FLAGS_ADDR, FRAME_BYTES_W, FRAME_H, FRAME_SCALE, FRAME_W, FRAMES_PER_STEP,
                   GRID_DIM, GRID_RADIUS, MAP_BONUS_MULT, MAP_CURIOSITY_TABLE, MAP_NAMES,
                   MEM_BADGE_OFFSET, MEM_BROCK_BATTLE_INDEX, MEM_POKEBALL_INDEX, MEMORY_DIM, MEMORY_REWARD, MON_FIELDS,
                   MON_HP, MON_LEVEL, MON_SPECIES, MON_STATUS, MON_TYPE1, MON_TYPE2, NUM_TRACKED_EVENTS,
                   PALETTE_SIZE, PARTY_SIZE, PATCH, POTION_ITEMS, PRESS_FRAMES, RAM_DIM, REC_BADGES, REC_BALLS,
                   REC_BATTLE, REC_ENEMY, REC_FACING, REC_MENU_CUR, REC_MENU_MAX, REC_MENU_X, REC_MENU_Y, REC_MONEY,
                   REC_PARTY, REC_POTIONS, REC_UI, REC_X, REC_Y, SCREEN_H, SCREEN_W, TRACKED_EVENTS,
                   RESPAWN_MAPS, UNPAID_EVENT_MASK, describe_memory_bit)

START_STATE = "PokemonRed.Start.state"
EPISODE_STEPS = 10000                  # step budget of an episode (~67 min of game time at 24 frames per step)
EPISODE_EXTENSION = 2500               # + steps per new memory bit
NOVELTY_MIN_RARITY = 0.1               # cross-episode novelty: floor of a square's / map's rarity 1 / sqrt(1 + n)
NOVELTY_ZERO_VISITS = 1000            #   ... and rarity 0 once this many earlier episodes visited it (0 = never)


# --- Palette compression --------------------------------------------------------------------------------------------
def palettize(rgb, size=PALETTE_SIZE):
    """[H, W, 3] uint8 -> (palette index [H, W], palette [size, 3]); extra colours map to the nearest kept one."""
    key = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)
    uniq, inv, counts = np.unique(key.ravel(), return_inverse=True, return_counts=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=-1).astype(np.int32)
    if len(uniq) > size:
        keep = np.argsort(-counts)[:size]
        d = ((cols[:, None, :] - cols[None, keep, :]) ** 2).sum(-1)
        inv = np.argmin(d, axis=1)[inv]
        cols = cols[keep]
    palette = np.zeros((size, 3), dtype=np.uint8)
    palette[:len(cols)] = cols
    return inv.reshape(rgb.shape[:2]).astype(np.uint8), palette, len(cols)


# per-process lookup tables of palettize_screen
_KEY_POS = np.zeros(1 << 15, dtype=np.intp)
_KEY_LUT = np.zeros(1 << 15, dtype=np.uint8)
_PIX_IDX = np.arange(SCREEN_H * SCREEN_W, dtype=np.intp)


def palettize_screen(screen_rgba, size=PALETTE_SIZE):
    """Fast palettize of PyBoy's RGBA screen via 15-bit GBC colour keys (exact fallback on collisions)
    -> (palette index [144, 160], palette [size, 3], number of colours used)."""
    if screen_rgba.shape != (SCREEN_H, SCREEN_W, 4) or not screen_rgba.flags["C_CONTIGUOUS"]:
        return palettize(screen_rgba[..., :3], size)
    v = screen_rgba.view(np.uint32).reshape(-1)                      # packed RGBA, R = low byte
    key = ((((v >> 3) & 0x1F) << 10) | (((v >> 11) & 0x1F) << 5) | ((v >> 19) & 0x1F)).astype(np.intp)
    uniq = np.flatnonzero(np.bincount(key, minlength=1 << 15))
    if len(uniq) > size:
        return palettize(screen_rgba[..., :3], size)
    _KEY_POS[key] = _PIX_IDX
    pos = _KEY_POS[uniq]
    _KEY_LUT[uniq] = np.arange(len(uniq), dtype=np.uint8)
    idx = _KEY_LUT[key]
    if not np.array_equal(v[pos][idx], v):                           # two colours shared a 15-bit key
        return palettize(screen_rgba[..., :3], size)
    palette = np.zeros((size, 3), dtype=np.uint8)
    palette[:len(uniq)] = screen_rgba[..., :3].reshape(-1, 3)[pos]
    return idx.reshape(SCREEN_H, SCREEN_W), palette, len(uniq)


_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def reduce_frame(idx, palette, n_colours):
    """Full-screen palette indices [144, 160] -> (packed 72 x 80 frame [72, 40] uint8, palette [16, 3]).

    The palette is reordered dark -> light, then every 2 x 2 pixel block keeps its most frequent colour; a tie keeps
    the darker one, so text strokes and outlines survive better than with plain subsampling. Indices stay palette
    indices (an average would mix colours and need RGB storage). Two pixels per byte, low nibble = left pixel."""
    order = np.argsort(palette[:n_colours].astype(np.float32) @ _LUMA, kind="stable")   # dark -> light
    remap = np.zeros(PALETTE_SIZE, dtype=np.uint8)
    remap[order] = np.arange(n_colours, dtype=np.uint8)
    pal = np.zeros_like(palette)
    pal[:n_colours] = palette[order]
    v = remap[idx].reshape(FRAME_H, FRAME_SCALE, FRAME_W, FRAME_SCALE).transpose(0, 2, 1, 3)
    v = v.reshape(FRAME_H, FRAME_W, -1)
    counts = (v[..., :, None] == v[..., None, :]).sum(-1)                # [72, 80, 4]: how often each pixel's colour
    score = counts.astype(np.int16) * 16 + (15 - v.astype(np.int16))   # most frequent, then darkest
    frame = np.take_along_axis(v, score.argmax(-1)[..., None], axis=-1)[..., 0]
    return (frame[:, 0::2] | (frame[:, 1::2] << 4)).astype(np.uint8), pal


# --- Environment ----------------------------------------------------------------------------------------------------
class PokemonRedEnv(gym.Env):
    """Gym env over PyBoy; RAM map from the pokered disassembly (github.com/pret/pokered, pokered.sym)."""

    # WRAM addresses
    RAM_FACING = 0xC109         # wSpritePlayerStateData1FacingDirection: 0 down, 4 up, 8 left, 0xC right
    RAM_MENU_TOP_Y = 0xCC24     # wTopMenuItemY: screen tile row of the open menu's first item
    RAM_MENU_TOP_X = 0xCC25     # wTopMenuItemX
    RAM_MENU_CUR = 0xCC26       # wCurrentMenuItem: cursor index (checked: START menu, DOWN moves it 0 -> 1 -> 2)
    RAM_MENU_MAX = 0xCC28       # wMaxMenuItem
    RAM_FONT_LOADED = 0xCFC4    # bit 0 set while a text box / menu is up
    RAM_ENEMY_MON = 0xCFE5      # wEnemyMon: species, HP (2, big endian), box level, status, type 1, type 2 ...
    RAM_ENEMY_LEVEL = 0xCFF3
    RAM_ENEMY_MAX_HP = 0xCFF4
    RAM_IS_IN_BATTLE = 0xD057   # 0 none, 1 wild, 2 trainer, 0xFF lost
    RAM_CUR_OPPONENT = 0xD059
    RAM_PARTY_COUNT = 0xD163
    RAM_PARTY_MONS = 0xD16B     # 44 bytes per Pokémon
    PARTY_MON_BYTES = 44
    RAM_NUM_BAG_ITEMS = 0xD31D
    RAM_BAG_ITEMS = 0xD31E      # (item id, quantity) pairs
    RAM_MONEY = 0xD347          # 3 BCD bytes
    RAM_BADGES = 0xD356
    RAM_MAP_ID = 0xD35E
    RAM_LAST_BLACKOUT_MAP = 0xD719  # wLastBlackoutMap: where a blackout returns (checked: 0x01 -> Viridian City)
    RAM_Y = 0xD361
    RAM_X = 0xD362

    BTN = {
        "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
        "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
        "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
        "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
        "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    }
    ACTIONS = ACTIONS
    ACTION_NAMES = ACTION_NAMES
    # Python-side state that step() reads: copied into / restored from an env snapshot (the emulator state aside). The
    # buttons are pressed and released within a step, so no joypad state survives between steps.
    SNAPSHOT_ATTRS = ("current_step", "max_step_limit", "memory", "_start_events", "_start_badges", "_start_pokeball",
                      "max_level_reward", "last_hp_fraction_sum", "last_party_count", "blackouts", "_blackout_latch",
                      "episode_visited_tiles", "episode_maps")

    def __init__(self, rom_path, state_path, verbose=False, window="SDL2", speed=0, frames_per_step=FRAMES_PER_STEP,
                 episode_steps=EPISODE_STEPS, episode_extension=EPISODE_EXTENSION, cross_episode_novelty=True,
                 novelty_min_rarity=NOVELTY_MIN_RARITY, novelty_zero_visits=NOVELTY_ZERO_VISITS, auto_respawn=True):
        super().__init__()
        self.state_path = state_path
        self.verbose = verbose
        self.frames_per_step = frames_per_step
        self._render_all = window == "SDL2"                          # render every frame when a human watches

        # The ROM goes in as an in-memory file: PyBoy then never reads or writes the "<rom>.ram" save file next to it
        # (every worker shared that file). Cartridge RAM comes from the start state anyway (reset() loads it).
        with open(rom_path, "rb") as f:
            rom = f.read()
        try:                                                         # sound_emulated=False skips the APU (~15% faster)
            self.pyboy = PyBoy(io.BytesIO(rom), window=window, sound_volume=0, sound_emulated=False, cgb=True)
        except TypeError:                                            # PyBoy without the sound_emulated option
            self.pyboy = PyBoy(io.BytesIO(rom), window=window, sound_volume=0, cgb=True)
        self.pyboy.set_emulation_speed(speed)

        self.init_steps = int(episode_steps)                         # step budget, + episode_extension per memory bit
        self.step_increase = int(episode_extension)
        self.reward_event_val = MEMORY_REWARD                        # per new memory bit (dreams: sparse reward head)
        self.reward_heal_mult = 5.0                                  # per party HP fraction healed
        self.reward_lvl_mult = 5.0                                   # per party level (x 1/4 beyond 22 levels)
        self.cross_episode_novelty = bool(cross_episode_novelty)     # False: every new square pays its full value
        self.novelty_min_rarity = float(novelty_min_rarity)
        self.novelty_zero_visits = int(novelty_zero_visits)          # n >= this: rarity 0 (0 = off)
        self.auto_respawn = bool(auto_respawn)                       # first entry of a RESPAWN_MAPS map this episode
        self.respawn_maps = RESPAWN_MAPS                             #   sets the blackout respawn there
        # cross-episode counts (map, x, y) / map id -> earlier episodes that visited it: set by the agent before a play
        # round and constant during it; what this env's episodes visit is reported back by pop_novelty
        self.novelty_tiles, self.novelty_maps = {}, {}
        self._delta_tiles, self._delta_maps = {}, {}

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(0, 255, (FRAME_H, FRAME_BYTES_W), np.uint8),
            "palette": spaces.Box(0, 255, (PALETTE_SIZE, 3), np.uint8),
            "blocks": spaces.Box(0, 255, (BLOCK_ROWS, BLOCK_COLS), np.uint8),
            "grid": spaces.Box(0, 1, (GRID_DIM,), np.float32),
            "ram": spaces.Box(0, 255, (RAM_DIM,), np.uint8),
            "memory": spaces.Box(0, 1, (MEMORY_DIM,), np.float32),
        })

        self.current_step = 0
        self.max_step_limit = self.init_steps
        self.memory = np.zeros(MEMORY_DIM, dtype=np.float32)
        self._start_events = np.zeros(NUM_TRACKED_EVENTS, dtype=bool)
        self._start_badges = 0
        self._start_pokeball = False
        self.max_level_reward = 0.0
        self.last_hp_fraction_sum = 0.0
        self.last_party_count = 0
        self.blackouts = 0
        self._blackout_latch = False
        self.episode_visited_tiles = set()
        self.episode_maps = set()

    # --- RAM helpers ------------------------------------------------------------------------------------------------
    def _mem(self, addr):
        """One RAM byte."""
        return self.pyboy.memory[addr]

    def _read16be(self, addr):
        """Big-endian 16-bit RAM read (Pokémon stats)."""
        return (self.pyboy.memory[addr] << 8) | self.pyboy.memory[addr + 1]

    def _get_current_position(self):
        """(map id, x, y) in step squares."""
        return self._mem(self.RAM_MAP_ID), self._mem(self.RAM_X), self._mem(self.RAM_Y)

    def _battle_kind(self):
        """0 overworld, 1 wild battle, 2 trainer battle (a lost battle, 0xFF, counts as wild)."""
        b = self._mem(self.RAM_IS_IN_BATTLE)
        return 0 if b == 0 else (2 if b == 2 else 1)

    def _ui_open(self):
        """True while a text box or menu is up."""
        return (self._mem(self.RAM_FONT_LOADED) & 1) == 1

    def _party(self):
        """[(species, type 1, type 2, level, HP, max HP, status)] of the party."""
        count = min(self._mem(self.RAM_PARTY_COUNT), PARTY_SIZE)
        mons = []
        for i in range(count):
            b = self.RAM_PARTY_MONS + i * self.PARTY_MON_BYTES
            mons.append((self._mem(b), self._mem(b + 5), self._mem(b + 6), self._mem(b + 0x21),
                         self._read16be(b + 1), self._read16be(b + 0x22), self._mem(b + 4)))
        return mons

    def _bag(self):
        """[(item id, quantity)] of the bag."""
        n = min(self._mem(self.RAM_NUM_BAG_ITEMS), 20)
        return [(self._mem(self.RAM_BAG_ITEMS + 2 * i), self._mem(self.RAM_BAG_ITEMS + 2 * i + 1)) for i in range(n)]

    def _item_counts(self):
        """(Poké Balls of any kind, HP-healing potions of any kind) in the bag."""
        bag = self._bag()
        return (sum(q for item, q in bag if item in BALL_ITEMS), sum(q for item, q in bag if item in POTION_ITEMS))

    def _money(self):
        """Money decoded from its 3 BCD bytes."""
        v = 0
        for i in range(3):
            b = self._mem(self.RAM_MONEY + i)
            v = v * 100 + (b >> 4) * 10 + (b & 0x0F)
        return v

    def _event_flags(self):
        """Tracked story event flags [NUM_TRACKED_EVENTS] bool."""
        block = np.frombuffer(bytes(self.pyboy.memory[EVENT_FLAGS_ADDR:EVENT_FLAGS_ADDR + EVENT_BYTES_READ]),
                              dtype=np.uint8)
        return np.unpackbits(block, bitorder="little")[TRACKED_EVENTS].astype(bool)

    # --- Observation ------------------------------------------------------------------------------------------------
    def _get_explored_grid(self, coord):
        """5x5 explored grid around the player, read before the current square is marked."""
        map_id, x, y = coord
        grid = np.zeros(GRID_DIM, dtype=np.float32)
        i = 0
        for dy in range(-GRID_RADIUS, GRID_RADIUS + 1):
            for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):
                grid[i] = 1.0 if (map_id, x + dx, y + dy) in self.episode_visited_tiles else 0.0
                i += 1
        return grid

    def _get_blocks(self):
        """[9, 10] overlay of the step squares: OAM sprite coverage bin + overworld-valid bit."""
        cover = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)
        for i in range(40):
            s = self.pyboy.get_sprite(i)
            if not s.on_screen:
                continue
            w, h = s.shape
            x0, y0 = max(s.x, 0), max(s.y, 0)
            x1, y1 = min(s.x + w, SCREEN_W), min(s.y + h, SCREEN_H)
            if x1 > x0 and y1 > y0:
                cover[y0:y1, x0:x1] = 1
        area = cover.reshape(BLOCK_ROWS, PATCH, BLOCK_COLS, PATCH).sum(axis=(1, 3))
        bins = np.digitize(area, (1, 64, 128)).astype(np.uint8)        # none / <25% / <50% / >=50%
        if self._battle_kind() == 0:
            bins |= np.uint8(1 << BLK_VALID_BIT)
        return bins

    def _get_ram_record(self, coord, party):
        """RAM record [RAM_DIM] uint8 (layout in game.py): mode, position, badges, money, items, menu, party,
        opponent."""
        r = np.zeros(RAM_DIM, dtype=np.uint8)
        battle = self._battle_kind()
        ui = self._ui_open()
        r[REC_BATTLE] = battle
        r[REC_UI] = ui
        r[REC_X], r[REC_Y] = min(coord[1], 255), min(coord[2], 255)
        r[REC_FACING] = (self._mem(self.RAM_FACING) >> 2) & 3
        r[REC_BADGES] = self._mem(self.RAM_BADGES)
        r[REC_MONEY] = min(int(round(18.0 * np.log1p(self._money()))), 255)
        balls, potions = self._item_counts()
        r[REC_BALLS], r[REC_POTIONS] = min(balls, 255), min(potions, 255)
        if ui:                                                       # the menu bytes keep stale values otherwise
            r[REC_MENU_CUR] = self._mem(self.RAM_MENU_CUR)
            r[REC_MENU_MAX] = self._mem(self.RAM_MENU_MAX)
            r[REC_MENU_Y], r[REC_MENU_X] = self._mem(self.RAM_MENU_TOP_Y), self._mem(self.RAM_MENU_TOP_X)
        for i, (species, t1, t2, level, hp, max_hp, status) in enumerate(party):
            b = REC_PARTY + i * MON_FIELDS
            r[b + MON_SPECIES], r[b + MON_TYPE1], r[b + MON_TYPE2], r[b + MON_LEVEL] = species, t1, t2, level
            r[b + MON_HP] = int(round(255.0 * min(hp, max_hp) / max_hp)) if max_hp > 0 else 0
            r[b + MON_STATUS] = status
        if battle:
            e = self.RAM_ENEMY_MON
            max_hp = self._read16be(self.RAM_ENEMY_MAX_HP)
            r[REC_ENEMY + MON_SPECIES] = self._mem(e)
            r[REC_ENEMY + MON_TYPE1], r[REC_ENEMY + MON_TYPE2] = self._mem(e + 5), self._mem(e + 6)
            r[REC_ENEMY + MON_LEVEL] = self._mem(self.RAM_ENEMY_LEVEL)
            r[REC_ENEMY + MON_HP] = int(round(255.0 * min(self._read16be(e + 1), max_hp) / max_hp)) if max_hp else 0
        return r

    def _observe(self, coord, grid, party):
        """Observation dict of the current frame."""
        idx, palette, n = palettize_screen(self.pyboy.screen.ndarray)
        frame, palette = reduce_frame(idx, palette, n)
        return {
            "frame": frame,
            "palette": palette,
            "blocks": self._get_blocks(),
            "grid": grid,
            "ram": self._get_ram_record(coord, party),
            "memory": self.memory.copy(),
            "map_id": int(coord[0]),
        }

    # --- Input ------------------------------------------------------------------------------------------------------
    def _tick(self, count, render_last=True):
        """Advances the emulator, rendering only the last frame unless a human watches."""
        if count <= 0:
            return
        if self._render_all:
            for _ in range(count):
                self.pyboy.tick(1, True)
        else:
            if count > 1:
                self.pyboy.tick(count - 1, False)
            self.pyboy.tick(1, render_last)

    def _apply_action(self, action):
        """Presses the action's button for PRESS_FRAMES frames, then releases it for the rest of the step."""
        press, release = self.BTN[self.ACTIONS[action]]
        self.pyboy.send_input(press)
        self._tick(PRESS_FRAMES, render_last=False)
        self.pyboy.send_input(release)
        self._tick(self.frames_per_step - PRESS_FRAMES)

    # --- Progress ---------------------------------------------------------------------------------------------------
    def _memory_bits(self):
        """Memory bits set now and not in the start state (story events, Brock battle, first ball, badges); the flags
        in UNPAID_EVENTS never set theirs."""
        bits = np.zeros(MEMORY_DIM, dtype=np.float32)
        bits[:NUM_TRACKED_EVENTS] = self._event_flags() & ~self._start_events & ~UNPAID_EVENT_MASK
        bits[MEM_BROCK_BATTLE_INDEX] = float(self._mem(self.RAM_IS_IN_BATTLE) == 2
                                             and self._mem(self.RAM_CUR_OPPONENT) == BROCK_OPPONENT_ID)
        bits[MEM_POKEBALL_INDEX] = float(self._item_counts()[0] > 0 and not self._start_pokeball)
        new_badges = self._mem(self.RAM_BADGES) & ~self._start_badges
        bits[MEM_BADGE_OFFSET:] = [(new_badges >> b) & 1 for b in range(MEMORY_DIM - MEM_BADGE_OFFSET)]
        return bits

    def _level_reward(self, total_level):
        """Level reward of a party level sum: 5 per level, x 1/4 beyond 22 (grinding pays less)."""
        val = total_level if total_level <= 22 else (total_level - 22) / 4 + 22
        return self.reward_lvl_mult * val

    @staticmethod
    def _hp_fraction_sum(party):
        """Sum of the party's HP fractions."""
        return sum(hp / max_hp for _, _, _, _, hp, max_hp, _ in party if max_hp > 0)

    # --- Gym API ----------------------------------------------------------------------------------------------------
    def step(self, action):
        """One agent step: act, then compute curiosity, rewards and the blackout flag. Never terminates: the episode
        ends by truncation when its step budget runs out."""
        self._apply_action(action)
        self.current_step += 1
        sparse_reward, standard_reward = 0.0, 0.0
        logs = []
        coord = self._get_current_position()
        map_id = coord[0]
        party = self._party()

        # square curiosity: a new square pays its map category x its rarity across episodes, the first square of a new
        # map MAP_BONUS_MULT x the category x the map's rarity more (rarity 1: the square / map was never visited)
        grid = self._get_explored_grid(coord)
        curiosity = 0.0
        if self._battle_kind() == 0 and coord not in self.episode_visited_tiles:
            self.episode_visited_tiles.add(coord)
            base = float(MAP_CURIOSITY_TABLE[map_id])
            curiosity = base * self._rarity(self.novelty_tiles.get(coord, 0))
            self._delta_tiles[coord] = self._delta_tiles.get(coord, 0) + 1
            if map_id not in self.episode_maps:
                self.episode_maps.add(map_id)
                curiosity += MAP_BONUS_MULT * base * self._rarity(self.novelty_maps.get(map_id, 0))
                self._delta_maps[map_id] = self._delta_maps.get(map_id, 0) + 1
                if self.verbose:
                    logs.append(f"\033[1;96m[MAP] entered {MAP_NAMES.get(map_id, hex(map_id))}\033[0m")
                if self.auto_respawn and map_id in self.respawn_maps:
                    self.pyboy.memory[self.RAM_LAST_BLACKOUT_MAP] = map_id         # blackouts now return here
                    if self.verbose:
                        logs.append(f"\033[96m[RESPAWN] {MAP_NAMES.get(map_id, hex(map_id))}\033[0m")

        # event memory: every new bit pays +50 once and extends the budget; bits are sticky (once set, they stay set)
        bits = self._memory_bits()
        new = (bits > 0.5) & (self.memory < 0.5)
        if new.any():
            self.memory = np.maximum(self.memory, bits)
            n_new = int(new.sum())
            for idx in np.flatnonzero(new):
                logs.append(f"\033[1;95m[EVENT] +{self.reward_event_val:.0f} | {describe_memory_bit(int(idx))}\033[0m")
            self.max_step_limit += n_new * self.step_increase
            sparse_reward += n_new * self.reward_event_val
        n_events = int(self.memory.sum())

        # heal and level-up rewards; a heal from 0 HP (the respawn after a blackout) pays nothing
        hp_sum = self._hp_fraction_sum(party)
        if hp_sum > self.last_hp_fraction_sum and len(party) == self.last_party_count and self.last_hp_fraction_sum > 0:
            standard_reward += self.reward_heal_mult * (hp_sum - self.last_hp_fraction_sum)
        self.last_hp_fraction_sum = hp_sum
        self.last_party_count = len(party)
        total_level = sum(m[3] for m in party)
        lvl_val = self._level_reward(total_level)
        if lvl_val > self.max_level_reward:
            standard_reward += lvl_val - self.max_level_reward
            self.max_level_reward = lvl_val
            if self.verbose:
                logs.append(f"\033[1;92m[LEVEL] party level sum {total_level}\033[0m")

        # blackout (every party Pokémon at 0 HP), latched until the game heals the party: logged and stored only
        all_fainted = len(party) > 0 and all(m[4] == 0 for m in party)
        blackout = all_fainted and not self._blackout_latch
        self._blackout_latch = all_fainted
        if blackout:
            self.blackouts += 1
            if self.verbose:
                logs.append(f"\033[1;91m[BLACKOUT] #{self.blackouts} at {MAP_NAMES.get(map_id, hex(map_id))} "
                            f"(step {self.current_step})\033[0m")

        truncated = self.current_step >= self.max_step_limit
        info = {
            "coord": coord, "curiosity": curiosity,
            "sparse_reward": sparse_reward, "standard_reward": standard_reward,
            "steps": self.current_step, "limit": self.max_step_limit, "events": n_events,
            "map_id": int(map_id), "blackout": bool(blackout), "blackouts": self.blackouts,
            "badges": bin(self._mem(self.RAM_BADGES)).count("1"), "party_level": total_level,
            "maps": len(self.episode_maps), "logs": logs,
        }
        return self._observe(coord, grid, party), sparse_reward + standard_reward, False, truncated, info

    def reset(self, seed=None, options=None):
        """Loads the start state and clears the episode's progress."""
        super().reset(seed=seed)
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
        self.current_step = 0
        self.max_step_limit = self.init_steps
        self.pyboy.tick()

        start = self._get_current_position()
        self.episode_visited_tiles = {start}                         # the spawn square is not a discovery
        self.episode_maps = {start[0]}
        self._start_events = self._event_flags()
        self._start_badges = self._mem(self.RAM_BADGES)
        self._start_pokeball = self._item_counts()[0] > 0
        self.memory = np.zeros(MEMORY_DIM, dtype=np.float32)
        self.memory = np.maximum(self.memory, self._memory_bits())   # e.g. a trainer battle running in the state
        party = self._party()
        self.last_hp_fraction_sum = self._hp_fraction_sum(party)
        self.last_party_count = len(party)
        self.max_level_reward = self._level_reward(sum(m[3] for m in party))
        self.blackouts = 0
        self._blackout_latch = len(party) > 0 and all(m[4] == 0 for m in party)

        info = {"coord": start, "curiosity": 0.0, "events": int(self.memory.sum()), "map_id": int(start[0]),
                "steps": 0, "limit": self.max_step_limit, "blackout": False, "blackouts": 0,
                "badges": bin(self._start_badges).count("1"), "party_level": sum(m[3] for m in party),
                "maps": 1, "logs": []}
        return self._observe(start, self._get_explored_grid(start), party), info

    def rgb_frame(self):
        """Full 144x160 RGB frame."""
        return np.ascontiguousarray(self.pyboy.screen.ndarray[:, :, :3])

    # --- Cross-episode novelty -------------------------------------------------------------------------------------
    def _rarity(self, n_past):
        """Rarity of a square / map that n_past earlier episodes visited: max(min_rarity, 1 / sqrt(1 + n_past)), and 0
        from n_past = novelty_zero_visits on (0 = never)."""
        if not self.cross_episode_novelty:
            return 1.0
        if 0 < self.novelty_zero_visits <= n_past:
            return 0.0
        return max(self.novelty_min_rarity, 1.0 / float(np.sqrt(1.0 + n_past)))

    def set_novelty(self, tiles, maps):
        """Cross-episode counts {(map, x, y): episodes}, {map id: episodes} for the next episodes; clears the visits
        not yet reported (e.g. those of the dream audit's real rollouts)."""
        self.novelty_tiles, self.novelty_maps = dict(tiles), dict(maps)
        self._delta_tiles, self._delta_maps = {}, {}

    def pop_novelty(self):
        """-> ({(map, x, y): episodes}, {map id: episodes}) visited by this env's episodes since the last call or
        set_novelty (each square / map counted once per episode; the spawn square is not counted)."""
        d = (self._delta_tiles, self._delta_maps)
        self._delta_tiles, self._delta_maps = {}, {}
        return d

    # --- Snapshots (dream audit) ------------------------------------------------------------------------------------
    def get_snapshot(self):
        """Complete state of the env between two steps: the emulator (PyBoy save state) and every Python-side
        internal that step() reads (copies). restore_snapshot + the same actions give the same observations,
        rewards and infos."""
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        py = {}
        for name in self.SNAPSHOT_ATTRS:
            v = getattr(self, name)
            py[name] = v.copy() if hasattr(v, "copy") else v
        return {"emulator": buf.getvalue(), "py": py}

    def restore_snapshot(self, snap):
        """Restores a get_snapshot state (from any env on the same ROM)."""
        self.pyboy.load_state(io.BytesIO(snap["emulator"]))
        for name, v in snap["py"].items():
            setattr(self, name, v.copy() if hasattr(v, "copy") else v)

    def close(self):
        """Stops the emulator (no save file written)."""
        self.pyboy.stop(save=False)


# --- Vectorised environments ----------------------------------------------------------------------------------------
def _step_env(env, action, snapshot=False):
    """Steps env; with snapshot the info also carries env.get_snapshot() of the state after the step."""
    out = env.step(action)
    if snapshot:
        out[4]["snapshot"] = env.get_snapshot()
    return out


def _env_worker(conn, rom_path, state_path, window, speed, verbose, frames_per_step, episode_steps,
                episode_extension, cross_episode_novelty=True, novelty_min_rarity=NOVELTY_MIN_RARITY,
                novelty_zero_visits=NOVELTY_ZERO_VISITS, auto_respawn=True):
    """Subprocess loop serving one env over a pipe; any error goes back to the main process with its traceback."""
    env = None
    try:
        env = PokemonRedEnv(rom_path=rom_path, state_path=state_path, window=window, speed=speed, verbose=verbose,
                            frames_per_step=frames_per_step, episode_steps=episode_steps,
                            episode_extension=episode_extension, cross_episode_novelty=cross_episode_novelty,
                            novelty_min_rarity=novelty_min_rarity, novelty_zero_visits=novelty_zero_visits,
                            auto_respawn=auto_respawn)
        conn.send(("__ready__", None))                               # handshake: the emulator started
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                action, snap = data if isinstance(data, tuple) else (data, False)
                conn.send(_step_env(env, action, snap))
            elif cmd == "reset":
                conn.send(env.reset())
            elif cmd == "restore":                                   # dream audit: continue from a snapshot
                env.restore_snapshot(data)
                conn.send(True)
            elif cmd == "rgb_frame":
                conn.send(env.rgb_frame())
            elif cmd == "set_novelty":                               # cross-episode counts for the next episodes
                env.set_novelty(*data)
                conn.send(True)
            elif cmd == "pop_novelty":                               # squares / maps visited since then
                conn.send(env.pop_novelty())
            elif cmd == "close":
                break
            else:
                conn.send(("__error__", f"unknown command {cmd!r}"))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            conn.send(("__error__", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


# Workers only run the emulator: one BLAS / OpenMP thread each (a many-core host would otherwise start one thread per
# core in every worker).
_WORKER_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class ParallelPokemonEnvs:
    """One PyBoy per process; step() sends every command before reading any reply."""

    def __init__(self, num_envs, rom_path, state_path, window="null", speed=0, verbose=False,
                 frames_per_step=FRAMES_PER_STEP, episode_steps=EPISODE_STEPS, episode_extension=EPISODE_EXTENSION,
                 cross_episode_novelty=True, novelty_min_rarity=NOVELTY_MIN_RARITY,
                 novelty_zero_visits=NOVELTY_ZERO_VISITS, auto_respawn=True):
        ctx = mp.get_context("spawn")                                # PyBoy is not fork-safe
        self.num_envs = num_envs
        self.frames_per_step = frames_per_step
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._parents, self._procs = [], []
        saved_env = {k: os.environ.get(k) for k in _WORKER_THREAD_VARS}
        os.environ.update({k: "1" for k in _WORKER_THREAD_VARS})     # inherited by the spawned workers only
        try:
            for _ in range(num_envs):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_env_worker, daemon=True,
                                   args=(child, rom_path, state_path, window, speed, verbose, frames_per_step,
                                         episode_steps, episode_extension, cross_episode_novelty,
                                         novelty_min_rarity, novelty_zero_visits, auto_respawn))
                proc.start()
                child.close()
                self._parents.append(parent)
                self._procs.append(proc)
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._wait_ready(timeout=300)

    def __len__(self):
        return self.num_envs

    def _wait_ready(self, timeout):
        """Waits until every emulator has started; a worker that fails raises its error here, not at the first step."""
        deadline = time.time() + timeout
        for i, p in enumerate(self._parents):
            try:
                if not p.poll(max(deadline - time.time(), 0)):
                    raise RuntimeError(f"Env worker {i} did not start within {timeout} s")
                msg = p.recv()
            except (EOFError, OSError):
                msg = None
            if not (isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__ready__"):
                err = self._worker_error(i, msg)
                self.close()
                raise err

    def _worker_error(self, i, msg=None):
        """RuntimeError describing why worker i stopped answering (its traceback, or how the process ended)."""
        if msg is None:
            try:
                if self._parents[i].poll(1.0):
                    msg = self._parents[i].recv()
            except (EOFError, OSError):
                pass
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__error__":
            return RuntimeError(f"Env worker {i} failed:\n{msg[1]}")
        proc = self._procs[i]
        proc.join(timeout=2)
        code = proc.exitcode
        if code is None:
            how = "is still alive but its pipe is closed"
        elif code < 0:
            how = f"was killed by signal {-code}" + (" (SIGKILL: often the out-of-memory killer)" if code == -9 else
                                                    " (SIGSEGV: crash inside the emulator)" if code == -11 else "")
        else:
            how = f"exited with code {code} (its traceback, if any, is in the Jupyter server log)"
        return RuntimeError(f"Env worker {i} {how}")

    def _send(self, i, msg):
        try:
            self._parents[i].send(msg)
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
            raise self._worker_error(i) from None

    def _recv(self, i):
        try:
            result = self._parents[i].recv()
        except (EOFError, ConnectionResetError, OSError):
            raise self._worker_error(i) from None
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str) and result[0] == "__error__":
            raise RuntimeError(f"Env worker {i} failed:\n{result[1]}")
        return result

    def reset(self):
        """Resets every env -> [(obs, info)]."""
        for i in range(self.num_envs):
            self._send(i, ("reset", None))
        return [self._recv(i) for i in range(self.num_envs)]

    def reset_one(self, i):
        """Resets env i -> (obs, info)."""
        self._send(i, ("reset", None))
        return self._recv(i)

    def rgb_frame(self, i):
        """Full RGB frame of env i."""
        self._send(i, ("rgb_frame", None))
        return self._recv(i)

    def restore(self, snapshots):
        """Env i continues from snapshots[i] (get_snapshot of any env on this ROM; None = left as it is)."""
        sent = [i for i, s in enumerate(snapshots) if s is not None]
        for i in sent:
            self._send(i, ("restore", snapshots[i]))
        for i in sent:
            self._recv(i)

    def set_novelty(self, tiles, maps):
        """Gives every env the cross-episode counts (PokemonRedEnv.set_novelty)."""
        for i in range(self.num_envs):
            self._send(i, ("set_novelty", (tiles, maps)))
        for i in range(self.num_envs):
            self._recv(i)

    def pop_novelty(self):
        """-> [(tiles, maps)] visited by each env since the last call (PokemonRedEnv.pop_novelty)."""
        for i in range(self.num_envs):
            self._send(i, ("pop_novelty", None))
        return [self._recv(i) for i in range(self.num_envs)]

    def step(self, action_idxs, active=None, snapshot=None):
        """Steps the active envs -> list of step results (None for inactive envs); envs flagged in `snapshot` also put
        their post-step snapshot into the info (taken in the worker)."""
        active = active or [True] * self.num_envs
        snapshot = snapshot or [False] * self.num_envs
        for i in range(self.num_envs):
            if active[i]:
                self._send(i, ("step", (int(action_idxs[i]), bool(snapshot[i]))))
        results = [None] * self.num_envs
        for i in range(self.num_envs):
            if active[i]:
                results[i] = self._recv(i)
        return results

    def close(self):
        """Stops every worker."""
        for p in self._parents:
            try:
                p.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


class SequentialPokemonEnvs:
    """Same interface as ParallelPokemonEnvs, in-process (debugging)."""

    def __init__(self, envs):
        self._envs = envs
        self.num_envs = len(envs)
        self.frames_per_step = envs[0].frames_per_step
        self.action_space = envs[0].action_space

    def __len__(self):
        return self.num_envs

    def reset(self):
        """Resets every env -> [(obs, info)]."""
        return [e.reset() for e in self._envs]

    def reset_one(self, i):
        """Resets env i -> (obs, info)."""
        return self._envs[i].reset()

    def rgb_frame(self, i):
        """Full RGB frame of env i."""
        return self._envs[i].rgb_frame()

    def restore(self, snapshots):
        """Env i continues from snapshots[i] (None = left as it is)."""
        for e, s in zip(self._envs, snapshots):
            if s is not None:
                e.restore_snapshot(s)

    def set_novelty(self, tiles, maps):
        """Gives every env the cross-episode counts (PokemonRedEnv.set_novelty)."""
        for e in self._envs:
            e.set_novelty(tiles, maps)

    def pop_novelty(self):
        """-> [(tiles, maps)] visited by each env since the last call (PokemonRedEnv.pop_novelty)."""
        return [e.pop_novelty() for e in self._envs]

    def step(self, action_idxs, active=None, snapshot=None):
        """Steps the active envs -> list of step results (None for inactive envs); `snapshot` as ParallelPokemonEnvs."""
        active = active or [True] * self.num_envs
        snapshot = snapshot or [False] * self.num_envs
        return [_step_env(e, int(action_idxs[i]), snapshot[i]) if active[i] else None
                for i, e in enumerate(self._envs)]

    def close(self):
        """Closes every env."""
        for e in self._envs:
            e.close()


def create_envs(num_envs, rom_path, state_dir="game", wind="null", speed=0, parallel=True,
                frames_per_step=FRAMES_PER_STEP, episode_steps=EPISODE_STEPS, episode_extension=EPISODE_EXTENSION,
                cross_episode_novelty=True, novelty_min_rarity=NOVELTY_MIN_RARITY,
                novelty_zero_visits=NOVELTY_ZERO_VISITS, auto_respawn=True):
    """Creates `num_envs` environments starting from PokemonRed.Start.state."""
    state_path = os.path.join(state_dir, START_STATE)
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Required start state not found: {state_path}. Only '{START_STATE}' is supported.")
    novelty = "off"
    if cross_episode_novelty:
        zero = f", 0 from {novelty_zero_visits:,} visits" if novelty_zero_visits > 0 else ""
        novelty = f"on (rarity floor {novelty_min_rarity:g}{zero})"
    print(f"\033[90m    [setup] launching {num_envs} emulators ({START_STATE}, parallel={parallel}, "
          f"{frames_per_step} frames/step, episodes of {episode_steps:,} steps + {episode_extension:,} per new "
          f"memory bit, blackouts respawn {'at the last newly entered town' if auto_respawn else 'as in the game'} "
          f"and are not terminal; cross-episode novelty {novelty})\033[0m")
    kw = dict(window=wind, speed=speed, frames_per_step=frames_per_step, episode_steps=episode_steps,
              episode_extension=episode_extension, cross_episode_novelty=cross_episode_novelty,
              novelty_min_rarity=novelty_min_rarity, novelty_zero_visits=novelty_zero_visits,
              auto_respawn=auto_respawn)
    if parallel:
        envs = ParallelPokemonEnvs(num_envs=num_envs, rom_path=rom_path, state_path=state_path, verbose=True, **kw)
    else:
        envs = SequentialPokemonEnvs([PokemonRedEnv(rom_path=rom_path, state_path=state_path, verbose=True, **kw)
                                      for _ in range(num_envs)])
    envs.rom_path, envs.state_path = rom_path, state_path
    return envs
