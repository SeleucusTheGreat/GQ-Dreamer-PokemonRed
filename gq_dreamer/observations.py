"""Model-side observation processing: the stored 72x80 frame, patches, sprite masks, RAM-record fields, the encoder's
side inputs and batching of env observations."""
import numpy as np
import torch
import torch.nn.functional as F

from .game import (BLK_SPRITE_MASK, BLOCK_COLS, BLOCK_ROWS, COORD_SCALE, FRAME_H, FRAME_W, GRID_DIM, ITEM_CAP,
                   MEMORY_DIM, MENU_CURSOR_CLASSES, MENU_X_SCALE, MENU_Y_SCALE, MON_FIELDS, MON_HP, MON_LEVEL,
                   MON_SPECIES, MON_STATUS, MON_TYPE1, MON_TYPE2, MONEY_SCALE, NUM_BATTLE_KINDS, NUM_TYPES, PARTY_SIZE,
                   REC_BADGES, REC_BALLS, REC_BATTLE, REC_ENEMY, REC_FACING, REC_MENU_CUR, REC_MENU_MAX, REC_MENU_X,
                   REC_MENU_Y, REC_MONEY, REC_PARTY, REC_POTIONS, REC_UI, REC_X, REC_Y, SQUARE_PX, TYPE_INDEX)

IMG_H, IMG_W = FRAME_H, FRAME_W                                       # 72 x 80: the encoder input and decoder target

# --- Side inputs ----------------------------------------------------------------------------------------------------
NUM_MAP_IDS = 256
MODE_DIM = 2                       # in battle, trainer battle
MAP_SIDE_DIM = MODE_DIM + 2        # + x, y
STATUS_DIM = 8 + 4 + 1             # badges, facing one-hot, money
MON_DIM = NUM_TYPES + 4            # type multi-hot, level / 100, HP fraction, status problem, present
ITEM_DIM = 4                       # Poké Balls, potions (min(n, 20) / 20), has a ball, has a potion
MENU_DIM = 1 + MENU_CURSOR_CLASSES + 3   # menu open, cursor one-hot, last index / 8, first item's screen y, x
GRID_BYTES = (GRID_DIM + 7) // 8
MEMORY_BYTES = (MEMORY_DIM + 7) // 8

# side tokens of the encoder (one each) and the config field of their dropout probability. The memory and the other
# vectors that are not on the screen are never dropped (their probability is 0 in the config).
SIDE_GROUPS = ("grid", "map", "status", "party", "enemy", "items", "menu", "memory")
N_SIDE_TOKENS = len(SIDE_GROUPS)


# --- Frames ---------------------------------------------------------------------------------------------------------
def autocast_dtype(device_type):
    """Current autocast dtype of this device, fp32 when autocast is off."""
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return torch.float32


def unpack_frame(packed):
    """[..., 72, 40] packed bytes -> [..., 72, 80] uint8 palette indices (low nibble = left pixel)."""
    return torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2)


def frame_rgb(frame, palette, dtype=torch.float32):
    """Packed frames [N, 72, 40] + palettes [N, P, 3] -> RGB [N, 3, 72, 80] in [0, 1]."""
    N = frame.shape[0]
    idx = unpack_frame(frame).reshape(N, -1, 1).long().expand(-1, -1, 3)
    pal = palette.to(dtype) / 255.0
    return torch.gather(pal, 1, idx).view(N, IMG_H, IMG_W, 3).permute(0, 3, 1, 2)


def frame_input(frame, palette, dtype=torch.float32, chunk=2048):
    """Packed frames + palettes -> encoder input [N, 3, 72, 80] in [-1, 1], written chunk by chunk in `dtype` without
    gradient (the pixels need none)."""
    N = frame.shape[0]
    out = torch.empty(N, 3, IMG_H, IMG_W, dtype=dtype, device=frame.device)
    with torch.no_grad():
        for i in range(0, N, chunk):
            out[i:i + chunk] = frame_rgb(frame[i:i + chunk], palette[i:i + chunk]).mul_(2.0).sub_(1.0)
    return out


def patch_grid(k=2):
    """(rows, cols, pixels per side, bottom pad, right pad) of the k x k-square patch grid of the 72x80 frame. The 9 x
    10 squares are padded below / right to a multiple of k (k = 2: 5 x 5 patches of 16 x 16 px, one padded square row;
    the player stands in the centre patch)."""
    rows, cols = -(-BLOCK_ROWS // k), -(-BLOCK_COLS // k)
    px = SQUARE_PX * k
    return rows, cols, px, rows * px - IMG_H, cols * px - IMG_W


def n_patches(k=2):
    """Number of k x k-square patches (25 for k = 2)."""
    rows, cols, *_ = patch_grid(k)
    return rows * cols


def patch_dim(k=2):
    """Pixel values per patch (768 for k = 2)."""
    return 3 * patch_grid(k)[2] ** 2


def patchify(img, k=2):
    """[N, 3, 72, 80] in [-1, 1] -> [N, n_patches, patch_dim], row-major; the padding is 0 (mid-grey)."""
    N = img.shape[0]
    rows, cols, px, pad_h, pad_w = patch_grid(k)
    x = F.pad(img, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else img
    x = x.reshape(N, 3, rows, px, cols, px)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(N, rows * cols, 3 * px * px)


def patch_pixel_mask(hide, k=2):
    """Patch mask [N, n_patches] -> pixel mask [N, 1, 72, 80] (True on the pixels of the hidden patches)."""
    N = hide.shape[0]
    rows, cols, px, *_ = patch_grid(k)
    m = hide.view(N, 1, rows, 1, cols, 1).expand(N, 1, rows, px, cols, px).reshape(N, 1, rows * px, cols * px)
    return m[..., :IMG_H, :IMG_W]


# --- Sprite masks ---------------------------------------------------------------------------------------------------
def sprite_tiles(blocks):
    """Overlay [N, 9, 10] -> bool [N, 9, 10]: squares that contain a sprite."""
    return (blocks.long() & BLK_SPRITE_MASK) > 0


def sprite_pixel_mask(blocks):
    """Overlay [N, 9, 10] -> bool [N, 72, 80]: pixels of sprite squares (8 px per square)."""
    return sprite_tiles(blocks).repeat_interleave(SQUARE_PX, dim=1).repeat_interleave(SQUARE_PX, dim=2)


def sprite_patches(blocks, k=2):
    """Overlay [N, 9, 10] -> bool [N, n_patches]: patches that contain a sprite square."""
    rows, cols, *_ = patch_grid(k)
    s = sprite_tiles(blocks).float()
    s = F.pad(s, (0, cols * k - BLOCK_COLS, 0, rows * k - BLOCK_ROWS))
    return F.max_pool2d(s[:, None], k)[:, 0].flatten(1) > 0


def sample_hide(blocks, mask_max=0.9, k=2):
    """MAE mask [N, n_patches]: each frame hides p ~ U(0, mask_max) of its background patches (sprite patches never)."""
    N, dev = blocks.shape[0], blocks.device
    p = torch.rand(N, 1, device=dev) * mask_max
    return (torch.rand(N, n_patches(k), device=dev) < p) & ~sprite_patches(blocks, k)


# --- RAM record -----------------------------------------------------------------------------------------------------
_TYPE_LUT = {}


def _type_multi_hot(t1, t2):
    """RAM type ids [...] x 2 -> multi-hot [..., 15] over the 15 real types."""
    if t1.device not in _TYPE_LUT:
        _TYPE_LUT[t1.device] = torch.from_numpy(TYPE_INDEX).to(t1.device)
    lut = _TYPE_LUT[t1.device]
    out = 0
    for t in (t1, t2):
        i = lut[t.long()]
        out = out + F.one_hot(i.clamp(min=0), NUM_TYPES).float() * (i >= 0).float()[..., None]
    return out.clamp(max=1.0)


def _mon_fields(rec):
    """[..., k >= 5] Pokémon fields -> (species long, present, features [..., NUM_TYPES + 2])."""
    species = rec[..., MON_SPECIES].long()
    present = (species > 0).float()
    feats = torch.cat([_type_multi_hot(rec[..., MON_TYPE1], rec[..., MON_TYPE2]) * present[..., None],
                       rec[..., MON_LEVEL:MON_LEVEL + 1].float() / 100.0,
                       rec[..., MON_HP:MON_HP + 1].float() / 255.0], dim=-1)
    return species, present, feats


def ram_fields(ram):
    """RAM record [..., RAM_DIM] uint8 -> dict of tensors (mode, coords, status, party, opponent, items, menu)."""
    r = ram.long()
    lead = r.shape[:-1]
    battle = r[..., REC_BATTLE].clamp(0, NUM_BATTLE_KINDS - 1)
    mode = torch.stack([(battle > 0).float(), (battle == 2).float()], dim=-1)
    coords = torch.stack([r[..., REC_X].float(), r[..., REC_Y].float()], dim=-1) / COORD_SCALE
    facing = r[..., REC_FACING].clamp(0, 3)
    badges = ((r[..., REC_BADGES:REC_BADGES + 1] >> torch.arange(8, device=r.device)) & 1).float()
    money = r[..., REC_MONEY].float() / MONEY_SCALE
    counts = torch.stack([r[..., REC_BALLS], r[..., REC_POTIONS]], dim=-1)
    items = counts.clamp(max=ITEM_CAP).float() / ITEM_CAP

    ui = r[..., REC_UI].clamp(0, 1).float()
    cursor = r[..., REC_MENU_CUR].clamp(0, MENU_CURSOR_CLASSES - 1)
    menu_pos = torch.stack([r[..., REC_MENU_MAX].clamp(max=MENU_CURSOR_CLASSES).float() / MENU_CURSOR_CLASSES,
                            r[..., REC_MENU_Y].float() / MENU_Y_SCALE, r[..., REC_MENU_X].float() / MENU_X_SCALE],
                           dim=-1).clamp(0.0, 1.0) * ui[..., None]

    party = r[..., REC_PARTY:REC_PARTY + PARTY_SIZE * MON_FIELDS].reshape(*lead, PARTY_SIZE, MON_FIELDS)
    p_species, p_present, p_feats = _mon_fields(party)
    p_status = (party[..., MON_STATUS] > 0).float() * p_present
    enemy = r[..., REC_ENEMY:REC_ENEMY + 5]
    e_species, _, e_feats = _mon_fields(enemy)
    e_present = (battle > 0).float()
    return {"battle": battle, "mode": mode, "coords": coords, "facing": facing, "badges": badges, "money": money,
            "items": items, "counts": counts, "ui": ui, "cursor": cursor, "menu_pos": menu_pos,
            "party_species": p_species, "party_present": p_present, "party_types": p_feats[..., :NUM_TYPES],
            "party_level": p_feats[..., NUM_TYPES], "party_hp": p_feats[..., NUM_TYPES + 1], "party_status": p_status,
            "party": torch.cat([p_feats, p_status[..., None], p_present[..., None]], dim=-1),
            "enemy_species": e_species * (battle > 0).long(), "enemy_present": e_present,
            "enemy_types": e_feats[..., :NUM_TYPES], "enemy_level": e_feats[..., NUM_TYPES],
            "enemy_hp": e_feats[..., NUM_TYPES + 1],
            "enemy": torch.cat([e_feats * e_present[..., None], torch.zeros_like(e_present)[..., None],
                                e_present[..., None]], dim=-1)}


def side_inputs(obs):
    """Observation dict -> the encoder's side inputs, one entry per SIDE_GROUPS token (party / enemy carry their
    species ids too)."""
    rf = ram_fields(obs["ram"])
    status = torch.cat([rf["badges"], F.one_hot(rf["facing"], 4).float(), rf["money"][..., None]], dim=-1)
    items = torch.cat([rf["items"], (rf["counts"] > 0).float()], dim=-1)
    menu = torch.cat([rf["ui"][..., None], F.one_hot(rf["cursor"], MENU_CURSOR_CLASSES).float() * rf["ui"][..., None],
                      rf["menu_pos"]], dim=-1)
    return {"grid": obs["grid"].float(), "map_id": obs["map_id"].long(),
            "map": torch.cat([rf["mode"], rf["coords"]], dim=-1), "status": status,
            "party_species": rf["party_species"], "party": rf["party"],
            "enemy_species": rf["enemy_species"], "enemy": rf["enemy"], "items": items, "menu": menu,
            "memory": obs["memory"].float()}


# --- Bit packing ----------------------------------------------------------------------------------------------------
def pack_bits(v):
    """Binary vector -> little-endian packed uint8 bytes."""
    return np.packbits(np.asarray(v) > 0.5, bitorder="little")


_SHIFTS = {}


def unpack_bits(p, n):
    """Packed bytes [..., B] uint8 -> float bits [..., n]."""
    if p.device not in _SHIFTS:
        _SHIFTS[p.device] = torch.arange(8, dtype=torch.uint8, device=p.device)
    bits = (p.unsqueeze(-1) >> _SHIFTS[p.device]) & 1
    return bits.reshape(*p.shape[:-1], p.shape[-1] * 8)[..., :n].float()


# --- Batching -------------------------------------------------------------------------------------------------------
def grid_input(o):
    """Env observation -> the model's grid vector [explored 25] (float32)."""
    return np.asarray(o["grid"], dtype=np.float32)


def stored_grid(grids):
    """Replay rows -> the model's grid [..., 25]: the unpacked explored bits."""
    return unpack_bits(grids, GRID_DIM)


def obs_to_tensors(obs_list, device):
    """List of env observations -> batched encoder inputs on device (frame packed, memory as float bits)."""
    return {
        "frame": torch.from_numpy(np.stack([o["frame"] for o in obs_list])).to(device),
        "palette": torch.from_numpy(np.stack([o["palette"] for o in obs_list])).to(device),
        "grid": torch.from_numpy(np.stack([grid_input(o) for o in obs_list])).float().to(device),
        "map_id": torch.tensor([int(o["map_id"]) for o in obs_list], dtype=torch.long, device=device),
        "ram": torch.from_numpy(np.stack([np.asarray(o["ram"], dtype=np.uint8) for o in obs_list])).to(device),
        "memory": torch.from_numpy(np.stack([o["memory"] for o in obs_list])).float().to(device),
    }
