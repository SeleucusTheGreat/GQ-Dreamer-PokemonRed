"""Pokémon Red constants: screen geometry, actions, maps, story events, the event memory, types, items and the RAM
record written by the env and read by the model."""
import numpy as np

# --- Screen geometry ------------------------------------------------------------------------------------------------
SCREEN_H, SCREEN_W = 144, 160                                  # full Game Boy screen (no HUD to crop)
PATCH = 16                                                     # one step square (2 x 2 background tiles)
BLOCK_ROWS, BLOCK_COLS = SCREEN_H // PATCH, SCREEN_W // PATCH  # 9 x 10 step squares
NUM_BLOCKS = BLOCK_ROWS * BLOCK_COLS                           # 90
PLAYER_ROW, PLAYER_COL = 4, 4                                  # the player always stands on this square
PALETTE_SIZE = 16                                              # colours stored per frame (4-bit indices)

# --- Stored / model frame: the screen reduced 2x (majority colour of each 2 x 2 pixel block) -------------------------
FRAME_SCALE = 2
FRAME_H, FRAME_W = SCREEN_H // FRAME_SCALE, SCREEN_W // FRAME_SCALE    # 72 x 80
SQUARE_PX = PATCH // FRAME_SCALE                                       # 8: one step square in the frame
FRAME_BYTES_W = FRAME_W // 2                                           # two 4-bit pixels per byte -> [72, 40]

# --- Control rate ---------------------------------------------------------------------------------------------------
FRAMES_PER_STEP = 24                                           # emulator frames per agent decision (400 ms)
PRESS_FRAMES = 8                                               # a button is held for 8 frames, then released

# --- Sprite overlay ("blocks") bit layout ---------------------------------------------------------------------------
BLK_SPRITE_MASK = 0x03                                         # bits 0-1: sprite coverage bin of the square
BLK_VALID_BIT = 2                                              # bit 2: overworld (not in battle)

# --- Actions --------------------------------------------------------------------------------------------------------
ACTIONS = ("up", "down", "left", "right", "a", "b")            # START / SELECT left out: no menu loops
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B")
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_A, ACTION_B = range(6)
MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)

# --- Maps (pokered constants/map_constants.asm) ---------------------------------------------------------------------
NUM_MAPS = 256
MAP_NAMES = {
    0x00: "Pallet Town", 0x01: "Viridian City", 0x02: "Pewter City", 0x03: "Cerulean City", 0x04: "Lavender Town",
    0x05: "Vermilion City", 0x06: "Celadon City", 0x07: "Fuchsia City", 0x08: "Cinnabar Island",
    0x09: "Indigo Plateau", 0x0A: "Saffron City", 0x0C: "Route 1", 0x0D: "Route 2", 0x0E: "Route 3", 0x0F: "Route 4",
    0x10: "Route 5", 0x11: "Route 6", 0x12: "Route 7", 0x13: "Route 8", 0x14: "Route 9", 0x15: "Route 10",
    0x16: "Route 11", 0x17: "Route 12", 0x18: "Route 13", 0x19: "Route 14", 0x1A: "Route 15", 0x1B: "Route 16",
    0x1C: "Route 17", 0x1D: "Route 18", 0x1E: "Route 19", 0x1F: "Route 20", 0x20: "Route 21", 0x21: "Route 22",
    0x22: "Route 23", 0x23: "Route 24", 0x24: "Route 25", 0x25: "Red's House 1F", 0x26: "Red's House 2F",
    0x27: "Blue's House", 0x28: "Oak's Lab", 0x29: "Viridian Pokécenter", 0x2A: "Viridian Mart",
    0x2B: "Viridian School House", 0x2C: "Viridian Nickname House", 0x2D: "Viridian Gym",
    0x2E: "Digletts Cave Route 2", 0x2F: "Viridian Forest North Gate", 0x30: "Route 2 Trade House",
    0x31: "Route 2 Gate", 0x32: "Viridian Forest South Gate", 0x33: "Viridian Forest", 0x34: "Museum 1F",
    0x35: "Museum 2F", 0x36: "Pewter Gym", 0x37: "Pewter Nidoran House", 0x38: "Pewter Mart",
    0x39: "Pewter Speech House", 0x3A: "Pewter Pokécenter", 0x3B: "Mt. Moon 1F", 0x3C: "Mt. Moon B1F",
    0x3D: "Mt. Moon B2F", 0x3E: "Cerulean Trashed House", 0x3F: "Cerulean Trade House", 0x40: "Cerulean Pokécenter",
    0x41: "Cerulean Gym", 0x42: "Bike Shop", 0x43: "Cerulean Mart", 0x44: "Mt. Moon Pokécenter",
    0x45: "Cerulean Trashed House Copy", 0x46: "Route 5 Gate", 0x47: "Underground Path Route 5", 0x48: "Daycare",
    0x49: "Route 6 Gate", 0x4A: "Underground Path Route 6", 0x4B: "Underground Path Route 6 Copy",
    0x4C: "Route 7 Gate", 0x4D: "Underground Path Route 7", 0x4E: "Underground Path Route 7 Copy",
    0x4F: "Route 8 Gate", 0x50: "Underground Path Route 8", 0x51: "Rock Tunnel Pokécenter", 0x52: "Rock Tunnel 1F",
    0x53: "Power Plant", 0x54: "Route 11 Gate 1F", 0x55: "Digletts Cave Route 11", 0x56: "Route 11 Gate 2F",
    0x57: "Route 12 Gate 1F", 0x58: "Bills House", 0x59: "Vermilion Pokécenter", 0x5A: "Pokemon Fan Club",
    0x5B: "Vermilion Mart", 0x5C: "Vermilion Gym", 0x5D: "Vermilion Pidgey House", 0x5E: "Vermilion Dock",
    0x5F: "S.S. Anne 1F", 0x60: "S.S. Anne 2F", 0x61: "S.S. Anne 3F", 0x62: "S.S. Anne B1F", 0x63: "S.S. Anne Bow",
    0x64: "S.S. Anne Kitchen", 0x65: "S.S. Anne Captains Room", 0x66: "S.S. Anne 1F Rooms",
    0x67: "S.S. Anne 2F Rooms", 0x68: "S.S. Anne B1F Rooms", 0x6C: "Victory Road 1F", 0x71: "Lances Room",
    0x76: "Hall Of Fame", 0x77: "Underground Path North South", 0x78: "Champions Room",
    0x79: "Underground Path West East", 0x7A: "Celadon Mart 1F", 0x7B: "Celadon Mart 2F", 0x7C: "Celadon Mart 3F",
    0x7D: "Celadon Mart 4F", 0x7E: "Celadon Mart Roof", 0x7F: "Celadon Mart Elevator", 0x80: "Celadon Mansion 1F",
    0x81: "Celadon Mansion 2F", 0x82: "Celadon Mansion 3F", 0x83: "Celadon Mansion Roof",
    0x84: "Celadon Mansion Roof House", 0x85: "Celadon Pokécenter", 0x86: "Celadon Gym", 0x87: "Game Corner",
    0x88: "Celadon Mart 5F", 0x89: "Game Corner Prize Room", 0x8A: "Celadon Diner", 0x8B: "Celadon Chief House",
    0x8C: "Celadon Hotel", 0x8D: "Lavender Pokécenter", 0x8E: "Pokemon Tower 1F", 0x8F: "Pokemon Tower 2F",
    0x90: "Pokemon Tower 3F", 0x91: "Pokemon Tower 4F", 0x92: "Pokemon Tower 5F", 0x93: "Pokemon Tower 6F",
    0x94: "Pokemon Tower 7F", 0x95: "Mr Fujis House", 0x96: "Lavender Mart", 0x97: "Lavender Cubone House",
    0x98: "Fuchsia Mart", 0x99: "Fuchsia Bills Grandpas House", 0x9A: "Fuchsia Pokécenter", 0x9B: "Wardens House",
    0x9C: "Safari Zone Gate", 0x9D: "Fuchsia Gym", 0x9E: "Fuchsia Meeting Room", 0x9F: "Seafoam Islands B1F",
    0xA0: "Seafoam Islands B2F", 0xA1: "Seafoam Islands B3F", 0xA2: "Seafoam Islands B4F",
    0xA3: "Vermilion Old Rod House", 0xA4: "Fuchsia Good Rod House", 0xA5: "Pokemon Mansion 1F", 0xA6: "Cinnabar Gym",
    0xA7: "Cinnabar Lab", 0xA8: "Cinnabar Lab Trade Room", 0xA9: "Cinnabar Lab Metronome Room",
    0xAA: "Cinnabar Lab Fossil Room", 0xAB: "Cinnabar Pokécenter", 0xAC: "Cinnabar Mart", 0xAD: "Cinnabar Mart Copy",
    0xAE: "Indigo Plateau Lobby", 0xAF: "Copycats House 1F", 0xB0: "Copycats House 2F", 0xB1: "Fighting Dojo",
    0xB2: "Saffron Gym", 0xB3: "Saffron Pidgey House", 0xB4: "Saffron Mart", 0xB5: "Silph Co 1F",
    0xB6: "Saffron Pokécenter", 0xB7: "Mr Psychics House", 0xB8: "Route 15 Gate 1F", 0xB9: "Route 15 Gate 2F",
    0xBA: "Route 16 Gate 1F", 0xBB: "Route 16 Gate 2F", 0xBC: "Route 16 Fly House", 0xBD: "Route 12 Super Rod House",
    0xBE: "Route 18 Gate 1F", 0xBF: "Route 18 Gate 2F", 0xC0: "Seafoam Islands 1F", 0xC1: "Route 22 Gate",
    0xC2: "Victory Road 2F", 0xC3: "Route 12 Gate 2F", 0xC4: "Vermilion Trade House", 0xC5: "Digletts Cave",
    0xC6: "Victory Road 3F", 0xC7: "Rocket Hideout B1F", 0xC8: "Rocket Hideout B2F", 0xC9: "Rocket Hideout B3F",
    0xCA: "Rocket Hideout B4F", 0xCB: "Rocket Hideout Elevator", 0xCF: "Silph Co 2F", 0xD0: "Silph Co 3F",
    0xD1: "Silph Co 4F", 0xD2: "Silph Co 5F", 0xD3: "Silph Co 6F", 0xD4: "Silph Co 7F", 0xD5: "Silph Co 8F",
    0xD6: "Pokemon Mansion 2F", 0xD7: "Pokemon Mansion 3F", 0xD8: "Pokemon Mansion B1F", 0xD9: "Safari Zone East",
    0xDA: "Safari Zone North", 0xDB: "Safari Zone West", 0xDC: "Safari Zone Center",
    0xDD: "Safari Zone Center Rest House", 0xDE: "Safari Zone Secret House", 0xDF: "Safari Zone West Rest House",
    0xE0: "Safari Zone East Rest House", 0xE1: "Safari Zone North Rest House", 0xE2: "Cerulean Cave 2F",
    0xE3: "Cerulean Cave B1F", 0xE4: "Cerulean Cave 1F", 0xE5: "Name Raters House", 0xE6: "Cerulean Badge House",
    0xE8: "Rock Tunnel B1F", 0xE9: "Silph Co 9F", 0xEA: "Silph Co 10F", 0xEB: "Silph Co 11F",
    0xEC: "Silph Co Elevator", 0xEF: "Trade Center", 0xF0: "Colosseum", 0xF5: "Loreleis Room", 0xF6: "Brunos Room",
    0xF7: "Agathas Room",
}
LAST_OUTDOOR_MAP = 0x24                                        # ids 0x00..0x24: towns and routes
DUNGEON_WORDS = ("Forest", "Mt. Moon", "S.S. Anne", "Rock Tunnel", "Power Plant", "Pokemon Tower", "Seafoam",
                 "Pokemon Mansion", "Safari Zone", "Victory Road", "Cerulean Cave", "Rocket Hideout", "Silph Co",
                 "Digletts Cave", "Underground Path")
ROOM_WORDS = ("Gate", "Pokécenter", "House", "Elevator", "Mart")  # small rooms, even when named after a dungeon
ENTRANCE_ROOMS = frozenset((0x2E, 0x55, 0x47, 0x4A, 0x4B, 0x4D, 0x4E, 0x50))   # Diglett's Cave / Underground entrances
BOSS_ROOMS = frozenset((0x71, 0x78, 0xF5, 0xF6, 0xF7))           # Elite Four and Champion
START_MAP_NAME = "Red's House 2F"

# --- Map curiosity: a new square pays its map category; the first visit of a map pays MAP_BONUS_MULT x more -------
CURIOSITY_VALUES = (0.05, 0.20, 0.50)                          # indoor room, town / route / dungeon, gym / E4
MAP_BONUS_MULT = 10.0
# Respawn helper (env auto_respawn): the maps with a Pokémon Center that a blackout can return to (the
# game's fly-warp table: every town but Pallet, Indigo Plateau, Route 4 = Mt. Moon's centre, Route 10 = Rock Tunnel's).
# The game moves its respawn point (wLastBlackoutMap) only when the player accepts a nurse's healing; the env moves it
# to each of these maps the first time an episode enters it.
RESPAWN_MAPS = frozenset({0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0F, 0x15})


def map_class_index(map_id):
    """Map category: 2 = gym / Elite Four, 1 = town, route or dungeon, 0 = house, shop, gate, centre."""
    name = MAP_NAMES.get(map_id, "")
    if "Gym" in name or map_id in BOSS_ROOMS:
        return 2
    if map_id in ENTRANCE_ROOMS or any(w in name for w in ROOM_WORDS):
        return 0
    if map_id <= LAST_OUTDOOR_MAP or any(w in name for w in DUNGEON_WORDS):
        return 1
    return 0


MAP_CLASS_INDEX = np.array([map_class_index(m) for m in range(NUM_MAPS)], dtype=np.int64)
MAP_CURIOSITY_TABLE = np.array([CURIOSITY_VALUES[c] for c in MAP_CLASS_INDEX], dtype=np.float32)

# --- Local grids (5x5 step squares centred on the player) -----------------------------------------------------------
# One channel, read before the current step is marked: explored (square visited this episode). A new square pays
# its map's curiosity; in dreams the grid head's explored centre gates the predicted curiosity. The encoder's grid
# token reads it and the grid head predicts it.
GRID_RADIUS = 2
GRID_SIDE = 2 * GRID_RADIUS + 1
GRID_DIM = GRID_SIDE * GRID_SIDE                               # 25 squares per channel
GRID_CENTER_INDEX = GRID_DIM // 2
GRID_INPUT_DIM = GRID_DIM                                      # explored 25

# --- Story events (pokered constants/event_constants.asm), sections reachable before Lt. Surge ----------------------
EVENT_FLAGS_ADDR = 0xD747                                      # wEventFlags: event e is bit e % 8 of byte e // 8
# The prologue (<= EVENT_GOT_POKEDEX) pays nothing. Flags already set in the start state (OAK_APPEARED_IN_PALLET,
# OAK_GOT_PARCEL, GOT_OAKS_PARCEL, 1ST_ROUTE22_RIVAL_BATTLE, ROUTE22_RIVAL_WANTS_BATTLE) stay 0 in the memory.
EVENT_SECTIONS = {
    "Pallet Town": {0x026: "PALLET_AFTER_GETTING_POKEBALLS_2", 0x027: "OAK_APPEARED_IN_PALLET"},
    "Viridian City": {0x028: "VIRIDIAN_GYM_OPEN", 0x029: "GOT_TM42", 0x038: "OAK_GOT_PARCEL",
        0x039: "GOT_OAKS_PARCEL", 0x050: "GOT_TM27", 0x051: "BEAT_VIRIDIAN_GYM_GIOVANNI",
        0x052: "BEAT_VIRIDIAN_GYM_TRAINER_0", 0x053: "BEAT_VIRIDIAN_GYM_TRAINER_1",
        0x054: "BEAT_VIRIDIAN_GYM_TRAINER_2", 0x055: "BEAT_VIRIDIAN_GYM_TRAINER_3",
        0x056: "BEAT_VIRIDIAN_GYM_TRAINER_4", 0x057: "BEAT_VIRIDIAN_GYM_TRAINER_5",
        0x058: "BEAT_VIRIDIAN_GYM_TRAINER_6", 0x059: "BEAT_VIRIDIAN_GYM_TRAINER_7"},
    "Pewter City": {0x068: "BOUGHT_MUSEUM_TICKET", 0x069: "GOT_OLD_AMBER", 0x072: "BEAT_PEWTER_GYM_TRAINER_0",
        0x076: "GOT_TM34", 0x077: "BEAT_BROCK"},
    "Cerulean City": {0x098: "BEAT_CERULEAN_RIVAL", 0x0A7: "BEAT_CERULEAN_ROCKET_THIEF",
        0x0BA: "BEAT_CERULEAN_GYM_TRAINER_0", 0x0BB: "BEAT_CERULEAN_GYM_TRAINER_1", 0x0BE: "GOT_TM11",
        0x0BF: "BEAT_MISTY", 0x0C0: "GOT_BICYCLE"},
    "Vermilion City": {0x151: "GOT_BIKE_VOUCHER", 0x156: "SEEL_FAN_BOAST", 0x157: "PIKACHU_FAN_BOAST",
        0x160: "2ND_LOCK_OPENED", 0x161: "1ST_LOCK_OPENED", 0x162: "BEAT_VERMILION_GYM_TRAINER_0",
        0x163: "BEAT_VERMILION_GYM_TRAINER_1", 0x164: "BEAT_VERMILION_GYM_TRAINER_2", 0x166: "GOT_TM24",
        0x167: "BEAT_LT_SURGE"},
    "Route 2": {0x3D8: "GOT_HM05"},
    "Route 3": {0x3E2: "BEAT_ROUTE_3_TRAINER_0", 0x3E3: "BEAT_ROUTE_3_TRAINER_1", 0x3E4: "BEAT_ROUTE_3_TRAINER_2",
        0x3E5: "BEAT_ROUTE_3_TRAINER_3", 0x3E6: "BEAT_ROUTE_3_TRAINER_4", 0x3E7: "BEAT_ROUTE_3_TRAINER_5",
        0x3E8: "BEAT_ROUTE_3_TRAINER_6", 0x3E9: "BEAT_ROUTE_3_TRAINER_7"},
    "Route 4": {0x3F2: "BEAT_ROUTE_4_TRAINER_0", 0x3FF: "BOUGHT_MAGIKARP"},
    "Route 6": {0x411: "BEAT_ROUTE_6_TRAINER_0", 0x412: "BEAT_ROUTE_6_TRAINER_1", 0x413: "BEAT_ROUTE_6_TRAINER_2",
        0x414: "BEAT_ROUTE_6_TRAINER_3", 0x415: "BEAT_ROUTE_6_TRAINER_4", 0x416: "BEAT_ROUTE_6_TRAINER_5"},
    "Route 11": {0x471: "BEAT_ROUTE_11_TRAINER_0", 0x472: "BEAT_ROUTE_11_TRAINER_1", 0x473: "BEAT_ROUTE_11_TRAINER_2",
        0x474: "BEAT_ROUTE_11_TRAINER_3", 0x475: "BEAT_ROUTE_11_TRAINER_4", 0x476: "BEAT_ROUTE_11_TRAINER_5",
        0x477: "BEAT_ROUTE_11_TRAINER_6", 0x478: "BEAT_ROUTE_11_TRAINER_7", 0x479: "BEAT_ROUTE_11_TRAINER_8",
        0x47A: "BEAT_ROUTE_11_TRAINER_9", 0x47F: "GOT_ITEMFINDER"},
    "Route 22": {0x520: "1ST_ROUTE22_RIVAL_BATTLE", 0x521: "2ND_ROUTE22_RIVAL_BATTLE",
        0x525: "BEAT_ROUTE22_RIVAL_1ST_BATTLE", 0x526: "BEAT_ROUTE22_RIVAL_2ND_BATTLE",
        0x527: "ROUTE22_RIVAL_WANTS_BATTLE"},
    "Route 24": {0x540: "GOT_NUGGET", 0x541: "BEAT_ROUTE24_ROCKET", 0x542: "BEAT_ROUTE_24_TRAINER_0",
        0x543: "BEAT_ROUTE_24_TRAINER_1", 0x544: "BEAT_ROUTE_24_TRAINER_2", 0x545: "BEAT_ROUTE_24_TRAINER_3",
        0x546: "BEAT_ROUTE_24_TRAINER_4", 0x547: "BEAT_ROUTE_24_TRAINER_5", 0x549: "NUGGET_REWARD_AVAILABLE"},
    "Route 25": {0x550: "MET_BILL", 0x551: "BEAT_ROUTE_25_TRAINER_0", 0x552: "BEAT_ROUTE_25_TRAINER_1",
        0x553: "BEAT_ROUTE_25_TRAINER_2", 0x554: "BEAT_ROUTE_25_TRAINER_3", 0x555: "BEAT_ROUTE_25_TRAINER_4",
        0x556: "BEAT_ROUTE_25_TRAINER_5", 0x557: "BEAT_ROUTE_25_TRAINER_6", 0x558: "BEAT_ROUTE_25_TRAINER_7",
        0x559: "BEAT_ROUTE_25_TRAINER_8", 0x55B: "USED_CELL_SEPARATOR_ON_BILL", 0x55C: "GOT_SS_TICKET",
        0x55D: "MET_BILL_2", 0x55E: "BILL_SAID_USE_CELL_SEPARATOR", 0x55F: "LEFT_BILLS_HOUSE_AFTER_HELPING"},
    "Viridian Forest": {0x562: "BEAT_VIRIDIAN_FOREST_TRAINER_0", 0x563: "BEAT_VIRIDIAN_FOREST_TRAINER_1",
        0x564: "BEAT_VIRIDIAN_FOREST_TRAINER_2"},
    "Mt. Moon": {0x571: "BEAT_MT_MOON_1_TRAINER_0", 0x572: "BEAT_MT_MOON_1_TRAINER_1",
        0x573: "BEAT_MT_MOON_1_TRAINER_2", 0x574: "BEAT_MT_MOON_1_TRAINER_3", 0x575: "BEAT_MT_MOON_1_TRAINER_4",
        0x576: "BEAT_MT_MOON_1_TRAINER_5", 0x577: "BEAT_MT_MOON_1_TRAINER_6", 0x579: "BEAT_MT_MOON_EXIT_SUPER_NERD",
        0x57A: "BEAT_MT_MOON_3_TRAINER_0", 0x57B: "BEAT_MT_MOON_3_TRAINER_1", 0x57C: "BEAT_MT_MOON_3_TRAINER_2",
        0x57D: "BEAT_MT_MOON_3_TRAINER_3", 0x57E: "GOT_DOME_FOSSIL", 0x57F: "GOT_HELIX_FOSSIL"},
    "S.S. Anne": {0x5C4: "BEAT_SS_ANNE_5_TRAINER_0", 0x5C5: "BEAT_SS_ANNE_5_TRAINER_1", 0x5E0: "GOT_HM01",
        0x5E1: "RUBBED_CAPTAINS_BACK", 0x5E2: "SS_ANNE_LEFT", 0x5E3: "WALKED_PAST_GUARD_AFTER_SS_ANNE_LEFT",
        0x5E4: "STARTED_WALKING_OUT_OF_DOCK", 0x5E5: "WALKED_OUT_OF_DOCK", 0x5F1: "BEAT_SS_ANNE_8_TRAINER_0",
        0x5F2: "BEAT_SS_ANNE_8_TRAINER_1", 0x5F3: "BEAT_SS_ANNE_8_TRAINER_2", 0x5F4: "BEAT_SS_ANNE_8_TRAINER_3",
        0x601: "BEAT_SS_ANNE_9_TRAINER_0", 0x602: "BEAT_SS_ANNE_9_TRAINER_1", 0x603: "BEAT_SS_ANNE_9_TRAINER_2",
        0x604: "BEAT_SS_ANNE_9_TRAINER_3", 0x611: "BEAT_SS_ANNE_10_TRAINER_0", 0x612: "BEAT_SS_ANNE_10_TRAINER_1",
        0x613: "BEAT_SS_ANNE_10_TRAINER_2", 0x614: "BEAT_SS_ANNE_10_TRAINER_3", 0x615: "BEAT_SS_ANNE_10_TRAINER_4",
        0x616: "BEAT_SS_ANNE_10_TRAINER_5"},
}
TRACKED_EVENTS = np.array(sorted(e for sec in EVENT_SECTIONS.values() for e in sec), dtype=np.int64)
EVENT_NAMES = {e: name for sec in EVENT_SECTIONS.values() for e, name in sec.items()}
EVENT_SECTION_OF = {e: s for s, sec in EVENT_SECTIONS.items() for e in sec}
NUM_TRACKED_EVENTS = len(TRACKED_EVENTS)                       # 134
EVENT_BYTES_READ = int(TRACKED_EVENTS.max()) // 8 + 1
# Tracked flags that never pay: they keep their place in the layout, but their bit stays 0.
# PALLET_AFTER_GETTING_POKEBALLS_2 depends on conditions the agent cannot see (no ball in the bag, fewer than 2 Pokémon
# owned), so paying it makes the agent circle between Oak and the lab door.
UNPAID_EVENTS = (0x026,)
UNPAID_EVENT_MASK = np.isin(TRACKED_EVENTS, UNPAID_EVENTS)
BEAT_BROCK_EVENT = 0x077

# --- Event memory: every bit that pays the +50 sparse reward once per episode ---------------------------------------
# It is part of the agent state s = [h, z, m] and of the TSSM input, and the memory head reads it back from z.
BADGE_NAMES = ("Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth")
MEM_EVENT_OFFSET = 0                                           # bits 0..133: story events
MEM_BROCK_BATTLE_INDEX = NUM_TRACKED_EVENTS                    # 134: the Brock battle started
MEM_POKEBALL_INDEX = NUM_TRACKED_EVENTS + 1                    # 135: first Poké Ball in the bag
MEM_BADGE_OFFSET = NUM_TRACKED_EVENTS + 2                      # 136..143: badges
MEMORY_DIM = MEM_BADGE_OFFSET + len(BADGE_NAMES)               # 144
MEMORY_BYTES = (MEMORY_DIM + 7) // 8                           # 18 packed bytes per stored row
MEMORY_REWARD = 50.0                                           # paid once per new bit (env, dream rule, sampling)
BROCK_OPPONENT_ID = 200 + 34                                   # wCurOpponent in the Brock battle (OPP_BROCK)


def describe_memory_bit(idx):
    """Readable name of a memory bit."""
    if idx < MEM_BROCK_BATTLE_INDEX:
        e = int(TRACKED_EVENTS[idx - MEM_EVENT_OFFSET])
        return f"{EVENT_NAMES[e]} (0x{e:03X}, {EVENT_SECTION_OF[e]}{', unpaid' if e in UNPAID_EVENTS else ''})"
    if idx == MEM_BROCK_BATTLE_INDEX:
        return "Brock battle started"
    if idx == MEM_POKEBALL_INDEX:
        return "first Poké Ball"
    return f"{BADGE_NAMES[idx - MEM_BADGE_OFFSET]} Badge"


# --- Pokémon, types and items ---------------------------------------------------------------------------------------
PARTY_SIZE = 6
NUM_SPECIES_IDS = 256                                          # internal species index (0 = empty slot)
TYPE_NAMES = ("Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost",
              "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon")
TYPE_RAM_IDS = (0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x08, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A)
NUM_TYPES = len(TYPE_NAMES)                                    # 15
TYPE_INDEX = np.full(256, -1, dtype=np.int64)                  # RAM type id -> 0..14 (-1 = none)
TYPE_INDEX[list(TYPE_RAM_IDS)] = np.arange(NUM_TYPES)
BALL_ITEMS = frozenset((0x01, 0x02, 0x03, 0x04))               # Master, Ultra, Great, Poké Ball
POTION_ITEMS = frozenset((0x10, 0x11, 0x12, 0x13, 0x14))       # Full Restore, Max / Hyper / Super / Potion
ITEM_CAP = 20                                                  # item counts are given to the model as min(n, 20) / 20

# --- RAM record: one uint8 vector per frame, written by the env and read by the model -------------------------------
REC_BATTLE = 0                     # 0 overworld, 1 wild battle, 2 trainer battle
REC_UI = 1                         # 1 while a text box or menu is open
REC_X, REC_Y = 2, 3                # map coordinates (step squares)
REC_FACING = 4                     # 0 down, 1 up, 2 left, 3 right
REC_BADGES = 5                     # badge bits
REC_MONEY = 6                      # round(18 * log(1 + money)), 0..249
REC_BALLS, REC_POTIONS = 7, 8      # bag counts (capped at 255)
REC_MENU_CUR = 9                   # menu cursor index (wCurrentMenuItem), 0 while no menu is open
REC_MENU_MAX = 10                  # last cursor index of the open menu (wMaxMenuItem)
REC_MENU_Y, REC_MENU_X = 11, 12    # screen tile of the menu's first item (wTopMenuItemY / X): which menu it is
REC_PARTY = 13                     # 6 slots x MON_FIELDS
MON_SPECIES, MON_TYPE1, MON_TYPE2, MON_LEVEL, MON_HP, MON_STATUS = range(6)   # HP = round(255 * HP / max HP)
MON_FIELDS = 6
REC_ENEMY = REC_PARTY + PARTY_SIZE * MON_FIELDS                # 49: opponent (species, types, level, HP)
RAM_DIM = REC_ENEMY + 5                                        # 54
NUM_BATTLE_KINDS = 3
MENU_CURSOR_CLASSES = 8            # cursor index given to the model as a one-hot over 0..7 (clamped)
COORD_SCALE = 64.0                 # coordinates are given to the model as x / 64
MONEY_SCALE = 250.0                # REC_MONEY / 250 lies in [0, 1]
MENU_Y_SCALE, MENU_X_SCALE = 18.0, 20.0   # screen tiles
