"""Verify the renderer-relevant prop set in ONE parse_ticks call (fast, no per-prop cost)."""
import warnings, json
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
TICKS = list(range(30000, 30100))

RENDER = [
    # transform
    "X", "Y", "Z", "pitch", "yaw",
    # vitals
    "health", "armor_value", "armor", "has_helmet", "has_defuser", "is_alive", "life_state",
    "is_defusing", "is_scoped", "is_walking", "is_airborne", "ducking", "duck_amount",
    # identity / team
    "player_name", "team_num", "player_steamid", "entity_id", "user_id", "rank",
    # economy
    "balance", "current_equip_value", "round_start_equip_value", "cash_spent_this_round",
    "start_balance", "total_cash_spent",
    # weapons
    "active_weapon", "active_weapon_name", "active_weapon_skin", "active_weapon_ammo",
    "active_weapon_original_owner", "item_def_idx", "weapon_paint_seed", "weapon_float",
    "weapon_stickers", "fall_back_paint_kit", "custom_name", "inventory", "inventory_as_ids",
    "total_ammo_left",
    # movement
    "velocity", "velocity_X", "velocity_Y", "velocity_Z", "max_speed", "fall_velo",
    "ducking", "in_crouch", "crouch_state", "stamina", "move_state",
    # flash / effects
    "flash_duration", "flash_max_alpha", "molotov_damage_time", "spotted", "approximate_spotted_by",
    # misc
    "shots_fired", "last_place_name", "ping", "score", "mvps", "kills_total", "deaths_total",
    "assists_total", "alive_time_total", "damage_total", "equipment_value_total",
    "is_bomb_planted", "is_freeze_period", "game_phase", "total_rounds_played",
    "round_win_reason", "round_win_status", "round_start_time", "game_time",
    "is_warmup_period", "team_rounds_total", "team_name", "is_bomb_dropped",
    "aim_punch_angle", "aim_punch_angle_vel", "agent_skin", "glove_paint_id",
    # buttons
    "FORWARD", "BACK", "LEFT", "RIGHT", "FIRE", "RIGHTCLICK", "RELOAD", "USE", "ZOOM", "WALK", "buttons",
    # usercommands
    "usercmd_viewangle_x", "usercmd_viewangle_y", "usercmd_forward_move",
    # known-bogus sanity checks
    "is_planting", "m_iAccount", "m_firePositions",
]

df = p.parse_ticks(RENDER, ticks=TICKS)
got = [c for c in df.columns if c not in ("steamid", "name", "tick")]
missing = [c for c in RENDER if c not in got and c not in ("steamid", "name", "tick")]
print("rows:", len(df))
print("\n=== AVAILABLE (%d) ===" % len(got))
for c in got:
    s = df[c]
    print(f"  {c:32s} {str(s.dtype):22s} nn={int(s.notna().sum()):6d} eg={[repr(v)[:60] for v in s.dropna().head(2).tolist()]}")
print("\n=== NOT AVAILABLE (%d) ===" % len(missing))
print(", ".join(missing))
print("\nfull columns:", list(df.columns))
df.head(20).to_csv("batch_props_head.csv", index=False)
