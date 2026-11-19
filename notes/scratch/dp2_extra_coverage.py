"""Which of the demoinfocs-reachable extras can demoparser2 also reach?

demoparser2 raises on unknown prop names, so each candidate is tested alone.
Target demo: ~/Downloads/test.dem (de_ancient).
"""
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 220)

TICKS = list(range(4800, 4900))

CANDIDATES = [
    # --- raw sendtable names demoinfocs reads for button state ---
    "m_pMovementServices.m_nButtonDownMaskPrev",
    "m_pMovementServices.m_nToggleButtonDownMask",
    "m_nButtonDownMaskPrev", "buttons", "buttonstate", "m_nButtons",
    # --- demoinfocs reads these for item identity / skins ---
    "m_iItemDefinitionIndex", "m_nFallbackPaintKit", "m_nFallbackSeed",
    "m_flFallbackWear", "m_iItemIDHigh", "m_iItemIDLow", "m_szCustomName",
    "m_iEntityQuality", "m_iEntityLevel",
    # --- spotting / visibility ---
    "m_bSpottedByMask", "m_bSpotted",
    # --- everything else demoinfocs exposes as a first-class accessor ---
    "is_walking", "is_scoped", "is_ducking", "is_airborne", "is_reloading",
    "is_planting", "is_defusing", "is_grabbing_hostage",
    "in_bomb_zone", "in_buy_zone",
    "money", "equipment_value", "equipment_value_round_start",
    "equipment_value_freeze_time_end", "money_spent_total",
    "ping", "score", "kills", "deaths", "assists", "mvps",
    "total_damage", "utility_damage", "crosshair_code", "clan_tag",
    "viewmodel_fov", "viewmodel_offset_x", "color",
    "rank", "rank_type", "competitive_wins",
    "m_angEyeAngles", "m_vecViewOffset", "eye_offset",
    "life_state", "m_lifeState", "is_alive",
    "shots_fired", "m_iShotsFired",
    "velocity", "velocity_X", "velocity_Y", "velocity_Z",
    "velocity_modifier", "m_flVelocityModifier",
    "approximate_spotted_by",
]

print("=" * 78)
print("parse_ticks single-prop coverage probe   demo =", DEMO)
print("=" * 78)
ok, bad = [], []
for c in CANDIDATES:
    try:
        df = p.parse_ticks(wanted_props=[c], ticks=TICKS[:20])
        n = len(df)
        extra = ""
        if c in df.columns and n:
            s = df[c]
            try:
                nn = int(s.notna().sum())
                uniq = s.dropna().unique()[:3]
                extra = f" nonnull={nn} sample={list(uniq)}"
            except Exception:
                extra = " (unprintable)"
        ok.append((c, n, list(df.columns), extra, c in df.columns))
    except BaseException as e:
        bad.append((c, str(e).split("\n")[0][:90]))

print("\n### COLUMN ACTUALLY RETURNED ###")
for c, n, cols, extra, present in ok:
    if present:
        print(f"  YES  {c:52s}{extra}")

print("\n### SILENTLY DROPPED (no error, no column) ###")
for c, n, cols, extra, present in ok:
    if not present:
        print(f"  DROP {c:52s} rows={n} returned_columns={cols}")

print("\n### RAISED ###")
for c, e in bad:
    print(f"  FAIL {c:52s} {e}")

print("\n" + "=" * 78)
print("parse_skins() columns")
print("=" * 78)
try:
    sk = p.parse_skins()
    print("rows:", len(sk))
    print("cols:", list(sk.columns))
    print(sk.head(10).to_string(index=False))
except BaseException as e:
    print("parse_skins failed:", e)

print("\n" + "=" * 78)
print("parse_grenades() bounce-related columns")
print("=" * 78)
try:
    g = p.parse_grenades()
    print("cols:", list(g.columns))
except BaseException as e:
    print("failed:", e)