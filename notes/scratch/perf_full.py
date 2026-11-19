"""Measure wall-clock time + memory for full-demo player parse and parse_grenades.

Also records the demo's total tick count / tick range.
"""
import time, resource, json, warnings
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


p = DemoParser(DEMO)
res = {}

PROPS = [
    "X", "Y", "Z", "pitch", "yaw", "health", "armor_value", "has_helmet", "has_defuser",
    "is_alive", "is_defusing", "is_scoped", "is_walking", "is_airborne", "ducking",
    "team_num", "balance", "current_equip_value", "active_weapon_name", "active_weapon_ammo",
    "inventory", "velocity", "velocity_X", "velocity_Y", "velocity_Z", "flash_duration",
    "shots_fired", "last_place_name", "is_bomb_planted", "game_phase", "total_rounds_played",
]

t0 = time.perf_counter()
df = p.parse_ticks(PROPS)          # ticks=None, players=None -> ALL ticks, ALL players
t1 = time.perf_counter()
res["parse_ticks_all"] = {
    "seconds": round(t1 - t0, 3),
    "rows": int(len(df)),
    "n_props": len(PROPS),
    "peak_rss_mb": round(rss_mb(), 1),
    "tick_min": int(df["tick"].min()),
    "tick_max": int(df["tick"].max()),
    "n_players": int(df["steamid"].nunique()),
    "n_ticks": int(df["tick"].nunique()),
}
print("parse_ticks(all ticks, all players, %d props): %.2fs rows=%d ticks=%d..%d players=%d peakRSS=%.0fMB"
      % (len(PROPS), t1 - t0, len(df), res["parse_ticks_all"]["tick_min"],
         res["parse_ticks_all"]["tick_max"], res["parse_ticks_all"]["n_players"], rss_mb()))

# tick continuity -> does the demo tick range have gaps?
t = df["tick"].drop_duplicates().sort_values()
d = t.diff().dropna()
res["tick_gaps"] = {"n_unique_ticks": int(len(t)), "max_gap": int(d.max()),
                    "span": int(t.max() - t.min() + 1),
                    "distinct_gaps": sorted(set(int(x) for x in d.unique()))[:12]}
print("tick continuity:", res["tick_gaps"])

# per-prop cost: a single X/Y/Z parse for the whole match
t0 = time.perf_counter(); d2 = p.parse_ticks(["X", "Y", "Z"]); t1 = time.perf_counter()
res["parse_ticks_xyz_all"] = {"seconds": round(t1 - t0, 3), "rows": int(len(d2)), "peak_rss_mb": round(rss_mb(), 1)}
print("parse_ticks(XYZ all ticks): %.2fs rows=%d" % (t1 - t0, len(d2)))

del df, d2

t0 = time.perf_counter(); g = p.parse_grenades(); t1 = time.perf_counter()
res["parse_grenades"] = {"seconds": round(t1 - t0, 3), "rows": int(len(g)), "peak_rss_mb": round(rss_mb(), 1)}
print("parse_grenades: %.2fs rows=%d peakRSS=%.0fMB" % (t1 - t0, len(g), rss_mb()))

t0 = time.perf_counter()
g2 = p.parse_grenades(extra=["m_vSmokeDetonationPos", "m_vSmokeColor", "m_flDetonateTime",
                             "m_nSmokeEffectTickBegin", "m_bDidSmokeEffect",
                             "m_nVoxelFrameDataSize", "m_nExplodeEffectTickBegin",
                             "m_vecExplodeEffectOrigin", "m_vInitialPosition", "m_vInitialVelocity",
                             "m_flThrowStrength", "m_nBounces", "m_hThrower"])
t1 = time.perf_counter()
res["parse_grenades_extra"] = {"seconds": round(t1 - t0, 3), "rows": int(len(g2)),
                              "cols": list(g2.columns), "peak_rss_mb": round(rss_mb(), 1)}
print("parse_grenades(extra=13): %.2fs cols=%s" % (t1 - t0, list(g2.columns)))

t0 = time.perf_counter(); h = p.parse_header(); t1 = time.perf_counter()
res["parse_header_s"] = round(t1 - t0, 4)
t0 = time.perf_counter(); pi = p.parse_player_info(); t1 = time.perf_counter()
res["parse_player_info_s"] = round(t1 - t0, 4)

json.dump(res, open("perf_result.json", "w"), indent=1, default=str)
print("\n", json.dumps(res, indent=1, default=str))
