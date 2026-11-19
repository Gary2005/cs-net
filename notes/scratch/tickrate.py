"""Derive tickrate rigorously and characterise tick semantics.

Methods:
 1) regress tick against grenade absolute detonation time (m_flDetonateTime, seconds)
 2) regress tick against the `game_time` player prop
 3) smoke detonate -> expired delta (known 18s? -> gives scale) and inferno 7s lifetime
 4) check whether tick 0 is warmup / pre-game and whether ticks are contiguous
"""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
out = {}

# ---- 1) m_flDetonateTime vs detonate event tick -------------------------------
g = p.parse_grenades(extra=["m_flDetonateTime", "m_nSmokeEffectTickBegin", "m_bDidSmokeEffect",
                            "m_vSmokeDetonationPos"])
g = g[g.grenade_type == "CSmokeGrenadeProjectile"].copy()
g["m_flDetonateTime"] = pd.to_numeric(g["m_flDetonateTime"], errors="coerce")
g["m_nSmokeEffectTickBegin"] = pd.to_numeric(g["m_nSmokeEffectTickBegin"], errors="coerce")
d = g[(g.m_flDetonateTime > 0) & (g.m_nSmokeEffectTickBegin > 0)]
pts = d.groupby(["grenade_entity_id", "m_nSmokeEffectTickBegin"])["m_flDetonateTime"].median().reset_index()
pts = pts.drop_duplicates("m_nSmokeEffectTickBegin")
print("smoke detonation sample points:", len(pts))
if len(pts) > 2:
    A = np.polyfit(pts["m_flDetonateTime"], pts["m_nSmokeEffectTickBegin"], 1)
    print("slope (ticks per second) =", A[0], " intercept =", A[1])
    out["tickrate_from_detonate_time"] = {"slope": float(A[0]), "intercept": float(A[1]),
                                          "n_points": int(len(pts))}

# ---- 2) game_time prop vs tick ------------------------------------------------
TICKS = list(range(30000, 30120))
gt = p.parse_ticks(["game_time"], ticks=TICKS)
gt = gt.drop_duplicates("tick").sort_values("tick")
print("\ngame_time vs tick (first 5):")
print(gt.head(5).to_string(index=False))
if len(gt) > 2:
    B = np.polyfit(gt["tick"], gt["game_time"], 1)
    print("game_time slope (s per tick) =", B[0], "=> tickrate =", 1 / B[0] if B[0] else None)
    out["game_time"] = {"s_per_tick": float(B[0]), "implied_tickrate": float(1 / B[0]),
                        "ticks_per_s": float(1 / B[0]), "n": int(len(gt))}

# ---- 3) smoke / inferno event deltas -----------------------------------------
det = p.parse_event("smokegrenade_detonate")
exp = p.parse_event("smokegrenade_expired")
m = det.merge(exp, on="entityid", suffixes=("_det", "_exp"))
m = m[m.tick_det < m.tick_exp]
m["dticks"] = m.tick_exp - m.tick_det
out["smoke_lifetime_ticks"] = {"median": float(m.dticks.median()), "min": int(m.dticks.min()),
                               "max": int(m.dticks.max()), "n": int(len(m))}
print("\nsmoke detonate->expired ticks:", out["smoke_lifetime_ticks"])

sb = p.parse_event("inferno_startburn")
ex = p.parse_event("inferno_expire")
mi = sb.merge(ex, on="entityid", suffixes=("_s", "_e"))
mi = mi[mi.tick_s < mi.tick_e]
mi["dticks"] = mi.tick_e - mi.tick_s
out["inferno_lifetime_ticks"] = {"median": float(mi.dticks.median()), "min": int(mi.dticks.min()),
                                 "max": int(mi.dticks.max()), "n": int(len(mi)),
                                 "value_counts": {int(k): int(v) for k, v in mi.dticks.value_counts().head(8).items()}}
print("inferno startburn->expire ticks:", out["inferno_lifetime_ticks"])

# ---- 4) tick 0 / warmup / contiguity ----------------------------------------
hd = p.parse_ticks(["game_time", "is_warmup_period", "total_rounds_played", "is_freeze_period"], ticks=list(range(0, 400, 40)))
print("\ntick 0..400 probe:")
print(hd.to_string(index=False))

# round boundaries
pre = p.parse_event("round_prestart")
off = p.parse_event("round_officially_ended")
fz = p.parse_event("round_freeze_end")
out["n_round_prestart"] = int(len(pre))
out["round_prestart_ticks"] = [int(x) for x in pre["tick"].tolist()]
out["round_freeze_end_ticks"] = [int(x) for x in fz["tick"].tolist()]
out["round_officially_ended_ticks"] = [int(x) for x in off["tick"].tolist()]
print("\nround_prestart ticks:", out["round_prestart_ticks"])
print("round_freeze_end ticks:", out["round_freeze_end_ticks"])
print("round_officially_ended ticks:", out["round_officially_ended_ticks"])

# last event tick = rough demo end
allt = []
for n in p.list_game_events():
    try:
        df = p.parse_event(n)
        if "tick" in df.columns and len(df):
            allt.append(int(df["tick"].max()))
    except Exception:
        pass
out["max_event_tick"] = max(allt)
print("\nMAX event tick in demo:", out["max_event_tick"])

json.dump(out, open("tickrate_result.json", "w"), indent=1, default=str)
print("\n", json.dumps(out, indent=1, default=str)[:2000])
