"""Debug: m_nSmokeEffectTickBegin tick space, event pairing by (entityid,position),
   m_VoxelFrameData content, prop_states semantics.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser, WantedPropState

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)

g = p.parse_grenades(extra=["m_nSmokeEffectTickBegin", "m_bDidSmokeEffect", "m_vSmokeDetonationPos",
                            "m_VoxelFrameData", "m_nVoxelFrameDataSize", "m_nVoxelUpdate",
                            "m_flDetonateTime"])
sm = g[g.grenade_type == "CSmokeGrenadeProjectile"].copy()
det = p.parse_event("smokegrenade_detonate")

print("=== A) for a few smoke projectile entities: raw rows around detonation")
for eid in [88, 89, 91]:
    sub = sm[sm.grenade_entity_id == eid].sort_values("tick")
    print(f"\n--- entity_id={eid}: {len(sub)} rows ticks {sub.tick.min()}..{sub.tick.max()}")
    print("    events with this entityid:", det[det.entityid == eid][["tick", "x", "y", "z"]].to_dict("records"))
    d = sub[sub.m_bDidSmokeEffect.astype(str) == "True"]
    print("    first row with m_bDidSmokeEffect=True:", d.head(1)[["tick", "m_nSmokeEffectTickBegin",
          "m_vSmokeDetonationPos", "m_VoxelFrameData", "m_nVoxelFrameDataSize", "m_nVoxelUpdate"]].to_dict("records"))
    print("    last row:", sub.tail(1)[["tick", "m_bDidSmokeEffect", "m_nVoxelFrameDataSize"]].to_dict("records"))

print("\n=== B) is m_nSmokeEffectTickBegin in the same tick space as parse_ticks 'tick'?")
# find a smoke detonation and compare with the nearest player tick where smoke is dawning
d1 = det.sort_values("tick").iloc[0]
print("first detonate event:", d1.to_dict())
cand = sm[(sm.m_bDidSmokeEffect.astype(str) == "True") & (sm.m_vSmokeDetonationPos.astype(str) != "[0.0, 0.0, 0.0]")]
print("rows with a real m_vSmokeDetonationPos:", len(cand))
print(cand.head(3)[["grenade_entity_id", "tick", "m_nSmokeEffectTickBegin", "m_vSmokeDetonationPos"]].to_string(index=False))
# For each detonate event, find the projectile entity-row whose detonation pos matches
print("\n=== C) pairing event <-> projectile row by position (tolerance 1.0)")
cand2 = cand.copy()
cand2["px"] = cand2.m_vSmokeDetonationPos.apply(lambda v: v[0])
cand2["py"] = cand2.m_vSmokeDetonationPos.apply(lambda v: v[1])
cand2["pz"] = cand2.m_vSmokeDetonationPos.apply(lambda v: v[2])
m = det.merge(cand2, left_on="entityid", right_on="grenade_entity_id", suffixes=("_ev", "_pr"))
m = m[(abs(m.x - m.px) < 1.0) & (abs(m.y - m.py) < 1.0) & (abs(m.z - m.pz) < 1.0)]
m["dt"] = m["m_nSmokeEffectTickBegin"].astype(int) - m["tick_ev"]
print("matched:", len(m))
print(m[["entityid", "tick_ev", "tick_pr", "m_nSmokeEffectTickBegin", "dt"]].head(10).to_string(index=False))
print("unique (m_nSmokeEffectTickBegin - event_tick):", sorted(set(m.dt.tolist()))[:10])

print("\n=== D) event tick vs m_nSmokeEffectTickBegin relationship on the SAME entity")
print("  => a constant offset means two different tick origins")

print("\n=== E) smoke expired pairing by (entityid, position)")
exp = p.parse_event("smokegrenade_expired")
mm = det.merge(exp, on=["entityid", "x", "y", "z"], suffixes=("_det", "_exp"))
mm = mm[mm.tick_det < mm.tick_exp]
mm["dticks"] = mm.tick_exp - mm.tick_det
print("paired:", len(mm), "of det", len(det), "exp", len(exp))
print("dticks value_counts:\n", mm.dticks.value_counts().head(8).to_string())
print("seconds @64:", sorted(set((mm.dticks / 64).round(3).tolist()))[:8])
print("unpaired detonations (round ended first):", len(det) - len(mm))

print("\n=== F) inferno startburn<->expire pairing by (entityid, position)")
sb = p.parse_event("inferno_startburn"); ex = p.parse_event("inferno_expire")
mi = sb.merge(ex, on=["entityid", "x", "y", "z"], suffixes=("_s", "_e"))
mi = mi[mi.tick_s < mi.tick_e]
mi["dticks"] = mi.tick_e - mi.tick_s
print("paired:", len(mi), "of", len(sb), len(ex))
print("dticks value_counts:\n", mi.dticks.value_counts().to_string())
print("seconds @64:", sorted(set((mi.dticks / 64).round(3).tolist())))

print("\n=== G) m_VoxelFrameData / m_nVoxelFrameDataSize actual value distribution")
for c in ["m_VoxelFrameData", "m_nVoxelFrameDataSize", "m_nVoxelUpdate"]:
    s = sm[c].dropna()
    print(f"  {c}: dtype={s.dtype} n={len(s)} unique={s.nunique()} min={s.min() if s.dtype!=object else 'obj'} max={s.max()}")
    print("    top values:", s.value_counts().head(6).to_dict())

print("\n=== H) per-tick smoke volume growth proxy (m_nVoxelFrameDataSize over time, one entity)")
sub = sm[sm.grenade_entity_id == 88].sort_values("tick")
ch = sub[sub.m_nVoxelFrameDataSize.astype(float) != 0]
print("ticks where voxel size != 0:", len(ch), "range", ch.tick.min() if len(ch) else None, ch.tick.max() if len(ch) else None)
print(ch[["tick", "m_nVoxelFrameDataSize", "m_nVoxelUpdate", "m_bDidSmokeEffect"]].head(8).to_string(index=False))
print("...")
print(ch[["tick", "m_nVoxelFrameDataSize", "m_nVoxelUpdate", "m_bDidSmokeEffect"]].tail(4).to_string(index=False))

print("\n=== I) prop_states semantics")
for ticks in ([30000, 30001, 30002], [43000, 43001, 43002]):
    raw = p.parse_ticks(["X", "is_alive"], ticks=ticks)
    print(f" ticks={ticks} raw is_alive:", raw.groupby("tick").is_alive.sum().to_dict())
    for st in (True, False):
        try:
            f = p.parse_ticks(["X"], ticks=ticks, prop_states=[WantedPropState("is_alive", st)])
            print(f"   prop_states is_alive={st} -> rows={len(f)}")
        except BaseException as e:
            print(f"   prop_states is_alive={st} ERR {e}")
    try:
        f = p.parse_ticks(["X"], ticks=ticks, prop_states=[("is_alive", True)])
        print("   tuple form ->", len(f))
    except BaseException as e:
        print("   tuple form ERR", type(e).__name__, str(e)[:120])
    try:
        f = p.parse_ticks(["X"], ticks=ticks, prop_states=[{"prop": "is_alive", "state": True}])
        print("   dict form ->", len(f))
    except BaseException as e:
        print("   dict form ERR", type(e).__name__, str(e)[:120])
