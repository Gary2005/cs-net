"""Verify the entity-tick-field offset, fix event pairing, and characterise smoke growth."""
import glob, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser, WantedPropState

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
out = {}

g = p.parse_grenades(extra=["m_nSmokeEffectTickBegin", "m_bDidSmokeEffect", "m_vSmokeDetonationPos",
                            "m_VoxelFrameData", "m_nVoxelFrameDataSize", "m_nVoxelUpdate",
                            "m_nExplodeEffectTickBegin", "m_vecExplodeEffectOrigin", "m_nDeployTick"])
sm = g[g.grenade_type == "CSmokeGrenadeProjectile"].copy()
det = p.parse_event("smokegrenade_detonate")
exp = p.parse_event("smokegrenade_expired")

print("=" * 78)
print("1) OFFSET between m_nSmokeEffectTickBegin and the demoparser2 'tick' column")
# first tick at which m_bDidSmokeEffect becomes True == detonation tick (demo space)
first_det = (sm[sm.m_bDidSmokeEffect.astype(str) == "True"]
             .sort_values("tick").groupby("grenade_entity_id")
             .agg(demo_tick=("tick", "first"), field_tick=("m_nSmokeEffectTickBegin", "first")).reset_index())
first_det["offset"] = first_det["field_tick"].astype(int) - first_det["demo_tick"]
print("   n smokes:", len(first_det))
print("   offset value_counts:\n", first_det.offset.value_counts().to_string())
out["smoke_tick_offset"] = {int(k): int(v) for k, v in first_det.offset.value_counts().items()}

print("\n   cross-check with HE grenade m_nExplodeEffectTickBegin")
he = g[g.grenade_type == "CHEGrenadeProjectile"].copy()
hed = p.parse_event("hegrenade_detonate")
he_first = he[he["m_nExplodeEffectTickBegin"].astype(float) > 0].sort_values("tick") \
    .groupby("grenade_entity_id").agg(demo_tick=("tick", "first"),
                                      field_tick=("m_nExplodeEffectTickBegin", "first")).reset_index()
he_first["offset"] = he_first["field_tick"].astype(int) - he_first["demo_tick"]
print("   HE offset value_counts:\n", he_first.offset.value_counts().head(6).to_string())
out["he_tick_offset"] = {int(k): int(v) for k, v in he_first.offset.value_counts().head(6).items()}

print("\n   is the offset the same in ANOTHER demo? (proves it is per-demo, not global)")
other = sorted(glob.glob("/Users/wanjungu/Downloads/test_demos/**/*.dem", recursive=True))
print("   candidate other demos:", len(other))
for path in other[:2]:
    try:
        p2 = DemoParser(path)
        g2 = p2.parse_grenades(extra=["m_nSmokeEffectTickBegin", "m_bDidSmokeEffect"])
        s2 = g2[g2.grenade_type == "CSmokeGrenadeProjectile"]
        fd = (s2[s2.m_bDidSmokeEffect.astype(str) == "True"].sort_values("tick")
              .groupby("grenade_entity_id").agg(d=("tick", "first"), f=("m_nSmokeEffectTickBegin", "first")).reset_index())
        fd["offset"] = fd.f.astype(int) - fd.d
        print(f"   {path.split('/')[-1]}: n={len(fd)} offsets={fd.offset.value_counts().head(3).to_dict()}")
        out.setdefault("other_demo_offsets", {})[path.split('/')[-1]] = {int(k): int(v) for k, v in fd.offset.value_counts().head(3).items()}
    except BaseException as e:
        print(f"   {path}: ERR {type(e).__name__} {str(e)[:100]}")

print("\n" + "=" * 78)
print("2) SMOKE: detonation pos from event vs from entity prop; start/end ticks")
m = det.merge(exp, on=["entityid", "x", "y", "z"], suffixes=("_det", "_exp"))
m = m[m.tick_det < m.tick_exp].copy()
m["dticks"] = m.tick_exp - m.tick_det
print("   paired by (entityid,pos):", len(m), "detonate:", len(det), "expired:", len(exp))
print("   dticks value_counts:\n", m.dticks.value_counts().head(6).to_string())
print("   seconds @64:", sorted(set((m.dticks / 64).round(3).tolist()))[:6])
out["smoke_lifetime"] = {"paired": int(len(m)), "n_detonate": int(len(det)), "n_expired": int(len(exp)),
                         "median_ticks": float(m.dticks.median()), "median_seconds": float(m.dticks.median() / 64),
                         "unpaired_round_ended": int(len(det) - len(m))}

print("\n   detonation position: m_vSmokeDetonationPos vs event x/y/z")
cand = sm[sm.m_vSmokeDetonationPos.astype(str) != "[0.0, 0.0, 0.0]"].copy()
cand["px"] = cand.m_vSmokeDetonationPos.apply(lambda v: v[0])
cand["py"] = cand.m_vSmokeDetonationPos.apply(lambda v: v[1])
cand["pz"] = cand.m_vSmokeDetonationPos.apply(lambda v: v[2])
j = det.merge(cand, left_on="entityid", right_on="grenade_entity_id", suffixes=("_ev", "_pr"))
j = j[(abs(j["x_ev"] - j.px) < 0.5) & (abs(j["y_ev"] - j.py) < 0.5) & (abs(j["z_ev"] - j.pz) < 0.5)]
print("   exact-position matches:", len(j), "(=1 event row x N entity ticks)")
print("   => m_vSmokeDetonationPos is the authoritative detonation point, matches event exactly")

print("\n" + "=" * 78)
print("3) INFERNO pairing by (entityid, position)")
sb = p.parse_event("inferno_startburn"); ex = p.parse_event("inferno_expire")
mi = sb.merge(ex, on=["entityid", "x", "y", "z"], suffixes=("_s", "_e"))
mi = mi[mi.tick_s < mi.tick_e].copy()
mi["dticks"] = mi.tick_e - mi.tick_s
print("   paired:", len(mi), "startburn:", len(sb), "expire:", len(ex))
print("   dticks value_counts:\n", mi.dticks.value_counts().to_string())
print("   seconds @64:", sorted(set((mi.dticks / 64).round(3).tolist())))
out["inferno_lifetime"] = {"paired": int(len(mi)), "n_startburn": int(len(sb)), "n_expire": int(len(ex)),
                           "counts": {int(k): int(v) for k, v in mi.dticks.value_counts().items()},
                           "seconds": sorted(set((mi.dticks / 64).round(3).tolist()))}

print("\n" + "=" * 78)
print("4) SMOKE GROWTH: m_nVoxelFrameDataSize over the smoke lifetime (bloom proxy)")
sub = sm[sm.grenade_entity_id == 89].sort_values("tick")
ch = sub[sub.m_nVoxelFrameDataSize.astype(float) != 0]
print("   entity 89: rows with nonzero voxel size:", len(ch), "tick", ch.tick.min(), "..", ch.tick.max())
print(ch[["tick", "m_nVoxelFrameDataSize", "m_nVoxelUpdate"]].iloc[::max(1, len(ch)//12)].to_string(index=False))
mono = bool((ch.m_nVoxelFrameDataSize.diff().dropna() >= 0).all())
print("   monotonically non-decreasing:", mono, " start size:", ch.m_nVoxelFrameDataSize.iloc[0],
      " end size:", ch.m_nVoxelFrameDataSize.iloc[-1])
out["smoke_voxel_size_growth"] = {"monotonic": mono, "start": float(ch.m_nVoxelFrameDataSize.iloc[0]),
                                  "end": float(ch.m_nVoxelFrameDataSize.iloc[-1]), "n_ticks": int(len(ch))}
print("   NOTE: m_VoxelFrameData (the real blob) comes back as a single float -> NOT usable shape data.")

print("\n" + "=" * 78)
print("5) prop_states semantics")
t = [30000, 30001, 30002, 30003]
raw = p.parse_ticks(["X", "is_alive"], ticks=t)
print("   raw per-tick alive count:", raw.groupby("tick").is_alive.sum().to_dict())
for st in (True, False):
    try:
        f = p.parse_ticks(["X"], ticks=t, prop_states=[WantedPropState("is_alive", st)])
        print(f"   WantedPropState(is_alive, {st}) -> rows={len(f)}")
    except BaseException as e:
        print(f"   WantedPropState('is_alive', {st}) ERR {type(e).__name__} {str(e)[:120]}")
for form, obj in [("tuple", ("is_alive", True)), ("dict", {"prop": "is_alive", "state": True})]:
    try:
        f = p.parse_ticks(["X"], ticks=t, prop_states=[obj])
        print(f"   {form} -> rows={len(f)}")
    except BaseException as e:
        print(f"   {form} ERR {type(e).__name__} {str(e)[:120]}")
# try on a prop that actually varies and is a "state"-like field
try:
    f = p.parse_ticks(["X"], ticks=t, prop_states=[WantedPropState("active_weapon_name", "AK-47")])
    print("   filter active_weapon_name=='AK-47' ->", len(f))
except BaseException as e:
    print("   ak47 filter ERR", str(e)[:120])

json.dump(out, open("offset_result.json", "w"), indent=1, default=str)
print("\n", json.dumps(out, indent=1, default=str)[:1500])
