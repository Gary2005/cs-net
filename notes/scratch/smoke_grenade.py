"""Smoke + grenade precise extraction.

Key issues addressed:
  * grenade_entity_id is REUSED across grenades -> must segment into tick-runs
  * smoke detonation pos / start tick / end tick
  * is smoke bloom (growth) recoverable? m_VoxelFrameData? m_nVoxelFrameDataSize?
  * inferno start/end paired CHRONOLOGICALLY (entityid is reused too)
  * unknown event name behaviour, parse_events(), prop_states
"""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser, WantedPropState

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 220)
out = {}

print("=" * 78)
print("1) does m_VoxelFrameData (real smoke volume) come through?")
base = set(p.parse_grenades().columns)
for prop in ["m_VoxelFrameData", "m_nVoxelFrameDataSize", "m_nVoxelUpdate", "m_vSmokeColor",
             "m_bInPostEffectTime", "m_bParticlesSpawned", "m_nSmokeEffectTickBegin",
             "m_bDidSmokeEffect", "m_vSmokeDetonationPos"]:
    g = p.parse_grenades(extra=[prop])
    added = [c for c in g.columns if c not in base]
    if added:
        s = g[added[0]]
        nz = int((s.notna() & ~s.astype(str).isin(["0", "0.0", "False", "[]"])).sum())
        print(f"   {prop:26s} -> OK   dtype={str(s.dtype):10s} nonzero={nz:7d} eg={s.dropna().head(2).tolist()}")
        out.setdefault("voxel", {})[prop] = {"dtype": str(s.dtype), "nonzero": nz,
                                            "eg": [str(v)[:120] for v in s.dropna().head(2).tolist()]}
    else:
        print(f"   {prop:26s} -> DROPPED")
        out.setdefault("voxel", {})[prop] = "DROPPED"

print("\n" + "=" * 78)
print("2) grenade entity-run segmentation (entity ids are reused)")
g = p.parse_grenades(extra=["m_vSmokeDetonationPos", "m_nSmokeEffectTickBegin", "m_bDidSmokeEffect",
                            "m_vecExplodeEffectOrigin", "m_nExplodeEffectTickBegin", "m_flDetonateTime",
                            "m_vInitialPosition", "m_vInitialVelocity", "m_flThrowStrength"])
g = g.sort_values(["grenade_entity_id", "tick"]).reset_index(drop=True)
# a new run starts when the tick gap > 1 OR the thrower changes
g["new_run"] = (g.grenade_entity_id.ne(g.grenade_entity_id.shift()) |
                g.tick.diff().ne(1) |
                g.steamid.ne(g.steamid.shift()))
g["run_id"] = g.new_run.cumsum()
proj = g[g.grenade_type.str.endswith("Projectile")]
runs = proj.groupby("run_id").agg(
    grenade_type=("grenade_type", "first"), entity_id=("grenade_entity_id", "first"),
    tick0=("tick", "min"), tick1=("tick", "max"), n=("tick", "size"),
    steamid=("steamid", "first"), name=("name", "first"),
    det_tick=("m_nSmokeEffectTickBegin", "max"), did_smoke=("m_bDidSmokeEffect", "max"),
    expl_tick=("m_nExplodeEffectTickBegin", "max"),
)
print("   projectile runs:", len(runs))
print("   runs by type:\n", runs.grenade_type.value_counts().to_string())
print("\n   EXAMPLE smoke runs (det_tick = m_nSmokeEffectTickBegin, in TICKS):")
sm = runs[runs.grenade_type == "CSmokeGrenadeProjectile"]
print(sm[["entity_id", "tick0", "tick1", "n", "det_tick", "did_smoke", "name"]].head(8).to_string(index=False))
out["n_proj_runs"] = int(len(runs))

print("\n   cross-check: m_nSmokeEffectTickBegin vs smokegrenade_detonate event tick")
det = p.parse_event("smokegrenade_detonate")
r = sm.dropna(subset=["det_tick"]).copy()
r["det_tick"] = r["det_tick"].astype(int)
j = r.merge(det[["tick", "entityid", "x", "y", "z"]], left_on=["entity_id", "det_tick"],
            right_on=["entityid", "tick"], how="left")
print("   matched rows:", int(j["x"].notna().sum()), "of", len(j))
print(j[["entity_id", "tick0", "tick1", "det_tick", "x", "y", "z"]].head(6).to_string(index=False))

print("\n" + "=" * 78)
print("3) molotov / incendiary: flight run + detonation point (fallback since fire cells are unavailable)")
mol = runs[runs.grenade_type.isin(["CMolotovProjectile", "CFlashbangProjectile",
                                    "CHEGrenadeProjectile", "CDecoyProjectile"])]
print(mol[["grenade_type", "entity_id", "tick0", "tick1", "n", "det_tick", "expl_tick", "name"]].head(10).to_string(index=False))

print("\n" + "=" * 78)
print("4) smoke detonate->expired, paired CHRONOLOGICALLY (entityid reuse makes id-join wrong)")
det = p.parse_event("smokegrenade_detonate").sort_values("tick").reset_index(drop=True)
exp = p.parse_event("smokegrenade_expired").sort_values("tick").reset_index(drop=True)
print("   n detonate:", len(det), "n expired:", len(exp))
k = min(len(det), len(exp))
dd = (exp["tick"].iloc[:k].to_numpy() - det["tick"].iloc[:k].to_numpy())
pos_match = np.allclose(det[["x", "y", "z"]].iloc[:k].to_numpy(),
                        exp[["x", "y", "z"]].iloc[:k].to_numpy(), atol=1.0)
print("   chronological pairing: position match =", pos_match)
print("   delta ticks: median=%.0f min=%d max=%d unique(head)=%s" %
      (np.median(dd), dd.min(), dd.max(), sorted(set(dd.tolist()))[:6]))
print("   => seconds @64tick: median=%.2f" % (np.median(dd) / 64))
out["smoke_delta_ticks_chrono"] = {"median": float(np.median(dd)), "min": int(dd.min()),
                                   "max": int(dd.max()), "position_match": bool(pos_match),
                                   "median_seconds": float(np.median(dd) / 64)}

print("\n5) inferno startburn->expire, chronological pairing")
sb = p.parse_event("inferno_startburn").sort_values("tick").reset_index(drop=True)
ex = p.parse_event("inferno_expire").sort_values("tick").reset_index(drop=True)
k = min(len(sb), len(ex))
di = (ex["tick"].iloc[:k].to_numpy() - sb["tick"].iloc[:k].to_numpy())
pos_match = np.allclose(sb[["x", "y", "z"]].iloc[:k].to_numpy(), ex[["x", "y", "z"]].iloc[:k].to_numpy(), atol=1.0)
print("   n startburn:", len(sb), "n expire:", len(ex), "position match =", pos_match)
vals, cnt = np.unique(di, return_counts=True)
print("   delta ticks histogram (top):", sorted(zip(cnt.tolist(), vals.tolist()), reverse=True)[:6])
print("   => seconds @64tick:", sorted(set((di / 64).round(3).tolist()))[:8])
out["inferno_delta_ticks_chrono"] = {"n": int(k), "position_match": bool(pos_match),
                                     "values": {int(v): int(c) for v, c in zip(vals, cnt)},
                                     "seconds": sorted(set((di / 64).round(3).tolist()))}

print("\n" + "=" * 78)
print("6) event API edge cases")
res = p.parse_event("definitely_not_an_event")
print("   parse_event(unknown) ->", type(res).__name__, repr(res)[:80])
res2 = p.parse_events(["player_death", "bomb_planted"])
print("   parse_events([...]) ->", type(res2).__name__, "len", len(res2), "items:",
      [(n, type(d).__name__, len(d)) for n, d in res2])
try:
    ps = p.parse_ticks(["X"], ticks=[30000, 30001], prop_states=[WantedPropState("is_alive", True)])
    print("   prop_states filter ->", len(ps), "rows (only is_alive==True):")
    print(ps.to_string(index=False))
    out["prop_states"] = {"rows": int(len(ps))}
except BaseException as e:
    print("   prop_states ERROR:", type(e).__name__, e)
    out["prop_states"] = {"ERR": str(e)[:200]}

json.dump(out, open("smoke_result.json", "w"), indent=1, default=str)
