"""Exhaustive attempt to obtain REAL inferno/fire cell data (molotov footprint grid).

Tests every plausible route in demoparser2 0.42.0 and records exactly what happens.
"""
import json, warnings
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
log = {}

FIRE_PROPS = ["m_firePositions", "m_fireParentPositions", "m_fireCount", "m_nFireCount",
              "m_bFireIsBurning", "m_nInfernoType", "m_nFireLifetime", "m_nFireEffectTickBegin",
              "m_BurnNormal", "m_fireXDelta", "m_fireYDelta", "m_fireZDelta"]

print("=" * 78)
print("A) does the demo actually CONTAIN fire fields? (list_updated_fields)")
fire_fields = [f for f in p.list_updated_fields() if "fire" in f.lower() or "nferno" in f]
print(f"   -> {fire_fields}")

print("\nB) route 1: parse_grenades(extra=FIRE_PROPS)")
base_cols = set(p.parse_grenades().columns)
g = p.parse_grenades(extra=FIRE_PROPS)
log["parse_grenades_extra"] = {"added_cols": [c for c in g.columns if c not in base_cols],
                               "grenade_types": sorted(g.grenade_type.unique().tolist())}
print("   added cols:", log["parse_grenades_extra"]["added_cols"])
print("   grenade_type values:", log["parse_grenades_extra"]["grenade_types"])

print("\nC) route 2: parse_grenades(grenades=False)")
try:
    g2 = p.parse_grenades(grenades=False)
    log["grenades_false"] = {"cols": list(g2.columns), "rows": int(len(g2)),
                             "types": sorted(g2.grenade_type.unique().tolist()) if "grenade_type" in g2.columns else None}
    print("   cols:", log["grenades_false"]["cols"], "rows:", log["grenades_false"]["rows"])
    print("   types:", log["grenades_false"]["types"])
except BaseException as e:
    log["grenades_false"] = {"ERR": str(e)[:200]}
    print("   ERROR:", e)

print("\nD) route 3: parse_grenades(grenades=False, extra=FIRE_PROPS)")
try:
    g3 = p.parse_grenades(grenades=False, extra=FIRE_PROPS)
    log["grenades_false_extra"] = {"added": [c for c in g3.columns if c not in base_cols],
                                   "types": sorted(g3.grenade_type.unique().tolist()) if "grenade_type" in g3.columns else None}
    print("   added:", log["grenades_false_extra"])
except BaseException as e:
    log["grenades_false_extra"] = {"ERR": str(e)[:200]}
    print("   ERROR:", e)

print("\nE) route 4: parse_ticks(FIRE_PROPS)  [is parse_ticks entity-scoped?]")
df = p.parse_ticks(FIRE_PROPS, ticks=[5600, 5617, 5700, 5800])
log["parse_ticks_fire"] = {"cols": [c for c in df.columns if c not in ("tick", "steamid", "name")]}
print("   returned non-player cols:", log["parse_ticks_fire"]["cols"], "  <-- empty means ignored")
print("   actual cols:", list(df.columns))
print(df.head(3).to_string(index=False))

print("\nF) route 5: fire-related EVENTS")
for ev in ["inferno_startburn", "inferno_expire", "inferno_extinguish", "inferno_burn"]:
    try:
        d = p.parse_event(ev)
        log.setdefault("events", {})[ev] = {"n": int(len(d)), "cols": list(d.columns)}
        print(f"   {ev:22s} n={len(d):4d} cols={list(d.columns)}")
    except BaseException as e:
        log.setdefault("events", {})[ev] = {"ERR": f"{type(e).__name__}: {str(e)[:120]}"}
        print(f"   {ev:22s} ERROR {type(e).__name__}: {str(e)[:120]}")

print("\nG) route 6: can we at least see HOW MANY fire cells / the effect origin?")
for prop in ["m_nFireEffectTickBegin", "m_vecExplodeEffectOrigin", "m_nExplodeEffectTickBegin",
             "mb_fireCount", "m_nFireCount ", "m_firePositions[0]"]:
    gg = p.parse_grenades(extra=[prop])
    added = [c for c in gg.columns if c not in base_cols]
    print(f"   {prop:26s} -> {'ADDED ' + str(added) if added else 'DROPPED'}")

print("\nH) other grenade effect props that DO survive (fallback for molotov impact point)")
probe = ["m_vecExplodeEffectOrigin", "m_nExplodeEffectTickBegin", "m_flDetonateTime",
         "m_vInitialPosition", "m_vInitialVelocity", "m_vSmokeDetonationPos", "m_nSmokeEffectTickBegin",
         "m_bDidSmokeEffect", "m_vSmokeColor", "m_nVoxelFrameDataSize", "m_nVoxelUpdate",
         "m_bIsLive", "m_nBounces", "m_flThrowStrength", "m_flCreateTime", "m_bInPostEffectTime",
         "m_nFireLifetime", "m_flSimulationTime", "m_nDeployTick"]
gg = p.parse_grenades(extra=probe)
added = [c for c in gg.columns if c not in base_cols]
for c in added:
    s = gg[c]
    nn = int(s.notna().sum())
    # how many rows are "meaningful" (not 0/False/empty)
    sstr = s.astype(str)
    nz = int((s.notna() & ~sstr.isin(["0", "0.0", "False", "[]", "[0.0, 0.0, 0.0]"])).sum())
    print(f"   {c:28s} {str(s.dtype):10s} nonnull={nn:7d} nonzero={nz:7d} eg={s.dropna().head(2).tolist()}")
log["extra_survived"] = added
print("\n   extra props that SURVIVED:", added)

json.dump(log, open("inferno_result.json", "w"), indent=1, default=str)
