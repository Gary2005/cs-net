"""Exactly what can demoparser2 obtain for (A) inferno fire cells and (B) smoke voxels?

Tests all three access paths for every relevant field in the 966-name vocabulary:
  1. parse_ticks(["<namespaced name>"])
  2. parse_grenades(grenades=True, extra=["<leaf name>"])
  3. parse_event("<event>", other=["<name>"])
"""
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"
p = DemoParser(DEMO)
vocab = sorted(p.list_updated_fields())

FIRE = re.compile(r"fire|inferno|burn|flame|molotov|incgrenade|incendiary", re.I)
SMOKE = re.compile(r"voxel|smoke", re.I)

fire_names = [n for n in vocab if FIRE.search(n)]
smoke_names = [n for n in vocab if SMOKE.search(n)]

print("=" * 100)
print(f"A) FIRE-related names in the 966-field vocabulary: {len(fire_names)}")
print("=" * 100)
for n in fire_names:
    print("   ", n)
print()
print(f"B) SMOKE/VOXEL-related names in the vocabulary: {len(smoke_names)}")
print("=" * 100)
for n in smoke_names:
    print("   ", n)

TICKS = list(range(30000, 30120))


def try_ticks(name):
    try:
        df = p.parse_ticks(wanted_props=[name], ticks=TICKS)
    except BaseException as e:
        return "RAISED", str(e).splitlines()[0][:40]
    if name not in df.columns:
        return "dropped", ""
    s = df[name]
    nn = int(s.notna().sum())
    if nn == 0:
        return "col, all-null", ""
    ex = s.dropna().iloc[0]
    return "OK", f"nonnull={nn} {type(ex).__name__} {str(ex)[:44]}"


def try_grenades(leaf):
    try:
        df = p.parse_grenades(grenades=True, extra=[leaf])
    except BaseException as e:
        return "RAISED", str(e).splitlines()[0][:40]
    if leaf not in df.columns:
        return "dropped", ""
    s = df[leaf]
    nn = int(s.notna().sum())
    if nn == 0:
        return "col, all-null", ""
    ex = s.dropna().iloc[0]
    return "OK", f"nonnull={nn} {type(ex).__name__} {str(ex)[:44]}"


print()
print("=" * 100)
print("PER-FIELD REACHABILITY")
print("=" * 100)
print(f"{'leaf name':34s} {'parse_ticks(namespaced)':30s} {'parse_grenades(extra=)':32s}")
for n in fire_names + smoke_names:
    leaf = n.split(".")[-1]
    if not leaf or leaf == n:
        continue
    a_state, a_det = try_ticks(n)
    b_state, b_det = try_grenades(leaf)
    print(f"{leaf:34s} {a_state:30s} {b_state:32s}")
    if a_state == "OK":
        print(f"{'':34s}   -> ticks: {a_det}")
    if b_state == "OK":
        print(f"{'':34s}   -> nades: {b_det}")

# ---------------------------------------------------------------- fire events
print()
print("=" * 100)
print("C) WHAT FIRE INFO *IS* OBTAINABLE — event layer")
print("=" * 100)
for ev in ("inferno_startburn", "inferno_expire"):
    try:
        df = p.parse_event(ev)
        print(f"\nparse_event({ev!r})  rows={len(df)}")
        print("   columns:", list(df.columns))
        if len(df):
            print(df.head(3).to_string(index=False))
    except BaseException as e:
        print(f"\nparse_event({ev!r}) failed: {e}")

print("\n-- can these be added as extra columns to the fire events? --")
for extra in ("m_nInfernoType", "m_fireCount", "m_bFireIsBurning", "Grenade.m_firePositions",
              "m_nFireLifetime", "m_DmgRadius", "m_flDamage", "m_vInitialPosition"):
    try:
        df = p.parse_event("inferno_startburn", other=[extra])
        got = [c for c in df.columns if extra.split(".")[-1] == c or c == extra]
        print(f"   other=[{extra!r}] -> {'RETURNED '+str(got) if got else 'dropped'}"
              f"    (all cols: {list(df.columns)})")
    except BaseException as e:
        print(f"   other=[{extra!r}] -> RAISED {str(e).splitlines()[0][:60]}")

# ------------------------------------------------------------- smoke time series
print()
print("=" * 100)
print("D) SMOKE: what the obtainable per-tick signals actually look like")
print("=" * 100)
g = p.parse_grenades(grenades=False, extra=[
    "m_nVoxelFrameDataSize", "m_nVoxelUpdate", "m_bDidSmokeEffect",
    "m_nSmokeEffectTickBegin", "m_VoxelFrameData", "m_vSmokeColor", "m_vSmokeDetonationPos",
])
smoke = g[g["grenade_type"].astype(str).str.contains("SmokeGrenadeProjectile", na=False)].copy()
print(f"smoke projectile rows: {len(smoke)}  distinct entities: {smoke['grenade_entity_id'].nunique()}")

# pick the entity with the biggest voxel payload
best = smoke.loc[smoke["m_nVoxelFrameDataSize"].idxmax(), "grenade_entity_id"]
one = smoke[smoke["grenade_entity_id"] == best].sort_values("tick")
print(f"\nentity {best}: {len(one)} rows, ticks {one['tick'].min()}..{one['tick'].max()}")
print("   detonation pos:", one["m_vSmokeDetonationPos"].dropna().iloc[0] if len(one) else None)
print("   color         :", one["m_vSmokeColor"].dropna().iloc[0] if len(one) else None)
print("   effectTickBegin:", one["m_nSmokeEffectTickBegin"].dropna().unique()[:4])
print("\n   bloom curve  (tick | voxelSize | voxelUpdate | didSmokeEffect | scalarVoxelFrameData):")
chg = one[(one["m_nVoxelFrameDataSize"] > 0) | (one["m_bDidSmokeEffect"] == True)]
for _, r in chg.head(30).iterrows():
    print(f"      {int(r['tick']):7d} | {r['m_nVoxelFrameDataSize']:9.0f} | "
          f"{r['m_nVoxelUpdate']:11.0f} | {str(r['m_bDidSmokeEffect']):5s} | {r['m_VoxelFrameData']}")

print("\n   distinct (size, scalar) pairs over the whole demo:")
pairs = g[["m_nVoxelFrameDataSize", "m_VoxelFrameData"]].dropna().drop_duplicates()
pairs = pairs[pairs["m_nVoxelFrameDataSize"] > 0].sort_values("m_nVoxelFrameDataSize")
print(pairs.head(30).to_string(index=False))
print(f"   ... {len(pairs)} distinct pairs total")
print("\n   is the scalar constant within one entity?")
for e in list(smoke["grenade_entity_id"].unique())[:6]:
    sub = smoke[smoke["grenade_entity_id"] == e]
    vals = sub["m_VoxelFrameData"].dropna().unique()
    print(f"      entity {e}: size max={sub['m_nVoxelFrameDataSize'].max():.0f} "
          f"scalar distinct={list(vals)[:8]}")