"""Probe 2: measure the real FOV fields, per player, and correlate with weapon.

Probe 1 showed:
  - CCSPlayerController.m_iDesiredFOV parses but is 0 in every one of 1.53M rows
  - the live props are on CCSPlayerPawn.CCSPlayer_CameraServices.*
  - demoparser2 silently DROPS unknown field names instead of raising, so any
    "OK" result with only steamid/name columns is meaningless

Now: what do those fields actually contain, and can we use them for the frustum?
"""

import glob
import json
from collections import Counter, defaultdict

from demoparser2 import DemoParser

DEM = sorted(glob.glob(r"D:\project\cs-net-v5\examples\demo\*.dem"))[1]  # inferno
print("demo:", DEM)
dp = DemoParser(DEM)

FIELDS = [
    "CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOV",
    "CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOVStart",
    "CCSPlayerPawn.CCSPlayer_CameraServices.m_flFOVRate",
    "CCSPlayerPawn.CCSPlayer_CameraServices.m_flFOVTime",
    "CCSPlayerPawn.m_flViewmodelFOV",
    "weapon",
    "is_alive",
]
df = dp.parse_ticks(FIELDS)
print("rows:", f"{len(df):,}")
print("columns:", list(df.columns))

FOV = "CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOV"
FOVS = "CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOVStart"
VM = "CCSPlayerPawn.m_flViewmodelFOV"

for col in (FOV, FOVS, VM):
    if col not in df.columns:
        print(f"!! {col} MISSING from the frame")
        continue
    s = df[col].dropna()
    print(f"\n--- {col}")
    print(f"    non-null {len(s):,} / {len(df):,}   min={s.min()}  max={s.max()}")
    vc = Counter(s.tolist())
    top = sorted(vc.items(), key=lambda kv: -kv[1])[:15]
    print(f"    distinct={len(vc)}  top15={top}")

# ---- correlate m_iFOV with the held weapon
if FOV in df.columns and "weapon" in df.columns:
    print("\n--- m_iFOV by weapon (which weapons ever move it off the default?)")
    per = defaultdict(Counter)
    for w, v in zip(df["weapon"].tolist(), df[FOV].tolist()):
        if v is None:
            continue
        per[w][v] += 1
    # weapon ids are ints; show only weapons with >1 distinct FOV value
    movers = {w: c for w, c in per.items() if len(c) > 1}
    print(f"    weapons seen: {len(per)}   weapons whose FOV varies: {len(movers)}")
    for w, c in sorted(movers.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
        tot = sum(c.values())
        top = sorted(c.items(), key=lambda kv: -kv[1])[:6]
        print(f"      weapon {w:>4}: n={tot:>7,}  {top}")

# ---- per player: does anyone sit at a non-90 FOV for a long stretch?
print("\n--- per-player time at each FOV value (top values)")
if "steamid" in df.columns:
    per_p = defaultdict(Counter)
    for sid, v in zip(df["steamid"].tolist(), df[FOV].tolist()):
        if v is None:
            continue
        per_p[sid][v] += 1
    for sid, c in sorted(per_p.items(), key=lambda kv: -sum(kv[1].values()))[:10]:
        tot = sum(c.values())
        top = sorted(c.items(), key=lambda kv: -kv[1])[:5]
        print(f"    {sid}: n={tot:>7,}  {top}")

# ---- how often does m_iFOV differ from its own most common value (i.e. scoping)?
print("\n--- how transient is it? (fraction of ticks away from that player's mode)")
if "steamid" in df.columns:
    for sid, c in sorted(per_p.items(), key=lambda kv: -sum(kv[1].values()))[:6]:
        tot = sum(c.values())
        mode, mc = c.most_common(1)[0]
        print(f"    {sid}: mode={mode} ({mc:,}/{tot:,} = {mc/tot:6.2%}), "
              f"away {tot-mc:,} ticks")
