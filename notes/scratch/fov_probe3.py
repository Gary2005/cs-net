"""Probe 3: does m_iFOV's non-default value track the held weapon?

If 40 / 10 / 15 are scope zoom levels, then those values must appear almost
exclusively while the player holds a sniper rifle.  That turns "40 is probably
the AWP zoom" from a guess into a measurement.
"""

import glob
from collections import Counter, defaultdict

from demoparser2 import DemoParser

DEM = sorted(glob.glob(r"D:\project\cs-net-v5\examples\demo\*.dem"))[1]
dp = DemoParser(DEM)

FOV = "CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOV"
df = dp.parse_ticks([FOV, "weapon_name", "is_alive"])
print("columns:", list(df.columns))
print("rows:", f"{len(df):,}")

if "weapon_name" not in df.columns:
    print("!! weapon_name missing -- demoparser2 dropped it silently")
    raise SystemExit(1)

# value -> which weapons co-occur
by_val = defaultdict(Counter)
for w, v in zip(df["weapon_name"].tolist(), df[FOV].tolist()):
    if v is None or w is None:
        continue
    by_val[v][str(w)] += 1

for v in sorted(by_val, key=lambda k: -sum(by_val[k].values())):
    c = by_val[v]
    tot = sum(c.values())
    top = c.most_common(6)
    share = ", ".join(f"{w}={n:,}({n/tot:5.1%})" for w, n in top)
    print(f"\n  m_iFOV={v:<3} n={tot:>9,}\n      {share}")

# The decisive cut: while holding a sniper, what is the FOV distribution?
print("\n=== FOV distribution while holding each sniper ===")
SNIPERS = ("awp", "ssg08", "scar20", "g3sg1")
for sn in SNIPERS:
    c = Counter()
    for w, v in zip(df["weapon_name"].tolist(), df[FOV].tolist()):
        if v is None or w is None:
            continue
        if str(w).replace("weapon_", "").lower() == sn:
            c[v] += 1
    if not c:
        print(f"  {sn:<8} (never held)")
        continue
    tot = sum(c.values())
    print(f"  {sn:<8} n={tot:>7,}  " + ", ".join(
        f"{v}:{n:,}({n/tot:5.1%})" for v, n in sorted(c.items(), key=lambda kv: -kv[1])))

# And the converse: while NOT holding a sniper, is 40/10/15 ever seen?
print("\n=== FOV distribution while holding a NON-sniper ===")
c = Counter()
for w, v in zip(df["weapon_name"].tolist(), df[FOV].tolist()):
    if v is None or w is None:
        continue
    if str(w).replace("weapon_", "").lower() not in SNIPERS:
        c[v] += 1
tot = sum(c.values())
print(f"  n={tot:,}  " + ", ".join(
    f"{v}:{n:,}({n/tot:6.3%})" for v, n in sorted(c.items(), key=lambda kv: -kv[1])))

# also: dead players -- does the FOV field go stale like the spotted mask does?
print("\n=== is the FOV field stale on dead players? ===")
c = defaultdict(Counter)
for a, v in zip(df["is_alive"].tolist(), df[FOV].tolist()):
    if v is None:
        continue
    c[bool(a)][v] += 1
for alive in (True, False):
    cc = c[alive]
    tot = sum(cc.values())
    print(f"  alive={alive}: n={tot:,}  " + ", ".join(
        f"{v}:{n:,}({n/tot:6.2%})" for v, n in sorted(cc.items(), key=lambda kv: -kv[1])))
