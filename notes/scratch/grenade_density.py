"""Measure demoparser2 `parse_grenades` density on one demo.

Question: how many samples per grenade does demoparser2 give, vs demoinfocs-golang
which only records throw / bounce / destroy keypoints?
"""
import sys
import time

from demoparser2 import DemoParser

demo = sys.argv[1] if len(sys.argv) > 1 else "test.dem"

p = DemoParser(demo)
t0 = time.perf_counter()
g = p.parse_grenades()
dt = time.perf_counter() - t0

print(f"demo={demo}")
print(f"parse_grenades() wall={dt:.2f}s")
print(f"shape={g.shape}")
print(f"columns={list(g.columns)}")
print()

# how is a single grenade identified?
for cand in ("grenade_entity_id", "entity_id", "grenade_type", "steamid", "name"):
    print(f"  has {cand}: {cand in g.columns}")

if "grenade_type" in g.columns:
    print()
    print("grenade_type counts:")
    print(g["grenade_type"].value_counts().to_string())

key = "grenade_entity_id" if "grenade_entity_id" in g.columns else None
if key:
    per = g.groupby(key).size()
    print()
    print(f"distinct {key}: {len(per)}")
    print(f"samples per grenade: min={per.min()} median={int(per.median())} max={per.max()} mean={per.mean():.1f}")
    # sample one grenade with a long life and show tick spacing
    longest = per.idxmax()
    sub = g[g[key] == longest].sort_values("tick")
    d = sub["tick"].diff().dropna()
    print(f"\nlongest grenade {key}={longest}: {len(sub)} samples over ticks {sub['tick'].min()}..{sub['tick'].max()}")
    print(f"  tick delta: min={int(d.min())} median={int(d.median())} max={int(d.max())}")
    print(sub.head(8)[["tick", "grenade_type", "x", "y", "z"]].to_string(index=False))

print()
print(">>> TOTAL ROWS:", len(g))