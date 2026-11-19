"""Map every one of demoparser2's 966 fields to retrievable / silently-dropped.

`parse_ticks` raises on a name outside the vocabulary but SILENTLY OMITS the
column for a name that is in the vocabulary yet not extractable. That failure
mode is invisible unless you check `name in df.columns`, so it is checked here
for the entire field list.
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"
OUT = sys.argv[2] if len(sys.argv) > 2 else "notes/scratch/dp2_field_retrievability.json"

p = DemoParser(DEMO)
fields = sorted(p.list_updated_fields())
print(f"scanning {len(fields)} fields ...", flush=True)

TICKS = list(range(30000, 30012))

ok, dropped, raised = [], [], []
t0 = time.perf_counter()
for i, name in enumerate(fields):
    if name in ("tick", "steamid", "name", ""):
        continue
    try:
        df = p.parse_ticks(wanted_props=[name], ticks=TICKS)
        (ok if name in df.columns else dropped).append(name)
    except BaseException as e:
        raised.append([name, str(e).splitlines()[0][:120]])
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(fields)}  ok={len(ok)} dropped={len(dropped)} raised={len(raised)}", flush=True)

dt = time.perf_counter() - t0
print(f"\ndone in {dt:.1f}s")
print(f"  retrievable       : {len(ok)}")
print(f"  silently dropped  : {len(dropped)}")
print(f"  raised            : {len(raised)}")

print("\n### SILENTLY DROPPED (in vocabulary, never returned) ###")
for n in dropped:
    print("   ", n)

print("\n### RAISED ###")
for n, e in raised:
    print(f"    {n}  ->  {e}")

json.dump(
    {"demo": DEMO, "ticks": [TICKS[0], TICKS[-1]], "total": len(fields),
     "ok": ok, "dropped": dropped, "raised": raised},
    open(OUT, "w"), indent=1,
)
print(f"\nwrote {OUT}")