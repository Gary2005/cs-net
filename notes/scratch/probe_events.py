"""Enumerate every event in the demo and dump its schema + row count + a sample row."""
import json, warnings
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)

names = sorted(p.list_game_events())
print("total events:", len(names))
res = {}
for n in names:
    try:
        df = p.parse_event(n)
        res[n] = {
            "n": int(len(df)),
            "cols": list(df.columns),
            "dtypes": {c: str(t) for c, t in zip(df.columns, df.dtypes)},
            "sample": df.head(2).to_dict("records"),
        }
        print(f"{n:32s} n={len(df):6d} cols={list(df.columns)}")
    except BaseException as e:
        res[n] = {"err": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"{n:32s} ERROR {e}")

json.dump(res, open("events_result.json", "w"), indent=1, ensure_ascii=False, default=str)

print("\n\n=========== SAMPLES ===========")
for n in names:
    r = res[n]
    if "err" in r:
        continue
    print(f"\n--- {n} (n={r['n']}) ---")
    print("cols:", r["cols"])
    for row in r["sample"]:
        print("  ", {k: v for k, v in row.items()})
