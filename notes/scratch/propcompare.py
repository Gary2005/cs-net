import numpy as np, pandas as pd
from demoparser2 import DemoParser
from demo_extract.extract import PLAYER_PROPS

p = DemoParser("/Users/wanjungu/Downloads/test.dem")
lo, hi = 1181, 1181 + 32*60
dense = p.parse_ticks(wanted_props=PLAYER_PROPS, ticks=list(range(lo, hi)))
print("dense rows", len(dense))
for step in (2, 8, 32, 64):
    st = list(range(lo, hi, step))
    a = p.parse_ticks(wanted_props=PLAYER_PROPS, ticks=st)
    m = dense.merge(a, on=["tick","steamid"], suffixes=("_d","_s"))
    bad = []
    for c in PLAYER_PROPS:
        if c not in dense.columns or c not in a.columns or c in ("tick","steamid"):
            continue
        cd, cs = m[c+"_d"], m[c+"_s"]
        if cd.dtype == object or str(cd.dtype) == "object":
            same = sum(1 for x, y in zip(cd, cs) if (x is None and y is None) or
                       (isinstance(x, list) and isinstance(y, list) and x == y) or
                       (x == y) or (isinstance(x,float) and isinstance(y,float) and x!=x and y!=y))
            n = len(cd) - same
        else:
            n = int((np.abs(np.asarray(cd,float) - np.asarray(cs,float)) > 1e-6).sum())
        if n:
            bad.append(f"{c}:{n}")
    print(f"step {step:3d} rows {len(m):5d}  mismatched columns -> {bad}")
