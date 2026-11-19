import numpy as np, pandas as pd
from demoparser2 import DemoParser
p = DemoParser("/Users/wanjungu/Downloads/test.dem")
PROPS = ["X","velocity","velocity_X","velocity_Y","velocity_Z","CCSPlayerPawn.m_fFlags"]
lo, hi, step = 1181, 1181 + 32*40, 32
strided = list(range(lo, hi, step))
dense   = list(range(lo, hi))
a = p.parse_ticks(wanted_props=PROPS, ticks=strided)
b = p.parse_ticks(wanted_props=PROPS, ticks=dense)
key = ["tick","steamid"]
m = b.merge(a, on=key, suffixes=("_dense","_strided"))
print("merged rows", len(m))
for c in ("velocity_X","velocity_Y","velocity_Z","X","velocity","CCSPlayerPawn.m_fFlags"):
    d = (m[c+"_dense"] - m[c+"_strided"]).abs()
    print(f"{c:28s} maxdiff {np.nanmax(d):12.4f}  n_diff {int((d>1e-6).sum())} / {len(d)}")
bad = m[(m["velocity_Z_dense"]-m["velocity_Z_strided"]).abs()>1e-6]
print(bad[["tick","steamid","velocity_Z_dense","velocity_Z_strided","X_dense","X_strided"]].head(10).to_string())
# 2D speed identity check on dense data
v2 = np.sqrt(b["velocity_X"]**2 + b["velocity_Y"]**2)
print("dense: |v - sqrt(vx^2+vy^2)| max", np.nanmax(np.abs(b["velocity"]-v2)))
v3 = np.sqrt(b["velocity_X"]**2 + b["velocity_Y"]**2 + b["velocity_Z"]**2)
print("dense: |v - |vxyz|| max", np.nanmax(np.abs(b["velocity"]-v3)))
