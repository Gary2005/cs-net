import numpy as np, pandas as pd
from demoparser2 import DemoParser
p = DemoParser("/Users/wanjungu/Downloads/test.dem")
df = p.parse_ticks(wanted_props=["velocity","velocity_X","velocity_Y","velocity_Z","X","Y","Z","CCSPlayerPawn.m_fFlags","is_alive"], ticks=list(range(20000,20600)))
print(list(df.columns))
for c in ("velocity","velocity_X","velocity_Y","velocity_Z"):
    s = pd.to_numeric(df[c], errors="coerce")
    print(f"{c:12s} min {s.min():10.2f} med {s.median():10.2f} max {s.max():10.2f} mean|.| {s.abs().mean():10.2f}")
v = np.sqrt(df["velocity_X"]**2+df["velocity_Y"]**2+df["velocity_Z"]**2)
print("speed from components: med", np.median(v), "max", v.max())
d = np.abs(df["velocity"] - v)
print("|velocity| vs speed: frac equal", float((d<1e-3).mean()), "max diff", d.max())
