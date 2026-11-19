import numpy as np, pandas as pd
from demoparser2 import DemoParser

p = DemoParser("/Users/wanjungu/Downloads/test.dem")
props = ["CCSPlayerPawn.m_fFlags", "X", "velocity_X", "is_alive", "total_rounds_played"]
df = p.parse_ticks(wanted_props=props)
print("cols", list(df.columns), len(df))
f = df["CCSPlayerPawn.m_fFlags"]
print("dtype", f.dtype, "nan", int(f.isna().sum()))
nan = df[f.isna()]
print("nan ticks", nan["tick"].nunique(), "rows", len(nan))
print("nan rounds", sorted(nan["total_rounds_played"].unique().tolist()))
print("nan alive", nan["is_alive"].value_counts(dropna=False).to_dict())
print("nan X notna", int(nan["X"].notna().sum()), "nan vx notna", int(nan["velocity_X"].notna().sum()))
fi = f.fillna(0).astype(np.int64)
print("bits present", sorted(pd.Series(fi).value_counts().to_dict().items()))
print("ONGROUND", int((fi & 1).sum()), "DUCK", int((fi & 2).sum()>>1), "FROZEN", int((fi & 32).sum()>>5), "CLIENT", int((fi & 128).sum()>>7))
# does the ONGROUND bit agree with hGroundEntity validity?
h = p.parse_ticks(wanted_props=["CCSPlayerPawn.m_hGroundEntity", "X"])
print("hGround dtype", h["CCSPlayerPawn.m_hGroundEntity"].dtype)
print(h["CCSPlayerPawn.m_hGroundEntity"].head().tolist())
