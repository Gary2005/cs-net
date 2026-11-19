import numpy as np, pandas as pd
from demoparser2 import DemoParser

p = DemoParser("/Users/wanjungu/Downloads/test.dem")
df = p.parse_ticks(wanted_props=[
    "CCSPlayerPawn.m_fFlags", "CCSPlayerPawn.m_hGroundEntity",
    "CCSPlayerPawn.CCSPlayer_MovementServices.m_flFallVelocity",
    "X", "Z", "velocity_Z", "is_alive", "total_rounds_played",
])
print("rows", len(df))
f = df["CCSPlayerPawn.m_fFlags"]
h = df["CCSPlayerPawn.m_hGroundEntity"]
fv = df["CCSPlayerPawn.CCSPlayer_MovementServices.m_flFallVelocity"]
print("hGround uniq sample", pd.Series(h).dropna().unique()[:12])
print("hGround nan", int(h.isna().sum()), "fv nan", int(fv.isna().sum()))
alive = df["is_alive"].fillna(0).astype(bool).to_numpy()
present = f.notna().to_numpy()
print("alive&present", int((alive&present).sum()), "present&~alive", int((present&~alive).sum()))
nanrows = df[f.isna()]
print("nan tick range", int(nanrows["tick"].min()), int(nanrows["tick"].max()))
print("nan rounds", nanrows["total_rounds_played"].value_counts(dropna=False).to_dict())
print("nan is_alive", nanrows["is_alive"].value_counts(dropna=False).to_dict())
print("nan X na", int(nanrows["X"].isna().sum()), "/", len(nanrows))
print("nan steamids", nanrows["steamid"].nunique())
# bit0 vs hGround on rows where pawn present
fi = f.fillna(0).astype(np.int64).to_numpy()
og = (fi & 1).astype(bool)
hv = h.fillna(0).to_numpy()
hvalid = (hv != 0) & (hv != 16777215) & (hv != -1)
m = present & alive
print("agree on alive+present:", float((og[m]==hvalid[m]).mean()), "n", int(m.sum()))
print("og=True hvalid=False", int((og[m]&~hvalid[m]).sum()), "og=False hvalid=True", int((~og[m]&hvalid[m]).sum()))
vz = df["velocity_Z"].to_numpy()
fva = fv.to_numpy()
air = m & ~og
print("airborne n", int(air.sum()), "|vz|>10 frac", float((np.abs(vz[air])>10).mean()))
print("ground |vz|>10 frac", float((np.abs(vz[m&og])>10).mean()))
print("air fv range", np.nanmin(fva[air]), np.nanmax(fva[air]))
print("air fv>0 frac", float(np.nanmean(fva[air]>0)), "air fv<0 frac", float(np.nanmean(fva[air]<0)))
print("ground fv>0 frac", float(np.nanmean(fva[m&og]>0)), "ground fv<0 frac", float(np.nanmean(fva[m&og]<0)))
print("bit16 frac", float(((fi>>16)&1)[m].mean()), "bit29 frac", float(((fi>>29)&1)[m].mean()))
print("duck frac", float(((fi>>1)&1)[m].mean()), "frozen frac", float(((fi>>5)&1)[m].mean()))
