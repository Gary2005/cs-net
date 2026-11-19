import glob
from collections import defaultdict
from demoparser2 import DemoParser
DEM = sorted(glob.glob(r"D:\project\cs-net-v5\examples\demo\*.dem"))[1]
dp = DemoParser(DEM)
FOV = "CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOV"
df = dp.parse_ticks([FOV]).sort_values(["steamid", "tick"])
print(f"rows {len(df):,}")
# how many CHANGES per player over the whole match (= cost of a sparse column)
tot = 0
for sid, g in df.groupby("steamid"):
    v = g[FOV].tolist()
    ch = sum(1 for i in range(1, len(v)) if v[i] != v[i-1])
    tot += ch
    print(f"  {sid}: {ch:>6,} changes / {len(v):,} ticks")
print(f"TOTAL changes over the match: {tot:,}")
print(f"a naive per-tick column would be {len(df):,} entries")
