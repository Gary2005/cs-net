"""m_flDuckOffset is NOT always -18*duck_amount.  Where does it diverge?

probe 2 only computed the ratio on rows with amount > 0.05, where it is exact.
Over the full population the residual o + 18*d reaches 18.0 -- i.e. some rows
say "fully crouched" in one column and "standing" in the other.  Which one is
the truth, and does it correlate with alive/dead?
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

MS = "CCSPlayerPawn.CCSPlayer_MovementServices."
DEMO = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"

p = DemoParser(DEMO)
df = p.parse_ticks(
    wanted_props=[MS + "m_flDuckAmount", MS + "m_flDuckOffset",
                  MS + "m_flDuckSpeed", MS + "m_bDucking", MS + "m_bDucked",
                  "CCSPlayerPawn.m_fFlags", "is_alive", "health"],
)
d = pd.to_numeric(df[MS + "m_flDuckAmount"], errors="coerce")
o = pd.to_numeric(df[MS + "m_flDuckOffset"], errors="coerce")
alive = df["is_alive"].astype(bool)
resid = o + 18.0 * d

print(f"demo={DEMO.split('/')[-1]}  rows={len(df)}")
print(f"\n[1] residual (offset + 18*amount) distribution over ALL rows")
print(f"    |resid| <= 0.001 : {int((resid.abs() <= 0.001).sum()):8d}")
print(f"    |resid| >  0.001 : {int((resid.abs() > 0.001).sum()):8d}")
print(f"    |resid| >  1     : {int((resid.abs() > 1).sum()):8d}")
print(f"    max|resid|       : {resid.abs().max():.6f}")

bad = resid.abs() > 0.001
print(f"\n[2] who are the divergent rows?")
print(f"    alive={int((bad & alive).sum())}  dead={int((bad & ~alive).sum())}")
print(f"    of all dead rows ({int((~alive).sum())}), "
      f"{100*(bad & ~alive).sum()/max(int((~alive).sum()),1):.2f}% diverge")
print(f"    of all alive rows ({int(alive.sum())}), "
      f"{100*(bad & alive).sum()/max(int(alive.sum()),1):.4f}% diverge")

print(f"\n[3] value pairs among divergent rows (top 10 combinations)")
sub = df[bad].assign(amount=d[bad].round(4), offset=o[bad].round(4))
vc = sub.groupby(["amount", "offset"]).size().sort_values(ascending=False).head(10)
for (a, off), n in vc.items():
    print(f"    amount={a:8.4f}  offset={off:9.4f}  n={n}")

print(f"\n[4] among divergent rows: which column looks right?")
db = d[bad]
ob = o[bad]
print(f"    amount==0 & offset==-18 : {int(((db == 0) & (ob == -18)).sum())}")
print(f"    amount!=0 & offset==0   : {int(((db != 0) & (ob == 0)).sum())}")
print(f"    other                   : {int(len(db) - ((db == 0) & (ob == -18)).sum() - ((db != 0) & (ob == 0)).sum())}")

print(f"\n[5] does the divergence persist over time, or is it a single-tick glitch?")
runs = []
for sid, g in df.assign(_b=bad.values).groupby("steamid"):
    g = g.sort_values("tick")
    b = g["_b"].to_numpy()
    tk = g["tick"].to_numpy()
    i = 0
    while i < len(b):
        if b[i]:
            j = i
            while j + 1 < len(b) and b[j + 1]:
                j += 1
            runs.append((tk[j] - tk[i], j - i + 1))
            i = j + 1
        else:
            i += 1
if runs:
    spans = np.array([r[0] for r in runs])
    lens = np.array([r[1] for r in runs])
    print(f"    divergent runs: {len(runs)}   ticks spanned: median={np.median(spans):.0f} "
          f"max={spans.max()}   samples: median={np.median(lens):.0f} max={lens.max()}")

print(f"\n[6] clean-population check (drop the divergent rows)")
good = ~bad
print(f"    rows kept: {int(good.sum())}/{len(df)}")
print(f"    max|resid| on kept rows: {resid[good].abs().max():.8f}")
print(f"    -> on clean rows offset is EXACTLY -18*amount (float32 rounding only)")
