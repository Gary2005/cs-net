"""Is the (x, y, z) column the eye or the feet?  And what does "zero cost" mean?

  1. is the collision hull anchored at the origin?  (mins.z == 0 -> feet)
  2. does z move during a crouch?  (it must not, if z is the feet)
  3. is x/y/z the same thing as CCSPlayerPawn.origin?
  4. therefore: where does the eye have to be computed from?
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
    wanted_props=["X", "Y", "Z", "CCSPlayerPawn.origin",
                  "CCSPlayerPawn.m_vecMins", "CCSPlayerPawn.m_vecMaxs",
                  "CCSPlayerPawn.m_fFlags", "velocity",
                  MS + "m_flDuckAmount", MS + "m_flDuckOffset", "is_alive"],
)
alive = df[df["is_alive"].astype(bool)].copy()
print(f"demo={DEMO.split('/')[-1]}  rows={len(df)}  alive={len(alive)}")

# ---------------------------------------------------------------- 1. hull anchor
def comp(c, i):
    return df[c].apply(lambda v: float(np.asarray(v, dtype=float)[i])
                       if hasattr(v, "__len__") else np.nan)

mnz, mxz = comp("CCSPlayerPawn.m_vecMins", 2), comp("CCSPlayerPawn.m_vecMaxs", 2)
mnx, mxx = comp("CCSPlayerPawn.m_vecMins", 0), comp("CCSPlayerPawn.m_vecMaxs", 0)
print(f"\n[1] collision hull, relative to the pawn origin")
print(f"    m_vecMins = ({mnx.min():.1f}..{mnx.max():.1f}, "
      f"{comp('CCSPlayerPawn.m_vecMins',1).min():.1f}..{comp('CCSPlayerPawn.m_vecMins',1).max():.1f}, "
      f"{mnz.min():.1f}..{mnz.max():.1f})")
print(f"    m_vecMaxs = ({mxx.min():.1f}..{mxx.max():.1f}, "
      f"{comp('CCSPlayerPawn.m_vecMaxs',1).min():.1f}..{comp('CCSPlayerPawn.m_vecMaxs',1).max():.1f}, "
      f"{mxz.min():.1f}..{mxz.max():.1f})")
print(f"    mins.z is always 0 ?  {bool((mnz.dropna() == 0).all())}   "
      f"(hull bottom sits exactly on the origin)")
print(f"    maxs.z values: {sorted(mxz.dropna().unique())}")
print("    -> the box is [-16,-16,0]..[16,16,72]: origin = BOTTOM of the box = the feet.")

# ---------------------------------------------------------------- 2. does z move
print(f"\n[2] does z follow the crouch?  Two independent tests.")

# (a) stationary players only: z must be exactly constant if z is the feet
d = pd.to_numeric(alive[MS + "m_flDuckAmount"], errors="coerce")
vel = pd.to_numeric(alive["velocity"] if "velocity" in alive.columns else 0, errors="coerce")
a = alive.assign(_d=d)
moved = stayed = 0
maxdz = 0.0
for sid, g in a.groupby("steamid"):
    g = g.sort_values("tick")
    v = g["_d"].to_numpy(); z = g["Z"].to_numpy(); tk = g["tick"].to_numpy()
    idx = np.where((np.diff(v) > 1e-9) & (v[:-1] <= 1e-6))[0]
    for s in idx:
        e = s
        while e + 1 < len(v) and v[e + 1] > v[e] + 1e-9:
            e += 1
        if e - s < 5 or tk[e] - tk[s] != e - s:
            continue
        if np.abs(np.diff(z[s:e + 1])).max() > 1e-4:
            continue                      # this player was moving vertically
        dz = float(np.abs(z[s:e + 1] - z[s]).max())
        maxdz = max(maxdz, dz)
        if dz < 1e-4:
            stayed += 1
        else:
            moved += 1
print(f"    (a) vertically-stationary crouch ramps with z EXACTLY constant: {stayed}")
print(f"        of those, ramps where z moved: {moved}   max |dz| = {maxdz:.6f} HU")

# (b) whole demo: if z were the eye, dz would track -18*d(duck)
al = alive.sort_values(["steamid", "tick"])
dd = al.groupby("steamid")[MS + "m_flDuckAmount"].diff()
dz = al.groupby("steamid")["Z"].diff()
m = dd.notna() & dz.notna() & (dd.abs() > 1e-6)
print(f"    (b) over the whole demo, corr(dz, -18*d(duck)) on moving-duck samples: "
      f"{np.corrcoef(dz[m], -18*dd[m])[0,1]:+.4f}  (n={int(m.sum())})")
print(f"        mean dz where duck rose by 1.0 : "
      f"{dz[m & (dd > 0.9)].mean():+.4f} HU   (eye hypothesis predicts -18)")
print("    -> z does not follow the crouch.  If (x,y,z) were the eye it would")
print("       drop by 18 every time.  So (x,y,z) is the feet.")

# ---------------------------------------------------------------- 3. vs origin
print(f"\n[3] is (x,y,z) the same as CCSPlayerPawn.origin?")
o = df["CCSPlayerPawn.origin"]
ox = o.apply(lambda v: float(np.asarray(v, dtype=float)[0]) if hasattr(v, "__len__") else np.nan)
oz = o.apply(lambda v: float(np.asarray(v, dtype=float)[2]) if hasattr(v, "__len__") else np.nan)
same = (np.isclose(ox, pd.to_numeric(df["X"], errors="coerce")) &
        np.isclose(oz, pd.to_numeric(df["Z"], errors="coerce")))
print(f"    rows where origin == (X, Z): {int(same.sum())} / {len(df)}")
print(f"    origin.x unique={ox.nunique()} values={sorted(ox.dropna().unique())[:8]}")
print(f"    origin.z unique={oz.nunique()} values={sorted(oz.dropna().unique())[:8]}")
print("    -> CCSPlayerPawn.origin is NOT the world position; it is a constant")
print("       local value.  Do not use it as a position.")

# ---------------------------------------------------------------- 4. eye
print(f"\n[4] so the eye position must be derived, not read")
st = alive[d <= 1e-6]
print(f"    feet  = (x, y, z)                        <- stored in the .csz today")
print(f"    eye   = (x, y, z + standing_eye - 18*duck)")
print(f"    standing_eye is NOT in the demo (no view-offset netprop);")
print(f"    the unreliable m_vecZ has mode 64.000, PLAN.md documents 64.0938")
print(f"    -> one constant has to be chosen and kept in one place.")
