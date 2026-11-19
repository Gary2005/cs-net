"""Does the demo actually carry crouch state, and is it extractable?

Checks, in order:
  1. which crouch-ish netprops come back non-null from parse_ticks
  2. value ranges / distributions (is m_flDuckAmount really continuous 0..1?)
  3. agreement with m_fFlags bit 2 (FL_DUCKING) and button-mask bit 0x4
  4. the strided-tick trap: does a strided tick list return another tick's value?
  5. derived eye height vs the documented 64.09 / 46.08 HU
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"

MS = "CCSPlayerPawn.CCSPlayer_MovementServices."
PROPS = [
    MS + "m_flDuckAmount",
    MS + "m_flDuckOffset",
    MS + "m_flDuckSpeed",
    MS + "m_flLastDuckTime",
    MS + "m_nDuckTimeMsecs",
    MS + "m_nCrouchState",
    MS + "m_bDucking",
    MS + "m_bDucked",
    MS + "m_bInCrouch",
    MS + "m_bDesiresDuck",
    MS + "m_flCrouchTransitionStartTime",
    MS + "m_flStamina",
    MS + "m_nButtonDownMaskPrev",
    "CCSPlayerPawn.m_fFlags",
    "CCSPlayerPawn.m_MoveType",
    "X", "Y", "Z", "health", "is_alive",
]

p = DemoParser(DEMO)
print(f"demo = {DEMO}")
print(f"tickrate = {p.parse_header().get('tickrate')}")

# ---------------------------------------------------------------- 1. presence
TICKS = list(range(30000, 31000))
df = p.parse_ticks(wanted_props=PROPS, ticks=TICKS)
print(f"\n[1] parse_ticks over {len(TICKS)} ticks -> {len(df)} rows")
missing = [c for c in PROPS if c not in df.columns]
print(f"    missing columns : {missing if missing else 'none'}")
short = MS + "m_flDuckAmount"
for c in PROPS:
    if c in df.columns:
        nn = df[c].notna().sum()
        print(f"    {c.split('.')[-1]:32s} dtype={str(df[c].dtype):8s} non-null={nn}/{len(df)}")

# ---------------------------------------------------------------- 2. values
print("\n[2] value ranges")
for c in [MS + "m_flDuckAmount", MS + "m_flDuckOffset", MS + "m_flDuckSpeed",
          MS + "m_nDuckTimeMsecs", MS + "m_nCrouchState", MS + "m_flStamina"]:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        print(f"    {c.split('.')[-1]:32s} min={s.min():10.4f} max={s.max():10.4f} "
              f"nunique={s.nunique():6d} nulls={s.isna().sum()}")

duck = pd.to_numeric(df[short], errors="coerce")
print("\n    m_flDuckAmount histogram (0.1 bins):")
hist, edges = np.histogram(duck.dropna(), bins=np.arange(0, 1.11, 0.1))
for h, e in zip(hist, edges):
    print(f"      {e:4.1f}-{e+0.1:4.1f}  {h:6d}  {'#' * int(60 * h / max(hist.max(), 1))}")

# ---------------------------------------------------------------- 3. agreement
print("\n[3] agreement between the redundant encodings")
flags = pd.to_numeric(df["CCSPlayerPawn.m_fFlags"], errors="coerce")
fl_duck = (flags.astype("Int64") & 0x2) != 0          # FL_DUCKING = bit 2
btn = pd.to_numeric(df[MS + "m_nButtonDownMaskPrev"], errors="coerce")
btn_duck = (btn.astype("Int64") & 0x4) != 0            # IN_DUCK = 0x4
in_crouch = df[MS + "m_bInCrouch"].astype(bool)
b_ducking = df[MS + "m_bDucking"].astype(bool)
b_ducked = df[MS + "m_bDucked"].astype(bool)

def agree(a, b, la, lb):
    a = pd.Series(a).fillna(False).astype(bool)
    b = pd.Series(b).fillna(False).astype(bool)
    print(f"    {la:28s} vs {lb:24s} agree={100*(a==b).mean():6.2f}%  "
          f"({la}={a.sum()}, {lb}={b.sum()})")

agree(fl_duck, in_crouch, "m_fFlags bit2", "m_bInCrouch")
agree(fl_duck, b_ducking, "m_fFlags bit2", "m_bDucking")
agree(fl_duck, b_ducked, "m_fFlags bit2", "m_bDucked")
agree(in_crouch, b_ducking, "m_bInCrouch", "m_bDucking")
agree(btn_duck, b_ducking, "button 0x4 (IN_DUCK)", "m_bDucking")
agree(btn_duck, in_crouch, "button 0x4 (IN_DUCK)", "m_bInCrouch")

print("    duck_amount conditioned on flags:")
print(f"      FL_DUCKING set   : mean={duck[fl_duck.values].mean():.4f} "
      f"min={duck[fl_duck.values].min():.4f} max={duck[fl_duck.values].max():.4f}")
print(f"      FL_DUCKING clear : mean={duck[~fl_duck.values].mean():.4f} "
      f"min={duck[~fl_duck.values].min():.4f} max={duck[~fl_duck.values].max():.4f}")

# ---------------------------------------------------------------- 4. stride trap
print("\n[4] strided-tick trap (the v6 velocity bug): contiguous vs stride=2")
t2 = list(range(30000, 30200))
d1 = p.parse_ticks(wanted_props=[short, "X"], ticks=t2)
d2 = p.parse_ticks(wanted_props=[short, "X"], ticks=t2[::2])
m = d1.merge(d2, on=["tick", "steamid"], suffixes=("_1", "_2"))
if len(m):
    diff = (m[short + "_1"] - m[short + "_2"]).abs()
    bad = (diff > 1e-6).sum()
    print(f"    overlapping (tick,steamid) rows : {len(m)}")
    print(f"    duck_amount mismatches          : {bad} "
          f"({100*bad/max(len(m),1):.2f}%)  max|diff|={diff.max():.6f}")
else:
    print("    no overlap to compare")

# ---------------------------------------------------------------- 5. eye height
print("\n[5] derived eye height (stand 64.0938 -> crouch 46.0762 HU)")
eye = 64.0938 - duck * (64.0938 - 46.0762)
print(f"    eye height at duck=0 : {eye[duck < 0.01].mean():.4f}")
print(f"    eye height at duck=1 : {eye[duck > 0.99].mean():.4f}")

# ---------------------------------------------------------------- 6. how common
print("\n[6] whole-demo crouch prevalence (all ticks, no filter)")
big = p.parse_ticks(wanted_props=[short, "CCSPlayerPawn.m_fFlags", "is_alive"],
                    ticks=None)
big = big[big["is_alive"].astype(bool)]
d = pd.to_numeric(big[short], errors="coerce")
f = (pd.to_numeric(big["CCSPlayerPawn.m_fFlags"], errors="coerce")
     .astype("Int64") & 0x2) != 0
print(f"    alive samples      : {len(big)}")
print(f"    duck_amount > 0.5  : {(d > 0.5).sum()} ({100*(d>0.5).mean():.2f}%)")
print(f"    duck_amount > 0    : {(d > 0).sum()} ({100*(d>0).mean():.2f}%)")
print(f"    FL_DUCKING set     : {f.sum()} ({100*f.mean():.2f}%)")
print(f"    transitions 0->1   : {(d.diff().abs() > 0.01).sum()}")
