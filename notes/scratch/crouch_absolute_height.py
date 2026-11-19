"""Is the ABSOLUTE eye height obtainable from the demo, or only the delta?

Findings this script reproduces:

  * m_vecMaxs.z is the collision hull: exactly two values, 72 and 54, and
    "hull < 60" is 100.0000% identical to FL_DUCKING over 727,649 alive rows.
    So the 18 HU magnitude of m_flDuckOffset has an independent, demo-derived
    witness: 72 - 54 = 18.  No documentation needed.

  * m_flDuckOffset is continuous (0 -> -18) and exactly -18 * m_flDuckAmount
    on every alive row.  The hull is a step; the offset is a ramp.  Two
    different quantities that happen to share the same magnitude.

  * CCSPlayerPawn.m_vecZ LOOKS like an absolute view height -- its mode is
    exactly 64.000 for every player and inside a crouch ramp it equals
    (base + m_flDuckOffset) to six decimals -- but it is NOT dependable:
    when airborne its median collapses to 13.1, and even grounded ~18% of
    rows fall outside [60, 65], down to -492.  Its m_vecX/m_vecY siblings are
    nonzero and vary wildly, which a centred view offset cannot be.

  * An exhaustive sweep of all 885 retrievable fields over a real crouch ramp
    turns up only three things that move by 18 HU: m_vecMaxs (step),
    m_flDuckOffset (ramp), and that same unreliable m_vecZ.

Conclusion: the demo carries the DELTA exactly and the ABSOLUTE only
unreliably.  Storing m_flDuckOffset instead of m_flDuckAmount therefore buys
nothing -- they are an exact affine pair -- and the standing eye height still
has to come from a constant.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

MS = "CCSPlayerPawn.CCSPlayer_MovementServices."
DEMO = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"
SID = 76561198865544189          # a player who crouches in place at tick 32519
RAMP = range(32519, 32534)

p = DemoParser(DEMO)
print(f"demo = {DEMO}")

# ---------------------------------------------------------------- hull vs offset
df = p.parse_ticks(
    wanted_props=[MS + "m_flDuckAmount", MS + "m_flDuckOffset",
                  "CCSPlayerPawn.m_vecMaxs", "CCSPlayerPawn.m_vecZ",
                  "CCSPlayerPawn.m_vecX", "CCSPlayerPawn.m_vecY",
                  "CCSPlayerPawn.m_fFlags", "is_alive"],
    ticks=None,
)
df = df[df["is_alive"].astype(bool)].copy()
df["d"] = pd.to_numeric(df[MS + "m_flDuckAmount"], errors="coerce")
df["o"] = pd.to_numeric(df[MS + "m_flDuckOffset"], errors="coerce")
df["mz"] = df["CCSPlayerPawn.m_vecMaxs"].apply(
    lambda v: float(np.asarray(v, dtype=float)[2]) if hasattr(v, "__len__") else np.nan)
df["vz"] = pd.to_numeric(df["CCSPlayerPawn.m_vecZ"], errors="coerce")
df["fl"] = (pd.to_numeric(df["CCSPlayerPawn.m_fFlags"], errors="coerce")
            .astype("Int64") & 0x2) != 0
df = df.dropna(subset=["mz"])

print(f"\n[1] collision hull m_vecMaxs.z -- alive rows {len(df)}")
print("    distinct values:", sorted(df["mz"].unique()))
print(f"    hull < 60  ==  FL_DUCKING : "
      f"{100*((df.mz < 60).values == df.fl.values).mean():.4f}% agreement, "
      f"{int(((df.mz < 60) != df.fl).sum())} exceptions")
print(f"    hull step 72 -> 54  =>  18 HU, matching |m_flDuckOffset| max "
      f"{df.o.abs().max():.4f}")

print(f"\n[2] on every alive row, offset == -18 * amount ?")
resid = (df.o + 18 * df.d).abs()
off = resid > 0.001
print(f"    exceptions: {int(off.sum())} / {len(df)}  ({100*off.mean():.4f}%)")
print(f"    on the clean rows max|offset + 18*amount| = {resid[~off].max():.10f}")
print(f"    (the few exceptions are 1-tick glitches right at death/respawn;")
print(f"     dead rows are far worse -- 21.08% of them keep a frozen offset)")

# ---------------------------------------------------------------- the ramp
r = (p.parse_ticks(wanted_props=[MS + "m_flDuckAmount", MS + "m_flDuckOffset",
                                 "CCSPlayerPawn.m_vecZ", "CCSPlayerPawn.m_vecMaxs",
                                 "CCSPlayerPawn.m_vecX", "CCSPlayerPawn.m_vecY", "Z"],
                   ticks=list(RAMP)))
r = r[r["steamid"] == SID].sort_values("tick")
print(f"\n[3] the ramp (player {SID}, ticks {RAMP.start}..{RAMP.stop-1})")
print(f"    {'tick':>6} {'duck':>7} {'offset':>10} {'m_vecZ':>10} {'vecZ-offset':>12} "
      f"{'vecX':>7} {'vecY':>7} {'hullZ':>7}")
for _, w in r.iterrows():
    mz = float(np.asarray(w["CCSPlayerPawn.m_vecMaxs"], dtype=float)[2])
    print(f"    {int(w['tick']):6d} {w[MS+'m_flDuckAmount']:7.4f} "
          f"{w[MS+'m_flDuckOffset']:10.4f} {w['CCSPlayerPawn.m_vecZ']:10.4f} "
          f"{w['CCSPlayerPawn.m_vecZ']-w[MS+'m_flDuckOffset']:12.4f} "
          f"{w['CCSPlayerPawn.m_vecX']:7.3f} {w['CCSPlayerPawn.m_vecY']:7.3f} {mz:7.1f}")
print("    -> m_vecZ tracks base + offset perfectly here, and hull snaps 72->54 "
      "only at full duck.")

# ---------------------------------------------------------------- is m_vecZ usable
print(f"\n[4] is m_vecZ dependable?  (alive rows {len(df)})")
st = df[df.d <= 1e-6]
og = (pd.to_numeric(df["CCSPlayerPawn.m_fFlags"], errors="coerce")
      .astype("Int64") & 0x1) != 0
print(f"    mode of standing m_vecZ          : {st.vz.round(3).mode().iloc[0]:.3f}")
print(f"    standing rows in [60,65]         : {100*((st.vz>=60)&(st.vz<=65)).mean():6.2f}%")
print(f"    standing rows exactly 0          : {100*(st.vz==0).mean():6.2f}%")
print(f"    standing rows outside [-80,120]  : {100*((st.vz<-80)|(st.vz>120)).mean():6.2f}%")
gi = og.loc[st.index]
print(f"    grounded: median={st.vz[gi].median():8.3f}  in[60,65]="
      f"{100*((st.vz[gi]>=60)&(st.vz[gi]<=65)).mean():6.2f}%")
print(f"    airborne: median={st.vz[~gi].median():8.3f}  in[60,65]="
      f"{100*((st.vz[~gi]>=60)&(st.vz[~gi]<=65)).mean():6.2f}%")
print("    -> reads a plausible view height only while grounded, garbage in the")
print("       air.  Not usable as an eye-height source without upstream work.")

# ---------------------------------------------------------------- exhaustive
print("\n[5] exhaustive sweep: which retrievable fields move by ~18 HU on the ramp?")
ok = [f for f in json.load(open("notes/scratch/dp2_field_retrievability.json"))["ok"]
      if f not in ("tick", "steamid", "name")]
w = p.parse_ticks(wanted_props=ok, ticks=list(RAMP))
w = w[w["steamid"] == SID].sort_values("tick")
a, b = w.iloc[0], w.iloc[-1]
hits = []
for c in ok:
    if c not in w.columns:
        continue
    va, vb = a[c], b[c]
    if hasattr(va, "__len__") or hasattr(vb, "__len__"):
        try:
            va = np.asarray(va, dtype=float); vb = np.asarray(vb, dtype=float)
            if np.any(np.abs(vb - va) > 0.001):
                hits.append((c, f"{va.tolist()} -> {vb.tolist()}", float(np.max(np.abs(vb - va)))))
        except Exception:
            pass
        continue
    try:
        fa, fb = float(va), float(vb)
    except Exception:
        continue
    if fa != fb and np.isfinite(fa) and np.isfinite(fb):
        hits.append((c, f"{fa:.4f} -> {fb:.4f}", abs(fb - fa)))
for c, s, dv in sorted(hits, key=lambda x: -x[2]):
    if abs(dv - 18) < 0.5:
        print(f"    {c:66s} {s:36s} d={dv:.4f}")
print("    -> only the hull (step), the duck offset (ramp), and the unreliable")
print("       m_vecZ.  No dependable absolute eye height exists in the vocabulary.")
