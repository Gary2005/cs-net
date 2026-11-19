"""Is the crouch transition continuous tick-by-tick, and is eye height enough?

  1. dump a few real stand->crouch ramps, tick by tick
  2. per-tick increment distribution: linear ramp or eased curve?
  3. does it ever jump (a tick where the value leaps)?
  4. how much of the ramp survives at 4 Hz / 2 Hz sampling?
  5. is eye height a lossless stand-in for duck_amount?
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
# contiguous window that contains real crouching
TICKS = list(range(30000, 40000))
df = p.parse_ticks(
    wanted_props=[MS + "m_flDuckAmount", MS + "m_flDuckOffset",
                  MS + "m_flDuckSpeed", MS + "m_bDucking", "is_alive"],
    ticks=TICKS,
)
df = df[df["is_alive"].astype(bool)].copy()
df["d"] = pd.to_numeric(df[MS + "m_flDuckAmount"], errors="coerce")
df["o"] = pd.to_numeric(df[MS + "m_flDuckOffset"], errors="coerce")
print(f"demo={DEMO.split('/')[-1]}  window={TICKS[0]}..{TICKS[-1]}  rows={len(df)}")

# ---------------------------------------------------------------- 1. dump ramps
print("\n[1] stand -> full-crouch ramps, tick by tick (first 4 found)")
shown = 0
for sid, g in df.groupby("steamid"):
    g = g.sort_values("tick")
    v = g["d"].to_numpy()
    tk = g["tick"].to_numpy()
    o = g["o"].to_numpy()
    sp = pd.to_numeric(g[MS + "m_flDuckSpeed"], errors="coerce").to_numpy()
    # rising edge: was ~0, becomes >0
    idx = np.where((v[1:] > v[:-1] + 1e-9) & (v[:-1] <= 1e-6))[0]
    for s in idx:
        # extend to where it hits 1.0
        end = s
        while end + 1 < len(v) and v[end + 1] > v[end] + 1e-9:
            end += 1
        span = end - s
        if span < 3:
            continue
        print(f"\n    steamid={sid}  ticks {tk[s]}..{tk[end]}  span={span} ticks "
              f"= {span/64*1000:.0f} ms   duckSpeed={sp[s]:.3f}")
        for i in range(s, min(end + 2, len(v))):
            eye = 64.0938 + o[i]
            step = v[i] - v[i - 1] if i > s else 0.0
            print(f"      tick {tk[i]:6d}  amount={v[i]:.6f}  offset={o[i]:9.6f}  "
                  f"eye={eye:9.6f}  d(step)={step:+.6f}")
        shown += 1
        if shown >= 4:
            break
    if shown >= 4:
        break

# ---------------------------------------------------------------- 2. increments
print("\n[2] per-tick increment statistics during rising ramps")
inc, spans, durs = [], [], []
for sid, g in df.groupby("steamid"):
    g = g.sort_values("tick")
    v = g["d"].to_numpy()
    tk = g["tick"].to_numpy()
    # only truly contiguous ticks
    idx = np.where((v[1:] > v[:-1] + 1e-9))[0]
    for s in idx:
        if v[s] <= 1e-6:
            end = s
            while end + 1 < len(v) and v[end + 1] > v[end] + 1e-9:
                end += 1
            if end > s and tk[end] - tk[s] == end - s:
                seg = v[s:end + 1]
                inc.extend(np.diff(seg).tolist())
                spans.append(end - s)
                durs.append(tk[end] - tk[s])
inc = np.array(inc)
spans = np.array(spans)
print(f"    ramps: {len(spans)}   ticks per ramp: median={np.median(spans):.0f} "
      f"min={spans.min()} max={spans.max()}")
print(f"    increments: n={len(inc)} mean={inc.mean():.6f} std={inc.std():.6f} "
      f"min={inc.min():.6f} max={inc.max():.6f}")
print(f"    -> if linear, std/mean should be ~0.  ratio={inc.std()/inc.mean():.4f}")
h, e = np.histogram(inc, bins=8)
for hh, ee in zip(h, e):
    print(f"      {ee:.5f}+  {hh:6d}  {'#'*int(60*hh/max(h.max(),1))}")

# ---------------------------------------------------------------- 3. jumps
print("\n[3] are there ticks where the value leaps (non-smooth)?")
jumps = 0
tot = 0
for sid, g in df.groupby("steamid"):
    g = g.sort_values("tick")
    v = g["d"].to_numpy()
    tk = g["tick"].to_numpy()
    contiguous = np.diff(tk) == 1
    dv = np.abs(np.diff(v))[contiguous]
    tot += len(dv)
    jumps += int((dv > 0.2).sum())
print(f"    contiguous tick pairs: {tot}   |delta| > 0.2 in one tick: {jumps} "
      f"({100*jumps/max(tot,1):.3f}%)")
print(f"    (a full 200 ms ramp at 64 tick = ~14 ticks, so ~0.07/tick if linear)")

# ---------------------------------------------------------------- 4. sampling
print("\n[4] what survives coarse sampling (the ramp is ~200 ms)")
for name, step in (("every tick", 1), ("stride=8 (8 Hz)", 8),
                   ("interval 0.25 (4 Hz)", 16), ("interval 0.5 (2 Hz)", 32)):
    sub = df[df["tick"] % step == 0]
    # a ramp is "seen" if some sample lands strictly between 0 and 1
    seen = 0
    for sid, g in sub.groupby("steamid"):
        v = g.sort_values("tick")["d"].to_numpy()
        seen += int(((v > 0.01) & (v < 0.99)).sum())
    mid = int(((df["d"] > 0.01) & (df["d"] < 0.99)).sum())
    print(f"    {name:22s} samples={len(sub):6d}  mid-transition samples={seen:5d} "
          f"(per-tick reference {mid})")

# ---------------------------------------------------------------- 5. lossless?
print("\n[5] eye height vs duck_amount")
o = df["o"].to_numpy()
d = df["d"].to_numpy()
mask = ~np.isnan(o) & ~np.isnan(d)
resid = (o[mask] + 18.0 * d[mask])
print(f"    residual of offset + 18*amount : max|.|={np.abs(resid).max():.8f}")
print(f"    -> eye height is an exact affine function of amount; storing either")
print(f"       loses nothing.  (eye = 64.0938 - 18*amount)")
