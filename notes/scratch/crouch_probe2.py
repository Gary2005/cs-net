"""Precision pass on the crouch signal: which column is the real one?

  1. m_flDuckAmount vs m_flDuckOffset — exact relation?
  2. eye height = 64.0938 - offset ?
  3. crouch transition duration (docs say 200 ms)
  4. FL_DUCKING vs duck_amount threshold: which to trust
  5. same numbers on a second demo
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

MS = "CCSPlayerPawn.CCSPlayer_MovementServices."
PROPS = [
    MS + "m_flDuckAmount",
    MS + "m_flDuckOffset",
    MS + "m_flDuckSpeed",
    MS + "m_bDucking",
    MS + "m_bDucked",
    MS + "m_bDesiresDuck",
    MS + "m_flStamina",
    MS + "m_nButtonDownMaskPrev",
    "CCSPlayerPawn.m_fFlags",
    "is_alive",
]


def probe(path, ticks=None):
    p = DemoParser(path)
    df = p.parse_ticks(wanted_props=PROPS, ticks=ticks)
    df = df[df["is_alive"].astype(bool)].copy()
    d = pd.to_numeric(df[MS + "m_flDuckAmount"], errors="coerce")
    o = pd.to_numeric(df[MS + "m_flDuckOffset"], errors="coerce")
    flags = pd.to_numeric(df["CCSPlayerPawn.m_fFlags"], errors="coerce").astype("Int64")
    fl = (flags & 0x2) != 0
    btn = pd.to_numeric(df[MS + "m_nButtonDownMaskPrev"], errors="coerce").astype("Int64")
    bd = (btn & 0x4) != 0

    print(f"\n===== {path.split('/')[-1]}  ({len(df)} alive samples) =====")
    print(f"[1] duck_offset / duck_amount ratio (amount>0.05):")
    m = d > 0.05
    ratio = (o[m] / d[m])
    print(f"    ratio mean={ratio.mean():.6f} std={ratio.std():.8f} "
          f"min={ratio.min():.6f} max={ratio.max():.6f}")
    print(f"    -> implied full-duck offset = {ratio.mean():.4f} HU")
    print(f"    offset range : {o.min():.4f} .. {o.max():.4f}")
    print(f"    amount range : {d.min():.4f} .. {d.max():.4f}")

    eye_from_offset = 64.0938 + o            # offset is <= 0
    eye_from_amount = 64.0938 - d * 18.0176
    print(f"[2] eye from offset  : {eye_from_offset.min():.4f} .. {eye_from_offset.max():.4f}")
    print(f"    eye from amount  : {eye_from_amount.min():.4f} .. {eye_from_amount.max():.4f}")
    print(f"    max |diff|       : {(eye_from_offset - eye_from_amount).abs().max():.6f}")

    print(f"[3] FL_DUCKING vs thresholds")
    for thr in (0.0, 0.5, 0.9, 0.99):
        t = d > thr
        print(f"    amount>{thr:<5} {t.sum():7d}   FL_DUCKING {fl.sum():7d}   "
              f"agree {100*(t.values == fl.values).mean():6.2f}%")
    print(f"    FL_DUCKING & amount<0.5 : {int((fl & (d < 0.5)).sum())} "
          f"(flag set but not really crouched)")
    print(f"    !FL_DUCKING & amount>0.5: {int((~fl & (d > 0.5)).sum())} "
          f"(crouched but flag clear)")
    print(f"    button 0x4 held         : {int(bd.sum())}")

    print(f"[4] transition duration (per player, contiguous runs)")
    durs = []
    for sid, g in df.assign(_d=d).groupby("steamid"):
        g = g.sort_values("tick")
        v = g["_d"].to_numpy()
        tk = g["tick"].to_numpy()
        idx = np.where(np.diff(v) > 1e-6)[0]
        starts = idx[v[idx] <= 1e-6]
        for s in starts:
            after = np.where(v[s + 1:] >= 0.999)[0]
            if len(after):
                e = s + 1 + after[0]
                durs.append(tk[e] - tk[s])
    if durs:
        durs = np.array(durs)
        print(f"    stand->full runs: {len(durs)}  median={np.median(durs):.0f} ticks "
              f"= {np.median(durs)/64*1000:.0f} ms   "
              f"p10={np.percentile(durs,10):.0f} p90={np.percentile(durs,90):.0f}")
    return d


probe(sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem")
probe(sys.argv[2] if len(sys.argv) > 2 else "/Users/wanjungu/Downloads/9207151584779083532_0.dem",
      ticks=list(range(20000, 60000, 1)))
