"""Probe: is the player's FOV available in a CS2 demo via demoparser2?

Question being answered: the visualizer draws every player's view frustum with a
hardcoded 90 deg horizontal FOV.  Is that number extractable from the demo
instead of invented?

Three angles of attack:
  1. what does demoparser2 expose at all (field lists)?
  2. does the raw demo bytes contain an FOV-ish prop name?
  3. if it parses, what does it actually look like over a match?
"""

import re
import sys
import glob
import json
import struct
from collections import Counter

DEM = sorted(glob.glob(r"D:\project\cs-net-v5\examples\demo\*.dem"))
print("demos:")
for d in DEM:
    print("  ", d)
if not DEM:
    sys.exit("no demos")

import demoparser2
print("\ndemoparser2 version:", getattr(demoparser2, "__version__", "?"))

from demoparser2 import DemoParser

# ---------------------------------------------------------------- 1. raw bytes
# The prop name lives in the demo's string table / class baselines as plain
# ASCII.  If it is not there at all, no parser can invent it.
print("\n=== 1. raw byte scan for FOV-ish names ===")
p = DEM[0]
with open(p, "rb") as fh:
    blob = fh.read()
print(f"file size: {len(blob):,} bytes")
for pat in (rb"m_iDesiredFOV", rb"DesiredFOV", rb"m_iFOV", rb"m_flFOV",
            rb"m_bIsScoped", rb"m_bResumeZoom", rb"m_iFOVStart"):
    hits = [m.start() for m in re.finditer(re.escape(pat), blob)]
    print(f"  {pat.decode():<16} {len(hits):>6} hits"
          + (f"   first at {hits[0]:#x}" if hits else ""))

# ------------------------------------------------------- 2. what demoparser2 has
print("\n=== 2. demoparser2 field surface ===")
dp = DemoParser(p)
for meth in ("list_updated_fields", "list_game_events", "list_props",
             "list_entities", "list_entity_names"):
    fn = getattr(dp, meth, None)
    print(f"  {meth:<22} {'yes' if fn else 'NO'}")

try:
    upd = dp.list_updated_fields()
    print(f"  list_updated_fields -> {type(upd).__name__} len={len(upd)}")
    if isinstance(upd, dict):
        fovish = {k: v for k, v in upd.items() if "fov" in k.lower()}
        print("    FOV-ish:", json.dumps(fovish)[:800])
    else:
        fovish = [x for x in upd if "fov" in str(x).lower()]
        print("    FOV-ish:", fovish[:40])
except Exception as e:
    print("  list_updated_fields failed:", type(e).__name__, e)

try:
    names = dp.list_entity_names()
    print(f"  list_entity_names -> {type(names).__name__} len={len(names)}")
    print("    sample:", list(names)[:25] if names is not None else None)
except Exception as e:
    print("  list_entity_names failed:", type(e).__name__, e)

# ------------------------------------------------------------- 3. try to parse
print("\n=== 3. try parsing candidate fields ===")
CANDIDATES = [
    "m_iDesiredFOV", "m_iFOV", "m_flFOV", "m_bIsScoped",
    "CCSPlayerController.m_iDesiredFOV", "CCSPlayerPawn.m_iDesiredFOV",
]
for c in CANDIDATES:
    try:
        df = dp.parse_ticks([c])
        cols = [x for x in df.columns if x != "tick"]
        nonnull = {col: int(df[col].notna().sum()) for col in cols}
        print(f"  OK   {c:<38} rows={len(df):,} cols={cols} nonnull={nonnull}")
        if cols:
            vc = Counter(df[cols[0]].dropna().tolist())
            top = sorted(vc.items(), key=lambda kv: -kv[1])[:12]
            print(f"        value counts (top 12): {top}")
    except Exception as e:
        print(f"  FAIL {c:<38} {type(e).__name__}: {str(e)[:140]}")
