import time, numpy as np
from demoparser2 import DemoParser
from demo_extract.extract import PLAYER_PROPS
VEL = ["velocity","velocity_X","velocity_Y","velocity_Z"]
OTHER = [p for p in PLAYER_PROPS if p not in VEL]
p = DemoParser("/Users/wanjungu/Downloads/test.dem")

def t(label, fn):
    a=time.perf_counter(); r=fn(); print(f"{label:52s} {time.perf_counter()-a:7.2f}s rows={len(r)}"); return r

sampled = list(range(1181, 101293, 64))     # ~1565 ticks, worst case for interval
sampled = sorted(set(sampled))
ref = t("REF: all props, ticks=None", lambda: p.parse_ticks(wanted_props=PLAYER_PROPS))
t("A: 28 props, strided 1.5k ticks", lambda: p.parse_ticks(wanted_props=PLAYER_PROPS, ticks=sampled))
t("B: 4 vel props, ticks=None", lambda: p.parse_ticks(wanted_props=VEL))
t("C: 4 vel props, contiguous range(min,max)", lambda: p.parse_ticks(wanted_props=VEL, ticks=list(range(sampled[0], sampled[-1]+1))))
t("D: 28 props, contiguous range(min,max)", lambda: p.parse_ticks(wanted_props=PLAYER_PROPS, ticks=list(range(sampled[0], sampled[-1]+1))))
t("E: 24 other props, strided", lambda: p.parse_ticks(wanted_props=OTHER, ticks=sampled))

# correctness: contiguous range == ticks=None for the 4 vel props
b = p.parse_ticks(wanted_props=VEL)
c = p.parse_ticks(wanted_props=VEL, ticks=list(range(sampled[0], sampled[-1]+1)))
m = b.merge(c, on=["tick","steamid"], suffixes=("_r","_t"))
n = sum(int(np.nansum(np.abs(np.asarray(m[x+"_r"],float)-np.asarray(m[x+"_t"],float))>1e-6)) for x in VEL)
print("contiguous vs ticks=None velocity mismatches:", n, "of", len(m)*4)
