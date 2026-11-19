import numpy as np
from demoparser2 import DemoParser
p = DemoParser("/Users/wanjungu/Downloads/test.dem")
PROPS = ["velocity","velocity_X","velocity_Y","velocity_Z"]
full = p.parse_ticks(wanted_props=PROPS)
def cmp(label, ticks):
    a = p.parse_ticks(wanted_props=PROPS, ticks=ticks)
    m = full.merge(a, on=["tick","steamid"], suffixes=("_r","_t"))
    n = 0
    for c in PROPS:
        d = np.abs(np.asarray(m[c+"_r"],float) - np.asarray(m[c+"_t"],float))
        n += int(np.nansum(d > 1e-6))
    print(f"{label:44s} rows {len(m):6d}  ndiff {n}")
cmp("[1245]", [1245])
cmp("[1245,1246]", [1245,1246])
cmp("[1245,1247]", [1245,1247])
cmp("[1245,1246,1248]", [1245,1246,1248])
cmp("[1245,1247,1248]", [1245,1247,1248])
cmp("range(1245,1248) (contiguous 3)", list(range(1245,1248)))
cmp("range(1181,1301) contiguous 120", list(range(1181,1301)))
cmp("range(1181,1301,2) strided 60", list(range(1181,1301,2)))
cmp("range(0,101293) all ticks", list(range(0,101293)))
cmp("range(1181,1301) + dup 1245", list(range(1181,1301))+[1245])
