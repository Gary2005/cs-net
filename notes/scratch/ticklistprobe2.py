import numpy as np
from demoparser2 import DemoParser
p = DemoParser("/Users/wanjungu/Downloads/test.dem")
PROPS = ["velocity","velocity_X","velocity_Y","velocity_Z"]
full = p.parse_ticks(wanted_props=PROPS)          # every tick, the reference
full = full.set_index(["tick","steamid"]).sort_index()
def cmp(label, ticks):
    a = p.parse_ticks(wanted_props=PROPS, ticks=ticks).set_index(["tick","steamid"])
    j = full.join(a, rsuffix="_x", how="inner")
    d = np.nanmax([np.abs(np.asarray(j[c],float) - np.asarray(j[c+"_x"],float)).max() for c in PROPS])
    n = sum(int((np.abs(np.asarray(j[c],float) - np.asarray(j[c+"_x"],float)) > 1e-6).sum()) for c in PROPS)
    print(f"{label:42s} rows {len(j):5d}  maxdiff {d:11.3f}  ndiff {n}")
cmp("contiguous 1181..1300 (120 ticks)", list(range(1181,1301)))
cmp("contiguous 1200..1320", list(range(1200,1320)))
cmp("strided 1181..1300 step 2 (60)", list(range(1181,1301,2)))
cmp("strided 1181..1420 step 4 (60)", list(range(1181,1421,4)))
cmp("single tick 1245", [1245])
cmp("two adjacent ticks 1245,1246", [1245,1246])
cmp("two ticks 1245,1247", [1245,1247])
cmp("contiguous 1245,1246,1247", [1245,1246,1247])
cmp("contiguous reversed 1247,1246,1245", [1247,1246,1245])
cmp("contiguous 1181..1300 + far 5000", list(range(1181,1301))+[5000])
