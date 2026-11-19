"""验证机制：velocity 是不是 = (上一行位置 - 上上行位置) * 64？

    python notes/scratch/velocity_mechanism.py /path/to/file.dem
"""
import sys
import numpy as np
from demoparser2 import DemoParser

demo = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"
P = DemoParser(demo)
COLS = ["X", "Y", "velocity_X", "velocity_Y", "velocity"]

ref = P.parse_ticks(wanted_props=["X", "Y"])              # 全部 tick，真位置
ref = ref.set_index(["tick", "steamid"]).sort_index()

for stride in (2, 8, 32):
    kept = list(range(1181, 1181 + stride * 200, stride))
    got = P.parse_ticks(wanted_props=COLS, ticks=kept)
    got = got.set_index(["tick", "steamid"]).sort_index()
    rows = []
    for i in range(3, len(kept) - 1):
        for sid in got.loc[kept[i]].index if hasattr(got.loc[kept[i]], "index") else []:
            pass
    # 逐 (kept 下标 i, 玩家) 比两种假设
    h1 = h2 = tot = 0
    h1_max = 0.0
    for i in range(3, len(kept) - 1):
        try:
            cur = got.loc[kept[i]]
            a, b, c = ref.loc[kept[i]], ref.loc[kept[i-1]], ref.loc[kept[i-2]]
        except KeyError:
            continue
        if isinstance(cur, np.ndarray) or getattr(cur, "ndim", 1) > 1:
            for sid in cur.index:
                if sid not in a.index or sid not in b.index or sid not in c.index:
                    continue
                v = float(cur.loc[sid, "velocity_X"])
                if not np.isfinite(v):
                    continue
                tot += 1
                p1 = (float(b.loc[sid, "X"]) - float(c.loc[sid, "X"])) * 64.0
                p2 = (float(a.loc[sid, "X"]) - float(b.loc[sid, "X"])) * 64.0
                h1 += abs(v - p1) < 1e-3
                h2 += abs(v - p2) < 1e-3
                h1_max = max(h1_max, abs(v))
    print(f"stride {stride:3d}: 共 {tot:5d} 个值  "
          f"H1(上两行) 命中 {h1:5d} ({h1/max(tot,1):6.1%})  "
          f"H2(当前+上一行) 命中 {h2:5d} ({h2/max(tot,1):6.1%})  "
          f"max|velocity_X| {h1_max:9.1f}  (真实上限 ≈250×stride = {250*stride})")
