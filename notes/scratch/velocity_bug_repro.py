"""最小复现：demoparser2 的 velocity* 在 tick 列表不连续时读错。

    python notes/scratch/velocity_bug_repro.py /path/to/file.dem
"""
import sys
import numpy as np
from demoparser2 import DemoParser

demo = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"
p = DemoParser(demo)
VEL = ["velocity", "velocity_X", "velocity_Y", "velocity_Z"]
OTHER = ["X", "Z", "velocity", "velocity_X", "velocity_Y", "velocity_Z",
         "CCSPlayerPawn.m_fFlags"]

# 参考：不传 ticks，demoparser2 自己遍历全部 tick
ref = p.parse_ticks(wanted_props=OTHER)
lo, hi = 1181, 1181 + 32 * 60          # 一段真实的回合内时间

print(f"{'tick 列表':<34} {'velocity 错':>12} {'X/Z/flags 错':>14}")
for label, ticks in [
    ("连续 range(1181, 3101)", list(range(lo, hi))),
    ("跳着 range(1181, 3101, 2)", list(range(lo, hi, 2))),
    ("跳着 range(1181, 3101, 8)", list(range(lo, hi, 8))),
    ("跳着 range(1181, 3101, 32)", list(range(lo, hi, 32))),
]:
    got = p.parse_ticks(wanted_props=OTHER, ticks=ticks)
    m = ref.merge(got, on=["tick", "steamid"], suffixes=("_r", "_g"))
    vel = sum(int(np.nansum(np.abs(np.asarray(m[c + "_r"], float)
                                   - np.asarray(m[c + "_g"], float)) > 1e-6))
              for c in VEL)
    oth = sum(int(np.nansum(np.abs(np.asarray(m[c + "_r"], float)
                                   - np.asarray(m[c + "_g"], float)) > 1e-6))
              for c in ("X", "Z", "CCSPlayerPawn.m_fFlags"))
    print(f"{label:<34} {vel:>6} / {len(m)*4:<5} {oth:>8} / {len(m)*3:<5}")

# 只看前几行，看看"错"长什么样
want = list(range(lo, lo + 32 * 4, 32))
a = ref[ref["tick"].isin(want)].sort_values(["tick", "steamid"])
b = p.parse_ticks(wanted_props=OTHER, ticks=want).sort_values(["tick", "steamid"])
j = a.merge(b, on=["tick", "steamid"], suffixes=("_正确", "_错"))
print("\n同一批 tick，两种读法（前 6 行）:")
print(j[["tick", "velocity_Z_正确", "velocity_Z_错",
         "X_正确"]].head(6).to_string(index=False))
