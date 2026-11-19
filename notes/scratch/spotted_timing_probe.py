"""调研：.csz 里的 spotted（m_bSpottedByMask）到底有没有时序延迟？

只读，不改任何东西。跑法：
    python notes/scratch/spotted_timing_probe.py visualizer/outputs/xxx.csz

要回答的问题：
  Q1  mask 的 0->1 / 1->0 跳变，在 tick 上是不是被"量化"到某个固定周期？
      引擎若按固定间隔重算 mask，所有跳变的 tick % N 会挤在同一个余数上。
  Q2  mask 的停留时长（dwell）分布 —— 有没有最短停留时间（去抖）？
  Q3  跳变时刻和"视角是否已经对准"的关系 —— 这就是延迟的直接估计。
"""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")

from tools.demo_extract import codec  # noqa: E402

HU = 0.0254  # 1 Hammer unit = 0.0254 m  (其实这里用 HU 更直观，不换算)


def load_rounds(path):
    d = codec.load(path)
    return d


def bit_matrix(rnd):
    """-> trans[t][i] = frozenset(能看到 i 的敌人编号)。ticks 直接来自 round。"""
    return rnd["players"]


def q1_residue(rnd, maxmod=64):
    """收集所有 mask 位的跳变 tick，看 tick % N 的分布。"""
    ticks = rnd["ticks"]
    n = len(ticks)
    P = rnd["players"]
    npl = len(P)
    # mask[t][i] = frozenset
    mask = [[frozenset(P[i]["spotted"][t]) for i in range(npl)] for t in range(n)]
    rises, falls = [], []
    for i in range(npl):
        for t in range(1, n):
            a, b = mask[t - 1][i], mask[t][i]
            for s in b - a:
                rises.append((ticks[t], i, s))
            for s in a - b:
                falls.append((ticks[t], i, s))
    return rises, falls, mask


def main(path):
    d = load_rounds(path)
    rounds = d["rounds"]
    names = [p.get("name") for p in d.get("players", [])]
    print(f"file: {path}")
    print(f"rounds: {len(rounds)}  players: {names}")

    all_rises, all_falls = [], []
    dwell_on, dwell_off = [], []
    for rnd in rounds:
        rises, falls, mask = q1_residue(rnd)
        all_rises += [(rnd["id"],) + r for r in rises]
        all_falls += [(rnd["id"],) + f for f in falls]
        # dwell
        n = len(rnd["ticks"])
        npl = len(mask[0])
        for i in range(npl):
            cur = bool(mask[0][i])
            start = 0
            for t in range(1, n):
                v = bool(mask[t][i])
                if v != cur:
                    (dwell_on if cur else dwell_off).append(t - start)
                    cur, start = v, t
            (dwell_on if cur else dwell_off).append(n - start)

    print(f"\nQ1  total rising edges {len(all_rises)}   falling edges {len(all_falls)}")
    print(f"    ({len(all_rises)/len(rounds):.1f} rises/round, {len(all_falls)/len(rounds):.1f} falls/round)")

    tickrate = 64.0
    print("\n    tick % N 分布（升沿；若引擎按固定周期重算，会挤在一个余数上）")
    for N in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 64):
        c = Counter(t % N for _, t, _, _ in all_rises)
        top, cnt = c.most_common(1)[0]
        # 均匀分布时期望 = total/N；给出"最高桶 / 期望"的倍数
        exp = len(all_rises) / N
        print(f"      N={N:3d}  max bucket {cnt:5d}/{len(all_rises)} "
              f"({cnt/len(all_rises)*100:5.1f}%, uniform={exp:.1f}, x{cnt/exp:.2f})  "
              f"buckets_used={len(c)}/{N}")

    # 同时看升沿/降沿合并
    both = [t for _, t, _, _ in all_rises] + [t for _, t, _, _ in all_falls]
    print("\n    升沿+降沿合并")
    for N in (4, 8, 16, 32, 64):
        c = Counter(t % N for t in both)
        top, cnt = c.most_common(1)[0]
        exp = len(both) / N
        print(f"      N={N:3d}  max bucket {cnt:5d}/{len(both)} "
              f"({cnt/len(both)*100:5.1f}%, uniform={exp:.1f}, x{cnt/exp:.2f})")

    print(f"\nQ2  dwell 分布（采样点= tick）")
    for label, arr in (("ON ", dwell_on), ("OFF", dwell_off)):
        if not arr:
            continue
        arr_s = sorted(arr)
        c = Counter(arr)
        print(f"    {label}: n={len(arr)} min={arr_s[0]} p25={arr_s[len(arr_s)//4]} "
              f"median={arr_s[len(arr_s)//2]} p75={arr_s[3*len(arr_s)//4]} max={arr_s[-1]}")
        print(f"         最短的 12 个停留时长: {arr_s[:12]}")
        print(f"         时长直方（<=6 tick）: {[c.get(k,0) for k in range(0,7)]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "visualizer/outputs/aurora-vs-vitality-m2-inferno.csz")
