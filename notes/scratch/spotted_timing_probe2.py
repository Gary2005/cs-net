"""调研 2：spotted 跳变的量化特征 + 几何延迟估计。

跑法: python notes/scratch/spotted_timing_probe2.py <csz>
"""
from __future__ import annotations

import math
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
from tools.demo_extract import codec  # noqa: E402


def collect(path):
    d = codec.load(path)
    rows = []          # (round_id, sample_idx, tick, target, viewer, kind)  kind: +1 rise / -1 fall
    dwell = []         # (kind, ticks_duration)
    gaps = 0
    total = 0
    for rnd in d["rounds"]:
        ticks = rnd["ticks"]
        n = len(ticks)
        P = rnd["players"]
        npl = len(P)
        total += n
        for k in range(1, n):
            if ticks[k] != ticks[k - 1] + 1:
                gaps += 1
        mask = [[frozenset(P[i]["spotted"][t]) for i in range(npl)] for t in range(n)]
        for i in range(npl):
            cur = bool(mask[0][i])
            start = 0
            for t in range(1, n):
                v = bool(mask[t][i])
                if v != cur:
                    for s in (mask[t][i] - mask[t - 1][i]):
                        rows.append((rnd["id"], t, ticks[t], i, s, +1))
                    for s in (mask[t - 1][i] - mask[t][i]):
                        rows.append((rnd["id"], t, ticks[t], i, s, -1))
                    dwell.append((cur, ticks[t] - ticks[start]))
                    cur, start = v, t
            dwell.append((cur, ticks[n - 1] - ticks[start]))
    return d, rows, dwell, gaps, total


def main(path):
    d, rows, dwell, gaps, total = collect(path)
    nrounds = len(d["rounds"])
    names = [p.get("name") for p in d.get("players", [])]
    print(f"file {path}")
    print(f"rounds {nrounds} players {names}")
    print(f"tick gaps inside rounds: {gaps}  (samples {total})  -> ticks are "
          f"{'contiguous' if gaps == 0 else 'NOT contiguous'}")

    print(f"\n== 跳变总数: {len(rows)}  ({len(rows)/nrounds:.1f}/round)")

    print("\n== 停留时长完整直方（按次数排序，前 25 个取值）")
    for kind, label in ((True, "ON "), (False, "OFF")):
        arr = [t for k, t in dwell if k is kind]
        c = Counter(arr)
        print(f"  {label} n={len(arr)}")
        for v, cnt in c.most_common(25):
            print(f"      {v:6d} ticks ({v/64:.3f}s)  x{cnt}")

    print("\n== 量化检验：停留时长能否被某个步长整除（残差集中度）")
    arr = [t for _, t in dwell]
    for step in (2, 4, 8, 10, 11, 16, 21, 32, 33, 64):
        res = Counter(t % step for t in arr)
        top, cnt = res.most_common(1)[0]
        exp = len(arr) / step
        print(f"   step={step:3d}  max residue bucket {cnt}/{len(arr)} "
              f"({cnt/len(arr)*100:5.1f}%  uniform={exp:.1f}  x{cnt/exp:.2f})")

    print("\n== 跳变时刻的 tick % N（按回合内 sample_idx 也算一遍）")
    for key, getter in (("tick", lambda r: r[2]), ("sample_idx", lambda r: r[1])):
        for N in (2, 4, 8, 16, 32):
            c = Counter(getter(r) % N for r in rows)
            top, cnt = c.most_common(1)[0]
            exp = len(rows) / N
            print(f"   {key:10s} N={N:3d}  max {cnt:4d}/{len(rows)} "
                  f"({cnt/len(rows)*100:5.1f}%  uniform={exp:.1f}  x{cnt/exp:.2f})")

    print("\n== 每个回合的跳变数")
    per = Counter(r[0] for r in rows)
    print("   ", dict(sorted(per.items())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "visualizer/outputs/aurora-vs-vitality-m2-inferno.csz")
