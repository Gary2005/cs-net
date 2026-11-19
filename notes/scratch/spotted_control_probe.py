"""调研 3：对照实验 —— spotted 的 33-tick 量化是"引擎对 spotting 限速"，
还是"demo/采样对某些属性限速"？

做法：在同一段 tick 上同时抓一堆属性，对每个属性算它的变化点分布，
看它是否也被量化到 33 tick 的网格上。

跑法: python notes/scratch/spotted_control_probe.py <dem> <start_tick> <n_ticks>
"""
from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from demoparser2 import DemoParser


def residue_report(label, ticks_of_change, N):
    if not ticks_of_change:
        return
    c = Counter(t % N for t in ticks_of_change)
    top, cnt = c.most_common(1)[0]
    exp = len(ticks_of_change) / N
    print(f"    {label:26s} n={len(ticks_of_change):5d}  "
          f"tick%{N} -> max bucket {cnt:4d} ({cnt/len(ticks_of_change)*100:5.1f}%, "
          f"uniform={exp:.1f}, x{cnt/exp:.2f})  distinct residues={len(c)}/{N}")


def main(demo, start, nticks):
    p = DemoParser(demo)
    hdr = p.parse_header()
    print("header:", {k: hdr[k] for k in list(hdr)[:8]})
    ticks = list(range(start, start + nticks))

    props = [
        "approximate_spotted_by",   # m_bSpottedByMask
        "spotted",                  # m_bSpotted
        "is_defusing",
        "is_scoped",
        "is_walking",
        "is_airborne",
        "has_defuser",
        "health",
        "armor",
        "shots_fired",
        "flash_duration",
        "active_weapon_name",
        "inventory",
        "X",
        "Y",
        "Z",
        "yaw",
        "pitch",
        "is_alive",
        "ducking",
    ]
    df = p.parse_ticks(props, ticks=ticks)
    print("df shape", df.shape, "cols", list(df.columns))
    df = df.sort_values(["steamid", "tick"])

    N = 33
    print(f"\n=== 每个属性的变化点是否落在 {N}-tick 网格上")
    print("    （变化点 = 该玩家该属性的值相对上一 tick 变了）")
    for col in props:
        if col not in df.columns:
            continue
        changes = []
        dwells = []
        for sid, g in df.groupby("steamid"):
            g = g.sort_values("tick")
            v = g[col].tolist()
            tk = g["tick"].tolist()
            if len(v) < 2:
                continue
            if isinstance(v[0], list):
                key = lambda x: tuple(sorted(str(i) for i in (x or [])))
            else:
                key = lambda x: (x != x) and "nan" or x   # NaN -> "nan"
            prev = key(v[0])
            start_i = 0
            for i in range(1, len(v)):
                k = key(v[i])
                if k != prev:
                    changes.append(tk[i])
                    dwells.append(tk[i] - tk[start_i])
                    prev, start_i = k, i
        if not changes:
            print(f"    {col:26s} (no change)")
            continue
        residue_report(col, changes, N)
        dm = sum(1 for d in dwells if d % N == 0)
        print(f"      {'':24s} dwell 是 {N} 的倍数的比例: {dm}/{len(dwells)} "
              f"({dm/len(dwells)*100:.1f}%)")

    print(f"\n=== 换个 N 再看 spotted 的变化点")
    for N2 in (2, 4, 8, 16, 32, 33, 64, 66):
        ch = []
        for sid, g in df.groupby("steamid"):
            g = g.sort_values("tick")
            v = [tuple(sorted(str(i) for i in (x or [])))
                 for x in g["approximate_spotted_by"].tolist()]
            tk = g["tick"].tolist()
            for i in range(1, len(v)):
                if v[i] != v[i - 1]:
                    ch.append(tk[i])
        residue_report(f"N={N2}", ch, N2)

    print("\n=== spotted 变化点的原始 tick（前 60 个）")
    ch = []
    for sid, g in df.groupby("steamid"):
        g = g.sort_values("tick")
        v = [tuple(sorted(str(i) for i in (x or [])))
             for x in g["approximate_spotted_by"].tolist()]
        tk = g["tick"].tolist()
        for i in range(1, len(v)):
            if v[i] != v[i - 1]:
                ch.append(tk[i])
    ch.sort()
    print("   ", ch[:60])
    print("    差值:", [ch[i] - ch[i - 1] for i in range(1, min(60, len(ch)))])


if __name__ == "__main__":
    demo = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1395
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    main(demo, start, n)
