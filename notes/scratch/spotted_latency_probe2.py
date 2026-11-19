"""调研 8：把结论落成可复核的数字 + 具体例子。

跑法: python notes/scratch/spotted_latency_probe2.py <csz>
"""
from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, ".")
from tools.demo_extract import codec  # noqa: E402

P = 33


def main(path):
    d = codec.load(path)
    names = [p.get("name") for p in d.get("players", [])]

    # tickrate 复核
    r0 = d["rounds"][0]
    dt = r0["ticks"][-1] - r0["ticks"][0]
    ds = r0["round_seconds"][-1] - r0["round_seconds"][0]
    print(f"tickrate 复核: round0 {dt} ticks / {ds:.4f} s -> {dt/ds:.4f} tick/s")
    print(f"网格周期 33 ticks = {33/(dt/ds):.4f} s\n")

    rows = []
    already = stale_le = stale_gt = never = 0
    for rnd in d["rounds"]:
        ticks = rnd["ticks"]
        idx_of = {t: i for i, t in enumerate(ticks)}
        n = len(ticks)
        for ev in rnd["events"]["damage"]:
            a, v, t = ev["a"], ev["v"], ev["t"]
            if a < 0 or v < 0 or a == v:
                continue
            i = idx_of.get(t)
            if i is None:
                continue
            sp = rnd["players"][v]["spotted"]
            if a in sp[i]:
                already += 1
                continue
            j = i
            while j < n and a not in sp[j]:
                j += 1
            if j >= n:
                never += 1
                continue
            delay = ticks[j] - t
            if delay <= 32:
                stale_le += 1
                rows.append((rnd["id"], t, names[a], names[v], delay,
                             bool(rnd["players"][v]["alive"][min(n - 1, i + 40)])))
            else:
                stale_gt += 1

    tot = already + stale_le + stale_gt + never
    print(f"伤害事件（attacker/victim 有效）: {tot}")
    print(f"  开枪那一刻 mask 已经是 1              : {already:4d} ({already/tot*100:5.1f}%)")
    print(f"  那一刻还是 0，<=32 tick 内变 1（网格内）: {stale_le:4d} ({stale_le/tot*100:5.1f}%)")
    print(f"  那一刻还是 0，>32 tick 才变 1          : {stale_gt:4d} ({stale_gt/tot*100:5.1f}%)")
    print(f"  整回合再也没变 1                       : {never:4d} ({never/tot*100:5.1f}%)")

    if rows:
        ds = sorted(r[4] for r in rows)
        print(f"\n<=32 tick 那批的延迟分布 n={len(ds)}: "
              f"min={ds[0]} median={ds[len(ds)//2]} max={ds[-1]} "
              f"mean={sum(ds)/len(ds):.2f} tick = {sum(ds)/len(ds)/64*1000:.0f} ms")

    print("\n具体例子（开枪了但数据里还是「没看见」）:")
    for r in rows[:12]:
        alive = "victim 之后还活着" if r[5] else "victim 之后死了"
        print(f"  r{r[0]:<2} tick {r[1]:<7} {r[2]:<9} -> {r[3]:<9} "
              f"延迟 {r[4]:2d} tick ({r[4]/64*1000:4.0f} ms)  [{alive}]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "visualizer/outputs/aurora-vs-vitality-m2-inferno.csz")
