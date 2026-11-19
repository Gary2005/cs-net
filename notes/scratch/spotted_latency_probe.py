"""调研 7：用"伤害事件"直接量延迟 —— 开枪的人一定看见了被打的人，
所以被打者身上的 spotted 位在那一 tick 就该是 1。实测它是不是还在等网格。

跑法: python notes/scratch/spotted_latency_probe.py <csz>
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

    already = 0
    stale = 0
    never = 0
    delays = []
    delays_mod = []
    # 每个回合的网格相位（用该回合所有跳变推）
    for rnd in d["rounds"]:
        ticks = rnd["ticks"]
        idx_of = {t: i for i, t in enumerate(ticks)}
        n = len(ticks)
        # 相位
        ph = Counter()
        for pl in rnd["players"]:
            sp = pl["spotted"]
            for k in range(1, n):
                if frozenset(sp[k]) != frozenset(sp[k - 1]):
                    ph[ticks[k] % P] += 1
        phase = ph.most_common(1)[0][0] if ph else None

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
            # 往后找第一次出现
            j = i
            while j < n and a not in sp[j]:
                j += 1
            if j >= n:
                never += 1
                continue
            stale += 1
            delays.append(ticks[j] - t)
            if phase is not None:
                delays_mod.append((ticks[j] - phase) % P)

    tot = already + stale + never
    print(f"file {path}")
    print(f"伤害事件（attacker/victim 都有效）: {tot}")
    print(f"  伤害那一刻 masked 已经是 1 : {already}  ({already/tot*100:.1f}%)")
    print(f"  伤害那一刻还是 0，之后才变 1: {stale}  ({stale/tot*100:.1f}%)   <-- 陈旧")
    print(f"  整回合都没变 1             : {never}  ({never/tot*100:.1f}%)")

    if delays:
        ds = sorted(delays)
        print(f"\n延迟（tick，从伤害到 mask 变 1）n={len(ds)}")
        print(f"  min={ds[0]} p25={ds[len(ds)//4]} median={ds[len(ds)//2]} "
              f"p75={ds[3*len(ds)//4]} max={ds[-1]}  mean={sum(ds)/len(ds):.2f}")
        print(f"  = {sum(ds)/len(ds)/64:.3f} s 平均，{ds[-1]/64:.3f} s 最大")
        c = Counter(delays)
        print("  直方（tick -> 次数）:")
        for k in sorted(c):
            print(f"     {k:3d} ({k/64:.3f}s) {'#'*c[k]} {c[k]}")

    if delays_mod:
        c = Counter(delays_mod)
        print(f"\n变 1 的 tick 相对网格相位 mod {P}（应该只有一个值）")
        print("  ", dict(sorted(c.items())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "visualizer/outputs/aurora-vs-vitality-m2-inferno.csz")
