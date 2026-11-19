"""调研 4：确认 spotted 更新网格的相位（period 33 ticks）是全局固定还是每个 life 重置。

跑法: python notes/scratch/spotted_phase_probe.py <csz>
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from tools.demo_extract import codec  # noqa: E402

P = 33


def main(path):
    d = codec.load(path)
    names = [p.get("name") for p in d.get("players", [])]
    glob = Counter()
    per_life_ok = 0
    per_life_bad = 0
    life_sizes = []
    examples_bad = []
    per_player = defaultdict(Counter)

    for rnd in d["rounds"]:
        ticks = rnd["ticks"]
        n = len(ticks)
        for i, pl in enumerate(rnd["players"]):
            sp = pl["spotted"]
            alive = pl["alive"]
            # 只在该玩家 alive 的连续段里看（死了 pawn 没了，mask 冻结）
            t = 1
            while t < n:
                if not alive[t] or not alive[t - 1]:
                    t += 1
                    continue
                # 找到这一段 alive 的起点
                s = t - 1
                while s > 0 and alive[s - 1]:
                    s -= 1
                e = t
                while e + 1 < n and alive[e + 1]:
                    e += 1
                # [s, e] 是连续 alive 段；收集里面的跳变
                ch = []
                prev = frozenset(sp[s])
                for k in range(s + 1, e + 1):
                    cur = frozenset(sp[k])
                    if cur != prev:
                        ch.append(ticks[k])
                        prev = cur
                if len(ch) >= 2:
                    res = set(c % P for c in ch)
                    life_sizes.append(len(ch))
                    if len(res) == 1:
                        per_life_ok += 1
                    else:
                        per_life_bad += 1
                        if len(examples_bad) < 6:
                            examples_bad.append(
                                (rnd["id"], names[i], ch[:14],
                                 sorted({c % P for c in ch})))
                for c in ch:
                    glob[c % P] += 1
                    per_player[names[i]][c % P] += 1
                t = e + 2

    total = sum(glob.values())
    print(f"file {path}   period P={P} ticks ({P/64:.4f}s)")
    print(f"\n全部跳变 tick % {P} 的分布（n={total}）")
    for r in range(P):
        c = glob.get(r, 0)
        bar = "#" * int(c / max(1, total) * 200)
        print(f"   r={r:2d}  {c:5d}  {c/total*100:6.2f}%  {bar}")

    print(f"\n连续 alive 段内、>=2 次跳变的段：{per_life_ok + per_life_bad}")
    print(f"   相位单一（全部同余）: {per_life_ok}")
    print(f"   相位不单一          : {per_life_bad}")
    for ex in examples_bad:
        print("     bad:", ex)

    print(f"\n每个玩家的 tick % {P} 主相位")
    for nm in names:
        c = per_player[nm]
        if not c:
            continue
        top, cnt = c.most_common(1)[0]
        tot = sum(c.values())
        print(f"   {nm:10s} n={tot:4d}  main r={top:2d} ({cnt}/{tot} = {cnt/tot*100:5.1f}%)"
              f"  residues={len(c)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "visualizer/outputs/aurora-vs-vitality-m2-inferno.csz")
