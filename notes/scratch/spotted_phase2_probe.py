"""调研 5：33-tick 网格的相位到底跟什么绑定？

- 同一回合里，所有玩家的相位一样吗？
- 相位跟回合开始 tick / 玩家 alive 段起点 有关吗？

跑法: python notes/scratch/spotted_phase2_probe.py <csz>
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

    print(f"{'round':>5} {'start_tick':>10} {'start%33':>8}   "
          f"每个玩家的相位（residue:次数）")
    within_round_ok = 0
    within_round_bad = 0
    spawn_offsets = []

    for rnd in d["rounds"]:
        ticks = rnd["ticks"]
        n = len(ticks)
        st = rnd["start_tick"]
        per = {}
        for i, pl in enumerate(rnd["players"]):
            sp = pl["spotted"]
            alive = pl["alive"]
            ch = []
            first_alive_tick = None
            for k in range(1, n):
                if alive[k] and not alive[k - 1] and first_alive_tick is None:
                    first_alive_tick = ticks[k]
                if alive[k] and alive[k - 1] and frozenset(sp[k]) != frozenset(sp[k - 1]):
                    ch.append(ticks[k])
            if ch:
                c = Counter(x % P for x in ch)
                per[names[i]] = c
                if first_alive_tick is not None and ch:
                    spawn_offsets.append((names[i], first_alive_tick % P,
                                          c.most_common(1)[0][0]))
        if not per:
            continue
        allres = set()
        for c in per.values():
            allres |= set(c)
        tag = "SAME" if len(allres) == 1 else "DIFF"
        if len(allres) == 1:
            within_round_ok += 1
        else:
            within_round_bad += 1
        show = "  ".join(f"{k[:4]}={dict(v.most_common(2))}" for k, v in list(per.items())[:5])
        print(f"{rnd['id']:>5} {st:>10} {st % P:>8}  [{tag}] {show}")

    print(f"\n同一回合内相位一致: {within_round_ok}  不一致: {within_round_bad}")

    print("\n相位 vs 该玩家本回合首次 alive 的 tick%33（看相位是不是 spawn 时定的）")
    same = sum(1 for _, s, ph in spawn_offsets if s == ph)
    print(f"   相等 {same}/{len(spawn_offsets)}")
    off = Counter((ph - s) % P for _, s, ph in spawn_offsets)
    print("   (相位 - 首次alive) mod 33 分布:", dict(sorted(off.items())))

    print("\n相位 vs 回合 start_tick%33")
    off2 = []
    for rnd in d["rounds"]:
        st = rnd["start_tick"]
        for i, pl in enumerate(rnd["players"]):
            sp = pl["spotted"]
            ch = [rnd["ticks"][k] for k in range(1, len(rnd["ticks"]))
                  if frozenset(sp[k]) != frozenset(sp[k - 1])]
            if ch:
                off2.append((ch[0] % P - st % P) % P)
    print("   (首次跳变 - start_tick) mod 33 分布:", dict(sorted(Counter(off2).items())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "visualizer/outputs/aurora-vs-vitality-m2-inferno.csz")
