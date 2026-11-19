"""探针：把「队伍 vs 火焰类型」对不上的那几个案例挖出来看清楚。

`molotov_team_probe.py` 发现 139 个火焰里 4 个不吻合。两种可能：

1. **配对错了** —— 两颗燃烧弹落点很近时，最近邻挑到了另一颗；
2. **真的不吻合** —— CS2 里可以捡起地上的投掷物再丢，所以 CT 丢出莫洛托夫
   是可能的。

这两种结论对代码的影响完全相反（前者可以放心用队伍区分，后者不行），
所以必须逐条看清楚，不能靠比例猜。

    python notes/scratch/molotov_exception_probe.py [base_url]
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5099"
MATCH_TICKS = 64
MATCH_HU = 200.0


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def d3(g, o) -> float:
    return ((g["x"] - o[0]) ** 2 + (g["y"] - o[1]) ** 2 + (g["z"] - o[2]) ** 2) ** 0.5


def main() -> int:
    s = get("/api/session")
    if not s.get("loaded"):
        print("服务器上没有加载 demo")
        return 1

    for idx in range(len(s.get("rounds") or [])):
        r = get(f"/api/round/{idx}")
        teams = r.get("teams") or []
        fires = r.get("fire") or []
        nades = (r.get("events") or {}).get("grenades") or []
        molotovs = [g for g in nades if g.get("ty") == "CMolotovProjectile"]
        if not molotovs or not fires:
            continue

        for f in fires:
            origin = f.get("origin")
            if not origin:
                continue
            ft, ftick = f.get("type_name"), int(f.get("first_tick"))
            cands = []
            for g in molotovs:
                dt = abs(int(g["t"]) - ftick)
                if dt <= MATCH_TICKS:
                    cands.append((dt, d3(g, origin), g))
            if not cands:
                continue
            cands.sort(key=lambda c: (c[0], c[1]))
            dt, dd, g = cands[0]
            if dd > MATCH_HU:
                continue
            team = teams[g["th"]] if 0 <= g["th"] < len(teams) else "?"
            expect = "incendiary" if team == "CT" else "molotov"
            if ft == expect:
                continue

            print(f"== 回合 idx={idx} eid={f['entity_id']} 火焰={ft} 引爆 tick={ftick} "
                  f"origin={[round(v,1) for v in origin]}")
            print(f"   投掷者队伍={team} -> 按队伍应判为 {expect}")
            for rank, (cdt, cdd, cg) in enumerate(cands[:4]):
                cteam = teams[cg["th"]] if 0 <= cg["th"] < len(teams) else "?"
                mark = "  <-- 选中" if rank == 0 else ""
                print(f"   cand{rank}: eid={cg['eid']:5d} th={cg['th']} ({cteam:2s}) "
                      f"dt={cdt:3d} d={cdd:7.1f} HU  t={cg['t']}{mark}")
            # 这颗投掷物的整条轨迹落在哪
            same = sorted(g2["t"] for g2 in molotovs if g2["eid"] == g["eid"])
            print(f"   该投掷物采样 tick 范围 {same[0]}..{same[-1]}（{len(same)} 点）")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
