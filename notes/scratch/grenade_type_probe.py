"""探针：从正在运行的 visualizer 服务器上读出真实投掷物类型字符串与每回合数量。

用途：`visualizer/static/js/visuals.js` 里要把飞行中的投掷物从"圆点"换成真实
GLB 模型，就得知道 `events.grenades[].ty` 到底是什么字符串 —— 它决定了
"类别 -> 模型文件"的映射表怎么写。**不猜，读服务器上的真数据。**

    python notes/scratch/grenade_type_probe.py [base_url]
"""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5099"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def main() -> int:
    s = get("/api/session")
    if not s.get("loaded"):
        print("服务器上没有加载 demo —— 先在 UI 里加载一个")
        return 1
    print(f"demo   : {s['meta'].get('demo')}  地图: {s.get('map')}")
    n_rounds = len(s.get("rounds") or [])
    print(f"回合数 : {n_rounds}")

    types: Counter[str] = Counter()
    per_round: list[int] = []
    eids: Counter[int] = Counter()
    bad_coord = 0
    no_eid = 0
    total = 0

    for i in range(n_rounds):
        r = get(f"/api/round/{i}")
        gs = (r.get("events") or {}).get("grenades") or []
        per_round.append(len(gs))
        for g in gs:
            total += 1
            types[str(g.get("ty"))] += 1
            eid = g.get("eid")
            if not isinstance(eid, int) or eid < 0:
                no_eid += 1
            else:
                eids[eid] += 1
            for k in ("x", "y", "z"):
                if g.get(k) is None:
                    bad_coord += 1
                    break

    print(f"\n投掷物采样点总数: {total}   （每回合中位数 {sorted(per_round)[len(per_round)//2]}）")
    print(f"坐标缺失         : {bad_coord}")
    print(f"eid 非法         : {no_eid}")
    print(f"不同 eid 个数    : {len(eids)}   同一 eid 最多出现 {max(eids.values()) if eids else 0} 次")

    print("\nty 字符串（这是映射表的键）:")
    for t, c in types.most_common():
        print(f"  {c:6d}  {t!r}")

    print("\n每回合投掷物采样点数（前 20 个回合）:")
    print("  " + " ".join(str(x) for x in per_round[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
