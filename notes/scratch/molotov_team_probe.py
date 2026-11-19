"""探针：验证「燃烧弹 / 莫洛托夫」的投掷物类名是同一个，能否用**投掷者队伍**区分。

背景：demo 里 `events.grenades[].ty` 是**引擎类名**（实测 5 种：
`CSmokeGrenadeProjectile` / `CHEGrenadeProjectile` / `CMolotovProjectile` /
`CFlashbangProjectile` / `CDecoyProjectile`），而莫洛托夫和燃烧弹共用
`CMolotovProjectile`。要挂对模型就得有个区分办法。

CS2 规则：莫洛托夫只有 T 能买、燃烧弹只有 CT 能买。这个探针**不靠规则**，
而是拿真实数据交叉验证：把每个火焰实例（`fire[].type_name` ∈ molotov/incendiary）
和"引爆时刻空间上最近的 CMolotov 投掷物采样"配起来，看
（投掷者队伍 → 火焰类型）是不是一对一的。

    python notes/scratch/molotov_team_probe.py [base_url]
"""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5099"
MATCH_TICKS = 64          # 引爆时刻前后容忍 1 秒
MATCH_HU = 200.0          # 位置容忍 200 HU


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def dist(a: dict, b: list[float]) -> float:
    return ((a["x"] - b[0]) ** 2 + (a["y"] - b[1]) ** 2 + (a["z"] - b[2]) ** 2) ** 0.5


def main() -> int:
    s = get("/api/session")
    if not s.get("loaded"):
        print("服务器上没有加载 demo")
        return 1

    pairs: Counter[tuple[str, str]] = Counter()
    unmatched = 0
    n_fire = 0
    sample_fire = None

    for idx in range(len(s.get("rounds") or [])):
        r = get(f"/api/round/{idx}")
        teams = r.get("teams") or []
        fires = r.get("fire") or []
        nades = (r.get("events") or {}).get("grenades") or []
        molotovs = [g for g in nades if g.get("ty") == "CMolotovProjectile"]
        if not molotovs:
            continue
        if sample_fire is None and fires:
            sample_fire = {k: v for k, v in fires[0].items() if k != "cells"}

        for f in fires:
            n_fire += 1
            ft = f.get("type_name")
            ftick = f.get("first_tick")
            origin = f.get("origin")
            if not origin:
                unmatched += 1
                continue
            best = None
            best_d = None
            for g in molotovs:
                dt = abs(int(g["t"]) - int(ftick))
                if dt > MATCH_TICKS:
                    continue
                d = dist(g, origin)
                if d > MATCH_HU:
                    continue
                # 先看时间，再看距离
                score = (dt, d)
                if best_d is None or score < best_d:
                    best_d, best = score, g
            if best is None:
                unmatched += 1
                continue
            th = best.get("th")
            team = teams[th] if isinstance(th, int) and 0 <= th < len(teams) else "?"
            pairs[(team, str(ft))] += 1

    print(f"火焰实例: {n_fire}   成功配对: {sum(pairs.values())}   未配对: {unmatched}")
    if sample_fire is not None:
        print(f"fire 实例字段示例: {json.dumps(sample_fire, ensure_ascii=False)[:300]}")
    print("\n(投掷者队伍, 火焰 type_name) 分布:")
    for (team, ft), c in sorted(pairs.items()):
        print(f"  {c:5d}  队伍={team:3s}  火焰类型={ft}")

    teams_seen = {t for (t, _) in pairs}
    kinds = {k for (_, k) in pairs}
    print()
    if len(pairs) <= 2 and len(teams_seen) <= 2:
        print("=> 队伍与火焰类型一一对应，可以用队伍区分模型")
        for (team, ft), _ in sorted(pairs.items()):
            print(f"   {team} -> {ft}")
    else:
        print("=> !! 不是一一对应，队伍不能用来区分模型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
