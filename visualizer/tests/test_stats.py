#!/usr/bin/env python3
"""
比赛统计（/api/stats）：合成用例 + 真实 .csz 的交叉验证。

这个功能有两类会出错、而且**都不会报错**的地方：

  1. **伤害口径**。`kills[].dmg` 只是致命那一枪（API.md §1.9 实测中位数 31），
     不是"造成的伤害"。用错不报错，只是 ADR 悄悄偏低 —— 而且偏低得很像真的。
     这里的判据是硬下界：**每死一个人至少掉 100 点血**，所以
     总伤害 ≥ 100 × 总死亡数。`kills[].dmg` 在真实数据上过不了这一关（见
     stats.py 的说明），`events.damage` 能过。
  2. **队伍划分**。半场会换边，按"第一回合站哪边"分会把同一队拆成两组。
     合成用例里专门造了一场换边的比赛。

真实文件那一半拿**另一份实现**（`app/metadata.py` 的记分板）对拍 K/D/A/爆头/
队伍/比分：两边算法写法不同（一个从 `kills` 数、一个也是从 `kills` 数，但分组和
胜负判定各自写了一遍），对上才说明不是自己验自己。

    python visualizer/tests/test_stats.py [/path/to/match.csz]
    CS2_CSZ=/path/to/match.csz python visualizer/tests/test_stats.py

没有 .csz 时只跑合成用例（真实数据那半打印 SKIP，不影响退出码）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "tools"), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visualizer import stats as st  # noqa: E402

# Windows 控制台默认是 GBK，中文表头和 "−" 这类字符会当场把测试打成
# UnicodeEncodeError（看着像功能坏了，其实是打印坏了）。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

FAILED: list[str] = []
PASSED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
    else:
        FAILED.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond else ""))


# ── 合成用例 ────────────────────────────────────────────────────────────────
#
# 4 个人、2 队、3 回合，第 2 回合换边（0/1 从 CT 去 T，2/3 反过来）。
# 事件里塞进了四种"边界"数据：
#   * 自伤（a == v）    —— 两边都不该算
#   * 世界伤害（a == -1）—— 只算承伤
#   * 助攻 / 爆头
#   * 某个选手某回合完全没数据（0 值）


def synthetic_data() -> dict:
    return {
        "format": "cs2.demo.v7",
        "meta": {"map": "de_test"},
        "players": [
            {"steamid": "1", "name": "A-one"},
            {"steamid": "2", "name": "A-two"},
            {"steamid": "3", "name": "B-one"},
            {"steamid": "4", "name": "B-two"},
        ],
        "rounds": [
            {   # 第 1 回合：0 和 1 在 CT，2 和 3 在 T
                "id": 1, "winner": "CT", "end_reason": "bomb_defused",
                "round_seconds": [0.0, 30.0],
                "teams": ["CT", "CT", "T", "T"],
                "events": {
                    "kills": [
                        # 0 杀掉 2（爆头），1 助攻
                        {"t": 100, "a": 0, "v": 2, "as": 1, "w": 1, "hs": True, "dmg": 100},
                        # 3 杀掉 0
                        {"t": 200, "a": 3, "v": 0, "as": -1, "w": 1, "hs": False, "dmg": 100},
                    ],
                    "damage": [
                        {"t": 90, "a": 0, "v": 2, "hp": 40, "w": 1},
                        {"t": 95, "a": 1, "v": 2, "hp": 20, "w": 1},
                        {"t": 100, "a": 0, "v": 2, "hp": 100, "w": 1},  # 致命一枪：只掉 40
                        {"t": 150, "a": 3, "v": 0, "hp": 30, "w": 1},
                        {"t": 200, "a": 3, "v": 0, "hp": 100, "w": 1},  # 致命一枪：只掉 70
                    ],
                },
            },
            {   # 第 2 回合：换边。自伤 / 世界伤害放在这一回合，免得污染第 1 回合的
                #      「承伤 ≤ 100 × 死亡数」守恒检查（世界伤害不算任何人的输出）。
                "id": 2, "winner": "T", "end_reason": "bomb_exploded",
                "round_seconds": [0.0, 41.5],
                "teams": ["T", "T", "CT", "CT"],
                "events": {
                    "kills": [
                        {"t": 300, "a": 2, "v": 0, "as": 3, "w": 2, "hs": False, "dmg": 100},
                    ],
                    "damage": [
                        # 按 tick 顺序（stats.py 假定事件已经按时间排好）
                        {"t": 290, "a": 0, "v": 0, "hp": 55, "w": 9},    # 自伤：不计分
                        {"t": 300, "a": 2, "v": 0, "hp": 100, "w": 2},   # 满血 100 → 掉光
                        {"t": 310, "a": -1, "v": 0, "hp": 17, "w": 0},   # 血槽已空 → 记 0
                        {"t": 360, "a": -1, "v": 1, "hp": 17, "w": 0},   # 世界伤害：只算承伤
                    ],
                },
            },
            {   # 第 3 回合：什么事件都没有（0 值那一路），T 赢
                "id": 3, "winner": "T", "end_reason": "t_win",
                "round_seconds": [0.0, 20.0],
                "teams": ["CT", "CT", "T", "T"],
                "events": {},
            },
        ],
    }


def test_synthetic() -> None:
    print("\n合成用例：换边 / 助攻 / 自伤 / 世界伤害")
    s = st.build_stats(synthetic_data())
    players = {p["name"]: p for p in s["players"]}

    check("回合数 = 3", s["rounds"] == 3)
    check("地图透传", s["map"] == "de_test")

    # ── 队伍：换边之后仍然是同一队（这正是 team_groups 存在的理由）────────
    check("分成两队", len(s["teams"]) == 2, f"{s['teams']}")
    check("A-one / A-two 同队", s["players"][0]["team"] == s["players"][1]["team"],
          f"{[p['team'] for p in s['players']]}")
    check("B-one / B-two 同队", s["players"][2]["team"] == s["players"][3]["team"])
    check("两队不是同一队", s["players"][0]["team"] != s["players"][2]["team"])
    check("每队 2 人", sorted(len(t["players"]) for t in s["teams"]) == [2, 2])
    check("队伍覆盖了每个选手且不重复",
          sorted(i for t in s["teams"] for i in t["players"]) == [0, 1, 2, 3])
    check("比分 = 按 winner 数出来的（CT 先的队 2 分）",
          s["teams"][0]["score"] == 2 and s["teams"][1]["score"] == 1,
          f"{[t['score'] for t in s['teams']]}")
    check("side0 记的是开场那一边", s["teams"][0]["side0"] == "CT"
          and s["teams"][1]["side0"] == "T", f"{[t['side0'] for t in s['teams']]}")

    # ── 合计 ──────────────────────────────────────────────────────────────
    #
    # 手工数一遍（这就是那份合成数据里发生的事；伤害按**剩余血量**截断）：
    #   r1: 0 杀 2（爆头，1 助攻）；3 杀 0
    #       B-one 掉血：40 + 20 + 100=160 → 截断成 40 + 20 + 40（血槽只有 100）
    #       A-one 掉血：30 + 100=130       → 截断成 30 + 70
    #   r2: 2 杀 0（3 助攻）—— A-one 满血，被打 100，正好掉光；另有自伤 55 与
    #       两个世界伤害 17（A-one / A-two 各一个，满血所以全额扣）
    #   r3: 无事发生
    # 于是 —— 0: 1杀2死, 造成 100, 承伤 100+100（世界伤害不计）
    #          1: 0杀0死1助攻, 造成 20, 承伤 17（世界伤害）
    #          2: 1杀1死, 造成 100, 承伤 100
    #          3: 1杀0死1助攻, 造成 100, 承伤 0
    a1 = players["A-one"]
    b1, b2 = players["B-one"], players["B-two"]
    check("A-one: K/D/A = 1/2/0", (a1["kills"], a1["deaths"], a1["assists"]) == (1, 2, 0),
          f"{a1['kills']}/{a1['deaths']}/{a1['assists']}")
    check("A-one: 1 个爆头", a1["headshots"] == 1)
    check("A-two: 1 个助攻、没有击杀", players["A-two"]["assists"] == 1
          and players["A-two"]["kills"] == 0)
    check("B-one: 1 杀 1 死 0 助攻", (b1["kills"], b1["deaths"], b1["assists"]) == (1, 1, 0),
          f"{b1['kills']}/{b1['deaths']}/{b1['assists']}")
    check("B-two: 1 杀 0 死 1 助攻", (b2["kills"], b2["deaths"], b2["assists"]) == (1, 0, 1),
          f"{b2['kills']}/{b2['deaths']}/{b2['assists']}")

    # A-one 造成：r1 打 B-one 40 + 40（最后一枪时对方只剩 40 血，所以只记 40）
    # ＋ r2 打 A-one 100；自伤那 55 不算
    check("致命一枪按剩余血量截断（A-one 伤害 = 80，不是 40+100）",
          a1["damage"] == 80, f"{a1['damage']}")
    # A-one 承受：r1 被 B-two 打 30 + 70（第 2 枪只掉 70），r2 被 B-one 打 100
    check("A-one 承伤 = 200（两回合各掉光一次）", a1["damage_taken"] == 200,
          f"{a1['damage_taken']}")
    check("A-two 承伤含世界伤害（17）", players["A-two"]["damage_taken"] == 17,
          f"{players['A-two']['damage_taken']}")
    check("每个选手的输出都对得上（世界伤害不算谁的输出）",
          [p["damage"] for p in s["players"]] == [80, 20, 100, 100],
          f"{[p['damage'] for p in s['players']]}")
    check("每个人的承伤也对得上",
          [p["damage_taken"] for p in s["players"]] == [200, 17, 100, 0],
          f"{[p['damage_taken'] for p in s['players']]}")
    # 截断之后：一个人的承伤不可能超过 100 × 他的死亡数（一回合一条命，死了才掉光 100）
    def _cap(r, i):
        return 100 * r["deaths"][i] if r["deaths"][i] else 100

    check("每回合每人的承伤 ≤ 100（没死）/ 100 × 死亡数（死了）",
          all(r["damage_taken"][i] <= _cap(r, i)
              for r in s["per_round"] for i in range(4)),
          f"{[(r['damage_taken'], r['deaths']) for r in s['per_round']]}")
    # 伤害守恒：截断是对称的，所以造成之和**正好**等于承受之和（世界伤害只进后者）
    check("造成之和 + 世界伤害 == 承受之和（截断后守恒）",
          sum(p["damage"] for p in s["players"]) + 17
          == sum(p["damage_taken"] for p in s["players"]),
          f"{sum(p['damage'] for p in s['players'])} vs "
          f"{sum(p['damage_taken'] for p in s['players'])}")

    # ── 派生公式（和玩家页一致）───────────────────────────────────────────
    check("ADR = 伤害 / 回合数（80/3 = 26.7）", a1["adr"] == 26.7, f"{a1['adr']}")
    check("KPR = 击杀 / 回合数（1/3 = 0.33）", a1["kpr"] == 0.33, f"{a1['kpr']}")
    check("K/D = 1/2 = 0.5", a1["kd"] == 0.5, f"{a1['kd']}")
    check("0 死亡时 K/D 退化成击杀数", b2["kd"] == 1.0, f"{b2['kd']}")
    check("爆头率 = 爆头/击杀 = 100%", a1["hs_pct"] == 100.0, f"{a1['hs_pct']}")
    check("0 击杀时爆头率是 0 而不是除零",
          all(p["hs_pct"] == 0 or p["kills"] for p in s["players"]))
    check("ADR / KPR 的分母是整场回合数，不是这个人出场的回合数",
          players["A-two"]["adr"] == round(20 / 3, 1), f"{players['A-two']['adr']}")

    # ── 每回合 ────────────────────────────────────────────────────────────
    pr = s["per_round"]
    check("每回合都有一条", len(pr) == 3)
    check("每回合的四个数组都按人数对齐",
          all(len(r[k]) == 4 for r in pr for k in ("kills", "deaths", "damage", "damage_taken")))
    check("回合 1 的击杀落在下标 0 和 3 上",
          pr[0]["kills"] == [1, 0, 0, 1], f"{pr[0]['kills']}")
    check("回合 1 的死亡落在下标 0 和 2 上",
          pr[0]["deaths"] == [1, 0, 1, 0], f"{pr[0]['deaths']}")
    check("回合 1 的伤害按人分开（助攻先削血，致命一枪只记剩下的）",
          pr[0]["damage"] == [80, 20, 0, 100], f"{pr[0]['damage']}")
    check("回合 1 的承伤按人分开（B-one 100 / A-one 100）",
          pr[0]["damage_taken"] == [100, 0, 100, 0], f"{pr[0]['damage_taken']}")
    check("回合 1 里单人承伤没有超过 100（死一次最多掉 100 血）",
          all(v <= 100 for v in pr[0]["damage_taken"]), f"{pr[0]['damage_taken']}")
    check("回合 2 的击杀 / 死亡按下标落位",
          pr[1]["kills"] == [0, 0, 1, 0] and pr[1]["deaths"] == [1, 0, 0, 0],
          f"{pr[1]['kills']} / {pr[1]['deaths']}")
    # 第 2 回合的三人各自掉血：A-one 满血被击杀 100、之后世界伤害 17 记 0（槽已空）；
    # A-two 只挨了一个世界伤害 17；B-two 没有事件
    check("回合 2 的承伤：血槽空了就不再扣，世界伤害只算承伤",
          pr[1]["damage_taken"] == [100, 17, 0, 0], f"{pr[1]['damage_taken']}")
    check("没有事件的回合全是 0",
          pr[2]["kills"] == [0, 0, 0, 0] and pr[2]["damage"] == [0, 0, 0, 0])
    check("回合时长取 round_seconds 最后一项", pr[1]["duration"] == 41.5, f"{pr[1]['duration']}")
    check("回合 winner / 结束原因透传",
          pr[1]["winner"] == "T" and pr[1]["end_reason"] == "bomb_exploded")

    # 每回合之和必须等于合计：两处对各写一遍，最容易在这里分叉
    for key, total_key in (("kills", "kills"), ("deaths", "deaths"),
                           ("damage", "damage"), ("damage_taken", "damage_taken")):
        summed = [sum(r[key][i] for r in pr) for i in range(4)]
        check(f"每回合的 {key} 之和 == 合计",
              summed == [p[total_key] for p in s["players"]],
              f"{summed} vs {[p[total_key] for p in s['players']]}")

    # 空数据不该炸
    empty = st.build_stats({"players": [], "rounds": [], "meta": {}})
    check("空数据返回空表而不是抛异常",
          empty["players"] == [] and empty["teams"] == [] and empty["per_round"] == [])


# ── 热身 ────────────────────────────────────────────────────────────────────
#
# 对战平台（5E）会把赛前热身整段录进 demo，而**热身和第一回合都报
# `total_rounds_played == 0`**。真实例子（g161-…_de_inferno.dem）：热身 71 次
# 击杀、正赛 115 次，不裁掉的话第一回合的 K/D 直接翻倍，而且不报任何错。
#
# 判据是"tick 早于本回合第一个采样点"—— `.csz` 里每个回合的 `ticks[0]` 就是它
# 自己的 `round_freeze_end`，比它早的事件只可能是热身。


def warmup_data() -> dict:
    ticks = list(range(1000, 1011))
    blank = {"x": [0.0] * 11, "y": [0.0] * 11, "z": [0.0] * 11, "yaw": [0.0] * 11,
             "pitch": [0.0] * 11, "hp": [100.0] * 11, "alive": [True] * 11}
    return {
        "format": "cs2.demo.v7",
        "meta": {"map": "de_test", "tickrate": 64},
        "players": [{"steamid": str(i), "name": f"P{i}"} for i in range(4)],
        "rounds": [{
            "id": 0, "winner": "CT", "end_reason": "t_killed",
            "ticks": ticks,
            "round_seconds": [round((x - 1000) / 64.0, 6) for x in ticks],
            "teams": ["CT", "CT", "T", "T"],
            "players": [dict(blank) for _ in range(4)],
            "events": {
                "kills": [
                    # 热身：全都发生在回合开始之前，只是被算进了 round 0 的桶里
                    {"t": 300, "a": 0, "v": 1, "as": -1, "w": 1, "hs": False, "dmg": 100},
                    {"t": 400, "a": 2, "v": 3, "as": -1, "w": 1, "hs": True, "dmg": 100},
                    # 正赛：唯一一条
                    {"t": 1005, "a": 0, "v": 2, "as": -1, "w": 1, "hs": False, "dmg": 100},
                ],
                "damage": [
                    {"t": 350, "a": 1, "v": 0, "hp": 40, "w": 1},
                    {"t": 1005, "a": 0, "v": 2, "hp": 100, "w": 1},
                ],
            },
        }],
    }


def test_warmup() -> None:
    print("\n热身不算进比赛数据（5E 这类平台会录热身）")
    s = st.build_stats(warmup_data(), smoke=False)
    p = {x["name"]: x for x in s["players"]}

    check("热身击杀不进 K/D",
          (p["P0"]["kills"], p["P2"]["kills"]) == (1, 0),
          f"{[(x['name'], x['kills']) for x in s['players']]}")
    check("热身死亡也不记",
          (p["P1"]["deaths"], p["P3"]["deaths"]) == (0, 0),
          f"{[(x['name'], x['deaths']) for x in s['players']]}")
    check("热身伤害不进 ADR / 承伤",
          (p["P0"]["damage_taken"], p["P2"]["damage_taken"]) == (0, 100),
          f"{[(x['name'], x['damage_taken']) for x in s['players']]}")
    check("正赛那一条照常算",
          (p["P0"]["kills"], p["P2"]["deaths"]) == (1, 1))
    check("被丢掉的条数报出来了（前端拿它解释和平台的差异）",
          (s["warmup_kills"], s["warmup_damage"]) == (2, 1),
          f"{s['warmup_kills']} / {s['warmup_damage']}")
    check("每回合的数组也是干净的",
          s["per_round"][0]["kills"] == [1, 0, 0, 0], f"{s['per_round'][0]['kills']}")
    # 没有 ticks 的数据（老 .csz / 手搓夹具）不裁，否则会把整场都算成热身
    legacy = warmup_data()
    legacy["rounds"][0].pop("ticks")
    check("没有 ticks 时不裁任何事件",
          st.build_stats(legacy, smoke=False)["warmup_kills"] == 0)


# ── 新指标：首杀 / 先手开枪 / 侧背身 / 穿烟 ──────────────────────────────────
#
# 合成用例（**和上面那场无关**，另造一场 4 人 / 1 回合的比赛），每个数都是手算的：
#
#   采样轴：ticks 100..110（11 帧，逐 tick），tickrate 64 → 4 s 窗口 = 256 tick
#   位置：0(100,0) 1(0,100) 2(0,0) 3(100,100)，整回合不动
#
#   击杀 A：t=108 玩家 0 杀 2
#       伤害   t=102  2 → 0      ← 受害者先开的枪
#              t=106  0 → 2      ← 击杀者第一次打到受害者（侧背身判定时刻）
#              t=108  0 → 2      （致命一枪）
#       受害者 2 在 t=106 的 yaw = 90°（朝 +Y），杀伤者 0 在 +X 方向 → 夹角 90°
#       期望：先手 = 2；侧背身 → 0 的 side_kills +1、2 的 side_deaths +1
#
#   击杀 B：t=109 玩家 3 杀 1
#       伤害   t=105  1 → 3      ← 先手 = 1
#              t=107  3 → 1      ← 判定时刻
#       受害者 1 在 t=107 的 yaw = 0°（朝 +X），杀人者 3 也在 +X → 夹角 0°，不算侧背身
#
#   另外塞三条**必须被排除**的击杀：自杀（a==v）、队友击杀（0 杀 1，两人都 CT）、
#   落在采样范围外的一条（t=9999）。


def extras_data() -> dict:
    n, t = 4, list(range(100, 111))
    secs = [round((x - 100) / 64.0, 6) for x in t]

    def rec(x, y, yaw=0.0, yaw_at=None):
        """一个玩家的列式记录；`yaw_at` 覆盖某一帧的 yaw。"""
        y = [yaw] * len(t)
        for tick, val in (yaw_at or {}).items():
            y[tick - 100] = val
        return {"x": [x] * len(t), "y": y, "z": [0.0] * len(t),
                "yaw": y, "pitch": [0.0] * len(t), "hp": [100.0] * len(t),
                "alive": [True] * len(t)}

    return {
        "format": "cs2.demo.v7",
        "meta": {"map": "de_test", "tickrate": 64},
        "players": [{"steamid": str(i), "name": f"P{i}"} for i in range(n)],
        "rounds": [{
            "id": 1, "winner": "CT", "end_reason": "t_killed",
            "ticks": t, "round_seconds": secs,
            "teams": ["CT", "CT", "T", "T"],
            "players": [
                rec(100, 0),                       # 0：靠 +X 一边
                rec(0, 100),                       # 1：靠 +Y 一边
                rec(0, 0, yaw_at={106: 90.0}),     # 2：被 0 打，判定那一刻朝 +Y
                rec(100, 100),                     # 3
            ],
            "events": {
                "kills": [
                    {"t": 108, "a": 0, "v": 2, "as": -1, "w": 1, "hs": False, "dmg": 100},
                    {"t": 109, "a": 3, "v": 1, "as": -1, "w": 1, "hs": False, "dmg": 100},
                    {"t": 109, "a": 1, "v": 1, "as": -1, "w": 9, "hs": False, "dmg": 50},
                    {"t": 109, "a": 0, "v": 1, "as": -1, "w": 1, "hs": False, "dmg": 30},
                    {"t": 9999, "a": 2, "v": 3, "as": -1, "w": 1, "hs": False, "dmg": 100},
                ],
                "damage": [
                    {"t": 102, "a": 2, "v": 0, "hp": 30, "w": 1},
                    {"t": 105, "a": 1, "v": 3, "hp": 20, "w": 1},
                    {"t": 106, "a": 0, "v": 2, "hp": 40, "w": 1},
                    {"t": 107, "a": 3, "v": 1, "hp": 35, "w": 1},
                    {"t": 108, "a": 0, "v": 2, "hp": 60, "w": 1},
                    {"t": 109, "a": 3, "v": 1, "hp": 80, "w": 1},
                ],
            },
        }],
    }


def test_extras() -> None:
    print("\n新指标：首杀 / 先手开枪 / 侧背身（合成用例，手算）")
    # 合成数据里没有地图几何，穿烟那一列整块关掉（见 build_stats 的 smoke 参数）
    s = st.build_stats(extras_data(), smoke=False)
    p = {x["name"]: x for x in s["players"]}
    r = s["per_round"][0]

    check("穿烟判定关掉时 smoke_ok = False（前端显示「—」）", s["smoke_ok"] is False)
    check("穿烟关掉时穿烟击杀全是 0",
          all(x["smoke_kills"] == 0 for x in s["players"]))
    check("首杀只记第一个击杀的击杀方（0）", p["P0"]["first_kills"] == 1
          and sum(x["first_kills"] for x in s["players"]) == 1,
          f"{[x['first_kills'] for x in s['players']]}")
    check("首死记第一个被击杀的人（2）", p["P2"]["first_deaths"] == 1
          and sum(x["first_deaths"] for x in s["players"]) == 1)
    check("击杀总表：自杀不算击杀，队友击杀 / 回合外的仍是老口径",
          [x["kills"] for x in s["players"]] == [2, 0, 1, 1],
          f"{[x['kills'] for x in s['players']]}")
    check("自杀仍然算死亡",
          [x["deaths"] for x in s["players"]] == [0, 3, 1, 1],
          f"{[x['deaths'] for x in s['players']]}")
    check("对枪击杀只数敌人的（自杀 / 队友击杀 / 回合外都不算）",
          [x["enemy_kills"] for x in s["players"]] == [1, 0, 0, 1],
          f"{[x['enemy_kills'] for x in s['players']]}")
    check("对枪死亡同理",
          [x["enemy_deaths"] for x in s["players"]] == [0, 1, 1, 0],
          f"{[x['enemy_deaths'] for x in s['players']]}")
    check("duels = 造成 + 被造成",
          all(x["duels"] == x["enemy_kills"] + x["enemy_deaths"] for x in s["players"]))
    check("先手开枪：受害者先开的枪算受害者（P2）", p["P2"]["openers"] == 1
          and p["P0"]["openers"] == 0,
          f"{[(x['name'], x['openers']) for x in s['players']]}")
    check("先手开枪：另一场对枪的先手是 P1", p["P1"]["openers"] == 1
          and p["P3"]["openers"] == 0)
    check("每场对枪恰好一个先手（合计 == 对枪数）",
          sum(x["openers"] for x in s["players"]) == 2,
          f"{sum(x['openers'] for x in s['players'])}")
    check("先手占比 = openers / duels",
          p["P0"]["opener_pct"] == 0.0 and p["P2"]["opener_pct"] == 100.0,
          f"{[(x['name'], x['opener_pct']) for x in s['players']]}")
    check("侧背身击杀：受害者朝 +Y、击杀者在 +X（90°）→ P0 记 1 个",
          p["P0"]["side_kills"] == 1, f"{p['P0']['side_kills']}")
    check("侧背身被击杀：同一件事的另一端 → P2 记 1 个", p["P2"]["side_deaths"] == 1)
    check("侧背身：另一个击杀的夹角是 0°（正对着），两边都不记",
          p["P3"]["side_kills"] == 0 and p["P1"]["side_deaths"] == 0,
          f"{p['P3']['side_kills']} / {p['P1']['side_deaths']}")
    check("侧背身击杀之和 == 被击杀之和（同一个事件的两个计数器）",
          sum(x["side_kills"] for x in s["players"])
          == sum(x["side_deaths"] for x in s["players"]) == 1)
    check("侧背身占比 = side_kills / enemy_kills",
          p["P0"]["side_kill_pct"] == 100.0 and p["P3"]["side_kill_pct"] == 0.0)
    check("侧背身被击杀占比 = side_deaths / enemy_deaths",
          p["P2"]["side_death_pct"] == 100.0 and p["P1"]["side_death_pct"] == 0.0)

    # ── 每回合那一份必须和合计对得上（两处各写一遍，最容易在这里分叉）────
    for key in st.EXTRA_KEYS:
        check(f"每回合的 {key} 和合计一致",
              r[key] == [x[key] for x in s["players"]],
              f"{r[key]} vs {[x[key] for x in s['players']]}")

    # 空 / 缺列不该炸：老数据里没有 ticks 和 players 列（合成用例就是这种）
    legacy = extras_data()
    legacy["rounds"][0].pop("ticks")
    legacy["rounds"][0].pop("players")
    ls = st.build_stats(legacy, smoke=False)
    check("没有 ticks / players 列时新指标给 0，不抛异常",
          all(x["enemy_kills"] == 0 and x["openers"] == 0 for x in ls["players"]))


# ── 真实文件 ────────────────────────────────────────────────────────────────
def _find_csz() -> str | None:
    if len(sys.argv) > 1:
        return sys.argv[1]
    env = os.environ.get("CS2_CSZ")
    if env:
        return env
    for cand in sorted((_ROOT / "app" / "data" / "csz").glob("*.csz")):
        return str(cand)
    return None


def test_real(csz: str) -> None:
    from demo_extract import codec
    from app.metadata import build_metadata

    print(f"\n真实文件：{Path(csz).name}")
    data = codec.load(csz)
    s = st.build_stats(data)
    n = len(s["players"])
    print(f"  {s['rounds']} 回合 · {n} 人 · "
          f"比分 {s['teams'][0]['score']}:{s['teams'][1]['score']}")

    check("人数 == .csz 名单长度", n == len(data.get("players") or []))
    check("每回合四个数组都是定长",
          all(len(r[k]) == n for r in s["per_round"]
              for k in ("kills", "deaths", "damage", "damage_taken")))

    # ── 伤害口径：按剩余血量截断之后，这几条必须同时成立 ─────────────────
    total_deaths = sum(p["deaths"] for p in s["players"])
    total_dmg = sum(p["damage"] for p in s["players"])
    total_taken = sum(p["damage_taken"] for p in s["players"])
    raw_dmg = sum(int(d.get("hp") or 0)
                  for r in data["rounds"]
                  for d in ((r.get("events") or {}).get("damage") or [])
                  if d.get("a") != d.get("v"))
    kill_dmg = sum(int(k.get("dmg") or 0)
                   for r in data["rounds"] for k in (r.get("events") or {}).get("kills") or [])
    # 截断之后单人单回合不可能超过 100（一回合只有一条命），且必须和面板逐格一致。
    # 顺带量出「世界伤害」（a == -1，摔落 / C4）：它只进承伤、不进任何人的造成。
    bad_cells = []
    world_dmg = 0
    for rnd in data["rounds"]:
        ev = rnd.get("events") or {}
        slot = [100] * n
        r_taken = [0] * n
        for d in ev.get("damage") or []:
            a, v = d.get("a", -1), d.get("v", -1)
            if a == v or not (0 <= v < n):
                continue                       # 自伤不算；受害者不在名单里也丢掉
            hit = min(int(d.get("hp") or 0), slot[v])
            slot[v] -= hit
            r_taken[v] += hit
            if a == -1:
                world_dmg += hit               # 世界伤害：只进承伤
        for i in range(n):
            if r_taken[i] > 100:
                bad_cells.append((rnd.get("id"), s["players"][i]["name"], r_taken[i]))
    # 截断之后单人单回合**玩家造成的**承伤不可能超过 100（一回合只有一条命）
    check("每回合每个人的承伤 ≤ 100 —— 「承伤 443」的护栏", not bad_cells,
          f"{bad_cells[:5]}")
    check(f"截断确实压低了名义伤害（{total_dmg} < {raw_dmg} = 原样累加）",
          total_dmg < raw_dmg, f"panel={total_dmg} raw={raw_dmg}")
    check(f"kills[].dmg 过不了这个下界（{kill_dmg} < {100 * total_deaths}）—— "
          "所以它不能当总伤害", kill_dmg < 100 * total_deaths,
          "如果这一条 FAIL，说明这份 demo 的数据集变了，stats.py 的口径要重新论证")
    # 截断是对称的：「造成之和」+「世界伤害」==「承受之和」（自伤两边都不进）。
    # 注意别写「总数 ≈ 100 × 击杀数」—— 受伤但没死的人也计入承伤，
    # 所以总和天然会超过 100 × 击杀数（mirage R0：5 次击杀、605 承伤）。
    check(f"造成之和 + 世界伤害({world_dmg}) == 承受之和（截断后守恒）",
          total_dmg + world_dmg == total_taken,
          f"{total_dmg} + {world_dmg} vs {total_taken}")

    # ── 和另一份实现（app/metadata.py 的记分板）对拍 ──────────────────────
    md = build_metadata(data, filename=Path(csz).name)
    sb = {p["steamid"]: p for p in json.loads(md["scoreboard_json"])}
    bad = []
    for p in s["players"]:
        ref = sb.get(p["steamid"])
        if ref is None:
            bad.append(f"{p['name']}: 记分板里没有")
            continue
        for mine, theirs, label in ((p["kills"], ref["kills"], "K"),
                                    (p["deaths"], ref["deaths"], "D"),
                                    (p["assists"], ref["assists"], "A"),
                                    (p["headshots"], ref["hs"], "HS"),
                                    (p["team"], ref["team"], "队伍")):
            if mine != theirs:
                bad.append(f"{p['name']} {label}: {mine} != {theirs}")
    check("K / D / A / 爆头 / 队伍 和库里的记分板逐项相同", not bad, "; ".join(bad[:6]))

    check("比分和记分板一致",
          [t["score"] for t in s["teams"]] == [md["team_a_score"], md["team_b_score"]],
          f"{[t['score'] for t in s['teams']]} vs "
          f"{[md['team_a_score'], md['team_b_score']]}")

    # ── 新指标：结构性的恒等式（不需要另一份实现就能验）─────────────────
    #
    # 这几条是"数错了立刻能看出来"的那一类：一回合只有一个首杀 / 一个首死；
    # 侧背身是同一件事的两个计数器（A 的击杀 == B 的被击杀）；每场对枪恰好一个
    # 先手；穿烟 / 侧背身都是击杀的子集。
    players = s["players"]
    rounds_n = s["rounds"]
    check("首杀合计 == 回合数 == 首死合计",
          sum(p["first_kills"] for p in players) == rounds_n
          == sum(p["first_deaths"] for p in players),
          f"{sum(p['first_kills'] for p in players)} / "
          f"{sum(p['first_deaths'] for p in players)} / {rounds_n}")
    check("侧背身击杀之和 == 侧背身被击杀之和",
          sum(p["side_kills"] for p in players) == sum(p["side_deaths"] for p in players),
          f"{sum(p['side_kills'] for p in players)} vs "
          f"{sum(p['side_deaths'] for p in players)}")
    check("每场对枪恰好一个先手（openers × 2 == duels 之和 + 缺数据的那几场）",
          2 * sum(p["openers"] for p in players) <= sum(p["duels"] for p in players),
          f"{sum(p['openers'] for p in players)} vs {sum(p['duels'] for p in players)}")
    check("对枪击杀 <= 击杀、对枪死亡 <= 死亡（自杀 / 队友 / 回合外被排掉）",
          all(p["enemy_kills"] <= p["kills"] and p["enemy_deaths"] <= p["deaths"]
              for p in players))
    check("侧背身 / 穿烟都是击杀的子集",
          all(p["side_kills"] <= p["enemy_kills"] and p["smoke_kills"] <= p["enemy_kills"]
              and p["side_deaths"] <= p["enemy_deaths"] for p in players))
    check("穿烟判定可用（有地图几何 / numba）", s["smoke_ok"] is True)
    check("比例字段和分子分母对得上",
          all(
              (p["opener_pct"] == (round(100.0 * p["openers"] / p["duels"], 1)
                                   if p["duels"] else 0.0))
              and (p["side_kill_pct"] == (round(100.0 * p["side_kills"] / p["enemy_kills"], 1)
                                          if p["enemy_kills"] else 0.0))
              and (p["side_death_pct"] == (round(100.0 * p["side_deaths"] / p["enemy_deaths"], 1)
                                           if p["enemy_deaths"] else 0.0))
              for p in players))
    for key in st.EXTRA_KEYS:
        check(f"每回合的 {key} 之和 == 合计",
              [sum(r[key][i] for r in s["per_round"]) for i in range(n)]
              == [p[key] for p in players])
    side_pct = (100.0 * sum(p["side_kills"] for p in players)
                / max(1, sum(p["enemy_kills"] for p in players)))
    smoke_pct = (100.0 * sum(p["smoke_kills"] for p in players)
                 / max(1, sum(p["enemy_kills"] for p in players)))
    open_pct = (100.0 * sum(p["openers"] for p in players)
                / max(1, sum(p["duels"] for p in players)))
    print(f"  新指标：侧背身 {side_pct:.1f}% · 穿烟 {smoke_pct:.1f}% · "
          f"先手开枪 {open_pct:.1f}% · 本场 {s['ms']:.0f} ms")

    # 库里的 ADR 用的是 kills[].dmg，两边**故意**不一样 —— 这里把它量出来，
    # 免得哪天有人"顺手对齐"把回放页也改回低口径。
    lib_dmg = sum(p["adr_damage"] for p in sb.values())
    check("库里的 ADR 口径（kills[].dmg）确实更低，且差距被记录在这里",
          lib_dmg < total_dmg,
          f"lib={lib_dmg} vs panel={total_dmg}")
    print(f"  伤害口径：面板（截断后）{total_dmg} / 原样累加 {raw_dmg} "
          f"（{total_dmg / raw_dmg:.2f}×） / 库里 {lib_dmg} "
          f"（{lib_dmg / total_dmg:.2f}×）")

    # ── 路由 ──────────────────────────────────────────────────────────────
    from visualizer import server as vis

    app = vis.create_app(maps_dir=None)
    client = app.test_client()
    check("没有会话时 /api/stats 返回 400", client.get("/api/stats").status_code == 400)

    with vis._lock:
        vis._clear_session()
        vis._session.update({"data": data, "source": Path(csz).name,
                             "csz_path": None, "maps_dir": None})
    try:
        res = client.get("/api/stats")
        payload = res.get_json()
        check("/api/stats 返回 200", res.status_code == 200, f"{res.status_code}")
        check("载荷里的合计和 build_stats 一致",
              payload["players"] == s["players"] and payload["per_round"] == s["per_round"])
        check("载荷里带上是哪一场", payload["source"] == Path(csz).name)
        check("结果缓存在会话里（第二次不再重算）",
              vis._session.get("stats") is not None
              and client.get("/api/stats").get_json()["players"] == s["players"])
        check("响应不许缓存", res.headers.get("Cache-Control") == "no-store",
              f"{res.headers.get('Cache-Control')}")
    finally:
        with vis._lock:
            vis._clear_session()


def test_assist_team_rule() -> None:
    """助攻只算敌人：队友误伤、再由对面补掉，事件里有 assister 也不计。"""
    print("\n助攻口径：只算敌人")
    data = extras_data()
    data["rounds"][0]["events"]["kills"].extend([
        # 3(T) 杀 1(CT)，4 号没有；assister 0 是 CT（和受害者同队）→ 不算
        {"t": 100, "a": 3, "v": 1, "as": 0, "w": 1, "hs": False, "dmg": 100},
        # assister 2 是 T（受害者的敌人）→ 算
        {"t": 101, "a": 3, "v": 1, "as": 2, "w": 1, "hs": False, "dmg": 100},
    ])
    s = st.build_stats(data, smoke=False)
    p = {x["name"]: x for x in s["players"]}
    check("队友误伤的 assister 不算助攻", p["P0"]["assists"] == 0,
          str(p["P0"]["assists"]))
    check("敌人的 assister 才算助攻", p["P2"]["assists"] == 1,
          str(p["P2"]["assists"]))


def main() -> int:
    test_synthetic()
    test_warmup()
    test_extras()
    test_assist_team_rule()
    csz = _find_csz()
    if not csz or not Path(csz).exists():
        print("\nSKIP: 没有 .csz（传一个路径或设 CS2_CSZ）—— 只跑了合成用例")
    else:
        test_real(csz)

    print(f"\n{PASSED} passed, {len(FAILED)} failed" + (f": {FAILED}" if FAILED else ""))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
