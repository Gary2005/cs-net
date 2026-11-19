#!/usr/bin/env python3
"""AI 数据统计（`app/ai/stats.py` + `/api/ai/stats` + 玩家页的两个统计端点）。

这里能抓到、而且**不会报错**的错有三类：

  1. **时间轴错位**。`build_round_stats` 要"击杀前一个 / 后一个采样点"，而
     `predict_round` 手里的 `samples` 是**采样下标**（0, 16, 32…）、回合事件里的
     `t` 是 **tick**（9808, 9824…）。把两者当同一个轴用不会抛异常：每一次击杀
     都会落到"最后一个采样点之后"，影响力全变成 ±0.002 那种假数字。
     所以下面用手算的用例把精确值钉死。
  2. **加减反了**。击杀的影响力是"后减前"、被击杀的也是"后减前"（所以一般是
     负数）。符号反了界面上看着一样"有正有负"，只有手算能发现。
  3. **聚合口径**。跨场 / 跨回合要"先把累计量相加、再除一次"，不是平均值的
     平均（否则 5 回合的一场会和 24 回合的一场一样重）。

    python app/tests/test_ai_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from app.ai import stats as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

PASSED = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}   {detail}" if detail else f"  FAIL {name}")


def close(a, b, tol=1e-6) -> bool:
    return a is not None and abs(float(a) - float(b)) <= tol


# ═══════════════════════════════════════════════════════════════════════════
# 合成用例：8 个采样点、10 个人、2 次击杀（其中一次是回合结束那一枪）
# ═══════════════════════════════════════════════════════════════════════════
#
#   ticks   0    16   32   48   64   80   96   112      (113 = 采样范围外那一枪)
#   win_ct  .5   .5   .5   .8   .8   .8   .9   0.2      (模型看错，T 最终赢)
#
#   击杀 1：t=48 玩家 0(CT) 杀 5(T)
#       击杀者：win_ct[32]=.5 → win_ct[48]=.8     → +0.30
#       受害者：1-.5=.5 → 1-.8=.2                 → -0.30
#   击杀 2：t=113（> ticks[-1]）玩家 5(T) 杀 1(CT)
#       击杀者：后一个采样点不存在 → 按回合胜负补（T 赢 → 1.0）；
#               前一个点 win_T[112] = 1-0.2 = 0.8  → +0.20
#       受害者：前一个点 win_CT[112] = 0.2，后一个点按胜负补 0.0 → -0.20
#
#   回合影响力：win_ct 补上结果（T 赢 → 0.0）后逐格差分
#       [0, 0, .3, 0, 0, .1, -.7, -.2]，每步记在"这一步开始时"还活着的人头上
#       （玩家 3 从第 5 个采样点起死了）：
#       CT 全程活着的 4 人：.3/5 + .1/4 - .7/4 - .2/4 = -0.14
#       玩家 3 活到第 4 个采样点：.3/5 = +0.06
#       T 5 人：-(.3 + .1 - .7 - .2)/5 = +0.10
#       十个人加起来 = 0
#
#   next_death[t][0] 全是 0.1；next_kill[t][0] 全是 0.3
#   玩家 3 在第 5 个采样点（tick 64）之后死了 → 他只有 4 个"活着的采样点"
#
#   对枪评分：击杀 1 的 P(击杀者赢)=0.9、P(受害者赢)=0.4
#             击杀 2 的 P(击杀者赢)=0.2、P(受害者赢)=0.75
#       玩家 0：1-0.9 = +0.10           （1 次）
#       玩家 5：-0.4 + (1-0.2) = +0.40  （2 次）
#       玩家 1：-0.75                   （1 次）


def fixture():
    ticks = np.array([0, 16, 32, 48, 64, 80, 96, 112], dtype=np.int64)
    secs = ticks.astype(np.float64) / 64.0
    frames = ticks.size
    alive = np.ones((frames, 10), dtype=bool)
    alive[4:, 3] = False                      # 玩家 3 从 tick 64 起是死的
    team = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)

    tensors = SimpleNamespace(ticks=ticks, seconds=secs, alive=alive, team=team)

    win_ct = np.array([0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.9, 0.2])
    nd = np.zeros((frames, 11), dtype=np.float64)
    nk = np.zeros((frames, 11), dtype=np.float64)
    nd[:, 0] = 0.1
    nk[:, 0] = 0.3
    spatial = {"win": win_ct, "next_death": nd, "next_kill": nk}

    rnd = {
        "ticks": ticks.tolist(),
        "round_seconds": secs.tolist(),
        "winner": "T",
        "teams": ["CT"] * 5 + ["T"] * 5,
        "players": [{"x": [0.0] * frames, "y": [0.0] * frames,
                     "alive": alive[:, p].tolist()} for p in range(10)],
        "events": {"kills": [
            {"t": 48, "a": 0, "v": 5, "w": 1, "hs": False, "dmg": 100},
            {"t": 113, "a": 5, "v": 1, "w": 1, "hs": False, "dmg": 100},
            {"t": 48, "a": 2, "v": 2, "w": 1},          # 自杀：不算
            {"t": 48, "a": 0, "v": 1, "w": 1},          # 队友击杀：不算
        ], "damage": []},
    }
    return tensors, rnd, spatial


def test_probes() -> None:
    print("\n击杀探针（击杀前 2 秒的采样点）")
    tensors, rnd, _ = fixture()
    probes = st.kill_probes(tensors, rnd)
    check("只认两场对枪（自杀 / 队友击杀被滤掉）", len(probes) == 2, str(probes))
    check("第一场是 (0 杀 5) @ t=48", probes[0]["a"] == 0 and probes[0]["v"] == 5
          and probes[0]["t"] == 48, str(probes[0]))
    check("第二场是回合结束那一枪 (5 杀 1) @ t=113",
          probes[1]["a"] == 5 and probes[1]["v"] == 1 and probes[1]["t"] == 113,
          str(probes[1]))
    check("探针落在「击杀前 2 秒」的采样点上（0.75 - 2 = 负数 → 夹到 0）",
          all(p["i"] == 0 for p in probes), str([p["i"] for p in probes]))

    # 击杀在回合中段：t=96 → 1.5 - 2 = -0.5 还是负数，所以再造一个靠后的
    ticks = np.arange(0, 64 * 5, 16, dtype=np.int64)
    tensors2 = SimpleNamespace(ticks=ticks,
                               seconds=ticks.astype(np.float64) / 64.0,
                               alive=np.ones((ticks.size, 10), bool),
                               team=np.array([0] * 5 + [1] * 5, np.int8))
    # 最后一帧是 304（下标 19），后退 8 个采样点 = 128 tick = 2 秒 → t = 176，
    # 探针再往前 2 秒 = 112 tick… 不，176-128=48 tick 不在采样点上，所以落到
    # 48 之前最近的一格？—— 见断言：tick 176 对应 2.75 s，减 2 s = 0.75 s，
    # 也就是第 3 个采样点（0、16、32、48 → 0.75 s 正好是 48）。
    rnd2 = {"ticks": ticks.tolist(), "winner": "CT",
            "teams": ["CT"] * 5 + ["T"] * 5,
            "events": {"kills": [{"t": 176, "a": 0, "v": 5}]}}
    probe = st.kill_probes(tensors2, rnd2)[0]
    check("击杀在 t=176（2.75 s）时，探针取「前 2 秒」= 0.75 s 那一格",
          probe["i"] == 3, str(probe))


def test_round_stats() -> None:
    print("\n一回合的五个指标（手算）")
    tensors, rnd, spatial = fixture()
    probes = st.kill_probes(tensors, rnd)
    probs = np.array([0.9, 0.2, 0.4, 0.75])       # 两个方向各两个
    out = st.build_round_stats(tensors, rnd, np.arange(tensors.ticks.size),
                               spatial, probes, probs)
    rows = {r["idx"]: r for r in out["players"]}
    check("每个人一行", len(out["players"]) == 10)

    check("击杀者 0 的击杀影响力 = +0.30（后减前）",
          close(rows[0]["kill_sum"], 0.30), str(rows[0]["kill_sum"]))
    check("回合结束那一枪：击杀者按真实胜负补 → +0.20（不是 0）",
          close(rows[5]["kill_sum"], 0.20), str(rows[5]["kill_sum"]))
    check("回合影响力：胜率涨落按在场人数均分（CT 每人 -0.14）",
          close(rows[0]["round_impact"], -0.14), str(rows[0]["round_impact"]))
    check("T 每个人拿到镜像的 +0.10", close(rows[7]["round_impact"], 0.10),
          str(rows[7]["round_impact"]))
    check("中途死掉的人只拿死前那几步（玩家 3 = .3/5）",
          close(rows[3]["round_impact"], 0.06), str(rows[3]["round_impact"]))
    check("十个人的回合影响力加起来是 0",
          abs(sum(r["round_impact"] for r in out["players"])) < 1e-9,
          str(sum(r["round_impact"] for r in out["players"])))

    check("站位死亡风险：活着的 8 个采样点各 0.1",
          close(rows[0]["risk_sum"], 0.8) and rows[0]["risk_n"] == 8,
          f"{rows[0]['risk_sum']} / {rows[0]['risk_n']}")
    check("站位击杀率：活着的 8 个采样点各 0.3",
          close(rows[0]["kr_sum"], 2.4) and rows[0]["kr_n"] == 8,
          f"{rows[0]['kr_sum']} / {rows[0]['kr_n']}")
    check("死去之后不再计入（玩家 3 只有 4 个活着的采样点）",
          rows[3]["risk_n"] == 4, str(rows[3]["risk_n"]))

    check("对枪评分：击杀者得 1 - p", close(rows[0]["duel_sum"], 0.1),
          str(rows[0]["duel_sum"]))
    check("对枪评分：受害者得 -p", close(rows[1]["duel_sum"], -0.75),
          str(rows[1]["duel_sum"]))
    check("一个人两次对枪的分相加", close(rows[5]["duel_sum"], 0.4)
          and rows[5]["duel_n"] == 2, f"{rows[5]['duel_sum']} / {rows[5]['duel_n']}")

    check("对枪次数合计 == 传统统计的 duels（一次击杀算两边）",
          sum(r["duel_n"] for r in out["players"]) == 4,
          str(sum(r["duel_n"] for r in out["players"])))

    # 概率个数对不上时不要瞎配对（宁可整块不给分，也不要拿错的人的分）
    out2 = st.build_round_stats(tensors, rnd, np.arange(tensors.ticks.size),
                                spatial, probes, np.array([0.9]))
    check("概率个数对不上时对枪分数整块留空",
          all(r["duel_n"] == 0 for r in out2["players"]))


def test_wipe_guard() -> None:
    print("\n某队全灭后沿用最后一个还有人活着的采样点")
    tensors, rnd, spatial = fixture()
    alive = tensors.alive.copy()
    alive[6:, 5:] = False                      # T 在第 7 个采样点之前全灭
    tensors = SimpleNamespace(**{**vars(tensors), "alive": alive})
    rnd = {**rnd, "winner": "CT"}
    out = st.build_round_stats(tensors, rnd, np.arange(tensors.ticks.size),
                               spatial, [], np.zeros(0))
    rows = {r["idx"]: r for r in out["players"]}
    check("没有 NaN / inf", all(np.isfinite(r["round_impact"]) for r in out["players"]))
    check("全灭之后那几步照算：CT = .06 + .025 - .175 + .2 = .11",
          close(rows[0]["round_impact"], 0.11), str(rows[0]["round_impact"]))
    check("沿用 5 个 T 的名单：-(.3 + .1 - .7 + .8)/5 = -.10",
          close(rows[5]["round_impact"], -0.10), str(rows[5]["round_impact"]))
    check("十个人加起来仍然是 0",
          abs(sum(r["round_impact"] for r in out["players"])) < 1e-9,
          str(sum(r["round_impact"] for r in out["players"])))


def test_attribution() -> None:
    print("\n涨落记在「涨落之前」那一帧的名单上")
    tensors, rnd, spatial = fixture()
    ticks = np.array([0, 16, 32, 48], dtype=np.int64)
    alive = np.ones((4, 10), dtype=bool)
    alive[3:, 0] = False                       # 玩家 0 在最后一个采样点之前死了
    alive[2:, 5:] = False                      # T 从第 3 个采样点起全灭
    tensors = SimpleNamespace(ticks=ticks, seconds=ticks / 64.0, alive=alive,
                              team=tensors.team)
    rnd = {**rnd, "ticks": ticks.tolist(), "winner": "CT",
           "events": {"kills": []}}
    spatial = {"win": np.array([0.5, 0.5, 0.5, 0.9]),
               "next_death": spatial["next_death"][:4],
               "next_kill": spatial["next_kill"][:4]}
    out = st.build_round_stats(tensors, rnd, np.arange(4), spatial, [], np.zeros(0))
    rows = {r["idx"]: r for r in out["players"]}
    check("0.4 那一跳记给涨落之前还活着的 5 个 CT（他下一步才死）",
          close(rows[0]["round_impact"], 0.08), str(rows[0]["round_impact"]))
    check("活到最后的人多拿最后一步：.08 + .1/4",
          close(rows[1]["round_impact"], 0.105), str(rows[1]["round_impact"]))
    check("T 全灭后沿用 5 个人：-(.4 + .1)/5 = -.10",
          close(rows[5]["round_impact"], -0.10), str(rows[5]["round_impact"]))
    check("十个人加起来是 0",
          abs(sum(r["round_impact"] for r in out["players"])) < 1e-9,
          str(sum(r["round_impact"] for r in out["players"])))


def test_aggregate() -> None:
    print("\n聚合（一场 / 跨场）")
    tensors, rnd, spatial = fixture()
    probes = st.kill_probes(tensors, rnd)
    probs = np.array([0.9, 0.2, 0.4, 0.75])
    one = {"stats": st.build_round_stats(tensors, rnd, np.arange(tensors.ticks.size),
                                         spatial, probes, probs)}

    agg = st.aggregate([one])
    p0 = agg["players"][0]
    check("平均击杀影响力 = 0.30 / 1", close(p0["kill_impact"], 0.3), str(p0))
    check("平均回合影响力 = -0.14 / 1 回合", close(p0["round_impact"], -0.14))
    check("站位死亡风险 = 0.8 / 8 = 0.1", close(p0["death_risk"], 0.1))
    check("站位击杀率 = 2.4 / 8 = 0.3", close(p0["kill_rate"], 0.3))
    check("对枪评分 = 0.1 / 1", close(p0["duel_score"], 0.1))
    check("每回合明细跟着走", len(p0["per_round"]) == 1
          and close(p0["per_round"][0]["kill_impact"], 0.3))
    check("原始求和留给跨场聚合", close(p0["totals"]["kill_sum"], 0.3)
          and close(p0["totals"]["round_sum"], -0.14), str(p0["totals"]))

    # 两场：一场 1 回合、一场 3 回合 —— 平均回合影响力必须是"总和 / 4"，
    # 不是 (-0.14/1 + -0.14/3) / 2
    two = st.aggregate([one, one, one])
    p0b = two["players"][0]
    check("三个回合的回合影响力 = 总和 / 3", close(p0b["round_impact"], -0.14),
          str(p0b["round_impact"]))
    three = st.aggregate([one, one, one])
    check("回合数累加", three["players"][0]["rounds"] == 3)

    comb = st.combine([
        {"totals": {"kill_sum": 1.0, "kill_n": 1, "round_sum": 2.0,
                    "risk_sum": 1.0, "risk_n": 10, "duel_sum": 1.0, "duel_n": 1},
         "rounds": 5},
        {"totals": {"kill_sum": 0.0, "kill_n": 1, "round_sum": 4.0,
                    "risk_sum": 2.0, "risk_n": 10, "duel_sum": 0.0, "duel_n": 1},
         "rounds": 15},
    ])
    check("跨场：击杀影响力 = 1.0 / 2", close(comb["kill_impact"], 0.5))
    check("跨场：回合影响力 = 6.0 / 20（不是两场平均值的平均）",
          close(comb["round_impact"], 0.3), str(comb["round_impact"]))
    check("跨场：站位风险 = 3.0 / 20", close(comb["death_risk"], 0.15))
    check("跨场：回合数相加", comb["rounds"] == 20)


def test_predict_match() -> None:
    print("\n整场：逐回合接起来 + 落盘缓存")
    from app.ai import service

    tensors, rnd, spatial = fixture()
    probes = st.kill_probes(tensors, rnd)
    probs = np.array([0.9, 0.2, 0.4, 0.75])
    round_stats = st.build_round_stats(tensors, rnd, np.arange(tensors.ticks.size),
                                       spatial, probes, probs)

    data = {
        "meta": {"map": "de_test"},
        "players": [{"steamid": f"id{p}", "name": f"P{p}"} for p in range(10)],
        "rounds": [rnd, rnd],
    }

    saved = service.predict_round
    calls = []

    def fake_predict(d, idx, *, csz_path=None, stride=16, refresh=False, progress=None):
        calls.append(idx)
        if idx == 1 and len(calls) > 1 and fake_predict.fail_second:
            raise RuntimeError("这个回合的数据坏了")
        return {"stats": round_stats, "ok": True}

    fake_predict.fail_second = True
    try:
        service.predict_round = fake_predict
        payload = st.predict_match(data, None, stride=16)
        check("两个回合都请求了", calls == [0, 1], str(calls))
        check("坏掉的那个回合被记进 errors（不拖垮整场）",
              len(payload["errors"]) == 1
              and payload["errors"][0]["round"] == 1, str(payload["errors"]))
        check("另一个回合照常汇总", payload["rounds"] == 1, str(payload["rounds"]))
        check("玩家行带上 steamid / name",
              payload["players"][0]["steamid"] == "id0"
              and payload["players"][0]["name"] == "P0")
        check("回合总数照实报（2 个，其中 1 个没算出来）",
              payload["rounds_total"] == 2)
        check("版本字段写进去了（缓存失效靠它）",
              payload["version"] == st.MATCH_VERSION
              and payload["round_version"] == service.CACHE_VERSION)

        # 进度：两个回合各占一半
        seen: list[tuple[float, str]] = []
        fake_predict.fail_second = False
        calls.clear()
        st.predict_match(data, None, stride=16,
                         progress=lambda pct, msg: seen.append((pct, msg)))
        check("进度从 0 走到 1", seen[0][0] >= 0 and seen[-1][0] >= 0.99,
              f"{seen[0]} … {seen[-1]}")
        check("进度带「回合 k/N」的人话", any("回合 2/2" in m for _, m in seen),
              str([m for _, m in seen][:3]))
    finally:
        service.predict_round = saved


def test_routes() -> None:
    print("\n路由（/api/ai/stats 与玩家页两个端点）")
    import json as _json

    from flask import Flask

    import visualizer.server as viz
    from app.ai import service, stats as ai_stats
    from app.ai.api import bp as ai_bp
    from app import stats_api

    app = Flask(__name__)
    app.register_blueprint(ai_bp)
    app.register_blueprint(stats_api.bp)
    client = app.test_client()

    # ── /api/ai/stats：没有会话 ────────────────────────────────────────
    saved_snap = viz.session_snapshot
    saved_predict = service.predict_round
    saved_instance = service.Predictor.instance
    try:
        viz.session_snapshot = lambda: None
        check("没有会话 → 400", client.get("/api/ai/stats").status_code == 400)

        tensors, rnd, spatial = fixture()
        probes = st.kill_probes(tensors, rnd)
        probs = np.array([0.9, 0.2, 0.4, 0.75])
        rs = st.build_round_stats(tensors, rnd, np.arange(tensors.ticks.size),
                                  spatial, probes, probs)
        data = {"meta": {"map": "de_test"}, "format": "cs2.demo.v7",
                "players": [{"steamid": f"id{p}", "name": f"P{p}"} for p in range(10)],
                "rounds": [rnd]}
        viz.session_snapshot = lambda: {"data": data, "csz_path": None, "source": "x"}

        class FakePredictor:
            model_key = "testmodel"

        service.Predictor.instance = classmethod(lambda cls: FakePredictor())

        calls = []

        def fake_predict(d, idx, *, csz_path=None, stride=16, refresh=False, progress=None):
            calls.append(idx)
            return {"stats": rs}

        service.predict_round = fake_predict

        # 用临时目录做缓存根（不许写进真的 app/data）。故意不用
        # `tempfile.mkdtemp()`：Windows 上它按 0o700 建目录，而某些受限环境
        # （比如带 ACL 沙箱的开发机）里那种目录连自己都写不进去，症状是下面
        # 三条缓存断言报 PermissionError，看着像"缓存没写"。默认权限建目录就没这事。
        import tempfile
        import uuid

        tmp = Path(tempfile.gettempdir()) / f"aistats-{uuid.uuid4().hex[:12]}"
        tmp.mkdir()
        saved_root = service.cache_root
        service.cache_root = lambda: tmp
        try:
            r = client.get("/api/ai/stats")
            check("整场统计 → 200", r.status_code == 200, str(r.status_code))
            payload = r.get_json()
            check("载荷里有五组数", all(k in payload["players"][0] for k in (
                "kill_impact", "round_impact", "death_risk", "kill_rate",
                "duel_score")), str(payload.get("players", [{}])[0]))
            check("第一次是现算的", payload.get("cached") is False, str(payload.get("cached")))
            check("每个回合都算了一次", calls == [0], str(calls))
            check("写了比赛级缓存", len(list(tmp.rglob("*_match.json"))) == 1,
                  str(list(tmp.rglob("*"))))

            calls.clear()
            r2 = client.get("/api/ai/stats")
            payload2 = r2.get_json()
            check("第二次直接读缓存（不再跑模型）", calls == [], str(calls))
            check("缓存命中标记", payload2.get("cached") is True)
            check("两次的结果一样",
                  payload2["players"][0]["duel_score"] == payload["players"][0]["duel_score"])
        finally:
            service.cache_root = saved_root

        # ── 玩家页：传统统计 ──────────────────────────────────────────
        fake_stats = {
            "ok": True, "version": stats_api.TRAD_VERSION, "smoke_ok": True,
            "rounds": 2, "map": "de_test",
            "players": [
                {"idx": 0, "steamid": "id0", "name": "P0",
                 "kills": 3, "deaths": 1, "assists": 0, "headshots": 1,
                 "damage": 300, "damage_taken": 100, "rounds": 2,
                 "enemy_kills": 3, "enemy_deaths": 1, "duels": 4,
                 "first_kills": 1, "first_deaths": 0, "openers": 2,
                 "side_kills": 1, "side_deaths": 1, "smoke_kills": 1},
            ],
        }
        saved_demos = stats_api._player_demos
        saved_trad = stats_api.demo_trad_stats
        stats_api._player_demos = lambda sid, limit=200: [
            {"id": 1, "map_name": "de_test", "match_date": "2026-01-01",
             "filename": "a.dem", "total_rounds": 2, "csz_path": __file__},
            {"id": 2, "map_name": "de_test", "match_date": "2026-01-02",
             "filename": "b.dem", "total_rounds": 2, "csz_path": ""},
        ]
        stats_api.demo_trad_stats = lambda csz, refresh=False, progress=None: fake_stats
        try:
            r = client.get("/api/library/players/id0/stats")
            check("玩家传统统计 → 200", r.status_code == 200, str(r.status_code))
            pl = r.get_json()
            check("4 场的合计里只有 1 场有数据", pl["totals"]["kills"] == 3,
                  str(pl["totals"].get("kills")))
            check("没有 .csz 的那一场记为错误", len(pl["errors"]) == 1
                  and pl["errors"][0]["error"], str(pl["errors"]))
            check("比例字段是重算的（先加再除）",
                  pl["totals"]["opener_pct"] == 50.0, str(pl["totals"]["opener_pct"]))
            check("K/D 也对", pl["totals"]["kd"] == 3.0, str(pl["totals"]["kd"]))
        finally:
            stats_api._player_demos = saved_demos
            stats_api.demo_trad_stats = saved_trad

        # ── 玩家页：AI 统计（读比赛级缓存）───────────────────────────
        match_payload = {
            "ok": True, "version": ai_stats.MATCH_VERSION,
            "round_version": service.CACHE_VERSION, "cached": True,
            "players": [{"idx": 0, "steamid": "id0", "name": "P0", "rounds": 2,
                         "totals": {"kill_sum": 0.4, "kill_n": 2, "round_sum": 0.6,
                                    "risk_sum": 1.0, "risk_n": 20,
                                    "duel_sum": 0.2, "duel_n": 2},
                         "kill_impact": 0.2, "round_impact": 0.3}],
        }
        stats_api._player_demos = lambda sid, limit=200: [
            {"id": 1, "map_name": "de_test", "match_date": "2026-01-01",
             "filename": "a.dem", "csz_path": __file__},
        ]
        saved_key = stats_api._demo_key
        saved_read = ai_stats.read_match_cache
        stats_api._demo_key = lambda csz: "deadbeef"
        ai_stats.read_match_cache = lambda path, round_version: dict(match_payload)
        try:
            r = client.get("/api/library/players/id0/ai")
            check("玩家 AI 统计 → 200", r.status_code == 200, str(r.status_code))
            ai = r.get_json()
            check("每个选手每场一行", len(ai["matches"]) == 1
                  and ai["matches"][0]["stats"]["rounds"] == 2)
            check("跨场汇总是从 totals 加出来的",
                  ai["totals"]["round_impact"] == 0.3,
                  str(ai["totals"]))
            check("带上是哪个选手", ai["steamid"] == "id0")
        finally:
            stats_api._demo_key = saved_key
            ai_stats.read_match_cache = saved_read
            stats_api._player_demos = saved_demos
    finally:
        viz.session_snapshot = saved_snap
        service.predict_round = saved_predict
        service.Predictor.instance = saved_instance


def main() -> int:
    print("AI 数据统计（app/ai/stats.py）")
    test_probes()
    test_round_stats()
    test_wipe_guard()
    test_attribution()
    test_aggregate()
    test_predict_match()
    test_routes()
    print(f"\n{PASSED} passed, {len(FAILED)} failed" + (f": {FAILED}" if FAILED else ""))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
