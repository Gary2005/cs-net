#!/usr/bin/env python3
"""`app/ai` 的单元测试：对枪窗口、窗口切片、进度、缓存、路由。

**不跑真模型**（那是 1.5 GB + 十几秒，属于端到端脚本
`visualizer/tests/test_ai_overlay.mjs` 的事）。这里量的是那些"错了也不报错、
只会悄悄画错"的地方：

* `duel_windows`：只认敌人、不认自杀 / 队友 / 队伍未知、弧线 = 死前 2 秒、
  终点取受害者**最后一个活帧**
* `plan_duels`：一枪一条窗口（锚在击杀帧、193 帧）、窗口里死过人就丢掉、列号换算
* `RoundTensors.aim_arrays`：补齐到固定 193 帧、`seconds` 对零、`frame_mask`、
  窗口起点 / 列号
* `Progress`：只增不减、finish 之后变成 result / error 事件
* 缓存：键跟着文件**内容**变（mtime 变了不算）、版本不对就作废、写进去能读回来
* 烟体积缓存：跟着存活期走（散了不许命中旧的栅格）、同回合复用 entity_id 的两颗烟
  不许互相覆盖
* 权重：`PINNED_WEIGHTS` 的扁平 `<run>.pt` + 同名 `<run>.yaml`（老布局也认），
  缺了要报出是哪条路径
* 路由：没有会话 / 模型不可用 / 数据不合规 各自的状态码；`?progress=1` 的 NDJSON 流

Run:  python app/tests/test_ai.py
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import uuid
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from app import config  # noqa: E402

_passed = 0
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  \u2713 {name}")
    else:
        _failed.append(name)
        print(f"  \u2717 {name}" + (f"\n      {detail}" if detail else ""))


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r}, want {want!r}")


def scratch() -> Path:
    base = _ROOT / ".build" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    return d


# ═══════════════════════════════════════════════════════════════════════════
# 对枪窗口
# ═══════════════════════════════════════════════════════════════════════════


def _round(frames: int = 600, deaths: dict[int, int] | None = None):
    """一个 600 帧 / 9.375 s 的假回合：`deaths = {受害者: 死在那一帧}`。"""
    from app.ai.features import N_PROJ, RoundTensors

    alive = np.ones((frames, 10), dtype=bool)
    for victim, at in (deaths or {}).items():
        alive[at:, victim] = False
    ticks = np.arange(frames, dtype=np.int64) + 1000
    tensors = RoundTensors(
        map_name="de_mirage",
        map_id=0,
        ticks=ticks,
        seconds=np.arange(frames, dtype=np.float64) / 64.0,
        feat=np.zeros((frames, 10, 20), np.float32),
        alive=alive,
        weapon=np.full((frames, 10), 3, np.int8),
        team=np.array([0] * 5 + [1] * 5, np.int8),
        obstruction=np.zeros((frames, 10, 10), np.int8),
        projectiles=np.zeros((frames, N_PROJ, 4), np.float32),
        proj_id=np.full((frames, N_PROJ), -1, np.int8),
        global_feat=np.zeros((frames, 5), np.float32),
    )
    rnd = {"ticks": ticks.tolist(), "events": {"kills": []}}
    return tensors, rnd


def _kill(tick: int, attacker: int, victim: int) -> dict:
    return {"t": tick, "a": attacker, "v": victim}


def test_duel_windows() -> None:
    from app.ai.service import DUEL_LEAD, duel_windows

    eq("弧线画死前 2 秒", DUEL_LEAD, 2.0)

    # 6 号在 400 帧杀了 3 号 → 弧线 = [400 帧前 2 秒, 398]（3 号最后一个活帧）
    tensors, rnd = _round(deaths={3: 400})
    rnd["events"]["kills"] = [_kill(1000 + 400, 6, 3)]
    duels = duel_windows(tensors, rnd)
    eq("一场击杀 = 一个窗口", len(duels), 1)
    d = duels[0]
    eq("A 端 = 受害者", d.a, 3)
    eq("B 端 = 击杀者", d.b, 6)
    eq("终点 = 受害者最后一个活帧（不是击杀帧）", d.i1, 399)
    eq("起点 = 死前 2 秒（6.234 - 2 = 4.234 s → 第 271 帧）", d.i0, 271)
    check("弧线正好 2 秒", abs((d.t1 - d.t0) - 2.0) < 0.0001, f"{d.t0}..{d.t1}")
    eq("as_dict 里两个端点和帧号都带出去",
       {k: d.as_dict()[k] for k in ("a", "b", "i0", "i1")},
       {"a": 3, "b": 6, "i0": 271, "i1": 399})

    # 各种"不该有对枪"的情况
    cases = {
        "队友击杀（6 杀 7，都是 T）": [_kill(1500, 6, 7)],
        "自杀 / 同一个人（a == v）": [_kill(1500, 3, 3)],
        "摔死 / 被 C4 炸死（a = -1）": [_kill(1500, -1, 3)],
        "事件 tick 不落在采样点上": [_kill(1500.5, 6, 3)],
    }
    for name, kills in cases.items():
        tensors2, rnd2 = _round(deaths={3: 400})
        rnd2["events"]["kills"] = kills
        eq(f"跳过：{name}", len(duel_windows(tensors2, rnd2)), 0)

    # 队伍未知：把人 7 的队伍抹成 -1
    from dataclasses import replace

    tensors3, rnd3 = _round(deaths={3: 400})
    tensors3 = replace(tensors3, team=np.array([0] * 5 + [1, 1, -1, 1, 1], np.int8))
    rnd3["events"]["kills"] = [_kill(1400, 7, 3)]
    eq("跳过：击杀者队伍未知", len(duel_windows(tensors3, rnd3)), 0)

    # 受害者在整段里没活过（缺数据）
    tensors4, rnd4 = _round()
    tensors4.alive[:, 3] = False
    rnd4["events"]["kills"] = [_kill(1000 + 400, 6, 3)]
    eq("跳过：受害者整段都没活过", len(duel_windows(tensors4, rnd4)), 0)

    # 回合最开头那一刀：窗口自动从第 0 帧开始（不可能有负下标）
    tensors5, rnd5 = _round(deaths={3: 40})
    rnd5["events"]["kills"] = [_kill(1000 + 40, 6, 3)]
    d5 = duel_windows(tensors5, rnd5)[0]
    eq("太靠前的击杀：窗口从 0 开始", d5.i0, 0)
    eq("…终点还是最后一个活帧", d5.i1, 39)

    # 多场对枪按"死得早"排序
    tensors6, rnd6 = _round(deaths={3: 300, 4: 400})
    rnd6["events"]["kills"] = [
        _kill(1000 + 400, 6, 4),
        _kill(1000 + 300, 7, 3),
    ]
    order = [d.a for d in duel_windows(tensors6, rnd6)]
    eq("多场对枪按死亡时间排序", order, [3, 4])


def test_round_end_kill() -> None:
    """回合结束那一枪（`t = ticks[-1] + 1`）也要有对枪窗口。

    "打死最后一个敌人、回合立刻结束"的那一枪比最后一个采样帧晚 1 tick（引擎在
    死亡那一 tick 就把回合计数加一），`round_events()` 会整条滤掉；不补的话每个
    回合的最后一杀都没有弧线。
    """
    from app.ai.features import AIM_WINDOW
    from app.ai.service import duel_windows, plan_duels

    tensors, rnd = _round()                       # 600 帧 / 9.375 s，谁都没死
    hi = int(tensors.ticks[-1])
    rnd["events"]["kills"] = [_kill(hi + 1, 6, 3)]
    duels = duel_windows(tensors, rnd)
    eq("回合结束那一枪算一场对枪", len(duels), 1)
    d = duels[0]
    eq("A 端 = 受害者", d.a, 3)
    eq("B 端 = 击杀者", d.b, 6)
    eq("挂在最后一帧上（受害者活到那一帧）", d.i1, tensors.frames - 1)
    eq("起点仍是死前 2 秒（9.359 - 2 = 7.359 s → 第 471 帧）", d.i0, 471)
    check("弧线正好 2 秒", abs((d.t1 - d.t0) - 2.0) < 0.0001, f"{d.t0}..{d.t1}")

    runs = plan_duels(tensors, duels)
    eq("一枪一条窗口（不是一帧一条）", len(runs), 1)
    run = runs[0]
    eq("窗口锚在击杀帧上（3 秒 → 193 帧）", run.start, d.i1 - (AIM_WINDOW - 1))
    eq("窗口固定 193 帧", run.length, AIM_WINDOW)
    eq("击杀帧是最后一列", run.column(d.i1), AIM_WINDOW - 1)
    eq("弧线第一列 = 窗口第 64 列（死前 2 秒）", run.column(d.i0), 64)

    grid = np.arange(0, tensors.frames, 16, dtype=np.int64)
    cols = [run.column(int(i)) for i in grid if d.i0 <= i <= d.i1]
    check("弧线上的采样点都落在窗口里", bool(cols) and min(cols) >= 64
          and max(cols) <= AIM_WINDOW - 1, str(cols))

    # 只补 +1：+2 已经不是"下一个 tick"，没有帧可挂
    tensors2, rnd2 = _round()
    rnd2["events"]["kills"] = [_kill(int(tensors2.ticks[-1]) + 2, 6, 3)]
    eq("晚 2 tick 的击杀不补", len(duel_windows(tensors2, rnd2)), 0)

    # 过滤器照旧：团队击杀 / 自杀 / 队伍未知都不算
    tensors3, rnd3 = _round()
    rnd3["events"]["kills"] = [_kill(int(tensors3.ticks[-1]) + 1, 6, 7)]
    eq("跳过：回合结束那一枪的团队击杀", len(duel_windows(tensors3, rnd3)), 0)

    # 中段一杀 + 回合结束那一枪：两场，最后一场挂在最后一帧
    tensors4, rnd4 = _round(deaths={3: 300})
    rnd4["events"]["kills"] = [
        _kill(int(tensors4.ticks[-1]) + 1, 6, 4),
        _kill(1000 + 300, 7, 3),
    ]
    ds = duel_windows(tensors4, rnd4)
    eq("两场：中段一杀 + 回合结束那一枪", [x.a for x in ds], [3, 4])
    eq("…最后一枪挂在最后一帧", ds[-1].i1, tensors4.frames - 1)


def test_assign_duels() -> None:
    from app.ai.features import AIM_WINDOW
    from app.ai.service import Duel, assign_duels, plan_duels

    d1 = Duel(a=3, b=6, i0=100, i1=300, t0=1.5, t1=4.6)
    d2 = Duel(a=4, b=7, i0=200, i1=480, t0=3.1, t1=7.5)
    samples = np.arange(0, 600, 16, dtype=np.int64)
    assigned = assign_duels([d1, d2], samples)
    by_frame = dict(zip(samples.tolist(), assigned))

    eq("第一场窗口里归第一场", by_frame[160].a, 3)
    eq("第二场窗口里归第二场", by_frame[480].a, 4)
    eq("重叠区（死得早的那场优先）", by_frame[240].a, 3)
    eq("窗口之前没有对枪", by_frame[0], None)
    eq("两场都结束之后没有对枪", by_frame[576], None)

    # 窗口里有人死过 → 这场对枪不喂模型（训练分布里双方全程活着）
    tensors, _ = _round(deaths={3: 340})
    d = Duel(a=3, b=6, i0=211, i1=339, t0=3.3, t1=5.3)   # 弧线 = 死前 2 秒
    runs = plan_duels(tensors, [d])
    eq("双方全程活着 → 给一条窗口", len(runs), 1)
    eq("窗口从击杀帧往回 193 帧", runs[0].start, 339 - (AIM_WINDOW - 1))
    eq("击杀帧是最后一列", runs[0].column(339), AIM_WINDOW - 1)
    eq("弧线第一列 = 第 64 列（死前 2 秒）", runs[0].column(d.i0), 64)

    # 击杀者在窗口里死过 → 整场丢掉
    tensors2, _ = _round(deaths={3: 340, 6: 100})
    eq("击杀者死过 → 跳过", len(plan_duels(tensors2, [d])), 0)

    # 回合最开头那一刀：窗口被帧 0 截短
    tensors3, _ = _round(deaths={3: 40})
    d3 = Duel(a=3, b=6, i0=0, i1=39, t0=0.0, t1=0.609)
    run3 = plan_duels(tensors3, [d3])[0]
    eq("太靠前的击杀：窗口从 0 开始", run3.start, 0)
    eq("…只有 40 帧（其余靠补零）", run3.length, 40)


# ═══════════════════════════════════════════════════════════════════════════
# 窗口切片
# ═══════════════════════════════════════════════════════════════════════════


def _tensors(frames: int = 300):
    """一个只有形状、内容全 0 的 RoundTensors（`aim_arrays` 只关心形状）。"""
    return _round(frames)[0]


def test_aim_arrays() -> None:
    from app.ai.features import AIM_WINDOW

    t = _tensors(300)
    eq("回合第一帧的窗口只有 1 帧", t.window_bounds(0), (0, 1))
    eq("最后一帧的窗口是完整 193 帧", t.window_bounds(299), (107, 300))

    arrays, starts, lengths = t.aim_arrays([(299, 4, 7), (0, 0, 9)])
    eq("T 固定 193（不再按 batch 内最长补齐）", arrays["ctx_player"].shape,
       (2, AIM_WINDOW, 10, 20))
    eq("每条窗口的起点", list(starts), [107, 0])
    eq("每个样本自己的真实帧数", list(lengths), [AIM_WINDOW, 1])
    eq("frame_mask 只标真帧", arrays["frame_mask"].sum(axis=1).tolist(),
       [AIM_WINDOW, 1])
    eq("最后一帧是最后一列（列号 = 帧号 - 起点）",
       int(299 - starts[0]), AIM_WINDOW - 1)
    eq("pair 按 (A, B) 原样写进去", arrays["pair"].tolist(), [[4, 7], [0, 9]])
    check("seconds 从窗口第一帧对零",
          abs(arrays["seconds"][0, 1] - 1 / 64.0) < 1e-7
          and abs(arrays["seconds"][0, 0]) < 1e-9,
          str(arrays["seconds"][0, :3]))
    eq("补齐帧的武器是 -1（不是沙鹰 0）", int(arrays["ctx_player_weapon"][1, 1, 0]), -1)
    eq("补齐帧的道具 id 是 -1", int(arrays["ctx_projectile_id"][1, 1, 0]), -1)
    eq("dashboard map_id 每行都有", arrays["map_id"].tolist(), [0, 0])

    # 两个短窗口：长度不同，但都补到固定的 193 帧
    arrays, starts, lengths = t.aim_arrays([(5, 1, 2), (9, 1, 2)])
    eq("短窗口各自长度", list(lengths), [6, 10])
    eq("短窗口都从帧 0 开始", list(starts), [0, 0])
    eq("T 恒等于 AIM_WINDOW", arrays["ctx_player"].shape[1], AIM_WINDOW)
    eq("短窗口的尾巴是补的", int(arrays["frame_mask"][0, 6:].sum()), 0)


def test_spatial_arrays() -> None:
    t = _tensors(20)
    a = t.spatial_arrays(np.array([0, 5, 5]))
    eq("ctx_player 形状", a["ctx_player"].shape, (3, 10, 20))
    eq("ctx_player_mask 是 bool", a["ctx_player_mask"].dtype, bool)
    eq("队伍列 broadcast 到 10 个槽位", a["ctx_player_team"].shape, (3, 10))
    eq("队伍列的值", a["ctx_player_team"][0].tolist(), [0] * 5 + [1] * 5)
    eq("map_id 每行一个", a["map_id"].tolist(), [0, 0, 0])


# ═══════════════════════════════════════════════════════════════════════════
# 进度
# ═══════════════════════════════════════════════════════════════════════════


def test_progress() -> None:
    from app.ai.service import Progress

    p = Progress()
    snap = p.snapshot()
    eq("开始是一条 progress 事件", snap["type"], "progress")
    eq("初始 0%", snap["pct"], 0.0)
    check("带一句人话", isinstance(snap["stage"], str) and bool(snap["stage"]), snap["stage"])

    p.report(0.4, "构建玩家特征…")
    eq("报上来的进度", p.snapshot()["pct"], 0.4)
    p.report(0.2, "两两遮挡…")
    eq("进度不倒退", p.snapshot()["pct"], 0.4)
    eq("阶段跟着最后一次", p.snapshot()["stage"], "两两遮挡…")
    p.report(2.0, "装不下")
    check("pct 封顶在 1 以下（1.0 留给 result）", p.snapshot()["pct"] < 1.0,
          str(p.snapshot()["pct"]))
    check("还没结束", not p.done)

    p.finish(payload={"ok": True, "round": 0})
    snap = p.snapshot()
    eq("finish 之后是 result", snap["type"], "result")
    eq("payload 原样带出来", snap["payload"]["round"], 0)
    check("done 了", p.done)

    q = Progress()
    q.report(0.3, "逐帧推理…")
    q.finish(error="这个 .csz 不是逐 tick 的", kind="features")
    snap = q.snapshot()
    eq("出错是 error 事件", snap["type"], "error")
    eq("错误里带着 kind", snap["kind"], "features")
    check("错误文本原样带出来", "逐 tick" in snap["error"], snap["error"])


# ═══════════════════════════════════════════════════════════════════════════
# 缓存
# ═══════════════════════════════════════════════════════════════════════════


def test_cache() -> None:
    from app.ai import service

    tmp = scratch()
    try:
        csz = tmp / "a.csz"
        csz.write_bytes(b"x" * 32)
        data = {"meta": {"map": "de_mirage"}, "format": "cs2.demo.v7",
                "rounds": [{"ticks": [1, 2, 3]}]}
        k1 = service.demo_key(csz, data)
        eq("同一个文件 → 同一个键", service.demo_key(csz, data), k1)

        csz.write_bytes(b"x" * 64)                  # size 变了
        k2 = service.demo_key(csz, data)
        check("文件变了 → 键也变", k2 != k1, f"{k1} vs {k2}")

        os.utime(csz, (1, 1))                       # mtime 变了（= 换了文件）
        k3 = service.demo_key(csz, data)
        check("mtime 变了 → 键也变", k3 != k2, f"{k2} vs {k3}")

        # 没有文件时退回"形状"，同一形状确定同一个键
        d1 = service.demo_key(None, data)
        eq("没有 .csz 时按形状回退（确定性）", service.demo_key(None, data), d1)
        check("形状键有前缀，不会和文件键混", d1.startswith("shape-"), d1)

        path = service.cache_path("demoX", "modelY", 16, 3, root=tmp / "ai")
        check("缓存文件名带 demo / 模型 / stride / 回合",
              path.name == "modelY_s16_r3.json", str(path))
        eq("缓存路径在 root/<demo>/ 下", path.parent.name, "demoX")

        payload = {"version": service.CACHE_VERSION, "ok": True, "round": 3,
                   "samples": [1, 2], "duel": {"a": [None, 1]}}
        service._write_cache(path, payload)
        back = service._read_cache(path)
        check("写进去能读回来，并且标记 cached", bool(back) and back["cached"] is True)
        eq("内容没变", back["round"], 3)

        bad = dict(payload, version=service.CACHE_VERSION + 1)
        service._write_cache(path, bad)
        eq("版本不对 → 当作没有缓存", service._read_cache(path), None)

        path.write_text("{not json", encoding="utf-8")
        eq("坏 JSON → 当作没有缓存", service._read_cache(path), None)

        eq("clear_cache 数对了", service.clear_cache(root=tmp / "ai"), 1)
        check("clear 之后目录没了", not (tmp / "ai").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pinned_models() -> None:
    from app.ai import models

    # 1) 版本开关就是这张表：两个任务、都指向扁平的 `<run>.pt`（旁边配 `<run>.yaml`）
    eq("写死了两个任务", sorted(models.PINNED_WEIGHTS), ["aim", "spatial_prediction"])
    eq(
        "aim 指向 20261119-013008.pt",
        models.PINNED_WEIGHTS["aim"],
        models.MODELS_DIR / "aim" / "20261119-013008.pt",
    )
    eq(
        "spatial 指向 20261113-112930.pt",
        models.PINNED_WEIGHTS["spatial_prediction"],
        models.MODELS_DIR / "spatial_prediction" / "20261113-112930.pt",
    )
    check("用的是 .pt", all(p.suffix == ".pt" for p in models.PINNED_WEIGHTS.values()))

    # 2) 扁平布局：`<run>.pt` + `<run>.yaml`，name 取文件名（缓存键里的 run）
    tmp = scratch()
    try:
        stem = tmp / "20260101-000000"
        stem.with_suffix(".pt").write_bytes(b"w")
        stem.with_suffix(".yaml").write_text("model: {}\n", encoding="utf-8")
        ref = models.pinned_run("aim", weights=stem.with_suffix(".pt"))
        eq("name 就是 run 名", ref.name, "20260101-000000")
        eq("config 取同名的 .yaml", ref.config, stem.with_suffix(".yaml"))
        eq("signature 记下权重文件名", ref.signature()["weights"], "20260101-000000.pt")
        eq("signature 记下 run", ref.signature()["run"], "20260101-000000")
        key = models.signature_key(ref, ref)
        eq("signature_key 是确定的", models.signature_key(ref, ref), key)
        check("signature_key 是短哈希", len(key) == 12, key)

        # 2a) 同样的文件被重写一遍（`hf_hub_download` 就是这么干的）不该换 key：
        #     缓存键跟着内容走，不跟 mtime 走。否则用户重下一遍权重，全部历史
        #     AI 缓存会集体 miss（文件还在，但模型指纹变了，一个都读不到）。
        os.utime(stem.with_suffix(".pt"), (1, 1))
        os.utime(stem.with_suffix(".yaml"), (1, 1))
        eq("mtime 变了但内容没变 → signature_key 不变", models.signature_key(ref, ref), key)
        stem.with_suffix(".yaml").write_text("model: {a: 1}\n", encoding="utf-8")
        check("config 内容变了 → signature_key 变",
              models.signature_key(ref, ref) != key, "换了 config 却还是同一个 key")
        eq("signature 记下内容指纹", len(ref.signature()["digest"]), 16)

        # 2b) 训练目录的老布局也认（`<run>/last.pt` + `<run>/config.yaml`）
        legacy = tmp / "20260102-000000"
        legacy.mkdir()
        (legacy / "last.pt").write_bytes(b"w")
        (legacy / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        old = models.pinned_run("aim", weights=legacy / "last.pt")
        eq("老布局 name 用目录名", old.name, "20260102-000000")
        eq("老布局 config 用 config.yaml", old.config, legacy / "config.yaml")

        # 3) 写错了要报出**哪条路径**，不能静默退回别的 run
        try:
            models.pinned_run("aim", weights=tmp / "gone" / "last.pt")
            check("权重不存在时抛 ModelUnavailable", False)
        except models.ModelUnavailable as exc:
            check("权重不存在时抛 ModelUnavailable", "gone" in str(exc), str(exc))
        (tmp / "lonely.pt").write_bytes(b"w")
        try:
            models.pinned_run("aim", weights=tmp / "lonely.pt")
            check("缺 config 时抛 ModelUnavailable", False)
        except models.ModelUnavailable as exc:
            check("缺 config 时抛 ModelUnavailable", "yaml" in str(exc), str(exc))
        try:
            models.pinned_run("depth")
            check("没登记的任务抛 ModelUnavailable", False)
        except models.ModelUnavailable as exc:
            check("没登记的任务抛 ModelUnavailable", "PINNED_WEIGHTS" in str(exc),
                  str(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 4) 表里写的那两个文件真的在（`models/` 不进 git，所以按目录在不在这判定）
    if models.MODELS_DIR.is_dir():
        for task in ("aim", "spatial_prediction"):
            ref = models.pinned_run(task)
            check(f"仓库里有 {task} 的权重", ref.weights.is_file(), str(ref.weights))
            check(f"… 且配置在权重旁边", ref.config.is_file(), str(ref.config))


# ═══════════════════════════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════════════════════════


def test_routes() -> None:
    import visualizer.server as viz
    from flask import Flask

    from app.ai import service
    from app.ai.api import bp as ai_bp
    from app.ai.features import AiFeaturesError
    from app.ai.models import ModelUnavailable

    app = Flask(__name__)
    app.register_blueprint(ai_bp)
    client = app.test_client()

    saved = viz.session_snapshot
    saved_predict = service.predict_round
    saved_status = service.Predictor.instance

    try:
        # 1) 没有会话
        viz.session_snapshot = lambda: None
        r = client.get("/api/ai/round/0")
        eq("没有会话 → 400", r.status_code, 400)

        # 2) 正常返回
        viz.session_snapshot = lambda: {"data": {"rounds": [{"ticks": [1]}]},
                                        "csz_path": None, "source": "x"}
        calls = {}

        def fake_predict(data, idx, *, csz_path=None, stride=service.DEFAULT_STRIDE,
                         refresh=False):
            calls.update(idx=idx, stride=stride, refresh=refresh, csz=csz_path)
            return {"ok": True, "round": idx, "samples": [1, 2], "big": "x" * 100}

        service.predict_round = fake_predict
        r = client.get("/api/ai/round/7?refresh=1&stride=32")
        eq("正常 → 200", r.status_code, 200)
        eq("刷新参数透传", calls["refresh"], True)
        eq("stride 透传", calls["stride"], 32)
        eq("回合号透传", calls["idx"], 7)
        eq("响应是 JSON", r.is_json, True)
        check("大一点的响应也带 no-store", r.headers.get("Cache-Control") == "no-store",
              str(r.headers.get("Cache-Control")))

        # 3) 模型不可用 → 503
        def boom_models(*a, **k):
            raise ModelUnavailable("没有权重")

        service.predict_round = boom_models
        r = client.get("/api/ai/round/0")
        eq("模型不可用 → 503", r.status_code, 503)
        eq("… 带上 kind=models", r.get_json().get("kind"), "models")

        # 4) 数据不合规 → 400
        def boom_features(*a, **k):
            raise AiFeaturesError("不是逐 tick 的")

        service.predict_round = boom_features
        r = client.get("/api/ai/round/0")
        eq("数据不合规 → 400", r.status_code, 400)
        eq("… 带上 kind=features", r.get_json().get("kind"), "features")

        # 7) `?progress=1`：NDJSON 流（若干 progress，最后一条 result）
        def stream_predict(data, idx, *, csz_path=None,
                           stride=service.DEFAULT_STRIDE, refresh=False, progress=None):
            import time as _t

            assert progress is not None, "流式路径必须把 progress 传下去"
            progress(0.1, "构建玩家特征…")
            _t.sleep(0.2)                   # 让生成器至少推一次，别整条流一次吐完
            progress(0.7, "逐帧推理…")
            return {"ok": True, "round": idx, "samples": [1], "streamed": True}

        service.predict_round = stream_predict
        r = client.get("/api/ai/round/3?progress=1")
        eq("流式预测 → 200", r.status_code, 200)
        eq("… 是 NDJSON", r.mimetype, "application/x-ndjson")
        eq("… 不让中间层缓冲", r.headers.get("X-Accel-Buffering"), "no")
        lines = [json.loads(x) for x in r.data.decode("utf-8").splitlines() if x.strip()]
        eq("第一条是进度事件", lines[0]["type"], "progress")
        check("进度事件带 stage / pct / elapsed",
              all(k in lines[0] for k in ("stage", "pct", "elapsed")), str(lines[0]))
        eq("最后一条是结果", lines[-1]["type"], "result")
        eq("结果里带完整 payload", lines[-1]["payload"]["streamed"], True)
        pcts = [e["pct"] for e in lines if e["type"] == "progress"]
        check("进度是递增的", pcts == sorted(pcts), str(pcts))

        # 8) 流式路径上的错误也走同一条流（HTTP 头早就发出去了，只能靠事件报错）
        def boom_stream(*a, progress=None, **k):
            raise ModelUnavailable("没有权重")

        service.predict_round = boom_stream
        r = client.get("/api/ai/round/3?progress=1")
        eq("流式失败也是 200（错误在流里）", r.status_code, 200)
        lines = [json.loads(x) for x in r.data.decode("utf-8").splitlines() if x.strip()]
        eq("… 最后一条是 error", lines[-1]["type"], "error")
        eq("… 带 kind=models", lines[-1]["kind"], "models")
        check("… 有错误文本", "权重" in (lines[-1].get("error") or ""), str(lines[-1]))

        # 5) status 路由：把 Predictor 换成一个不会加载模型的桩
        class FakePredictor:
            def status(self):
                return {"ok": True, "stride": 16, "loaded": False,
                        "runs": {"aim": {"run": "A"}, "spatial": {"run": "S"}}}

        service.Predictor.instance = classmethod(lambda cls: FakePredictor())
        r = client.get("/api/ai/status")
        eq("status → 200", r.status_code, 200)
        eq("status 里带 stride", r.get_json().get("stride"), 16)
        check("status 有耗时字段", "ms" in r.get_json(), str(r.get_json()))

        # 6) status 路由：目录都没有时也是一句人话（200 + ok=false）
        def no_models(cls):
            raise ModelUnavailable("models 目录不存在")

        service.Predictor.instance = classmethod(no_models)
        r = client.get("/api/ai/status")
        eq("没有产物 → 200（不是 500）", r.status_code, 200)
        eq("… ok=false", r.get_json().get("ok"), False)
    finally:
        viz.session_snapshot = saved
        service.predict_round = saved_predict
        service.Predictor.instance = saved_status


def test_cache_migration() -> None:
    """模型指纹从 mtime 换成内容之后，老前缀的缓存要被搬到新前缀。

    这是"用户重下一遍权重，AI 缓存全没了"的补救：磁盘上那些文件还是同一份权重
    的结果，不搬就得再冷启动一次（一场比赛几分钟）。
    """
    from app.ai import models, service

    tmp = scratch()
    try:
        files = {}
        for tag, body in (("aim", b"a" * 64), ("sp", b"s" * 64)):
            w = tmp / f"{tag}.pt"
            y = tmp / f"{tag}.yaml"
            w.write_bytes(body)
            y.write_text("model: {}\n", encoding="utf-8")
            files[tag] = (w, y)
        aim = models.RunRef(task="aim", name="aim", weights=files["aim"][0],
                            config=files["aim"][1])
        sp = models.RunRef(task="spatial_prediction", name="sp",
                           weights=files["sp"][0], config=files["sp"][1])

        old = service._legacy_model_key(aim, sp)
        check("旧公式算得出指纹", bool(old) and len(old) == 12, str(old))

        root = tmp / "ai"
        demo = root / "demoX"
        demo.mkdir(parents=True)
        for name in (f"{old}_s16_r0.json", f"{old}_s16_match.json",
                     "other_s16_r0.json"):
            (demo / name).write_text("{}", encoding="utf-8")

        saved_root = service.cache_root
        service.cache_root = lambda: root
        try:
            service._migrate_legacy_model_prefix(aim, sp, "newkey000001")
            names = sorted(p.name for p in demo.iterdir())
            eq("老前缀的文件都搬到新前缀",
               names, ["newkey000001_s16_match.json", "newkey000001_s16_r0.json",
                       "other_s16_r0.json"])

            # 幂等：再跑一次没有东西可搬，也不该报错
            service._migrate_legacy_model_prefix(aim, sp, "newkey000001")
            eq("再跑一次什么都不变", sorted(p.name for p in demo.iterdir()), names)
        finally:
            service.cache_root = saved_root
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stream_work_shares_one_job() -> None:
    """同一个 subject 的两次请求 = 一个任务，第二个跟着看。

    回放页和玩家页、或者两个浏览器点同一个回合时，**不能**排两次推理：
    模型只有一份，多跑一次就是几十秒的 CPU/显存白烧。任务本身进 AI 队列，
    所以库页面的「任务」抽屉能看到它。
    """
    print("\nstream_work 去重 + 任务可见")
    import threading

    from flask import Flask

    from app.ai import queue as ai_queue
    from app.ai.api import stream_work

    app = Flask(__name__)
    subject = ("test", "shared", uuid.uuid4().hex)
    release = threading.Event()
    calls = {"n": 0}

    def work(say) -> dict:
        calls["n"] += 1
        say(0.2, "构建玩家特征…")
        release.wait(timeout=10.0)
        return {"ok": True, "value": 7}

    @app.get("/work")
    def _work():
        return stream_work(work, name="t", label="测试 AI 任务", subject=subject)

    results: list[list[dict]] = []

    def client_call() -> None:
        with app.test_client() as c:
            body = c.get("/work").data.decode("utf-8")
        results.append([json.loads(x) for x in body.splitlines() if x.strip()])

    t1 = threading.Thread(target=client_call, name="w1")
    t1.start()
    # 等第一个请求把任务挂上（worker 是懒启动的），再发第二个。
    import time as _t

    for _ in range(100):
        if ai_queue.manager.list():
            break
        _t.sleep(0.02)
    t2 = threading.Thread(target=client_call, name="w2")
    t2.start()
    _t.sleep(0.3)                      # 让第二个请求走到 find()/attach
    release.set()
    t1.join(timeout=20)
    t2.join(timeout=20)

    eq("两次请求只跑了一次推理", calls["n"], 1)
    eq("两个请求都拿到了结果", [r[-1]["type"] for r in results], ["result", "result"])
    eq("结果一样", [r[-1]["payload"]["value"] for r in results], [7, 7])
    check("每个流都先推过进度",
          all(any(e["type"] == "progress" for e in r) for r in results),
          str([[e["type"] for e in r] for r in results]))

    jobs = [j for j in ai_queue.manager.list()
            if j["subject"] == list(subject)]
    eq("队列里只有一个任务", len(jobs), 1)
    eq("任务类型是 ai（抽屉按它分组）", jobs[0]["kind"], "ai")
    eq("标题带上算什么", jobs[0]["label"], "测试 AI 任务")
    eq("算完了", jobs[0]["state"], "done")
    eq("结果挂在任务上（下一次点开是缓存）", jobs[0]["result"]["value"], 7)


# ═══════════════════════════════════════════════════════════════════════════
# 整回合遮挡：稀疏构建 + 磁盘缓存
# ═══════════════════════════════════════════════════════════════════════════

#: `view_render` 把 three 空间的 OBJ 换成 game 空间用的比例（`(gx,gy,gz)=(tz,tx,ty)/S`）。
_OBJ_SCALE = 0.0254


def _wall_obj(path: Path) -> Path:
    """一面 ``x = 0`` 的墙（game 空间），写成 OBJ。

    OBJ 是 three 空间：``(tx,ty,tz) = (gy,gz,gx)·S``，和 `scripts/extract_maps.py`
    拆出来的地图同轴 —— 这样稀疏构建和精确算法看的是同一面墙，比较才有意义。
    """
    s = _OBJ_SCALE
    game = [
        (0.0, -200.0, -16.0),
        (0.0, 200.0, -16.0),
        (0.0, 200.0, 96.0),
        (0.0, -200.0, 96.0),
    ]
    lines = [f"v {gy * s:.6f} {gz * s:.6f} {gx * s:.6f}" for gx, gy, gz in game]
    lines += ["f 1 2 3", "f 1 3 4"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _wall_round(frames: int = 120) -> dict:
    """2 个玩家：0 号一直站在墙左边，1 号从左穿到右（穿墙，只为制造一次单调翻转）。"""
    xs = np.linspace(60.0, -60.0, frames)
    z = [0.0] * frames

    def player(x) -> dict:
        return {
            "x": list(x),
            "y": z,
            "z": z,
            "crouch": z,
            "crouching": [False] * frames,
            "alive": [True] * frames,
        }

    return {
        "id": 0,
        "ticks": list(range(1000, 1000 + frames)),
        "round_seconds": [i / 64.0 for i in range(frames)],
        "players": [player([-60.0] * frames), player(xs)],
        "smoke": [],
        "fire": [],
        "events": {},
    }


def test_build_round_sparse() -> None:
    """`build_round` 和精确逐帧算法在"单调变化"的回合上必须逐位相同。"""
    from scripts.training_data import obstruction as obst

    d = scratch()
    obj = _wall_obj(d / "wall.obj")
    saved = obst.load_map
    obst.load_map = lambda _name: saved("tt_wall", obj_path=str(obj))
    try:
        rnd = _wall_round()
        exact = obst.Occluder(rnd, "tt_wall", 2).window(0, len(rnd["ticks"]))
        sparse = obst.build_round(rnd, "tt_wall", 2)

        eq("形状和精确一致", sparse.shape, exact.shape)
        check("确实有被墙挡死的帧", int((exact[:, 0, 1] == 2).sum()) > 0)
        check("也确实有通的帧", int((exact[:, 0, 1] == 0).sum()) > 0)
        eq("单调穿越：稀疏 == 精确", int((sparse != exact).sum()), 0)

        # 锚点稀疏到 16 帧，二分仍能定位到那一帧（变化只有一次，不会"翻了又翻回来"）
        eq("stride=16 也精确", int((obst.build_round(rnd, "tt_wall", 2, stride=16) != exact).sum()), 0)
        # 对角线永远是 0
        eq("对角线是 0", int(np.abs(np.einsum("tii->ti", sparse)).sum()), 0)

        # 死亡：死人的行 / 列必须是 0，且和精确值一致
        rnd2 = _wall_round()
        rnd2["players"][1]["alive"][60:] = [False] * 60
        exact2 = obst.Occluder(rnd2, "tt_wall", 2).window(0, 120)
        sparse2 = obst.build_round(rnd2, "tt_wall", 2)
        eq("有死亡的回合：稀疏 == 精确", int((sparse2 != exact2).sum()), 0)
        eq(
            "死人的行 / 列全是 0",
            int(np.abs(sparse2[60:, 1, :]).sum() + np.abs(sparse2[60:, :, 1]).sum()),
            0,
        )
    finally:
        obst.load_map = saved


def _smoke_instance(eid: int, first: int, last: int, lo: int, det: list[float]) -> dict:
    """一颗最小的烟：8 个种子点围成小立方体（凸包退化不了）。

    journal 的字节布局和 `tools/demo_extract/tests/test_roundtrip.py` 里的
    `_keyframe` / `_record` 一样：8 字节一个种子 + 空的 mask 表。
    """
    from demo_extract import voxel

    seeds = [(lo + dx, lo + dy, lo + dz)
             for dx in (0, 4) for dy in (0, 4) for dz in (0, 4)]
    payload = bytearray([0, voxel.KEYFRAME, len(seeds)])
    for x, y, z in seeds:
        payload += bytes([x, y, z, 0, 0, 0, 0, 0])
    payload += struct.pack("<H", 0)
    j = struct.pack("<HH", 0, len(payload)) + bytes(payload)
    return {
        "entity_id": eid,
        "detonation": det,
        "first_tick": first,
        "last_tick": last,
        "capacity": 3072,
        "changes": [{"tick": first, "size": len(j), "update": 0,
                     "mode": voxel.MODE_APPEND, "data": j}],
    }


def test_smoke_volumes_cache() -> None:
    """烟体积缓存必须跟着存活期走，且能区分同回合复用 entity_id 的两颗烟。

    两条都是"错了不报错、只把 `.obs` 悄悄算错"的 bug（实测：补上存活期检查后重算
    单个回合，有数百帧、上千个玩家对的标签从 1 变回 0，且全部落在场上没有存活烟的
    时段）：

    * `voxel._record_index()` 只做二分，`tick > last_tick` 时照样返回最后一条记录
      的序号 —— 缓存键里没有存活期，"烟散了"就会命中"烟还在"的那一格，绕过
      `journal_at()` 的守卫。
    * 键里没有 `first_tick`，同回合复用的 entity_id 会互相覆盖（`rec` 是各自
      `changes` 下标，天然会撞）。
    """
    from demo_extract import voxel
    from scripts.training_data import obstruction as obst

    a = _smoke_instance(7, 1000, 1010, 8, [100.0, 200.0, 50.0])
    b = _smoke_instance(7, 2000, 2010, 20, [-300.0, 400.0, 80.0])
    rnd = {"smoke": [a, b]}

    eq("存活期内有一颗烟", obst.smoke_volumes(rnd, 1010)[0].shape[0], 1)
    eq("散了之后一颗都不该有", obst.smoke_volumes(rnd, 1011)[0].shape[0], 0)
    check("journal_at 在 last_tick+1 是 None", voxel.journal_at(a, 1011) is None)
    check("_record_index 在 last_tick+1 仍是最后一条",
          voxel._record_index(a, 1011) == voxel._record_index(a, 1010))

    cache: dict = {}
    eq("先查存活期最后一帧", obst.smoke_volumes(rnd, 1010, cache)[0].shape[0], 1)
    eq("再查消散之后（缓存不许命中）", obst.smoke_volumes(rnd, 1011, cache)[0].shape[0], 0)

    cache2: dict = {}
    eq("先查第一颗烟", obst.smoke_volumes(rnd, 1005, cache2)[0].shape[0], 1)
    det, _ = obst.smoke_volumes(rnd, 2005, cache2)
    eq("第二颗烟生效时只剩它自己", det.shape[0], 1)
    check("拿到的是第二颗烟的爆点", bool(np.allclose(det[0], b["detonation"])),
          f"{det[0].tolist()} vs {b['detonation']}")


def test_obstruction_cache() -> None:
    """缓存：写完能读、对不上就重建、命中不碰几何、删 csz 一起清。"""
    from scripts.training_data import obstruction as obst
    from scripts.training_data import obstruction_cache as oc
    from app import obstruction as app_obst

    d = scratch()
    csz = d / "demo__ab12cd34.csz"
    csz.write_bytes(b"not a real csz")
    rnd = {"id": 3, "ticks": list(range(40))}
    arr = (np.arange(40 * 10 * 10, dtype=np.int8) % 3).reshape(40, 10, 10)

    calls = {"n": 0}
    # `ensure()` 是**调用时**才从 obstruction 里取 build_round 的（为了不把 numba
    # 拖进"只想摸文件"的调用方），所以补丁打在 obstruction 上，不是打在这里。
    saved_build = obst.build_round

    def fake_build(_rnd, _map, _n, **_kw):
        calls["n"] += 1
        return arr

    obst.build_round = fake_build
    saved_map = obst.load_map
    try:
        got = oc.ensure(csz, rnd, "de_mirage", 10)
        eq("没有缓存 → 建一次", calls["n"], 1)
        check("返回的数组和算出来的一致", np.array_equal(got, arr))
        check("缓存文件落盘了", oc.round_path(csz, 3).is_file())

        # 命中缓存时**不许碰几何**：把 load_map 换成"一调就炸"
        obst.load_map = lambda _n, **_kw: (_ for _ in ()).throw(RuntimeError("不许碰几何"))
        got2 = app_obst.ensure_round(csz, rnd, "de_mirage")
        eq("第二次 → 不再算", calls["n"], 1)
        check("读回来的数组一致", np.array_equal(got2, arr))
        obst.load_map = saved_map

        check("has() 认这个缓存", oc.has(csz, 3, 40, map_name="de_mirage"))
        check("帧数对不上 → 不认", not oc.has(csz, 3, 41, map_name="de_mirage"))
        check("帧数对不上 → load 返回 None", oc.load(csz, 3, 41, map_name="de_mirage") is None)
        check("地图名换了 → 不认（basis 变了）",
              not oc.has(csz, 3, 40, map_name="de_anubis"))

        # 算法版本一变，老缓存自动作废
        oc.CACHE_VERSION += 1
        check("CACHE_VERSION +1 → 不认", not oc.has(csz, 3, 40, map_name="de_mirage"))
        oc.CACHE_VERSION -= 1

        # 坏文件 = 没命中，不是异常
        oc.round_path(csz, 3).write_bytes(b"garbage")
        check("坏缓存 → 当没命中", oc.load(csz, 3, 40, map_name="de_mirage") is None)
        oc.save(csz, 3, arr, map_name="de_mirage")
        check("覆盖之后又能读", np.array_equal(oc.load(csz, 3, 40, map_name="de_mirage"), arr))

        # 整个 demo 一轮：两个回合，一个命中一个新建
        data = {
            "meta": {"map": "de_mirage"},
            "rounds": [rnd, {"id": 4, "ticks": list(range(40))}],
        }
        info = app_obst.build_demo(csz, data)
        eq("build_demo 覆盖两个回合", info["rounds"], 2)
        eq("其中一个命中缓存", info["cached"], 1)
        eq("另一个新建", info["built"], 1)
        eq("没有失败", info["failed"], [])
        check("第二个回合也有缓存文件", oc.round_path(csz, 4).is_file())

        # 删 csz 时缓存目录一起没
        app_obst.discard(csz)
        csz.unlink()
        check("discard 之后目录没了", not oc.cache_dir(csz).exists())
        check("discard 之后再 load → None", oc.load(csz, 3, 40, map_name="de_mirage") is None)
    finally:
        obst.build_round = saved_build
        obst.load_map = saved_map


def main() -> int:
    print("app/ai 单元测试")
    for name, fn in [
        ("duel_windows", test_duel_windows),
        ("round end kill", test_round_end_kill),
        ("assign_duels", test_assign_duels),
        ("aim_arrays", test_aim_arrays),
        ("spatial_arrays", test_spatial_arrays),
        ("progress", test_progress),
        ("cache", test_cache),
        ("cache migration", test_cache_migration),
        ("models pinned", test_pinned_models),
        ("routes", test_routes),
        ("stream_work jobs", test_stream_work_shares_one_job),
        ("build_round sparse", test_build_round_sparse),
        ("smoke volumes cache", test_smoke_volumes_cache),
        ("obstruction cache", test_obstruction_cache),
    ]:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _failed.append(f"{name} raised {type(exc).__name__}")
            print(f"  \u2717 {name} raised {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{_passed} passed, {len(_failed)} failed")
    for name in _failed:
        print(f"  FAILED: {name}")
    return 0 if not _failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
