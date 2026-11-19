#!/usr/bin/env python3
"""`app/ai` 的单元测试：对枪窗口、窗口切片、进度、缓存、路由。

**不跑真模型**（那是 1.5 GB + 十几秒，属于端到端脚本
`visualizer/tests/test_ai_overlay.mjs` 的事）。这里量的是那些"错了也不报错、
只会悄悄画错"的地方：

* `duel_windows`：只认敌人、不认自杀 / 队友 / 队伍未知、窗口 = 死前 3 秒、
  终点取受害者**最后一个活帧**
* `assign_duels` / `duel_entries`：窗口重叠时归谁、4 秒窗口里死过人就不喂模型
* `RoundTensors.aim_arrays`：补齐、`seconds` 对零、`frame_mask`、窗口边界
* `Progress`：只增不减、finish 之后变成 result / error 事件
* 缓存：键跟着文件**内容**变（mtime 变了不算）、版本不对就作废、写进去能读回来
* 权重：`PINNED_WEIGHTS` 的扁平 `<run>.pt` + 同名 `<run>.yaml`（老布局也认），
  缺了要报出是哪条路径
* 路由：没有会话 / 模型不可用 / 数据不合规 各自的状态码；`?progress=1` 的 NDJSON 流

Run:  python app/tests/test_ai.py
"""

from __future__ import annotations

import json
import os
import shutil
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

    eq("对枪窗口是死前 3 秒", DUEL_LEAD, 3.0)

    # 6 号在 400 帧杀了 3 号 → 窗口 = [400 帧前 3 秒, 398]（3 号最后一个活帧）
    tensors, rnd = _round(deaths={3: 400})
    rnd["events"]["kills"] = [_kill(1000 + 400, 6, 3)]
    duels = duel_windows(tensors, rnd)
    eq("一场击杀 = 一个窗口", len(duels), 1)
    d = duels[0]
    eq("A 端 = 受害者", d.a, 3)
    eq("B 端 = 击杀者", d.b, 6)
    eq("终点 = 受害者最后一个活帧（不是击杀帧）", d.i1, 399)
    eq("起点 = 死前 3 秒（6.234 - 3 = 3.234 s → 第 207 帧）", d.i0, 207)
    check("窗口正好 3 秒", abs((d.t1 - d.t0) - 3.0) < 0.0001, f"{d.t0}..{d.t1}")
    eq("as_dict 里两个端点和帧号都带出去",
       {k: d.as_dict()[k] for k in ("a", "b", "i0", "i1")},
       {"a": 3, "b": 6, "i0": 207, "i1": 399})

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
    from app.ai.service import assign_duels, duel_entries, duel_windows

    tensors, rnd = _round()                       # 600 帧 / 9.375 s，谁都没死
    hi = int(tensors.ticks[-1])
    rnd["events"]["kills"] = [_kill(hi + 1, 6, 3)]
    duels = duel_windows(tensors, rnd)
    eq("回合结束那一枪算一场对枪", len(duels), 1)
    d = duels[0]
    eq("A 端 = 受害者", d.a, 3)
    eq("B 端 = 击杀者", d.b, 6)
    eq("挂在最后一帧上（受害者活到那一帧）", d.i1, tensors.frames - 1)
    eq("起点仍是死前 3 秒（9.359 - 3 = 6.359 s → 第 407 帧）", d.i0, 407)
    check("窗口正好 3 秒", abs((d.t1 - d.t0) - 3.0) < 0.0001, f"{d.t0}..{d.t1}")

    grid = np.arange(0, tensors.frames, 16, dtype=np.int64)
    entries, _ = duel_entries(tensors, grid, assign_duels(duels, grid))
    eq("窗口里的采样点都喂了模型", len(entries), 12)
    check("最后一个采样点也有对枪预测",
          bool(entries) and entries[-1] == (int(grid[-1]), 3, 6), str(entries[-1:]))

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
    from app.ai.service import Duel, assign_duels, duel_entries

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

    # 4 秒窗口里有人死过 → 不喂模型（训练分布里双方全程活着）
    tensors, _ = _round(deaths={3: 340})
    d = Duel(a=3, b=6, i0=200, i1=339, t0=3.1, t1=5.3)
    grid = np.arange(0, 600, 16, dtype=np.int64)
    entries, where = duel_entries(tensors, grid, assign_duels([d], grid))
    inside = [i for i in range(0, 600, 16) if 200 <= i <= 339]
    eq("窗口内的采样点都喂了模型", len(entries), len(inside))
    check("配对是（受害者, 击杀者）", all(e[1:] == (3, 6) for e in entries),
          str(entries[:3]))
    eq("where 是它们在 samples 里的位置", where[0], list(grid).index(inside[0]))

    # 击杀者在窗口里死过 → 这一段整个跳过
    tensors2, _ = _round(deaths={3: 340, 6: 100})
    probe = np.array([288, 296, 304], dtype=np.int64)
    entries2, _ = duel_entries(tensors2, probe, assign_duels([d], probe))
    eq("窗口里死过人就跳过", len(entries2), 0)


# ═══════════════════════════════════════════════════════════════════════════
# 窗口切片
# ═══════════════════════════════════════════════════════════════════════════


def _tensors(frames: int = 300):
    """一个只有形状、内容全 0 的 RoundTensors（`aim_arrays` 只关心形状）。"""
    return _round(frames)[0]


def test_aim_arrays() -> None:
    t = _tensors(300)
    eq("回合第一帧的窗口只有 1 帧", t.window_bounds(0), (0, 1))
    eq("最后一帧的窗口是完整 257 帧", t.window_bounds(299), (43, 300))

    arrays, lengths = t.aim_arrays([(299, 4, 7), (0, 0, 9)])
    eq("补齐到 batch 内最长的窗口", arrays["ctx_player"].shape, (2, 257, 10, 20))
    eq("每个样本自己的真实长度", list(lengths), [257, 1])
    eq("frame_mask 只标真帧", arrays["frame_mask"].sum(axis=1).tolist(), [257, 1])
    eq("pair 按 (A, B) 原样写进去", arrays["pair"].tolist(), [[4, 7], [0, 9]])
    check("seconds 从窗口第一帧对零",
          abs(arrays["seconds"][0, 1] - 1 / 64.0) < 1e-7
          and abs(arrays["seconds"][0, 0]) < 1e-9,
          str(arrays["seconds"][0, :3]))
    eq("补齐帧的武器是 -1（不是沙鹰 0）", int(arrays["ctx_player_weapon"][1, 1, 0]), -1)
    eq("补齐帧的道具 id 是 -1", int(arrays["ctx_projectile_id"][1, 1, 0]), -1)
    eq("dashboard map_id 每行都有", arrays["map_id"].tolist(), [0, 0])

    # 两个窗口都短、长度不同 → 仍然按最长补齐
    arrays, lengths = t.aim_arrays([(5, 1, 2), (9, 1, 2)])
    eq("短窗口各自长度", list(lengths), [6, 10])
    eq("短窗口的 Tmax", arrays["ctx_player"].shape[1], 10)


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
