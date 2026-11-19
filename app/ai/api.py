"""AI 预测的 HTTP 路由。

    GET /api/ai/status         → 权重在哪、什么设备、有没有装 torch（**不加载权重**）
    GET /api/ai/models         → 写死的那几份权重文件在不在（**不联网**）
    POST /api/ai/models/download
                               → 起一个后台任务把缺的权重下下来（走 jobs 队列）
    GET /api/ai/round/<idx>    → 一个回合的完整预测（首次 15~25 秒，之后读缓存）
    GET /api/ai/round/<idx>?progress=1
                               → 同上，但边算边推 NDJSON（一行一个事件，进度条用）
    GET /api/ai/stats          → 整场比赛的 AI 统计（五个平均值，逐人）
    GET /api/ai/stats?progress=1
                               → 同上，边算边推 NDJSON（每场最多几分钟）

为什么要 `?progress=1` 这一条流
-----------------------------
一次冷分析二十几秒（构建特征 + 两两遮挡 + 两次前向），同步请求在这段时间里
一个字节都不发，浏览器只能挂在那儿转圈 —— 用户不知道是卡死了还是要等一分钟。
统计更狠：一整场比赛 24 个回合，冷启动是**几分钟**。所以加了 `progress=1`：
响应立刻开始，每 ~80 ms 推一行

    {"type":"progress","pct":0.42,"stage":"两两遮挡 2048/6828 帧…","elapsed":3.1}
    {"type":"result","payload":{...},"elapsed":23.9}
    {"type":"error","error":"…","kind":"models"}

真的预测跑在 AI 队列的 worker 线程里（`app.ai.queue`），生成器只读任务的
`step` / `progress` 快照。不带这个参数时行为完全不变（纯 JSON），命令行 / 老的
调用方不受影响。

`/api/ai/stats` 的三层缓存
--------------------------
1. 每个回合自己的载荷（`service.predict_round` 的磁盘缓存）—— 统计只是把
   **已经有缓存**的回合读回来，不会重复跑模型；
2. 整场的汇总（`stats.match_cache_path`，一个几十 KB 的 JSON）—— 第二次打开
   统计页是几十毫秒；
3. `?refresh=1` 才会真的重算（换了模型 / 想复核时用）。

为什么这些路由不在 `visualizer/server.py` 里
------------------------------------------
它们要读"当前回放的那场比赛"（`_session["data"]`），而那个 dict 是 visualizer
的。所以这里只向它要一份**只读快照**（`session_snapshot()`），依赖方向仍然是
`app → visualizer`，单独跑 `python visualizer/server.py` 时这套东西根本不会被
import（那台 server 不需要模型也能放回放）。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from flask import Blueprint, Response, jsonify, request

bp = Blueprint("ai", __name__)

#: `payload` 的 JSON 有 130 KB 量级（一个回合 400 多个预测点 × 5 个头），
#: `jsonify` 会先 `json.dumps` 一遍再让 Flask 再 dumps 一遍，没必要。
_NO_STORE = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}

#: 推送间隔。比一帧（16 ms）慢、比人的耐心（1 s）快得多；20 秒的分析大约
#: 250 行，每行几十字节。
_TICK = 0.08

#: 下载权重的任务类型（出现在顶栏的「任务」抽屉里，和提取任务同一套队列）。
MODELS_JOB_KIND = "models"


def _json(payload: dict[str, Any]):
    from flask import current_app

    resp = current_app.response_class(
        json.dumps(payload, ensure_ascii=False), mimetype="application/json"
    )
    resp.headers.update(_NO_STORE)
    return resp


@bp.get("/api/ai/status")
def api_ai_status():
    from . import service
    from .models import ModelUnavailable

    t0 = time.perf_counter()
    try:
        status = service.Predictor.instance().status()
    except ModelUnavailable as exc:  # 连目录都没扫到
        return _json({"ok": False, "reason": str(exc)})
    status["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    if status.get("ok"):
        # 页面刚打开就会调这个接口 —— 趁用户还在挑回合，把 numba 遮挡内核和这张
        # 地图的 BVH 在后台编好（第一次要 ~10 s，不该算在"点一下"的头上）。
        try:
            from visualizer.server import session_snapshot

            snap = session_snapshot()
            map_name = ((snap or {}).get("data") or {}).get("meta", {}).get("map")
        except Exception:  # noqa: BLE001 —— 预热是锦上添花，不许影响状态查询
            map_name = None
        if map_name:
            service.warmup(str(map_name))
    return _json(status)


# ═══════════════════════════════════════════════════════════════════════════
# 权重：本地有没有 / 缺了就下
#
# 两份 `.pt` 合计 665 MB，`models/` 因此不进 git —— 干净检出之后 AI 面板一定
# 不可用。库页面的横幅问的就是这两个路由：**先查本地**（`inspect()` 只碰磁盘，
# 不发请求，页面一打开就调），缺了才让人点按钮，点完把活儿交给 jobs 队列。
#
# 下载本身在 `scripts/download_models.py` 里（命令行 `python
# scripts/download_models.py` 走的是同一个函数），所以这里不重复实现"从哪个
# 站点下、断点续传怎么写"—— 网页和命令行不可能下出不一样的东西。
# ═══════════════════════════════════════════════════════════════════════════


@bp.get("/api/ai/models")
def api_ai_models():
    """本地那几份写死的权重在不在（`app/ai/models.py::PINNED_WEIGHTS`）。

    不联网：`inspect()` 只看文件在不在、像不像一份完整的权重。远端大小是
    下载脚本 `--check` 才关心的事 —— 每次开页面都去探测一遍 HF 会把首屏拖慢
    好几秒，而且国内大概率还连不上。
    """
    from scripts.download_models import inspect

    from app.jobs import manager

    state = inspect()
    # 正在下 / 刚下完的那个任务一起给页面，横幅直接用它的 step + progress，
    # 不必再为"下载进度"造第二套接口。
    state["download"] = next(
        (job for job in manager.list() if job.get("kind") == MODELS_JOB_KIND), None
    )
    return _json(state)


@bp.post("/api/ai/models/download")
def api_ai_models_download():
    """把缺的权重下到 `models/` 里，跑在后台任务里。

    同步做这件事不行：665 MB 在慢网上要几分钟到几十分钟，浏览器的 fetch 会
    先超时，中间还看不到任何进度。所以 POST 立刻返回 `job_id`，页面去
    `/api/library/jobs/<id>` 读 step / progress（顶栏的「任务」抽屉也是同一份）。

    已经齐了就不排队（除非 `?force=1`）；已经有一个在跑就把它还回去 ——
    连点两下按钮不该排出两个 665 MB 的下载。
    """
    from app.jobs import manager
    from scripts.download_models import download, inspect

    force = request.args.get("force") in ("1", "true", "yes")
    if inspect()["ok"] and not force:
        return _json({"ok": True, "job_id": None, "already": True})

    for job in manager.list(active_only=True):
        if job.get("kind") == MODELS_JOB_KIND:
            return _json({"ok": True, "job_id": job["id"], "already": True})

    def run(job):
        # `job.set_step(一句话, 0..1)` 是 jobs 队列的进度 API；下载脚本通过
        # `progress` 回调把同一个动作报上来。
        return download(
            force=force,
            progress=lambda frac, msg: job.set_step(msg, frac),
        )

    job = manager.submit(MODELS_JOB_KIND, "下载 AI 模型", run)
    return _json({"ok": True, "job_id": job.id})


# ═══════════════════════════════════════════════════════════════════════════
# 慢活 → NDJSON 流（回合预测 / 整场统计共用）
# ═══════════════════════════════════════════════════════════════════════════


def stream_work(
    fn: Callable[[Callable[[float, str], None]], Any],
    *,
    name: str = "ai",
    label: str = "AI 分析",
    subject: Any = None,
) -> Response:
    """把一段慢活排进 AI 队列，边算边把进度写成 NDJSON。

    `fn(say)` 是真正的工作，`say(pct, stage)` 就是任务进度；它的返回值作为最后
    一条 `result` 事件的 `payload`。`ModelUnavailable` / `AiFeaturesError` 会带上
    `kind`，其余异常统一成 `kind="internal"` —— HTTP 头早就发出去了，只能在流里
    报错。

    **活是在 AI 队列里跑的**（`app.ai.queue`，一个 worker），所以
    `/library` 顶栏的「任务」抽屉能看到它，和提取任务并排：谁在算什么回合、
    进度多少、失败了为什么，都不必开第二个页面去猜。

    `subject` 相同的请求不会再排一个任务：第二个（第三个…）请求直接**跟着看**
    已有的那个任务。两个浏览器点同一个回合，只会有一次推理。

    不带 `progress=1` 的同步路径不走这里（它直接算，行为和以前一样）。
    """
    from . import queue as ai_queue

    job = ai_queue.find(subject)
    if job is None:
        job = ai_queue.submit(
            label,
            lambda worker: fn(lambda pct, stage: worker.set_step(stage, pct)),
            subject=subject,
        )
        if job is None:                    # 刚好有人抢先提交了同一个 subject
            job = ai_queue.find(subject)

    def events():
        while True:
            if job is None:                # 理论上到不了：只有 subject=None 才会
                yield json.dumps({"type": "error", "error": "AI 任务提交失败",
                                  "kind": "internal", "elapsed": 0.0},
                                 ensure_ascii=False) + "\n"
                return
            snap = job.to_dict()
            state = snap["state"]
            if state == "done":
                yield json.dumps({"type": "result", "payload": snap["result"],
                                  "elapsed": snap["elapsed"]},
                                 ensure_ascii=False) + "\n"
                return
            if state in ("error", "cancelled"):
                yield json.dumps({
                    "type": "error",
                    "error": snap["error"] or "已取消",
                    "kind": snap.get("error_kind") or ("cancelled" if state == "cancelled" else "internal"),
                    "job_id": snap["id"],
                    "elapsed": snap["elapsed"],
                }, ensure_ascii=False) + "\n"
                return
            yield json.dumps({
                "type": "progress",
                "pct": round(float(snap["progress"] or 0.0), 4),
                "stage": snap["step"],
                "job_id": snap["id"],
                "elapsed": snap["elapsed"],
            }, ensure_ascii=False) + "\n"
            time.sleep(_TICK)

    return Response(events(), mimetype="application/x-ndjson", headers=_NO_STORE)


def _wants_stream() -> bool:
    return request.args.get("progress") in ("1", "true", "yes")


def _wants_refresh() -> bool:
    return request.args.get("refresh") in ("1", "true", "yes")


# ═══════════════════════════════════════════════════════════════════════════
# 一个回合
# ═══════════════════════════════════════════════════════════════════════════


@bp.get("/api/ai/round/<int:idx>")
def api_ai_round(idx: int):
    from visualizer.server import session_snapshot

    from . import service
    from .features import AiFeaturesError
    from .models import ModelUnavailable

    snap = session_snapshot()
    if snap is None:
        return jsonify({"error": "还没有加载数据"}), 400

    refresh = _wants_refresh()
    stride = request.args.get("stride", type=int) or service.DEFAULT_STRIDE
    if _wants_stream():
        meta = (snap.get("data") or {}).get("meta") or {}
        # 同一个回合 + 同一个 stride 只算一次：统计页和回放页同时点，第二个跟着看。
        subject = ("ai", "round", service.demo_key(snap.get("csz_path"), snap["data"]),
                   idx, stride, bool(refresh))
        return stream_work(
            lambda say: service.predict_round(
                snap["data"], idx, csz_path=snap.get("csz_path"),
                stride=stride, refresh=refresh, progress=say,
            ),
            name=f"ai-round-{idx}",
            label=f"AI 回合 {idx + 1} · {meta.get('map') or '未知地图'}",
            subject=subject,
        )
    try:
        payload = service.predict_round(
            snap["data"],
            idx,
            csz_path=snap.get("csz_path"),
            stride=stride,
            refresh=refresh,
        )
    except ModelUnavailable as exc:
        return jsonify({"error": str(exc), "kind": "models"}), 503
    except AiFeaturesError as exc:
        return jsonify({"error": str(exc), "kind": "features"}), 400
    return _json(payload)


# ═══════════════════════════════════════════════════════════════════════════
# 整场比赛的 AI 统计
# ═══════════════════════════════════════════════════════════════════════════


@bp.get("/api/ai/stats")
def api_ai_stats():
    """整场比赛的 AI 统计（五个平均值，逐人）+ 每回合明细。

    冷启动 = 这一场每个回合各跑一次预测（几分钟）；有回合缓存时 ≈ 读 24 个
    JSON + 汇总，几十毫秒。整场汇总自己也有磁盘缓存（`match_cache_path`）。
    """
    from visualizer.server import session_snapshot

    from . import service, stats as stats_mod
    from .models import ModelUnavailable

    snap = session_snapshot()
    if snap is None:
        return jsonify({"error": "还没有加载数据"}), 400

    refresh = _wants_refresh()
    stride = request.args.get("stride", type=int) or service.DEFAULT_STRIDE
    data = snap["data"]
    source = snap.get("source")

    try:
        model_key = service.Predictor.instance().model_key
    except ModelUnavailable as exc:
        return jsonify({"error": str(exc), "kind": "models"}), 503

    path = stats_mod.match_cache_path(
        service.demo_key(snap.get("csz_path"), data), model_key, stride
    )
    if not refresh:
        hit = stats_mod.read_match_cache(path, round_version=service.CACHE_VERSION)
        if hit is not None:
            return _json(dict(hit, source=source, elapsed=0.0))

    def run(say) -> dict:
        t0 = time.perf_counter()
        payload = stats_mod.predict_match(
            data, snap.get("csz_path"), stride=stride, refresh=refresh, progress=say,
        )
        stats_mod.write_match_cache(path, payload)
        return dict(payload, source=source,
                    elapsed=round(time.perf_counter() - t0, 1))

    if _wants_stream():
        meta = (data or {}).get("meta") or {}
        subject = ("ai", "match", service.demo_key(snap.get("csz_path"), data),
                   stride, bool(refresh))
        return stream_work(run, name="ai-stats",
                           label=f"AI 整场统计 · {meta.get('map') or '未知地图'}",
                           subject=subject)
    try:
        return _json(run(None))
    except ModelUnavailable as exc:
        return jsonify({"error": str(exc), "kind": "models"}), 503
