"""AI 预测的 HTTP 路由。

    GET /api/ai/status         → 权重在哪、什么设备、有没有装 torch（**不加载权重**）
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

真的预测跑在后台线程里，生成器只读一个加锁的 `Progress` 快照。不带这个参数
时行为完全不变（纯 JSON），命令行 / 老的调用方不受影响。

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
import threading
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
# 慢活 → NDJSON 流（回合预测 / 整场统计共用）
# ═══════════════════════════════════════════════════════════════════════════


def stream_work(
    fn: Callable[[Callable[[float, str], None]], Any], *, name: str = "ai"
) -> Response:
    """把一段慢活放到后台线程，边算边把进度写成 NDJSON。

    `fn(say)` 是真正的工作，`say(pct, stage)` 就是 `Progress.report`；它的返回值
    作为最后一条 `result` 事件的 `payload`。`ModelUnavailable` / `AiFeaturesError`
    会带上 `kind`，其余异常统一成 `kind="internal"` —— HTTP 头早就发出去了，
    只能在流里报错。

    路由里同步跑也行（不带 `progress=1` 就是纯 JSON），这里只是把"同一段活"
    的两种跑法分开写，免得两边各写一遍错误分类。
    """
    from . import service
    from .features import AiFeaturesError
    from .models import ModelUnavailable

    box = service.Progress()

    def work() -> None:
        try:
            box.finish(payload=fn(box.report))
        except ModelUnavailable as exc:
            box.finish(error=str(exc), kind="models")
        except AiFeaturesError as exc:
            box.finish(error=str(exc), kind="features")
        except Exception as exc:  # noqa: BLE001 —— 一条人话比 500 页面有用
            box.finish(error=f"{type(exc).__name__}: {exc}", kind="internal")

    threading.Thread(target=work, name=name, daemon=True).start()

    def events():
        while True:
            ev = box.snapshot()
            yield json.dumps(ev, ensure_ascii=False) + "\n"
            if ev["type"] in ("result", "error"):
                return
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
        return stream_work(
            lambda say: service.predict_round(
                snap["data"], idx, csz_path=snap.get("csz_path"),
                stride=stride, refresh=refresh, progress=say,
            ),
            name=f"ai-round-{idx}",
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
        return stream_work(run, name="ai-stats")
    try:
        return _json(run(None))
    except ModelUnavailable as exc:
        return jsonify({"error": str(exc), "kind": "models"}), 503
