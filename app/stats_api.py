"""玩家页的两份统计：**传统数据**和 **AI 数据**（每场明细 + 跨场合计）。

为什么要单独一个 blueprint
--------------------------
玩家页看的是"这个人**所有比赛**的平均"：一个选手在库里可能出现在 4 场比赛里，
每场 17~24 个回合。这些数必须跨文件算，而 `app/api.py` 是库的元数据接口
（只读 SQLite），不适合塞一个会解码 `.csz`、会跑模型的端点进来。

三层缓存，代价逐层下降
----------------------
1. **传统统计**（`app/data/stats/<demo_key>.json`）：解码一次 `.csz` + 扫事件
   （含"穿烟"要建地图几何），第一场 5~10 s，之后每场几十毫秒。键是 `.csz` 的
   size+mtime（和 AI 缓存同一套 `service.demo_key`），换了文件自动失效。
2. **AI 统计**（`app/data/ai/<demo>/<model>_s16_match.json`）：一场比赛每个回合各
   跑一次模型，冷启动是**几分钟**；有了回合缓存之后 ≈ 读 N 个 JSON。这一层由
   `app/ai/stats.py` 管，这里只是按 SteamID 取自己那一行。
3. 两个端点都支持 `?progress=1`（NDJSON），页面拿它画进度条 —— 冷启动那几分钟
   不能只让用户看一个转圈。

为什么不在库入库时就算好
------------------------
入库（`app/ingest.py`）只解元数据和记分板，不碰逐 tick 的数据；把"传统统计"
塞进去会让每份 demo 的入库时间翻倍，而绝大多数时候没人看统计。AI 那一侧更
不可能：它依赖当前权重（`PINNED_WEIGHTS`），重训一次就要全部重算。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, request

from . import config, db

bp = Blueprint("stats", __name__)

#: 传统统计缓存文件的格式版本（改了 `build_stats` 的字段就 +1）。
TRAD_VERSION = 1

#: 跨场求和时直接相加的字段（比例和派生值最后重算）。
TRAD_SUM = (
    "kills", "deaths", "assists", "headshots", "damage", "damage_taken",
    "rounds", "enemy_kills", "enemy_deaths", "duels",
    "first_kills", "first_deaths", "openers",
    "side_kills", "side_deaths", "smoke_kills",
)


def _log(msg: str) -> None:
    print(f"[stats] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 磁盘缓存（一场比赛一份）
# ═══════════════════════════════════════════════════════════════════════════


def trad_cache_root() -> Path:
    return Path(config.DATA_DIR) / "stats"


def _demo_key(csz_path: str | Path | None) -> str:
    """和 AI 缓存同一套键（`.csz` 的 size+mtime）；懒 import 是为了不在启动时
    把 torch / numba 那一串拖进来。"""
    from .ai.service import demo_key

    return demo_key(csz_path, {})


def _read(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != TRAD_VERSION:
        return None
    if not payload.get("ok"):
        return None
    return payload


def _write(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # 缓存写不进去不该让统计失败
        _log(f"缓存写入失败（不影响本次结果）：{exc}")


def demo_trad_stats(
    csz_path: str | Path, *, refresh: bool = False, progress=None
) -> dict:
    """一场比赛的传统统计（`visualizer.stats.build_stats` 的结果 + 磁盘缓存）。

    Raises:
        OSError: `.csz` 不见了 / 读不了
        Exception: `build_stats` 自己抛的（数据坏了）
    """
    from demo_extract import codec
    from visualizer.stats import build_stats

    key = _demo_key(csz_path)
    path = trad_cache_root() / f"{key}.json"
    if not refresh:
        hit = _read(path)
        if hit is not None:
            hit["cached"] = True
            return hit
    if progress:
        progress(0.05, "解码 .csz…")
    data = codec.load(str(csz_path))
    if progress:
        progress(0.6, "统计击杀 / 伤害 / 站位…")
    payload = dict(build_stats(data))
    payload.update({"ok": True, "version": TRAD_VERSION, "demo": key, "cached": False})
    _write(path, payload)
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# 传统统计的跨场合计
# ═══════════════════════════════════════════════════════════════════════════


def _pct(num: float, den: float) -> float:
    return round(num * 100.0 / den, 1) if den else 0.0


def sum_trad(rows: list[dict]) -> dict:
    """多场的选手行 → 合计（先加再除，不是平均的平均）。

    `rows` 是 `build_stats()["players"]` 里的那些行（每场一行，同一个选手）。
    """
    acc: dict[str, float] = {k: 0.0 for k in TRAD_SUM}
    for row in rows:
        for key in TRAD_SUM:
            acc[key] += float(row.get(key) or 0.0)
    kills, deaths = acc["kills"], acc["deaths"]
    rounds = acc["rounds"]
    ek, ed = acc["enemy_kills"], acc["enemy_deaths"]
    out = {key: int(val) for key, val in acc.items()}
    out.update({
        "matches": len(rows),
        "kd": round(kills / deaths, 2) if deaths else float(kills),
        "kpr": round(kills / rounds, 2) if rounds else 0.0,
        "adr": round(acc["damage"] / rounds, 1) if rounds else 0.0,
        "hs_pct": _pct(acc["headshots"], kills),
        "opener_pct": _pct(acc["openers"], ek + ed),
        "side_kill_pct": _pct(acc["side_kills"], ek),
        "side_death_pct": _pct(acc["side_deaths"], ed),
        "smoke_kill_pct": _pct(acc["smoke_kills"], ek),
    })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════════════════


def _wants_stream() -> bool:
    return request.args.get("progress") in ("1", "true", "yes")


def _match_label(demo: dict) -> str:
    return str(demo.get("display_name") or demo.get("filename") or f"#{demo['id']}")


def _player_demos(steamid: str, limit: int = 200) -> list[dict]:
    result = db.list_demos(steamid=steamid, sort="date", limit=limit, offset=0)
    return list(result.get("items") or [])


def _run(fn, *, name: str):
    """慢活两种跑法：NDJSON 流（带进度）或纯 JSON。"""
    from .ai.api import stream_work

    if _wants_stream():
        return stream_work(fn, name=name)
    try:
        payload = fn(None)
    except Exception as exc:  # noqa: BLE001
        from .ai.models import ModelUnavailable

        kind = "models" if isinstance(exc, ModelUnavailable) else "internal"
        return jsonify({"error": f"{type(exc).__name__}: {exc}", "kind": kind}), (
            503 if kind == "models" else 500
        )
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.get("/api/library/players/<steamid>/stats")
def api_player_stats(steamid: str):
    """这位选手每一场的**传统统计**明细 + 跨场合计。

    第一次打开会把库里他打过的每场比赛各解码一次（每场 5~10 s，之后走磁盘
    缓存）。`?progress=1` 走 NDJSON，进度按场报。
    """
    demos = _player_demos(steamid)
    refresh = request.args.get("refresh") in ("1", "true", "yes")
    total_n = len(demos)

    def run(say) -> dict:
        t0 = time.perf_counter()
        say_ = say or (lambda _p, _m: None)
        matches: list[dict] = []
        rows: list[dict] = []
        errors: list[dict] = []
        smoke_ok = True
        for i, demo in enumerate(demos):
            label = _match_label(demo)
            csz = demo.get("csz_path")

            def sub(frac: float, msg: str, _i: int = i) -> None:
                say_((_i + min(max(frac, 0.0), 1.0)) / max(1, total_n),
                     f"第 {_i + 1}/{total_n} 场 · {label} · {msg}")

            base = {"demo_id": demo.get("id"), "map": demo.get("map_name"),
                    "date": demo.get("match_date"), "label": label,
                    "rounds": int(demo.get("total_rounds") or 0)}
            if not csz or not Path(csz).is_file():
                errors.append({**base, "error": "这场比赛没有 .csz 文件"})
                matches.append({**base, "stats": None})
                continue
            try:
                st = demo_trad_stats(csz, refresh=refresh, progress=sub)
            except Exception as exc:                  # noqa: BLE001
                _log(f"{label} 统计失败：{type(exc).__name__}: {exc}")
                errors.append({**base, "error": f"{type(exc).__name__}: {exc}"})
                matches.append({**base, "stats": None})
                continue
            row = next((p for p in st.get("players") or []
                        if str(p.get("steamid")) == str(steamid)), None)
            smoke_ok = smoke_ok and bool(st.get("smoke_ok"))
            if row is None:
                errors.append({**base, "error": "这场比赛里没有这位选手"})
                matches.append({**base, "stats": None})
                continue
            rows.append(row)
            matches.append({**base, "cached": bool(st.get("cached")),
                            "rounds": int(st.get("rounds") or base["rounds"]),
                            "stats": row})
            say_((i + 1) / max(1, total_n), f"第 {i + 1}/{total_n} 场 · {label} · 完成")

        return {
            "ok": True,
            "steamid": str(steamid),
            "matches": matches,
            "totals": sum_trad(rows),
            "smoke_ok": bool(smoke_ok and rows),
            "errors": errors,
            "elapsed": round(time.perf_counter() - t0, 1),
        }

    return _run(run, name="player-stats")


@bp.get("/api/library/players/<steamid>/ai")
def api_player_ai(steamid: str):
    """这位选手每一场的 **AI 统计**明细 + 跨场合计。

    ⚠ 第一次调用会**真的跑模型**：库里他打过的每一场、每一回合各一次
    （20 回合 ≈ 5~8 分钟）。之后走 `app/data/ai/` 的磁盘缓存，几十毫秒。
    所以界面上这一栏是**点了「AI 数据」页签才请求**的。
    """
    from .ai import service, stats as ai_stats
    from .ai.models import ModelUnavailable

    demos = _player_demos(steamid)
    stride = request.args.get("stride", type=int) or service.DEFAULT_STRIDE
    refresh = request.args.get("refresh") in ("1", "true", "yes")
    total_n = len(demos)

    try:
        model_key = service.Predictor.instance().model_key
    except ModelUnavailable as exc:
        return jsonify({"error": str(exc), "kind": "models", "state": "unavailable"}), 503

    def run(say) -> dict:
        from demo_extract import codec

        t0 = time.perf_counter()
        say_ = say or (lambda _p, _m: None)
        matches: list[dict] = []
        entries: list[dict] = []
        errors: list[dict] = []
        for i, demo in enumerate(demos):
            label = _match_label(demo)
            csz = demo.get("csz_path")

            def sub(frac: float, msg: str, _i: int = i) -> None:
                say_((_i + min(max(frac, 0.0), 1.0)) / max(1, total_n),
                     f"第 {_i + 1}/{total_n} 场 · {label} · {msg}")

            base = {"demo_id": demo.get("id"), "map": demo.get("map_name"),
                    "date": demo.get("match_date"), "label": label}
            if not csz or not Path(csz).is_file():
                errors.append({**base, "error": "这场比赛没有 .csz 文件"})
                matches.append({**base, "stats": None})
                continue
            key = _demo_key(csz)
            path = ai_stats.match_cache_path(key, model_key, stride)
            hit = None if refresh else ai_stats.read_match_cache(
                path, round_version=service.CACHE_VERSION
            )
            if hit is None:
                sub(0.0, "解码…")
                data = codec.load(str(csz))
                hit = ai_stats.predict_match(data, csz, stride=stride,
                                             refresh=refresh, progress=sub)
                ai_stats.write_match_cache(path, hit)
            row = next((p for p in hit.get("players") or []
                        if str(p.get("steamid")) == str(steamid)), None)
            if row is None:
                errors.append({**base, "error": "这场比赛里没有这位选手"})
                matches.append({**base, "stats": None})
                continue
            entries.append({"totals": row.get("totals") or {},
                            "rounds": int(row.get("rounds") or 0)})
            matches.append({
                **base,
                "cached": bool(hit.get("cached")),
                "rounds": int(row.get("rounds") or 0),
                "stats": {k: v for k, v in row.items() if k != "per_round"},
            })
            say_((i + 1) / max(1, total_n), f"第 {i + 1}/{total_n} 场 · {label} · 完成")

        totals = ai_stats.combine(entries)
        return {
            "ok": True,
            "steamid": str(steamid),
            "matches": matches,
            "totals": totals,
            "stride": stride,
            "probe_sec": ai_stats.KILL_PROBE_SEC,
            "errors": errors,
            "elapsed": round(time.perf_counter() - t0, 1),
        }

    return _run(run, name="player-ai")
