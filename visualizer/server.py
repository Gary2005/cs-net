#!/usr/bin/env python3
"""
cs-net-v5 — 3D 回放工作台（demo_extract 版）。

这个 server 只做一件事：**把 ``tools/demo_extract`` 提取出来的数据画出来**，
重点是逐 tick 的玩家轨迹、**真实火焰格子**、**真实烟雾体素**。

和 cs-net（旧版）的区别
------------------------
* 数据源只有 ``tools/demo_extract``：``.dem`` 用 ``demo_extract.extract`` 解析，
  ``.csz`` 用 ``demo_extract.codec.load`` 读回。旧 ``demo_parser`` 已完全移除。
* **默认每个 tick 都采**（≈64 Hz），也可以按秒间隔 / tick 步长降采样来对比。
* 没有任何模型推理：checkpoint / 路径预测 / spatial-only / 低概率扫描全部删除。
* 没有 JSON / json.gz 打包下载，只有新的压缩格式 ``.csz``。
* **没有上传页**。回放只能从 Demo 库进（``/replay?batch=<id>``）；以前那个
  拖拽 .dem / 浏览本机文件的独立页面已经删掉，``POST /api/load`` 现在只认
  ``path``，不再收 multipart 文件。

数据怎么送到浏览器
------------------
整份 all-tick 数据的 JSON 有 160+ MB，不可能整包塞给前端。所以：

* ``POST /api/load`` 只在服务端把 demo 解析进内存（后台线程 + 进度轮询），
  返回**摘要**（地图、回合列表、采样方式、烟/火数量……）。
* ``GET /api/round/<idx>`` 按需返回**单个回合**的紧凑载荷：玩家列用
  base64 的 Float32/Int16/Uint8 数组，烟/火用原始 journal 与格子快照。
* 前端切回合时才拉那一回合。

单独跑这个 server（不带 Demo 库）时，``/`` 就是回放页；没有 ``?batch=``
也没有会话时它会显示一屏「去 Demo 库」的引导。

用法::

    conda activate cs-net-v5
    # 完整应用（Demo 库 + 回放），推荐：
    python -m app.server --port 5000
    # 只要回放工作台：
    python visualizer/server.py --host 127.0.0.1 --port 5000

    # 地图默认在 ../cs-net/maps/optimized_obj_files，也可以显式指定：
    python visualizer/server.py --maps-dir /path/to/optimized_obj_files
"""

from __future__ import annotations

import base64
import json
import mimetypes
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from flask import Flask, jsonify, make_response, redirect, render_template, request, send_file

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _PROJECT_ROOT / "tools"
for _p in (str(_PROJECT_ROOT), str(_TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_extract import codec  # noqa: E402
from demo_extract import extract as extract_demo  # noqa: E402

# 绝对导入，不用 `from . import`：这个文件既能当模块（app.server 里）也能当脚本
# （`python visualizer/server.py`）跑，脚本模式下没有包上下文。和上面 demo_extract
# 同样的理由。
from visualizer import stats as stats_mod  # noqa: E402
from visualizer.banner import startup  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = _THIS_DIR / "outputs"


def _resolve_maps_dir(explicit: Optional[str]) -> Optional[Path]:
    """找地图 OBJ 目录。

    优先用命令行显式给的；否则依次尝试仓库内 ``maps/optimized_obj_files``、
    隔壁 ``cs-net`` 仓库的同名目录。找不到时返回 None（前端会退化成
    一块空地 + 网格，不致命）。
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [
        _PROJECT_ROOT / "maps" / "optimized_obj_files",
        _PROJECT_ROOT.parent / "cs-net" / "maps" / "optimized_obj_files",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.obj")):
            return c.resolve()
    return None


# ── base64 列编码 ───────────────────────────────────────────────────────────


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _f32(values) -> str:
    return _b64(np.asarray(values, dtype="<f4"))


def _i16(values) -> str:
    return _b64(np.asarray(values, dtype="<i2"))


def _u8(values) -> str:
    return _b64(np.asarray(values, dtype=np.uint8))


def _bools(values) -> str:
    return _u8(np.asarray(values, dtype=np.uint8))


def _crouch_u8(values) -> str:
    """0..1 crouch amount -> one byte per sample.

    The .csz keeps the full float32; this is transport only.  255 levels over
    the 18 HU of eye travel is 0.071 HU per step, so the eye position the
    browser reconstructs is off by at most 0.035 HU = 0.9 mm.  Quantising here
    keeps the per-round payload at ~78 KB instead of ~420 KB for the biggest
    round, and matches how shots/footsteps/jump/land already travel.

    Missing (a v6 file) reads as all-zero, i.e. standing.
    """
    a = np.asarray(list(values) if values is not None else [], dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0)
    return _u8(np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8))


# ── 会话状态 ────────────────────────────────────────────────────────────────

_lock = threading.Lock()
#: 统计的**计算**锁（和 `_lock` 分开：算一次统计要几秒，不能堵住回放的接口）。
_stats_lock = threading.Lock()

# 当前会话：{"data": dict, "source": str, "csz_path": Path|None, "maps_dir": Path|None}
_session: dict[str, Any] = {}
_round_cache: dict[int, str] = {}  # round idx -> 已 json.dumps 的载荷


def _clear_session() -> None:
    _session.clear()
    _round_cache.clear()


def session_snapshot() -> Optional[dict]:
    """当前会话的**只读**句柄：`{"data", "csz_path", "source"}`，没有会话时 ``None``。

    给 `app.ai` 的预测路由用（`app.ai.api` 从这边拿数据，依赖方向仍然是
    app → visualizer）。返回的 `data` 是那次 `codec.load()` 的整个 dict，调用方
    只许读：视觉层还在用它，改一个字段就会把画面改坏。

    锁只在这一小段里拿 —— 推理要跑好几秒，攥着 `_lock` 会把 `/api/round` 一起
    堵死。`_clear_session()` 只是 `dict.clear()`，之后旧引用仍然是一个完整的
    旧文件，正好够一次"预测完就把结果写进磁盘缓存"。
    """
    with _lock:
        if not _session:
            return None
        data = _session.get("data")
        if data is None:
            return None
        return {
            "data": data,
            "csz_path": _session.get("csz_path"),
            "source": _session.get("source"),
        }


# 后台解析任务：job_id -> {"state", "step", "error", "t0", "source"}
_jobs: dict[str, dict] = {}
# 记录哪些 job 的摘要已经发过（摘要只在第一次 done 轮询里返回）
_jobs_sent: set[str] = set()


def _set_job(job_id: str, **kw) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(kw)


# ── 摘要 / 载荷构建 ─────────────────────────────────────────────────────────


def _summary(data: dict, source: str) -> dict:
    meta = data.get("meta", {}) or {}
    rounds = []
    for i, r in enumerate(data.get("rounds", [])):
        rounds.append(
            {
                "idx": i,
                "id": r.get("id"),
                "teams": r.get("teams"),
                "winner": r.get("winner"),
                "end_reason": r.get("end_reason"),
                "sample_count": len(r.get("ticks", [])),
                "duration": (r.get("round_seconds") or [0])[-1] if r.get("round_seconds") else 0,
                "smoke_count": len(r.get("smoke", [])),
                "fire_count": len(r.get("fire", [])),
            }
        )
    sampled = meta.get("sampled_ticks")
    total = meta.get("total_ticks")
    # spotted（谁看得见谁）是可选列：`--no-spotted` 提取的文件没有。
    # 前端拿它画"选中玩家的可见敌人"，所以要能分辨"没有这一列"和"这一列全是空"。
    first_players = (data.get("rounds") or [{}])[0].get("players") or [{}]
    has_spotted = "spotted" in (first_players[0] or {})
    return {
        "source": source,
        "format": data.get("format"),
        "map": meta.get("map"),
        "tickrate": meta.get("tickrate"),
        "players": data.get("players", []),
        "weapons": data.get("weapons", {}),
        "places": data.get("places", {}),
        "meta": meta,
        "has_spotted": has_spotted,
        "skipped_rounds": data.get("skipped_rounds", []),
        "rounds": rounds,
        "totals": {
            "rounds": len(rounds),
            "sampled_ticks": sampled,
            "total_ticks": total,
            "fire": sum(r["fire_count"] for r in rounds),
            "smoke": sum(r["smoke_count"] for r in rounds),
        },
        "sampling": _sampling_label(meta),
    }


def _sampling_label(meta: dict) -> str:
    if meta.get("stride"):
        return f"stride={meta['stride']}"
    if meta.get("interval"):
        return f"interval={meta['interval']}s"
    return "every tick"


def _list_changes(values, default=None) -> list:
    """变长列表列 → 只在**变化时**记一条 ``{"i": 采样下标, "v": [...]}``。

    `inventory` 和 `spotted` 都是 per-tick 的变长列表，逐采样点原样发太肥
    （10 人 × 几千采样）。两者都是长时间不变的稀疏信号，所以只发变化点，
    前端按二分查找取"最后一个 i <= 当前采样"的值 —— 和 .csz 里的稀疏
    inventory 是同一套办法。

    实测一整场 24 回合的 `spotted` 只有 740 个变化点，压缩后可以忽略不计。
    """
    out = []
    prev = default
    for i, items in enumerate(values or []):
        cur = list(items)
        if cur != prev:
            out.append({"i": i, "v": cur})
            prev = cur
    return out


def _player_payload(p: dict) -> dict:
    n = len(p.get("x") or [])
    flags = np.zeros(n, dtype=np.uint8)
    flags |= (np.asarray(p.get("helmet") or [False] * n, dtype=np.uint8) & 1) << 0
    flags |= (np.asarray(p.get("defuser") or [False] * n, dtype=np.uint8) & 1) << 1
    flags |= (np.asarray(p.get("alive") or [False] * n, dtype=np.uint8) & 1) << 2
    flags |= (np.asarray(p.get("on_ground") or [False] * n, dtype=np.uint8) & 1) << 3
    # v7: FL_DUCKING, i.e. the collision hull has snapped 72 -> 54.  Free bit.
    flags |= (np.asarray(p.get("crouching") or [False] * n, dtype=np.uint8) & 1) << 4

    # inventory / spotted 只在变化时记录（稀疏），重建成本很低
    inv_changes = _list_changes(p.get("inventory"))
    spot_changes = _list_changes(p.get("spotted"))

    return {
        "x": _f32(p.get("x") or []),
        "y": _f32(p.get("y") or []),
        "z": _f32(p.get("z") or []),
        "yaw": _f32(p.get("yaw") or []),
        "pitch": _f32(p.get("pitch") or []),
        "v": _f32(p.get("v") or []),
        "flash": _f32(p.get("flash") or []),
        "flash_alpha": _f32(p.get("flash_alpha") or []),
        "hp": _i16(p.get("hp") or []),
        "armor": _i16(p.get("armor") or []),
        "weapon": _i16(p.get("weapon") or []),
        "place": _i16(p.get("place") or []),
        "shots": _u8(p.get("shots") or []),
        "footsteps": _u8(p.get("footsteps") or []),
        "jump": _u8(p.get("jump") or []),
        "land": _u8(p.get("land") or []),
        "crouch": _crouch_u8(p.get("crouch")),
        "flags": _u8(flags),
        "inventory": inv_changes,
        # 能看到这个玩家的**敌人**编号（逐采样点）。旧文件用 --no-spotted 提取的
        # 话这里是空表，前端就不会画眼睛图标。
        "spotted": spot_changes,
    }


def _events_payload(ev: dict) -> dict:
    """v6 的烟/火区间用 ``s0/s1`` 秒，这里补上 ``ts/te`` 便于前端统一处理。"""
    out = dict(ev or {})
    for key in ("smokes", "infernos"):
        norm = []
        for it in ev.get(key, []) or []:
            d = dict(it)
            d["ts"] = d.get("s0")
            d["te"] = d.get("s1")
            norm.append(d)
        out[key] = norm
    return out


def _fire_payload(rnd: dict) -> list:
    out = []
    for f in rnd.get("fire", []) or []:
        out.append(
            {
                "entity_id": f.get("entity_id"),
                "inferno_type": f.get("inferno_type"),
                "type_name": f.get("type_name"),
                "origin": f.get("origin"),
                "first_tick": f.get("first_tick"),
                "last_tick": f.get("last_tick"),
                "changes": [
                    {
                        "tick": c.get("tick"),
                        "count": c.get("count"),
                        "pos": c.get("pos"),
                        "normal": c.get("normal"),
                        "parent": c.get("parent"),
                        "burning": c.get("burning"),
                    }
                    for c in f.get("changes", []) or []
                ],
            }
        )
    return out


def _smoke_payload(rnd: dict) -> list:
    out = []
    for s in rnd.get("smoke", []) or []:
        out.append(
            {
                "entity_id": s.get("entity_id"),
                "detonation": s.get("detonation"),
                "first_tick": s.get("first_tick"),
                "last_tick": s.get("last_tick"),
                "capacity": s.get("capacity"),
                "changes": [
                    {
                        "tick": c.get("tick"),
                        "size": c.get("size"),
                        "update": c.get("update"),
                        "mode": c.get("mode"),
                        "data": base64.b64encode(c.get("data") or b"").decode("ascii"),
                    }
                    for c in s.get("changes", []) or []
                ],
            }
        )
    return out


def _bomb_position_payload(rnd: dict) -> str:
    """``bomb_position`` 里有 None（没拿在手上）；编成 NaN 塞进 Float32 数组。"""
    flat = []
    for item in rnd.get("bomb_position", []) or []:
        if item is None:
            flat.extend([float("nan")] * 3)
        else:
            flat.extend([float(item[0]), float(item[1]), float(item[2])])
    return _f32(flat)


def _build_round_payload(data: dict, idx: int) -> dict:
    rnd = data["rounds"][idx]
    meta = data.get("meta", {}) or {}
    n = len(rnd.get("ticks", []))
    return {
        "idx": idx,
        "id": rnd.get("id"),
        "teams": rnd.get("teams"),
        "winner": rnd.get("winner"),
        "end_reason": rnd.get("end_reason"),
        "start_tick": rnd.get("start_tick"),
        "tickrate": meta.get("tickrate") or 64.0,
        "sample_count": n,
        "ticks": _b64(np.asarray(rnd.get("ticks", []), dtype="<i4")),
        "bomb_planted_time": rnd.get("bomb_planted_time"),
        "bomb_planted": _bools(rnd.get("bomb_planted", [])),
        "bomb_dropped": _bools(rnd.get("bomb_dropped", [])),
        "bomb_position": _bomb_position_payload(rnd),
        "missing_samples": rnd.get("missing_samples", []),
        "players": [_player_payload(p) for p in rnd.get("players", [])],
        "events": _events_payload(rnd.get("events", {}) or {}),
        "fire": _fire_payload(rnd),
        "smoke": _smoke_payload(rnd),
    }


# ── 后台解析 ────────────────────────────────────────────────────────────────


def _concise_error(exc: BaseException, limit: int = 400) -> str:
    """把异常压成一句人话，给前端 toast 用。

    demo_extract 的 SideChannelError 会把整个 Go panic 栈（几十 KB）塞进 message，
    直接丢给前端只会得到一坨没人看得懂的提示。这里只留最前面的关键几行。
    """
    msg = str(exc).strip() or type(exc).__name__
    lines = [ln.rstrip() for ln in msg.splitlines() if ln.strip()]
    keep: list[str] = []
    for ln in lines:
        # Go 的 goroutine 栈帧对使用者没有意义，遇到就停
        if ln.startswith(("goroutine ", "created by ", "runtime.", "reflect.")):
            break
        if "/pkg/mod/" in ln or ln.startswith(("panic(", "github.com/")):
            continue
        keep.append(ln)
        if len(keep) >= 4:
            break
    text = "\n".join(keep) if keep else lines[0]
    if len(text) > limit:
        text = text[:limit].rstrip() + " …"
    if len(msg) > len(text) + 50:
        text += "\n（完整堆栈在服务端日志）"
    # 这一类错误几乎都是 demoinfocs 太旧，直接给出可操作的提示
    if "unable to find existing entity" in msg or "unexpected team" in msg:
        text += ("\n提示：火/烟辅助程序的 demoinfocs 版本过旧（需要 v5.3.0）。"
                 "重新编译：bash tools/demo_extract/build_sidechannel.sh")
    return text


def _run_load_job(job_id: str, path: Path, source_name: str,
                  interval: Optional[float], stride: Optional[int],
                  maps_dir: Optional[Path]) -> None:
    try:
        if path.suffix.lower() == ".csz":
            _set_job(job_id, state="parsing", step="读取 .csz（新的压缩格式）…")
            t0 = time.time()
            data = codec.load(path)
            csz_path = path
            _set_job(job_id, step=f".csz 读取完成（{time.time() - t0:.1f}s）")
        else:
            kwargs: dict[str, Any] = {}
            if stride:
                kwargs["stride"] = int(stride)
            elif interval:
                kwargs["interval"] = float(interval)
            mode = "every tick" if not kwargs else str(kwargs)
            _set_job(job_id, state="parsing", step=f"demo_extract.extract（{mode}）…")
            t0 = time.time()
            data = extract_demo(str(path), **kwargs)
            parse_s = time.time() - t0
            _set_job(job_id, step=f"解析完成（{parse_s:.1f}s），正在写 .csz…")
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            csz_path = OUTPUTS_DIR / (path.stem + ".csz")
            t1 = time.time()
            codec.save(data, csz_path)
            _set_job(
                job_id,
                step=f"已写 {csz_path.name}（{csz_path.stat().st_size / 1e6:.1f} MB，"
                     f"{time.time() - t1:.1f}s）",
            )
        summary = _summary(data, source_name)
        with _lock:
            _clear_session()
            _session.update(
                {"data": data, "source": source_name, "csz_path": csz_path,
                 "maps_dir": maps_dir}
            )
        _set_job(job_id, state="done", step="就绪", summary=summary)
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        traceback.print_exc()
        _set_job(job_id, state="error", error=_concise_error(exc))


# ── Flask app ───────────────────────────────────────────────────────────────


def create_app(maps_dir: Optional[Path] = None, *, default_page: str = "replay") -> Flask:
    """Build the visualizer Flask app.

    ``default_page`` decides what ``/`` serves.  The library front-end
    (:mod:`app.server`) passes ``"library"`` so ``/`` redirects to the demo
    library and the replay page lives at ``/replay``; on its own the app keeps
    its original behaviour, with the replay page at ``/``.
    """
    # Windows' registry-backed mimetypes database has no entry for .webp, so
    # Flask serves the weapon textures as application/octet-stream. Browsers
    # sniff the content and cope, but three.js's loader and any strict proxy
    # should see the real type.
    mimetypes.add_type("image/webp", ".webp")
    mimetypes.add_type("model/gltf-binary", ".glb")

    app = Flask(
        __name__,
        template_folder=str(_THIS_DIR / "templates"),
        static_folder=str(_THIS_DIR / "static"),
    )
    # Jinja 的 auto_reload 默认跟随 debug（这里是 False），所以**模板改完不重启
    # 就一直是旧的**。这个坑很隐蔽：`/` 的 Cache-Control: no-cache 只解决了浏览器
    # 那一半，服务器端仍在用进程启动时编译好的 index.html —— 于是新加的
    # `$('toggle-xxx')` 拿到 null，`bindEvents()` 直接抛，整个页面挂掉，
    # 而浏览器里看到的是"新 JS + 旧 HTML"。改模板的开发流程必须能生效。
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    resolved_maps = _resolve_maps_dir(str(maps_dir) if maps_dir else None)

    def _maps_now() -> Optional[Path]:
        """现在这个进程该用哪个地图目录（启动时没找到就再找一遍）。

        `resolved_maps` 是**启动时**算的，而 `maps/optimized_obj_files/*.obj`
        并不进 git —— 用户很可能是在 app 已经跑起来之后，点库页面顶上那条
        「解压地图」横幅才把它解出来的。不在这里重算的话，`/api/map/<name>`
        会一直 404，回放里始终是一块空地，非得重启一次才看得见建筑。

        找到之后就记住（`nonlocal`），所以解压完最多每个请求多一次 `glob`，
        之后和原来一样。找不到时返回 None，路由照旧 404。
        """
        nonlocal resolved_maps
        if resolved_maps is None:
            resolved_maps = _resolve_maps_dir(str(maps_dir) if maps_dir else None)
        return resolved_maps

    # ── 页面 ────────────────────────────────────────────────────────────
    def _render_replay():
        found = _maps_now()
        resp = make_response(render_template(
            "index.html",
            maps_available=found is not None,
            maps_dir=str(found) if found else "",
            # 顶栏的导航指向 Demo 库和玩家页，那两个路由是 app.server 注册的。
            # 单独跑这个 server 时它们不存在，所以那时候不画导航 —— 与其给两
            # 个 404 的链接，不如老实只留品牌块。
            in_app=default_page == "library",
        ))
        # 静态文件 Flask 默认就带 Cache-Control: no-cache（会带 ETag 重新校验），
        # 但模板渲染出来的 HTML 没有 —— 浏览器会启发式缓存它。改了 index.html
        # 之后刷新拿到旧页面、而 main.js 已经是新的，就会出现
        # "$('toggle-xxx') is null" 这种莫名其妙的报错。这里补上。
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.route("/replay")
    def replay():
        """回放页。和 `/` 是同一个页面，只是路径固定。"""
        return _render_replay()

    @app.route("/")
    def index():
        if default_page == "library":
            return redirect("/library")
        return _render_replay()

    # ── 浏览器自测回传（仅测试用）────────────────────────────────────────
    #
    # `static/selftest.html` 跑完之后把结果 POST 到这里，测试脚本再 GET 读走。
    #
    # 为什么不直接用 `chrome --dump-dom`：那个必须配 `--virtual-time-budget`，
    # 而虚拟时钟在页面"看起来空闲"时会**瞬间跳到预算末尾** —— 加载 3 MB GLB +
    # 九张 1024² 贴图这种真正异步的活会在完成前就被掐断（实测 sprite 能过、
    # 武器 GLB 必超时）。回传 + 轮询走真实时间，没有这个问题。
    #
    # 只在内存里存最近一次结果，不落盘；带长度上限，避免被当成存储滥用。
    _selftest_result: dict = {}

    @app.route("/api/_selftest", methods=["POST", "GET"])
    def api_selftest():
        if request.method == "GET":
            return jsonify(_selftest_result or {"done": False})
        body = request.get_data(cache=False)
        if len(body) > 512 * 1024:
            return jsonify({"error": "too large"}), 413
        try:
            _selftest_result.clear()
            _selftest_result.update(json.loads(body.decode("utf-8")))
        except (ValueError, UnicodeDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True})

    # ── 地图 ────────────────────────────────────────────────────────────
    @app.route("/api/maps")
    def api_maps():
        found = _maps_now()
        if found is None:
            return jsonify({"maps": [], "dir": None})
        return jsonify(
            {
                "maps": sorted(p.stem for p in found.glob("*.obj")),
                "dir": str(found),
            }
        )

    @app.route("/api/map/<name>")
    def api_map(name: str):
        found = _maps_now()
        if found is None:
            return jsonify({"error": "地图目录未配置（--maps-dir）"}), 404
        safe = Path(name).name
        path = found / f"{safe}.obj"
        if not path.is_file():
            return jsonify({"error": f"no map {safe}"}), 404
        return send_file(path, mimetype="text/plain")

    # 地图的"每三角形颜色"旁车文件（`bake_map_colors.py` 烘的，见
    # 离线资产管线）。OBJ 本身是 Open3D 导出的纯几何，
    # 没有 vt / usemtl，浏览器端靠这个文件把游戏材质的颜色贴回去。
    #
    # 格式：magic "MCLR" + u16 版本 + u16 flags + u32 三角形数 + 每面 3 字节 sRGB。
    # 没有这个文件时返回 404，前端退回原来的水泥灰顶点色 —— 地图仍然能看。
    @app.route("/api/map/<name>/colors")
    def api_map_colors(name: str):
        found = _maps_now()
        if found is None:
            return jsonify({"error": "地图目录未配置（--maps-dir）"}), 404
        safe = Path(name).name
        path = found / f"{safe}.colors.bin"
        if not path.is_file():
            return jsonify({"error": f"no baked colours for {safe}"}), 404
        return send_file(path, mimetype="application/octet-stream")

    # ── 加载（后台任务）─────────────────────────────────────────────────
    @app.route("/api/load", methods=["POST"])
    def api_load():
        # 只认「本机某个 .dem / .csz 的路径」。以前还收 multipart 上传的文件，
        # 那是独立上传页的入口；页面删掉之后这条路没人走了 —— 回放只能从
        # Demo 库进，而库里的 demo 本来就在本机。
        #
        # interval / stride 仍然认：它们是 demo_extract 的采样参数，直接用
        # API 加载一个 .dem 时还有用。界面不再传，因为库里的 .csz 在提取时
        # 就已经把采样定死了。
        body = request.get_json(silent=True) or {}
        raw = request.form.get("path") or body.get("path")
        if not raw:
            return jsonify({"error": "缺少 path"}), 400
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            return jsonify({"error": f"文件不存在: {path}"}), 400
        if path.suffix.lower() not in (".dem", ".csz"):
            return jsonify({"error": f"不支持的文件类型 {path.suffix}"}), 400
        source_name = path.name
        interval = request.form.get("interval") or body.get("interval")
        stride = request.form.get("stride") or body.get("stride")

        try:
            interval_f = float(interval) if interval not in (None, "", "0") else None
            stride_i = int(stride) if stride not in (None, "", "0") else None
        except (TypeError, ValueError):
            return jsonify({"error": "interval / stride 必须是数字"}), 400
        if interval_f is not None and interval_f <= 0:
            interval_f = None
        if stride_i is not None and stride_i <= 1:
            stride_i = None

        job_id = uuid.uuid4().hex[:12]
        with _lock:
            _jobs[job_id] = {
                "state": "queued", "step": "排队中…", "t0": time.time(),
                "source": source_name, "error": None, "summary": None,
            }
        threading.Thread(
            target=_run_load_job,
            args=(job_id, path, source_name, interval_f, stride_i, resolved_maps),
            daemon=True,
        ).start()
        return jsonify({"job_id": job_id, "source": source_name})

    @app.route("/api/load/status")
    def api_load_status():
        job_id = request.args.get("job_id", "")
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return jsonify({"error": "job not found"}), 404
            out = dict(job)
        out["elapsed"] = round(time.time() - out.pop("t0", time.time()), 1)
        # 摘要只发一次，避免轮询重复传大对象
        if out.get("state") == "done" and job_id not in _jobs_sent:
            _jobs_sent.add(job_id)
        elif out.get("state") == "done":
            out.pop("summary", None)
        return jsonify(out)

    # ── 会话 ────────────────────────────────────────────────────────────
    @app.route("/api/session")
    def api_session():
        with _lock:
            if not _session:
                return jsonify({"loaded": False})
            summary = _summary(_session["data"], _session["source"])
            summary["loaded"] = True
            summary["has_csz"] = bool(_session.get("csz_path"))
        return jsonify(summary)

    # ── 整场统计（统计弹窗）──────────────────────────────────────────────
    @app.route("/api/stats")
    def api_stats():
        """整场比赛的统计：每人的合计 + 每回合的击杀/死亡/伤害/承伤 + 新指标。

        在这里算而不是让前端拉全部回合再自己加：回合载荷是按需拉的，24 个回合
        加起来是几十 MB，而这份数据本来就在服务端内存里（见 stats.py 的说明）。
        结果缓存在会话里 —— 弹窗反复开关不该重算，换一场比赛（_clear_session）
        自然就丢了。

        ⚠ **计算必须在 `_lock` 外面**：`build_stats` 要判「穿烟击杀」，第一次会
        导入 numba / 抽几何 / 建地图 BVH（实测 3~8 s，之后每场 0.3~1 s）。攥着会话
        锁做这件事，这十几秒里 `/api/round`、`/api/session` 全被堵死 —— 而统计弹窗
        只是盖在回放页上面的一层，底下的回放还得能跑。锁里只做"取快照"和"写回"。
        """
        with _lock:
            if not _session:
                return jsonify({"error": "还没有加载数据"}), 400
            data = _session["data"]
            source = _session.get("source")
            payload = _session.get("stats")
        if payload is None:
            # 两个窗口同时点开就只算一次；另一个等锁、然后直接拿结果。
            with _stats_lock:
                with _lock:
                    payload = _session.get("stats")
                    stale = _session.get("data") is not data
                if payload is None and stale:
                    return jsonify({"error": "比赛已被替换，请重新打开"}), 409
                if payload is None:
                    payload = stats_mod.build_stats(data)
                    with _lock:
                        if _session.get("data") is data:   # 别把旧比赛的结果写进新会话
                            _session["stats"] = payload
        payload = dict(payload, source=source)
        resp = app.response_class(json.dumps(payload), mimetype="application/json")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/round/<int:idx>")
    def api_round(idx: int):
        with _lock:
            if not _session:
                return jsonify({"error": "还没有加载数据"}), 400
            data = _session["data"]
            rounds = data.get("rounds", [])
            if not (0 <= idx < len(rounds)):
                return jsonify({"error": f"回合越界: {idx}"}), 404
            cached = _round_cache.get(idx)
            if cached is None:
                cached = json.dumps(_build_round_payload(data, idx))
                _round_cache[idx] = cached
        resp = app.response_class(cached, mimetype="application/json")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/download")
    def api_download():
        with _lock:
            path = _session.get("csz_path")
            source = _session.get("source") if _session else None
        if not path or not Path(path).is_file():
            return jsonify({"error": "还没有可下载的 .csz"}), 404
        return send_file(
            path,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name=(Path(source).stem if source else "replay") + ".csz",
        )

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        with _lock:
            had = bool(_session)
            rounds = len(_round_cache)
            _clear_session()
        return jsonify({"cleared": had, "rounds": rounds})

    return app


def banner(host: str, port: int, *, maps: Optional[Path],
           columns: Optional[int] = None, color: Optional[bool] = None) -> str:
    """启动横幅：logo 字符画 + 信息框（对齐 / 版式 / 颜色见 visualizer/banner.py）。"""
    return startup(
        [
            ("Open:", f"http://{host}:{port}/"),
            ("Maps:", str(maps) if maps else "(未找到，回退到空地 + 网格)"),
            ("提取:", "demo_extract（采样在提取时定死，这里只读 .csz）"),
            ("烟/火:", "真实烟雾体素 + 真实火焰格子（demoinfocs side-channel）"),
        ],
        title="cs-net-v5 — demo_extract 3D 回放工作台（逐 tick）",
        columns=columns,
        color=color,
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="cs-net-v5 — demo_extract 3D 回放工作台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--maps-dir", default=None,
                    help="optimized_obj_files 目录；默认自动找仓库内 maps/ 或隔壁 cs-net/maps/")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app = create_app(maps_dir=Path(args.maps_dir) if args.maps_dir else None)
    maps = _resolve_maps_dir(args.maps_dir)
    print()
    print(banner(args.host, args.port, maps=maps))
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
