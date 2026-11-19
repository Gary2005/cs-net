"""aim 数据集的可视化 web app —— 标准库 HTTP server（不用 Flask）。

    python -m scripts.training_data_visualizer.aim
    python -m scripts.training_data_visualizer.aim --port 5118 --open
    python -m scripts.training_data_visualizer.aim --data-dir data/demos/prepared/aim
    python -m scripts.training_data_visualizer.aim --model-dir models/aim/20261108-131429

为什么不用 Flask
---------------
这个工具的目标是"任何一台有 numpy + Pillow 的机器都能开"：它只读已经写好的
`.npz`，不需要 `demoparser2` / `numba` / 一堆 web 依赖。标准库的
`ThreadingHTTPServer` 对"本机单人点开看数据"完全够用。

接口
----
    GET /                             前端页面
    GET /static/app.js, style.css     前端资源（白名单，只这两个）
    GET /api/samples                  扫描结果 + 每个 split 的数量
    GET /api/sample/<split>/<name>    npz → JSON 载荷（逐帧全场 + 遮挡矩阵 + 道具）
    GET /api/radar/<map>              地图俯视底图 → PNG（没有就 404）
    GET /api/weapon-icons             武器名 → 图标 URL（给逐帧 HUD 用）
    GET /weapon-icons/<file>.png      CS2 HUD 武器图标（直接复用 3D 回放器的素材）
    GET /api/model                    当前加载的模型（目录 / 权重 / 设备 / 参数 / 指标）
    GET /api/predict/<split>/<name>   这条样本的 T 个预测点（每帧一个，见 predict.py）
    GET /api/health                   只有 `{"ok": true}`，给脚本探活用

**没有 `/api/view/...` 了**：样本里不再有 192×256×5 的视角渲染图（那是旧版的首帧
快照），换成逐帧的 `obstruction (T,10,10)` —— 谁挡着谁这件事现在是一个矩阵，直接
塞进 JSON 就行，不需要服务端渲染 PNG。

武器图标**不是** aim 数据集的一部分：npz 里只有 `atk_weapon` / `tgt_weapon` 的
全局武器下标（`constants.WEAPON_NAMES`），图是 3D 回放器那份
`visualizer/static/assets/weapon_icons/`（CS2 官方 HUD 素材，已 git 跟踪）。
这里只做一层"名字 → 文件"的查表转发，所以素材目录不存在时降级成空表，
前端就只显示武器名、不显示图标。

模型预测也是**可选**的：npz 本身和它无关，`predict.py` 里才 import torch。
`--model-dir` 指定目录（默认取 `models/` 下最新的一个 run，用 `best.pt`），
`--no-model` 整个关掉 —— 没有 torch 的机器上其他功能照常。

所有 JSON 都是 `Cache-Control: no-store`；PNG 也一样（样本文件随时会被重新
`prepare`，缓存住就会看到旧图）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import threading
import traceback
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import data as aim_data
from . import predict as aim_predict

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: 仓库根目录（`scripts/training_data_visualizer/aim/server.py` → 上三级）。
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: CS2 HUD 武器图标：和 3D 回放器共用同一份素材，不往本工具里再复制一遍。
WEAPON_ICON_DIR = _REPO_ROOT / "visualizer" / "static" / "assets" / "weapon_icons"

#: 只放行 `weapon_xxx.png` —— 这条正则同时就是目录穿越的防线。
_ICON_NAME_RE = re.compile(r"^weapon_[a-z0-9_]+\.png$")

#: `map.json` 漏登记的条目（文件在、名字没写）。CS2 里 CZ75 的文件名是 `cz75a`；
#: 3D 回放器那份 map.json 里没有它，所以这里单独补一条（不影响那边）。
_ICON_ALIASES = {"CZ75 Auto": "weapon_cz75a_png.png"}

#: 缓存上限。单个样本的载荷约 150 KB（逐帧全场 + 遮挡矩阵），够来回翻几十个样本。
_SAMPLE_CACHE = 6
#: 每条预测约 T×2 个数（不到 4 KB），但推理本身要跑一次模型，多留几条。
_PREDICT_CACHE = 24


# ── 状态 ────────────────────────────────────────────────────────────────────


def load_weapon_icons() -> dict[str, str]:
    """`map.json` 里的"显示名 → 图标文件名" → `{显示名: /weapon-icons/xxx.png}`。

    素材目录缺失 / map.json 坏掉时返回空表（前端就退化成只显示武器名），
    不让一个锦上添花的图标把整页搞挂。
    """
    try:
        raw = json.loads((WEAPON_ICON_DIR / "map.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for name, files in raw.items():
        if not isinstance(files, list):
            continue
        for fname in files:
            if _ICON_NAME_RE.match(str(fname)) and (WEAPON_ICON_DIR / fname).is_file():
                out[str(name)] = f"/weapon-icons/{fname}"
                break
    for alias, fname in _ICON_ALIASES.items():
        if alias not in out and _ICON_NAME_RE.match(fname) and (WEAPON_ICON_DIR / fname).is_file():
            out[alias] = f"/weapon-icons/{fname}"
    return out


class AppState:
    """进程级状态：数据目录 + 模型 + 三层 LRU 缓存。"""

    def __init__(
        self,
        data_dir: Path,
        *,
        use_model: bool = True,
        model_dir: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir
        #: 模型：`use_model=False`（`--no-model`）时连 torch 都不 import；
        #: `model_dir=None` 表示自动挑 `models/` 下最新的一个 run。
        self.use_model = use_model
        self.model_dir = model_dir
        self.lock = threading.Lock()
        self.samples: Optional[list[dict[str, Any]]] = None
        self.sample_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self.radar_cache: dict[str, Optional[bytes]] = {}
        self.predict_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._weapon_icons: Optional[dict[str, str]] = None
        self._predictor: Optional[aim_predict.Predictor] = None

    # -- 武器图标 ----------------------------------------------------------
    def weapon_icons(self) -> dict[str, str]:
        with self.lock:
            if self._weapon_icons is None:
                self._weapon_icons = load_weapon_icons()
            return self._weapon_icons

    # -- 模型 --------------------------------------------------------------
    def predictor(self) -> aim_predict.Predictor:
        """解析并缓存 `Predictor`。**不加载权重**（那是 `info()` / `predict()` 里的事）。"""
        with self.lock:
            if self._predictor is None:
                self._predictor = aim_predict.Predictor(
                    aim_predict.resolve_run_dir(self.model_dir)
                )
            return self._predictor

    def model_info(self) -> dict[str, Any]:
        """`/api/model`：模型目录 / 权重 / 设备 / 参数 / 训练指标。

        用不了时返回 `{"ok": false, "reason": ...}` 而不是报错：页面上要显示一句
        "为什么没有预测"，不能整块变成 500。
        """
        if not self.use_model:
            return {"ok": False, "reason": "启动时加了 --no-model"}
        try:
            return self.predictor().info()
        except aim_predict.ModelUnavailable as exc:
            return {"ok": False, "reason": str(exc)}

    def prediction(self, split: str, name: str) -> dict[str, Any]:
        """`/api/predict/...`：这条样本的 T 个预测点（每帧一个）。按样本 LRU。"""
        key = f"{split}/{name}"
        with self.lock:
            hit = self.predict_cache.get(key)
            if hit is not None:
                self.predict_cache.move_to_end(key)
                return hit

        if not self.use_model:
            raise aim_predict.ModelUnavailable("启动时加了 --no-model")
        path = aim_data.sample_path(self.data_dir, split, name)   # 顺带做路径校验
        out = self.predictor().predict(path)
        with self.lock:
            self.predict_cache[key] = out
            self.predict_cache.move_to_end(key)
            while len(self.predict_cache) > _PREDICT_CACHE:
                self.predict_cache.popitem(last=False)
        return out

    # -- 样本列表 ----------------------------------------------------------
    def sample_list(self) -> list[dict[str, Any]]:
        with self.lock:
            if self.samples is None:
                self.samples = aim_data.scan_samples(self.data_dir)
            return self.samples

    def rescan(self) -> list[dict[str, Any]]:
        with self.lock:
            self.samples = aim_data.scan_samples(self.data_dir)
            self.sample_cache.clear()
            self.predict_cache.clear()
            return self.samples

    # -- 载荷 --------------------------------------------------------------
    def payload(self, split: str, name: str) -> dict[str, Any]:
        key = f"{split}/{name}"
        with self.lock:
            hit = self.sample_cache.get(key)
            if hit is not None:
                self.sample_cache.move_to_end(key)
                return hit

        path = aim_data.sample_path(self.data_dir, split, name)
        payload = aim_data.load_sample(path, split=split, name=name, data_dir=self.data_dir)
        with self.lock:
            self.sample_cache[key] = payload
            self.sample_cache.move_to_end(key)
            while len(self.sample_cache) > _SAMPLE_CACHE:
                self.sample_cache.popitem(last=False)
        return payload

    # -- 雷达 --------------------------------------------------------------
    def radar_png(self, map_name: str) -> Optional[bytes]:
        with self.lock:
            if map_name in self.radar_cache:
                return self.radar_cache[map_name]
        info = aim_data.radar_info(map_name)
        blob: Optional[bytes] = None
        if info is not None:
            try:
                blob = Path(info["file"]).read_bytes()
            except OSError:
                blob = None
        with self.lock:
            self.radar_cache[map_name] = blob
        return blob


# ── HTTP ────────────────────────────────────────────────────────────────────


class AimHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, state: AppState) -> None:
        super().__init__(addr, AimHandler)
        self.state = state

    def handle_error(self, request, client_address) -> None:
        """浏览器关掉 keep-alive 连接时不要打一整段 traceback。

        Chrome/Edge 刷新或关页面时会直接 RST 掉空闲连接，`socketserver` 默认把
        `ConnectionResetError` 当成异常打印。本地工具里这只意味着"用户刷新了"。
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class AimHandler(BaseHTTPRequestHandler):
    server_version = "aim-visualizer/1.0"
    protocol_version = "HTTP/1.1"

    # -- 输出工具 -----------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str,
              *, extra: Optional[dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        raw = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {}
        if len(raw) > 2048 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
            raw = gzip.compress(raw, compresslevel=6)
            headers["Content-Encoding"] = "gzip"
        self._send(status, raw, "application/json; charset=utf-8", extra=headers)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    # -- 路由 ---------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html", "text/html; charset=utf-8")
            if path == "/favicon.ico":
                # 浏览器默认会要 .ico；给它同一张 SVG（Content-Type 已经声明）
                return self._static("favicon.svg", "image/svg+xml")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):], None)
            if path.startswith("/weapon-icons/"):
                return self._weapon_icon(path[len("/weapon-icons/"):])
            if path.startswith("/api/"):
                return self._api(path, query)
            self._error(404, f"没有这个路径: {path}")
        except FileNotFoundError as exc:
            self._error(404, str(exc))
        except ValueError as exc:
            self._error(400, str(exc))
        except aim_predict.ModelUnavailable as exc:
            # 没权重 / 没装 torch：不是故障，是"这条功能用不了"，单独一个状态码。
            self._error(503, str(exc))
        except Exception:  # pragma: no cover - 给前端一句人话 + 服务端堆栈
            traceback.print_exc()
            self._error(500, "服务端异常，完整堆栈见启动这个工具的终端")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _static(self, name: str, content_type: Optional[str]) -> None:
        # 白名单：不接受 `../`、子目录、任意扩展名。
        allowed = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "application/javascript; charset=utf-8",
            "style.css": "text/css; charset=utf-8",
            "favicon.svg": "image/svg+xml",
        }
        if name not in allowed:
            return self._error(404, f"没有这个静态文件: {name}")
        path = STATIC_DIR / name
        if not path.is_file():
            return self._error(404, f"静态文件不存在: {name}")
        self._send(200, path.read_bytes(), content_type or allowed[name])

    def _weapon_icon(self, name: str) -> None:
        """`/weapon-icons/weapon_ak47_png.png` —— 只认 `weapon_*.png` 这一个形状。"""
        if not _ICON_NAME_RE.match(name):
            return self._error(404, f"不是合法图标名: {name}")
        path = WEAPON_ICON_DIR / name
        if not path.is_file():
            return self._error(404, f"图标不存在: {name}")
        self._send(200, path.read_bytes(), "image/png")

    def _api(self, path: str, query: dict[str, list[str]]) -> None:
        state: AppState = self.server.state  # type: ignore[attr-defined]
        parts = [p for p in path.split("/") if p]
        # parts[0] == "api"
        route = parts[1] if len(parts) > 1 else ""

        if route == "health":
            return self._json({"ok": True})

        if route == "samples":
            samples = state.sample_list()
            counts: dict[str, int] = {s: 0 for s in aim_data.SPLITS}
            for s in samples:
                counts[s["split"]] = counts.get(s["split"], 0) + 1
            maps = sorted({s["map"] for s in samples})
            return self._json(
                {
                    "data_dir": str(state.data_dir),
                    "samples": samples,
                    "counts": counts,
                    "total": len(samples),
                    "maps": maps,
                    "radar": {m: aim_data.radar_info(m) is not None for m in maps},
                    "fov": {"h": aim_data.FOV_H_DEG, "v": aim_data.FOV_V_DEG},
                }
            )

        if route == "rescan":
            state.rescan()
            return self._json({"ok": True, "total": len(state.sample_list())})

        if route == "weapon-icons":
            icons = state.weapon_icons()
            return self._json({
                "dir": str(WEAPON_ICON_DIR),
                "count": len(icons),
                "icons": icons,
            })

        if route == "model":
            return self._json(state.model_info())

        if route == "predict" and len(parts) >= 4:
            split, name = parts[2], "/".join(parts[3:])
            return self._json(state.prediction(split, name))

        if route == "sample" and len(parts) >= 4:
            split, name = parts[2], "/".join(parts[3:])
            return self._json(state.payload(split, name))

        if route == "radar" and len(parts) == 3:
            map_name = parts[2]
            if not map_name.replace("_", "").isalnum():
                raise ValueError(f"非法地图名: {map_name!r}")
            png = state.radar_png(map_name)
            if png is None:
                return self._error(404, f"没有 {map_name} 的雷达底图")
            return self._send(200, png, "image/png")

        return self._error(404, f"没有这个接口: {path}")

    # -- 日志 ---------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # 默认格式会带上客户端地址和日期，本地工具看着太吵。只留 "GET /api/... 200"。
        try:
            sys.stderr.write(f"  {args[0]} {args[1]} {args[2]}\n")
        except (IndexError, TypeError):
            pass


# ── 启动 ────────────────────────────────────────────────────────────────────


def _print_banner(host: str, port: int, state: AppState) -> None:
    samples = state.sample_list()
    counts: dict[str, int] = {s: 0 for s in aim_data.SPLITS}
    for s in samples:
        counts[s["split"]] = counts.get(s["split"], 0) + 1
    maps = sorted({s["map"] for s in samples})
    print()
    print("  aim · 训练数据可视化")
    print("  " + "─" * 62)
    print(f"  地址:     http://{host}:{port}/")
    print(f"  数据:     {state.data_dir}")
    print(f"  样本:     {len(samples)} 个"
          f"（train {counts['train']} / valid {counts['valid']} / test {counts['test']}）")
    # 模型只解析目录（不 import torch），真正加载要等第一次预测。
    if not state.use_model:
        print("  模型:     关掉了（--no-model）")
    else:
        try:
            run = aim_predict.resolve_run_dir(state.model_dir)
            weights = aim_predict.weights_of(run)
            print(f"  模型:     {run.name} · {weights.name if weights else '?'}"
                  "（第一次预测时才加载权重）")
        except aim_predict.ModelUnavailable as exc:
            print(f"  模型:     ⚠ {exc}")
    if maps:
        radar = [m for m in maps if aim_data.radar_info(m) is not None]
        print(f"  地图:     {', '.join(maps)}")
        if len(radar) < len(maps):
            missing = [m for m in maps if m not in radar]
            print(f"  雷达:     缺 {' / '.join(missing)}（俯视图退化成网格）")
    else:
        print("  样本:     ⚠ 一个都没扫到，先跑 "
              "`python -m scripts.training_data.prepare --task aim`")
    print("  " + "─" * 62)
    print("  Ctrl+C 退出")
    print()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.training_data_visualizer.aim",
        description="aim 数据集（prepared/*.npz）的逐帧可视化",
    )
    parser.add_argument("--host", default="127.0.0.1", help="默认 127.0.0.1（只给本机看）")
    parser.add_argument("--port", type=int, default=5117)
    parser.add_argument(
        "--data-dir", default=None,
        help=f"样本根目录，默认 {aim_data.DEFAULT_DATA_DIR}",
    )
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument(
        "--model-dir", default=None,
        help="模型目录（含 config.yaml + best.pt / last.pt），"
             "默认取 models/ 和 models/aim/ 下最新的一个 run；也可以只给 run 名字",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="关掉模型预测面板（不加载 torch，页面其余功能照常）",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else aim_data.DEFAULT_DATA_DIR
    if not data_dir.is_dir():
        print(f"⚠ 数据目录不存在: {data_dir}", file=sys.stderr)

    state = AppState(data_dir, use_model=not args.no_model, model_dir=args.model_dir)
    try:
        httpd = AimHTTPServer((args.host, args.port), state)
    except OSError as exc:
        print(f"⚠ 监听 {args.host}:{args.port} 失败：{exc}", file=sys.stderr)
        print("  换一个端口：--port 5118", file=sys.stderr)
        raise SystemExit(1)

    _print_banner(args.host, args.port, state)
    if args.open:
        webbrowser.open(f"http://{args.host}:{args.port}/")

    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n  已退出")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
