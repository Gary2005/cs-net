"""spatial_prediction 数据集的可视化 web app —— 标准库 HTTP server（不用 Flask）。

    python -m scripts.training_data_visualizer.spatial_prediction
    python -m scripts.training_data_visualizer.spatial_prediction --port 5119 --open
    python -m scripts.training_data_visualizer.spatial_prediction \\
        --data-dir data/demos/prepared/spatial_prediction

为什么不用 Flask
---------------
这个工具的目标是"任何一台有 numpy 的机器都能开"：它只读已经写好的 `.npz`，不需要
`demoparser2` / `numba` / 任何 web 依赖。标准库的 `ThreadingHTTPServer` 对"本机单人
点开看数据"完全够用。

接口
----
    GET /                            前端页面
    GET /static/app.js, style.css    前端资源（白名单，只这三个）
    GET /api/samples                 文件清单（每个文件几行样本）+ 每个 split 的数量
    GET /api/sample/<split>/<name>?k=N  这个文件的第 N 行 → JSON 载荷（单帧 + 5 个标签）
    GET /api/radar/<map>             地图俯视底图 → PNG（没有就 404）
    GET /api/weapon-icons            武器名 → 图标 URL（给玩家表用）
    GET /weapon-icons/<file>.png     CS2 HUD 武器图标（复用 3D 回放器的素材）
    GET /api/health                  只有 `{"ok": true}`，给脚本探活用
    GET /api/rescan                  重新扫磁盘（prepare 重跑之后用）

`?k=` 是这个工具和 aim 那版最大的区别：**一个 npz 里装着 K 行样本**（一个回合被随机
抽到的 K 帧，见 `tasks/spatial_prediction.py`），所以要指定行号。`k` 缺省就是 0
（每个文件的第一行），越界返回 400 而不是硬崩。

所有 JSON 都是 `Cache-Control: no-store`；PNG 也一样（样本文件随时会被重新
`prepare`，缓存住就会看到旧图）。JSON 大于 2 KB 且客户端接受 gzip 时压缩。
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

from . import data as sp_data

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: 仓库根目录（`scripts/training_data_visualizer/spatial_prediction/server.py` → 上三级）。
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: CS2 HUD 武器图标：和 3D 回放器共用同一份素材，不往本工具里再复制一遍。
WEAPON_ICON_DIR = _REPO_ROOT / "visualizer" / "static" / "assets" / "weapon_icons"

#: 只放行 `weapon_xxx.png` —— 这条正则同时就是目录穿越的防线。
_ICON_NAME_RE = re.compile(r"^weapon_[a-z0-9_]+\.png$")

#: `map.json` 漏登记的条目（文件在、名字没写）。CS2 里 CZ75 的文件名是 `cz75a`；
#: 3D 回放器那份 map.json 里没有它，所以这里单独补一条（不影响那边）。
_ICON_ALIASES = {"CZ75 Auto": "weapon_cz75a_png.png"}

#: 缓存上限。单帧载荷比 aim 的窗口小一个数量级（~15 KB），但翻页会连着换行，
#: 所以按"样本"缓存，不按文件。
_SAMPLE_CACHE = 24


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
    """进程级状态：数据目录 + 两层 LRU 缓存。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.lock = threading.Lock()
        self.files: Optional[list[dict[str, Any]]] = None
        self.sample_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self.radar_cache: dict[str, Optional[bytes]] = {}
        self._weapon_icons: Optional[dict[str, str]] = None

    # -- 武器图标 ----------------------------------------------------------
    def weapon_icons(self) -> dict[str, str]:
        with self.lock:
            if self._weapon_icons is None:
                self._weapon_icons = load_weapon_icons()
            return self._weapon_icons

    # -- 文件清单 ----------------------------------------------------------
    def file_list(self) -> list[dict[str, Any]]:
        with self.lock:
            if self.files is None:
                self.files = sp_data.scan_files(self.data_dir)
            return self.files

    def rescan(self) -> list[dict[str, Any]]:
        with self.lock:
            self.files = sp_data.scan_files(self.data_dir)
            self.sample_cache.clear()
            return self.files

    # -- 载荷 --------------------------------------------------------------
    def payload(self, split: str, name: str, k: int) -> dict[str, Any]:
        key = f"{split}/{name}#{k}"
        with self.lock:
            hit = self.sample_cache.get(key)
            if hit is not None:
                self.sample_cache.move_to_end(key)
                return hit

        path = sp_data.sample_path(self.data_dir, split, name)
        payload = sp_data.load_sample(path, split=split, name=name, k=k)
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
        info = sp_data.radar_info(map_name)
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


class SpatialHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, state: AppState) -> None:
        super().__init__(addr, SpatialHandler)
        self.state = state

    def handle_error(self, request, client_address) -> None:
        """浏览器关掉 keep-alive 连接时不要打一整段 traceback。"""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class SpatialHandler(BaseHTTPRequestHandler):
    server_version = "spatial-prediction-visualizer/1.0"
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
        except (ValueError, IndexError) as exc:
            # 行号越界 / 非法样本名 / npz 不是这个任务造的：都是"请求有问题"，
            # 不是服务端故障，所以 400 而不是 500。
            self._error(400, str(exc))
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
            files = state.file_list()
            by_split: dict[str, dict[str, int]] = {
                s: {"files": 0, "samples": 0} for s in sp_data.SPLITS
            }
            for f in files:
                bucket = by_split.setdefault(f["split"], {"files": 0, "samples": 0})
                bucket["files"] += 1
                bucket["samples"] += int(f["n"])
            maps = sorted({f["map"] for f in files})
            return self._json(
                {
                    "data_dir": str(state.data_dir),
                    "files": files,
                    "counts": by_split,
                    "total_files": len(files),
                    "total_samples": sum(int(f["n"]) for f in files),
                    "maps": maps,
                    "radar": {m: sp_data.radar_info(m) is not None for m in maps},
                    "labels": list(sp_data.LABELS),
                    "team_names": {str(k): v for k, v in sp_data.TEAM_NAMES.items()},
                    "density": sp_data.DENSITY,
                }
            )

        if route == "rescan":
            state.rescan()
            return self._json({"ok": True, "total_files": len(state.file_list())})

        if route == "weapon-icons":
            icons = state.weapon_icons()
            return self._json({
                "dir": str(WEAPON_ICON_DIR),
                "count": len(icons),
                "icons": icons,
            })

        if route == "sample" and len(parts) >= 4:
            split = parts[2]
            name = "/".join(parts[3:])
            raw_k = (query.get("k") or ["0"])[0]
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                raise ValueError(f"k 必须是整数，收到 {raw_k!r}") from None
            return self._json(state.payload(split, name, k))

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
    files = state.file_list()
    counts = {s: {"files": 0, "samples": 0} for s in sp_data.SPLITS}
    for f in files:
        counts.setdefault(f["split"], {"files": 0, "samples": 0})
        counts[f["split"]]["files"] += 1
        counts[f["split"]]["samples"] += int(f["n"])
    maps = sorted({f["map"] for f in files})
    n_samples = sum(int(f["n"]) for f in files)
    print()
    print("  spatial_prediction · 训练数据可视化")
    print("  " + "─" * 66)
    print(f"  地址:     http://{host}:{port}/")
    print(f"  数据:     {state.data_dir}")
    print(f"  样本:     {n_samples} 帧（{len(files)} 个 npz）"
          f"  train {counts['train']['samples']} / "
          f"valid {counts['valid']['samples']} / test {counts['test']['samples']}")
    print(f"  采样:     density = {sp_data.DENSITY:g} s（造数据那侧的常量，样本里不存）")
    if maps:
        radar = [m for m in maps if sp_data.radar_info(m) is not None]
        print(f"  地图:     {', '.join(maps)}")
        if len(radar) < len(maps):
            missing = [m for m in maps if m not in radar]
            print(f"  雷达:     缺 {' / '.join(missing)}（俯视图退化成网格）")
    else:
        print("  样本:     ⚠ 一个都没扫到，先跑 "
              "`python -m scripts.training_data.prepare --task spatial_prediction`")
    print("  " + "─" * 66)
    print("  Ctrl+C 退出")
    print()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.training_data_visualizer.spatial_prediction",
        description="spatial_prediction 数据集（单帧 + 5 个未来标签）的逐样本可视化",
    )
    parser.add_argument("--host", default="127.0.0.1", help="默认 127.0.0.1（只给本机看）")
    parser.add_argument("--port", type=int, default=5118)
    parser.add_argument(
        "--data-dir", default=None,
        help=f"样本根目录，默认 {sp_data.DEFAULT_DATA_DIR}",
    )
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args(argv)

    data_dir = (
        Path(args.data_dir).expanduser().resolve() if args.data_dir else sp_data.DEFAULT_DATA_DIR
    )
    if not data_dir.is_dir():
        print(f"⚠ 数据目录不存在: {data_dir}", file=sys.stderr)

    state = AppState(data_dir)
    try:
        httpd = SpatialHTTPServer((args.host, args.port), state)
    except OSError as exc:
        print(f"⚠ 监听 {args.host}:{args.port} 失败：{exc}", file=sys.stderr)
        print("  换一个端口：--port 5119", file=sys.stderr)
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
