"""双击启动器：起服务、打开浏览器、在托盘里留一个能退出的图标。

打包成 exe 之后的入口就是它（``cs-net.exe`` → ``app.desktop.main``）。开发态也能
用，和原来的 ``python -m app.server`` 等价，只是多了"自动找端口 + 开浏览器"：

    python -m app.desktop              # 托盘图标 + 开浏览器（双击时的样子）
    python -m app.desktop --window     # 弹那个小窗口（没有托盘时的退路）
    python -m app.desktop --no-window  # 完全没有界面（调试 / CI）
    python -m app.desktop --packaging-selftest   # 打包后的自检，见 app/selftest.py

为什么不用 ``app.run()``
------------------------
``flask.Flask.run`` 没有"停"的接口，打包后的窗口关掉就只剩任务管理器能杀。
这里用 ``werkzeug.serving.make_server``，``shutdown()`` 之后线程会自己退出来，
退出时还能顺手把 jobs 队列和目录监听停干净。

几个刻意的选择
--------------
* **不死磕 5000 端口**：默认 ``--port 0``，让系统分配。5000 在 Windows 上被
  占用是常事（别的东西、另一个实例、上一次没退干净的僵尸）。
* **先确认服务活了再开浏览器**：``/healthz`` 返回 200 才 ``webbrowser.open``，
  否则玩家看到的是"无法连接"，然后以为程序坏了。
* **第二个实例不开服务**：``server.json`` 里记着第一个实例的地址，第二个直接
  把浏览器指过去然后退出。双击两下不会出现两个进程抢同一个库。
* **日志落在数据目录**：``<data>/logs/desktop.log``。``--noconsole`` 之后崩溃是
  静默的，第一次启动失败必须留下能看的东西，外加一个弹窗指过去。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from . import config, paths

log = logging.getLogger("cs-net.desktop")

#: 单实例用的命名互斥体（Windows）。没释放就说明已经有一个在跑了。
_MUTEX_NAME = "Local\\cs-net-v5-single-instance"
_mutex = None

#: 等 /healthz 的上限。冷启动要初始化数据库、起 watcher，实测 1~3 s。
_STARTUP_TIMEOUT = 40.0


# ── 小工具 ──────────────────────────────────────────────────────────────────


def free_port(host: str = "127.0.0.1") -> int:
    """让内核挑一个空闲端口，然后立刻放掉。

    理论上有"放掉之后被别的进程抢走"的窗口，实际窄到可以忽略；真撞上了
    ``make_server`` 会抛 ``OSError``，外面会重试一次。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def local_url(host: str, port: int, page: str = "/library") -> str:
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{shown}:{port}{page}"


def _state_path() -> Path:
    return Path(config.DATA_DIR) / "server.json"


def _health_url(url: str) -> str:
    """``server.json`` 里存的是库页面地址，健康检查要的是同一个 origin 下的 /healthz。

    直接 ``url + "/healthz"`` 会变成 ``/library/healthz``（404），曾经因此把
    "已经有一个实例在跑"误判成"上次没退干净"，还顺手把记录删了。
    """
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}/healthz"


def _healthy(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(_health_url(url), timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _running_instance() -> Optional[str]:
    """已经在跑的那个实例的库页面地址；没有就 None。"""
    try:
        state = json.loads(_state_path().read_text(encoding="utf-8"))
        url = str(state["url"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if _healthy(url):
        return url
    # 上次没退干净留下的记录：删掉，当成"没在跑"。
    try:
        _state_path().unlink()
    except OSError:
        pass
    return None


def _write_state(url: str) -> None:
    try:
        _state_path().parent.mkdir(parents=True, exist_ok=True)
        _state_path().write_text(
            json.dumps({"pid": os.getpid(), "url": url}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:          # 只读目录：单实例功能降级，但服务照起
        log.warning("写不了 %s（%s），单实例检测跳过", _state_path(), exc)


def _clear_state() -> None:
    try:
        _state_path().unlink()
    except OSError:
        pass


def _acquire_single_instance() -> bool:
    """拿到锁返回 True；已经有一个实例在跑返回 False。"""
    global _mutex
    if os.name != "nt":
        return True
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _mutex = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


def _force_utf8_console() -> None:
    """让中文 / ✓✗ 在 GBK 控制台上不炸。

    开发态的 `python -m app.desktop` 能靠 `PYTHONIOENCODING` 兜住，但**冻结后的
    exe 不认那个环境变量**（PyInstaller 的 bootloader 自己配 stdio），在 cp936 的
    PowerShell/cmd 里 print 一个 `✓` 就是 `UnicodeEncodeError` —— 自检第一条就崩，
    看起来像"打包失败"，其实是编码。所以入口一进来就显式改一次。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def setup_logging(verbose: bool = False) -> Path:
    """日志写到 ``<data>/logs/desktop.log``，滚 1 MB × 3。返回日志文件路径。"""
    log_dir = Path(config.DATA_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "desktop.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fh = RotatingFileHandler(path, maxBytes=1 << 20, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(fh)
    # 打包成 --noconsole 之后 stderr 是 None，加 StreamHandler 会炸。
    if sys.stderr is not None and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    ):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(sh)
    return path


def _notify_error(text: str, log_path: Optional[Path]) -> None:
    """没有控制台时，错误得让人看见 —— 一个弹窗 + 日志路径。"""
    body = text + (f"\n\n日志：{log_path}" if log_path else "")
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("cs-net v5 启动失败", body)
        root.destroy()
    except Exception:  # noqa: BLE001 - 弹窗失败也不能再抛
        print(body, file=sys.stderr or sys.stdout)


def _hide_own_console() -> None:
    """双击启动时把附带的那块控制台藏起来。

    包是 console 子系统的（不然 `--packaging-selftest` 没有输出、崩溃也看不到栈），
    但玩家双击时多一块黑窗口很怪。只在**控制台属于本进程独占**时才藏 —— 从终端
    里跑时 ``GetConsoleProcessList`` 会返回 ≥2（终端 + 我们），那时藏掉就把用户
    自己的输出弄没了。CI 里也是这条路径，所以不会影响日志。
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        pids = (ctypes.c_uint * 8)()
        if kernel32.GetConsoleProcessList(pids, 8) != 1:
            return
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.WinDLL("user32").ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:  # noqa: BLE001 - 藏不掉控制台不是错误
        pass


def open_in_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception as exc:  # noqa: BLE001 - 开不了浏览器不影响服务
        log.warning("打开浏览器失败（%s），请手动访问 %s", exc, url)


def open_folder(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except OSError as exc:
        log.warning("打开目录 %s 失败：%s", path, exc)


# ── 资产状态（窗口里显示一行，省得玩家猜"为什么 AI 是灰的"）───────────────


def assets_summary() -> str:
    bits: list[str] = []
    try:
        from scripts.extract_maps import inspect as inspect_maps

        state = inspect_maps(config.MAPS_DIR)
        maps = state.get("maps", [])
        zips = state.get("zips", 0)
        if zips == 0:
            objs = sorted(Path(config.MAPS_DIR).glob("*.obj"))
            bits.append(f"地图 {len(objs)} 张")
        else:
            complete = sum(1 for m in maps if m.get("exists"))
            bits.append(f"地图 {complete}/{zips}" + ("" if complete == zips else "（未解压）"))
    except Exception:  # noqa: BLE001 - 只是展示，坏了就不显示
        pass
    try:
        from .ai.models import PINNED_WEIGHTS

        have = sum(1 for p in PINNED_WEIGHTS.values() if Path(p).is_file())
        bits.append("权重 就绪" if have == len(PINNED_WEIGHTS) else "权重 未下载")
    except Exception:  # noqa: BLE001
        pass
    return " · ".join(bits)


# ── 窗口 ────────────────────────────────────────────────────────────────────


def _ui_icon():
    """托盘 / 窗口用的图标。打包时 `packaging/cs-net.png` 随 datas 进来。

    拿不到就返回 None —— 图标缺失不该让程序起不来（任务栏会用默认图标）。
    """
    path = Path(paths.RESOURCES_DIR) / "packaging" / "cs-net.png"
    try:
        from PIL import Image

        if path.is_file():
            return Image.open(path)
    except Exception as exc:  # noqa: BLE001
        log.debug("图标读不出来（%s）：%s", path, exc)
    return None


def _relax_pystray_filter() -> None:
    """让 pystray 跳过 `ChangeWindowMessageFilterEx`。

    那一步只是"提权进程也还能收到 WM_TASKBARCREATED"的可选优化，但它在**冻结后的
    exe** 里返回 ``ERROR_ACCESS_DENIED``（开发态 python 里没事），pystray 把它当
    致命错误抛出来，于是打包后托盘永远起不来、退回那个小窗口。我们本来就不是提权
    进程，跳过它没有任何副作用；explorer 重启那一幕也用 NIM_ADD 重加一次就够了
    （pystray 自己还有 WM_TASKBARCREATED 的处理）。
    """
    try:
        from pystray._util import win32 as _w

        def _noop(*_args, **_kwargs):
            return 1

        _w.ChangeWindowMessageFilterEx = _noop
    except Exception as exc:  # noqa: BLE001 - 换别的 pystray 版本就静默跳过
        log.debug("没法给 pystray 打补丁（%s），按它本来的路子来", exc)


def show_tray(url: str, summary: str, on_quit) -> bool:
    """托盘图标 + 菜单，**不弹窗口**。返回 False 表示托盘起不来。

    为什么默认走托盘：双击一个本地工具，最想要的是"浏览器打开、然后别烦我"。
    但总得留一个退出入口 —— 没有窗口也没有托盘的进程只能去任务管理器杀，
    那是 bug 级别的体验。
    """
    try:
        import pystray
    except Exception as exc:  # noqa: BLE001
        log.warning("托盘不可用（%s）", exc)
        return False

    image = _ui_icon()
    if image is None:
        log.warning("没有图标，托盘不建")
        return False

    _relax_pystray_filter()

    def _open(icon, item) -> None:
        open_in_browser(url)

    def _data(icon, item) -> None:
        open_folder(Path(config.DATA_DIR))

    def _quit(icon, item) -> None:
        icon.stop()

    icon = pystray.Icon(
        "cs-net",
        image,
        f"cs-net v5 — {url}" + (f"\n{summary}" if summary else ""),
        menu=pystray.Menu(
            pystray.MenuItem("打开界面", _open, default=True),
            pystray.MenuItem("打开数据目录", _data),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", _quit),
        ),
    )
    try:
        # Windows 上 run() 自己起消息循环，阻塞到 icon.stop()。setup 回调在循环
        # 起来之后执行 —— 那时才知道托盘真活了，再藏控制台就不会把唯一的退路弄没。
        def _setup(ic) -> None:
            try:
                ic.visible = True
            except Exception:  # noqa: BLE001
                pass
            _hide_own_console()

        icon.run(setup=_setup)
    except Exception as exc:  # noqa: BLE001
        # 带上栈：托盘在打包后失败过一次（WinError 5），只有 message 没法定位。
        log.warning("托盘跑不起来（%s: %s），退回窗口", type(exc).__name__, exc,
                    exc_info=True)
        return False
    on_quit()
    return True


def _apply_window_icon(root) -> None:
    """窗口左上角 / 任务栏的图标。拿不到就算了，不该因此打不开窗口。"""
    try:
        ico = Path(paths.RESOURCES_DIR) / "packaging" / "cs-net.ico"
        if os.name == "nt" and ico.is_file():
            root.iconbitmap(str(ico))
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import ImageTk

        image = _ui_icon()
        if image is not None:
            photo = ImageTk.PhotoImage(image)
            root.iconphoto(True, photo)
            root._cs_net_icon = photo      # 留住引用，否则会被 GC 掉变空白
    except Exception:  # noqa: BLE001
        pass


def show_window(url: str, summary: str, on_quit) -> None:
    """一个小窗口：显示地址、能开浏览器 / 开数据目录、点关闭就退出。"""
    import tkinter as tk

    root = tk.Tk()
    root.title("cs-net v5")
    _apply_window_icon(root)
    root.resizable(False, False)
    root.configure(padx=18, pady=14)

    tk.Label(root, text="cs-net v5 正在运行", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    link = tk.Entry(root, width=42, relief="flat", justify="left")
    link.insert(0, url)
    link.configure(state="readonly", readonlybackground=root.cget("bg"))
    link.pack(anchor="w", pady=(6, 0))
    if summary:
        tk.Label(root, text=summary, fg="#555").pack(anchor="w", pady=(2, 0))

    stopped = {"done": False}

    def quit_app() -> None:
        if stopped["done"]:
            return
        stopped["done"] = True
        try:
            root.destroy()
        finally:
            on_quit()

    row = tk.Frame(root)
    row.pack(anchor="w", pady=(14, 0))
    tk.Button(row, text="打开界面", width=12, command=lambda: open_in_browser(url)).pack(side="left")
    tk.Button(row, text="打开数据目录", width=12,
              command=lambda: open_folder(Path(config.DATA_DIR))).pack(side="left", padx=8)
    tk.Button(row, text="退出", width=8, command=quit_app).pack(side="left")

    tk.Label(root, text="关闭这个窗口 = 停止服务。", fg="#888").pack(anchor="w", pady=(12, 0))
    root.protocol("WM_DELETE_WINDOW", quit_app)
    # 窗口已经建起来了，藏控制台是安全的：万一 Tk 起不来会在上面抛异常、走控制台
    # 兜底，那条路上不会藏。
    _hide_own_console()
    root.mainloop()


# ── 主流程 ──────────────────────────────────────────────────────────────────


def serve(host: str, port: int):
    """起 HTTP 服务，返回 ``(server, thread, app)``。"""
    from werkzeug.serving import make_server

    from .server import create_app

    app = create_app(start_background=True)
    srv = make_server(host, port, app, threaded=True)
    thread = threading.Thread(target=srv.serve_forever, name="http", daemon=True)
    thread.start()
    return srv, thread, app


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cs-net",
        description="cs-net v5 · Demo 库 + 3D 回放（双击启动器）",
    )
    from . import __version__

    ap.add_argument("--version", action="version", version=f"cs-net {__version__}")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = 让系统挑空闲端口")
    ap.add_argument("--maps-dir", default=None)
    ap.add_argument("--window", action="store_true",
                    help="弹那个小窗口（默认不弹，只在托盘里留一个图标）")
    ap.add_argument("--no-tray", action="store_true",
                    help="不建托盘图标（没给 --window 时会退回窗口，不然就没法退出了）")
    ap.add_argument("--no-window", action="store_true",
                    help="完全没有界面（调试 / CI；Ctrl+C 退出）")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--no-watch", action="store_true", help="不启动目录监听")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--packaging-selftest", action="store_true",
                    help="打包后自检：起服务、打关键路由、import 重依赖，然后退出")
    args = ap.parse_args(argv)

    log_path: Optional[Path] = None
    _force_utf8_console()
    try:
        try:
            log_path = setup_logging(args.verbose)
        except OSError as exc:
            # 只读的数据目录（比如被人装进 Program Files 又没退到 %LOCALAPPDATA%）：
            # 日志没了，但服务该起还得起，至少让 stdout 留着这句话。
            print(f"[cs-net] 日志初始化失败（{exc}），继续启动", file=sys.stderr or sys.stdout)

        if args.packaging_selftest:
            from . import selftest

            return selftest.run(verbose=args.verbose)

        # 单实例：不管有没有界面，第二个都该直接打开浏览器然后退出 —— 两个进程
        # 抢同一个 SQLite 库不会有好事。
        running = _running_instance()
        if running:
            log.info("已经有一个实例在跑：%s", running)
            if not args.no_browser:
                open_in_browser(running)
            return 0

        if not _acquire_single_instance():
            running = _running_instance()
            if running:
                log.info("已经有一个实例在跑：%s", running)
                if not args.no_browser:
                    open_in_browser(running)
                return 0
            # 端口记录丢了但锁还在：让用户自己判断，总比悄悄起第二个好。
            log.warning("检测到另一个 cs-net 正在运行，但没有可用的地址记录")
            return 2

        if args.no_watch:
            config.Config().update({"watch_enabled": False})

        host, port = args.host, args.port or free_port(args.host)
        srv, thread, _app = serve(host, port)
        url = local_url(host, srv.server_port)
        _write_state(url)
        log.info("服务已启动：%s（数据目录 %s）", url, config.DATA_DIR)

        deadline = time.time() + _STARTUP_TIMEOUT
        while time.time() < deadline and not _healthy(url):
            time.sleep(0.2)
        if not _healthy(url):
            raise RuntimeError(f"服务起来了但 /healthz 一直没响应：{url}")

        if not args.no_browser:
            open_in_browser(url)

        def shutdown() -> None:
            log.info("正在退出…")
            try:
                from . import jobs, watcher

                jobs.manager.stop()
                watcher.watcher.stop()
            except Exception as exc:  # noqa: BLE001 - 退出路径上不能再抛
                log.warning("后台任务停止时报错：%s", exc)
            try:
                srv.shutdown()
                srv.server_close()
            except Exception as exc:  # noqa: BLE001
                log.warning("关闭 HTTP 服务时报错：%s", exc)
            _clear_state()
            log.info("已退出")

        summary = assets_summary()

        # 三种界面形态：托盘（默认）> 窗口（--window）> 纯后台（--no-window）。
        # 托盘里没有"窗口"，正是双击一个本地工具时最舒服的样子；但它必须有，
        # 否则进程就没有退出的入口了。
        ui = "none" if args.no_window else ("window" if args.window else "tray")
        if ui == "tray" and args.no_tray:
            ui = "window"

        if ui == "tray":
            if show_tray(url, summary, shutdown):
                return 0
            log.warning("托盘起不来，改用窗口")
            ui = "window"

        if ui == "window":
            try:
                show_window(url, summary, shutdown)
                return 0
            except Exception as exc:  # noqa: BLE001 - 没有 tkinter 就退化成控制台
                log.warning("窗口打不开（%s），退化成控制台模式；Ctrl+C 退出", exc)

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            shutdown()
        return 0
    except Exception as exc:  # noqa: BLE001 - 这里是最后一道防线
        log.exception("启动失败")
        _notify_error(f"{type(exc).__name__}: {exc}", log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
