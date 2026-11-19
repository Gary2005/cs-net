"""打包后的自检：``cs-net.exe --packaging-selftest``。

这不是单元测试，是**给 PyInstaller 产物做的体检**。它跑在冻结后的 exe 里，专门
抓那些"本地源码跑得好好的、打进包就少东西"的故障 —— 这类故障不会在 `run_all.ps1`
里出现，只会在玩家点开 AI 面板时变成一个 500：

* PyInstaller 是按 `import` 静态分析收模块的，`torch` / `numba` / `demoparser2`
  这些带原生扩展的包经常要隐藏 import 才收得全；
* `app/ai/features.py` 在模块级就 `from scripts.training_data.tasks.aim import _Round`，
  它背后串着 numba、地图几何、`.csz` 解码 —— 少一个环，整个 AI 面板 500，
  但库页面和 3D 回放照常，很容易漏掉；
* 模板 / 静态资源是 `datas` 收的，路径写错了页面会 404 而不是抛异常。

检查四组：

1. **重依赖 import**：torch、numba、llvmlite、demoparser2、polars、pyarrow、
   zstandard、PIL、pandas
2. **两条最容易漏的链**：`demo_extract.extract`（解压）与 `app.ai.features`（AI）；
   外加真解析一张 demo —— demoparser2 在 Rust 里 import polars / pyarrow，
   PyInstaller 看不见，只有真解析一次能证明它们在包里
3. **真的 HTTP 服务**：随机端口起服务，打 `/healthz`、`/library`、`/players`、
   `/replay`、`/api/library/list`、`/api/library/maps`、`/api/ai/models`、`/api/maps`
4. **状态报告（不算失败）**：数据目录 / 地图解压进度 / 权重在不在 / side-channel

退出码：0 全部硬检查通过。最后打印一行 ``SELFTEST_JSON {...}``，CI 直接读。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

_passed = 0
_checks: list[dict[str, Any]] = []


def _record(name: str, ok: bool, detail: str = "", hard: bool = True) -> None:
    global _passed
    if ok:
        _passed += 1
    _checks.append({"name": name, "ok": ok, "detail": detail, "hard": hard})
    mark = "\u2713" if ok else ("\u2717" if hard else "-")
    tail = f"  {detail}" if detail else ""
    print(f"  {mark} {name}{tail}")


def _attempt(name: str, fn: Callable[[], Optional[str]], *, hard: bool = True) -> None:
    try:
        detail = fn() or ""
        _record(name, True, detail, hard)
    except Exception as exc:  # noqa: BLE001 - 自检要把失败本身变成一行输出
        _record(name, False, f"{type(exc).__name__}: {exc}", hard)


def _ensure_tools_on_path() -> None:
    """开发态 ``demo_extract`` 躺在 ``tools/`` 下，不是装好的包。

    冻结态它是包内模块（``_internal/demo_extract``），``_internal/tools`` 根本
    不存在，这个函数什么都不做。
    """
    from . import paths

    tools = paths.RESOURCES_DIR / "tools"
    if tools.is_dir() and str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


# ── 1. 重依赖 ───────────────────────────────────────────────────────────────


def _import_version(module: str, attr: str = "__version__") -> Callable[[], str]:
    def check() -> str:
        mod = __import__(module)
        return f"{getattr(mod, attr, '?')}"

    return check


def _check_demo_extract() -> str:
    from demo_extract import codec, extract, sidechannel, voxel  # noqa: F401

    binary = sidechannel.binary_path()
    if not binary.is_file():
        # 开发机没跑过 build_sidechannel.sh 是常事，但在**发行包**里这等于核心功能废了
        # （火/烟的字段只有 Go 辅助程序给得了），所以只有冻结态才算硬失败。
        from . import paths

        if paths.is_frozen():
            raise FileNotFoundError(
                f"发行包里没有 side-channel 辅助程序：{binary}"
            )
        return f"binary={binary.name} present=False（开发机没编译，打包时由 CI 提供）"
    return f"binary={binary.name} present=True"

def _check_ai_chain() -> str:
    # 这一条会拖进 torch、numba、地图几何、训练侧的数据结构；缺任何一个都在这儿炸，
    # 而不是等玩家点开 AI 面板。
    from app.ai import features, models, service  # noqa: F401

    return f"pinned weights={len(models.PINNED_WEIGHTS)}"


def _selftest_demo() -> Optional[Path]:
    """自检要真解析的那张 demo：``CSNET_SELFTEST_DEMO`` > 仓库里的 examples/demo。

    发行包里不带 demo（每个 300 MB），所以 zip 里这条自动跳过；CI 会显式指一张，
    那才是"打出来的包真能解析"的证据。
    """
    env = os.environ.get("CSNET_SELFTEST_DEMO", "").strip()
    if env:
        return Path(env)
    from . import paths

    if paths.is_frozen():
        return None
    for candidate in sorted((paths.RESOURCES_DIR / "examples" / "demo").glob("*.dem")):
        return candidate
    return None


def _check_demo_parse(demo: Path) -> str:
    """真的让 demoparser2 解析一次，拿回一张表。

    这是**唯一**能挡住 "polars / pyarrow 没进包" 的检查：demoparser2 是纯 Rust
    扩展，它在 Rust 里 ``PyImport_ImportModule("polars")``，PyInstaller 的静态分析
    完全看不见 —— 于是源码环境里一切正常，玩家点"提取"却拿到
    ``ModuleNotFoundError: No module named 'polars'``（0.42.2 上真的发生过）。
    """
    from demoparser2 import DemoParser

    if not Path(demo).is_file():
        raise FileNotFoundError(demo)
    parser = DemoParser(str(demo))
    header = parser.parse_header()
    df = parser.parse_ticks(wanted_props=["tick"])
    rows = len(df)
    if rows <= 0:
        raise RuntimeError(f"parse_ticks 返回 0 行（map={header.get('map_name')}）")
    return f"{Path(demo).name}: {rows} 行（{type(df).__name__}）"


def _check_numba_parallel() -> str:
    """真的编译并跑一个 ``parallel=True`` 内核。

    只 ``import numba`` 证明不了什么：并行层（tbb / omp / workqueue）是运行时才
    选、才加载的，PyInstaller 少收一个 DLL 在这里才会暴露。遮挡矩阵
    （``obstruction.py::_pair_kernel``）就是 ``parallel=True``，它起不来 = AI 面板
    第一次点就 500。
    """
    import numpy as np
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def _total(a):
        acc = 0.0
        for i in prange(a.shape[0]):
            acc += a[i]
        return acc

    got = float(_total(np.ones(1000, dtype=np.float64)))
    if abs(got - 1000.0) > 1e-6:
        raise RuntimeError(f"并行内核算错了：{got}")
    return "parallel kernel ok"


# ── 2. 状态（不算失败）───────────────────────────────────────────────────────


def _report_state() -> None:
    from app import config, paths

    _record("路径", True, f"frozen={paths.is_frozen()} portable={paths.is_portable()}", hard=False)
    _record("数据目录", True, str(config.DATA_DIR), hard=False)
    _record("地图目录", True, str(config.MAPS_DIR), hard=False)
    _record("权重目录", True, str(paths.MODELS_DIR), hard=False)

    try:
        from scripts.extract_maps import inspect as inspect_maps

        state = inspect_maps(config.MAPS_DIR)
        maps = state.get("maps", [])
        zips = state.get("zips", 0)
        if zips == 0:
            # 发行包里没有 `*.obj.zip`（207 MB 没必要带），地图是解压好直接发的。
            # 这时 inspect() 只能说"目录里有 0 个 zip"，所以自己数一遍 .obj。
            objs = sorted(Path(config.MAPS_DIR).glob("*.obj"))
            _record("地图", bool(objs),
                    f"{len(objs)} 个 .obj 已就位（随包解压，无 zip）", hard=False)
        else:
            complete = sum(1 for m in maps if m.get("exists"))
            _record("地图", complete == zips,
                    f"{complete}/{zips} 解压（{state.get('unpacked_bytes', 0) // (1 << 20)} MB）",
                    hard=False)
    except Exception as exc:  # noqa: BLE001
        _record("地图", False, f"{type(exc).__name__}: {exc}", hard=False)

    try:
        from app.ai.models import PINNED_WEIGHTS

        missing = [p.name for p in PINNED_WEIGHTS.values() if not p.is_file()]
        _record("权重", not missing,
                "两份都在" if not missing else f"缺 {', '.join(missing)}（lite 版可以缺）",
                hard=False)
    except Exception as exc:  # noqa: BLE001
        _record("权重", False, f"{type(exc).__name__}: {exc}", hard=False)


# ── 3. HTTP ─────────────────────────────────────────────────────────────────


def _fetch(url: str, timeout: float = 20.0) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def _http_checks(base: str) -> None:
    def json_ok(path: str) -> Callable[[], str]:
        def check() -> str:
            status, body = _fetch(base + path)
            payload = json.loads(body.decode("utf-8"))
            if status != 200:
                raise RuntimeError(f"HTTP {status}")
            if not isinstance(payload, (dict, list)):
                raise RuntimeError(f"不是 JSON 对象：{type(payload).__name__}")
            return f"HTTP {status}"

        return check

    def page_ok(path: str, marker: str) -> Callable[[], str]:
        def check() -> str:
            status, body = _fetch(base + path)
            text = body.decode("utf-8", "replace")
            if status != 200:
                raise RuntimeError(f"HTTP {status}")
            if marker not in text:
                raise RuntimeError(f"页面里没有 {marker!r}（模板丢了？）")
            return f"HTTP {status}"

        return check

    _attempt("GET /healthz", json_ok("/healthz"))
    _attempt("GET /library", page_ok("/library", "cs-net"))
    _attempt("GET /players", page_ok("/players", "cs-net"))
    _attempt("GET /replay", page_ok("/replay", "cs-net"))
    _attempt("GET /api/library/list", json_ok("/api/library/list"))
    _attempt("GET /api/library/maps", json_ok("/api/library/maps"))
    _attempt("GET /api/maps", json_ok("/api/maps"))

    def models_agree() -> str:
        """权重在本地，`/api/ai/models` 就必须说 ok。

        这条专门挡 full 版那个 bug：权重躺在 exe 旁边，但 `download_models`
        默认根目录还按 `__file__` 算成了 `_internal/`，于是横幅永远喊
        "缺 4 个文件、合计约 665 MB"。
        """
        from app.ai.models import PINNED_WEIGHTS

        _status, body = _fetch(base + "/api/ai/models")
        payload = json.loads(body.decode("utf-8"))
        local = all(Path(p).is_file() for p in PINNED_WEIGHTS.values())
        if local and not payload.get("ok"):
            missing = ", ".join(payload.get("missing") or [])
            raise RuntimeError(f"权重就在本地，接口却说缺：{missing}")
        return f"ok={payload.get('ok')} 本地权重={'在' if local else '未下载'}"

    _attempt("GET /api/ai/models 与磁盘一致", models_agree)


def run(verbose: bool = False) -> int:
    """跑完整自检，返回进程退出码。"""
    # 冻结后的 exe 不认 PYTHONIOENCODING，在 GBK 控制台上 print ✓ 会抛
    # UnicodeEncodeError（详见 app/desktop.py::_force_utf8_console）。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    _ensure_tools_on_path()
    print("[packaging selftest]")

    print("  -- 依赖 --")
    for module in ("numpy", "pandas", "flask", "PIL", "zstandard",
                   "demoparser2", "polars", "pyarrow", "numba", "llvmlite",
                   "torch", "huggingface_hub", "pystray"):
        _attempt(f"import {module}", _import_version(module))

    print("  -- 内部链路 --")
    _attempt("demo_extract (解压链)", _check_demo_extract)
    _attempt("app.ai (预测链)", _check_ai_chain)
    _attempt("numba 并行内核", _check_numba_parallel)
    demo = _selftest_demo()
    if demo is None:
        _record("demoparser2 解析（polars/pyarrow）", True,
                "跳过：没有 demo（发行包不带；CI 用 CSNET_SELFTEST_DEMO 指定）",
                hard=False)
    else:
        _attempt("demoparser2 解析（polars/pyarrow）",
                 lambda: _check_demo_parse(demo))

    print("  -- HTTP --")
    srv = thread = None
    try:
        from werkzeug.serving import make_server

        from app.server import create_app

        app = create_app(start_background=False)
        srv = make_server("127.0.0.1", 0, app, threaded=True)
        thread = threading.Thread(target=srv.serve_forever, name="selftest-http", daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{srv.server_port}"
        _http_checks(base)
    except Exception as exc:  # noqa: BLE001
        _record("起 HTTP 服务", False, f"{type(exc).__name__}: {exc}")
    finally:
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:  # noqa: BLE001
                pass

    print("  -- 状态 --")
    _report_state()

    hard_failed = [c["name"] for c in _checks if not c["ok"] and c["hard"]]
    print()
    print(f"{_passed}/{len(_checks)} 项通过")
    if hard_failed:
        print("失败：" + "、".join(hard_failed))
    print("SELFTEST_JSON " + json.dumps(
        {"ok": not hard_failed, "passed": _passed, "total": len(_checks),
         "failed": hard_failed, "checks": _checks},
        ensure_ascii=False,
    ))
    return 0 if not hard_failed else 1


if __name__ == "__main__":
    raise SystemExit(run())
