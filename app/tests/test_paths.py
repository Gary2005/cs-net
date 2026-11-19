#!/usr/bin/env python3
"""`app/paths.py`：程序目录 / 数据目录的探测。

这一层唯一的验收标准是"开发态和以前**一字不差**，冻结态按规则落到该落的地方"。
所以这里的检查都绕着同一个问题：把 ``sys.frozen`` / ``sys.executable`` /
``%LOCALAPPDATA%`` 换成打包后的样子，**不真的打包**，看它选哪个目录。

量的是这些：

* 开发态：``BUNDLE_DIR`` 是仓库根、数据在 ``app/data``、地图在 ``maps/`` 下
  —— 和 `app.config` 里那几个常量必须完全相等（下游把它们绑进默认参数了）
* 冻结态 + exe 旁边能写：数据、地图、权重都在 exe 旁边（绿色版，整目录拷走）
* 冻结态 + exe 旁边不能写（模拟 Program Files）：数据退到 ``%LOCALAPPDATA%``，
  权重跟着退，地图**不**退（它是随包发的资源，只读也能用）
* ``CSNET_DATA_DIR`` 谁都得让路
* 探测本身不留垃圾：写完要删掉探针文件，重复调用结果一致

Run:  python app/tests/test_paths.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
# `demo_extract` 不是安装好的包，就躺在 tools/ 下（app/ingest.py 也是这么加的）。
sys.path.insert(0, str(_ROOT / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from app import config, paths  # noqa: E402

_passed = 0
_failed: list[str] = []
_skipped: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  \u2713 {name}")
    else:
        _failed.append(name)
        print(f"  \u2717 {name}" + (f"\n      {detail}" if detail else ""))


def eq(name: str, got: object, want: object) -> None:
    check(name, got == want, f"got {got!r}, want {want!r}")


def skip(name: str, why: str) -> None:
    _skipped.append(name)
    print(f"  - {name}（跳过：{why}）")


# ── 伪造冻结态 ──────────────────────────────────────────────────────────────

_SENTINEL = object()


class Frozen:
    """临时把进程装成 PyInstaller 打出来的样子，退出时全部还原。

    只改 ``sys.frozen`` / ``sys.executable`` / 两个环境变量 —— 这正是
    :mod:`app.paths` 判断"我在哪种形态里"的全部输入。
    """

    def __init__(self, exe: Path, local_app_data: Path, data_env: str | None = None):
        self.exe = exe
        self.local_app_data = local_app_data
        self.data_env = data_env

    def __enter__(self) -> "Frozen":
        self._frozen = getattr(sys, "frozen", _SENTINEL)
        self._exe = sys.executable
        self._lad = os.environ.get("LOCALAPPDATA", _SENTINEL)
        self._denv = os.environ.get("CSNET_DATA_DIR", _SENTINEL)

        sys.frozen = True
        sys.executable = str(self.exe)
        os.environ["LOCALAPPDATA"] = str(self.local_app_data)
        if self.data_env is None:
            os.environ.pop("CSNET_DATA_DIR", None)
        else:
            os.environ["CSNET_DATA_DIR"] = self.data_env
        paths.reset_cache()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._frozen is _SENTINEL:
            del sys.frozen
        else:
            sys.frozen = self._frozen
        sys.executable = self._exe
        if self._lad is _SENTINEL:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = str(self._lad)
        if self._denv is _SENTINEL:
            os.environ.pop("CSNET_DATA_DIR", None)
        else:
            os.environ["CSNET_DATA_DIR"] = str(self._denv)
        paths.reset_cache()


def _scratch() -> Path:
    """临时的"exe 目录"父级。

    不用 ``tempfile.mkdtemp()``：它在 Windows 上按 0o700 建目录，而这份工作区
    的沙箱 ACL 只对默认权限建出来的目录放行，mkdtemp 的目录连写都写不进去
    （`PermissionError`，和权限位无关）。``uuid`` + 默认 ``mkdir()`` 没有这个毛病。
    """
    base = _ROOT / ".build" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"paths-{uuid.uuid4().hex[:12]}"
    path.mkdir()
    return path


# ── 1. 开发态：和以前完全一样 ───────────────────────────────────────────────


def test_dev_mode() -> None:
    print("dev mode / 开发态不动")
    eq("not frozen", paths.is_frozen(), False)
    eq("bundle dir is the repo root", paths.BUNDLE_DIR, _ROOT)
    eq("resource root is the repo root", paths.RESOURCES_DIR, _ROOT)
    eq("data dir is app/data", paths.DATA_DIR, _ROOT / "app" / "data")
    eq("maps dir", paths.MAPS_DIR, _ROOT / "maps" / "optimized_obj_files")
    eq("models dir", paths.MODELS_DIR, _ROOT / "models")
    eq("examples dir", paths.EXAMPLES_DIR, _ROOT / "examples")
    eq("not portable", paths.PORTABLE, False)
    eq("is_portable()", paths.is_portable(), False)

    # app.config 必须把它们原样贴过去 —— 一堆调用方在 import 时就把
    # config.DATA_DIR 绑进了默认参数，这里错了不会有人报错，只会往错目录写。
    eq("config.DATA_DIR", config.DATA_DIR, paths.DATA_DIR)
    eq("config.DEFAULT_DATA_DIR", config.DEFAULT_DATA_DIR, paths.DATA_DIR)
    eq("config.ROOT", config.ROOT, paths.RESOURCES_DIR)
    eq("config.MAPS_DIR", config.MAPS_DIR, paths.MAPS_DIR)
    eq("config.CONFIG_PATH", config.CONFIG_PATH, paths.DATA_DIR / "config.json")
    eq("config.LIBRARY_DB", config.LIBRARY_DB, paths.DATA_DIR / "library.db")

    # 老数据的迁移只在 app/data 里找，别在冻结态指到 _internal 去。
    eq("legacy data dir", config.LEGACY_DATA_DIR, _ROOT / "data")


# ── 2. 冻结态 + exe 旁边能写 ────────────────────────────────────────────────


def test_frozen_portable() -> None:
    print("frozen mode / exe 旁边可写")
    tmp = _scratch()
    app_dir = tmp / "cs-net-5.0.0"
    app_dir.mkdir()
    exe = app_dir / "cs-net.exe"
    exe.write_bytes(b"MZ")

    with Frozen(exe, tmp / "localappdata"):
        eq("frozen", paths.is_frozen(), True)
        eq("bundle dir is the exe's folder", paths.bundle_dir(), app_dir)
        eq("resource root without _internal", paths.resources_dir(), app_dir)
        eq("data sits next to the exe", paths.data_dir(), app_dir / "data")
        check("the data dir was created", (app_dir / "data").is_dir())
        eq("portable", paths.is_portable(), True)
        eq("maps next to the exe", paths.maps_dir(),
           app_dir / "maps" / "optimized_obj_files")
        eq("models next to the exe", paths.models_dir(), app_dir / "models")
        eq("no probe file left behind",
           list((app_dir / "data").glob(".cs-net-write-probe")), [])
        eq("idempotent", paths.data_dir(), paths.data_dir())

        # PyInstaller 真打包时会带上 `_internal/`，资源根要让给它。
        (app_dir / "_internal").mkdir()
        eq("resource root under _internal", paths.resources_dir(),
           app_dir / "_internal")


# ── 3. 冻结态 + exe 旁边不能写（模拟 Program Files）──────────────────────────


def test_frozen_readonly() -> None:
    print("frozen mode / exe 旁边不可写")
    tmp = _scratch()
    # 把"安装目录"做成一个**文件**：在里面 mkdir 会抛 NotADirectoryError，
    # 这正是只读目录在探测眼里的样子，而且不用改 ACL（Windows 上改 ACL 要管理员）。
    blocked = tmp / "program-files"
    blocked.write_text("not a directory", encoding="utf-8")
    exe = blocked / "cs-net.exe"
    lad = tmp / "localappdata"

    with Frozen(exe, lad):
        eq("bundle dir is still the exe's folder", paths.bundle_dir(), blocked)
        eq("data falls back to %LOCALAPPDATA%", paths.data_dir(), lad / "cs-net")
        eq("not portable", paths.is_portable(), False)
        eq("models fall back into the data dir", paths.models_dir(),
           lad / "cs-net" / "models")
        eq("maps do not move", paths.maps_dir(),
           blocked / "maps" / "optimized_obj_files")

    # 没有 LOCALAPPDATA（非 Windows 或者被清了）也不能炸。
    with Frozen(exe, tmp / "unused"):
        os.environ.pop("LOCALAPPDATA", None)
        paths.reset_cache()
        want = Path.home() / ".cs-net"
        eq("home fallback", paths.data_dir(), want)


# ── 4. CSNET_DATA_DIR 谁都得让路 ────────────────────────────────────────────


def test_env_override() -> None:
    print("CSNET_DATA_DIR / 环境变量优先")
    tmp = _scratch()
    app_dir = tmp / "cs-net-5.0.0"
    app_dir.mkdir()
    exe = app_dir / "cs-net.exe"
    exe.write_bytes(b"MZ")
    elsewhere = tmp / "D-drive" / "cs-net-data"

    with Frozen(exe, tmp / "localappdata", data_env=str(elsewhere)):
        eq("env wins over the portable dir", paths.data_dir(), elsewhere)
        eq("and the app is not portable", paths.is_portable(), False)
        eq("models still follow the bundle", paths.models_dir(), app_dir / "models")

    # 开发态也认（测试和"库放 NAS 上"靠它）。
    with Frozen(exe, tmp / "localappdata", data_env=str(elsewhere)):
        os.environ.pop("CSNET_DATA_DIR", None)
        paths.reset_cache()
        eq("unset env -> back to portable", paths.data_dir(), app_dir / "data")


# ── 5. 辅助程序的名字 ───────────────────────────────────────────────────────


def test_sidechannel_name() -> None:
    print("side-channel / 可执行文件名")
    from demo_extract import sidechannel

    if os.name == "nt":
        eq("windows needs a .exe", sidechannel.BINARY_NAME, "cs2sidechannel.exe")
        check("binary_path ends with .exe",
              str(sidechannel.binary_path()).endswith(".exe"))
    else:
        eq("posix keeps the bare name", sidechannel.BINARY_NAME, "cs2sidechannel")


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    for fn in (
        test_dev_mode,
        test_frozen_portable,
        test_frozen_readonly,
        test_env_override,
        test_sidechannel_name,
    ):
        print()
        fn()

    print()
    if _failed:
        print(f"FAILED {len(_failed)} / {_passed + len(_failed)}")
        for name in _failed:
            print(f"  - {name}")
        return 1
    print(f"all {_passed} checks passed" + (f" ({len(_skipped)} skipped)" if _skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
