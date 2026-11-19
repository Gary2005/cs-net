"""程序目录 / 数据目录 —— 打包成 exe 之后，哪些地方是只读的、哪些能写。

开发态（从源码跑）和以前完全一样：``BUNDLE_DIR`` 是仓库根，数据在 ``app/data``。

PyInstaller ``--onedir`` 打出来的包是另一种形状：exe 旁边只有 ``_internal/``，
而整包可能被解压到任何地方 —— 玩家自己挑的文件夹（可写），或者 ``Program Files``
（只读）。所以资源和使用者的库必须彻底分开：

======================  ===========================  ========================================
冻结态                   例子                          说明
======================  ===========================  ========================================
``BUNDLE_DIR``          ``D:\\cs-net-5.0.0\\``        exe 所在目录
``RESOURCES_DIR``       ``D:\\cs-net-5.0.0\\_internal\\``  模板 / 静态资源 / 内置的包
``DATA_DIR``            ``D:\\cs-net-5.0.0\\data\\``  能写就在这里（绿色版，换机器整目录拷走）
                        ``%LOCALAPPDATA%\\cs-net\\``  exe 旁边写不了时退到这里
``MAPS_DIR``            ``BUNDLE_DIR\\maps\\optimized_obj_files\\``  地图几何（随包发）
``MODELS_DIR``          ``BUNDLE_DIR\\models\\``      权重；写不了才退回 ``DATA_DIR\\models``
======================  ===========================  ========================================

``CSNET_DATA_DIR`` 覆盖数据目录，两种模式下都认 —— 测试靠它指到临时目录，也顺便
支持"库放在移动硬盘 / NAS 上"。

**为什么探测可写而不是放一个标记文件**：同一个 zip 解压到 D 盘可写、解压到
``Program Files`` 不可写，而玩家把文件夹搬个位置，标记文件就再也对不上了。
探测的代价只有一次 ``mkdir`` + 一次写删，而且只在冻结态发生。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

#: 探测可写时写的临时文件名。失败（只读目录）时它根本写不出来。
_PROBE_NAME = ".cs-net-write-probe"

#: 缓存：探测有副作用，别在每次 import 都做一遍。测试用 :func:`reset_cache`。
_CACHE: dict[str, object] = {}


def is_frozen() -> bool:
    """是不是打出来的（PyInstaller / Nuitka）可执行文件，而不是 ``python -m``。"""
    if getattr(sys, "frozen", False):
        return True
    return bool(globals().get("__compiled__"))


def bundle_dir() -> Path:
    """exe 所在目录（开发态是仓库根）。资源、地图、权重都相对它找。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resources_dir() -> Path:
    """模板 / 静态资源 / 内置包所在的根。

    PyInstaller 把收集来的数据放在 exe 旁边的 ``_internal/``；开发态没有这个
    目录，仓库根就是资源根。两边的下游代码（``Path(__file__)`` 推模板路径那些）
    因此都不用改。
    """
    root = bundle_dir()
    if is_frozen():
        internal = root / "_internal"
        if internal.is_dir():
            return internal
    return root


def _writable(path: Path) -> bool:
    """``path`` 能不能建、能不能写。只读目录会抛 ``PermissionError`` / ``OSError``。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / _PROBE_NAME
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except (OSError, ValueError):
        return False


def _local_app_data() -> Path:
    """``%LOCALAPPDATA%\\cs-net``；非 Windows 退到 ``~/.cs-net``（macOS 以后要用）。"""
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "cs-net"
    return Path.home() / ".cs-net"


def data_dir() -> Path:
    """用户的库、配置、``.csz``、缩略图、AI 缓存放哪。

    顺序：``CSNET_DATA_DIR`` > 便携目录（exe 旁边的 ``data\\``，能写才用）>
    ``%LOCALAPPDATA%\\cs-net``。开发态固定是 ``app/data``，不做任何探测 ——
    ``app/data`` 是 gitignore 的，建了也无所谓，但没必要让单测在仓库里留文件。
    """
    cached = _CACHE.get("data")
    if cached is not None:
        return Path(str(cached))

    env = os.environ.get("CSNET_DATA_DIR")
    if env:
        chosen = Path(env).expanduser()
    elif not is_frozen():
        chosen = Path(__file__).resolve().parent / "data"
    else:
        portable = bundle_dir() / "data"
        chosen = portable if _writable(portable) else _local_app_data()

    _CACHE["data"] = chosen
    return chosen


def is_portable() -> bool:
    """数据目录是不是就在 exe 旁边（绿色版）。只对冻结态有意义。"""
    if not is_frozen():
        return False
    try:
        return data_dir() == bundle_dir() / "data"
    except OSError:
        return False


def maps_dir() -> Path:
    """地图几何目录（``*.obj`` 与 ``*.obj.zip`` 放一起，和 ``extract_maps.py`` 一致）。

    冻结态和 exe 并排、**不在** ``_internal`` 里：943 MB 的东西塞进内部目录
    只会让玩家以为那是个缓存，而且 ``extract_maps.py`` 的横幅要能往里写。
    """
    return bundle_dir() / "maps" / "optimized_obj_files"


def models_dir() -> Path:
    """两份权重放哪。

    包里的 ``models\\`` 能写就用它（full 版就是这么发的，lite 版第一次运行也往
    这里下）；写不了（只读安装位置）才退回 ``DATA_DIR\\models``。
    """
    root = bundle_dir() / "models"
    if not is_frozen():
        return root
    cached = _CACHE.get("models")
    if cached is not None:
        return Path(str(cached))
    chosen = root if _writable(root) else data_dir() / "models"
    _CACHE["models"] = chosen
    return chosen


def examples_dir() -> Path:
    """仓库自带的示例 demo 目录（只有开发和测试用得到）。"""
    return bundle_dir() / "examples"


def reset_cache() -> None:
    """忘掉探测结果。测试换 ``sys.frozen`` / ``CSNET_DATA_DIR`` 之后要调一次。"""
    _CACHE.clear()


#: 下面这些是 import 时算好的快照，给 ``app.config`` 直接用 —— 它的模块级常量
#: 必须在那里就是最终值（一堆调用方在 import 时就把它们绑进默认参数了）。
BUNDLE_DIR = bundle_dir()
RESOURCES_DIR = resources_dir()
DATA_DIR = data_dir()
MAPS_DIR = maps_dir()
MODELS_DIR = models_dir()
EXAMPLES_DIR = examples_dir()
PORTABLE = is_portable()

__all__ = [
    "BUNDLE_DIR",
    "DATA_DIR",
    "EXAMPLES_DIR",
    "MAPS_DIR",
    "MODELS_DIR",
    "PORTABLE",
    "RESOURCES_DIR",
    "bundle_dir",
    "data_dir",
    "examples_dir",
    "is_frozen",
    "is_portable",
    "maps_dir",
    "models_dir",
    "reset_cache",
    "resources_dir",
]
