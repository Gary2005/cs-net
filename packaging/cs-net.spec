# -*- mode: python ; coding: utf-8 -*-
"""cs-net v5 → Windows onedir 包。

        pyinstaller packaging/cs-net.spec --noconfirm

产物在 ``dist/cs-net/``：``cs-net.exe`` + ``_internal/``。**不要**加 ``--onefile``
—— 3 万文件、1.5 GB 的包每次双击都要解压到 ``%TEMP%``，冷启动半分钟，还容易
被杀软当可疑行为。

这里的每一块都对应一个"少了它就只在某个页面上 500"的地方：

* ``pathex`` 里加 ``tools/``：``visualizer/server.py`` 是在运行时把 ``tools`` 塞进
  sys.path 再 ``from demo_extract import ...`` 的，PyInstaller 不会执行那段代码，
  只能靠 pathex 让它找到这个"不是安装包"的模块；
* ``demo_extract`` / ``scripts.training_data`` / ``training`` 用 ``collect_submodules``
  收全：它们之间是互相 import 的网状结构（``app/ai/features.py`` → ``tasks/aim``
  → ``obstruction`` → ``view_render``），漏一个 AI 面板就 500；
* 模板和静态资源走 ``datas``，冻结后落在 ``_internal/`` 里，和 ``app/paths.py``
  的 ``RESOURCES_DIR`` 对得上；
* Go 辅助程序落到 ``demo_extract/sidechannel_go/``：``sidechannel.HERE`` 在冻结后
  就是 ``_internal/demo_extract``，它从旁边的 ``sidechannel_go/`` 找二进制；
* ``maps/`` 和 ``models/`` **不在这里**。它们是发行版的运行期数据，由
  ``packaging/package.ps1`` 在构建之后拷到 exe 旁边（地图本来就解压好，
  lite 版不带权重）。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent          # noqa: F821 - SPECPATH 由 PyInstaller 注入
TOOLS = ROOT / "tools"

# ── 随包的运行期数据 ────────────────────────────────────────────────────────
datas = [
    (str(ROOT / "app" / "templates"), "app/templates"),
    (str(ROOT / "app" / "static"), "app/static"),
    (str(ROOT / "visualizer" / "templates"), "visualizer/templates"),
    # 295 MB：武器模型、贴图、雷达图。派生自 Valve 的内容，玩家机器上重建不了。
    (str(ROOT / "visualizer" / "static"), "visualizer/static"),
]

# 图标：exe 的图标在 EXE(icon=) 里，这两份是给托盘 / 小窗口运行时读的。
for _icon in ("cs-net.ico", "cs-net.png"):
    _path = ROOT / "packaging" / _icon
    if _path.is_file():
        datas.append((str(_path), "packaging"))
    else:
        print(f"[cs-net.spec] 警告：{_path} 不存在（图标会缺席，功能不受影响）")

_sidechannel = TOOLS / "demo_extract" / "sidechannel_go" / (
    "cs2sidechannel.exe" if __import__("os").name == "nt" else "cs2sidechannel"
)
if _sidechannel.is_file():
    datas.append((str(_sidechannel), "demo_extract/sidechannel_go"))
else:
    print(f"[cs-net.spec] 警告：{_sidechannel} 不存在，包里将没有火/烟辅助程序")

# demoparser2 是**纯 Rust 扩展**：它在 Rust 里 `PyImport_ImportModule("polars")`
# 和 ("pyarrow")，PyInstaller 的静态分析完全看不见。源码环境一切正常，玩家的
# "提取"却在 parse_ticks 那一行炸 `ModuleNotFoundError: No module named 'polars'`
# （0.42.2 上真的发生过，`--packaging-selftest` 也看不出来，因为它只 import）。
# 所以整包收进来：polars 的 .py 只是壳，真正的大头是 `_polars_runtime_32` 里那个
# 160 MB 的 Rust 核心；pyarrow 是 demoparser2 造 pandas DataFrame 用的。
_extra_binaries: list = []
_extra_hidden: list = []
for _pkg in ("polars", "_polars_runtime_32", "pyarrow"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    _extra_binaries += _b
    _extra_hidden += _h

hiddenimports = (
    collect_submodules("app")
    + collect_submodules("demo_extract")
    + collect_submodules("scripts.training_data")
    + collect_submodules("training")
    # pystray 在 import 时按平台 `from . import _win32`，静态分析不一定跟得进
    # ctypes 后端；显式列出来，免得打出来的包"托盘不可用"。
    + ["pystray", "pystray._win32"]
    + _extra_hidden
)

# ── 明确不要的 ──────────────────────────────────────────────────────────────
# 训练 / 画图 / notebook 那一套：运行时一个都不 import，收进来纯属浪费体积。
# `matplotlib` 尤其重要 —— numpy/pandas 会尝试 import 它，勾进来就拖一串 Qt。
excludes = [
    "app.tests",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    "torch.utils.tensorboard",
]

a = Analysis(  # noqa: F821 - PyInstaller 注入
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT), str(TOOLS)],
    binaries=_extra_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cs-net",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,       # 保留控制台：--packaging-selftest 的输出、崩溃时的栈都靠它
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "packaging" / "cs-net.ico"),
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="cs-net",
)
