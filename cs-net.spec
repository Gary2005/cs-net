# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CS-Net desktop app.

Usage:
    pyinstaller cs-net.spec
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent

# Collect all model weight files (.pt, .json, .txt) under cs-net-models/.
model_datas = []
models_dir = _root / "cs-net-models"
if models_dir.is_dir():
    for f in models_dir.rglob("*"):
        if f.is_file():
            dest = str(f.relative_to(_root).parent)
            model_datas.append((str(f), dest))

# Collect callout configs.
callout_datas = []
callouts_dir = _root / "config" / "callouts"
if callouts_dir.is_dir():
    for f in callouts_dir.glob("*.yaml"):
        callout_datas.append((str(f), "config/callouts"))

# Collect assets.
asset_datas = []
assets_dir = _root / "assets"
if assets_dir.is_dir():
    for f in assets_dir.rglob("*"):
        if f.is_file():
            dest = str(f.relative_to(_root).parent)
            asset_datas.append((str(f), dest))

a = Analysis(
    ["launcher.py"],
    pathex=[str(_root)],
    binaries=[],
    datas=(
        # Templates and static files for the Flask web app.
        [
            (str(_root / "demo_analysis" / "templates" / "index.html"), "demo_analysis/templates"),
            (str(_root / "demo_analysis" / "static" / "styles.css"), "demo_analysis/static"),
            (str(_root / "demo_analysis" / "static" / "app.js"), "demo_analysis/static"),
            (str(_root / "demo_analysis" / "static" / "radar.js"), "demo_analysis/static"),
        ]
        # Static viewer (WASM-based demo viewer).
        + [
            (str(f), str(f.relative_to(_root).parent))
            for f in (_root / "demo_analysis" / "static" / "viewer").rglob("*")
            if f.is_file()
        ]
        # Map overview images.
        + [
            (str(f), str(f.relative_to(_root).parent))
            for f in (_root / "demo_analysis" / "static" / "overviews").glob("*.png")
        ]
        # Model weights, configs, assets.
        + model_datas
        + callout_datas
        + asset_datas
    ),
    hiddenimports=[
        "torch",
        "transformers",
        "transformers.models.auto",
        "tokenizers",
        "flask",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "yaml",
        "huggingface_hub",
        "agent_framework",
        "agent_framework.openai",
        "demoparser2",
        "polars",
        "wandb",
        "peft",
        "accelerate",
        "safetensors",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "IPython",
        "ipykernel",
        "jupyter",
        "notebook",
        "sentry_sdk",
        "playwright",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cs-net",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_root / "assets" / "logo.png") if (_root / "assets" / "logo.png").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="cs-net",
)
