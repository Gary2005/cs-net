"""把 `visualizer/static/logo.svg` 做成 Windows 用的图标。

只在换 logo 时手动跑一次，产物 `packaging/cs-net.ico` + `packaging/cs-net.png`
是要提交的（CI 上没有浏览器可以渲染 SVG）。

    python packaging/make_icon.py

为什么用 Edge 而不是 cairosvg：cairosvg 在 Windows 上要 GTK 的 cairo DLL，
为了一个 200 KB 的图标不值；Edge 是 Windows 自带的，`--headless --screenshot`
一行就够。渲染出来是透明底的 RGBA，Pillow 再居中放进正方形、导出多尺寸 ICO。

图标是一个宽横排字（131.76 × 101.11），方形图标里必然留白 —— 16×16 的时候只能
认出"三条彩色线"，这是横排字的固有代价，不换 logo 就只能这样。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "visualizer" / "static" / "logo.svg"
ICO = ROOT / "packaging" / "cs-net.ico"
PNG = ROOT / "packaging" / "cs-net.png"

#: 渲染多大再缩。256 是 ICO 里最大的一帧，两倍余量足够 LANCZOS 缩到 16 px。
_RENDER = 512
#: 内容占画布的比例。留一点边，任务栏里不至于顶到边。
_FILL = 0.94
_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def _browser() -> str:
    for path in _CANDIDATES:
        if Path(path).is_file():
            return path
    raise SystemExit("找不到 Edge / Chrome，没法把 SVG 渲染成位图")


def render_svg(out_png: Path) -> None:
    uri = "file:///" + str(SVG).replace("\\", "/")
    with tempfile.TemporaryDirectory(prefix="cs-net-icon-profile-") as profile:
        subprocess.run(
            [
                _browser(),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                # 已经在跑的 Edge 会吞掉 --screenshot 并直接退出（exit 1），
                # 给一份独立 profile 就不会去连那个实例。
                f"--user-data-dir={profile}",
                "--default-background-color=00000000",   # 透明底
                f"--window-size={_RENDER},{_RENDER}",
                f"--screenshot={out_png}",
                uri,
            ],
            check=True,
            capture_output=True,
        )


def main() -> int:
    import argparse

    from PIL import Image

    ap = argparse.ArgumentParser(description="logo.svg -> packaging/cs-net.ico + .png")
    ap.add_argument(
        "--source", type=Path, default=None,
        help="跳过渲染，直接用这张 PNG（透明的方形图）。Edge 已经开着的时候 "
             "--headless 会卡住，这时先把 SVG 截出来再用这个参数。",
    )
    args = ap.parse_args()

    if args.source is not None:
        if not args.source.is_file():
            raise SystemExit(f"没有 {args.source}")
        source = Image.open(args.source).convert("RGBA")
    else:
        if not SVG.is_file():
            raise SystemExit(f"没有 {SVG}")
        with tempfile.TemporaryDirectory(prefix="cs-net-icon-") as tmp:
            raw = Path(tmp) / "logo.png"
            render_svg(raw)
            source = Image.open(raw).convert("RGBA")

    content = source.crop(source.getbbox())          # 去掉透明留白
    canvas = Image.new("RGBA", (_RENDER, _RENDER), (0, 0, 0, 0))
    box_w = int(_RENDER * _FILL)
    scale = box_w / content.width
    resized = content.resize((box_w, max(1, round(content.height * scale))), Image.LANCZOS)
    canvas.alpha_composite(
        resized,
        ((_RENDER - resized.width) // 2, (_RENDER - resized.height) // 2),
    )

    canvas.save(PNG)
    canvas.save(ICO, sizes=_SIZES)
    print(f"{PNG.name}: {PNG.stat().st_size / 1024:.0f} KB")
    print(f"{ICO.name}: {ICO.stat().st_size / 1024:.0f} KB, sizes={_SIZES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
