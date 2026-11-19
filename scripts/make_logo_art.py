#!/usr/bin/env python3
"""
把 ``visualizer/static/logo.svg`` 渲染成 CLI 用的字符画，生成
``visualizer/logo_art.py``。

为什么要有这个脚本
------------------
命令行横幅里那个 logo 是**从 SVG 生成**的，不是手画的：手画的字符画跟矢量图对不上，
设计改颜色 / 改形状之后没人会记得回去修。所以流程是
``logo.svg --(headless Chrome 截图)--> PNG --(面积平均 + 品牌色量化)--> 字符网格``，
产物是 ``visualizer/logo_art.py`` 里的几个字符串常量（不依赖运行时的光栅化库）。

怎么画
------
每个字符格子 = **上下两个像素**，用半个方块 ``▀`` 画：前景色是上半格、背景色是下半格。
这样一格能表达两种颜色，竖直分辨率翻倍 —— 否则同一个网格下 logo 会被压扁成一半高。

尺寸必须守住宽高比：终端一个字符格大约是 **1:2**（宽:高），半高块让一格覆盖两个像素行，
所以"像素网格的宽高比"要等于 logo 的宽高比。参数 ``--cols``（字符列数）定下来之后，
行数由 SVG 的包围盒反推，不用手调。

用法::

    python scripts/make_logo_art.py                 # 默认 44 列，写 visualizer/logo_art.py
    python scripts/make_logo_art.py --cols 32       # 换尺寸
    python scripts/make_logo_art.py --preview       # 只在终端里看一眼，不写文件

需要 Chrome/Edge（截图）和 Pillow（缩放）。找不到就退出码 2（= 没跑成，不是"通过"）。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "visualizer" / "static" / "logo.svg"
OUT = ROOT / "visualizer" / "logo_art.py"

try:
    import numpy as np
    from PIL import Image
except ImportError as _exc:      # 报清楚缺哪个，别等到缩图那一步才炸
    np = None
    Image = None
    _IMPORT_ERROR: Exception | None = _exc
else:
    _IMPORT_ERROR = None

# Windows 上 Chrome 的默认位置 + 环境变量覆盖，和 visualizer/tests 里那几支测试同一份
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

#: 截图用的窗口边长。**不能用目标尺寸**：窗口小到一定程度时 Chrome 会按一个最小
#: 布局宽度排版，SVG 就被裁掉一截（实测 352 px 时右侧被切、宽高比从 1.30 变成 0.94，
#: 生成出来的字符画整个是歪的）。固定截一张大图再按面积平均缩下来，既躲开这个坑，
#: 本身也就是超采样。
RENDER_PX = 1400

#: 低于这个 alpha 的格子当作"没有内容"（透明）。抗锯齿边缘会留下一点淡淡的颜色，
#: 不滤掉的话字符画外面会糊一层脏边；但太高会把 logo 的**细笔画**（N 那两根竖线
#: 只有 ~3 个 SVG 单位宽 ≈ 1.2 格）整段吃掉 —— 实测 0.2 是这条线上最好的：
#: 「cs」不糊、NET 的竖线不断。见 .build/logo_cutoffs.png 那种对比图。
ALPHA_CUTOFF = 0.2

#: 默认字符列数。44 列 ≈ 17 行：再小「NET」的细笔画会断成虚线，再大横幅太高。
DEFAULT_COLS = 44


def find_chrome() -> str | None:
    import os

    cands = [os.environ.get("CHROME_PATH")] + CHROME_CANDIDATES
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def svg_palette(svg_text: str) -> list[str]:
    """SVG 里 ``<style>`` 的 fill 颜色，按出现顺序去重。

    从 SVG 里读而不是写死：设计改色之后重新生成一次就同步了。
    """
    out: list[str] = []
    for hexv in re.findall(r"fill:\s*(#[0-9a-fA-F]{6})", svg_text):
        if hexv.lower() not in [c.lower() for c in out]:
            out.append(hexv.lower())
    return out


def svg_aspect(svg_text: str) -> float:
    """``viewBox`` 的宽高比，用来检查截图没有被裁。"""
    m = re.search(r'viewBox="([\d.\-\s,]+)"', svg_text)
    if not m:
        raise RuntimeError("SVG 里没有 viewBox")
    x0, y0, w, h = (float(v) for v in re.split(r"[\s,]+", m.group(1).strip()))
    return w / h


def screenshot(chrome: str, svg: Path, out: Path, w: int, h: int) -> None:
    """用 headless Chrome 把 SVG 截成带透明通道的 PNG。

    SVG 只有 viewBox、没有 width/height，所以浏览器按 ``preserveAspectRatio`` 缩放到
    窗口里（居中、留透明边），之后再按 alpha 包围盒裁掉那些边。
    """
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={w},{h}",
        # ARGB，alpha=00 → 背景透明
        "--default-background-color=00000000",
        f"--screenshot={out}",
        svg.resolve().as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not out.exists():
        raise RuntimeError(f"Chrome 没有产出截图：\n{proc.stdout}\n{proc.stderr}")


def render_grid(png: Path, cols: int, expect_aspect: float | None = None):
    """PNG → 目标像素网格的 (rgb, alpha)。做**预乘**的面积平均，否则透明像素
    （0,0,0,0）会把边缘往黑色拉。

    ``expect_aspect`` 是 SVG 的宽高比：截图被浏览器裁过的话这个检查会当场拦下来 ——
    裁过的图**看起来**也能生成字符画，只是整个是歪的，肉眼很难发现。
    """
    im = Image.open(png).convert("RGBA")
    alpha = im.getchannel("A")
    bbox = alpha.point(lambda a: 255 if a > 8 else 0).getbbox()
    if bbox is None:
        raise RuntimeError("截图整张都是透明的 —— SVG 没渲染出来")
    im = im.crop(bbox)

    aspect = im.width / im.height
    if expect_aspect and abs(aspect - expect_aspect) / expect_aspect > 0.05:
        raise RuntimeError(
            f"截图宽高比 {aspect:.3f} 和 SVG 的 {expect_aspect:.3f} 差太多 —— "
            f"多半是窗口太小被浏览器裁了（把 RENDER_PX 调大）")

    # 宽高比定行数：字符格宽:高 ≈ 1:2，一个格子装两个像素行
    pw = cols
    ph = round(cols * im.height / im.width)
    if ph % 2:
        ph += 1

    arr = np.asarray(im).astype(np.float64) / 255.0
    a = arr[..., 3]
    pm = arr[..., :3] * a[..., None]

    def _resize(chan: "np.ndarray") -> "np.ndarray":
        img = Image.fromarray(chan.astype(np.float32), mode="F")
        return np.asarray(img.resize((pw, ph), Image.BOX), dtype=np.float64)

    pm_small = np.stack([_resize(pm[..., i]) for i in range(3)], axis=-1)
    a_small = _resize(a)
    rgb = np.where(a_small[..., None] > 1e-9, pm_small / np.maximum(a_small[..., None], 1e-9), 0.0)
    return np.clip(rgb, 0, 1), np.clip(a_small, 0, 1), (pw, ph)


def quantise(rgb, alpha, palette_rgb, cutoff: float = ALPHA_CUTOFF):
    """每个像素 → 一个字符（调色板字母，或 ``.`` 表示透明）。

    量化到品牌色，不是为了省字节，而是**字符画在这个尺寸上更清楚**：抗锯齿边缘的
    中间色在半格方块里会变成一片灰雾，人眼反而看不出形状。
    """
    pal = np.asarray(palette_rgb, dtype=np.float64)          # (n, 3) 0..1
    dist = ((rgb[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    nearest = dist.argmin(-1)
    letters = np.array(list("1234567890"))[nearest]
    letters = np.where(alpha < cutoff, ".", letters)
    return letters


def to_rows(letters, palette_keys) -> list[str]:
    """像素网格 → 字符行：一格两个字符（上、下）。"""
    rows: list[str] = []
    h, w = letters.shape
    for y in range(0, h, 2):
        top = letters[y]
        bottom = letters[y + 1] if y + 1 < h else np.full(w, ".", dtype="<U1")
        rows.append("".join(t + b for t, b in zip(top, bottom)))
    return rows


def ansi_lines(rows: list[str], palette_rgb: dict) -> list[str]:
    """预览用的 ANSI 行 —— **直接用 CLI 那支渲染器**（``visualizer.banner.ansi_logo``），
    不在这里另写一份：预览和真正打出来的必须是同一套代码，否则预览好看、实际难看。"""
    sys.path.insert(0, str(ROOT))
    from visualizer.banner import ansi_logo

    palette = {k: tuple(int(round(c * 255)) for c in v) for k, v in palette_rgb.items()}
    return ansi_logo(rows, palette)


def write_module(rows: list[str], palette_hex: list[str], cols: int) -> None:
    """写出 ``visualizer/logo_art.py``。

    换行交给 Python 的文本模式：Windows 上写出 CRLF，git 里存的还是 LF
    （``core.autocrlf``），生成器不用管这件事。
    """
    rgb = []
    for hexv in palette_hex:
        rgb.append(tuple(int(hexv[i:i + 2], 16) for i in (1, 3, 5)))
    body = "\n".join(f'    "{r}",' for r in rows)
    OUT.write_text(
        '"""CLI 横幅里的 logo 字符画 —— **由 scripts/make_logo_art.py 生成，不要手改**。\n'
        "\n"
        "每个字符是半个像素高：一行字符 = 上下两个像素（渲染时用 ``▀``，前景=上、背景=下）。\n"
        "字母是 PALETTE 里的颜色键，``.`` 是透明。\n"
        "\n"
        "重新生成::\n"
        "\n"
        "    python scripts/make_logo_art.py\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        f"#: 源文件（改这个再去重新生成，别手改这里的字符串）\n"
        f'SOURCE = "visualizer/static/logo.svg"\n'
        f"\n"
        f"#: 每个字符单元格的宽度（= 每行字符串长度 / 2）\n"
        f"WIDTH = {cols}\n"
        f"\n"
        f"#: 像素行数（= len(ROWS) * 2）\n"
        f"PIXEL_ROWS = {len(rows) * 2}\n"
        "\n"
        "#: 颜色键 → RGB（取自 SVG 里的 fill）\n"
        "PALETTE = {\n"
        + "".join(f'    "{k}": {str(v) + ",":<15} # {h}\n'
                  for k, v, h in zip("1234567890", rgb, palette_hex))
        + "}\n"
        "\n"
        "#: 半高像素网格，一行一格字符对（上、下）\n"
        "ROWS = [\n"
        f"{body}\n"
        "]\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="logo.svg → CLI 字符画")
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS,
                    help=f"字符列数（默认 {DEFAULT_COLS}，≈17 行）")
    ap.add_argument("--preview", action="store_true", help="只在终端里看，不写文件")
    args = ap.parse_args()

    chrome = find_chrome()
    if not chrome:
        print("SKIP: 找不到 Chrome/Edge（设 CHROME_PATH）")
        return 2
    if _IMPORT_ERROR is not None:
        print(f"SKIP: {_IMPORT_ERROR}（需要 numpy + Pillow）")
        return 2

    svg_text = SVG.read_text(encoding="utf-8")
    palette_hex = svg_palette(svg_text)
    if not palette_hex:
        print("FAIL: SVG 里没找到 fill 颜色")
        return 1
    palette_rgb = {
        k: tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        for k, h in zip("1234567890", palette_hex)
    }

    print(f"chrome : {chrome}")
    print(f"svg    : {SVG.relative_to(ROOT)}  {palette_hex}")
    aspect = svg_aspect(svg_text)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "logo.png"
        screenshot(chrome, SVG, png, RENDER_PX, RENDER_PX)
        rgb, alpha, (pw, ph) = render_grid(png, args.cols, expect_aspect=aspect)

    letters = quantise(rgb, alpha, [palette_rgb[k] for k in palette_rgb])
    rows = to_rows(letters, list(palette_rgb))

    bad = {len(r) for r in rows}
    if bad != {args.cols * 2}:
        print(f"FAIL: 行长度不一致 {bad}（应为 {args.cols * 2}）")
        return 1

    ink = sum(1 for r in rows for ch in r if ch != ".")
    print(f"grid   : {pw}x{ph} 像素 → {len(rows)} 行 x {args.cols} 格，"
          f"有内容 {ink}/{len(rows) * args.cols * 2} 个半像素")

    if args.preview:
        for line in ansi_lines(rows, palette_rgb):
            print(line)
        return 0

    write_module(rows, palette_hex, args.cols)
    print(f"wrote  : {OUT.relative_to(ROOT)}")
    for line in ansi_lines(rows, palette_rgb):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
