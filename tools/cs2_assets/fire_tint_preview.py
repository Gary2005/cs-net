#!/usr/bin/env python3
"""Composite the three CS2 fire sprite layers the way three.js actually will.

Why this exists: the three flame sequences are **grayscale luminance masks**
(measured mean per-channel spread 1.5-1.6 out of 255), so the colour has to come
from a per-layer `MeshBasicMaterial.color` tint. Picking those numbers by eye is
guesswork; this reproduces the renderer's maths instead and writes a preview PNG.

It models, per layer, in this order (matching three r160):

  1. sRGB -> linear decode of the texture   (`texture.colorSpace = SRGBColorSpace`)
  2. multiply by `color` (the tint, allowed >1) and `opacity`  -- in LINEAR space
  3. ACES filmic tone mapping with the renderer's exposure
  4. linear -> sRGB encode for output
  5. additive blend: contribution = out_rgb * (tex_alpha * opacity)

Note step 3 happens **per layer, before blending** -- that is how the GPU does
it (tone mapping is in the fragment shader, blending is in the framebuffer), and
it is why a layer whose tint is too hot saturates to white on its own instead of
being rolled off together with the others.

Usage:
    python fire_tint_preview.py [--out preview.png] [--frame 16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

DEFAULT_ASSETS = ROOT / "visualizer" / "static" / "assets" / "sprites"

# Must stay in sync with FIRE_LAYERS in visualizer/static/js/visuals.js
LAYERS = [
    # seq,              scale, opacity, fps, phase, tint (linear, >1 = gain)
    ("flames",           0.55, 0.95, 24, 0.00, (3.40, 2.00, 0.70)),
    ("flame_omni",       0.85, 0.62, 18, 0.37, (2.60, 1.05, 0.20)),
    ("flame_wisp",       1.20, 0.34, 13, 0.71, (1.60, 0.42, 0.07)),
]

EXPOSURE = 0.95   # scene.js: renderer.toneMappingExposure


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _rrt_odt_fit(v: np.ndarray) -> np.ndarray:
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / b


ACES_IN = np.array([
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
])
ACES_OUT = np.array([
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
])


def aces_filmic(rgb: np.ndarray, exposure: float = EXPOSURE) -> np.ndarray:
    """three.js ACESFilmicToneMapping, operating on (..., 3) linear RGB."""
    c = rgb * (exposure / 0.6)
    c = c @ ACES_IN.T
    c = _rrt_odt_fit(c)
    c = c @ ACES_OUT.T
    return np.clip(c, 0.0, 1.0)


def load_frame(assets: Path, seq: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((assets / seq).glob(f"{seq}_seq*_*.png"))
    if not files:
        raise SystemExit(f"no frames under {assets / seq}")
    im = Image.open(files[idx % len(files)]).convert("RGBA")
    a = np.asarray(im, dtype=np.float64) / 255.0
    return a[..., :3], a[..., 3]


def composite(assets: Path, frame: int, canvas: int = 384) -> tuple[np.ndarray, dict]:
    """Return (H,W,3) sRGB composite plus per-layer diagnostics.

    The three sequences have **different pixel sizes** (78x120, 108x128,
    200x160) because they are separate textures, but in the renderer each one
    covers a quad whose size is `scale` relative to the fire patch. So they are
    not composited pixel-for-pixel: each is resampled onto a common canvas at its
    own relative size and centred, which is what the three quads look like from
    directly above.
    """
    biggest = max(L[1] for L in LAYERS)
    rgb_sum = np.zeros((canvas, canvas, 3), dtype=np.float64)
    alpha_max = np.zeros((canvas, canvas), dtype=np.float64)
    diag = {}
    for seq, scale, opacity, _fps, _phase, tint in LAYERS:
        tex, alpha = load_frame(assets, seq, frame)
        lin = srgb_to_linear(tex)
        # 2. tint + opacity in linear space
        c = lin * np.array(tint)[None, None, :] * opacity
        # 3. per-layer tone map, 4. encode
        out = linear_to_srgb(aces_filmic(c))
        # 5. additive: src_rgb * src_alpha, src_alpha = tex_alpha * opacity
        contrib = out * (alpha * opacity)[..., None]

        # 铺到该层的 quad 大小上（相对最大的那层），居中
        side = max(1, int(round(canvas * scale / biggest)))
        plane = np.zeros((canvas, canvas, 3), dtype=np.float64)
        pa = np.zeros((canvas, canvas), dtype=np.float64)
        off = (canvas - side) // 2
        ci = np.asarray(Image.fromarray((np.clip(contrib, 0, 1) * 255).astype(np.uint8))
                        .resize((side, side), Image.LANCZOS), dtype=np.float64) / 255.0
        ai = np.asarray(Image.fromarray((np.clip(alpha * opacity, 0, 1) * 255).astype(np.uint8))
                        .resize((side, side), Image.LANCZOS), dtype=np.float64) / 255.0
        plane[off:off + side, off:off + side] = ci
        pa[off:off + side, off:off + side] = ai

        rgb_sum += plane
        alpha_max = np.maximum(alpha_max, pa)
        diag[seq] = {
            "quad_px": side,
            "tex_mean_linear": float(lin.mean()),
            "layer_mean_srgb": float(out.mean()),
            "layer_max": float(out.max()),
        }
    return np.clip(rgb_sum, 0.0, 1.0), diag, alpha_max


def describe(name: str, img: np.ndarray, alpha_mask: np.ndarray | None = None) -> str:
    flat = img.reshape(-1, 3)
    mask = None
    if alpha_mask is not None:
        mask = alpha_mask.reshape(-1) > 0.02
        if mask.any():
            flat = flat[mask]
    r, g, b = flat.mean(axis=0)
    # 亮度加权平均色相：只统计"看得见"的像素
    bright = flat[flat.max(axis=1) > 0.25]
    if len(bright):
        br, bg, bb = bright.mean(axis=0)
    else:
        br = bg = bb = 0.0
    # "有多白"：HSV 饱和度低的像素占比 —— 用户说的"火是白的"就是这个
    mx = flat.max(axis=1)
    mn = flat.min(axis=1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    vis = mx > 0.15                       # 真正看得见的像素
    n_vis = int(vis.sum())
    washed = float((sat[vis] < 0.20).mean() * 100) if n_vis else 0.0
    mid = float((sat[vis] < 0.35).mean() * 100) if n_vis else 0.0
    # 平均色相（只对看得见且有色相的像素）
    hue = _mean_hue(flat[vis]) if n_vis else 0.0
    return (f"{name:<16} mean=({r:.3f},{g:.3f},{b:.3f}) "
            f"bright=({br:.3f},{bg:.3f},{bb:.3f})  hue={hue:5.1f}deg  "
            f"低饱和(<0.20)={washed:4.1f}%  <0.35={mid:4.1f}%  n_vis={n_vis}")


def _mean_hue(rgb: np.ndarray) -> float:
    """Circular mean hue in degrees (0=red, 30=orange, 60=yellow)."""
    import colorsys
    ang = []
    for px in rgb[:: max(1, len(rgb) // 4000)]:
        h, s, v = colorsys.rgb_to_hsv(*px)
        if s > 0.05 and v > 0.05:
            ang.append(h * 2 * np.pi)
    if not ang:
        return 0.0
    a = np.array(ang)
    return float((np.arctan2(np.sin(a).mean(), np.cos(a).mean()) % (2 * np.pi))
                 * 180 / np.pi)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    ap.add_argument("--out", type=Path, default=ROOT / "notes" / "scratch" / "fire_tint_preview.png")
    ap.add_argument("--frame", type=int, default=16, help="frame index into each sequence")
    ap.add_argument("--tints", default="",
                    help='override tints: "r,g,b;r,g,b;r,g,b" in core,filler,edge order')
    ap.add_argument("--frames", default="",
                    help="comma-separated frame indices -> write a filmstrip instead")
    ap.add_argument("--canvas", type=int, default=384)
    args = ap.parse_args(argv[1:])

    if args.tints:
        parts = [p for p in args.tints.split(";") if p.strip()]
        if len(parts) != len(LAYERS):
            raise SystemExit(f"--tints needs {len(LAYERS)} groups, got {len(parts)}")
        for i, p in enumerate(parts):
            LAYERS[i] = (*LAYERS[i][:5], tuple(float(x) for x in p.split(",")))

    if not args.assets.is_dir():
        raise SystemExit(f"no sprite assets at {args.assets}")

    print(f"layers (tint is LINEAR, >1 = additive gain; exposure={EXPOSURE}):")
    for seq, scale, opacity, fps, phase, tint in LAYERS:
        print(f"  {seq:<14} scale={scale:<5} opacity={opacity:<5} {fps:>2}fps "
              f"phase={phase:<5} tint=({tint[0]:.2f},{tint[1]:.2f},{tint[2]:.2f})")
    print()

    img, diag, vis_mask = composite(args.assets, args.frame, args.canvas)
    for seq, d in diag.items():
        print(f"  {seq:<14} quad={d['quad_px']:>3}px  tex_mean_linear={d['tex_mean_linear']:.3f}  "
              f"after tint+ACES: mean={d['layer_mean_srgb']:.3f} max={d['layer_max']:.3f}")
    print()
    print(describe("COMPOSITE", img))
    print(describe("COMPOSITE(vis)", img, vis_mask))

    # 预览：合成图 + 每层单独一张，横排
    tiles = [img]
    for seq, _s, opacity, _f, _p, tint in LAYERS:
        tex, alpha = load_frame(args.assets, seq, args.frame)
        lin = srgb_to_linear(tex)
        out = linear_to_srgb(aces_filmic(lin * np.array(tint)[None, None, :] * opacity))
        tiles.append(out * (alpha * opacity)[..., None])
    H = max(t.shape[0] for t in tiles)
    W = sum(t.shape[1] for t in tiles) + 8 * (len(tiles) - 1)
    sheet = np.zeros((H, W, 3), dtype=np.float64)
    x = 0
    for t in tiles:
        sheet[:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 8
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print(f"\npreview (composite | core | filler | edge) -> {args.out}")

    # 多帧胶片条：确认不是"只有这一帧好看"
    if args.frames:
        idxs = [int(x) for x in args.frames.split(",") if x.strip()]
        rows = [composite(args.assets, fi, args.canvas)[0] for fi in idxs]
        strip = np.concatenate([np.pad(r, ((0, 0), (0, 6), (0, 0))) for r in rows], axis=1)
        sp = args.out.with_name(args.out.stem + "_frames" + args.out.suffix)
        Image.fromarray((np.clip(strip, 0, 1) * 255).astype(np.uint8)).save(sp)
        print(f"filmstrip frames {idxs} -> {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
