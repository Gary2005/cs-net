#!/usr/bin/env python3
"""Measure how much a lossy WebP round-trip actually distorts a normal map.

Size alone is not a result: a normal map that shrinks 3x but gains shading
noise is a regression. This decodes the compressed image back and measures the
per-pixel angular deviation of the reconstructed surface normal against the
original, plus a worst-region ASCII view so artefacts are visible without a
browser.

Normal maps here are stored in the usual tangent-space convention, so the
encoded RGB is mapped back to a unit vector before comparing.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

from PIL import Image

from encode_textures import classify


def decode_normal(im: Image.Image) -> list[tuple[float, float, float]]:
    """RGB -> unit normal vectors.

    Note: CS2 normal maps export as RGBA. Converting to RGB first is required --
    reading an RGBA image as if it were RGB shifts the channels and makes the
    blue channel read the alpha, which silently yields a near-constant field.
    """
    px = im.convert("RGB").load()
    w, h = im.size
    out = []
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            n = (r / 255 * 2 - 1, g / 255 * 2 - 1, b / 255 * 2 - 1)
            ln = math.sqrt(sum(v * v for v in n))
            out.append(tuple(v / ln for v in n) if ln > 1e-6 else (0.0, 0.0, 1.0))
    return out


def angular_error(a, b) -> float:
    d = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(d))


def ascii_map(errors: list[float], w: int, h: int, cols: int = 60) -> str:
    rows = max(1, int(cols * h / w / 2))
    chars = " .:-=+*#%@"
    lines = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            x0, x1 = c * w // cols, max(c * w // cols + 1, (c + 1) * w // cols)
            y0, y1 = r * h // rows, max(r * h // rows + 1, (r + 1) * h // rows)
            vals = [errors[y * w + x] for y in range(y0, y1) for x in range(x0, x1)]
            m = max(vals) if vals else 0.0
            # 0 deg -> blank, >=6 deg -> solid
            line += chars[min(9, int(m / 6.0 * 9.999))]
        lines.append(line)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="a normal-map PNG")
    ap.add_argument("--max-dim", type=int, default=1024)
    args = ap.parse_args(argv[1:])

    im = Image.open(args.src)
    scale = min(1.0, args.max_dim / max(im.size))
    if scale < 1.0:
        im = im.resize((max(1, round(im.width * scale)),
                        max(1, round(im.height * scale))), Image.LANCZOS)

    # Guard against degenerate inputs: a 1x1 or flat normal map makes every
    # encoder look perfect and the measurement meaningless.
    ref = decode_normal(im)
    spread = max(abs(v) for n in ref for v in n) - min(v for n in ref for v in n)
    if spread < 1e-3:
        print(f"{args.src.name}: DEGENERATE normal map (near-constant field) "
              f"-- nothing to measure")
        return 2

    print(f"file      : {args.src.name}")
    print(f"analysed  : {im.size[0]}x{im.size[1]}  ({len(ref):,} texels)")
    print()
    print(f"{'encoding':<16}{'bytes':>10}{'mean°':>9}{'p99°':>8}{'max°':>8}{'>2°':>9}{'>5°':>9}")

    for label, fmt, opts in (
        ("png (ref)", "PNG", {}),
        ("webp-lossless", "WEBP", {"lossless": True, "method": 6}),
        ("webp-q99", "WEBP", {"quality": 99, "method": 6}),
        ("webp-q99-444", "WEBP", {"quality": 99, "method": 6, "subsampling": "4:4:4"}),
        ("webp-q95-444", "WEBP", {"quality": 95, "method": 6, "subsampling": "4:4:4"}),
        ("webp-q90-444", "WEBP", {"quality": 90, "method": 6, "subsampling": "4:4:4"}),
        ("webp-q90", "WEBP", {"quality": 90, "method": 6}),
        ("jpeg-q95", "JPEG", {"quality": 95, "subsampling": 0}),
        ("jpeg-q92", "JPEG", {"quality": 92, "subsampling": 0}),
    ):
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format=fmt, **opts)
        nbytes = buf.tell()
        got = Image.open(io.BytesIO(buf.getvalue()))
        cur = decode_normal(got)
        errs = sorted(angular_error(a, b) for a, b in zip(ref, cur))
        n = len(errs)
        mean = sum(errs) / n
        p99 = errs[min(n - 1, int(n * 0.99))]
        mx = errs[-1]
        over2 = sum(1 for e in errs if e > 2) / n * 100
        over5 = sum(1 for e in errs if e > 5) / n * 100
        print(f"{label:<16}{nbytes:>10,}{mean:>9.3f}{p99:>8.2f}{mx:>8.2f}{over2:>8.1f}%{over5:>8.1f}%")

        if label == "webp-q90":
            print()
            print("  worst-region angular error map (webp-q90, blank=0deg, @>=6deg):")
            for line in ascii_map(errs, im.size[0], im.size[1]).splitlines():
                print("    " + line)
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
