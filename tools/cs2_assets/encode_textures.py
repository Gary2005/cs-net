#!/usr/bin/env python3
"""Downscale + recompress extracted CS2 textures for web delivery.

Source2Viewer writes every texture at full source resolution as PNG -- 2048^2
for agents and up to 4096^2 for weapons -- which is 49 MB for one agent's
materials and 143 MB for six weapons' worth. VRF has no resize or recompress
flag, so this post-process is the only lever.

Format choice is per texture *class*, because the encoders are not
interchangeable (all measured on `glove_hardknuckle_normal`, 512^2, angular
error in degrees):

    encoding        bytes   mean    p99    max   >2deg  >5deg
    png (ref)      494,971  0.000   0.00   0.00   0.0%   0.0%
    webp-lossless  414,682  0.000   0.00   0.00   0.0%   0.0%
    webp-q99       138,992  3.603  19.25  71.55  48.6%  22.5%
    webp-q90        75,034  3.975  19.54  80.59  58.4%  24.6%
    jpeg-q95       205,344  1.664   4.90   9.38  31.0%   0.9%

So lossy WebP is **not** usable for normal maps (24.6% of texels off by >5deg
even at q99 -- shading noise), while JPEG q95 keeps that to 0.9%. Colour maps
are the opposite: lossy WebP wins by a wide margin.

Resulting policy:
    colour          -> WebP q90        (8 agent maps: 4.8 MB -> 0.3 MB)
    normal          -> WebP lossless   (10 agent maps: 6.9 MB -> 5.5 MB)
    orm/ao/mask     -> WebP lossless   (18 agent maps: 2.4 MB -> 1.4 MB)

Usage:
    python encode_textures.py <src-dir> --out <dir> [--max-dim 1024] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

CLASSIFIERS: list[tuple[str, re.Pattern[str]]] = [
    ("normal", re.compile(r"normal", re.I)),
    ("orm", re.compile(r"(_orm|_ao_|ambientocclusion|metal|rough|mask|cloth)", re.I)),
    ("color", re.compile(r"color|albedo|diffuse", re.I)),
]

# class -> (extension, PIL format, save kwargs)
POLICY = {
    "color": (".webp", "WEBP", {"quality": 90, "method": 6}),
    "normal": (".webp", "WEBP", {"lossless": True, "method": 6}),
    "orm": (".webp", "WEBP", {"lossless": True, "method": 6}),
    "other": (".webp", "WEBP", {"lossless": True, "method": 6}),
}


def classify(name: str) -> str:
    for label, pat in CLASSIFIERS:
        if pat.search(name):
            return label
    return "other"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv[1:])

    files = sorted(args.src.rglob("*.png"))
    if not files:
        print(f"no PNGs under {args.src}", file=sys.stderr)
        return 1

    stats: dict[str, list[int]] = {}
    written = 0
    for f in files:
        cls = classify(f.name)
        before = f.stat().st_size
        im = Image.open(f)
        scale = min(1.0, args.max_dim / max(im.size))
        if scale < 1.0:
            im = im.resize((max(1, round(im.width * scale)),
                            max(1, round(im.height * scale))), Image.LANCZOS)

        ext, fmt, opts = POLICY[cls]
        if not args.dry_run:
            rel = f.relative_to(args.src).parent
            d = args.out / rel
            d.mkdir(parents=True, exist_ok=True)
            target = d / (f.stem + ext)
            im.save(target, format=fmt, **opts)
            after = target.stat().st_size
            written += 1
        else:
            import io
            buf = io.BytesIO()
            im.save(buf, format=fmt, **opts)
            after = buf.tell()

        s = stats.setdefault(cls, [0, 0, 0])
        s[0] += 1
        s[1] += before
        s[2] += after

    tb = sum(v[1] for v in stats.values())
    ta = sum(v[2] for v in stats.values())
    print(f"{'class':<8}{'n':>5}{'before':>12}{'after':>12}{'ratio':>9}   policy")
    for cls, (n, b, a) in sorted(stats.items()):
        ext, fmt, opts = POLICY[cls]
        desc = f"WEBP q{opts['quality']}" if "quality" in opts else "WEBP lossless"
        print(f"{cls:<8}{n:>5}{b/1048576:>10.1f}MB{a/1048576:>10.1f}MB{a/b*100:>8.1f}%   {desc}")
    print(f"{'TOTAL':<8}{sum(v[0] for v in stats.values()):>5}"
          f"{tb/1048576:>10.1f}MB{ta/1048576:>10.1f}MB{ta/tb*100:>8.1f}%")
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    else:
        print(f"\nwrote {written} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
