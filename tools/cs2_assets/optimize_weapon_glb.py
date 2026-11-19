#!/usr/bin/env python3
"""Shrink the staged weapon GLBs for web delivery.

VRF writes weapon textures at full source resolution -- 4096^2 PNGs, 15-18 MB
each -- so six weapons come to 167 MB. That is not shippable to a browser.

Two things have to happen together, which is why this is not just a resize:

  1. the PNGs are downscaled and re-encoded (colour lossy WebP, normal/ORM
     lossless -- see encode_textures.py for the measurements behind that split)
  2. the GLB's `images[].uri` entries are rewritten to the new filenames, since
     glTF references textures by name and a renamed file simply 404s

Rewriting the GLB means rebuilding the container: the JSON chunk changes length,
so the 12-byte header and the chunk table must be regenerated and each chunk
padded back to a 4-byte boundary.

Usage:
    python optimize_weapon_glb.py <weapon-dir> [<weapon-dir> ...]
        [--max-dim 1024] [--dry-run]

Accepts several directories so a whole tree can be done in one pass. Each
directory is optimized **once**: the source PNGs are deleted as they are
converted, so a second run finds nothing to do and says so instead of
dividing by zero.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import struct
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_textures import classify, POLICY  # noqa: E402

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def read_glb(path: Path):
    raw = path.read_bytes()
    magic, version, _length = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67:
        raise ValueError(f"{path.name}: not a GLB (magic={magic:#x})")
    off, chunks = 12, []
    while off < len(raw):
        clen, ctype = struct.unpack("<II", raw[off:off + 8])
        chunks.append((ctype, raw[off + 8:off + 8 + clen]))
        off += 8 + clen
    gltf = json.loads(chunks[0][1].decode("utf-8"))
    return version, gltf, chunks


def write_glb(path: Path, version: int, gltf: dict, chunks: list) -> int:
    """Rebuild the container, padding chunks to 4 bytes as the spec requires."""
    payload = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    body = struct.pack("<II", len(payload), JSON_CHUNK) + payload
    for ctype, data in chunks[1:]:
        pad = b"\x00" * (-len(data) % 4)
        body += struct.pack("<II", len(data) + len(pad), ctype) + data + pad
    out = struct.pack("<III", 0x46546C67, version, 12 + len(body)) + body
    path.write_bytes(out)
    return len(out)


def optimize_dir(d: Path, max_dim: int, dry_run: bool) -> dict | None:
    glbs = [p for p in d.glob("*.glb") if "physics" not in p.stem]
    if not glbs:
        print(f"{d.name}: no GLB", file=sys.stderr)
        return None
    glb_path = glbs[0]

    before_glb = glb_path.stat().st_size
    pngs = sorted(d.glob("*.png"))
    if not pngs:
        print(f"{d.name}: already optimized (no PNG left) -- skipped")
        return None
    before_tex = sum(p.stat().st_size for p in pngs)

    version, gltf, chunks = read_glb(glb_path)

    # 1. encode each texture; remember the rename
    renames: dict[str, str] = {}
    after_tex = 0
    for p in pngs:
        cls = classify(p.name)
        ext, fmt, opts = POLICY[cls]
        im = Image.open(p)
        scale = min(1.0, max_dim / max(im.size))
        if scale < 1.0:
            im = im.resize((max(1, round(im.width * scale)),
                            max(1, round(im.height * scale))), Image.LANCZOS)
        target = p.with_suffix(ext)
        if dry_run:
            buf = io.BytesIO()
            im.save(buf, format=fmt, **opts)
            after_tex += buf.tell()
        else:
            im.save(target, format=fmt, **opts)
            after_tex += target.stat().st_size
            if target.name != p.name:
                p.unlink()
        renames[p.name] = target.name

    # 2. point the glTF at the new filenames
    changed = 0
    for im in gltf.get("images", []):
        uri = im.get("uri")
        if uri and uri in renames and renames[uri] != uri:
            im["uri"] = renames[uri]
            changed += 1

    # 3. rebuild the container
    if dry_run:
        after_glb = before_glb
    else:
        after_glb = write_glb(glb_path, version, gltf, chunks)

    return {
        "name": d.name,
        "tex_before": before_tex, "tex_after": after_tex,
        "glb_before": before_glb, "glb_after": after_glb,
        "renames": changed,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+", help="weapon asset director(ies)")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv[1:])

    rows, skipped = [], 0
    for d in args.dirs:
        if not d.is_dir():
            print(f"{d}: not a directory", file=sys.stderr)
            skipped += 1
            continue
        r = optimize_dir(d, args.max_dim, args.dry_run)
        if r:
            rows.append(r)
        else:
            skipped += 1

    if not rows:
        print(f"\nnothing to do ({skipped} director(ies) skipped)")
        return 0

    tb = sum(r["tex_before"] for r in rows)
    ta = sum(r["tex_after"] for r in rows)
    gb = sum(r["glb_before"] for r in rows)
    ga = sum(r["glb_after"] for r in rows)
    worst = sorted(rows, key=lambda r: r["tex_before"] + r["glb_before"],
                   reverse=True)[:5]
    print(f"\n{'weapon':<22}{'before':>10}{'after':>10}{'ratio':>8}  uri-refs")
    for r in worst:
        b = r["tex_before"] + r["glb_before"]
        a = r["tex_after"] + r["glb_after"]
        print(f"{r['name']:<22}{b/1048576:>8.1f}MB{a/1048576:>8.1f}MB{a/b*100:>7.1f}%"
              f"{r['renames']:>9}")
    print(f"{'...':<22}")
    print(f"{'TOTAL (' + str(len(rows)) + ' dirs)':<22}"
          f"{(tb+gb)/1048576:>8.1f}MB{(ta+ga)/1048576:>8.1f}MB"
          f"{(ta+ga)/(tb+gb)*100:>7.1f}%")
    if skipped:
        print(f"({skipped} director(ies) skipped)")
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
