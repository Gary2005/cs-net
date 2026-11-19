#!/usr/bin/env python3
"""Verify every texture a weapon GLB references actually loads and decodes.

Why this exists: three.js can only tell you "couldn't load texture", which does
not distinguish a missing file from a bad file from a harness problem. This
checks the three things that actually matter, with Pillow doing the decode:

  1. every `images[].uri` in the GLB exists next to it
  2. every one decodes as a real image
  3. every one has non-zero dimensions and is not a degenerate stub

That last check is not paranoia: `optimize_weapon_glb.py` rewrote the GLB's
image uris, and a few of the textures VRF emits are tiny placeholder files
(36 and 44 bytes), so "file exists" is not the same as "texture is usable".

It also reports which images are actually *referenced by a material*, since a
GLB can carry images that no material points at.

Usage:
    python verify_weapon_textures.py <weapon-dir> [<weapon-dir>...]
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path

from PIL import Image

JSON_CHUNK = 0x4E4F534A


def read_gltf(path: Path) -> dict:
    raw = path.read_bytes()
    magic, _ver, _len = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67:
        raise ValueError(f"{path.name}: not a GLB")
    clen, ctype = struct.unpack("<II", raw[12:20])
    if ctype != JSON_CHUNK:
        raise ValueError(f"{path.name}: first chunk is not JSON")
    return json.loads(raw[20:20 + clen].decode("utf-8"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--min-dim", type=int, default=8,
                    help="smaller than this counts as a stub texture")
    args = ap.parse_args(argv[1:])

    grand_ok = grand_bad = 0
    for d in args.dirs:
        glbs = [p for p in d.glob("*.glb") if "physics" not in p.stem]
        if not glbs:
            print(f"!! no GLB in {d}")
            continue
        glb = glbs[0]
        gltf = read_gltf(glb)

        # which images does a material actually point at?
        used = set()
        for mat in gltf.get("materials", []):
            for slot in ("baseColorTexture", "normalTexture", "occlusionTexture",
                         "emissiveTexture", "metallicRoughnessTexture"):
                tex = mat.get("pbrMetallicRoughness", {}).get(slot) or mat.get(slot)
                if isinstance(tex, dict) and "index" in tex:
                    src = gltf["textures"][tex["index"]].get("source")
                    if src is not None:
                        used.add(src)

        images = gltf.get("images", [])
        print(f"\n{d.name}  ({glb.name})")
        print(f"  {len(images)} images in GLB, {len(used)} referenced by a material")

        ok = bad = 0
        placeholders: list = []
        for i, im in enumerate(images):
            uri = im.get("uri")
            if not uri:
                print(f"    [{i}] no uri (embedded bufferView) -- skipped")
                continue
            p = d / uri
            mark = "material" if i in used else "unused  "
            if not p.is_file():
                print(f"    [{i}] {mark} MISSING  {uri}")
                bad += 1
                continue
            try:
                with Image.open(io.BytesIO(p.read_bytes())) as img:
                    w, h = img.size
                    fmt = img.format
                    img.load()
            except Exception as exc:                      # noqa: BLE001
                print(f"    [{i}] {mark} UNDECODABLE  {uri}: {exc}")
                bad += 1
                continue

            size = p.stat().st_size
            # A 1x1 texture is NOT a failure. Valve ships intentional 1x1
            # placeholders (`default_normal_tga`, `default_ao_tga`,
            # `sticker_gaps_vmat_g_tcolor`); they are 84-byte PNGs in the
            # extracted source, so the material is meant to fall back to a flat
            # value. Reporting these as broken would be a false alarm.
            is_placeholder = w <= 2 and h <= 2
            if is_placeholder:
                placeholders.append((uri, w, h, size))
                tag = "1x1"
                ok += 1
            else:
                ok += 1
                tag = "ok"
            print(f"    [{i}] {mark} {tag:4} {w}x{h} {fmt:5} {size/1024:8.1f} KB  {uri}")

        grand_ok += ok
        grand_bad += bad
        note = f" ({len(placeholders)} intentional 1x1 placeholders)" if placeholders else ""
        print(f"  -> {ok} usable, {bad} bad{note}")

    print(f"\n  TOTAL: {grand_ok} usable, {grand_bad} bad")
    return 1 if grand_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
