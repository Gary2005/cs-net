#!/usr/bin/env python3
"""Inspect a glTF/GLB produced by Source2Viewer-CLI without loading it into Blender.

Prints asset counts, the animation-name list, per-accessor sizes and which
buffer views dominate the file. That is what decides whether an export is
web-deliverable or needs trimming.
"""

from __future__ import annotations

import json
import struct
import sys
from collections import Counter
from pathlib import Path

COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, f"not a GLB: magic={magic:#x}"
    off, chunks = 12, {}
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunks[ctype] = data[off:off + clen]
        off += clen
    gltf = json.loads(chunks[0x4E4F534A].decode("utf-8"))
    return gltf, chunks.get(0x004E4942, b"")


def acc_bytes(gltf: dict, idx: int) -> int:
    a = gltf["accessors"][idx]
    return a["count"] * COMPONENT_SIZE.get(a["componentType"], 4) * TYPE_COUNT.get(a["type"], 1)


def main(argv: list[str]) -> int:
    path = Path(argv[1])
    gltf, binchunk = read_glb(path)

    print(f"file            : {path.name}  ({path.stat().st_size / 1048576:,.1f} MB)")
    print(f"glTF version    : {gltf['asset'].get('version')}  generator={gltf['asset'].get('generator')}")
    print(f"BIN chunk       : {len(binchunk) / 1048576:,.1f} MB")
    for key in ("scenes", "nodes", "meshes", "skins", "materials", "textures", "images",
                "animations", "accessors", "bufferViews", "cameras"):
        if key in gltf:
            print(f"{key:<16}: {len(gltf[key])}")

    if gltf.get("skins"):
        for i, sk in enumerate(gltf["skins"][:3]):
            print(f"  skin[{i}] joints={len(sk.get('joints', []))} name={sk.get('name')!r}")

    # mesh primitive / vertex accounting
    total_verts = total_tris = 0
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None:
                total_verts += gltf["accessors"][pos]["count"]
            if "indices" in prim:
                total_tris += gltf["accessors"][prim["indices"]]["count"] // 3
    print(f"vertices        : {total_verts:,}")
    print(f"triangles       : {total_tris:,}")

    anims = gltf.get("animations", [])
    print(f"\nanimations ({len(anims)}):")
    for i, an in enumerate(anims):
        ch = an.get("channels", [])
        smp = an.get("samplers", [])
        nbytes = sum(acc_bytes(gltf, s["input"]) + acc_bytes(gltf, s["output"]) for s in smp)
        # longest keyframe time
        tmax = 0.0
        for s in smp[:1]:
            acc = gltf["accessors"][s["input"]]
            if "max" in acc:
                tmax = acc["max"][0]
        print(f"  [{i:>3}] {an.get('name','<unnamed>')[:70]:<72} ch={len(ch):>4} bytes={nbytes/1048576:>8.2f} MB  t<= {tmax:.2f}s")

    # what dominates
    print("\nlargest bufferViews:")
    views = sorted(enumerate(gltf.get("bufferViews", [])), key=lambda kv: -kv[1]["byteLength"])[:12]
    for i, v in views:
        users = [f"acc:{j}" for j, a in enumerate(gltf.get("accessors", [])) if a.get("bufferView") == i]
        print(f"  bv[{i:>4}] {v['byteLength']/1048576:>8.2f} MB  target={v.get('target')}  {', '.join(users[:4])}")

    # attribute usage census
    attrs: Counter[str] = Counter()
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs.update(prim.get("attributes", {}).keys())
    print(f"\nvertex attributes: {dict(attrs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
