#!/usr/bin/env python3
"""Prune a Source2Viewer agent GLB down to a web-deliverable size.

Why this exists
---------------
`Source2Viewer-CLI --gltf_export_animations` on a CS2 agent embeds **every**
clip in the model's animation set -- for `ctm_sas` that is 2091 clips and a
1062 MB GLB. A replay viewer needs perhaps 30 of them. VRF's own
`--gltf_animation_list` only matches exact clip names (verified: a `*` glob
matches nothing), so the practical route is: export everything once, then keep
only the clips you need.

What it does
------------
* selects animations by regex against the clip name (keep-list and/or drop-list)
* rebuilds the binary buffer from only the accessors those animations need, so
  pruned clip data is genuinely removed rather than orphaned
* keeps all mesh/skin/material/image data untouched (pass-through)
* re-packs the JSON + BIN chunks with correct 4-byte alignment

Usage
-----
    python prune_glb.py in.glb out.glb --keep 'world/rifle' --keep 'idle'
    python prune_glb.py in.glb out.glb --drop 'ui_anims'
    python prune_glb.py in.glb out.glb --list          # just print clip names
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT_FMT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"{path}: not a GLB (magic={magic:#x})")
    if version != 2:
        raise ValueError(f"{path}: unsupported glTF container version {version}")
    off, chunks = 12, {}
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunks[ctype] = data[off:off + clen]
        off += clen
    return json.loads(chunks[CHUNK_JSON].decode("utf-8")), chunks.get(CHUNK_BIN, b"")


def write_glb(path: Path, gltf: dict, blob: bytes) -> None:
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    bn = blob + b"\x00" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + (8 + len(bn) if bn else 0)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(js), CHUNK_JSON) + js
    if bn:
        out += struct.pack("<II", len(bn), CHUNK_BIN) + bn
    path.write_bytes(bytes(out))


def accessor_bytes(gltf: dict, blob: bytes, idx: int) -> tuple[bytes, int]:
    """Return (raw bytes, element byte stride) for one accessor."""
    acc = gltf["accessors"][idx]
    n = COMPONENT_SIZE[acc["componentType"]] * TYPE_COUNT[acc["type"]]
    if "bufferView" not in acc:
        return b"\x00" * (n * acc["count"]), n
    bv = gltf["bufferViews"][acc["bufferView"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or n
    if stride == n:
        return blob[start:start + n * acc["count"]], n
    # de-interleave
    out = bytearray()
    for i in range(acc["count"]):
        out += blob[start + i * stride: start + i * stride + n]
    return bytes(out), n


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?")
    ap.add_argument("--keep", action="append", default=[],
                    help="regex; a clip is kept when it matches ANY --keep (default: keep all)")
    ap.add_argument("--drop", action="append", default=[],
                    help="regex; a clip is dropped when it matches ANY --drop")
    ap.add_argument("--list", action="store_true", help="print clip names and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv[1:])

    gltf, blob = read_glb(args.src)
    anims = gltf.get("animations", [])

    if args.list:
        for a in anims:
            print(a.get("name", "<unnamed>"))
        return 0
    if args.dst is None:
        ap.error("dst is required unless --list is given")

    keeps = [re.compile(p) for p in args.keep]
    drops = [re.compile(p) for p in args.drop]

    def wanted(name: str) -> bool:
        if any(d.search(name) for d in drops):
            return False
        if keeps:
            return any(k.search(name) for k in keeps)
        return True

    kept_idx = [i for i, a in enumerate(anims) if wanted(a.get("name", ""))]
    if not kept_idx:
        print("error: filter matched no animations", file=sys.stderr)
        return 2

    # --- collect every accessor we must retain -------------------------------
    # glTF points at accessors through a fixed set of key names (the value is the
    # index), so match on those rather than hoping the key is spelled "accessor".
    ACCESSOR_KEYS = {
        "indices", "inverseBindMatrices",
        "POSITION", "NORMAL", "TANGENT",
    }
    ACCESSOR_KEY_PREFIXES = ("TEXCOORD_", "COLOR_", "JOINTS_", "WEIGHTS_", "_")

    def is_accessor_key(k: str) -> bool:
        return k in ACCESSOR_KEYS or k.startswith(ACCESSOR_KEY_PREFIXES)

    mesh_acc: set[int] = set()

    def walk(node, key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, int) and is_accessor_key(k):
                    mesh_acc.add(v)
                else:
                    walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)

    walk({k: v for k, v in gltf.items() if k != "animations"})

    anim_acc: set[int] = set()
    for i in kept_idx:
        for s in anims[i].get("samplers", []):
            anim_acc.add(s["input"])
            anim_acc.add(s["output"])

    keep_acc = mesh_acc | anim_acc

    # --- rebuild the buffer, accessor by accessor ----------------------------
    new_blob = bytearray()
    new_views: list[dict] = []
    new_accessors: list[dict] = []
    acc_map: dict[int, int] = {}

    for old_idx in sorted(keep_acc):
        acc = dict(gltf["accessors"][old_idx])
        raw, _ = accessor_bytes(gltf, blob, old_idx)
        while len(new_blob) % 4:
            new_blob += b"\x00"
        view = {"buffer": 0, "byteOffset": len(new_blob), "byteLength": len(raw)}
        old_bv = gltf["bufferViews"][acc["bufferView"]] if "bufferView" in acc else {}
        if "target" in old_bv:
            view["target"] = old_bv["target"]
        new_blob += raw
        new_views.append(view)
        acc.pop("bufferView", None)
        acc.pop("byteOffset", None)
        acc["bufferView"] = len(new_views) - 1
        acc_map[old_idx] = len(new_accessors)
        new_accessors.append(acc)

    # --- rewrite animations --------------------------------------------------
    new_anims = []
    for i in kept_idx:
        a = dict(anims[i])
        a["samplers"] = [
            {"input": acc_map[s["input"]], "output": acc_map[s["output"]],
             **({"interpolation": s["interpolation"]} if "interpolation" in s else {})}
            for s in a.get("samplers", [])
        ]
        new_anims.append(a)

    gltf["animations"] = new_anims
    gltf["accessors"] = new_accessors
    gltf["bufferViews"] = new_views
    gltf["buffers"] = [{"byteLength": len(new_blob)}]

    # Remap every remaining accessor reference (mesh attributes, indices, morph
    # targets, skin inverseBindMatrices, ...) to its new index. Animation
    # samplers were remapped above, so skip that subtree.
    def remap(node, key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, int) and is_accessor_key(k):
                    node[k] = acc_map[v]
                else:
                    remap(v, k)
        elif isinstance(node, list):
            for v in node:
                remap(v, key)

    remap({k: v for k, v in gltf.items() if k != "animations"})

    write_glb(args.dst, gltf, bytes(new_blob))

    if not args.quiet:
        before = args.src.stat().st_size
        after = args.dst.stat().st_size
        print(f"animations : {len(anims)} -> {len(new_anims)}")
        print(f"accessors  : {len(gltf.get('accessors', []))} (kept {len(keep_acc)})")
        print(f"size       : {before/1048576:,.1f} MB -> {after/1048576:,.1f} MB "
              f"({after/before*100:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
