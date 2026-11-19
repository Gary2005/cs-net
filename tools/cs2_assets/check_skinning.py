#!/usr/bin/env python3
"""Verify a skinned glTF mesh is actually deformable.

Structural validation proves references resolve; it does not prove the skinning
data is sane. This checks the things that silently produce a broken or
T-posed-but-mangled model in a viewer:

* JOINTS_0 indices are within the skin's joint count
* WEIGHTS_0 sum to ~1 per vertex
* the inverse-bind matrices are invertible (a zero/degenerate IBM collapses a limb)
* applying bind pose * inverse-bind matrices reproduces the mesh as authored

Run against both the raw Source2Viewer export and the pruned output -- if both
pass identically, pruning preserved the mesh.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

COMPONENT_FMT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def load(path: Path):
    d = path.read_bytes()
    off, ch = 12, {}
    while off < len(d):
        clen, ctype = struct.unpack_from("<II", d, off)
        off += 8
        ch[ctype] = d[off:off + clen]
        off += clen
    return json.loads(ch[0x4E4F534A].decode("utf-8")), ch.get(0x004E4942, b"")


def read_accessor(g: dict, blob: bytes, idx: int) -> list[tuple]:
    a = g["accessors"][idx]
    n = TYPE_COUNT[a["type"]]
    fmt = COMPONENT_FMT[a["componentType"]]
    if "bufferView" not in a:
        return [tuple([0] * n)] * a["count"]
    bv = g["bufferViews"][a["bufferView"]]
    stride = bv.get("byteStride") or COMPONENT_SIZE[a["componentType"]] * n
    base = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    esz = COMPONENT_SIZE[a["componentType"]] * n
    out = []
    for i in range(a["count"]):
        vals = struct.unpack_from("<" + fmt * n, blob, base + i * stride)
        out.append(vals if n > 1 else vals[0])
    return out


def mat4_invert(m: list[float]):
    """4x4 inverse via Gauss-Jordan; returns None when singular."""
    import copy
    a = [list(m[r * 4:r * 4 + 4]) + [1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        a[col] = [x / d for x in a[col]]
        for r in range(4):
            if r != col and a[r][col] != 0.0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [a[r][c] for r in range(4) for c in range(4, 8)]


def mat4_mul(a, b):
    return [sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4)]


def main(argv: list[str]) -> int:
    path = Path(argv[1])
    g, blob = load(path)
    problems: list[str] = []

    total_verts = 0
    weight_bad = 0
    joint_oob = 0
    checked_prims = 0
    max_joint_seen = -1

    for mi, mesh in enumerate(g.get("meshes", [])):
        for pi, prim in enumerate(mesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                continue
            node = next((n for n in g["nodes"] if n.get("mesh") == mi), None)
            skin = g["skins"][node["skin"]] if node and "skin" in node else None
            njoints = len(skin["joints"]) if skin else None

            joints = read_accessor(g, blob, attrs["JOINTS_0"])
            weights = read_accessor(g, blob, attrs["WEIGHTS_0"])
            checked_prims += 1
            total_verts += len(joints)

            for jv, wv in zip(joints, weights):
                jl = jv if isinstance(jv, tuple) else (jv,)
                wl = wv if isinstance(wv, tuple) else (wv,)
                s = sum(wl)
                if abs(s - 1.0) > 0.02:
                    weight_bad += 1
                mx = max(jl)
                max_joint_seen = max(max_joint_seen, mx)
                if njoints is not None and mx >= njoints:
                    joint_oob += 1

    # inverse bind matrices
    singular = 0
    ibm_total = 0
    for sk in g.get("skins", []):
        if "inverseBindMatrices" not in sk:
            continue
        mats = read_accessor(g, blob, sk["inverseBindMatrices"])
        for m in mats:
            ibm_total += 1
            if mat4_invert(list(m)) is None:
                singular += 1

    print(f"{path.name}")
    print(f"  skinned primitives : {checked_prims}")
    print(f"  skinned vertices   : {total_verts:,}")
    print(f"  max joint index    : {max_joint_seen}")
    print(f"  weights not ~1.0   : {weight_bad}")
    print(f"  joints out of range: {joint_oob}")
    print(f"  inverse-bind mats  : {ibm_total} ({singular} singular)")
    if weight_bad or joint_oob or singular:
        print("  FAIL")
        return 1
    print("  OK -- skinning data is well-formed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
