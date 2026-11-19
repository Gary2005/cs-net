#!/usr/bin/env python3
"""Structural validation of a glTF/GLB: does every reference actually resolve?

Catches the failure modes a pruner can introduce -- dangling accessor or
bufferView indices, bufferViews running past the end of the buffer, animation
samplers whose input/output accessors are gone, and channels pointing at nodes
that no longer exist. Run this after any rewrite; a GLB can look fine in a JSON
dump and still be unusable.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

GLB_MAGIC = 0x46546C67
COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def load(path: Path) -> tuple[dict, bytes]:
    d = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", d, 0)
    if magic != GLB_MAGIC:
        raise ValueError("not a GLB")
    if length != len(d):
        raise ValueError(f"header length {length} != file size {len(d)}")
    off, ch = 12, {}
    while off < length:
        clen, ctype = struct.unpack_from("<II", d, off)
        off += 8
        ch[ctype] = d[off:off + clen]
        off += clen
    return json.loads(ch[0x4E4F534A].decode("utf-8")), ch.get(0x004E4942, b"")


def main(argv: list[str]) -> int:
    path = Path(argv[1])
    g, blob = load(path)
    errors: list[str] = []
    warnings: list[str] = []

    n_acc = len(g.get("accessors", []))
    n_bv = len(g.get("bufferViews", []))
    n_nodes = len(g.get("nodes", []))
    buf_len = g["buffers"][0]["byteLength"]

    if buf_len != len(blob):
        errors.append(f"buffer byteLength {buf_len} != BIN chunk size {len(blob)}")

    # bufferViews inside the buffer
    for i, bv in enumerate(g.get("bufferViews", [])):
        end = bv.get("byteOffset", 0) + bv["byteLength"]
        if end > buf_len:
            errors.append(f"bufferView[{i}] ends at {end} > buffer {buf_len}")
        if bv.get("buffer", 0) != 0:
            errors.append(f"bufferView[{i}] references buffer {bv.get('buffer')} (only 0 exists)")

    # accessors resolve and fit
    for i, a in enumerate(g.get("accessors", [])):
        if "bufferView" in a:
            if a["bufferView"] >= n_bv:
                errors.append(f"accessor[{i}] bufferView {a['bufferView']} out of range ({n_bv})")
                continue
            bv = g["bufferViews"][a["bufferView"]]
            need = COMPONENT_SIZE[a["componentType"]] * TYPE_COUNT[a["type"]]
            stride = bv.get("byteStride") or need
            span = (a["count"] - 1) * stride + need if a["count"] else 0
            avail = bv["byteLength"] - a.get("byteOffset", 0)
            if span > avail:
                errors.append(
                    f"accessor[{i}] needs {span} bytes but bufferView has {avail}")
        if a.get("sparse"):
            for which in ("indices", "values"):
                sub = a["sparse"].get(which, {})
                if "bufferView" in sub and sub["bufferView"] >= n_bv:
                    errors.append(f"accessor[{i}].sparse.{which} bufferView out of range")

    # mesh attribute / index references
    for mi, mesh in enumerate(g.get("meshes", [])):
        for pi, prim in enumerate(mesh.get("primitives", [])):
            for key, val in list(prim.get("attributes", {}).items()):
                if val >= n_acc:
                    errors.append(f"mesh[{mi}].prim[{pi}].{key} accessor {val} >= {n_acc}")
            if "indices" in prim and prim["indices"] >= n_acc:
                errors.append(f"mesh[{mi}].prim[{pi}].indices accessor {prim['indices']} >= {n_acc}")
            for ti, tgt in enumerate(prim.get("targets", []) or []):
                for key, val in tgt.items():
                    if val >= n_acc:
                        errors.append(f"mesh[{mi}].prim[{pi}].target[{ti}].{key} accessor {val} >= {n_acc}")

    # skins
    for si, sk in enumerate(g.get("skins", [])):
        if "inverseBindMatrices" in sk and sk["inverseBindMatrices"] >= n_acc:
            errors.append(f"skin[{si}].inverseBindMatrices {sk['inverseBindMatrices']} >= {n_acc}")
        for j in sk.get("joints", []):
            if j >= n_nodes:
                errors.append(f"skin[{si}] joint node {j} >= {n_nodes}")

    # animations
    for ai, an in enumerate(g.get("animations", [])):
        samplers = an.get("samplers", [])
        for k, s in enumerate(samplers):
            for which in ("input", "output"):
                if s[which] >= n_acc:
                    errors.append(f"anim[{ai}]{an.get('name','')[:40]} sampler[{k}].{which} accessor {s[which]} >= {n_acc}")
        for c in an.get("channels", []):
            if c["sampler"] >= len(samplers):
                errors.append(f"anim[{ai}] channel sampler {c['sampler']} >= {len(samplers)}")
            if c["target"]["node"] >= n_nodes:
                errors.append(f"anim[{ai}] channel node {c['target']['node']} >= {n_nodes}")

    # images / textures
    for i, im in enumerate(g.get("images", [])):
        if "bufferView" in im and im["bufferView"] >= n_bv:
            errors.append(f"image[{i}] bufferView {im['bufferView']} >= {n_bv}")
        if "uri" not in im and "bufferView" not in im:
            errors.append(f"image[{i}] has neither uri nor bufferView")
    for i, t in enumerate(g.get("textures", [])):
        if t.get("source", 0) >= len(g.get("images", [])):
            errors.append(f"texture[{i}] source {t.get('source')} out of range")

    external = [im["uri"] for im in g.get("images", []) if "uri" in im]
    if external:
        warnings.append(f"{len(external)} image(s) use an external URI "
                        f"(e.g. {external[0][:70]}) -- those files must ship alongside")

    print(f"{path.name}: {path.stat().st_size/1048576:,.2f} MB | "
          f"acc={n_acc} bv={n_bv} nodes={n_nodes} "
          f"anims={len(g.get('animations', []))} meshes={len(g.get('meshes', []))}")
    if warnings:
        for w in warnings:
            print(f"  warn : {w}")
    if errors:
        for e in errors[:25]:
            print(f"  ERROR: {e}")
        print(f"  ... {len(errors)} error(s) total")
        return 1
    print("  OK -- all references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
