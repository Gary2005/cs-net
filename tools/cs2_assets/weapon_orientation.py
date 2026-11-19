#!/usr/bin/env python3
"""Work out which way a weapon model points, from its geometry.

The renderers can show a human which way the barrel faces, but code cannot read
a PNG. This measures it instead, which is what the mount code actually needs.

Two independent metrics are reported, because neither works for everything:

1. **Cross-section** -- slice the vertices along the long axis and compare the
   cross-sectional extent at the two ends. The muzzle/blade-tip end of a
   straight object is much thinner than the stock/handle end. This is the
   reliable one for guns and straight knives.

   It fails on a **curved** blade: the karambit's blade sweeps sideways, so its
   z-extent (0.195) is nearly as large as its y-extent (0.249), and the test
   picks the ring end instead of the tip.

2. **Farthest vertex from the centroid** -- the tip of anything is its most
   distant point. This is curvature-independent, so it settles the karambit.
   But it fails on guns: the AK's stock is farther from the centroid than its
   muzzle (the heavy receiver pulls the centroid forward), so it reports -z.

So: trust (1) for guns, (2) for knives, and when they disagree, say so.

Usage:
    python weapon_orientation.py <model.glb> [more.glb ...] [--bins 10]

With more than one model it prints a compact table instead of full slice dumps.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
        5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_glb(path: Path):
    raw = path.read_bytes()
    off, chunks = 12, []
    while off < len(raw):
        clen, ctype = struct.unpack("<II", raw[off:off + 8])
        chunks.append((ctype, raw[off + 8:off + 8 + clen]))
        off += 8 + clen
    gltf = json.loads(chunks[0][1].decode("utf-8"))
    bin_chunk = next((c[1] for c in chunks[1:] if c[0] == 0x004E4942), b"")
    return gltf, bin_chunk


def read_positions(gltf, blob, prim):
    """Yield (x, y, z) for every vertex of a primitive, in node-local space."""
    acc_i = prim.get("attributes", {}).get("POSITION")
    if acc_i is None:
        return []
    acc = gltf["accessors"][acc_i]
    bv = gltf["bufferViews"][acc["bufferView"]]
    fmt, size = COMP[acc["componentType"]]
    n = NCOMP[acc["type"]]
    stride = bv.get("byteStride") or size * n
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    out = []
    for i in range(acc["count"]):
        o = base + i * stride
        out.append(struct.unpack_from("<" + fmt * n, blob, o))
    return out


def load_points(path: Path) -> list[tuple]:
    gltf, blob = load_glb(path)
    pts = []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pts.extend(read_positions(gltf, blob, prim))
    return pts


def measure(path: Path, bins_n: int = 10) -> dict | None:
    pts = load_points(path)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    ext = {"x": max(xs) - min(xs), "y": max(ys) - min(ys), "z": max(zs) - min(zs)}
    axis = max(ext, key=ext.get)
    series = {"x": xs, "y": ys, "z": zs}
    others = [a for a in "xyz" if a != axis]

    lo, hi = min(series[axis]), max(series[axis])
    span = hi - lo
    bins: list[list] = [[] for _ in range(bins_n)]
    for p in pts:
        t = (p["xyz".index(axis)] - lo) / span if span else 0.0
        bins[min(bins_n - 1, int(t * bins_n))].append(p)

    prof = []
    for b in bins:
        if not b:
            prof.append(0.0)
            continue
        prof.append(max(max(p["xyz".index(o)] for p in b)
                        - min(p["xyz".index(o)] for p in b) for o in others))

    thin_dir = None
    if prof[0] or prof[-1]:
        thin_dir = f"+{axis}" if prof[-1] < prof[0] else f"-{axis}"

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)
    far = max(pts, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2)
    d = [far[0] - cx, far[1] - cy, far[2] - cz]
    dn = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5 or 1.0
    dom_i = max(range(3), key=lambda i: abs(d[i]))
    tip_dir = f"{'+' if d[dom_i] > 0 else '-'}{'xyz'[dom_i]}"
    tip_share = abs(d[dom_i]) / dn

    above = {}
    for o in others:
        v = series[o]
        c = (max(v) + min(v)) / 2
        above[o] = sum(1 for x in v if x > c) / len(v)

    # Which perpendicular axis is the TALL one. This -- not `above` -- is the
    # evidence for "which way is up": a gun is taller (grip to sights) than it is
    # wide, so the perpendicular axis with the larger extent is the vertical one.
    #
    # `above` measures where the ORIGIN sits inside the bounding box, which is a
    # different thing and varies wildly: the AWP's origin is nearly centred in y
    # (48% above), the M249's sits near the top (13% above). Reading either as
    # "up" gives the wrong answer. For guns the origin is deliberately at the
    # grip, which is why `anchor: 'origin'` is used and why `above` is skewed.
    tall = max(others, key=lambda o: ext[o]) if others else None
    short = min(others, key=lambda o: ext[o]) if others else None

    return {
        "name": path.stem,
        "verts": len(pts),
        "ext": ext,
        "axis": axis,
        "thin_dir": thin_dir,
        "thin_ends": (prof[0], prof[-1]),
        "tip_dir": tip_dir,
        "tip_share": tip_share,
        "above": above,
        "tall": tall,
        "tall_ratio": (ext[tall] / ext[short]) if short and ext[short] else 0.0,
        "prof": prof,
        "others": others,
    }


def print_detail(m: dict) -> None:
    print(f"{m['name']}: {m['verts']:,} vertices")
    print(f"  extent  x={m['ext']['x']:.3f}  y={m['ext']['y']:.3f}  z={m['ext']['z']:.3f}  "
          f"-> long axis = {m['axis']}")
    print(f"\n  slices along {m['axis']}:")
    print(f"    {'bin':>3} {'extent':>9}")
    for k, e in enumerate(m["prof"]):
        print(f"    {k:>3} {e:>9.3f}")
    lo_e, hi_e = m["thin_ends"]
    print(f"\n  cross-section at lo end = {lo_e:.3f}, at hi end = {hi_e:.3f}")
    print(f"  -> [cross-section] the THIN end is at {m['thin_dir']}")
    print(f"  -> [farthest vertex] TIP direction = {m['tip_dir']} "
          f"({m['tip_share']*100:.0f}% of the offset)")
    print(f"  -> [tall axis] {m['tall']} is {m['tall_ratio']:.1f}x the other "
          f"perpendicular axis -> vertical axis = {m['tall']}")
    print()
    for o, f in m["above"].items():
        print(f"  {o}: {f*100:.0f}% of vertices above centre "
              f"(origin placement, NOT the up direction)")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", type=Path, nargs="+")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args(argv[1:])

    results = []
    for p in args.glb:
        m = measure(p, args.bins)
        if m is None:
            print(f"{p}: no POSITION data", file=sys.stderr)
            continue
        results.append(m)

    if len(results) == 1:
        print_detail(results[0])
        return 0

    print(f"{'model':<26}{'axis':>5}{'cross-sec':>11}{'tip':>6}{'share':>7}"
          f"{'tall':>6}{'ratio':>7}  agree")
    disagree = 0
    for m in sorted(results, key=lambda r: r["name"]):
        # cross-section and farthest-vertex agree if both name the same signed axis
        agree = m["thin_dir"] == m["tip_dir"]
        if not agree:
            disagree += 1
        print(f"{m['name']:<26}{m['axis']:>5}{m['thin_dir'] or '?':>11}"
              f"{m['tip_dir']:>6}{m['tip_share']*100:>6.0f}%"
              f"{m['tall'] or '?':>6}{m['tall_ratio']:>6.1f}x"
              f"  {'yes' if agree else 'NO'}")
    print(f"\n{len(results)} models, {disagree} where the two metrics disagree")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
