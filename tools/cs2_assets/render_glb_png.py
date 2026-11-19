#!/usr/bin/env python3
"""Render a GLB to a PNG with a real perspective camera and depth buffer.

The ASCII rasteriser in render_glb_ascii.py is fine for "is this animating" but
useless for "does this model look right": it is orthographic, has no shading,
and collapses a rifle viewed down its barrel into an unreadable blob. Deciding
whether a weapon's bind pose is correctly assembled needs a real image.

Reuses the glTF plumbing (load_glb, world_matrices, accessor reads, matrix ops)
from render_glb_ascii so there is one parser, not two.

Two output modes:
  * normal   -- shaded 3/4 view, for judging overall silhouette
  * by-joint -- each vertex tinted by its dominant joint index, for spotting
                parts that are detached or left at the origin

Usage:
    python render_glb_png.py <model.glb> --out out.png [--view 3q|front|side|top]
                             [--mode normal|by-joint] [--size 900]
                             [--clip NAME] [--time 0.0] [--turntable 4]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_glb_ascii import (  # noqa: E402
    Anim, gather_vertices, load_glb, m_apply, read_accessor, world_matrices,
)

VIEWS = {
    "3q": (35.0, 24.0),      # azimuth, elevation in degrees
    "front": (0.0, 0.0),
    "side": (90.0, 0.0),
    "top": (0.0, 88.0),
    "back": (180.0, 10.0),
}

# distinct hues for the by-joint mode
JOINT_COLOURS = [
    (0.90, 0.25, 0.25), (0.25, 0.75, 0.35), (0.30, 0.45, 0.95), (0.95, 0.75, 0.20),
    (0.75, 0.35, 0.85), (0.25, 0.80, 0.85), (0.95, 0.55, 0.20), (0.55, 0.55, 0.60),
    (0.85, 0.35, 0.55), (0.45, 0.85, 0.55), (0.35, 0.55, 0.75), (0.80, 0.80, 0.30),
]


def gather_triangles(g, blob, world):
    """World-space triangles with per-triangle normals.

    Indices are required: without them we would have to guess the winding and
    the result is a soup. Every VRF export carries them.
    """
    tris = []
    for mi, mesh in enumerate(g["meshes"]):
        node = next((n for n in g["nodes"] if n.get("mesh") == mi), None)
        if node is None:
            continue
        node_i = g["nodes"].index(node)
        node_world = world[node_i]
        skin = g["skins"][node["skin"]] if "skin" in node else None

        for prim in mesh["primitives"]:
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs or "indices" not in prim:
                continue
            pos = read_accessor(g, blob, attrs["POSITION"])
            idx = [int(i[0]) for i in read_accessor(g, blob, prim["indices"])]

            joints = weights = None
            if skin and "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                joints = read_accessor(g, blob, attrs["JOINTS_0"])
                weights = read_accessor(g, blob, attrs["WEIGHTS_0"])
                ibm = read_accessor(g, blob, skin["inverseBindMatrices"])
                jn = skin["joints"]
                jm = [m_mul(world[jn[k]], ibm[k]) for k in range(len(jn))]

            world_pts, joint_ids = [], []
            for vi, p in enumerate(pos):
                if joints is None:
                    world_pts.append(m_apply(node_world, p))
                    joint_ids.append(-1)
                    continue
                jv, wv = joints[vi], weights[vi]
                acc = [0.0, 0.0, 0.0]
                best, bestw = -1, -1.0
                for k in range(4):
                    w = wv[k]
                    if w <= 0:
                        continue
                    q = m_apply(jm[jv[k]], p)
                    acc[0] += q[0] * w
                    acc[1] += q[1] * w
                    acc[2] += q[2] * w
                    if w > bestw:
                        best, bestw = int(jv[k]), w
                world_pts.append(tuple(acc))
                joint_ids.append(best)

            for t in range(0, len(idx) - 2, 3):
                a, b, c = idx[t], idx[t + 1], idx[t + 2]
                if a >= len(world_pts) or b >= len(world_pts) or c >= len(world_pts):
                    continue
                tris.append((world_pts[a], world_pts[b], world_pts[c],
                             joint_ids[a]))
    return tris


def normal_of(p0, p1, p2):
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (nx / ln, ny / ln, nz / ln) if ln > 1e-12 else (0.0, 0.0, 1.0)


def render(tris, size, azim, elev, mode, label):
    if not tris:
        return None
    pts = [p for t in tris for p in t[:3]]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)

    a, e = math.radians(azim), math.radians(elev)
    ca, sa, ce, se = math.cos(a), math.sin(a), math.cos(e), math.sin(e)
    # camera basis: right, up, forward(view direction)
    right = (ca, 0.0, -sa)
    up = (-sa * se, ce, -ca * se)
    fwd = (sa * ce, se, ca * ce)

    # fit
    r = 0.0
    for p in pts:
        d = (p[0] - cx, p[1] - cy, p[2] - cz)
        r = max(r, math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
    if r <= 1e-9:
        r = 1.0
    dist = r * 2.6
    f = (size * 0.5) / math.tan(math.radians(22.0))

    zbuf = [1e30] * (size * size)
    img = Image.new("RGB", (size, size), (16, 18, 22))
    px = img.load()

    def project(p):
        dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
        # camera at -fwd*dist looking toward +fwd
        vx = dx * right[0] + dy * right[1] + dz * right[2]
        vy = dx * up[0] + dy * up[1] + dz * up[2]
        vz = dx * fwd[0] + dy * fwd[1] + dz * fwd[2] + dist
        if vz <= 1e-6:
            return None
        return (size * 0.5 + f * vx / vz, size * 0.5 - f * vy / vz, vz)

    light = (-0.35, 0.75, 0.55)
    ln = math.sqrt(sum(c * c for c in light))
    light = tuple(c / ln for c in light)

    for p0, p1, p2, jid in tris:
        s = [project(p0), project(p1), project(p2)]
        if any(v is None for v in s):
            continue
        n = normal_of(p0, p1, p2)
        # two-sided shading
        lam = abs(n[0] * light[0] + n[1] * light[1] + n[2] * light[2])
        shade = 0.18 + 0.82 * lam

        if mode == "by-joint" and jid >= 0:
            base = JOINT_COLOURS[jid % len(JOINT_COLOURS)]
        else:
            base = (0.72, 0.74, 0.78)
        col = tuple(int(max(0, min(255, c * shade * 255))) for c in base)

        x0, y0 = s[0][0], s[0][1]
        x1, y1 = s[1][0], s[1][1]
        x2, y2 = s[2][0], s[2][1]
        minx, maxx = max(0, int(min(x0, x1, x2))), min(size - 1, int(max(x0, x1, x2)) + 1)
        miny, maxy = max(0, int(min(y0, y1, y2))), min(size - 1, int(max(y0, y1, y2)) + 1)
        if minx > maxx or miny > maxy:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        for yy in range(miny, maxy + 1):
            for xx in range(minx, maxx + 1):
                l0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / den
                l1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / den
                l2 = 1.0 - l0 - l1
                if l0 < -1e-9 or l1 < -1e-9 or l2 < -1e-9:
                    continue
                z = l0 * s[0][2] + l1 * s[1][2] + l2 * s[2][2]
                o = yy * size + xx
                if z < zbuf[o]:
                    zbuf[o] = z
                    px[xx, yy] = col

    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.text((8, 6), label, fill=(210, 215, 225))
    return img


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--view", default="3q", choices=sorted(VIEWS))
    ap.add_argument("--mode", default="normal", choices=["normal", "by-joint"])
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--turntable", type=int, default=0,
                    help="if >0, render N evenly spaced azimuths into one strip")
    args = ap.parse_args(argv[1:])

    g, blob = load_glb(args.glb)
    anims = g.get("animations", [])

    override = {}
    t = 0.0
    if anims:
        anim = None
        if args.clip:
            anim = next((a for a in anims if args.clip in a.get("name", "")), None)
        if anim is None:
            anim = max(anims, key=lambda a: Anim(g, blob, a).duration)
        A = Anim(g, blob, anim)
        t = max(0.0, min(args.time, A.duration))
        override = A.pose(t)

    world = world_matrices(g, override)
    tris = gather_triangles(g, blob, world)
    print(f"{args.glb.name}: {len(tris):,} triangles, "
          f"{'animated t=%.3fs' % t if anims else 'static bind pose'}")

    az, el = VIEWS[args.view]
    if args.turntable > 0:
        imgs = []
        for i in range(args.turntable):
            a = az + i * (360.0 / args.turntable)
            lab = f"{args.glb.stem}  az={a:.0f} el={el:.0f}  [{args.mode}]"
            imgs.append(render(tris, args.size, a, el, args.mode, lab))
        imgs = [i for i in imgs if i]
        if not imgs:
            print("nothing to draw")
            return 1
        strip = Image.new("RGB", (args.size * len(imgs), args.size), (16, 18, 22))
        for i, im in enumerate(imgs):
            strip.paste(im, (i * args.size, 0))
        strip.save(args.out)
    else:
        lab = f"{args.glb.stem}  az={az:.0f} el={el:.0f}  [{args.mode}]"
        im = render(tris, args.size, az, el, args.mode, lab)
        if im is None:
            print("nothing to draw")
            return 1
        im.save(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
