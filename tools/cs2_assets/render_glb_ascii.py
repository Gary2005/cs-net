#!/usr/bin/env python3
"""Software-rasterise a skinned GLB to prove it actually renders and deforms.

The structural and skinning validators prove references resolve and weights are
sane, but neither proves the model *draws*. This does: it evaluates a clip at a
chosen time, applies linear-blend skinning on the CPU, and renders a depth-shaded
ASCII image from a front and a side view.

Run the same GLB at two different times -- if the silhouettes differ, animation
evaluation and skinning both work end to end.

No GPU, no browser, no third-party dependency: pure Python + math.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

COMPONENT_FMT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


# ---------------------------------------------------------------- glTF loading

def load_glb(path: Path):
    d = path.read_bytes()
    off, ch = 12, {}
    while off < len(d):
        clen, ctype = struct.unpack_from("<II", d, off)
        off += 8
        ch[ctype] = d[off:off + clen]
        off += clen
    return json.loads(ch[0x4E4F534A].decode("utf-8")), ch.get(0x004E4942, b"")


def read_accessor(g, blob, idx):
    a = g["accessors"][idx]
    n = TYPE_COUNT[a["type"]]
    fmt = COMPONENT_FMT[a["componentType"]]
    if "bufferView" not in a:
        return [(0.0,) * n] * a["count"]
    bv = g["bufferViews"][a["bufferView"]]
    esz = COMPONENT_SIZE[a["componentType"]] * n
    stride = bv.get("byteStride") or esz
    base = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    return [struct.unpack_from("<" + fmt * n, blob, base + i * stride) for i in range(a["count"])]


# ------------------------------------------------------------------- 4x4 maths

def m_identity():
    return [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]


def m_mul(a, b):
    return [sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4)]


def m_from_trs(t, q, s):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    sx, sy, sz = s
    return [
        (1 - 2 * (yy + zz)) * sx, (2 * (xy - wz)) * sx, (2 * (xz + wy)) * sx, 0,
        (2 * (xy + wz)) * sy, (1 - 2 * (xx + zz)) * sy, (2 * (yz - wx)) * sy, 0,
        (2 * (xz - wy)) * sz, (2 * (yz + wx)) * sz, (1 - 2 * (xx + yy)) * sz, 0,
        t[0], t[1], t[2], 1.0,
    ]


def m_apply(m, p):
    x, y, z = p
    return (
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    )


# --------------------------------------------------------------- animation eval

class Anim:
    """Samples one glTF animation at a given time, per node."""

    def __init__(self, g, blob, anim):
        self.g, self.blob = g, blob
        self.channels = []
        for c in anim.get("channels", []):
            s = anim["samplers"][c["sampler"]]
            times = [v[0] if isinstance(v, tuple) else v
                     for v in read_accessor(g, blob, s["input"])]
            vals = read_accessor(g, blob, s["output"])
            self.channels.append((c["target"]["node"], c["target"]["path"], times, vals))
        self.duration = max((t[-1] for _, _, t, _ in self.channels if t), default=0.0)

    @staticmethod
    def _sample(times, vals, t, is_quat):
        if not times:
            return None
        if t <= times[0]:
            return vals[0]
        if t >= times[-1]:
            return vals[-1]
        lo, hi = 0, len(times) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if times[mid] <= t:
                lo = mid
            else:
                hi = mid
        span = times[hi] - times[lo]
        f = 0.0 if span <= 0 else (t - times[lo]) / span
        a, b = vals[lo], vals[hi]
        if is_quat:
            # shortest-arc nlerp
            dot = sum(x * y for x, y in zip(a, b))
            if dot < 0:
                b = tuple(-y for y in b)
            out = tuple(x + (y - x) * f for x, y in zip(a, b))
            n = math.sqrt(sum(v * v for v in out)) or 1.0
            return tuple(v / n for v in out)
        return tuple(x + (y - x) * f for x, y in zip(a, b))

    def pose(self, t):
        """Return {node_index: (translation, rotation, scale)} overrides."""
        out: dict[int, dict] = {}
        for node, path, times, vals in self.channels:
            is_quat = path == "rotation"
            v = self._sample(times, vals, t, is_quat)
            if v is None:
                continue
            out.setdefault(node, {})[path] = v
        return out


def node_local(g, idx, override):
    n = g["nodes"][idx]
    if "matrix" in n and not override:
        return list(n["matrix"])
    t = override.get("translation") or tuple(n.get("translation", (0.0, 0.0, 0.0)))
    q = override.get("rotation") or tuple(n.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    s = override.get("scale") or tuple(n.get("scale", (1.0, 1.0, 1.0)))
    return m_from_trs(t, q, s)


def world_matrices(g, override_by_node):
    n = len(g["nodes"])
    parent = [-1] * n
    for i, node in enumerate(g["nodes"]):
        for c in node.get("children", []):
            parent[c] = i
    order, stack = [], [i for i in range(n) if parent[i] == -1]
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(g["nodes"][i].get("children", []))
    world = [None] * n
    for i in order:
        local = node_local(g, i, override_by_node.get(i, {}))
        world[i] = local if parent[i] == -1 else m_mul(world[parent[i]], local)
    return world


# ------------------------------------------------------------------- rasteriser

def render(positions, cols, rows, title):
    if not positions:
        print(f"  {title}: no geometry")
        return
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    spanx = (maxx - minx) or 1.0
    spany = (maxy - miny) or 1.0
    spanz = (maxz - minz) or 1.0

    zbuf = [[None] * cols for _ in range(rows)]
    for x, y, z in positions:
        # view from -Z, up = +Y  (glTF is Y-up)
        px = int((x - minx) / spanx * (cols - 1))
        py = int((maxy - y) / spany * (rows - 1))
        depth = (z - minz) / spanz
        if 0 <= px < cols and 0 <= py < rows:
            if zbuf[py][px] is None or depth < zbuf[py][px]:
                zbuf[py][px] = depth

    chars = "@%#*+=-:. "
    print(f"  {title}  bbox {spanx:.1f} x {spany:.1f} x {spanz:.1f}")
    for r in range(rows):
        line = ""
        for c in range(cols):
            d = zbuf[r][c]
            line += " " if d is None else chars[min(9, int(d * 9.999))]
        print("    " + line)


def gather_vertices(g, blob, world):
    """Collect world-space vertices, applying linear blend skinning when present.

    A mesh with JOINTS_0/WEIGHTS_0 but no skin (weapon exports) falls through to
    the rigid path on purpose: there is no inverseBindMatrices to apply, and the
    bind pose is the assembled weapon.
    """
    out = []
    for mi, mesh in enumerate(g["meshes"]):
        node = next((n for n in g["nodes"] if n.get("mesh") == mi), None)
        if node is None:
            continue
        skin = g["skins"][node["skin"]] if "skin" in node else None
        node_world = world[g["nodes"].index(node)]
        for prim in mesh["primitives"]:
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = read_accessor(g, blob, attrs["POSITION"])
            if not skin or "JOINTS_0" not in attrs:
                for p in pos:
                    out.append(m_apply(node_world, p))
                continue
            joints = read_accessor(g, blob, attrs["JOINTS_0"])
            weights = read_accessor(g, blob, attrs["WEIGHTS_0"])
            ibm = read_accessor(g, blob, skin["inverseBindMatrices"])
            jn = skin["joints"]
            # jointMatrix = world[joint] * inverseBind
            jm = [m_mul(world[jn[k]], ibm[k]) for k in range(len(jn))]
            for p, jv, wv in zip(pos, joints, weights):
                acc = [0.0, 0.0, 0.0]
                for k in range(4):
                    w = wv[k]
                    if w == 0:
                        continue
                    q = m_apply(jm[jv[k]], p)
                    acc[0] += q[0] * w
                    acc[1] += q[1] * w
                    acc[2] += q[2] * w
                out.append(tuple(acc))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", type=Path)
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--clip", default=None, help="substring of the animation name")
    ap.add_argument("--cols", type=int, default=54)
    ap.add_argument("--rows", type=int, default=30)
    args = ap.parse_args(argv[1:])

    g, blob = load_glb(args.glb)
    anims = g.get("animations", [])

    # Weapon models carry no animation tracks -- they are static props. Render
    # them in bind pose instead of refusing, otherwise the only way to eyeball a
    # weapon's geometry is to open it in a browser.
    if not anims:
        print(f"{args.glb.name}: no animations -- rendering static bind pose")
        world = world_matrices(g, {})
        out = gather_vertices(g, blob, world)
        print(f"vertices  : {len(out):,}")
        render(out, args.cols, args.rows, "static bind pose")
        return 0

    anim = None
    if args.clip:
        for a in anims:
            if args.clip in a.get("name", ""):
                anim = a
                break
        if anim is None:
            print(f"{args.glb.name}: no clip matching {args.clip!r}")
            print(f"  {len(anims)} clip(s) available, e.g.:")
            for a in anims[:15]:
                print(f"    {a.get('name')}")
            return 1
    else:
        # prefer the longest-duration clip so motion is obvious
        anim = max(anims, key=lambda a: Anim(g, blob, a).duration)

    A = Anim(g, blob, anim)
    t = max(0.0, min(args.time, A.duration))
    print(f"clip      : {anim.get('name')}")
    print(f"duration  : {A.duration:.3f}s   sampling t={t:.3f}s")

    override = A.pose(t)
    world = world_matrices(g, override)
    out = gather_vertices(g, blob, world)

    print(f"vertices  : {len(out):,}")
    render(out, args.cols, args.rows, f"front view @ t={t:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
