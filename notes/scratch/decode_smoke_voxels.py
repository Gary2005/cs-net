#!/usr/bin/env python3
"""Decode CS2 `m_VoxelFrameData` (smoke) and verify against real smoke geometry.

Format reverse-engineered by DrEAmSs59/CS2-insight-agent
(`backend/app/features/demo_analysis/smoke_voxel_decode.py`, PolyForm
Noncommercial 1.0.0).  Re-implemented here from the published *format
description* for independent verification against our own probe data.

Journal layout
--------------
    record : u16le seq | u16le payload_len | payload
    payload[0] = phase, payload[1] = frame_type
      type 3 KEYFRAME : u8 seed_count,
                        seed_count x 8B [x, y, z, density, pad x 4],
                        u16 mask_count,
                        mask_count x 10B [cell_morton u16, voxel_mask u64]
      type 2 DELTA    : u16 mask_count, mask_count x 10B replacement masks
      type 0 HEARTBEAT: no geometry change (payload == 00 00 00)

Grid
----
    32 x 32 x 32.  Both Morton levels interleave X, Y, Z.
    cell_index uses 3 bits/axis -> 8x8x8 = 512 cells.
    voxel_mask's 64 bits use 2 bits/axis -> 4x4x4 sub-voxels per cell.
    => 8^3 * 4^3 = 32^3 = 32768.

World
-----
    world = detonation + (grid - 15.5) * 12.0     (12 HU per voxel)
    Mask bits give the cloud *boundary*; the enclosed interior is found by
    flood-filling from the six outer faces.

Input: the JSON produced by  `DUMP_VOXELS=1 ./goprobe/probe <demo>`
       (one line ``VOXELJSON {"n":..,"det":"[x y z]","vals":[..]}``)
"""
from __future__ import annotations

import json
import sys
from collections import deque

VOXEL_GRID_DIM = 32
VOXEL_WORLD_SIZE = 12.0
VOXEL_GRID_CENTER = (VOXEL_GRID_DIM - 1) / 2.0  # 15.5
_ROW_MASK = (1 << VOXEL_GRID_DIM) - 1

KEYFRAME, DELTA, HEARTBEAT = 3, 2, 0


def split_journal(raw: bytes, size: int):
    """Split the property bytes into (seq, payload) records."""
    frames, off = [], 0
    while off + 4 <= size:
        seq = raw[off] | (raw[off + 1] << 8)
        length = raw[off + 2] | (raw[off + 3] << 8)
        end = off + 4 + length
        if end > size:
            raise ValueError(f"record@{off} len={length} overruns size={size}")
        frames.append((seq, raw[off + 4:end]))
        off = end
    return frames, off


def demorton3(index: int, bits: int):
    x = y = z = 0
    for b in range(bits):
        x |= ((index >> (3 * b)) & 1) << b
        y |= ((index >> (3 * b + 1)) & 1) << b
        z |= ((index >> (3 * b + 2)) & 1) << b
    return x, y, z


def apply_keyframe(payload: bytes):
    count = payload[2]
    off, seeds = 3, []
    for _ in range(count):
        x, y, z, density = payload[off], payload[off + 1], payload[off + 2], payload[off + 3]
        seeds.append((x, y, z, density))
        off += 8
    mask_count = payload[off] | (payload[off + 1] << 8)
    off += 2
    masks = []
    for _ in range(mask_count):
        cell = payload[off] | (payload[off + 1] << 8)
        m = int.from_bytes(payload[off + 2:off + 10], "little")
        masks.append((cell, m))
        off += 10
    return seeds, masks, off


def _spread_row(seed: int, empty: int) -> int:
    reached = seed & empty & _ROW_MASK
    while True:
        nxt = (reached | ((reached << 1) & _ROW_MASK) | (reached >> 1)) & empty
        if nxt == reached:
            return reached
        reached = nxt


def rebuild_grid(seeds, masks) -> list[list[int]]:
    """Return occupied[x][y] as a 32-bit Z row, with enclosed interior filled."""
    occ = [[0] * VOXEL_GRID_DIM for _ in range(VOXEL_GRID_DIM)]

    def mark(x, y, z):
        if 0 <= x < 32 and 0 <= y < 32 and 0 <= z < 32:
            occ[x][y] |= 1 << z

    for x, y, z, _ in seeds:
        mark(x, y, z)
    for cell, mask in masks:
        cx, cy, cz = demorton3(cell, 3)
        for bit in range(64):
            if (mask >> bit) & 1:
                sx, sy, sz = demorton3(bit, 2)
                mark(cx * 4 + sx, cy * 4 + sy, cz * 4 + sz)

    empty = [[_ROW_MASK ^ occ[x][y] for y in range(32)] for x in range(32)]
    outside = [[0] * 32 for _ in range(32)]
    pending, queued = deque(), set()

    def add(x, y, bits):
        old = outside[x][y]
        new = _spread_row(old | bits, empty[x][y])
        if new == old:
            return
        outside[x][y] = new
        if (x, y) not in queued:
            queued.add((x, y))
            pending.append((x, y))

    for y in range(32):
        add(0, y, empty[0][y])
        add(31, y, empty[31][y])
    for x in range(32):
        add(x, 0, empty[x][0])
        add(x, 31, empty[x][31])
        for y in range(32):
            add(x, y, (1 | (1 << 31)) & empty[x][y])
    while pending:
        x, y = pending.popleft()
        queued.discard((x, y))
        bits = outside[x][y]
        if x > 0:
            add(x - 1, y, bits)
        if x < 31:
            add(x + 1, y, bits)
        if y > 0:
            add(x, y - 1, bits)
        if y < 31:
            add(x, y + 1, bits)

    return [[occ[x][y] | (_ROW_MASK ^ outside[x][y]) for y in range(32)] for x in range(32)]


def to_world(x, y, z, origin):
    return tuple(origin[i] + (v - VOXEL_GRID_CENTER) * VOXEL_WORLD_SIZE
                 for i, v in enumerate((x, y, z)))


def main(path: str) -> int:
    line = open(path).read().strip()
    d = json.loads(line[len("VOXELJSON "):])
    raw = bytes(d["vals"])
    det = [float(v) for v in d["det"].strip("[]").split()]

    declared = int(sys.argv[2]) if len(sys.argv) > 2 else 802
    print(f"declared m_nVoxelFrameDataSize = {declared}")
    print(f"detonation = ({det[0]:.2f}, {det[1]:.2f}, {det[2]:.2f})\n")

    frames, consumed = split_journal(raw, declared)
    print(f"journal: {len(frames)} record(s), consumed {consumed}/{declared} bytes")
    for seq, payload in frames:
        kind = {KEYFRAME: "KEYFRAME", DELTA: "DELTA", HEARTBEAT: "HEARTBEAT"}.get(
            payload[1] if len(payload) > 1 else -1, "?")
        print(f"  seq={seq} phase={payload[0]} type={kind} payload_len={len(payload)}")
        if payload[1] != KEYFRAME:
            continue
        seeds, masks, used = apply_keyframe(payload)
        print(f"  seeds={len(seeds)} masks={len(masks)} consumed={used}/{len(payload)}")
        print(f"  seed coord range: "
              f"x{min(s[0] for s in seeds)}-{max(s[0] for s in seeds)} "
              f"y{min(s[1] for s in seeds)}-{max(s[1] for s in seeds)} "
              f"z{min(s[2] for s in seeds)}-{max(s[2] for s in seeds)}")

        occ = rebuild_grid(seeds, masks)
        pts = []
        for x in range(32):
            for y in range(32):
                bits = occ[x][y]
                while bits:
                    z = (bits & -bits).bit_length() - 1
                    pts.append(to_world(x, y, z, det))
                    bits &= bits - 1

        r3 = [sum((p[i] - det[i]) ** 2 for i in range(3)) ** 0.5 for p in pts]
        rh = [((p[0] - det[0]) ** 2 + (p[1] - det[1]) ** 2) ** 0.5 for p in pts]
        c = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
        zs = [p[2] for p in pts]
        print(f"\n  occupied voxels = {len(pts)}")
        print(f"  world centroid = ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"
              f"  offset from detonation = ({c[0]-det[0]:+.1f}, {c[1]-det[1]:+.1f}, {c[2]-det[2]:+.1f})")
        print(f"  horizontal radius: max={max(rh):.1f} HU")
        print(f"  3D radius:         max={max(r3):.1f} HU")
        print(f"  vertical extent:   {min(zs):.1f} .. {max(zs):.1f}  (height {max(zs)-min(zs):.1f} HU)")
        print("\n  expected: 3D radius ~149-166 HU; centroid above detonation; grounded lens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/voxel.json"))