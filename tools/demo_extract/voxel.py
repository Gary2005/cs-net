"""
Smoke voxel journals: reconstruction and decoding.

The stored form is the *raw* journal, so nothing here is needed to get the data
back — these helpers turn it into geometry.  Everything is derived, never
stored, which is why the file stays small.

Journal layout (reverse-engineered by DrEAmSs59/CS2-insight-agent, PolyForm
Noncommercial 1.0.0; re-implemented here from the format description and
verified against our own probe data)
--------------------------------------------------------------------------
    record  : u16le seq | u16le payload_len | payload
    payload[0] = phase, payload[1] = frame_type
      type 3 KEYFRAME : u8 seed_count,
                        seed_count x 8B [x, y, z, density, pad x 4],
                        u16 mask_count,
                        mask_count x 10B [cell_morton u16, voxel_mask u64]
      type 2 DELTA    : u16 mask_count, mask_count x 10B replacement masks
      type 0 HEARTBEAT: no geometry change

Grid
----
    32 x 32 x 32 = 32768 voxels.
    `cell_morton` interleaves X, Y, Z with 3 bits/axis -> one of 8x8x8 = 512
    cells.  `voxel_mask`'s 64 bits interleave 2 bits/axis -> 4x4x4 = 64
    sub-voxels inside that cell.

World
-----
    world = detonation + (grid - 15.5) * 12.0        (12 HU per voxel)

The masks describe the cloud *surface*, and they sample it sparsely: on real
demos only ~83% of the (x, y) columns hold one contiguous Z run, the average
column spans ~4 voxels, and no single cell ever has all 64 of its sub-voxels
set.  Consequence: the empty space inside the shell stays connected to the
outside, so the obvious "flood fill inwards from the six outer faces" recovers
**zero** interior voxels — 59 of 82 smokes in `test.dem` gained nothing at all.
That approach is not implemented here; it was tried and removed.

`solid_from_shell` therefore fills the **convex hull** of the sampled surface:
every grid cell inside `conv(shell)` becomes occupied.  It is parameter-free —
there is nothing to tune — but it is still a *rendering* choice, not a fact
about the format: the interior is simply not in the data.  See the warning on
`solid_from_shell`.

Nothing in this module writes to a `.csz`; the fill only ever affects what the
visualizer draws.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, Iterable, Iterator

import numpy as np

#: 凸包后端：`"qhull"`（scipy，C 实现）或 `"python"`（本文件的纯 stdlib 后备）。
#:
#: 两者算同一个凸包（`test_roundtrip.py` 的 `voxel / hull backend` 组逐格对拍），
#: 但满云（~1350 点）实测 **147 ms → 2.2 ms**。装没装 scipy 决定用哪个，而"结果对、
#: 只是慢 50 倍"是最难自己发现的那类问题，所以 `prepare` 开跑前会打一行
#: `[烟雾] 凸包后端 …`；库模块本身不往 stdout 写东西。
try:
    from scipy.spatial import ConvexHull
    from scipy.spatial import QhullError
except ImportError:  # pragma: no cover - 取决于环境
    ConvexHull = None
    QhullError = Exception
HULL_BACKEND = "qhull" if ConvexHull is not None else "python"

VOXEL_GRID_DIM = 32
VOXEL_WORLD_SIZE = 12.0
VOXEL_GRID_CENTER = (VOXEL_GRID_DIM - 1) / 2.0  # 15.5

#: Tolerance for "is this grid cell inside the convex hull?".
#:
#: The hull's faces pass exactly through grid points, so cells can sit exactly
#: on the boundary.  Measured over 82 real smokes, the fill is *identical* for
#: any value from 1e-12 to 1e-3, so this only needs to absorb floating-point
#: noise between the Python and JavaScript hull implementations.  Do not raise
#: it past ~1e-3 or the fill starts growing.
HULL_EPS = 1e-6


KEYFRAME, DELTA, HEARTBEAT = 3, 2, 0
FRAME_TYPE_NAMES = {KEYFRAME: "keyframe", DELTA: "delta", HEARTBEAT: "heartbeat"}

MODE_APPEND, MODE_FULL = "append", "full"


# ── journal parsing (raw bytes -> records) ───────────────────────────────────


def split_journal(raw: bytes, size: int | None = None) -> tuple[list[tuple[int, bytes]], int]:
    """Split journal bytes into ``[(seq, payload), ...]`` and bytes consumed."""
    if size is None:
        size = len(raw)
    size = min(size, len(raw))
    frames: list[tuple[int, bytes]] = []
    off = 0
    while off + 4 <= size:
        seq = raw[off] | (raw[off + 1] << 8)
        length = raw[off + 2] | (raw[off + 3] << 8)
        end = off + 4 + length
        if end > size:
            break
        frames.append((seq, raw[off + 4:end]))
        off = end
    return frames, off


def frame_type(payload: bytes) -> int:
    return payload[1] if len(payload) > 1 else -1


def apply_keyframe(payload: bytes) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]], int]:
    """Parse a KEYFRAME payload -> (seeds, masks, bytes consumed)."""
    if len(payload) < 3:
        return [], [], len(payload)
    count = payload[2]
    off = 3
    seeds: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        if off + 8 > len(payload):
            break
        seeds.append((payload[off], payload[off + 1], payload[off + 2], payload[off + 3]))
        off += 8
    if off + 2 > len(payload):
        return seeds, [], off
    mask_count = payload[off] | (payload[off + 1] << 8)
    off += 2
    masks: list[tuple[int, int]] = []
    for _ in range(mask_count):
        if off + 10 > len(payload):
            break
        cell = payload[off] | (payload[off + 1] << 8)
        m = int.from_bytes(payload[off + 2:off + 10], "little")
        masks.append((cell, m))
        off += 10
    return seeds, masks, off


def apply_delta(payload: bytes) -> list[tuple[int, int]]:
    """Parse a DELTA payload -> replacement (cell_morton, voxel_mask) entries."""
    if len(payload) < 3:
        return []
    mask_count = payload[2] | (payload[3] << 8) if len(payload) >= 4 else 0
    off = 4
    masks: list[tuple[int, int]] = []
    for _ in range(mask_count):
        if off + 10 > len(payload):
            break
        cell = payload[off] | (payload[off + 1] << 8)
        m = int.from_bytes(payload[off + 2:off + 10], "little")
        masks.append((cell, m))
        off += 10
    return masks


# ── Morton ──────────────────────────────────────────────────────────────────


def demorton3(index: int, bits: int) -> tuple[int, int, int]:
    x = y = z = 0
    for b in range(bits):
        x |= ((index >> (3 * b)) & 1) << b
        y |= ((index >> (3 * b + 1)) & 1) << b
        z |= ((index >> (3 * b + 2)) & 1) << b
    return x, y, z


def morton3(x: int, y: int, z: int, bits: int) -> int:
    out = 0
    for b in range(bits):
        out |= ((x >> b) & 1) << (3 * b)
        out |= ((y >> b) & 1) << (3 * b + 1)
        out |= ((z >> b) & 1) << (3 * b + 2)
    return out


# ── geometry ─────────────────────────────────────────────────────────────────


def boundary_voxels(journal: bytes, size: int | None = None) -> set[tuple[int, int, int]]:
    """Voxels explicitly named by the journal's masks (the cloud boundary)."""
    frames, _ = split_journal(journal, size)
    out: set[tuple[int, int, int]] = set()
    for _, payload in frames:
        ft = frame_type(payload)
        if ft == KEYFRAME:
            seeds, masks, _ = apply_keyframe(payload)
            for x, y, z, _d in seeds:
                out.add((x, y, z))
            for cell, mask in masks:
                _apply_mask(out, cell, mask)
        elif ft == DELTA:
            for cell, mask in apply_delta(payload):
                _apply_mask(out, cell, mask)
    return out


def _apply_mask(out: set[tuple[int, int, int]], cell: int, mask: int) -> None:
    if mask == 0:
        return
    cx, cy, cz = demorton3(cell, 3)
    while mask:
        bit = (mask & -mask).bit_length() - 1
        sx, sy, sz = demorton3(bit, 2)
        out.add((cx * 4 + sx, cy * 4 + sy, cz * 4 + sz))
        mask &= mask - 1


def solid_rows(journal: bytes, size: int | None = None) -> list[list[int]]:
    """Per-(x,y) 32-bit Z occupancy rows of the *solid* cloud.

    See `solid_from_shell`; this is the row representation of the same thing.
    """
    rows = [[0] * VOXEL_GRID_DIM for _ in range(VOXEL_GRID_DIM)]
    for x, y, z in solid_from_shell(boundary_voxels(journal, size)):
        rows[x][y] |= 1 << z
    return rows


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _dist2(a, b):
    d = _sub(a, b)
    return _dot(d, d)


def _line_dist2(p, a, b):
    """Squared distance of `p` from the line ab, scaled by |ab|^2."""
    c = _cross(_sub(b, a), _sub(p, a))
    return _dot(c, c)


def convex_hull_halfspaces(
    points: Iterable[tuple[float, float, float]],
) -> list[tuple[float, float, float, float]] | None:
    """Half-spaces of ``conv(points)`` as ``(nx, ny, nz, c)``.

    The convention is ``n.p - c <= 0`` **inside**, matching three.js's ``Face``
    (``normal`` / ``constant``) so the JavaScript port in
    ``visualizer/static/js/smoke-voxels.js`` produces the same hull.

    Returns ``None`` when the points are fewer than 4 or all coplanar/collinear.

    实现走 Qhull（`scipy.spatial.ConvexHull`，C 代码）；没装 scipy 时回落到本文件
    里的纯 Python beneath-beyond（`_halfspaces_python`）。两者算同一个凸包，只是一
    个是 C 一个是一行一行 Python —— 实测 1800 个点 **~28 ms → ~1 ms**，而逐帧遮挡
    每颗烟每个 journal 记录都要重建一次，这是那条路径上最大的一块。当前用的是哪个
    后端看模块常量 `HULL_BACKEND`。
    """
    pts = list(dict.fromkeys(tuple(p) for p in points))
    if len(pts) < 4:
        return None
    if ConvexHull is not None:
        return _halfspaces_qhull(pts)
    return _halfspaces_python(pts)


def _halfspaces_qhull(pts: list[tuple[float, float, float]]):
    """Qhull 版。``equations`` 一行是 ``[n, offset]``，满足 ``n·p + offset <= 0``
    为内部，所以换算成本文件的约定时要取 **``c = -offset``**（漏掉负号会把凸包内外
    颠倒，填出"云外面整个世界"或直接空）。"""
    arr = np.asarray(pts, dtype=np.float64)
    try:
        hull = ConvexHull(arr)
    except QhullError:
        return None                                   # 共面 / 共线 / 重复点
    # Qhull 会把共面的面片三角化，同一个平面可能出现多次；按平面去重（保留原始浮点，
    # 只用四舍五入后的值当键），后面的逐列 min/max 能少算一半。
    seen: dict[tuple[float, float, float, float], tuple[float, float, float, float]] = {}
    for nx, ny, nz, offset in hull.equations:
        key = (round(float(nx), 12), round(float(ny), 12),
               round(float(nz), 12), round(float(offset), 12))
        if key not in seen:
            seen[key] = (float(nx), float(ny), float(nz), float(-offset))
    return list(seen.values())


def _halfspaces_python(
    pts: list[tuple[float, float, float]],
) -> list[tuple[float, float, float, float]] | None:
    """纯 stdlib 的增量（beneath-beyond）构造：Qhull 不可用时的后备。

    和 Qhull 版算的是同一个凸包（`test_roundtrip.py` 里有两者逐格对拍）。
    """

    # ── initial tetrahedron ────────────────────────────────────────────────
    i0 = 0
    i1 = max(range(len(pts)), key=lambda i: _dist2(pts[i], pts[i0]))
    if _dist2(pts[i1], pts[i0]) <= HULL_EPS:
        return None
    i2 = max(range(len(pts)), key=lambda i: _line_dist2(pts[i], pts[i0], pts[i1]))
    if _line_dist2(pts[i2], pts[i0], pts[i1]) <= HULL_EPS:
        return None
    n012 = _cross(_sub(pts[i1], pts[i0]), _sub(pts[i2], pts[i0]))
    i3 = max(range(len(pts)), key=lambda i: abs(_dot(n012, _sub(pts[i], pts[i0]))))
    if abs(_dot(n012, _sub(pts[i3], pts[i0]))) <= HULL_EPS:
        return None

    # The tetrahedron's centroid stays strictly inside the hull forever, so it
    # is a reliable "which way is out" reference for winding every new face.
    centre = tuple(sum(pts[i][k] for i in (i0, i1, i2, i3)) / 4.0 for k in range(3))

    def oriented(a: int, b: int, c: int) -> tuple[int, int, int]:
        n = _cross(_sub(pts[b], pts[a]), _sub(pts[c], pts[a]))
        if _dot(n, _sub(pts[a], centre)) < 0:
            b, c = c, b
        return (a, b, c)

    faces = [
        oriented(i0, i1, i2),
        oriented(i0, i1, i3),
        oriented(i0, i2, i3),
        oriented(i1, i2, i3),
    ]
    on_hull = {i0, i1, i2, i3}

    for ip in range(len(pts)):
        if ip in on_hull:
            continue
        p = pts[ip]
        visible = [
            fi
            for fi, (a, b, c) in enumerate(faces)
            if _dot(_cross(_sub(pts[b], pts[a]), _sub(pts[c], pts[a])), _sub(p, pts[a]))
            > HULL_EPS
        ]
        if not visible:
            continue
        on_hull.add(ip)

        # Horizon = edges used by exactly one visible face (interior edges are
        # shared by two of them and cancel out).
        edges: dict[tuple[int, int], int] = {}
        for fi in visible:
            a, b, c = faces[fi]
            for e in ((a, b), (b, c), (c, a)):
                rev = (e[1], e[0])
                if rev in edges:
                    edges[rev] += 1
                else:
                    edges[e] = edges.get(e, 0) + 1
        horizon = [e for e, count in edges.items() if count == 1]

        for fi in sorted(visible, reverse=True):
            faces.pop(fi)
        for a, b in horizon:
            faces.append(oriented(a, b, ip))

    out = []
    for a, b, c in faces:
        n = _cross(_sub(pts[b], pts[a]), _sub(pts[c], pts[a]))
        length = _dot(n, n) ** 0.5
        if length <= HULL_EPS:
            continue
        n = (n[0] / length, n[1] / length, n[2] / length)
        out.append((n[0], n[1], n[2], _dot(n, pts[a])))
    return out or None


def convex_hull_fill(
    cells: Iterable[tuple[int, int, int]], size: int | None = None
) -> set[tuple[int, int, int]]:
    """Every grid cell inside ``conv(cells)``.

    For a convex body a vertical line crosses it in exactly one interval, so
    this is a per-column fill like `solid_from_shell` used to do — but the
    interval comes from the hull's lower/upper surface instead of from the
    lowest/highest *sampled* voxel.  That is what makes it parameter-free: the
    hull interpolates across the gaps in the sparse sampling by itself.

    The near-horizontal half-spaces (``|nz| <= eps``) carry no Z information;
    they are what bounds the hull's shadow on the (x, y) plane, so they are
    checked separately.  Skipping that check makes the fill leak outside the
    hull along the whole grid.

    返回 `set` 只是为了兼容；热路径要用 `convex_hull_mask`，省掉 `set` 那趟往返。
    """
    shell = set(cells)
    dim = size or VOXEL_GRID_DIM
    up, down, flat = _split_halfspaces(convex_hull_halfspaces(shell))
    if up is None:
        return shell
    mask = _fill_from_halfspaces(up, down, flat, dim)
    return {(int(x), int(y), int(z)) for x, y, z in np.argwhere(mask)}


def convex_hull_mask(
    cells: Iterable[tuple[int, int, int]], size: int | None = None
) -> np.ndarray:
    """``(dim, dim, dim)`` bool：``conv(cells)`` 内的格子。

    和 `convex_hull_fill` **逐格相同**（`test_roundtrip.py` 里有对拍），但整张 32×32
    一次算完 —— 原来那版是 1024 次 Python 循环、每列再跑一遍 min/max 生成器，实测
    几 ms，现在是亚毫秒。

    只返回栅格、不建 `set`：热路径（`solid_mask` → 烟的 32³ 栅格）本来就要一个布尔
    数组，绕一趟 `set` 反而更贵 —— 5459 个 tuple 约 1～2 ms，比填充本身还大。

    退化（点太少 / 共面 / 没有上下面）时返回**点云自身的栅格**，和
    `convex_hull_fill` 返回 `shell` 是同一个意思。越界坐标裁到 `0..dim-1`
    （journal 的 keyframe seed 可能是 `dim`）。
    """
    shell = set(cells)
    dim = size or VOXEL_GRID_DIM
    up, down, flat = _split_halfspaces(convex_hull_halfspaces(shell))
    if up is None:
        out = np.zeros((dim, dim, dim), dtype=bool)
        pts = np.asarray(sorted(shell), dtype=np.int64).reshape(-1, 3)
        if pts.size:
            np.clip(pts, 0, dim - 1, out=pts)
            out[pts[:, 0], pts[:, 1], pts[:, 2]] = True
        return out
    return _fill_from_halfspaces(up, down, flat, dim)


def _split_halfspaces(halfspaces):
    """`(nx, ny, nz, c)` 列表 → `(up, down, flat)` 三个 ndarray；拿不出上/下表面时
    第一个返回 `None`（调用方退化成"只有壳"）。"""
    if not halfspaces:
        return None, None, None
    arr = np.asarray(halfspaces, dtype=np.float64).reshape(-1, 4)
    up = arr[arr[:, 2] > HULL_EPS]
    down = arr[arr[:, 2] < -HULL_EPS]
    flat = arr[np.abs(arr[:, 2]) <= HULL_EPS]
    if up.size == 0 or down.size == 0:
        return None, None, None
    return up, down, flat


def _fill_from_halfspaces(up: np.ndarray, down: np.ndarray, flat: np.ndarray,
                          dim: int) -> np.ndarray:
    """半空间 → `(dim, dim, dim)` 布尔栅格：每列取 `[lo, hi]` 填。"""
    # 网格坐标先铺成 (dim, dim)，再用半空间 (m, 1, 1) 广播成 (m, dim, dim)。
    xy = np.arange(dim, dtype=np.float64)
    gx, gy = np.meshgrid(xy, xy, indexing="ij")

    keep = np.ones((dim, dim), dtype=bool)
    if flat.size:
        # |nz| ≈ 0 的竖直面只约束 (x, y)：整列都在影子外就直接扔
        outside = np.any(
            flat[:, 0, None, None] * gx + flat[:, 1, None, None] * gy
            - flat[:, 3, None, None] > HULL_EPS,
            axis=0,
        )
        keep &= ~outside

    hi = np.min(
        (up[:, 3, None, None] - up[:, 0, None, None] * gx - up[:, 1, None, None] * gy)
        / up[:, 2, None, None],
        axis=0,
    )
    lo = np.max(
        (down[:, 3, None, None] - down[:, 0, None, None] * gx - down[:, 1, None, None] * gy)
        / down[:, 2, None, None],
        axis=0,
    )
    keep &= hi >= lo - HULL_EPS

    # 每列填 z ∈ [ceil(lo), floor(hi)] ∩ [0, dim-1]；keep 为假的列 lo/hi 无意义，
    # 先夹到栅格内再取整，免得 inf / 越界值把 int64 撑坏。
    zlo = np.clip(np.ceil(lo - HULL_EPS), 0, dim - 1).astype(np.int64)
    zhi = np.clip(np.floor(hi + HULL_EPS), 0, dim - 1).astype(np.int64)
    z = np.arange(dim)[None, None, :]
    return keep[..., None] & (z >= zlo[..., None]) & (z <= zhi[..., None])


def solid_from_shell(shell: Iterable[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    """Reconstruct a cloud-shaped volume from the stored surface shell.

    Fills the convex hull of the sampled surface (`convex_hull_fill`).  The
    journal only stores the cloud's *boundary*, so the interior is invented
    either way; a convex hull is at least the simplest rule that needs no
    tuned constant.

    .. warning::
       **Rendering heuristic, not format.**  The interior is *not* in the data
       and the upstream decoder never tries to recover it (it takes min/max Z
       per column, for 2D radar contours only).  A convex hull also removes
       every concavity, which the real cloud may well have.

       反过来"用壳做遮挡才准"也是错的，别被 `boundary_voxels` 这个名字骗了：
       掩码是**稀疏采样**，云内部本身就有成片的空洞（一颗满云：点名 1787 格、
       凸包 5459 格，点名格只有 6.5% 的 6 邻域全占，p50 邻居数 4/6），DDA 会从洞里
       穿过去 —— 实测按壳判遮挡会把 **45.8%** 的"被烟挡"记成"通"。所以烟体积就用
       这里的凸包（和渲染/可视化画的同一个），别拿 `boundary_voxels` 做遮挡判定。
       `tools/demo_extract/README.md` §6 和 `training_data/obstruction.py::smoke_volumes`
       里有完整测量。
    """
    return convex_hull_fill(shell)


def solid_voxels(
    journal: bytes, size: int | None = None
) -> list[tuple[int, int, int]]:
    return sorted(solid_from_shell(boundary_voxels(journal, size)))


def solid_mask(journal: bytes, size: int | None = None) -> np.ndarray:
    """`solid_voxels()` 的栅格版：`(dim, dim, dim)` bool，索引顺序 `[x, y, z]`。

    给热路径用（`recipes.smoke_grid_at(solid=True)`）：那里本来就要一个布尔栅格，
    走 `solid_voxels()` 会先 `sorted(set)` 出 5000 多个 tuple、再逐个写回数组，
    实测比填充本身还贵。越界坐标裁到 `0..dim-1`。

    等价性：`set(map(tuple, np.argwhere(solid_mask(j))))` 和
    `set(solid_voxels(j))` 在坐标都在栅格内时完全相同（`test_roundtrip.py` 有对拍）。
    """
    return convex_hull_mask(boundary_voxels(journal, size), size)


def grid_to_world(
    x: int, y: int, z: int, detonation: Iterable[float]
) -> tuple[float, float, float]:
    """Grid voxel index -> world position in Hammer units."""
    ox, oy, oz = detonation
    return (
        ox + (x - VOXEL_GRID_CENTER) * VOXEL_WORLD_SIZE,
        oy + (y - VOXEL_GRID_CENTER) * VOXEL_WORLD_SIZE,
        oz + (z - VOXEL_GRID_CENTER) * VOXEL_WORLD_SIZE,
    )


def world_voxels(
    journal: bytes, detonation: Iterable[float], size: int | None = None, solid: bool = True
) -> list[tuple[float, float, float]]:
    """Cloud as world-space points, ready to feed a renderer or a voxel CNN."""
    cells = solid_voxels(journal, size) if solid else sorted(boundary_voxels(journal, size))
    return [grid_to_world(x, y, z, detonation) for x, y, z in cells]


# ── per-tick reconstruction from the stored change records ───────────────────


def _record_index(inst: dict, tick: int) -> int:
    ticks = [r["tick"] for r in inst["changes"]]
    i = bisect_right(ticks, tick) - 1
    return i


def journal_at(inst: dict, tick: int, cache: dict | None = None) -> bytes | None:
    """The smoke's journal as of `tick`, or None when the smoke is not alive.

    `cache` is an optional dict used to memoise the fold; pass one when walking
    many ticks over the same instance.
    """
    if tick < inst["first_tick"] or tick > inst["last_tick"]:
        return None
    recs = inst["changes"]
    if not recs:
        return b""
    i = _record_index(inst, tick)
    if i < 0:
        return None
    key = ("fold", inst["entity_id"], inst["first_tick"], i)
    if cache is not None and key in cache:
        return cache[key]
    last = cache.get(("last", inst["entity_id"], inst["first_tick"])) if cache else None
    if last is not None and last[0] <= i:
        j, buf = last
    else:
        j, buf = -1, b""
    while j < i:
        j += 1
        r = recs[j]
        buf = buf + r["data"] if r["mode"] == MODE_APPEND else bytes(r["data"])
    if cache is not None:
        cache[key] = buf
        cache[("last", inst["entity_id"], inst["first_tick"])] = (j, buf)
    return buf


def state_at(inst: dict, tick: int) -> tuple[int, int] | None:
    """``(m_nVoxelFrameDataSize, m_nVoxelUpdate)`` at `tick`, or None if absent."""
    if tick < inst["first_tick"] or tick > inst["last_tick"]:
        return None
    recs = inst["changes"]
    if not recs:
        return (0, 0)
    i = _record_index(inst, tick)
    if i < 0:
        return None
    r = recs[i]
    return (r["size"], r["update"])


def change_ticks(inst: dict) -> list[int]:
    return [r["tick"] for r in inst["changes"]]


def iter_journals(inst: dict) -> Iterator[tuple[int, bytes]]:
    """Yield ``(tick, journal)`` for every stored change, folding incrementally."""
    buf = b""
    for r in inst["changes"]:
        buf = buf + r["data"] if r["mode"] == MODE_APPEND else bytes(r["data"])
        yield r["tick"], buf


def journal_stats(journal: bytes, size: int | None = None) -> dict[str, Any]:
    """Diagnostics: record mix and how much of the journal parses."""
    frames, consumed = split_journal(journal, size)
    counts = {KEYFRAME: 0, DELTA: 0, HEARTBEAT: 0}
    other = 0
    for _, payload in frames:
        ft = frame_type(payload)
        if ft in counts:
            counts[ft] += 1
        else:
            other += 1
    return {
        "records": len(frames),
        "keyframe": counts[KEYFRAME],
        "delta": counts[DELTA],
        "heartbeat": counts[HEARTBEAT],
        "other": other,
        "consumed": consumed,
        "declared": size if size is not None else len(journal),
    }