"""视角渲染图：从某个玩家的眼睛往 **192 × 256** 个方向发一条射线，每条射线记 5 个通道。

    (192, 256, 5) float32 —— 5 个通道依次是 depth / smoke / enemy / teammate / target

    通道        含义（全部是**游戏单位 HU**，离眼睛的直线距离）
    ----------  --------------------------------------------------------------
    0 depth     到**地图 mesh** 的距离（最近的那个三角形）
    1 smoke     到**烟雾体素**的距离（进入第一颗被占据的体素）
    2 enemy     到**敌人**包围盒的距离（**含**第 4 个通道的那个目标，取最近的）
    3 teammate  到**队友**包围盒的距离（不含自己）
    4 target    到**这一次的目标**包围盒的距离（攻击方看受害者、受害者看攻击方）

    任何一条射线没打到东西就是哨兵 `VIEW_FAR = 1e9`（"无限远"）。

## 为什么和 `visualizer/` 里的数字对齐

角度范围和包围盒**刻意和浏览器那套一致**，这样离线和在线看到的是同一个世界：

* 视锥：水平 **90°**（`visibility.js::DEFAULT_FOV_DEG`，就是 `m_iFOV` 的默认值），
  宽高比 **4:3**（`192:256` 的行列比），于是垂直
  `2·atan(tan(45°)/(4/3)) = 73.7398°`。
  和 `visibility.js` 的默认 16:9（垂直 58.7°）不同——**这里按你给的 256:192 像素网格走 4:3**，
  水平角不动。
* 采样：**角度均匀**，不是透视投影。第 k 列的 yaw 偏移 =
  `(256/2 - 0.5 - k) · (90°/256)`，第 j 行的 pitch 偏移同理。
  也就是说这是一张"等角距"的 yaw×pitch 图（视锥内），不是针孔相机的那张画——
  你要求的正是"yaw 分 256 格、pitch 分 192 格"。

  列的方向 = 屏幕的左右（**第 0 列在最左边**）。因为 Source 的 yaw 是从上方看
  **逆时针**增大的（yaw 0 看 +X，yaw +90 看 +Y），玩家的**右手**方向是
  `f × up = (sin y, −cos y, 0)`（和 `visibility.js::viewBasis()` 的 r 同一个式子），
  所以"视线往左 10°"对应 yaw **+10°**、对应屏幕**左边**的列。写成式子是
  `delta_yaw = −(k + 0.5 − 256/2) · (90°/256)`——负号不能省：省掉就变成
  左右镜像的图（照镜子的第一人称），和真实画面、浏览器 POV 相反。
  俯仰那一条没有这个问题：pitch 正 = 向下看，所以第 j 行 = 屏幕第 j 行（往下）。
* 包围盒：**世界轴对齐**的方盒，半宽 **16 HU**、高 **72（站）/ 54（蹲）**，
  `z` 是**脚底**（`.csz` 的 x/y/z 就是脚底）。高度看 `crouching`（布尔阶跃），
  和 `visibility.js::hullHeightFor()` 一致；不用 `crouch` 那个连续量——那是**相机**眼高的斜坡。
* 眼睛：`z + eye_z`。`.csz` 里**没有** `eye_z` 这个字段（`round_tensors()` 的
  `eye_z` 是算出来的），所以这里直接调 `recipes.eye_height()` =
  `64.0938 - 18·crouch`，别在这儿抄常数。
* 烟雾：journal → 32³ 体素 → **实心**（凸包填充），和浏览器默认的 `solid=True` 一致；
  格心世界坐标 = `detonation + (grid - 15.5) · 12`。

## 实现

射线求交是 **numba** 编译的：地图的三角形先建成 BVH（`map-raycast.js` 同一套
"质心最长轴中点切"，叶子 8 个三角形），再逐射线遍历。BVH 只在**换地图**时建一次
（`load_map()` 有 `lru_cache`）：mirage 的 122 万三角形 = 解析 1.0 s + 建树 2.6 s，
45.6 万个节点、约 23 MB。

实测（mirage，32 线程）：一张 192×256 的图 **20 ms**（49,152 条射线，绝大部分花在
地图 BVH 上），每张图另外约 1 ms 的组装成本。四份 csz 全量造 532 个样本（1064 张图）
一共 169 s，其中渲染只占 ~21 s（剩下是 csz 加载、烟雾凸包和 npz 压缩）。

## 语义上要知道的三件事

1. **enemy 包含 target。** 对手就是敌方的一员，所以逐像素恒有 `enemy <= target`：
   `target` 不是"另一类东西"，它只是把指定的那个对手单独再给一份，好让模型不必从
   一堆敌人里猜"哪个才是目标"。
2. 三个包围盒通道是**纯几何**，不考虑墙、烟、别的人挡在前面（那是通道 0/1 的事）。
   一张图里 `depth < target` 就等于"目标被墙挡住了"，模型自己组合。
3. `teammate` **不含自己**，也不含任何敌方；两队的划分看 `.csz` 的 `teams`，
   队伍未知（`-1`）的人两边都不算。

## 烟雾那一通道的已知饱和

`solid=True` 填的是**表面点云的凸包**（渲染选择，不是格式事实，见 `voxel.py`），
凸包比真实云大一圈，所以站在烟边上也可能"眼睛已经在实心体素里"——那时整张图
的 smoke 通道全是 0（距离 0 = 你在烟里）。实测 1064 张图里 44% 完全没烟；
有烟的 595 张里 **6.1% 是这种全 0**，其余正常给出到云面的距离。
要"必须为真"的遮挡判定，**也**用这个实心栅格（`obstruction.smoke_volumes` 就是调
这里的）—— 别改成 `voxel.boundary_voxels()`：壳是稀疏采样、内部有整片空洞，按它判
遮挡会把 45.8% 的"被烟挡"记成"通"（实测见 `obstruction.py::smoke_volumes` 的表）。
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from numba import njit, prange

#: ⚠ **故意关掉 numba 的磁盘缓存**（每个 `@njit` 都写着 `cache=False`）。
#:
#: numba 的缓存文件按**源文件**存，但里面记的是**导入时用的模块名**。这个仓库有两个
#: 合法的导入路径 —— `python -m scripts.training_data.prepare` 得到
#: `scripts.training_data.view_render`，而把 `scripts/` 塞进 `sys.path` 的探针脚本
#: 得到 `training_data.view_render`。两条路径共用同一个缓存文件，后跑的那条会在
#: 装载时 `ModuleNotFoundError: No module named 'training_data'` 直接炸掉。
#: 代价是每个进程多 ~10 s 编译（一次），比"换个入口就崩"划算。

__all__ = [
    "VIEW_ROWS", "VIEW_COLS", "VIEW_CHANNELS", "VIEW_SHAPE", "VIEW_FAR",
    "FOV_H_DEG", "FOV_V_DEG", "ASPECT", "FOV_V_DEG_EXACT",
    "HULL_HALF_HU", "HULL_STAND_HU", "HULL_CROUCH_HU",
    "VOXEL_CELL", "VOXEL_DIM",
    "MapMesh", "load_map", "render_view", "hull_boxes", "smoke_volumes",
    "eye_of", "directions",
]

# ── 画布 ────────────────────────────────────────────────────────────────────

#: 行 = pitch 格数，列 = yaw 格数。
VIEW_ROWS = 192
VIEW_COLS = 256

#: 通道名（第 3 轴的下标就是这个元组的顺序）。
VIEW_CHANNELS: tuple[str, ...] = ("depth", "smoke", "enemy", "teammate", "target")

#: 一张图的形状。多一个通道就改这里 + `_render_kernel` 里那句赋值。
VIEW_SHAPE = (VIEW_ROWS, VIEW_COLS, len(VIEW_CHANNELS))

#: 没打到任何东西时的哨兵距离（HU）。1e9 ≈ 2539 km，比任何地图都大得多。
VIEW_FAR = 1e9

# ── 视锥 ────────────────────────────────────────────────────────────────────

#: 水平 FOV（度）。**和 `visibility.js::DEFAULT_FOV_DEG` 保持一致**（= `m_iFOV` 默认值）。
FOV_H_DEG = 90.0

#: 宽高比 = 列数 / 行数 = 4:3。
ASPECT = VIEW_COLS / VIEW_ROWS

#: 垂直 FOV（度）= `2·atan(tan(FOV_H/2) / ASPECT)` ≈ 73.7398°。
FOV_V_DEG_EXACT = 2.0 * math.degrees(math.atan(math.tan(math.radians(FOV_H_DEG / 2.0)) / ASPECT))
FOV_V_DEG = float(FOV_V_DEG_EXACT)

# ── 玩家包围盒（照抄 `visibility.js`）────────────────────────────────────────

#: 碰撞盒半宽（HU）：32/2。世界轴对齐，不随 yaw 转。
HULL_HALF_HU = 16.0

#: 站立 / 蹲下时的盒高（HU）。蹲下用 `crouching`（布尔）**阶跃**，不是 `crouch` 的斜坡。
HULL_STAND_HU = 72.0
HULL_CROUCH_HU = 54.0

# ── 烟雾体素（照抄 `smoke-voxels.js` / `voxel.py`）──────────────────────────

VOXEL_DIM = 32
VOXEL_CELL = 12.0
# 格子 i 覆盖 `det + (i-16)·CELL .. det + (i-15)·CELL`，所以整个烟是一个
# `32·CELL = 384 HU` 的立方体，中心在引爆点（`voxel.grid_to_world()` 的格心约定
# 是 `det + (i-15.5)·CELL`，两者差半格，`_dda_dist` 用的是**格边界**）。

# ── BVH 参数（和 `map-raycast.js` 同值）──────────────────────────────────────

_LEAF_SIZE = 8
_MAX_DEPTH = 48
_EPS = 1e-9
_BARY_EPS = 1e-9

#: `scene.js::SCALE`：three 空间是"米"，game 空间是 HU。OBJ 是 three 空间。
_SCALE = 0.0254

_STACK = 128


# ═══════════════════════════════════════════════════════════════════════════
# 射线 / 三角形（`map-raycast.js` 的逐行移植，只是换成 numba）
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠ 所有做浮点除法的 kernel 都写了 `error_model="numpy"`。numba 的默认模型是
# `"python"`：`1.0 / 0.0` 直接抛 `ZeroDivisionError`（实测 numba 0.68）。而这里的几何
# 代码是照 JS 写的，**靠 ±inf / NaN 被比较自然吃掉**（`_slab` 的注释、`_cast` 的
# `1.0/dx`、`_box_dist` 的 `1.0/d`）。`directions()` 真的会给出**精确的 0**：
# de_cache 有一帧 yaw=113.73046875，第 60 列的 `cos_y` 在浮点里正好舍入成 0.0，
# 于是 `1.0/dx` 抛 ZeroDivisionError，被 numba 的并行区包装成
# `SystemError: ... returned a result with an exception set`，看不出真因。
# 改成 numpy 语义后 `1/±0 = ±inf`，和浏览器里逐字节一致。


@njit(cache=False, inline="always", error_model="numpy")
def _ray_tri(verts, t, ox, oy, oz, dx, dy, dz):
    """Möller–Trumbore，**双面**。命中返回 `t`（方向归一化时就是距离），否则 -1。"""
    i = t * 9
    ax = verts[i]
    ay = verts[i + 1]
    az = verts[i + 2]
    e1x = verts[i + 3] - ax
    e1y = verts[i + 4] - ay
    e1z = verts[i + 5] - az
    e2x = verts[i + 6] - ax
    e2y = verts[i + 7] - ay
    e2z = verts[i + 8] - az

    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if det > -_EPS and det < _EPS:            # 射线和三角形共面
        return -1.0

    inv = 1.0 / det
    tx = ox - ax
    ty = oy - ay
    tz = oz - az
    u = (tx * px + ty * py + tz * pz) * inv
    if u < -_BARY_EPS or u > 1.0 + _BARY_EPS:
        return -1.0

    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < -_BARY_EPS or u + v > 1.0 + _BARY_EPS:
        return -1.0

    return (e2x * qx + e2y * qy + e2z * qz) * inv


@njit(cache=False, inline="always")
def _slab(node_min, node_max, ni, ox, oy, oz, invx, invy, invz):
    """射线 vs AABB。返回进入距离（起点在盒内时为 0），没命中返回 -1。"""
    b = ni * 3
    t0 = (node_min[b] - ox) * invx
    t1 = (node_max[b] - ox) * invx
    tmin = t0 if t0 < t1 else t1
    tmax = t1 if t0 < t1 else t0

    t0 = (node_min[b + 1] - oy) * invy
    t1 = (node_max[b + 1] - oy) * invy
    if t0 > t1:
        s = t0
        t0 = t1
        t1 = s
    if t0 > tmin:
        tmin = t0
    if t1 < tmax:
        tmax = t1
    if tmin > tmax:
        return -1.0

    t0 = (node_min[b + 2] - oz) * invz
    t1 = (node_max[b + 2] - oz) * invz
    if t0 > t1:
        s = t0
        t0 = t1
        t1 = s
    if t0 > tmin:
        tmin = t0
    if t1 < tmax:
        tmax = t1
    if tmin > tmax:
        return -1.0

    return 0.0 if tmin < 0.0 else tmin


@njit(cache=False, error_model="numpy")
def _cast(verts, order, node_min, node_max, node_left, node_right, node_start,
          node_count, ox, oy, oz, dx, dy, dz, far):
    """最近的地图交点距离；没打到返回 `far`。"""
    invx = 1.0 / dx
    invy = 1.0 / dy
    invz = 1.0 / dz

    stack = np.empty(_STACK, np.int32)
    sp = 0
    stack[sp] = 0
    sp += 1

    best = far
    found = False

    while sp > 0:
        sp -= 1
        ni = stack[sp]
        tmin = _slab(node_min, node_max, ni, ox, oy, oz, invx, invy, invz)
        if tmin < 0.0 or tmin > best:
            continue

        c = node_count[ni]
        if c > 0:
            s = node_start[ni]
            for k in range(s, s + c):
                t = order[k]
                hit = _ray_tri(verts, t, ox, oy, oz, dx, dy, dz)
                if hit >= 0.0 and hit < best:
                    best = hit
                    found = True
        else:
            left = node_left[ni]
            right = node_right[ni]
            tl = _slab(node_min, node_max, left, ox, oy, oz, invx, invy, invz)
            tr = _slab(node_min, node_max, right, ox, oy, oz, invx, invy, invz)
            l_ok = tl >= 0.0 and tl <= best
            r_ok = tr >= 0.0 and tr <= best
            if l_ok and r_ok:
                # 远的先压栈 → 近的先弹出来，能更早收紧 best
                if tl <= tr:
                    stack[sp] = right
                    sp += 1
                    stack[sp] = left
                    sp += 1
                else:
                    stack[sp] = left
                    sp += 1
                    stack[sp] = right
                    sp += 1
            elif l_ok:
                stack[sp] = left
                sp += 1
            elif r_ok:
                stack[sp] = right
                sp += 1

    return best if found else far


@njit(cache=False, inline="always", error_model="numpy")
def _box_dist(ox, oy, oz, dx, dy, dz, bx, by, bz, half, height, far):
    """射线到**世界轴对齐方块**的距离（进入距离 = 到盒面的距离）。

    `bx/by/bz` 是**脚底**（盒底），盒 = `[b±half] × [bz, bz+height]`，和
    `visibility.js::rayHull()` 同一套 slab。没打到 / 在背后 → `far`。
    """
    lo0 = bx - half
    hi0 = bx + half
    lo1 = by - half
    hi1 = by + half
    lo2 = bz
    hi2 = bz + height

    tmin = 0.0
    tmax = far

    # 逐轴 slab。方向分量恰好是 0 时 `inv` 是 ±inf，NaN 会被比较自然忽略
    # （和 JS 一样：贴着面这种零测度情形不影响结果）。
    for axis in range(3):
        if axis == 0:
            o, d, lo, hi = ox, dx, lo0, hi0
        elif axis == 1:
            o, d, lo, hi = oy, dy, lo1, hi1
        else:
            o, d, lo, hi = oz, dz, lo2, hi2

        if d > -1e-12 and d < 1e-12:
            if o < lo or o > hi:
                return far
            continue

        inv = 1.0 / d
        a = (lo - o) * inv
        b = (hi - o) * inv
        if a > b:
            t = a
            a = b
            b = t
        if a > tmin:
            tmin = a
        if b < tmax:
            tmax = b
        if tmin > tmax:
            return far

    return tmin


@njit(cache=False, inline="always", error_model="numpy")
def _dda_dist(grid, cell, detx, dety, detz, ox, oy, oz, dx, dy, dz, max_dist):
    """射线穿过一颗烟的体素栅格，返回**进入第一个被占据体素的距离**；没穿到返回 `max_dist`。

    Amanatides–Woo 3D DDA，和 `visibility.js::ddaBlocked()` 同构，只多了"记下每格的
    进入参数"（那个函数只回答 true/false，这里要距离）。
    """
    dim = grid.shape[0]
    span = dim * cell
    half = span * 0.5
    bx = detx - half
    by = dety - half
    bz = detz - half

    # ① AABB 拒绝：绝大多数射线根本不在这个 384 HU 的立方里
    t0 = 0.0
    t1 = max_dist

    if dx > -1e-12 and dx < 1e-12:
        if ox < bx or ox > bx + span:
            return max_dist
    else:
        inv = 1.0 / dx
        a = (bx - ox) * inv
        b = (bx + span - ox) * inv
        if a > b:
            t = a
            a = b
            b = t
        if a > t0:
            t0 = a
        if b < t1:
            t1 = b
        if t0 > t1:
            return max_dist

    if dy > -1e-12 and dy < 1e-12:
        if oy < by or oy > by + span:
            return max_dist
    else:
        inv = 1.0 / dy
        a = (by - oy) * inv
        b = (by + span - oy) * inv
        if a > b:
            t = a
            a = b
            b = t
        if a > t0:
            t0 = a
        if b < t1:
            t1 = b
        if t0 > t1:
            return max_dist

    if dz > -1e-12 and dz < 1e-12:
        if oz < bz or oz > bz + span:
            return max_dist
    else:
        inv = 1.0 / dz
        a = (bz - oz) * inv
        b = (bz + span - oz) * inv
        if a > b:
            t = a
            a = b
            b = t
        if a > t0:
            t0 = a
        if b < t1:
            t1 = b
        if t0 > t1:
            return max_dist

    if t1 < 0.0:
        return max_dist

    # ② 稍微推进一点，保证起点严格在盒内（贴着面进的时候 floor 会落到 dim）
    t0 += 1e-6
    if t0 > t1:
        return max_dist

    px = ox + dx * t0
    py = oy + dy * t0
    pz = oz + dz * t0
    gx = (px - bx) / cell
    gy = (py - by) / cell
    gz = (pz - bz) / cell
    i = int(math.floor(gx))
    j = int(math.floor(gy))
    k = int(math.floor(gz))
    if i < 0:
        i = 0
    elif i >= dim:
        i = dim - 1
    if j < 0:
        j = 0
    elif j >= dim:
        j = dim - 1
    if k < 0:
        k = 0
    elif k >= dim:
        k = dim - 1

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

    adx = -dx if dx < 0 else dx
    ady = -dy if dy < 0 else dy
    adz = -dz if dz < 0 else dz
    tdx = cell / adx if step_x != 0 else math.inf
    tdy = cell / ady if step_y != 0 else math.inf
    tdz = cell / adz if step_z != 0 else math.inf
    tmax_x = t0 + ((i + 1 - gx if step_x > 0 else gx - i) * cell) / adx if step_x != 0 else math.inf
    tmax_y = t0 + ((j + 1 - gy if step_y > 0 else gy - j) * cell) / ady if step_y != 0 else math.inf
    tmax_z = t0 + ((k + 1 - gz if step_z > 0 else gz - k) * cell) / adz if step_z != 0 else math.inf

    t_enter = t0
    for _ in range(3 * dim + 8):
        if grid[i, j, k]:
            return t_enter

        if tmax_x <= tmax_y:
            if tmax_x <= tmax_z:
                if tmax_x > t1:
                    return max_dist
                t_enter = tmax_x
                i += step_x
                if i < 0 or i >= dim:
                    return max_dist
                tmax_x += tdx
            else:
                if tmax_z > t1:
                    return max_dist
                t_enter = tmax_z
                k += step_z
                if k < 0 or k >= dim:
                    return max_dist
                tmax_z += tdz
        elif tmax_y <= tmax_z:
            if tmax_y > t1:
                return max_dist
            t_enter = tmax_y
            j += step_y
            if j < 0 or j >= dim:
                return max_dist
            tmax_y += tdy
        else:
            if tmax_z > t1:
                return max_dist
            t_enter = tmax_z
            k += step_z
            if k < 0 or k >= dim:
                return max_dist
            tmax_z += tdz

    return max_dist


# ═══════════════════════════════════════════════════════════════════════════
# 整张图：一条射线一个像素
# ═══════════════════════════════════════════════════════════════════════════


@njit(parallel=True, cache=False, error_model="numpy")
def _render_kernel(dx, dy, dz, ox, oy, oz,
                   verts, order, node_min, node_max, node_left, node_right,
                   node_start, node_count,
                   enemy, mate, tgt,
                   smoke_det, smoke_grid,
                   out):
    """`out` 形状 `(n, 5)`，第 0..4 列 = depth smoke enemy teammate target。"""
    n = dx.size
    far = VIEW_FAR
    n_enemy = enemy.shape[0]
    n_mate = mate.shape[0]
    n_tgt = tgt.shape[0]
    n_smoke = smoke_grid.shape[0]

    for p in prange(n):
        ex = ox
        ey = oy
        ez = oz
        ddx = dx[p]
        ddy = dy[p]
        ddz = dz[p]

        # ① 地图
        out[p, 0] = _cast(verts, order, node_min, node_max, node_left, node_right,
                          node_start, node_count, ex, ey, ez, ddx, ddy, ddz, far)

        # ② 烟（多颗取最近的）
        best = far
        for v in range(n_smoke):
            t = _dda_dist(smoke_grid[v], VOXEL_CELL,
                          smoke_det[v, 0], smoke_det[v, 1], smoke_det[v, 2],
                          ex, ey, ez, ddx, ddy, ddz, far)
            if t < best:
                best = t
        out[p, 1] = best

        # ③④⑤ 三个包围盒通道
        best = far
        for q in range(n_enemy):
            t = _box_dist(ex, ey, ez, ddx, ddy, ddz,
                          enemy[q, 0], enemy[q, 1], enemy[q, 2],
                          HULL_HALF_HU, enemy[q, 3], far)
            if t < best:
                best = t
        out[p, 2] = best

        best = far
        for q in range(n_mate):
            t = _box_dist(ex, ey, ez, ddx, ddy, ddz,
                          mate[q, 0], mate[q, 1], mate[q, 2],
                          HULL_HALF_HU, mate[q, 3], far)
            if t < best:
                best = t
        out[p, 3] = best

        best = far
        for q in range(n_tgt):
            t = _box_dist(ex, ey, ez, ddx, ddy, ddz,
                          tgt[q, 0], tgt[q, 1], tgt[q, 2],
                          HULL_HALF_HU, tgt[q, 3], far)
            if t < best:
                best = t
        out[p, 4] = best


# ═══════════════════════════════════════════════════════════════════════════
# 地图：OBJ → game 空间的三角形汤 → BVH
# ═══════════════════════════════════════════════════════════════════════════


@njit(cache=False)
def _build_nodes(tri_min, tri_max, order,
                 node_min, node_max, node_left, node_right,
                 node_start, node_count, next_node):
    """中序遍历建树（显式栈，不用递归）。划分策略和 `map-raycast.js` 一致。"""
    n = order.size

    cap = _STACK
    st_lo = np.empty(cap, np.int64)
    st_hi = np.empty(cap, np.int64)
    st_dep = np.empty(cap, np.int64)
    st_par = np.empty(cap, np.int64)
    st_side = np.empty(cap, np.int64)

    st_lo[0] = 0
    st_hi[0] = n
    st_dep[0] = 0
    st_par[0] = -1
    st_side[0] = 0
    sp = 1

    while sp > 0:
        sp -= 1
        lo = st_lo[sp]
        hi = st_hi[sp]
        dep = st_dep[sp]
        par = st_par[sp]
        side = st_side[sp]

        ni = next_node[0]
        next_node[0] = ni + 1
        if par >= 0:
            if side == 0:
                node_left[par] = ni
            else:
                node_right[par] = ni

        # 节点包围盒 + 质心包围盒
        mnx = 1e30
        mny = 1e30
        mnz = 1e30
        mxx = -1e30
        mxy = -1e30
        mxz = -1e30
        cnx = 1e30
        cny = 1e30
        cnz = 1e30
        cxx = -1e30
        cxy = -1e30
        cxz = -1e30
        for k in range(lo, hi):
            t = order[k]
            a = t * 3
            ax = tri_min[a]
            ay = tri_min[a + 1]
            az = tri_min[a + 2]
            bx = tri_max[a]
            by = tri_max[a + 1]
            bz = tri_max[a + 2]
            if ax < mnx:
                mnx = ax
            if ay < mny:
                mny = ay
            if az < mnz:
                mnz = az
            if bx > mxx:
                mxx = bx
            if by > mxy:
                mxy = by
            if bz > mxz:
                mxz = bz
            gx = (ax + bx) * 0.5
            gy = (ay + by) * 0.5
            gz = (az + bz) * 0.5
            if gx < cnx:
                cnx = gx
            if gy < cny:
                cny = gy
            if gz < cnz:
                cnz = gz
            if gx > cxx:
                cxx = gx
            if gy > cxy:
                cxy = gy
            if gz > cxz:
                cxz = gz

        node_min[ni * 3] = mnx
        node_min[ni * 3 + 1] = mny
        node_min[ni * 3 + 2] = mnz
        node_max[ni * 3] = mxx
        node_max[ni * 3 + 1] = mxy
        node_max[ni * 3 + 2] = mxz
        node_left[ni] = -1
        node_right[ni] = -1
        node_start[ni] = lo
        node_count[ni] = 0

        count = hi - lo
        if count <= _LEAF_SIZE or dep >= _MAX_DEPTH:
            node_count[ni] = count
            continue

        exx = cxx - cnx
        eyy = cxy - cny
        ezz = cxz - cnz
        axis = 0
        ext = exx
        if eyy > ext:
            axis = 1
            ext = eyy
        if ezz > ext:
            axis = 2
            ext = ezz

        if ext <= 0.0:                       # 所有质心重合，切不动了
            node_count[ni] = count
            continue

        split = (cnx + cxx) * 0.5 if axis == 0 else ((cny + cxy) * 0.5 if axis == 1 else (cnz + cxz) * 0.5)

        i = lo
        j = hi - 1
        while i <= j:
            t = order[i]
            c = (tri_min[t * 3 + axis] + tri_max[t * 3 + axis]) * 0.5
            if c < split:
                i += 1
            else:
                tmp = order[i]
                order[i] = order[j]
                order[j] = tmp
                j -= 1
        mid = i
        if mid == lo or mid == hi:            # 退化：按下标中位数硬切
            mid = (lo + hi) >> 1

        # 左先入栈、右后入栈 —— 反正两个孩子的节点号都在弹出时写回
        st_lo[sp] = mid
        st_hi[sp] = hi
        st_dep[sp] = dep + 1
        st_par[sp] = ni
        st_side[sp] = 1
        sp += 1
        st_lo[sp] = lo
        st_hi[sp] = mid
        st_dep[sp] = dep + 1
        st_par[sp] = ni
        st_side[sp] = 0
        sp += 1


class MapMesh:
    """一张地图的三角形汤 + BVH。用 `load_map(map_name)` 拿（有缓存）。"""

    __slots__ = ("path", "verts", "tri_count", "order", "node_min", "node_max",
                 "node_left", "node_right", "node_start", "node_count", "nodes",
                 "bounds")

    def __init__(self, path: Path, verts: np.ndarray, tri_count: int) -> None:
        self.path = Path(path)
        self.verts = verts                      # (tri*9,) float32，game 空间 HU
        self.tri_count = int(tri_count)
        self.order = np.empty(0, np.uint32)
        self.node_min = np.empty(0, np.float32)
        self.node_max = np.empty(0, np.float32)
        self.node_left = np.empty(0, np.int32)
        self.node_right = np.empty(0, np.int32)
        self.node_start = np.empty(0, np.uint32)
        self.node_count = np.empty(0, np.uint32)
        self.nodes = 0
        self.bounds = (np.zeros(3, np.float32), np.zeros(3, np.float32))

    def build(self) -> MapMesh:
        """建 BVH。约 1 s（mirage 122 万三角形）。"""
        tri_min, tri_max = _tri_bounds(self.verts, self.tri_count)
        self.order = np.arange(self.tri_count, dtype=np.uint32)
        max_nodes = 2 * self.tri_count + 1
        node_min = np.empty(max_nodes * 3, np.float32)
        node_max = np.empty(max_nodes * 3, np.float32)
        node_left = np.empty(max_nodes, np.int32)
        node_right = np.empty(max_nodes, np.int32)
        node_start = np.empty(max_nodes, np.uint32)
        node_count = np.empty(max_nodes, np.uint32)
        counter = np.zeros(1, np.int64)
        _build_nodes(tri_min, tri_max, self.order,
                     node_min, node_max, node_left, node_right,
                     node_start, node_count, counter)
        n = int(counter[0])
        self.nodes = n
        self.node_min = node_min[: n * 3].copy()
        self.node_max = node_max[: n * 3].copy()
        self.node_left = node_left[:n].copy()
        self.node_right = node_right[:n].copy()
        self.node_start = node_start[:n].copy()
        self.node_count = node_count[:n].copy()
        self.bounds = (tri_min.reshape(-1, 3).min(0), tri_max.reshape(-1, 3).max(0))
        return self

    def cast(self, origin, dirs: np.ndarray, far: float = VIEW_FAR) -> np.ndarray:
        """`(N,3)` 单位方向 → `(N,)` 最近的地图距离（没打到 = `far`）。调试/对拍用。"""
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        dirs = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1, 3)
        out = np.empty(len(dirs), np.float32)
        _cast_many(self.verts, self.order, self.node_min, self.node_max,
                   self.node_left, self.node_right, self.node_start, self.node_count,
                   float(origin[0]), float(origin[1]), float(origin[2]),
                   dirs, float(far), out)
        return out


@njit(cache=False)
def _tri_bounds(verts, tri_count):
    tri_min = np.empty(tri_count * 3, np.float32)
    tri_max = np.empty(tri_count * 3, np.float32)
    for t in range(tri_count):
        i = t * 9
        ax = verts[i]
        ay = verts[i + 1]
        az = verts[i + 2]
        bx = verts[i + 3]
        by = verts[i + 4]
        bz = verts[i + 5]
        cx = verts[i + 6]
        cy = verts[i + 7]
        cz = verts[i + 8]
        tri_min[t * 3] = min(ax, min(bx, cx))
        tri_min[t * 3 + 1] = min(ay, min(by, cy))
        tri_min[t * 3 + 2] = min(az, min(bz, cz))
        tri_max[t * 3] = max(ax, max(bx, cx))
        tri_max[t * 3 + 1] = max(ay, max(by, cy))
        tri_max[t * 3 + 2] = max(az, max(bz, cz))
    return tri_min, tri_max


@njit(parallel=True, cache=False)
def _cast_many(verts, order, node_min, node_max, node_left, node_right,
               node_start, node_count, ox, oy, oz, dirs, far, out):
    for i in prange(out.size):
        out[i] = _cast(verts, order, node_min, node_max, node_left, node_right,
                       node_start, node_count,
                       ox, oy, oz, dirs[i, 0], dirs[i, 1], dirs[i, 2], far)


def read_obj_tris(path: str | Path) -> tuple[np.ndarray, int]:
    """OBJ → `(tri*9,) float32`，**已经换到 game 空间（HU）**。

    只有 `v ` 和 `f ` 两种行（这些 OBJ 是 Open3D 导出的，没有 `vt` / `vn` / 材质），
    多边形按扇形三角化（和 `vis_common.mjs::readObjSoup` 一样）。

    three 空间 → game 空间是 `(gx,gy,gz) = (tz,tx,ty)/SCALE`（`scene.js::gameToThree` 的逆）。
    先在 three 空间三角化再换轴，省得每行都置换一次。
    """
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")
    v_lines = [ln[2:] for ln in lines if ln[:2] == b"v "]
    f_lines = [ln[2:] for ln in lines if ln[:2] == b"f "]
    if not v_lines:
        raise ValueError(f"OBJ 里没有顶点：{path}")

    v = np.array(b" ".join(v_lines).split(), dtype=np.float64)
    if v.size % 3:
        raise ValueError(f"OBJ 的 `v` 行不完整：{path}")
    v = v.reshape(-1, 3)
    nv = len(v)

    fb = b" ".join(f_lines)
    # `a` / `a/b` / `a/b/c` 都只取第一段（这些地图没有 `vt`/`vn`，但别的地图可能带）
    if b"/" in fb:
        faces = [tok.split(b"/")[0] for tok in fb.split()]
    else:
        faces = fb.split()
    if len(faces) % 3:
        # 慢路径：有多边形。按行做扇形三角化（和 `vis_common.mjs::readObjSoup` 一样）
        tri: list[bytes] = []
        for ln in f_lines:
            tok = [t.split(b"/")[0] for t in ln.split()]
            for k in range(1, len(tok) - 1):
                tri += [tok[0], tok[k], tok[k + 1]]
        faces = tri
    if not faces:
        raise ValueError(f"OBJ 里没有面：{path}")
    idx = np.array(faces, dtype=np.int64) - 1
    if idx.size % 3:
        raise ValueError(f"OBJ 的面解析不出来：{path}（{idx.size} 个顶点下标）")
    idx = idx.reshape(-1, 3)
    ok = (idx >= 0).all(axis=1) & (idx < nv).all(axis=1)
    idx = idx[ok]

    # three → game：gx = tz, gy = tx, gz = ty，再除 SCALE
    g = v[:, [2, 0, 1]] / _SCALE
    tris = g[idx].astype(np.float32)
    return np.ascontiguousarray(tris.reshape(-1)), len(idx)


#: `maxsize=8`：**按地图名的数量给，不是按 csz 的数量**。原来写 4，全量语料有 8 张图
#: （mirage 一张就 45.6 万节点 / 23 MB）时，一个 worker 转到第 5 张就会把第 1 张挤掉，
#: 等它再回头处理第 1 张的文件时**重新建一次 BVH**（3.6 s + 23 MB）。最坏情况就是每份
#: csz 重建一次，8000 份 ≈ 8 核·小时白烧。8 张全常驻 ≈ 200～400 MB/进程，相对
#: `workers × 1.7 GB` 的 csz 峰值可以忽略。
@lru_cache(maxsize=8)
def load_map(map_name: str, obj_path: str | None = None) -> MapMesh:
    """`'de_mirage'` → 建好 BVH 的 `MapMesh`（同一进程里每种地图只加载一次）。

    地图 OBJ 来自 `maps/optimized_obj_files/<map>.obj`；`obj_path` 只是给测试替换用的。
    """
    if obj_path is None:
        from .vis_oracle import map_obj_for

        obj_path = str(map_obj_for(map_name))
    verts, tri_count = read_obj_tris(obj_path)
    return MapMesh(Path(obj_path), verts, tri_count).build()


# ═══════════════════════════════════════════════════════════════════════════
# 位姿 / 包围盒 / 烟雾
# ═══════════════════════════════════════════════════════════════════════════


def _col(rnd: dict, p: int, key: str):
    return rnd["players"][p][key]


def eye_of(rnd: dict, p: int, idx: int) -> np.ndarray:
    """第 `idx` 帧第 `p` 个玩家的**眼睛**世界坐标（`z + eye_z`，HU）。

    眼高**不在** `rnd["players"][p]` 里：它是 `recipes.eye_height()` 现算的
    `STANDING_EYE_HEIGHT_HU - CROUCH_EYE_DROP_HU · crouch`（`64.0938 - 18·crouch`，
    蹲到底 46.0938）。直接调那个函数，别在这里抄常数 —— `round_tensors()["eye_z"]`
    用的就是它，两边必须是同一个式子。
    """
    from demo_extract import recipes

    crouch = _col(rnd, p, "crouch")[idx]
    if crouch is None or not np.isfinite(crouch):
        crouch = 0.0
    eye_z = recipes.STANDING_EYE_HEIGHT_HU - recipes.CROUCH_EYE_DROP_HU * float(crouch)
    z = float(_col(rnd, p, "z")[idx])
    return np.array(
        [float(_col(rnd, p, "x")[idx]), float(_col(rnd, p, "y")[idx]), z + eye_z],
        dtype=np.float64,
    )


def hull_boxes(rnd: dict, idx: int, players: Sequence[int]) -> np.ndarray:
    """一组玩家在第 `idx` 帧的包围盒 `(N, 4)`：`[x, y, z, 盒高]`（脚底坐标，HU）。"""
    out = np.empty((len(players), 4), dtype=np.float32)
    for i, p in enumerate(players):
        out[i, 0] = float(_col(rnd, p, "x")[idx])
        out[i, 1] = float(_col(rnd, p, "y")[idx])
        out[i, 2] = float(_col(rnd, p, "z")[idx])
        crouching = _col(rnd, p, "crouching")[idx]
        out[i, 3] = HULL_CROUCH_HU if bool(crouching) else HULL_STAND_HU
    return out


def smoke_volumes(rnd: dict, tick: int, cache: dict | None = None):
    """这一刻所有生效烟雾的 `(det (nv,3) float32, grid (nv,32,32,32) uint8)`。

    `solid=True`（凸包填充），和浏览器默认一致。`cache` 用来摊掉那次纯 Python
    凸包（实测 **101 ms/颗/个 change index**，见 `spatial.py`）—— 传同一个 dict
    进来的话，同一颗烟的同一个"变更序号"只算一次。

    ## 缓存的两条硬性要求

    1. **存活期必须在查缓存之前判。** `voxel._record_index()` 只做二分：``tick >
       last_tick``（烟已经散了）时它照样返回最后一条记录的序号，于是"散之前"和
       "散之后"是**同一个键**，命中的是散之前那张栅格。`voxel.journal_at()` 里有
       同一条 ``first_tick <= tick <= last_tick`` 的守卫，但缓存命中时根本走不到
       那里 —— 漏了这一步，烟散了之后会一直被当成有烟。
    2. **`first_tick` 也是键的一部分。** 同一个回合里 `entity_id` 会被复用（Source
       的实体槽回收），只按 ``(entity_id, rec)`` 缓存的话，后一颗烟会覆盖 / 读到
       前一颗烟的栅格（`rec` 是各自 `changes` 列表里的下标，天然会撞）。
    """
    from demo_extract import recipes
    from demo_extract import voxel as _voxel

    dts: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    for inst in rnd.get("smoke") or []:
        det = inst.get("detonation")
        if det is None:
            continue                               # 放不进世界坐标：和浏览器一样丢掉
        first_tick = int(inst["first_tick"])
        if tick < first_tick or tick > int(inst["last_tick"]):
            continue                               # 还没炸 / 已经散了：和 journal_at 同一条守卫
        key = None
        if cache is not None:
            rec = _voxel._record_index(inst, tick)          # noqa: SLF001 —— 只为缓存键
            key = ("smoke", inst["entity_id"], first_tick, rec)
            hit = cache.get(key)
            if hit is not None:
                if hit is False:
                    continue                       # 记过"这一刻这颗烟还没有几何"
                dts.append(hit[0])
                grids.append(hit[1])
                continue
        grid = recipes.smoke_grid_at(inst, tick, solid=True)
        if grid is None or not grid.any():
            if cache is not None:
                cache[key] = False
            continue
        det_arr = np.asarray(det, dtype=np.float32).reshape(3)
        g = np.ascontiguousarray(grid, dtype=np.uint8)
        if cache is not None:
            cache[key] = (det_arr, g)
        dts.append(det_arr)
        grids.append(g)

    if not dts:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), np.uint8))
    return np.stack(dts), np.stack(grids)


def directions(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """整张图的单位方向 `(dx, dy, dz)`，各 `(192, 256)`。

    第 k 列的 yaw = `yaw + (256/2 − 0.5 − k) · (90°/256)`，
    第 j 行的 pitch = `pitch + (j + 0.5 − 192/2) · (V_FOV/192)`。

    **列的方向就是屏幕的左右**：第 0 列 = 最左 = 视线往左 45°。Source 的 yaw 从
    上方看逆时针增大（`yaw 0` 看 `+X`，`yaw +90` 看 `+Y`），玩家的右手是
    `f × up = (sin y, −cos y, 0)`（`visibility.js::viewBasis()` 的 r），所以 yaw
    变大 = 往**左**看 = 往屏幕**左**边去，式子里那个负号就是这件事。别把它"顺手
    改成正号"：那样画出来是左右镜像的第一人称（真实画面里在左边的墙会跑到右边）。

    方向公式和 `visibility.js::viewBasis()` 的 forward 完全一样：
    `f = (cos y·cos p, sin y·cos p, −sin p)`（**pitch > 0 是向下看**，Source 约定）。
    角度用和角公式展开（`cos(y+dy) = cos y cos dy − sin y sin dy`），
    这样一张图只要 4 次三角函数（预计算的 dy 表另算）。
    """
    # 注意 (COLS/2 − 0.5 − k)：k=0 是 +44.82°（最左），k=255 是 −44.82°（最右）。
    dyaw = (VIEW_COLS / 2.0 - 0.5 - np.arange(VIEW_COLS, dtype=np.float64)) * (FOV_H_DEG / VIEW_COLS)
    dpitch = (np.arange(VIEW_ROWS, dtype=np.float64) + 0.5 - VIEW_ROWS / 2.0) * (FOV_V_DEG / VIEW_ROWS)

    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cdy, sdy = np.cos(np.radians(dyaw)), np.sin(np.radians(dyaw))
    cdp, sdp = np.cos(np.radians(dpitch)), np.sin(np.radians(dpitch))

    cos_y = cy * cdy - sy * sdy              # (COLS,)
    sin_y = sy * cdy + cy * sdy
    cos_p = cp * cdp - sp * sdp              # (ROWS,)
    sin_p = sp * cdp + cp * sdp

    dx = cos_y[None, :] * cos_p[:, None]
    dy = sin_y[None, :] * cos_p[:, None]
    dz = np.broadcast_to(-sin_p[:, None], (VIEW_ROWS, VIEW_COLS))
    return dx, dy, dz


def render_view(
    map_name: str,
    origin: np.ndarray,
    yaw: float,
    pitch: float,
    *,
    enemy: np.ndarray | None = None,
    teammate: np.ndarray | None = None,
    target: np.ndarray | None = None,
    smoke: tuple[np.ndarray, np.ndarray] | None = None,
    mesh: MapMesh | None = None,
) -> np.ndarray:
    """一张 `(192, 256, 5) float32` 的视角渲染图（5 个通道都是 HU 距离，1e9 = 无限远）。

    Args:
        map_name: `'de_mirage'` 这种；决定用哪张地图的 mesh（`load_map()` 有缓存）。
        origin: 眼睛的世界坐标（HU），`z` 已经是眼睛高度（见 `eye_of()`）。
        yaw, pitch: 度。pitch > 0 向下看（Source 约定）。
        enemy, teammate, target: `(N, 4)` 的 `[x, y, z, 盒高]`（脚底），
            `target` 通常就 1 行。三个都可以不传 = 那个通道全是 1e9。
        smoke: `smoke_volumes()` 的返回值；不传 = 没有烟。
        mesh: 已经加载好的地图（省一次查缓存）。不传就 `load_map(map_name)`。
    """
    if mesh is None:
        mesh = load_map(map_name)

    dx, dy, dz = directions(yaw, pitch)
    n = VIEW_ROWS * VIEW_COLS
    out = np.empty((n, len(VIEW_CHANNELS)), dtype=np.float32)

    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    enemy = empty_boxes if enemy is None else np.ascontiguousarray(enemy, dtype=np.float32).reshape(-1, 4)
    teammate = empty_boxes if teammate is None else np.ascontiguousarray(teammate, dtype=np.float32).reshape(-1, 4)
    target = empty_boxes if target is None else np.ascontiguousarray(target, dtype=np.float32).reshape(-1, 4)

    if smoke is None:
        det = np.zeros((0, 3), np.float32)
        grids = np.zeros((0, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), np.uint8)
    else:
        det, grids = smoke
        det = np.ascontiguousarray(det, dtype=np.float32).reshape(-1, 3)
        grids = np.ascontiguousarray(grids, dtype=np.uint8)
        if grids.size == 0:
            grids = np.zeros((0, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), np.uint8)

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    _render_kernel(
        np.ascontiguousarray(dx, dtype=np.float64).reshape(n),
        np.ascontiguousarray(dy, dtype=np.float64).reshape(n),
        np.ascontiguousarray(dz, dtype=np.float64).reshape(n),
        float(origin[0]), float(origin[1]), float(origin[2]),
        mesh.verts, mesh.order, mesh.node_min, mesh.node_max,
        mesh.node_left, mesh.node_right, mesh.node_start, mesh.node_count,
        enemy, teammate, target, det, grids, out,
    )
    return out.reshape(VIEW_SHAPE)
