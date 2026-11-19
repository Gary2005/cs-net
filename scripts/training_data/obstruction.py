"""玩家两两遮挡：**i 的眼睛能不能看见 j 的身体**，以及「是什么挡住了」。

    obstruction[i][j] ∈ {0, 1, 2}      0 = 没挡  1 = 被烟挡  2 = 被地图 mesh 挡

`aim` 任务把它按帧存成 `(T, 10, 10)` int8（见 `tasks/aim.py`）。这是「掩体」这件事
**唯一**可靠的表达方式：位置轨迹里有墙的坐标，但没有"这两个人之间隔着墙"这个关系，
而那正是瞄准之前必须知道的东西。

## 怎么判

照 `visualizer/static/js/visibility.js` 的 `pairVisibility()`：

1. 目标玩家取**包围盒的 8 个顶点**（世界轴对齐、半宽 16 HU、高 72 站 / 54 蹲，
   和 `view_render.py::hull_boxes` 同一个盒子；`z` 是**脚底**）。
2. 从观察者的**眼睛**（`z + 眼高`，眼高 = `64.0938 - 18·crouch`）向每个顶点发一条
   射线，分类：
   * 打到地图 mesh，且交点**比顶点更近** → 2；
   * 否则穿过烟雾体素（Amanatides–Woo 3D DDA，进入第一颗实心体素的距离更近）→ 1；
   * 都没有 → 0。
3. `obstruction[i][j] = min(8 个顶点)` —— 只要有一个顶点看得见就算「没挡全」，
   这就是"人不是一条线、露出一只手也能被打"的意思。

**2 的优先级高于 1**（和浏览器里 `else if` 的链式判断一致）：同一条射线既穿烟又被
墙挡住时记 2。取 `min` 之后，含义是"这 8 个顶点里最轻的那个遮挡"，所以
`0` 表示至少露出一个角，`1` 表示全被烟挡死（没有墙），`2` 表示全被墙挡死。

**不做视野（FOV）判定**：这是"挡不挡"，不是"在不在视野里"。朝哪看是模型自己的
输入（`yaw/pitch` 在 20 列里），要不要转过脸去是它该学的事。

## 为什么不是"视线打到包围盒中心"

中心一条射线会把"半个身子露在墙角外"判成看不见，而那恰恰是最常见的击杀场景。
8 个顶点是浏览器那套判定的原样搬过来，两边看到的必须是同一个世界。

## 代价

一帧 = `n_alive × (n_alive-1) × 8` 条射线（10 个人全活 = 720 条），每条射线
"BVH 求交 + 逐烟 DDA"。实测（mirage，39% 的帧有烟）单帧 **3~6 ms**（单线程），
一条 `T=142` 的样本约 **0.5 s**；`Occluder` 在**回合级**缓存结果，所以同一回合里
互相重叠的几条击杀窗口只算一遍。

死人不算：`obstruction[i][j]` 只有在 `i` 和 `j` 当帧都活着时才有意义，其余一律 0，
配合 `ctx_player_mask` 使用（0 = 死亡）。

## 和 `view_render.py` 的关系

BVH、烟雾体素、包围盒、眼高**全部复用**那边，一行都不重写 —— 那套几何必须和浏览器
逐字节一致（`view_render.py` 的 docstring 里有对齐说明）。这里只加了一个"逐对
8 顶点分类"的外壳。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .view_render import (
    HULL_CROUCH_HU,
    HULL_HALF_HU,
    HULL_STAND_HU,
    VOXEL_CELL,
    VOXEL_DIM,
    _cast,
    _dda_dist,
    load_map,
    smoke_volumes as _vr_smoke_volumes,
)

#: 三个取值的人话名（下标就是存进 npz 的值）。
OBSTRUCT_NAMES: tuple[str, str, str] = ("无遮挡", "烟", "墙")

#: 判定"打到了"的距离容差（HU）。顶点常常就贴在墙 / 烟面上（人身高 72 HU、烟是
#: 384 HU 的立方体），不留容差的话 `t == dist` 的浮点抖动会让同一场景时挡时通。
_TOL = 1.0

#: 一个盒子 8 个顶点。
_N_CORNER = 8


@njit(parallel=True, cache=False, error_model="numpy")
def _pair_kernel(
    eyes, corners,
    verts, order, node_min, node_max, node_left, node_right, node_start, node_count,
    smoke_det, smoke_grid, out,
):
    """`out[i,j]` = i 看 j 的遮挡类别（0/1/2）；i == j 留 0。

    `eyes (n,3)` 眼睛世界坐标，`corners (n,8,3)` 包围盒顶点，`out (n,n) int8`。
    `eyes` / `corners` 只装**当帧活着**的玩家（死人的行由调用方在外面填 0）。
    """
    n = eyes.shape[0]
    n_smoke = smoke_grid.shape[0]

    for i in prange(n):
        ex = eyes[i, 0]
        ey = eyes[i, 1]
        ez = eyes[i, 2]
        for j in range(n):
            if j == i:
                continue
            best = 2
            for c in range(8):
                dx = corners[j, c, 0] - ex
                dy = corners[j, c, 1] - ey
                dz = corners[j, c, 2] - ez
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                if dist < 1e-6:
                    best = 0                      # 顶点就在眼睛上：贴脸，无遮挡
                    break
                dx /= dist
                dy /= dist
                dz /= dist

                # 把 `far` 压到 dist：BVH 只关心比顶点更近的交点，剪枝更狠
                t = _cast(verts, order, node_min, node_max, node_left, node_right,
                          node_start, node_count, ex, ey, ez, dx, dy, dz, dist)
                if t < dist - _TOL:
                    kind = 2
                else:
                    kind = 0
                    for v in range(n_smoke):
                        s = _dda_dist(smoke_grid[v], VOXEL_CELL,
                                      smoke_det[v, 0], smoke_det[v, 1], smoke_det[v, 2],
                                      ex, ey, ez, dx, dy, dz, dist)
                        if s < dist - _TOL:
                            kind = 1
                            break
                if kind < best:
                    best = kind
                    if best == 0:
                        break                     # 露了一个角，不用再看其余顶点
            out[i, j] = best
    return out


def pair_matrix(
    eyes: np.ndarray, corners: np.ndarray, mesh, smoke
) -> np.ndarray:
    """一次算完 `n` 个玩家两两的遮挡 `(n, n) int8`。

    Args:
        eyes: `(n, 3)` 眼睛的世界坐标（HU，z 已经加了眼高）。
        corners: `(n, 8, 3)` 每个玩家包围盒的 8 个顶点（脚底那个盒子，HU）。
        mesh: `view_render.load_map()` 的结果。
        smoke: `view_render.smoke_volumes()` 的 `(det (m,3), grid (m,32,32,32))`。
    """
    eyes = np.ascontiguousarray(eyes, dtype=np.float64).reshape(-1, 3)
    corners = np.ascontiguousarray(corners, dtype=np.float64).reshape(-1, _N_CORNER, 3)
    n = eyes.shape[0]
    out = np.zeros((n, n), dtype=np.int8)
    if n < 2:
        return out

    det, grids = smoke
    det = np.ascontiguousarray(det, dtype=np.float32).reshape(-1, 3)
    grids = np.ascontiguousarray(grids, dtype=np.uint8)
    if grids.size == 0:
        grids = np.zeros((0, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), np.uint8)

    _pair_kernel(
        eyes, corners,
        mesh.verts, mesh.order, mesh.node_min, mesh.node_max,
        mesh.node_left, mesh.node_right, mesh.node_start, mesh.node_count,
        det, grids, out,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 从 `rnd` 取几何（和 `view_render.eye_of` / `hull_boxes` 同一套公式）
# ═══════════════════════════════════════════════════════════════════════════


class _Geometry:
    """一个回合的逐帧眼睛 / 包围盒，**预先把整回合算好**（每条击杀都要用）。

    眼高依赖 `crouch`（连续量，`recipes.eye_height()`），盒高依赖 `crouching`
    （布尔阶跃）—— 这两个不是同一个量，别合并（`view_render.py` 里有说明）。
    """

    def __init__(self, rnd: dict, n_pl: int) -> None:
        from demo_extract import recipes

        self.n_pl = n_pl
        self.n = len(rnd["round_seconds"])
        self.x = np.zeros((n_pl, self.n), np.float64)
        self.y = np.zeros((n_pl, self.n), np.float64)
        self.z = np.zeros((n_pl, self.n), np.float64)
        self.eye_z = np.zeros((n_pl, self.n), np.float64)
        self.hull_h = np.zeros((n_pl, self.n), np.float64)
        self.alive = np.zeros((n_pl, self.n), bool)

        for p in range(min(n_pl, len(rnd["players"]))):
            rec = rnd["players"][p]
            self.x[p] = _f(rec["x"])
            self.y[p] = _f(rec["y"])
            self.z[p] = _f(rec["z"])
            crouch = np.clip(np.nan_to_num(_f(rec["crouch"])), 0.0, 1.0)
            self.eye_z[p] = (
                recipes.STANDING_EYE_HEIGHT_HU - recipes.CROUCH_EYE_DROP_HU * crouch
            )
            crouching = np.asarray(rec["crouching"], dtype=bool)
            self.hull_h[p] = np.where(crouching, HULL_CROUCH_HU, HULL_STAND_HU)
            self.alive[p] = np.asarray(rec["alive"], dtype=bool)

    def eyes(self, idx: int, players: list[int]) -> np.ndarray:
        """`(len(players), 3)` 眼睛世界坐标（z 已加眼高）。"""
        p = np.asarray(players, dtype=np.int64)
        return np.stack(
            [self.x[p, idx], self.y[p, idx], self.z[p, idx] + self.eye_z[p, idx]], axis=-1
        )

    def corners(self, idx: int, players: list[int]) -> np.ndarray:
        """`(len(players), 8, 3)` 包围盒 8 个顶点（`z` 是脚底那一层）。"""
        p = np.asarray(players, dtype=np.int64)
        cx = self.x[p, idx][:, None]                               # (n,1)
        cy = self.y[p, idx][:, None]
        cz = self.z[p, idx][:, None]
        hh = self.hull_h[p, idx][:, None]
        hw = HULL_HALF_HU

        # 顺序按 (z, y, x) 展开，和 `visibility.js::playerCorners` 一致（顺序不影响
        # 结果，只是照抄，方便和对拍脚本逐顶点比较）。
        xx = cx + np.array([[-hw, +hw]])                           # (n,2)
        yy = cy + np.array([[-hw, +hw]])
        out = np.empty((len(p), _N_CORNER, 3), np.float64)
        k = 0
        for zz in (cz, cz + hh):
            for j in range(2):
                for i in range(2):
                    out[:, k, 0] = xx[:, i]
                    out[:, k, 1] = yy[:, j]
                    out[:, k, 2] = zz[:, 0]
                    k += 1
        return out


def _f(v) -> np.ndarray:
    """原始列 → float64 数组（`None` / NaN 变 0）。"""
    return np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)


def smoke_volumes(rnd: dict, tick: int, cache: dict | None = None):
    """这一刻所有生效烟雾的 `(det (n,3) float32, grid (n,32,32,32) uint8)`。

    直接转发 `view_render.smoke_volumes()` —— **标签和画面必须用同一个烟体积**，
    两个实现迟早会漂移（这个文件原来就是自己抄了一份，然后顺手把 `solid` 改成了
    `False`，见下）。

    那个 `solid=False`（只取 journal 点名的边界体素）的理由是"射线要进去就必须先
    穿过边界那一层壳"。**实测不成立**：掩码是稀疏采样，云内部本身就有成片的空洞。
    一颗满云：点名 1787 格、凸包 5459 格（3.05×），点名格只有 **6.5%** 是 6 邻域全占
    （凸包 74%），p50 邻居数 4/6 —— DDA 直接从洞里穿过去。代价（4 份 csz、912 帧
    有烟、61 152 个有向对，`tmp/smoke_close_probe.py`）：

    | 烟栅格        | 判成"烟" | 比凸包漏判 | 漏判占"凸包烟" | ms/颗  |
    |---------------|---------:|-----------:|---------------:|-------:|
    | 边界壳        |      219 |        185 |         45.8%  |   0.55 |
    | 壳膨胀 2 格   |      409 |         32 |          7.9%  |   0.62 |
    | 凸包          |      404 |         —  |           0%   |   3.9  |

    "只取壳"会把近一半"其实被烟挡着"记成"通"（方向单向，只会漏不会多）。膨胀 k 格
    是当时留的后路（k=2 体积和凸包相当、几乎免费），但形状和可视化画的不一致，会在
    图上看出"烟明明挡着却显示通"，所以最终选凸包。

    ⚠ 那张表里"凸包 3.9 ms"是 `voxel.convex_hull_halfspaces` 换成 Qhull、填充换成
    numpy 之后的数。当初改成 `solid=False` 的理由是"凸包 35.7 ms，比壳慢 65 倍"——
    那个 65 倍来自纯 Python 凸包，现在只剩 **7 倍**，前提已经不成立了。谁再看到
    "遮挡就先用壳顶一下"的旧说法，先看 `voxel.HULL_BACKEND` 是不是 `"qhull"`。

    留给测试/打桩一个替换点（`tmp/` 里量速度就是把它换掉的），所以不在导入处直接
    alias —— 换掉的是**这个模块的**全局名。
    """
    return _vr_smoke_volumes(rnd, tick, cache)


class Occluder:
    """一个回合的遮挡计算器：**逐帧缓存**，同一帧只算一次。

    用法（`tasks/aim.py` 里就是这个）：

        occ = Occluder(rnd, map_name, n_pl)
        m = occ.matrix(first)                 # (10,10) int8，死人的行/列全是 0

    一个回合里几条击杀窗口互相重叠（同一段帧被 2~4 条样本用到），所以缓存放在
    回合级而不是样本级 —— 实测省 2~3 倍。
    """

    def __init__(self, rnd: dict, map_name: str, n_pl: int) -> None:
        self.rnd = rnd
        self.n_pl = n_pl
        self.geom = _Geometry(rnd, n_pl)
        self.mesh = load_map(map_name)
        self.ticks = np.asarray(rnd["ticks"], dtype=np.int64)
        self._smoke_cache: dict = {}
        self._frames: dict[int, np.ndarray] = {}

    def matrix(self, idx: int) -> np.ndarray:
        """第 `idx` 帧的 `(n_pl, n_pl)` int8 遮挡矩阵（死人的行/列留 0）。"""
        hit = self._frames.get(idx)
        if hit is not None:
            return hit

        out = np.zeros((self.n_pl, self.n_pl), np.int8)
        alive = [p for p in range(self.n_pl) if self.geom.alive[p, idx]]
        if len(alive) > 1:
            smoke = smoke_volumes(self.rnd, int(self.ticks[idx]), self._smoke_cache)
            m = pair_matrix(
                self.geom.eyes(idx, alive), self.geom.corners(idx, alive), self.mesh, smoke
            )
            out[np.ix_(alive, alive)] = m
        self._frames[idx] = out
        return out

    def window(self, start: int, stop: int) -> np.ndarray:
        """`[start, stop)` → `(stop-start, n_pl, n_pl)` int8。"""
        idx = range(start, stop)
        return np.stack([self.matrix(i) for i in idx]) if len(idx) else np.zeros(
            (0, self.n_pl, self.n_pl), np.int8
        )
