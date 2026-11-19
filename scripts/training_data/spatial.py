"""fire / smoke 空间特征。**默认不用**，因为很贵。

实测代价（本机）：

    recipes.active_fire(rnd, tick)                  ~1.3 ms/次
    smoke_occupancy_at(inst, tick, solid=True)       3.9 ms/次  ← Qhull 凸包；没装 scipy 时 ~150 ms
    smoke_occupancy_at(inst, tick, solid=False)      2.4 ms/次

一颗烟有 ~1300 个 tick，**逐 tick 调用 `solid=True` 会跑几小时**。
正确做法：只在 journal 的 change tick（一颗烟约 200 次）上预计算，存成稀疏
数组，训练时按 tick 查表。

⚠ 语义警告：journal 只存了烟雾的**稀疏采样**（云内部本身就有成片空洞），
`solid=True` 填的是这些点的**凸包**——严格说也是渲染选择，不是格式事实。但实测
按壳（`solid=False`）判遮挡会把 **45.8%** 的"被烟挡"记成"通"，所以遮挡/视线/
子弹阻挡也要用 `solid=True`：标签和画面必须是同一个体积。测量见
`obstruction.py::smoke_volumes`。
"""

from __future__ import annotations

import numpy as np

from . import _paths  # noqa: F401

from demo_extract import recipes  # noqa: E402


def smoke_occupancy_at(inst: dict, tick: int, solid: bool = True) -> np.ndarray | None:
    """单颗烟的 (32,32,32) bool 占据栅格，或 None（那颗烟此刻不在）。

    `inst` 来自 `rnd["smoke"][k]`。见模块 docstring 的代价与语义警告。
    """
    return recipes.smoke_grid_at(inst, tick, solid=solid)


def smoke_points_at(inst: dict, tick: int, solid: bool = True) -> np.ndarray | None:
    """单颗烟的世界坐标体素中心 (N, 3)，Hammer 单位。"""
    return recipes.smoke_points_at(inst, tick, solid=solid)


def fire_points_at(rnd: dict, tick: int) -> np.ndarray:
    """这一刻所有燃烧中的火焰格世界坐标 (N, 3)，Hammer 单位。~1.3 ms/次。"""
    cells = recipes.active_fire(rnd, tick)
    if not cells:
        return np.zeros((0, 3), dtype=np.float32)
    pts = [c["points"][c["burning"]] if c["burning"].any() else c["points"] for c in cells]
    return np.concatenate(pts, axis=0).astype(np.float32)


def fire_volume_at(rnd: dict, tick: int, resolution: float = 32.0) -> dict:
    """把这一刻所有火焰体素化成一个世界空间占据栅格。

    返回 `{"origin", "resolution", "grid", "shape"}`。~0.8 ms/次。
    """
    return recipes.fire_volume_at(rnd, tick, resolution=resolution)
