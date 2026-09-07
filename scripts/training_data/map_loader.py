"""
地图几何体加载器 — Open3D BVH 射线检测。

惰性加载优化后的 OBJ 文件，构建 RaycastingScene，按地图名缓存。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .config import (
    SCALE,
    DEPTH_EYE_HEIGHT,
    game_forward_to_obj,
    game_to_obj,
    obj_dist_to_game,
)


class MapGeometry:
    """单个地图的射线检测场景。"""

    def __init__(self, obj_path: Path):
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        if len(mesh.triangles) == 0:
            raise RuntimeError(f"Map OBJ 无效或为空: {obj_path}")

        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(t_mesh)
        self.name = obj_path.stem
        self._num_triangles = len(mesh.triangles)

    def cast_rays(
        self,
        origins: np.ndarray,      # [N, 3] OBJ空间坐标
        directions: np.ndarray,    # [N, 3] OBJ空间方向向量
    ) -> np.ndarray:
        """
        批量射线检测。

        Returns:
            distances: [N] 命中距离（OBJ 单位），未命中为 inf
        """
        rays = np.concatenate([origins.astype(np.float32),
                               directions.astype(np.float32)], axis=1)
        ans = self._scene.cast_rays(rays)
        return ans["t_hit"].numpy()  # [N], inf = miss

    def __repr__(self) -> str:
        return f"MapGeometry({self.name}, {self._num_triangles} tris)"


# ── 全局缓存 ──────────────────────────────────────────────────────────────────

_cache: Dict[str, MapGeometry] = {}


def get_map_geometry(map_name: str, maps_dir: Path) -> MapGeometry:
    """
    获取地图几何体（惰性加载 + 缓存）。

    Args:
        map_name: 地图名，如 "de_dust2"
        maps_dir: OBJ 文件目录
    """
    if map_name not in _cache:
        obj_path = maps_dir / f"{map_name}.obj"
        if not obj_path.exists():
            raise FileNotFoundError(f"地图 OBJ 不存在: {obj_path}")
        _cache[map_name] = MapGeometry(obj_path)
    return _cache[map_name]


def clear_cache():
    """清空缓存（释放内存）。"""
    _cache.clear()


def player_raycast_batch(
    map_geom: MapGeometry,
    origins_game: np.ndarray,     # [N, 3] 游戏空间坐标
    yaws_deg: np.ndarray,         # [N] 偏航角（度）
    pitches_deg: np.ndarray,      # [N] 俯仰角（度）
    max_dist: float = 5000.0,
) -> np.ndarray:
    """
    批量从玩家视角做射线检测。

    对每个 (origin, yaw, pitch) 发射一条沿前方向的射线。

    Args:
        origins_game: 玩家游戏坐标 [N, 3]
        yaws_deg: 偏航角 [N]
        pitches_deg: 俯仰角 [N]
        max_dist: 最大距离（游戏单位）

    Returns:
        distances: [N] 游戏单位距离，inf = 未命中
    """
    N = len(origins_game)
    assert len(yaws_deg) == N
    assert len(pitches_deg) == N

    # 转换 origin 到 OBJ 空间
    origins_obj = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        gx, gy, gz = origins_game[i]
        # 眼睛高度
        gz += DEPTH_EYE_HEIGHT
        ox, oy, oz = game_to_obj(gx, gy, gz)
        origins_obj[i] = (ox, oy, oz)

    # 转换 direction 到 OBJ 空间
    directions_obj = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        ox, oy, oz = game_forward_to_obj(yaws_deg[i], pitches_deg[i])
        directions_obj[i] = (ox, oy, oz)

    # 批量射线检测
    dists_obj = map_geom.cast_rays(origins_obj, directions_obj)

    # 转换回游戏单位，裁剪 max_dist
    dists_game = np.where(
        np.isfinite(dists_obj),
        obj_dist_to_game(dists_obj),
        np.inf,
    )
    dists_game = np.clip(dists_game, 0.0, max_dist)

    return dists_game
