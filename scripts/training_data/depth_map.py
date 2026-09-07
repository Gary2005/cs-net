"""
深度图生成 — 简化方向深度 + 完整深度缓冲。

简化深度：从玩家视角向 N 个方向发射射线，记录距离。
完整深度：渲染 64×48 深度缓冲（训练时实时生成，此处提供接口）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import (
    DEPTH_DIRECTIONS,
    DEPTH_MAX_DIST,
    DEPTH_N_DIRECTIONS,
    game_forward_to_obj,
    normalize_position,
)
from .map_loader import MapGeometry, player_raycast_batch


def compute_directional_depth(
    map_geom: MapGeometry,
    player_positions: np.ndarray,       # [T, 10, 3] 游戏坐标
    player_yaws: np.ndarray,            # [T, 10] 度
    player_pitches: np.ndarray,         # [T, 10] 度
    player_alive: np.ndarray,           # [T, 10] bool
    n_directions: int = DEPTH_N_DIRECTIONS,
    max_dist: float = DEPTH_MAX_DIST,
) -> np.ndarray:
    """
    为所有存活玩家生成简化方向深度。

    64 个方向在玩家局部坐标系中固定分布（刚性球体），
    随玩家的 yaw/pitch 整体旋转。dpitch=0 始终形成
    垂直于视线的赤道圆盘，而非等纬度偏移。

    Args:
        map_geom: 地图几何体
        player_positions: [T, 10, 3] 玩家游戏坐标
        player_yaws: [T, 10] 偏航角（度）
        player_pitches: [T, 10] 俯仰角（度）
        player_alive: [T, 10] 是否存活
        n_directions: 射线方向数
        max_dist: 最大距离

    Returns:
        depth: [T, 10, n_directions] 归一化距离值 [0, 1]
        mask: [T, 10] bool
    """
    T, N = player_alive.shape
    assert player_positions.shape == (T, N, 3)
    assert player_yaws.shape == (T, N)

    # ── 预计算 64 条射线在玩家局部坐标系中的 OBJ 方向 ──
    local_dirs = np.zeros((n_directions, 3), dtype=np.float32)
    for d, (dyaw, dpitch) in enumerate(DEPTH_DIRECTIONS[:n_directions]):
        ox, oy, oz = game_forward_to_obj(dyaw, dpitch)
        local_dirs[d] = (ox, oy, oz)

    depth = np.zeros((T, N, n_directions), dtype=np.float32)
    mask = np.zeros((T, N), dtype=bool)

    # 找出所有存活的 (tick, player) 对
    alive_indices = np.argwhere(player_alive)  # [K, 2]
    K = len(alive_indices)
    if K == 0:
        return depth, mask

    # 所有存活玩家的射线一次性拼接，合并成单次批量 raycast
    # （每玩家 n_directions 条射线，射线方向随 yaw/pitch 旋转）
    all_origins = np.empty((K * n_directions, 3), dtype=np.float32)
    all_yaws = np.empty((K * n_directions,), dtype=np.float32)
    all_pitches = np.empty((K * n_directions,), dtype=np.float32)

    for k, (ti, pi) in enumerate(alive_indices):
        pos = player_positions[ti, pi]
        yaw = float(player_yaws[ti, pi])
        pitch = float(player_pitches[ti, pi])

        # ── 构建旋转矩阵 R = R_y(yaw) @ R_x(-pitch) ──
        # 将玩家局部坐标系的方向旋转到世界 OBJ 空间
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)

        # R_y(yaw)
        Ry = np.array([
            [cos_y, 0, sin_y],
            [0,     1, 0],
            [-sin_y, 0, cos_y],
        ], dtype=np.float32)
        # R_x(-pitch): game +pitch = UP → Three.js +Y
        Rx = np.array([
            [1, 0,      0],
            [0, cos_p,  sin_p],
            [0, -sin_p, cos_p],
        ], dtype=np.float32)

        R = Ry @ Rx  # [3, 3]

        # 旋转所有局部方向 → 世界 OBJ 方向
        world_dirs = local_dirs @ R.T  # [D, 3]

        # ── 从世界方向反推绝对 yaw/pitch ──
        norms = np.linalg.norm(world_dirs, axis=1, keepdims=True)
        world_dirs = world_dirs / np.maximum(norms, 1e-8)
        # OBJ: ox=sin(yaw)*cos(pitch), oy=sin(pitch), oz=cos(yaw)*cos(pitch)
        ray_pitches = np.degrees(np.arcsin(np.clip(world_dirs[:, 1], -1.0, 1.0)))
        ray_yaws = np.degrees(np.arctan2(world_dirs[:, 0], world_dirs[:, 2])) % 360.0

        # 为该玩家构建 n_directions 条射线（所有射线从同一原点发出）
        idx = k * n_directions
        all_origins[idx:idx + n_directions] = pos
        all_yaws[idx:idx + n_directions] = ray_yaws
        all_pitches[idx:idx + n_directions] = ray_pitches

    # 单次批量射线检测（Open3D 内部并行，比逐玩家调用快得多）
    dists_all = player_raycast_batch(
        map_geom, all_origins, all_yaws, all_pitches, max_dist
    )

    # log 归一化（扩展近处，压缩远处，与人眼感知一致）
    for k, (ti, pi) in enumerate(alive_indices):
        idx = k * n_directions
        dists = dists_all[idx:idx + n_directions]
        depth[ti, pi, :] = np.log(dists + 1.0) / math.log(max_dist + 1.0)
        mask[ti, pi] = True

    return depth, mask


def compute_full_depth_buffer(
    map_geom: MapGeometry,
    player_position: np.ndarray,    # [3] 游戏坐标
    player_yaw: float,              # 度
    player_pitch: float,            # 度
    resolution: int = 64,
    hfov: float = 90.0,
    vfov: float = 60.0,
    max_dist: float = DEPTH_MAX_DIST,
) -> np.ndarray:
    """
    单玩家完整深度缓冲（64×48 或自定义分辨率）。

    用于训练时实时生成，不建议存入数据集（数据量太大）。

    Args:
        map_geom: 地图几何体
        player_position: [3] 游戏坐标
        player_yaw: 偏航角（度）
        player_pitch: 俯仰角（度）
        resolution: 正方形分辨率（宽=高）
        hfov: 水平视场角
        vfov: 垂直视场角

    Returns:
        depth_buffer: [resolution, resolution] 归一化深度
    """
    import numpy as np

    # 计算分辨率（宽稍大于高以适应 hfov/vfov 比例）
    v_res = resolution
    h_res = int(resolution * hfov / vfov)

    # 生成像素方向
    xs = np.linspace(-1.0, 1.0, h_res)
    ys = np.linspace(-1.0, 1.0, v_res)
    xv, yv = np.meshgrid(xs, ys)

    # 水平角：hfov/2 * pixel_x，垂直角：vfov/2 * pixel_y
    h_angles = xv * (hfov / 2.0)  # [v_res, h_res]
    v_angles = yv * (vfov / 2.0)  # [v_res, h_res]

    # 展平
    h_flat = h_angles.ravel()
    v_flat = v_angles.ravel()
    n_rays = len(h_flat)

    # 每条射线的绝对偏航/俯仰
    ray_yaws = np.full(n_rays, player_yaw, dtype=np.float32) + h_flat.astype(np.float32)
    ray_pitches = np.full(n_rays, player_pitch, dtype=np.float32) + v_flat.astype(np.float32)

    origins = np.tile(player_position.reshape(1, 3), (n_rays, 1))

    dists = player_raycast_batch(
        map_geom, origins, ray_yaws, ray_pitches, max_dist,
    )

    depth_buffer = (np.log(dists + 1.0) / math.log(max_dist + 1.0)).reshape(v_res, h_res)
    return depth_buffer.astype(np.float32)
