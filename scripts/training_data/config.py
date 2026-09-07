"""
CS2 训练数据管线 — 配置文件。

地图中心、坐标范围、武器/投掷物索引、常量。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

# ── 地图配置 ──────────────────────────────────────────────────────────────────

MapCenter = Tuple[float, float, float]

MAP_CONFIG: Dict[str, Dict] = {
    "de_mirage":  {"center": (-605.8900146484375, -866.8900146484375, -171.6199951171875)},
    "de_dust2":   {"center": (-199.0, 977.0, 32.220001220703125)},
    "de_inferno": {"center": (481.07000732421875, 1396.47998046875, 137.91000366210938)},
    "de_nuke":    {"center": (265.9599914550781, -772.5, -381.8999938964844)},
    "de_overpass":{"center": (-2027.3900146484375, -812.9000244140625, 324.95001220703125)},
    "de_ancient": {"center": (-435.5, -348.0, 43.650001525878906)},
    "de_anubis":  {"center": (-77.38999938964844, 618.9000244140625, -6.800000190734863)},
    "de_cache":   {"center": (724.157958984375, 394.74676513671875, 1757.4903564453125)},
}

MAP_RANGES = {
    "x": (-5000.0, 5000.0),
    "y": (-5000.0, 5000.0),
    "z": (-2000.0, 2000.0),
}

MAP_NAME_TO_IDX = {name: i for i, name in enumerate(MAP_CONFIG.keys())}
MAP_IDX_TO_NAME = {i: name for name, i in MAP_NAME_TO_IDX.items()}

# ── 坐标系转换（游戏 ↔ OBJ/Three.js）────────────────────────────────────────

SCALE = 0.0254  # 英寸/游戏单位
VELOCITY_SCALE = 1.0 / 32.0  # demoparser2 速度 → 游戏单位/秒（约 33x 缩放）

def game_to_obj(gx: float, gy: float, gz: float) -> Tuple[float, float, float]:
    """游戏坐标 (X东, Y北, Z上) → OBJ坐标 (Three.js Y-up)."""
    return (gy * SCALE, gz * SCALE, gx * SCALE)


def obj_to_game(ox: float, oy: float, oz: float) -> Tuple[float, float, float]:
    """OBJ坐标 → 游戏坐标."""
    inv = 1.0 / SCALE
    return (oz * inv, ox * inv, oy * inv)


def obj_dist_to_game(dist_obj: float) -> float:
    """OBJ空间距离 → 游戏单位."""
    return dist_obj / SCALE


def game_forward_to_obj(yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
    """
    玩家前方向量：游戏空间 → OBJ空间。

    游戏空间 forward: (cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch))
    OBJ空间: (game_y*SCALE, game_z*SCALE, game_x*SCALE)
    缩放系数约掉后归一化。
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    gx = math.cos(yaw) * cp
    gy = math.sin(yaw) * cp
    gz = math.sin(pitch)
    # 转 OBJ: (gy, gz, gx)，SCALE 在归一化中抵消
    ox, oy, oz = gy, gz, gx
    norm = math.sqrt(ox * ox + oy * oy + oz * oz)
    if norm < 1e-8:
        return (0.0, 0.0, 1.0)
    return (ox / norm, oy / norm, oz / norm)


# ── 归一化工具 ────────────────────────────────────────────────────────────────

def clip_and_scale(value: float, rng: Tuple[float, float]) -> float:
    """将值裁剪到范围并归一化到 [-1, 1]."""
    lo, hi = rng
    if value < lo:
        value = lo
    elif value > hi:
        value = hi
    return value / max(abs(lo), abs(hi))


def log_norm_signed(value: float, max_dist: float = 5000.0) -> float:
    """符号保留的对数归一化。

    将距离值压缩到 [-1, 1]，保留正负方向，对数刻度让近处更敏感。

    Args:
        value: 原始有符号距离（游戏单位）
        max_dist: 最大距离，超过此值按对数饱和处理
    """
    sign = 1.0 if value >= 0.0 else -1.0
    abs_clipped = min(abs(value), max_dist)
    return sign * math.log(abs_clipped + 1.0) / math.log(max_dist + 1.0)


def normalize_position(gx: float, gy: float, gz: float, map_name: str) -> Tuple[float, float, float]:
    """游戏坐标 → 归一化坐标（以地图中心为原点，按 MAP_RANGES 缩放）."""
    cx, cy, cz = MAP_CONFIG.get(map_name, {}).get("center", (0.0, 0.0, 0.0))
    return (
        clip_and_scale(gx - cx, MAP_RANGES["x"]),
        clip_and_scale(gy - cy, MAP_RANGES["y"]),
        clip_and_scale(gz - cz, MAP_RANGES["z"]),
    )


def denormalize_position(nx: float, ny: float, nz: float, map_name: str) -> Tuple[float, float, float]:
    """归一化坐标 → 游戏坐标（normalize_position 的逆运算）."""
    cx, cy, cz = MAP_CONFIG.get(map_name, {}).get("center", (0.0, 0.0, 0.0))
    return (
        nx * max(abs(MAP_RANGES["x"][0]), abs(MAP_RANGES["x"][1])) + cx,
        ny * max(abs(MAP_RANGES["y"][0]), abs(MAP_RANGES["y"][1])) + cy,
        nz * max(abs(MAP_RANGES["z"][0]), abs(MAP_RANGES["z"][1])) + cz,
    )


# ── 武器/投掷物 ───────────────────────────────────────────────────────────────

# 从旧项目的 tokenizer.yaml 提取（合并 demo_parser V2 的武器名）
WEAPON_NAMES = [
    "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18",
    "AK-47", "AUG", "AWP", "FAMAS", "Galil AR", "M4A4",
    "M4A1-S", "SG 553", "SCAR-20", "G3SG1", "SSG 08",
    "MAC-10", "MP9", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon",
    "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev",
    "P2000", "USP-S", "P250", "CZ75 Auto", "Tec-9", "R8 Revolver",
    "Zeus x27", "C4 Explosive", "Knife",
    "High Explosive Grenade", "Flashbang", "Smoke Grenade",
    "Molotov", "Incendiary Grenade", "Decoy Grenade",
]

WEAPON_TO_IDX = {name: i for i, name in enumerate(WEAPON_NAMES)}

# 投掷物类型（用于 projectile token）
PROJECTILE_TYPES = [
    "smoke", "inferno", "he", "flashbang", "decoy", "molotov",
]
PROJECTILE_TYPE_TO_IDX = {name: i for i, name in enumerate(PROJECTILE_TYPES)}


def weapon_name_to_idx(name: str) -> int:
    """武器名 → 索引。未知武器返回 Knife 的索引。"""
    return WEAPON_TO_IDX.get(name, WEAPON_TO_IDX.get("Knife", 0))


# ── 深度图配置 ────────────────────────────────────────────────────────────────

DEPTH_N_DIRECTIONS = 64       # 简化方向深度：射线数量
DEPTH_MAX_DIST = 5000.0       # 最远射线距离（游戏单位）
DEPTH_EYE_HEIGHT = 64.0       # 玩家眼睛高度（游戏单位，Z 轴偏移）

# 64 个方向的定义 (yaw_offset, pitch_offset) — 5 层同心圈
#   中心层（0°）: 24 条，每 15° 一条
#   ±30° 层   : 各 12 条，每 30° 一条
#   ±60° 层   : 各  8 条，每 45° 一条
DEPTH_DIRECTIONS = [
    # === +60° 层  8 条 ===
    (0.0,   60.0),
    (45.0,  60.0),
    (90.0,  60.0),
    (135.0, 60.0),
    (180.0, 60.0),
    (225.0, 60.0),
    (270.0, 60.0),
    (315.0, 60.0),
    # === +30° 层 12 条 ===
    (0.0,   30.0),
    (30.0,  30.0),
    (60.0,  30.0),
    (90.0,  30.0),
    (120.0, 30.0),
    (150.0, 30.0),
    (180.0, 30.0),
    (210.0, 30.0),
    (240.0, 30.0),
    (270.0, 30.0),
    (300.0, 30.0),
    (330.0, 30.0),
    # === 0° 中心层 24 条 ===
    (0.0,    0.0),
    (15.0,   0.0),
    (30.0,   0.0),
    (45.0,   0.0),
    (60.0,   0.0),
    (75.0,   0.0),
    (90.0,   0.0),
    (105.0,  0.0),
    (120.0,  0.0),
    (135.0,  0.0),
    (150.0,  0.0),
    (165.0,  0.0),
    (180.0,  0.0),
    (195.0,  0.0),
    (210.0,  0.0),
    (225.0,  0.0),
    (240.0,  0.0),
    (255.0,  0.0),
    (270.0,  0.0),
    (285.0,  0.0),
    (300.0,  0.0),
    (315.0,  0.0),
    (330.0,  0.0),
    (345.0,  0.0),
    # === -30° 层 12 条 ===
    (0.0,   -30.0),
    (30.0,  -30.0),
    (60.0,  -30.0),
    (90.0,  -30.0),
    (120.0, -30.0),
    (150.0, -30.0),
    (180.0, -30.0),
    (210.0, -30.0),
    (240.0, -30.0),
    (270.0, -30.0),
    (300.0, -30.0),
    (330.0, -30.0),
    # === -60° 层  8 条 ===
    (0.0,   -60.0),
    (45.0,  -60.0),
    (90.0,  -60.0),
    (135.0, -60.0),
    (180.0, -60.0),
    (225.0, -60.0),
    (270.0, -60.0),
    (315.0, -60.0),
]

# ── 特征维度常量 ──────────────────────────────────────────────────────────────

N_PLAYERS = 10
N_MAX_PROJECTILES = 16
N_TOKENS = N_PLAYERS + 1 + N_MAX_PROJECTILES  # 10 + 1 + 16 = 27
N_PLAYER_RELATIONS = 9           # 每个玩家最多 9 个关系
N_PLAYER_STATE_FEATURES = 14     # 玩家状态特征数
N_RELATION_FEATURES = 14         # 关系特征数（dx,dy,dz,log_dist,teammate,enemy,spotted_me,spotted_by_me,cos/sin(d_theta_xy),cos/sin(d_theta_z),j_alive,j_hp）
N_SOUND_FEATURES = 2             # 声音特征数
N_BOMB_STATE_FEATURES = 4        # 炸弹状态特征数
N_PROJ_FEATURES = 3              # 投掷物位置特征数（pos only, type & dur separate）

# ── 武器/子弹配置 (来自旧 tokenizer.yaml) ────────────────────────────────────
# 这些用于模型侧的 embedding 初始化，训练数据管线中用不到但保留以供参考

N_WEAPONS = len(WEAPON_NAMES)
N_PROJECTILE_TYPES = len(PROJECTILE_TYPES)
N_MAPS = len(MAP_CONFIG)

# ── 旧 tokenizer.yaml 路径（仅用于兼容参考）──────────────────────────────────
OLD_TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "cs-net" / "demoparser_utils" / "tokenizer.yaml"
