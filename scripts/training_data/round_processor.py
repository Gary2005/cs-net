"""
回合处理器 — 一个 V2 回合 → 一个训练 sample。

组合 feature_builder + depth_map + label_builder。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import (
    N_PLAYERS,
    N_TOKENS,
    MAP_NAME_TO_IDX,
    DEPTH_N_DIRECTIONS,
    DEPTH_MAX_DIST,
)
from .feature_builder import (
    build_player_features,
    build_global_features,
    build_projectile_features,
)
from .label_builder import build_labels
from .depth_map import compute_directional_depth
from .map_loader import MapGeometry


def process_round(
    round_data: dict,
    map_geom: Optional[MapGeometry] = None,
    source_file: str = "",
    match_teams: Optional[list] = None,
    players_meta: Optional[list] = None,
    tick_interval: float = 0.25,
    compute_depth: bool = True,
    places: Optional[dict] = None,
) -> dict:
    """
    处理一个 V2 回合 → 一个训练 sample。

    如果提供了 map_geom 且 compute_depth=True，会生成简化方向深度。

    Args:
        round_data: V2 回合 dict
        map_geom: 地图几何体（None 则跳过深度图）
        source_file: 来源文件名
        match_teams: 比赛队伍名 [team1, team2]
        players_meta: 玩家元数据列表 [{steamid, name}, ...]
        tick_interval: tick 间隔（秒）
        compute_depth: 是否生成深度图
        places: V2 顶层 places 字典 {name: index}，用于确定包点标签

    Returns:
        sample: 包含所有特征张量和元数据的字典
    """
    T = len(round_data["ticks"])
    if T == 0:
        raise ValueError("空回合：ticks 列表为空")

    map_name = round_data.get("map", round_data.get("map_name", "unknown"))

    # ── 构建特征 ────────────────────────────────────────────────────────
    player_feats = build_player_features(round_data)
    global_feats = build_global_features(round_data)
    proj_feats = build_projectile_features(round_data)
    # ── 标签 ────────────────────────────────────────────────────────────
    labels = build_labels(round_data, places=places)

    # ── 深度图 ──────────────────────────────────────────────────────────
    if compute_depth and map_geom is not None:
        # 从原始数据提取原始游戏坐标（深度图需要真实坐标做射线检测）
        raw_positions = np.zeros((T, N_PLAYERS, 3), dtype=np.float32)
        player_yaws = np.zeros((T, N_PLAYERS), dtype=np.float32)
        player_pitches = np.zeros((T, N_PLAYERS), dtype=np.float32)
        player_alive_raw = np.zeros((T, N_PLAYERS), dtype=bool)

        for p_idx, pdata in enumerate(round_data["players"]):
            n_ticks = min(T, len(pdata.get("x", [])))
            for t in range(n_ticks):
                raw_positions[t, p_idx] = (
                    float(pdata["x"][t]),
                    float(pdata["y"][t]),
                    float(pdata["z"][t]),
                )
                player_yaws[t, p_idx] = float(pdata["yaw"][t])
                player_pitches[t, p_idx] = float(pdata["pitch"][t])
                player_alive_raw[t, p_idx] = bool(pdata["alive"][t])

        depth, depth_mask = compute_directional_depth(
            map_geom,
            raw_positions,      # 原始游戏坐标
            player_yaws,
            player_pitches,
            player_alive_raw,
        )
    else:
        depth = np.zeros((T, N_PLAYERS, DEPTH_N_DIRECTIONS), dtype=np.float32)
        depth_mask = np.zeros((T, N_PLAYERS), dtype=bool)

    # ── 组装 sample ────────────────────────────────────────────────────

    # 从 label 张量提取 bombsite / win_reason 字符串（供可视化用）
    bombsite_arr = labels.get("label_bombsite")
    win_reason_arr = labels.get("label_win_reason")
    bombsite_str = _bombsite_to_str(bombsite_arr)
    win_reason_str = _win_reason_to_str(win_reason_arr)

    source_match_id = Path(source_file).stem.replace(".json", "")

    sample_key = (
        f"{source_match_id}__round{round_data.get('id', '?')}"
        f"_{random.randint(0, int(1e9))}"
    )

    # 游戏时间线（每个 tick 的 round_seconds）
    round_seconds = np.array(round_data.get("round_seconds", []), dtype=np.float32)

    sample = {
        "__key__": sample_key,

        "round_seconds": round_seconds,

        # 玩家特征
        "player_pos": player_feats["player_pos"],
        "player_alive_mask": player_feats["player_alive_mask"],
        "player_state": player_feats["player_state"],
        "player_inv": player_feats["player_inv"],
        "player_inv_mask": player_feats["player_inv_mask"],
        "player_rel_f": player_feats["player_rel_f"],
        "player_rel_i": player_feats["player_rel_i"],
        "player_rel_mask": player_feats["player_rel_mask"],
        "player_sound": player_feats["player_sound"],

        # 深度图
        "player_depth": depth,
        "player_depth_mask": depth_mask,

        # 全局/Bomb
        "bomb_pos": global_feats["bomb_pos"],
        "bomb_state": global_feats["bomb_state"],
        "map_idx": global_feats["map_idx"],

        # 投掷物
        "proj_pos": proj_feats["proj_pos"],
        "proj_type": proj_feats["proj_type"],
        "proj_dur": proj_feats["proj_dur"],
        "proj_mask": proj_feats["proj_mask"],
        "proj_is_active": proj_feats["proj_is_active"],

        # Token 掩码
        # 标签
        "label_winrate": labels["label_winrate"],
        "label_nxt_kill": labels["label_nxt_kill"],
        "label_nxt_death": labels["label_nxt_death"],
        "label_alive_end": labels["label_alive_end"],
        "label_bombsite": labels["label_bombsite"],
        "label_win_reason": labels["label_win_reason"],
        "label_camera": labels["label_camera"],

        # 元数据
        "meta": {
            "format": "cs2.training.v5",  # v5: label_camera 位移分量使用世界对齐坐标系
            "source_file": source_file,
            "match_teams": match_teams or ["?", "?"],
            "map_name": map_name,
            "map_idx": MAP_NAME_TO_IDX.get(map_name, -1),
            "round_id": round_data.get("id", -1),
            "teams": round_data.get("teams", ["?"] * N_PLAYERS),
            "winner": round_data.get("winner", "?"),
            "end_reason": round_data.get("end_reason", "?"),
            "bombsite": bombsite_str,
            "win_reason": win_reason_str,
            "players": players_meta or [],
            "T": T,
            "tick_interval": tick_interval,
            "depth_config": {
                "n_directions": DEPTH_N_DIRECTIONS,
                "max_dist": DEPTH_MAX_DIST,
            },
        },
    }

    return sample


# ── 标签 → 字符串 转换工具 ─────────────────────────────────────────────────

def _bombsite_to_str(arr: "np.ndarray | None") -> str:
    """将 label_bombsite 张量转换为可读字符串。"""
    if arr is None:
        return "?"
    unique = set(int(v) for v in np.unique(arr))
    if 0 in unique:
        return "A"
    elif 1 in unique:
        return "B"
    return "?"


def _win_reason_to_str(arr: "np.ndarray | None") -> str:
    """将 label_win_reason 张量转换为可读字符串。"""
    if arr is None:
        return "?"
    val = int(arr[0]) if len(arr) > 0 else 5
    mapping = {
        0: "CT全灭",
        1: "T全灭",
        2: "炸弹爆炸",
        3: "炸弹拆除",
        4: "时间耗尽",
    }
    return mapping.get(val, "其他")
