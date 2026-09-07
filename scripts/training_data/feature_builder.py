"""
特征构建器 — 从 V2 回合数据构建所有特征数组。

构建：玩家位置、状态、背包、关系、声音特征，
      全局/Bomb 特征，投掷物特征，Token 掩码。
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import (
    N_PLAYERS,
    N_MAX_PROJECTILES,
    N_TOKENS,
    N_PLAYER_RELATIONS,
    N_PLAYER_STATE_FEATURES,
    N_RELATION_FEATURES,
    N_SOUND_FEATURES,
    N_BOMB_STATE_FEATURES,
    MAP_NAME_TO_IDX,
    MAP_CONFIG,
    PROJECTILE_TYPE_TO_IDX,
    VELOCITY_SCALE,
    log_norm_signed,
    normalize_position,
    weapon_name_to_idx,
)


def build_player_features(round_data: dict) -> Dict[str, np.ndarray]:
    """
    从 V2 回合数据构建所有特征数组。

    V2 回合格式（demo_parser V2）:
      - round_data["players"][p]["x"] 等是长度为 T 的列式数组
      - round_data["tick_interval"] 之类在顶层

    Returns 包含所有特征张量的字典。
    """
    players = round_data["players"]
    T = len(round_data["ticks"])
    map_name = round_data.get("map", round_data.get("map_name", "unknown"))

    # ── 初始化 ────────────────────────────────────────────────────────────
    pos = np.zeros((T, N_PLAYERS, 3), dtype=np.float32)
    alive_mask = np.zeros((T, N_PLAYERS), dtype=bool)

    state = np.zeros((T, N_PLAYERS, N_PLAYER_STATE_FEATURES), dtype=np.float32)

    inv = np.zeros((T, N_PLAYERS, 9), dtype=np.int32)
    inv_mask = np.zeros((T, N_PLAYERS, 9), dtype=bool)

    rel_f = np.zeros((T, N_PLAYERS, N_PLAYER_RELATIONS, N_RELATION_FEATURES), dtype=np.float32)
    rel_i = np.zeros((T, N_PLAYERS, N_PLAYER_RELATIONS), dtype=np.int32)
    rel_mask = np.zeros((T, N_PLAYERS, N_PLAYER_RELATIONS), dtype=bool)

    sound = np.zeros((T, N_PLAYERS, N_SOUND_FEATURES), dtype=np.float32)

    # ── 填充玩家特征 ──────────────────────────────────────────────────────
    for p_idx, pdata in enumerate(players):
        n_ticks = len(pdata["x"])
        # 如果有 tick 数不一致，取较小值（容错）
        actual_T = min(T, n_ticks)

        for t in range(actual_T):
            alive = bool(pdata["alive"][t])
            if not alive:
                continue

            # 位置 — 列式数组直接索引
            px = float(pdata["x"][t])
            py = float(pdata["y"][t])
            pz = float(pdata["z"][t])
            nx, ny, nz = normalize_position(px, py, pz, map_name)
            pos[t, p_idx] = (nx, ny, nz)
            alive_mask[t, p_idx] = True

            # 状态特征（14 个）
            hp = float(pdata["hp"][t])
            armor = int(pdata["armor"][t])
            helmet = bool(pdata["helmet"][t])
            defuser = bool(pdata["defuser"][t])
            flash_dur = float(pdata["flash"][t])
            flash_alpha = float(pdata["flash_alpha"][t])
            pitch = float(pdata["pitch"][t])
            yaw = float(pdata["yaw"][t])
            vx = float(pdata["vx"][t])
            vy = float(pdata["vy"][t])
            vz = float(pdata["vz"][t])

            # 速度分解：摄像机相对方向
            yaw_rad = math.radians(yaw)
            fwd_x = math.cos(yaw_rad)
            fwd_y = math.sin(yaw_rad)
            right_x = math.sin(yaw_rad)
            right_y = -math.cos(yaw_rad)

            # demoparser2 velocity 约为游戏单位的 ~33x，缩放后与游戏速度一致
            v_forward = (vx * fwd_x + vy * fwd_y) * VELOCITY_SCALE
            v_right = (vx * right_x + vy * right_y) * VELOCITY_SCALE
            v_vert = vz * VELOCITY_SCALE

            # 队伍（从 round_data["teams"] 获取）
            teams = round_data.get("teams", ["?"] * N_PLAYERS)
            is_ct = 1.0 if (p_idx < len(teams) and teams[p_idx] == "CT") else 0.0

            state[t, p_idx] = [
                hp / 100.0,
                armor / 100.0,
                float(helmet),
                float(defuser),
                flash_dur / 5.0,
                flash_alpha / 255.0,
                math.cos(math.radians(pitch)),
                math.sin(math.radians(pitch)),
                math.cos(yaw_rad),
                math.sin(yaw_rad),
                is_ct,
                log_norm_signed(v_forward, 500.0),
                log_norm_signed(v_right, 500.0),
                log_norm_signed(v_vert, 500.0),
            ]

            # 背包
            inventory = pdata["inventory"][t]
            if isinstance(inventory, list) and len(inventory) > 0:
                # V2 format: inventory 已经是 weapon_idx 的列表
                n_items = min(len(inventory), 9)
                for wi, wid in enumerate(inventory[:9]):
                    inv[t, p_idx, wi] = int(wid)
                    inv_mask[t, p_idx, wi] = True

            # 声音（bool：本 tick 是否开火/有脚步）
            shots = pdata.get("shots", [0] * n_ticks)[t]
            footsteps = pdata.get("footsteps", [0] * n_ticks)[t]
            sound[t, p_idx] = [
                float(float(shots) > 0),
                float(float(footsteps) > 0),
            ]

    # ── 填充玩家间关系 ────────────────────────────────────────────────────
    for p_idx in range(N_PLAYERS):
        # 找到其他 9 个玩家的索引
        other_indices = [j for j in range(N_PLAYERS) if j != p_idx]
        assert len(other_indices) == N_PLAYER_RELATIONS

        for rel_slot, j_idx in enumerate(other_indices):
            for t in range(T):
                if not alive_mask[t, p_idx] or not alive_mask[t, j_idx]:
                    continue

                pi_pos = players[p_idx]
                pj_pos = players[j_idx]
                if t >= len(pi_pos["x"]) or t >= len(pj_pos["x"]):
                    continue

                # 眼睛坐标差值（Z +64 眼睛高度）
                eye_z_i = float(pi_pos["z"][t]) + 64.0
                eye_z_j = float(pj_pos["z"][t]) + 64.0
                dx = float(pj_pos["x"][t]) - float(pi_pos["x"][t])
                dy = float(pj_pos["y"][t]) - float(pi_pos["y"][t])
                dz = eye_z_j - eye_z_i
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                dist_clipped = max(0.0, min(float(dist), 5000.0))
                dist_log = math.log(dist_clipped + 1.0) / math.log(5001.0)

                # 旋转到观测者眼睛坐标系（3D，含 pitch）
                my_yaw = float(pi_pos["yaw"][t])
                my_pitch = float(pi_pos["pitch"][t])
                yaw_rad = math.radians(my_yaw)
                pitch_rad = math.radians(my_pitch)
                cos_y = math.cos(yaw_rad)
                sin_y = math.sin(yaw_rad)
                cos_p = math.cos(pitch_rad)
                sin_p = math.sin(pitch_rad)

                # 摄像机基向量投影
                d_forward = dx * cos_p * cos_y + dy * cos_p * sin_y + dz * sin_p
                d_right   = dx * sin_y - dy * cos_y
                d_up      = -dx * cos_y * sin_p - dy * sin_y * sin_p + dz * cos_p

                teams = round_data.get("teams", ["?"] * N_PLAYERS)
                my_team = teams[p_idx] if p_idx < len(teams) else "?"
                their_team = teams[j_idx] if j_idx < len(teams) else "?"
                is_teammate = float(my_team == their_team)
                is_enemy = float(my_team != their_team)

                # spotted
                my_spotted = pi_pos.get("spotted", [[]] * T)[t]
                their_spotted = pj_pos.get("spotted", [[]] * T)[t]
                if isinstance(my_spotted, list):
                    spotted_by_me = float(j_idx in my_spotted)
                else:
                    spotted_by_me = 0.0
                if isinstance(their_spotted, list):
                    spotted_me = float(p_idx in their_spotted)
                else:
                    spotted_me = 0.0

                # 视线角差（复用摄像机相对距离）
                # d_theta_xy: 视线方向与目标方向在摄像机 forward-right 平面内的夹角
                xy_norm = math.hypot(d_forward, d_right)
                if xy_norm > 1e-6:
                    cos_theta_xy = d_forward / xy_norm
                    cos_theta_xy = max(-1.0, min(1.0, cos_theta_xy))
                    d_theta_xy = math.degrees(math.acos(cos_theta_xy))
                else:
                    d_theta_xy = 0.0

                # d_theta_z: 目标相对于摄像机 forward-right 平面的仰角
                if xy_norm > 1e-6:
                    d_theta_z = math.degrees(math.atan2(d_up, xy_norm))
                else:
                    d_theta_z = 90.0 if d_up > 0 else (-90.0 if d_up < 0 else 0.0)

                rel_f[t, p_idx, rel_slot] = [
                    log_norm_signed(d_forward),
                    log_norm_signed(d_right),
                    log_norm_signed(d_up),
                    dist_log,
                    is_teammate,
                    is_enemy,
                    spotted_by_me,
                    spotted_me,
                    math.cos(math.radians(d_theta_xy)),
                    math.sin(math.radians(d_theta_xy)),
                    math.cos(math.radians(d_theta_z)),
                    math.sin(math.radians(d_theta_z)),
                    float(bool(pj_pos.get("alive", [False] * T)[t])),
                    float(pj_pos.get("hp", [0] * T)[t]) / 100.0,
                ]
                rel_i[t, p_idx, rel_slot] = j_idx
                rel_mask[t, p_idx, rel_slot] = True

    return {
        "player_pos": pos,
        "player_alive_mask": alive_mask,
        "player_state": state,
        "player_inv": inv,
        "player_inv_mask": inv_mask,
        "player_rel_f": rel_f,
        "player_rel_i": rel_i,
        "player_rel_mask": rel_mask,
        "player_sound": sound,
    }


def build_global_features(round_data: dict) -> Dict[str, np.ndarray]:
    """构建全局/Bomb 特征（Token 10）。"""
    T = len(round_data["ticks"])
    map_name = round_data.get("map", round_data.get("map_name", "unknown"))
    map_idx = MAP_NAME_TO_IDX.get(map_name, 0)

    bomb_pos = np.zeros((T, 3), dtype=np.float32)
    bomb_state = np.zeros((T, N_BOMB_STATE_FEATURES), dtype=np.float32)
    map_idx_arr = np.full(T, map_idx, dtype=np.int32)

    round_seconds = round_data.get("round_seconds", [0.0] * T)
    bomb_planted_arr = round_data.get("bomb_planted", [False] * T)
    bomb_dropped_arr = round_data.get("bomb_dropped", [False] * T)
    bomb_positions = round_data.get("bomb_position", [None] * T)

    # 计算 bomb_planted_duration：从 planted 事件起算
    bomb_planted_time = round_data.get("bomb_planted_time")
    bomb_events = round_data.get("events", {}).get("bomb", [])

    for t in range(T):
        actual_t = min(t, len(round_seconds) - 1, len(bomb_planted_arr) - 1)

        # Bomb 位置
        bp = bomb_positions[actual_t] if actual_t < len(bomb_positions) else None
        if bp is not None and len(bp) == 3:
            nx, ny, nz = normalize_position(float(bp[0]), float(bp[1]), float(bp[2]), map_name)
            bomb_pos[t] = (nx, ny, nz)

        # Bomb 状态
        is_planted = float(bomb_planted_arr[actual_t] if actual_t < len(bomb_planted_arr) else False)
        is_dropped = float(bomb_dropped_arr[actual_t] if actual_t < len(bomb_dropped_arr) else False)

        # planted duration
        planted_dur = 0.0
        if is_planted and bomb_planted_time is not None:
            current_time = round_seconds[actual_t] if actual_t < len(round_seconds) else 0.0
            planted_dur = max(0.0, current_time - bomb_planted_time)

        round_time = round_seconds[actual_t] if actual_t < len(round_seconds) else 0.0

        bomb_state[t] = [
            round_time / 160.0,
            is_planted,
            is_dropped,
            planted_dur / 40.0,
        ]

    return {
        "bomb_pos": bomb_pos,
        "bomb_state": bomb_state,
        "map_idx": map_idx_arr,
    }


def _normalize_grenade_type(raw_type: str) -> str:
    """将 V2 类名（CHEGrenadeProjectile）标准化为短名（he/flashbang 等）。"""
    t = raw_type.lower()
    if "flashbang" in t:
        return "flashbang"
    if "hegrenade" in t:
        return "he"
    if "molotov" in t:
        return "molotov"
    if "incgrenade" in t or "incendiary" in t:
        return "molotov"
    if "smoke" in t:
        return "smoke"
    if "decoy" in t:
        return "decoy"
    # 已经是短名（smoke/inferno 云），原样返回
    return t


def _grenade_priority(grenade_type: str) -> int:
    """投掷物优先级（数值越大越优先保留）。

    活跃烟雾/火焰 > 飞行闪光 > 飞行手雷 > 飞行火 > 飞行烟 > 诱饵

    注意：活跃烟雾/火焰云在调用方硬编码 prio=5，
    此函数仅处理飞行中的投掷物类型。
    """
    t = _normalize_grenade_type(grenade_type)
    if t == "flashbang":
        return 4
    if t == "he":
        return 3
    if t == "molotov":
        return 2
    if t == "smoke":
        return 1
    if t == "decoy":
        return 0
    return 0


def build_projectile_features(round_data: dict) -> Dict[str, np.ndarray]:
    """构建投掷物特征（Token 11–26）。

    当活跃投掷物超过 N_MAX_PROJECTILES 个时，按优先级保留：
      活跃烟雾/火焰 > 飞行闪光 > 飞行HE > 飞行火 > 飞行烟 > 诱饵

    溢出时记录告警。
    """
    T = len(round_data["ticks"])
    map_name = round_data.get("map", round_data.get("map_name", "unknown"))

    proj_pos = np.zeros((T, N_MAX_PROJECTILES, 3), dtype=np.float32)
    proj_type = np.full((T, N_MAX_PROJECTILES), -1, dtype=np.int32)
    proj_dur = np.zeros((T, N_MAX_PROJECTILES), dtype=np.float32)
    proj_mask = np.zeros((T, N_MAX_PROJECTILES), dtype=bool)
    proj_is_active = np.zeros((T, N_MAX_PROJECTILES), dtype=np.int32)

    events = round_data.get("events", {})
    grenades = events.get("grenades", [])
    smokes = events.get("smokes", [])
    infernos = events.get("infernos", [])

    ticks = round_data["ticks"]
    round_seconds = round_data.get("round_seconds", [0.0] * T)

    overflow_count = 0

    for t in range(T):
        tick = ticks[t] if t < len(ticks) else 0
        current_time = round_seconds[t] if t < len(round_seconds) else 0.0

        # 收集本 tick 所有活跃投掷物 → (优先级, type_idx, pos, dur)
        candidates = []

        # 飞行手雷
        for g in grenades:
            if g.get("t") == tick:
                gtype_raw = str(g.get("ty", ""))
                gtype = _normalize_grenade_type(gtype_raw)
                candidates.append((
                    _grenade_priority(gtype),
                    PROJECTILE_TYPE_TO_IDX.get(gtype, -1),
                    (float(g["x"]), float(g["y"]), float(g["z"])),
                    0.0,    # 飞行道具无持续时间
                    False,  # 飞行道具
                ))

        # 活跃烟雾（te 为空时默认 18s，与 filter_data 一致）
        for s in smokes:
            ts = s.get("ts", 0)
            te = s.get("te")
            if te is None:
                te = ts + 18.0
            if ts <= current_time < te:
                rem = max(0.0, te - current_time)
                candidates.append((
                    5,  # 最高优先级
                    PROJECTILE_TYPE_TO_IDX.get("smoke", 0),
                    (float(s["x"]), float(s["y"]), float(s["z"])),
                    rem / 25.0,
                    True,  # 活跃烟雾
                ))

        # 活跃火焰（te 为空时默认 7s，与 filter_data 一致）
        for inf in infernos:
            ts = inf.get("ts", 0)
            te = inf.get("te")
            if te is None:
                te = ts + 7.0
            if ts <= current_time < te:
                rem = max(0.0, te - current_time)
                candidates.append((
                    5,  # 最高优先级
                    PROJECTILE_TYPE_TO_IDX.get("inferno", 1),
                    (float(inf["x"]), float(inf["y"]), float(inf["z"])),
                    rem / 25.0,
                    True,  # 活跃火焰
                ))

        # 按优先级降序排列，取前 N_MAX_PROJECTILES
        candidates.sort(key=lambda x: x[0], reverse=True)
        total = len(candidates)
        kept = min(total, N_MAX_PROJECTILES)

        if total > N_MAX_PROJECTILES:
            overflow_count += 1

        for slot in range(kept):
            _, ptype, (px, py, pz), dur, is_active = candidates[slot]
            nx, ny, nz = normalize_position(px, py, pz, map_name)
            proj_pos[t, slot] = (nx, ny, nz)
            proj_type[t, slot] = ptype
            proj_dur[t, slot] = dur
            proj_mask[t, slot] = True
            proj_is_active[t, slot] = 1 if is_active else 0

    # overflow_count 仅作统计，不再打印警告（常见于大量投掷物的回合）

    return {
        "proj_pos": proj_pos,
        "proj_type": proj_type,
        "proj_dur": proj_dur,
        "proj_mask": proj_mask,
        "proj_is_active": proj_is_active,
    }


# token_dead_mask 已删除，用 ~player_alive_mask 替代
