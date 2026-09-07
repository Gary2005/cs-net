"""
标签构建器 — 从 V2 回合数据生成标签。

任务：
  1. winrate     — 回合获胜方（0=CT, 1=T）
  2. nxt_kill    — 下一个获得击杀的人（int 0-9，10=无）
  3. nxt_death   — 下一个死亡的人（int 0-9，10=无）
  4. alive_end   — 回合结束时是否存活（per-player binary）
  5. bombsite    — 炸弹安放包点（0=A, 1=B, 2=未知/未安放）
  6. win_reason  — 获胜原因（0=CT全灭, 1=T全灭, 2=炸弹爆炸, 3=炸弹拆除, 4=时间耗尽, 5=其他）
  7. camera      — 每个 tick 相对上一 tick 的相机运动（预训练标签）
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

# end_reason → 获胜原因标签映射
WIN_REASON_MAP = {
    "ct_killed":      0,   # CT 全灭（T 获胜）
    "t_killed":       1,   # T 全灭（CT 获胜）
    "bomb_exploded":  2,   # 炸弹爆炸（T 获胜）
    "bomb_defused":   3,   # 炸弹拆除（CT 获胜）
    "time_ran_out":   4,   # 时间耗尽
}


def _normalize_angle_rad(diff_rad: float) -> float:
    """将角度差（弧度）归一化到 [-pi, pi]."""
    return (diff_rad + math.pi) % (2.0 * math.pi) - math.pi


def build_camera_labels(round_data: dict) -> Dict[str, np.ndarray]:
    """
    为每个 tick 计算相机相对运动标签（预训练用）。

    对于每个玩家，label_camera[t] 预测 tick t → t+1 的相机运动。
    位移分量使用世界对齐坐标系（v5）：
      - d_forward, d_right 在水平面上，仅由 yaw 决定，不受 pitch 影响
      - d_up = 纯世界 Z 轴位移（不受视角影响）
    is_alive[t] = tick t+1 是否存活。

    标签向量 10 维：
      (d_forward, d_right, d_up, cos(d_pitch), sin(d_pitch), cos(d_yaw), sin(d_yaw),
       is_alive, is_firing, end)

    end = 1 表示该 label 有效（对应 tick 有真实数据，非 padding）。
    end = 0 只出现在 padding 位置或末 tick（T-1 无 t+1）。

    Loss 计算顺序：先 BCE(end)，end=1 则再 BCE(is_alive)，is_alive=1 则再 MSE(相机运动)。
    """
    T = len(round_data["ticks"])
    N = 10

    label_camera = np.zeros((T, N, 10), dtype=np.float32)

    players = round_data["players"]

    for p_idx, pdata in enumerate(players):
        n_ticks = len(pdata["x"])
        actual_T = min(T, n_ticks)

        for t in range(actual_T - 1):  # t → t+1 的预测
            alive_t = bool(pdata["alive"][t])
            if not alive_t:
                continue  # t 死亡，无法计算 t → t+1 的运动

            alive_next = bool(pdata["alive"][t + 1])
            # end=1: 有真实的 t+1 数据（padding 位置 end=0）
            is_end = 1.0
            is_alive = 1.0 if alive_next else 0.0

            if alive_next:
                # 双方存活 → 计算完整相机运动
                dx = float(pdata["x"][t + 1]) - float(pdata["x"][t])
                dy = float(pdata["y"][t + 1]) - float(pdata["y"][t])
                dz = float(pdata["z"][t + 1]) - float(pdata["z"][t])

                if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
                    continue

                yaw_t = float(pdata["yaw"][t])
                pitch_t = float(pdata["pitch"][t])
                if not (math.isfinite(yaw_t) and math.isfinite(pitch_t)):
                    continue

                yaw_rad = math.radians(yaw_t)
                cos_y = math.cos(yaw_rad); sin_y = math.sin(yaw_rad)

                # 世界对齐位移：forward/right 在水平面上，up = 纯世界 Z
                d_forward = dx * cos_y + dy * sin_y
                d_right   = dx * sin_y - dy * cos_y
                d_up      = dz

                yaw_next = float(pdata["yaw"][t + 1])
                pitch_next = float(pdata["pitch"][t + 1])
                if not (math.isfinite(yaw_next) and math.isfinite(pitch_next)):
                    continue

                d_pitch_rad = math.radians(pitch_next) - math.radians(pitch_t)
                d_yaw_rad = _normalize_angle_rad(math.radians(yaw_next) - math.radians(yaw_t))

                n_shots = pdata.get("shots", [0] * n_ticks)
                is_firing = 1.0 if (t + 1 < len(n_shots) and float(n_shots[t + 1]) > 0) else 0.0

                label_camera[t, p_idx] = [
                    d_forward, d_right, d_up,
                    math.cos(d_pitch_rad), math.sin(d_pitch_rad),
                    math.cos(d_yaw_rad), math.sin(d_yaw_rad),
                    is_alive, is_firing, is_end,
                ]
            else:
                # t 存活但 t+1 死亡：只标记 end + is_alive=0，不计算运动
                label_camera[t, p_idx] = [
                    0, 0, 0, 1, 0, 1, 0,   # 运动全零，角度变化清零
                    is_alive, 0, is_end,     # is_alive=0, is_firing=0, end=1
                ]

    return {"label_camera": label_camera}


def build_labels(round_data: dict, places: dict | None = None) -> Dict[str, np.ndarray]:
    """
    为回合中每个 tick 计算标签。

    Args:
        round_data: V2 回合数据
        places: V2 顶层 places 字典 {name: index}，用于确定包点

    Returns:
        label_winrate:   [T]    float32  — 0=CT获胜, 1=T获胜
        label_nxt_kill:  [T]    int32    — 下一个击杀者 0-9，10=无
        label_nxt_death: [T]    int32    — 下一个死亡者 0-9，10=无
        label_alive_end: [T,10] float32  — 1.0=回合结束时存活
        label_bombsite:  [T]    int32    — 0=A, 1=B, 2=未知/未安放
        label_win_reason:[T]    int32    — 获胜原因索引
    """
    T = len(round_data["ticks"])
    N = 10

    kills = round_data.get("events", {}).get("kills", [])
    teams = round_data.get("teams", ["?"] * N)
    winner = round_data.get("winner", "")

    # ── 1. Winrate：回合层面 ─────────────────────────────────────────────────
    # 0 = CT wins, 1 = T wins
    winrate = np.full(T, 1 if winner == "T" else 0, dtype=np.float32)

    # ── 2. nxt_kill / nxt_death：每 tick 的下一个事件 ────────────────────────
    nxt_kill = np.full(T, 10, dtype=np.int32)
    nxt_death = np.full(T, 10, dtype=np.int32)

    ticks_arr = round_data["ticks"]
    t_min = ticks_arr[0]

    # 过滤：跳过 t_min 之前的 kill（跨回合污染），保留 t_max 之后的 kill
    kills_filtered = [
        k for k in kills
        if k["t"] > t_min
    ]
    kill_ticks = sorted([k["t"] for k in kills_filtered])
    kill_by_tick = {k["t"]: k for k in kills_filtered}

    for t_idx in range(T):
        current_tick = ticks_arr[t_idx]

        for kt in kill_ticks:
            if kt <= current_tick:
                continue

            k = kill_by_tick[kt]
            a = k.get("a", -1)
            v = k.get("v", -1)

            if 0 <= a < N and 0 <= v < N:
                nxt_kill[t_idx] = a
                nxt_death[t_idx] = v
            break  # 只取第一个未来 kill

    # ── 3. Alive at round end ────────────────────────────────────────────────
    alive_end = np.zeros((T, N), dtype=np.float32)
    died = set()
    for k in kills_filtered:
        v = k.get("v", -1)
        if 0 <= v < N:
            died.add(v)

    for p in range(N):
        alive_end[:, p] = 0.0 if p in died else 1.0

    # ── 4. Bombsite：炸弹安放在哪个包点 ─────────────────────────────────────
    # 0=A, 1=B, 2=未知/未安放
    label_bombsite = _build_bombsite_label(round_data, places, T, ticks_arr)

    # ── 5. Win reason：获胜原因 ──────────────────────────────────────────────
    end_reason = round_data.get("end_reason", "")
    win_reason_val = WIN_REASON_MAP.get(end_reason, 5)  # 5 = 其他/未知
    label_win_reason = np.full(T, win_reason_val, dtype=np.int32)

    labels = {
        "label_winrate": winrate,
        "label_nxt_kill": nxt_kill,
        "label_nxt_death": nxt_death,
        "label_alive_end": alive_end,
        "label_bombsite": label_bombsite,
        "label_win_reason": label_win_reason,
    }

    # ── 7. Camera：预训练相机运动标签 ───────────────────────────────────────
    camera_labels = build_camera_labels(round_data)
    labels.update(camera_labels)

    return labels


def _build_bombsite_label(
    round_data: dict,
    places: dict | None,
    T: int,
    ticks_arr: np.ndarray,
) -> np.ndarray:
    """
    确定本回合炸弹安放的包点（最终结果）。

    通过炸弹安放事件，查找安放时 T 方存活玩家所在的 place 名称，
    判断是 BombsiteA 还是 BombsiteB。
    标签为回合级常量：即使安放前的 tick 也填入最终安放的包点，
    与 label_winrate 的语义一致（让模型学会预测）。

    Returns:
        [T] int32 — 0=A, 1=B, 2=未知/未安放
    """
    label = np.full(T, 2, dtype=np.int32)  # 默认：未知/未安放

    if places is None:
        return label

    # 构建 place_name → index 的查找（用于快速判断）
    place_reverse = {idx: name for name, idx in places.items()}

    # 找到炸弹安放事件
    bomb_events = round_data.get("events", {}).get("bomb", [])
    plant_event = None
    for be in bomb_events:
        if be.get("e") == "planted":
            plant_event = be
            break

    if plant_event is None:
        return label  # 没有安放炸弹，全部为 2

    plant_tick = plant_event["t"]

    # 找到安放 tick 在 ticks_arr 中的索引
    import bisect
    idx = bisect.bisect_left(ticks_arr, plant_tick)
    # 取最近的 tick
    if idx == 0:
        plant_idx = 0
    elif idx >= len(ticks_arr):
        plant_idx = len(ticks_arr) - 1
    else:
        plant_idx = idx if abs(ticks_arr[idx] - plant_tick) < abs(ticks_arr[idx - 1] - plant_tick) else idx - 1

    # 查看安放时 T 方存活玩家的 place，判断是哪个包点
    teams = round_data.get("teams", [])
    players = round_data.get("players", [])

    for p_idx, pdata in enumerate(players):
        if p_idx >= len(teams):
            break
        if teams[p_idx] != "T":
            continue

        # 检查该玩家在安放时是否存活
        alive_arr = pdata.get("alive", [])
        if plant_idx >= len(alive_arr) or not alive_arr[plant_idx]:
            continue

        # 获取该玩家在安放时的 place
        place_arr = pdata.get("place", [])
        if plant_idx >= len(place_arr):
            continue
        place_val = place_arr[plant_idx]
        if place_val is None:
            continue

        place_name = place_reverse.get(place_val, "")
        if "BombsiteA" in place_name or "A Site" in place_name:
            label[:] = 0   # 整回合都是 A
            return label
        elif "BombsiteB" in place_name or "B Site" in place_name:
            label[:] = 1   # 整回合都是 B
            return label

    # fallback：没通过 place 找到，返回未知
    return label
