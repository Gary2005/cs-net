"""
预训练窗口提取器 — 从 round-level WDS 样本提取固定长度滑动窗口。

每个窗口: N_TICKS 输入 tick + (N_TICKS-1) 输出 tick = 2*N_TICKS-1 总 tick。
输入特征右对齐（pad 在开头），输出标签左对齐（pad 在末尾）。

用法:
    from scripts.training_data.pretrain_processor import PretrainWindowExtractor

    extractor = PretrainWindowExtractor(n_ticks=64, stride=16)
    for round_sample in decode_round_samples(...):
        for window in extractor.extract_windows(round_sample):
            # window 是完整的 pretrain sample dict，T=64
            ...
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set

import numpy as np

# ── 常量 ──────────────────────────────────────────────────────────────────────────

DEFAULT_N_TICKS = 64
DEFAULT_STRIDE = 16
DEFAULT_MIN_INPUT_TICKS = 32
DEFAULT_MIN_OUTPUT_TICKS = 1

# 所有已知的 numpy key（与 wds_writer.py 保持同步）
NUMPY_KEYS: Set[str] = {
    "round_seconds",
    "player_pos", "player_alive_mask",
    "player_state",
    "player_inv", "player_inv_mask",
    "player_rel_f", "player_rel_i", "player_rel_mask",
    "player_sound",
    "player_depth", "player_depth_mask",
    "player_depth_labels", "player_alive_mask_labels",  # decoder per-step depth conditioning
    "player_pos_labels", "player_angle_labels",        # decoder per-step xyz & angle conditioning
    "bomb_pos",
    "bomb_state",
    "map_idx",
    "proj_pos",
    "proj_type", "proj_dur", "proj_mask", "proj_is_active",
    "label_winrate",
    "label_nxt_kill", "label_nxt_death",
    "label_alive_end",
    "label_bombsite", "label_win_reason",
    "label_camera",
}

# 在 padding 位置应填 True 的特殊 mask（表示"无效/忽略"）
_PAD_TRUE_MASKS: Set[str] = set()

# int32 类型的 key（非 mask）
_INT_KEYS: Set[str] = {
    "player_inv", "player_rel_i",
    "proj_type", "proj_is_active", "map_idx",
    "label_nxt_kill", "label_nxt_death",
    "label_bombsite", "label_win_reason",
}

# int key 在 padding 位置的默认值
_INT_PAD_VALUES: Dict[str, int] = {
    "label_nxt_kill": 10,
    "label_nxt_death": 10,
    "label_bombsite": 2,
    "label_win_reason": 5,
    "proj_type": -1,
}

# 哪些 key 末尾带 _mask（用于判断 bool 类型）
_BOOL_SUFFIX = "_mask"

# label key 集合（需要从输入窗口切片保留的）
_LABEL_KEYS: Set[str] = {
    "label_winrate", "label_nxt_kill", "label_nxt_death",
    "label_alive_end", "label_bombsite", "label_win_reason",
}

# 回合级常量 label（对所有 tick 值相同），slice 时直接复制
_ROUND_CONSTANT_LABELS: Set[str] = {
    "label_winrate", "label_alive_end", "label_bombsite", "label_win_reason",
}


class PretrainWindowExtractor:
    """
    从 round-level 样本提取固定长度预训练窗口。

    Args:
        n_ticks:        输入/输出窗口 tick 数（默认 64）
        stride:         滑动步长（默认 16，75% 重叠）
        min_input_ticks: 最少有效输入 tick 数
        min_output_ticks: 最少有效输出 tick 数
        require_camera: 如果 True，缺少 camera label 的样本将报错而非跳过
    """

    def __init__(
        self,
        n_ticks: int = DEFAULT_N_TICKS,
        stride: int = DEFAULT_STRIDE,
        min_input_ticks: int = DEFAULT_MIN_INPUT_TICKS,
        min_output_ticks: int = DEFAULT_MIN_OUTPUT_TICKS,
        require_camera: bool = False,
        jitter: bool = True,
    ):
        self.n_ticks = n_ticks
        self.stride = stride
        self.min_input_ticks = min_input_ticks
        self.min_output_ticks = min_output_ticks
        self.require_camera = require_camera
        self.jitter = jitter
        self.total_ticks = n_ticks * 2 - 1  # 127 (64 input + 63 output ticks)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def extract_windows(self, sample: dict) -> List[dict]:
        """
        从一个 round sample 提取所有有效预训练窗口。

        Args:
            sample: 解码后的 round-level sample dict

        Returns:
            windows: 每个元素是一个 T=64 的 pretrain sample dict
        """
        # 检查 camera label 是否存在
        if "label_camera" not in sample:
            msg = (
                f"Sample '{sample.get('__key__', '?')}' missing camera labels. "
                f"Re-run create_training_data.py with updated code to generate them."
            )
            if self.require_camera:
                raise ValueError(msg)
            warnings.warn(msg)
            return []

        T = sample["player_pos"].shape[0]
        if T < self.min_input_ticks:
            return []  # 回合太短

        meta = sample.get("meta", {})
        round_seconds = sample.get("round_seconds", None)
        tick_interval = meta.get("tick_interval", 0.25)

        # 滑动窗口起点范围: s >= 0（不 left-pad，不做假数据）
        start_min = 0
        start_max = T - self.n_ticks

        # 随机起始偏移 + 每步抖动（避免窗口位置的规律性）
        rng = np.random.RandomState()
        jitter_start = rng.randint(0, self.stride + 1) if self.jitter else 0
        jitter_step = max(1, self.stride // 2)

        windows = []
        s = start_min + jitter_start
        while s <= start_max:
            # 输入窗口在 round 中的范围
            input_real_start = max(0, s)
            input_real_end = min(T, s + self.n_ticks)
            n_valid_input = input_real_end - input_real_start

            # 输出窗口在 round 中的范围
            output_real_start = max(0, s + self.n_ticks)
            output_real_end = min(T, s + self.total_ticks)
            n_valid_output = output_real_end - output_real_start

            if n_valid_input < self.min_input_ticks:
                s += self.stride
                continue
            if n_valid_output < self.min_output_ticks:
                s += self.stride
                continue

            window = self._build_window(
                sample, s, T,
                input_real_start, input_real_end,
                output_real_start, output_real_end,
                round_seconds, tick_interval, meta,
            )
            windows.append(window)

            # 下一步：基础 stride + 随机抖动
            jitter = rng.randint(-jitter_step, jitter_step + 1) if self.jitter else 0
            s += self.stride + jitter

        return windows

    # ── 窗口构建 ──────────────────────────────────────────────────────────────

    def _build_window(
        self,
        sample: dict,
        s: int,
        T: int,
        input_start: int,
        input_end: int,
        output_start: int,
        output_end: int,
        round_seconds: Optional[np.ndarray],
        tick_interval: float,
        meta: dict,
    ) -> dict:
        """构建一个预训练窗口 sample（s>=0，无左侧 padding）。"""
        n = self.n_ticks
        # s >= 0 保证 pad_before 恒为 0（无假数据）
        pad_after_input = max(0, s + n - T)
        pad_after_output = max(0, s + self.total_ticks - T)

        window: dict = {}

        # ── 所有 numpy 特征 key：切片 + padding ──────────────────────────
        for key in sorted(NUMPY_KEYS):
            if key not in sample:
                continue

            arr = sample[key]

            if key == "label_camera":
                # 127 个 label：condition t∈[s,s+63] 需要的最大 label 是 label[s+126]
                label_end = min(output_end, input_start + self.total_ticks)
                window[key] = self._slice_array(
                    arr, input_start, label_end,
                    pad_before=0,
                    pad_after=pad_after_output,
                    total_length=self.total_ticks,
                )
            elif key == "player_depth":
                # 原有：input 窗口 (64 ticks)
                window[key] = self._slice_array(
                    arr, input_start, input_end,
                    pad_before=0,
                    pad_after=pad_after_input,
                    total_length=n,
                )
                # 新增：完整 label 范围 (127 ticks)，用于 decoder per-step depth conditioning
                label_end = min(output_end, input_start + self.total_ticks)
                window["player_depth_labels"] = self._slice_array(
                    arr, input_start, label_end,
                    pad_before=0,
                    pad_after=pad_after_output,
                    total_length=self.total_ticks,
                )
            elif key == "player_alive_mask":
                # 原有：input 窗口 (64 ticks)
                window[key] = self._slice_array(
                    arr, input_start, input_end,
                    pad_before=0,
                    pad_after=pad_after_input,
                    total_length=n,
                )
                # 修正：alive_mask 语义是 True=存活，padding 应为 False（死亡/未知）
                valid_input = input_end - input_start
                window[key][valid_input:] = False

                # 新增：完整 label 范围 (127 ticks)，用于判断 depth 有效性
                label_end = min(output_end, input_start + self.total_ticks)
                window["player_alive_mask_labels"] = self._slice_array(
                    arr, input_start, label_end,
                    pad_before=0,
                    pad_after=pad_after_output,
                    total_length=self.total_ticks,
                )
                valid_label = label_end - input_start
                window["player_alive_mask_labels"][valid_label:] = False
            elif key == "player_pos":
                # 原有：input 窗口 (64 ticks)
                window[key] = self._slice_array(
                    arr, input_start, input_end,
                    pad_before=0,
                    pad_after=pad_after_input,
                    total_length=n,
                )
                # 新增：完整 label 范围 (127 ticks)，用于 decoder per-step xyz conditioning
                label_end = min(output_end, input_start + self.total_ticks)
                window["player_pos_labels"] = self._slice_array(
                    arr, input_start, label_end,
                    pad_before=0,
                    pad_after=pad_after_output,
                    total_length=self.total_ticks,
                )
            elif key == "player_state":
                # 原有：input 窗口 (64 ticks)
                window[key] = self._slice_array(
                    arr, input_start, input_end,
                    pad_before=0,
                    pad_after=pad_after_input,
                    total_length=n,
                )
                # 新增：角度标签 (127 ticks, 4 dims: cos(yaw), sin(yaw), cos(pitch), sin(pitch))
                label_end = min(output_end, input_start + self.total_ticks)
                state_label_slice = arr[input_start:label_end]  # [L, 10, 14]
                angle_slice = state_label_slice[..., [8, 9, 6, 7]]  # reorder: yaw first
                window["player_angle_labels"] = self._slice_array(
                    angle_slice, 0, label_end - input_start,
                    pad_before=0,
                    pad_after=pad_after_output,
                    total_length=self.total_ticks,
                )
            elif key == "player_depth_labels" or key == "player_alive_mask_labels":
                # 这些 key 在上面 player_depth/player_alive_mask 分支中已处理，跳过
                continue
            elif key == "player_pos_labels" or key == "player_angle_labels":
                # 这些 key 在上面 player_pos/player_state 分支中已处理，跳过
                continue
            elif key in _LABEL_KEYS:
                if key in _ROUND_CONSTANT_LABELS:
                    # 回合常量：直接复制到输入窗口长度
                    ref_val = arr[0] if arr.ndim == 1 else arr[0]
                    out = np.zeros_like(arr, shape=(n,) + arr.shape[1:])
                    out[:] = ref_val if np.isscalar(ref_val) or arr.ndim == 1 else ref_val
                    window[key] = out
                else:
                    # 逐 tick 标签：从输入窗口切片
                    window[key] = self._slice_array(
                        arr, input_start, input_end,
                        pad_before=0,
                        pad_after=pad_after_input,
                        total_length=n,
                    )
            else:
                # 特征张量：从输入窗口切片（左对齐，无左侧 pad）
                window[key] = self._slice_array(
                    arr, input_start, input_end,
                    pad_before=0,
                    pad_after=pad_after_input,
                    total_length=n,
                )

        # ── 坐标系转换：v4 相机坐标系 → v5 世界对齐 ─────────────────
        # 仅当 round 数据为 v4 或更早（相机坐标）时才转换。
        # v5+ round 数据的 label_camera 已经是世界对齐，跳过。
        source_format = (sample.get("meta") or {}).get("format", "")
        needs_conversion = not source_format.startswith("cs2.training.v5") and \
                           not source_format.startswith("cs2.training.v6") and \
                           not source_format.startswith("cs2.training.v7") and \
                           not source_format.startswith("cs2.training.v8") and \
                           not source_format.startswith("cs2.training.v9")
        if needs_conversion and "label_camera" in window and "player_state" in sample:
            label_len = self.total_ticks
            label_s = input_start
            label_e = min(output_end, input_start + self.total_ticks)
            real_label_ticks = label_e - label_s

            ps = sample["player_state"]  # [T_round, 10, 14]
            pitch_cos = ps[label_s:label_e, :, 6].copy()   # cos(pitch), [L, 10]
            pitch_sin = ps[label_s:label_e, :, 7].copy()   # sin(pitch), [L, 10]

            # 对 padding 位置填充 pitch=0（cos=1, sin=0），不做旋转
            if pad_after_output > 0:
                pad_cos = np.ones((pad_after_output, 10), dtype=np.float32)
                pad_sin = np.zeros((pad_after_output, 10), dtype=np.float32)
                pitch_cos = np.concatenate([pitch_cos, pad_cos], axis=0)
                pitch_sin = np.concatenate([pitch_sin, pad_sin], axis=0)

            d_fwd = window["label_camera"][..., 0].copy()
            d_up_old = window["label_camera"][..., 2].copy()

            # 绕 pitch 反向旋转：(d_forward, d_up) → 世界对齐
            window["label_camera"][..., 0] = d_fwd * pitch_cos - d_up_old * pitch_sin
            window["label_camera"][..., 2] = d_fwd * pitch_sin + d_up_old * pitch_cos
            # index 1 (d_right) 不变——right 向量 (sin_y, -cos_y, 0) 天然水平

        # ── 辅助 mask ───────────────────────────────────────────────────
        output_mask = np.zeros(n, dtype=bool)
        real_output_len = output_end - output_start
        output_mask[:real_output_len] = True
        window["output_mask"] = output_mask

        # ── 下游任务标签：每玩家在每个窗口 tick 之后是否还能获得击杀 ──────
        # last_kill[p] = 该玩家本回合最后一次击杀的 round tick；-1 = 无击杀
        # label_future_kill[i, p] = last_kill[p] > (s + i)（窗口内 tick i 之后还能击杀？）
        # 下游 future_kill 任务在 cond 时刻（窗口起点 t）锚定：取 label_future_kill[t]，
        # 即"cond 之后任意时刻（含最后一个 tick 之后，哨兵 T）获得击杀=1"——所有预测点同一标签。
        last_kill = self._last_kill_ticks(sample)          # [10]
        fkill = np.zeros((self.total_ticks, 10), dtype=bool)
        for i in range(self.total_ticks):
            fkill[i] = last_kill > (s + i)
        window["label_future_kill"] = fkill

        # ── 玩家阵营（0=CT, 1=T, -1=未知）──
        # 供下游 CT 胜率聚合指标使用（按阵营分组存活玩家的 winrate 预测）。
        # 转成数值数组 key 而非放在 meta：player shuffle 时随玩家维度同步。
        teams_list = (meta.get("teams") or []) if isinstance(meta, dict) else []
        window["player_teams"] = np.array(
            [0 if t == "CT" else (1 if t == "T" else -1) for t in teams_list],
            dtype=np.int64,
        )

        # ── tick 时间 ──────────────────────────────────────────────────
        window["tick_times_input"] = self._build_tick_times(
            round_seconds, input_start, input_end,
            0, pad_after_input, n, tick_interval,
        )
        window["tick_times_output"] = self._build_tick_times(
            round_seconds, output_start, output_end,
            pad_before=0, pad_after=pad_after_output,
            total_length=n, tick_interval=tick_interval,
        )

        # ── 元数据 ─────────────────────────────────────────────────────
        window["__key__"] = (
            f"{sample.get('__key__', 'unknown')}_s{s}"
        )
        window["meta"] = {
            **meta,
            "format": "cs2.training.pretrain.v5",  # v5: label_camera d_up 改为世界坐标
            "source_sample_key": sample.get("__key__", ""),
            "window_start": s,
            "n_valid_input": input_end - input_start,
            "n_valid_output": output_end - output_start,
            "n_ticks_config": n,
            "stride": self.stride,
            "round_T": T,
        }

        return window

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _last_kill_ticks(self, sample: dict) -> np.ndarray:
        """从 label_nxt_death 重建击杀序列，返回每玩家最后击杀 tick [10]。

        用 label_nxt_death 的"变化点"作为击杀事件：每个 victim 变化 = 一次击杀
        （连杀时 killer 不变但 victim 变，nxt_kill 无法区分，nxt_death 可以；
        victim 交叉验证：每个变化点的 victim 在之后确实死亡）。
        击杀者 = 变化点 tick **之前**的 label_nxt_kill 值（nk[t-1]，因为 nk[t]
        已经指向下一次击杀）。-1 = 该玩家本回合无击杀。
        """
        last = np.full(10, -1, dtype=np.int64)
        nk = sample.get("label_nxt_kill")
        nd = sample.get("label_nxt_death")
        if nk is None or nd is None or len(nk) == 0:
            return last
        T = len(nk)

        def _record(t: int, victim: int):
            # victim 在变化点 t 被击杀，击杀者是变化前的 nxt_kill
            killer = int(nk[t - 1])
            if 0 <= killer < 10:
                last[killer] = t

        prev_d = -1
        for t in range(T):
            cur_d = int(nd[t])
            if cur_d != prev_d:
                if prev_d != -1 and prev_d != 10:
                    _record(t, prev_d)
                prev_d = cur_d
        # 末尾残留：nxt_death 从头到尾保持同一个 victim 且无变化点，
        # 说明该击杀发生在最后一个记录 tick 之后（kt > T-1，超出回合数据）。
        # 记录为哨兵 tick T：保证在 t=T-1 判断"之后还能击杀"仍为 True
        # （满足"包含最后一个 tick 之后"的语义）。
        if prev_d != -1 and prev_d != 10:
            _record(T, prev_d)
        return last

    def _slice_array(
        self,
        arr: np.ndarray,
        real_start: int,
        real_end: int,
        pad_before: int,
        pad_after: int,
        total_length: int,
    ) -> np.ndarray:
        """
        从 arr 切片 [real_start:real_end]（沿 dim=0），并前后 pad 到 total_length。

        Args:
            arr:          原始数组，shape[0] = T
            real_start:   切片起点
            real_end:     切片终点
            pad_before:   开头 pad 数量
            pad_after:    末尾 pad 数量
            total_length: 目标长度

        Returns:
            长度 = total_length 的数组
        """
        is_bool = arr.dtype == bool or (arr.dtype == np.dtype("bool"))
        is_int = arr.dtype in (np.int32, np.int64, np.dtype("int32"), np.dtype("int64"))

        # 确定 pad 值
        if is_bool:
            pad_val = True   # bool mask: pad with True (ignore)
        elif is_int:
            pad_val = 0  # 默认，调用方可通过 key 判断覆盖
        else:
            pad_val = 0.0

        # 切片
        real_slice = arr[real_start:real_end]

        # 构建 padded 数组
        out_shape = (total_length,) + arr.shape[1:]
        if is_bool:
            out = np.zeros(out_shape, dtype=bool)
            if pad_val:  # True pad for special masks
                out[:] = pad_val
        elif is_int:
            out = np.zeros(out_shape, dtype=arr.dtype)
        else:
            out = np.zeros(out_shape, dtype=arr.dtype)

        # 放置真实数据（右对齐：真实数据从 pad_before 位置开始）
        out[pad_before:pad_before + (real_end - real_start)] = real_slice

        return out

    def _build_tick_times(
        self,
        round_seconds: Optional[np.ndarray],
        real_start: int,
        real_end: int,
        pad_before: int,
        pad_after: int,
        total_length: int,
        tick_interval: float,
    ) -> np.ndarray:
        """构建 tick_times 数组，填充 pad 位置用外推值。"""
        times = np.zeros(total_length, dtype=np.float32)

        if round_seconds is not None and len(round_seconds) > 0:
            real_len = real_end - real_start
            if real_len > 0:
                times[pad_before:pad_before + real_len] = round_seconds[real_start:real_end]

            # 对 padded 位置做简单外推（保持时间连续性，方便 time encoding）
            if pad_before > 0 and real_len > 0:
                # 开头 pad: 从第一个真实时间往前推
                first_time = round_seconds[real_start]
                for i in range(pad_before - 1, -1, -1):
                    first_time -= tick_interval
                    times[i] = max(0.0, first_time)

            if pad_after > 0 and real_len > 0:
                # 末尾 pad: 从最后一个真实时间往后推
                last_time = round_seconds[real_end - 1]
                start_idx = pad_before + real_len
                for i in range(start_idx, total_length):
                    last_time += tick_interval
                    times[i] = last_time

        return times


def extract_windows_from_sample(
    sample: dict,
    n_ticks: int = DEFAULT_N_TICKS,
    stride: int = DEFAULT_STRIDE,
    **kwargs,
) -> List[dict]:
    """
    便捷函数：从一个 round sample 提取所有预训练窗口。

    等价于 PretrainWindowExtractor(n_ticks, stride, **kwargs).extract_windows(sample)
    """
    extractor = PretrainWindowExtractor(
        n_ticks=n_ticks,
        stride=stride,
        **kwargs,
    )
    return extractor.extract_windows(sample)


# ── 玩家编号随机化 ──────────────────────────────────────────────────────────────

# 所有 player 维度（dim=1 或 dim=2 取决于 key）需要 shuffle 的 key 列表
_PLAYER_DIM1_KEYS = {
    "player_pos", "player_state", "player_inv", "player_inv_mask",
    "player_alive_mask", "player_alive_mask_labels",
    "player_depth", "player_depth_labels",
    "player_pos_labels", "player_angle_labels",
    "label_alive_end", "label_future_kill",   # per-player 标签，需随玩家 shuffle 同步
}
# 形状 [10] 的一维 per-player key（如 player_teams），也需随玩家 shuffle 同步
_PLAYER_DIM0_KEYS = {
    "player_teams",
}
_PLAYER_DIM2_KEYS = {
    "player_rel_f", "player_rel_i", "player_rel_mask",
}
_PLAYER_LABEL_KEYS = {
    "label_camera",
}


def shuffle_players(window: dict, rng: Optional[np.random.RandomState] = None) -> dict:
    """
    对单个窗口的玩家维度做随机 permutation。

    修改所有 player 相关的 feature tensor 和 label tensor，
    同时重映射 player_rel_i 中的索引。
    """
    if rng is None:
        rng = np.random.RandomState()

    perm = rng.permutation(10)
    inv_perm = np.argsort(perm)

    for key in _PLAYER_DIM1_KEYS:
        if key in window and window[key].ndim >= 2 and window[key].shape[1] == 10:
            window[key] = window[key][:, perm, ...]

    for key in _PLAYER_DIM2_KEYS:
        if key in window and window[key].ndim >= 3 and window[key].shape[1] == 10:
            window[key] = window[key][:, perm, ...]
            if key == "player_rel_i":
                # 重映射关系中的玩家索引
                window[key] = inv_perm[window[key].clip(0, 9)]

    for key in _PLAYER_DIM0_KEYS:
        if key in window and window[key].ndim == 1 and window[key].shape[0] == 10:
            window[key] = window[key][perm]

    for key in _PLAYER_LABEL_KEYS:
        if key in window and window[key].ndim >= 2 and window[key].shape[1] == 10:
            window[key] = window[key][:, perm, ...]

    return window
