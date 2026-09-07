"""
CS2 预训练模型 — 纯离散 token 预测版。

架构：
  Token Embedder → Spatial Transformer → Temporal Transformer → Token Decoder → CE Loss

每个时刻的相机运动用 7 个离散 token 表示，一次性预测 n_future_ticks 个时刻。
Token 序列: [continue][d_forward][d_right][d_up][d_pitch][d_yaw][fire]
Ground truth 使用残差修正防止离散化误差累积。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training_data.config import N_MAPS, N_WEAPONS, N_PROJECTILE_TYPES, N_PLAYERS, N_MAX_PROJECTILES, N_TOKENS

# torch.compile 下把 keep_ratio 抽样放到 eager 域执行：
# torch.randperm 是随机 op，若进入 dynamo 编译图，speculation 重启分析时会因
# 随机结果不同而走不同路径，报 "SpeculationLog diverged" AssertionError。
try:
    from torch._dynamo import disable as _torch_dynamo_disable
except ImportError:  # 极老 torch 兜底：无 dynamo 时原样返回
    def _torch_dynamo_disable(fn):
        return fn


# ═══════════════════════════════════════════════════════════════════════════════════
# Camera Tokenizer — 连续相机运动 ↔ 离散 token 序列
# ═══════════════════════════════════════════════════════════════════════════════════

class CameraTokenizer:
    """
    将连续相机运动标签转换为离散 token 序列，支持残差修正。

    每个 tick 的 7 个 token 布局（单 token 覆盖完整有符号范围）:
      [0] continue   (1=继续, 0=停止)
      [1] d_forward  (signed [-128,128] → tok MOVE_OFFSET + bin)
      [2] d_right    (signed [-128,128] → tok MOVE_OFFSET + bin)
      [3] d_up       (signed [-128,128] → tok MOVE_OFFSET + bin)
      [4] d_pitch    (signed [-90,90] → tok PITCH_OFFSET + bin)
      [5] d_yaw      (signed [-180,180] → tok YAW_OFFSET + bin)
      [6] fire       (binary: 0/1 → tok FIRE_0/FIRE_1)

    特殊 token: PAD (id=0), 用于 continue=0（死亡/结束）之后的填充。
    """

    def __init__(self, move_range: int = 128, move_grid_size: float = 1.0,
                 angle_grid_size: float = 1.0, use_residual_correction: bool = True):
        self.move_range = move_range
        self.move_grid_size = move_grid_size
        self.angle_grid_size = angle_grid_size
        self.use_residual_correction = use_residual_correction

        # Signed full-range bin counts
        self.n_move_values = int(2 * move_range / move_grid_size) + 1   # covers [-128, 128]
        self.n_pitch_values = int(2 * 90 / angle_grid_size) + 1          # covers [-90, 90]
        self.n_yaw_values = int(2 * 180 / angle_grid_size) + 1           # covers [-180, 180]

        # Half-ranges for encode/decode shift
        self._half_move = float(move_range)   # 128.0
        self._half_pitch = 90.0
        self._half_yaw = 180.0

        # ── Token ID 偏移 ──
        self.PAD = 0
        self.CONTINUE_0 = 1
        self.CONTINUE_1 = 2
        self.MOVE_OFFSET = 3
        self.PITCH_OFFSET = self.MOVE_OFFSET + self.n_move_values
        self.YAW_OFFSET = self.PITCH_OFFSET + self.n_pitch_values
        self.FIRE_0 = self.YAW_OFFSET + self.n_yaw_values
        self.FIRE_1 = self.FIRE_0 + 1
        self.vocab_size = self.FIRE_1 + 1

        # 每个 tick: depth + xyz_abs + angle_abs + 7 camera tokens = 10 positions per group
        self.TOKENS_PER_TICK = 7
        self.TOKENS_PER_GROUP = 10
        self.FIRE_TOKEN_INDEX = 6  # fire token 在 tick 内的位置（0-indexed）

    def _encode_move(self, val: float) -> int:
        """移动距离（有符号 [-move_range, move_range]）→ token id。"""
        clamped = max(-self.move_range, min(val, self.move_range))
        bin_idx = int(round((clamped + self._half_move) / self.move_grid_size))
        bin_idx = max(0, min(bin_idx, self.n_move_values - 1))
        return self.MOVE_OFFSET + bin_idx

    def _encode_d_pitch(self, val_deg: float) -> int:
        """d_pitch（有符号 [-90, 90]）→ token id。"""
        clamped = max(-90.0, min(val_deg, 90.0))
        bin_idx = int(round((clamped + self._half_pitch) / self.angle_grid_size))
        bin_idx = max(0, min(bin_idx, self.n_pitch_values - 1))
        return self.PITCH_OFFSET + bin_idx

    def _encode_d_yaw(self, val_deg: float) -> int:
        """d_yaw（有符号 [-180, 180]）→ token id。"""
        clamped = max(-180.0, min(val_deg, 180.0))
        bin_idx = int(round((clamped + self._half_yaw) / self.angle_grid_size))
        bin_idx = max(0, min(bin_idx, self.n_yaw_values - 1))
        return self.YAW_OFFSET + bin_idx

    def _decode_move(self, tok: int) -> float:
        """move token → 移动距离值（有符号）。"""
        bin_idx = tok - self.MOVE_OFFSET
        if 0 <= bin_idx < self.n_move_values:
            return bin_idx * self.move_grid_size - self._half_move
        return 0.0

    def _decode_angle(self, tok: int, offset: int, n_values: int, half_range: float) -> float:
        """角度 token → 角度值（度，有符号）。"""
        bin_idx = tok - offset
        if 0 <= bin_idx < n_values:
            return bin_idx * self.angle_grid_size - half_range
        return 0.0

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def encode_tick(self, label_10d, accumulated: list) -> list[int]:
        """
        将一个 tick 的 10D 连续标签编码为 7 个 token（带残差修正）。

        Args:
            label_10d: [10] tensor (d_fwd,d_right,d_up, cos/sin pitch, cos/sin yaw, alive, fire, end)
            accumulated: list of 5 accumulated decoded values [fwd, right, up, pitch_deg, yaw_deg]

        Returns:
            list of 7 int token ids
        """
        import math as _math

        d_fwd = float(label_10d[0])
        d_right = float(label_10d[1])
        d_up = float(label_10d[2])
        # cos/sin pitch → delta pitch in degrees
        dp_rad = _math.atan2(float(label_10d[4]), float(label_10d[3]))
        dp_deg = dp_rad * (180.0 / _math.pi)
        # cos/sin yaw → delta yaw in degrees
        dy_rad = _math.atan2(float(label_10d[6]), float(label_10d[5]))
        dy_deg = dy_rad * (180.0 / _math.pi)

        is_alive = float(label_10d[7])
        is_fire = float(label_10d[8])
        is_end = float(label_10d[9])

        # ── 残差修正: 用实际累计位移加上已累积残差 ──
        residual_fwd = d_fwd + accumulated[0]
        residual_right = d_right + accumulated[1]
        residual_up = d_up + accumulated[2]
        residual_pitch = dp_deg + accumulated[3]
        residual_yaw = dy_deg + accumulated[4]

        # 单 token 编码（完整有符号范围）
        tok_fwd = self._encode_move(residual_fwd)
        tok_right = self._encode_move(residual_right)
        tok_up = self._encode_move(residual_up)
        tok_pitch = self._encode_d_pitch(residual_pitch)
        tok_yaw = self._encode_d_yaw(residual_yaw)

        # 更新 accumulated: 原始值 - 解码值 = 量化误差
        decoded_fwd = self._decode_move(tok_fwd)
        decoded_right = self._decode_move(tok_right)
        decoded_up = self._decode_move(tok_up)
        decoded_pitch = self._decode_angle(tok_pitch, self.PITCH_OFFSET, self.n_pitch_values, self._half_pitch)
        decoded_yaw = self._decode_angle(tok_yaw, self.YAW_OFFSET, self.n_yaw_values, self._half_yaw)

        accumulated[0] = accumulated[0] + d_fwd - decoded_fwd
        accumulated[1] = accumulated[1] + d_right - decoded_right
        accumulated[2] = accumulated[2] + d_up - decoded_up
        accumulated[3] = accumulated[3] + dp_deg - decoded_pitch
        accumulated[4] = accumulated[4] + dy_deg - decoded_yaw

        continue_value = 1.0 if (is_alive > 0.5 and is_end > 0.5) else 0.0
        tokens = [
            self.CONTINUE_1 if continue_value > 0.5 else self.CONTINUE_0,
            tok_fwd,
            tok_right,
            tok_up,
            tok_pitch,
            tok_yaw,
            self.FIRE_1 if is_fire > 0.5 else self.FIRE_0,
        ]
        return tokens

    # ── 向量化编码辅助方法（GPU 上并行处理 N 个样本）──────────────────────────

    def _vec_encode_move(self, val: torch.Tensor) -> torch.Tensor:
        """对 [*] tensor 做 signed move 量化编码，返回 long tensor。"""
        clamped = torch.clamp(val, min=-self.move_range, max=self.move_range)
        bin_idx = torch.round((clamped + self._half_move) / self.move_grid_size).long()
        bin_idx = torch.clamp(bin_idx, min=0, max=self.n_move_values - 1)
        return bin_idx + self.MOVE_OFFSET

    def _vec_encode_pitch(self, val_deg: torch.Tensor) -> torch.Tensor:
        """对 [*] tensor 做 signed pitch 量化编码，返回 long tensor。"""
        clamped = torch.clamp(val_deg, min=-90.0, max=90.0)
        bin_idx = torch.round((clamped + self._half_pitch) / self.angle_grid_size).long()
        bin_idx = torch.clamp(bin_idx, min=0, max=self.n_pitch_values - 1)
        return bin_idx + self.PITCH_OFFSET

    def _vec_encode_yaw(self, val_deg: torch.Tensor) -> torch.Tensor:
        """对 [*] tensor 做 signed yaw 量化编码，返回 long tensor。"""
        clamped = torch.clamp(val_deg, min=-180.0, max=180.0)
        bin_idx = torch.round((clamped + self._half_yaw) / self.angle_grid_size).long()
        bin_idx = torch.clamp(bin_idx, min=0, max=self.n_yaw_values - 1)
        return bin_idx + self.YAW_OFFSET

    def _vec_decode_move(self, tok: torch.Tensor) -> torch.Tensor:
        """对 [*] long tensor 解码 move → float 有符号位移值。"""
        bin_idx = (tok - self.MOVE_OFFSET).clamp(0, self.n_move_values - 1)
        return bin_idx.float() * self.move_grid_size - self._half_move

    def _vec_decode_angle(self, tok: torch.Tensor, offset: int, n_values: int, half_range: float) -> torch.Tensor:
        """对 [*] long tensor 解码角度 → float 有符号角度值。"""
        bin_idx = (tok - offset).clamp(0, n_values - 1)
        return bin_idx.float() * self.angle_grid_size - half_range

    # ── 主编码方法 ────────────────────────────────────────────────────────────

    def encode_sequence(self, labels, n_future_ticks: int) -> torch.Tensor:
        """
        将一个窗口的连续标签编码为带残差修正的 token 序列（GPU 向量化版）。

        优化策略:
          1. atan2/角度转换一次性 GPU 向量化
          2. continue / fire 一次性向量化写入（不依赖残差修正）
          3. cumprod 预计算 per-tick active mask，避免逐 tick bookkeeping
          4. 循环仅处理连续值 tokens（残差修正依赖项）
          5. 全程零 GPU↔CPU 同步

        Args:
            labels: [N, n_future_ticks, 10]  连续相机运动标签
            n_future_ticks: 预测的未来 tick 数

        Returns:
            [N, seq_len] token ids, seq_len = 7 * n_future_ticks
        """
        import math as _math
        N = labels.shape[0]
        device = labels.device
        seq_len = self.TOKENS_PER_TICK * n_future_ticks
        tpt = self.TOKENS_PER_TICK  # 7

        # ── 第1步: GPU 向量化预计算所有 tick 的原始值 ──
        # 索引: 0=d_fwd, 1=d_right, 2=d_up, 3/4=cos/sin pitch, 5/6=cos/sin yaw, 7=alive, 8=fire, 9=end
        d_fwd   = labels[..., 0]                                                   # [N, ticks]
        d_right = labels[..., 1]
        d_up    = labels[..., 2]
        dp_deg  = torch.atan2(labels[..., 4], labels[..., 3]) * (180.0 / _math.pi) # [N, ticks]
        dy_deg  = torch.atan2(labels[..., 6], labels[..., 5]) * (180.0 / _math.pi) # [N, ticks]
        is_alive_v = labels[..., 7]                                                # [N, ticks]
        is_fire_v  = labels[..., 8]
        is_end_v   = labels[..., 9]

        # ── 第2步: 初始化 & 常量 ──
        all_tokens = torch.full((N, seq_len), self.PAD, dtype=torch.long, device=device)
        accum = torch.zeros(N, 5, device=device)  # [N, 5]: fwd, right, up, pitch, yaw

        _PAD = torch.tensor(self.PAD, dtype=torch.long, device=device)
        _c1 = torch.tensor(self.CONTINUE_1, dtype=torch.long, device=device)
        _c0 = torch.tensor(self.CONTINUE_0, dtype=torch.long, device=device)
        _f1  = torch.tensor(self.FIRE_1, dtype=torch.long, device=device)
        _f0  = torch.tensor(self.FIRE_0, dtype=torch.long, device=device)

        # ── 第3步: 一次性写入 continue / fire（不依赖残差修正）──
        tick_offsets = torch.arange(n_future_ticks, device=device) * tpt  # [ticks]

        # continue[t] = alive[t] & end[t]，任一为 0 都表示该 player 序列停止。
        continue_v = (is_alive_v > 0.5) & (is_end_v > 0.5)   # [N, ticks]

        # cum_ok[t] = ∏ continue_v[:t]：tick t 之前全部 continue
        cum_ok = torch.cat([
            torch.ones(N, 1, dtype=torch.bool, device=device),
            torch.cumprod(continue_v.long(), dim=1)[:, :-1].bool(),
        ], dim=1)  # [N, ticks]

        # continue token 在停止 tick 写 0，之后写 PAD。
        continue_tok = torch.where(continue_v, _c1, _c0)       # [N, ticks]
        all_tokens[:, tick_offsets] = torch.where(cum_ok, continue_tok, _PAD)

        # 只有 continue 且之前全部 continue 的 tick 才写 fire / 连续值。
        active_mask = cum_ok & continue_v                        # [N, ticks]

        # fire at pos FIRE_TOKEN_INDEX (=6), 13, 20, ...  — 仅 active_mask 写入
        fire_tok = torch.where(is_fire_v > 0.5, _f1, _f0)        # [N, ticks]
        all_tokens[:, tick_offsets + self.FIRE_TOKEN_INDEX] = torch.where(active_mask, fire_tok, _PAD)

        # ── 第4步: 连续值 tokens（positions 1-5: d_forward, d_right, d_up, d_pitch, d_yaw）──
        N_VALS = 5  # fwd, right, up, pitch, yaw — each a single signed token
        if self.use_residual_correction:
            # 逐 tick 残差修正：编码 → 解码 → 累积残差 → 下一 tick
            for tick in range(n_future_ticks):
                active = active_mask[:, tick]  # [N]
                if not active.any():
                    break
                offset = tick * tpt

                cur_fwd   = d_fwd[:, tick]
                cur_right = d_right[:, tick]
                cur_up    = d_up[:, tick]
                cur_pitch = dp_deg[:, tick]
                cur_yaw   = dy_deg[:, tick]

                res_fwd   = cur_fwd   + accum[:, 0]
                res_right = cur_right + accum[:, 1]
                res_up    = cur_up    + accum[:, 2]
                res_pitch = cur_pitch + accum[:, 3]
                res_yaw   = cur_yaw   + accum[:, 4]

                # 单 token 编码（完整有符号范围）
                t_fwd   = self._vec_encode_move(res_fwd)
                t_right = self._vec_encode_move(res_right)
                t_up    = self._vec_encode_move(res_up)
                t_pitch = self._vec_encode_pitch(res_pitch)
                t_yaw   = self._vec_encode_yaw(res_yaw)

                # ── 解码 → 更新 accumulated（仅 active 样本）──
                d_fwd_dec   = self._vec_decode_move(t_fwd)
                d_right_dec = self._vec_decode_move(t_right)
                d_up_dec    = self._vec_decode_move(t_up)
                d_pitch_dec = self._vec_decode_angle(t_pitch, self.PITCH_OFFSET, self.n_pitch_values, self._half_pitch)
                d_yaw_dec   = self._vec_decode_angle(t_yaw,   self.YAW_OFFSET,   self.n_yaw_values,   self._half_yaw)

                accum[:, 0] = torch.where(active, accum[:, 0] + cur_fwd   - d_fwd_dec,   accum[:, 0])
                accum[:, 1] = torch.where(active, accum[:, 1] + cur_right - d_right_dec, accum[:, 1])
                accum[:, 2] = torch.where(active, accum[:, 2] + cur_up    - d_up_dec,    accum[:, 2])
                accum[:, 3] = torch.where(active, accum[:, 3] + cur_pitch - d_pitch_dec, accum[:, 3])
                accum[:, 4] = torch.where(active, accum[:, 4] + cur_yaw   - d_yaw_dec,   accum[:, 4])

                # ── 写入 tokens 2-6（5 个连续值 token）──
                tokens_vals = torch.stack([
                    t_fwd, t_right, t_up, t_pitch, t_yaw,
                ], dim=1)  # [N, 5]
                mask_2d = active.unsqueeze(1).expand(-1, N_VALS)
                all_tokens[:, offset + 1:offset + 1 + N_VALS] = torch.where(
                    mask_2d, tokens_vals, all_tokens[:, offset + 1:offset + 1 + N_VALS],
                )
        else:
            # 直接编码（无残差修正）：纯 GPU 向量化，零 Python 循环
            _t_fwd   = self._vec_encode_move(d_fwd)        # [N, ticks]
            _t_right = self._vec_encode_move(d_right)
            _t_up    = self._vec_encode_move(d_up)
            _t_pitch = self._vec_encode_pitch(dp_deg)
            _t_yaw   = self._vec_encode_yaw(dy_deg)

            # 堆叠所有 5 个 token → [N, ticks, 5] → [N, ticks*5]
            tokens_all = torch.stack([
                _t_fwd, _t_right, _t_up, _t_pitch, _t_yaw,
            ], dim=-1).reshape(N, n_future_ticks * N_VALS)

            # 构建目标位置索引: tick t 的 token k 写入 t*tpt + 1 + k
            t_idx = torch.arange(n_future_ticks, device=device)                   # [ticks]
            k_idx = torch.arange(N_VALS, device=device)                             # [5]
            target_pos = (t_idx.unsqueeze(1) * tpt + 1 + k_idx.unsqueeze(0)).reshape(-1)  # [ticks*5]

            # active_mask [N, ticks] → expand → [N, ticks*5]
            active_flat = active_mask.unsqueeze(-1).expand(-1, -1, N_VALS).reshape(N, -1)

            all_tokens[:, target_pos] = torch.where(
                active_flat, tokens_all, all_tokens[:, target_pos],
            )

        return all_tokens

    def decode_tick(self, tokens: list[int]) -> dict:
        """将一个 tick 的 7 个 token 解码回连续值字典。"""
        continue_value = 1 if tokens[0] == self.CONTINUE_1 else 0
        result = {
            'continue': continue_value,
            'is_end': continue_value,
            'is_alive': continue_value,
            'd_forward': self._decode_move(tokens[1]),
            'd_right': self._decode_move(tokens[2]),
            'd_up': self._decode_move(tokens[3]),
            'd_pitch_deg': self._decode_angle(tokens[4], self.PITCH_OFFSET, self.n_pitch_values, self._half_pitch),
            'd_yaw_deg': self._decode_angle(tokens[5], self.YAW_OFFSET, self.n_yaw_values, self._half_yaw),
            'fire': 1 if tokens[6] == self.FIRE_1 else 0,
        }
        return result

    def decode_sequence(self, token_ids: torch.Tensor,
                        n_future_ticks: int) -> torch.Tensor:
        """
        将 token 序列解码回 10D 连续标签（纯 GPU 向量化）。

        Args:
            token_ids: [N, seq_len] token ids
            n_future_ticks: 预测的未来 tick 数

        Returns:
            [N, n_future_ticks, 10]  10D 标签 (d_fwd,d_right,d_up, cos/sin pitch, cos/sin yaw, alive, fire, end)
        """
        import math as _math
        N = token_ids.shape[0]
        device = token_ids.device
        seq_len = n_future_ticks * self.TOKENS_PER_TICK

        # ── Reshape: [N, n_ticks, 7] ──
        t = token_ids[:, :seq_len].reshape(N, n_future_ticks, self.TOKENS_PER_TICK)

        # ── PAD mask: any PAD in a tick → invalid ──
        pad_mask = (t == self.PAD).any(dim=-1)  # [N, n_ticks]

        # ── Binary tokens ──
        continue_value = (t[..., 0] == self.CONTINUE_1)
        fire = (t[..., self.FIRE_TOKEN_INDEX] == self.FIRE_1)

        # ── Signed magnitude helpers (direct decode from single token) ──
        def _vec_move(tok: torch.Tensor) -> torch.Tensor:
            return (tok - self.MOVE_OFFSET).clamp(0, self.n_move_values - 1).float() * self.move_grid_size - self._half_move

        def _vec_pitch(tok: torch.Tensor) -> torch.Tensor:
            return (tok - self.PITCH_OFFSET).clamp(0, self.n_pitch_values - 1).float() * self.angle_grid_size - self._half_pitch

        def _vec_yaw(tok: torch.Tensor) -> torch.Tensor:
            return (tok - self.YAW_OFFSET).clamp(0, self.n_yaw_values - 1).float() * self.angle_grid_size - self._half_yaw

        # ── Decode displacements (single signed token each) ──
        d_forward = _vec_move(t[..., 1])
        d_right   = _vec_move(t[..., 2])
        d_up      = _vec_move(t[..., 3])
        d_pitch   = _vec_pitch(t[..., 4])
        d_yaw     = _vec_yaw(t[..., 5])

        # ── PAD override: dead/invalid → zero ──
        continue_value = torch.where(pad_mask, torch.zeros_like(continue_value), continue_value)

        # ── Build output: [N, n_ticks, 10] ──
        labels = torch.zeros(N, n_future_ticks, 10, device=device)
        labels[..., 0] = d_forward
        labels[..., 1] = d_right
        labels[..., 2] = d_up
        p_rad = d_pitch * (_math.pi / 180.0)
        labels[..., 3] = torch.cos(p_rad)
        labels[..., 4] = torch.sin(p_rad)
        y_rad = d_yaw * (_math.pi / 180.0)
        labels[..., 5] = torch.cos(y_rad)
        labels[..., 6] = torch.sin(y_rad)
        labels[..., 7] = continue_value.float()
        labels[..., 8] = fire.float()
        labels[..., 9] = continue_value.float()

        return labels

    def get_binary_token(self, name: str, value: bool) -> int:
        """便捷方法: 获取二元 token id。"""
        if name in ('continue', 'is_end', 'is_alive'):
            return self.CONTINUE_1 if value else self.CONTINUE_0
        elif name == 'fire':
            return self.FIRE_1 if value else self.FIRE_0
        raise ValueError(f"Unknown binary token: {name}")


# ═══════════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class PretrainConfig:
    d_model: int = 256
    n_spatial_layers: int = 4
    n_temporal_layers: int = 4
    n_decoder_layers: int = 2
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    n_depth_ray_layers: int = 2  # DepthRayEncoder 的 transformer 层数

    # Data constants (synced with config.py)
    n_players: int = N_PLAYERS
    n_projectiles: int = N_MAX_PROJECTILES
    n_tokens: int = N_TOKENS
    n_maps: int = N_MAPS
    n_weapons: int = N_WEAPONS
    n_proj_types: int = N_PROJECTILE_TYPES
    n_ticks: int = 64               # 输入/输出 tick 数（预测未来 n_ticks 个 tick）
    label_dim: int = 10             # d_forward/right/up(3) + cos/sin pitch(2) + cos/sin yaw(2) + is_alive + is_firing + end

    # ── 离散 token 化参数 ──
    move_range: float = 128.0       # d_forward/right/up 最大移动距离 (clamp 到 [0, move_range])
    move_grid_size: float = 1.0     # 移动 token 分辨率 (单位/token)
    angle_grid_size: float = 1.0    # 角度 token 分辨率 (度/token)
    use_residual_correction: bool = True  # 是否使用残差修正（逐 tick 编码-解码-积累，修正离散化误差）

    # Derived
    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads


# ═══════════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """2-layer MLP: input_dim → hidden → output_dim, GELU.

    特征编码层，不做 dropout —— 正则化由下游 Transformer 层负责。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Adapter(nn.Module):
    """Bottleneck adapter: d → d//2 → GELU → d.

    用于共享组件的梯度解耦：同一份 embedding/encoder 被多处复用时，
    插入 Adapter 让每个下游路径有独立的轻量投影层，避免梯度冲突。
    统一使用 PyTorch 默认 kaiming 初始化。
    """

    def __init__(self, d_model: int):
        super().__init__()
        hidden = d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sinusoidal_time_encoding(times: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Sinusoidal encoding based on continuous time values.

    Args:
        times: [..., T] float, round_seconds for each tick
        d_model: embedding dimension

    Returns:
        [..., T, d_model]
    """
    *batch_dims, T = times.shape
    device = times.device

    # 使用原始秒数 [0, ~160]，不做归一化。
    # 最高频分量 (i=0) 波长 = 2π ≈ 6.28s，每个 tick (0.25s) 的相位变化 ≈ 14°，
    # 16  tick 窗口内覆盖约 64% 周期，tick 间可有效区分。
    # 低频分量波长 ≈ 600s，捕获 round 阶段信息（早期 vs 晚期）。
    position = times.unsqueeze(-1)  # [..., T, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(*batch_dims, T, d_model, device=device)
    pe[..., 0::2] = torch.sin(position * div_term)
    pe[..., 1::2] = torch.cos(position * div_term)
    return pe


# ═══════════════════════════════════════════════════════════════════════════════════
# 1a. Depth Ray Encoder
# ═══════════════════════════════════════════════════════════════════════════════════

class DepthRayEncoder(nn.Module):
    """
    Small transformer over 64 depth rays per player-tick.

    Each ray: [log_dist, cos(yaw), sin(yaw), cos(pitch), sin(pitch)].
    Rays attend to each other to learn spatial patterns (walls, gaps, corridors).

    No masking needed: dead players' output is discarded by torch.where in the embedder,
    and alive players always have all 64 rays valid.

    Input:  depth  [N, 64, 5]

    Output: [N, d]  mean-pooled ray features
    """

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        d = cfg.d_model
        self.ray_proj = nn.Linear(5, d)                # per-ray projection (angles already encode direction)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_depth_ray_layers)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:   # [N, 64, 5] → [N, d]
        x = self.ray_proj(depth)                              # [N, 64, d]
        x = self.transformer(x)                               # [N, 64, d]
        return x.mean(dim=-2)                                 # [N, d]


# ═══════════════════════════════════════════════════════════════════════════════════
# 1b. Token Embedder
# ═══════════════════════════════════════════════════════════════════════════════════

class TokenEmbedder(nn.Module):
    """
    将 batch 中的结构化特征映射为 27 个 d 维 token。

    Player (0-9):  MLP1(pos,map) + MLP2(state) + Σ Emb1(inv) + Σ MLP5(rel, id_emb)
                   + MLP_sound(sound) + DepthRayEncoder(depth_rays) + id_emb
                   Dead: DeadEmbedding + id_emb
    Bomb (10):     MLP1(pos,map) + MLP4(bomb_state, time)
    Proj (11-26):  MLP1(pos,map) + MLP3(type, dur, is_active)   (empty→zero)
    """

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        d = cfg.d_model
        self.d = d

        # ── Embedding tables ──
        self.map_emb = nn.Embedding(cfg.n_maps, d)
        self.player_id_emb = nn.Embedding(cfg.n_players, d)
        self.weapon_emb = nn.Embedding(cfg.n_weapons, d)
        self.proj_type_emb = nn.Embedding(cfg.n_proj_types, d)
        self.dead_emb = nn.Parameter(torch.randn(1, d) * 0.02)

        # player_id_emb adapters: 同一个 ID embedding 用于 self 和 relation 两种上下文
        self.pid_self_adapter = Adapter(d)  # inline: identity embed → self-identity
        self.pid_rel_adapter  = Adapter(d)   # inline: identity embed → relation-target

        # ── MLPs ──
        hidden = d * 2
        self.mlp1 = MLP(3 + d, hidden, d)   # pos(3) + map_emb(d) → d（player/bomb/proj 共享）
        # mlp1 按实体类型分离的 adapter
        self.mlp1_player_adapter = Adapter(d)  # inline: shared mlp1 → player
        self.mlp1_bomb_adapter   = Adapter(d)    # inline: shared mlp1 → bomb
        self.mlp1_proj_adapter   = Adapter(d)    # inline: shared mlp1 → projectile
        self.mlp2 = MLP(14, hidden, d)       # state(14) → d
        self.mlp3 = MLP(d + 2, hidden, d)    # type_emb(d) + dur(1) + is_active(1) → d
        self.mlp4 = MLP(5, hidden, d)        # bomb_state(4) + time(1) → d
        self.mlp5 = MLP(14 + d, hidden, d)   # rel(14) + id_emb(d) → d
        self.mlp_sound = MLP(2, hidden, d)    # sound(2) → d
        self.depth_encoder = DepthRayEncoder(cfg)          # 64-ray transformer → d
        self.depth_enc_adapter = Adapter(d)                 # 将共享 depth 表示投影到 encoder 求和空间
        self.mlp_angle = MLP(4, hidden, d)                   # cos/sin yaw + cos/sin pitch (4) → d

        # 其余层用 PyTorch 默认 kaiming 初始化

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: collated pretrain batch dict

        Returns:
            tokens: [B, T, 27, d]
        """
        B, T = batch["player_pos"].shape[:2]
        d = self.d
        device = batch["player_pos"].device

        tokens = torch.zeros(B, T, 27, d, device=device)

        # ── Player tokens (0-9) ──
        pos = batch["player_pos"]                                        # [B,T,10,3]
        state = batch["player_state"]                                    # [B,T,10,14]
        inv = batch["player_inv"].long()                                 # [B,T,10,9]
        inv_mask = batch["player_inv_mask"].bool()                       # [B,T,10,9]
        rel_f = batch["player_rel_f"]                                    # [B,T,10,9,14]
        rel_i = batch["player_rel_i"].long()                             # [B,T,10,9]
        rel_mask = batch["player_rel_mask"].bool()                       # [B,T,10,9]
        alive = batch["player_alive_mask"].bool()                        # [B,T,10]
        map_idx = batch["map_idx"].long()                                # [B,T]

        map_emb = self.map_emb(map_idx)                                  # [B,T,d]
        map_emb_exp = map_emb.unsqueeze(2).expand(-1, -1, 10, -1)       # [B,T,10,d]
        id_emb = self.player_id_emb.weight.unsqueeze(0).unsqueeze(0)     # [1,1,10,d]
        id_emb = self.pid_self_adapter(id_emb)                            # 适配 self-identity 语义

        # MLP1: pos(3) + map_emb(d) → d
        mlp1_in = torch.cat([pos, map_emb_exp], dim=-1)                 # [B,T,10,3+d]
        mlp1_out = self.mlp1(mlp1_in)                                   # [B,T,10,d]
        mlp1_out = self.mlp1_player_adapter(mlp1_out)                     # 适配 player 实体

        # MLP2: state(14) → d
        mlp2_out = self.mlp2(state)                                     # [B,T,10,d]

        # Emb1: sum over valid weapon slots
        inv_emb = self.weapon_emb(inv.clamp(0, self.weapon_emb.num_embeddings - 1))  # [B,T,10,9,d]
        inv_emb = inv_emb * inv_mask.unsqueeze(-1).float()               # zero invalid
        inv_sum = inv_emb.sum(dim=3)                                     # [B,T,10,d]

        # MLP5: sum over valid relations
        id_emb_j = self.player_id_emb(rel_i.clamp(0, 9))                # [B,T,10,9,d]
        id_emb_j = self.pid_rel_adapter(id_emb_j)                         # 适配 relation-target 语义
        mlp5_in = torch.cat([rel_f, id_emb_j], dim=-1)                  # [B,T,10,9,14+d]
        mlp5_out = self.mlp5(mlp5_in)                                    # [B,T,10,9,d]
        mlp5_out = mlp5_out * rel_mask.unsqueeze(-1).float()             # zero invalid
        mlp5_sum = mlp5_out.sum(dim=3)                                   # [B,T,10,d]

        # MLP_sound: footstep/gunshot sound features (2) → d
        sound = batch["player_sound"]                                    # [B,T,10,2]
        sound_out = self.mlp_sound(sound)                                # [B,T,10,d]

        # DepthRayEncoder: 64-ray transformer → mean pool → d
        depth = batch["player_depth"]                                    # [B,T,10,64,5]
        N = B * T * 10
        depth_out = self.depth_encoder(
            depth.reshape(N, 64, 5),
        ).reshape(B, T, 10, self.d)                                      # [B,T,10,d]
        depth_out = self.depth_enc_adapter(depth_out)                     # 适配 encoder 求和空间

        # Alive player embedding
        player_emb = (mlp1_out + mlp2_out + inv_sum + mlp5_sum
                      + sound_out + depth_out + id_emb)                  # [B,T,10,d]

        # Dead player embedding
        dead_emb = self.dead_emb.unsqueeze(0).unsqueeze(0).expand(B, T, 10, -1) + id_emb

        # Select alive vs dead
        player_emb = torch.where(
            alive.unsqueeze(-1),
            player_emb,
            dead_emb,
        )
        tokens[:, :, :10, :] = player_emb

        # ── Bomb token (10) ──
        bomb_pos = batch["bomb_pos"]                                     # [B,T,3]
        bomb_state = batch["bomb_state"]                                 # [B,T,4]
        tick_times = batch["tick_times_input"]                            # [B,T]
        round_time = (tick_times / 160.0).unsqueeze(-1)                  # [B,T,1]

        bomb_mlp1_in = torch.cat([bomb_pos, map_emb], dim=-1)           # [B,T,3+d]
        bomb_mlp1_out = self.mlp1(bomb_mlp1_in)                          # [B,T,d]
        bomb_mlp1_out = self.mlp1_bomb_adapter(bomb_mlp1_out)             # 适配 bomb 实体
        bomb_mlp4_in = torch.cat([bomb_state, round_time], dim=-1)       # [B,T,5]
        bomb_mlp4_out = self.mlp4(bomb_mlp4_in)                          # [B,T,d]
        tokens[:, :, 10, :] = bomb_mlp1_out + bomb_mlp4_out

        # ── Projectile tokens (11-26) ──
        proj_pos = batch["proj_pos"]                                     # [B,T,16,3]
        proj_type = batch["proj_type"].long()                            # [B,T,16]
        proj_dur = batch["proj_dur"].unsqueeze(-1)                       # [B,T,16,1]
        proj_is_active = batch["proj_is_active"].unsqueeze(-1).float()   # [B,T,16,1] 区分飞行/落地
        proj_mask = batch["proj_mask"].bool()                            # [B,T,16]

        map_emb_proj = map_emb.unsqueeze(2).expand(-1, -1, 16, -1)      # [B,T,16,d]
        proj_mlp1_in = torch.cat([proj_pos, map_emb_proj], dim=-1)      # [B,T,16,3+d]
        proj_mlp1_out = self.mlp1(proj_mlp1_in)                          # [B,T,16,d]
        proj_mlp1_out = self.mlp1_proj_adapter(proj_mlp1_out)             # 适配 projectile 实体

        type_idx = proj_type.clamp(0, self.proj_type_emb.num_embeddings - 1)
        type_emb = self.proj_type_emb(type_idx)                          # [B,T,16,d]
        proj_mlp3_in = torch.cat([type_emb, proj_dur, proj_is_active], dim=-1)  # [B,T,16,d+2]
        proj_mlp3_out = self.mlp3(proj_mlp3_in)                          # [B,T,16,d]

        proj_emb = proj_mlp1_out + proj_mlp3_out                         # [B,T,16,d]
        proj_emb = proj_emb * proj_mask.unsqueeze(-1).float()            # zero invalid
        tokens[:, :, 11:27, :] = proj_emb

        return tokens


# ═══════════════════════════════════════════════════════════════════════════════════
# 2. Spatial Transformer
# ═══════════════════════════════════════════════════════════════════════════════════

class SpatialTransformer(nn.Module):
    """
    27 个 token 之间 self-attention，提取 10 个 player embedding。
    """

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_spatial_layers)

    def forward(
        self,
        tokens: torch.Tensor,           # [B, T, 27, d]
        attn_mask: torch.Tensor,        # [B, T, 27]  True = ignore
    ) -> torch.Tensor:                  # [B, T, 10, d]
        B, T, N, d = tokens.shape

        # Reshape: treat each tick independently
        x = tokens.reshape(B * T, N, d)           # [B*T, 27, d]
        mask = attn_mask.reshape(B * T, N)         # [B*T, 27]

        # key_padding_mask: only empty projectile slots are masked
        # Every tick has >= 11 valid tokens → no fully-masked rows
        x = self.transformer(x, src_key_padding_mask=mask)

        # Extract player tokens (0-9)
        x = x[:, :10, :]                           # [B*T, 10, d]
        x = x.reshape(B, T, 10, d)                 # [B, T, 10, d]
        return x


# ═══════════════════════════════════════════════════════════════════════════════════
# 3. Temporal Transformer
# ═══════════════════════════════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    """
    每个玩家独立的 causal self-attention over time。
    加上 continuous sinusoidal time encoding。
    """

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.d_model = cfg.d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_temporal_layers)

    def forward(
        self,
        player_emb: torch.Tensor,         # [B, T, 10, d]
        tick_times: torch.Tensor,          # [B, T]
    ) -> torch.Tensor:                     # [B, T, 10, d]
        B, T, P, d = player_emb.shape
        device = player_emb.device

        # Reshape: treat each player independently
        x = player_emb.permute(0, 2, 1, 3).reshape(B * P, T, d)   # [B*10, T, d]

        # Time encoding
        tick_times_exp = tick_times.unsqueeze(1).expand(-1, P, -1).reshape(B * P, T)
        time_enc = sinusoidal_time_encoding(tick_times_exp, d)      # [B*10, T, d]
        x = x + time_enc.to(device)

        # Causal mask: can only see past
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # All T ticks are valid (no padding), no key_padding_mask needed
        x = self.transformer(x, mask=causal_mask)

        # Reshape back
        x = x.reshape(B, P, T, d).permute(0, 2, 1, 3)              # [B, T, 10, d]
        return x


# ═══════════════════════════════════════════════════════════════════════════════════
# 4. Token Decoder — 纯离散 token 预测
# ═══════════════════════════════════════════════════════════════════════════════════

class DecoderKVCache:
    """AR 增量解码的 KV cache（按 decoder block 预分配，逐位置写入）。"""

    def __init__(
        self,
        n_blocks: int,
        n: int,
        n_heads: int,
        seq_len: int,
        head_dim: int,
        device: torch.device,
    ):
        self.k = [
            torch.zeros(n, n_heads, seq_len, head_dim, device=device)
            for _ in range(n_blocks)
        ]
        self.v = [
            torch.zeros(n, n_heads, seq_len, head_dim, device=device)
            for _ in range(n_blocks)
        ]
        self.length = 0  # 已写入 cache 的位置数（位置 0..length-1）


class CausalDecoderBlock(nn.Module):
    """单个 causal self-attention + FFN block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor,
                return_attention: bool = False):
        # Pre-LN: norm before each sub-layer, cleaner residual path
        attn_out, attn_weights = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=causal_mask,
            need_weights=return_attention,
            average_attn_weights=True,  # [N, seq, seq] averaged over heads
        )
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        if return_attention:
            return x, attn_weights
        return x

    @torch.no_grad()
    def forward_cached(
        self,
        x_p: torch.Tensor,      # [N, 1, d] 当前预测位置 p 的输入
        k: torch.Tensor,        # [N, n_heads, seq_len, head_dim] 本 block 的 K cache
        v: torch.Tensor,        # [N, n_heads, seq_len, head_dim] 本 block 的 V cache
        pos: int,               # 当前位置索引
    ) -> torch.Tensor:
        """
        增量解码：只处理位置 pos，K/V 复用 cache。

        与 forward 的数学完全一致（Pre-LN + causal self-attention + FFN），
        但 attention 只对已缓存位置 0..pos 计算，FFN 只算当前一个位置。
        """
        N = x_p.shape[0]
        d = self.norm1.normalized_shape[0]
        n_heads = self.self_attn.num_heads
        head_dim = d // n_heads

        h = self.norm1(x_p)
        w = self.self_attn.in_proj_weight
        b = self.self_attn.in_proj_bias

        q = F.linear(h, w[:d], b[:d]).view(N, 1, n_heads, head_dim)
        kk = F.linear(h, w[d:2 * d], b[d:2 * d]).view(N, 1, n_heads, head_dim)
        vv = F.linear(h, w[2 * d:], b[2 * d:]).view(N, 1, n_heads, head_dim)

        # 写入当前位置的 K/V，再对 0..pos 做 causal attention
        k[:, :, pos:pos + 1, :] = kk.transpose(1, 2)
        v[:, :, pos:pos + 1, :] = vv.transpose(1, 2)
        L = pos + 1

        # 与 nn.MultiheadAttention 完全一致的张量布局和 bmm 运算
        # （MHA: [N*h, L, hd] 上的 bmm + scaling + softmax + bmm）
        q3 = q.reshape(N * n_heads, 1, head_dim)
        k3 = k[:, :, :L, :].reshape(N * n_heads, L, head_dim)
        v3 = v[:, :, :L, :].reshape(N * n_heads, L, head_dim)
        scores = torch.bmm(q3, k3.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        if self.training and self.dropout.p > 0:
            attn = self.dropout(attn)
        out = torch.bmm(attn, v3)                                  # [N*h, 1, hd]
        out = out.reshape(N, n_heads, 1, head_dim).transpose(1, 2).reshape(N, 1, d)
        attn_out = self.self_attn.out_proj(out)

        x = x_p + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class TokenDecoder(nn.Module):
    """
    纯离散 token 解码器: player embedding → 8×n_future_ticks 个 token。

    GPT-style teacher forcing:
      输入: [cond, tok_0, tok_1, ..., tok_{seq-2}]
      输出: [logit_tok_0, ..., logit_tok_{seq-1}]
      目标: [tok_0, ..., tok_{seq-1}]

    推理时 autoregressive 逐 token 生成。
    """

    def __init__(self, cfg: PretrainConfig, tokenizer: CameraTokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.n_future_ticks = cfg.n_ticks
        self.seq_len = tokenizer.TOKENS_PER_GROUP * cfg.n_ticks  # 10 * n_ticks
        self.d_model = cfg.d_model

        # Token embedding table
        self.token_emb = nn.Embedding(self.vocab_size, cfg.d_model)

        # Condition projection: player embedding → d_model
        self.cond_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # Depth adapter: 将共享 depth 表示投影到 decoder token 空间
        self.depth_dec_adapter = Adapter(cfg.d_model)

        # XYZ adapter: 将共享 mlp1 输出投影到 decoder xyz token 空间
        self.xyz_dec_adapter = Adapter(cfg.d_model)

        # Angle adapter: 将共享 mlp_angle 输出投影到 decoder angle token 空间
        self.angle_dec_adapter = Adapter(cfg.d_model)

        # Position encodings for decoder positions (0 = cond, 1..seq_len = token/depth positions)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_len + 1, cfg.d_model) * 0.02
        )

        # Causal decoder blocks
        self.decoder_blocks = nn.ModuleList([
            CausalDecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_decoder_layers)
        ])

        # Output head: d_model → vocab_size
        self.head = nn.Linear(cfg.d_model, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        # token_emb 用小 std 防止初始 logits 方差过大
        nn.init.normal_(self.token_emb.weight, std=0.02)
        # pos_encoding 创建时已 randn*0.02；cond_proj/head/decoder_blocks 用默认 kaiming

    def forward(
        self,
        conditions: torch.Tensor,           # [N, d]  player embedding
        gt_tokens: torch.Tensor,            # [N, n_ticks * 7]  ground truth camera token ids
        depth_ctx: Optional[torch.Tensor] = None,  # [N, n_ticks, d]  per-tick depth
        xyz_ctx: Optional[torch.Tensor] = None,    # [N, n_ticks, d]  per-tick absolute xyz
        angle_ctx: Optional[torch.Tensor] = None,  # [N, n_ticks, d]  per-tick absolute angle
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练: teacher forcing, 一次前向得到所有位置的 logits。

        Args:
            return_hidden: True 时额外返回 decoder 各位置的 hidden state
                [N, seq_len, d]（下游任务在特定位置接预测头用）。

        Returns:
            logits: [N, seq_len, vocab_size]
            flat_targets: [N, seq_len]  expanded targets with PAD at conditioning positions
            (return_hidden=True 时) hidden: [N, seq_len, d]
        """
        N = conditions.shape[0]
        device = conditions.device
        seq_len = self.seq_len  # 10 * n_ticks
        tpt = self.tokenizer.TOKENS_PER_TICK   # 7
        tpg = self.tokenizer.TOKENS_PER_GROUP  # 10

        # ── Step 1: Build flat_targets with PAD at conditioning positions ──
        # Per tick group (10 位置): [depth(PAD)@0, xyz(PAD)@1, angle(PAD)@2,
        #                            camera0..6@3..9]   ← 7 个相机 token 预测位置
        flat_targets = torch.full((N, seq_len), self.tokenizer.PAD,
                                  dtype=gt_tokens.dtype, device=device)
        camera_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        # 3 conditioning slots (depth, xyz, angle) per tick → PAD, not predicted
        for offset in range(3):
            cond_indices = torch.arange(self.n_future_ticks, device=device) * tpg + offset
            camera_mask[cond_indices] = False
        flat_targets[:, camera_mask] = gt_tokens           # 7 camera tokens per tick

        # ── Step 2: Build decoder input (GPT left-shift) ──
        token_emb = self.token_emb(flat_targets[:, :-1])              # [N, seq_len-1, d]
        cond_emb = self.cond_proj(conditions).unsqueeze(1)             # [N, 1, d]
        decoder_input = torch.cat([cond_emb, token_emb], dim=1)        # [N, seq_len, d]
        decoder_input = decoder_input + self.pos_encoding[:, :seq_len, :]

        # ── Step 3: Inject conditioning tokens ──
        # Layout per tick group (tick*10 + offset)，10 位置 = 3 条件槽 + 7 相机 token：
        #   flat_targets(输出目标): depth(PAD)@0, xyz(PAD)@1, angle(PAD)@2,
        #                           camera tokens @3..9（预测位置）
        #   decoder_input(输入侧, GPT 左移): cond@0, depth_ctx@1, xyz_ctx@2,
        #                           angle_ctx@3, camera token embedding @4..10
        # 预测规则：位置 i → flat_targets[i]；camera_mask 在 0/1/2 置 False（不预测）
        if depth_ctx is not None:
            assert depth_ctx.shape[1] == self.n_future_ticks, \
                f"depth_ctx ticks {depth_ctx.shape[1]} != n_future_ticks {self.n_future_ticks}"
            depth_ctx = self.depth_dec_adapter(depth_ctx)       # 适配 decoder token 空间
            depth_indices = torch.arange(self.n_future_ticks, device=device) * tpg + 1
            decoder_input[:, depth_indices, :] = \
                depth_ctx + self.pos_encoding[:, depth_indices, :]

        if xyz_ctx is not None:
            assert xyz_ctx.shape[1] == self.n_future_ticks
            xyz_ctx = self.xyz_dec_adapter(xyz_ctx)
            xyz_indices = torch.arange(self.n_future_ticks, device=device) * tpg + 2
            decoder_input[:, xyz_indices, :] = \
                xyz_ctx + self.pos_encoding[:, xyz_indices, :]

        if angle_ctx is not None:
            assert angle_ctx.shape[1] == self.n_future_ticks
            angle_ctx = self.angle_dec_adapter(angle_ctx)
            angle_indices = torch.arange(self.n_future_ticks, device=device) * tpg + 3
            decoder_input[:, angle_indices, :] = \
                angle_ctx + self.pos_encoding[:, angle_indices, :]

        # ── Step 4: Causal decoder ──
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        out = decoder_input
        for block in self.decoder_blocks:
            out = block(out, attn_mask)                              # [N, seq_len, d]

        # Predict: position i → token_i
        logits = self.head(out)                                      # [N, seq_len, vocab_size]
        if return_hidden:
            return logits, flat_targets, out
        return logits, flat_targets

    @torch.no_grad()
    def init_generate(self, conditions: torch.Tensor) -> torch.Tensor:
        """初始化 AR 生成状态: 预分配全 seq_len 的零张量。

        AR 生成全程使用 causal mask，未来位置被完全屏蔽，无需 PAD token embedding。
        """
        N = conditions.shape[0]
        device = conditions.device
        x = torch.zeros(N, self.seq_len, self.d_model, device=device)
        x = x + self.pos_encoding[:, :self.seq_len, :]
        # 覆写 position 0 为 condition
        x[:, 0:1, :] = self.cond_proj(conditions).unsqueeze(1) + self.pos_encoding[:, :1, :]
        return x

    def new_kv_cache(self, n: int, device: torch.device) -> DecoderKVCache:
        """为一次 AR 生成创建预分配的 KV cache（需在 init_generate 后调用）。"""
        return DecoderKVCache(
            n_blocks=len(self.decoder_blocks),
            n=n,
            n_heads=self.cfg.n_heads,
            seq_len=self.seq_len,
            head_dim=self.d_model // self.cfg.n_heads,
            device=device,
        )

    @torch.no_grad()
    def seed_cache(self, x: torch.Tensor, kv_cache: DecoderKVCache):
        """将位置 0（player condition）写入 cache。"""
        if kv_cache.length == 0:
            self._append_cache_position(x, kv_cache, 0)

    @torch.no_grad()
    def _append_cache_position(self, x: torch.Tensor, kv_cache: DecoderKVCache, pos: int):
        """把 x 中位置 pos 的 K/V 依次写入各 block 的 cache。"""
        inp = x[:, pos:pos + 1, :]
        for i, block in enumerate(self.decoder_blocks):
            inp = block.forward_cached(inp, kv_cache.k[i], kv_cache.v[i], pos)
        kv_cache.length = pos + 1

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,      # [N, 1, vocab]
        temperature: float,
        top_k: int,
        top_p: float,
        argmax: bool,
    ) -> torch.Tensor:
        """Temperature + top-k + top-p 采样；argmax=True 时直接取 argmax。"""
        t = max(temperature, 1e-6)
        scaled = logits.squeeze(1) / t                            # [N, vocab_size]
        if argmax:
            return scaled.argmax(dim=-1)
        if top_k > 0:
            k = min(top_k, scaled.shape[-1])
            top_vals, _ = torch.topk(scaled, k, dim=-1)
            thresh = top_vals[:, -1:]
            scaled = torch.where(scaled >= thresh, scaled,
                                 torch.full_like(scaled, -1e10))

        probs = F.softmax(scaled, dim=-1)
        if top_p > 0.0 and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            # 标准 top-p：保留累计概率刚好超过 p 的那个边界 token
            remove = cumsum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove = remove.scatter(1, sorted_idx, remove)
            probs = probs.masked_fill(remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, 1).squeeze(-1)             # [N]

    @torch.no_grad()
    def generate_group(
        self,
        x: torch.Tensor,                    # [N, seq_len, d]  预分配全长度 decoder state
        tick_idx: int,                      # 当前 tick 索引
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        depth_emb: Optional[torch.Tensor] = None,  # [N, d]  depth
        xyz_emb: Optional[torch.Tensor] = None,    # [N, d]  absolute xyz
        angle_emb: Optional[torch.Tensor] = None,  # [N, d]  absolute angle (cos/sin yaw/pitch)
        kv_cache: Optional[DecoderKVCache] = None,  # 增量解码 cache；None 走原 full-run 路径
        argmax: bool = False,                       # True 时确定性取 argmax（评估用）
        return_logp: bool = False,                  # True 时额外返回每 token 的 log p
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成一个 tick group：写 depth + xyz + angle + 8 camera tokens 到预分配的 x 中。

        x 已在 init_generate 中预分配为 [N, seq_len, d]（PAD + pos_enc 填充）。
        本方法直接覆写对应位置，不再 cat。

        传入 kv_cache 时使用增量解码：每个 token 只计算当前预测位置，
        attention 复用已缓存 K/V，不再对完整序列重算（结果与原路径一致）。

        return_logp=True 时返回 (tick_tokens, x, tick_logp)：
            tick_logp: [N, TOKENS_PER_TICK] 每个采样 token 在**未缩放**分布下的
                       log p（模型自评分：对自身预测路径的置信度；与 temperature
                       无关，恒 ≤ log max(p)）。

        Returns:
            tick_tokens: [N, TOKENS_PER_TICK] 采样的 camera token ids
            x:           [N, seq_len, d]  更新后的状态（原地修改）
            tick_logp:   [N, TOKENS_PER_TICK]（仅 return_logp=True 时返回）
        """
        N = x.shape[0]
        device = x.device
        tpt = self.tokenizer.TOKENS_PER_TICK   # 7
        tpg = self.tokenizer.TOKENS_PER_GROUP  # 10
        group_start = tick_idx * tpg           # 0, 10, 20, ...
        _full_len = self.seq_len

        tick_tokens = torch.full((N, tpt), self.tokenizer.PAD,
                                  dtype=torch.long, device=device)
        if return_logp:
            tick_logp = torch.zeros(N, tpt, device=device)

        # ── Step 1: Write 3 conditioning tokens ──
        # Layout: depth at group+1, xyz at group+2, angle at group+3
        cond_slots = [
            (1, depth_emb),   # offset 1: depth
            (2, xyz_emb),     # offset 2: absolute xyz
            (3, angle_emb),   # offset 3: absolute angle
        ]
        for offset, emb in cond_slots:
            seq_pos = group_start + offset
            if seq_pos < _full_len:
                if emb is not None:
                    token = emb.unsqueeze(1)                        # [N, 1, d]
                else:
                    token = torch.zeros(N, 1, self.d_model, device=device)
                token = token + self.pos_encoding[:, seq_pos:seq_pos + 1, :]
                x[:, seq_pos:seq_pos + 1, :] = token

        # ── KV cache 增量路径：只计算当前预测位置 ──
        if kv_cache is not None:
            # 补齐所有已写入但尚未入 cache 的位置
            # （预测位置之前的内容可能是 TF 阶段写入的 cond/GT token，逐一入 cache）
            while kv_cache.length < group_start + 4 and kv_cache.length < _full_len:
                self._append_cache_position(x, kv_cache, kv_cache.length)

            for step in range(tpt):
                pred_pos = group_start + 3 + step
                if pred_pos >= _full_len:
                    break
                out = x[:, pred_pos:pred_pos + 1, :]
                for i, block in enumerate(self.decoder_blocks):
                    out = block.forward_cached(out, kv_cache.k[i], kv_cache.v[i], pred_pos)
                logits = self.head(out)                              # [N, 1, vocab_size]
                sampled = self._sample_logits(logits, temperature, top_k, top_p, argmax)
                tick_tokens[:, step] = sampled
                if return_logp:
                    # 未缩放分布下的 log p（与 temperature/top-k/top-p 无关）
                    tick_logp[:, step] = torch.log_softmax(
                        logits, dim=-1).squeeze(1).gather(
                            1, sampled.unsqueeze(1)).squeeze(1)

                write_pos = pred_pos + 1
                if write_pos < _full_len:
                    next_emb = self.token_emb(sampled).unsqueeze(1)  # [N, 1, d]
                    next_emb = next_emb + self.pos_encoding[:, write_pos:write_pos + 1, :]
                    x[:, write_pos:write_pos + 1, :] = next_emb
                    self._append_cache_position(x, kv_cache, write_pos)
            if return_logp:
                return tick_tokens, x, tick_logp
            return tick_tokens, x

        # ── Step 2: Generate 8 camera tokens ──
        # camera step s: predict from pos group+3+s, write to pos group+4+s
        # （无 cache 的 full-run 路径，保持与训练一致的实现，可作对照）
        attn_mask = torch.triu(
            torch.full((_full_len, _full_len), float("-inf"), device=device),
            diagonal=1,
        )
        for step in range(tpt):
            pred_pos = group_start + 3 + step  # position whose output is used for prediction
            out = x
            for block in self.decoder_blocks:
                out = block(out, attn_mask)

            last_out = out[:, pred_pos:pred_pos + 1, :]             # [N, 1, d]
            logits = self.head(last_out)                              # [N, 1, vocab_size]
            sampled = self._sample_logits(logits, temperature, top_k, top_p, argmax)
            tick_tokens[:, step] = sampled
            if return_logp:
                tick_logp[:, step] = torch.log_softmax(
                    logits, dim=-1).squeeze(1).gather(
                        1, sampled.unsqueeze(1)).squeeze(1)

            # 覆写 camera token 到 x 的对应位置
            write_pos = pred_pos + 1  # token goes to next position
            if write_pos < _full_len:
                next_emb = self.token_emb(sampled).unsqueeze(1)       # [N, 1, d]
                next_emb = next_emb + self.pos_encoding[:, write_pos:write_pos + 1, :]
                x[:, write_pos:write_pos + 1, :] = next_emb

        if return_logp:
            return tick_tokens, x, tick_logp
        return tick_tokens, x

    @torch.no_grad()
    def generate(
        self,
        conditions: torch.Tensor,           # [N, d]
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        depth_ctx: Optional[torch.Tensor] = None,  # [N, n_ticks, d]
        xyz_ctx: Optional[torch.Tensor] = None,    # [N, n_ticks, d]
        angle_ctx: Optional[torch.Tensor] = None,  # [N, n_ticks, d]
    ) -> torch.Tensor:
        """
        Autoregressive 推理: cond → 逐 tick group 生成完整序列。
        内部使用 init_generate + generate_group。

        Returns:
            tokens: [N, n_ticks * TOKENS_PER_TICK] 生成的 camera token ids
        """
        x = self.init_generate(conditions)
        device = conditions.device
        kv_cache = self.new_kv_cache(conditions.shape[0], device)
        self.seed_cache(x, kv_cache)
        if depth_ctx is not None:
            assert depth_ctx.shape[1] == self.n_future_ticks, \
                f"depth_ctx ticks {depth_ctx.shape[1]} != n_future_ticks {self.n_future_ticks}"
            depth_ctx = self.depth_dec_adapter(depth_ctx)  # 适配 decoder token 空间
        if xyz_ctx is not None:
            assert xyz_ctx.shape[1] == self.n_future_ticks
            xyz_ctx = self.xyz_dec_adapter(xyz_ctx)
        if angle_ctx is not None:
            assert angle_ctx.shape[1] == self.n_future_ticks
            angle_ctx = self.angle_dec_adapter(angle_ctx)
        n_ticks = depth_ctx.shape[1] if depth_ctx is not None else self.n_future_ticks

        all_tokens = []
        for tick in range(n_ticks):
            depth_emb = depth_ctx[:, tick, :] if depth_ctx is not None else None
            xyz_emb = xyz_ctx[:, tick, :] if xyz_ctx is not None else None
            angle_emb = angle_ctx[:, tick, :] if angle_ctx is not None else None
            tick_tokens, x = self.generate_group(
                x, tick, temperature, top_k, top_p,
                depth_emb=depth_emb, xyz_emb=xyz_emb, angle_emb=angle_emb,
                kv_cache=kv_cache,
            )
            all_tokens.append(tick_tokens)

        return torch.cat(all_tokens, dim=1)                            # [N, n_ticks * 7]


# ═══════════════════════════════════════════════════════════════════════════════════
# 5. Full Model
# ═══════════════════════════════════════════════════════════════════════════════════

@_torch_dynamo_disable
def sample_keep_indices(N: int, keep_ratio: float, device) -> torch.Tensor:
    """从 N 个序列中随机保留 n_keep 个的索引（每步重新抽样）。

    keep_ratio >= 1.0 时返回 torch.arange(N)（全保留，等价于不抽样），
    保证调用方代码路径统一（无 None 分支），dynamo 只看到一个图输入 keep_idx。

    本函数在 eager 域执行（torch.compile 的 dynamo 不追踪内部）：
    torch.randperm 是随机 op，若进入编译图，dynamo speculation 重启分析时
    会因随机结果不同走不同路径，报 "SpeculationLog diverged"。
    """
    if keep_ratio >= 1.0:
        return torch.arange(N, device=device)
    n_keep = max(1, int(N * keep_ratio))
    return torch.randperm(N, device=device)[:n_keep]


class CS2PretrainModel(nn.Module):
    """完整的预训练模型 — 纯离散 token 预测版。"""

    def __init__(self, cfg: Optional[PretrainConfig] = None):
        super().__init__()
        self.cfg = cfg or PretrainConfig()
        self.embedder = TokenEmbedder(self.cfg)
        self.spatial = SpatialTransformer(self.cfg)
        self.temporal = TemporalTransformer(self.cfg)
        self.tokenizer = CameraTokenizer(
            move_range=self.cfg.move_range,
            move_grid_size=self.cfg.move_grid_size,
            angle_grid_size=self.cfg.angle_grid_size,
            use_residual_correction=self.cfg.use_residual_correction,
        )
        self.decoder = TokenDecoder(self.cfg, self.tokenizer)

    def _build_spatial_mask(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """构建 spatial attention mask [B, T, 27], True = ignore."""
        B, T = batch["player_pos"].shape[:2]
        device = batch["player_pos"].device
        valid = torch.ones(B, T, 27, device=device, dtype=torch.bool)
        proj_mask = batch["proj_mask"].bool()
        valid[:, :, 11:27] = proj_mask
        return ~valid

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        global_step: int = 0,
        keep_ratio: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch:  collated pretrain batch dict
            labels: [B, n_ticks*2-1, 10, 10] ground truth.
                    如果为 None，使用 autoregressive 推理。
            keep_ratio: <1.0 时从 N=B*T*10 个序列（player-tick 预测任务）中随机保留
                        一部分，只对保留子集跑 decoder —— 减少重复序列的冗余计算
                        （decoder 是主要 FLOPs 来源）。每步重新抽样，训练均匀覆盖。

        Returns:
            dict with: token_logits, loss, metrics, player_emb
        """
        B, T = batch["player_pos"].shape[:2]
        cfg = self.cfg

        # 1. Token embedding
        tokens = self.embedder(batch)                                    # [B, T, 27, d]

        # 2. Spatial transformer
        attn_mask = self._build_spatial_mask(batch)
        player_emb = self.spatial(tokens, attn_mask)                     # [B, T, 10, d]

        # 3. Temporal transformer
        tick_times = batch["tick_times_input"]
        player_emb = self.temporal(player_emb, tick_times)               # [B, T, 10, d]

        # 4. Decoder: 每个 player embedding → 8×n_future_ticks 个 token
        N = B * T * cfg.n_players
        conditions = player_emb.reshape(N, cfg.d_model)

        # keep_ratio: 随机保留 n_keep 个序列（每步重新抽样；eager 域执行，避开 dynamo 分歧）
        keep_idx = sample_keep_indices(N, keep_ratio, conditions.device)
        conditions = conditions[keep_idx]

        # ── Per-tick depth context for AR decoder (inject at fire token positions) ──
        depth_ctx = None
        if "player_depth_labels" in batch:
            depth_labels = batch["player_depth_labels"]               # [B, total_ticks, 10, 64, 5]
            B_d, total, P, R, F = depth_labels.shape
            assert B_d == B, f"depth_labels batch {B_d} != player batch {B}"
            assert P == cfg.n_players, f"depth_labels players {P} != {cfg.n_players}"
            # 先编码所有唯一的 depth（每 round_tick × player 一次），再 unfold 切片避免重复计算
            depth_all = depth_labels.reshape(B * total * P, R, F)    # [B*total*10, 64, 5]
            depth_enc = self.embedder.depth_encoder(depth_all)        # [B*total*10, d]
            depth_enc = depth_enc.reshape(B, total, P, cfg.d_model)   # [B, total_ticks, 10, d]
            # 滑窗切片：每个样本取 n_ticks 个 future tick 的 depth
            depth_enc = depth_enc.permute(0, 2, 1, 3)                 # [B, 10, total_ticks, d]
            depth_windows = depth_enc.unfold(2, cfg.n_ticks, 1)       # [B, 10, T, d, n_ticks]  unfold追加维度在末尾
            assert depth_windows.shape[2] == T, \
                f"depth unfold T={depth_windows.shape[2]} != input T={T}"
            depth_ctx = depth_windows.permute(0, 2, 1, 4, 3).reshape(N, cfg.n_ticks, cfg.d_model)
            # [B, T, 10, n_ticks, d] → [N, n_ticks, d]

        # ── Per-tick absolute xyz & angle context (static, repeated like depth) ──
        xyz_ctx, angle_ctx = self._build_abs_context(batch, N, T, cfg)

        result = {"player_emb": player_emb}

        if labels is not None:
            # ── 构造 ground truth token 序列（残差修正）──
            # labels: [B, n_ticks*2-1, 10, 10]
            # 对每个输入 tick t，取 labels[t : t+n_ticks]
            label_windows = labels.unfold(1, cfg.n_ticks, 1)          # [B, T, 10, 10, n_ticks]
            label_windows = label_windows.permute(0, 1, 2, 4, 3)             # [B, T, 10, n_ticks, 10]
            decoder_labels = label_windows.reshape(N, cfg.n_ticks, cfg.label_dim)
            decoder_labels = decoder_labels[keep_idx]

            # 残差修正编码 → token ids
            gt_tokens = self.tokenizer.encode_sequence(decoder_labels,
                                                       cfg.n_ticks)   # [n_keep, seq_len]

            # Teacher forcing forward (returns logits + expanded targets with PAD at depth positions)
            token_logits, flat_targets = self.decoder(
                conditions, gt_tokens,
                depth_ctx=depth_ctx[keep_idx]
                if depth_ctx is not None else depth_ctx,
                xyz_ctx=xyz_ctx[keep_idx],
                angle_ctx=angle_ctx[keep_idx],
            )

            # Loss (depth positions are PAD → auto-ignored by ignore_index)
            loss, metrics = self._compute_loss(token_logits, flat_targets)

            result["token_logits"] = token_logits
            result["gt_tokens"] = flat_targets
            result["loss"] = loss
            result["metrics"] = metrics
        else:
            # Inference
            token_logits = self.decoder.generate(conditions,
                                                  depth_ctx=depth_ctx,
                                                  xyz_ctx=xyz_ctx,
                                                  angle_ctx=angle_ctx)
            result["token_logits"] = token_logits

        return result

    @torch.no_grad()
    def get_player_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """下游任务接口：返回每个 tick 每个玩家的 embedding [B, T, 10, d]."""
        tokens = self.embedder(batch)
        attn_mask = self._build_spatial_mask(batch)
        player_emb = self.spatial(tokens, attn_mask)
        tick_times = batch["tick_times_input"]
        player_emb = self.temporal(player_emb, tick_times)
        return player_emb

    def _build_abs_context(
        self,
        batch: Dict[str, torch.Tensor],
        N: int,                          # B * T * n_players
        T: int,                          # n_ticks (input window size)
        cfg,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        用未来每个 tick 的真实 xyz 和角度作为 per-tick conditioning（与 depth 一致）。

        batch 必须包含 player_pos_labels 和 player_angle_labels。

        Returns:
            xyz_ctx:   [N, n_ticks, d]
            angle_ctx: [N, n_ticks, d]
        """
        assert "player_pos_labels" in batch, \
            "batch 缺少 player_pos_labels，请确认数据管线已更新（pretrain_processor.py）"
        assert "player_angle_labels" in batch, \
            "batch 缺少 player_angle_labels，请确认数据管线已更新（pretrain_processor.py）"

        map_idx = batch["map_idx"].long()        # [B, T]
        P = cfg.n_players
        n_ticks = cfg.n_ticks

        # ── xyz context ──────────────────────────────────────────────────
        pos_labels = batch["player_pos_labels"]     # [B, total_ticks, 10, 3]
        B_l, total, P_l, _ = pos_labels.shape
        assert B_l == N // (T * P), f"pos_labels batch mismatch"
        assert P_l == P

        # Encode all positions with map embedding
        pos_flat = pos_labels.reshape(B_l * total * P, 3)
        map_sample = map_idx[:, 0]                             # [B] — 同 round 内不变
        map_emb_sample = self.embedder.map_emb(map_sample)      # [B, d]
        map_emb_exp = map_emb_sample[:, None, None, :].expand(B_l, total, P, -1)
        map_emb_flat = map_emb_exp.reshape(B_l * total * P, self.cfg.d_model)

        xyz_enc = self.embedder.mlp1(torch.cat([pos_flat, map_emb_flat], dim=-1))
        xyz_enc = xyz_enc.reshape(B_l, total, P, self.cfg.d_model)
        xyz_enc = xyz_enc.permute(0, 2, 1, 3)                 # [B, 10, total_ticks, d]
        xyz_windows = xyz_enc.unfold(2, n_ticks, 1)            # [B, 10, T, d, n_ticks]
        assert xyz_windows.shape[2] == T, \
            f"xyz unfold T={xyz_windows.shape[2]} != input T={T}"
        xyz_ctx = xyz_windows.permute(0, 2, 1, 4, 3).reshape(N, n_ticks, self.cfg.d_model)

        # ── angle context ────────────────────────────────────────────────
        angle_labels = batch["player_angle_labels"]  # [B, total_ticks, 10, 4]
        B_l2, total2, P_l2, _ = angle_labels.shape
        assert P_l2 == P

        angle_flat = angle_labels.reshape(B_l2 * total2 * P, 4)
        angle_enc = self.embedder.mlp_angle(angle_flat)
        angle_enc = angle_enc.reshape(B_l2, total2, P, self.cfg.d_model)
        angle_enc = angle_enc.permute(0, 2, 1, 3)             # [B, 10, total_ticks, d]
        angle_windows = angle_enc.unfold(2, n_ticks, 1)       # [B, 10, T, d, n_ticks]
        angle_ctx = angle_windows.permute(0, 2, 1, 4, 3).reshape(N, n_ticks, self.cfg.d_model)

        return xyz_ctx, angle_ctx

    def _compute_loss(
        self,
        token_logits: torch.Tensor,     # [N, seq_len, vocab_size]
        gt_tokens: torch.Tensor,        # [N, seq_len]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        纯 Cross-Entropy loss over all token positions.
        """
        N, seq_len, vocab = token_logits.shape
        device = token_logits.device

        # Cross-entropy over all positions
        loss = F.cross_entropy(
            token_logits.reshape(-1, vocab),
            gt_tokens.reshape(-1),
            ignore_index=self.tokenizer.PAD,   # 忽略 PAD 位置
        )

        # ── Metrics ──
        with torch.no_grad():
            pred_tokens = torch.argmax(token_logits, dim=-1)
            non_pad_mask = gt_tokens != self.tokenizer.PAD
            total_non_pad = non_pad_mask.sum().float()
            acc = (pred_tokens[non_pad_mask] == gt_tokens[non_pad_mask]).float().mean() \
                if total_non_pad > 0 else torch.tensor(0.0, device=device)

        metrics = {"token_acc": acc}
        return loss, metrics

# ═══════════════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing CS2PretrainModel (discrete tokens + per-tick depth)...")
    cfg = PretrainConfig(
        d_model=64, n_spatial_layers=1, n_temporal_layers=1,
        n_decoder_layers=1, n_heads=4, d_ff=128,
        n_depth_ray_layers=1,
        n_ticks=16,
        move_range=128.0, move_grid_size=1.0, angle_grid_size=1.0,
    )
    model = CS2PretrainModel(cfg)

    B, T = 1, cfg.n_ticks
    device = torch.device("cpu")

    # Build synthetic batch
    batch = {
        "player_pos": torch.randn(B, T, 10, 3),
        "player_state": torch.randn(B, T, 10, 14),
        "player_inv": torch.randint(0, 44, (B, T, 10, 9)),
        "player_inv_mask": torch.ones(B, T, 10, 9, dtype=torch.bool),
        "player_rel_f": torch.randn(B, T, 10, 9, 14),
        "player_rel_i": torch.randint(0, 10, (B, T, 10, 9)),
        "player_rel_mask": torch.ones(B, T, 10, 9, dtype=torch.bool),
        "player_alive_mask": torch.ones(B, T, 10, dtype=torch.bool),
        "player_depth": torch.randn(B, T, 10, 64, 5),
        "player_sound": torch.randn(B, T, 10, 2),
        "bomb_pos": torch.randn(B, T, 3),
        "bomb_state": torch.randn(B, T, 4),
        "map_idx": torch.zeros(B, T, dtype=torch.long),
        "proj_pos": torch.randn(B, T, 16, 3),
        "proj_type": torch.randint(0, 6, (B, T, 16)),
        "proj_dur": torch.rand(B, T, 16),
        "proj_mask": torch.ones(B, T, 16, dtype=torch.bool),
        "proj_is_active": torch.zeros(B, T, 16, dtype=torch.float32),
        "tick_times_input": torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 0.25,
    }

    # Labels: 2*n_ticks - 1 = 127 ticks
    n_label_ticks = cfg.n_ticks * 2 - 1
    batch["player_depth_labels"] = torch.randn(B, n_label_ticks, 10, 64, 5)
    batch["player_pos_labels"] = torch.randn(B, n_label_ticks, 10, 3)
    # angle: [cos(yaw), sin(yaw), cos(pitch), sin(pitch)]
    batch["player_angle_labels"] = torch.randn(B, n_label_ticks, 10, 4)
    labels = torch.randn(B, n_label_ticks, 10, cfg.label_dim)
    labels[..., 7] = (labels[..., 7] > 0).float()   # is_alive
    labels[..., 8] = (labels[..., 8] > 0).float()   # fire
    labels[..., 9] = (labels[..., 9] > -0.5).float()  # end (bias toward end=1)

    model.eval()
    with torch.no_grad():
        out = model(batch, labels)

    print(f"  vocab_size:      {model.tokenizer.vocab_size}")
    print(f"  seq_len:         {model.decoder.seq_len} (= 10 × {cfg.n_ticks})")
    print(f"  token_logits:    {out['token_logits'].shape}")
    print(f"  gt_tokens:       {out['gt_tokens'].shape}")
    print(f"  loss:            {out['loss'].item():.4f}")

    metrics = out["metrics"]
    for k in sorted(metrics.keys()):
        print(f"  {k}: {metrics[k].item():.4f}")

    # Test inference
    out_inf = model(batch, labels=None)
    print(f"  inference token_logits: {out_inf['token_logits'].shape}")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params / 1e6:.1f}M")
    print("✓ All shape checks passed")
