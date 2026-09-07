"""
预训练模型推理引擎。

加载 checkpoint，对 round-level 样本运行推理，输出预测轨迹（世界坐标）。

用法:
    from scripts.prediction_engine import PredictionEngine
    engine = PredictionEngine("config/pretrain-a100.yaml", "examples/checkpoints/step_0035000.pt")
    result = engine.predict_at_tick(sample, query_tick=120)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pretrain_model import CS2PretrainModel, PretrainConfig
from training_data.torch_dataset import (
    augment_depth_with_angles,
    sample_to_torch,
    N_PLAYERS,
    N_DEPTH_DIRS,
)
from training_data.config import (
    DEPTH_DIRECTIONS,
    DEPTH_MAX_DIST,
    MAP_NAME_TO_IDX,
    denormalize_position,
    game_forward_to_obj,
    game_to_obj,
    normalize_position,
)
from training_data.map_loader import MapGeometry, get_map_geometry, player_raycast_batch


# ═══════════════════════════════════════════════════════════════════════════════════
# 下游任务（winrate / future_kill / alive_end）
# ═══════════════════════════════════════════════════════════════════════════════════

DOWNSTREAM_TASKS = ("winrate", "future_kill", "alive_end")

TASK_LABELS = {
    "winrate": "队伍胜率",
    "future_kill": "未来击杀",
    "alive_end": "回合末存活",
}


class _TaskHead(torch.nn.Module):
    """下游任务线性头（fc: d_model → 1），参数键与 ckpt head_state 一致。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = torch.nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden).squeeze(-1)


def _last_kill_ticks(nxt_kill, nxt_death) -> np.ndarray:
    """从 label_nxt_kill / label_nxt_death 重建每玩家最后击杀 tick [10]。

    与 pretrain_processor.PretrainWindowExtractor._last_kill_ticks 完全一致：
    nxt_death 的"变化点" = 一次击杀事件（连杀时 killer 不变但 victim 变，
    nxt_kill 无法区分，nxt_death 可以）；击杀者 = 变化点 tick **之前**的
    nxt_kill 值（nk[t-1]）。-1 = 该玩家本回合无击杀；
    末尾残留记哨兵 tick T（保证在 t=T-1 判断"之后还能击杀"为 True）。
    """
    last = np.full(N_PLAYERS, -1, dtype=np.int64)
    if nxt_kill is None or nxt_death is None or len(nxt_kill) == 0:
        return last
    T = len(nxt_kill)

    def _record(t: int, victim: int):
        killer = int(nxt_kill[t - 1])
        if 0 <= killer < N_PLAYERS:
            last[killer] = t

    prev_d = -1
    for t in range(T):
        cur_d = int(nxt_death[t])
        if cur_d != prev_d:
            if prev_d != -1 and prev_d != 10:
                _record(t, prev_d)
            prev_d = cur_d
    if prev_d != -1 and prev_d != 10:
        _record(T, prev_d)
    return last


# ═══════════════════════════════════════════════════════════════════════════════════
# 轨迹积分工具
# ═══════════════════════════════════════════════════════════════════════════════════

def _camera_basis(yaw: float, pitch: float) -> Tuple[
    Tuple[float, float, float],  # forward (includes pitch vertical component)
    Tuple[float, float, float],  # right (horizontal, perpendicular to forward_xy)
    Tuple[float, float, float],  # up (world Z)
]:
    """
    相机基向量。

    完整相机基向量（含 pitch）：
      F = (cos(y)·cos(p), sin(y)·cos(p), sin(p))     — 相机视线方向
      R = (sin(y), -cos(y), 0)                        — 水平面内，垂直于 F_xy
      U = (-cos(y)·sin(p), -sin(y)·sin(p), cos(p))    — 相机上方向 (cross(R, F))

      世界位移恢复（camera_relative_up=True，v4 相机坐标系标签）：
        wx = d_fwd·cos(y)·cos(p) + d_right·sin(y) - d_up·cos(y)·sin(p) = dx
        wy = d_fwd·sin(y)·cos(p) - d_right·cos(y) - d_up·sin(y)·sin(p) = dy
        wz = d_fwd·sin(p) + d_up·cos(p) = dz

    水平基向量（pitch=0）：
      F = (cos(y), sin(y), 0)     — 水平面内 forward
      U = (0, 0, 1)                — 世界 Z 轴 (= 相机上方向退化到水平时)

      世界位移恢复（camera_relative_up=False，v5 世界对齐标签）：
        wx = d_fwd·cos(y) + d_right·sin(y)
        wz = d_up                   # d_up = pure dz
    """
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    forward = (cy * cp, sy * cp, sp)
    right = (sy, -cy, 0.0)
    up = (-cy * sp, -sy * sp, cp)    # camera up = cross(right, forward)

    return forward, right, up


def _extract_yaw_pitch(state: np.ndarray) -> Tuple[float, float]:
    """从 player_state[14] 提取 yaw, pitch（弧度）。"""
    pitch = math.atan2(float(state[7]), float(state[6]))
    yaw = math.atan2(float(state[9]), float(state[8]))
    return yaw, pitch


def _angle_from_cos_sin(cos_val: float, sin_val: float) -> float:
    """从 cos/sin 恢复角度增量（弧度）。"""
    return math.atan2(sin_val, cos_val)


def integrate_trajectory(
    start_pos: Tuple[float, float, float],
    start_yaw: float,
    start_pitch: float,
    deltas: np.ndarray,          # [N, 10] — 预测或 GT 标签
    start_step: int = 0,
    max_steps: int = 64,
    camera_relative_up: bool = False,  # labels are world-aligned: d_forward/d_right horizontal, d_up = dz
    return_angles: bool = False,
) -> Tuple[
    List[Tuple[float, float, float]],  # 世界坐标点
    List[float],                        # alive 概率
    List[float],                        # firing 概率
    Optional[List[float]],              # yaw（弧度，每点一个）
    Optional[List[float]],              # pitch（弧度，每点一个）
]:
    """
    将世界对齐 delta 序列积分到世界坐标轨迹（v5 坐标系约定）。

    d_forward, d_right 在水平面上（仅由 yaw 决定），d_up 为纯世界 Z。

    Args:
        start_pos:   起始位置 (gx, gy, gz) 游戏单位
        start_yaw:   起始 yaw（弧度）
        start_pitch: 起始 pitch（弧度）
        deltas:      [N, 10] 预测标签，dims:
                       0-2: d_forward, d_right, d_up
                       3-4: cos(d_pitch), sin(d_pitch)
                       5-6: cos(d_yaw), sin(d_yaw)
                       7:   is_alive (sigmoid logit)
                       8:   is_firing (sigmoid logit)
                       9:   end (sigmoid logit)
        start_step:  从 deltas 的第几步开始（跳过 end=0 的步）
        max_steps:   最多积分多少步

    Returns:
        (points, alive_probs, firing_probs, end_stopped_at)
        return_angles=True 时额外返回 (yaws_rad, pitches_rad)，长度与 points 一致。
        其中 yaws/pitches 记录的是移动到此点所用的航向（即该点的相机朝向）。
        end_stopped_at: step index where end<0 triggered, or -1 if all steps valid
    """
    px, py, pz = start_pos
    yaw, pitch = start_yaw, start_pitch

    points: List[Tuple[float, float, float]] = []
    alive_probs: List[float] = []
    firing_probs: List[float] = []
    yaws_rad: List[float] = []
    pitches_rad: List[float] = []
    end_stopped_at = -1

    N = min(deltas.shape[0] - start_step, max_steps)
    for i in range(start_step, start_step + N):
        d = deltas[i]
        # 检查 alive / end 信号
        alive_val = float(d[7])
        end_val = float(d[9])
        if alive_val < 0.5 or end_val < 0.5:  # 死亡或结束 → 停止积分
            end_stopped_at = i
            break

        # 记录当前朝向：这是移动到下一个点所使用的航向，可视化为该点的相机方向
        yaws_rad.append(yaw)
        pitches_rad.append(pitch)

        d_forward = float(d[0])
        d_right = float(d[1])
        d_up = float(d[2])

        # labels are world-aligned: d_forward/d_right in horizontal plane, d_up = pure dz
        if camera_relative_up:
            forward, right, up = _camera_basis(yaw, pitch)
        else:
            forward, right, up = _camera_basis(yaw, 0.0)  # 水平 forward，d_up = 纯世界 Z

        # 世界空间位移
        wx = d_forward * forward[0] + d_right * right[0] + d_up * up[0]
        wy = d_forward * forward[1] + d_right * right[1] + d_up * up[1]
        wz = d_forward * forward[2] + d_right * right[2] + d_up * up[2]

        px += wx
        py += wy
        pz += wz

        # 更新角度
        d_pitch = _angle_from_cos_sin(float(d[3]), float(d[4]))
        d_yaw = _angle_from_cos_sin(float(d[5]), float(d[6]))
        pitch += d_pitch
        yaw += d_yaw

        # 裁剪 pitch 到合理范围（避免数值溢出）
        pitch = max(-math.pi / 2, min(math.pi / 2, pitch))

        points.append((px, py, pz))
        alive_probs.append(float(d[7]))   # logit — 前端做 sigmoid
        firing_probs.append(float(d[8]))

    if return_angles:
        return points, alive_probs, firing_probs, end_stopped_at, yaws_rad, pitches_rad
    return points, alive_probs, firing_probs, end_stopped_at


# ═══════════════════════════════════════════════════════════════════════════════════
# 推理引擎
# ═══════════════════════════════════════════════════════════════════════════════════

class PredictionEngine:
    """加载预训练模型并运行推理。"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        maps_dir: str = "maps/optimized_obj_files",
        model_overrides: Optional[dict] = None,
    ):
        self.device = torch.device(device)
        self.maps_dir = Path(maps_dir)
        if not self.maps_dir.is_absolute():
            self.maps_dir = (_PROJECT_ROOT / self.maps_dir).resolve()

        # 加载配置
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}

        # 应用 CLI 覆盖（--predict-override key=value）
        if model_overrides:
            yaml_cfg.update(model_overrides)
            print(f"[PredictionEngine] Overrides applied: {model_overrides}")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_state = ckpt.get("model", ckpt)

        # 剥离 torch.compile 的 _orig_mod. 前缀
        new_state = {}
        for k, v in ckpt_state.items():
            new_state[k.replace("_orig_mod.", "")] = v
        ckpt_state = new_state

        model_cfg = PretrainConfig(
            d_model=yaml_cfg.get("d_model", 256),
            n_spatial_layers=yaml_cfg.get("n_spatial_layers", 4),
            n_temporal_layers=yaml_cfg.get("n_temporal_layers", 4),
            n_decoder_layers=yaml_cfg.get("n_decoder_layers", 2),
            n_heads=yaml_cfg.get("n_heads", 8),
            d_ff=yaml_cfg.get("d_ff", 1024),
            dropout=yaml_cfg.get("dropout", 0.1),
            n_depth_ray_layers=yaml_cfg.get("n_depth_ray_layers", 2),
            n_ticks=yaml_cfg.get("n_ticks", 64),
            move_range=yaml_cfg.get("move_range", 128.0),
            move_grid_size=yaml_cfg.get("move_grid_size", 1.0),
            angle_grid_size=yaml_cfg.get("angle_grid_size", 1.0),
        )
        self.model_cfg = model_cfg

        # 预计算局部射线方向（OBJ 空间），与 depth_map.py 一致
        self._local_dirs = np.zeros((N_DEPTH_DIRS, 3), dtype=np.float32)
        for d, (dyaw, dpitch) in enumerate(DEPTH_DIRECTIONS):
            ox, oy, oz = game_forward_to_obj(dyaw, dpitch)
            self._local_dirs[d] = (ox, oy, oz)

        # 加载模型
        self.model = CS2PretrainModel(model_cfg).to(self.device)

        missing, unexpected = self.model.load_state_dict(ckpt_state, strict=False)
        if missing:
            print(f"[PredictionEngine] Missing keys (using fresh init): {missing}")
        if unexpected:
            print(f"[PredictionEngine] Unexpected keys (ignored): {unexpected}")
        self.model.eval()
        print(f"[PredictionEngine] Loaded checkpoint step={ckpt.get('global_step', '?')} "
              f"params={sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")

        # 下游任务状态（apply_downstream_state 后生效）
        # 注意：路径预测始终使用预训练底座 self.model；
        #       下游模型（_down_model）只在指标计算时单独前向，不改动 self.model。
        self.downstream_task = None      # "winrate" / "future_kill" / "alive_end"
        self.downstream_step = None
        self.downstream_state = None     # 下游 merged 权重（CPU state dict）
        self.downstream_head_state = None
        self._down_model = None          # 指标计算专用下游模型实例（懒加载）
        self._down_head = None
        self._down_model_task = None     # _down_model 当前装的哪个任务的权重

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    def _get_starting_state(
        self, torch_sample: dict, start_idx: int, map_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取 query_tick 时刻 10 个玩家的游戏坐标位置和角度。

        Returns:
            pos_game:  [10, 3] float64 游戏坐标
            yaw_deg:   [10] float64 度
            pitch_deg: [10] float64 度
        """
        player_pos = torch_sample["player_pos"]       # [round_T, 10, 3]
        player_state = torch_sample["player_state"]   # [round_T, 10, 14]

        pos_game = np.zeros((N_PLAYERS, 3), dtype=np.float64)
        yaw_deg = np.zeros(N_PLAYERS, dtype=np.float64)
        pitch_deg = np.zeros(N_PLAYERS, dtype=np.float64)

        for p in range(N_PLAYERS):
            nx, ny, nz = player_pos[start_idx, p].tolist()
            gx, gy, gz = denormalize_position(nx, ny, nz, map_name)
            pos_game[p] = (gx, gy, gz)
            state = player_state[start_idx, p].numpy()
            yaw, pitch = _extract_yaw_pitch(state)
            yaw_deg[p] = math.degrees(yaw)
            pitch_deg[p] = math.degrees(pitch)

        return pos_game, yaw_deg, pitch_deg

    def _convert_labels_v4_to_v5(
        self, labels_orig: torch.Tensor, pitch_deg: np.ndarray, n_ticks: int,
    ) -> torch.Tensor:
        """将 v4 (camera-relative up) 标签转换为 v5 (world-aligned up = dz)。

        labels_orig: [10, n_ticks, 10] 原始标签
        pitch_deg:   [10] 每个玩家的当前 pitch
        """
        import math as _math
        labels_v5 = labels_orig.clone()
        for p in range(N_PLAYERS):
            pitch_acc = float(pitch_deg[p])
            for tick in range(n_ticks):
                d_fwd_v4 = float(labels_orig[p, tick, 0])
                d_up_v4 = float(labels_orig[p, tick, 2])
                cp = _math.cos(_math.radians(pitch_acc))
                sp = _math.sin(_math.radians(pitch_acc))
                labels_v5[p, tick, 0] = d_fwd_v4 * cp - d_up_v4 * sp
                labels_v5[p, tick, 2] = d_fwd_v4 * sp + d_up_v4 * cp
                dp = _math.atan2(float(labels_orig[p, tick, 4]),
                                 float(labels_orig[p, tick, 3]))
                pitch_acc += _math.degrees(dp)
                pitch_acc = max(-90.0, min(90.0, pitch_acc))
        return labels_v5

    def _apply_delta(
        self, pos_game: np.ndarray, yaw_deg: np.ndarray, pitch_deg: np.ndarray,
        alive_arr: np.ndarray, label_10d: np.ndarray, tick_idx: int,
        camera_relative_up: bool = False,
    ):
        """用 10D 标签更新游戏坐标位置、角度和存活状态（原地修改）。

        label_10d: [10, N, 10] — player × tick × 10D
        camera_relative_up: True for v4 (d_up along camera up), False for v5 (d_up = dz)
        """
        import math as _math

        n_rows = label_10d.shape[0]   # 10（全玩家）或 K（单玩家多采样）
        for p in range(n_rows):
            if not alive_arr[p]:
                continue
            d_fwd = float(label_10d[p, tick_idx, 0])
            d_right = float(label_10d[p, tick_idx, 1])
            d_up = float(label_10d[p, tick_idx, 2])
            dp_rad = _math.atan2(float(label_10d[p, tick_idx, 4]),
                                 float(label_10d[p, tick_idx, 3]))
            dy_rad = _math.atan2(float(label_10d[p, tick_idx, 6]),
                                 float(label_10d[p, tick_idx, 5]))
            is_alive = float(label_10d[p, tick_idx, 7])
            is_end = float(label_10d[p, tick_idx, 9])

            yaw = _math.radians(yaw_deg[p])

            if camera_relative_up:
                pitch = _math.radians(pitch_deg[p])
                forward, right, up = _camera_basis(yaw, pitch)
                pos_game[p, 0] += d_fwd * forward[0] + d_right * right[0] + d_up * up[0]
                pos_game[p, 1] += d_fwd * forward[1] + d_right * right[1] + d_up * up[1]
                pos_game[p, 2] += d_fwd * forward[2] + d_right * right[2] + d_up * up[2]
            else:
                cos_y, sin_y = _math.cos(yaw), _math.sin(yaw)
                pos_game[p, 0] += d_fwd * cos_y + d_right * sin_y
                pos_game[p, 1] += d_fwd * sin_y - d_right * cos_y
                pos_game[p, 2] += d_up

            yaw_deg[p] += _math.degrees(dy_rad)
            pitch_deg[p] = max(-90.0, min(90.0, pitch_deg[p] + _math.degrees(dp_rad)))

            # alive=0 表示死亡，end=0 表示停止；两者任一触发都不再继续更新状态
            if is_alive < 0.5 or is_end < 0.5:
                alive_arr[p] = False

    def _compute_depth_emb(self, map_geom, pos_game, yaw_deg, pitch_deg, alive_arr,
                           return_raw: bool = False):
        """Raycast + depth encoding → [10, d] tensor on device.

        return_raw=True 时返回 (emb, raw)，raw 为 encoder 之前的 [10, 64, 5]
        张量（下游指标需要在另一模型上重新编码，必须保留原始输入）。
        """
        from training_data.depth_map import compute_directional_depth

        if map_geom is None or not alive_arr.any():
            return (None, None) if return_raw else None

        depth_raw, _ = compute_directional_depth(
            map_geom,
            pos_game[np.newaxis, :, :],
            yaw_deg[np.newaxis, :],
            pitch_deg[np.newaxis, :],
            alive_arr[np.newaxis, :],
        )
        depth_aug = augment_depth_with_angles(
            {"player_depth": depth_raw}
        )["player_depth"][0]  # [10, 64, 5]
        if isinstance(depth_aug, np.ndarray):
            depth_aug = torch.from_numpy(depth_aug)
        emb = self.model.embedder.depth_encoder(
            depth_aug.to(self.device)
        )  # [10, d]
        if return_raw:
            return emb, depth_aug
        return emb

    def _compute_xyz_emb(self, pos_game: np.ndarray, map_name: str) -> torch.Tensor:
        """Encode absolute game-unit positions → [N, d] using mlp1 + xyz_dec_adapter."""
        from training_data.config import normalize_position

        n = pos_game.shape[0]   # 10（全玩家）或 K（单玩家多采样）
        norm_pos = np.zeros((n, 3), dtype=np.float32)
        for p in range(n):
            nx, ny, nz = normalize_position(pos_game[p, 0], pos_game[p, 1], pos_game[p, 2], map_name)
            norm_pos[p] = (nx, ny, nz)

        pos_t = torch.from_numpy(norm_pos).to(self.device)            # [N, 3]
        map_id = MAP_NAME_TO_IDX.get(map_name, 0)
        map_emb = self.model.embedder.map_emb(
            torch.full((n,), map_id, dtype=torch.long, device=self.device)
        )  # [N, d]

        mlp_in = torch.cat([pos_t, map_emb], dim=-1)                  # [10, 3+d]
        out = self.model.embedder.mlp1(mlp_in)                         # [10, d]
        # adapter 由 decoder 内部调用，这里不重复
        return out

    def _compute_angle_emb(self, yaw_deg: np.ndarray, pitch_deg: np.ndarray) -> torch.Tensor:
        """Encode absolute yaw/pitch (degrees) → [10, d] using mlp_angle."""
        import math as _math
        yaw_rad = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)

        angle_in = np.stack([
            np.cos(yaw_rad), np.sin(yaw_rad),
            np.cos(pitch_rad), np.sin(pitch_rad),
        ], axis=-1)  # [10, 4]
        angle_t = torch.from_numpy(angle_in.astype(np.float32)).to(self.device)
        return self.model.embedder.mlp_angle(angle_t)                  # [10, d]

    # ── 下游任务（winrate / future_kill / alive_end）─────────────────────

    def load_downstream_checkpoint(self, checkpoint_path: str) -> str:
        """加载下游任务完整微调 checkpoint（merged 权重 + 线性头）。

        checkpoint 格式（finetune_lora.py save_finetune_ckpt）:
          {task, n_ticks, pred_pos, global_step,
           peft_state: 完整模型 state_dict（全量微调 / LoRA 已 merge 的权重）,
           head_state: {fc.weight, fc.bias}}
        下游权重只用于**指标计算**；路径预测始终使用预训练底座模型。
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        task = ckpt.get("task")
        if task not in DOWNSTREAM_TASKS:
            raise ValueError(f"未知下游任务: {task!r}")
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["peft_state"].items()}
        self.apply_downstream_state(state, ckpt["head_state"], task,
                                    step=ckpt.get("global_step"))
        return task

    def apply_downstream_state(self, state: dict, head_state: dict, task: str,
                               step=None) -> None:
        """缓存下游任务 merged 权重 + 线性头（仅用于指标计算）。

        不修改 self.model：路径预测始终用预训练底座，指标曲线在计算时
        才用下游模型（_ensure_down_model 懒加载的独立实例）前向。
        """
        self.downstream_state = state
        self.downstream_head_state = head_state
        self.downstream_task = task
        self.downstream_step = step
        # 下次指标计算时重建/重载下游模型实例
        self._down_model = None
        self._down_head = None
        self._down_model_task = None
        print(f"[PredictionEngine] downstream task={task} step={step} "
              f"head=({head_state['fc.weight'].shape[1]}→1) "
              f"（仅指标；路径用预训练底座）")

    def clear_downstream(self):
        """清除下游任务状态（路径预测不受影响，始终是预训练底座）。"""
        self.downstream_task = None
        self.downstream_step = None
        self.downstream_state = None
        self.downstream_head_state = None
        self._down_model = None
        self._down_head = None
        self._down_model_task = None

    def _ensure_down_model(self, task: str):
        """懒加载 / 按需切换指标计算专用下游模型实例。

        单实例复用：任务切换时 load_state_dict 换权重（247MB，仅切换时发生）。
        Returns: (down_model, down_head)
        """
        if self.downstream_state is None or self.downstream_task != task:
            raise ValueError(
                f"下游任务 {task} 未加载（请先 apply_downstream_state / "
                f"load_downstream_checkpoint）")
        if self._down_model is not None and self._down_model_task == task:
            return self._down_model, self._down_head

        if self._down_model is None:
            self._down_model = CS2PretrainModel(self.model_cfg).to(self.device)
        missing, unexpected = self._down_model.load_state_dict(
            self.downstream_state, strict=False)
        if missing:
            print(f"[PredictionEngine] downstream missing keys: {missing}")
        if unexpected:
            print(f"[PredictionEngine] downstream unexpected keys: {unexpected}")
        self._down_model.eval()

        if self._down_head is None:
            self._down_head = _TaskHead(self.model_cfg.d_model).to(self.device).eval()
        self._down_head.load_state_dict(self.downstream_head_state)

        self._down_model_task = task
        return self._down_model, self._down_head

    @torch.no_grad()
    def _task_metric_forward(
        self, conditions, tokens, depth_ctx, xyz_ctx, angle_ctx,
        model, head,
    ) -> np.ndarray:
        """decoder 全前向 → 预测点 hidden → 线性头 → sigmoid 概率。

        conditions: [N, d]            条件 embedding（cond 时刻，须来自 model）
        tokens:     [N, n_ticks*7]    路径 token（GT 或 AR 生成）
        depth/xyz/angle_ctx: [N, n_ticks, d] 或 None（训练同款逐 tick 条件）
        model/head: 计算用的模型与线性头（指标场景传下游模型）

        Returns: probs [N, n_points]（n_points = n_ticks + 1）
        """
        n_ticks = self.model_cfg.n_ticks
        tpg = self.model.tokenizer.TOKENS_PER_GROUP  # 10
        pred_pos = [0] + [k * tpg + 9 for k in range(n_ticks)]
        _, _, hidden = model.decoder(
            conditions, tokens,
            depth_ctx=depth_ctx, xyz_ctx=xyz_ctx, angle_ctx=angle_ctx,
            return_hidden=True,
        )  # [N, seq_len, d]
        logits = head(hidden[:, pred_pos, :])   # [N, n_points]
        return torch.sigmoid(logits).cpu().numpy()

    def _encode_metric_raw_ctx(self, depth_raw_list, xyz_raw_list, angle_raw_list,
                               map_name, model, n_rows):
        """把 AR 逐 tick 捕获的**原始条件**编码成 [n_rows, n_ticks, d] 预-adapter 张量。

        与训练 / GT（_build_gt_task_contexts）同款编码：
          depth: embedder.depth_encoder（[*, 64, 5] → d）
          xyz:   normalize_position + map_emb + mlp1
          angle: cos/sin(yaw,pitch) → mlp_angle
        在指定 model（指标场景为下游模型）的 embedder 上执行。
        None 条目（整 tick 全员死亡 / 无地图几何）补零，与旧 _stack 行为一致。

        depth_raw_list:  [n_ticks] of [n_rows, 64, 5] tensor | None
        xyz_raw_list:    [n_ticks] of [n_rows, 3] np（游戏坐标）| None
        angle_raw_list:  [n_ticks] of (yaw_deg[n_rows], pitch_deg[n_rows]) | None
        Returns: (depth_t, xyz_t, angle_t) — [n_rows, n_ticks, d] device tensors
        """
        n_ticks = self.model_cfg.n_ticks
        d = self.model_cfg.d_model
        device = self.device
        emb = model.embedder

        # ── depth ──
        depth_ticks = []
        for raw in depth_raw_list:
            if raw is None:
                raw = torch.zeros(n_rows, 64, 5, device=device)
            else:
                raw = raw.to(device)
            depth_ticks.append(raw)
        depth_flat = torch.stack(depth_ticks, dim=1).reshape(n_rows * n_ticks, 64, 5)
        depth_t = emb.depth_encoder(depth_flat).reshape(n_rows, n_ticks, d)

        # ── xyz（游戏坐标 → 归一化 → mlp1(+map_emb)）──
        xyz_ticks = []
        for raw in xyz_raw_list:
            if raw is None:
                raw = np.zeros((n_rows, 3), dtype=np.float32)
            xyz_ticks.append(raw)
        xyz_np = np.stack(xyz_ticks, axis=1)            # [n_rows, n_ticks, 3]
        map_id = MAP_NAME_TO_IDX.get(map_name, 0)
        norm_flat = np.zeros((n_rows * n_ticks, 3), dtype=np.float32)
        for i in range(n_rows * n_ticks):
            r, t = divmod(i, n_ticks)
            nx, ny, nz = normalize_position(
                xyz_np[r, t, 0], xyz_np[r, t, 1], xyz_np[r, t, 2], map_name)
            norm_flat[i] = (nx, ny, nz)
        pos_t = torch.from_numpy(norm_flat).to(device)
        map_emb = emb.map_emb(
            torch.full((n_rows * n_ticks,), map_id, dtype=torch.long, device=device))
        xyz_t = emb.mlp1(torch.cat([pos_t, map_emb], dim=-1)).reshape(n_rows, n_ticks, d)

        # ── angle ──
        ang_ticks = []
        for raw in angle_raw_list:
            if raw is None:
                ang_ticks.append(np.zeros((n_rows, 4), dtype=np.float32))
            else:
                yaw_deg, pitch_deg = raw
                yaw_r = np.radians(yaw_deg)          # float64
                pitch_r = np.radians(pitch_deg)
                ang_ticks.append(np.stack([
                    np.cos(yaw_r), np.sin(yaw_r),
                    np.cos(pitch_r), np.sin(pitch_r)], axis=-1))
        ang_np = np.stack(ang_ticks, axis=1).reshape(n_rows * n_ticks, 4)
        angle_t = emb.mlp_angle(
            torch.from_numpy(ang_np.astype(np.float32)).to(device)
        ).reshape(n_rows, n_ticks, d)

        return depth_t, xyz_t, angle_t

    def _build_gt_task_contexts(self, torch_sample, query_tick, map_name,
                                player_idx: Optional[int] = None,
                                model=None):
        """训练同款 GT 条件上下文：窗口 ticks q..q+n-1 的 depth/xyz/angle。

        player_idx=None → 全部玩家 [10, n_ticks, d]；否则单个玩家 [1, n_ticks, d]。
        窗口不足 n_ticks 时与训练一致补零（这些位置的输出会被 mask 掉）。
        model: 编码所用模型（指标场景传下游模型），默认 self.model。
        Returns: (depth_t, xyz_t, angle_t) — device tensors，可能为 None
        """
        model = model or self.model
        n_ticks = self.model_cfg.n_ticks
        T = torch_sample["player_pos"].shape[0]
        end = min(query_tick + n_ticks, T)
        n = end - query_tick
        device = self.device
        d = self.model_cfg.d_model

        if player_idx is None:
            P = N_PLAYERS
            sel = slice(None)
        else:
            P = 1
            sel = slice(player_idx, player_idx + 1)

        def _pad(slice_arr, pad_after):
            if pad_after <= 0:
                return slice_arr
            z = torch.zeros((pad_after,) + slice_arr.shape[1:], dtype=slice_arr.dtype)
            return torch.cat([slice_arr, z], dim=0)

        depth_t = None
        if "player_depth" in torch_sample:
            depth = torch_sample["player_depth"]          # [T, 10, 64, 5]
            d_slice = _pad(depth[query_tick:end][:, sel], n_ticks - n)
            d_flat = d_slice.permute(1, 0, 2, 3).contiguous().view(P * n_ticks, 64, 5)
            d_enc = model.embedder.depth_encoder(d_flat.to(device))
            depth_t = d_enc.reshape(P, n_ticks, d)

        pos = torch_sample["player_pos"]                  # [T, 10, 3] 归一化坐标
        p_slice = _pad(pos[query_tick:end][:, sel], n_ticks - n)
        p_flat = p_slice.reshape(P * n_ticks, 3).to(device)
        map_id = MAP_NAME_TO_IDX.get(map_name, 0)
        map_emb = model.embedder.map_emb(
            torch.full((P * n_ticks,), map_id, dtype=torch.long, device=device))
        xyz_t = model.embedder.mlp1(torch.cat([p_flat, map_emb], dim=-1))
        xyz_t = xyz_t.reshape(P, n_ticks, d)

        state = torch_sample["player_state"]              # [T, 10, 14]
        # 角度标签 = [cos(yaw), sin(yaw), cos(pitch), sin(pitch)]（与训练一致）
        a_slice = _pad(state[query_tick:end][:, sel][..., [8, 9, 6, 7]], n_ticks - n)
        a_flat = a_slice.reshape(P * n_ticks, 4).to(device)
        angle_t = model.embedder.mlp_angle(a_flat).reshape(P, n_ticks, d)

        return depth_t, xyz_t, angle_t

    def _downstream_labels(self, metric, torch_sample, query_tick, teams) -> List[float]:
        """每个玩家的锚定目标值（0/1）。

        winrate 显示为 P(该玩家队伍获胜)：CT 玩家 = 1 - label，T 玩家 = label。
        """
        if metric == "winrate":
            wr = float(torch_sample["label_winrate"][0]) \
                if "label_winrate" in torch_sample else 0.0
            return [wr if t == "T" else (1.0 - wr) if t == "CT" else wr
                    for t in teams]
        if metric == "alive_end":
            ae = torch_sample["label_alive_end"]          # [T, 10]
            return [float(ae[0, p]) for p in range(N_PLAYERS)]
        if metric == "future_kill":
            nk = torch_sample.get("label_nxt_kill")
            nd = torch_sample.get("label_nxt_death")
            last = _last_kill_ticks(
                nk.numpy() if nk is not None else None,
                nd.numpy() if nd is not None else None,
            )
            # cond 锚定：cond 之后任意时刻（含最后一个 tick 之后）获得击杀 = 1
            return [1.0 if last[p] > query_tick else 0.0 for p in range(N_PLAYERS)]
        raise ValueError(f"未知指标: {metric}")

    @staticmethod
    def _flip_winrate(probs: np.ndarray, teams) -> np.ndarray:
        """head 输出 P(T 胜) → 显示 P(该玩家队伍胜)（CT 玩家取 1-p）。"""
        out = probs.copy()
        for p in range(probs.shape[0]):
            if teams[p] == "CT":
                out[p] = 1.0 - out[p]
        return out

    @staticmethod
    def _json_curve(probs_1d, mask_1d) -> dict:
        """单条曲线 → JSON（masked 位置 probs=None）。"""
        return {
            "probs": [None if not mask_1d[j] else float(probs_1d[j])
                      for j in range(len(probs_1d))],
            "mask": [bool(mask_1d[j]) for j in range(len(mask_1d))],
        }

    def _score_generated_logp(self, logp_pt, cont, initial_alive) -> List[Optional[dict]]:
        """对 AR 生成的预测路径计算模型自评分（与 GT 的 pre_tick 同口径）。

        Args:
            logp_pt:       [10, n_ticks, 7] 每 token 的 log p（未缩放分布）
            cont:          [10, n_ticks] 每 tick 的 continue token id
            initial_alive: [10] cond 时刻是否存活

        有效 tick 判定与 GT 编码一致：停止 tick（continue != CONTINUE_1）本身
        计入，其后的 tick 排除（PAD 语义）；玩家 cond 时刻死亡 → None。

        Returns:
            [10] dict 或 None（死亡玩家）：
              per_tick = 有效 tick 的每 tick 每 token 平均 log p 的 tick 等权均值
              total    = 有效 token 的 log p 之和
              tokcount = 有效 token 数
              ticks    = 有效 tick 数
        """
        import numpy as _np
        n_ticks = cont.shape[1]
        c1 = self.model.tokenizer.CONTINUE_1
        scores: List[Optional[dict]] = [None] * N_PLAYERS
        for p in range(N_PLAYERS):
            if not initial_alive[p]:
                continue
            c = cont[p]                                   # [n_ticks]
            stopped = _np.flatnonzero(c != c1)
            # 有效 tick = 0..first_stop（含停止 tick）；无停止 → 全部
            n_valid = int(stopped[0]) + 1 if stopped.size else n_ticks
            valid = _np.arange(n_ticks) < n_valid
            lp = logp_pt[p][valid]                        # [n_valid, 7]
            tick_means = lp.mean(axis=-1)                 # [n_valid] 每 tick 均值
            scores[p] = {
                "per_tick": float(tick_means.mean()),
                "total": float(lp.sum()),
                "tokcount": int(valid.sum() * lp.shape[1]),
                "ticks": int(valid.sum()),
            }
        return scores

    # ── 主推理方法 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_at_tick(
        self,
        sample: dict,
        query_tick: int,
        temperature: float = 0.0,
        teacher_forcing_ticks: int = 0,
        metric: Optional[str] = None,
        return_logp: bool = False,
    ) -> dict:
        """在指定 tick 运行推理，返回预测轨迹 vs 真值。

        Args:
            sample:             解码后的 round-level sample（numpy 数组）
            query_tick:         当前 timeline tick 位置（0-indexed）
            teacher_forcing_ticks: 前 N 个 tick 用 GT token 替代 AR 生成
            metric:             下游指标（"winrate"/"future_kill"/"alive_end"），
                                非 None 时结果附带 metrics。路径预测始终用预训练底座；
                                指标曲线由下游模型（apply_downstream_state 提供）评估。
            return_logp:        同时计算模型对自身 AR 预测路径的自评分
                                （未缩放分布下生成 token 的 log p），附加到每个
                                trajectory.pred_logp：
                                  per_tick = tick 等权平均（停止 tick 计入，其后排除，
                                             与 GT 路径的 pre_tick 同口径）
                                  total    = 有效 token 的 log p 之和
                                  tokcount = 有效 token 数
                                  ticks    = 有效 tick 数
                                死亡玩家为 None。

        Returns:
            {query_tick, input_T, output_T, map_name, trajectories: [...],
             metrics: {...}|None}
        """
        from training_data.map_loader import get_map_geometry

        meta = sample.get("meta", {})
        round_T = meta.get("T", 0)
        map_name = meta.get("map_name", "unknown")
        tick_interval = meta.get("tick_interval", 0.25)
        n_ticks = self.model_cfg.n_ticks

        # ── 检测标签坐标系版本 ─────────────────────────────
        source_format = meta.get("format", "")
        local_labels_are_v4 = not source_format.startswith("cs2.training.v5") and \
                              not source_format.startswith("cs2.training.v6") and \
                              not source_format.startswith("cs2.training.v7") and \
                              not source_format.startswith("cs2.training.v8") and \
                              not source_format.startswith("cs2.training.v9")

        # ── 1. 预处理 ───────────────────────────────────
        if "player_depth" in sample and sample["player_depth"].ndim == 3:
            sample = augment_depth_with_angles(sample)
        torch_sample = sample_to_torch(sample)

        # ── 2. 输入窗口 ─────────────────────────────────
        input_end = min(query_tick + 1, round_T)
        input_len = min(n_ticks, input_end)
        input_start = input_end - input_len
        batch = self._build_batch(torch_sample, input_start, input_end, tick_interval)

        player_emb = self.model.get_player_embeddings(batch)  # [1, T_input, 10, d]
        T_input = player_emb.shape[1]
        conditions = player_emb[0, -1, :, :].to(self.device)  # [10, d]

        # ── 下游指标：收集 AR 逐 tick 原始条件（指标阶段用下游模型重新编码）──
        metric_ctx = None
        if metric is not None:
            if metric not in DOWNSTREAM_TASKS:
                raise ValueError(f"未知指标: {metric}")
            metric_ctx = {
                "depth_raw_list": [],
                "xyz_raw_list": [],
                "angle_raw_list": [],
                "tokens_list": [],
            }

        # ── 3. 起始状态 ─────────────────────────────────
        start_idx = input_end - 1
        pos_game, yaw_deg, pitch_deg = self._get_starting_state(
            torch_sample, start_idx, map_name)
        alive_arr = torch_sample["player_alive_mask"][query_tick].bool().numpy()
        initial_alive = alive_arr.copy()  # snapshot for trajectory output
        initial_pos = pos_game.copy()     # snapshot before loop modifies it
        initial_yaw = yaw_deg.copy()
        initial_pitch = pitch_deg.copy()

        # ── 4. 地图几何（用于 raycast depth）────────────
        map_geom = None
        try:
            map_geom = get_map_geometry(map_name, self.maps_dir)
        except FileNotFoundError:
            pass

        # ── 5. 准备 GT labels（用于 TF + 最终对比）────────
        label_camera = torch_sample["label_camera"]  # [round_T, 10, 10]
        gt_end = min(query_tick + n_ticks, round_T)
        gt_labels_10d = label_camera[query_tick:gt_end]  # [≤n_ticks, 10, 10]
        gt_available = gt_labels_10d.shape[0]
        if gt_available < n_ticks:
            gt_labels_padded = torch.cat([
                gt_labels_10d,
                torch.zeros(n_ticks - gt_available, 10, 10)
            ], dim=0)
        else:
            gt_labels_padded = gt_labels_10d[:n_ticks]
        gt_labels_orig = gt_labels_padded.permute(1, 0, 2).clone()  # [10, n_ticks, 10]

        if local_labels_are_v4:
            gt_labels_v5 = self._convert_labels_v4_to_v5(gt_labels_orig, pitch_deg, n_ticks)
        else:
            gt_labels_v5 = gt_labels_orig

        gt_tokens_all = self.model.tokenizer.encode_sequence(gt_labels_v5, n_ticks)
        gt_tokens_per_tick = gt_tokens_all.reshape(N_PLAYERS, n_ticks, self.model.tokenizer.TOKENS_PER_TICK)

        # ── 6. 逐 tick 生成 ──────────────────────────────
        tf_ticks = min(teacher_forcing_ticks, n_ticks)
        tpg = self.model.tokenizer.TOKENS_PER_GROUP  # 10

        decoder_x = self.model.decoder.init_generate(conditions)
        kv_cache = self.model.decoder.new_kv_cache(conditions.shape[0], self.device)
        self.model.decoder.seed_cache(decoder_x, kv_cache)
        all_preds_list = []

        # return_logp：收集每 tick 每个生成 token 的 log p 与 continue token
        # （用于对模型自身预测路径打分，与 GT 的 pre_tick 口径对齐）
        pred_logp_tokens = None
        pred_continue = None
        if return_logp:
            tpt = self.model.tokenizer.TOKENS_PER_TICK
            pred_logp_tokens = torch.zeros(
                N_PLAYERS, n_ticks, tpt, device=self.device)
            pred_continue = torch.zeros(
                N_PLAYERS, n_ticks, dtype=torch.long, device=self.device)

        for tick in range(n_ticks):
            is_tf = tick < tf_ticks

            if metric_ctx is not None:
                depth_emb, depth_raw = self._compute_depth_emb(
                    map_geom, pos_game, yaw_deg, pitch_deg, alive_arr,
                    return_raw=True)
            else:
                depth_emb = self._compute_depth_emb(
                    map_geom, pos_game, yaw_deg, pitch_deg, alive_arr)

            # ── Compute xyz/angle conditioning embeddings ──
            xyz_emb = self._compute_xyz_emb(pos_game, map_name) if alive_arr.any() else None
            angle_emb = self._compute_angle_emb(yaw_deg, pitch_deg) if alive_arr.any() else None

            # 下游指标：收集 adapter / encoder 之前的原始条件
            # （路径由预训练底座生成，指标阶段需用下游模型的 encoder 重新编码）
            if metric_ctx is not None:
                metric_ctx["depth_raw_list"].append(depth_raw)
                metric_ctx["xyz_raw_list"].append(
                    pos_game.copy() if xyz_emb is not None else None)
                metric_ctx["angle_raw_list"].append(
                    (yaw_deg.copy(), pitch_deg.copy()) if angle_emb is not None else None)

            # ── Apply decoder adapters (project shared encoder → decoder token space) ──
            # 训练时 TokenDecoder.forward 和 generate() 都会过 adapter；
            # predict_at_tick 直接调 generate_group，需要在此手动过一遍。
            dec = self.model.decoder
            if depth_emb is not None:
                depth_emb = dec.depth_dec_adapter(depth_emb)
            if xyz_emb is not None:
                xyz_emb = dec.xyz_dec_adapter(xyz_emb)
            if angle_emb is not None:
                angle_emb = dec.angle_dec_adapter(angle_emb)

            if is_tf:
                tf_cam_tokens = gt_tokens_per_tick[:, tick, :].to(self.device)  # [10, 7]

                # Write 3 conditioning tokens: depth(tick*10+1), xyz(tick*10+2), angle(tick*10+3)
                for offset, emb in [(1, depth_emb), (2, xyz_emb), (3, angle_emb)]:
                    seq_pos = tick * tpg + offset
                    if seq_pos < self.model.decoder.seq_len:
                        tok = (emb if emb is not None
                               else torch.zeros(N_PLAYERS, self.model_cfg.d_model, device=self.device))
                        decoder_x[:, seq_pos:seq_pos + 1, :] = \
                            tok.unsqueeze(1) + self.model.decoder.pos_encoding[:, seq_pos:seq_pos + 1, :]

                # Write 7 camera tokens starting at tick*10+4
                for step in range(self.model.tokenizer.TOKENS_PER_TICK):
                    write_pos = tick * tpg + 4 + step
                    if write_pos < self.model.decoder.seq_len:
                        emb = self.model.decoder.token_emb(tf_cam_tokens[:, step]).unsqueeze(1)
                        decoder_x[:, write_pos:write_pos + 1, :] = \
                            emb + self.model.decoder.pos_encoding[:, write_pos:write_pos + 1, :]

                all_preds_list.append(gt_labels_v5[:, tick, :].cpu().numpy()[:, np.newaxis, :])
                if metric_ctx is not None:
                    metric_ctx["tokens_list"].append(tf_cam_tokens)
                if tick < n_ticks - 1 and alive_arr.any():
                    self._apply_delta(pos_game, yaw_deg, pitch_deg, alive_arr,
                                      gt_labels_orig.numpy(), tick,
                                      camera_relative_up=local_labels_are_v4)
            else:
                if return_logp:
                    tick_tokens, decoder_x, tick_logp = \
                        self.model.decoder.generate_group(
                            decoder_x, tick, depth_emb=depth_emb,
                            xyz_emb=xyz_emb, angle_emb=angle_emb,
                            temperature=temperature,
                            kv_cache=kv_cache,
                            return_logp=True,
                        )
                    pred_logp_tokens[:, tick, :] = tick_logp
                    pred_continue[:, tick] = tick_tokens[:, 0]
                else:
                    tick_tokens, decoder_x = self.model.decoder.generate_group(
                        decoder_x, tick, depth_emb=depth_emb,
                        xyz_emb=xyz_emb, angle_emb=angle_emb,
                        temperature=temperature,
                        kv_cache=kv_cache,
                    )
                tick_10d = self.model.tokenizer.decode_sequence(
                    tick_tokens, 1,
                ).cpu().numpy()  # [10, 1, 10]
                all_preds_list.append(tick_10d)
                if metric_ctx is not None:
                    metric_ctx["tokens_list"].append(tick_tokens)

                # 诊断摘要（每 tick 一行，pred 只统计窗口开始时活着的玩家）
                alive_count = int(alive_arr.sum())
                end_vals = tick_10d[:, 0, 9]  # [10] — 1=继续, 0=停止
                alive_vals = tick_10d[:, 0, 7]  # [10] — 1=存活, 0=死亡
                n_continue = int((end_vals[initial_alive] > 0.5).sum())
                n_alive_pred = int((alive_vals[initial_alive] > 0.5).sum())
                n_total = int(initial_alive.sum())
                print(f"  [AR tick={tick}] alive={alive_count}/{n_total} continue={n_continue}/{n_total} "
                      f"alive_pred={n_alive_pred}/{n_total}")

                if tick < n_ticks - 1 and alive_arr.any():
                    self._apply_delta(pos_game, yaw_deg, pitch_deg, alive_arr,
                                      tick_10d, 0, camera_relative_up=False)

        all_preds_np = np.concatenate(all_preds_list, axis=1)  # [10, n_ticks, 10]

        # ── 6b. 模型自评分（return_logp）：AR 预测路径的 log p ─────────
        pred_logp_scores = None
        if return_logp and pred_logp_tokens is not None:
            pred_logp_scores = self._score_generated_logp(
                pred_logp_tokens.cpu().numpy(),
                pred_continue.cpu().numpy(),
                initial_alive,
            )

        # ── 7. GT labels 用于对比 ────────────────────────
        gt_labels = label_camera[query_tick:gt_end].numpy()  # [≤n_ticks, 10, 10]

        # ── 8. 积分轨迹 ──────────────────────────────────
        trajectories = []
        for p in range(N_PLAYERS):
            start_pos = (float(initial_pos[p, 0]), float(initial_pos[p, 1]), float(initial_pos[p, 2]))
            start_yaw = math.radians(initial_yaw[p])
            start_pitch = math.radians(initial_pitch[p])

            if not initial_alive[p]:
                # Dead at query tick: no prediction, GT also empty
                trajectories.append({
                    "player_idx": p,
                    "is_alive": False,
                    "start_pos": list(start_pos),
                    "pred_points": [],
                    "gt_points": [],
                    "pred_yaw": [],
                    "pred_pitch": [],
                    "gt_yaw": [],
                    "gt_pitch": [],
                    "pred_alive": [],
                    "gt_alive": [],
                    "pred_firing": [],
                    "gt_firing": [],
                    "pred_steps": 0,
                    "gt_steps": 0,
                    "end_stopped_at": 0,
                    "teacher_forcing_steps": 0,
                    "pred_logp": None,
                })
                continue

            pred_deltas = all_preds_np[p]  # [n_ticks, 10]
            pred_points, pred_alive, pred_firing, end_stop, pred_yaws, pred_pitches = integrate_trajectory(
                start_pos, start_yaw, start_pitch, pred_deltas,
                start_step=0, max_steps=n_ticks,
                camera_relative_up=False,  # model outputs v5
                return_angles=True,
            )

            gt_deltas = gt_labels[:, p, :]  # [≤n_ticks, 10]
            gt_points, gt_alive, gt_firing, _, gt_yaws, gt_pitches = integrate_trajectory(
                start_pos, start_yaw, start_pitch, gt_deltas,
                start_step=0, max_steps=gt_available,
                camera_relative_up=local_labels_are_v4,
                return_angles=True,
            )

            trajectories.append({
                "player_idx": p,
                "is_alive": bool(initial_alive[p]),
                "start_pos": list(start_pos),
                "pred_points": [list(pt) for pt in pred_points],
                "gt_points": [list(pt) for pt in gt_points],
                # integrate_trajectory 返回弧度，转成度（前端与玩家 state.yaw 同为度）
                "pred_yaw": [math.degrees(y) for y in pred_yaws],
                "pred_pitch": [math.degrees(x) for x in pred_pitches],
                "gt_yaw": [math.degrees(y) for y in gt_yaws],
                "gt_pitch": [math.degrees(x) for x in gt_pitches],
                "pred_alive": pred_alive,
                "gt_alive": gt_alive,
                "pred_firing": pred_firing,
                "gt_firing": gt_firing,
                "pred_steps": len(pred_points),
                "gt_steps": len(gt_points),
                "end_stopped_at": end_stop,
                "teacher_forcing_steps": min(tf_ticks, len(pred_points)),
                "pred_logp": (
                    pred_logp_scores[p] if pred_logp_scores is not None else None
                ),
            })

        # ── 9. 下游指标（GT teacher-forcing + 预测路径）──────────────────
        # 路径由预训练底座生成；指标用下游模型（独立实例）在生成路径上评估
        metrics_payload = None
        if metric_ctx is not None:
            teams = meta.get("teams") or ["?"] * N_PLAYERS
            down, down_head = self._ensure_down_model(metric)

            # 下游模型自己的条件 embedding（路径生成仍用预训练底座的 conditions）
            down_conditions = down.get_player_embeddings(batch)[0, -1, :, :].to(self.device)
            gt_ctx = self._build_gt_task_contexts(
                torch_sample, query_tick, map_name, model=down)

            # GT 曲线：真实路径 token + 训练同款条件 → 下游模型 head
            gt_probs = self._task_metric_forward(
                down_conditions, gt_tokens_all.to(self.device), *gt_ctx,
                model=down, head=down_head)

            # 预测曲线：预训练底座 AR 生成的 token + 生成路径逐 tick 原始条件
            #            → 下游模型 encoder 重新编码 → 下游模型 head
            N_ = conditions.shape[0]
            depth_t, xyz_t, angle_t = self._encode_metric_raw_ctx(
                metric_ctx["depth_raw_list"], metric_ctx["xyz_raw_list"],
                metric_ctx["angle_raw_list"], map_name, down, N_)
            pred_tokens = torch.cat(metric_ctx["tokens_list"], dim=1)   # [N_, n_ticks*7]
            pred_probs = self._task_metric_forward(
                down_conditions, pred_tokens, depth_t, xyz_t, angle_t,
                model=down, head=down_head)

            # mask：沿路径存活的位置（j=0 → cond 存活；j>=1 → 第 j-1 步存活）
            n_points = n_ticks + 1
            gt_mask = np.zeros((N_PLAYERS, n_points), dtype=bool)
            pred_mask = np.zeros((N_PLAYERS, n_points), dtype=bool)
            for traj in trajectories:
                p = traj["player_idx"]
                if not traj["is_alive"]:
                    continue
                gt_mask[p, 0] = True
                pred_mask[p, 0] = True
                ga = traj["gt_alive"]
                pa = traj["pred_alive"]
                for j in range(1, n_points):
                    if j - 1 < len(ga) and ga[j - 1] > 0.5:
                        gt_mask[p, j] = True
                    if j - 1 < len(pa) and pa[j - 1] > 0.5:
                        pred_mask[p, j] = True

            if metric == "winrate":
                gt_probs = self._flip_winrate(gt_probs, teams)
                pred_probs = self._flip_winrate(pred_probs, teams)
            labels = self._downstream_labels(metric, torch_sample, query_tick, teams)

            def _players(curves_probs, curves_mask):
                return {"players": [
                    self._json_curve(curves_probs[p], curves_mask[p])
                    for p in range(N_PLAYERS)
                ]}

            metrics_payload = {
                "task": metric,
                "task_label": TASK_LABELS.get(metric, metric),
                "n_points": n_points,
                "labels": [float(x) for x in labels],
                "teams": [str(t) for t in teams],
                "gt": _players(gt_probs, gt_mask),
                "pred": _players(pred_probs, pred_mask),
            }

        return {
            "query_tick": query_tick,
            "input_T": T_input,
            "output_T": min(n_ticks, round_T - query_tick),
            "map_name": map_name,
            "trajectories": trajectories,
            "metrics": metrics_payload,
        }

    @torch.no_grad()
    def predict_at_tick_sampled(
        self,
        sample: dict,
        query_tick: int,
        num_samples: int = 4,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> dict:
        """
        对每个玩家采样多条轨迹（从 angle-grid 分布中采样而非 argmax）。

        Args:
            sample:      解码后的 round sample
            query_tick:  当前 tick 位置
            num_samples: 每个玩家采样轨迹数量
            temperature: softmax 温度（>1=更多样，<1=更确定）
            top_k:       >0 时只从 top-k bins 中采样

        Returns:
            {query_tick, trajectories_by_sample: [[traj, ...], ...], ...}
        """
        meta = sample.get("meta", {})
        round_T = meta.get("T", 0)
        map_name = meta.get("map_name", "unknown")
        tick_interval = meta.get("tick_interval", 0.25)

        if "player_depth" in sample and sample["player_depth"].ndim == 3:
            sample = augment_depth_with_angles(sample)

        torch_sample = sample_to_torch(sample)
        input_end = min(query_tick + 1, round_T)
        n_ticks = self.model_cfg.n_ticks
        input_len = min(n_ticks, input_end)
        input_start = input_end - input_len

        batch = self._build_batch(torch_sample, input_start, input_end, tick_interval)
        player_emb = self.model.get_player_embeddings(batch)
        T_input = player_emb.shape[1]

        last_emb = player_emb[0, -1, :, :]  # [10, d]
        conditions = last_emb.to(self.device)

        start_idx = input_end - 1
        player_pos = torch_sample["player_pos"]
        player_state = torch_sample["player_state"]
        alive_mask = torch_sample["player_alive_mask"][query_tick].bool()

        base_positions = np.zeros((N_PLAYERS, 3), dtype=np.float64)
        base_yaws = np.zeros(N_PLAYERS, dtype=np.float64)
        base_pitches = np.zeros(N_PLAYERS, dtype=np.float64)
        alive_arr = np.zeros(N_PLAYERS, dtype=bool)
        for p in range(N_PLAYERS):
            nx, ny, nz = player_pos[start_idx, p].tolist()
            gx, gy, gz = denormalize_position(nx, ny, nz, map_name)
            base_positions[p] = (gx, gy, gz)
            state = player_state[start_idx, p].numpy()
            yaw, pitch = _extract_yaw_pitch(state)
            base_yaws[p] = math.degrees(yaw)
            base_pitches[p] = math.degrees(pitch)
            alive_arr[p] = bool(alive_mask[p])

        # Build per-tick depth context
        depth_ctx = None
        if "player_depth" in torch_sample:
            depth_data = torch_sample["player_depth"]                     # [round_T, 10, 64, 5]
            d_slice = depth_data[query_tick:query_tick + n_ticks]
            if d_slice.shape[0] == 0:
                d_slice = depth_data[-1:].expand(n_ticks, -1, -1, -1)
            elif d_slice.shape[0] < n_ticks:
                pad_len = n_ticks - d_slice.shape[0]
                d_slice = torch.cat([d_slice, d_slice[-1:].expand(pad_len, -1, -1, -1)], dim=0)
            d_flat = d_slice.permute(1, 0, 2, 3).contiguous().view(10 * n_ticks, 64, 5)
            d_enc = self.model.embedder.depth_encoder(d_flat.to(self.device))
            depth_ctx = d_enc.reshape(10, n_ticks, self.model_cfg.d_model)

        # Build static xyz/angle context (same start pos/angle repeated; generate() applies adapters)
        xyz_ctx = None
        angle_ctx = None
        if alive_arr.any():
            xyz_emb = self._compute_xyz_emb(base_positions, map_name)          # [10, d]
            xyz_ctx = xyz_emb.unsqueeze(1).expand(10, n_ticks, self.model_cfg.d_model)
            angle_emb = self._compute_angle_emb(base_yaws, base_pitches)       # [10, d]
            angle_ctx = angle_emb.unsqueeze(1).expand(10, n_ticks, self.model_cfg.d_model)

        # Sample K trajectories: call generate() K times with different seeds
        trajectories_by_sample = []
        for k in range(num_samples):
            token_ids = self.model.decoder.generate(
                conditions, temperature=temperature, top_k=top_k, top_p=top_p,
                depth_ctx=depth_ctx, xyz_ctx=xyz_ctx, angle_ctx=angle_ctx,
            )
            all_10d = self.model.tokenizer.decode_sequence(
                token_ids, n_ticks,
            ).cpu().numpy()  # [10, n_ticks, 10]

            # Model outputs are v5 world-aligned (pretrain_processor rotated training labels).
            # No rotation needed — integrate_trajectory handles with camera_relative_up=False.

            trajs = []
            for p in range(N_PLAYERS):
                start_pos = (base_positions[p, 0], base_positions[p, 1], base_positions[p, 2])
                start_yaw = math.radians(base_yaws[p])
                start_pitch = math.radians(base_pitches[p])
                pred_points, pred_alive, pred_firing, end_stop, pred_yaws, pred_pitches = integrate_trajectory(
                    start_pos, start_yaw, start_pitch, all_10d[p],
                    start_step=0, max_steps=n_ticks, camera_relative_up=False,
                    return_angles=True,
                )
                trajs.append({
                    "player_idx": p,
                    "is_alive": bool(alive_mask[p]),
                    "start_pos": list(start_pos),
                    "pred_points": [list(pt) for pt in pred_points],
                    # 弧度 → 度（与 predict_at_tick 一致）
                    "pred_yaw": [math.degrees(y) for y in pred_yaws],
                    "pred_pitch": [math.degrees(x) for x in pred_pitches],
                    "pred_alive": pred_alive,
                    "pred_firing": pred_firing,
                    "pred_steps": len(pred_points),
                    "end_stopped_at": end_stop,
                })
            trajectories_by_sample.append(trajs)

        return {
            "query_tick": query_tick,
            "input_T": T_input,
            "output_T": min(n_ticks, round_T - query_tick),
            "map_name": map_name,
            "num_samples": num_samples,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "trajectories_by_sample": trajectories_by_sample,
        }

    @torch.no_grad()
    def predict_at_tick_player_sampled(
        self,
        sample: dict,
        query_tick: int,
        player_idx: int = 0,
        num_samples: int = 8,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        metric: Optional[str] = None,
    ) -> dict:
        """
        对单个玩家并行采样多条路径（decoder batch = num_samples 行）。

        与 predict_at_tick / predict_at_tick_sampled 的区别：
          - 只解码目标玩家：decoder 的 batch 行 = K 份相同条件 embedding，
            每行独立采样 token → K 条互不相关的轨迹（同起点、同历史条件）。
          - 其他 9 个玩家不进 decoder（不生成它们的轨迹，省算力）。
          - 每 tick 用各采样行自己的积分状态重算 depth/xyz/angle 条件，
            因此 K 条轨迹会因各自采样到的动作而自然发散。

        Args:
            sample:      解码后的 round-level sample（numpy 数组）
            query_tick:  当前 timeline tick 位置（0-indexed）
            player_idx:  目标玩家索引 [0, 10)
            num_samples: 采样条数（decoder batch 大小）
            temperature: softmax 温度（>0；0=argmax 时所有采样相同）
            top_k:       >0 时只从 top-k bins 中采样
            top_p:       (0,1] 时 nucleus sampling
            metric:      下游指标（非 None 时附带 metrics：K 条采样 + GT）

        Returns:
            {query_tick, input_T, output_T, map_name, player_idx,
             num_samples, temperature, top_k, top_p,
             is_alive, start_pos, samples: [{pred_points, pred_yaw,
             pred_pitch, pred_alive, pred_firing, pred_steps,
             end_stopped_at}], gt: {...}|None, metrics: {...}|None}
        """
        from training_data.map_loader import get_map_geometry

        meta = sample.get("meta", {})
        round_T = meta.get("T", 0)
        map_name = meta.get("map_name", "unknown")
        tick_interval = meta.get("tick_interval", 0.25)
        n_ticks = self.model_cfg.n_ticks

        if not (0 <= player_idx < N_PLAYERS):
            raise ValueError(f"player_idx {player_idx} out of range [0, {N_PLAYERS})")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        # ── 检测标签坐标系版本 ─────────────────────────────
        source_format = meta.get("format", "")
        local_labels_are_v4 = not source_format.startswith("cs2.training.v5") and \
                              not source_format.startswith("cs2.training.v6") and \
                              not source_format.startswith("cs2.training.v7") and \
                              not source_format.startswith("cs2.training.v8") and \
                              not source_format.startswith("cs2.training.v9")

        # ── 1. 预处理 ───────────────────────────────────
        if "player_depth" in sample and sample["player_depth"].ndim == 3:
            sample = augment_depth_with_angles(sample)
        torch_sample = sample_to_torch(sample)

        # ── 2. 输入窗口 ─────────────────────────────────
        input_end = min(query_tick + 1, round_T)
        input_len = min(n_ticks, input_end)
        input_start = input_end - input_len
        batch = self._build_batch(torch_sample, input_start, input_end, tick_interval)

        player_emb = self.model.get_player_embeddings(batch)  # [1, T_input, 10, d]
        T_input = player_emb.shape[1]

        # 目标玩家的条件 embedding 复制 K 份 → decoder batch = K
        cond = player_emb[0, -1, player_idx, :]  # [d]
        conditions = cond.unsqueeze(0).repeat(num_samples, 1)  # [K, d]

        # 下游指标：收集 AR 逐 tick 原始条件（指标阶段用下游模型重新编码）
        teams = meta.get("teams") or ["?"] * N_PLAYERS
        metric_ctx = None
        if metric is not None:
            if metric not in DOWNSTREAM_TASKS:
                raise ValueError(f"未知指标: {metric}")
            metric_ctx = {
                "depth_raw_list": [],
                "xyz_raw_list": [],
                "angle_raw_list": [],
                "tokens_list": [],
            }

        # ── 3. 起始状态（K 行相同起点）────────────────
        start_idx = input_end - 1
        player_pos = torch_sample["player_pos"]       # [round_T, 10, 3]
        player_state = torch_sample["player_state"]   # [round_T, 10, 14]
        nx, ny, nz = player_pos[start_idx, player_idx].tolist()
        gx, gy, gz = denormalize_position(nx, ny, nz, map_name)
        state = player_state[start_idx, player_idx].numpy()
        yaw, pitch = _extract_yaw_pitch(state)   # 弧度

        pos_game = np.tile(np.array([[gx, gy, gz]], dtype=np.float64), (num_samples, 1))  # [K, 3]
        yaw_deg = np.full(num_samples, math.degrees(yaw), dtype=np.float64)
        pitch_deg = np.full(num_samples, math.degrees(pitch), dtype=np.float64)
        alive_arr = np.full(num_samples, bool(
            torch_sample["player_alive_mask"][query_tick, player_idx].item()), dtype=bool)
        initial_alive = bool(alive_arr[0])
        start_pos = (float(gx), float(gy), float(gz))
        # 注意：_extract_yaw_pitch 返回弧度，integrate_trajectory 也接受弧度，直接使用（勿再转一次）
        start_yaw = yaw
        start_pitch = pitch

        # 目标玩家在 query_tick 已死亡 → 无采样轨迹
        if not initial_alive:
            return {
                "query_tick": query_tick,
                "input_T": T_input,
                "output_T": 0,
                "map_name": map_name,
                "player_idx": player_idx,
                "num_samples": num_samples,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "is_alive": False,
                "start_pos": list(start_pos),
                "samples": [],
                "gt": None,
                "metrics": None,
            }

        # ── 4. 地图几何（用于 raycast depth）────────────
        map_geom = None
        try:
            map_geom = get_map_geometry(map_name, self.maps_dir)
        except FileNotFoundError:
            pass

        # ── 5. GT（仅目标玩家，用于对比）──────────────
        label_camera = torch_sample["label_camera"]  # [round_T, 10, 10]
        gt_end = min(query_tick + n_ticks, round_T)
        gt_available = max(0, gt_end - query_tick)
        gt_deltas = label_camera[query_tick:gt_end, player_idx, :].numpy()  # [≤n_ticks, 10]

        # 下游指标：GT token（v4→v5 转换后 encode，与训练一致）
        gt_tokens_single = None
        if metric_ctx is not None and gt_available > 0:
            pad_n = n_ticks - gt_available
            if pad_n > 0:
                gt_full = np.concatenate(
                    [gt_deltas, np.zeros((pad_n, 10), dtype=np.float32)], axis=0)
            else:
                gt_full = gt_deltas[:n_ticks]
            if local_labels_are_v4:
                # 单玩家 v4→v5（与 _convert_labels_v4_to_v5 一致，pitch 逐 tick 累加）
                pitch_acc = math.degrees(pitch)
                for tick in range(n_ticks):
                    cp = math.cos(math.radians(pitch_acc))
                    sp = math.sin(math.radians(pitch_acc))
                    d_fwd = float(gt_full[tick, 0])
                    d_up = float(gt_full[tick, 2])
                    gt_full[tick, 0] = d_fwd * cp - d_up * sp
                    gt_full[tick, 2] = d_fwd * sp + d_up * cp
                    dp = math.atan2(float(gt_full[tick, 4]), float(gt_full[tick, 3]))
                    pitch_acc += math.degrees(dp)
                    pitch_acc = max(-90.0, min(90.0, pitch_acc))
            gt_v5 = torch.from_numpy(gt_full).unsqueeze(0)   # [1, n_ticks, 10]
            gt_tokens_single = self.model.tokenizer.encode_sequence(gt_v5, n_ticks)

        # ── 6. AR 生成（batch = K）──────────────────────
        decoder_x = self.model.decoder.init_generate(conditions)
        kv_cache = self.model.decoder.new_kv_cache(conditions.shape[0], self.device)
        self.model.decoder.seed_cache(decoder_x, kv_cache)
        all_preds_list = []

        for tick in range(n_ticks):
            if metric_ctx is not None:
                depth_emb, depth_raw = self._compute_depth_emb(
                    map_geom, pos_game, yaw_deg, pitch_deg, alive_arr,
                    return_raw=True)
            else:
                depth_emb = self._compute_depth_emb(
                    map_geom, pos_game, yaw_deg, pitch_deg, alive_arr)
            xyz_emb = self._compute_xyz_emb(pos_game, map_name) if alive_arr.any() else None
            angle_emb = self._compute_angle_emb(yaw_deg, pitch_deg) if alive_arr.any() else None

            # 下游指标：收集 encoder / adapter 之前的原始条件
            if metric_ctx is not None:
                metric_ctx["depth_raw_list"].append(depth_raw)
                metric_ctx["xyz_raw_list"].append(
                    pos_game.copy() if xyz_emb is not None else None)
                metric_ctx["angle_raw_list"].append(
                    (yaw_deg.copy(), pitch_deg.copy()) if angle_emb is not None else None)

            # Apply decoder adapters（与 predict_at_tick 一致）
            dec = self.model.decoder
            if depth_emb is not None:
                depth_emb = dec.depth_dec_adapter(depth_emb)
            if xyz_emb is not None:
                xyz_emb = dec.xyz_dec_adapter(xyz_emb)
            if angle_emb is not None:
                angle_emb = dec.angle_dec_adapter(angle_emb)

            tick_tokens, decoder_x = self.model.decoder.generate_group(
                decoder_x, tick,
                depth_emb=depth_emb, xyz_emb=xyz_emb, angle_emb=angle_emb,
                temperature=temperature, top_k=top_k, top_p=top_p,
                kv_cache=kv_cache,
            )
            tick_10d = self.model.tokenizer.decode_sequence(
                tick_tokens, 1,
            ).cpu().numpy()  # [K, 1, 10]
            all_preds_list.append(tick_10d)
            if metric_ctx is not None:
                metric_ctx["tokens_list"].append(tick_tokens)

            if tick < n_ticks - 1 and alive_arr.any():
                self._apply_delta(pos_game, yaw_deg, pitch_deg, alive_arr,
                                  tick_10d, 0, camera_relative_up=False)

        all_preds_np = np.concatenate(all_preds_list, axis=1)  # [K, n_ticks, 10]

        # ── 7. 积分 K 条轨迹 ──────────────────────────
        samples = []
        for k in range(num_samples):
            pred_deltas = all_preds_np[k]  # [n_ticks, 10]
            pred_points, pred_alive, pred_firing, end_stop, pred_yaws, pred_pitches = \
                integrate_trajectory(
                    start_pos, start_yaw, start_pitch, pred_deltas,
                    start_step=0, max_steps=n_ticks,
                    camera_relative_up=False,  # model outputs v5
                    return_angles=True,
                )
            samples.append({
                "pred_points": [list(pt) for pt in pred_points],
                "pred_yaw": [math.degrees(y) for y in pred_yaws],
                "pred_pitch": [math.degrees(x) for x in pred_pitches],
                "pred_alive": pred_alive,
                "pred_firing": pred_firing,
                "pred_steps": len(pred_points),
                "end_stopped_at": end_stop,
            })

        # ── 8. GT 轨迹（目标玩家，用于对比）──────────
        gt = None
        if initial_alive and gt_available > 0:
            gt_points, gt_alive, gt_firing, _, gt_yaws, gt_pitches = integrate_trajectory(
                start_pos, start_yaw, start_pitch, gt_deltas,
                start_step=0, max_steps=gt_available,
                camera_relative_up=local_labels_are_v4,
                return_angles=True,
            )
            gt = {
                "gt_points": [list(pt) for pt in gt_points],
                "gt_yaw": [math.degrees(y) for y in gt_yaws],
                "gt_pitch": [math.degrees(x) for x in gt_pitches],
                "gt_alive": gt_alive,
                "gt_firing": gt_firing,
                "gt_steps": len(gt_points),
            }

        # ── 9. 下游指标（K 条采样 + 目标玩家 GT）──────────
        # 采样路径由预训练底座生成；指标用下游模型（独立实例）在采样路径上评估
        metrics_payload = None
        if metric_ctx is not None:
            down, down_head = self._ensure_down_model(metric)
            down_emb_all = down.get_player_embeddings(batch)[0, -1, :, :]  # [10, d]

            # 预测曲线：K 条采样各自生成的 token（预训练底座）+ 各自路径
            #            原始条件 → 下游模型 encoder 重新编码 → head
            K = conditions.shape[0]
            down_conditions = down_emb_all[player_idx].unsqueeze(0).repeat(K, 1).to(self.device)
            depth_t, xyz_t, angle_t = self._encode_metric_raw_ctx(
                metric_ctx["depth_raw_list"], metric_ctx["xyz_raw_list"],
                metric_ctx["angle_raw_list"], map_name, down, K)
            pred_tokens = torch.cat(metric_ctx["tokens_list"], dim=1)   # [K, n_ticks*7]
            pred_probs = self._task_metric_forward(
                down_conditions, pred_tokens, depth_t, xyz_t, angle_t,
                model=down, head=down_head,
            )  # [K, n_points]

            # 每条采样 mask：沿该条路径存活的位置
            n_points = n_ticks + 1
            pred_mask = np.zeros((K, n_points), dtype=bool)
            for k in range(K):
                pred_mask[k, 0] = True
                pa = samples[k]["pred_alive"]
                for j in range(1, n_points):
                    if j - 1 < len(pa) and pa[j - 1] > 0.5:
                        pred_mask[k, j] = True

            # GT 曲线（目标玩家，teacher forcing，下游模型）
            gt_probs = None
            gt_mask = np.zeros((1, n_points), dtype=bool)
            if gt_tokens_single is not None:
                cond_single = down_emb_all[player_idx:player_idx + 1, :].to(self.device)
                gt_ctx = self._build_gt_task_contexts(
                    torch_sample, query_tick, map_name, player_idx, model=down)
                gt_probs = self._task_metric_forward(
                    cond_single, gt_tokens_single.to(self.device), *gt_ctx,
                    model=down, head=down_head)
                gt_mask[0, 0] = True
                if gt is not None:
                    ga = gt["gt_alive"]
                    for j in range(1, n_points):
                        if j - 1 < len(ga) and ga[j - 1] > 0.5:
                            gt_mask[0, j] = True

            if metric == "winrate":
                pred_probs = self._flip_winrate(pred_probs, [teams[player_idx]] * K)
                if gt_probs is not None:
                    gt_probs = self._flip_winrate(gt_probs, [teams[player_idx]])
            labels = self._downstream_labels(metric, torch_sample, query_tick, teams)

            metrics_payload = {
                "task": metric,
                "task_label": TASK_LABELS.get(metric, metric),
                "n_points": n_points,
                "label": float(labels[player_idx]),
                "team": str(teams[player_idx]),
                "gt": self._json_curve(gt_probs[0], gt_mask[0]) if gt_probs is not None else None,
                "samples": [self._json_curve(pred_probs[k], pred_mask[k])
                            for k in range(K)],
            }

        return {
            "query_tick": query_tick,
            "input_T": T_input,
            "output_T": min(n_ticks, round_T - query_tick),
            "map_name": map_name,
            "player_idx": player_idx,
            "num_samples": num_samples,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "is_alive": initial_alive,
            "start_pos": list(start_pos),
            "samples": samples,
            "gt": gt,
            "metrics": metrics_payload,
        }

    def _build_batch(
        self,
        sample: dict,
        input_start: int,
        input_end: int,
        tick_interval: float = 0.25,
    ) -> dict:
        """
        从 round sample 截取输入窗口，构建 batch=1 的字典。
        """
        T = input_end - input_start  # ≤ n_ticks

        batch = {}
        for key, tensor in sample.items():
            if key in ("meta", "__key__") or not isinstance(tensor, torch.Tensor):
                continue
            # 沿 dim=0 切片，然后加 batch dim
            sliced = tensor[input_start:input_end]  # [T, ...]
            batch[key] = sliced.unsqueeze(0).to(self.device)  # [1, T, ...]

        # 构建 tick_times_input（如果 sample 中没有）
        if "tick_times_input" not in batch:
            if "round_seconds" in batch:
                batch["tick_times_input"] = batch["round_seconds"]
            else:
                times = torch.arange(T, dtype=torch.float32) * tick_interval
                batch["tick_times_input"] = times.unsqueeze(0).to(self.device)

        return batch


# ═══════════════════════════════════════════════════════════════════════════════════
# 独立测试
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import webdataset as wds
    from training_data.torch_dataset import decode_sample

    import argparse
    ap = argparse.ArgumentParser(description="Test prediction engine")
    ap.add_argument("--config", default="config/pretrain-a100.yaml")
    ap.add_argument("--checkpoint", default="examples/checkpoints/step_0035000.pt")
    ap.add_argument("--data-dir", default="examples/dataset")
    ap.add_argument("--tick", type=int, default=200)
    ap.add_argument("--maps-dir", default="maps/optimized_obj_files", help="OBJ file directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None, help="Save JSON result to file")
    args = ap.parse_args()

    # 加载引擎
    engine = PredictionEngine(args.config, args.checkpoint, device=args.device,
                              maps_dir=args.maps_dir)

    # 读取一个 sample
    shard = sorted(Path(args.data_dir, "test").glob("shards-*.tar"))[0]
    ds = wds.WebDataset([str(shard)], shardshuffle=False)
    for raw in ds:
        sample = decode_sample(raw)
        break

    meta = sample.get("meta", {})
    round_T = meta.get("T", 0)
    query_tick = min(args.tick, round_T - 1) if round_T > 0 else args.tick
    print(f"Sample: map={meta.get('map_name')}, T={round_T}, query_tick={query_tick}")

    # 推理
    result = engine.predict_at_tick(sample, query_tick)

    print(f"Input T: {result['input_T']}, Output T: {result['output_T']}")
    for traj in result["trajectories"]:
        if traj["is_alive"]:
            print(f"  Player {traj['player_idx']}: "
                  f"pred {traj['pred_steps']} steps, "
                  f"gt {traj['gt_steps']} steps, "
                  f"start_pos=({traj['start_pos'][0]:.0f}, "
                  f"{traj['start_pos'][1]:.0f}, {traj['start_pos'][2]:.0f})")

    if args.output:
        # 保存为 JSON（numpy 转换）
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
