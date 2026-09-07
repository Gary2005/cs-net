#!/usr/bin/env python3
"""
Teacher-forcing / Auto-regressive evaluation for CS2 pretrained model on local data.

用法:
    # Teacher-forcing 评估
    python scripts/test_pretrain.py \
        --config config/pretrain-a100.yaml \
        --checkpoint /path/to/checkpoint.pt \
        --data-dir examples/dataset \
        --key "parivision-vs-big-m3-ancient__round19_323472067" \
        --tick 0

    # Auto-regressive 评估（逐 tick 自回归 + 实时渲染深度图）
    python scripts/test_pretrain.py \
        --config config/pretrain-a100.yaml \
        --checkpoint /path/to/checkpoint.pt \
        --data-dir examples/dataset \
        --key "parivision-vs-big-m3-ancient__round19_323472067" \
        --tick 0 \
        --AR
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import webdataset as wds
import zstandard as zstd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pretrain_model import CS2PretrainModel, PretrainConfig
from training_data.torch_dataset import (
    augment_depth_with_angles,
    sample_to_torch,
)
from training_data.pretrain_processor import PretrainWindowExtractor
from training_data.config import denormalize_position, DEPTH_EYE_HEIGHT, N_PLAYERS

_dctx = zstd.ZstdDecompressor()

# ── 每个 tick 的 7 个 camera token 名称（单 token 覆盖完整有符号范围）────────
TOKEN_NAMES = [
    "continue",      # 0
    "d_forward",     # 1
    "d_right",       # 2
    "d_up",          # 3
    "d_pitch",       # 4
    "d_yaw",         # 5
    "fire",          # 6
]


# ═══════════════════════════════════════════════════════════════════════════════════
# Sample loading
# ═══════════════════════════════════════════════════════════════════════════════════

def find_sample(data_dir: str, key: str) -> dict:
    """在 WDS dataset 中根据 __key__ 查找指定 sample。"""
    data_root = Path(data_dir)
    searched = []

    for split in ("train", "test"):
        split_dir = data_root / split
        shards = sorted(split_dir.glob("shards-*.tar"))
        searched.append(str(split_dir))
        if not shards:
            continue

        for shard_path in shards:
            print(f"  Scanning {shard_path.name}...")
            dataset = wds.WebDataset([str(shard_path)], empty_check=False)
            for raw in dataset:
                sample_key = raw.get("__key__", "")
                if sample_key == key:
                    sample: dict = {}
                    for k, v in raw.items():
                        if k == "__key__":
                            sample["__key__"] = v
                        elif k.endswith(".npy.zst"):
                            name = k[:-8]
                            sample[name] = np.load(io.BytesIO(_dctx.decompress(v)))
                        elif k.endswith(".json.zst"):
                            sample["meta"] = json.loads(_dctx.decompress(v))
                    return sample

    raise ValueError(
        f"Sample with key '{key}' not found in {searched}."
    )


def extract_window_at_tick(sample: dict, target_tick: int, n_ticks: int) -> tuple:
    """
    从 round sample 中提取预训练窗口，以 target_tick 作为最后一个 condition tick。

    窗口起始 s = max(0, target_tick - n_ticks + 1)。

    Returns:
        (window, s, condition_pos)
    """
    T = sample["player_pos"].shape[0]
    total_ticks = n_ticks * 2 - 1
    s = max(0, target_tick - n_ticks + 1)
    condition_pos = target_tick - s

    if target_tick < 0 or target_tick >= T:
        raise ValueError(f"target_tick={target_tick} out of range [0, {T})")
    if s + total_ticks > T:
        raise ValueError(
            f"target_tick={target_tick} need {total_ticks} label ticks from s={s}, "
            f"but round only has {T} ticks."
        )

    sample = augment_depth_with_angles(sample)

    extractor = PretrainWindowExtractor(n_ticks=n_ticks, stride=1, jitter=False)
    round_seconds = sample.get("round_seconds", None)
    meta = sample.get("meta", {})
    tick_interval = meta.get("tick_interval", 0.25)

    input_start = s
    input_end = min(T, s + n_ticks)
    output_start = min(T, s + n_ticks)
    output_end = min(T, s + total_ticks)

    window = extractor._build_window(
        sample, s, T,
        input_start, input_end,
        output_start, output_end,
        round_seconds, tick_interval, meta,
    )
    return window, s, condition_pos


# ═══════════════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════════════

def build_model(yaml_cfg: dict, checkpoint_path: str, device: torch.device):
    """根据 yaml 配置构建模型并加载 checkpoint。"""
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
        use_residual_correction=yaml_cfg.get("use_residual_correction", True),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt.get("model", ckpt)
    print(f"Checkpoint step: {ckpt.get('global_step', '?')}")

    # 剥离 torch.compile 的 _orig_mod. 前缀
    new_state = {}
    for k, v in ckpt_state.items():
        new_state[k.replace("_orig_mod.", "")] = v
    ckpt_state = new_state

    model = CS2PretrainModel(model_cfg)

    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing:
        print(f"  Missing keys (using fresh init): {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f}M, Vocab: {model.tokenizer.vocab_size}")
    return model, model_cfg


# ═══════════════════════════════════════════════════════════════════════════════════
# Teacher-Forcing 评估
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_teacher_forcing(model: CS2PretrainModel, window: dict, device: torch.device,
                             condition_pos: int = 0) -> dict:
    """运行 teacher-forcing 评估。"""
    n_ticks = model.cfg.n_ticks
    tokenizer = model.tokenizer
    tpt = tokenizer.TOKENS_PER_TICK   # 7
    tpg = tokenizer.TOKENS_PER_GROUP  # 10

    batch = sample_to_torch(window)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)

    labels = batch["label_camera"]
    out = model(batch, labels)

    token_logits = out["token_logits"]
    gt_tokens = out["gt_tokens"]
    loss = out["loss"].item()

    N = token_logits.shape[0]
    seq_len = tpg * n_ticks

    pred_tokens = torch.argmax(token_logits, dim=-1)
    non_pad = gt_tokens != tokenizer.PAD
    correct = pred_tokens == gt_tokens

    total_correct = (correct & non_pad).sum().item()
    total_non_pad = non_pad.sum().item()
    overall_acc = total_correct / total_non_pad if total_non_pad > 0 else 0.0

    per_seq_pos = []
    for pos in range(seq_len):
        mask = non_pad[:, pos]
        n_valid = mask.sum().item()
        if n_valid > 0:
            per_seq_pos.append((pos, correct[:, pos][mask].float().mean().item(), n_valid))

    cond_offset = tpg - tpt  # 3 conditioning slots per tick
    per_tick_token = {}
    for tick in range(n_ticks):
        for tok_idx in range(tpt):
            seq_pos = tick * tpg + cond_offset + tok_idx
            if seq_pos >= seq_len:
                continue
            mask = non_pad[:, seq_pos]
            n_valid = mask.sum().item()
            if n_valid > 0:
                acc = correct[:, seq_pos][mask].float().mean().item()
                per_tick_token[(tick, TOKEN_NAMES[tok_idx])] = (acc, n_valid)

    per_tick_acc = []
    for tick in range(n_ticks):
        tick_accs = [per_tick_token[(tick, name)][0]
                     for name in TOKEN_NAMES if (tick, name) in per_tick_token]
        per_tick_acc.append(sum(tick_accs) / len(tick_accs) if tick_accs else 0.0)

    per_token_type_acc = {}
    for name in TOKEN_NAMES:
        accs = [per_tick_token[(t, name)][0]
                for t in range(n_ticks) if (t, name) in per_tick_token]
        per_token_type_acc[name] = sum(accs) / len(accs) if accs else 0.0

    return {
        "loss": loss, "overall_acc": overall_acc,
        "total_correct": total_correct, "total_non_pad": total_non_pad,
        "per_tick_acc": per_tick_acc, "per_token_type_acc": per_token_type_acc,
        "per_tick_token": per_tick_token, "per_seq_pos": per_seq_pos,
        "condition_pos": condition_pos,
        "token_logits": token_logits, "gt_tokens": gt_tokens,
        "pred_tokens": pred_tokens, "correct": correct, "non_pad": non_pad,
        "N": N, "n_ticks": n_ticks, "tpt": tpt, "tpg": tpg,
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# AR 辅助函数：状态提取、轨迹积分、深度渲染
# ═══════════════════════════════════════════════════════════════════════════════════

def _extract_yaw_pitch(state: np.ndarray):
    """从 player_state[14] 提取 yaw, pitch（弧度）。
    state layout: [hp, armor, helmet, defuser, flash_dur, flash_alpha,
                   cos(pitch), sin(pitch), cos(yaw), sin(yaw), is_CT,
                   log(v_fwd), log(v_right), log(v_vert)]
    """
    pitch = math.atan2(float(state[7]), float(state[6]))
    yaw = math.atan2(float(state[9]), float(state[8]))
    return yaw, pitch


def get_starting_state(sample_torch: dict, tick_idx: int, map_name: str):
    """获取指定 tick 时刻 10 个玩家的游戏坐标、角度和存活状态。

    Returns:
        pos_game:   [10, 3] float64 游戏坐标
        yaw_deg:    [10]    float64 度
        pitch_deg:  [10]    float64 度
        alive_arr:  [10]    bool
    """
    player_pos = sample_torch["player_pos"]       # [T, 10, 3]
    player_state = sample_torch["player_state"]   # [T, 10, 14]
    alive_mask = sample_torch["player_alive_mask"]  # [T, 10]

    pos_game = np.zeros((N_PLAYERS, 3), dtype=np.float64)
    yaw_deg = np.zeros(N_PLAYERS, dtype=np.float64)
    pitch_deg = np.zeros(N_PLAYERS, dtype=np.float64)
    alive_arr = np.zeros(N_PLAYERS, dtype=bool)

    for p in range(N_PLAYERS):
        nx, ny, nz = player_pos[tick_idx, p].tolist()
        gx, gy, gz = denormalize_position(nx, ny, nz, map_name)
        pos_game[p] = (gx, gy, gz)
        state = player_state[tick_idx, p].numpy()
        yaw, pitch = _extract_yaw_pitch(state)
        yaw_deg[p] = math.degrees(yaw)
        pitch_deg[p] = math.degrees(pitch)
        alive_arr[p] = bool(alive_mask[tick_idx, p])

    return pos_game, yaw_deg, pitch_deg, alive_arr


def apply_delta_v5(pos_game: np.ndarray, yaw_deg: np.ndarray, pitch_deg: np.ndarray,
                   alive_arr: np.ndarray, label_10d: np.ndarray, tick_idx: int):
    """用 v5（世界对齐）10D 标签更新玩家位置、角度和存活状态（原地修改）。

    v5: d_forward/d_right 在水平面上（仅由 yaw 决定），d_up = 纯世界 dz。
    end=0 表示停止，标记玩家死亡。

    label_10d: [10, N, 10] — player × tick × 10D
    """
    for p in range(N_PLAYERS):
        if not alive_arr[p]:
            continue
        d_fwd = float(label_10d[p, tick_idx, 0])
        d_right = float(label_10d[p, tick_idx, 1])
        d_up = float(label_10d[p, tick_idx, 2])
        dp_rad = math.atan2(float(label_10d[p, tick_idx, 4]),
                           float(label_10d[p, tick_idx, 3]))
        dy_rad = math.atan2(float(label_10d[p, tick_idx, 6]),
                           float(label_10d[p, tick_idx, 5]))
        is_alive = float(label_10d[p, tick_idx, 7])
        is_end = float(label_10d[p, tick_idx, 9])

        cos_y = math.cos(math.radians(yaw_deg[p]))
        sin_y = math.sin(math.radians(yaw_deg[p]))

        # v5: d_forward/d_right 在水平面
        pos_game[p, 0] += d_fwd * cos_y + d_right * sin_y
        pos_game[p, 1] += d_fwd * sin_y - d_right * cos_y
        pos_game[p, 2] += d_up  # 纯 world Z

        yaw_deg[p] += math.degrees(dy_rad)
        pitch_deg[p] = max(-89.0, min(89.0,
                          pitch_deg[p] + math.degrees(dp_rad)))

        # end=0 → 停止；alive=0 → 死亡
        if is_end < 0.5 or is_alive < 0.5:
            alive_arr[p] = False


def _compute_xyz_emb(model: CS2PretrainModel, pos_game: np.ndarray,
                     map_name: str, device: torch.device):
    """Encode absolute positions → [10, d_model] using mlp1 (no adapter; adapter applied in generate_group)."""
    from training_data.config import normalize_position, MAP_NAME_TO_IDX

    norm_pos = np.zeros((N_PLAYERS, 3), dtype=np.float32)
    for p in range(N_PLAYERS):
        nx, ny, nz = normalize_position(pos_game[p, 0], pos_game[p, 1], pos_game[p, 2], map_name)
        norm_pos[p] = (nx, ny, nz)

    pos_t = torch.from_numpy(norm_pos).to(device)
    map_id = MAP_NAME_TO_IDX.get(map_name, 0)
    map_emb = model.embedder.map_emb(
        torch.full((N_PLAYERS,), map_id, dtype=torch.long, device=device)
    )
    return model.embedder.mlp1(torch.cat([pos_t, map_emb], dim=-1))  # [10, d]


def _compute_angle_emb(model: CS2PretrainModel, yaw_deg: np.ndarray,
                       pitch_deg: np.ndarray, device: torch.device):
    """Encode yaw/pitch → [10, d_model] using mlp_angle."""
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    angle_in = np.stack([
        np.cos(yaw_rad), np.sin(yaw_rad),
        np.cos(pitch_rad), np.sin(pitch_rad),
    ], axis=-1).astype(np.float32)
    return model.embedder.mlp_angle(torch.from_numpy(angle_in).to(device))  # [10, d]


def render_depth_emb(model: CS2PretrainModel, map_geom,
                     pos_game: np.ndarray, yaw_deg: np.ndarray,
                     pitch_deg: np.ndarray, alive_arr: np.ndarray,
                     device: torch.device):
    """渲染当前玩家位置的深度图，返回 depth embedding [10, d_model]。

    流程与训练数据创建一致：
      1. 64 方向射线检测 → log 归一化距离 [0, 1]
      2. 拼接角度编码 → [10, 64, 5]
      3. DepthRayEncoder → [10, d_model]
    """
    from training_data.depth_map import compute_directional_depth

    if map_geom is None or not alive_arr.any():
        return None

    # Step 1: 射线检测（与训练数据创建完全一致）
    depth_raw, _ = compute_directional_depth(
        map_geom,
        pos_game[np.newaxis, :, :],     # [1, 10, 3]
        yaw_deg[np.newaxis, :],          # [1, 10]
        pitch_deg[np.newaxis, :],        # [1, 10]
        alive_arr[np.newaxis, :],        # [1, 10]
    )  # → [1, 10, 64] log-normalized

    # Step 2: 拼角度编码 → [10, 64, 5]（与 augment_depth_with_angles 一致）
    depth_aug = augment_depth_with_angles(
        {"player_depth": depth_raw}
    )["player_depth"][0]  # [10, 64, 5]

    # Step 3: DepthRayEncoder
    if isinstance(depth_aug, np.ndarray):
        depth_aug = torch.from_numpy(depth_aug)
    depth_emb = model.embedder.depth_encoder(
        depth_aug.to(device)
    )  # [10, d_model]

    return depth_emb


# ═══════════════════════════════════════════════════════════════════════════════════
# Auto-Regressive 评估
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_group_argmax(model, x, tick_idx, kv_cache, depth_emb, xyz_emb, angle_emb):
    """
    和 TokenDecoder.generate_group 的 argmax 路径一致，走 KV cache 增量解码。
    """
    # generate_group 的输入约定是已经过 decoder adapter 的 embedding；
    # 评估侧 render_depth_emb / _compute_*_emb 返回原始 embedding，需在此手动过一遍
    # （与旧版 _generate_group_argmax 行为保持一致）。
    dec = model.decoder
    if depth_emb is not None:
        depth_emb = dec.depth_dec_adapter(depth_emb)
    if xyz_emb is not None:
        xyz_emb = dec.xyz_dec_adapter(xyz_emb)
    if angle_emb is not None:
        angle_emb = dec.angle_dec_adapter(angle_emb)
    return model.decoder.generate_group(
        x, tick_idx,
        depth_emb=depth_emb, xyz_emb=xyz_emb, angle_emb=angle_emb,
        kv_cache=kv_cache, argmax=True,
    )


@torch.no_grad()
def evaluate_autoregressive(
    model: CS2PretrainModel,
    sample: dict,                # 原始 round-level numpy sample
    window: dict,                # 预训练窗口
    s: int,                      # 窗口在 round 中的起始 tick
    condition_pos: int,          # target_tick 在 condition window 中的位置
    target_tick: int,            # 用户指定的目标 tick
    device: torch.device,
    maps_dir: str,
) -> dict:
    """
    Auto-regressive 评估：逐 tick 自回归生成，实时渲染深度图。

    流程：
      1. 从 sample 获取 target_tick 时刻 10 个玩家的起始状态（pos, yaw, pitch, alive）
      2. 从 condition window 获取 player embedding（只用最后一个 tick）
      3. 逐 tick AR 循环：
         a. 用当前玩家位置 raycast 64 方向深度
         b. encode 深度 → depth_emb
         c. decode.generate_group → 一个 tick 的 7 个 camera token
         d. decode tokens → 10D label
         e. apply delta 更新位置/角度
         f. 检查 end/alive → 停止该玩家
      4. 与 ground truth 对比
    """
    from training_data.map_loader import get_map_geometry

    n_ticks = model.cfg.n_ticks
    meta = sample.get("meta", {})
    map_name = meta.get("map_name", "unknown")
    round_T = meta.get("T", 0)

    # ── 1. 起始状态 ────────────────────────────────────────────
    torch_sample = sample_to_torch(sample)
    pos_game, yaw_deg, pitch_deg, alive_arr = get_starting_state(
        torch_sample, target_tick, map_name)

    # 保存初始快照用于输出
    initial_pos = pos_game.copy()
    initial_yaw = yaw_deg.copy()
    initial_pitch = pitch_deg.copy()
    initial_alive = alive_arr.copy()
    alive_history = [alive_arr.copy()]  # 每 tick 的存活状态

    print(f"  Starting state: {alive_arr.sum()}/10 players alive")
    for p in range(N_PLAYERS):
        if alive_arr[p]:
            print(f"    P{p}: pos=({pos_game[p,0]:.1f}, {pos_game[p,1]:.1f}, {pos_game[p,2]:.1f})  "
                  f"yaw={yaw_deg[p]:.1f}°  pitch={pitch_deg[p]:.1f}°")

    # ── 2. Player embeddings from condition window ─────────────
    batch = sample_to_torch(window)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)

    # 获取 player embeddings，取 condition_pos 位置（即 target_tick 对应的位置）
    player_emb = model.get_player_embeddings(batch)  # [1, T, 10, d]
    conditions = player_emb[0, condition_pos, :, :]    # [10, d]
    print(f"  Condition from window position {condition_pos}, emb shape={list(conditions.shape)}")

    # ── 3. 加载地图几何体 ──────────────────────────────────────
    map_geom = None
    try:
        map_geom = get_map_geometry(map_name, Path(maps_dir))
        print(f"  Loaded map geometry: {map_geom}")
    except FileNotFoundError:
        print(f"  WARNING: Map OBJ not found at {maps_dir}/{map_name}.obj, "
              f"depth will be zeros.")

    # ── 4. Ground truth labels（用于对比）───────────────────────
    # GT 从 window 的 label_camera 中取：labels 覆盖 [s, s+total_ticks)
    # 我们要对比从 target_tick 开始的 n_ticks 个 tick
    total_ticks_in_window = window["label_camera"].shape[0]  # 2*n_ticks - 1
    gt_start_in_window = condition_pos  # label_camera 中对应 target_tick 的位置
    gt_end_in_window = min(gt_start_in_window + n_ticks, total_ticks_in_window)
    gt_labels = window["label_camera"][gt_start_in_window:gt_end_in_window]  # [≤n_ticks, 10, 10]

    # ── 5. AR 循环 ─────────────────────────────────────────────
    decoder_x = model.decoder.init_generate(conditions)
    kv_cache = model.decoder.new_kv_cache(conditions.shape[0], device)
    model.decoder.seed_cache(decoder_x, kv_cache)
    all_preds = []        # list of [10, 1, 10] numpy
    all_pred_tokens = []  # list of [10, 7] torch

    for tick in range(n_ticks):
        n_alive = alive_arr.sum()
        if n_alive == 0:
            print(f"  [AR tick={tick}] All players stopped → breaking")
            # 后续 tick 填 zeros
            zeros = np.zeros((N_PLAYERS, 1, 10), dtype=np.float32)
            zeros[:, 0, 9] = 0.0  # end=0
            zeros[:, 0, 7] = 0.0  # alive=0
            all_preds.append(zeros)
            alive_history.append(alive_arr.copy())
            continue

        # a) 渲染深度图（与训练时 storage depth 一致，均走 compute_directional_depth）
        depth_emb = render_depth_emb(
            model, map_geom, pos_game, yaw_deg, pitch_deg, alive_arr, device)

        # b) 计算 xyz/angle 上下文（与训练时 _build_abs_context 一致）
        xyz_emb = _compute_xyz_emb(model, pos_game, map_name, device)
        angle_emb = _compute_angle_emb(model, yaw_deg, pitch_deg, device)

        # c) AR 生成一个 tick
        # 不使用 sampling，直接用 argmax（与 TF 评估一致）
        tick_tokens, decoder_x = _generate_group_argmax(
            model, decoder_x, tick, kv_cache,
            depth_emb=depth_emb, xyz_emb=xyz_emb, angle_emb=angle_emb,
        )  # tick_tokens: [10, 7]

        all_pred_tokens.append(tick_tokens)

        # c) 解码 → 10D label
        tick_10d = model.tokenizer.decode_sequence(
            tick_tokens, 1
        )  # [10, 1, 10]
        tick_np = tick_10d.cpu().numpy()
        all_preds.append(tick_np)

        # d) 诊断输出
        end_vals = tick_np[:, 0, 9]
        alive_vals = tick_np[:, 0, 7]
        n_continue = int((end_vals > 0.5).sum())
        n_alive_pred = int((alive_vals > 0.5).sum())
        alive_indices = np.where(alive_arr)[0]
        print(f"  [AR tick={tick}] alive_players={alive_indices.tolist()}  "
              f"continue_pred={n_continue}/10  alive_pred={n_alive_pred}/10")

        # e) 更新状态
        if tick < n_ticks - 1 and alive_arr.any():
            apply_delta_v5(pos_game, yaw_deg, pitch_deg, alive_arr, tick_np, 0)

        # 记录存活状态
        alive_history.append(alive_arr.copy())

    # ── 6. 汇总预测 ────────────────────────────────────────────
    all_preds_np = np.concatenate(all_preds, axis=1)  # [10, n_ticks, 10]

    # ── 7. 与 GT 逐 token 对比 ──────────────────────────────────
    # GT labels: [n_ticks, 10, 10] — 直接可用
    gt_labels_tensor = torch.from_numpy(
        gt_labels.transpose(1, 0, 2)  # [10, n_ticks, 10]
    ).to(device)
    gt_len = gt_labels.shape[0]  # actual available GT ticks

    # 用 tokenizer 编码 GT（向量化）
    gt_tokens_all = model.tokenizer.encode_sequence(gt_labels_tensor, n_ticks)  # [10, 8*n_ticks]
    gt_tokens_2d = gt_tokens_all.reshape(N_PLAYERS, n_ticks, model.tokenizer.TOKENS_PER_TICK)

    # 对比预测 tokens 与 GT tokens（只对比 AR 实际生成的 tick）
    ar_ticks = len(all_pred_tokens)
    per_tick_ar_acc = []
    per_tick_token_ar = {}  # (tick, token_name) → acc

    for tick in range(ar_ticks):
        pred_t = all_pred_tokens[tick].cpu()  # [10, 7]
        gt_t = gt_tokens_2d[:, tick, :].cpu()  # [10, 7]
        tick_correct = 0
        tick_total = 0
        for tok_idx in range(model.tokenizer.TOKENS_PER_TICK):
            # 只统计初始存活 + AR 过程中仍在预测的玩家
            mask = alive_history[tick]  # 该 tick 开始时存活的玩家
            n_valid = mask.sum()
            if n_valid > 0:
                correct_tok = (pred_t[mask, tok_idx] == gt_t[mask, tok_idx]).sum().item()
                acc_tok = correct_tok / n_valid
                per_tick_token_ar[(tick, TOKEN_NAMES[tok_idx])] = acc_tok
                tick_correct += correct_tok
                tick_total += n_valid
        if tick_total > 0:
            per_tick_ar_acc.append(tick_correct / tick_total)
        else:
            per_tick_ar_acc.append(0.0)

    # 补全不足 n_ticks 的 tick
    while len(per_tick_ar_acc) < n_ticks:
        per_tick_ar_acc.append(0.0)

    # ── 7b. Per-dimension MAE（连续值误差，单位 = game units / 度）──
    per_dim_mae = {}  # dim_name → [errors...]
    for tick in range(ar_ticks):
        mask = alive_history[tick]
        if not mask.any():
            continue
        pred_tick = all_preds_np[mask, tick, :]       # [n_alive, 10]
        gt_tick = gt_labels[tick][mask, :]             # [n_alive, 10]

        # 位移 MAE（直接比较 d_forward, d_right, d_up）
        for dim_idx, dim_name in [(0, "d_forward"), (1, "d_right"), (2, "d_up")]:
            errs = np.abs(pred_tick[:, dim_idx] - gt_tick[:, dim_idx])
            per_dim_mae.setdefault(dim_name, []).extend(errs.tolist())

        # 角度 MAE（从 cos/sin 还原角度再比）
        for base_idx, dim_name in [(3, "d_pitch"), (5, "d_yaw")]:
            pred_rad = np.arctan2(pred_tick[:, base_idx + 1], pred_tick[:, base_idx])
            gt_rad   = np.arctan2(gt_tick[:, base_idx + 1], gt_tick[:, base_idx])
            ang_err_deg = np.abs(np.degrees(np.arctan2(
                np.sin(pred_rad - gt_rad), np.cos(pred_rad - gt_rad))))
            per_dim_mae.setdefault(dim_name, []).extend(ang_err_deg.tolist())

    per_dim_mae_mean = {k: np.mean(v) for k, v in per_dim_mae.items()}

    # ── 8. 计算轨迹（世界坐标点）────────────────────────────────
    # 用预测 labels 积分出世界坐标轨迹
    traj_pos = pos_game.copy()  # 从初始位置重新积分
    traj_yaw = yaw_deg.copy()
    traj_pitch = pitch_deg.copy()
    traj_alive = initial_alive.copy()
    pred_positions = {p: [(float(initial_pos[p, 0]),
                            float(initial_pos[p, 1]),
                            float(initial_pos[p, 2]))]
                      for p in range(N_PLAYERS)}
    gt_positions = {p: [(float(initial_pos[p, 0]),
                          float(initial_pos[p, 1]),
                          float(initial_pos[p, 2]))]
                     for p in range(N_PLAYERS)}

    for tick in range(ar_ticks):
        apply_delta_v5(traj_pos, traj_yaw, traj_pitch, traj_alive, all_preds_np, tick)
        for p in range(N_PLAYERS):
            pred_positions[p].append((float(traj_pos[p, 0]),
                                       float(traj_pos[p, 1]),
                                       float(traj_pos[p, 2])))

    # GT 轨迹
    gt_pos = initial_pos.copy()
    gt_yaw = initial_yaw.copy()
    gt_pitch = initial_pitch.copy()
    gt_alive = initial_alive.copy()
    for tick in range(gt_len):
        apply_delta_v5(gt_pos, gt_yaw, gt_pitch, gt_alive,
                       gt_labels[:].transpose(1, 0, 2), tick)
        for p in range(N_PLAYERS):
            gt_positions[p].append((float(gt_pos[p, 0]),
                                     float(gt_pos[p, 1]),
                                     float(gt_pos[p, 2])))

    return {
        "map_name": map_name,
        "target_tick": target_tick,
        "initial_pos": initial_pos,
        "initial_yaw": initial_yaw,
        "initial_pitch": initial_pitch,
        "initial_alive": initial_alive,
        "alive_history": alive_history,
        "ar_ticks": ar_ticks,
        "n_ticks": n_ticks,
        "per_tick_ar_acc": per_tick_ar_acc,
        "per_tick_token_ar": per_tick_token_ar,
        "pred_positions": pred_positions,
        "gt_positions": gt_positions,
        "gt_len": gt_len,
        "all_preds_np": all_preds_np,
        "gt_labels": gt_labels,
        # raw tokens for position-by-position print
        "all_pred_tokens": all_pred_tokens,       # list of [10, 7] torch
        "gt_tokens_2d": gt_tokens_2d,              # [10, n_ticks, 8] torch
        "tokenizer": model.tokenizer,
        "per_dim_mae_mean": per_dim_mae_mean,
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# 结果打印
# ═══════════════════════════════════════════════════════════════════════════════════

def print_tf_results(results: dict):
    """打印 Teacher-Forcing 评估结果。"""
    n_ticks = results["n_ticks"]
    tpt = results["tpt"]
    cp = results["condition_pos"]

    print()
    print("=" * 76)
    print("  TEACHER-FORCING EVALUATION RESULTS")
    print("=" * 76)
    print(f"  Samples (N=B×T×players): {results['N']}")
    print(f"  Seq length ({tpg}×n_ticks): {tpg * n_ticks}")
    print()
    print(f"  ── Overall ──")
    print(f"  Loss:                  {results['loss']:.4f}")
    print(f"  Token accuracy:        {results['overall_acc']:.4f}  "
          f"({results['total_correct']}/{results['total_non_pad']} correct/total)")
    print()

    print(f"  ── Per-tick accuracy (avg over {tpt} token types) ──")
    print(f"  ★ condition_pos={cp} = predictions starting from the target tick ★")
    header = f"  {'Tick':>5s}"
    for i in range(n_ticks):
        marker = "★" if i == cp else " "
        header += f" {marker}{i:4d}"
    print(header)
    acc_line = f"  {'acc':>5s}"
    for i, acc in enumerate(results["per_tick_acc"]):
        marker = ">" if i == cp else " "
        acc_line += f" {marker}{acc:.4f}"
    print(acc_line)
    print()

    print(f"  ── Target condition_pos={cp} per-token detail ──")
    print(f"  {'Token':>16s}  {'Accuracy':>10s}  {'Correct':>8s}  {'Count':>6s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*8}  {'-'*6}")
    cp_accs = []
    for name in TOKEN_NAMES:
        key = (cp, name)
        if key in results["per_tick_token"]:
            acc, n = results["per_tick_token"][key]
            n_correct = int(round(acc * n))
            cp_accs.append(acc)
            bar = "#" * int(acc * 40)
            print(f"  {name:>16s}  {acc:.4f}  {bar}  {n_correct:6d}  {n:5d}")
        else:
            print(f"  {name:>16s}  {'N/A':>10s}")
    if cp_accs:
        print(f"  {'─'*16}  {'─'*10}")
        print(f"  {'avg':>16s}  {sum(cp_accs)/len(cp_accs):.4f}")
    print()

    print(f"  ── Per-token-type accuracy (avg over all {n_ticks} condition ticks) ──")
    print(f"  {'Token':>16s}  {'Overall':>10s}  {'@cp={}'.format(cp):>10s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}")
    for name in TOKEN_NAMES:
        overall_acc = results["per_token_type_acc"].get(name, 0.0)
        cp_key = (cp, name)
        cp_a = results["per_tick_token"][cp_key][0] if cp_key in results["per_tick_token"] else None
        bar = "#" * int(overall_acc * 30)
        cp_str = f"{cp_a:.4f}" if cp_a is not None else "N/A"
        print(f"  {name:>16s}  {overall_acc:.4f}  {bar}  {cp_str}")
    print()

    print(f"  ── Per-tick × per-token (best/worst 3 per tick) ──")
    for tick in range(n_ticks):
        items = []
        for name in TOKEN_NAMES:
            key = (tick, name)
            if key in results["per_tick_token"]:
                items.append((name, results["per_tick_token"][key][0]))
        if items:
            items.sort(key=lambda x: -x[1])
            best3 = ", ".join(f"{n}={a:.2f}" for n, a in items[:3])
            worst3 = ", ".join(f"{n}={a:.2f}" for n, a in items[-3:])
            print(f"  tick {tick:2d} (avg={results['per_tick_acc'][tick]:.4f})  "
                  f"best: [{best3}]  worst: [{worst3}]")
    # ── Position-by-position TF: Player 0 at condition_pos ──
    token_logits = results["token_logits"]  # [N, seq_len, vocab]
    gt_tokens = results["gt_tokens"]
    pred_tokens = results["pred_tokens"]
    tpg = results["tpg"]
    cond_offset = tpg - tpt
    n_ticks = results["n_ticks"]
    cp = results["condition_pos"]
    p_idx = 0  # Player 0
    sample_idx = cp * N_PLAYERS + p_idx

    print(f"  ── TF Position-by-position: Player {p_idx}, condition_pos={cp} ──")
    for tick in range(n_ticks):
        print(f"  ── Tick {tick} ──")
        for tok_idx, name in enumerate(TOKEN_NAMES):
            seq_pos = tick * tpg + cond_offset + tok_idx
            if seq_pos >= gt_tokens.shape[1]:
                continue
            p = int(pred_tokens[sample_idx, seq_pos])
            g = int(gt_tokens[sample_idx, seq_pos])
            ok = "✓" if p == g else "✗"
            print(f"    pos {tok_idx:2d} ({name:>14s}):  pred={p:4d}  gt={g:4d}  {ok}")
        print()

    print()
    print("=" * 76)


def print_ar_results(results: dict):
    """打印 Auto-Regressive 评估结果。"""
    n_ticks = results["n_ticks"]
    ar_ticks = results["ar_ticks"]
    tpt = len(TOKEN_NAMES)  # 7

    print()
    print("=" * 76)
    print("  AUTO-REGRESSIVE EVALUATION RESULTS")
    print("=" * 76)
    print(f"  Map: {results['map_name']}, target_tick: {results['target_tick']}")
    print(f"  AR generated: {ar_ticks}/{n_ticks} ticks")
    print()

    initial_alive = np.where(results["initial_alive"])[0]
    print(f"  Initially alive players: {initial_alive.tolist()} "
          f"({len(initial_alive)}/10)")

    # Per-tick AR accuracy
    print(f"\n  ── Per-tick token accuracy (AR vs GT) ──")
    header = f"  {'Tick':>5s}"
    for i in range(ar_ticks):
        header += f"  {i:5d}"
    print(header)
    acc_line = f"  {'acc':>5s}"
    for acc in results["per_tick_ar_acc"][:ar_ticks]:
        acc_line += f"  {acc:.4f}"
    print(acc_line)
    print()

    # Alive status per tick
    print(f"  ── Alive players per tick ──")
    for tick in range(min(ar_ticks + 1, len(results["alive_history"]))):
        alive = results["alive_history"][tick]
        alive_list = np.where(alive)[0].tolist()
        print(f"  tick {tick:2d}: alive={alive_list} ({len(alive_list)}/10)")
    print()

    # Per-tick × per-token detail matrix
    print(f"  ── Per-tick × per-token accuracy (AR) ──")
    # Header row: token names (abbreviated)
    TOKEN_ABBR = ["cont", "fwd", "r", "up", "pitch", "yaw", "fire"]
    header = f"  {'Tick':>5s} {'alive':>5s}"
    for abbr in TOKEN_ABBR:
        header += f"  {abbr:>6s}"
    print(header)
    sep = f"  {'-'*5} {'-'*5}"
    for _ in TOKEN_ABBR:
        sep += f"  {'-'*6}"
    print(sep)

    for tick in range(ar_ticks):
        n_alive = int(results["alive_history"][tick].sum())
        line = f"  {tick:5d} {n_alive:5d}"
        for tok_idx, name in enumerate(TOKEN_NAMES):
            acc = results["per_tick_token_ar"].get((tick, name), None)
            if acc is not None:
                line += f"  {acc:.4f}"
            else:
                line += f"  {'-':>6s}"
        # append per-tick avg
        tick_avg = results["per_tick_ar_acc"][tick]
        line += f"  | avg={tick_avg:.4f}"
        print(line)
    print()

    # Per-token-type AR accuracy (average across all ticks)
    print(f"  ── Per-token-type AR accuracy (avg over {ar_ticks} ticks) ──")
    print(f"  {'Token':>16s}  {'AR':>10s}  {'TF@cp=0':>10s}  (TF for reference)")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}")
    for name in TOKEN_NAMES:
        accs = [results["per_tick_token_ar"].get((t, name), None)
                for t in range(ar_ticks)]
        accs = [a for a in accs if a is not None]
        if accs:
            avg = sum(accs) / len(accs)
            bar = "#" * int(avg * 30)
            print(f"  {name:>16s}  {avg:.4f}  {bar}")
        else:
            print(f"  {name:>16s}  {'N/A':>10s}")
    print()

    # Per-dimension MAE (continuous error in game units / degrees)
    maes = results.get("per_dim_mae_mean", {})
    if maes:
        print(f"  ── Per-dimension MAE (absolute error, {ar_ticks}-tick avg over alive players) ──")
        print(f"  {'Dimension':>16s}  {'MAE':>10s}  {'Unit':>10s}")
        print(f"  {'-'*16}  {'-'*10}  {'-'*10}")
        dim_units = {"d_forward": "units", "d_right": "units", "d_up": "units",
                     "d_pitch": "deg", "d_yaw": "deg"}
        for name in ["d_forward", "d_right", "d_up", "d_pitch", "d_yaw"]:
            if name in maes:
                unit = dim_units.get(name, "")
                print(f"  {name:>16s}  {maes[name]:8.2f}   {unit:>10s}")
        print()

    # ── Position-by-position detail (first alive player) ──
    all_pred_tokens = results.get("all_pred_tokens", [])
    gt_tokens_2d = results.get("gt_tokens_2d", None)
    tokenizer = results.get("tokenizer", None)

    if all_pred_tokens and gt_tokens_2d is not None and tokenizer is not None:
        alive_at_start = np.where(results["initial_alive"])[0]
        if len(alive_at_start) > 0:
            show_p = int(alive_at_start[0])
            print(f"  ── Position-by-position: Player {show_p} ──")
            print(f"  (pred=model output, gt=ground truth, ✓=correct)")

            for tick in range(results["ar_ticks"]):
                pred_t = all_pred_tokens[tick][show_p].cpu().numpy()
                gt_t = gt_tokens_2d[show_p, tick].cpu().numpy()
                was_alive = results["alive_history"][tick][show_p]
                status = "ALIVE" if was_alive else "STOPPED"

                print(f"  ── Tick {tick} ({status}) ──")
                for tok_idx, name in enumerate(TOKEN_NAMES):
                    p = int(pred_t[tok_idx])
                    g = int(gt_t[tok_idx])
                    ok = "✓" if p == g else "✗"
                    print(f"    pos {tok_idx:2d} ({name:>14s}):  pred={p:4d}  gt={g:4d}  {ok}")
                print()

    # Trajectory summary
    print(f"  ── Trajectory summary (per player) ──")
    all_ade = []
    all_fde = []
    for p in range(N_PLAYERS):
        if not results["initial_alive"][p]:
            print(f"  P{p}: DEAD at start")
            continue
        pred_pts = results["pred_positions"][p]
        gt_pts = results["gt_positions"][p]
        pred_end = pred_pts[-1]
        gt_end = gt_pts[-1] if len(gt_pts) > 0 else pred_pts[0]
        fde = math.sqrt(
            (pred_end[0] - gt_end[0])**2 +
            (pred_end[1] - gt_end[1])**2 +
            (pred_end[2] - gt_end[2])**2
        )
        # ADE: average per-tick displacement error
        n_steps = min(len(pred_pts), len(gt_pts))
        ade_sum = 0.0
        for s in range(n_steps):
            pp, gp = pred_pts[s], gt_pts[s]
            ade_sum += math.sqrt((pp[0]-gp[0])**2 + (pp[1]-gp[1])**2 + (pp[2]-gp[2])**2)
        ade = ade_sum / n_steps if n_steps > 0 else 0.0
        all_ade.append(ade)
        all_fde.append(fde)
        print(f"  P{p}: pred_steps={len(pred_pts)-1}  gt_steps={min(len(gt_pts)-1, results['gt_len'])}  "
              f"ADE={ade:.1f}u  FDE={fde:.1f}u  "
              f"pred_end=({pred_end[0]:.0f},{pred_end[1]:.0f},{pred_end[2]:.0f})  "
              f"gt_end=({gt_end[0]:.0f},{gt_end[1]:.0f},{gt_end[2]:.0f})")
    if all_ade:
        print(f"  ── avg ADE={sum(all_ade)/len(all_ade):.1f}u  avg FDE={sum(all_fde)/len(all_fde):.1f}u "
              f"(over {len(all_ade)} alive players)")

    # Per-tick ADE: how error accumulates over time
    print(f"\n  ── Per-tick displacement error (avg over alive players) ──")
    print(f"  {'Tick':>5s}  {'ADE@t(u)':>10s}  {'FDE@t(u)':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}")
    for tick in range(1, ar_ticks + 1):
        tick_ade_sum = 0.0
        tick_fde_sum = 0.0
        tick_count = 0
        for p in range(N_PLAYERS):
            if not results["initial_alive"][p]:
                continue
            pred_pts = results["pred_positions"][p]
            gt_pts = results["gt_positions"][p]
            if tick < len(pred_pts) and tick < len(gt_pts):
                pp, gp = pred_pts[tick], gt_pts[tick]
                err = math.sqrt((pp[0]-gp[0])**2 + (pp[1]-gp[1])**2 + (pp[2]-gp[2])**2)
                tick_ade_sum += err
                tick_fde_sum += err
                tick_count += 1
        if tick_count > 0:
            tick_ade = tick_ade_sum / tick_count
            tick_fde = tick_fde_sum / tick_count
            bar = "█" * int(tick_ade / 50)
            print(f"  {tick:5d}  {tick_ade:10.1f}  {tick_fde:10.1f}  {bar}")
        else:
            print(f"  {tick:5d}  {'N/A':>10s}")

    print()
    print("=" * 76)


# ═══════════════════════════════════════════════════════════════════════════════════
# Decoder Attention Weight 分析
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_decoder_attention(model, window, condition_pos, device, output_html):
    """提取 decoder 每层 attention weight 并生成 HTML 热力图。

    对第一个 tick group (positions 0-10) 的每个 camera token，
    可视化它看 condition 和前面 token 的注意力分布。
    """
    batch = sample_to_torch(window)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)

    tpt = model.tokenizer.TOKENS_PER_TICK  # 7
    tpg = model.tokenizer.TOKENS_PER_GROUP  # 10
    n_ticks = model.cfg.n_ticks

    # Get player embeddings
    player_emb = model.get_player_embeddings(batch)  # [1, T, 10, d]
    # Use a single player (first alive one at condition_pos)
    labels = batch["label_camera"]
    alive_mask = batch["player_alive_mask"]

    # Pick first alive player
    p_idx = 0
    for p in range(10):
        if alive_mask[0, condition_pos, p] > 0.5:
            p_idx = p
            break

    condition = player_emb[0, condition_pos, p_idx:p_idx+1, :]  # [1, d]
    print(f"Analyzing Player {p_idx} at condition_pos={condition_pos}")

    # Build decoder input (same as TokenDecoder.forward)
    dec = model.decoder
    N = 1
    seq_len = dec.seq_len

    # Build flat_targets with GT tokens for this player
    label_cam = labels[0, condition_pos:condition_pos+n_ticks, p_idx, :]  # [n_ticks, 10]
    gt_labels = label_cam.unsqueeze(0)  # [1, n_ticks, 10]
    gt_tokens = model.tokenizer.encode_sequence(gt_labels, n_ticks)  # [1, 8*n_ticks]

    # Build decoder input: GPT left-shift
    flat_targets = torch.full((N, seq_len), model.tokenizer.PAD, dtype=torch.long, device=device)
    camera_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    for offset in range(3):
        cond_indices = torch.arange(n_ticks, device=device) * tpg + offset
        camera_mask[cond_indices] = False
    flat_targets[:, camera_mask] = gt_tokens

    token_emb = dec.token_emb(flat_targets[:, :-1])
    cond_emb = dec.cond_proj(condition).unsqueeze(1)
    decoder_input = torch.cat([cond_emb, token_emb], dim=1)  # [1, seq_len, d]
    decoder_input = decoder_input + dec.pos_encoding[:, :seq_len, :]

    # Build conditioning context for this player
    # Depth: use zero embedding for simplicity (attention pattern doesn't depend on values)
    if "player_depth_labels" in batch:
        try:
            depth_all = batch["player_depth_labels"][0, :, p_idx, :, :]  # [total_ticks, 64, 5]
            depth_enc = model.embedder.depth_encoder(depth_all.to(device))  # [total_ticks, d]
            depth_windows = depth_enc.unfold(0, n_ticks, 1).permute(0, 2, 1)  # [T, n_ticks, d]
            depth_ctx = depth_windows[condition_pos:condition_pos+1].reshape(1, n_ticks, model.cfg.d_model)
            depth_ctx = dec.depth_dec_adapter(depth_ctx)
        except Exception:
            depth_ctx = torch.zeros(1, n_ticks, model.cfg.d_model, device=device)
        depth_indices = torch.arange(n_ticks, device=device) * tpg + 1
        decoder_input[:, depth_indices, :] = depth_ctx + dec.pos_encoding[:, depth_indices, :]

    # xyz context — use per-tick future position labels, same as training `_build_abs_context`
    from training_data.config import MAP_NAME_TO_IDX
    map_name = window.get("meta", {}).get("map_name", "de_ancient")
    if "player_pos_labels" in batch:
        pos_win = batch["player_pos_labels"][0, condition_pos:condition_pos + n_ticks, p_idx, :]
    else:
        window_ticks = batch["player_pos"].shape[1]
        tick_indices = list(range(condition_pos, min(condition_pos + n_ticks, window_ticks)))
        norm_pos = np.zeros((n_ticks, 3), dtype=np.float32)
        for i, t in enumerate(tick_indices):
            if t < window_ticks:
                norm_pos[i] = (
                    batch["player_pos"][0, t, p_idx, 0].item(),
                    batch["player_pos"][0, t, p_idx, 1].item(),
                    batch["player_pos"][0, t, p_idx, 2].item(),
                )
        pos_win = torch.from_numpy(norm_pos).to(device)
    pos_t = pos_win
    map_id = MAP_NAME_TO_IDX.get(map_name, 0)
    map_emb = model.embedder.map_emb(
        torch.full((n_ticks,), map_id, dtype=torch.long, device=device))
    xyz_emb = model.embedder.mlp1(torch.cat([pos_t, map_emb], dim=-1))
    xyz_emb = dec.xyz_dec_adapter(xyz_emb).unsqueeze(0)  # [1, n_ticks, d]
    xyz_indices = torch.arange(n_ticks, device=device) * tpg + 2
    decoder_input[:, xyz_indices, :] = xyz_emb + dec.pos_encoding[:, xyz_indices, :]

    # angle context — use per-tick future angle labels, same as training `_build_abs_context`
    if "player_angle_labels" in batch:
        angle_in = batch["player_angle_labels"][0, condition_pos:condition_pos + n_ticks, p_idx, :]
    else:
        window_ticks = batch["player_pos"].shape[1]
        state = batch["player_state"][0, min(condition_pos, window_ticks - 1), p_idx, :]
        cp_sp = state[6:8].unsqueeze(0).repeat(n_ticks, 1)  # fallback: same angle for all ticks
        cs_y = state[8:10].unsqueeze(0).repeat(n_ticks, 1)
        angle_in = torch.cat([cs_y, cp_sp], dim=-1).to(device)  # [n_ticks, 4]
    angle_emb = model.embedder.mlp_angle(angle_in)
    angle_emb = dec.angle_dec_adapter(angle_emb).unsqueeze(0)  # [1, n_ticks, d]
    angle_indices = torch.arange(n_ticks, device=device) * tpg + 3
    decoder_input[:, angle_indices, :] = angle_emb + dec.pos_encoding[:, angle_indices, :]

    # Causal mask
    attn_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    # Run decoder blocks, capturing attention weights
    layer_weights = []
    x = decoder_input
    for block in dec.decoder_blocks:
        x, w = block(x, attn_mask, return_attention=True)
        layer_weights.append(w.squeeze(0).cpu().numpy())  # [seq_len, seq_len]

    n_layers = len(layer_weights)

    # Build position labels
    cam_names = ["continue", "d_forward", "d_right", "d_up", "d_pitch", "d_yaw"]
    pos_labels = []
    for tick in range(n_ticks):
        gs = tick * tpg
        if tick == 0:
            pos_labels.append("cond")
        else:
            pos_labels.append(f"T{tick-1}:fire")
        pos_labels.append(f"T{tick}:depth")
        pos_labels.append(f"T{tick}:xyz")
        pos_labels.append(f"T{tick}:angle")
        for cn in cam_names:
            pos_labels.append(f"T{tick}:{cn}")
    pos_labels = pos_labels[:seq_len]

    # Generate HTML
    # Define sections to show
    sections = [
        ("Tick 0-1", 0, min(tpg * 2, seq_len)),
        ("Tick 10-11", max(0, tpg * 10 - 1), min(tpg * 12 + 1, seq_len)),
    ]

    html_content = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Decoder Attention Weights — Player """ + str(p_idx) + """</title>
<style>
body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #ffaa00; font-size: 16px; }
h2 { color: #58a6ff; font-size: 14px; margin-top: 24px; }
h3 { color: #8b949e; font-size: 12px; margin-top: 16px; }
.heatmap { overflow-x: auto; margin: 8px 0; }
.heatmap table { border-collapse: collapse; font-size: 8px; }
.heatmap td { padding: 0; }
.lbl-r { font-size: 8px; padding-right: 4px; white-space: nowrap; text-align: right; color: #8b949e; }
.hdr { font-size: 7px; writing-mode: vertical-lr; color: #8b949e; padding: 2px 0; }
.label-cond { color: #58a6ff; font-weight: bold; }
.label-cam  { color: #ffaa00; font-weight: bold; }
.row-cond { background: rgba(88,166,255,0.08); }
.row-cam  { background: rgba(255,170,0,0.06); }
.summary-table { font-size: 11px; border-collapse: collapse; margin: 8px 0; }
.summary-table td, .summary-table th { padding: 4px 8px; text-align: left; border-bottom: 1px solid #21262d; }
</style></head><body>
<h1>Decoder Attention Weights — Player """ + str(p_idx) + """</h1>
<p>Each cell = attention weight from row (query) to column (key).<br>
Bright yellow = highest weight in row. Black = zero. Gray = masked (future).<br>
Conditioning: <span class="label-cond">blue</span>. Camera tokens: <span class="label-cam">orange</span>.</p>
"""

    for section_name, section_start, section_end in sections:
        show_positions = section_end - section_start
        if show_positions <= 0:
            continue

        for layer_idx in range(n_layers):
            # Extract the section from the full weight matrix
            w_full = layer_weights[layer_idx]
            w_sec = w_full[section_start:section_end, section_start:section_end]

            html_content += f'<h2>Layer {layer_idx+1}/{n_layers} — {section_name} (pos {section_start}-{section_end-1})</h2>'
            html_content += '<div class="heatmap"><table>'

            # Column headers
            html_content += '<tr><td></td>'
            for c in range(show_positions):
                label = pos_labels[section_start + c]
                cls = 'label-cond' if any(x in label for x in ['cond', 'depth', 'xyz', 'angle']) else 'label-cam'
                html_content += f'<td class="hdr {cls}">{label}</td>'
            html_content += '</tr>\n'

            for r in range(show_positions):
                label = pos_labels[section_start + r]
                cls = 'label-cond' if any(x in label for x in ['cond', 'depth', 'xyz', 'angle']) else 'label-cam'
                html_content += f'<tr><td class="lbl-r {cls}">{label}</td>'

                # Per-row max for normalization (only visible positions)
                row = w_sec[r, :]
                visible = row[:r+1]  # causal: can only see 0..r
                row_max = visible.max() if visible.max() > 0 else 1.0

                for c in range(show_positions):
                    val = row[c]
                    if c > r:
                        color = "#161b22"
                    else:
                        # Normalize by row max so small weights spread across many positions are still visible
                        intensity = int(255 * (val / max(row_max, 1e-8)))
                        color = f"rgb({intensity},{max(0,intensity-60)},0)"
                    html_content += f'<td style="background:{color};width:18px;height:18px" title="Q:{pos_labels[section_start+r]} K:{pos_labels[section_start+c]} w={val:.4f}"></td>'
                html_content += '</tr>\n'
            html_content += '</table></div>\n'

    # Summary for camera tokens at tick 10
    html_content += '<h2>Top-attended positions for Tick 10 Camera Tokens</h2>'
    tick10_start = tpg * 10
    for layer_idx in [0, n_layers - 1]:
        w = layer_weights[layer_idx]
        html_content += f'<h3>Layer {layer_idx+1}</h3>'
        html_content += '<table class="summary-table"><tr><th>Token</th><th>Top-3 attended positions</th></tr>'
        for tok_offset in range(4, 10):  # camera tokens at positions 4-9 within tick
            query_pos = tick10_start + tok_offset
            if query_pos >= seq_len:
                break
            query_name = pos_labels[query_pos]
            row = w[query_pos, :]
            visible = row[:query_pos + 1].copy()
            visible[query_pos] = 0  # exclude self
            top3 = np.argsort(visible)[-3:][::-1]
            items = []
            for idx in top3:
                items.append(f'{pos_labels[idx]} ({row[idx]:.3f})')
            html_content += f'<tr><td class="label-cam">{query_name}</td><td>{" ← ".join(items)}</td></tr>'
        html_content += '</table>'

    html_content += '</body></html>'

    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"\nSaved attention analysis to {output_html}")
    print(f"Layers: {n_layers}, positions shown: {show_positions}/{seq_len}")


# ═══════════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CS2 pretrained model evaluation (TF / AR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Teacher-forcing
  python scripts/test_pretrain.py --config config/pretrain-a100.yaml \\
      --checkpoint latest.pt --data-dir examples/dataset \\
      --key "parivision-vs-big-m3-ancient__round19_323472067" --tick 0

  # Auto-regressive
  python scripts/test_pretrain.py --config config/pretrain-a100.yaml \\
      --checkpoint latest.pt --data-dir examples/dataset \\
      --key "parivision-vs-big-m3-ancient__round19_323472067" --tick 0 --AR
        """,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="examples/dataset")
    parser.add_argument("--key", required=True)
    parser.add_argument("--tick", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--AR", action="store_true",
                        help="Enable auto-regressive evaluation (default: teacher-forcing)")
    parser.add_argument("--maps-dir", default="maps/optimized_obj_files",
                        help="Directory containing map OBJ files (default: maps/optimized_obj_files)")
    parser.add_argument("--analyze-attention", type=str, default=None, metavar="OUTPUT_HTML",
                        help="Analyze decoder attention weights and save heatmap HTML")
    args = parser.parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load config
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f) or {}
    n_ticks = yaml_cfg.get("n_ticks", 16)
    print(f"Config: {args.config}, n_ticks={n_ticks}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, model_cfg = build_model(yaml_cfg, args.checkpoint, device)

    # Find sample
    print(f"\nSearching for sample: {args.key}")
    sample = find_sample(args.data_dir, args.key)
    T = sample["player_pos"].shape[0]
    meta = sample.get("meta", {})
    map_name = meta.get("map_name", "unknown")
    print(f"  Found: {T} ticks, map={map_name}, round={meta.get('round_id', '?')}")

    # Extract window
    print(f"\nExtracting window at target_tick={args.tick}...")
    window, s, condition_pos = extract_window_at_tick(sample, args.tick, n_ticks)
    print(f"  Window start s={s}, condition=[{s}, {s + n_ticks})")
    print(f"  target_tick={args.tick} → condition_pos={condition_pos} "
          f"(last_cond_tick={s + n_ticks - 1})")

    if args.analyze_attention:
        print(f"\nAnalyzing decoder attention weights...")
        analyze_decoder_attention(
            model, window, condition_pos, device, args.analyze_attention)
        print("Done.")
        return

    if args.AR:
        # ── Auto-Regressive ────────────────────────────────────
        print(f"\nRunning AUTO-REGRESSIVE evaluation...")
        print(f"  Maps dir: {args.maps_dir}")
        results = evaluate_autoregressive(
            model, sample, window, s, condition_pos, args.tick,
            device, args.maps_dir,
        )
        print_ar_results(results)
    else:
        # ── Teacher-Forcing ────────────────────────────────────
        print(f"\nRunning teacher-forcing evaluation...")
        results = evaluate_teacher_forcing(
            model, window, device, condition_pos=condition_pos)
        print_tf_results(results)

    print("Done.")


if __name__ == "__main__":
    main()
