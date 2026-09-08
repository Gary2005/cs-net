#!/usr/bin/env python3
"""
spatial-only 下游任务微调 — 单 tick 局面 → 每玩家一个概率（单任务）。

与 downstream/finetune_lora.py 的核心区别：
  - 不需要路径（不用 temporal / decoder / teacher-forcing 轨迹）
  - 输入 = **单个 tick 的完整局面**（10 玩家 + 炸弹 + 投掷物，27 tokens）
  - 模型 = 预训练 embedder + SpatialTransformer（**全量微调**），
    之后每个玩家的 embedding 过一个全连接 → 1 个 logit（二分类）
  - 数据不需要窗口：每个 sample = 单个 tick（完整局面），
    数据增强（player shuffle）也是每个 tick 独立进行。
  - 玩家在该 tick 已死亡 → mask 掉，不参与 loss。

一个模型只训练一个任务，用 --task 切换（与 downstream/finetune_lora.py 一致）：

  winrate      预测该玩家队伍的胜率 [0,1]   （回合级：label_winrate，0=CT 1=T）
  alive_end    预测该玩家最终存活概率 [0,1] （回合级：label_alive_end）
  future_kill  预测该玩家在**本 tick 之后任意时刻**（死前）获得击杀的概率
               （从 label_nxt_kill/death 重建最后击杀 tick，last_kill > t；
                与 pretrain_processor._last_kill_ticks 同口径）

用法示例:
    conda activate cs2demo
    python scripts/downstream-spatial-only/finetune_spatial_only.py \
        --config config/finetune-spatial-only-a100.yaml \
        --checkpoint /Users/wanjungu/Downloads/cs-net-v4-pro/cs-net-v4-pro.pt \
        --task winrate --batch-size 256 --max-steps 2000 --device cpu
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/downstream-spatial-only
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # 仓库根目录
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for _p in (str(_SCRIPTS_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_pretrain import build_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_data.torch_dataset import (
    augment_depth_with_angles,
    decode_sample,
    pretrain_collate_fn,
    sample_to_torch,
)
from training_data.pretrain_processor import shuffle_players

TASKS = ["winrate", "alive_end", "future_kill"]
N_PLAYERS = 10

# embedder 需要的输入 key（decode 后只保留这些 + 标签，尽早丢大数组）
_INPUT_KEYS: List[str] = [
    "player_pos", "player_state",
    "player_inv", "player_inv_mask",
    "player_rel_f", "player_rel_i", "player_rel_mask",
    "player_sound", "player_depth",
    "player_alive_mask",
    "bomb_pos", "bomb_state", "map_idx",
    "proj_pos", "proj_type", "proj_dur", "proj_mask", "proj_is_active",
]


def last_kill_ticks(nxt_kill: np.ndarray, nxt_death: np.ndarray) -> np.ndarray:
    """从 label_nxt_kill / label_nxt_death 重建每玩家最后击杀 tick [10]。

    与 pretrain_processor.PretrainWindowExtractor._last_kill_ticks 完全一致：
    nxt_death 的"变化点" = 一次击杀事件；击杀者 = 变化点 tick **之前**的
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


# ═══════════════════════════════════════════════════════════════════════
# 单 tick IterableDataset
# ═══════════════════════════════════════════════════════════════════════

class SingleTickDataset(torch.utils.data.IterableDataset):
    """读取 round 级 WDS，每个 sample = 单个 tick 的完整局面。

    对每个 round 的每个 tick 生成一个 sample（输入切片 [t:t+1] 保持 T=1 维），
    每个 tick 独立做 player shuffle（增强粒度 = tick，非窗口）。

    Args:
        data_dir:       Round WDS 目录（含 train/ test/）
        split:          "train" | "test" | "both"
        shuffle_buffer: tick 级 shuffle buffer（0 = 不 shuffle，顺序流式）
        augment_depth:  是否增强深度图（[T,10,64] → [T,10,64,5]）
        tick_stride:    采样步长（1 = 每 tick 都取；2 = 隔一个取一个）
        keep_ratio:     训练保留比例：每个 tick 以 keep_ratio 概率保留
                        （伯努利，每 epoch 重新掷硬币 → 多 epoch 覆盖全程
                        更多时刻，随机性 = 隐式增强）。1.0 = 全部保留。
                        同 round 相邻 tick 高度相关（输入几乎相同、winner
                        标签恒定），全量会让训练样本大量近重复 → 过拟合。
                        每 round 至少保留 1 个 tick。
        max_samples:    最大 tick 数（调试用，None = 全部）
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        shuffle_buffer: int = 20000,
        augment_depth: bool = True,
        tick_stride: int = 1,
        keep_ratio: float = 1.0,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.augment_depth = augment_depth
        self.tick_stride = max(1, tick_stride)
        self.keep_ratio = float(keep_ratio)
        assert 0.0 < self.keep_ratio <= 1.0, f"keep_ratio 需在 (0, 1]，got {keep_ratio}"
        self.max_samples = max_samples
        self.urls = self._gather_urls()

    def _gather_urls(self) -> List[str]:
        urls: List[str] = []
        for sp in (["train", "test"] if self.split == "both" else [self.split]):
            split_dir = self.data_dir / sp
            if not split_dir.is_dir():
                continue
            for shard in sorted(split_dir.glob("shards-*.tar")):
                urls.append(str(shard))
        if not urls:
            raise FileNotFoundError(f"No shards in {self.data_dir}/[{self.split}]")
        return urls

    def __iter__(self):
        import webdataset as wds

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_urls = self.urls
        else:
            per_worker = len(self.urls) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.urls)
            my_urls = self.urls[start:end]
        if not my_urls:
            return

        pipeline = wds.WebDataset(
            my_urls,
            shardshuffle=self.shuffle_buffer if self.shuffle_buffer > 0 else False,
            empty_check=False,
        )
        if self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        pipeline = pipeline.map(decode_sample)

        # tick 级 buffer：打散同一 round 内的连续 tick
        tick_buffer: List[dict] = []
        rng = np.random.RandomState()

        n_yield = 0
        n_rounds = 0
        n_round_ticks = 0          # 该 worker 看到的原始 tick 总数（cap 前）
        for sample in pipeline:
            if "player_pos" not in sample or "label_alive_end" not in sample:
                continue
            T = sample["player_pos"].shape[0]
            if T <= 0:
                continue
            if self.augment_depth:
                sample = augment_depth_with_angles(sample)

            last_kill = last_kill_ticks(
                sample.get("label_nxt_kill"), sample.get("label_nxt_death"))

            # player_teams：0=CT 1=T -1=未知（随 player shuffle 同步）
            meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
            teams_list = meta.get("teams") or []
            player_teams = np.array(
                [0 if t == "CT" else (1 if t == "T" else -1) for t in teams_list],
                dtype=np.int64,
            )

            # tick 候选：先按 stride 粗采样，再按 keep_ratio 伯努利保留。
            # 同 round 相邻 tick 高度相关（输入几乎不变、winner 恒定），
            # 全量 → 训练样本大量近重复 → 过拟合。keep_ratio < 1 时每个
            # tick 独立以 keep_ratio 概率保留（每 epoch 重新掷硬币，多
            # epoch 覆盖全程更多时刻）；每 round 至少保留 1 个 tick。
            full = np.arange(0, T, self.tick_stride)
            cand = full
            if self.keep_ratio < 1.0:
                cand = full[rng.rand(len(full)) < self.keep_ratio]
                if len(cand) == 0:
                    cand = full[[rng.randint(len(full))]]
            n_rounds += 1
            n_round_ticks += T

            for t in cand:
                st: dict = {}
                for key in _INPUT_KEYS:
                    if key in sample:
                        st[key] = sample[key][t:t + 1]          # 保持 [1, ...] 维度
                if "tick_times_input" not in st:
                    st["tick_times_input"] = np.asarray(
                        sample.get("round_seconds", np.zeros(T, dtype=np.float32))[t],
                        dtype=np.float32).reshape(1)
                # 标签（回合常量取 tick t；future_kill 按 tick t 锚定）
                st["label_alive_end"] = sample["label_alive_end"][t:t + 1]      # [1,10]
                st["label_future_kill"] = (last_kill > t).astype(np.float32)[None]  # [1,10]
                st["label_winrate"] = sample["label_winrate"][t:t + 1]          # [1]
                st["player_teams"] = player_teams.copy()                        # [10]

                # 每个 tick 独立 player shuffle
                st = shuffle_players(st)
                st["__key__"] = f"{sample.get('__key__', '?')}_t{t}"
                st["meta"] = {**meta, "tick": t, "round_T": T}

                if self.shuffle_buffer > 0:
                    tick_buffer.append(st)
                    if len(tick_buffer) >= self.shuffle_buffer:
                        idx = rng.randint(len(tick_buffer))
                        yield sample_to_torch(tick_buffer.pop(idx))
                        n_yield += 1
                else:
                    yield sample_to_torch(st)
                    n_yield += 1
                if self.max_samples is not None and n_yield >= self.max_samples:
                    return

            del sample  # 释放 round 大数组

        while tick_buffer:
            idx = rng.randint(len(tick_buffer))
            yield sample_to_torch(tick_buffer.pop(idx))
            n_yield += 1
            if self.max_samples is not None and n_yield >= self.max_samples:
                return

        # 每 worker 结束时打印一次本 worker 的数据统计（epoch 结束可见）
        if worker_info is None or worker_info.id == 0:
            ratio_txt = f", keep_ratio={self.keep_ratio}" if self.keep_ratio < 1.0 else ""
            print(f"[data] split={self.split} worker={worker_info.id if worker_info else 0} "
                  f"rounds={n_rounds} raw_ticks={n_round_ticks} "
                  f"sampled_ticks={n_yield} avg_raw_tick/round="
                  f"{n_round_ticks / max(n_rounds, 1):.0f}{ratio_txt}")

    def __len__(self) -> int:
        return len(self.urls)


# ═══════════════════════════════════════════════════════════════════════
# 模型：embedder + spatial（全量微调）+ 单任务头
# ═══════════════════════════════════════════════════════════════════════

class SpatialOnlyModel(nn.Module):
    """包装 embedder + SpatialTransformer（temporal / decoder 不参与）。"""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.embedder = base_model.embedder
        self.spatial = base_model.spatial

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """返回每玩家 spatial embedding [B, T, 10, d]（T=1）。"""
        tokens = self.embedder(batch)                          # [B, T, 27, d]
        B, T = batch["player_pos"].shape[:2]
        device = batch["player_pos"].device
        valid = torch.ones(B, T, 27, device=device, dtype=torch.bool)
        valid[:, :, 11:27] = batch["proj_mask"].bool()
        attn_mask = ~valid                                     # True = ignore
        player_emb = self.spatial(tokens, attn_mask)           # [B, T, 10, d]
        return player_emb


class TaskHead(nn.Module):
    """每玩家 embedding [N, d] → 1 个 logit（二分类）。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb).squeeze(-1)                        # [N]


def build_targets(batch: Dict[str, torch.Tensor], task: str) -> tuple:
    """返回 (targets [B*10], mask [B*10])。

    mask = 该 tick 存活的玩家（死亡玩家不参与 loss，与 finetune_lora 一致）。
    """
    B = batch["player_pos"].shape[0]
    alive = batch["player_alive_mask"].bool().reshape(-1)      # [B*10]
    if task == "winrate":
        # 回合级常量：所有玩家同一 label（该玩家队伍的胜率）
        t = batch["label_winrate"].float().reshape(-1, 1).expand(
            -1, N_PLAYERS).reshape(-1)
    elif task == "alive_end":
        t = batch["label_alive_end"].float().reshape(-1)       # [B*10]
    elif task == "future_kill":
        t = batch["label_future_kill"].float().reshape(-1)     # [B*10]
    else:
        raise ValueError(f"未知任务: {task}")
    return t, alive


@torch.no_grad()
def compute_ct_winrate(
    logits: torch.Tensor,      # [B*10]
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """用每个存活玩家的 winrate 预测聚合出 CT 胜率并评估（单 tick 版）。

    p = sigmoid(logit) = P(T 胜)；CT 方胜率 = 存活 CT 玩家的 (1-p) 均值，
    T 方胜率 = 存活 T 玩家的 p 均值；归一化 ct = CT/(CT+T)，预测 CT 胜 ⇔ ct>0.5。
    某方在 cond tick 无存活玩家 → 该方胜率记 0。
    """
    B = batch["player_pos"].shape[0]
    p = torch.sigmoid(logits).reshape(B, N_PLAYERS)            # [B,10] P(T胜)
    alive = batch["player_alive_mask"].bool().reshape(B, N_PLAYERS)
    teams = batch["player_teams"]                              # [B,10]
    known = teams >= 0
    is_ct = (teams == 0) & alive & known
    is_t = (teams == 1) & alive & known
    n_ct = is_ct.float().sum(-1)
    n_t = is_t.float().sum(-1)
    ct_avg = ((1 - p) * is_ct.float()).sum(-1) / n_ct.clamp(min=1)
    t_avg = (p * is_t.float()).sum(-1) / n_t.clamp(min=1)
    ct_avg = torch.where(n_ct > 0, ct_avg, torch.zeros_like(ct_avg))
    t_avg = torch.where(n_t > 0, t_avg, torch.zeros_like(t_avg))
    denom = ct_avg + t_avg
    ct = torch.where(denom > 0, ct_avg / denom.clamp(min=1e-9),
                     torch.full_like(denom, 0.5))
    true_ct = (batch["label_winrate"].float().reshape(-1, 1)[:, 0] == 0)      # [B]
    valid = denom > 0
    n_valid = int(valid.sum().item())
    acc = float((((ct > 0.5) == true_ct)[valid]).float().mean()) if n_valid else float("nan")
    return {
        "ct_winrate_acc": acc,
        "ct_winrate_mean": float(ct[valid].mean()) if n_valid else float("nan"),
        "true_ct_winrate": float(true_ct.float().mean()),
        "n_valid": n_valid,
    }


# ═══════════════════════════════════════════════════════════════════════
# 配置 / 参数
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # 数据
    "data_dir": "examples/dataset",
    "split": "train",
    "shuffle_buffer": 20000,
    "tick_stride": 1,
    "keep_ratio": 1.0,
    "max_samples": 0,
    # 模型架构（与预训练 checkpoint 一致；--config 可覆盖）
    "d_model": 768, "n_spatial_layers": 5, "n_temporal_layers": 3,
    "n_decoder_layers": 6, "n_depth_ray_layers": 3, "n_heads": 12,
    "d_ff": 3072, "dropout": 0.1,
    "move_range": 128.0, "move_grid_size": 4.0, "angle_grid_size": 5.0,
    "n_ticks": 16, "use_residual_correction": True,
    # 预训练 checkpoint
    "checkpoint": "",
    # 任务（一个模型只训练一个任务）
    "task": "winrate",
    # 训练
    "device": "cpu",
    "batch_size": 256,
    "num_workers": 0,
    "lr": 1e-4,
    "warmup_steps": 200,
    "lr_schedule": "cosine",
    "max_steps": 2000,
    "grad_clip": 1.0,
    "log_interval": 50,
    "seed": 0,
    # 保存 / 验证 / wandb
    "save_dir": "outputs/finetune_spatial_only",
    "save_interval": 1000,
    "val_interval": 500,
    "val_max_samples": 2000,
    "wandb": False,
    "wandb_project": "cs2-downstream-spatial",
    "wandb_name": None,
    "wandb_entity": None,
    # 加速
    "use_amp": True,
    "use_tf32": True,
    "use_compile": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="spatial-only 下游任务微调（单 tick 局面 → 单任务概率）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="",
                        help="配置 yaml（模型架构 + 训练超参 + 任务；CLI 参数可覆盖任意字段）")
    parser.add_argument("--checkpoint", default=None, help="预训练 checkpoint 路径（必填）")
    parser.add_argument("--data-dir", default=None, help="回合级 WebDataset 目录（含 train/ test/）")
    parser.add_argument("--split", choices=["train", "test", "both"], default=None)
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="下游任务：winrate / alive_end / future_kill")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多用多少个 tick（调试，0 = 全部）")
    parser.add_argument("--tick-stride", type=int, default=None,
                        help="tick 采样步长（1 = 每 tick 都取；2 = 隔一个）")
    parser.add_argument("--keep-ratio", type=float, default=None,
                        help="训练保留比例：每个 tick 以 keep_ratio 概率保留"
                             "（1.0 = 全部；防同 round 近重复样本过拟合）")
    parser.add_argument("--device", default=None, help="cpu / mps / cuda")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-interval", type=int, default=None, help="每 N 步保存（0=关闭）")
    parser.add_argument("--val-interval", type=int, default=None, help="每 N 步验证 test（0=关闭）")
    parser.add_argument("--val-max-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-amp", action="store_true", help="禁用 BF16 混合精度")
    parser.add_argument("--no-tf32", action="store_true", help="禁用 TF32 + cudnn.benchmark")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg.update(yaml_cfg)
    for k, v in vars(args).items():
        if v is not None and k not in (
            "config", "no_wandb", "no_amp", "no_tf32", "no_compile",
        ):
            cfg[k] = v
    if args.no_wandb:
        cfg["wandb"] = False
    if args.no_amp:
        cfg["use_amp"] = False
    if args.no_tf32:
        cfg["use_tf32"] = False
    if args.no_compile:
        cfg["use_compile"] = False
    return cfg


def resolve_device(device_str: str) -> torch.device:
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("[warn] MPS 不可用，回退到 CPU")
        return torch.device("cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: Optional[int],
                     schedule: str):
    def lr_lambda(step: int) -> float:
        step = max(1, step)
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if total_steps is None or schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════════
# 训练 / 评估
# ═══════════════════════════════════════════════════════════════════════

def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """二分类 AUC（Mann-Whitney U 统计量，rank-based，无 sklearn 依赖）。

    y_true:  0/1 标签
    y_score: 任意实数 logit/概率（越高越像正类）
    平局用平均秩处理；正/负类缺一 → NaN。
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # 平均秩（1-based，平局共享平均秩）
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    n = len(y_score)
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg
        i = j + 1

    # U = 正样本秩和 − n_pos(n_pos+1)/2；AUC = U / (n_pos · n_neg)
    u = ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


@torch.no_grad()
def evaluate(loader, model: nn.Module, head: nn.Module, cfg: dict,
             device: torch.device, max_batches: int = 0) -> Dict[str, float]:
    """在 test loader 上评估：loss / acc / pos_rate / AUC；winrate 附 ct_winrate 聚合。"""
    model.eval()
    head.eval()
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if cfg.get("use_amp", True) and device.type == "cuda" else nullcontext())
    task = cfg["task"]

    total_loss = 0.0
    total_acc = 0.0
    total_pos = 0.0
    n_valid = 0
    n_batches = 0
    ct = {"acc": 0.0, "mean": 0.0, "true": 0.0, "n": 0}
    logits_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    # winrate 是队伍级标签：AUC 按 **tick 级聚合**（每 tick 存活玩家 logit 均值
    # = 该 tick 一个预测 vs 该 tick 的 label_winrate），避免 10 个玩家共享同一
    # 标签造成的样本不独立/重复计数。alive_end / future_kill 标签是玩家级，
    # 保持 player-level AUC。
    tick_logits: List[float] = []
    tick_targets: List[float] = []

    for batch in loader:
        if max_batches > 0 and n_batches >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with amp_ctx:
            emb = model(batch)                                   # [B, 1, 10, d]
            logits = head(emb.reshape(-1, emb.shape[-1]))        # [B*10]
            targets, mask = build_targets(batch, task)
        mask = mask.to(device)
        m = mask & targets.isfinite()
        n = int(m.sum().item())
        if n > 0:
            l = F.binary_cross_entropy_with_logits(logits[m], targets[m].to(device))
            a = ((logits[m] > 0).float() == targets[m].to(device)).float().mean()
            total_loss += l.item()
            total_acc += a.item()
            total_pos += targets[m].float().mean().item()
            n_valid += n
            if task == "winrate":
                # tick 级聚合（存活玩家 logit 均值 → 每 tick 一个预测）
                B = batch["player_pos"].shape[0]
                lg = logits.detach().reshape(B, N_PLAYERS)
                alv = batch["player_alive_mask"].bool().reshape(B, N_PLAYERS)
                wr = batch["label_winrate"].float().reshape(-1)      # [B]
                for b in range(B):
                    alive_idx = alv[b]
                    if alive_idx.any():
                        tick_logits.append(lg[b][alive_idx].mean().item())
                        tick_targets.append(wr[b].item())
            else:
                logits_list.append(logits[m].detach().cpu().float())
                targets_list.append(targets[m].detach().cpu().float())
        if task == "winrate" and "player_teams" in batch:
            cm = compute_ct_winrate(logits, batch)
            ct["acc"] += cm["ct_winrate_acc"] * max(cm["n_valid"], 1)
            ct["mean"] += cm["ct_winrate_mean"] * max(cm["n_valid"], 1)
            ct["true"] += cm["true_ct_winrate"] * max(cm["n_valid"], 1)
            ct["n"] += cm["n_valid"]
        n_batches += 1

    model.train()
    head.train()
    if n_batches == 0:
        return {"loss": float("nan"), "acc": float("nan"), "n_batches": 0.0}
    out: Dict[str, float] = {
        "loss": total_loss / n_batches,
        "acc": total_acc / n_batches,
        "pos_rate": total_pos / n_batches,
        "n_valid": float(n_valid),
        "n_batches": float(n_batches),
    }
    if task == "winrate":
        if tick_targets:
            out["auc"] = binary_auc(np.asarray(tick_targets), np.asarray(tick_logits))
            out["auc_n"] = float(len(tick_targets))   # tick 数（评估单元）
    elif logits_list:
        out["auc"] = binary_auc(
            torch.cat(targets_list).numpy(),
            torch.cat(logits_list).numpy(),
        )
    if ct["n"] > 0:
        out["ct_winrate_acc"] = ct["acc"] / ct["n"]
        out["ct_winrate_mean"] = ct["mean"] / ct["n"]
        out["true_ct_winrate"] = ct["true"] / ct["n"]
    return out


def save_ckpt(path: Path, model: nn.Module, head: nn.Module, yaml_cfg: dict,
              task: str, global_step: int) -> None:
    """保存全量微调后的 embedder+spatial（剥离 _orig_mod.）+ head + 配置。"""
    model_state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({
        "task": task,
        "global_step": global_step,
        "model_state": model_state,
        "head_state": head.state_dict(),
        "config": yaml_cfg,
    }, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    if not cfg["checkpoint"]:
        raise ValueError("缺少预训练 checkpoint：请用 --checkpoint 传入")
    task = cfg["task"]
    if task not in TASKS:
        raise ValueError(f"未知任务: {task}（可选 {TASKS}）")

    device = resolve_device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    print(f"Device: {device} | task: {task}")

    if device.type == "cuda" and cfg.get("use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("TF32 + cudnn.benchmark: ON")

    yaml_cfg = cfg
    d_model = int(cfg["d_model"])

    # ── 模型：预训练底座 → 只取 embedder + spatial，全量微调 ──
    base_model, model_cfg = build_model(yaml_cfg, cfg["checkpoint"], device)
    print(f"预训练底座已加载: d_model={model_cfg.d_model}")

    model = SpatialOnlyModel(base_model)
    head = TaskHead(d_model).to(device)

    # 冻结 temporal / decoder（不参与优化）；embedder+spatial 全量微调
    for p in base_model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(True)
    for p in head.parameters():
        p.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad] + \
                list(head.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"可训练参数: {n_train / 1e6:.2f}M（embedder + spatial 全量微调 + 单任务头）")

    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], betas=(0.9, 0.95))
    total_steps = cfg["max_steps"] if cfg["max_steps"] > 0 else None
    scheduler = get_lr_scheduler(optimizer, cfg["warmup_steps"], total_steps,
                                 cfg["lr_schedule"])
    print(f"LR schedule: warmup={cfg['warmup_steps']}, "
          f"{cfg['lr_schedule']}{' → 0 (total ' + str(total_steps) + ')' if total_steps else ''}")

    if cfg.get("use_compile", True):
        model = torch.compile(model)
        print("torch.compile: ON")

    # ── Wandb ──
    use_wandb = cfg["wandb"]
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("[warn] wandb 未安装，跳过")
            use_wandb = False
    if use_wandb:
        run_name = cfg["wandb_name"] or f"{task}_spatial_only_d{d_model}"
        wandb.init(project=cfg["wandb_project"], name=run_name,
                   entity=cfg["wandb_entity"], config={
                       "task": task, "lr": cfg["lr"],
                       "warmup_steps": cfg["warmup_steps"],
                       "lr_schedule": cfg["lr_schedule"],
                       "max_steps": cfg["max_steps"],
                       "batch_size": cfg["batch_size"], "d_model": d_model,
                       "checkpoint": Path(cfg["checkpoint"]).name,
                   })
        print(f"Wandb: {wandb.run.name}")

    # ── 数据 ──
    max_samples = cfg["max_samples"] if cfg["max_samples"] > 0 else None
    keep_ratio = cfg.get("keep_ratio", 1.0)
    ds = SingleTickDataset(
        cfg["data_dir"], split=cfg["split"],
        shuffle_buffer=cfg["shuffle_buffer"] if max_samples is None else 0,
        augment_depth=True, tick_stride=cfg["tick_stride"],
        keep_ratio=keep_ratio, max_samples=max_samples,
    )
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], collate_fn=pretrain_collate_fn,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )
    print(f"Data: {cfg['data_dir']}/{cfg['split']}"
          f"（tick_stride={cfg['tick_stride']}，"
          f"keep_ratio={keep_ratio if keep_ratio < 1.0 else 'all'}）")

    # ── test 集（val 全量，确定性由 val_max_samples 控制）──
    test_loader = None
    if cfg["val_interval"] > 0:
        try:
            val_max = cfg.get("val_max_samples", 2000)
            test_ds = SingleTickDataset(
                cfg["data_dir"], split="test",
                shuffle_buffer=0, augment_depth=True,
                tick_stride=cfg["tick_stride"],
                keep_ratio=1.0,  # val 全量，保证指标可复现可比
                max_samples=val_max if val_max > 0 else None,
            )
            test_loader = DataLoader(
                test_ds, batch_size=cfg["batch_size"], collate_fn=pretrain_collate_fn,
                num_workers=0, pin_memory=(device.type == "cuda"),
            )
            print(f"Val: {cfg['data_dir']}/test（每次最多 {val_max} 个 tick）")
        except FileNotFoundError:
            print("[warn] 没有 test 集，val_interval 已关闭")
            cfg["val_interval"] = 0

    # ── 保存目录（不同 task 放不同子目录）──
    save_dir = Path(cfg["save_dir"])
    if not save_dir.is_absolute():
        save_dir = _PROJECT_ROOT / save_dir
    save_dir = save_dir / task
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(cfg["checkpoint"]).name.replace("latest_", "").replace(".pt", "")

    # ── 训练循环 ──
    # 数据是 IterableDataset（每轮 epoch 流式遍历一遍 = 一个 pass）。
    # max_steps > 0 时循环重启迭代器跑多个 epoch，直到累计步数达标；
    # max_steps = 0 时保持原语义：只跑一遍数据。
    model.train()
    head.train()
    global_step = 0
    epoch = 0
    t0 = time.time()
    win = {"loss": 0.0, "acc": 0.0, "pos": 0.0, "n": 0}
    win_steps = 0
    pbar = tqdm(total=cfg["max_steps"] if cfg["max_steps"] > 0 else None,
                desc=f"spatial-only [{task}]", unit="step", dynamic_ncols=True)

    while cfg["max_steps"] <= 0 or global_step < cfg["max_steps"]:
        epoch += 1
        if epoch > 1:
            print(f"\n── epoch {epoch} 开始（global_step={global_step}）──")
        batch_in_epoch = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                       if cfg.get("use_amp", True) and device.type == "cuda" else nullcontext())
            with amp_ctx:
                emb = model(batch)                                  # [B, 1, 10, d]
                logits = head(emb.reshape(-1, emb.shape[-1]))       # [B*10]
                targets, mask = build_targets(batch, task)
                mask = mask.to(device)
                m = mask & targets.isfinite()
                if m.any():
                    loss = F.binary_cross_entropy_with_logits(
                        logits[m], targets[m].to(device))
                else:
                    loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_in_epoch += 1

            with torch.no_grad():
                n_valid = int(m.sum().item())
                acc = ((logits[m] > 0).float() == targets[m].to(device)).float().mean().item() \
                    if n_valid else float("nan")
                pos = targets[m].float().mean().item() if n_valid else float("nan")
                lr_now = scheduler.get_last_lr()[0]

            # ── 累积 log 窗口 ──
            if n_valid > 0:
                win["loss"] += loss.item() * n_valid
                win["acc"] += acc * n_valid
                win["pos"] += pos * n_valid
                win["n"] += n_valid
            win_steps += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{acc:.3f}",
                "lr": f"{lr_now:.1e}",
            })
            pbar.update(1)

            if global_step % cfg["log_interval"] == 0:
                elapsed = time.time() - t0
                avg_loss = win["loss"] / max(win["n"], 1)
                avg_acc = win["acc"] / max(win["n"], 1)
                avg_pos = win["pos"] / max(win["n"], 1)
                print(f"step {global_step:5d} | {elapsed:.1f}s | lr {lr_now:.2e} | "
                      f"loss {avg_loss:.4f} | acc {avg_acc:.3f} | pos_rate {avg_pos:.3f} "
                      f"(n={win['n']})")
                if use_wandb:
                    wandb.log({
                        "step": global_step,
                        "train/lr": lr_now,
                        "train/loss": avg_loss,
                        "train/acc": avg_acc,
                        "train/pos_rate": avg_pos,
                    })
                win = {"loss": 0.0, "acc": 0.0, "pos": 0.0, "n": 0}
                win_steps = 0

            # ── 周期验证 ──
            if cfg["val_interval"] > 0 and global_step % cfg["val_interval"] == 0:
                val_metrics = evaluate(test_loader, model, head, cfg, device)
                n_vb = int(val_metrics["n_batches"])
                if n_vb == 0:
                    print(f"  [val] step {global_step:5d} | ⚠ test loader 空")
                else:
                    msg = (f"  [val] step {global_step:5d} | loss {val_metrics['loss']:.4f} "
                           f"| acc {val_metrics['acc']:.3f} | pos_rate {val_metrics['pos_rate']:.3f} "
                           f"| auc {val_metrics.get('auc', float('nan')):.3f} "
                           f"(n_batch={n_vb})")
                    if "ct_winrate_acc" in val_metrics:
                        msg += (f" | ct_winrate {val_metrics['ct_winrate_acc']:.3f} "
                                f"(n={int(val_metrics['n_valid'])})")
                    print(msg)
                    if use_wandb:
                        wandb.log({f"val/{k}": v for k, v in val_metrics.items()
                                   if k != "n_batches"} | {"step": global_step})

            # ── 周期保存 ──
            if cfg["save_interval"] > 0 and global_step % cfg["save_interval"] == 0:
                step_path = save_dir / f"{task}_spatial_only_{ckpt_name}_step_{global_step:07d}.pt"
                save_ckpt(step_path, model, head, yaml_cfg, task, global_step)
                print(f"  ✓ Saved {step_path.name}")

            if cfg["max_steps"] > 0 and global_step >= cfg["max_steps"]:
                break

        # 一个 epoch 结束：max_steps=0 只跑一遍；否则继续下一 epoch
        if cfg["max_steps"] <= 0 or global_step >= cfg["max_steps"]:
            break
        if batch_in_epoch == 0:
            print("[warn] loader 为空，终止训练（检查 data_dir）")
            break

    pbar.close()
    if use_wandb:
        wandb.finish()

    out_path = save_dir / f"{task}_spatial_only_{ckpt_name}.pt"
    save_ckpt(out_path, model, head, yaml_cfg, task, global_step)
    print(f"\nSaved -> {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
