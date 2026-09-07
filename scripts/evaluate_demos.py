#!/usr/bin/env python3
"""
批量评估 demo 目录（按子文件夹分级别）中的每位选手：

  1) 预训练模型：对"选手的未来轨迹"的 teacher-forcing 对数概率
     （对选手活着的每个时刻，取其后 n_ticks 个 tick 的 GT 相机 token 序列，
      用训练同款条件（未来 depth/xyz/angle + 窗口化 player embedding）做
      teacher-forcing，得到该轨迹被模型预测的（对数）概率；对所有活着的
      时刻平均）。

  2) 三个下游模型：对"选手的未来轨迹"的指标影响
     （winrate / future_kill / alive_end）：用对应的下游微调模型对 GT 轨迹
      逐 tick teacher-forcing 打分，得到沿轨迹的指标曲线；对所有活着的
      时刻平均）。

输出：汇总 4 张图（x = 每个级别一列、虚线分隔，点按地图着色，选手名标注）
+ 每个地图一个子文件夹（同样 4 张图，点按级别着色，没有该地图数据的级别
列留空）+ results.json（所有数值，可离线重画图）。

用法示例:
    python scripts/evaluate_demos.py \
        --demos-dir /Users/wanjungu/Downloads/test_demos \
        --model-dir /Users/wanjungu/Downloads/cs-net-v4-preview \
        --device mps \
        --out outputs/demo_eval

    # 只重画图（读 results.json，可切换汇总方式）
    python scripts/evaluate_demos.py --replot --out outputs/demo_eval \
        --summarize end --prob per_tick
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm  # noqa: E402

# matplotlib 缓存目录设为可写（沙箱 / HOME 可能不可写）
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache-eval-demos")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_parser import parse_demo  # noqa: E402
from replay_tool.filter import filter_data  # noqa: E402
from training_data.round_processor import process_round  # noqa: E402
from training_data.torch_dataset import (  # noqa: E402
    augment_depth_with_angles,
    sample_to_torch,
)
from training_data.map_loader import get_map_geometry  # noqa: E402
from training_data.config import MAP_NAME_TO_IDX, N_PLAYERS  # noqa: E402
from create_training_data import _convert_inventory_indices  # noqa: E402
from prediction_engine import (  # noqa: E402
    PredictionEngine,
    _TaskHead,
    DOWNSTREAM_TASKS,
    TASK_LABELS,
)
from pretrain_model import CS2PretrainModel, sinusoidal_time_encoding  # noqa: E402

DOWNSTREAM_PATTERNS = {
    "winrate": ("win-rate", "winrate"),
    "future_kill": ("future-kill", "future_kill"),
    "alive_end": ("alive-end", "alive_end"),
}

LEVEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

TICK_INTERVAL = 0.25

# 解析缓存版本：demo_parser / 过滤策略变更时递增，旧缓存自动失效
CACHE_VERSION = "v3"


# ═══════════════════════════════════════════════════════════════════════
# 模型发现 / 加载
# ═══════════════════════════════════════════════════════════════════════

def discover_models(model_dir: Path,
                    want: Optional[set] = None) -> Tuple[Path, Dict[str, Path]]:
    """发现预训练底座 + 需要的下游模型。

    want: 需要的任务集合（{"pretrain", *DOWNSTREAM_TASKS} 的子集）。
    底座总是需要（engine 配置/tokenizer 依赖它）；未选中的下游模型不加载。
    """
    model_dir = Path(model_dir)
    if want is None:
        want = {"pretrain", *DOWNSTREAM_TASKS}
    pts = sorted(model_dir.glob("*.pt"))
    if not pts:
        raise SystemExit(f"模型目录中没有 .pt 文件: {model_dir}")
    # 所有下游模式文件名（即使本次不跑也要排除，避免把下游误当底座）
    down_names_all = {
        p.name for p in pts
        for pats in DOWNSTREAM_PATTERNS.values() for pat in pats if pat in p.name
    }
    down: Dict[str, Path] = {}
    for task in DOWNSTREAM_TASKS:
        if task not in want:
            continue
        for p in pts:
            if any(pat in p.name for pat in DOWNSTREAM_PATTERNS[task]):
                down[task] = p
                break
        if task not in down:
            raise SystemExit(f"未在 {model_dir} 中找到下游模型（{task}）")
    base = next((p for p in pts if p.name not in down_names_all), None)
    if base is None:
        raise SystemExit(f"未在 {model_dir} 中找到预训练底座 checkpoint")
    return base, down


def load_down_model(engine: PredictionEngine, ckpt_path: Path, task: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    got_task = ckpt.get("task")
    if got_task != task:
        raise SystemExit(f"{ckpt_path.name}: task={got_task!r} != {task!r}")
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["peft_state"].items()}
    model = CS2PretrainModel(engine.model_cfg).to(engine.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [{task}] missing keys: {missing}")
    if unexpected:
        print(f"  [{task}] unexpected keys: {unexpected}")
    model.eval()
    head = _TaskHead(engine.model_cfg.d_model).to(engine.device).eval()
    head.load_state_dict(ckpt["head_state"])
    step = ckpt.get("global_step", "?")
    print(f"  [{task}] loaded step={step} head=({head.fc.weight.shape[1]}→1)")
    return model, head


# ═══════════════════════════════════════════════════════════════════════
# demo 发现 / 解析缓存
# ═══════════════════════════════════════════════════════════════════════

def discover_demos(demos_dir: Path, level_names: Optional[List[str]]) -> List[Tuple[str, List[Path]]]:
    demos_dir = Path(demos_dir)
    levels: List[Tuple[str, List[Path]]] = []
    subdirs = sorted(
        d for d in demos_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    for d in subdirs:
        if level_names and d.name not in level_names:
            continue
        demos = sorted(d.rglob("*.dem"))
        if demos:
            levels.append((d.name, demos))
    if not levels:
        root_demos = sorted(demos_dir.glob("*.dem"))
        if root_demos:
            levels.append((demos_dir.name, root_demos))
    if not levels:
        raise SystemExit(f"目录中没有 .dem 文件: {demos_dir}")
    return levels


def get_demo_samples(
    demo_path: Path,
    cache_dir: Path,
    maps_dir: Path,
    interval: float = TICK_INTERVAL,
) -> Tuple[List[dict], List[str], str]:
    """解析一个 .dem → 每回合 process_round 后的 numpy sample 列表（带缓存）。"""
    stat = os.stat(demo_path)
    # CACHE_VERSION：解析/过滤策略变更时递增，使旧缓存自动失效（保证数据干净）
    key = hashlib.sha1(
        f"{CACHE_VERSION}|{demo_path}|{interval}|{stat.st_size}|{stat.st_mtime_ns}"
        .encode()
    ).hexdigest()[:16]
    cache_file = cache_dir / f"{key}.pt.gz"
    if cache_file.exists():
        with gzip.open(cache_file, "rb") as f:
            obj = pickle.load(f)
        return obj["samples"], obj["player_names"], obj["map_name"]

    print(f"  解析 demo {demo_path.name} ...")
    t0 = time.time()
    data = parse_demo(str(demo_path), interval=interval, verbose=False)
    filter_data(data)
    map_name = data.get("map", "unknown")
    player_names = [
        p.get("name", f"P{i}") for i, p in enumerate(data.get("players", []))
    ]
    samples: List[dict] = []
    rounds = data.get("rounds", [])
    for ri, round_data in enumerate(rounds):
        rd = copy.deepcopy(round_data)
        rd["map_name"] = map_name
        rd["map"] = map_name
        try:
            _convert_inventory_indices(
                {"weapons": data.get("weapons", {}), "rounds": [rd]}
            )
            map_geom = get_map_geometry(map_name, maps_dir)
            sample = process_round(
                rd,
                map_geom=map_geom,
                source_file=demo_path.name,
                match_teams=None,
                players_meta=data.get("players"),
                tick_interval=interval,
                compute_depth=True,
                places=data.get("places"),
            )
        except Exception as exc:
            print(f"    警告：round {ri} 跳过（{exc}）")
            continue
        samples.append(sample)
    print(f"  解析完成：{len(samples)} 个回合，用时 {time.time() - t0:.1f}s")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_file, "wb") as f:
        pickle.dump(
            {"samples": samples, "player_names": player_names, "map_name": map_name},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return samples, player_names, map_name


# ═══════════════════════════════════════════════════════════════════════
# 回合级评估（核心）
# ═══════════════════════════════════════════════════════════════════════

def _build_tick_batch(ts: dict, s0: int, s1: int, device: torch.device,
                      interval: float) -> dict:
    """构建单段 tick 区间 [s0, s1) 的 batch [1, s1-s0, ...]（与 engine._build_batch 一致）。"""
    batch: Dict[str, torch.Tensor] = {}
    for key, tensor in ts.items():
        if key in ("meta", "__key__") or not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 0:
            continue
        batch[key] = tensor[s0:s1].unsqueeze(0).to(device)
    if "tick_times_input" not in batch:
        if "round_seconds" in batch:
            batch["tick_times_input"] = batch["round_seconds"]
        else:
            times = torch.arange(s1 - s0, dtype=torch.float32) * interval
            batch["tick_times_input"] = times.unsqueeze(0).to(device)
    return batch


def _build_window_batch(ts: dict, starts: List[int], n_ticks: int,
                        device: torch.device, interval: float) -> dict:
    """把多个 16-tick 窗口堆成 batch [B, 16, ...]（与 engine._build_batch 一致）。"""
    B = len(starts)
    batch: Dict[str, torch.Tensor] = {}
    for key, tensor in ts.items():
        if key in ("meta", "__key__") or not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 0:
            continue
        slices = [tensor[s:s + n_ticks] for s in starts]
        batch[key] = torch.stack(slices, dim=0).to(device)
    if "tick_times_input" not in batch:
        if "round_seconds" in batch:
            batch["tick_times_input"] = batch["round_seconds"]
        else:
            times = torch.arange(n_ticks, dtype=torch.float32) * interval
            batch["tick_times_input"] = times.unsqueeze(0).expand(B, -1).to(device)
    return batch


def _window_conditions(model: CS2PretrainModel, ts: dict, query_ticks,
                       win_batch: int, device: torch.device,
                       interval: float = TICK_INTERVAL,
                       seg_ticks: int = 64,
                       players: Optional[List[int]] = None) -> torch.Tensor:
    """对每个 query tick t，用 16-tick 因果窗口 [t-15, t+1) 计算 player embedding
    （训练 / engine 同款条件）。返回 [T, 10, d]（players=None）或 [T, P, d]。

    players 指定时只计算这些玩家的 temporal 条件（embedder/spatial 仍是全量共享，
    无法按玩家裁剪），返回 [T, P, d]；用于"只看单个玩家"的扫描省算力。

    内存优化：
      - embedder（含 depth encoder）与 spatial 逐 tick 独立，按 seg_ticks 分块
        处理整回合（避免大回合一次编码 10k 条深度射线 → MPS OOM）；
      - 随后仅对每个窗口跑廉价的因果 temporal transformer。
    """
    n_ticks = model.cfg.n_ticks
    d = model.cfg.d_model
    T = ts["player_pos"].shape[0]
    P = len(players) if players is not None else 10
    sel = slice(None) if players is None else players
    conds = torch.zeros(T, P, d, device=device)
    ticks = sorted(set(int(t) for t in query_ticks))
    if not ticks:
        return conds

    # 1) embedder + spatial：整回合一次，按 tick 分块（逐 tick 独立，与窗口无关）
    spat_parts = []
    for s0 in range(0, T, seg_ticks):
        s1 = min(s0 + seg_ticks, T)
        b = _build_tick_batch(ts, s0, s1, device, interval)
        tok = model.embedder(b)                    # [1, seg, 27, d]
        m = model._build_spatial_mask(b)           # [1, seg, 27]
        spat_parts.append(model.spatial(tok, m)[0])
    spat = torch.cat(spat_parts, dim=0)            # [T, 10, d]

    # 2) temporal：按 16-tick 因果窗口分批（绝对 round_seconds 时间编码）
    windows: Dict[int, List[Tuple[int, int]]] = {}
    for t in ticks:
        s = max(0, t - n_ticks + 1)
        windows.setdefault(s, []).append((t, t - s))
    starts = sorted(windows)
    round_seconds = ts.get("round_seconds")
    for i in range(0, len(starts), win_batch):
        s_list = starts[i:i + win_batch]
        B = len(s_list)
        x = torch.stack([spat[s:s + n_ticks][:, sel] for s in s_list])   # [B,P,16,d]
        x = x.permute(0, 2, 1, 3).reshape(B * P, n_ticks, d)       # [B*P,16,d]
        if round_seconds is not None:
            times = torch.stack(
                [round_seconds[s:s + n_ticks] for s in s_list]
            ).to(device)
        else:
            times = (torch.arange(n_ticks, dtype=torch.float32) * interval
                     ).unsqueeze(0).expand(B, -1).to(device)
        times_exp = times.unsqueeze(1).expand(-1, P, -1).reshape(B * P, n_ticks)
        x = x + sinusoidal_time_encoding(times_exp, d)
        causal = torch.triu(
            torch.ones(n_ticks, n_ticks, device=device, dtype=torch.bool),
            diagonal=1,
        )
        x = model.temporal.transformer(x, mask=causal)               # [B*P,16,d]
        x = x.reshape(B, P, n_ticks, d).permute(0, 2, 1, 3)         # [B,16,P,d]
        for bi, s in enumerate(s_list):
            for t, pos in windows[s]:
                conds[t] = x[bi, pos]
    return conds


def _round_ctx(model: CS2PretrainModel, ts: dict, map_name: str,
               device: torch.device,
               seg_ticks: int = 64,
               players: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """整回合逐 tick 的 depth / xyz / angle 编码（训练同款 encoder）。

    players=None → [T, 10, d]（全量）；指定 → 只编码这些玩家，返回 [T, P, d]
    （depth encoder 按玩家裁剪，单玩家扫描省 ~90% depth 编码）。

    depth encoder 按 seg_ticks 分块，避免大回合峰值内存过大。
    """
    T = ts["player_pos"].shape[0]
    d = model.cfg.d_model
    P = len(players) if players is not None else 10
    sel = slice(None) if players is None else players
    emb = model.embedder
    depth = ts["player_depth"]                       # [T,10,64,5]
    depth_parts = []
    for s0 in range(0, T, seg_ticks):
        s1 = min(s0 + seg_ticks, T)
        dd = depth[s0:s1][:, sel]                    # [seg, P, 64, 5]
        dd = dd.reshape((s1 - s0) * P, 64, 5).to(device)
        depth_parts.append(emb.depth_encoder(dd).reshape(s1 - s0, P, d))
    depth_enc = torch.cat(depth_parts, dim=0)        # [T, P, d]

    pos = ts["player_pos"][:, sel].to(device)        # [T, P, 3] 已归一化
    map_id = MAP_NAME_TO_IDX.get(map_name, 0)
    map_emb = emb.map_emb(
        torch.full((T * P,), map_id, dtype=torch.long, device=device))
    xyz_enc = emb.mlp1(
        torch.cat([pos.reshape(T * P, 3), map_emb], dim=-1)
    ).reshape(T, P, d)

    state = ts["player_state"][:, sel].to(device)    # [T, P, 14]
    ang = state[..., [8, 9, 6, 7]].reshape(T * P, 4)   # [cos(yaw),sin(yaw),cos(pitch),sin(pitch)]
    angle_enc = emb.mlp_angle(ang).reshape(T, P, d)
    return depth_enc, xyz_enc, angle_enc


def _task_slices(enc: torch.Tensor, tasks: List[Tuple[int, int]],
                 n_ticks: int) -> torch.Tensor:
    return torch.stack([enc[t:t + n_ticks, p] for t, p in tasks])


@torch.no_grad()
def evaluate_round(engine: PredictionEngine, down_models: Dict[str, Tuple],
                   ts: dict, map_name: str, teams: List[str],
                   batch_K: int, win_batch: int,
                   seg_ticks: int = 64,
                   tick_sample: float = 1.0,
                   seed: int = 12345,
                   want: Optional[set] = None,
                   round_idx: int = 0) -> Tuple[Optional[dict], Optional[dict]]:
    """评估一个回合，返回 (聚合统计, 逐任务原始数据)。

    tick_sample < 1.0 时，只随机保留该比例（种子固定、可复现）的"活着的时刻"
    任务，均值仍是无偏估计，但大幅减少计算量。

    want: 要算的模型集合（{"pretrain", *DOWNSTREAM_TASKS} 的子集）；
    未选中的模型跳过计算，输出中对应条目保持空（n=0）。

    raw（逐任务，K=本回合任务数，数组按任务对齐）：
      round  [K] int32        回合序号（round_idx）
      tick   [K] int32        tick t（回合内）
      player [K] int32        选手 0-9
      pre_tick [K,16] f32     预训练：路径每个 tick 的每 token 平均对数概率
      pre_total [K] f32       预训练：整条路径总对数概率（非 PAD token 之和）
      pre_tokcount [K] int32  预训练：非 PAD token 数
      pre_tick_max [K,16] f32 预训练：同一 mask 下每 tick 的 log max(p) 平均
                              （校准用：log p - log max(p) = 模型认可度，0=最优）
      pre_total_max [K] f32   预训练：整条路径 log max(p) 之和（非 PAD token）
      curves [K,3,17] f32     下游：winrate(己方,CT翻转)/future_kill/alive_end
                              j=0..16 概率曲线
      alive [K,17] bool       j=0..16 存活 mask（j=0 恒 True）

    Returns:
        (out, raw) 或 (None, None)（无任务/回合太短时）
    """
    n_ticks = engine.model_cfg.n_ticks
    device = engine.device
    if want is None:
        want = {"pretrain", *DOWNSTREAM_TASKS}

    def _maybe_empty_cache():
        if device.type == "mps":
            torch.mps.empty_cache()

    T = ts["player_pos"].shape[0]
    t_max = T - n_ticks
    if t_max < 0:
        return None, None
    alive = ts["player_alive_mask"].numpy()          # [T,10]

    if tick_sample >= 1.0:
        tasks = [(t, p) for t in range(t_max + 1) for p in range(N_PLAYERS)
                 if alive[t, p]]
    else:
        # 采样：固定种子，按比例随机保留"活着的时刻"（均值无偏）
        rng = np.random.default_rng(seed)
        tasks = [(t, p) for t in range(t_max + 1) for p in range(N_PLAYERS)
                 if alive[t, p] and rng.random() < tick_sample]
    if not tasks:
        return None, None
    K = len(tasks)
    task_t = np.array([t for t, _ in tasks], dtype=np.int64)
    task_p = np.array([p for _, p in tasks], dtype=np.int64)

    # GT 相机 token（逐任务 16-tick 窗口编码，残差从 0 开始 = 训练同款）
    labels = torch.stack(
        [ts["label_camera"][t:t + n_ticks, p] for t, p in tasks]).to(device)
    tokens = engine.model.tokenizer.encode_sequence(labels, n_ticks)  # [K, 112]
    del labels

    out = {
        "pretrain": {
            p: {"n": 0, "sum_total": 0.0, "sum_tokcount": 0, "sum_pertick": 0.0}
            for p in range(N_PLAYERS)
        },
        "metrics": {
            m: {
                p: {"n": 0, "mean_sum": 0.0, "end_sum": 0.0, "delta_sum": 0.0,
                    "delta_n": 0,
                    "curve_sum": np.zeros(17), "curve_count": np.zeros(17)}
                for p in range(N_PLAYERS)
            }
            for m in DOWNSTREAM_TASKS
        },
    }

    def _chunks():
        for i0 in range(0, K, batch_K):
            yield i0, slice(i0, min(i0 + batch_K, K))

    # ── 1) 预训练：轨迹对数概率 ─────────────────────────────────────────
    per_tick_logp = np.zeros((K, n_ticks))           # 逐 tick 每 token 平均
    per_tick_logp_max = np.zeros((K, n_ticks))       # 逐 tick 每 token 平均 log max(p)
    if "pretrain" in want:
        conds = _window_conditions(engine.model, ts, range(t_max + 1), win_batch,
                                   device, seg_ticks=seg_ticks)
        d_enc, x_enc, a_enc = _round_ctx(engine.model, ts, map_name, device,
                                         seg_ticks=seg_ticks)
        cond_t = conds[task_t, task_p]                 # [K, d]
        pad = engine.model.tokenizer.PAD
        per_tot = np.zeros(K)
        per_tot_max = np.zeros(K)                      # 路径 log max(p) 之和
        per_pertick = np.zeros(K)
        per_tokcount = np.zeros(K)
        tick_idx = np.arange(n_ticks)[:, None] * 10 + np.arange(3, 10)  # [16,7] 每 tick 相机 token 位置（tick*10+3..9）
        for i0, idx in _chunks():
            logits, flat = engine.model.decoder(
                cond_t[idx], tokens[idx],
                depth_ctx=_task_slices(d_enc, tasks[i0:i0 + batch_K], n_ticks),
                xyz_ctx=_task_slices(x_enc, tasks[i0:i0 + batch_K], n_ticks),
                angle_ctx=_task_slices(a_enc, tasks[i0:i0 + batch_K], n_ticks),
            )
            logp = torch.log_softmax(logits, dim=-1)   # [k,160,vocab]
            logp_max = logp.max(-1).values             # [k,160] 每 token 的 log max(p)
            mask = flat != pad                          # [k,160]
            gathered = logp.gather(-1, flat.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            vals = (gathered * mask).sum(-1).cpu().numpy()
            vals_max = (logp_max * mask).sum(-1).cpu().numpy()
            cnts = mask.sum(-1).cpu().numpy()
            k = idx.stop - idx.start
            per_tot[i0:i0 + k] = vals
            per_tot_max[i0:i0 + k] = vals_max
            per_tokcount[i0:i0 + k] = cnts
            per_pertick[i0:i0 + k] = vals / np.maximum(cnts, 1)
            # 逐 tick：每个 tick 的 7 个相机 token 的掩码均值（路径上的值）
            gm = (gathered * mask).cpu().numpy()       # [k,160]
            gm_max = (logp_max * mask).cpu().numpy()   # [k,160]
            cm = mask.cpu().numpy().astype(np.float32)
            ts_ = gm[:, tick_idx].sum(-1)              # [k,16]
            ts_max_ = gm_max[:, tick_idx].sum(-1)      # [k,16]
            tc_ = cm[:, tick_idx].sum(-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_tick_logp[i0:i0 + k] = np.where(tc_ > 0, ts_ / tc_, np.nan)
                per_tick_logp_max[i0:i0 + k] = np.where(
                    tc_ > 0, ts_max_ / tc_, np.nan)
        for i, (t, p) in enumerate(tasks):
            d_ = out["pretrain"][p]
            d_["n"] += 1
            d_["sum_total"] += float(per_tot[i])
            d_["sum_tokcount"] += int(per_tokcount[i])
            d_["sum_pertick"] += float(per_pertick[i])
        _maybe_empty_cache()

    # ── 2) 存活 mask：17 点（模型无关，pretrain-only 也写入）────────────
    # j=0 恒 True（cond tick 必存活）；j>=1 = tick t+j 存活且在回合内
    alive_raw = np.zeros((K, n_ticks + 1), bool)
    j17 = np.arange(n_ticks + 1)[None, :]            # [1,17]
    idx17 = task_t[:, None] + j17                     # [K,17]
    a_ok = idx17 < T
    a_ic = np.clip(idx17, 0, T - 1)
    alive_raw = a_ok & alive[a_ic, task_p[:, None]]   # [K,17]
    alive_raw[:, 0] = True

    # ── 3) 下游指标：GT 轨迹打分 ────────────────────────────────────────
    pred_pos = [0] + [k * 10 + 9 for k in range(n_ticks)]
    curves_raw = np.zeros((K, len(DOWNSTREAM_TASKS), n_ticks + 1), np.float32)
    for mi, m in enumerate(DOWNSTREAM_TASKS):
        if m not in want:
            continue
        down, head = down_models[m]
        conds_m = _window_conditions(down, ts, range(t_max + 1), win_batch,
                                     device, seg_ticks=seg_ticks)
        d_m, x_m, a_m = _round_ctx(down, ts, map_name, device,
                                   seg_ticks=seg_ticks)
        cond_m = conds_m[task_t, task_p]
        probs = np.zeros((K, n_ticks + 1))
        for i0, idx in _chunks():
            _, _, hidden = down.decoder(
                cond_m[idx], tokens[idx],
                depth_ctx=_task_slices(d_m, tasks[i0:i0 + batch_K], n_ticks),
                xyz_ctx=_task_slices(x_m, tasks[i0:i0 + batch_K], n_ticks),
                angle_ctx=_task_slices(a_m, tasks[i0:i0 + batch_K], n_ticks),
                return_hidden=True,
            )
            h = head(hidden[:, pred_pos, :])
            probs[i0:idx.stop] = torch.sigmoid(h).cpu().numpy()
        if m == "winrate":
            for p in range(N_PLAYERS):
                if p < len(teams) and teams[p] == "CT":
                    sel = task_p == p
                    probs[sel] = 1.0 - probs[sel]
        curves_raw[:, mi, :] = probs
        for i, (t, p) in enumerate(tasks):
            mask = alive_raw[i]
            curve = probs[i]
            mm = out["metrics"][m][p]
            mm["curve_sum"] += np.where(mask, curve, 0.0)
            mm["curve_count"] += mask
            fut = mask[1:]
            if fut.any():
                fut_vals = curve[1:][fut]
                mm["mean_sum"] += float(fut_vals.mean())
                mm["end_sum"] += float(fut_vals[-1])
                mm["n"] += 1
            # delta = 第 4 秒（j=n_ticks）与 cond（j=0）的变化量，中间无所谓。
            # 所有任务都算（含窗口内阵亡：curve[16] 取模型在该时刻的输出，
            # 如 alive_end 阵亡后≈0，delta 即为"沿轨迹的下降量"）。
            mm["delta_sum"] += float(curve[n_ticks] - curve[0])
            mm["delta_n"] += 1
        _maybe_empty_cache()

    # ── 逐任务原始数据（供以后画任意图，无需重算） ──────────────────────
    raw = {
        "round": np.full(K, round_idx, dtype=np.int32),
        "tick": task_t.astype(np.int32),
        "player": task_p.astype(np.int32),
    }
    if "pretrain" in want:
        raw["pre_tick"] = per_tick_logp.astype(np.float32)
        raw["pre_total"] = per_tot.astype(np.float32)
        raw["pre_tokcount"] = per_tokcount.astype(np.int32)
        raw["pre_tick_max"] = per_tick_logp_max.astype(np.float32)
        raw["pre_total_max"] = per_tot_max.astype(np.float32)
    raw["alive"] = alive_raw          # 模型无关，无条件写入（下游也复用同一份）
    if want & set(DOWNSTREAM_TASKS):
        raw["curves"] = curves_raw
    # 每任务每 tick 位移（游戏单位，0.25s 间隔）——与 pre_tick 逐 tick 对齐，
    # 画图时可用它做位移加权/过滤（静止 tick 模型几乎必对，稀释信号）。
    # player_pos 为归一化坐标：x/y 范围 ±5000 → ±1，z ±2000 → ±1。
    raw["disp"] = _task_tick_displacement(
        ts, task_t, task_p, n_ticks, T).astype(np.float32)
    return out, raw


def _task_tick_displacement(
    ts: dict, task_t: np.ndarray, task_p: np.ndarray,
    n_ticks: int, T: int,
) -> np.ndarray:
    """每任务每 tick 位移 [K, 16]（游戏单位，tick j → j+1，0.25s 间隔）。

    向量化差分：归一化坐标差分 → 游戏单位
      u = sqrt((5000·Δx_norm)² + (5000·Δy_norm)² + (2000·Δz_norm)²)
    只有"两端都存活"的 tick 才算位移：
      - 越界（tick+j+1 >= T）记 0（该 tick 不参与加权）；
      - 死亡 tick（alive→dead）记 0：死后 player_pos 清零（feature_builder
        跳过死亡 tick，位置保持 (0,0,0)），alive→dead 的"位移"是跳回原点的
        伪影（数千单位），不是真实移动，计入会严重污染位移加权。
    """
    pos = ts["player_pos"].numpy()                    # [T,10,3] 归一化
    alive = ts["player_alive_mask"].numpy()           # [T,10]
    K = len(task_t)
    # 任务窗口 [t, t+16] 的起点/终点索引
    j = np.arange(n_ticks + 1)[None, :]               # [1,17]
    idx = task_t[:, None] + j                          # [K,17]
    p = task_p[:, None]                                # [K,1]
    ok = idx < T                                       # [K,17]
    idx_c = np.clip(idx, 0, T - 1)
    seq = pos[idx_c, p, :]                             # [K,17,3]
    d = np.diff(seq, axis=1)                           # [K,16,3]
    scale = np.array([5000.0, 5000.0, 2000.0])
    u = np.linalg.norm(d * scale, axis=-1)             # [K,16]
    # 终点存活才有真实移动（起点=cond 必存活；死亡后两端都归零 → 差 0）
    end_alive = alive[idx_c, p][:, 1:]                 # [K,16] tick t+j+1 存活
    valid = ok[:, 1:] & ok[:, :-1] & end_alive         # [K,16]
    return np.where(valid, u, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

def _accumulate_match(out: Optional[dict], acc: dict) -> None:
    if out is None:
        return
    for p in range(N_PLAYERS):
        pa = acc["players"][p]
        pr = out["pretrain"][p]
        pa["pre"]["n"] += pr["n"]
        pa["pre"]["sum_total"] += pr["sum_total"]
        pa["pre"]["sum_tokcount"] += pr["sum_tokcount"]
        pa["pre"]["sum_pertick"] += pr["sum_pertick"]
        for m in DOWNSTREAM_TASKS:
            mm = out["metrics"][m][p]
            dst = pa["met"][m]
            dst["n"] += mm["n"]
            dst["mean_sum"] += mm["mean_sum"]
            dst["end_sum"] += mm["end_sum"]
            dst["delta_sum"] += mm["delta_sum"]
            dst["delta_n"] += mm["delta_n"]
            dst["curve_sum"] += mm["curve_sum"]
            dst["curve_count"] += mm["curve_count"]


def _finalize_match(acc: dict) -> dict:
    players = []
    for p in range(N_PLAYERS):
        pa = acc["players"][p]
        pre = pa["pre"]
        n = pre["n"]
        pre_out = {"n_ticks": n}
        if n > 0:
            pre_out["per_tick"] = pre["sum_pertick"] / n
            pre_out["per_token"] = (
                pre["sum_total"] / pre["sum_tokcount"]
                if pre["sum_tokcount"] > 0 else float("nan"))
            pre_out["total_per_tick"] = pre["sum_total"] / n
            pre_out["n_tokens"] = pre["sum_tokcount"]
        met_out = {}
        for m in DOWNSTREAM_TASKS:
            mm = pa["met"][m]
            d = {"n_ticks": mm["n"], "n_delta": mm["delta_n"]}
            if mm["n"] > 0:
                d["mean"] = mm["mean_sum"] / mm["n"]
                d["end"] = mm["end_sum"] / mm["n"]
                with np.errstate(invalid="ignore", divide="ignore"):
                    curve = np.where(
                        mm["curve_count"] > 0,
                        mm["curve_sum"] / np.maximum(mm["curve_count"], 1),
                        np.nan,
                    )
                d["curve"] = [float(x) for x in curve]
            if mm["delta_n"] > 0:
                d["delta"] = mm["delta_sum"] / mm["delta_n"]
            met_out[m] = d
        players.append({
            "idx": p,
            "name": pa["name"],
            "pretrain": pre_out,
            "metrics": met_out,
        })
    return {"players": players}


def run(args) -> dict:
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS 不可用，回退到 CPU")
        device = torch.device("cpu")

    want: set = (set(args.models) if args.models
                 else {"pretrain", *DOWNSTREAM_TASKS})
    base_ckpt, down_ckpts = discover_models(Path(args.model_dir), want)
    print(f"预训练底座: {base_ckpt.name}")
    for t, p in down_ckpts.items():
        print(f"下游 {t}: {p.name}")

    engine = PredictionEngine(args.config, str(base_ckpt), device=device,
                              maps_dir=args.maps_dir)
    down_models: Dict[str, Tuple] = {}
    for t in down_ckpts:
        down_models[t] = load_down_model(engine, down_ckpts[t], t)

    levels = discover_demos(Path(args.demos_dir), args.levels)
    if args.max_matches > 0:
        levels = [(name, demos[:args.max_matches]) for name, demos in levels]

    # ── --new-only：只评估新增 demo，旧场次沿用已有 results.json ──────────
    old: Optional[dict] = None
    old_names: Dict[str, set] = {}
    if args.new_only:
        rp = Path(args.out) / "results.json"
        if not rp.exists():
            raise SystemExit(f"--new-only 需要已有 results.json：{rp}")
        with open(rp, "r", encoding="utf-8") as f:
            old = json.load(f)
        oc = old.get("config", {})
        # 采样参数必须与上次一致，否则新旧数据混采样。
        # --models 允许子集：新场次缺失的指标不会出现在对应图/汇总中（图保持旧数据）。
        for k in ("tick_sample", "seed", "max_rounds"):
            if oc.get(k) != getattr(args, k):
                raise SystemExit(
                    f"--new-only：参数 {k}={getattr(args, k)!r} 与已有结果 "
                    f"{oc.get(k)!r} 不一致。请保持一致，或全量重跑（删掉 results.json）")
        old_models = sorted(oc.get("models") or ["pretrain", *DOWNSTREAM_TASKS])
        if old_models != sorted(want):
            print(f"    注意：--models {sorted(want)} 与已有结果 {old_models} 不同；"
                  f"新场次缺失的指标不会出现在对应图/汇总中（图保持旧数据）")
        for lv, ms in old.get("matches", {}).items():
            old_names[lv] = {m["match"] for m in ms}

    if old is not None:
        results = old          # 沿用旧结果，只追加新场次
    else:
        results = {
            "levels": [name for name, _ in levels],
            "config": {
                "demos_dir": args.demos_dir, "model_dir": args.model_dir,
                "device": str(device), "interval": TICK_INTERVAL,
                "batch_K": args.batch, "win_batch": args.win_batch,
                "seg_ticks": args.seg_ticks,
                "tick_sample": args.tick_sample,
                "seed": args.seed,
                "models": sorted(want),
                "group_by": args.group_by,
                "save_raw": not args.no_save_raw,
                "max_rounds": args.max_rounds,
            },
            "matches": {},
        }

    mi = 0
    skipped_old = 0
    for level_name, demos in levels:
        if old is not None:
            existing = old_names.get(level_name, set())
            run_demos = [d for d in demos if d.name not in existing]
            skipped_old += len(demos) - len(run_demos)
            demos = run_demos
            if not demos:
                print(f"  [{level_name}] 无新增 demo，跳过")
                continue
        else:
            results["matches"][level_name] = []
        results["matches"].setdefault(level_name, [])
        for demo_path in tqdm(demos, desc=f"[{level_name}]", position=0):
            mi += 1
            t0 = time.time()
            try:
                samples, player_names, map_name = get_demo_samples(
                    demo_path, Path(args.cache_dir), Path(args.maps_dir))
            except Exception as exc:
                print(f"    警告：跳过 {demo_path.name}（解析失败：{exc}）")
                import traceback
                traceback.print_exc()
                continue
            try:
                match, raw = _process_match(engine, down_models, samples,
                                            player_names, map_name, args, want)
            except Exception as exc:
                print(f"    警告：跳过 {demo_path.name}（评估失败：{exc}）")
                import traceback
                traceback.print_exc()
                continue
            match["match"] = demo_path.name
            match["rounds_used"] = min(len(samples), args.max_rounds or len(samples))
            match["map"] = map_name
            results["matches"][level_name].append(match)
            if raw is not None and not args.no_save_raw:
                _save_raw(raw, Path(args.out) / "raw", demo_path, level_name,
                          player_names, map_name)
            if not args.quiet:
                print(f"    [{level_name}] {demo_path.name}: "
                      f"{match['rounds_used']} 回合，{time.time() - t0:.1f}s")
            # 每场比赛后落盘一次（崩溃可续）
            _save_results(results, Path(args.out))
        # end for demos
    # end for levels

    if old is not None:
        # 合并级别列表：旧级别 + 本次新增的级别（图表按此顺序排列）
        seen = set(results["levels"])
        for lv in results["matches"]:
            if lv not in seen:
                results["levels"].append(lv)
                seen.add(lv)
        if skipped_old:
            print(f"跳过 {skipped_old} 场（已在 results.json 中，未重新计算）")
    return results


def _process_match(engine, down_models, samples, player_names, map_name, args,
                   want: Optional[set] = None):
    """评估一场比赛的全部回合，返回 (选手级汇总 dict, 逐任务原始数据 dict 或 None)。"""
    acc = {"players": [
        {"name": player_names[p] if p < len(player_names) else f"P{p}",
         "pre": {"n": 0, "sum_total": 0.0, "sum_tokcount": 0,
                 "sum_pertick": 0.0},
         "met": {m: {"n": 0, "mean_sum": 0.0, "end_sum": 0.0,
                     "delta_sum": 0.0, "delta_n": 0,
                     "curve_sum": np.zeros(17),
                     "curve_count": np.zeros(17)}
                 for m in DOWNSTREAM_TASKS}}
        for p in range(N_PLAYERS)
    ]}
    if args.max_rounds > 0:
        samples = samples[:args.max_rounds]
    raw_parts = []
    rng = tqdm(range(len(samples)), desc=f"    rounds",
               position=1, leave=False)
    for ri in rng:
        sample = samples[ri]
        if "player_depth" in sample and sample["player_depth"].ndim == 3:
            sample = augment_depth_with_angles(sample)
        ts = sample_to_torch(sample)
        teams = sample.get("meta", {}).get("teams", ["?"] * N_PLAYERS)
        out, raw = evaluate_round(engine, down_models, ts, map_name,
                                  teams, args.batch, args.win_batch,
                                  seg_ticks=args.seg_ticks,
                                  tick_sample=args.tick_sample, seed=args.seed,
                                  want=want, round_idx=ri)
        if out is not None:
            _accumulate_match(out, acc)
            if raw is not None:
                raw_parts.append(raw)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    match = _finalize_match(acc)
    if raw_parts:
        raw = {k: np.concatenate([p[k] for p in raw_parts], axis=0)
               for k in raw_parts[0]}
    else:
        raw = None
    return match, raw


def _save_raw(raw: dict, raw_dir: Path, demo_path: Path, level_name: str,
              player_names: List[str], map_name: str) -> Path:
    """逐任务原始数据落盘为 .npz（float32 压缩，供以后画任意图）。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{level_name}__{demo_path.stem}.npz"
    data = {
        "round": raw["round"], "tick": raw["tick"], "player": raw["player"],
        "map_name": np.array([map_name]),
        "player_names": np.array(player_names),
    }
    for k in ("pre_tick", "pre_total", "pre_tokcount", "curves", "alive"):
        if k in raw:
            data[k] = raw[k]
    np.savez_compressed(out, **data)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════════════

def _setup_cjk_font() -> None:
    from matplotlib import font_manager
    candidates = ["PingFang SC", "Hiragino Sans GB", "Heiti SC",
                  "Arial Unicode MS", "Microsoft YaHei", "SimHei",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_charts(results: dict, out_dir: Path, summarize: str = "delta",
                prob: str = "per_tick", group_by: str = "level") -> None:
    """画 4 张图。group_by="match"：每场比赛一列并标注选手名；
    "level"：每个文件夹（级别）一列。"""
    _setup_cjk_font()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = results["levels"]
    n_levels = len(levels)
    colors = {lv: LEVEL_COLORS[i % len(LEVEL_COLORS)]
              for i, lv in enumerate(levels)}

    charts = [
        ("pretrain", "1_pretrain_logprob.png",
         "预训练模型：选手未来轨迹的平均对数概率",
         ("平均每 tick 校准认可度（log p - log max p，tick 等权，0 = 模型最认可）"
          if prob.startswith("calib") else
          "平均每 tick 对数概率（tick 等权，越高 = 越被模型认可）"),
         None),
        ("winrate", "2_winrate.png",
         "下游 win-rate：选手未来轨迹的胜率影响",
         "队伍胜率 4 秒变化（j=16 - j=0，CT 已翻转）",
         (-1.0, 1.0)),
        ("future_kill", "3_future_kill.png",
         "下游 future-kill：选手未来轨迹的击杀影响",
         "未来击杀概率 4 秒变化（j=16 - j=0）",
         (-1.0, 1.0)),
        ("alive_end", "4_alive_end.png",
         "下游 alive-end：选手未来轨迹的存活影响",
         "存活概率 4 秒变化（j=16 - j=0）",
         (-1.0, 1.0)),
    ]

    # ── 每级别取值辅助 ────────────────────────────────────────────────────
    def _pl_weight(key: str, pl: dict) -> float:
        """组内平均/颜色深浅的权重：预训练=存活 tick 数（tick 等权），
        下游=任务数。"""
        if key == "pretrain":
            w = (pl["pretrain"].get("n_ticks_alive")
                 or pl["pretrain"].get("n_ticks") or 0)
        else:
            w = pl["metrics"].get(key, {}).get("n_ticks") or 0
        return float(w)

    def _values_for(key: str, lv: str):
        """该级别所有选手的 (名字, 值, 权重) 列表（权重=tick 数，组内平均用它）。"""
        out = []
        for match in results["matches"].get(lv, []):
            for pl in match["players"]:
                if key == "pretrain":
                    v = pl["pretrain"].get(prob)
                else:
                    v = pl["metrics"].get(key, {}).get(summarize)
                w = _pl_weight(key, pl)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    out.append((pl["name"], float(v), w))
        return out

    def _lv_weighted_mean(pairs):
        """组内平均：按每个选手的存活 tick 数加权（tick 多 = 存活时间长 = 证据更多）。"""
        if not pairs:
            return float("nan")
        ws = np.array([w for _, w in pairs], dtype=np.float64)
        vs = np.array([v for v, _ in pairs], dtype=np.float64)
        if ws.sum() <= 0:
            return float(np.mean(vs))
        return float(np.sum(vs * ws) / np.sum(ws))

    def _point_colors(lv: str, weights, max_w: float):
        """散点颜色：tick 越多的玩家颜色越深（alpha 随 tick 数线性加深）。"""
        alphas = [0.35 + 0.65 * (w / max_w) if max_w > 0 else 0.5
                  for w in weights]
        return [to_rgba(colors[lv], a) for a in alphas]

    def _finalize(fig, ax, fname, title, ylabel, ylim, xlabel, lvl_means,
                  lvl_order=None):
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        handles = []
        order = lvl_order if lvl_order is not None else levels
        for lv in order:
            n_m = len(results["matches"].get(lv, []))
            handles.append(plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=colors[lv], markersize=8,
                label=f"{lv}（{n_m} 场）"))
        for lv, mv in lvl_means:
            handles.append(plt.Line2D(
                [0], [0], color=colors[lv], linewidth=1.6,
                linestyle="-", label=f"{lv} 平均 = {mv:.3f}"))
        ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  图已保存: {out_path}")

    if group_by == "match":
        # 每场比赛一列（原布局：10 名选手 + 名字标注）
        n_matches_total = sum(len(results["matches"].get(lv, [])) for lv in levels)
        for key, fname, title, ylabel, ylim in charts:
            max_w = max((_pl_weight(key, pl)
                         for lv in levels
                         for m in results["matches"].get(lv, [])
                         for pl in m["players"]), default=1.0) or 1.0
            fig, ax = plt.subplots(
                figsize=(max(10.0, 0.5 * n_matches_total + 4.0), 6.8))
            x = 0.0
            lvl_means = []
            for li, lv in enumerate(levels):
                matches = results["matches"].get(lv, [])
                lvl_pts = []
                for match in matches:
                    pts = []
                    for pl in match["players"]:
                        if key == "pretrain":
                            v = pl["pretrain"].get(prob)
                        else:
                            v = pl["metrics"].get(key, {}).get(summarize)
                        w = _pl_weight(key, pl)
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            pts.append((pl["name"], float(v), w))
                            lvl_pts.append((float(v), w))
                    if not pts:
                        continue   # 该场此指标无数据（--models 子集），不占列
                    xs = [x + (pi - 4.5) * 0.06 for pi in range(len(pts))]
                    ys = [v for _, v, _ in pts]
                    ax.scatter(xs, ys,
                               c=_point_colors(lv, [w for _, _, w in pts], max_w),
                               s=24, edgecolors="white", linewidths=0.4, zorder=3)
                    for (px, py, name) in zip(xs, ys, [n for n, _, _ in pts]):
                        ax.annotate(name, (px, py),
                                    textcoords="offset points",
                                    xytext=(0, 3 + (len(name) % 3) * 2),
                                    fontsize=5.2, ha="center",
                                    color=colors[lv], alpha=0.9)
                    x += 1.0
                if lvl_pts:
                    lvl_means.append((lv, _lv_weighted_mean(lvl_pts)))
                if li < n_levels - 1:
                    ax.axvline(x - 0.5, color="#bbbbbb", linewidth=0.8,
                               linestyle="--", alpha=0.6)
            ax.set_xlim(-0.8, x - 0.2)
            if not lvl_means:
                print(f"    {fname}: 无数据（该模型未运行），图已保存但为空")
            _finalize(fig, ax, fname, title, ylabel, ylim,
                      "比赛（每场比赛一列，10 名选手）", lvl_means)
        return

    # ── 默认：每个文件夹（级别）一列，列按均值升序排列 ────────────────────
    rng = np.random.default_rng(20240818)  # 固定抖动种子，重画保持一致
    for key, fname, title, ylabel, ylim in charts:
        # 只保留本图有数据的级别（--models 子集时，缺指标的级别不出现空列）
        def _lv_mean(lv):
            return _lv_weighted_mean([(v, w) for _, v, w in _values_for(key, lv)])
        lv_order = sorted(
            [lv for lv in levels if _values_for(key, lv)], key=_lv_mean)
        n_cols = len(lv_order)
        max_w = max((w for lv in lv_order
                     for _, _, w in _values_for(key, lv)), default=1.0) or 1.0
        fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * n_cols + 5.0), 6.8))
        lvl_means = []
        for li, lv in enumerate(lv_order):
            pts = _values_for(key, lv)
            if not pts:
                continue
            vals = [v for _, v, _ in pts]
            xs = li + rng.uniform(-0.32, 0.32, size=len(pts))
            ax.scatter(xs, vals,
                       c=_point_colors(lv, [w for _, _, w in pts], max_w),
                       s=22, edgecolors="white", linewidths=0.4, zorder=3)
            if len(pts) <= 30:
                for (px, py, name) in zip(xs, vals, [n for n, _, _ in pts]):
                    ax.annotate(name, (px, py),
                                textcoords="offset points",
                                xytext=(0, 3), fontsize=5.5, ha="center",
                                color=colors[lv], alpha=0.9)
            else:
                print(f"    {fname}: {lv} 有 {len(pts)} 个点，"
                      f"省略选手名标注（想看名字请用 --group-by match）")
            mv = _lv_weighted_mean([(v, w) for _, v, w in pts])
            lvl_means.append((lv, mv))
            ax.axhline(mv, color=colors[lv], linewidth=1.4,
                       linestyle="-", alpha=0.85, zorder=2)
            ax.annotate(f"均值 {mv:.3f}", (li + 0.36, mv),
                        fontsize=8, color=colors[lv], ha="left", va="center",
                        alpha=0.9)
        if not lvl_means:
            print(f"    {fname}: 无数据（该模型未运行），图已保存但为空")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([
            f"{lv}\n({len(results['matches'].get(lv, []))} 场 · "
            f"{len(_values_for(key, lv))} 人)" for lv in lv_order], fontsize=9)
        if n_cols > 0:
            ax.set_xlim(-0.6, n_cols - 0.4)
        _finalize(fig, ax, fname, title, ylabel, ylim,
                  "级别（每个文件夹一列，按均值升序）", lvl_means, lv_order)


# ═══════════════════════════════════════════════════════════════════════
# 结果保存 / 汇总
# ═══════════════════════════════════════════════════════════════════════

def _save_results(results: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


def print_summary(results: dict, prob: str = "per_tick") -> None:
    print("\n" + "=" * 72)
    print("汇总（每级别：所有选手的均值 ± 标准差，均值按存活 tick 数加权 = tick 等权）")
    print("    下游 = delta：第 4 秒 (j=16) vs cond (j=0) 的概率变化")
    if prob.startswith("calib"):
        print(f"    预训练口径 = {prob}：log p - log max(p)（0 = 模型最认可的动作）")
    print("=" * 72)
    for lv in results["levels"]:
        matches = results["matches"].get(lv, [])
        if not matches:
            continue
        # (值, 权重) 对；预训练权重 = 存活 tick 数（tick 等权），下游 = 任务数
        pre_pairs, wr_pairs, fk_pairs, ae_pairs = [], [], [], []
        for match in matches:
            for pl in match["players"]:
                pv = pl["pretrain"].get(prob)
                if pv is not None and not (isinstance(pv, float) and np.isnan(pv)):
                    w = (pl["pretrain"].get("n_ticks_alive")
                         or pl["pretrain"].get("n_ticks") or 1)
                    pre_pairs.append((pv, w))
                for m, key in (("winrate", wr_pairs), ("future_kill", fk_pairs),
                               ("alive_end", ae_pairs)):
                    v = pl["metrics"].get(m, {}).get("delta")
                    if v is not None and not np.isnan(v):
                        w = pl["metrics"].get(m, {}).get("n_ticks") or 1
                        key.append((v, w))
        def _fmt(pairs):
            if not pairs:
                return "n/a"
            ws = np.array([w for _, w in pairs], dtype=np.float64)
            vs = np.array([v for v, _ in pairs], dtype=np.float64)
            a = float(np.sum(vs * ws) / np.sum(ws)) if ws.sum() > 0 \
                else float(np.mean(vs))
            s = float(np.sqrt(np.sum(ws * (vs - a) ** 2) / np.sum(ws))) \
                if ws.sum() > 0 else float(np.std(vs))
            return f"{a:.4f} ± {s:.4f}  (n={len(pairs)})"
        print(f"\n[{lv}] ({len(matches)} 场)")
        pre_label = ("预训练 校准认可度(log p-log max p) : "
                     if prob.startswith("calib")
                     else "预训练 轨迹平均对数概率 : ")
        print(f"  {pre_label}{_fmt(pre_pairs)}")
        print(f"  win-rate     delta 影响 : {_fmt(wr_pairs)}")
        print(f"  future-kill  delta 影响 : {_fmt(fk_pairs)}")
        print(f"  alive-end    delta 影响 : {_fmt(ae_pairs)}")


# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="批量评估 demo 目录：预训练轨迹概率 + 3 个下游指标影响")
    ap.add_argument("--demos-dir", default="/Users/wanjungu/Downloads/test_demos",
                    help="demo 根目录（每个子文件夹 = 一个级别）")
    ap.add_argument("--model-dir",
                    default="/Users/wanjungu/Downloads/cs-net-v4-preview",
                    help="模型目录（自动读取预训练底座 + 3 个下游模型）")
    ap.add_argument("--config", default="config/pretrain-a100.yaml")
    ap.add_argument("--maps-dir", default="maps/optimized_obj_files")
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--out", default="outputs/demo_eval",
                    help="输出目录（4 张图 + results.json + cache/）")
    ap.add_argument("--cache-dir", default=None, help="解析缓存目录（默认 <out>/cache）")
    ap.add_argument("--batch", type=int, default=64,
                    help="decoder 批大小（内存紧张就调小，例如 2/8/16；跑得动可调大 128/256）")
    ap.add_argument("--win-batch", type=int, default=16,
                    help="条件窗口批大小（每批 B 个 16-tick 窗口）")
    ap.add_argument("--seg-ticks", type=int, default=64,
                    help="embedder 分块 tick 数（控制深度编码峰值内存，越大越快但更吃内存）")
    ap.add_argument("--tick-sample", type=float, default=1.0,
                    help="活着的时刻采样比例 (0,1]：例如 0.2 = 只算 20%% 的 tick，"
                         "速度快约 5 倍，均值仍无偏；1.0 = 全量")
    ap.add_argument("--seed", type=int, default=12345,
                    help="采样随机种子（保证可复现）")
    ap.add_argument("--levels", nargs="*", default=None,
                    help="只评估指定子文件夹")
    ap.add_argument("--max-matches", type=int, default=0, help="每级别最多场数（0=全部）")
    ap.add_argument("--max-rounds", type=int, default=0, help="每场最多回合数（0=全部）")
    ap.add_argument("--quiet", action="store_true", help="减少输出")
    ap.add_argument("--replot", action="store_true",
                    help="只从 results.json 重画图（不重新计算）")
    ap.add_argument("--summarize", choices=["mean", "end", "delta"], default="delta",
                    help="下游指标汇总方式（重画图时生效）：delta=第 4 秒 vs cond 的"
                         "变化（默认）；mean=未来存活 tick 平均；end=最后存活 tick 值")
    ap.add_argument("--prob",
                    choices=["per_tick", "per_token", "total_per_tick",
                             "calib", "calib_total_per_tick"],
                    default="per_tick",
                    help="轨迹概率汇总方式（重画图时生效）：per_tick=tick 等权；"
                         "per_token=token 等权；total_per_tick=路径总 logp 的路径等权；"
                         "calib=log p - log max(p) 的 tick 等权（0 = 模型最认可的"
                         "动作，剥离整体置信度）；calib_total_per_tick=同样的校准"
                         "差值按路径等权")
    ap.add_argument("--group-by", choices=["level", "match"], default="level",
                    help="图的 x 轴分组：level=每个文件夹一列（默认）；"
                         "match=每场比赛一列并标注选手名（重画图时生效）")
    ap.add_argument("--models", nargs="*",
                    choices=["pretrain", *DOWNSTREAM_TASKS], default=None,
                    help="只跑指定的模型（默认全部）。例：--models pretrain 只跑预训练；"
                         "--models winrate alive_end 只跑这两个下游。"
                         "未跑的模型结果为空，图/汇总自动跳过")
    ap.add_argument("--new-only", action="store_true",
                    help="只评估新增的 demo（对比已有 results.json，旧场次直接沿用、"
                         "不重新计算）。要求 tick-sample / seed / max-rounds 与上次"
                         "一致，否则拒绝；--models 可为子集，新场次缺失的指标"
                         "不出现在对应图/汇总中（图保持旧数据）")
    ap.add_argument("--no-save-raw", action="store_true",
                    help="不保存逐任务原始数据（默认保存到 <out>/raw/*.npz，"
                         "含每任务的逐 tick 对数概率与 3 个下游曲线，"
                         "供以后画任意图，无需重算）")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    if args.cache_dir is None:
        args.cache_dir = str(out_dir / "cache")

    if args.replot:
        rp = out_dir / "results.json"
        if not rp.exists():
            raise SystemExit(f"results.json 不存在: {rp}（先运行一次完整评估）")
        with open(rp, "r", encoding="utf-8") as f:
            results = json.load(f)
        plot_charts(results, out_dir, args.summarize, args.prob, args.group_by)
        print_summary(results, prob=args.prob)
        return

    t_start = time.time()
    results = run(args)
    _save_results(results, out_dir)
    plot_charts(results, out_dir, args.summarize, args.prob, args.group_by)
    print_summary(results, prob=args.prob)
    print(f"\n总用时: {(time.time() - t_start) / 60:.1f} 分钟")
    print(f"结果: {out_dir}/results.json")


if __name__ == "__main__":
    main()
