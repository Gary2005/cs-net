#!/usr/bin/env python3
"""
跨多个 sample / tick 评估预训练 checkpoint 的综合表现。

和 test_pretrain.py 的区别：
  - test_pretrain.py 只围绕单个 --tick 做一次评估；
  - 本脚本会遍历数据集里的多个 sample，并在每个 round 上采样多个 query tick，
    汇总 Teacher-Forcing 和 / 或 Auto-Regressive 指标，便于比较不同 step 的 checkpoint。

用法示例:
    python scripts/evaluate_pretrain.py \
        --config config/pretrain-a100.yaml \
        --checkpoint checkpoints/pretrain_61/step_0100000.pt \
        --data-dir examples/dataset \
        --split train \
        --max-samples 10 \
        --tick-step 16 \
        --mode both \
        --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import webdataset as wds
import yaml


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training_data.wds_reader import decode_sample
from test_pretrain import (
    build_model,
    evaluate_teacher_forcing,
    evaluate_autoregressive,
    extract_window_at_tick,
    N_PLAYERS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrain checkpoints over multiple samples/ticks"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="examples/dataset")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="最多评估多少个 round；0=全部")
    parser.add_argument("--tick-step", type=int, default=16,
                        help="每个 round 中每隔多少 tick 评估一次")
    parser.add_argument("--mode", choices=["tf", "ar", "both"], default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maps-dir", default="maps/optimized_obj_files")
    parser.add_argument("--quiet", action="store_true",
                        help="抑制 AR 逐 tick 诊断输出")
    parser.add_argument("--output-json", type=str, default=None,
                        help="将汇总指标写入 JSON 文件")
    return parser.parse_args()


def iter_shards(data_dir: Path, split: str):
    if split == "both":
        splits = ["train", "test"]
    else:
        splits = [split]
    for sp in splits:
        sp_dir = data_dir / sp
        if sp_dir.is_dir():
            for shard in sorted(sp_dir.glob("shards-*.tar")):
                yield shard


def iter_samples(data_dir: Path, split: str, max_samples: int):
    count = 0
    for shard in iter_shards(data_dir, split):
        dataset = wds.WebDataset([str(shard)], shardshuffle=False, empty_check=False)
        for raw in dataset:
            sample = decode_sample(raw)
            if "label_camera" not in sample or "player_pos" not in sample:
                continue
            yield sample
            count += 1
            if max_samples > 0 and count >= max_samples:
                return


def _safe_tick_range(round_T: int, n_ticks: int, tick_step: int):
    """返回能构造完整未来窗口的 query tick 列表。"""
    ticks = []
    if round_T <= 0:
        return ticks
    # extract_window_at_tick 要求 target_tick 之后还有 n_ticks-1 个未来 tick
    max_valid = round_T - 1
    for tick in range(0, round_T, tick_step):
        if tick + n_ticks - 1 >= round_T:
            continue
        if tick <= max_valid:
            ticks.append(tick)
    return ticks


def _align_ade_fde(pred_points, gt_points):
    """从预测/GT 轨迹点计算 ADE 和 FDE，跳过起始点。"""
    if not pred_points or not gt_points:
        return 0.0, 0.0
    n = min(len(pred_points), len(gt_points))
    if n <= 1:
        return 0.0, 0.0
    errs = []
    for i in range(1, n):
        pp, gp = pred_points[i], gt_points[i]
        errs.append(math.sqrt((pp[0]-gp[0])**2 + (pp[1]-gp[1])**2 + (pp[2]-gp[2])**2))
    ade = sum(errs) / len(errs)
    fde = errs[-1]
    return ade, fde


def _run_tf(model, window, condition_pos, device):
    res = evaluate_teacher_forcing(model, window, device, condition_pos=condition_pos)
    metrics = {
        "loss": float(res["loss"]),
        "token_acc": float(res["overall_acc"]),
    }
    for name, acc in res["per_token_type_acc"].items():
        metrics[f"token_acc_{name}"] = float(acc)
    return metrics


def _run_ar(model, sample, window, s, condition_pos, target_tick, device, maps_dir, quiet):
    stdout_cm = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with stdout_cm:
        res = evaluate_autoregressive(
            model, sample, window, s, condition_pos, target_tick,
            device, maps_dir,
        )

    metrics = {}
    tick_accs = [a for a in res["per_tick_ar_acc"] if a > 0]
    if tick_accs:
        metrics["ar_token_acc"] = float(np.mean(tick_accs))
    else:
        metrics["ar_token_acc"] = 0.0

    all_ade, all_fde = [], []
    for p in range(N_PLAYERS):
        if not res["initial_alive"][p]:
            continue
        ade, fde = _align_ade_fde(res["pred_positions"][p], res["gt_positions"][p])
        all_ade.append(ade)
        all_fde.append(fde)
    if all_ade:
        metrics["ar_ade"] = float(np.mean(all_ade))
        metrics["ar_fde"] = float(np.mean(all_fde))
    else:
        metrics["ar_ade"] = 0.0
        metrics["ar_fde"] = 0.0
    return metrics


def main():
    args = parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = _PROJECT_ROOT / data_dir

    with open(args.config, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f) or {}
    n_ticks = yaml_cfg.get("n_ticks", 16)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_cfg = build_model(yaml_cfg, args.checkpoint, device)

    tf_metrics = []
    ar_metrics = []
    total_windows = 0
    total_rounds = 0

    for sample in iter_samples(data_dir, args.split, args.max_samples):
        total_rounds += 1
        meta = sample.get("meta", {})
        round_T = meta.get("T", sample["player_pos"].shape[0])
        ticks = _safe_tick_range(round_T, n_ticks, args.tick_step)
        if not ticks:
            continue

        for target_tick in ticks:
            try:
                window, s, condition_pos = extract_window_at_tick(
                    sample, target_tick, n_ticks
                )
            except ValueError:
                continue

            total_windows += 1

            if args.mode in ("tf", "both"):
                tf_metrics.append(_run_tf(model, window, condition_pos, device))
            if args.mode in ("ar", "both"):
                ar_metrics.append(_run_ar(
                    model, sample, window, s, condition_pos, target_tick,
                    device, args.maps_dir, args.quiet,
                ))

        print(f"  round {total_rounds}: {len(ticks)} windows")

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_rounds": total_rounds,
        "n_windows": total_windows,
        "mode": args.mode,
    }

    def _aggregate(rows):
        if not rows:
            return {}
        keys = rows[0].keys()
        out = {}
        for key in keys:
            vals = [r[key] for r in rows if key in r]
            if vals:
                out[key] = float(np.mean(vals))
        return out

    tf_summary = _aggregate(tf_metrics) if tf_metrics else {}
    ar_summary = _aggregate(ar_metrics) if ar_metrics else {}
    summary["teacher_forcing"] = tf_summary
    summary["autoregressive"] = ar_summary

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / out_path
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
