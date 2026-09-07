#!/usr/bin/env python3
"""
CS2 预训练数据管线 — CLI 入口。

从 round-level WebDataset shards 提取固定长度滑动窗口，
生成 pretrain WebDataset shards。

每个 pretrain sample = 64 tick 输入 + 64 tick 输出（相机运动标签）。

Usage:
    python scripts/create_pretrain_data.py \
        --input-dir data/dataset \
        --output-dir data/pretrain_dataset \
        --n-ticks 64 --stride 16 \
        --workers 4 --verbose
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set

import tqdm

# 确保项目根目录和 scripts/ 目录下的包可导入
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training_data.pretrain_processor import PretrainWindowExtractor
from training_data.wds_reader import decode_sample, scan_shards
from training_data.wds_writer import (
    create_wds_writer,
    find_start_shard,
    write_samples,
)


def _worker_process_shard(args: tuple) -> tuple:
    """
    工作进程入口：处理一个 round shard，返回所有窗口 sample。

    必须在模块顶层定义（ProcessPoolExecutor 的 spawn 要求）。

    Args:
        args: (shard_path, n_ticks, stride, min_input_ticks, min_output_ticks, require_camera)

    Returns:
        (shard_name, windows, error_msg_or_None)
    """
    (
        shard_path, n_ticks, stride,
        min_input_ticks, min_output_ticks, require_camera,
    ) = args

    try:
        import webdataset as wds

        extractor = PretrainWindowExtractor(
            n_ticks=n_ticks,
            stride=stride,
            min_input_ticks=min_input_ticks,
            min_output_ticks=min_output_ticks,
            require_camera=require_camera,
        )

        windows = []
        skipped_no_camera = 0
        skipped_too_short = 0
        total_rounds = 0

        # 跳过空 shard
        if shard_path.stat().st_size < 1024:
            return (shard_path.name, [], None)

        dataset = wds.WebDataset(
            [str(shard_path)], shardshuffle=False, empty_check=False
        )

        for raw in dataset:
            sample = decode_sample(raw)
            total_rounds += 1

            # 检查 camera label
            if "label_camera" not in sample:
                skipped_no_camera += 1
                continue

            T = sample["player_pos"].shape[0]
            if T < min_input_ticks:
                skipped_too_short += 1
                continue

            try:
                round_windows = extractor.extract_windows(sample)
                windows.extend(round_windows)
            except Exception:
                # 单个回合失败不中断整个 shard
                continue

            # 释放原始 sample 的大块内存
            del sample

        return (
            shard_path.name,
            windows,
            None,
            {
                "total_rounds": total_rounds,
                "skipped_no_camera": skipped_no_camera,
                "skipped_too_short": skipped_too_short,
            },
        )

    except Exception as exc:
        msg = f"{exc}\n{traceback.format_exc()}"
        return (shard_path.name, [], msg, {})


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Pretraining Data Pipeline — Round shards → Window shards"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Round-level WebDataset 目录（含 train/ test/ 子目录）"
    )
    parser.add_argument(
        "--output-dir", default="data/pretrain_dataset",
        help="Pretrain WebDataset 输出目录"
    )
    parser.add_argument(
        "--n-ticks", type=int, default=64,
        help="输入/输出窗口 tick 数（默认 64）"
    )
    parser.add_argument(
        "--stride", type=int, default=16,
        help="滑动窗口步长（默认 16）"
    )
    parser.add_argument(
        "--min-input-ticks", type=int, default=32,
        help="最少有效输入 tick 数（默认 32）"
    )
    parser.add_argument(
        "--min-output-ticks", type=int, default=1,
        help="最少有效输出 tick 数（默认 1）"
    )
    parser.add_argument(
        "--max-shard-size", type=int, default=5,
        help="每个输出 shard 最大大小（GB，默认 5）"
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="并行 worker 数（0=自动检测 CPU 核心数）"
    )
    parser.add_argument(
        "--require-camera", action="store_true",
        help="缺少 camera label 时报错而非跳过"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只统计不写入"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input-dir not found: {input_dir}")
        sys.exit(1)

    # ── 扫描 shard ──────────────────────────────────────────────────────────
    shard_map = scan_shards(input_dir)
    all_shards = shard_map.get("train", []) + shard_map.get("test", [])
    if not all_shards:
        print(f"No shards found in {input_dir}/{{train,test}}/")
        sys.exit(1)

    train_shards = shard_map.get("train", [])
    test_shards = shard_map.get("test", [])

    print(f"Found {len(train_shards)} train + {len(test_shards)} test shards "
          f"in {input_dir}")
    print(f"Config: n_ticks={args.n_ticks}, stride={args.stride}, "
          f"min_input={args.min_input_ticks}, min_output={args.min_output_ticks}")
    print(f"Output: {output_dir}/")

    # 统计：总 round 数
    if args.verbose or args.dry_run:
        # 采样统计
        import webdataset as wds
        total_rounds_est = 0
        sample_shards = all_shards[:3]
        for sp in sample_shards:
            if sp.stat().st_size < 1024:
                continue
            ds = wds.WebDataset([str(sp)], shardshuffle=False, empty_check=False)
            count = 0
            for _ in ds:
                count += 1
                if count >= 50:
                    break
            total_rounds_est += count
        avg_per_shard = total_rounds_est / max(len(sample_shards), 1)
        est_total = int(avg_per_shard * len(all_shards))
        print(f"Estimated ~{est_total} round samples total "
              f"(avg {avg_per_shard:.0f}/shard)")

    if args.dry_run:
        print("Dry run — no output written.")
        return

    # ── 并行处理 shard ──────────────────────────────────────────────────────
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) // 2)
    if args.verbose:
        print(f"Using {workers} workers")

    # 分别处理 train 和 test，保持 split 一致
    for split_name, shards in [("train", train_shards), ("test", test_shards)]:
        if not shards:
            continue

        start_shard = find_start_shard(output_dir, split_name)
        sink = create_wds_writer(
            output_dir, split_name,
            maxsize=args.max_shard_size * 1024 ** 3,
            start_shard=start_shard,
        )

        try:
            worker_args = [
                (
                    sp, args.n_ticks, args.stride,
                    args.min_input_ticks, args.min_output_ticks,
                    args.require_camera,
                )
                for sp in shards
            ]

            total_windows = 0
            total_rounds_processed = 0
            total_skipped_camera = 0
            total_skipped_short = 0
            completed = 0

            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_shard = {
                    executor.submit(_worker_process_shard, wargs): wargs[0].name
                    for wargs in worker_args
                }

                pbar = tqdm.tqdm(
                    as_completed(future_to_shard),
                    total=len(shards),
                    desc=f"Processing {split_name}",
                    unit="shard",
                )
                for future in pbar:
                    future_to_shard.pop(future)
                    shard_name, windows, error, stats = future.result()
                    completed += 1

                    if error:
                        pbar.write(f"  ❌ [{completed}/{len(shards)}] {shard_name}: {error}")
                        continue

                    total_rounds_processed += stats.get("total_rounds", 0)
                    total_skipped_camera += stats.get("skipped_no_camera", 0)
                    total_skipped_short += stats.get("skipped_too_short", 0)

                    if windows:
                        write_samples(sink, windows)
                        total_windows += len(windows)

                    # 释放本轮内存
                    del windows

                    if completed % 10 == 0:
                        gc.collect()

                    pbar.set_postfix(
                        windows=total_windows,
                        rounds=total_rounds_processed,
                    )

            print(f"\n  [{split_name}] Done: {total_windows} windows "
                  f"from {total_rounds_processed} rounds "
                  f"({total_skipped_camera} no-camera, "
                  f"{total_skipped_short} too-short)")

        finally:
            sink.close()

    print(f"\nDone! Output: {output_dir}/")
    print(f"  Train: {output_dir}/train/shards-*.tar")
    print(f"  Test:  {output_dir}/test/shards-*.tar")


if __name__ == "__main__":
    main()
