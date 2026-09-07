#!/usr/bin/env python3
"""
CS2 训练数据管线 — CLI 入口。

将 V2 JSON 文件（demo_parser 输出）转换为 WebDataset shard，
每个 sample 为一个完整回合的时序数据。

Usage:
    python scripts/create_training_data.py \
        --input-dir data/demos/json \
        --maps-dir maps/optimized_obj_files \
        --output-dir data/dataset \
        --test-split 0.02 \
        --seed 42 \
        --verbose
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

import tqdm

# 确保项目根目录和 scripts/ 目录下的包可导入
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training_data.config import MAP_CONFIG, MAP_NAME_TO_IDX, weapon_name_to_idx
from training_data.map_loader import get_map_geometry, clear_cache
from training_data.round_processor import process_round
from training_data.wds_writer import (
    create_wds_writer,
    find_start_shard,
    write_samples,
)


def load_v2_json(file_path: Path) -> dict:
    """加载 V2 JSON 文件（支持 .json 和 .json.gz）。"""
    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


def _convert_inventory_indices(data: dict) -> None:
    """将 V2 JSON 的动态 weapon 索引转换为 config.py 的规范 WEAPON_TO_IDX。

    V2 JSON 的 weapon_lookup 是按 demo 中出现武器字母序动态生成的，
    与 config.py WEAPON_NAMES 的固定索引不一致。此函数将 inventory
    中的动态索引原地替换为规范索引。
    """
    weapon_lookup: dict[str, int] = data.get("weapons", {})
    if not weapon_lookup:
        return
    # 反向映射: dynamic_idx → weapon_name
    idx_to_name: dict[int, str] = {idx: name for name, idx in weapon_lookup.items()}

    for rd in data.get("rounds", []):
        for p in rd.get("players", []):
            inv_list = p.get("inventory", [])
            for t, inv_per_tick in enumerate(inv_list):
                if isinstance(inv_per_tick, list):
                    inv_list[t] = [
                        weapon_name_to_idx(idx_to_name.get(int(di), "Knife"))
                        for di in inv_per_tick
                    ]


def find_json_files(input_dir: Path) -> List[Path]:
    """递归查找所有 V2 JSON 文件。"""
    files = []
    for pattern in ["*.json.gz", "*.json"]:
        files.extend(input_dir.rglob(pattern))
    # 去重：如果有 .json 和 .json.gz，优先 .gz
    seen = set()
    result = []
    for f in sorted(files):
        stem = f.name.replace(".gz", "").replace(".json", "") + ".json"
        if stem not in seen:
            seen.add(stem)
            result.append(f)
    return result


def load_done(done_file: Path) -> Set[str]:
    """加载已处理的文件列表。"""
    if not done_file.exists():
        return set()
    return set(
        line.strip()
        for line in done_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def mark_done(done_file: Path, filename: str):
    """标记文件已处理。"""
    with done_file.open("a", encoding="utf-8") as f:
        f.write(filename + "\n")


def process_match_file(
    file_path: Path,
    map_obj_dir: Path,
    tick_interval: float,
    compute_depth: bool,
    verbose: bool,
) -> List[dict]:
    """
    处理一个 V2 JSON 文件 → 该场比赛所有回合的 sample 列表。

    Returns:
        samples: 每个元素是一个 round sample dict
    """
    if verbose:
        print(f"  Loading: {file_path.name}")

    data = load_v2_json(file_path)

    # 将 V2 JSON 的动态 weapon 索引转换为 config.py 规范索引
    _convert_inventory_indices(data)

    # 验证格式（不再使用 filter_data，保留原始 kill 事件以正确标注最后一个 tick）
    fmt = data.get("format", "")
    if "v2" not in fmt and fmt != "cs2.demo.v2":
        print(f"  ⚠ Unknown format '{fmt}' in {file_path.name}, trying anyway...")

    map_name = data.get("map", data.get("map_name", "unknown"))
    if map_name not in MAP_CONFIG:
        print(f"  Unknown map '{map_name}' in {file_path.name}, skipping.")
        return []

    # 跳过没有 OBJ 的地图
    obj_path = map_obj_dir / f"{map_name}.obj"
    if not obj_path.exists():
        if verbose:
            print(f"  Skipping {file_path.name}: no OBJ for map '{map_name}'")
        return []

    # 加载地图几何体（OBJ 已在上方验证存在）
    map_geom = None
    if compute_depth:
        map_geom = get_map_geometry(map_name, map_obj_dir)

    # 比赛元数据
    source_name = file_path.name
    players_meta = data.get("players", [])
    places = data.get("places", {})
    match_teams = [source_name]  # 从文件名推断，后续可改进

    rounds = data.get("rounds", [])
    if not rounds:
        print(f"  ⚠ No rounds in {file_path.name}, skipping.")
        return []

    samples = []
    skipped = 0

    for rd in rounds:
        # 注入 map_name（V2 有些字段在顶层）
        if "map_name" not in rd and "map" not in rd:
            rd["map_name"] = map_name
        if "map" not in rd:
            rd["map"] = map_name

        # 跳过空回合
        if len(rd.get("ticks", [])) == 0:
            skipped += 1
            continue

        # 跳过炸弹位置缺失的回合（bomb_position 存在 null）
        bomb_positions = rd.get("bomb_position", [])
        if bomb_positions and any(bp is None for bp in bomb_positions):
            skipped += 1
            continue

        try:
            sample = process_round(
                rd,
                map_geom=map_geom,
                source_file=source_name,
                match_teams=match_teams,
                players_meta=players_meta,
                tick_interval=tick_interval,
                compute_depth=compute_depth,
                places=places,
            )
            samples.append(sample)
        except Exception as exc:
            skipped += 1
            if verbose:
                print(f"    ⚠ Round {rd.get('id', '?')} failed: {exc}")

    if verbose:
        n_rounds = len(rounds)
        print(f"    {n_rounds - skipped}/{n_rounds} rounds processed"
              + (f" ({skipped} skipped)" if skipped else ""))

    return samples


def _worker_process_file(args: tuple) -> tuple:
    """
    工作进程入口：处理单个 JSON 文件，返回所有回合的 sample 列表。

    必须在模块顶层定义（ProcessPoolExecutor 的 spawn 要求）。

    Args:
        args: (file_path, maps_dir, tick_interval, compute_depth, verbose)

    Returns:
        (filename, samples, error_msg_or_None)
    """
    file_path, maps_dir, tick_interval, compute_depth, verbose = args
    try:
        samples = process_match_file(
            file_path, maps_dir, tick_interval, compute_depth, verbose
        )
        return (file_path.name, samples, None)
    except Exception as exc:
        msg = f"{exc}"
        if verbose:
            msg = f"{exc}\n{traceback.format_exc()}"
        return (file_path.name, [], msg)


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Training Data Pipeline — V2 JSON → WebDataset"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="V2 JSON 文件目录（.json 或 .json.gz）"
    )
    parser.add_argument(
        "--maps-dir", default="maps/optimized_obj_files",
        help="优化后的地图 OBJ 文件目录"
    )
    parser.add_argument(
        "--output-dir", default="data/dataset",
        help="WebDataset shard 输出目录"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.02,
        help="每个回合分配到测试集的概率（默认 0.02 = 2%%）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，用于 train/test 划分的可复现性（默认 42）"
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="禁用深度图生成（加速处理）"
    )
    parser.add_argument(
        "--tick-interval", type=float, default=0.25,
        help="Tick 采样间隔（秒）"
    )
    parser.add_argument(
        "--max-shard-size", type=int, default=5,
        help="每个 shard 最大大小（GB）"
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="并行处理 worker 数（0=自动检测 CPU 核心数）"
    )
    parser.add_argument(
        "--done-file", default="",
        help="已处理文件列表（断点续传）"
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
    maps_dir = Path(args.maps_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input-dir not found: {input_dir}")
        sys.exit(1)

    if not maps_dir.exists():
        print(f"Error: maps-dir not found: {maps_dir}")
        sys.exit(1)

    compute_depth = not args.no_depth

    # ── 查找 JSON 文件 ──────────────────────────────────────────────────
    json_files = find_json_files(input_dir)
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files in {input_dir}")
    if compute_depth:
        print(f"Depth maps: enabled (simplified, {args.maps_dir})")
    else:
        print("Depth maps: disabled")

    # ── 断点续传 ────────────────────────────────────────────────────────
    done_file = Path(args.done_file) if args.done_file else output_dir / "processed.txt"
    processed = load_done(done_file)
    pending = [f for f in json_files if f.name not in processed]
    print(f"Already processed: {len(processed)}, pending: {len(pending)}")

    if args.dry_run:
        print(f"Dry run — would process {len(pending)} files")
        # 统计回合数
        total_rounds = 0
        for fp in pending[:5]:  # 只采样前 5 个
            try:
                data = load_v2_json(fp)
                total_rounds += len(data.get("rounds", []))
            except Exception:
                pass
        print(f"Sample: ~{total_rounds} rounds in first 5 files")
        return

    # ── 随机种子 ──────────────────────────────────────────────────────────
    random.seed(args.seed)

    # ── 训练/测试划分 ────────────────────────────────────────────────────
    # 每个回合按概率随机分配到 test（默认 5%）
    test_split = args.test_split

    # ── 创建 ShardWriter ────────────────────────────────────────────────
    start_train = find_start_shard(output_dir, "train")
    start_test = find_start_shard(output_dir, "test")

    train_sink = create_wds_writer(
        output_dir, "train",
        maxsize=args.max_shard_size * 1024 ** 3,
        start_shard=start_train,
    )
    test_sink = create_wds_writer(
        output_dir, "test",
        maxsize=args.max_shard_size * 1024 ** 3,
        start_shard=start_test,
    )

    print(f"Train shards start at: {start_train:05d}")
    print(f"Test shards start at:  {start_test:05d}")

    # ── 并行处理文件 ────────────────────────────────────────────────────
    workers = args.workers if args.workers > 0 else os.cpu_count() or 4
    if args.verbose:
        print(f"Using {workers} workers")

    try:
        train_count = 0
        test_count = 0
        total_count = 0
        completed = 0
        map_rounds = {}        # map_name → round count
        last_summary_pct = 0

        worker_args = [
            (fp, maps_dir, args.tick_interval, compute_depth, False)
            for fp in pending
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # dict 需要在 as_completed 外部持有以支持 pop（见下方），
            # 每次处理完一个 future 立刻 pop 释放其缓存的 result，
            # 避免所有 results 积压在内存中被 OOM Killer 杀掉。
            future_to_file = {
                executor.submit(_worker_process_file, wargs): wargs[0].name
                for wargs in worker_args
            }

            pbar = tqdm.tqdm(
                as_completed(future_to_file),
                total=len(pending),
                desc="Processing",
                unit="file",
            )
            for future in pbar:
                # 立即 pop 释放 dict 对 future 的引用，防止 result 内存累积
                future_to_file.pop(future)
                filename, samples, error = future.result()
                completed += 1

                if error:
                    pbar.write(f"  ❌ [{completed}/{len(pending)}] {filename}: {error}")
                    continue

                if not samples:
                    continue

                # 统计每地图回合数
                for s in samples:
                    mn = s.get("meta", {}).get("map_name", "unknown")
                    map_rounds[mn] = map_rounds.get(mn, 0) + 1

                for sample in samples:
                    if random.random() < test_split:
                        write_samples(test_sink, [sample])
                        test_count += 1
                    else:
                        write_samples(train_sink, [sample])
                        train_count += 1

                total_count += len(samples)
                mark_done(done_file, filename)

                # 主动释放本轮的大块 numpy 数组
                del samples

                # 每 50 个文件强制回收一次内存
                if completed % 50 == 0:
                    gc.collect()

                pbar.set_postfix(train=train_count, test=test_count)

                # 每 5% 打印一次地图回合分布
                current_pct = int(completed / len(pending) * 100)
                current_milestone = (current_pct // 5) * 5
                if current_milestone > last_summary_pct and current_milestone >= 5:
                    last_summary_pct = current_milestone
                    pbar.write(f"\n  ── 地图回合分布 ({current_milestone}%) ──")
                    for mn in sorted(map_rounds.keys(), key=lambda k: -map_rounds[k]):
                        pbar.write(f"    {mn:15s}  {map_rounds[mn]:6d} rounds")
                    pbar.write("")

    finally:
        train_sink.close()
        test_sink.close()

    print(f"\nDone!")
    print(f"  Train samples: {train_count}")
    print(f"  Test samples:  {test_count}")
    print(f"  Total samples: {total_count}")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
