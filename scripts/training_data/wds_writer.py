"""
WebDataset 写入器 — ShardWriter 封装。

处理 numpy 压缩 + JSON 元数据写入。
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Iterable, Set

import numpy as np
import webdataset as wds
import zstandard as zstd

# 压缩器（复用，level=3 平衡速度与压缩率）
_cctx = zstd.ZstdCompressor(level=3)

# 需要作为 .npy.zst 存储的 key
NUMPY_KEYS: Set[str] = {
    "round_seconds",
    "player_pos", "player_alive_mask",
    "player_state",
    "player_inv", "player_inv_mask",
    "player_rel_f", "player_rel_i", "player_rel_mask",
    "player_sound",
    "player_depth", "player_depth_mask",
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


def _get_dtype(key: str):
    """根据 key 后缀确定 numpy dtype。"""
    if key.endswith("_mask"):
        return bool
    if key.endswith("_i") or key == "proj_type" or key.startswith("player_inv"):
        return np.int32
    if key.startswith("map_idx"):
        return np.int32
    if key.startswith("label_nxt_kill") or key.startswith("label_nxt_death"):
        return np.int32
    if key.startswith("label_bombsite") or key.startswith("label_win_reason"):
        return np.int32
    # 默认 float32
    return np.float32


def encode_sample(sample: dict) -> dict:
    """
    将 sample dict 编码为 WebDataset 可写格式。

    - numpy 数组 → .npy.zst（zstandard 压缩）
    - meta → .json.zst
    """
    encoded = {}
    for key, value in sample.items():
        if key == "__key__":
            encoded[key] = value
            continue

        if key in NUMPY_KEYS:
            out_key = f"{key}.npy.zst"
            dtype = _get_dtype(key)
            arr = np.asarray(value, dtype=dtype)

            # 写入 buffer 并压缩
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            compressed = _cctx.compress(buf.getvalue())
            encoded[out_key] = compressed
            continue

        if key == "meta":
            out_key = "meta.json.zst"
            raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
            compressed = _cctx.compress(raw)
            encoded[out_key] = compressed
            continue

        # 其他值（不应该出现）
        encoded[key] = value

    return encoded


def create_wds_writer(
    output_dir: Path,
    split: str,
    maxsize: int = 5 * 1024 ** 3,  # 5 GB
    start_shard: int = 0,
) -> wds.ShardWriter:
    """
    创建 WebDataset ShardWriter。

    Args:
        output_dir: 输出根目录
        split: "train" 或 "test"
        maxsize: 每个 shard 最大字节数
        start_shard: 起始 shard 编号
    """
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(split_dir / "shards-%05d.tar")
    return wds.ShardWriter(pattern, start_shard=start_shard, maxsize=maxsize)


def find_start_shard(output_dir: Path, split: str) -> int:
    """查找已有 shard 的最大编号 + 1。"""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    max_idx = -1
    for path in split_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        # shards-00001.tar
        if name.startswith("shards-") and name.endswith(".tar"):
            try:
                idx = int(name[7:12])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    return max_idx + 1


def write_samples(
    sink: wds.ShardWriter,
    samples: Iterable[dict],
    verbose: bool = False,
):
    """批量写入 samples 到 ShardWriter。"""
    count = 0
    for sample in samples:
        encoded = encode_sample(sample)
        sink.write(encoded)
        count += 1
    if verbose and count > 0:
        print(f"  Wrote {count} samples")
