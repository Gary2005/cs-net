"""
PyTorch Dataset / DataLoader 集成。

提供 IterableDataset + collate_fn，可直接接入训练循环。

用法:
    from scripts.training_data.torch_dataset import CS2Dataset, collate_fn

    ds = CS2Dataset("data/dataset", split="train", shuffle_buffer=5000)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
    )
    for batch in loader:
        # batch["player_pos"]:  [B, max_T, 10, 3]
        # batch["T_mask"]:      [B, max_T]        True=valid tick
        ...

Token 结构（27 tokens）:
    tokens  0– 9: 玩家 (10), 每玩家含 pos/state/inv/rel/depth/sound
    token     10: 炸弹/全局 (1)
    tokens 11–26: 投掷物 (16), 每投掷物含 pos/type/dur

深度图增强:
    原始数据存储为 [T,10,64]（仅 log 距离），开启 augment_depth=True 后
    自动扩展为 [T,10,64,5]，每条射线包含:
        [log_dist, cos(yaw_offset), sin(yaw_offset), cos(pitch_offset), sin(pitch_offset)]
    让模型感知每条射线的空间方向。
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.utils.data
import webdataset as wds
import zstandard as zstd

# ── 常量 ──────────────────────────────────────────────────────────────────────────

N_PLAYERS = 10
N_PROJECTILES = 16
N_TOKENS = 27  # 10 players + 1 bomb + 16 projectiles
N_PLAYER_STATE_DIMS = 14
N_PLAYER_INV_SLOTS = 9
N_RELATIONS = 9
N_RELATION_DIMS = 14
N_SOUND_DIMS = 2
N_DEPTH_DIRS = 64
N_DEPTH_DIR_ENCODINGS = 4   # cos(yaw), sin(yaw), cos(pitch), sin(pitch)
N_DEPTH_FEATURES = 5         # log_dist + 4 angular encodings
N_BOMB_STATE_DIMS = 4

_dctx = zstd.ZstdDecompressor()

# ═══════════════════════════════════════════════════════════════════════════════════
# 深度射线方向定义 & 角度编码（预计算常量）
# ═══════════════════════════════════════════════════════════════════════════════════

# 64 条射线的方向定义 (yaw_offset_deg, pitch_offset_deg) — 5 层同心圈
#   中心层（ 0°）: 24 条，每 15° 一条
#   ±30° 层     : 各 12 条，每 30° 一条
#   ±60° 层     : 各  8 条，每 45° 一条
_DEPTH_RAW_DIRECTIONS: List[Tuple[float, float]] = [
    # === +60° 层  8 条 ===
    (0.0,   60.0), (45.0,  60.0), (90.0,  60.0), (135.0, 60.0),
    (180.0, 60.0), (225.0, 60.0), (270.0, 60.0), (315.0, 60.0),
    # === +30° 层 12 条 ===
    (0.0,   30.0), (30.0,  30.0), (60.0,  30.0), (90.0,  30.0),
    (120.0, 30.0), (150.0, 30.0), (180.0, 30.0), (210.0, 30.0),
    (240.0, 30.0), (270.0, 30.0), (300.0, 30.0), (330.0, 30.0),
    # === 0° 中心层 24 条 ===
    (0.0,    0.0), (15.0,   0.0), (30.0,   0.0), (45.0,   0.0),
    (60.0,   0.0), (75.0,   0.0), (90.0,   0.0), (105.0,  0.0),
    (120.0,  0.0), (135.0,  0.0), (150.0,  0.0), (165.0,  0.0),
    (180.0,  0.0), (195.0,  0.0), (210.0,  0.0), (225.0,  0.0),
    (240.0,  0.0), (255.0,  0.0), (270.0,  0.0), (285.0,  0.0),
    (300.0,  0.0), (315.0,  0.0), (330.0,  0.0), (345.0,  0.0),
    # === -30° 层 12 条 ===
    (0.0,   -30.0), (30.0,  -30.0), (60.0,  -30.0), (90.0,  -30.0),
    (120.0, -30.0), (150.0, -30.0), (180.0, -30.0), (210.0, -30.0),
    (240.0, -30.0), (270.0, -30.0), (300.0, -30.0), (330.0, -30.0),
    # === -60° 层  8 条 ===
    (0.0,   -60.0), (45.0,  -60.0), (90.0,  -60.0), (135.0, -60.0),
    (180.0, -60.0), (225.0, -60.0), (270.0, -60.0), (315.0, -60.0),
]


def _build_depth_dir_encodings() -> np.ndarray:
    """
    预计算 64 条射线方向的角度编码。

    Returns:
        [64, 4] float32 数组，每行:
        [cos(yaw), sin(yaw), cos(pitch), sin(pitch)]

    所有角度以度为单位，yaw/pitch 均为相对玩家朝向的偏移量。
    yaw_offset=0 表示玩家正前方，pitch_offset=0 表示水平方向。
    """
    enc = np.empty((N_DEPTH_DIRS, N_DEPTH_DIR_ENCODINGS), dtype=np.float32)
    for i, (yaw_deg, pitch_deg) in enumerate(_DEPTH_RAW_DIRECTIONS):
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        enc[i, 0] = math.cos(yaw)
        enc[i, 1] = math.sin(yaw)
        enc[i, 2] = math.cos(pitch)
        enc[i, 3] = math.sin(pitch)
    return enc


# 模块级预计算常量（numpy）
_DEPTH_DIR_ENC_NP: np.ndarray = _build_depth_dir_encodings()

# torch 版本（延迟创建，避免无 torch 环境下导入失败）
_DEPTH_DIR_ENC_TORCH: Optional[torch.Tensor] = None


def _get_depth_dir_enc_torch() -> torch.Tensor:
    """获取 torch 版深度方向编码（懒加载）。"""
    global _DEPTH_DIR_ENC_TORCH
    if _DEPTH_DIR_ENC_TORCH is None:
        _DEPTH_DIR_ENC_TORCH = torch.from_numpy(_DEPTH_DIR_ENC_NP)
    return _DEPTH_DIR_ENC_TORCH


def augment_depth_with_angles(sample: dict) -> dict:
    """
    将 player_depth 从 [T, 10, 64] 扩展为 [T, 10, 64, 5]。

    每条射线的特征:
        [log_dist, cos(yaw), sin(yaw), cos(pitch), sin(pitch)]

    角度编码让模型感知每条射线在玩家局部球坐标系中的方向，
    从而理解深度图的几何结构（如正前方 vs 侧方 vs 头顶）。

    同时也处理 player_depth_labels（用于 decoder per-step depth conditioning）。

    如果 sample 中没有 player_depth 或已经是 4D，不做处理直接返回。
    """
    for depth_key in ("player_depth", "player_depth_labels"):
        if depth_key not in sample:
            continue

        depth = sample[depth_key]
        if depth.ndim != 3:
            continue

        T, N, D = depth.shape
        assert D == N_DEPTH_DIRS, f"Expected {N_DEPTH_DIRS} depth dirs, got {D}"

        # 角度编码: [64, 4] → [1, 1, 64, 4] → [T, 10, 64, 4]
        if isinstance(depth, np.ndarray):
            dir_enc = _DEPTH_DIR_ENC_NP  # [64, 4]
            dir_expanded = np.broadcast_to(dir_enc[np.newaxis, np.newaxis, :, :],
                                            (T, N, N_DEPTH_DIRS, N_DEPTH_DIR_ENCODINGS))
            depth_aug = np.concatenate([depth[..., np.newaxis], dir_expanded], axis=-1)
        else:
            # torch tensor
            dir_enc = _get_depth_dir_enc_torch()  # [64, 4]
            dir_expanded = dir_enc.unsqueeze(0).unsqueeze(0).expand(T, N, -1, -1)
            depth_aug = torch.cat([depth.unsqueeze(-1), dir_expanded], dim=-1)

        sample[depth_key] = depth_aug  # [T, 10, 64, 5]
    return sample


# ═══════════════════════════════════════════════════════════════════════════════════
# Key 分类
# ═══════════════════════════════════════════════════════════════════════════════════

# 所有 numpy key（与 wds_writer.py 保持同步）
NUMPY_KEYS: Set[str] = {
    "player_pos", "player_alive_mask",
    "player_state",
    "player_inv", "player_inv_mask",
    "player_rel_f", "player_rel_i", "player_rel_mask",
    "round_seconds",
    "player_sound",
    "player_depth", "player_depth_mask",
    "player_depth_labels", "player_alive_mask_labels",  # decoder per-step depth
    "player_pos_labels", "player_angle_labels",        # decoder per-step xyz & angle
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
    "output_mask",
    "tick_times_input", "tick_times_output",
}

# bool 类型的 mask key（padding 位置填 False）
BOOL_KEYS: Set[str] = {
    k for k in NUMPY_KEYS if k.endswith("_mask")
}

# 特殊 mask: 在 padding 位置应填 True（表示"无效/忽略"）
_PAD_TRUE_MASKS: Set[str] = set()

# int32 类型的 key（非 mask）
INT_KEYS: Set[str] = {
    "player_inv", "player_rel_i",
    "proj_type", "proj_is_active", "map_idx",
    "label_nxt_kill", "label_nxt_death",
    "label_bombsite", "label_win_reason",
    "player_teams",   # 0=CT, 1=T, -1=未知（下游 CT 胜率聚合用）
}

# int key 在 padding 位置的默认值
_INT_PAD_VALUES: Dict[str, int] = {
    "label_nxt_kill": 10,   # 10 = 无更多击杀
    "label_nxt_death": 10,  # 10 = 无更多死亡
    "label_bombsite": 2,    # 2 = 未知
    "label_win_reason": 5,  # 5 = 其他
    "proj_type": -1,        # -1 = 空槽
}

# label key
LABEL_KEYS: Set[str] = {
    "label_winrate", "label_nxt_kill", "label_nxt_death",
    "label_alive_end", "label_bombsite", "label_win_reason",
    "label_camera",
}

# 按 token 类型分组的 key
PLAYER_KEYS: Set[str] = {
    "player_pos", "player_alive_mask",
    "player_state",
    "player_inv", "player_inv_mask",
    "player_rel_f", "player_rel_i", "player_rel_mask",
    "player_sound",
    "player_depth", "player_depth_mask",
}

BOMB_KEYS: Set[str] = {
    "bomb_pos",
    "bomb_state",
    "map_idx",
}

PROJ_KEYS: Set[str] = {
    "proj_pos",
    "proj_type", "proj_dur", "proj_mask", "proj_is_active",
}


# ═══════════════════════════════════════════════════════════════════════════════════
# 解码
# ═══════════════════════════════════════════════════════════════════════════════════

def decode_sample(raw: dict) -> dict:
    """
    解码一个 WebDataset sample。

    - .npy.zst → numpy array
    - .json.zst → dict（存为 "meta"）
    - __key__, __url__ 保留原始值
    """
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        if key == "__key__":
            result["__key__"] = value
            continue
        if key.startswith("__"):
            continue
        if key.endswith(".npy.zst"):
            name = key[:-8]
            decompressed = _dctx.decompress(value)
            result[name] = np.load(io.BytesIO(decompressed))
        elif key.endswith(".json.zst"):
            result["meta"] = json.loads(_dctx.decompress(value))
    return result


def sample_to_torch(sample: dict) -> dict:
    """
    将解码后的 numpy sample 转为 torch.Tensor。

    保留 meta / __key__ 原样。
    """
    result: Dict[str, Any] = {}
    for key, value in sample.items():
        if key in ("meta", "__key__"):
            result[key] = value
            continue
        if isinstance(value, np.ndarray):
            if key in INT_KEYS:
                result[key] = torch.from_numpy(value).long()
            elif key in BOOL_KEYS:
                result[key] = torch.from_numpy(value).bool()
            else:
                result[key] = torch.from_numpy(value).float()
        else:
            result[key] = value
    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# collate
# ═══════════════════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[dict]) -> dict:
    """
    将变长 T 的 sample 列表 padding 到 batch 内最大 T。

    除 meta / __key__ / __url__ 外，所有张量沿 dim=0 (T) padding。

    Returns:
        batched dict，额外加入：
        - "T_mask": [B, max_T] bool — True=有效 tick, False=padding
        - "meta": list[dict] — 未合并
    """
    if not batch:
        return {}

    T_vals = [s["player_pos"].shape[0] for s in batch]
    max_T = max(T_vals)
    B = len(batch)

    result: Dict[str, Any] = {}
    result["T_mask"] = torch.zeros(B, max_T, dtype=torch.bool)
    result["meta"] = []

    # 收集所有 tensor key（跳过 meta/__key__）
    all_keys = sorted(set().union(*[
        {k for k in s if k not in ("meta", "__key__") and isinstance(s[k], torch.Tensor)}
        for s in batch
    ]))

    for key in all_keys:
        tensors = []
        for s in batch:
            t = s.get(key)
            if t is None:
                shapes = [s2[key].shape for s2 in batch if key in s2]
                if not shapes:
                    continue
                ref_shape = list(shapes[0])
                ref_shape[0] = T_vals[batch.index(s)]
                t = torch.zeros(ref_shape, dtype=torch.float32)
            tensors.append(t)

        if not tensors:
            continue

        # 决定 pad value
        if key in _PAD_TRUE_MASKS:
            pad_val = True
            dtype_override = torch.bool
        elif key in BOOL_KEYS:
            pad_val = False
            dtype_override = torch.bool
        elif key in INT_KEYS:
            pad_val = _INT_PAD_VALUES.get(key, 0)
            dtype_override = torch.long
        else:
            pad_val = 0.0
            dtype_override = None

        # Padding
        padded = []
        for i, t in enumerate(tensors):
            cur_T = t.shape[0]
            if cur_T < max_T:
                pad_len = max_T - cur_T
                pad_shape = list(t.shape)
                pad_shape[0] = pad_len

                if t.dtype == torch.bool or dtype_override == torch.bool:
                    pad_tensor = torch.full(pad_shape, pad_val, dtype=torch.bool)
                elif dtype_override == torch.long:
                    pad_tensor = torch.full(pad_shape, pad_val, dtype=torch.long)
                else:
                    pad_tensor = torch.zeros(pad_shape, dtype=t.dtype)

                t = torch.cat([t, pad_tensor], dim=0)

            if dtype_override is not None:
                t = t.to(dtype_override)

            padded.append(t)

        result[key] = torch.stack(padded, dim=0)  # [B, max_T, ...]

    # 构建 T_mask
    for i, T in enumerate(T_vals):
        result["T_mask"][i, :T] = True

    # 动态生成 token_pad_mask（True = pad token，由 T_mask 取反扩展得到）
    B_dim, max_T_dim = result["T_mask"].shape
    result["token_pad_mask"] = ~result["T_mask"].unsqueeze(-1).expand(B_dim, max_T_dim, 27)

    result["meta"] = [s.get("meta", {}) for s in batch]

    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# IterableDataset
# ═══════════════════════════════════════════════════════════════════════════════════

class CS2Dataset(torch.utils.data.IterableDataset):
    """
    CS2 训练数据 IterableDataset（基于 WebDataset）。

    Args:
        data_dir:       数据集根目录（含 train/ test/ 子目录）
        split:          "train" | "test" | "both"
        shuffle_buffer: WebDataset 内部 shuffle buffer 大小（0=不 shuffle）
        augment_depth:  是否将深度图从 [T,10,64] 扩展为 [T,10,64,5]，
                        加入每条射线的方向角度编码 cos/sin(yaw, pitch)。
                        推荐开启（默认 True），让模型感知深度图的空间结构。
        max_samples:    最大样本数（None=全部），用于调试

    Worker 切分:
        多 worker 时，每个 worker 处理不同的 shard 子集，避免重复读取。

    Example:
        ds = CS2Dataset("data/dataset", split="train", shuffle_buffer=5000)
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn,
                           num_workers=4, pin_memory=True)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        shuffle_buffer: int = 1000,
        augment_depth: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.augment_depth = augment_depth
        self.max_samples = max_samples

        self.urls = self._gather_urls()

    def _gather_urls(self) -> List[str]:
        """收集指定 split 下的所有 shard tar 文件。"""
        urls: List[str] = []
        for sp in (["train", "test"] if self.split == "both" else [self.split]):
            split_dir = self.data_dir / sp
            if not split_dir.is_dir():
                continue
            shards = sorted(split_dir.glob("shards-*.tar"))
            for shard in shards:
                urls.append(str(shard))
        if not urls:
            raise FileNotFoundError(
                f"No shards found in {self.data_dir}/[{self.split}]"
            )
        return urls

    def __iter__(self) -> Iterator[dict]:
        """迭代器，支持多 worker 自动切分。"""
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

        count = 0
        for sample in pipeline:
            # 深度图角度增强（在 numpy 域做，避免 torch 转换后再操作）
            if self.augment_depth:
                sample = augment_depth_with_angles(sample)

            yield sample_to_torch(sample)
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                break

    def __len__(self) -> int:
        """返回 shard 数量（非 sample 数，流式读取）。"""
        return len(self.urls)


# ═══════════════════════════════════════════════════════════════════════════════════
# Pretrain IterableDataset — 从 round 样本实时切窗口，不落盘
# ═══════════════════════════════════════════════════════════════════════════════════

class CS2PretrainDataset(torch.utils.data.IterableDataset):
    """
    预训练 IterableDataset：读取 round 级 WDS，在线切 64-tick 窗口。

    不做窗口级落盘——窗口在 __iter__ 中实时提取后立即 yield，
    避免 300GB+ 数据翻倍。

    Args:
        data_dir:       Round WDS 目录（含 train/ test/）
        split:          "train" | "test" | "both"
        n_ticks:        窗口 tick 数（默认 64）
        stride:         滑动步长（默认 16，75% 重叠）
        shuffle_buffer: 窗口级 shuffle buffer 大小（0=不 shuffle）
        augment_depth:  是否增强深度图
        max_samples:    最大窗口数（调试用）
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        n_ticks: int = 64,
        stride: int = 16,
        shuffle_buffer: int = 10000,
        augment_depth: bool = True,
        max_samples: Optional[int] = None,
        jitter: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.n_ticks = n_ticks
        self.stride = stride
        self.shuffle_buffer = shuffle_buffer
        self.augment_depth = augment_depth
        self.max_samples = max_samples
        self.jitter = jitter

        # 延迟导入，避免循环依赖
        from .pretrain_processor import PretrainWindowExtractor, shuffle_players
        self._shuffle_players = shuffle_players
        self._extractor = PretrainWindowExtractor(
            n_ticks=n_ticks,
            stride=stride,
            min_input_ticks=n_ticks,
            min_output_ticks=1,
            jitter=jitter,
        )

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

    def __iter__(self) -> Iterator[dict]:
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

        # 窗口 buffer：收集窗口后 shuffle，打散同一 round 内的相邻窗口
        window_buffer: List[dict] = []
        rng = np.random.RandomState()

        count = 0
        for sample in pipeline:
            if "label_camera" not in sample:
                continue

            if self.augment_depth:
                sample = augment_depth_with_angles(sample)

            # 提取窗口（numpy 域，纯切片，快）
            windows = self._extractor.extract_windows(sample)
            del sample  # 释放 round 的大块内存

            for w in windows:
                # 每个窗口独立 player shuffle
                w = self._shuffle_players(w)

                if self.shuffle_buffer > 0:
                    window_buffer.append(w)
                    if len(window_buffer) >= self.shuffle_buffer:
                        idx = rng.randint(len(window_buffer))
                        yield sample_to_torch(window_buffer.pop(idx))
                else:
                    yield sample_to_torch(w)

                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    return

        # 排空 buffer
        while window_buffer:
            idx = rng.randint(len(window_buffer))
            yield sample_to_torch(window_buffer.pop(idx))
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                return


def pretrain_collate_fn(batch: List[dict]) -> dict:
    """
    Pretrain 样本 collate：T 固定为 n_ticks，直接 stack 即可。
    """
    if not batch:
        return {}

    result: Dict[str, Any] = {}
    result["meta"] = []

    all_keys = sorted(set().union(*[
        {k for k in s if k not in ("meta", "__key__") and isinstance(s[k], torch.Tensor)}
        for s in batch
    ]))

    for key in all_keys:
        tensors = [s[key] for s in batch]
        result[key] = torch.stack(tensors, dim=0)

    result["meta"] = [s.get("meta", {}) for s in batch]
    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════════════

def create_dataloader(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = 8,
    shuffle_buffer: int = 5000,
    augment_depth: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    max_samples: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    一键创建 DataLoader。

    Args:
        data_dir:         数据集根目录
        split:            "train" | "test" | "both"
        batch_size:       batch 大小
        shuffle_buffer:   webdataset shuffle buffer
        augment_depth:    是否增强深度图（加入射线方向角度编码）
        num_workers:      DataLoader worker 数（0=主进程）
        pin_memory:       是否 pin memory（GPU 训练时推荐 True）
        prefetch_factor:  每个 worker 预取 batch 数
        max_samples:      最大样本数限制（调试用）

    Returns:
        配置好的 DataLoader
    """
    ds = CS2Dataset(
        data_dir=data_dir,
        split=split,
        shuffle_buffer=shuffle_buffer,
        augment_depth=augment_depth,
        max_samples=max_samples,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def create_pretrain_dataloader(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = 8,
    n_ticks: int = 64,
    stride: int = 16,
    shuffle_buffer: int = 10000,
    augment_depth: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    max_samples: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    一键创建预训练 DataLoader（从 round WDS 实时切窗口，不落盘）。

    Args:
        data_dir:         Round WDS 目录（含 train/ test/）
        split:            "train" | "test" | "both"
        batch_size:       batch 大小
        n_ticks:          窗口 tick 数（默认 64）
        stride:           滑动步长（默认 16）
        shuffle_buffer:   窗口级 shuffle buffer（默认 10000）
        augment_depth:    是否增强深度图
        num_workers:      DataLoader worker 数
        pin_memory:       是否 pin memory
        prefetch_factor:  预取 batch 数
        max_samples:      最大窗口数（调试用）

    Returns:
        配置好的 DataLoader，每个 batch 包含 64-tick 预训练窗口。
    """
    ds = CS2PretrainDataset(
        data_dir=data_dir,
        split=split,
        n_ticks=n_ticks,
        stride=stride,
        shuffle_buffer=shuffle_buffer,
        augment_depth=augment_depth,
        max_samples=max_samples,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=pretrain_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# 调试/检查入口
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CS2Dataset loading")
    parser.add_argument("--data-dir", required=True, help="Path to dataset dir")
    parser.add_argument("--split", default="train", choices=["train", "test", "both"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--no-augment-depth", action="store_true",
                        help="Disable depth angular augmentation")
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    print(f"Loading from: {args.data_dir}/{args.split}")
    print(f"Batch size: {args.batch_size}, workers: {args.num_workers}")
    print(f"Depth augment: {not args.no_augment_depth}")

    loader = create_dataloader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle_buffer=0 if args.no_shuffle else 1000,
        augment_depth=not args.no_augment_depth,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    for i, batch in enumerate(loader):
        print(f"\n{'='*60}")
        print(f"Batch {i}:")
        print(f"  T_mask shape: {batch['T_mask'].shape}  "
              f"(max_T={batch['T_mask'].sum(dim=1).int().tolist()})")

        for key in sorted(batch.keys()):
            val = batch[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key:25s}  {str(list(val.shape)):30s}  {str(val.dtype):15s}")
            elif key == "meta":
                metas = val
                print(f"  meta: {len(metas)} samples")
                for m in metas[:2]:
                    print(f"    map={m.get('map_name','?')}  T={m.get('T','?')}  "
                          f"winner={m.get('winner','?')}  round={m.get('round_id','?')}")

        if i >= 2:
            break

    print("\n✓ Data loading test passed!")
