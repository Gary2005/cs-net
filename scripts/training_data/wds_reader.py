"""
WebDataset 读取器 — 解码、验证、统计。

用于 visualize_wds.py 的底层工具。
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import webdataset as wds
import zstandard as zstd

_dctx = zstd.ZstdDecompressor()

# 所有已知的 numpy key（与 wds_writer.py 保持同步）
NUMPY_KEYS = {
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

# 已知的 label key
LABEL_KEYS = {
    "label_winrate",
    "label_nxt_kill", "label_nxt_death",
    "label_alive_end",
    "label_bombsite", "label_win_reason",
    "label_camera",
}

# state 各维度的名称
STATE_DIM_NAMES = [
    "hp/100", "armor/100", "helmet", "defuser",
    "flash_dur/5", "flash_alpha/255",
    "cos(pitch)", "sin(pitch)", "cos(yaw)", "sin(yaw)",
    "is_CT",
    "log(v_forward)", "log(v_right)", "log(v_vert)",
]

# relation 各维度的名称
REL_DIM_NAMES = [
    "log(d_forward)", "log(d_right)", "log(d_up)", "log_dist",
    "is_teammate", "is_enemy",
    "spotted_by_me", "spotted_me",
    "cos(d_theta_xy)", "sin(d_theta_xy)",
    "cos(d_theta_z)", "sin(d_theta_z)",
    "j_alive", "j_hp/100",
]

# bomb_state 各维度名称
BOMB_STATE_DIM_NAMES = [
    "round_time/160", "is_planted", "is_dropped", "planted_dur/40",
]

# sound 各维度名称
SOUND_DIM_NAMES = ["is_firing", "has_footsteps"]

# 每个数据 key 对应的 mask key，用于统计时过滤无效条目
_MASK_FOR_KEY = {
    "player_pos": "player_alive_mask",
    "player_state": "player_alive_mask",
    "player_inv": "player_inv_mask",
    "player_rel_f": "player_rel_mask",
    "player_rel_i": "player_rel_mask",
    "player_sound": "player_alive_mask",
    "player_depth": "player_depth_mask",
    "proj_pos": "proj_mask",
    "proj_type": "proj_mask",
    "proj_dur": "proj_mask",
}

# 投掷物类型名称
PROJ_TYPE_NAMES = {
    0: "smoke", 1: "inferno", 2: "he",
    3: "flashbang", 4: "decoy", 5: "molotov",
}


def decode_sample(raw: dict) -> dict:
    """
    解码一个 WebDataset sample。

    - .npy.zst → numpy array
    - .json.zst → dict (keyed as "meta")
    """
    result = {}
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


def scan_shards(data_dir: Path) -> Dict[str, List[Path]]:
    """
    扫描 data_dir 下的 train/ test/ 子目录，返回 shard 文件列表。

    Returns:
        {"train": [Path, ...], "test": [Path, ...]}
    """
    result = {"train": [], "test": []}
    for split in ("train", "test"):
        split_dir = data_dir / split
        if split_dir.is_dir():
            shards = sorted(split_dir.glob("shards-*.tar"))
            result[split] = shards
    return result


def read_shard_keys(shard_path: Path) -> List[str]:
    """读取一个 shard 中所有 sample 的 key（不解码内容）。"""
    # 跳过空 shard
    if shard_path.stat().st_size < 1024:
        return []
    dataset = wds.WebDataset([str(shard_path)], shardshuffle=False, empty_check=False)
    keys = []
    try:
        for sample in dataset:
            key = sample.get("__key__", "")
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="replace")
            keys.append(key)
    except ValueError:
        pass  # empty shard
    return keys


def read_shard_samples(
    shard_path: Path,
    max_samples: Optional[int] = None,
) -> Iterator[dict]:
    """逐 sample 解码读取一个 shard。"""
    if shard_path.stat().st_size < 1024:
        return
    dataset = wds.WebDataset([str(shard_path)], shardshuffle=False, empty_check=False)
    count = 0
    try:
        for raw in dataset:
            if max_samples is not None and count >= max_samples:
                break
            yield decode_sample(raw)
            count += 1
    except ValueError:
        pass  # empty shard


def validate_sample(sample: dict) -> List[dict]:
    """
    校验单个 sample，返回错误/警告列表。

    每个错误 dict: {"severity": "error"|"warning", "field": str, "message": str}
    """
    issues = []
    meta = sample.get("meta", {})

    # ── 检查 meta ──
    if not meta:
        issues.append({"severity": "error", "field": "meta", "message": "Missing metadata"})
        return issues

    expected_T = meta.get("T", 0)
    if expected_T <= 0:
        issues.append({"severity": "error", "field": "meta.T", "message": f"Invalid T={expected_T}"})

    # ── 逐个 numpy 字段检查 ──
    for key in sorted(NUMPY_KEYS):
        if key not in sample:
            issues.append({"severity": "warning", "field": key, "message": "Field missing from sample"})
            continue

        arr = sample[key]

        # NaN / Inf 检查
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        if nan_count > 0:
            issues.append({
                "severity": "error", "field": key,
                "message": f"Contains {nan_count} NaN values ({nan_count/arr.size*100:.2f}%)",
            })
        if inf_count > 0:
            issues.append({
                "severity": "error", "field": key,
                "message": f"Contains {inf_count} Inf values ({inf_count/arr.size*100:.2f}%)",
            })

        # nxt_kill / nxt_death: 检查值在 0-10 范围内（10=无更多击杀/死亡）
        if key in ("label_nxt_kill", "label_nxt_death"):
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                out_range = np.sum((valid < 0) | (valid > 10))
                if out_range > 0:
                    issues.append({
                        "severity": "error", "field": key,
                        "message": f"Values out of [0,10] range: {out_range}",
                    })

        # 位置字段：检查范围 [-1, 1]
        if key in ("player_pos",):
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                out_low = np.sum(valid < -1.01)
                out_high = np.sum(valid > 1.01)
                if out_low > 0 or out_high > 0:
                    issues.append({
                        "severity": "warning", "field": key,
                        "message": f"Out of [-1,1] range: {out_low} below, {out_high} above",
                    })

        # 深度图：检查是否在 [0, 1]
        if key == "player_depth":
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                neg = np.sum(valid < -0.01)
                above = np.sum(valid > 1.01)
                zeros = np.sum(np.abs(valid) < 1e-6)
                if neg > 0:
                    issues.append({
                        "severity": "error", "field": key,
                        "message": f"Negative depth values: {neg}",
                    })
                if zeros > len(valid) * 0.5:
                    all_zero = zeros == len(valid)
                    issues.append({
                        "severity": "info" if all_zero else "warning", "field": key,
                        "message": f"{'All' if all_zero else 'High ratio of'} zero depth: {zeros/len(valid)*100:.1f}%"
                        + (" (--no-depth used?)" if all_zero else ""),
                    })

    # ── 一致性检查 ──
    # cos/sin 一致性检查 (player_state dim 6-7, 8-9)
    if "player_state" in sample and "player_alive_mask" in sample:
        state = sample["player_state"]
        smask = sample["player_alive_mask"]
        if smask.any():
            valid_state = state[smask]
            if valid_state.shape[-1] >= 10:
                # pitch: dim 6,7
                cp = valid_state[:, 6]
                sp = valid_state[:, 7]
                pitch_mag = cp**2 + sp**2
                bad_pitch = np.sum(np.abs(pitch_mag - 1.0) > 0.1)
                if bad_pitch > 0:
                    issues.append({
                        "severity": "warning", "field": "player_state[6:8]",
                        "message": f"cos²+sin²(pitch) ≠ 1 for {bad_pitch} values",
                    })
                # yaw: dim 8,9
                cy = valid_state[:, 8]
                sy = valid_state[:, 9]
                yaw_mag = cy**2 + sy**2
                bad_yaw = np.sum(np.abs(yaw_mag - 1.0) > 0.1)
                if bad_yaw > 0:
                    issues.append({
                        "severity": "warning", "field": "player_state[8:10]",
                        "message": f"cos²+sin²(yaw) ≠ 1 for {bad_yaw} values",
                    })

    # label_nxt_kill/death: 检查 inf 值
    for lk in ("label_nxt_kill", "label_nxt_death"):
        if lk in sample:
            arr = sample[lk]
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                neg = np.sum(finite < 0)
                if neg > 0:
                    issues.append({
                        "severity": "error", "field": lk,
                        "message": f"Negative tick values: {neg}",
                    })

    return issues


# ── 统计工具 ──────────────────────────────────────────────────────────────────


def _histogram_ascii(values: np.ndarray, n_bins: int = 10) -> List[str]:
    """生成 ASCII 柱状图字符串列表。"""
    if len(values) == 0:
        return ["  (no data)"]

    hist, bin_edges = np.histogram(values, bins=n_bins)
    max_bar = 40
    total = len(values)
    lines = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        count = hist[i]
        pct = count / total * 100
        bar_len = int(count / max(hist.max(), 1) * max_bar)
        bar = "█" * bar_len + "░" * (max_bar - bar_len) if bar_len > 0 else "░" * max_bar
        lines.append(f"    [{lo:.3f}, {hi:.3f}): {bar} {pct:.1f}%")
    return lines


def _print_dim_stats(name: str, values: np.ndarray, dim_name: str = ""):
    """打印单个维度的统计信息。"""
    label = f"{name} {dim_name}" if dim_name else name
    count = len(values)
    if count == 0:
        print(f"  {label}: (no data)")
        return

    mean_v = float(np.mean(values))
    median_v = float(np.median(values))
    std_v = float(np.std(values))
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    nan_c = int(np.sum(np.isnan(values)))
    inf_c = int(np.sum(np.isinf(values)))
    zero_pct = float(np.sum(np.abs(values) < 1e-6) / count * 100)

    print(f"  {label}:")
    print(f"    count: {count:,}  mean: {mean_v:.4f}  median: {median_v:.4f}  std: {std_v:.4f}")
    print(f"    min: {min_v:.4f}  max: {max_v:.4f}  NaN: {nan_c}  Inf: {inf_c}  zero%: {zero_pct:.1f}%")

    # 柱状图
    finite_vals = values[np.isfinite(values)]
    if len(finite_vals) > 1:
        lines = _histogram_ascii(finite_vals)
        for line in lines:
            print(line)


def compute_and_print_stats(
    samples_iter: Iterator[dict],
    max_samples: Optional[int] = None,
    show_histograms: bool = True,
) -> dict:
    """
    遍历 samples 并打印综合统计信息。

    Returns:
        统计摘要 dict
    """
    # ── 聚合 buffer ──
    # 对于大数组字段，不存储全部数据，而是增量聚合
    # 对于小字段，收集所有值
    accum: Dict[str, List[np.ndarray]] = {}

    map_counts: Dict[str, int] = {}
    T_values: List[int] = []
    end_reasons: Dict[str, int] = {}
    winners: Dict[str, int] = {}
    total_samples = 0
    total_errors = 0
    total_warnings = 0

    # 采样预算
    sample_count = 0
    for sample in samples_iter:
        if max_samples is not None and sample_count >= max_samples:
            break
        sample_count += 1
        total_samples += 1

        meta = sample.get("meta", {})

        # Map stats
        map_name = meta.get("map_name", "unknown")
        map_counts[map_name] = map_counts.get(map_name, 0) + 1

        # T stats
        T = meta.get("T", 0)
        if T > 0:
            T_values.append(T)

        # End reason / winner
        end_reason = meta.get("end_reason", "unknown")
        end_reasons[end_reason] = end_reasons.get(end_reason, 0) + 1
        winner = meta.get("winner", "unknown")
        winners[winner] = winners.get(winner, 0) + 1

        # ── 收集各字段的值用于统计 ──
        for key in NUMPY_KEYS:
            if key not in sample:
                continue
            arr = sample[key].astype(np.float64)

            # 对于多维特征保留最后一维（用于逐维度统计），其余 flatten
            preserve_last_dim = arr.ndim >= 2 and key in (
                "player_pos", "player_state", "player_rel_f",
                "proj_pos", "player_sound", "player_depth",
                "bomb_pos", "bomb_state",
                "player_inv", "proj_type",
            )
            if preserve_last_dim:
                flat = arr.reshape(-1, arr.shape[-1])
            else:
                flat = arr.ravel()

            # 用对应的 mask 过滤无效条目（dead 玩家、不存在的投射物等）
            mask_key = _MASK_FOR_KEY.get(key)
            if mask_key and mask_key in sample:
                mask_flat = sample[mask_key].ravel()
                if len(mask_flat) == len(flat):
                    flat = flat[mask_flat]

            if len(flat) > 0:
                if key not in accum:
                    accum[key] = []
                # 限制每个样本收集的数量以避免 OOM
                max_per_sample = 1_000_000
                if len(flat) > max_per_sample:
                    indices = np.random.choice(len(flat), max_per_sample, replace=False)
                    flat = flat[indices]
                accum[key].append(flat)

        # 校验
        issues = validate_sample(sample)
        for iss in issues:
            if iss["severity"] == "error":
                total_errors += 1
            else:
                total_warnings += 1

    # ── 打印 Layer 1: Dataset Overview ──
    print("\n" + "=" * 72)
    print("  Layer 1: Dataset Overview")
    print("=" * 72)
    print(f"\nTotal samples analyzed: {total_samples}")
    print(f"Validation: {total_errors} errors, {total_warnings} warnings")

    # Map distribution
    if map_counts:
        print("\nMap Distribution:")
        max_bar = 40
        max_count = max(map_counts.values())
        for name in sorted(map_counts.keys(), key=lambda k: -map_counts[k]):
            count = map_counts[name]
            pct = count / total_samples * 100
            bar_len = int(count / max_count * max_bar)
            bar = "█" * bar_len
            print(f"  {name:15s} {bar} {count:,} ({pct:.1f}%)")

    # T distribution
    if T_values:
        T_arr = np.array(T_values)
        print(f"\nT (num ticks) Distribution:")
        print(f"  Min: {T_arr.min():.0f}  Max: {T_arr.max():.0f}  "
              f"Mean: {T_arr.mean():.1f}  Median: {np.median(T_arr):.0f}  Std: {T_arr.std():.1f}")
        if show_histograms and len(T_arr) > 1:
            lines = _histogram_ascii(T_arr, 10)
            for line in lines:
                print(line)

    # End reason distribution
    if end_reasons:
        print("\nEnd Reason Distribution:")
        for reason in sorted(end_reasons.keys(), key=lambda k: -end_reasons[k]):
            count = end_reasons[reason]
            pct = count / total_samples * 100
            print(f"  {reason:20s}: {count:,} ({pct:.1f}%)")

    # Winner distribution
    if winners:
        print("\nWinner Distribution:")
        for w in sorted(winners.keys(), key=lambda k: -winners[k]):
            count = winners[w]
            pct = count / total_samples * 100
            print(f"  {w:5s}: {count:,} ({pct:.1f}%)")

    # ── 打印 Layer 2: Per-Dimension Feature Statistics ──
    print("\n" + "=" * 72)
    print("  Layer 2: Input Feature Distributions")
    print("=" * 72)

    # player_pos — 3 维
    _print_feature_dims(accum, "player_pos", ["x", "y", "z"], show_histograms)

    # player_state — 14 维
    _print_feature_dims(accum, "player_state", STATE_DIM_NAMES, show_histograms)
    # circular encoding check
    if "player_state" in accum:
        all_vals = np.concatenate(accum["player_state"])
        if all_vals.ndim >= 2 and all_vals.shape[-1] >= 10:
            cp, sp = all_vals[:, 6], all_vals[:, 7]
            mag_p = cp**2 + sp**2
            bad_p = np.sum(np.abs(mag_p - 1.0) > 0.1)
            print(f"  ⚠ cos²+sin²(pitch) check: {bad_p}/{len(mag_p)} values deviate >0.1 from 1.0")
            cy, sy = all_vals[:, 8], all_vals[:, 9]
            mag_y = cy**2 + sy**2
            bad_y = np.sum(np.abs(mag_y - 1.0) > 0.1)
            print(f"  ⚠ cos²+sin²(yaw) check:   {bad_y}/{len(mag_y)} values deviate >0.1 from 1.0")

    # player_inv
    _print_feature_dims(accum, "player_inv", ["weapon_idx"], show_histograms)

    # player_rel_f — 14 维
    _print_feature_dims(accum, "player_rel_f", REL_DIM_NAMES, show_histograms)

    # player_sound — 2 维
    _print_feature_dims(accum, "player_sound", SOUND_DIM_NAMES, show_histograms)

    # player_depth — 64 维（按层汇总）
    if "player_depth" in accum:
        all_depth = np.concatenate(accum["player_depth"])
        print(f"\n  player_depth (overall):")
        _print_dim_stats_single("all 64 dirs", all_depth.ravel())
        if show_histograms and all_depth.size > 1:
            for line in _histogram_ascii(all_depth.ravel(), 10):
                print(line)
        # 按层
        if all_depth.shape[-1] == 64:
            layers = {
                "+60° (dims 0-7)":   slice(0, 8),
                "+30° (dims 8-19)":  slice(8, 20),
                "0° (dims 20-43)":   slice(20, 44),
                "-30° (dims 44-55)": slice(44, 56),
                "-60° (dims 56-63)": slice(56, 64),
            }
            for layer_name, sl in layers.items():
                layer_vals = all_depth[:, sl].ravel()
                if len(layer_vals) > 0:
                    print(f"    {layer_name}: mean={np.mean(layer_vals):.4f}, "
                          f"median={np.median(layer_vals):.4f}, "
                          f"zero%={np.sum(np.abs(layer_vals)<1e-6)/len(layer_vals)*100:.1f}%")

    # bomb_pos
    _print_feature_dims(accum, "bomb_pos", ["x", "y", "z"], show_histograms)
    # bomb_state
    _print_feature_dims(accum, "bomb_state", BOMB_STATE_DIM_NAMES, show_histograms)

    # map_idx
    _print_feature_dims(accum, "map_idx", ["map_idx"], show_histograms)

    # proj_pos
    _print_feature_dims(accum, "proj_pos", ["x", "y", "z"], show_histograms)
    # proj_type
    if "proj_type" in accum:
        all_types = np.concatenate(accum["proj_type"])
        print(f"\n  proj_type distribution:")
        valid = all_types[all_types >= 0]
        total_valid = len(valid)
        for type_idx in sorted(PROJ_TYPE_NAMES.keys()):
            count = np.sum(valid == type_idx)
            pct = count / max(total_valid, 1) * 100
            print(f"    {PROJ_TYPE_NAMES[type_idx]:12s} ({type_idx}): {count:,} ({pct:.1f}%)")
    # proj_dur
    _print_feature_dims(accum, "proj_dur", ["dur/25"], show_histograms)

    # mask 覆盖率
    print(f"\n  Mask Coverage (% True):")
    for mask_key in ["player_alive_mask", "player_inv_mask",
                      "player_rel_mask", "player_depth_mask",
                      "proj_mask",
                      "player_alive_mask"]:
        if mask_key in accum:
            all_vals = np.concatenate(accum[mask_key])
            pct_true = np.sum(all_vals > 0.5) / len(all_vals) * 100
            print(f"    {mask_key:25s}: {pct_true:.1f}% True")

    # ── 打印 Layer 3: Label Distributions ──
    print("\n" + "=" * 72)
    print("  Layer 3: Label Distributions")
    print("=" * 72)

    _print_label_stats(accum, "label_winrate", "winrate", show_histograms)
    _print_label_stats(accum, "label_nxt_kill", "ticks to next kill", show_histograms)
    _print_label_stats(accum, "label_nxt_death", "ticks to next death", show_histograms)
    _print_label_stats(accum, "label_alive_end", "alive at end", show_histograms)

    # ── 返回值 ──
    return {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "map_counts": map_counts,
        "T_stats": {
            "min": int(np.min(T_values)) if T_values else 0,
            "max": int(np.max(T_values)) if T_values else 0,
            "mean": float(np.mean(T_values)) if T_values else 0,
            "median": float(np.median(T_values)) if T_values else 0,
        },
    }


def _print_feature_dims(
    accum: dict,
    key: str,
    dim_names: List[str],
    show_histograms: bool,
):
    """打印一个多维特征字段的各维度统计。"""
    if key not in accum:
        print(f"\n  {key}: (no data)")
        return

    all_vals = np.concatenate(accum[key])
    if all_vals.size == 0:
        print(f"\n  {key}: (no valid data)")
        return

    print(f"\n  {key} [{all_vals.shape[-1]} dims]:")

    if all_vals.ndim == 1:
        _print_dim_stats_single(key, all_vals)
        if show_histograms and len(all_vals) > 1:
            for line in _histogram_ascii(all_vals, 10):
                print(line)
        return

    # 多维：逐维度打印
    ndims = all_vals.shape[-1]
    for d in range(ndims):
        dim_name = dim_names[d] if d < len(dim_names) else f"dim{d}"
        dim_vals = all_vals[..., d].ravel()
        _print_dim_stats_single(f"dim {d} ({dim_name})", dim_vals)
    if show_histograms and ndims <= 5:
        for d in range(ndims):
            dim_name = dim_names[d] if d < len(dim_names) else f"dim{d}"
            dim_vals = all_vals[..., d].ravel()
            print(f"    dim {d} ({dim_name}) histogram:")
            for line in _histogram_ascii(dim_vals, 10):
                print(line)


def _print_dim_stats_single(label: str, values: np.ndarray):
    """打印单个维度的统计。"""
    if len(values) == 0:
        print(f"    {label}: (no data)")
        return
    mean_v = float(np.mean(values))
    median_v = float(np.median(values))
    std_v = float(np.std(values))
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    nan_c = int(np.sum(np.isnan(values)))
    inf_c = int(np.sum(np.isinf(values)))
    zero_pct = float(np.sum(np.abs(values) < 1e-6) / len(values) * 100)

    print(f"    {label}:")
    print(f"      count={len(values):,}  mean={mean_v:.4f}  median={median_v:.4f}  "
          f"std={std_v:.4f}  min={min_v:.4f}  max={max_v:.4f}  NaN={nan_c}  Inf={inf_c}  zero%={zero_pct:.1f}%")


def _print_label_stats(
    accum: dict,
    key: str,
    desc: str,
    show_histograms: bool,
):
    """打印一个 label 字段的统计。"""
    if key not in accum:
        print(f"\n  {key} ({desc}): (no data)")
        return

    all_vals = np.concatenate(accum[key])
    if all_vals.size == 0:
        print(f"\n  {key} ({desc}): (no data)")
        return

    if all_vals.ndim == 1:
        print(f"\n  {key} ({desc}) [{len(all_vals)} values]:")
    else:
        print(f"\n  {key} ({desc}) [{all_vals.shape[-1]} dims]:")

    if all_vals.ndim == 1:
        _print_dim_stats_single("overall", all_vals)
        if key in ("label_nxt_kill", "label_nxt_death"):
            # 分类标签：0-9=目标玩家，10=无
            no_event_pct = np.sum(all_vals > 9.5) / all_vals.size * 100
            print(f"      No more events (label=10): {no_event_pct:.1f}%")
            valid = all_vals[all_vals <= 9.5]
            if len(valid) > 0:
                unique, counts = np.unique(valid.astype(int), return_counts=True)
                target_str = "  ".join(f"P{i}:{c/len(valid)*100:.1f}%" for i, c in zip(unique, counts))
                print(f"      Target distribution: {target_str}")
        elif show_histograms and len(all_vals) > 1:
            finite = all_vals[np.isfinite(all_vals)]
            if len(finite) > 1:
                for line in _histogram_ascii(finite, 10):
                    print(line)
        return

    ndims = all_vals.shape[-1]
    # 汇总
    overall = all_vals.ravel()
    _print_dim_stats_single(f"overall (all players)", overall)

    # Special handling per label type
    if key in ("label_nxt_kill", "label_nxt_death"):
        # 分类标签：值 0-9 = 目标玩家，10 = 无更多事件
        no_event_pct = np.sum(all_vals > 9.5) / all_vals.size * 100
        print(f"      No more events (label=10): {no_event_pct:.1f}%")
        valid = all_vals[all_vals <= 9.5]
        if len(valid) > 0:
            print(f"      Has next event: {100-no_event_pct:.1f}% ({len(valid):,} values)")
            unique, counts = np.unique(valid.astype(int), return_counts=True)
            target_str = "  ".join(f"P{i}:{c/len(valid)*100:.1f}%" for i, c in zip(unique, counts))
            print(f"      Target distribution: {target_str}")
        return  # skip generic handling

    elif key in ("label_winrate", "label_alive_end"):
        # Binary/semi-binary: show per-player breakdown
        for d in range(ndims):
            d_vals = all_vals[..., d].ravel()
            _print_dim_stats_single(f"dim {d}", d_vals)

    if show_histograms and ndims <= 5 and key not in ("label_nxt_kill", "label_nxt_death"):
        overall_finite = overall[np.isfinite(overall)]
        if len(overall_finite) > 1:
            print(f"      Histogram (overall):")
            for line in _histogram_ascii(overall_finite, 10):
                print(line)
