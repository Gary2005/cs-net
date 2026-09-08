#!/usr/bin/env python3
"""
验证 cs-net-v4 模型 checkpoint 能正确加载（下载模型后运行）。

用法:
    python scripts/test_checkpoints.py --models-dir checkpoints
    python scripts/test_checkpoints.py --models-dir checkpoints --device cpu
    python scripts/test_checkpoints.py --models-dir checkpoints --forward
        # --forward: 额外跑 ① 合成局面的 16 tick 自回归路径预测
        #            ② 内置回合数据 examples/json/test.json.gz 的完整 pipeline
        #            路径预测（CPU 上约 1-3 分钟，默认关闭）

检查内容:
  [1] 路径预测 ckpt (cs-net-v4-pro.pt):
      - checkpoint 结构 / global_step / 参数量（Pro ≈ 138.7M）
      - 与 config/pretrain-a100-pro.yaml 架构完全匹配（无 missing / unexpected 键）
      - 所有权重为有限值
      - PredictionEngine 完整加载路径可用
      （可选 --forward）合成局面跑一次 16 tick 自回归路径预测
  [2] spatial-only ckpt（winrate / alive_end / future_kill）:
      - SpatialOnlyPredictor 自动发现并加载 3 个任务模型（d_model=768）
      - 合成回合（16 tick）逐 tick 推理，输出全部为有限值
  [3] 内置回合数据完整 pipeline（examples/json/test.json.gz，de_mirage）:
      - json.gz → filter_data → _convert_inventory_indices → process_round
        （与训练/可视化工具完全相同的预处理链路）
      - spatial-only 对真实回合逐 tick 推理，输出全部为有限值
      （可选 --forward）对真实回合跑一次自回归路径预测
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pretrain_model import CS2PretrainModel, PretrainConfig  # noqa: E402
from prediction_engine import PredictionEngine  # noqa: E402
from spatial_only_predictor import SpatialOnlyPredictor  # noqa: E402
from training_data.config import MAP_NAME_TO_IDX  # noqa: E402

PRETRAIN_FILE = "cs-net-v4-pro.pt"
SPATIAL_TASKS = ("winrate", "alive_end", "future_kill")
PRO_PARAMS_M = 138.7  # Pro 架构实测参数量（d_model=768）
PIPELINE_DATA = _PROJECT_ROOT / "examples" / "json" / "test.json.gz"
PIPELINE_MAX_TICKS = 64  # pipeline 测试只处理真实回合前 64 个 tick（保持测试快速）


def _cfg_from_yaml(path: str) -> PretrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return PretrainConfig(
        d_model=int(cfg.get("d_model", 768)),
        n_spatial_layers=int(cfg.get("n_spatial_layers", 5)),
        n_temporal_layers=int(cfg.get("n_temporal_layers", 3)),
        n_decoder_layers=int(cfg.get("n_decoder_layers", 6)),
        n_heads=int(cfg.get("n_heads", 12)),
        d_ff=int(cfg.get("d_ff", 3072)),
        dropout=float(cfg.get("dropout", 0.1)),
        n_depth_ray_layers=int(cfg.get("n_depth_ray_layers", 3)),
        n_ticks=int(cfg.get("n_ticks", 16)),
        move_range=float(cfg.get("move_range", 128.0)),
        move_grid_size=float(cfg.get("move_grid_size", 4.0)),
        angle_grid_size=float(cfg.get("angle_grid_size", 5.0)),
        use_residual_correction=bool(cfg.get("use_residual_correction", True)),
    )


def _make_synthetic_sample(T: int = 16, map_name: str = "de_cache") -> dict:
    """构造形状与真实 round sample 完全一致的合成样本（无需数据即可跑通推理）。"""
    rng = np.random.default_rng(0)
    map_idx = MAP_NAME_TO_IDX[map_name]
    sample = {
        "player_pos": rng.uniform(-1500, 1500, (T, 10, 3)).astype(np.float32),
        "player_state": rng.uniform(-1, 1, (T, 10, 14)).astype(np.float32),
        "player_inv": rng.integers(0, 30, (T, 10, 9)).astype(np.int32),
        "player_inv_mask": rng.random((T, 10, 9)) > 0.3,
        "player_rel_f": rng.uniform(-1, 1, (T, 10, 9, 14)).astype(np.float32),
        "player_rel_i": rng.integers(0, 60, (T, 10, 9)).astype(np.int32),
        "player_rel_mask": rng.random((T, 10, 9)) > 0.3,
        "player_sound": rng.uniform(-1, 1, (T, 10, 2)).astype(np.float32),
        "player_depth": rng.uniform(0, 1, (T, 10, 64)).astype(np.float32),
        "player_depth_mask": rng.random((T, 10)) > 0.1,
        "player_alive_mask": np.ones((T, 10), dtype=bool),
        "bomb_pos": rng.uniform(-1500, 1500, (T, 3)).astype(np.float32),
        "bomb_state": rng.uniform(0, 1, (T, 4)).astype(np.float32),
        "map_idx": np.full((T,), map_idx, dtype=np.int32),
        "proj_pos": rng.uniform(-1500, 1500, (T, 16, 3)).astype(np.float32),
        "proj_type": np.zeros((T, 16), dtype=np.int32),
        "proj_dur": np.zeros((T, 16), dtype=np.float32),
        "proj_mask": np.zeros((T, 16), dtype=bool),
        "proj_is_active": np.zeros((T, 16), dtype=np.float32),
        # labels
        "label_camera": rng.uniform(-0.2, 0.2, (T, 10, 10)).astype(np.float32),
        "label_alive_end": rng.random((T, 10)).astype(np.float32),
        "label_nxt_kill": np.full((T,), 10, dtype=np.int32),
        "label_nxt_death": np.full((T,), 10, dtype=np.int32),
        "label_winrate": rng.random(T).astype(np.float32),
        "round_seconds": (np.arange(T, dtype=np.float32) * 0.25),
        "meta": {
            "map_name": map_name,
            "T": T,
            "tick_interval": 0.25,
            "format": "cs2.training.v5",
            "teams": ["CT"] * 5 + ["T"] * 5,
            "winner": "T",
        },
    }
    return sample


def _slice_round(sample: dict, n_ticks: int) -> dict:
    """把 round sample 截断到前 n_ticks 个 tick（所有 [T, ...] 数组沿 dim0 切片）。"""
    out = {}
    for k, v in sample.items():
        if k == "meta":
            meta = dict(v)
            meta["T"] = int(n_ticks)
            out[k] = meta
        elif isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == sample["meta"]["T"]:
            out[k] = v[:n_ticks]
        else:
            out[k] = v
    return out


def check_pretrain(config_path: str, ckpt_path: Path, device: str,
                   maps_dir: str, run_forward: bool) -> PredictionEngine:
    print(f"\n[1/3] 路径预测 checkpoint: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model" in ckpt, "checkpoint 缺少 'model' 键（不是预训练模型格式）"
    step = ckpt.get("global_step", "?")
    print(f"  global_step = {step}")

    # 剥离 torch.compile 的 _orig_mod. 前缀，与 PredictionEngine 一致
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    del ckpt

    cfg = _cfg_from_yaml(config_path)
    model = CS2PretrainModel(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)

    assert not unexpected, f"Unexpected keys: {unexpected[:5]}"
    # 完整预训练 ckpt 应包含全部权重；spatial-only 才有 missing（但这里不是）
    assert not missing, f"Missing keys: {missing[:5]}"
    print(f"  架构匹配: d_model={cfg.d_model}, 无 missing / unexpected 键")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    assert abs(n_params - PRO_PARAMS_M) < 1.0, \
        f"参数量 {n_params:.1f}M 与 Pro 架构 ({PRO_PARAMS_M}M) 不符"
    print(f"  参数量: {n_params:.1f}M (Pro 架构预期 ≈{PRO_PARAMS_M}M)")

    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    assert finite, "存在非有限权重（NaN/Inf）"
    print("  所有权重有限值 ✓")
    del model, state

    # PredictionEngine 完整加载路径（config + ckpt → 推理引擎）
    engine = PredictionEngine(config_path, str(ckpt_path), device=device,
                              maps_dir=maps_dir)
    print(f"  PredictionEngine 加载成功 (device={device})")

    if run_forward:
        print("  --forward: 合成局面 16 tick 自回归路径预测（CPU 约 1-3 分钟）...")
        sample = _make_synthetic_sample(T=16)
        result = engine.predict_at_tick(sample, query_tick=0)
        n_alive = sum(1 for t in result["trajectories"] if t["is_alive"])
        n_pred = sum(1 for t in result["trajectories"]
                     if t["is_alive"] and t["pred_steps"] > 0)
        assert n_pred == n_alive, \
            f"预测轨迹数 ({n_pred}) != 存活玩家数 ({n_alive})"
        print(f"  AR 推理完成: {n_alive} 名存活玩家全部产出 {result['output_T']} tick 预测轨迹 ✓")

    print("  ✓ 路径预测 checkpoint 加载测试通过")
    return engine


def check_spatial(models_dir: Path, device: str) -> SpatialOnlyPredictor:
    print(f"\n[2/3] spatial-only checkpoints: {models_dir}")

    predictor = SpatialOnlyPredictor(str(models_dir), device=device)
    assert set(predictor.tasks) == set(SPATIAL_TASKS), \
        f"发现任务 {sorted(predictor.tasks)}，预期 {sorted(SPATIAL_TASKS)}"
    print(f"  已加载任务模型: {predictor.tasks}")

    # 合成 16 tick 回合，逐 tick 全任务推理（无路径/decoder，速度很快）
    sample = _make_synthetic_sample(T=16)
    out = predictor.predict_round_full(sample, chunk=16)
    assert out["T"] == 16 and len(out["ticks"]) == 16
    for tick in out["ticks"]:
        for task in SPATIAL_TASKS:
            vals = tick[task]
            assert vals is not None and all(
                v is None or (isinstance(v, float) and math.isfinite(v))
                for v in vals), f"tick {tick['tick']} {task} 存在非有限/非法值"
    print("  合成回合 16 tick 全任务推理: 输出全部为有限值 ✓")
    print("  ✓ spatial-only checkpoint 加载测试通过")
    return predictor


def check_pipeline(data_path: Path, engine: PredictionEngine,
                   predictor: SpatialOnlyPredictor, maps_dir: str,
                   run_forward: bool) -> None:
    """内置回合数据完整 pipeline：json.gz → 预处理 → process_round → 模型推理。

    与训练 / 可视化工具完全相同的预处理链路：
      json.gz → filter_data → _convert_inventory_indices → process_round。
    """
    print(f"\n[3/3] 回合数据完整 pipeline: {data_path.name}")

    from create_training_data import _convert_inventory_indices
    from replay_tool.filter import filter_data
    from training_data.map_loader import get_map_geometry
    from training_data.round_processor import process_round

    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("format") == "cs2.demo.v2", \
        f"测试数据格式不符: {data.get('format')}"
    map_name = data.get("map", "unknown")
    print(f"  数据: {map_name}, {len(data.get('rounds', []))} 回合, "
          f"{len(data.get('players', []))} 玩家")

    filter_data(data)
    round_data = data["rounds"][0]

    # 与训练管线对齐：动态 weapon 索引 → config.py 规范索引
    _convert_inventory_indices({"weapons": data.get("weapons", {}),
                                "rounds": [round_data]})

    map_geom = None
    try:
        map_geom = get_map_geometry(map_name, Path(maps_dir))
        print(f"  地图几何: {map_name} (深度图启用)")
    except FileNotFoundError:
        print(f"  ⚠ 未找到 {map_name} 的地图 OBJ，深度图跳过")

    sample = process_round(
        round_data,
        map_geom=map_geom,
        source_file=data_path.name,
        match_teams=None,
        players_meta=data.get("players"),
        tick_interval=0.25,
        compute_depth=map_geom is not None,
        places=data.get("places"),
    )
    round_T = int(sample["meta"]["T"])
    sample = _slice_round(sample, min(round_T, PIPELINE_MAX_TICKS))
    print(f"  预处理完成: {map_name}, {round_T} tick → 测试 {sample['meta']['T']} tick")

    # spatial-only 对真实局面逐 tick 推理
    out = predictor.predict_round_full(sample, chunk=16)
    assert len(out["ticks"]) == sample["meta"]["T"]
    n_finite = 0
    for tick in out["ticks"]:
        ok = all(
            tick[task] is not None and all(
                v is None or (isinstance(v, float) and math.isfinite(v))
                for v in tick[task])
            for task in SPATIAL_TASKS)
        n_finite += int(ok)
    assert n_finite == len(out["ticks"]), \
        f"pipeline 推理: {n_finite}/{len(out['ticks'])} tick 输出有效"
    print(f"  spatial-only 逐 tick 推理: {n_finite}/{len(out['ticks'])} tick 全部有限值 ✓")

    if run_forward:
        print("  --forward: 对真实局面跑自回归路径预测（CPU 约 1-3 分钟）...")
        query_tick = max(0, sample["meta"]["T"] // 2)
        result = engine.predict_at_tick(sample, query_tick=query_tick)
        n_alive = sum(1 for t in result["trajectories"] if t["is_alive"])
        n_pred = sum(1 for t in result["trajectories"]
                     if t["is_alive"] and t["pred_steps"] > 0)
        assert n_pred == n_alive
        print(f"  AR 推理完成: {n_alive} 名存活玩家全部产出轨迹 ✓")

    print("  ✓ 回合数据完整 pipeline 测试通过")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", default="checkpoints",
                    help="模型目录（默认 checkpoints/）")
    ap.add_argument("--config", default="config/pretrain-a100-pro.yaml",
                    help="路径预测模型架构配置")
    ap.add_argument("--maps-dir", default="maps/optimized_obj_files",
                    help="地图 OBJ 目录（深度图 raycast 用）")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--forward", action="store_true",
                    help="额外跑自回归路径预测（合成局面 + 内置回合数据）")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        print(f"模型目录不存在: {models_dir}（先运行 "
              f"python scripts/download_checkpoints.py）", file=sys.stderr)
        return 1

    pretrain_path = models_dir / PRETRAIN_FILE
    if not pretrain_path.exists():
        print(f"缺少 {PRETRAIN_FILE}（先运行 "
              f"python scripts/download_checkpoints.py）", file=sys.stderr)
        return 1

    torch.set_num_threads(max(1, torch.get_num_threads()))
    engine = check_pretrain(args.config, pretrain_path, args.device,
                            args.maps_dir, args.forward)
    predictor = check_spatial(models_dir, args.device)

    if PIPELINE_DATA.exists():
        check_pipeline(PIPELINE_DATA, engine, predictor,
                       args.maps_dir, args.forward)
    else:
        print(f"\n[3/3] 跳过 pipeline 测试：未找到 {PIPELINE_DATA.name}")

    print("\n✅ 全部测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
