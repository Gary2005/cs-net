#!/usr/bin/env python3
"""
验证 cs-net-v4 模型 checkpoint 能正确加载（下载模型后运行）。

用法:
    python scripts/test_checkpoints.py --models-dir checkpoints
    python scripts/test_checkpoints.py --models-dir checkpoints --device cpu
    python scripts/test_checkpoints.py --models-dir checkpoints --forward
        # --forward: 额外对路径预测模型跑一次合成局面的 16 tick 自回归推理
        #            （CPU 上约 1-3 分钟，默认关闭）

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
"""

from __future__ import annotations

import argparse
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


def _cfg_from_yaml(path: str) -> PretrainConfig:
    with open(path, "r") as f:
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


def check_pretrain(config_path: str, ckpt_path: Path, device: str,
                   maps_dir: str, run_forward: bool) -> None:
    print(f"\n[1/2] 路径预测 checkpoint: {ckpt_path.name}")

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


def check_spatial(models_dir: Path, device: str) -> None:
    print(f"\n[2/2] spatial-only checkpoints: {models_dir}")

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
                    help="额外对路径预测模型跑一次合成局面自回归推理")
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
    check_pretrain(args.config, pretrain_path, args.device,
                   args.maps_dir, args.forward)
    check_spatial(models_dir, args.device)

    print("\n✅ 全部 checkpoint 加载测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
