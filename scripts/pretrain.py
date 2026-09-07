#!/usr/bin/env python3
"""
CS2 预训练 — 训练入口。

Usage:
    python scripts/pretrain.py --config config/pretrain.yaml
    python scripts/pretrain.py --config config/pretrain.yaml --batch-size 4 --lr 5e-5
    python scripts/pretrain.py --data-dir data/dataset --device cuda  # 直接用默认参数
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pretrain_model import CS2PretrainModel, PretrainConfig
from training_data.torch_dataset import CS2PretrainDataset, pretrain_collate_fn


# ── defaults ──────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data_dir": "data/dataset",
    "device": "cuda",
    "batch_size": 8,
    "epochs": 100,
    "lr": 1e-4,
    "warmup_steps": 1000,
    "total_steps": 600000,
    "max_grad_norm": 1.0,
    "grad_accum_steps": 1,
    "save_dir": "checkpoints/pretrain",
    "resume": "",
    "log_interval": 50,
    "save_interval": 5000,
    "val_interval": 2000,
    "max_samples": 0,
    "n_ticks": 64,
    "stride": 16,
    "jitter": True,
    "shuffle_buffer": 10000,
    "num_workers": 8,
    "keep_ratio": 1.0,      # 序列保留比例：从每步 N=B*T*10 个序列任务中随机保留的比例（<1 省 decoder 计算）
    # 优化开关（--no-xxx 可关闭）
    "use_amp": True,        # BF16 混合精度
    "use_tf32": True,       # TF32 + cudnn.benchmark
    "use_compile": True,    # torch.compile
    # model
    "d_model": 256,
    "n_spatial_layers": 4,
    "n_temporal_layers": 4,
    "n_decoder_layers": 2,
    "n_depth_ray_layers": 2,
    "n_heads": 8,
    "d_ff": 1024,
    "dropout": 0.1,
    # discrete tokenization
    "move_range": 128.0,
    "move_grid_size": 1.0,
    "angle_grid_size": 1.0,
    "use_residual_correction": True,  # 残差修正（逐 tick 编码-解码-积累）
    # wandb
    "wandb": False,
    "wandb_project": "cs2-pretrain",
    "wandb_name": None,
    "wandb_entity": None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="CS2 Pretraining")
    parser.add_argument("--config", type=str, default="", help="YAML 配置文件路径")
    # 以下参数可覆盖 config 文件中的值
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--val-interval", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-spatial-layers", type=int)
    parser.add_argument("--n-temporal-layers", type=int)
    parser.add_argument("--n-decoder-layers", type=int)
    parser.add_argument("--n-depth-ray-layers", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--n-ticks", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--shuffle-buffer", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--keep-ratio", type=float, help="序列保留比例（0,1]；<1 时每步随机保留 B*T*10 序列的一部分，省 decoder 计算")
    parser.add_argument("--no-amp", action="store_true", help="禁用 BF16 AMP")
    parser.add_argument("--no-tf32", action="store_true", help="禁用 TF32 + cudnn.benchmark")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile")
    parser.add_argument("--no-jitter", action="store_true", help="禁用窗口滑动随机抖动")
    parser.add_argument("--no-residual-correction", action="store_true", help="禁用残差修正（加速数据加载）")
    parser.add_argument("--n-deg-grid", type=int, help="角度网格 bin 数量")
    parser.add_argument("--output-dim", type=int, help="每 bin 输出维度")
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-name", type=str)
    parser.add_argument("--wandb-entity", type=str)
    return parser.parse_args()


def load_config(args) -> dict:
    """加载 YAML config，CLI 参数覆盖。"""
    cfg = dict(DEFAULT_CONFIG)

    if args.config:
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        if yaml_cfg:
            cfg.update(yaml_cfg)

    # CLI 参数覆盖（只覆盖显式传入的，None 表示未传入）
    _no_prefix_args = {"config", "no_wandb", "no_amp", "no_tf32", "no_compile", "no_jitter"}
    cli_overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k not in _no_prefix_args}
    cfg.update(cli_overrides)

    if args.no_wandb:
        cfg["wandb"] = False
    if args.no_amp:
        cfg["use_amp"] = False
    if args.no_tf32:
        cfg["use_tf32"] = False
    if args.no_compile:
        cfg["use_compile"] = False
    if args.no_jitter:
        cfg["jitter"] = False
    if args.no_residual_correction:
        cfg["use_residual_correction"] = False

    return cfg


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay to 0."""
    import math

    def lr_lambda(step: int) -> float:
        step = max(1, step)
        if step < warmup_steps:
            return float(step) / warmup_steps                      # linear warmup
        if step >= total_steps:
            return 0.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))          # cosine decay → 0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, loader, device, use_amp: bool = True, global_step: int = 0):
    model.eval()
    total_loss, count = 0.0, 0
    metric_sums = {}
    for batch in tqdm(iter(loader), desc="  Validating", unit="step", leave=False):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        labels = batch["label_camera"]
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with amp_ctx:
            out = model(batch, labels, global_step=global_step)
        total_loss += out["loss"].item()
        for k, v in out["metrics"].items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v.item()
        count += 1
    model.train()
    result = {"loss": total_loss / max(count, 1),
              **{k: v / max(count, 1) for k, v in metric_sums.items()}}
    return result


def main():
    args = parse_args()
    cfg = load_config(args)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── CUDA 后端优化 ──────────────────────────────────────────────────
    if device.type == "cuda" and cfg.get("use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("TF32 + cudnn.benchmark: ON")
    # ── 完整配置输出 ─────────────────────────────────────────────────
    print("━" * 56)
    print("  Model Config:")
    print(f"    d_model={cfg['d_model']}, n_heads={cfg.get('n_heads', 8)}, d_ff={cfg.get('d_ff', 1024)}")
    print(f"    n_spatial_layers={cfg['n_spatial_layers']}, n_temporal_layers={cfg['n_temporal_layers']}")
    print(f"    n_decoder_layers={cfg['n_decoder_layers']}, n_depth_ray_layers={cfg.get('n_depth_ray_layers', 2)}")
    print(f"    dropout={cfg.get('dropout', 0.1)}, n_ticks={cfg['n_ticks']}, stride={cfg.get('stride', 16)}")
    print(f"    move_range={cfg.get('move_range', 128)}, move_grid_size={cfg.get('move_grid_size', 1.0)}, angle_grid_size={cfg.get('angle_grid_size', 1.0)}")
    print("  Training Config:")
    print(f"    batch_size={cfg['batch_size']}, grad_accum={cfg.get('grad_accum_steps', 1)}")
    print(f"    lr={cfg['lr']}, warmup={cfg.get('warmup_steps', 1000)}, max_grad_norm={cfg.get('max_grad_norm', 1.0)}")
    print(f"    epochs={cfg['epochs']}, use_amp={cfg.get('use_amp', True)}, use_compile={cfg.get('use_compile', True)}")
    print(f"    jitter={cfg.get('jitter', True)}, num_workers={cfg.get('num_workers', 8)}")
    print(f"    shuffle_buffer={cfg.get('shuffle_buffer', 10000)}, max_samples={cfg.get('max_samples', 0)}")
    print(f"    keep_ratio={cfg.get('keep_ratio', 1.0)}（每步 N=B*T*10 个序列中保留的比例）")
    print("━" * 56)

    # ── Wandb ─────────────────────────────────────────────────────────────
    use_wandb = cfg.get("wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "cs2-pretrain"),
            name=cfg.get("wandb_name"),
            entity=cfg.get("wandb_entity"),
            config=cfg,
        )
        print(f"Wandb: {wandb.run.name}")

    # ── Model ──────────────────────────────────────────────────────────────
    model_cfg = PretrainConfig(
        d_model=cfg["d_model"],
        n_spatial_layers=cfg["n_spatial_layers"],
        n_temporal_layers=cfg["n_temporal_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        n_depth_ray_layers=cfg.get("n_depth_ray_layers", 2),
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        n_ticks=cfg["n_ticks"],
        move_range=cfg.get("move_range", 128.0),
        move_grid_size=cfg.get("move_grid_size", 1.0),
        angle_grid_size=cfg.get("angle_grid_size", 1.0),
        use_residual_correction=cfg.get("use_residual_correction", True),
    )
    model = CS2PretrainModel(model_cfg).to(device)
    base_model = model
    if cfg.get("use_compile", True):
        model = torch.compile(model)  # default mode，跳过有问题的 CUDA Graphs
        print("torch.compile: ON")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M")
    print(f"Vocab size: {model.tokenizer.vocab_size}, seq_len: {model.decoder.seq_len} (= {model.tokenizer.TOKENS_PER_GROUP} × {cfg['n_ticks']})")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95))
    scheduler = get_lr_scheduler(optimizer, cfg["warmup_steps"], cfg["total_steps"])

    # ── DataLoaders ────────────────────────────────────────────────────────
    max_samples = cfg["max_samples"] if cfg["max_samples"] > 0 else None
    train_ds = CS2PretrainDataset(
        cfg["data_dir"], split="train",
        n_ticks=cfg["n_ticks"], stride=cfg["stride"],
        shuffle_buffer=cfg["shuffle_buffer"], augment_depth=True,
        max_samples=max_samples, jitter=cfg["jitter"],
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        collate_fn=pretrain_collate_fn, num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    test_loader = None
    test_dir = Path(cfg["data_dir"]) / "test"
    if test_dir.exists() and any(test_dir.glob("shards-*.tar")):
        val_max = cfg.get("val_max_samples", 2000)  # 只跑 2000 个窗口
        test_ds = CS2PretrainDataset(
            cfg["data_dir"], split="test",
            n_ticks=cfg["n_ticks"], stride=cfg["stride"],
            shuffle_buffer=0, augment_depth=True, max_samples=val_max,
            jitter=False,  # val/test 不需要随机抖动
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg["batch_size"],
            collate_fn=pretrain_collate_fn, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch, global_step = 0, 0
    if cfg["resume"]:
        # 加载到 CPU，避免 checkpoint 本身（模型+optimizer 约 3× 模型大小）占 GPU 显存
        ckpt = torch.load(cfg["resume"], map_location="cpu", weights_only=False)
        # 剥离 torch.compile 保存时带上的 _orig_mod. 前缀，
        # 并加载到底层原始模型，避免 use_compile 开关变化时静默加载失败。
        ckpt_state = ckpt["model"]
        stripped_state = {}
        for k, v in ckpt_state.items():
            stripped_state[k.replace("_orig_mod.", "")] = v

        # strict=False: 兼容旧 checkpoint 缺少 dead_depth_emb 等新参数
        missing, unexpected = base_model.load_state_dict(stripped_state, strict=False)
        if missing:
            print(f"  [resume] Missing keys (using fresh init): {missing}")
        if unexpected:
            print(f"  [resume] Unexpected keys (ignored): {unexpected}")
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]

        # 计算当前步数 scheduler 理论上应该给出的 lr
        def _expected_lr(step: int) -> float:
            step = max(1, step)
            if step < cfg["warmup_steps"]:
                return cfg["lr"] * float(step) / cfg["warmup_steps"]
            if step >= cfg["total_steps"]:
                return 0.0
            progress = (step - cfg["warmup_steps"]) / max(1, cfg["total_steps"] - cfg["warmup_steps"])
            return cfg["lr"] * 0.5 * (1.0 + math.cos(math.pi * progress))

        ckpt_lr = ckpt["optimizer"]["param_groups"][0]["lr"]
        expected_lr = _expected_lr(global_step)
        if abs(ckpt_lr - expected_lr) < 1e-6:
            # lr 与 schedule 吻合 → 正常恢复
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            else:
                for _ in range(global_step):
                    scheduler.step()
            print(f"Resumed from {cfg['resume']} (epoch {start_epoch}, step {global_step}, "
                  f"lr={ckpt_lr:.2e} — optimizer/scheduler restored)")
        else:
            # lr 被用户故意修改了，optimizer 动量不适用于新 lr
            scheduler.last_epoch = global_step
            print(f"Resumed from {cfg['resume']} (epoch {start_epoch}, step {global_step}, "
                  f"lr {ckpt_lr:.2e} → {cfg['lr']:.2e} (base) — optimizer/scheduler reset)")

        # 显式释放 checkpoint dict，避免 GC 延迟导致显存残留
        del ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ── Training Loop ──────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    done = False

    for epoch in range(start_epoch, cfg["epochs"]):
        if done:
            break
        epoch_loss, epoch_steps = 0.0, 0
        # 窗口内累计值（用于 log_interval 平均）
        win_loss = 0.0
        win_metrics = {}
        win_steps = 0
        t0 = time.time()
        pbar = tqdm(iter(train_loader), desc=f"Epoch {epoch:3d}", unit="step")

        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch["label_camera"]
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if cfg.get("use_amp", True) else nullcontext()
            with amp_ctx:
                out = model(batch, labels, global_step=global_step,
                            keep_ratio=cfg.get("keep_ratio", 1.0))

            loss = out["loss"] / cfg["grad_accum_steps"]
            loss.backward()

            epoch_loss += out["loss"].item()
            epoch_steps += 1

            # 窗口累计（用于 log_interval 平均）
            win_loss += out["loss"].item()
            for k, v in out["metrics"].items():
                win_metrics[k] = win_metrics.get(k, 0.0) + v.item()
            win_steps += 1

            # 梯度累积：每 grad_accum_steps 步更新一次
            if epoch_steps % cfg["grad_accum_steps"] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step >= cfg["total_steps"]:
                    tqdm.write(f"✓ Reached total_steps={cfg['total_steps']}, stopping.")
                    done = True
                    break

                # ── log / save / val（仅在 optimizer step 后执行）──
                if global_step % cfg["log_interval"] == 0:
                    elapsed = time.time() - t0
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = win_loss / win_steps
                    avg_metrics = {k: v / win_steps for k, v in win_metrics.items()}

                    msg = (f"step {global_step:6d} | loss {avg_loss:.4f} "
                           f"token_acc {avg_metrics.get('token_acc', 0):.3f} "
                           f"| lr {lr:.2e} | {elapsed:.1f}s")
                    tqdm.write(msg)
                    wandb_log = {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        **{f"train/{k}": v for k, v in avg_metrics.items()},
                        "step": global_step,
                    }
                    if use_wandb:
                        wandb.log(wandb_log)
                    t0 = time.time()
                    # 重置窗口
                    win_loss = 0.0
                    win_metrics.clear()
                    win_steps = 0

                if global_step % cfg["save_interval"] == 0:
                    path = f"{cfg['save_dir']}/step_{global_step:07d}.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "epoch": epoch, "global_step": global_step}, path)
                    tqdm.write(f"✓ Saved {path}")

                if global_step % cfg["val_interval"] == 0:
                    # 保存 latest checkpoint（覆盖式，用于恢复/推理）
                    latest_path = f"{cfg['save_dir']}/latest.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "epoch": epoch, "global_step": global_step}, latest_path)

                    if test_loader is not None:
                        val_metrics = validate(model, test_loader, device, use_amp=cfg.get("use_amp", True), global_step=global_step)
                        tqdm.write(f"  [val] step {global_step:6d} | loss {val_metrics['loss']:.4f} "
                              f"token_acc {val_metrics.get('token_acc', 0):.3f}")
                        if use_wandb:
                            wandb.log({f"val/{k}": v for k, v in val_metrics.items()} | {"step": global_step})

            # 实时显示 loss（不打断进度条）
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "step": global_step,
                "loss": f"{out['loss'].item():.3f}",
                "acc": f"{out['metrics'].get('token_acc', torch.tensor(0)).item():.2f}",
                "lr": f"{lr_now:.2e}",
            })

        avg_loss = epoch_loss / max(epoch_steps, 1)
        tqdm.write(f"Epoch {epoch:3d} | loss {avg_loss:.4f}")
        if use_wandb:
            wandb.log({"epoch": epoch, "epoch/loss": avg_loss})

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "global_step": global_step},
                   f"{cfg['save_dir']}/epoch_{epoch:03d}.pt")

    final_path = f"{cfg['save_dir']}/final.pt"
    torch.save({"model": model.state_dict(), "global_step": global_step}, final_path)
    print(f"✓ Saved final checkpoint to {final_path}")

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
