import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import yaml
import os
import wandb

from models.model3_space_only import CSModelV3
from dataset.model3_wds_space_only import build_dataloader, move_to_device


# =========================
# Loss
# =========================
def get_loss_fn(task):
    if task in ["winrate", "duel", "alive_in_the_end"]:
        return nn.BCEWithLogitsLoss()
    elif task in ["nxt_kill", "nxt_death"]:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError


def compute_loss(task, out, label, loss_fn):
    if task in ["winrate", "duel", "alive_in_the_end"]:
        return loss_fn(out, label.float())
    elif task in ["nxt_kill", "nxt_death"]:
        return loss_fn(out, label.long())
    else:
        raise ValueError


# =========================
# Eval
# =========================
@torch.no_grad()
def evaluate(model, dataloader, device, task, test_samples, batch_size):
    model.eval()
    loss_fn = get_loss_fn(task)

    total_loss = 0
    total_acc = 0
    step = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="eval", total=(test_samples + batch_size - 1) // batch_size, dynamic_ncols=True):
        batch = move_to_device(batch, device)
        out, label = model(batch)
        loss = compute_loss(task, out, label, loss_fn)

        # 计算准确率
        if task in ["winrate", "duel"]:
            # 二分类，out: (B,), label: (B,)
            preds = (torch.sigmoid(out) > 0.5).long()
            acc = (preds == label.long()).float().mean().item()
            total_acc += acc * label.size(0)
            total_samples += label.size(0)
        elif task in ["nxt_kill", "nxt_death"]:
            # 多分类，out: (B, C), label: (B,)
            preds = torch.argmax(out, dim=-1)
            acc = (preds == label.long()).float().mean().item()
            total_acc += acc * label.size(0)
            total_samples += label.size(0)
        elif task == "alive_in_the_end":
            # 多标签二分类，out: (B, 10), label: (B, 10)
            preds = (torch.sigmoid(out) > 0.5).long()
            acc = (preds == label.long()).float().mean().item()
            total_acc += acc * label.numel()
            total_samples += label.numel()
        else:
            acc = 0.0

        total_loss += loss.item()
        step += 1

    avg_loss = total_loss / step
    avg_acc = total_acc / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


# =========================
# Train
# =========================
def train(device=None, save_path="ckpt", dataset_path="data", cfg_path="config/model3.yaml"):

    cfg = yaml.safe_load(open(cfg_path))

    # ===== wandb 初始化 =====
    wandb.init(
        project="model3",
        name=f"train_{cfg['task']}",
        config=cfg,
        reinit=True
    )

    print("="*50)
    print("Configuration:")
    print(yaml.dump(cfg))
    print("="*50)

    task = cfg["task"]
    train_cfg = cfg["Training"]

    # 优先使用传入的 device，否则自动检测
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # ===== model =====
    model = CSModelV3(cfg).to(device)

    # if device == "cuda":
    #     print("Using torch.compile...")
    #     model = torch.compile(model, mode="max-autotune")
    
    torch.set_float32_matmul_precision('high')
    print("Set float32 matmul precision to high")

    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    # ===== dataloader =====
    train_loader = build_dataloader(
        dataset_path,
        split="train",
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        task=task,
    )

    val_loader = build_dataloader(
        dataset_path,
        split="test",
        batch_size=train_cfg["batch_size"],
        num_workers=0,
        task=task,
    )

    # ===== optimizer =====
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=1e-4,
    )

    # ===== scheduler =====
    total_steps = (train_cfg["train_samples"] // train_cfg["batch_size"]) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warm_up_ratio"])
    min_lr = train_cfg["min_lr"]
    base_lr = train_cfg["lr"]

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))
        return max(min_lr / base_lr, cosine_decay.item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = get_loss_fn(task)

    os.makedirs(save_path, exist_ok=True)

    best_val = float("inf")
    best_acc = 0.0
    global_step = 0

    # =========================
    # training loop
    # =========================

    number_batches = train_cfg["train_samples"] // train_cfg["batch_size"]
    
    print(f"Number of batches per epoch: {number_batches}")

    # evaluate before training
    print(f"\n[Before Training] Running evaluation...")
    val_loss, val_acc = evaluate(model, val_loader, device, task, test_samples=train_cfg["test_samples"], batch_size=train_cfg["batch_size"])
    print(f"[Before Training] val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": -1, "step": global_step})

    for epoch in range(train_cfg["epochs"]):
        print(f"\n===== Epoch {epoch} =====")
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=number_batches, dynamic_ncols=True)

        sum_loss = 0.0
        sum_acc = 0.0
        sum_steps = 0

        for batch in pbar:
            batch = move_to_device(batch, device)

            out, label = model(batch)
            loss = compute_loss(task, out, label, loss_fn)

            # 计算训练准确率
            if task in ["winrate", "duel"]:
                preds = (torch.sigmoid(out) > 0.5).long()
                acc = (preds == label.long()).float().mean().item()
            elif task in ["nxt_kill", "nxt_death"]:
                preds = torch.argmax(out, dim=-1)
                acc = (preds == label.long()).float().mean().item()
            elif task == "alive_in_the_end":
                preds = (torch.sigmoid(out) > 0.5).long()
                acc = (preds == label.long()).float().mean().item()
            else:
                acc = 0.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            
            sum_loss += loss.item()
            sum_acc += acc
            sum_steps += 1

            current_lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}", "avg_loss": f"{sum_loss/sum_steps:.4f}", "avg_acc": f"{sum_acc/sum_steps:.4f}", "lr": f"{current_lr:.6f}"})

            # wandb log

            wandb.log({"train/loss": loss.item(), "train/acc": acc, "train/avg_loss": sum_loss/sum_steps, "train/avg_acc": sum_acc/sum_steps, "epoch": epoch, "step": global_step, "lr": current_lr})

            # =========================
            # 🔥 step-based eval
            # =========================
            if global_step % train_cfg["test_interval"] == 0:
                print(f"\n[Step {global_step}] Running evaluation...")

                val_loss, val_acc = evaluate(model, val_loader, device, task, test_samples=train_cfg["test_samples"], batch_size=train_cfg["batch_size"])
                model.train()

                print(f"[Step {global_step}] val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch, "step": global_step})

                # ===== save latest =====
                torch.save(model.state_dict(), f"{save_path}/latest_{task}.pt")

                # ===== save best =====
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), f"{save_path}/best_{task}.pt")
                    print("Saved BEST model")
                # 也可以根据准确率保存最佳模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), f"{save_path}/best_acc_{task}.pt")
                    print("Saved BEST ACC model")

        wandb.log({"train/epoch_loss": sum_loss/sum_steps, "train/epoch_acc": sum_acc/sum_steps, "epoch": epoch, "step": global_step, "lr": current_lr})

        # =========================
        # 🔥 epoch-based eval
        # =========================
        print(f"\n[Epoch {epoch}] Running evaluation...")

        val_loss, val_acc = evaluate(model, val_loader, device, task, test_samples=train_cfg["test_samples"], batch_size=train_cfg["batch_size"])
        model.train()

        print(f"[Epoch {epoch}] val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch, "step": global_step})

        # ===== save latest =====
        torch.save(model.state_dict(), f"{save_path}/latest_{task}.pt")

        # ===== save best =====
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{save_path}/best_{task}.pt")
            print("Saved BEST model")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{save_path}/best_acc_{task}.pt")
            print("Saved BEST ACC model")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps，优先级高于自动检测")
    parser.add_argument("--save_path", type=str, default="ckpt", help="模型权重保存路径")
    parser.add_argument("--dataset_path", type=str, default="data", help="数据集路径")
    parser.add_argument("--cfg_path", type=str, default="config/model3.yaml", help="模型配置文件路径")
    args = parser.parse_args()
    train(device=args.device, save_path=args.save_path, dataset_path=args.dataset_path, cfg_path=args.cfg_path)


# python -m scripts.train3 --device "cuda" --save_path chkpt3 --dataset_path /data/dataset --cfg_path config/model3_win.yaml