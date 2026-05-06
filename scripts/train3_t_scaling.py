import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from dataset.model3_wds_space_only import build_dataloader, move_to_device
from models.model3_space_only import CSModelV3


# =========================
# metrics
# =========================

def binary_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    probs = probs.view(-1)
    labels = labels.view(-1).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        if mask.any():
            conf = probs[mask].mean()
            acc = labels[mask].mean()
            ece = ece + (mask.float().mean() * (acc - conf).abs())
    return ece.item()


def multiclass_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=-1)
    acc = (pred == labels).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (conf >= lo) & (conf < hi)
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        if mask.any():
            conf_avg = conf[mask].mean()
            acc_avg = acc[mask].mean()
            ece = ece + (mask.float().mean() * (acc_avg - conf_avg).abs())
    return ece.item()


# =========================
# loss & metrics
# =========================

def get_loss_fn(task: str) -> nn.Module:
    if task in ["winrate", "duel", "alive_in_the_end"]:
        return nn.BCEWithLogitsLoss()
    if task in ["nxt_kill", "nxt_death"]:
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported task: {task}")


def compute_metrics(task: str, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    loss_fn = get_loss_fn(task)

    if task in ["winrate", "duel"]:
        loss = loss_fn(logits, labels.float()).item()
        probs = torch.sigmoid(logits)
        acc = (probs > 0.5).long().eq(labels.long()).float().mean().item()
        ece = binary_ece(probs, labels)
        return loss, ece, acc

    if task in ["nxt_kill", "nxt_death"]:
        loss = loss_fn(logits, labels.long()).item()
        probs = torch.softmax(logits, dim=-1)
        acc = probs.argmax(dim=-1).eq(labels.long()).float().mean().item()
        ece = multiclass_ece(probs, labels.long())
        return loss, ece, acc

    if task == "alive_in_the_end":
        loss = loss_fn(logits, labels.float()).item()
        probs = torch.sigmoid(logits)
        acc = (probs > 0.5).long().eq(labels.long()).float().mean().item()
        ece = binary_ece(probs, labels)
        return loss, ece, acc

    raise ValueError(f"Unsupported task: {task}")


# =========================
# loading
# =========================

def find_checkpoint(folder: str) -> str:
    for name in sorted(os.listdir(folder)):
        if name.endswith((".pth", ".pt")):
            return os.path.join(folder, name)
    raise RuntimeError(f"No checkpoint found in {folder}")


def load_model_and_cfg(model_dir: str, cfg_path: str, device: torch.device):
    ckpt = find_checkpoint(model_dir)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = CSModelV3(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    return model, cfg, ckpt


# =========================
# calibration
# =========================

def fit_temperature(task: str, logits: torch.Tensor, labels: torch.Tensor, device: torch.device):
    logits = logits.to(device)
    labels = labels.to(device)

    log_T = nn.Parameter(torch.zeros((), device=device))

    if task in ["winrate", "duel"]:
        loss_fn = nn.BCEWithLogitsLoss()

        def scale(x):
            return x / torch.exp(log_T)

        def loss_val():
            return loss_fn(scale(logits), labels.float())

    elif task in ["nxt_kill", "nxt_death"]:
        loss_fn = nn.CrossEntropyLoss()

        def scale(x):
            return x / torch.exp(log_T)

        def loss_val():
            return loss_fn(scale(logits), labels.long())

    elif task == "alive_in_the_end":
        loss_fn = nn.BCEWithLogitsLoss()

        def scale(x):
            return x / torch.exp(log_T)

        def loss_val():
            return loss_fn(scale(logits), labels.float())

    else:
        raise ValueError(f"Unsupported task: {task}")

    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = loss_val()
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = torch.exp(log_T).detach().cpu().item()
    return temperature


def collect_logits_labels(model, dataloader, device, max_samples=None):
    logits_list = []
    labels_list = []
    total = 0

    for batch in tqdm(dataloader, desc="collect", dynamic_ncols=True):
        batch = move_to_device(batch, device)
        out, label = model(batch)
        logits_list.append(out.detach().cpu())
        labels_list.append(label.detach().cpu())
        total += label.shape[0]
        if max_samples is not None and total >= max_samples:
            break

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    if max_samples is not None and logits.shape[0] > max_samples:
        logits = logits[:max_samples]
        labels = labels[:max_samples]
    return logits, labels


def apply_temperature(logits: torch.Tensor, temperature: float):
    return logits / float(temperature)


def update_yaml(cfg_path: str, temperature: float):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    calib = cfg.setdefault("calibration", {})
    calib["temperature_scaling"] = {
        "temperature": float(temperature),
    }

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Temperature scaling with bias on test split")
    parser.add_argument("--dataset_path", required=True, help="WebDataset root with train/test shards")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on test samples")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_root = os.path.join(repo_root, "cs-net-models")
    config_root = os.path.join(repo_root, "config")

    head_map = {
        "alive": ("alive", "model3_alive_space_only.yaml"),
        "duel": ("duel", "model3_duel_space_only.yaml"),
        "nxt_death": ("nxt_death", "model3_death_space_only.yaml"),
        "nxt_kill": ("nxt_kill", "model3_kill_space_only.yaml"),
        "win_rate": ("win_rate", "model3_win_space_only.yaml"),
    }

    for head, (model_dir_name, cfg_name) in head_map.items():
        model_dir = os.path.join(models_root, model_dir_name)
        cfg_path = os.path.join(config_root, cfg_name)

        print("=" * 60)
        print(f"Calibrating: {head}")
        model, cfg, ckpt = load_model_and_cfg(model_dir, cfg_path, device)
        task = cfg.get("task")
        if task is None:
            raise RuntimeError(f"Missing task in config: {cfg_path}")

        test_samples = None
        if isinstance(cfg.get("Training"), dict):
            test_samples = cfg["Training"].get("test_samples")
        if args.max_samples is not None:
            test_samples = args.max_samples if test_samples is None else min(test_samples, args.max_samples)

        dataloader = build_dataloader(
            args.dataset_path,
            split="test",
            batch_size=cfg["Training"]["batch_size"],
            num_workers=0,
            task=task,
        )

        logits, labels = collect_logits_labels(model, dataloader, device, max_samples=test_samples)

        loss_before, ece_before, acc_before = compute_metrics(task, logits, labels)
        temperature = fit_temperature(task, logits, labels, device)

        logits_scaled = apply_temperature(logits, temperature)
        loss_after, ece_after, acc_after = compute_metrics(task, logits_scaled, labels)

        update_yaml(cfg_path, temperature)

        print(f"Model: {os.path.basename(ckpt)}")
        print(f"Config: {cfg_path}")
        print(f"Task: {task}")
        print(f"T: {temperature:.6f}")
        print(f"Before  loss={loss_before:.6f}  ECE={ece_before:.6f}  acc={acc_before:.6f}")
        print(f"After   loss={loss_after:.6f}  ECE={ece_after:.6f}  acc={acc_after:.6f}")


if __name__ == "__main__":
    main()
