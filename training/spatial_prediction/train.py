"""spatial_prediction 训练入口 —— 超参全在 `config.yaml` 里，代码里不写死。

    # 默认配置（`training/spatial_prediction/config.yaml`）
    python -m training.spatial_prediction.train -c training/spatial_prediction/config.yaml

    # 临时改一项（按 YAML 解析：`1e-4` / `true` / `[6,8]` 都行）
    python -m training.spatial_prediction.train -c training/spatial_prediction/config.yaml \
        --set train.lr=1e-4

    # 续训 / 冒烟
    python -m training.spatial_prediction.train -c ... --resume runs/spatial/<name>/last.pt
    python -m training.spatial_prediction.train -c ... --dry-run

## 配置怎么读

`config.yaml` 分 `data` / `model` / `train` 三段：`model` 那段直接 unpack 成
`SpatialConfig`（字段名就是 `model.py` 里的那些），`data` / `train` 段在这里用。命令行
`--set a.b=c` 在 `yaml.safe_load` 之后覆盖，所以值的写法和配置文件里一样。

日志走 wandb（`train/*`、`valid/*`）+ tqdm，每个 run 在 `train.save_dir/<name>` 下留一份
**实际生效的配置** `config.yaml` 和 `last.pt` /（有 valid 时）`best.pt`。`name` 留空就用
时间戳，不会覆盖上一次。

## 数据

一个样本 = **一帧**，行与行之间没有时间轴（`dataset.py` 的模块 docstring 讲了这个设计）。
所以 `len(ds)` 就是样本数，batch 是纯 stack，没有 padding / `frame_mask` / 截断 —— 和 aim
那边"每个样本 257 帧、要补到 batch 内最长"完全是两回事。

`data.buffer_rows` 决定打散方式（默认 8192 = `ShuffleBufferSampler`，`null` = `RandomSampler`）。
真实规模上这个开关是**翻倍**的差别：282,888 个文件 / 24.2M 行时，全局随机是 145 ms/步
（一半时间在等盘），按文件顺序读 + 内存打散是 69 ms/步。理由和实测都在 `dataset.py`。

## 一个 epoch 大概多久（实测）

RTX 3090 24 G，bf16，`d_model=512` / `layers=6` / B=512 / `num_workers=8`，数据在 F: 这块
980 PRO 上，**24,248,254 行**（= 47,359 步）：

    模型本身（数据已在显存里）     55.1 ms/步   9,294 行/s   → 43 min/epoch  峰值 2.11 GiB
    真实训练循环（buffer 打散）    69 ms/步     7.4k 行/s    → 54 min/epoch  峰值 2.4 GiB
    真实训练循环（shuffle=true）  145 ms/步     3.5k 行/s    → 115 min/epoch
    纯前向（评估用）               17.2 ms/步   29,852 行/s  → 全量 valid（1.4M 行）约 1 分钟

10 个 epoch ≈ **9 小时**（含每轮一次全量 valid）。想缩：先砍 `layers`（6 → 4 省 1/3），再砍
`d_model`（512 → 384 省一半算力），最后才是 `batch_size` —— 见 `model.py` 成本一节。

⚠ Windows 上显存超了**不会报 OOM**：驱动把它换成共享内存，单步从 55 ms 变成几秒，表现是
"卡住不动"。看到 ms/步 一个数量级地涨，先查 `data.batch_size`。反过来，**B=1024 只值 9%**
（101 ms/步 / 10,137 行/s），B=2048 已经完全不值（206 ms / 9,941 行/s，8.2 GiB）。

## loss 与指标的口径

5 个 head 的 loss 见 `model.spatial_loss`；权重在 `train.loss_weights`（默认全 1）。这里只
提醒两件容易看错的事：

* `train/acc_alive` / `train/acc_can_kill` **只在活人上算**（死人那两个标签恒 False，不掩掉
  的话分母里一半是"免费的 0"，acc 会虚高到 0.8 以上）。
* `train/acc_next_death` / `train/acc_next_kill` 是 11 类，**纯瞎猜是 9.1%**（`ln 11 = 2.398`
  的 CE）。它们不像 aim 的 acc 那样天生贴 0.5，**低于 0.11 就是没学到**。

## 为什么 valid 按回合计权

行不独立：一个回合几十上百行共享 `ct_win`（整回合常量）、`alive_end` 也高度相似，而行数
随回合长度变（K 中位 86、最长 5,897）。直接对行取平均 = 按"回合有多长"给回合计权，
长回合（往往是残局拖很久的那种）会占掉大部分指标。所以 valid 上用 `ds.sample_weights()`
（每行 `1/K`，每个回合总权重相等）加权：`acc = Σ(w·对) / Σw`，这就是"回合级"的准确率。

训练日志里**不**加权（每行等权）—— 那是 loss 的口径，也是 GPU 实际看到的东西；两者不一致
是有意的，别拿 `train/acc_*` 和 `valid/acc_*` 直接比绝对值。

评估要 `shuffle=False` + `drop_last=False`：顺序读让行号连续，才能按下标切 `sample_weights()`
（`pos` 累加），也保证一条样本都不漏。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import torch
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import yaml

from training.spatial_prediction.dataset import SpatialDataset, make_loader
from training.spatial_prediction.model import (
    NO_EVENT,
    SpatialConfig,
    SpatialModel,
    loss_weights_of,
    spatial_correct,
    spatial_loss,
)

# ---------------------------------------------------------------------------- 配置


def load_config(path: str | Path) -> dict:
    """读 `config.yaml`（空文件当空 dict，缺的段下面用 `.get` 兜底）。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_override(cfg: dict, spec: str) -> None:
    """`"train.lr=1e-4"` → 改 `cfg` 里对应的项（值按 YAML 解析，不是全当字符串）。"""
    key, sep, raw = spec.partition("=")
    if not sep:
        raise SystemExit(f"--set 需要 KEY=VALUE，收到 {spec!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        node = node[p]
    node[leaf] = yaml.safe_load(raw)


def save_config(cfg: dict, path: Path) -> None:
    """把**实际跑的那份**配置落在 run 目录里（`--set` 之后、默认值补全之前）。"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------- 数据


def try_split(
    root: str, split: str, *, max_cached_files: int | None = None
) -> SpatialDataset | None:
    """取一个 split，没有（目录不存在 / 空的）就返回 `None`。

    评估**不截断、不换方向**：这里本来就没有 aim 那种随机截断和 A/B 对调，`__getitem__`
    是纯读取，所以同一个 ckpt 评两次拿到的是同一个数（这点和 aim 不同）。
    """
    try:
        return SpatialDataset(root, split=split, max_cached_files=max_cached_files)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------- 训练


def make_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """权重衰减只加在矩阵乘法的权重上：bias / LayerNorm / Embedding 不衰减。"""
    emb = {
        id(p)
        for m in model.modules()
        if isinstance(m, torch.nn.Embedding)
        for p in m.parameters()
    }
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith("bias") or id(p) in emb else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_scale(step: int, total: int, warmup: int, min_frac: float) -> float:
    """线性 warmup + 余弦退火到 `min_frac`。`step` 是已经走完的优化步数。"""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p))


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@contextlib.contextmanager
def maybe_autocast(dtype: torch.dtype | None, device: torch.device):
    """`dtype=None` 时啥也不做（fp32 训练）。"""
    if dtype is None:
        yield
    else:
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield


class MetricStore:
    """按「分子 / 分母」累加指标，**只在 `ratio()` 时 sync 一次**。

    为什么不每个 batch 算一个 acc 再平均：那是"按 batch 等权"，而 batch 里的活人数、
    行的回合长度都不一样（valid 最后一个 batch 还更短），加权口径会偏。存分子分母、
    最后一次相除才是精确的 `Σ(w·对) / Σw`。

    为什么不在每个 step 上 `.item()`：每个 `.item()` 都是一次 GPU→CPU 同步，10 个指标
    × 每个 step 会把训练循环拖成同步执行（实测 B=512 时单步只有 ~20 ms，10 次同步能占掉
    一成以上）。这里分子分母都留在 GPU 上，累加是纯粹的 tensor 加。
    """

    def __init__(self) -> None:
        self._num: dict[str, torch.Tensor] = {}
        self._den: dict[str, torch.Tensor] = {}

    def add(self, key: str, num: torch.Tensor, den: torch.Tensor) -> None:
        self._num[key] = self._num.get(key, 0.0) + num
        self._den[key] = self._den.get(key, 0.0) + den

    def ratio(self, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        for k, num in self._num.items():
            den = self._den[k].clamp_min(1e-9)
            out[f"{prefix}{k}"] = float(num / den)
        return out

    def clear(self) -> None:
        self._num.clear()
        self._den.clear()

    def __len__(self) -> int:
        return len(self._num)


def add_batch(
    store: MetricStore,
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    total: torch.Tensor,
    parts: dict[str, torch.Tensor],
    weights: torch.Tensor | None = None,
) -> None:
    """一个 batch 的「5 个 acc + 5 个预测正类比例 + 6 个 loss」全部按分子/分母累加。

    `weights (B,)` 给了就按行加权 —— valid 上用 `ds.sample_weights()`（每个回合总权重
    相等）。训练时不给，就是每行等权，和 loss 的口径一致。

    ⚠ 分项 loss 用**行数**做权重（batch 均值 × n），而 `alive` / `can_kill` 的真实分母是
    活人数；两者只在"每个 batch 的活人比例都差不多"时严格相等。这是日志口径，够用；
    真要逐项精确，得让 `spatial_loss` 把分子分母也吐出来。
    """
    n = batch["ct_win"].shape[0]
    one = torch.ones(n, dtype=torch.float32, device=total.device)
    w = one if weights is None else weights.to(torch.float32)
    alive = batch["ctx_player_mask"].to(torch.float32)
    #: 逐人指标的分母：活人 × 行权重（死人那两个标签恒 False，不掩掉会让 acc 虚高）。
    mw = alive * w[:, None]
    corr = spatial_correct(out, batch)

    for k in ("win", "next_death", "next_kill"):
        hit = corr[k].to(torch.float32)
        store.add(f"acc_{k}", (hit * w).sum(), w.sum())
    for k in ("alive", "can_kill"):
        hit = corr[k].to(torch.float32)
        store.add(f"acc_{k}", (hit * mw).sum(), mw.sum())

    store.add("pred_win", ((out["logit_win"] > 0).to(torch.float32) * w).sum(), w.sum())
    for k in ("alive", "can_kill"):
        store.add(
            f"pred_{k}",
            ((out[f"logit_{k}"] > 0).to(torch.float32) * mw).sum(),
            mw.sum(),
        )
    for k in ("next_death", "next_kill"):
        hit = (out[f"logit_{k}"].argmax(-1) == NO_EVENT).to(torch.float32)
        store.add(f"pred_{k}", (hit * w).sum(), w.sum())

    # loss 按行数加权：batch 均值 × n 再除以 Σn，等价于"所有行的平均"。
    den = torch.as_tensor(float(n), dtype=torch.float32, device=total.device)
    store.add("loss", total.detach() * den, den)
    for k, v in parts.items():
        store.add(f"loss_{k}", v.detach() * den, den)


@torch.no_grad()
def evaluate(
    model: SpatialModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    sample_weights: torch.Tensor | None,
    loss_weights: dict[str, float],
    max_batches: int | None = None,
    prefix: str = "valid/",
) -> dict[str, float]:
    """一个 split 上的指标：`acc_*` 按**回合计权**，`loss*` 按行等权（和训练同口径）。

    `prefix` 只用来区分 `valid/*` 和 `test/*`（同一个函数，不给两个 split 各写一份）。
    """
    model.eval()
    store = MetricStore()
    pos = 0
    for i, batch in enumerate(tqdm(loader, desc="valid", unit="batch", leave=False,
                                   disable=None)):
        n = batch["ct_win"].shape[0]
        w = None
        if sample_weights is not None:
            w = sample_weights[pos:pos + n]
        pos += n
        batch = to_device(batch, device)
        with maybe_autocast(amp_dtype, device):
            out = model(batch)
            total, parts = spatial_loss(out, batch, loss_weights)
        add_batch(store, out, batch, total, parts, weights=w)
        if max_batches and i + 1 >= max_batches:
            break
    model.train()
    return store.ratio(prefix)


def evaluate_split(
    model: SpatialModel,
    root: str,
    split: str,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    loss_weights: dict[str, float],
    *,
    cache: int | None,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    run_dir: Path,
    max_batches: int | None = None,
) -> tuple[dict[str, float], str]:
    """训练结束后在 `test` 上跑一遍**最好**那个 checkpoint → `({test/*}, ckpt 文件名)`。

    `test` 只在最后用一次（每 epoch 都测就等于拿它调参）。`best.pt` 不存在（没开验证）时
    退回 `last.pt`，返回的第二个值就是实际用的那个文件名 —— 别让"这是 best 还是 last"
    靠猜。
    """
    ds = try_split(root, split, max_cached_files=cache)
    if ds is None:
        print(f"[spatial] {root}/{split} 是空的，跳过")
        return {}, ""
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    loader = make_loader(
        ds, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=False,
        pin_memory=pin_memory, buffer_rows=None,
    )
    weights = torch.from_numpy(ds.sample_weights()).to(device)
    metrics = evaluate(
        model, loader, device, amp_dtype, weights, loss_weights,
        max_batches=max_batches, prefix=f"{split}/",
    )
    return metrics, ckpt_path.name


def train(cfg: dict) -> dict[str, float]:
    """按配置训练，返回最好的那个 valid 指标（没有 valid 时是 `{"valid/loss": inf}`）。"""
    dc, tc = cfg.get("data", {}), cfg.get("train", {})
    mc = dict(cfg.get("model", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")            # TF32（fp32 那部分算子）
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- run 目录 / wandb ----
    name = tc.get("name") or time.strftime("%Y%m%d-%H%M%S")
    save_dir = Path(tc.get("save_dir", "runs/spatial"))
    run_dir = save_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    wb = None
    wcfg = tc.get("wandb") or {}
    if wcfg.get("enabled"):
        import wandb

        wb = wandb.init(
            project=wcfg.get("project", "cs-net-spatial"),
            name=name,
            config=cfg,
            mode=wcfg.get("mode", "online"),
            dir=str(run_dir),
        )

    # ---- 模型 ----
    model = SpatialModel(SpatialConfig(**mc)).to(device)
    loss_weights = loss_weights_of(tc.get("loss_weights"))
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[spatial] {n_param / 1e6:.2f} M 参数, run={run_dir}")
    print(
        f"[spatial] {model.cfg.layers} 层全连接自注意力（d={model.cfg.d_model}, "
        f"h={model.cfg.n_heads}），token = 4 个语境 + 10 个玩家 + P 个道具；"
        f"遮挡 bias {tuple(model.ob_bias.shape)}"
    )
    print(f"[spatial] loss 权重：{loss_weights}")

    # ---- 数据 ----
    cache = dc.get("max_cached_files")
    train_ds = SpatialDataset(
        dc["root"], split="train", max_cached_files=cache
    )
    valid_ds = try_split(dc["root"], "valid", max_cached_files=cache)
    print(
        f"[spatial] train: {len(train_ds.files)} 回合 / {len(train_ds):,} 行"
        f"（K 中位 {int(train_ds.counts.mean())}）"
        + (
            f"；valid: {len(valid_ds.files)} 回合 / {len(valid_ds):,} 行"
            if valid_ds is not None
            else f"；{dc['root']}/valid 是空的，跳过验证（只用 train loss）"
        )
    )

    batch_size = int(dc.get("batch_size", 512))
    accum = max(1, int(dc.get("grad_accum", 1)))
    workers = int(dc.get("num_workers", 8))
    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=bool(dc.get("pin_memory", True)),
        seed=dc.get("seed"),
        buffer_rows=dc.get("buffer_rows", 8192),
    )
    train_loader = make_loader(train_ds, shuffle=True, drop_last=True, **common)
    valid_loader = (
        make_loader(valid_ds, shuffle=False, drop_last=False, **common)
        if valid_ds is not None
        else None
    )
    #: valid 的逐行权重（每行 `1/K`，每个回合总权重相等）。只在评估时用；训练侧 24M 行
    #: 建这张表要 97 MB，没必要。
    sample_weights = (
        torch.from_numpy(valid_ds.sample_weights()).to(device)
        if valid_ds is not None
        else None
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise SystemExit(f"dataset 只有 {len(train_ds)} 行，batch_size={batch_size} 凑不出一批")
    epochs = int(tc.get("epochs", 1))
    total_steps = steps_per_epoch * epochs
    warmup = int(total_steps * float(tc.get("warmup_frac", 0.02)))

    # ---- 优化器 / 调度 / AMP ----
    optimizer = torch.optim.AdamW(
        make_param_groups(model, float(tc.get("weight_decay", 0.05))),
        lr=float(tc.get("lr", 3e-4)),
        betas=tuple(tc.get("betas", (0.9, 0.95))),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_scale(step, total_steps, warmup, float(tc.get("min_lr_frac", 0.1))),
    )
    amp = str(tc.get("amp", "bf16")).lower()
    if amp not in ("bf16", "fp16", "off"):
        raise SystemExit(f"train.amp 只能是 bf16 / fp16 / off，收到 {amp!r}")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[amp]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    # ---- 续训 ----
    start_epoch, global_step = 0, 0
    best = {"valid/loss": float("inf")}
    if tc.get("resume"):
        ckpt = torch.load(tc["resume"], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, global_step = int(ckpt["epoch"]), int(ckpt["step"])
        # 把历史最好的分数接过来，续训时别用刚开始的 inf 把 `best.pt` 覆盖掉。
        best = ckpt.get("best", best)
        print(f"[spatial] 从 {tc['resume']} 续训：epoch {start_epoch}, step {global_step}")

    # ---- 训练 ----
    max_steps = tc.get("max_steps")
    grad_clip = float(tc.get("grad_clip", 1.0))
    log_every = max(1, int(tc.get("log_every", 50)))
    print_every = max(1, int(tc.get("print_every", 200)))
    eval_every = max(1, int(tc.get("eval_every", 1)))
    eval_max_batches = tc.get("eval_max_batches")

    for epoch in range(start_epoch, epochs):
        model.train()
        window, epoch_store = MetricStore(), MetricStore()
        skipped = 0
        rows = 0
        t_epoch = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pbar = tqdm(
            total=steps_per_epoch, desc=f"epoch {epoch + 1}/{epochs}", unit="step",
            disable=None,
        )
        print(
            f"[spatial] epoch {epoch + 1}/{epochs} 开始：{steps_per_epoch} 步"
            f"（B={batch_size} × accum {accum}）× {epochs - epoch} 个 epoch 之一"
            f"，共 {total_steps:,} 步"
        )
        optimizer.zero_grad(set_to_none=True)
        n_micro = 0
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            rows += batch["ct_win"].shape[0]
            n_micro += 1
            with maybe_autocast(amp_dtype, device):
                out = model(batch)
                loss, parts = spatial_loss(out, batch, loss_weights)

            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.update(1)
                continue

            scaler.scale(loss / accum).backward()

            # 累积满 `accum` 个微批、或这个 epoch 就剩这几个，就更新一次参数。
            last_micro = (i + 1) == steps_per_epoch
            step_now = (i + 1) % accum == 0 or last_micro
            grad_norm = None
            if step_now:
                scaler.unscale_(optimizer)
                grad_norm = float(nn_utils.clip_grad_norm_(model.parameters(), grad_clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            with torch.no_grad():
                add_batch(window, out, batch, loss, parts)
                add_batch(epoch_store, out, batch, loss, parts)
            pbar.update(1)

            if step_now and global_step % log_every == 0:
                logs = {"train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm, "epoch": epoch}
                logs |= window.ratio("train/")
                if wb is not None:
                    wb.log(logs, step=global_step)
                window.clear()

            if step_now and global_step % print_every == 0:
                # 这几个数是从 epoch 开头累加到现在（不是最近 N 步），所以变化很慢 ——
                # 想看"最近发生了什么"就看 tqdm 的 postfix 或者把 print_every 调小。
                m = epoch_store.ratio()
                wall = max(1e-9, time.perf_counter() - t_epoch)
                eta = (steps_per_epoch - i - 1) / max(1e-9, (i + 1) / wall)
                print(
                    f"  step {global_step:,}（epoch 累计）  loss {m['loss']:.4f} "
                    f"(win {m['loss_win']:.3f} alive {m['loss_alive']:.3f} "
                    f"can {m['loss_can_kill']:.3f} nd {m['loss_next_death']:.3f} "
                    f"nk {m['loss_next_kill']:.3f})  "
                    f"acc win {m['acc_win']:.3f} alive {m['acc_alive']:.3f} "
                    f"can {m['acc_can_kill']:.3f} nd {m['acc_next_death']:.3f} "
                    f"nk {m['acc_next_kill']:.3f}  "
                    f"pred nd={m['pred_next_death']:.3f} nk={m['pred_next_kill']:.3f}  "
                    f"{rows / wall:,.0f} 行/s  {optimizer.param_groups[0]['lr']:.1e}  "
                    f"eta {eta / 60:.0f} min"
                )
                pbar.set_postfix(
                    loss=f"{m['loss']:.4f}", nd=f"{m['acc_next_death']:.3f}",
                    nk=f"{m['acc_next_kill']:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

            if max_steps and global_step >= int(max_steps):
                break

        pbar.close()
        # 进度条被关掉时（stderr 不是终端）就靠这几行看进度。
        wall = max(1e-9, time.perf_counter() - t_epoch)
        m = epoch_store.ratio()
        perf = f"{wall / max(1, n_micro) * 1e3:.0f} ms/步, {rows / wall:,.0f} 行/s"
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**30
            perf += f", 峰值显存 {peak:.1f} GiB"
        print(
            f"[spatial] epoch {epoch + 1} train: loss {m['loss']:.4f} "
            f"acc win {m['acc_win']:.3f} alive {m['acc_alive']:.3f} "
            f"can {m['acc_can_kill']:.3f} nd {m['acc_next_death']:.3f} "
            f"nk {m['acc_next_kill']:.3f} | {perf}"
        )
        if skipped:
            print(f"[spatial] epoch {epoch + 1}: 跳过 {skipped} 个非有限 loss 的微批")

        # ---- 验证 / 存档 ----
        metrics: dict[str, float] = {}
        if valid_loader is not None and (epoch + 1) % eval_every == 0:
            metrics = evaluate(
                model, valid_loader, device, amp_dtype, sample_weights,
                loss_weights, max_batches=eval_max_batches,
            )
            print(
                f"[spatial] epoch {epoch + 1} valid（按回合计权）: "
                f"loss {metrics['valid/loss']:.4f} "
                f"acc win {metrics['valid/acc_win']:.3f} "
                f"alive {metrics['valid/acc_alive']:.3f} "
                f"can {metrics['valid/acc_can_kill']:.3f} "
                f"nd {metrics['valid/acc_next_death']:.3f} "
                f"nk {metrics['valid/acc_next_kill']:.3f}"
            )
            if wb is not None:
                wb.log(metrics, step=global_step)

        # 存档：`last.pt` 每个 epoch 覆盖；`best.pt` 只在 valid loss 变好时写。
        is_best = bool(metrics) and metrics["valid/loss"] < best.get(
            "valid/loss", float("inf")
        )
        if is_best:
            best = metrics
        ckpt = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            epoch=epoch + 1,
            step=global_step,
            best=best,
            cfg=cfg,
        )
        torch.save(ckpt, run_dir / "last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "best.pt")

        if max_steps and global_step >= int(max_steps):
            print(f"[spatial] 到 max_steps={max_steps}，停")
            break

    # ---- test（可选）：用最好的那个 checkpoint 跑一遍没见过的那 16,017 个回合 ----
    if tc.get("eval_test"):
        test_metrics, ckpt_name = evaluate_split(
            model, dc["root"], "test", device, amp_dtype, loss_weights,
            cache=cache, batch_size=batch_size, workers=workers,
            pin_memory=bool(dc.get("pin_memory", True)),
            run_dir=run_dir, max_batches=eval_max_batches,
        )
        if test_metrics:
            print(
                f"[spatial] test（按回合计权，ckpt={ckpt_name}）: "
                f"loss {test_metrics['test/loss']:.4f} "
                f"acc win {test_metrics['test/acc_win']:.3f} "
                f"alive {test_metrics['test/acc_alive']:.3f} "
                f"can {test_metrics['test/acc_can_kill']:.3f} "
                f"nd {test_metrics['test/acc_next_death']:.3f} "
                f"nk {test_metrics['test/acc_next_kill']:.3f}"
            )
            if wb is not None:
                wb.log(test_metrics, step=global_step)

    if wb is not None:
        wb.finish()
    if best.get("valid/loss", float("inf")) < float("inf"):
        print(f"[spatial] best valid: {best}")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="训练 spatial_prediction（单帧 5 个标签）")
    ap.add_argument(
        "-c", "--config", default="training/spatial_prediction/config.yaml",
        help="配置文件",
    )
    ap.add_argument("--set", action="append", nargs="+", default=[], metavar="KEY=VALUE",
                    help="覆盖配置，可一次给多个（例：--set train.lr=1e-4 data.batch_size=256）")
    ap.add_argument("--resume", default=None, help="从 last.pt / best.pt 续训")
    ap.add_argument("--no-wandb", action="store_true", help="不连 wandb")
    ap.add_argument("--dry-run", action="store_true", help="1 epoch / 6 步的冒烟测试")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for spec in [s for group in args.set for s in group]:
        apply_override(cfg, spec)
    cfg.setdefault("train", {}).setdefault("wandb", {})
    if args.no_wandb:
        cfg["train"]["wandb"]["enabled"] = False
    if args.resume:
        cfg["train"]["resume"] = args.resume
    if args.dry_run:
        cfg.setdefault("data", {}).update(
            batch_size=8, grad_accum=1, num_workers=0, buffer_rows=256
        )
        cfg["train"].update(
            epochs=1, max_steps=6, name="dry-run", eval_max_batches=2,
            print_every=2, log_every=2,
        )
        cfg["train"]["wandb"]["enabled"] = False
    train(cfg)


if __name__ == "__main__":
    main()
