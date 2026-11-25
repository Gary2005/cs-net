"""aim 训练入口 —— 超参全在 `config.yaml` 里，代码里不写死。

    # 默认配置（`training/aim/config.yaml`）
    python -m training.aim.train -c training/aim/config.yaml

    # 临时改一项（按 YAML 解析：`1e-4` / `true` / `[6,8]` 都行）
    python -m training.aim.train -c training/aim/config.yaml --set train.lr=1e-4

    # 续训 / 冒烟
    python -m training.aim.train -c training/aim/config.yaml --resume runs/aim/<name>/last.pt
    python -m training.aim.train -c training/aim/config.yaml --dry-run

## 配置怎么读

`config.yaml` 分 `data` / `model` / `train` 三段：`model` 那段直接 unpack 成
`AimConfig`（字段名就是 `model.py` 里的那些），`data` / `train` 段在这里用。命令行
`--set a.b=c` 在 `yaml.safe_load` 之后覆盖，所以值的写法和配置文件里一样。

日志走 wandb（`train/loss`、`train/lr`、`train/grad_norm`、`valid/*`）+ tqdm，
每个 run 在 `train.save_dir/<name>` 下留一份**实际生效的配置** `config.yaml` 和
`last.pt` /（有 valid 时）`best.pt`。`name` 留空就用时间戳，不会覆盖上一次。

## 数据

样本是逐条从 npz 现读的（`np.load` 只解压用到的成员，一条 `T=257` 折成 tensor 约
**0.38 MB**），200k 条一个 epoch ≈ **76 GB** 顺序读，所以 `num_workers` 给到 10 左右、
`pin_memory: true`。真正贵的是 zlib 解压那一下（在 worker 里，不占显卡）。

batch 就是 `shuffle=True` 直接切，**没有长度分桶**：`AimDataset` 的随机截断
（`T' ~ U{6, 257}`）在 `__getitem__` 里现摇，训练侧拿不到"这批要多长"的先验，扫一遍
npz 量出来的长度恒是 257（原始窗口是定长），分桶等于没分。代价是 batch 内 `Tmax`
基本贴着 257、padding 浪费 ≈ 48%，省下来的是数据量不是时间。

想把那半天真省掉，得把摇 `T'` 的位置从 `__getitem__` 挪到采样器（按桶摇、桶内排序），
那是以后要加的东西，不是被删掉的那个 —— 被删的版本只按**静态**长度分桶，跟随机截断
天然冲突。

## 验证：valid 也开随机截断（和训练同一个分布）

`try_split` 不传 `augment=False`，所以 valid 和 train 走**同一条** `__getitem__`：每条
样本每轮从 `[6, T]` 里摇一个 `T'`、只留最后 `T'` 帧、`seconds` 对零。这是刻意的 ——
固定 4 秒窗口只是训练分布里 `T' = T` 那一个点（252 个里的 1 个），拿它当 valid 等于
换个分布评，而且偏乐观（历史最长）。截断是真的会改输入的：`k` 变小（`time_emb[k]` 变）、
因果注意力能回看的历史也短了，所以不是"同一批输入换个平均方式"。

`best.pt` 因此也按这个 `valid/loss` 选，和训练目标同一个口径。代价是 valid 的 `T'` 每轮
重摇，同一个 ckpt 评两次的数会差一点 —— 训练那边每轮也重摇，所以这是同一个口径下的
抽样噪声，不是新引入的偏差。嫌抖就多评几次取平均（或把 valid 做大），别抠小数点后第三位。

## 一个 epoch 大概多久（24 G 卡 + `d_model: 1024` 的 67 M 参数 + 23 万条）

**已实测**（RTX 3090 24 G，bf16，`d_model=1024` / `rel_layers=2` / `antisym: true` /
`T=257`，假 batch、不含 I/O）：B=8 → 3.8 GiB / 119 ms，B=16 → 6.5 GiB / 216 ms，
B=32 → 12.0 GiB / 414 ms，B=64 → 22.5 GiB / **5.3 s**。显存随 batch **线性**涨（帧内那一头
是 `B×T_real` 行一起算的），24 G 的上限就在 B=32 附近。每 epoch ≈ **1.5 小时**，和 batch
几乎无关，所以别靠加 batch 换速度，要更大的有效 batch 就抬 `grad_accum`。20 epoch ≈ 30 h。

⚠ 上面这张表是 `rel_layers=2` 时测的，本配置用 3（帧内那一路 ×4/3）。按算式总算力 +17%：
每 epoch ≈ **1.75 小时**、20 epoch ≈ **35 小时**、帧内占全模型约 **57%**。这是算出来的，
**没有重测** —— 第一次跑请照旧先量一次 ms/微步。

（同样这份代码 `d_model=640` 时是 26 M / B=32 → 204 ms / 25 min 每 epoch；1024 大约是它的
2.5 倍算力、2.6 倍参数。）

⚠ B=64 起在 Windows 上**不会报 OOM**：驱动把它换成共享内存，单步从 0.4 s 变成 **5.3 s**，
表现是"卡住不动"。看到 ms/微步 一个数量级地涨，先查 batch。

算力的算式（下面这个仍然成立，只是已经被实测覆盖）：

    帧内     T × 2 个查询 × (9 活人 + P 道具 + 3 个小项) × (1 + rel_layers) d²
    时间     T × temporal_layers × 12 d²

帧内那一头是**每帧两个查询**：一帧只算 A、B 两个人，每人 `9 个活人 + P 个道具 +
map / global / 自己` 那么多次 MLP，一次 `(1 + rel_layers) d²`（本配置 `rel_layers=3`，
即 `4 d²`）。所以它比旧的 2 层空间（`S×12d²`，`S ≈ 15~22`）便宜三倍左右，也跟着
**不再随 batch 里的道具数剧烈变化**：`P` 只影响"求和/池化项多几项"，不影响 token 数。
`rel_layers` 是帧内唯一的深度旋钮，在 `rel_layers=3` 的基数上每加一层大约 +14% 总算力
（见 `model.py` 的 `rel_layers` 一段）。
`P` 取 batch 内有道具槽位的最大数（槽位上限 32；实测均值 3.7、p95 9、max 12），一般落在 10 左右；
它是帧内成本里唯一随数据变的量。第一次跑先用 `--set data.batch_size=32` 量出 ms/微步和
峰值显存，再决定要不要抬。

真 OOM 就往下调 `batch_size`、抬 `grad_accum`，有效 batch 别变 —— 模型里没有对 batch 的
假设（帧内那一头把 `(B,T)` 摊平成 `B·T_valid` 一次算完，`B` 多大都行）。

I/O：每 epoch 从盘上读 200k × ~0.38 MB ≈ **76 GB**（旧版带视角图时是 400 GB），NVMe 上
`num_workers: 10` 绰绰有余；真正吃 CPU 的是 worker 里的 zlib 解压。
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

from training.aim.dataset import AimDataset, collate_aim
from training.aim.model import AimConfig, AimModel, aim_loss

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


def make_loader(
    ds: AimDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """训练 `shuffle=True` + `drop_last`（最后那批不满的丢掉，每个 epoch 步数固定），
    评估 `shuffle=False` 且保留尾批（valid 的样本一条都别丢）。"""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=collate_aim,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def try_split(root: str, split: str) -> AimDataset | None:
    """取一个 split，没有就返回 `None`（现在盘上 valid/test 还是空的）。

    **和训练同一个分布**：不传 `augment=False`，所以 valid 也做随机截断（`T' ~ U{6, T}`，
    每轮重摇）。固定 4 秒窗口只是训练分布里的一个点，拿它当 valid 等于换个分布评 ——
    见 `dataset.py` 模块 docstring 第 3 条。

    **固定 A/B 方向**（`swap=False`，永远是磁盘上的"1 = 击杀者"）：`antisym` 开着时
    两个方向的 logit 精确互为相反数，固定方向的 acc 和摇出来的完全一样；关着时摇方向
    会多一份噪声，而 acc 对这个对称性并不敏感。
    """
    try:
        return AimDataset(root, split=split, swap=False)
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


@torch.no_grad()
def batch_stats(
    probs: torch.Tensor, mask: torch.Tensor, label: torch.Tensor
) -> dict[str, float]:
    """一个 batch 的指标，**每条样本权重相同**。

    `aim_loss` 现在是逐点等权（长窗口贡献更多预测点），这里的 `acc` 仍是"每条样本先在
    窗口内平均"——两者口径不同，好处是 `acc` 不会被长窗口带偏、仍能和旧 run 直接比。

    `acc` 是"每条样本先在 `T` 个预测点内部平均、再对 batch 平均"，所以长轨迹
    不会因为有更多预测点而占更大权重。`acc_first` = 只看完第 0 帧那一列，
    `acc_last` = 看完整个窗口那一列 —— 两者差多少直接说明模型有没有用上这段轨迹。
    `prob_pos` / `prob_neg` 是正负样本的平均概率，都该往 0 / 1 两头跑。
    """
    pos = label.to(probs.dtype)[:, None].expand_as(probs) > 0.5
    neg = ~pos
    n = mask.sum(1).clamp_min(1)
    last = probs[torch.arange(probs.shape[0], device=probs.device), mask.sum(1) - 1]
    # 每条样本的标签只有一个，所以它要么全是正样本点、要么全是负的：按"有这类点的
    # 样本"取平均，不然正样本的 batch 里 `prob_neg` 会除 0 变 nan。
    p_pos, n_pos = (probs * (pos & mask)).sum(1), (pos & mask).sum(1)
    p_neg, n_neg = (probs * (neg & mask)).sum(1), (neg & mask).sum(1)
    p_pos, p_neg = p_pos[n_pos > 0] / n_pos[n_pos > 0], p_neg[n_neg > 0] / n_neg[n_neg > 0]
    return {
        "acc": ((((probs > 0.5) == pos) & mask).sum(1) / n).mean().item(),
        "acc_first": ((probs[:, 0] > 0.5) == pos[:, 0]).float().mean().item(),
        "acc_last": ((last > 0.5) == pos[:, 0]).float().mean().item(),
        "prob_pos": p_pos.mean().item() if p_pos.numel() else float("nan"),
        "prob_neg": p_neg.mean().item() if p_neg.numel() else float("nan"),
    }


@torch.no_grad()
def evaluate(
    model: AimModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    """valid/test 上的指标（`valid/loss` + `batch_stats` 那几项）。

    只负责**跑一遍 loader**。loader 背后是随机截断过的样本（`try_split` 不再传
    `augment=False`），所以这里的 `acc_first` / `acc_last` 是"随机长度窗口的第 0 列 /
    击杀那一列"，和训练时 `train/acc_first` / `train/acc_last` 同一个口径。
    另外多记一个 `valid/ticks`（平均可见帧数）：截断生效时它应该 ≈ `E[T'] ≈ 131`，
    而不是 257 —— 它是"valid 真的在截断"最直接的一个数。
    """
    model.eval()
    sums: dict[str, float] = {}
    cnt: dict[str, int] = {}
    ticks = 0.0
    rows = 0
    for batch in tqdm(loader, desc="valid", unit="batch", leave=False, disable=None):
        batch = to_device(batch, device)
        with maybe_autocast(amp_dtype, device):
            out = model(batch)
        mask = out["pred_mask"]
        ticks += float(mask.sum().item())
        rows += int(mask.shape[0])
        stats = batch_stats(out["probs"].float(), mask, batch["label"])
        stats["loss"] = float(aim_loss(out["logits"], batch["label"], mask))
        for k, v in stats.items():
            if v == v:                      # nan 跳过：评估固定方向后 `prob_neg` 恒为 nan
                sums[k] = sums.get(k, 0.0) + v
                cnt[k] = cnt.get(k, 0) + 1

    model.train()
    metrics = {f"valid/{k}": v / max(1, cnt[k]) for k, v in sums.items()}
    metrics["valid/ticks"] = ticks / max(1, rows)
    return metrics


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
    save_dir = Path(tc.get("save_dir", "runs/aim"))
    run_dir = save_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    wb = None
    wcfg = tc.get("wandb") or {}
    if wcfg.get("enabled"):
        import wandb

        wb = wandb.init(
            project=wcfg.get("project", "cs-net-aim"),
            name=name,
            config=cfg,
            mode=wcfg.get("mode", "online"),
            dir=str(run_dir),
        )

    # ---- 模型 / 数据 ----
    model = AimModel(AimConfig(**mc)).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[aim] {n_param / 1e6:.3f} M 参数, run={run_dir}")
    if model.cfg.antisym:
        print("[aim] antisym 开：logit = w·h(A,B) − w·h(B,A)，两个方向精确反对称")
    pool_desc = {
        "sum": "聚合 = 求和（每条关系等权）",
        "attn": "聚合 = 注意力池化（打分头零初始化，起点等同求和）",
    }[model.cfg.pool]
    print(
        f"[aim] 帧内：proj + 四项关系 MLP（map / global / 9 个活人 / 双向遮挡 / 道具，"
        f"各 {model.cfg.rel_layers} 层带残差），{pool_desc}；"
        f"时间：{model.cfg.temporal_layers} 层因果 transformer"
    )

    # A/B 对调喂的是同一个二选一的另一个方向。`antisym` 打开之后两个方向的梯度完全
    # 相同（`dataset.py` 模块 docstring 末尾有推导），摇它只是把同一件事说两遍 ——
    # 步数不变，省的是那份冗余，不是算力。
    swap_on = bool(dc.get("swap_ab", True))
    if not (model.cfg.antisym or swap_on):
        raise SystemExit(
            "[aim] `data.swap_ab: false` 必须配 `model.antisym: true`。\n"
            "    标签恒为 True 时，单方向读出可以靠一个常数 logit 把 loss 压到 0"
            "（`acc` 还会显示 1.0），这是退化解；\n"
            "    只有 z = w·h(A,B) − w·h(B,A) 这种两个方向相减的读出才排得掉它。"
        )
    if model.cfg.antisym and swap_on:
        print("[aim] ⚠ antisym 和 swap_ab 同时开着：对调只是在把同一件事说两遍，"
              "建议 data.swap_ab=false（步数不用改）")
    train_ds = AimDataset(dc["root"], split="train", swap=swap_on)
    valid_ds = try_split(dc["root"], "valid")
    if valid_ds is None:
        print(f"[aim] {dc['root']}/valid 是空的，跳过验证（只用 train loss）")

    batch_size = int(dc.get("batch_size", 32))
    accum = max(1, int(dc.get("grad_accum", 1)))
    common = dict(
        num_workers=int(dc.get("num_workers", 0)),
        pin_memory=bool(dc.get("pin_memory", True)),
    )
    train_loader = make_loader(
        train_ds, batch_size=batch_size, shuffle=True, **common
    )
    valid_loader = (
        make_loader(valid_ds, batch_size=batch_size, shuffle=False, **common)
        if valid_ds is not None
        else None
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise SystemExit(f"dataset 只有 {len(train_ds)} 条，batch_size={batch_size} 凑不出一批")
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
        state = ckpt["model"]
        # `sum` → `attn` 是**允许**跨档续训的：`attn` 只多出 `pool_*.{norm,score}`，而分数头
        # 零初始化时两档逐位相同（见 `model.RelPool`）。反方向不行（多了 key），
        # 其他结构超参变了也不行 —— 那些都必须重训。
        new_keys = {
            k for k in model.state_dict() if k.startswith(("pool_pair.", "pool_prop."))
        }
        tolerant = bool(new_keys) and new_keys.isdisjoint(state)
        try:
            model.load_state_dict(state, strict=not tolerant)
        except RuntimeError as e:
            raise SystemExit(
                "[aim] ckpt 和当前结构对不上（改过 rel_layers / temporal_layers / pool 等"
                "结构超参就会这样；从 `pool: attn` 退回 `sum` 也不行，反方向可以）。\n"
                f"    {e}"
            ) from e
        if tolerant:
            print(
                f"[aim] 续训：ckpt 是 `sum` 档，本模型是 `{model.cfg.pool}` 档，"
                f"多出的 {len(new_keys)} 个 key 用默认值（分数头为 0 = 起点等同 `sum`）"
            )
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, global_step = int(ckpt["epoch"]), int(ckpt["step"])
        # 把历史最好的分数接过来，续训时别用刚开始的 inf 把 `best.pt` 覆盖掉。
        best = ckpt.get("best", best)
        print(f"[aim] 从 {tc['resume']} 续训：epoch {start_epoch}, step {global_step}")

    # ---- 训练 ----
    max_steps = tc.get("max_steps")
    grad_clip = float(tc.get("grad_clip", 1.0))
    log_every = max(1, int(tc.get("log_every", 20)))
    eval_every = max(1, int(tc.get("eval_every", 1)))

    for epoch in range(start_epoch, epochs):
        model.train()
        run = dict(loss=0.0, rows=0, skipped=0)
        # 一个 log 窗口内的指标累计（累加 + 计数，nan 跳过：关掉 swap_ab 之后
        # `prob_neg` 恒为 nan）。`acc_last` 是这里最该看的那个数，见 `batch_stats`。
        win_sum: dict[str, float] = {}
        win_cnt: dict[str, int] = {}
        cum = dict(loss=0.0, acc=0.0, rows=0)  # 整个 epoch 的累计，给最后那行 print 用
        t_epoch = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pbar = tqdm(
            total=steps_per_epoch, desc=f"epoch {epoch + 1}/{epochs}", unit="step",
            disable=None,
        )
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            with maybe_autocast(amp_dtype, device):
                out = model(batch)
                loss = aim_loss(out["logits"], batch["label"], out["pred_mask"])

            if not torch.isfinite(loss):
                run["skipped"] += 1
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

            # tqdm / wandb 上的 loss、acc 都按**每条样本权重相同**累计，和 `aim_loss`
            # 的口径一致（长轨迹不因为预测点多而被放大）。
            with torch.no_grad():
                stats = batch_stats(out["probs"].float(), out["pred_mask"], batch["label"])
                run["loss"] += float(loss)
                run["rows"] += 1
                cum["loss"] += float(loss)
                cum["acc"] += stats["acc"]
                cum["rows"] += 1
                for k, v in stats.items():
                    if v == v:                          # nan 跳过
                        win_sum[k] = win_sum.get(k, 0.0) + v
                        win_cnt[k] = win_cnt.get(k, 0) + 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{run['loss'] / max(1, run['rows']):.4f}",
                acc=f"{win_sum.get('acc', 0.0) / max(1, win_cnt.get('acc', 0)):.3f}",
                last=f"{win_sum.get('acc_last', 0.0) / max(1, win_cnt.get('acc_last', 0)):.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if step_now and global_step % log_every == 0:
                logs = {
                    "train/loss": run["loss"] / max(1, run["rows"]),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                    "epoch": epoch,
                }
                # `train/acc` 是把最多 257 个 tick 摊平的平均，而信息全在末尾（实测 A 看见
                # B：首帧 3.6% → 末帧 86.0%），所以它天生贴着 0.5、看不出进展。
                # `acc_last` 才是和 `valid/acc_last` 同一个口径、真正要看的那个数；
                # `acc_first` 一起打出来，两者之差说明模型有没有用上这段轨迹。
                logs |= {f"train/{k}": win_sum[k] / max(1, win_cnt[k]) for k in win_sum}
                if wb is not None:
                    wb.log(logs, step=global_step)
                # 窗口内的累计值清零，别让 tqdm 上的数字是整个 epoch 的平均。
                run.update(loss=0.0, rows=0)
                win_sum.clear()
                win_cnt.clear()

            if max_steps and global_step >= int(max_steps):
                break

        pbar.close()
        # 进度条被关掉时（stderr 不是终端）就靠这几行看进度，每个 epoch 一条。
        # 后两个数是**调参的唯一依据**：显存不够先降 `batch_size`、抬 `grad_accum`；
        # 太慢按这个顺序砍：`rel_layers`（1 就够跑）、`temporal_layers`、`d_model`。
        n_step = max(1, cum["rows"])
        wall = max(1e-9, time.perf_counter() - t_epoch)
        samples = cum["rows"] * batch_size
        perf = (
            f"{wall / n_step * 1e3:.0f} ms/微步, {samples / wall:.0f} 样本/s"
        )
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**30
            perf += f", 峰值显存 {peak:.1f} GiB"
        print(
            f"[aim] epoch {epoch + 1} train: loss {cum['loss'] / max(1, cum['rows']):.4f} "
            f"acc {cum['acc'] / max(1, cum['rows']):.3f} | {perf}"
        )
        if run["skipped"]:
            print(f"[aim] epoch {epoch + 1}: 跳过 {run['skipped']} 个非有限 loss 的微批")

        # ---- 验证 / 存档 ----
        metrics = {}
        if valid_loader is not None and (epoch + 1) % eval_every == 0:
            metrics = evaluate(model, valid_loader, device, amp_dtype)
            print(
                f"[aim] epoch {epoch + 1} valid: loss {metrics['valid/loss']:.4f} "
                f"acc {metrics['valid/acc']:.3f} "
                f"(first {metrics['valid/acc_first']:.3f} / "
                f"last {metrics['valid/acc_last']:.3f}, "
                f"平均 {metrics['valid/ticks']:.0f} 帧)"
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
            print(f"[aim] 到 max_steps={max_steps}，停")
            break

    if wb is not None:
        wb.finish()
    if best.get("valid/loss", float("inf")) < float("inf"):
        print(f"[aim] best valid: {best}")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="训练 aim（P(A 击杀 B)）")
    ap.add_argument("-c", "--config", default="training/aim/config.yaml", help="配置文件")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖配置，可重复（例：--set train.lr=1e-4）")
    ap.add_argument("--resume", default=None, help="从 last.pt / best.pt 续训")
    ap.add_argument("--no-wandb", action="store_true", help="不连 wandb")
    ap.add_argument("--dry-run", action="store_true", help="1 epoch / 4 步的冒烟测试")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for spec in args.set:
        apply_override(cfg, spec)
    cfg.setdefault("train", {}).setdefault("wandb", {})
    if args.no_wandb:
        cfg["train"]["wandb"]["enabled"] = False
    if args.resume:
        cfg["train"]["resume"] = args.resume
    if args.dry_run:
        cfg.setdefault("data", {}).update(batch_size=4, grad_accum=1, num_workers=0)
        cfg["train"].update(epochs=1, max_steps=4, name="dry-run")
        cfg["train"]["wandb"]["enabled"] = False
    train(cfg)


if __name__ == "__main__":
    main()
