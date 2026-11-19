"""aim 任务的 torch 数据集：把 `scripts/training_data` 写出来的 npz 变成训练 batch。

    from training.aim.dataset import AimDataset, collate_aim

    ds = AimDataset("data/demos/prepared/aim", split="train")
    dl = DataLoader(ds, batch_size=32, collate_fn=collate_aim, num_workers=8)

目录约定（`scripts/training_data/prepare.py` 的输出）：`root/train`、`root/valid`、
`root/test` 三个子目录，样本是**一次击杀一个文件**的 npz。`root` 也可以直接给一个装满
npz 的目录（没有 `train/` 子目录时按"整个目录就是一个 split"处理）。

## 一条样本 = 「窗口内每一帧的全场状态」+ 一个 bool 标签

磁盘上的字段（详细语义见 `scripts/training_data/tasks/aim.py` 的模块 docstring）：

    ctx_player        (T, 10, 20) float32  逐帧 10 个玩家的状态，列序 = AIM_FEATURES
    ctx_player_mask   (T, 10)     int8     0 死 / 1 A(击杀者) / 2 B(受害者) / 3 其他存活
    ctx_player_weapon (T, 10)     int8     逐帧武器全局下标，-1 = 无武器或已死
    obstruction       (T, 10, 10) int8     i 看 j：0 通 / 1 烟 / 2 墙（有方向）
    ctx_projectile    (T, 32, 4)  float32  逐帧道具 x y z t，全 [0,1]
    ctx_projectile_id (T, 32)     int8     -1 = 空槽
    global_feat       (T, 5)      float32  逐帧 [炸弹 x y z, 安放秒数/50, 回合秒数/200]
    seconds           (T,)        float32  **可见窗口**第一帧起算的秒数，第一个恒为 0.0。
                                           未截断时就是磁盘上那 4 秒；训练截断后可见窗口
                                           是它的后 T' 帧，被对零成 `0..(T'-1)/64` ——
                                           对零正是这套增强的目的，见下文
    map_id            ()          int8
    label             ()          bool     True = A 击杀了 B

**`ctx_player_mask` 里的 1 / 2 就是标签本身**（1 = 击杀者，2 = 受害者）。模型不需要它：
`collate_aim` 把"谁是 A、谁是 B"抽成 `pair (B,2)` 的**索引**，mask 塌成 0 死 / 1 活。
好处是标签不再躺在特征通道里，而且帧内那一头对 10 个玩家完全一视同仁（细节见
`model.py` 的模块 docstring）。磁盘格式没变，不用重做数据。

**所有逐帧字段，没有例外**（早先的 `global_feat` 是首帧快照，已经改成 `(T,5)`）。

**A 和 B 是谁只由 `ctx_player_mask` 的 1 / 2 决定**，不再有 `atk` / `tgt` 两条独立轨迹。
好处是 10 个人天然在同一个张量里（"谁看得见谁""谁在旁边"这类关系才写得出来），
代价是模型必须自己从 mask 里把人挑出来。

## A/B 随机对调

磁盘上 `mask == 1` 永远是击杀者、`2` 永远是受害者，所以 100% 是正样本。`__getitem__`
**每次均匀随机**决定方向，对调就是把 mask 里的 1 和 2 互换：

    未对调（label=True ）  A = 击杀者   B = 受害者
    对调  （label=False）  A = 受害者   B = 击杀者

只有 mask 需要换：`ctx_player` / `ctx_player_weapon` / `obstruction` / `ctx_projectile`
都是**按玩家下标**存的绝对量，和 A/B 谁是谁无关（`obstruction[i][j]` 是 i 看 j，
对调之后仍然成立）。这是新布局比旧版（要同时换 `atk`/`tgt`、`atk_view`/`tgt_view`、
`atk_weapon`/`tgt_weapon`）干净的地方。

注意这是**同一条击杀的两个方向**，不是两份独立数据，别据此高估有效样本量。

⚠ 它**不去偏**：标签边际被摇成 50/50，但"特征 ↔ 标签"的条件相关一点没动（交换后
`P(下标小的那个是 A | 标签=1)` 仍然不等于 `| 标签=0`）。想让偏斜真的没用，得靠划分
（`prepare.demo_split` 按 demo 整场切，换个 demo 槽位就换了个人）。

`AimDataset(swap=False)` 可以关掉它，**但必须同时 `model.antisym: true`**：标签恒为 True
时，单方向的读出可以靠一个常数 logit 把 loss 压到 0（`acc` 还显示 1.0），而
`z = w·h(A,B) − w·h(B,A)` 两个方向相减正好把这个退化解排掉（`train.py` 里会拦）。

## 训练增强：随机截断（`AimDataset(augment=True)`，默认开）

`__getitem__` 读出整条 4 秒窗口后，再从 `[min_ticks, T]` 里**均匀**取一个 `T'`
（默认 `min_ticks=6`），**保留最后 `T'` 帧**、把前面的丢掉，并把 `seconds` 对零：

    T' = 6    ──────────────────────────────────▶  只看最后 78 ms
    T' = 257                                         整段 4 秒
    两者在模型眼里都从 `k = 0` 开始

## 为什么时间要一起对零（这套增强的真正动机）

窗口的终点固定在受害者最后一个活帧，也就是击杀那一刻。所以只要 `seconds` 是**绝对**
时间，模型就永远知道击杀发生在哪：最后一帧恒是 `k = 256`，`256 - k` 就是"离击杀还有
几个 tick"。而因果注意力下模型**看不到自己序列的总长**（padding 在末尾、`frame_mask`
也没喂进去），`k` 是它唯一能判断"我是不是快看到结局了"的依据 —— 绝对时间正好把这条
信息喂满。固定 4 秒窗口的害处就在这里：模型不需要懂战局，它只要知道"现在是第 256
帧"就该往"要死了"那边调概率。

对零之后 `k` 只剩"我已经看了多少帧"这一个含义：一个 6 帧样本的第 0 帧和一个 257 帧
样本的第 0 帧在模型眼里完全一样，**击杀在哪一帧变成隐变量**，模型只能靠画面里的
动势（谁在瞄谁、谁有枪线）去判断，这正是要的东西。

代价：模型失去"离结局多久"这条校准信息（早帧本来就更该不确定），而且
`acc_first` / `acc_last` 的含义变成"`k=0` / 最长那一帧"，跨窗口长度的可比性下降。
这是有意的取舍，不是副作用。

另外三个机械层面的后果，改下游时要知道：

1. `collate_aim` 的补齐**重新用上了**（"batch 内全是 257、一帧都不用补"只对未截断成立）。
2. `T'` 每个样本每轮现摇，所以 batch 内 `Tmax` 基本都贴着 257（实测 `B=8` 时
   `210~257`，`B=64` 约 254），padding 浪费 ≈ `1 - E[T']/Tmax ≈ 48%`。想省掉这半天得把
   摇 `T'` 的位置挪到采样器（按桶摇、桶内排序），训练侧现在做不到 —— 原来的静态长度
   分桶已经被删掉了，它和随机截断天然冲突（扫 npz 量出来的长度恒是 257）。
3. 评估（valid/test）**必须** `augment=False`：`acc_first` 要的是"整条 4 秒窗口的第 0
   帧"，不是随机截断之后的第一个可见帧，否则每次评估的 `T'` 都不一样、跨 run 没法比。

## 变长与 padding

磁盘上的窗口固定 4 秒，64 Hz 下 `T = 257` 是**定值**（实测 449 条全是 257）。变长只有
两个来源：上面那个训练增强（`T' ∈ [6, 257]`，训练时必然触发），以及 `i1` 太靠前、
窗口被截到回合开头（现有 449 条里一条都没有，`T < 257` 仍然按变长处理）。两种都靠
`collate_aim`：逐帧字段右侧补到 batch 内最长的 `T`，并给出 `frame_mask (B, T) bool`
（True = 真数据）。

补齐值不是一律 0，逐个字段不同（错一个就是不报错的假数据）：

    ctx_player / ctx_projectile / global_feat  0    全 0 是"死人 / 空槽 / 未知"，配合 mask 无歧义
    ctx_player_mask                       0    = 死亡，和"补齐"自然一致
    ctx_player_weapon / ctx_projectile_id -1   **0 是 Desert Eagle**，不能当 padding
    obstruction                           0    = 无遮挡；补齐帧由 frame_mask 排除
    seconds                               0    = 可见窗口起点（合法值），靠 frame_mask 排除

一整条 `T=257` 的样本折成 tensor 是 **0.38 MB**（实测 4 条 1.53 MB），`B=64, T=257`
时一个 batch ≈ 24.5 MB，比旧版两张视角图（2 × `(B,192,256,5)` float32 ≈ 250 MB）小
一个量级 —— 这是这一版值得重造数据的直接原因之一。训练时 `T` 按 batch 内的 `Tmax`
算，截断越短越省。

## 逐条返回的字段

见上面那张表，另外 `collate_aim` 会补两个字段：

    frame_mask  (B, Tmax)  bool   True = 真数据（末尾补的是 False）
    pair        (B, 2)     int64  (A 的槽位, B 的槽位)；`ctx_player_mask` 因此塌成 0/1

`pair` 是 **per-sample** 的（窗口内双方一直活着），不是逐帧的。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

#: 会被 `collate_aim` 右侧补齐到 batch 内最长 `T` 的字段 →
#: `(每帧形状, dtype, 补齐值)`。顺序无所谓，只是为了少写一遍循环。
#:
#: 补齐值必须逐字段给：武器 / 道具 id 的 -1 和 0 是两个不同的东西（0 = Desert Eagle、
#: "-1 = 没有"），一律补 0 会凭空造出一片手持沙鹰的幽灵玩家。
_SEQ_FIELDS: tuple[tuple[str, tuple[int, ...], torch.dtype, float], ...] = (
    ("ctx_player", (10, 20), torch.float32, 0.0),
    ("ctx_player_mask", (10,), torch.int8, 0),
    ("ctx_player_weapon", (10,), torch.int8, -1),
    ("obstruction", (10, 10), torch.int8, 0),
    ("ctx_projectile", (32, 4), torch.float32, 0.0),
    ("ctx_projectile_id", (32,), torch.int8, -1),
    ("global_feat", (5,), torch.float32, 0.0),
    ("seconds", (), torch.float32, 0.0),
)

#: 形状和 `T` 无关、拼 batch 时直接 stack 的字段（现在只剩 `map_id`：`global_feat`
#: 改成逐帧之后就归 `_SEQ_FIELDS` 了）。
_FIXED_FIELDS: tuple[str, ...] = ("map_id",)

#: 样本里要读出来交给训练的 key（npz 里还有 `hs` / `dmg`，训练用不上，丢掉）。
_KEYS: frozenset[str] = frozenset(f for f, *_ in _SEQ_FIELDS) | frozenset(
    _FIXED_FIELDS
)

#: `ctx_player_mask` 的取值（和 `tasks/aim.py` 的 `MASK_*` 一致）。
MASK_DEAD, MASK_ATK, MASK_TGT, MASK_OTHER = 0, 1, 2, 3

#: 随机截断的下限（`AimDataset.__getitem__`）。6 帧 = 最后 78 ms，再短连"举枪"都看不到，
#: 而且 `T'` 太小的时候 batch 内的 `obstruction`/`weapon` 切片会退化成纯静态快照。
DEFAULT_MIN_TICKS = 6


def _tensor(a: np.ndarray) -> torch.Tensor:
    """npz 里的数组 → tensor，**不拷贝**（`np.load` 出来的数组本来就可写）。

    0 维数组（`map_id`）也照常转成 0 维 tensor，拼 batch 时再 stack 成 `(B,)`。
    注意 `np.ascontiguousarray` 会把 0 维抬成 `(1,)`，所以要 reshape 回原形状。
    """
    return torch.from_numpy(np.ascontiguousarray(a).reshape(a.shape))


def load_arrays(path: str | Path) -> dict[str, torch.Tensor]:
    """读一条 npz → `__getitem__` 返回的那种"不补齐、不换位"的字段字典。

    方向固定为磁盘上的那个：`mask == 1` 是击杀者、`label=True`。推理 / 可视化要用
    **固定方向**时调这个，别用 `AimDataset`（它每次随机对调 A/B，那是训练要的均衡）。
    """
    with np.load(path) as z:
        item = {k: _tensor(z[k]) for k in z.files if k in _KEYS}
    item["label"] = torch.tensor(True)          # A = mask==1 = 击杀者
    return item


def swap_ab(item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """**原地**把 A/B 对调（mask 里的 1 ↔ 2）并让标签取反，返回同一个 dict。

    只改 `ctx_player_mask` 这一张表：其余字段都是"按玩家下标"的绝对量
    （`ctx_player` 是第 p 个玩家的状态、`obstruction[i][j]` 是 i 看 j、武器同理），
    跟"谁是 A"无关。旧版要成对换 `atk`↔`tgt`、`atk_view`↔`tgt_view`、
    `atk_weapon`↔`tgt_weapon`，换漏一个就是静默错位。
    """
    m = item["ctx_player_mask"]
    atk = m == MASK_ATK
    tgt = m == MASK_TGT
    item["ctx_player_mask"] = torch.where(
        atk, torch.full_like(m, MASK_TGT), torch.where(tgt, torch.full_like(m, MASK_ATK), m)
    )
    item["label"] = torch.tensor(False)
    return item


def last_ticks(item: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    """**原地**只留最后 `n` 帧，并把 `seconds` 对零（第一帧恒为 0.0）。

    切的是**尾巴**：窗口终点固定在击杀那一刻，丢掉前面的帧 = "只看最近 n 帧"，标签和
    数据都还是击杀前那一小段。

    `seconds` 对零是这套增强的**核心**，不是顺手：见模块 docstring 的"为什么时间要
    对零"。一句话——绝对秒数会把"离击杀还有多久"直接喂给模型（窗口终点恒是击杀，
    于是 `256 - k` 就是答案），那时间 bias 一点没去掉。

    `n >= T` 时原样返回（`load_arrays` 出来的整条窗口就是最常见的"不截断"；它的
    `seconds[0]` 本来就是 0，对零是对恒等操作）。
    """
    if n >= int(item["seconds"].shape[0]):
        return item
    for key, *_ in _SEQ_FIELDS:
        item[key] = item[key][-n:]
    # 两者都是 1/64 的整数倍（`< 4.1`），差值精确可表示，没有累积误差。
    item["seconds"] = item["seconds"] - item["seconds"][0]
    return item


class AimDataset(Dataset):
    """一个 split 的 aim 样本集合。

    参数
    ----
    root
        `data/demos/prepared/aim` 这样的任务根目录（下面有 `train` / `valid` / `test`），
        或者直接装 npz 的目录。
    split
        取哪个子目录：`"train"`（默认）/ `"valid"` / `"test"`。给 `None` 表示三个
        split 合并（做推理 / 统计时用）。`root` 本身就是样本目录时忽略这个参数。
    augment
        是否做**随机截断**（保留最后 `T'` 帧，`T' ~ U{min_ticks, T}`），默认开。
        训练用默认值；评估 / 推理 / 统计**必须**传 `False`，否则指标随每次抽样抖动。
    min_ticks
        截断下限，默认 6。`T <= min_ticks` 的样本不截断。
    swap
        是否随机对调 A/B（每次 50%），默认开。它是对称性的来源，不是噪声。
        `model.antisym: true` 时它变成纯冗余（见模块 docstring 末尾），应当传 `False`。
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        *,
        augment: bool = True,
        min_ticks: int = DEFAULT_MIN_TICKS,
        swap: bool = True,
    ) -> None:
        root = Path(root)
        if (root / "train").is_dir():           # 任务根目录：按 split 取子目录
            splits = ("train", "valid", "test") if split is None else (split,)
            self.samples = sorted(p for s in splits for p in (root / s).glob("*.npz"))
        else:                                    # 直接给的样本目录：整个目录算一个 split
            self.samples = sorted(root.glob("*.npz"))
        if not self.samples:
            raise FileNotFoundError(f"{root} 下没有 npz 样本（split={split!r}）")
        if min_ticks < 1:
            raise ValueError(f"min_ticks 必须 >= 1，收到 {min_ticks}")
        self.root = root
        self.split = split
        self.augment = augment
        self.min_ticks = min_ticks
        self.swap = swap

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = load_arrays(self.samples[i])
        if self.augment:
            t = int(item["seconds"].shape[0])
            if t > self.min_ticks:
                # `randint` 上界不含，所以 +1 才是闭区间 [min_ticks, T]。
                last_ticks(item, int(np.random.randint(self.min_ticks, t + 1)))
        if self.swap and np.random.random() < 0.5:
            swap_ab(item)
        return item

    def __repr__(self) -> str:
        return (
            f"AimDataset(root={str(self.root)!r}, split={self.split!r}, n={len(self)}, "
            f"augment={self.augment}, min_ticks={self.min_ticks}, swap={self.swap})"
        )


def collate_aim(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """变长样本拼 batch：时间维补到 batch 内最长，并给出 `frame_mask`。

    补出来的帧在逐帧字段里都是**该字段的 padding 值**（见 `_SEQ_FIELDS`），用
    `frame_mask`（注意力靠"补齐在末尾"天然看不到它，所以模型里不用再传 padding
    mask；**损失用模型返回的 `pred_mask`**，现在它就是 `frame_mask` 本身 —— 每帧一个
    预测点，没有额外那列"看了全部帧"，见 `training/aim/model.py`）。
    **不要**假设补齐长度是某个常数，也别按 `T` 做 reshape（训练开了随机截断，
    `Tmax` 每个 batch 都不一样）。

    另外把"谁是 A、谁是 B"从 `ctx_player_mask` 里提出来变成 `pair (B,2)` 的下标，
    `ctx_player_mask` 本身塌成 0 死 / 1 活 —— 见函数末尾。
    """
    b = len(samples)
    tmax = max(int(s["seconds"].shape[0]) for s in samples)
    out: dict[str, torch.Tensor] = {}
    frame_mask = torch.zeros(b, tmax, dtype=torch.bool)
    for i, s in enumerate(samples):
        t = int(s["seconds"].shape[0])
        frame_mask[i, :t] = True
    out["frame_mask"] = frame_mask

    for key, feat, dtype, pad in _SEQ_FIELDS:
        buf = torch.full((b, tmax, *feat), pad, dtype=dtype)
        for i, s in enumerate(samples):
            t = int(s[key].shape[0])
            buf[i, :t] = s[key]
        out[key] = buf

    for key in _FIXED_FIELDS:
        out[key] = torch.stack([s[key] for s in samples])
    out["label"] = torch.stack([s["label"] for s in samples])

    # ---- 角色 → 索引：`pair (B,2)`，mask 塌成 0/1 ----
    # 磁盘上"谁是 A、谁是 B"写在每个玩家的行里（`ctx_player_mask` 的 1 / 2）。模型不要
    # 这个：10 个玩家共用一套 `proj`，谁是 A 由帧内那一头按**下标**查询。所以这里把
    # 角色抽成两个索引，mask 只留 0 死 / 1 活。
    #
    # 取第 0 帧的索引就够：窗口内 A 和 B 一直活着（`aim.py` 造数据时
    # `rr.alive[a, win].all()` 卡过），所以 pair 在整条样本上是常量，模型也只按
    # per-sample 索引取（见 `model._pair_vectors`）。
    mask = out["ctx_player_mask"]
    out["pair"] = torch.stack(
        [
            (mask[:, 0] == MASK_ATK).float().argmax(-1),
            (mask[:, 0] == MASK_TGT).float().argmax(-1),
        ],
        dim=-1,
    )
    out["ctx_player_mask"] = (mask > 0).to(mask.dtype)
    return out
