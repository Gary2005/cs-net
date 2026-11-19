"""spatial_prediction 任务的 torch 数据集：**npz 里的一行 = 一个单帧样本**。

    from training.spatial_prediction.dataset import SpatialDataset, make_loader

    ds = SpatialDataset("data/demos/prepared/spatial_prediction", split="train")
    dl = make_loader(ds, batch_size=64, num_workers=4)

    for batch in dl:
        # batch["ctx_player"] (B,10,20) / batch["ctx_player_mask"] (B,10)
        # batch["next_death"]  (B,) long     11 类，10 = 之后没有人死
        ...

目录约定（`scripts/training_data/prepare.py` 的输出）：`root/train`、`root/valid`、
`root/test` 三个子目录。`root` 也可以直接给一个装满 npz 的目录（没有 `train/`
子目录时按"整个目录就是一个 split"处理）。

## 磁盘上「一个回合一个 npz、K 行」，这里摊平成「一行一个样本」

    train/de_anubis_f0_r1_n84.npz          ← 84 = K = 这个回合被抽到的帧数（1 s 一帧）
    ├── ctx_player        (84,10,20) float32
    ├── ctx_player_mask   (84,10)    bool        True = 这一帧活着
    ├── ...
    └── next_kill         (84,)      int8

`SpatialDataset` 只读文件名里的 `_n{K}`（**不打开文件**，和框架的 `n_samples()` 同一条
规矩）建一张「扁平下标 → (第几个文件, 第几行)」的表，于是 `len(ds) = Σ K`（本仓库实测
78 个文件 / **7135 行**），`ds[i]` 返回去掉 K 维之后的一行：

    __getitem__   ctx_player (10,20)   obstruction (10,10)   ctx_projectile (32,4)
                  ctx_player_mask (10) ctx_player_team (10)   ctx_projectile_id (32)
                  ctx_player_weapon (10)  global_feat (5)     seconds () / frame_index ()
                  ct_win ()  alive_end (10)  can_kill (10)  next_death ()  next_kill ()

**为什么不按回合取**：这个任务的标签本来就是单帧的（「这一帧之后会发生什么」），模型
也只吃一帧 —— 磁盘上的 K 维是"同一回合里 1 秒一个的若干时刻"，**不是** aim 那种 4 秒
时间窗，行与行之间没有因果顺序要建模。所以：

* batch 里**不需要 padding，也没有 `frame_mask`**（aim 那一套补齐 / 随机截断在这没有
  对应物，`collate_spatial` 只是 stack）；
* `K` 不参与训练：它只决定文件多大、以及读一行要解压多少（见"读取代价"）；
* 想要整回合的 `(K,...)` 张量（推理 / 统计分析用）调 `load_file(path)`，不要再自己
  `np.load`。

## 5 个标签怎么吃

    磁盘 key      __getitem__    collate 后      训练侧
    ct_win        ()  bool       (B,) bool       BCE（或和下面几个一起做多任务）
    alive_end     (10) bool      (B,10) bool     逐玩家 BCE，**用 ctx_player_mask 掩**
    can_kill      (10) bool      (B,10) bool     逐玩家 BCE，**用 ctx_player_mask 掩**
    next_death    ()  int8       (B,) long       11 类 CrossEntropy，**不掩**
    next_kill     ()  int8       (B,) long       11 类 CrossEntropy，**不掩**

两条掩码规则的理由，都是"别教模型背废话"：

1. `alive_end` / `can_kill` 是**逐玩家**的 bool。当前帧已经死的人，这两个标签**恒
   False**（死人活不到回合结束、也拿不到击杀），这一半分是靠 `ctx_player_mask` 一眼
   就能推出来的 —— 掩掉它，loss 才只奖励"对活人的判断"。`collate_spatial` 已经把
   `ctx_player_mask` 原样留下（它**就是** alive），`loss = (bce * mask).sum() / mask.sum()`。
2. `next_death` / `next_kill` 是**单值 11 类**（0..9 = 玩家，`10` = 这一帧之后本回合
   再没有死亡 / 击杀），**不要**把 10 掩掉：它是有信息量的类，而且回合末段必然大量出现
   （实测占 **11.0%**，`ln 11 = 2.398` 就是纯瞎猜的 CE 基线）。用 `-1` 当哨兵会被
   `CrossEntropyLoss` 直接拒掉（"target out of bounds"），所以造数侧定成 10。

`can_kill` 和 `alive_end` 是**两条独立**的监督信号，别用一条推另一条（赢下回合但死在
最后 0.1 秒的人 `alive_end=False`、`can_kill` 也可能 False；两者没有包含关系）。

实测的先验（7135 行 / 71 350 人次，`density=1 s`、4 个 demo / 78 回合），写 loss / 判
断模型有没有学到东西时对着看：

    平均存活 0.809                  ← `ctx_player_mask` 的均值（每帧 8.1 人活着）
    平均终局存活 0.369              ← `alive_end` 的均值（回合结束时剩 3.7 人）
    当前帧活着、但之后会死 54.3%     ← 这才是 `alive_end` 真正要预测的那部分
    平均还能击杀 0.322              ← `can_kill` 的均值（每帧只有 3.2 个人以后还会拿击杀）
    `next_death` / `next_kill` = 10 各 **11.0%**

所以**只看活人**的话，`alive_end` 的正负比约 **46 : 54**（54.3% 的人之后会死），`can_kill`
约 **40 : 60** —— 两个都是偏负的，别拿"50% 基线"去比，也别用 `pos_weight=1` 的默认口径
解释 loss 曲线。

### `alive_end ⊆ ctx_player_mask`，别把这条读成 bug

`ctx_player_mask` 是**当前帧**的存活，`alive_end` 是**回合结束**时的存活。回合结束瞬间
中枪的人（"打死最后一个敌人、回合立刻结束"那一枪，`ticks[-1] + 1`，实测 78 个回合里
49 个有）在当前帧还活着、但 `alive_end` 已经 False —— 这正是最有信息量的那一类样本，
造数侧已经把它算进标签了（`scripts/training_data/tasks/_util.py::late_kills`）。所以：

    (alive_end & ~ctx_player_mask) 必须为空       # 当前帧死的人不可能活到最后
    (alive_end &  ctx_player_mask) 可以为空        # ← 不是 bug

## 四个容易踩的坑

1. **同一回合的行高度相关**。`ct_win` 是**整个文件**的常量，`alive_end` / `next_*` 在
   回合后半段也趋同，而一个回合有几十上百行（长回合行多，抽帧是 demo 级均匀的）。后果
   *行级 acc / 行级 loss 会被长回合并重复计权*，valid/test 上一个 batch 甚至可能整批
   `ct_win=True`。看指标用 `ds.sample_weights()`（每个回合总权重相等，权重和 = 回合数），
   别只看 `mean(正确)`。
2. **`seconds` / `frame_index` 是回合内的绝对时间**，不是"离结局还有多久"的相对量。
   它**不是泄漏**（推理时你确实知道现在是第几秒），但它是条很强的捷径：回合末段几乎
   所有人都已死、`next_* = 10`，模型完全可以只读表盘就压低 loss，而不看画面。想验证
   有没有走这条捷径就做 ablation（`seconds` 置 0 重训，或只用 `alive` 的 baseline）。
   `frame_index` 是 `rnd["ticks"]` 的下标（比 `seconds` 精确到 tick），**只建议回查原始
   数据用，别喂模型**。`round_id` 同理（demo 内的回合序号，跨 demo 没有可比性）。
   `map_id` 可以喂（embedding）。
3. **`ctx_player_weapon` / `ctx_projectile_id` / `ctx_player_team` 的 `-1` 是"槽位空
   / 队伍未知"**，和 `next_*` 的 `10`（没有下一次）完全是两件事。做 id embedding 时
   `-1` 要单独一个下标（`-1 → 0`，真实 id 往后挪），别直接 `nn.Embedding(...)`——负下标
  在 `nn.Embedding` 里是"从尾部算"，**不会报错**，会静默错位。
4. **槽位（0..9）是 demo 内固定的玩家身份，不是随机标签** —— 同一场 demo 里每个槽位
   对应同一个人（上下半场换边，所以每个槽位的队伍是 ~50/50 的 CT/T）。这带来两个后果：

   * **标签的槽位先验严重不均**：`next_kill` 里 P0 占 **15.8%**、P8 只占 4.6%（10 类
     均匀应是 10%）；`next_death` 里 P8 占 12.0%、P0 只占 4.1%。所以"永远猜 P0"的
     CE 是 **1.85**、猜 P8 是 **2.12**，比 `ln 11 = 2.398` 好得多 —— 模型如果拿到任何
     玩家身份编码（slot embedding / 玩家 id），它会先学这个捷径。
     **当前这套输入里没有玩家身份**（`ctx_player` 是特征、`row 序` 不进模型），所以先验
     现在学不到；以后加 slot embedding 时要意识到它会把 1.85 这个先验直接送进去。
   * **这 4 个 demo 是同一场比赛（同 10 名选手）**，按 demo 划分的 train/valid/test 里
     选手是重叠的 —— 那个 valid 衡量的是"换地图"，不是"换人"。要看泛化得换一场比赛
     的 demo 进 valid/test。

## 没有任何增强

aim 的随机截断（保留最后 `T'` 帧 + `seconds` 对零）是为了把"击杀发生在窗口第几帧"变成
隐变量；这里一条样本就是一帧，没有时间轴可截，也就没有那个泄漏要治（`seconds` 那条
捷径是上面第 2 条说的问题，不能用"对零"解决）。A/B 对调那种对称性增强也不存在 ——
这个任务里 10 个玩家是**同一个张量里的 10 行**，没有"谁是主角"的选择要做。

所以 `__getitem__` 是纯读取：**不摇任何东西**，同一个下标每次返回同一条数据（这点和
aim 不同，写 collate / 采样器时别按"每次都不一样"来设计）。

## 打散：`RandomSampler` 还是 shuffle buffer

打散在**采样器**里，不在 `__getitem__` 里，两种都能用：

* `make_loader(shuffle=True, buffer_rows=None)` —— `DataLoader` 自带的 `RandomSampler`，
  每个 epoch 一个 `randperm(len(ds))`。语义最干净，但**一行一个样本 + 一个回合一个 npz +
  取一行要解整个文件**这三件事凑在一起会要命：随机顺序下几乎每行都落在不同文件上。
  本机实测（282,888 个文件 / 24,248,254 行，NVMe）：`_load_file` 冷读一个文件 **5.1 ms**、
  热读 **1.3 ms**。单独测 loader（16 worker、B=64）只有 **2,339 行/s**；放进真实训练循环
  （B=512、8 worker、3090）稳态 **145 ms/步 = 3.5k 行/s**，一个 epoch 约 **115 分钟**，
  其中 GPU 只要 55 ms/步 —— 一半以上时间在等盘。
* `make_loader(buffer_rows=8192)`（**默认**）—— `ShuffleBufferSampler`：文件顺序随机、
  行在内存里打散，一个文件一个 epoch 只解压一次。同一条训练循环稳态 **69 ms/步 =
  7.4k 行/s**，一个 epoch 约 **54 分钟**，基本贴着 GPU 的上限（9.3k 行/s）。

两种都是**每行恰好出现一次、不重不漏**（`drop_last=True` 时末尾不满 batch 的几十行丢掉，
换 epoch 会换成别的行被丢掉）。buffer 版本里一个 batch 的 512 行是从 8,192 行的窗口
（≈ 95 个回合）里随机抽的，和 `RandomSampler` 的 batch 组成没有可观测差别 —— 它**不是**
"按回合发牌"，batch 里仍然混着几十个回合。

`seed` 会固定采样器的随机序列（同 seed 重建 loader，第一批就是同一批，冒烟里钉了这条），
每个 epoch 自动往前推一格。断点续训不追求逐 batch 复现 —— 要精确接上就得把前 n 个 epoch
的 `iter()` 空跑一遍，不值。

⚠ 行级洗牌对"行"一视同仁，但行并不独立：同一回合的几十上百行共享 `ct_win`（整回合常量）、
`alive_end` 也高度相似。实测 78 个回合、B=64 时一个 batch 平均 1.47 行/回合、最多能挤进
6 行，batch 间 `ct_win` 比例标准差 0.055。**回合数到几千以上时这个偏差变得可以忽略**
（几千个回合 ÷ 64 ≈ 每 batch 有几十个回合），所以不用为它写按回合发牌的采样器。
在回合数很少的数据集上训练/评估时，用 `sample_weights()` 按回合计权来抵消这份重复计权。

## 读取代价与缓存

`.npz` 是 zip，`z[key]` 会把**整个成员**解出来（`mmap_mode` 对 npz 无效），所以"取一行"
其实要解整个文件（一个回合、压缩后 18–44 KB，实测平均 33 KB，K 中位 91）。实测（本机）：

    冷读一个文件（整文件解压，缓存必 miss）      1.34 ms/文件 → 全 miss 走一遍 9.5 s
    缓存命中切一行                               25–33 μs/行
    单进程一个 epoch（全库 7135 行，含 collate）  0.18 s（25 μs/行）
    4 worker 稳态一个 epoch                      0.08–0.14 s（11–19 μs/行）
    4 worker 第 1 个 epoch（含 spawn 起 4 个进程） 1.8–5.6 s（抖动大，看稳态）

（这一组是 78 个回合那条小数据、文件已经在页缓存里的数。**真实规模的首触代价是
5.1 ms/文件** —— 282,888 个文件里绝大多数这一遍才第一次被打开，见上面"打散"一节。）

**缓存策略**（`max_cached_files`）：`None`（默认）= **文件数 ≤ 256 时整个数据集全缓存**
（78 个文件 × ~136 KB ≈ **10.4 MB / worker**），超过才退成 `DEFAULT_CACHE_FILES = 32` 个
的 LRU。这条不是微调，是开局就踩到的坑：32 个文件的 LRU 在 78 个文件上只有 **41% 命中率**，
一个 epoch 5.4 s；全缓存之后 100% 命中，**0.2–0.34 s / epoch**（约 16 倍）。随机访问尤其
吃这个 —— 一个 batch 的 64 行几乎落在 64 个不同的文件上，小 LRU 基本必然全 miss。

⚠ **`max_cached_files` 和打散方式必须配套**。LRU 只在"同一批行反复落回同几个文件"时才有
用：`RandomSampler` + 282,888 个文件 + 32 个槽位的 LRU ≈ 0% 命中率，等于每行都整文件解压
（训练循环里就是上面那个 145 ms/步）。`ShuffleBufferSampler` 把"一个文件的所有行"排在一起，
窗口内的文件数 ≈ `buffer_rows / 平均 K`（8,192 / 86 ≈ **95**），所以给 256 个槽位就够
（每个 worker 约 30 MB）。两者是同一个问题的两端，改一个就要看另一个。

（绝对秒数会随机器负载抖动 1~2 倍，看**行级 μs**更稳：命中约 25 μs/行、miss 约 1.3 ms/
文件。）

解压后一帧约 1.5 KB（`ctx_player` 800 B + `ctx_projectile` 512 B + `obstruction` 100 B +
其余约 90 B），所以一个回合文件 ≈ `K × 1.5 KB`。和 `codec.load()`（11–15 s/demo）不是一个
量级，**现在不值得为它换存储**。⚠ 真实规模（282,888 个文件 / 24.2M 行）上这两个数是：
盘上 **8.6 GB**、解压后 **~36 GB**、平均 **30.8 KB / 文件**（K 中位 86）—— 36 GB 装不进
内存，所以"整个数据集全缓存"只在几万个回合以下成立。`ShuffleBufferSampler` 把"取一行"
的代价从"解压一个文件"摊回"解压该文件的 1/86"，是这一档数据量下最省事的解；再往上
（文件数 > 1M、每 epoch 要读好几遍）才值得换成 `.npy` + 每回合一个目录 / 一个 split 一列
一个 mmap 文件，让随机取行不再碰 zlib。

⚠ 两个缓存相关的后果：

* `__getitem__` 返回的是**缓存里那张 tensor 的视图**（`t[row]` 不拷贝）。别在训练循环
  里对它做原地修改（`.mul_()` / `[mask] = 0` 之类），会污染缓存、而且污染跨 batch 传播。
  需要改就 `.clone()`。
* Windows 上 `num_workers > 0` 是 **spawn**，dataset 会被 pickle 到每个 worker。所以
  实现了 `__getstate__` 把缓存丢掉（否则 10.7 MB 张量 × worker 数白拷一遍），worker 里
  从零开始攒，`ds.cache.hits` 在父进程看不到。

## 返回的字段（就是磁盘字段，去掉 K 维）

见上面 `__getitem__` 那张表，另外 `map_id () int8` / `round_id () int16` 是**整个文件
一份**的标量（同文件的所有行相同），`__getitem__` 直接把缓存里那个 0 维 tensor 交出去。
`collate_spatial` 把它们 stack 成 `(B,)`。

0 维字段（`seconds` / `frame_index` / `ct_win` / `next_death` / `next_kill` / `map_id` /
`round_id`）在 `__getitem__` 里是**真 0 维 tensor**（`torch.Size([])`），stack 之后才是
`(B,)` —— 和 `scripts/training_data/dataset.py::to_torch()` 的 `(1,)` 不一样，那边是
`np.ascontiguousarray` 抬维的副作用；这个模块从 torch 直接切片，没有那层。取标量用
`item["next_death"].item()`，取 batch 用 `batch["next_death"][b]`。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# ═══════════════════════════════════════════════════════════════════════════
# 常量（和 scripts/training_data/tasks/spatial_prediction.py 一致，不 import 那边：
# 那个模块在上面挂着 numba / demoparser，训练进程不该被拖进来）
# ═══════════════════════════════════════════════════════════════════════════

#: 玩家数 = 一行特征 / 一个标签列的数量。
N_PLAYERS = 10

#: `next_death` / `next_kill` 的"这一帧之后没有下一次事件"。
NO_EVENT = N_PLAYERS          # 10，所以是 11 类（0..9 是人，10 = 无）

#: `ctx_player_team` 的取值（和 `constants.TEAM_TO_IDX` 一致）。
TEAM_CT = 0
TEAM_T = 1
TEAM_UNKNOWN = -1

#: 逐帧字段（磁盘上是 `(K, ...)`，`__getitem__` 去掉 K）→ `(行内形状, dtype)`。
#:
#: 顺序 = 磁盘上的字段顺序，只是为了 `__repr__` / smoke 打印好看，语义上无序。
_ROW_FIELDS: tuple[tuple[str, tuple[int, ...], torch.dtype], ...] = (
    ("ctx_player", (N_PLAYERS, 20), torch.float32),
    ("ctx_player_mask", (N_PLAYERS,), torch.bool),
    ("ctx_player_team", (N_PLAYERS,), torch.int8),
    ("ctx_player_weapon", (N_PLAYERS,), torch.int8),
    ("obstruction", (N_PLAYERS, N_PLAYERS), torch.int8),
    ("ctx_projectile", (32, 4), torch.float32),
    ("ctx_projectile_id", (32,), torch.int8),
    ("global_feat", (5,), torch.float32),
    ("seconds", (), torch.float32),
    ("frame_index", (), torch.int32),
    ("ct_win", (), torch.bool),
    ("alive_end", (N_PLAYERS,), torch.bool),
    ("can_kill", (N_PLAYERS,), torch.bool),
    ("next_death", (), torch.int8),
    ("next_kill", (), torch.int8),
)

#: 每个文件一份的标量（`__getitem__` 原样把 0 维 tensor 交出去）。
_FILE_FIELDS: tuple[tuple[str, torch.dtype], ...] = (
    ("map_id", torch.int8),
    ("round_id", torch.int16),
)

#: 喂模型的输入列（模型不会用到下面那 5 个标签）。
FEATURE_KEYS: tuple[str, ...] = (
    "ctx_player", "ctx_player_mask", "ctx_player_team", "ctx_player_weapon",
    "obstruction", "ctx_projectile", "ctx_projectile_id", "global_feat",
    "seconds", "frame_index", "map_id",
)

#: 5 个标签。
LABEL_KEYS: tuple[str, ...] = ("ct_win", "alive_end", "can_kill", "next_death", "next_kill")

#: 一条样本里的全部 key（`collate_spatial` 按这个顺序 stack，dict 顺序稳定）。
ALL_KEYS: tuple[str, ...] = tuple(k for k, *_ in _ROW_FIELDS) + tuple(
    k for k, _ in _FILE_FIELDS
)

#: 文件名末尾的 `_n{K}`（K = 这个文件里有几行样本）。造数 / 读数的唯一契约之一。
_N_RE = re.compile(r"_n(\d+)$")

#: 缓存策略：**文件数不超过这个值时全缓存**（每文件解压后约 `K × 1.5 KB`，78 个文件
#: ≈ 10.7 MB）；超过时才退成 `DEFAULT_CACHE_FILES` 个的 LRU。
#:
#: 为什么值得全缓存：一个回合一个文件、一行一个样本，所以「把整个数据集装进内存」在
#: 几十个回合的规模上只要十几 MB —— 而随机访问时 32 个文件的 LRU 在 78 个文件上只有约
#: 41% 命中率，一个 epoch 要多花 3 倍时间。
CACHE_ALL_MAX_FILES = 256

#: 文件数超过 `CACHE_ALL_MAX_FILES` 时的默认 LRU 容量（每个 worker 各一份）。
DEFAULT_CACHE_FILES = 32


def _tensor(a: np.ndarray) -> torch.Tensor:
    """npz 里的数组 → tensor，**不拷贝**（`np.load` 解出来的数组本来就可写）。

    `np.ascontiguousarray` 在 numpy 2.x 里保证 `ndim >= 1`（0 维会被抬成 `(1,)`），
    所以最后要 `reshape` 回原形状 —— 否则 `map_id` 会变成 `(1,)` 而不是 `()`。
    """
    return torch.from_numpy(np.ascontiguousarray(a).reshape(a.shape))


def count_of(path: str | Path) -> int:
    """从文件名解析 `_n{K}`。**这是造数侧和读侧的契约**，缺了就直接报错，不猜 1 行。"""
    m = _N_RE.search(Path(path).stem)
    if m is None:
        raise ValueError(
            f"{Path(path).name} 的文件名里没有 `_n{{K}}`（K = 文件里有几行样本）。"
            f"这是 `tasks/spatial_prediction.py::write_round` 写盘时的契约；"
            f"手工丢进来的 npz 得自己补上（`..._n1.npz` 表示一行）。"
        )
    k = int(m.group(1))
    if k < 1:
        raise ValueError(f"{Path(path).name}: K={k}，一个文件至少一行（K=0 造数侧不写文件）")
    return k


def load_file(path: str | Path) -> dict[str, torch.Tensor]:
    """读**一整个回合**的 npz → `{key: (K, ...) tensor}`（含 `map_id` / `round_id` 0 维）。

    推理 / 统计 / 可视化回查用这个（返回磁盘布局，行就是帧）；训练用 `SpatialDataset`
    （它按行切片，并且有 LRU 缓存）。返回的 tensor 是**新解压出来的**，可以随便改。
    """
    rows, scalars, _ = _load_file(Path(path))
    out = dict(rows)
    out.update(scalars)
    return out


def _load_file(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """`path` → `(逐帧字段 dict, 标量字段 dict, K)`，并校验形状 / dtype / K 三方一致。

    校验放在这里（每个文件每个 worker 一次，走缓存）而不是 `__getitem__`：形状错了是
    **数据问题**，应该在第一次碰到这个文件时就大声报出来，而不是训练到第 3 个 epoch
    才在 collate 里炸成一句看不懂的 stack 报错。
    """
    k = count_of(path)
    with np.load(path) as z:
        files = set(z.files)
        missing = [key for key, *_ in _ROW_FIELDS if key not in files] + [
            key for key, _ in _FILE_FIELDS if key not in files
        ]
        if missing:
            raise ValueError(
                f"{path.name} 缺字段 {missing}（旧版数据？重新 prepare）"
            )
        rows: dict[str, torch.Tensor] = {}
        for key, feat, dtype in _ROW_FIELDS:
            t = _tensor(z[key])
            if t.shape[1:] != feat:
                raise ValueError(f"{path.name}: {key} 的行内形状是 {tuple(t.shape[1:])}，期望 {feat}")
            if t.shape[0] != k:
                raise ValueError(
                    f"{path.name}: {key} 有 {t.shape[0]} 行，文件名里的 K 是 {k}"
                )
            if t.dtype != dtype:
                raise ValueError(f"{path.name}: {key} 的 dtype 是 {t.dtype}，期望 {dtype}")
            rows[key] = t
        scalars: dict[str, torch.Tensor] = {}
        for key, dtype in _FILE_FIELDS:
            t = _tensor(z[key])
            if t.ndim != 0:
                raise ValueError(f"{path.name}: {key} 应该是 0 维标量，实际 {tuple(t.shape)}")
            if t.dtype != dtype:
                raise ValueError(f"{path.name}: {key} 的 dtype 是 {t.dtype}，期望 {dtype}")
            scalars[key] = t
    return rows, scalars, k


class _FileCache:
    """按文件路径的 LRU（`capacity <= 0` = 关闭缓存）。

    `hits` / `misses` 只用于 smoke / 性能测量；`num_workers > 0` 时每个 worker 各一份，
    父进程看不到（见模块 docstring）。
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.hits = 0
        self.misses = 0
        self._items: OrderedDict[Path, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]] = (
            OrderedDict()
        )

    def get(self, path: Path):
        if self.capacity <= 0:
            self.misses += 1
            return _load_file(path)
        try:
            item = self._items[path]
        except KeyError:
            self.misses += 1
            item = _load_file(path)
            self._items[path] = item
            if len(self._items) > self.capacity:
                self._items.popitem(last=False)      # 淘汰最久没用的
            return item
        self.hits += 1
        self._items.move_to_end(path)
        return item


class SpatialDataset(Dataset):
    """一个 split 的单帧样本集合：`len(ds) = 所有文件的 K 之和`，`ds[i]` = 第 i 行。

    参数
    ----
    root
        `data/demos/prepared/spatial_prediction` 这样的任务根目录（下面有 `train` /
        `valid` / `test`），或者直接装 npz 的目录。
    split
        取哪个子目录：`"train"`（默认）/ `"valid"` / `"test"`。给 `None` 表示三个
        split 合并（推理 / 统计用）。`root` 本身就是样本目录时忽略这个参数。
    max_cached_files
        每个 worker 缓存几个解压后的回合文件。`None`（默认）= **文件数 ≤ 256 时全缓存**
        （本仓库 78 个文件 ≈ 10.7 MB / worker，顺序读和随机读都是 100% 命中），否则
        `DEFAULT_CACHE_FILES = 32` 个的 LRU。给 0 表示不缓存（每次取一行都重新解压，
        只在测 I/O 成本时用）。
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        *,
        max_cached_files: int | None = None,
    ) -> None:
        root = Path(root)
        if (root / "train").is_dir():           # 任务根目录：按 split 取子目录
            splits = ("train", "valid", "test") if split is None else (split,)
            files = sorted(p for s in splits for p in (root / s).glob("*.npz"))
        else:                                    # 直接给的样本目录：整个目录算一个 split
            files = sorted(root.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"{root} 下没有 npz 样本（split={split!r}）")

        self.files: tuple[Path, ...] = tuple(files)
        #: 每个文件的行数（`_n{K}`）。`counts.sum() == len(ds)`。
        self.counts = np.array([count_of(p) for p in files], dtype=np.int64)
        #: 每个样本所在的文件下标（`(N,) int32`），评估时按回合均衡计权要用。
        self.groups = np.repeat(
            np.arange(len(self.files), dtype=np.int32), self.counts
        )
        #: `(F+1,)`，`offsets[j]` = 第 j 个文件的第一行在扁平下标里的位置。
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)
        self.root = root
        self.split = split
        # 缓存容量：小数据集直接全装（见 CACHE_ALL_MAX_FILES 的理由），大的走 LRU。
        if max_cached_files is None:
            capacity = len(files) if len(files) <= CACHE_ALL_MAX_FILES else DEFAULT_CACHE_FILES
        else:
            capacity = int(max_cached_files)
        self.cache_capacity = capacity              # `__setstate__` 靠它重建缓存
        self.cache = _FileCache(capacity)

    # ── 长度 / 下标 ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def file_index(self, i: int) -> int:
        """扁平下标 → 文件下标。``searchsorted(offsets, i, "right") - 1``：`offsets`
        是**左闭**起点，`i` 落在第 j 个区间里就返回 j。"""
        return int(np.searchsorted(self.offsets, i, side="right")) - 1

    def row_index(self, i: int) -> int:
        """扁平下标 → 该文件内的行号。"""
        f = self.file_index(i)
        return int(i) - int(self.offsets[f])

    def sample_weights(self) -> np.ndarray:
        """每个样本的权重 = `1 / 该回合的行数` → **每个回合总权重相等**（和 = 文件数）。

        行级 acc / loss 会被长回合重复计权（一个回合几十上百行、长短差 5 倍），valid /
        test 上用它算才是"回合级"的指标：

            right = (logits.argmax(-1) == target)          # (B,) 或 (B,10)
            acc = (right.float() * w).sum() / w.sum()

        训练要不要用它当 loss 权重是另一回事（它会把短回合的梯度放大），这里只提供数字。
        """
        return (1.0 / self.counts[self.groups]).astype(np.float32)

    # ── 取样本 ───────────────────────────────────────────────────────────

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        i = int(i)
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"样本下标 {i} 越界：这个 split 一共 {n} 个样本")
        f = self.file_index(i)
        row = i - int(self.offsets[f])
        rows, scalars, _ = self.cache.get(self.files[f])
        # 逐帧字段取第 row 行（**视图**，不拷贝：见模块 docstring 的"缓存相关后果"）。
        item: dict[str, torch.Tensor] = {key: t[row] for key, t in rows.items()}
        # 整个文件一份的标量，同文件的所有行共用同一个 0 维 tensor。
        item.update(scalars)
        return item

    # ── 杂项 ─────────────────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        """pickle 时丢掉缓存。Windows 上 `num_workers > 0` 是 spawn，dataset 会被拷给
        每个 worker（以及 `persistent_workers` 的每次重建），缓存里那几 MB 张量拷过去
        没有意义 —— worker 里反正要从零开始攒。"""
        state = self.__dict__.copy()
        state["cache"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.cache = _FileCache(getattr(self, "cache_capacity", DEFAULT_CACHE_FILES))

    def __repr__(self) -> str:
        return (
            f"SpatialDataset(root={str(self.root)!r}, split={self.split!r}, "
            f"files={len(self.files)}, n={len(self)}, "
            f"cache={self.cache.capacity})"
        )


def collate_spatial(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """把 B 个单帧样本 stack 成 batch。**没有 padding、没有 `frame_mask`**（每条样本
    形状完全一样，这是"一行一个样本"的直接好处）。

    唯一做的转换：`next_death` / `next_kill` 从 `int8` 转 `torch.long` ——
    `CrossEntropyLoss` 只吃 long（int8 会报 "expected scalar type Long"）。
    其余字段 dtype 原样保留（`ctx_player_mask` / `ct_win` / `alive_end` / `can_kill` 是
    **真 bool**，可以直接当 mask 乘、也可以喂 `BCEWithLogitsLoss`）。
    """
    if not samples:
        raise ValueError("空 batch：DataLoader 的 drop_last 应该拦住它，检查一下采样器")
    out = {key: torch.stack([s[key] for s in samples]) for key in ALL_KEYS}
    for key in ("next_death", "next_kill"):
        out[key] = out[key].long()
    return out


class ShuffleBufferSampler(Sampler[int]):
    """按**文件顺序**读、行在内存里打散：一个 epoch 仍然不重不漏，但每个文件只解压一次。

    为什么不是 `RandomSampler`：一行一个样本、一个回合一个 npz、取一行要解整个文件 ——
    随机顺序下几乎每一行都落在不同文件上。实测（282,888 个文件 / 24.2M 行，NVMe）：单独测
    loader（16 worker、B=64）只有 **2,339 行/s**，真实训练循环里稳态 3.5k 行/s。这里的做法
    和 `tf.data` 的 `shuffle(buffer_size)` 是同一件事：文件顺序随机 → 逐个文件把行号灌进
    缓冲区 → 缓冲区满了整块打散再吐出去。于是每行**恰好出现一次**，而它的位置被限制在
    "读到它时"所在的那个窗口里（窗口 ≈ `buffer_rows` 行 ≈ 95 个回合）。

    ⚠ 和全局 `randperm` 的差别只有"行出现在什么位置"：全局洗牌里一行可以去任何位置，这里
    只能在它被读到之后。对"一个 epoch 均匀走一遍、batch 内混着几十个回合"这个用途没有
    影响（`buffer_rows` 越大越接近全局洗牌），但**不要**拿它当"从整个数据集里无放回抽 K 行"
    的证明 —— 要那个语义（比如严格的分层抽样）就用 `RandomSampler`。

    ⚠ 多 worker 时每个 worker 各有一份 `_FileCache`，而 DataLoader 是按 batch 轮流发给
    worker 的，所以一个文件仍然会被**每个 worker 各解压一次**（`num_workers` 倍）。8 个
    worker 时 282k 文件 × 8 ≈ 225 万次解压，摊到 8 个进程上约 1,400 s —— 但训练循环里它
    被 GPU 的前向反传盖住了（实测稳态 69 ms/步，其中 GPU 自己就 55 ms）。真要把这 8 倍也
    省掉，得把 dataset 换成 `IterableDataset` 按 worker 分片，代价是丢掉 `DataLoader` 那套
    采样/长度语义 —— 现在还不到那一步。

    `seed=None` 用系统熵；给了 seed 就固定整条序列（同 seed 重建 = 逐 epoch 相同）。每个
    `iter()` 往前推一格，所以换 epoch 会重新洗（和 `RandomSampler` 一样）。
    """

    def __init__(
        self,
        ds: Dataset,
        *,
        buffer_rows: int = 8192,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> None:
        if buffer_rows < 1:
            raise ValueError(f"buffer_rows 必须是正整数（None = 用 RandomSampler），收到 {buffer_rows}")
        for attr in ("counts", "offsets"):
            if not hasattr(ds, attr):
                raise TypeError(
                    f"ShuffleBufferSampler 需要 {type(ds).__name__} 提供 `{attr}`"
                    f"（`SpatialDataset` 有；自定义 dataset 请自己实现或者用 shuffle=True）"
                )
        self.ds = ds
        self.buffer_rows = int(buffer_rows)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.ds)

    def _flush(self, pending: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
        """把攒下的几段行号拼起来打散（每段内部已经是随机的，这里只做段间混合）。"""
        block = pending[0] if len(pending) == 1 else np.concatenate(pending)
        if self.shuffle and block.size > 1:
            block = block[rng.permutation(block.size)]
        return block

    def __iter__(self):
        # 每个 epoch 从 `seed + epoch` 重新起一条随机流：同 seed 重建的 loader 前几个 epoch
        # 完全一样（冒烟里钉了这条），而同一个 loader 的第 2 次 `iter()` 一定换序。
        # epoch 在**开头**就自增：中途 `break`（比如 `max_steps`）之后不重跑同一个顺序。
        epoch, self.epoch = self.epoch, self.epoch + 1
        rng = np.random.default_rng(
            None if self.seed is None else int(self.seed) + epoch
        )
        offsets, counts = self.ds.offsets, self.ds.counts
        order = (
            rng.permutation(len(counts)) if self.shuffle else np.arange(len(counts))
        )
        pending: list[np.ndarray] = []
        rows = 0
        for f in order:
            start, n = int(offsets[f]), int(counts[f])
            part = np.arange(start, start + n, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(part)                     # 文件内部也打散
            pending.append(part)
            rows += n
            if rows >= self.buffer_rows:
                yield from self._flush(pending, rng).tolist()
                pending, rows = [], 0
        if pending:
            yield from self._flush(pending, rng).tolist()


def make_loader(
    ds: Dataset,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    seed: int | None = None,
    buffer_rows: int | None = 8192,
    prefetch_factor: int = 4,
) -> DataLoader:
    """`SpatialDataset` → 训练 / 评估用的 DataLoader（参数就是 `train.py` 里那几行）。

    * `shuffle=True`（默认）+ `buffer_rows=8192` → `ShuffleBufferSampler`：文件顺序随机、
      行在内存里打散，一个 epoch 不重不漏，而每个文件只解压一次。
    * `buffer_rows=None` → `DataLoader` 自带的 `RandomSampler`（全局 `randperm`）。语义最
      干净，但文件多的时候每行都要整文件解压（见模块 docstring 的"打散"一节），只在几十个
      回合的小数据集上快。
    * `shuffle=False`（评估）→ 顺序读，配 `drop_last=False` 能保证指标覆盖全部样本，而且
      **行号连续**：`train.py` 的按回合计权就是靠这个特性按下标切 `sample_weights()` 的。
    * `drop_last=True` 只在训练用；评估要 `False`（否则最后一个不满的 batch 被丢掉，
      指标统计不全）——`collate_spatial` 对不满的 batch 一样能 stack。
    * `seed` 固定采样器的随机序列（同一个 seed 重建 loader，前几个 epoch 的顺序完全一样）。
    * `num_workers > 0` 时开 `persistent_workers`：Windows 上每个 epoch 重建进程要重新
      spawn + 重新攒文件缓存，稳态下这段时间是白扔的。
    """
    kwargs: dict = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    common = dict(
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_spatial,
        **kwargs,
    )
    if shuffle and buffer_rows:
        sampler = ShuffleBufferSampler(ds, buffer_rows=buffer_rows, seed=seed)
        return DataLoader(ds, sampler=sampler, **common)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(ds, shuffle=shuffle, generator=generator, **common)


if __name__ == "__main__":
    import shutil
    import sys
    import time

    from torch.utils.data import BatchSampler, RandomSampler

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/demos/prepared/spatial_prediction"
    )
    if not root.is_dir():
        raise SystemExit(f"没有这个目录：{root}（先跑 prepare，或用参数指定）")

    # ── 1. 索引：只读文件名就能算出总样本数 ───────────────────────────────
    ds = SpatialDataset(root, split=None, max_cached_files=0)
    print(f"[1/7] {ds}")
    assert len(ds) == int(ds.counts.sum()) == len(ds.groups)
    assert len(ds.offsets) == len(ds.files) + 1 and ds.offsets[0] == 0
    for i in (0, len(ds) // 2, len(ds) - 1):       # 扁平下标 ⇄ (文件, 行) 自洽
        f, row = ds.file_index(i), ds.row_index(i)
        assert 0 <= row < ds.counts[f]
        assert ds.offsets[f] <= i < ds.offsets[f + 1]
        assert ds.files[f].name.endswith(f"_n{int(ds.counts[f])}.npz")
    print(f"      K（每回合行数）：min={ds.counts.min()} 中位={int(np.median(ds.counts))} "
          f"max={ds.counts.max()}，{len(ds.files)} 个回合文件 = {len(ds)} 行")

    # ── 2. 逐行读全库，检查逐行不变式 + 统计 5 个标签 ─────────────────────
    # 用**开着缓存**的 dataset 顺序读：这是训练时最理想的访问模式（一个回合的 K 行连着
    # 读），也是"一个 epoch 到底多慢"的下界；冷读代价在第 3 节单独量。
    warm = SpatialDataset(root, split=None)          # 默认：78 个文件全缓存
    n = len(warm)
    alive_n = alive_end_n = can_kill_n = 0
    nd_hist = np.zeros(NO_EVENT + 1, dtype=np.int64)
    nk_hist = np.zeros(NO_EVENT + 1, dtype=np.int64)
    dies_later = 0               # 当前帧活着、但回合结束前会死的人次（含回合结束那一枪）
    team_bad = 0
    t0 = time.perf_counter()
    for i in range(n):
        it = warm[i]
        m, end, can = it["ctx_player_mask"], it["alive_end"], it["can_kill"]
        nd, nk = int(it["next_death"]), int(it["next_kill"])
        assert m.shape == end.shape == can.shape == (N_PLAYERS,), i
        assert not (end & ~m).any(), f"行 {i}: 当前帧已死的人却活到了回合结束"
        assert not (can & ~m).any(), f"行 {i}: 当前帧已死的人却能拿击杀"
        assert 0 <= nd <= NO_EVENT and 0 <= nk <= NO_EVENT, (i, nd, nk)
        # next_death == 10 ⟺ 当帧活着的人全都活到了最后（造数侧钉住的不变式）
        assert (nd == NO_EVENT) == bool((m == end).all()), (i, nd)
        if nd != NO_EVENT:
            assert bool(m[nd]) and not bool(end[nd]), (i, nd)   # 下一个死者：现在活、最后死
        if nk != NO_EVENT:
            assert bool(can[nk]), (i, nk)                      # 下一个击杀者必须在 can_kill 里
        assert (nk == NO_EVENT) == (not bool(can.any())), (i, nk)
        team_bad += int((~torch.isin(it["ctx_player_team"],
                                     torch.tensor([TEAM_CT, TEAM_T, TEAM_UNKNOWN]))).sum())
        alive_n += int(m.sum())
        alive_end_n += int(end.sum())
        can_kill_n += int(can.sum())
        dies_later += int((m & ~end).sum())
        nd_hist[nd] += 1
        nk_hist[nk] += 1
    dt = time.perf_counter() - t0
    print(f"[2/7] 全库 {n} 行逐行读完并校验：{dt:.2f} s（{dt / n * 1e3:.3f} ms/行，"
          f"缓存 {warm.cache.capacity} 个文件，命中 {warm.cache.hits} / miss {warm.cache.misses}）")
    assert team_bad == 0, f"ctx_player_team 出现非法值 {team_bad} 次"

    n_pl = n * N_PLAYERS
    print(f"      平均存活 {alive_n / n_pl:.3f}（{alive_n} / {n_pl} 人次）")
    print(f"      平均终局存活 {alive_end_n / n_pl:.3f}，平均还能击杀 {can_kill_n / n_pl:.3f}")
    print(f"      当前帧活着、但回合结束前会死：{dies_later} 人次"
          f"（占全部活人 {dies_later / alive_n:.1%}，含回合结束那一枪）")
    for name, hist in (("next_death", nd_hist), ("next_kill", nk_hist)):
        share = hist / hist.sum()
        #: 槽位是 demo 内固定的**玩家身份**（上下半场换边，所以每槽 ~50/50 队伍），
        #: 同一场比赛里"强手"的槽位会明显偏向某些下标 —— 这就是槽位先验。
        top = int(share[:NO_EVENT].argmax())
        print(f"      {name}：10 = {share[NO_EVENT]:.1%}；槽位先验最大 P{top} = "
              f"{share[top]:.1%}（多数类 CE {np.log(1 / share[top]):.3f}）")
    print(f"      纯瞎猜的 CE 基线：ln 11 = {np.log(NO_EVENT + 1):.3f}")

    # ── 3. 读取代价：整文件解压（冷）vs 缓存命中 ─────────────────────────
    plain = SpatialDataset(root, split=None, max_cached_files=0)
    t0 = time.perf_counter()
    for f in range(len(plain.files)):                  # 每行一个不同文件 → 必 miss
        plain[int(plain.offsets[f])]
    dt_all_files = time.perf_counter() - t0
    print(f"[3/7] 冷读 {len(plain.files)} 个文件：{dt_all_files / len(plain.files) * 1e3:.2f}"
          f" ms/文件（整文件解压，K 中位 {int(np.median(plain.counts))}）→ 一个 epoch "
          f"{dt_all_files / len(plain.files) * n:.1f} s（全 miss，最坏情况）")

    cached = SpatialDataset(root, split=None)
    t0 = time.perf_counter()
    for i in range(n):
        cached[i]
    dt_hit = time.perf_counter() - t0
    print(f"[3/7] 顺序读全库（缓存命中）：{dt_hit:.2f} s / {n} 行"
          f"（{dt_hit / n * 1e6:.0f} μs/行，命中率 "
          f"{cached.cache.hits / (cached.cache.hits + cached.cache.misses):.1%}）")

    # ── 4. DataLoader：形状 / dtype / collate ─────────────────────────────
    train = SpatialDataset(root, split=None)
    batch = next(iter(make_loader(train, batch_size=64, num_workers=0)))
    print(f"[4/7] 一个 batch（B=64）：")
    for key in ALL_KEYS:
        t = batch[key]
        print(f"      {key:18s} {str(tuple(t.shape)):14s} {str(t.dtype):14s}")
    assert batch["ctx_player"].shape == (64, N_PLAYERS, 20)
    assert batch["ctx_player"].dtype == torch.float32
    assert batch["ctx_player_mask"].dtype == torch.bool
    assert batch["next_death"].shape == (64,) and batch["next_death"].dtype == torch.long
    assert 0 <= int(batch["next_death"].min()) and int(batch["next_death"].max()) <= NO_EVENT
    assert 0 <= int(batch["next_kill"].min()) and int(batch["next_kill"].max()) <= NO_EVENT
    assert batch["ct_win"].dtype == torch.bool and batch["alive_end"].dtype == torch.bool
    # 掩码规则的自检：死人那两半标签恒 False，掩掉之后剩下的才是"要预测的"
    m = batch["ctx_player_mask"]
    assert not (batch["alive_end"] & ~m).any() and not (batch["can_kill"] & ~m).any()
    assert int(m.sum()) > 0

    # ── 5. 一个 epoch 的 wall time（含 collate），0 / 4 worker ─────────────
    # `num_workers > 0` 在 Windows 上是 spawn，第一个 epoch 含 4 个进程各自 import
    # torch（实测 1~2 s × 并行），所以第 2 个 epoch 才是稳态 —— 训练循环里 DataLoader
    # 只建一次，要报的是稳态那个数。
    for workers in (0, 4):
        dl = make_loader(train, batch_size=64, num_workers=workers, shuffle=True, seed=0)
        for epoch in range(1, 3):
            t0 = time.perf_counter()
            nb = 0
            for _ in dl:
                nb += 1
            dt = time.perf_counter() - t0
            tag = "第 1 个（含 worker 启动）" if epoch == 1 else "稳态"
            print(f"[5/7] num_workers={workers} {tag}：{nb} 个 batch（{nb * 64} 行）"
                  f"{dt:.2f} s（{dt / max(1, nb * 64) * 1e6:.0f} μs/行，含 collate）"
                  + ("  ← 稳态" if workers and epoch == 2 else ""))

    # ── 6. 洗牌：行级随机、一个 epoch 不重不漏、seed 可复现 ────────────────
    gen = torch.Generator()
    gen.manual_seed(0)
    batches = list(BatchSampler(RandomSampler(train, generator=gen), 64, True))
    idx = np.concatenate([np.asarray(b) for b in batches])
    assert len(np.unique(idx)) == len(idx), "一个 epoch 里有重复行"
    assert len(idx) == len(train) - len(train) % 64, "drop_last 只该丢末尾那几十行"
    per_batch = [len(np.unique(train.groups[b])) for b in batches]
    print(f"[6/7] 行级洗牌（B=64）：{len(batches)} 个 batch、覆盖 {len(idx)} / {len(train)} 行，"
          f"batch 内回合数 均值 {np.mean(per_batch):.1f}（最少 {min(per_batch)}）")
    # 同一个 seed 的两个 loader 第一批一样；同一个 loader 的第二个 epoch 换一批
    a = next(iter(make_loader(train, batch_size=8, num_workers=0, seed=0)))
    b = next(iter(make_loader(train, batch_size=8, num_workers=0, seed=0)))
    assert torch.equal(a["seconds"], b["seconds"]), "同 seed 的 loader 第一批不一致"
    dl2 = make_loader(train, batch_size=8, num_workers=0, seed=0)
    e0, e1 = next(iter(dl2))["seconds"], next(iter(dl2))["seconds"]
    assert not torch.equal(e0, e1), "第二个 epoch 的行没变"
    print("      同 seed 可复现、换 epoch 会重新洗")

    # shuffle buffer：不重不漏、同 seed 可复现、窗口内确实是打散的
    s1 = ShuffleBufferSampler(train, buffer_rows=1024, seed=0)
    idx2 = np.fromiter(s1, dtype=np.int64, count=len(train))
    s2 = ShuffleBufferSampler(train, buffer_rows=1024, seed=0)
    idx3 = np.fromiter(s2, dtype=np.int64, count=len(train))
    assert len(idx2) == len(train) and np.unique(idx2).size == len(train), "buffer 采样器漏行/重行"
    assert np.array_equal(idx2, idx3), "同 seed 的 buffer 采样器不一致"
    assert not np.array_equal(idx2, np.fromiter(s2, dtype=np.int64, count=len(train))), \
        "buffer 采样器的第 2 个 epoch 没换序"
    first = train.groups[idx2[:1024]]
    print(f"      shuffle buffer（1024 行窗口）：{np.unique(first).size} 个回合挤在第一个窗口里，"
          f"窗口内相邻两行的文件不同占比 {(first[1:] != first[:-1]).mean():.1%}")

    # ── 7. 整回合读取 + 按回合均衡的权重 ─────────────────────────────────
    f0 = load_file(train.files[0])
    k0 = int(train.counts[0])
    assert f0["ctx_player"].shape == (k0, N_PLAYERS, 20)
    assert f0["map_id"].ndim == 0 and f0["round_id"].ndim == 0
    w = train.sample_weights()
    assert w.shape == (len(train),) and np.isclose(w.sum(), len(train.files))
    per_round = np.bincount(train.groups, weights=w)
    assert np.allclose(per_round, 1.0)
    print(f"[7/7] load_file({train.files[0].name}) → ctx_player "
          f"{tuple(f0['ctx_player'].shape)}，map_id {int(f0['map_id'])} / "
          f"round_id {int(f0['round_id'])}")
    print(f"      sample_weights：和 = {w.sum():.1f}（= {len(train.files)} 个回合），"
          f"每个回合权重和 = {per_round.min():.3f}..{per_round.max():.3f}")

    # ── 收尾：契约检查，文件名缺 `_n{K}` 必须报错，不能猜 1 行 ───────────────
    bad_dir = Path("tmp") / "spatial_ds_bad_name"
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad = bad_dir / "de_anubis_f0_r0.npz"        # 故意不带 _n{K}
    with np.load(train.files[0]) as z:
        np.savez_compressed(bad, **{k: z[k] for k in z.files})
    try:
        SpatialDataset(bad_dir, split=None)
    except ValueError as e:
        print(f"[冒烟] 坏文件名被拒绝：{str(e)[:70]}…")
    else:
        raise AssertionError("文件名缺 `_n{K}` 却没有报错")
    finally:
        shutil.rmtree(bad_dir, ignore_errors=True)

    print("[冒烟] 全部通过")
