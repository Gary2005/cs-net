"""空间预测数据集：**一个采样帧 → 一个样本**，标签是「这个回合后来发生了什么」。

    python -m scripts.training_data.prepare --task spatial_prediction
    python -m scripts.training_data --task spatial_prediction       # 跑通检查
    python -m scripts.training_data.tasks.spatial_prediction        # 自带冒烟测试

和 `aim` 的关系：**输入侧一模一样，只是没有窗口、标签换了**
------------------------------------------------------------
`tasks/aim.py` 解决的问题是「看 4 秒历史，猜这个人会不会被打死」；这个任务解决的是
「**只看这一帧**，猜接下来会发生什么」。所以：

    输入  和 aim 逐列一致：`AIM_FEATURES` 那 20 列 + 逐帧武器 + 两两遮挡 + 道具 +
          全局状态。**唯一的区别是没有 T 轴**：aim 是 `(T,10,20)`，这里是 `(10,20)`
          （文件里是 `(K,10,20)`，K = 这个回合被抽到的帧数，见下）。
    标签  5 个未来量，见下面的表格。

复用是**在 `write_round` 里惰性 import** 的（`from .aim import _Round`）：任务模块会被
DataLoader 的每个 worker import，模块级 import 会把 `..features` → `..store` →
`demo_extract` 拖进训练进程（见 `tasks/base.py` 的规矩 3）。放进函数里，`read_sample`
那一侧就只依赖 numpy。`aim.py` 的 `_Round` 正好是「整回合 × 全场」的原料，一次算好、
按帧取用，比在这里重写一份 10 个玩家 × 20 列的拼装逻辑可靠得多 —— 两边**不会漂移**。

采样：`DENSITY`
---------------
`DENSITY = 1.0`（秒）= **期望采样间隔**，语义和 aim 的 `LOOKBACK` 一样，是个模块级常量。
一个 demo 打 T 秒 → 抽 `T / DENSITY` 帧，在所有回合的所有采样帧上**均匀随机、不放回**：

    total = Σ 每个回合的帧数                      # demo 级，不是回合级
    want  = round(total / (DENSITY × 64))         # 64 Hz → DENSITY 秒 = 64 帧
    chosen = 升序( rng.choice(total, want, replace=False) )

随机数种子由 **csz 文件名** 派生（`blake2b(f"{SAMPLING_SEED}:{name}")`）：同一个文件、
同一个 `SAMPLING_SEED` 抽到的是同一批帧，换进程 / 换 `--workers` / 重跑都一致。**不用
Python 的 `hash()`**（`PYTHONHASHSEED` 一换就变），也不吃 `--seed`（那个是分 split 的）。

⚠ 抽帧是**按整个 demo 均匀**的，不是"每个回合各抽自己的时长/density"。两种做法样本
总数一样，区别在分布：按 demo 抽时，某个回合抽到几帧是**随机的**（二项分布，期望 =
该回合时长 / density），长回合自然拿得多；按回合抽则把每个回合的帧数钉死，短回合会被
相对过量采样、长回合被相对欠采样。这里选前者 —— 它和"从所有帧里均匀随机选"这句话
字面一致，而且不会因为回合长短不同而扭曲样本分布。

## 一个回合一个文件，文件里 K 行样本

`prepare` 是逐回合调 `write_round` 的，而抽帧计划是 demo 级的（上面那个 `total` 要跨
回合求和），所以 `write_round` 里会**重算一遍整个 demo 的计划**再取本回合那段：

    文件  {地图}_f{fi}_r{rid}_n{K}.npz      K = 这个回合被抽到的帧数（K=0 不写文件）
    样本  第 k 行 = 文件里的第 k 行（k 由 `read_sample` 从扁平下标推出来）

为什么不是「一个样本一个文件」（aim 就是这样）：density=1 s 时一个 demo 有几千个样本，
几万个 demo 就是**上千万个小文件**。一个回合一个文件则只有「回合数」那么多（实测一局
20 个回合左右），而且 `np.savez_compressed` 一次压缩几百行比压几百次小文件省得多。

代价是 `read_sample` 不能再用 `samples[i]`：`i` 要先映射到「第几个文件 + 第几行」。
文件名的 `_n{K}` 就是为此存在的 —— `n_samples()` 只读文件名就能算出总样本数（框架的
硬规矩：不许打开样本文件）。映射表在 `Task` 实例上缓存一份（`len + 首尾文件名` 做键），
否则每次取样本都要重扫一遍目录。

⚠ **文件名里的 K 是造和读之间契约的一部分。** 手工往目录里扔一个没有 `_n{K}` 的 npz，
`n_samples()` 会直接报错（而不是猜 1 行、然后静默读错）。

## 5 个标签

全部是**这一帧之后**（严格 `i > t`，不含当前帧）的因果量：

    key         shape       dtype   含义
    ct_win      ()          bool    CT 是否赢下这个回合（`rnd["winner"] == "CT"`）
    alive_end   (10,)       bool    回合**结束**时还活着的人（= 最终存活）
    can_kill    (10,)       bool    这一帧之后**还会拿到击杀**的人（之前的击杀不算）
    next_death  ()          int8    这一帧之后**第一个死**的玩家下标，10 = 没有了
    next_kill   ()          int8    这一帧之后**第一个拿到击杀**的玩家下标，10 = 没有了

`next_death` / `next_kill` 满足「**必须当前帧还活着**」这条：找事件时逐个检查事件当事人
在当帧的 `alive`，不活着就跳过（正常数据里不会出现，这是给切片器偶发的不一致兜底）。

**`10` = "没有下一次"**（`NO_EVENT = N_PLAYERS`），因为玩家下标只可能是 0..9。所以这两个
标签是**规规矩矩的 11 类**（0..9 是人，10 = 无），训练侧直接 `CE(logits, target)`，不用掩码。
用 `-1` 当哨兵会被 `CrossEntropyLoss` 直接拒掉（"target out of bounds"），而"把 -1 的行
掩掉"又白丢了一条本来能监督「这个回合不会再有人死了」的样本。

`next_death` / `next_kill` 的**其余**字段（`ctx_player_weapon` / `ctx_projectile_id` /
`ctx_player_team`）仍然是 `-1`，那是"槽位空"，和"没有下一次"是两回事，别混。
同理 `ct_win` 只在 `winner ∈ {"CT","T"}` 时才写 —— `winner` 是空串的回合整回合不产出。

几处口径（想改就改这里，别在训练侧猜）：

* **团队击杀算击杀**：`next_kill` / `can_kill` 只看「`events.kills` 里 `a == p`」，
  不区分 a 和 v 是不是同一队。CS2 的记分板不算团队击杀，但这里问的是"下一个杀人的
  是谁"，所以算。想排除就在 `frame_labels()` 里加 `if team[a] != team[v]: continue`。
* **非玩家击杀（`a == -1`，摔死 / C4）算死亡**：`next_death` 只要求 `v` 是有效玩家，
  不要求 `a` 有效 —— 被 C4 炸死一样是"下一个死的人"。
* **`alive_end` 是回合结束时的 `alive`**（最后一帧 + 那之后的一枪），不是"没被击杀"：
  赢下回合但死在最后 0.1 秒的人算 False；拆包成功时还活着的 T 算 True。和游戏里的
  "存活"一致。
* **回合结束瞬间的那一枪也算**（`_util.late_kills()`）：`end_reason ∈ {t_killed, ct_killed}`
  的回合，最后一枪在 `ticks[-1] + 1`，比最后一帧晚 1 tick，`features.round_events()`
  的 `t <= ticks[-1]` 会把它滤掉。它会以"最后一帧之后"的身份参与标签，所以
  `alive_end` 里中枪那位是 `False`、末尾几行的 `next_death` 也指得到他。
  实测 78 个回合里 **49 个**是这种情况（63%），丢掉的话这 49 个样本的存活标签是反的。
  反过来，`bomb_exploded` 的回合**没有**这件事可修：被炸死的人在 csz 的击杀事件流里
  根本没有 `player_death`（实测 16 个爆炸回合里 `hi + 1` 处一条都没有），只能以最后
  一帧的 `alive` 为准。
* `can_kill` 最后与当帧 `alive` 做了 AND：死人不可能"还能拿到击杀"。正常数据下这步是
  空操作，它只是保证标签自洽（真出现"死人后来击杀"就是提取层出了问题）。

## 样本里的 key

    输入（和 aim 逐列一致，只是少了 T 轴；文件里全部多一个 K 维）
    ctx_player        (K,10,20) float32  逐帧全场状态，列序 = `aim.AIM_FEATURES`，
                                         每列都在 [0,1]。死人的行**不保证是 0**，
                                         只认 `ctx_player_mask`。
    ctx_player_mask   (K,10)    bool      True = 这一帧活着（含"没进场" = False）。
                                         ⚠ 是**真 bool**，不是 0/1/2/3 的编码：
                                         `dataset.to_torch()` 对 `*_mask` 和 bool 都转
                                         `torch.bool`，所以它拿到手就能直接当
                                         `key_padding_mask` / `attn_mask` 用。
    ctx_player_team   (K,10)    int8     队伍：0 = CT / 1 = T / -1 = 未知
                                         （和 `constants.TEAM_TO_IDX` 一致）。
                                         常量列（沿时间不变），**死人也保留队伍** ——
                                         "谁是哪队的"和"他死没死"是两个正交的量。
    ctx_player_weapon (K,10)    int8     逐帧武器（`constants.WEAPON_NAMES` 全局下标），
                                         -1 = 没有武器 / 已死亡
    ctx_player_inv    (K,10,43) uint8    **背包**多热，下标同样落在 `WEAPON_NAMES`：
                                         这一帧这个人身上带着哪些东西（枪 / 投掷物 /
                                         C4 / 刀）。死人的行恒全 0（实测 225,836 行
                                         死人里 0 行带东西）。
                                         `ctx_player_weapon` 是**当前手持**的那一把
                                         （切刀 / 拿着雷时会变成 Knife / 那颗雷，实测
                                         占存活样本的 26.6% / 9.0%），这张表才是
                                         "他接下来还能用什么"。
                                         ⚠ csz 里这张表是**去重**的，闪光 2 颗 = 1 颗。
    obstruction       (K,10,10) int8     当帧两两遮挡 0 通 / 1 烟 / 2 墙，有方向，
                                         死人的行 / 列全 0（语义见 `obstruction.py`）
    ctx_projectile    (K,32,4)  float32  当帧道具 `[x y z t]`，四列都在 [0,1]
    ctx_projectile_id (K,32)    int8     -1 = 空槽，0..6 见 `aim._PROJ_*`
    global_feat       (K,5)     float32  `[炸弹 xyz, 距安放秒数/50, 回合时间/200]`
    seconds           (K,)      float32  当帧在**本回合**里的秒数（0 = round_freeze_end）
    frame_index       (K,)      int32    当帧在 `rnd["ticks"]` 里的下标（回查原始数据用）
    map_id            ()        int8     地图编号（`constants.MAP_TO_IDX`）
    round_id          ()        int16    回合号（调试 / 回查用；不是列表下标）

    标签
    ct_win       (K,)    bool
    alive_end    (K,10)  bool
    can_kill     (K,10)  bool
    next_death   (K,)    int8    10 = 本回合没有下一次死亡
    next_kill    (K,)    int8    10 = 本回合没有下一个击杀

`read_sample()` 返回的是**一行**：上面每个数组都去掉 K 维（`ct_win` / `next_death` /
`map_id` 变成 0 维数组），另外 `seconds` / `frame_index` 是标量。`K=1` 的文件和
"一个样本一个文件"读起来没有区别。

⚠ 经过 `SampleDataset` 之后，0 维的 key 会变成 **`(1,)`** 而不是 `()`：`to_torch()`
里用的是 `np.ascontiguousarray()`，它在 numpy 2.x 里保证 `ndim >= 1`（0 维会被抬成一
维）。所以取标量要 `.squeeze(-1)` / `[:, 0]`，collate 之后是 `(B,)`。这是框架行为，
不是这里写错了（aim 的 `hs` / `dmg` 一样）。

## 存储与代价

一帧样本解压后约 **1.9 KB**：`ctx_player` 800 B（10×20×4）、`ctx_projectile` 512 B
（32×4×4）、`ctx_player_inv` 430 B（10×43×1，绝大多数是 0）、`obstruction` 100 B、
`global_feat` 20 B、武器 10 B、道具 id 32 B，再加上标签 / `seconds` / `frame_index`
约 30 B。压缩后按 aim 的经验（~5×，`ctx_projectile` / `ctx_player_inv` 这种大片 0
压得最狠）约 **0.35 KB**。density=1 s 时一个 30 分钟的 demo ≈ 1800 帧 ≈
**0.6 MB**。想更省就调大 `DENSITY`，样本数和体积都是线性缩的。

实测（本仓库 4 个 demo / 78 回合 / 7135 样本，`DENSITY=1.0`）：平均 **0.36 KB/样本**、
**33 KB/文件**（K≈91 行）、一局约 **0.6 MB**、**0.42 MB/分钟 demo**。抽一帧的遮挡
开销混在 `codec.load()`（11–15 s）里，4 进程并行时每个 demo 22–40 s 造完。
（这几个数是**加 `ctx_player_inv` 之前**量的；多热本身按 +29% 估，但压缩率会把它吃掉
一部分，真要报数就重造一批再量。）

读样本：`read_sample` 实测 **2.3 ms/样本**（K≈91）—— 解的是**整个 npz**，所以和 K 成
正比（见上）。单进程一个 epoch 7135 个样本约 **16 s**，4 个 worker 约 4 s。数据量再
大一个量级（百万样本）就该换成 `.npy` + 每回合一个目录，让 `mmap` 只碰一行。

造数成本 = aim 的「整回合特征 + 遮挡」里的特征那一半：遮挡进缓存之后不再逐帧付
（`obstruction_cache.ensure()` 一次建好整回合，这里只按抽到的帧花式索引）。所以
**density 直接决定造数时间**：density=1 s 时一局约 1800 帧，开销和一次
`codec.load()`（11–15 s）相比可以忽略。

## 几条容易踩的

* **`DENSITY` 没有 CLI 参数**：`prepare` 不认任务级选项，改密度就改这个常量。
* **抽帧计划按 demo 重算**：每个回合的 `write_round` 都会把所有回合的帧数加一遍。
  一个 demo 几十个回合、几万帧，这点计算在 12 s 的 `codec.load()` 面前可以忽略。
* **体检不合格的回合**（`round_health()`）在 `write_round` 之前就被框架跳过了，但它
  仍然占着抽帧名额 → 实际样本数可能比 `want` 少一点点。这是故意的：抽帧计划只跟
  demo 有关，不该因为某个回合的健康状况整体抖动。
* **炸弹位置缺失的帧会被丢掉**（`global_feat` 没有可填的值；aim 也是整条跳过）。
  实测缺失只出现在回合最开头，随机抽到的概率极低。
* 训练侧**不要**把 `next_death` / `next_kill` 当 10 类问题：它们是 **11 类**，第 11 类
  （下标 10）就是"这一帧之后不会再有人死 / 拿到击杀"，是**有信息量的标签**，别掩掉。
  也不要用 `alive_end` 反推 `can_kill` —— 两者是独立的监督信号。
* **`ctx_player_inv` 是去重的**（csz 里 `extract.py` 写盘那一步就 `set()` 过了），所以
  闪光 2 颗 = 1 颗。想拿到计数只能重抽 demo，重造 npz 拿不回来。这不是这里省掉的：
  源数据里确实有重复（实测整局粗扫 5.2% 的采样行是 `Flashbang ×2`），是提取层折掉的。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ._util import kills, late_kills
from .base import Task

# ═══════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════

#: 玩家数。和 `constants.N_PLAYERS` 一致，但**不能**在这里 import 那个模块：
#: `constants` 会连带 `demo_extract`，而本模块会被 DataLoader 的每个 worker import。
#: 造数据那一侧（`write_round`）用的是 `aim._Round`，它按 `N_PLAYERS` 补到 10 行。
N_PLAYERS = 10

#: `next_death` / `next_kill` 的"这一帧之后没有下一次"= `10`。
#:
#: 为什么不是 `-1`：玩家下标只可能是 0..9，所以 10 是个天然的"无"。这样标签就是干净的
#: **11 类**，训练侧 `CE(logits, target)` 直接吃，不用掩码；`-1` 会被 `CrossEntropyLoss`
#: 拒掉，而"掩掉这些行"会白丢一条能监督「不会再有人死」的样本。
#: 代价是画图 / 打印时要把 10 当"无"处理（本模块和可视化都按这个约定写）。
NO_EVENT = N_PLAYERS

#: 采样率。`round_seconds` 实测逐帧 +1/64 s，所以 `DENSITY` 秒 = `DENSITY × 64` 帧。
TICK_HZ = 64.0

#: 期望采样间隔（秒），和 aim 的 `LOOKBACK` 一个性质：**改这一个常量就够了**。
#:
#: 1 s 的意思是「平均每秒一个样本」。调大 → 样本更少、更省盘、造得更快，但相邻样本
#: 之间的动作细节丢得更多（一个 3 s 的拉枪在 1 s 采样下只有 3 个点）。从 1 开始试，
#: 觉得"太密/太稀"直接改这里，别在训练脚本里做二次下采样（那等于白造）。
DENSITY = 1.0

#: 抽帧随机种子的前缀。想换一批帧（同时保留原来的数据做对比）就改它。
#: ⚠ 它**不**吃 `prepare --seed`：那个是分 split 的，换了会让 val/test 换图。
SAMPLING_SEED = 0

#: `ctx_player_team` 的取值：和 `constants.TEAM_TO_IDX` 一致（CT=0 / T=1 / -1 未知）。
#: 它是**常量列**（沿时间不变，死人也保留队伍），和 `ctx_player` 第 15 列同一个来源。
#: 单独存一份是因为第 15 列是 [0,1] 的浮点 —— -1 会被 clip 成 0，"队伍未知"就和 CT
#: 分不开了。
TEAM_CT = 0
TEAM_T = 1
TEAM_UNKNOWN = -1
TEAM_NAMES: dict[int, str] = {TEAM_CT: "CT", TEAM_T: "T", TEAM_UNKNOWN: "?"}

#: `read_sample()` 返回的**逐行** key（0 维标量和 K 维数组都在这里）。
ROW_KEYS: tuple[str, ...] = (
    "ctx_player", "ctx_player_mask", "ctx_player_team", "ctx_player_weapon",
    "ctx_player_inv", "obstruction",
    "ctx_projectile", "ctx_projectile_id", "global_feat",
    "seconds", "frame_index",
    "ct_win", "alive_end", "can_kill", "next_death", "next_kill",
)

#: 每个文件一份、每个样本都带着的标量 key。
SCALAR_KEYS: tuple[str, ...] = ("map_id", "round_id")

#: 样本文件名末尾的 `_n{K}`（K = 这个文件里有几行样本）。造和读的唯一契约之一。
_COUNT_RE = re.compile(r"_n(\d+)$")


# ═══════════════════════════════════════════════════════════════════════════
# 文件名 ⇄ 扁平下标
# ═══════════════════════════════════════════════════════════════════════════


def sample_counts(samples: Sequence[Path]) -> list[int]:
    """每个样本文件里有几行样本（= 文件名末尾的 `_n{K}`）。

    `n_samples()` 和 `read_sample()` 都只靠这个函数算偏移，所以"一个文件几行"只有
    一处定义。名字里没有 `_n{K}` 的文件**直接报错**：静默按 1 行处理会让 `len(ds)` 和
    磁盘上的行数对不上，然后要 shuffle 到越界才炸（而且那个错看起来像 dataset 的锅）。
    """
    out: list[int] = []
    for p in samples:
        m = _COUNT_RE.search(p.stem)
        if m is None:
            raise ValueError(
                f"{p.name} 的文件名里没有 `_n<K>`（K = 这个文件装了几行样本）。\n"
                f"  spatial_prediction 的文件名形如 de_mirage_f0_r3_n97.npz；\n"
                f"  别的任务留下的 / 手工拼的文件请删掉，或者重新跑：\n"
                f"    python -m scripts.training_data.prepare --task spatial_prediction --overwrite"
            )
        out.append(int(m.group(1)))
    return out


def sample_offsets(counts: Sequence[int]) -> np.ndarray:
    """`[每个文件的行数]` → `(n_files+1,)` 的累积偏移（`starts[-1]` = 总样本数）。

    `i` 落在 `[starts[f], starts[f+1])` 里就说明它是第 `f` 个文件的第 `i-starts[f]` 行。
    """
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(np.asarray(counts, dtype=np.int64), out=starts[1:])
    return starts


def _allocate(
    lengths: Sequence[int],
    name: str,
    *,
    density: float | None = None,
    seed: int = SAMPLING_SEED,
) -> list[np.ndarray]:
    """抽帧计划：`[每个回合的帧数]` → `[每个回合被抽到的帧下标]`（升序）。

    **纯函数**（只依赖 numpy + 文件名），所以冒烟测试能直接钉住 density 的语义：

        total = Σ lengths
        want  = max(1, round(total / (density × 64)))       # density 秒 = 64 帧
        chosen = 升序随机不放回地取 want 个 ∈ [0, total)

    种子从 `f"{seed}:{name}"` 单调派生（`blake2b`，跨进程 / 跨机器稳定）。`name` 是
    **csz 文件名**，不是 `file_idx`：往目录里加文件会让下标整体平移，抽到的帧就变了。

    `lengths` 里的 0（没有采样点的回合）不会分到任何帧；`total == 0` 时返回全空。
    """
    d = DENSITY if density is None else float(density)
    lens = [int(n) for n in lengths]
    total = int(sum(lens))
    if total <= 0 or not lens:
        return [np.zeros(0, dtype=np.int64) for _ in lens]
    if d <= 0:
        raise ValueError(f"density 必须 > 0，现在是 {d!r}")

    want = int(round(total / (d * TICK_HZ)))
    want = max(1, min(want, total))          # 至少 1 帧：density 比整局还长也别空手

    digest = hashlib.blake2b(f"{seed}:{name}".encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    chosen = np.sort(rng.choice(total, size=want, replace=False)).astype(np.int64)

    out: list[np.ndarray] = []
    base = 0
    for n in lens:
        lo = int(np.searchsorted(chosen, base, side="left"))
        hi = int(np.searchsorted(chosen, base + n, side="left"))
        out.append(chosen[lo:hi] - base)
        base += n
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 标签（纯函数，不碰 store / demo_extract）
# ═══════════════════════════════════════════════════════════════════════════


def frame_labels(
    events: Iterable[dict],
    alive: np.ndarray,
    idx: np.ndarray,
    *,
    n_players: int = N_PLAYERS,
) -> dict[str, np.ndarray]:
    """「这一帧之后会发生什么」的 4 个标签（`ct_win` 和它无关，由调用方补）。

    Args:
        events: `[{"i": 采样下标, "a": 击杀者, "v": 受害者, ...}]`，`i` 是 `rnd["ticks"]`
            里的下标。**必须已经滤到本回合**（用 `_util.kills(rnd)`，它转发
            `features.round_events()` 做过滤）。顺序无所谓，这里按 `i` 稳定排序；同一帧的
            多条事件按原始顺序（换人互换时"下一个"取先出现的
            那条，测试里钉住了这一点）。
            `i >= alive.shape[1]` 表示"发生在**最后一帧之后**"：它照样参与每一行的
            "下一个"，并且它的受害者不算活到回合结束。回合结束瞬间的那一枪就是这样
            （见 `_util.late_kills()`）。
        alive: `(n_players, n_frames)` bool 的整回合存活矩阵（`aim._Round.alive`）。
        idx: `(K,)` 采样帧下标（升序、去重）。
        n_players: 输出宽度（默认 10；`alive` 行数不够时缺的列按"一直死"处理）。

    Returns:
        `{"alive_end": (K,P) bool, "can_kill": (K,P) bool,
          "next_death": (K,) int8, "next_kill": (K,) int8}`

    `next_death` / `next_kill` 用 `NO_EVENT`（= 10）表示**这一帧之后本回合再也没有**死亡 /
    击杀。它是合法且**有信息量**的标签（回合末尾必然出现），所以是 11 类而不是 10 类，
    训练侧不用掩码。

    实现上只做「每个采样帧扫一遍它之后的事件」：一个回合的事件数是 10~30 条，采样帧
    几百个，总开销可以忽略。用 `searchsorted` 把"之后"的起点一次算好，避免 O(K×E) 的
    条件判断。
    """
    t_idx = np.asarray(idx, dtype=np.int64)
    k = int(t_idx.size)
    alive = np.asarray(alive, dtype=bool)
    rows = min(int(n_players), int(alive.shape[0]) if alive.ndim == 2 else 0)
    n_frames = int(alive.shape[1]) if alive.ndim == 2 else 0

    # 事件按 i 排序（`kills()` 给的基本有序，但别假定）；只要带 i 的。
    ev = sorted((e for e in events if e.get("i") is not None), key=lambda e: int(e["i"]))
    ev_i = np.fromiter((int(e["i"]) for e in ev), dtype=np.int64, count=len(ev))
    # 发生在整回合最后一帧之后的事件（回合结束瞬间的那一枪）。`searchsorted` 天然把它
    # 排在所有 `t_idx` 之后，所以它只影响"下一个"的搜索；`alive_end` 另外单独处理。
    late_victims = [int(e["v"]) for e in ev if int(e["i"]) >= n_frames]

    next_death = np.full(k, NO_EVENT, dtype=np.int8)
    next_kill = np.full(k, NO_EVENT, dtype=np.int8)
    can_kill = np.zeros((k, int(n_players)), dtype=bool)
    alive_here = np.zeros((k, int(n_players)), dtype=bool)
    if rows:
        alive_here[:, :rows] = alive[:rows, t_idx].T

    # start[row] = 第一个 i > t 的事件的下标（严格大于：当前帧的事件不算"之后"）
    start = (
        np.searchsorted(ev_i, t_idx, side="right")
        if ev_i.size
        else np.zeros(k, dtype=np.int64)
    )

    for row in range(k):
        living = alive_here[row]
        death = NO_EVENT
        kill = NO_EVENT
        for e in ev[int(start[row]):]:
            a = int(e.get("a", -1))
            v = int(e.get("v", -1))
            # 当事人当帧必须活着 —— 这就是「下一个死亡的玩家必须当前帧还活着」。
            # 正常数据里恒成立；不成立说明提取 / 切片有洞，跳过它而不是写出错标签。
            if death == NO_EVENT and 0 <= v < rows and living[v]:
                death = v
            if kill == NO_EVENT and 0 <= a < rows and living[a]:
                kill = a
            if 0 <= a < int(n_players):
                can_kill[row, a] = True
        next_death[row] = death
        next_kill[row] = kill

    # 死人不可能"还能拿到击杀"。正常数据下这是空操作，兜住"死后还有击杀事件"的坏数据。
    can_kill &= alive_here

    alive_end = np.zeros((k, int(n_players)), dtype=bool)
    if rows:
        alive_end[:, :rows] = alive[:rows, -1][None, :]
    # 最后一帧之后才死的人，不能算"活到回合结束" —— 否则 `alive_end` 会把
    # "打死最后一个敌人、回合随即结束"的那一枪算漏，而那正是最有信息量的样本。
    for v in late_victims:
        if 0 <= v < int(n_players):
            alive_end[:, v] = False

    return {
        "alive_end": alive_end,
        "can_kill": can_kill,
        "next_death": next_death,
        "next_kill": next_kill,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 任务
# ═══════════════════════════════════════════════════════════════════════════


class SpatialPrediction(Task):
    ext = ".npz"

    desc = (
        f"单帧预测：density={DENSITY:g}s 在整局上随机抽帧，标签 = CT 胜 / 终局存活 / "
        "还能否击杀 / 下一个死亡 / 下一个击杀（一个回合一个 npz，K 行样本）"
    )

    def __init__(self) -> None:
        #: `samples` 的偏移缓存。键是 `(文件数, 首文件名, 尾文件名)` —— 同一个
        #: `SampleDataset` 每个 `__getitem__` 都会把同一个 list 传进来，所以命中率是 100%；
        #: 换 split / 重新 prepare 之后键一定变。不缓存的话每取一个样本都要重扫目录
        #: （几千个文件名 × 每个 epoch），这才是慢的地方。
        self._cache_key: tuple | None = None
        self._starts: np.ndarray | None = None

    # ── 扁平下标 ⇄ 文件/行 ────────────────────────────────────────────────

    def _index(self, samples: Sequence[Path]) -> np.ndarray:
        key = (
            len(samples),
            samples[0].name if samples else "",
            samples[-1].name if samples else "",
        )
        if self._cache_key != key or self._starts is None:
            self._starts = sample_offsets(sample_counts(samples))
            self._cache_key = key
        return self._starts

    def n_samples(self, samples: list[Path]) -> int:
        """总样本数 = 所有文件的 `_n{K}` 之和。**只读文件名**，不打开 npz。"""
        return int(self._index(samples)[-1])

    def read_sample(self, samples: list[Path], i: int) -> Any:
        """第 `i` 个样本 = 第 `f` 个文件的第 `k` 行。

        ⚠ `.npz` 是 zip，`z[key]` 会把**整个成员**解出来（`mmap_mode` 对 npz 无效），
        所以取一行其实解了整个文件（一个回合几十到几百行、压缩后几十 KB）。这是
        "一个回合一个文件"的代价：`read_sample` 的耗时和 K 成正比，不是和 1 行成正比。
        真嫌慢就换 `ext=".npy"` + 每个回合一个目录的布局（框架支持，见 `tasks/base.py`），
        这里先不复杂化。
        """
        starts = self._index(samples)
        total = int(starts[-1])
        if not 0 <= i < total:
            raise IndexError(
                f"样本下标 {i} 越界：这个 split 一共 {total} 个样本"
                f"（{len(samples)} 个文件）"
            )
        f = int(np.searchsorted(starts, i, side="right")) - 1
        k = int(i - int(starts[f]))
        with np.load(samples[f]) as z:
            out = {key: np.asarray(z[key][k]) for key in ROW_KEYS}
            for key in SCALAR_KEYS:
                out[key] = np.asarray(z[key][()])
        return out

    # ── 造数据 ────────────────────────────────────────────────────────────

    def write_round(self, store, *, fi: int, rid: int, split: str, path: Path) -> None:
        """把这个回合被抽到的帧写成 `{path.stem}_n{K}.npz`。K=0 时一个文件都不写。"""
        # 惰性 import：见模块 docstring。`_PROJ_MAX` 也从 aim 拿，保证 32 行的上限只有
        # 一处定义（写死一个 32 在这里，改了 aim 那边就会静默错位）。
        from ..constants import MAP_TO_IDX
        from .aim import _PROJ_MAX, _Round

        map_name = store.map_name(fi)
        map_id = MAP_TO_IDX.get(map_name)
        if map_id is None:
            return          # 不在这 8 张图里：normalize_xyz 会静默失真，整回合不产出

        from ..obstruction_cache import ensure as ensure_obstruction
        rnd = store.round(fi, rid)
        winner = str(rnd.get("winner") or "")
        if winner not in ("CT", "T"):
            return          # 胜负未知 → `ct_win` 没有真值，整个回合不产出

        idx = self._picks(store, fi).get(rid)
        if idx is None or idx.size == 0:
            return          # 这个回合一个帧都没抽到（短回合 / 其他回合占了名额）

        csz = getattr(store, "paths", None)
        obstruction = ensure_obstruction(
            csz[fi] if csz else None, rnd, map_name, N_PLAYERS
        )
        rr = _Round(store, fi, rnd, map_name, obstruction=obstruction)
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < rr.n_frames)]
        if idx.size == 0:
            return
        # 炸弹位置是逐帧存的，缺的帧 `global_feat` 没有可填的值（填 0.5 = 凭空造一颗
        # 地图中心的 C4）。实测 456 662 帧里只有 8 帧缺，随机抽到的概率极低。
        idx = idx[rr.bomb_ok[idx]]
        if idx.size == 0:
            return

        k = int(idx.size)
        team = rr.team.astype(np.int8)                                   # (10,) 0/1/-1
        alive_here = rr.alive[:, idx].T                                  # (K,10)

        # 道具逐帧取：`projectiles()` 是 (T,32,4)，这里每次只要一帧（窗口长度 1）。
        proj = np.zeros((k, _PROJ_MAX, 4), dtype=np.float32)
        proj_id = np.full((k, _PROJ_MAX), -1, dtype=np.int8)
        for row, i in enumerate(idx):
            p, pid = rr.projectiles(slice(int(i), int(i) + 1))
            proj[row] = p[0]
            proj_id[row] = pid[0]

        labels = frame_labels(
            kills(rnd) + late_kills(rnd, n_frames=rr.n_frames), rr.alive, idx
        )

        np.savez_compressed(
            path.with_name(f"{path.stem}_n{k}{path.suffix}"),
            ctx_player=np.ascontiguousarray(rr.feat[:, idx].transpose(1, 0, 2)),  # (K,10,20)
            ctx_player_mask=alive_here.astype(bool),                             # (K,10) True=活
            ctx_player_team=np.tile(team, (k, 1)),                               # (K,10) 0/1/-1
            ctx_player_weapon=np.ascontiguousarray(rr.weapon[:, idx].T),         # (K,10) -1=无
            # 背包：**当前手持**那把也在里面（实测 0 例外），模型侧按 `weapon` 那一列扣掉。
            ctx_player_inv=rr.inventory_at(idx),                                 # (K,10,43) uint8
            obstruction=rr.obstruction_at(idx),                                  # (K,10,10)
            ctx_projectile=proj,                                                 # (K,32,4) [0,1]
            ctx_projectile_id=proj_id,                                           # (K,32)
            global_feat=rr.global_feat(slice(0, rr.n_frames))[idx],              # (K,5)
            seconds=rr.secs[idx].astype(np.float32),                             # (K,)
            frame_index=idx.astype(np.int32),                                    # (K,)
            ct_win=np.full(k, winner == "CT", dtype=bool),                       # (K,)
            alive_end=labels["alive_end"],                                       # (K,10)
            can_kill=labels["can_kill"],                                         # (K,10)
            next_death=labels["next_death"],                                     # (K,) 10=无
            next_kill=labels["next_kill"],                                       # (K,) 10=无
            map_id=np.int8(map_id),
            round_id=np.int16(rid),
        )

    def _picks(self, store, fi: int) -> dict[int, np.ndarray]:
        """整个 demo 的抽帧计划 `{round_id: 本回合的帧下标}`。

        每个回合都会重算一遍（`write_round` 一次只拿到一个 `rid`，而 `total` 要跨回合
        求和）。代价是 `Σ len(round_seconds)` 的加法 + 一次 `rng.choice`，和 12 s 的
        `codec.load()` 不是一个量级 —— 别为它加缓存，缓存才是下次出 bug 的地方。
        """
        rids = list(store.round_ids(fi))
        lengths = [len(store.round(fi, rid).get("round_seconds") or ()) for rid in rids]
        per_round = _allocate(lengths, store.paths[fi].name)
        return dict(zip(rids, per_round))


TASK = SpatialPrediction()


# ═══════════════════════════════════════════════════════════════════════════
# 冒烟测试：python -m scripts.training_data.tasks.spatial_prediction
# ═══════════════════════════════════════════════════════════════════════════
#
# 三段：
#   ① `frame_labels()` 的语义（纯 numpy，秒回）
#   ② `_allocate()` 的 density / 可复现性（纯 numpy，秒回）
#   ③ 真 csz → 造 → 读：用 1 个 csz 造一份到 `tmp/training_data/<任务>`，逐条读回来
#      检查形状 / dtype / mask 与 alive 一致 / 标签自洽。
#
# ⚠ import 全写在 `if` 里面：任务模块会被每个 worker import，写在外面会让每个 worker
#   都连带 `..prepare`（`CszStore` / `demo_extract`）。
if __name__ == "__main__":
    import shutil

    from .._paths import REPO_ROOT
    from ..constants import N_WEAPONS
    from ..dataset import SampleDataset
    from ..prepare import SPLITS, build
    from .aim import _PROJ_MAX

    def _check_labels() -> None:
        # 玩家 3 在第 3 帧死；玩家 2 在第 5 帧死。
        alive = np.ones((N_PLAYERS, 8), dtype=bool)
        alive[3, 3:] = False
        alive[2, 5:] = False
        events = [
            {"i": 3, "a": 1, "v": 3},      # 1 杀 3
            {"i": 5, "a": 4, "v": 2},      # 4 杀 2
            {"i": 5, "a": -1, "v": 0},     # C4 / 摔死：a 无效，但 v 是真的死亡
        ]
        idx = np.array([0, 2, 3, 4, 5])
        out = frame_labels(events, alive, idx)

        assert out["next_death"].tolist() == [3, 3, 2, 2, 10], out["next_death"].tolist()
        assert out["next_kill"].tolist() == [1, 1, 4, 4, 10], out["next_kill"].tolist()
        assert out["next_death"].max() == NO_EVENT == N_PLAYERS
        # 11 类的下界：10 是唯一的"无"，不会出现 11 这种下标
        assert (out["next_death"] <= NO_EVENT).all() and (out["next_kill"] <= NO_EVENT).all()
        # can_kill：t<3 时 1 和 4 都还有击杀；t=3/4 只剩 4；t=5 没了
        want_can = np.zeros((5, N_PLAYERS), dtype=bool)
        want_can[0, [1, 4]] = want_can[1, [1, 4]] = True
        want_can[2, 4] = want_can[3, 4] = True
        assert np.array_equal(out["can_kill"], want_can), out["can_kill"]
        # alive_end = 最后一帧的 alive：2 和 3 都是 False
        want_end = np.ones((5, N_PLAYERS), dtype=bool)
        want_end[:, [2, 3]] = False
        assert np.array_equal(out["alive_end"], want_end)
        assert out["next_death"].dtype == np.int8

        # 当事人当帧已经不活着（数据不一致）：跳过它，取下一个。
        alive2 = np.ones((N_PLAYERS, 6), dtype=bool)
        alive2[3, :] = False                       # 3 全程是死的
        out2 = frame_labels(
            [{"i": 2, "a": 0, "v": 3}, {"i": 4, "a": 0, "v": 5}],
            alive2,
            np.array([0]),
        )
        assert out2["next_death"][0] == 5, out2["next_death"]

        # 第 0 帧的事件不算"之后"：i == t 必须被排除。
        out3 = frame_labels([{"i": 0, "a": 0, "v": 1}], np.ones((N_PLAYERS, 3), bool), np.array([0]))
        assert out3["next_death"][0] == NO_EVENT and out3["next_kill"][0] == NO_EVENT

        # 同一帧的两条事件（互换）：按列表顺序取"先出现"的那条 —— 这是确定性的来源，
        # 靠的是 `sorted(..., key=i)` 的稳定性。
        out4 = frame_labels(
            [{"i": 2, "a": 0, "v": 1}, {"i": 2, "a": 1, "v": 0}],
            np.ones((N_PLAYERS, 4), bool),
            np.array([0]),
        )
        assert out4["next_death"][0] == 1 and out4["next_kill"][0] == 0, out4

        # 回合结束瞬间的击杀：`i = n_frames`（最后一帧之后，没有对应的帧）。
        # 它必须参与每一行的"下一个"，并且让受害者不算活到回合结束。
        alive5 = np.ones((N_PLAYERS, 6), dtype=bool)
        out5 = frame_labels(
            [{"i": 3, "a": 1, "v": 2}, {"i": 6, "a": 1, "v": 4}],
            alive5,
            np.array([0, 3, 5]),
        )
        assert out5["next_death"].tolist() == [2, 4, 4], out5["next_death"].tolist()
        assert out5["next_kill"].tolist() == [1, 1, 1], out5["next_kill"].tolist()
        assert not out5["alive_end"][:, 4].any(), "最后一帧之后死的人还算活着"
        assert out5["alive_end"][:, 0].all()
        # 那一枪的击杀者对所有行都"还能拿到击杀"
        assert out5["can_kill"][:, 1].all() and out5["can_kill"].sum(axis=1).tolist() == [1, 1, 1]
        print("[1/3] frame_labels：标签语义 OK")

    def _check_allocate() -> None:
        lengths = [640, 640, 320]                  # 10 s + 10 s + 5 s = 25 s → 25 帧
        picks = _allocate(lengths, "a.csz", density=1.0)
        # 抽帧是**整局均匀**的，所以每个回合拿到几帧是随机的（期望 = 时长/density）：
        # 这里只钉"总数"和"每帧唯一 / 不越界"，不钉每回合的个数。
        assert sum(len(p) for p in picks) == 25, [len(p) for p in picks]
        assert all(np.all(np.diff(p) > 0) for p in picks), "同一回合里不能重复抽同一帧"
        assert all(p.size == 0 or (p.min() >= 0 and p.max() < n)
                   for p, n in zip(picks, lengths))
        bases = np.cumsum([0] + lengths[:-1])
        allp = np.concatenate([p + b for p, b in zip(picks, bases)])
        assert allp.size == np.unique(allp).size == 25, "跨回合抽到了同一帧"
        # 可复现：同名字 → 同一批；换种子 → 换一批
        again = _allocate(lengths, "a.csz", density=1.0)
        assert all(np.array_equal(a, b) for a, b in zip(picks, again))
        other = _allocate(lengths, "a.csz", density=1.0, seed=1)
        assert not all(np.array_equal(a, b) for a, b in zip(picks, other)), "种子没生效"
        # density 变大 → 样本变少（1600 帧 / 2 s / 64 = 12.5 → 12）；整局只有 1 帧时至少 1 个
        assert sum(len(p) for p in _allocate(lengths, "a.csz", density=2.0)) == 12, \
            sum(len(p) for p in _allocate(lengths, "a.csz", density=2.0))
        assert sum(len(p) for p in _allocate([1, 0], "a.csz", density=100.0)) == 1
        assert sum(len(p) for p in _allocate([], "a.csz")) == 0
        assert sample_offsets([3, 0, 2]).tolist() == [0, 3, 3, 5]
        print("[2/3] _allocate：density / 可复现 / 边界 OK")

    def _check_dataset(out: Path) -> int:
        """逐**文件**读（不是逐样本）：同一个 npz 的 K 行是同一回合的时间序列，只有一次
        性拿到整段，才能检查"随时间单调"这类跨行不变式。"""
        total = 0
        first = None
        n_late = 0           # 最后一帧之后还有人死的文件数（回合结束瞬间那一枪）
        for split in SPLITS:
            if not any((out / split).glob("*.npz")):
                # 空 split 是**合法产物**（整张图都分到别的 split 去了），
                # `SampleDataset` 在空目录上会直接抛 FileNotFoundError。
                print(f"[3/3] {split:<5} 空（合法，跳过）")
                continue
            ds = SampleDataset(out / split, task="spatial_prediction")
            n = len(ds)
            total += n
            counts = sample_counts(ds.samples)
            assert n == sum(counts) == ds.task.n_samples(ds.samples)
            assert n == 0 or len(ds.samples) > 0
            base = np.cumsum([0] + counts[:-1])
            if first is None and n:
                first = ds[0]
            print(f"[3/3] {split:<5} {len(ds.samples):>3} 个文件 / {n:>5} 个样本")
            for j, path in enumerate(ds.samples):
                m = _COUNT_RE.search(path.stem)
                if m is None or int(m.group(1)) != counts[j]:
                    raise AssertionError(f"{path.name}: 文件名里的 _n 和实际行数对不上")
                k = counts[j]
                with np.load(path) as z:
                    a = {key: z[key] for key in z.files}
                where = f"{split}/{path.name}"

                # 形状 / dtype（写盘时就要对，不是训练时才发现）
                assert a["ctx_player"].shape == (k, N_PLAYERS, 20), where
                assert a["ctx_player"].dtype == np.float32, where
                assert a["ctx_player_mask"].shape == (k, N_PLAYERS), where
                assert a["ctx_player_mask"].dtype == bool, where
                assert a["ctx_player_team"].shape == (k, N_PLAYERS), where
                assert a["ctx_player_weapon"].shape == (k, N_PLAYERS), where
                assert a["ctx_player_inv"].shape == (k, N_PLAYERS, N_WEAPONS), where
                assert a["ctx_player_inv"].dtype == np.uint8, where
                assert a["obstruction"].shape == (k, N_PLAYERS, N_PLAYERS), where
                assert a["ctx_projectile"].shape == (k, _PROJ_MAX, 4), where
                assert a["ctx_projectile_id"].shape == (k, _PROJ_MAX), where
                assert a["global_feat"].shape == (k, 5), where
                for key in ("seconds", "frame_index", "ct_win", "next_death", "next_kill"):
                    assert a[key].shape == (k,), (where, key, a[key].shape)
                for key in ("alive_end", "can_kill"):
                    assert a[key].shape == (k, N_PLAYERS), (where, key)
                assert a["map_id"].shape == () and a["round_id"].shape == (), where

                # 输入：aim 的契约（[0,1]、死人 weapon = -1、队伍 0/1/-1）
                p = a["ctx_player"]
                assert 0.0 <= p.min() and p.max() <= 1.0, (where, p.min(), p.max())
                mask = a["ctx_player_mask"]
                team = a["ctx_player_team"]
                assert set(np.unique(team).tolist()) <= {TEAM_CT, TEAM_T, TEAM_UNKNOWN}, where
                assert (a["ctx_player_weapon"][~mask] == -1).all(), f"{where}: 死人带武器"
                inv = a["ctx_player_inv"]
                assert set(np.unique(inv).tolist()) <= {0, 1}, f"{where}: 背包不是 0/1"
                assert not inv[~mask].any(), f"{where}: 死人背包非空"
                # 手里那把一定也在背包里（这条是"扣掉 held"的前提，见 `inventory_at`）
                held = a["ctx_player_weapon"].astype(np.int64)
                armed = mask & (held >= 0)
                row_idx = np.arange(k)[:, None]
                assert inv[row_idx, np.arange(N_PLAYERS)[None, :],
                           np.where(armed, held, 0)][armed].all(), \
                    f"{where}: 手里那把不在背包里"
                assert set(np.unique(a["obstruction"]).tolist()) <= {0, 1, 2}, where
                proj, pid = a["ctx_projectile"], a["ctx_projectile_id"]
                assert 0.0 <= proj.min() and proj.max() <= 1.0, (where, proj.min(), proj.max())
                assert (pid >= -1).all(), where
                assert (proj[pid < 0] == 0.0).all(), f"{where}: 空槽的道具不是全 0"

                # 行 = 帧，必须按时间**严格**递增（抽帧时排过序，写盘顺序不能乱）
                fi, sec = a["frame_index"], a["seconds"]
                assert fi.dtype == np.int32 and sec.dtype == np.float32, where
                assert (np.diff(fi.astype(np.int64)) > 0).all(), f"{where}: frame_index 不是递增"
                assert (np.diff(sec.astype(np.float64)) > 0).all(), f"{where}: seconds 不是递增"
                assert np.allclose(np.diff(sec.astype(np.float64)) % (1.0 / TICK_HZ), 0.0,
                                   atol=1e-6), f"{where}: seconds 不是 1/64 的整数倍"

                # 标签：一个回合内恒定 / 单调
                assert a["ct_win"].all() or not a["ct_win"].any(), f"{where}: ct_win 不恒定"
                end = a["alive_end"]
                assert (end == end[0]).all(), f"{where}: alive_end 随时间变了"
                # `alive_end` 是**回合结束**时的存活，不是最后一帧样本的存活：抽到的最后
                # 一帧之后可能还有几十帧（含回合结束瞬间那一枪），所以只要求
                # end ⊆ mask[-1]（人死了不会复活）。
                assert not (end[0] & ~mask[-1]).any(), f"{where}: alive_end 和最后一帧的 alive 矛盾"
                if (mask[-1] & ~end[0]).any():
                    n_late += 1
                can = a["can_kill"]
                assert not (can & ~mask).any(), f"{where}: 死人还能拿击杀"
                assert (can[1:] <= can[:-1]).all(), f"{where}: can_kill 不是单调收缩"

                nd, nk = a["next_death"].astype(np.int64), a["next_kill"].astype(np.int64)
                # 11 类：0..9 是玩家，10 是"没有下一次"。出现 -1 或 11 都是写错了。
                assert ((nd >= 0) & (nd <= NO_EVENT)).all(), where
                assert ((nk >= 0) & (nk <= NO_EVENT)).all(), where
                rows = np.arange(k)
                has_d = nd < NO_EVENT
                assert mask[rows[has_d], nd[has_d]].all(), f"{where}: 下一个死者当帧不活着"
                assert (~end[rows[has_d], nd[has_d]]).all(), f"{where}: 下一个死者最后还活着"
                # 没有"下一个死者" ⟺ 当帧活着的人全都活到了最后
                assert (nd == NO_EVENT).tolist() == list(map(bool, (mask == end).all(axis=1))), \
                    f"{where}: next_death 的 10 和 alive_end 不一致"
                has_k = nk < NO_EVENT
                assert mask[rows[has_k], nk[has_k]].all(), f"{where}: 下一个击杀者当帧不活着"
                assert can[rows[has_k], nk[has_k]].all(), f"{where}: 下一个击杀者不在 can_kill 里"
                assert (nk == NO_EVENT).tolist() == list(map(bool, ~can.any(axis=1))), \
                    f"{where}: next_kill 的 10 和 can_kill 不一致"
                assert a["next_death"].dtype == np.int8 and a["next_kill"].dtype == np.int8

                # `read_sample` 的扁平下标 ⇄ (文件, 行) 映射：抽查首尾两行
                for off, want in ((0, int(fi[0])), (k - 1, int(fi[-1]))):
                    got = ds[int(base[j]) + off]["frame_index"]
                    assert int(np.asarray(got).reshape(-1)[0]) == want, \
                        f"{where}: read_sample 映射错了（off={off}）"

        # 单独确认 `to_torch` 之后的 dtype（训练侧真正看到的）：抽一个样本就够了，
        # 形状/dtype 的规则和内容无关。
        if n_late:
            print(f"[3/3] 最后一帧之后还有人死的文件：{n_late} 个"
                  f"（回合结束瞬间那一枪，`alive_end` 应记成 False）")
        if first is None:
            return total
        for key in ("ctx_player_mask", "ct_win"):
            assert np.asarray(first[key]).dtype == np.dtype(bool), key
        for key in ("ctx_player_team", "ctx_player_weapon", "next_death", "next_kill"):
            assert np.asarray(first[key]).dtype.kind == "i", (key, first[key])
        # 标量样本在 `to_torch` 之后是 (1,)（`np.ascontiguousarray` 会把 0 维强行
        # 变 1 维），不是 ()。训练侧用 `.squeeze(-1)` / `[:, 0]` 取。
        for key in ("seconds", "frame_index", "ct_win", "next_death", "next_kill"):
            assert np.asarray(first[key]).size == 1, (key, first[key].shape)
        return total

    TASK_NAME = Path(__file__).stem
    CSZ_DIR = REPO_ROOT / "data" / "demos" / "csz"
    OUT = REPO_ROOT / "tmp" / "training_data" / TASK_NAME

    _check_labels()
    _check_allocate()

    shutil.rmtree(OUT, ignore_errors=True)          # 每次从零开始，不用手动清
    counts = build(CSZ_DIR, OUT, task=TASK_NAME, limit=1, workers=1)
    print(f"[冒烟] 造出来 {counts}")

    n_samples = _check_dataset(OUT)
    # `build()` 返回的是**文件**数（一个回合一个 npz），不是样本数 —— 这两个数只在
    # 每个回合恰好抽到 1 帧时才会相等，别拿来互相校验。
    n_files = sum(1 for split in SPLITS for _ in (OUT / split).glob("*.npz"))
    assert sum(counts.values()) == n_files, (counts, n_files)
    print(f"[冒烟] 共 {n_files} 个文件 / {n_samples} 个样本，全部通过")
