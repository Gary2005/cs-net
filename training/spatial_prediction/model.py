"""spatial_prediction 模型：**一帧 → 一组 token → 一个 transformer → 5 个 head**。

    tokens  = [CLS_WIN, CLS_KILL, map, global, player×10, projectile×P]        S = 14 + P
                └──── 4 个语境 token ────┘  └ 活人才是 key ┘ └ 非空槽才是 key ┘
                              │
                    S 个 token 全连接自注意力 ×L 层（player↔player 之间加遮挡 bias）
                              │
      ┌───────────┬───────────┼──────────────────┬──────────────────┐
    CLS_WIN    player×10   player×10          player×10 + CLS_KILL
      │            │           │                    │
    ct_win     alive_end    can_kill          next_death / next_kill
    （2 类）    （逐人 2 类） （逐人 2 类）       （11 类，第 11 类 = 没有下一次）

一帧就是一个样本，没有时间轴：batch 里每条样本**形状完全一样**（10 个玩家槽位 + 32 个道具
槽位都是定长），所以没有 padding、没有 `frame_mask`，`collate_spatial` 纯 stack。

## 为什么这一版要帧内注意力，而 aim 不要

aim 的帧内只有两个查询（A、B），答案是"这两个人里谁死"，`cat(x_A, x_B)` 的拼接顺序就把
"谁是谁"说清楚了，注意力在那里买不到东西（见 `training/aim/model.py` 的"为什么不要帧内
注意力"）。

这个任务不一样：**没有主角**。要答的是"10 个人里谁下一个死"、"CT 会不会赢"、"谁之后还能
拿到击杀"，全是**跨玩家的比较**，而且答案要同时看所有玩家和所有道具（一颗烟落在包点上，
对 10 个人的影响各不相同）。这正是自注意力擅长的事：每个 token 在每一层都能重新汇总
"整帧现在是什么局面"，比较在注意力里直接发生，不用先池化成一个向量再解。

代价是 token 里有大量"不该被看见"的东西（死人、空道具槽），所以 mask 是这份代码里最需要
小心的地方 —— 见下面两节。

## token 从哪来

    token         数量   内容                                            来自
    CLS_WIN        1     可学习向量（`cls_win`）                          —
    CLS_KILL       1     可学习向量（`cls_kill`）                          —
    map            1     `map_emb(map_id)`                                每文件一个标量
    global         1     `global_proj(Fourier(global_feat[5]))`           炸弹 xyz / 安放秒 / 回合秒
    player        10     `player_proj(Fourier(ctx_player[20]))`           轨迹 / 朝向 / 速度 / 血量
                          `+ weapon_emb(+1) + team_emb(+1)`               / 武器 / 队伍
    projectile     P     `proj_proj(Fourier(ctx_projectile[4]))`           道具 xyz + 已飞/已生效秒数
                          `+ proj_type(+1)`

每个非 CLS token 再加一个**类型 embedding**（`type_emb`，4 种）：token 的种类不靠位置编码
表达（这份模型**完全没有位置编码** —— 10 个玩家和 P 个道具本来就是无序集合），而是靠这个
可学习向量 + 各自的内容投影区分。

`FourierFeatures` 和 player / projectile / global 三套投影是**从 `training/aim/model.py`
直接 import 复用的**（同一个类，不是抄一份）：两边喂的是同一套 `ctx_player` / `ctx_projectile`
/ `global_feat` 列，列序和 `[0,1]` 归一化完全一致（`aim.py::AIM_FEATURES`）。武器表
（43）、地图表（8）、道具类型表（7）也是同一张表，所以常量一起 import 过来。

### 每帧多少个 token（实测）

    alive/frame   均值 7.62（p5 3 / p50 8 / p95 10）     ← 10 个玩家槽位里活着的
    proj/frame    均值 3.19（p95 9 / max 18），16.4% 的帧一个道具都没有
    S = 14 + P    均值 17.2（p95 23 / max 31）

`P` 取**本 batch 内**的最大占用数（`_projectile_tokens` 把非空槽挤到前面再切），所以同一帧
的表示和 batch 里别的样本带多少道具无关。数据里道具槽本来就是前缀排布的（25,224 行里
0 例外），挤一次只是给"以后乱序写盘"兜底。

## mask：谁参与注意力

    token         是 key 的条件          为什么
    CLS_WIN        恒是                  —
    CLS_KILL       恒是                  —
    map            恒是                  —
    global         恒是                  —
    player i       `ctx_player_mask[i]`   死人没有任何可观测状态；磁盘上死人那一行**不保证是
                                          0**（实测 `|feat|` 均值 0.343），不 mask 掉就是往模型
                                          里喂残留位置值
    projectile q   `ctx_projectile_id[q] >= 0`   空槽的 `[x y z t]` 全是 0，是"原点"而不是
                                          "这里没有东西"，不 mask 掉模型会以为原点长期有个道具

实现是一张 `(B, H, S, S)` 的加性掩码（`_mask_bias`），**8 层共用同一份**：被 mask 的 key
那一列填 `MASK_NEG`。**每一行至少有 4 个语境 token 是开的**，所以不存在"整行全被 mask →
softmax 全零"的 nan 路径。死人的 token 照样算（作为 query 它还在跑），只是它的输出没人读 —— 5 个 head
都用 `ctx_player_mask` 选活人。

磁盘上还有两条恒等式（实测 25,224 帧里 0 例外），这份代码依赖它们：
`obstruction` 的对角恒 0、死人的行和列恒 0。

## 遮挡：作为注意力 bias，不是 token

`obstruction[i][j]` 是**有向**的（i 的眼睛看 j 的身体），实测 1.25% 的活人对两个方向不一致。
它进注意力的方式是给 `player_i → player_j` 这个位置加一个**可学习的加性 bias**：

    bias[l][h][i][j] = ob_bias[l][h][class(ob[i][j])]   class: 0 通 / 1 烟 / 2 墙 / 3 自己

每层每个头一组 4 个标量，一共 `layers × n_heads × 4` = 512 个参数 —— 让每层的每个头自己决定
"墙有多挡视线"。**逐层独立，不是 8 层共用一张表**。共用（早先的写法）有两个问题：

* 同一个 bias 在 8 层各加一次，有效偏置被放大 L 倍：这 64 个数就成了全网络唯一的低维公共开
  关，一旦长大，8 层的注意力会**同时**被推向饱和，等于在 loss 上造出一条特别尖的方向。
* 各层需要的量级本来就不一样：浅层 LayerNorm 后的权重和 logits 都小，一个 `b = 2` 就把
  它的 softmax 压死了；深层 logits 已经很大，同样的 2 几乎没有影响。一份绝对量喂 8 层是错
  配的。

实测（活人有向对，去对角）：

    通 13.30%   烟 0.70%   墙 86.01%          ← 和 aim 那份数据的 10.6% / 2.1% / 87.3% 不同，
                                                这张图/这批 demo 看得见的时候更多

为什么是 bias 而不是"给每条边一个 token"或者"给每一对单独一个向量"：

* **顶点只有 10 个，边有 90 条**。把边做成 token 会让 S 从 17 涨到 107，而 90 条边里绝大
  多数是"墙"，信息量只有 3 个离散档次 —— 用一个 `(layers, n_heads, 4)` 的 bias 表达同样的
  东西，参数少 4 个数量级。
* **bias 是加在 logits 上的，不占表示维度**。注意力分数本来就是"这个 key 对这条 query 有
  多重要"，遮挡正好是这个语义；换成乘性的门（`score * 0`）会连"烟雾弹后面那个人完全看
  不见，但他自己会动、有枪声"这类间接证据一起抹掉。
* **按头分开**：单个 bias 只能表达"墙一律降权"，分头之后可以出现"某个头专门看被墙挡住的
  人"（绕后/穿烟）这种情况，而这正是这个任务里最值钱的信息之一。初始化全 0 —— 一开始等价
  于没有遮挡 bias，让模型自己长出来（和 ALiBi 的写法一致）。

`class = 3`（自己）是必要的：对角线上 `ob[i][i]` 恒 0，读作"通"，不单独分出来会让"自己看
自己"和"我看得见另一个没被挡的人"共用同一个 bias。死人的行列本来就是 0，但它们整个 token
已经被 mask 掉了，所以不用管。

## 5 个 head

全部是"2 层 MLP（`d → head_hidden → 1`）+ GELU + dropout"，只是读哪个 token 不一样：

    head            读的 token              输出         loss
    ct_win          CLS_WIN                 (B,)  logit   BCE
    alive_end       每个活着的 player       (B,10) logit 逐人 BCE，mask 掉死人
    can_kill        每个活着的 player       (B,10) logit 逐人 BCE，mask 掉死人
    next_death      player ×10 + CLS_KILL   (B,11) logit  CE（11 类）
    next_kill       player ×10 + CLS_KILL   (B,11) logit  CE（11 类）

### `next_death` / `next_kill` 为什么是"10 个玩家 + 1 个 CLS_KILL"

`[CLS_KILL]` 那个位置**就是 `NO_EVENT = 10` 这一类**：它的 logit 和 10 个玩家的 logit 拼在
一起做 11 类 softmax，谁最大就是"下一个死/杀的是谁"，`[CLS_KILL]` 最大就是"之后没有了"。
这样"谁"和"有没有"是**同一个 softmax 里的竞争**，不需要另开一个"会不会发生"的二分类头再
和"是谁"相乘 —— 两条独立的 1-vs-rest 信号在"末段大家都快死了"的区域会互相打架。

两个 head 共用同一个 `[CLS_KILL]` 读出位置、但**各有一组 MLP 参数**（`head_nd_event` /
`head_nk_event`）：`next_death` 和 `next_kill` 是两条独立的监督信号（见数据侧文档），
"之后没人死"和"之后没人杀"不是一回事（最后一个 CT 活着但不会再开枪）。

死人的列在 CE 之前被填成 `finfo.min`：它们本来就不可能是答案（造数侧保证了
`next_*` 指向的人当前帧还活着），不填的话 softmax 会把概率分给一份"从来没发生过的未来"，
白白摊薄 10 个真候选。`10 = NO_EVENT` 实测只占 **9.0%**（`next_death`）/ **9.5%**
（`next_kill`），所以这 11 类**接近均匀**，`ln 11 = 2.398` 就是纯瞎猜的 CE 基线
（旧的那份 4 demo 抽样里 `NO_EVENT` 占 11.0%、还有"永远猜 P0"能到 1.85 的捷径，这批
13,451 个 demo 的数据没有那个偏斜：`next_kill` 最热门的类也只占 10.1%）。

### `alive_end` / `can_kill` 为什么必须 mask 死人

当前帧已经死的人这两个标签恒 False，这一半分靠 `ctx_player_mask` 一眼就能推出来；掩掉之后
loss 才只奖励"对活人的判断"。实测（全库 24.2M 行 / 242M 人次）：`alive_end` 均值
**0.320**、`can_kill` 均值 **0.314**，而只看活人的话正负比大约是 **42 : 58** 和
**41 : 59** —— 两个都是偏负的，别拿"50% 基线"去比。

## 没喂进模型的字段

* `seconds` / `frame_index` / `round_id`：回合内的绝对时间/序号。它们**不是泄漏**（推理时
  你确实知道现在是第几秒），但是很强的捷径：回合末段几乎所有人都死了、`next_* = 10`，
  模型可以只读表盘就压低 loss 而不看画面。`frame_index` / `round_id` 是"回查原始数据"用
  的（`round_id` 跨 demo 没有可比性）。
  ⚠ 不过 `global_feat` 第 5 列**就是** `回合时间/200`，是数据侧一起给的；想验证"模型有没
  有只读表盘"，就把它置 0 重训再做 ablation，而不是指望这份代码里不出现它。
* 玩家**槽位编号**（0..9 的身份）：这 4 个 demo 是同一场比赛的话槽位是固定的人，加
  `slot_emb` 会把"P0 更容易拿击杀"这种先验直接送进去（旧抽样里能到 1.85 的 CE）。这批
  13,451 个 demo 上标签已经接近均匀，先验这条捷径不值钱，但它仍然是"记住了选手"而不是
  "看懂了局面"，所以不加。
* `ctx_player` 第 15 列（team 的浮点版）是冗余的：单独给了 `ctx_player_team`（-1/0/1），
  用 embedding 表达比 [0,1] 的浮点列干净 —— 尤其是 -1（未知）会被 clip 成 0 和 CT 混掉。

## 成本

    token 数 S ≈ 17（p95 23），层数 L，宽度 d：
        每层 12 d² × S  MAC（注意力 4d² + MLP 8d²），乘 L
        一帧 ≈ L × S × 12 d² = 6 × 17 × 12 × 512² ≈ 0.32 G MAC（前向）

`d_model=512` / `layers=6` 时 **20.12 M 参数**（`train.py` 启动时打真实值）。它比 aim 的
67 M 小，是因为这里没有 257 个时间步：**一个样本就是一个 token 序列**，而 aim 一个样本是
257 个这样的序列。

实测（RTX 3090 24 G，bf16，数据已经在显存里，纯前向+反传+`AdamW.step`）：

    B=512    55.1 ms/步   9,294 行/s   峰值 2.11 GiB      只前向 17.2 ms → 29,852 行/s
    B=1024   101 ms/步   10,137 行/s   峰值 3.88 GiB
    B=2048   206 ms/步    9,941 行/s   峰值 8.19 GiB
    B=512 fp32  115 ms/步  4,435 行/s（bf16 是它的 2.1 倍）

加成 **2.1 倍**就到头了（55 ms 里有一半是小 kernel 的启动开销和 LayerNorm，不是 FLOP），
所以 `batch_size` 从 512 抬到 1024 只值 9%，到 2048 反而白占显存。**加容量要加在
`layers` / `d_model` 上，不是 batch 上** —— 这两个才是真正乘在每个 token 上的。

⚠ 一帧的计算量小到"数据比模型贵"：24.2M 行 / 9.3k 行/s ≈ **43 分钟/epoch** 全是 GPU 时间，
所以数据管线必须能喂满 9.3k 行/s，否则每个 epoch 都在等盘（`dataset.py` 的
`ShuffleBufferSampler` 就是为这件事存在的：实测喂得动）。


⚠ 注意力是**全连接、无因果遮罩**的：`[CLS_WIN]` 看到所有玩家，玩家也看到 `[CLS_WIN]`。
这不是自回归模型，"看到别人的答案"不成立 —— 5 个 head 的输出都是同一帧的标签，没有先后。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# aim 的帧内特征编码器 + 三张 id 表。**同一个模块，不是复制**：两边喂的是同一套列
# （`aim.py::AIM_FEATURES` / `PROJECTILE_FEATURES` / `constants.WEAPON_NAMES`）。
from training.aim.model import (
    FourierFeatures,
    N_GLOBAL,
    N_MAPS,
    N_PLAYER_FEAT,
    N_PROJ_FEAT,
    N_PROJ_TYPE,
    N_WEAPONS,
)

# ═══════════════════════════════════════════════════════════════════════════
# 常量：和 `training/spatial_prediction/dataset.py` 的磁盘布局对齐
# ═══════════════════════════════════════════════════════════════════════════

#: 玩家槽位数（死人也占位）。
N_PLAYERS = 10

#: `next_death` / `next_kill` 的"这一帧之后没有下一次" = 10，所以是 11 类。
NO_EVENT = N_PLAYERS

#: `ctx_player_team` 的取值 -1 / 0 / 1 → embedding 下标 0 / 1 / 2。
N_TEAMS = 3

#: 遮挡的类别：0 通 / 1 烟 / 2 墙 / 3 自己（对角）。死人的行列是 0，但它们整个 token 被
#: mask 掉了，所以不需要第 5 类。
N_EDGE_CLASS = 4
EDGE_SELF = N_EDGE_CLASS - 1

#: token 类型：player / map / global / projectile。CLS 两个各自是一个独立参数，不占这里。
N_TOKEN_TYPE = 4
T_PLAYER, T_MAP, T_GLOBAL, T_PROJ = 0, 1, 2, 3

#: 序列开头的 4 个固定位置，顺序是 `[CLS_WIN, CLS_KILL, map, global]`，之后才是 10 个玩家
#: （固定占 10 格，死人也占位）和 P 个道具。所以玩家 token 恒在 `N_PREFIX..N_PREFIX+10`，
#: 5 个 head 里只有这一步下标是写死的。
IDX_CLS_WIN, IDX_CLS_KILL = 0, 1
N_PREFIX = 4

#: 注意力里"这个 key 不可见"的加性常数。**必须是有限值**：`finfo(dtype).min` 之后还要和
#: 遮挡 bias 相加（`-3.39e38 + (-10)` 在 bf16 里就溢出成 `-inf` 了），而 `-1e4` 在
#: fp32/fp16/bf16 里都安全，`exp(-1e4 - max)` 又精确下溢成 0 —— 只要 logits 不到 1e4 量级
#: （实测 O(10)）它就和"完全看不见"等价。
MASK_NEG = -1.0e4

# 上面那几个 `N_*` 是从 aim 借的（同一套提取代码），这里再钉一遍和本任务磁盘布局一致。
# 形状真变了说明两边的数据契约已经错开，早点炸比训到一半再炸好。
assert (N_PLAYER_FEAT, N_PROJ_FEAT, N_GLOBAL) == (20, 4, 5), (
    f"aim 的特征列数变了：player={N_PLAYER_FEAT} proj={N_PROJ_FEAT} global={N_GLOBAL}，"
    f"和 dataset._ROW_FIELDS 的 (20, 4, 5) 不一致"
)
assert N_TEAMS == 3 and NO_EVENT == 10


@dataclass
class SpatialConfig:
    """模型超参。`config.yaml` 的 `model:` 段直接 unpack 成它。"""

    d_model: int = 512
    n_heads: int = 8
    #: 全连接自注意力的层数（没有因果遮罩）。S ≈ 17 时每层 `12 d² × S`。
    layers: int = 6
    #: FFN 隐层 = `d_model * ffn_mult`。
    ffn_mult: float = 4.0
    dropout: float = 0.1
    #: 5 个 head 的隐层宽度（`d_model → head_hidden → 1`）。
    head_hidden: int = 256
    #: `[0,1]` 小特征的 Fourier 展开：每通道频率个数（见 `FourierFeatures`）。
    player_freq: int = 8
    proj_freq: int = 8
    global_freq: int = 8
    fourier_f_min: float = 1.0
    #: 多小的差异还能被线性分辨：`f=128` 时 0.004 的差异就有 ~90° 相位差
    #: （位置列 ≈ 10 HU、hp ≈ 0.4%）。
    fourier_f_max: float = 128.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} 不能被 n_heads={self.n_heads} 整除")
        if self.layers < 1:
            raise ValueError(f"layers 必须 >= 1，收到 {self.layers}")
        if self.head_hidden < 1:
            raise ValueError(f"head_hidden 必须 >= 1，收到 {self.head_hidden}")


class SelfAttention(nn.Module):
    """全连接多头自注意力 + 加性 mask/bias（唯一的使用者就是 `Block`）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """`x (B,S,d)`，`bias (B,H,S,S)` 加性（已被 mask 的 key 是 `MASK_NEG`）。"""
        b, s, _ = x.shape
        q, k, v = (
            self.qkv(x)
            .view(b, s, 3, self.n_heads, self.d_head)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        # autocast 下 q/k/v 是 bf16，而 SDPA 要求 attn_mask 和 query 同 dtype。逐层那张
        # `(B,H,S,S)` 已经是目标 dtype 了（`_mask_bias` 直接按 dtype 建、`_edge_bias` 也是），
        # 这里的判断只在手工传错 dtype 时兜底。
        if bias.dtype != q.dtype:
            bias = bias.to(q.dtype)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.proj(y.transpose(1, 2).reshape(b, s, -1))


class Block(nn.Module):
    """pre-LN 残差块：`x + attn(ln(x))`，再 `x + mlp(ln(x))`。"""

    def __init__(self, cfg: SpatialConfig) -> None:
        super().__init__()
        d = cfg.d_model
        hidden = int(d * cfg.ffn_mult)
        self.norm1 = nn.LayerNorm(d)
        self.attn = SelfAttention(d, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), bias)
        return x + self.mlp(self.norm2(x))


class Head(nn.Module):
    """`d → head_hidden → 1`，GELU + dropout。

    两层而不是一层：第一层是"在 d 维里挑出这个 head 关心的那几个方向"，第二层才是"把这些
    证据加权成一条 logit"。单层线性等价于在最粗的表示上直接取一个投影方向，头与头之间只能
    靠不同方向错开，没法各自做"合取"（"他血少 **并且** 在包点里"）。
    """

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(..., d)` → `(...)`。"""
        return self.net(x).squeeze(-1)


class SpatialModel(nn.Module):
    """一帧 → `[CLS_WIN, CLS_KILL, map, global, player×10, projectile×P]` → 5 个 logit 张量。"""

    def __init__(self, cfg: SpatialConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or SpatialConfig()
        d = cfg.d_model

        # ---- token 的输入投影（player / projectile / global 三套复用 aim 的写法）----
        self.fourier_player = FourierFeatures(
            N_PLAYER_FEAT, cfg.player_freq, cfg.fourier_f_min, cfg.fourier_f_max
        )
        self.fourier_proj = FourierFeatures(
            N_PROJ_FEAT, cfg.proj_freq, cfg.fourier_f_min, cfg.fourier_f_max
        )
        self.fourier_global = FourierFeatures(
            N_GLOBAL, cfg.global_freq, cfg.fourier_f_min, cfg.fourier_f_max
        )
        self.player_proj = nn.Linear(self.fourier_player.out_dim, d)
        self.global_proj = nn.Linear(self.fourier_global.out_dim, d)
        self.proj_proj = nn.Linear(self.fourier_proj.out_dim, d)
        self.map_emb = nn.Embedding(N_MAPS, d)
        #: 道具类型：`-1`（空槽）落到槽位 0。
        self.proj_type = nn.Embedding(N_PROJ_TYPE + 1, d)
        #: 武器：`-1`（无武器 / 已死）落到槽位 0，`padding_idx=0` 让它恒为 0 向量。
        self.weapon_emb = nn.Embedding(N_WEAPONS + 1, d, padding_idx=0)
        #: 队伍：`-1`（未知）→ 0，CT → 1，T → 2。
        self.team_emb = nn.Embedding(N_TEAMS, d)
        #: token 类型。4 个可学习向量，用 `weight[i]` 直接取（不建新的 index tensor）。
        self.type_emb = nn.Embedding(N_TOKEN_TYPE, d)

        # ---- 两个 CLS ----
        #: 初始化 0.02（同 GPT-2 的 `wpe` / aim 的 `time_emb`）：一开始接近 0，让注意力的
        #: 第一层先看到"真实的局面"，CLS 自己随训练长出来。
        self.cls_win = nn.Parameter(torch.zeros(d))
        self.cls_kill = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.cls_win, std=0.02)
        nn.init.normal_(self.cls_kill, std=0.02)

        # ---- 遮挡 bias：`(layers, H, 4)`，初始 0（= 一开始没有遮挡先验）----
        #: **逐层独立**：浅层和深层需要的最优量级不一样，共用一张表等于给全网络一个低维公共
        #: 开关（见文件头"遮挡"一节）。8×16×4 = 512 个数，相对 103 M 是免费的。
        self.ob_bias = nn.Parameter(torch.zeros(cfg.layers, cfg.n_heads, N_EDGE_CLASS))

        # ---- 主干 ----
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm_f = nn.LayerNorm(d)

        # ---- 5 个 head（`next_*` 拆成"玩家那一半"+"NO_EVENT 那一半"）----
        hh = cfg.head_hidden
        self.head_win = Head(d, hh, cfg.dropout)
        self.head_alive = Head(d, hh, cfg.dropout)
        self.head_cankill = Head(d, hh, cfg.dropout)
        self.head_nd_player = Head(d, hh, cfg.dropout)
        self.head_nd_event = Head(d, hh, cfg.dropout)
        self.head_nk_player = Head(d, hh, cfg.dropout)
        self.head_nk_event = Head(d, hh, cfg.dropout)

    # ────────────────────────────────────────────────── token 构造

    def _projectile_tokens(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(B,P,d)` 的道具 token + `(B,P)` 的"这个槽真有东西"标记。

        `P` 取本 batch 的最大占用数（实测均值 3.19 / p95 9 / max 18）。数据里道具槽本来就是
        前缀排布的（25,224 行 0 例外），先挤再切只是给"以后乱序写盘"兜底 —— 直接
        `[:, :P]` 在乱序时会**静默**丢掉非空槽。
        """
        pid = batch["ctx_projectile_id"].long()               # (B,32)
        feat = batch["ctx_projectile"]                        # (B,32,4)
        filled = pid >= 0
        n_prop = max(1, int(filled.sum(-1).max()))
        # 空槽的 id 是 -1、特征全 0，稳定排序后它们必然落在后面。
        order = torch.argsort((~filled).to(torch.int8), dim=-1, stable=True)[:, :n_prop]
        pid = pid.gather(1, order)                            # (B,P)
        feat = feat.gather(1, order.unsqueeze(-1).expand(-1, -1, N_PROJ_FEAT))
        vec = (
            self.proj_proj(self.fourier_proj(feat))
            + self.proj_type(pid + 1)
            + self.type_emb.weight[T_PROJ]
        )
        return vec, pid >= 0

    def _tokens(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """→ `tokens (B,S,d)`、`valid (B,S) bool`（False = 这个 token 不当 key）。"""
        b = batch["ctx_player_mask"].shape[0]
        d = self.cfg.d_model
        alive = batch["ctx_player_mask"].bool()                       # (B,10)

        # 玩家：自己的 20 列 + 武器 + 队伍 + 类型。
        pv = (
            self.player_proj(self.fourier_player(batch["ctx_player"]))
            + self.weapon_emb(batch["ctx_player_weapon"].long() + 1)
            + self.team_emb(batch["ctx_player_team"].long() + 1)
            + self.type_emb.weight[T_PLAYER]
        )                                                             # (B,10,d)

        # map / global：每帧各一个。
        mp = (self.map_emb(batch["map_id"].long())
              + self.type_emb.weight[T_MAP]).unsqueeze(1)             # (B,1,d)
        gl = (self.global_proj(self.fourier_global(batch["global_feat"]))
              + self.type_emb.weight[T_GLOBAL]).unsqueeze(1)          # (B,1,d)

        pj, pj_ok = self._projectile_tokens(batch)                    # (B,P,d), (B,P)

        cls = torch.stack(
            [self.cls_win, self.cls_kill], dim=0
        ).to(pv.dtype).unsqueeze(0).expand(b, -1, -1)                  # (B,2,d)
        tokens = torch.cat([cls, mp, gl, pv, pj], dim=1)               # (B,S,d)

        valid = torch.cat(
            [
                torch.ones(b, N_PREFIX, dtype=torch.bool, device=alive.device),
                alive,
                pj_ok,
            ],
            dim=1,
        )                                                             # (B,S)
        assert tokens.shape[1] == N_PREFIX + N_PLAYERS + pj.shape[1]
        return tokens, valid

    def _edge_class(self, obstruction: torch.Tensor) -> torch.Tensor:
        """`(B,10,10)` 的遮挡 → `(B,10,10)` 的类别（对角 → `EDGE_SELF`）。

        `.long()` 已经拷了一份，所以后面的 `masked_fill` 不会改到 batch 里的原张量。
        死人的行列不单独给类：它们整个 token 已经被 mask 掉，bias 加在被 mask 的位置上没有
        意义（随后的 `masked_fill` 会把它盖成负无穷）。
        """
        cls = obstruction.long().clamp(0, N_EDGE_CLASS - 2)
        eye = torch.eye(N_PLAYERS, dtype=torch.bool, device=cls.device)
        return cls.masked_fill(eye, EDGE_SELF)

    def _mask_bias(self, valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """`(B,H,S,S)` 的加性掩码（**8 层共用同一份**）：被 mask 的 key 那一列 `MASK_NEG`，其余 0。

        为什么用 `MASK_NEG` 而不是 `finfo(dtype).min`：掩码之后还要**逐层**把遮挡 bias 加进
        player 那一块，`-3.39e38 + (-10)` 在 bf16/fp32 里都会溢出成 `-inf`；而 `-1e4` 加上任
        何 `O(10)` 的 bias 仍然是有限值，`exp(-1e4 - max)` 又精确为 0。这里保证了每一行至少
        有 4 个语境 token 是开的，所以本来也不会出现"整行 -inf → softmax nan"。
        """
        b, s = valid.shape
        bias = torch.zeros(b, self.cfg.n_heads, s, s, dtype=dtype, device=valid.device)
        return bias.masked_fill(~valid[:, None, None, :], MASK_NEG)

    def _edge_bias(
        self, oh: torch.Tensor, layer: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """第 `layer` 层的 `(B,H,10,10)` 遮挡 bias。`oh` 是 `(B,10,10,4)` 的 one-hot。"""
        return torch.einsum("bijc,hc->bhij", oh, self.ob_bias[layer].to(oh.dtype)).to(dtype)

    # ────────────────────────────────────────────────── 前向

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """返回 5 个 logit 张量：

            logit_win         (B,)      二分类（sigmoid）
            logit_alive       (B,10)    逐人二分类，死人的那一列无意义（loss 里 mask 掉）
            logit_can_kill     (B,10)    同上
            logit_next_death  (B,11)    0..9 = 玩家，10 = NO_EVENT（CE，死人的列是 -inf）
            logit_next_kill   (B,11)    同上
        """
        alive = batch["ctx_player_mask"].bool()
        tokens, valid = self._tokens(batch)                           # (B,S,d), (B,S)
        dtype = tokens.dtype
        base = self._mask_bias(valid, dtype)                          # (B,H,S,S)，8 层共用
        oh = F.one_hot(
            self._edge_class(batch["obstruction"]), N_EDGE_CLASS
        ).to(torch.float32)                                           # (B,10,10,4)
        players = slice(N_PREFIX, N_PREFIX + N_PLAYERS)

        h = tokens
        for i, blk in enumerate(self.blocks):
            # 每层只在 player↔player 那一块上加自己的 bias；被 mask 的格子还是 MASK_NEG。
            bias = base.clone()
            bias[:, :, players, players] += self._edge_bias(oh, i, dtype)
            h = blk(h, bias)
        h = self.norm_f(h)

        win = h[:, IDX_CLS_WIN]                                       # (B,d)
        kill = h[:, IDX_CLS_KILL]
        players = h[:, N_PREFIX:N_PREFIX + N_PLAYERS]                  # (B,10,d)

        # CE 之前把死人的列压到 -inf（float32 里做，别让 bf16 的 finfo.min 参与 log_softmax）。
        neg = torch.finfo(torch.float32).min
        nd = self.head_nd_player(players).float().masked_fill(~alive, neg)
        nk = self.head_nk_player(players).float().masked_fill(~alive, neg)
        return {
            "logit_win": self.head_win(win),
            "logit_alive": self.head_alive(players),
            "logit_can_kill": self.head_cankill(players),
            "logit_next_death": torch.cat(
                [nd, self.head_nd_event(kill).float()[:, None]], dim=1
            ),
            "logit_next_kill": torch.cat(
                [nk, self.head_nk_event(kill).float()[:, None]], dim=1
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# loss / 指标
# ═══════════════════════════════════════════════════════════════════════════

#: 5 个 head 的默认权重。`config.yaml` 的 `train.loss_weights` 覆盖它。
DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "win": 1.0,
    "alive": 1.0,
    "can_kill": 1.0,
    "next_death": 1.0,
    "next_kill": 1.0,
}


def spatial_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """5 个 head 的 loss 加权和 → `(total, {name: 标量 loss})`（返回的都是 tensor）。

    * `win`：整 batch 一个 BCE。`ct_win` 是整回合常量，一个 batch 里正负混着才是常态
      （13,451 个 demo 洗牌之后），偶尔整批同号也不会炸。
    * `alive` / `can_kill`：逐人 BCE，**除以 `ctx_player_mask` 的和**而不是画面数 ——
      死人那 8 个位置（平均 2.4 人）不进分母，否则 loss 会被"本来就能推出来的 0"稀释。
      代价是每行的权重等于"这行有几个活人"：人少的残局贡献小。这是有意的，因为
      `next_*` 的 CE 也是每行一个预测，两个口径在"每人次"这个尺度上对齐。
    * `next_*`：11 类 CE，不掩 `NO_EVENT`（它是有信息量的类，实测占 9%），死人的列已在
      `forward` 里压成 `-inf`。

    BCE 的 logits 都先 `.float()`：autocast 下它们是 bf16，`binary_cross_entropy_with_logits`
    在 bf16 上做 log/exp 会掉精度（尤其是 `alive_end` 这种偏负的标签，正样本只占 32%）。
    """
    w = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})
    alive = batch["ctx_player_mask"].bool()
    m = alive.to(out["logit_alive"].dtype)

    def masked_bce(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logit.float(), label.to(torch.float32), reduction="none"
        )
        return (bce * m).sum() / m.sum().clamp_min(1.0)

    parts = {
        "win": F.binary_cross_entropy_with_logits(
            out["logit_win"].float(), batch["ct_win"].to(torch.float32)
        ),
        "alive": masked_bce(out["logit_alive"], batch["alive_end"]),
        "can_kill": masked_bce(out["logit_can_kill"], batch["can_kill"]),
        "next_death": F.cross_entropy(
            out["logit_next_death"].float(), batch["next_death"]
        ),
        "next_kill": F.cross_entropy(
            out["logit_next_kill"].float(), batch["next_kill"]
        ),
    }
    total = sum(w[k] * v for k, v in parts.items())
    return total, parts


def loss_weights_of(cfg_weights: dict | None) -> dict[str, float]:
    """配置里的 `loss_weights` → 完整字典（缺的项用默认 1.0，未知的键直接报错）。"""
    w = dict(DEFAULT_LOSS_WEIGHTS)
    for k, v in (cfg_weights or {}).items():
        if k not in w:
            raise SystemExit(
                f"train.loss_weights 里有未知的 head {k!r}，可选：{sorted(w)}"
            )
        w[k] = float(v)
    return w


def spatial_correct(
    out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """每个 head 的**逐行**对错，形状和标签一致：`(B,)` / `(B,10)`。

    `train.py` 用它做两件事：训练日志里按行平均，valid 里按回合计权（`ds.sample_weights()`）
    —— 两个口径共用同一份"谁对了"，不会因为两处各写一遍而错开。

    逐人两项已经把死人那几列 AND 掉了（`alive_end` / `can_kill` 在死人身上恒 False，
    不掩掉的话分母里一半是"免费的 0"，acc 会虚高到 0.8 以上）。
    """
    alive = batch["ctx_player_mask"].bool()
    return {
        "win": (out["logit_win"] > 0).squeeze(-1) == batch["ct_win"],
        "alive": ((out["logit_alive"] > 0) == batch["alive_end"]) & alive,
        "can_kill": ((out["logit_can_kill"] > 0) == batch["can_kill"]) & alive,
        "next_death": out["logit_next_death"].argmax(-1) == batch["next_death"],
        "next_kill": out["logit_next_kill"].argmax(-1) == batch["next_kill"],
    }
