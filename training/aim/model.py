"""aim 模型：**帧内关系**（MLP 相加）+ **跨帧因果**（时间 transformer），每帧输出 `P(A 击杀 B)`。

    ┌──────── 每帧独立（帧内）─────────────┐      ┌──── 跨帧因果（时间）────────┐
    x_p = norm( proj(p) + Σ MLP(关系项) )         x_A, x_B ──▶ f ──▶ 帧向量
         p = A、B 各算一遍                         └─── T 个 ──▶ logit_t ────┘

一条 `collate_aim` 的 batch 先按**帧**摊平，帧内那一头一次算完所有真实帧、每帧给出 A、B
两个向量，拼成这一帧的向量，再折回 `(B,T,d)` 交给因果的时间 transformer。

为什么这么分：一帧里的关系（10 个人谁看得见谁、谁拿什么枪、炸弹在哪）是**空间**问题；
帧与帧之间"谁在收枪、谁在绕后"才是**时间**问题。旧版把两者混在一条因果序列里（每帧
3 个 token + 两张视角图），既贵又让"同帧关系"和"跨帧历史"抢同一份注意力预算。
上一版给帧内配了一个 2 层空间 transformer，这一版连它也没有了 —— 理由见下面
"为什么不要帧内注意力"。

## 帧内那一头：`proj + Σ MLP`，没有注意力

    x_A = norm_f( proj(player_A)                                        自己的轨迹/朝向/血量/武器
                + MLP_1(player_A, map)                                  地图 id 的 embedding
                + MLP_3(player_A, global)                               炸弹位置/安放时长/回合时间
                + Σ_{i≠A, i 活} MLP_2(player_A, player_i, ob[A][i], ob[i][A])   每帧最多 9 个活人
                + Σ_{prop 非空}  MLP_4(player_A, prop)                   每帧 0~32 个道具 )

最后两个 `Σ` 是**聚合**：`pool: sum` 就是上面写的求和，`pool: attn` 换成带掩码的注意力
池化（见下面"聚合"一段）。`map` / `global` 每帧恒一个，没有聚合可言。

每一项的输入都是"查询玩家自己的向量 + 一个关系对象的向量"**拼接**，过一个 `rel_layers`
层的 MLP：第 1 层把 `2d (+8)` 压到 `d`，后面每层宽度不变、各带一条残差
（`h = h + Dropout(GELU(Linear(h)))`，见 `RelMLP`），输出和 `proj(player_A)` 同维相加；
`norm_f` 是最后那一层 LayerNorm。`x_B` 用**同一套**参数、同样跑一遍，只是查询换成 B。

* 只有两类项需要**聚合**（其他活人 / 道具）：它们的数量逐帧变（活人 0~9、道具 0~32），
  求和天然置换不变 —— 换几个、按什么顺序进来，加出来的向量都一样（`attn` 也一样：
  权重跟着项一起换，输出不变）。map / global 每帧恒一个。
  代价是**和的量级跟着数量走**（只剩 2 个活人时就只有 2 项，满编 9 项），而 `norm_f` 是
  唯一负责把尺度拉回来的东西。好在"尺度"和"分配"是可以分开动的两个旋钮：`pool: attn`
  把分配交给 softmax、同时乘回项数保住计数（见"聚合"一段）；只想动尺度就退一步换均值
  （除以有效项数）。
* 凡是写 `player_A` 的地方都指 `proj(player_A)`，也就是
  `player_proj(F(ctx_player)) + weapon_emb`；所以主项是"这个人自己长什么样"，
  其余四项是把他放进这一帧的语境里。
* 死人不进聚合（`alive` 掩掉），死人那一行整行换成可学习的 `dead` 向量 —— 磁盘上死人
  那行**不保证是 0**，不换掉就是往模型里喂残留位置值。A、B 在窗口内恒活（`aim.py` 造数据
  时 `rr.alive[a, win].all()` 卡过），所以查询方不会走到 `dead`；那一行只是兜底。
* 每次前向只算 **A、B 两个查询**，不逐个算 10 个人：帧向量只要这两个，其余 8 个人的向量
  没有去处，算了也是白算。这一点直接决定了成本（见下）。

## 为什么不要帧内注意力

1. **帧内没有序列**。token 的顺序只是槽位编号，注意力在这一层能学到的东西里，一大半是
   "这个 token 是谁、对面是谁" —— 而 `pair` 的下标和 `cat(x_A, x_B)` 的拼接顺序已经把
   这件事免费说清楚（见下面"帧向量"），不需要花一层注意力去学。
2. **同样的关系，算力差三倍**。旧版每帧 `1 + 1 + 10 + P` 个 token（`P` 取 batch 内最大
   占用，`S ≈ 15~22`），一层 ≈ `S × 12 d²`；现在每帧 2 个查询、每个查询
   `9 个活人 + P 个道具 + 3 个小项` 次 MLP，每次 `(1 + rel_layers) d²`（`rel_layers=2`
   即 `3 d²`，`config.yaml` 用的 3 即 `4 d²`）。`d = 640` 时：
   2 层空间 ≈ **154 M MAC/帧**（`2 × 15.7 × 12d²`），现在 ≈ **39~54 M**（`P` 从 3.7 到 10）。
   参数量 帧内 ≈ **9.8 M → 4.9 M**。后两个数是 `rel_layers=2` 的口径，`L=3` 就是
   `×4/3`：≈ **52~72 M**、参数 6.6 M。⚠ 都是按 `d²` 手算的估算，实测以 `train.py`
   每个 epoch 打的那行 `ms/微步 / 样本每秒` 为准。
3. **没有 mask 陷阱**。旧版要为"空道具槽"建注意力 key mask、`P` 取 batch 内最大值，同一帧
   的表示会跟着 batch 里别的样本带多少道具变；现在空槽根本不进聚合，一帧的表示只由这一帧
   自己决定。同理，没有 padding token 抢权重的问题。
4. 代价是**关系之间不再互相影响**：注意力里 10 条关系在同一轮 softmax 里竞争、能组合出
   "A 看得见 B 但 B 看不见 A，而且 B 背对" 这种二阶结构；现在是每条关系独立过一个 MLP
   再相加，帧内的交互被压平成加法。这类组合只能留给后面每帧一个向量的时间 transformer。

这就是这次改动的赌注：任务里真正值钱的是**少数几条显式关系**（谁看得见谁、拿什么枪），
不是 token 之间自由的二阶交互。如果拆完发现 acc 掉得多，缺的多半就是第 4 条。

## 聚合：`pool: sum | attn`（求和 vs 注意力池化）

上面第 4 条说的"关系之间不再互相影响"，不必用帧内注意力去补 —— 那条贵在**给 10 个玩家
token 都算一遍表示再扔掉 8 个**，而真正缺的只是"候选之间没有竞争"。所以这里只改**最后那
一步聚合**，`RelMLP` 一行不动：

    sum  ：x += Σ_i u_i                      u_i = MLP(关系 i)
    attn ：x += Σ_i (n · a_i) u_i            a = softmax_i( w·LN(u_i) + b )，n = Σ ok

实现见 `RelPool`。要点：

* **乘回 `n`**：输入里没有任何人数 / 道具计数特征，`Σ ok` 是这条轴唯一携带"有几个"的
  地方。纯 softmax 权重和恒为 1，等于把这条通道删掉；乘回 `n` 后总权重恒等于项数，
  计数不丢、只是分配变成可学的。
* **分数过 `LN`**：`LN(u)` 的模长被钉在 `√d` 附近，于是 softmax 的温度**只由 `‖w‖`
  决定**（一个显式的、受损失优化的量），而不是"主干输出模长"这种被 `norm_f` 免费吸收、
  没有压力不长大、最后把 `a` 推到 one-hot 的隐变量。顺带 `w` 变成可读的"相关度方向"。
* **初始化即等价**：`w = b = 0` → `a_i = 1/n` → `n·a_i = 1`，所以 `attn` 档在 step 0 与
  `sum` 档**逐位相同**（fp32 下 `(1/n)·n` 恰好回 1，`n ≤ 12` 实测全中），`sum` 的
  checkpoint 可以直接续训。
* **成本**：打分读的是主干已经算出来的 `u`，每个关系 `O(d)` 对值那一路的 `O(d²)`，`d =
  1024` 时整帧约 **+0.07%**、参数 +6d+2。要 `H` 头就把 `Linear(d, 1)` 的 `1` 换成 `H`。
* **买到了什么**：候选之间的**竞争**（"谁相关"从每个关系各自决定变成集合级决定），
  以及"最近 / 最威胁的那个"这类近似 argmax 的语义。**没买到**：`u_i` 之间仍然不交互
  （value 只看自己那一对），所以"二阶组合"还是缺的，这里补的是"二阶权重"。
* **还缺上下文**：打分只看 `cat(q, pv_j, 两向遮挡)`，看不到 `global`（回合时间 / 炸弹），
  而池化发生在时间 transformer 之前 —— 所以"谁相关"在整个回合里是同一个方向。真要补，
  把 `gl` 加进打分输入（+`d` 参数），别靠"加深"补。

⚠ **这一档还没有实测收益**（`sum` 是上一版 0.811 的口径）。看 `valid/acc_last` 会被"恰好
一方看得见"那 79.5% 的一阶子集稀释，要一起看：`a` 的熵（贴着 `log n` = 打分头没学到）、
`argmax_j a` 是否随帧变化（集中在固定槽位 = 塌缩）、以及按子集拆（`恰好一方看得见` /
`两方都看不见` / `≥3 个活人参与`）—— 预期收益在第三档。

## `rel_layers`：帧内 MLP 要多深

`rel_layers` 数的是**线性层**个数（GELU 也是每层一个：第 1 层一个，之后每个残差块一个）。
往里加层买到的是"合取"—— "i 很近 **并且** 拿步枪 **并且** 我侧身"：

* `L=1`：只有第 1 层那个 GELU，非线性发生在**输入坐标**上。合取也能逼近，但只能靠 GELU
  在 0 附近的二次项硬凑（`GELU'' ≠ 0`，`(x+y)² − (x−y)²` 那套）；而曲率只在 `|x| ≲ 2`
  那一段明显，得靠学出来的 bias / 缩放恰好落进去 —— 数值上脆、参数上贵。
* `L=2`（默认）：第二个 GELU 作用在**学到的特征**上（`GELU(W₂·h₁)`），它的一个单元就是
  "`α h_i + β h_j − θ` 过阈值"，即特征与特征之间的与门。合取从"凑出来"变成"直接建"，
  这就是默认取 2 的理由。
* `L≥3`：只在需要"合取的合取"（分层组合）时才有边际收益，而每加一层都乘在**每次求值**上：
  每次 `+d²`。四项一起加是帧内 **+33%**（总 +16~17%，随 `P` 变）；只加 `mlp_pair` 是
  帧内 +14~19%（总 +6~10%，参数 +0.4 M）。本仓库的 `config.yaml` 现在走的就是 `L=3`
  （四项一起），`AimConfig` 的默认值仍是 2 —— 想回退改 dataclass 那一行，别只改 YAML。

残差是给"深"兜底的：没有它，每加一层都要重学一遍"输出 ≈ 输入"，深了反而更难训；有了它，
浅层学到的东西能原封不动穿到出口，`rel_layers` 才真是"加容量"的旋钮（代价是最后那一层
带着一条恒等通路，出口不再是一个干净的线性投影 —— 这里无所谓，反正后面还有 `norm_f`）。

四项共用一个值。想只加深 `mlp_pair`（唯一真吃二阶交互的一项）更划算：把那一行的
`RelMLP(..., n_rel, ...)` 换成 `RelMLP(..., 3, ...)` 就行，别四项一起加。

## 遮挡：进 `MLP_2` 的输入，两个方向都给

`obstruction[t][i][j]` 是**有向**的（i 的眼睛看 j 的身体，两人眼睛位置不同），实测 7.91%
的 A↔B 帧两个方向不一致。所以 `MLP_2` 同时吃 `ob[A][i]` 和 `ob[i][A]`，各自 one-hot 成
4 维一起拼进输入（合计 `2d + 8` 维）。类别：

    0 通   1 烟   2 墙   3 无效（i == j，或 i / j 已死）

实测（449 条 / 115,393 帧，活人之间的有向对，`i != j`）：

    通 10.64%   烟 2.10%   墙 87.26%        每帧"通或烟"的对数：均值 7.1 / p95 16
    A 看见 B：全帧平均 18.83%（首帧 3.6% → 末帧 86.0%）  其中 6.24% 是隔着烟看见的

后两个数就是时间 transformer 存在的理由：**"终于看见"这件事是突变，不是渐变**；
而烟占了"A 非墙"的 1/4，不是边角情况。旧 449 条上，只有一方看得见时看见的是杀手的占
**79.5%** —— 这是任务里信号最强的一条，所以帧内不做注意力可以，`ob` 不能跟着一起扔。

对角 `[i][i]` 恒为 0（实测 893,102 个活人·帧里 0 烟 0 墙），死人的行列也恒为 0 而 0 读作
"通" —— 这两件事都必须并进类 3，否则模型会读到"我完全看得见一个已经死掉的人"。
聚合那一步（`sum` 的求和、`attn` 的掩码）已经把 `i == j` 和死人排除了，所以类 3 在这里
主要是兜底（数据漏标记时才会出现）。

`ob == 0` **不等于"看得见"**：它只表示"眼睛到身体 8 个角里至少露出一个"，且**没算 FOV**。
所以这块特征是"物理上没挡"，不是"正在看着" —— 对任务是好事（不泄漏朝向），但别当
visibility 真值用。

## 帧向量：A、B 拼起来，不池化

    x_A, x_B = 上面两个查询各出一个      # (n, d)
    u = pair_proj(cat(x_A, x_B))         # (n, d) ← 时间 transformer 的输入

为什么不做池化（旧版的 CLS，或者"10 个状态取平均"）：这一帧要答的是"**这两个人里谁死**"，
一个二选一；池化会把 10 个玩家加 P 个道具压成一个向量，答案得先在里面"存下谁是 A、
谁是 B"再解出来（"a 看得见 b"这条也只占 1/12 的份量）。换成拼接，pair 的身份由拼接顺序
直接表达，不占维度、也不用学。

拼接顺序表示的就是"这次问的是哪个方向"，**它和标签是同一件事**：`cat(x_A,x_B)` 配
`label=True`（A 是击杀者），`cat(x_B,x_A)` 配 `label=False`。`dataset.swap_ab` 就是在这
两个方向之间摇；`pair (B,2)` 由 `collate_aim` 从 `ctx_player_mask` 里取出来。

## `antisym`：两个方向的 logit 相减（可选，默认关）

    开：z_t = w·h_t(A,B) − w·h_t(B,A)     交换 A/B 时 z **精确取反**
    关：z_t = w·h_t(A,B)                  对称性靠 data.swap_ab 摇出来

必须放在时间那一头**之后**，两个理由：

* 两人距离 / 双向遮挡 / 各自武器这些是**对称**证据，在帧向量上先做差会把它们减掉一半
  —— 而它们是判"谁死"的主要依据；
* 时间 transformer 里的 LayerNorm + GELU + bias 都不是奇函数，反对称穿不过去。

实现上把两个方向拼成 `(2B,T,d)` 一次前向、末尾再减，没有额外内核开销。代价是时间那一头
跑两遍 ≈ **+40~50%** 算力（时间侧现在占了全模型大约一半 —— 帧内改成 MLP 之后它不再便宜到
可以忽略）。打开之后 `data.swap_ab` 变成纯
冗余（`BCE(z,1)` 与 `BCE(−z,0)` 的梯度完全相同），应当关掉 —— **步数不用改**，省掉的是
重复的那一遍，不是算力。

## 时间那一头

`(B,T,d)` 上跑因果自注意力，每个 tick 直接过 `Linear(d,1)`，`logits` 就是 `(B,T)`。
位置只有 `time_emb[round(seconds × 64)]`（一张 `time_bins × d` 的表）加在帧向量上。
`time_bins = 257` 要装下 `k = 0..256`（写 256 会在最长样本上越界）。

**时间那一头没有 padding mask**：补齐帧补在末尾，因果注意力让真帧永远看不到它们。
而且帧内那一头**只在真帧上算**（先 `frame_mask` 选出来再散回去）—— 训练开了随机截断
之后实时帧数平均只有 52%，这一步省掉一半帧内开销。

帧内那一头不需要任何 mask：空道具槽被 `Σ` 跳过，死人被 `alive` 跳过，用的都是乘法/加法
掩码，没有"某一行全被屏蔽"的 nan 风险（时间那一头的补零行有同样的问题，但因果遮罩保证
它不参与）。

`pred_mask` 就是 `frame_mask`：每帧一个预测点，没有额外的"看完整个窗口"那一列。

## 和上一版（空间 transformer）不兼容

`spatial.*` / `edge_proj` / `edge_bias` / `slot_emb` 这些 key 全没了，`state_dict` 对不上，
旧 checkpoint 只能重训（`train.py --resume` 会直接报错）。`model.spatial_layers` 这个超参
也一起删了：帧内**没有注意力层数**可调，深度是 `rel_layers`、宽度是 `d_model`。

`pool: attn` 额外多出 `pool_pair.*` / `pool_prop.*`（两条轴各一个 `LayerNorm` + 一个
`Linear`）；`pool: sum` 一个都不多，所以上一版的 checkpoint 在 `sum` 档下仍然逐 key 相同。
`sum → attn` 可以续训（分数头零初始化，起点等价），`attn → sum` 不行（多了 key）。

## 为什么不再有视角图

旧版每帧（实际是首帧）渲染两张 `(192,256,5)` 的视角图，448 个 token，占条件 token 的
一半。实测它几乎不值钱（449 条 / 898 行末帧 acc）：把视角换成**别的样本**的视角
0.778 → 0.773（-0.006），只在"B 在 A 45° 视锥内"的子集里掉 0.017；换成全空图掉 0.024，
那是分布外惩罚而不是丢信息。原因不在分辨率：图取的是窗口**第一帧**，而击杀在末尾，
真正值钱的"谁挡着谁"现在是逐帧的 `obstruction`，直接给。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

#: 和 `scripts/training_data/tasks/aim.py` 写出的样本对齐。
#: ⚠ `ctx_player_mask` 磁盘上是 0 死 / 1 A(击杀者) / 2 B(受害者) / 3 其他活人，但
#: `dataset.collate_aim` 把角色搬进了 `pair (B,2)`，模型这边只看到 0 死 / 1 活。
N_PLAYERS = 10              # 玩家槽位数（死人也占位）
N_PLAYER_FEAT = 20          # `ctx_player` 的列数（AIM_FEATURES）
N_PROJ = 32                 # `ctx_projectile` 的槽位数
N_PROJ_FEAT = 4             # x y z t
N_GLOBAL = 5                # `global_feat` 的列数
N_WEAPONS = 43              # `constants.WEAPON_NAMES` 的长度；-1 = 无武器
N_MAPS = 8                  # `constants.MAP_TO_IDX` 的大小
N_PROJ_TYPE = 7             # `ctx_projectile_id` 的取值 0..6（-1 = 空槽）
N_EDGE_CLASS = 4            # 通 / 烟 / 墙 / 无效（自己或已死）


@dataclass
class AimConfig:
    """模型超参。`config.yaml` 的 `model:` 段直接 unpack 成它。"""

    d_model: int = 256
    n_heads: int = 8
    #: 帧内每个关系项 MLP 的层数：第 1 层 `in_dim → d_model`，其后每层 `d_model → d_model`
    #: 且带一条残差（见 `RelMLP`）。`1` = 单隐层；`2` 和上一版同宽同深；3 以上才便宜地
    #: 表达"特征合取"（见模块 docstring 的 `rel_layers` 一段）。每加一层 ≈ 帧内 +`d²`/求值。
    rel_layers: int = 2
    #: 两条可变长关系轴（其他活人 / 道具）的**聚合**方式：
    #: `"sum"` = 直接求和（旧行为，`state_dict` 与加这个开关之前逐 key 相同）；
    #: `"attn"` = 带掩码的注意力池化（见 `RelPool` 和模块 docstring 的"聚合"一段）。
    #: 换档会改 `state_dict`：`attn` 多出 `pool_pair.*` / `pool_prop.*`。从 `sum` 的
    #: checkpoint 换到 `attn` 可以续训（分数头零初始化 → 起点与 `sum` 逐位相同）。
    pool: str = "sum"
    #: 时间 transformer 的层数。序列长度是 `T`，和帧内项数无关；帧内那一头的深度是
    #: `rel_layers`，两者互不影响（一个管"跨帧看多深"，一个管"帧内的关系算多细"）。
    temporal_layers: int = 6
    #: 两个方向各跑一遍时间 transformer、末尾相减（见模块 docstring 的 `antisym` 一段）。
    #: 打开后 `data.swap_ab` 应关掉（步数不用改）；代价 +40~50% 算力（时间侧跑两遍，
    #: 而时间侧现在占了全模型大约一半）。
    antisym: bool = False
    #: FFN 隐层 = `d_model * ffn_mult`。
    ffn_mult: float = 4.0
    dropout: float = 0.1
    #: `[0,1]` 小特征的 Fourier 展开：每通道频率个数。
    #: 输出维数 = 原维数 × `(1 + 2 × n_freq)`，开头的 `1` 是原始值（见 `FourierFeatures`）。
    player_freq: int = 8
    proj_freq: int = 8
    global_freq: int = 8
    fourier_f_min: float = 1.0
    #: 多小的差异还能被线性分辨：`f=128` 时 0.004 的差异就有 ~90° 相位差
    #: （位置列 ≈ 10 HU、hp ≈ 0.4%、角度 ≈ 1.4°）。
    fourier_f_max: float = 128.0
    #: 数据的时间分辨率（帧/秒），`seconds` 永远是它的整数倍。
    tick_hz: float = 64.0
    #: 时间位置表的大小。窗口 4 秒 → `k = seconds × 64 ∈ 0..256` 共 257 个。
    time_bins: int = 257

    def __post_init__(self) -> None:
        # `range(0)` 也建得出来：不拦的话 `0` / 负数会被当成"只有第 1 层"静默跑起来，
        # 消融时就分不清是"层数没生效"还是"这一项没用"。
        if self.rel_layers < 1:
            raise ValueError(
                f"rel_layers 必须 >= 1（1 = 单隐层），收到 {self.rel_layers}"
            )
        # 同 `rel_layers`：拼错的字符串会被 `RelPool` 当成"不是 attn"静默建成 sum，
        # 消融时就分不清是"池化没生效"还是"这一项没用"。
        if self.pool not in ("sum", "attn"):
            raise ValueError(f'pool 只能是 "sum" / "attn"，收到 {self.pool!r}')


class FourierFeatures(nn.Module):
    """`x ∈ [0,1]^C` → **原始值 + 每通道多频率 `sin/cos`**，输出 `C * (1 + 2 * n_freq)` 维。

    这批小特征全在 `[0,1]`，直接进一个宽 `Linear` 的话模型很难从单个标量里分辨
    "1° 和 3° 的差别"和"0.9 和 0.91 的差别"—— 两个尺度被压成一个数。每个通道先摊成
    `n_freq` 个频率的 `sin/cos`（频率在 `[f_min, f_max]` 上几何等距，`f` = 整个 `[0,1]`
    区间振荡的周期数），不同分辨率各自占一个可学习方向。

    输出排布：**原始值整段在最前**，之后每个通道占一个 `[sin f1..fk, cos f1..fk]` 块：

        `[x_1 … x_C | ch1: sin, cos | ch2: sin, cos | … ]`
          └─ `C` 维 ─┘└──────── `2 * C * n_freq` 维 ────────┘

    ## 为什么还要把原始值拼回去

    `f_min = 1` 时最低频那两个通道（`sin 2πx` / `cos 2πx`）合起来已经是 `x` 的双射，
    所以原始值在**信息上完全冗余**。拼它是为了给下游那个 `Linear` 一条**直通路径**：
    没有它，模型想要一个 `x` 的线性项，得先用两层权重把 `atan2(sin, cos)` 反解出来；
    有了它，`x` 的那一列权重就是纯粹的斜率。代价是每个通道 +1 维输入（整模型
    `+29 d` 参数，可忽略），换掉的是"学习恒等映射"这件本来没必要学的活。

    所以：原始值和展开**不是二选一**，前者是后者的低频极限，后者负责高频分辨率。

    宽度直接用 `self.out_dim`（别在外面手算，`C + 2 * C * n_freq`）。
    """

    def __init__(
        self, n_ch: int, n_freq: int, f_min: float = 1.0, f_max: float = 128.0
    ) -> None:
        super().__init__()
        if n_freq <= 0:
            raise ValueError(f"n_freq 必须是正整数，收到 {n_freq}")
        freqs = torch.logspace(
            math.log10(f_min), math.log10(f_max), n_freq, dtype=torch.float32
        )
        #: `(n_freq,)`，x（`[..., C]`）右乘成 `[..., C, n_freq]`；所有通道共用一组频率。
        self.register_buffer("freqs", freqs * 2 * math.pi)
        self.n_ch, self.n_freq = n_ch, n_freq
        self.out_dim = n_ch * (1 + 2 * n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(-1) * self.freqs                    # (..., C, n_freq)
        # `cat` 不做 dtype 提升（混着 bf16 / fp32 会直接报错），所以对齐到 `z` 的 dtype；
        # fp32 输入时 `x.to` 是 no-op。现在调用方传的都是 fp32（autocast 只在 matmul 处转）。
        return torch.cat(
            [x.to(z.dtype), torch.cat([z.sin(), z.cos()], dim=-1).flatten(-2)],
            dim=-1,
        )


class SelfAttention(nn.Module):
    """因果多头自注意力（时间那一头唯一的使用者）。"""

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 不能被 n_heads={n_heads} 整除")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = (
            self.qkv(x)
            .view(b, t, 3, self.n_heads, self.d_head)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        # 因果遮罩交给内核（`is_causal=True`）：不建 `(T,T)` 的 mask，也不要求 T 是 2 的幂。
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(y.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    """pre-LN 残差块：`x + attn(ln(x))`，再 `x + mlp(ln(x))`。"""

    def __init__(self, cfg: AimConfig) -> None:
        super().__init__()
        hidden = int(cfg.d_model * cfg.ffn_mult)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RelMLP(nn.Module):
    """帧内的一项关系：`cat(查询玩家, 关系对象) → d`，`rel_layers` 层、宽度不变的层带残差。

    第 1 层把 `in_dim` 压到 `d_model` —— 帧内四项都要在这个宽度上和 `proj(player)` 相加，
    所以维度只能在这里变，它也就没有残差可加：

        h = Dropout(GELU(Linear(in_dim, d)(x)))        第 1 层
        h = h + Dropout(GELU(Linear(d, d)(h)))         第 2..rel_layers 层，各一条残差

    所以 `rel_layers=1` 是"单隐层 MLP"（合取只能靠 GELU 的二次项硬凑），`2` 和上一版同宽
    同深（尾部多一个 GELU 和一条残差），而第二个 GELU 已经作用在学到的特征上 —— 合取从
    这里就变便宜了。3 以上是"合取的合取"，边际收益递减（见模块 docstring 的 `rel_layers`
    一段）。

    不做特殊初始化（不置零也不缩小）：残差链的起点就是第 1 层的输出，而把它压到 0 会让
    后面每层的隐层第一步都没有梯度（`dL/dW_hidden ∝ W_out`），要等输出层自己长出来才学得动。
    量级由最后的 `norm_f` 兜底。
    """

    def __init__(
        self, in_dim: int, d_model: int, n_layers: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"rel_layers 必须 >= 1，收到 {n_layers}")
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
            for _ in range(n_layers - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return h


class RelPool(nn.Module):
    """帧内两条可变长关系轴的聚合：`sum`（求和）或 `attn`（带掩码的注意力池化）。

        x = Σ_i u_i                              sum
        x = Σ_i (n · a_i) u_i                    attn，a = softmax_i( w·LN(u_i) + b )

    `u` 是 `RelMLP` 的输出 `(n, K, d)`，`ok` 是这一批里真有东西的槽位 `(n, K)`。`K` 取
    batch 内的最大占用，**真实项数 `n = Σ ok` 逐帧变**（活人 1~9、道具 0~32），而它是这条轴
    唯一携带"有几个"的地方 —— 输入里没有任何计数特征（`AIM_FEATURES` / `global_feat`
    都没有人数）。

    ## 为什么乘回 `n`

    纯 softmax 的权重和恒为 1，等于把"这一帧有 3 个人还是 9 个人"这条信息删掉 —— 而这条
    信息只从这里过。乘回 `n = Σ ok` 之后**总权重恒等于项数**，计数一点没丢，变的只是项与
    项之间的**分配**。`w = b = 0` 时 `a_i = 1/n`，乘回 `n` 得 1，于是 `attn` 与 `sum` 在
    初始化时**逐位相同**（`(1/n)·n` 在 fp32 下恰好回 1，`n ≤ 12` 实测全中 —— 前提是
    softmax 在 fp32 里算，见 `forward` 那行注释），所以 `sum` 的 checkpoint 能直接喂给
    `attn`（见 `train.py --resume` 那段）。

    ## 为什么分数要过 `LN`

    `LN(u)` 的每一维都是单位方差、模长恒在 `√d` 附近，于是 `w·LN(u)` 的取值范围（`±‖w‖√d`
    之内）**只由 `‖w‖` 决定** —— softmax 的温度也就只由 `w` 决定，而它直接受损失优化。
    喂裸 `u` 的话温度会变成"主干输出的模长"—— 而主干那一路的尺度被 `norm_f` 免费吸收、
    没有任何压力不长大，长到某处 `a` 就塌成 one-hot，`K-1` 项再也拿不到梯度。`LN` 把这根
    线切断，顺带让分数变成"`u` 与方向 `w` 的余弦相似度"：`w` 是能读出来的。

    ## 掩码：`-1e4` 而不是 `-inf`

    道具轴可以是**全空**（`n = 0`，实测均值 3.7、但一枪没开的帧就是 0），全 `-inf` 的
    softmax 是 NaN；而且 `torch.where` 的反向是一次乘法，`NaN × 0 = NaN` 会漏进梯度。
    用有限大负数时退化的那一行 `a` 是全 `1/K` 均匀、再乘一次 `ok` 就恒为 0，和 `sum` 档
    在同一个输入上的结果一致（都是 0）。

    ## 成本

    打分的输入是主干**已经算出来**的 `u`，所以每个关系只花 `O(d)`（`LN` ≈2d +
    `Linear(d, 1)` d+1），而值那一路是 `O(d²)`（`rel_layers=3` 时 `4 d²`）—— 差 `1/d`。
    `d = 1024` 时整帧约 **+0.07%**（两条轴合计 +6d+2 ≈ 6 K 参数）。要加宽成 `H` 头，
    就是把下面的 `1` 换成 `H`（参数 `H·d`，仍然可忽略）。

    这条通路**还没有实测收益**（见模块 docstring 的"聚合"一段）：要盯的是 `a` 的熵和
    `argmax_j a` 的分布，塌缩时先降 lr / 给 `a` 加 dropout，别急着加深。
    """

    #: 挡空槽用的有限大负数（为什么不是 `-inf` 见类 docstring）。
    _MASK = -1e4

    def __init__(self, d_model: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode == "attn":
            # `sum` 档**一个参数都不建**：这样加这个开关之前的 checkpoint 仍然逐 key 相同。
            self.norm = nn.LayerNorm(d_model)
            self.score = nn.Linear(d_model, 1)
            # 零初始化：`a` 均匀 → `n·a_i = 1` → 起点就是 `sum`。残差/主干那边不需要
            # 特殊初始化：分数只从 `u` 里线性读出，`u` 在 step 0 就已经是有意义的特征。
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)

    def forward(self, u: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        """`u (n,K,d)` + `ok (n,K) bool` → `(n,d)`。"""
        if self.mode == "sum":
            return (u * ok.unsqueeze(-1)).sum(1)
        s = self.score(self.norm(u)).squeeze(-1)                  # (n,K)
        # softmax 落到 fp32 再算：bf16 里 `1/9` 是 0.111328（相对误差 0.2%），9 个乘回后
        # 总权重是 9.018 而不是 9；分数之间只差一点点时量化还会更糟。fp32 下 `(1/n)·n`
        # 在 n <= 12 时恰好回 1（实测），所以"初始化等于 sum"是**逐位**的。
        a = s.masked_fill(~ok, self._MASK).float().softmax(-1).to(u.dtype)
        a = a * ok                                                # 全空行 → 全 0
        return (u * (a * ok.sum(-1, keepdim=True).to(u.dtype)).unsqueeze(-1)).sum(1)


class AimModel(nn.Module):
    """`P(A 击杀 B)`：帧内用 MLP 把关系加进 A、B 各自的向量，时间 transformer 每帧出 logit。"""

    def __init__(self, cfg: AimConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or AimConfig()
        d = cfg.d_model

        # ---- 帧内那一头的输入 ----
        self.fourier_player = FourierFeatures(
            N_PLAYER_FEAT, cfg.player_freq, cfg.fourier_f_min, cfg.fourier_f_max
        )
        self.fourier_proj = FourierFeatures(
            N_PROJ_FEAT, cfg.proj_freq, cfg.fourier_f_min, cfg.fourier_f_max
        )
        self.fourier_global = FourierFeatures(
            N_GLOBAL, cfg.global_freq, cfg.fourier_f_min, cfg.fourier_f_max
        )
        self.map_emb = nn.Embedding(N_MAPS, d)
        self.player_proj = nn.Linear(self.fourier_player.out_dim, d)
        self.global_proj = nn.Linear(self.fourier_global.out_dim, d)
        self.proj_proj = nn.Linear(self.fourier_proj.out_dim, d)
        #: 道具类型：`-1`（空槽）落到槽位 0。
        self.proj_type = nn.Embedding(N_PROJ_TYPE + 1, d)
        #: 武器：`-1`（无武器 / 已死）落到槽位 0。
        self.weapon_emb = nn.Embedding(N_WEAPONS + 1, d, padding_idx=0)
        #: 死人的替身（磁盘上死人那行不保证是 0）。聚合时死人是被掩掉的，这里只是兜底。
        self.dead = nn.Parameter(torch.zeros(d))

        # ---- 帧内的四项关系（见模块 docstring 的公式）----
        #: `map` / `global` / `prop` 的输入都是 `cat(2 个 d)`；`pair` 再多两向遮挡的
        #: one-hot（2 × 4 类）→ `2d + 8`。宽度统一 `d`，深度统一 `rel_layers`。
        n_rel = cfg.rel_layers
        self.mlp_map = RelMLP(2 * d, d, n_rel, cfg.dropout)
        self.mlp_global = RelMLP(2 * d, d, n_rel, cfg.dropout)
        self.mlp_prop = RelMLP(2 * d, d, n_rel, cfg.dropout)
        self.mlp_pair = RelMLP(2 * d + 2 * N_EDGE_CLASS, d, n_rel, cfg.dropout)

        #: 两条**数量逐帧变**的轴怎么合成一个向量：`sum` 时不建参数，`attn` 时各一个
        #: 零初始化的标量打分头（见 `RelPool`）。`map` / `global` 每帧恒一个，没有池化可言。
        self.pool_pair = RelPool(d, cfg.pool)
        self.pool_prop = RelPool(d, cfg.pool)

        #: 聚合（求和 / 注意力池化）之后那一次 LayerNorm（帧内那一头的出口）。
        self.norm_f = nn.LayerNorm(d)
        #: 读出头：`cat(x_A, x_B) → d`。两层而不是一层：帧向量要同时装下"对称证据"和
        #: "这一对是谁、往哪个方向问"，单个线性层只能在 concat 的两半上各做一次加权和。
        self.pair_proj = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # ---- 时间那一头 ----
        self.temporal = nn.ModuleList(
            [Block(cfg) for _ in range(cfg.temporal_layers)]
        )
        self.norm_t = nn.LayerNorm(d)
        #: 时间位置：`Embedding(257, d)`，索引 = 帧号。这是模型唯一的位置编码。
        #: 初始化 0.02（同 GPT-2 的 `wpe`）：一开始接近 0，位置信息随训练长出来。
        self.time_emb = nn.Embedding(cfg.time_bins, d)
        nn.init.normal_(self.time_emb.weight, std=0.02)

        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)

    # --------------------------------------------------------- 帧内：关系项

    def _edge_class(self, ob: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """`(n,10,10)` 的遮挡 → `(n,10,10)` 的类别，第 4 类是"无效"。

        无效 = `j` 是自己（对角恒 0，实测 893,102 个活人·帧里 0 烟 0 墙）或 `j` 已死
        （死人的行列恒 0，而 0 读作"通"）。不并进去，模型就会读到"我完全看得见一个
        已经死掉的人"。聚合本身已经排除了这两类（`sum` 靠求和项、`attn` 靠掩码），所以这里
        是数据漏标记时的兜底。
        """
        eye = torch.eye(N_PLAYERS, dtype=torch.bool, device=ob.device)   # (10,10)
        null = eye | (~alive).unsqueeze(-2)                              # (n,10,10)
        cls = ob.long().clamp_(0, 2)
        return cls.masked_fill_(null, N_EDGE_CLASS - 1)

    def _prop_vectors(
        self, batch: dict[str, torch.Tensor], idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(n, P, d)` 的道具向量 + `(n, P)` 的"这个槽真有东西"标记。

        ⚠ 磁盘上的道具槽**不是前缀排布**（449 条里 410 条中间有洞），所以必须先把
        `id >= 0` 的挤到前面再切，不能直接 `[:, :P]`。`P` 取本 batch 的最大占用数，
        上限是槽位数 `N_PROJ = 32`（实测均值 3.7 / p95 9 / max 12）。

        空槽由调用方用 `ok` 挡在聚合之外 —— 所以**同一帧的表示和 batch 里别的样本带多少
        道具无关**；旧版用注意力时必须靠 key mask 才能做到这件事。
        """
        pid = batch["ctx_projectile_id"][idx]               # (n,N_PROJ)
        feat = batch["ctx_projectile"][idx]                 # (n,N_PROJ,4)
        filled = pid >= 0
        n_prop = max(1, int(filled.sum(-1).max()))          # 本 batch 最多几个
        order = torch.argsort((~filled).to(torch.int8), dim=-1, stable=True)[:, :n_prop]
        pid = pid.gather(1, order)
        feat = feat.gather(1, order.unsqueeze(-1).expand(-1, -1, N_PROJ_FEAT))
        vec = self.proj_proj(self.fourier_proj(feat)) + self.proj_type(pid.long() + 1)
        return vec, pid >= 0

    def _pair_vectors(
        self, batch: dict[str, torch.Tensor], idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """每帧的 A、B 向量 → 两个方向的帧向量 `(n,d)`，`(A,B)` 在前。

        `n = Σ frame_mask`：只有真实帧进来，补齐帧在外层散回去（见 `forward`）。
        `pair` 是 **per-sample** 的：窗口内 A 和 B 一直活着（`aim.py` 造数据时
        `rr.alive[a, win].all()` 卡过这一点），所以不需要逐帧索引，按 `frame_mask`
        展开就行。`x_A` 和 `x_B` 走的是**同一套**参数 —— 谁是 A 只体现在查询哪个下标。

        只算这两个查询：其余 8 个人的向量没有去处，算了白算（这是帧内成本的大头）。
        """
        n = int(idx.sum())
        rows = torch.arange(n, device=idx.device)

        feat = batch["ctx_player"][idx]                     # (n,10,20)
        wpn = batch["ctx_player_weapon"][idx].long()        # (n,10)
        alive = batch["ctx_player_mask"][idx] > 0           # (n,10)
        #: `proj(player)`：自己的轨迹 / 朝向 / 血量（Fourier 展开）+ 武器。
        pv = self.player_proj(self.fourier_player(feat)) + self.weapon_emb(wpn + 1)
        # `dead` 显式转成 pv 的 dtype：`torch.where` 会做类型提升，bf16 下的
        # `cat(bf16, fp32)` 会直接报错。
        pv = torch.where(alive.unsqueeze(-1), pv, self.dead.to(pv.dtype))  # (n,10,d)

        aut = self._edge_class(batch["obstruction"][idx], alive)      # (n,10,10) 类别
        # `map_id` 是 per-sample 的 `(B,)`，`map_emb` 出来是 `(B,d)`；bool 掩码要求
        # "覆盖到的维"完全等形，所以先摊成 `(B,T,d)` 再用 `(B,T)` 的 `idx` 选帧。
        b, t = idx.shape
        mp = self.map_emb(batch["map_id"].long())                    # (B,d)
        mp = mp.unsqueeze(1).expand(b, t, -1)[idx]                   # (n,d)
        gl = self.global_proj(self.fourier_global(batch["global_feat"][idx]))
        pr, pr_ok = self._prop_vectors(batch, idx)                  # (n,P,d), (n,P)
        n_prop = pr.shape[1]
        d = pv.shape[-1]

        def state(p: torch.Tensor) -> torch.Tensor:
            """一个查询玩家的帧内向量（`p` 是每个真实帧的目标槽位，`(n,)`）。"""
            q = pv[rows, p]                                     # (n,d)
            x = q + self.mlp_map(torch.cat([q, mp], dim=-1))
            x = x + self.mlp_global(torch.cat([q, gl], dim=-1))

            # 和其他人：第 j 行是 `(q, pv_j, ob[p][j], ob[j][p])` —— 两个方向的遮挡都给，
            # 因为 `ob` 是有向的（7.91% 的 A↔B 帧两向不一致）。
            slots = torch.arange(N_PLAYERS, device=p.device)
            out_cls = aut[rows[:, None], p[:, None], slots]     # (n,10) ob[p][j]
            in_cls = aut[rows[:, None], slots, p[:, None]]      # (n,10) ob[j][p]
            pair_in = torch.cat(
                [
                    q[:, None, :].expand(n, N_PLAYERS, d),
                    pv,
                    F.one_hot(out_cls, N_EDGE_CLASS).to(q.dtype),
                    F.one_hot(in_cls, N_EDGE_CLASS).to(q.dtype),
                ],
                dim=-1,
            )                                                   # (n,10,2d+8)
            keep = alive & ~F.one_hot(p, N_PLAYERS).bool()      # 活人、且不是自己
            x = x + self.pool_pair(self.mlp_pair(pair_in), keep)

            # 道具：空槽不进池化（`_prop_vectors` 已把空的挤到后面并给了 `pr_ok`；
            # 一枪没开的帧就是 `Σ ok = 0`，见 `RelPool` 的掩码一段）。
            prop_in = torch.cat([q[:, None, :].expand(n, n_prop, d), pr], dim=-1)
            x = x + self.pool_prop(self.mlp_prop(prop_in), pr_ok)
            return self.norm_f(x)

        a = batch["pair"][:, 0:1].expand_as(idx)[idx]           # (n,)
        b = batch["pair"][:, 1:2].expand_as(idx)[idx]
        xa, xb = state(a), state(b)
        return (
            self.pair_proj(torch.cat([xa, xb], dim=-1)),
            self.pair_proj(torch.cat([xb, xa], dim=-1)),
        )

    # -------------------------------------------------------------------- 前向

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """返回 `{"logits": (B,T), "probs": (B,T), "pred_mask": (B,T) bool}`。

        第 `t` 列 = 看完第 `0..t` 帧的预测（每帧一个预测点）。
        """
        fm = batch["frame_mask"]                            # (B,T) bool
        idx = fm                                            # 同一张表：选帧 + 散回去
        fwd, rev = self._pair_vectors(batch, idx)           # (n,d) 各一个
        b, t = fm.shape

        # 帧号 = `round(seconds × 64)`：`seconds` 精确是 1/64 的整数倍，没有量化误差；
        # 随机截断之后 `seconds` 已被对零，所以 k 只表示"看了多少帧"，与击杀的远近无关。
        k = torch.round(batch["seconds"][idx].float() * self.cfg.tick_hz).long()
        te = self.time_emb(k.clamp_(0, self.cfg.time_bins - 1))
        fwd, rev = fwd + te, rev + te

        h = fwd.new_zeros(b, t, fwd.shape[-1])
        h[fm] = fwd                                         # 补齐帧保持 0，因果看不到
        if self.cfg.antisym:
            # 两个方向各跑一遍时间 transformer（拼成 (2B,T,d) 一次前向），出两个 logit
            # 再相减 → 交换 A/B 时精确取反。放在时间那一头**之后**的理由见模块 docstring。
            h_rev = fwd.new_zeros(b, t, fwd.shape[-1])
            h_rev[fm] = rev
            h = torch.cat([h, h_rev], dim=0)
            for blk in self.temporal:
                h = blk(h)
            logits = self.head(self.norm_t(h)).squeeze(-1)   # (2B,T)
            logits = logits[:b] - logits[b:]
        else:
            for blk in self.temporal:
                h = blk(h)
            logits = self.head(self.norm_t(h)).squeeze(-1)    # (B,T)
        return {"logits": logits, "probs": logits.sigmoid(), "pred_mask": fm}


def aim_loss(
    logits: torch.Tensor, label: torch.Tensor, pred_mask: torch.Tensor
) -> torch.Tensor:
    """逐预测点 BCE，**所有预测点等权**：`sum(loss * mask) / sum(mask)`。

    以前是"每条样本内部先平均、再对 batch 平均"。改成逐点等权是因为随机截断的 `T'`
    是**独立于标签**均匀抽的（见 `dataset.AimDataset`），"长窗口多贡献"不再等于"按交火
    时长给样本加权"——那是固定窗口时代的问题了。换过来有两个直接后果：

    * 短窗口不再被放大：旧口径下 `T'=6` 的样本每帧权重是 `T'=257` 的 **43 倍**，而 6 帧
      几乎不含信息，等于让噪声支配编码器。
    * 末段信号变强：旧口径下每个预测点的权重是 `1/T'`，`k<6` 的点单点权重是 `k>=200`
      的 **3.5 倍**（`E[1/T' | T'>k]` = 0.0153 vs 0.0044），逐点等权把每个点拉成 1。
      按"这一行总共收到多少梯度"算，`k>=200` 涨了 **1.7~2 倍**、早段掉一半。
      ⚠ `k=256` 只涨 2 倍，治不了根：它仍只有 `T'=257`（0.4%）的样本产生，真要让末帧
      那行训到得让 `T'` 偏向长窗口，不是靠改 loss 归一化。

    代价：梯度尺度随 batch 的 `Σ T'` 波动（±10% 量级；`T'` 与标签无关，所以是无偏抖动），
    `loss` 的绝对值也不再和旧 run 可比。`acc` 的口径没动（仍是每条样本等权），见
    `train.batch_stats`。

    `antisym` 开着时标签的含义不变（A 到底是不是击杀者）：模型算的是
    `z = w·h(A,B) − w·h(B,A)`，两个方向各一遍再减，所以 `swap_ab` 关掉、标签恒为 True
    也不会退化 —— 它就是"每对里二选一"的那个一。
    """
    target = label.to(logits.dtype).unsqueeze(1).expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    m = pred_mask.to(loss.dtype)
    return (loss * m).sum() / m.sum().clamp_min(1.0)
