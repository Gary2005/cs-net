"""瞄准数据集：**每次击杀 → 一个样本**，样本里是「窗口内每一帧的**全场**状态」。

一个回合会写成 N 个文件（N = 这个回合里能用的击杀数，0 也合法）：

    {地图}_f{fi}_r{rid}_k{序号}.npz

    python -m scripts.training_data.prepare --task aim
    python -m scripts.training_data --task aim

窗口是 `[受害者最后一个活帧 - LOOKBACK, 受害者最后一个活帧]`（两端都含），跨度
**固定 4 秒**。采样是 64 Hz 的整数倍（实测 `seconds` 逐帧 +0.015625 s），所以
`T = 4 × 64 + 1 = 257` —— **整个数据集成一个定值**（实测 449 条：min = p50 = max = 257）。
如果 `i1` 太靠前、窗口被截到回合开头，`window_before` 会从 0 开始、`T < 257`（现有
449 条里一条都没有），所以下游仍然按变长处理（`collate_aim` 的补齐逻辑保留，只是通常
补 0 帧）。
终点不用击杀帧 `k["i"]`：那一帧受害者已经死了，所有位置列都是 0，放进输入既是纯噪声、
又接近标签泄漏。真正的瞄准目标就是死前最后一帧的姿态。

**回合结束那一枪也算一个样本**（`_util.late_kills()`）：`end_reason ∈
{t_killed, ct_killed}` 的回合，最后一枪落在 `ticks[-1] + 1` —— 引擎在死亡那一 tick
才把 `total_rounds_played` 加一，所以那一帧的采样已经算进**下一个**回合了，
`kills()` 只收 `ticks[0] <= t <= ticks[-1]`，会把它整条滤掉。实测 4 份 demo 的 78 个
回合里有 **49 个（63%）** 是这么结束的：不补的话"打死最后一个敌人"这一枪既不在训练
集里、回放页最后 3 秒也没有对枪弧线（推理侧同一件事见 `app/ai/service.py::_round_kills`）。
它的窗口终点仍是受害者最后一个活帧（`last_alive()` 会自己往前找），所以
`T = 257` 这条约定不变。

早先版本在 `[0.1, 4]` 秒上**随机**取跨度，理由是不想让"标签恒落在窗口末端"变成模型
可以直接利用的规律。改成定长 4 秒的收益很实在：`T` 一致之后 batch 内不用补齐（`(B,T,…)`
一次成型，省掉 copy 和 padding 的显存），而且"只看末端"这条捷径本来就被**逐帧预测**
挡住了 —— 每个预测点各算一次 BCE，只读末帧的模型在早帧上一样要交损失。真要防它，
把随机 span 加回来即可，数据这一侧的接口不用动。

## 为什么是「全场 × 每帧」

旧版每个样本只存 **atk / tgt 两个人**的逐帧轨迹 + 其余 8 个人的**首帧快照** +
两张**首帧**视角渲染图。两处硬伤，都是结构性的：

1. **另外 8 个人只有一帧。** 击杀经常是第三方造成的 —— 队友补枪、被别人打残、绕后
   夹击。首帧快照里连"他后来往哪走了"都没有，模型不可能学到这些。
2. **视角图是首帧渲染的。** 击杀在窗口末尾，中间那 0.1–4 s 的走位、拉枪、绕掩体
   一点不在图里。实测（449 条样本）：78% 的帧目标落在视野框里，但**只有 17.6%
   真没被墙/烟挡住**；这 17.6% 里信号是硬的（"只有一方看得见"的 39 条中，看见的
   那个人最后就是杀手的占 79.5%），可就算吃满也只值 **+0.2 个点**。图本身不值钱，
   值钱的是"谁挡着谁"这个**关系**。

所以现在：逐帧存 10 个玩家的状态、逐帧存遮挡矩阵、逐帧存道具，**不存渲染图**。
一帧一个人 20 个数、一对人 1 个遮挡值，比一张 192×256×5 的图小 3 个数量级，
信息却更贴近任务。

换成遮挡矩阵之后这个结论更硬了（449 条实测）：固定 4 秒窗口的**第一帧**只有
**3.6%** 的样本里 A 看得见 B，**死前最后一帧**是 **86.0%** —— 首帧视角图恰好取在
信息量最低的那一刻，而击杀本身就发生在"终于看见了"的瞬间。

（这两个数在烟雾改成凸包体积之后变过：壳版是 4.9% / 91.5%。原因是按壳判遮挡会漏掉
近一半"其实被烟挡着"的格子，而隔着烟打死人恰好是"最后一帧看起来没挡"的一种。现在
全量 449 条的活人有向对里 **烟占 2.10%**（墙 87.26%、通 10.64%），壳版只有 1.08%。）

## 样本里的 key

    ctx_player       (T, 10, 20) float32  逐帧全场状态，列序 = AIM_FEATURES，**每列都在
                                          [0,1]**（映射见 `_SCALE`）。第 p 行永远是
                                          `rnd["players"][p]`（和击杀事件的 a / v 同一套
                                          下标）。死人的行**不保证是 0**（位置列可能是
                                          0、也可能残留最后一帧的值），**只认
                                          `ctx_player_mask`**，别从数值上猜生死。
    ctx_player_mask  (T, 10)     int8     每一格的角色 / 生死：
                                            0 = 死亡（或没进场）
                                            1 = **A，击杀者**
                                            2 = **B，受害者**
                                            3 = 活着，但不是这一对
                                          A、B 由击杀事件决定，**沿窗口恒定**（窗口成立
                                          的前提就是这两人全程活着）。
                                          ⚠ 3 是必须的：没有它，"活着但不是当事人"和
                                          "死亡"就分不开，模型会以为场上只有两个人。
    ctx_player_weapon (T, 10)    int8     逐帧武器（`constants.WEAPON_NAMES` 的全局下标）。
                                          -1 = 没有武器 / 已死亡。跨 demo 可比：本文件的
                                          下标已经过 `store.weapon_lut(fi)` 换算。
    obstruction      (T, 10, 10) int8     逐帧两两遮挡：`obstruction[t][i][j]` = 从 i 的
                                          **眼睛**看 j 的**身体**有没有被挡住
                                            0 = 没挡（至少露出一个角）
                                            1 = 被烟雾挡住
                                            2 = 被地图 mesh 挡住
                                          不看视野（FOV）—— 朝哪看是模型自己的输入。
                                          判据、8 个顶点、`min` 的取法见
                                          `training_data/obstruction.py`。
                                          ⚠ **0 ≠ "看得见"**：0 只说明"中间没有墙 / 烟"，
                                          对方可能正在背对着（`yaw` 差 138° 也是 0）。
                                          `visibility.js::pairVisibility()` 里
                                          `BLOCK_FOV` 是**另一个位**，没有进这张表。
                                          ⚠ **有方向**：`[i][j]` 和 `[j][i]` 可以不同
                                          （两个人的眼睛位置不一样）。死人的行/列全 0。
    ctx_projectile   (T, 32, 4)  float32  逐帧道具快照，每行 `[x y z t]`，四列都在 [0,1]：
                                          x y z = 世界坐标（和 ctx_player 同一套地图
                                          归一化，再 (v+1)/2）；
                                          t = 该道具已经存在了多少秒 / 100 再 clip。
                                          不足 32 行补 0，配合 `ctx_projectile_id` 的 -1；
                                          超过 32 个时按 id 从小到大保留（先丢 id 大的）。
    ctx_projectile_id (T, 32)    int8     逐帧每行是什么，-1 = 空：
                                            0 落地烟  1 落地火  2 飞行闪光  3 飞行诱饵
                                            4 飞行手雷  5 飞行燃烧瓶  6 飞行烟
                                          "飞行"只算**位置还在变**的采样（烟/诱饵落地、
                                          HE 爆炸后实体仍会被逐帧报告上千帧，那段不算）。
    global_feat      (T, 5)      float32  逐帧全局状态 `[x y z 安放时长 回合时间]`，五列都在
                                          [0,1]：
                                          x y z = 炸弹位置（同一套地图归一化）；
                                          第 4 列 = 距安放的秒数 / 50（未安放恒 0）；
                                          第 5 列 = 回合时间秒数 / 200（0 点 =
                                          `round_freeze_end`）。
                                          ⚠ **逐帧**：炸弹会被搬运、掉落、安放，回合时间
                                          更是在走；只存一帧就等于把"后来发生了什么"
                                          抹掉，模型还得自己拿 `seconds` 去外推。
                                          ⚠ 未安放时 xyz 是**携带者**位置 —— 对敌方形同
                                          隐藏信息；⚠ 掉在地上的那段坐标冻结在掉落点。
                                          ⚠ 窗口内任一帧的 bomb 位置缺失时**整条样本跳过**
                                          （实测 456 662 帧里只有 8 帧缺，都在回合最开头，
                                          所以这条几乎不丢样本）。
    seconds          (T,)        float32  相对窗口第一帧的秒数，第一个恒为 0.0
    map_id           ()          int8     地图编号（`constants.MAP_TO_IDX`，0..7）。不在那
                                          8 张图里的回合**整回合跳过**。
    hs / dmg         ()          bool / int64   爆头 / 这一击的伤害（能 > 100）

武器列**没有**混进 `ctx_player` 那个 float32 矩阵：离散下标混进去既会变浮点、又不好查表。
遮挡矩阵同理单独存 int8（它也不是 [0,1] 的连续量，混进去会破坏"所有列都在 [0,1]"这条
模型侧依赖的性质）。

## 存储

一条 `T=257` 的样本解压后 **371 KB**：`ctx_player` 201 KB、`ctx_projectile` 129 KB
（绝大多数行是全 0）、`obstruction` 25 KB、`global_feat` 5 KB。旧的视角图是
2 × 1.97 MB = **3.9 MB**。`np.savez_compressed` 之后实测 **p50 66 KB**（28–98 KB，
449 条共 28.7 MB）；旧版单样本 320 KB（视角图压不动）。

造数成本上遮挡是唯一的增量：一帧 `n_alive²×8` 条射线，实测 **0.5 ms/帧**（10 人全活，
mirage），一条 `T=257` 的样本纯遮挡约 0.13 s；`Occluder` 按**回合**缓存帧结果，所以同一
回合里互相重叠的几条击杀窗口只算一遍。四份 csz 全量重造 **159 秒 / 449 条**
（≈0.35 s 一条，含解码；旧版随机窗口是 132 秒）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..constants import MAP_TO_IDX, N_PLAYERS, TEAM_TO_IDX, TEAM_UNKNOWN
from ..features import normalize_xyz
from ..obstruction import Occluder
from ._util import alive_track, kills, last_alive, late_kills, track, window_before
from .base import Task

#: 直接从 `rnd["players"][p]` 取、之后**只做缩放**的列组。
_XYZ_KEYS = ("x", "y", "z")
_ANG_KEYS = ("yaw", "pitch")
_VEL_KEYS = ("vx", "vy", "vz")
_MISC_KEYS = ("hp", "crouch", "armor", "helmet")

#: 样本里 F 轴的最终列序（`ctx_player` 的最后一维就是它）。
#:
#: yaw / pitch 存的是 **sin/cos 四列**，不是角度：yaw 在 ±180 处跳变、pitch 在
#: ±89 处折返，原始角度是模型插值不了的不连续量。代价是这两个量从 2 列变 4 列。
#:
#: `team`（第 15 列）是**常量列**（CT=0 / T=1，沿时间不变）：整回合只有一个值，拼进
#: 矩阵只是为了让"一个张量带走全部输入"。代价是同一 bit 重复 T 遍。
#:
#: 最后五列 `shot` / `since_shot` / `has_kill` / `blinded` / `defuser` 是**因果状态列**：
#: 第 i 行的值只依赖第 i 帧及之前的数据，所以窗口切片不会泄露未来。
#:
#: ⚠ `blinded` / `defuser` 是玩家**私有**状态（对手看不到谁被闪、谁有钳子）；现在 10
#: 个人都有这一列，做单视角模型时别把别人那两列当成可观测输入。
AIM_FEATURES: tuple[str, ...] = (
    "x", "y", "z",
    "yaw_sin", "yaw_cos", "pitch_sin", "pitch_cos",
    "vx", "vy", "vz",
    "hp", "crouch", "armor", "helmet",
    "team",
    "shot", "since_shot", "has_kill",
    "blinded", "defuser",
)

#: `ctx_player_mask` 的取值。0 是死亡，1/2 是这一对的双方，3 是"活着但不相干"。
MASK_DEAD = 0
MASK_ATK = 1
MASK_TGT = 2
MASK_OTHER = 3
MASK_NAMES: tuple[str, str, str, str] = ("死亡", "A(击杀者)", "B(受害者)", "其他存活")

#: 把每一列压到 **[0, 1]** 的仿射系数：`v01 = clip(v * scale + bias, 0, 1)`。
#: 顺序和 `AIM_FEATURES` 一一对应（下面有 assert 兜着）。
#:
#: 有符号的列用 `0.5 / half` + `0.5`：把 `[-half, +half]` 映到 `[0, 1]`，0 正好落在
#: 0.5。`half` 的来源：
#:
#:     x y z      `normalize_xyz()` 之后已在 [-1, 1]        → half = 1
#:     sin / cos  值域本来就是 [-1, 1]                      → half = 1
#:     vx vy      实测水平速度 ≤ 282 HU/s                   → half = 300
#:     vz         实测最大 ±1152 HU/s                       → half = 1200
#:     hp armor   0..100                                    → ÷100
#:     crouch     原生 0..1（m_flDuckAmount）               → 原样
#:     helmet     bool                                       → 原样
#:     team       CT=0 / T=1                                 → 原样
#:     shot       0/1                                         → 原样
#:     has_kill   0/1                                         → 原样
#:     since_shot 秒（`_SHOT_TIME_SPAN` 为饱和值）             → ÷100
#:     blinded    0/1                                         → 原样
#:     defuser    0/1                                         → 原样
#:
#: clip 只兜极端值（实测最大速度就贴在 half 附近），正常样本几乎不会被裁。
_SHOT_TIME_SPAN = 100.0   # since_shot 的归一化分母，也是"本回合还没开过枪"的饱和值

_SCALE = np.array(
    [
        0.5, 0.5, 0.5,                             # x y z
        0.5, 0.5, 0.5, 0.5,                        # yaw_sin yaw_cos pitch_sin pitch_cos
        0.5 / 300.0, 0.5 / 300.0, 0.5 / 1200.0,    # vx vy vz（HU/s）
        1.0 / 100.0, 1.0, 1.0 / 100.0, 1.0,        # hp crouch armor helmet
        1.0,                                       # team
        1.0, 1.0 / _SHOT_TIME_SPAN, 1.0,           # shot since_shot has_kill
        1.0,                                       # blinded
        1.0,                                       # defuser
    ],
    dtype=np.float32,
)
_BIAS = np.array(
    [
        0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5,
        0.0, 0.0, 0.0, 0.0,
        0.0,
        0.0, 0.0, 0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
assert len(_SCALE) == len(_BIAS) == len(AIM_FEATURES), "映射表必须和 AIM_FEATURES 对齐"

#: 窗口跨度（秒），**固定值**：窗口 = `[死前最后一帧 - LOOKBACK, 死前最后一帧]`。
#: 写进 `task.json` 的 `desc` 也是从它拼的，改这里就够了。
#:
#: 64 Hz 采样下 `T = LOOKBACK × 64 + 1 = 257`，**全样本同长**（早先是 0.1–4 s 随机，
#: `T` 在 7–253 之间飘，每个 batch 都要补到最长）。
#:
#: ⚠ 改这个值要联动 `training/aim/model.py` 的 `time_bins`：那是帧号
#: `k = seconds × tick_hz` 的查表槽数，要装下 `0..LOOKBACK×64` 共 257 个值。
LOOKBACK = 4.0


#: 度 → 弧度。
_DEG2RAD = np.float32(np.pi / 180.0)

#: `ctx_projectile` 每行的列序：x y z t，**四列都在 [0,1]**。
PROJECTILE_FEATURES: tuple[str, ...] = ("x", "y", "z", "t")

#: 一行最多放几个道具。超出时按 id 从小到大保留（也就是先丢 id 大的）。
_PROJ_MAX = 32

#: 时间归一化：`elapsed_seconds / 100` 再 clip 到 [0,1]。
#: 用 100 而不是 30：最长的烟实测 22 s、火 ~18 s、投掷物飞行 ≤4.2 s，30 其实也够，
#: 但 100 留了 3 倍余量（换地图 / 换节奏都不会贴到 1）。
_PROJ_TIME_SPAN = 100.0

#: `global_feat` 第 4 列的时间归一化：`距安放的秒数 / 50` 再 clip。
#: C4 计时 40 s，实测安放后 40 s 内回合一定结束，50 留一点余量，正常不会贴到 1。
_BOMB_TIME_SPAN = 50.0

#: `global_feat` 第 5 列的时间归一化：`回合时间秒数 / 200` 再 clip。
#: 0 点 = 本回合的 `round_freeze_end`（`rnd["round_seconds"]` 就是这么算的）。
_ROUND_TIME_SPAN = 200.0

#: `ctx_projectile_id` 的取值：0/1 是"已经生效"的，2..6 是"还在飞"的。
_PROJ_LANDED_SMOKE = 0      # 落地烟：位置 = smokegrenade_detonate 的 x,y,z
_PROJ_LANDED_FIRE = 1       # 落地火：位置 = rnd["fire"][i]["origin"]
_PROJ_FLYING_SMOKE = 6
_PROJ_FLYING: dict[str, int] = {
    "CFlashbangProjectile": 2,
    "CDecoyProjectile": 3,
    "CHEGrenadeProjectile": 4,
    "CMolotovProjectile": 5,
    "CSmokeGrenadeProjectile": _PROJ_FLYING_SMOKE,
}


def _player_arrays(rnd: dict, p: int, map_name: str, team_id: int) -> np.ndarray:
    """某个玩家的整回合特征 `(T, len(AIM_FEATURES))`，**每一列都在 [0, 1]**。

    `team_id` 是 `constants.TEAM_TO_IDX` 的编号（CT=0 / T=1），作为**常量列**拼进去
    （第 15 列，沿时间不变）。

    最后五列是**逐帧状态量**，只用第 i 帧及之前的数据：
        shot       本帧是否开枪（`players[p]["shots"]`，0/1）
        since_shot 距上次开枪的秒数（秒数除以 `_SHOT_TIME_SPAN` 后 clip）；
                   本回合还没开过枪 → 直接取饱和值 1.0
        has_kill   截至本帧是否已取得过击杀（一旦为 1，之后恒为 1）
        blinded    本帧是否被闪光弹致盲（`players[p]["flash"]` 剩余秒数 > 0，0/1）。
                   `flash` 缺失（NaN）时按"没被闪"处理——`NaN > 0` 本来就是 False。
        defuser    本帧是否带着拆弹器（`players[p]["defuser"]` 直接就是 0/1 bool）。

    每一步都产生新数组（`track()` 的 `np.stack`、`normalize_xyz()` 的 copy、
    `np.concatenate`、`np.clip`），所以不会污染 `rnd` 缓存。
    """
    xyz = normalize_xyz(track(rnd, p, _XYZ_KEYS), map_name)   # 减中心、裁 ±5000/±2000、除范围
    ang = track(rnd, p, _ANG_KEYS)                            # (T, 2) **度**
    vel = track(rnd, p, _VEL_KEYS)                            # (T, 3) HU/s，缩放交给 _SCALE
    misc = track(rnd, p, _MISC_KEYS)                          # hp, crouch, armor, helmet

    n = len(xyz)

    # ── 状态列：开枪 / 距上次开枪 / 是否已击杀 / 是否被闪 ────────────────
    shot = np.asarray(rnd["players"][p]["shots"], dtype=np.float64)   # (T,) 0/1
    secs = np.asarray(rnd["round_seconds"], dtype=np.float64)
    # last[i] = 第 i 帧（含）之前最后一次开枪的采样下标，-1 = 本回合还没开过枪
    last = np.maximum.accumulate(np.where(shot > 0, np.arange(n), -1))
    since_shot = np.full(n, _SHOT_TIME_SPAN, dtype=np.float64)        # 没开过枪 → 饱和 1.0
    fired = last >= 0
    since_shot[fired] = secs[fired] - secs[last[fired]]

    has_kill = np.zeros(n, dtype=np.float64)
    for e in kills(rnd):                       # 已滤掉热身等不属于本回合的击杀
        if e["a"] == p:
            has_kill[e["i"]:] = 1.0

    # 被闪：`flash` 是**剩余秒数**（实测 0..5.129）。`flash_alpha` 是最大亮度（p50
    # 已经是 255，没被闪时也是 255），不能拿来判"是否被闪"。NaN（缺采样）按未被闪。
    flash = np.asarray(rnd["players"][p]["flash"], dtype=np.float64)
    blinded = (flash > 0.0).astype(np.float64)

    # 拆弹器：`defuser` 本来就是逐帧 bool（提取层的 `has_defuser`），直接转 float。
    defuser = np.asarray(rnd["players"][p]["defuser"], dtype=np.float64)

    yaw, pitch = ang[:, 0] * _DEG2RAD, ang[:, 1] * _DEG2RAD
    raw = np.concatenate(
        [
            xyz,
            np.stack([np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)], axis=-1),
            vel,
            misc,
            np.full((n, 1), team_id, dtype=np.float32),   # team：常量列
            shot[:, None],
            since_shot[:, None],
            has_kill[:, None],
            blinded[:, None],
            defuser[:, None],
        ],
        axis=-1,
    ).astype(np.float32)

    # `track()` 把原始 `None` 变成 nan（实测 round 0 开头有 2 帧没有速度）。不填就
    # 违反了"每一列都在 [0,1]"，还会把 nan 带进 loss。填 0 的语义：有符号列正好回到
    # 0.5（中性），而这些帧本来就没有可用信息。
    raw = np.nan_to_num(raw, nan=0.0)

    # 逐列仿射到 [0,1] 再裁剪（有符号列映射成 0.5 为中心）。
    return np.clip(raw * _SCALE + _BIAS, 0.0, 1.0).astype(np.float32)


def _global_weapon(rnd: dict, p: int, lut: np.ndarray) -> np.ndarray:
    """某个玩家的武器 `(T,)` int8 —— **全局**下标（-1 = 无/未知）。

    `rnd["players"][p]["weapon"]` 存的是**本文件**编号，跨 demo 不可比（实测同一个
    12 在 4 个文件里分别是 High Explosive Grenade / Huntsman Knife / Glock-18 /
    Incendiary Grenade）。`lut` 来自 `store.weapon_lut(fi)`。

    ⚠ 必须先按 `>= 0` 掩掉再索引：`-1` 是负索引，`lut[-1]` 会**静默**取到表里最后
    一个武器。实测 `weapon == -1` 约占 10% 的采样（死人 / 无武器）。
    """
    w = np.asarray(rnd["players"][p]["weapon"], dtype=np.int64)
    out = np.full(w.shape, -1, dtype=np.int8)
    ok = w >= 0
    out[ok] = lut[w[ok]].astype(np.int8)
    return out


class _Round:
    """一个回合的"全场 × 时间"原料，**整回合算一次**，每条击杀切片取走自己要的窗口。

    这样做的原因很直接：一个回合里 10 个玩家 × 每帧 20 个特征，逐条击杀各算一遍会把
    同一个人重复算 2~5 次（一个回合平均产出 5 条样本）。而每帧的遮挡矩阵由 `Occluder`
    自己做**逐帧**缓存，重叠窗口也只算一遍。
    """

    def __init__(self, store, fi: int, rnd: dict, map_name: str, *,
                 on_player=None, on_stage=None) -> None:
        # `on_player(p)` / `on_stage(name)` 是给网页进度条用的（默认 None：造训练
        # 数据时没有任何额外开销，也不影响产出 —— 它们只在某一步算完之后被调，
        # 什么时候调不改变任何数值）。这一版最慢的一步是下面 `Occluder` 的逐帧
        # 几何 + 地图 BVH，光靠"玩家 1/10"是看不出它在干什么的。
        self.rnd = rnd
        self.map_name = map_name
        self.secs = np.asarray(rnd["round_seconds"], dtype=np.float64)
        self.ticks = np.asarray(rnd["ticks"], dtype=np.int64)
        self.n_frames = len(self.ticks)
        self.n_pl = min(N_PLAYERS, len(rnd["players"]))
        self.lut = store.weapon_lut(fi)
        teams = rnd.get("teams") or []
        #: `(N_PLAYERS,)` int8：CT=0 / T=1 / -1 未知。整回合不变。
        self.team = np.full(N_PLAYERS, TEAM_UNKNOWN, np.int8)
        for p in range(self.n_pl):
            self.team[p] = TEAM_TO_IDX.get(teams[p] if p < len(teams) else "?", TEAM_UNKNOWN)

        #: `(N_PLAYERS, T, 20)`，缺的玩家（不足 10 人）留 0。
        self.feat = np.zeros((N_PLAYERS, self.n_frames, len(AIM_FEATURES)), np.float32)
        #: `(N_PLAYERS, T)` bool
        self.alive = np.zeros((N_PLAYERS, self.n_frames), bool)
        #: `(N_PLAYERS, T)` int8 全局武器下标
        self.weapon = np.full((N_PLAYERS, self.n_frames), -1, np.int8)

        for p in range(self.n_pl):
            self.feat[p] = _player_arrays(rnd, p, map_name, self.team[p])[: self.n_frames]
            self.alive[p] = alive_track(rnd, p)[: self.n_frames]
            self.weapon[p] = _global_weapon(rnd, p, self.lut)[: self.n_frames]
            if on_player is not None:
                on_player(p)
        # 死人不可能拿着枪：位置列可以是残留值，但武器别留下假信号。
        self.weapon[~self.alive] = -1

        if on_stage is not None:
            on_stage("准备地图 / 逐帧遮挡几何")
        self.occluder = Occluder(rnd, map_name, N_PLAYERS)

        # ── 道具的原料（整回合只取一次）────────────────────────────────────
        if on_stage is not None:
            on_stage("整理道具 / 手雷原料")
        ev = rnd.get("events") or {}
        self.grenades: list[dict] = ev.get("grenades") or []
        self.g_ticks = np.fromiter(
            (g["t"] for g in self.grenades), dtype=np.int64, count=len(self.grenades)
        )
        # 每个 (类型, eid) 第一次被采样的 tick ≈ 出手时刻（实测离投掷者 ~75 HU）。
        # 列表已按 t 排序，所以 setdefault 就是"首次出现"。
        self.throw_tick: dict[tuple[str, int], int] = {}
        for g in self.grenades:
            self.throw_tick.setdefault((g["ty"], g["eid"]), int(g["t"]))

        # 每个投掷物最后一次"位置变化"的 tick。之后的采样是**静置尾**：烟/诱饵落地、
        # HE 爆炸之后，实体还会被逐帧报告几百到上千帧（实测烟 1414 帧、诱饵 1085、
        # HE 321），这一段不能再算"飞行中"。
        self.stop_tick: dict[tuple[str, int], int] = {}
        groups: dict[tuple[str, int], list[dict]] = {}
        for g in self.grenades:
            groups.setdefault((g["ty"], g["eid"]), []).append(g)
        for key, gs in groups.items():
            p = np.array([[g["x"], g["y"], g["z"]] for g in gs], dtype=np.float64)
            moved = np.flatnonzero(np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-4)
            # moved[-1] 是"最后一次移动"那一步的**前一个**采样，所以要 +1
            self.stop_tick[key] = (
                int(gs[int(moved[-1]) + 1]["t"]) if moved.size else int(gs[0]["t"]) - 1
            )

        self.smoke_events: list[dict] = ev.get("smokes") or []
        self.fire_insts: list[dict] = rnd.get("fire") or []

        # ── 炸弹：整回合位置（先归一化好）+ 安放时刻 ─────────────────────
        bomb_pos = rnd.get("bomb_position") or []
        bomb_xyz = np.full((self.n_frames, 3), np.nan, dtype=np.float32)
        known = [i for i, q in enumerate(bomb_pos[: self.n_frames]) if q is not None]
        if known:
            bomb_xyz[known] = np.asarray([bomb_pos[i] for i in known], dtype=np.float32)
        self.bomb_ok = np.zeros(self.n_frames, bool)
        self.bomb_ok[known] = True
        self.bomb_norm = np.nan_to_num(
            normalize_xyz(bomb_xyz, map_name) * 0.5 + 0.5, nan=0.5
        ).astype(np.float32)
        # 安放时刻只认 `bomb_planted` 这个**逐 tick 列**，不用 `bomb_planted_time`
        # 字段：后者会把上一回合的安放事件算进本回合（实测 de_mirage round 7 拿到
        # s = -22.2 s，而该回合根本没下包）。
        planted = np.flatnonzero(np.asarray(rnd.get("bomb_planted") or [], dtype=bool))
        self.bomb_planted_s = float(self.secs[planted[0]]) if planted.size else None

    # ------------------------------------------------------------------ 切片

    def mask(self, sl: slice, atk: int, tgt: int) -> np.ndarray:
        """`(T, 10)` int8：0 死 / 1 A / 2 B / 3 其他存活。"""
        alive = self.alive[:, sl].T                       # (T, 10)
        out = np.where(alive, np.int8(MASK_OTHER), np.int8(MASK_DEAD)).astype(np.int8)
        out[:, atk] = MASK_ATK
        out[:, tgt] = MASK_TGT
        return out

    def projectiles(self, sl: slice) -> tuple[np.ndarray, np.ndarray]:
        """窗口内**逐帧**的道具快照 `(T,32,4)` + `(T,32)`。

        和旧版逐帧语义完全一样，只是从"只取第一帧"变成"每一帧都取"：
        xyz 和 `ctx_player` 用同一套地图归一化再 `(v+1)/2`；t = 该道具已经存在了多少
        秒 / 100。不足 32 行补 0、id 填 -1；超过 32 行时按 id 升序保留（先丢 id 大的）。
        """
        n = sl.stop - sl.start
        out = np.zeros((n, _PROJ_MAX, len(PROJECTILE_FEATURES)), dtype=np.float32)
        out_id = np.full((n, _PROJ_MAX), -1, dtype=np.int8)

        for t_out, i in enumerate(range(sl.start, sl.stop)):
            tick = int(self.ticks[i])
            now = float(self.secs[i])
            ids: list[int] = []
            xyz: list[tuple[float, float, float]] = []
            starts: list[int] = []                  # 起始 tick，最后统一换成秒

            # ① 生效中的烟（0）：用事件引爆点，和烟体素用的是同一个坐标系
            landed: set[int] = set()
            for s in self.smoke_events:
                if s["t0"] <= tick and (s["t1"] is None or tick <= s["t1"]):
                    landed.add(int(s["eid"]))
                    ids.append(_PROJ_LANDED_SMOKE)
                    xyz.append((s["x"], s["y"], s["z"]))
                    starts.append(int(s["t0"]))

            # ② 生效中的火（1）：origin 是起火点；匹配不到 origin 的放不进世界坐标
            for inst in self.fire_insts:
                origin = inst.get("origin")
                if origin is None:
                    continue
                if inst["first_tick"] <= tick <= inst["last_tick"]:
                    ids.append(_PROJ_LANDED_FIRE)
                    xyz.append((float(origin[0]), float(origin[1]), float(origin[2])))
                    starts.append(int(inst["first_tick"]))

            # ③ 还在飞的道具（2..6）：只取这一 tick 的采样行
            lo = int(np.searchsorted(self.g_ticks, tick, "left"))
            hi = int(np.searchsorted(self.g_ticks, tick, "right"))
            for g in self.grenades[lo:hi]:
                pid = _PROJ_FLYING.get(g["ty"])
                if pid is None:
                    continue                        # 未来新增的类型：宁可不放
                if g["t"] > self.stop_tick[(g["ty"], g["eid"])]:
                    continue                        # 已经落地/炸了：静置尾不算飞行
                if pid == _PROJ_FLYING_SMOKE and int(g["eid"]) in landed:
                    continue                        # 烟已落地：上面按 0 收过了
                ids.append(pid)
                xyz.append((g["x"], g["y"], g["z"]))
                starts.append(self.throw_tick[(g["ty"], g["eid"])])

            if not ids:
                continue
            keep = sorted(range(len(ids)), key=lambda q: ids[q])[:_PROJ_MAX]  # 先丢 id 大的
            pts = normalize_xyz(np.asarray([xyz[q] for q in keep], dtype=np.float32), self.map_name)
            out[t_out, : len(keep), :3] = pts * 0.5 + 0.5    # 和 ctx_player 同一套 [0,1]
            elapsed = now - np.interp(
                np.asarray([starts[q] for q in keep], dtype=np.float64), self.ticks, self.secs
            )
            out[t_out, : len(keep), 3] = np.clip(elapsed / _PROJ_TIME_SPAN, 0.0, 1.0)
            out_id[t_out, : len(keep)] = [ids[q] for q in keep]
        return out, out_id

    def global_feat(self, sl: slice) -> np.ndarray:
        """`(T, 5)` **逐帧**全局状态：炸弹位置 + 已安放多久 + 回合时间。

        三者本来都是逐帧量（炸弹会被搬运 / 掉落 / 安放，计时器一直在走），所以这里
        直接按窗口切片，不再只取首帧：

            out[:, :3] = bomb_norm[sl]                          # 世界坐标 → [0,1]
            out[:, 3]  = (secs[sl] - 安放时刻) / 50             # 未安放恒 0
            out[:, 4]  = secs[sl] / 200                         # 回合时间

        `np.clip` 只兜极端值（安放后 40 s 内回合必定结束，`/50` 正常到不了 1）。
        """
        n = sl.stop - sl.start
        out = np.empty((n, 5), dtype=np.float32)
        out[:, :3] = self.bomb_norm[sl]
        if self.bomb_planted_s is None:
            out[:, 3] = 0.0                    # 本回合没下包
        else:
            out[:, 3] = np.clip(
                (self.secs[sl] - self.bomb_planted_s) / _BOMB_TIME_SPAN, 0.0, 1.0
            )
        out[:, 4] = np.clip(self.secs[sl] / _ROUND_TIME_SPAN, 0.0, 1.0)
        return out


class Aim(Task):
    ext = ".npz"

    # 从 LOOKBACK 拼，改常量时不会留下 desc 和实际窗口不一致
    desc = (
        f"击杀前固定 {LOOKBACK:g} 秒（{int(LOOKBACK * 64) + 1} 帧）的**逐帧全场**状态 + "
        "两两遮挡（瞄准数据集，一次击杀一个样本，含回合结束那一枪）"
    )

    def n_samples(self, samples: list[Path]) -> int:
        return len(samples)

    def write_round(self, store, *, fi: int, rid: int, split: str, path: Path) -> None:
        map_name = store.map_name(fi)                              # 'de_mirage'
        map_id = MAP_TO_IDX.get(map_name)
        if map_id is None:
            return      # 不在这 8 张图里：normalize_xyz 会静默失真，整个回合一条都不产出

        rnd = store.round(fi, rid)
        rr = _Round(store, fi, rnd, map_name)
        secs = rr.secs

        count = 0
        # `kills()` 已滤掉不属于本回合的（round 0 的热身等）；`late_kills()` 再补上
        # "打死最后一个敌人、回合立刻结束"那一枪（`t = ticks[-1] + 1`，实测 63% 的
        # 回合这么结束）。它挂在最后一帧之后，`last_alive` 会自己找到死前一帧。
        for k in kills(rnd) + late_kills(rnd, n_frames=rr.n_frames):
            a, v = k["a"], k["v"]
            if a < 0 or v < 0 or a == v:
                continue                           # 摔死 / 被 C4 炸死 / 自杀：没有对手
            if a >= rr.n_pl or v >= rr.n_pl:
                continue

            ta, tv = int(rr.team[a]), int(rr.team[v])
            if ta == tv or min(ta, tv) < 0:
                continue                           # 团队击杀 / 队伍未知：只留敌人

            i1 = k["i"]                            # 击杀事件落在哪个采样点
            jv = last_alive(rr.alive[v], i1)       # 死前最后一个活帧（击杀帧位置列是 0）
            if jv < 0:
                continue                           # 这段里一次没活过：缺数据 / 没进场
            # 收尾那一枪（`i1 == n_frames`，没有对应帧）：受害者必须活到最后一帧，
            # 窗口才是"死前一帧"。不是的话说明这份数据被截断了 —— 跳过，免得拿更早
            # 那次死亡再造一条重复样本。
            if i1 >= rr.n_frames and jv != rr.n_frames - 1:
                continue

            # 窗口跨度固定 4 秒（`LOOKBACK`）：64 Hz 下 T = 257，全样本同长
            win = window_before(secs, jv, LOOKBACK)
            if not (rr.alive[v, win].all() and rr.alive[a, win].all()):
                continue                           # 窗口内双方死过 / 缺数据：整段不留
            # 炸弹位置逐帧存，所以窗口内**每一帧**都得有（`bomb_position[i] is not
            # None`）。缺失只在回合最开头出现（实测 456 662 帧里 8 帧），这时整条样本
            # 跳过、不写文件 —— `global_feat` 没有可填的值，填 0.5 等于凭空造一颗
            # 地图中心的炸弹。
            if not rr.bomb_ok[win].all():
                continue

            ctx_player = np.ascontiguousarray(rr.feat[:, win].transpose(1, 0, 2))  # (T,10,20)
            ctx_mask = rr.mask(win, a, v)                                        # (T,10)
            ctx_weapon = np.ascontiguousarray(rr.weapon[:, win].T)               # (T,10)
            obstruction = rr.occluder.window(win.start, win.stop)                # (T,10,10)
            ctx_proj, ctx_proj_id = rr.projectiles(win)                          # (T,32,4)/(T,32)

            count += 1
            np.savez_compressed(
                path.with_name(f"{path.stem}_k{count:03d}{path.suffix}"),
                ctx_player=ctx_player,                 # (T,10,20) [0,1]，死人的行别当真
                ctx_player_mask=ctx_mask,              # (T,10) 0 死 / 1 A / 2 B / 3 其他
                ctx_player_weapon=ctx_weapon,          # (T,10) 全局下标，-1 = 无/死
                obstruction=obstruction,               # (T,10,10) 0 通 / 1 烟 / 2 墙
                ctx_projectile=ctx_proj,               # (T,32,4) x y z t，全 [0,1]
                ctx_projectile_id=ctx_proj_id,         # (T,32) -1 = 空
                global_feat=rr.global_feat(win),       # (T,5) 逐帧：xyz 安放 回合时间
                seconds=(secs[win] - secs[win.start]).astype(np.float32),  # (T,)
                map_id=np.int8(map_id),                # () 0..7
                hs=np.bool_(k["hs"]),
                dmg=np.int64(k["dmg"]),
            )

    def read_sample(self, samples: list[Path], i: int) -> Any:
        with np.load(samples[i]) as z:
            return {k: z[k] for k in z.files}


TASK = Aim()

if __name__ == "__main__":
    pass
