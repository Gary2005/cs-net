"""参考实现：怎么把回合数据切成"输入特征 / 标签"（**窗口形状**）。

⚠ 你现在写的取数逻辑在 `prepare.py`（造数据）和 `dataset.py` 的 `SampleDataset`
（读数据）里，**不一定要用这个文件的写法**。
这个文件是"一个窗口 = 一个样本"那套留下的参考实现：`build_window()` /
`build_targets()` 的签名都带 `WindowRef`。留着它是因为里面有两样东西值得照用：

    round_events() / events_in_window()   ← 事件必须过滤，不过滤标签是错的（见 docstring）
    各处 docstring 里的字段表、量级、坑

切法你自己定（整回合 / 单 tick / 别的），但**事件过滤一定要做**。

写之前，先看这两份现成的文档（`tools/demo_extract/` 下，写得很全）：

    README.md  §3 输出长什么样    —— 整个 dict 的字段树，一页看完
               §4 怎么用          —— 速览，含 v6/v7 新增列的用法
               §5 火焰 / §6 烟雾  —— 几何怎么还原
               §8 已知取舍，请务必读

    API.md     §1.5–1.7 回合/玩家/事件的完整字段表
               §1.9 每个字段的量级（**实测**）  ← 写归一化、做数据体检必看
               §4.10 几个必须知道的坑
               §7 速查表          —— "我想……用这个" 的对照表

契约（如果你沿用下面的写法）
----------------------------
两个函数都返回 `dict[str, np.ndarray]`，**每个数组第 0 维必须是 T**
（本窗口的采样数）。key 的命名决定它转 torch 后的 dtype：

    key 以 `_mask` 结尾  → torch.bool
    key 在 INT_KEYS 里    → torch.long     （见 dataset.py）
    其它                  → torch.float32

窗口参数（`WindowRef`）也可以自己造：`WindowRef(file_idx, round_id, start, stop, start_tick)`
就是五个整数，`start`/`stop` 是回合内采样下标。想按窗口取数就自己拼一个传进来。

你拿到的东西
------------
    store    : CszStore          —— 取原始数据 / 事件 / fire / smoke
    ref      : WindowRef         —— 窗口位置：file_idx, round_id, start, stop, start_tick
    tensors  : dict[str, ndarray] —— 已经算好的整回合张量（见 store.round_tensors）

`tensors` 里每个数组都是**整个回合**的长度 T_round，所以你要自己切：

    a, b = ref.start, ref.stop
    tensors["players"][a:b]      # (T, 10, 20) 本窗口的玩家特征

可用的 key（`CszStore` 已校验格式为 v7，所以**这些全都有**）：

    players    (T,10,20) float32   列序见 constants.COL
    alive      (T,10)    bool
    yaw_sincos (T,10,2)  float32   sin/cos(yaw)
    seconds    (T,)      float64
    ticks      (T,)      int64
    teams      (10,)     int8      0=CT 1=T -1=未知
    inventory  (T,10,43) bool      全局武器词表
    spotted    (T,10,10) bool      [t,i,j] = "j 能看见 i"（注意是转置！）
    on_ground  (T,10)    bool
    jump/land  (T,10)    int64
    crouch     (T,10)    float32   0..1 连续（相机斜坡）
    crouching  (T,10)    bool      FL_DUCKING，碰撞盒已收到 54（阶跃）
    eye_z      (T,10)    float32   眼高（相对脚，HU）

几条必须做对的
--------------
* **yaw 不要直接喂角度**——它在 ±180 处跳变，模型没法插值。用 `yaw_sincos`。
* **位置要归一化**，否则 ±5000 的原始 HU 值会让训练很难收敛。见 `normalize_xyz()`。
  实测区间表在 `API.md` §1.9（位置 ±2300、水平速度 ≤282、`vz` ≤1152）。
* **稀疏的 0/1 列别做 min-max**——`shots`/`footsteps`/`jump`/`land` 的 p50 和 p99
  都是 0，缩放会把信号压没（`API.md` §1.9）。
* **`crouch` 和 `crouching` 是两个不同的量**，别用 `crouch > 0.5` 代替后者：
  前者是相机连续斜坡（用于视角），后者是碰撞盒阶跃 72→54（用于命中盒）。
* **`x,y,z` 是脚底不是眼睛**。眼位 = `z + eye_z`。
* 死人的位置列是 0，**配合 `alive` 掩码用**，别让模型以为他站在地图原点。
"""

from __future__ import annotations

import numpy as np

from .constants import (
    COL,
    MAP_CENTER,
    MAP_RANGE_X,
    MAP_RANGE_Y,
    MAP_RANGE_Z,
    N_PLAYERS,
)
from .store import CszStore, WindowRef

from demo_extract import recipes


# ═══════════════════════════════════════════════════════════════════════════
# 小工具（feature 专用，随便改）
# ═══════════════════════════════════════════════════════════════════════════


def normalize_xyz(xyz: np.ndarray, map_name: str) -> np.ndarray:
    """(..., 3) 的 Hammer 坐标 → 以地图中心为原点、裁到 ±MAP_RANGE 的归一化坐标。

    未知地图会退化成"不减去中心"（只用全局范围裁剪），不会报错。
    """
    cx, cy, cz = MAP_CENTER.get(map_name, (0.0, 0.0, 0.0))
    out = np.array(xyz, dtype=np.float32, copy=True)
    out[..., 0] = np.clip(out[..., 0] - cx, -MAP_RANGE_X, MAP_RANGE_X) / MAP_RANGE_X
    out[..., 1] = np.clip(out[..., 1] - cy, -MAP_RANGE_Y, MAP_RANGE_Y) / MAP_RANGE_Y
    out[..., 2] = np.clip(out[..., 2] - cz, -MAP_RANGE_Z, MAP_RANGE_Z) / MAP_RANGE_Z
    return out


def col1(tensors: dict, key: str, a: int, b: int) -> np.ndarray:
    """取一个 (T,10) 列，切成窗口并加尾轴成 (T,10,1) float32。

    这些 key 由 `CszStore.round_tensors()` 保证存在（格式在加载时已校验为 v7），
    所以这里不做存在性判断——缺列应该在加载时就报错，而不是在这里补 0。
    """
    return tensors[key][a:b, :, None].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 事件 → 窗口内下标（做标签用；**不滤会错，见 docstring**）
# ═══════════════════════════════════════════════════════════════════════════


def round_events(
    rnd: dict, kind: str = "kills", *, only_sampled: bool = True
) -> list[dict]:
    """某个事件列表，滤到回合自己的采样范围内，并附上采样下标 `i`。

    ⚠ **必须滤，否则标签是错的。** 实测 4 个文件：`events` 是按"上一个回合结束
    到这个回合结束"分桶的，所以**每个回合的事件列表里都有落在自己采样范围外的
    条目**：

    | 文件 | 击杀落在采样范围外 | 伤害落在采样范围外 |
    |---|---|---|
    | de_mirage | 43 / 162 (26.5%) | 114 / 630 (18.1%) |
    | de_nuke | 62 / 186 (33.3%) | 148 / 658 (22.5%) |
    | de_inferno | 25 / 163 (15.3%) | 60 / 710 (8.5%) |
    | de_dust2 | 70 / 222 (31.5%) | 137 / 769 (17.8%) |

    重灾区是 **round 0**：它把**赛前热身**的击杀全吃进来了
    （de_nuke 的 round 0 有 52 条击杀，其中 **45 条发生在回合开始之前**；
    de_dust2 是 56 条里 49 条）。不滤的话，round 0 会拿到一个荒谬的标签，
    而且这些 tick 在采样时间轴上根本不存在。

    另外每个回合末尾还常有一条 `tick = last_tick + 1` 的事件（回合结束瞬间），
    它也不在采样范围内。

    返回 list[dict]：原事件的所有字段 + `"i"`（该 tick 的采样下标）。
    落在范围外的**直接丢掉**（不会给你 `i = -1` 的条目）。

    `sound` 事件是例外——实测它 100% 落在采样范围内，滤不滤都一样。

    Args:
        kind: `kills` / `damage` / `bomb` / `grenades` / `sound` / `smokes` / `infernos`
        only_sampled: False 则不过滤，但 `i` 可能是 -1（不建议）。
    """
    ticks = rnd["ticks"]
    lo, hi = ticks[0], ticks[-1]
    out: list[dict] = []
    for e in rnd.get("events", {}).get(kind, []):
        t = e.get("t")
        if t is None:
            continue
        if only_sampled and not (lo <= t <= hi):
            continue
        i = recipes.sample_index(rnd, t)
        if i < 0:
            continue
        out.append({**e, "i": i})
    return out


def events_in_window(rnd: dict, ref: WindowRef, kind: str = "kills") -> list[dict]:
    """`round_events()` 的结果再滤到窗口内（`ref.start <= i < ref.stop`）。

    做"这个窗口里发生了什么"的标签时用这个，`e["i"] - ref.start` 就是窗口内下标。
    """
    return [e for e in round_events(rnd, kind) if ref.start <= e["i"] < ref.stop]


# ═══════════════════════════════════════════════════════════════════════════
# ★ 输入特征
# ═══════════════════════════════════════════════════════════════════════════


def build_window(
    store: CszStore, ref: WindowRef, tensors: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """构造输入。返回的每个数组第 0 维必须是 T。

    下面是**能跑通的参考实现**，不是最优特征集。它演示了必须做对的几件事：
    角度用 sin/cos、位置归一化、存活掩码、队伍。

    建议保留的 key（下游/调试都按这个约定）：
        players     (T, 10, F)
        alive_mask  (T, 10)
        team        (T, 10)
    """
    a, b = ref.start, ref.stop
    T = b - a
    raw = tensors["players"][a:b]        # (T, 10, 20)
    alive = tensors["alive"][a:b]        # (T, 10)
    yaw_sc = tensors["yaw_sincos"][a:b]  # (T, 10, 2)

    # 位置：归一化
    xyz = normalize_xyz(raw[:, :, COL["x"]:COL["z"] + 1], store.map_name(ref.file_idx))

    # pitch 也做 sin/cos（和 yaw 同理）
    pitch = np.radians(raw[:, :, COL["pitch"]]).astype(np.float32)
    pitch_sc = np.stack([np.cos(pitch), np.sin(pitch)], axis=-1)

    # 拼成 (T, 10, 16)
    feats = np.concatenate(
        [
            xyz,                                             # 3
            yaw_sc,                                          # 2
            pitch_sc,                                        # 2
            (raw[:, :, COL["hp"]:COL["hp"] + 1] / 100.0).astype(np.float32),      # 1
            (raw[:, :, COL["armor"]:COL["armor"] + 1] / 100.0).astype(np.float32),  # 1
            alive[:, :, None].astype(np.float32),            # 1
            raw[:, :, COL["flash"]:COL["flash"] + 1].astype(np.float32),          # 1
            (raw[:, :, COL["vx"]:COL["vz"] + 1] / 1000.0).astype(np.float32),     # 3
            col1(tensors, "on_ground", a, b),                # 1
            col1(tensors, "crouch", a, b),                   # 1
        ],
        axis=-1,
    )

    team = np.broadcast_to(tensors["teams"][None, :], (T, N_PLAYERS)).copy()

    return {
        "players": feats,                                    # (T, 10, 16) float32
        "alive_mask": alive,                                 # (T, 10)    bool
        "team": team,                                        # (T, 10)    int64
        # 用不上就删：这一刻谁能看见谁，[t,i,j] = j 看见 i
        "spotted": tensors["spotted"][a:b],                  # (T, 10, 10) bool
    }


# ═══════════════════════════════════════════════════════════════════════════
# ★ 标签
# ═══════════════════════════════════════════════════════════════════════════


def build_targets(
    store: CszStore, ref: WindowRef, tensors: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """构造标签。默认返回空 dict —— 这里放你的监督信号。

    现成可用的素材（注释掉的几行是能直接跑的示例）：

        # 自监督 / 世界模型：预测下一步状态
        # "next_players": tensors["players"][a+1:b+1]

        # 下一 tick 是否还活着
        # "next_alive_mask": tensors["alive"][a+1:b+1]

    需要事件时用 `store.round(ref.file_idx, ref.round_id)` 取原始回合 dict：

        rnd = store.round(ref.file_idx, ref.round_id)
        rnd["winner"]                    # 'T' / 'CT' / ''（回合胜负）
        rnd["bomb_planted"]              # (T_round,) bool
        rnd["bomb_position"]             # (T_round,) [x,y,z] or None
        rnd["events"]["kills"]           # [{t, a, v, w, hs, dmg, ...}]  t = tick
        rnd["events"]["damage"]          # [{t, a, v, hp, w}]
        rnd["events"]["grenades"]        # [{t, ty, eid, th, x, y, z}]

    ⚠ **别直接遍历 `rnd["events"]["kills"]`**——里面混着不属于这个回合的事件
    （round 0 会吃进赛前热身的几十条击杀），直接用会让标签错得很隐蔽。
    用上面那个 `events_in_window()`，它已经滤掉并给了窗口内下标：

        rnd = store.round(ref.file_idx, ref.round_id)
        for k in events_in_window(rnd, ref, "kills"):
            local_i = k["i"] - ref.start     # 窗口内第几个采样
            # k["a"] = 攻击者玩家下标, k["v"] = 被杀者, k["w"] = 武器下标, k["hs"] = 爆头

    事件里的 `t` 是真实 tick。想自己对齐就用：

        i = recipes.sample_index(rnd, event_tick)   # -1 = 不在采样里，必须处理

    ⚠ **动作标签不在 csz 里。** 想让 BC 用真实按键，得重新抽一次 demo，
    在 `tools/demo_extract/extract.py` 的 `PLAYER_PROPS` 里加一行：

        "CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtonDownMaskPrev"

    已实测 13 个键位全在，且两个解析器一致。
    注意 `Zoom` 位整份 demo 从未置起，判断开镜要用 `Attack2`(0x800)。
    """
    return {}
