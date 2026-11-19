"""多个任务共用的只读小工具（`rnd` 上的原始 dict → 数组）。

这个文件不是任务：文件名以 `_` 开头，`task_files()` 会跳过它，所以它不会出现在
`--list` / `--todos` 里，也不会被 `--task _util` 找到。

**模块级只 import 标准库 + numpy。** 任务模块会被 DataLoader 的每个 worker
import（见 `base.py` 的"几条实际的坑"），所以这里不把抽取栈拖进来。
唯一的例外是 `kills()`：它在**函数内部**才 `from ..features import round_events`
（见那个函数的注释），这样只想要 `track()` / `window_before()` 的任务不受影响，
而且无论 `scripts/training_data/__init__.py` 以后是不是改成惰性导入都成立。

约定：**所有函数都是只读的**，不改传进来的 `rnd`，返回的数组都是新对象
（`track()` 的 `np.stack` 必然复制，可以安全地就地归一化）。
"""

from __future__ import annotations

import numpy as np

#: `track()` 默认取的列。就是 `constants.FEATURES`（v7 的 20 列）里做瞄准最常用的
#: 那几列，顺序即输出数组的列序。要别的列自己传 `keys=`。
TRACK_KEYS: tuple[str, ...] = (
    "x", "y", "z",          # 脚底坐标，HU
    "yaw", "pitch",         # 度（喂模型请用 sin/cos，见下方 yaw_sincos）
    "vx", "vy", "vz",       # HU/s
    "hp",                   # 0..100
)


# ═══════════════════════════════════════════════════════════════════════════
# 玩家列 → 数组
# ═══════════════════════════════════════════════════════════════════════════


def track(
    rnd: dict, p: int, keys: tuple[str, ...] = TRACK_KEYS
) -> np.ndarray:
    """某个玩家的若干列 → `(T, len(keys))` float32。

    `rnd["players"][p]` 是列式记录（每个 key 一个长度 T 的 list），这个函数只碰
    **那一个玩家**：比 `recipes.player_tensor()` 那种 `(T,10,F)` 省 10 倍，也比
    `round_tensors()`（14 个 key、约 9.2 MB/回合）省得多。

    ```python
    atk = track(rnd, k["a"])                    # (T, 9)
    tgt = track(rnd, k["v"], ("x", "y", "z"))   # (T, 3)
    ```

    * 原始列里的 `None` 会变成 `nan`（`np.asarray(..., dtype=np.float32)` 的语义，
      和 `recipes.player_tensor()` 一致）。
    * `p` 越界会抛 `IndexError`；`keys` 为空会抛 `ValueError`（`np.stack` 不接受空列表）。
    """
    rec = rnd["players"][p]
    return np.stack([np.asarray(rec[k], dtype=np.float32) for k in keys], axis=-1)


def alive_track(rnd: dict, p: int) -> np.ndarray:
    """某个玩家的存活掩码 `(T,)` bool。

    ⚠ 死人的位置列是 0，不是 NaN —— 拿到 `False` 的帧别当有效坐标用。
    """
    return np.asarray(rnd["players"][p]["alive"], dtype=bool)


def yaw_sincos(rnd: dict, p: int) -> np.ndarray:
    """某个玩家的 `(T, 2)` = `[sin(yaw), cos(yaw)]`。

    别直接喂 yaw 角度：它在 ±180 处跳变，模型没法插值。pitch 同理。
    """
    yaw = np.radians(track(rnd, p, ("yaw",))[:, 0].astype(np.float64))
    return np.stack([np.sin(yaw), np.cos(yaw)], axis=-1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 事件
# ═══════════════════════════════════════════════════════════════════════════


def kills(rnd: dict, kind: str = "kills") -> list[dict]:
    """本回合的事件（默认击杀），已滤掉不属于这个回合的，并附上采样下标 `i`。

    直接转发给 `features.round_events()` —— 判据只有一份，不在这里重复实现。

    ⚠ **必须滤。** `events` 是按「上一回合结束 → 本回合结束」分桶的，round 0 会把
    赛前热身的几十条击杀全吃进来，回合末尾还有一条 `tick = last_tick + 1`
    （实测 15–33% 的击杀落在自己回合的采样范围外，见 `features.round_events`）。

    返回 `list[dict]`：原事件的所有字段 + `"i"`。字段见 `tools/demo_extract/API.md`
    §1.7（击杀：`t/a/v/as/w/hs/ts/ab/ai/af/dmg`）。

    ⚠ **回合末尾那条 `tick = last_tick + 1` 的收尾击杀不在这里**（正是它导致"最后
    一个击杀没有对枪预测"）。要的话再并上 `late_kills(rnd, n_frames=...)`，别自己
    去遍历原始列表 —— 那会把热身 / 别的回合的击杀一起捞进来。

    ⚠ `round_events()` 要求事件 tick **精确**落在某个采样点上，否则整条丢掉。
    用 `--stride` 抽出来的 csz 大部分事件的 tick 都不在采样点上，此时会静默少样本
    （`prepare` 的进度行只报回合数，看不出来）。要"吸到之后第一个采样点"就得自己
    写，或改 `features.round_events`。

    Args:
        kind: `kills` / `damage` / `bomb` / `grenades` / `sound` / `smokes` / `infernos`
    """
    # 函数内 import：`features` 顶上有 `from .store import CszStore` 和
    # `from demo_extract import recipes`，模块级 import 会把抽取栈拖进每个 worker。
    # 放这里的效果是：只有真的取事件时才付这笔钱（而那时一定在 prepare 进程里）。
    from ..features import round_events

    return round_events(rnd, kind)


def late_kills(rnd: dict, *, n_frames: int) -> list[dict]:
    """回合结束瞬间的那一枪 → 采样下标 `n_frames`（= 最后一帧**之后**）。

    `kills()` / `features.round_events()` 只保留 `ticks[0] <= t <= ticks[-1]`，而
    "打死最后一个敌人、回合立刻结束"的那一枪落在 `ticks[-1] + 1`：引擎在死亡那一
    tick 才把 `total_rounds_played` 加一，所以那一帧的采样已经归到**下一个**回合了。
    实测 4 份 demo 的 78 个回合里有 49 个（63%）是这么结束的（`end_reason ∈
    {t_killed, ct_killed}`），不补的话这一枪一条样本都没有 —— 回放页的「AI 对枪」
    也会在最后 3 秒整段空着（`app/ai/service.py::_round_kills` 是同一件事的推理侧版本）。

    下标给 `n_frames` 而不是 `n_frames - 1`：它表达的是"发生在最后一帧之后"，
    具体怎么用由标签侧决定：

    * `spatial_prediction.frame_labels()` 把 `i >= n_frames` 当"最后一帧之后"：
      中枪那位因此不算活到回合结束，末尾几行的 `next_death` 也指得到他；
    * `aim` 那边 `last_alive(alive[..., i])` 会自己往前找到受害者最后一个活帧，
      所以窗口终点仍是"死前一帧"，和正常样本同一种形状。

    判据就一条 `t == ticks[-1] + 1`：引擎在死亡那一 tick 才把 `total_rounds_played`
    加一，所以"收尾那一枪"必然落在最后一个采样 tick 的**下一个** tick 上。实测 4 份
    demo 里所有落在采样范围之外的击杀，偏差**恰好都是 +1**（没有 +2 或更大）；真出现
    更大的偏差说明这份 demo 被截断了，这时没有对应的帧、也不知道真实时间，不做猜测。
    """
    hi = int(rnd["ticks"][-1]) + 1
    out: list[dict] = []
    for e in (rnd.get("events") or {}).get("kills") or ():
        t = e.get("t")
        if t is not None and int(t) == hi:
            out.append({**e, "i": int(n_frames)})
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 时间轴
# ═══════════════════════════════════════════════════════════════════════════


def window_before(seconds: np.ndarray, i1: int, span: float = 4.0) -> slice:
    """`[i1 之前 span 秒, i1]` 的采样切片，含 `i1` 那一帧。

    `seconds` 必须是单调递增的时间轴，通常是 `rnd["round_seconds"]`
    （**不要**拿 tick 除以 64 代替：每份 demo 有一个固定偏移，而且 `--interval`
    采样会让 tick 差和秒差不成比例）。

    用 `searchsorted(..., "left")` 取「第一个 ≥ 目标时刻」的采样，所以窗口
    **至少** `span` 秒；`i1` 太靠前时自动从 0 开始（此时短于 `span` 秒，要不要
    这条样本由调用方决定 —— 框架不会替你判断）。

    ```python
    win = window_before(rnd["round_seconds"], k["i"])
    atk = track(rnd, k["a"])[win]      # (T, 9)
    ```
    """
    if not (0 <= i1 < len(seconds)):
        raise IndexError(f"i1={i1} 超出时间轴长度 {len(seconds)}")
    i0 = int(np.searchsorted(seconds, seconds[i1] - span, side="left"))
    return slice(i0, i1 + 1)


def last_alive(alive: np.ndarray, upto: int) -> int:
    """`alive[:upto+1]` 里最后一个 `True` 的下标；一个都没有则 `-1`。

    用途：击杀事件的那一帧，受害者**已经死了**（位置列全是 0）。要"死前一刻"
    的姿态就得往前找最后一帧活着的采样：

    ```python
    i1 = k["i"]
    jv = last_alive(alive_track(rnd, k["v"]), i1)
    if jv < 0:
        continue                       # 这个窗口里一次都没活过，别用
    tgt = track(rnd, k["v"])[jv]       # (9,)
    ```

    ⚠ 返回 `-1` 是"整段都没活过"，调用方必须处理 —— 别拿它去索引数组
    （Python 的负索引会静默给你最后一帧）。
    """
    idx = np.flatnonzero(np.asarray(alive[: upto + 1], dtype=bool))
    return int(idx[-1]) if idx.size else -1


if __name__ == "__main__":  # python -m scripts.training_data.tasks._util
    _rnd = {
        "ticks": [100, 101, 102, 103, 104],
        "round_seconds": [0.0, 0.015625, 0.03125, 0.046875, 0.0625],
        "players": [
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0] * 5, "z": [0.0] * 5,
             "yaw": [0.0] * 5, "pitch": [0.0] * 5, "vx": [0.0] * 5, "vy": [0.0] * 5,
             "vz": [0.0] * 5, "hp": [100.0] * 5, "alive": [True] * 4 + [False]},
        ],
        "events": {"kills": [{"t": 104, "a": 0, "v": 0, "w": 1}, {"t": 999, "a": 0, "v": 0}]},
    }
    assert track(_rnd, 0).shape == (5, len(TRACK_KEYS))
    assert alive_track(_rnd, 0).tolist() == [True, True, True, True, False]
    assert yaw_sincos(_rnd, 0).shape == (5, 2)
    assert [e["i"] for e in kills(_rnd)] == [4]          # t=999 在回合外，被滤掉
    # 回合结束那一枪（`ticks[-1] + 1`）不在 `kills()` 里，由 `late_kills()` 补上，
    # 下标给"帧数"（= 最后一帧之后）
    _rnd["events"]["kills"].append({"t": 105, "a": 0, "v": 1})
    assert [e["i"] for e in kills(_rnd)] == [4]
    assert [(e["a"], e["v"], e["i"]) for e in late_kills(_rnd, n_frames=5)] == [(0, 1, 5)]
    assert late_kills({"ticks": [1, 2], "events": {"kills": []}}, n_frames=2) == []
    assert window_before(_rnd["round_seconds"], 4) == slice(0, 5)
    assert last_alive(alive_track(_rnd, 0), 4) == 3
    assert last_alive(np.zeros(5, bool), 4) == -1
    print("[OK] tasks/_util.py 自检通过")
