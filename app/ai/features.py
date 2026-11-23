"""把**已经读进内存的 demo dict** 变成两个模型要的输入张量。

这里刻意不复刻一份特征公式：`ctx_player` 的 20 列、道具快照、全局列全部由
`scripts/training_data/tasks/aim.py` 的 `_Round` 算，和造训练数据用的是
**同一段代码**。抄一份到这里，两边迟早会漂：特征是训练和推理之间唯一的契约。

**遮挡不在这个契约的现算路径上**：逐帧两两遮挡是训练和推理共用的、只依赖 csz
几何 + 地图 mesh 的量，提取消耗完就落在 `<csz>.obs/r<回合>.npz`（见
`training_data/obstruction_cache.py`），这里只读数组。命中缓存时**完全不碰地图
几何**：不建 `Occluder`、不加载 BVH。缓存缺失才整回合重算再写回。

输入从哪来
----------
`visualizer/server.py` 的会话里就有 `codec.load()` 出来的整个 demo dict
（`_session["data"]`，和 `.csz` 里的内容一模一样）。所以不需要为了预测再
load 一次 10 秒的文件，也不需要造 npz。

`_Round` 只向 store 要一样东西（`weapon_lut`：本文件武器下标 → 全局下标），
所以这里用一个 12 行的 `_WeaponLut` 冒充 store，不构造 `CszStore` —— 后者会
把整个 `.csz` 再解一遍。

两个消费方
----------
* `spatial_arrays(idx)`：单帧样本，`(B, ...)`，直接喂 `SpatialModel`；
* `aim_arrays(windows)`：**一条对枪一个格子**，`(B, AIM_WINDOW, ...)`，喂
  `AimModel`，一次前向吃整条 3 秒窗口，列号由调用方按 `帧号 - start` 取。

后者是**固定 193 帧、右侧补齐**的（和 `collate_aim` 同一个约定）：因果注意力看不
到补齐帧，`seconds` 从窗口第一帧的 0 开始，所以窗口越短越像训练时随机截断出来
的样本。为什么不是训练的 257 帧，见下面 `AIM_WINDOW` 的说明。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.training_data.constants import (  # noqa: E402
    MAP_TO_IDX,
    N_PLAYERS,
    weapon_name_to_idx,
)
from scripts.training_data.store import _invert_weapon_index  # noqa: E402
from scripts.training_data.tasks.aim import _Round  # noqa: E402

#: 推理时每条 aim 窗口的帧数 = `3 s × 64 Hz + 1`。
#:
#: **故意比训练的 257 短**，不是因为省时间，是因为 `AimConfig.time_bins = 257` 是
#: 帧号 `k` 的查表上限，而训练时每条窗口的终点**恒是击杀帧**：
#:
#: * `k = 256` 在训练集里**只可能**出现在"最后一帧"，于是这一格等于把"下一帧就要
#:   死"直接写进位置编码 —— 模型学到 `k=256 ⇒ 击杀即将发生`；
#: * 一般地说，"第 k 列正好是击杀帧"的条件概率是 `1/(257-k)`：`k=256` 是 **1**，
#:   `k=192` 只有 1/65。
#:
#: 所以推理窗口取 193 帧（`[死前最后一帧 - 192, 死前最后一帧]`），最后一列是
#: `k=192`，把"位置编码本身就是答案"的那一格彻底躲开。屏幕上的弧线画最后 2 秒
#: （列 `64..192`），第一个点已经看了 64 帧 = 1 秒的历史；对枪评分探针取它后一帧
#: （满窗口第 65 列，死前 2 秒）。
#:
#: 三个数一起改才成立：这里、`AimConfig.time_bins`（必须 ≥ 193），以及
#: `scripts/training_data/tasks/aim.py::LOOKBACK`（训练窗口，仍是 4 s 257 帧）。
AIM_WINDOW = 193

#: 弧线/探针在窗口里的起点偏移：`2 s × 64 Hz`。也就是"死前 2 秒"这一列离窗口
#: 末尾的距离；`service.DUEL_LEAD` 和它必须一致。
ARC_FROM_END = 2 * 64

#: 每个采样点的列数：`ctx_projectile` 的行数（`_Round.projectiles` 的上限）。
N_PROJ = 32


class AiFeaturesError(RuntimeError):
    """这份数据没法治成模型输入（地图不认识 / 不是逐 tick 抽样 ……）。

    消息是给人看的，路由层直接 400/503 返回；`kind` 给任务队列和 NDJSON 流
    分类（和 `ModelUnavailable.kind == "models"` 对应）。
    """

    kind = "features"


@dataclass(frozen=True)
class RoundTensors:
    """一个回合的整份输入（同一个回合的多个窗口/多帧共用，只算一次）。"""

    map_name: str
    map_id: int
    ticks: np.ndarray          # (T,) int64
    seconds: np.ndarray        # (T,) float64，回合内秒数
    feat: np.ndarray           # (T, 10, 20) float32，AIM_FEATURES，全在 [0,1]
    alive: np.ndarray          # (T, 10) bool
    weapon: np.ndarray         # (T, 10) int8，全局武器下标，-1 = 无/死
    inv: np.ndarray            # (T, 10, 43) uint8，背包多热（0/1，下标 = WEAPON_NAMES）
    team: np.ndarray           # (10,)   int8，0=CT 1=T -1=未知
    obstruction: np.ndarray    # (T, 10, 10) int8，0 通 / 1 烟 / 2 墙
    projectiles: np.ndarray    # (T, 32, 4) float32，xyz 全 [0,1]，第 4 列 = 已存在秒数/100
    proj_id: np.ndarray        # (T, 32) int8，-1 = 空槽
    global_feat: np.ndarray    # (T, 5) float32，炸弹 xyz + 安放秒/50 + 回合秒/200

    @property
    def frames(self) -> int:
        return int(self.ticks.shape[0])

    # ── spatial：单帧 ──────────────────────────────────────────────────────

    def spatial_arrays(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """`idx`（采样下标，可重复）→ `SpatialModel` 的 batch（numpy）。"""
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        b = int(idx.size)
        return {
            "ctx_player": np.ascontiguousarray(self.feat[idx], dtype=np.float32),
            # collate_spatial 给的是 bool；`SpatialModel.forward` 里直接 `.bool()`
            "ctx_player_mask": np.ascontiguousarray(self.alive[idx], dtype=bool),
            "ctx_player_team": np.broadcast_to(
                self.team[None, :], (b, N_PLAYERS)
            ).astype(np.int8),
            "ctx_player_weapon": np.ascontiguousarray(self.weapon[idx], dtype=np.int8),
            "ctx_player_inv": np.ascontiguousarray(self.inv[idx], dtype=np.uint8),
            "obstruction": np.ascontiguousarray(self.obstruction[idx], dtype=np.int8),
            "ctx_projectile": np.ascontiguousarray(self.projectiles[idx], dtype=np.float32),
            "ctx_projectile_id": np.ascontiguousarray(self.proj_id[idx], dtype=np.int8),
            "global_feat": np.ascontiguousarray(self.global_feat[idx], dtype=np.float32),
            "map_id": np.full(b, self.map_id, dtype=np.int8),
        }

    # ── aim：3 秒窗口（锚在击杀帧）─────────────────────────────────────────

    def window_bounds(self, i: int) -> tuple[int, int]:
        """以帧 `i` 结尾的窗口 `[start, stop)`：最多 `AIM_WINDOW` 帧。"""
        stop = min(max(int(i), 0), self.frames - 1) + 1
        return max(0, stop - AIM_WINDOW), stop

    def aim_arrays(
        self, entries: Iterable[tuple[int, int, int]]
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """`(窗口终点, A, B)` 的列表 → `AimModel` 的 batch + 每条的起点 / 长度。

        **一次前向吃整条窗口**：`(i1, a, b)` 只给窗口终点（受害者最后一个活帧），
        窗口是 `[i1 - (AIM_WINDOW-1), i1]`，和训练样本一样锚在击杀那一刻。要第 `i`
        帧的预测就读第 `i - start` 列（`start` 由本函数返回）—— 不再为每一帧另开
        一条"以该帧结尾"的窗口（那样既贵 10 倍，又把 `k` 挪到训练里只属于击杀帧
        的那几格，见 `AIM_WINDOW` 的注释）。

        补齐按 `collate_aim` 的约定：**右侧**补，`ctx_projectile_id` / 武器补 -1，
        `seconds` 从 0 开始 —— 回合最开头那几刀（`i1 < AIM_WINDOW-1`）窗口被截短，
        补出来的尾巴 `frame_mask=False`、因果注意力看不到。

        Returns:
            `(batch, starts, lengths)`：`starts[r]` 是第 r 条窗口的第一帧；
            `lengths[r]` 是真实帧数（其余是补的零）。
        """
        entries = list(entries)
        b = len(entries)
        if b == 0:
            raise ValueError("aim_arrays 需要至少一个窗口")
        bounds = [self.window_bounds(i) for i, _, _ in entries]
        starts = np.array([start for start, _ in bounds], dtype=np.int64)
        lengths = np.array([stop - start for start, stop in bounds], dtype=np.int64)
        tmax = AIM_WINDOW

        ctx_player = np.zeros((b, tmax, N_PLAYERS, 20), dtype=np.float32)
        mask = np.zeros((b, tmax, N_PLAYERS), dtype=np.int8)
        weapon = np.full((b, tmax, N_PLAYERS), -1, dtype=np.int8)
        obstruction = np.zeros((b, tmax, N_PLAYERS, N_PLAYERS), dtype=np.int8)
        projectiles = np.zeros((b, tmax, N_PROJ, 4), dtype=np.float32)
        proj_id = np.full((b, tmax, N_PROJ), -1, dtype=np.int8)
        global_feat = np.zeros((b, tmax, 5), dtype=np.float32)
        seconds = np.zeros((b, tmax), dtype=np.float32)
        frame_mask = np.zeros((b, tmax), dtype=bool)
        pair = np.zeros((b, 2), dtype=np.int64)

        for row, ((_, a, other), (start, stop)) in enumerate(zip(entries, bounds)):
            t = stop - start
            ctx_player[row, :t] = self.feat[start:stop]
            mask[row, :t] = self.alive[start:stop]
            weapon[row, :t] = self.weapon[start:stop]
            obstruction[row, :t] = self.obstruction[start:stop]
            projectiles[row, :t] = self.projectiles[start:stop]
            proj_id[row, :t] = self.proj_id[start:stop]
            global_feat[row, :t] = self.global_feat[start:stop]
            # 和训练时的随机截断同一个口径：窗口第一帧 = 0 秒。
            seconds[row, :t] = self.seconds[start:stop] - self.seconds[start]
            frame_mask[row, :t] = True
            pair[row] = (a, other)

        return {
            "ctx_player": ctx_player,
            "ctx_player_mask": mask,
            "ctx_player_weapon": weapon,
            "obstruction": obstruction,
            "ctx_projectile": projectiles,
            "ctx_projectile_id": proj_id,
            "global_feat": global_feat,
            "seconds": seconds,
            "frame_mask": frame_mask,
            "pair": pair,
            "map_id": np.full(b, self.map_id, dtype=np.int8),
        }, starts, lengths


class _WeaponLut:
    """冒充 `CszStore` 的最小接口：`_Round` 只调用 `weapon_lut(fi)`。

    `_invert_weapon_index` 是 `CszStore.weapon_lut` 里用的同一个函数，所以这里
    算出来的映射和造训练数据时一模一样（包括"未知名落到 Knife"那条兜底）。
    """

    def __init__(self, data: dict) -> None:
        names = _invert_weapon_index(data)
        self._lut = np.array(
            [weapon_name_to_idx(n) if n else -1 for n in names], dtype=np.int16
        )

    def weapon_lut(self, _file_idx: int = 0) -> np.ndarray:
        return self._lut


def build_round_tensors(
    data: dict,
    round_idx: int,
    *,
    csz_path: Optional[str | Path] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> RoundTensors:
    """一个回合的整份模型输入（特征 + 遮挡 + 道具 + 全局列）。

    `data` 是 `codec.load()` 的结果。遮挡走磁盘缓存（`<csz>.obs/r<回合>.npz`，
    由 `obstruction_cache.ensure()` 在提取 / 第一次用到时建好）：**命中就只是一次
    读盘，不建地图 BVH、不发射线**，所以这个函数从原来的 3~4 秒降到约 0.1 秒
    （实测：`_Round` 30 ms + 道具 61 ms + 读缓存 10 ms）。缓存缺失时会整回合重算
    （约 0.5 秒）再写回，而不是只算需要的窗口。

    同一回合重复调用应当走 `service` 的内存缓存，不要重新调这个函数。

    `progress(分数, 说明)` 报的是**本函数内部**的进度（0..1）。
    """
    say = progress or (lambda _frac, _msg: None)
    rounds = data.get("rounds") or []
    if not 0 <= round_idx < len(rounds):
        raise AiFeaturesError(f"回合越界：{round_idx}（共 {len(rounds)} 个）")
    rnd = rounds[round_idx]

    meta = data.get("meta") or {}
    map_name = str(meta.get("map") or "")
    map_id = MAP_TO_IDX.get(map_name)
    if map_id is None:
        raise AiFeaturesError(
            f"地图 {map_name!r} 不在训练用的 8 张图里（{', '.join(sorted(MAP_TO_IDX))}），"
            "坐标归一化会失真，不能喂模型"
        )

    ticks = np.asarray(rnd.get("ticks") or [], dtype=np.int64)
    if ticks.size < 2:
        raise AiFeaturesError("这个回合没有采样点")
    # 模型的时间位置 = `seconds × 64`。只有逐 tick 抽样的文件才满足"一步 = 1/64 秒"；
    # stride / interval 抽样会让同一个 4 秒窗口跨几十秒，喂进去和训练分布差太远。
    step = float(np.median(np.diff(ticks)))
    if abs(step - 1.0) > 1e-6:
        raise AiFeaturesError(
            f"这个 .csz 不是逐 tick 的（采样步长 {step:g} tick）："
            "AI 预测需要每 tick 都采的提取结果"
        )

    say(0.0, "读取逐帧遮挡缓存…")
    from ..obstruction import ensure_round

    frames = int(ticks.size)
    obstruction = ensure_round(
        csz_path,
        rnd,
        map_name,
        progress=lambda frac, msg: say(0.85 * max(0.0, min(1.0, frac)), msg),
    )
    if obstruction.shape != (frames, N_PLAYERS, N_PLAYERS):
        raise AiFeaturesError(
            f"遮挡缓存形状不对：{obstruction.shape}，回合是 {frames} 帧 × "
            f"{N_PLAYERS} 人（清掉 <csz>.obs 会重算）"
        )

    say(0.86, "构建玩家特征…")
    n_pl = max(1, min(N_PLAYERS, len(rnd.get("players") or [])))
    rr = _Round(
        _WeaponLut(data), 0, rnd, map_name,
        obstruction=obstruction,
        on_player=lambda p: say(0.86 + 0.08 * (p + 1) / n_pl, f"玩家特征 {p + 1}/{n_pl}…"),
        on_stage=lambda name: say(0.95, f"{name}…"),
    )
    n = rr.n_frames

    say(0.97, "整理道具 / 全局列…")
    projectiles, proj_id = rr.projectiles(slice(0, n))
    global_feat = rr.global_feat(slice(0, n))
    seconds = np.asarray(rnd["round_seconds"], dtype=np.float64)[:n]

    return RoundTensors(
        map_name=map_name,
        map_id=int(map_id),
        ticks=ticks[:n],
        seconds=seconds,
        feat=rr.feat[:, :n].transpose(1, 0, 2).copy(),   # (T,10,20)：窗口切片要连续内存
        alive=np.ascontiguousarray(rr.alive[:, :n].T),
        weapon=np.ascontiguousarray(rr.weapon[:, :n].T),
        # 背包：和造训练数据走**同一个** `_Round.inventory_at`，所以两边不可能错开。
        inv=rr.inventory_at(np.arange(n)),
        team=np.asarray(rr.team, dtype=np.int8),
        obstruction=obstruction,
        projectiles=projectiles,
        proj_id=proj_id,
        global_feat=global_feat,
    )
