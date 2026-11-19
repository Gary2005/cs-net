"""把**已经读进内存的 demo dict** 变成两个模型要的输入张量。

这里刻意不复刻一份特征公式：`ctx_player` 的 20 列、遮挡矩阵、道具快照、全局列
全部由 `scripts/training_data/tasks/aim.py` 的 `_Round` 算，和造训练数据用的是
**同一段代码**。抄一份到这里，两边迟早会漂：特征是训练和推理之间唯一的契约。

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
* `aim_arrays(entries)`：4 秒窗口样本，`(B, Tmax, ...)`，喂 `AimModel`。

后者是**右侧补齐**的（和 `collate_aim` 同一个约定）：因果注意力看不到补齐帧，
`seconds` 从窗口第一帧的 0 开始，所以窗口越短越像训练时随机截断出来的样本。
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

#: aim 模型的窗口长度 = `LOOKBACK(4 s) × 64 Hz + 1`。和 `tasks/aim.py::LOOKBACK`、
#: `AimConfig.time_bins` 是同一个数，改一个就要改另外两个。
AIM_WINDOW = 257

#: 每个采样点的列数：`ctx_projectile` 的行数（`_Round.projectiles` 的上限）。
N_PROJ = 32


#: 遮挡分块的帧数：`Occluder` 逐帧缓存，所以分块算和整段算的数值完全一样，
#: 分块只是为了每 ~0.25 s 能报一次进度（网页的进度条靠它）。
OCC_CHUNK = 512


class AiFeaturesError(RuntimeError):
    """这份数据没法治成模型输入（地图不认识 / 不是逐 tick 抽样 ……）。

    消息是给人看的，路由层直接 400/503 返回。
    """


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
            "obstruction": np.ascontiguousarray(self.obstruction[idx], dtype=np.int8),
            "ctx_projectile": np.ascontiguousarray(self.projectiles[idx], dtype=np.float32),
            "ctx_projectile_id": np.ascontiguousarray(self.proj_id[idx], dtype=np.int8),
            "global_feat": np.ascontiguousarray(self.global_feat[idx], dtype=np.float32),
            "map_id": np.full(b, self.map_id, dtype=np.int8),
        }

    # ── aim：4 秒窗口 ──────────────────────────────────────────────────────

    def window_bounds(self, i: int) -> tuple[int, int]:
        """以采样 `i` 结尾的窗口 `[start, stop)`：最多 `AIM_WINDOW` 帧。"""
        stop = min(max(int(i), 0), self.frames - 1) + 1
        return max(0, stop - AIM_WINDOW), stop

    def aim_arrays(
        self, entries: Iterable[tuple[int, int, int]]
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """`(i, a, b)` 的列表 → `AimModel` 的 batch + 每个样本的"最后一列"下标。

        `a` / `b` 是**这一帧**的两个主角（模型只按这两个下标查询）。窗口内两人
        是否一直活着由调用方保证（见 `service._duel_entries`）：模型训练时窗口
        内双方恒活，喂进去一个中途死掉的窗口不在分布里。

        补齐按 `collate_aim` 的约定：右侧补，`ctx_projectile_id` / 武器补 -1，
        `seconds` 从 0 开始 —— 这样短窗口（回合刚开始）和训练时的随机截断同分布。
        """
        entries = list(entries)
        b = len(entries)
        if b == 0:
            raise ValueError("aim_arrays 需要至少一个窗口")
        bounds = [self.window_bounds(i) for i, _, _ in entries]
        lengths = np.array([stop - start for start, stop in bounds], dtype=np.int64)
        tmax = int(lengths.max())

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
        }, lengths


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
    progress: Optional[Callable[[float, str], None]] = None,
) -> RoundTensors:
    """一个回合的整份模型输入（特征 + 遮挡 + 道具 + 全局列）。

    `data` 是 `codec.load()` 的结果。耗时（RTX 3090 那台机器上实测，一个
    ~6800 采样 / 106 s 的回合）≈ **3~4 s**，其中 90% 是两两遮挡的射线；同一
    回合重复调用应当走 `service` 的内存缓存，不要重新调这个函数。

    `progress(分数, 说明)` 报的是**本函数内部**的进度（0..1），调用方自己映射到
    整体进度上：遮挡按 `OCC_CHUNK` 帧分块算，玩家特征按人报（各 1/10），所以
    一次 7 秒的构建能报出二十来次，进度条不会僵在那儿。
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

    say(0.0, "构建玩家特征 / 遮挡几何…")
    n_pl = max(1, min(N_PLAYERS, len(rnd.get("players") or [])))
    rr = _Round(
        _WeaponLut(data), 0, rnd, map_name,
        # 逐人报一次（一个人约 5 ms，主要是让进度条早点动起来）；真正花时间的是
        # `_Round.__init__` 里的 `Occluder`（逐帧几何 + 地图 BVH），所以这一小段
        # 只占 10%，剩下的留给遮挡。
        on_player=lambda p: say(0.02 + 0.10 * (p + 1) / n_pl, f"玩家特征 {p + 1}/{n_pl}…"),
        on_stage=lambda name: say(0.13, f"{name}…"),
    )
    n = rr.n_frames

    obstruction = np.empty((n, N_PLAYERS, N_PLAYERS), np.int8)
    for start in range(0, n, OCC_CHUNK):
        stop = min(n, start + OCC_CHUNK)
        obstruction[start:stop] = rr.occluder.window(start, stop)
        say(0.14 + 0.84 * stop / n, f"两两遮挡 {stop}/{n} 帧…")

    say(0.99, "整理道具 / 全局列…")
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
        team=np.asarray(rr.team, dtype=np.int8),
        obstruction=obstruction,
        projectiles=projectiles,
        proj_id=proj_id,
        global_feat=global_feat,
    )
