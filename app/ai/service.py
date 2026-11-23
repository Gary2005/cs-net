"""把两个模型接进回放页：**一回合一次**算完，按回合缓存。

为什么是"整回合一次"
------------------
一次前向能吃掉整条序列（aim 是因果的，spatial 是单帧并行的），所以最贵的不是
模型而是**准备输入**：逐帧两两遮挡（~0.5 ms/帧）、4 秒窗口切片。这些量都只和
回合有关，所以按回合算一次、整场回放重复使用 —— 拖时间轴、切回合回来，都是
纯查表，一帧的代价是几次数组取值。

缓存分三层
---------
1. **进程内 · 输入张量**：挂在会话的 `data` dict 上（`data["_ai_tensors"]`），
   换一场比赛（`_clear_session`）自然就没了。一个回合约 10 MB，只留最近 3 个。
2. **进程内 · 模型**：`Predictor` 单例，两个模型常驻显存（约 0.7 GB）。
3. **磁盘 · 预测结果**：`app/data/ai/<demo>/<models>_s<stride>_r<回合>.json`。
   键里带 `.csz` 的 size+mtime 和权重的**内容**指纹（大小 + 头尾 1 MB），所以
   "换了 demo"或"换了模型"都会自动失效；而"把同样的权重重下一遍"不会 ——
   见 `models.file_fingerprint`。重复点同一回合 = 读一个 130 KB 的 JSON，毫秒级。

预测了什么（spatial 的 5 个头全给）
----------------------------------
spatial（一帧一个预测）：

    win_ct       这一帧的 CT 最终赢面
    alive        逐人"活到回合结束"的概率
    can_kill     逐人"之后还能拿到击杀"的概率
    next_death   下一个死的是谁（11 类：0..9 是玩家，10 = 之后没人死）
    next_kill    下一个杀人的是谁（同上）

aim（一个对枪窗口一个预测）：`P(A 击杀 B)`。**A = 受害者，B = 击杀者**，也就是
网页上那条弧线的两端；弧上的游标 = A 的胜率。

对枪：一枪一条窗口，锚在击杀那一刻（`DUEL_LEAD` / `AIM_WINDOW`）
------------------------------------------------------------------
训练数据里 aim 的样本是「[受害者最后一个活帧 - 4 s, 受害者最后一个活帧]」，label
在整条窗口的**每一帧**上都算一个样本。所以推理时也**一枪一条窗口**：窗口终点恒是
受害者最后一个活帧，屏幕上的 2 秒弧线是这条窗口的最后 128 列。两个人用的是**这一
回合真实的击杀双方**（来自 `.csz` 的 kills 事件）：aim 模型是"这两个人谁会赢"，
喂一对"模型猜的对枪"进去既不是训练分布，也会在没有对枪的时候乱跳。谁"即将被打死"
这件事由 spatial 的 next_death / next_kill 在指标表里回答。

窗口取 193 帧（3 秒）而不是训练的 257 帧，理由见 `features.AIM_WINDOW`：257 是
`time_bins` 的上限，训练时窗口终点恒是击杀帧，于是 `k=256` 只在"最后一帧"出现 ——
位置编码本身就把答案写了进去。193 帧把那一格躲开，最后一列是 `k=192`。

**以前不是这么算的**：老代码是"每个采样点开一条以该帧结尾的 4 秒窗口"，于是弧线
最早那个点（死前 3 秒）看到的其实是 `[死前 7 秒, 死前 3 秒]`，而且 `k=256` ——
既是训练里没出现过的状态，又把 71 个窗口算完只取最后一列（约 10 倍的浪费）。
改成一枪一条之后 aim 从 ~15 s 降到 ~1.8 s（实测 13900K / CPU torch）。

**回合结束那一枪也要有弧线**（`_round_kills`）
--------------------------------------------
"打死最后一个敌人、回合立刻结束"的那一枪落在 `ticks[-1] + 1`：引擎在死亡那一
tick 就把 `total_rounds_played` 加一，所以那一帧的采样已经归到下一个回合了。
`round_events()` 只收 `ticks[0] <= t <= ticks[-1]`，于是这**每个回合的最后一杀**
连一场对枪都不会有 —— 实测 4 份 demo 的 78 个回合里有 49 个（63%）是这么结束
的。`_round_kills()` 把它挂到最后一帧上（受害者那一帧还活着、下一个 tick 才死，
和训练样本的"最后一个活帧"完全同分布），其余口径不变。

**AI 数据统计也在这个文件里算完**（`stats` 段）
----------------------------------------------
`predict_round` 顺手把统计页「AI 数据」那五个数算出来（`app/ai/stats.py`）：
击杀 / 回合影响力、站位死亡风险、站位击杀率、对枪评分。它们全都建立在**同一次**
前向之上 —— spatial 的逐帧胜率 / next_death / next_kill 已经在那儿了，对枪评分
只是从对枪窗口里读一列（满窗口第 65 列 = 死前 2 秒），**不再多跑一次 aim**。所以这些数
和回放页的弧线是同一批预测，不会出现"统计里的胜率和画面上的对不上"；而且它们和
预测一起进同一个缓存文件，换页签、开关弹窗都不用再跑模型。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from . import models as model_lib
from . import stats as stats_mod
from .features import AIM_WINDOW, AiFeaturesError, RoundTensors, build_round_tensors
from .models import ModelUnavailable

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from scripts.training_data.features import round_events  # noqa: E402
from scripts.training_data.tasks._util import last_alive  # noqa: E402

#: 网页上的时间分辨率：每 16 个采样点（逐 tick 的 demo 就是 4 Hz）算一次。
#: 4 Hz 对"看一眼谁要打起来"足够了，而代价是下面这些数的线性函数：
#: 6800 采样的回合 ≈ 427 个预测点，spatial 约 2.5 s（aim 与采样点无关，一枪一次）。
DEFAULT_STRIDE = 16

#: 结果 JSON 的格式版本。改了 payload 结构就 +1，旧缓存自动作废。
#:   v2：spatial 多了 `can_kill`、对枪改成"死前 3 秒的真实击杀双方"、去掉 `conf`
#:   v3：回合结束那一枪（`t = ticks[-1] + 1`）也算一场对枪，见 `_round_kills`
#:   v4：多了 `stats`（AI 数据统计：击杀/回合影响力、站位风险、对枪评分），
#:       见 `app/ai/stats.py`
#:   v5：回合影响力改成「逐采样点胜率涨落按在场人数均分」，`stats` 里的
#:       `death_sum` / `death_n` 没了
#:   v6：回合影响力改记到「涨落之前」那一帧的存活名单上，某队全灭后沿用最后一个
#:       还有人活着的帧
#:   v7：aim 改成一枪一条窗口（锚在击杀帧、193 帧）、弧线画死前 2 秒、对枪评分取
#:       同一条窗口的第 65 列 —— 预测值全变了，旧缓存必须作废
#:   v8：烟雾遮挡缓存修好了（烟散了不再按有烟算），`obstruction` 这个模型输入变了，
#:       同一份 csz 的预测值跟着变 —— 旧缓存必须作废。顺带把 `app/ai/stats.py` 的
#:       比赛级缓存一起作废（那边用同一个 `round_version`）。
CACHE_VERSION = 8

#: 弧线在屏幕上画多长（秒）：从**死前 2 秒**画到击杀那一帧。
#:
#: 为什么是 2 而不是 3：窗口是 193 帧 = 3 秒（`features.AIM_WINDOW`），取最后 2 秒
#: 就留出 1 秒历史 —— 弧线第一个点已经看了 64 帧数据，而不是从"刚有一帧"开始猜。
#: 这 1 秒同时也是 `stats.KILL_PROBE_SEC`：对枪评分要的就是"模型只看了 1 秒时怎么
#: 判断"，探针落在弧线起点后一帧（满窗口第 65 列，`features.ARC_FROM_END = 128`）。
DUEL_LEAD = 2.0

#: 进程内最多留几个回合的输入张量（一个约 10 MB）。
_TENSORS_KEEP = 3

#: 整体进度里各阶段的区间（实测：特征 4.2 s、遮挡 3.4 s、spatial 0.6 s、
#: aim 0.3 s、加载权重 1.4 s、首次 numba 编译 +7 s）。进度按阶段权重线性映射，
#: 前端再做平滑 —— 见 `visualizer/static/js/main.js::ensureAiRound`。
_STAGE_LOAD = (0.01, 0.06)
_STAGE_FEAT = (0.06, 0.62)
_STAGE_SPATIAL = (0.62, 0.86)
_STAGE_AIM = (0.86, 0.97)


def _log(msg: str) -> None:
    print(f"[ai] {msg}", flush=True)


def _sub(span: tuple[float, float], frac: float) -> float:
    """`span` 区间里的 `frac`（0..1）位置。"""
    lo, hi = span
    return lo + (hi - lo) * min(1.0, max(0.0, frac))


# ═══════════════════════════════════════════════════════════════════════════
# 进度（给 `/api/ai/round?progress=1` 的 NDJSON 流用）
# ═══════════════════════════════════════════════════════════════════════════


class Progress:
    """一次预测的进度快照。

    写的人在后台线程（真正跑模型的那个），读的人在 Flask 的流式生成器里 ——
    所以只有一个锁 + 几个字段，不做队列。`pct` 只增不减（阶段权重是单调的，
    偶尔乱序也不会让进度条倒退）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pct = 0.0
        self._stage = "排队中…"
        self._t0 = time.perf_counter()
        self._payload: Optional[dict[str, Any]] = None
        self._error: Optional[str] = None
        self._kind: Optional[str] = None
        self._done = False

    # ── 写 ────────────────────────────────────────────────────────────────

    def report(self, pct: float, stage: str) -> None:
        with self._lock:
            self._pct = max(self._pct, min(0.999, float(pct)))
            self._stage = stage

    def finish(self, payload: Optional[dict[str, Any]] = None,
               error: Optional[str] = None, kind: Optional[str] = None) -> None:
        with self._lock:
            self._pct = 1.0 if payload is not None else self._pct
            self._stage = "完成" if payload is not None else "失败"
            self._payload = payload
            self._error = error
            self._kind = kind
            self._done = True

    # ── 读 ────────────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    def snapshot(self) -> dict[str, Any]:
        """一次事件（NDJSON 的一行）。"""
        with self._lock:
            elapsed = round(time.perf_counter() - self._t0, 2)
            if self._payload is not None:
                return {"type": "result", "payload": self._payload, "elapsed": elapsed}
            if self._error is not None:
                return {"type": "error", "error": self._error,
                        "kind": self._kind, "elapsed": elapsed}
            return {"type": "progress", "pct": round(self._pct, 4),
                    "stage": self._stage, "elapsed": elapsed}

# ═══════════════════════════════════════════════════════════════════════════
# 模型：一次加载，常驻
# ═══════════════════════════════════════════════════════════════════════════


class Predictor:
    """`aim` + `spatial` 两个模型和它们的那把锁。"""

    _instance: Optional["Predictor"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundle: Optional[model_lib.Bundle] = None
        self._error: Optional[str] = None
        self._aim_ref: Optional[model_lib.RunRef] = None
        self._spatial_ref: Optional[model_lib.RunRef] = None
        self._model_key: Optional[str] = None

    @classmethod
    def instance(cls) -> "Predictor":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 发现 / 加载 ────────────────────────────────────────────────────────

    def refs(self) -> tuple[model_lib.RunRef, model_lib.RunRef]:
        """写死的那两份 aim / spatial 权重。找不到就抛 `ModelUnavailable`。"""
        if self._aim_ref is None or self._spatial_ref is None:
            self._aim_ref = model_lib.pinned_run("aim")
            self._spatial_ref = model_lib.pinned_run("spatial_prediction")
            self._model_key = model_lib.signature_key(self._aim_ref, self._spatial_ref)
            _migrate_legacy_model_prefix(
                self._aim_ref, self._spatial_ref, self._model_key
            )
        return self._aim_ref, self._spatial_ref

    @property
    def model_key(self) -> str:
        self.refs()
        assert self._model_key is not None
        return self._model_key

    def bundle(self, progress: Optional[Callable[[float, str], None]] = None
               ) -> model_lib.Bundle:
        """加载（幂等）。慢：两个 checkpoint 加起来 ~1.5 GB。"""
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            aim_ref, spatial_ref = self.refs()
            _log(f"加载模型 aim={aim_ref.name} spatial={spatial_ref.name} …")
            bundle = model_lib.Bundle(aim_ref, spatial_ref)
            bundle.load(progress=progress)
            self._bundle = bundle
            info = bundle.info()
            _log(
                f"模型就绪 {info.get('device')} · {info.get('gpu') or 'cpu'} · "
                f"{bundle.loaded_ms:.0f} ms · aim {bundle.meta['aim']['params'] / 1e6:.1f}M "
                f"+ spatial {bundle.meta['spatial']['params'] / 1e6:.1f}M 参数"
            )
            return bundle

    def status(self) -> dict[str, Any]:
        """给页面看的可用性（**不加载权重**，只查 `PINNED_WEIGHTS` 那两个文件在不在）。"""
        try:
            aim_ref, spatial_ref = self.refs()
        except ModelUnavailable as exc:
            return {"ok": False, "reason": str(exc), "stride": DEFAULT_STRIDE}
        out: dict[str, Any] = {
            "ok": True,
            "reason": None,
            "stride": DEFAULT_STRIDE,
            "aim_window": AIM_WINDOW,
            "duel_lead": DUEL_LEAD,
            "loaded": self._bundle is not None,
            "runs": {
                "aim": {**aim_ref.signature(), "path": str(aim_ref.weights)},
                "spatial": {
                    **spatial_ref.signature(),
                    "path": str(spatial_ref.weights),
                },
            },
        }
        if self._bundle is not None:
            out.update(self._bundle.info())
        return out

    # ── 推理 ───────────────────────────────────────────────────────────────

    def _autocast(self):
        bundle = self.bundle()
        if bundle.device == "cuda":
            return bundle.torch.autocast("cuda", dtype=bundle.torch.bfloat16)
        return _nullcontext()

    def run_spatial(
        self,
        tensors: RoundTensors,
        idx: np.ndarray,
        *,
        chunk: int = 256,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, np.ndarray]:
        """逐帧跑 spatial，返回 5 个头（都是概率）。

        `win` / `alive` / `can_kill` / `next_death` / `next_kill`。

        死人的列在这一步就被清成 0（`next_death` / `next_kill` 再按行归一化）：
        模型的 forward 不做这件事，训练时是在 loss 里把死人的列填 `-inf` 掩掉
        的。不掩的话网页上会出现"某个已经死掉的人有 3% 概率成为下一个死的人"。
        """
        bundle = self.bundle()
        torch = bundle.torch
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        win = np.zeros(idx.size, dtype=np.float32)
        alive = np.zeros((idx.size, 10), dtype=np.float32)
        cankill = np.zeros((idx.size, 10), dtype=np.float32)
        nd = np.zeros((idx.size, 11), dtype=np.float32)
        nk = np.zeros((idx.size, 11), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, idx.size, chunk):
                part = idx[start:start + chunk]
                arrays = tensors.spatial_arrays(part)
                batch = {
                    k: torch.from_numpy(np.ascontiguousarray(v)).to(bundle.device)
                    for k, v in arrays.items()
                }
                with self._autocast():
                    out = bundle.spatial(batch)
                b = part.size
                live = tensors.alive[part]                      # (b,10) bool
                win[start:start + b] = out["logit_win"].float().sigmoid().cpu().numpy()
                alive[start:start + b] = (
                    out["logit_alive"].float().sigmoid().cpu().numpy() * live
                )
                cankill[start:start + b] = (
                    out["logit_can_kill"].float().sigmoid().cpu().numpy() * live
                )
                for arr, key in ((nd, "logit_next_death"), (nk, "logit_next_kill")):
                    p = out[key].float().softmax(-1).cpu().numpy()
                    p[:, :10] *= live
                    total = p.sum(axis=1, keepdims=True)
                    arr[start:start + b] = p / np.where(total > 0, total, 1.0)
                if progress is not None:
                    progress(min(idx.size, start + b), int(idx.size))
        return {"win": win, "alive": alive, "can_kill": cankill,
                "next_death": nd, "next_kill": nk}

    def run_aim(
        self,
        tensors: RoundTensors,
        windows: list[tuple[int, int, int]],
        *,
        chunk: int = 32,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """`[(窗口终点, A, B), …]` → 概率矩阵 `(行, AIM_WINDOW)`，行 = 窗口。

        **返回整条窗口**（不是最后一列）：一次前向里所有列都算出来了，调用方按
        `列号 = 帧号 - starts[row]` 去取（`starts` 见 `RoundTensors.aim_arrays`）。
        以前每个采样点开一条窗口、只留最后一列，等于把 256/257 的结果扔掉。
        """
        if not windows:
            return np.zeros((0, AIM_WINDOW), dtype=np.float32)
        bundle = self.bundle()
        torch = bundle.torch
        out = np.zeros((len(windows), AIM_WINDOW), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(windows), chunk):
                part = windows[start:start + chunk]
                arrays, _starts, _lengths = tensors.aim_arrays(part)
                batch = {
                    k: torch.from_numpy(np.ascontiguousarray(v)).to(bundle.device)
                    for k, v in arrays.items()
                }
                with self._autocast():
                    result = bundle.aim(batch)
                probs = result["probs"].float().cpu().numpy()
                out[start:start + len(part)] = probs
                if progress is not None:
                    progress(min(len(windows), start + len(part)), len(windows))
        return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 对枪窗口
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Duel:
    """一场对枪：谁死、谁杀，以及屏幕上的弧线区间 `[i0, i1]`。

    `a` 是受害者（弧线的 A 端，也就是游标的 0 点），`b` 是击杀者。
    `[i0, i1]` 是**画弧线**的那 2 秒（`DUEL_LEAD`），不是模型窗口：模型窗口是
    `[i1 - (AIM_WINDOW-1), i1]`（3 秒），见 :class:`DuelRun`。
    """

    a: int      # 受害者的槽位 0..9
    b: int      # 击杀者的槽位 0..9
    i0: int     # 弧线第一个采样帧（死前 `DUEL_LEAD` 秒）
    i1: int     # 弧线最后一帧 = 受害者最后一个活帧
    t0: float   # `i0` 的回合内秒数
    t1: float   # `i1` 的回合内秒数

    def as_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "i0": self.i0, "i1": self.i1,
                "t0": round(self.t0, 3), "t1": round(self.t1, 3)}


@dataclass(frozen=True)
class DuelRun:
    """一场对枪占 `run_aim` 结果里的哪一行，以及这条窗口盖住哪几帧。

    `start` 是窗口第一帧（`[start, i1]`，正常是 193 帧，回合开头那几刀会被帧 0
    截短），于是"第 `i` 帧的预测"= 第 `i - start` 列。
    """

    duel: Duel
    row: int
    start: int
    length: int

    def column(self, i: int) -> int:
        """第 `i` 帧在这条窗口里的列号（调用方保证 `start <= i <= i1`）。"""
        return int(i) - self.start


def _round_kills(rnd: dict) -> list[dict]:
    """本回合的击杀，**含"回合结束那一枪"**，每条都带采样下标 `i`。

    `round_events()` 只收 `ticks[0] <= t <= ticks[-1]` 的事件，而"打死最后一个
    敌人、回合立刻结束"的那一枪恰好落在 `ticks[-1] + 1`，在采样轴之外。它通常仍
    带着刚结束回合的 `total_rounds_played`（同一 tick 的 `player_death` 在
    `round_end` 之前发出）；如果某份 demo 把它记到下一回合名下，extract 现在也会
    归回上一回合。两种情况下都靠 `t == ticks[-1] + 1` 找回
    （`scripts/training_data/tasks/_util.py::late_kills` 记的是同一件事）。实测
    4 份 demo / 78 个回合：

    * 49 个回合（63%）是这种结束方式（`end_reason ∈ {t_killed, ct_killed}`）；
    * 所有"落在采样范围之后"的击杀，与最后一帧的距离**恰好都是 1 tick**；
    * 这些回合受害者都活到最后一个采样帧（i1 = frames-1），所以把它当成"发生在
      最后一帧"不是近似。

    这里只补 `t == ticks[-1] + 1` 这一种：采样步长恒为 1 tick
    （`build_round_tensors` 卡了这一条），所以它对应的就是"下一个 tick"，模型看到
    的是训练样本里再普通不过的一帧 —— 受害者还活着、下一 tick 才被杀。两个回合
    之间更晚的 `post` 事件没有帧可挂，只进传统 KDA，不进 AI；最后一个回合结束
    之后的事件不属于任何回合，也不会出现在这里。
    """
    out = round_events(rnd, "kills")
    ticks = rnd.get("ticks") or []
    if not ticks:
        return out
    hi = int(ticks[-1])
    for e in (rnd.get("events") or {}).get("kills") or ():
        t = e.get("t")
        if t is not None and int(t) == hi + 1:
            out.append({**e, "i": len(ticks) - 1})
    return out


def duel_windows(
    tensors: RoundTensors, rnd: dict, *, lead: float = DUEL_LEAD
) -> list[Duel]:
    """这一回合所有"值得预测"的对枪 = 每一次**真实的**击杀 + 它之前 `lead` 秒。

    过滤规则和造训练数据时一模一样（`tasks/aim.py::write_round`）：不要自杀 /
    队友击杀 / 队伍未知，窗口里双方必须全程活着，`i1` 取受害者**最后一个活帧**
    （击杀事件那一帧受害者已经死了，位置列全是 0）。

    **回合结束那一枪也算**（`_round_kills`）：它挂在最后一帧上，`i1` 就是那一帧
    —— 受害者还活着、下一个 tick 才被打死。不补的话"打死最后一个敌人、回合立刻
    结束"的回合永远没有最后一场对枪（实测 63% 的回合是这样结束的）。

    返回按"死得早"排序 —— `assign_duels` 靠这个顺序裁决重叠的窗口。
    """
    alive = tensors.alive
    n_pl = alive.shape[1]
    secs = tensors.seconds
    out: list[Duel] = []
    for k in _round_kills(rnd):
        a, v, i_kill = int(k.get("a", -1)), int(k.get("v", -1)), int(k.get("i", -1))
        if a < 0 or v < 0 or a == v or i_kill < 0:
            continue                       # 摔死 / 被 C4 炸死 / 自杀：没有对手
        if a >= n_pl or v >= n_pl:
            continue
        ta, tv = int(tensors.team[a]), int(tensors.team[v])
        if ta == tv or min(ta, tv) < 0:
            continue                       # 团队击杀 / 队伍未知：只留敌人
        i1 = last_alive(alive[:, v], i_kill)
        if i1 < 0:
            continue                       # 这段里一次没活过：缺数据 / 没进场
        i0 = int(np.searchsorted(secs, secs[i1] - float(lead), side="left"))
        out.append(Duel(a=v, b=a, i0=i0, i1=int(i1),
                        t0=float(secs[i0]), t1=float(secs[i1])))
    out.sort(key=lambda d: (d.i1, d.i0))
    return out


def assign_duels(
    duels: Iterable[Duel], samples: np.ndarray
) -> list[Optional[Duel]]:
    """每个采样点归哪一场对枪（窗口不重叠时一眼可见，重叠时取"死得最早"的那场）。

    重叠只出现在**换人**的瞬间（比如 A 死了 0.3 秒后 B 被补枪）：那时候前一场
    已经结束、后一场还没到"死前 2 秒"以内的部分，取更近的那场才不会让弧线
    在两个人之间来回跳。
    """
    duels = list(duels)
    out: list[Optional[Duel]] = [None] * len(samples)
    for k, i in enumerate(samples):
        best: Optional[Duel] = None
        for d in duels:
            if d.i0 <= i <= d.i1 and (best is None or (d.i1, d.i0) < (best.i1, best.i0)):
                best = d
        out[k] = best
    return out


def plan_duels(tensors: RoundTensors, duels: Iterable[Duel]) -> list[DuelRun]:
    """给每场对枪分配 `run_aim` 里的一行：窗口 `[i1-192, i1]`，锚在击杀帧。

    跳过窗口里有人死过的：模型只在"双方全程活着"的窗口上训过（`tasks/aim.py`
    造数据时用 `alive[a, win].all()` 卡过这一条）。受害者的 i1 就是它最后一个
    活帧，所以正常情况下只有"击杀者在这一段里已经死了"（补枪 / 换人）才会命中
    这里 —— 那场对枪就不给弧线，免得喂一条分布外的窗口。

    返回的顺序就是 `run_aim` 结果的**行号**（`DuelRun.row`）。
    """
    out: list[DuelRun] = []
    for d in duels:
        start = max(0, d.i1 - (AIM_WINDOW - 1))
        stop = d.i1 + 1
        if not (tensors.alive[start:stop, d.a].all()
                and tensors.alive[start:stop, d.b].all()):
            continue
        out.append(DuelRun(duel=d, row=len(out), start=start, length=stop - start))
    return out


def probe_value(
    runs: Iterable[DuelRun], probe: dict, probs: np.ndarray
) -> Optional[float]:
    """`kill_probes()` 的一条探针 → 对应窗口里那一列的概率（`P(受害者赢)`）。

    匹配条件：同一对（受害者、击杀者）、且探针帧落在这一场的弧线区间里。
    同一对可能打两场（先 A 杀 B、后来 B 杀 A），取 `i1` 更近的那一场。

    返回 `None` = 这场对枪没有窗口（`plan_duels` 跳过了），调用方不计入分母。
    """
    victim, killer, i = int(probe["v"]), int(probe["a"]), int(probe["i"])
    best: Optional[DuelRun] = None
    for r in runs:
        if r.duel.a != victim or r.duel.b != killer:
            continue
        if not (r.duel.i0 <= i <= r.duel.i1):
            continue
        if best is None or r.duel.i1 < best.duel.i1:
            best = r
    if best is None:
        return None
    col = min(AIM_WINDOW - 1, max(0, best.column(i)))
    return float(probs[best.row, col])


# ═══════════════════════════════════════════════════════════════════════════
# 缓存
# ═══════════════════════════════════════════════════════════════════════════


def cache_root() -> Path:
    """`app/data/ai/`（跟着 `config.DATA_DIR` 走，测试可以把它指到临时目录）。"""
    from .. import config

    return Path(config.DATA_DIR) / "ai"


def _legacy_model_key(
    aim: model_lib.RunRef, spatial: model_lib.RunRef
) -> Optional[str]:
    """**旧公式**（`size + mtime`）算出来的模型指纹。

    只给 :func:`_migrate_legacy_model_prefix` 用。权重指纹换成内容指纹之后，
    磁盘上那些预测结果还是同一份权重的结果，只是前缀是旧公式算的。
    """
    try:
        payload = []
        for task, ref in (("aim", aim), ("spatial_prediction", spatial)):
            st = ref.weights.stat()
            payload.append({
                "task": task,
                "run": ref.name,
                "weights": ref.weights.name,
                "bytes": st.st_size,
                "mtime": st.st_mtime_ns,
                "config_mtime": ref.config.stat().st_mtime_ns,
            })
    except OSError:
        return None
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _migrate_legacy_model_prefix(
    aim: model_lib.RunRef, spatial: model_lib.RunRef, new_key: str
) -> None:
    """把缓存文件名里的旧模型前缀改成新的（一次性；找不到就什么都不做）。

    不搬的话，用户换公式那一刻会**再**冷启动一次（一场比赛几分钟），尽管
    `app/data/ai/<demo>/` 里躺着的是同一份权重的结果。搬只是重命名：同一场
    demo 的同一个回合，payload 格式没变。
    """
    old_key = _legacy_model_key(aim, spatial)
    if not old_key or old_key == new_key:
        return
    root = cache_root()
    if not root.is_dir():
        return
    moved = 0
    for demo_dir in root.iterdir():
        if not demo_dir.is_dir():
            continue
        for path in demo_dir.glob(f"{old_key}_*"):
            try:
                path.replace(demo_dir / (new_key + path.name[len(old_key):]))
            except OSError:  # 文件被占用 / 只读：下次启动再试，不影响结果
                continue
            moved += 1
    if moved:
        _log(f"AI 缓存前缀迁移 {old_key} -> {new_key}（{moved} 个文件）")


def demo_key(csz_path: Optional[str | Path], data: dict) -> str:
    """".csz 的 size+mtime" 的短哈希；没有文件时退回"内容形状"。

    这一份**故意还带 mtime**（权重那一侧已经换成内容指纹了）：`.csz` 是本机
    产物，"重新解析"就是真的换了文件，让缓存作废是对的。换成内容指纹要再搬一遍
    `app/data/ai/<demo>/` 和 `app/data/stats/` 的目录名，收益只是"重提取出一份
    一模一样的 .csz 也能命中" —— 不值。

    形状回退只用于 `/api/load` 直接吃内存的情况（现在的流程都会落 .csz），
    所以不追求唯一 —— 但同一个形状的两场比赛会共用缓存，这一点必须写在
    docstring 里，别当成 bug。
    """
    path = Path(csz_path) if csz_path else None
    try:
        if path is not None and path.is_file():
            st = path.stat()
            raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
            return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    except OSError:
        pass
    rounds = data.get("rounds") or []
    raw = "|".join(
        [
            str((data.get("meta") or {}).get("map")),
            str(data.get("format")),
            str(len(rounds)),
            str(sum(len(r.get("ticks") or []) for r in rounds)),
        ]
    )
    return "shape-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def cache_path(
    demo: str, model_key: str, stride: int, round_idx: int, root: Optional[Path] = None
) -> Path:
    base = Path(root) if root is not None else cache_root()
    return base / demo / f"{model_key}_s{stride}_r{round_idx}.json"


def _read_cache(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return None
    if not payload.get("ok"):
        return None
    payload["cached"] = True
    return payload


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError as exc:  # 缓存写不进去不该让预测失败
        _log(f"缓存写入失败（不影响本次结果）：{exc}")


# ═══════════════════════════════════════════════════════════════════════════
# 张量缓存（挂在会话的 data dict 上）
# ═══════════════════════════════════════════════════════════════════════════

_COMPUTE_LOCK = threading.Lock()


def _tensors_of(
    data: dict, round_idx: int, *, csz_path: Optional[str | Path] = None, progress=None
) -> RoundTensors:
    store: Optional[dict[int, RoundTensors]] = data.get("_ai_tensors")
    if store is None:
        store = {}
        data["_ai_tensors"] = store
    cached = store.get(round_idx)
    if cached is not None:
        return cached
    tensors = build_round_tensors(data, round_idx, csz_path=csz_path, progress=progress)
    store[round_idx] = tensors
    # 只留最近几个回合：一个约 10 MB，回合多的比赛不该被这一层吃掉内存
    while len(store) > _TENSORS_KEEP:
        oldest = next(iter(store))
        if oldest == round_idx:
            break
        store.pop(oldest, None)
    return tensors


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════


def _round_array(values: Iterable[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def _probs_matrix(mat: np.ndarray, digits: int = 4) -> list[list[float]]:
    return [[round(float(v), digits) for v in row] for row in mat]


def predict_round(
    data: dict,
    round_idx: int,
    *,
    csz_path: Optional[str | Path] = None,
    stride: int = DEFAULT_STRIDE,
    refresh: bool = False,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """一个回合的完整预测（含缓存）。**会阻塞**：首次约 15~25 秒。

    Args:
        progress: `progress(0..1, 一句话)`。界面靠它画进度条；不传就是静默。

    Raises:
        ModelUnavailable: 没有权重 / 没有 torch
        AiFeaturesError: 数据不满足模型前提（地图不认识、不是逐 tick 抽样）
    """
    say = progress or (lambda _pct, _msg: None)
    predictor = Predictor.instance()
    aim_ref, spatial_ref = predictor.refs()
    model_key = predictor.model_key
    key = demo_key(csz_path, data)
    stride = max(1, int(stride))
    path = cache_path(key, model_key, stride, round_idx)

    if not refresh:
        cached = _read_cache(path)
        if cached is not None:
            say(1.0, "命中磁盘缓存")
            return cached

    # 同一时刻只算一个回合：GPU 只有一张，而第二个人等一会儿就能直接吃缓存。
    if not _COMPUTE_LOCK.acquire(timeout=0.05):
        say(0.0, "另一个回合正在分析，排队中…")
        _COMPUTE_LOCK.acquire()
    try:
        if not refresh:
            cached = _read_cache(path)
            if cached is not None:
                say(1.0, "命中磁盘缓存")
                return cached

        t0 = time.perf_counter()
        rounds = data.get("rounds") or []
        if not 0 <= round_idx < len(rounds):
            raise AiFeaturesError(f"回合越界：{round_idx}（共 {len(rounds)} 个）")
        frames = len(rounds[round_idx].get("ticks") or [])
        n_samples = (frames + stride - 1) // stride

        say(_sub(_STAGE_LOAD, 0), "加载模型…")
        predictor.bundle(
            progress=lambda frac, msg: say(_sub(_STAGE_LOAD, frac), msg)
        )

        _log(
            f"分析回合 {round_idx}：{frames} 采样 → {n_samples} 个预测点"
            f"（stride={stride}）"
        )
        tensors = _tensors_of(
            data, round_idx,
            csz_path=csz_path,
            progress=lambda frac, msg: say(_sub(_STAGE_FEAT, frac), msg),
        )
        samples = np.arange(0, tensors.frames, stride, dtype=np.int64)
        if samples.size == 0:
            raise AiFeaturesError("这个回合没有采样点")

        spatial = predictor.run_spatial(
            tensors, samples,
            progress=lambda done, total: say(
                _sub(_STAGE_SPATIAL, done / max(1, total)),
                f"逐帧推理 {done}/{total}…",
            ),
        )

        duels = duel_windows(tensors, rounds[round_idx])
        assigned = assign_duels(duels, samples)
        runs = plan_duels(tensors, duels)
        # 掉一场"窗口里一个采样点都没有"的对枪（补枪时两场窗口重叠，死得早的那场
        # 会占满重叠部分）：留着它只会让界面说"这场有预测"但一个点都画不出来。
        shown = {id(d) for d in assigned if d is not None}
        duels = [d for d in duels if id(d) in shown]
        probs = predictor.run_aim(
            tensors, [(r.duel.i1, r.duel.a, r.duel.b) for r in runs],
            progress=lambda done, total: say(
                _sub(_STAGE_AIM, done / max(1, total)),
                f"对枪推理 {done}/{total}…",
            ),
        )

        # ── 对枪评分：同一条窗口的第 65 列（死前 2 秒）────────────────────
        # 不再为探针另跑一次前向：`kill_probes()` 只是把"哪一枪、哪一帧"点出来，
        # 概率从刚才那张矩阵里读（见 `probe_value`）。
        probes = stats_mod.kill_probes(tensors, rounds[round_idx])
        if probes:
            say(_sub(_STAGE_AIM, 0.99), f"对枪评分 {len(probes)} 场…")
            for q in probes:
                value = probe_value(runs, q, probs)
                if value is not None:
                    q["p"] = value
        round_stats = stats_mod.build_round_stats(
            tensors, rounds[round_idx], samples, spatial, probes
        )

        by_duel = {id(r.duel): r for r in runs}
        duel_a: list[Optional[int]] = [None] * samples.size
        duel_b: list[Optional[int]] = [None] * samples.size
        duel_p: list[Optional[float]] = [None] * samples.size
        win0: list[Optional[float]] = [None] * samples.size
        win1: list[Optional[float]] = [None] * samples.size
        for k, (i, duel) in enumerate(zip(samples, assigned)):
            if duel is None:
                continue
            win0[k] = round(duel.t0, 3)
            win1[k] = round(duel.t1, 3)
            run = by_duel.get(id(duel))
            if run is None:
                continue          # 窗口里死过人：这场对枪不给弧线
            duel_a[k] = duel.a
            duel_b[k] = duel.b
            duel_p[k] = float(probs[run.row, run.column(int(i))])

        say(0.99, "写缓存…")
        payload = {
            "ok": True,
            "version": CACHE_VERSION,
            "cached": False,
            "round": round_idx,
            "stride": stride,
            "frames": tensors.frames,
            "map": tensors.map_name,
            "map_id": tensors.map_id,
            "tick_hz": 64.0,
            "samples": [int(tensors.ticks[i]) for i in samples],
            "seconds": _round_array(tensors.seconds[samples], 3),
            "models": {
                "label": f"aim {aim_ref.name} · spatial {spatial_ref.name}",
                "aim": {**aim_ref.signature(), "path": str(aim_ref.weights)},
                "spatial": {
                    **spatial_ref.signature(),
                    "path": str(spatial_ref.weights),
                },
                "device": predictor.bundle().device,
            },
            "spatial": {
                "win_ct": _round_array(spatial["win"]),
                "alive": _probs_matrix(spatial["alive"]),
                "can_kill": _probs_matrix(spatial["can_kill"]),
                "next_death": _probs_matrix(spatial["next_death"]),
                "next_kill": _probs_matrix(spatial["next_kill"]),
            },
            "duels": [d.as_dict() for d in duels],
            "duel": {
                "a": duel_a,
                "b": duel_b,
                "p": [None if p is None else round(p, 4) for p in duel_p],
                "t0": win0,
                "t1": win1,
            },
            "stats": round_stats,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
        _write_cache(path, payload)
        _log(
            f"回合 {round_idx} 完成：{payload['ms']:.0f} ms · "
            f"{len(duels)} 场对枪 / {sum(1 for a in duel_a if a is not None)} 个预测点 · "
            f"{len(probes)} 场对枪评分"
        )
        say(1.0, "完成")
        return payload
    finally:
        _COMPUTE_LOCK.release()


def clear_cache(root: Optional[Path] = None) -> int:
    """删掉磁盘缓存，返回删了几个文件（测试和"换模型后想重算"用）。"""
    import shutil

    base = Path(root) if root is not None else cache_root()
    if not base.is_dir():
        return 0
    count = sum(1 for p in base.rglob("*.json"))
    shutil.rmtree(base, ignore_errors=True)
    return count


#: 预热只跑一次。
_warmed = False

def warmup(map_name: Optional[str] = None) -> None:
    """后台预热：把 numba 遮挡内核和地图 BVH 编好（第一次要 ~10 s）。

    遮挡内核是 `@njit(cache=False)`（见 `view_render.py` 里为什么故意关掉磁盘
    缓存），所以**每个进程的第一次**遮挡计算要多付 ~7 s 编译；地图 BVH 每个
    进程第一次也要几秒。这些和用户点了哪个回合无关，所以在页面刚打开、`/api/ai/
    status` 被调的时候扔到后台线程里做掉。失败就静默（没地图 / 没 numba →
    AI 本来也用不了，让真正的请求去报那条人话）。
    """
    global _warmed
    if _warmed:
        return
    _warmed = True

    def _run() -> None:
        global _warmed
        try:
            import numpy as _np

            from scripts.training_data.obstruction import load_map, pair_matrix

            t0 = time.perf_counter()
            mesh = load_map(map_name or "de_dust2")
            pair_matrix(
                _np.array([[0.0, 0.0, 100.0], [64.0, 0.0, 100.0]]),
                _np.zeros((2, 8, 3)),
                mesh,
                (_np.zeros((0, 3), _np.float32),
                 _np.zeros((0, 32, 32, 32), _np.uint8)),
            )
            _log(f"预热完成 {map_name or 'de_dust2'} · "
                 f"{(time.perf_counter() - t0) * 1000:.0f} ms")
        except Exception as exc:  # noqa: BLE001 —— 预热失败不该影响任何功能
            # 把"只跑一次"的标记放回去：失败最常见的原因是**地图还没解压**
            # （`maps/*.obj` 是用户跑起来之后点横幅才解出来的），那之后下一次
            # 打开回放页应该再试一次，否则这一进程永远等不到预热。
            _warmed = False
            _log(f"预热跳过（{type(exc).__name__}: {exc}）")

    threading.Thread(target=_run, name="ai-warmup", daemon=True).start()
