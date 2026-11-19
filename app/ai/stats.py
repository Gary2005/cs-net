"""AI 数据统计：把**一个回合的预测**折算成「每个选手的几个数」，再按场 / 按选手聚合。

数据从哪来
----------
不重新跑模型：`app/ai/service.py::predict_round()` 的载荷里已经有一整回合的
spatial（逐采样点的 `win_ct` / `next_death` / `next_kill` / `alive`）和每场对枪的
aim 概率。这个模块只做两件事：

1. `kill_probes()` —— 从回合事件里挑出「对枪评分」要问模型的那几个时刻（击杀前
   2 秒），交给 `predict_round` 顺手多跑一次 aim；
2. `build_round_stats()` —— 把预测和事件对起来，算出这一回合每个人的五个数。

「一场对枪」的定义和传统统计共用一份（`visualizer.stats.duel_kills()`）：只算敌人
之间的击杀，自杀 / 队友误杀 / 落在采样范围外的事件都不进分母。

五个指标怎么算
--------------
**回合影响力**：把这一回合的 CT 胜率曲线按采样点做差分。`win[0..S-1]` 是逐采样点
的 CT 胜率，末尾补一格真实结果（CT 赢 1、输 0）得到 `win[0..S]`，第 k 步的涨落
`d[k] = win[k+1] - win[k]` 记在**第 k 个采样点**的存活名单上：那时还有 x 个 CT、
y 个 T，每个 CT `+d[k]/x`、每个 T `-d[k]/y`（涨落之后才死的人也要承担这一步）。
已经死掉的人不再累计；**某一队已经全灭**（炸弹已下、等拆包）时沿用最后一个还有人
活着的采样点的名单。这一步加起来正好是 0，所以每回合十个人加起来严格是 0。
平均值 = Σ(每回合) / 回合数。

**击杀影响力**：设 `win_p[t]` = 选手 p 那一队在第 t 个采样点的胜率（`win_ct` 按
他是 CT / T 折一下，队伍未知时恒 0.5）。对 p 造成的每次击杀，取击杀事件**前一个
采样点**和**后一个采样点**的 `win_p`，差就是这一次的收益。平均值 = Σ / 次数。

⚠ 击杀**正好落在最后一个采样点上**（实测 63% 的回合是"最后一个敌人一死回合就
结束"）时没有"后一个采样点"：这时**按回合胜负补** —— 赢了补 1.0、输了补 0.0。
落在第一个采样点之前的击杀没有"前一个采样点"，用 0.5（开局前的先验，和"什么都
没发生"一致）。

**平均站位死亡风险**：p 活着的每个采样点上 `next_death[p]` 的平均值 —— 模型对
"下一个死的是谁"给出的概率，只在活人之间归一化。**平均站位击杀率**同理，用
`next_kill[p]`。这两个都是"站在这个位置、这个时刻"的风险 / 收益，逐帧取平均。

**对枪评分**：对 p 造成的每次击杀，问模型「击杀发生前 2 秒，p 赢下这场对枪的
概率」p̂，这一次的分数是 `1 - p̂`（模型越不看好、越说明这一枪打得漂亮）；p 被击杀
时问「2 秒前 p 赢的概率」p̂，分数是 `-p̂`（模型越看好、这次死越亏）。
对枪评分 = （Σ 击杀分数 + Σ 被击杀分数）/（击杀次数 + 被击杀次数）。

为什么是"前 2 秒"
------------------
网页上的对枪是从死前 3 秒开始看的（`service.DUEL_LEAD`），而模型的窗口是往回看
4 秒（`features.AIM_WINDOW`）。取击杀前 2 秒那一刻的预测，等于让模型只看到
`[死前 2 秒 - 4 秒, 死前 2 秒]` —— 也就是"人还没死"的那段，天然不包含已发生的结果。
一个击杀同时算两个人：击杀者的 `1 - p̂` 和受害者的 `-p̂`，所以
`duel_n` 的合计 == 传统统计里 `duels` 的合计。

一条击杀同时是"击杀者的收益"和"受害者的损失"，两个方向各问模型一次
（`(A, B)` 和 `(B, A)`），不做 `1 - p` 的翻转：模型在两种朝向下不是严格互补的，
而训练集本来就是 50% 交换过的，两边都在分布内。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from visualizer.stats import duel_kills

#: 对枪评分取"击杀发生前几秒"的模型预测。
KILL_PROBE_SEC = 2.0

#: 击杀落在第一个采样点之前时的"击杀前胜率"（开局先验）。
WIN_PRIOR = 0.5

#: `build_round_stats()` 返回的累计字段：求和 + 计数，聚合就是逐项相加。
#: 回合影响力不进这里 —— 它一个回合就一个值，聚合时单独加 `round_sum`。
SUM_KEYS: tuple[str, ...] = (
    "kill_sum", "kill_n",
    "risk_sum", "risk_n",
    "kr_sum", "kr_n",
    "duel_sum", "duel_n",
)


def kill_probes(tensors, rnd: dict) -> list[dict]:
    """本回合每一场对枪的「击杀前 `KILL_PROBE_SEC` 秒」采样点。

    返回 `[{a, v, t, i}, ...]`：`a` 击杀 `v`，`t` 是击杀 tick，`i` 是问模型用的
    采样下标（`seconds <= 击杀时刻 - 2 s` 的最后一个采样点，夹在 `[0, frames-1]`）。

    回合结束那一枪（`t == ticks[-1] + 1`，采样轴之外）用"最后一帧 + 1 tick"当成
    击杀时刻，所以它的 `i` 落在最后几帧上 —— 和 `service._round_kills()` 把那一枪
    挂到最后一帧是同一件事。
    """
    ticks = np.asarray(tensors.ticks, dtype=np.int64)
    secs = np.asarray(tensors.seconds, dtype=np.float64)
    frames = int(ticks.size)
    if frames < 1:
        return []
    tick_hz = 64.0
    out: list[dict] = []
    for k in duel_kills(rnd, int(tensors.alive.shape[1])):
        t = int(k["t"])
        ik = int(np.searchsorted(ticks, t, side="right")) - 1
        if ik < frames - 1:
            ik = max(0, ik)
            t_sec = float(secs[ik])
            if t > int(ticks[ik]):                    # 落在两个采样点之间
                t_sec += (t - int(ticks[ik])) / tick_hz
        else:
            # 击杀事件在采样范围之外（回合结束那一枪）：从最后一个采样点往后推
            t_sec = float(secs[-1]) + max(0, t - int(ticks[-1])) / tick_hz
        target = t_sec - KILL_PROBE_SEC
        i = int(np.searchsorted(secs, target, side="right")) - 1
        out.append({"a": int(k["a"]), "v": int(k["v"]), "t": t,
                    "i": int(min(max(i, 0), frames - 1))})
    return out


def _team_win(win_ct: np.ndarray, team: np.ndarray, p: int) -> np.ndarray:
    """选手 `p` 那一队的胜率序列（CT = `win_ct`，T = `1 - win_ct`，未知 = 0.5）。"""
    side = int(team[p]) if p < team.size else -1
    if side == 0:
        return win_ct
    if side == 1:
        return 1.0 - win_ct
    return np.full(win_ct.shape, WIN_PRIOR, dtype=np.float64)


def _won_ct(rnd: dict) -> Optional[float]:
    """回合结果换成 CT 的 1.0 / 0.0；`winner` 不是 T/CT 时给 None。"""
    winner = str(rnd.get("winner") or "").strip().upper()
    if winner == "CT":
        return 1.0
    if winner == "T":
        return 0.0
    return None


def _won(rnd: dict, team: np.ndarray, p: int) -> float:
    """回合结束时 `p` 那一队到底赢没赢（用来补"没有下一个采样点"的那一步）。"""
    ct = _won_ct(rnd)
    side = int(team[p]) if p < team.size else -1
    if ct is None:
        return WIN_PRIOR
    if side == 0:
        return ct
    if side == 1:
        return 1.0 - ct
    return WIN_PRIOR


def _carry_alive(mask: np.ndarray) -> np.ndarray:
    """逐帧的存活名单；某一帧已经全灭时沿用最后一个还有人活着的帧。

    炸弹已下、等拆包的阶段一方可能已经全灭，但胜率曲线还在动 —— 那几步必须有人
    来分，否则涨落会白丢（回合影响力也就不再是 0 和的）。
    """
    alive_here = mask.any(axis=1)
    src = np.where(alive_here, np.arange(alive_here.size), -1)
    np.maximum.accumulate(src, out=src)
    keep = src >= 0
    return np.where(keep[:, None], mask[np.maximum(src, 0)], False)


def build_round_stats(
    tensors,
    rnd: dict,
    samples: np.ndarray,
    spatial: dict[str, np.ndarray],
    probes: list[dict],
    probs: np.ndarray,
) -> dict:
    """一回合的 AI 统计（逐选手的求和 + 计数，聚合时直接相加）。

    Args:
        samples: `(S,)` 这一回合跑过 spatial 的**采样下标**（升序，即
            `np.arange(0, frames, stride)`）。tick 轴由 `tensors.ticks[samples]`
            现算 —— 这里千万别把 tick 当成下标用（那会让每次击杀都落到"回合最后
            一个采样点之后"，影响力全变成 ±1 分钱那么小）。
        spatial: `Predictor.run_spatial()` 的返回值（`win` / `next_death` / `next_kill`）。
        probes: `kill_probes()` 的结果。
        probs: `len(probes) * 2` 个概率 —— 前一半是 `P(击杀者赢)`，后一半是
            `P(受害者赢)`（同一个 `i`，两个方向各问一次）。

    返回 `{"players": [{idx, kill_sum, ..., duel_sum, duel_n}, ...]}`。
    """
    n = int(tensors.alive.shape[1])
    idx = np.asarray(samples, dtype=np.int64).reshape(-1)
    ticks = np.asarray(tensors.ticks, dtype=np.int64)[idx]      # 采样点的 tick
    win_ct = np.asarray(spatial["win"], dtype=np.float64)
    nd = np.asarray(spatial["next_death"], dtype=np.float64)
    nk = np.asarray(spatial["next_kill"], dtype=np.float64)
    team = tensors.team
    alive = np.asarray(tensors.alive[idx], dtype=bool)           # (S, n)

    acc: dict[int, dict[str, float]] = {
        p: {k: 0.0 for k in SUM_KEYS} for p in range(n)
    }

    # ── 站位风险 / 击杀率：活着的每个采样点各取一次概率，再平均 ──────────
    for p in range(n):
        live = alive[:, p]
        cnt = int(live.sum())
        if cnt:
            acc[p]["risk_sum"] += float(nd[live, p].sum())
            acc[p]["kr_sum"] += float(nk[live, p].sum())
        acc[p]["risk_n"] += cnt
        acc[p]["kr_n"] += cnt

    # ── 击杀影响力：跨过击杀那一 tick 的胜率差 ────────────────────────────
    def win_at(p: int, t: int) -> tuple[float, float]:
        """击杀 tick `t` 前后两个采样点的 `p` 队胜率。

        `after` 取**第一个 `>= t` 的采样点**：击杀那一 tick 的采样里受害者已经是死的
        （`alive` 就是 False，`aim` 的窗口终点也因此往前找一帧），所以那一帧已经在
        "事后"了。`before` 是它前一个采样点。击杀落在采样点之间时，这一段就是
        "跨过这一枪"的胜率差 —— 最大 0.25 s（`stride / 64`）的量化误差，不再往两边
        多取一格：多取会把别的交火也算进这一枪的头上。
        """
        series = _team_win(win_ct, team, p)
        after = int(np.searchsorted(ticks, t, side="left"))     # 第一个 >= t
        before = after - 1
        w_before = float(series[before]) if before >= 0 else WIN_PRIOR
        w_after = float(series[after]) if after < ticks.size else _won(rnd, team, p)
        return w_before, w_after

    for k in duel_kills(rnd, n):
        a, t = int(k["a"]), int(k["t"])
        b, af = win_at(a, t)
        acc[a]["kill_sum"] += af - b
        acc[a]["kill_n"] += 1

    # ── 回合影响力：逐采样点的胜率涨落，由这一步开始时活着的人均分 ────────
    side = np.asarray(team, dtype=np.int64)
    winner = _won_ct(rnd)
    end = float(win_ct[-1]) if winner is None else winner   # 看不出谁赢就不补
    delta = np.diff(np.append(win_ct, end))    # delta[k] = win[k+1] - win[k]

    ct_mask = _carry_alive(alive & (side == 0))
    t_mask = _carry_alive(alive & (side == 1))
    ct_n = ct_mask.sum(axis=1)
    t_n = t_mask.sum(axis=1)
    live_step = (ct_n > 0) & (t_n > 0)   # 整场都没出现过的一方：没法分
    ct_share = np.where(live_step, delta / np.maximum(ct_n, 1), 0.0)
    t_share = np.where(live_step, -delta / np.maximum(t_n, 1), 0.0)
    impact = (ct_mask.astype(np.float64).T @ ct_share
              + t_mask.astype(np.float64).T @ t_share)

    # ── 对枪评分：击杀前 2 秒的模型预测 ───────────────────────────────────
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probes and probs.size == len(probes) * 2:
        half = len(probes)
        for j, pr in enumerate(probes):
            p_killer = float(probs[j])                 # P(击杀者赢)
            p_victim = float(probs[half + j])          # P(受害者赢)
            acc[pr["a"]]["duel_sum"] += 1.0 - p_killer
            acc[pr["a"]]["duel_n"] += 1
            acc[pr["v"]]["duel_sum"] += -p_victim
            acc[pr["v"]]["duel_n"] += 1

    players = []
    for p in range(n):
        row = {"idx": p}
        row.update({k: (int(v) if k.endswith("_n") else round(float(v), 6))
                    for k, v in acc[p].items()})
        row["round_impact"] = round(float(impact[p]), 6)
        players.append(row)
    return {
        "players": players,
        "probe_sec": KILL_PROBE_SEC,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 聚合（一场比赛 / 一个选手跨场）
# ═══════════════════════════════════════════════════════════════════════════


#: 聚合用的累计字段：SUM_KEYS 再加上"每回合影响力之和"。
TOTAL_KEYS: tuple[str, ...] = SUM_KEYS + ("round_sum",)


def _blank_totals() -> dict[str, float]:
    return {k: 0.0 for k in TOTAL_KEYS}


def _add_totals(into: dict[str, float], row: dict) -> None:
    for k in SUM_KEYS:
        into[k] += float(row.get(k) or 0.0)
    into["round_sum"] += float(row.get("round_impact") or 0.0)


def _avg(total: dict[str, float], sum_key: str, n_key: str) -> float:
    n = total.get(n_key) or 0
    return round(total.get(sum_key, 0.0) / n, 4) if n else 0.0


def derive(totals: dict[str, float], rounds: int) -> dict:
    """求和 → 界面上的五个平均值（外加各自的分母，方便显示 "n=12"）。"""
    return {
        # 1. 平均击杀影响力
        "kill_impact": _avg(totals, "kill_sum", "kill_n"),
        "kill_n": int(totals.get("kill_n") or 0),
        # 2. 平均回合影响力（分母是回合数，不是击杀数）
        "round_impact": (round(totals.get("round_sum", 0.0) / rounds, 4) if rounds else 0.0),
        # 3. 平均站位死亡风险 / 4. 平均站位击杀率（0~1，界面转成 %）
        "death_risk": _avg(totals, "risk_sum", "risk_n"),
        "risk_n": int(totals.get("risk_n") or 0),
        "kill_rate": _avg(totals, "kr_sum", "kr_n"),
        "kr_n": int(totals.get("kr_n") or 0),
        # 5. 对枪评分
        "duel_score": _avg(totals, "duel_sum", "duel_n"),
        "duel_n": int(totals.get("duel_n") or 0),
    }


def aggregate(payloads: list[dict]) -> dict:
    """多个回合的载荷 → 一场（或跨场）的 AI 统计。

    Args:
        payloads: `predict_round()` 的返回值（必须都带 `stats`）。

    返回 `{"players": [...], "rounds": n}`。每个选手是五个平均值 + 各自的分母 +
    **原始求和**（`totals`，跨场聚合时直接相加）+ 每回合明细（`per_round`，下标
    就是 `payloads` 的下标，缺的回合是 `None`）。

    ⚠ `rounds` 是这个人**打过**的回合数（他在那个回合里没数据就不算），而
    `round_impact` 的分母也是它 —— 和"平均每回合影响力"的字面意思一致。
    """
    n = 0
    for pay in payloads:
        for row in ((pay.get("stats") or {}).get("players") or []):
            n = max(n, int(row["idx"]) + 1)
    totals = [_blank_totals() for _ in range(n)]
    per_round: list[list[dict | None]] = [[] for _ in range(n)]
    played = [0] * n

    for pay in payloads:
        rows = {int(r["idx"]): r for r in ((pay.get("stats") or {}).get("players") or [])}
        for p in range(n):
            row = rows.get(p)
            if row is None:
                per_round[p].append(None)
                continue
            _add_totals(totals[p], row)
            played[p] += 1
            per_round[p].append({
                "kill_impact": (
                    round(float(row["kill_sum"]) / row["kill_n"], 4) if row["kill_n"] else None
                ),
                "kill_n": int(row["kill_n"]),
                "round_impact": round(float(row.get("round_impact") or 0.0), 4),
                "death_risk": (
                    round(float(row["risk_sum"]) / row["risk_n"], 4) if row["risk_n"] else None
                ),
                "kill_rate": (
                    round(float(row["kr_sum"]) / row["kr_n"], 4) if row["kr_n"] else None
                ),
                "duel_score": (
                    round(float(row["duel_sum"]) / row["duel_n"], 4) if row["duel_n"] else None
                ),
                "duel_n": int(row["duel_n"]),
            })

    players = []
    for p in range(n):
        out = {"idx": p, "rounds": played[p],
               "totals": {k: round(float(v), 6) for k, v in totals[p].items()}}
        out.update(derive(totals[p], played[p]))
        out["per_round"] = per_round[p]
        players.append(out)
    return {"players": players, "rounds": len(payloads),
            "probe_sec": KILL_PROBE_SEC}


def combine(entries: list[dict]) -> dict:
    """多场比赛里**同一个选手**的 `{"totals", "rounds"}` → 跨场合计。

    玩家页的"所有比赛平均"就是这件事：先把每场的累计量相加，再除一次 —— 不是
    把各场的平均值再平均（那样 5 回合的一场会和 24 回合的一场一样重）。
    """
    total = _blank_totals()
    rounds = 0
    for e in entries:
        for k in TOTAL_KEYS:
            total[k] += float((e.get("totals") or {}).get(k) or 0.0)
        rounds += int(e.get("rounds") or 0)
    out = {"rounds": rounds,
           "totals": {k: round(float(v), 6) for k, v in total.items()}}
    out.update(derive(total, rounds))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 一场比赛：把每个回合的预测接起来 + 落盘缓存
# ═══════════════════════════════════════════════════════════════════════════

#: 比赛级缓存的格式版本。`service.CACHE_VERSION` 变了（回合载荷换口径）时这个
#: 汇总也会过期，所以文件里连那个版本号一起存，读的时候比一下。
#:   v3：回合影响力记到「涨落之前」那一帧的存活名单上，某队全灭后沿用最后一个
#:       还有人活着的帧
MATCH_VERSION = 3


def match_cache_path(demo: str, model_key: str, stride: int,
                     root=None):
    """`app/data/ai/<demo>/<models>_s<stride>_match.json`。

    和回合缓存放在同一个目录：一场比赛的全部分析产物（回合载荷 + 汇总）在一起，
    删 `app/data/ai` 就一起没了。
    """
    from .service import cache_root

    base = Path(root) if root is not None else cache_root()
    return base / demo / f"{model_key}_s{stride}_match.json"


def read_match_cache(path: Path, *, round_version: int):
    """读比赛级缓存；版本不对 / 文件坏了 / 回合缓存已经换代 → None。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != MATCH_VERSION:
        return None
    if int(payload.get("round_version") or 0) != int(round_version):
        return None
    if not payload.get("ok"):
        return None
    payload["cached"] = True
    return payload


def write_match_cache(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError as exc:  # 缓存写不进去不该让统计失败
        print(f"[ai] 比赛级缓存写入失败（不影响本次结果）：{exc}", flush=True)


def predict_match(
    data: dict,
    csz_path=None,
    *,
    stride=None,
    refresh: bool = False,
    progress=None,
) -> dict:
    """一场比赛的 AI 统计：逐回合 `predict_round`（有缓存就是读文件）+ 汇总。

    返回的 `players` 每行都带 `steamid` / `name`（从 `data["players"]` 取），
    玩家页拿它按 SteamID 找自己那行。

    `progress(分数, 说明)` 报整体进度；每个回合内部的进度映射到 `[r/N, (r+1)/N]`。
    单个回合算不出来（数据坏了 / 不是逐 tick 的）只会记进 `errors` 并跳过，
    不让整场统计变成 500。
    """
    from . import service

    say = progress or (lambda _pct, _msg: None)
    stride = int(stride or service.DEFAULT_STRIDE)
    n_round = len(data.get("rounds") or [])

    payloads: list[dict] = []
    errors: list[dict] = []
    for ri in range(n_round):
        # 每个回合开始先报一次：`predict_round` 命中磁盘缓存时一步都不报，
        # 进度条会一直停在上一格，看着像卡住。
        say(ri / max(1, n_round), f"回合 {ri + 1}/{n_round} · 准备…")

        def sub(frac: float, msg: str, _ri: int = ri) -> None:
            say((_ri + min(max(frac, 0.0), 1.0)) / max(1, n_round),
                f"回合 {_ri + 1}/{n_round} · {msg}")

        try:
            payloads.append(service.predict_round(
                data, ri, csz_path=csz_path, stride=stride, refresh=refresh,
                progress=sub,
            ))
        except Exception as exc:                      # noqa: BLE001
            errors.append({"round": ri,
                           "error": f"{type(exc).__name__}: {exc}"})
            say((ri + 1) / max(1, n_round),
                f"回合 {ri + 1}/{n_round} · 跳过（{exc}）")

    agg = aggregate(payloads)
    roster = list(data.get("players") or [])
    for row in agg["players"]:
        idx = int(row["idx"])
        p = roster[idx] if 0 <= idx < len(roster) else {}
        row["steamid"] = str(p.get("steamid") or "")
        row["name"] = str(p.get("name") or "")
    say(1.0, "汇总…")
    agg.update({
        "ok": True,
        "cached": False,
        "version": MATCH_VERSION,
        "round_version": service.CACHE_VERSION,
        "stride": stride,
        "model_key": service.Predictor.instance().model_key,
        "map": (data.get("meta") or {}).get("map"),
        "rounds_total": n_round,
        "errors": errors,
    })
    return agg
