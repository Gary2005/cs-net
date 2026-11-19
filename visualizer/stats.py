"""整场比赛的统计（每场 / 每回合）—— 回放页右侧那个「统计」弹窗的数据。

数据源只有**已经在内存里**的 demo_extract 结果（``_session["data"]``）：

* ``rounds[i].events.kills``  —— 每次击杀：攻击者、受害者、助攻、是否爆头
* ``rounds[i].events.damage`` —— 每一次伤害事件：谁打的、谁挨的、报了多少
* ``rounds[i].players[p][...]`` —— 逐采样点的位置 / 视角（新指标要用）

为什么在服务端算
----------------
回合载荷是**按需**拉的（一个回合 100 KB ~ 1 MB）。统计却要跨全部回合 —— 让浏览器
把 24 个回合全拉下来再自己加，等于为了几个数字把整场比赛又传一遍。``data`` 本来
就在服务端内存里，扫一遍 events 是毫秒级的，所以算一次、缓存进会话。

「伤害」为什么必须按剩余血量截断
--------------------------------
``events.damage[].hp`` 是 ``player_hurt`` 报的**这一枪的伤害**：按武器伤害算出来的
名义值，**不因血槽不够而截断**。于是冒出单回合单人「承伤 443」这种数 —— 实测
（cache R3）: woxic 用 AWP 把满血的 ropz 一枪打死，报的伤害是 443，而那个人这个
回合只挨了这一下，血槽只有 100。AWP 满蓄力 / 穿墙会把名义伤害放大好几倍。

所以这里给每个选手每回合开一个血槽（初值 100），逐枪取

    take = min(报的伤害, 剩余血槽)

记的是「实际扣掉的血」，而不是武器的名义伤害。实测 4 份真实 demo，截断后
Σ承伤 / 真实掉血（逐采样 hp 下降量）= 1.02 ~ 1.05。游戏内记分板也是这个口径。

口径与本仓库其他两处的关系
--------------------------
* ``app/metadata.py`` 存进库的 ``adr_damage`` 用 ``kills[].dmg``（只有致命那一枪），
  比真实伤害低一个量级，见 ``visualizer/README.md`` §3.5 的对照表。
* HLTV 系的 ADR 把名义伤害**原样累加**，不截断。所以本面板的 ADR 会比 HLTV 口径低
  20% 左右（anubis 15604 → 12377），换来的是「每回合每条命最多 100」这个能对账的
  事实，以及「造成之和 + 世界伤害 == 承受之和」的守恒。两种口径都合理，这里选了后者。

自伤（``a == v``）两边都不算；世界伤害（``a == -1``，摔落 / C4）只算承伤、不算
任何人的输出 —— 但它同样扣血，所以也要扣血槽（漏掉这一步截断会偏大 1~5%）。

新指标：首杀 / 先手开枪 / 侧背身 / 穿烟
--------------------------------------
除了 K / D / A / 伤害，面板还要五个「事件级」的指标。它们的共同前提是**一场对枪**
（= 一次击杀的击杀方和被杀方），所以先定下「什么算一场对枪」：

* 自杀（``a == v``）、世界伤害致死（``a == -1``）、队友击杀（两人同一队）**都不算**
  —— 这些没有对手，"谁先打谁"无从谈起；
* 只收**落在本回合采样范围内**的事件（外加回合结束那一枪 ``ticks[-1] + 1``，和
  ``app/ai/service.py::_round_kills`` 同一件事）。``events`` 是按「上一回合结束 →
  本回合结束」分桶的，round 0 里还混着赛前热身的几十条击杀；K / D 那几列沿用老
  口径（和库里的记分板逐项相等），新指标不收热身。

于是每个选手每回合有四个分母（前端把比例显示成 "7/14"，四个数都给了）：

    enemy_kills    造成对枪击杀的次数（= 下面三个「击杀占比」的分母）
    enemy_deaths   被对面击杀的次数（= 侧背身被击杀的分母）
    duels          enemy_kills + enemy_deaths（= 先手开枪占比的分母）

**首杀 / 首死**：本回合第一个造成对枪击杀的人 / 第一个被对面击杀的人。一回合最多
各一个，所以这两列加起来的全部选手之和 == 回合数。

**先手开枪占比**：对每场对枪，取**死亡前 4 秒**里两人之间最早的伤害事件 —— 谁先把
子弹打在对面身上，这场对枪的先手就是他（开枪但没打中不算，因为这里只认伤害事件）。
分母是 ``duels``。注意是「这场对枪里的先手」，不是「这一回合的第一个开枪」。

**侧背身击杀 / 被击杀**：判定时刻是**击杀者第一次对受害者造成伤害**的那一帧 ——
取受害者当时的 ``yaw``（水平朝向），和「受害者眼睛 → 击杀者眼睛」这条水平线段比
夹角，**≥ 40°** 就算受害者没在看那个人（侧身或背身）。两个计数器是同一件事的两端：
A 对 B 打出一个侧背身，就给 A 的 ``side_kills`` +1、同时给 B 的 ``side_deaths`` +1。
所以「A 的侧背身击杀数之和」==「B 的侧背身被击杀数之和」。

**穿烟击杀**：被判定的那一帧是**受害者死前的最后一个活帧**（和 AI 的 ``aim`` 窗口
终点、``spatial`` 的标签是同一天），判据是**击杀者到受害者的连线有没有被烟挡住** ——
直接用 AI 的遮挡矩阵（``scripts/training_data/obstruction.py::Occluder``，0 通 / 1 烟 /
2 墙），取 ``obstruction[a][v] == 1``。同一套几何意味着「面板说这是穿烟击杀」和
「模型看到的那一帧是穿烟」永远是同一个事实。

代价：``Occluder`` 要给每帧算两两射线（3~6 ms/帧），这里一个回合只算有击杀的那几帧
（实测 24 回合的整场 ≈ 0.6~1.5 s，首次还要加 1~3 s 建地图 BVH）。所以它是**懒加载 +
可失败**的：地图不认识 / 没装 numba 时 ``smoke_ok = False``，前端那一列显示 "—"，
其余指标照常 —— 传统统计不能因为一个可选列整块不可用。
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np

#: 空手/世界造成的伤害在事件里没有攻击者（``a == -1``）；自伤（``a == v``，自己的
#: 火 / 雷 / 摔）两边都不算 —— 它既不是"输出"，也不该记成"被打"。
WORLD = -1

#: 侧背身判定的水平夹角阈值（度）：受害者**朝向**和**看向击杀者的方向**差这么多，
#: 就算没在看那个人（侧身 / 背身）。需求里写的就是 40。
SIDE_BACK_DEG = 40.0

#: 先手开枪 / 侧背身的观察窗口（秒）：死亡前这么久之内的事件才看。
KILL_WINDOW_SEC = 4.0


def _idx(value: Any) -> int:
    """事件里的玩家下标 → int；缺失/非法一律当成"没有人"（-1）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _side(round_: dict, i: int) -> str:
    teams = round_.get("teams") or []
    if not (0 <= i < len(teams)):
        return "?"
    side = str(teams[i]).strip().upper()
    return side if side in ("T", "CT") else "?"


# ── 队伍划分 ────────────────────────────────────────────────────────────────


def team_groups(rounds: list[dict], n: int) -> list[list[int]]:
    """把选手下标分成两队。

    规则：**任何一回合都没有站在对面的两个人，就是队友。** 半场换边（同一队上下半场
    分属 CT / T）和只打了一半的替补都能扛住 —— "第一回合站哪边"这种朴素分法会把
    换边之后的同一队拆成两组。

    这和 ``app/metadata.py`` 里算记分板用的是同一条规则（那里的注释一样）。两边
    各留一份实现：``app`` 依赖 ``visualizer``，反过来不成立（单独跑
    ``visualizer/server.py`` 时 ``app`` 根本不存在），而这条规则一共三十行、
    不依赖任何东西。
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sides = [[_side(rnd, i) for i in range(n)] for rnd in rounds]

    for i in range(n):
        for j in range(i + 1, n):
            agree = disagree = False
            for row in sides:
                si, sj = row[i], row[j]
                if si == "?" or sj == "?":
                    continue
                if si == sj:
                    agree = True
                else:
                    disagree = True
                if agree and disagree:
                    break
            # 从没同边过的两个人不算队友：不然两个从未同场的人会被并进一队
            if agree and not disagree:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    # 人多的队在前，其次按下标，结果稳定可复现
    return sorted(buckets.values(), key=lambda g: (-len(g), g[0]))


def _winner_group(round_: dict, groups: list[list[int]]) -> Optional[int]:
    """这一回合是哪个队赢的；看不出来（winner 不是 T/CT）返回 None。"""
    winner = str(round_.get("winner") or "").strip().upper()
    if winner not in ("T", "CT"):
        return None
    for gi, group in enumerate(groups):
        for i in group:
            side = _side(round_, i)
            if side == "?":
                continue
            if side == winner:
                return gi
            return 1 - gi if len(groups) == 2 else None
    return None


def _first_side(rounds: list[dict], group: list[int]) -> str:
    """这一队**开场站哪边**（只用来给表头一个 CT / T 的颜色和说明）。

    换边之后同一队会站另一边，所以这里刻意只取第一次看到的那一边，别当成"这队就是
    CT"。
    """
    if not group:
        return "?"
    idx = group[0]
    for rnd in rounds:
        side = _side(rnd, idx)
        if side != "?":
            return side
    return "?"


# ═══════════════════════════════════════════════════════════════════════════
# 事件级的原料：采样轴 / 位置 / 视角
# ═══════════════════════════════════════════════════════════════════════════


class _Geo:
    """一回合的采样轴 + 按选手懒加载的位置 / 视角 / 存活列。

    只碰**用得到的那几个选手**（一回合 8 次击杀最多涉及 8 个人），也不构造
    ``(T,10,20)`` 那种整块张量 —— 统计只要几个数，用不着把整回合的特征都摆出来。
    每列都是原始 HU / 度，不做归一化（角度与尺度无关）。
    """

    def __init__(self, rnd: dict) -> None:
        self._rnd = rnd
        self.ticks = np.asarray(rnd.get("ticks") or [], dtype=np.int64)
        self._pos: dict[int, np.ndarray] = {}
        self._yaw: dict[int, np.ndarray] = {}
        self._alive: dict[int, np.ndarray] = {}

    # ── 坐标轴 ────────────────────────────────────────────────────────────

    def idx_at(self, tick: int) -> int:
        """`<= tick` 的最后一个采样下标；早于第一个采样返回 -1。"""
        if not self.ticks.size:
            return -1
        return int(np.searchsorted(self.ticks, int(tick), side="right")) - 1

    # ── 选手列 ────────────────────────────────────────────────────────────

    def xy(self, p: int, i: int) -> tuple[float, float]:
        arr = self._pos.get(p)
        if arr is None:
            rec = self._rnd["players"][p]
            arr = np.stack(
                [np.asarray(rec["x"], dtype=np.float64),
                 np.asarray(rec["y"], dtype=np.float64)], axis=-1
            )
            self._pos[p] = arr
        return float(arr[i, 0]), float(arr[i, 1])

    def yaw(self, p: int, i: int) -> float:
        arr = self._yaw.get(p)
        if arr is None:
            arr = np.asarray(self._rnd["players"][p]["yaw"], dtype=np.float64)
            self._yaw[p] = arr
        return float(arr[i])

    def alive(self, p: int) -> np.ndarray:
        arr = self._alive.get(p)
        if arr is None:
            arr = np.asarray(self._rnd["players"][p]["alive"], dtype=bool)
            self._alive[p] = arr
        return arr

    def last_alive_idx(self, p: int, tick: int) -> int:
        """`<= tick` 的最后一个**活着**的采样下标（死前最后一帧）；没有则 -1。"""
        alive = self.alive(p)
        i = min(self.idx_at(tick), alive.size - 1)
        if i < 0:
            return -1
        idx = np.flatnonzero(alive[: i + 1])
        return int(idx[-1]) if idx.size else -1


def duel_kills(rnd: dict, n: int) -> list[dict]:
    """本回合的「对枪击杀」：有敌人对手、两人都在名单里、落在采样范围内。

    返回 `[{t, a, v}, ...]`，按时间排序（同一 tick 保持原顺序，``sorted`` 是稳定的）。
    过滤理由见模块 docstring 的「新指标」一节。

    这是**唯一**一份"什么算一场对枪"的定义：传统统计的五个新指标和 AI 统计
    （`app/ai/stats.py`）都从这里取，两边的 ``enemy_kills`` / ``duels`` / 对枪评分
    的分母才是同一个数。
    """
    ticks = rnd.get("ticks") or []
    lo = int(ticks[0]) if ticks else 0
    hi = int(ticks[-1]) + 1 if ticks else -1
    out: list[dict] = []
    for k in (rnd.get("events") or {}).get("kills") or []:
        a, v = _idx(k.get("a")), _idx(k.get("v"))
        t = k.get("t")
        if t is None or not (0 <= a < n and 0 <= v < n) or a == v:
            continue                                  # 摔死 / C4 / 自杀：没有对手
        sa, sv = _side(rnd, a), _side(rnd, v)
        if sa != "?" and sa == sv:
            continue                                  # 队友击杀：不是对枪
        if not (lo <= int(t) <= hi):
            continue                                  # 热身 / 别的回合的事件
        out.append({"t": int(t), "a": a, "v": v})
    out.sort(key=lambda e: e["t"])
    return out


def _pair_damage(
    rnd: dict, a: int, v: int, lo: int, hi: int
) -> list[tuple[int, int]]:
    """`a` ↔ `v` 之间、tick 在 `[lo, hi]` 里的伤害事件，按时间排序。

    返回 `[(tick, 开枪的人), ...]`；`开枪的人` 一定是 `a` 或 `v`（第三个方向的伤害
    不算"这场对枪的先手"）。
    """
    out: list[tuple[int, int]] = []
    for d in (rnd.get("events") or {}).get("damage") or []:
        t = d.get("t")
        if t is None or not (lo <= int(t) <= hi):
            continue
        da, dv = _idx(d.get("a")), _idx(d.get("v"))
        if da == a and dv == v:
            out.append((int(t), a))
        elif da == v and dv == a:
            out.append((int(t), v))
    out.sort()
    return out


def _side_back_angle(geo: _Geo, victim: int, other: int, tick: int) -> Optional[float]:
    """受害者「朝向」与「看向对方」的水平夹角（度，0~180）；算不出来返回 None。

    Source 的 ``forward = (cos yaw·cos pitch, sin yaw·cos pitch, -sin pitch)``，
    水平投影就是 ``(cos yaw, sin yaw)``（见 ``view_render.directions`` 的注释）。
    夹角用 ``atan2(|叉积|, 点积)``，天然处理 ±180 的绕回。
    """
    i = geo.idx_at(tick)
    if i < 0:
        return None
    vx, vy = geo.xy(victim, i)
    ox, oy = geo.xy(other, i)
    dx, dy = ox - vx, oy - vy
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return None                                   # 两个人重合：方向没有意义
    yaw = math.radians(geo.yaw(victim, i))
    fx, fy = math.cos(yaw), math.sin(yaw)
    return math.degrees(math.atan2(abs(fx * dy - fy * dx), fx * dx + fy * dy))


# ═══════════════════════════════════════════════════════════════════════════
# 穿烟：AI 的遮挡矩阵
# ═══════════════════════════════════════════════════════════════════════════


class _Smoke:
    """「这条线有没有被烟挡住」—— 和 AI 遮挡矩阵**同一段代码**，只是只算一对人。

    `Occluder.matrix()` 一次算全场的 10×10（720 条射线），而这里一个回合最多十几
    场击杀、每场只关心**一对人**，所以直接调它底下那两个函数
    （`_Geometry` / `pair_matrix` / `smoke_volumes`）算 2×2：几何和判定内核一模
    一样，射线数从 720 降到 16（实测 45 ms/帧 → 1 ms 量级）。

    只为「穿烟击杀」服务，所以走**懒 import**：`scripts.training_data` 的
    `__init__` 会把 torch / numba / 抽取栈一起拖进来（那个包自己的注释写了这件事），
    而传统统计本来一个模型依赖都不需要。建不起来（没地图 / 没 numba）时调用方把
    `smoke_ok` 设成 False，前端那一列显示 "—"。
    """

    def __init__(self, data: dict) -> None:
        from scripts.training_data.obstruction import (
            _Geometry,
            pair_matrix,
            smoke_volumes,
        )
        from scripts.training_data.view_render import load_map

        map_name = str((data.get("meta") or {}).get("map") or "")
        if not map_name:
            raise ValueError("这份数据没有地图名，穿烟判定需要地图几何")
        self._geom_cls = _Geometry
        self._pair = pair_matrix
        self._smoke_volumes = smoke_volumes
        self._load_map = load_map
        self._map_name = map_name
        self._mesh = None
        self._rounds: dict[int, tuple[Any, dict]] = {}

    def _round(self, ri: int, rnd: dict):
        hit = self._rounds.get(ri)
        if hit is None:
            if self._mesh is None:
                self._mesh = self._load_map(self._map_name)
            geom = self._geom_cls(rnd, self._n_pl(rnd))
            hit = (geom, {})
            self._rounds[ri] = hit
        return hit

    @staticmethod
    def _n_pl(rnd: dict) -> int:
        from scripts.training_data.constants import N_PLAYERS

        return min(N_PLAYERS, len(rnd.get("players") or []))

    def blocked(self, ri: int, rnd: dict, a: int, v: int, idx: int) -> bool:
        """击杀者 `a` 到受害者 `v`（`idx` 那一帧）的连线是不是被**烟**挡住。

        判据就是遮挡矩阵的取值：`obstruction[a][v] == 1`（1 = 烟，2 = 墙，0 = 通）。
        用 2×2 的 `pair_matrix` 取 `[0][1]`，和 `Occluder.matrix()` 里同一格逐位相同
        —— 每一对人是独立算的。
        """
        if idx < 0:
            return False
        geom, cache = self._round(ri, rnd)
        ticks = rnd.get("ticks") or []
        if not (0 <= idx < len(ticks)):
            return False
        m = self._pair(
            geom.eyes(idx, [a, v]),
            geom.corners(idx, [a, v]),
            self._mesh,
            self._smoke_volumes(rnd, int(ticks[idx]), cache),
        )
        return int(m[0, 1]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 一回合的新指标
# ═══════════════════════════════════════════════════════════════════════════

#: `_round_extras()` 返回的数组名，顺序即前端表格的列序（见 main.js）。
EXTRA_KEYS: tuple[str, ...] = (
    "enemy_kills", "enemy_deaths",
    "first_kills", "first_deaths",
    "openers", "side_kills", "side_deaths", "smoke_kills",
)


def _blank_extras(n: int) -> dict[str, list[int]]:
    return {k: [0] * n for k in EXTRA_KEYS}


def _round_extras(
    rnd: dict, n: int, *, tickrate: float, smoke: Optional[_Smoke], round_idx: int
) -> dict[str, list[int]]:
    """一回合的五个新指标（按选手下标的定长数组）。

    `smoke` 为 None 时跳过穿烟判定（那一列保持 0，前端靠 `smoke_ok` 区分
    "没有"和"不可用"）。
    """
    out = _blank_extras(n)
    geo = _Geo(rnd)
    kills = duel_kills(rnd, n)
    if not kills:
        return out

    first = kills[0]
    out["first_kills"][first["a"]] = 1
    out["first_deaths"][first["v"]] = 1

    span = KILL_WINDOW_SEC * float(tickrate or 64.0)

    for k in kills:
        a, v, t = k["a"], k["v"], k["t"]
        out["enemy_kills"][a] += 1
        out["enemy_deaths"][v] += 1

        hits = _pair_damage(rnd, a, v, int(t - span), t)
        if hits:
            # 先手开枪：两人之间最早的那一枪（开枪没打中不算 —— 这里只有伤害事件）
            out["openers"][hits[0][1]] += 1
            # 侧背身：按**击杀者第一次打到受害者**的那一帧判受害者朝向
            first_hit = next((h for h in hits if h[1] == a), None)
            if first_hit is not None:
                ang = _side_back_angle(geo, v, a, first_hit[0])
                if ang is not None and ang >= SIDE_BACK_DEG:
                    out["side_kills"][a] += 1
                    out["side_deaths"][v] += 1

        if smoke is not None:
            # 穿烟：受害者死前最后一个活帧，击杀者到他的连线过不过烟
            i = geo.last_alive_idx(v, t)
            if i >= 0 and smoke.blocked(round_idx, rnd, a, v, i):
                out["smoke_kills"][a] += 1

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════


def build_stats(data: dict, *, smoke: bool = True) -> dict:
    """把一份 demo_extract 结果算成统计面板要的那几个数。

    返回结构（字段名就是前端用的名字）::

        {
          "map": "de_anubis", "format": "cs2.demo.v7", "rounds": 17,
          "players": [{idx, steamid, name, team, kills, deaths, assists,
                       headshots, damage, damage_taken, rounds,
                       kd, kpr, adr, hs_pct,
                       enemy_kills, enemy_deaths, duels,
                       first_kills, first_deaths,
                       openers, opener_pct,
                       side_kills, side_kill_pct,
                       side_deaths, side_death_pct,
                       smoke_kills, smoke_kill_pct}, ...],
          "teams":   [{idx, label, score, side0, players: [下标...]}, ...],
          "per_round": [{idx, id, winner, end_reason, duration,
                         kills: [...], deaths: [...],
                         damage: [...], damage_taken: [...],
                         enemy_kills: [...], enemy_deaths: [...],
                         first_kills: [...], first_deaths: [...],
                         openers: [...], side_kills: [...],
                         side_deaths: [...], smoke_kills: [...]}, ...],
          "smoke_ok": True,     # False = 穿烟判定不可用（无地图 / 无 numba）
          "ms": 1234.5,
        }

    ``per_round[*]`` 里那些数组都是**按选手下标对齐**的定长数组（长度 = 人数），
    所以前端拿它当矩阵画：行是选手、列是回合。

    ``damage`` / ``damage_taken``（以及每回合的同名数组）都是**按剩余血量截断之后**
    的值：单回合单人承伤 ≤ 100（一回合一条命），而且
    「所有人的 damage 之和」+「世界伤害」== 「所有人的 damage_taken 之和」
    —— 世界伤害只进后者，自伤两边都不进。ADR 也用它算。

    一个人一回合只有一条命（死了要到下一回合才复活），所以「一回合每条命最多掉 100」
    就是这条规则的全部内容，不需要按死亡重置血槽。注意别把「总和」当成
    `100 × 击杀数`：受伤但没死的人也计入承伤。

    派生公式和玩家页（``app/db.py`` 的 ``_PLAYER_VIEW``）一致，这样同一个人在回放页
    和玩家页看到的 K/D、KPR、HS% 不会是两套算法：K/D = K/D（D=0 时退化成 K）、
    KPR = K/回合数、ADR = 伤害/回合数、爆头率 = 爆头/击杀。分母用**整场比赛的回合数**
    （不是这个人出场的回合数），和库里一样。

    新指标的口径见模块 docstring；比例类字段（``*_pct``）是 0~100 的百分数，
    分母为 0 时是 0.0（不是 "—"：前端还有分子 / 分母可以显示成 "0/0"）。

    Args:
        smoke: 算不算「穿烟击杀」。关掉的话一个回合只要扫 events（毫秒级），
            测试和不需要那一列的调用方可以省掉建地图几何的几秒钟。
    """
    t_start = time.perf_counter()
    _warned = False
    roster = list(data.get("players") or [])
    rounds = list(data.get("rounds") or [])
    n = len(roster)
    r_total = len(rounds)
    tickrate = float((data.get("meta") or {}).get("tickrate") or 64.0)

    groups = team_groups(rounds, n) if n else []
    team_of = [0] * n
    for gi, group in enumerate(groups[:2]):
        for i in group:
            team_of[i] = gi

    kills = [0] * n
    deaths = [0] * n
    assists = [0] * n
    headshots = [0] * n
    dealt = [0] * n
    taken = [0] * n
    per_round: list[dict] = []

    # 新指标的累计（和每回合的数组是同一份数据，最后再拼比例）
    totals = _blank_extras(n)

    # 穿烟要建地图几何：地图不认识 / 没装 numba 就整块降级，不影响其余指标
    probe: Optional[_Smoke] = None
    smoke_ok = False
    if smoke and n and rounds:
        try:
            probe = _Smoke(data)
            smoke_ok = True
        except Exception:                          # noqa: BLE001 —— 见 docstring
            probe = None
            smoke_ok = False

    for ri, rnd in enumerate(rounds):
        events = rnd.get("events") or {}
        rk = [0] * n
        rd = [0] * n
        r_dealt = [0] * n
        r_taken = [0] * n
        # 每个选手这一回合的剩余血量。回合开场满血，所以每回合重置成 100；
        # 事件报的是"这一枪打掉多少"，不是"血槽还剩多少"，必须靠这个槽来截断。
        #
        # 一条血槽就够：标准 CS2 一个回合每人只有一条命，死了不会在同一回合复活
        # （复活发生在下一个回合，而这里的槽本来就是按回合重建的）。所以这个回合
        # 的承伤天然满足 0 ≤ 承伤 ≤ 100（没死的人也能挨打，只是掉不到 0）。
        slot = [100] * n

        for k in events.get("kills") or []:
            a, v, s = _idx(k.get("a")), _idx(k.get("v")), _idx(k.get("as"))
            if 0 <= a < n:
                kills[a] += 1
                rk[a] += 1
                if k.get("hs"):
                    headshots[a] += 1
            if 0 <= v < n:
                deaths[v] += 1
                rd[v] += 1
            # 自杀时 extract 已经把 as 写成 -1，这里不需要再排一次
            if 0 <= s < n:
                assists[s] += 1

        for d in events.get("damage") or []:
            a, v = _idx(d.get("a")), _idx(d.get("v"))
            hp = int(d.get("hp") or 0)
            if a == v:
                continue                      # 自伤：不算输出也不算承受
            # 伤害事件报的是**武器的名义伤害**，致命一击 / 穿墙时会远超血槽。
            # 按剩余血槽截断，记的才是「实际扣掉的血」（见模块 docstring）。
            # 受害者不在名单里（数据脏）时这条事件无处归属，整条丢掉 ——
            # 只记进"输出"会让「造成之和 == 承受之和」对不上。
            if not (0 <= v < n):
                continue
            hit_taken = min(hp, slot[v])
            slot[v] -= hit_taken
            taken[v] += hit_taken
            r_taken[v] += hit_taken
            if 0 <= a < n:
                dealt[a] += hit_taken         # 世界伤害（a == WORLD）不算任何人的输出
                r_dealt[a] += hit_taken

        try:
            extra = _round_extras(rnd, n, tickrate=tickrate,
                                  smoke=probe, round_idx=ri)
        except Exception as exc:               # noqa: BLE001 —— 见 docstring
            # 一个回合算不出来不该让整个面板 500（这五列是增量信息），但**必须**
            # 在服务端留一条痕：静默吞掉的话，一个形状不对的列会让所有新指标
            # 一直是 0，界面上看着跟"这局没人侧身"一模一样。
            if not _warned:
                _warned = True
                import traceback

                print(f"[stats] 回合 {ri} 的新指标计算失败，这一场的新列全部置 0："
                      f"{type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
            extra = _blank_extras(n)
            probe = None                      # 别让后面每个回合都再炸一次
            smoke_ok = False
        for key in EXTRA_KEYS:
            for i in range(n):
                totals[key][i] += extra[key][i]

        secs = rnd.get("round_seconds") or []
        per_round.append({
            "idx": ri,
            "id": rnd.get("id"),
            "winner": str(rnd.get("winner") or ""),
            "end_reason": str(rnd.get("end_reason") or ""),
            "duration": round(float(secs[-1]), 1) if len(secs) else 0.0,
            "kills": rk,
            "deaths": rd,
            "damage": r_dealt,
            "damage_taken": r_taken,
            **extra,
        })

    def pct(num: int, den: int) -> float:
        return round(num * 100.0 / den, 1) if den else 0.0

    players = []
    for i, p in enumerate(roster):
        k, d, a = kills[i], deaths[i], assists[i]
        ek, ed = totals["enemy_kills"][i], totals["enemy_deaths"][i]
        sk, sd = totals["side_kills"][i], totals["side_deaths"][i]
        smk = totals["smoke_kills"][i]
        players.append({
            "idx": i,
            "steamid": str(p.get("steamid") or ""),
            "name": str(p.get("name") or ""),
            "team": team_of[i],
            "kills": k,
            "deaths": d,
            "assists": a,
            "headshots": headshots[i],
            "damage": dealt[i],
            "damage_taken": taken[i],
            "rounds": r_total,
            "kd": round(k / d, 2) if d else float(k),
            "kpr": round(k / r_total, 2) if r_total else 0.0,
            "adr": round(dealt[i] / r_total, 1) if r_total else 0.0,
            "hs_pct": round(headshots[i] * 100.0 / k, 1) if k else 0.0,
            # ── 新指标（分母都给了，前端显示成 "7/14"）──────────────────
            "enemy_kills": ek,
            "enemy_deaths": ed,
            "duels": ek + ed,
            "first_kills": totals["first_kills"][i],
            "first_deaths": totals["first_deaths"][i],
            "openers": totals["openers"][i],
            "opener_pct": pct(totals["openers"][i], ek + ed),
            "side_kills": sk,
            "side_kill_pct": pct(sk, ek),
            "side_deaths": sd,
            "side_death_pct": pct(sd, ed),
            "smoke_kills": smk,
            "smoke_kill_pct": pct(smk, ek),
        })

    teams = []
    for gi, group in enumerate(groups[:2]):
        teams.append({
            "idx": gi,
            "label": "队伍 A" if gi == 0 else "队伍 B",
            "score": sum(1 for rnd in rounds if _winner_group(rnd, groups) == gi),
            "side0": _first_side(rounds, group),
            "players": list(group),
        })

    meta = data.get("meta") or {}
    return {
        "map": meta.get("map"),
        "format": data.get("format"),
        "rounds": r_total,
        "players": players,
        "teams": teams,
        "per_round": per_round,
        "smoke_ok": bool(smoke_ok),
        "ms": round((time.perf_counter() - t_start) * 1000.0, 1),
    }
