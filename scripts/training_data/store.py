"""第 1 层：`CszStore` —— 加载 .csz、按 round 缓存张量、数据体检。

这一层不关心你要什么特征，只负责"把原始数据变成能快速切片的东西"。
"""

from __future__ import annotations

import struct
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import _paths  # noqa: F401
from .constants import (
    FEATURES,
    MAX_PLAUSIBLE_INSTANCES,
    MIN_PLAUSIBLE_MEDIAN_SPAN,
    N_WEAPONS,
    REQUIRED_FORMAT,
    TEAM_TO_IDX,
    TEAM_UNKNOWN,
    WEAPON_TO_IDX,
    weapon_name_to_idx,
)

from demo_extract import codec, recipes, voxel  # noqa: E402


@dataclass(frozen=True)
class WindowRef:
    """一个窗口的定位信息。Dataset 的 `__getitem__` 就是把它换成张量。"""

    file_idx: int      # 在 CszStore.paths 里的下标
    round_id: int      # 回合 id（**不是**列表下标，跳过的回合会留空号）
    start: int         # 窗口在 round["ticks"] 里的起始采样下标
    stop: int          # 结束（不含）
    start_tick: int    # 对应的真实 tick，方便回查事件


def _invert_weapon_index(data: dict) -> list[str | None]:
    """`data["weapons"]`（名字 → 本文件下标）的逆表：下标 → 名字。

    ⚠ 每个 csz 只收录**该文件出现过的**武器并重新从 0 编号（实测 34/31/36/35 项），
    所以下标和名字的对应关系**每个文件都不一样**。
    """
    out: list[str | None] = [None] * len(data["weapons"])
    for name, i in data["weapons"].items():
        if 0 <= i < len(out):
            out[i] = name
    return out


def merge_split_instances(insts: list[dict], *, max_gap: int = 8) -> tuple[list[dict], int]:
    """把同一回合里**被切片器切碎**的 fire/smoke 实例合并回一条。

    返回 `(合并后的列表, 改写了多少个 change 的 mode)`。

    ## 为什么要合并

    抽取器写 side-channel 时，同一团火/烟偶尔会被拆成多条背靠背的实例，全部
    顶着**同一个 `entity_id`**，每条只覆盖 0–19 tick，中间留 2–3 tick 的缝。
    实测 4 个 demo、89 个回合，`fire + smoke` 一共 **5913 条实例，其实只有
    917 个真的实体**；最惨的一回合（inferno round 18）一个 eid 被切成 227 条。

    后果不是"多了几条"——`rnd["fire"]` 的区间之间有缝，`recipes.fire_state_at`
    在缝里返回 `None`，**约 18% 的帧火凭空消失**，而且不报任何错。

    ## 为什么不能直接拼 changes

    smoke 的碎片开头往往是一条 `mode="append"` 但 `data` 已经是**整份快照**的
    记录（判据：`len(data) == size`）。直接首尾相接会让 fold 变成 `buf + buf`，
    实测 journal 从 2591 字节涨到 46 万字节。

    所以这里的规则是：**遇到 `append` 且 `len(data) == size` 的记录，改成 `full`**
    （替换而不是追加）。改完之后有一条可验证的不变式：把 changes 依次 fold，
    **每一步 `len(buf)` 都必须等于该记录自带的 `size`**。`CszStore` 会真的验
    这一条（见 `round_health`），所以这不是"猜出来的修复"。

    ## 实测（4 个 demo / 89 回合，逐 eid 逐 tick 对比合并前后）

    * 实例数 **5913 → 917**
    * 缝里"修回"了 **5032 个 tick** 的 fire/smoke
    * **冲突 0**：凡是合并前能算出值的 tick，合并后逐字节/逐格完全一致
    * 不变式 **101733 / 101733** 全过

    `fire` 没有 mode，合并是纯拼接，天然无损。

    ## 参数

        max_gap: 两条实例的间隔小于等于它才认为是同一条被切开的。
                 实测所有碎片的间隔都在 2–3 tick（`fire`）或 2–4 tick（`smoke`），
                 而真正"另一颗弹复用同一个 eid"在回合内**一次都没出现**过
                 （跨回合才会复用）。8 已经很宽松；调大没有收益，只会扩大
                 误合并的风险。
    """
    by_eid: dict[int, list[dict]] = defaultdict(list)
    for inst in insts:
        by_eid[inst["entity_id"]].append(inst)

    out: list[dict] = []
    n_fixed = 0
    for _eid, group in by_eid.items():
        group.sort(key=lambda i: i["first_tick"])
        cur: dict | None = None
        for inst in group:
            changes: list[dict] = []
            for c in inst["changes"]:
                # fire 的 change 没有 mode/size（它存的是绝对格坐标，天然自洽）
                if (
                    "mode" in c
                    and c["mode"] == voxel.MODE_APPEND
                    and len(c["data"]) == c["size"]
                ):
                    c = {**c, "mode": voxel.MODE_FULL}
                    n_fixed += 1
                changes.append(c)

            if cur is not None and inst["first_tick"] - cur["last_tick"] <= max_gap:
                cur = {
                    **cur,
                    "last_tick": max(cur["last_tick"], inst["last_tick"]),
                    "changes": cur["changes"] + changes,
                }
            else:
                if cur is not None:
                    out.append(cur)
                cur = {**inst, "changes": changes}
        if cur is not None:
            out.append(cur)

    for inst in out:
        inst["changes"].sort(key=lambda c: c["tick"])
    out.sort(key=lambda i: i["first_tick"])
    return out, n_fixed


def _journal_issue(insts: list[dict]) -> str | None:
    """smoke 的 journal 自检：fold 每一步的 `len(buf)` 都得等于记录自带的 `size`。

    这条不变式在**合并前**就成立（每条碎片自己都是自洽的），所以它不是用来发现
    切碎的，而是用来确认**合并没把事情弄坏**——一个"拼错了但长度看着对"的结果
    会在这里现形。返回 `None` 表示没问题。
    """
    for inst in insts:
        buf = b""
        for rec in inst["changes"]:
            if "mode" not in rec:
                continue
            buf = (
                buf + rec["data"]
                if rec["mode"] == voxel.MODE_APPEND
                else bytes(rec["data"])
            )
            if len(buf) != rec["size"]:
                return (
                    f"smoke eid={inst['entity_id']} tick={rec['tick']}: "
                    f"journal fold 后 {len(buf)} 字节，记录自称 {rec['size']} —— 合并出错"
                )
    return None


#: `peek_meta()` 默认只读文件头这么多字节。实测 256 KB 压缩数据能解出 ~1.3 MB 原文，
#: 而 `format` + `meta` 加起来只有 ~1.5 KB —— 留了几百倍余量。
#: 调到更小也能跑，但没必要：读 256 KB 和读 16 KB 都是一次 read syscall 的量级。
PEEK_BYTES = 256 * 1024


def peek_meta(path: str | Path, *, max_read: int = PEEK_BYTES) -> dict | None:
    """只解 csz 的**文件头**，拿 `format` + `meta`（地图名就在 `meta["map"]` 里）。

    为什么需要它
    ------------
    `codec.load()` 一个 9 MB 的 csz 要 **11–15 s**，因为它把整个对象图解出来
    （几百万个 Python float，实测常驻 ×96）。但顶层 dict 的**前两个 key 就是
    `format` 和 `meta`**（`demo_extract/extract.py` 末尾的 `result`），而整份 payload
    是一个 zstd 帧 —— 所以只要解压开头那段就够了：实测 **6 ms**，比全量 load 快
    两千倍，内存只有几百 KB。

    用途：`prepare.py` 要按地图平衡抽样、还要在开跑前打印"每张图几个 csz 参与计算"，
    这些都只要地图名。为此把每个 csz 全量 load 一遍等于**把 30 小时变成 60 小时**，
    所以必须走这条路。

    只认前两个 key
    --------------
    按顺序解顶层 dict 的 key，拿到 `meta` 就停 —— 后面的 `players` / `rounds` 一律不碰。
    所以它依赖"`format`/`meta` 排在最前面"这个事实。key 顺序真变了的话，它会返回
    `None`（而不是解出半份脏数据），调用方自己兜底。

    Returns:
        `{"format": ..., "meta": {...}}`（至少含 `meta`）；解不出来返回 `None` ——
        文件不是 CSZ5、容器版本不对、开头不是 dict、前缀不够长、没装 `zstandard`……
        **别假定它一定成功**，`None` 是正常返回值的一种。
    """
    try:
        with open(Path(path), "rb") as f:
            head = f.read(max(int(max_read), 9))
    except OSError:
        return None

    if len(head) < 9 or not head.startswith(codec.MAGIC) or head[4] != codec.VERSION:
        return None
    (raw_len,) = struct.unpack("<I", head[5:9])
    body = head[9:]
    if raw_len and codec.zstd is not None:
        try:
            # 只喂前缀：zstd 的流式接口接受"帧还没结束"的输入，返回已经能解出来的那部分。
            body = codec.zstd.ZstdDecompressor().decompressobj().decompress(body)
        except Exception:
            return None

    try:
        r = codec._Reader(body)
        if r.u8() != codec.T_DICT:
            return None
        out: dict[str, Any] = {}
        for _ in range(r.uvarint()):
            key = r.take(r.uvarint()).decode("utf-8")
            out[key] = codec._dec(r)   # 前缀不够长会抛 ValueError → 下面返回 None
            if key == "meta":
                break
        return out if "meta" in out else None
    except Exception:
        # 前缀里没有 meta（key 顺序变了 / 前缀太短）、或者文件本身坏了。
        # 这里刻意宽：它只是个"探一下"的函数，答案就是"探不到"，不该把调用方炸掉。
        return None


class CszStore:
    """一组 .csz + 按 round 的惰性张量缓存。

    为什么要这一层（实测）：

    * `codec.load()` 一个 11 MB 的 csz 要 **11–15 秒**——瓶颈是把 ~240 万个
      float 解成 Python `float` 对象，不是解压。所以**一个进程只 load 一次**。
    * load 之后，把一个回合转成 `(T,10,20)` float32 只要 ~50 ms，所以缓存
      round 张量很划算。实测量级（4 个 demo 共 89 回合）：一个回合 14 个 key
      合计 **平均 9.2 MB**，全部 **817 MB**。`players` 一个 key 占一半左右。

    加载时还会把被切片器切碎的 fire/smoke 合并回一条（`merge_splits=True`），
    实测 5913 条碎片 -> 917 条真实例，逐 tick 对比零冲突。见
    `merge_split_instances`。

    Args:
        paths: .csz 路径列表。
        max_cached_rounds: 最多缓存多少个回合（LRU）。单回合约 5 MB，默认 8。
        interp_nan: 是否插值速度列的 NaN（**建议 True**，见 README §5.6）。
        merge_splits: 加载时是否把切碎的 fire/smoke 合并回去（**建议 True**，
            见 `merge_split_instances`）。关掉的话你会拿到原始碎片。
        merge_gap: 判定"同一条被切开"的最大 tick 间隔。
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        max_cached_rounds: int = 8,
        interp_nan: bool = True,
        merge_splits: bool = True,
        merge_gap: int = 8,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("CszStore 需要至少一个 .csz 路径")
        self.max_cached_rounds = max_cached_rounds
        self.interp_nan = interp_nan
        self.merge_splits = merge_splits
        self.merge_gap = merge_gap

        self._data: dict[int, dict] = {}
        self._round_cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] = OrderedDict()
        self._health: dict[tuple[int, int], str | None] = {}
        self._merge_stats: dict[int, dict[str, int]] = {}
        self._merge_bad: dict[tuple[int, int], str] = {}
        #: `map_name_fast()` 的缓存（每个文件一个字符串，几 KB 的事，`clear_cache()` 不清它）
        self._maps: dict[int, str] = {}

    @classmethod
    def from_dir(cls, csz_dir: str | Path, **kw: Any) -> "CszStore":
        """把一个目录下所有 `*.csz` 收进来。"""
        d = Path(csz_dir)
        paths = sorted(d.glob("*.csz"))
        if not paths:
            raise FileNotFoundError(f"{d} 下没有 *.csz")
        return cls(paths, **kw)

    # ── 加载 ───────────────────────────────────────────────────────────────

    def data(self, file_idx: int) -> dict:
        """整个文件的原始 dict（惰性加载，之后常驻）。

        加载时校验格式版本：不是 v7 直接抛错。**不静默降级**——
        老格式缺 on_ground/jump/land/crouch，补 0 会让那几列变成常量，
        训练完才发现，很难查。缺列就该重新抽一次 demo。

        加载后还会做一件事：把切碎的 fire/smoke 合并回去
        （`merge_splits=True` 时，见 `merge_split_instances`）。这是**唯一**一处
        规范化，所以 `round()` / `round_tensors()` / 任何直接拿 `rnd["fire"]`
        的下游都不可能拿到碎片版本。
        """
        if file_idx not in self._data:
            d = codec.load(self.paths[file_idx])
            fmt = d.get("format")
            if fmt != REQUIRED_FORMAT:
                raise ValueError(
                    f"{self.paths[file_idx].name}: 格式是 {fmt!r}，需要 {REQUIRED_FORMAT!r}。\n"
                    f"  {REQUIRED_FORMAT} 才有 on_ground / jump / land / crouch 这几列。\n"
                    f"  老文件重新抽一遍即可：\n"
                    f"    python -m demo_extract <match.dem> -o <match.csz>"
                )
            if self.merge_splits:
                self._merge_file(file_idx, d)
            self._data[file_idx] = d
        return self._data[file_idx]

    # ── 切碎的 fire/smoke 合并 ─────────────────────────────────────────────

    def _merge_file(self, file_idx: int, d: dict) -> None:
        before = after = fixed = 0
        for rnd in d.get("rounds", []):
            key = (file_idx, rnd["id"])
            for kind in ("fire", "smoke"):
                insts = rnd.get(kind)
                if not insts:
                    continue
                merged, n_fixed = merge_split_instances(insts, max_gap=self.merge_gap)
                before += len(insts)
                after += len(merged)
                fixed += n_fixed
                rnd[kind] = merged
            # 合并完顺手验一遍不变式；坏了就把这个回合标成不健康，
            # 而不是让脏数据继续往下走。
            issue = _journal_issue(rnd.get("smoke") or [])
            if issue is not None:
                self._merge_bad[key] = issue
        self._merge_stats[file_idx] = {
            "rounds": len(d.get("rounds", [])),
            "before": before,
            "after": after,
            "fixed": fixed,
        }

    def merge_stats(self) -> dict[str, int]:
        """合并了多少东西 —— 给你确认这一步真的发生了。

            {'rounds': 89, 'before': 5913, 'after': 917, 'fixed': 3847,
             'bad_rounds': 0}

        `before -> after` 掉得多是**正常**的（实测最惨的一回合 930 -> 4），
        因为碎片本来就不是真的实例。`bad_rounds` 才是要盯的：它非 0 说明有
        回合合并后自检没过，那几回合会被 `round_health()` 标成不合格。
        注意只统计**已经 load 过**的文件。
        """
        return {
            "rounds": sum(s["rounds"] for s in self._merge_stats.values()),
            "before": sum(s["before"] for s in self._merge_stats.values()),
            "after": sum(s["after"] for s in self._merge_stats.values()),
            "fixed": sum(s["fixed"] for s in self._merge_stats.values()),
            "bad_rounds": len(self._merge_bad),
        }

    def round(self, file_idx: int, round_id: int) -> dict:
        """某回合的原始 dict。

        回合可能因为 `no round_freeze_end` 被抽取器跳过，那样它**不在文件里**
        （不是空回合）。用 `round_ids()` 查有哪些。
        """
        return recipes.round_by_id(self.data(file_idx), round_id)

    def round_ids(self, file_idx: int) -> list[int]:
        return recipes.round_ids(self.data(file_idx))

    def meta(self, file_idx: int) -> dict:
        return self.data(file_idx)["meta"]

    def map_name(self, file_idx: int) -> str:
        """地图名，如 `'de_mirage'`。**原样返回** demo 里写的字符串，不做归一。

        所以 `de_Ancient` 就是 `de_Ancient`，`MAP_TO_IDX` / `MAP_CENTER` 里没有它 ——
        `aim` 任务查不到 `map_id` 就整回合跳过，`prepare.py` 在抽样阶段就把这种 csz
        整个筛掉了。
        """
        return str(self.meta(file_idx)["map"])

    def map_name_fast(self, file_idx: int) -> str:
        """地图名，但**不 load 整个文件**（`peek_meta()`，实测 6 ms vs 11–15 s）。

        结果和 `map_name()` **完全一样**（都是 `meta["map"]` 原样），只是走文件头快照。
        给 `prepare.py` 的按地图平衡抽样 / 开跑前的"每张图几个 csz"统计用 —— 那里
        一个 csz 都还没 load，为了拿地图名去 load 一遍等于把整个任务的时间翻倍。

        兜底：`peek_meta()` 返回 `None`（key 顺序变了、前缀不够长……）就退回
        `map_name()` **全量 load 一次**，然后 `clear_cache()` 把那 ~1.7 GB 丢掉 ——
        调用方多半是父进程，那里本来不留任何文件。这条兜底正常永远走不到
        （4 个真实文件实测全走快路），但"探不到就报错"比"悄悄少一个地图"好。

        每个文件只探一次，结果缓存在 `_maps` 里（`clear_cache()` 不清它）。
        """
        cached = self._maps.get(file_idx)
        if cached is None:
            meta = peek_meta(self.paths[file_idx])
            map_meta = (meta or {}).get("meta") or {}
            if map_meta.get("map") is not None:
                cached = str(map_meta["map"])
            else:                                    # 快路失败 → 全量 load 兜底
                cached = self.map_name(file_idx)
                self.clear_cache()
            self._maps[file_idx] = cached
        return cached

    # ── 体检 ───────────────────────────────────────────────────────────────

    def round_health(self, file_idx: int, round_id: int) -> str | None:
        """None = 这个回合的 fire/smoke 可信；否则返回不合格原因。

        体检是在**合并之后**做的（`data()` 里先合并），所以实测那三个回合
        （inferno 18/19/20）现在是健康的——它们的碎片已经被拼回一条完整的
        inferno（930 -> 4 条，span 回到 ~1281）。这里剩下的是**兜底**：

        1. **合并自检失败**。`merge_split_instances` 有一条可验证的不变式
           （fold 每一步 `len(buf)` 都得等于记录自带的 `size`）。自检不过说明
           合并没修好，那就不能信。
        2. **形状仍然不对**。碎片之间的缝如果超过 `merge_gap`（默认 8 tick）
           就拼不起来，这时下面的两条老判据会兜住它：
             * 实例数 > 64 —— 正常回合最多 18 个 inferno / 17 个 smoke
             * span 中位数 < 100 tick —— 正常是 ~1285 (火) / ~1519 (烟)

        为什么要留这两条：碎片化不是"要么全好要么全坏"，它是个连续谱。实测
        inferno 那个文件 23 个回合里有 12 个被切过，最轻的只切了一刀。合并把
        全谱都修了，但**修不了**的情况必须还有人喊。

        玩家状态（位置/视角/速度/血量）**不受影响**，坏的只有 fire/smoke。
        所以"要不要因为这条丢掉整个回合"是**调用方的策略**，不是 store 的策略 ——
        这里只回答"健不健康"，不替你决定：`prepare.py` 在写盘前问一次
        （`if store.round_health(fi, rid) is not None: 跳过`），`selftest.py` 只是列出来。
        不做空间特征的话，随便你怎么处理。
        """
        key = (file_idx, round_id)
        if key in self._health:
            return self._health[key]

        # 先 round()，它会把文件 load + 合并一遍，_merge_bad 才有内容
        rnd = self.round(file_idx, round_id)

        why: str | None = self._merge_bad.get(key)
        for kind in ("fire", "smoke"):
            if why is not None:
                break
            insts = rnd.get(kind) or []
            if len(insts) > MAX_PLAUSIBLE_INSTANCES:
                why = (f"{kind}: {len(insts)} 个实例"
                       f"（正常 ≤ 18 火 / 17 烟，阈值 {MAX_PLAUSIBLE_INSTANCES}）")
                break
            spans = [i["last_tick"] - i["first_tick"] for i in insts]
            if spans and float(np.median(spans)) < MIN_PLAUSIBLE_MEDIAN_SPAN:
                why = (f"{kind}: span 中位数 {np.median(spans):.0f} tick"
                       f"（正常 ~1285 火 / ~1519 烟，阈值 {MIN_PLAUSIBLE_MEDIAN_SPAN}）")
                break
        self._health[key] = why
        return why

    # ── round 张量缓存 ─────────────────────────────────────────────────────

    def round_tensors(self, file_idx: int, round_id: int) -> dict[str, np.ndarray]:
        """把一个回合预计算成标准张量，按 LRU 缓存。

        `features.build_window()` 拿到的就是它——**在这里一次算好、反复切片**，
        比在窗口里反复调 `recipes.*` 快得多（那些函数每次都重建 (T,10,F)）。

        返回的 key（格式已在 `data()` 里校验为 v7，所以**全都有**）：

            players    (T,10,20) float32  FEATURES 的列，NaN 已按 interp_nan 处理
            alive      (T,10)    bool
            yaw_sincos (T,10,2)  float32  sin/cos(yaw) —— 别直接喂 yaw 角度
            seconds    (T,)      float64  回合内秒数
            ticks      (T,)      int64
            teams      (10,)     int8     0=CT 1=T -1=未知
            inventory  (T,10,N_WEAPONS) bool  已映射到**全局**武器词表
            spotted    (T,10,10) bool     [t,i,j] = "j 能看见 i"（注意方向！）
            on_ground  (T,10)    bool
            jump       (T,10)    int64    起跳次数
            land       (T,10)    int64    落地次数
            crouch     (T,10)    float32  0..1 连续（相机斜坡）
            crouching  (T,10)    bool     FL_DUCKING，碰撞盒已收到 54（阶跃）
            eye_z      (T,10)    float32  眼高（相对脚，HU）
        """
        key = (file_idx, round_id)
        cached = self._round_cache.get(key)
        if cached is not None:
            self._round_cache.move_to_end(key)
            return cached

        rnd = self.round(file_idx, round_id)
        data = self.data(file_idx)
        nan_mode = "interp" if self.interp_nan else "keep"

        out: dict[str, np.ndarray] = {
            "players": recipes.player_tensor(rnd, FEATURES, nan=nan_mode),
            "alive": recipes.alive_mask(rnd),
            "yaw_sincos": recipes.yaw_sincos(rnd),
            "seconds": recipes.seconds_of(rnd),
            "ticks": recipes.ticks_of(rnd),
            "teams": np.asarray(
                [TEAM_TO_IDX.get(t, TEAM_UNKNOWN) for t in rnd.get("teams", [])],
                dtype=np.int8,
            ),
            "inventory": self._inventory_global(data, rnd),
            "spotted": recipes.spotted_multihot(rnd),
            # v7 的地面/下蹲列。格式已校验过，直接取，不做存在性判断——
            # 缺列应该在上面就报错，而不是在这里悄悄少几个 key。
            "on_ground": recipes.on_ground_mask(rnd),
            "jump": recipes.jump_mask(rnd),
            "land": recipes.land_mask(rnd),
            "crouch": recipes.crouch_amount(rnd),
            "crouching": recipes.crouching_mask(rnd),
            "eye_z": recipes.eye_height(rnd),
        }

        self._round_cache[key] = out
        self._round_cache.move_to_end(key)
        while len(self._round_cache) > self.max_cached_rounds:
            self._round_cache.popitem(last=False)
        return out

    # ── 武器：本文件下标 ↔ 全局下标 ────────────────────────────────────────
    #
    # 为什么需要这一组：`round_tensors()["players"][:, :, COL["weapon"]]` 存的是
    # **本文件下标**（`recipes.player_tensor` 直接抄原始编号）。实测同一个下标
    # 12 在 4 个文件里分别是 High Explosive Grenade / Huntsman Knife /
    # Glock-18 / Incendiary Grenade —— 跨 demo 直接用是错的。
    #
    # `inventory` 不用管这个：它已经在 round_tensors() 里转成全局多热了。

    def weapon_names(self, file_idx: int) -> list[str | None]:
        """本文件的武器下标 → 名字。**每个文件的表都不一样**，别跨文件缓存。"""
        return _invert_weapon_index(self.data(file_idx))

    def weapon_lut(self, file_idx: int) -> np.ndarray:
        """本文件武器下标 → **全局**武器下标（`constants.WEAPON_NAMES`，43 项）。

        用它把 `players[:, :, COL["weapon"]]` 转成跨 demo 可比的下标：

            lut = store.weapon_lut(ref.file_idx)      # len = 本文件武器数
            w = tensors["players"][:, :, COL["weapon"]].astype(np.int64)
            valid = w >= 0                            # -1 = 未知/无武器
            glob = np.full(w.shape, -1, dtype=np.int16)
            glob[valid] = lut[w[valid]]

        ⚠ **别写 `lut[w]`。** `w` 里有 `-1`（实测占 10% 左右的 tick，死人/无武器），
        拿 -1 去索引 numpy 会取到**最后一个元素**（Python 负索引语义），
        静默给你一个错武器。必须先按 `w >= 0` 掩掉。

        ⚠ 不在全局表里的名字一律落到 `Knife`（下标 36）。实测 4 个文件里这种名字
        有 18 个，**全是刀的变体**（Butterfly / Flip / Karambit / M9 Bayonet /
        Huntsman / Skeleton / Stiletto / Talon / Ursus，以及 `knife_flip` 这类
        内部名），所以这个兜底正是想要的。想核对有没有意外落表的，用
        `unmapped_weapons()`。
        """
        names = self.weapon_names(file_idx)
        return np.array(
            [weapon_name_to_idx(n) if n else -1 for n in names], dtype=np.int16
        )

    def unmapped_weapons(self, file_idx: int) -> list[str]:
        """本文件里**不在全局表**里的武器名 —— 这些会被算成 `Knife`。

        给你核对用。实测 4 个文件的结果全是刀；如果哪天这里冒出非刀的名字
        （比如 `World`，它确实在 `data["weapons"]` 里但从不进 inventory），
        说明全局表缺条目，该去 `constants.WEAPON_NAMES` 补。
        """
        return [n for n in self.weapon_names(file_idx) if n and n not in WEAPON_TO_IDX]

    def _inventory_global(self, data: dict, rnd: dict) -> np.ndarray:
        """inventory 多热，下标是**全局**武器词表。

        `recipes.inventory_multihot(rnd, n_items=len(data["weapons"]))` 给的是
        本文件编号，跨 demo 不可比。这里先把 本文件下标 → 武器名 → 全局下标。
        """
        local_names = _invert_weapon_index(data)

        T = len(rnd["ticks"])
        out = np.zeros((T, len(rnd["players"]), N_WEAPONS), dtype=bool)
        for p, rec in enumerate(rnd["players"]):
            for t, items in enumerate(rec.get("inventory", [])):
                for local_i in items:
                    if 0 <= local_i < len(local_names) and local_names[local_i]:
                        out[t, p, weapon_name_to_idx(local_names[local_i])] = True
        return out

    def clear_cache(self) -> None:
        """放掉已加载的文件和 round 缓存（长跑训练里手动控内存用）。

        `_maps`（`map_name_fast()` 缓存的每文件地图名）**不清**：几 KB 的字符串，
        和这里要放掉的几百 MB 不是一回事。
        """
        self._data.clear()
        self._round_cache.clear()
        self._health.clear()
        self._merge_stats.clear()
        self._merge_bad.clear()

    # ── Windows / spawn 下的序列化 ─────────────────────────────────────────
    #
    # ⚠ 这条在 Windows 上是必需的，不是优化。
    # Windows 的 DataLoader 用 spawn 起 worker，会**把 dataset 整个 pickle 一遍**
    # 发给每个子进程。如果主进程已经 load 过 csz，`_data` 里那几百万个 Python
    # float 会被序列化 N 次（每个 worker 一次），启动几十秒甚至 OOM。
    # 这里把缓存从序列化状态里摘掉：worker 起来后按需重新 load（各付一次
    # ~12 s），但至少不会重复传输。

    def __getstate__(self) -> dict:
        st = self.__dict__.copy()
        st["_data"] = {}
        st["_round_cache"] = OrderedDict()
        st["_health"] = {}
        # 这两个也跟着 `_data` 走：它们描述的是"本进程 load 过什么"，
        # 留着会和新进程重新 load 的结果对不上（见 merge_stats 的 docstring）。
        st["_merge_stats"] = {}
        st["_merge_bad"] = {}
        return st

    def __setstate__(self, st: dict) -> None:
        self.__dict__.update(st)
