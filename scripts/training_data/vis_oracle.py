"""`vis_oracle.mjs` 的 Python 客户端：**把一个局面直接问成「A 能不能看见 B」**。

    from scripts.training_data.vis_oracle import VisOracle, player_state

    with VisOracle("de_mirage") as o:                  # 加载地图 ~1 s，进程只做一次
        o.can_see(a, b)                                # a/b 用 player_state(...) 造
        o.detail(a, b)["corners"]                      # 想要每个角落的明细
        o.load_round(store, fi, rid)                   # 或直接把 csz 某个回合灌进去
        o.matrix(tick=1234)                            # 那一帧整张 10×10

为什么要常驻进程：一次判定的成本是 **几何 ~8 µs/对** + **加载地图 ~1 s**。前者躲不掉，
后者必须摊掉 —— 所以这里起一个 node 进程一直留着，见 `js/vis_oracle.mjs`。

**训练侧完全不碰它** —— 它只在**造数据**（`write_round`）的时候用：把要用的可见性抄进样本，
训练照旧只认样本文件（`read_sample` 不许 import `..store`）。见 README §3.9。

⚠ 三个字段/语义坑（客户端已经封住，但你得知道）：
  1. **观察者**的眼高看 `eye_height`，没有就按 **连续** 的 `crouch` ∈ [0,1] 插值；
     **目标**的盒子高度看 `crouching`（**布尔**）→ 72/54 HU **阶跃**。字段名不对称，
     所以用 `player_state()` 一次填齐，别手写 dict。
  2. `pairVisibility()` **自己不查 `alive`** —— 死人照算。本客户端补上：任一方死了直接
     `False`（`detail()` 里给 `dead=True`）。
  3. `can_see(a, b)` 是「**a 看得见 b**」，方向别和 `.csz` 的 `spotted[目标] ∋ 观察者` 搞混。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import _paths

__all__ = [
    "VisOracle", "player_state", "algo_ver", "map_obj_for",
    "ORACLE_JS", "GOLDEN", "MAPS_DIR",
]

ORACLE_JS = Path(__file__).resolve().parent / "js" / "vis_oracle.mjs"
GOLDEN = Path(__file__).resolve().parent / "oracle_golden.npz"
MAPS_DIR = _paths.REPO_ROOT / "maps" / "optimized_obj_files"
JS_DIR = _paths.REPO_ROOT / "visualizer" / "static" / "js"

#: 决定判定的 JS 文件。任何一个改了，冻结快照就该重做（`--freeze`）。
JS_FILES = ("replay-core.js", "visibility.js", "map-raycast.js", "smoke-voxels.js", "scene.js")


def _sha1(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def algo_ver(map_obj: Path) -> str:
    """`js:xxxx obj:yyyy` —— 记进样本/快照，几何一改就能一眼看出来。

    （这个函数原来在 `vis_prepare.py` 里、和离线表一起；表删了，它搬到这里。）
    """
    return f"js:{_sha1([JS_DIR / f for f in JS_FILES])} obj:{_sha1([map_obj])}"


def map_obj_for(map_name: str, maps_dir: str | Path | None = None) -> Path:
    """地图名 → `maps/optimized_obj_files/<名>.obj`。找不到就把有哪些地图列出来。"""
    d = Path(maps_dir) if maps_dir else MAPS_DIR
    p = d / f"{map_name}.obj"
    if not p.is_file():
        have = sorted(q.stem for q in d.glob("*.obj"))
        raise FileNotFoundError(
            f"没有地图几何 {p}\n（csz 里的 meta['map'] = {map_name!r}；{d} 下有 {have}）"
        )
    return p


def player_state(
    x: float,
    y: float,
    z: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
    *,
    alive: bool = True,
    crouch: float = 0.0,
    crouching: bool | None = None,
    eye_height: float | None = None,
) -> dict[str, Any]:
    """一个玩家在某一瞬的状态（HU，game 空间）。**蹲的三个字段一次填齐**。

    Args:
        x, y, z: **脚底**坐标 —— `.csz` 的 x/y/z 就是脚底，不是眼睛（眼睛 = z + eye_height）。
        yaw, pitch: 度。yaw 0 → +X；**pitch > 0 是向下看**（Source 约定，和 `visuals.js` 一致）。
        alive: 死人直接判不可见。
        crouch: 蹲下量 **0..1 连续**，只影响**观察者**的眼高（64.0938 − 18·crouch）。
        crouching: **布尔**，只影响**目标/遮挡物**的碰撞盒高度（72 / 54 HU 阶跃）。
            不传就按 `crouch > 0.5` 猜 —— 这是猜的，确定知道就显式给。
        eye_height: 显式眼高（相对脚底，HU）。给了就优先于 `crouch`。

    Returns:
        node 那边 `pairVisibility()` 直接吃的 dict。
    """
    st: dict[str, Any] = {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "alive": bool(alive),
        "crouch": float(crouch),
        "crouching": bool(crouch > 0.5) if crouching is None else bool(crouching),
    }
    if eye_height is not None:
        st["eye_height"] = float(eye_height)
    return st


class VisOracle:
    """常驻 node 进程 + 协议封装。`with` 用它，或者自己 `close()`。"""

    def __init__(
        self,
        map_name: str,
        *,
        maps_dir: str | Path | None = None,
        node: str | None = None,
        solid: bool = True,
        fov_deg: float | None = None,
        aspect: float | None = None,
        inset: float | None = None,
        exact: bool = False,
        block_players: bool = False,
        timeout: float = 900.0,
        quiet: bool = False,
    ) -> None:
        self.map_name = str(map_name)
        self.obj = self._resolve_obj(self.map_name, maps_dir)
        self.solid = bool(solid)
        self.timeout = float(timeout)

        exe = node or shutil.which("node")
        if not exe:
            raise RuntimeError(
                "找不到 node。装一个，或者 VisOracle(..., node=r'C:\\...\\node.exe') / "
                "给 `python -m scripts.training_data.vis_oracle --node <路径>`。"
            )
        cmd = [str(exe), str(ORACLE_JS), "--obj", str(self.obj)]
        if not self.solid:
            cmd.append("--boundary")
        if fov_deg is not None:
            cmd += ["--fov", str(fov_deg)]
        if aspect is not None:
            cmd += ["--aspect", str(aspect)]
        if inset is not None:
            cmd += ["--inset", str(inset)]
        if exact:
            cmd.append("--exact")
        if block_players:
            cmd.append("--block-players")

        self._cmd = cmd
        self._id = 0
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if quiet else None,   # 地图加载那行日志值得看见
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.info = self._rpc({"op": "ping"})       # 顺便确认它真的起来了

    # ── 生命周期 ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_obj(map_name: str, maps_dir: str | Path | None) -> Path:
        return map_obj_for(map_name, maps_dir)

    def close(self) -> None:
        p = getattr(self, "_proc", None)
        if p is None or p.poll() is not None:
            return
        try:
            p.stdin.write(json.dumps({"id": -1, "op": "quit"}) + "\n")
            p.stdin.flush()
            p.wait(timeout=5)
        except Exception:                                    # noqa: BLE001
            p.kill()
        finally:
            for s in (p.stdin, p.stdout):
                try:
                    s.close()
                except Exception:                            # noqa: BLE001
                    pass

    def __enter__(self) -> VisOracle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:                               # 兜底
        try:
            self.close()
        except Exception:                                    # noqa: BLE001
            pass

    def _rpc(self, req: dict[str, Any]) -> dict[str, Any]:
        if self._proc.poll() is not None:
            raise RuntimeError(f"node 已经退出了（rc={self._proc.returncode}）")
        self._id += 1
        req = {**req, "id": self._id}
        try:
            self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError(f"node 挂了（管道断了）：{e}") from e

        t0 = time.perf_counter()
        while True:
            if time.perf_counter() - t0 > self.timeout:
                raise TimeoutError(f"等 node 超过 {self.timeout}s（请求 {req['op']}）")
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"node 没回话就退出了（rc={self._proc.poll()}），请求 op={req['op']}"
                )
            msg = json.loads(line)
            if msg.get("id") != self._id:
                continue                      # 只可能是我自己发的请求，容错跳过
            if "error" in msg:
                raise RuntimeError(f"oracle（op={req['op']}）：{msg['error']}")
            return msg

    # ── 问 ──────────────────────────────────────────────────────────────────

    def can_see(
        self,
        a: dict[str, Any] | int,
        b: dict[str, Any] | int,
        *,
        tick: int | None = None,
        smoke: Sequence[dict[str, Any]] | None = None,
        smoke_key: str | None = None,
        others: Sequence[dict[str, Any]] | None = None,
        opts: dict[str, Any] | None = None,
    ) -> bool:
        """**a 看得见 b 吗。** `a`/`b` 是 `player_state(...)`，或者灌过回合后的整数下标。"""
        return bool(self.detail(a, b, tick=tick, smoke=smoke, smoke_key=smoke_key,
                                others=others, opts=opts)["visible"])

    def detail(
        self,
        a: dict[str, Any] | int,
        b: dict[str, Any] | int,
        *,
        tick: int | None = None,
        smoke: Sequence[dict[str, Any]] | None = None,
        smoke_key: str | None = None,
        others: Sequence[dict[str, Any]] | None = None,
        opts: dict[str, Any] | None = None,
        corners: bool = False,
    ) -> dict[str, Any]:
        """同上，但把 `clear / reason / occluders / corners` 一起给回来。

        `reason` 是位或：1 FOV / 2 地图 / 4 烟 / 8 玩家（见 `visibility.js` 的 `BLOCK_*`）。
        """
        o = dict(opts or {})
        if corners:
            o["corners"] = True
        req = {"op": "pair", "a": a, "b": b, **self._opt_keys(tick, smoke, smoke_key, others, o)}
        return self._rpc(req)

    @staticmethod
    def _opt_keys(
        tick: int | None,
        smoke: Sequence[dict[str, Any]] | None,
        smoke_key: str | None,
        others: Sequence[dict[str, Any]] | None,
        opts: dict[str, Any] | None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {}
        if tick is not None:
            req["tick"] = int(tick)
        if smoke is not None:
            req["smoke"] = list(smoke)
            req["smoke_key"] = smoke_key if smoke_key is not None else f"len{len(smoke)}"
        if others is not None:
            req["others"] = list(others)
        if opts:
            req["opts"] = dict(opts)
        return req

    def can_see_many(
        self,
        pairs: Sequence[Sequence[Any]],
        *,
        tick: int | None = None,
        smoke: Sequence[dict[str, Any]] | None = None,
        smoke_key: str | None = None,
        others: Sequence[dict[str, Any]] | None = None,
        opts: dict[str, Any] | None = None,
    ) -> list[bool]:
        """一次问多对：`[(a, b), ...]` → `[bool, ...]`。

        **同一帧问几十个对就用这个**：一趟往返固定 ~50 µs、几何只要 ~8 µs/对，
        循环调 `can_see()` 的 90 次光 IPC 就要 4.5 ms。
        """
        return [bool(r.get("visible")) for r in self.pairs_detail(
            pairs, tick=tick, smoke=smoke, smoke_key=smoke_key, others=others, opts=opts,
        )]

    def pairs_detail(
        self,
        pairs: Sequence[Sequence[Any]],
        *,
        tick: int | None = None,
        smoke: Sequence[dict[str, Any]] | None = None,
        smoke_key: str | None = None,
        others: Sequence[dict[str, Any]] | None = None,
        opts: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """同 `can_see_many`，但每对都带上 `clear / reason / dead`。"""
        req = {"op": "pairs", "pairs": [list(p) for p in pairs],
               **self._opt_keys(tick, smoke, smoke_key, others, opts)}
        return list(self._rpc(req)["results"])

    def states(self, tick: int) -> list[dict[str, Any]]:
        """当前回合 `tick` 那帧的 10 个玩家状态（`getAllPlayerStates()` 的原样输出）。"""
        return list(self._rpc({"op": "states", "tick": int(tick)})["states"])

    def matrix(
        self,
        states: Sequence[dict[str, Any]] | None = None,
        *,
        tick: int | None = None,
        smoke: Sequence[dict[str, Any]] | None = None,
        smoke_key: str | None = None,
    ) -> np.ndarray:
        """整张矩阵：返回 `(n,) uint16`，**bit a = 「这个观察者看得见 a」**（和 `.csz` 表同向）。

        不给 `states` 就必须先 `load_round()` + 给 `tick`。走的是 `createVisibilityEvaluator`
        —— 和出全量表那条路**同一份代码**（`pair` 走的是 `pairVisibility`，带角落明细）。
        """
        req: dict[str, Any] = {"op": "matrix"}
        if states is not None:
            req["states"] = list(states)
        if tick is not None:
            req["tick"] = int(tick)
        if smoke is not None:
            req["smoke"] = list(smoke)
            req["smoke_key"] = smoke_key if smoke_key is not None else f"len{len(smoke)}"
        return np.asarray(self._rpc(req)["masks"], dtype=np.uint16)

    def load_round(self, store: Any, fi: int, rid: int) -> dict[str, Any]:
        """把 csz 里某个回合灌给 node（一次 3–5 MB 的 JSON）。

        ⚠ 用的是 `store.data(fi)`（**合并过的** fire/smoke）；浏览器那边 `codec.load()` 是
        原始碎片，所以碎片回合上两者会有差别。
        """
        from visualizer.server import _build_round_payload   # 懒导入：别把 flask 拖进训练进程

        data = store.data(fi)
        idx = next(
            (i for i, r in enumerate(data["rounds"]) if int(r["id"]) == int(rid)), None
        )
        if idx is None:
            raise KeyError(f"{store.paths[fi].name} 里没有回合 {rid}")
        return self._rpc({
            "op": "load_round", "rid": int(rid), "payload": _build_round_payload(data, idx),
        })


# ── 冻结快照 ────────────────────────────────────────────────────────────────


def _golden_ticks(ticks: np.ndarray, per_round: int) -> list[int]:
    """均匀抽 `per_round` 个 tick（覆盖到有烟和没烟的段，别只取开头）。"""
    step = max(1, int(ticks.size) // max(1, per_round))
    return [int(t) for t in ticks[::step][:per_round]]


def _freeze(args: argparse.Namespace) -> int:
    """重做冻结快照 `oracle_golden.npz`。

    ⚠ **冻结的是"当前行为"，不是真值。** 这个快照抓的是"改动带来的不知情行为变化"。
    第一版（mirage，2 回合 × 32 帧）是先和当时还在的离线表**逐位对拍过**才冻下来的；
    以后 `visibility.js` 有意改了、快照失效了，才用它重冻 —— 重冻要在提交信息里写清楚
    为什么，因为一重冻，那段行为就再没人盯着了。
    """
    from .store import CszStore

    store = CszStore.from_dir(_paths.REPO_ROOT / "data" / "demos" / "csz")
    if not 0 <= args.file < len(store.paths):
        print(f"没有第 {args.file} 个 csz（目录里 {len(store.paths)} 个）")
        return 2
    fi = args.file
    csz = store.paths[fi]
    map_name = store.map_name(fi)
    rids = [int(r) for r in store.round_ids(fi)[: max(1, args.rounds)]]

    rows_r: list[int] = []
    rows_t: list[int] = []
    rows_m: list[np.ndarray] = []
    with VisOracle(
        map_name, maps_dir=args.maps_dir, node=args.node, solid=not args.boundary,
        fov_deg=args.fov, aspect=args.aspect, inset=args.inset,
        exact=args.exact, block_players=args.block_players,
    ) as o:
        for rid in rids:
            o.load_round(store, fi, rid)
            ticks = np.asarray(store.round_tensors(fi, rid)["ticks"])
            for tk in _golden_ticks(ticks, args.per_round):
                rows_r.append(rid)
                rows_t.append(tk)
                rows_m.append(o.matrix(tick=tk))
        info = o.info

    np.savez_compressed(
        GOLDEN,
        csz=np.asarray(csz.name),
        csz_size=np.asarray(csz.stat().st_size),
        map=np.asarray(map_name),
        algo=np.asarray(algo_ver(map_obj_for(map_name, args.maps_dir))),
        fov=np.asarray(float(info["opts"]["fovDeg"])),
        aspect=np.asarray(float(info["opts"]["aspect"])),
        inset=np.asarray(float(info["opts"]["inset"])),
        exact=np.asarray(bool(info["opts"]["exactCount"])),
        block=np.asarray(bool(info["opts"]["blockPlayers"])),
        solid=np.asarray(bool(info["solid"])),
        round=np.asarray(rows_r, dtype=np.int64),
        tick=np.asarray(rows_t, dtype=np.int64),
        masks=np.stack(rows_m),
    )
    print(f"写好 {GOLDEN.name}：{len(rows_r)} 帧 / {len(rids)} 回合（{map_name}），"
          f"{GOLDEN.stat().st_size} 字节")
    return 0


# ── 自测 ────────────────────────────────────────────────────────────────────


def _selftest(args: argparse.Namespace) -> int:
    """拿冻结快照对拍，再验内部一致性和成本。

    ⚠ golden 是**行为快照**，不是外部真值：这个 oracle 已经没有外部真值了（离线表删掉了，
    能当参照的只剩浏览器那侧 `visualizer/tests/test_geo_visibility.mjs`）。所以这些断言
    是"和快照一致"，不是"算得对"。
    """
    from .store import CszStore

    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))

    if not shutil.which(args.node or "node") and args.node is None:
        print("找不到 node —— 装一个，或者 --node <路径>")
        return 2
    if not GOLDEN.is_file():
        print(f"没有冻结快照 {GOLDEN.name} —— 先跑一次 --freeze")
        return 3

    g = np.load(GOLDEN)
    rmap = str(g["map"])
    csz_name = str(g["csz"])
    rround, rtick, rmasks = g["round"], g["tick"], g["masks"]

    store = CszStore.from_dir(_paths.REPO_ROOT / "data" / "demos" / "csz")
    names = [p.name for p in store.paths]
    if csz_name not in names:
        print(f"快照要的是 {csz_name}，data/demos/csz 里没有它")
        return 3
    fi = names.index(csz_name)
    print(f"[oracle 自测] 快照 {GOLDEN.name}：{len(rtick)} 帧 / "
          f"{len(set(int(r) for r in rround))} 回合，地图 {rmap}")

    print("\n[1] 起进程 + 地图（开关照快照来）")
    o = VisOracle(
        rmap, maps_dir=args.maps_dir, node=args.node, solid=bool(g["solid"]),
        fov_deg=float(g["fov"]), aspect=float(g["aspect"]), inset=float(g["inset"]),
        exact=bool(g["exact"]), block_players=bool(g["block"]),
    )
    try:
        check("ping 回来了", bool(o.info.get("map")), f"{o.info.get('tris')} 三角形")
        check("csz 没换过", store.paths[fi].stat().st_size == int(g["csz_size"]),
              f"{store.paths[fi].stat().st_size} 字节")
        av = algo_ver(map_obj_for(rmap, args.maps_dir))
        same = av == str(g["algo"])
        check("几何版本和快照一致", same,
              f"{av} vs {g['algo']}"
              + ("" if same else "  ← JS 改过了，快照过期：对不上的帧就是被改动影响到的"))

        # 先把 csz 文件读进来（`round_ids` 会触发 load）—— 否则第一趟 `load_round` 会把
        # 十几秒的解压算进下面的计时里，看起来像"灌一个回合要 6 秒"。
        t_warm = time.perf_counter()
        store.round_ids(fi)
        warm_ms = (time.perf_counter() - t_warm) * 1000

        print(f"\n[2] 逐位对拍 {len(rtick)} 帧（oracle vs 冻结快照）")
        bad = 0
        vis_total = 0
        load_ms = 0.0
        occ_ms = 0.0
        ev_ms = 0.0
        cur = None
        t0 = time.perf_counter()
        for rid, tk, want in zip(rround, rtick, rmasks):
            rid, tk = int(rid), int(tk)
            if rid != cur:                      # 换回合才重灌载荷（一个回合一次）
                t1 = time.perf_counter()
                lr = o.load_round(store, fi, rid)
                load_ms += (time.perf_counter() - t1) * 1000
                cur = rid
            m = o._rpc({"op": "matrix", "tick": tk})      # 用 _rpc 是为了拿到耗时分解
            got = np.asarray(m["masks"], dtype=np.uint16)
            occ_ms += float(m.get("occ_ms", 0.0))
            ev_ms += float(m.get("ev_ms", 0.0))
            vis_total += sum(bin(int(x)).count("1") for x in got)
            if not np.array_equal(got, want):
                bad += 1
                if bad <= 2:
                    print(f"       r{rid} tick {tk} 不一致："
                          f"oracle={list(map(int, got))} 快照={list(map(int, want))}")
        dt = time.perf_counter() - t0
        n = max(1, len(rtick))
        check("每一帧的 10 行 bitmask 都和快照完全相同", bad == 0,
              f"{len(rtick)} 帧 / {vis_total} 个可见对，"
              f"{(dt * 1000 - load_ms) / n:.2f} ms/帧（其中烟 {occ_ms / n:.2f} + 几何 "
              f"{ev_ms / n:.2f}，含 IPC）")

        # 后面几步只需要第一个回合
        rid = int(rround[0])
        if cur != rid:
            lr = o.load_round(store, fi, rid)
        sel = [int(t) for r, t in zip(rround, rtick) if int(r) == rid]

        print("\n[3] pair 路径和 matrix 路径一致（visibility.js 里是两份代码）")
        mismatch = 0
        dead_cases = 0
        n_cmp = 0
        t_batch = 0.0
        for tk in sel[: max(1, len(sel) // 10)]:
            sts = o.states(tk)
            rowmat = o.matrix(tick=tk)
            order = [(b, a) for b in range(len(sts)) for a in range(len(sts)) if a != b]
            # ⚠ 必须带 `tick`：这样烟遮挡体走的是这个回合那份（和 matrix 一致）。
            #    不带 tick 的话两条路会看不同的烟，烟真挡住人的地方就会假报不一致。
            t0 = time.perf_counter()
            got = o.can_see_many([(sts[b], sts[a]) for b, a in order], tick=tk)
            t_batch += time.perf_counter() - t0
            for (b, a), g in zip(order, got):
                n_cmp += 1
                bit = bool(int(rowmat[b]) >> a & 1)
                if bool(g) != bit:
                    mismatch += 1
                if sts[a].get("alive") is False and bit:
                    mismatch += 1                        # 目标已死却判可见 = 这个位该是 0
                if sts[a].get("alive") is False and not bit:
                    dead_cases += 1
        check("两种实现给出的答案完全相同", mismatch == 0,
              f"{n_cmp} 个方向 0 不一致；覆盖到「目标已死」的负例 {dead_cases} 个")

        sts0 = o.states(sel[0])
        n = args.bench
        t0 = time.perf_counter()
        for _ in range(n):
            o.can_see(sts0[0], sts0[1])
        per = (time.perf_counter() - t0) / n * 1e6
        batch_per = t_batch / max(1, n_cmp) * 1e6
        check("批量入口明显省 IPC", batch_per < per, f"{batch_per:.0f} µs/对 vs 单对往返 {per:.0f} µs")

        print("\n[4] 单对成本（常驻进程里，不含加载地图）")
        check("一趟往返（JSON + IPC + 几何）在微秒级", per < 5000,
              f"{per:.0f} µs/对；几何本身 ~8 µs，其余是 IPC")

        print("\n[5] 语义：alive / yaw 真的被用了")
        al = [dict(s) for s in sts0]
        al[1] = {**al[1], "alive": False}
        check("目标死了 → False", o.can_see(al[0], al[1]) is False)
        check("观察者死了 → False", o.can_see(al[1], al[0]) is False)
        yaw0 = {**sts0[0], "yaw": 0.0, "pitch": 0.0}
        yaw1 = {**sts0[0], "yaw": 180.0, "pitch": 0.0}
        n0 = sum(o.can_see(yaw0, s) for s in sts0 if s is not sts0[0])
        n1 = sum(o.can_see(yaw1, s) for s in sts0 if s is not sts0[0])
        check("转 180° 看见的人数会变（yaw 生效）", n0 != n1, f"yaw0 → {n0} 人，yaw180 → {n1} 人")
    finally:
        o.close()

    print("\n" + ("自测全部通过" if ok else "有失败！"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scripts.training_data.vis_oracle",
        description="单局面可见性：A 能不能看见 B（常驻 node 进程）",
    )
    ap.add_argument("--selftest", action="store_true",
                    help="和冻结快照（oracle_golden.npz）逐位对拍 + 一致性 + 成本")
    ap.add_argument("--freeze", action="store_true",
                    help="重做冻结快照（只在 JS 有意改了之后用；见 _freeze 的说明）")
    ap.add_argument("--maps-dir", default=None, help="地图 obj 目录")
    ap.add_argument("--node", default=None, help="node 可执行文件路径")
    ap.add_argument("--limit", type=int, default=60, help="自测抽查多少个 tick（默认 60）")
    ap.add_argument("--bench", type=int, default=200, help="自测里单对计时的次数（默认 200）")
    # 下面这些只有 --freeze 用得上
    ap.add_argument("--file", type=int, default=0, help="--freeze：第几个 csz（默认 0）")
    ap.add_argument("--rounds", type=int, default=2, help="--freeze：抽几个回合（默认 2）")
    ap.add_argument("--per-round", type=int, default=32, help="--freeze：每回合抽几帧（默认 32）")
    ap.add_argument("--boundary", action="store_true", help="--freeze：烟用边界（默认实心）")
    ap.add_argument("--fov", type=float, default=None, help="--freeze：水平 FOV（默认按 JS）")
    ap.add_argument("--aspect", type=float, default=None, help="--freeze：宽高比（默认按 JS）")
    ap.add_argument("--inset", type=float, default=None, help="--freeze：内缩 HU（默认按 JS）")
    ap.add_argument("--exact", action="store_true", help="--freeze：8 个角全算（默认提前退出）")
    ap.add_argument("--block-players", action="store_true", help="--freeze：玩家互相挡")
    args = ap.parse_args(argv)

    if args.freeze:
        return _freeze(args)
    if args.selftest:
        return _selftest(args)
    ap.print_help()
    print("\n（当库用：from scripts.training_data.vis_oracle import VisOracle, player_state）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
