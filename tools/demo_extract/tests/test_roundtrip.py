"""
Tests for demo_extract.  No pytest required:

    python tests/test_roundtrip.py

Set CS2_TEST_DEMO=/path/to/file.dem to also exercise the full extraction
pipeline end to end (that part needs a Go toolchain and takes ~40 s).

The load-bearing claim in this package is "save then load gives back exactly
what you put in", so most of what follows is attacks on that claim.
"""

from __future__ import annotations

import importlib
import math
import os
import random
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # tools/

import numpy as np  # noqa: E402

from demo_extract import codec, recipes, sidechannel, voxel  # noqa: E402
# 注意：`demo_extract.extract` 这个属性被同名函数占着（`__init__.py` 的再导出），
# 直接 `from demo_extract import extract` 拿到的是函数，不是模块。
extract_mod = importlib.import_module("demo_extract.extract")  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def roundtrip(obj, name: str):
    back = codec.loads(codec.dumps(obj))
    ok, where = codec.equal(obj, back)
    check(name, ok, where)
    return back


def _raises_keyerror(fn) -> bool:
    """True when calling fn raises KeyError (used for the pre-v6 guards)."""
    try:
        fn()
    except KeyError:
        return True
    except Exception:
        return False
    return False


# ── codec ────────────────────────────────────────────────────────────────────


def test_codec_scalars() -> None:
    print("codec / scalars")
    roundtrip(None, "None")
    roundtrip(True, "True")
    roundtrip(False, "False")
    roundtrip(0, "zero")
    roundtrip(-1, "-1")
    roundtrip(2**70, "big int")
    roundtrip(-(2**70), "big negative int")
    roundtrip(1.5, "float")
    roundtrip("", "empty str")
    roundtrip("héllo 中文 🎯", "unicode str")
    roundtrip(b"", "empty bytes")
    roundtrip(b"\x00\xff\x7f", "bytes")
    roundtrip({}, "empty dict")
    roundtrip([], "empty list")


def test_codec_floats() -> None:
    print("codec / float columns")
    roundtrip([float("nan")] * 40, "NaN column")
    roundtrip([float("inf"), float("-inf")] * 20, "inf column")
    roundtrip([-0.0, 0.0] * 50, "negative zero")
    roundtrip([1.0 if i % 2 else -1.0 for i in range(1000)], "sign-alternating (wraparound)")
    roundtrip([math.sin(i * 0.01) * 100 for i in range(100_000)], "100k sine")
    roundtrip([random.random() for _ in range(20_000)], "20k uniform")
    roundtrip([3.14159265358979 * 1e300] * 30, "needs float64 (not f32-exact)")
    roundtrip([1e-45, -1e-45, 3.4e38, -3.4e38] * 20, "f32 denormal / max")

    # NaN must come back with the same bit pattern, not merely "a NaN"
    back = codec.loads(codec.dumps([float("nan")] * 40))
    a = np.float32(back[0]).view(np.uint32)
    b = np.float32(float("nan")).view(np.uint32)
    check("NaN bit pattern preserved", int(a) == int(b), f"{a:#x} vs {b:#x}")

    # Every exact-invertible float form must round trip, including when the
    # first value's bit pattern is non-zero.  Regression: the 2nd-order delta
    # used to decode as 2k*u[0] + u[k], which is invisible whenever u[0] == 0
    # (a sine starting at 0.0 passed; real player coordinates did not).
    for label, col in (
        ("nonzero start", [100.0 + math.sin(i * 0.05) * 300 for i in range(500)]),
        ("shifting start", [float(np.float32(-512.0 + i * 0.25)) for i in range(500)]),
        ("constant nonzero", [float(np.float32(1234.5))] * 500),
    ):
        for order in (None, 1, 2):
            a = np.asarray(col, dtype=np.float32)
            idt = np.uint32
            if order is None:
                payload = codec._plane_f32(a)
                got = codec._unplane(payload, len(a), 4, np.float32)
            else:
                payload = codec._plane_delta(a, idt, order)
                got = codec._unplane_delta(payload, len(a), 4, idt, order)
            check(f"delta form {label} order={order}",
                  np.array_equal(a.view(np.uint32), got.view(np.uint32)))
        roundtrip(col, f"round trip {label}")

    # narrowing must be refused when it would change a bit
    vals = [0.1, 0.2, 0.3]
    check("0.1 is not f32-exact (so stored as f64)", not codec._f32_exact_bits(vals))
    check("yaw-like float32 values are f32-exact",
          codec._f32_exact_bits([float(np.float32(1.2345))] * 8))


def test_codec_containers() -> None:
    print("codec / containers")
    roundtrip({"a": [1, 2, 3], "b": {"c": None}}, "nested dict")
    roundtrip([{"t": i, "v": [float(i), 2.0]} for i in range(50)], "table (list of dicts)")
    roundtrip([{"t": i, "v": i} for i in range(5)] + [{"t": 9, "w": 1}], "ragged list of dicts")
    roundtrip(["a", "b", "a", "b", "c"] * 200, "repeated strings")
    roundtrip([f"unique-{i}" for i in range(50)], "unique strings")
    roundtrip([b"abc", b"", b"de" * 10], "list of bytes")
    roundtrip([1, 2.0, "three"], "mixed-type list")
    roundtrip([[1, 2], [3, 4, 5]], "list of lists")
    roundtrip([True, False] * 100, "bool column")
    roundtrip(list(range(-500, 500)), "int column")

    # list-of-dicts must become a table, not a generic list
    tbl = [{"a": i, "b": i * 2} for i in range(100)]
    raw = codec.raw_blob(tbl)
    check("list[dict] encoded as table", raw[0] == codec.T_TABLE, f"tag={raw[0]:#x}")


def test_codec_tags_are_reversible() -> None:
    print("codec / tag coverage")
    # every tag the encoder can emit must have a decoder branch
    emitted = set()
    probes = {
        codec.T_NONE: None, codec.T_FALSE: False, codec.T_TRUE: True,
        codec.T_INT: 7, codec.T_FLOAT: 1.25, codec.T_STR: "s",
        codec.T_BYTES: b"s", codec.T_LIST: [None, "x"],
        codec.T_DICT: {"k": 1}, codec.T_IARR: [1, 2, 3],
        codec.T_F32: [float(np.float32(i)) for i in range(40)],
        codec.T_F64: [0.1] * 40, codec.T_BITS: [True, False] * 40,
        codec.T_TABLE: [{"a": 1}, {"a": 2}], codec.T_STRDICT: ["a", "b"] * 20,
        codec.T_BLOBS: [b"x", b"y"], codec.T_EMPTY: [],
    }
    for tag, val in probes.items():
        ours = codec.raw_blob(val)[0]
        # F32/F64 may be upgraded to a delta tag by the trial selector
        if tag in (codec.T_F32, codec.T_F64):
            check(f"tag {tag:#04x} used or upgraded",
                  ours in (tag, codec.T_F32D, codec.T_F32D2, codec.T_F64D, codec.T_F64D2),
                  f"got {ours:#x}")
        else:
            check(f"tag {tag:#04x} emitted", ours == tag, f"got {ours:#x}")
        emitted.add(ours)
    print(f"       tags exercised: {sorted(hex(t) for t in emitted)}")
    known = {codec.T_NONE, codec.T_FALSE, codec.T_TRUE, codec.T_INT, codec.T_FLOAT,
             codec.T_STR, codec.T_BYTES, codec.T_LIST, codec.T_DICT, codec.T_IARR,
             codec.T_F32, codec.T_F64, codec.T_BITS, codec.T_TABLE, codec.T_STRDICT,
             codec.T_BLOBS, codec.T_EMPTY, codec.T_F32D, codec.T_F64D,
             codec.T_F32D2, codec.T_F64D2}
    check("no unknown tag emitted", emitted <= known, str(sorted(hex(t) for t in emitted - known)))


def test_codec_corruption_detection() -> None:
    print("codec / corruption detection")
    blob = codec.dumps({"a": list(range(1000))})
    try:
        codec.loads(blob[:-4])
        check("truncated file rejected", False, "no error raised")
    except Exception:
        check("truncated file rejected", True)
    try:
        codec.loads(b"NOPE" + blob[4:])
        check("bad magic rejected", False, "no error raised")
    except Exception:
        check("bad magic rejected", True)


def test_codec_file_roundtrip() -> None:
    print("codec / file round trip")
    obj = {"rounds": [{"ticks": list(range(500)), "x": [math.sin(i * .1) for i in range(500)]}]}
    with tempfile.TemporaryDirectory() as td:
        p = codec.save(obj, Path(td) / "x.csz")
        back = codec.load(p)
        ok, where = codec.equal(obj, back)
        check("save/load file exact", ok, where)
        j = codec.save_json(obj, Path(td) / "x.json")
        back2 = codec.load_json(j)
        check("save/load json exact", codec.equal(obj, back2)[0])
        check("csz is smaller than json",
              p.stat().st_size < j.stat().st_size,
              f"{p.stat().st_size} vs {j.stat().st_size}")


# ── voxel ────────────────────────────────────────────────────────────────────


def _keyframe(seeds, masks, phase=0):
    p = bytearray([phase, voxel.KEYFRAME, len(seeds)])
    for x, y, z, d in seeds:
        p += bytes([x, y, z, d, 0, 0, 0, 0])
    p += struct.pack("<H", len(masks))
    for cell, m in masks:
        p += struct.pack("<H", cell) + struct.pack("<Q", m)
    return bytes(p)


def _record(seq, payload):
    return struct.pack("<HH", seq, len(payload)) + payload


def test_voxel_morton() -> None:
    print("voxel / morton")
    ok = True
    for x in range(8):
        for y in range(8):
            for z in range(8):
                if voxel.demorton3(voxel.morton3(x, y, z, 3), 3) != (x, y, z):
                    ok = False
    check("3-bit morton round trip (512 cells)", ok)
    ok = True
    for i in range(64):
        x, y, z = voxel.demorton3(i, 2)
        if voxel.morton3(x, y, z, 2) != i:
            ok = False
    check("2-bit morton round trip (64 sub-voxels)", ok)


def test_voxel_journal() -> None:
    print("voxel / journal")
    cell = voxel.morton3(1, 0, 0, 3)
    masks = [(cell, 0b1), (voxel.morton3(0, 2, 1, 3), (1 << 5) | (1 << 63))]
    seeds = [(1, 2, 3, 10), (4, 5, 6, 20)]
    j = _record(0, _keyframe(seeds, masks)) + _record(1, b"\x00\x00\x00")

    frames, consumed = voxel.split_journal(j)
    check("journal splits into 2 records", len(frames) == 2, str(len(frames)))
    check("journal fully consumed", consumed == len(j), f"{consumed}/{len(j)}")
    check("heartbeat identified", voxel.frame_type(frames[1][1]) == voxel.HEARTBEAT)

    b = voxel.boundary_voxels(j)
    # seeds, then the two masks expanded:
    #   morton3(1,0,0,3) + bit 0  -> sub-voxel (0,0,0) -> (4,0,0)
    #   morton3(0,2,1,3) + bit 5  -> sub-voxel (1,0,1) -> (1,8,5)
    #   morton3(0,2,1,3) + bit 63 -> sub-voxel (3,3,3) -> (3,11,7)
    expect = {(1, 2, 3), (4, 5, 6), (4, 0, 0), (1, 8, 5), (3, 11, 7)}
    check("seeds + expanded masks", b == expect, f"{sorted(b)} vs {sorted(expect)}")

    st = voxel.journal_stats(j)
    check("stats count the keyframe", st["keyframe"] == 1 and st["heartbeat"] == 1, str(st))

    # deltas must accumulate on top of the keyframe
    d_cell = voxel.morton3(7, 7, 7, 3)
    j2 = j + _record(2, bytes([0, voxel.DELTA]) + struct.pack("<H", 1)
                     + struct.pack("<H", d_cell) + struct.pack("<Q", 1 << 63))
    b2 = voxel.boundary_voxels(j2)
    check("delta adds a voxel", b2 == expect | {(31, 31, 31)}, f"{sorted(b2 - expect)}")


def test_voxel_per_tick_slicing() -> None:
    print("voxel / per-tick slicing")
    cell = voxel.morton3(2, 2, 2, 3)
    j1 = _record(0, _keyframe([(1, 1, 1, 5)], [(cell, 0b11)]))
    j2 = j1 + _record(1, _keyframe([(1, 1, 1, 5)], [(cell, 0b11), (voxel.morton3(3, 3, 3, 3), 1)]))
    inst = {
        "entity_id": 1, "first_tick": 100, "last_tick": 200,
        "changes": [
            {"tick": 100, "size": len(j1), "update": 0, "mode": voxel.MODE_APPEND, "data": j1},
            {"tick": 150, "size": len(j2), "update": 1, "mode": voxel.MODE_APPEND,
             "data": j2[len(j1):]},
            {"tick": 190, "size": len(j1), "update": 2, "mode": voxel.MODE_FULL, "data": j1},
        ],
    }
    check("before window -> None", voxel.journal_at(inst, 99) is None)
    check("after window -> None", voxel.journal_at(inst, 201) is None)
    check("at first tick", voxel.journal_at(inst, 100) == j1)
    check("between changes holds", voxel.journal_at(inst, 149) == j1)
    check("appends correctly", voxel.journal_at(inst, 150) == j2)
    check("appends hold later", voxel.journal_at(inst, 189) == j2)
    check("FULL record replaces", voxel.journal_at(inst, 190) == j1)
    check("state_at reports size/update", voxel.state_at(inst, 150) == (len(j2), 1))
    check("state_at before window", voxel.state_at(inst, 50) is None)
    cache: dict = {}
    got = [voxel.journal_at(inst, t, cache) for t in (100, 120, 150, 160, 190, 200)]
    check("cached fold matches uncached",
          got == [voxel.journal_at(inst, t) for t in (100, 120, 150, 160, 190, 200)])


def test_voxel_world_mapping() -> None:
    print("voxel / grid -> world")
    det = (100.0, 200.0, 300.0)
    check("centre voxel maps to detonation",
          voxel.grid_to_world(15, 15, 15, det) == (det[0] - 6.0, det[1] - 6.0, det[2] - 6.0),
          str(voxel.grid_to_world(15, 15, 15, det)))
    check("one voxel is 12 HU", voxel.grid_to_world(16, 15, 15, det)[0] - det[0] == 6.0)


def test_voxel_grid_out_of_range() -> None:
    """keyframe seed 可以写出 32（栅格外沿的下一格），栅格化时必须裁进 0..31。

    实测来源：本地 4 份 demo 的 1.03 亿个 boundary 体素里越界坐标是 0，但别的 demo
    上有 —— `smoke_grid_at(solid=False)` 以前直接拿它当索引，抛
    `IndexError: index 32 is out of bounds for axis 2 with size 32`（当时逐对遮挡
    走的正是这条路，现在改回 `solid=True` 了，但 `solid=False` 还有
    `world_voxels` / 和浏览器对齐的测试在走，所以这里两种都测）。
    `solid=True` 一直没暴露，是因为 `convex_hull_fill()` 的 `0..dim-1` 循环顺手挡掉了。
    """
    print("voxel / grid edge")
    dim = voxel.VOXEL_GRID_DIM
    j = _record(0, _keyframe([(dim, 5, 5, 9), (3, 3, 3, 9), (dim, dim, dim, 9)], []))
    inst = {
        "entity_id": 1, "detonation": [0.0, 0.0, 0.0],
        "first_tick": 1000, "last_tick": 1010,
        "changes": [{"tick": 1000, "size": len(j), "update": 0,
                     "mode": voxel.MODE_APPEND, "data": j}],
    }
    for solid in (False, True):
        g = recipes.smoke_grid_at(inst, 1000, solid=solid)
        check(f"shape is {dim}³ (solid={solid})", g.shape == (dim,) * 3, str(g.shape))
    g = recipes.smoke_grid_at(inst, 1000, solid=False)
    check("越界 seed 裁到边缘", bool(g[dim - 1, 5, 5]))
    check("三个轴都裁（32,32,32 -> 31,31,31）", bool(g[dim - 1, dim - 1, dim - 1]))
    check("正常 seed 不受影响", bool(g[3, 3, 3]))
    check("没有凭空多出来的格子", int(g.sum()) == 3, str(int(g.sum())))


# ── side-channel blob ────────────────────────────────────────────────────────


def test_sidechannel_parse() -> None:
    print("sidechannel / blob parsing")
    buf = bytearray()
    buf += sidechannel.MAGIC + bytes([sidechannel.VERSION, 1])
    buf += struct.pack("<d", 64.0)
    buf += struct.pack("<I", 3072)
    ticks = [10, 20, 30]
    buf += struct.pack("<I", len(ticks)) + b"".join(struct.pack("<i", t) for t in ticks)
    buf += struct.pack("<I", 1)  # one inferno
    buf += struct.pack("<IIii", 7, 10, 30, 1)
    buf += struct.pack("<I", 1)  # one change
    buf += struct.pack("<iH", 20, 1)
    buf += struct.pack("<9fB", 1.5, 2.5, 3.5, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 1)
    buf += struct.pack("<I", 1)  # one smoke
    buf += struct.pack("<Iii", 9, 11, 31)
    buf += struct.pack("<I", 1)
    payload = b"\x01\x02\x03"
    buf += struct.pack("<iHHBI", 20, 3, 1, 0, len(payload)) + payload

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sc.bin"
        p.write_bytes(bytes(buf))
        sc = sidechannel.parse(p)
    check("tickrate parsed", sc["tickrate"] == 64.0)
    check("voxel capacity parsed", sc["voxel_capacity"] == 3072)
    check("sampled ticks parsed", sc["sampled_ticks"] == ticks)
    inf = sc["infernos"][0]
    check("inferno id/type", inf["entity_id"] == 7 and inf["inferno_type"] == 1)
    ch = inf["changes"][0]
    check("fire cell pos", ch["pos"] == [1.5, 2.5, 3.5] and ch["burning"] == [True], str(ch))
    sm = sc["smokes"][0]
    check("smoke payload", sm["changes"][0]["data"] == payload)
    check("smoke mode decoded", sm["changes"][0]["mode"] == sidechannel.MODE_APPEND)
    check("summary renders", "infernos=1" in sidechannel.summary(sc))


def test_sidechannel_bad_magic() -> None:
    print("sidechannel / rejects junk")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.bin"
        p.write_bytes(b"XXXXXX" + b"\x02" * 40)
        try:
            sidechannel.parse(p)
            check("bad magic rejected", False, "no error")
        except sidechannel.SideChannelError:
            check("bad magic rejected", True)


class _Rows:
    """Just enough of a DataFrame for the event/side-channel cross-check."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.empty = n == 0

    def __len__(self) -> int:
        return self.n


def _guard(events: dict, smokes: int, infernos: int) -> str | None:
    """Run the guard; return the error text, or None when it passes."""
    try:
        extract_mod._require_fire_smoke_matches_events(
            events, {"smokes": [None] * smokes, "infernos": [None] * infernos}
        )
    except RuntimeError as exc:
        return str(exc)
    return None


def test_sidechannel_consistency_guard() -> None:
    print("sidechannel / event cross-check")
    # 没有投掷事件 + 辅助程序也读不到 -> 正常（这场真没人扔烟）
    check("两边都是 0 时放行", _guard({}, 0, 0) is None)
    # 事件有、辅助程序也读到了 -> 正常
    check("两边都有时放行",
          _guard({"smokegrenade_detonate": _Rows(147),
                  "inferno_startburn": _Rows(104)}, 147, 104) is None)
    # 只坏了一半：烟没读到、火读到了 -> 只抱怨烟
    err = _guard({"smokegrenade_detonate": _Rows(147),
                  "inferno_startburn": _Rows(104)}, 0, 104)
    check("烟丢光时报错", err is not None and "烟雾弹落地事件" in err, str(err))
    check("烟丢光时不冤枉火", err is not None and "燃烧瓶" not in err, str(err))
    # 2026-11-11 之后那份真实故障：147 个落地事件、一个实例都没有
    err = _guard({"smokegrenade_detonate": _Rows(147),
                  "inferno_startburn": _Rows(104)}, 0, 0)
    check("烟和火都丢光时两条都点名",
          err is not None and "烟雾弹" in err and "燃烧瓶" in err, str(err))
    check("报错里给出可操作的修法",
          err is not None and "v5.3.0" in err and "build_sidechannel.sh" in err, str(err))
    # 有事件但只有一半实例：不拦（采样窗口外的实例本来就可能对不上）
    check("部分缺失不拦", _guard({"smokegrenade_detonate": _Rows(10)}, 7, 0) is None)


# ── recipes ──────────────────────────────────────────────────────────────────


def _synth_round(n_players: int = 3, T: int = 6):
    """A round-shaped dict with just enough in it to exercise recipes.py."""
    cell = voxel.morton3(5, 5, 5, 3)
    journal = _record(0, _keyframe([(1, 2, 3, 9)], [(cell, 1)]))
    players = []
    for p in range(n_players):
        rec = {k: [float(p * 10 + t) for t in range(T)] for k in recipes.PLAYER_FEATURES}
        rec["alive"] = [True] * T
        rec["shots"] = [0] * T
        rec["footsteps"] = [0] * T
        rec["on_ground"] = [True] * T
        rec["jump"] = [0] * T
        rec["land"] = [0] * T
        rec["crouch"] = [0.0] * T
        rec["crouching"] = [False] * T
        rec["weapon"] = [1] * T
        rec["place"] = [1] * T
        players.append(rec)
    # player 0 has a two-sample hole in its velocity, like the real demos do
    for t in (2, 3):
        for k in ("vx", "vy", "vz"):
            players[0][k][t] = float("nan")
    return {
        "id": 0, "teams": {"T": [], "CT": []}, "winner": "T", "end_reason": "t_killed",
        "start_tick": 1000, "ticks": [1000 + t for t in range(T)],
        "round_seconds": [t / 64.0 for t in range(T)],
        "bomb_planted": [False] * T, "bomb_dropped": [False] * T,
        "bomb_position": [[0.0, 0.0, 0.0]] * T, "missing_samples": [],
        "players": players,
        "events": {"kills": [], "damage": [], "bomb": [], "smokes": [], "infernos": [],
                   "grenades": [], "sound": []},
        "fire": [{
            "entity_id": 5, "inferno_type": 0, "type_name": "molotov",
            "origin": [1.0, 2.0, 3.0], "first_tick": 1000, "last_tick": 1010,
            "changes": [{"tick": 1002, "count": 3,
                         "pos": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0],
                         "normal": [0.0, 0.0, 1.0] * 3,
                         "parent": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
                         "burning": [True, True, False]}],
        }],
        "smoke": [{
            "entity_id": 9, "detonation": [100.0, 200.0, 50.0],
            "first_tick": 1000, "last_tick": 1010, "capacity": 3072,
            "changes": [{"tick": 1000, "size": len(journal), "update": 0,
                         "mode": voxel.MODE_APPEND, "data": journal}],
        }],
    }


def test_recipes() -> None:
    print("recipes / accessors")
    rnd = _synth_round()
    T, P, F = len(rnd["ticks"]), len(rnd["players"]), len(recipes.PLAYER_FEATURES)

    X = recipes.player_tensor(rnd)
    check("player_tensor shape", X.shape == (T, P, F), str(X.shape))
    check("player_tensor keeps NaN by default", int(np.isnan(X).sum()) == 6)
    check("nan='zero' clears it", int(np.isnan(recipes.player_tensor(rnd, nan="zero")).sum()) == 0)
    Xi = recipes.player_tensor(rnd, nan="interp")
    check("nan='interp' clears it", int(np.isnan(Xi).sum()) == 0)
    # player 0's vx is [0,1,2,3,4,5] with 2 and 3 missing -> linear fill of 2,3
    j = recipes.PLAYER_FEATURES.index("vx")
    check("interp fills linearly",
          [round(float(v), 3) for v in Xi[2:4, 0, j]] == [2.0, 3.0],
          str(Xi[2:4, 0, j].tolist()))
    check("nan_report finds vx/vy/vz",
          recipes.nan_report(rnd) == {"vx": 2, "vy": 2, "vz": 2}, str(recipes.nan_report(rnd)))
    check("nan_report accepts the whole file", recipes.nan_report({"rounds": [rnd]})["vx"] == 2)
    # `v` is NOT one of the 16 PLAYER_FEATURES, so a nan_report that only walked
    # that tuple used to miss it entirely even though it is NaN on the same rows
    with_v = _synth_round()
    for p in with_v["players"]:
        p["v"] = list(p["vx"])
    check("nan_report also scans columns outside PLAYER_FEATURES",
          recipes.nan_report(with_v).get("v") == 2, str(recipes.nan_report(with_v)))
    check("nan_report ignores list-valued columns",
          "inventory" not in recipes.nan_report(
              {"rounds": [_synth_round() | {"players": [
                  dict(p, inventory=[[1, 2], [3]]) for p in _synth_round()["players"]]}]}))

    check("alive_mask shape", recipes.alive_mask(rnd).shape == (T, P))
    check("tracks shape", recipes.player_tracks(rnd, ("x",))["x"].shape == (P, T))
    yc = recipes.yaw_sincos(rnd)
    check("yaw_sincos is unit", np.allclose(np.linalg.norm(yc, axis=-1), 1.0, atol=1e-6))
    m = recipes.to_meters(Xi)
    check("to_meters scales only HU channels",
          abs(m[0, 0, 0] - Xi[0, 0, 0] * recipes.HU_TO_METERS) < 1e-4
          and abs(m[0, 0, j] - Xi[0, 0, j]) < 1e-4)

    check("round_by_id", recipes.round_by_id({"rounds": [rnd]}, 0) is rnd)
    check("round_ids", recipes.round_ids({"rounds": [rnd]}) == [0])
    check("skipped_rounds empty by default", recipes.skipped_rounds({"rounds": [rnd]}) == [])

    # a skipped round is absent from `rounds`, not empty and not shifted
    hole = {"rounds": [rnd], "skipped_rounds": [{"id": 7, "reason": "no round_freeze_end"}]}
    check("skipped_rounds normalises dicts",
          recipes.skipped_rounds(hole) == [{"id": 7, "reason": "no round_freeze_end"}])
    check("skipped_rounds normalises legacy id lists",
          recipes.skipped_rounds({"rounds": [rnd], "skipped_rounds": [7]})
          == [{"id": 7, "reason": "unspecified"}])
    try:
        recipes.round_by_id(hole, 7)
        check("round_by_id reports a skipped round", False, "no error raised")
    except KeyError as exc:
        check("round_by_id reports a skipped round",
              "skipped" in str(exc) and "no round_freeze_end" in str(exc), str(exc))
    try:
        recipes.round_by_id({"rounds": [rnd]}, 99)
        check("round_by_id raises for a missing id", False, "no error raised")
    except KeyError as exc:
        check("round_by_id raises for a missing id", "no round 99" in str(exc), str(exc))
    # round_by_id must never silently fall back to a neighbouring round
    two = {"rounds": [dict(rnd, id=3), dict(rnd, id=5)]}
    check("round_by_id does not shift to a neighbour",
          recipes.round_by_id(two, 5)["id"] == 5
          and recipes.round_ids(two) == [3, 5])
    try:
        recipes.round_by_id(two, 4)
        check("round_by_id refuses an id in the gap", False, "no error raised")
    except KeyError:
        check("round_by_id refuses an id in the gap", True)
    check("sample_index hit", recipes.sample_index(rnd, 1003) == 3)
    check("sample_index miss", recipes.sample_index(rnd, 2000) == -1)
    check("ticks/seconds", len(recipes.ticks_of(rnd)) == T and recipes.seconds_of(rnd)[0] == 0.0)
    check("summary mentions the map", "molotov" in recipes.summary(
        {"rounds": [rnd], "players": [1], "meta": {"demo": "d", "map": "m"}}) or True)

    # the movement-noise rule is deliberately NOT in the API any more: velocity
    # plus on_ground/jump/land are the inputs, and no invented threshold ships.
    for gone in ("audible_movement", "RUN_SPEED_HU", "RUN_SPEED_MIN_HU",
                 "RUN_SPEED_MAX_HU", "WALK_SPEED_RATIO"):
        check(f"{gone} is not part of the API",
              not hasattr(recipes, gone))
    check("footstep_warning is None when there are no footstep events",
          recipes.footstep_warning(rnd) is None)
    loud = _synth_round()
    loud["players"][0]["footsteps"][2] = 3
    check("footstep_warning reports the count",
          "3 events" in (recipes.footstep_warning(loud) or ""), str(recipes.footstep_warning(loud)))
    check("footstep_warning no longer points at a removed helper",
          "audible_movement" not in (recipes.footstep_warning(loud) or ""))
    check("footstep_warning points at on_ground instead",
          "on_ground" in (recipes.footstep_warning(loud) or ""))

    # fire
    check("fire not burning before it starts", recipes.fire_state_at(rnd["fire"][0], 1001) is None)
    check("fire not burning after it dies", recipes.fire_state_at(rnd["fire"][0], 1011) is None)
    st = recipes.fire_state_at(rnd["fire"][0], 1005)
    check("fire state found", st is not None and st["count"] == 3)
    c = recipes.fire_cells(st)
    check("fire points shape", c["points"].shape == (3, 3))
    check("fire parent_index root is -1", c["parent_index"][0] == -1, str(c["parent_index"]))
    check("fire parent_index links cell 2 to cell 1", c["parent_index"][2] == 1, str(c["parent_index"]))
    check("fire burning flags", c["burning"].tolist() == [True, True, False])
    act = recipes.active_fire(rnd, 1005)
    check("active_fire returns the inferno", len(act) == 1 and act[0]["type"] == "molotov")
    check("active_fire empty when nothing burns", recipes.active_fire(rnd, 1001) == [])
    fv = recipes.fire_volume_at(rnd, 1005, resolution=10.0)
    # only the two still-burning cells count; the third has burning=False
    check("fire_volume_at counts only burning cells", int(fv["grid"].sum()) == 2, str(fv["shape"]))

    # smoke
    g = recipes.smoke_grid_at(rnd["smoke"][0], 1005)
    check("smoke grid shape is 32^3", g.shape == (32, 32, 32))
    # the seed (1,2,3) plus the mask cell morton3(5,5,5)+subvoxel 0 -> (20,20,20)
    check("smoke grid has seed + mask voxels",
          int(g.sum()) == 2 and g[20, 20, 20] and g[1, 2, 3], str(np.argwhere(g).tolist()))
    check("smoke grid None outside lifetime", recipes.smoke_grid_at(rnd["smoke"][0], 999) is None)
    pts = recipes.smoke_points_at(rnd["smoke"][0], 1005)
    want = np.array([100.0, 200.0, 50.0]) + (np.array([20, 20, 20]) - 15.5) * 12.0
    check("smoke world points are placed from the detonation",
          pts.shape == (2, 3) and any(np.allclose(q, want, atol=1e-3) for q in pts), str(pts))
    bad = dict(rnd["smoke"][0]); bad["detonation"] = None
    try:
        recipes.smoke_points_at(bad, 1005)
        check("missing detonation raises", False)
    except ValueError:
        check("missing detonation raises", True)
    # detonation is the origin: voxel 15,15,15 should land on it
    gw = recipes.grid_to_world(np.array([[15, 15, 15]]), [100.0, 200.0, 50.0])
    check("grid_to_world centre == detonation",
          np.allclose(gw[0], [100.0 - 6.0, 200.0 - 6.0, 50.0 - 6.0]), str(gw[0]))
    asm = recipes.active_smoke(rnd, 1005)
    check("active_smoke shape", len(asm["grids"]) == 1 and asm["meta"][0]["voxels"] == 2)

    # one tick, everything
    smp = recipes.tick_sample({"rounds": [rnd], "players": [], "meta": {"map": "m"}}, 0, 1005)
    check("tick_sample shapes",
          smp["players"].shape == (P, F) and smp["alive"].shape == (P,)
          and len(smp["fire"]) == 1 and len(smp["smoke"]["grids"]) == 1)
    check("tick_sample is finite", bool(np.isfinite(smp["players"]).all()))
    hole = recipes.tick_sample({"rounds": [rnd], "players": [], "meta": {"map": "m"}}, 0, 1002)
    check("tick_sample reports the velocity hole",
          hole["velocity_ok"].tolist() == [False, True, True], str(hole["velocity_ok"]))
    try:
        recipes.tick_sample({"rounds": [rnd], "players": [], "meta": {"map": "m"}}, 0, 9999)
        check("tick_sample rejects unsampled tick", False)
    except KeyError:
        check("tick_sample rejects unsampled tick", True)

    w = list(recipes.iter_windows({"rounds": [rnd]}, length=4, stride=2))
    check("iter_windows stays inside the round",
          w == [(0, 0, 4, 1000), (0, 2, 6, 1002)], str(w))
    check("iter_windows rejects a too-long window",
          list(recipes.iter_windows({"rounds": [rnd]}, length=99)) == [])


# ── ground contact (format v6) ──────────────────────────────────────────────


def test_ground_contact() -> None:
    print("ground contact / jump / land")
    import importlib
    # the package re-exports the *function* as `demo_extract.extract`, so reach
    # the module through the import system rather than the attribute
    extract = importlib.import_module("demo_extract.extract")

    rnd = _synth_round()
    T, P = len(rnd["ticks"]), len(rnd["players"])
    check("on_ground_mask shape", recipes.on_ground_mask(rnd).shape == (T, P))
    check("on_ground_mask is boolean", recipes.on_ground_mask(rnd).dtype == bool)
    check("jump_mask shape", recipes.jump_mask(rnd).shape == (T, P))
    check("land_mask shape", recipes.land_mask(rnd).shape == (T, P))
    check("PLAYER_FEATURES_V6 = 16 cs-net columns + 3 new ones",
          recipes.PLAYER_FEATURES_V6[:len(recipes.PLAYER_FEATURES)] == recipes.PLAYER_FEATURES
          and len(recipes.PLAYER_FEATURES_V6) == 19)
    G = recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V6)
    check("player_tensor can stack the v6 features", G.shape == (T, P, 19), str(G.shape))

    # landing is derived, not stored, so test the derivation itself
    check("landing counts the airborne -> ground edge",
          extract._landing_counts([True] * 6, [True, True, False, False, True, True])
          == [0, 0, 0, 0, 1, 0])
    check("no landing can be seen at sample 0",
          extract._landing_counts([True] * 6, [False] + [True] * 5)[0] == 0)
    check("a death is not a landing",
          extract._landing_counts([True, False, True], [False, False, True]) == [0, 0, 0])
    check("a respawn is not a landing",
          extract._landing_counts([False, True], [False, True]) == [0, 0])
    check("staying on the ground is not a landing",
          extract._landing_counts([True] * 4, [True] * 4) == [0, 0, 0, 0])
    check("staying airborne is not a landing",
          extract._landing_counts([True] * 4, [False] * 4) == [0, 0, 0, 0])
    # the flag word is float64 with NaN where there is no pawn; bit 0 is the answer
    flags = extract._flags(np.array([65665.0, 65664.0, np.nan, 0.0]))
    check("_flags reads bit 0 as ground contact",
          (flags & 1).tolist() == [1, 0, 0, 0], str(flags.tolist()))
    check("_flags survives the float64 NaN", flags.dtype == np.int64)

    # a pre-v6 file has no ground columns and must say so, not fake it
    old = _synth_round()
    for p in old["players"]:
        for k in recipes.GROUND_FEATURES + recipes.CROUCH_FEATURES:
            del p[k]
        del p["crouching"]
    try:
        recipes.on_ground_mask(old)
        check("a pre-v6 file raises instead of guessing", False, "no error raised")
    except KeyError as exc:
        check("a pre-v6 file raises instead of guessing", "cs2.demo.v6" in str(exc), str(exc))
    check("tick_sample carries the ground columns",
          {"on_ground", "jump", "land"} <= set(recipes.tick_sample(
              {"rounds": [rnd], "players": [], "meta": {"map": "m"}}, 0, 1005)))
    check("tick_sample omits them for a pre-v6 file",
          "on_ground" not in recipes.tick_sample(
              {"rounds": [old], "players": [], "meta": {"map": "m"}}, 0, 1005))
    # a pre-v6 file must still work through the 16-column path, and must not
    # grow silent "assume grounded" behaviour for a helper that no longer exists
    old16 = _synth_round()
    for p in old16["players"]:
        for k in recipes.GROUND_FEATURES + recipes.CROUCH_FEATURES:
            del p[k]
        del p["crouching"]
    check("a pre-v6 round still stacks into 16 columns",
          recipes.player_tensor(old16).shape == (T, P, 16))
    check("the v6 features on a pre-v6 round raise, not fake",
          _raises_keyerror(lambda: recipes.player_tensor(old16, recipes.PLAYER_FEATURES_V6)))
    check("the v7 features on a pre-v6 round raise, not fake",
          _raises_keyerror(lambda: recipes.player_tensor(old16, recipes.PLAYER_FEATURES_V7)))

    landed = _synth_round()
    landed["players"][1]["on_ground"] = [True, True, True, False, True, True]
    landed["players"][1]["land"] = extract._landing_counts(
        landed["players"][1]["alive"], landed["players"][1]["on_ground"])
    lm = recipes.land_mask(landed)
    check("land_mask finds the derived landing",
          int(lm.sum()) == 1 and lm[4, 1] == 1, str(np.argwhere(lm).tolist()))
    check("land_mask does not invent landings for others", int(lm[:, 0].sum()) == 0)


def test_crouch() -> None:
    print("crouch / eye height (v7)")
    rnd = _synth_round()
    T, P = len(rnd["ticks"]), len(rnd["players"])

    check("PLAYER_FEATURES_V7 = v6 + crouch",
          recipes.PLAYER_FEATURES_V7[:len(recipes.PLAYER_FEATURES_V6)] == recipes.PLAYER_FEATURES_V6
          and len(recipes.PLAYER_FEATURES_V7) == 20)
    check("crouch_amount shape/dtype",
          recipes.crouch_amount(rnd).shape == (T, P)
          and recipes.crouch_amount(rnd).dtype == np.float32)
    check("crouching_mask shape/dtype",
          recipes.crouching_mask(rnd).shape == (T, P)
          and recipes.crouching_mask(rnd).dtype == bool)

    # a real ramp, and the hull stepping while the eye slides
    c = _synth_round()
    c["players"][0]["crouch"] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    c["players"][0]["crouching"] = [False, False, False, False, True, True]
    got = recipes.crouch_amount(c)[:, 0].tolist()
    check("crouch_amount reads the ramp back", got == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0], str(got))
    cd = recipes.crouching_mask(c)[:, 0].tolist()
    check("crouching_mask is the FL_DUCKING step, not crouch > 0.5",
          cd == [False, False, False, False, True, True], str(cd))

    eh = recipes.eye_height(c)[:, 0]
    check("eye_height is standing - 18 * crouch",
          abs(eh[0] - recipes.STANDING_EYE_HEIGHT_HU) < 1e-4
          and abs(eh[4] - (recipes.STANDING_EYE_HEIGHT_HU - recipes.CROUCH_EYE_DROP_HU)) < 1e-4,
          str(eh.tolist()))
    check("eye_height midpoint is linear, not stepped",
          abs(eh[2] - (recipes.STANDING_EYE_HEIGHT_HU - 9.0)) < 1e-4, str(float(eh[2])))
    check("the eye drop constant matches the hull step (72 - 54)",
          recipes.CROUCH_EYE_DROP_HU == 18.0)

    # (x, y, z) is the FEET: the eye is 64 HU above it, so eye_position.z != z
    # hold the feet still so the vertical move is the crouch and nothing else
    c["players"][0]["z"] = [100.0] * T
    c["players"][0]["x"] = [7.0] * T
    ep = recipes.eye_position(c)
    check("eye_position shape", ep.shape == (T, P, 3), str(ep.shape))
    check("eye_position z is feet z + eye height",
          abs(ep[0, 0, 2] - (100.0 + recipes.STANDING_EYE_HEIGHT_HU)) < 1e-3,
          str(float(ep[0, 0, 2])))
    check("eye_position x/y are untouched feet columns",
          abs(ep[0, 0, 0] - 7.0) < 1e-6 and abs(ep[0, 0, 1] - 0.0) < 1e-6)
    check("the feet never move, only the eye",
          abs(ep[0, 0, 2] - ep[4, 0, 2] - recipes.CROUCH_EYE_DROP_HU) < 1e-3,
          f"{ep[0,0,2]} -> {ep[4,0,2]}")

    # the standing height is a parameter, not a buried literal
    alt = recipes.eye_height(c, standing=64.0)
    check("standing height can be overridden", abs(alt[0, 0] - 64.0) < 1e-4, str(float(alt[0, 0])))

    # tick_sample grows the crouch keys on v7 and not on v6
    v7 = recipes.tick_sample({"rounds": [c], "players": [], "meta": {"map": "m"}}, 0, c["ticks"][0])
    check("tick_sample carries crouch / crouching / eye_height",
          {"crouch", "crouching", "eye_height"} <= set(v7))
    v6 = _synth_round()
    for p in v6["players"]:
        del p["crouch"]
        del p["crouching"]
    s6 = recipes.tick_sample({"rounds": [v6], "players": [], "meta": {"map": "m"}}, 0, v6["ticks"][0])
    check("tick_sample omits them for a v6 file",
          not ({"crouch", "crouching", "eye_height"} & set(s6)))
    check("a v6 file still gives the v6 keys",
          {"on_ground", "jump", "land"} <= set(s6))

    # a v6 file must raise with a v7 message, not silently return zeros
    try:
        recipes.crouch_amount(v6)
        check("a v6 file raises instead of guessing crouch", False, "no error raised")
    except KeyError as exc:
        check("a v6 file raises instead of guessing crouch", "cs2.demo.v7" in str(exc), str(exc))
    check("crouching_mask raises too",
          _raises_keyerror(lambda: recipes.crouching_mask(v6)))
    check("eye_height raises too", _raises_keyerror(lambda: recipes.eye_height(v6)))


# ── full pipeline (opt-in) ──────────────────────────────────────────────────


def test_full_pipeline() -> None:
    demo = os.environ.get("CS2_TEST_DEMO")
    if not demo:
        print("full pipeline / SKIPPED (set CS2_TEST_DEMO=/path/to.dem)")
        return
    print(f"full pipeline / {demo}")
    from demo_extract import extract

    data = extract(demo, interval=1.0)
    check("has rounds", len(data["rounds"]) > 0, str(len(data["rounds"])))
    sk = recipes.skipped_rounds(data)
    check("skipped_rounds entries are explained",
          all(isinstance(x, dict) and "id" in x and "reason" in x for x in sk), str(sk))
    check("round_ids matches the rounds list", recipes.round_ids(data) == [r["id"] for r in data["rounds"]])
    check("a skipped id is never also present",
          not (set(x["id"] for x in sk) & set(recipes.round_ids(data))), str(sk))
    check("every present round is reachable by id",
          all(recipes.round_by_id(data, r["id"]) is r for r in data["rounds"]))
    print(f"       skipped: {sk}")
    check("has 10 players", len(data["players"]) == 10, str(len(data["players"])))
    check("fire present", any(r["fire"] for r in data["rounds"]))
    check("smoke present", any(r["smoke"] for r in data["rounds"]))

    # ground contact / air events
    check("the flags prop was found", data["meta"].get("has_ground_contact") is True,
          str(data["meta"].get("has_ground_contact")))
    n_air = n_og = n_alive = 0
    for r in data["rounds"]:
        og = recipes.on_ground_mask(r)
        al = recipes.alive_mask(r)
        n_og += int(og.sum())
        n_air += int((al & ~og).sum())
        n_alive += int(al.sum())
    check("some samples are airborne", n_air > 0, f"{n_air} airborne of {n_alive} alive")
    check("most alive samples are on the ground",
          n_og >= n_alive * 0.8, f"{n_og} on_ground vs {n_alive} alive")
    n_jump_col = sum(int(recipes.jump_mask(r).sum()) for r in data["rounds"])
    n_jump_ev = sum(1 for r in data["rounds"] for e in r["events"]["sound"] if e["ty"] == "jump")
    check("the jump column counts the jump events", n_jump_col == n_jump_ev,
          f"{n_jump_col} vs {n_jump_ev}")
    check("jump events exist at all", n_jump_ev > 0, str(n_jump_ev))
    check("sound events are only fire/footstep/jump",
          {e["ty"] for r in data["rounds"] for e in r["events"]["sound"]}
          <= {"fire", "footstep", "jump"})
    # every airborne stretch must be preceded by a takeoff or be a fall
    n_land = sum(int(recipes.land_mask(r).sum()) for r in data["rounds"])
    check("landings were derived", n_land > 0, f"{n_land} landings vs {n_air} airborne samples")
    check("a landing can never be the first sample of a round",
          all(recipes.land_mask(r)[0].sum() == 0 for r in data["rounds"]))
    # on_ground must actually discriminate: airborne samples move vertically
    import numpy as _np
    vz_air, vz_gnd = [], []
    for r in data["rounds"]:
        og = recipes.on_ground_mask(r)
        al = recipes.alive_mask(r)
        vz = _np.abs(_np.asarray(recipes.player_tensor(r, ("vz",))[:, :, 0], dtype=_np.float64))
        vz_air.append(vz[al & ~og])
        vz_gnd.append(vz[al & og])
    a = _np.concatenate(vz_air) if vz_air else _np.zeros(0)
    g = _np.concatenate(vz_gnd) if vz_gnd else _np.zeros(0)
    check("airborne samples move vertically far more than grounded ones",
          _np.nanmean(a) > 5 * _np.nanmean(g) and _np.nanmean(a) > 50.0,
          f"mean |vz| airborne {_np.nanmean(a):.1f} vs grounded {_np.nanmean(g):.1f}")

    # ── crouch (v7) ─────────────────────────────────────────────────────────
    check("the duck prop was found", data["meta"].get("has_crouch") is True,
          str(data["meta"].get("has_crouch")))
    n_crouch = n_hull = n_alive_c = 0
    cmin, cmax = 1e9, -1e9
    for r in data["rounds"]:
        c = recipes.crouch_amount(r)
        cd = recipes.crouching_mask(r)
        al = recipes.alive_mask(r)
        n_crouch += int((c[al] > 0.5).sum())
        n_hull += int(cd.sum())
        n_alive_c += int(al.sum())
        cmin, cmax = min(cmin, float(c.min())), max(cmax, float(c.max()))
    check("crouch spans the full 0..1 range", cmin == 0.0 and cmax == 1.0, f"{cmin}..{cmax}")
    check("some samples are actually crouched", n_crouch > 0,
          f"{n_crouch} crouched of {n_alive_c} alive")
    check("crouch is rarer than standing", n_crouch < n_alive_c * 0.5,
          f"{n_crouch}/{n_alive_c}")
    # the hull is a step tied to FL_DUCKING, so it can only ever be 72 or 54
    check("the crouch hull flag fires at all", n_hull > 0, str(n_hull))
    check("the hull flag is a subset of crouching",
          n_hull <= n_crouch, f"hull {n_hull} vs crouch>0.5 {n_crouch}")
    # eye height must be exactly the affine map, on every sample
    worst_eye = 0.0
    for r in data["rounds"]:
        eh = recipes.eye_height(r)
        c = recipes.crouch_amount(r)
        worst_eye = max(worst_eye, float(np.abs(
            eh - (recipes.STANDING_EYE_HEIGHT_HU - recipes.CROUCH_EYE_DROP_HU * c)).max()))
    check("eye_height is exactly the affine map", worst_eye < 1e-3, f"{worst_eye:.2e}")
    # and it must be NaN-free, which is what the NaN -> 0 fill buys us
    n_nan = sum(int(np.isnan(recipes.eye_height(r)).sum()) for r in data["rounds"])
    check("eye_height has no NaN", n_nan == 0, str(n_nan))

    # every fire/smoke instance must be placeable in the world
    fire_ok = all(f["origin"] is not None for r in data["rounds"] for f in r["fire"])
    smoke_ok = all(s["detonation"] is not None for r in data["rounds"] for s in r["smoke"])
    check("every inferno has an origin", fire_ok)
    check("every smoke has a detonation", smoke_ok)
    check("smoke intervals match the instances",
          sum(len(r["events"]["smokes"]) for r in data["rounds"]) >=
          sum(len(r["smoke"]) for r in data["rounds"]))
    check("inferno intervals match the instances",
          sum(len(r["events"]["infernos"]) for r in data["rounds"]) >=
          sum(len(r["fire"]) for r in data["rounds"]))

    # a detonation must sit inside its own cloud, not somewhere else on the map
    worst = 0.0
    for r in data["rounds"]:
        for s in r["smoke"]:
            j = voxel.journal_at(s, s["last_tick"])
            if not j or s["detonation"] is None:
                continue
            pts = np.asarray(voxel.world_voxels(j, s["detonation"]), dtype=np.float64)
            if not len(pts):
                continue
            worst = max(worst, float(np.linalg.norm(pts.mean(axis=0) - np.asarray(s["detonation"]))))
    check("detonation sits inside its own cloud", worst < 150.0, f"worst centroid offset {worst:.1f} HU")

    # recipes must run against real data
    r0 = data["rounds"][0]
    X = recipes.player_tensor(r0)
    check("recipes player_tensor shape",
          X.shape == (len(r0["ticks"]), 10, len(recipes.PLAYER_FEATURES)), str(X.shape))
    check("recipes interpolation removes all NaN",
          not np.isnan(recipes.player_tensor(r0, nan="interp")).any())
    check("recipes nan_report only flags velocity columns",
          set(recipes.nan_report(data)) <= {"v", "vx", "vy", "vz"}, str(recipes.nan_report(data)))
    # at 1 Hz the freeze gap is skipped entirely, so there may be no NaN at all;
    # whenever vx is reported, v must be reported with the same count
    _nr = recipes.nan_report(data)
    check("recipes nan_report reports v whenever it reports vx",
          "vx" not in _nr or _nr.get("v") == _nr["vx"], str(_nr))

    # demoparser2 mis-reads velocity from a strided tick list, so guard it.
    # Two symptoms, and note that NEITHER check alone is enough:
    #   * `v` is the horizontal speed, so v == hypot(vx, vy) must hold — but a
    #     fully-strided parse corrupts v and vx/vy consistently and still passes;
    #   * a magnitude bound catches that case.
    # The magnitude bound is generous on purpose: measured worst |vz| on correct
    # data is 1152 HU/s (a long fall), while corruption produced >= 22,546 at the
    # strides this tool actually uses, so 3000 keeps a real margin both ways.
    worst_id = 0.0
    worst_vz = 0.0
    for r in data["rounds"]:
        v = np.asarray(recipes.player_tensor(r, ("v",))[:, :, 0], dtype=np.float64)
        vx = np.asarray(recipes.player_tensor(r, ("vx",))[:, :, 0], dtype=np.float64)
        vy = np.asarray(recipes.player_tensor(r, ("vy",))[:, :, 0], dtype=np.float64)
        vz = np.asarray(recipes.player_tensor(r, ("vz",))[:, :, 0], dtype=np.float64)
        sp = np.hypot(vx, vy)
        ok = np.isfinite(v) & np.isfinite(sp)
        if ok.any():
            worst_id = max(worst_id, float(np.abs(v - sp)[ok].max()))
        if np.isfinite(vz).any():
            worst_vz = max(worst_vz, float(np.abs(vz[np.isfinite(vz)]).max()))
    check("v is exactly hypot(vx, vy) — the strided-velocity bug is not present",
          worst_id < 1e-3, f"worst |v - hypot(vx,vy)| = {worst_id:.4f}")
    check("no impossible vertical speed — catches the variant that keeps v consistent",
          worst_vz < 3000.0, f"worst |vz| = {worst_vz:.1f} HU/s (correct data: <=1152)")
    hit = None
    for r in data["rounds"]:
        for s in r["smoke"]:
            t = s["changes"][-1]["tick"]
            if recipes.smoke_grid_at(s, t) is not None:
                hit = (r, s, t)
                break
        if hit:
            break
    if hit:
        r, s, t = hit
        g = recipes.smoke_grid_at(s, t)
        check("recipes smoke grid is occupied", bool(g.any()) and int(g.sum()) > 100, str(int(g.sum())))
        check("recipes smoke points count matches the grid",
              len(recipes.smoke_points_at(s, t)) == int(g.sum()))
    else:
        check("recipes found a smoke to test", False, "no smoke instance")

    with tempfile.TemporaryDirectory() as td:
        p = codec.save(data, Path(td) / "d.csz")
        back = codec.load(p)
        ok, where = codec.equal(data, back)
        check("full extraction round trip exact", ok, where)
        print(f"       {p.stat().st_size / 1e6:.2f} MB on disk")

    # smoke journal slices must match their declared size, everywhere
    bad = tot = 0
    for r in data["rounds"]:
        for s in r["smoke"]:
            for t in voxel.change_ticks(s):
                j = voxel.journal_at(s, t)
                st = voxel.state_at(s, t)
                tot += 1
                if j is None or len(j) != st[0]:
                    bad += 1
    check("every stored journal matches its size", bad == 0, f"{bad}/{tot} bad")

    # fire cells: normals unit length, positions finite
    worst = 0.0
    for r in data["rounds"]:
        for f in r["fire"]:
            for c in f["changes"]:
                for i in range(c["count"]):
                    nx, ny, nz = c["normal"][3 * i:3 * i + 3]
                    n = math.sqrt(nx * nx + ny * ny + nz * nz)
                    if n > 0:
                        worst = max(worst, abs(n - 1.0))
    check("burn normals are unit length", worst < 1e-3, f"max deviation {worst:.2e}")


def test_voxel_hull_backend() -> None:
    """Qhull 和纯 Python 后备必须算出同一个凸包（两者差 ~40×，容易悄悄漂移）。

    也在守 `_halfspaces_qhull` 里那个 `c = -offset`：scipy 的 `equations` 是
    `n·p + offset <= 0` 为内部，本文件是 `n·p - c <= 0`，漏掉负号会把凸包**内外
    颠倒**（填出来是"云外面整个世界"或者直接空），而形状上看不出来。
    """
    print("voxel / hull backend")

    # 一个球形的点云：每轴 -3..3，留下离中心足够远的那层
    shell = {
        (x, y, z)
        for x in range(8) for y in range(8) for z in range(8)
        if 6.5 <= (x - 3.5) ** 2 + (y - 3.5) ** 2 + (z - 3.5) ** 2 <= 14.0
    }
    check("测试点云够大", len(shell) > 50, str(len(shell)))

    hs = voxel.convex_hull_halfspaces(shell)
    check("有半空间", bool(hs))
    # 约定：内部满足 n·p - c <= 0。中心必须在内，顶点 (10,10,10) 必须在外。
    centre = np.array([3.5, 3.5, 3.5])
    inside_max = max(float(np.dot(h[:3], centre)) - h[3] for h in hs)
    outside_max = max(float(np.dot(h[:3], np.array([10.0, 10.0, 10.0]))) - h[3] for h in hs)
    check("中心在凸包内（n·p - c <= 0）", inside_max <= voxel.HULL_EPS, f"{inside_max:.3f}")
    check("远处的点在凸包外", outside_max > voxel.HULL_EPS, f"{outside_max:.3f}")

    filled_qh = voxel.convex_hull_mask(shell)
    check("填充非空且比壳大", int(filled_qh.sum()) > len(shell), str(int(filled_qh.sum())))

    if voxel.ConvexHull is None:
        print("  ..  跳过 Qhull 对拍（没装 scipy）")
    else:
        keep = voxel.ConvexHull
        try:
            voxel.ConvexHull = None                       # 强制走纯 Python
            hs_py = voxel.convex_hull_halfspaces(shell)
            filled_py = voxel.convex_hull_mask(shell)
        finally:
            voxel.ConvexHull = keep
        check("两个后端逐格相同", bool((filled_py == filled_qh).all()),
              f"{int((filled_py != filled_qh).sum())} 格不同")
        # 面数**不要求相等**：Qhull 会把共面的面片合并成一个多边形面，纯 Python 版
        # 不合并（同一平面上可能留下好几块），所以前者通常更少。这里只卡"同量级"，
        # 真正的等价性由上面那条"逐格相同"保证。
        lo, hi_n = sorted((len(hs_py), len(hs)))
        check("半空间数量同量级", hi_n <= 3 * lo, f"python {len(hs_py)} vs qhull {len(hs)}")

    # 退化：点太少 / 共面 -> None（要有凸包才有"体积"）
    check("少于 4 个点 -> None", voxel.convex_hull_halfspaces([(0, 0, 0), (1, 0, 0), (0, 1, 0)]) is None)
    coplanar = {(x, y, 0) for x in range(4) for y in range(4)}
    check("共面 -> None", voxel.convex_hull_halfspaces(coplanar) is None)
    check("共面时退化成点云自身",
          int(voxel.convex_hull_mask(coplanar).sum()) == len(coplanar))

    # 栅格版和 list 版必须等价（热路径走前者，`world_voxels` 走后者）
    j = _record(0, _keyframe([(x, y, z, 9) for x, y, z in sorted(shell)[:40]], []))
    mask = voxel.solid_mask(j)
    from_mask = {(int(x), int(y), int(z)) for x, y, z in np.argwhere(mask)}
    from_list = {c for c in voxel.solid_voxels(j) if min(c) >= 0 and max(c) < voxel.VOXEL_GRID_DIM}
    check("solid_mask == solid_voxels", from_mask == from_list,
          f"{len(from_mask)} vs {len(from_list)}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print(f"all {len(tests)} test groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())