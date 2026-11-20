"""
Per-frame CS2 demo extraction — the main entry point.

What this is
------------
The cs-net v2 extractor, plus the two things demoparser2 cannot reach at all:
inferno fire-cell geometry and the smoke voxel journal.  It answers the same
question cs-net asked — "what did every player look like at every sampled tick,
and what happened?" — but it can be sampled at full tick rate (64/s) instead of
4/s, and it carries the fire/smoke state so a renderer or a spatial model has
something real to draw.

    >>> from demo_extract import extract
    >>> data = extract("match.dem")                 # every tick (~64 Hz)
    >>> data = extract("match.dem", interval=0.25)  # 4 Hz, cs-net's cadence
    >>> data = extract("match.dem", stride=8)       # every 8th tick
    >>> from demo_extract import codec
    >>> codec.save(data, "match.csz")
    >>> again = codec.load("match.csz")
    >>> codec.equal(data, again)                    # exact

Cost of the two parsers
-----------------------
The demo is read twice: once by demoparser2 (players, events, grenades) and once
by the demoinfocs-golang side-channel (fire cells, smoke journals).  Measured on
a 110 MB / 18-round demo: 7.3 s + 5.4 s.  That is the price of not forking
demoparser2.

Sampling
--------
Ticks are selected per round, so a round's sample index 0 is its own
`round_freeze_end`, not the demo's tick 0.

* ``stride=1`` (or ``interval=None``) — every tick, ~64 samples/second.
* ``interval`` — seconds between samples, nearest actual tick wins (this is
  cs-net's `interval=0.5`, i.e. 2 Hz).

Every consumer of the result treats a sample as "the state at tick T"; nothing
in the file depends on there being exactly one sample per tick.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from demoparser2 import DemoParser

from . import sidechannel as sc

FORMAT_VERSION = "cs2.demo.v7"

# The pawn's continuous crouch amount, 0 = standing, 1 = fully crouched.
# It is the only crouch netprop in CS2 that carries information: over 2.0M
# sampled rows across two demos, `m_bInCrouch`, `m_nCrouchState` and
# `m_nDuckTimeMsecs` are *identically zero* (unique value count 1), and
# `m_bDucking` / `m_bDucked` are transient key states rather than "is crouched".
# `m_flDuckOffset` is exactly -18 x this value, so storing both would be pure
# redundancy -- the offset is derived on the read side instead.
DUCK_PROP = "CCSPlayerPawn.CCSPlayer_MovementServices.m_flDuckAmount"

TEAM_MAP = {2: "T", 3: "CT"}

# demoparser2 is a Rust extension: a panic inside it surfaces as a
# `pyo3_runtime.PanicException`, which derives from BaseException rather than
# Exception.  `except Exception` therefore does *not* catch it, and an
# uncaught one kills the whole scrape instead of skipping one demo.
# Known upstream trigger: a demo whose game event carries more values than its
# own event-list descriptor defines keys -- demoparser2 0.42.0 indexes
# `event_desc.keys[i]` with `i` from `event.keys.len()` and panics with
# "index out of bounds".  Fixed upstream after 0.42.0 by PR #351, unreleased.
# Panics are deterministic per (demo, event), so catching them here is safe:
# the same call can never succeed on a retry.
PANIC_TYPES = ("PanicException",)


def _failure_note(exc: BaseException) -> str:
    """One-line description of an extract failure, naming Rust panics clearly."""
    kind = type(exc).__name__
    detail = str(exc).strip() or traceback.format_exc().strip().splitlines()[-1]
    if kind in PANIC_TYPES:
        return (f"{kind} in demoparser2 (upstream bug, not a corrupt file): "
                f"{detail}")
    return f"{kind}: {detail}"


EVENTS_ALWAYS = [
    "round_freeze_end",
    "round_end",
    "player_death",
    "player_hurt",
    "weapon_fire",
    "player_footstep",
    "player_jump",
    "bomb_planted",
    "bomb_exploded",
    "bomb_defused",
    "smokegrenade_detonate",
    "smokegrenade_expired",
    "inferno_startburn",
    "inferno_expire",
]

# Per-tick player properties.  Same set cs-net v2 asks for, so nothing that the
# training pipeline already consumes can go missing.
PLAYER_PROPS = [
    "X", "Y", "Z",
    "pitch", "yaw",
    "velocity", "velocity_X", "velocity_Y", "velocity_Z",
    "health", "armor", "has_helmet", "has_defuser",
    "is_alive", "weapon_name", "inventory", "inventory_as_ids",
    "flash_duration", "flash_max_alpha",
    "last_place_name", "approximate_spotted_by",
    "team_num", "steamid", "name",
    "game_time", "total_rounds_played",
    "is_bomb_planted", "is_bomb_dropped",
    # v6: ground contact.  Bit 0 (FL_ONGROUND) of the pawn's flag word agrees
    # 100.0000% with "m_hGroundEntity is a valid handle" over 727,649 samples,
    # and flags is one small integer per sample instead of a handle.
    "CCSPlayerPawn.m_fFlags",
    # v7: continuous crouch.  Verified stride-safe (stride 2 / stride 16 /
    # 200 random gaps all match a contiguous parse bit for bit), so unlike
    # velocity it can ride the normal strided pass.
    DUCK_PROP,
]

# The four columns demoparser2 cannot read from a strided tick list.  They get
# their own pass with no `ticks` filter; everything else is read from `sampled`
# directly.  Keeping the two lists in sync is automatic, so a prop added above
# is safe by default.
VELOCITY_PROPS = ["velocity", "velocity_X", "velocity_Y", "velocity_Z"]
STATIC_PROPS = [p for p in PLAYER_PROPS if p not in VELOCITY_PROPS]

# The two columns the round/sampling pass needs, and nothing else.
BASIC_PROPS = ["game_time", "total_rounds_played"]

# Normalise event weapon short-names to the display names used in tick data, so
# one weapon has exactly one index.  Lifted verbatim from cs-net v2: the two
# parsers disagree on naming and this table is what reconciles them.
WEAPON_NAME_NORMALISE: dict[str, str] = {
    "ak47": "AK-47", "ak47_txz12": "AK-47",
    "m4a1": "M4A4", "m4a1_txz12": "M4A4",
    "m4a1_silencer": "M4A1-S",
    "famas": "FAMAS", "famas_txz12": "FAMAS",
    "galilar": "Galil AR", "galilar_txz12": "Galil AR",
    "aug": "AUG",
    "sg556": "SG 553",
    "awp": "AWP", "awp_txz12": "AWP",
    "scar20": "SCAR-20",
    "g3sg1": "G3SG1",
    "ssg08": "SSG 08",
    "mac10": "MAC-10", "mac10_txz12": "MAC-10",
    "mp9": "MP9", "mp9_txz12": "MP9",
    "mp7": "MP7", "mp7_txz12": "MP7",
    "mp5": "MP5-SD",
    "ump45": "UMP-45", "ump45_txz12": "UMP-45",
    "p90": "P90", "p90_txz12": "P90",
    "bizon": "PP-Bizon", "bizon_txz12": "PP-Bizon",
    "glock": "Glock-18",
    "hkp2000": "P2000",
    "usp_silencer": "USP-S",
    "p250": "P250",
    "deagle": "Desert Eagle",
    "elite": "Dual Berettas",
    "fiveseven": "Five-SeveN",
    "tec9": "Tec-9", "tec9_txz12": "Tec-9",
    "cz75a": "CZ75 Auto",
    "revolver": "R8 Revolver",
    "nova": "Nova", "nova_txz12": "Nova",
    "xm1014": "XM1014", "xm1014_txz12": "XM1014",
    "mag7": "MAG-7", "mag7_txz12": "MAG-7",
    "sawedoff": "Sawed-Off", "sawedoff_txz12": "Sawed-Off",
    "m249": "M249", "m249_txz12": "M249",
    "negev": "Negev", "negev_txz12": "Negev",
    "hegrenade": "High Explosive Grenade",
    "flashbang": "Flashbang",
    "smokegrenade": "Smoke Grenade",
    "molotov": "Molotov",
    "incgrenade": "Incendiary Grenade",
    "decoy": "Decoy Grenade",
    "taser": "Zeus x27",
    "c4": "C4 Explosive",
    "planted_c4": "C4 Explosive",
    "knife": "Knife",
    "knife_t": "Knife",
    "knife_butterfly": "Butterfly Knife",
    "knife_m9_bayonet": "M9 Bayonet",
    "knife_karambit": "Karambit",
    "knife_huntsman": "Huntsman Knife",
    "knife_nomad": "Nomad Knife",
    "knife_shadow_daggers": "Shadow Daggers",
    "world": "World",
    "inferno": "Incendiary Grenade",
}

# m_nInfernoType: measured on real demos.
INFERNO_TYPE_NAMES = {0: "molotov", 1: "incendiary"}

# Sanity bound for pairing a detonate/startburn with its expire, in ticks.
# Measured lifetimes on a real demo: every smoke pair is exactly 22.06 s,
# every inferno pair is 5.50 s (incendiary) or 7.02 s (molotov).  60 s
# clears all of those with room to spare while still rejecting a mis-pair.
MAX_PROJECTILE_LIFETIME_TICKS = 60 * 64


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ──────────────────────────────────────────────────────────────────────────────


def _num(s) -> str:
    """Any steamid representation -> canonical decimal string."""
    if isinstance(s, str):
        return s
    try:
        return str(int(s))
    except (TypeError, ValueError):
        return str(s)


def _normalise_weapon_name(name: str) -> str:
    if not name:
        return name
    if name.startswith("weapon_"):
        name = name[7:]
    return WEAPON_NAME_NORMALISE.get(name, name)


def _f32_or_f64(a: np.ndarray) -> list:
    """Column -> plain Python list of floats, NaN preserved."""
    return [float(x) for x in np.asarray(a, dtype=np.float64)]


def _numcol0(pdf, name: str) -> list:
    """Like `_numcol` but NaN becomes 0.

    `m_flDuckAmount` is NaN only where the player has no pawn at all (dead and
    respawning).  "Not crouching" is the right reading there, and a NaN would
    poison the eye-height arithmetic on the read side.
    """
    if name not in pdf.columns:
        return [0.0] * len(pdf)
    v = np.nan_to_num(np.asarray(pdf[name].to_numpy(), dtype=np.float64), nan=0.0)
    return _f32_or_f64(v)


def _ints(a, fill: int = 0) -> list:
    """Column -> ints.  Anything absent/NaN becomes `fill` (recorded separately)."""
    out = []
    for x in np.asarray(a):
        try:
            if x is None or (isinstance(x, float) and x != x):
                out.append(fill)
            else:
                out.append(int(x))
        except (TypeError, ValueError):
            out.append(fill)
    return out


def _bools(a) -> list:
    out = []
    for x in np.asarray(a):
        if x is None:
            out.append(False)
        elif isinstance(x, float) and x != x:
            out.append(False)
        else:
            out.append(bool(x))
    return out


def _empty_or_df(v) -> bool:
    return v is None or isinstance(v, list) or (hasattr(v, "empty") and v.empty)


def _event_rows(df) -> int:
    return 0 if _empty_or_df(df) else len(df)


def _require_fire_smoke_matches_events(events: dict, side: dict) -> None:
    """Raise when the side-channel saw nothing while the event stream says otherwise.

    demoinfocs 的实体流一旦读偏，`IgnorePacketEntitiesPanic` 会把坏包丢掉、`ParseToEnd()`
    仍然返回 nil —— 于是产出一份完全合法、但 fire/smoke 全空的 .csz：`infernos=0`、
    `smokes=0`、`voxel_capacity=0`，而同一份 demo 的 `smokegrenade_detonate` /
    `inferno_startburn` 事件一条不少。2026-11-11 的 CS2 更新就是这么沉默地坏掉的：
    它给 CCSPlayerController 加了 fixed8 编码的 uint8 字段，demoinfocs v5.2.0 及更早
    当 varint 读，值 >= 128 就多吃一个字节，整包错位。
    唯一的症状是网页上不再画烟和火，很容易被当成"这场没人扔"，所以这里拿事件流对账，
    宁可让这次提取失败（带着原因），也不写一份没有火/烟的文件。
    """
    smoke_events = _event_rows(events.get("smokegrenade_detonate"))
    fire_events = _event_rows(events.get("inferno_startburn"))
    problems = []
    if smoke_events and not side["smokes"]:
        problems.append(f"{smoke_events} 个烟雾弹落地事件，但辅助程序一个烟雾实例都没读到")
    if fire_events and not side["infernos"]:
        problems.append(f"{fire_events} 个燃烧瓶/燃烧弹起火事件，但辅助程序一个火焰实例都没读到")
    if not problems:
        return
    raise RuntimeError(
        "side-channel 的火/烟和事件流对不上（"
        + "；".join(problems)
        + "）。多半是 Go 辅助程序太旧、解不开这份 demo 的实体流："
        "2026-11-11 之后的 demo 要 demoinfocs-golang >= v5.3.0，"
        "用 tools/demo_extract/build_sidechannel.sh 重编译一次"
    )


def _pick_ticks(
    ticks: np.ndarray,
    round_secs: np.ndarray,
    interval: float | None,
    stride: int | None,
    start_offset: float,
) -> np.ndarray:
    """Choose sample indices within one round.

    stride  -> every Nth actual tick (stride=1 means every tick)
    interval-> nearest actual tick to each multiple of `interval` seconds
    """
    if stride is not None:
        return np.arange(0, len(ticks), max(int(stride), 1))
    if not len(ticks):
        return np.array([], dtype=int)
    if interval is None or interval <= 0:
        return np.arange(len(ticks))
    t_end = float(round_secs[-1])
    if t_end <= start_offset:
        return np.array([len(ticks) - 1])
    targets = np.arange(start_offset, t_end, interval)
    if not len(targets):
        return np.array([len(ticks) - 1])
    # nearest actual tick for every target, vectorised
    pos = np.searchsorted(round_secs, targets)
    pos = np.clip(pos, 1, len(round_secs) - 1)
    left = round_secs[pos - 1]
    right = round_secs[pos]
    pick = np.where(np.abs(right - targets) < np.abs(left - targets), pos, pos - 1)
    return np.unique(pick)


# ──────────────────────────────────────────────────────────────────────────────
# main entry point
# ──────────────────────────────────────────────────────────────────────────────


def extract(
    demo_path: str | Path,
    interval: float | None = None,
    *,
    stride: int | None = None,
    start_offset: float = 0.0,
    include_fire_smoke: bool = True,
    include_inventory: bool = True,
    include_spotted: bool = True,
    build_dir: str | Path | None = None,
    auto_build: bool = True,
    workdir: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Parse a CS2 `.dem` into the v6 structure.

    Parameters
    ----------
    demo_path
        The `.dem` file.
    interval
        Seconds between samples.  ``None`` (default) means every tick.
        ``0.25`` gives cs-net's old 4 Hz cadence.
    stride
        Sample every Nth tick.  Mutually exclusive with `interval`; takes
        precedence.  ``stride=1`` is equivalent to ``interval=None``.
    start_offset
        Seconds into the round at which interval sampling starts.
    include_fire_smoke
        Run the demoinfocs side-channel.  Costs a second full demo parse.
    include_inventory, include_spotted
        Drop these per-tick columns to shrink the output; they are the two
        largest and the least used.
    """
    t0 = time.perf_counter()
    demo_path = Path(demo_path)
    if not demo_path.is_file():
        raise FileNotFoundError(demo_path)

    parser = DemoParser(str(demo_path))
    header = parser.parse_header()
    map_name = str(header.get("map_name", "unknown"))

    # ─ pass 1: two cheap columns, to find rounds and pick the sample ticks ──
    df_basic = parser.parse_ticks(wanted_props=BASIC_PROPS)
    df_basic = (
        df_basic[["tick", "game_time", "total_rounds_played"]]
        .drop_duplicates(subset="tick")
        .sort_values("tick")
        .reset_index(drop=True)
    )
    df_basic["tick"] = df_basic["tick"].astype(np.int64)
    df_basic["total_rounds_played"] = df_basic["total_rounds_played"].astype(np.int64)

    freeze_df = parser.parse_event("round_freeze_end")
    round_end_df = parser.parse_event("round_end")
    freeze_ticks = (
        sorted(int(t) for t in freeze_df["tick"].tolist()) if not _empty_or_df(freeze_df) else []
    )

    basic_tick = df_basic["tick"].to_numpy()
    basic_round = df_basic["total_rounds_played"].to_numpy()
    basic_time = df_basic["game_time"].to_numpy(dtype=np.float64)

    round_start_tick: dict[int, int] = {}
    round_start_time: dict[int, float] = {}
    for ft in freeze_ticks:
        i = int(np.searchsorted(basic_tick, ft))
        if i >= len(basic_tick):
            continue
        rid = int(basic_round[i])
        round_start_tick[rid] = ft
        round_start_time[rid] = float(basic_time[i])

    # round -> the ticks belonging to it, and the window it occupies
    order = np.argsort(basic_round, kind="stable")
    rounds_sorted = basic_round[order]
    boundaries = np.flatnonzero(np.diff(rounds_sorted)) + 1
    groups = np.split(order, boundaries)

    round_ticks_all: dict[int, np.ndarray] = {}
    round_window: dict[int, tuple[int, int]] = {}
    for g in groups:
        if not len(g):
            continue
        rid = int(basic_round[g[0]])
        tk = np.sort(basic_tick[g])
        round_ticks_all[rid] = tk
        round_window[rid] = (int(tk[0]), int(tk[-1]))

    # ── sample selection ────────────────────────────────────────────────────
    sampled: list[int] = []
    for rid in sorted(round_ticks_all):
        if rid not in round_start_time:
            continue  # warmup / pre-match: no round_freeze_end to anchor on
        # A round's timeline starts at its own round_freeze_end.  Without this
        # clip the group for round 0 also swallows the pre-match warmup, which
        # is thousands of ticks with negative round_seconds.
        tk_all = round_ticks_all[rid]
        tk = tk_all[tk_all >= round_start_tick[rid]]
        if not len(tk):
            continue
        round_window[rid] = (int(tk[0]), int(tk[-1]))
        tm = basic_time[np.searchsorted(basic_tick, tk)]
        secs = tm - round_start_time[rid]
        idx = _pick_ticks(tk, secs, interval, stride, start_offset)
        sampled.extend(int(tk[i]) for i in idx)
    sampled = sorted(set(sampled))
    if not sampled:
        raise RuntimeError("no ticks sampled — empty or corrupt demo?")

    if verbose:
        print(f"demo       : {demo_path.name}")
        print(f"map        : {map_name}")
        print(f"rounds     : {len(round_ticks_all)}")
        print(f"sampled    : {len(sampled)} ticks "
              f"({len(sampled) / max(1, len(df_basic)):.0%} of all ticks)")

    # ─ pass 2: the real per-tick pull, restricted to the sampled ticks ─────
    #
    # demoparser2 mis-reads the four velocity columns whenever `ticks` is not a
    # single contiguous run: at stride 2 already ~84% of samples are wrong, and
    # the values that come back are velocities from some other tick (verified
    # against a contiguous parse; X/Y/Z, flags and every other prop are exact).
    # So velocity is pulled in its own pass with no tick filter at all, which is
    # also the cheapest correct path (0.35 s for a whole demo, versus 2.8 s for
    # an explicit contiguous tick list).  Everything else is fine strided.
    df = parser.parse_ticks(wanted_props=STATIC_PROPS, ticks=sampled)
    if df is None or len(df) == 0:
        raise RuntimeError("parse_ticks returned nothing for the sampled ticks")
    df = df.copy()
    df["tick"] = df["tick"].astype(np.int64)
    df["steamid"] = df["steamid"].map(_num)

    vel = parser.parse_ticks(wanted_props=VELOCITY_PROPS)
    if vel is None or len(vel) == 0:
        raise RuntimeError("parse_ticks returned nothing for velocity")
    vel = vel.copy()
    vel["tick"] = vel["tick"].astype(np.int64)
    vel["steamid"] = vel["steamid"].map(_num)
    vel = vel[["tick", "steamid"] + [c for c in VELOCITY_PROPS if c in vel.columns]]
    df = df.merge(vel, on=["tick", "steamid"], how="left", validate="one_to_one")
    df["total_rounds_played"] = df["total_rounds_played"].astype(np.int64)

    # ── events ──────────────────────────────────────────────────────────────
    other = ["total_rounds_played", "game_time"]
    events: dict[str, Any] = {}
    for name in EVENTS_ALWAYS:
        try:
            events[name] = parser.parse_event(name, other=other)
        except BaseException as exc:  # depends on demo content; see PANIC_TYPES
            # A Rust panic is always worth a line even when not verbose: it is
            # the only signal that this demo's event set is incomplete.
            if verbose or type(exc).__name__ in PANIC_TYPES:
                print(f"  event {name} unavailable: {_failure_note(exc)}")
            events[name] = None

    grenades_df = None
    try:
        grenades_df = parser.parse_grenades(grenades=False)
        if isinstance(grenades_df, list):
            grenades_df = None
    except BaseException as exc:  # pragma: no cover
        if verbose or type(exc).__name__ in PANIC_TYPES:
            print(f"  parse_grenades unavailable: {_failure_note(exc)}")

    # ─ identity: 10 players, stable index order ────────────────────────────
    first_tick_rows = df[df["tick"] == sampled[0]]
    steamids = list(dict.fromkeys(first_tick_rows["steamid"].tolist()))
    if len(steamids) != 10:
        # fall back to every steamid seen, still deterministic
        steamids = list(dict.fromkeys(df["steamid"].tolist()))
    player_map = {sid: i for i, sid in enumerate(steamids)}
    name_by_sid: dict[str, str] = {}
    for sid, nm in zip(first_tick_rows["steamid"], first_tick_rows["name"]):
        name_by_sid.setdefault(sid, str(nm))
    for sid, nm in zip(df["steamid"], df["name"]):
        name_by_sid.setdefault(sid, str(nm))
    player_metas = [{"steamid": sid, "name": name_by_sid.get(sid, "")} for sid in steamids]

    weapon_lookup, place_lookup = _build_lookups(df, events, include_inventory)
    if verbose:
        print(f"players    : {', '.join(p['name'] for p in player_metas)}")
        print(f"weapons    : {len(weapon_lookup)}  places: {len(place_lookup)}")

    # ─ bomb carrier, per tick (forward scan over the whole demo) ───────────
    bomb_carrier = _bomb_carrier(parser, sampled, verbose)

    # ── fire / smoke side-channel ───────────────────────────────────────────
    side = None
    if include_fire_smoke:
        # never write next to the demo: dataset mounts are often read-only
        tmp = Path(workdir) if workdir else Path(tempfile.gettempdir())
        blob = tmp / f".{demo_path.stem}.sidechannel.bin"
        try:
            sc.run(demo_path, sampled, blob, build_dir=build_dir,
                    auto_build=auto_build, verbose=verbose)
            side = sc.parse(blob)
        finally:
            blob.unlink(missing_ok=True)
        if not side["f32_exact"]:
            raise RuntimeError(
                "side-channel: an inferno coordinate did not survive the f32 "
                "round trip — refusing to write a lossy file"
            )
        if verbose:
            print(f"sidechannel: {sc.summary(side)}")
        _require_fire_smoke_matches_events(events, side)

    # ── per-round assembly ──────────────────────────────────────────────────
    rounds_data: list[dict] = []
    skipped: list[dict] = []

    def skip(rid: int, reason: str) -> None:
        skipped.append({"id": rid, "reason": reason})

    for rid in sorted(round_ticks_all):
        if rid not in round_start_time:
            # No round_freeze_end, so there is no origin for this round's
            # timeline.  This used to be a bare `continue`, which dropped the
            # round without recording it — leaving a hole in `rounds` that
            # `skipped_rounds` did not explain.
            skip(rid, "no round_freeze_end, so the round has no time origin")
            continue
        win = round_window[rid]
        sub = df[(df["total_rounds_played"] == rid) & (df["tick"] >= win[0])]
        sub = sub[sub["tick"].isin(sampled)]
        if sub.empty:
            skip(rid, "no sampled ticks fell inside this round")
            continue
        try:
            rd = _build_round(
                rid=rid,
                sub=sub,
                sampled=sampled,
                round_start_tick=round_start_tick[rid],
                round_start_time=round_start_time[rid],
                round_end_df=round_end_df,
                player_map=player_map,
                player_metas=player_metas,
                weapon_lookup=weapon_lookup,
                place_lookup=place_lookup,
                events=events,
                grenades_df=grenades_df,
                bomb_carrier=bomb_carrier,
                side=side,
                include_inventory=include_inventory,
                include_spotted=include_spotted,
            )
            rounds_data.append(rd)
        except Exception as exc:
            skip(rid, f"{type(exc).__name__}: {exc}")
            if verbose:
                print(f"  round {rid} skipped: {exc}")

    result: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "meta": {
            "demo": demo_path.name,
            "map": map_name,
            "interval": interval,
            "stride": stride,
            "start_offset": start_offset,
            "sampled_ticks": len(sampled),
            "total_ticks": int(len(df_basic)),
            "has_fire_smoke": side is not None,
            "has_ground_contact": "CCSPlayerPawn.m_fFlags" in df.columns,
            "has_crouch": DUCK_PROP in df.columns,
            "sidechannel_tickrate": side["tickrate"] if side else None,
            "voxel_capacity": side["voxel_capacity"] if side else None,
            "inferno_type_names": {str(k): v for k, v in INFERNO_TYPE_NAMES.items()},
        },
        "players": player_metas,
        "weapons": weapon_lookup,
        "places": place_lookup,
        "rounds": rounds_data,
        "skipped_rounds": skipped,
    }
    if verbose:
        print(f"done       : {len(rounds_data)} rounds"
              + (f" ({len(skipped)} skipped: {[s['id'] for s in skipped]})" if skipped else "")
              + f" in {time.perf_counter() - t0:.1f}s")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# lookups / bomb carrier
# ──────────────────────────────────────────────────────────────────────────────


def _build_lookups(df, events: dict, include_inventory: bool) -> tuple[dict, dict]:
    weapons: set[str] = set()
    places: set[str] = set()

    for w in df["weapon_name"].dropna().unique():
        if str(w):
            weapons.add(str(w))
    for p in df["last_place_name"].dropna().unique():
        places.add(str(p))
    if include_inventory:
        # inventory is a list-valued column; hash as tuples so it can be deduped
        inv_col = df["inventory"].map(lambda x: tuple(x) if isinstance(x, list) else None)
        for inv in inv_col.dropna().unique():
            for item in inv:
                weapons.add(WEAPON_NAME_NORMALISE.get(str(item), str(item)))

    for name in ("player_death", "player_hurt", "weapon_fire"):
        edf = events.get(name)
        if edf is None or isinstance(edf, list) or "weapon" not in getattr(edf, "columns", []):
            continue
        for w in edf["weapon"].dropna().unique():
            if str(w):
                weapons.add(_normalise_weapon_name(str(w)))

    return (
        {n: i for i, n in enumerate(sorted(weapons))},
        {n: i for i, n in enumerate(sorted(places))},
    )


def _bomb_carrier(parser: DemoParser, sampled: list[int], verbose: bool) -> dict:
    """tick -> (x, y, z, round) for whoever is holding the C4."""
    hi = max(sampled)
    lo = min(sampled)
    try:
        inv = parser.parse_ticks(
            wanted_props=["inventory", "X", "Y", "Z", "total_rounds_played"],
            ticks=list(range(lo, hi + 1)),
        )
    except Exception as exc:  # pragma: no cover
        if verbose:
            print(f"  bomb carrier scan failed: {exc}")
        return {}
    out: dict[int, tuple[float, float, float, int]] = {}
    for tick, inventory, x, y, z, rid in zip(
        inv["tick"], inv["inventory"], inv["X"], inv["Y"], inv["Z"],
        inv["total_rounds_played"],
    ):
        if isinstance(inventory, list) and "C4 Explosive" in inventory:
            out[int(tick)] = (float(x), float(y), float(z), int(rid))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# per-round assembly
# ──────────────────────────────────────────────────────────────────────────────


def _build_round(
    *,
    rid: int,
    sub,
    sampled: list[int],
    round_start_tick: int,
    round_start_time: float,
    round_end_df,
    player_map: dict,
    player_metas: list,
    weapon_lookup: dict,
    place_lookup: dict,
    events: dict,
    grenades_df,
    bomb_carrier: dict,
    side: dict | None,
    include_inventory: bool,
    include_spotted: bool,
) -> dict:
    uniq = sub.drop_duplicates(subset="tick").sort_values("tick")
    ticks_list = [int(t) for t in uniq["tick"]]
    tick_arr = np.asarray(ticks_list, dtype=np.int64)
    round_seconds = [float(g - round_start_time) for g in uniq["game_time"]]
    n = len(ticks_list)

    teams: list[str] = []
    df_first = sub[sub["tick"] == ticks_list[0]]
    team_by_idx: dict[int, str] = {}
    for sid, tn in zip(df_first["steamid"], df_first["team_num"]):
        idx = player_map.get(sid)
        if idx is not None and tn == tn and int(tn) in TEAM_MAP:
            team_by_idx[idx] = TEAM_MAP[int(tn)]
    teams = [team_by_idx.get(i, "?") for i in range(len(player_metas))]

    enemy_sids: dict[str, set[str]] = {}
    for sid in player_map:
        i = player_map[sid]
        mine = teams[i] if i < len(teams) else "?"
        enemy_sids[sid] = {
            s for s in player_map if teams[player_map[s]] != mine and mine != "?"
        }

    # ── per-player columns ──────────────────────────────────────────────────
    players_data: list[dict] = []
    missing: list[list[int]] = []
    for sid in player_map:
        pdf = sub[sub["steamid"] == sid].sort_values("tick").set_index("tick")
        if len(pdf) != n:
            pdf = pdf.reindex(tick_arr)
            miss = [i for i in range(n) if pdf["steamid"].isna().iloc[i]]
            for i in miss:
                missing.append([i, player_map[sid]])

        def col(name, default=np.nan):
            if name not in pdf.columns:
                return np.full(n, default)
            v = pdf[name].to_numpy()
            return v

        rec = {
            "x": _numcol(pdf, "X"),
            "y": _numcol(pdf, "Y"),
            "z": _numcol(pdf, "Z"),
            "yaw": _numcol(pdf, "yaw"),
            "pitch": _numcol(pdf, "pitch"),
            "v": _numcol(pdf, "velocity"),
            "vx": _numcol(pdf, "velocity_X"),
            "vy": _numcol(pdf, "velocity_Y"),
            "vz": _numcol(pdf, "velocity_Z"),
            "hp": _ints(col("health")),
            "armor": _ints(col("armor")),
            "helmet": _bools(np.nan_to_num(col("has_helmet"), nan=0)),
            "defuser": _bools(np.nan_to_num(col("has_defuser"), nan=0)),
            "alive": _bools(np.nan_to_num(col("is_alive"), nan=0)),
            "on_ground": _bools(_flags(col("CCSPlayerPawn.m_fFlags")) & 1),
            # v7: two *different* crouch quantities, both stored on purpose.
            #   crouch    - continuous 0..1.  This is the camera/eye ramp: 18 HU
            #               of travel spread over ~13 ticks (203 ms measured,
            #               matching Valve's documented 200 ms).
            #   crouching - FL_DUCKING, i.e. the collision hull has actually
            #               snapped 72 -> 54.  The hull is a STEP, not a ramp:
            #               "hull maxs.z < 60" agrees with this bit on
            #               100.0000% of 727,649 alive samples, and its height
            #               only ever takes the two values 72 and 54.
            # Drawing a smoothly shrinking hitbox would be wrong; drawing a
            # smoothly descending eye is right.  Hence both.
            "crouch": _numcol0(pdf, DUCK_PROP),
            "crouching": _bools(_flags(col("CCSPlayerPawn.m_fFlags")) & 2),
            "weapon": [
                weapon_lookup.get(_normalise_weapon_name(str(w)), -1) if str(w) else -1
                for w in col("weapon_name")
            ],
            "place": [place_lookup.get(str(p), -1) for p in col("last_place_name")],
            "flash": _numcol(pdf, "flash_duration"),
            "flash_alpha": _numcol(pdf, "flash_max_alpha"),
        }
        if include_inventory:
            rec["inventory"] = [
                sorted({
                    weapon_lookup.get(WEAPON_NAME_NORMALISE.get(str(it), str(it)), -1)
                    for it in (inv if isinstance(inv, list) else [])
                } - {-1})
                for inv in col("inventory")
            ]
        if include_spotted:
            ese = enemy_sids.get(sid, set())
            rec["spotted"] = [_spot(v, ese, player_map) for v in col("approximate_spotted_by")]
        players_data.append(rec)

    # ── shots / footsteps per sample window ─────────────────────────────────
    fire_counts, sound_events = _sound_events(
        events.get("weapon_fire"), rid, ticks_list, round_start_tick, player_map, weapon_lookup, "fire"
    )
    foot_counts, foot_events = _sound_events(
        events.get("player_footstep"), rid, ticks_list, round_start_tick, player_map, weapon_lookup, "footstep"
    )
    jump_counts, jump_events = _sound_events(
        events.get("player_jump"), rid, ticks_list, round_start_tick, player_map, weapon_lookup, "jump"
    )
    sound_events.extend(foot_events)
    sound_events.extend(jump_events)
    sound_events.sort(key=lambda e: (e["t"], e["ty"]))
    for i, sid in enumerate(player_map):
        players_data[i]["shots"] = fire_counts.get(sid, [0] * n)
        players_data[i]["footsteps"] = foot_counts.get(sid, [0] * n)
        players_data[i]["jump"] = jump_counts.get(sid, [0] * n)
        players_data[i]["land"] = _landing_counts(
            players_data[i]["alive"], players_data[i]["on_ground"]
        )

    # ─ events ──────────────────────────────────────────────────────────────
    kills = _kills(events.get("player_death"), rid, player_map, weapon_lookup,
                   round_start_tick)
    damage = _damage(events.get("player_hurt"), rid, player_map, weapon_lookup,
                     round_start_tick)
    bomb_events = _bomb_events(events, rid, round_start_time, round_start_tick)
    grenades = _grenades(grenades_df, ticks_list, player_map)

    bomb_planted_time = next((b["s"] for b in bomb_events if b["e"] == "planted"), None)
    smoke_iv, inferno_iv = _projectile_intervals(events, rid, round_start_time,
                                                 round_start_tick)

    # ─ fire / smoke for this round ─────────────────────────────────────────
    fire: list[dict] = []
    smoke: list[dict] = []
    if side is not None:
        lo, hi = ticks_list[0], ticks_list[-1]
        fire = [
            {
                "entity_id": f["entity_id"],
                "inferno_type": f["inferno_type"],
                "type_name": INFERNO_TYPE_NAMES.get(f["inferno_type"], "unknown"),
                "origin": _match_origin(inferno_iv, f["entity_id"], f["first_tick"]),
                "first_tick": f["first_tick"],
                "last_tick": f["last_tick"],
                "changes": f["changes"],
            }
            for f in side["infernos"]
            if lo <= f["first_tick"] <= hi
        ]
        smoke = [
            {
                "entity_id": s["entity_id"],
                "detonation": _match_origin(smoke_iv, s["entity_id"], s["first_tick"]),
                "first_tick": s["first_tick"],
                "last_tick": s["last_tick"],
                "capacity": side["voxel_capacity"],
                "changes": s["changes"],
            }
            for s in side["smokes"]
            if lo <= s["first_tick"] <= hi
        ]

    return {
        "id": rid,
        "teams": teams,
        "start_tick": round_start_tick,
        "winner": _end_value(round_end_df, round_start_tick, "winner"),
        "end_reason": _end_value(round_end_df, round_start_tick, "reason"),
        "bomb_planted_time": bomb_planted_time,
        "ticks": ticks_list,
        "round_seconds": round_seconds,
        "bomb_planted": _bools(uniq["is_bomb_planted"].to_numpy()),
        "bomb_dropped": _bools(uniq["is_bomb_dropped"].to_numpy()),
        "bomb_position": _bomb_positions(ticks_list, bomb_carrier, rid),
        "missing_samples": missing,
        "players": players_data,
        "events": {
            "kills": kills,
            "damage": damage,
            "bomb": bomb_events,
            "smokes": smoke_iv,
            "infernos": inferno_iv,
            "grenades": grenades,
            "sound": sound_events,
        },
        "fire": fire,
        "smoke": smoke,
    }


def _numcol(pdf, name: str) -> list:
    if name not in pdf.columns:
        return [0.0] * len(pdf)
    return _f32_or_f64(pdf[name].to_numpy())


def _flags(a) -> np.ndarray:
    """Pawn flag word -> int64.  The column arrives as float64 and is NaN when
    the player has no pawn at all (dead and respawning); NaN becomes 0, i.e.
    "no flags", which reads as airborne and not ducking."""
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0).astype(np.int64)


def _landing_counts(alive, on_ground) -> list[int]:
    """Sample indices where a player stops being airborne.

    CS2 has no landing event, so this is derived from the ground-contact column:
    sample ``i`` counts when the player was alive and airborne at ``i-1`` and is
    alive and on the ground at ``i``.  Exact at the default per-tick sampling;
    at a coarse ``--interval`` a short hop can start and end between two samples
    and is not seen.
    """
    n = len(on_ground)
    out = [0] * n
    for i in range(1, n):
        if alive[i] and alive[i - 1] and on_ground[i] and not on_ground[i - 1]:
            out[i] = 1
    return out


def _spot(spotted_list, enemies: set[str], player_map: dict) -> list[int]:
    if not isinstance(spotted_list, list):
        return []
    out = set()
    for s in spotted_list:
        cs = _num(s)
        if cs in enemies and cs in player_map:
            out.add(player_map[cs])
    return sorted(out)


def _end_value(round_end_df, start_tick: int, column: str) -> str:
    """First round_end after this round's freeze_end carries this round's result."""
    if _empty_or_df(round_end_df) or column not in getattr(round_end_df, "columns", []):
        return ""
    rows = round_end_df[round_end_df["tick"] > start_tick]
    if rows.empty:
        return ""
    return str(rows.iloc[0][column])


def _sound_events(event_df, rid, ticks_list, round_start_tick, player_map, weapon_lookup, kind):
    n = len(ticks_list)
    counts: dict[str, list[int]] = {sid: [0] * n for sid in player_map}
    out: list[dict] = []
    if _empty_or_df(event_df):
        return counts, out
    sub = event_df[event_df["total_rounds_played"] == rid]
    if sub.empty:
        return counts, out
    tk = np.asarray(ticks_list, dtype=np.int64)
    et = sub["tick"].to_numpy(dtype=np.int64)
    si = np.searchsorted(tk, et, side="left")
    ok = (et > round_start_tick) & (et <= ticks_list[-1]) & (si < n)
    sids = sub["user_steamid"].map(_num).to_numpy()
    weapons = sub["weapon"].to_numpy() if "weapon" in sub.columns else None
    for j in np.flatnonzero(ok):
        sid = str(sids[j])
        if sid not in player_map:
            continue
        counts[sid][int(si[j])] += 1
        if kind == "fire":
            out.append({
                "t": int(et[j]),
                "ty": "fire",
                "p": player_map[sid],
                "w": weapon_lookup.get(_normalise_weapon_name(str(weapons[j])), -1) if weapons is not None else -1,
                "sil": bool(sub["silenced"].to_numpy()[j]) if "silenced" in sub.columns else False,
            })
        else:
            out.append({"t": int(et[j]), "ty": kind, "p": player_map[sid]})
    return counts, out


def _round_events(event_df, rid: int, round_start_tick: int):
    """Rows that belong to *this* round: right round number **and** after its freeze.

    Filtering on ``total_rounds_played`` alone is not enough.  The pre-match
    warmup also reports round 0, and platforms that record it (5E does: a real
    Inferno demo has 71 warmup kills against 115 in the match itself) put
    hundreds of kills in that bucket -- all of which used to be credited to
    round 0 of the match.  A round's own ``round_freeze_end`` is its real
    origin, and nothing before it can belong to it: players cannot shoot
    during freeze time, so the only events there are warmup ones.

    Returns ``None`` when there is nothing to iterate, so callers can keep the
    single ``if ... is None: return []`` shape.
    """
    if _empty_or_df(event_df):
        return None
    sub = event_df[event_df["total_rounds_played"] == rid]
    if sub.empty:
        return None
    keep = sub["tick"].to_numpy(dtype=np.int64) > int(round_start_tick)
    return sub[keep]


def _kills(event_df, rid, player_map, weapon_lookup, round_start_tick: int = 0) -> list[dict]:
    sub = _round_events(event_df, rid, round_start_tick)
    if sub is None:
        return []
    out = []
    for row in sub.itertuples(index=False):
        d = row._asdict() if hasattr(row, "_asdict") else None
        get = (lambda k, dv=None: d.get(k, dv)) if d else (lambda k, dv=None: getattr(row, k, dv))
        attacker = player_map.get(_num(get("attacker_steamid")), -1)
        assister = player_map.get(_num(get("assister_steamid")), -1)
        out.append({
            "t": int(get("tick")),
            "a": attacker,
            "v": player_map.get(_num(get("user_steamid")), -1),
            "as": assister if assister != attacker else -1,
            "w": weapon_lookup.get(_normalise_weapon_name(str(get("weapon", ""))), -1),
            "hs": bool(get("headshot", False)),
            "ts": bool(get("thrusmoke", False)),
            "ab": bool(get("attackerblind", False)),
            "ai": bool(get("attackerinair", False)),
            "af": bool(get("assistedflash", False)),
            "dmg": int(get("dmg_health", 0) or 0),
        })
    out.sort(key=lambda k: k["t"])
    return out


def _damage(event_df, rid, player_map, weapon_lookup, round_start_tick: int = 0) -> list[dict]:
    sub = _round_events(event_df, rid, round_start_tick)
    if sub is None:
        return []
    out = []
    for row in sub.itertuples(index=False):
        d = row._asdict() if hasattr(row, "_asdict") else None
        get = (lambda k, dv=None: d.get(k, dv)) if d else (lambda k, dv=None: getattr(row, k, dv))
        out.append({
            "t": int(get("tick")),
            "a": player_map.get(_num(get("attacker_steamid")), -1),
            "v": player_map.get(_num(get("user_steamid")), -1),
            "hp": int(get("dmg_health", 0) or 0),
            "w": weapon_lookup.get(_normalise_weapon_name(str(get("weapon", ""))), -1),
        })
    out.sort(key=lambda x: x["t"])
    return out


def _bomb_events(events: dict, rid: int, round_start_time: float,
                 round_start_tick: int = 0) -> list[dict]:
    out = []
    for key, label in (("bomb_planted", "planted"), ("bomb_exploded", "exploded"),
                       ("bomb_defused", "defused")):
        df = events.get(key)
        sub = _round_events(df, rid, round_start_tick)
        if sub is None:
            continue
        for row in sub.itertuples(index=False):
            out.append({
                "t": int(row.tick),
                "e": label,
                "s": float(row.game_time) - round_start_time,
            })
    out.sort(key=lambda b: b["t"])
    return out


def _grenades(grenades_df, ticks_list: list[int], player_map: dict) -> list[dict]:
    if grenades_df is None or len(grenades_df) == 0:
        return []
    tk = np.asarray(ticks_list, dtype=np.int64)
    keep = grenades_df[grenades_df["tick"].isin(ticks_list)]
    if keep.empty:
        return []
    keep = keep[keep["x"].notna() & keep["y"].notna() & keep["z"].notna()]
    out = []
    for row in keep.itertuples(index=False):
        out.append({
            "t": int(row.tick),
            "ty": str(row.grenade_type),
            "eid": int(row.grenade_entity_id),
            "th": player_map.get(_num(row.steamid), -1),
            "x": float(row.x),
            "y": float(row.y),
            "z": float(row.z),
        })
    out.sort(key=lambda g: g["t"])
    return out


def _projectile_intervals(events: dict, rid: int, round_start_time: float,
                          round_start_tick: int = 0) -> tuple[list, list]:
    """Smoke / inferno lifetime intervals, paired from detonate + expire events.

    These carry the **detonation position** — the origin a smoke's voxel grid is
    relative to.  Without it the voxels cannot be placed in the world:
    `m_vSmokeDetonationPos` on the entity itself is mostly [0,0,0] and is not
    usable, so the event position is the only source.

    Both the event tick and the round-relative second are kept.  The entity
    ticks the side-channel reports are offset from event ticks by a per-demo
    constant, so instances are matched to intervals by entity id + nearest tick.

    Pairing on entity id alone is not safe (ids get recycled).  An implausible
    gap leaves the interval unclosed or drops it, rather than mis-pairing.
    """
    smokes: list[dict] = []
    infernos: list[dict] = []
    open_smoke: dict[int, dict] = {}
    open_inferno: dict[int, dict] = {}

    def begin(row) -> dict:
        return {
            "eid": int(row.entityid),
            "t0": int(row.tick),
            "s0": float(row.game_time) - round_start_time,
            "x": float(row.x), "y": float(row.y), "z": float(row.z),
            "t1": None, "s1": None,
        }

    for name, df, table, live in (
        ("smokegrenade_detonate", events.get("smokegrenade_detonate"), smokes, open_smoke),
        ("smokegrenade_expired", events.get("smokegrenade_expired"), smokes, open_smoke),
        ("inferno_startburn", events.get("inferno_startburn"), infernos, open_inferno),
        ("inferno_expire", events.get("inferno_expire"), infernos, open_inferno),
    ):
        if _empty_or_df(df) or rid not in set(df["total_rounds_played"].tolist()):
            continue
        starting = name.endswith("_detonate") or name.endswith("_startburn")
        for row in df[df["total_rounds_played"] == rid].sort_values("tick").itertuples(index=False):
            # 热身里的烟 / 火也不属于 round 0：它们会在回合开场凭空出现在地图上
            if int(row.tick) <= int(round_start_tick):
                continue
            eid = int(row.entityid)
            if starting:
                live[eid] = begin(row)
            else:
                iv = live.pop(eid, None)
                if iv is not None and 0 < int(row.tick) - iv["t0"] < MAX_PROJECTILE_LIFETIME_TICKS:
                    iv["t1"] = int(row.tick)
                    iv["s1"] = float(row.game_time) - round_start_time
                    table.append(iv)

    # still smoking / still burning when the round ended
    smokes.extend(open_smoke.values())
    infernos.extend(open_inferno.values())
    smokes.sort(key=lambda v: v["t0"])
    infernos.sort(key=lambda v: v["t0"])
    return smokes, infernos


def _match_origin(intervals: list, eid: int, first_tick: int, window: int = 4096):
    """Detonation position of the interval for `eid` closest in time to `first_tick`.

    `first_tick` is an entity tick and intervals are keyed on event ticks; the
    two differ by a per-demo constant, so "closest" is the right test.  Returns
    None rather than a wrong position when nothing falls inside `window`.
    """
    best = None
    best_d = None
    for iv in intervals:
        if iv["eid"] != eid:
            continue
        d = abs(iv["t0"] - first_tick)
        if best_d is None or d < best_d:
            best_d, best = d, iv
    if best is None or best_d is None or best_d > window:
        return None
    return [best["x"], best["y"], best["z"]]


def _bomb_positions(ticks_list: list[int], bomb_carrier: dict, rid: int) -> list:
    carrier_ticks = sorted(t for t, v in bomb_carrier.items() if v[3] == rid)
    if not carrier_ticks:
        return [None] * len(ticks_list)
    out = []
    ci = 0
    last = None
    for t in ticks_list:
        while ci < len(carrier_ticks) and carrier_ticks[ci] <= t:
            x, y, z, _ = bomb_carrier[carrier_ticks[ci]]
            last = [float(x), float(y), float(z)]
            ci += 1
        out.append(list(last) if last else None)
    return out