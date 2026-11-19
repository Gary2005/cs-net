"""
Using what you saved — the accessors you actually want.

`codec.load` gives back a plain dict, so you can always walk it by hand.  This
module is the layer on top: pull a round, stack players into a tensor, and turn
the fire/smoke change-lists back into geometry at any tick.

    from demo_extract import codec, recipes

    data  = codec.load("match.csz")
    rnd   = recipes.round_by_id(data, 7)

    P     = recipes.player_tensor(rnd)          # (T, P, F) float32
    names = recipes.PLAYER_FEATURES             # what the F axis means
    mask  = recipes.alive_mask(rnd)             # (T, P) bool

    # v6 files also carry ground contact and the air events.  These are the
    # columns a model needs to decide for itself whether a step was audible.
    G     = recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V6)   # (T, P, 19)
    og    = recipes.on_ground_mask(rnd)         # (T, P) bool
    jmp   = recipes.jump_mask(rnd)              # (T, P) int, takeoffs
    lnd   = recipes.land_mask(rnd)              # (T, P) int, touchdowns

    # v7 files also carry the crouch.  Note the two are different quantities:
    # `crouch` is the continuous camera ramp, `crouching` is the FL_DUCKING
    # bit, i.e. the collision hull has snapped 72 -> 54.
    C     = recipes.crouch_amount(rnd)          # (T, P) float32 0..1
    cd    = recipes.crouching_mask(rnd)         # (T, P) bool
    eye   = recipes.eye_height(rnd)             # (T, P) float32, HU above feet

    s     = recipes.tick_sample(data, 7, rnd["ticks"][1200])
    s["players"]        # (10, F)  that tick only
    s["on_ground"]      # (10,) bool
    s["crouch"]         # (10,) float32 0..1
    s["eye_height"]     # (10,) float32 HU above the feet
    s["fire"]["points"] # (N, 3)   world coordinates, HU
    s["smoke"]["grids"] # (S, 32, 32, 32) bool occupancy

Units: positions and velocities are Hammer units (1 HU = 1.905 cm).  Multiply by
`HU_TO_METERS` (0.0254, matching cs-net's SCALE) to get metres.

Note on ticks: samples are whatever you asked the extractor for.  At the default
(every tick) consecutive samples are 1/64 s apart; at `--interval 0.25` they are
0.25 s apart.  `round["ticks"]` is the authority — never assume a stride.
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Iterable, Sequence

import numpy as np

from . import voxel

HU_TO_METERS = 0.0254  # cs-net's SCALE: 1 Hammer unit in metres

# The F axis of player_tensor(), in order.  These are exactly the 16 columns
# cs-net v2 consumes, so a v6 file still feeds the old pipeline unchanged.
PLAYER_FEATURES = (
    "x", "y", "z",              # world position, HU
    "yaw", "pitch",             # view angles, degrees
    "vx", "vy", "vz",           # velocity, HU/s
    "hp", "armor",              # 0..100
    "flash", "flash_alpha",     # blind remaining seconds / max alpha
    "weapon", "place",          # indices into data["weapons"] / data["places"]
    "shots", "footsteps",       # counts inside this sample window
)
# Added in format v6.  Ground contact plus the two discrete air events.  Feed
# these next to the 16 above and the model can work out for itself whether a
# movement step was loud (on the ground and fast) or silent.
GROUND_FEATURES = (
    "on_ground",                # 1 = touching the world, 0 = airborne
    "jump",                     # takeoff, from the player_jump event
    "land",                     # touchdown, derived from on_ground
)
PLAYER_FEATURES_V6 = PLAYER_FEATURES + GROUND_FEATURES
# Added in format v7.  The continuous crouch amount, 0 = standing, 1 = fully
# crouched.  Kept out of the v6 tuple so the 19-column tensor stays bit-stable,
# and because the two crouch quantities the demo carries are *not* the same
# thing -- see `crouching_mask` and `eye_height`.
CROUCH_FEATURES = (
    "crouch",                   # 0..1, continuous
)
PLAYER_FEATURES_V7 = PLAYER_FEATURES_V6 + CROUCH_FEATURES

# Standing eye height above the pawn origin, in Hammer units.
#
# This is the one number in the whole crouch chain that the demo does not state
# authoritatively, so it is named and parameterised rather than inlined.
# What the demo *does* pin down exactly:
#   * `m_flDuckOffset == -18.0 * crouch` on 727,649 alive samples (31
#     exceptions, 0.0043%, all single-tick death/respawn glitches);
#   * the collision hull `m_vecMaxs.z` only ever takes two values, 72 and 54,
#     so the 18 HU of travel is corroborated independently of the offset.
# The absolute value comes from outside.  Valve's documented CS2 figure is
# 64.0938 (PLAN.md section 4.4).  The only candidate in the demo,
# `CCSPlayerPawn.m_vecZ`, has mode exactly 64.000 -- but it reads garbage while
# airborne (median 13.1) and ~18% of grounded rows fall outside [60, 65], so it
# is not trustworthy enough to prefer.  The two differ by 0.0938 HU = 2.4 mm,
# which is invisible at any zoom this renderer supports.
STANDING_EYE_HEIGHT_HU = 64.0938
CROUCH_EYE_DROP_HU = 18.0
# Channels in PLAYER_FEATURES that are lengths in Hammer units.
HU_FEATURES = ("x", "y", "z", "vx", "vy", "vz")


# ── rounds and ticks ─────────────────────────────────────────────────────────


def round_by_id(data: dict, round_id: int) -> dict:
    """The round with this `id`, or raise `KeyError`.

    A round that could not be extracted is **not in the file at all** — it is
    not an empty round, and it does not shift the other rounds' ids.  It is
    listed in ``data["skipped_rounds"]`` as ``{"id": ..., "reason": ...}``, and
    the error raised here repeats that reason when it has one.

    ```python
    recipes.round_by_id(data, 7)
    # KeyError: round 7 was skipped during extraction:
    #           no round_freeze_end, so the round has no time origin
    ```

    Use :func:`round_ids` to see what is actually available.
    """
    for r in data["rounds"]:
        if r["id"] == round_id:
            return r
    why = next((s for s in skipped_rounds(data) if s["id"] == round_id), None)
    if why is not None:
        raise KeyError(f"round {round_id} was skipped during extraction: {why['reason']}")
    raise KeyError(f"no round {round_id} in this file; it has {round_ids(data)}")


def round_ids(data: dict) -> list[int]:
    """Ids of the rounds actually present, ascending.

    This is the list `round_by_id` accepts.  It is **not** necessarily
    ``range(n)``: a skipped round leaves a gap, which is why you look a round up
    by id instead of by list index.
    """
    return [r["id"] for r in data["rounds"]]


def skipped_rounds(data: dict) -> list[dict]:
    """``[{"id": int, "reason": str}, ...]`` for rounds that could not be extracted.

    Empty on a clean demo.  Accepts files written before this became a list of
    dicts (when it was a bare list of ids) and normalises them.
    """
    out: list[dict] = []
    for s in data.get("skipped_rounds", []) or []:
        out.append(s if isinstance(s, dict) else {"id": s, "reason": "unspecified"})
    return out


def ticks_of(rnd: dict) -> np.ndarray:
    """Sampled ticks of a round, as int64."""
    return np.asarray(rnd["ticks"], dtype=np.int64)


def seconds_of(rnd: dict) -> np.ndarray:
    """Round-relative seconds of each sample.  `seconds_of(r)[0]` is always 0."""
    return np.asarray(rnd["round_seconds"], dtype=np.float64)


def sample_index(rnd: dict, tick: int) -> int:
    """Index of `tick` in the round's sample list, or -1 if it is not sampled."""
    i = bisect.bisect_left(rnd["ticks"], tick)
    return i if i < len(rnd["ticks"]) and rnd["ticks"][i] == tick else -1


def iter_samples(rnd: dict) -> Iterable[tuple[int, int, float]]:
    """Yield ``(sample_index, tick, seconds)`` for every sample in a round."""
    for i, (t, s) in enumerate(zip(rnd["ticks"], rnd["round_seconds"])):
        yield i, t, s


# ── players ─────────────────────────────────────────────────────────────────


def _fill_nan(a: np.ndarray, method: str) -> np.ndarray:
    """Fill NaN runs along axis 0.  `method` is 'zero' or 'interp'."""
    if method == "zero":
        return np.nan_to_num(a, nan=0.0)
    if method != "interp":
        raise ValueError(f"unknown nan method {method!r}")
    out = a.copy()
    T = out.shape[0]
    flat = out.reshape(T, -1)          # one column per (player, feature) pair
    idx = np.arange(T)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = np.isnan(col)
        if not bad.any():
            continue
        good = ~bad
        if not good.any():
            col[:] = 0.0
            continue
        # np.interp clamps at both ends, which is what we want: a gap at the
        # start of a round holds the first known value instead of exploding.
        col[:] = np.interp(idx, idx[good], col[good])
    return out


# NOTE: there used to be an `audible_movement()` helper here, plus the
# WALK_SPEED_RATIO / RUN_SPEED_*_HU constants it was built from.  It is gone on
# purpose.  The plateaus it was derived from are real measurements (per-weapon
# run speeds at 202..252 HU/s, shift-walk ratio a median of 0.515), but the
# threshold itself -- the midpoint of the resulting gap -- is a number we made
# up, not something the game or the data states.  Baking an invented constant
# into the API is worse than handing over the inputs and letting the model (or
# the reader) decide, so what remains is the raw material: velocity, on_ground,
# jump, land.  The measurements live on as prose in README section 8.1.


def on_ground_mask(rnd: dict) -> np.ndarray:
    """``(T, P)`` bool — was this player touching the ground at this sample.

    This is the raw flag, so it says nothing about being alive: a dead player
    whose ragdoll is resting on the floor reports ``True``, and the only rows
    that report ``False`` while dead are the ones where the pawn is gone
    entirely (mid-respawn).  Combine with :func:`alive_mask` when you want "an
    alive player standing on something".  Only present in format v6 or later —
    older files raise `KeyError`.
    """
    return _mask(rnd, "on_ground").astype(bool)


def jump_mask(rnd: dict) -> np.ndarray:
    """``(T, P)`` int — jump takeoffs in this sample window.

    From the CS2 `player_jump` event, so it is exact and complete.  At the
    default per-tick sampling this is 0/1; with a coarse ``--interval`` a value
    above 1 means the window contained several jumps.
    """
    return _mask(rnd, "jump")


def land_mask(rnd: dict) -> np.ndarray:
    """``(T, P)`` int — landings in this sample window.

    CS2 has no landing event, so the extractor derives this from the
    ``on_ground`` column: an alive player who was airborne at ``i-1`` and is on
    the ground at ``i``.  Exact at the default per-tick sampling; at a coarse
    ``--interval`` a hop that starts and ends between two samples is invisible.
    """
    return _mask(rnd, "land")


def _mask(rnd: dict, key: str) -> np.ndarray:
    T, P = len(rnd["ticks"]), len(rnd["players"])
    try:
        return np.asarray(
            [rnd["players"][p][key] for p in range(P)], dtype=np.int64
        ).reshape(P, T).T.copy()
    except KeyError:
        raise KeyError(
            f"this file has no `{key}` column — it was written in a format older "
            f"than cs2.demo.v6; re-extract the demo to get ground contact"
        ) from None


# ── crouch (format v7) ───────────────────────────────────────────────────────


def crouch_amount(rnd: dict) -> np.ndarray:
    """``(T, P)`` float32 — 0.0 standing, 1.0 fully crouched.

    The continuous camera/eye ramp, straight from
    ``CCSPlayerPawn.CCSPlayer_MovementServices.m_flDuckAmount``.  It is the only
    crouch netprop in CS2 that carries information: ``m_bInCrouch``,
    ``m_nCrouchState`` and ``m_nDuckTimeMsecs`` are identically zero across
    2.0M sampled rows in two demos.

    The ramp is genuinely continuous -- every tick moves, there is no plateau
    in the middle, and a full stand-to-crouch takes a median of 13 ticks
    (203 ms at 64 tick, matching Valve's documented 200 ms).  It is *not* a
    straight line: the per-tick step grows from about 0.076 to 0.082, a mild
    ease-in.  So use these values rather than fitting ``t / 0.2``.

    Only present in format v7 or later; older files raise `KeyError`.
    """
    return _col(rnd, "crouch", np.float32, "cs2.demo.v7", "the crouch amount")


def crouching_mask(rnd: dict) -> np.ndarray:
    """``(T, P)`` bool — has the collision hull actually snapped 72 -> 54.

    This is ``FL_DUCKING`` (bit 2 of the pawn flag word), and it is **not**
    ``crouch_amount > 0.5``.  The two disagree on purpose:

    * the hull is a **step** — ``m_vecMaxs.z`` only ever takes the values 72
      and 54, and "hull < 60" agrees with this bit on 100.0000% of 727,649
      alive samples;
    * the eye offset is a **ramp** — ``m_flDuckOffset`` slides continuously
      from 0 to -18.

    Use this for hitbox / collision / "can he fit through that gap", and
    :func:`eye_height` for the camera.  Drawing a smoothly shrinking hitbox
    would be wrong, because the real one does not shrink smoothly.

    Only present in format v7 or later; older files raise `KeyError`.
    """
    return _col(rnd, "crouching", bool, "cs2.demo.v7", "the crouch state")


def eye_height(rnd: dict, standing: float = STANDING_EYE_HEIGHT_HU) -> np.ndarray:
    """``(T, P)`` float32 — eye height above the pawn origin, in Hammer units.

    ``standing - 18 * crouch``, i.e. 64.0938 standing down to 46.0938 fully
    crouched.  See :data:`STANDING_EYE_HEIGHT_HU` for why the standing value is
    a parameter and where it comes from.

    Note the pawn origin is the **feet**, not the eye: the collision hull is
    ``[-16, -16, 0] .. [16, 16, 72]`` relative to it, and a crouch leaves the
    origin's Z untouched (185 of 186 vertically-stationary ramps measure a Z
    delta of exactly 0.0).
    """
    return (standing - CROUCH_EYE_DROP_HU * crouch_amount(rnd)).astype(np.float32)


def eye_position(rnd: dict, standing: float = STANDING_EYE_HEIGHT_HU) -> np.ndarray:
    """``(T, P, 3)`` float32 — world-space eye position, Hammer units.

    The feet come from the stored ``x`` / ``y`` / ``z``; only the vertical
    offset is derived.  ``(x, y, z)`` is the feet -- do not mistake it for the
    eye, which is 64 HU higher and drops 18 HU when crouching.
    """
    out = player_tensor(rnd, ("x", "y", "z")).copy()   # (T, P, 3)
    out[:, :, 2] += eye_height(rnd, standing)
    return out.astype(np.float32)


def _col(rnd: dict, key: str, dtype, introduced: str, what: str) -> np.ndarray:
    T, P = len(rnd["ticks"]), len(rnd["players"])
    try:
        return np.asarray(
            [rnd["players"][p][key] for p in range(P)], dtype=dtype
        ).reshape(P, T).T.copy()
    except KeyError:
        raise KeyError(
            f"this file has no `{key}` column — it was written in a format older "
            f"than {introduced}; re-extract the demo to get {what}"
        ) from None


def footstep_warning(data_or_round: dict) -> str | None:
    """Describe what the `footsteps` column actually holds, if it is present.

    Returns None when there is nothing to warn about.  Kept as a function so
    callers can surface it without duplicating the numbers.
    """
    rounds = data_or_round["rounds"] if "rounds" in data_or_round else [data_or_round]
    n = sum(int(np.asarray(p["footsteps"], dtype=np.int64).sum())
            for r in rounds for p in r["players"])
    if n == 0:
        return None
    return (
        f"`footsteps` holds {n} events for this data. The CS2 `player_footstep` "
        "game event fires around jump takeoffs and landings, not on ordinary "
        "running footsteps (96.9% of them are within 1s of the same player "
        "jumping, vs a 10.4% base rate). Expect roughly 50x fewer than the real "
        "number of footsteps. For movement noise use velocity together with "
        "`on_ground`; the `jump` and `land` columns cover the air events."
    )


def nan_report(data_or_round: dict) -> dict:
    """Per-feature NaN counts, so you know what needs filling before training.

    On every demo measured so far NaN appears **only** in the velocity columns,
    always as exactly two consecutive samples hitting all ten players at once — a
    server freeze, not death (dead players still report a velocity).  It is under
    1% of samples and safe to interpolate across.

    Every numeric column is scanned, not just the 16 in `PLAYER_FEATURES`, so
    columns you would otherwise forget (notably `v`, the speed magnitude, which
    is *not* one of the 16) are reported too.
    """
    rounds = data_or_round["rounds"] if "rounds" in data_or_round else [data_or_round]
    out: dict = {}
    for r in rounds:
        for p in r["players"]:
            for k, col in p.items():
                if not isinstance(col, list):
                    continue
                # list-valued columns (inventory, spotted) are lists of lists
                if col and not isinstance(col[0], (int, float)):
                    continue
                n = sum(1 for x in col if isinstance(x, float) and math.isnan(x))
                if n:
                    out[k] = out.get(k, 0) + n
    return out


def player_tensor(
    rnd: dict,
    features: Sequence[str] = PLAYER_FEATURES,
    dtype=np.float32,
    nan: str = "keep",
) -> np.ndarray:
    """``(T, P, F)`` array of player state for one round.

    T = samples in this round, P = players (index order matches
    ``data["players"]``), F = `features`.  This is the natural input for a
    sequence model; transpose to ``(P, T, F)`` if you prefer per-player tracks.

    NaN is kept by default — see :func:`nan_report`; pass ``nan="interp"`` or
    ``nan="zero"`` to fill it.  A sample the demo entirely lacks is listed in
    ``round["missing_samples"]``.
    """
    T = len(rnd["ticks"])
    P = len(rnd["players"])
    out = np.zeros((T, P, len(features)), dtype=dtype)
    for j, key in enumerate(features):
        for p, rec in enumerate(rnd["players"]):
            col = rec[key]
            out[:, p, j] = np.asarray(col, dtype=dtype)
    if nan != "keep":
        out = _fill_nan(out, nan).astype(dtype)
    return out


def player_tracks(rnd: dict, features: Sequence[str] = PLAYER_FEATURES) -> dict[str, np.ndarray]:
    """``{feature: (P, T)}`` — per-player tracks instead of one big tensor."""
    T = len(rnd["ticks"])
    P = len(rnd["players"])
    return {
        k: np.asarray([rnd["players"][p][k] for p in range(P)], dtype=np.float32).reshape(P, T)
        for k in features
    }


def alive_mask(rnd: dict) -> np.ndarray:
    """``(T, P)`` bool: was this player alive at this sample."""
    T, P = len(rnd["ticks"]), len(rnd["players"])
    return np.asarray(
        [rnd["players"][p]["alive"] for p in range(P)], dtype=bool
    ).reshape(P, T).T.copy()


def yaw_sincos(rnd: dict) -> np.ndarray:
    """``(T, P, 2)`` sin/cos of yaw.

    Feed this to a network instead of raw degrees: yaw wraps at +-180, so the
    raw value has a discontinuity a model cannot interpolate across.
    """
    yaw = np.radians(player_tensor(rnd, ("yaw",))[:, :, 0].astype(np.float64))
    return np.stack([np.sin(yaw), np.cos(yaw)], axis=-1).astype(np.float32)


def to_meters(arr: np.ndarray, features: Sequence[str] = PLAYER_FEATURES) -> np.ndarray:
    """Scale the Hammer-unit channels of a player tensor in place-safe fashion."""
    out = np.array(arr, dtype=np.float64, copy=True)
    for j, key in enumerate(features):
        if key in HU_FEATURES:
            out[..., j] *= HU_TO_METERS
    return out


def inventory_multihot(rnd: dict, n_items: int | None = None) -> np.ndarray:
    """``(T, P, n_items)`` bool — which items each player is carrying.

    `inventory` is a variable-length list of item ids, so it cannot go into a
    numeric tensor.  This is the shape a model actually wants.  Item ids index
    ``data["weapons"]`` (verified: every id in a real demo is < len(weapons)).
    A dead player carries nothing, so an all-False row is normal.

    **Pass `n_items` explicitly when you combine rounds.**  The default is
    inferred from the item ids that actually occur, and that varies per round —
    measured on one demo: 28 for most rounds, 26 for two of them, 25 for one.
    Use ``n_items=len(data["weapons"])`` for a width that is stable across the
    whole file.
    """
    T, P = len(rnd["ticks"]), len(rnd["players"])
    if n_items is None:
        n_items = 1 + max(
            (i for p in rnd["players"] for lst in p["inventory"] for i in lst), default=-1
        )
    out = np.zeros((T, P, n_items), dtype=bool)
    for p, rec in enumerate(rnd["players"]):
        for t, items in enumerate(rec["inventory"]):
            for i in items:
                out[t, p, i] = True
    return out


def spotted_multihot(rnd: dict) -> np.ndarray:
    """``(T, P, P)`` bool — ``[t, i, j]`` is "player **j** can see player **i**".

    `spotted` is a variable-length list of player indices, so like `inventory`
    it needs its own shape.  Empty everywhere means nobody was visible.

    ⚠️ The direction is the opposite of what this docstring said before.  The
    extractor stores ``spotted[i] = "the enemies that can see i"`` (API.md
    §1.6), so the loop below — ``out[t, i, j] = j in spotted[i]`` — marks
    ``[t, i, j]`` as **j sees i**, i.e. the matrix is the transpose of "i sees
    j".  Verified independently on de_inferno: among pairs where ``j`` appears
    in ``spotted[i]`` one-way (``i not in spotted[j]``), j's view direction
    points at i within 20° for 94.9% of them, while i's points at j for only
    52.2%.  So j really is the one looking.

    If you want "what can i see", use ``sp[t, :, i]`` (a column), or
    ``sp.transpose(0, 2, 1)[t, i]``.
    """
    T, P = len(rnd["ticks"]), len(rnd["players"])
    out = np.zeros((T, P, P), dtype=bool)
    for i, rec in enumerate(rnd["players"]):
        for t, seen in enumerate(rec["spotted"]):
            for j in seen:
                out[t, i, j] = True
    return out


def round_winner(rnd: dict) -> str:
    """'T' / 'CT' / '' — derived from round_end, may be empty in odd demos."""
    return rnd.get("winner", "")


# ── fire ─────────────────────────────────────────────────────────────────────


def fire_state_at(inst: dict, tick: int) -> dict | None:
    """The inferno's cell state at `tick`, or None when it is not burning.

    Cheap and exact: the extractor stored a snapshot at every sampled tick where
    the state differed, so "the last snapshot at or before `tick`" *is* the state
    (verified against demoinfocs frame by frame).
    """
    if not (inst["first_tick"] <= tick <= inst["last_tick"]):
        return None
    tk = [c["tick"] for c in inst["changes"]]
    i = bisect.bisect_right(tk, tick) - 1
    return inst["changes"][i] if i >= 0 else None


def fire_cells(state: dict) -> dict[str, np.ndarray]:
    """Explode one snapshot into ``(N, 3)`` points / normals / parents + flags."""
    n = state["count"]
    pts = np.asarray(state["pos"], dtype=np.float32).reshape(n, 3)
    nrm = np.asarray(state["normal"], dtype=np.float32).reshape(n, 3)
    par = np.asarray(state["parent"], dtype=np.float32).reshape(n, 3)
    # cell 0 is its own parent; -1 means "this cell is a root of the spread tree"
    parent_idx = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        d = np.linalg.norm(pts - par[i], axis=1)
        d[i] = np.inf  # never its own parent unless nothing else matches
        j = int(np.argmin(d))
        if d[j] < 1.0:
            parent_idx[i] = j
    return {
        "points": pts,
        "normals": nrm,
        "parents": par,
        "parent_index": parent_idx,
        "burning": np.asarray(state["burning"], dtype=bool),
    }


def active_fire(rnd: dict, tick: int) -> list[dict]:
    """Every burning inferno at `tick`, as world-space cell arrays."""
    out = []
    for inst in rnd.get("fire", []):
        st = fire_state_at(inst, tick)
        if st is None or st["count"] == 0:
            continue
        cells = fire_cells(st)
        cells["entity_id"] = inst["entity_id"]
        cells["type"] = inst["type_name"]
        cells["origin"] = inst.get("origin")
        out.append(cells)
    return out


# ── smoke ───────────────────────────────────────────────────────────────────


def smoke_grid_at(inst: dict, tick: int, solid: bool = True) -> np.ndarray | None:
    """``(32, 32, 32)`` bool occupancy of one smoke at `tick`, or None if absent.

    Index order is ``[x, y, z]`` in the smoke's own grid; use
    :func:`grid_to_world` (or `inst["detonation"]`) to place it.

    ⚠ 坐标**可能越界一格**：JOURNAL 的 keyframe seed 是原始 u8 坐标，云体贴着栅格外沿
    时会写出 `32`，而栅格只有 `0..31`。这里把越界坐标裁回边缘而不是丢掉 —— 那一格确实
    在云体上，只是这个 32³ 的栅格装不下它；丢掉会在最外层留一圈 12 HU 的空壳，让射线
    从边缘漏过去（遮挡偏"通"），裁进来最多让云在那一面厚一格（偏"挡"）。
    `solid=True` 走的是 `convex_hull_fill()`，它本来就按 `0..dim-1` 填，所以这条路
    以前不会崩；`solid=False` 直接拿边界体素索引赋值，不裁就是
    `IndexError: index 32 is out of bounds for axis 2 with size 32` —— 这条分支现在还有
    `world_voxels(solid=False)` / `smoke_points_at()` / 和浏览器对齐的测试在走
    （逐帧遮挡原本用它，后来改回 `solid=True` 了，原因见
    `training_data/obstruction.py::smoke_volumes`）。
    """
    j = voxel.journal_at(inst, tick)
    if j is None:
        return None
    if solid:
        # 直接拿栅格：`solid_voxels()` 会先吐 5000 多个 tuple 再写回数组，
        # 那趟 `set` 比凸包填充本身还贵（凸包那步现在已经进 Qhull 了）。
        return voxel.solid_mask(j)
    g = np.zeros((voxel.VOXEL_GRID_DIM,) * 3, dtype=bool)
    cells = sorted(voxel.boundary_voxels(j))
    if cells:
        a = np.asarray(cells, dtype=np.int32)
        np.clip(a, 0, voxel.VOXEL_GRID_DIM - 1, out=a)
        g[a[:, 0], a[:, 1], a[:, 2]] = True
    return g


def smoke_points_at(inst: dict, tick: int, solid: bool = True) -> np.ndarray | None:
    """``(N, 3)`` world-space voxel centres of one smoke at `tick`."""
    j = voxel.journal_at(inst, tick)
    if j is None:
        return None
    det = inst.get("detonation")
    if det is None:
        raise ValueError(
            f"smoke entity {inst['entity_id']} has no detonation position, so its "
            "voxels cannot be placed in world space"
        )
    pts = voxel.world_voxels(j, det, solid=solid)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def active_smoke(rnd: dict, tick: int, solid: bool = True) -> dict[str, Any]:
    """Every smoke alive at `tick`: grids, world points and per-smoke metadata."""
    grids, pts, meta = [], [], []
    for inst in rnd.get("smoke", []):
        g = smoke_grid_at(inst, tick, solid=solid)
        if g is None or not g.any():
            continue
        grids.append(g)
        pts.append(smoke_points_at(inst, tick, solid=solid))
        meta.append({
            "entity_id": inst["entity_id"],
            "detonation": inst.get("detonation"),
            "voxels": int(g.sum()),
        })
    return {"grids": grids, "points": pts, "meta": meta}


def grid_to_world(
    grid_xyz: np.ndarray, detonation: Sequence[float]
) -> np.ndarray:
    """``(N, 3)`` integer grid indices -> ``(N, 3)`` world positions in HU."""
    a = np.asarray(grid_xyz, dtype=np.float64)
    return a * voxel.VOXEL_WORLD_SIZE + (
        np.asarray(detonation, dtype=np.float64)
        - voxel.VOXEL_GRID_CENTER * voxel.VOXEL_WORLD_SIZE
    )


# ── the whole picture at one tick ────────────────────────────────────────────


def tick_sample(data: dict, round_id: int, tick: int, smoke_solid: bool = True) -> dict:
    """Everything the extractor knows about one tick, ready for a model or a renderer.

    Returns a dict with:
        players      (P, F) float32        `PLAYER_FEATURES` columns
        alive        (P,) bool
        velocity_ok  (P,) bool            False where vx/vy/vz were NaN
        yaw_sincos   (P, 2) float32
        on_ground    (P,) bool            v6 files only
        jump         (P,) int             v6 files only, takeoffs in this window
        land         (P,) int             v6 files only, landings in this window
        crouch       (P,) float32         v7 files only, 0..1
        crouching    (P,) bool            v7 files only, hull has snapped to 54
        eye_height   (P,) float32         v7 files only, HU above the feet
        fire         list of dicts         points/normals/parent_index/burning
        smoke        {"grids", "points", "meta"}
        meta         tick / seconds / map / round / teams
    """
    rnd = round_by_id(data, round_id)
    i = sample_index(rnd, tick)
    if i < 0:
        raise KeyError(f"tick {tick} is not a sampled tick of round {round_id}")
    P = len(rnd["players"])
    vi = [PLAYER_FEATURES.index(k) for k in ("vx", "vy", "vz")]
    vel = player_tensor(rnd)[i][:, vi]
    out = {
        "players": player_tensor(rnd, nan="interp")[i],
        "alive": alive_mask(rnd)[i],
        "velocity_ok": ~np.isnan(vel).any(axis=1),
        "yaw_sincos": yaw_sincos(rnd)[i],
        "fire": active_fire(rnd, tick),
        "smoke": active_smoke(rnd, tick, solid=smoke_solid),
        "meta": {
            "tick": tick,
            "seconds": rnd["round_seconds"][i],
            "map": data["meta"]["map"],
            "round": round_id,
            "teams": rnd["teams"],
            "winner": rnd.get("winner", ""),
            "sample_index": i,
            "samples_in_round": len(rnd["ticks"]),
            "n_players": P,
        },
    }
    if "on_ground" in rnd["players"][0]:
        out["on_ground"] = on_ground_mask(rnd)[i]
        out["jump"] = jump_mask(rnd)[i]
        out["land"] = land_mask(rnd)[i]
    if "crouch" in rnd["players"][0]:
        out["crouch"] = crouch_amount(rnd)[i]
        out["crouching"] = crouching_mask(rnd)[i]
        out["eye_height"] = eye_height(rnd)[i]
    return out


def fire_volume_at(rnd: dict, tick: int, resolution: float = 32.0) -> dict[str, Any]:
    """Voxelise every burning inferno into one occupancy grid in world space.

    `resolution` is the voxel edge in Hammer units.  Handy when you want fire as
    a spatial field rather than a point cloud.
    """
    cells = active_fire(rnd, tick)
    if not cells:
        return {"origin": None, "resolution": resolution, "grid": np.zeros((0, 0, 0), bool),
                "shape": (0, 0, 0)}
    pts = np.concatenate([c["points"][c["burning"]] if c["burning"].any() else c["points"]
                          for c in cells], axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    shape = tuple(int(math.floor((hi[k] - lo[k]) / resolution)) + 1 for k in range(3))
    grid = np.zeros(shape, dtype=bool)
    idx = np.floor((pts - lo) / resolution).astype(int)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return {"origin": lo, "resolution": resolution, "grid": grid, "shape": shape}


# ─ dataset-level helpers ────────────────────────────────────────────────────


def summary(data: dict) -> str:
    """One-paragraph description of a loaded file."""
    rounds = data["rounds"]
    n_fire = sum(len(r.get("fire", [])) for r in rounds)
    n_smoke = sum(len(r.get("smoke", [])) for r in rounds)
    n_samples = sum(len(r["ticks"]) for r in rounds)
    m = data["meta"]
    return (
        f"{m.get('demo')} on {m.get('map')}: {len(rounds)} rounds, "
        f"{len(data['players'])} players, {n_samples} samples, "
        f"{n_fire} infernos, {n_smoke} smokes "
        f"(sampling interval={m.get('interval')}, stride={m.get('stride')})"
    )


def iter_windows(
    data: dict, length: int, stride: int = 1, round_id: int | None = None
) -> Iterable[tuple[int, int, int, int]]:
    """Yield ``(round_id, start_index, stop_index, start_tick)`` for training windows.

    Windows never cross a round boundary, so a sequence is always inside one
    round's timeline.
    """
    for rnd in data["rounds"]:
        if round_id is not None and rnd["id"] != round_id:
            continue
        T = len(rnd["ticks"])
        # number of windows is T - length + 1, and zero when length > T
        for a in range(0, max(0, T - length + 1), stride):
            yield rnd["id"], a, a + length, rnd["ticks"][a]


def smoke_lifetime(rnd_or_data, entity_id: int) -> dict | None:
    """Find a smoke instance by entity id across a round or a whole file."""
    rounds = rnd_or_data["rounds"] if "rounds" in rnd_or_data else [rnd_or_data]
    for r in rounds:
        for s in r.get("smoke", []):
            if s["entity_id"] == entity_id:
                return s
    return None