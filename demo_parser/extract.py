"""
CS2 Demo Parser — Optimized V2 JSON Format.

Converts .dem files to a compact, ML-friendly JSON structure.

Key optimizations over the V1 format:
  1. Players referenced by index (0-9) instead of 64-bit steamid strings
  2. Columnar arrays (e.g. "x": [1,2,3]) instead of per-tick dicts
  3. Kill/damage/bomb events stored as a timeline, not repeated in every tick
  4. Smoke/inferno stored as {start, end} intervals, not per-tick active lists
  5. Weapon and place names mapped to integer indices with a lookup table
"""

from __future__ import annotations

import bisect
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from demoparser2 import DemoParser

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FORMAT_VERSION = "cs2.demo.v2"

TEAM_MAP = {2: "T", 3: "CT"}

# Normalise event weapon names (short/technical) to tick-data names (display).
# demoparser2 uses different conventions for events vs tick data.
WEAPON_NAME_NORMALISE: dict[str, str] = {
    # rifles
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
    # SMGs
    "mac10": "MAC-10", "mac10_txz12": "MAC-10",
    "mp9": "MP9", "mp9_txz12": "MP9",
    "mp7": "MP7", "mp7_txz12": "MP7",
    "mp5": "MP5-SD",
    "ump45": "UMP-45", "ump45_txz12": "UMP-45",
    "p90": "P90", "p90_txz12": "P90",
    "bizon": "PP-Bizon", "bizon_txz12": "PP-Bizon",
    # pistols
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
    # heavy
    "nova": "Nova", "nova_txz12": "Nova",
    "xm1014": "XM1014", "xm1014_txz12": "XM1014",
    "mag7": "MAG-7", "mag7_txz12": "MAG-7",
    "sawedoff": "Sawed-Off", "sawedoff_txz12": "Sawed-Off",
    "m249": "M249", "m249_txz12": "M249",
    "negev": "Negev", "negev_txz12": "Negev",
    # grenades
    "hegrenade": "High Explosive Grenade",
    "flashbang": "Flashbang",
    "smokegrenade": "Smoke Grenade",
    "molotov": "Molotov",
    "incgrenade": "Incendiary Grenade",
    "decoy": "Decoy Grenade",
    # equipment
    "taser": "Zeus x27",
    "c4": "C4 Explosive",
    "planted_c4": "C4 Explosive",
    # knives & misc
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

# Properties requested from the demo parser for each tick.
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
]

# Lightweight props for the global meta pass (identity, round mapping, lookups).
# The heavy PLAYER_PROPS are fetched per-round for fault isolation.
META_PROPS = [
    "steamid", "name", "team_num", "total_rounds_played",
    "game_time", "weapon_name", "inventory", "last_place_name",
    "is_bomb_planted", "is_bomb_dropped",
]


# ──────────────────────────────────────────────────────────────────────────────
# Tick sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_ticks(parser: DemoParser, interval: float = 0.5) -> list[int]:
    """
    Sample tick indices at fixed time intervals within each round.

    For each round, sample every `interval` seconds starting at round_seconds = 0.5.
    Uses nearest-neighbour matching to pick the closest actual tick.
    """
    df = parser.parse_ticks(wanted_props=["game_time", "total_rounds_played"])
    df = (
        df[["tick", "game_time", "total_rounds_played"]]
        .drop_duplicates()
        .sort_values("tick")
    )

    # Round start events
    round_starts = parser.parse_event("round_freeze_end")
    round_start_ticks = sorted(
        round_starts["tick"].to_numpy().astype(int).tolist()
    )

    start_df = df[df["tick"].isin(round_start_ticks)]
    round_start_time: dict[int, float] = {}
    for _, row in start_df.iterrows():
        round_start_time[int(row["total_rounds_played"])] = float(row["game_time"])

    all_ticks: list[int] = []

    for round_id, grp in df.groupby("total_rounds_played"):
        round_id = int(round_id)
        if round_id not in round_start_time:
            continue

        start = round_start_time[round_id]
        times = grp["game_time"].to_numpy()
        ticks = grp["tick"].to_numpy()
        round_secs = times - start

        if len(round_secs) == 0:
            continue

        t_end = round_secs[-1]
        targets = np.arange(0.5, t_end, interval)

        idx = 0
        for t in targets:
            while idx + 1 < len(round_secs) and round_secs[idx + 1] < t:
                idx += 1
            if idx + 1 < len(round_secs):
                pick = (
                    idx + 1
                    if abs(round_secs[idx + 1] - t) < abs(round_secs[idx] - t)
                    else idx
                )
            else:
                pick = idx
            all_ticks.append(int(ticks[pick]))

    return sorted(set(all_ticks))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _player_index_map(df_first_tick) -> dict:
    """Build steamid (str) → index (0–9) mapping from the first sampled tick.

    Expects the steamid column to already be converted to str.
    """
    steamids = df_first_tick["steamid"].unique()
    if len(steamids) != 10:
        raise RuntimeError(f"Expected 10 players, got {len(steamids)}")
    return {str(sid): i for i, sid in enumerate(steamids)}


def _player_metas(df_first_tick) -> list[dict]:
    """Build player metadata list (index-aligned with _player_index_map).

    Only stores identity (steamid + name).  Team is stored per-round
    because teams swap sides at halftime (round 12).
    """
    metas: list[dict] = []
    for _, row in df_first_tick.iterrows():
        metas.append({
            "steamid": str(row["steamid"]),
            "name": str(row["name"]),
        })
    return metas


def _canonical_sid(sid) -> str:
    """Convert a steamid from any representation to canonical string."""
    if isinstance(sid, str):
        return sid
    return str(int(sid))


def _build_lookups(df_all_ticks, death_events, damage_events,
                   weapon_fire_df=None) -> tuple[dict[str, int], dict[str, int]]:
    """Build weapon name → idx and place name → idx lookups.

    Normalises event weapon short-names (e.g. "ak47", "weapon_ak47") to the
    same display names used in tick data (e.g. "AK-47"), so each weapon has
    exactly one index.
    """
    weapon_names: set[str] = set()
    place_names: set[str] = set()

    # Tick data — weapon display names and inventory items
    for _, row in df_all_ticks.iterrows():
        w = row["weapon_name"]
        if w is not None and str(w) != "":
            weapon_names.add(str(w))
        place_names.add(str(row["last_place_name"]))
        inv = row["inventory"]
        if isinstance(inv, list):
            for item in inv:
                name = str(item)
                # Tick inventory uses display names, but belt items may be short names
                name = WEAPON_NAME_NORMALISE.get(name, name)
                weapon_names.add(name)

    # Events — normalise short names (including "weapon_" prefix) to display names
    for events_df in [death_events, damage_events, weapon_fire_df]:
        if events_df is not None and hasattr(events_df, "columns") and "weapon" in events_df.columns:
            for w in events_df["weapon"].unique():
                if w is not None and str(w) != "":
                    name = _normalise_weapon_name(str(w))
                    weapon_names.add(name)

    weapon_lookup = {name: i for i, name in enumerate(sorted(weapon_names))}
    place_lookup = {name: i for i, name in enumerate(sorted(place_names))}
    return weapon_lookup, place_lookup


def _build_bomb_positions(
    ticks: list[int],
    bomb_carrier: dict[int, tuple[float, float, float, int]],
    round_id: int,
) -> list[list[float] | None]:
    """
    Build a per-tick array of bomb carrier positions for a round.

    Uses a forward scan (O(N+M)) instead of backward search (O(N*M)).
    """
    carrier_ticks = sorted(
        t for t, (_, _, _, r) in bomb_carrier.items() if r == round_id
    )

    if not carrier_ticks:
        return [None] * len(ticks)

    result: list[list[float] | None] = []
    c_idx = 0
    last_pos: list[float] | None = None

    for t in ticks:
        while c_idx < len(carrier_ticks) and carrier_ticks[c_idx] <= t:
            x, y, z, _ = bomb_carrier[carrier_ticks[c_idx]]
            last_pos = [float(x), float(y), float(z)]
            c_idx += 1
        # Copy to avoid mutation issues
        result.append(list(last_pos) if last_pos else None)

    return result


def _filter_spotted(spotted_list, enemy_steamids: set[str], player_map: dict) -> list[int]:
    """Filter spotted_by to only enemy players, return as player indices."""
    if not isinstance(spotted_list, list):
        return []
    result = []
    for sid in spotted_list:
        csid = _canonical_sid(sid)
        if csid in enemy_steamids and csid in player_map:
            result.append(player_map[csid])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (continued)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_weapon_name(name: str) -> str:
    """Strip ``weapon_`` prefix and normalise event names → display names."""
    if not name:
        return name
    if name.startswith("weapon_"):
        name = name[7:]
    return WEAPON_NAME_NORMALISE.get(name, name)


def _weapon_id(name: str, weapon_lookup: dict[str, int]) -> int:
    """Look up a weapon name, normalising event short-names → display names."""
    if not name:
        return -1
    name = _normalise_weapon_name(name)
    return weapon_lookup.get(name, -1)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_demo(
    demo_path: str,
    interval: float = 0.5,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Parse a CS2 .dem file into the optimized V2 JSON format.

    Parameters
    ----------
    demo_path : str
        Path to the .dem file.
    interval : float
        Sampling interval in seconds (default 0.5).
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        The parsed match data in V2 format.  See docs/demo-json-format.md
        for the full schema.
    """
    t_start = time.perf_counter()
    parser = DemoParser(demo_path)

    # ── Header ────────────────────────────────────────────────────────────
    header = parser.parse_header()
    map_name = header.get("map_name", "unknown")

    # ── Round boundaries ──────────────────────────────────────────────────
    df_basic = parser.parse_ticks(wanted_props=["game_time", "total_rounds_played"])
    df_basic = (
        df_basic[["tick", "game_time", "total_rounds_played"]]
        .drop_duplicates()
        .sort_values("tick")
    )

    round_starts = parser.parse_event("round_freeze_end")
    round_start_ticks = sorted(
        round_starts["tick"].to_numpy().astype(int).tolist()
    )

    round_start_time: dict[int, float] = {}
    round_start_tick: dict[int, int] = {}
    start_df = df_basic[df_basic["tick"].isin(round_start_ticks)]
    for _, row in start_df.iterrows():
        rid = int(row["total_rounds_played"])
        round_start_time[rid] = float(row["game_time"])
        round_start_tick[rid] = int(row["tick"])

    rounds_df = parser.parse_event("round_end")

    # ── Build round info (winner / reason) ────────────────────────────────
    round_info: dict[int, dict[str, str]] = {}
    for _, row in rounds_df.iterrows():
        rid = None
        for i in range(len(round_start_tick)):
            if round_start_tick.get(i) is not None and row["tick"] > round_start_tick[i]:
                rid = i
        if rid is not None:
            round_info[rid] = {
                "winner": str(row["winner"]),
                "end_reason": str(row["reason"]),
            }

    # ── Sample ticks ──────────────────────────────────────────────────────
    all_ticks = sample_ticks(parser, interval)
    if not all_ticks:
        raise RuntimeError(
            "No ticks sampled — the demo may be empty or corrupted."
        )

    if verbose:
        print(f"Demo        : {demo_path}")
        print(f"Map         : {map_name}")
        print(f"Sampled ticks: {len(all_ticks)}")

    # ── Phase 1: lightweight meta pass (all ticks) ────────────────────────
    # Pull only what's needed for identity, tick→round mapping, and lookups.
    # The full PLAYER_PROPS are fetched per-round in phase 2 for fault isolation.
    df_meta = parser.parse_ticks(wanted_props=META_PROPS, ticks=all_ticks)
    df_meta["steamid"] = df_meta["steamid"].apply(lambda x: str(int(x)))

    # Player identity from the first sampled tick
    df_first_tick = df_meta[df_meta["tick"] == all_ticks[0]]
    player_map = _player_index_map(df_first_tick)
    player_metas = _player_metas(df_first_tick)

    if verbose:
        names = ", ".join(p["name"] for p in player_metas)
        print(f"Players     : {names}")

    # ── Parse events (global, separate from tick data) ────────────────────
    death_events = parser.parse_event(
        "player_death", other=["total_rounds_played", "game_time"]
    )
    damage_events = parser.parse_event(
        "player_hurt", other=["total_rounds_played", "game_time"]
    )
    weapon_fire_df = parser.parse_event(
        "weapon_fire", other=["total_rounds_played", "game_time"]
    )
    footstep_df = parser.parse_event(
        "player_footstep", other=["total_rounds_played", "game_time"]
    )

    # Build lookups from meta pass + events
    weapon_lookup, place_lookup = _build_lookups(
        df_meta, death_events, damage_events, weapon_fire_df
    )

    bomb_planted_df = parser.parse_event(
        "bomb_planted", other=["total_rounds_played", "game_time"]
    )
    bomb_exploded_df = parser.parse_event(
        "bomb_exploded", other=["total_rounds_played", "game_time"]
    )
    bomb_defused_df = parser.parse_event(
        "bomb_defused", other=["total_rounds_played", "game_time"]
    )

    smoke_inferno_events = parser.parse_events(
        [
            "smokegrenade_detonate",
            "smokegrenade_expired",
            "inferno_startburn",
            "inferno_expire",
        ],
        other=["total_rounds_played", "game_time"],
    )

    df_grenades = parser.parse_grenades(grenades=False)

    # ── Bomb carrier tracker (scan all ticks) ─────────────────────────────
    all_ticks_full = list(range(0, max(all_ticks) + 1))
    df_inv = parser.parse_ticks(
        wanted_props=["inventory", "X", "Y", "Z", "total_rounds_played"],
        ticks=all_ticks_full,
    )
    bomb_carrier: dict[int, tuple[float, float, float, int]] = {}
    for _, row in df_inv.iterrows():
        if isinstance(row["inventory"], list) and "C4 Explosive" in row["inventory"]:
            bomb_carrier[int(row["tick"])] = (
                float(row["X"]), float(row["Y"]), float(row["Z"]),
                int(row["total_rounds_played"]),
            )

    # ── Group ticks by round ──────────────────────────────────────────────
    tick_to_round: dict[int, int] = {}
    for _, row in df_meta[["tick", "total_rounds_played"]].drop_duplicates().iterrows():
        tick_to_round[int(row["tick"])] = int(row["total_rounds_played"])

    rounds_ticks: dict[int, list[int]] = defaultdict(list)
    for t in all_ticks:
        rid = tick_to_round.get(t, -1)
        if rid >= 0:
            rounds_ticks[rid].append(t)

    # ── Pre-compute team assignments per round ─────────────────────────────
    # 干净比赛里 10 名玩家应从开局就在队（team_num = 2/3）。
    # 首 tick 出现 NaN / 无效队伍 / 缺人 = 数据异常，整回合跳过，不修复不凑数。
    round_teams_map: dict[int, list[str]] = {}
    for rid in rounds_ticks:
        first_tick = min(rounds_ticks[rid])
        df_rt = df_meta[df_meta["tick"] == first_tick]
        if df_rt.empty:
            if verbose:
                print(f"Warning: no tick data for round {rid}, skipping")
            continue
        teams_for_round: list[str] = ["?"] * 10
        clean = True
        for _, row in df_rt.iterrows():
            idx = player_map.get(str(row["steamid"]), -1)
            if idx < 0:
                clean = False
                break
            tn = row["team_num"]
            if tn != tn or int(tn) not in TEAM_MAP:  # NaN 或 0/1（未分配/观战）
                clean = False
                break
            teams_for_round[idx] = TEAM_MAP[int(tn)]
        if not clean or any(t == "?" for t in teams_for_round):
            if verbose:
                print(f"Warning: round {rid} 队伍数据异常，跳过")
            continue
        round_teams_map[rid] = teams_for_round

    # ── Build per-round data ──────────────────────────────────────────────
    rounds_data: list[dict] = []
    skipped_rounds: list[int] = []
    for rid in sorted(rounds_ticks.keys()):
        ticks = sorted(rounds_ticks[rid])
        info = round_info.get(rid, {})
        start_time = round_start_time.get(rid, 0.0)
        round_teams = round_teams_map.get(rid)
        if round_teams is None:
            if verbose:
                print(f"Warning: no team info for round {rid}, skipping")
            continue

        try:
            # Phase 2: fetch full tick data for this round only
            df_round = parser.parse_ticks(
                wanted_props=PLAYER_PROPS, ticks=ticks
            )
            df_round["steamid"] = df_round["steamid"].apply(
                lambda x: str(int(x))
            )
            rd = _build_round(
                df_round=df_round,
                round_start_time=start_time,
                round_start_tick=round_start_tick.get(rid, 0),
                player_map=player_map,
                round_teams=round_teams,
                weapon_lookup=weapon_lookup,
                place_lookup=place_lookup,
                death_events=death_events,
                damage_events=damage_events,
                weapon_fire_df=weapon_fire_df,
                footstep_df=footstep_df,
                bomb_planted_df=bomb_planted_df,
                bomb_exploded_df=bomb_exploded_df,
                bomb_defused_df=bomb_defused_df,
                smoke_inferno_events=smoke_inferno_events,
                df_grenades=df_grenades,
                bomb_carrier=bomb_carrier,
                round_info=info,
            )
            rounds_data.append(rd)
        except Exception as exc:
            skipped_rounds.append(rid)
            if verbose:
                print(f"Warning: skipping round {rid} due to error: {exc}")
            continue

    # ── Assemble result ───────────────────────────────────────────────────
    result: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "map": map_name,
        "players": player_metas,
        "weapons": weapon_lookup,
        "places": place_lookup,
        "rounds": rounds_data,
    }

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(f"Rounds      : {len(rounds_data)}"
              + (f" ({len(skipped_rounds)} skipped: {skipped_rounds})"
                 if skipped_rounds else ""))
        print(f"Done in {elapsed:.1f}s")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-round builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_round(
    *,
    df_round,
    round_start_time: float,
    round_start_tick: int,
    player_map: dict,
    round_teams: list[str],
    weapon_lookup: dict[str, int],
    place_lookup: dict[str, int],
    death_events,
    damage_events,
    weapon_fire_df,
    footstep_df,
    bomb_planted_df,
    bomb_exploded_df,
    bomb_defused_df,
    smoke_inferno_events,
    df_grenades,
    bomb_carrier: dict,
    round_info: dict,
) -> dict:
    """Build columnar round data for a single round.

    Parameters
    ----------
    df_round : DataFrame
        Full PLAYER_PROPS for this round's ticks only (already pre-filtered).
    round_start_tick : int
        Tick at which the round started (round_freeze_end).
    round_teams : list[str]
        Team for each player index, e.g. ["CT","CT",...,"T","T"].
        Length = 10, order matches the header ``players`` array.
    """

    # ── Derive ticks / round_id from the already-filtered DataFrame ────────
    df_uniq = df_round.drop_duplicates(subset="tick").sort_values("tick")
    ticks_list = [int(t) for t in df_uniq["tick"]]
    round_id = int(df_round.iloc[0]["total_rounds_played"])

    # ── Tick-level arrays ─────────────────────────────────────────────────
    round_seconds = [
        float(row["game_time"] - round_start_time)
        for _, row in df_uniq.iterrows()
    ]
    bomb_planted_arr = [bool(v) for v in df_uniq["is_bomb_planted"]]
    bomb_dropped_arr = [bool(v) for v in df_uniq["is_bomb_dropped"]]

    # ── Bomb positions ────────────────────────────────────────────────────
    bomb_positions = _build_bomb_positions(ticks_list, bomb_carrier, round_id)

    # ── Player states (columnar) ──────────────────────────────────────────
    player_order = list(player_map.keys())  # steamids in index order
    players_data: list[dict] = []

    # Pre-compute enemy steamid sets for this round
    _enemy_sets: dict[str, set[str]] = {}
    for sid in player_order:
        idx = player_map.get(sid, -1)
        team = round_teams[idx] if 0 <= idx < len(round_teams) else "?"
        _enemy_sets[sid] = {
            s for s in player_order
            if round_teams[player_map.get(s, -1)] != team
        }

    # ── Sound events (weapon_fire & player_footstep) ───────────────────────
    sound_events: list[dict] = []
    num_samples = len(ticks_list)

    # Per-player counts: maps steamid → list[int] of length num_samples
    fire_counts: dict[str, list[int]] = {sid: [0] * num_samples for sid in player_order}
    foot_counts: dict[str, list[int]] = {sid: [0] * num_samples for sid in player_order}

    # Helper: assign an event tick to a sample window index
    def _sample_index(event_tick: int) -> int | None:
        """Return the sample index whose window covers *event_tick*."""
        if event_tick <= round_start_tick or event_tick > ticks_list[-1]:
            return None
        return bisect.bisect_left(ticks_list, event_tick)

    # Process weapon_fire events
    if not (isinstance(weapon_fire_df, list) or weapon_fire_df.empty):
        f_round = weapon_fire_df[weapon_fire_df["total_rounds_played"] == round_id]
        for _, row in f_round.iterrows():
            t = int(row["tick"])
            sid = str(int(row["user_steamid"]))
            if sid not in player_map:
                continue
            si = _sample_index(t)
            if si is not None:
                fire_counts[sid][si] += 1
            weapon = str(row.get("weapon", ""))
            silenced = bool(row.get("silenced", False))
            sound_events.append({
                "t":  t,
                "ty": "fire",
                "p":  player_map[sid],
                "w":  _weapon_id(weapon, weapon_lookup),
                "sil": silenced,
            })

    # Process player_footstep events
    if not (isinstance(footstep_df, list) or footstep_df.empty):
        fs_round = footstep_df[footstep_df["total_rounds_played"] == round_id]
        for _, row in fs_round.iterrows():
            t = int(row["tick"])
            sid = str(int(row["user_steamid"]))
            if sid not in player_map:
                continue
            si = _sample_index(t)
            if si is not None:
                foot_counts[sid][si] += 1
            sound_events.append({
                "t":  t,
                "ty": "footstep",
                "p":  player_map[sid],
            })

    sound_events.sort(key=lambda se: se["t"])

    for steamid in player_order:
        pdf = df_round[df_round["steamid"] == steamid].sort_values("tick")

        enemy_sids = _enemy_sets.get(steamid, set())

        # Convert numpy NaN → 0.0 (covers both Python float and np.floating)
        def _safe_float(arr, idx):
            v = arr[idx]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    return 0.0
            return float(v)

        n = len(pdf)
        x_arr   = pdf["X"].to_numpy()
        y_arr   = pdf["Y"].to_numpy()
        z_arr   = pdf["Z"].to_numpy()
        yaw_arr = pdf["yaw"].to_numpy()
        pt_arr  = pdf["pitch"].to_numpy()
        v_arr   = pdf["velocity"].to_numpy()
        vx_arr  = pdf["velocity_X"].to_numpy()
        vy_arr  = pdf["velocity_Y"].to_numpy()
        vz_arr  = pdf["velocity_Z"].to_numpy()
        hp_arr  = pdf["health"].to_numpy()
        ar_arr  = pdf["armor"].to_numpy()
        he_arr  = pdf["has_helmet"].to_numpy()
        df_arr  = pdf["has_defuser"].to_numpy()
        al_arr  = pdf["is_alive"].to_numpy()
        wn_arr  = pdf["weapon_name"].to_numpy()
        inv_arr = pdf["inventory"].to_numpy()
        fl_arr  = pdf["flash_duration"].to_numpy()
        fa_arr  = pdf["flash_max_alpha"].to_numpy()
        pl_arr  = pdf["last_place_name"].to_numpy()
        sp_arr  = pdf["approximate_spotted_by"].to_numpy()

        players_data.append({
            "x":        [_safe_float(x_arr, i) for i in range(n)],
            "y":        [_safe_float(y_arr, i) for i in range(n)],
            "z":        [_safe_float(z_arr, i) for i in range(n)],
            "yaw":      [_safe_float(yaw_arr, i) for i in range(n)],
            "pitch":    [_safe_float(pt_arr, i) for i in range(n)],
            "v":        [_safe_float(v_arr, i) for i in range(n)],
            "vx":       [_safe_float(vx_arr, i) for i in range(n)],
            "vy":       [_safe_float(vy_arr, i) for i in range(n)],
            "vz":       [_safe_float(vz_arr, i) for i in range(n)],
            "hp":       [int(hp_arr[i]) for i in range(n)],
            "armor":    [int(ar_arr[i]) for i in range(n)],
            "helmet":   [bool(he_arr[i]) for i in range(n)],
            "defuser":  [bool(df_arr[i]) for i in range(n)],
            "alive":    [bool(al_arr[i]) for i in range(n)],
            "weapon":   [_weapon_id(str(wn_arr[i]), weapon_lookup) for i in range(n)],
            "inventory": [
                [_weapon_id(str(item), weapon_lookup) for item in (inv if isinstance(inv, list) else [])]
                for inv in inv_arr
            ],
            "flash":    [float(fl_arr[i]) for i in range(n)],
            "flash_alpha": [float(fa_arr[i]) for i in range(n)],
            "place":    [place_lookup.get(str(pl_arr[i]), -1) for i in range(n)],
            "spotted":  [
                _filter_spotted(sp_arr[i], enemy_sids, player_map)
                for i in range(n)
            ],
            "shots":     fire_counts.get(steamid, [0] * n),
            "footsteps": foot_counts.get(steamid, [0] * n),
        })

    # ── Events ────────────────────────────────────────────────────────────

    # Kills
    kills: list[dict] = []
    if not (isinstance(death_events, list) or death_events.empty):
        de_round = death_events[death_events["total_rounds_played"] == round_id]
        for _, row in de_round.iterrows():
            attacker = player_map.get(str(row["attacker_steamid"]), -1)
            assister = player_map.get(str(row.get("assister_steamid", "")), -1)
            kills.append({
                "t":  int(row["tick"]),
                "a":  attacker,
                "v":  player_map.get(str(row["user_steamid"]), -1),
                "as": assister if assister != attacker else -1,
                "w":  _weapon_id(str(row["weapon"]), weapon_lookup),
                "hs": bool(row["headshot"]),
                "ts": bool(row["thrusmoke"]),
                "ab": bool(row["attackerblind"]),
                "ai": bool(row["attackerinair"]),
                "af": bool(row["assistedflash"]),
                "dmg": int(row["dmg_health"]),
            })
        kills.sort(key=lambda k: k["t"])

    # Damage
    damage: list[dict] = []
    if not (isinstance(damage_events, list) or damage_events.empty):
        dm_round = damage_events[damage_events["total_rounds_played"] == round_id]
        for _, row in dm_round.iterrows():
            damage.append({
                "t":  int(row["tick"]),
                "a":  player_map.get(str(row["attacker_steamid"]), -1),
                "v":  player_map.get(str(row["user_steamid"]), -1),
                "hp": int(row["dmg_health"]),
                "w":  _weapon_id(str(row["weapon"]), weapon_lookup),
            })
        damage.sort(key=lambda d: d["t"])

    # Bomb events
    bomb_events: list[dict] = []
    for df_src, event_label in [
        (bomb_planted_df, "planted"),
        (bomb_exploded_df, "exploded"),
        (bomb_defused_df, "defused"),
    ]:
        # parse_event may return an empty list if the event never occurs
        if isinstance(df_src, list) or (hasattr(df_src, "empty") and df_src.empty):
            continue
        sub = df_src[df_src["total_rounds_played"] == round_id]
        for _, row in sub.iterrows():
            bomb_events.append({
                "t": int(row["tick"]),
                "e": event_label,
                "s": round(
                    float(row["game_time"]) - round_start_time, 2
                ),
            })
    bomb_events.sort(key=lambda b: b["t"])

    # Grenade entities (in-flight) — stored as sparse events
    grenades: list[dict] = []
    for t in ticks_list:
        gr = df_grenades[df_grenades["tick"] == t]
        for _, row in gr.iterrows():
            if np.isnan(row["x"]) or np.isnan(row["y"]) or np.isnan(row["z"]):
                continue
            grenades.append({
                "t":  int(t),
                "ty": str(row["grenade_type"]),
                "th": player_map.get(str(row["steamid"]), -1),
                "x":  float(row["x"]),
                "y":  float(row["y"]),
                "z":  float(row["z"]),
            })

    # Smoke / inferno intervals
    smokes, infernos = _build_projectile_intervals(
        smoke_inferno_events, round_id, round_start_time
    )

    # Bomb planted time (round-relative seconds, one value)
    bomb_planted_time: float | None = None
    for be in bomb_events:
        if be["e"] == "planted":
            bomb_planted_time = be["s"]
            break

    return {
        "id": round_id,
        "teams": round_teams,
        "winner": round_info.get("winner", ""),
        "end_reason": round_info.get("end_reason", ""),
        "bomb_planted_time": bomb_planted_time,
        "ticks": ticks_list,
        "round_seconds": round_seconds,
        "bomb_planted": bomb_planted_arr,
        "bomb_dropped": bomb_dropped_arr,
        "bomb_position": bomb_positions,
        "players": players_data,
        "events": {
            "kills": kills,
            "damage": damage,
            "bomb": bomb_events,
            "grenades": grenades,
            "smokes": smokes,
            "infernos": infernos,
            "sound": sound_events,
        },
    }


def _build_projectile_intervals(
    smoke_inferno_events, round_id: int, round_start_time: float
) -> tuple[list[dict], list[dict]]:
    """Convert smoke/inferno start/end events into closed [ts, te] intervals."""
    smokes: list[dict] = []
    infernos: list[dict] = []

    active_smokes: dict[int, Any] = {}
    active_infernos: dict[int, Any] = {}

    for name, df in smoke_inferno_events:
        for _, row in df.iterrows():
            if int(row["total_rounds_played"]) != round_id:
                continue
            eid = int(row["entityid"])

            if name == "smokegrenade_detonate":
                active_smokes[eid] = row
            elif name == "smokegrenade_expired":
                if eid in active_smokes:
                    start = active_smokes.pop(eid)
                    smokes.append({
                        "ts": round(float(start["game_time"]) - round_start_time, 2),
                        "te": round(float(row["game_time"]) - round_start_time, 2),
                        "x":  float(start["x"]),
                        "y":  float(start["y"]),
                        "z":  float(start["z"]),
                    })
            elif name == "inferno_startburn":
                active_infernos[eid] = row
            elif name == "inferno_expire":
                if eid in active_infernos:
                    start = active_infernos.pop(eid)
                    infernos.append({
                        "ts": round(float(start["game_time"]) - round_start_time, 2),
                        "te": round(float(row["game_time"]) - round_start_time, 2),
                        "x":  float(start["x"]),
                        "y":  float(start["y"]),
                        "z":  float(start["z"]),
                    })

    # Don't forget still-active projectiles at round end
    for row in active_smokes.values():
        smokes.append({
            "ts": round(float(row["game_time"]) - round_start_time, 2),
            "te": None,  # still active at round end
            "x":  float(row["x"]),
            "y":  float(row["y"]),
            "z":  float(row["z"]),
        })
    for row in active_infernos.values():
        infernos.append({
            "ts": round(float(row["game_time"]) - round_start_time, 2),
            "te": None,
            "x":  float(row["x"]),
            "y":  float(row["y"]),
            "z":  float(row["z"]),
        })

    return smokes, infernos


# ──────────────────────────────────────────────────────────────────────────────
# Serialization
# ──────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types and replaces NaN/Infinity with null."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def iterencode(self, o, _one_shot=False):
        """Override to catch any NaN/Inf that slips past default()."""
        for chunk in super().iterencode(o, _one_shot):
            yield chunk


def save_demo_json(
    data: dict,
    output_path: str,
    compact: bool = False,
    compress: bool = False,
) -> Path:
    """
    Save parsed demo data to a JSON file.

    Parameters
    ----------
    data : dict
        Parsed match data from ``parse_demo()``.
    output_path : str
        Destination file path.
    compact : bool
        If True, omit indentation (smaller file).
    compress : bool
        If True, gzip-compress the output (appends ``.gz`` if not already present).

    Returns
    -------
    Path
        The actual path written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    indent = None if compact else 2

    if compress:
        json_bytes = json.dumps(
            data, ensure_ascii=False, cls=_NumpyEncoder, indent=indent
        ).encode("utf-8")
        if path.suffix != ".gz":
            path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(str(path), "wb") as f:
            f.write(json_bytes)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, cls=_NumpyEncoder, indent=indent)

    return path
