import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import torch
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)


os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
UPLOAD_DIR = ROOT_DIR / "demo_analysis" / "uploads"
OUTPUT_DIR = ROOT_DIR / "demo_analysis" / "outputs"
MODEL_ROOT = ROOT_DIR / "cs-net-models"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
ANALYSIS_JOBS: dict[str, dict[str, Any]] = {}
ANALYSIS_LOCK = threading.Lock()

VIEWER_DIR = Path(__file__).resolve().parent / "static" / "viewer"


def _has_running_jobs() -> bool:
    with ANALYSIS_LOCK:
        for job in ANALYSIS_JOBS.values():
            if job.get("status") in {"queued", "running"}:
                return True
    return False


def cleanup_runtime_artifacts(clear_state: bool = True) -> dict[str, int]:
    """Delete stale uploaded demos/results and optionally reset in-memory state."""
    removed_uploads = 0
    removed_outputs = 0

    for path in UPLOAD_DIR.glob("*"):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
            removed_uploads += 1

    for path in OUTPUT_DIR.glob("*"):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
            removed_outputs += 1

    if clear_state:
        with ANALYSIS_LOCK:
            ANALYSIS_JOBS.clear()
        ANALYSIS_CACHE.clear()

    return {
        "uploads": removed_uploads,
        "outputs": removed_outputs,
    }


def choose_default_device() -> str:
    return "cpu"


ALL_HEAD_SUBDIRS = ("alive", "nxt_kill", "nxt_death", "win_rate", "duel")
REQUIRED_HEAD_SUBDIRS = ("win_rate",)  # only win_rate is mandatory; other heads fall back


def _has_ckpt_and_yaml(path: Path) -> bool:
    """True if `path` directly contains a checkpoint (.pt/.pth) and a non-tokenizer yaml."""
    if not path.is_dir():
        return False
    has_yaml = any(
        p.suffix == ".yaml" and "tokenizer" not in p.name.lower()
        for p in path.iterdir()
    )
    has_ckpt = any(p.suffix in {".pt", ".pth"} for p in path.iterdir())
    return has_yaml and has_ckpt


def _looks_like_model_root(path: Path) -> bool:
    """A model root has (at minimum) a `win_rate/` subdir."""
    if not path.exists() or not path.is_dir():
        return False
    return all((path / sub).is_dir() for sub in REQUIRED_HEAD_SUBDIRS)


def normalize_model_root(path_str: str) -> Path | None:
    """
    Accept any of:
      - A model root containing `win_rate/` (and optionally other heads).
      - A bare legacy checkpoint dir (win_rate itself) — passed through as-is;
        the inference script auto-detects it as a single-head legacy layout.
      - A path *inside* a model root (e.g. `.../win_rate`) — normalize to its parent.
    """
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    if _looks_like_model_root(path):
        return path
    parent = path.parent
    if _looks_like_model_root(parent):
        return parent
    # Legacy: user points directly at a single-head checkpoint dir (e.g. win_rate/).
    if _has_ckpt_and_yaml(path):
        return path
    return None


def discover_model_paths() -> list[str]:
    options: list[str] = []
    if _looks_like_model_root(MODEL_ROOT):
        options.append(str(MODEL_ROOT))
    # Legacy fallback: surface MODEL_ROOT/win_rate if only that exists.
    legacy = MODEL_ROOT / "win_rate"
    if not options and _has_ckpt_and_yaml(legacy):
        options.append(str(legacy))
    return options


def choose_default_model_path(model_options: list[str]) -> str:
    if model_options:
        return model_options[0]
    if MODEL_ROOT.exists() and MODEL_ROOT.is_dir():
        return str(MODEL_ROOT)
    return ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def round_key_sorter(item: tuple[str, Any]) -> int:
    key = item[0]
    if key.isdigit():
        return int(key)
    return 10**9


def build_round_summary(round_data: dict[str, Any]) -> dict[str, Any]:
    player_data = round_data.get("player_data", [])
    final_snapshot = player_data[-1] if player_data else {}
    per_player = []

    for player, stat in final_snapshot.items():
        kill_contribution = safe_float(stat.get("kill_contribution", 0.0))
        tactical_contribution = safe_float(stat.get("tactical_contribution", 0.0))
        total = kill_contribution + tactical_contribution
        per_player.append(
            {
                "player": player,
                "kill_contribution": kill_contribution,
                "tactical_contribution": tactical_contribution,
                "total_contribution": total,
            }
        )

    per_player.sort(key=lambda x: x["total_contribution"], reverse=True)

    return {
        "final_snapshot": final_snapshot,
        "per_player": per_player,
    }


def build_team_swings(win_rate: list[dict[str, Any]], horizon: float = 5.0) -> dict[str, Any]:
    if len(win_rate) < 2:
        return {
            "largest_team1_drop_5s": None,
            "largest_team1_rise_5s": None,
        }

    largest_drop = {"delta": 0.0, "start": None, "end": None}
    largest_rise = {"delta": 0.0, "start": None, "end": None}

    for i in range(len(win_rate)):
        start_t = safe_float(win_rate[i].get("round_seconds", 0.0))
        start_wr = safe_float(win_rate[i].get("team1_win_rate", 0.0))

        j = i + 1
        while j < len(win_rate):
            end_t = safe_float(win_rate[j].get("round_seconds", 0.0))
            if end_t - start_t > horizon:
                break
            end_wr = safe_float(win_rate[j].get("team1_win_rate", 0.0))
            delta = end_wr - start_wr

            if delta < largest_drop["delta"]:
                largest_drop = {"delta": delta, "start": start_t, "end": end_t}
            if delta > largest_rise["delta"]:
                largest_rise = {"delta": delta, "start": start_t, "end": end_t}
            j += 1

    return {
        "largest_team1_drop_5s": largest_drop if largest_drop["start"] is not None else None,
        "largest_team1_rise_5s": largest_rise if largest_rise["start"] is not None else None,
    }


def build_advanced_metrics(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-player advanced metrics + a |swing|-sorted kill ranking.

    player_stats per player:
      avg_kill_opp        — mean of next_kill[idx]   across all ticks they were in
      avg_death_opp       — mean of next_death[idx]
      avg_survive_chance  — mean of alive_pred[idx]
      hard_win_rate       — kills with difficulty > 1 / hard duels they were involved in
      easy_win_rate       — kills with 0 < difficulty < 1 / easy duels they were involved in
      highlight_rate      — fraction of rounds where their total contribution >= HIGHLIGHT_THR

    A duel is "hard" for the attacker when difficulty > 1 (model thought they would lose)
    and "easy" when difficulty < 1. The victim sees the inverse — losing a fight where they
    were favored counts as a missed easy duel; losing a fight where they were the underdog
    counts as a missed hard duel. We keep the convention symmetric so each player's
    hard_win_rate + easy_win_rate is bounded in [0, 1] regardless of role.
    """
    HIGHLIGHT_THRESHOLD = 0.20

    kill_ranking: list[dict[str, Any]] = []
    agg: dict[str, dict[str, Any]] = {}

    def ensure(name: str, team_hint: str) -> dict[str, Any]:
        if name not in agg:
            agg[name] = {
                "team": team_hint,
                "kill_sum": 0.0,
                "death_sum": 0.0,
                "survive_sum": 0.0,
                "tick_n": 0,
                "hard_kills": 0,
                "hard_attempts": 0,
                "easy_kills": 0,
                "easy_attempts": 0,
                "highlights": 0,
                "rounds": 0,
            }
        elif agg[name]["team"] == "Unknown" and team_hint != "Unknown":
            agg[name]["team"] = team_hint
        return agg[name]

    for rd in rounds:
        ticks = rd.get("ticks") or []
        first_tick = ticks[0] if ticks else {}
        players_info = first_tick.get("players_info") or []
        name_to_idx: dict[str, int] = {}
        for i, p in enumerate(players_info):
            name = p.get("name")
            if name and i < 10:
                name_to_idx[name] = i

        team1_set = set(rd.get("team1_players") or [])
        team2_set = set(rd.get("team2_players") or [])

        def team_of(name: str) -> str:
            if name in team1_set:
                return "team1"
            if name in team2_set:
                return "team2"
            return "Unknown"

        for k in rd.get("kills") or []:
            attacker = k.get("killer", "Unknown")
            victim = k.get("victim", "Unknown")
            kill_ranking.append(
                {
                    "round": rd.get("round_id"),
                    "round_seconds": safe_float(k.get("round_seconds", 0.0)),
                    "attacker": attacker,
                    "victim": victim,
                    "swing": safe_float(k.get("kill_impact", 0.0)),
                    "difficulty": safe_float(k.get("difficulty", 0.0)),
                }
            )

            difficulty = safe_float(k.get("difficulty", 0.0))
            attacker_entry = ensure(attacker, team_of(attacker))
            victim_entry = ensure(victim, team_of(victim))
            if difficulty > 1.0:
                attacker_entry["hard_attempts"] += 1
                attacker_entry["hard_kills"] += 1
                victim_entry["easy_attempts"] += 1  # victim was favored but lost
            elif 0.0 < difficulty < 1.0:
                attacker_entry["easy_attempts"] += 1
                attacker_entry["easy_kills"] += 1
                victim_entry["hard_attempts"] += 1  # victim was the underdog and still lost

        for name, idx in name_to_idx.items():
            entry = ensure(name, team_of(name))
            entry["rounds"] += 1
            for tk in ticks:
                nk = tk.get("next_kill") or []
                nd = tk.get("next_death") or []
                ap = tk.get("alive_pred") or []
                if idx < len(nk):
                    entry["kill_sum"] += safe_float(nk[idx])
                if idx < len(nd):
                    entry["death_sum"] += safe_float(nd[idx])
                if idx < len(ap):
                    entry["survive_sum"] += safe_float(ap[idx])
                entry["tick_n"] += 1

        for item in rd.get("round_summary", {}).get("per_player", []):
            name = item.get("player")
            if not name:
                continue
            entry = ensure(name, team_of(name))
            if safe_float(item.get("total_contribution", 0.0)) >= HIGHLIGHT_THRESHOLD:
                entry["highlights"] += 1

    player_stats: list[dict[str, Any]] = []
    for name, a in agg.items():
        ticks_n = max(1, a["tick_n"])
        rounds_n = max(1, a["rounds"])
        hard_n = a["hard_attempts"]
        easy_n = a["easy_attempts"]
        player_stats.append(
            {
                "player": name,
                "team": a["team"],
                "avg_kill_opp": a["kill_sum"] / ticks_n,
                "avg_death_opp": a["death_sum"] / ticks_n,
                "avg_survive_chance": a["survive_sum"] / ticks_n,
                "hard_win_rate": (a["hard_kills"] / hard_n) if hard_n > 0 else 0.0,
                "easy_win_rate": (a["easy_kills"] / easy_n) if easy_n > 0 else 0.0,
                "highlight_rate": a["highlights"] / rounds_n,
                "rounds": int(a["rounds"]),
                "hard_attempts": int(hard_n),
                "easy_attempts": int(easy_n),
            }
        )

    player_stats.sort(
        key=lambda x: (-x["hard_win_rate"], -x["avg_kill_opp"], -x["highlight_rate"])
    )
    kill_ranking.sort(key=lambda x: abs(safe_float(x.get("swing", 0.0))), reverse=True)

    return {
        "kill_ranking": kill_ranking,
        "player_stats": player_stats,
    }


def build_dashboard_payload(raw_results: dict[str, Any]) -> dict[str, Any]:
    rounds = []
    errors: dict[str, Any] = {}
    player_totals: dict[str, dict[str, float]] = {}
    player_team: dict[str, str] = {}
    team1_round_wins = 0
    team2_round_wins = 0
    team1_roster: set[str] = set()
    team2_roster: set[str] = set()

    # Establish stable team identity by the first valid round roster.
    for key, val in sorted(raw_results.items(), key=round_key_sorter):
        if not key.isdigit() or not isinstance(val, dict):
            continue
        ct_players = val.get("CT_players", [])
        t_players = val.get("T_players", [])
        if ct_players and t_players:
            team1_roster = set(ct_players)
            team2_roster = set(t_players)
            break

    for key, val in sorted(raw_results.items(), key=round_key_sorter):
        if key.startswith("error_round_"):
            errors[key] = val
            continue
        if not key.isdigit() or not isinstance(val, dict):
            continue

        round_id = int(key)
        raw_win_rate = val.get("win_rate", [])
        kills = val.get("kills", [])
        ct_players = val.get("CT_players", [])
        t_players = val.get("T_players", [])
        winner_side = val.get("winner", "Unknown")
        player_data = val.get("player_data", [])
        start_inventory = val.get("start_inventory", [])

        ct_set = set(ct_players)
        t_set = set(t_players)

        team1_on_ct = len(ct_set & team1_roster) >= len(t_set & team1_roster)
        if team1_on_ct:
            round_team1_players = ct_players
            round_team2_players = t_players
        else:
            round_team1_players = t_players
            round_team2_players = ct_players

        for p in round_team1_players:
            player_team[p] = "team1"
        for p in round_team2_players:
            player_team[p] = "team2"

        if winner_side == "CT":
            winner_team = "team1" if team1_on_ct else "team2"
        elif winner_side == "T":
            winner_team = "team2" if team1_on_ct else "team1"
        else:
            winner_team = "Unknown"

        if winner_team == "team1":
            team1_round_wins += 1
        elif winner_team == "team2":
            team2_round_wins += 1

        win_rate = []
        for wr in raw_win_rate:
            ct_win_rate = safe_float(wr.get("ct_win_rate", 0.0))
            team1_win_rate = ct_win_rate if team1_on_ct else 1.0 - ct_win_rate
            win_rate.append(
                {
                    "round_seconds": safe_float(wr.get("round_seconds", 0.0)),
                    "ct_win_rate": ct_win_rate,
                    "team1_win_rate": team1_win_rate,
                }
            )

        round_summary = build_round_summary(val)
        swings = build_team_swings(win_rate)

        for item in round_summary["per_player"]:
            player = item["player"]
            if player not in player_totals:
                player_totals[player] = {
                    "rounds": 0.0,
                    "sum_kill": 0.0,
                    "sum_tactical": 0.0,
                    "sum_total": 0.0,
                }
            player_totals[player]["rounds"] += 1.0
            player_totals[player]["sum_kill"] += item["kill_contribution"]
            player_totals[player]["sum_tactical"] += item["tactical_contribution"]
            player_totals[player]["sum_total"] += item["total_contribution"]

        rounds.append(
            {
                "round_id": round_id,
                "winner": winner_team,
                "winner_side": winner_side,
                "team1_on_ct": team1_on_ct,
                "team1_players": round_team1_players,
                "team2_players": round_team2_players,
                "ct_players": ct_players,
                "t_players": t_players,
                "win_rate": win_rate,
                "kills": kills,
                "player_data": player_data,
                "start_inventory": start_inventory,
                "round_summary": round_summary,
                "swings": swings,
                "map_name": val.get("map_name"),
                "ticks": val.get("ticks", []),
            }
        )

    rounds.sort(key=lambda x: x["round_id"])

    overall = []
    for player, stat in player_totals.items():
        rounds_count = max(stat["rounds"], 1.0)
        avg_kill = stat["sum_kill"] / rounds_count
        avg_tactical = stat["sum_tactical"] / rounds_count
        avg_total = stat["sum_total"] / rounds_count
        overall.append(
            {
                "player": player,
                "team": player_team.get(player, "Unknown"),
                "avg_kill_contribution": avg_kill,
                "avg_tactical_contribution": avg_tactical,
                "avg_total_contribution": avg_total,
                "rounds": int(rounds_count),
            }
        )

    overall.sort(key=lambda x: x["avg_total_contribution"], reverse=True)

    match_winner = "team1" if team1_round_wins > team2_round_wins else "team2"
    if team1_round_wins == team2_round_wins:
        match_winner = "Tie"

    match_loser = "team2" if match_winner == "team1" else "team1"
    if match_winner == "Tie":
        match_loser = "Tie"

    winners = [x for x in overall if x["team"] == match_winner]
    losers = [x for x in overall if x["team"] == match_loser]

    mvp = winners[0] if winners else None
    svp = losers[0] if losers else None

    advanced = build_advanced_metrics(rounds)

    return {
        "rounds": rounds,
        "overall": overall,
        "errors": errors,
        "advanced": advanced,
        "match": {
            "team1_round_wins": team1_round_wins,
            "team2_round_wins": team2_round_wins,
            "winner": match_winner,
            "loser": match_loser,
            "team1_players": sorted(team1_roster),
            "team2_players": sorted(team2_roster),
            "mvp": mvp,
            "svp": svp,
        },
    }


MAX_FEATURED_ROUNDS = 6
MAX_KILLS_PER_FEATURED_ROUND = 5
MAX_KILL_RANKING_ENTRIES = 10


def build_llm_payload(dashboard: dict[str, Any]) -> dict[str, Any]:
    """
    Compact payload for the LLM. We only keep the most decisive per-round data and
    rely on the caller's prompt + whitelist to block hallucinations. Large raw
    arrays (win_rate curves, per-tick player states) are dropped.
    """

    def signed_percent(value: float) -> str:
        return f"{safe_float(value, 0.0) * 100.0:+.2f}%"

    def format_contrib_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted = []
        for item in items or []:
            kill_contrib = safe_float(item.get("kill_contribution", item.get("avg_kill_contribution", 0.0)))
            tactical_contrib = safe_float(item.get("tactical_contribution", item.get("avg_tactical_contribution", 0.0)))
            total_contrib = safe_float(item.get("total_contribution", item.get("avg_total_contribution", 0.0)))
            entry = {
                "player": item.get("player"),
                "kill_pct": signed_percent(kill_contrib),
                "tactical_pct": signed_percent(tactical_contrib),
                "total_pct": signed_percent(total_contrib),
            }
            if item.get("team") is not None:
                entry["team"] = item["team"]
            if item.get("rounds") is not None:
                entry["rounds"] = item["rounds"]
            formatted.append(entry)
        return formatted

    def nearest_wr(win_rate_points: list[dict[str, Any]], t: float) -> float:
        if not win_rate_points:
            return 0.0
        best = win_rate_points[0]
        best_gap = abs(safe_float(best.get("round_seconds", 0.0)) - t)
        for point in win_rate_points[1:]:
            gap = abs(safe_float(point.get("round_seconds", 0.0)) - t)
            if gap < best_gap:
                best = point
                best_gap = gap
        return safe_float(best.get("team1_win_rate", 0.0))

    def kill_team_label(killer: str, team1: list[str], team2: list[str]) -> str:
        if killer in team1:
            return "team1"
        if killer in team2:
            return "team2"
        return "unknown"

    def half_score(half_rounds: list[dict[str, Any]]) -> dict[str, int]:
        t1 = t2 = 0
        for item in half_rounds:
            winner = item.get("winner")
            if winner == "team1":
                t1 += 1
            elif winner == "team2":
                t2 += 1
        return {"team1": t1, "team2": t2}

    def make_half_meta(name: str, half_rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not half_rounds:
            return None
        first = half_rounds[0]
        team1_on_ct = bool(first.get("team1_on_ct", False))
        round_ids = [int(x.get("round_id", 0)) for x in half_rounds if isinstance(x.get("round_id"), int)]
        return {
            "name": name,
            "team1_side": "CT" if team1_on_ct else "T",
            "team2_side": "T" if team1_on_ct else "CT",
            "team1_role": "defense" if team1_on_ct else "attack",
            "team2_role": "attack" if team1_on_ct else "defense",
            "round_start": min(round_ids) if round_ids else None,
            "round_end": max(round_ids) if round_ids else None,
            "score": half_score(half_rounds),
        }

    source_rounds = dashboard.get("rounds", [])

    # Two-pass: build full per-round features (lean), then select top rounds and
    # compact everyone else into a 1-line summary.
    full_rounds: list[dict[str, Any]] = []
    for rd in source_rounds:
        win_rate = rd.get("win_rate", [])
        wr_values = [safe_float(x.get("team1_win_rate", 0.0)) for x in win_rate]
        team1_players = rd.get("team1_players", [])
        team2_players = rd.get("team2_players", [])
        team1_on_ct = bool(rd.get("team1_on_ct", False))
        team1_side = "CT" if team1_on_ct else "T"
        team2_side = "T" if team1_on_ct else "CT"

        start_wr = wr_values[0] if wr_values else 0.0
        end_wr = wr_values[-1] if wr_values else 0.0
        max_wr = max(wr_values) if wr_values else 0.0
        min_wr = min(wr_values) if wr_values else 0.0

        kills_sorted = sorted(rd.get("kills", []), key=lambda x: safe_float(x.get("round_seconds", 0.0)))
        transitions = []
        for kill in kills_sorted:
            t = safe_float(kill.get("round_seconds", 0.0))
            before = nearest_wr(win_rate, t - 0.12)
            after = nearest_wr(win_rate, t + 0.12)
            delta = after - before
            killer = kill.get("killer", "Unknown")
            victim = kill.get("victim", "Unknown")
            transitions.append(
                {
                    "t": round(t, 2),
                    "killer": killer,
                    "killer_team": kill_team_label(killer, team1_players, team2_players),
                    "victim": victim,
                    "weapon": kill.get("weapon", "Unknown"),
                    "hs": bool(kill.get("headshot", False)),
                    "assister": kill.get("assister"),
                    "difficulty": round(safe_float(kill.get("difficulty", 0.0)), 3),
                    "wr_delta_pct": round(delta * 100.0, 1),
                }
            )

        # Interestingness = biggest absolute win-rate swing driven by a single kill.
        peak_delta = max((abs(x["wr_delta_pct"]) for x in transitions), default=0.0)

        full_rounds.append(
            {
                "round_id": rd.get("round_id"),
                "winner": rd.get("winner"),
                "team1_side": team1_side,
                "team2_side": team2_side,
                "wr_start_pct": round(start_wr * 100.0, 1),
                "wr_end_pct": round(end_wr * 100.0, 1),
                "wr_max_pct": round(max_wr * 100.0, 1),
                "wr_min_pct": round(min_wr * 100.0, 1),
                "peak_kill_delta_pct": peak_delta,
                "kill_transitions": transitions,
                "final_contrib": format_contrib_items(
                    rd.get("round_summary", {}).get("per_player", [])
                ),
            }
        )

    # Select the featured rounds (most decisive) and compact the rest.
    sorted_by_interest = sorted(
        full_rounds, key=lambda r: r["peak_kill_delta_pct"], reverse=True
    )
    featured_ids = {r["round_id"] for r in sorted_by_interest[:MAX_FEATURED_ROUNDS]}

    featured_rounds: list[dict[str, Any]] = []
    other_rounds: list[dict[str, Any]] = []
    for r in full_rounds:
        if r["round_id"] in featured_ids:
            trimmed_transitions = sorted(
                r["kill_transitions"],
                key=lambda x: abs(x["wr_delta_pct"]),
                reverse=True,
            )[:MAX_KILLS_PER_FEATURED_ROUND]
            trimmed_transitions.sort(key=lambda x: x["t"])
            featured_rounds.append(
                {**r, "kill_transitions": trimmed_transitions}
            )
        else:
            other_rounds.append(
                {
                    "round_id": r["round_id"],
                    "winner": r["winner"],
                    "team1_side": r["team1_side"],
                    "team2_side": r["team2_side"],
                    "wr_start_pct": r["wr_start_pct"],
                    "wr_end_pct": r["wr_end_pct"],
                    "kill_count": len(r["kill_transitions"]),
                }
            )

    featured_rounds.sort(key=lambda r: r["round_id"])
    other_rounds.sort(key=lambda r: r["round_id"])

    # Halves
    first_half_rounds: list[dict[str, Any]] = []
    second_half_rounds: list[dict[str, Any]] = []
    if source_rounds:
        boundaries = [0]
        prev_flag = bool(source_rounds[0].get("team1_on_ct", False))
        for idx in range(1, len(source_rounds)):
            curr_flag = bool(source_rounds[idx].get("team1_on_ct", False))
            if curr_flag != prev_flag:
                boundaries.append(idx)
            prev_flag = curr_flag
        boundaries.append(len(source_rounds))
        first_half_rounds = source_rounds[boundaries[0]:boundaries[1]]
        if len(boundaries) > 2:
            second_half_rounds = source_rounds[boundaries[1]:boundaries[2]]
        elif len(boundaries) > 1:
            second_half_rounds = source_rounds[boundaries[1]:]

    match_halves = {
        "first_half": make_half_meta("first_half", first_half_rounds),
        "second_half": make_half_meta("second_half", second_half_rounds),
    }

    # Advanced metrics summary — trimmed to keep the prompt compact.
    advanced = dashboard.get("advanced", {}) or {}
    adv_kill_ranking = [
        {
            "round": k.get("round"),
            "t": round(safe_float(k.get("round_seconds", 0.0)), 2),
            "attacker": k.get("attacker"),
            "victim": k.get("victim"),
            "swing_pct": signed_percent(k.get("swing", 0.0)),
            "difficulty": round(safe_float(k.get("difficulty", 0.0)), 3),
        }
        for k in (advanced.get("kill_ranking") or [])[:MAX_KILL_RANKING_ENTRIES]
    ]
    adv_player_stats = [
        {
            "player": p.get("player"),
            "team": p.get("team"),
            "avg_kill_opp": round(safe_float(p.get("avg_kill_opp", 0.0)), 3),
            "avg_death_opp": round(safe_float(p.get("avg_death_opp", 0.0)), 3),
            "avg_survive_chance": round(safe_float(p.get("avg_survive_chance", 0.0)), 3),
            "hard_win_rate": round(safe_float(p.get("hard_win_rate", 0.0)), 3),
            "easy_win_rate": round(safe_float(p.get("easy_win_rate", 0.0)), 3),
            "highlight_rate": round(safe_float(p.get("highlight_rate", 0.0)), 3),
        }
        for p in (advanced.get("player_stats") or [])
    ]

    match_info = dashboard.get("match", {}) or {}
    whitelist = {
        "team1_players": sorted(match_info.get("team1_players", []) or []),
        "team2_players": sorted(match_info.get("team2_players", []) or []),
        "valid_round_ids": sorted([r["round_id"] for r in full_rounds if r["round_id"] is not None]),
    }

    return {
        "match": {
            "team1_round_wins": match_info.get("team1_round_wins"),
            "team2_round_wins": match_info.get("team2_round_wins"),
            "winner": match_info.get("winner"),
            "mvp": (match_info.get("mvp") or {}).get("player"),
            "svp": (match_info.get("svp") or {}).get("player"),
        },
        "match_halves": match_halves,
        "whitelist": whitelist,
        "overall_player_averages": format_contrib_items(dashboard.get("overall", [])),
        "featured_rounds": featured_rounds,
        "other_rounds": other_rounds,
        "advanced_kill_ranking": adv_kill_ranking,
        "advanced_player_stats": adv_player_stats,
    }


def build_chat_completion_url(base_url: str) -> str:
    normalized = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def build_llm_prompts(llm_data: dict[str, Any], language: str) -> tuple[str, str]:
    """
    Two harness defenses to reduce hallucinations:
      1. The system prompt lists the ONLY valid player names and round IDs. Anything
         outside that set is forbidden.
      2. The user prompt demands that every numeric claim be traceable to a JSON
         field name the model must cite (e.g., featured_rounds[].kill_transitions[].wr_delta_pct).
    """
    lang = (language or "zh").strip().lower()
    if lang not in {"zh", "en"}:
        lang = "zh"

    whitelist = llm_data.get("whitelist", {}) or {}
    team1_players = whitelist.get("team1_players", []) or []
    team2_players = whitelist.get("team2_players", []) or []
    valid_round_ids = whitelist.get("valid_round_ids", []) or []

    all_players = sorted(set(team1_players) | set(team2_players))
    data_json = json.dumps(llm_data, ensure_ascii=False)

    if lang == "en":
        system_prompt = (
            "You are a professional CS2 tactical analyst. Produce an insightful English review.\n\n"
            "STRICT ANTI-HALLUCINATION RULES:\n"
            f"- Valid player names (do NOT invent others): {all_players}\n"
            f"- team1 roster: {team1_players}\n"
            f"- team2 roster: {team2_players}\n"
            f"- Valid round IDs: {valid_round_ids}\n"
            "- Every numeric claim (win rate, swing, difficulty, contribution) MUST come directly "
            "from the JSON data. Do not fabricate values, do not round aggressively.\n"
            "- If a field is missing, say so instead of guessing.\n"
            "- Do not invent kills, clutches, weapons, or round outcomes that are absent from the JSON.\n"
        )
        user_prompt = (
            "Deliver the following sections (concise, well-structured markdown):\n"
            "1) Halves & score: cite match_halves.first_half / .second_half verbatim (sides and score).\n"
            "2) Team narrative: tempo, consistency, collapse/comeback moments. Cite overall_player_averages + match.\n"
            "3) Featured rounds: walk through each round in featured_rounds.\n"
            "   For each, state team1_side/team2_side, wr_start_pct → wr_end_pct, and the 1-3 most decisive "
            "   kill_transitions (use killer, victim, weapon, wr_delta_pct, difficulty).\n"
            "4) Per-player review: cite advanced_player_stats (avg_kill_opp, avg_survive_chance, hard_win_rate) "
            "   alongside overall_player_averages. Only comment on players in the whitelist.\n"
            "5) Fun stats from advanced_kill_ranking (top swings and their difficulty).\n"
            "6) MVP/SVP check: compare match.mvp / match.svp to advanced_player_stats; confirm or disagree with data.\n"
            "7) Three actionable improvement suggestions.\n\n"
            "Data (JSON):\n" + data_json
        )
        return system_prompt, user_prompt

    system_prompt = (
        "你是专业的 CS2 战术分析师，请输出中文复盘。\n\n"
        "严格防幻觉规则：\n"
        f"- 合法玩家名（不得发明其它名字）：{all_players}\n"
        f"- team1 阵容：{team1_players}\n"
        f"- team2 阵容：{team2_players}\n"
        f"- 合法回合 ID：{valid_round_ids}\n"
        "- 任何数字（胜率、swing、难度、贡献）必须直接来自下方 JSON，不许编造或过度取整。\n"
        "- 如果某字段缺失，直接说明“数据中未提供”，不要猜。\n"
        "- 不要编造 JSON 里没有的击杀、残局、武器、回合结果。\n"
    )
    user_prompt = (
        "请按如下结构输出（markdown，精炼）：\n"
        "1) 上/下半场阵营与比分：严格以 match_halves.first_half / .second_half 为准。\n"
        "2) 团队叙事：节奏、稳定性、崩盘/翻盘瞬间。引用 overall_player_averages 与 match。\n"
        "3) 精选回合讲解：遍历 featured_rounds 列表。\n"
        "   每一回合写出 team1_side / team2_side、wr_start_pct → wr_end_pct，并挑 1-3 个 kill_transitions 里 |wr_delta_pct| 最大的击杀展开（注明 killer、victim、weapon、wr_delta_pct、difficulty）。\n"
        "4) 玩家点评：结合 advanced_player_stats（avg_kill_opp / avg_survive_chance / hard_win_rate）与 overall_player_averages，只评论白名单里的玩家。\n"
        "5) 趣味数据：从 advanced_kill_ranking 里挑 top swing 的击杀，说明对应 difficulty。\n"
        "6) MVP/SVP 复核：对照 advanced_player_stats 看 match.mvp / match.svp 是否合理，给出数据支持或反对。\n"
        "7) 三条可执行改进建议。\n\n"
        "以下是结构化数据(JSON)：\n" + data_json
    )
    return system_prompt, user_prompt


def _append_job_log(job_id: str, line: str) -> None:
    clean = line.rstrip("\n")
    with ANALYSIS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if not job:
            return
        job["logs"].append(clean)
        if len(job["logs"]) > 300:
            job["logs"] = job["logs"][-300:]


def _update_job_progress_by_log(job_id: str, line: str) -> None:
    total_match = re.search(r"Total rounds with sampled ticks:\s*(\d+)", line)
    round_match = re.search(r"Processing round\s+(\d+)\s+with", line)

    with ANALYSIS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if not job:
            return

        if "Loading model from" in line:
            job["phase"] = "加载模型中"
            job["message"] = "正在加载胜率模型..."
        elif "Sampling ticks" in line or "Processing demo" in line:
            job["phase"] = "解析 Demo 中"
            job["message"] = "正在解析 demo (通常 30~60s)"
        elif "Extracted states for" in line and "Starting inference" in line:
            job["phase"] = "模型推理中"
            job["message"] = "开始逐回合推理 (通常不到 1 分钟)"
        elif "Results saved to" in line:
            job["phase"] = "结果收集中"
            job["message"] = "结果已生成，正在整理数据..."

        if total_match:
            job["total_rounds"] = int(total_match.group(1))

        if round_match:
            current_round = int(round_match.group(1))
            job["current_round"] = current_round
            total_rounds = int(job.get("total_rounds", 0))
            if total_rounds > 0:
                finished = int(job.get("processed_rounds", 0)) + 1
                finished = min(finished, total_rounds)
                job["processed_rounds"] = finished
                progress = int(100 * finished / total_rounds)
                job["progress"] = max(min(progress, 99), 1)
                job["message"] = f"正在处理第 {current_round} 局 ({finished}/{total_rounds})"


def _run_analysis_job(
    job_id: str,
    upload_path: Path,
    model_path: str,
    device: str,
    batch_size: str,
    output_path: Path,
) -> None:
    with ANALYSIS_LOCK:
        if job_id not in ANALYSIS_JOBS:
            return
        ANALYSIS_JOBS[job_id]["status"] = "running"
        ANALYSIS_JOBS[job_id]["phase"] = "准备中"
        ANALYSIS_JOBS[job_id]["message"] = "任务已启动，正在准备执行..."

    cmd = [
        sys.executable,
        "-m",
        "demo_analysis.get_round_win_rate",
        "--demo_path",
        str(upload_path),
        "--model_root",
        model_path,
        "--device",
        device,
        "--output",
        str(output_path),
        "--batch_size",
        batch_size,
    ]
    if device == "cpu":
        cmd.append("--skip_duel")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )

        stdout_lines: list[str] = []
        if proc.stdout is not None:
            for line in proc.stdout:
                stdout_lines.append(line)
                _append_job_log(job_id, line)
                _update_job_progress_by_log(job_id, line)

        return_code = proc.wait()

        if return_code != 0:
            with ANALYSIS_LOCK:
                job = ANALYSIS_JOBS.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["phase"] = "失败"
                    job["message"] = "分析脚本运行失败"
                    job["error"] = "\n".join(job.get("logs", [])[-80:])[-4000:]
            return

        if not output_path.exists():
            with ANALYSIS_LOCK:
                job = ANALYSIS_JOBS.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["phase"] = "失败"
                    job["message"] = "分析结果文件不存在"
                    job["error"] = "分析结果文件不存在"
            return

        with output_path.open("r", encoding="utf-8") as f:
            raw_results = json.load(f)

        dashboard = build_dashboard_payload(raw_results)
        analysis_id = uuid.uuid4().hex
        ANALYSIS_CACHE[analysis_id] = {
            "dashboard": dashboard,
            "raw": raw_results,
            "source_file": str(upload_path),
            "result_file": str(output_path),
            "stdout": "".join(stdout_lines),
        }

        with ANALYSIS_LOCK:
            job = ANALYSIS_JOBS.get(job_id)
            if job:
                job["status"] = "succeeded"
                job["phase"] = "完成"
                job["message"] = "分析完成"
                job["progress"] = 100
                job["analysis_id"] = analysis_id
                job["dashboard"] = dashboard
    except Exception as exc:
        with ANALYSIS_LOCK:
            job = ANALYSIS_JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["phase"] = "失败"
                job["message"] = "后台任务异常"
                job["error"] = str(exc)


@app.route("/")
def index():
    # Keep disk usage bounded for local usage: clear stale artifacts on each open.
    if not _has_running_jobs():
        cleanup_runtime_artifacts(clear_state=True)

    model_options = discover_model_paths()
    return render_template(
        "index.html",
        default_device=choose_default_device(),
        model_options=model_options,
        default_model_path=choose_default_model_path(model_options),
    )


@app.get("/assets/<path:filename>")
def asset_file(filename: str):
    return send_from_directory(ASSETS_DIR, filename)


# ---- third_party 2D replay viewer (built SPA mounted under /viewer/) ----

@app.get("/viewer/")
@app.get("/viewer/player")
def viewer_index():
    """Serve the SPA shell for the viewer's home and /player routes."""
    return send_from_directory(VIEWER_DIR, "index.html")


@app.get("/viewer/<path:filename>")
def viewer_assets(filename: str):
    """Serve any built asset (JS/CSS/PNG/wasm/worker.js) under /viewer/."""
    target = (VIEWER_DIR / filename).resolve()
    try:
        target.relative_to(VIEWER_DIR.resolve())
    except ValueError:
        return jsonify({"error": "invalid path"}), 404
    if not target.is_file():
        # Unknown sub-route → fall back to SPA shell so client routing still works.
        return send_from_directory(VIEWER_DIR, "index.html")
    return send_from_directory(VIEWER_DIR, filename)


def build_viewer_timeline(dashboard: dict[str, Any]) -> dict[str, Any]:
    """Flatten per-tick predictions into the shape the 2D viewer's overlays expect.

    The bundled viewer (`static/viewer/`) reads a `winrateurl` JSON and drives its
    win-rate curve, next-kill / next-death bars, and duel matrix from the entry
    that matches the current round + round_seconds. Fields consumed by the viewer:
    round, seconds, ct_win_rate, alive_pred (10), next_kill (11), next_death (11),
    duel (10x10), players_info[{name}]. See `winRateData.js:normalizeTimelineEntry`.
    """
    timeline: list[dict[str, Any]] = []
    for rd in dashboard.get("rounds") or []:
        round_id = rd.get("round_id")
        for tick in rd.get("ticks") or []:
            seconds = tick.get("round_seconds")
            if seconds is None:
                continue
            players = [
                {"name": p.get("name")}
                for p in (tick.get("players_info") or [])
                if p.get("name")
            ]
            timeline.append(
                {
                    "round": round_id,
                    "seconds": seconds,
                    "ct_win_rate": tick.get("ct_win_rate"),
                    "alive_pred": tick.get("alive_pred"),
                    "next_kill": tick.get("next_kill"),
                    "next_death": tick.get("next_death"),
                    "duel": tick.get("duel"),
                    "players_info": players,
                }
            )
    return {
        "meta": {"roundBase": 1},
        "timeline": timeline,
    }


@app.get("/api/winrate_timeline/<analysis_id>")
def serve_winrate_timeline(analysis_id: str):
    """Serve the per-tick prediction timeline consumed by the 2D viewer overlays."""
    if not re.fullmatch(r"[0-9a-fA-F]{8,64}", analysis_id):
        return jsonify({"error": "bad analysis_id"}), 400
    entry = ANALYSIS_CACHE.get(analysis_id)
    if not entry:
        return jsonify({"error": "analysis not found"}), 404
    return jsonify(build_viewer_timeline(entry["dashboard"]))


@app.get("/api/demo_file/<run_id>")
@app.get("/api/demo_file/<run_id>.dem")
def serve_uploaded_demo(run_id: str):
    """Stream the originally uploaded .dem file so the in-browser parser can fetch it.

    Accepts both `<run_id>` and `<run_id>.dem` — the 2D viewer's WASM parser picks
    demo format from the URL suffix, so we surface `.dem` by default.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{8,64}", run_id):
        return jsonify({"error": "bad run_id"}), 400
    matches = sorted(UPLOAD_DIR.glob(f"{run_id}_*"))
    if not matches:
        return jsonify({"error": "demo not found"}), 404
    target = matches[0]
    return send_from_directory(
        UPLOAD_DIR,
        target.name,
        as_attachment=False,
        download_name=target.name.split("_", 1)[-1],
        mimetype="application/octet-stream",
    )


@app.post("/api/analyze")
def analyze_demo():
    dem_file = request.files.get("demo_file")
    if dem_file is None or dem_file.filename == "":
        return jsonify({"error": "请上传 .dem 文件"}), 400

    if _has_running_jobs():
        return jsonify({"error": "已有任务正在运行，请等待完成后再上传新 demo"}), 409

    cleanup_runtime_artifacts(clear_state=True)

    model_path = request.form.get("model_path", "").strip()
    device = request.form.get("device", choose_default_device()).strip()
    batch_size = request.form.get("batch_size", "32").strip()

    if not model_path:
        return jsonify({"error": "请先选择模型根目录"}), 400

    model_dir = normalize_model_root(model_path)
    if model_dir is None:
        return jsonify({
            "error": (
                f"模型根目录无效: {model_path}\n"
                f"至少需要 win_rate 子目录（或直接传入 win_rate 目录）；"
                f"alive / nxt_kill / nxt_death / duel 缺失时会自动走 fallback。"
            )
        }), 400

    model_path = str(model_dir)

    run_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{run_id}_{Path(dem_file.filename).name}"
    output_path = OUTPUT_DIR / f"{run_id}.json"
    dem_file.save(upload_path)

    job_id = uuid.uuid4().hex
    with ANALYSIS_LOCK:
        ANALYSIS_JOBS[job_id] = {
            "status": "queued",
            "phase": "排队中",
            "message": "任务已提交，等待执行",
            "progress": 0,
            "logs": [
                "已接收任务，准备开始分析...",
                "提示: 解析 demo 通常 30~60s，模型推理通常不到 10 分钟。",
            ],
            "error": "",
            "total_rounds": 0,
            "processed_rounds": 0,
            "current_round": None,
            "analysis_id": None,
            "dashboard": None,
            "run_id": run_id,
        }

    worker = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, upload_path, model_path, device, batch_size, output_path),
        daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id})


@app.get("/api/analyze_status/<job_id>")
def analyze_status(job_id: str):
    with ANALYSIS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if not job:
            return jsonify({"error": "无效的 job_id"}), 404

        logs = job.get("logs", [])
        payload = {
            "status": job.get("status", "unknown"),
            "phase": job.get("phase", ""),
            "message": job.get("message", ""),
            "progress": job.get("progress", 0),
            "total_rounds": job.get("total_rounds", 0),
            "processed_rounds": job.get("processed_rounds", 0),
            "current_round": job.get("current_round"),
            "logs": "\n".join(logs[-120:]),
            "error": job.get("error", ""),
            "analysis_id": job.get("analysis_id"),
            "run_id": job.get("run_id"),
        }

        if job.get("status") == "succeeded":
            payload["dashboard"] = job.get("dashboard")

    return jsonify(payload)


@app.post("/api/llm_summary")
def llm_summary():
    body = request.get_json(silent=True) or {}
    analysis_id = body.get("analysis_id", "")
    api_key = body.get("api_key", "").strip()
    model_name = body.get("model_name", "").strip()
    base_url = body.get("base_url", "https://api.openai.com/v1").strip()
    temperature = safe_float(body.get("temperature", 0.95), 0.95)
    language = body.get("language", "zh").strip().lower()

    if not analysis_id or analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "无效的 analysis_id，请先完成 DEM 分析"}), 400
    if not api_key:
        return jsonify({"error": "请提供 API Key"}), 400
    if not model_name:
        return jsonify({"error": "请提供模型名称"}), 400

    dashboard = ANALYSIS_CACHE[analysis_id]["dashboard"]
    llm_data = build_llm_payload(dashboard)
    system_prompt, user_prompt = build_llm_prompts(llm_data, language)

    payload = {
        "model": model_name,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        url = build_chat_completion_url(base_url)
        last_exc: Exception | None = None
        resp = None

        # Retry transient timeout / 5xx issues to improve stability on slower providers.
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(20, 300),
                )

                if resp.status_code >= 500 and attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break
            except requests.Timeout as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                return (
                    jsonify(
                        {
                            "error": "LLM 接口请求超时，请稍后重试",
                            "detail": str(exc),
                        }
                    ),
                    504,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break

        if resp is None:
            return jsonify({"error": f"请求 LLM 接口异常: {last_exc}"}), 500

        if resp.status_code >= 400:
            return (
                jsonify(
                    {
                        "error": "LLM 接口调用失败",
                        "status_code": resp.status_code,
                        "detail": resp.text[:4000],
                    }
                ),
                500,
            )

        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "模型返回为空")
        )
        return jsonify({"summary": content})
    except requests.RequestException as exc:
        return jsonify({"error": f"请求 LLM 接口异常: {exc}"}), 500


@app.post("/api/llm_summary_stream")
def llm_summary_stream():
    body = request.get_json(silent=True) or {}
    analysis_id = body.get("analysis_id", "")
    api_key = body.get("api_key", "").strip()
    model_name = body.get("model_name", "").strip()
    base_url = body.get("base_url", "https://api.openai.com/v1").strip()
    temperature = safe_float(body.get("temperature", 0.95), 0.95)
    language = body.get("language", "zh").strip().lower()

    if not analysis_id or analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "无效的 analysis_id，请先完成 DEM 分析"}), 400
    if not api_key:
        return jsonify({"error": "请提供 API Key"}), 400
    if not model_name:
        return jsonify({"error": "请提供模型名称"}), 400

    dashboard = ANALYSIS_CACHE[analysis_id]["dashboard"]
    llm_data = build_llm_payload(dashboard)
    system_prompt, user_prompt = build_llm_prompts(llm_data, language)

    payload = {
        "model": model_name,
        "temperature": temperature,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = build_chat_completion_url(base_url)
    try:
        upstream = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(20, 600),
            stream=True,
        )
    except requests.Timeout as exc:
        return jsonify({"error": "LLM 接口请求超时，请稍后重试", "detail": str(exc)}), 504
    except requests.RequestException as exc:
        return jsonify({"error": f"请求 LLM 接口异常: {exc}"}), 500

    if upstream.status_code >= 400:
        return (
            jsonify(
                {
                    "error": "LLM 接口调用失败",
                    "status_code": upstream.status_code,
                    "detail": upstream.text[:4000],
                }
            ),
            500,
        )

    @stream_with_context
    def generate_text_stream():
        try:
            for raw_line in upstream.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text is None:
                    text = choice.get("message", {}).get("content", "")

                if text:
                    yield text
        finally:
            upstream.close()

    return Response(generate_text_stream(), content_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)
