import json
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CALLOUT_CONFIG_PATH = ROOT_DIR / "config" / "map_callouts.yaml"


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


def load_map_callouts() -> dict[str, Any]:
    if not CALLOUT_CONFIG_PATH.exists():
        return {}
    with CALLOUT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def normalize_callout(
    map_name: str | None,
    raw_place: Any,
    callouts: dict[str, Any],
) -> dict[str, Any]:
    raw = "" if raw_place is None else str(raw_place).strip()
    if not raw:
        return {"raw": None, "name": "数据未提供", "callout_source": "missing"}

    aliases: dict[str, str] = {}
    for section in ("default", map_name or ""):
        cfg = callouts.get(section, {}) if section else {}
        aliases.update(cfg.get("aliases", {}) or {})

    compact = raw.replace(" ", "").replace("_", "")
    lookup = {str(k): str(v) for k, v in aliases.items()}
    lookup.update({str(k).replace(" ", "").replace("_", ""): str(v) for k, v in aliases.items()})
    if raw in lookup:
        return {"raw": raw, "name": lookup[raw], "callout_source": "mapped"}
    if compact in lookup:
        return {"raw": raw, "name": lookup[compact], "callout_source": "mapped"}
    return {"raw": raw, "name": raw, "callout_source": "raw"}


def nearest_point(points: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    if not points:
        return None
    best = points[0]
    best_gap = abs(safe_float(best.get("round_seconds", 0.0)) - t)
    for point in points[1:]:
        gap = abs(safe_float(point.get("round_seconds", 0.0)) - t)
        if gap < best_gap:
            best = point
            best_gap = gap
    return best


def count_alive(players: list[dict[str, Any]]) -> dict[str, int]:
    score = {"CT": 0, "T": 0}
    for p in players or []:
        if not bool(p.get("is_alive", False)):
            continue
        side = p.get("team_num")
        if side in score:
            score[side] += 1
    return score


def summarize_utility(tick: dict[str, Any] | None) -> dict[str, Any]:
    projectiles = (tick or {}).get("projectiles") or []
    entity_grenades = (tick or {}).get("entity_grenades") or []
    smokes = sum(1 for p in projectiles if p.get("type") == "smokegrenade")
    infernos = sum(1 for p in projectiles if p.get("type") == "inferno")
    return {
        "active_smokes": smokes,
        "active_infernos": infernos,
        "flying_grenades": len(entity_grenades),
        "bomb_planted": bool((tick or {}).get("is_bomb_planted", False)),
    }


def team_label_for_player(name: str, team1_players: list[str], team2_players: list[str]) -> str:
    if name in team1_players:
        return "team1"
    if name in team2_players:
        return "team2"
    return "unknown"


def summarize_inventory(items: list[dict[str, Any]]) -> dict[str, Any]:
    rifles = {
        "AK-47", "M4A4", "M4A1-S", "AUG", "SG 553", "FAMAS", "Galil AR",
        "AWP", "SSG 08", "G3SG1", "SCAR-20",
    }
    pistols = {
        "Glock-18", "USP-S", "P2000", "P250", "Desert Eagle", "Five-SeveN",
        "Tec-9", "CZ75-Auto", "Dual Berettas", "R8 Revolver",
    }
    grenades = {
        "Flashbang", "High Explosive Grenade", "Smoke Grenade", "Molotov",
        "Incendiary Grenade", "Decoy Grenade",
    }
    by_side: dict[str, dict[str, Any]] = {}
    for item in items or []:
        side = item.get("team_num", "Unknown")
        inv = set(item.get("inventory") or [])
        entry = by_side.setdefault(
            side,
            {"rifles": 0, "pistols": 0, "awps": 0, "grenades": 0, "helmets": 0, "kits": 0},
        )
        entry["rifles"] += int(bool(inv & rifles))
        entry["pistols"] += int(bool(inv & pistols))
        entry["awps"] += int("AWP" in inv)
        entry["grenades"] += len(inv & grenades)
        entry["helmets"] += int(bool(item.get("has_helmet", False)))
        entry["kits"] += int(bool(item.get("has_defuser", False)))
    return by_side


def build_tactical_rounds(rounds: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    callouts = load_map_callouts()
    coverage = {"mapped": 0, "raw": 0, "missing": 0}
    tactical_rounds: list[dict[str, Any]] = []

    for rd in rounds:
        ticks = rd.get("ticks") or []
        win_rate = rd.get("win_rate") or []
        map_name = rd.get("map_name")
        team1_players = rd.get("team1_players") or []
        team2_players = rd.get("team2_players") or []
        first_tick = ticks[0] if ticks else {}
        first_players = first_tick.get("players_info") or []
        name_to_idx = {
            p.get("name"): idx
            for idx, p in enumerate(first_players[:10])
            if p.get("name") is not None
        }

        def player_at(tick: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
            for p in (tick or {}).get("players_info") or []:
                if p.get("name") == name:
                    return p
            return None

        def location_for(tick: dict[str, Any] | None, name: str) -> dict[str, Any]:
            p = player_at(tick, name)
            loc = normalize_callout(map_name, (p or {}).get("last_place_name"), callouts)
            coverage[loc["callout_source"]] += 1
            return loc

        def wr_at(t: float) -> float:
            point = nearest_point(win_rate, t)
            return safe_float((point or {}).get("team1_win_rate", 0.0))

        timeline: list[dict[str, Any]] = []
        for kill in sorted(rd.get("kills") or [], key=lambda x: safe_float(x.get("round_seconds", 0.0))):
            t = safe_float(kill.get("round_seconds", 0.0))
            before_tick = nearest_point(ticks, t - 0.12)
            after_tick = nearest_point(ticks, t + 0.12)
            killer = kill.get("killer", "Unknown")
            victim = kill.get("victim", "Unknown")
            killer_team = team_label_for_player(killer, team1_players, team2_players)
            before_wr = wr_at(t - 0.12)
            after_wr = wr_at(t + 0.12)
            wr_delta = (after_wr - before_wr) * 100.0
            difficulty = safe_float(kill.get("difficulty", 0.0))
            killer_loc = location_for(before_tick, killer)
            victim_loc = location_for(before_tick, victim)
            killer_snapshot = player_at(before_tick, killer) or {}
            victim_snapshot = player_at(before_tick, victim) or {}
            duel_prob = None
            a_idx = name_to_idx.get(killer)
            v_idx = name_to_idx.get(victim)
            duel = (before_tick or {}).get("duel") or []
            if a_idx is not None and v_idx is not None and a_idx < len(duel):
                row = duel[a_idx]
                if isinstance(row, list) and v_idx < len(row) and row[v_idx] != "/":
                    duel_prob = round(safe_float(row[v_idx], 0.0), 3)
            killer_swing = wr_delta if killer_team == "team1" else -wr_delta
            location_sources = [
                killer_loc.get("callout_source"),
                victim_loc.get("callout_source"),
            ]

            timeline.append(
                {
                    "t": round(t, 2),
                    "type": "kill",
                    "summary_facts": {
                        "killer": killer,
                        "victim": victim,
                        "weapon": kill.get("weapon", "Unknown"),
                        "headshot": bool(kill.get("headshot", False)),
                        "assister": kill.get("assister"),
                        "assistedflash": bool(kill.get("assistedflash", False)),
                        "thrusmoke": bool(kill.get("thrusmoke", False)),
                    },
                    "players": {
                        "killer": {
                            "name": killer,
                            "team": killer_team,
                            "side": killer_snapshot.get("team_num"),
                            "health": killer_snapshot.get("health"),
                            "weapon": killer_snapshot.get("weapon_name"),
                        },
                        "victim": {
                            "name": victim,
                            "team": team_label_for_player(victim, team1_players, team2_players),
                            "side": victim_snapshot.get("team_num"),
                            "health": victim_snapshot.get("health"),
                            "weapon": victim_snapshot.get("weapon_name"),
                        },
                    },
                    "locations": {"killer": killer_loc, "victim": victim_loc},
                    "alive_score": count_alive((before_tick or {}).get("players_info") or []),
                    "utility_state": summarize_utility(before_tick),
                    "wr_before_pct": round(before_wr * 100.0, 1),
                    "wr_after_pct": round(after_wr * 100.0, 1),
                    "wr_delta_pct": round(wr_delta, 1),
                    "difficulty": round(difficulty, 3),
                    "duel_context": {"killer_vs_victim": duel_prob},
                    "evaluation_context": {
                        "killer_team_swing_pct": round(killer_swing, 1),
                        "winner_to_praise": killer,
                        "loser_to_critique": victim,
                        "location_sources": location_sources,
                        "has_reliable_locations": "missing" not in location_sources,
                        "notes": "Use win-rate swing, difficulty, and duel_context to judge this kill; do not use fixed labels.",
                    },
                }
            )

        planted_time = None
        for tk in ticks:
            if tk.get("bomb_planted_time") is not None:
                planted_time = safe_float(tk.get("bomb_planted_time"))
                break
            if tk.get("is_bomb_planted"):
                planted_time = safe_float(tk.get("round_seconds", 0.0))
                break
        if planted_time is not None:
            plant_tick = nearest_point(ticks, planted_time)
            wr_before = wr_at(planted_time - 0.12)
            wr_after = wr_at(planted_time + 0.12)
            timeline.append(
                {
                    "t": round(planted_time, 2),
                    "type": "bomb_planted",
                    "summary_facts": {"bomb_position": (plant_tick or {}).get("bomb_position")},
                    "players": {},
                    "locations": {},
                    "alive_score": count_alive((plant_tick or {}).get("players_info") or []),
                    "utility_state": summarize_utility(plant_tick),
                    "wr_before_pct": round(wr_before * 100.0, 1),
                    "wr_after_pct": round(wr_after * 100.0, 1),
                    "wr_delta_pct": round((wr_after - wr_before) * 100.0, 1),
                    "difficulty": None,
                    "duel_context": {},
                    "evaluation_context": {
                        "notes": "Bomb plant event only; evaluate from win-rate movement and alive_score if useful.",
                    },
                }
            )

        timeline.sort(key=lambda x: x["t"])
        wr_values = [safe_float(x.get("team1_win_rate", 0.0)) for x in win_rate]
        round_takeaway = {
            "winner": rd.get("winner"),
            "event_count": len(timeline),
            "largest_swing_pct": max((abs(safe_float(x.get("wr_delta_pct", 0.0))) for x in timeline), default=0.0),
        }
        tactical_rounds.append(
            {
                "round_id": rd.get("round_id"),
                "map_name": map_name,
                "winner": rd.get("winner"),
                "team1_side": "CT" if bool(rd.get("team1_on_ct", False)) else "T",
                "team2_side": "T" if bool(rd.get("team1_on_ct", False)) else "CT",
                "economy_summary": summarize_inventory(rd.get("start_inventory") or []),
                "wr_start_pct": round((wr_values[0] if wr_values else 0.0) * 100.0, 1),
                "wr_end_pct": round((wr_values[-1] if wr_values else 0.0) * 100.0, 1),
                "timeline": timeline,
                "round_takeaway": round_takeaway,
            }
        )

    return tactical_rounds, coverage


def build_advanced_metrics(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-player advanced metrics + a |swing|-sorted kill ranking.
    """
    highlight_threshold = 0.20

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
                victim_entry["easy_attempts"] += 1
            elif 0.0 < difficulty < 1.0:
                attacker_entry["easy_attempts"] += 1
                attacker_entry["easy_kills"] += 1
                victim_entry["hard_attempts"] += 1

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
            if safe_float(item.get("total_contribution", 0.0)) >= highlight_threshold:
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
    tactical_rounds, map_callout_coverage = build_tactical_rounds(rounds)

    return {
        "rounds": rounds,
        "overall": overall,
        "errors": errors,
        "advanced": build_advanced_metrics(rounds),
        "tactical_rounds": tactical_rounds,
        "map_callout_coverage": map_callout_coverage,
        "match": {
            "team1_round_wins": team1_round_wins,
            "team2_round_wins": team2_round_wins,
            "winner": match_winner,
            "loser": match_loser,
            "team1_players": sorted(team1_roster),
            "team2_players": sorted(team2_roster),
            "mvp": winners[0] if winners else None,
            "svp": losers[0] if losers else None,
        },
    }


def analyze_raw_results(raw_results: dict[str, Any]) -> dict[str, Any]:
    return build_dashboard_payload(raw_results)


def analyze_output_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw_results = json.load(f)
    return analyze_raw_results(raw_results)
