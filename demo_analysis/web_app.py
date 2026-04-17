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


def choose_default_device() -> str:
    return "cpu"


def discover_model_paths() -> list[str]:
    options: list[str] = []
    if not MODEL_ROOT.exists():
        return options

    for child in sorted(MODEL_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if "win_rate" not in child.name.lower():
            continue
        has_pth = any(p.suffix == ".pth" for p in child.iterdir())
        has_yaml = any(
            p.suffix == ".yaml" and "tokenizer" not in p.name.lower()
            for p in child.iterdir()
        )
        if has_pth and has_yaml:
            options.append(str(child))

    return options


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

    return {
        "rounds": rounds,
        "overall": overall,
        "errors": errors,
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


def build_llm_payload(dashboard: dict[str, Any]) -> dict[str, Any]:
    def signed_percent(value: float) -> str:
        num = safe_float(value, 0.0) * 100.0
        return f"{num:+.2f}%"

    def format_contrib_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted = []
        for item in items or []:
            kill_contrib = safe_float(item.get("kill_contribution", item.get("avg_kill_contribution", 0.0)))
            tactical_contrib = safe_float(item.get("tactical_contribution", item.get("avg_tactical_contribution", 0.0)))
            total_contrib = safe_float(item.get("total_contribution", item.get("avg_total_contribution", 0.0)))

            cloned = dict(item)
            cloned["kill_contribution_pct"] = signed_percent(kill_contrib)
            cloned["tactical_contribution_pct"] = signed_percent(tactical_contrib)
            cloned["total_contribution_pct"] = signed_percent(total_contrib)
            cloned["avg_kill_contribution_pct"] = signed_percent(
                safe_float(item.get("avg_kill_contribution", kill_contrib))
            )
            cloned["avg_tactical_contribution_pct"] = signed_percent(
                safe_float(item.get("avg_tactical_contribution", tactical_contrib))
            )
            cloned["avg_total_contribution_pct"] = signed_percent(
                safe_float(item.get("avg_total_contribution", total_contrib))
            )
            formatted.append(cloned)
        return formatted

    def nearest_win_rate_at_time(win_rate_points: list[dict[str, Any]], t: float) -> float:
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

    def kill_team_label(killer: str, team1_players: list[str], team2_players: list[str]) -> str:
        if killer in team1_players:
            return "team1"
        if killer in team2_players:
            return "team2"
        return "unknown"

    def half_score(half_rounds: list[dict[str, Any]]) -> dict[str, int]:
        team1 = 0
        team2 = 0
        for item in half_rounds:
            winner = item.get("winner")
            if winner == "team1":
                team1 += 1
            elif winner == "team2":
                team2 += 1
        return {"team1": team1, "team2": team2}

    def make_half_meta(name: str, half_rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not half_rounds:
            return None

        first = half_rounds[0]
        team1_on_ct = bool(first.get("team1_on_ct", False))
        team1_side = "CT" if team1_on_ct else "T"
        team2_side = "T" if team1_on_ct else "CT"
        team1_role = "defense" if team1_on_ct else "attack"
        team2_role = "attack" if team1_on_ct else "defense"
        round_ids = [int(x.get("round_id", 0)) for x in half_rounds if isinstance(x.get("round_id"), int)]

        return {
            "name": name,
            "team1_side": team1_side,
            "team2_side": team2_side,
            "team1_role": team1_role,
            "team2_role": team2_role,
            "round_start": min(round_ids) if round_ids else None,
            "round_end": max(round_ids) if round_ids else None,
            "score": half_score(half_rounds),
        }

    rounds_data = []
    source_rounds = dashboard.get("rounds", [])
    for rd in source_rounds:
        win_rate = rd.get("win_rate", [])
        wr_values = [safe_float(x.get("team1_win_rate", 0.0)) for x in win_rate]
        team1_players = rd.get("team1_players", [])
        team2_players = rd.get("team2_players", [])
        team1_on_ct = bool(rd.get("team1_on_ct", False))
        team1_side = "CT" if team1_on_ct else "T"
        team2_side = "T" if team1_on_ct else "CT"
        team1_role = "defense" if team1_on_ct else "attack"
        team2_role = "attack" if team1_on_ct else "defense"
        round_start_inventory = rd.get("start_inventory", [])
        kills = sorted(rd.get("kills", []), key=lambda x: safe_float(x.get("round_seconds", 0.0)))

        if wr_values:
            start_wr = wr_values[0]
            end_wr = wr_values[-1]
            max_wr = max(wr_values)
            min_wr = min(wr_values)
        else:
            start_wr = end_wr = max_wr = min_wr = 0.0

        win_rate_timeline_text = [
            f"本回合阵营: team1={team1_side}({team1_role}), team2={team2_side}({team2_role})",
            f"回合开始 team1 胜率 {start_wr * 100:.1f}%",
            f"回合结束 team1 胜率 {end_wr * 100:.1f}%",
            f"回合内最高 {max_wr * 100:.1f}% / 最低 {min_wr * 100:.1f}%",
        ]

        kill_win_rate_transitions = []
        for kill in kills:
            t = safe_float(kill.get("round_seconds", 0.0))
            before = nearest_win_rate_at_time(win_rate, t - 0.12)
            after = nearest_win_rate_at_time(win_rate, t + 0.12)
            delta = after - before
            killer = kill.get("killer", "Unknown")
            assister = kill.get("assister")
            assister_text = assister if assister else "None"
            victim = kill.get("victim", "Unknown")
            weapon = kill.get("weapon", "Unknown")
            headshot = bool(kill.get("headshot", False))
            headshot_text = "Yes" if headshot else "No"
            killer_team = kill_team_label(killer, team1_players, team2_players)

            kill_win_rate_transitions.append(
                {
                    "round_seconds": t,
                    "killer": killer,
                    "assister": assister,
                    "assister_text": assister_text,
                    "victim": victim,
                    "weapon": weapon,
                    "headshot": headshot,
                    "headshot_text": headshot_text,
                    "killer_team": killer_team,
                    "team1_win_rate_before": before,
                    "team1_win_rate_after": after,
                    "team1_win_rate_delta": delta,
                    "description": (
                        f"{t:.2f}s {killer}({killer_team}) 击杀 {victim}({weapon}), "
                        f"assister={assister_text}, headshot={headshot_text}, "
                        f"team1 胜率 {before * 100:.1f}% -> {after * 100:.1f}% "
                        f"(变化 {delta * 100:+.1f}%)"
                    ),
                }
            )

        if kill_win_rate_transitions:
            win_rate_timeline_text.extend(
                [x["description"] for x in kill_win_rate_transitions]
            )

        rounds_data.append(
            {
                "round_id": rd.get("round_id"),
                "winner": rd.get("winner"),
                "winner_side": rd.get("winner_side"),
                "team1_side": team1_side,
                "team2_side": team2_side,
                "team1_role": team1_role,
                "team2_role": team2_role,
                "team1_players": team1_players,
                "team2_players": team2_players,
                "round_start_inventory": round_start_inventory,
                "win_rate_start": start_wr,
                "win_rate_end": end_wr,
                "win_rate_max": max_wr,
                "win_rate_min": min_wr,
                "swings": rd.get("swings", {}),
                "kills": kills,
                "kill_win_rate_transitions": kill_win_rate_transitions,
                "win_rate_timeline_text": win_rate_timeline_text,
                "round_final_player_contribution": format_contrib_items(
                    rd.get("round_summary", {}).get("per_player", [])
                ),
            }
        )

    # Derive first/second half by side-switch boundaries to reduce LLM hallucination.
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

        first_start = boundaries[0]
        first_end = boundaries[1] if len(boundaries) > 1 else len(source_rounds)
        first_half_rounds = source_rounds[first_start:first_end]

        if len(boundaries) > 2:
            second_start = boundaries[1]
            second_end = boundaries[2]
            second_half_rounds = source_rounds[second_start:second_end]
        elif len(boundaries) > 1:
            second_half_rounds = source_rounds[boundaries[1]:]

    match_halves = {
        "first_half": make_half_meta("first_half", first_half_rounds),
        "second_half": make_half_meta("second_half", second_half_rounds),
    }

    return {
        "match": dashboard.get("match", {}),
        "match_halves": match_halves,
        "overall_player_averages": format_contrib_items(dashboard.get("overall", [])),
        "rounds": rounds_data,
    }


def build_chat_completion_url(base_url: str) -> str:
    normalized = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def build_llm_prompts(llm_data: dict[str, Any], language: str) -> tuple[str, str]:
    lang = (language or "zh").strip().lower()
    if lang not in {"zh", "en"}:
        lang = "zh"

    if lang == "en":
        system_prompt = (
            "You are a professional CS2 tactical analyst. "
            "Use round win-rate curves, kill events, and player contribution data "
            "to produce an insightful and highly readable analysis in English. "
            "Focus on key rounds, high-impact kills, clutch turnarounds, and team momentum shifts."
        )
        user_prompt = (
            "Please complete the following tasks:\n"
            "1) Summarize each player's performance (highlight rounds, weak rounds, key kills, tactical value).\n"
            "2) Summarize each team (tempo, consistency, collapse points, comeback points).\n"
            "3) First explicitly report first-half and second-half side assignment and score: who is CT/T and attack/defense, and the half score for each side.\n"
            "   You MUST use match_halves.first_half and match_halves.second_half as the source of truth.\n"
            "4) round_start_inventory is optional context: use it only when it helps explain key rounds (economy/loadout advantages).\n"
            "5) Provide interesting stats (e.g., biggest 5-second drops, low-probability comebacks, high-probability throws).\n"
            "6) Select ONLY the most interesting/key rounds (about 3-6 rounds), instead of covering every round.\n"
            "   Prioritize rounds with major win-rate swings, high-impact kills, clutch/comeback moments, and tactical turning points.\n"
            "7) For each selected round, explicitly state attack/defense side (team1/team2 and T/CT), opening/ending win rate, and key kill transitions.\n"
            "   Prefer information from kill_win_rate_transitions and win_rate_timeline_text.\n"
            "   When describing contribution numbers, prioritize *_pct fields (signed percentage strings).\n"
            "   When describing kills, explicitly include assister_text and headshot_text (even when no assister).\n"
            "8) Provide MVP/SVP rationale and verify consistency with statistics.\n"
            "9) End with 3 actionable improvement suggestions.\n\n"
            "Structured data (JSON):\n"
            + json.dumps(llm_data, ensure_ascii=False)
        )
        return system_prompt, user_prompt

    system_prompt = (
        "你是专业的 CS2 战术分析师。"
        "请根据每回合胜率曲线、击杀事件、玩家贡献数据，"
        "输出有洞察力、可读性强的中文总结。"
        "要重点指出关键回合、高影响力击杀、残局翻盘、以及团队波动。"
    )
    user_prompt = (
        "请完成以下任务：\n"
        "1) 按玩家逐个总结表现（亮点回合、低谷回合、关键击杀、战术价值）。\n"
        "2) 按队伍总结（节奏、稳定性、崩盘时刻、翻盘时刻）。\n"
        "3) 先明确写出上半场和下半场的阵营归属与比分：谁是 CT/T、谁是进攻/防守，以及上下半场分别比分。\n"
        "   这部分必须以 match_halves.first_half 和 match_halves.second_half 为唯一依据。\n"
        "4) round_start_inventory 仅作为辅助上下文：只在解释关键回合时按需引用（例如经济局/装备优势），不需要逐回合罗列。\n"
        "5) 输出趣味数据（如5秒内胜率暴跌、低胜率翻盘、高胜率被翻盘）。\n"
        "6) 不需要逐回合全覆盖，只挑最有趣/最关键的回合进行讲解（建议 3-6 个回合）。\n"
        "   优先选择胜率波动大、关键击杀、残局翻盘、战术转折明显的回合。\n"
        "7) 对于每个被选中的回合，明确写出攻防归属（team1/team2 及 T/CT），并说明开局/结尾胜率与关键击杀前后变化。\n"
        "   优先使用 kill_win_rate_transitions 和 win_rate_timeline_text 中的信息。\n"
        "   描述贡献数值时优先使用 *_pct 字段（已带正负号和百分号）。\n"
        "   描述击杀时必须显式写出 assister_text 和 headshot_text（无助攻也要写出来）。\n"
        "8) 给出 MVP/SVP 评价理由，并验证是否与统计结果一致。\n"
        "9) 最后给出 3 条可执行改进建议。\n\n"
        "以下是结构化数据(JSON)：\n"
        + json.dumps(llm_data, ensure_ascii=False)
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
        "--model_path",
        model_path,
        "--device",
        device,
        "--output",
        str(output_path),
        "--batch_size",
        batch_size,
    ]

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
    return render_template(
        "index.html",
        default_device=choose_default_device(),
        model_options=discover_model_paths(),
    )


@app.get("/assets/<path:filename>")
def asset_file(filename: str):
    return send_from_directory(ASSETS_DIR, filename)


@app.post("/api/analyze")
def analyze_demo():
    dem_file = request.files.get("demo_file")
    if dem_file is None or dem_file.filename == "":
        return jsonify({"error": "请上传 .dem 文件"}), 400

    model_path = request.form.get("model_path", "").strip()
    device = request.form.get("device", choose_default_device()).strip()
    batch_size = request.form.get("batch_size", "32").strip()

    if not model_path:
        return jsonify({"error": "请先选择模型目录"}), 400

    model_dir = Path(model_path)
    if not model_dir.exists() or not model_dir.is_dir():
        return jsonify({"error": f"模型目录不存在或不是文件夹: {model_path}"}), 400

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
                "提示: 解析 demo 通常 30~60s，模型推理通常不到 1 分钟。",
            ],
            "error": "",
            "total_rounds": 0,
            "processed_rounds": 0,
            "current_round": None,
            "analysis_id": None,
            "dashboard": None,
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
