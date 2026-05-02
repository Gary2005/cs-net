import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator

from agent_framework import Message
from agent_framework.openai import OpenAIChatCompletionClient

from demo_analysis.high_level_analysis import safe_float


MAX_FEATURED_ROUNDS = 6
MAX_KILLS_PER_FEATURED_ROUND = 5
MAX_KILL_RANKING_ENTRIES = 10


@dataclass(frozen=True)
class LlmSummaryConfig:
    api_key: str
    model_name: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.95
    language: str = "zh"


def build_llm_payload(dashboard: dict[str, Any]) -> dict[str, Any]:
    """
    Compact payload for the LLM. Large raw arrays are dropped so reports stay
    focused on decisive rounds, players, and advanced metrics.
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
            featured_rounds.append({**r, "kill_transitions": trimmed_transitions})
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
        "match_halves": {
            "first_half": make_half_meta("first_half", first_half_rounds),
            "second_half": make_half_meta("second_half", second_half_rounds),
        },
        "whitelist": whitelist,
        "overall_player_averages": format_contrib_items(dashboard.get("overall", [])),
        "featured_rounds": featured_rounds,
        "other_rounds": other_rounds,
        "advanced_kill_ranking": adv_kill_ranking,
        "advanced_player_stats": adv_player_stats,
    }


def build_llm_prompts(llm_data: dict[str, Any], language: str) -> tuple[str, str]:
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


def _normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized + "/"


def _make_client(config: LlmSummaryConfig) -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        base_url=_normalize_base_url(config.base_url),
        api_key=config.api_key,
        model=config.model_name,
    )


async def llm_summary_stream(
    dashboard: dict[str, Any],
    config: LlmSummaryConfig,
) -> AsyncIterator[str]:
    llm_data = build_llm_payload(dashboard)
    system_prompt, user_prompt = build_llm_prompts(llm_data, config.language)
    client = _make_client(config)
    messages = [
        Message("system", [system_prompt]),
        Message("user", [user_prompt]),
    ]
    options = {"temperature": config.temperature}

    stream = client.get_response(messages, stream=True, options=options)
    async for chunk in stream:
        text = getattr(chunk, "text", "")
        if text:
            yield text


async def llm_summary(
    dashboard: dict[str, Any],
    config: LlmSummaryConfig,
) -> str:
    chunks = []
    async for chunk in llm_summary_stream(dashboard, config):
        chunks.append(chunk)
    return "".join(chunks)


def llm_summary_sync(
    dashboard: dict[str, Any],
    config: LlmSummaryConfig,
) -> str:
    return asyncio.run(llm_summary(dashboard, config))


def llm_summary_stream_sync(
    dashboard: dict[str, Any],
    config: LlmSummaryConfig,
) -> Iterator[str]:
    loop = asyncio.new_event_loop()
    async_iter = llm_summary_stream(dashboard, config)
    try:
        while True:
            try:
                yield loop.run_until_complete(async_iter.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.run_until_complete(async_iter.aclose())
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
