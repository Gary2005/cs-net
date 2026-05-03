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
MAX_TIMELINE_EVENTS_PER_ROUND = 12
MAX_DETAILED_TACTICAL_ROUNDS = 8
MIN_DETAILED_SWING_PCT = 15.0
MAX_BRIEF_TIMELINE_EVENTS_PER_ROUND = 2


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
    tactical_rounds = []
    for rd in dashboard.get("tactical_rounds", []) or []:
        timeline = rd.get("timeline") or []
        tactical_rounds.append(
            {
                "round_id": rd.get("round_id"),
                "map_name": rd.get("map_name"),
                "winner": rd.get("winner"),
                "team1_side": rd.get("team1_side"),
                "team2_side": rd.get("team2_side"),
                "economy_summary": rd.get("economy_summary"),
                "wr_start_pct": rd.get("wr_start_pct"),
                "wr_end_pct": rd.get("wr_end_pct"),
                "timeline": timeline[:MAX_TIMELINE_EVENTS_PER_ROUND],
                "round_takeaway": rd.get("round_takeaway"),
            }
        )
    detailed_candidates = [
        rd
        for rd in tactical_rounds
        if safe_float((rd.get("round_takeaway") or {}).get("largest_swing_pct", 0.0))
        >= MIN_DETAILED_SWING_PCT
    ]
    detailed_candidates.sort(
        key=lambda rd: safe_float((rd.get("round_takeaway") or {}).get("largest_swing_pct", 0.0)),
        reverse=True,
    )
    detailed_ids = {
        rd.get("round_id") for rd in detailed_candidates[:MAX_DETAILED_TACTICAL_ROUNDS]
    }
    if tactical_rounds and not detailed_ids:
        strongest = max(
            tactical_rounds,
            key=lambda rd: safe_float((rd.get("round_takeaway") or {}).get("largest_swing_pct", 0.0)),
        )
        detailed_ids = {strongest.get("round_id")}

    detailed_tactical_rounds = [
        rd for rd in tactical_rounds if rd.get("round_id") in detailed_ids
    ]
    detailed_tactical_rounds.sort(key=lambda rd: safe_float(rd.get("round_id"), 10**9))

    brief_tactical_rounds = []
    for rd in tactical_rounds:
        if rd.get("round_id") in detailed_ids:
            continue
        takeaway = rd.get("round_takeaway") or {}
        timeline = rd.get("timeline") or []
        brief_events = sorted(
            timeline,
            key=lambda ev: abs(safe_float(ev.get("wr_delta_pct", 0.0))),
            reverse=True,
        )[:MAX_BRIEF_TIMELINE_EVENTS_PER_ROUND]
        brief_events.sort(key=lambda ev: safe_float(ev.get("t", 0.0)))
        brief_tactical_rounds.append(
            {
                "round_id": rd.get("round_id"),
                "map_name": rd.get("map_name"),
                "winner": rd.get("winner"),
                "team1_side": rd.get("team1_side"),
                "team2_side": rd.get("team2_side"),
                "wr_start_pct": rd.get("wr_start_pct"),
                "wr_end_pct": rd.get("wr_end_pct"),
                "largest_swing_pct": takeaway.get("largest_swing_pct"),
                "event_count": takeaway.get("event_count"),
                "key_events": brief_events,
            }
        )
    brief_tactical_rounds.sort(key=lambda rd: safe_float(rd.get("round_id"), 10**9))

    whitelist = {
        "team1_players": sorted(match_info.get("team1_players", []) or []),
        "team2_players": sorted(match_info.get("team2_players", []) or []),
        "valid_round_ids": sorted([r["round_id"] for r in full_rounds if r["round_id"] is not None]),
    }
    map_names = sorted(
        {
            str(r.get("map_name"))
            for r in source_rounds
            if r.get("map_name") is not None
        }
    )

    return {
        "match": {
            "map_name": map_names[0] if len(map_names) == 1 else map_names,
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
        "map_name": map_names[0] if len(map_names) == 1 else map_names,
        "map_callout_coverage": dashboard.get("map_callout_coverage", {}),
        "tactical_rounds": detailed_tactical_rounds,
        "detailed_tactical_rounds": detailed_tactical_rounds,
        "brief_tactical_rounds": brief_tactical_rounds,
        "overall_player_averages": format_contrib_items(dashboard.get("overall", [])),
        "featured_rounds": [] if detailed_tactical_rounds else featured_rounds,
        "other_rounds": [] if detailed_tactical_rounds else other_rounds,
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
            "- For locations, only use locations.*.name from detailed_tactical_rounds / brief_tactical_rounds. "
            "If callout_source is missing, say location data is unavailable.\n"
        )
        user_prompt = (
            "Deliver the following sections (concise, well-structured markdown):\n"
            "1) Halves & score: cite match_halves.first_half / .second_half verbatim (sides and score).\n"
            "2) Detailed turning rounds: iterate through detailed_tactical_rounds in round order. For each round, state sides, "
            "economy_summary, wr_start_pct -> wr_end_pct, then narrate timeline events using provided locations, alive_score, "
            "utility_state, weapon, wr_delta_pct, difficulty, duel_context, and evaluation_context. For each kill, write a natural judgment "
            "based on win-rate swing, difficulty, and duel probability; you may praise the winner or criticize the loser. Do not use fixed labels.\n"
            "3) Brief ordinary rounds: iterate through brief_tactical_rounds in round order. Give one sentence per round using winner, "
            "wr_start_pct -> wr_end_pct, largest_swing_pct, and at most key_events. Do not expand these rounds.\n"
            "4) Team narrative: tempo, consistency, collapse/comeback moments. Cite overall_player_averages + match.\n"
            "5) Per-player review: cite advanced_player_stats alongside overall_player_averages, and use evaluation_context/difficulty/duel_context "
            "from detailed_tactical_rounds when judging decisions. Only comment on players in the whitelist.\n"
            "6) Fun stats from advanced_kill_ranking (top swings and their difficulty).\n"
            "7) MVP/SVP check: compare match.mvp / match.svp to advanced_player_stats; confirm or disagree with data.\n"
            "8) Three actionable improvement suggestions.\n\n"
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
        "- 点位只能使用 detailed_tactical_rounds / brief_tactical_rounds 中 locations.*.name；如果 callout_source 是 missing，就写“点位数据未提供”。\n"
        "- 不要使用固定操作评价标签；每个 kill 都要根据胜率变化、difficulty、duel_context 和 evaluation_context 自然评价。\n"
        "- 评价可以夸击杀方，也可以批评被击杀方，但必须能从 JSON 数据推出。\n"
    )
    user_prompt = (
        "请按如下结构输出（markdown，中文战报风格，信息要具体）：\n"
        "1) 上/下半场阵营与比分：严格以 match_halves.first_half / .second_half 为准。\n"
        "2) 转折回合详写：只遍历 detailed_tactical_rounds，并按 round_id 顺序写。每回合说明攻防方、开局装备概况、wr_start_pct → wr_end_pct，"
        "再按 timeline 写每个关键事件；必须写出提供的点位、人数、utility_state、武器、胜率变化、difficulty、duel_context 和 evaluation_context。"
        "每个 kill 都要给一句自然评价：结合击杀方胜率收益、对枪难度、duel 胜率，判断是漂亮发挥、关键补枪、被抓失误、冒险成功，或信息不足；不要套用固定标签。"
        "如果 timeline 没有点位，就明确写点位数据未提供；不要补写 JSON 没有的爆弹、前压、残局或道具。\n"
        "3) 普通回合速览：遍历 brief_tactical_rounds，每回合只写一句话，使用 winner、wr_start_pct → wr_end_pct、largest_swing_pct 和最多 key_events；不要展开成详细战报。\n"
        "4) 团队叙事：节奏、稳定性、崩盘/翻盘瞬间。引用 overall_player_averages 与 match。\n"
        "5) 玩家点评：结合 advanced_player_stats、overall_player_averages，以及 detailed_tactical_rounds 里的 evaluation_context / difficulty / duel_context，评价每个人的关键操作质量；只评论白名单里的玩家。\n"
        "6) 趣味数据：从 advanced_kill_ranking 里挑 top swing 的击杀，说明对应 difficulty。\n"
        "7) MVP/SVP 复核：对照 advanced_player_stats 看 match.mvp / match.svp 是否合理，给出数据支持或反对。\n"
        "8) 三条可执行改进建议。\n\n"
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
