import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator

from agent_framework import ChatResponse, ChatResponseUpdate, Message
from agent_framework.openai import OpenAIChatCompletionClient

from demo_analysis.high_level_analysis import safe_float


MAX_FEATURED_ROUNDS = 6
MAX_KILLS_PER_FEATURED_ROUND = 5
MAX_KILL_RANKING_ENTRIES = 10
MAX_TIMELINE_EVENTS_PER_ROUND = 12
MAX_DETAILED_TACTICAL_ROUNDS = 8
MIN_DETAILED_SWING_PCT = 15.0
MAX_BRIEF_TIMELINE_EVENTS_PER_ROUND = 2

ZH_SYSTEM_PROMPT_TEMPLATE = """你是专业的 CS2 战术分析师，请输出中文复盘。

严格防幻觉与写作规则：
- 合法玩家名（不得发明其它名字）：{all_players}
- team1 阵容：{team1_players}
- team2 阵容：{team2_players}
- 合法展示回合 ID：{valid_round_ids}
- 当前 CS2 使用 MR12：展示回合 1-12 是上半场，13-24 是下半场；12:12 后进入加时。
- 加时是 MR3：每个加时块 6 回合，3 回合后换边；展示回合 25+ 都属于加时，绝不能算进下半场。
- 任何数字（胜率、swing、难度、贡献）必须直接来自下方 JSON，不许编造或过度取整。
- 如果某字段缺失，直接说明“数据中未提供”，不要猜。
- 不要编造 JSON 里没有的击杀、残局、武器、回合结果、爆弹、前压或道具。
- 报点必须基于 killer_location / victim_location 里的点位数据。callout_candidates 是候选点位列表；每项的 name 是点位名，distance 是当前位置到该点位边界的距离，size 是点位多边形面积，数值越大代表点位范围越大。优先使用 distance 为 0 的覆盖点位和距离更近的小点位组合自然报点：如果一个点被大点位包含，同时靠近某个小点位，可以写成“大点位小点位附近/位置”。不得发明候选外的点位。如果 callout_source 是 missing，就写“点位数据未提供”。
- 报点 few-shot 示例：候选为 [{{name:"B 包", distance:0, size:250000}}, {{name:"大箱", distance:38, size:12000}}]，可写“B 包大箱附近”；候选为 [{{name:"A 区", distance:0, size:300000}}, {{name:"三箱", distance:0, size:18000}}]，可写“A 区三箱位”；候选为 [{{name:"中路", distance:0, size:220000}}, {{name:"VIP", distance:90, size:25000}}]，可写“中路靠 VIP 一侧”。
- 不要向用户展示内部技术字段名，包括 detailed_tactical_rounds、brief_tactical_rounds、wr_start、wr_end、wr_start_pct、wr_end_pct、hard_win_rate、easy_win_rate、highlight_rate、evaluation_context、duel_context、callout_candidates。
- 评价可以夸击杀方，也可以批评被击杀方，但必须能从 JSON 数据推出。
- 回合级胜率只写“某队在开局胜率 X% 的情况下获胜”；不要写终局 0.0%/100.0%，也不要展示 wr_start/wr_end 这样的字段名。
- 详细回合的每个关键击杀可以、而且应该展示事件级数字：击杀方收益、Team1 胜率曲线变化、击杀难度、对决预测胜率。展示数字时用自然语言，不要写 JSON 字段名。
"""

ZH_USER_PROMPT_TEMPLATE = """请按如下结构输出（markdown，中文战报风格，信息要具体）：
1) 上/下半场与加时：严格以 match_halves.first_half、match_halves.second_half 和 match_overtimes 为准。下半场只包含展示回合 13-24；如果有加时，单独列出。
2) 转折回合详写：只遍历 JSON 里的详细转折回合列表，并按 round_display_id 顺序写。标题只写“回合 X”，不要写数据来源字段名。每回合先说明攻防方、开局装备概况，并读取 winner_start_win_rate_pct 但输出成“胜方开局胜率”，不要写字段名。然后按 timeline 写关键事件；对每个 kill 必须写清：击杀者、被击杀者、武器、点位、击杀方收益、Team1 胜率曲线变化、击杀难度、对决预测胜率，并基于这些数字给一句自然评价。
3) 普通回合速览：遍历普通回合列表，每回合只写一句话，优先使用 brief_summary_hint 概括该回合，不要展开成详细战报。
4) 团队叙事：节奏、稳定性、崩盘/翻盘瞬间。引用 overall_player_averages 与 match。
5) 玩家点评：结合 advanced_player_stats、overall_player_averages，以及详细回合里的 killer_team_swing_pct / difficulty / duel_win_rate。解释困难枪胜率、简单枪胜率、高光回合占比等用户友好名词；只评论白名单里的玩家。
6) 趣味数据：从 advanced_kill_ranking 里挑 top swing 的击杀，说明对应 difficulty。
7) MVP/SVP 复核：对照 advanced_player_stats 看 match.mvp / match.svp 是否合理，给出数据支持或反对。
8) 三条可执行改进建议。

以下是结构化数据(JSON)：
{data_json}"""

EN_SYSTEM_PROMPT_TEMPLATE = """You are a professional CS2 tactical analyst. Produce an insightful English review.

Strict anti-hallucination and writing rules:
- Valid player names (do NOT invent others): {all_players}
- team1 roster: {team1_players}
- team2 roster: {team2_players}
- Valid display round IDs: {valid_round_ids}
- Current CS2 uses MR12: display rounds 1-12 are the first half, 13-24 are the second half; overtime starts after 12-12.
- Overtime uses MR3: each overtime block has 6 rounds and sides switch after 3 rounds. Display rounds 25+ are overtime and must never be counted as the second half.
- Every numeric claim (win rate, swing, difficulty, contribution) MUST come directly from the JSON data. Do not fabricate values or round aggressively.
- If a field is missing, say so instead of guessing.
- Do not invent kills, clutches, weapons, round outcomes, utility, pushes, or tactics absent from the JSON.
- Location callouts must be grounded in killer_location / victim_location. callout_candidates is a list of candidate locations; name is the location name, distance is the distance from the current position to that location boundary, and size is polygon area, where larger values mean broader regions. Prefer distance 0 containing regions and nearby smaller landmarks together: if a point is inside a broad location and close to a smaller one, you may combine them naturally as "broad location near smaller landmark". Do not invent locations outside the candidates. If callout_source is missing, say location data is unavailable.
- Location few-shot examples: candidates [{{name:"B Site", distance:0, size:250000}}, {{name:"Van", distance:35, size:12000}}] -> "B Site near Van"; candidates [{{name:"A Site", distance:0, size:300000}}, {{name:"Triple", distance:0, size:18000}}] -> "A Site Triple"; candidates [{{name:"Mid", distance:0, size:220000}}, {{name:"Window", distance:80, size:25000}}] -> "Mid toward Window".
- Do not expose internal technical field names to the user, including detailed_tactical_rounds, brief_tactical_rounds, wr_start, wr_end, wr_start_pct, wr_end_pct, hard_win_rate, easy_win_rate, highlight_rate, evaluation_context, duel_context, callout_candidates.
- Round-level win-rate narration must say: “Team X won from an opening win probability of Y%”. Do not show terminal 0.0%/100.0% curve values or wr_start/wr_end field names.
- For detailed rounds, each key kill should show event-level numbers: killer-side gain, Team1 curve change, kill difficulty, and duel predicted win rate. Use natural language, not JSON field names.
"""

EN_USER_PROMPT_TEMPLATE = """Deliver the following sections (concise, well-structured markdown):
1) Halves and overtime: use match_halves.first_half, match_halves.second_half, and match_overtimes. The second half only contains display rounds 13-24; list overtime separately when present.
2) Detailed turning rounds: iterate through the detailed turning round list in round_display_id order. Headings should say only “Round X”; do not mention internal source field names. State sides and opening economy; read winner_start_win_rate_pct but write it as “winner opening win probability” without exposing the field name. Then narrate timeline events; for each kill, include killer, victim, weapon, locations, killer-side gain, Team1 curve change, kill difficulty, duel predicted win rate, and one natural judgment grounded in those numbers.
3) Brief ordinary rounds: iterate through the ordinary round list. Give one sentence per round, preferably using brief_summary_hint. Do not expand them into full reports.
4) Team narrative: tempo, consistency, collapse/comeback moments. Cite overall_player_averages + match.
5) Per-player review: cite advanced_player_stats, overall_player_averages, and detailed-round killer_team_swing_pct / difficulty / duel_win_rate. Explain user-friendly terms such as hard-duel win rate, easy-duel win rate, and highlight share. Only comment on whitelisted players.
6) Fun stats from advanced_kill_ranking: top swings and their difficulty.
7) MVP/SVP check: compare match.mvp / match.svp to advanced_player_stats; confirm or disagree with data.
8) Three actionable improvement suggestions.

Data (JSON):
{data_json}"""


@dataclass(frozen=True)
class LlmSummaryConfig:
    api_key: str
    model_name: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.95
    language: str = "zh"


def build_llm_payload(
    dashboard: dict[str, Any],
    max_detailed_rounds: int | None = None,
    language: str = "zh",
) -> dict[str, Any]:
    """
    Compact payload for the LLM. Large raw arrays are dropped so reports stay
    focused on decisive rounds, players, and advanced metrics.
    """

    source_rounds = dashboard.get("rounds", [])
    output_lang = "en" if (language or "").strip().lower() == "en" else "zh"

    def difficulty_value(raw: Any) -> float | None:
        # compute_kill_difficulty returns -1 as a sentinel when data is unavailable
        v = safe_float(raw, -1.0)
        return None if v < 0 else round(v, 3)

    def localize_location(loc: Any) -> Any:
        if not isinstance(loc, dict):
            return loc
        candidates_key = "callout_candidates_en" if output_lang == "en" else "callout_candidates_cn"
        candidates = loc.get(candidates_key) or loc.get("callout_candidates")
        compact = {
            "callout_source": loc.get("callout_source"),
            "callout_candidates": candidates or [],
        }
        if not compact["callout_candidates"] and loc.get("name") is not None:
            compact["name"] = loc.get("name")
        return compact

    def localize_event_locations(event: Any) -> Any:
        if not isinstance(event, dict):
            return event
        updated = dict(event)
        if "killer_location" in updated:
            updated["killer_location"] = localize_location(updated["killer_location"])
        if "victim_location" in updated:
            updated["victim_location"] = localize_location(updated["victim_location"])
        if "difficulty" in updated:
            updated["difficulty"] = difficulty_value(updated["difficulty"])
        return updated

    def localize_timeline(timeline: Any) -> list[Any]:
        return [localize_event_locations(event) for event in (timeline or [])]
    try:
        detailed_round_limit = (
            MAX_DETAILED_TACTICAL_ROUNDS
            if max_detailed_rounds is None
            else max(0, int(max_detailed_rounds))
        )
    except (TypeError, ValueError):
        detailed_round_limit = MAX_DETAILED_TACTICAL_ROUNDS
    raw_round_ids = [
        x.get("round_id")
        for x in source_rounds
        if isinstance(x.get("round_id"), int)
    ]
    raw_round_ids.extend(
        x.get("round_id")
        for x in (dashboard.get("tactical_rounds", []) or [])
        if isinstance(x.get("round_id"), int)
    )
    raw_round_ids.extend(
        x.get("round")
        for x in ((dashboard.get("advanced", {}) or {}).get("kill_ranking") or [])
        if isinstance(x.get("round"), int)
    )
    round_display_offset = 1 if raw_round_ids and min(raw_round_ids) == 0 else 0

    def display_round_id(round_id: Any) -> Any:
        if isinstance(round_id, int):
            return round_id + round_display_offset
        return round_id

    def signed_percent(value: float) -> str:
        return f"{safe_float(value, 0.0) * 100.0:+.2f}%"

    def percent_label(value: float) -> str:
        return f"{safe_float(value, 0.0) * 100.0:.1f}%"

    def display_team(team: Any) -> str:
        if team == "team1":
            return "Team1"
        if team == "team2":
            return "Team2"
        return str(team or "Unknown")

    def side_for_winner(round_data: dict[str, Any], winner: Any) -> str:
        if winner == "team1":
            return str(round_data.get("team1_side", "Unknown"))
        if winner == "team2":
            return str(round_data.get("team2_side", "Unknown"))
        return "Unknown"

    def winner_start_pct(winner: Any, team1_start_pct: Any) -> float:
        team1_pct = safe_float(team1_start_pct, 0.0)
        if winner == "team1":
            return round(team1_pct, 1)
        if winner == "team2":
            return round(100.0 - team1_pct, 1)
        return 0.0

    def loser_start_pct(winner: Any, team1_start_pct: Any) -> float:
        team1_pct = safe_float(team1_start_pct, 0.0)
        if winner == "team1":
            return round(100.0 - team1_pct, 1)
        if winner == "team2":
            return round(team1_pct, 1)
        return 0.0

    def add_winner_context(round_data: dict[str, Any]) -> dict[str, Any]:
        winner = round_data.get("winner")
        start_pct = round_data.get("wr_start_pct", 0.0)
        return {
            **round_data,
            "winner_team": winner,
            "winner_label": display_team(winner),
            "winner_side": side_for_winner(round_data, winner),
            "winner_start_win_rate_pct": winner_start_pct(winner, start_pct),
            "loser_start_win_rate_pct": loser_start_pct(winner, start_pct),
        }

    def brief_event_text(event: dict[str, Any]) -> str:
        if event.get("type") == "kill":
            facts = event.get("summary_facts") or {}
            killer = event.get("killer", facts.get("killer", "Unknown"))
            victim = event.get("victim", facts.get("victim", "Unknown"))
            weapon = event.get("weapon", facts.get("weapon", "Unknown"))
            swing = safe_float(
                event.get(
                    "killer_team_swing_pct",
                    (event.get("evaluation_context") or {}).get(
                        "killer_team_swing_pct",
                        event.get("wr_delta_pct", 0.0),
                    ),
                ),
            )
            return f"{killer} 用 {weapon} 击杀 {victim}，击杀方收益 {swing:+.1f}%"
        if event.get("type") == "bomb_planted":
            return "下包改变回合走势"
        return "关键事件数据较少"

    def brief_summary_hint(round_data: dict[str, Any]) -> str:
        winner = display_team(round_data.get("winner"))
        side = side_for_winner(round_data, round_data.get("winner"))
        start_pct = winner_start_pct(round_data.get("winner"), round_data.get("wr_start_pct", 0.0))
        swing = safe_float(round_data.get("largest_swing_pct", 0.0))
        events = round_data.get("key_events") or []
        event_bits = [brief_event_text(event) for event in events]
        event_text = "；".join(event_bits) if event_bits else "关键事件数据较少"
        return (
            f"回合 {round_data.get('round_display_id')}：{winner}({side}) "
            f"在开局胜率 {start_pct:.1f}% 的情况下获胜，最大摆动 {swing:.1f}%；{event_text}。"
        )

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
        round_ids = [
            int(display_round_id(x.get("round_id")))
            for x in half_rounds
            if isinstance(x.get("round_id"), int)
        ]
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
                    "difficulty": difficulty_value(kill.get("difficulty")),
                    "wr_delta_pct": round(delta * 100.0, 1),
                }
            )

        peak_delta = max((abs(x["wr_delta_pct"]) for x in transitions), default=0.0)
        full_rounds.append(
            add_winner_context(
                {
                    "round_id": rd.get("round_id"),
                    "round_display_id": display_round_id(rd.get("round_id")),
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
                    "round_display_id": r["round_display_id"],
                    "winner": r["winner"],
                    "winner_team": r["winner_team"],
                    "winner_label": r["winner_label"],
                    "winner_side": r["winner_side"],
                    "team1_side": r["team1_side"],
                    "team2_side": r["team2_side"],
                    "wr_start_pct": r["wr_start_pct"],
                    "wr_end_pct": r["wr_end_pct"],
                    "winner_start_win_rate_pct": r["winner_start_win_rate_pct"],
                    "loser_start_win_rate_pct": r["loser_start_win_rate_pct"],
                    "kill_count": len(r["kill_transitions"]),
                }
            )

    featured_rounds.sort(key=lambda r: r["round_id"])
    other_rounds.sort(key=lambda r: r["round_id"])

    first_half_rounds = [
        rd
        for rd in source_rounds
        if isinstance(rd.get("round_id"), int)
        and 1 <= display_round_id(rd.get("round_id")) <= 12
    ]
    second_half_rounds = [
        rd
        for rd in source_rounds
        if isinstance(rd.get("round_id"), int)
        and 13 <= display_round_id(rd.get("round_id")) <= 24
    ]
    overtime_rounds = [
        rd
        for rd in source_rounds
        if isinstance(rd.get("round_id"), int)
        and display_round_id(rd.get("round_id")) >= 25
    ]
    match_overtimes = []
    max_display_round = max(
        [display_round_id(rd.get("round_id")) for rd in overtime_rounds],
        default=24,
    )
    for block_start in range(25, max_display_round + 1, 6):
        block_rounds = [
            rd
            for rd in overtime_rounds
            if block_start <= display_round_id(rd.get("round_id")) <= block_start + 5
        ]
        if not block_rounds:
            continue
        first_three = [
            rd
            for rd in block_rounds
            if block_start <= display_round_id(rd.get("round_id")) <= block_start + 2
        ]
        second_three = [
            rd
            for rd in block_rounds
            if block_start + 3 <= display_round_id(rd.get("round_id")) <= block_start + 5
        ]
        block_meta = make_half_meta(
            f"overtime_{((block_start - 25) // 6) + 1}",
            block_rounds,
        )
        if block_meta:
            block_meta["side_periods"] = [
                x
                for x in [
                    make_half_meta("first_3_rounds", first_three),
                    make_half_meta("second_3_rounds", second_three),
                ]
                if x
            ]
            match_overtimes.append(block_meta)

    advanced = dashboard.get("advanced", {}) or {}
    adv_kill_ranking = [
        {
            "round": k.get("round"),
            "round_display_id": display_round_id(k.get("round")),
            "t": round(safe_float(k.get("round_seconds", 0.0)), 2),
            "attacker": k.get("attacker"),
            "victim": k.get("victim"),
            "swing_pct": signed_percent(k.get("swing", 0.0)),
            "difficulty": difficulty_value(k.get("difficulty")),
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
            "avg_kill_opp_label": "平均击杀机会",
            "avg_death_opp_label": "平均阵亡威胁",
            "avg_survive_chance_label": "平均存活率",
            "hard_duel_win_rate_label": "困难枪胜率",
            "easy_duel_win_rate_label": "简单枪胜率",
            "highlight_rate_label": "高光回合占比",
            "hard_duel_win_rate_pct": percent_label(p.get("hard_win_rate", 0.0)),
            "easy_duel_win_rate_pct": percent_label(p.get("easy_win_rate", 0.0)),
            "highlight_rate_pct": percent_label(p.get("highlight_rate", 0.0)),
        }
        for p in (advanced.get("player_stats") or [])
    ]

    match_info = dashboard.get("match", {}) or {}
    tactical_rounds = []
    for rd in dashboard.get("tactical_rounds", []) or []:
        timeline = localize_timeline(rd.get("timeline") or [])
        tactical_rounds.append(
            add_winner_context(
                {
                    "round_id": rd.get("round_id"),
                    "round_display_id": display_round_id(rd.get("round_id")),
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
        rd.get("round_id") for rd in detailed_candidates[:detailed_round_limit]
    }
    if detailed_round_limit > 0 and tactical_rounds and not detailed_ids:
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
        timeline = localize_timeline(rd.get("timeline") or [])
        brief_events = sorted(
            timeline,
            key=lambda ev: abs(safe_float(ev.get("wr_delta_pct", 0.0))),
            reverse=True,
        )[:MAX_BRIEF_TIMELINE_EVENTS_PER_ROUND]
        brief_events.sort(key=lambda ev: safe_float(ev.get("t", 0.0)))
        brief_tactical_rounds.append(
            {
                "round_id": rd.get("round_id"),
                "round_display_id": rd.get("round_display_id"),
                "map_name": rd.get("map_name"),
                "winner": rd.get("winner"),
                "winner_team": rd.get("winner_team"),
                "winner_label": rd.get("winner_label"),
                "winner_side": rd.get("winner_side"),
                "team1_side": rd.get("team1_side"),
                "team2_side": rd.get("team2_side"),
                "wr_start_pct": rd.get("wr_start_pct"),
                "wr_end_pct": rd.get("wr_end_pct"),
                "winner_start_win_rate_pct": rd.get("winner_start_win_rate_pct"),
                "loser_start_win_rate_pct": rd.get("loser_start_win_rate_pct"),
                "largest_swing_pct": takeaway.get("largest_swing_pct"),
                "event_count": takeaway.get("event_count"),
                "key_events": brief_events,
            }
        )
        brief_tactical_rounds[-1]["brief_summary_hint"] = brief_summary_hint(
            brief_tactical_rounds[-1]
        )
    brief_tactical_rounds.sort(key=lambda rd: safe_float(rd.get("round_id"), 10**9))

    whitelist = {
        "team1_players": sorted(match_info.get("team1_players", []) or []),
        "team2_players": sorted(match_info.get("team2_players", []) or []),
        "valid_round_ids": sorted([r["round_display_id"] for r in full_rounds if r["round_display_id"] is not None]),
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
        "match_overtimes": match_overtimes,
        "round_id_display_offset": round_display_offset,
        "whitelist": whitelist,
        "map_name": map_names[0] if len(map_names) == 1 else map_names,
        "map_callout_coverage": dashboard.get("map_callout_coverage", {}),
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
        system_prompt = EN_SYSTEM_PROMPT_TEMPLATE.format(
            all_players=all_players,
            team1_players=team1_players,
            team2_players=team2_players,
            valid_round_ids=valid_round_ids,
        )
        user_prompt = EN_USER_PROMPT_TEMPLATE.format(data_json=data_json)
        return system_prompt, user_prompt

    system_prompt = ZH_SYSTEM_PROMPT_TEMPLATE.format(
        all_players=all_players,
        team1_players=team1_players,
        team2_players=team2_players,
        valid_round_ids=valid_round_ids,
    )
    user_prompt = ZH_USER_PROMPT_TEMPLATE.format(data_json=data_json)
    return system_prompt, user_prompt


def _log_token_usage(response: ChatResponse[Any] | None) -> None:
    usage = (response.usage_details if response is not None else None) or {}
    print(
        "[llm_summary] "
        f"input_tokens={usage.get('input_token_count', 'unknown')}, "
        f"output_tokens={usage.get('output_token_count', 'unknown')}, "
        f"total_tokens={usage.get('total_token_count', 'unknown')}"
    )


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
    max_detailed_rounds: int | None = None,
) -> AsyncIterator[str]:
    llm_data = build_llm_payload(
        dashboard,
        max_detailed_rounds=max_detailed_rounds,
        language=config.language,
    )
    system_prompt, user_prompt = build_llm_prompts(llm_data, config.language)
    client = _make_client(config)
    messages = [
        Message("system", [system_prompt]),
        Message("user", [user_prompt]),
    ]
    options = {"temperature": config.temperature}

    stream = client.get_response(messages, stream=True, options=options)
    chunks: list[ChatResponseUpdate] = []
    try:
        async for chunk in stream:
            chunks.append(chunk)
            text = getattr(chunk, "text", "")
            if text:
                yield text
    finally:
        response = ChatResponse.from_updates(chunks) if chunks else None
        _log_token_usage(response)


async def llm_summary(
    dashboard: dict[str, Any],
    config: LlmSummaryConfig,
    max_detailed_rounds: int | None = None,
) -> str:
    chunks = []
    async for chunk in llm_summary_stream(
        dashboard,
        config,
        max_detailed_rounds=max_detailed_rounds,
    ):
        chunks.append(chunk)
    return "".join(chunks)


def llm_summary_sync(
    dashboard: dict[str, Any],
    config: LlmSummaryConfig,
    max_detailed_rounds: int | None = None,
) -> str:
    return asyncio.run(
        llm_summary(
            dashboard,
            config,
            max_detailed_rounds=max_detailed_rounds,
        )
    )


def llm_summary_stream_sync(
    dashboard: dict[str, Any],
    config: LlmSummaryConfig,
    max_detailed_rounds: int | None = None,
) -> Iterator[str]:
    loop = asyncio.new_event_loop()
    async_iter = llm_summary_stream(
        dashboard,
        config,
        max_detailed_rounds=max_detailed_rounds,
    )
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
