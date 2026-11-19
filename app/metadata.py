"""
Derive library metadata from an extracted demo.

Everything here works on the dict that ``demo_extract.codec.load`` returns, so
it can run against a freshly extracted demo or an existing ``.csz`` without
re-parsing the original.

What is *not* in the data, and therefore has to be inferred or left to the user:

* **team names** -- ``.csz`` carries only ``{steamid, name}`` per player.  The
  scoreboard is rebuilt from per-round sides plus the round winner, and the two
  team *labels* are guessed from the filename (``A-vs-B``) and otherwise left as
  "Team A"/"Team B" for the user to rename.
* **match source** -- not recorded anywhere reachable; inferred from the
  filename and overridable in the UI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

#: `A-vs-B`, `A_vs_B`, `A vs B` -- how HLTV-style archives name their files.
_VS_RE = re.compile(r"^(?P<a>[A-Za-z0-9_.]+?)[\s_-]+vs[\s_-]+(?P<b>[A-Za-z0-9_.]+)", re.I)

#: Trailing noise in downloaded filenames we should not treat as a team name.
_TEAM_NOISE = {"m1", "m2", "m3", "m4", "m5", "map1", "map2", "map3", "map4", "map5"}


def _title(name: str) -> str:
    """`vitality` -> `Vitality`, `aurora` -> `Aurora`, leaving odd casing alone."""
    if not name:
        return name
    if name.islower() or name.isupper():
        return name.capitalize()
    return name


def teams_from_filename(filename: str) -> tuple[Optional[str], Optional[str]]:
    """Guess ``(team_a, team_b)`` from a filename, or ``(None, None)``.

    Only the leading ``A-vs-B`` prefix is used, so
    ``aurora-vs-vitality-m2-inferno.dem`` yields ``("Aurora", "Vitality")``
    while an unrelated name yields nothing rather than a wrong answer.
    """
    stem = Path(filename).stem
    m = _VS_RE.match(stem)
    if not m:
        return None, None
    a, b = m.group("a"), m.group("b")
    if a.lower() in _TEAM_NOISE or b.lower() in _TEAM_NOISE:
        return None, None
    return _title(a), _title(b)


# ── team grouping ────────────────────────────────────────────────────────────


class _UnionFind:
    """Minimal union-find; 10 players means performance is irrelevant."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _team_groups(rounds: list[dict], n_players: int) -> list[list[int]]:
    """Split player indices into teams.

    Players are teammates when their side agrees in *every* round where both
    sides are known.  That is robust to the halftime swap (a team is always on
    one side at a time) and to a stand-in who only plays part of the match,
    where a naive "side in round 0" grouping would mis-assign them.
    """
    uf = _UnionFind(n_players)
    sides: list[list[str]] = []
    for rnd in rounds:
        teams = rnd.get("teams") or []
        sides.append([str(teams[i]) if i < len(teams) else "?" for i in range(n_players)])

    for i in range(n_players):
        for j in range(i + 1, n_players):
            agree = disagree = False
            for row in sides:
                si, sj = row[i], row[j]
                if si == "?" or sj == "?":
                    continue
                if si == sj:
                    agree = True
                else:
                    disagree = True
                if agree and disagree:
                    break
            # Teammates: never seen on opposite sides.  Requiring at least one
            # agreement stops two players who never overlapped from merging.
            if agree and not disagree:
                uf.union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(n_players):
        buckets.setdefault(uf.find(i), []).append(i)
    # Largest group first, then by index, so the result is deterministic.
    return sorted(buckets.values(), key=lambda g: (-len(g), g[0]))


def _round_winner_group(rnd: dict, groups: list[list[int]]) -> Optional[int]:
    """Which group index won this round, or None if it cannot be told."""
    winner = str(rnd.get("winner") or "").strip().upper()
    if winner not in ("T", "CT"):
        return None
    teams = rnd.get("teams") or []
    for gi, group in enumerate(groups):
        for idx in group:
            side = str(teams[idx]) if idx < len(teams) else "?"
            if side.upper() in ("T", "CT"):
                return gi if side.upper() == winner else (1 - gi if len(groups) == 2 else None)
    return None


# ── watched players ──────────────────────────────────────────────────────────


def match_watched(
    scoreboard: list[dict[str, Any]],
    watched_players: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Players from ``scoreboard`` whose name matches the watched list.

    Matching is a case-insensitive **substring** test, so a half-typed entry
    ("Zyw") still finds the player -- at the cost of a short entry matching by
    accident ("zy" also hits "Crazy").  At most one entry per player: a name
    matched by two watched entries is reported once.

    This is its own function because the watched list is editable *after* a demo
    has been extracted.  The API re-derives ``watched_json`` from the stored
    scoreboard through here, so there is exactly one implementation of the rule
    and the highlight cannot drift from what extraction produced.
    """
    wanted = [str(w).strip() for w in (watched_players or []) if str(w or "").strip()]
    if not wanted:
        return []
    watched: list[dict[str, Any]] = []
    for entry in scoreboard:
        name = str(entry.get("name") or "")
        low = name.lower()
        for w in wanted:
            if w.lower() in low:
                watched.append({
                    "name": name,
                    "matched": w,
                    "team": entry.get("team"),
                    "kills": entry.get("kills"),
                    "deaths": entry.get("deaths"),
                    "assists": entry.get("assists"),
                })
                break
    return watched


# ── the main entry point ─────────────────────────────────────────────────────


def build_metadata(
    data: dict,
    *,
    filename: str,
    watched_players: Optional[list[str]] = None,
    clans: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Turn an extracted demo into the fields the library row needs.

    ``clans`` maps steamid -> real team name (see :mod:`app.probe`).  When it is
    supplied the two teams are labelled with their actual names, which is the
    only way to get them right: ``.csz`` stores just ``{steamid, name}``, so
    without it the assignment of names to score columns would be a coin flip.

    Returns keys matching :mod:`app.db` columns (``players_json`` etc. hold
    ready-to-store JSON strings).
    """
    import json

    meta = data.get("meta") or {}
    roster = list(data.get("players") or [])
    rounds = list(data.get("rounds") or [])
    n = len(roster)

    groups = _team_groups(rounds, n) if n else []
    if len(groups) != 2:
        # A match with a stand-in can genuinely produce three overlapping
        # groups; fall back to the round-0 side split so the UI still has two
        # teams to show rather than an empty scoreboard.
        groups = _groups_by_first_round(rounds, n)

    scores = [0, 0]
    for rnd in rounds:
        gi = _round_winner_group(rnd, groups)
        if gi is not None and 0 <= gi < len(scores):
            scores[gi] += 1

    team_of = [0] * n
    for gi, group in enumerate(groups[:2]):
        for idx in group:
            if idx < n:
                team_of[idx] = gi

    # ── per-player scoreboard ───────────────────────────────────────────────
    kills = [0] * n
    deaths = [0] * n
    assists = [0] * n
    headshots = [0] * n
    damage = [0] * n

    for rnd in rounds:
        events = rnd.get("events") or {}
        for k in events.get("kills") or []:
            a, v, s = k.get("a", -1), k.get("v", -1), k.get("as", -1)
            if 0 <= a < n:
                kills[a] += 1
                damage[a] += int(k.get("dmg") or 0)
                if k.get("hs"):
                    headshots[a] += 1
            if 0 <= v < n:
                deaths[v] += 1
            if 0 <= s < n:
                assists[s] += 1

    scoreboard = []
    for i, p in enumerate(roster):
        k, d, a = kills[i], deaths[i], assists[i]
        scoreboard.append({
            "steamid": str(p.get("steamid") or ""),
            "name": str(p.get("name") or ""),
            "team": team_of[i],
            "kills": k,
            "deaths": d,
            "assists": a,
            "kd": round(k / d, 2) if d else float(k),
            "hs": headshots[i],
            "adr_damage": damage[i],
        })

    players_out = [
        {
            "steamid": str(p.get("steamid") or ""),
            "name": str(p.get("name") or ""),
            "team": team_of[i],
        }
        for i, p in enumerate(roster)
    ]

    # ── watched players ─────────────────────────────────────────────────────
    watched = match_watched(scoreboard, watched_players)

    # ── duration / date ─────────────────────────────────────────────────────
    tickrate = float(meta.get("tickrate") or meta.get("sidechannel_tickrate") or 64.0)
    total_ticks = int(meta.get("total_ticks") or 0)
    duration_mins = round(total_ticks / tickrate / 60.0, 1) if tickrate else None

    team_a, team_b = _team_labels(groups, roster, clans, filename)
    return {
        "map_name": meta.get("map"),
        "total_rounds": len(rounds),
        "duration_mins": duration_mins,
        "players_json": json.dumps(players_out, ensure_ascii=False),
        "scoreboard_json": json.dumps(scoreboard, ensure_ascii=False),
        "watched_json": json.dumps(watched, ensure_ascii=False),
        "team_a_name": team_a,
        "team_b_name": team_b,
        "team_a_score": scores[0] if len(scores) > 0 else None,
        "team_b_score": scores[1] if len(scores) > 1 else None,
        "server_name": meta.get("server_name"),
    }


def _team_labels(
    groups: list[list[int]],
    roster: list[dict],
    clans: Optional[dict[str, str]],
    filename: str,
) -> tuple[str, str]:
    """Name the two teams, preferring the demo's own clan names.

    Order of preference, most trustworthy first:

    1. the real clan name carried by the demo (``CCSTeam.m_szClanTeamname``)
    2. an ``A-vs-B`` filename
    3. "Team A" / "Team B"

    Step 1 is what keeps the score columns correct: the filename tells you
    *which* two teams played but not which one is group 0, whereas the clan
    field is attached to the players themselves.
    """
    labels: list[Optional[str]] = [None, None]

    if clans:
        for gi, group in enumerate(groups[:2]):
            tally: dict[str, int] = {}
            for idx in group:
                if idx >= len(roster):
                    continue
                clan = clans.get(str(roster[idx].get("steamid") or ""))
                if clan:
                    tally[clan] = tally.get(clan, 0) + 1
            if tally:
                labels[gi] = max(tally.items(), key=lambda kv: (kv[1], kv[0]))[0]

    if not any(labels):
        fa, fb = teams_from_filename(filename)
        labels = [fa, fb]

    return labels[0] or "Team A", labels[1] or "Team B"


def _groups_by_first_round(rounds: list[dict], n: int) -> list[list[int]]:
    """Fallback split: the two sides as they stood in the first round."""
    first = (rounds[0].get("teams") if rounds else None) or []
    ct = [i for i in range(n) if i < len(first) and str(first[i]).upper() == "CT"]
    t = [i for i in range(n) if i < len(first) and str(first[i]).upper() == "T"]
    rest = [i for i in range(n) if i not in ct and i not in t]
    # Anything unassigned goes to the smaller side so the split stays 2 groups.
    for i in rest:
        (ct if len(ct) <= len(t) else t).append(i)
    return [g for g in (ct, t) if g]


def match_date_from_file(path: Path) -> Optional[str]:
    """The file's mtime, formatted for display.

    Named ``match_date`` by the database column, but it is **not** the match
    date: a CS2 demo carries no wall-clock timestamp, so the best a fresh
    download offers is when it landed on disk.  The UI labels it 文件时间 for
    that reason -- do not put a calendar icon or "比赛日期" back on it.
    """
    try:
        import time

        return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
    except OSError:
        return None
