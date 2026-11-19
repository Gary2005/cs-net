"""
Cheap demo inspection that does not require a full extraction.

Two reads, both fast enough to run the moment a file is discovered:

* ``parse_header()`` -- the demo header. Measured at well under a millisecond,
  and it carries ``map_name`` and ``server_name``.  The server name is what
  makes source classification (5E / 完美 / 官匹 / FACEIT / ...) trustworthy,
  because it comes from the server rather than from a filename the user may
  have renamed.
* ``parse_ticks([CCSTeam.m_szClanTeamname, team_num, name], ticks=[...])`` --
  ~0.4 s, and it is the *only* place real team names exist.  ``.csz`` stores
  just ``{steamid, name}`` per player, so without this the scoreboard would
  have to label teams "Team A"/"Team B".

Everything here is best-effort: a demo that cannot be read returns what was
obtained plus an ``error`` string, and the caller falls back to filename
heuristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

#: Ticks probed for team names.  Tick 0 is included because the clan fields are
#: already populated there; the others are a fallback for demos whose warmup
#: starts later.
_PROBE_TICKS = [0, 64, 128, 256]

#: Exceptions that mean "the process is going away", not "this demo is bad".
_FATAL = (KeyboardInterrupt, SystemExit, GeneratorExit)

# Why the handlers below catch ``BaseException`` rather than ``Exception``:
#
# demoparser is Rust, and a malformed or truncated demo makes it *panic*.  PyO3
# surfaces that as ``PanicException``, which deliberately inherits from
# **BaseException** -- so the obvious ``except Exception`` lets it straight
# through.  One half-downloaded ``.dem`` would then abort a whole directory
# scan, and kill the watcher thread outright when it happened there.
#
# It cannot be caught by name either: the type is raised but is *not* exposed as
# an attribute of ``demoparser2`` or ``pyo3_runtime``, so there is nothing to
# put in an ``except`` clause.  Enumerating it is a dead end; catching
# ``BaseException`` and re-raising the genuine shutdown signals is the only
# form that actually holds.  ``app/tests/test_library.py`` triggers a real panic
# and asserts the type is not an ``Exception`` subclass, so this reasoning is
# checked rather than merely asserted.


def probe_demo(path: str | Path) -> dict[str, Any]:
    """Read header + team names from ``path`` without extracting.

    Returns::

        {
          "ok": bool,
          "map_name": str | None,
          "server_name": str | None,
          "teams": {steamid: clan_name},      # real names, when present
          "players": [{steamid, name, team_num}],
          "error": str | None,
        }

    Never raises for a bad demo -- see :func:`panic_exception_types` for why
    that takes a ``BaseException`` handler rather than an ``Exception`` one.
    """
    path = Path(path)
    out: dict[str, Any] = {
        "ok": False,
        "map_name": None,
        "server_name": None,
        "teams": {},
        "players": [],
        "error": None,
    }
    try:
        import demoparser2
    except ImportError as exc:  # pragma: no cover - environment problem
        out["error"] = f"demoparser2 不可用：{exc}"
        return out

    try:
        parser = demoparser2.DemoParser(str(path))
        header = parser.parse_header() or {}
        out["map_name"] = header.get("map_name")
        out["server_name"] = header.get("server_name")
    except _FATAL:
        raise
    except BaseException as exc:  # any parser failure is non-fatal
        out["error"] = _describe(exc, "读取 demo 头失败")
        return out

    try:
        df = parser.parse_ticks(
            ["CCSTeam.m_szClanTeamname", "team_num", "name"], ticks=_PROBE_TICKS
        )
        if df is not None and len(df):
            players: dict[str, dict[str, Any]] = {}
            # NOTE: use to_dict("records"), not itertuples().  pandas renames
            # columns that are not valid Python identifiers, and
            # "CCSTeam.m_szClanTeamname" is one -- itertuples would silently
            # hand back `_1` and every clan lookup would miss.
            for d in df.to_dict("records"):
                sid = str(d.get("steamid", ""))
                if not sid or sid == "nan":
                    continue
                clan = d.get("CCSTeam.m_szClanTeamname")
                entry = players.setdefault(
                    sid, {"steamid": sid, "name": str(d.get("name") or ""), "team_num": None}
                )
                if clan and not entry.get("clan"):
                    entry["clan"] = str(clan).strip()
                tn = d.get("team_num")
                if tn is not None and tn == tn:  # not NaN
                    entry["team_num"] = int(tn)
            out["players"] = list(players.values())
            out["teams"] = {
                sid: p["clan"] for sid, p in players.items() if p.get("clan")
            }
        out["ok"] = True
    except _FATAL:
        raise
    except BaseException as exc:
        # Header data is still useful, so keep it and report the partial failure.
        out["error"] = _describe(exc, "读取队伍名失败")
        out["ok"] = True

    return out


def _describe(exc: BaseException, prefix: str) -> str:
    """A one-line message.  Panic text is multi-line Rust output."""
    text = " ".join(str(exc).split())
    return f"{prefix}：{type(exc).__name__}: {text[:200]}"


def team_names_from_probe(probe: dict[str, Any]) -> list[str]:
    """Distinct real team names, most common first."""
    counts: dict[str, int] = {}
    for clan in (probe.get("teams") or {}).values():
        if clan:
            counts[clan] = counts.get(clan, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def clan_by_steamid(probe: dict[str, Any]) -> dict[str, str]:
    return dict(probe.get("teams") or {})
