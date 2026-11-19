"""
SQLite storage for the demo library.

One row per demo.  The row carries three kinds of data that arrive at different
times, which is why the table is wider than it first looks:

* **identity** -- path / size / md5, known the moment a file is discovered
* **parsed metadata** -- map, score, roster, scoreboard; only known after the
  demo has been extracted to ``.csz`` (or read back from one)
* **user annotations** -- display name, notes, source override; edited by hand
  and never overwritten by a re-parse

``status`` tracks how far along a row is::

    discovered -> extracting -> ready
                      |
                      +------> error

Threading: the server runs Flask with ``threaded=True`` plus a watcher thread,
so every call opens its own short-lived connection (WAL mode makes concurrent
reads safe) and writes are serialised by a module lock.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config

#: A demo whose file is present but which has not been extracted yet.
STATUS_DISCOVERED = "discovered"
STATUS_EXTRACTING = "extracting"
STATUS_READY = "ready"
STATUS_ERROR = "error"
STATUS_MISSING = "missing"

ALL_STATUSES = [
    STATUS_DISCOVERED,
    STATUS_EXTRACTING,
    STATUS_READY,
    STATUS_ERROR,
    STATUS_MISSING,
]

STATUS_LABELS = {
    STATUS_DISCOVERED: "待入库",
    STATUS_EXTRACTING: "提取中",
    STATUS_READY: "已入库",
    STATUS_ERROR: "失败",
    STATUS_MISSING: "文件丢失",
}

_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS demo_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,
    filename        TEXT    NOT NULL,
    file_size       INTEGER,
    content_md5     TEXT,
    csz_path        TEXT,
    status          TEXT    NOT NULL DEFAULT 'discovered',
    source          TEXT,
    watch_root      TEXT,

    map_name        TEXT,
    total_rounds    INTEGER,
    duration_mins   REAL,
    match_date      TEXT,  -- the file's mtime, NOT the match date; see metadata.match_date_from_file
    server_name     TEXT,
    team_a_name     TEXT,
    team_b_name     TEXT,
    team_a_score    INTEGER,
    team_b_score    INTEGER,
    players_json    TEXT,
    scoreboard_json TEXT,
    watched_json    TEXT,

    display_name    TEXT,
    remark          TEXT,

    added_at        TEXT    NOT NULL,
    updated_at      TEXT,
    parsed_at       TEXT,
    error_msg       TEXT,
    thumb_path      TEXT,
    thumb_key       TEXT
);

CREATE INDEX IF NOT EXISTS idx_demo_md5      ON demo_files(content_md5);
CREATE INDEX IF NOT EXISTS idx_demo_status   ON demo_files(status);
CREATE INDEX IF NOT EXISTS idx_demo_map      ON demo_files(map_name);
CREATE INDEX IF NOT EXISTS idx_demo_added    ON demo_files(added_at DESC);
CREATE INDEX IF NOT EXISTS idx_demo_csz      ON demo_files(csz_path);

CREATE TABLE IF NOT EXISTS zip_extract_state (
    zip_path   TEXT PRIMARY KEY,
    mtime_ns   INTEGER,
    size_bytes INTEGER,
    updated_at TEXT
);

-- Steam avatars, one row per SteamID64 seen in the library.  Kept in its own
-- table rather than on demo_files because a player appears in many demos and
-- the picture belongs to the person, not to the match.
CREATE TABLE IF NOT EXISTS player_avatars (
    steamid      TEXT PRIMARY KEY,
    persona_name TEXT,
    avatar_url   TEXT,
    file_path    TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending | ok | error
    error_msg    TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    fetched_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_avatar_status ON player_avatars(status);
"""


# ── connection helpers ───────────────────────────────────────────────────────


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path or config.LIBRARY_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── one-off: the library moved from <repo>/data to <repo>/app/data ───────────
#
# ``<repo>/data/`` is for training data, so the web app's own files moved under
# ``app/``.  Nothing in the database records a relative path -- ``csz_path``,
# ``thumb_path`` and the avatar files are absolute -- so an existing install
# would come up with every row pointing at a directory that no longer holds
# anything.  The demos would still be on disk; the library just would not know.

#: Every column that holds an absolute path into the data directory.
_PATH_COLUMNS = (
    ("demo_files", "path"),
    ("demo_files", "csz_path"),
    ("demo_files", "thumb_path"),
    ("demo_files", "watch_root"),
    ("player_avatars", "file_path"),
    ("zip_extract_state", "zip_path"),
)


def migrate_legacy_data_dir() -> Optional[Path]:
    """Move a pre-``app/data`` library into place, once.

    Returns the new directory when a move happened, else ``None``.

    Three things keep this from ever touching anything that is not ours:

    * both paths must be the *default* ones.  A test points ``DATA_DIR`` at a
      scratch directory, and without this check the migration would cheerfully
      move the developer's real library into it.
    * the legacy directory must contain a ``library.db`` with a ``demo_files``
      table.  Training data does not, so a ``data/`` full of parquet files is
      left alone.
    * it only runs when the target does not exist yet, so it is idempotent and
      can never overwrite a live library.
    """
    legacy = config.LEGACY_DATA_DIR
    target = config.DATA_DIR
    if target != config.DEFAULT_DATA_DIR or legacy == target:
        return None
    if target.exists() or not legacy.is_dir():
        return None

    src_db = legacy / "library.db"
    if not src_db.is_file():
        return None
    try:
        probe = sqlite3.connect(str(src_db), timeout=5.0)
        try:
            tables = {r[0] for r in probe.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
        finally:
            probe.close()
    except sqlite3.Error:
        return None
    if "demo_files" not in tables:
        return None

    # Move first, rewrite second: the database travels with the directory, so a
    # crash in between leaves a library whose paths are stale but whose files
    # are all in the new place -- re-running fixes it.
    shutil.move(str(legacy), str(target))

    old, new = str(legacy), str(target)
    prefix = old + os.sep
    with _write_lock, connect() as conn:
        for table, column in _PATH_COLUMNS:
            if table not in tables:
                continue
            rows = conn.execute(
                f"SELECT rowid, {column} FROM {table} "
                f"WHERE {column} IS NOT NULL AND {column} <> ''"
            ).fetchall()
            updates = [
                (new + str(value)[len(old):], rowid)
                for rowid, value in rows
                if str(value).startswith(prefix)
            ]
            if updates:
                conn.executemany(
                    f"UPDATE {table} SET {column} = ? WHERE rowid = ?", updates
                )
        conn.commit()
    return target


def init_db(db_path: Optional[Path] = None) -> None:
    with _write_lock, connect(db_path) as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


#: Columns the API is allowed to write, so a stray key cannot inject SQL.
_WRITABLE = {
    "path", "filename", "file_size", "content_md5", "csz_path", "status",
    "source", "watch_root", "map_name", "total_rounds", "duration_mins",
    "match_date", "server_name", "team_a_name", "team_b_name",
    "team_a_score", "team_b_score", "players_json", "scoreboard_json",
    "watched_json", "display_name", "remark", "added_at", "updated_at",
    "parsed_at", "error_msg", "thumb_path", "thumb_key",
}

#: JSON-encoded columns, decoded on read.
_JSON_COLUMNS = ("players_json", "scoreboard_json", "watched_json")


# ── reads ────────────────────────────────────────────────────────────────────


def _row_to_dict(row: sqlite3.Row, decode: bool = True) -> dict[str, Any]:
    out = dict(row)
    if decode:
        for col in _JSON_COLUMNS:
            raw = out.get(col)
            if raw:
                try:
                    out[col.removesuffix("_json")] = json.loads(raw)
                except ValueError:
                    out[col.removesuffix("_json")] = None
            else:
                out[col.removesuffix("_json")] = None
    out["status_label"] = STATUS_LABELS.get(out.get("status") or "", out.get("status"))
    return out


def get_demo(demo_id: int, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM demo_files WHERE id = ?", (demo_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_by_path(path: str, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM demo_files WHERE path = ?", (path,)).fetchone()
    return _row_to_dict(row) if row else None


def get_by_md5(md5: str, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    if not md5:
        return None
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM demo_files WHERE content_md5 = ? LIMIT 1", (md5,)
        ).fetchone()
    return _row_to_dict(row) if row else None


# ── writes ───────────────────────────────────────────────────────────────────


def upsert_demo(
    path: str,
    *,
    db_path: Optional[Path] = None,
    **fields: Any,
) -> int:
    """Insert a row for ``path`` or update the existing one. Returns the id.

    Only keys in :data:`_WRITABLE` are honoured.  ``added_at`` is set once, on
    first insert, so re-discovering a file does not reshuffle the library.
    """
    clean = {k: v for k, v in fields.items() if k in _WRITABLE and k != "path"}
    with _write_lock, connect(db_path) as conn:
        row = conn.execute("SELECT id FROM demo_files WHERE path = ?", (path,)).fetchone()
        if row:
            demo_id = int(row["id"])
            if clean:
                clean["updated_at"] = _now()
                sets = ", ".join(f"{k} = ?" for k in clean)
                conn.execute(
                    f"UPDATE demo_files SET {sets} WHERE id = ?",
                    (*clean.values(), demo_id),
                )
        else:
            clean.setdefault("added_at", _now())
            cols = ["path", *clean.keys()]
            placeholders = ", ".join("?" for _ in cols)
            cur = conn.execute(
                f"INSERT INTO demo_files ({', '.join(cols)}) VALUES ({placeholders})",
                (path, *clean.values()),
            )
            demo_id = int(cur.lastrowid)
        conn.commit()
    return demo_id


def update_demo(demo_id: int, *, db_path: Optional[Path] = None, **fields: Any) -> bool:
    clean = {k: v for k, v in fields.items() if k in _WRITABLE and k != "path"}
    if not clean:
        return False
    clean["updated_at"] = _now()
    sets = ", ".join(f"{k} = ?" for k in clean)
    with _write_lock, connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE demo_files SET {sets} WHERE id = ?", (*clean.values(), demo_id)
        )
        conn.commit()
        return cur.rowcount > 0


def delete_demo(demo_id: int, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Remove a library row. Returns the row that was deleted, or None."""
    with _write_lock, connect(db_path) as conn:
        row = conn.execute("SELECT * FROM demo_files WHERE id = ?", (demo_id,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM demo_files WHERE id = ?", (demo_id,))
        conn.commit()
    return _row_to_dict(row)


def all_paths(db_path: Optional[Path] = None) -> set[str]:
    with connect(db_path) as conn:
        return {r["path"] for r in conn.execute("SELECT path FROM demo_files")}


def scoreboard_rows(db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Rows that have a scoreboard, with the raw ``watched_json`` alongside.

    Only extracted rows have a scoreboard, so only those have anything to
    re-derive.  ``watched_json`` comes back as the stored string rather than a
    decoded value so a caller can compare before writing and skip rows that
    would not change.
    """
    with connect(db_path) as conn:
        return [
            {"id": r["id"],
             "scoreboard_json": r["scoreboard_json"],
             "watched_json": r["watched_json"]}
            for r in conn.execute(
                "SELECT id, scoreboard_json, watched_json FROM demo_files "
                "WHERE scoreboard_json IS NOT NULL "
                "AND scoreboard_json NOT IN ('', '[]')"
            )
        ]


def set_watched(updates: Iterable[tuple[str, int]],
                db_path: Optional[Path] = None) -> int:
    """Overwrite the derived ``watched_json`` column for several rows at once.

    ``updates`` is ``(watched_json, demo_id)`` pairs.  Deliberately writes only
    that column: re-deriving it is not a change to the demo, and routing this
    through :func:`update_demo` would stamp ``updated_at`` on every row in the
    library each time the user edits the watched list.
    """
    rows = list(updates)
    if not rows:
        return 0
    with _write_lock, connect(db_path) as conn:
        conn.executemany(
            "UPDATE demo_files SET watched_json = ? WHERE id = ?", rows
        )
        conn.commit()
    return len(rows)


def reset_stale_extracting(db_path: Optional[Path] = None) -> list[int]:
    """Move rows left in ``extracting`` back to ``discovered``.

    Extraction only ever happens in this process, so at startup nothing can be
    in flight -- any row still reading "extracting" was interrupted by a crash,
    a kill, or a power cut.  Left alone it stays that way forever: the UI shows
    a permanent "提取中", and nothing will ever finish it because the watcher
    only queues *newly discovered* files.

    Returns the ids that were reset, so the caller can log or re-queue them.
    """
    with _write_lock, connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM demo_files WHERE status = ?", (STATUS_EXTRACTING,)
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        if ids:
            conn.execute(
                f"UPDATE demo_files SET status = ?, error_msg = NULL, updated_at = ? "
                f"WHERE id IN ({', '.join('?' for _ in ids)})",
                (STATUS_DISCOVERED, _now(), *ids),
            )
            conn.commit()
    return ids


def unfinished_ids(db_path: Optional[Path] = None) -> list[int]:
    """Rows that are supposed to end up with a ``.csz`` but do not have one yet.

    Used to pick up work orphaned by a restart, so a demo does not sit at
    "待入库" forever just because it was discovered during a session that died
    before auto-extract got to it.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, csz_path FROM demo_files "
            "WHERE status IN (?, ?) ORDER BY id",
            (STATUS_DISCOVERED, STATUS_EXTRACTING),
        ).fetchall()
    return [
        int(r["id"]) for r in rows
        if not (r["csz_path"] and Path(r["csz_path"]).is_file())
    ]


def all_md5(db_path: Optional[Path] = None) -> set[str]:
    with connect(db_path) as conn:
        return {
            r["content_md5"]
            for r in conn.execute(
                "SELECT content_md5 FROM demo_files WHERE content_md5 IS NOT NULL"
            )
        }


# ── queries ──────────────────────────────────────────────────────────────────

#: Sort key -> ORDER BY clause.  Everything falls back to newest-first.
SORTS = {
    "library": """
        CASE status
            WHEN 'extracting' THEN 0
            WHEN 'discovered' THEN 1
            WHEN 'ready'      THEN 2
            WHEN 'error'      THEN 3
            ELSE 4
        END, added_at DESC, id DESC
    """,
    "date": "COALESCE(match_date, added_at) DESC, id DESC",
    "added": "added_at DESC, id DESC",
    "size": "file_size DESC, id DESC",
    "duration": "duration_mins DESC, id DESC",
    "rounds": "total_rounds DESC, id DESC",
    "map": "map_name COLLATE NOCASE ASC, added_at DESC",
    "name": "COALESCE(NULLIF(display_name, ''), filename) COLLATE NOCASE ASC",
    "source": "source COLLATE NOCASE ASC, added_at DESC",
}


def list_demos(
    *,
    q: Optional[str] = None,
    statuses: Optional[Iterable[str]] = None,
    sources: Optional[Iterable[str]] = None,
    map_name: Optional[str] = None,
    player: Optional[str] = None,
    steamid: Optional[str] = None,
    watched_only: bool = False,
    sort: str = "library",
    limit: int = 60,
    offset: int = 0,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Filtered, sorted, paginated library query. Returns ``{items, total, ...}``.

    ``player`` is a fuzzy name match (what the search box does); ``steamid`` is
    an exact roster match, which is what a player profile needs so that two
    people who share a nickname cannot bleed into each other's match list.
    """
    where: list[str] = []
    params: list[Any] = []

    if q:
        # One search box, so it has to cover what people actually type: the
        # file they remember, the name they gave it, a note, the map -- and,
        # most of all, a player.  ``players_json`` holds the roster, so a
        # substring match finds "ZywOo" without a join.
        #
        # The roster JSON carries the in-game name only, so the Steam persona
        # needs its own EXISTS through ``player_avatars``: the player pages are
        # titled by the persona, and a name the user can read on one page has to
        # be findable on the other.
        where.append(
            "(filename LIKE ? OR IFNULL(display_name, '') LIKE ? "
            "OR IFNULL(remark, '') LIKE ? OR IFNULL(map_name, '') LIKE ? "
            "OR IFNULL(players_json, '') LIKE ? "
            "OR IFNULL(team_a_name, '') LIKE ? OR IFNULL(team_b_name, '') LIKE ? "
            "OR IFNULL(server_name, '') LIKE ? "
            "OR EXISTS (SELECT 1 FROM json_each("
            "  CASE WHEN json_valid(players_json) THEN players_json ELSE '[]' END"
            ") p JOIN player_avatars pa "
            "  ON pa.steamid = json_extract(p.value, '$.steamid') "
            "  WHERE IFNULL(pa.persona_name, '') LIKE ?))"
        )
        like = f"%{q}%"
        params += [like] * 9

    status_list = [s for s in (statuses or []) if s in ALL_STATUSES]
    if status_list:
        where.append(f"status IN ({', '.join('?' for _ in status_list)})")
        params += status_list

    source_list = [s for s in (sources or []) if s]
    if source_list:
        where.append(f"source IN ({', '.join('?' for _ in source_list)})")
        params += source_list

    if map_name:
        where.append("map_name = ?")
        params.append(map_name)

    if player:
        # roster is stored as JSON; a substring match is good enough and keeps
        # the query index-free but cheap at library scale.
        where.append("IFNULL(players_json, '') LIKE ?")
        params.append(f"%{player}%")

    if steamid:
        # json_each over a malformed value raises, so the CASE keeps a
        # hand-edited row from taking the whole page down with it.
        where.append(
            "EXISTS (SELECT 1 FROM json_each("
            "  CASE WHEN json_valid(players_json) THEN players_json ELSE '[]' END"
            ") p WHERE json_extract(p.value, '$.steamid') = ?)"
        )
        params.append(str(steamid))

    if watched_only:
        where.append("IFNULL(watched_json, '') NOT IN ('', '[]')")

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    order = SORTS.get(sort, SORTS["library"])
    limit = max(1, min(int(limit or 60), 1000))
    offset = max(0, int(offset or 0))

    with connect(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM demo_files {clause}", params
        ).fetchone()["n"]
        rows = conn.execute(
            f"SELECT * FROM demo_files {clause} ORDER BY {order} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()

    return {
        "items": [_row_to_dict(r) for r in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "sort": sort if sort in SORTS else "library",
    }


def distinct_maps(db_path: Optional[Path] = None) -> list[str]:
    with connect(db_path) as conn:
        return [
            r["map_name"]
            for r in conn.execute(
                "SELECT DISTINCT map_name FROM demo_files "
                "WHERE map_name IS NOT NULL AND map_name != '' ORDER BY map_name"
            )
        ]


def stats(db_path: Optional[Path] = None) -> dict[str, Any]:
    with connect(db_path) as conn:
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM demo_files GROUP BY status"
            )
        }
        by_source = {
            r["source"] or "本地": r["n"]
            for r in conn.execute(
                "SELECT source, COUNT(*) AS n FROM demo_files GROUP BY source"
            )
        }
        total = conn.execute("SELECT COUNT(*) AS n FROM demo_files").fetchone()["n"]
        size = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) AS s FROM demo_files"
        ).fetchone()["s"]
    return {
        "total": int(total),
        "bytes": int(size),
        "by_status": by_status,
        "by_source": by_source,
        "maps": distinct_maps(db_path),
    }


# ── players ──────────────────────────────────────────────────────────────────
#
# The player pages need "every player across every demo, with totals".  There is
# deliberately no second table for that: `scoreboard_json` already holds one
# entry per player per match, and SQLite's json_each turns it into rows on the
# fly.  A separate table would be a cache that could disagree with the demos it
# was derived from, and would need rebuilding whenever a demo was re-parsed.
#
# `demo_files` is still the outer table so the aggregation stays inside the
# library; a demo deleted from the library stops counting immediately.

#: One row per (demo, player).  ``json_valid`` guards a hand-edited row.
_PLAYER_ROWS = """
WITH sb AS (
    SELECT
        json_extract(s.value, '$.steamid')    AS steamid,
        json_extract(s.value, '$.name')       AS name,
        json_extract(s.value, '$.team')       AS team,
        json_extract(s.value, '$.kills')      AS kills,
        json_extract(s.value, '$.deaths')     AS deaths,
        json_extract(s.value, '$.assists')    AS assists,
        json_extract(s.value, '$.hs')         AS hs,
        json_extract(s.value, '$.adr_damage') AS dmg,
        d.id                                  AS demo_id,
        d.added_at                            AS added_at,
        d.match_date                          AS match_date,
        d.map_name                            AS map_name,
        d.total_rounds                        AS rounds,
        d.team_a_score                        AS score_a,
        d.team_b_score                        AS score_b
    FROM demo_files d, json_each(
        CASE WHEN json_valid(d.scoreboard_json) THEN d.scoreboard_json ELSE '[]' END
    ) s
    WHERE json_extract(s.value, '$.steamid') IS NOT NULL
      AND json_extract(s.value, '$.steamid') <> ''
)
"""

#: Totals per player.  ``team`` is 0 for the left-hand team and 1 for the other,
#: which is all the win/loss split needs -- both scores are on the same row.
#:
#: Two of the columns come from outside the scoreboard:
#:
#: * ``persona`` is the Steam profile name, which no demo carries.  It is joined
#:   in from ``player_avatars`` so the player list can be titled by it and
#:   searched by it.  ``MAX`` is safe because the row is one per SteamID.
#: * ``names_json`` is every in-game name this SteamID has used, newest first.
#:   A correlated subquery rather than an aggregate over the group: the *order*
#:   is the point, and ``json_group_array(DISTINCT ...)`` would hand back
#:   whatever order SQLite's DISTINCT b-tree produced.
_PLAYER_TOTALS = _PLAYER_ROWS + """
SELECT
    a.steamid AS steamid,
    (SELECT b.name FROM sb b WHERE b.steamid = a.steamid
      ORDER BY b.added_at DESC, b.demo_id DESC LIMIT 1) AS name,
    MAX(pa.persona_name) AS persona,
    (SELECT json_group_array(n) FROM (
        SELECT b.name AS n FROM sb b
        WHERE b.steamid = a.steamid AND IFNULL(b.name, '') <> ''
        GROUP BY b.name
        ORDER BY MAX(b.added_at) DESC, MAX(b.demo_id) DESC
     )) AS names_json,
    (SELECT b.demo_id FROM sb b WHERE b.steamid = a.steamid
      ORDER BY b.added_at DESC, b.demo_id DESC LIMIT 1) AS last_demo_id,
    (SELECT b.map_name FROM sb b WHERE b.steamid = a.steamid
      ORDER BY b.added_at DESC, b.demo_id DESC LIMIT 1) AS last_map,
    MAX(COALESCE(a.match_date, a.added_at)) AS last_seen,
    COUNT(*)                                AS matches,
    COALESCE(SUM(a.kills), 0)               AS kills,
    COALESCE(SUM(a.deaths), 0)              AS deaths,
    COALESCE(SUM(a.assists), 0)             AS assists,
    COALESCE(SUM(a.hs), 0)                  AS headshots,
    COALESCE(SUM(a.dmg), 0)                 AS damage,
    COALESCE(SUM(a.rounds), 0)              AS rounds,
    COUNT(DISTINCT a.map_name)              AS map_count,
    SUM(CASE WHEN (a.team = 0 AND a.score_a > a.score_b)
              OR (a.team = 1 AND a.score_b > a.score_a) THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN (a.team = 0 AND a.score_a < a.score_b)
              OR (a.team = 1 AND a.score_b < a.score_a) THEN 1 ELSE 0 END) AS losses
FROM sb a
LEFT JOIN player_avatars pa ON pa.steamid = a.steamid
GROUP BY a.steamid
"""

#: Derived numbers.  ADR and KPR divide by rounds rather than by matches, so a
#: 30-round match does not count the same as a 13-0 stomp.
_PLAYER_VIEW = f"""
SELECT *,
    CASE WHEN deaths  > 0 THEN ROUND(kills * 1.0 / deaths, 2) ELSE kills END AS kd,
    CASE WHEN rounds  > 0 THEN ROUND(kills * 1.0 / rounds, 2) ELSE 0 END      AS kpr,
    CASE WHEN rounds  > 0 THEN ROUND(damage * 1.0 / rounds, 1) ELSE 0 END     AS adr,
    CASE WHEN kills   > 0 THEN ROUND(headshots * 100.0 / kills, 1) ELSE 0 END AS hs_pct,
    CASE WHEN matches > 0 THEN ROUND(wins * 100.0 / matches, 1) ELSE 0 END    AS win_pct
FROM ({_PLAYER_TOTALS})
"""

#: Player sort key -> ORDER BY.  ``kd`` and ``adr`` are computed above.
PLAYER_SORTS = {
    "matches": "matches DESC, kills DESC, steamid ASC",
    "kills": "kills DESC, matches DESC, steamid ASC",
    "kd": "kd DESC, matches DESC, steamid ASC",
    "adr": "adr DESC, matches DESC, steamid ASC",
    "last": "last_seen DESC, kills DESC, steamid ASC",
    "name": "name COLLATE NOCASE ASC, steamid ASC",
}


def _json_names(raw: Optional[str]) -> list[str]:
    """Decode the ``names_json`` column into a list of usable names."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(value, list):
        return []
    return [str(n) for n in value if isinstance(n, str) and n.strip()]


def list_players(
    *,
    q: Optional[str] = None,
    steamid: Optional[str] = None,
    watched_names: Optional[Iterable[str]] = None,
    sort: str = "matches",
    limit: int = 60,
    offset: int = 0,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Every player in the library with their totals. Returns ``{items, total}``.

    Names come from the most recent demo the player appears in, because in-game
    names change and the newest one is the one a user will recognise.  The full
    history is on each item as ``names``.

    ``q`` is a plain substring test over all four things a user might type: the
    newest in-game name, any earlier in-game name, the Steam persona, and the
    SteamID.  ``search_players`` is the ranked version of the same idea.

    ``watched_names`` is the config's watched list, passed in rather than read
    here so the query layer stays free of settings.  Matching is the same
    case-insensitive substring test the library highlight uses.
    """
    where: list[str] = []
    params: list[Any] = []
    if steamid:
        where.append("steamid = ?")
        params.append(str(steamid))
    if q:
        # ``names_json`` covers the names this player no longer uses: searching
        # "Wicadia" has to keep finding someone who has since renamed, or the
        # history this page displays would be unreachable.
        where.append(
            "(IFNULL(name, '') LIKE ? OR IFNULL(persona, '') LIKE ? "
            "OR IFNULL(names_json, '') LIKE ? OR steamid LIKE ?)"
        )
        like = f"%{q}%"
        params += [like, like, like, like]
    names = [str(w).strip() for w in (watched_names or []) if str(w or "").strip()]
    if names:
        parts = []
        for name in names[:50]:
            parts.append("LOWER(IFNULL(name, '')) LIKE ?")
            params.append(f"%{name.lower()}%")
        where.append(f"({' OR '.join(parts)})")
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    order = PLAYER_SORTS.get(sort, PLAYER_SORTS["matches"])
    limit = max(1, min(int(limit or 60), 1000))
    offset = max(0, int(offset or 0))

    with connect(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM ({_PLAYER_VIEW}) {clause}", params
        ).fetchone()["n"]
        rows = conn.execute(
            f"SELECT * FROM ({_PLAYER_VIEW}) {clause} ORDER BY {order} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()

    return {
        "items": [_player_row(r) for r in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "sort": sort if sort in PLAYER_SORTS else "matches",
    }


def _player_row(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["names"] = _json_names(out.pop("names_json", None))
    return out


def get_player(steamid: str, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    result = list_players(steamid=steamid, limit=1, db_path=db_path)
    return result["items"][0] if result["items"] else None


def player_names(steamid: str, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Every in-game name one player has used, newest first.

    This is the list behind the profile's name history.  ``matches`` counts the
    demos the name appears in, which is what tells a one-off typo apart from a
    handle somebody actually played under.
    """
    sql = _PLAYER_ROWS + """
    SELECT name,
           COUNT(*)                            AS matches,
           MAX(COALESCE(match_date, added_at)) AS last_seen,
           MAX(added_at)                       AS last_added
    FROM sb
    WHERE steamid = ? AND IFNULL(name, '') <> ''
    GROUP BY name
    ORDER BY last_added DESC, MAX(demo_id) DESC, name ASC
    """
    with connect(db_path) as conn:
        rows = conn.execute(sql, (str(steamid),)).fetchall()
    return [dict(r) for r in rows]


# ── ranked player search ─────────────────────────────────────────────────────
#
# The list filter in :func:`list_players` is a substring test, which is the
# right thing for a grid that also has to report a total.  This is the other
# half: a handful of *ranked* candidates for the suggestion box, so a
# half-remembered handle still finds its player.  Scoring happens in Python
# because a subsequence match ("wcda" -> "Wicadia") is not expressible in LIKE,
# and the whole player view is small enough (one row per SteamID in the library)
# to score in full rather than guess a SQL prefilter that could drop the answer.

#: What a hit in each field is worth.  The player list is titled by the Steam
#: persona, so it edges out an in-game name; a SteamID is only ever a
#: tie-breaker between people whose names are equally bad matches.
_FIELD_WEIGHT = {"persona": 1.0, "name": 0.97, "steamid": 0.90}

#: Shorter than this and a numeric query matches every SteamID64 there is --
#: they all start "7656119", so "7656" would rank the whole library equally.
_MIN_STEAMID_QUERY = 5


def _fuzzy_score(needle: str, hay: str) -> float:
    """How well ``needle`` matches ``hay``, from 0.0 to 1.0. Case-insensitive.

    Four tiers, best first: the whole string, a prefix, a substring, and a
    subsequence.  The tiers are ordered by score, not just by check order, so a
    scattered subsequence can never outrank a real substring.  A substring is
    penalised by how far in it starts: "ke" belongs to a name that starts with
    it before it belongs to ``amınaKE``.
    """
    if not needle or not hay:
        return 0.0
    n, h = needle.lower(), hay.lower()
    if h == n:
        return 1.0
    pos = h.find(n)
    if pos == 0:
        return 0.95
    if pos > 0:
        return max(0.55, 0.82 - pos * 0.01)

    idx = 0
    first = last = -1
    for i, ch in enumerate(h):
        if idx < len(n) and ch == n[idx]:
            if first < 0:
                first = i
            last = i
            idx += 1
    if idx < len(n):
        return 0.0
    # Tightly packed matches are worth more than characters hunted out of a
    # long string; the ceiling stays below the substring floor.
    return 0.25 + 0.25 * (len(n) / (last - first + 1))


def _match_player(query: str, player: dict[str, Any]) -> tuple[float, str, str]:
    """Best ``(score, field, text)`` for one player. Score 0 means no match."""
    best: tuple[float, str, str] = (0.0, "", "")

    def consider(field: str, text: Any) -> None:
        nonlocal best
        value = str(text or "")
        if not value:
            return
        score = _fuzzy_score(query, value) * _FIELD_WEIGHT[field]
        if score > best[0]:
            best = (score, field, value)

    consider("persona", player.get("persona"))
    consider("name", player.get("name"))
    for alt in player.get("names") or []:
        consider("name", alt)
    steamid = str(player.get("steamid") or "")
    if steamid and query.isdigit() and len(query) >= _MIN_STEAMID_QUERY:
        consider("steamid", steamid)
    return best


def search_players(
    q: str,
    *,
    limit: int = 8,
    db_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Ranked candidates for a player search, best match first.

    Ties are broken by match count, so of two players a query matches equally
    well the one with a real history comes first.
    """
    query = str(q or "").strip()
    if not query:
        return []
    with connect(db_path) as conn:
        rows = conn.execute(f"SELECT * FROM ({_PLAYER_VIEW})").fetchall()

    scored: list[dict[str, Any]] = []
    for row in rows:
        player = _player_row(row)
        score, field, text = _match_player(query, player)
        if score <= 0:
            continue
        player["score"] = round(score, 4)
        player["match"] = field
        player["match_text"] = text
        scored.append(player)

    scored.sort(key=lambda p: (-p["score"], -int(p["matches"] or 0), p["steamid"]))
    return scored[: max(1, min(int(limit or 8), 50))]


def player_maps(steamid: str, db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Per-map breakdown for one player, busiest map first."""
    sql = _PLAYER_ROWS + """
    SELECT map_name,
           COUNT(*)                  AS matches,
           COALESCE(SUM(kills), 0)   AS kills,
           COALESCE(SUM(deaths), 0)  AS deaths,
           COALESCE(SUM(assists), 0) AS assists,
           COALESCE(SUM(hs), 0)      AS headshots,
           COALESCE(SUM(dmg), 0)     AS damage,
           COALESCE(SUM(rounds), 0)  AS rounds,
           SUM(CASE WHEN (team = 0 AND score_a > score_b)
                     OR (team = 1 AND score_b > score_a) THEN 1 ELSE 0 END) AS wins
    FROM sb
    WHERE steamid = ?
    GROUP BY map_name
    ORDER BY matches DESC, map_name ASC
    """
    with connect(db_path) as conn:
        rows = conn.execute(sql, (str(steamid),)).fetchall()
    out = []
    for r in rows:
        row = dict(r)
        rounds = row.get("rounds") or 0
        deaths = row.get("deaths") or 0
        kills = row.get("kills") or 0
        row["kd"] = round(kills / deaths, 2) if deaths else float(kills)
        row["adr"] = round((row.get("damage") or 0) / rounds, 1) if rounds else 0.0
        row["win_pct"] = round(row["wins"] * 100.0 / row["matches"], 1) if row["matches"] else 0.0
        out.append(row)
    return out


def distinct_steamids(db_path: Optional[Path] = None) -> list[str]:
    """Every SteamID64 the library knows about, for the avatar sync."""
    with connect(db_path) as conn:
        rows = conn.execute(
            _PLAYER_ROWS + "SELECT DISTINCT steamid FROM sb ORDER BY steamid"
        ).fetchall()
    return [r["steamid"] for r in rows]


# ── avatars ──────────────────────────────────────────────────────────────────

AVATAR_PENDING = "pending"
AVATAR_OK = "ok"
AVATAR_ERROR = "error"

_AVATAR_WRITABLE = {
    "persona_name", "avatar_url", "file_path", "status",
    "error_msg", "attempts", "fetched_at",
}


def get_avatar(steamid: str, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM player_avatars WHERE steamid = ?", (str(steamid),)
        ).fetchone()
    return dict(row) if row else None


def avatars_for(steamids: Iterable[str],
                db_path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """Avatar rows for a set of SteamIDs, keyed by SteamID.

    Chunked because SQLite caps how many placeholders one statement may carry,
    and a library page can easily name a few hundred players.
    """
    ids = list(dict.fromkeys(str(s) for s in steamids if s))
    out: dict[str, dict[str, Any]] = {}
    if not ids:
        return out
    with connect(db_path) as conn:
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            rows = conn.execute(
                f"SELECT * FROM player_avatars WHERE steamid IN "
                f"({', '.join('?' for _ in chunk)})",
                chunk,
            ).fetchall()
            for row in rows:
                out[row["steamid"]] = dict(row)
    return out


def upsert_avatar(steamid: str, *, db_path: Optional[Path] = None,
                  bump_attempts: bool = False, **fields: Any) -> None:
    """Insert or update one avatar row. ``fetched_at`` defaults to now."""
    sid = str(steamid or "").strip()
    if not sid:
        return
    clean = {k: v for k, v in fields.items() if k in _AVATAR_WRITABLE}
    with _write_lock, connect(db_path) as conn:
        row = conn.execute(
            "SELECT attempts FROM player_avatars WHERE steamid = ?", (sid,)
        ).fetchone()
        if bump_attempts:
            # sqlite3.Row has no .get(), hence the explicit test.
            clean["attempts"] = (int(row["attempts"]) if row else 0) + 1
        clean.setdefault("fetched_at", _now())
        if row:
            sets = ", ".join(f"{k} = ?" for k in clean)
            conn.execute(
                f"UPDATE player_avatars SET {sets} WHERE steamid = ?",
                (*clean.values(), sid),
            )
        else:
            cols = ["steamid", *clean.keys()]
            conn.execute(
                f"INSERT INTO player_avatars ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)})",
                (sid, *clean.values()),
            )
        conn.commit()


def _age_hours(stamp: Optional[str]) -> Optional[float]:
    """Hours since a ``_now()`` timestamp, or None when it cannot be read."""
    if not stamp:
        return None
    try:
        return (time.time() - time.mktime(time.strptime(str(stamp), "%Y-%m-%d %H:%M:%S"))) / 3600.0
    except (TypeError, ValueError, OverflowError):
        return None


def stale_steamids(
    steamids: Iterable[str],
    *,
    ttl_days: float = 7.0,
    error_retry_hours: float = 6.0,
    force: bool = False,
    db_path: Optional[Path] = None,
) -> list[str]:
    """Which of these SteamIDs actually need a network lookup.

    Fresh means: an ``ok`` row whose file is still on disk and younger than
    ``ttl_days``, or an ``error`` row younger than ``error_retry_hours``.  A row
    whose cached file has been deleted is always stale, otherwise a stray
    ``rm app/data/avatars`` would leave the UI showing broken images until the TTL
    happened to expire.
    """
    ids = list(dict.fromkeys(str(s) for s in steamids if s))
    rows = avatars_for(ids, db_path)
    out: list[str] = []
    for sid in ids:
        if force:
            out.append(sid)
            continue
        row = rows.get(sid)
        if row is None:
            out.append(sid)
            continue
        age = _age_hours(row.get("fetched_at"))
        if row.get("status") == AVATAR_OK:
            if age is not None and age < ttl_days * 24 and row.get("file_path") \
                    and Path(row["file_path"]).is_file():
                continue
            out.append(sid)
        elif row.get("status") == AVATAR_ERROR:
            if age is not None and age < error_retry_hours:
                continue
            out.append(sid)
        else:
            out.append(sid)
    return out


def avatar_stats(db_path: Optional[Path] = None) -> dict[str, int]:
    """Counts for the settings page: how many faces are cached, and how many failed."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM player_avatars GROUP BY status"
        ).fetchall()
    by = {r["status"]: int(r["n"]) for r in rows}
    return {
        "known": sum(by.values()),
        "ok": by.get(AVATAR_OK, 0),
        "error": by.get(AVATAR_ERROR, 0),
    }


# ── cache bookkeeping ────────────────────────────────────────────────────────
#
# The settings page can wipe a whole cache directory at once.  These are the
# other half of that: after the files are gone the rows must stop pointing at
# them, or the library keeps claiming a demo is extracted while its replay data
# has been deleted.  They are separate functions rather than one generic
# "null this column" so each can say what it does to ``status``.


def forget_csz(db_path: Optional[Path] = None) -> int:
    """Drop every ``csz_path`` after the ``.csz`` files were deleted.

    A ``ready`` row goes back to ``discovered``: it is not ready any more.
    Those rows are then picked up by the next scan (``unfinished_ids``), so a
    bulk clear is recoverable rather than a one-way loss.
    """
    with _write_lock, connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE demo_files SET csz_path = NULL, updated_at = ?, "
            "status = CASE WHEN status = ? THEN ? ELSE status END "
            "WHERE csz_path IS NOT NULL AND csz_path <> ''",
            (_now(), STATUS_READY, STATUS_DISCOVERED),
        )
        conn.commit()
        return cur.rowcount


def forget_thumbs(db_path: Optional[Path] = None) -> int:
    """Drop every ``thumb_path``/``thumb_key`` after the PNGs were deleted.

    ``thumb_key`` goes too: it records which ``.csz`` the picture was built
    from, and a stale key would look like a cache hit for a thumbnail that is
    not there.  The next extraction or refresh rebuilds both.
    """
    with _write_lock, connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE demo_files SET thumb_path = NULL, thumb_key = NULL "
            "WHERE thumb_path IS NOT NULL OR thumb_key IS NOT NULL"
        )
        conn.commit()
        return cur.rowcount


def forget_avatar_files(db_path: Optional[Path] = None) -> int:
    """Mark every cached avatar as pending after its image was deleted.

    ``persona_name`` survives on purpose.  It is text the app already spent a
    round trip on, the player list is titled by it, and dropping it would blank
    every Steam name on a page that has nothing to do with images.  ``pending``
    rather than ``ok`` because there is no file any more, which is also what
    makes :func:`stale_steamids` re-fetch it.

    Only rows that claim a file are touched, so a lookup that failed for its own
    reason keeps the message that explains it.
    """
    with _write_lock, connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE player_avatars SET file_path = NULL, status = ?, error_msg = NULL "
            "WHERE file_path IS NOT NULL AND file_path <> ''",
            (AVATAR_PENDING,),
        )
        conn.commit()
        return cur.rowcount


# ── zip bookkeeping ──────────────────────────────────────────────────────────


def zip_seen(zip_path: str, mtime_ns: int, size: int,
             db_path: Optional[Path] = None) -> bool:
    """True when this exact zip has already been unpacked."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT mtime_ns, size_bytes FROM zip_extract_state WHERE zip_path = ?",
            (zip_path,),
        ).fetchone()
    return bool(row and row["mtime_ns"] == mtime_ns and row["size_bytes"] == size)


def mark_zip(zip_path: str, mtime_ns: int, size: int,
             db_path: Optional[Path] = None) -> None:
    with _write_lock, connect(db_path) as conn:
        conn.execute(
            "INSERT INTO zip_extract_state (zip_path, mtime_ns, size_bytes, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(zip_path) DO UPDATE SET "
            "mtime_ns = excluded.mtime_ns, size_bytes = excluded.size_bytes, "
            "updated_at = excluded.updated_at",
            (zip_path, mtime_ns, size, _now()),
        )
        conn.commit()
