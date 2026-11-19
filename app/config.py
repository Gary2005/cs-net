"""
Application configuration for the cs-net demo library.

Everything lives under ``app/data/`` (gitignored):

    app/data/config.json     user settings (watch roots, watched players, sampling)
    app/data/library.db      SQLite demo library
    app/data/csz/            extracted ``.csz`` files
    app/data/thumbs/         generated radar thumbnails
    app/data/avatars/        Steam avatars downloaded for the player pages
    app/data/recycle/        ``.dem`` files the user chose to remove

It sits under ``app/`` and not at the repo root because ``<repo>/data/`` is
where training data lives; the two have nothing to do with each other.  An
install from before that split is carried over by
:func:`app.db.migrate_legacy_data_dir`.

打包成 exe 之后它们不在仓库里了：数据目录是 exe 旁边的 ``data\\``（绿色版）
或 ``%LOCALAPPDATA%\\cs-net``，资源则在 ``_internal\\`` 里。规则全在
:mod:`app.paths`，这里只是把算好的结果贴成模块常量 —— 下游照旧写 ``config.DATA_DIR``。

The config is a plain JSON file read/written through :class:`Config`.  A
``threading.Lock`` guards read-modify-write so the watcher thread and an HTTP
request cannot clobber each other.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from . import paths

# ── paths ────────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).resolve().parent
#: 资源根：模板、静态资源、内置包。开发态是仓库根，冻结后是 exe 旁边的
#: ``_internal\\``（见 :mod:`app.paths`）。
ROOT = paths.RESOURCES_DIR

DATA_DIR = paths.DATA_DIR
#: What :data:`DATA_DIR` is when nothing has overridden it.  The migration in
#: :func:`app.db.migrate_legacy_data_dir` compares against this, so a test that
#: points ``DATA_DIR`` at a scratch directory can never be mistaken for a real
#: install and have the developer's library moved into it.
DEFAULT_DATA_DIR = DATA_DIR
#: Where the library lived before it moved under ``app/``.  Only read by that
#: same migration, which leaves the directory alone unless it really is a
#: library -- a training-data ``data/`` has no ``demo_files`` table.  冻结态下
#: 这个路径不存在（它是 ``_internal\data``），迁移自然什么都不做。
LEGACY_DATA_DIR = ROOT / "data"

CONFIG_PATH = DATA_DIR / "config.json"
LIBRARY_DB = DATA_DIR / "library.db"
CSZ_DIR = DATA_DIR / "csz"
THUMB_DIR = DATA_DIR / "thumbs"
AVATAR_DIR = DATA_DIR / "avatars"
MAPS_DIR = paths.MAPS_DIR
EXAMPLES_DIR = paths.EXAMPLES_DIR


# ── demo sources ─────────────────────────────────────────────────────────────

#: Display order and labels for the platforms the library knows about.
SOURCE_LABELS = {
    "5E": "5E 对战平台",
    "完美": "完美世界",
    "官匹": "官方匹配",
    "FACEIT": "FACEIT",
    "ESL": "ESL",
    "ESEA": "ESEA",
    "BLAST": "BLAST",
    "PGL": "PGL",
    "StarLadder": "StarLadder",
    "本地": "本地 / 其他",
}

SOURCE_ORDER = [
    "5E", "完美", "官匹", "FACEIT", "ESL", "ESEA",
    "BLAST", "PGL", "StarLadder", "本地",
]

#: Best-effort candidate download directories, offered as one-click suggestions
#: in the UI.  These are *candidates* -- every one is probed with
#: :func:`probe_path` before it is offered, and none is ever watched implicitly.
#: ``%VAR%`` segments are expanded with the process environment.
PLATFORM_PATH_CANDIDATES: dict[str, list[str]] = {
    "官匹": [
        r"%ProgramFiles(x86)%\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays",
        r"%ProgramFiles%\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays",
        r"D:\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays",
        r"E:\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays",
        r"F:\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays",
        r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays",
        r"E:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays",
    ],
    "5E": [
        r"%USERPROFILE%\Documents\5EPlay\demo",
        r"%USERPROFILE%\Documents\5EPlay",
        r"%USERPROFILE%\Documents\5E\demo",
        r"%USERPROFILE%\Downloads\5EPlay",
        r"%LOCALAPPDATA%\5EPlay\demo",
    ],
    "完美": [
        r"%USERPROFILE%\Documents\完美世界\demo",
        r"%USERPROFILE%\Documents\PerfectWorld\demo",
        r"%USERPROFILE%\Documents\完美世界竞技平台\demo",
        r"%USERPROFILE%\Downloads\完美世界",
        r"%LOCALAPPDATA%\PerfectWorld\demo",
    ],
    "FACEIT": [
        r"%USERPROFILE%\Documents\FACEIT\demo",
        r"%USERPROFILE%\Documents\FACEIT",
        r"%USERPROFILE%\Downloads\FACEIT",
        r"%APPDATA%\FACEIT\demo",
    ],
}

#: Substring -> source, matched against the demo's ``server_name`` header.
#: Checked before the filename rules because a server name is far more
#: trustworthy than a filename the user may have renamed.
SOURCE_SERVER_RULES: list[tuple[str, str]] = [
    ("faceit", "FACEIT"),
    ("5eplay", "5E"),
    ("5ewin", "5E"),
    ("wanmei", "完美"),
    ("perfectworld", "完美"),
    ("完美世界", "完美"),
    ("starladder", "StarLadder"),
    ("blast", "BLAST"),
    ("pgl", "PGL"),
    ("esea", "ESEA"),
    ("esl", "ESL"),
    ("valve", "官匹"),
    ("matchmaking", "官匹"),
]

#: Filename fragments -> source.  Checked in order, first match wins.
#: Mirrors what the platform clients actually name their downloads; the user
#: can always override the source by hand in the library UI.
SOURCE_FILENAME_RULES: list[tuple[str, str]] = [
    ("faceit", "FACEIT"),
    ("5eplay", "5E"),
    ("5ewin", "5E"),
    ("wanmei", "完美"),
    ("perfectworld", "完美"),
    ("pvp.", "完美"),
    ("match730", "官匹"),
    ("matchmaking", "官匹"),
    ("valve", "官匹"),
    ("starladder", "StarLadder"),
    ("esl", "ESL"),
    ("esea", "ESEA"),
    ("blast", "BLAST"),
    ("pgl", "PGL"),
]

#: Filename *patterns* for the cases a substring test gets wrong.
#:
#: 5E names its replays ``g<matchid>-...`` and FACEIT uses ``<id>_team...``;
#: neither contains a usable keyword.  Conversely a bare ``"5e"`` substring is
#: too short to match safely anywhere in a name, so it is deliberately absent
#: from the list above and handled by an anchored pattern here instead.
SOURCE_FILENAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^g\d+[-_]", re.I), "5E"),
    (re.compile(r"^\d+_team", re.I), "FACEIT"),
    (re.compile(r"^5e[-_]", re.I), "5E"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}_", re.I), "5E"),
]


def expand(path: str) -> str:
    """Expand ``%VAR%`` (Windows) and ``~`` in a user-supplied path."""
    return os.path.expanduser(os.path.expandvars(path))


def probe_path(path: str) -> dict[str, Any]:
    """Validate a candidate directory and count what is in it.

    Returns ``{valid, path, demo_count, zip_count, error?}``.  Used both by the
    "suggested directories" list and by the UI when a user types a path, so a
    typo is reported as a message instead of silently watching nothing.
    """
    resolved = expand(path)
    p = Path(resolved)
    try:
        if not p.exists():
            return {"valid": False, "path": resolved, "error": "目录不存在"}
        if not p.is_dir():
            return {"valid": False, "path": resolved, "error": "不是一个目录"}
        demos = zips = 0
        for entry in p.iterdir():
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            if suffix == ".dem":
                demos += 1
            elif suffix == ".zip":
                zips += 1
        return {
            "valid": True,
            "path": str(p.resolve()),
            "demo_count": demos,
            "zip_count": zips,
        }
    except OSError as exc:
        return {"valid": False, "path": resolved, "error": f"无法访问：{exc}"}


def suggested_paths() -> list[dict[str, Any]]:
    """Every platform candidate that actually exists on this machine."""
    out: list[dict[str, Any]] = []
    for source, candidates in PLATFORM_PATH_CANDIDATES.items():
        for raw in candidates:
            info = probe_path(raw)
            if info.get("valid"):
                out.append({**info, "source": source})
    return out


def infer_source(*names: str) -> str:
    """Guess the platform from a ``server_name`` and/or a filename.

    Server name is checked first (see :data:`SOURCE_SERVER_RULES`), then the
    filename.  Pass the server name first when you have it::

        infer_source(header.get("server_name", ""), path.name)

    A server name that matches nothing does not fall through to "本地"
    immediately -- the filename is still tried, because plenty of demos come
    from a server nobody has a rule for yet.
    """
    for raw in names:
        if not raw:
            continue
        low = str(raw).lower()
        for needle, source in SOURCE_SERVER_RULES:
            if needle in low:
                return source

    for raw in names:
        if not raw:
            continue
        name = str(raw)
        for pattern, source in SOURCE_FILENAME_PATTERNS:
            if pattern.search(name):
                return source
        low = name.lower()
        for needle, source in SOURCE_FILENAME_RULES:
            if needle in low:
                return source
    return "本地"


# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "watch_roots": [],          # [{path, depth, source, enabled, label}]
    "watch_enabled": True,
    "watch_interval_sec": 5.0,
    "watched_players": [],      # names highlighted in the UI
    "sampling": {"mode": "tick", "interval": 0.25, "stride": 8},
    "auto_extract": True,       # extract to .csz as soon as a demo is ingested
    "make_thumbnails": True,
    "extract_workers": 1,
    # Steam avatars.  A demo carries SteamID64s, not pictures, so these are
    # fetched from steamcommunity.com (no API key, no login -- see app/avatars).
    # Off means the player pages fall back to name initials.
    "fetch_avatars": True,
    "avatar_ttl_days": 7,       # re-fetch a known avatar after this long
    "avatar_delay_ms": 350,     # gap between lookups, to stay polite
    # Not exposed in the UI: a lookup that failed (no proxy, private profile)
    # is retried sooner than a successful one, but not on every page load.
    "avatar_error_retry_hours": 6,
}


def _merge_defaults(loaded: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge saved config over the defaults, keeping unknown keys."""
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in (loaded or {}).items():
        if key == "sampling" and isinstance(value, dict):
            cfg["sampling"] = {**DEFAULTS["sampling"], **value}
        else:
            cfg[key] = value
    return cfg


class Config:
    """Thread-safe JSON-backed settings store.

    ``path`` defaults to :data:`CONFIG_PATH` *resolved at call time*, not bound
    as a default argument -- otherwise the module constant would be frozen into
    the signature at import and could never be redirected (which is exactly
    what the tests need to do to keep off the real config file).
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or CONFIG_PATH)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self.reload()

    # -- io ----------------------------------------------------------------
    def reload(self) -> dict[str, Any]:
        with self._lock:
            loaded: dict[str, Any] = {}
            try:
                if self.path.is_file():
                    loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # A corrupt config must not take the server down; fall back to
                # defaults and let the next save() overwrite it.
                loaded = {}
            self._data = _merge_defaults(loaded)
            return copy.deepcopy(self._data)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)

    # -- access ------------------------------------------------------------
    def all(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return copy.deepcopy(self._data.get(key, default))

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial update and persist it. Returns the new config."""
        with self._lock:
            for key, value in (patch or {}).items():
                if key == "sampling" and isinstance(value, dict):
                    self._data["sampling"] = {**self._data.get("sampling", {}), **value}
                else:
                    self._data[key] = value
            self.save()
            return copy.deepcopy(self._data)

    # -- derived -----------------------------------------------------------
    def extract_kwargs(self) -> dict[str, Any]:
        """Translate the sampling config into ``demo_extract.extract`` kwargs.

        ``stride`` wins over ``interval`` (same precedence as the extractor),
        and "every tick" is expressed by passing neither.
        """
        sampling = self.get("sampling", {}) or {}
        mode = sampling.get("mode", "tick")
        if mode == "stride":
            stride = int(sampling.get("stride") or 8)
            return {} if stride <= 1 else {"stride": stride}
        if mode == "interval":
            interval = float(sampling.get("interval") or 0.25)
            return {} if interval <= 0 else {"interval": interval}
        return {}

    def sampling_label(self) -> str:
        sampling = self.get("sampling", {}) or {}
        mode = sampling.get("mode", "tick")
        if mode == "stride":
            return f"每 {int(sampling.get('stride') or 8)} tick"
        if mode == "interval":
            return f"每 {sampling.get('interval')} 秒"
        return "每个 tick"
