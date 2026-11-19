"""
Disk usage of everything the app keeps for itself.

The library stores absolute paths and nothing else, so the bytes the app
actually owns live in a handful of directories under ``app/data``:

    csz/        one .csz per parsed demo -- by far the biggest thing here
    thumbs/     radar thumbnails, plus a per-map render cache in thumbs/_maps
    avatars/    Steam profile pictures
    recycle/    .dem files the user chose to remove

The original ``.dem`` files are deliberately *not* in this list.  They stay in
the user's watch directory, and the library only points at them; deleting one
is a per-demo decision made in the Demo 库, not a cache clear.

Clearing a cache is a two-part operation: the files go, and the rows that
pointed at them are updated so the library does not keep claiming data it no
longer has.  Both halves are here so they cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from . import config, db

#: Everything the settings page is allowed to wipe, in display order.
CLEANABLE = ("csz", "thumbs", "avatars", "recycle")


def recycle_dir() -> Path:
    return config.DATA_DIR / "recycle"


def _measure(root: Path) -> tuple[int, int]:
    """``(files, bytes)`` under ``root``, ignoring anything unreadable."""
    files = 0
    total = 0
    if not root.is_dir():
        return files, total
    for entry in root.rglob("*"):
        try:
            if not entry.is_file():
                continue
            total += entry.stat().st_size
        except OSError:
            continue
        files += 1
    return files, total


def _unlink_tree(root: Path) -> dict[str, Any]:
    """Delete every file under ``root``, then the directories left behind.

    Returns ``{files, bytes, errors}``.  A file that will not delete is
    reported rather than raised: one locked avatar must not stop the other
    three hundred from going.
    """
    removed = 0
    freed = 0
    errors: list[str] = []
    if not root.is_dir():
        return {"files": 0, "bytes": 0, "errors": errors}

    for entry in sorted(root.rglob("*")):
        try:
            if not entry.is_file():
                continue
            size = entry.stat().st_size
            entry.unlink()
        except OSError as exc:
            errors.append(f"{entry.name}：{exc}")
            continue
        removed += 1
        freed += size

    # Deepest first, so ``thumbs/_maps`` empties before ``thumbs`` is tried.
    dirs = [e for e in root.rglob("*") if e.is_dir()]
    for entry in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            entry.rmdir()
        except OSError:
            pass
    return {"files": removed, "bytes": freed, "errors": errors}


def _entry(key: str, label: str, path: Path, note: str,
           *, removable: bool = True) -> dict[str, Any]:
    files, total = _measure(path)
    return {
        "key": key,
        "label": label,
        "path": str(path),
        "files": files,
        "bytes": total,
        "removable": removable,
        "note": note,
    }


def overview() -> dict[str, Any]:
    """Sizes of the caches, plus what the library points at."""
    entries = [
        _entry("csz", "提取文件 .csz", config.CSZ_DIR, "清掉后需要重新提取"),
        _entry("thumbs", "雷达缩略图", config.THUMB_DIR, "清掉后会按需重建"),
        _entry("avatars", "Steam 头像", config.AVATAR_DIR, "清掉后会重新从 Steam 取"),
        _entry("recycle", "回收站", recycle_dir(), "清掉后无法恢复"),
        _db_entry(),
    ]
    library = db.stats()
    return {
        "data_dir": str(config.DATA_DIR),
        "total_bytes": sum(e["bytes"] for e in entries),
        "entries": entries,
        # Informational: the originals live in the watch directories, which are
        # outside app/data, and they are removed per demo from the library.
        "demos": {"count": int(library.get("total") or 0),
                  "bytes": int(library.get("bytes") or 0)},
    }


def _db_entry() -> dict[str, Any]:
    """The SQLite file and its WAL sidecars, as one row."""
    files = 0
    total = 0
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(config.LIBRARY_DB) + suffix)
        try:
            if p.is_file():
                total += p.stat().st_size
                files += 1
        except OSError:
            continue
    return {
        "key": "db",
        "label": "库数据库",
        "path": str(config.LIBRARY_DB),
        "files": files,
        "bytes": total,
        "removable": False,
        "note": "库记录本身",
    }


def _clean_csz() -> dict[str, Any]:
    result = _unlink_tree(config.CSZ_DIR)
    result["rows"] = db.forget_csz()
    return result


def _clean_thumbs() -> dict[str, Any]:
    result = _unlink_tree(config.THUMB_DIR)
    result["rows"] = db.forget_thumbs()
    return result


def _clean_avatars() -> dict[str, Any]:
    result = _unlink_tree(config.AVATAR_DIR)
    result["rows"] = db.forget_avatar_files()
    return result


def _clean_recycle() -> dict[str, Any]:
    # Nothing in the database points at the recycle bin, so there is no second
    # half here.
    return _unlink_tree(recycle_dir())


_CLEANERS = {
    "csz": _clean_csz,
    "thumbs": _clean_thumbs,
    "avatars": _clean_avatars,
    "recycle": _clean_recycle,
}


def clean(targets: Iterable[str]) -> dict[str, Any]:
    """Wipe the named caches.  Raises ``ValueError`` for an unknown target."""
    wanted = [t for t in dict.fromkeys(str(x or "") for x in targets) if t]
    unknown = [t for t in wanted if t not in CLEANABLE]
    if unknown:
        raise ValueError(f"不能清理：{'、'.join(unknown)}")
    if not wanted:
        raise ValueError("没有可清理的目标")

    cleaned = {key: _CLEANERS[key]() for key in wanted}
    return {"cleaned": cleaned, "overview": overview()}
