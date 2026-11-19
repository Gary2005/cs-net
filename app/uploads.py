"""
Browser uploads: turn a dropped ``.dem`` into a library row.

The watcher is for demos that *appear* in a folder; this is for demos the user
hands over directly -- drag a file onto the library page, or pick several in the
file dialog.  Three things follow from that difference:

* **The bytes arrive through HTTP.**  The browser sends the file once, and it is
  copied straight into ``data/imports`` while the MD5 is computed on the same
  pass.  Nothing is held in memory: a 400 MB demo is 400 MB of disk, not RAM.

* **A second copy is a bug, not a feature.**  ``.dem`` files are big and people
  download the same match twice, so before a file is even renamed into place its
  digest is looked up in the library.  A hit means "already here": the uploaded
  copy is deleted, no row is created, and no extraction is queued.  That answer
  is available *before* the parse would have started, which is the whole reason
  the hash is computed during the upload.  "Already here" means the file is
  still on disk, though -- a row whose ``.dem`` was deleted must not block the
  one upload that could put that match back.

* **Uploading is an explicit request to add this demo**, so it is queued for
  extraction even when ``auto_extract`` is off (that setting governs what the
  directory watcher does on its own).

``.zip`` is accepted because 5E hands out archives rather than demos; the
members are unpacked next to it and each one goes through the same duplicate
check.  A member that turns out to be a duplicate is deleted again -- it only
existed for the few milliseconds it took to hash it.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional

from . import config, db, ingest

#: Read/write block.  4 MB keeps the loop cheap without making progress coarse.
CHUNK = 4 << 20
#: Per-file ceiling.  A demo is 50-400 MB; a 5E archive wrapping several matches
#: is the largest thing worth accepting, and 2 GB is comfortably above that.
MAX_FILE_BYTES = 2 * 1024**3
#: Ceiling for one request (the browser sends every selected file in it).
MAX_REQUEST_BYTES = 6 * 1024**3
#: How many files one request may carry.  The UI shows a row per file, so this is
#: also a limit on how long that list gets.
MAX_FILES = 40

DEMO_SUFFIXES = (".dem",)
ARCHIVE_SUFFIXES = (".zip",)
ALLOWED_SUFFIXES = DEMO_SUFFIXES + ARCHIVE_SUFFIXES

_PART_PREFIX = ".upload-"
_PART_SUFFIX = ".part"
#: A ``.part`` older than this is garbage from a cancelled upload or a killed
#: process, not something still being written.
_PART_MAX_AGE = 24 * 3600


class UploadRejected(Exception):
    """Retrying will not help: wrong format, empty file, or too large."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UploadFailed(Exception):
    """Writing or registering failed.  Not the file's fault; a retry may work."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ── where uploads live ───────────────────────────────────────────────────────


def imports_dir() -> Path:
    """``data\\imports`` -- the folder uploaded demos are copied into."""
    return config.imports_dir()


def _abs(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _prune_partials() -> None:
    """Drop half-written files left behind by a cancelled upload."""
    root = imports_dir()
    if not root.is_dir():
        return
    cutoff = time.time() - _PART_MAX_AGE
    for entry in root.glob(f"{_PART_PREFIX}*{_PART_SUFFIX}"):
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            continue


# ── shapes the API returns ───────────────────────────────────────────────────


def _existing_view(row: dict[str, Any]) -> dict[str, Any]:
    """What the UI needs to say "this demo is already in the library"."""
    return {
        "demo_id": row.get("id"),
        "name": row.get("display_name") or row.get("filename"),
        "filename": row.get("filename"),
        "path": row.get("path"),
        "status": row.get("status"),
        "status_label": row.get("status_label"),
        "map": row.get("map_name"),
        "added_at": row.get("added_at"),
    }


def _item(name: str, size: Optional[int], status: str, *,
          reason: Optional[str] = None,
          demos: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    return {
        "name": name,
        "size": size,
        "status": status,
        "reason": reason,
        "demos": list(demos or []),
    }


def _demo(name: str, status: str, *, demo_id: Optional[int] = None,
          existing: Optional[dict[str, Any]] = None,
          reason: Optional[str] = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "demo_id": demo_id,
        "existing": existing,
        "reason": reason,
    }


# ── writing the bytes ────────────────────────────────────────────────────────


def _write(stream: BinaryIO, dest: Path) -> tuple[str, int]:
    """Copy ``stream`` to ``dest``, hashing as it goes.  Returns ``(md5, size)``.

    One pass over the bytes, which is what makes the duplicate check free: by
    the time the file is on disk its digest is already known, so the caller can
    answer "already in the library?" before doing anything expensive with it.
    """
    digest = hashlib.md5()
    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                block = stream.read(CHUNK)
                if not block:
                    break
                size += len(block)
                if size > MAX_FILE_BYTES:
                    raise UploadRejected(
                        f"文件超过 {MAX_FILE_BYTES >> 30} GB，先确认一下选错了没有"
                    )
                digest.update(block)
                out.write(block)
    except UploadRejected:
        _unlink(dest)
        raise
    except OSError as exc:
        _unlink(dest)
        raise UploadFailed(f"写入失败：{exc}") from exc
    return digest.hexdigest(), size


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# ── registering ──────────────────────────────────────────────────────────────


def _register(path: Path, md5: str, name: str) -> dict[str, Any]:
    """Register ``path`` unless its content is already in the library.

    The digest check comes first and uses the digest we just computed, so a
    second copy of the same match costs one hash and nothing else -- not a row,
    not a ``.csz``, not a parse.  It asks for a *live* twin: a row whose file was
    deleted must not make the upload impossible.
    """
    twin = db.live_by_md5(md5)
    if twin is not None:
        return _demo(name, "duplicate", existing=_existing_view(twin))

    try:
        demo_id, is_new = ingest.register_demo(
            path, watch_root=str(imports_dir()), md5=md5
        )
    except (OSError, ValueError) as exc:
        raise UploadFailed(f"入库失败：{type(exc).__name__}: {exc}") from exc
    row = db.get_demo(demo_id) or {}
    # The file on disk carries the digest in its name (so two different demos
    # called ``match.dem`` cannot overwrite each other); what the library shows
    # should be the name the user uploaded.  Set once, never on a re-register --
    # after that it belongs to them.
    if is_new and name and name != row.get("filename"):
        db.update_demo(demo_id, display_name=name)
        row["display_name"] = name
    return _demo(
        name,
        "imported",
        demo_id=demo_id,
        existing={
            "demo_id": demo_id,
            "map": row.get("map_name"),
            "source": row.get("source"),
            "filename": row.get("filename"),
        },
    )


def _import_archive(archive: Path, name: str, size: int) -> dict[str, Any]:
    """Unpack a ``.zip`` and register every ``.dem`` inside it."""
    before = {p.name for p in archive.parent.glob("*.dem")}
    members, error = ingest.extract_zip_demos(archive)
    if not members and error:
        return _item(name, size, "failed", reason=error)
    if not members:
        return _item(name, size, "rejected",
                     reason="压缩包里没有 .dem 文件")

    demos: list[dict[str, Any]] = []
    for member in members:
        try:
            digest = ingest.file_md5(member)
        except OSError as exc:
            demos.append(_demo(member.name, "failed", reason=str(exc)))
            continue
        try:
            entry = _register(member, digest, member.name)
        except UploadFailed as exc:
            demos.append(_demo(member.name, "failed", reason=exc.reason))
            continue
        if entry["status"] == "duplicate":
            # A member that was just unpacked and turned out to be a copy of
            # something already in the library has no reason to stay on disk.
            twin_path = (entry.get("existing") or {}).get("path")
            if not twin_path or Path(twin_path) != _abs(member):
                if member.name not in before:
                    _unlink(member)
        demos.append(entry)

    statuses = {d["status"] for d in demos}
    if statuses == {"imported"}:
        status = "imported"
    elif "imported" in statuses:
        status = "partial"
    elif statuses == {"duplicate"}:
        status = "duplicate"
    else:
        status = "failed"
    return _item(name, size, status, demos=demos)


# ── the entry points ─────────────────────────────────────────────────────────


def import_one(stream: BinaryIO, filename: str) -> dict[str, Any]:
    """Save one uploaded file, register it, and describe what happened.

    Raises :class:`UploadRejected` / :class:`UploadFailed`; the batch entry point
    turns those into per-file rows so one bad file cannot fail the others.
    """
    name = Path(filename or "").name or "upload.dem"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected("只能上传 .dem 或 .zip")

    root = imports_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UploadFailed(f"上传目录不可写：{exc}") from exc

    partial = root / f"{_PART_PREFIX}{uuid.uuid4().hex[:12]}{_PART_SUFFIX}"
    md5, size = _write(stream, partial)
    if size == 0:
        _unlink(partial)
        raise UploadRejected("文件是空的")

    if suffix in ARCHIVE_SUFFIXES:
        # The archive itself never gets a library row, so its digest is only
        # useful as part of the file name.
        final = root / f"{ingest.safe_stem(name, 60)}__{md5[:8]}{suffix}"
        try:
            os.replace(partial, final)
        except OSError as exc:
            _unlink(partial)
            raise UploadFailed(f"写入失败：{exc}") from exc
        return _import_archive(final, name, size)

    final = root / f"{ingest.safe_stem(name, 60)}__{md5[:8]}{suffix}"
    twin = db.live_by_md5(md5)
    if twin is not None:
        # Already in the library: drop the copy we just wrote.  ``final`` is
        # left alone -- when the same name and content were uploaded before it
        # *is* a row's file, and deleting it would break that row.
        _unlink(partial)
        return _item(name, size, "duplicate", demos=[
            _demo(name, "duplicate", existing=_existing_view(twin))
        ])

    try:
        os.replace(partial, final)
    except OSError as exc:
        _unlink(partial)
        raise UploadFailed(f"写入失败：{exc}") from exc

    try:
        entry = _register(final, md5, name)
    except UploadFailed:
        _unlink(final)  # no row points at it; leaving it would be a leak
        raise
    return _item(name, size, "imported" if entry["status"] == "imported" else "duplicate",
                 demos=[entry])


def import_uploads(files: Iterable[Any]) -> dict[str, Any]:
    """Handle one request's worth of uploads and queue the extractions."""
    uploads = list(files)
    _prune_partials()

    items: list[dict[str, Any]] = []
    for upload in uploads[:MAX_FILES]:
        name = Path(getattr(upload, "filename", "") or "").name or "upload.dem"
        try:
            items.append(import_one(upload.stream, name))
        except UploadRejected as exc:
            items.append(_item(name, None, "rejected", reason=exc.reason))
        except UploadFailed as exc:
            items.append(_item(name, None, "failed", reason=exc.reason))
        except Exception as exc:  # noqa: BLE001 - one bad file must not fail the batch
            items.append(_item(name, None, "failed",
                               reason=f"{type(exc).__name__}: {exc}"))

    demos = [d for item in items for d in item.get("demos") or []]
    imported = [d for d in demos if d["status"] == "imported" and d.get("demo_id")]
    duplicates = [d for d in demos if d["status"] == "duplicate"]

    job_ids: list[str] = []
    skipped: list[int] = []
    if imported:
        job_ids = ingest.queue_extraction(
            [int(d["demo_id"]) for d in imported], skipped=skipped
        )

    return {
        "received": len(items),
        "truncated": max(0, len(uploads) - MAX_FILES),
        "imported": len(imported),
        "duplicates": len(duplicates),
        "rejected": sum(1 for item in items if item["status"] == "rejected"),
        "failed": sum(1 for item in items if item["status"] == "failed"),
        "queued": len(job_ids),
        "job_ids": job_ids,
        "already_queued": len(skipped),
        "items": items,
    }
