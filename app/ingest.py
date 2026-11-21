"""
Discovery and ingestion: turn a ``.dem`` on disk into a library row.

The pipeline has two deliberate stages, matching how the reference tool works
and how the data actually becomes available:

1. **Register** (fast).  Walk a watch root, hash the file, read the demo header
   and team names via :mod:`app.probe`, and write a ``discovered`` row.  This is
   cheap enough to run over a whole directory tree on every scan, and it is why
   the library can show map / source / teams before anything is extracted.
2. **Extract** (slow).  ``demo_extract.extract`` -> ``.csz`` -> scoreboard,
   roster, watched players, radar thumbnail.  Only this stage needs the heavy
   parser, and it runs on the job queue.

Content deduplication happens at registration: if a byte-identical demo is
already in the library with a valid ``.csz``, the new row points at the same
``.csz`` instead of extracting a second time.  ``delete_csz`` therefore only
unlinks a file when no other row still references it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import avatars, config, db, jobs, metadata, probe, radar

#: Exceptions that mean "the process is going away", not "this demo is bad".
_FATAL = (KeyboardInterrupt, SystemExit, GeneratorExit)

# ── module plumbing ──────────────────────────────────────────────────────────

_TOOLS = config.ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


def _demo_extract():
    """Import ``demo_extract`` lazily so a missing dep cannot break startup."""
    from demo_extract import codec, extract  # noqa: PLC0415

    return codec, extract


# ── small helpers ────────────────────────────────────────────────────────────

_SAFE = re.compile(r"[^A-Za-z0-9_.\u4e00-\u9fff-]+")


def safe_stem(name: str, limit: int = 80) -> str:
    """A filesystem-safe stem that still reads like the original name."""
    stem = Path(name).stem
    stem = _SAFE.sub("_", stem).strip("_")
    return (stem or "demo")[:limit]


def file_md5(path: Path, chunk: int = 4 << 20) -> str:
    """MD5 of a file, read in 4 MB blocks so a 500 MB demo does not load whole."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def csz_path_for(dem_path: Path, md5: str) -> Path:
    """Where the ``.csz`` for this demo lives.

    The digest is part of the name so two copies of the same demo land on one
    file, and the original stem stays readable for anyone browsing the folder.
    """
    return config.CSZ_DIR / f"{safe_stem(dem_path.name)}__{md5[:8]}.csz"


def thumb_path_for(demo_id: int) -> Path:
    return config.THUMB_DIR / f"demo_{demo_id}.png"


def _rel_or_abs(p: Path) -> str:
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


# ── registration ─────────────────────────────────────────────────────────────


def register_demo(
    path: str | Path,
    *,
    watch_root: Optional[str] = None,
    probe_file: bool = True,
    force_hash: bool = False,
    md5: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> tuple[int, bool]:
    """Register (or refresh) a ``.dem`` in the library.

    Returns ``(demo_id, is_new)``.  A row that is already ``ready`` keeps its
    parsed metadata and user annotations -- only identity fields are refreshed,
    so a re-scan never destroys an edited display name or a note.

    Hashing a 300 MB demo costs about a second, which is far too much to repeat
    on every watcher poll, so an unchanged file reuses the digest already on its
    row.  Pass ``force_hash=True`` (the UI's "强制重新扫描") to override.

    ``md5`` is a digest the caller already computed while the bytes were being
    written -- that is how an upload lands on disk, one pass, hash as it goes.
    Supplying it saves reading the same 300 MB a second time just to hash it.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such demo: {p}")

    abs_path = _rel_or_abs(p)
    existing = db.get_by_path(abs_path, db_path)
    is_new = existing is None
    size = p.stat().st_size

    fields: dict[str, Any] = {
        "filename": p.name,
        "file_size": size,
        "match_date": metadata.match_date_from_file(p),
    }
    if watch_root:
        fields["watch_root"] = watch_root

    # -- hash + content dedup ------------------------------------------------
    if md5 is None and not force_hash and existing and existing.get("content_md5") and \
            existing.get("file_size") == size:
        md5 = existing["content_md5"]
    elif md5 is None:
        try:
            md5 = file_md5(p)
        except OSError:
            md5 = None

    if md5:
        fields["content_md5"] = md5
        twin = db.get_by_md5(md5, db_path)
        if twin and twin["id"] != (existing or {}).get("id") and twin.get("csz_path"):
            if Path(twin["csz_path"]).is_file():
                # Same bytes, already extracted: reuse everything derived from
                # the .csz rather than parsing the same demo twice.
                fields.setdefault("csz_path", twin["csz_path"])
                for key in (
                    "map_name", "total_rounds", "duration_mins", "server_name",
                    "team_a_name", "team_b_name", "team_a_score", "team_b_score",
                    "players_json", "scoreboard_json",
                ):
                    if twin.get(key) is not None:
                        fields.setdefault(key, twin[key])
                fields.setdefault("status", db.STATUS_READY)
                fields.setdefault("parsed_at", twin.get("parsed_at"))

    # -- cheap probe ---------------------------------------------------------
    if probe_file and "status" not in fields:
        info = probe.probe_demo(p)
        if info.get("map_name"):
            fields["map_name"] = info["map_name"]
        if info.get("server_name"):
            fields["server_name"] = info["server_name"]
        fields["source"] = config.infer_source(info.get("server_name") or "", p.name)
    elif "source" not in fields and is_new:
        fields["source"] = config.infer_source(p.name)

    if is_new:
        fields["status"] = fields.get("status", db.STATUS_DISCOVERED)
    elif existing and existing["status"] == db.STATUS_MISSING:
        # The file came back.
        fields["status"] = (
            db.STATUS_READY if existing.get("csz_path") else db.STATUS_DISCOVERED
        )
        fields["error_msg"] = None

    # A user-edited source must survive a re-scan.
    if existing and existing.get("source") and existing.get("status") == db.STATUS_READY:
        fields.pop("source", None)

    demo_id = db.upsert_demo(abs_path, db_path=db_path, **fields)
    return demo_id, is_new


# ── zip handling ─────────────────────────────────────────────────────────────


def extract_zip_demos(
    zip_path: Path, *, db_path: Optional[Path] = None
) -> tuple[list[Path], Optional[str]]:
    """Unpack every ``.dem`` inside a ``.zip`` next to the archive.

    Platform clients (5E in particular) hand you a zip rather than a demo.
    Returns ``(extracted_paths, error)``.  An already-extracted member is reused
    when its size matches, so a re-scan does not duplicate files.
    """
    out: list[Path] = []
    try:
        st = zip_path.stat()
    except OSError as exc:
        return out, f"无法读取 zip：{exc}"

    if db.zip_seen(_rel_or_abs(zip_path), st.st_mtime_ns, st.st_size, db_path):
        # Already unpacked; still report the members that exist so a scan can
        # register them if the library row was deleted.
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    if info.filename.lower().endswith(".dem"):
                        cand = zip_path.parent / Path(info.filename).name
                        if cand.is_file():
                            out.append(cand)
        except (OSError, zipfile.BadZipFile):
            pass
        return out, None

    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".dem"):
                    continue
                target = zip_path.parent / Path(info.filename).name
                if target.exists() and target.stat().st_size == info.file_size:
                    out.append(target)
                    continue
                if target.exists():
                    # Never clobber: keep both, clearly named.
                    target = target.with_name(
                        f"{target.stem}_fromzip_{safe_stem(zip_path.name, 24)}_{len(out)}.dem"
                    )
                tmp = target.with_suffix(".dem.partial")
                with zf.open(info) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=4 << 20)
                tmp.replace(target)
                out.append(target)
    except zipfile.BadZipFile as exc:
        return out, f"zip 损坏或缺少中央目录（部分 5E 回放包如此）：{exc}"
    except OSError as exc:
        return out, f"解压失败：{exc}"

    db.mark_zip(_rel_or_abs(zip_path), st.st_mtime_ns, st.st_size, db_path)
    return out, None


# ── scanning ─────────────────────────────────────────────────────────────────


def _iter_candidates(root: Path, depth: int) -> Iterable[Path]:
    """Yield ``.dem`` / ``.zip`` files under ``root``, at most ``depth`` deep."""
    if depth <= 0:
        for entry in sorted(root.iterdir()):
            if entry.is_file() and entry.suffix.lower() in (".dem", ".zip"):
                yield entry
        return
    root_depth = len(root.parts)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if len(path.parts) - root_depth > depth:
            continue
        if path.suffix.lower() in (".dem", ".zip"):
            yield path


def scan_roots(
    roots: Optional[list[dict]] = None,
    *,
    progress: Optional[Callable[[str, Optional[float]], None]] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Walk every enabled watch root and register what is found.

    Returns a summary ``{scanned, registered, skipped, zips, errors, new_ids}``.
    ``new_ids`` is what the caller queues for extraction.
    """
    cfg = config.Config() if roots is None else None
    if roots is None:
        roots = [r for r in (cfg.get("watch_roots") or []) if r.get("enabled", True)]

    def report(msg: str, pct: Optional[float] = None) -> None:
        if progress:
            progress(msg, pct)

    summary: dict[str, Any] = {
        "scanned": 0, "registered": 0, "skipped": 0,
        "zips": 0, "errors": [], "new_ids": [], "recovered": 0,
    }

    for i, root in enumerate(roots):
        raw = root.get("path")
        if not raw:
            continue
        base = Path(config.expand(raw))
        if not base.is_dir():
            summary["errors"].append(f"{raw}：目录不存在")
            continue
        depth = int(root.get("depth", 2) or 0)
        report(f"扫描 {base} …", i / max(1, len(roots)))
        try:
            candidates = list(_iter_candidates(base, depth))
        except OSError as exc:
            summary["errors"].append(f"{base}：{exc}")
            continue

        for cand in candidates:
            if cand.suffix.lower() == ".zip":
                summary["zips"] += 1
                extracted, err = extract_zip_demos(cand, db_path=db_path)
                if err:
                    summary["errors"].append(f"{cand.name}：{err}")
                candidates_extra = extracted
            else:
                candidates_extra = [cand]

            for dem in candidates_extra:
                summary["scanned"] += 1
                try:
                    demo_id, is_new = register_demo(
                        dem, watch_root=str(base), db_path=db_path
                    )
                except (OSError, ValueError) as exc:
                    summary["errors"].append(f"{dem.name}：{exc}")
                    continue
                if is_new:
                    summary["registered"] += 1
                    summary["new_ids"].append(demo_id)
                else:
                    summary["skipped"] += 1

    # Pick up work orphaned by a restart: a demo that was discovered (or was
    # mid-extraction) when the process died has no .csz and would otherwise sit
    # at "待入库" forever, because nothing re-queues an existing row.
    orphans = [i for i in db.unfinished_ids(db_path) if i not in set(summary["new_ids"])]
    if orphans:
        summary["recovered"] = len(orphans)
        summary["new_ids"].extend(orphans)

    report("扫描完成", 1.0)
    return summary


# ── the extraction pipeline ──────────────────────────────────────────────────


def _ticker(job: jobs.Job, base: str):
    """Context manager that keeps a job's step text counting up while it blocks.

    ``demo_extract.extract`` gives no progress callbacks, so the honest thing to
    show is which phase we are in and how long it has been running.
    """
    import contextlib
    import threading

    @contextlib.contextmanager
    def _cm():
        stop = threading.Event()
        t0 = time.time()

        def run() -> None:
            while not stop.wait(1.0):
                job.set_step(f"{base}（已用 {time.time() - t0:.0f}s）")

        th = threading.Thread(target=run, daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()

    return _cm()


def run_extraction(demo_id: int, job: jobs.Job) -> dict[str, Any]:
    """Extract one demo to ``.csz`` and fill in everything derived from it."""
    row = db.get_demo(demo_id)
    if row is None:
        raise ValueError(f"demo {demo_id} 不在库里")

    src = Path(row["path"])
    if not src.is_file():
        db.update_demo(demo_id, status=db.STATUS_MISSING, error_msg="源文件不存在")
        raise FileNotFoundError(f"源文件不存在：{src}")

    db.update_demo(demo_id, status=db.STATUS_EXTRACTING, error_msg=None)

    codec, extract_demo = _demo_extract()
    cfg = config.Config()
    kwargs = cfg.extract_kwargs()

    job.set_step("读取 demo 头与队伍名…", 0.05)
    info = probe.probe_demo(src)

    job.set_step(f"解析 demo（{cfg.sampling_label()}）…", 0.1)
    t0 = time.time()
    # side-channel 的暂存文件（`.csz` 那步之前的一个 blob）默认落在
    # `tempfile.gettempdir()`，也就是 `%TEMP%`。这在两种机器上会直接失败：
    # 杀软 / 组策略禁止未知 exe 往 `%TEMP%` 写、以及沙箱只放行工作区。
    # 报到用户眼前就是 `side-channel failed (exit 1): ... Access is denied`。
    # 放到数据目录里（和 `.csz` 同一个盘、同样的权限），一条命令都不用加。
    scratch = Path(config.DATA_DIR) / "tmp"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
    except OSError:
        scratch = Path(config.DATA_DIR)      # 数据目录本身可写就够
    try:
        with _ticker(job, f"解析 demo（{cfg.sampling_label()}）"):
            data = extract_demo(str(src), workdir=scratch, **kwargs)
    except _FATAL:
        db.update_demo(demo_id, status=db.STATUS_ERROR, error_msg="已中断")
        raise
    except BaseException as exc:
        # The row must not be left reading "提取中" forever.  BaseException
        # because demoparser2 panics on a truncated or corrupt demo and PyO3
        # raises that as PanicException, which is not an Exception.
        db.update_demo(
            demo_id, status=db.STATUS_ERROR,
            error_msg=probe._describe(exc, "解析失败"),
        )
        raise
    parse_s = time.time() - t0

    job.set_step(f"解析完成（{parse_s:.0f}s），写出 .csz…", 0.7)
    md5 = row.get("content_md5") or file_md5(src)
    csz = csz_path_for(src, md5)
    codec.save(data, csz)

    # 逐帧遮挡缓存：跟着 csz 一起落盘。放在这里而不是"第一次点 AI 的时候"，
    # 是因为它只依赖 csz 里的几何 + 地图 mesh，和模型、采样窗口都无关 —— 算一次
    # 就够，之后推理 / 造数都只是读数组（推理从 3~4 s 降到 0.1 s，且不再建 BVH）。
    job.set_step("缓存逐帧遮挡…", 0.72)
    try:
        from . import obstruction as _obstruction

        cache_info = _obstruction.build_demo(
            csz,
            data,
            progress=lambda frac, msg: job.set_step(msg, 0.72 + 0.11 * float(frac)),
        )
        if cache_info.get("failed"):
            # 单个回合失败不算提取失败：那一回合下次用到时会重算。
            print(
                f"[obstruction] demo {demo_id} 有 {len(cache_info['failed'])} "
                f"个回合的遮挡缓存没建成"
            )
    except Exception as exc:  # noqa: BLE001 - 缓存失败绝不能让提取失败
        print(f"[obstruction] demo {demo_id} 遮挡缓存失败：{type(exc).__name__}: {exc}")

    job.set_step("统计比分与选手数据…", 0.85)
    md = metadata.build_metadata(
        data,
        filename=src.name,
        watched_players=cfg.get("watched_players") or [],
        clans=probe.clan_by_steamid(info),
    )
    if info.get("server_name"):
        md["server_name"] = info["server_name"]

    thumb = None
    if cfg.get("make_thumbnails", True):
        job.set_step("生成雷达缩略图…", 0.92)
        try:
            out = thumb_path_for(demo_id)
            if radar.render_demo_thumbnail(
                data,
                map_name=md.get("map_name") or "",
                team_names=(md.get("team_a_name") or "Team A", md.get("team_b_name") or "Team B"),
                score=(md.get("team_a_score"), md.get("team_b_score")),
                out_path=out,
            ):
                thumb = str(out)
        except Exception as exc:  # noqa: BLE001 - a thumbnail must never fail ingest
            print(f"[thumb] demo {demo_id} 缩略图失败：{type(exc).__name__}: {exc}")

    # The user may have corrected the source by hand; only fill it in when the
    # row has nothing yet.
    updates: dict[str, Any] = {
        **md,
        "csz_path": str(csz),
        "content_md5": md5,
        "status": db.STATUS_READY,
        "parsed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_msg": None,
    }
    if thumb:
        updates["thumb_path"] = thumb
        updates["thumb_key"] = md5
    if not row.get("source") or row.get("source") == "本地":
        updates["source"] = config.infer_source(
            info.get("server_name") or "", src.name
        )
    db.update_demo(demo_id, **updates)

    # Steam avatars for the roster that just became known.  Deliberately not
    # part of this job: it is network-bound, and one job that both parses and
    # fetches would hold the extraction queue for the duration of ten HTTP
    # round trips.  queue_sync coalesces, so a scan that finishes 60 demos
    # still results in about two avatar jobs, not 60.
    try:
        avatars.queue_sync()
    except Exception as exc:  # noqa: BLE001 - a face must never fail an extract
        print(f"[avatars] 排队失败：{type(exc).__name__}: {exc}")

    job.set_step("完成", 1.0)
    return {
        "demo_id": demo_id,
        "csz": str(csz),
        "map": md.get("map_name"),
        "rounds": md.get("total_rounds"),
        "score": [md.get("team_a_score"), md.get("team_b_score")],
        "parse_seconds": round(parse_s, 1),
        "csz_mb": round(csz.stat().st_size / 1e6, 1) if csz.is_file() else None,
    }


def _json_list(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except ValueError:
        return []


def resync_watched() -> dict[str, int]:
    """Re-derive ``watched_json`` for every row from its stored scoreboard.

    ``watched_json`` is derived data: the scoreboard crossed with the config's
    watched list.  The library cards, the ``watched=1`` filter and the
    thumbnail all read that stored copy rather than re-matching on every
    request, so editing the list leaves every existing row stale -- unfollowing
    a player kept the old matches highlighted until each one was refreshed by
    hand, which is exactly backwards: the user just said they are not
    interested any more.

    No ``.csz`` is needed.  The scoreboard is already in the database, so this
    is JSON work over rows (milliseconds for a library of any plausible size),
    not a re-parse, and it can run inside the request that changed the list.

    Rows whose value would not change are left alone, so this does not churn
    timestamps on a library-wide scale.
    """
    wanted = config.Config().get("watched_players") or []
    rows = db.scoreboard_rows()
    updates: list[tuple[str, int]] = []
    for row in rows:
        try:
            scoreboard = json.loads(row["scoreboard_json"] or "[]")
        except ValueError:
            continue
        if not isinstance(scoreboard, list):
            continue
        fresh = metadata.match_watched(scoreboard, wanted)
        encoded = json.dumps(fresh, ensure_ascii=False)
        if encoded != (row["watched_json"] or "[]"):
            updates.append((encoded, row["id"]))
    db.set_watched(updates)
    return {"checked": len(rows), "changed": len(updates)}


def refresh_metadata(demo_id: int, job: Optional[jobs.Job] = None) -> dict[str, Any]:
    """Re-derive scoreboard/roster/thumbnail from an existing ``.csz``.

    Used after the user edits the watched-player list, and after a manual
    re-parse.  Does not re-read the original demo.
    """
    row = db.get_demo(demo_id)
    if row is None:
        raise ValueError(f"demo {demo_id} 不在库里")
    csz = row.get("csz_path")
    if not csz or not Path(csz).is_file():
        raise FileNotFoundError("这个 Demo 还没有 .csz，请先提取")

    if job:
        job.set_step("读取 .csz…", 0.2)
    codec, _ = _demo_extract()
    data = codec.load(csz)

    cfg = config.Config()
    info = probe.probe_demo(row["path"]) if Path(row["path"]).is_file() else {}

    if job:
        job.set_step("重算比分与选手数据…", 0.6)
    md = metadata.build_metadata(
        data,
        filename=row["filename"],
        watched_players=cfg.get("watched_players") or [],
        clans=probe.clan_by_steamid(info),
    )
    # Team names the user edited by hand win over a re-derivation.
    if row.get("team_a_name") and row["team_a_name"] not in ("", "Team A"):
        md["team_a_name"] = row["team_a_name"]
    if row.get("team_b_name") and row["team_b_name"] not in ("", "Team B"):
        md["team_b_name"] = row["team_b_name"]

    updates = {**md, "status": db.STATUS_READY, "error_msg": None}

    if cfg.get("make_thumbnails", True):
        if job:
            job.set_step("重建缩略图…", 0.85)
        try:
            out = thumb_path_for(demo_id)
            if radar.render_demo_thumbnail(
                data,
                map_name=md.get("map_name") or "",
                team_names=(md["team_a_name"], md["team_b_name"]),
                score=(md.get("team_a_score"), md.get("team_b_score")),
                out_path=out,
            ):
                updates["thumb_path"] = str(out)
                updates["thumb_key"] = row.get("content_md5")
        except Exception as exc:  # noqa: BLE001
            print(f"[thumb] demo {demo_id} 重建失败：{type(exc).__name__}: {exc}")

    db.update_demo(demo_id, **updates)
    if job:
        job.set_step("完成", 1.0)
    return {"demo_id": demo_id, "watched": len(_json_list(md.get("watched_json")))}


# ── queue helpers ────────────────────────────────────────────────────────────


def extraction_subject(demo_id: int, stage: str = "extract") -> tuple:
    """The queue key that means "this demo, this kind of work".

    Keyed on the demo's *path* rather than its row id: the id is only unique
    inside one library, while the job queue is process-wide and outlives a
    library swap (the test suite points ``DATA_DIR`` at a fresh directory per
    group, and every one of them numbers its rows from 1 again -- an id-keyed
    subject made a brand-new demo inherit the refusal meant for an unrelated
    queued job).
    """
    row = db.get_demo(int(demo_id))
    return ("demo", str(row["path"]) if row else str(demo_id), stage)


def queue_extraction(demo_ids: Iterable[int], *, label: Optional[str] = None,
                     force: bool = False,
                     skipped: Optional[list[int]] = None) -> list[str]:
    """Submit extraction jobs for the given demo ids. Returns the job ids.

    A demo that already has a valid ``.csz`` is refreshed from it rather than
    re-parsed, so "提取" on a ready row is cheap and idempotent.

    ``force=True`` skips that shortcut and re-parses the ``.dem`` regardless.
    Without it there is **no way to re-extract a ready row**, which is what you
    need after changing the sampling settings: the shortcut would happily
    refresh metadata from the old, coarsely-sampled ``.csz`` and report success
    while the sampling setting had no effect at all.

    A demo whose extraction is **already queued or running** is skipped -- its id
    lands in ``skipped`` instead of getting a second job.  This is not a nicety:
    four different callers can ask for the same demo in the same few seconds
    (the watcher's poll, the manual 扫描's orphan sweep, the 提取 button, and
    打开回放's "queue the missing ``.csz``"), and each job is a full parse of a
    file that takes 30-90 s.  A refresh of a ready row is a *different* subject,
    so it is still allowed alongside an in-flight re-parse.
    """
    job_ids: list[str] = []
    for demo_id in demo_ids:
        demo = int(demo_id)
        row = db.get_demo(demo)
        if row is None:
            continue
        has_csz = bool(row.get("csz_path")) and Path(row["csz_path"]).is_file()
        name = row.get("display_name") or row["filename"]
        # A refresh only re-reads the .csz; it does not parse the demo, so it is
        # allowed to sit next to an extraction instead of being folded into it.
        stage = "refresh" if (has_csz and not force) else "extract"
        job = jobs.manager.submit_unique(
            stage,
            f"{'刷新' if stage == 'refresh' else ('重新解析' if has_csz else '提取')} {name}",
            (lambda j, i=demo: refresh_metadata(i, j)) if stage == "refresh"
            else (lambda j, i=demo: run_extraction(i, j)),
            subject=extraction_subject(demo, stage),
        )
        if job is None:
            if skipped is not None:
                skipped.append(demo)
            continue
        job_ids.append(job.id)
    return job_ids


def delete_csz_if_unreferenced(csz_path: Optional[str],
                               db_path: Optional[Path] = None) -> bool:
    """Unlink a ``.csz`` only when no other library row still points at it."""
    if not csz_path:
        return False
    with db.connect(db_path) as conn:
        others = conn.execute(
            "SELECT COUNT(*) AS n FROM demo_files WHERE csz_path = ?", (csz_path,)
        ).fetchone()["n"]
    if others:
        return False
    try:
        Path(csz_path).unlink(missing_ok=True)
    except OSError:
        return False
    from . import obstruction as _obstruction

    _obstruction.discard(csz_path)
    return True
