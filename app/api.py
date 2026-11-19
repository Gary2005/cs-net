"""
HTTP API for the demo library.

Registered as a blueprint on the app built by :mod:`app.server`, which also
mounts the existing visualizer.  Everything lives under ``/api/library`` so it
cannot collide with the visualizer's own ``/api/*`` routes.

The one endpoint that reaches across the boundary is ``POST /api/library/open``:
it turns a library selection into a *batch* and returns the URL of the replay
page.  The replay page then loads each demo through the visualizer's existing
``POST /api/load``, so nothing about the replay data path changes.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, jsonify, request, send_file

from . import avatars, config, db, ingest, jobs, radar, storage, watcher

bp = Blueprint(
    "library",
    __name__,
    static_folder=str(Path(__file__).resolve().parent / "static"),
    static_url_path="/lib-static",
    template_folder=str(Path(__file__).resolve().parent / "templates"),
)


# ── batches: a library selection handed to the replay page ───────────────────

_batches: dict[str, dict[str, Any]] = {}
_batch_lock = threading.Lock()
_BATCH_TTL = 6 * 3600


def _prune_batches() -> None:
    cutoff = time.time() - _BATCH_TTL
    with _batch_lock:
        for key in [k for k, v in _batches.items() if v["created"] < cutoff]:
            _batches.pop(key, None)


def _batch_view(batch_id: str) -> Optional[dict[str, Any]]:
    with _batch_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return None
        items = []
        for item in batch["items"]:
            row = db.get_demo(item["demo_id"])
            if row is None:
                continue
            csz = row.get("csz_path")
            ready = bool(csz) and Path(csz).is_file()
            items.append({
                "demo_id": row["id"],
                "name": row.get("display_name") or row["filename"],
                "filename": row["filename"],
                "map": row.get("map_name"),
                "status": row["status"],
                "status_label": row["status_label"],
                "ready": ready,
                "csz_path": csz if ready else None,
                "source": row.get("source"),
                "team_a_name": row.get("team_a_name"),
                "team_b_name": row.get("team_b_name"),
                "team_a_score": row.get("team_a_score"),
                "team_b_score": row.get("team_b_score"),
                "thumb": f"/api/library/thumb/{row['id']}" if row.get("thumb_path") else None,
            })
        return {"batch_id": batch_id, "created": batch["created"], "items": items}


# ── helpers ──────────────────────────────────────────────────────────────────


def _bad(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _int_arg(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


def _body() -> dict[str, Any]:
    return request.get_json(silent=True) or {}


# ── avatars / players ────────────────────────────────────────────────────────
#
# The list endpoint ships the avatars it can already serve, so a card never has
# to make ten extra round trips.  Only players with a cached picture are listed;
# everything else falls back to an initial, which is also what happens when the
# machine cannot reach Steam at all.


def _avatar_view(row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not row:
        return None
    path = row.get("file_path")
    has_file = bool(path) and Path(path).is_file()
    buster = ""
    if has_file:
        try:
            buster = f"?v={int(Path(path).stat().st_mtime)}"
        except OSError:
            buster = ""
    return {
        "steamid": row.get("steamid"),
        "persona": row.get("persona_name") or None,
        "status": row.get("status"),
        "error": row.get("error_msg"),
        "fetched_at": row.get("fetched_at"),
        "url": f"/api/library/avatar/{row['steamid']}{buster}" if has_file else None,
    }


def _avatars_for(steamids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sid, row in db.avatars_for(steamids).items():
        view = _avatar_view(row)
        if view and view["url"]:
            out[sid] = view
    return out


def _roster_ids(items: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in items:
        for player in (item.get("players") or []):
            sid = str(player.get("steamid") or "")
            if sid:
                ids.append(sid)
    return ids


def _mark_watched(players: list[dict[str, Any]]) -> None:
    """Flag the players whose in-game name matches the watched list.

    Matching is by substring, case-insensitively, exactly like the library's
    highlight, so the two pages cannot disagree about who is "watched".
    """
    wanted = [str(w).lower() for w in (config.Config().get("watched_players") or []) if w]
    for player in players:
        name = str(player.get("name") or "").lower()
        player["watched"] = bool(name) and any(w in name for w in wanted)


# ── library rows ─────────────────────────────────────────────────────────────


@bp.get("/api/library/list")
def api_list():
    statuses = [s for s in (request.args.get("status") or "").split(",") if s]
    sources = [s for s in (request.args.get("source") or "").split(",") if s]
    result = db.list_demos(
        q=(request.args.get("q") or "").strip() or None,
        statuses=statuses,
        sources=sources,
        map_name=(request.args.get("map") or "").strip() or None,
        player=(request.args.get("player") or "").strip() or None,
        steamid=(request.args.get("steamid") or "").strip() or None,
        watched_only=request.args.get("watched") == "1",
        sort=request.args.get("sort") or "library",
        limit=_int_arg("limit", 60),
        offset=_int_arg("offset", 0),
    )
    result["avatars"] = _avatars_for(_roster_ids(result["items"]))
    return jsonify(result)


@bp.get("/api/library/stats")
def api_stats():
    return jsonify(db.stats())


@bp.get("/api/library/demo/<int:demo_id>")
def api_demo(demo_id: int):
    row = db.get_demo(demo_id)
    if row is None:
        return _bad("demo 不存在", 404)
    row["thumb_url"] = f"/api/library/thumb/{demo_id}" if row.get("thumb_path") else None
    row["csz_exists"] = bool(row.get("csz_path")) and Path(row["csz_path"]).is_file()
    return jsonify(row)


#: Fields a user may edit by hand.  Anything else in a PATCH body is ignored.
_EDITABLE = {
    "display_name": 512,
    "remark": 4000,
    "source": 64,
    "team_a_name": 128,
    "team_b_name": 128,
    "match_date": 32,
}


@bp.patch("/api/library/demo/<int:demo_id>")
def api_update(demo_id: int):
    row = db.get_demo(demo_id)
    if row is None:
        return _bad("demo 不存在", 404)
    body = _body()
    updates: dict[str, Any] = {}
    for field, limit in _EDITABLE.items():
        if field not in body:
            continue
        value = body[field]
        if value is None:
            updates[field] = None
            continue
        text = str(value).strip()
        if len(text) > limit:
            return _bad(f"{field} 超长（最多 {limit} 字）")
        updates[field] = text or None

    # Swapping the two score columns is a common correction, so accept it here
    # rather than making the UI do two round trips.
    if body.get("swap_teams"):
        updates["team_a_name"] = row.get("team_b_name")
        updates["team_b_name"] = row.get("team_a_name")
        updates["team_a_score"] = row.get("team_b_score")
        updates["team_b_score"] = row.get("team_a_score")

    if not updates:
        return _bad("没有可更新的字段")
    db.update_demo(demo_id, **updates)
    return jsonify(db.get_demo(demo_id))


@bp.delete("/api/library/demo/<int:demo_id>")
def api_delete(demo_id: int):
    """Remove a library row.

    ``?delete_file=1`` also moves the original ``.dem`` to the recycle bin
    rather than deleting it outright -- a mis-click should not destroy a 400 MB
    demo the user may not be able to download again.

    The row's ``.csz`` and thumbnail always go with it: nothing else can reach
    them once the row is gone (scanning only consults the database), so leaving
    them behind would just strand files on disk.  ``csz_removed`` is False when
    another row still points at the same ``.csz``.
    """
    row = db.get_demo(demo_id)
    if row is None:
        return _bad("demo 不存在", 404)
    delete_file = request.args.get("delete_file") == "1"

    moved = None
    if delete_file:
        moved = _recycle(row["path"])
        if moved is None:
            return _bad("源文件被占用，无法移动。记录未删除", 409)

    removed = db.delete_demo(demo_id)
    if removed is None:
        return _bad("删除失败", 500)
    csz_removed = ingest.delete_csz_if_unreferenced(removed.get("csz_path"))

    thumb_removed = False
    thumb = removed.get("thumb_path")
    if thumb:
        thumb_file = Path(thumb)
        thumb_removed = thumb_file.is_file()
        try:
            thumb_file.unlink(missing_ok=True)
        except OSError:
            thumb_removed = False

    return jsonify({
        "deleted": demo_id,
        "file_moved_to": moved,
        # _recycle() returns the original path when the file was already gone,
        # so only a different path means something actually moved.
        "recycled": moved is not None and moved != row["path"],
        "csz_removed": csz_removed,
        "thumb_removed": thumb_removed,
    })


def _recycle(path: str) -> Optional[str]:
    """Move a file into ``app/data/recycle/``. Returns the new path, or None."""
    src = Path(path)
    if not src.is_file():
        return str(src)  # already gone; treat as success
    dest_dir = config.DATA_DIR / "recycle"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            dest = dest_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
        src.replace(dest)
        return str(dest)
    except OSError:
        return None


@bp.get("/api/library/thumb/<int:demo_id>")
def api_thumb(demo_id: int):
    row = db.get_demo(demo_id)
    if row is None:
        return _bad("demo 不存在", 404)
    path = row.get("thumb_path")
    if request.args.get("rebuild") == "1" or not path or not Path(path).is_file():
        job = jobs.manager.submit(
            "thumb",
            f"重建缩略图 #{demo_id}",
            lambda j, i=demo_id: ingest.refresh_metadata(i, j),
        )
        return _bad(f"缩略图不存在，已提交重建任务 {job.id}", 404)
    resp = send_file(path, mimetype="image/png", conditional=True)
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp


# ── players ──────────────────────────────────────────────────────────────────
#
# A player page is three reads: the totals, the per-map split, and the match
# list -- the last of which is just the library query with an exact roster
# filter, so a profile can reuse every filter the library already has.


@bp.get("/api/library/players")
def api_players():
    cfg = config.Config()
    result = db.list_players(
        q=(request.args.get("q") or "").strip() or None,
        watched_names=(cfg.get("watched_players") or [])
        if request.args.get("watched") == "1" else None,
        sort=request.args.get("sort") or "matches",
        limit=_int_arg("limit", 60),
        offset=_int_arg("offset", 0),
    )
    _mark_watched(result["items"])
    result["avatars"] = _avatars_for([p["steamid"] for p in result["items"]])
    result["watched_players"] = cfg.get("watched_players") or []
    result["avatar_stats"] = db.avatar_stats()
    return jsonify(result)


@bp.get("/api/library/players/search")
def api_players_search():
    """Ranked candidates for the player search box.

    Separate from ``/api/library/players?q=`` on purpose: that one filters the
    grid and has to stay consistent with its own total, this one only has to
    put the right few names at the top.
    """
    q = (request.args.get("q") or "").strip()
    limit = max(1, min(_int_arg("limit", 8), 30))
    items = db.search_players(q, limit=limit)
    _mark_watched(items)
    return jsonify({
        "q": q,
        "items": items,
        "avatars": _avatars_for([p["steamid"] for p in items]),
    })


@bp.get("/api/library/players/<steamid>")
def api_player(steamid: str):
    row = db.get_player(steamid)
    if row is None:
        return _bad("没有这个玩家的记录", 404)
    _mark_watched([row])
    return jsonify({
        "player": row,
        "maps": db.player_maps(steamid),
        "name_history": db.player_names(steamid),
        "avatar": _avatar_view(db.get_avatar(steamid)),
        "demos": db.list_demos(steamid=steamid, limit=1)["total"],
    })


@bp.patch("/api/library/players/<steamid>")
def api_player_update(steamid: str):
    """Add or remove a player from the watched list.

    The watched list is a list of *names*, not SteamIDs -- it is also used to
    highlight players who only appear in a scoreboard -- so this writes the
    player's current in-game name and reports back whether that actually made
    them watched.  Removing only drops an exact name match: a partial entry the
    user typed by hand ("Zyw") still matches, and the response says so rather
    than pretending the player is gone.
    """
    row = db.get_player(steamid)
    if row is None:
        return _bad("没有这个玩家的记录", 404)
    body = _body()
    if "watched" not in body:
        return _bad("没有可更新的字段")
    name = str(row.get("name") or "").strip()
    if not name:
        return _bad("这个玩家没有可用的名字")

    cfg = config.Config()
    watched = [str(w).strip() for w in (cfg.get("watched_players") or []) if str(w).strip()]
    low = name.lower()
    if body["watched"]:
        if low not in {w.lower() for w in watched}:
            watched.append(name)
    else:
        watched = [w for w in watched if w.lower() != low]
    cfg.update({"watched_players": watched})
    watcher.watcher.poke()
    # The library cards read a per-row copy of the match, so following from a
    # profile has to refresh it too -- otherwise unfollowing here leaves the
    # old matches highlighted on the library page with nothing to explain it.
    resynced = ingest.resync_watched()

    fresh = [w.lower() for w in watched]
    return jsonify({
        "watched": any(w in low for w in fresh),
        "watched_players": watched,
        "watched_resynced": resynced,
    })


@bp.post("/api/library/avatars/sync")
def api_avatars_sync():
    """Queue a Steam avatar lookup.

    No ``steamids`` means every player in the library.  When a sync is already
    queued or running this returns ``queued: false`` and folds the request into
    a follow-up run, so a caller never has to poll and retry.
    """
    body = _body()
    raw = body.get("steamids")
    if raw is not None and not isinstance(raw, list):
        return _bad("steamids 必须是数组")
    clean = None
    if isinstance(raw, list) and raw:
        clean = [str(s).strip() for s in raw if str(s or "").strip()][:200]
    job_id = avatars.queue_sync(clean, force=bool(body.get("force")))
    return jsonify({
        "job_id": job_id,
        "queued": job_id is not None,
        "note": None if job_id else "已有同步任务在跑，完成后会再跑一轮",
    })


@bp.get("/api/library/avatar/<steamid>")
def api_avatar(steamid: str):
    path = avatars.local_avatar(steamid)
    if path is None:
        return _bad("没有这个玩家的头像", 404)
    resp = send_file(path, conditional=True)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


# ── extraction ───────────────────────────────────────────────────────────────


@bp.post("/api/library/extract")
def api_extract():
    """Queue extraction. ``{"force": true}`` re-parses instead of refreshing.

    Without ``force`` a row that already has a ``.csz`` only gets its metadata
    re-derived from that file -- cheap, but it ignores the sampling settings,
    so it cannot be used to re-extract after changing them.
    """
    body = _body()
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        return _bad("ids 必须是非空数组")
    clean: list[int] = []
    for raw in ids[:200]:
        try:
            clean.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not clean:
        return _bad("没有有效的 demo id")
    already: list[int] = []
    job_ids = ingest.queue_extraction(clean, force=bool(body.get("force")),
                                      skipped=already)
    if not job_ids:
        if already:
            # Not an error: the very same click twice, or a scan that already
            # picked this row up.  Saying "queued 0" with a reason is more
            # useful than a 400 that reads like the demo does not exist.
            return jsonify({
                "queued": 0, "job_ids": [], "already_queued": len(already),
                "note": "已经在队列里了，等它跑完",
            })
        # Every id was unknown.  Silently returning "queued: 0" makes a scripted
        # caller think it worked, and the UI would print "已排队 0 个".
        return _bad("没有可提取的 Demo", 400)
    return jsonify({"queued": len(job_ids), "job_ids": job_ids,
                    "already_queued": len(already)})


@bp.post("/api/library/demo/<int:demo_id>/refresh")
def api_refresh(demo_id: int):
    if db.get_demo(demo_id) is None:
        return _bad("demo 不存在", 404)
    job = jobs.manager.submit(
        "refresh", f"刷新 #{demo_id}", lambda j: ingest.refresh_metadata(demo_id, j)
    )
    return jsonify({"job_id": job.id})


# ── open in the replay page ──────────────────────────────────────────────────


@bp.post("/api/library/open")
def api_open():
    """Turn a selection into a replay batch.

    Missing ``.csz`` files are queued for extraction, so the replay page can
    show the whole selection immediately and enable each entry as it lands.
    """
    body = _body()
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        return _bad("ids 必须是非空数组")

    clean: list[int] = []
    for raw in ids[:100]:
        try:
            clean.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not clean:
        return _bad("没有有效的 demo id")

    rows = [(i, db.get_demo(i)) for i in clean]
    rows = [(i, r) for i, r in rows if r is not None]
    if not rows:
        # Otherwise this builds an empty batch and sends the browser to a replay
        # page with nothing in it, which looks like the app is broken.
        return _bad("没有找到这些 Demo", 400)

    _prune_batches()
    batch_id = uuid.uuid4().hex[:12]
    with _batch_lock:
        _batches[batch_id] = {
            "created": time.time(),
            "items": [{"demo_id": i} for i, _ in rows],
        }

    missing = [
        i for i, row in rows
        if not row.get("csz_path") or not Path(row["csz_path"]).is_file()
    ]
    already: list[int] = []
    job_ids = ingest.queue_extraction(missing, skipped=already) if missing else []

    view = _batch_view(batch_id) or {"batch_id": batch_id, "items": []}
    return jsonify({
        **view,
        "queued": len(job_ids),
        "job_ids": job_ids,
        "already_queued": len(already),
        "replay_url": f"/replay?batch={batch_id}",
    })


@bp.get("/api/library/batch/<batch_id>")
def api_batch(batch_id: str):
    view = _batch_view(batch_id)
    if view is None:
        return _bad("批次不存在或已过期", 404)
    return jsonify(view)


# ── watch roots ──────────────────────────────────────────────────────────────


@bp.get("/api/library/watch")
def api_watch_status():
    return jsonify(watcher.watcher.status())


@bp.put("/api/library/watch")
def api_watch_update():
    body = _body()
    cfg = config.Config()
    patch: dict[str, Any] = {}

    if "roots" in body:
        roots = body["roots"]
        if not isinstance(roots, list):
            return _bad("roots 必须是数组")
        cleaned = []
        for raw in roots[:50]:
            if not isinstance(raw, dict) or not raw.get("path"):
                continue
            path = config.expand(str(raw["path"]))
            # Deliberately *not* required to exist.  A demo directory on an
            # external or network drive that is not mounted right now is a
            # normal thing to configure; the watcher simply skips it until it
            # reappears.  The UI probes the path first and tells the user how
            # many demos it found, so a typo is caught there.
            cleaned.append({
                "path": path,
                "depth": max(0, min(int(raw.get("depth", 2) or 0), 32)),
                "source": raw.get("source") or config.infer_source(path),
                "enabled": bool(raw.get("enabled", True)),
                "label": str(raw.get("label") or "")[:64],
            })
        patch["roots"] = None
        patch["watch_roots"] = cleaned

    for key, caster in (
        ("enabled", lambda v: bool(v)),
        ("interval", lambda v: max(1.0, min(float(v), 300.0))),
    ):
        if key in body:
            try:
                patch["watch_enabled" if key == "enabled" else "watch_interval_sec"] = caster(body[key])
            except (TypeError, ValueError):
                return _bad(f"{key} 取值无效")
    patch.pop("roots", None)

    if not patch:
        return _bad("没有可更新的字段")
    cfg.update(patch)
    watcher.watcher.restart()
    return jsonify(watcher.watcher.status())


@bp.post("/api/library/watch/inspect")
def api_watch_inspect():
    body = _body()
    path = body.get("path")
    if not path:
        return _bad("缺少 path")
    return jsonify(config.probe_path(str(path)))


@bp.get("/api/library/watch/suggest")
def api_watch_suggest():
    return jsonify({"suggestions": config.suggested_paths()})


@bp.post("/api/library/watch/scan")
def api_watch_scan():
    """Full manual scan. Skips the stability gate -- the user is asserting the
    files are complete -- and reports progress through the job queue.

    Whether the newly found demos are extracted follows the ``auto_extract``
    setting, so a manual scan and the background watcher behave identically.
    Pass ``extract`` explicitly to override it for one call.
    """
    body = _body()
    force = bool(body.get("force", True))
    cfg = config.Config()
    extract = body.get("extract")
    if extract is None:
        extract = bool(cfg.get("auto_extract", True))
    extract = bool(extract)

    def run(job: jobs.Job):
        if force:
            # Register straight from a directory walk rather than through
            # poll(), so newly-dropped files are picked up on the first click.
            summary = ingest.scan_roots(
                progress=lambda msg, pct=None: job.set_step(msg, pct)
            )
        else:
            summary = watcher.watcher.poll(force=False)
        summary["extracted"] = extract
        new_ids = summary.get("new_ids") or []
        if new_ids and extract:
            # `new_ids` is not just "found this time": scan_roots also folds in
            # every row still without a .csz (work orphaned by a restart), which
            # includes the row the watcher queued a second ago.  queue_extraction
            # skips those, so report what actually got queued rather than
            # promising one job per id.
            already: list[int] = []
            job_ids = ingest.queue_extraction(new_ids, skipped=already)
            summary["job_ids"] = job_ids
            summary["already_queued"] = len(already)
            if job_ids:
                job.set_step(f"排队提取 {len(job_ids)} 个 demo…", 0.95)
            else:
                job.set_step("这些 Demo 已经在队列里了", 1.0)
        elif new_ids:
            job.set_step(f"已入库 {len(new_ids)} 个 demo（自动提取已关闭）", 1.0)
        return summary

    job = jobs.manager.submit("scan", "扫描监听目录", run)
    return jsonify({"job_id": job.id})


# ── storage ──────────────────────────────────────────────────────────────────


@bp.get("/api/library/storage")
def api_storage():
    return jsonify(storage.overview())


@bp.post("/api/library/storage/clean")
def api_storage_clean():
    """Wipe one or more caches.

    Refused while anything is running: an extraction writes its ``.csz`` at the
    end of a parse that can take half a minute, and deleting the directory
    underneath it would either fail the job or leave a file nobody points at.
    """
    body = _body()
    targets = body.get("targets")
    if not isinstance(targets, list) or not targets:
        return _bad("targets 必须是非空数组")
    if jobs.manager.has_active():
        return _bad("有后台任务在跑，等它结束再清理", 409)
    try:
        result = storage.clean(targets)
    except ValueError as exc:
        return _bad(str(exc))
    return jsonify(result)


# ── config + jobs ────────────────────────────────────────────────────────────


@bp.get("/api/library/config")
def api_config_get():
    cfg = config.Config()
    data = cfg.all()
    data["sources"] = config.SOURCE_ORDER
    data["source_labels"] = config.SOURCE_LABELS
    data["sampling_label"] = cfg.sampling_label()
    data["maps_available"] = radar.available_maps()
    data["avatars"] = db.avatar_stats()
    return jsonify(data)


@bp.put("/api/library/config")
def api_config_put():
    body = _body()
    allowed = {
        "watched_players", "sampling", "auto_extract",
        "make_thumbnails", "watch_enabled", "watch_interval_sec",
        "fetch_avatars", "avatar_ttl_days",
    }
    patch = {k: v for k, v in body.items() if k in allowed}
    if "watched_players" in patch:
        raw = patch["watched_players"]
        if not isinstance(raw, list):
            return _bad("watched_players 必须是数组")
        patch["watched_players"] = [
            str(x).strip() for x in raw[:100] if str(x).strip()
        ]
    if "sampling" in patch:
        s = patch["sampling"]
        if not isinstance(s, dict):
            return _bad("sampling 必须是对象")
        mode = s.get("mode", "tick")
        if mode not in ("tick", "interval", "stride"):
            return _bad("sampling.mode 只能是 tick / interval / stride")
    if "avatar_ttl_days" in patch:
        try:
            patch["avatar_ttl_days"] = max(1, min(int(patch["avatar_ttl_days"]), 365))
        except (TypeError, ValueError):
            return _bad("avatar_ttl_days 取值无效")
    if not patch:
        return _bad("没有可更新的字段")
    cfg = config.Config()
    cfg.update(patch)
    watcher.watcher.poke()
    data = cfg.all()
    if "watched_players" in patch:
        # Every existing row holds a copy of the old match, and the cards read
        # that copy.  Re-derive it here so the highlight follows the list
        # instead of waiting for a per-demo refresh.
        data["watched_resynced"] = ingest.resync_watched()
    return jsonify(data)


@bp.get("/api/library/jobs")
def api_jobs():
    return jsonify({
        "jobs": jobs.manager.list(active_only=request.args.get("active") == "1"),
        "pending": jobs.manager.pending_count(),
    })


@bp.get("/api/library/jobs/<job_id>")
def api_job(job_id: str):
    job = jobs.manager.get(job_id)
    if job is None:
        return _bad("任务不存在", 404)
    return jsonify(job.to_dict())


@bp.post("/api/library/jobs/clear")
def api_jobs_clear():
    return jsonify({"cleared": jobs.manager.clear_finished()})


# ── 地图几何：没解压就解压 ───────────────────────────────────────────────────
#
# `maps/optimized_obj_files/` 里进 git 的是 `.obj.zip`（8 个，约 207 MB），解压
# 出来是约 943 MB 的 `.obj`。也就是说干净检出之后这个目录里只有 zip：3D 回放
# 会退化成一块空地加网格，雷达缩略图也画不出来。
#
# 它和"AI 权重还没下载"是同一类事：仓库里缺的东西跑一次补上。所以两个接口
# 照抄 `/api/ai/models*` 的形状：先查本地（`inspect()` 只 stat，不联网也不解压），
# 缺了才让页面出按钮，点完把活儿交给 jobs 队列。
#
# 解压本身在 `scripts/extract_maps.py` 里 —— 命令行 `python scripts/extract_maps.py`
# 走的是同一个函数，网页不重复实现"怎么写 .part、怎么原子改名"。

#: 解压地图的任务类型（顶栏「任务」抽屉里看到的也是它）。
MAPS_JOB_KIND = "maps"


@bp.get("/api/library/maps")
def api_maps():
    """地图几何解压了没（只碰磁盘，不进 zip 解压）。

    横幅的显隐就靠这次查询，所以响应带 `no-store`：浏览器缓存住一个旧状态，
    用户会看到一个点了没反应的按钮。
    """
    from scripts.extract_maps import inspect

    state = inspect(config.MAPS_DIR)
    # 正在解压 / 刚解压完的那个任务一起给页面，横幅直接用它的 step + progress，
    # 不必再为"解压进度"造第二套接口。
    state["extract"] = next(
        (job for job in jobs.manager.list() if job.get("kind") == MAPS_JOB_KIND), None
    )
    resp = jsonify(state)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.post("/api/library/maps/extract")
def api_maps_extract():
    """把 `maps/*.obj.zip` 解压出来，跑在后台任务里。

    943 MB 的解压不能同步做：浏览器的 fetch 会先超时，中间也看不到进度。所以
    POST 立刻返回 `job_id`，页面去 `/api/library/jobs/<id>` 读 step / progress。

    都解压好了就不排队（除非 `?force=1`）；已经有一个在跑就把它还回去 ——
    连点两下按钮不该把同一批文件解两遍。
    """
    from scripts.extract_maps import extract, inspect

    force = request.args.get("force") in ("1", "true", "yes")
    root = config.MAPS_DIR
    if inspect(root)["ok"] and not force:
        return jsonify({"ok": True, "job_id": None, "already": True})

    for job in jobs.manager.list(active_only=True):
        if job.get("kind") == MAPS_JOB_KIND:
            return jsonify({"ok": True, "job_id": job["id"], "already": True})

    def run(job):
        # `job.set_step(一句话, 0..1)` 是 jobs 队列的进度 API；解压脚本通过
        # `progress` 回调把真实字节进度报上来。
        return extract(
            root=root,
            force=force,
            progress=lambda frac, msg: job.set_step(msg, frac),
        )

    job = jobs.manager.submit(MAPS_JOB_KIND, "解压地图几何", run)
    return jsonify({"ok": True, "job_id": job.id})
