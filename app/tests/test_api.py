#!/usr/bin/env python3
"""
HTTP-level checks for ``/api/library``.

Everything here runs against a **scratch data directory**, so it needs no demo
file and never touches the real library.  It covers the surface that the
end-to-end run cannot cheaply reach: settings round-trips, directory
validation, the recycle path, and the error responses.

Run:  python app/tests/test_api.py
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from app import config  # noqa: E402

_passed = 0
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  \u2713 {name}")
    else:
        _failed.append(name)
        print(f"  \u2717 {name}" + (f"\n      {detail}" if detail else ""))


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r}, want {want!r}")


@contextlib.contextmanager
def scratch():
    base = _ROOT / ".build" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def scratch_app():
    """Redirect every data path into a scratch dir, then build the app.

    ``config.Config`` resolves ``CONFIG_PATH`` at call time, and ``db``/``ingest``
    read ``config.*`` at call time too, so patching the module attributes is
    enough to keep the real library untouched.
    """
    saved = {k: getattr(config, k) for k in
             ("DATA_DIR", "CONFIG_PATH", "LIBRARY_DB", "CSZ_DIR", "THUMB_DIR",
              "AVATAR_DIR")}
    with scratch() as tmp:
        config.DATA_DIR = tmp
        config.CONFIG_PATH = tmp / "config.json"
        config.LIBRARY_DB = tmp / "library.db"
        config.CSZ_DIR = tmp / "csz"
        config.THUMB_DIR = tmp / "thumbs"
        config.AVATAR_DIR = tmp / "avatars"
        for d in (config.CSZ_DIR, config.THUMB_DIR, config.AVATAR_DIR):
            d.mkdir(parents=True, exist_ok=True)
        try:
            # Imported late so the patched paths are what the modules see.
            from app import jobs, server, watcher
            app = server.create_app(start_background=False)
            app.config["TESTING"] = True
            # The scan endpoint runs through the job queue, so the queue has to
            # actually work here -- create_app(start_background=False) leaves it
            # stopped on purpose.
            jobs.manager.start()
            try:
                yield app.test_client(), tmp
            finally:
                jobs.manager.stop()
                watcher.watcher.stop()
        finally:
            for k, v in saved.items():
                setattr(config, k, v)


def seed(client, tmp: Path, name: str = "a.dem", **fields):
    """Put a row in the library without extracting anything."""
    from app import db
    path = tmp / name
    if not path.exists():
        path.write_bytes(b"fake demo bytes")
    payload = {"filename": name, "file_size": path.stat().st_size,
               "status": db.STATUS_READY, "source": "5E", "map_name": "de_mirage",
               "team_a_name": "Team A", "team_b_name": "Team B",
               "team_a_score": 13, "team_b_score": 7,
               "players_json": json.dumps([{"name": "ZywOo", "team": 0}])}
    payload.update(fields)
    return db.upsert_demo(str(path), **payload)


#: Two real SteamID64s, so the avatar code paths get ids they accept.
PLAYER_A = "76561198113666193"
PLAYER_B = "76561199074863582"

#: The smallest thing the avatar sniffer accepts as a JPEG.
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


@contextlib.contextmanager
def stub_network():
    """Keep the avatar sync from reaching Steam during the API tests.

    The sync endpoint really does run on the job queue here, so without this a
    test would depend on steamcommunity.com being reachable -- and on how fast
    it answers.
    """
    from app import avatars

    def fake(url, *, accept="application/json", timeout=10.0, limit=1 << 22):
        if "miniprofile" in url:
            return json.dumps({
                "persona_name": "Test Persona",
                "avatar_url": "https://avatars.fastly.steamstatic.com/t_full.jpg",
            }).encode("utf-8")
        return JPEG

    original = avatars._http_get
    avatars._http_get = fake
    try:
        yield
    finally:
        avatars._http_get = original


# ── checks ───────────────────────────────────────────────────────────────────


def test_basics(c) -> None:
    print("\nbasics")
    r = c.get("/healthz")
    eq("healthz", r.status_code, 200)
    eq("healthz reports zero demos", r.get_json()["demos"], 0)

    eq("library page", c.get("/library").status_code, 200)
    eq("replay page", c.get("/replay").status_code, 200)
    eq("root redirects to the library", c.get("/").status_code, 302)
    check("redirect target", "/library" in (c.get("/").headers.get("Location") or ""))

    html = c.get("/library").get_data(as_text=True)
    for needle in ('id="content"', 'id="settings-modal"', 'id="activity-drawer"',
                   'id="selection-bar"', "/lib-static/js/library.js",
                   "/lib-static/css/library.css"):
        check(f"library.html contains {needle}", needle in html)

    # The batch selector belongs to the replay page, not the library page.
    replay = c.get("/replay").get_data(as_text=True)
    check("replay page has the batch selector", 'id="batch-select"' in replay)
    check("replay page links back to the library", 'href="/library"' in replay)

    check("library css served", c.get("/lib-static/css/library.css").status_code == 200)
    check("library js served", c.get("/lib-static/js/library.js").status_code == 200)
    check("visualizer css still served", c.get("/static/css/style.css").status_code == 200)
    check("visualizer api still mounted", c.get("/api/maps").status_code in (200, 404, 500))


def test_demo_crud(c, tmp: Path) -> None:
    print("\ndemo CRUD")
    did = seed(c, tmp, "a.dem", display_name="决赛图三", remark="很精彩")
    seed(c, tmp, "b.dem", map_name="de_inferno", source="FACEIT")

    row = c.get(f"/api/library/demo/{did}").get_json()
    eq("detail display name", row["display_name"], "决赛图三")
    eq("detail players decoded", row["players"][0]["name"], "ZywOo")
    eq("thumb_url is null without a thumbnail", row["thumb_url"], None)
    eq("csz_exists false", row["csz_exists"], False)

    eq("404 for a missing demo", c.get("/api/library/demo/99999").status_code, 404)
    eq("404 patch", c.patch("/api/library/demo/99999", json={"remark": "x"}).status_code, 404)
    eq("404 delete", c.delete("/api/library/demo/99999").status_code, 404)

    print("\npatch")
    r = c.patch(f"/api/library/demo/{did}", json={"remark": "改了"})
    eq("patch remark", r.get_json()["remark"], "改了")
    r = c.patch(f"/api/library/demo/{did}", json={"remark": "   "})
    eq("whitespace clears the field", r.get_json()["remark"], None)
    r = c.patch(f"/api/library/demo/{did}", json={"display_name": "x" * 600})
    eq("over-long value rejected", r.status_code, 400)
    check("rejection explains which field", "display_name" in r.get_json()["error"])
    eq("unknown field is ignored, not fatal",
       c.patch(f"/api/library/demo/{did}", json={"path": r"D:\evil.dem"}).status_code, 400)
    eq("path was not changed",
       c.get(f"/api/library/demo/{did}").get_json()["path"].endswith("a.dem"), True)

    print("\nswap teams")
    r = c.patch(f"/api/library/demo/{did}", json={"swap_teams": True}).get_json()
    eq("names swapped", (r["team_a_name"], r["team_b_name"]), ("Team B", "Team A"))
    eq("scores swapped", (r["team_a_score"], r["team_b_score"]), (7, 13))
    r = c.patch(f"/api/library/demo/{did}", json={"swap_teams": True}).get_json()
    eq("swapping twice restores", (r["team_a_score"], r["team_b_score"]), (13, 7))

    print("\nthumbnail")
    eq("no thumbnail -> 404 + rebuild queued",
       c.get(f"/api/library/thumb/{did}").status_code, 404)
    eq("thumb 404 for missing demo", c.get("/api/library/thumb/99999").status_code, 404)

    print("\ndelete")
    r = c.delete(f"/api/library/demo/{did}")
    eq("delete ok", r.status_code, 200)
    eq("nothing recycled (delete_file not set)", r.get_json()["file_moved_to"], None)
    check("source file kept", (tmp / "a.dem").is_file())
    eq("library now has 1", c.get("/api/library/stats").get_json()["total"], 1)

    # delete_file=1 moves the .dem into app/data/recycle/
    did2 = seed(c, tmp, "c.dem")
    r = c.delete(f"/api/library/demo/{did2}?delete_file=1")
    moved = Path(r.get_json()["file_moved_to"])
    check("file moved to recycle", "recycle" in str(moved) and moved.is_file(), str(moved))
    check("original path is gone", not (tmp / "c.dem").is_file())


def test_list_and_filters(c, tmp: Path) -> None:
    print("\nlist / filters / pagination")
    seed(c, tmp, "m1.dem", map_name="de_anubis", source="StarLadder",
         team_a_name="Team Vitality", team_b_name="Aurora Gaming", file_size=100)
    seed(c, tmp, "m2.dem", map_name="de_inferno", source="StarLadder",
         team_a_name="Team Vitality", team_b_name="Aurora Gaming", file_size=300)
    seed(c, tmp, "m3.dem", map_name="de_mirage", source="FACEIT",
         team_a_name="Natus Vincere", team_b_name="FaZe", file_size=200)

    j = c.get("/api/library/list").get_json()
    eq("total", j["total"], 3)
    eq("default sort is by newest first", len(j["items"]), 3)
    eq("sort=size desc", [i["file_size"] for i in
                          c.get("/api/library/list?sort=size").get_json()["items"]],
       [300, 200, 100])
    eq("sort=size ascending is not offered",
       c.get("/api/library/list?sort=size").get_json()["sort"], "size")

    eq("q by player", c.get("/api/library/list?q=ZywOo").get_json()["total"], 3)
    eq("q by team", c.get("/api/library/list?q=NaVi").get_json()["total"], 0)
    eq("q by team substring", c.get("/api/library/list?q=Natus").get_json()["total"], 1)
    eq("q by map", c.get("/api/library/list?q=inferno").get_json()["total"], 1)
    eq("q by filename", c.get("/api/library/list?q=m2").get_json()["total"], 1)
    eq("q miss", c.get("/api/library/list?q=zzz").get_json()["total"], 0)

    eq("filter source", c.get("/api/library/list?source=FACEIT").get_json()["total"], 1)
    eq("filter two sources",
       c.get("/api/library/list?source=FACEIT,StarLadder").get_json()["total"], 3)
    eq("filter map", c.get("/api/library/list?map=de_anubis").get_json()["total"], 1)
    eq("filter status ready", c.get("/api/library/list?status=ready").get_json()["total"], 3)
    eq("filter status error", c.get("/api/library/list?status=error").get_json()["total"], 0)
    eq("unknown status ignored",
       c.get("/api/library/list?status=bogus").get_json()["total"], 3)

    eq("pagination limit", len(c.get("/api/library/list?limit=2").get_json()["items"]), 2)
    eq("pagination offset",
       c.get("/api/library/list?limit=2&offset=2").get_json()["items"][0]["filename"], "m1.dem")
    eq("absurd limit clamped", c.get("/api/library/list?limit=99999").get_json()["limit"], 1000)
    eq("negative offset clamped", c.get("/api/library/list?offset=-5").get_json()["offset"], 0)
    eq("garbage limit falls back", c.get("/api/library/list?limit=abc").get_json()["limit"], 60)
    eq("unknown sort falls back", c.get("/api/library/list?sort=nope").get_json()["sort"], "library")

    eq("stats total", c.get("/api/library/stats").get_json()["total"], 3)
    eq("stats maps", sorted(c.get("/api/library/stats").get_json()["maps"]),
       ["de_anubis", "de_inferno", "de_mirage"])


def test_watch(c, tmp: Path) -> None:
    print("\nwatch directories")
    root = tmp / "demos"
    root.mkdir()
    (root / "x.dem").write_bytes(b"x")
    (root / "y.dem").write_bytes(b"x")
    (root / "z.zip").write_bytes(b"x")

    print("  inspect")
    j = c.post("/api/library/watch/inspect", json={"path": str(root)}).get_json()
    check("valid", j["valid"])
    eq("demo count", j["demo_count"], 2)
    eq("zip count", j["zip_count"], 1)
    j = c.post("/api/library/watch/inspect", json={"path": str(tmp / "nope")}).get_json()
    check("missing dir invalid", not j["valid"])
    r = c.post("/api/library/watch/inspect", json={"path": ""})
    eq("empty path rejected", r.status_code, 400)
    j = c.post("/api/library/watch/inspect", json={"path": str(root / "x.dem")}).get_json()
    check("a file is not a directory", not j["valid"])

    print("  put / get round trip")
    r = c.put("/api/library/watch", json={
        "roots": [{"path": str(root), "depth": 1, "source": "5E", "enabled": True}],
        "enabled": False, "interval": 30,
    })
    eq("put ok", r.status_code, 200)
    j = c.get("/api/library/watch").get_json()
    eq("root persisted", j["roots"][0]["path"], str(root))
    eq("depth persisted", j["roots"][0]["depth"], 1)
    eq("enabled persisted", j["enabled"], False)
    eq("interval persisted", j["interval"], 30.0)
    check("disabled watcher reports inactive", j["active"] is False)

    print("  validation")
    # A path that does not exist right now is still accepted on purpose: a demo
    # directory on an unmounted external drive is a normal thing to configure.
    r = c.put("/api/library/watch", json={"roots": [{"path": str(tmp / "nope")}]})
    eq("missing root accepted", r.status_code, 200)
    eq("missing root stored for later", len(r.get_json()["roots"]), 1)
    eq("empty root rejected or pruned",
       len(c.put("/api/library/watch", json={"roots": [{"path": ""}]}).get_json()["roots"]), 0)
    r = c.put("/api/library/watch", json={"roots": "not a list"})
    eq("non-list roots rejected", r.status_code, 400)
    eq("interval clamped up",
       c.put("/api/library/watch", json={"interval": 0}).get_json()["interval"], 1.0)
    eq("interval clamped down",
       c.put("/api/library/watch", json={"interval": 99999}).get_json()["interval"], 300.0)

    print("  suggestions")
    j = c.get("/api/library/watch/suggest").get_json()
    check("suggest returns a list", isinstance(j["suggestions"], list))
    for s in j["suggestions"]:
        check(f"suggestion {s['path']} has a source", bool(s.get("source")))

    print("  scan with extraction disabled")
    c.put("/api/library/watch", json={
        "roots": [{"path": str(root), "depth": 1, "source": "5E", "enabled": True}],
        "enabled": True, "interval": 5,
    })
    c.put("/api/library/config", json={"auto_extract": False})
    jid = c.post("/api/library/watch/scan", json={"force": True}).get_json()["job_id"]
    import time
    for _ in range(60):
        st = c.get(f"/api/library/jobs/{jid}").get_json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.25)
    eq("scan job finished", st["state"], "done")
    res = st["result"]
    eq("found both demos", res["scanned"], 2)
    eq("registered both", res["registered"], 2)
    eq("extraction skipped", res.get("extracted"), False)
    eq("no extract jobs queued", "job_ids" in res, False)
    eq("library now has 2", c.get("/api/library/stats").get_json()["total"], 2)
    check("enabled watcher with roots reports active",
          c.get("/api/library/watch").get_json()["active"] is True)

    print("  rescan is idempotent")
    jid = c.post("/api/library/watch/scan", json={"force": True}).get_json()["job_id"]
    for _ in range(60):
        st = c.get(f"/api/library/jobs/{jid}").get_json()
        if st["state"] in ("done", "error"):
            break
        time.sleep(0.25)
    eq("nothing new on rescan", st["result"]["registered"], 0)
    eq("still 2 rows", c.get("/api/library/stats").get_json()["total"], 2)


def test_config(c) -> None:
    print("\nconfig")
    j = c.get("/api/library/config").get_json()
    check("sources listed", "5E" in j["sources"] and "FACEIT" in j["sources"])
    eq("source labels present", j["source_labels"].get("5E"), "5E 对战平台")
    check("maps available", isinstance(j["maps_available"], list))
    check("sampling label", isinstance(j["sampling_label"], str) and j["sampling_label"])
    eq("default sampling mode", j["sampling"]["mode"], "tick")

    r = c.put("/api/library/config", json={
        "watched_players": ["ZywOo", " donk ", "", "m0NESY"],
        "sampling": {"mode": "stride", "stride": 16, "interval": 0.25},
        "auto_extract": False, "make_thumbnails": False,
    })
    eq("put ok", r.status_code, 200)
    j = c.get("/api/library/config").get_json()
    eq("watched trimmed and de-blanked", j["watched_players"], ["ZywOo", "donk", "m0NESY"])
    eq("sampling mode", j["sampling"]["mode"], "stride")
    eq("sampling stride", j["sampling"]["stride"], 16)
    eq("auto_extract", j["auto_extract"], False)
    eq("make_thumbnails", j["make_thumbnails"], False)

    print("  validation")
    eq("invalid sampling mode rejected",
       c.put("/api/library/config", json={"sampling": {"mode": "nonsense"}}).status_code, 400)
    eq("mode unchanged after rejection",
       c.get("/api/library/config").get_json()["sampling"]["mode"], "stride")
    eq("non-object sampling rejected",
       c.put("/api/library/config", json={"sampling": "tick"}).status_code, 400)
    eq("non-array watched_players rejected",
       c.put("/api/library/config", json={"watched_players": "a,b"}).status_code, 400)
    eq("empty patch rejected", c.put("/api/library/config", json={}).status_code, 400)

    print("  unknown keys ignored")
    c.put("/api/library/config", json={"library_db": r"D:\evil.db", "watched_players": []})
    j = c.get("/api/library/config").get_json()
    eq("unknown key not stored", "library_db" in j, False)


def test_jobs_and_errors(c) -> None:
    print("\njobs + error paths")
    j = c.get("/api/library/jobs").get_json()
    check("jobs list", isinstance(j["jobs"], list))
    eq("404 for an unknown job", c.get("/api/library/jobs/nope").status_code, 404)
    eq("clear finished", c.post("/api/library/jobs/clear").status_code, 200)

    eq("extract with no ids", c.post("/api/library/extract", json={"ids": []}).status_code, 400)
    eq("extract with no body", c.post("/api/library/extract").status_code, 400)
    eq("extract with unknown ids",
       c.post("/api/library/extract", json={"ids": [99999]}).status_code, 400)
    eq("open with no ids", c.post("/api/library/open", json={"ids": []}).status_code, 400)
    eq("open with unknown ids",
       c.post("/api/library/open", json={"ids": [99999]}).status_code, 400)
    eq("refresh unknown demo",
       c.post("/api/library/demo/99999/refresh").status_code, 404)
    eq("batch unknown id", c.get("/api/library/batch/deadbeef").status_code, 404)

    print("  bad json body does not 500")
    r = c.post("/api/library/extract", data="not json",
               content_type="application/json")
    check("malformed body handled", r.status_code < 500, f"{r.status_code}")


def test_batch_handoff(c, tmp: Path) -> None:
    print("\nbatch handoff")
    from app import db
    ids = [seed(c, tmp, f"h{i}.dem", map_name=f"de_m{i}") for i in range(3)]
    # Give one of them a real .csz so `ready` has something to be true about.
    csz = tmp / "h0.csz"
    csz.write_bytes(b"CSZ5")
    db.update_demo(ids[0], csz_path=str(csz), status=db.STATUS_READY)

    r = c.post("/api/library/open", json={"ids": ids}).get_json()
    check("batch id returned", bool(r["batch_id"]))
    eq("replay url", r["replay_url"], f"/replay?batch={r['batch_id']}")
    eq("all three listed", len(r["items"]), 3)
    eq("queued the two without a .csz", r["queued"], 2)

    j = c.get(f"/api/library/batch/{r['batch_id']}").get_json()
    by_id = {i["demo_id"]: i for i in j["items"]}
    eq("first is ready", by_id[ids[0]]["ready"], True)
    eq("second is not", by_id[ids[1]]["ready"], False)
    eq("unready has no csz_path", by_id[ids[1]]["csz_path"], None)
    eq("ready has a csz_path", by_id[ids[0]]["csz_path"], str(csz))

    # A batch must survive the demo being renamed in the library.
    c.patch(f"/api/library/demo/{ids[0]}", json={"display_name": "重命名了"})
    j = c.get(f"/api/library/batch/{r['batch_id']}").get_json()
    eq("rename reflected", j["items"][0]["name"], "重命名了")

    # ...but not deletion.
    c.delete(f"/api/library/demo/{ids[0]}")
    j = c.get(f"/api/library/batch/{r['batch_id']}").get_json()
    eq("deleted demo drops out of the batch", len(j["items"]), 2)


def test_delete_semantics(c, tmp: Path) -> None:
    """The delete response has to say what actually happened on disk.

    The UI reports this back to the user verbatim, so a wrong flag here turns
    into a lie in a toast ("moved to the recycle bin" when nothing moved).
    """
    print("\ndelete: cached artefacts")
    csz = tmp / "shared__deadbeef.csz"
    csz.write_bytes(b"csz payload")
    thumb = tmp / "demo_1.png"
    thumb.write_bytes(b"\x89PNG fake")
    did = seed(c, tmp, "d.dem", csz_path=str(csz), thumb_path=str(thumb))

    r = c.delete(f"/api/library/demo/{did}").get_json()
    eq("csz removed", r["csz_removed"], True)
    check("csz file is gone", not csz.is_file())
    eq("thumb removed", r["thumb_removed"], True)
    check("thumb file is gone", not thumb.is_file())
    eq("nothing recycled without delete_file", r["recycled"], False)

    print("\ndelete: a .csz shared with another row survives")
    shared = tmp / "shared__cafebabe.csz"
    shared.write_bytes(b"csz payload")
    first = seed(c, tmp, "e1.dem", csz_path=str(shared))
    second = seed(c, tmp, "e2.dem", csz_path=str(shared))
    r = c.delete(f"/api/library/demo/{first}").get_json()
    eq("csz kept while another row points at it", r["csz_removed"], False)
    check("shared csz still on disk", shared.is_file())
    r = c.delete(f"/api/library/demo/{second}").get_json()
    eq("last referrer removes it", r["csz_removed"], True)
    check("shared csz gone once unreferenced", not shared.is_file())

    print("\ndelete: recycled only when a file actually moved")
    did = seed(c, tmp, "f.dem")
    r = c.delete(f"/api/library/demo/{did}?delete_file=1").get_json()
    eq("recycled flag set", r["recycled"], True)
    eq("file_moved_to points into recycle", "recycle" in r["file_moved_to"], True)

    # A row whose source vanished (status "missing") must still be deletable
    # with delete_file=1, and must not claim it recycled anything.
    from app import db
    did = seed(c, tmp, "g.dem")
    (tmp / "g.dem").unlink()
    db.update_demo(did, status=db.STATUS_MISSING)
    r = c.delete(f"/api/library/demo/{did}?delete_file=1")
    eq("missing source still deletes", r.status_code, 200)
    eq("missing source reports nothing recycled", r.get_json()["recycled"], False)

    print("\ndelete: a source that cannot be moved aborts the whole delete")
    did = seed(c, tmp, "h.dem")
    # Make the recycle *directory* impossible to create.  That reaches the same
    # OSError branch a file locked by another process takes, but deterministically
    # -- Windows file locking is not something a test can rely on.
    recycle = tmp / "recycle"
    shutil.rmtree(recycle, ignore_errors=True)
    recycle.write_bytes(b"not a directory")
    try:
        r = c.delete(f"/api/library/demo/{did}?delete_file=1")
        eq("409 when the source cannot be moved", r.status_code, 409)
        eq("row survives the failure", c.get(f"/api/library/demo/{did}").status_code, 200)
        check("source file left untouched", (tmp / "h.dem").is_file())
    finally:
        recycle.unlink(missing_ok=True)


def test_force_reextract(c, tmp: Path) -> None:
    """A ready row must be re-parseable on demand.

    Without ``force`` the endpoint refreshes metadata out of the existing .csz,
    which never looks at the sampling settings.  So after changing sampling, the
    plain extract button reported success while changing nothing at all.
    """
    print("\nextract vs re-parse")
    csz = tmp / "ready__deadbeef.csz"
    csz.write_bytes(b"csz payload")
    did = seed(c, tmp, "ready.dem", csz_path=str(csz))

    r = c.post("/api/library/extract", json={"ids": [did]}).get_json()
    eq("a ready row refreshes by default",
       c.get(f"/api/library/jobs/{r['job_ids'][0]}").get_json()["kind"], "refresh")

    r = c.post("/api/library/extract", json={"ids": [did], "force": True}).get_json()
    eq("force re-parses it",
       c.get(f"/api/library/jobs/{r['job_ids'][0]}").get_json()["kind"], "extract")

    plain = seed(c, tmp, "plain.dem")
    r = c.post("/api/library/extract", json={"ids": [plain]}).get_json()
    eq("a row with no .csz always parses",
       c.get(f"/api/library/jobs/{r['job_ids'][0]}").get_json()["kind"], "extract")
    r = c.post("/api/library/extract", json={"ids": [plain], "force": True}).get_json()
    eq("force on a row with no .csz is harmless",
       c.get(f"/api/library/jobs/{r['job_ids'][0]}").get_json()["kind"], "extract")


def test_players_api(c, tmp: Path) -> None:
    """The player index, the profile, and the avatar bytes they point at."""
    print("\nplayer index")
    from app import config as cfg_mod, db

    roster = [{"steamid": PLAYER_A, "name": "ZywOo", "team": 0},
              {"steamid": PLAYER_B, "name": "donk", "team": 1}]
    scoreboard = [
        {"steamid": PLAYER_A, "name": "ZywOo", "team": 0, "kills": 20, "deaths": 10,
         "assists": 4, "hs": 9, "adr_damage": 1800, "kd": 2.0},
        {"steamid": PLAYER_B, "name": "donk", "team": 1, "kills": 10, "deaths": 18,
         "assists": 2, "hs": 5, "adr_damage": 900, "kd": 0.56},
    ]
    seed(c, tmp, "p1.dem", total_rounds=20, team_a_score=13, team_b_score=7,
         players_json=json.dumps(roster), scoreboard_json=json.dumps(scoreboard))

    r = c.get("/api/library/players").get_json()
    eq("two players", r["total"], 2)
    eq("the index ships an avatars map", r["avatars"], {})
    eq("and the cache counters", r["avatar_stats"]["known"], 0)
    by = {p["steamid"]: p for p in r["items"]}
    eq("totals are aggregated", by[PLAYER_A]["kills"], 20)
    eq("and the derived numbers too", by[PLAYER_A]["adr"], 90.0)
    check("nobody is watched yet", by[PLAYER_A]["watched"] is False)
    eq("searching the index", c.get("/api/library/players?q=donk").get_json()["total"], 1)

    print("\nwatched")
    cfg_mod.Config().update({"watched_players": ["zywoo"]})
    r = c.get("/api/library/players").get_json()
    by = {p["steamid"]: p for p in r["items"]}
    check("matched case-insensitively", by[PLAYER_A]["watched"] is True)
    check("the other one is not", by[PLAYER_B]["watched"] is False)
    eq("the filter narrows to them",
       c.get("/api/library/players?watched=1").get_json()["total"], 1)
    cfg_mod.Config().update({"watched_players": []})

    print("\nfollowing from a profile")
    r = c.patch(f"/api/library/players/{PLAYER_A}", json={"watched": True}).get_json()
    check("the response says they are watched", r["watched"] is True)
    eq("written into the config", cfg_mod.Config().get("watched_players"), ["ZywOo"])
    r = c.patch(f"/api/library/players/{PLAYER_A}", json={"watched": False}).get_json()
    check("and can be undone", r["watched"] is False)
    eq("gone from the config", cfg_mod.Config().get("watched_players"), [])

    print("\nthe library cards follow the list")
    # The cards read a per-row copy of the match.  Writing only the config left
    # already-parsed demos highlighted for a player the user had just
    # unfollowed, which is backwards: they said they are not interested any
    # more, and the highlight was the only thing that did not hear it.
    def cards() -> list:
        return c.get("/api/library/list").get_json()["items"][0]["watched"]

    eq("nothing highlighted to start with", cards(), [])
    r = c.patch(f"/api/library/players/{PLAYER_A}", json={"watched": True}).get_json()
    eq("following reports the rows it resynced", r["watched_resynced"]["changed"], 1)
    eq("the card highlight appears", [w["name"] for w in cards()], ["ZywOo"])
    eq("and the library filter agrees",
       c.get("/api/library/list?watched=1").get_json()["total"], 1)
    c.patch(f"/api/library/players/{PLAYER_A}", json={"watched": False})
    eq("unfollowing clears it with no per-demo refresh", cards(), [])
    eq("and the filter empties",
       c.get("/api/library/list?watched=1").get_json()["total"], 0)

    print("\nsaving the list in settings resyncs too")
    r = c.put("/api/library/config", json={"watched_players": ["donk"]}).get_json()
    eq("the save reports the resync", r["watched_resynced"]["changed"], 1)
    eq("the highlight follows the new name", [w["name"] for w in cards()], ["donk"])
    r = c.put("/api/library/config", json={"watched_players": ["donk"]}).get_json()
    eq("saving the same list again changes nothing", r["watched_resynced"]["changed"], 0)
    r = c.put("/api/library/config", json={"auto_extract": True}).get_json()
    check("an unrelated save does not resync", "watched_resynced" not in r)
    c.put("/api/library/config", json={"watched_players": []})

    # A partial entry typed by hand still matches, so the response must not
    # claim the player was unwatched when the highlight is still there.
    cfg_mod.Config().update({"watched_players": ["Zyw"]})
    r = c.patch(f"/api/library/players/{PLAYER_A}", json={"watched": False}).get_json()
    check("a hand-typed partial entry keeps matching", r["watched"] is True)
    check("and the card still shows it", [w["name"] for w in cards()], ["ZywOo"])
    cfg_mod.Config().update({"watched_players": []})

    eq("an unknown player is a 404",
       c.patch("/api/library/players/76561190000000000",
               json={"watched": True}).status_code, 404)
    eq("an empty patch is rejected",
       c.patch(f"/api/library/players/{PLAYER_A}", json={}).status_code, 400)

    print("\nprofile")
    r = c.get(f"/api/library/players/{PLAYER_A}").get_json()
    eq("the player", r["player"]["name"], "ZywOo")
    eq("one map row", len(r["maps"]), 1)
    eq("their demo count", r["demos"], 1)
    check("no avatar cached yet", r["avatar"] is None)
    eq("an unknown steamid is a 404",
       c.get("/api/library/players/76561190000000000").status_code, 404)

    print("\navatar bytes")
    eq("nothing to serve yet",
       c.get(f"/api/library/avatar/{PLAYER_A}").status_code, 404)
    f = config.AVATAR_DIR / f"{PLAYER_A}.jpg"
    f.write_bytes(JPEG)
    db.upsert_avatar(PLAYER_A, status=db.AVATAR_OK, file_path=str(f),
                     persona_name="Herbus doubledoor")
    resp = c.get(f"/api/library/avatar/{PLAYER_A}")
    eq("served", resp.status_code, 200)
    eq("as a jpeg", resp.headers["Content-Type"], "image/jpeg")
    check("with the cached bytes", resp.data == JPEG)
    check("and a cache header", "max-age" in resp.headers.get("Cache-Control", ""))
    eq("a steamid that is not a steamid is a 404",
       c.get("/api/library/avatar/not-a-steamid").status_code, 404)

    print("\nthe library list ships avatars")
    r = c.get("/api/library/list").get_json()
    check("the cached one is included", PLAYER_A in r["avatars"], list(r["avatars"]))
    check("with a cache-busted url",
          r["avatars"][PLAYER_A]["url"].startswith(f"/api/library/avatar/{PLAYER_A}?v="),
          r["avatars"][PLAYER_A]["url"])
    eq("and the persona", r["avatars"][PLAYER_A]["persona"], "Herbus doubledoor")
    check("the player with no file is left out", PLAYER_B not in r["avatars"])
    eq("the profile filter works on the library",
       c.get(f"/api/library/list?steamid={PLAYER_A}").get_json()["total"], 1)
    eq("and excludes the other player",
       c.get(f"/api/library/list?steamid=76561190000000000").get_json()["total"], 0)


def test_avatar_sync_api(c, tmp: Path) -> None:
    """The sync endpoint queues work and folds duplicate requests."""
    print("\navatar sync endpoint")
    from app import avatars, config as cfg_mod, db, jobs

    # The job syncs "every player in the library", so the library needs one.
    roster = [{"steamid": PLAYER_A, "name": "ZywOo", "team": 0}]
    seed(c, tmp, "sync.dem", players_json=json.dumps(roster),
         scoreboard_json=json.dumps([{**roster[0], "kills": 1, "deaths": 1,
                                      "assists": 0, "hs": 0, "adr_damage": 0}]))

    cfg_mod.Config().update({"avatar_delay_ms": 0})
    with stub_network():
        r = c.post("/api/library/avatars/sync", json={}).get_json()
        check("a job is queued", r["queued"] is True, r)
        check("with an id", bool(r["job_id"]))

        original = jobs.manager.has_active
        jobs.manager.has_active = lambda kind=None: True
        try:
            again = c.post("/api/library/avatars/sync", json={}).get_json()
        finally:
            jobs.manager.has_active = original
        check("a second request folds into the running one", again["queued"] is False)
        check("and says why", bool(again["note"]), again)

        eq("a bad payload is rejected",
           c.post("/api/library/avatars/sync", json={"steamids": "abc"}).status_code, 400)

        # Let the queue drain so this group cannot leak work into the next one.
        deadline = time.time() + 10
        while jobs.manager.pending_count() and time.time() < deadline:
            time.sleep(0.05)
        avatars._again = False

    check("the sync really cached something", db.avatar_stats()["ok"] >= 1,
          db.avatar_stats())
    check("and wrote files", any(config.AVATAR_DIR.glob("*.jpg")),
          list(config.AVATAR_DIR.iterdir()))

    # Turning the setting off is honoured by a plain sync, but not by force.
    cfg_mod.Config().update({"fetch_avatars": False})
    with stub_network():
        r = c.post("/api/library/avatars/sync", json={}).get_json()
        check("still queues", r["queued"] is True)
        deadline = time.time() + 10
        while jobs.manager.pending_count() and time.time() < deadline:
            time.sleep(0.05)
        avatars._again = False
    cfg_mod.Config().update({"fetch_avatars": True})

    print("\nconfig surface")
    cfg = c.get("/api/library/config").get_json()
    check("the config exposes the avatar setting", "fetch_avatars" in cfg)
    check("and the cache counters", "avatars" in cfg)
    r = c.put("/api/library/config", json={"fetch_avatars": False,
                                           "avatar_ttl_days": 3}).get_json()
    eq("the setting can be turned off", r["fetch_avatars"], False)
    eq("and the ttl clamped and stored", r["avatar_ttl_days"], 3)
    eq("an absurd ttl is clamped, not rejected",
       c.put("/api/library/config", json={"avatar_ttl_days": 99999}).get_json()
       ["avatar_ttl_days"], 365)
    eq("nonsense is rejected",
       c.put("/api/library/config", json={"avatar_ttl_days": "soon"}).status_code, 400)


def _player_row(steamid: str, name: str, team: int = 0, kills: int = 10) -> dict:
    return {"steamid": steamid, "name": name, "team": team, "kills": kills,
            "deaths": 5, "assists": 1, "hs": 4, "adr_damage": 900}


def _roster(*rows: dict) -> list:
    return [{"steamid": r["steamid"], "name": r["name"], "team": r["team"]}
            for r in rows]


def test_player_search(c, tmp: Path) -> None:
    """Search reaches the in-game name, the Steam persona and the SteamID."""
    print("\nplayer search")
    from app import db

    rows = [_player_row(PLAYER_A, "Wicadia"), _player_row(PLAYER_B, "donk", team=1)]
    seed(c, tmp, "s1.dem", players_json=json.dumps(_roster(*rows)),
         scoreboard_json=json.dumps(rows))
    db.upsert_avatar(PLAYER_A, persona_name="am\u0131naKE", status=db.AVATAR_OK)

    r = c.get("/api/library/players/search", query_string={"q": "wic"}).get_json()
    eq("an in-game name finds the player", [p["steamid"] for p in r["items"]], [PLAYER_A])
    eq("and the response says which field matched", r["items"][0]["match"], "name")
    eq("the query is echoed back", r["q"], "wic")
    check("avatars ride along", isinstance(r["avatars"], dict))
    eq("the item carries every in-game name", r["items"][0]["names"], ["Wicadia"])
    eq("and the Steam persona", r["items"][0]["persona"], "am\u0131naKE")

    r = c.get("/api/library/players/search", query_string={"q": "am\u0131na"}).get_json()
    eq("the Steam persona finds them too", [p["steamid"] for p in r["items"]], [PLAYER_A])
    eq("named as a persona match", r["items"][0]["match"], "persona")

    r = c.get("/api/library/players/search", query_string={"q": PLAYER_A}).get_json()
    eq("a full SteamID finds them", [p["steamid"] for p in r["items"]], [PLAYER_A])
    eq("named as a steamid match", r["items"][0]["match"], "steamid")

    eq("an empty query returns nothing",
       c.get("/api/library/players/search?q=").get_json()["items"], [])
    eq("a query nobody matches returns nothing",
       c.get("/api/library/players/search?q=zzzzzz").get_json()["items"], [])
    eq("the limit is clamped, not rejected",
       c.get("/api/library/players/search?q=a&limit=9999").status_code, 200)

    print("\nthe grid filter reaches the same fields")
    eq("by persona", c.get("/api/library/players",
                           query_string={"q": "am\u0131na"}).get_json()["total"], 1)
    eq("by in-game name", c.get("/api/library/players?q=wic").get_json()["total"], 1)
    eq("by SteamID", c.get(f"/api/library/players?q={PLAYER_A}").get_json()["total"], 1)

    print("\nthe profile carries the name history")
    r = c.get(f"/api/library/players/{PLAYER_A}").get_json()
    eq("history", [n["name"] for n in r["name_history"]], ["Wicadia"])
    eq("with a match count", r["name_history"][0]["matches"], 1)
    eq("and the player itself is titled by the persona",
       r["player"]["persona"], "am\u0131naKE")


def test_storage(c, tmp: Path) -> None:
    """The settings page measures the caches and can wipe them."""
    print("\nstorage overview")
    from app import config as cfg_mod, db

    did = seed(c, tmp, "s.dem")
    csz = cfg_mod.CSZ_DIR / "s__abcd1234.csz"
    csz.write_bytes(b"x" * 2048)
    thumb = cfg_mod.THUMB_DIR / "demo_1.png"
    thumb.write_bytes(b"y" * 512)
    maps = cfg_mod.THUMB_DIR / "_maps"
    maps.mkdir(exist_ok=True)
    (maps / "de_mirage.v1.png").write_bytes(b"z" * 256)
    avatar = cfg_mod.AVATAR_DIR / f"{PLAYER_A}.jpg"
    avatar.write_bytes(JPEG)
    recycle = cfg_mod.DATA_DIR / "recycle"
    recycle.mkdir(exist_ok=True)
    (recycle / "gone.dem").write_bytes(b"d" * 4096)

    db.update_demo(did, csz_path=str(csz), thumb_path=str(thumb),
                   thumb_key="abcd1234", status=db.STATUS_READY)
    db.upsert_avatar(PLAYER_A, status=db.AVATAR_OK, file_path=str(avatar),
                     persona_name="am\u0131naKE")

    r = c.get("/api/library/storage").get_json()
    by = {e["key"]: e for e in r["entries"]}
    eq("the .csz is counted", by["csz"]["files"], 1)
    eq("with its bytes", by["csz"]["bytes"], 2048)
    eq("thumbnails include the per-map cache", by["thumbs"]["files"], 2)
    eq("and their bytes", by["thumbs"]["bytes"], 768)
    eq("avatars are counted", by["avatars"]["files"], 1)
    eq("so is the recycle bin", by["recycle"]["files"], 1)
    eq("the database is listed but not clearable", by["db"]["removable"], False)
    check("the total covers every entry",
          r["total_bytes"] >= 2048 + 768 + 4096, r["total_bytes"])
    eq("the original demos are reported separately", r["demos"]["count"], 1)
    check("the data directory is named", str(cfg_mod.DATA_DIR) in r["data_dir"])

    print("\nclearing one cache")
    r = c.post("/api/library/storage/clean", json={"targets": ["csz"]})
    eq("accepted", r.status_code, 200)
    body = r.get_json()
    eq("one file removed", body["cleaned"]["csz"]["files"], 1)
    eq("bytes freed", body["cleaned"]["csz"]["bytes"], 2048)
    check("the file is gone", not csz.exists())
    row = db.get_demo(did)
    eq("the row stops pointing at it", row["csz_path"], None)
    eq("and is no longer ready", row["status"], db.STATUS_DISCOVERED)
    eq("the response carries a fresh overview",
       {e["key"]: e for e in body["overview"]["entries"]}["csz"]["files"], 0)

    print("\nclearing several at once")
    body = c.post("/api/library/storage/clean",
                  json={"targets": ["thumbs", "avatars", "recycle"]}).get_json()
    eq("thumbnails removed", body["cleaned"]["thumbs"]["files"], 2)
    eq("the per-map cache directory went too", maps.exists(), False)
    eq("the avatar file removed", body["cleaned"]["avatars"]["files"], 1)
    eq("the recycle bin emptied", body["cleaned"]["recycle"]["files"], 1)
    row = db.get_demo(did)
    eq("the thumbnail path was dropped", row["thumb_path"], None)
    eq("and the cache key with it", row["thumb_key"], None)
    av = db.get_avatar(PLAYER_A)
    eq("the avatar row is pending again", av["status"], db.AVATAR_PENDING)
    eq("with no file path", av["file_path"], None)
    eq("but the Steam name survives", av["persona_name"], "am\u0131naKE")

    print("\nbad input")
    eq("no targets", c.post("/api/library/storage/clean", json={}).status_code, 400)
    eq("targets that are not a list",
       c.post("/api/library/storage/clean", json={"targets": "csz"}).status_code, 400)
    eq("the database cannot be cleared",
       c.post("/api/library/storage/clean", json={"targets": ["db"]}).status_code, 400)
    r = c.post("/api/library/storage/clean", json={"targets": ["nonsense"]})
    eq("an unknown target", r.status_code, 400)
    check("with a message naming it", "nonsense" in r.get_json()["error"],
          r.get_json())


def main() -> int:
    # Each group gets a fresh app + data dir, so a row seeded by one group
    # cannot quietly change another group's totals.
    groups = [
        ("basics", test_basics, False),
        ("demo CRUD", test_demo_crud, True),
        ("delete semantics", test_delete_semantics, True),
        ("extract vs re-parse", test_force_reextract, True),
        ("list/filters", test_list_and_filters, True),
        ("config", test_config, False),
        ("watch", test_watch, True),
        ("jobs/errors", test_jobs_and_errors, False),
        ("batch handoff", test_batch_handoff, True),
        ("players/avatars", test_players_api, True),
        ("player search", test_player_search, True),
        ("storage", test_storage, True),
        # Last: it puts real work on the job queue, and draining it here keeps
        # a straggler from running against the next group's scratch dir.
        ("avatar sync endpoint", test_avatar_sync_api, True),
    ]
    for name, fn, needs_tmp in groups:
        try:
            with scratch_app() as (c, tmp):
                fn(c, tmp) if needs_tmp else fn(c)
        except Exception as exc:  # noqa: BLE001
            _failed.append(f"{name} raised {type(exc).__name__}")
            print(f"\n  \u2717 {name} raised {type(exc).__name__}: {exc}")

    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        for name in _failed:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
