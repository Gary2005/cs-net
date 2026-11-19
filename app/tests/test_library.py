#!/usr/bin/env python3
"""
Unit checks for the demo library that need no demo file and no extraction.

Covers the parts where a silent mistake would be expensive:

* source inference -- server name must beat the filename, because a user can
  rename a file but cannot fake the server it came from
* team grouping -- must survive the halftime swap *and* a stand-in who only
  plays part of the match, which is where a naive "side in round 0" split
  mis-assigns players
* scoreboard derivation -- round wins, K/D/A, watched-player matching
* the SQLite layer -- including that a re-scan does not clobber a hand-edited
  display name

Run:  python app/tests/test_library.py
Exit code is non-zero if anything fails.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

# The Windows console defaults to GBK, which cannot encode the check marks or
# the Chinese field names below. Force UTF-8 rather than degrade the output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from app import config, db, metadata  # noqa: E402


@contextlib.contextmanager
def scratch():
    """A temp directory *inside the workspace*.

    ``tempfile.TemporaryDirectory()`` puts it under the platform temp area,
    where this environment refuses both SQLite file creation and the eventual
    cleanup, so the tests would fail for reasons that have nothing to do with
    the code under test.
    """
    base = _ROOT / ".build" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def scratch_env():
    """A scratch dir *and* every config path pointed into it.

    Needed for anything that goes through a code path without a ``db_path``
    argument -- the watcher is the main one, since in production there is only
    ever one library and its DB calls take the default.  Without this the
    watcher happily registers test files into the real ``app/data/library.db`` and
    overwrites the real ``app/data/config.json``.
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
        config.CSZ_DIR.mkdir(parents=True, exist_ok=True)
        config.THUMB_DIR.mkdir(parents=True, exist_ok=True)
        config.AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        db.init_db()
        try:
            yield tmp
        finally:
            for k, v in saved.items():
                setattr(config, k, v)

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


# ── source inference ─────────────────────────────────────────────────────────


def test_source_inference() -> None:
    print("\nsource inference")
    eq("server name wins over filename",
       config.infer_source("FACEIT London", "5e_ranked_123.dem"), "FACEIT")
    eq("5E server", config.infer_source("5EPlay Competitive", ""), "5E")
    eq("perfect world server", config.infer_source("完美世界 竞技平台", ""), "完美")
    eq("wanmei", config.infer_source("wanmei arena", ""), "完美")
    eq("valve / matchmaking", config.infer_source("Valve Matchmaking", ""), "官匹")
    eq("starladder", config.infer_source("StarLadder StarSeries Fall 2026", ""), "StarLadder")
    eq("filename fallback", config.infer_source("", "g123-5e-match.dem"), "5E")
    eq("faceit filename", config.infer_source("", "faceit_match_1.dem"), "FACEIT")
    eq("unknown -> local", config.infer_source("some random server", "x.dem"), "本地")
    eq("empty", config.infer_source("", ""), "本地")

    print("\nteam names from filename")
    eq("hltv style", metadata.teams_from_filename("aurora-vs-vitality-m2-inferno.dem"),
       ("Aurora", "Vitality"))
    eq("underscore style", metadata.teams_from_filename("Navi_vs_Faze.dem"), ("Navi", "Faze"))
    eq("space style", metadata.teams_from_filename("G2 vs Spirit map1.dem"), ("G2", "Spirit"))
    eq("no vs -> none", metadata.teams_from_filename("random_scrim.dem"), (None, None))
    eq("noise rejected", metadata.teams_from_filename("m1-vs-m2.dem"), (None, None))


# ── synthetic demo ───────────────────────────────────────────────────────────


def make_demo(*, swap_at: int = 2, stand_in: bool = False, n_rounds: int = 4) -> dict:
    """A minimal but structurally real extracted demo.

    ``teams`` is per round and per player, exactly as ``demo_extract`` emits it,
    so the grouping code is exercised on the same shape it sees in production.
    """
    names = ["ZywOo", "flameZ", "mezii", "apEX", "ropz",
             "Wicadia", "kyxsan", "Jimpphat", "woxic", "XANTARES"]
    steamids = [str(76561198000000000 + i) for i in range(10)]
    players = [{"steamid": steamids[i], "name": names[i]} for i in range(10)]

    rounds = []
    for rid in range(n_rounds):
        first_half = rid < swap_at
        teams = []
        for i in range(10):
            if i < 5:
                side = "CT" if first_half else "T"
            else:
                side = "T" if first_half else "CT"
            if stand_in and i == 9 and rid < swap_at:
                side = "?"          # did not play yet
            teams.append(side)

        # CT wins every round. Because the sides swap at halftime, that is 2
        # rounds for each team -- which is the point: a scoreboard built from
        # "who was CT in round 0" would come out 4:0 here.
        winner = "CT"
        kills = []
        if rid == 0:
            kills.append({"t": 100, "a": 0, "v": 5, "as": 1, "w": 10, "hs": True, "dmg": 100})
            kills.append({"t": 200, "a": 5, "v": 0, "as": -1, "w": 10, "hs": False, "dmg": 100})

        rounds.append({
            "id": rid,
            "teams": teams,
            "winner": winner,
            "end_reason": "bomb_exploded",
            "events": {"kills": kills, "damage": []},
            "players": [{"x": [0.0], "y": [0.0]} for _ in range(10)],
        })

    return {
        "format": "cs2.demo.v7",
        "meta": {"demo": "t.dem", "map": "de_mirage", "total_ticks": 64000,
                 "sidechannel_tickrate": 64.0},
        "players": players,
        "weapons": {},
        "places": {},
        "rounds": rounds,
        "skipped_rounds": [],
    }


def test_team_grouping() -> None:
    print("\nteam grouping")
    data = make_demo()
    groups = metadata._team_groups(data["rounds"], 10)
    eq("two groups", len(groups), 2)
    check("each group has 5", sorted(len(g) for g in groups) == [5, 5],
          f"got {[len(g) for g in groups]}")
    eq("group 0 is players 0-4", sorted(groups[0]), [0, 1, 2, 3, 4])

    print("\nteam grouping with a stand-in (only plays the 2nd half)")
    data = make_demo(stand_in=True)
    groups = metadata._team_groups(data["rounds"], 10)
    eq("still two groups", len(groups), 2)
    eq("stand-in joins the right team", sorted(groups[0]), [0, 1, 2, 3, 4])
    eq("team 1 has 5 including the stand-in", sorted(groups[1]), [5, 6, 7, 8, 9])


def test_metadata() -> None:
    print("\nmetadata / scoreboard")
    data = make_demo()
    clans = {
        "76561198000000000": "Team Vitality", "76561198000000001": "Team Vitality",
        "76561198000000002": "Team Vitality", "76561198000000003": "Team Vitality",
        "76561198000000004": "Team Vitality",
        "76561198000000005": "Aurora Gaming", "76561198000000006": "Aurora Gaming",
        "76561198000000007": "Aurora Gaming", "76561198000000008": "Aurora Gaming",
        "76561198000000009": "Aurora Gaming",
    }
    md = metadata.build_metadata(data, filename="a-vs-b.dem",
                                 watched_players=["ZywOo", "wicadia"], clans=clans)
    eq("map", md["map_name"], "de_mirage")
    eq("rounds", md["total_rounds"], 4)
    eq("duration", md["duration_mins"], 16.7)
    eq("team A name from clans", md["team_a_name"], "Team Vitality")
    eq("team B name from clans", md["team_b_name"], "Aurora Gaming")
    eq("score 2:2", (md["team_a_score"], md["team_b_score"]), (2, 2))

    sb = {r["name"]: r for r in json.loads(md["scoreboard_json"])}
    eq("ZywOo kills", sb["ZywOo"]["kills"], 1)
    eq("ZywOo deaths", sb["ZywOo"]["deaths"], 1)
    eq("ZywOo assists", sb["ZywOo"]["assists"], 0)
    eq("ZywOo headshot", sb["ZywOo"]["hs"], 1)
    eq("flameZ assisted", sb["flameZ"]["assists"], 1)
    eq("Wicadia kills", sb["Wicadia"]["kills"], 1)

    watched = json.loads(md["watched_json"])
    eq("two watched players matched", len(watched), 2)
    check("watched is case-insensitive",
          {w["name"] for w in watched} == {"ZywOo", "Wicadia"}, str(watched))

    print("\nmetadata without clan names falls back to the filename")
    md2 = metadata.build_metadata(data, filename="aurora-vs-vitality-m1-mirage.dem")
    eq("team A", md2["team_a_name"], "Aurora")
    eq("team B", md2["team_b_name"], "Vitality")

    print("\nmetadata with neither source of names")
    md3 = metadata.build_metadata(data, filename="scrim.dem")
    eq("placeholder A", md3["team_a_name"], "Team A")
    eq("placeholder B", md3["team_b_name"], "Team B")


# ── database ─────────────────────────────────────────────────────────────────


def test_database() -> None:
    print("\ndatabase")
    with scratch() as tmp:
        dbf = Path(tmp) / "lib.db"
        db.init_db(dbf)

        did = db.upsert_demo(r"D:\demos\a.dem", db_path=dbf, filename="a.dem",
                             file_size=100, status=db.STATUS_DISCOVERED,
                             source="5E", map_name="de_mirage")
        check("insert returns an id", did > 0)

        row = db.get_demo(did, dbf)
        eq("round trip filename", row["filename"], "a.dem")
        eq("status label", row["status_label"], "待入库")

        # upsert on the same path updates rather than duplicating
        did2 = db.upsert_demo(r"D:\demos\a.dem", db_path=dbf, status=db.STATUS_READY,
                              total_rounds=24)
        eq("upsert is idempotent", did2, did)
        eq("total rows", db.stats(dbf)["total"], 1)
        eq("updated field", db.get_demo(did, dbf)["total_rounds"], 24)

        # JSON columns decode
        db.update_demo(did, db_path=dbf, players_json=json.dumps(
            [{"name": "ZywOo", "team": 0}], ensure_ascii=False))
        eq("players decoded", db.get_demo(did, dbf)["players"][0]["name"], "ZywOo")

        # a user-edited display name survives a metadata-style update
        db.update_demo(did, db_path=dbf, display_name="决赛图三")
        db.update_demo(did, db_path=dbf, map_name="de_inferno")
        eq("display name kept", db.get_demo(did, dbf)["display_name"], "决赛图三")

        db.upsert_demo(r"D:\demos\b.dem", db_path=dbf, filename="b.dem",
                       status=db.STATUS_READY, source="FACEIT", file_size=200,
                       map_name="de_dust2")
        db.upsert_demo(r"D:\demos\c.dem", db_path=dbf, filename="c.dem",
                       status=db.STATUS_ERROR, source="5E", file_size=300)

        eq("filter by source", db.list_demos(sources=["5E"], db_path=dbf)["total"], 2)
        eq("filter by status", db.list_demos(statuses=["ready"], db_path=dbf)["total"], 2)
        eq("filter by map", db.list_demos(map_name="de_dust2", db_path=dbf)["total"], 1)
        eq("search filename", db.list_demos(q="b.dem", db_path=dbf)["total"], 1)
        eq("search display name", db.list_demos(q="决赛", db_path=dbf)["total"], 1)

        # The single search box must find a player: that is the most common
        # thing anyone types into a demo library.
        db.update_demo(did, db_path=dbf, players_json=json.dumps(
            [{"name": "ZywOo", "team": 0}, {"name": "apEX", "team": 0}]))
        db.update_demo(did, db_path=dbf, team_a_name="Team Vitality",
                       team_b_name="Aurora Gaming", server_name="StarLadder Fall 2026")
        eq("search player name", db.list_demos(q="ZywOo", db_path=dbf)["total"], 1)
        eq("search partial player name", db.list_demos(q="zywo", db_path=dbf)["total"], 1)
        eq("search team name", db.list_demos(q="Vitality", db_path=dbf)["total"], 1)
        eq("search server name", db.list_demos(q="StarLadder", db_path=dbf)["total"], 1)
        eq("search miss", db.list_demos(q="donk", db_path=dbf)["total"], 0)
        eq("player= param still works", db.list_demos(player="apEX", db_path=dbf)["total"], 1)

        sizes = [r["file_size"] for r in db.list_demos(sort="size", db_path=dbf)["items"]]
        eq("sort by size desc", sizes, [300, 200, 100])

        listed = db.list_demos(limit=2, db_path=dbf)
        eq("limit respected", len(listed["items"]), 2)
        eq("total is unpaginated", listed["total"], 3)

        eq("stats maps", sorted(db.stats(dbf)["maps"]), ["de_dust2", "de_inferno"])

        removed = db.delete_demo(did, dbf)
        check("delete returns the row", removed is not None and removed["id"] == did)
        eq("gone after delete", db.get_demo(did, dbf), None)
        eq("total after delete", db.stats(dbf)["total"], 2)


def test_probe_path() -> None:
    print("\nwatch-path validation")
    with scratch() as tmp:
        p = Path(tmp)
        (p / "one.dem").write_bytes(b"x")
        (p / "two.dem").write_bytes(b"x")
        (p / "three.zip").write_bytes(b"x")
        info = config.probe_path(tmp)
        check("valid dir", info["valid"], str(info))
        eq("counts demos", info["demo_count"], 2)
        eq("counts zips", info["zip_count"], 1)

        missing = config.probe_path(str(p / "nope"))
        check("missing dir invalid", not missing["valid"])
        check("missing dir explains why", "不存在" in (missing.get("error") or ""))

        asfile = config.probe_path(str(p / "one.dem"))
        check("a file is not a directory", not asfile["valid"])


def test_probe_survives_garbage() -> None:
    """A truncated or fake .dem must not raise out of probe_demo.

    demoparser2 is Rust and *panics* on a malformed demo; PyO3 surfaces that as
    ``PanicException``, which inherits from ``BaseException``.  A plain
    ``except Exception`` therefore lets it escape, and one half-downloaded file
    would abort a directory scan -- or kill the watcher thread outright.
    """
    print("\nprobe on garbage files")
    from app import probe

    with scratch() as tmp:
        # Establish the hazard with evidence rather than by importing a name:
        # the panic type is raised but is not exposed as a module attribute, so
        # the only way to learn what it is, is to make demoparser panic.
        bad = Path(tmp) / "truncated.dem"
        bad.write_bytes(b"x")
        import demoparser2
        panic_type = None
        try:
            demoparser2.DemoParser(str(bad)).parse_header()
        except BaseException as exc:  # noqa: BLE001
            panic_type = type(exc)
        check("a malformed demo really does make demoparser raise", panic_type is not None)
        if panic_type is not None:
            check(f"{panic_type.__name__} is NOT an Exception subclass",
                  not issubclass(panic_type, Exception),
                  "if this ever changes, the BaseException handlers in "
                  "probe/ingest/jobs/watcher can be relaxed")
            check("and it is not importable by name, so `except` cannot list it",
                  not hasattr(demoparser2, panic_type.__name__))

    with scratch() as tmp:
        cases = {
            "empty.dem": b"",
            "tiny.dem": b"x",
            "short_magic.dem": b"PBDEMS2\x00",
            "random.dem": bytes(range(256)) * 8,
        }
        for name, blob in cases.items():
            f = Path(tmp) / name
            f.write_bytes(blob)
            try:
                info = probe.probe_demo(f)
            except BaseException as exc:  # noqa: BLE001
                check(f"probe_demo survives {name}", False,
                      f"raised {type(exc).__name__}: {exc}")
                continue
            check(f"probe_demo survives {name}", True)
            check(f"{name} reports ok=False", info["ok"] is False)
            check(f"{name} carries an error message", bool(info["error"]),
                  f"got {info!r}")
            check(f"{name} returns an empty team map", info["teams"] == {})
            check(f"{name} still returns the documented keys",
                  set(info) == {"ok", "map_name", "server_name", "teams",
                                "players", "error"})


def test_scan_survives_garbage() -> None:
    """A scan over a directory containing junk must still register the good file."""
    print("\nscan over a mixed directory")
    with scratch() as tmp:
        dbf = Path(tmp) / "lib.db"
        db.init_db(dbf)
        root = Path(tmp) / "demos"
        root.mkdir()
        (root / "broken.dem").write_bytes(b"not a demo at all")
        (root / "empty.dem").write_bytes(b"")

        from app import ingest
        summary = ingest.scan_roots(
            [{"path": str(root), "depth": 1, "source": "本地", "enabled": True}],
            db_path=dbf,
        )
        eq("both files were seen", summary["scanned"], 2)
        eq("both were registered", summary["registered"], 2)
        eq("no exception escaped the scan", summary["errors"], [])
        rows = db.list_demos(db_path=dbf)["items"]
        eq("two rows in the library", len(rows), 2)
        # They are registered, just without metadata -- a later manual extract
        # is what will surface the real error, and that is the right time.
        check("junk rows have no map", all(r["map_name"] is None for r in rows),
              str([r["map_name"] for r in rows]))


def test_restart_recovery() -> None:
    """Work interrupted by a crash must be picked up again, not left stuck."""
    print("\nrestart recovery")
    with scratch() as tmp:
        dbf = Path(tmp) / "lib.db"
        db.init_db(dbf)
        root = Path(tmp) / "demos"
        root.mkdir()
        (root / "a.dem").write_bytes(b"x")

        from app import ingest
        roots = [{"path": str(root), "depth": 1, "source": "本地", "enabled": True}]

        # Simulate a crash mid-extraction: the row is left in `extracting`.
        did = db.upsert_demo(str(root / "a.dem"), db_path=dbf, filename="a.dem",
                             status=db.STATUS_EXTRACTING, file_size=1)
        eq("row starts out extracting", db.get_demo(did, dbf)["status"], "extracting")

        reset = db.reset_stale_extracting(dbf)
        eq("stale row was reset", reset, [did])
        eq("status is discovered again", db.get_demo(did, dbf)["status"], "discovered")
        eq("resetting twice is a no-op", db.reset_stale_extracting(dbf), [])

        # A rescan must re-queue it -- it has no .csz, so it is unfinished work.
        eq("unfinished work is visible", db.unfinished_ids(dbf), [did])
        summary = ingest.scan_roots(roots, db_path=dbf)
        eq("nothing newly registered", summary["registered"], 0)
        eq("the orphan was recovered", summary["recovered"], 1)
        check("and is queued for extraction", did in summary["new_ids"],
              str(summary["new_ids"]))

        # A row that already has a real .csz is NOT re-queued.
        csz = Path(tmp) / "a.csz"
        csz.write_bytes(b"CSZ5")
        db.update_demo(did, db_path=dbf, status=db.STATUS_DISCOVERED, csz_path=str(csz))
        eq("finished work is not unfinished", db.unfinished_ids(dbf), [])
        summary = ingest.scan_roots(roots, db_path=dbf)
        eq("nothing recovered this time", summary["recovered"], 0)

        # Neither is a row the user deliberately left alone.
        db.update_demo(did, db_path=dbf, status=db.STATUS_ERROR)
        eq("error rows are not auto-retried", db.unfinished_ids(dbf), [])
        db.update_demo(did, db_path=dbf, status=db.STATUS_MISSING)
        eq("missing rows are not auto-retried", db.unfinished_ids(dbf), [])


def test_watcher_does_not_requeue() -> None:
    """The watcher must queue a demo once, not on every poll.

    A row stays "discovered" from the moment it is queued until a worker
    actually starts it, so a naive orphan sweep would re-queue the same demo
    every few seconds for as long as a long extraction sat in the queue.
    """
    print("\nwatcher re-queue guard")
    with scratch_env() as tmp:
        root = Path(tmp) / "demos"
        root.mkdir()
        (root / "a.dem").write_bytes(b"x")

        from app import watcher as watcher_mod

        cfg = config.Config()
        cfg.update({
            "watch_roots": [{"path": str(root), "depth": 1, "source": "本地",
                             "enabled": True}],
            "watch_enabled": True,
            "auto_extract": True,
        })

        w = watcher_mod.DemoWatcher()
        # Poll directly rather than starting the thread, so the test is
        # deterministic. `force=True` skips the stability gate.
        w.poll(force=True)
        first = dict(w.status()["last_result"])
        eq("first poll found the file", first["new"], 1)

        # The row is still "discovered" -- no worker has run -- so a second poll
        # would see it as unfinished work again.
        eq("the row is still discovered",
           db.list_demos()["items"][0]["status"], "discovered")
        w.poll(force=True)
        second = dict(w.status()["last_result"])
        eq("second poll queues nothing again", second["queued"], 0)
        eq("second poll finds nothing new", second["new"], 0)

        w.poll(force=True)
        eq("third poll also queues nothing", w.status()["last_result"]["queued"], 0)

        # A brand-new file is still queued, even after the guard is populated.
        (root / "b.dem").write_bytes(b"x")
        w.poll(force=True)
        eq("a genuinely new file is still queued", w.status()["last_result"]["queued"], 1)


def test_config_defaults() -> None:
    """New settings have to survive a config file written by an older build.

    ``_merge_defaults`` is what makes that true, so the avatar keys are checked
    through it rather than by reading :data:`config.DEFAULTS` directly: a config
    that predates the feature has to come back with the feature switched on.
    """
    print("\nconfig defaults")
    with scratch_env():
        cfg = config.Config()
        for key in ("fetch_avatars", "avatar_ttl_days", "avatar_delay_ms",
                    "avatar_error_retry_hours"):
            check(f"{key} has a default", key in cfg.all())

        check("avatars are on by default", cfg.get("fetch_avatars") is True)
        eq("the default ttl is a week", cfg.get("avatar_ttl_days"), 7)

        # Simulate a config.json written before the avatar keys existed.
        config.CONFIG_PATH.write_text(
            json.dumps({"watched_players": ["ZywOo"], "sampling": {"mode": "stride"}}),
            encoding="utf-8",
        )
        old = config.Config()
        eq("an unknown key is left alone", old.get("watched_players"), ["ZywOo"])
        check("the new key still appears", old.get("fetch_avatars") is True)
        eq("a partial sampling block is merged, not replaced",
           old.get("sampling"), {"mode": "stride", "interval": 0.25, "stride": 8})

        # The avatar dir is its own place under data/, not mixed into thumbs.
        eq("avatars live in app/data/avatars",
           config.AVATAR_DIR, config.DATA_DIR / "avatars")


def test_stop_drops_queued_jobs() -> None:
    """Stopping the queue must not leave work for the next start to resume.

    The API suite builds and tears down the whole app once per group in one
    process, and the job manager is a module-level singleton.  ``stop()`` used
    to leave the queue intact, so a group that queued more jobs than the single
    worker could finish handed the leftovers to the *next* group -- where the
    data dir had already been swapped, so an extraction meant for a fake demo
    landed on the next group's freshly seeded row and flipped it to ``error``.
    That showed up as three unrelated failures in the list/filter group.
    """
    print("\nstopping the job queue")
    from app import jobs

    manager = jobs.JobManager(workers=1)
    ran: list[str] = []

    # Block the worker on a job it has already started, so everything queued
    # behind it is still waiting when stop() is called.
    release = threading.Event()
    first = manager.submit("slow", "slow", lambda job: release.wait(timeout=5.0))
    for i in range(3):
        manager.submit("later", f"later-{i}", lambda job, n=i: ran.append(f"later-{n}"))
    manager.start()

    deadline = time.time() + 5.0
    while first.state != "running" and time.time() < deadline:
        time.sleep(0.01)
    eq("the first job is running", first.state, "running")

    manager.stop(timeout=0.1)
    release.set()
    manager.stop(timeout=2.0)

    eq("nothing queued behind it ran", ran, [])
    eq("the in-flight job was allowed to finish", first.state, "done")
    queued = [j for j in manager.list() if j["kind"] == "later"]
    eq("the dropped jobs are still listed", len(queued), 3)
    check("and every one of them reads cancelled",
          all(j["state"] == "cancelled" for j in queued),
          f"{[j['state'] for j in queued]}")
    check("with a reason the UI can show",
          all(j["error"] for j in queued))

    # The regression: a restart must not pick the dropped work back up.
    manager.start()
    manager.stop(timeout=2.0)
    eq("a restart does not resurrect them", ran, [])
    eq("the queue is empty", manager.pending_count(), 0)


def test_match_watched() -> None:
    """The rule that decides who counts as a watched player."""
    print("\nwatched matching")
    sb = [
        {"name": "ZywOo", "team": 0, "kills": 19, "deaths": 7, "assists": 7},
        {"name": "ropz", "team": 1, "kills": 10, "deaths": 12, "assists": 3},
        {"name": "Crazy", "team": 1, "kills": 1, "deaths": 2, "assists": 0},
    ]
    names = lambda wanted: [w["name"] for w in metadata.match_watched(sb, wanted)]  # noqa: E731

    eq("no list means nobody", metadata.match_watched(sb, []), [])
    eq("and None is the same", metadata.match_watched(sb, None), [])
    eq("blank entries are ignored", metadata.match_watched(sb, ["", "   "]), [])
    eq("an exact name", names(["ZywOo"]), ["ZywOo"])
    eq("case-insensitively", names(["zywoo"]), ["ZywOo"])
    eq("a half-typed entry still finds them", names(["Zyw"]), ["ZywOo"])
    eq("surrounding spaces are trimmed", names(["  ropz  "]), ["ropz"])

    # Substring matching is a deliberate trade: a short entry finds the player
    # you meant and anyone else whose name contains those letters.  Pinned here
    # so the rule cannot change without someone noticing.
    eq("a short entry matches by accident", names(["zy"]), ["ZywOo", "Crazy"])
    eq("two entries for one player count once", names(["Zyw", "ZywOo"]), ["ZywOo"])
    eq("two players are both reported", names(["Zyw", "ropz"]), ["ZywOo", "ropz"])

    entry = metadata.match_watched(sb, ["Zyw"])[0]
    eq("the entry records which list item matched", entry["matched"], "Zyw")
    eq("and carries the stats the card shows",
       (entry["kills"], entry["deaths"], entry["assists"], entry["team"]), (19, 7, 7, 0))


def test_watched_resync() -> None:
    """Editing the watched list has to update rows that are already parsed.

    ``watched_json`` is derived data stored per row, and the library cards read
    that copy.  Writing only the config left every existing demo highlighted for
    a player the user had just unfollowed.
    """
    print("\nwatched resync")
    from app import ingest

    def scoreboard(*names: str) -> str:
        return json.dumps([
            {"name": n, "team": i % 2, "kills": 10 + i, "deaths": 5, "assists": 1}
            for i, n in enumerate(names)
        ])

    with scratch_env():
        def add(name: str, sb: Optional[str], status: str, watched: str,
                day: int) -> None:
            p = config.DATA_DIR / name
            p.write_bytes(b"x")
            db.upsert_demo(str(p), filename=name, status=status,
                           scoreboard_json=sb, watched_json=watched,
                           map_name="de_mirage", added_at=f"2026-01-{day:02d} 00:00:00")

        add("a.dem", scoreboard("ZywOo", "ropz"), db.STATUS_READY,
            json.dumps(metadata.match_watched(
                json.loads(scoreboard("ZywOo", "ropz")), ["ZywOo"])), 1)
        add("b.dem", scoreboard("donk"), db.STATUS_READY, "[]", 2)
        # A row that was never extracted: no scoreboard, nothing to re-derive.
        add("c.dem", None, db.STATUS_DISCOVERED, None, 3)

        def cards() -> dict[str, list[str]]:
            return {
                r["filename"]: [w["name"] for w in (r.get("watched") or [])]
                for r in db.list_demos(limit=10)["items"]
            }

        def order() -> dict[str, str]:
            return {r["filename"]: r["added_at"]
                    for r in db.list_demos(limit=10)["items"]}

        order_before = order()
        eq("a.dem starts highlighted", cards()["a.dem"], ["ZywOo"])
        eq("b.dem does not", cards()["b.dem"], [])

        # The seeded state already agrees with the config, so this pass is the
        # no-op case and pins "does not churn rows that are already right".
        config.Config().update({"watched_players": ["ZywOo"]})
        r = ingest.resync_watched()
        eq("only rows with a scoreboard are checked", r["checked"], 2)
        eq("an already-correct library is left alone", r["changed"], 0)

        config.Config().update({"watched_players": []})
        r = ingest.resync_watched()
        eq("unfollowing changed one row", r["changed"], 1)
        eq("a.dem is no longer highlighted", cards()["a.dem"], [])
        eq("the library filter agrees", db.list_demos(watched_only=True)["total"], 0)
        eq("the sort key was not touched", order(), order_before)

        config.Config().update({"watched_players": ["donk"]})
        eq("following the other player", ingest.resync_watched()["changed"], 1)
        eq("b.dem is now highlighted", cards()["b.dem"], ["donk"])
        eq("and a.dem is not", cards()["a.dem"], [])
        eq("the filter follows", db.list_demos(watched_only=True)["total"], 1)
        eq("running it again changes nothing", ingest.resync_watched()["changed"], 0)

        # A row whose scoreboard is not valid JSON must not take the pass down:
        # one bad row would otherwise stop the whole library from following the
        # list, and the user would have no way to tell why.
        with db.connect() as conn:
            conn.execute("UPDATE demo_files SET scoreboard_json = ? WHERE filename = ?",
                         ("{not json", "a.dem"))
            conn.commit()
        config.Config().update({"watched_players": ["ZywOo"]})
        r = ingest.resync_watched()
        eq("the malformed row is still counted", r["checked"], 2)
        eq("but skipped, so only the good row changed", r["changed"], 1)
        eq("b.dem followed the new list", cards()["b.dem"], [])
        eq("the unextracted row still has no highlight", cards()["c.dem"], [])


def test_migrate_legacy_data_dir() -> None:
    """An install from before the move to ``app/data`` has to follow it.

    Paths in the database are absolute, so moving the directory on its own
    would leave every row pointing at files that are no longer there: the
    library would read as empty even though all the demos were still on disk.

    The negative cases matter more than the positive one.  The migration walks
    a real directory, so it has to be impossible for it to fire on a test
    scratch directory or on the repo root's training data.
    """
    print("\nmoving the data directory")
    keys = ("DATA_DIR", "DEFAULT_DATA_DIR", "LEGACY_DATA_DIR", "LIBRARY_DB",
            "CSZ_DIR", "THUMB_DIR", "AVATAR_DIR", "CONFIG_PATH")
    saved = {k: getattr(config, k) for k in keys}

    def build_library(root: Path) -> dict:
        """A tiny but structurally real library rooted at ``root``."""
        root.mkdir(parents=True, exist_ok=True)
        for sub in ("csz", "thumbs", "avatars"):
            (root / sub).mkdir(exist_ok=True)
        paths = {
            "csz": root / "csz" / "d__abc.csz",
            "thumb": root / "thumbs" / "demo_1.png",
            "avatar": root / "avatars" / "76561198113666193.jpg",
        }
        for p in paths.values():
            p.write_bytes(b"x")
        # Built and closed by hand rather than through db.connect(): an open
        # connection holds a handle on the directory and shutil.move would fail
        # on Windows with "being used by another process".
        conn = sqlite3.connect(str(root / "library.db"))
        conn.executescript(db.SCHEMA)
        conn.execute(
            "INSERT INTO demo_files (path, filename, csz_path, thumb_path, "
            "watch_root, added_at) VALUES (?,?,?,?,?,?)",
            (str(root.parent / "demos" / "d.dem"), "d.dem",
             str(paths["csz"]), str(paths["thumb"]),
             str(root.parent / "demos"), "2026-01-01 00:00:00"),
        )
        conn.execute(
            "INSERT INTO player_avatars (steamid, file_path, status) VALUES (?,?,?)",
            ("76561198113666193", str(paths["avatar"]), "ok"),
        )
        conn.commit()
        conn.close()
        return paths

    try:
        with scratch() as base:
            legacy = base / "data"
            target = base / "app" / "data"
            old_paths = build_library(legacy)

            config.DATA_DIR = target
            config.DEFAULT_DATA_DIR = target
            config.LEGACY_DATA_DIR = legacy
            config.LIBRARY_DB = target / "library.db"

            eq("it reports where the library went",
               db.migrate_legacy_data_dir(), target)
            check("the old directory is gone", not legacy.exists())
            check("the .csz came along",
                  (target / "csz" / "d__abc.csz").is_file())

            row = db.get_demo(1)
            eq("csz_path was rewritten", row["csz_path"],
               str(target / "csz" / "d__abc.csz"))
            eq("thumb_path was rewritten", row["thumb_path"],
               str(target / "thumbs" / "demo_1.png"))
            eq("a path outside the data dir is left alone", row["path"],
               str(base / "demos" / "d.dem"))
            eq("watch_root too", row["watch_root"], str(base / "demos"))
            eq("avatar files were rewritten",
               db.get_avatar("76561198113666193")["file_path"],
               str(target / "avatars" / "76561198113666193.jpg"))

            eq("running it a second time is a no-op",
               db.migrate_legacy_data_dir(), None)

            # ── the two ways this must refuse ────────────────────────────────
            #
            # 1) The repo root's data/ holds training data.  It has no
            #    demo_files table, so it is not ours and must not be moved.
            training = base / "training"
            (training / "data").mkdir(parents=True)
            (training / "data" / "metadata-0.parquet").write_bytes(b"PAR1")
            config.LEGACY_DATA_DIR = training / "data"
            config.DATA_DIR = training / "app-data"
            config.DEFAULT_DATA_DIR = training / "app-data"
            eq("training data is not mistaken for a library",
               db.migrate_legacy_data_dir(), None)
            check("and it is still where it was",
                  (training / "data" / "metadata-0.parquet").is_file())

            # 2) Every other test points DATA_DIR at a scratch directory.  If
            #    the guard did not compare against the default, this call would
            #    move the developer's real library into the scratch dir.
            guarded = base / "guarded"
            build_library(guarded)
            config.LEGACY_DATA_DIR = guarded
            config.DATA_DIR = base / "scratch"
            config.DEFAULT_DATA_DIR = base / "expected-default"
            eq("an overridden DATA_DIR is never migrated into",
               db.migrate_legacy_data_dir(), None)
            check("the library it pointed at is untouched",
                  (guarded / "library.db").is_file())
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def main() -> int:
    test_source_inference()
    test_team_grouping()
    test_metadata()
    test_database()
    test_probe_path()
    test_probe_survives_garbage()
    test_scan_survives_garbage()
    test_restart_recovery()
    test_watcher_does_not_requeue()
    test_config_defaults()
    test_migrate_legacy_data_dir()
    test_match_watched()
    test_watched_resync()
    test_stop_drops_queued_jobs()
    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        for name in _failed:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
