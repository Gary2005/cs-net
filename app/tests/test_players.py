#!/usr/bin/env python3
"""
Player aggregation and Steam avatar cache.

Two things are checked here that the HTTP tests cannot reach:

* the SQL that turns per-demo scoreboards into a player index, including the
  cases that are easy to get subtly wrong -- two players sharing a nickname,
  a demo that was never extracted, a player whose in-game name changed between
  matches, and a win counted from the wrong side of the score
* the avatar pipeline with the network replaced, so the interesting paths
  (a host that is not Steam, bytes that are not an image, a lookup that failed
  yesterday) are exercised without depending on steamcommunity.com being up

Run:  python app/tests/test_players.py
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
def scratch_env():
    """Redirect every data path into a scratch dir for the duration."""
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
        from app import db
        db.init_db()
        try:
            yield tmp
        finally:
            for k, v in saved.items():
                setattr(config, k, v)


#: A real SteamID64 (ZywOo's account) and its Steam account id.
ZYWOO = "76561198113666193"
ZYWOO_ACC = 153400465
DONK = "76561199074863582"

#: Smallest thing that passes the image sniff.
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def seed_demo(tmp: Path, name: str, *, map_name: str, rounds: int,
              score=(13, 7), roster=(), added_at: str = "2026-01-01 00:00:00"):
    """One library row with a scoreboard, without extracting anything."""
    from app import db
    path = tmp / name
    path.write_bytes(b"fake demo")
    players = [{"steamid": p["steamid"], "name": p["name"], "team": p["team"]}
               for p in roster]
    scoreboard = [{
        "steamid": p["steamid"], "name": p["name"], "team": p["team"],
        "kills": p.get("kills", 0), "deaths": p.get("deaths", 0),
        "assists": p.get("assists", 0), "hs": p.get("hs", 0),
        "adr_damage": p.get("dmg", 0),
    } for p in roster]
    demo_id = db.upsert_demo(
        str(path), filename=name, file_size=10, status=db.STATUS_READY,
        map_name=map_name, total_rounds=rounds,
        team_a_score=score[0], team_b_score=score[1],
        players_json=json.dumps(players, ensure_ascii=False),
        scoreboard_json=json.dumps(scoreboard, ensure_ascii=False),
        added_at=added_at,
    )
    return demo_id


# ── aggregation ──────────────────────────────────────────────────────────────


def test_player_totals() -> None:
    print("\nplayer totals")
    from app import db
    with scratch_env() as tmp:
        seed_demo(tmp, "m1.dem", map_name="de_mirage", rounds=20, score=(13, 7), roster=[
            {"steamid": ZYWOO, "name": "ZywOo", "team": 0,
             "kills": 20, "deaths": 10, "assists": 4, "hs": 9, "dmg": 1800},
            {"steamid": DONK, "name": "donk", "team": 1,
             "kills": 10, "deaths": 18, "assists": 2, "hs": 5, "dmg": 900},
        ], added_at="2026-01-01 00:00:00")
        seed_demo(tmp, "m2.dem", map_name="de_inferno", rounds=30, score=(13, 10), roster=[
            {"steamid": ZYWOO, "name": "ZywOo", "team": 1,
             "kills": 24, "deaths": 20, "assists": 6, "hs": 12, "dmg": 2400},
            {"steamid": DONK, "name": "donk", "team": 0,
             "kills": 30, "deaths": 15, "assists": 3, "hs": 15, "dmg": 2700},
        ], added_at="2026-02-01 00:00:00")

        result = db.list_players()
        eq("two players", result["total"], 2)
        by_id = {p["steamid"]: p for p in result["items"]}
        z = by_id[ZYWOO]

        eq("matches summed", z["matches"], 2)
        eq("kills summed", z["kills"], 44)
        eq("deaths summed", z["deaths"], 30)
        eq("assists summed", z["assists"], 10)
        eq("headshots summed", z["headshots"], 21)
        eq("damage summed", z["damage"], 4200)
        eq("rounds summed", z["rounds"], 50)
        eq("K/D", z["kd"], 1.47)
        eq("KPR divides by rounds, not matches", z["kpr"], 0.88)
        eq("ADR divides by rounds", z["adr"], 84.0)
        eq("headshot rate", z["hs_pct"], 47.7)
        eq("distinct maps", z["map_count"], 2)

        # The win/loss split has to follow the player's side, not team A.
        eq("won the match on team 0", z["wins"], 1)
        eq("lost the match on team 1", z["losses"], 1)
        eq("win rate", z["win_pct"], 50.0)

        d = by_id[DONK]
        eq("the other side of the same two matches", (d["wins"], d["losses"]), (1, 1))

        print("\nname and recency")
        eq("name comes from the most recent demo", z["name"], "ZywOo")
        eq("last map is from the most recent demo", z["last_map"], "de_inferno")
        check("last seen is the newest date",
              str(z["last_seen"]).startswith("2026-02-01"), z["last_seen"])

        # A rename must show the new name, not whichever row SQLite scanned first.
        seed_demo(tmp, "m3.dem", map_name="de_anubis", rounds=20, score=(13, 2), roster=[
            {"steamid": ZYWOO, "name": "ZywOo (renamed)", "team": 0, "kills": 1},
        ], added_at="2026-03-01 00:00:00")
        eq("a rename wins",
           db.get_player(ZYWOO)["name"], "ZywOo (renamed)")
        eq("the rename counts as a third match", db.get_player(ZYWOO)["matches"], 3)


def test_player_queries() -> None:
    print("\nplayer search / sort / filters")
    from app import db
    with scratch_env() as tmp:
        seed_demo(tmp, "a.dem", map_name="de_mirage", rounds=20, score=(13, 7), roster=[
            {"steamid": ZYWOO, "name": "ZywOo", "team": 0, "kills": 20, "deaths": 10,
             "dmg": 1800},
            {"steamid": DONK, "name": "donk", "team": 1, "kills": 5, "deaths": 20},
        ])

        eq("search by name", db.list_players(q="zywoo")["total"], 1)
        eq("search is case-insensitive", db.list_players(q="ZYW")["total"], 1)
        eq("search by steamid", db.list_players(q="76561199074863582")["total"], 1)
        eq("search that matches nothing", db.list_players(q="nobody")["total"], 0)

        eq("sort by kd puts the better player first",
           db.list_players(sort="kd")["items"][0]["steamid"], ZYWOO)
        eq("sort by kills agrees here",
           db.list_players(sort="kills")["items"][0]["steamid"], ZYWOO)
        eq("sort by name is alphabetical",
           db.list_players(sort="name")["items"][0]["name"], "donk")
        eq("an unknown sort falls back instead of erroring",
           db.list_players(sort="nonsense")["sort"], "matches")

        eq("watched filter keeps the match", db.list_players(watched_names=["Zyw"])["total"], 1)
        eq("watched filter drops the rest", db.list_players(watched_names=["m0NESY"])["total"], 0)
        eq("watched filter is case-insensitive",
           db.list_players(watched_names=["zywoo"])["total"], 1)

        eq("paging", len(db.list_players(limit=1, offset=1)["items"]), 1)
        eq("paging reports the full total", db.list_players(limit=1, offset=1)["total"], 2)

        print("\nper-map breakdown")
        maps = db.player_maps(ZYWOO)
        eq("one map", len(maps), 1)
        eq("map name", maps[0]["map_name"], "de_mirage")
        eq("map ADR", maps[0]["adr"], 90.0)
        eq("map win rate", maps[0]["win_pct"], 100.0)

        print("\nunknown player")
        check("get_player returns None", db.get_player("76561190000000000") is None)
        eq("player_maps is empty", db.player_maps("76561190000000000"), [])

        print("\ndemos for one player")
        eq("the roster filter finds the demo",
           db.list_demos(steamid=ZYWOO)["total"], 1)
        eq("and only that player's", db.list_demos(steamid=DONK)["total"], 1)
        eq("an unknown steamid matches nothing",
           db.list_demos(steamid="76561190000000000")["total"], 0)

        # Same nickname, different account: the fuzzy `player` filter merges
        # them, the exact `steamid` one must not.
        other = "76561190000000123"
        seed_demo(tmp, "b.dem", map_name="de_cache", rounds=20, score=(13, 7), roster=[
            {"steamid": other, "name": "ZywOo", "team": 0, "kills": 1},
        ])
        eq("the name filter cannot tell them apart",
           db.list_demos(player="ZywOo")["total"], 2)
        eq("the steamid filter can", db.list_demos(steamid=ZYWOO)["total"], 1)
        eq("the player index keeps them separate", db.list_players()["total"], 3)


def test_name_history_and_persona() -> None:
    """The player list is titled by the Steam persona, and keeps every name.

    A demo carries the in-game name only, so the persona has to be joined in
    from ``player_avatars``.  A player who renamed must not lose the name the
    user still recognises from the demo it was played under, either -- that is
    what ``names`` is for, and why the search box has to consult it.
    """
    print("\nname history and persona")
    from app import db
    with scratch_env() as tmp:
        for i, (name, added) in enumerate((
            ("Wicadia", "2026-01-01 00:00:00"),
            ("Wicadia", "2026-02-01 00:00:00"),
            ("W1c", "2026-03-01 00:00:00"),
        ), start=1):
            seed_demo(tmp, f"m{i}.dem", map_name="de_mirage", rounds=20,
                      score=(13, 7), roster=[
                          {"steamid": ZYWOO, "name": name, "team": 0, "kills": 5},
                      ], added_at=added)

        eq("with no avatar row there is no persona", db.get_player(ZYWOO)["persona"], None)
        db.upsert_avatar(ZYWOO, persona_name="am\u0131naKE", status=db.AVATAR_OK)

        p = db.get_player(ZYWOO)
        eq("the newest in-game name is still ``name``", p["name"], "W1c")
        eq("the Steam persona is joined in", p["persona"], "am\u0131naKE")
        eq("every name is kept, newest first", p["names"], ["W1c", "Wicadia"])

        history = db.player_names(ZYWOO)
        eq("the history lists both names",
           [h["name"] for h in history], ["W1c", "Wicadia"])
        eq("and counts the demos each was used in",
           [h["matches"] for h in history], [1, 2])
        eq("an unknown player has no history", db.player_names("76561190000000000"), [])

        print("\nthe list filter reaches all three fields")
        eq("by Steam persona", db.list_players(q="am\u0131na")["total"], 1)
        eq("by a name the player no longer uses",
           db.list_players(q="Wicadia")["total"], 1)
        eq("by SteamID", db.list_players(q=ZYWOO)["total"], 1)
        eq("a name nobody has", db.list_players(q="nobody")["total"], 0)

        print("\nthe library search reaches the persona too")
        eq("a Steam name finds that player's demos",
           db.list_demos(q="am\u0131naKE")["total"], 3)
        eq("an old in-game name still finds them",
           db.list_demos(q="Wicadia")["total"], 2)
        eq("a name nobody used finds nothing",
           db.list_demos(q="nobody")["total"], 0)


def test_player_search_ranking() -> None:
    """Ranked candidates, so a half-remembered handle still finds its player."""
    print("\nranked player search")
    from app import db
    from app.db import _fuzzy_score as score

    print("\nthe scorer")
    eq("an identical string", score("zywoo", "ZywOo"), 1.0)
    eq("case does not matter", score("ZYWOO", "zywoo"), 1.0)
    check("a prefix beats a substring", score("zy", "ZywOo") > score("zy", "Crazy"),
          f"{score('zy', 'ZywOo')} vs {score('zy', 'Crazy')}")
    check("a substring beats a scattered subsequence",
          score("cda", "Wicadia") > score("wcda", "Wicadia"),
          f"{score('cda', 'Wicadia')} vs {score('wcda', 'Wicadia')}")
    check("a subsequence still matches", score("wcda", "Wicadia") > 0)
    eq("unrelated text scores nothing", score("donk", "ZywOo"), 0.0)
    eq("an empty query scores nothing", score("", "ZywOo"), 0.0)
    eq("an empty candidate scores nothing", score("zywoo", ""), 0.0)

    with scratch_env() as tmp:
        seed_demo(tmp, "a.dem", map_name="de_mirage", rounds=20, score=(13, 7), roster=[
            {"steamid": ZYWOO, "name": "Wicadia", "team": 0, "kills": 30},
            {"steamid": DONK, "name": "Crazy", "team": 1, "kills": 5},
        ], added_at="2026-01-01 00:00:00")
        seed_demo(tmp, "a2.dem", map_name="de_inferno", rounds=20, score=(13, 7), roster=[
            {"steamid": ZYWOO, "name": "Wicadia", "team": 0, "kills": 20},
        ], added_at="2026-01-15 00:00:00")
        # A second account that has only ever played one match under the same
        # nickname: the tie between the two must break on match count.
        twin = "76561190000000123"
        seed_demo(tmp, "b.dem", map_name="de_cache", rounds=20, score=(13, 7), roster=[
            {"steamid": twin, "name": "Wicadia", "team": 0, "kills": 1},
        ], added_at="2026-02-01 00:00:00")

        eq("an exact name is the top hit",
           db.search_players("Wicadia")[0]["steamid"], ZYWOO)
        eq("and reports the field", db.search_players("Wicadia")[0]["match"], "name")
        eq("a prefix reaches it", db.search_players("wic")[0]["steamid"], ZYWOO)
        eq("a subsequence reaches it",
           db.search_players("wcda")[0]["steamid"], ZYWOO)
        eq("both accounts are candidates for the shared nickname",
           len(db.search_players("Wicadia")), 2)
        eq("the tie breaks on match count",
           db.search_players("Wicadia")[0]["matches"], 2)
        eq("the substring case-insensitively hits the other player",
           db.search_players("zy")[0]["steamid"], DONK)

        print("\nsteam ids")
        eq("a full SteamID is a hit", db.search_players(ZYWOO)[0]["steamid"], ZYWOO)
        eq("and is reported as one", db.search_players(ZYWOO)[0]["match"], "steamid")
        eq("a prefix of one is a hit",
           db.search_players(ZYWOO[:12])[0]["steamid"], ZYWOO)
        # Every SteamID64 starts "7656119", so a short numeric query would rank
        # the entire library equally and drown out the names.
        eq("a number too short to be an id matches nothing", db.search_players("7656"), [])

        print("\nthe steam persona")
        db.upsert_avatar(DONK, persona_name="Herbus doubledoor", status=db.AVATAR_OK)
        eq("a persona finds its player",
           db.search_players("herbus")[0]["steamid"], DONK)
        eq("and is reported as a persona", db.search_players("herbus")[0]["match"], "persona")
        eq("the matched text comes back", db.search_players("herbus")[0]["match_text"],
           "Herbus doubledoor")

        print("\nshape of the result")
        eq("nothing matches nothing", db.search_players("zzzzzz"), [])
        eq("an empty query is not a search", db.search_players("   "), [])
        eq("limit is honoured", len(db.search_players("a", limit=1)), 1)
        eq("limit cannot go negative", len(db.search_players("a", limit=-5)), 1)
        hits = db.search_players("a", limit=10)
        check("results are ordered by score",
              all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1)),
              [h["score"] for h in hits])


def test_unextracted_and_broken_rows() -> None:
    """A row with no scoreboard must not appear, or crash the aggregation."""
    print("\nrows the aggregation has to survive")
    from app import db
    with scratch_env() as tmp:
        seed_demo(tmp, "good.dem", map_name="de_mirage", rounds=20, score=(13, 7), roster=[
            {"steamid": ZYWOO, "name": "ZywOo", "team": 0, "kills": 20, "deaths": 10},
        ])
        path = tmp / "pending.dem"
        path.write_bytes(b"x")
        db.upsert_demo(str(path), filename="pending.dem", status=db.STATUS_DISCOVERED)
        path2 = tmp / "broken.dem"
        path2.write_bytes(b"x")
        db.upsert_demo(str(path2), filename="broken.dem", status=db.STATUS_READY,
                       scoreboard_json="{not json at all", players_json="also not json")

        eq("only the extracted demo contributes players", db.list_players()["total"], 1)
        eq("malformed json does not take the page down",
           db.list_demos(steamid=ZYWOO)["total"], 1)
        eq("distinct_steamids skips the junk", db.distinct_steamids(), [ZYWOO])
        eq("stats still work", db.stats()["total"], 3)


# ── steam ids and urls ───────────────────────────────────────────────────────


def test_steam_id_helpers() -> None:
    print("\nsteamid plumbing")
    from app import avatars
    eq("account id", avatars.account_id(ZYWOO), ZYWOO_ACC)
    check("empty is not an id", avatars.account_id("") is None)
    check("words are not an id", avatars.account_id("ZywOo") is None)
    check("a short number is not an id", avatars.account_id("12345") is None)
    check("an account id is not a steamid64", avatars.account_id("153400465") is None)
    check("below the base is rejected",
          avatars.account_id(str(avatars.STEAM_ID64_BASE)) is None)

    print("\navatar url allowlist")
    for url in (
        "https://avatars.fastly.steamstatic.com/abc_full.jpg",
        "https://avatars.akamai.steamstatic.com/abc_full.jpg",
        "https://avatars.cloudflare.steamstatic.com/abc_full.jpg",
        "https://steamcdn-a.akamaihd.net/steamcommunity/public/images/avatars/ab/abc_full.jpg",
        "//avatars.fastly.steamstatic.com/abc_full.jpg",
    ):
        check(f"allowed: {url[:52]}", avatars.allowed_avatar_url(url) != "")

    for url in (
        "http://avatars.fastly.steamstatic.com/abc.jpg",   # not https
        "https://evil.example.com/abc.jpg",
        "https://steamstatic.com.evil.example/abc.jpg",
        "https://notsteamstatic.com/abc.jpg",
        "https://avatars.fastly.steamstatic.com.evil.example/abc.jpg",
        "",
    ):
        check(f"rejected: {url[:52] or '(empty)'}", avatars.allowed_avatar_url(url) == "")

    print("\nimage sniffing")
    eq("jpeg", avatars.image_ext(JPEG), "jpg")
    eq("png", avatars.image_ext(PNG), "png")
    eq("gif", avatars.image_ext(b"GIF89a" + b"\x00" * 20), "gif")
    eq("webp", avatars.image_ext(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8), "webp")
    check("html is not an image",
          avatars.image_ext(b"<!DOCTYPE html><html>") is None)
    check("empty is not an image", avatars.image_ext(b"") is None)


# ── the fetch pipeline, with the network replaced ────────────────────────────


@contextlib.contextmanager
def stub_http(handler):
    """Replace the module's one network call.  Keeps every test offline."""
    from app import avatars
    original = avatars._http_get
    calls: list[str] = []

    def fake(url, *, accept="application/json", timeout=10.0, limit=1 << 22):
        calls.append(url)
        return handler(url, accept)

    avatars._http_get = fake
    try:
        yield calls
    finally:
        avatars._http_get = original


def _profile(payload: dict, image: bytes = JPEG):
    def handler(url, accept):
        if "miniprofile" in url:
            return json.dumps(payload).encode("utf-8")
        return image
    return handler


def test_fetch_and_store() -> None:
    print("\nfetch and store")
    from app import avatars, db
    payload = {"persona_name": "Herbus doubledoor",
               "avatar_url": "https://avatars.fastly.steamstatic.com/abc_full.jpg"}

    with scratch_env():
        with stub_http(_profile(payload)) as calls:
            result = avatars.fetch_and_store(ZYWOO)

        check("reported success", result["ok"], result.get("error", ""))
        eq("asked the mini-profile for the right account",
           calls[0], f"https://steamcommunity.com/miniprofile/{ZYWOO_ACC}/json")
        eq("then asked the CDN", calls[1], payload["avatar_url"])

        path = Path(result["path"])
        check("the file is on disk", path.is_file())
        eq("named after the steamid", path.stem, ZYWOO)
        eq("extension comes from the bytes", path.suffix, ".jpg")
        eq("the bytes are the ones served", path.read_bytes(), JPEG)

        row = db.get_avatar(ZYWOO)
        eq("status", row["status"], db.AVATAR_OK)
        eq("persona kept", row["persona_name"], "Herbus doubledoor")
        eq("attempt counted", row["attempts"], 1)
        check("a timestamp was written", bool(row["fetched_at"]))
        eq("stats", db.avatar_stats()["ok"], 1)

        print("\na second extension replaces the first")
        with stub_http(_profile(payload, image=PNG)):
            result = avatars.fetch_and_store(ZYWOO)
        eq("now a png", Path(result["path"]).suffix, ".png")
        eq("the old jpg is gone", list(config.AVATAR_DIR.glob(f"{ZYWOO}.jpg")), [])
        eq("only one file left", len(list(config.AVATAR_DIR.glob(f"{ZYWOO}.*"))), 1)
        eq("attempts accumulate", db.get_avatar(ZYWOO)["attempts"], 2)


def test_fetch_failures() -> None:
    """Every failure has to land as a row the UI can explain."""
    print("\nfailures are recorded, not raised")
    from app import avatars, db

    with scratch_env():
        with stub_http(lambda url, accept: b"not json"):
            r = avatars.fetch_and_store(ZYWOO)
        check("bad json fails", not r["ok"])
        row = db.get_avatar(ZYWOO)
        eq("recorded as an error", row["status"], db.AVATAR_ERROR)
        check("with a reason", "JSON" in row["error_msg"], row["error_msg"])
        eq("no file was written", list(config.AVATAR_DIR.glob(f"{ZYWOO}.*")), [])

        # A URL on somebody else's host must be refused before it is fetched.
        with stub_http(_profile({"persona_name": "x",
                                 "avatar_url": "https://evil.example.com/a.jpg"})) as calls:
            r = avatars.fetch_and_store(DONK)
        check("an off-Steam avatar url fails", not r["ok"])
        eq("and is never fetched", len(calls), 1)
        check("reason names the url problem", "头像地址" in r["error"], r["error"])

        # Right host, wrong bytes.
        with stub_http(_profile(
            {"persona_name": "x",
             "avatar_url": "https://avatars.fastly.steamstatic.com/a.jpg"},
            image=b"<html>rate limited</html>",
        )):
            r = avatars.fetch_and_store(DONK)
        check("non-image bytes fail", not r["ok"])
        check("reason says so", "图片" in r["error"], r["error"])

        print("\na network error is one line, not a traceback")
        def boom(url, accept):
            raise avatars.AvatarError("连不上 steamcommunity.com（timed out）")
        with stub_http(boom):
            r = avatars.fetch_and_store(DONK)
        check("still a recorded failure", not r["ok"])
        check("no newline in the message", "\n" not in r["error"], repr(r["error"]))
        eq("two errors counted", db.avatar_stats()["error"], 2)


def test_staleness() -> None:
    print("\nwhat needs re-fetching")
    from app import avatars, db
    fresh = time.strftime("%Y-%m-%d %H:%M:%S")
    old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 30 * 86400))
    recent_error = time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(time.time() - 3600))
    old_error = time.strftime("%Y-%m-%d %H:%M:%S",
                              time.localtime(time.time() - 48 * 3600))

    with scratch_env():
        eq("an unknown steamid is stale", db.stale_steamids([ZYWOO]), [ZYWOO])

        f = config.AVATAR_DIR / f"{ZYWOO}.jpg"
        f.write_bytes(JPEG)
        db.upsert_avatar(ZYWOO, status=db.AVATAR_OK, file_path=str(f), fetched_at=fresh)
        eq("a fresh avatar is skipped", db.stale_steamids([ZYWOO]), [])

        db.upsert_avatar(ZYWOO, status=db.AVATAR_OK, file_path=str(f), fetched_at=old)
        eq("an old avatar is re-fetched", db.stale_steamids([ZYWOO]), [ZYWOO])

        db.upsert_avatar(ZYWOO, status=db.AVATAR_OK, file_path=str(f), fetched_at=fresh)
        f.unlink()
        eq("a fresh row whose file vanished is re-fetched",
           db.stale_steamids([ZYWOO]), [ZYWOO])

        db.upsert_avatar(DONK, status=db.AVATAR_ERROR, error_msg="x",
                         fetched_at=recent_error)
        eq("a recent failure is left alone", db.stale_steamids([DONK]), [])
        db.upsert_avatar(DONK, status=db.AVATAR_ERROR, error_msg="x",
                         fetched_at=old_error)
        eq("an old failure is retried", db.stale_steamids([DONK]), [DONK])

        eq("force re-fetches everything",
           sorted(db.stale_steamids([ZYWOO, DONK], force=True)), sorted([ZYWOO, DONK]))
        eq("the input is de-duplicated",
           db.stale_steamids([ZYWOO, ZYWOO, ZYWOO], force=True), [ZYWOO])

        print("\nlocal_avatar")
        db.upsert_avatar(ZYWOO, status=db.AVATAR_OK, file_path=str(f))
        check("no file means no avatar", avatars.local_avatar(ZYWOO) is None)
        f.write_bytes(JPEG)
        eq("a file means an avatar", avatars.local_avatar(ZYWOO), f)

        # A file that outlived a wiped row is still served.
        with db.connect() as conn:
            conn.execute("DELETE FROM player_avatars WHERE steamid = ?", (ZYWOO,))
            conn.commit()
        eq("the file alone is enough", avatars.local_avatar(ZYWOO), f)


def test_sync_offline() -> None:
    """The whole batch path, with the network replaced."""
    print("\nbatch sync")
    from app import avatars, db
    with scratch_env():
        seed_demo(config.DATA_DIR, "s.dem", map_name="de_mirage", rounds=20,
                  score=(13, 7), roster=[
                      {"steamid": ZYWOO, "name": "ZywOo", "team": 0, "kills": 1},
                      {"steamid": DONK, "name": "donk", "team": 1, "kills": 1},
                  ])
        eq("both players are in the library",
           sorted(db.distinct_steamids()), sorted([ZYWOO, DONK]))

        payload = {"persona_name": "p",
                   "avatar_url": "https://avatars.fastly.steamstatic.com/a.jpg"}
        with stub_http(_profile(payload)) as calls:
            result = avatars.sync(delay_ms=0)
        eq("both fetched", result["ok"], 2)
        eq("two lookups plus two downloads", len(calls), 4)
        eq("nothing failed", result["failed"], 0)

        with stub_http(_profile(payload)) as calls:
            result = avatars.sync(delay_ms=0)
        eq("a second run has nothing to do", result["total"], 0)
        eq("and makes no requests", len(calls), 0)

        with stub_http(_profile(payload)) as calls:
            result = avatars.sync([ZYWOO], force=True, delay_ms=0)
        eq("force re-fetches one", result["ok"], 1)
        eq("one lookup plus one download", len(calls), 2)

        print("\nwhen the setting is off")
        config.Config().update({"fetch_avatars": False})
        with stub_http(_profile(payload)) as calls:
            result = avatars.sync(force=True, delay_ms=0)
        eq("force still works", result["ok"], 2)
        with stub_http(_profile(payload)) as calls:
            result = avatars.sync(delay_ms=0)
        check("a plain run is skipped", result["skipped"])
        eq("and makes no requests", len(calls), 0)
        eq("with a reason for the UI", result["note"], "头像获取已关闭")

        print("\none failure does not stop the batch")
        def flaky(url, accept):
            if "miniprofile" in url and str(ZYWOO_ACC) in url:
                raise avatars.AvatarError("boom")
            return json.dumps(payload).encode() if "miniprofile" in url else JPEG
        with stub_http(flaky):
            result = avatars.sync(force=True, delay_ms=0)
        eq("the other one still landed", result["ok"], 1)
        eq("the failure is counted", result["failed"], 1)
        eq("and reported by steamid", result["errors"][0].split("：")[0], ZYWOO)


def test_queue_coalescing() -> None:
    """A scan that finishes 60 demos must not queue 60 avatar jobs.

    ``has_active`` is stubbed rather than a real job submitted: the module-level
    manager is not started here, and a queued job would never settle.
    """
    print("\njob coalescing")
    from app import avatars, jobs
    with scratch_env():
        original = jobs.manager.has_active
        jobs.manager.has_active = lambda kind=None: True
        try:
            check("a request while a sync is in flight is folded in",
                  avatars.queue_sync() is None)
            check("and asks for a follow-up run", avatars._again)
        finally:
            jobs.manager.has_active = original
            avatars._again = False

        check("with nothing in flight a request is queued",
              isinstance(avatars.queue_sync(), str))
        check("the follow-up flag was not set by that",
              avatars._again is False)


def main() -> int:
    test_player_totals()
    test_player_queries()
    test_name_history_and_persona()
    test_player_search_ranking()
    test_unextracted_and_broken_rows()
    test_steam_id_helpers()
    test_fetch_and_store()
    test_fetch_failures()
    test_staleness()
    test_sync_offline()
    test_queue_coalescing()
    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        for name in _failed:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
