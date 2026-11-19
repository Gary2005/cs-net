"""
Steam avatars for the players found in the library.

A CS2 demo carries no pictures -- it carries SteamID64s.  The face lives on
Steam's CDN, and Steam's public mini-profile endpoint hands out both the persona
name and the avatar URL **without an API key and without a login**:

    https://steamcommunity.com/miniprofile/<account_id>/json

``account_id`` is SteamID64 minus the 76561197960265728 base.  That is the whole
trick, and it is what the reference tool does; its optional Steam Web API key is
a fallback this does not need.

Four deliberate choices:

* **Fetched on a background job, never inside a request.**  Ten players at
  roughly 0.4 s each is four seconds of latency on a page render.
* **Stored under ``app/data/avatars/`` instead of hotlinked.**  The page then keeps
  working offline, and the browser is not made to talk to a third party on every
  single load.
* **A failed lookup is a normal outcome, not an error.**  ``steamcommunity.com``
  is not reachable from mainland China without a proxy, so failure records the
  reason, the UI falls back to an initial, and nothing about ingestion breaks.
* **The persona name is kept separate from the in-game name.**  They differ
  more often than not for pros -- the demo says ``ZywOo`` while the Steam
  account behind that SteamID is somebody else entirely.  The library keeps
  showing the name the demo used and offers the persona as a subtitle.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from . import config, db, jobs

#: SteamID64 for account id 0.  Every individual account is this plus its id.
STEAM_ID64_BASE = 76561197960265728

MINIPROFILE_URL = "https://steamcommunity.com/miniprofile/{account_id}/json"

#: Hard cap on a downloaded avatar.  A real one is 10-30 KB; anything past this
#: is a broken response or somebody else's idea of a joke.
MAX_AVATAR_BYTES = 4 << 20

DEFAULT_TIMEOUT = 10.0
DEFAULT_DELAY_MS = 350

#: Only Steam's own CDNs.  The URL comes from a third party, so it is checked
#: before it is fetched -- otherwise a spoofed response could point the server
#: at anything on the network.
_ALLOWED_HOSTS = ("steamcdn-a.akamaihd.net",)
_ALLOWED_SUFFIXES = (".steamstatic.com",)

_USER_AGENT = "cs-net/1.0 (+demo library; steam avatar fetch)"

#: magic bytes -> extension.  The extension comes from the bytes, never from the
#: URL: Steam serves the same picture from paths ending ``_full.jpg`` and
#: ``_medium.jpg``, and an animated avatar is a ``.gif`` under either.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
)


class AvatarError(RuntimeError):
    """A lookup that failed for a reason worth showing the user."""


# ── steam id plumbing ────────────────────────────────────────────────────────


def account_id(steamid: Any) -> Optional[int]:
    """SteamID64 -> Steam account id, or None when it is not one."""
    text = str(steamid or "").strip()
    if not text.isdigit() or not 15 <= len(text) <= 20:
        return None
    value = int(text) - STEAM_ID64_BASE
    return value if value > 0 else None


def allowed_avatar_url(value: Any) -> str:
    """Return ``value`` only when it is an https URL on a Steam CDN."""
    url = str(value or "").strip()
    if url.startswith("//"):
        url = f"https:{url}"
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if parsed.scheme != "https":
        return ""
    host = (parsed.hostname or "").lower()
    ok = host in _ALLOWED_HOSTS or any(host.endswith(s) for s in _ALLOWED_SUFFIXES)
    return url if ok else ""


def image_ext(data: bytes) -> Optional[str]:
    """Sniff the image type, or None when the bytes are not an image we know."""
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def avatar_path(steamid: Any, ext: str) -> Path:
    """Where this player's avatar is cached.  One file per SteamID."""
    text = str(steamid or "").strip()
    if not text.isdigit():
        raise AvatarError("SteamID 必须是数字")
    return config.AVATAR_DIR / f"{text}.{ext}"


def local_avatar(steamid: Any) -> Optional[Path]:
    """The cached avatar file for ``steamid``, if one exists."""
    row = db.get_avatar(str(steamid))
    path = (row or {}).get("file_path")
    if path and Path(path).is_file():
        return Path(path)
    # The database is authoritative, but a file can outlive a wiped row.
    text = str(steamid or "").strip()
    if not text.isdigit():
        return None
    for candidate in sorted(config.AVATAR_DIR.glob(f"{text}.*")):
        if candidate.suffix != ".part" and candidate.is_file():
            return candidate
    return None


# ── the network ──────────────────────────────────────────────────────────────


def _http_get(url: str, *, accept: str = "application/json",
              timeout: float = DEFAULT_TIMEOUT,
              limit: int = MAX_AVATAR_BYTES) -> bytes:
    """One GET.  Isolated so the tests can replace it and stay offline."""
    req = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept": accept}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise AvatarError(f"响应超过 {limit // 1024} KB，已放弃")
    return data


def fetch_summary(steamid: Any, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Persona name + avatar URL for one SteamID64.  Raises :class:`AvatarError`."""
    acc = account_id(steamid)
    if acc is None:
        raise AvatarError("不是有效的 SteamID64")
    url = MINIPROFILE_URL.format(account_id=acc)
    try:
        raw = _http_get(url, accept="application/json", timeout=timeout)
    except AvatarError:
        raise
    except urllib.error.HTTPError as exc:
        raise AvatarError(f"Steam 返回 HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", None) or exc
        raise AvatarError(f"连不上 steamcommunity.com（{reason}）") from exc

    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except ValueError as exc:
        raise AvatarError(f"Steam 返回的不是 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise AvatarError("Steam 返回的结构不认识")

    avatar = allowed_avatar_url(payload.get("avatar_url") or payload.get("avatarfull"))
    if not avatar:
        raise AvatarError("Steam 没有返回可用的头像地址")
    return {
        "steamid": str(steamid).strip(),
        "persona_name": str(payload.get("persona_name") or "").strip(),
        "avatar_url": avatar,
    }


def _write_atomic(steamid: str, ext: str, data: bytes) -> Path:
    config.AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    dest = avatar_path(steamid, ext)
    tmp = dest.with_name(f"{dest.name}.part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    # A previous copy may have had a different extension (jpg -> gif).
    for other in config.AVATAR_DIR.glob(f"{steamid}.*"):
        if other != dest and other.suffix != ".part":
            try:
                other.unlink()
            except OSError:
                pass
    return dest


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:200] or type(exc).__name__


def fetch_and_store(steamid: Any, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Look one player up and cache the picture.  Never raises.

    Returns ``{steamid, ok, persona_name?, path?, error?}``.  The failure is also
    written to the database, so the UI can explain *why* a face is missing
    instead of just showing an initial forever.
    """
    sid = str(steamid or "").strip()
    try:
        summary = fetch_summary(sid, timeout=timeout)
        try:
            data = _http_get(summary["avatar_url"], accept="image/*", timeout=timeout)
        except AvatarError:
            raise
        except urllib.error.HTTPError as exc:
            raise AvatarError(f"下载头像失败：HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            reason = getattr(exc, "reason", None) or exc
            raise AvatarError(f"下载头像失败：{reason}") from exc
        ext = image_ext(data)
        if not ext:
            raise AvatarError("下载到的内容不是图片")
        path = _write_atomic(sid, ext, data)
    except (AvatarError, OSError) as exc:
        message = _one_line(exc)
        db.upsert_avatar(sid, status=db.AVATAR_ERROR, error_msg=message, bump_attempts=True)
        return {"steamid": sid, "ok": False, "error": message}

    db.upsert_avatar(
        sid,
        persona_name=summary["persona_name"] or None,
        avatar_url=summary["avatar_url"],
        file_path=str(path),
        status=db.AVATAR_OK,
        error_msg=None,
        bump_attempts=True,
    )
    return {
        "steamid": sid,
        "ok": True,
        "persona_name": summary["persona_name"],
        "path": str(path),
    }


# ── batch sync ───────────────────────────────────────────────────────────────


def sync(
    steamids: Optional[Iterable[Any]] = None,
    *,
    force: bool = False,
    job: Optional[jobs.Job] = None,
    delay_ms: Optional[int] = None,
) -> dict[str, Any]:
    """Fetch every missing or stale avatar.

    ``steamids=None`` means "every player in the library", which is what makes
    the job cheap to coalesce: it re-reads the roster when it starts, so one job
    covers demos extracted while it was queued.

    ``force=True`` ignores the cache and re-downloads, and also overrides the
    ``fetch_avatars`` setting -- that is the "I turned it on just now, go" path.
    """
    cfg = config.Config()
    if not cfg.get("fetch_avatars", True) and not force:
        return {"total": 0, "ok": 0, "failed": 0, "skipped": True,
                "note": "头像获取已关闭"}

    if steamids is None:
        candidates = db.distinct_steamids()
    else:
        candidates = [str(s).strip() for s in steamids if str(s or "").strip()]

    todo = db.stale_steamids(
        candidates,
        ttl_days=float(cfg.get("avatar_ttl_days") or 7),
        error_retry_hours=float(cfg.get("avatar_error_retry_hours") or 6),
        force=force,
    )
    delay = (delay_ms if delay_ms is not None
             else int(cfg.get("avatar_delay_ms") or DEFAULT_DELAY_MS)) / 1000.0

    ok = failed = 0
    errors: list[str] = []
    for i, sid in enumerate(todo):
        if job:
            job.set_step(f"获取头像 {i + 1}/{len(todo)}…", i / len(todo))
        result = fetch_and_store(sid)
        if result["ok"]:
            ok += 1
        else:
            failed += 1
            if len(errors) < 10:
                errors.append(f"{sid}：{result['error']}")
        if delay and i < len(todo) - 1:
            time.sleep(delay)

    if job:
        job.set_step("完成", 1.0)
    return {"total": len(todo), "ok": ok, "failed": failed, "skipped": False,
            "errors": errors}


# ── job plumbing ─────────────────────────────────────────────────────────────

#: One avatar job at a time, however many demos finish at once.  A scan that
#: registers 60 demos fires 60 "extraction done" triggers; without coalescing
#: that is 60 queued jobs each looking up the same roster.
_queue_lock = threading.Lock()
_again = False


def _submit(steamids: Optional[list[str]], force: bool) -> str:
    job = jobs.manager.submit(
        "avatars",
        "同步 Steam 头像",
        lambda j: _run(j, steamids, force),
    )
    return job.id


def _run(job: jobs.Job, steamids: Optional[list[str]], force: bool) -> dict[str, Any]:
    global _again
    try:
        return sync(steamids, force=force, job=job)
    finally:
        with _queue_lock:
            again, _again = _again, False
        if again:
            # Submitted directly rather than through queue_sync: this job is
            # still "running" as far as the manager is concerned, so the active
            # check would suppress the very job it is trying to schedule.
            _submit(None, False)


def queue_sync(steamids: Optional[Iterable[Any]] = None, *,
               force: bool = False) -> Optional[str]:
    """Queue an avatar sync, or fold into the one already queued.

    Returns the job id, or None when a sync was already in flight -- in which
    case a fresh full sync is scheduled to run right after it, so the caller
    never has to retry.
    """
    global _again
    clean = None
    if steamids is not None:
        clean = [str(s).strip() for s in steamids if str(s or "").strip()][:200]
    with _queue_lock:
        if jobs.manager.has_active("avatars"):
            _again = True
            return None
        return _submit(clean, force)
