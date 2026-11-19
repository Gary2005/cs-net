"""
Watch demo download directories and auto-import what appears.

Implementation choice: **polling, not ``watchdog``.**  The project's conda env
does not have ``watchdog`` installed, and adding a compiled dependency to watch
a handful of directories is a poor trade.  A poll every few seconds that stats
the tree is cheap, behaves identically on Windows and Linux, and -- unlike
filesystem events -- cannot miss files that were dropped in while the server
was not running.

Two guards matter more than the polling itself:

* **Stability gate.**  A demo being downloaded grows while it is written.
  A file is only registered once its ``(size, mtime)`` has been identical for
  two consecutive polls, so a half-written 400 MB demo is never ingested.
* **Stat cache.**  ``register_demo`` would otherwise re-hash every file on every
  poll.  The watcher keeps its own ``{path: (size, mtime_ns)}`` snapshot and
  only touches the library when something actually changed.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import config, db, ingest

#: Exceptions that mean "the process is going away"; the poll loop must not
#: treat those as a recoverable scan error.
_FATAL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class DemoWatcher:
    """Background poller over the configured watch roots."""

    def __init__(self, on_change: Optional[Callable[[str], None]] = None) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._on_change = on_change

        # path -> (size, mtime_ns) as of the previous poll
        self._snapshot: dict[str, tuple[int, int]] = {}
        # path -> (size, mtime_ns) seen once but not yet confirmed stable
        self._pending: dict[str, tuple[int, int]] = {}

        self._last_scan: Optional[float] = None
        self._last_result: dict[str, Any] = {}
        self._last_error: Optional[str] = None
        self._scanning = False

        #: Demo ids this process has already auto-queued.  See the comment in
        #: :meth:`poll` -- this is what stops a queued-but-not-yet-started demo
        #: from being re-queued on every poll.
        self._queued_ids: set[int] = set()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="demo-watcher", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            t = self._thread
        if t and t.is_alive():
            t.join(timeout=3.0)

    def restart(self) -> None:
        self.stop()
        self.start()

    def poke(self) -> None:
        """Ask for an immediate poll (used after a config change)."""
        self._wake.set()

    # -- status ------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        cfg = config.Config()
        alive = bool(self._thread and self._thread.is_alive())
        enabled = bool(cfg.get("watch_enabled", True))
        roots = cfg.get("watch_roots") or []
        return {
            # `running` is just "the thread is alive".  It stays alive while
            # watching is switched off so that re-enabling takes effect on the
            # next tick instead of needing a restart -- hence `active`, which is
            # the one to show a user.
            "running": alive,
            "active": alive and enabled and any(r.get("enabled", True) for r in roots),
            "enabled": enabled,
            "interval": cfg.get("watch_interval_sec", 5.0),
            "roots": roots,
            "last_scan": self._last_scan,
            "last_result": self._last_result,
            "last_error": self._last_error,
            "scanning": self._scanning,
            "pending": len(self._pending),
        }

    # -- the poll ----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            cfg = config.Config()
            interval = float(cfg.get("watch_interval_sec", 5.0) or 5.0)
            interval = max(1.0, min(interval, 300.0))

            if cfg.get("watch_enabled", True) and (cfg.get("watch_roots") or []):
                try:
                    self.poll(cfg)
                except _FATAL:
                    raise
                except BaseException as exc:  # noqa: BLE001 - never kill the thread
                    # BaseException: a demoparser panic on one corrupt file is
                    # a PanicException, which is not an Exception.  Catching
                    # only Exception here would let a single bad demo stop
                    # directory watching for the rest of the session, silently.
                    self._last_error = f"{type(exc).__name__}: {exc}"

            self._wake.wait(timeout=interval)
            self._wake.clear()

    def _iter_files(self, cfg: dict) -> dict[str, tuple[int, int]]:
        """Stat every candidate under every enabled root."""
        found: dict[str, tuple[int, int]] = {}
        for root in cfg.get("watch_roots") or []:
            if not root.get("enabled", True):
                continue
            raw = root.get("path")
            if not raw:
                continue
            base = Path(config.expand(raw))
            if not base.is_dir():
                continue
            depth = int(root.get("depth", 2) or 0)
            try:
                for path in ingest._iter_candidates(base, depth):
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    if st.st_size <= 0:
                        continue  # still being created
                    found[str(path)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                continue
        return found

    def poll(self, cfg: Optional[config.Config] = None, *, force: bool = False) -> dict[str, Any]:
        """One pass: detect new/changed files, register stable ones, import them.

        ``force=True`` skips the stability gate (used by the manual
        "立即扫描" button, where the user is asserting the files are complete).
        """
        cfg = cfg or config.Config()
        self._scanning = True
        try:
            current = self._iter_files(cfg)
            known = self._snapshot

            # -- what is new or changed? -----------------------------------
            changed = {
                p: sig for p, sig in current.items()
                if known.get(p) != sig
            }

            # -- stability gate --------------------------------------------
            stable: list[str] = []
            next_pending: dict[str, tuple[int, int]] = {}
            for p, sig in changed.items():
                if force or self._pending.get(p) == sig:
                    stable.append(p)
                else:
                    next_pending[p] = sig
            self._pending = next_pending

            # -- disappeared ------------------------------------------------
            gone = [p for p in known if p not in current]
            marked_missing = 0
            for p in gone:
                row = db.get_by_path(p)
                if row and row["status"] != db.STATUS_MISSING:
                    db.update_demo(row["id"], status=db.STATUS_MISSING)
                    marked_missing += 1

            # -- register + queue -------------------------------------------
            new_ids: list[int] = []
            errors: list[str] = []
            for p in stable:
                path = Path(p)
                try:
                    if path.suffix.lower() == ".zip":
                        extracted, err = ingest.extract_zip_demos(path)
                        if err:
                            errors.append(f"{path.name}：{err}")
                        targets = extracted
                    else:
                        targets = [path]

                    for dem in targets:
                        try:
                            demo_id, is_new = ingest.register_demo(
                                dem, watch_root=str(dem.parent)
                            )
                        except (OSError, ValueError) as exc:
                            errors.append(f"{dem.name}：{exc}")
                            continue
                        if is_new:
                            new_ids.append(demo_id)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{path.name}：{type(exc).__name__}: {exc}")

            self._snapshot = current

            auto = bool(cfg.get("auto_extract", True))
            queued: list[str] = []
            if auto:
                # Newly discovered files, plus anything left unfinished by a
                # restart (a row with no .csz that no longer has a file event to
                # trigger it).  Without the second half, a demo discovered in a
                # session that died before auto-extract reached it would sit at
                # "待入库" until someone happened to click 扫描.
                #
                # `_queued_ids` is what makes this safe.  A row stays
                # "discovered" from the moment it is queued until a worker
                # actually picks it up, so without the guard the poll would
                # re-queue the same demo every few seconds while a long
                # extraction sat in the queue.  Once a demo has been queued in
                # this process it is never auto-queued again -- a retry after a
                # failure is the user's call, via the UI's 提取 button.
                #
                # `dict.fromkeys` de-duplicates: a file discovered on this very
                # poll is in `new_ids` *and* in `unfinished_ids`, and queuing it
                # twice would start two extractions of the same demo.
                candidates = dict.fromkeys((*new_ids, *db.unfinished_ids()))
                todo = [i for i in candidates if i not in self._queued_ids]
                if todo:
                    self._queued_ids.update(todo)
                    queued = ingest.queue_extraction(todo)

            self._last_scan = time.time()
            self._last_result = {
                "found": len(current),
                "new": len(new_ids),
                "queued": len(queued),
                "missing": marked_missing,
                "errors": errors[:20],
            }
            self._last_error = None
            if (new_ids or marked_missing) and self._on_change:
                try:
                    self._on_change("changed")
                except Exception:  # noqa: BLE001
                    pass
            return self._last_result
        finally:
            self._scanning = False


#: Process-wide watcher. The API layer starts it.
watcher = DemoWatcher()
