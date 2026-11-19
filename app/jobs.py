"""
A small in-process job queue.

Extracting a demo takes anywhere from seconds to minutes, so every long
operation (extraction, directory scan, thumbnail rebuild) runs on a worker
thread and the browser polls for progress.  This mirrors the pattern the
existing visualizer uses for ``/api/load``, but generalised so the library can
have several kinds of job and a real queue behind them.

Progress is coarse by design: ``demo_extract`` does not report progress from
inside its parse, so a job reports the *step* it is on plus elapsed time rather
than inventing a percentage.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, Callable, Optional

#: How many finished jobs to keep for the UI's activity list.
_HISTORY = 40

#: Exceptions that mean "the process is going away", which a worker must not
#: convert into a job failure.
_FATAL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class Job:
    """One unit of background work."""

    __slots__ = (
        "id", "kind", "label", "subject", "state", "step", "error", "error_kind",
        "result", "created_at", "started_at", "finished_at", "progress",
    )

    def __init__(self, kind: str, label: str, subject: Any = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.label = label
        #: Opaque key naming *what* this job works on, e.g. ``("demo", 7, "extract")``.
        #: Two jobs with the same subject do the same thing to the same thing, so
        #: the queue can refuse the second one -- see :meth:`JobManager.submit_unique`.
        self.subject = subject
        self.state = "queued"          # queued | running | done | error | cancelled
        self.step = "排队中…"
        self.error: Optional[str] = None
        #: 一类异常可能自带 `kind`（`app.ai.models.ModelUnavailable.kind == "models"`），
        #: 界面靠它把"没下权重"和"数据不合规"分成两种提示，而不是只显示一段红字。
        self.error_kind: Optional[str] = None
        self.result: Any = None
        self.progress: Optional[float] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    # -- progress API handed to the worker ---------------------------------
    def set_step(self, step: str, progress: Optional[float] = None) -> None:
        self.step = step
        if progress is not None:
            self.progress = max(0.0, min(1.0, float(progress)))

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        end = self.finished_at or now
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "subject": list(self.subject) if isinstance(self.subject, tuple) else self.subject,
            "state": self.state,
            "step": self.step,
            "progress": self.progress,
            "error": self.error,
            "error_kind": self.error_kind,
            "result": self.result,
            "elapsed": round(end - (self.started_at or self.created_at), 1),
            "created_at": self.created_at,
        }


class JobManager:
    """Fixed-size worker pool with a FIFO queue and a bounded history."""

    def __init__(self, workers: int = 1) -> None:
        self._workers = max(1, int(workers))
        self._lock = threading.Lock()
        self._queue: deque[tuple[Job, Callable[[Job], Any]]] = deque()
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._threads: list[threading.Thread] = []
        self._wake = threading.Event()
        self._stopping = False
        self._running = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Start the worker threads. Safe to call again after :meth:`stop`.

        The restartability matters: the test suite builds and tears down the
        whole app several times in one process, and a manager that could only
        ever run once would leave every later job stuck at "queued".
        """
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            if self._threads:
                return
            self._stopping = False
            for i in range(self._workers):
                t = threading.Thread(target=self._loop, name=f"job-worker-{i}", daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self, timeout: float = 3.0) -> None:
        """Ask the workers to exit and wait briefly for them.

        A job already in flight is *not* interrupted -- ``_loop`` only checks
        the flag between jobs -- so ``timeout`` bounds the wait rather than
        guaranteeing the threads are gone. They are daemons, so a worker still
        draining a long extraction cannot hold up interpreter shutdown.

        Jobs still sitting in the queue are dropped and marked cancelled.  They
        were never started, and keeping them would mean a later :meth:`start`
        silently resumes work that belongs to whoever queued it -- which in the
        test suite is a *different* scratch data dir, so a straggler extraction
        would land on the next group's database and flip a freshly seeded row to
        ``error``.  That is a real bug, not just a test artefact: "stop" that
        leaves pending work behind is a lie about what happened.
        """
        self._stopping = True
        self._wake.set()
        with self._lock:
            dropped = [job for job, _ in self._queue]
            self._queue.clear()
            for job in dropped:
                job.state = "cancelled"
                job.step = "已取消"
                job.error = "服务已停止"
                job.finished_at = time.time()
            threads = list(self._threads)
        for t in threads:  # joined outside the lock: _loop takes it too
            if t.is_alive():
                t.join(timeout=timeout)
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]

    # -- public API --------------------------------------------------------
    def submit(self, kind: str, label: str, fn: Callable[[Job], Any],
               *, subject: Any = None) -> Job:
        job = self._submit(kind, label, fn, subject=subject, unique=False)
        assert job is not None      # unique=False never refuses
        return job

    def submit_unique(self, kind: str, label: str, fn: Callable[[Job], Any],
                      *, subject: Any) -> Optional[Job]:
        """Submit unless something for ``subject`` is already queued or running.

        Returns ``None`` when the work is already spoken for.  This is what keeps
        one demo from being extracted twice: a manual 扫描 recovers every row
        that has no ``.csz`` yet, which includes the row a watcher poll queued
        two seconds ago, and handing that id to :func:`app.ingest.queue_extraction`
        again used to start a second full parse of the same file (58 s + 76 s for
        one demo in the report that found this).

        The check and the insert happen under one lock, so two threads racing to
        queue the same demo cannot both win.
        """
        return self._submit(kind, label, fn, subject=subject, unique=True)

    def _submit(self, kind: str, label: str, fn: Callable[[Job], Any],
                *, subject: Any, unique: bool) -> Optional[Job]:
        job = Job(kind, label, subject)
        with self._lock:
            if unique and subject is not None and self._active_for(subject):
                return None
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > _HISTORY:
                old = self._order.popleft()
                stale = self._jobs.get(old)
                # Never evict something still in flight.
                if stale is not None and stale.state in ("queued", "running"):
                    self._order.append(old)
                    break
                self._jobs.pop(old, None)
            self._queue.append((job, fn))
        self._wake.set()
        return job

    def _active_for(self, subject: Any) -> bool:
        """Is a job for ``subject`` queued or running? (Call with ``_lock`` held.)"""
        return any(
            j.subject == subject and j.state in ("queued", "running")
            for j in self._jobs.values()
        )

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def find_active(self, subject: Any) -> Optional[Job]:
        """The queued/running job for ``subject``, if any.

        Used by callers that do not want to *replace* the work but to **watch**
        it: two browsers (or the replay page plus the stats tab) asking for the
        same round share one inference and one set of progress numbers instead
        of queueing a second identical job.
        """
        if subject is None:
            return None
        with self._lock:
            for j in self._jobs.values():
                if j.subject == subject and j.state in ("queued", "running"):
                    return j
        return None

    def list(self, active_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [self._jobs[i] for i in self._order if i in self._jobs]
        if active_only:
            jobs = [j for j in jobs if j.state in ("queued", "running")]
        return [j.to_dict() for j in reversed(jobs)]

    def has_active(self, kind: Optional[str] = None) -> bool:
        with self._lock:
            return any(
                j.state in ("queued", "running") and (kind is None or j.kind == kind)
                for j in self._jobs.values()
            )

    def pending_count(self) -> int:
        with self._lock:
            return sum(
                1 for j in self._jobs.values() if j.state in ("queued", "running")
            )

    def clear_finished(self) -> int:
        with self._lock:
            done = [
                jid for jid, j in self._jobs.items()
                if j.state not in ("queued", "running")
            ]
            for jid in done:
                self._jobs.pop(jid, None)
            self._order = deque(i for i in self._order if i in self._jobs)
        return len(done)

    # -- worker ------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stopping:
            item: Optional[tuple[Job, Callable[[Job], Any]]] = None
            with self._lock:
                if self._queue:
                    item = self._queue.popleft()
                    self._running += 1
            if item is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue

            job, fn = item
            job.state = "running"
            job.started_at = time.time()
            job.set_step("开始…")
            try:
                job.result = fn(job)
                if job.state != "cancelled":
                    job.state = "done"
                    job.set_step(job.step if job.step != "开始…" else "完成", 1.0)
            except _FATAL:
                # Never swallow a real shutdown request.
                job.state = "error"
                job.error = "已中断"
                raise
            except BaseException as exc:  # noqa: BLE001 - surfaced to the UI
                # BaseException, not Exception, on purpose: demoparser2 is Rust
                # and panics on a malformed demo, and PyO3 surfaces that as
                # PanicException, which inherits from BaseException.  Catching
                # only Exception would let one corrupt .dem kill the worker
                # thread and wedge the queue with every later job stuck at
                # "queued" -- with the failing job left reading "running".
                job.state = "error"
                job.error = _concise(exc)
                # 异常类可以自带 `kind`（AI 那边是 models / features），界面用它
                # 区分"去下权重"和"这场数据算不了"，而不是只显示一段红字。
                job.error_kind = getattr(exc, "kind", None)
                job.step = "失败"
                traceback.print_exc()
            finally:
                job.finished_at = time.time()
                with self._lock:
                    self._running -= 1


def _concise(exc: BaseException, limit: int = 400) -> str:
    """Compress an exception into one actionable line for a toast.

    ``demo_extract``'s ``SideChannelError`` embeds a whole Go panic stack;
    pasting tens of KB into a toast helps nobody.
    """
    msg = str(exc).strip() or type(exc).__name__
    lines = [ln.rstrip() for ln in msg.splitlines() if ln.strip()]
    keep: list[str] = []
    for ln in lines:
        if ln.startswith(("goroutine ", "created by ", "runtime.", "reflect.")):
            break
        if "/pkg/mod/" in ln or ln.startswith(("panic(", "github.com/")):
            continue
        keep.append(ln)
        if len(keep) >= 4:
            break
    text = "\n".join(keep) if keep else (lines[0] if lines else type(exc).__name__)
    if len(text) > limit:
        text = text[:limit].rstrip() + " …"
    if "unable to find existing entity" in msg or "unexpected team" in msg:
        text += (
            "\n提示：火/烟辅助程序（sidechannel）的 demoinfocs 版本过旧，"
            "需要 v5.3.0（2026-11-11 之后录的 demo 尤其）。"
            "重新编译：bash tools/demo_extract/build_sidechannel.sh"
        )
    return text


#: Process-wide queue. One worker by default: extraction is CPU- and
#: memory-heavy and parallel runs mostly fight each other.
manager = JobManager(workers=1)
