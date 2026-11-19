"""
End-to-end pipeline: scrape HLTV → download demo → parse to .csz.

Connects to your **real Chrome** via CDP (Chrome DevTools Protocol)
so Cloudflare sees a real browser fingerprint with your existing cookies.
Playwright's built-in Chromium is only used as a last-resort fallback.

The crawler is carried over verbatim from cs-net-dev.  The only v5 change is
the data-conversion step: demos are parsed with ``tools/demo_extract`` and
written as ``.csz`` (columnar binary + zstd) instead of the old ``.json.gz``.
Sampling defaults to **every tick** (~64 Hz); ``--interval`` / ``--stride``
thin it out.

Usage:
    python -m scraper --setup          # one-time: login to HLTV
    python -m scraper --count 10       # scrape + download + parse 10 demos
    python -m scraper --auto-launch    # auto-start Chrome if not running
    python -m scraper --count 10 --interval 0.25   # 4 Hz, smaller files
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import rarfile

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None  # type: ignore[assignment]

# ── v5 data conversion ──────────────────────────────────────────
# Everything above is the crawler, unchanged.  Only the parse/save step
# differs from cs-net-dev: v5 stores ``.csz`` via ``tools/demo_extract``.
# Put the project root and ``tools/`` on sys.path so ``demo_extract``
# imports without needing PYTHONPATH to be set by hand.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _PROJECT_ROOT / "tools"
for _p in (str(_PROJECT_ROOT), str(_TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_extract import codec  # noqa: E402
from demo_extract import extract as extract_demo  # noqa: E402


# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

HLTV_RESULTS = "https://www.hltv.org/results"

DEFAULT_OUTPUT_DIR = "data/demos"
DEFAULT_CDP_URL = "http://127.0.0.1:9222"

# How many fresh (not-yet-downloaded) candidates to accumulate
# before the scraper stops fetching new pages.
CANDIDATE_THRESHOLD = 64

_SYSTEM = __import__("platform").system()

if _SYSTEM == "Windows":
    DEFAULT_CHROME_PATH = (
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    )
    DEFAULT_USER_DATA_DIR = str(Path.home() / "chrome-hltv")
elif _SYSTEM == "Darwin":
    DEFAULT_CHROME_PATH = (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    DEFAULT_USER_DATA_DIR = "/tmp/chrome-hltv"
else:
    # Linux: try multiple common Chrome paths
    _LINUX_CHROME_CANDIDATES = [
        "google-chrome-stable", "google-chrome",
        "chromium-browser", "chromium",
        "/usr/bin/google-chrome-stable", "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser", "/usr/bin/chromium",
        "/snap/bin/chromium",
    ]
    _found = None
    for _c in _LINUX_CHROME_CANDIDATES:
        _found = shutil.which(_c)
        if _found:
            break
    DEFAULT_CHROME_PATH = _found or "google-chrome"
    DEFAULT_USER_DATA_DIR = "/tmp/chrome-hltv"


# ────────────────────────────────────────────────────────────────
# Error helpers
# ────────────────────────────────────────────────────────────────

# demoparser2 is a Rust extension: its panics surface as
# `pyo3_runtime.PanicException`, which derives from BaseException, not
# Exception.  A handler that catches only Exception lets one bad demo kill the
# whole scrape, so every per-demo boundary below catches BaseException.
PANIC_TYPES = ("PanicException",)


def _exc_note(exc: BaseException) -> str:
    """One-line failure description, naming demoparser2 panics clearly."""
    kind = type(exc).__name__
    detail = str(exc).strip() or traceback.format_exc().strip().splitlines()[-1]
    if kind in PANIC_TYPES:
        return (f"{kind} in demoparser2 (upstream bug, not a corrupt file): "
                f"{detail}")
    return f"{kind}: {detail}"


# ────────────────────────────────────────────────────────────────
# CDP helpers (connect to real Chrome)
# ────────────────────────────────────────────────────────────────

def _resolve_ws_endpoint(cdp_url: str) -> str:
    """Convert a http:// CDP URL to a ws:// endpoint."""
    if cdp_url.startswith(("ws://", "wss://")):
        return cdp_url
    base = cdp_url.rstrip("/")
    with urllib.request.urlopen(f"{base}/json/version", timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ws_url = data.get("webSocketDebuggerUrl")
    if not ws_url:
        raise RuntimeError(f"No webSocketDebuggerUrl at {base}/json/version")
    return ws_url


def _launch_chrome(
    chrome_path: str = DEFAULT_CHROME_PATH,
    user_data_dir: str = DEFAULT_USER_DATA_DIR,
    port: int = 9222,
    headless: bool = False,
) -> subprocess.Popen | None:
    """Launch Chrome with remote debugging enabled."""
    chrome = Path(chrome_path)
    if not chrome.exists():
        print(f"Chrome not found at: {chrome_path}", file=sys.stderr)
        return None

    udd = Path(user_data_dir)
    udd.mkdir(parents=True, exist_ok=True)

    # Clean stale lock files
    for lock_name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
        lock = udd / lock_name
        if lock.exists():
            try:
                lock.unlink()
            except OSError:
                pass

    # Kill stale Chrome using this profile
    if _SYSTEM == "Windows":
        subprocess.run(
            ["taskkill", "/F", "/IM", "chrome.exe"], capture_output=True
        )
    else:
        subprocess.run(
            ["pkill", "-f", f"--user-data-dir={user_data_dir}"],
            capture_output=True,
        )
    time.sleep(1)

    cmd = [
        str(chrome),
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if headless:
        cmd.append("--headless=new")

    return subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


# ────────────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────────────

class DemoPipeline:
    """Scrape, download, and parse CS2 demos from HLTV in one pass.

    Connects to your **real Chrome** via CDP so Cloudflare sees
    a genuine browser fingerprint.  Playwright's built-in Chromium
    is only used as a fallback when CDP is unavailable and
    ``auto_launch`` is off.
    """

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        user_data_dir: str = DEFAULT_USER_DATA_DIR,
        cdp_url: str = DEFAULT_CDP_URL,
        chrome_path: str = DEFAULT_CHROME_PATH,
        auto_launch: bool = False,
        headless: bool = False,
        workers: int = 2,
        page_timeout: int = 60_000,
        download_timeout: int = 900_000,
    ):
        self.output_dir = Path(output_dir)
        self.user_data_dir = user_data_dir
        self.cdp_url = cdp_url
        self.chrome_path = chrome_path
        self.auto_launch = auto_launch
        self.headless = headless
        self.workers = min(workers, 4)
        self.page_timeout = page_timeout          # ms
        self.download_timeout = download_timeout  # ms

        # Directory layout
        self.csz_dir = self.output_dir / "csz"
        self.tmp_dir = self.output_dir / "tmp"
        self.done_file = self.output_dir / "done.json"
        self.candidates_file = self.output_dir / "candidates.json"

        for d in [self.output_dir, self.csz_dir, self.tmp_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Internal Chrome process handle (for CDP auto-launch)
        self._chrome_proc: subprocess.Popen | None = None

    # ── public API ──────────────────────────────────────────────

    def run(
        self,
        count: int = 10,
        interval: float | None = None,
        stride: int | None = None,
        min_stars: int = 0,
        verbose: bool = False,
    ) -> int:
        """
        Scrape, download, and parse `count` demos.

        Sampling for the v5 ``.csz`` conversion: ``interval=None`` (default)
        samples every tick; ``interval=0.25`` gives 4 Hz; ``stride=N`` samples
        every Nth tick and takes precedence over `interval`.

        Candidate URLs are persisted to disk so scraping can resume
        from the last page offset across runs.  New pages are fetched
        incrementally — only when the candidate list runs empty.

        Downloads run sequentially on the main thread (Cloudflare-safe).
        Parse runs in parallel — each demo starts parsing immediately
        after download, overlapping with the next download.

        Returns the number of successfully processed demos.
        """
        if sync_playwright is None:
            raise RuntimeError("playwright not installed.")

        # Clean up stale Playwright artifacts from previous runs
        self._clean_playwright_artifacts()

        done = self._load_done()

        if verbose:
            print(f"Output dir      : {self.output_dir}")
            print(f"User data dir  : {self.user_data_dir}")
            print(f"CDP URL         : {self.cdp_url}")
            print(f"Chrome path     : {self.chrome_path}")
            print(f"Done already    : {len(done)}")
            print(f"Page timeout    : {self.page_timeout // 1000}s")
            print(f"Download timeout: {self.download_timeout // 1000}s")
            print()

        # ── Phase 1: Load / refill candidates ──────────────────────
        with sync_playwright() as p:
            browser, is_own = self._connect_browser(p)
            try:
                context = browser.contexts[0] if browser.contexts else browser.new_context()

                last_offset, candidates = self._load_candidates()

                # Filter out already-done matches and persist the clean list
                candidates = [
                    c for c in candidates if c["match_id"] not in done
                ]
                self._save_candidates(last_offset, candidates)

                # If no candidates, scrape fresh pages
                if not candidates:
                    print("Candidate list empty — scraping fresh pages…")
                    last_offset, candidates = self._refill_candidates(
                        context, last_offset, min_stars, done
                    )
                    # Re-filter and persist
                    candidates = [
                        c for c in candidates if c["match_id"] not in done
                    ]
                    self._save_candidates(last_offset, candidates)

                if not candidates:
                    print("\nNo new matches to download.")
                    return 0

                # ── Phase 2: Download + Parse (pipelined) ─────────
                processed = 0
                lock = threading.Lock()
                parse_futures: dict[concurrent.futures.Future, dict] = {}
                _slots = threading.BoundedSemaphore(self.workers * 3)

                # Timing / ETA state
                dl_times: list[float] = []        # last N download durations
                t0 = time.time()                   # pipeline start
                QUEUE_MAX = self.workers * 3

                bar_width = 30

                def _eta_str() -> str:
                    """Build ETA string from rolling download average."""
                    if len(dl_times) < 2:
                        return ""
                    avg_dl = sum(dl_times) / len(dl_times)
                    remaining = count - d_idx
                    dl_left = remaining * avg_dl
                    q = len(parse_futures)
                    parse_lag = (q / max(self.workers, 1)) * avg_dl * 1.5
                    eta_sec = dl_left + parse_lag
                    if eta_sec < 60:
                        return f" | ETA: {eta_sec:.0f}s"
                    elif eta_sec < 3600:
                        return f" | ETA: {eta_sec/60:.0f}m{eta_sec%60:.0f}s"
                    else:
                        h = int(eta_sec / 3600)
                        m = int((eta_sec % 3600) / 60)
                        return f" | ETA: {h}h{m}m"

                def _progress_bar() -> str:
                    n = bar_width * d_idx // max(count, 1)
                    return f"[{'█' * n}{'░' * (bar_width - n)}]"

                print(f"Downloading {count} matches "
                      f"(parse: {self.workers} workers, "
                      f"queue: max {QUEUE_MAX})…\n")

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.workers
                ) as executor:
                    d_idx = 0
                    while d_idx < count:
                        # Refill if candidate list is exhausted
                        if not candidates:
                            print("\n  🔄 candidate list empty — "
                                  "scraping more pages…")
                            last_offset, candidates = self._refill_candidates(
                                context, last_offset, min_stars, done
                            )
                            candidates = [
                                c for c in candidates
                                if c["match_id"] not in done
                            ]
                            self._save_candidates(last_offset, candidates)
                            if not candidates:
                                print("  ⚠ no more candidates available.")
                                break

                        match = candidates.pop(0)
                        self._save_candidates(last_offset, candidates)

                        mid = match["match_id"]
                        team_str = (
                            f"{match['team1']} vs {match['team2']}"
                        )

                        # Block if parse queue is full — slots are
                        # auto-released via future.add_done_callback
                        # from worker threads, so the main thread is
                        # never stuck waiting for itself to release.
                        q_now = len(parse_futures)
                        if q_now >= QUEUE_MAX:
                            print(f"  ⏳ parse queue full ({q_now}) — "
                                  f"pausing download until a parse "
                                  f"finishes…")
                        _slots.acquire()
                        # Report any parses that completed while we waited
                        processed += self._report_done_parses(
                            parse_futures, lock, done, verbose
                        )

                        # Download (main thread, sequential)
                        t_dl_start = time.time()
                        rar_path = self._download_one(context, match)
                        t_dl_end = time.time()
                        dl_dur = t_dl_end - t_dl_start

                        q = len(parse_futures)

                        if rar_path:
                            d_idx += 1  # only count successful downloads
                            dl_times.append(dl_dur)
                            if len(dl_times) > 20:
                                dl_times = dl_times[-20:]
                            bar = _progress_bar()
                            eta = _eta_str()
                            print(f"⬇ {bar} [{d_idx}/{count}] {team_str} "
                                  f"({dl_dur:.0f}s) [q:{q}]{eta}")
                            f = executor.submit(
                                self._parse_one, match, rar_path,
                                interval, stride, verbose,
                            )
                            # Auto-release slot when parse completes
                            # (worker thread).
                            f.add_done_callback(
                                lambda fut: _slots.release()
                            )
                            parse_futures[f] = match
                        else:
                            bar = _progress_bar()
                            print(f"❌ {bar} [{d_idx}/{count}] {team_str} "
                                  f"({dl_dur:.0f}s) — skipped")
                            _slots.release()

                        # Report completed parses so far (non-blocking)
                        processed += self._report_done_parses(
                            parse_futures, lock, done, verbose
                        )

                # Wait for remaining parses
                processed += self._await_remaining(
                    parse_futures, lock, done, verbose
                )

                # Save final state (remaining candidates)
                self._save_candidates(last_offset, candidates)

            finally:
                if is_own:
                    browser.close()

        # Cleanup auto-launched Chrome
        if self._chrome_proc is not None:
            self._chrome_proc.terminate()
            self._chrome_proc = None

        elapsed = time.time() - t0
        if elapsed < 60:
            ts = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            ts = f"{elapsed/60:.0f}m{elapsed%60:.0f}s"
        else:
            ts = f"{elapsed/3600:.1f}h"
        print(f"\nDone. {processed} parsed ({len(done)} total) in {ts}")
        return processed

    # ── parse-result helpers ─────────────────────────────────────

    def _report_done_parses(
        self,
        futures: dict,
        lock: threading.Lock,
        done: set,
        verbose: bool,
    ) -> int:
        """Print results for completed parse futures.  Returns success count."""
        n = 0
        for f in list(futures):
            if not f.done():
                continue
            match = futures.pop(f)
            n += self._log_parse_result(f, match, lock, done, verbose)
            # Slot is now released via future.add_done_callback (worker thread).
        return n

    def _await_remaining(
        self,
        futures: dict,
        lock: threading.Lock,
        done: set,
        verbose: bool,
    ) -> int:
        """Wait for remaining parse futures.  Returns success count."""
        n = 0
        for f in concurrent.futures.as_completed(list(futures)):
            match = futures.pop(f, None)
            if match is None:
                continue
            n += self._log_parse_result(f, match, lock, done, verbose)
            # Slot is now released via future.add_done_callback (worker thread).
        return n

    def _log_parse_result(
        self,
        future: concurrent.futures.Future,
        match: dict,
        lock: threading.Lock,
        done: set,
        verbose: bool,
    ) -> int:
        """Log a single parse result.  Returns 1 on success, 0 on failure."""
        mid = match["match_id"]
        display = (
            f"{match['team1']} vs {match['team2']} "
            f"({match.get('event','?')})"
        )
        try:
            result = future.result()
            with lock:
                if result:
                    self._mark_done(mid, result)
                    done.add(mid)
                    print(f"✅ {display}")
                    return 1
                else:
                    print(f"❌ {display}")
                    return 0
        except Exception as exc:
            with lock:
                print(f"💥 {display} — {exc}")
            return 0

    # ── setup ───────────────────────────────────────────────────

    def setup(self) -> None:
        """Launch Chrome and open HLTV for manual login / Cloudflare."""
        with sync_playwright() as p:
            browser, is_own = self._connect_browser(p)
            try:
                context = browser.contexts[0] if browser.contexts else browser.new_context()
                page = context.new_page()
                page.goto(
                    "https://www.hltv.org/results",
                    wait_until="domcontentloaded",
                    timeout=60000,
                )
                print("\n" + "=" * 50)
                print("  Solve Cloudflare & log in to HLTV in Chrome.")
                print("  Then press Enter to finish setup.")
                print("=" * 50 + "\n")
                input()
                page.close()
            finally:
                if is_own:
                    browser.close()

        print("Setup complete. Run: python -m scraper --count 10")

    # ── browser connection ───────────────────────────────────────

    def _connect_browser(self, p) -> tuple[Any, bool]:
        """
        Connect to a real Chrome via CDP.
        Returns (browser, is_owned_by_us).

        Tries CDP first (user's real Chrome with existing cookies).
        Falls back to auto-launching or Playwright's own Chromium.
        """
        # 1. Try CDP (connect to already-running real Chrome)
        try:
            ws = _resolve_ws_endpoint(self.cdp_url)
            browser = p.chromium.connect_over_cdp(ws)
            print(f"Connected to Chrome via CDP: {self.cdp_url}")
            return browser, False
        except Exception:
            pass

        # 2. Auto-launch real Chrome with remote debugging, then CDP
        if self.auto_launch:
            self._chrome_proc = _launch_chrome(
                self.chrome_path, self.user_data_dir, headless=self.headless
            )
            if self._chrome_proc is not None:
                time.sleep(3)
                try:
                    ws = _resolve_ws_endpoint(self.cdp_url)
                    browser = p.chromium.connect_over_cdp(ws)
                    print(f"Auto-launched Chrome + CDP: {self.cdp_url}")
                    return browser, True
                except Exception:
                    pass

        # 3. Fallback: Playwright's built-in Chromium (easily detected)
        print("⚠ CDP unavailable, falling back to Playwright Chromium")
        print("  (Cloudflare may block this — use --auto-launch for real Chrome)")
        browser = p.chromium.launch(
            headless=self.headless,
            args=["--no-first-run", "--no-default-browser-check"],
        )
        return browser, True

    # ── scraping ─────────────────────────────────────────────────

    def _scrape_page(
        self, context, offset: int, min_stars: int
    ) -> list[dict]:
        """Scrape a single HLTV results page.  Returns all match rows."""
        url = f"{HLTV_RESULTS}?offset={offset}"
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded",
                      timeout=self.page_timeout)
            _human_scroll(page, steps=4)

            try:
                page.wait_for_selector("div.result-con", timeout=10000)
            except Exception:
                print("  ⚠ Cloudflare block — solve in Chrome, press Enter…")
                input()
                page.goto(url, wait_until="domcontentloaded",
                          timeout=self.page_timeout)
                page.wait_for_selector("div.result-con", timeout=15000)

            return self._extract_rows(page, min_stars)
        finally:
            page.close()

    def _refill_candidates(
        self, context, last_offset: int, min_stars: int, done: set[str]
    ) -> tuple[int, list[dict]]:
        """Scrape pages until we have CANDIDATE_THRESHOLD fresh candidates.

        Returns updated (last_offset, candidates).  Saves state to disk
        after each page so an interrupted run can resume.
        """
        candidates: list[dict] = []
        seen: set[str] = set()

        while True:
            print(f"  🔍 scraping page offset={last_offset} …")
            rows = self._scrape_page(context, last_offset, min_stars)

            added = 0
            for r in rows:
                mid = r["match_id"]
                if mid not in seen:
                    seen.add(mid)
                    candidates.append(r)
                    added += 1

            last_offset += 100
            self._save_candidates(last_offset, candidates)

            new_count = sum(
                1 for c in candidates if c["match_id"] not in done
            )
            print(f"  📄 page done: +{added} rows, "
                  f"{len(candidates)} total ({new_count} fresh)")

            if new_count >= CANDIDATE_THRESHOLD:
                break

            # Safety limit — generous cap for incremental mode
            if last_offset > 100_000:
                print("  ⚠ reached safety limit (offset 100k)")
                break

            time.sleep(random.uniform(5, 10))

        return last_offset, candidates

    def _extract_rows(self, page, min_stars: int) -> list[dict]:
        """Extract match info from a results page."""
        return page.evaluate(
            """(minStars) => {
                const matches = [];
                document.querySelectorAll('div.result-con').forEach(con => {
                    const a = con.querySelector('a.a-reset[href*="/matches/"]');
                    if (!a) return;
                    const href = a.href;
                    const m = href.match(/\\/matches\\/(\\d+)/);
                    if (!m) return;
                    const matchId = m[1];

                    // Stars filter
                    const stars = con.querySelectorAll('.stars i.fa, .stars i.fas').length;
                    if (stars < minStars) return;

                    matches.push({
                        match_id: matchId,
                        url: href,
                        team1: (con.querySelector('.team1')?.innerText || '').trim(),
                        team2: (con.querySelector('.team2')?.innerText || '').trim(),
                        event: (con.querySelector('.event-name')?.innerText || '').trim(),
                        stars: stars
                    });
                });
                return matches;
            }""",
            min_stars,
        )

    def _download_one(self, context, match: dict) -> Path | None:
        """
        Open match page, find demo download link, download .rar.
        Called from the main thread — no concurrency issues.
        """
        match_url = match["url"]
        mid = match["match_id"]
        key = f"{mid}-{match['team1']}-vs-{match['team2']}"
        rar_path = self.tmp_dir / f"{key}.rar"

        page = context.new_page()
        try:
            # 1. Load match page (with retry)
            for attempt in range(3):
                try:
                    page.goto(match_url, wait_until="domcontentloaded",
                              timeout=self.page_timeout)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    wait = (attempt + 1) * 10
                    print(f"   🔄 [{key}] retry {attempt+1}/3 after {wait}s")
                    time.sleep(wait)

            _human_scroll(page, steps=3)
            time.sleep(random.uniform(1, 2))

            demo_btn = page.locator("a[data-demo-link]")
            if demo_btn.count() == 0:
                print(f"   ⚠ [{key}] no demo link")
                return None

            demo_path = demo_btn.first.get_attribute("data-demo-link")
            if not demo_path:
                print(f"   ⚠ [{key}] no data-demo-link attr")
                return None

            download_url = (
                f"https://www.hltv.org{demo_path}"
                if demo_path.startswith("/")
                else demo_path
            )

            # 2. Download .rar
            with page.expect_download(timeout=self.download_timeout) as dl_info:
                try:
                    page.goto(download_url, wait_until="commit")
                except Exception:
                    pass  # "Download is starting" is expected

            download = dl_info.value
            download.save_as(str(rar_path))

            # Clean Playwright's temp file immediately — without this
            # /tmp/playwright-artifacts-* balloons to gigabytes after
            # downloading hundreds of RARs.
            try:
                download.delete()
            except Exception:
                pass

            if not rar_path.exists() or rar_path.stat().st_size < 1000:
                print(f"   ⚠ [{key}] download too small")
                return None

            return rar_path

        except Exception as exc:
            print(f"   ⚠ [{key}] {exc}")
            return None

        finally:
            page.close()

    def _parse_one(
        self,
        match: dict,
        rar_path: Path,
        interval: float | None,
        stride: int | None,
        verbose: bool,
    ) -> dict | None:
        """
        Extract .rar and parse .dem files to .csz (demo_extract format).
        Called from a worker thread — no browser access needed.
        """
        mid = match["match_id"]
        key = f"{mid}-{match['team1']}-vs-{match['team2']}"
        dem_files: list[Path] = []
        saved: list[str] = []

        try:
            rf = rarfile.RarFile(rar_path)
            rf.extractall(self.tmp_dir)
            dem_files = [
                self.tmp_dir / f
                for f in rf.namelist()
                if f.endswith(".dem")
            ]

            if not dem_files:
                print(f"   ⚠ [{key}] no .dem in rar")
                return None

            for dem_path in dem_files:
                try:
                    out_name = f"{key}+{dem_path.stem}.csz"
                    out_path = self.csz_dir / out_name

                    # v5: per-tick by default, with real fire geometry and
                    # smoke voxels.  interval/stride thin the sampling out.
                    data = extract_demo(
                        str(dem_path), interval=interval, stride=stride,
                    )
                    actual_path = codec.save(data, out_path)

                    if verbose:
                        size_kb = actual_path.stat().st_size / 1024
                        print(f"   📊 [{key}] {actual_path.name} ({size_kb:.0f} KB)")

                    saved.append(str(actual_path.name))

                except BaseException as exc:
                    # BaseException, not Exception: demoparser2 is a Rust
                    # extension and its panics arrive as PanicException, which
                    # derives from BaseException.  Catching only Exception let
                    # one bad demo kill the entire scrape.
                    print(f"   ⚠ [{key}] parse failed: {dem_path.name}: "
                          f"{_exc_note(exc)}")

        except BaseException as exc:
            print(f"   ⚠ [{key}] {_exc_note(exc)}")

        finally:
            # Clean up temp files
            rar_path.unlink(missing_ok=True)
            for d in dem_files:
                d.unlink(missing_ok=True)

        return {
            "match_id": mid,
            "csz": saved,
            "team1": match["team1"],
            "team2": match["team2"],
            "event": match.get("event", ""),
        } if saved else None

    @staticmethod
    def _clean_playwright_artifacts() -> None:
        """Remove stale Playwright download temp dirs from previous runs."""
        for pattern in ("playwright-artifacts-*", ".org.chromium.Chromium.*"):
            for path in Path("/tmp").glob(pattern):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except Exception:
                    pass

    # ── candidate persistence ────────────────────────────────────

    def _load_candidates(self) -> tuple[int, list[dict]]:
        """Return (last_offset, candidates) from the state file."""
        if not self.candidates_file.exists():
            return 0, []
        try:
            with open(self.candidates_file) as f:
                data = json.load(f)
            return data.get("last_offset", 0), data.get("candidates", [])
        except Exception:
            return 0, []

    def _save_candidates(self, last_offset: int, candidates: list[dict]) -> None:
        """Persist scraping state to disk."""
        with open(self.candidates_file, "w") as f:
            json.dump(
                {"last_offset": last_offset, "candidates": candidates},
                f, ensure_ascii=False, indent=2,
            )

    # ── done tracking ───────────────────────────────────────────

    def _load_done(self) -> set[str]:
        if not self.done_file.exists():
            return set()
        try:
            with open(self.done_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                return {str(d["match_id"]) for d in data}
        except Exception:
            pass
        return set()

    def _mark_done(self, match_id: str, result: dict) -> None:
        done = []
        if self.done_file.exists():
            try:
                with open(self.done_file) as f:
                    done = json.load(f)
            except Exception:
                done = []
        done.append(result)
        with open(self.done_file, "w") as f:
            json.dump(done, f, ensure_ascii=False, indent=2)


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────

def _human_scroll(page, steps: int = 5) -> None:
    for _ in range(steps):
        delta = random.randint(200, 600)
        page.mouse.wheel(0, delta)
        time.sleep(random.uniform(0.3, 0.8))


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Scrape HLTV, download demos, and parse to JSON — all in one."
    )
    ap.add_argument(
        "--count", "-n", type=int, default=10,
        help="Number of demos to download and process (default: 10)",
    )
    ap.add_argument(
        "--output-dir", "-o", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--interval", "-i", type=float, default=None,
        help="Seconds between samples for .csz extraction "
             "(default: every tick, ~64 Hz; 0 also means every tick)",
    )
    ap.add_argument(
        "--stride", "-s", type=int, default=None,
        help="Sample every Nth tick — overrides --interval (1 = every tick)",
    )
    ap.add_argument(
        "--min-stars", type=int, default=0,
        help="Minimum HLTV star rating filter (0 = all)",
    )
    ap.add_argument(
        "--user-data-dir", default=DEFAULT_USER_DATA_DIR,
        help="Chrome user data directory for persistent login session",
    )
    ap.add_argument(
        "--cdp-url", default=DEFAULT_CDP_URL,
        help=f"Chrome DevTools Protocol URL (default: {DEFAULT_CDP_URL})",
    )
    ap.add_argument(
        "--chrome-path", default=DEFAULT_CHROME_PATH,
        help="Path to real Chrome executable (for CDP + auto-launch)",
    )
    ap.add_argument(
        "--auto-launch", action="store_true",
        help="Auto-launch real Chrome with remote debugging if not running",
    )
    ap.add_argument(
        "--headless", action="store_true",
        help="Launch Chrome in headless mode (more likely to trigger Cloudflare)",
    )
    ap.add_argument(
        "--setup", action="store_true",
        help="Launch Chrome for HLTV login setup, then exit",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    ap.add_argument(
        "--workers", "-w", type=int, default=2,
        help="Parallel parse workers (default: 2, max: 4)",
    )
    ap.add_argument(
        "--page-timeout", type=int, default=60,
        help="Page load timeout in seconds (default: 60)",
    )
    ap.add_argument(
        "--download-timeout", type=int, default=900,
        help="Demo download timeout in seconds (default: 900)",
    )

    args = ap.parse_args(argv)

    pipeline = DemoPipeline(
        output_dir=args.output_dir,
        user_data_dir=args.user_data_dir,
        cdp_url=args.cdp_url,
        chrome_path=args.chrome_path,
        auto_launch=args.auto_launch,
        headless=args.headless,
        page_timeout=args.page_timeout * 1000,
        download_timeout=args.download_timeout * 1000,
        workers=args.workers,
    )

    if args.setup:
        pipeline.setup()
        return

    pipeline.run(
        count=args.count,
        interval=args.interval,
        stride=args.stride,
        min_stars=args.min_stars,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
