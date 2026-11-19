"""
HLTV match scraper.

Connects to a real Chrome browser via CDP (Chrome DevTools Protocol)
to pass Cloudflare checks. Outputs structured JSON with match metadata.

Usage:
    python -m scraper.hltv --pages 5 --output matches.json

Setup (one-time):
    1. Open Chrome with remote debugging:
       macOS:
         /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\
           --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-hltv
       Linux / WSL:
         google-chrome-stable \\
           --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-hltv
    2. Navigate to https://www.hltv.org and log in.
    3. The scraper will reuse this Chrome session on subsequent runs.

    Or let the scraper auto-launch Chrome with --auto-launch.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None  # type: ignore[assignment]


# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

HLTV_RESULTS = "https://www.hltv.org/results"

DEFAULT_CDP_URL = "http://127.0.0.1:9222"

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
        "google-chrome-stable",
        "google-chrome",
        "chromium-browser",
        "chromium",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
    ]
    _found = None
    for _candidate in _LINUX_CHROME_CANDIDATES:
        _found = shutil.which(_candidate)
        if _found:
            break
    DEFAULT_CHROME_PATH = _found or "google-chrome"
    DEFAULT_USER_DATA_DIR = "/tmp/chrome-hltv"


# ────────────────────────────────────────────────────────────────
# Chrome helpers
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


def launch_chrome(
    chrome_path: str = DEFAULT_CHROME_PATH,
    user_data_dir: str = DEFAULT_USER_DATA_DIR,
    port: int = 9222,
    headless: bool = False,
) -> subprocess.Popen | None:
    """Launch Chrome with remote debugging enabled.  Returns the process handle."""
    chrome = Path(chrome_path)
    if not chrome.exists():
        print(f"Chrome not found at: {chrome_path}", file=sys.stderr)
        return None

    udd = Path(user_data_dir)
    udd.mkdir(parents=True, exist_ok=True)

    # Clean all lock files from previous killed Chrome
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
        # macOS and Linux both have pkill
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
# Scraper
# ────────────────────────────────────────────────────────────────

class HLTVScraper:
    """Scrape HLTV results pages for match URLs and metadata."""

    def __init__(
        self,
        cdp_url: str = DEFAULT_CDP_URL,
        chrome_path: str = DEFAULT_CHROME_PATH,
        user_data_dir: str = DEFAULT_USER_DATA_DIR,
        auto_launch: bool = False,
        headless: bool = False,
    ):
        self.cdp_url = cdp_url
        self.chrome_path = chrome_path
        self.user_data_dir = user_data_dir
        self.auto_launch = auto_launch
        self.headless = headless
        self._chrome_proc: subprocess.Popen | None = None

    # ── public API ──────────────────────────────────────────────

    def setup(self) -> None:
        """
        Launch Chrome (headed) and open HLTV for manual login / Cloudflare.
        After solving the challenge, Chrome stays open for subsequent scrapes.
        """
        if sync_playwright is None:
            raise RuntimeError("playwright not installed.")

        # Always headed for initial setup
        self._chrome_proc = launch_chrome(
            self.chrome_path, self.user_data_dir, headless=False
        )
        if self._chrome_proc is None:
            raise RuntimeError("Failed to launch Chrome.")

        time.sleep(3)

        with sync_playwright() as p:
            ws = _resolve_ws_endpoint(self.cdp_url)
            browser = p.chromium.connect_over_cdp(ws)
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.new_page()

            print("Opening HLTV.org in Chrome…")
            page.goto("https://www.hltv.org/results",
                       wait_until="domcontentloaded", timeout=60000)

            print("\n" + "=" * 50)
            print("  Solve the Cloudflare challenge & log in to HLTV.")
            print("  Chrome will stay open for future scrapes.")
            print("=" * 50 + "\n")
            print("Press Enter when done…")
            input()

            page.close()
            browser.close()

        print("Setup complete. You can now run: python -m scraper.hltv")

    def scrape(
        self,
        pages: int = 10,
        offset_step: int = 100,
        output: str | Path = "matches.json",
        min_stars: int = 0,
        existing_file: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Scrape HLTV results and return match data.

        Parameters
        ----------
        pages : int
            Number of results pages to scrape (each has ~100 matches).
        offset_step : int
            Offset between pages (HLTV uses 100).
        output : str | Path
            Where to save the JSON output.
        min_stars : int
            Minimum star rating (0 = all matches, 1+ = only premium matches).
        existing_file : str | None
            Path to an existing JSON output to deduplicate against.

        Returns
        -------
        list[dict]
        """
        if sync_playwright is None:
            raise RuntimeError(
                "playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        existing_ids = self._load_existing_ids(existing_file)
        seen: set[str] = set()
        matches: list[dict[str, Any]] = []

        with sync_playwright() as p:
            browser, is_own = self._connect_browser(p)
            try:
                context = browser.contexts[0] if browser.contexts else browser.new_context()

                for page_idx in range(pages):
                    offset = page_idx * offset_step
                    url = f"{HLTV_RESULTS}?offset={offset}"
                    print(f"[{page_idx + 1}/{pages}] {url}")

                    page = context.new_page()
                    try:
                        new_matches = self._scrape_page(
                            page, url, existing_ids, seen, min_stars
                        )
                        matches.extend(new_matches)
                        print(f"  → {len(new_matches)} new matches"
                              + (f" (total: {len(matches)})" if matches else ""))

                        # Rate limiting
                        if page_idx < pages - 1:
                            delay = random.uniform(8, 15)
                            print(f"  … waiting {delay:.0f}s")
                            time.sleep(delay)
                    finally:
                        page.close()
            finally:
                if is_own:
                    browser.close()

        # Save output
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)

        print(f"\nDone. {len(matches)} matches saved to {output}")

        # Cleanup
        if self._chrome_proc is not None:
            self._chrome_proc.terminate()

        return matches

    # ── internals ───────────────────────────────────────────────

    def _connect_browser(self, p) -> tuple[Any, bool]:
        """Connect to Chrome.  Returns (browser, is_owned_by_us)."""
        # Try CDP first
        try:
            ws = _resolve_ws_endpoint(self.cdp_url)
            browser = p.chromium.connect_over_cdp(ws)
            print(f"Connected to Chrome via CDP: {self.cdp_url}")
            return browser, False
        except Exception:
            pass

        # Auto-launch if allowed
        if self.auto_launch:
            self._chrome_proc = launch_chrome(
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

        raise RuntimeError(
            "Cannot connect to Chrome.\n"
            "Start Chrome manually:\n"
            f'  "{self.chrome_path}" --remote-debugging-port=9222 '
            f"--user-data-dir={self.user_data_dir}\n"
            "Or use --auto-launch to let the scraper start Chrome."
        )

    def _scrape_page(
        self,
        page,
        url: str,
        existing_ids: set[str],
        seen: set[str],
        min_stars: int,
    ) -> list[dict[str, Any]]:
        """Scrape one results page."""
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        self._human_scroll(page, steps=5)

        # Check for Cloudflare challenge
        try:
            cf_present = page.evaluate(
                "() => document.body.innerText.includes('verify you are a human')"
                + " || document.title.includes('Attention Required')"
                + " || document.title.includes('Just a moment')"
            )
            if cf_present:
                print("  ⚠ Cloudflare challenge detected!")
                print("  → Please solve the challenge in Chrome, then press Enter…")
                input()
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                self._human_scroll(page, steps=5)
        except Exception:
            pass

        # Wait for match rows
        try:
            page.wait_for_selector("div.result-con", timeout=15000)
        except Exception:
            print("  ⚠ No match rows found (still blocked?)")
            return []

        return page.evaluate(
            """({minStars, existingIds}) => {
                const matches = [];
                let currentDate = '';

                // HLTV groups matches under date headers.
                // Walk all children of the results container.
                const container = document.querySelector('div.results-holder')
                    || document.querySelector('div.results')
                    || document.body;

                // Collect date headers and match rows
                const allRows = document.querySelectorAll('div.result-con');
                allRows.forEach(con => {
                    const a = con.querySelector('a.a-reset[href*="/matches/"]');
                    if (!a) return;

                    const href = a.href;
                    const m = href.match(/\\/matches\\/(\\d+)/);
                    if (!m) return;
                    const matchId = m[1];

                    // Stars
                    const starEls = con.querySelectorAll('.stars i.fa, .stars i.fas, .stars i.far');
                    let stars = starEls.length;

                    if (stars < minStars) return;
                    if (existingIds.includes(matchId)) return;

                    // Teams
                    const t1el = con.querySelector('.team1');
                    const t2el = con.querySelector('.team2');
                    const team1 = t1el ? t1el.innerText.trim() : '';
                    const team2 = t2el ? t2el.innerText.trim() : '';

                    // Score
                    const s1 = con.querySelector('.score-won')?.innerText.trim() || '';
                    const s2 = con.querySelector('.score-lost')?.innerText.trim() || '';

                    // Event
                    const evt = con.querySelector('.event-name')?.innerText.trim() || '';

                    // Map text (usually in .map-text or .map element)
                    const mapEl = con.querySelector('.map-text, .map');
                    const mapText = mapEl ? mapEl.innerText.trim() : '';

                    // Date — look for nearest preceding date header
                    let date = '';
                    let prev = con.previousElementSibling;
                    while (prev && !date) {
                        const headline = prev.querySelector('.standard-headline');
                        if (headline) date = headline.innerText.trim();
                        prev = prev.previousElementSibling;
                    }

                    matches.push({
                        match_id: matchId,
                        url: href,
                        team1: team1,
                        team2: team2,
                        score: s1 + '-' + s2,
                        event: evt,
                        stars: stars,
                        map: mapText,
                        date: date
                    });
                });
                return matches;
            }""",
            {"minStars": min_stars, "existingIds": list(existing_ids)},
        )

    def _human_scroll(self, page, steps: int = 5) -> None:
        """Simulate human-like scrolling."""
        for _ in range(steps):
            delta = random.randint(200, 600)
            page.mouse.wheel(0, delta)
            time.sleep(random.uniform(0.3, 0.8))

    @staticmethod
    def _load_existing_ids(file_path: str | None) -> set[str]:
        """Load match IDs from an existing JSON output for dedup."""
        if not file_path:
            return set()
        path = Path(file_path)
        if not path.exists():
            return set()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return {str(m["match_id"]) for m in data if "match_id" in m}
        except Exception:
            pass
        return set()


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Scrape HLTV results pages for match URLs and metadata."
    )
    ap.add_argument(
        "--pages", "-n", type=int, default=10,
        help="Number of results pages to scrape (default: 10, ~1000 matches)",
    )
    ap.add_argument(
        "--offset-step", type=int, default=100,
        help="HLTV offset step (default: 100)",
    )
    ap.add_argument(
        "--output", "-o", default="matches.json",
        help="Output JSON file (default: matches.json)",
    )
    ap.add_argument(
        "--min-stars", type=int, default=0,
        help="Minimum star rating filter (0 = all)",
    )
    ap.add_argument(
        "--existing", default=None,
        help="Existing JSON to deduplicate against",
    )
    ap.add_argument(
        "--cdp-url", default=DEFAULT_CDP_URL,
        help="Chrome DevTools Protocol URL",
    )
    ap.add_argument(
        "--chrome-path", default=DEFAULT_CHROME_PATH,
        help="Path to Chrome executable",
    )
    ap.add_argument(
        "--user-data-dir", default=DEFAULT_USER_DATA_DIR,
        help="Chrome user data directory (cookies/sessions persist here)",
    )
    ap.add_argument(
        "--auto-launch", action="store_true",
        help="Auto-launch Chrome if not already running",
    )
    ap.add_argument(
        "--headless", action="store_true",
        help="Launch Chrome in headless mode (may trigger Cloudflare)",
    )
    ap.add_argument(
        "--setup", action="store_true",
        help="Launch Chrome and open HLTV for manual login / Cloudflare setup",
    )

    args = ap.parse_args(argv)

    scraper = HLTVScraper(
        cdp_url=args.cdp_url,
        chrome_path=args.chrome_path,
        user_data_dir=args.user_data_dir,
        auto_launch=args.auto_launch,
        headless=args.headless,
    )

    if args.setup:
        scraper.setup()
        return

    scraper.scrape(
        pages=args.pages,
        offset_step=args.offset_step,
        output=args.output,
        min_stars=args.min_stars,
        existing_file=args.existing,
    )


if __name__ == "__main__":
    main()
