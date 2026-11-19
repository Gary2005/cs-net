#!/usr/bin/env python3
"""Fast verification of the cs-net-v5 environment.

Ported from ``cs-net-dev/scripts/check_env.py`` and adapted to v5's dependency
set (no open3d / webdataset / torch — v5 has no training code; ``demo_extract``
and the Go side-channel replace ``demo_parser``).

Usage:
    python scripts/check_env.py           # core: scraper + demo_extract
    python scripts/check_env.py --all     # + visualizer / scratch-probe extras

Exit code is 0 when everything checked out, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

_PLATFORM_KIND = {"Linux": "ELF", "Darwin": "Mach-O", "Windows": "PE"}

_failed: list[str] = []


# ── output helpers ───────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def fail(msg: str, hint: str | None = None) -> None:
    print(f"  \u2717 {msg}")
    if hint:
        for line in hint.strip().splitlines():
            print(f"      {line}")
    _failed.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")


# ── checks ───────────────────────────────────────────────────────────────────

def check_python() -> None:
    v = sys.version_info
    label = f"python {v.major}.{v.minor}.{v.micro}  ({sys.executable})"
    if v >= (3, 10):
        ok(label)
    else:
        fail(f"{label} — need >= 3.10")


def check_modules() -> None:
    for mod, name in (
        ("numpy", "numpy"),
        ("demoparser2", "demoparser2"),
        ("zstandard", "zstandard"),
        ("rarfile", "rarfile"),
    ):
        try:
            __import__(mod)
            ok(name)
        except ImportError:
            fail(f"{name} — MISSING", "pip install demoparser2 numpy zstandard rarfile")


def check_demo_extract() -> None:
    """``tools/demo_extract`` must import — it is the v5 data-conversion core."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    try:
        from demo_extract import codec, extract  # noqa: F401
    except Exception as exc:
        fail(
            f"demo_extract — {type(exc).__name__}: {exc}",
            "the package lives in tools/demo_extract; fix the import above "
            "(usually a missing dependency or a stale .pyc)",
        )
        return
    ok("demo_extract (codec + extract)")


def _binary_kind(path: Path) -> str:
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head == b"\x7fELF":
        return "ELF"
    if head in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
                b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce"):
        return "Mach-O"
    if head[:2] == b"MZ":
        return "PE"
    return "unknown"


def check_sidechannel() -> None:
    """The Go helper is the one v5-only moving part — and the one that breaks
    silently when the repo's macOS binary is used on Linux."""
    try:
        from demo_extract import sidechannel
    except Exception:
        return  # already reported by check_demo_extract
    path = sidechannel.binary_path()
    if not path.is_file():
        fail(
            f"side-channel binary missing: {path}",
            "bash tools/demo_extract/build_sidechannel.sh\n"
            "(needs a Go toolchain; set CS2_SIDECHANNEL_BIN to use a prebuilt one)",
        )
        return

    kind = _binary_kind(path)
    want = _PLATFORM_KIND.get(platform.system(), "?")
    if kind != want:
        fail(
            f"side-channel is {kind} but this machine is {want}",
            "it was built on another OS (the repo ships a macOS arm64 binary) — "
            "rebuild here:\n  bash tools/demo_extract/build_sidechannel.sh",
        )
        return

    try:
        proc = subprocess.run([str(path)], capture_output=True, text=True, timeout=30)
    except OSError as exc:
        fail(f"side-channel will not start — {exc}")
        return
    if proc.returncode == 2 and "usage" in (proc.stderr or "").lower():
        ok(f"side-channel binary ({kind}, runs)")
    else:
        fail(
            f"side-channel exited {proc.returncode} with no usage text",
            "rebuild it: bash tools/demo_extract/build_sidechannel.sh",
        )


def _pinned_chromium_revision() -> str | None:
    try:
        import playwright
    except ImportError:
        return None
    spec = Path(playwright.__file__).parent / "driver" / "package" / "browsers.json"
    try:
        data = json.loads(spec.read_text())
        return next(b["revision"] for b in data["browsers"] if b["name"] == "chromium")
    except Exception:
        return None


def check_chromium() -> None:
    """Playwright pins its browser revision to the pip version, so upgrading
    the package silently invalidates an already-downloaded browser."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        fail("playwright — MISSING", "pip install playwright\npython -m playwright install chromium")
        return

    rev = _pinned_chromium_revision()
    rev_label = f"revision {rev}" if rev else "revision unknown"

    with sync_playwright() as p:
        exe = Path(p.chromium.executable_path)
        if not exe.exists():
            fail(
                f"playwright chromium not downloaded ({rev_label})",
                f"expected at: {exe}\n"
                "python -m playwright install chromium\n"
                f"{_playwright_deps_hint()}\n"
                "note: upgrading the playwright pip package bumps the pinned revision "
                "and does NOT re-download the browser",
            )
            return
        try:
            browser = p.chromium.launch(headless=True)
            browser.close()
        except Exception as exc:
            fail(
                f"playwright chromium failed to launch ({rev_label}) — {exc}",
                _playwright_deps_hint(),
            )
            return
    ok(f"playwright chromium ({rev_label}, launch OK)")


def _unrar_hint() -> str:
    """Install instructions for the archive tool, per platform.

    ``rarfile`` shells out to ``unrar``, so the hint has to match the host --
    telling a Windows user to run ``sudo apt install`` is worse than no hint.
    """
    if platform.system() == "Windows":
        return (
            "winget install RARLab.WinRAR      # provides unrar.exe\n"
            "winget install 7zip.7zip          # alternative: add 7z.exe to PATH\n"
            "(only needed for .rar-packed demos; plain .dem files need nothing)"
        )
    if platform.system() == "Darwin":
        return "brew install unrar             # or: brew install sevenzip"
    return (
        "sudo apt install unrar        # Debian/Ubuntu\n"
        "sudo dnf install unrar        # Fedora (RPM Fusion)"
    )


def _playwright_deps_hint() -> str:
    """Playwright's system-library install command is Linux-only."""
    if platform.system() == "Linux":
        return "sudo python -m playwright install-deps chromium"
    return "(Windows/macOS: bundled browser needs no extra system libraries)"


def check_unrar() -> None:
    if shutil.which("unrar"):
        ok("unrar")
    elif shutil.which("7z") or shutil.which("7za"):
        # rarfile can be pointed at 7-Zip, so this is usable, just not default.
        ok("7-Zip present (rarfile can use it via UNRAR_TOOL=7z)")
    else:
        fail(
            "unrar — not found (rarfile needs it to unpack .rar demos)",
            _unrar_hint(),
        )


def check_extras() -> None:
    for mod, name, hint in (
        ("flask", "flask (visualizer)", "pip install flask"),
        ("pandas", "pandas (notes/scratch probes only)", "pip install pandas"),
    ):
        try:
            __import__(mod)
            ok(name)
        except ImportError:
            fail(f"{name} — MISSING", hint)


# ── main ─────────────────────────────────────────────────────────────────────

def _force_utf8_console() -> None:
    """Make the ✓/✗ output survive a non-UTF-8 console.

    On Windows the console defaults to a legacy code page (GBK here), and
    printing ``\\u2713`` raises ``UnicodeEncodeError: 'gbk' codec can't encode``.
    That is a hard crash before any check runs, so it has to be handled before
    the first print. macOS/Linux default to UTF-8 and are unaffected.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main() -> int:
    _force_utf8_console()
    ap = argparse.ArgumentParser(description="Verify the cs-net-v5 environment.")
    ap.add_argument("--all", action="store_true",
                    help="also check visualizer / scratch-probe extras")
    args = ap.parse_args()

    print("cs-net-v5 environment check")
    print(f"repo: {ROOT}")

    section("Python")
    check_python()

    section("Core modules")
    check_modules()

    section("Data conversion")
    check_demo_extract()
    check_sidechannel()

    section("Browser (scraper)")
    check_chromium()

    section("Archive tool")
    check_unrar()

    if args.all:
        section("Extras")
        check_extras()

    print()
    if _failed:
        print(f"\u274c {len(_failed)} issue(s) — see above.")
        return 1
    print("\u2705 All good!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
