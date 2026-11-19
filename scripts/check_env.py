#!/usr/bin/env python3
"""Fast verification of the cs-net environment.

Checks the Python version, the extraction tool (``tools/demo_extract``) and its
Go side-channel -- the one moving part that fails silently when a binary built
for another OS is reused. With ``--all`` it also checks the optional pieces the
web app and the dataset inspectors need.

Usage:
    python scripts/check_env.py           # core: demo_extract + side-channel
    python scripts/check_env.py --all     # + Flask / pandas

Exit code is 0 when everything checked out, 1 otherwise.
"""

from __future__ import annotations

import argparse
import platform
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


def note(msg: str, hint: str | None = None) -> None:
    """Not a failure: something optional is missing and the app degrades."""
    print(f"  - {msg}")
    if hint:
        for line in hint.strip().splitlines():
            print(f"      {line}")


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


def check_maps() -> None:
    """The map geometry: only the zipped form ships with the repo.

    Not a failure when it is still packed -- the app degrades to an empty patch
    of ground and the library page offers a one-click unpack -- but "the replay
    has no buildings" is much easier to answer here than by reading the console.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from scripts.extract_maps import inspect
    except Exception as exc:  # noqa: BLE001 - 报告出来比抛栈好
        fail(f"scripts/extract_maps.py — {type(exc).__name__}: {exc}")
        return

    state = inspect()
    if not state["maps"]:
        note("map geometry — no *.obj.zip in maps/optimized_obj_files")
        return
    total = len(state["maps"])
    if state["ok"]:
        ok(f"map geometry ({total}/{total} unpacked, {_mb(state['present_bytes'])})")
        return
    note(
        f"map geometry {total - len(state['missing'])}/{total} unpacked "
        f"({_mb(state['unpacked_bytes'] - state['present_bytes'])} left)",
        "the replay falls back to an empty ground and thumbnails stay blank\n"
        "unpack it: python scripts/extract_maps.py   (or the banner in the library page)",
    )


def _mb(n: int | None) -> str:
    if not n:
        return "0 MB"
    return f"{n / (1024 * 1024):.0f} MB"


def check_extras() -> None:
    for mod, name, hint in (
        ("flask", "flask (web app)", "pip install flask"),
        ("pandas", "pandas (web app probe)", "pip install pandas"),
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
                    help="also check the web app extras")
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
    check_maps()

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
