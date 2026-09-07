#!/usr/bin/env python3
"""Fast verification of environment — run: python scripts/check_env.py"""

import sys


def _check(pkg, name=None):
    try:
        __import__(pkg)
        print(f"  ✓ {name or pkg}")
        return True
    except ImportError:
        print(f"  ✗ {name or pkg}  — MISSING")
        return False


def main():
    print("Checking Python environment...\n")
    ok = True

    # Core
    ok &= _check("numpy")
    ok &= _check("demoparser2", "demoparser2")
    ok &= _check("playwright.sync_api", "playwright")

    # Browser
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        print("  ✓ playwright chromium (launch OK)")
    except Exception as exc:
        print(f"  ✗ playwright chromium — {exc}")
        ok = False

    # Rar
    ok &= _check("rarfile")

    # Optional
    ok &= _check("open3d")

    print(f"\n{'✅ All good!' if ok else '❌ Some issues — see above.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
