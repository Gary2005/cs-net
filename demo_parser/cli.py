"""
CLI entry point for demo → JSON conversion.

Usage:
    python -m demo_parser --demo match.dem --out output.json
    python -m demo_parser -d match.dem -o out.json -i 0.25 -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extract import parse_demo, save_demo_json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Convert CS2 .dem files to optimized V2 JSON format."
    )
    ap.add_argument(
        "--demo", "-d",
        required=True,
        help="Path to input .dem file",
    )
    ap.add_argument(
        "--out", "-o",
        required=True,
        help="Path to output JSON file",
    )
    ap.add_argument(
        "--interval", "-i",
        type=float,
        default=0.25,
        help="Tick sampling interval in seconds (default: 0.25)",
    )
    ap.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON without indentation (smaller file)",
    )
    ap.add_argument(
        "--compress", "-z",
        action="store_true",
        help="Gzip-compress the output (.gz)",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )

    args = ap.parse_args(argv)

    demo_path = Path(args.demo)
    if not demo_path.exists():
        print(f"Error: demo file not found: {args.demo}", file=sys.stderr)
        sys.exit(1)

    data = parse_demo(
        str(demo_path),
        interval=args.interval,
        verbose=args.verbose,
    )

    out_path = save_demo_json(
        data,
        args.out,
        compact=args.compact,
        compress=args.compress,
    )

    if args.verbose:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Output      : {out_path} ({size_mb:.1f} MB)")
