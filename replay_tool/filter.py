#!/usr/bin/env python3
"""
JSON post-processing filter for CS2 demo data.

Fixes common issues with the raw V2 JSON output:
  1. Smoke/inferno durations — when ``te`` is null, set default durations
     (smoke = 18 s, inferno = 7 s).
  2. Cross-round projectiles — remove smokes and infernos with ``ts < 0``
     (leftover from warmup / previous round).
  3. Trims per-round event lists to only contain events within the round's
     tick range.

Usage:
    python filter.py input.json -o output.json
    python -m replay_tool.filter input.json -o output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SMOKE_DURATION = 18   # seconds
INFERNO_DURATION = 7  # seconds


def filter_data(data: dict[str, Any]) -> dict[str, Any]:
    """Apply all filters to the V2 JSON data (mutates in-place)."""
    if data.get("format") != "cs2.demo.v2":
        raise ValueError(f"Expected cs2.demo.v2 format, got {data.get('format')}")

    for round_data in data.get("rounds", []):
        _fix_projectile_durations(round_data)
        _remove_cross_round(round_data)
        _clip_events_to_round(round_data)

    return data


def _fix_projectile_durations(round_data: dict) -> None:
    """Set default durations for smokes (18 s) and infernos (7 s) when te is null."""
    for s in round_data.get("events", {}).get("smokes", []):
        if s.get("te") is None:
            s["te"] = round(s["ts"] + SMOKE_DURATION, 2)

    for inf in round_data.get("events", {}).get("infernos", []):
        if inf.get("te") is None:
            inf["te"] = round(inf["ts"] + INFERNO_DURATION, 2)


def _remove_cross_round(round_data: dict) -> None:
    """Remove smokes and infernos that started before the round (ts < 0)."""
    events = round_data.get("events", {})
    events["smokes"] = [s for s in events.get("smokes", []) if s["ts"] >= 0]
    events["infernos"] = [
        inf for inf in events.get("infernos", []) if inf["ts"] >= 0
    ]


def _clip_events_to_round(round_data: dict) -> None:
    """Remove kill/damage/bomb/sound/grenade events outside the round's tick range."""
    ticks = round_data.get("ticks", [])
    if not ticks:
        return
    t_min, t_max = ticks[0], ticks[-1]

    events = round_data.get("events", {})
    for key in ("kills", "damage", "bomb", "grenades", "sound"):
        if key in events:
            events[key] = [e for e in events[key] if t_min < e["t"] <= t_max]


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Post-process CS2 demo JSON: fix durations, remove cross-round events."
    )
    ap.add_argument("input", help="Path to input JSON file (.json or .json.gz)")
    ap.add_argument("--out", "-o", required=True, help="Output JSON file path")
    ap.add_argument("--compact", action="store_true", help="Minified output")
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load
    if input_path.suffix == ".gz":
        import gzip
        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

    # Filter
    filter_data(data)

    # Save
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.compact else 2
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Filtered → {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
