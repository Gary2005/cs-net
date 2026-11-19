#!/usr/bin/env python3
"""Which agent/weapon models do the example demos actually reference?

A replay viewer only needs the models the demo uses, so this enumerates the
`player_*` model paths and weapon names appearing in the provided demos. Uses
demoparser2 for the tick-level model column, and falls back to reporting what
the parser exposes if the column is absent.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from demoparser2 import DemoParser


def main(argv: list[str]) -> int:
    demos = [Path(p) for p in argv[1:]]
    if not demos:
        d = Path("examples/demo")
        demos = sorted(d.glob("*.dem"))
    if not demos:
        print("no demos found")
        return 1

    for dem in demos:
        print(f"\n=== {dem.name} ({dem.stat().st_size / 1048576:,.0f} MB) ===")
        try:
            parser = DemoParser(str(dem))
            header = parser.parse_header()
            print(f"  map={header.get('map_name')}  tickrate/interval ok")
        except Exception as exc:
            print(f"  header failed: {type(exc).__name__}: {exc}")
            continue

        # what per-tick columns exist that name a model?
        try:
            fields = parser.list_updated_fields()
            model_like = sorted(f for f in fields if "model" in f.lower() or "item" in f.lower())
            print(f"  model/item-ish fields ({len(model_like)}): {model_like[:20]}")
        except Exception as exc:
            print(f"  list_updated_fields failed: {exc}")

        for prop in ("model", "Player.model", "active_weapon_name", "weapon_name"):
            try:
                df = parser.parse_ticks([prop])
            except Exception:
                continue
            if df is None or df.empty or prop not in df.columns:
                continue
            vals = [v for v in df[prop].dropna().unique() if isinstance(v, str) and v]
            if not vals:
                continue
            print(f"  {prop}: {len(vals)} distinct")
            for v, n in Counter(df[prop].dropna()).most_common(15):
                print(f"     {n:>8}  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
