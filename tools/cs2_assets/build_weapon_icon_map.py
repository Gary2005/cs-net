#!/usr/bin/env python3
"""Build the demo-weapon-name -> HUD icon mapping from the game's own schema.

The visualizer sees weapon names as the demo exposes them (``active_weapon_name``,
e.g. "AK-47", "Butterfly Knife"), but the extracted icons are named by internal
item class (``weapon_ak47_png.png``). Nothing links the two except the game's
item schema, so this derives the table instead of hand-maintaining it.

Chain, all read from the VPK via Source2Viewer-CLI:

    items_game.txt   items -> { item_class, item_name ("#LOC_KEY"), image_inventory }
    csgo_english.txt LOC_KEY -> display name

Note the mapping is *not* injective: several loc keys can share a display name
(SFUI_WPNHUD_Deagle and SFUI_WPNHUD_DesertEagle are both "Desert Eagle"), so the
reverse lookup keeps every candidate and the caller decides.

Usage:
    python build_weapon_icon_map.py \
        --items-game <items_game.txt> \
        --english    <csgo_english.txt> \
        --icons      <extracted base_weapons dir> \
        --out        <weapon_icons.json> \
        [--demo-names <names.txt>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Valve's KeyValues format is not JSON: values are quoted strings, keys may
# repeat, and comments use //. A full parser is overkill -- the three fields we
# want always appear as simple quoted pairs on one line.
KV_RE = re.compile(r'^\s*"([^"]+)"\s+"([^"]*)"\s*$')


def iter_kv(path: Path):
    """Yield (key, value) for every simple quoted pair in a KeyValues file."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.lstrip().startswith("//"):
                continue
            m = KV_RE.match(line)
            if m:
                yield m.group(1), m.group(2)


def load_english(path: Path) -> dict[str, str]:
    """Loc tokens, keyed case-insensitively.

    items_game's ``item_name`` does not reliably match the token's own casing:
    it says ``#SFUI_WPNHUD_FLASHBANG`` while the token is ``SFUI_WPNHUD_Flashbang``.
    A case-sensitive lookup silently drops those weapons, so fold case and keep
    the first spelling seen.
    """
    out: dict[str, str] = {}
    for k, v in iter_kv(path):
        out.setdefault(k.lower(), v)
    return out


# Item entries are keyed inconsistently: guns use "item_class", knives use
# "name" (with a numeric block key above them). Accept both.
CLASS_KEYS = ("item_class", "name")


def load_items(path: Path) -> list[dict[str, str]]:
    """Collect each item's class/name/image triple.

    Walks the line stream tracking the current block, because the anchor key
    tells us we are inside a real item entry rather than one of the many nested
    sub-blocks that also carry "name"/"image_inventory" keys.
    """
    items: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for key, value in iter_kv(path):
        if key in CLASS_KEYS:
            if cur.get("item_class"):
                items.append(cur)
            # "name" also appears for non-item things (paint kits, prefabs);
            # only start a block if it looks like a weapon class.
            cur = {"item_class": value} if value.startswith("weapon_") else {}
        elif key in ("item_name", "image_inventory", "model_player") and cur:
            cur.setdefault(key, value)
    if cur.get("item_class"):
        items.append(cur)
    return items


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-game", type=Path, required=True)
    ap.add_argument("--english", type=Path, required=True)
    ap.add_argument("--icons", type=Path, required=True,
                    help="directory of extracted *_png.png HUD icons")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--demo-names", type=Path,
                    help="optional newline-separated weapon names to coverage-check")
    args = ap.parse_args(argv[1:])

    english = load_english(args.english)
    items = load_items(args.items_game)
    print(f"items_game : {len(items):,} item entries")
    print(f"english    : {len(english):,} loc tokens")

    icons = {p.name for p in args.icons.glob("*.png")}
    print(f"icons      : {len(icons)} files")

    # display name -> [icon filenames]
    by_display: dict[str, list[str]] = {}
    unresolved: list[str] = []
    resolved = 0
    for it in items:
        cls = it.get("item_class", "")
        image = it.get("image_inventory", "")
        if not cls or not image:
            continue
        icon = Path(image).name + "_png.png"
        if icon not in icons:
            continue
        resolved += 1
        loc = it.get("item_name", "")
        display = english.get(loc.lstrip("#").lower())
        if not display:
            # Knives and grenades have loc keys that follow no consistent
            # pattern (SFUI_WPNHUD_Knife_Butterfly, SFUI_WPNHUD_KnifeM9,
            # SFUI_WPNHUD_knife_skeleton), so fall back to probing the
            # conventions before giving up on the item.
            stem = Path(image).name
            tail = stem.replace("weapon_", "")
            for cand in (
                f"sfui_wpnhud_{stem}",
                f"sfui_wpnhud_{tail}",
                f"sfui_wpnhud_knife_{tail.replace('knife_', '')}",
                f"sfui_wpnhud_knife{tail.replace('knife_', '')}",
            ):
                if cand in english:
                    display = english[cand]
                    break
        if not display:
            unresolved.append(it.get("item_class", "?"))
            continue
        by_display.setdefault(display, []).append(icon)

    print(f"resolved   : {resolved} items -> an extracted icon")
    print(f"display    : {len(by_display)} distinct names")
    if unresolved:
        print(f"no loc key : {len(unresolved)} -> {sorted(set(unresolved))}")

    table = {k: sorted(set(v)) for k, v in sorted(by_display.items())}

    if args.demo_names and args.demo_names.is_file():
        names = [n.strip() for n in args.demo_names.read_text(
            encoding="utf-8").splitlines() if n.strip()]
        hit = [n for n in names if n in table]
        miss = [n for n in names if n not in table]
        print()
        print(f"demo coverage: {len(hit)}/{len(names)}")
        if miss:
            print("  MISSING:")
            for n in miss:
                print(f"    {n!r}")
        amb = {n: table[n] for n in hit if len(table[n]) > 1}
        if amb:
            print(f"  AMBIGUOUS ({len(amb)}):")
            for n, v in amb.items():
                print(f"    {n!r} -> {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
