#!/usr/bin/env python3
"""Categorise a `Source2Viewer-CLI --vpk_list` dump of pak01_dir.vpk.

Input lines look like:
    agents/models/ctm_fbi/ctm_fbi.vmdl_c CRC:00ee832574 size:902896

Writes a per-category summary plus the raw matching paths, so the numbers in
notes/research-gameassets.md can be re-derived from scratch.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

LINE_RE = re.compile(r"^(?P<path>\S+)(?:\s+CRC:(?P<crc>[0-9a-f]+))?(?:\s+size:(?P<size>\d+))?\s*$")

# (category, path prefix predicate, extension filter or None)
CATEGORIES: list[tuple[str, str, str | None]] = [
    ("agents/player models (.vmdl_c)", "agents/models/", "vmdl_c"),
    ("agents/ other", "agents/", None),
    ("weapon world/view models", "weapons/models/", "vmdl_c"),
    ("weapon models (all ext)", "weapons/models/", None),
    ("weapons/ other", "weapons/", None),
    ("particles/ (all)", "particles/", None),
    ("particles/ .vpcf_c", "particles/", "vpcf_c"),
    ("particles/ .vsnap_c", "particles/", "vsnap_c"),
    ("particles/inferno_fx/", "particles/inferno_fx/", None),
    ("particles/explosions_fx/", "particles/explosions_fx/", None),
    ("particles/characters/", "particles/characters/", None),
    ("materials/particle/ .vtex_c", "materials/particle/", "vtex_c"),
    ("materials/particle/ .vmat_c", "materials/particle/", "vmat_c"),
    ("materials/ (all)", "materials/", None),
    ("panorama/images/ (weapon icons etc)", "panorama/images/", None),
    ("panorama/ (all)", "panorama/", None),
    ("models/ (props, world)", "models/", None),
    ("models/ .vmdl_c", "models/", "vmdl_c"),
    ("scripts/items/", "scripts/items/", None),
    ("scripts/", "scripts/", None),
    ("shaders/", "shaders/", None),
    ("sounds/", "sounds/", None),
    ("maps/ (inside pak01)", "maps/", None),
    ("resource/", "resource/", None),
]


def main(argv: list[str]) -> int:
    src = Path(argv[1] if len(argv) > 1 else "vpk_full_list.txt")
    outdir = src.parent
    rows: list[tuple[str, int]] = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        rows.append((m.group("path"), int(m.group("size") or 0)))

    print(f"parsed {len(rows)} entries from {src.name}")

    ext_counter: Counter[str] = Counter()
    for p, _ in rows:
        ext = p.rsplit(".", 1)[-1] if "." in p.rsplit("/", 1)[-1] else "<none>"
        ext_counter[ext] += 1

    summary: list[str] = ["# pak01_dir.vpk inventory", "", f"total entries: {len(rows)}", ""]
    summary.append("## top 25 extensions by file count")
    summary.append("")
    summary.append("| ext | count |")
    summary.append("|---|---:|")
    for ext, n in ext_counter.most_common(25):
        summary.append(f"| `.{ext}` | {n} |")
    summary.append("")

    summary.append("## categories")
    summary.append("")
    summary.append("| category | files | total MB |")
    summary.append("|---|---:|---:|")
    detail: dict[str, list[tuple[str, int]]] = {}
    for label, prefix, ext in CATEGORIES:
        hits = [(p, s) for p, s in rows
                if p.startswith(prefix) and (ext is None or p.endswith("." + ext))]
        detail[label] = hits
        mb = sum(s for _, s in hits) / 1024 / 1024
        summary.append(f"| {label} | {len(hits)} | {mb:,.1f} |")
    summary.append("")

    # per-directory breakdown for the biggest asset families
    for family in ("agents/models/", "weapons/models/", "particles/inferno_fx/",
                   "particles/explosions_fx/", "materials/particle/"):
        sub: Counter[str] = Counter()
        for p, _ in rows:
            if p.startswith(family):
                rest = p[len(family):]
                sub[rest.split("/", 1)[0]] += 1
        summary.append(f"## `{family}` — {len(sub)} subdirectories")
        summary.append("")
        for name, n in sub.most_common(40):
            summary.append(f"- `{name}` — {n}")
        summary.append("")

    (outdir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")

    for label, hits in detail.items():
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        with (outdir / f"list_{slug}.txt").open("w", encoding="utf-8") as fh:
            for p, s in sorted(hits):
                fh.write(f"{p}\t{s}\n")

    print(f"wrote SUMMARY.md + {len(detail)} list files to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
