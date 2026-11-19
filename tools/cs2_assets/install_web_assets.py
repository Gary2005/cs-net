#!/usr/bin/env python3
"""Copy the extracted CS2 assets into the visualizer's static tree.

The visualizer serves `visualizer/static/`, so anything the browser must fetch
has to live under `visualizer/static/assets/`. This stages the four asset groups
there and writes a `manifest.json` describing the sprite sequences, because the
frame order and loop flags live in VRF's `.mks` sidecars rather than in the
filenames.

These are Valve's copyrighted assets, and the staged result **is tracked** (see
the bottom of `.gitignore`): a fresh clone does not have to rebuild them, so
this script only has to run when the assets themselves change. That exception
exists because the rebuild path below is not a one-liner -- the write-up is in
`notes/research-gameassets.md` §9.0.1.

Re-running needs a local CS2 install, `tools/cs2_assets/extract_cs2_assets.py`,
and `--probe` pointed at a tree laid out the way this script reads it
(`<probe>/pipeline_out/weapon_icons`, `<probe>/particles_out/particles/
materials/particle/...`, plus `weapons_index.json` at the probe root). The
extractor's default `--out assets` produces none of those paths, so matching
this script takes three runs with three different `--out`s; the layout that
exists on the author's machine is `notes/scratch/asset_probe/`.

Usage:
    python tools/cs2_assets/install_web_assets.py \
        --probe notes/scratch/asset_probe \
        --dest  visualizer/static/assets \
        [--weapon-icons] [--smoke] [--fire] [--weapons ak47,awp,...]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FRAME_RE = re.compile(r'^\s*frame\s+(\S+)\s+(\d+)\s*$')
SEQ_RE = re.compile(r'^\s*sequence\s+(\d+)\s*$')


def rel_to(path: Path, base: Path) -> str:
    """`path` relative to `base`, or the raw path if it is not underneath.

    `--dest` may be given as a relative path, in which case `relative_to(ROOT)`
    raises; printing a path is never worth crashing the whole staging run over.
    """
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def parse_mks(path: Path) -> list[dict]:
    """Parse a VRF .mks sprite-sequence sidecar.

    Format is one `sequence N` header, an optional LOOP marker, then
    `frame <file> <duration>` lines. Frame order matters and is only recorded
    here, so never sort these by filename.
    """
    seqs: list[dict] = []
    cur: dict | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("//"):
            continue
        m = SEQ_RE.match(line)
        if m:
            cur = {"index": int(m.group(1)), "loop": False, "frames": []}
            seqs.append(cur)
            continue
        if cur is None:
            continue
        if line.strip().upper() == "LOOP":
            cur["loop"] = True
            continue
        m = FRAME_RE.match(line)
        if m:
            cur["frames"].append({"file": m.group(1), "duration": int(m.group(2))})
    return seqs


def png_size(path: Path) -> tuple[int, int]:
    """Width/height from the IHDR chunk, without pulling in Pillow."""
    head = path.read_bytes()[:24]
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"not a PNG: {path}")
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def kv_pairs(text: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Split a VDF file into its flat keys and its one-level nested blocks.

    CS2's overview files are two levels at most: the map block holds
    ``pos_x``/``pos_y``/``scale``, and ``verticalsections`` holds one block per
    altitude band (``default``, ``lower``).  A full VDF parser would be
    overkill; quoted lines and braces are the whole format.
    """
    clean = "\n".join(line.split("//")[0] for line in text.splitlines())
    flat: dict[str, str] = {}
    blocks: dict[str, dict[str, str]] = {}
    stack: list[str] = []
    for raw in clean.splitlines():
        line = raw.strip()
        if not line or line == "{":
            continue
        if line == "}":
            if stack:
                stack.pop()
            continue
        m = re.fullmatch(r'"([^"]+)"\s+"([^"]*)"', line)
        if m:
            key, val = m.group(1), m.group(2)
            if len(stack) <= 1:      # inside the map block itself
                flat[key] = val
            else:                    # inside verticalsections/<band>
                blocks.setdefault(stack[-1], {})[key] = val
            continue
        m = re.fullmatch(r'"([^"]+)"', line)
        if m:
            stack.append(m.group(1))
    return flat, blocks


def build_radar_group(src: Path, overviews: Path) -> dict[str, dict]:
    """``<map>_radar_psd.png`` + ``resource/overviews/<map>.txt`` -> one table.

    The transform is not derivable from the image, so a radar without its
    overview is skipped rather than guessed at.  ``*_lower`` images never have
    their own overview: they reuse the parent map's pos/scale and are selected
    by altitude, which is what ``verticalsections`` encodes.
    """
    pngs = {p.name.removesuffix("_radar_psd.png"): p
            for p in src.glob("*_radar_psd.png")}
    table: dict[str, dict] = {}
    for base in sorted(pngs):
        if base.endswith("_lower"):
            continue
        txt = overviews / f"{base}.txt"
        if not txt.is_file():
            print(f"  !! {base}: no overview {txt.name} -- skipped", file=sys.stderr)
            continue
        flat, blocks = kv_pairs(txt.read_text(encoding="utf-8", errors="replace"))
        try:
            pos_x = float(flat["pos_x"])
            pos_y = float(flat["pos_y"])
            scale = float(flat["scale"])
        except (KeyError, ValueError):
            print(f"  !! {base}: overview lacks pos_x/pos_y/scale", file=sys.stderr)
            continue
        if scale <= 0:
            continue
        w, h = png_size(pngs[base])
        entry: dict = {
            "image": f"{base}.png",
            "pos_x": pos_x,
            "pos_y": pos_y,
            "scale": scale,
            "width": w,
            "height": h,
        }
        lower_base = f"{base}_lower"
        if lower_base in pngs:
            try:
                below = float((blocks.get("lower") or {}).get("AltitudeMax", "nan"))
            except ValueError:
                below = float("nan")
            if below == below:                      # not NaN
                entry["lower"] = f"{lower_base}.png"
                entry["lower_below_z"] = below
            else:
                print(f"  ( {base}: lower image present but no AltitudeMax )",
                      file=sys.stderr)
        table[base] = entry
    return table


def copy_sequence(src_dir: Path, base: str, dest_dir: Path) -> dict | None:
    """Copy one sprite sequence (its .mks + every referenced PNG)."""
    mks = src_dir / f"{base}.mks"
    if not mks.is_file():
        return None
    seqs = parse_mks(mks)
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_seqs = []
    for s in seqs:
        kept = []
        for fr in s["frames"]:
            src = src_dir / fr["file"]
            if not src.is_file():
                continue
            shutil.copy2(src, dest_dir / fr["file"])
            kept.append(fr)
        if kept:
            out_seqs.append({"index": s["index"], "loop": s["loop"], "frames": kept})

    if not out_seqs:
        return None
    return {
        "base": base,
        "dir": str(dest_dir.name),
        "sequences": out_seqs,
        "frame_count": sum(len(s["frames"]) for s in out_seqs),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path,
                    default=ROOT / "notes" / "scratch" / "asset_probe")
    ap.add_argument("--dest", type=Path,
                    default=ROOT / "visualizer" / "static" / "assets")
    ap.add_argument("--weapon-icons", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--weapons", default="", help="comma-separated weapon keys")
    ap.add_argument("--radars", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv[1:])

    if args.all:
        args.weapon_icons = args.smoke = args.fire = args.radars = True

    dest = args.dest
    manifest: dict = {"weapon_icons": {}, "sprites": {}, "weapons": []}
    # reuse an existing manifest so partial runs do not drop other groups
    mpath = dest / "manifest.json"
    if mpath.is_file():
        try:
            manifest.update(json.loads(mpath.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    # ── weapon icons ---------------------------------------------------------
    if args.weapon_icons:
        src = args.probe / "pipeline_out" / "weapon_icons"
        pngs = sorted(src.rglob("*.png"))
        if not pngs:
            print(f"!! no weapon icons under {src}", file=sys.stderr)
        else:
            out = dest / "weapon_icons"
            out.mkdir(parents=True, exist_ok=True)
            for p in pngs:
                shutil.copy2(p, out / p.name)
            print(f"weapon_icons : {len(pngs)} PNG -> {rel_to(out, ROOT)}")

        # the display-name -> icon table derived from items_game.txt
        for cand in (args.probe / "weapon_icon_map.json",
                     args.probe / "pipeline_out" / "weapon_icon_map.json"):
            if cand.is_file():
                shutil.copy2(cand, dest / "weapon_icons" / "map.json")
                n = len(json.loads(cand.read_text(encoding="utf-8")))
                manifest["weapon_icons"] = {"map": "weapon_icons/map.json",
                                            "names": n}
                print(f"weapon_icons : map.json ({n} display names)")
                break
        else:
            print("  (no weapon_icon_map.json -- run build_weapon_icon_map.py)",
                  file=sys.stderr)

    # ── smoke ----------------------------------------------------------------
    if args.smoke:
        smoke_src = (args.probe / "particles_out" / "particles" / "materials"
                     / "particle" / "smoke" / "smokeburst")
        # Only the sprite the voxel smoke actually renders. flwmix is a
        # SPRITECARD_TEXTURE_ANIMMOTIONVEC flow map, not a second visible sheet.
        for base, role in (("smokeloop_i_0_sc_hardedge", "albedo"),):
            info = copy_sequence(smoke_src, base, dest / "sprites" / base)
            if info:
                info["role"] = role
                manifest["sprites"][base] = info
                print(f"smoke        : {base} {info['frame_count']} frames")
            else:
                print(f"!! smoke sequence {base} not found under {smoke_src}",
                      file=sys.stderr)

    # ── fire -----------------------------------------------------------------
    if args.fire:
        pdir = args.probe / "particles_out" / "particles" / "materials" / "particle"
        for base, sub in (("flames", "flames"),
                          ("flame_omni", "flames"),
                          ("flame_wisp", "flames"),
                          ("fire_onsurface", "fire_onsurface")):
            src_dir = pdir / sub
            info = copy_sequence(src_dir, base, dest / "sprites" / base)
            if info:
                manifest["sprites"][base] = info
                print(f"fire         : {base} {info['frame_count']} frames")
            else:
                print(f"  (fire sequence {base} absent -- skipped)")

    # ── weapon models --------------------------------------------------------
    if args.weapons:
        wanted = [w.strip() for w in args.weapons.split(",") if w.strip()]

        # Resolve key -> GLB through the exact index written by the extractor.
        # Do NOT search: the old `w in p.stem` match picked
        # `weapon_rif_m4a1_silencer.glb` for the key `m4a1` (it is also a
        # substring of it) and `rglob` order decided the winner.
        #
        # Index paths are relative to `--out` of the extractor (== args.probe),
        # and already start with `weapons/`, so join against the probe root.
        index: dict[str, str] = {}
        idx_path = args.probe / "weapons_index.json"
        if idx_path.is_file():
            try:
                index = json.loads(idx_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                index = {}
        else:
            print(f"!! {idx_path} missing -- re-run extract_cs2_assets.py "
                  f"(--weapons writes it)", file=sys.stderr)

        # Keep the manifest free of stale entries from earlier runs.
        manifest["weapons"] = [w for w in manifest.get("weapons", [])
                               if w.get("key") in index]

        for w in wanted:
            rel = index.get(w)
            glb = (args.probe / rel) if rel else None
            if not glb or not glb.is_file():
                print(f"!! weapon model {w!r} not in the extract index "
                      f"({glb if rel else 'no index entry'})", file=sys.stderr)
                continue
            # The GLB references its textures by bare filename, so the whole
            # directory has to travel together.
            out = dest / "weapons" / glb.parent.name
            out.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in glb.parent.iterdir():
                if f.is_file() and f.suffix.lower() in (".glb", ".png", ".webp"):
                    if f.suffix.lower() == ".glb" and "physics" in f.stem:
                        continue
                    shutil.copy2(f, out / f.name)
                    n += 1
            manifest["weapons"] = [e for e in manifest["weapons"]
                                   if e.get("name") != glb.stem]
            manifest["weapons"].append({
                "name": glb.stem,
                "key": w,
                "dir": str(out.relative_to(dest)),
                "glb": glb.name,
                "files": n,
            })
            print(f"weapon       : {w:18s} {glb.stem} ({n} files -> {rel_to(out, ROOT)})")

    # ── radar ----------------------------------------------------------------
    if args.radars:
        src = args.probe / "radar" / "panorama" / "images" / "overheadmaps"
        overviews = args.probe / "radar" / "resource" / "overviews"
        table = build_radar_group(src, overviews) if src.is_dir() else {}
        if not table:
            print(f"!! no radar images under {src}", file=sys.stderr)
        else:
            out = dest / "radar"
            out.mkdir(parents=True, exist_ok=True)
            # Source names keep Valve's `_radar_psd` suffix; the staged names
            # are the map key, which is what map.json and the frontend use.
            copied = 0
            for base, entry in sorted(table.items()):
                for suffix in ("", "_lower"):
                    if suffix and not entry.get("lower"):
                        continue
                    shutil.copy2(src / f"{base}{suffix}_radar_psd.png",
                                 out / f"{base}{suffix}.png")
                    copied += 1
            (out / "map.json").write_text(
                json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
            manifest["radar"] = {"map": "radar/map.json", "maps": len(table)}
            print(f"radar        : {copied} PNG / {len(table)} maps -> "
                  f"{rel_to(out, ROOT)}")

    dest.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    print(f"\nmanifest -> {rel_to(mpath, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
