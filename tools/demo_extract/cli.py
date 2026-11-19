"""
Command line front-end.

    python -m demo_extract match.dem
    python -m demo_extract match.dem -o match.csz
    python -m demo_extract match.dem --interval 0.25
    python -m demo_extract match.dem --stride 8 --json match.json.gz
    python -m demo_extract --inspect match.csz
    python -m demo_extract --selftest match.dem
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import codec, voxel
from .extract import extract


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _inspect(path: Path) -> int:
    t0 = time.perf_counter()
    data = codec.load(path)
    dt = time.perf_counter() - t0
    size = path.stat().st_size
    meta = data.get("meta", {})
    print(f"file        : {path}")
    print(f"on disk     : {_human(size)}   load {dt * 1000:.0f} ms")
    print(f"format      : {data.get('format')}")
    print(f"map         : {meta.get('map')}   demo {meta.get('demo')}")
    print(f"sampling    : interval={meta.get('interval')} stride={meta.get('stride')} "
          f"-> {meta.get('sampled_ticks')} ticks of {meta.get('total_ticks')}")
    print(f"players     : {len(data.get('players', []))}")
    print(f"rounds      : {len(data.get('rounds', []))}")
    nfire = sum(len(r.get("fire", [])) for r in data.get("rounds", []))
    nsmoke = sum(len(r.get("smoke", [])) for r in data.get("rounds", []))
    nfsnap = sum(len(f["changes"]) for r in data.get("rounds", []) for f in r.get("fire", []))
    nvox = sum(len(s["changes"]) for r in data.get("rounds", []) for s in r.get("smoke", []))
    print(f"fire        : {nfire} infernos, {nfsnap} change-snapshots")
    print(f"smoke       : {nsmoke} instances, {nvox} voxel records")
    if data.get("rounds") and "on_ground" in data["rounds"][0].get("players", [{}])[0]:
        njump = sum(sum(int(x) for x in p["jump"])
                    for r in data["rounds"] for p in r["players"])
        nland = sum(sum(int(x) for x in p["land"])
                    for r in data["rounds"] for p in r["players"])
        print(f"ground      : yes (on_ground/jump/land) — {njump} jumps, {nland} landings")
    else:
        print("ground      : no (pre-v6 file, no on_ground/jump/land)")
    return 0


def _selftest(demo: Path, args) -> int:
    """Extract, save, load, compare — and prove the smoke journals round trip."""
    print(f"extracting {demo} ...")
    data = extract(
        demo,
        interval=args.interval,
        stride=args.stride,
        build_dir=args.build_dir,
        verbose=True,
    )
    out = args.output or (Path("/tmp") / (demo.stem + ".csz"))
    codec.save(data, out, level=args.level)
    size = Path(out).stat().st_size
    raw = len(codec.raw_blob(data))
    print(f"\nsaved       : {out}  ({_human(size)}, uncompressed body {_human(raw)}, "
          f"ratio {raw / max(size, 1):.1f}x)")

    t0 = time.perf_counter()
    back = codec.load(out)
    print(f"loaded      : {time.perf_counter() - t0:.2f} s")
    ok, where = codec.equal(data, back)
    print(f"round trip  : {'EXACT' if ok else 'MISMATCH at ' + where}")

    # smoke journals: the one thing where size depends on cleverness, so prove it
    checked = bad = 0
    for r in data["rounds"]:
        for s in r["smoke"]:
            for tick in voxel.change_ticks(s):
                j = voxel.journal_at(s, tick)
                st = voxel.state_at(s, tick)
                checked += 1
                if j is None or len(j) != st[0]:
                    bad += 1
    print(f"voxel slice : {checked - bad}/{checked} journals match their declared size")

    # velocity integrity: demoparser2 returns velocities from the wrong tick when
    # it is handed a strided tick list, so prove the workaround is in effect.
    worst_id = worst_vz = 0.0
    for r in data["rounds"]:
        v = np.asarray([p["v"] for p in r["players"]], dtype=np.float64)
        vx = np.asarray([p["vx"] for p in r["players"]], dtype=np.float64)
        vy = np.asarray([p["vy"] for p in r["players"]], dtype=np.float64)
        vz = np.asarray([p["vz"] for p in r["players"]], dtype=np.float64)
        sp = np.hypot(vx, vy)
        ok_m = np.isfinite(v) & np.isfinite(sp)
        if ok_m.any():
            worst_id = max(worst_id, float(np.abs(v - sp)[ok_m].max()))
        fin = np.isfinite(vz)
        if fin.any():
            worst_vz = max(worst_vz, float(np.abs(vz[fin]).max()))
    # 3000 HU/s: correct data tops out at ~1152 (a long fall), corruption at
    # >= 22,546 for the strides this tool uses.  See tests/test_roundtrip.py.
    vel_ok = worst_id < 1e-2 and worst_vz < 3000.0
    print(f"velocity    : {'SANE' if vel_ok else 'CORRUPT'} "
          f"(max |v - hypot(vx,vy)| = {worst_id:.4f}, max |vz| = {worst_vz:.0f} HU/s)")
    return 0 if (ok and bad == 0 and vel_ok) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="demo_extract",
        description="Extract per-frame CS2 demo data (players, events, fire, smoke voxels).",
    )
    ap.add_argument("demo", nargs="?", help="path to a .dem (or .csz with --inspect)")
    ap.add_argument("-o", "--output", type=Path, help="output .csz path")
    ap.add_argument("--interval", type=float, default=None,
                    help="seconds between samples (default: every tick)")
    ap.add_argument("--stride", type=int, default=None,
                    help="sample every Nth tick (1 = every tick)")
    ap.add_argument("--start-offset", type=float, default=0.0,
                    help="seconds into the round where interval sampling begins")
    ap.add_argument("--level", type=int, default=codec.DEFAULT_LEVEL,
                    help="zstd compression level (default 19)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write human-readable JSON here")
    ap.add_argument("--no-fire-smoke", action="store_true",
                    help="skip the demoinfocs side-channel (much smaller, loses fire/smoke)")
    ap.add_argument("--no-inventory", action="store_true",
                    help="drop the per-tick inventory column")
    ap.add_argument("--no-spotted", action="store_true",
                    help="drop the per-tick spotted-by column")
    ap.add_argument("--build-dir", type=Path, default=None,
                    help="where the Go side-channel binary lives")
    ap.add_argument("--build", action="store_true",
                    help="rebuild the Go side-channel binary, then exit")
    ap.add_argument("--inspect", action="store_true",
                    help="read a .csz and describe it, without re-parsing the demo")
    ap.add_argument("--selftest", action="store_true",
                    help="extract, save, load, and verify exactness")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.build:
        from . import sidechannel

        p = sidechannel.build(args.build_dir, verbose=True)
        print(f"built {p}")
        return 0

    if args.inspect:
        if not args.demo:
            ap.error("--inspect needs a .csz path")
        return _inspect(Path(args.demo))

    if not args.demo:
        ap.error("a .dem path is required")
    demo = Path(args.demo)
    if not demo.is_file():
        ap.error(f"no such file: {demo}")

    if args.selftest:
        return _selftest(demo, args)

    data = extract(
        demo,
        interval=args.interval,
        stride=args.stride,
        start_offset=args.start_offset,
        include_fire_smoke=not args.no_fire_smoke,
        include_inventory=not args.no_inventory,
        include_spotted=not args.no_spotted,
        build_dir=args.build_dir,
        verbose=not args.quiet,
    )
    out = args.output or demo.with_suffix(".csz")
    codec.save(data, out, level=args.level)
    size = Path(out).stat().st_size
    raw = len(codec.raw_blob(data))
    print(f"wrote {out}  ({_human(size)} on disk, {_human(raw)} uncompressed, "
          f"{raw / max(size, 1):.1f}x)")
    if args.json:
        codec.save_json(data, args.json)
        print(f"wrote {args.json}  ({_human(Path(args.json).stat().st_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())