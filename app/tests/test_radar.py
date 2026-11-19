#!/usr/bin/env python3
"""
Radar thumbnail checks.

The thing worth testing here is the **coordinate transform**. ``scene.js``
converts game units to three.js with ``three = (gy, gz, gx) * SCALE`` and drops
the map OBJ into the scene untransformed, so the radar has to invert that
exactly.  Get it wrong -- swap the axes, forget the flip -- and the thumbnail
still *renders*, it just draws players walking through walls.  Nothing crashes.

So the load-bearing test is the last one: project real demo player positions
through the radar transform and require that they land on rendered geometry,
then require that the four wrong axis assignments land on it *less* often.  It
needs a ``.csz`` and is skipped when none is available.

    python app/tests/test_radar.py
    CS2_CSZ=/path/to/match.csz python app/tests/test_radar.py
"""

from __future__ import annotations

import contextlib
import inspect
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

# The Windows console defaults to GBK, which cannot encode the check marks below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from app import config, radar  # noqa: E402


@contextlib.contextmanager
def scratch():
    """A temp directory inside the workspace; see test_library.scratch."""
    base = _ROOT / ".build" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

_passed = 0
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  \u2713 {name}")
    else:
        _failed.append(name)
        print(f"  \u2717 {name}" + (f"\n      {detail}" if detail else ""))


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r}, want {want!r}")


# ── the transform ────────────────────────────────────────────────────────────


def test_axis_mapping() -> None:
    print("\naxis mapping (must match scene.js: three = (gy, gz, gx) * SCALE)")
    s = radar.SCALE
    eq("SCALE matches scene.js", s, 0.0254)

    # An OBJ vertex (ox, oy, oz) corresponds to demo (oz/S, ox/S, oy/S).
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],      # +x in obj  -> +y in demo
        [0.0, 2.0, 0.0],      # +y in obj  -> +z in demo (height)
        [0.0, 0.0, 3.0],      # +z in obj  -> +x in demo
    ], dtype=np.float32)
    uv, y = radar._uv_y(verts)

    # Vertices arrive as float32, so compare with a tolerance rather than ==.
    check("obj +x -> demo.y", np.isclose(uv[1][1], 1.0 / s, rtol=1e-6),
          f"{uv[1][1]} vs {1.0 / s}")
    check("obj +y -> height", np.isclose(y[2], 2.0 / s, rtol=1e-6),
          f"{y[2]} vs {2.0 / s}")
    check("obj +z -> demo.x", np.isclose(uv[3][0], 3.0 / s, rtol=1e-6),
          f"{uv[3][0]} vs {3.0 / s}")
    check("obj y is not a map coordinate", abs(uv[2][0]) < 1e-6 and abs(uv[2][1]) < 1e-6,
          f"got {uv[2]}")
    eq("origin", (float(uv[0][0]), float(uv[0][1]), float(y[0])), (0.0, 0.0, 0.0))


def test_pixel_mapping() -> None:
    print("\nworld -> pixel mapping")
    r = radar.MapRadar(np.zeros((64, 64, 3), dtype=np.uint8),
                       (-1000.0, 3000.0, -2000.0, 2000.0), 64)

    # The extent corners must land exactly on the image corners, which pins
    # both the scale and the direction of each axis.
    px, py = r.world_to_px(-1000.0, 2000.0)
    check("top-left corner (u0, v1)", (px, py) == (0.0, 0.0), f"{px}, {py}")
    px, py = r.world_to_px(3000.0, -2000.0)
    check("bottom-right corner (u1, v0)", (px, py) == (64.0, 64.0), f"{px}, {py}")
    px, py = r.world_to_px(1000.0, 0.0)
    check("centre", (px, py) == (32.0, 32.0), f"{px}, {py}")

    # +y in game space must be *up* the image, i.e. a smaller row index.
    _, py_low = r.world_to_px(0.0, -2000.0)
    _, py_high = r.world_to_px(0.0, 2000.0)
    check("+y maps upward (smaller row)", py_high < py_low, f"{py_high} vs {py_low}")

    # +x in game space must run left -> right.
    px_left, _ = r.world_to_px(-1000.0, 0.0)
    px_right, _ = r.world_to_px(3000.0, 0.0)
    check("+x maps rightward", px_right > px_left, f"{px_right} vs {px_left}")

    # Vectors, because that is how the thumbnail renders traces.
    PX, PY = r.world_to_px(np.array([-1000.0, 3000.0]), np.array([-2000.0, 2000.0]))
    check("accepts arrays", PX.shape == (2,) and PY.shape == (2,), f"{PX.shape}, {PY.shape}")
    check("array result matches scalar result",
          (float(PX[1]), float(PY[0])) == (64.0, 64.0), f"{PX}, {PY}")

    # A degenerate extent must not divide by zero.
    flat = radar.MapRadar(np.zeros((8, 8, 3), dtype=np.uint8), (0.0, 0.0, 0.0, 0.0), 8)
    fx, fy = flat.world_to_px(0.0, 0.0)
    check("degenerate extent stays finite", np.isfinite(fx) and np.isfinite(fy), f"{fx}, {fy}")


def test_trace_runs() -> None:
    """Traces are drawn per life: no chord across a death or a round boundary.

    A single polyline per player across all rounds used to join the end of one
    round (wherever the player died) to the start of the next (a spawn point),
    which drew a long straight line across the map on every round change.
    """
    print("\ntrace runs (one per life, never across rounds)")
    demo = {
        "rounds": [
            {"teams": ["CT"], "players": [{
                "x": [0.0, 10.0, 10.0, 10.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "alive": [True, True, False, False],
            }]},
            {"teams": ["CT"], "players": [{
                "x": [1000.0, 1010.0],
                "y": [500.0, 500.0],
                "alive": [True, True],
            }]},
        ],
    }
    runs = radar._trace_runs(demo)
    eq("each round is its own run", len(runs), 2)
    eq("the first run stops at the death", runs[0][1], [(0.0, 0.0), (10.0, 0.0)])
    eq("the next round starts a separate run", runs[1][1][0], (1000.0, 500.0))
    check("no run contains both the death spot and the spawn",
          all(not (0.0 in [x for x, _ in pts] and 1000.0 in [x for x, _ in pts])
              for _, pts in runs))
    eq("team is carried on the run", runs[0][0], "CT")

    # Payloads without `alive` (the older/synthetic case) keep every sample.
    bare = {"rounds": [{"players": [
        {"x": [0.0, 1.0], "y": [0.0, 1.0]}]}]}
    runs = radar._trace_runs(bare)
    eq("missing alive keeps the whole trace", len(runs), 1)
    eq("missing team falls back to '?'", runs[0][0], "?")
    eq("both points survive", len(runs[0][1]), 2)

    # A NaN position is a gap, not a point to interpolate through.
    gap = {"rounds": [{"teams": ["T"], "players": [{
        "x": [0.0, 1.0, float("nan"), 5.0, 6.0],
        "y": [0.0, 1.0, float("nan"), 5.0, 6.0],
    }]}]}
    runs = radar._trace_runs(gap)
    eq("a NaN splits the run", [len(pts) for _, pts in runs], [2, 2])


def test_synthetic_map() -> None:
    print("\nrendering a synthetic map")
    with scratch() as tmp:
        maps = Path(tmp) / "maps"
        maps.mkdir()
        # A 2x2 unit floor at y=0, plus a 1-unit-tall block in one corner.
        obj = maps / "de_test.obj"
        obj.write_text(
            "v -1 0 -1\nv 1 0 -1\nv 1 0 1\nv -1 0 1\n"
            "v 0 1 0\nv 0.5 1 0\nv 0.5 1 0.5\nv 0 1 0.5\n"
            "f 1 2 3\nf 1 3 4\n"
            "f 5 6 7\nf 5 7 8\n",
            encoding="utf-8",
        )
        old_maps, old_thumbs = config.MAPS_DIR, config.THUMB_DIR
        config.MAPS_DIR = maps
        config.THUMB_DIR = Path(tmp) / "thumbs"
        try:
            r = radar.get_map_radar("de_test")
            check("radar rendered", r is not None)
            if r is None:
                return
            filled = r.filled_mask()
            frac = float(filled.mean())
            check("floor covers most of the frame", 0.5 < frac < 1.0, f"filled={frac:.3f}")

            # The floor spans obj x,z in [-1,1] -> demo x,y in [-1/S, 1/S].
            half = 1.0 / radar.SCALE
            px, py = r.world_to_px(0.0, 0.0)
            inside = 0 <= px < r.size and 0 <= py < r.size
            check("map centre is inside the image", inside, f"{px}, {py}")
            check("map centre is on geometry", bool(filled[int(py), int(px)]),
                  f"pixel ({int(px)}, {int(py)})")

            # A point well outside the map must be background, not geometry.
            px2, py2 = r.world_to_px(half * 4, half * 4)
            out = not (0 <= px2 < r.size and 0 <= py2 < r.size) or \
                not filled[int(py2), int(px2)]
            check("far outside point is background", out)

            # Cache round trip
            r2 = radar.get_map_radar("de_test")
            check("cached extent matches",
                  r2 is not None and (r2.u0, r2.u1, r2.v0, r2.v1) == (r.u0, r.u1, r.v0, r.v1))
            check("unknown map returns None", radar.get_map_radar("de_nope") is None)

            print("\ndemo thumbnail")
            from PIL import Image

            demo = {
                "players": [{"name": "ZywOo"}, {"name": "ropz"}],
                "rounds": [{
                    "teams": ["CT", "T"],
                    "players": [
                        {"x": [-0.5, 0.0, 0.5], "y": [0.0, 0.2, 0.0]},
                        {"x": [0.5, 0.0, -0.5], "y": [0.0, -0.2, 0.0]},
                    ],
                }],
            }
            shot = Path(tmp) / "thumb.png"
            check("thumbnail written", radar.render_demo_thumbnail(
                demo, map_name="de_test", team_names=("A", "B"), score=(13, 7),
                out_path=shot) and shot.is_file())

            # Watched players used to get their own pink trace baked into the
            # image.  A baked highlight cannot follow an editable list, so
            # unfollowing someone left the line behind until every .csz was
            # reloaded.  The feature is gone on purpose; these keep it gone.
            pink = sum(1 for r, g, b in Image.open(shot).convert("RGB").getdata()
                       if abs(r - 219) < 30 and abs(g - 88) < 34 and abs(b - 116) < 34)
            eq("no watched-player trace colour in the image", pink, 0)

            # The traces still have to be drawn -- a blank overlay would also
            # pass the check above.
            bare = Path(tmp) / "bare.png"
            radar.render_demo_thumbnail(
                {**demo, "rounds": []}, map_name="de_test",
                team_names=("A", "B"), score=(13, 7), out_path=bare)
            check("the team traces are still drawn",
                  Image.open(shot).tobytes() != Image.open(bare).tobytes())

            params = inspect.signature(radar.render_demo_thumbnail).parameters
            check("the renderer no longer takes a watched list",
                  "watched_names" not in params, f"{sorted(params)}")
            check("and the watch colour is gone", not hasattr(radar, "_WATCH_COLOR"))
        finally:
            config.MAPS_DIR, config.THUMB_DIR = old_maps, old_thumbs


# ── the load-bearing test ────────────────────────────────────────────────────


def find_csz() -> Path | None:
    env = os.environ.get("CS2_CSZ")
    if env and Path(env).is_file():
        return Path(env)
    for pattern in ("visualizer/outputs/*.csz", "app/data/csz/*.csz", ".build/smoke/*.csz"):
        hits = sorted(_ROOT.glob(pattern))
        if hits:
            return hits[0]
    return None


def test_real_alignment() -> None:
    """Do demo player positions land on the map geometry?"""
    print("\nreal-demo alignment")
    csz = find_csz()
    if csz is None:
        print("  SKIP  no .csz available (set CS2_CSZ, or extract one first)")
        return
    try:
        sys.path.insert(0, str(_ROOT / "tools"))
        from demo_extract import codec
        data = codec.load(csz)
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP  could not read {csz}: {type(exc).__name__}: {exc}")
        return

    map_name = (data.get("meta") or {}).get("map")
    r = radar.get_map_radar(map_name or "")
    if r is None:
        print(f"  SKIP  no map OBJ for {map_name!r}")
        return
    print(f"  using {csz.name}  map={map_name}")

    xs, ys = [], []
    for rnd in data.get("rounds") or []:
        for p in rnd.get("players") or []:
            xs.append(np.asarray(p.get("x") or [], dtype=np.float64))
            ys.append(np.asarray(p.get("y") or [], dtype=np.float64))
    if not xs:
        print("  SKIP  demo has no player positions")
        return
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    check("demo has player samples", len(X) > 1000, f"{len(X)}")

    filled = r.filled_mask()

    def hit(ax, ay) -> float:
        px, py = r.world_to_px(ax, ay)
        ix = np.clip(px.astype(int), 0, r.size - 1)
        iy = np.clip(py.astype(int), 0, r.size - 1)
        return float(filled[iy, ix].mean())

    correct = hit(X, Y)
    print(f"      correct transform      : {correct:.4f}")
    controls = {
        "swapped (y,x)": hit(Y, X),
        "flip x   (-x,y)": hit(-X, Y),
        "flip y   (x,-y)": hit(X, -Y),
        "flip both": hit(-X, -Y),
    }
    for name, val in controls.items():
        print(f"      control {name:<16}: {val:.4f}")

    check("correct transform finds the map", correct > 0.80, f"only {correct:.3f}")
    worst = max(controls.values())
    check("correct beats every wrong axis assignment", correct > worst + 0.05,
          f"correct={correct:.3f} best control={worst:.3f} ({[k for k, v in controls.items() if v == worst][0]})")


def main() -> int:
    test_axis_mapping()
    test_pixel_mapping()
    test_trace_runs()
    test_synthetic_map()
    test_real_alignment()
    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        for name in _failed:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
