"""
Radar base images for thumbnails: CS2's own radar art, with an OBJ fallback.

Where the image comes from
--------------------------
The game's radar art is staged per map into
``visualizer/static/assets/radar/`` (checked in with the repo): the art itself
(``panorama/images/overheadmaps/<map>_radar_psd``) and ``map.json``, which holds
the transform read out of ``resource/overviews/<map>.txt``.  That is the same
image and the same numbers the in-game radar uses, so a player marker lands on
the right floor by construction rather than by tuning.

Before that existed this module rendered its own plate from the map OBJ.  That
path is still here as the fallback for a map whose art was never staged (and for
a checkout that predates the staged assets), and it is why the coordinate
derivation below still matters -- both paths produce the same ``MapRadar``
contract: an image, the game-space rectangle it covers, and a mask.

Coordinate system
-----------------
``scene.js`` converts game units to three.js with::

    three = (gy, gz, gx) * SCALE        SCALE = 0.0254

and drops the OBJ into the scene untransformed, so the OBJ *is* in three.js
space.  Inverting that, for a vertex ``(ox, oy, oz)`` of the map::

    demo.x = oz / SCALE
    demo.y = ox / SCALE
    demo.z = oy / SCALE

Both radars are top-down projections using ``u = demo.x`` (image column) and
``v = demo.y`` (image row, flipped so +y is up), which is also what the overview
files mean by ``pos_x``/``pos_y``/``scale``.  ``app/tests/test_radar.py`` checks
this by asserting that demo player positions land on map pixels, and that the
four wrong axis assignments do not.

How the fallback image is built
-------------------------------
de_mirage is 738k vertices / 1.2M triangles, and a per-triangle Python loop is
far too slow for an ingest path.  Two stages, both vectorised:

1. **Sample the surface.**  Every triangle gets a number of barycentric samples
   proportional to its *projected pixel area*, so a huge floor triangle and a
   tiny wall triangle contribute comparably to coverage.  This is what makes
   the result solid -- splatting only the vertices leaves large flat floors
   almost empty, which is exactly where players walk.
2. **Reduce to a height field.**  Samples are scattered into per-pixel buffers
   with a sort-and-overwrite trick::

       order = argsort(y); buf[pix[order]] = y[order]   # last write wins = max

   giving the highest and lowest surface per pixel, plus a coverage count.

Shading then uses the floor height for a terrain ramp and hillshade, and
darkens pixels whose floor-to-ceiling gap is large so buildings read as mass.

Results are cached per map on disk, so a map is rendered once and every demo on
it reuses the same base image.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import config

#: Resolution of the cached per-map radar.
RADAR_SIZE = 1024
#: Size of the per-demo thumbnail written to disk.
THUMB_SIZE = (640, 400)
#: Barycentric samples per pixel of projected triangle area.  Higher = smoother
#: coverage at a linear cost in time.
SAMPLE_DENSITY = 3.0
#: Hard ceiling on total samples, to bound memory on very dense maps.
MAX_SAMPLES = 6_000_000
#: Triangles processed per chunk; keeps peak memory around a few tens of MB.
_CHUNK = 60_000

SCALE = 0.0254  # game units -> three.js units, must match visualizer/static/js/scene.js

# Bump when the rendering changes: the PNG is cached on disk and a stale one
# is indistinguishable from a wrong transform at a glance.  v3 = area-weighted
# sampling without the per-triangle cap.
_CACHE_VERSION = "v3"


# ── OBJ reading ──────────────────────────────────────────────────────────────


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an OBJ into ``(vertices (N,3) float32, triangles (M,3) int32)``.

    Only ``v`` and ``f`` lines exist in these files; ``f`` entries may be
    ``a``, ``a/b`` or ``a/b/c`` and faces are fan-triangulated so quads and
    ngons are handled too.
    """
    raw = path.read_bytes()
    lines = raw.split(b"\n")

    verts = np.array([ln[2:].split() for ln in lines if ln[:2] == b"v "], dtype=np.float32)
    if verts.size == 0:
        raise ValueError(f"no vertices in {path}")

    tris: list[tuple[int, int, int]] = []
    for ln in lines:
        if ln[:2] != b"f ":
            continue
        try:
            idx = [int(p.split(b"/")[0]) - 1 for p in ln[2:].split()]
        except ValueError:
            continue
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))

    if not tris:
        return verts, np.zeros((0, 3), dtype=np.int32)

    tri = np.asarray(tris, dtype=np.int64)
    n = len(verts)
    keep = np.all((tri >= 0) & (tri < n), axis=1)
    return verts, tri[keep].astype(np.int32)


def _uv_y(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OBJ vertices -> radar space. See the module docstring for the derivation."""
    u = verts[:, 2] / SCALE
    v = verts[:, 0] / SCALE
    y = verts[:, 1] / SCALE
    return np.stack([u, v], axis=1).astype(np.float64), y.astype(np.float64)


# ── surface sampling ─────────────────────────────────────────────────────────


def _sample_surface(
    verts: np.ndarray, tris: np.ndarray, size: int, extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted barycentric sampling of the mesh, in pixel space.

    Returns ``(flat_pixel_index, y, weight)`` -- ``weight`` is unused by the
    current shading but kept so callers can density-weight later.
    """
    u0, u1, v0, v1 = extent
    du, dv = u1 - u0, v1 - v0

    uv, y = _uv_y(verts)

    def to_px(uv_arr: np.ndarray) -> np.ndarray:
        """(..., 2) radar-space -> (..., 2) pixel-space, keeping the leading shape."""
        px = (uv_arr[..., 0] - u0) / du * size
        py = (v1 - uv_arr[..., 1]) / dv * size  # +v is up
        return np.stack([px, py], axis=-1)

    if len(tris) == 0:
        px = to_px(uv)
        ix = np.clip(px[:, 0].astype(np.int64), 0, size - 1)
        iy = np.clip(px[:, 1].astype(np.int64), 0, size - 1)
        return iy * size + ix, y, np.ones(len(y))

    rng = np.random.default_rng(12345)  # deterministic: same map -> same image

    # Pass 1: projected area per triangle, in pixels (shoelace).  Sampling
    # effort should follow on-screen size, not triangle count -- a map's floor
    # is a handful of huge triangles and its detail is thousands of tiny ones.
    areas = np.empty(len(tris), dtype=np.float64)
    for start in range(0, len(tris), _CHUNK):
        t = tris[start:start + _CHUNK]
        p = to_px(uv[t])                       # (K, 3, 2)
        ax = p[:, 1, 0] - p[:, 0, 0]
        ay = p[:, 1, 1] - p[:, 0, 1]
        bx = p[:, 2, 0] - p[:, 0, 0]
        by = p[:, 2, 1] - p[:, 0, 1]
        areas[start:start + _CHUNK] = np.abs(ax * by - bx * ay) * 0.5

    # Pass 2: allocate.  Density drives the count; the global budget is the
    # only ceiling.  There used to be a hard per-triangle cap of 4096 here,
    # which silently starved any map whose geometry is a few large planes --
    # a 2x2-unit test floor came out 1.5% covered instead of 92%.
    k = np.maximum(areas * SAMPLE_DENSITY, 1.0)
    total = float(k.sum())
    if total > MAX_SAMPLES:
        k *= MAX_SAMPLES / total
    k = np.ceil(np.maximum(k, 1.0)).astype(np.int64)  # never skip a triangle
    total = int(k.sum())
    if total == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    all_idx: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for start in range(0, len(tris), _CHUNK):
        t = tris[start:start + _CHUNK]
        p = to_px(uv[t])                       # (K, 3, 2)
        ty = y[t]                              # (K, 3)
        kk = k[start:start + _CHUNK]
        n_here = int(kk.sum())
        if n_here == 0:
            continue

        # Uniform point in the triangle via the standard sqrt trick.
        r1 = np.sqrt(rng.random(n_here))
        r2 = rng.random(n_here)
        w0 = 1.0 - r1
        w1 = r1 * (1.0 - r2)
        w2 = r1 * r2

        rep = np.repeat(np.arange(len(t), dtype=np.int64), kk)
        px = w0 * p[rep, 0, 0] + w1 * p[rep, 1, 0] + w2 * p[rep, 2, 0]
        py = w0 * p[rep, 0, 1] + w1 * p[rep, 1, 1] + w2 * p[rep, 2, 1]
        sy = w0 * ty[rep, 0] + w1 * ty[rep, 1] + w2 * ty[rep, 2]

        ix = np.clip(px.astype(np.int64), 0, size - 1)
        iy = np.clip(py.astype(np.int64), 0, size - 1)
        all_idx.append(iy * size + ix)
        all_y.append(sy)

    if not all_idx:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    return np.concatenate(all_idx), np.concatenate(all_y), np.ones(
        sum(len(a) for a in all_idx), dtype=np.float64
    )


def _reduce_height_field(
    flat: np.ndarray, y: np.ndarray, n_cells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter samples into per-pixel ``(top, bottom, count)``.

    Sorting by height and letting the last write win is the whole rasteriser:
    ascending order leaves the maximum, descending leaves the minimum.  It is
    done per chunk against ``-inf``/``+inf`` sentinels and folded with
    ``maximum``/``minimum`` so peak memory stays bounded.
    """
    top = np.full(n_cells, -np.inf, dtype=np.float64)
    bot = np.full(n_cells, np.inf, dtype=np.float64)
    count = np.zeros(n_cells, dtype=np.int64)

    step = 2_000_000
    for s in range(0, len(flat), step):
        f = flat[s:s + step]
        v = y[s:s + step]
        order = np.argsort(v, kind="stable")

        buf = np.full(n_cells, -np.inf, dtype=np.float64)
        buf[f[order]] = v[order]
        np.maximum(top, buf, out=top)

        buf.fill(np.inf)
        buf[f[order[::-1]]] = v[order[::-1]]
        np.minimum(bot, buf, out=bot)

        count += np.bincount(f, minlength=n_cells)

    return top, bot, count


# ── shading ──────────────────────────────────────────────────────────────────


class MapRadar:
    """A top-down base image for one map, plus its world transform."""

    def __init__(self, image: np.ndarray, extent: tuple[float, float, float, float],
                 size: int = RADAR_SIZE, mask: Optional[np.ndarray] = None) -> None:
        self.image = image                     # (size, size, 3) uint8
        self.u0, self.u1, self.v0, self.v1 = extent
        self.size = size
        #: Pixels that carry map geometry.  Radar art knows this exactly (its
        #: alpha channel); a rendered plate only has a background colour, so the
        #: mask is derived from that in :meth:`filled_mask` instead.
        self._mask = mask

    def world_to_px(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Game (x, y) -> image pixel (col, row). Accepts scalars or arrays."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        du = max(self.u1 - self.u0, 1e-6)
        dv = max(self.v1 - self.v0, 1e-6)
        px = (x - self.u0) / du * self.size
        py = (self.v1 - y) / dv * self.size  # +v is up
        return px, py

    def filled_mask(self) -> np.ndarray:
        """Pixels that carry geometry (i.e. are not the background plate)."""
        if self._mask is not None:
            return self._mask
        bg = np.array([14, 18, 23], dtype=np.int16)
        return np.abs(self.image.astype(np.int16) - bg).sum(axis=2) > 12


def _render_radar(verts: np.ndarray, tris: np.ndarray, size: int = RADAR_SIZE) -> MapRadar:
    uv, _ = _uv_y(verts)
    u0, u1 = float(uv[:, 0].min()), float(uv[:, 0].max())
    v0, v1 = float(uv[:, 1].min()), float(uv[:, 1].max())
    pad = 0.02 * max(u1 - u0, v1 - v0, 1.0)
    extent = (u0 - pad, u1 + pad, v0 - pad, v1 + pad)

    flat, y, _ = _sample_surface(verts, tris, size, extent)
    n_cells = size * size
    top, bot, count = _reduce_height_field(flat, y, n_cells)

    top = top.reshape(size, size)
    bot = bot.reshape(size, size)
    filled = count.reshape(size, size) > 0

    floor = _fill_holes(bot, filled)
    valid = floor[filled]
    if valid.size:
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-3:
        hi = lo + 1.0
    norm = np.clip((floor - lo) / (hi - lo), 0, 1)

    shade = _hillshade(floor)

    # Low ground light, high ground slightly darker -> reads as relief.
    lum = (0.60 + 0.32 * (1.0 - norm)) * shade

    # Mass: a large floor-to-ceiling gap means a building/wall block.
    gap = np.where(filled, top - bot, 0.0)
    if filled.any():
        ref = max(float(np.percentile(gap[filled], 95)), 1.0)
    else:
        ref = 1.0
    gnorm = np.clip(gap / ref, 0, 1)
    lum *= 1.0 - 0.55 * gnorm

    img = np.zeros((size, size, 3), dtype=np.float64)
    img[..., 0] = 0.055
    img[..., 1] = 0.070
    img[..., 2] = 0.090
    tone = np.stack([lum * 0.94, lum * 0.97, lum * 1.00], axis=-1)
    img = np.where(filled[..., None], tone, img)

    return MapRadar((np.clip(img, 0, 1) * 255).astype(np.uint8), extent, size)


def _fill_holes(a: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """Nearest-valid fill by repeated dilation; a handful of passes is plenty."""
    out = np.where(filled, a, 0.0)
    known = filled.copy()
    for _ in range(8):
        if known.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(out.shape, dtype=np.float64)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            kshift = np.roll(np.roll(known, dy, axis=0), dx, axis=1)
            acc += np.where(kshift, shifted, 0.0)
            cnt += kshift
        newly = (~known) & (cnt > 0)
        out[newly] = acc[newly] / cnt[newly]
        known |= newly
    return out


def _hillshade(floor: np.ndarray) -> np.ndarray:
    """Gentle hillshade over the floor height field.

    A pixel is a fraction of a world unit, so the raw gradient is enormous and
    a textbook hillshade saturates into noise; the gradient is damped first.
    """
    gy, gx = np.gradient(floor)
    gx *= 0.35
    gy *= 0.35
    az = math.radians(315.0)
    alt = math.radians(45.0)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return 0.72 + 0.28 * np.clip(shaded, 0.0, 1.0)


# ── per-map cache ────────────────────────────────────────────────────────────


def map_obj_path(map_name: str) -> Optional[Path]:
    if not map_name:
        return None
    p = config.MAPS_DIR / f"{map_name}.obj"
    return p if p.is_file() else None


#: Staged CS2 radar art (checked in).  Same files the browser loads for the
#: replay minimap, so the thumbnail and the minimap cannot drift apart.
RADAR_DIR = config.ROOT / "visualizer" / "static" / "assets" / "radar"

_staged_cache: dict[str, Any] | None = None
_staged_mtime: float = -1.0


def _staged_table() -> dict[str, Any]:
    """``radar/map.json``, re-read when it changes."""
    global _staged_cache, _staged_mtime
    path = RADAR_DIR / "map.json"
    if not path.is_file():
        return {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _staged_cache is None or mtime != _staged_mtime:
        try:
            _staged_cache = json.loads(path.read_text(encoding="utf-8"))
            _staged_mtime = mtime
        except (OSError, ValueError):
            return {}
    return _staged_cache or {}


def _load_staged_radar(map_name: str) -> Optional[MapRadar]:
    """Radar art + the overview transform, or None when this map was not staged.

    These images are RGBA with a transparent surround; the mask comes from that
    alpha, and the RGB is composited over the card plate up front so every
    consumer sees the same opaque image.
    """
    from PIL import Image

    info = _staged_table().get(map_name)
    if not info:
        return None
    path = RADAR_DIR / info["image"]
    if not path.is_file():
        return None

    src = Image.open(path).convert("RGBA")
    w, h = src.size
    if w != h:                                  # MapRadar's transform is square
        return None
    alpha = np.asarray(src.getchannel("A"))
    plate = Image.new("RGBA", (w, h), (10, 14, 20, 255))
    img = np.asarray(Image.alpha_composite(plate, src).convert("RGB"))

    scale = float(info["scale"])
    pos_x = float(info["pos_x"])
    pos_y = float(info["pos_y"])
    extent = (pos_x, pos_x + w * scale, pos_y - h * scale, pos_y)
    return MapRadar(img, extent, w, mask=alpha > 16)


def _cache_path(map_name: str) -> Path:
    return config.THUMB_DIR / "_maps" / f"{map_name}.{_CACHE_VERSION}.png"


def available_maps() -> list[str]:
    if not config.MAPS_DIR.is_dir():
        return []
    return sorted(p.stem for p in config.MAPS_DIR.glob("*.obj"))


def get_map_radar(map_name: str, *, rebuild: bool = False) -> Optional[MapRadar]:
    """Return the radar for ``map_name``, staging radar art preferred.

    CS2 ships a hand-drawn radar per map (`panorama/images/overheadmaps/
    <map>_radar_psd.vtex_c`) plus the exact transform in `resource/overviews/
    <map>.txt`; when those are staged the OBJ render is not used at all --
    the art is what the game itself shows, and it lines up to the pixel.

    Falls back to rendering the OBJ grey-model, and to ``None`` when there is
    neither, so callers can draw a plain card rather than fail.
    """
    if map_name and not rebuild:
        staged = _load_staged_radar(map_name)
        if staged is not None:
            return staged

    from PIL import Image

    obj = map_obj_path(map_name)
    if obj is None:
        return None

    cache = _cache_path(map_name)
    extent_file = cache.with_suffix(".extent")
    if cache.is_file() and extent_file.is_file() and not rebuild:
        try:
            img = np.asarray(Image.open(cache).convert("RGB"))
            u0, u1, v0, v1 = (float(x) for x in extent_file.read_text().split())
            return MapRadar(img, (u0, u1, v0, v1), img.shape[0])
        except (OSError, ValueError):
            pass  # fall through and re-render

    verts, tris = _read_obj(obj)
    radar = _render_radar(verts, tris)

    cache.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(radar.image).save(cache, format="PNG", optimize=True)
    extent_file.write_text(
        f"{radar.u0} {radar.u1} {radar.v0} {radar.v1}", encoding="utf-8"
    )
    return radar


# ── per-demo thumbnail ───────────────────────────────────────────────────────


_TEAM_COLORS = {0: (255, 196, 84), 1: (86, 182, 255)}

#: Cap on trace points per player per round.  A per-tick round is 5-7k samples
#: at 64 Hz; 240 keeps the map readable and the card cheap to draw.
_TRACE_POINTS = 240


def _trace_runs(data: dict) -> list[tuple[str, list[tuple[float, float]]]]:
    """One ``(team, [(x, y), ...])`` run per player life, in game coordinates.

    A trace must never be one polyline per player index across the whole match.
    Each round starts at a spawn point while the previous round ended wherever
    the player happened to die, so joining them draws a straight chord across
    the map -- with ~16 round boundaries and 10 players that is roughly 160 long
    chords, which is what made the library cards look like everyone was
    teleporting back to spawn.  Splitting on the round boundary is the fix;
    filtering dead players alone is not enough, because a player who survives to
    the end of a round has no dead gap at the boundary at all.

    Dead samples are dropped for the same reason they are useless: the
    extractor keeps the corpse's position frozen at the death spot instead of
    NaN, so they add a stationary tail and nothing else.  Missing ``alive``
    (older/synthetic payloads) is treated as "alive throughout".
    """
    runs: list[tuple[str, list[tuple[float, float]]]] = []
    for rnd in data.get("rounds") or []:
        teams = rnd.get("teams") or []
        for i, p in enumerate(rnd.get("players") or []):
            xs = p.get("x") or []
            ys = p.get("y") or []
            if not xs or not ys:
                continue
            alive = p.get("alive") or []
            team = teams[i] if i < len(teams) else "?"
            step = max(1, len(xs) // _TRACE_POINTS)
            run: list[tuple[float, float]] = []
            for k in range(0, min(len(xs), len(ys)), step):
                if k < len(alive) and not alive[k]:
                    if len(run) > 1:
                        runs.append((team, run))
                    run = []
                    continue
                try:
                    x, y = float(xs[k]), float(ys[k])
                except (TypeError, ValueError):
                    x = y = math.nan
                if not (math.isfinite(x) and math.isfinite(y)):
                    if len(run) > 1:
                        runs.append((team, run))
                    run = []
                    continue
                run.append((x, y))
            if len(run) > 1:
                runs.append((team, run))
    return runs


def _thumb_crop(radar: MapRadar, W: int, H: int):
    """Crop box (radar pixels) + the game extent it covers, for a ``W:H`` card.

    Centred on the map's own content rather than the image, because the art has
    a transparent margin whose size differs per map; clamping keeps the box
    inside the image when a map hugs one edge.
    """
    n = radar.size
    mask = radar.filled_mask()
    ys, xs = np.nonzero(mask)
    if len(xs):
        cx = (float(xs.min()) + float(xs.max())) / 2.0
        cy = (float(ys.min()) + float(ys.max())) / 2.0
    else:
        cx = cy = n / 2.0

    if W >= H:
        cw, ch = float(n), n * H / W
    else:
        cw, ch = n * W / H, float(n)
    left = min(max(cx - cw / 2.0, 0.0), n - cw)
    top = min(max(cy - ch / 2.0, 0.0), n - ch)
    box = (int(round(left)), int(round(top)),
           int(round(left + cw)), int(round(top + ch)))

    du = radar.u1 - radar.u0
    dv = radar.v1 - radar.v0
    x0 = radar.u0 + (box[0] / n) * du
    x1 = radar.u0 + (box[2] / n) * du
    y1 = radar.v1 - (box[1] / n) * dv
    y0 = radar.v1 - (box[3] / n) * dv
    return box, (x0, x1, y0, y1)


def render_demo_thumbnail(
    data: dict,
    *,
    map_name: str,
    team_names: tuple[str, str] = ("Team A", "Team B"),
    score: tuple[Optional[int], Optional[int]] = (None, None),
    out_path: Path,
) -> bool:
    """Composite a per-demo thumbnail: radar + player traces + score badge.

    Traces come from the demo, so two matches on the same map get visibly
    different thumbnails.  They are drawn one *life* at a time (see
    :func:`_trace_runs`): joining rounds or deaths would draw straight chords
    across the map instead of paths.  Returns True when an image was written.

    Watched players used to be traced on top in their own colour.  That is gone
    on purpose: the image is baked at extraction time, so editing the watched
    list left a pink line for a player the user had just unfollowed, and the
    only way to fix it was to reload every ``.csz`` (7-17 s each).  The card
    already highlights them, and the card can follow the list for free.
    """
    from PIL import Image, ImageDraw

    radar = get_map_radar(map_name)
    if radar is None:
        return False

    W, H = THUMB_SIZE
    # Crop (never squash) to the card's aspect.  Radar art is square in world
    # space, so resizing it straight to 640x400 would stretch the map by 1.6x
    # horizontally -- and the card's `object-fit: cover` then crops it again.
    box, (u0, u1, v0, v1) = _thumb_crop(radar, W, H)
    base = (Image.fromarray(radar.image)
            .crop(box).resize((W, H), Image.LANCZOS).convert("RGB"))
    # Dim the radar so traces and text stay readable on top of it.  Radar art is
    # far darker than the old rendered plate, so it needs much less of this --
    # at 0.45 the map itself disappeared and the card became a ball of lines.
    base = Image.blend(base, Image.new("RGB", (W, H), (6, 9, 13)), 0.22)
    draw = ImageDraw.Draw(base, "RGBA")

    du = u1 - u0
    dv = v1 - v0

    def to_px(x: float, y: float) -> tuple[float, float]:
        return ((x - u0) / du * W, (v1 - y) / dv * H)

    # One polyline per life, and only over points that survive the crop.  A
    # point outside the card is a break, not a straight line back into view.
    for team, pts in _trace_runs(data):
        color = _TEAM_COLORS[0 if team == "CT" else 1]
        seg: list[tuple[float, float]] = []
        for x, y in pts:
            px, py = to_px(x, y)
            if 0 <= px < W and 0 <= py < H:
                seg.append((px, py))
            else:
                if len(seg) > 1:
                    draw.line(seg, fill=(*color, 46), width=1)
                seg = []
        if len(seg) > 1:
            draw.line(seg, fill=(*color, 46), width=1)

    a, b = score
    badge = (
        f"{team_names[0]}  {a} : {b}  {team_names[1]}"
        if a is not None and b is not None
        else f"{team_names[0]}  vs  {team_names[1]}"
    )
    _text_with_backdrop(draw, (10, 9), badge, size=15)
    if map_name:
        _text_with_backdrop(draw, (10, H - 27), map_name, size=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".png.tmp")
    base.save(tmp, format="PNG", optimize=True)
    tmp.replace(out_path)
    return True


_FONT_CACHE: dict[int, Any] = {}


def _load_font(size: int):
    """A CJK-capable font when one exists, else Pillow's bitmap default.

    Team and player names in this library are frequently Chinese, and the
    built-in default renders those as empty boxes.
    """
    from PIL import ImageFont

    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhl.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    font = None
    for c in candidates:
        if Path(c).is_file():
            try:
                font = ImageFont.truetype(c, size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _text_with_backdrop(draw, xy, text: str, size: int = 14) -> None:
    """Draw text on a translucent plate so it survives a busy background."""
    font = _load_font(size)
    try:
        box = draw.textbbox(xy, text, font=font)
    except (ValueError, AttributeError):
        return
    pad = 4
    draw.rectangle(
        (box[0] - pad, box[1] - pad + 1, box[2] + pad, box[3] + pad),
        fill=(4, 7, 11, 190),
    )
    draw.text(xy, text, font=font, fill=(226, 240, 252, 255))
