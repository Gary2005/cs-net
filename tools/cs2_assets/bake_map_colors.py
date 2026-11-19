#!/usr/bin/env python3
"""Bake per-triangle colours for the optimized map OBJs out of CS2's own materials.

The OBJs in ``maps/optimized_obj_files/`` were exported by Open3D as a bare
triangle mesh: no ``vt``, no ``vn``, no ``usemtl`` (verified: 0 of each). The web
viewer therefore paints them with a synthetic "concrete" vertex colour and they
look like a grey clay model.

The game still has all of it. A CS2 map VPK stores the world as ``worldnodes/``
- 784 aggregated meshes for de_inferno - each with ``TEXCOORD_0`` and, in the
99% case, exactly one material. This script reconstructs the missing link:

  1. export every world node to glTF (Source2Viewer resolves the .vmat and dumps
     its textures for us; decompiling a .vmat_c directly crashes on the current
     shader version, see ``_note_shader_bug``),
  2. build a KD-tree over the *original* triangle centroids,
  3. for every triangle of the optimized OBJ, take the nearest original
     triangle, inherit its material and sample that material's base colour
     texture at its UV,
  4. flatten to one colour per material, quantise to a small palette, and give
     every face a smoothed normal,
  5. make duplicate faces share colour + normal, so their z-fighting cannot be
     seen: exact vertex-coincident pairs *and* faces that merely overlap within
     2 mm (brush soup emits the same surface from two brushes, often with a
     different triangulation or a hair-thin offset),
  6. write ``<map>.colors.bin`` - three bytes of sRGB per triangle (+ three bytes
     of int8 normal per triangle in v2).

The two meshes live in the same coordinate space (de_inferno: the game export
and the OBJ have bounding boxes that agree to 1 cm), so no transform is fitted.

Output format (little endian)::

    magic   b"MCLR"
    version u16 = 1
    flags   u16 = 0
    count   u32        number of triangles
    rgb     u8[count*3] sRGB, one colour per triangle, in OBJ face order

``version`` 2 appends ``count*3`` bytes of int8 normals (see ``write_sidecar``).

Why one colour per triangle instead of a texture atlas: the OBJ averages ~10 cm
per triangle, so this is equivalent to a ~10 cm/texel colour map, and it needs
no vendor-specific retopology. See notes/research-gameassets.md for the write-up.
The sidecar is per *face* because the OBJ is a bare triangle soup with no vertex
index of its own; OBJLoader expands it to one vertex per corner anyway, so the
face order is the only stable key between the two sides.

Examples
--------
    python tools/cs2_assets/bake_map_colors.py --map de_inferno
    python tools/cs2_assets/bake_map_colors.py --map de_mirage --force
    python tools/cs2_assets/bake_map_colors.py --map de_inferno --preview out.png
    python tools/cs2_assets/bake_map_colors.py --map de_inferno --restyle
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CLI = ROOT / "tools" / "Source2Viewer-CLI" / "Source2Viewer-CLI.exe"
DEFAULT_GAME = Path(
    r"F:\Program Files\steam\steamapps\common\Counter-Strike Global Offensive"
)
DEFAULT_MAPS = ROOT / "maps" / "optimized_obj_files"
DEFAULT_WORK = ROOT / ".build" / "mapmat" / "work"

#: Cell size of the coarse prefilter grid, in metres. Only used to pick export
#: batches; the nearest-neighbour search itself is a KD-tree.
BATCH = 40

#: 贴花 / 覆盖层（`materials/.../decals/`、`.../overlay/`）**默认不参与**回投影。
#:
#: 它们在游戏里是半透明的独立几何，贴着地面几个毫米；而旁车是"每面一个不透明
#: 颜色"，于是这些面就变成地面上一片杂色的三角形 —— 实测 de_inferno 的地砖上那层
#: "碎三角"主要就是它们（784 个 worldnode 里约 250 个是 overlay）。跳过之后，这些
#: 面会匹配到它们底下那块真地面，颜色就干净了。
DECAL_PATTERNS = (
    "/decals/", "_decal", "/overlay/", "overlay_", "/scorch", "/blood",
)

#: Sidecar constants. Bump VERSION if the layout changes.
#:   v1 = 颜色（每面 3 字节 sRGB）
#:   v2 = 颜色 + 法线（每面再 3 字节 int8），见 `write_sidecar`
MAGIC = b"MCLR"
VERSION = 2

#: Colour textures are downsampled to this before sampling. 256 keeps the
#: material's tone and most of its per-face variation for ~150 KB in memory.
TEX_SIZE = 256

COMPONENT = {5126: ("<f4", 4), 5123: ("<u2", 2), 5125: ("<u4", 4), 5121: ("<u1", 1)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _note_shader_bug() -> str:
    """The CLI version vendored here cannot decompile .vmat_c on CS2's current
    shaders (``vcsFileVersion 72`` -> UnexpectedMagicException). Going through
    ``--gltf_export_materials`` works because that path swallows the failure and
    falls back to the material's raw parameters, which is all we need."""
    return "only glTF material export works with this CLI build"


# ── Source2Viewer plumbing ──────────────────────────────────────────────────
class Cli:
    def __init__(self, exe: Path, game: Path):
        self.exe = exe
        self.game = game
        #: Working directory for the calls. The CLI drops a `exceptions.txt` in
        #: the CWD whenever a material fails to decompile (it does, on CS2's
        #: current shaders), so point it at the work dir instead of the repo root.
        self.cwd: Path | None = None
        if not exe.is_file():
            raise SystemExit(f"Source2Viewer-CLI not found at {exe}")
        gi = game / "game" / "csgo" / "gameinfo.gi"
        if not gi.is_file():
            raise SystemExit(f"gameinfo.gi not found at {gi} (wrong --game?)")
        self.gameinfo = gi

    def call(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(self.exe), "--game", str(self.gameinfo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(self.cwd) if self.cwd else None,
        )

    def list_world_nodes(self, map_vpk: Path, map_name: str) -> list[str]:
        proc = self.call(["-i", str(map_vpk), "-l", "-f", f"maps/{map_name}/worldnodes/"])
        nodes = []
        for line in proc.stdout.splitlines():
            m = re.match(r"^(\S+worldnodes/\S+?\.vmdl_c)\s", line)
            if m:
                nodes.append(m.group(1))
        return sorted(nodes)

    def export(self, map_vpk: Path, nodes: list[str], outdir: Path) -> float:
        """Export one batch of models to GLB with materials + textures."""
        t0 = time.time()
        outdir.mkdir(parents=True, exist_ok=True)
        proc = self.call([
            "-i", str(map_vpk),
            "-f", ",".join(nodes),
            "-o", str(outdir) + "\\",
            "-d",
            "--gltf_export_format", "glb",
            "--gltf_export_materials",
            "--threads", "4",
        ])
        # The CLI exits 0 even when individual files fail; a hard failure shows
        # up as *no* GLBs at all, which the caller detects.
        if proc.returncode != 0 and "Dump written" not in proc.stdout:
            sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        return time.time() - t0


# ── GLB reading ─────────────────────────────────────────────────────────────
def read_glb(path: Path) -> list[dict]:
    """Minimal GLB reader: positions, TEXCOORD_0, indices, per-primitive
    material name and the material's base colour image URI."""
    raw = path.read_bytes()
    magic, _ver, _length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67:
        raise ValueError(f"not a GLB: {path}")
    off = 12
    clen, _ctype = struct.unpack_from("<II", raw, off)
    off += 8
    doc = json.loads(raw[off : off + clen])
    boff = off + clen
    blen, _btype = struct.unpack_from("<II", raw, boff)
    boff += 8
    binc = raw[boff : boff + blen]

    acc = doc.get("accessors", [])
    views = doc.get("bufferViews", [])

    def accessor(i: int) -> np.ndarray:
        a = acc[i]
        bv = views[a["bufferView"]]
        dtype, _ = COMPONENT[a["componentType"]]
        n = NCOMP[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        flat = np.frombuffer(binc, dtype=dtype, count=a["count"] * n, offset=start)
        return flat.reshape(a["count"], n)

    mats = doc.get("materials", [])
    images = doc.get("images", [])
    textures = doc.get("textures", [])

    def material_of(prim: dict) -> tuple[str, str | None]:
        mi = prim.get("material")
        if mi is None or mi >= len(mats):
            return "<none>", None
        m = mats[mi]
        # extras.vmat.Name is the full resource path (stable across maps);
        # the glTF name is only the file stem.
        name = ((m.get("extras") or {}).get("vmat") or {}).get("Name") or m.get("name", "?")
        uri = None
        pbr = m.get("pbrMetallicRoughness") or {}
        bc = pbr.get("baseColorTexture")
        if bc:
            try:
                src = textures[bc["index"]]["source"]
                uri = images[src].get("uri")
            except (KeyError, IndexError):
                uri = None
        return name, uri

    out = []
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos_acc = prim["attributes"].get("POSITION")
            if pos_acc is None:
                continue
            pos = accessor(pos_acc).astype(np.float32)
            idx = accessor(prim["indices"]).ravel() if "indices" in prim else None
            uv_acc = prim["attributes"].get("TEXCOORD_0")
            uv = accessor(uv_acc).astype(np.float32) if uv_acc is not None else None
            name, uri = material_of(prim)
            out.append({"pos": pos, "idx": idx, "uv": uv, "material": name, "image": uri})
    return out


# ── Stage 1: export + build the reference triangle table ────────────────────
def is_decal(material: str) -> bool:
    name = material.lower()
    return any(p in name for p in DECAL_PATTERNS)


def stage_reference(cli: Cli, map_vpk: Path, map_name: str, work: Path,
                    limit: int | None, force: bool, keep_decals: bool = False) -> dict:
    ref_npz = work / "reference.npz"
    ref_json = work / "reference.json"
    tex_dir = work / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)

    if ref_npz.is_file() and ref_json.is_file() and not force:
        cached = json.loads(ref_json.read_text(encoding="utf-8"))
        # 贴花开关会改变参与几何本身，缓存不能跨模式复用；地图 VPK 换过（游戏更新）
        # 也要重建，否则会拿旧版本的材质去配新几何。
        same_vpk = cached.get("vpk_sig") == file_sig(map_vpk).tolist()
        if (bool(cached.get("keep_decals")) == keep_decals
                and bool(cached.get("limited")) == bool(limit) and same_vpk):
            print(f"[ref] cached: {ref_npz} ({ref_npz.stat().st_size / 1e6:.0f} MB)")
            return cached
        print("[ref] cache was built with different options or another game build, rebuilding")

    nodes = cli.list_world_nodes(map_vpk, map_name)
    if limit:
        nodes = nodes[:limit]
    print(f"[ref] {len(nodes)} world node(s) to export")
    cli.cwd = work

    centroids: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    mat_ids: list[np.ndarray] = []
    materials: list[str] = []
    mat_index: dict[str, int] = {}
    mat_image: dict[str, str] = {}
    seen_images: set[str] = set()
    tmp = work / "batch"

    t_start = time.time()
    for b in range(0, len(nodes), BATCH):
        chunk = nodes[b : b + BATCH]
        # Fresh directory per batch: the CLI writes a PNG for *every* texture
        # slot of every material it touches (colour, normal, height, two layers
        # for blend materials - ~10 MB each at 2048²). Letting that accumulate
        # over the whole map is ~10 GB, so colour PNGs are kept and the rest is
        # deleted batch by batch.
        if tmp.exists():
            for p in sorted(tmp.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
        else:
            tmp.mkdir(parents=True)
        dt = cli.export(map_vpk, chunk, tmp)

        glbs = sorted(tmp.rglob("*.glb"))
        needed: set[str] = set()
        skipped = 0
        for glb in glbs:
            for prim in read_glb(glb):
                if not keep_decals and is_decal(prim["material"]):
                    skipped += 1
                    continue
                pos, idx, uv = prim["pos"], prim["idx"], prim["uv"]
                if idx is None:
                    tri = pos.reshape(-1, 3, 3)
                else:
                    tri = pos[idx.reshape(-1, 3)]
                n_tri = tri.shape[0]
                if n_tri == 0:
                    continue
                name = prim["material"]
                if name not in mat_index:
                    mat_index[name] = len(materials)
                    materials.append(name)
                mid = mat_index[name]
                if prim["image"]:
                    mat_image.setdefault(name, prim["image"])
                    needed.add(prim["image"])
                centroids.append(tri.mean(axis=1))
                e1 = tri[:, 1] - tri[:, 0]
                e2 = tri[:, 2] - tri[:, 0]
                nrm = np.cross(e1, e2)
                ln = np.linalg.norm(nrm, axis=1)
                ln[ln == 0] = 1.0
                nrm /= ln[:, None]
                normals.append((nrm * 127).astype(np.int8))
                if uv is not None:
                    if idx is None:
                        uvt = uv.reshape(-1, 3, 2)
                    else:
                        uvt = uv[idx.reshape(-1, 3)]
                    uvs.append(uvt.mean(axis=1).astype(np.float32))
                else:
                    uvs.append(np.full((n_tri, 2), np.nan, dtype=np.float32))
                mat_ids.append(np.full(n_tri, mid, dtype=np.uint16))

        # Keep exactly the files the glTF materials point at, drop everything
        # else (normal / height / orm maps are ~10 MB each at 2048²).
        for png in sorted(tmp.rglob("*.png")):
            if png.name not in needed:
                png.unlink()
                continue
            dst = tex_dir / png.name
            if dst.name in seen_images:
                png.unlink()
            else:
                seen_images.add(dst.name)
                png.replace(dst)
        for leftover in sorted(tmp.rglob("*"), reverse=True):
            if leftover.is_file():
                leftover.unlink()
            elif leftover.is_dir():
                leftover.rmdir()
        print(f"[ref] batch {b // BATCH + 1}/{(len(nodes) + BATCH - 1) // BATCH}: "
              f"{len(glbs)} glb, {dt:.1f}s, {sum(c.shape[0] for c in centroids):,} tris"
              + (f", {skipped} decal mesh(es) skipped" if skipped else ""))

    ref = {
        "centroid": np.concatenate(centroids),
        "normal": np.concatenate(normals),
        "uv": np.concatenate(uvs),
        "mat": np.concatenate(mat_ids),
    }
    np.savez_compressed(ref_npz, **ref)
    meta = {
        "map": map_name,
        "nodes": len(nodes),
        "triangles": int(ref["centroid"].shape[0]),
        "materials": materials,
        "material_image": mat_image,
        "keep_decals": keep_decals,
        "limited": bool(limit),
        "vpk_sig": file_sig(map_vpk).tolist(),
        "seconds": round(time.time() - t_start, 1),
    }
    ref_json.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[ref] {meta['triangles']:,} triangles, {len(materials)} materials, "
          f"{len(seen_images)} colour maps, {meta['seconds']}s")
    return meta


# ── Stage 2: turn the colour PNGs into a material -> rgb LUT ────────────────
def stage_textures(work: Path, meta: dict, force: bool) -> np.ndarray:
    from PIL import Image

    out = work / "textures.npz"
    materials = meta["materials"]
    if out.is_file() and not force:
        data = np.load(out)
        if data["lut"].shape[0] == len(materials):
            print(f"[tex] cached: {out.name}")
            return data["lut"]

    lut = np.full((len(materials), TEX_SIZE, TEX_SIZE, 3), 200, dtype=np.uint8)
    # Load every distinct file once: several materials share one colour map
    # (barrel_kit_01 has five tint/blend variants over the same texture), so
    # deleting a file after the first material used it starves the rest.
    cache: dict[str, np.ndarray] = {}
    have = 0
    for mi, name in enumerate(materials):
        image = meta["material_image"].get(name)
        if not image:
            continue
        if image not in cache:
            p = work / "tex" / image
            if not p.is_file():
                cache[image] = None
                continue
            with Image.open(p) as im:
                # Textures stay in sRGB; the viewer converts to linear on upload.
                small = im.convert("RGB").resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS)
                cache[image] = np.asarray(small, dtype=np.uint8)
            p.unlink()
        if cache[image] is not None:
            lut[mi] = cache[image]
            have += 1
    np.savez_compressed(out, lut=lut)
    print(f"[tex] {have}/{len(materials)} materials have a base colour map "
          f"({len(cache)} distinct files)")
    return lut


# ── Stage 3: the OBJ's own triangles ────────────────────────────────────────
def file_sig(path: Path) -> np.ndarray:
    """(大小, 修改时间) —— 缓存的失效依据。"""
    st = path.stat()
    return np.array([st.st_size, st.st_mtime_ns], dtype=np.int64)


def load_obj(path: Path, cache: Path, force: bool) -> tuple[np.ndarray, np.ndarray]:
    """读 OBJ 的顶点和三角形（带缓存）。

    缓存按**文件签名**失效：你改过 OBJ（哪怕只是删了几个面）之后，`obj.npz` 里的
    顶点/面顺序就作废了 —— 而这正是"颜色按 f 行顺序逐个对齐"的前提。签名对不上
    就重新解析，不会拿旧数据去配新几何。
    """
    sig = file_sig(path)
    if cache.is_file() and not force:
        try:
            data = np.load(cache)
            if "sig" in data.files and np.array_equal(data["sig"], sig):
                return data["verts"], data["tris"]
            print("[obj] OBJ changed since the cache was written, re-parsing")
        except Exception:                      # 缓存坏了就重来，别让烘焙卡在这
            print("[obj] cache unreadable, re-parsing")
    t0 = time.time()
    verts = []
    tris = []
    polys = 0
    with path.open("rb") as f:
        for line in f:
            if line[0] == 118 and line[1] == 32:            # "v "
                verts.append(line[2:].split())
            elif line[0] == 102 and line[1] == 32:          # "f "
                parts = line[2:].split()
                if len(parts) < 3:
                    continue
                # 索引可能是负数（相对当前顶点数），OBJLoader 也支持
                nv = len(verts)
                idx = []
                for p in parts:
                    i = int(p.split(b"/")[0])
                    idx.append(i - 1 if i > 0 else nv + i)
                if len(idx) == 3:
                    tris.append(idx)
                else:
                    # 多边形要按 OBJLoader 的扇面三角化展开（v0, vj-1, vj），
                    # 否则面序和浏览器里的几何对不上，颜色会从第一个四边形开始错位
                    polys += 1
                    for j in range(2, len(idx)):
                        tris.append([idx[0], idx[j - 1], idx[j]])
    v = np.array(verts, dtype=np.float32)
    t = np.array(tris, dtype=np.int32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, verts=v, tris=t, sig=sig)
    print(f"[obj] {len(v):,} verts / {len(t):,} tris in {time.time() - t0:.1f}s"
          + (f" ({polys:,} polygons fan-triangulated)" if polys else ""))
    return v, t


# ── Stage 4: transfer ───────────────────────────────────────────────────────
def smooth_face_colors(centroid: np.ndarray, best: np.ndarray, ref_mat: np.ndarray,
                       rgb: np.ndarray, radius: float, k: int = 5,
                       chunk: int = 250_000) -> np.ndarray:
    """按"邻域里占多数的材质"重涂每个面，去掉逐三角形的材质跳变。

    为什么需要：OBJ 是减面版（de_inferno 8.42M → 3.29M），一个三角形经常横跨两三种
    材质（路缘、土带、贴花），最近三角形匹配就会逐面跳 —— 画面上就是地砖上撒了一层
    细碎的三角色块。取邻域里最多的那个材质，边界只会被推移 ≤ radius，但跳变没了。

    实现：在 OBJ 自己的面重心上建 KD-tree，查 k 个邻居，取它们材质的众数，
    然后直接把**那个邻居的颜色**搬过来 —— 颜色仍然来自游戏贴图，不是平均值。
    """
    from scipy.spatial import cKDTree

    m = len(rgb)
    face_mat = np.full(m, -1, dtype=np.int32)
    hit = best >= 0
    face_mat[hit] = ref_mat[best[hit]].astype(np.int32)
    tree = cKDTree(centroid)
    dist, idx = tree.query(centroid, k=k, distance_upper_bound=radius, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    found = np.isfinite(dist)

    out = rgb.copy()
    ar = np.arange(chunk)
    for i in range(0, m, chunk):
        sl = slice(i, min(i + chunk, m))
        nb = idx[sl]
        ok = found[sl]
        safe = np.where(ok, nb, 0)
        nbm = np.where(ok, face_mat[safe], -1)
        # 众数：每个邻居"有多少个邻居和它同材质"，取最大的那个（并列时取先出现的）
        counts = np.zeros(nbm.shape, dtype=np.int16)
        for j in range(nbm.shape[1]):
            counts += (nbm == nbm[:, j:j + 1]).astype(np.int16)
        pick = counts.argmax(axis=1)
        rows = np.arange(nbm.shape[0])
        use = ok[rows, pick]
        out[i + rows[use]] = rgb[safe[rows[use], pick[use]]]
    changed = int((out != rgb).any(axis=1).sum())
    print(f"[smooth] {changed:,}/{m:,} faces repainted from their neighbourhood "
          f"(k={k}, r={radius} m)")
    return out


def median_smooth_colors(centroid: np.ndarray, rgb: np.ndarray, radius: float,
                         k: int = 9, chunk: int = 250_000) -> np.ndarray:
    """对颜色做一次"邻域中值"滤波，把贴图里的碎斑抹成大片。

    为什么还要这一步：材质粒度解决不了"同一块贴图上的斑驳" —— 比如地面那张 blend
    贴图本身就有明暗土块，逐三角形采样出来就是撒胡椒面；压到 8 色之后这些斑块只是
    变成"几块不同的颜色"，不会消失。中值滤波在保留材质边界的同时把这种斑驳合并掉
    （中值比均值更保边）。

    颜色仍然是"游戏里真实采到的颜色"之一，只是取自邻域里居中的那个。
    """
    from scipy.spatial import cKDTree

    m = len(rgb)
    tree = cKDTree(centroid)
    dist, idx = tree.query(centroid, k=k, distance_upper_bound=radius, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    found = np.isfinite(dist)
    safe = np.where(found, idx, 0)
    out = rgb.copy()
    for i in range(0, m, chunk):
        sl = slice(i, min(i + chunk, m))
        nb = rgb[safe[sl]]                       # (n,k,3)
        med = np.median(nb, axis=1)
        out[sl] = np.clip(np.rint(med), 0, 255).astype(np.uint8)
    changed = int((out != rgb).any(axis=1).sum())
    print(f"[smooth] {changed:,}/{m:,} faces median-smoothed (k={k}, r={radius} m)")
    return out


def match(verts: np.ndarray, tris: np.ndarray, ref: dict, lut: np.ndarray,
          radius: float, normal_cos: float) -> tuple[np.ndarray, dict, np.ndarray]:
    from scipy.spatial import cKDTree

    tri = verts[tris]                       # (M, 3, 3)
    centroid = tri.mean(axis=1)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1)
    ln[ln == 0] = 1.0
    nrm /= ln[:, None]

    t0 = time.time()
    tree = cKDTree(ref["centroid"])
    dist, idx = tree.query(centroid, k=4, distance_upper_bound=radius, workers=-1)
    print(f"[match] KD-tree built + queried in {time.time() - t0:.1f}s")

    ref_n = ref["normal"].astype(np.float32) / 127.0
    mat = ref["mat"]
    uv = ref["uv"]

    best = np.full(len(tris), -1, dtype=np.int64)
    best_d = np.full(len(tris), np.inf, dtype=np.float32)
    valid = np.isfinite(dist)
    for k in range(dist.shape[1]):
        cand = np.where(valid[:, k], idx[:, k], 0)
        d = np.where(valid[:, k], dist[:, k], np.inf)
        # normals are unsigned: a wall's back face shares the same surface
        agree = np.abs((ref_n[cand] * nrm).sum(axis=1)) >= normal_cos
        ok = valid[:, k] & agree & (d < best_d)
        best[ok] = cand[ok]
        best_d[ok] = d[ok]
    # second pass: anything that found no normal-aligned candidate falls back to
    # the plain nearest centroid (decal/overlay meshes on a wall, thin trim)
    miss = best < 0
    if miss.any():
        d2, i2 = tree.query(centroid[miss], k=1, distance_upper_bound=radius * 3, workers=-1)
        ok2 = np.isfinite(d2)
        sub = np.where(miss)[0][ok2]
        best[sub] = i2[ok2]
        best_d[sub] = d2[ok2]

    found = best >= 0
    rgb = np.full((len(tris), 3), 168, dtype=np.uint8)     # neutral grey fallback
    sel = best[found]
    mi = mat[sel].astype(np.int64)
    uu = uv[sel]
    have_uv = np.isfinite(uu[:, 0]) & np.isfinite(uu[:, 1])
    px = np.zeros((sel.shape[0], 3), dtype=np.uint8)
    if have_uv.any():
        ti = mi[have_uv]
        tx = np.clip((uu[have_uv, 0] % 1.0) * (TEX_SIZE - 1), 0, TEX_SIZE - 1).astype(np.int32)
        ty = np.clip((uu[have_uv, 1] % 1.0) * (TEX_SIZE - 1), 0, TEX_SIZE - 1).astype(np.int32)
        px[have_uv] = lut[ti, ty, tx]
    if (~have_uv).any():
        ti = mi[~have_uv]
        px[~have_uv] = lut[ti].reshape(len(ti), -1, 3).mean(axis=1).astype(np.uint8)
    rgb[found] = px

    stats = {
        "faces": int(len(tris)),
        "matched": int(found.sum()),
        "matched_pct": round(100.0 * found.sum() / max(1, len(tris)), 3),
        "no_uv": int((~have_uv).sum()),
        "distance_m": {
            "median": round(float(np.median(best_d[found])), 4) if found.any() else None,
            "p99": round(float(np.percentile(best_d[found], 99)), 4) if found.any() else None,
            "max": round(float(best_d[found].max()), 4) if found.any() else None,
        },
        "materials_used": int(len(np.unique(mat[best[found]]))) if found.any() else 0,
    }
    return rgb, stats, best


def srgb_to_linear_u8(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_u8(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(c * 255.0), 0, 255).astype(np.uint8)


def flat_by_material(verts: np.ndarray, tris: np.ndarray, best: np.ndarray,
                     ref_mat: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """每个材质**一个**颜色（按面积加权平均），彻底去掉贴图斑驳。

    这是"再压一档"的做法：连同一块贴图上的明暗土块也平均掉，整块材质变成一个纯色。
    配合 `--palette` 就得到"每个材质一个色号、全图 N 个颜色"的画法 —— 边界落在材质
    分界上，所以看起来像手工涂色的模型，而不是贴图的马赛克。

    ⚠️ 必须在**线性空间**里平均：直接平均 sRGB 值会整体偏亮（sRGB 是非线性的，
    "灰泥 + 污渍"的平均会被抬成浅米色），整张图发白。第一版就是这么错的。
    """
    m = len(rgb)
    hit = best >= 0
    mat = np.full(m, -1, dtype=np.int64)
    mat[hit] = ref_mat[best[hit]].astype(np.int64)
    tri = verts[tris]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = hit & (area > 0)
    idx = mat[keep]
    w = area[keep]
    lin = srgb_to_linear_u8(rgb[keep])
    slots = int(mat.max()) + 1
    out = rgb.copy()
    mean_lin = np.zeros((slots, 3), dtype=np.float64)
    for k in range(3):
        tot = np.bincount(idx, weights=w, minlength=slots)
        acc = np.bincount(idx, weights=w * lin[:, k], minlength=slots)
        mean_lin[:, k] = np.where(tot > 0, acc / np.maximum(tot, 1e-12), 0.0)
    out[keep] = linear_to_srgb_u8(mean_lin[idx])
    used = len(np.unique(mat[keep]))
    print(f"[flat] {used} materials -> {used} flat colours "
          f"({keep.sum():,}/{m:,} faces)")
    return out


def match_and_clean(verts: np.ndarray, tris: np.ndarray, ref: dict, lut: np.ndarray,
                    radius: float, normal_cos: float, smooth: float,
                    flat: bool = False) -> tuple[np.ndarray, dict]:
    """匹配 + 邻域众数去跳变 +（可选）按材质压平。`smooth <= 0` 跳过众数那步。"""
    rgb, stats, best = match(verts, tris, ref, lut, radius, normal_cos)
    if smooth > 0:
        centroid = verts[tris].mean(axis=1)
        rgb = smooth_face_colors(centroid, best, ref["mat"], rgb, smooth)
    if flat:
        rgb = flat_by_material(verts, tris, best, ref["mat"], rgb)
    return rgb, stats


# ── 调色板：把几万种颜色压成 N 种 ───────────────────────────────────────────
#
# 直接从 256² 贴图上按 UV 采样，会得到几万种颜色、相邻三角形之间差几个色阶 ——
# 肉眼看到的就是"一片一片细碎的三角形"。压到 N 个颜色之后，同一块材料的绝大多数
# 面会落到同一个色号上，三角剖分就看不见了；剩下的边界正好落在真正的材质分界上。
#
# 在 **OkLab** 里做 k 均值：欧氏距离在 Lab 里大致等于人眼感知差异，直接对 sRGB
# 做 k 均值会把暗部的差异放大（同样是 20/255 的差，暗部看着差很多）。
def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)


def quantize(rgb: np.ndarray, n: int, seed: int = 0, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """把 (M,3) 的 sRGB 压成 n 种颜色。

    k 均值 ++ 初始化 + Lloyd 迭代，采样 20 万个面跑，再整体归类 —— 全量 300 万面
    跑 30 轮也只要几秒，没必要上 mini-batch。返回 (新颜色, 每个面的色号)。
    """
    if n < 2 or len(rgb) <= n:
        return rgb.copy(), np.zeros(len(rgb), dtype=np.uint16)
    rng = np.random.default_rng(seed)
    lab = srgb_to_oklab(rgb)
    step = max(1, len(lab) // 200_000)
    samp = lab[::step]

    # k 均值++：按离已选中心的距离平方抽样，避免一开始就把两个中心选在同一块材料上
    centers = np.empty((n, 3), dtype=np.float32)
    centers[0] = samp[rng.integers(len(samp))]
    d2 = ((samp - centers[0]) ** 2).sum(1)
    for i in range(1, n):
        total = d2.sum()
        pick = rng.random() * total if total > 0 else 0.0
        idx = int(np.searchsorted(np.cumsum(d2), pick))
        idx = min(idx, len(samp) - 1)
        centers[i] = samp[idx]
        d2 = np.minimum(d2, ((samp - centers[i]) ** 2).sum(1))

    for _ in range(iters):
        assign = np.argmin(((samp[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)
        moved = 0.0
        for i in range(n):
            sel = samp[assign == i]
            if len(sel):
                new = sel.mean(0)
                moved = max(moved, float(np.abs(new - centers[i]).max()))
                centers[i] = new
        if moved < 1e-6:
            break

    idx = np.argmin(((lab[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)
    # 色号的颜色取"这一簇里所有面的平均值"（在 sRGB 里平均），比 Lab 质心反变换更稳
    out = np.empty((len(rgb), 3), dtype=np.uint8)
    palette = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        sel = idx == i
        if sel.any():
            mean = rgb[sel].astype(np.float32).mean(0)
            palette[i] = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
            out[sel] = palette[i]
        else:
            palette[i] = np.clip(np.rint(centers[i] * 255), 0, 255).astype(np.uint8)
    return out, idx.astype(np.uint16)


# ── 法线 + 重合面统一 ───────────────────────────────────────────────────────
def face_normals(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每个面的单位法线 + 面积（面积给下面的加权用）。"""
    tri = verts[tris]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    ln = np.linalg.norm(cross, axis=1)
    ln[ln == 0] = 1.0
    return cross / ln[:, None], area


def smooth_face_normals(verts: np.ndarray, tris: np.ndarray, cos_limit: float = 0.5) -> np.ndarray:
    """把"每面自己算出来的法线"换成"和邻居平均过的法线"+ 折角保护。

    为什么需要：OBJ 没有 `vn`，OBJLoader 又把它展开成非索引几何，`computeVertexNormals()`
    只能给出**逐面**法线 —— 一块平地上几千个三角形的法线差个 1~2°，光照下就是一片
    明暗不同的小三角。颜色压到 8 种之后，剩下的"碎三角形"主要就是它，而不是颜色。

    做法：先按顶点把相邻面的法线（按面积加权）累加成"顶点光滑法线"，再让每个面取它三个
    顶点光滑法线的平均；**但只平均和这个面夹角小于 `cos_limit` 的那些**（默认 60°），
    这样箱子角、台阶边不会糊成圆的。

    返回 int8 量化后的 (M,3)，和 sRGB 一样按 /127 还原。
    """
    n, area = face_normals(verts, tris)
    v = len(verts)
    acc = np.empty((v, 3), dtype=np.float64)
    weighted = (n * area[:, None])
    for k in range(3):
        acc[:, k] = np.bincount(tris.ravel(), weights=np.repeat(weighted[:, k], 3), minlength=v)
    ln = np.linalg.norm(acc, axis=1)
    ln[ln == 0] = 1.0
    vn = acc / ln[:, None]

    corner = vn[tris]                                  # (M,3,3)
    agree = (corner * n[:, None, :]).sum(-1) > cos_limit
    cnt = agree.sum(1)
    summed = (corner * agree[..., None]).sum(1)
    bad = cnt == 0
    if bad.any():
        summed[bad] = n[bad]
        cnt[bad] = 1
    out = summed / cnt[:, None]
    ln = np.linalg.norm(out, axis=1)
    ln[ln == 0] = 1.0
    out /= ln[:, None]
    return np.clip(np.rint(out * 127), -127, 127).astype(np.int8)


def unify_coincident_faces(verts: np.ndarray, tris: np.ndarray,
                           rgb: np.ndarray, nrm: np.ndarray,
                           quantum: float = 1e-3) -> tuple[np.ndarray, np.ndarray, dict]:
    """让"同一块表面在 OBJ 里存在两份"的面共用同一个颜色和法线。

    为什么需要：OBJ 是碰撞网格（brush soup），同一块表面经常由两个 brush 各出一份，
    顶点位置**完全重合**。渲染时两份的深度也完全一样，逐像素谁赢取决于光栅化/浮点的
    末位误差 —— 画面就是一片横向细条纹（z-fighting），镜头一动就闪。

    本来两份涂同一个颜色时根本看不出来；但"逐面平滑法线"是按各自的面邻域算的，两份会
    得到不同的法线，光照下差好几级 —— 于是这份 z-fighting 就露出来了。色板压到几种
    颜色也救不了，因为差异在**光照**，不在颜色（实测 de_inferno：44,447 个多余面，
    其中 888 组两份的法线差 1°~30°）。

    做法：按顶点位置（`quantum` 米，默认 1 mm）把面分组；只统一"同向"（和组内第一个
    面夹角 < 30°）的那批 —— 背靠背的两份、T 型接缝各画各的，把它们平均到一起会把面
    翻到墙里去。每簇取**面积最大的那个面**的颜色、按面积加权的平均法线，写回簇内所有
    面。颜色取自簇内已有的面，所以不会把色板撑大。

    返回 (rgb, nrm, stats)。
    """
    if nrm is None:
        return rgb, nrm, {"skipped": "no normals"}

    m = len(tris)
    # 统一用 float32 量化：`obj.npz` 缓存里可能是 float32、也可能是老版本的 float64，
    # 直接按 float64 量化会在 0.5 mm 的边界上偶尔差一格，把同一组重复面拆开。
    v32 = np.asarray(verts, dtype=np.float32).astype(np.float64)
    q = np.rint(v32 / quantum).astype(np.int64)
    tri_q = q[tris]                                     # (M, 3, 3) —— 三个顶点

    # 三个顶点按字典序排序，做成与顶点顺序无关的键
    a, b, c = tri_q[:, 0], tri_q[:, 1], tri_q[:, 2]

    def gt(u, v):
        return (u[:, 0] > v[:, 0]) | (
            (u[:, 0] == v[:, 0])
            & ((u[:, 1] > v[:, 1]) | ((u[:, 1] == v[:, 1]) & (u[:, 2] > v[:, 2]))))

    for _ in range(2):
        sw = gt(a, b)
        a, b = np.where(sw[:, None], b, a), np.where(sw[:, None], a, b)
        sw = gt(b, c)
        b, c = np.where(sw[:, None], c, b), np.where(sw[:, None], b, c)
    sw = gt(a, b)
    a, b = np.where(sw[:, None], b, a), np.where(sw[:, None], a, b)

    # 两路独立的滚动哈希：单路 64 位哈希在 330 万个面上会有极少数碰撞，碰撞会把两块
    # 不同的几何并成一组，然后按"重复面"去统一颜色/法线 —— 那是错的。
    key = np.zeros(m, dtype=np.uint64)
    key2 = np.zeros(m, dtype=np.uint64)
    for arr in (a, b, c):
        for k in range(3):
            col = arr[:, k].astype(np.uint64)
            key = key * np.uint64(1000003) + col + np.uint64(7)
            key2 = key2 * np.uint64(1000033) + col + np.uint64(11)
    pair = np.stack([key, key2], axis=1)

    uniq, first, inv, counts = np.unique(
        pair, axis=0, return_index=True, return_inverse=True, return_counts=True)

    n = nrm.astype(np.float64) / 127.0
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    rep = n[first][inv]                                 # 每组代表法线
    same = ((n * rep).sum(1) >= np.cos(np.radians(30.0))).astype(np.int64)
    cluster = inv.astype(np.int64) * 2 + same
    ncl = len(uniq) * 2

    tri = np.asarray(verts, dtype=np.float64)[tris]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    # 面积加权平均法线（分簇）
    acc = np.empty((ncl, 3), dtype=np.float64)
    for k in range(3):
        acc[:, k] = np.bincount(cluster, weights=n[:, k] * area, minlength=ncl)
    ln = np.linalg.norm(acc, axis=1)
    ln[ln == 0] = 1.0
    mean_n = acc / ln[:, None]

    # 每簇取面积最大的那个面的颜色（不平均，免得把贴花/污渍的颜色抹到整块地面上）
    order = np.argsort(-area, kind="stable")
    cu, cfirst, cinv_sorted = np.unique(cluster[order], return_index=True, return_inverse=True)
    rep_rgb = rgb[order[cfirst]]
    rep_n = mean_n[cu]
    # `cinv_sorted` 是「按面积排序之后」的下标，映射回原来的面顺序
    cinv = np.empty(m, dtype=np.int64)
    cinv[order] = cinv_sorted

    changed = 0
    multi = counts > 1
    if multi.any():
        sel = multi[inv] & (same == 1)                  # 重复组里、同向的那些面
        changed = int(sel.sum())
        rgb = rgb.copy()
        rgb[sel] = rep_rgb[cinv][sel]
        nrm = nrm.copy()
        nrm[sel] = np.clip(np.rint(rep_n[cinv][sel] * 127), -127, 127).astype(np.int8)

    stats = {
        "coincident_groups": int(multi.sum()),
        "faces_unified": changed,
        "max_extra_faces": int((counts[multi] - 1).sum()) if multi.any() else 0,
    }
    print(f"[coincident] {stats['coincident_groups']:,} duplicate face groups, "
          f"{changed:,} faces made identical "
          f"({stats['max_extra_faces']:,} of them are redundant copies)")
    return rgb, nrm, stats


def unify_overlapping_faces(verts: np.ndarray, tris: np.ndarray,
                            rgb: np.ndarray, nrm: np.ndarray,
                            max_offset: float = 2e-3, max_angle: float = 20.0,
                            cell: float = 0.05) -> tuple[np.ndarray, np.ndarray, dict]:
    """把"同一块表面被两份几何盖住"的面统一成同一个颜色 + 法线。

    和 `unify_coincident_faces` 的区别：那个只认**顶点完全重合**，这个认**覆盖面**。
    brush soup 里更常见的是差 1~2 mm 的两层笔刷，或者同一个四边形被另一种对角线切开
    —— 顶点对不上，但两个三角形盖住了同一片区域，深度上只差零点几毫米，近处就会
    z-fighting。它们的颜色可能一样（调色板只有 8 色），但**逐面平滑法线**是按各自的
    邻域算的，两层能差 100° 以上，光照差好几级 —— 这就是"若隐若现"的那层。

    判据（对每一对面 i/j，几何法线夹角 < `max_angle`、重心到对方平面 < `max_offset`）：
      * i 的重心落在 j 的三角形内（重心距离 < 2 mm 也算，兜住两种对角线切法）；
      * 或者两者重心几乎重合。
    选中之后让 i 跟随"盖住它的面里面积最大的那个"，形成一棵只指向更大面积的森林
    （面积相同按编号破平局），所以没有环，也不会把一整块地板连成一片。每簇取面积最大的
    面的颜色、簇内（朝向自己几何法线的）平滑法线按面积加权平均。

    返回 (rgb, nrm, stats)。
    """
    if nrm is None:
        return rgb, nrm, {"skipped": "no normals"}

    m = len(tris)
    t0 = time.time()
    v = np.asarray(verts, dtype=np.float64)
    tri = v[tris]
    cen = tri.mean(axis=1)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cr = np.cross(e1, e2)
    ln = np.linalg.norm(cr, axis=1)
    area = 0.5 * ln
    ln[ln == 0] = 1.0
    g = cr / ln[:, None]

    ns = nrm.astype(np.float64) / 127.0
    ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-9)

    # 重心做空间哈希，只比同一个格子及其邻居（格子 5 cm，够覆盖 2 cm 内的两层）
    q = np.floor(cen / cell).astype(np.int64) + 100_000
    K = np.int64(1_000_000)
    key = (q[:, 0] * K + q[:, 1]) * K + q[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]

    cos_ang = np.cos(np.radians(max_angle))
    keep_i, keep_j = [], []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for dx, dy, dz in offsets:
        tgt = ks + np.int64((dx * K + dy) * K + dz)
        lo = np.searchsorted(ks, tgt, "left")
        hi = np.searchsorted(ks, tgt, "right")
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            continue
        ii = np.repeat(lo, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        jj = np.repeat(np.arange(len(ks)), cnt)
        fi = order[ii]
        fj = order[jj]
        # 同一个面、法线/平面不像的先扔掉（99% 的候选对在这一步就被筛掉）
        d2 = ((cen[fi] - cen[fj]) ** 2).sum(1)
        close = d2 < (cell * 1.5) ** 2
        par = (g[fi] * g[fj]).sum(1) > cos_ang
        off = np.abs(((cen[fj] - cen[fi]) * g[fi]).sum(1))
        ok = close & par & (off < max_offset) & (fi != fj)
        fi, fj = fi[ok], fj[ok]
        if not fi.size:
            continue
        # i 的重心是否落在 j 的三角形内（j 的平面基底上的重心坐标）
        d = cen[fi] - tri[fj, 0]
        d11 = (e1[fj] * e1[fj]).sum(1)
        d12 = (e1[fj] * e2[fj]).sum(1)
        d22 = (e2[fj] * e2[fj]).sum(1)
        det = d11 * d22 - d12 * d12
        good = det > 1e-12
        u = np.zeros(len(fi))
        w = np.zeros(len(fi))
        eg1, eg2, dg = e1[fj], e2[fj], d
        u[good] = (d22[good] * (dg[good] * eg1[good]).sum(1)
                   - d12[good] * (dg[good] * eg2[good]).sum(1)) / det[good]
        w[good] = (d11[good] * (dg[good] * eg2[good]).sum(1)
                   - d12[good] * (dg[good] * eg1[good]).sum(1)) / det[good]
        inside = good & (u >= -1e-3) & (w >= -1e-3) & (u + w <= 1.001)
        # 重心几乎重合也算（两种对角线切出来的三角形互相都盖不住对方重心的情形）
        near = d2[ok] < 4e-6
        sel = inside | near
        keep_i.append(fi[sel])
        keep_j.append(fj[sel])

    if not keep_i:
        print("[overlap] 0 overlapping face pairs")
        return rgb, nrm, {"overlap_pairs": 0, "faces_unified": 0, "clusters": 0}
    fi = np.concatenate(keep_i)
    fj = np.concatenate(keep_j)
    t_pairs = time.time()
    # 求连通分量（并查集）：只跟"面积最大的那个"是不够的 —— A 被 B、C 同时盖住时
    # A 只跟 B 走，C 却谁也不跟，A 和 C 就会分属两簇、颜色仍然不一致。用标签传播
    # 把所有互相盖住的面连成一簇，簇内才可能真正"同一个样子"。
    rep = np.arange(m, dtype=np.int64)
    rounds = 0
    for rounds in range(1, 65):
        prev = rep.copy()                 # 必须 copy：`np.minimum.at` 是原地改的，
        np.minimum.at(rep, fi, rep[fj])   # `prev = rep` 会跟着一起变，比较就永远相等，
        np.minimum.at(rep, fj, rep[fi])   # 循环提前退出、分量只合并了一半。
        rep = rep[rep]
        if np.array_equal(rep, prev):
            break

    gid, inv = np.unique(rep, return_inverse=True)
    sizes = np.bincount(inv)
    ng = len(gid)

    # 每簇的颜色：面积最大的那个面（和重合面统一用同一套规则，不平均）
    order_a = np.argsort(-area, kind="stable")
    cu, cfirst = np.unique(inv[order_a], return_index=True)
    donor_rgb = np.empty((ng, 3), np.uint8)
    donor_rgb[cu] = rgb[order_a[cfirst]]

    # 每簇的法线：只收"朝向自己几何法线 60° 以内"的平滑法线（被折角带偏的不投票）
    valid = (ns * g[gid][inv]).sum(1) >= np.cos(np.radians(60.0))
    acc = np.empty((ng, 3), dtype=np.float64)
    wts = area * valid
    for k in range(3):
        acc[:, k] = np.bincount(inv, weights=ns[:, k] * wts, minlength=ng)
    ln2 = np.linalg.norm(acc, axis=1)
    fallback = ln2 < 1e-9
    ln2[fallback] = 1.0
    mean_n = acc / ln2[:, None]
    mean_n[fallback] = g[gid][fallback]

    changed = sizes[inv] > 1
    n_changed = int(changed.sum())
    if n_changed:
        rgb = rgb.copy()
        rgb[changed] = donor_rgb[inv[changed]]
        nrm = nrm.copy()
        nrm[changed] = np.clip(np.rint(mean_n[inv[changed]] * 127), -127, 127).astype(np.int8)

    stats = {
        "overlap_pairs": int(len(fi)),
        "clusters": int((sizes > 1).sum()),
        "faces_unified": n_changed,
        "rounds": rounds,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[overlap] {stats['overlap_pairs']:,} overlapping face pairs, "
          f"{stats['clusters']:,} clusters, {n_changed:,} faces made identical "
          f"({t_pairs - t0:.0f}s candidate scan, {rounds} label rounds, "
          f"{stats['seconds']}s total)")
    return rgb, nrm, stats


# ── Stage 5: sidecar ────────────────────────────────────────────────────────
def write_sidecar(path: Path, rgb: np.ndarray, normals: np.ndarray | None = None,
                  palette: int = 0) -> None:
    """写旁车文件。带法线时用 v2 布局（颜色之后追加 每面 3 字节 int8 法线）。

    `flags` 记调色板大小（0 = 没压色），测试和前端据此知道"应该是几种颜色"。
    """
    version = 2 if normals is not None else 1
    header = MAGIC + struct.pack("<HHI", version, palette if palette > 1 else 0, len(rgb))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(header)
        f.write(rgb.astype(np.uint8).tobytes())
        if normals is not None:
            f.write(normals.astype(np.int8).tobytes())
    tmp.replace(path)          # 原地重写也要原子，别让浏览器读到半个文件
    print(f"[out] {path} ({path.stat().st_size / 1e6:.2f} MB, {len(rgb):,} triangles, v{version}"
          + (f", {palette} colours)" if palette > 1 else ")"))


def read_sidecar(path: Path) -> tuple[np.ndarray, np.ndarray | None, int]:
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != MAGIC:
        raise SystemExit(f"{path}: not a MCLR sidecar")
    version, flags, count = struct.unpack_from("<HHI", raw, 4)
    body = 12 + count * 3
    if len(raw) not in (body, body + count * 3):
        raise SystemExit(f"{path}: {len(raw)} bytes does not match {count} triangles")
    rgb = np.frombuffer(raw, dtype=np.uint8, offset=12, count=count * 3).reshape(-1, 3).copy()
    nrm = None
    if version >= 2 and len(raw) == body + count * 3:
        nrm = np.frombuffer(raw, dtype=np.int8, offset=body, count=count * 3).reshape(-1, 3).copy()
    return rgb, nrm, flags


def preview(verts: np.ndarray, tris: np.ndarray, rgb: np.ndarray, out: Path) -> None:
    """Top-down colour preview, so the bake can be eyeballed without a browser.

    Splats face centroids instead of rasterising triangles: 3.3M PIL polygons
    would take minutes, while a vectorised forward pass is instant and still
    shows whether walls, roofs and floors got the right palette.
    """
    from PIL import Image

    size = 1400
    centroid = verts[tris].mean(axis=1)
    lo = centroid[:, [0, 2]].min(axis=0)
    hi = centroid[:, [0, 2]].max(axis=0)
    span = float((hi - lo).max()) or 1.0
    px = ((centroid[:, 0] - lo[0]) / span * (size - 1)).astype(np.int32)
    py = ((centroid[:, 2] - lo[1]) / span * (size - 1)).astype(np.int32)
    img = np.full((size, size, 3), 20, dtype=np.uint8)
    # paint high Y last so roofs sit on top of the floors they cover
    order = np.argsort(centroid[:, 1])
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx = np.clip(px[order] + dx, 0, size - 1)
        yy = np.clip(py[order] + dy, 0, size - 1)
        img[yy, xx] = rgb[order]
    Image.fromarray(img).save(out)
    print(f"[out] preview -> {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="map name, e.g. de_inferno")
    ap.add_argument("--game", type=Path, default=DEFAULT_GAME, help="CS2 install root")
    ap.add_argument("--maps", type=Path, default=DEFAULT_MAPS, help="maps/optimized_obj_files")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK, help="cache directory")
    ap.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    ap.add_argument("--out", type=Path, default=None, help="sidecar path (default <maps>/<map>.colors.bin)")
    ap.add_argument("--radius", type=float, default=0.75, help="max match distance in metres")
    ap.add_argument("--normal-cos", type=float, default=0.7, help="normal agreement threshold")
    ap.add_argument("--limit", type=int, default=None, help="only export the first N nodes (debug)")
    ap.add_argument("--preview", type=Path, default=None, help="write a top-down PNG preview")
    ap.add_argument("--palette", type=int, default=8,
                    help="quantise to this many colours (0 = keep the per-material colours)")
    ap.add_argument("--no-normals", action="store_true",
                    help="skip the smoothed per-face normals (v1 sidecar, colours only)")
    ap.add_argument("--no-unify", action="store_true",
                    help="keep coincident duplicate faces shaded independently (they z-fight)")
    ap.add_argument("--no-overlap", action="store_true",
                    help="skip the 'overlapping surface' pass (only exact-coincident faces unified)")
    ap.add_argument("--no-flat", action="store_true",
                    help="keep per-face texture sampling (default: one flat colour per material)")
    ap.add_argument("--smooth", type=float, default=0.35,
                    help="neighbourhood radius (m) for the material majority vote; 0 = off")
    ap.add_argument("--keep-decals", action="store_true",
                    help="let decal / overlay materials vote (default: skip them, they speckle)")
    ap.add_argument("--restyle", action="store_true",
                    help="only re-quantise + re-normal an existing sidecar (no export, no match)")
    ap.add_argument("--force", action="store_true", help="ignore cached stages")
    args = ap.parse_args(argv)

    # 全部转绝对路径：CLI 是带 cwd 跑的（见 `Cli.cwd`），相对路径会被解释两遍
    args.maps = args.maps.expanduser().resolve()
    args.work = args.work.expanduser().resolve()
    obj = args.maps / f"{args.map}.obj"
    if not obj.is_file():
        raise SystemExit(f"no such OBJ: {obj}")

    out = args.out.resolve() if args.out else (args.maps / f"{args.map}.colors.bin")
    work = args.work / args.map

    # 换调色板 / 补法线都不需要重新导出几何：旁车里已经是"每面一个颜色"，法线从 OBJ
    # 算（~10 秒/张）。所以可以拿 8 / 12 / 16 各出一版直接比。
    if args.restyle:
        if not out.is_file():
            raise SystemExit(f"no sidecar to restyle: {out}")
        rgb, _old, _pal = read_sidecar(out)
        before = len(np.unique(rgb, axis=0))
        verts, tris = load_obj(obj, work / "obj.npz", False)
        if len(tris) != len(rgb):
            raise SystemExit(f"{out}: {len(rgb)} colours vs {len(tris)} faces in {obj.name}")
        # 色板数量没变就别再压一次：k 均值在"已经压好的 8 个颜色"上重跑会轻微漂移，
        # 而 --restyle 的常见用途只是补法线 / 统一重合面。
        if args.palette > 1 and _pal != args.palette:
            rgb, idx = quantize(rgb, args.palette)
            counts = np.bincount(idx, minlength=args.palette)
            print(f"[palette] {before} -> {args.palette} colours "
                  f"({', '.join(f'{c / len(rgb):.0%}' for c in counts)})")
        nrm = None if args.no_normals else smooth_face_normals(verts, tris)
        if nrm is not None and not args.no_unify:
            rgb, nrm, _uni = unify_coincident_faces(verts, tris, rgb, nrm)
            if not args.no_overlap:
                rgb, nrm, _ov = unify_overlapping_faces(verts, tris, rgb, nrm)
        write_sidecar(out, rgb, nrm, args.palette)
        return 0

    map_vpk = args.game / "game" / "csgo" / "maps" / f"{args.map}.vpk"
    if not map_vpk.is_file():
        raise SystemExit(f"no such map VPK: {map_vpk}")

    work.mkdir(parents=True, exist_ok=True)
    cli = Cli(args.cli.resolve(), args.game)

    meta = stage_reference(cli, map_vpk, args.map, work, args.limit, args.force,
                           keep_decals=args.keep_decals)
    lut = stage_textures(work, meta, args.force)
    ref = np.load(work / "reference.npz")
    verts, tris = load_obj(obj, work / "obj.npz", args.force)
    if len(tris) != meta["triangles"] and args.limit:
        print("[warn] reference is a --limit subset; colours will be partial")
    rgb, stats = match_and_clean(verts, tris, ref, lut, args.radius, args.normal_cos,
                                 args.smooth, not args.no_flat)
    print("[match] " + json.dumps(stats))
    (work / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")

    if args.palette > 1:
        before = len(np.unique(rgb, axis=0))
        rgb, idx = quantize(rgb, args.palette)
        counts = np.bincount(idx, minlength=args.palette)
        print(f"[palette] {before} -> {args.palette} colours "
              f"({', '.join(f'{c / len(rgb):.0%}' for c in counts)})")

    nrm = None if args.no_normals else smooth_face_normals(verts, tris)
    if nrm is not None and not args.no_unify:
        rgb, nrm, _uni = unify_coincident_faces(verts, tris, rgb, nrm)
        if not args.no_overlap:
            rgb, nrm, _ov = unify_overlapping_faces(verts, tris, rgb, nrm)
    write_sidecar(out, rgb, nrm, args.palette)
    if args.preview:
        preview(verts, tris, rgb, args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
