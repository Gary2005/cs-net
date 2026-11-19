#!/usr/bin/env python3
"""test_map_colors.py — 地图材质颜色旁车文件（`<map>.colors.bin`）的自洽性检查。

不需要浏览器、不需要服务器：直接对拍 `maps/optimized_obj_files/` 里的
`<map>.obj` 和旁边烘好的 `<map>.colors.bin`（由
`离线资产管线` 生成）。

为什么不只看文件头：这个文件是**按 OBJ 的 `f` 行顺序**逐面存的，所以"面数对得上"
就是它能正确贴回去的唯一前提。面数错了不会崩，只会整张图错位一个面的颜色 ——
那种错误肉眼要放大才看得出来，这里直接数 `f` 行。

检查项：
  1. 头：magic "MCLR" / 版本 1~2 / 长度 = 12 + 3×面数（v2 再加 3×面数）；
  2. 面数 = OBJ 里 `f` 行的数量（OBJ 全是三角形，`f` 行数就是三角形数）；
  3. 颜色不是"一片灰"：没压色板时要求不同颜色数 ≥ 1000；压过色板（header.flags > 1）就
     要求 ≤ 色板大小 —— 压到 12 色之后"颜色多样性"这个指标本身就反过来了；
  4. 没有大面积的中性灰兜底色（烘焙时没匹配上的面会被涂成 (168,168,168)），占比 < 5%；
  5. v2：**顶点完全重合的重复面必须同色同法线**（否则两份 z-fighting，镜头一动就闪）；
  6. v2：**互相盖住（2 mm 内、共面）的面也必须同色同法线** —— 换三角化的同一块表面、
     差零点几毫米的两层笔刷，深度同样会打架。

用法：

    python visualizer/tests/test_map_colors.py [maps_dir]

`maps_dir` 默认 `maps/optimized_obj_files`。没有任何 `.colors.bin` 时打印 SKIP 并退出 0
（干净的 checkout 里没有这些文件，它们和 OBJ 一样是本地生成的）。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAGIC = b"MCLR"
#: v1 = 颜色；v2 = 颜色 + 每个面 3 字节 int8 平滑法线
MAX_VERSION = 2
HEADER = 12
#: bake 脚本给"没匹配到任何原始三角形"的面涂的兜底色
FALLBACK = (168, 168, 168)


def count_faces(obj_path: Path) -> int:
    """数 OBJ 里**三角化之后**的面数。分块读，130 MB 的图也就一两秒。

    多边形按扇面展开成 n-2 个三角形，和 OBJLoader / `bake_map_colors.load_obj`
    的做法一致 —— 只数 `f ` 行会把四边形算成一个面，然后报"颜色数和面数差 515"，
    看起来像烘焙错了。（de_mirage 的 OBJ 里就有 515 个四边形。）

    块边界要留 **2 字节**重叠：模式是 `\\nf `（3 字节），只留 1 字节的话"块尾正好
    停在 `\\nf`"时那个换行就被两边都丢掉了。
    """
    n = 0
    rest = b""
    with obj_path.open("rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            lines = (rest + chunk).split(b"\n")
            rest = lines.pop()          # 最后一段可能是半行，留到下一块
            for line in lines:
                if line[:2] == b"f ":
                    k = len(line.split()) - 1
                    if k >= 3:
                        n += k - 2
    if rest[:2] == b"f ":
        k = len(rest.split()) - 1
        if k >= 3:
            n += k - 2
    return n


def load_geometry(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读 OBJ 的顶点位置和（扇面三角化后的）面索引。

    只解析 ``v`` / ``f`` 两种行 —— 这个 OBJ 没有 ``vt`` / ``vn`` / ``usemtl``，
    所以这就是全部内容。130 MB 的图大约 3~5 秒。
    """
    verts = bytearray()
    faces: list[tuple[int, int, int]] = []
    with obj_path.open("rb") as f:
        for line in f:
            if line[:2] == b"v ":
                verts += line[2:]
            elif line[:2] == b"f ":
                idx = [int(t.split(b"/")[0]) - 1 for t in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    # float32 和烘焙端保持一致（`bake_map_colors.load_obj` 也是 float32）：位置量化到
    # 1 mm 时，float64 和 float32 在 0.5 mm 边界上偶尔会差一格，导致同一组重复面被
    # 拆成两组 —— 那是"测试和烘焙各用一套数"，不是真的没统一。
    v = np.fromstring(bytes(verts), dtype=np.float32, sep=" ").reshape(-1, 3)
    return v, np.asarray(faces, dtype=np.int64)


def check_coincident(obj_path: Path, rgb: np.ndarray, nrm: np.ndarray) -> list[str]:
    """顶点完全重合的重复面，必须画成同一个样子（颜色 + 法线）。

    为什么：碰撞网格是 brush soup，同一块表面经常由两个 brush 各出一份。两份的深度
    完全一样，逐像素谁赢取决于浮点末位误差 —— 画面上是一闪一闪的横向条纹。把两份
    涂成同一个颜色 + 同一个法线之后，谁赢都一样，这份 z-fighting 就看不见了
    （见 `bake_map_colors.unify_coincident_faces`）。

    只比"同向"（夹角 < 30°）的那批：背靠背的两份本来就只画得出一个面。
    """
    v, f = load_geometry(obj_path)
    if len(f) != len(rgb):
        return []                                  # 面数对不上，上面已经报过了

    # 先转 float64 再除：float32 直接除以 1e-3 会先按 float32 舍入，正好落在 0.5 mm
    # 边界上的坐标可能和烘焙端（float64 除法）差一格 —— 两边分组就会错开。
    q = np.rint(np.asarray(v, dtype=np.float32).astype(np.float64) / 1e-3).astype(np.int64)
    tri = q[f]                                     # (M, 3, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

    def gt(u, w):
        return (u[:, 0] > w[:, 0]) | (
            (u[:, 0] == w[:, 0])
            & ((u[:, 1] > w[:, 1]) | ((u[:, 1] == w[:, 1]) & (u[:, 2] > w[:, 2]))))

    for _ in range(2):
        sw = gt(a, b)
        a, b = np.where(sw[:, None], b, a), np.where(sw[:, None], a, b)
        sw = gt(b, c)
        b, c = np.where(sw[:, None], c, b), np.where(sw[:, None], b, c)
    sw = gt(a, b)
    a, b = np.where(sw[:, None], b, a), np.where(sw[:, None], a, b)

    # 两路独立的滚动哈希：单路 64 位哈希在 330 万个面上会有极少数碰撞，碰撞会把
    # 两块**不同**的几何并成一组，然后报"重复面颜色不一致"的假警报。
    key = np.zeros(len(f), dtype=np.uint64)
    key2 = np.zeros(len(f), dtype=np.uint64)
    for arr in (a, b, c):
        for k in range(3):
            col = arr[:, k].astype(np.uint64)
            key = key * np.uint64(1000003) + col + np.uint64(7)
            key2 = key2 * np.uint64(1000033) + col + np.uint64(11)
    pair = np.stack([key, key2], axis=1)

    _u, first, inv, counts = np.unique(
        pair, axis=0, return_index=True, return_inverse=True, return_counts=True)
    if not (counts > 1).any():
        return []

    n = nrm.astype(np.float64) / 127.0
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    rep = n[first][inv]
    same = (n * rep).sum(1) >= np.cos(np.radians(30.0))
    dup = (counts > 1)[inv] & same

    bad_colour = bad_normal = 0
    idx = np.flatnonzero(dup)
    if len(idx):
        # 同一组里，"同向"的那批面必须**互相**一模一样（不是和组代表比 —— 统一时用的
        # 是这批面的平均法线，可以合理地不等于任何一个成员）。取组内第一个同向面当基准。
        g = inv[idx]
        order = np.argsort(g, kind="stable")
        gs = g[order]
        heads = np.r_[True, gs[1:] != gs[:-1]]
        ref = idx[order][heads]                       # 每组第一个同向面
        ref_face = ref[np.searchsorted(gs[heads], g)]
        bad_colour = int(
            (np.abs(rgb[idx].astype(int) - rgb[ref_face].astype(int)).max(1) > 0).sum())
        # 和 overlap 检查同一个口径：>30° 才算"看得出来"。簇是按"和组代表差 30° 以内"
        # 分的，簇内已经统一成同一个平均法线；残留的几度差异来自两个预测器在 30° 边界
        # 上对同一个面分了不同的簇 —— 不是没统一，而是判据本身的边界。报出来但不判失败。
        ang = np.degrees(np.arccos(
            np.clip((n[idx] * n[ref_face]).sum(1), -1, 1)))
        bad_normal = int((ang > 30.0).sum())
        borderline = int((ang > 1.0).sum())

    problems = []
    if bad_colour:
        problems.append(f"{obj_path.name}: {bad_colour:,} coincident faces still differ in colour")
    if bad_normal:
        problems.append(f"{obj_path.name}: {bad_normal:,} coincident faces still differ in normal")
    print(f"  coincident faces: {int(dup.sum()):,} checked, "
          f"{bad_colour:,} colour / {borderline:,} borderline ({bad_normal:,} >15°) normal mismatches")
    return problems


def check_overlapping(obj_path: Path, rgb: np.ndarray, nrm: np.ndarray) -> list[str]:
    """互相盖住的共面三角形必须画成同一个样子（颜色 + 法线）。

    为什么：`check_coincident` 只管"顶点完全重合"。brush soup 里更常见的是差 1~2 mm
    的两层笔刷、或者同一个四边形被另一种对角线切开 —— 顶点对不上，但两个三角形盖住
    同一片区域，深度只差零点几毫米。颜色多半本来就一样（8 色调色板），可**逐面平滑
    法线**是按各自邻域算的，两层能差 100° 以上，光照差好几级，近处就是"若隐若现"。

    判据（和 `bake_map_colors.unify_overlapping_faces` 一致）：
      * 两个面的几何法线夹角 < 20°，重心到对方平面 < 2 mm；
      * 面 i 的重心落在面 j 的三角形内，或两者重心几乎重合。
    """
    v, f = load_geometry(obj_path)
    if len(f) != len(rgb):
        return []

    tri = np.asarray(v, dtype=np.float64)[f]
    cen = tri.mean(axis=1)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cr = np.cross(e1, e2)
    ln = np.linalg.norm(cr, axis=1)
    ln[ln == 0] = 1.0
    g = cr / ln[:, None]

    cell = 0.05
    q = np.floor(cen / cell).astype(np.int64) + 100_000
    K = np.int64(1_000_000)
    key = (q[:, 0] * K + q[:, 1]) * K + q[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cos_ang = np.cos(np.radians(20.0))

    pairs_i, pairs_j = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                tgt = ks + np.int64((dx * K + dy) * K + dz)
                lo = np.searchsorted(ks, tgt, "left")
                hi = np.searchsorted(ks, tgt, "right")
                cnt = hi - lo
                tot = int(cnt.sum())
                if tot == 0:
                    continue
                ii = np.repeat(lo, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
                jj = np.repeat(np.arange(len(ks)), cnt)
                fi, fj = order[ii], order[jj]
                d2 = ((cen[fi] - cen[fj]) ** 2).sum(1)
                ok = ((d2 < (cell * 1.5) ** 2) & (fi != fj)
                      & ((g[fi] * g[fj]).sum(1) > cos_ang)
                      & (np.abs(((cen[fj] - cen[fi]) * g[fi]).sum(1)) < 2e-3))
                fi, fj = fi[ok], fj[ok]
                if not len(fi):
                    continue
                d = cen[fi] - tri[fj, 0]
                d11 = (e1[fj] * e1[fj]).sum(1)
                d12 = (e1[fj] * e2[fj]).sum(1)
                d22 = (e2[fj] * e2[fj]).sum(1)
                det = d11 * d22 - d12 * d12
                good = det > 1e-12
                u = np.zeros(len(fi))
                w = np.zeros(len(fi))
                u[good] = (d22[good] * (d[good] * e1[fj][good]).sum(1)
                           - d12[good] * (d[good] * e2[fj][good]).sum(1)) / det[good]
                w[good] = (d11[good] * (d[good] * e2[fj][good]).sum(1)
                           - d12[good] * (d[good] * e1[fj][good]).sum(1)) / det[good]
                sel = good & (u >= -1e-3) & (w >= -1e-3) & (u + w <= 1.001)
                sel |= d2[ok] < 4e-6
                pairs_i.append(fi[sel])
                pairs_j.append(fj[sel])

    problems = []
    if not pairs_i:
        print("  overlapping faces: 0 pairs")
        return problems
    fi = np.concatenate(pairs_i)
    fj = np.concatenate(pairs_j)
    # 统一之后同一簇的面完全一样。留 1° 的软阈值只用来报"擦边"数量：检测器是
    # "重心落在对方三角形内"，在曲面上偶尔会把两个相邻面配成一对（各差几度），
    # 这些不是重复面，不该判失败。真正会看出来的是 >15° 的明暗差。
    n = nrm.astype(np.float64) / 127.0
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    bad_colour = int(
        (np.abs(rgb[fi].astype(int) - rgb[fj].astype(int)).max(1) > 1).sum())
    ang = np.degrees(np.arccos(np.clip((n[fi] * n[fj]).sum(1), -1, 1)))
    soft = int((ang > 1.0).sum())
    bad_normal = int((ang > 15.0).sum())
    if bad_colour:
        problems.append(f"{obj_path.name}: {bad_colour:,} overlapping faces differ in colour")
    if bad_normal:
        problems.append(f"{obj_path.name}: {bad_normal:,} overlapping faces differ in normal")
    print(f"  overlapping faces: {len(fi):,} pairs, "
          f"{bad_colour:,} colour / {soft:,} borderline ({bad_normal:,} >15°) normal mismatches")
    return problems


def check_one(obj_path: Path, bin_path: Path) -> list[str]:
    problems: list[str] = []
    raw = bin_path.read_bytes()
    if len(raw) < HEADER:
        return [f"{bin_path.name}: {len(raw)} bytes, shorter than the {HEADER}-byte header"]
    if raw[:4] != MAGIC:
        problems.append(f"{bin_path.name}: bad magic {raw[:4]!r}, expected {MAGIC!r}")
    version, flags, count = struct.unpack_from("<HHI", raw, 4)
    if version not in (1, 2):
        problems.append(f"{bin_path.name}: version {version}, expected 1 or 2")
    body = HEADER + count * 3
    # v2 在颜色后面追加"每面 3 字节 int8 法线"
    if len(raw) not in (body, body + count * 3):
        problems.append(
            f"{bin_path.name}: {len(raw)} bytes != {body} (v1) or {body + count * 3} (v2)"
        )
        return problems
    if version == 2 and len(raw) != body + count * 3:
        problems.append(f"{bin_path.name}: header says v2 but there are no normals")
    if version == 1 and len(raw) != body:
        problems.append(f"{bin_path.name}: header says v1 but the file is longer")

    faces = count_faces(obj_path)
    if faces != count:
        problems.append(f"{bin_path.name}: {count:,} colours vs {faces:,} faces in {obj_path.name}")

    rgb = np.frombuffer(raw, dtype=np.uint8, offset=HEADER, count=count * 3).reshape(-1, 3)
    distinct = len(np.unique(rgb, axis=0))
    # flags 记调色板大小：压过色的话"颜色种类数"就应该 ≤ 那个数，没压才要求足够多样
    if flags > 1:
        if distinct > flags:
            problems.append(f"{bin_path.name}: {distinct} colours but the palette is {flags}")
        if distinct < 2:
            problems.append(f"{bin_path.name}: a single colour after quantising to {flags}")
    elif distinct < 1000:
        problems.append(f"{bin_path.name}: only {distinct} distinct colours (a grey wash?)")

    c = rgb.astype(np.int16)
    spread = c.max(axis=1) - c.min(axis=1)
    # 只报数，不做判定：de_nuke 的 albedo **本身就是灰白的金属板**（实测只有 1.1% 的面
    # 有明显色相），拿它当"烘坏了"会误伤。真正的"烘坏了"由下面的"颜色种类数"和
    # "兜底色占比"兜住 —— 合成的那套水泥灰是恒定色相，最多几千个亮度档，而且
    # 全部落在同一个灰阶上。
    tinted = float((spread > 25).mean())          # 25/255 ≈ 0.1

    flat = int((rgb == np.array(FALLBACK, dtype=np.uint8)).all(axis=1).sum())
    share = flat / max(1, len(rgb))
    if share > 0.05:
        problems.append(f"{bin_path.name}: {share:.1%} of faces are the (168,168,168) fallback")

    print(
        f"  {bin_path.name:<26} {count:>10,} faces  {distinct:>6} colours  "
        f"{tinted:>5.1%} tinted  {share:>5.2%} fallback"
    )

    if version == 2 and len(raw) == body + count * 3:
        nrm = np.frombuffer(raw, dtype=np.int8, offset=body,
                            count=count * 3).reshape(-1, 3)
        problems += check_coincident(obj_path, rgb, nrm)
        problems += check_overlapping(obj_path, rgb, nrm)
    return problems


def main(argv: list[str]) -> int:
    maps_dir = Path(argv[1]) if len(argv) > 1 else ROOT / "maps" / "optimized_obj_files"
    if not maps_dir.is_dir():
        print(f"SKIP: no maps directory at {maps_dir}")
        return 0
    sidecars = sorted(maps_dir.glob("*.colors.bin"))
    if not sidecars:
        print(f"SKIP: no baked colours under {maps_dir} "
              f"(run 离线资产管线 --map <name>)")
        return 0

    problems: list[str] = []
    missing: list[str] = []
    for bin_path in sidecars:
        obj_path = bin_path.with_name(bin_path.name.replace(".colors.bin", ".obj"))
        if not obj_path.is_file():
            missing.append(f"{bin_path.name}: no matching {obj_path.name}")
            continue
        problems += check_one(obj_path, bin_path)

    for m in missing:
        print(f"  WARN {m}")

    print()
    if problems:
        for p in problems:
            print(f"FAIL {p}")
        return 1
    print(f"PASS: {len(sidecars)} map colour sidecar(s) consistent with their OBJ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
