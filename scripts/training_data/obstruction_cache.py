"""整回合两两遮挡的磁盘缓存：``<csz>.obs/r<回合>.npz``。

为什么单独一层
--------------
遮挡是造数 / 推理链路里最贵的一步（精确算法实测 0.57 ms/帧、一个 5666 帧的回合
2~6 秒，占 app 侧单回合推理耗时的 90%），而它只依赖 ``.csz`` 里的几何 + 地图
mesh，**和模型、和采样窗口都无关**。所以它在 csz 落盘那一步算一次、按回合存下来，
下游（训练造数 / AI 推理）只读数组：

    解析 csz ──► build_round() ──► <csz>.obs/r<id>.npz
                                     │
              ┌──────────────────────┴──────────────────────┐
        aim / spatial 造数                        app AI 推理
        （切片就是张量）                          （不碰几何、不建 BVH）

缓存键
------
:func:`basis` 把**所有会改变数值的东西**揉进一个短哈希：缓存格式版本、锚点间隔、
包围盒 / 眼高 / 体素常量、以及地图 OBJ 的 ``size + mtime``。任何一项变了，老的
``.npz`` 就不会被认（:func:`load` 返回 ``None``，调用方重算并覆盖）。回合帧数也
单独比一次 —— 同一个 csz 重新抽一遍（采样步长变了）会让帧数对不上。

缓存文件坏了 / 被删了都只是"没命中"，不是错误。

**这个模块本身很轻**：几何 / 射线（numba、地图 BVH）和 ``build_round`` 都是用到
才 import 的，所以 ``import obstruction_cache`` 只是 numpy + 几个常量 —— 删缓存、
查有没有缓存这些"只想摸文件"的调用方不会因此拖进 numba。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import _paths  # noqa: F401  （把 tools/ 加进 sys.path，demo_extract 才可导入）
from .constants import N_PLAYERS

#: 缓存格式 / 算法版本。**改了 ``build_round`` 的规则或几何常量就 +1**，老缓存会
#: 在下一次 :func:`load` 时被认成不命中。
#:
#: v2：修掉 `view_render.smoke_volumes()` 的缓存绕过存活期检查（烟散了还按有烟算）
#: 和 entity_id 复用时互相覆盖 —— 两者都会直接污染已经落盘的 ``.obs``，所以老缓存
#: 必须整批作废重算。
CACHE_VERSION = 2

#: 缓存目录后缀：``match__ab12cd34.csz`` → ``match__ab12cd34.obs/``。
CACHE_SUFFIX = ".obs"


def _default_stride() -> int:
    """锚点间隔的默认值住在 :data:`obstruction.PAIR_STRIDE`（算法侧）。"""
    from .obstruction import PAIR_STRIDE

    return int(PAIR_STRIDE)


def _stride(value: Optional[int]) -> int:
    return _default_stride() if value is None else max(1, int(value))


# ═══════════════════════════════════════════════════════════════════════════
# 路径 / 键
# ═══════════════════════════════════════════════════════════════════════════


def cache_dir(csz_path: str | Path) -> Path:
    """缓存目录：和 ``.csz`` 并排，名字换个后缀。

    并排而不是塞进一个总的缓存目录，是为了"删 csz 就顺手删缓存"能一眼写对，
    而且两个消费方（app 的 ``data/csz``、训练的 ``data/demos/csz``）不需要各自
    配置一个路径。
    """
    return Path(csz_path).with_suffix(CACHE_SUFFIX)


def round_path(csz_path: str | Path, round_id: int) -> Path:
    return cache_dir(csz_path) / f"r{int(round_id):02d}.npz"


def basis(map_name: str, stride: Optional[int] = None) -> str:
    """一份缓存的"数值身份"：常量 + 地图几何 + 算法版本，短哈希。"""
    from demo_extract import recipes

    from .obstruction import _TOL
    from .view_render import (
        HULL_CROUCH_HU,
        HULL_HALF_HU,
        HULL_STAND_HU,
        VOXEL_CELL,
        VOXEL_DIM,
    )

    h = hashlib.sha1()
    parts = [
        f"v{CACHE_VERSION}",
        f"stride={_stride(stride)}",
        f"tol={_TOL}",
        f"hull={HULL_HALF_HU},{HULL_STAND_HU},{HULL_CROUCH_HU}",
        f"vox={VOXEL_CELL},{VOXEL_DIM}",
        f"eye={recipes.STANDING_EYE_HEIGHT_HU},{recipes.CROUCH_EYE_DROP_HU}",
    ]
    try:
        p = _map_obj(map_name)
        st = p.stat()
        parts.append(f"map={p.name}:{st.st_size}:{st.st_mtime_ns}")
    except OSError:
        # 地图缺失时上层本来就会报错（build_round 里的 load_map 会抛）；这里只求
        # 别让哈希本身炸掉。
        parts.append(f"map={map_name}:missing")
    h.update("|".join(parts).encode("utf-8"))
    return h.hexdigest()[:16]


def _map_obj(map_name: str) -> Path:
    from .vis_oracle import map_obj_for

    return map_obj_for(map_name)


# ═══════════════════════════════════════════════════════════════════════════
# 读写
# ═══════════════════════════════════════════════════════════════════════════


def _meta(csz_path: str | Path | None, round_id: int) -> Optional[dict]:
    """只读 npz 的头部三样（不会把整张数组读进来）。坏文件 = ``None``。"""
    if not csz_path:
        return None
    path = round_path(csz_path, round_id)
    try:
        with np.load(path) as z:
            return {
                "basis": str(z["basis"].item()),
                "stride": int(z["stride"]),
                "frames": int(z["frames"]),
            }
    except (OSError, KeyError, ValueError):
        return None


def has(
    csz_path: str | Path | None,
    round_id: int,
    frames: int,
    *,
    map_name: str,
    stride: Optional[int] = None,
) -> bool:
    """有可用缓存吗（**不读数组本体**，给"命中/新建"统计和预检用）。"""
    meta = _meta(csz_path, round_id)
    if meta is None:
        return False
    stride = _stride(stride)
    return (
        meta["basis"] == basis(map_name, stride)
        and meta["stride"] == stride
        and meta["frames"] == int(frames)
    )


def load(
    csz_path: str | Path | None,
    round_id: int,
    frames: int,
    *,
    map_name: str,
    stride: Optional[int] = None,
    n_pl: int = N_PLAYERS,
) -> Optional[np.ndarray]:
    """命中就返回 ``(frames, n_pl, n_pl) int8``，否则 ``None``。

    任何异常都当"没命中"：缓存坏掉不该让造数 / 推理失败，重算一遍就好。
    """
    if not has(csz_path, round_id, frames, map_name=map_name, stride=stride):
        return None
    try:
        with np.load(round_path(csz_path, round_id)) as z:
            arr = z["obstruction"]
    except (OSError, KeyError, ValueError):
        return None
    if arr.shape != (int(frames), n_pl, n_pl):
        return None
    if arr.dtype != np.int8:
        arr = arr.astype(np.int8)
    return np.ascontiguousarray(arr)


def save(
    csz_path: str | Path | None,
    round_id: int,
    arr: np.ndarray,
    *,
    map_name: str,
    stride: Optional[int] = None,
) -> Optional[Path]:
    """原子写。写不进去（权限 / 只读盘）就当没发生过，返回 ``None``。"""
    if not csz_path:
        return None
    stride = _stride(stride)
    path = round_path(csz_path, round_id)
    tmp = path.with_suffix(".npz.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 传文件对象而不是路径：`np.savez_compressed("x.npz.tmp")` 会自己再补一个
        # `.npz`，写出来的名字和 `os.replace` 找的对不上。
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                obstruction=np.ascontiguousarray(arr, dtype=np.int8),
                basis=np.array(basis(map_name, stride)),
                stride=np.int64(stride),
                frames=np.int64(arr.shape[0]),
                map_name=np.array(str(map_name)),
            )
        os.replace(tmp, path)
        return path
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════


def ensure(
    csz_path: str | Path | None,
    rnd: dict,
    map_name: str,
    n_pl: int = N_PLAYERS,
    *,
    stride: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> np.ndarray:
    """这个回合的遮挡：命中缓存就读，否则整回合重算并写回。

    参数 ``rnd`` 是 ``recipes.round_by_id()`` / ``data["rounds"][i]`` 的那个 dict；
    回合 id 从 ``rnd["id"]`` 取（缺失时退回 0 —— 只有测试里的手搓 dict 会这样）。

    ``csz_path`` 为空（内存里的 data、没落盘的临时 demo）时**不读写磁盘**，
    照样按整回合算一遍返回。
    """
    from .obstruction import build_round          # numba，进到这里才 import

    say = progress or (lambda _frac, _msg: None)
    stride = _stride(stride)
    frames = len(rnd.get("ticks") or ())
    round_id = int(rnd.get("id", 0))

    if csz_path:
        say(0.0, "读取遮挡缓存…")
        hit = load(csz_path, round_id, frames, map_name=map_name, stride=stride, n_pl=n_pl)
        if hit is not None:
            say(1.0, "命中遮挡缓存")
            return hit

    arr = build_round(rnd, map_name, n_pl, stride=stride, progress=progress)
    if csz_path:
        save(csz_path, round_id, arr, map_name=map_name, stride=stride)
    return arr
