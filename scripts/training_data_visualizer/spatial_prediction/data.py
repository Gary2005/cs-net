"""npz → 可视化载荷（纯数据层，不碰 HTTP）。

这个文件只做一件事：把 `tasks/spatial_prediction.py` 写出来的样本**逆变换**回人看得懂
的量，并压成一份 JSON 友好的 dict。前端拿到之后不再做任何"数据语义"上的换算（除了
画图）。

和 aim 的可视化器是什么关系
---------------------------
结构几乎一样（同样的深色 web app、同样的逆变换套路），但**目标不同**：

    aim   一条样本 = 一段 257 帧的窗口，重点是"逐帧轨迹 + 第一人称视角 + 模型曲线"
    这里  一条样本 = **一帧**，重点是"这一帧长什么样 + 5 个标签对不对"

所以这里没有第一人称、没有时间轴、没有模型面板；取代它们的是：

    · 一张俯视雷达（10 个人的位置 / 朝向 / 血量 / 存活）
    · 一组**标签卡片**（CT 胜 / 下一个死亡 / 下一个击杀 / 终局存活 / 还能击杀）
    · 一张遮挡矩阵和一张 10×20 的原始特征表（查"模型到底看到了什么"）

为什么要逆变换
--------------
样本里的 `ctx_player` 每一列都在 **[0,1]**（`v01 = clip(v*scale+bias, 0,1)`，见
`tasks/aim.py::_SCALE`）。0..1 的矩阵适合训练，不适合人眼检查：x=0.6147 这种值看不出
玩家在哪、朝哪。这里逐列反解：

    x y z          (v01*2-1) * MAP_RANGE + MAP_CENTER      → 世界坐标 HU
    yaw/pitch      atan2(sin, cos)                          → 度
    vx vy vz       (v01*2-1) * 300 / 1200                   → HU/s
    hp armor       v01 * 100                                → 点
    since_shot     v01 * 100                                → 秒（>=99.9 表示饱和）
    crouch/helmet/team/shot/has_kill/blinded/defuser        → 原样

**这张表必须和 `tasks/aim.py` 的 `_SCALE` / `_BIAS` 同步。** 那边改了列序或半宽，
这里不改就会画出"看起来有东西、其实是错的"图 —— 比报错危险得多。所以：

* 列序只认 `FEATURES`（本文件，一字不差地抄自 `AIM_FEATURES`）；
* `selftest.py` 在能 import 到 `scripts.training_data` 时会**逐值对拍**
  `AIM_FEATURES` / `_SCALE` / `_BIAS` / `MAP_CENTER` / `WEAPON_NAMES`（装了解析器才做，
  没装就跳过 —— 这个工具本身不依赖它们）；
* 样本列数比 `FEATURES` 少时不猜：缺的列按 0 解出来，同时把缺了哪几列写进
  `warnings`，前端顶部会飘一条警告。

## 标签的"体检"也在这里做

样本是**单帧**，但同一个文件里的 K 行是**同一个回合按时间升序**排的。这给了一组很硬的
不变式，`build_payload()` 会检查并把违反的写进 `warnings`（不抛错 —— 让人看见，而不是
让整页打不开）：

    ct_win      整个文件必须是常量（一个回合只有一个胜负）
    alive_end   整个文件必须是常量（"回合结束时还活着"和问的是哪一帧无关）
    next_death  一旦变成 10（之后没有死亡了），后面所有行都必须是 10
    next_kill   同上
    can_kill    随时间是**单调收缩**的：后面还能拿到的击杀只会更少，不会变多

这五条只要有任意一条大面积不成立，就说明造数那侧的"未来"算错了。它们比"形状对了"
有用得多 —— 形状错了会崩，标签错了只会让 loss 好看。

## 为什么不 import `scripts.training_data`
`scripts/training_data/__init__.py` 会连带 import `demo_extract` → `demoparser2`，
而这个工具故意只依赖 **numpy**（读 npz）—— 它要在任何一台装好训练依赖、但没装解析器
的机器上都能开。所以常量在这里有一份镜像。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── 仓库路径 ────────────────────────────────────────────────────────────────

#: scripts/training_data_visualizer/spatial_prediction/data.py → 仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 样本目录（和 `prepare.py --task spatial_prediction` 的默认输出一致）
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "demos" / "prepared" / "spatial_prediction"

#: 地图雷达图缓存（`app/radar.py` 生成，`app/data/` 是 gitignore 的）
RADAR_DIR = PROJECT_ROOT / "app" / "data" / "thumbs" / "_maps"


# ═══════════════════════════════════════════════════════════════════════════
# 常量镜像（来源：scripts/training_data/{tasks/aim.py,constants.py}）
# ═══════════════════════════════════════════════════════════════════════════

#: 和 `tasks/aim.py::AIM_FEATURES` **逐字一致**（`spatial_prediction` 原样复用）。
FEATURES: tuple[str, ...] = (
    "x", "y", "z",
    "yaw_sin", "yaw_cos", "pitch_sin", "pitch_cos",
    "vx", "vy", "vz",
    "hp", "crouch", "armor", "helmet",
    "team",
    "shot", "since_shot", "has_kill",
    "blinded", "defuser",
)

#: `v01 = clip(v * SCALE + BIAS, 0, 1)`，逐列对应 `FEATURES`。
SCALE = np.array(
    [
        0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5 / 300.0, 0.5 / 300.0, 0.5 / 1200.0,
        1.0 / 100.0, 1.0, 1.0 / 100.0, 1.0,
        1.0,
        1.0, 1.0 / 100.0, 1.0,
        1.0,
        1.0,
    ],
    dtype=np.float64,
)
BIAS = np.array(
    [
        0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5,
        0.0, 0.0, 0.0, 0.0,
        0.0,
        0.0, 0.0, 0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
assert len(SCALE) == len(BIAS) == len(FEATURES)

#: `constants.MAP_CENTER`（**全表**，8 张图）。别拿玩家位置去校准它。
MAP_CENTER: dict[str, tuple[float, float, float]] = {
    "de_mirage": (-605.8900146484375, -866.8900146484375, -171.6199951171875),
    "de_dust2": (-199.0, 977.0, 32.220001220703125),
    "de_inferno": (481.07000732421875, 1396.47998046875, 137.91000366210938),
    "de_nuke": (265.9599914550781, -772.5, -381.8999938964844),
    "de_overpass": (-2027.3900146484375, -812.9000244140625, 324.95001220703125),
    "de_ancient": (-435.5, -348.0, 43.650001525878906),
    "de_anubis": (-77.38999938964844, 618.9000244140625, -6.800000190734863),
    "de_cache": (724.157958984375, 394.74676513671875, 1757.4903564453125),
}

#: `MAP_TO_IDX` 是 `sorted(MAP_CENTER)` 派生的，所以反查也用排序后的顺序。
MAP_NAMES: tuple[str, ...] = tuple(sorted(MAP_CENTER))

#: `constants.MAP_RANGE_X/Y/Z`（Hammer 单位）。
MAP_RANGE = (5000.0, 5000.0, 2000.0)

#: `constants.WEAPON_NAMES`（全局武器词表，跨 demo 可比）。
WEAPON_NAMES: tuple[str, ...] = (
    "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18",
    "AK-47", "AUG", "AWP", "FAMAS", "Galil AR", "M4A4",
    "M4A1-S", "SG 553", "SCAR-20", "G3SG1", "SSG 08",
    "MAC-10", "MP9", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon",
    "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev",
    "P2000", "USP-S", "P250", "CZ75 Auto", "Tec-9", "R8 Revolver",
    "Zeus x27", "C4 Explosive", "Knife",
    "High Explosive Grenade", "Flashbang", "Smoke Grenade",
    "Molotov", "Incendiary Grenade", "Decoy Grenade",
)

#: `ctx_player_team` 的取值（和 `constants.TEAM_TO_IDX` / 任务里的 `TEAM_*` 一致）。
TEAM_CT = 0
TEAM_T = 1
TEAM_UNKNOWN = -1
TEAM_NAMES: dict[int, str] = {TEAM_CT: "CT", TEAM_T: "T", TEAM_UNKNOWN: "?"}

#: `next_death` / `next_kill` 的"这一帧之后没有下一次"（= 玩家数）。玩家下标只可能是
#: 0..9，所以 10 是天然的"无"，标签是干净的 11 类。和 `tasks/spatial_prediction.py`
#: 的 `NO_EVENT` 必须一致。
NO_EVENT = 10

#: `obstruction[i][j]` 的取值：i 看 j 有没有被挡住。
OBSTRUCTION_NAMES: dict[int, str] = {0: "通", 1: "烟", 2: "墙"}

#: 5 个标签的人话名（前端卡片标题用同一份文案，别在 JS 里再抄一遍）。
LABELS: tuple[dict[str, str], ...] = (
    {"key": "ct_win", "title": "CT 获胜", "kind": "bool",
     "note": "回合结束时 CT 是否赢下这一局"},
    {"key": "next_death", "title": "下一个死亡", "kind": "player",
     "note": "这一帧之后第一个死的人（10 = 本回合不会再有人死）"},
    {"key": "next_kill", "title": "下一个击杀", "kind": "player",
     "note": "这一帧之后第一个拿到击杀的人（10 = 本回合不会再有人击杀）"},
    {"key": "alive_end", "title": "终局存活", "kind": "mask",
     "note": "回合**结束**时还活着的人（10 个 bool）；回合结束瞬间中枪的算死"},
    {"key": "can_kill", "title": "还能击杀", "kind": "mask",
     "note": "这一帧之后还能拿到击杀的人（之前的击杀不算，10 个 bool）"},
)

#: 时间归一化的分母（`tasks/aim.py` 里同名常量）。
_BOMB_TIME_SPAN = 50.0
_PROJ_TIME_SPAN = 100.0
_SHOT_TIME_SPAN = 100.0
_ROUND_TIME_SPAN = 200.0

#: `tasks/spatial_prediction.py::DENSITY` 的默认值 —— 只用来在页面上写一句提示，
#: 真正的采样间隔在造数据那侧，样本里不存它。
DENSITY = 1.0

#: `view_render.py` / `obstruction.py` 的几何常量：眼睛高度、包围盒。
STANDING_EYE_HU = 64.0938
CROUCH_EYE_DROP_HU = 18.0
HULL_HALF_HU = 16.0
HULL_STAND_HU = 72.0
HULL_CROUCH_HU = 54.0

#: `ctx_projectile_id` → 人话。
PROJECTILE_TYPES: dict[int, str] = {
    0: "落地烟", 1: "落地火", 2: "飞行闪光", 3: "飞行诱饵",
    4: "飞行手雷", 5: "飞行燃烧瓶", 6: "飞行烟",
}

#: split 子目录名（和 `prepare.py` 的 `SPLITS` 一致）。
SPLITS = ("train", "valid", "test")

#: 一个文件里装几行样本：文件名末尾的 `_n{K}`（`tasks/spatial_prediction.py` 的契约）。
_COUNT_RE = re.compile(r"_n(\d+)$")
_ROUND_RE = re.compile(r"_r(\d+)(?:_n\d+)?$")

#: 必须有的 key。缺一个就说明这个 npz 不是 spatial_prediction 造的 —— 明确报错，
#: 不要"能解多少解多少"（那样只会画出一堆看起来合理、其实全是错位的图）。
REQUIRED_KEYS: tuple[str, ...] = (
    "ctx_player", "ctx_player_mask", "ctx_player_team", "ctx_player_weapon", "obstruction",
    "ctx_projectile", "ctx_projectile_id", "global_feat", "seconds", "frame_index",
    "ct_win", "alive_end", "can_kill", "next_death", "next_kill", "map_id", "round_id",
)


# ═══════════════════════════════════════════════════════════════════════════
# 扫描样本
# ═══════════════════════════════════════════════════════════════════════════


def map_from_name(name: str) -> str:
    """`de_anubis_f0_r0_n97.npz` → `de_anubis`（只看文件名，不打开）。"""
    stem = name[:-4] if name.endswith(".npz") else name
    return stem.split("_f", 1)[0]


def round_from_name(name: str) -> Optional[int]:
    """`de_anubis_f0_r7_n97.npz` → `7`；解析不出来返回 None（不猜）。"""
    m = _ROUND_RE.search(name[:-4] if name.endswith(".npz") else name)
    return int(m.group(1)) if m else None


def rows_of_name(name: str) -> Optional[int]:
    """文件名里的 `_n{K}`；没有就返回 None（`scan_files` 会把它当错误报出来）。"""
    m = _COUNT_RE.search(name[:-4] if name.endswith(".npz") else name)
    return int(m.group(1)) if m else None


def scan_files(data_dir: Path) -> list[dict[str, Any]]:
    """扫 `{train,valid,test}/*.npz`，**不打开文件**（列表要快）。

    每项：`{id, split, name, map, round, n, bytes, mtime}`。`n` 是文件名里的 `_n{K}`
    （= 这个文件里装了几行样本），前端用它把"文件列表"摊平成"样本列表"。

    名字里没有 `_n{K}` 的文件直接报错：静默当 1 行会让"一共几个样本"和实际行数对不上，
    然后要翻到那一条才炸。
    """
    out: list[dict[str, Any]] = []
    for split in SPLITS:
        folder = data_dir / split
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.npz")):
            try:
                st = path.stat()
            except OSError:
                continue
            n = rows_of_name(path.name)
            if n is None:
                raise ValueError(
                    f"{path.name} 的文件名里没有 `_n<K>`（K = 这个文件装了几行样本）。\n"
                    f"  spatial_prediction 的样本形如 de_mirage_f0_r3_n97.npz。\n"
                    f"  用 --data-dir 指向 prepared/spatial_prediction（不是 aim 的目录）"
                )
            out.append(
                {
                    "id": f"{split}/{path.name}",
                    "split": split,
                    "name": path.name,
                    "map": map_from_name(path.name),
                    "round": round_from_name(path.name),
                    "n": n,
                    "bytes": st.st_size,
                    "mtime": int(st.st_mtime),
                }
            )
    return out


def radar_info(map_name: str) -> Optional[dict[str, Any]]:
    """这张图有没有现成的雷达底图（`app/radar.py` 的缓存）。

    返回 `{file, extent: [u0,u1,v0,v1], url}`；没有就 None（前端退化成坐标网格）。
    `extent` 用的是**游戏坐标**：`u = x`、`v = y`。
    """
    if not map_name:
        return None
    matches = sorted(RADAR_DIR.glob(f"{map_name}.*.png"))
    if not matches:
        return None
    png = matches[-1]
    extent_path = png.with_suffix(".extent")
    if not extent_path.is_file():
        return None
    try:
        parts = [float(x) for x in extent_path.read_text(encoding="utf-8").split()]
    except (OSError, ValueError):
        return None
    if len(parts) != 4:
        return None
    return {
        "file": str(png),
        "extent": parts,
        "url": f"/api/radar/{map_name}",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 逆变换
# ═══════════════════════════════════════════════════════════════════════════


def _round(a: np.ndarray, digits: int = 3) -> list:
    """ndarray → 有限浮点 list（JSON 不接受 NaN/Inf）。"""
    return np.round(
        np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
        digits,
    ).tolist()


def vec_to_world(v01: np.ndarray, map_name: str) -> np.ndarray:
    """`[0,1]` 归一化坐标 → 世界坐标 HU，`(..., 3)` 一起处理。"""
    v = np.asarray(v01, dtype=np.float64) * 2.0 - 1.0
    cx, cy, cz = MAP_CENTER.get(map_name, (0.0, 0.0, 0.0))
    rx, ry, rz = MAP_RANGE
    out = np.empty_like(v)
    out[..., 0] = v[..., 0] * rx + cx
    out[..., 1] = v[..., 1] * ry + cy
    out[..., 2] = v[..., 2] * rz + cz
    return out


def decode_player(raw: np.ndarray, map_name: str) -> dict[str, list[float]]:
    """`(N, F)` 的 [0,1] 特征 → 物理量字典（每个值都是长度 N 的 list）。

    `N` 是这一帧的玩家数（正常 10）。`F` 可以是 18 / 19 / 20（旧样本没有
    blinded / defuser），缺的列按 0 解出来（= "没有"），由 `build_payload` 的
    warning 告诉用户该重造样本。
    """
    v = np.asarray(raw, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] < 18:
        raise ValueError(f"玩家特征形状不对：{v.shape}，至少要有 18 列")
    n = v.shape[0]
    un = v * 2.0 - 1.0

    yaw = np.degrees(np.arctan2(un[:, 3], un[:, 4]))
    pitch = np.degrees(np.arctan2(un[:, 5], un[:, 6]))
    crouch = np.clip(v[:, 11], 0.0, 1.0)
    cx, cy, cz = MAP_CENTER.get(map_name, (0.0, 0.0, 0.0))

    return {
        "x": _round(un[:, 0] * MAP_RANGE[0] + cx),
        "y": _round(un[:, 1] * MAP_RANGE[1] + cy),
        "z": _round(un[:, 2] * MAP_RANGE[2] + cz),
        "yaw": _round(yaw, 4),
        "pitch": _round(pitch, 4),
        "yaw_sin": _round(un[:, 3], 4),
        "yaw_cos": _round(un[:, 4], 4),
        "pitch_sin": _round(un[:, 5], 4),
        "pitch_cos": _round(un[:, 6], 4),
        "vx": _round(un[:, 7] * 300.0, 2),
        "vy": _round(un[:, 8] * 300.0, 2),
        "vz": _round(un[:, 9] * 1200.0, 2),
        "hp": _round(v[:, 10] * 100.0, 3),
        "crouch": _round(crouch, 4),
        "armor": _round(v[:, 12] * 100.0, 3),
        "helmet": _round(v[:, 13], 3),
        "team": _round(v[:, 14], 3),
        "shot": _round(v[:, 15], 3),
        "since_shot": _round(v[:, 16] * _SHOT_TIME_SPAN, 3),
        "has_kill": _round(v[:, 17], 3),
        "blinded": _round(_optional_col(v, 18, n), 3),
        "defuser": _round(_optional_col(v, 19, n), 3),
        # 由 crouch 派生的两个量，前端画图直接用，省得两边各写一次公式。
        "eye_z": _round(STANDING_EYE_HU - CROUCH_EYE_DROP_HU * crouch, 4),
        "box_h": _round(HULL_STAND_HU - (HULL_STAND_HU - HULL_CROUCH_HU) * crouch, 4),
    }


def _optional_col(v: np.ndarray, idx: int, n: int) -> np.ndarray:
    """取第 `idx` 列；旧样本没有这一列时整列 0（= "没有 / 未记录"），不猜。"""
    return v[:, idx] if v.shape[1] > idx else np.zeros(n)


def weapon_name(idx: int) -> str:
    """全局武器下标 → 名字；`-1` = 没有武器（死人 / 空手）。"""
    if 0 <= idx < len(WEAPON_NAMES):
        return WEAPON_NAMES[idx]
    return "" if idx < 0 else f"weapon#{idx}"


def _projectiles(rows: np.ndarray, ids: np.ndarray, map_name: str) -> list[dict[str, Any]]:
    """`(32,4)` + `(32,)` **这一帧**的道具 → 世界坐标 + 类型 + 已存在的秒数。"""
    out: list[dict[str, Any]] = []
    for r in range(min(len(rows), len(ids))):
        pid = int(ids[r])
        if pid < 0:
            continue
        p = vec_to_world(np.asarray(rows[r, :3], dtype=np.float64), map_name)
        elapsed = float(rows[r, 3]) * _PROJ_TIME_SPAN
        out.append(
            {
                "slot": r,
                "id": pid,
                "type": PROJECTILE_TYPES.get(pid, f"未知({pid})"),
                "x": round(float(p[0]), 2),
                "y": round(float(p[1]), 2),
                "z": round(float(p[2]), 2),
                "seconds": round(elapsed, 2),
                "clipped": bool(rows[r, 3] >= 0.999),
            }
        )
    return out


def _global_frame(global_feat: np.ndarray, map_name: str) -> dict[str, Any]:
    """`(5,)` 的逐帧全局状态 → 炸弹 / 回合时间的物理量（单帧，全部是标量）。"""
    g = np.asarray(global_feat, dtype=np.float64).reshape(-1)
    if g.size < 4:
        raise ValueError(f"global_feat 至少要有 4 列，实际 {g.size}")
    xyz = vec_to_world(g[None, :3], map_name)[0]
    planted = float(g[3])
    out: dict[str, Any] = {
        "per_frame": True,
        "x": round(float(xyz[0]), 2),
        "y": round(float(xyz[1]), 2),
        "z": round(float(xyz[2]), 2),
        "planted": bool(planted > 0.0),
        "planted_seconds": round(planted * _BOMB_TIME_SPAN, 2),
        "planted_clipped": bool(planted >= 0.999),
        "round_seconds": None,
        "round_clipped": None,
    }
    if g.size >= 5:
        out["round_seconds"] = round(float(g[4]) * _ROUND_TIME_SPAN, 2)
        out["round_clipped"] = bool(g[4] >= 0.999)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 标签体检（同一个文件的 K 行是同一回合按时间升序）
# ═══════════════════════════════════════════════════════════════════════════


def _sequence_warnings(arrays: dict[str, np.ndarray]) -> list[str]:
    """对**整个文件**的标签序列做不变式检查，返回人话警告（不抛错）。

    这几条是"未来标签"独有的硬约束，形状对了不代表标签对了，而标签错了只会让 loss
    好看 —— 所以这里必须能看见。
    """
    out: list[str] = []
    try:
        ct = np.asarray(arrays["ct_win"], dtype=bool).reshape(-1)
        end = np.asarray(arrays["alive_end"], dtype=bool)
        nd = np.asarray(arrays["next_death"]).reshape(-1)
        nk = np.asarray(arrays["next_kill"]).reshape(-1)
        can = np.asarray(arrays["can_kill"], dtype=bool)
    except (KeyError, ValueError):
        return out
    if ct.size <= 1:
        return out

    if not np.all(ct == ct[0]):
        bad = int(np.flatnonzero(ct != ct[0])[0])
        out.append(f"ct_win 在文件内不是常量（第 {bad} 行开始变了）：一个回合只有一个胜负")
    if end.ndim == 2 and end.shape[0] > 1 and not np.all(end == end[0]):
        bad = int(np.flatnonzero(np.any(end != end[0], axis=1))[0])
        out.append(
            f"alive_end 在文件内不是常量（第 {bad} 行开始变了）："
            "它和问的是哪一帧无关，应该是整条样本同一个值"
        )
    for key, arr in (("next_death", nd), ("next_kill", nk)):
        none_at = np.flatnonzero(arr == NO_EVENT)
        if none_at.size and not np.all(arr[none_at[0]:] == NO_EVENT):
            back = int(np.flatnonzero(arr[none_at[0]:] != NO_EVENT)[0] + none_at[0])
            out.append(
                f"{key} 在第 {none_at[0]} 行已经是 10（之后没有了），第 {back} 行又出现了玩家下标："
                "时间倒流了？"
            )
    if can.ndim == 2 and can.shape[0] > 1:
        grew = np.argwhere(np.diff(can.astype(np.int8), axis=0) > 0)
        if grew.size:
            row, pl = int(grew[0][0]), int(grew[0][1])
            out.append(
                f"can_kill 在第 {row + 1} 行对玩家 {pl} 从 False 变成了 True："
                "时间越往后还能拿到的击杀只会更少"
            )
    # 有人"之后还能拿击杀"却查不到下一个击杀者：只可能是那个击杀者当帧不活着
    if can.ndim == 2 and nk.size == can.shape[0] and np.any(can):
        bad = np.flatnonzero(can.any(axis=1) & (nk == NO_EVENT))
        if bad.size:
            out.append(
                f"第 {int(bad[0])} 行有 can_kill=True 但 next_kill=10："
                "有人之后会击杀，却找不到下一个击杀者"
            )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 样本 → 载荷
# ═══════════════════════════════════════════════════════════════════════════


def build_payload(
    arrays: dict[str, Any],
    *,
    split: str,
    name: str,
    k: int,
    path_label: str = "",
    file_bytes: int = 0,
) -> dict[str, Any]:
    """一个 npz（**整个文件**的数组）+ 行号 `k` → 给前端的完整载荷。

    故意和"读文件"分开：`selftest.py` 用内存里的合成数组就能验完整条逆变换 + 标签体检，
    不需要往磁盘上写夹具。
    """
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(
            f"{name} 缺 {', '.join(missing)} —— 这个 npz 不是 spatial_prediction 造的。\n"
            f"  aim / example 的样本布局完全不同；用 --data-dir 指向 "
            f"prepared/spatial_prediction"
        )

    warnings: list[str] = []
    ctx = np.asarray(arrays["ctx_player"], dtype=np.float32)
    if ctx.ndim != 3:
        raise ValueError(f"{name}: ctx_player 必须是 (K,10,F)，实际 {ctx.shape}")
    n_rows, n_pl, n_features = ctx.shape
    if not 0 <= int(k) < n_rows:
        raise ValueError(f"{name}: 行号 {k} 越界（这个文件有 {n_rows} 行）")

    mask = np.asarray(arrays["ctx_player_mask"]).astype(bool)[k]
    team = np.asarray(arrays["ctx_player_team"]).astype(np.int8)[k]
    weapon = np.asarray(arrays["ctx_player_weapon"]).astype(np.int8)[k]
    obstruction = np.asarray(arrays["obstruction"]).astype(np.int8)[k]
    proj = np.asarray(arrays["ctx_projectile"], dtype=np.float32)[k]
    proj_id = np.asarray(arrays["ctx_projectile_id"]).astype(np.int8)[k]
    global_feat = np.asarray(arrays["global_feat"], dtype=np.float32)[k]
    seconds = float(np.asarray(arrays["seconds"]).reshape(-1)[k])
    frame_index = int(np.asarray(arrays["frame_index"]).reshape(-1)[k])
    ct_win = bool(np.asarray(arrays["ct_win"]).reshape(-1)[k])
    alive_end = np.asarray(arrays["alive_end"], dtype=bool)[k]
    can_kill = np.asarray(arrays["can_kill"], dtype=bool)[k]
    next_death = int(np.asarray(arrays["next_death"]).reshape(-1)[k])
    next_kill = int(np.asarray(arrays["next_kill"]).reshape(-1)[k])
    map_id = int(np.asarray(arrays["map_id"]).reshape(()))
    round_id = int(np.asarray(arrays["round_id"]).reshape(()))

    if mask.shape != (n_pl,):
        warnings.append(f"ctx_player_mask 形状 {mask.shape} 和 {n_pl} 个玩家不一致")
    if team.shape != (n_pl,):
        warnings.append(f"ctx_player_team 形状 {team.shape} 和 {n_pl} 个玩家不一致")
    if weapon.shape != (n_pl,):
        warnings.append(f"ctx_player_weapon 形状 {weapon.shape} 和 {n_pl} 个玩家不一致")
    if obstruction.shape != (n_pl, n_pl):
        warnings.append(f"obstruction 形状 {obstruction.shape} 和 (10,10) 不一致")
    if alive_end.shape != (n_pl,) or can_kill.shape != (n_pl,):
        warnings.append("alive_end / can_kill 的长度和玩家数不一致")
    if n_features != len(FEATURES):
        missing_cols = "、".join(FEATURES[n_features:]) if n_features < len(FEATURES) else ""
        warnings.append(
            f"样本是 {n_features} 维，当前 FEATURES 是 {len(FEATURES)} 维"
            + (f"：缺 {missing_cols}（旧样本，按 0 显示，需要重新 prepare --overwrite）"
               if missing_cols else "")
        )
    if np.any(team < 0):
        warnings.append(
            "有玩家的队伍是 -1（未知）：`rnd[\"teams\"]` 可能缺人。"
            "`ctx_player` 第 15 列存的是 [0,1] 浮点，-1 会被压成 0，别拿它当队伍"
        )
    for label, idx in (("next_death", next_death), ("next_kill", next_kill)):
        # 11 类：0..9 是玩家，`NO_EVENT`（10）是"没有下一次"。别的值都是写错了。
        if not 0 <= idx <= NO_EVENT:
            warnings.append(
                f"{label}={idx} 不是合法取值（0..{n_pl - 1} 是玩家，{NO_EVENT} = 没有下一次）"
            )
        elif idx < n_pl and not bool(mask[idx]):
            warnings.append(
                f"{label}={idx} 是**已经死了**的玩家 —— 标签违反了"
                "「下一个死亡 / 击杀的人必须当前帧还活着」"
            )
    if np.any(can_kill & ~mask):
        bad = int(np.flatnonzero(can_kill & ~mask)[0])
        warnings.append(f"can_kill[{bad}]=True 但这个人当前帧是死的")
    if np.any(~mask & (weapon != -1)):
        warnings.append("有玩家已经死了却还带着武器（weapon != -1）")

    map_name = MAP_NAMES[map_id] if 0 <= map_id < len(MAP_NAMES) else map_from_name(name)
    if not 0 <= map_id < len(MAP_NAMES):
        warnings.append(f"map_id={map_id} 不在 0..{len(MAP_NAMES) - 1}，回退到文件名")
    elif map_from_name(name) != map_name:
        warnings.append(f"文件名里的地图（{map_from_name(name)}）和 map_id（{map_name}）不一致")

    global_dec = _global_frame(global_feat, map_name)
    if global_dec["round_seconds"] is not None:
        drift = abs(float(global_dec["round_seconds"]) - seconds)
        if drift > 0.02:
            warnings.append(
                f"`seconds`（{seconds:.3f} s）和 global_feat 的回合时间"
                f"（{global_dec['round_seconds']:.3f} s）差了 {drift:.3f} s：两者应当同步"
            )

    warnings.extend(_sequence_warnings(arrays))

    radar = radar_info(map_name)
    if radar is None:
        warnings.append(f"没有 {map_name} 的雷达底图，俯视图退化成坐标网格")

    phys = decode_player(ctx[k], map_name)

    # 队伍以 `ctx_player_team` 为准（-1 = 未知）。**不**用 `ctx_player` 第 15 列兜底：
    # 那一列是 [0,1] 浮点，-1 已经和 CT 分不开了 —— 真出现 -1 就是上面那条警告。
    teams = [TEAM_NAMES.get(int(team[p]), "?") for p in range(n_pl)]

    return {
        "id": f"{split}/{name}",
        "split": split,
        "name": name,
        "row": int(k),
        "rows": int(n_rows),
        "path": path_label,
        "bytes": int(file_bytes),
        "map_id": int(map_id),
        "map_name": map_name,
        "round_id": int(round_id),
        "frame_index": frame_index,
        "seconds": round(seconds, 4),
        "n_players": int(n_pl),
        "n_features": int(n_features),
        "features": list(FEATURES[:n_features]),
        # 单帧 → channel-major 的**扁平** list：`players.x[p]` 就是第 p 个人。
        "players": {
            "alive": mask.tolist(),
            "team": team.tolist(),
            "teams": teams,
            "weapon": weapon.tolist(),
            "weapon_names": [weapon_name(int(w)) for w in weapon],
            "phys": phys,
            # (10, F) 的 [0,1] 原值：特征矩阵默认显示它（物理值由前端切）
            "raw": _round(ctx[k], 4),
        },
        "labels": {
            "ct_win": ct_win,
            "alive_end": alive_end.tolist(),
            "can_kill": can_kill.tolist(),
            "next_death": next_death,
            "next_kill": next_kill,
            "next_death_team": teams[next_death] if 0 <= next_death < n_pl else None,
            "next_kill_team": teams[next_kill] if 0 <= next_kill < n_pl else None,
            "alive_end_n": int(alive_end.sum()),
            "can_kill_n": int(can_kill.sum()),
        },
        "labels_meta": list(LABELS),
        "obstruction": obstruction.tolist(),
        "obstruction_names": {str(key): val for key, val in OBSTRUCTION_NAMES.items()},
        "projectiles": _projectiles(proj, proj_id, map_name),
        "global": global_dec,
        "radar": radar,
        "weapons": {"names": list(WEAPON_NAMES)},
        "files": [
            {"key": key, "shape": list(np.shape(arrays[key])),
             "dtype": str(np.asarray(arrays[key]).dtype)}
            for key in sorted(arrays)
        ],
        "warnings": warnings,
        # 前端绘图用的几何常量（别在 JS 里再抄一遍数字）
        "geometry": {
            "hull_half": HULL_HALF_HU,
            "hull_stand": HULL_STAND_HU,
            "hull_crouch": HULL_CROUCH_HU,
            "eye_stand": STANDING_EYE_HU,
            "hu_to_m": 0.0254,
        },
        "density": DENSITY,
    }


def load_sample(
    path: Path,
    *,
    split: str,
    name: str,
    k: int = 0,
) -> dict[str, Any]:
    """读一个 npz（整个文件）→ 取第 `k` 行 → `build_payload()`。

    ⚠ npz 是 zip：`z[key]` 会把**整个成员**解出来（`mmap_mode` 对 npz 无效），所以翻
    同一文件的第 2 行和第 97 行代价一样。可视化一次只看一行，这个代价可以接受；
    训练侧要连读几千帧的话请看 `tasks/spatial_prediction.py` 里 `read_sample` 的说明。
    """
    with np.load(path) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    label = (
        path.relative_to(PROJECT_ROOT).as_posix()
        if path.is_relative_to(PROJECT_ROOT) else str(path)
    )
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return build_payload(
        arrays, split=split, name=name, k=k, path_label=label, file_bytes=size
    )


# ═══════════════════════════════════════════════════════════════════════════
# 给 selftest 用的小工具
# ═══════════════════════════════════════════════════════════════════════════


def invert_columns(raw: np.ndarray) -> np.ndarray:
    """`(N,F)` 的 [0,1] → **未缩放**的列（`(v01 - bias) / scale`）。只给测试用。"""
    return (np.asarray(raw, dtype=np.float64) - BIAS[: raw.shape[1]]) / SCALE[: raw.shape[1]]


def encode_row(phys: dict[str, float]) -> np.ndarray:
    """物理量 → 20 列 [0,1]（用 SCALE/BIAS 正向编码，和 `tasks/aim.py` 一致）。

    只给 selftest / 造合成样本用；**别**在生产代码里用它"反推"样本。
    """
    raw = np.zeros(len(FEATURES), dtype=np.float64)
    cx, cy, cz = MAP_CENTER.get("de_mirage", (0.0, 0.0, 0.0))
    raw[0] = (phys["x"] - cx) / MAP_RANGE[0] * 0.5 + 0.5
    raw[1] = (phys["y"] - cy) / MAP_RANGE[1] * 0.5 + 0.5
    raw[2] = (phys["z"] - cz) / MAP_RANGE[2] * 0.5 + 0.5
    raw[3] = np.sin(np.radians(phys["yaw"])) * 0.5 + 0.5
    raw[4] = np.cos(np.radians(phys["yaw"])) * 0.5 + 0.5
    raw[5] = np.sin(np.radians(phys["pitch"])) * 0.5 + 0.5
    raw[6] = np.cos(np.radians(phys["pitch"])) * 0.5 + 0.5
    raw[7] = phys["vx"] * SCALE[7] + BIAS[7]
    raw[8] = phys["vy"] * SCALE[8] + BIAS[8]
    raw[9] = phys["vz"] * SCALE[9] + BIAS[9]
    raw[10] = phys["hp"] / 100.0
    raw[11] = phys["crouch"]
    raw[12] = phys["armor"] / 100.0
    raw[13] = phys["helmet"]
    raw[14] = phys["team"]
    raw[15] = phys["shot"]
    raw[16] = min(phys["since_shot"] / 100.0, 1.0)
    raw[17] = phys["has_kill"]
    raw[18] = phys["blinded"]
    raw[19] = phys["defuser"]
    return np.clip(raw, 0.0, 1.0)


def sample_path(data_dir: Path, split: str, name: str) -> Path:
    """`split` / `name` → 磁盘路径。**拒绝路径穿越**（name 只能是该 split 下的文件名）。"""
    if split not in SPLITS:
        raise ValueError(f"未知 split: {split!r}")
    if not name or "/" in name or "\\" in name or not name.endswith(".npz"):
        raise ValueError(f"非法样本名: {name!r}")
    folder = (data_dir / split).resolve()
    path = (folder / name).resolve()
    if path.parent != folder:
        raise ValueError(f"样本路径越界: {name!r}")
    if not path.is_file():
        raise FileNotFoundError(f"样本不存在: {path}")
    return path
