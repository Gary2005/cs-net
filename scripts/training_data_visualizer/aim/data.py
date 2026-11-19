"""npz → 可视化载荷（纯数据层，不碰 HTTP）。

这个文件只做一件事：把 `aim.py` 写出来的样本**逆变换**回人看得懂的量，并压成
一份 JSON 友好的 dict。前端拿到之后不再做任何"数据语义"上的换算（除了画图）。

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
这里不改就会画出"看起来有东西、其实是错的"图——比报错危险得多。所以：

* 列序只认 `AIM_FEATURES`（本文件 `FEATURES`，一字不差地抄过来）；
* `selftest.py` 会在能 import 到 `scripts.training_data.constants` 时对拍
  `MAP_CENTER` / `WEAPON_NAMES`；
* 样本列数比 `FEATURES` 少时不猜：缺的列按 0 解出来，同时把缺了哪几列写进
  `warnings`，前端顶部会飘一条警告。

## 这一版相对旧版的三个变化

1. **没有视角渲染图了**（`atk_view` / `tgt_view` 从样本里删掉了，换成了逐帧的
   `obstruction`）。所以这里不再有 `/api/view/...` 那套 PNG，也没有视图的分位统计。
2. **全场逐帧**：`ctx_player (T,10,20)` + `ctx_player_mask (T,10)` + 逐帧武器。载荷里
   把它压成 **channel-major** 的 `players`（每个物理量一个 `(T,10)` 矩阵），比
   `(T,10,21)` 的嵌套字典小得多，前端取"第 t 帧第 p 个人"就是 `players.x[t][p]`。
3. **A / B 不再是两条独立轨迹**：谁是 A 只看 `ctx_player_mask == 1`。载荷里仍然
   给出 `atk` / `tgt` 两个"单人视图"（就是玩家 A / B 那一行），因为 HUD、第一人称
   画布、特征表都只需要这两个人；但同时给出 `players`，让"旁边还有谁"也能画。

## 为什么不 import `scripts.training_data`
`scripts/training_data/__init__.py` 会连带 import `demo_extract` → `demoparser2`，
而这个工具故意只依赖 **numpy**（读 npz）——它要在任何一台装好训练依赖、但没装
解析器的机器上都能开。所以常量在这里有一份镜像。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── 仓库路径 ────────────────────────────────────────────────────────────────

#: scripts/training_data_visualizer/aim/data.py → 仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 样本目录（和 `prepare.py --task aim` 的默认输出一致）
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "demos" / "prepared" / "aim"

#: 地图雷达图缓存（`app/radar.py` 生成，`app/data/` 是 gitignore 的）
RADAR_DIR = PROJECT_ROOT / "app" / "data" / "thumbs" / "_maps"


# ═══════════════════════════════════════════════════════════════════════════
# 常量镜像（来源：scripts/training_data/{tasks/aim.py,constants.py}）
# ═══════════════════════════════════════════════════════════════════════════

#: 和 `tasks/aim.py::AIM_FEATURES` **逐字一致**。前 18 列是最早就有的，第 19 列
#: `blinded`、第 20 列 `defuser` 是后加的；读旧样本时按 `features[:n]` 截断。
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

#: `ctx_player_mask` 的取值（和 `tasks/aim.py` 的 `MASK_*` 一致）。
MASK_NAMES: dict[int, str] = {0: "死亡", 1: "A(击杀者)", 2: "B(受害者)", 3: "其他存活"}

#: `obstruction[t][i][j]` 的取值：i 看 j 有没有被挡住。
OBSTRUCTION_NAMES: dict[int, str] = {0: "通", 1: "烟", 2: "墙"}

#: `tasks/aim.py`：时间归一化的分母，逆变换时乘回去。
_BOMB_TIME_SPAN = 50.0
_PROJ_TIME_SPAN = 100.0
_SHOT_TIME_SPAN = 100.0
_ROUND_TIME_SPAN = 200.0

#: `tasks/aim.py::LOOKBACK`：窗口跨度**固定 4 秒**（早先在 [0.1,4] 上随机取）。
LOOKBACK = 4.0
#: 采样率：`round_seconds` 实测逐帧 +1/64 s，所以固定窗口的帧数 `= LOOKBACK × 64 + 1 = 257`。
TICK_HZ = 64.0
#: 窗口应有的帧数（跨到回合开头被截短时会更少，`load_sample` 会提醒）。
EXPECTED_FRAMES = int(LOOKBACK * TICK_HZ) + 1

#: `view_render.py` / `obstruction.py` 的几何常量：眼睛高度、包围盒。
STANDING_EYE_HU = 64.0938
CROUCH_EYE_DROP_HU = 18.0
HULL_HALF_HU = 16.0
HULL_STAND_HU = 72.0
HULL_CROUCH_HU = 54.0

#: 第一人称画布用的 FOV（和浏览器 `visibility.js::DEFAULT_FOV_DEG` 一致）。
FOV_H_DEG = 90.0
FOV_V_DEG = 73.7398

#: `ctx_projectile_id` → 人话。
PROJECTILE_TYPES: dict[int, str] = {
    0: "落地烟",
    1: "落地火",
    2: "飞行闪光",
    3: "飞行诱饵",
    4: "飞行手雷",
    5: "飞行燃烧瓶",
    6: "飞行烟",
}

TEAM_NAMES = {0: "CT", 1: "T"}
SPLITS = ("train", "valid", "test")


# ═══════════════════════════════════════════════════════════════════════════
# 扫描样本
# ═══════════════════════════════════════════════════════════════════════════


def map_from_name(name: str) -> str:
    """`de_anubis_f0_r0_k001.npz` → `de_anubis`。"""
    head = name.split("_f", 1)[0]
    return head if head in MAP_CENTER else head


def scan_samples(data_dir: Path) -> list[dict[str, Any]]:
    """扫 `{train,valid,test}/*.npz`，**不打开文件**（列表要快）。

    元信息（帧数 / 地图 / 武器）等真正选中样本时再从 npz 里读。几百个样本的文件名
    扫描是毫秒级，逐个 `np.load` 就不是了。
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
            out.append(
                {
                    "id": f"{split}/{path.name}",
                    "split": split,
                    "name": path.name,
                    "map": map_from_name(path.name),
                    "bytes": st.st_size,
                    "mtime": int(st.st_mtime),
                }
            )
    return out


def radar_info(map_name: str) -> Optional[dict[str, Any]]:
    """这张图有没有现成的雷达底图（`app/radar.py` 的缓存）。

    返回 `{file, extent: [u0,u1,v0,v1], url}`；没有就 None（前端退化成网格）。
    `extent` 用的是**游戏坐标**：`u = x`、`v = y`（`app/radar.py::_uv_y`）。
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
    return np.round(np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0,
                                  posinf=0.0, neginf=0.0), digits).tolist()


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
    """`(T, F)` 的 [0,1] 特征 → 物理量字典（每个值都是长度 T 的 list）。

    `F` 可以是 18 / 19 / 20（旧样本没有 blinded / defuser）。缺的列按 0 解出来
    （= "没有"），由 `load_sample` 里的 warning 告诉用户该重造样本。
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
        # 文件里存的是 4 个 sin/cos 列（不是角度）；这里还原成 [-1,1] 的原始值，
        # 特征表里能直接和"0.5 + 0.5·sin"对上。
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


def decode_players(raw: np.ndarray, map_name: str) -> dict[str, list[list[float]]]:
    """`(T, 10, F)` → `{通道: (T,10) 的嵌套 list}`（channel-major）。

    绕一圈复用 `decode_player`：先把 `(T,10)` 摊平成 `(T*10, F)` 解一次，再折回去。
    10 倍的行数换"只写一份反解公式"，这个买卖值得 —— 两份公式迟早对不上。
    """
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"ctx_player 必须是三维 (T,10,F)，实际 {a.shape}")
    t, n, _ = a.shape
    flat = decode_player(a.reshape(t * n, -1), map_name)
    return {
        k: [v[t_i * n:(t_i + 1) * n] for t_i in range(t)]
        for k, v in flat.items()
    }


def _weapon_series(values: np.ndarray) -> tuple[list[dict[str, Any]], list[int]]:
    """全局武器下标 → (legend, 每帧在 legend 里的位置)；-1（无/未知）保持 -1。"""
    idx = [int(x) for x in np.asarray(values).reshape(-1)]
    used = sorted({i for i in idx if i >= 0})
    legend = []
    pos = {}
    for i in used:
        name = WEAPON_NAMES[i] if 0 <= i < len(WEAPON_NAMES) else f"weapon#{i}"
        pos[i] = len(legend)
        legend.append({"idx": i, "name": name})
    return legend, [pos.get(i, -1) for i in idx]


def _projectiles(rows: np.ndarray, ids: np.ndarray, map_name: str) -> list[dict[str, Any]]:
    """`(32,4)` + `(32,)` **某一帧**的道具 → 世界坐标 + 类型 + 已存在的秒数。"""
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
                # 时间归一化是 clip 的：贴着 1 说明已经过去 100 s 以上（实际不会）
                "clipped": bool(rows[r, 3] >= 0.999),
            }
        )
    return out


def _projectile_frames(
    rows: np.ndarray, ids: np.ndarray, map_name: str
) -> list[list[dict[str, Any]]]:
    """`(T,32,4)` + `(T,32)` → 每帧一个道具列表。"""
    return [
        _projectiles(rows[i], ids[i], map_name) for i in range(len(rows))
    ]


def _global_frames(
    global_feat: np.ndarray, map_name: str, n_frames: int
) -> dict[str, list[Any]]:
    """`global_feat` `(T,5)` → 逐帧的炸弹 / 回合时间（channel-major，和 `players` 一致）。

    * `(T,5)`：现在的布局，五列逐帧。
    * `(5,)` 或 `(4,)`：旧样本只有**首帧**一帧。这时把那一帧**广播**成 `T` 帧 ——
      注意广播只是为了让前端能按下标取值，它**不是**数据里真有的逐帧信息；前端按
      `per_frame=False` 明确写"首帧"，`load_sample` 也会在警告条里提示重造。

    第 5 列"回合时间"是后来加的：旧样本只有 4 列，此时 `round_seconds` 全 `None`
    （前端显示"未记录"），而不是把 0 当成"回合刚开始"。
    """
    g = np.asarray(global_feat, dtype=np.float64)
    per_frame = g.ndim == 2 and g.shape[0] == n_frames
    if g.ndim == 1:                      # 旧布局：单帧快照
        g = np.repeat(g[None, :], max(n_frames, 1), axis=0)
    elif not per_frame:                  # 帧数对不上：按首帧广播，别让前端越界
        g = np.repeat(g[:1], max(n_frames, 1), axis=0)
    n, n_col = g.shape

    xyz = vec_to_world(g[:, :3], map_name) if n_col >= 4 else np.zeros((n, 3))
    planted = g[:, 3] if n_col >= 4 else np.zeros(n)
    out: dict[str, list[Any]] = {
        "per_frame": per_frame,
        "x": _round(xyz[:, 0], 2),
        "y": _round(xyz[:, 1], 2),
        "z": _round(xyz[:, 2], 2),
        "planted": [bool(v > 0.0) for v in planted],
        "planted_seconds": _round(planted * _BOMB_TIME_SPAN, 2),
        "planted_clipped": [bool(v >= 0.999) for v in planted],
        "round_seconds": None,
        "round_clipped": None,
    }
    if n_col >= 5:
        out["round_seconds"] = _round(g[:, 4] * _ROUND_TIME_SPAN, 2)
        out["round_clipped"] = [bool(v >= 0.999) for v in g[:, 4]]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 样本 → 载荷
# ═══════════════════════════════════════════════════════════════════════════

#: 旧版样本里存在、新版删掉的 key。读到它们就明确报"需要重造"，别硬解。
_LEGACY_KEYS = ("atk", "tgt", "atk_view", "tgt_view", "atk_weapon", "tgt_weapon")


def load_sample(path: Path, *, split: str, name: str, data_dir: Path) -> dict[str, Any]:
    """读一个 npz，返回给前端的完整载荷。"""
    warnings: list[str] = []
    with np.load(path) as z:
        files = z.files
        if any(k in files for k in _LEGACY_KEYS):
            raise ValueError(
                f"{name} 是旧版样本（含 {'/'.join(k for k in _LEGACY_KEYS if k in files)}）："
                "新版删掉了 atk/tgt/视角图，改成逐帧全场 + 遮挡矩阵。"
                "跑 `python -m scripts.training_data.prepare --task aim --overwrite` 重造"
            )
        ctx_raw = np.asarray(z["ctx_player"], dtype=np.float32)
        ctx_mask = np.asarray(z["ctx_player_mask"]).astype(np.int8)
        ctx_weapon = np.asarray(z["ctx_player_weapon"]).astype(np.int8)
        obstruction = np.asarray(z["obstruction"]).astype(np.int8)
        proj = np.asarray(z["ctx_projectile"], dtype=np.float32)
        proj_id = np.asarray(z["ctx_projectile_id"]).astype(np.int8)
        global_feat = np.asarray(z["global_feat"], dtype=np.float32)
        seconds = np.asarray(z["seconds"], dtype=np.float64).reshape(-1)
        map_id = int(np.asarray(z["map_id"]).reshape(()))
        hs = bool(np.asarray(z["hs"]).reshape(()))
        dmg = int(np.asarray(z["dmg"]).reshape(()))
        shapes = {k: list(z[k].shape) for k in files}
        dtypes = {k: str(z[k].dtype) for k in files}

    if ctx_raw.ndim != 3:
        raise ValueError(f"{name}: ctx_player 必须是 (T,10,F)，实际 {ctx_raw.shape}")
    n, n_pl, n_features = ctx_raw.shape
    if ctx_mask.shape != (n, n_pl):
        warnings.append(f"ctx_player_mask 形状 {ctx_mask.shape} 和 ctx_player 不一致")
    if ctx_weapon.shape != (n, n_pl):
        warnings.append("ctx_player_weapon 形状和 ctx_player 不一致")
    if obstruction.shape != (n, n_pl, n_pl):
        warnings.append(f"obstruction 形状 {obstruction.shape} 和 (T,10,10) 不一致")
    if proj.shape[:1] != (n,) or proj_id.shape[:1] != (n,):
        warnings.append("ctx_projectile 的帧数和 ctx_player 不一致")
    if len(seconds) != n:
        warnings.append(f"seconds 长度 {len(seconds)} != 帧数 {n}")
    if n_features != len(FEATURES):
        missing = "、".join(FEATURES[n_features:]) if n_features < len(FEATURES) else ""
        warnings.append(
            f"样本是 {n_features} 维，当前 AIM_FEATURES 是 {len(FEATURES)} 维"
            + (f"：缺 {missing}（旧样本，按 0 显示，需要重新 prepare --overwrite）" if missing else "")
        )
    if global_feat.ndim != 2 or global_feat.shape[0] != n:
        warnings.append(
            f"global_feat 形状 {list(global_feat.shape)} 不是 (T,5)："
            "单帧快照是旧样本，只能广播到每帧显示；重新 prepare --overwrite 才会逐帧"
        )
    if global_feat.shape[-1] < 5:
        warnings.append("global_feat 只有 4 列（旧样本，没有回合时间）：重新 prepare --overwrite")
    if n != EXPECTED_FRAMES:
        warnings.append(
            f"T={n} 帧，固定 4 秒窗口应该是 {EXPECTED_FRAMES} 帧："
            "要么是旧样本（跨度 0.1–4 s 随机取），要么窗口被截到回合开头"
        )

    map_name = MAP_NAMES[map_id] if 0 <= map_id < len(MAP_NAMES) else map_from_name(name)
    if not 0 <= map_id < len(MAP_NAMES):
        warnings.append(f"map_id={map_id} 不在 0..{len(MAP_NAMES) - 1}，回退到文件名")
    elif map_from_name(name) != map_name:
        warnings.append(
            f"文件名里的地图（{map_from_name(name)}）和 map_id={map_id}（{map_name}）不一致"
        )

    players = decode_players(ctx_raw, map_name)

    # A / B 的下标：mask 里 1 / 2 的位置。窗口成立的前提是这两个人全程活着，所以
    # 每帧都应当是同一个下标；不一致就说明造数那侧的约束破了，报出来而不是猜。
    atk_index = tgt_index = None
    first = ctx_mask[0] if n else np.zeros(n_pl, np.int8)
    a_rows = np.flatnonzero(first == 1)
    b_rows = np.flatnonzero(first == 2)
    if a_rows.size == 1:
        atk_index = int(a_rows[0])
    if b_rows.size == 1:
        tgt_index = int(b_rows[0])
    if atk_index is None or tgt_index is None:
        warnings.append(
            "第一帧的 ctx_player_mask 里找不到唯一的 A(=1) / B(=2)：样本可能不是 aim 任务的"
        )
    else:
        for label, p, want in (("A", atk_index, 1), ("B", tgt_index, 2)):
            if not np.all(ctx_mask[:, p] == want):
                warnings.append(
                    f"{label}（第 {p} 位）的 mask 沿窗口不恒为 {want}："
                    "窗口内该玩家死过，样本不该产出"
                )

    # 单人视图：把 A / B 那一行单独切出来，供 HUD / 第一人称画布 / 特征表使用。
    atk_row = ctx_raw[:, atk_index] if atk_index is not None else np.zeros((n, n_features), np.float32)
    tgt_row = ctx_raw[:, tgt_index] if tgt_index is not None else np.zeros((n, n_features), np.float32)
    atk_weapon = ctx_weapon[:, atk_index] if atk_index is not None else np.full(n, -1, np.int8)
    tgt_weapon = ctx_weapon[:, tgt_index] if tgt_index is not None else np.full(n, -1, np.int8)
    atk_legend, atk_weapon_series = _weapon_series(atk_weapon)
    tgt_legend, tgt_weapon_series = _weapon_series(tgt_weapon)

    radar = radar_info(map_name)
    if radar is None:
        warnings.append(f"没有 {map_name} 的雷达底图，俯视图退化成坐标网格")

    mask_names = {int(v): MASK_NAMES.get(int(v), f"未知({int(v)})") for v in np.unique(ctx_mask)}

    return {
        "id": f"{split}/{name}",
        "split": split,
        "name": name,
        "path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
        "bytes": path.stat().st_size,
        "map_id": map_id,
        "map_name": map_name,
        "frames": n,
        "n_players": n_pl,
        "n_features": n_features,
        "features": list(FEATURES[:n_features]),
        "fps": round((n - 1) / float(seconds[-1]), 2) if n > 1 and seconds[-1] > 0 else None,
        "duration": round(float(seconds[-1]), 3) if len(seconds) else 0.0,
        "seconds": _round(seconds, 4),
        # A / B 的单人视图（固定方向：A = mask==1 = 磁盘上的击杀者）
        "atk": {"index": atk_index, "raw": _round_matrix(atk_row, 4), "phys": decode_player(atk_row, map_name)},
        "tgt": {"index": tgt_index, "raw": _round_matrix(tgt_row, 4), "phys": decode_player(tgt_row, map_name)},
        # 全场逐帧（channel-major）
        "players": {
            "mask": ctx_mask.tolist(),
            "weapon": ctx_weapon.tolist(),
            "phys": players,
            "mask_names": mask_names,
        },
        # 逐帧两两遮挡：(T,10,10) 的 0/1/2
        "obstruction": obstruction.tolist(),
        "obstruction_names": {str(k): v for k, v in OBSTRUCTION_NAMES.items()},
        # 逐帧道具
        "projectiles": _projectile_frames(proj, proj_id, map_name) if n else [],
        "global": _global_frames(global_feat, map_name, n),
        "weapons": {
            "atk": atk_legend,
            "tgt": tgt_legend,
            "atk_series": atk_weapon_series,
            "tgt_series": tgt_weapon_series,
            "names": list(WEAPON_NAMES),
        },
        "hs": hs,
        "dmg": dmg,
        "radar": radar,
        "files": [
            {"key": k, "shape": shapes[k], "dtype": dtypes[k]} for k in sorted(files)
        ],
        "warnings": warnings,
        # 前端绘图用的几何常量（别在 JS 里再抄一遍数字）
        "geometry": {
            "fov_h": FOV_H_DEG,
            "fov_v": FOV_V_DEG,
            "hull_half": HULL_HALF_HU,
            "hull_stand": HULL_STAND_HU,
            "hull_crouch": HULL_CROUCH_HU,
            "eye_stand": STANDING_EYE_HU,
            "hu_to_m": 0.0254,
        },
    }


def _round_matrix(a: np.ndarray, digits: int) -> list[list[float]]:
    """`(T,F)` float32 → 嵌套 list（保留 4 位小数，够人看、又能显著压缩 JSON）。"""
    return np.round(np.nan_to_num(np.asarray(a, dtype=np.float64)), digits).tolist()


# ═══════════════════════════════════════════════════════════════════════════
# 给 selftest 用的小工具
# ═══════════════════════════════════════════════════════════════════════════


def invert_columns(raw: np.ndarray) -> np.ndarray:
    """`(T,F)` 的 [0,1] → **未缩放**的列（`(v01 - bias) / scale`）。

    只给测试用：检查 `decode_player` 和 `_SCALE`/`_BIAS` 确实是互逆的。
    对 `scale == 0` 的列（不存在）会爆炸，这里不兜底。
    """
    return (np.asarray(raw, dtype=np.float64) - BIAS[: raw.shape[1]]) / SCALE[: raw.shape[1]]


def angles_from_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """直接从原始 [0,1] 特征里还原 `(yaw, pitch)` 度数，给测试和画图共用。"""
    v = np.asarray(raw, dtype=np.float64)
    return (
        np.degrees(np.arctan2(v[:, 3] * 2.0 - 1.0, v[:, 4] * 2.0 - 1.0)),
        np.degrees(np.arctan2(v[:, 5] * 2.0 - 1.0, v[:, 6] * 2.0 - 1.0)),
    )


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
