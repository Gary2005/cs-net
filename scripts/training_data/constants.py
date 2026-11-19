"""常量：特征列、地图中心、武器词表、体检阈值。

想换特征列 / 加地图 / 调阈值，都在这个文件里改，不影响别处。
"""

from __future__ import annotations

from . import _paths  # noqa: F401  —— 副作用：把 tools/ 加进 sys.path

from demo_extract import recipes  # noqa: E402

# ── 特征列 ─────────────────────────────────────────────────────────────────

N_PLAYERS = 10

#: 要求的文件格式。**只认 v7**——手上 4 个文件全是 v7。
#: 老格式缺 on_ground/jump/land/crouch 这几列，加载时直接报错，不静默补 0
#: （补 0 的话那几列全变成常量，训练完才发现，很难查）。
REQUIRED_FORMAT = "cs2.demo.v7"

#: 每 tick 的玩家特征列，顺序即 `player_tensor` 的 F 轴。
#: v7 = 20 列：
#:   0 x  1 y  2 z | 3 yaw  4 pitch | 5 vx  6 vy  7 vz | 8 hp  9 armor
#:   10 flash  11 flash_alpha | 12 weapon  13 place | 14 shots  15 footsteps
#:   16 on_ground  17 jump  18 land | 19 crouch
#:
#: ⚠ `recipes.player_tensor()` 的**默认参数是最老的 16 列集合**（为了向后兼容，
#: 让 v6 文件还能喂老管线），所以取值时必须显式传 FEATURES，别用默认值——
#: 默认值会悄悄少掉 on_ground/jump/land/crouch 这 4 列。
FEATURES = recipes.PLAYER_FEATURES_V7

#: 各列在 FEATURES 里的下标，给 features.py 用（别在那边写魔法数字）。
COL = {name: i for i, name in enumerate(FEATURES)}

# ── 坐标归一化 ─────────────────────────────────────────────────────────────

#: 地图中心（Hammer 单位）。**全表**抄自 dev 仓库
#: `cs-net/scripts/training_data/config.py` 的 MAP_CONFIG（8 张，和那边
#: `docs/training-data-format.md` §11 的表一致）。加图去那边抄，别自己编。
#:
#: ⚠ **这不是"可玩区域的中心"**，是地图自己的坐标原点偏移。实测手上 4 张图：
#: 玩家位置 bbox 中心和表里差 8–17 HU（mirage / inferno / dust2），但
#: **nuke 的 x 差了 496 HU**（玩家 bbox 中心 762，表里 266）。
#: 所以别拿"玩家都在哪"去校准这张表——它保证的是"数值落在 ±5000 内有界"，
#: 不是"几何上正中"。
#:
#: 前 4 张有 demo 可以对照，后 4 张（overpass / ancient / anubis / cache）
#: **手上没有 demo，没验证过**，只是原样抄过来。等爬到了这些图再核对。
MAP_CENTER: dict[str, tuple[float, float, float]] = {
    "de_mirage":   (-605.8900146484375, -866.8900146484375, -171.6199951171875),
    "de_dust2":    (-199.0, 977.0, 32.220001220703125),
    "de_inferno":  (481.07000732421875, 1396.47998046875, 137.91000366210938),
    "de_nuke":     (265.9599914550781, -772.5, -381.8999938964844),
    "de_overpass": (-2027.3900146484375, -812.9000244140625, 324.95001220703125),
    "de_ancient":  (-435.5, -348.0, 43.650001525878906),
    "de_anubis":   (-77.38999938964844, 618.9000244140625, -6.800000190734863),
    "de_cache":    (724.157958984375, 394.74676513671875, 1757.4903564453125),
}

#: 地图名 → 编号（0..7），由 `MAP_CENTER` 的**排序键**派生：以后往表里插新图，
#: 已有编号不会整体平移（用定义顺序就会）。
#:
#: ⚠ 只有这 8 张图有编号。数据集里遇到别的地图应当**整回合跳过**，而不是存 -1：
#: `normalize_xyz()` 对未知地图是静默兜底的（不减中心、只按 ±5000 裁），那种坐标
#: 已经失真，混进训练集会污染全局分布。
#:
#: 这张表同时也是"哪些地图算标准图"的唯一出处：`prepare.py` 在抽样阶段就用
#: `name in MAP_TO_IDX` 把别的地图**整个 csz 筛掉**（和 `tasks/aim.py` 里
#: `MAP_TO_IDX.get(map_name)` 查不到就 `return` 是同一个判据，只是提前了）。
MAP_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(sorted(MAP_CENTER))}


#: 裁剪范围（Hammer 单位），同样是 config.py 的值。
MAP_RANGE_X = 5000.0
MAP_RANGE_Y = 5000.0
MAP_RANGE_Z = 2000.0

#: 1 Hammer unit = 0.0254 m，和 cs-net 的 SCALE 一致。
HU_TO_METERS = recipes.HU_TO_METERS

# ── 武器词表 ───────────────────────────────────────────────────────────────
#
# ⚠ 每个 csz 里的 `data["weapons"]` 是**按该文件出现过的武器重新编号**的
# （实测 4 个文件分别 34/31/36/35 项），所以 `player["weapon"]` 的下标
# **跨文件不可比**。要跨 demo 训练就必须落到武器名、再映射到这份全局表。

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
WEAPON_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(WEAPON_NAMES)}
N_WEAPONS = len(WEAPON_NAMES)


def weapon_name_to_idx(name: str) -> int:
    """武器名 → 全局下标。未知武器落到 Knife（和 dev 仓库行为一致）。"""
    if name.startswith("weapon_"):
        name = name[7:]
    return WEAPON_TO_IDX.get(name, WEAPON_TO_IDX["Knife"])


#: 队伍编码，和 dev 仓库一致。
TEAM_TO_IDX = {"CT": 0, "T": 1}
TEAM_UNKNOWN = -1

# ── 回合体检阈值 ───────────────────────────────────────────────────────────
#
# 这俩是**兜底**，不是主判据。
#
# 主判据在 store.merge_split_instances()：加载时先把被切片器切碎的 fire/smoke
# 合并回一条（实测 5913 条碎片 -> 917 条真实例），再加一条自检不变式。合并之后
# 实测 89 个回合全部健康，所以下面这两条现在一个都不会触发。
#
# 留着是因为碎片化是个连续谱：切得比 merge_gap 还散的话就拼不起来，那时
# round_health() 靠这两条兜住（见它的 docstring）。

MAX_PLAUSIBLE_INSTANCES = 64
MIN_PLAUSIBLE_MEDIAN_SPAN = 100
