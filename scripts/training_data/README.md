# CSZ → 训练数据：怎么用，以及坑在哪

`data/demos/csz/` 下那 4 个文件是 `demo_extract` 抽出来的 **cs2.demo.v7** 格式。
这份文档说清楚三件事：**文件里到底有什么**、**怎么取出来**、**构造成 torch Dataset 时哪里会踩坑**。

代码分成**框架**和**任务**两半，**你要写的是任务**：`tasks/<任务名>.py`，一个任务一个
文件（它决定一个回合 dump 成什么、怎么读回来）。`prepare.py`（把 csz dump 成
`prepared/<task>/{train,valid,test}/`，按**整个 demo** 划分）和 `dataset.py`
（把某个 split 读成张量）都是框架，**不用改**。

**一份 csz 可以造出任意多个任务的数据集**，各自写自己的目录，互不干扰。

```bash
python -m scripts.training_data --task example          # 体检：csz + 任务 + 数据三层
python -m scripts.training_data.prepare --task example  # 造数据
python -m scripts.training_data --todos                 # 列出还没写的 TODO（秒回，不碰 csz）
```

写自己的任务：

```bash
copy scripts\training_data\tasks\_template.py scripts\training_data\tasks\mytask.py
```

---

## 造好的样本怎么用人眼检查

每个任务一个**独立可跑**的可视化 app（只依赖 numpy，不碰 csz / `demo_extract`）：

```bash
# aim：一条样本 = 257 帧的击杀窗口
python -m scripts.training_data_visualizer.aim     # http://127.0.0.1:5117/

# spatial_prediction：一条样本 = 一帧 + 5 个未来标签
python -m scripts.training_data_visualizer.spatial_prediction   # http://127.0.0.1:5118/
```

**aim**：逐帧看 A / B 各自的第一人称（目标/敌人包围盒、准星轨迹、开枪时间点）、
**当前帧**的 10×10 遮挡矩阵（`obstruction`：通 / 烟 / 墙）和逐帧道具表、逐帧全场玩家表
（`ctx_player` / `ctx_player_mask` / `ctx_player_weapon`）、逐帧炸弹与回合时间
（`global_feat`），以及当前帧 20 列的原值 + 逆变换值。说明见
[`training_data_visualizer/aim/README.md`](../training_data_visualizer/aim/README.md)。

**spatial_prediction**：一条样本是**一帧**（`ctx_player` 没有 T 轴），重点是
**5 个标签对不对** —— `ct_win` / `alive_end` / `can_kill` / `next_death` / `next_kill`。
页面上有标签卡片、俯视雷达、遮挡矩阵和 10×20 原始特征表；`data.py` 还会对同一个文件的
K 行做一组**标签不变式体检**（`ct_win` / `alive_end` 恒定、`next_*` 一旦为 10 不能
变回来、`can_kill` 单调收缩…），违反的会飘在页面顶部。说明见
[`training_data_visualizer/spatial_prediction/README.md`](../training_data_visualizer/spatial_prediction/README.md)。

---

## 0. 文档地图：提取数据的权威说明在别处

这个目录的文档只讲**怎么把它接进 torch**。**`.csz` 本身怎么解析、每个字段什么
含义，权威说明在 `tools/demo_extract/` 下**——那两份写得比这里全得多，别重复造：

| 文档 | 什么时候看 | 关键小节 |
|---|---|---|
| [`tools/demo_extract/README.md`](../../tools/demo_extract/README.md) | 想搞懂文件里有什么、怎么用 | **§3 输出长什么样**（整棵字段树，一页看完）<br>**§4 怎么用**（4.1–4.8，含 v6/v7 新列）<br>§5 火焰 / §6 烟雾（几何怎么还原）<br>**§8 已知取舍，请务必读**<br>§9 和 cs-net v2 的字段对应 |
| [`tools/demo_extract/API.md`](../../tools/demo_extract/API.md) | 查具体函数、写特征 | §1.5–1.7 回合/玩家/事件完整字段表<br>**§1.9 每个字段的量级（实测）** ← 写归一化必看<br>**§4.10 几个必须知道的坑**<br>**§7 速查表**（"我想……用这个"） |
| [`tools/demo_extract/codec.py`](../../tools/demo_extract/codec.py) | 想知道 `.csz` 二进制格式本身 | 模块 docstring 有完整布局（`CSZ5` 容器、tag 表、无损保证） |
| [`tools/demo_extract/voxel.py`](../../tools/demo_extract/voxel.py) | 想还原烟雾体素 | 模块 docstring 有 journal 记录格式、Morton 编码、世界坐标公式 |
| [`tools/demo_extract/recipes.py`](../../tools/demo_extract/recipes.py) | 用访问器 | 每个函数的 docstring 都很详细，含实测数字和警告 |
| 本目录 `README.md` | 接 torch、切样本、踩过的坑 | §3.9 单个局面的可见性、§4 代码结构、§5 八个坑 |

**一句话版本**：`README.md` 是教程，`API.md` 是手册，`recipes.py` 的 docstring 是速查。
本文件只补它们没讲的**训练侧**问题（缓存、切分、跨文件词表、坏回合）。

> 想直接看某个文件里有什么，不用写代码：
> ```bash
> python -m demo_extract --inspect data/demos/csz/xxx.csz
> ```

---

## 1. 你手上这 4 个文件的实际内容

| 文件 | 地图 | 回合 | 采样数 |
|---|---|---|---|
| `2392135-Imperial Academy-vs-MINLATE…m1-mirage.csz` | de_mirage | 20 | 120,614 |
| `2392136-Hashiras-vs-HEROIC…m2-nuke.csz` | de_nuke | 22 | 120,692 |
| `2392136-Hashiras-vs-HEROIC…m3-inferno.csz` | de_inferno | 23 | 121,414 |
| `2392142-MOUZ NXT-vs-zajezdzacze…m1-dust2.csz` | de_dust2 | 24 | 118,744 |
| **合计** | 4 张图 | **89** | **481,464** |

全部是：

* **per-tick（64 Hz）**——`interval=None, stride=None`，相邻采样差 1/64 秒
* **10 个玩家**，每回合固定 10 条记录，索引顺序和 `data["players"]` 一致
* `has_fire_smoke=True` / `has_ground_contact=True` / `has_crouch=True`——**信息最全的一档**
* 每个文件最后一个回合（id = 20/22/23/24）都被跳过了，原因写死在
  `skipped_rounds` 里：`no round_freeze_end, so the round has no time origin`。
  **这是正常的**，不是数据损坏。

> 回合 id 不是 `range(n)`。被跳过的回合**不在文件里**（不是空回合），
> 所以要用 `recipes.round_ids(data)` 查有哪些、用 `round_by_id()` 取，
> 不要拿 `data["rounds"][i]` 当 id 用。

---

## 2. 数据长什么样：三层结构

```
data
├── format    "cs2.demo.v7"
├── meta      map / interval / sampled_ticks / tickrate …
├── players   [{"steamid", "name"} × 10]          ← 玩家身份，索引即 P 轴
├── weapons   {"AK-47": 0, …}                     ← ⚠ 每个文件独立编号，见 §5.3
├── places    {"BombsiteA": 0, …}
├── skipped_rounds [{"id", "reason"}]
└── rounds    [ 每个回合一个 dict ]
    ├── id, teams[10], winner, end_reason, start_tick
    ├── ticks[T]  round_seconds[T]                ← T 的权威来源，别假设 stride
    ├── bomb_planted[T]  bomb_dropped[T]  bomb_position[T]
    ├── missing_samples [[sample_idx, player_idx]]
    ├── players[10]   ← 列式：每个 key 都是长度 T 的 list
    │   x y z | yaw pitch | v vx vy vz | hp armor
    │   helmet defuser alive | weapon place | flash flash_alpha
    │   shots footsteps | on_ground jump land | crouch crouching
    │   inventory[T][变长]  spotted[T][变长]
    ├── events  {kills, damage, bomb, smokes, infernos, grenades, sound}
    ├── fire[实例]   ← 每个 inferno：origin, type_name, first/last_tick, changes[]
    └── smoke[实例]  ← 每个烟：detonation, first/last_tick, changes[]
```

`fire` / `smoke` 存的是**变化快照**，不是逐 tick 全量：

* `fire[i]["changes"][k]` = `{tick, count, pos[16×3], normal[16×3], parent[16×3], burning[16]}`
  —— 一次快照最多 16 个燃烧格，带三维坐标 + 表面法向 + 蔓延树父节点
* `smoke[i]["changes"][k]` = `{tick, size, update, mode, data}` —— 原始 journal 字节，
  `mode` 是 `"append"` 或 `"full"`，要按序折叠才能还原到某 tick 的体素云

这是这个格式最值钱的部分：`demoparser2` **拿不到**火焰格子坐标和烟雾体素内容
（`docs/PLAN.md` §7.1–7.2），只有 `demoinfocs` 这条 side-channel 有。

---

## 3. 常见任务：要拿 X 就这么写

> **这一节是写给 `tasks/<名>.py` 的 `write_round()` 的**——下面每一段都是"造数据时
> 怎么把 csz 里的东西取出来"的原材料。（`read_sample()` 里只读你写出去的文件，
> 不碰 csz。）
>
> 这一节是**照着抄**的。每个函数的完整说明（参数、返回值）在
> [`tools/demo_extract/API.md`](../../tools/demo_extract/API.md) §4，
> 不知道用什么函数就查它 §7 的速查表。
>
> 下面的数字都是**从你 `data/demos/csz/` 第一个文件（de_mirage，回合 0）
> 实测跑出来的**，不是编的——你可以照着核对。

### 3.0 先搞清楚三件事，不然一定拿错

**① 玩家用下标，不用名字。** `data["players"]` 是 10 个 `{steamid, name}`，
顺序固定，索引就是所有张量的 P 轴：

```python
data["players"]            # [{'steamid': ..., 'name': 'tee'}, ...]
data["players"][1]["name"] # 'hyskeee'   ← 下标 1 是这个人
```

本文件实测顺序：`['tee','hyskeee','sk1tt','27aZe','sh1Fu-','POLO','MAGILA-','SBT','ta9z','kragh']`

**② 武器和地名也是下标，而且要反查才拿到名字。** 字典是 `名字 -> 下标`，
所以反查要自己翻转：

```python
idx2name  = {i: n for n, i in data["weapons"].items()}   # 下标 -> 武器名
idx2place = {i: n for n, i in data["places"].items()}    # 下标 -> 地名
```

⚠ **每个文件的编号是独立的**（实测 34/31/36/35 项），跨 demo 训练必须走
`constants.WEAPON_NAMES` 那份全局词表，见 §5.3。

**③ tick 和"采样下标"不是一回事。** `rnd["ticks"]` 是真实 tick（如 7668），
采样下标是它在列表里的位置（0, 1, 2…）。张量按**采样下标**索引，
事件按**真实 tick** 记时间。换算：

```python
i = recipes.sample_index(rnd, tick)   # 真实 tick -> 采样下标；-1 = 不在采样里
t = rnd["ticks"][i]                   # 反过来
```

**④ `store.round(file_idx, round_id)` 的两个参数是两个不同的东西。**
`store.round(0, 0)` 读起来像"把 0 四舍五入"，其实两个 `0` 毫无关系：

| 参数 | 含义 | 取值来源 |
|---|---|---|
| `file_idx` | **第几个文件** | `store.paths` 的下标（按文件名排序） |
| `round_id` | **那个文件里的哪个回合** | `store.round_ids(file_idx)` 里的值 |

```python
store.paths                          # 4 个文件，下标就是 file_idx
# 0 → ...m1-mirage.csz   (de_mirage)
# 1 → ...m2-nuke.csz     (de_nuke)
# 2 → ...m3-inferno.csz  (de_inferno)
# 3 → ...m1-dust2.csz    (de_dust2)

store.round_ids(0)                   # [0, 1, 2, ..., 19]
store.round(0, 0)                    # 文件 0（de_mirage）的第 0 回合
store.round(2, 0)                    # 文件 2（de_inferno）的第 0 回合  ← 完全是另一场
store.round(3, 12)                   # 文件 3（de_dust2）的第 12 回合
```

`store.round(0, 0)` 实测拿到的是：de_mirage、`start_tick=7668`、
`T=3750` 个采样、`winner='T'`。

⚠ **`round_id` 不是列表下标。** 被跳过的回合**根本不在 `rounds` 列表里**，
所以有洞的时候 `data["rounds"][i]` 会错位。你手上 4 个文件恰好都是连续的
0..N-1（跳过的都是最后一个 id，20/22/23/24），所以两种写法看着一样，
但别养成习惯：

```python
store.round(0, 0)                        # ✓ 安全
recipes.round_by_id(store.data(0), 0)    # ✓ 等价，库里的写法
store.data(0)["rounds"][0]               # ✗ 有洞就错位

store.round(0, 20)   # KeyError: round 20 was skipped during extraction:
                     #           no round_freeze_end, so the round has no time origin
```

> 想遍历全部回合用 `for rid in store.round_ids(fi):`，别用 `range(...)`。

---

### 3.1 拿玩家坐标

```python
rnd = store.round(0, 0)                      # 文件 0 的回合 0
p, t = 1, 0                                  # 玩家 1、第 0 个采样点

x = rnd["players"][p]["x"][t]                # 1216.0
y = rnd["players"][p]["y"][t]                # -307.0
z = rnd["players"][p]["z"][t]                # -164.815   ← 脚底！
```

实测这个点是 `tick=7668, 0.00s`（回合刚开始，出生点）：

| 想要 | 怎么拿 | 实测值 |
|---|---|---|
| 脚底坐标 | `x, y, z` | `1216.0, -307.0, -164.815` |
| **眼睛**坐标 | `z + eye_z` | `-164.815 + 64.094 = -100.721` |
| 视角 | `yaw`, `pitch` | `94.139°, 8.349°` |
| 速度 | `vx, vy, vz` | `0.0, 0.0, 0.0`（出生点没动） |
| 水平速度 | `v` | `0.0`（**不是 3D 模长**，见 §5.7） |
| 朝向向量 | `yaw_sincos` | `sin/cos(94.139°)` |

**推荐走 `round_tensors()`**，它把整回合一次算好（含 NaN 插值），切片更快：

```python
tens = store.round_tensors(0, 0)
tens["players"][t, p, 0:3]      # array([1216. , -307. , -164.81], float32)
tens["eye_z"][t, p]             # 64.094
```

**单位是 Hammer unit**：1 HU = 1.905 cm。乘 `HU_TO_METERS`(0.0254) 得米。
**没归一化**——原始 HU 值约 ±2300，喂模型前要自己缩放（见 §3.3 和 §4）。

---

### 3.2 拿玩家装备 / 背包 / 血量

装备存的是**下标列表**，要反查名字：

```python
rnd = store.round(0, 0)
p, t = 1, 0
idx2name = {i: n for n, i in data["weapons"].items()}

inv_idx = rnd["players"][p]["inventory"][t]        # [2, 9, 11, 25]
inv_names = [idx2name[i] for i in inv_idx]         # 名字
```

实测（玩家 `hyskeee`，回合刚开始）：

| 想要 | 怎么拿 | 实测值 |
|---|---|---|
| 背包（本地下标） | `inventory[t]` | `[2, 9, 11, 25]` |
| 背包（名字） | 反查 `idx2name` | `['Butterfly Knife', 'Flashbang', 'Glock-18', 'Smoke Grenade']` |
| **手持武器** | `weapon[t]` | 下标 `2` → `Butterfly Knife`（`-1` = 没武器/死亡/观战） |
| **所在区域** | `place[t]` | 下标 `19` → `TSpawn` |
| 血量 | `hp[t]` | `100`（死人是 0，**不是缺值**） |
| 护甲 | `armor[t]` | `0`（不穿甲就是 0） |
| 头盔 | `helmet[t]` | `False` |
| 拆弹器 | `defuser[t]` | `False` |
| 闪光剩余秒 | `flash[t]` | `0.0`（最多约 5.1 s） |
| 闪光亮度 | `flash_alpha[t]` | `0–255` 字节 |
| 是否活着 | `alive[t]` | `True` |

**要注意的两个坑**：

* `inventory` 是**变长列表**（实测 0–8 项，中位 3；死人是空列表），
  不是定长数组。要喂模型就得自己 pad 或用 multi-hot。
* `weapon` 是**当前手持**，`inventory` 是**身上所有**。两者不是一个东西。

**做模型输入时用 multi-hot**（定长、可直接喂）：

```python
inv = recipes.inventory_multihot(rnd, n_items=len(data["weapons"]))  # (T,10,34)
# 或者用本项目的全局词表版本（跨 demo 可比，推荐）：
inv = tens["inventory"]                    # (T,10,43) bool，下标是 WEAPON_NAMES

g = tens["inventory"][t, p]                # 某个时刻某人的装备
np.flatnonzero(g).tolist()                 # [3, 36, 38, 39]
[WEAPON_NAMES[i] for i in np.flatnonzero(g)]
# ['Glock-18', 'Knife', 'Flashbang', 'Smoke Grenade']
```

> 注意同一个东西在两套词表里下标不同：本地下标 `2` 是 `Butterfly Knife`，
> 全局词表里 `Knife` 是 `36`。**这就是为什么不能直接混用两套下标。**

**把本地下标转成全局下标**（跨 demo 训练必须做，接口是 `store.weapon_lut()`）：

```python
lut = store.weapon_lut(0)                  # 长度 = 本文件武器数（mirage 是 34）
lut[2]                                     # 2 = Butterfly Knife -> 36 = Knife
lut[12]                                    # 12 = High Explosive Grenade -> 37

# 整列转换（COL["weapon"] = 12）：
w = tens["players"][:, :, COL["weapon"]].astype(np.int64)
valid = w >= 0                             # -1 = 没武器/死亡/观战，实测占 20.8% 的 tick
glob = np.full(w.shape, -1, dtype=np.int16)
glob[valid] = lut[w[valid]]
```

配套还有两个查表用的：

```python
store.weapon_names(0)        # 下标 -> 名字，长度 34
store.unmapped_weapons(0)    # 不在全局表里、会被算成 Knife 的名字
```

**刀的收敛是自动的**（你问的那个）：实测 4 个文件里出现 9 种刀
（Butterfly / Flip / Huntsman / Karambit / M9 Bayonet / Skeleton / Stiletto /
Talon / Ursus），加上 `knife_flip` 这类内部名和裸 `Knife`，**全部映射到全局 36 = Knife**。

---

### 3.3 拿整回合的张量（所有玩家 × 所有时刻）

```python
tens = store.round_tensors(0, 0)
tens["players"].shape           # (3750, 10, 20) = (T, 10玩家, 20特征)
```

**记住 20 列的顺序**（`constants.COL` 可以按名字取，别写魔法数字）：

| 下标 | 列 | 下标 | 列 |
|---|---|---|---|
| 0,1,2 | `x, y, z` | 10,11 | `flash, flash_alpha` |
| 3,4 | `yaw, pitch` | 12,13 | `weapon, place` |
| 5,6,7 | `vx, vy, vz` | 14,15 | `shots, footsteps` |
| 8,9 | `hp, armor` | 16,17,18,19 | `on_ground, jump, land, crouch` |

常见切片：

```python
tens["players"][:, p, 0:3]      # (T, 3)      玩家 p 一整回合的坐标轨迹
tens["players"][t, :, 0:3]      # (10, 3)     某一刻 10 个人的坐标
tens["players"][t, p, COL["hp"]]# 标量
```

实测 `t=0` 那一刻 10 个人的坐标（回合刚开始，全在出生点附近）：

```
[[-1656.0, -1976.0,  -267.5],    # 0  tee      CT 出生点
 [ 1216.0,  -307.0,  -164.8],    # 1  hyskeee  T 出生点
 [-1720.0, -1896.0,  -267.6],
 [ 1136.0,   -64.0,  -164.5],
 [ 1136.0,    32.0,  -164.8],
 [-1776.0, -1976.0,  -264.0],
 [-1656.0, -1800.0,  -267.2],
 [-1776.0, -1800.0,  -264.0],
 [ 1136.0,  -256.0,  -164.8],
 [ 1216.0,  -115.0,  -164.0]]    # 9  kragh    T 出生点
```

`tens` 里其它可用的 key（都是整回合长度，自己切 `[a:b]`）：
`alive` `yaw_sincos` `seconds` `ticks` `teams` `inventory` `spotted`
`on_ground` `jump` `land` `crouch` `crouching` `eye_z`。

---

### 3.4 拿某一 tick 的完整画面（含火焰/烟雾几何）

```python
s = recipes.tick_sample(data, 0, rnd["ticks"][1200])
s["players"]              # (10, F)  那一刻所有人
s["on_ground"]            # (10,)    bool
s["crouch"]               # (10,)    0..1
s["eye_height"]           # (10,)
s["fire"]                 # 那一刻在烧的火，含几何
s["smoke"]["grids"]       # (S, 32, 32, 32) bool 占据栅格
```

火焰/烟雾的几何还原（坐标、法向、蔓延树）见
[`tools/demo_extract/README.md`](../../tools/demo_extract/README.md) §5 / §6，
以及本目录 `spatial.py`。⚠ 烟雾的 `solid=True` 是 **101 ms/次**，别逐 tick 调，见 §7。

---

### 3.5 拿事件（击杀/伤害/炸弹）—— ⚠ 必须过滤

```python
rnd = store.round(0, 0)
rnd["events"]["kills"]      # [{t, a, v, w, hs, dmg, ...}]   t = 真实 tick
rnd["events"]["damage"]     # [{t, a, v, hp, w}]
rnd["events"]["grenades"]   # [{t, ty, eid, th, x, y, z}]
rnd["events"]["sound"]      # [{t, ty, p, w, sil}]
rnd["winner"]               # 'T' / 'CT' / ''
rnd["bomb_planted"]         # (T_round,) bool
rnd["bomb_position"]        # (T_round,) [x,y,z] 或 None
```

**别直接遍历 `rnd["events"]["kills"]`！** 它混着不属于这个回合的事件——
**round 0 会把赛前热身的击杀全吃进来**：

| 文件 | round 0 原始击杀 | 其中发生在回合开始前 | 真正在回合内 |
|---|---|---|---|
| de_mirage | 33 | **27** | 5 |
| de_nuke | 52 | **45** | 7 |
| de_inferno | 16 | **11** | 4 |
| de_dust2 | 56 | **49** | 7 |

全文件范围看，**15–33% 的击杀和 8–23% 的伤害事件落在自己回合的采样范围之外**。
不滤的话这些 tick 在采样时间轴上根本不存在，标签会错得很隐蔽
（`sample_index()` 会返回 `-1`，但你不一定检查）。

用 `features.round_events()` / `events_in_window()`，它已经滤掉并给了采样下标：

```python
from scripts.training_data.features import round_events, events_in_window

# 整回合：滤掉不属于这个回合的
for k in round_events(rnd, "kills"):
    k["t"]        # 真实 tick
    k["i"]        # 采样下标，保证 0 <= i < len(rnd["ticks"])
    k["a"], k["v"]  # 攻击者 / 被杀者 的玩家下标
    k["w"], k["hs"] # 武器下标 / 是否爆头

# 只要窗口内的（做标签用）
for k in events_in_window(rnd, ref, "kills"):
    local_i = k["i"] - ref.start     # 窗口内第几个采样
```

实测 round 0 过滤后只剩 5 条：`t=9416, 9446, 9496, 9611, 11322`。

> ⚠ **回合结束的那一枪不在 `round_events()` 里**：它落在 `ticks[-1] + 1`（引擎在死亡
> 那一 tick 才把 `total_rounds_played` 加一，那一帧的采样已经算进下一个回合了）。
> "打死最后一个敌人、回合立刻结束"的回合里，这一枪不补就一条样本都没有 —— 实测
> 78 个回合里有 49 个（63%）是这么结束的。需要它就用
> `tasks/_util.py::late_kills(rnd, n_frames=...)`：它把这类事件映射成下标 `n_frames`
> （= 最后一帧之后），`aim` 和 `spatial_prediction` 都靠它。

> `sound` 事件是例外：实测 4 个文件**100% 落在采样范围内**（2667/2667），
> 滤不滤都一样。但统一走 helper 不会有坏处。

想自己对齐就用 `i = recipes.sample_index(rnd, tick)`，**记得处理 `-1`**。

---

### 3.6 拿"谁看得见谁"

```python
sp = tens["spotted"]        # (T, 10, 10) bool
sp[t, i, j]                 # 时刻 t：**j 能看见 i**   ← 注意方向是反的！
sp[t, :, i]                 # 时刻 t：**i 能看见谁**   ← 通常你要的是这个
sp[t, i, :].sum()           # 有几个人看得见 i
```

⚠ 这是**引擎**写的，只有敌方、被限速在 33 tick 的网格上、平均晚 ~178 ms，而且
**死亡不清零**。要"这一刻到底看不看得见"（含队友、逐采样点、没有延迟）就用 §3.9 的
**`VisOracle`** 现问；`spotted` 只当粗糙参考。

---

### 3.7 切样本

窗口切法自己定（整回合 / 单 tick / 别的都行），要滑动窗口就用库里的：

```python
for rid, a, b, start_tick in recipes.iter_windows(data, length=64, stride=16):
    # a, b 是采样下标；保证不跨回合
    X = store.round_tensors(0, rid)["players"][a:b]      # (64, 10, 20)
```

`iter_windows` 保证**不跨回合边界**——别自己按 tick 切，跨回合的窗口会把上一回合
的死亡状态喂进下一回合，是典型的静默脏数据。长度不足的回合尾部会被丢掉（不是 pad）。

切完怎么塞进 Dataset 见 §4。回合健不健康用 `store.round_health(fi, rid)` 查
（`None` = 健康；现在 89 个回合全是，原因见 §5.2）。

---

### 3.8 特征的语义细节（这几个不看会做错）

| 列 | 说明 |
|---|---|
| `x, y, z` | **脚底**，不是眼睛。眼位 = `z + eye_height()` |
| `crouch` vs `crouching` | 前者是**连续斜坡**（0→1 约 13 tick / 203 ms，用于相机）；后者是 `FL_DUCKING` **阶跃**（碰撞盒 72→54，用于命中盒）。**两者故意不同**，别用 `crouch > 0.5` 代替 |
| `footsteps` | ⚠ **不能当脚步用**。CS2 的 `player_footstep` 事件只在起跳/落地附近触发，实测比真实脚步少约 50 倍。判断移动噪音要用 `velocity` + `on_ground` |
| `spotted` | ⚠ 方向是 `[t, i, j]` = **j 能看见 i**（是转置）。想要"i 能看见谁"用 `sp[:, :, i]`。只有敌方 + 引擎有延迟，要真值用 §3.9 的 `VisOracle` |
| `jump` / `land` | `jump` 来自真实事件；`land` 是从 `on_ground` 推的（CS2 没有落地事件） |
| `weapon` / `place` | 本文件词表的下标，`-1` = 未知。反查见 §3.0 |
| `v` | **水平速度**，不是 3D 模长（`v == hypot(vx,vy)`，见 §5.7） |
| `inventory` | **变长列表**（0–8 项），不是定长数组，要 pad 或 multi-hot |
| `hp` | 死人是 `0`，**不是缺值** |
| `alive` | 死人的位置列是 `0`，**配合 `alive_mask` 用**，别让模型以为他站在地图原点 |

---

### 3.9 只问一个局面：`VisOracle`（A 能不能看见 B）

一帧 90 个有序对、一个回合 3750 帧。但很多时候你只想知道**某一个局面**下谁看得见谁 ——
那就别全算，直接问。

> **这里以前还有一条"离线预计算全表"的路子**（`vis_prepare.py` → `data/vis/*.npz` →
> `store.vis_round()`，L2 变化点压缩）。它和这个是**同一个引擎**、同一份 `visibility.js`，
> 区别只是"算一次存下来"还是"用的时候现算"；因为要的通常是少数局面，现在只留现算这一条，
> 那条路的代码（`vis.py` / `vis_check.py` / `vis_prepare.py` / `js/vis_offline.mjs`）都删了。

```python
from scripts.training_data.vis_oracle import VisOracle, player_state

with VisOracle("de_mirage") as o:            # 加载地图 ~1 s，进程只做一次
    a = player_state(x=100, y=200, z=0, yaw=90, pitch=5)    # 脚底坐标，HU
    b = player_state(x=900, y=200, z=0)
    o.can_see(a, b)                          # ← **A 看得见 B 吗**
    o.detail(a, b)["corners"]                # 每个角落的明细 + reason 位（FOV/地图/烟/玩家）
    o.can_see_many([(a, b), (b, a)])         # 多对一次往返（省 IPC）

    o.load_round(store, fi, rid)             # 或者：把 csz 某个回合灌进去（3–5 MB，0.2 s）
    o.matrix(tick=1234)                      # 那一帧整张 10×10
```

`js/vis_oracle.mjs` 是常驻 node 进程（stdin/stdout，一行一个 JSON 请求），`vis_oracle.py`
是它的 Python 客户端；零件（OBJ 解析 / BVH / 烟体素 / SCALE）在 `js/vis_common.mjs`，
和浏览器里跑的是**同一份**。

**它只在造数据这一侧**：在 `write_round()` 里问，把要用的可见性抄进样本。要点两条 ——
把 oracle 做成**模块级单例**（地图只在变化时加载一次，不是每个回合一次），以及**只问样本
真正要的那几个 tick**。`read_sample()` 不许 import `..store`，所以训练侧永远不碰 node。

**实测成本**（mirage，单线程；这台机器两次测得差 ~2 倍，所以给区间）：

| 事 | 成本 | 说明 |
| --- | --- | --- |
| 加载地图 | **~1.0 s** | 只在起进程时付一次（解析 0.65 s + 建树 0.32 s） |
| 灌一个回合 | 0.1–0.2 s | 3–5 MB 的载荷（首次还要等 csz 解压 ~13 s） |
| 单对 `can_see` | **60–140 µs** | 几何只占 ~8 µs，**其余全是 IPC** |
| 一帧 90 对 `pairs` | **25–45 µs/对** | 一次往返问完，别循环问 |
| 一帧整张 `matrix`，**没烟** | **~1–3 ms** | 纯几何 |
| 一帧整张 `matrix`，**有烟** | **+10–45 ms** | 见下 |

**烟是唯一的大头**（实测分解：`烟 30.89 + 几何 2.52` ms/帧）。机制在 `SmokeVoxelCache`：
它按**烟雾变化点**增量折叠，每条烟每碰到一个新的变化点要 ~10–20 ms（折叠 + 重建体素）。
所以：

* **跳着问会贵**：每 117 帧问一次 = 一次折叠掉中间所有变化点，实测 **30–48 ms/帧**；
  连续问时大部分帧命中缓存，只要 **1–1.3 ms/帧**。
* **问第二次几乎免费**：结果按变化点缓存 —— 把同一批 tick 倒着再问一遍 = **1.9 ms/帧**。
* 但稀疏采样仍然划算：一个回合 3750 帧全算要 4.9 ms/帧 ≈ 18 s，稀疏问 32 帧 ≈
  32 × 30 ms ≈ **1 s/回合**，而且**只为你真的问的那几个 tick 付钱**。

**四个坑**（前三个客户端已经封住，但你得知道）：

1. **观察者**眼高看 `eye_height`，没有就按**连续**的 `crouch` ∈ [0,1] 插值
   （64.0938 − 18·crouch）；**目标**的碰撞盒高度看 `crouching`（**布尔**，72/54 **阶跃**）。
   字段名不对称 → 用 `player_state()` 造，别手写 dict。
2. `pairVisibility()` **自己不查 `alive`** —— 死人照算。`VisOracle` 补上（`detail()` 里 `dead=True`）。
3. `can_see(a, b)` 是「**a 看得见 b**」，和 `.csz` 的 `spotted[目标] ∋ 观察者` 方向相反。
4. **不传 `tick` 就没有烟**：裸状态查询默认只算地图遮挡。要烟就传 `tick`（回合模式）
   或 `smoke=[...]`（裸模式，再给个 `smoke_key`，免得每次重建体素）。

**验收**：`python -m scripts.training_data.vis_oracle --selftest` —— 起进程、把 csz 里两个
真回合灌进去、和 **`oracle_golden.npz` 那份冻结快照逐位对拍**（64 帧 / 274 个可见对），
再验 `pair` 路径和 `matrix` 路径（`visibility.js` 里是两份独立实现）在 270 个方向上答案
完全相同、`alive`/`yaw` 真的生效、单对成本在微秒级。

⚠ **那份快照是"行为快照"，不是真值**：它是当初和那张离线表逐位对拍过（64/64）才冻下来的。
表删了之后，oracle 就没有别的独立参照了（只剩浏览器那侧
`visualizer/tests/test_geo_visibility.mjs`），所以它抓的是"改动带来的不知情行为变化"。
`visibility.js` 有意改了、对不上了，就用 `--freeze` 重冻，并在提交信息里写明为什么。

---

## 4. 代码结构

```
scripts/
├── __init__.py
└── training_data/
    ├── tasks/         ★★★ 你改这里 ★★★  一个任务一个文件（换个任务 = 加个文件）
    │   ├── example.py    完整可跑的示例：一个回合 → 一个 .npz
    │   ├── _template.py  空白模板（带 TODO），复制成 <你的任务名>.py
    │   └── base.py       任务契约：要写哪三个方法、能用什么、坑在哪
    ├── prepare.py     造数据框架 **不用改**：逐文件单遍、处理完就丢、按 demo 分 split、
    │                  --overwrite 重跑；唯一 import demo_extract 的地方
    ├── dataset.py     读数据框架 **不用改**：SampleDataset(data_dir, task=...)
    │                  + to_torch() / INT_KEYS
    ├── constants.py   常量：特征列 COL / 地图中心 / 武器词表 / 体检阈值
    ├── _paths.py      把仓库 tools/ 加进 sys.path（让 demo_extract 可导入）
    ├── store.py       CszStore —— 加载、合并切碎的 fire/smoke、缓存、体检
    │                  （地基，别删；prepare.py 用它）
    ├── features.py    参考实现：窗口切法 + round_events() / events_in_window()
    │                  （里面的字段表、量级、坑照用；切法你不必照抄）
    ├── spatial.py     fire / smoke 空间特征（可选，很贵，默认不用）
    ├── vis_oracle.py  单个局面的可见性：VisOracle.can_see(a, b) / player_state()（§3.9）
    ├── oracle_golden.npz  冻结的行为快照（64 帧，`--selftest` 用它；`--freeze` 重做）
    ├── js/            node 那侧的两个脚本（几何代码一行都没有，全 import app 的）
    │   ├── vis_common.mjs   OBJ→BVH / 烟体素 / SCALE
    │   └── vis_oracle.mjs   常驻进程：stdin/stdout 一行一个 JSON（pair/pairs/matrix/…）
    ├── selftest.py    跑通检查：python -m scripts.training_data --task <名>
    └── __main__.py    python -m scripts.training_data
```

**一份 csz 可以造出任意多个数据集**，一个数据集就是一个任务（`tasks/<名>.py`）：

```bash
python -m scripts.training_data.prepare --task example    # → prepared/example/
python -m scripts.training_data.prepare --task mytask     # → prepared/mytask/
```

任务名只决定"样本长什么样"，**不参与划分**：`demo_split()` 只吃文件名和 `seed`，
所以同一个 `seed` 下两个任务的 train/valid/test 是同一批 demo，val loss 可以跨任务比。
别给不同任务换 `seed`。

职责划分：

* **`tasks/<名>.py`** ★ 你写：一个回合 dump 成什么（`write_round`）、怎么把它读回来
  （`read_sample`）、一共几个样本（`n_samples`）。造和读在**同一个文件**里，
  布局改了必须两边一起改，放在一起才不会漂
* **`prepare.py`** 造数据框架：取数、体检过滤、**样本属于哪个 split**、逐文件单遍、
  `--overwrite` 重跑。**唯一**碰 `store.py` / `demo_extract` 的地方，跑一次就行，不进训练循环
* **`dataset.py`** 读数据框架：glob 文件名，`__len__` / `__getitem__` 转发给任务。
  只认磁盘上的样本文件，不 import `store.py`
* **`store.py`** 不关心你要什么特征，只把原始数据变成"能快速切片的东西"
* **`features.py`** 参考实现，不参与你的调用链；只有 `round_events()` 那套事件过滤
  是真该照用的（不过滤标签就是错的，见 §5.8）

### 为什么 `__len__` 由造数据那一侧定义

样本数不是 csz 目录决定的，是造数据那侧决定的——丢哪些不健康的回合、一个回合切几个
样本，都是那边的选择。如果 `dataset.py` 的参数还是 csz 目录，`__len__` 就只能
**把预处理逻辑再实现一遍去"预判"**。两边逻辑迟早不一致，而且是静默不一致：
`len(ds)` 报 89、磁盘上其实只有 86 个文件，shuffle 到越界才炸。

现在这个"两边"被收进了**同一个任务文件**：`n_samples(samples)` 和
`read_sample(samples, i)` 就在隔壁，对不上时一眼能看到两处定义。

> **`samples` 是样本文件（`.npz`），不是 csz 文件。** 这份契约里最容易搞混的就是
> "文件"指哪一个，所以两侧的参数名刻意分开了：
>
> | 叫什么 | 是什么 | 在哪一侧 | 怎么访问 |
> |---|---|---|---|
> | csz 文件 | 原始 `.csz`（`data/demos/csz/`） | 只在 `write_round`（造） | `fi` 下标 + `store`：`store.paths[fi]`、`store.round_tensors(fi, rid)` —— 任务**永远拿不到它的 `Path`** |
> | 样本文件 | 造出来的 `.npz`（`prepared/<task>/<split>/`） | 只在 `n_samples` / `read_sample`（读） | 形参 `samples`，即 `samples[i]` |
> | 样本 | 逻辑上的第 i 条数据 | `read_sample(samples, i)` | 边界由 `n_samples(samples)` 定 |
>
> 两个参数集合**不重叠**：造的时候没有 `samples`，读的时候没有 `store` / `fi`。
> （以前叫 `files`，两边都能套，已经全部改名。）

| | 决定什么 | 输入 | 依赖 |
|---|---|---|---|
| `tasks/<名>.py` | 样本数、样本内容、dtype、文件名 | 框架传进来的 `CszStore` | `store.py` → `demo_extract`（**只在 `write_round` 里**） |
| `prepare.py` | 写哪些回合、样本进哪个 split、重跑 | csz 目录 + 任务名 | `store.py` → `demo_extract` |
| `dataset.py` | 怎么把文件读成张量 | **某个 split 的文件夹** + 任务名 | 只有 numpy（+ torch） |

附带的好处：数据文件夹能单独搬走，训练侧不需要 `tools/` 在 `sys.path` 上、
也不需要装 `demoparser2`。

> ⚠ **实测：这条现在做不到。** 只要 import 了这个包里的任何东西就会连带
> `demo_extract` —— 包括 `import scripts.training_data.dataset`（实测 `sys.modules`
> 里有 `demo_extract.codec/voxel/recipes/sidechannel/extract`）。原因是 Python 导入
> 子模块前一定先执行父包的 `__init__.py`，而它自己就急切地 `from .store import ...`。
> 想真的断开，得把 `__init__.py` 改成惰性导入（PEP 562 的模块级 `__getattr__`）。
> 在那之前，"训练侧不装 `demoparser2`"这条好处**不成立**。

### 只认 v7，代码里没有版本分支

`tools/demo_extract/` 是个**读不同年代文件的库**，所以它同时提供
`PLAYER_FEATURES`(16 列) / `_V6`(19) / `_V7`(20) 三套列，并对老文件抛 KeyError。
那是库的契约，它的测试专门盯着这个行为（`test_roundtrip.py`：*"a pre-v6 file
raises instead of guessing, not fake"*）。

**但训练侧不需要这些分支**——你手上 4 个文件全是 v7。所以这里：

* `constants.FEATURES = recipes.PLAYER_FEATURES_V7`，只有这一套
* `constants.REQUIRED_FORMAT = "cs2.demo.v7"`
* `CszStore.data()` 加载时校验格式，**不是 v7 直接抛错**：

  ```
  格式是 'cs2.demo.v5'，需要 'cs2.demo.v7'。
    cs2.demo.v7 才有 on_ground / jump / land / crouch 这几列。
    老文件重新抽一遍即可：
      python -m demo_extract <match.dem> -o <match.csz>
  ```

* `round_tensors()` 无条件产出全部 14 个 key，不做 `if key in ...` 判断
* `features.py` 里没有"缺列就补 0"的兜底

> 这条**不是洁癖，是防一个真实的静默 bug**。之前的版本里有个 `_opt()` 会在列缺失时
> 补 0——那样 `on_ground` / `crouch` 会整列变成常量 0，模型照常训练、loss 照常下降，
> 你要到很久以后才发现那 4 列根本没起作用。**缺列就该在加载时炸掉。**
>
> 顺带提醒：`recipes.player_tensor(rnd)` 的**默认参数是 16 列的老集合**
> （为了向后兼容），所以取值时必须显式传 `FEATURES`，别用默认值。

### 你要写的部分①：`tasks/<任务名>.py`（造数据那一半）

一个任务 = 一个文件。契约（`tasks/base.py`）要求三个方法：

| 方法 | 干什么 | 规矩 |
|---|---|---|
| `n_samples(samples)` | 一共多少个样本 | **必须便宜**：只看文件名，不许开样本文件 |
| `write_round(store, *, fi, rid, split, path)` | 把一个回合写成一个样本文件 | 造数据侧唯一的出口；算完立刻写、立刻丢 |
| `read_sample(samples, i)` | 把第 i 个样本读回来 | 返回 `dict[str, ndarray]` 就自动过 `to_torch()` |

`sample_name()`（文件名规则）和 `ext` / `pattern` 有默认实现，一般不用动。
**完整可跑的例子是 `tasks/example.py`，照着抄**；空白骨架（带 TODO）在
`tasks/_template.py`。

```bash
copy tasks\_template.py tasks\mytask.py     # 文件名就是 --task 的值
python -m scripts.training_data --todos     # 列出还没写的 TODO（秒回）
```

```python
# tasks/mytask.py —— 一个回合一个 npz，只装 4 个 key
class MyTask(Task):
    ext = ".npz"
    desc = "一句话说清这个任务要什么"
    KEYS = ("players", "alive", "teams", "ticks")

    def n_samples(self, samples):               # ① 一共几个样本（samples = .npz 文件）
        return len(samples)

    def write_round(self, store, *, fi, rid, split, path):   # ② 落盘（fi = csz 下标）
        tensors = store.round_tensors(fi, rid)
        np.savez(path, **{k: tensors[k] for k in KEYS})

    def read_sample(self, samples, i):          # ③ 读回来
        with np.load(samples[i]) as z:
            return {k: z[k] for k in z.files}
```

### 一个 csz 出几个样本 / 也可以是 0 个

框架不预设"一个 csz 几个样本"——它只是**对每个回合调一次 `write_round`**，
每个回合写几个文件、乃至不写，都是你的实现决定的：

| 你要的 | 怎么写 | 支持 |
|---|---|---|
| 一个回合一个样本 | `n_samples = len(samples)`，一回合存一个文件 | ✅ `tasks/example.py` 就是（实测一个 csz 出 20～24 个） |
| 一个回合 K 个窗口 | 造的时候**仍一回合一个文件**，读的时候按 `n_samples = len(samples) * K`、`samples[i // K]` + `i % K` 拆 | ✅ 推荐：滑窗、预测未来 N tick 都这么做 |
| **一个回合 K 个文件** | `write_round` 里自己派生兄弟文件名，`n_samples` 用默认的 `len(samples)` | ✅ 见下 |
| 0 个 | 什么都不写；整张图的回合全被体检过滤掉就是这种 | ✅ 该 csz 的样本数记 0 |
| 整个 split 0 个 | `--valid 0`，或分到这张图的样本为空 | ✅ 空目录照样能构造，`len(ds) == 0`（见 §4） |

**一个回合写 K 个文件**：`write_round` 每次只拿到**一个** `path`，但没有任何东西限制你
写几个 —— 兄弟名字自己派生就行：

```python
def write_round(self, store, *, fi, rid, split, path):
    tensors = store.round_tensors(fi, rid)
    for k in range(K):
        p = path.with_name(f"{path.stem}_p{k}{path.suffix}")
        np.savez(p, ...)          # 框架传进来的 path 本身不写也行，没人检查它

def n_samples(self, samples):     # 默认实现就是对的
    return len(samples)
```

为什么这是安全的：框架**不数 `write_round` 被调了几次**，它只数磁盘上的样本文件
（`prepare.py` 最后 glob 一遍报出来，`dataset.py` 的 `len(ds)` 用同一套 glob，两边永远
一致）。实测一个回合写 3 个文件：20 个回合 → 磁盘 60 个文件 → `prepare` 报 `train=60`、
`len(ds) == 60`，`ds[0]`/`ds[1]`/`ds[2]` 取到同一个回合的三个不同窗口。

三条要注意的：

* **文件名唯一性由你负责**：`sample_name()` 只有 `(fi, rid)` 两个维度，算不出 K 个名字。
* **`n_samples` 别乱改**：默认 `len(samples)` 就是对的，写成 `len(samples) // K` 反而错。
* 想让**框架**驱动这个循环（`write_round` 每个分片被调一次、名字统一由 `sample_name`
  给）才需要加 `part: int` 参数 + `n_parts(store, *, fi, rid)`；自己派生名字则一行框架
  代码都不用动。

> 还是那句：**读侧切窗口能解决的事，不要搬到写侧**——多一个文件就多一份磁盘 I/O，
> 而且每个 epoch 都要多读一遍。

> 0 个样本这一条对 `SampleDataset` 有个尾巴：`len(ds) == 0` 时
> `DataLoader(ds, shuffle=True)` 会报 `num_samples=0`。训练脚本里要
> `if len(ds) == 0: 跳过这个 split`。

#### 任务文件里除了 `Task` 子类还能写什么

**什么都能写** —— 它就是个普通 Python 模块。框架只从里面拿两样东西：模块末尾的
`TASK = <你的类>()`（必须是 `Task` 子类）和文件名（= 任务名）。所以模块级常量、
辅助函数、辅助类、自己的共用模块都可以：

```python
# tasks/mytask.py
import math
from pathlib import Path

import numpy as np

from ._util import rel_xy          # 自己写的共用模块（下划线开头 → 不是任务）
from .base import Task

KEYS = ("players", "alive")        # example.py 就是这么写的
SCALE = 1.0 / 64

def _pick(store, fi, rid):         # 只给这个任务用的辅助函数
    t = store.round_tensors(fi, rid)
    return rel_xy(t["players"]), t["alive"]

class MyTask(Task):
    ...

TASK = MyTask()                    # ← 必须有，且只能有一个
```

实测过的三条：

* **多个任务共用的代码放 `tasks/_util.py`。** 下划线开头 → `task_files()` 跳过它，
  不会被当成任务、也不会出现在 `--help` / `--todos` 里（`base.py` 同理）。
* **模块级代码在每个进程里各执行一次。** 同一个进程 import 两次只执行一次，换个进程
  再来一次（实测：连续 3 个进程 → 计数 1 / 2 / 3）。Windows 的 DataLoader 是 spawn，
  **每个 worker 都是新进程** → 你的模块级代码在 N 个 worker 里各跑一遍。所以模块级
  只放常量和函数定义；大查表写成函数里的惰性缓存；别在模块级 `print` 或写文件。
* **`store` 不能在模块级 import**（它是造数据侧的东西）。它只是 `write_round` 的第一个
  参数；要写类型标注就用字符串或 `if TYPE_CHECKING:`（`base.py` 自己就是这么干的）。
* **可以在任务文件里写 `if __name__ == "__main__":` 冒烟测试**（很推荐）。见下一节。

#### 任务自带的冒烟测试（**每个任务各写各的**）

冒烟测试写在**任务文件末尾的 `if __name__ == "__main__":`** 里，不在别处：

```bash
python -m scripts.training_data.tasks.example      # 在仓库根目录执行，必须 -m
```

框架**不提供**共用的冒烟测试 —— 每个任务要检查什么、要不要画图存盘，只有你自己知道
（有的任务要画可视化图，有的要核对标签分布，有的只想打印几个张量）。共用的只有
`build(limit=N)` 和 `SampleDataset` 两个入口，其余你自己写。

`example.py` 末尾是一份完整可跑的例子，`_template.py` 末尾是留了 TODO 的骨架，骨架
长这样：

```python
if __name__ == "__main__":
    import shutil

    from .._paths import REPO_ROOT
    from ..dataset import SampleDataset
    from ..prepare import SPLITS, build

    TASK = Path(__file__).stem          # 从文件名取任务名，复制到别的任务不用改
    N_CSZ = 2                           # 1 个约 13 s
    CSZ_DIR = REPO_ROOT / "data" / "demos" / "csz"
    OUT = REPO_ROOT / "tmp" / "training_data" / TASK

    shutil.rmtree(OUT, ignore_errors=True)          # 每次从零开始，不用手动清
    counts = build(CSZ_DIR, OUT, task=TASK, limit=N_CSZ)
    print(f"[冒烟] 造出来 {counts}")

    for split in SPLITS:
        ds = SampleDataset(OUT / split, task=TASK)
        print(f"[冒烟] {split:<5} len={len(ds)}")
        for i in range(len(ds)):
            s = ds[i]
            assert ...                  # ★ 你这个任务该满足的东西 ★
```

实测输出（`example.py`，2 个 csz，约 26 s）：

```
[冒烟] task=example
[冒烟] csz =D:\project\cs-net-v5\data\demos\csz
[冒烟] 产出=D:\project\cs-net-v5\tmp\training_data\example
[   1/2] 2392135-...mirage.csz  train  rounds=20  体检滤掉 0    12.9s  累计 20 回合
[   2/2] 2392136-...nuke.csz    train  rounds=22  体检滤掉 0    15.3s  累计 42 回合
[冒烟] 造出来 {'train': 42, 'valid': 0, 'test': 0}
[冒烟] train len=42
[冒烟]   de_mirage_f0_r0.npz: players(3750, 10, 20)  alive(3750, 10)  teams(10,)  ticks(3750,)
[冒烟] valid len=0
[冒烟] test  len=0
[冒烟] ✅ 通过
```

几点：

* **`tmp/training_data/<任务>/` 每次被整体删掉重建**（实测：上次 1 个 csz 的 20 个文件 +
  一个手塞的 `垃圾文件.txt`，跑完变成 42 个文件、垃圾文件没了）。`tmp/` 已在
  `.gitignore` 里。
* **import 全写在 `if` 里面**。任务模块会被 DataLoader 的每个 worker import，写在文件
  顶部会让每个 worker 都连带 import `..prepare`（`CszStore` / `demo_extract`）。
* **必须用 `-m` 跑**。直接 `python scripts/training_data/tasks/example.py` 会炸在
  `ImportError: attempted relative import with no known parent package`（实测）。
* 断言失败会带**文件名和下标**（`train[0] de_mirage_f0_r0.npz: ...`）并返回 1，所以
  能直接定位到是哪个回合的问题。
* 画图要自己 `pip install matplotlib`（本机的 conda 环境里还没装）；服务器上记得
  `matplotlib.use("Agg")`，`example.py` 末尾有注释掉的示例。
* 想跑全量就别用冒烟测试，直接跑 `prepare.py`（4 个 csz 约 50 s，正式的全量约 30 h）。

`write_round` 里能用的东西（都在传进来的 `store` 上，不用 import 任何东西）：

| | |
|---|---|
| `store.round_tensors(fi, rid)` | 一个回合预计算好的 14 个张量（首选，最快），键见 §3.3 |
| `store.round(fi, rid)` | 原始回合 dict：`events` / `kills` / `fire` / `smoke` / `winner` … |
| `store.map_name(fi)` | `'de_mirage'` |
| `store.weapon_lut(fi)` / `weapon_names(fi)` | 武器下标 → 全局下标 / 名字（§5.3 的坑） |

§3 那一整节"要拿 X 就这么写"就是给你写 `write_round` 用的原材料。

跑一次：

```bash
python -m scripts.training_data.prepare --task mytask
# -> task=mytask  train=42  valid=23  test=24  -> data/demos/prepared/mytask
```

进度用 **tqdm**，一个 csz 一格；`--workers 4`（默认）并行跑 4 个 csz：

```
csz:  43%|████▎     | 3/7 [01:12<01:36, 20.0s/csz]
2392135-Impe...-m1-mirage.csz  train  rounds=20  体检滤掉 0  38.1s  累计 20 回合
```

* 进度条走 **stderr**，每个 csz 跑完那行总结走 **stdout** —— `> 日志.txt` 留下的是干净的
  文件级日志；stderr 被重定向时 tqdm 会自动关掉（不往日志里刷控制字符）。
* `--workers 1` 回到单进程（`--limit 1` 时也直接跑在父进程里，栈更干净）。
* 没装 tqdm 也能跑，只是没有进度条（`import` 处有兜底）。

签名里的 `--train` / `--valid` / `--test` / `--seed` 是**划分要用的输入**——`build()` 只替你
校验"三个比例加起来必须等于 1"（不合法抛 `ValueError`，`main()` 会打印成 `[!] ...`
而不是甩栈）。

产出结构（`--out-dir` 默认就是 `prepared/<task>`，**一个任务一个目录**）：

```
data/demos/prepared/mytask/
├── task.json   {"task": "mytask", "seed": 0, ...}   ← 记着是谁造的，读的时候会核对
├── train/      de_mirage_f0_r3.npz  ...
├── valid/      ...
└── test/       ...
```

取数（load 哪个文件、有哪些回合）、体检过滤、划分**都已经是框架的事了**。
`prepare.py` 里那个循环长这样，你不需要改它：

```python
for fi in range(n_files):                      # 一个 csz 一个 csz 地过
    split = demo_split(store.paths[fi].name, seed=seed, ...)   # 整张图进同一个 split
    ids = store.round_ids(fi)                  # ← 到这才 load 第 fi 个文件
    for rid in ids:
        if store.round_health(fi, rid) is not None:
            continue                           # 不合格的回合不写进磁盘
        path = out_dir / split / t.sample_name(store, fi=fi, rid=rid)
        t.write_round(store, fi=fi, rid=rid, split=split, path=path)   # ★ 你的代码
    store.clear_cache()                        # ← 丢掉这个文件的 ~1 GB
```

**那行 `continue` 现在一个都不会滤掉**（89 个回合全健康，见 §5.2），但它必须留着：
过滤是个便宜且必要的习惯。哪天你换了新 demo、切片器真出了修不好的碎，你不希望脏
标签被**永久写进磁盘** —— 那时连"原始 csz 还在，重抽一遍"这条退路都没了。

#### 先跑一小批：`--limit`（也是冒烟测试偷懒用的）

```bash
python -m scripts.training_data.prepare --task mytask --limit 2 --out-dir /tmp/看看
```

只处理**排序后的前 N 个** csz。一个 csz 要 11–15 s、全量 30 小时，先跑 2 个看看产出
对不对很划算。划分只跟文件名有关，所以**前 N 个的 split 和全量跑时完全一致**（不是
随机抽样）。产物里的 `task.json` 会同时记 `n_files`（目录里一共几个）和 `files_used`
（这次跑了几个），免得看的人以为样本丢了。

任务文件末尾的冒烟测试就是 `build(..., limit=2)` + 读回来，见 §4。

#### 只用一部分 csz：`--count` / `--map-balance`

全量跑一次是 30 小时 + ~1.9 TB。想先抽一小批把 pipeline 和 loss 曲线跑通：

```bash
# 随机抽 8 个 csz（按 --seed 抽，同 seed + 同目录 = 同一批，可复现）
python -m scripts.training_data.prepare --task mytask --count 8 --out-dir /tmp/小批

# 8 张地图每张各 8 个（count / 地图数 = 64/8，向下取整），不够的随机补
python -m scripts.training_data.prepare --task mytask --count 64 --map-balance
```

* `--count N`：只用 N 个 csz。`N` 不小于候选数时**不生效**（等于全用）。
* `--map-balance`：按地图均分名额，**只在 `count < 候选数` 时生效**。目录里有 K 张图
  → 每张图先拿 `count // K` 个；某张图不够就全拿；还差的名额从没被选中的文件里随机补。
  ⚠ `count < K` 时配额是 0，平衡退化成纯随机（开跑时会打 `[!]` 提醒）。
* `--maps`：哪些地图的 csz 才参与，**默认只跑 `constants.MAP_TO_IDX` 里的那 8 张**。
  判据就是 `map_name in MAP_TO_IDX`，和 `tasks/aim.py` 里 `MAP_TO_IDX.get(map_name)`
  查不到就 `return` 是同一个。真实目录里混着 `de_Ancient` / `de_cache_d` /
  `de_cbble_d` / `de_inferno_d_prefab` / `nip_aim_map` 这类图（实测一个 15712 个 csz
  的目录里有 16 个、分属 4 张图），它们的坐标对不上 `MAP_CENTER`
  （`normalize_xyz()` 静默失真）、`aim` 任务还会**一条样本都不产出** —— 所以默认
  **整个 csz 都不参与、不占名额**。名字精确比较：`de_Ancient` 不是 `de_ancient`，
  照样筛掉。
  - `--maps all`：不筛（所有出现过的地图都要）；
  - `--maps de_anubis,de_mirage`：只跑这几张。
* 它和 `--limit` 是两回事，**不能同时给**：`--limit` 是"排序后的前 N 个"（不看内容，
  冒烟测试用），`--count` 是"随机 N 个"。
* 划分仍然只跟**文件名**有关（`demo_split()`），所以抽样只决定"哪些 demo 在里面"，
  不会改变某个文件进哪个 split。

为什么按地图平衡：爬下来的比赛里地图很不均匀（热门图比冷门图多几倍），纯随机抽
`count` 个很容易抽出一个"mirage 占 40%"的数据集，val loss 就只反映热门图。

⚠ 这几条要先知道"每个 csz 是哪张图"，而地图名在 `meta` 里 —— 但**没有**为此把 csz
全量 load 一遍（那会把 30 小时变成 60 小时），走的是 `store.map_name_fast()`：
只解文件头的 `format` + `meta`，实测 **6 ms/个**（`store.peek_meta()`）。

开跑前会把**每张图有几个 csz 参与计算**印出来（噪声单独列出来）：

```
[地图] 8 张图参与（另有 4 张非标准图当噪声筛掉）；参与计算 2048/15696 个候选 csz（count=2048，按地图平衡），目录里一共 15712 个
       de_ancient     256 / 2521
       de_anubis      219 / 1310
       ...
[!] 另有 4 张非标准地图共 16 个 csz 当噪声跳过（要全要：--maps all；只挑几张：--maps de_anubis,de_mirage）：
       nip_aim_map             13
       de_cache_d               1
       ...
```

`task.json` 里也记着（`maps` = 每张图 `used`/`available`，`maps_excluded` = 被当噪声
筛掉的，`map_filter` / `count` / `map_balance` 原样），以及 **train / valid / test 各自
用了哪些 csz**（`splits.<split>.csz` 文件名列表 + `n_csz`）和各自的最终样本数
`n_samples` —— 拿到产物就能复盘"这份数据是哪几个 csz 造的"，不用去猜
`de_mirage_f3_r7.npz` 里的 `f3` 是哪个文件。

#### 重跑：`--overwrite`，没有断点续跑

**这个框架没有 `--resume`。** 目标目录里有上次的样本就拒绝开工，要重跑就明说：

```bash
python -m scripts.training_data.prepare --task mytask --overwrite    # 删掉重写
python -m scripts.training_data.prepare --task mytask --out-dir <新目录>   # 推荐
```

`--overwrite` 只删**这个工具自己认识的东西**：`train/` `valid/` `test/` 和 `task.json`。
**别的文件和目录一律不碰** —— 实测目录里混着 `notes.txt` 和 `extra_dir/keep.bin` 时，
清完之后它们原样还在。所以把 `--out-dir` 误写成父目录 `prepared/` 也只会删到那个
目录下的 `train/valid/test/task.json`，碰不到各任务自己的子目录；而如果那个目录的
`task.json` 写的是别的任务，检查会先拒绝。

⚠ 代价知情：**跑 30 小时的任务打断了就是从零重来**。实测解析 csz 占整个任务
**95% 以上**（4 个文件 46.4 s，其中写 89 个样本只占 1–2 s），所以重来的成本几乎就是
一次全量重跑。这是 2026-09 的明确取舍：不为"跳过已完成文件"维护一套断点标记
（它得记 csz 名、任务名、任务代码指纹、划分配置、样本数，还要处理"样本被手工删了
但标记还在"这类对不上的情况）。真要能续跑，按文件切片、一个进程一个 `--out-dir`
是更省事的办法（划分只看文件名，多进程各算各的）。

**别吞单文件解析错误**：哪个 csz 坏了就直接崩，最后一行进度就是崩溃点 —— 修好那个
文件再整体重跑。

#### 为什么是"逐文件单遍"（这条是硬约束，不是风格）

csz 是**一个 zstd 帧包住整个对象图**（`codec.loads()` 先 `read_bytes()` 整文件、
整帧解压、再解出整个 dict）。所以：

* **想知道一个文件里有哪些回合，就必须先把它整个 load 进内存。** 没有"只读清单"
  这种便宜路径 —— 所以"第一遍只数、第二遍再处理"等于把整个目录解析两遍。
* load 之后是 **Python 对象图**，膨胀极凶：实测 4 个 csz 一共 **44.1 MB**，全部
  load 完常驻 **4233 MB** —— **×96**。
* 于是"先把所有回合列成清单再统一处理"（两段式）在 **100 GB** 上是 **9.6 TB 常驻**，
  必然 OOM。

单遍跑则是恒定内存：

| | 峰值常驻 |
|---|---|
| 两段式（先列清单再处理） | 4 个文件 **4717 MB**，线性涨；100 GB → 9.6 TB |
| **逐文件 + `clear_cache()`** | **1674 MB**，4 个文件轮流跑**不涨** |

`clear_cache()` 实测把 RSS 从 1517 打回 534 MB（基线 484），下一个文件重新用这块内存。
所以峰值只取决于**最大的那一个文件**，和 csz 总量无关。

> 唯一的小代价：`merge_stats()` 之后只反映最后一个文件（它跟着 `_data` 一起被清）。

这条约束是按**进程**算的：`--workers N` 就是 N 个进程各自"单遍"，互不通信（划分只看
文件名，谁算都一样）。所以内存 ≈ **N × 1.7 GB**，`--workers 4` 约 7 GB。本机实测（32 核、
4 个 csz）：`--workers 1` 约 152 s、`2` 107 s、`4` **72 s**（≈2.1×）。再往上加收益递减 ——
瓶颈是 `codec.load()` 造几百万个 Python float 时的内存带宽，不是核数（把 `NUMBA_NUM_THREADS`
压到 8 试过，没有区别：渲染只占每文件的 ~5 s/38 s）。

#### 划分：一个 demo（= 一个 csz = 一张地图）整场进同一个 split

划分落在**文件**上：一个 csz 里所有回合必然进同一个 split，不会被拆到两边。哈希算在
**文件名**上：

```python
split = demo_split(store.paths[fi].name, seed=seed, train=train, valid=valid)
```

* **不能用 `file_idx`**：往 csz 目录里加文件会让下标整体平移，已分好的 demo 会换
  split，两次实验的 val loss 就没法比。**文件名是稳定的**。
* **不能用回合**：同一个 demo 的相邻回合共享走位习惯、经济、对手打法，按回合随机
  划分会让 valid/test 的 loss **比真实泛化能力好看，而且不报任何错** —— 这种偏差
  最难查，因为它只让数字变好。按文件划分从根上没有这个问题。

| | 按回合随机（老方案） | 按 demo（一个 csz）划分（现在） |
|---|---|---|
| 划分粒度 | 一个回合 | **一张地图（一个文件）** |
| train 里出现过的回合 | 也会出现在 valid/test | **绝不**出现在 valid/test |
| 4 个文件能切出的比例 | 0.9 / 0.05 / 0.05 能命中 | 只能整张给，`2 : 1 : 1` |
| val/test 数字回答的问题 | "我代码写对了吗" | "模型泛化好不好" |

**现在只有 4 个 demo，`seed=0` 的实测结果是：**

```
fi=0  2392135-...-m1-mirage.csz    20 回合  -> train
fi=1  2392136-...-m2-nuke.csz      22 回合  -> train
fi=2  2392136-...-m3-inferno.csz   23 回合  -> valid
fi=3  2392142-...-m1-dust2.csz     24 回合  -> test
                                 -> {"train": 42, "valid": 23, "test": 24}
```

**这次三个 split 都非空**，但那是运气 —— 4 个文件 × 0.05 ≈ 0.2，本来大概率是空的。
换个 `--seed` 就可能变空，**那是预期行为，不是 bug**（`--seed 2 / 3 / 42` 实测就是
全部进 train）。现在的目标是跑通 pipeline；等 csz 攒到几千个再在服务器上重跑，
比例自然就准了。

⚠ **已知取舍：粒度就是一张地图。** 上面 fi=1（nuke）和 fi=2（inferno）**来自同一场
比赛**（`2392136`），现在被分到了 train 和 valid —— 同一批选手、同一套打法跨了 split。
这是"一张图一个单位"的必然代价。想按比赛绑在一起，把哈希输入换成比赛 id：

```python
split = demo_split(store.paths[fi].name.split("-", 1)[0], seed=seed, ...)
```

那样"一张图一个单位"就变成"一场比赛一个单位"，代价是粒度更粗（这 4 个文件只有
3 场比赛，切出来更容易出现空 split）。

> 换回按回合划分：哈希输入改成 `f"{name}:{rid}"` —— 但那样 leakage 就回来了。

#### 为什么不用 `rng.permutation` 按比例切

因为那要求**先知道总样本数**，而知道总数就要求先把所有文件 load 一遍（见上）。
而且 `rng.permutation` 是**全局排列**，没法多进程分片跑 —— 单进程在 100 GB 上是
**~28 小时**（`codec.load()` 实测 11–15 s × 9300 个文件）。

哈希方案两个都能做到：

```python
digest = hashlib.blake2b(f"{seed}:{group}".encode(), digest_size=8).digest()
u = int.from_bytes(digest, "big") / 2**64      # 稳定映射到 [0, 1)
```

* 不需要知道总数 → 单遍流式
* 纯函数 → 8 个进程各管一批文件、各算各的，写同一个 `out_dir`
* 换 `seed` 就换一套划分；比例偏差在 20 万回合下 < 0.5%

### 你要写的部分②：`SampleDataset` 怎么用（读数据那一半）

框架已经写好了，你要写的只有 `tasks/<名>.py` 里的 `read_sample()`。用法：

```python
from scripts.training_data import SampleDataset

ds_train = SampleDataset("data/demos/prepared/mytask/train", task="mytask")
# 构造不读样本，只 glob 文件名 + 问一次 n_samples(samples) → 瞬时

len(ds_train)     # -> task.n_samples(ds_train.samples)
ds_train[0]       # -> to_torch(task.read_sample(ds_train.samples, 0))
```

`ds_train.samples` 是**样本文件（`.npz`）**的列表，不是 csz 文件——这个类的参数
`data_dir` 也永远是造好的数据文件夹，csz 只在 `prepare.py` 那侧出现。

参数是**某个 split 的子文件夹**（不是 csz 目录，也不是 `prepared` /
`prepared/mytask` 根目录）。传错会报"没有样本文件"——`glob` 不递归、只看这一层，
报错信息里会把子文件夹列出来提醒你。

例外是**真空的 split**：整个 split 一张图都没分到（4 个 demo 按 0.9/0.05/0.05 切
就会出现），或者分到的图所有回合都被体检过滤了。这种目录旁边有对得上的
`task.json`，所以构造**不报错**，`len(ds) == 0`：

```python
if len(ds_valid) == 0:          # 必须写这一句
    ...                          # 否则 DataLoader(shuffle=True) 报 num_samples=0
```

`task=` 是**必填**的，而且会核对：造数据时在 `out_dir/task.json` 里记了任务名，
传错直接报错。理由很实际 —— 换个任务，样本文件的布局就变了（`read_sample` 是两个
不同函数），**读错不会报错，只会静静地读出别的字段**：

```
data/demos/prepared/example/train 是任务 'example' 造出来的，你却按 'mytask' 去读。
```

`read_sample` 返回 `dict[str, ndarray]` 就够 —— `__getitem__` 会自动过一遍
`to_torch()`：`*_mask`/bool → bool，`INT_KEYS` 里或整型 → long，其余 → float32。
返回 tuple / 自定义对象就原样交给 collate。

**文件名是你和 `prepare.py` 之间的唯一契约**：默认规则是
`{地图}_f{fi}_r{rid}{ext}`，要加字段（tick 范围、样本权重）就覆盖
`Task.sample_name()`；需要一份完整的清单就在 `write_round` 时写个 `index.csv`，
`n_samples` 改成读它 —— 但**只许读小元数据，不许读样本本身**。

`n_samples()` 和 `read_sample()` 必须对得上，否则要 shuffle 到越界才报错。好在这两个
方法就在同一个文件里，改一个的时候另一个就在隔壁。

三条别踩的（详见 §5）：

* `n_samples()` 里只许 glob 文件名 + 读很小的元数据，别把样本读进内存（§5.5）
* 别在 `Task` 实例上挂大对象：Windows 的 DataLoader 用 spawn，会把 dataset 整个
  pickle 给每个 worker
* 用事件做标签必须过滤，原始 `rnd["events"]["kills"]` 里混着别的回合的事件（§5.8）
  —— 这件事发生在 `write_round` 那侧，别在 `read_sample` 里补救

### 参考实现：`features.py`（不参与你的调用链）

`features.py` 是"一个窗口 = 一个样本"那套留下的参考，签名带 `WindowRef`：
`build_window()` 给输入、`build_targets()` 给标签。切法你不必照抄，但里面两样值得照用：

* `round_events(rnd, kind)` / `events_in_window(rnd, ref, kind)` —— 事件过滤，**必用**
* 各处的字段表、实测量级（`API.md` §1.9）、坑

想按窗口取数就自己拼一个 `WindowRef(file_idx, round_id, start, stop, start_tick)`
（就是五个整数）传进去，不需要任何额外设施。

### 完整用法（`example` 任务在本机实测跑通）

```bash
# ① 造数据（跑一次；想重跑就 --overwrite，或换个 --out-dir）
python -m scripts.training_data.prepare --task example
# [OK] task=example  train=42  valid=23  test=24  -> data/demos/prepared/example

# ② 体检（csz + 任务 + 数据三层一起看）
python -m scripts.training_data --task example
```

```python
# ③ 训练时读
from torch.utils.data import DataLoader
from scripts.training_data import SampleDataset

ds_train = SampleDataset("data/demos/prepared/example/train", task="example")
ds_valid = SampleDataset("data/demos/prepared/example/valid", task="example")

loader = DataLoader(
    ds_train,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

写自己的任务：

```bash
copy scripts\training_data\tasks\_template.py scripts\training_data\tasks\mytask.py
python -m scripts.training_data --todos          # 列出 mytask.py 里还没写的 TODO
```

任务契约在 `tasks/base.py`，完整例子在 `tasks/example.py`；`prepare.py` /
`dataset.py` 是框架，不用改。

读磁盘样本时 `num_workers` 不用再迁就"文件数"了——瓶颈从解码（§5.1）变成 IO，
4–8 都可以，自己试。`persistent_workers=True` 省掉每个 epoch 重启 worker 的开销。

`__getitem__` 返回 `dict[str, Tensor]` 的话，torch 的默认 collate 会直接把同名字段
stack 成 batch，不用自己写 collate。

---

## 5. 坑：实测出来的 8 条

### 5.1 `codec.load()` 要 11–15 秒 —— 绝对不能放 `__getitem__`

实测（本机）：

| 操作 | 耗时 / 体积 |
|---|---|
| `codec.load()` 一个 csz | **10.7 – 14.9 s** |
| 把某回合转成 `(T,10,20)` float32 | ~50 ms |
| 一个回合 14 个 key 全算出来 | 平均 **9.2 MB** |
| 4 个 demo 全部 89 回合 | **817 MB** |

瓶颈不是解压，是把 ~240 万个 float 解成 Python `float` 对象。所以：

* **一个进程只 load 一次**（`CszStore` 已经这么做）
* 训练循环里只切片，不重算（`round_tensors()` 已经 LRU 缓存）
* 想根治就**离线预处理**——`prepare.py` 就是干这个的，dump 成 `.npz`（每回合一个
  文件），训练侧从"每 worker 12 s × 文件数"变成顺序读 **~800 MB**。
  4 个 demo 现在还能忍，爬到 100 个就必须做

  这笔账值得算清楚：4 文件 × 12 s = **48 s 每个 worker 各付一次**（Windows spawn
  下 `__getstate__` 会清掉已加载的 `_data`，所以是每个 worker 重付，不是共享），
  4 个 worker 就是 4×48 s 的 CPU；而 `round_tensors()` 一个 epoch 只有
  89 × 50 ms ≈ **4.5 s**。**瓶颈在解码，不在算张量**，所以别去优化
  `round_tensors()`，要优化就把它挪到训练前。

  ⚠ 但 817 MB 是**全部 14 个 key**。只 dump 训练真正用到的几个 key 能砍掉一大半
  （`players` 一个就占一半，而 `inventory` 那种 bool 用 `np.packbits` 还能再压 8×）。

### 5.2 切碎的 fire/smoke —— `CszStore` 已经在 load 时修好了

`...m3-inferno.csz` 的 **round 18 / 19 / 20**：side-channel 的 journal 被**逐帧切碎**了。

| | 健康回合（如 round 11） | 碎回合（18 / 20，**合并前**） |
|---|---|---|
| inferno 实例数 | 9 | **930 / 1222** |
| 唯一 entity_id | 9 | **4 / 6** |
| 单个实例 span | ~1285 tick | **0 – 19 tick** |
| 每实例 change 数 | 11 | **1** |

同一个实体被拆成上千条碎片，**全部顶着同一个 `entity_id`**，碎片之间留 2–3 tick 的缝。
**危险的地方在于它看起来是正常的**——线长没错、每个碎片的 `count` 都是 1–16、燃烧格
坐标也合法。真正的损失是**覆盖**：`recipes.fire_state_at()` 在缝里返回 `None`，实测
**约 18% 的帧火凭空消失**，而且不报任何错。

**你不用管它了。** `CszStore` 加载时会把它们合并回一条（`merge_splits=True`，默认开）：

```python
store = CszStore.from_dir("data/demos/csz")
store.merge_stats()
# {'rounds': 89, 'before': 5913, 'after': 917, 'fixed': 3847, 'bad_rounds': 0}
```

实测（4 个 demo / 89 回合，逐 eid 逐 tick 对比合并前后）：

| | |
|---|---|
| 实例数 | **5913 → 917**（碎片本来就不是真的实例） |
| 缝里修回的 tick | **5032** |
| 内容冲突 | **0** —— 合并前能算出值的 tick，合并后逐字节 / 逐格完全一致 |
| 自检不变式 | **101733 / 101733** 全过 |

所以 `round_health(fi, rid)` 现在对 89 个回合**全部返回 `None`**，`prepare.py` 循环里
那行 `continue` 一个都不会滤掉。老的形状判据（实例数 > 64、span 中位数 < 100）留着
当兜底：哪天碎片之间的缝超过 `merge_gap`（默认 8 tick）就拼不起来，那时它还会喊。

细节（为什么不能直接把 `changes` 首尾相接、`merge_gap` 怎么定）见
`store.merge_split_instances()` 的 docstring。**玩家状态（位置/视角/速度/血量）
从头到尾都不受影响**，碎的只有 fire/smoke。

### 5.3 武器/地点下标**跨文件不可比**

每个 csz 的 `data["weapons"]` 只包含**该文件出现过的**武器，编号从 0 开始：

```
mirage 34 项 / nuke 31 项 / inferno 36 项 / dust2 35 项
```

所以 `player["weapon"]` 的 `3` 在 mirage 是 `C4 Explosive`、在 nuke 可能是别的。
实测同一个本地下标 `12`：mirage 是 `High Explosive Grenade`、nuke 是
`Huntsman Knife`、inferno 是 `Glock-18`、dust2 是 `Incendiary Grenade`。

跨 demo 训练必须转成全局下标。三个入口：

| 拿到的东西 | 状态 | 怎么用 |
|---|---|---|
| `tens["inventory"]` | **已经是全局**（`(T,10,43)` 多热） | 直接用，不用转 |
| `tens["players"][:,:,COL["weapon"]]` | **是本地下标** | `lut = store.weapon_lut(fi)`，见 §3.2 |
| `rnd["players"][p]["inventory"][t]` | **是本地下标**（变长列表） | 同上 |

`constants.WEAPON_NAMES`（43 项，抄自 dev 仓库 `config.py`）。
实测 4 个文件里 `AK-47 → 4, AWP → 6` 完全一致，9 种刀全部 `→ 36 (Knife)`。

⚠ **`weapon_name_to_idx()` 对未知名字静默返回 Knife。** 对刀来说这是想要的
（实测 18 个落表名字全是刀），但 `'World'` 也在各个文件的 `data["weapons"]` 里
——它同样会被算成 Knife。`'World'` 从不进 inventory，所以现在没污染数据；
但**换一批 demo 或 CS2 出新武器时，这个静默兜底会给你 Knife 而不是报错**。
用 `store.unmapped_weapons(fi)` 定期核对。

⚠ **`lut[w]` 不能直接写**：`w` 里的 `-1`（实测 20.8% 的 tick）会被当成
Python 负索引，取到 lut 的最后一个元素。这 4 个文件里"最后一个武器"碰巧都是刀，
所以错误值恰好也是 `Knife`，**bug 不会当场显形**——换个末尾不是刀的 csz 就会
静默给出完全错的武器。必须先 `w >= 0` 掩一下，见 §3.2。

### 5.4 切分要按 demo 切，不能按回合切

**这条已经在 `prepare.py` 里落实了** —— `demo_split(store.paths[fi].name, ...)` 把
**一个 csz（一张地图）的所有回合**分到同一个 split（做法和理由见 §4"划分"）。这里
说的是**为什么必须这样**：

如果你做滑动窗口：`stride=16, n_ticks=64` → 相邻窗口有 **75% 重叠**。按窗口 / 按回合
随机切 train/val，验证集里几乎每个窗口都能在训练集找到"前 48 tick 完全一样"的兄弟，
**验证 loss 会好看但毫无意义**。按 demo 切就没这个问题。

⚠ 但 demo 级划分粒度很粗：`seed=0` 的实测是 `train=42, valid=23, test=24` 回合
（2 / 1 / 1 个文件），而且**换个 seed 可能就切出空的 valid**（`--seed 2/3/42` 实测
全部进 train）。样本量太小，**别拿它调超参**。要正经训练先多爬几十个 demo ——
几千个 demo 下 `0.9/0.05/0.05` 自然就准了。

### 5.5 Windows 上 DataLoader 会 pickle 整个 dataset

Windows 用 `spawn` 起 worker，主进程的 dataset 会被序列化 N 份。
如果主进程已经 load 过 csz，那是几百万个 Python 对象 × N 个 worker。
`CszStore.__getstate__` 已经把缓存从序列化里摘掉了。

⚠ 这条对 `SampleDataset` **同样适用**，而且更隐蔽：如果你在任务的 `n_samples()` /
`__init__` 里建了一份"每个样本一个 Python 对象"的索引（`list[tuple]` 之类），
它会跟着 dataset 被 pickle 给**每个 worker**。实测 1750 万个窗口的 `list[tuple]`
是 **2245 MB**，8 个 worker 就是 **17.6 GB** —— 光索引就 OOM。用 `np.cumsum`
前缀和只要 **2 MB**。

另外：**每个 worker 会为自己碰到的 csz 各付一次 ~12 s load**（那是 `CszStore` 的
行为，只在你把 csz 读进 worker 时才相关；`SampleDataset` 读的是 npz，不受影响）。
读磁盘样本的 `num_workers` 只受 IO 带宽限制，4–8 都合理。

### 5.6 速度列有少量 NaN

每个文件的速度列有 580–700 个 NaN（`v/vx/vy/vz` 各一份），
占全部采样的 **0.05%**，形态是"连续两个采样、10 个玩家同时 NaN"——
是服务器卡顿，不是死亡（死人也有速度）。

`CszStore(interp_nan=True)`（默认）已经按列线性插值填掉了；
`nan="keep"` 会留着，你自己处理。其它列**没有** NaN。

> **顺带一提，你文件里的速度是可信的。** `demoparser2` 有个上游 bug：
> 只要传给 `parse_ticks(ticks=...)` 的列表不是**一整段连续 tick**，
> 四个 `velocity*` 列就会返回别的 tick 的速度（stride=2 时 84% 是错的，
> 数值上会被放大成约 stride 倍，实测出现过 ±22,546 HU/s）。
> 详见 `API.md` §4.10。
>
> `extract.py` 已经绕过它了——速度单独走一次不带 `ticks` 过滤的解析再合并回来
> （`VELOCITY_PROPS` / `STATIC_PROPS`），所以 csz 里的是修好的值。
> 自检两条：`v == hypot(vx, vy)` 处处成立，且真人 `|vz|` 不会超过 ~1200 HU/s。
> 你手上 4 个文件都过——实测最大 `|vz|` 是 1152，正是长距离下落的物理上限。

### 5.7 `v` 是水平速度，不是 3D 速度

`v == hypot(vx, vy)` 精确成立（偏差是 f32 舍入），但
`v != |(vx, vy, vz)|`（实测最大差 728 HU/s）。它是独立的一列信息，
要么保留要么丢掉，**但别用 3D 模长去"推导"它**。见 `API.md` §4.10。

另外 `bomb_position` 也不是每 tick 的精确 C4 位置，而是"最后已知的携带者位置"
（实测最远偏离任何玩家 594 HU）。要精确位置就自己从 `inventory` 里找持有
`C4 Explosive` 的人，再取那一 tick 的 `x/y/z`。

### 5.8 ⚠ 事件列表里有不属于本回合的事件

这条单独列出来，因为它是**最容易静默做错标签**的一条（详见 §3.5）。

`rnd["events"]["kills"]` 是按"上一个回合结束 → 这个回合结束"分桶的，
所以 **round 0 会把赛前热身的击杀全吃进来**：de_nuke 的 round 0 有 52 条击杀，
**45 条发生在回合开始之前**；de_dust2 是 56 条里 49 条。

全文件范围：

| 文件 | 击杀落在采样范围外 | 伤害落在采样范围外 |
|---|---|---|
| de_mirage | 43 / 162 (26.5%) | 114 / 630 (18.1%) |
| de_nuke | 62 / 186 (33.3%) | 148 / 658 (22.5%) |
| de_inferno | 25 / 163 (15.3%) | 60 / 710 (8.5%) |
| de_dust2 | 70 / 222 (31.5%) | 137 / 769 (17.8%) |

这些 tick 在采样时间轴上不存在，`recipes.sample_index()` 会返回 `-1`。
**用 `round_events()` / `events_in_window()`**，它们已经滤掉并给了采样下标 `i`；
回合结束的那一枪（`ticks[-1] + 1`）不在这两个里面，用 `tasks/_util.py::late_kills()` 拿。

`sound` 事件是例外，实测 100% 落在范围内。

---

## 6. 动作标签：不在 csz 里，但拿得到

如果你要做行为克隆 `(state, action)`：**当前 csz 没有动作标签**，
`extract.py` 的 `PLAYER_PROPS` 里没抽按键。

但 `docs/PLAN.md` §7.6 ④ 已经实测确认：**逐 tick 的 13 位按键掩码是能拿到的**，
而且两个解析器完全一致：

```
demoparser2 : CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtonDownMaskPrev
demoinfocs  : Player.ButtonsPressedState
```

13 个位：`W, A, S, D, jump, crouch, walk, reload, fire, scope, use, inspect, scoreboard`
（实测出现次数：W 390,069 / D 135,370 / A 133,592 / walk 120,488 / fire 85,328 …）

**做法**：在 `tools/demo_extract/extract.py` 的 `PLAYER_PROPS` 里加一行，
重新抽一次 demo，然后在 `features.build_targets()` 里把它做成动作标签。
注意 `Zoom`(0x400000000) 和 `TurnLeft/Right` 整份 demo **从未置起**，
判断"开镜"要用 `Attack2`(0x800)。

鼠标位移（`usercmd_mouse_dx/dy`）**不在**实体属性层，需要自己解 packet 层的
`svc_UserCmds`（protobuf + codegen-delta），成本高，见 §7.5。

---

## 7. fire / smoke 空间特征的代价

实测（本机）：

| 调用 | 耗时 |
|---|---|
| `recipes.active_fire(rnd, tick)` | ~1.3 ms/次 |
| `recipes.smoke_grid_at(inst, tick, solid=True)` | **3.9 ms/次**（Qhull；没装 scipy 时 ~150 ms） |
| `recipes.smoke_grid_at(inst, tick, solid=False)` | 2.4 ms/次 |

`solid=True` 走的是凸包填充（`voxel.convex_hull_halfspaces` → `scipy.spatial.ConvexHull`，
填充用 numpy 向量化；没装 scipy 时回落到纯 Python 版，同一个凸包但慢 ~40×）。
一颗烟有 ~1300 个 tick，**逐 tick 调用会跑几小时**。正确做法：只在 journal 的 change tick
（一颗烟约 200 次）上预计算，存稀疏数组，训练时查表。

⚠ 语义警告：journal 只存了烟雾的**稀疏采样**（云内部本身就有成片空洞），
`solid=True` 填的是这些点的**凸包**——严格说也是渲染选择，不是数据事实。但实测
（4 份 csz、912 帧有烟、61 152 个有向对）：按壳（`solid=False`）判遮挡会把
**45.8%** 的"被烟挡"记成"通"（只漏不多），凸包 0 漏判，代价 0.55 → 35.7 ms/颗。
所以**遮挡/视线/子弹阻挡一律 `solid=True`**，和渲染、可视化用同一个体积；
`solid=False` 只用来省时间或做对照。

`spatial.py` 里已经封好这几个函数，默认**不**在 `features.py` 里调用。

⚠ 这条和 §3.9 的 oracle 现在是**一致**的：oracle 要和 app 逐位对齐，而 app 的
`visuals.js` 默认 `smokeSolid = true`（实心），`VisOracle` 的默认也正是
`solid=True`；`VisOracle(..., solid=False)` 留作开关，用来复现"只取壳"的差异。

---

## 8. 环境

当前 conda env `cs-net-v5` 里有：`torch 2.11.0+cu128`、`numpy 2.5.3`、`pandas 3.0.6`、
`demoparser2 0.42.0`、`zstandard 0.25.0`（`flask 3.1.3` 只有 `visualizer/` 那个应用在用）。

**torch 已经装好了，CUDA 直接能用** —— 实测 `torch.cuda.is_available()` → `True`，
capability `(8, 6)`，卡是 `NVIDIA GeForce RTX 3090`。所以 `dataset.py` 里
`HAS_TORCH = True`，`SampleDataset[i]` 返回的就是 torch 张量（`to_torch()` 自动跑一遍），
可以直接丢进 `DataLoader`。

本机 GPU 是 RTX 3090（24 GB，sm_86），驱动 `nvidia-smi` 报
CUDA UMD 13.4 —— 任何 ≤13.4 的 CUDA 构建都能跑（向下兼容）。**不需要另装 CUDA
Toolkit**，pip 的 wheel 自带 runtime + cuDNN。要重装 / 换版本时：

```bash
D:\conda\envs\cs-net-v5\python.exe -m pip install torch ^
  --index-url https://download.pytorch.org/whl/cu128
```

验证（实测输出就是 `2.11.0+cu128 12.8 True (8, 6) NVIDIA GeForce RTX 3090`）：

```bash
D:\conda\envs\cs-net-v5\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))"
```

报"No matching distribution"就换 `cu126` / `cu124` 再试（本机驱动都兼容）。

没有 torch 也能跑数据层（`HAS_TORCH = False` 时 `to_torch()` 原样返回 numpy，
`SampleDataset` 退化成普通基类）—— 但那是应急，别拿它当常态。

> 提示：Windows 控制台默认 GBK，代码里所有 print 都用 ASCII 符号。
> 如果你的终端能显示 UTF-8，设 `$env:PYTHONIOENCODING="utf-8"` 即可。

---

## 9. 和 dev 仓库的关系

`D:\project\cs-net\scripts\training_data\` 里有一套完整的
`feature_builder / label_builder / depth_map / wds_writer / torch_dataset`，
但它是从 `.json.gz` 读的，**不能直接搬**。可以复用的是：

* `config.py` 的 `MAP_CONFIG`（地图中心，8 张全抄了；前 4 张有 demo 可对照，
  后 4 张没验证过）、`WEAPON_NAMES`、`DEPTH_DIRECTIONS` —— 已经抄进 `constants.py`
  * ⚠ 这个"中心"不是可玩区域中心，是地图原点偏移：nuke 的玩家 bbox 中心实测
    比表里偏 496 HU。它保证坐标有界，不保证几何居中，别自己"修正"它
* `feature_builder.py` 的特征定义和 14 维 state / 9×14 关系的布局
  —— 搬进 `features.py` 时形状约定 `(T, P, F)` 是一致的
* `label_builder.py` 的标签定义（胜率、下次击杀/死亡、炸弹点）
  —— 搬进 `features.build_targets()`
* `torch_dataset.py` 的 WebDataset + shuffle buffer 流式方案
  —— 数据量大到内存放不下时再上
