# demo_extract — 逐帧 CS2 demo 提取（含火焰与烟雾体素）

把一份 `.dem` 变成"每一帧每个玩家长什么样、场上发生了什么"的结构化数据，
**并且带上了 demoparser2 拿不到的火焰几何和烟雾体素**。

输出可以直接保存成一个文件，加载回来**和保存前逐位相同**（不是近似，不是四舍五入）。

---

## 1. 它解决什么问题

cs-net v2 的 `extract.py` 已经在做逐帧提取，但它有两个限制：

| 限制 | 这份工具怎么做 |
|---|---|
| 每 0.5 秒采一次（≈2 Hz），一场比赛只有两千多个采样点 | **默认每个 tick 都采（≈64 Hz）**，也可以按秒间隔或 tick 步长降采样 |
| 没有火焰、没有烟雾体积 | 通过一个小的 demoinfocs-golang 辅助程序专门取这两样 |
| 不知道"这一步有没有发出移动脚步声" | v6 直接给出 `on_ground` / `jump` / `land`，和速度一起喂给模型自己学 |
| 保存成 JSON 有 183 MB | 自带列式二进制编解码 + zstd，同一份数据 7.0 MB（**比 JSON 小 26 倍**） |

实测（`test.dem`，110 MB，de_ancient，18 回合，全场 101,293 tick）：

| 采样方式 | 采样点数 | 磁盘大小 | 编码后未压缩 | 压缩比 |
|---|---|---|---|---|
| **每 tick（默认，≈64 Hz）** | **74,003** | **7.0 MB** | 46.0 MB | 6.5× |
| `--interval 0.25`（4 Hz） | 4,632 | 0.9 MB | 3.4 MB | 3.7× |
| `--interval 0.5`（2 Hz，cs-net 旧节奏） | 2,322 | 0.6 MB | 1.9 MB | 3.6× |

v6 相比 v5 只多了 `CCSPlayerPawn.m_fFlags` 一个 prop 和三列派生量，
**文件反而更小了一点**（因为速度列修好了，值更规整、更好压）：
每 tick 采样时磁盘大小一样是 7.0 MB，4 Hz 从 984 KB 降到 928 KB，2 Hz 从 584 KB 降到 551 KB。

"未压缩"指的是**二进制编码之后、还没过 zstd 的大小**，不是 JSON 的大小。
同一份数据存成 JSON 是 **183 MB**（所以 codec 相对 JSON 是 26 倍，不是 6.3 倍）。

其中**火焰 + 烟雾一共只占 0.08 MB**——它们不是体积瓶颈，玩家的位置/朝向才是。
所以"要不要带火焰烟雾"几乎不影响文件大小，默认就带着。

---

## 2. 快速开始

```bash
cd tools/demo_extract
./build_sidechannel.sh                 # 编译 Go 辅助程序（需要 Go 工具链）

export PYTHONPATH=$(cd .. && pwd)      # 让 python 找得到 demo_extract 包

# 每个 tick 都采（默认）
python -m demo_extract match.dem -o match.csz

# 4 Hz
python -m demo_extract match.dem --interval 0.25 -o match.csz

# 每 8 个 tick 采一次
python -m demo_extract match.dem --stride 8 -o match.csz

# 看一眼文件里有什么（不需要重新解析 demo）
python -m demo_extract --inspect match.csz

# 提取 → 保存 → 加载 → 逐位比对，自检
python -m demo_extract match.dem --selftest
```

Python 里用：

```python
from demo_extract import extract, codec, voxel

data = extract("match.dem")            # 每 tick
codec.save(data, "match.csz")
back = codec.load("match.csz")

codec.equal(data, back)                # -> (True, '$')
```

跑测试：

```bash
python tests/test_roundtrip.py
CS2_TEST_DEMO=/path/to.dem python tests/test_roundtrip.py   # 额外跑完整流水线
```

---

## 3. 输出长什么样

顶层就是一个普通 dict，可以直接 `json.dump`（`codec.save_json` 会帮你处理 bytes）。

```
data
├─ format        "cs2.demo.v7"
├─ meta          地图名、tickrate、采样方式、总 tick 数……
├─ players       10 个玩家 {steamid, name}（索引固定，下面都用索引指代玩家）
├─ weapons       武器名 -> 整数编号
├─ places        地名   -> 整数编号
├─ skipped_rounds 被跳过的回合 + 原因，如 [{'id':18,'reason':'no round_freeze_end...'}]
│                 **被跳过的回合不在 rounds 里**，取它要用 round_by_id 并接 KeyError
└─ rounds[]      每个回合
   ├─ id / teams / winner / end_reason
   ├─ start_tick          本回合 round_freeze_end 的 tick
   ├─ ticks[]             本回合被采样的 tick（升序）
   ├─ round_seconds[]     相对回合开始的秒数，第一个一定是 0.0
   ├─ bomb_planted[]      每 tick 的 bool
   ├─ bomb_dropped[]      每 tick 的 bool
   ├─ bomb_position[]     每 tick 的 C4 位置（见 §8 的说明）
   ├─ missing_samples[]   缺数据的 (采样序号, 玩家序号)，正常为空
   ├─ players[10]         每人一组**列式数组**，长度都等于 len(ticks)
   │   ├─ x y z                    世界坐标（Hammer 单位）
   │   ├─ yaw pitch                视角（度）
   │   ├─ v vx vy vz               速度大小 / 三个分量（HU/s）
   │   ├─ hp armor helmet defuser alive
   │   ├─ weapon                   当前手持武器 -> weapons 编号
   │   ├─ place                    所在区域 -> places 编号
   │   ├─ flash flash_alpha        闪光剩余时间 / 最大亮度
   │   ├─ inventory                身上所有物品的编号（去重、升序）
   │   ├─ spotted                  能看到他的敌人玩家序号
   │   ├─ shots footsteps          本采样窗口内的开枪数 / 脚步声数
   │   ├─ on_ground                本采样是否**踩在地面上**（v6 新增，见 §4.7）
   │   ├─ jump                     本采样窗口内的**起跳**次数（v6 新增）
   │   ├─ land                     本采样窗口内的**落地**次数（v6 新增）
   │   ├─ crouch                   下蹲量 0..1，**连续**（v7 新增，见 §4.8）
   │   └─ crouching                碰撞盒是否已跳到 54，**阶跃**（v7 新增）
   ├─ events
   │   ├─ kills[]     t a v as w hs ts ab ai af dmg
   │   ├─ damage[]    t a v hp w
   │   ├─ bomb[]      t e s（e = planted / exploded / defused）
   │   ├─ grenades[]  t ty eid th x y z（飞行中的投掷物，逐 tick）
   │   ├─ smokes[]    eid t0 t1 s0 s1 x y z（起烟区间 + **引爆点**）
   │   ├─ infernos[]  eid t0 t1 s0 s1 x y z（起火区间 + **起火点**）
   │   └─ sound[]     t ty p w sil（ty = fire / footstep / jump）
   ├─ fire[]      火焰（见 §5）
   │   └─ origin[]   起火点 x y z —— 火焰坐标就是围绕它给的
   └─ smoke[]     烟雾（见 §6）
       └─ detonation[] 引爆点 x y z —— **体素网格的原点，没有它无法定位烟雾**
```

**坐标单位**：Hammer 单位（HU）。1 HU = 1.905 cm，地图 16 HU = 1 英尺。
cs-net 的训练侧用 `SCALE = 0.0254` 把 HU 转成米，保持一致即可。

**每一列大概是什么数量级？** 位置约 ±2000 HU、速度 0~250 HU/s、`vz` 最多到 ±1150、
`hp/armor` 0~100、`flash_alpha` 是 0~255 字节、火焰每快照最多 16 格、
烟盖住约 800~1800 个体素（网格 32³）……
**完整的实测区间表在 [`API.md` §1.9](API.md)**，写归一化、做数据体检（怎么一眼看出数据坏了）
都用得上。那张表是量出来的，不是估的，可以用 `notes/scratch/measure_ranges.py` 复现。

---

## 4. 怎么用

> 每个函数的完整说明（参数、返回值、真实例子）在 **[`API.md`](API.md)**。
> 这一节是速览，`API.md` 是手册。

`codec.load` 回来就是一个普通 dict，你随时可以手写循环去翻。但**多数时候你要的是这几件事**，
所以 `recipes.py` 把它们包好了（`from demo_extract import recipes`）：

### 4.1 把玩家堆成一个张量

```python
X = recipes.player_tensor(rnd)      # (T, P, F) float32，T=采样数, P=10 个玩家, F=16 个特征
recipes.PLAYER_FEATURES             # F 轴每一列是什么，顺序固定

# v6 还多了三列地面/空中信号，想要 19 列就用这个：
G = recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V6)   # (T, P, 19)
```

F 轴就是：

```
x y z | yaw pitch | vx vy vz | hp armor | flash flash_alpha | weapon place | shots footsteps
```

`PLAYER_FEATURES_V6` 在上面 16 列后面接上 `on_ground jump land`（见 §4.7）。

想要每个玩家一条时间线（`(P, T)`）就用 `recipes.player_tracks(rnd)`。

**朝向别直接喂角度**：yaw 在 ±180 处会跳变，模型没法插值。用
`recipes.yaw_sincos(rnd)` → `(T, P, 2)` 的 sin/cos。

**单位**：位置和速度都是 Hammer 单位。`recipes.to_meters(X)` 会只把 `x y z vx vy vz`
乘上 `HU_TO_METERS = 0.0254`（和 cs-net 的 `SCALE` 一致），其余列不动。

### 4.2 取某一 tick 的全部信息

```python
s = recipes.tick_sample(data, round_id=7, tick=45130)

s["players"]      # (10, 16)  这一 tick 的玩家状态，已经填好 NaN
s["alive"]        # (10,)     bool
s["velocity_ok"]  # (10,)     bool，哪几个人的速度本来是缺的
s["yaw_sincos"]   # (10, 2)
s["on_ground"]    # (10,)     bool，谁踩在地上（v6 才有）
s["jump"]         # (10,)     int，这一 tick 谁起跳了（v6 才有）
s["land"]         # (10,)     int，这一 tick 谁落地了（v6 才有）
s["fire"]         # list，每个是一处正在烧的火
s["smoke"]        # {"grids": [...], "points": [...], "meta": [...]}
s["meta"]         # tick / seconds / map / round / teams ...
```

这就是"渲染一帧"或"喂一个决策模型"需要的东西。tick 必须是采样过的 tick，
不在里面会直接 `KeyError`（用 `recipes.sample_index(rnd, tick)` 先问，返回 `-1` 就是没有）。

### 4.3 火焰：想知道"哪几格在烧"

```python
for c in recipes.active_fire(rnd, tick):
    c["points"]        # (N, 3) 世界坐标 —— 直接可以画
    c["normals"]       # (N, 3) 每格表面法线
    c["parent_index"]  # (N,)   蔓延树：第 i 格是从第 parent_index[i] 格烧过来的，-1 是根
    c["burning"]       # (N,)   bool
    c["type"]          # "molotov" / "incendiary"
```

想要"火场占哪些体素"而不是点云：`recipes.fire_volume_at(rnd, tick, resolution=32.0)`。

### 4.4 烟雾：想知道"烟盖住了哪里"

```python
sm = recipes.active_smoke(rnd, tick)
sm["grids"]   # 每个 (32,32,32) 的 bool 占据网格
sm["points"]  # 每个 (N, 3) 的世界坐标体素中心
sm["meta"]    # entity_id / detonation / voxels
```

只想要一个烟：

```python
g = recipes.smoke_grid_at(inst, tick)         # (32,32,32) bool，包含内部
p = recipes.smoke_points_at(inst, tick)       # (N,3) 世界坐标
# 网格索引 -> 世界坐标（想自己挑体素时用）
w = recipes.grid_to_world(np.argwhere(g), inst["detonation"])
```

`inst["detonation"]` 是**引爆点**，也就是网格原点 —— 这是把体素放进世界坐标的必需品。

### 4.5 切训练窗口

```python
for rid, a, b, tick0 in recipes.iter_windows(data, length=64, stride=32):
    X = recipes.player_tensor(recipes.round_by_id(data, rid))[a:b]   # (64, 10, 16)
```

窗口**不会跨回合**，所以一段序列一定在同一个回合时间轴里。

### 4.6 缺值：只有速度会缺

```python
recipes.nan_report(data)     # -> {'v': 360, 'vx': 360, 'vy': 360, 'vz': 360}
X = recipes.player_tensor(rnd, nan="interp")   # 或 nan="zero"，默认 "keep"
```

实测**只有 4 个速度列会有 NaN**（`v` 和 `vx/vy/vz`），其他列从不缺。
而且形态很规整：总是**连续两个采样点**、**10 个人同时**缺 ——
是服务器卡了一下（freeze），**和死亡无关**（死人也照样报速度）。
全场占比 <1%，线性插值即可。

> `v` 是**速度大小**，它**不在** `PLAYER_FEATURES` 那 16 列里（16 列里是
> `vx vy vz` 三个分量）。所以 `nan_report` 不能只扫那 16 列，否则会漏掉它 ——
> 现在它扫**每个数值列**，包括你自己加进去的。

注意回合第一个采样点（`round_freeze_end` 那一刻）通常就在这个缺口里，
`nan="interp"` 会取"后面第一个已知值"往前填，不会外推出奇怪的值。

### 4.7 地面接触与空中事件（v6 新增，给模型自己判断"这一步响不响"）

以前想判断"这个人有没有发出移动脚步声"，只能拿速度硬套一个阈值。现在不用了 ——
文件里直接给出**原始物理信号**，让模型自己学：

```python
og  = recipes.on_ground_mask(rnd)   # (T, P) bool  这一步踩在地上吗
jmp = recipes.jump_mask(rnd)        # (T, P) int   这一步起跳了几次（事件，精确）
lnd = recipes.land_mask(rnd)        # (T, P) int   这一步落地了几次（推导）

# 想要一次拿全 19 列（16 个老特征 + on_ground / jump / land）：
G = recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V6)   # (T, P, 19)
recipes.GROUND_FEATURES             # ('on_ground', 'jump', 'land')
```

有了"速度 + 是否踩地 + 起跳/落地"，模型能自己推出：踩地且速度高 = 有脚步声；
按 Shift 走 = 速度低 = 无声；在空中 = 根本没有移动脚步声（但起跳/落地本身有声）。
比我们写死的阈值更准，而且不会被"某个武器走速不一样"这种细节坑到。

三个信号的可靠性：

| 信号 | 来源 | 可靠性 |
|---|---|---|
| `on_ground` | `CCSPlayerPawn.m_fFlags` 的 **bit 0**（`FL_ONGROUND`） | 和 `m_hGroundEntity` 是否为有效句柄在 **727,649 个采样点上 100.0000% 一致** |
| `jump` | CS2 `player_jump` 事件 | **精确**，事件自带 tick，粗采样也不会漏（会累加到同一个采样窗口） |
| `land` | 从 `on_ground` 的 `False -> True` 跳变推出来 | 默认每 tick 采样时精确；CS2 **没有落地事件**，粗采样可能漏掉短跳 |

校验（`tests/test_roundtrip.py` 里有）：在空中时 `|vz|` 平均 **142 HU/s**，
踩地时只有 **12 HU/s**，差 **11.6 倍** —— `on_ground` 确实在区分"下落"和"站着"。

注意 `on_ground` 是**原始标志**：死人的尸体只要贴地也会报 `True`。
要"活人站在地上"就自己写 `alive & on_ground` —— 这个组合**没有**被封成函数，
因为一旦封成函数就得回答"多快算有声音"，而那个阈值我们没有出处（见 §8.1）。

### 4.8 下蹲与眼高（v7 新增）

**一句话**：demo 能精确给的是**下蹲量**（连续 0..1）和**碰撞盒状态**（阶跃），
但**站立足眼高的绝对值拿不到**，只能取文档值。两个量**不是一回事**，别互相替代。

```python
c   = recipes.crouch_amount(rnd)    # (T, P) float32  0..1，连续
cd  = recipes.crouching_mask(rnd)   # (T, P) bool     碰撞盒是否已跳到 54
eye = recipes.eye_height(rnd)       # (T, P) float32  眼高（相对脚底，HU）
pos = recipes.eye_position(rnd)     # (T, P, 3)       眼睛的**世界坐标**

# 想要一次拿全 20 列（v6 的 19 个 + crouch）：
C = recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V7)   # (T, P, 20)
recipes.CROUCH_FEATURES             # ('crouch',)
```

| 量 | 来源 | 行为 |
|---|---|---|
| `crouch` | `m_flDuckAmount` | **连续斜坡**，0 → 1 中位数 **13 tick（203 ms @64 tick）** |
| `crouching` | `m_fFlags` 的 **bit 2**（`FL_DUCKING`） | **阶跃**：碰撞盒 72 → 54，中间没有 55..71 |
| `eye_height` | `64.0938 − 18 × crouch` | **连续**，64.09 → 46.09 |

两个量会**不一致**，而且这是对的：碰撞盒跟的是**下蹲键的意图**，不是当前高度。
实测"盒子还是 72 时 `crouch` 最大到过 0.9998"、"盒子已经是 54 时 `crouch` 最小到过
0.7516"。所以判定"能不能钻过去 / 打不打得到"用 `crouching`，画相机用 `eye_height`。

`(x, y, z)` 是**脚底不是眼睛**（碰撞盒是 `[-16,-16,0]..[16,16,72]`，`m_vecMins.z` 恒为 0），
所以 `eye_position` 只给 z 加了眼高。186 段垂直静止的下蹲斜坡里 185 段的 z 完全不变，
也印证了这点。

> ⚠️ **站立足眼高 64.0938 是文档值。** CS2 没有网络化任何视点偏移字段
> （没有 `m_vecViewOffset`），demo 里唯一的候选 `CCSPlayerPawn.m_vecZ` 众数是
> 64.000 但它空中读数是垃圾。两者差 **0.0938 HU = 2.4 mm**。常数在
> `recipes.STANDING_EYE_HEIGHT_HU` / `CROUCH_EYE_DROP_HU`。
>
> 18 HU 这个行程则是**实测精确**的：`m_flDuckOffset == -18 × crouch` 在 727,649 个
> 存活采样上成立，且碰撞盒 72 − 54 = 18 独立佐证。

> ⚠️ 老文件取这些会抛 `KeyError`（写明是 `cs2.demo.v7` 开始有的），
> 不会返回一堆 0。`recipes.tick_sample()` 对 v7 文件多返回
> `crouch` / `crouching` / `eye_height` 三个键。

体积代价实测（`test.dem` 每 tick）：7.371 MB → **7.411 MB（+0.54%）**。
完整的论证和实测数字见 `API.md` §4.12。

## 5. 火焰（fire）

### 为什么需要辅助程序

Source 2 把定长数组**摊平成带下标的属性名**：火焰格子不叫 `m_firePositions`，
而是 `m_firePositions.0000` … `m_firePositions.0015`。读裸名字只会得到一个空数组。
demoparser2 的三条取值路径（`parse_ticks` / `parse_grenades(extra=)` / `parse_event(other=)`）
对这 8 个字段**全部丢弃**，所以只能换解析器。

### 依赖版本：demoinfocs-golang **v5.3.0**

辅助程序用 [demoinfocs-golang](https://github.com/markus-wa/demoinfocs-golang) 读这两个字段，
版本要求写在 `sidechannel_go/go.mod`：**`/v5` v5.3.0**（v4 线已弃用）。

用 v4（哪怕是 v4 线的最后一版 v4.5.1）在较新的 demo 上会直接崩：

```
error: unable to find existing entity 259
error: unable to find existing entity 229
error: unexpected team "\xe31\x03\xe6,\x9bV\xa7\x8d\x81U8\xb0\xbeoy\x1b\xf8E\x15\xf9&D\xa9]SD"
```

**不是 demo 坏了** —— 同一份 demo 用 demoparser2 单独跑能正常读出 24 回合 / 10 个玩家 /
166,018 个采样 tick。是 v4 的实体流解码跟不上 patch 14180 之后的变化，字节读偏了：
先是 delta 更新引用到不存在的实体，接着 `m_szTeamname` 读出乱码，
最后 `ParseToEnd()` 返回错误、整场提取失败。

升级到 v5 后，同一份 550 MB / de_inferno demo：`entity_panics=0`，144 个火焰实例、
170 个烟雾实例、`f32_exact=true`。而在原本就能解析的 demo 上，v4 与 v5 的输出
**262,219 字节逐字节完全一致** —— 升级不改变任何已有行为，只修复读不了的。

#### v5.2.0 的死法更阴：解不开，但**不报错**

2026-11-11 的 CS2 更新给 `CCSPlayerController` 加了三个 **fixed8 编码**的 `uint8` 字段
（`m_recentKillQueue` / `m_nFirstKill` / `m_nKillCount`）。v5.2.0 及更早把它们当 varint 读，
值 ≥ 128 就多吃一个字节，从那次更新起每个 `PacketEntities` 包都错位：

```
error in OnPacketEntities: nextByte: insufficient buffer (697 of 697)
error in OnPacketEntities: unable to find existing entity 2      ← 之后每个包都这样
```

`IgnorePacketEntitiesPanic` 会把坏包一个个丢掉继续跑，**`ParseToEnd()` 照样返回 nil**，
于是写出一个完全合法、但 `fire` / `smoke` 全空的 `.csz`：`infernos=0`、`smokes=0`、
`voxel_capacity=0`，而同一份 demo 的 `smokegrenade_detonate` / `inferno_startburn`
一条不少。唯一的症状是回放页上不放烟和火 —— 很容易被当成"这场没人扔"。

修法就是 v5.3.0（[release notes](https://github.com/markus-wa/demoinfocs-golang/releases/tag/v5.3.0)：
fixed8/16/32 编码按 N 个原始 bit 读）。实测同一份 de_mirage（2026-11-12 录的）：
烟 147 / 火 104，和事件数一模一样；升级前是 0 / 0。老 demo 的输出**一个字节没变**
（`infernos=80 fire_records=94 smokes=98 smoke_records=1230 bytes=129948`，
v5.2.0 与 v5.3.0 逐字节相同）。

从今往后这种"静默空文件"还有一道闸：`extract.py` 的
`_require_fire_smoke_matches_events()` 拿事件流对账 —— **事件里有落地/起火、辅助程序却
一个实例都没读到**时直接让这次提取失败，并在错误里写明要重编译辅助程序。
宁可失败，也不写一份没有火/烟的文件。

`sidechannel_go/main.go` 里另有两处保险：

* `ParserConfig.IgnorePacketEntitiesPanic = true`（默认开，`-ignore-entity-panic=false` 关掉）。
  个别 demo 的 PacketEntities 会引用从未创建过的实体下标，demoinfocs 默认直接 panic 让整场失败；
  打开后该包被丢弃、解析继续，只在 stderr 留一条 `ParserWarn`，末尾汇总成 `entity_panics=N`。
* 配置必须从 `demoinfocs.DefaultParserConfig` 出发再改字段。直接写 `ParserConfig{...}`
  会把 `MsgQueueBufferSize` 变成 0，含义是"强制串行解析"，大 demo 上会明显变慢。

### 数据形态

```python
for f in round["fire"]:
    f["entity_id"]       # 实体 id
    f["inferno_type"]    # 0 = 燃烧瓶, 1 = 燃烧弹（辅助程序实测）
    f["type_name"]       # "molotov" / "incendiary"
    f["origin"]          # [x, y, z] 起火点（格子坐标就是围绕它给的）
    f["first_tick"], f["last_tick"]   # 存在区间
    f["changes"]         # 状态变化快照，只在"变了"的时候记一条
```

每条 `changes[i]`：

```python
c["tick"]      # 这一状态从哪个 tick 开始
c["count"]     # 此时有几格在烧（最多 16）
c["pos"]       # 展平的 3*count 个 float：第 i 格的 x,y,z
c["normal"]    # 展平的 3*count 个 float：第 i 格的表面法线
c["parent"]    # 展平的 3*count 个 float：第 i 格是从哪一格蔓延过来的
c["burning"]   # count 个 bool：这一格还在烧吗
```

用的时候按"最后一次变化"取值：

```python
import bisect

def fire_at(f, tick):
    """回到某个 tick 的火焰状态；不存活则返回 None。"""
    if not (f["first_tick"] <= tick <= f["last_tick"]):
        return None
    ticks = [c["tick"] for c in f["changes"]]
    i = bisect.bisect_right(ticks, tick) - 1
    if i < 0:
        return None
    return f["changes"][i]

def cells(c):
    return [(c["pos"][3*i], c["pos"][3*i+1], c["pos"][3*i+2]) for i in range(c["count"])]
```

### 为什么要用 `changes` 而不是每 tick 存一份

实测这场比赛的 58 处火焰一共经历了 **73,653 个实体帧**，但只有 **737 个不同的状态**——
**占 1%**。逐帧存等于把 99% 的空间浪费在重复上。

### 别用 2D 凸包当火区

`m_firePositions` 是真正的 3D 点。实测 56 处火焰里：

* 竖直跨度最大 **124.3 HU**（能顺着楼梯/箱子往上爬）
* 37.5% 的火焰法线是**倾斜**的，最斜的接近 83°（贴着墙烧）
* 对顶点做平面拟合，残差最大 **53.5 HU**

所以 `ConvexHull2D()` 那种 XY 投影会直接丢掉一百多 HU 的竖直结构。

`parent` 字段给出的是**蔓延树**（第 0 格的 parent 是它自己），可以直接画火焰的扩散方向。

### 但 16 格的坐标是**固定**的

两场比赛（`test.dem` 58 处 + de_inferno 144 处，共 202 处火焰）逐格比对：
**没有任何一格的坐标在燃烧期间移动过**（每个下标在整个存活期只有 1 个不同的坐标）。
火焰"蔓延"只体现在 `burning` / `count` 从 1 涨到 16（实测 1→2→3→5→6→8→10→12→14→16），
最后一帧全部置 `false`。也就是说这 16 个点是**预先算好的地面燃烧点**，不是会动的粒子。

配合法向看：15,418 个燃烧格法向里 96% 的 `normal.y ≥ 0.9`（中位数 ≈ 1.000），
只有 3~4% 是明显倾斜的（最斜的几乎贴墙）。所以大多数火是**平铺在表面上的一滩**，
按法向立锥体画出来就会变成一排"地刺"——`visualizer` 因此改成贴合表面的平面贴片。

---

## 6. 烟雾体素（smoke）

### 数据形态

```python
for s in round["smoke"]:
    s["entity_id"]
    s["detonation"]    # [x, y, z] 引爆点 —— 体素网格的原点
    s["first_tick"], s["last_tick"]
    s["capacity"]      # = 3072，日志缓冲区容量
    s["changes"]       # 每次"内容变了"记一条
```

每条 `changes[i]`：

```python
c["tick"]     # 从哪个 tick 开始
c["size"]     # m_nVoxelFrameDataSize，此刻日志的有效字节数
c["update"]   # m_nVoxelUpdate，更新计数器
c["mode"]     # "append"（接在上一条后面）或 "full"（整体替换）
c["data"]     # bytes，要追加/替换的内容
```

### 怎么还原任意一帧的体积

```python
from demo_extract import voxel

j = voxel.journal_at(s, tick)      # 该 tick 的完整日志；不在存活期返回 None
st = voxel.state_at(s, tick)       # (size, update)

# 1) 边界体素（日志里明确点名的格子；稀疏采样，**不是**一层水密的面）
boundary = voxel.boundary_voxels(j)

# 2) 实体体积（把上面那批格子当点云，填满它的凸包）
solid = voxel.solid_voxels(j)

# 3) 转成世界坐标，喂给渲染器或体素网络
points = voxel.world_voxels(j, s["detonation"], solid=True)
```

**为什么 `solid_voxels` 不是"洪泛填充"**：掩码存的是云的**表面**，而且采样很稀——
真实 demo 上平均每 (x,y) 列只有 ~4 个 Z（83% 的列是**一段**连续 Z），
没有任何一个格子的 64 位全满（出现最多的 `0xf0f0f0f000000000` 也只占 8%）。
所以空腔和外界是连通的，从六个外表面洪泛填充**一个内部体素都补不出来**：
`test.dem` 82 个烟里 59 个增益为 0，最大也只有 20；de_inferno 那场 169 个烟同理（最大 24）。

因此 `solid_voxels` 走的是 `solid_from_shell`，它填满**表面点云的凸包**
（`convex_hull_fill`）。选凸包是因为它**没有任何可调参数**：凸体的竖直截面一定是一个
区间，所以"填满凸包"就等于逐列填满，而每列的上下界由凸包自己插值出来 —— 稀疏采样之间
的洞会被自动补上，不需要先"膨胀"一遍。

**凸包算在哪**：`convex_hull_halfspaces` 走 `scipy.spatial.ConvexHull`（Qhull，C 代码），
没装 scipy 时回落到文件里的纯 Python beneath-beyond；`convex_hull_fill` 的逐列填充是
numpy 向量化的。两处都是纯计算优化，输出和原来**逐格相同**（`tests/test_roundtrip.py`
的 `voxel / hull backend` 组用"两个后端对拍 + 中心/远处点的内外判定"守着这个等式）。
满云（1356 个点）实测：**151 ms → 2.8 ms**；端到端 `smoke_grid_at(solid=True)` 3.9 ms/次，
其中 1 ms 是 journal 解码。当前用哪个后端看 `voxel.HULL_BACKEND`。

`eid 1352` 从 1618 个表面体素变成 **4207** 个实心体素，三视图轮廓填充率从
0.68~0.80 涨到 0.81~0.84。实测两个 demo 的最大体素数 6408（`test.csz` 5428）。

凸包同样**不是"数据"**：它只是"把表面包起来"这个最朴素规则的产物，而且它会
**把所有凹陷一并填平** —— 真实的烟云未必是凸的。

（这条洪泛填充路径**试过并且删掉了** —— 它在真实数据上退化成边界，留着只会让人以为
还有第二条重建路线。上面那个 `59/82 增益为 0` 就是它被删的原因。）

> 顺带记一笔排查结论：我们逐位对过上游参考实现
> `CS2-insight-agent/backend/app/features/demo_analysis/smoke_voxel_decode.py`，
> 掩码的 Morton 解码、六面播种、`occupied | (FULL ^ outside)`、世界坐标公式**完全一致**；
> 也暴力试过 cell/sub 的 Morton 与线性组合、6 种轴排列、8 种 sub 形状、u64 字节序反转、
> 位反转等解码，没有任何一种能把这份数据解成"实心球"。结论是数据本身就是稀疏表面，
> 不是解码错误。

> ### ⚠️ `solid_from_shell` 是渲染启发式，不是格式事实
>
> **参考实现自己也没做 3D 实心体。** 它的 `project_smoke_grid_to_cells` 虽然调了
> `_rebuild_occupied_grid`（洪泛填充），但只取每列的 `min_z` / `max_z` 两个点就输出了，
> docstring 明确说是喂给 **2D 雷达轮廓**用的
> （*"Keep the lowest and highest occupied Z for each XY column so multi-level radar
> filtering can select the appropriate layer before contouring."*）。
> 它从来不需要、也没有回答"这 32³ 该不该是实心"。
>
> 所以分层看：
>
> | 层 | 性质 |
> |---|---|
> | journal 格式 / Morton 解码 / 世界坐标公式 | 社区逆向（`CS2-insight-agent`），我们逐位对齐；**可信**，但不是 Valve 官方文档 |
> | 洪泛填充 | 参考实现里就有；在真实数据上退化成边界，**本项目已弃用并删除** |
> | **填满表面点云的凸包**（`solid_from_shell` / `convex_hull_fill`） | **本项目自己加的**；没有可调参数是它的优点，但"烟雾是凸的"这个假设本身没有数据依据 |
>
> 我们没能 100% 排除"存在第三种解码方式能直接解出实心体"，也没找到 Valve 或社区
> 对这份数据的权威格式说明。
>
> **做遮挡/可见性计算请用 `solid_voxels`（凸包），不要用 `boundary_voxels`。**
> 这里原先写的是反的，实测推翻了：壳是稀疏采样，云**内部**就有成片空洞
> （一颗满云点名 1787 格 / 凸包 5459 格，点名格只有 6.5% 的 6 邻域全占，
> p50 邻居数 4/6），DDA 会从洞里穿过去。4 份 csz、912 帧有烟、61 152 个有向对上比：
> 按壳判遮挡把 **45.8%** 的"被烟挡"记成"通"（只漏不多），凸包 0 漏判，
> 代价是每颗烟 0.55 ms → 3.9 ms（换 Qhull + 向量化填充之前是 35～150 ms，取决于云的大小，
> 见上一条）。既然渲染画的也是凸包，标签就该用同一个体积，
> 完整测量和取舍见 `scripts/training_data/obstruction.py::smoke_volumes`。
> "烟雾是凸的"这个假设依然是**本项目自己加的**，只是它比"壳是水密的"更接近事实。
>
> ### 🎨 `solid_voxels` 只影响画面，不参与 `.csz` 的读写
>
> `voxel.py` 整个模块都是**读取端**的辅助代码：`.csz` 里的字节是 `sidechannel_go`
> 采集、`codec.py` 写出来的，`boundary_voxels` / `solid_voxels` / `convex_hull_fill`
> 只是把这些字节翻译成几何。改这里的任何东西，`.csz` 都不会变一个字节。

引爆点不用自己找：`s["detonation"]` 就是从 `smokegrenade_detonate` 事件里带过来的，
而且实体自身的 `m_vSmokeDetonationPos` 基本都是 `[0,0,0]`（实测不可用），
所以事件是唯一的来源。要原始区间（起止 tick / 秒 + 位置）就看
`round["events"]["smokes"]` 和 `round["events"]["infernos"]`，字段是
`eid t0 t1 s0 s1 x y z`。

网格到世界的换算：

```
world = detonation + (grid - 15.5) * 12.0        # 每个体素 12 HU
```

日志格式（`u16 seq | u16 len | payload`，`payload[1]` 是类型：
`3` KEYFRAME / `2` DELTA / `0` HEARTBEAT）和 32³ 网格的 Morton 编码细节写在
`voxel.py` 的模块注释里。

### 烟雾是**逐帧生长**的，不是一张静态快照

实测一颗烟从 0 长到 1618 个表面体素（实心重建后 5424 个）、日志从 0 字节长到 2511 字节，中间有 **203 次内容变化**。
所以**不需要**用 BFS 去合成膨胀过程——原始数据里就有。

### 为什么只存"变化"

烟雾日志是**只追加**的：新记录接在后面，已经发布的字节不会被改写。
实测 83 个烟雾实体、约 15,000 个采样帧里，"已发布前缀被改写"只发生了一次——
而那一次是**实体 id 被复用**（同一 id 装了两个不同的烟）。所以：

* 正常情况下只存**新增的尾巴**（`mode == "append"`）
* 一旦发现不是前缀延续，就整段存下来（`mode == "full"`），
  这条规则对 id 复用天然安全，不需要额外判断

实测 16,108 条记录全部 100% 通过 `len(journal_at(s, t)) == state_at(s, t)[0]` 的自检。

---

## 7. 存储格式：为什么小，为什么无损

### 无损的保证方式

`codec.loads(codec.dumps(x)) == x` 是**逐位相同**，不是近似：

* **浮点列**只有在每个值都能**逐位**通过 f64 → f32 → f64 往返时才存成 float32
  （NaN 的载荷位也按位比较），否则自动升成 float64。**没有量化、没有四舍五入。**
* 整数列存成"差分 + zigzag 变长整数"，对任意精度 Python 整数都是精确的。
* 没有"有损模式"这个选项存在。

### 压缩手段

| 手段 | 效果 |
|---|---|
| 列式编码（`list[dict]` 同构 → 按列存） | `inventory` 从 4.74 MB 压到 0.01 MB |
| float32 按**字节平面**重排（4 个字节各自连续） | 让 zstd 按字节做熵编码 |
| **按位模式做整数差分**再平面化 | X 列 −27%，yaw −11% |
| **二阶差分**（对位置类更好） | X 列再 −15% |
| 逐列**试压缩选最优**（在 plain / 一阶 / 二阶 里挑） | 位置类选二阶，角度/速度类选一阶；整体再 −3.6% |
| 布尔列位打包、字符串字典化 | |
| 最后整块 zstd level 19 | 总 6.5× |

> 试压缩选最优是有实测依据的：一开始想用"零字节更多就更可压缩"这种启发式，
> 但实测 yaw 的零字节**更少**却**更大** 6%，所以改成真的压一下再比。

### 容错

文件头写了未压缩长度，加载时会校验；截断、坏 magic、版本不符都会直接报错，
而不是悄悄返回半个结果。

---

## 8. 已知取舍，请务必读

0. ⚠️ **`footsteps` 这个字段基本没用，别拿它当"脚步声"用。**
   它来自 CS2 的 `player_footstep` 事件，而**这个事件不是在正常跑动时触发的** ——
   实测它 96.9% 都发生在"同一名玩家起跳前后 1 秒内"，而跑动中的普通采样点
   只有 10.4% 满足这个条件（**9.4 倍富集**）。数量上差 **50 倍**：
   按跑动距离和步幅算应该有约 14,336 次，实际只有 **291** 次。
   两个独立解析器（demoparser2 和 demoinfocs）数出来都是 **360** 次（全场，
   含热身），所以**不是解析器漏了，是游戏本来就只发这么点**。

   **v6 起不用再猜了**：文件里直接给了 `on_ground`（是否踩地）、`jump`（起跳）、
   `land`（落地）。把速度和这三个信号一起喂给模型，它自己就能学出
   "什么时候会发出移动噪声"，不需要我们替它写规则。详见 §8.1。

1. ⚠️ **demoparser2 的 `velocity*` 四个字段不能用"跳着采"的 tick 列表读。**
   这是本项目发现的一个上游 bug：只要传给 `parse_ticks(ticks=...)` 的列表
   **不是一整段连续 tick**，`velocity / velocity_X / velocity_Y / velocity_Z`
   就会返回**别的 tick 上的速度**。实测 stride=2 时就已经有 **84%** 的采样点是错的
   （X/Y/Z、`m_fFlags` 以及其余所有字段都完全正确，只有这 4 个坏）。

   **根因已经定位到源码。** `collect_velocity` 不是读当前行，而是把**已经收集到
   `output` 里的最后两行**位置捞出来做差再乘 64：

   ```rust
   Ok(Variant::F32((last[i] * 64.0) - (prev[i] * 64.0)))
   ```

   跳着采的时候，这两行之间差着**整整一个 stride**，但照样乘 64 ——
   于是速度被放大了约 **stride 倍**。数值验证（`notes/scratch/velocity_mechanism.py`）：
   在 stride = 2 / 8 / 32 下，`velocity_X` 都 **100.0%** 等于
   `(X(上一个保留行) − X(上上个保留行)) × 64`。这也解释了为什么 2 Hz（stride 32）
   的文件里会出现 ±8000 HU/s ≈ 250 × 32。

   同样这也解释了"tick 列表只有 1~2 个元素时速度列直接消失"——凑不出两行。

   所以 `extract.py` 把速度单独拆成一次**不带 `ticks` 参数**的解析
   （`VELOCITY_PROPS` / `STATIC_PROPS`），再按 `(tick, steamid)` 合并回去。
   这条路径不仅正确，还更快：整份 demo 4 个速度字段只要 0.35 s，
   而显式传一段连续 tick 列表要 2.8 s。

   **别指望升级能修掉它。**
   * PyPI 上 `demoparser2` 最新就是 **0.42.0**，也就是我们装的这个（2026-08-08 发布）。
   * 上游仓库是 [LaihoE/demoparser](https://github.com/LaihoE/demoparser)，
     不是同名的 `osztenkurden/cs2parser`（那是另一个 TS/Node 库，版本号 v2.x）。
   * 上游确实有一处相关改动：[PR #353](https://github.com/LaihoE/demoparser/pull/353)
     （2026-09-03 合进 `main`，**尚未发版**）。但它是为了解决"速度 + 事件一起要时
     内存暴涨 300 MB"，不是修这个 bug；作者自己写明
     "`parse_ticks` passes `wanted_events: vec![]`, so the filter behaves exactly as
     before"，而且那段代码在过滤掉的 tick 上补历史的条件是
     `event_with_velocity && !parse_projectiles` —— 而 `parse_ticks` 没有事件，
     这个标志是 `false`，所以历史照样不补，**跳着采仍然错**。
   * 真要上游修，正确改法是把那个补历史的条件从 `event_with_velocity` 换成
     `needs_velocity`。需要的话可以照这个提 issue。

   自检方法（`tests/test_roundtrip.py` 里也有）：
   `v` 就是**水平速度**，所以 `v == hypot(vx, vy)` 必须处处成立；
   而且真人不可能有 |vz| > 1200 HU/s 的时候。
   修之前这两条都会挂（`|vz|` 能到 **22,546**）。

2. **同一份 demo 会被解析两遍**（demoparser2 一次 ≈7.3 s，demoinfocs 一次 ≈5.4 s）。
   这是不 fork demoparser2 的代价。几十份 demo 无所谓，几千份时值得考虑给
   demoparser2 打补丁。

3. **`v` 是水平速度大小，不是 3D 速度**。修掉上面那个 bug 之后，
   `v == hypot(vx, vy)` 是**精确成立**的（实测最大偏差 2e-5，就是 f32 舍入），
   但 `v != |(vx, vy, vz)|`（最大差 728 HU/s）。它是独立信息，必须留着。

4. **`round_seconds` 不能从 tick 推出来**。`game_time` 和 tick 之间有一个**每份 demo 固定的偏移**
   （这份是 1181）。虽然这里不是瓶颈，但别假设 `seconds = tick / 64`。

5. **每个回合的时间轴从它自己的 `round_freeze_end` 开始**。`total_rounds_played == 0`
   这一组会把赛前热身一起吃进来（这份 demo 有 1181 个 tick 的热身），
   所以采样时做了裁剪。这是"采样点占全部 tick 的 73%"的原因——剩下的 27% 是
   回合之间的准备时间。

6. **`bomb_position` 是"最后已知的 C4 携带者位置"**，不是每 tick 精确位置，也不是
   C4 实体的位置（和 cs-net 语义一致：没人持有的 tick 沿用上一个位置）。
   拿着的时候和那人的 `x/y/z` 完全相等；安放后冻结在安放点，离蹲着拆包的人
   ≤ 80 HU，是对的；**掉在地上（`bomb_dropped`）时只是"掉落点"**——C4 脱手后还会
   带着玩家的速度滑/掉一段，实测离真正的落点中位数 ~175 HU、最多 456 HU
   （4 份 demo、95 次掉落→拾取）。详见 `API.md` §4.10。

7. **`m_bDidSmokeEffect` / 烟雾颜色等字段没进这份 schema**。它们可以用
   `parse_grenades(extra=[...])` 拿到，但和"体素内容"不是一回事，需要的话再加。

8. **辅助程序会拒绝写出有损文件**：如果某个火焰坐标没能通过 f32 往返，它会直接报错，
   而不是默默写一个精度受损的文件。

9. **烟雾/火焰的存活时长是游戏常量，可以用来校验**。实测每一对
   detonate→expire 都是**精确的 22.06 秒**；火焰是**5.50 秒**（燃烧弹）
   或 **7.02 秒**（燃烧瓶）。配对时按 `(实体 id, 回合号)` 一起做键——
   只按实体 id 配会跨回合错配（id 会被复用），实测那样会多出 4 对
   相差 ±400 秒的垃圾配对。

10. **只有 4 个速度列会缺值**（`v` 和 `vx/vy/vz`），其他列从不缺。形态总是**连续两个采样点、
   10 个人同时缺**——是服务器卡顿，**不是死亡**（死人也报速度）。全场 <1%，
   见 §4.6。回合第一个采样点通常就落在这个缺口里。
   注意 `--interval` 越粗越容易整段跳过这个缺口（1 Hz 和 4 Hz 下实测一个都不剩）。

11. **`detonation` 是按"实体 id + 最近的 tick"匹配出来的**，不是原样拷贝。
    实体 tick 和事件 tick 之间有一个每份 demo 固定的偏移，所以只能比"谁更近"；
    匹配窗口是 60 秒，超出就留 `None` 而不是塞一个错的位置。
    实测 `test.dem` 82/82 个烟雾、58/58 处火焰都匹配上了，而且每个烟雾云的质心
    离自己的引爆点平均 41 HU、最远 90 HU（云本来就比引爆点大，这是合理的）。

    **但不是每份 demo 都 100%。** 一份 550 MB / de_inferno 的 SourceTV demo 上是
    **169/170**（火焰 144/144 全中）：漏掉的那个 `entity_id=86` 在**最后一个回合**，
    只活了 646 tick、只有 1 条 journal 记录 —— demo 在烟雾还没结束时录完了，
    自然没有对应的 `smokegrenade_detonate`。所以**消费端必须处理 `detonation is None`**：
    `recipes` 层会抛 `ValueError`（见 API.md §6.4），渲染器应该跳过而不是拿 `[0,0,0]` 兜底 ——
    否则会在地图原点凭空多出一团雾。

12. **`on_ground` 是"原始的踩地标志"，不等于"活着站在地上"**。死人只要尸体还在地板上
    也会报 `True`；只有"尸体/实体都还没生成"的那几行才是 `False`。
    想要"活人站在地上"就写 `alive & on_ground`。
    这份 demo 里：活着的采样点 523,183 个，其中 **17,818 个（3.41%）在空中**。

13. **`land` 是推出来的，不是事件**。CS2 没有落地事件，所以 `land` 是
    "上一个采样点活着且在空中，这一个采样点活着且踩地"。默认每 tick 采样时是精确的；
    用 `--interval 0.25` 这种粗采样时，一个短跳可能整个发生在两个采样点之间，
    那就看不见了（`jump` 不受影响，它是事件，永远精确）。
    这份 demo 实测：起跳 **415** 次、落地 **451** 次（多出来的 36 次是走下台阶/边缘掉落，
    没有按键起跳，所以只有落地没有起跳）。


### 8.1 为什么 `footsteps` 不能当脚步声用（实测调研）

**结论：它不是跑动脚步声，而是"落地声"性质的稀有事件。别用它做移动噪声信号。**

| 检验 | 结果 | 说明 |
|---|---|---|
| 全场事件数 | **360** | 含热身；落在回合内的 291 次 |
| 按跑动距离/步幅应有的数量 | **~14,336** | 差 **50 倍** |
| 两个解析器是否一致 | demoparser2 = 360，demoinfocs = 360 | **不是解析器的问题** |
| 触发时的速度 | 最低 135，中位 **245** HU/s | 只在**跑动**时触发，站立时一次都没有 |
| 与"同一玩家起跳"的关系 | **96.9%** vs 基线 **10.4%** | **9.4 倍富集** |
| 5 秒窗口内有几个不同玩家触发 | 平均 **1.22** 个 | 同期平均有 **6.35** 个玩家在跑动 |

所以它不是"道具投掷的声音"（投掷时速度不固定，而实测 100 HU/s 以下**一次都没有**），
但它**也不是**正常的跑动脚步声 —— 它几乎只在玩家起跳/落地那一下触发。

**替代方案：不要造阈值，把原料交出去。** CS2 里按住 Shift 走路是无声的，
所以"响不响"大致就是"跑还是走 + 在不在空中"。但**我们不再提供一个阈值函数**——
原因见下。

### 为什么这里没有任何 `audible_movement` / `RUN_SPEED_HU`

这个工具**曾经**有一个 `audible_movement(rnd, threshold=167.0)`，还有配套的
`RUN_SPEED_HU` / `WALK_SPEED_RATIO` / `RUN_SPEED_MIN_HU` / `RUN_SPEED_MAX_HU`。
**已经全部删掉了。**

删掉的理由不是那些观测不对，而是**那个 167.0 是我们自己编的**：

| | 性质 |
|---|---|
| 跑速台阶 202…252、走速台阶 102…132、走/跑比中位数 0.515 | **实测**（这份 demo，按武器取速度分布众数） |
| **"阈值 = 空带中点 167.0"** | **没有任何出处** —— 游戏没说、数据也没说，只是我挑了一个让两边余量相等的数 |

把这个数写进 API，等于把一个人的选择伪装成事实，而且**它会被下游当成真值用**。
这个项目对数值的要求是"要么有出处，要么别进代码"，所以：

* 留着的是**原材料**：`v` / `vx` `vy` `vz` + `on_ground` + `jump` `land`（§4.7）；
* 留在文档里的是**实测观测**（上面那张表的数字），谁想自己定规则可以拿去用；
* 从代码里彻底消失的是**那个没有出处的阈值**。

真要用一条规则，正确的做法是把它当**超参**、用你的下游指标去调，而不是从我们这里
接一个常数。这也正是把速度 + 踩地直接喂给模型更好的原因：模型自己会找到那条线，
而且会按武器、按状态（走 / 蹲走 / 空中 / 被打飞）分别去找。

> 顺带一提，这个常数已经翻过一次车：第一版是 135.0，从直方图谷底**目测**的，
> 只比实测最大走速 132 高 3 HU/s，一有噪声就会翻。改成 167 只是把它挪到空带中点，
> **依然是编的**。既然两次都是编的，就不该留在 API 里。

`footsteps` 字段仍然保留（和 cs-net 对齐），但建议直接忽略。
`recipes.footstep_warning(data)` 会返回一段说明当前数据里这个字段有多不可靠的文字。

> 唯一一处**代码层面**的痕迹是 `_landing_counts`（extract.py）：它判断"落地"
> 用的是 `on_ground` 的 **0/1 翻转**，不是速度阈值——那是硬事实，不是调参。


---

## 9. 和 cs-net v2 的字段对应

`PLAYER_PROPS` 这一份**逐字照抄** cs-net v2 的清单，所以训练侧已经在消费的字段一个都不会少：

```
X Y Z pitch yaw velocity velocity_X/Y/Z health armor has_helmet has_defuser
is_alive weapon_name inventory inventory_as_ids flash_duration flash_max_alpha
last_place_name approximate_spotted_by team_num steamid name game_time
total_rounds_played is_bomb_planted is_bomb_dropped
```

输出 schema 也尽量对齐 v2 的命名（`x/y/z/yaw/pitch/v/vx/vy/vz/hp/armor/helmet/defuser/
alive/weapon/inventory/place/flash/flash_alpha/spotted/shots/footsteps`）。

v6 另外加了 `CCSPlayerPawn.m_fFlags` 一个 prop，派生出的 `on_ground / jump / land`
三列是**新增**的，老字段一个都没动 —— 所以 v6 文件可以直接喂给老的 cs-net 训练侧
（用 `recipes.PLAYER_FEATURES` 那 16 列），想用新信号再切 `PLAYER_FEATURES_V6`。

和 v2 有意的差别：

* `round_seconds` 和事件秒数**不做两位小数四舍五入**（v2 会 round 到 2 位，那是不可逆的）
* 玩家每个采样点都严格对齐到 `ticks`，缺数据记在 `missing_samples` 里，
  而不是让数组悄悄变短（v2 是后者）
* 多了 `fire` / `smoke` / `meta` / `skipped_rounds`

---

## 10. 文件清单

```
tools/demo_extract/
├─ __init__.py              公开 API
├─ __main__.py              python -m demo_extract
├─ extract.py               主流程：parse_ticks + 事件 + 按回合组装
├─ sidechannel.py           调用并解析 Go 辅助程序的二进制输出
├─ sidechannel_go/main.go   火焰 16 格几何 + 烟雾日志的提取器
├─ voxel.py                 烟雾日志解析、逐帧还原、Morton 解码、网格->世界
├─ codec.py                 无损列式二进制编解码 + zstd
├─ recipes.py               上层取用：张量、单 tick、火焰/烟雾几何、训练窗口
├─ API.md                   **每个函数的详细说明**（含真实数据例子）
├─ cli.py                   命令行
├─ build_sidechannel.sh     编译辅助程序
└─ tests/test_roundtrip.py  测试（含完整流水线与逐位往返自检）
```

依赖：`demoparser2`、`pandas`、`numpy`、`zstandard`，以及构建辅助程序用的 Go 工具链。