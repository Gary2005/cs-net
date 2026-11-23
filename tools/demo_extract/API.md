# API 参考 —— 每个函数到底在干什么

这份文档假设你已经有了一个 `.csz` 文件。里面每个函数都配了**真实跑出来的例子**，
数字来自 `test.dem`（de_ancient，18 回合，10 个玩家，74,003 个采样点）。

先读第 1 节，不然 `round_by_id` 这类函数会看不懂。

---

## 1. 先搞清楚数据长什么样

### 1.1 为什么要按"回合"切开

一场 CS2 比赛不是一条连续的时间线。中间有热身、有回合之间的准备时间、有换边。
如果把整场当成一条线，模型会学到"上一帧在 A 点、下一帧突然瞬移到 B 点"这种假模式
（那其实是回合重置）。

所以文件的结构是：

```
整份 demo
└─ rounds[]        每个回合独立一条时间线
   ├─ ticks[]      这个回合里被采样的 tick，升序
   ├─ players[10]  每个玩家一组数组，长度都等于 len(ticks)
   ├─ events       这个回合发生的事
   ├─ fire[]       这个回合的火焰
   └─ smoke[]      这个回合的烟雾
```

**回合内的时间轴从它自己的 `round_freeze_end`（冻结结束、可以动了）开始算 0 秒。**

### 1.2 `round_id` 是什么，为什么不能直接下标

这是最容易踩的坑，也是 `round_by_id` 存在的唯一理由。

每个回合有一个 `id`，它就是游戏自己的计数器 `total_rounds_played`：
第 1 个回合是 0，第 2 个是 1，以此类推。

问题在于：**解析失败的回合会被丢掉**。比如某个回合没有 `round_freeze_end` 事件
（热身被切断、demo 从半路开始录），那这个回合就没法确定时间轴起点，只能跳过。
跳过一个之后：

```
游戏里的回合号:  0   1   2   3   4   5
data["rounds"]: [0] [1] [3] [4] [5]      ← 2 被跳过了
下标:            0   1   2   3   4
```

这时候 `data["rounds"][2]` 拿到的是 **id=3** 那个回合，不是 id=2。
下标和回合号对不上，而且**不同 demo 对不上的方式还不一样**。

所以：

```python
rnd = recipes.round_by_id(data, 7)   # ✅ 给我 id=7 的回合
rnd = data["rounds"][7]              # ❌ 这是"第 8 个存下来的回合"
```

跳过了哪些、为什么，记在 `data["skipped_rounds"]` 里：

```python
recipes.skipped_rounds(data)
# -> []                                      干净的话是空的
# -> [{'id': 7, 'reason': 'no round_freeze_end, so the round has no time origin'}]
```

**被跳过的回合是"不存在"，不是"空回合"。** 它不在 `data["rounds"]` 里，
也不会让别的回合的 id 变小 —— 洞就是洞。所以 `round_by_id(data, 7)`
在被跳过时会**抛 `KeyError`**，详见 §4.1。

> **实测例子**：`test.dem` 里 `total_rounds_played` 有 0..18 共 19 组，
> 其中 **id=18 被跳过了**（那组没有 `round_freeze_end`）。所以：
>
> ```python
> recipes.round_ids(data)         # [0, 1, ..., 17]        18 个回合
> recipes.skipped_rounds(data)    # [{'id': 18, 'reason': 'no round_freeze_end, ...'}]
> data["rounds"][17]["id"]        # 17   ← 这次恰好下标和 id 相等
> ```
>
> 这次 `id == 下标` 成立，**但那是巧合**：跳过的恰好是最后一个（id=18），
> 所以洞落在末尾，没有让前面的错位。要是跳过的是中间的回合，
> `data["rounds"][i]["id"]` 就不再等于 `i` 了。**永远别用下标。**

### 1.3 玩家是"索引"，不是名字

`data["players"]` 是一个长度 10 的列表：

```python
[{'steamid': '76561199517672765', 'name': '我的阿梓喵在奥林匹斯飞天滋崩'},
 {'steamid': '76561198...',        'name': '已老实（@…@）'},
 ...]
```

**整份文件里所有涉及玩家的地方都用这个列表的下标（0..9）指代玩家**，
比如 `events.kills` 里的 `a`（attacker）和 `v`（victim）、`spotted` 里的编号。
索引在整个回合内固定，所以可以直接当"第几号玩家"用。

注意列表顺序是**解析顺序**，不分组（不是前 5 个 T、后 5 个 CT）。
谁在哪个队看 `round["teams"]`，它也是按同样的索引排的：

```python
round["teams"]   # ['T','T','CT','CT','T','T','CT','CT','T','CT']
#                   ↑ 玩家 0   1   2   3 ...
```

### 1.4 武器和地名也是"索引"

```python
data["weapons"]   # {'AK-47': 0, 'AWP': 1, 'C4 Explosive': 2, 'Decoy Grenade': 3, ...}
data["places"]    # {'': 0, 'Alley': 1, 'BombsiteA': 2, 'BombsiteB': 3, ...}
```

`player["weapon"]` 存的是编号，`player["place"]` 也是。
要还原名字就 `list(data["weapons"])[idx]`。`places` 里的 `''`（0 号）表示"不在任何命名区域"。

### 1.5 一个回合里的完整字段

```python
rnd["id"]                 # 3            游戏回合号
rnd["teams"]              # 10 个 'T'/'CT'
rnd["start_tick"]         # 19574        本回合 round_freeze_end 的 tick
rnd["winner"]             # 'T'
rnd["end_reason"]         # 'ct_killed'
rnd["bomb_planted_time"]  # 31.83 或 None（相对回合开始的秒）
rnd["ticks"]              # [19574, 19575, ...]  本回合的采样 tick
rnd["round_seconds"]      # [0.0, 0.015625, ...] 和 ticks 一一对应，第一个恒为 0
rnd["bomb_planted"]       # [False, False, ...]  每采样点一个 bool
rnd["bomb_dropped"]       # 同上
rnd["bomb_position"]      # [[x,y,z], ...]       每采样点的 C4 位置（见 §4.10 的说明）
rnd["missing_samples"]    # [(采样下标, 玩家下标), ...]  正常是空的
rnd["players"]            # 10 个 dict，每个是 25 条等长数组（v6；老文件是 22 条）
rnd["events"]             # 见 §1.6
rnd["fire"]               # 见 §5
rnd["smoke"]              # 见 §6
```

### 1.6 每个玩家记录里有什么

25 条数组，长度都等于 `len(rnd["ticks"])`：

| 字段 | 类型 | 含义 |
|---|---|---|
| `x` `y` `z` | float | 世界坐标，Hammer 单位 |
| `yaw` `pitch` | float | 视角，度。yaw ∈ (-180,180]，pitch ∈ [-89,89] |
| `v` | float | **水平速度大小**，HU/s（`= hypot(vx, vy)`，精确成立；**不等于** 3D 速度，见 §4.10） |
| `vx` `vy` `vz` | float | 速度分量，HU/s |
| `hp` `armor` | float | 0..100 |
| `helmet` `defuser` | bool | 有没有头盔 / 拆弹器 |
| `alive` | bool | 活着吗 |
| `weapon` | int | 手持武器 → `data["weapons"]` 编号 |
| `place` | int | 所在区域 → `data["places"]` 编号 |
| `flash` `flash_alpha` | float | 被闪剩余秒数 / 最大亮度 |
| `inventory` | list[int] | 身上所有物品编号（去重升序，**可能是空表**） |
| `spotted` | list[int] | 能看到他的**敌人**玩家编号。⚠ 这一列是**从 `CCSPlayerPawn.m_bSpottedByMask` 推出来的**（不是引擎那个 `m_bSpotted` bool —— 两者实测 100.0000% 冗余），而且**提取时就按队伍过滤过**，所以永远不会有队友。⚠ 引擎写这个字段被限速在 **33 tick = 0.516 s 的网格**上（实测，同一回合 10 人共用一个相位），所以它比真实观测**平均慢约 178 ms**；要逐帧的纯几何可见性用 `visualizer/static/js/visibility.js`（README §4.9） |
| `shots` | float | 本采样窗口内的开枪数 |
| `footsteps` | float | ⚠️ **不要当脚步声用** —— 见 §4.3，它是落地声性质的稀有事件 |
| `on_ground` | bool | **v6**：这一采样点是否踩在地面上（原始标志，死人贴地也算 True） |
| `jump` | int | **v6**：本采样窗口内的**起跳**次数（`player_jump` 事件，精确） |
| `land` | int | **v6**：本采样窗口内的**落地**次数（从 `on_ground` 跳变推导） |
| `crouch` | float | **v7**：下蹲量 0..1，**连续**（`m_flDuckAmount`）。眼高 = 64.0938 − 18·crouch |
| `crouching` | bool | **v7**：`FL_DUCKING`，碰撞盒是否已从 72 跳到 54。**阶跃**，≠ `crouch > 0.5` |

`on_ground` / `jump` / `land` 只有 `format >= "cs2.demo.v6"` 的文件才有，
`crouch` / `crouching` 只有 `format >= "cs2.demo.v7"` 才有；老文件取它们会抛
`KeyError`，报错信息里会写明是哪个版本开始有的。详见 §4.11、§4.12。

**每一列大概是什么数量级？** → 见 **§1.9**（实测区间表，写归一化和做数据体检都用得上）。

### 1.7 事件

```python
e = rnd["events"]
e["kills"]     # [{'t':2502, 'a':6, 'v':4, 'as':-1, 'w':5, 'hs':True,
                 #   'ts':False, 'ab':False, 'ai':False, 'af':False, 'dmg':115}, ...]
e["damage"]    # [{'t':1877, 'a':9, 'v':1, 'hp':1, 'w':12}, ...]
e["bomb"]      # [{'t':3219, 'e':'planted', 's':31.83},
                 #  {'t':5063, 'e':'defused', 's':60.64}, ...]
e["smokes"]    # [{'eid':1352,'t0':2115,'t1':3527,'s0':14.58,'s1':36.64,
                 #   'x':662.83,'y':483.76,'z':125.13}, ...]
e["infernos"]  # 同样的结构，起火点
e["grenades"]  # [{'t':1728,'ty':'CMolotovProjectile','eid':1354,'th':9,
                 #   'x':692.8,'y':533.3,'z':190.1}, ...]   飞行中的投掷物，逐 tick
e["sound"]     # [{'t':1213,'ty':'fire','p':4,'w':13,'sil':False}, ...]
               # ty ∈ {'fire', 'footstep', 'jump'}
```

`ty` 的三种值：

* `'fire'` —— `weapon_fire`，**开枪**，带 `w`（武器编号）和 `sil`（消音器）
* `'footstep'` —— `player_footstep`，⚠️ 稀有且语义可疑，见 §4.3
* `'jump'` —— `player_jump`，**起跳**（v6 新增），只带 `p`

字段名的意思：

* `t` = tick（**注意**：是原始 tick，不是回合内秒数）
* `s` = 相对回合开始的秒数
* `a` / `v` = attacker / victim 的玩家索引（`-1` 表示世界/自己，比如摔死、被 C4 炸死）
* `w` = 武器编号，`dmg` = 伤害
* `hs` = headshot，`ts` = through smoke，`ab` = attacker blind，
  `ai` = attacker invisible，`af` = attacker flashed
* `as` = assister 的玩家索引（`-1` = 无）
* `eid` = 实体 id，`ty` = 类型名，`th` = 投掷物类型编号

**回合结束后的有效交火（`post`）。** CS2 在 `round_end` 那一 tick 就把
`total_rounds_played` 加一，但下一回合真正开始（`round_freeze_end`）还有几秒；
这几秒里玩家仍在非热身、非冻结状态，可以继续交火，**游戏内终场记分板也会计入**。
这些 `player_death` / `player_hurt` 事件在新文件里被归到**刚刚结束的回合**
（`rnd["events"]["kills"]` / `["damage"]`），并多带一个 `"post": True`：

```python
k = {"t": 34607, "a": 4, "v": 7, ..., "post": True}   # 回合结束后、下一回合前
k["t"] > rnd["ticks"][-1]                             # 它没有对应的采样帧
```

* **传统 KDA / 记分板统计要收**：`app/metadata.py`、`visualizer/stats.py` 的
  K / D / A 都按 tick 下界过滤，不设上界，所以这些事件天然计入。
* **采样 / 训练标签不要收**：它们没有采样帧。`recipes.sample_index()` 会返回
  `-1`，`features.round_events()` / `events_in_window()` 按
  `ticks[0] <= t <= ticks[-1]` 过滤，已经自动排除。`tasks/_util.py::late_kills()`
  只补 `t == ticks[-1] + 1` 的收尾一枪，不会把更晚的 `post` 事件当成标签。
* 只归并**两个回合之间**的交火（本回合 `round_end` → 下一回合 `round_freeze_end`）。
  第一个回合开始之前的热身、最后一个回合结束之后的击杀/自杀/清理死亡都不属于任何
  回合，一律不收。
* `meta["post_round_events"]` 为 `True` 表示这份文件带上述语义（老文件没有这个键，
  行为仍是旧的：这些事件被丢弃）。
* 自杀（`a == v`）在原始事件里存在，但传统 KDA 只把它算作死亡，不算击杀 ——
  和游戏内记分板一致。

### 1.8 tick、秒、采样下标，三者别混

一份 demo 里同时存在三种"时间"：

| 说法 | 例子 | 怎么来的 |
|---|---|---|
| **tick** | `19574` | 全局递增的整数，64 tick/秒 |
| **秒** | `0.0` | `round_seconds`，相对本回合 `round_freeze_end` |
| **采样下标** | `0` | `rnd["ticks"]` 里的位置，取决于你采样多密 |

换算**不要自己算**：

* `round_seconds` 和 tick 之间有一个**每份 demo 固定的偏移**（这份是 1181），
  所以 `秒 ≠ tick / 64`。想要秒就查 `round_seconds`。
* 采样下标和 tick 之间差一个采样步长。默认每个 tick 都采（下标差 1 = tick 差 1），
  但 `--interval 0.25` 之后下标差 1 = tick 差 16。**永远以 `rnd["ticks"]` 为准。**

---

### 1.9 每个字段的量级（**实测**，不是估计）

用来做两件事：写归一化、以及**一眼看出数据坏了**（比如速度出现 8000 就知道踩了 §4.10 那个坑）。

> **这些数字是从真实文件量出来的**（`test.dem` / de_ancient / 18 回合 / 74,003 个采样点
> × 10 个玩家 = 740,030 行）。
> **不是规范值**：位置那几行随地图变，事件计数随比赛变。
> 凡是有硬边界或来源的，我在"性质"列单独标出来了 —— 没标的就只是这份 demo 的观测。

**三列怎么读：** `min` / `max` 就是最小最大值；**`p50` 是中位数**
（第 50 百分位数：**一半的值比它小**）。这里**故意不给平均值** ——
这些列严重偏斜，平均值会被极端值带偏：

| 字段 | 平均值 | **p50（中位数）** | 说明 |
|---|---|---|---|
| `v` | 97.2 | **69.0** | 38% 的采样点是 0（站着不动），把平均值抬得没意义 |
| `hp` | 64.2 | **100** | 29% 是 0（死人），但活人几乎都是满血 |
| `vz` | 2.5 | **0** | 56.5% 精确为 0 |
| `shots` | 0.01 | **0** | 99.5% 是 0 |

所以看 `v` 要读成："中位数 69 HU/s，38% 的时间站着不动，最快 282 HU/s"，
而不是"平均 97"。`p50 = 0` 的列（`shots` / `footsteps` / `jump` / `land` / `vz`）
**一半以上都是 0**，做归一化时别用 min-max。

> 想更细就自己算：`np.percentile(col, [1, 50, 99])` 或 `np.median(col)`。
> 测量脚本里打印的就是 p1 / p50 / p99 三个点。

#### 玩家列（单位：HU / 度 / HU·s⁻¹）

| 字段 | min | p50 | max | 性质 |
|---|---|---|---|---|
| `x` | −2267 | 113.7 | 1396 | **地图相关** |
| `y` | −2327 | −286.1 | 1728 | **地图相关** |
| `z` | −175.3 | 82.0 | 273.9 | **地图相关**（de_ancient 高差约 450） |
| `yaw` | −180 | −0.90 | 180 | 硬边界 (−180, 180]，**会环绕**，见 `yaw_sincos` |
| `pitch` | −59.3 | 2.0 | 68.2 | 引擎上限 ±89；**这份 demo 没人看那么斜** |
| `v` | 0 | 69.0 | **282** | 水平速度。武器上限 250，>250 只出现在被加速/爆炸推动时 |
| `vx` | −274.5 | 0 | 262.5 | |
| `vy` | −265 | 0 | 265.5 | |
| `vz` | **−1152** | 0 | 1154 | 下落/起跳。**绝大多数采样点是 0** |
| `hp` | 0 | 100 | 100 | 死人是 0，不是缺值 |
| `armor` | 0 | 96 | 100 | 不穿甲时 0 |
| `flash` | 0 | 0 | 5.129 | 被闪剩余**秒数** |
| `flash_alpha` | 0 | 255 | 255 | 闪光亮度，**0–255 字节** |
| `weapon` | −1 | 8 | 27 | 索引；**−1 = 没有武器**（死亡/观战） |
| `place` | 0 | 7 | 18 | 索引；这份 demo 有 19 个地名 |
| `shots` | 0 | 0 | 1 | 每 tick 采样时是 0/1；**只有 0.525% 的采样点 >0** |
| `footsteps` | 0 | 0 | 1 | 只有 **0.039%** >0（全场 291 次，见 §4.3） |
| `on_ground` | 0 | 1 | 1 | **97.2% 是 1**（含死人贴地的尸体） |
| `jump` | 0 | 0 | 1 | 全场 **415** 次 |
| `land` | 0 | 0 | 1 | 全场 **451** 次 |

布尔列里 True 的占比：`helmet` **56.3%**、`defuser` **13.8%**、`alive` **70.7%**。

列表列的长度：`inventory` **0–8**（p50 3，死人是 0）、`spotted` **0–3**（p50 0）。

> 三个"稀疏"列值得注意：`shots` / `footsteps` / `jump` / `land` 的 p50 和 p99 **都是 0**，
> 也就是 99% 的采样点是 0。**归一化时别用 min-max**，会把这些信号压成 0。

#### 回合级

| 字段 | min | p50 | max | 性质 |
|---|---|---|---|---|
| `round_seconds` | 0 | 32.1 | 123 | 第一项**恒为 0.0** |
| `ticks` | 1181 | — | 100271 | 全场绝对 tick；起点随 demo 变 |
| `samples/round` | 2216 | 3669 | 7875 | 每 tick 采样时；粗采样按比例缩 |
| `missing_samples` | 0 | 0 | **0** | 这份 demo 一个都没缺 |
| `bomb_planted_time` | 25.97 | 30.43 | 110.2 | 秒；没下包的是 `None` |
| `bomb_position.z` | −175.3 | 57.4 | 196.3 | x/y 同坐标量级 |

#### 事件

| 字段 | 量级 | 备注 |
|---|---|---|
| `kills` | **7.28 次/回合** | 全场 131 次 |
| `kills[].dmg` | 8 – **544** | 中位 31；**能超过 100**（过杀） |
| `damage[].hp` | 0 – 544 | 中位 24 |
| `grenades[].x` | −2281 – 1517 | 和玩家坐标同量级 |
| `grenades[].z` | −59 – **1082** | **飞行中可以很高**，别假设贴地 |
| `sound` 的 `fire` | **215.8 次/回合** | 全场 3885 |
| `sound` 的 `jump` | **23.1 次/回合** | 全场 415，和 `jump` 列总数一致 |
| `sound` 的 `footstep` | **16.2 次/回合** | 全场 291，稀有且语义可疑（§4.3） |

#### 火焰 / 烟雾

| 指标 | min | p50 | max | 备注 |
|---|---|---|---|---|
| 每快照燃烧格数 | 1 | 10 | **16** | 硬上限 16（`m_firePositions.0000…0015`） |
| 每个火的快照数 | 9 | 13 | 19 | 只在"变了"的时候记一条 |
| **火的区间时长** | 2.41 s | **5.50 s** | **7.03 s** | 燃烧瓶 ≈7.0 s、燃烧弹 ≈5.5 s（**游戏常量**） |
| 火焰格子离 origin | 0 | 86 HU | 164 HU | 格子坐标是**围绕 origin** 给的 |
| `normal` 的模长 | 1 | 1 | 1 | **精确为 1**，不用再归一化 |
| 同一 tick 最多几处火 | — | — | **7** | 别假设只有 1 处 |
| **烟雾 detonate→expire** | **22.062 s** | 22.062 | 22.062 | **游戏常量**：63 个配对区间**全部精确相等** |
| 每个烟的快照数 | 10 | 203 | 203 | |
| 单条 journal 字节 | 0 | 1734 | 2741 | 解析出来就是这个大小 |
| **烟盖住的体素数** | 765 | 1178 | 1801 | 网格 32³ = 32768 格，**实占 2–5%** |
| 烟的网格最高 z 下标 | 20 | 27 | 27 | 网格局部坐标 0–31 |

> "火的区间时长"是**实体存续**区间（2.41–7.03 s），"真的在烧"只会更短：
> 按快照里 `burning=True` 量出来是 0.14–6.92 s，中位 1.17 s ——
> 因为很多火还没烧完回合就结束了，或者被烟/雷灭掉。

#### 怎么用这些数

```python
# 1) 归一化：位置按地图、速度按武器上限，别用全局 min-max
X = recipes.player_tensor(rnd, nan="interp")
X[..., :3] /= 3000.0          # 位置：HU，量级 1e3
X[..., 5:8] /= 250.0          # 速度：武器上限，量级 1e2
# 2) 稀疏的 0/1 列（shots/footsteps/jump/land）原样喂，别缩放

# 3) 数据体检：这些数一超，就是数据坏了而不是模型坏了
assert np.nanmax(np.abs(X[..., 8])) <= 1152.0 * 1.5   # vz，实测上限 1152
assert np.nanmax(np.abs(X[..., 0])) < 1e5             # x，地图量级
```

---

## 2. 三个模块的分工

```
codec     存 / 取        dict  <->  磁盘上的字节
recipes   取用（重点）    从 dict 里把你要的东西拿出来，变成数组/几何
voxel     底层          烟雾日志的字节级解析（recipes 已经包好了，一般不用直接碰）
sidechannel 底层        Go 辅助程序的调用（你不需要碰）
```

日常只 import 前两个：

```python
from demo_extract import codec, recipes
```

---

## 3. `codec` —— 保存和加载

### `codec.save(data, path, level=19)` / `codec.load(path, verify=True)`

把 dict 编码写盘 / 读盘解码。`load` 回来和 `save` 之前**逐位相同**。

```python
codec.save(data, "match.csz")     # 7.0 MB
data = codec.load("match.csz")    # 约 4.3 秒
```

`level` 是 zstd 压缩等级。实测 19 和 22 的差别可以忽略（0.749 vs 0.746 MB），
所以默认 19。想快一点可以降到 10 左右，文件会大一些。

### `codec.dumps(data, level=19)` / `codec.loads(blob, verify=True)`

同上，但不碰磁盘，直接 bytes。

### `codec.equal(a, b, path="$")`

逐位比较两个结构，返回 `(是否相同, 不同的位置)`。

```python
codec.equal(data, codec.load("match.csz"))    # -> (True, '$')
codec.equal(data, other)                       # -> (False, '$.rounds[7].players[7].x[1]')
```

NaN 会被当成相等（都是 NaN 就算相同），因为这是"数据一致"的比较，
不是 IEEE 语义的比较。自己写 `a == b` 会在有 NaN 的地方误报。

### `codec.save_json(data, path, indent=1)` / `codec.load_json(path)`

存成人类可读的 JSON。**会大很多**（同一份数据 **183 MB** vs 7.0 MB，26 倍），
而且 `bytes` 字段（烟雾体素）要 base64，所以只适合调试。

### `codec.raw_blob(data)`

编码成未压缩的字节流，用来分析"每一段占多少空间"。调试用。

### `codec.VERSION` / `codec.DEFAULT_LEVEL`

```python
codec.VERSION        # 2     T_* 编码标签的版本；不是 schema 版本，别和数据格式搞混
codec.DEFAULT_LEVEL  # 19    默认 zstd 等级（save / dumps 都用它）
```

> 数据格式版本看 `data["format"]`（现在是 `"cs2.demo.v7"`），
> 编解码器版本看 `codec.VERSION`。两个独立演进：加一个字段只需要改前者。
>
> `codec` 里的 `T_*` 编码标签（`T_INT` / `T_F32D2` / `T_TABLE` …）
> 属于实现细节，正常用不到；只有 `tests/test_roundtrip.py` 里的标签覆盖测试会碰。

---

## 4. `recipes` —— 取用

### 4.1 定位回合和采样点

#### `recipes.round_by_id(data, round_id) -> dict`

**按游戏回合号取一个回合**。见 §1.2 为什么不能用下标。

```python
rnd = recipes.round_by_id(data, 7)
rnd["ticks"][:3]        # [45130, 45131, 45132]
```

**找不到就抛 `KeyError`，永远不返回 `None`。** 有两种"找不到"，报错信息不一样：

```python
# (a) 这个回合存在，但提取时失败了 —— 它根本不在文件里
recipes.round_by_id(data, 7)
# KeyError: round 7 was skipped during extraction:
#           no round_freeze_end, so the round has no time origin

# (b) 这个 id 压根不存在（比如你传了 99）
recipes.round_by_id(data, 99)
# KeyError: no round 99 in this file; it has [0, 1, 2, ..., 17]
```

**它不返回"空回合"，也不返回相邻回合。** 想要不抛异常的写法：

```python
ids = recipes.round_ids(data)              # 实际有哪些 id，升序
if 7 in ids:
    rnd = recipes.round_by_id(data, 7)
```

#### `recipes.round_ids(data) -> list[int]`

**文件里实际存在的回合号**，升序。这就是 `round_by_id` 能接受的合法输入。

```python
recipes.round_ids(data)    # test.dem: [0, 1, 2, ..., 17]（18 个）
# 跳过的回合留下一个洞，比如 [0, 1, 2, 3, 4, 5, 6, 8, 9, ...]（缺 7）
```

**注意它不是 `range(n)`。** 被跳过的回合留下一个洞，
而且 `data["rounds"][i]["id"]` 和 `i` 不再相等 —— 这就是不能用下标的原因。

#### `recipes.skipped_rounds(data) -> list[dict]`

**哪些回合没能提取出来、为什么。**

```python
recipes.skipped_rounds(data)
# -> []  干净的文件
# -> test.dem 实际就是：[{'id': 18,
#        'reason': 'no round_freeze_end, so the round has no time origin'}]
```

可能的 `reason`：

| reason | 意思 |
|---|---|
| `no round_freeze_end, so the round has no time origin` | 没有冻结结束事件，这个回合的时间轴没有起点，`round_seconds` 无从算起 |
| `no sampled ticks fell inside this round` | 采样间隔太粗，这个回合里一个采样点都没落上（比如 `--interval 5` 碰上短回合） |
| `XxxError: ...` | 组装这个回合时抛了异常，后面跟着原始信息 |

这个函数对**旧文件**也兼容（早期版本 `skipped_rounds` 是纯 id 列表，
会被规整成 `{'id': ..., 'reason': 'unspecified'}`）。

#### `recipes.sample_index(rnd, tick) -> int`

**某个 tick 是第几个采样点**。不在采样点里返回 `-1`（不抛异常，方便判断）。

```python
recipes.sample_index(rnd, 45130)    # 0
recipes.sample_index(rnd, 45131)    # 1
recipes.sample_index(rnd, 45129)    # -1  ← 不是采样点
```

想拿"离这个 tick 最近的那个采样点"就自己用 `bisect`，这个函数只做精确匹配。

#### `recipes.ticks_of(rnd) -> np.ndarray[int64]`

把 `rnd["ticks"]` 转成 numpy 数组。

#### `recipes.seconds_of(rnd) -> np.ndarray[float64]`

把 `rnd["round_seconds"]` 转成 numpy 数组。`[0]` 恒为 `0.0`。

#### `recipes.iter_samples(rnd) -> Iterator[(下标, tick, 秒)]`

按顺序遍历一个回合的所有采样点。写"逐帧处理"的循环时用它，
比手动 `zip` 清楚。

```python
for i, tick, sec in recipes.iter_samples(rnd):
    ...   # i=0, tick=19574, sec=0.0 / i=1, tick=19575, sec=0.015625 ...
```

#### `recipes.round_winner(rnd) -> str`

返回 `'T'` / `'CT'` / `''`。就是 `rnd["winner"]` 的安全版本。

---

### 4.2 玩家状态 → 张量

#### `recipes.player_tensor(rnd, features=PLAYER_FEATURES, dtype=np.float32, nan="keep") -> np.ndarray`

**一个回合的玩家状态，堆成一个 `(T, P, F)` 数组。** 这是训练的主输入。

* `T` = 这个回合的采样点数
* `P` = 10（顺序和 `data["players"]` 一致）
* `F` = 你选的字段数，默认 16

```python
X = recipes.player_tensor(rnd)
X.shape                  # (3423, 10, 16)   ← 回合 3 每 tick 采样
X[0, 0, :]               # 第 0 个采样点、0 号玩家的 16 个特征
X[:, :, 0]               # 所有采样点、所有玩家的 x 坐标，形状 (T, 10)
```

默认的 16 列，顺序**固定**（用 `recipes.PLAYER_FEATURES` 取）：

```
下标:  0   1   2    3     4      5   6   7    8    9      10     11
字段:  x   y   z   yaw  pitch   vx  vy  vz   hp  armor  flash  flash_alpha
下标: 12    13     14     15
字段: weapon place shots footsteps
```

`features=` 可以换成任意**标量**字段的组合。比如你想把速度大小、头盔、拆弹器也放进去：

```python
X = recipes.player_tensor(rnd, features=(
    "x","y","z","yaw","pitch","vx","vy","vz","v",
    "hp","armor","helmet","defuser","alive",
))
X.shape                  # (3423, 10, 14)
```

`helmet`/`defuser`/`alive` 是 bool，会被转成 0.0/1.0。
**`inventory` 和 `spotted` 不能放进来**（它们是变长列表）—— 用 §4.4 的两个 multi-hot。

v6 的三列新信号（`on_ground` / `jump` / `land`）也可以放进来，直接用一个现成的常量：

```python
X = recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V6)
X.shape                  # (3423, 10, 19)  ← 老 16 列 + on_ground + jump + land
```

`PLAYER_FEATURES_V6` 的前 16 项**逐项等于** `PLAYER_FEATURES`，顺序没变，
所以"用 16 列训练"和"用 19 列训练"的代码可以共用同一套下标。

`nan=` 控制缺值怎么处理：

| 值 | 行为 |
|---|---|
| `"keep"`（默认） | 保留 NaN，让你自己决定 |
| `"zero"` | NaN → 0 |
| `"interp"` | 沿时间轴线性插值；缺口在开头就取后面第一个已知值往前填 |

```python
X = recipes.player_tensor(rnd, nan="interp")   # 训练时一般用这个
```

#### `recipes.player_tracks(rnd, features=PLAYER_FEATURES) -> dict[str, np.ndarray]`

和上面同一份数据，但**每个字段单独一个 `(P, T)` 数组**。
适合画图（一条线一个玩家）或者逐字段做归一化。

```python
tr = recipes.player_tracks(rnd, ("x", "y", "z"))
tr["x"].shape            # (10, 3423)   ← 10 个玩家 × 时间
tr["x"][3]               # 3 号玩家的 x 轨迹
```

#### `recipes.alive_mask(rnd) -> np.ndarray[bool]`

`(T, P)` 的布尔数组：这个采样点这个玩家活着吗。

```python
m = recipes.alive_mask(rnd)
m[0]                     # 回合开始时 10 个都是 True
m.sum(axis=1)[:5]        # 前 5 个采样点各有多少人活着
```

训练时**通常要拿它做 mask**，因为死人还留在原地、速度是 0，
不 mask 的话模型会把"尸体"当成站着不动的人。

#### `recipes.yaw_sincos(rnd) -> np.ndarray`

`(T, P, 2)`，yaw 的 `[sin, cos]`。

**为什么要这个**：yaw 是 ±180 度的角度，179° 和 -179° 实际只差 2°，
但数值上差 358。模型没法跨过这个跳变去插值。换成 sin/cos 就没有跳变了。

```python
yc = recipes.yaw_sincos(rnd)      # (3423, 10, 2)
yc[..., 0]                        # sin
yc[..., 1]                        # cos
```

pitch 不需要（它不环绕，范围只有 -89..89）。

#### `recipes.to_meters(arr, features=PLAYER_FEATURES) -> np.ndarray`

把张量里的**长度类字段**从 Hammer 单位换成米，其他列不动。

```python
Xm = recipes.to_meters(X)         # x y z vx vy vz 乘 0.0254，其余 10 列原样
```

会改的字段是 `x y z vx vy vz`（`recipes.HU_FEATURES`）。
返回 float64 的**新数组**，不改原数组。

```python
recipes.HU_TO_METERS   # 0.0254
recipes.HU_FEATURES    # ('x','y','z','vx','vy','vz')
```

`0.0254` 就是 cs-net 用的 `SCALE`。1 Hammer 单位 = 1.905 cm。
（注意：从 v6 起 `PLAYER_FEATURES_V6` 多了三列，但 `HU_FEATURES` 没变——
`on_ground` / `jump` / `land` 不是长度，不该被缩放。）

#### `recipes.nan_report(data_or_round) -> dict[str, int]`

**哪些字段有缺值、缺多少个。** 参数给整个 `data` 或单个 `rnd` 都行。

```python
recipes.nan_report(data)     # {'v': 360, 'vx': 360, 'vy': 360, 'vz': 360}
```

实测**只有 4 个速度列会缺**（`v` 和 `vx/vy/vz`），其他列从不缺。形态很规整：
总是**连续两个采样点**、**10 个人同时缺** —— 是服务器卡了一下，
**和死亡无关**（死人也照样报速度）。全场占比 <1%。
回合的第一个采样点（`round_freeze_end`）通常就在这个缺口里。

> ⚠️ **它扫的是每个数值列，不是 `PLAYER_FEATURES`。** 这两者不一样：
> `v`（速度大小）**不在**那 16 列里（16 列里只有分量 `vx vy vz`），
> 但它和分量在同一批采样点上缺。以前这里只遍历 `PLAYER_FEATURES`，
> 结果是**静默漏报 `v`**（实测该报 360 却报 0）。现在 `inventory` / `spotted`
> 这类"列表的列表"会被跳过，其余数值列一律计入 —— 包括你自己塞进去的列。

---

### 4.3 `footsteps` 是坏的，别拿它当脚步声

> 这一节解释一个**会误导训练**的字段。结论先给：
> **`footsteps` 不是跑动脚步声，它几乎只在玩家起跳/落地时触发。**
> v6 已经用 `on_ground` / `jump` / `land`（§4.11）取代了它，
> 所以这一节现在主要是"为什么不要用它"。

它来自 CS2 的 `player_footstep` 游戏事件。实测（`test.dem`，18 回合）：

| 检验 | 结果 |
|---|---|
| 全场事件数 | **360**（落在回合内 291） |
| 按跑动距离 ÷ 步幅应有的数量 | **~14,336** → 差 **50 倍** |
| demoparser2 数出来 | 360 |
| demoinfocs（独立实现）数出来 | **360** ← 所以不是解析器漏了 |
| 触发时的速度 | 最低 135，中位 **245** HU/s（只在跑动时） |
| 与"同一玩家起跳"的关系 | **96.9%**，基线 **10.4%** → **9.4 倍富集** |
| 5 秒窗口内触发的不同玩家数 | 平均 **1.22**，同期平均 **6.35** 个玩家在跑动 |

所以它既**不是**"道具投掷的声音"（投掷时速度不固定，而实测 100 HU/s 以下一次都没有），
也**不是**正常跑动脚步声 —— 它是跳跃/落地那一下的声音。

#### 那"有没有发出脚步声"怎么办？—— 没有函数，只有原料

**v6 起这里不再提供任何阈值函数。** 曾经有一个 `audible_movement(rnd, threshold=167.0)`
和配套的 `RUN_SPEED_HU` / `WALK_SPEED_RATIO` / `RUN_SPEED_MIN_HU` / `RUN_SPEED_MAX_HU`，
**已经全部删除**，取它们会 `AttributeError`（测试里有断言守着）。

删掉的理由：观测是真的，**阈值是编的**。

| | 性质 |
|---|---|
| 跑速台阶 202…252、走速台阶 102…132、走/跑比中位数 0.515 | **实测**（这份 demo，按武器取众数，空中样本已剔除） |
| "阈值 = 空带中点 = 167.0" | **查无出处** —— 既不是游戏常量，也不是数据里读出来的，是拍的一个让两边余量相等的数 |

把这个数封进 API，等于把一个人的选择伪装成事实，下游还会当它是真值。
所以：**原材料留下，编出来的常数拿掉。**

正确的用法是直接给模型原始信号（§4.11）：

```python
X = recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V6)   # (T, 10, 19)
# 里面同时有 velocity(vx,vy,vz) 和 on_ground / jump / land
```

模型自己会学出那条线，而且会**按武器、按状态分开学**（走 / 蹲走 / 空中 / 被打飞），
比一个全局常数准。

非要一条规则的话，把它当**超参**用你自己的下游指标去调，别从工具里接一个常数。
实测观测（想自己定规则的话可以参考）：跑速 202/212/222/227/232/237/242/247/252 HU/s
（AWP 最慢、刀最快），走速 102/112/117/122/127/132 HU/s，走/跑比中位数 0.515。
注意"走 ≤ 132、跑 ≥ 202"这条空带是这份 demo 的观测，**换武器/版本要重新量**。

#### `recipes.footstep_warning(data_or_round) -> str | None`

返回一段说明当前数据里 `footsteps` 有多不可靠的文字（含实际事件数），
数据里没有 footstep 事件时返回 `None`。

```python
print(recipes.footstep_warning(data))
# `footsteps` holds 291 events for this data. The CS2 `player_footstep` game
# event fires around jump takeoffs and landings, not on ordinary running
# footsteps ... For movement noise use velocity together with `on_ground` ...
```

---

### 4.4 物品和视野 → multi-hot

#### `recipes.inventory_multihot(rnd, n_items=None) -> np.ndarray[bool]`

`(T, P, n_items)`：每个人每个时刻身上有哪些东西。

`inventory` 是变长列表，塞不进数值张量，所以给它一个独立形状。

```python
inv = recipes.inventory_multihot(rnd)     # (3423, 10, 28)
inv[t, p]                                 # 这个时刻 p 号玩家的 28 维 0/1 向量
np.flatnonzero(inv[t, p])                 # -> [8, 13, 27]
[list(data["weapons"])[i] for i in np.flatnonzero(inv[t, p])]
# -> ['Flashbang', 'Glock-18', 'Shadow Daggers']
```

编号索引进 `data["weapons"]`。

> ⚠️ **跨回合拼数据时必须显式传 `n_items`。** 默认值是按"这个回合里实际出现过的
> 最大编号"推的，而**不同回合推出来的宽度不一样** —— 实测同一份 demo：
> 大部分回合是 28，第 9、10 回合是 26，第 11 回合是 25。
> 想要全文件统一宽度就用 `n_items=len(data["weapons"])`（这份是 32）。
>
> ```python
> W = len(data["weapons"])
> inv = recipes.inventory_multihot(rnd, n_items=W)   # 恒定 (T, 10, 32)
> ```

**死人身上什么都没有，所以整行全 False 是正常的**（实测回合 0 里
17,613 / 38,820 行是全 False）。

#### `recipes.spotted_multihot(rnd) -> np.ndarray[bool]`

`(T, P, P)`：`[t, i, j]` 表示"t 时刻，**j 能看到 i**"。

`spotted` 记录的是**能看到这个玩家的敌人**（§1.6），所以 `spotted[i] ∋ j` ⟺ j 看得见 i。
`spotted_multihot` 是照这个顺序直接铺开的，于是 `[t, i, j]` 是"**j 看 i**"，
**不是**"i 看 j" —— 想要"i 能看见谁"要取**列** `sp[t, :, i]`（或
`sp.transpose(0, 2, 1)[t, i]`）。

方向是在 de_inferno 上用几何独立验过的：把 `spotted[i]` 里单向出现的
（`i ∉ spotted[j]`）(i, j) 对挑出来，量 j 的朝向和 j→i 连线的夹角 ——
471 对里 94.9% 在 20° 以内，而 i 的朝向只有 52.2% 在 20° 以内。确实是 **j 在看 i**。

```python
sp = recipes.spotted_multihot(rnd)        # (3423, 10, 10)
sp[t, :, 3]                               # t 时刻，谁看得见 3 号（一个列）
sp[t, 3]                                  # t 时刻，3 号看得见谁 —— 见上面，别搞反
sp.sum(axis=(1, 2))                       # 每个时刻有多少对可见关系
```

对角线永远是 False（`spotted` 里不含自己，实测确认）。

> ⚠️ 前端可视化里的眼睛图标（`visualizer/README.md` §4.8）用的是同一条反向关系：
> 选中 S 之后要遍历敌人 E、查 `S ∈ spotted[E]`，也就是 `sp[t, E, S]`。

---

### 4.5 单 tick 的完整快照

#### `recipes.tick_sample(data, round_id, tick, smoke_solid=True) -> dict`

**一个 tick 的全部信息**，已经组装好。渲染一帧、或者给决策模型喂一步，
要的就是这个。

`tick` 必须是采样过的 tick，否则抛 `KeyError`（先用 `sample_index` 查）。

```python
s = recipes.tick_sample(data, round_id=3, tick=21285)

s["players"]      # (10, 16)  这一 tick 的玩家状态，NaN 已用 interp 填好
s["alive"]        # (10,)     bool
s["velocity_ok"]  # (10,)     bool —— 哪几个人的速度本来是缺的
s["yaw_sincos"]   # (10, 2)
s["on_ground"]    # (10,)     bool —— v6 文件才有；老文件这个键不存在
s["jump"]         # (10,)     int  —— v6 文件才有
s["land"]         # (10,)     int  —— v6 文件才有
s["crouch"]       # (10,)     f32  —— v7 文件才有，0..1 连续
s["crouching"]    # (10,)     bool —— v7 文件才有，碰撞盒已跳到 54
s["eye_height"]   # (10,)     f32  —— v7 文件才有，眼高（相对脚底，HU）
s["fire"]         # list，每个是一处正在烧的火（见 §4.6）
s["smoke"]        # {"grids": [...], "points": [...], "meta": [...]}（见 §4.7）
s["meta"]         # 见下
```

老文件（`cs2.demo.v5` 及更早）**不会**返回 `on_ground` / `jump` / `land` 三个键，
而不是返回一堆 `False` —— 用 `"on_ground" in s` 就能判断文件版本。
同理 v6 文件不会返回 `crouch` / `crouching` / `eye_height`，
用 `"crouch" in s` 判断是不是 v7（见 §4.12）。

`s["meta"]`：

```python
{'tick': 21285, 'seconds': 26.734375, 'map': 'de_ancient', 'round': 3,
 'teams': ['T','T','CT',...], 'winner': 'T', 'sample_index': 1711,
 'samples_in_round': 3423, 'n_players': 10}
```

注意 `s["players"]` 用的是 `nan="interp"`（和 `player_tensor` 的默认 `"keep"` 不同），
因为这是给"直接用"的场景；而 `velocity_ok` 告诉你哪几个值本来是缺的。

---

### 4.6 火焰

#### `recipes.active_fire(rnd, tick) -> list[dict]`

**这一 tick 所有还在烧的火**，每个都已经拆成世界坐标数组。

```python
for c in recipes.active_fire(rnd, 21285):
    c["points"]        # (N, 3) float32 世界坐标 —— 直接可以画
    c["normals"]       # (N, 3) float32 每格表面法线（单位向量）
    c["parents"]       # (N, 3) float32 每格的"从哪烧过来"坐标
    c["parent_index"]  # (N,)   int32   蔓延树：第 i 格来自第 parent_index[i] 格，-1 是根
    c["burning"]       # (N,)   bool    这一格还在烧吗
    c["entity_id"]     # 实体 id
    c["type"]          # 'molotov' / 'incendiary'
    c["origin"]        # [x,y,z] 起火点
```

最多 16 格。实测一例：`entity_id=687`，`molotov`，
`origin=[-1157.88, 600.21, 73.03]`，16 格全在烧，
`parent_index=[-1,0,0,1,2,3,1,2,6,0,1,2,4,6,10,0]` —— 一棵正常的蔓延树。

**`burning` 会变 False**：火快灭的时候部分格子先熄，`count` 不变但 `burning` 变了。
所以别只看 `count`。

#### `recipes.fire_state_at(inst, tick) -> dict | None`

单个火焰实例在某 tick 的**原始快照**（没拆成数组）。
不在存活期（`first_tick`..`last_tick` 之外）返回 `None`。

```python
f = rnd["fire"][0]
recipes.fire_state_at(f, 1000)    # None（还没开始烧）
recipes.fire_state_at(f, 21285)   # {'tick': 21285, 'count': 16, 'pos': [...], ...}
```

内部就是"取 `tick` 之前最后一个变化快照"。**这是精确的** ——
因为提取时在每个状态变化的采样点都存了一份，所以"最后一个不晚于 tick 的快照"
就是那一 tick 的真实状态（已用独立的 demoinfocs 逐帧探针交叉验证过）。

#### `recipes.fire_cells(state) -> dict[str, np.ndarray]`

把 `fire_state_at` 返回的那一个快照拆成数组。
`active_fire` 内部就是调它，你一般不用直接调，除非要自己遍历快照。

展平的 `pos`/`normal`/`parent` 是 `3*count` 个 float，这个函数把它 reshape 成 `(N,3)`，
并且**顺便算出 `parent_index`**（原始数据里 `parent` 存的是坐标不是下标，
所以这里是按"坐标距离 < 1 HU"反查出来的）。

#### `recipes.fire_volume_at(rnd, tick, resolution=32.0) -> dict`

**把火场体素化成一个占据网格**，而不是点云。

```python
fv = recipes.fire_volume_at(rnd, 21285, resolution=32.0)
fv["grid"]        # (nx, ny, nz) bool
fv["origin"]      # 网格原点的世界坐标
fv["resolution"]  # 体素边长，HU
```

`resolution` 是体素边长（Hammer 单位）。**只统计 `burning=True` 的格子**。
没有火的时候返回空网格（`shape` 是 `(0,0,0)`）。

---

### 4.7 烟雾

#### `recipes.smoke_grid_at(inst, tick, solid=True) -> np.ndarray | None`

**一个烟在某 tick 的 32×32×32 占据网格。** 不在存活期返回 `None`。

```python
g = recipes.smoke_grid_at(inst, 21285)
g.shape                  # (32, 32, 32)
g.sum()                  # 1291   ← 占据多少个体素
np.argwhere(g)           # 体素的网格坐标 (N, 3)，整数 0..31
```

`solid=True`（默认）返回**实体体积**（把边界围住的内部也填上）；
`solid=False` 只返回**边界体素**（日志里明确点名的那些）。

两者差得**很多**，别以为只差几个格子：一颗满云的点名集合 1787 格 → 凸包 5459 格
（**3.05×**）。原因见 §6：掩码是稀疏采样，云内部本身就有成片空洞，点名格只有 6.5%
是 6 邻域全占。所以**不用在两者之间随便挑**：渲染和遮挡都用 `solid=True`（同一个
体积），`solid=False` 只用于调试对照 —— 按它判遮挡会漏掉 45.8% 的"被烟挡"。

#### `recipes.smoke_points_at(inst, tick, solid=True) -> np.ndarray | None`

同上，但直接返回**世界坐标** `(N, 3)`，喂渲染器或点云网络。

```python
p = recipes.smoke_points_at(inst, 21285)
p.shape                  # (1291, 3)
p.min(axis=0), p.max(axis=0)   # 世界坐标包围盒
```

**需要 `inst["detonation"]`。** 没有就抛 `ValueError` —— 因为
`world = detonation + (grid - 15.5) * 12.0`，没有引爆点就没法定位。

#### `recipes.active_smoke(rnd, tick, solid=True) -> dict`

**这一 tick 所有还活着的烟。**

```python
sm = recipes.active_smoke(rnd, 21285)
sm["grids"]    # list，每个 (32,32,32) bool
sm["points"]   # list，每个 (N,3) 世界坐标
sm["meta"]     # list，每个 {'entity_id':710, 'detonation':[...], 'voxels':1291}
```

一个 tick 上可能同时有好几个烟（实测某一 tick 有 2 个：1291 + 1164 个体素）。
网格是空的（`not g.any()`）会被跳过。

#### `recipes.grid_to_world(grid_xyz, detonation) -> np.ndarray`

网格整数坐标 → 世界坐标。想自己挑体素（比如只要下半部分）时用。

```python
g = recipes.smoke_grid_at(inst, tick)
low = np.argwhere(g)[:, 2] < 20              # 只要低处的体素
w = recipes.grid_to_world(np.argwhere(g)[low], inst["detonation"])
```

换算公式就是 `world = detonation + (grid - 15.5) * 12.0`，每个体素 12 HU。

#### `recipes.smoke_lifetime(rnd_or_data, entity_id) -> dict | None`

按实体 id 找一个烟雾实例（给单个 `rnd` 或整个 `data` 都行）。

```python
inst = recipes.smoke_lifetime(data, 710)
inst["first_tick"], inst["last_tick"]   # (19681, 21452)  存活区间
```

注意**实体 id 会被复用**（同一个 id 先后装了两个不同的烟），
这个函数返回**第一个**匹配的。要精确就用 `round["smoke"]` 自己筛。

---

### 4.8 训练窗口

#### `recipes.iter_windows(data, length, stride=1, round_id=None) -> Iterator[(回合号, 起, 止, 起始tick)]`

**把时间线切成定长窗口。** 窗口**不会跨回合**，所以一段序列一定在同一个回合里。

```python
for rid, a, b, tick0 in recipes.iter_windows(data, length=64, stride=32):
    X = recipes.player_tensor(recipes.round_by_id(data, rid))[a:b]   # (64, 10, 16)
```

* `length` = 窗口长度（采样点数）
* `stride` = 相邻窗口起点差多少（`stride < length` 就有重叠）
* `round_id=` 只切某一个回合
* 某个回合比 `length` 还短就跳过（不产窗口）

`(起, 止)` 是左闭右开，直接拿去切片：`X[a:b]`。

---

### 4.9 概览

#### `recipes.summary(data) -> str`

一句话描述整个文件。

```python
print(recipes.summary(data))
# test.dem on de_ancient: 18 rounds, 10 players, 74003 samples,
# 58 infernos, 82 smokes (sampling interval=None, stride=None)
```

---

### 4.10 几个必须知道的坑

**⚠️ demoparser2 的 `velocity*` 不能用"跳着采"的 tick 列表读（上游 bug）。**
只要传给 `parse_ticks(ticks=...)` 的列表**不是一整段连续 tick**，
`velocity / velocity_X / velocity_Y / velocity_Z` 就会返回**别的 tick 上的速度**。
实测 stride=2 时已经有 **84%** 的采样点是错的；`X/Y/Z`、`m_fFlags`
以及其余所有字段都完全正确，**只有这 4 个坏**。

**根因**：`collect_velocity` 取的是 `output` 里**最后两行**位置做差再乘 64
（`(last * 64.0) - (prev * 64.0)`）。跳着采时这两行隔着一个 stride，
但仍然乘 64，所以速度被放大成约 **stride 倍**——2 Hz（stride 32）的文件里
出现 ±8000 HU/s ≈ 250 × 32 就是这么来的。数值上，`velocity_X` 在
stride = 2 / 8 / 32 下都 **100%** 等于 `(X(上一个保留行) − X(上上个保留行)) × 64`。
tick 列表只有 1~2 个元素时速度列直接消失，也是因为凑不出两行。

**升级修不掉它**：PyPI 最新版就是 0.42.0（= 我们装的这个）；
上游 [LaihoE/demoparser PR #353](https://github.com/LaihoE/demoparser/pull/353)
合进了 `main` 但未发版，而且它是治内存的，作者明确说 `parse_ticks` 的过滤行为
"exactly as before"（补历史的判断条件是 `event_with_velocity`，而 `parse_ticks`
不带事件，所以为 `false`）。

`extract.py` 的对策是把速度拆成一次**不带 `ticks` 参数**的解析
（`VELOCITY_PROPS` / `STATIC_PROPS`），再按 `(tick, steamid)` 合并回去：

```python
extract.VELOCITY_PROPS   # ['velocity','velocity_X','velocity_Y','velocity_Z']
extract.STATIC_PROPS     # PLAYER_PROPS 去掉上面四个
```

自检：`v` 是**水平速度**，所以 `v == hypot(vx, vy)` 必须处处成立；
而且真人不可能有 `|vz| > 1200 HU/s` 的时候。修之前这两条都会挂
（`|vz|` 能到 **22,546**）。

**`bomb_position` 不是每 tick 的精确位置，也不是 C4 实体的位置。** 它是"最后已知的
C4 携带者位置"（和 cs-net 语义一致）：`_bomb_carrier` 逐 tick 扫 `inventory` 里有没有
`C4 Explosive`，没人持有的 tick 由 `_bomb_positions` 沿用上一个位置。

- **拿在手上**：和那个人的 `x/y/z` 逐帧完全相等（4 份 demo 上 100.0000%）。
- **安放之后**：冻结在安放者的位置。这时冻结是**对的**（C4 不再动），离蹲着拆包的
  人 ≤ 80 HU——这个残差就是"玩家原点 vs 炸弹实体"的固定差（13 次 defuse 实测，
  中位数 25~57 HU）。
- **掉在地上**（`bomb_dropped`，本仓库 4 份 demo 占 **13.7%~24.2%** 的采样帧）：
  坐标只是**掉落的那个点**，不是 C4 的落点。C4 脱手后带着玩家的速度滑/掉一段，
  所以"掉落 → 被某人捡起"的 95 次里，冻结点和捡包人（必须站在 C4 上）差
  **中位数 ~175 HU、p90 ~360 HU、最多 456 HU**，且偏移方向就是掉落者当时的速度
  方向（余弦 0.96~0.98）；dz 有时 0（平地滑），有时 −70~−130 HU（掉到台阶下）。
  这段时间坐标一个数都不变（最长实测冻结 61 s）。
- 回合刚开始、还没人捡起来的时候是 `None`。

C4 是独立的 `C_C4` 实体，demoparser 的公开 API 读不到它的世界坐标
（`parse_grenades` 不暴露、`bomb_planted` 事件连 x/y/z 都没有），所以这套"跟着
携带者走"的做法不是省事，是当前唯一拿得到的近似。要判"现在是不是掉在地上"，
`bomb_dropped` 那一列在 4 份 demo 上有 95.3%~100% 的帧和"没人持有"一致
（偶尔整回合对不上），但不是 100%。

**`v` 是水平速度，不是 3D 速度。** 修掉上面那个 bug 之后，
`v == hypot(vx, vy)` 精确成立（最大偏差 2e-5，就是 f32 舍入），
但 `v != |(vx, vy, vz)|`（最大差 728 HU/s）。它是独立信息，要么保留要么丢掉，
但别用 3D 模长去"推导"。

**`round_seconds` 不能从 tick 算。** 见 §1.8。

**每个回合的时间轴从自己的 `round_freeze_end` 开始。** 赛前热身（这份 demo 有 1181 个 tick）
已经被裁掉了，所以"采样点占全部 tick 的 73%"。

---

### 4.11 地面接触与空中事件（v6）

给模型"这一步响不响"的**原始信号**，而不是替它写死的规则。

#### `recipes.PLAYER_FEATURES_V6`

```python
recipes.PLAYER_FEATURES      # 16 个，和 cs-net v2 完全一致
recipes.GROUND_FEATURES      # ('on_ground', 'jump', 'land')
recipes.PLAYER_FEATURES_V6   # 上面两个拼起来，19 个
```

`player_tensor(rnd, recipes.PLAYER_FEATURES_V6)` → `(T, P, 19)`。
老的 16 列一个没动、顺序也没变，所以 v6 文件可以直接喂给老的训练侧。

#### `recipes.on_ground_mask(rnd) -> np.ndarray[bool]`

`(T, P)`：这一采样点是否踩在地面上。

来源是 `CCSPlayerPawn.m_fFlags` 的 **bit 0（`FL_ONGROUND`）**。
选它是因为它和"`m_hGroundEntity` 是不是有效句柄"在
**727,649 个采样点上 100.0000% 一致**，而标志字只是一个整数。

```python
og = recipes.on_ground_mask(rnd)     # (3423, 10) bool
```

> ⚠️ 这是**原始标志**，不等于"活着站在地上"。死人的尸体只要贴地也报 `True`。
> 要"活人站在地上"就写 `alive & on_ground`。
> 这份 demo：活着的采样点 523,183 个，其中 **17,818 个（3.41%）在空中**。

自检：空中时 `|vz|` 平均 **142 HU/s**，踩地时只有 **12 HU/s**，差 **11.6 倍**。

#### `recipes.jump_mask(rnd) -> np.ndarray[int]`

`(T, P)`：本采样窗口内的**起跳**次数。来自 CS2 `player_jump` 事件，**精确**。
默认每 tick 采样时是 0/1；粗采样时可能 >1（一个窗口里跳了好几次）。

#### `recipes.land_mask(rnd) -> np.ndarray[int]`

`(T, P)`：本采样窗口内的**落地**次数。

> ⚠️ **这是推出来的，不是事件。** CS2 没有落地事件。定义是
> "上一个采样点活着且在空中，这一个采样点活着且踩地"。
> 默认每 tick 采样时是精确的；`--interval 0.25` 这种粗采样下，
> 一个短跳可能整个发生在两个采样点之间，那就看不见了
> （`jump` 不受影响，它是事件，永远精确）。
>
> 这份 demo 实测：起跳 **415** 次、落地 **451** 次。多出来的 36 次是
> 走下台阶/边缘掉落 —— 没有按键起跳，所以只有落地没有起跳。

#### 老文件会明确报错，不会假装有数据

```python
recipes.on_ground_mask(old_v5_round)
# KeyError: this file has no `on_ground` column — it was written in a format
#           older than cs2.demo.v6; re-extract the demo to get ground contact
```

`recipes.tick_sample()` 对 v6 文件会多返回 `on_ground` / `jump` / `land` 三个键，
对老文件则不会返回它们（而不是返回一堆 `False`）。


### 4.12 下蹲与眼高（v7）

> **一句话**：demo 里能精确拿到的是**下蹲量**（连续 0..1）和**碰撞盒状态**（阶跃），
> 但**站立足眼高的绝对值拿不到** —— 只能从文档取。这两个量是**不同的东西**，
> 不要互相替代。

#### `recipes.PLAYER_FEATURES_V7`

```python
recipes.CROUCH_FEATURES      # ('crouch',)
recipes.PLAYER_FEATURES_V7   # v6 的 19 个 + crouch，20 个
```

`player_tensor(rnd, recipes.PLAYER_FEATURES_V7)` → `(T, P, 20)`。
v6 的 19 列一个没动、顺序也没变。

#### `recipes.crouch_amount(rnd) -> np.ndarray[float32]`

`(T, P)`，0.0 = 完全站立，1.0 = 完全蹲下。来源
`CCSPlayerPawn.CCSPlayer_MovementServices.m_flDuckAmount`。

**这是连续的，而且真的每 tick 都在动**：一次完整的站→蹲中位数 **13 tick
（64 tick 下 203 ms，和 Valve 文档的 200 ms 吻合）**。但它**不是直线** ——
每 tick 的增量从约 0.076 涨到 0.082（轻微 ease-in，速率涨约 8%），
所以别用 `t / 0.2` 去拟合，直接用这些值。

```python
c = recipes.crouch_amount(rnd)       # (3423, 10) float32
```

> ⚠️ **粗采样会把这条斜坡吃掉一大半。** 约 200 ms 的过渡期里，
> 每 tick 有 938 个中间态采样点；`stride=8` 只剩 114 个（12%），
> `interval=0.25`（4 Hz）剩 56 个（6%），`interval=0.5`（2 Hz）剩 27 个（2.9%）。
> 要看下蹲过程就用默认的每 tick 采样。

> ⚠️ **只有 `m_flDuckAmount` 能用。** 同一个 pawn 上还有
> `m_flDuckOffset`（= −18 × crouch，完全冗余）、`m_bInCrouch`、`m_nCrouchState`、
> `m_nDuckTimeMsecs`（这三个在 2 个 demo、约 200 万行里**恒为 0**，是死字段）。
> `m_flDuckOffset` 在**死后会冻住**：21.08% 的死亡行 `amount=0` 但 `offset=-18`，
> 中位持续 838 tick（13 秒），最长 6,479 tick。`m_flDuckAmount` 则干净地归零 ——
> 这就是存 amount 而不是 offset 的原因。

#### `recipes.crouching_mask(rnd) -> np.ndarray[bool]`

`(T, P)`：碰撞盒是否**已经**从 72 跳到 54。来源 `m_fFlags` 的
**bit 2（`FL_DUCKING`）**。

**这是阶跃，不是 `crouch > 0.5`**，两者会不一致：

* 碰撞盒 `m_vecMaxs.z` **只取两个值**：72（站）和 54（蹲），中间没有 55..71。
  `maxs.z < 60` 与这个 bit 在 **727,649 个存活采样点上 100.0000% 一致**；
* 但 `FL_DUCKING` 并**不是** crouch 的纯函数 —— 实测"盒子还是 72 时 crouch 最大
  到过 0.9998"、"盒子已经是 54 时 crouch 最小到过 0.7516"，因为标志跟的是
  **下蹲键的意图**，不只是当前高度。

所以：**判定"能不能钻过去/打不打得到"用这个**，画相机用 `eye_height`。
把碰撞盒画成渐变的是错的，真实的盒子不渐变。

#### `recipes.eye_height(rnd, standing=64.0938) -> np.ndarray[float32]`

`(T, P)`：眼睛相对**脚底**的高度，HU。就是 `standing - 18 * crouch`。

#### `recipes.eye_position(rnd, standing=64.0938) -> np.ndarray[float32]`

`(T, P, 3)`：眼睛的**世界坐标**，HU。`x` / `y` 直接用存的，只有 `z` 加上眼高。

> ⚠️ **`(x, y, z)` 是脚底，不是眼睛。** 三条独立证据：
> 碰撞盒是 `[-16,-16,0] .. [16,16,72]`（`m_vecMins.z` 在 1,012,930 行上恒为 0），
> 所以盒子从原点往**上**长；186 段垂直静止的下蹲斜坡里 **185 段的 z 完全不变**
> （眼会掉 18 HU，脚不会）；`visuals.js` 本来就是当脚底用的。
> 别去用 `CCSPlayerPawn.origin`（全 1,012,930 行恒为 `[32, 48, -68]`，不是世界坐标）。

#### 站立足眼高：唯一一个 demo 说不准的数

| 来源 | 值 |
|---|---|
| Valve 文档（`PLAN.md` §4.4） | **64.0938** ← 默认取这个 |
| `CCSPlayerPawn.m_vecZ` 的众数 | 64.000 |

`m_vecZ` **不可信**：空中时中位数只有 13.141（只有 4.88% 落在 [60,65]），
地面时中位数 62.895 且 18% 越界；`m_vecX` / `m_vecY` 更是乱跳。
它像是"基址 + `m_flDuckOffset`"的别名，站着的时候能凑对，别的时候不行。
CS2 的 966 个 netprop 里**没有任何视点偏移字段**（没有 `m_vecViewOffset`），
demoinfocs 的 `PositionEyes()` 在 Source 2 上直接 panic。
所以绝对值只能来自文档，两者差 0.0938 HU = **2.4 mm**，画面上不可见。

常数集中在 `recipes.STANDING_EYE_HEIGHT_HU` / `recipes.CROUCH_EYE_DROP_HU`
和 `replay-core.js` 的 `STANDING_EYE_HEIGHT` / `CROUCH_EYE_DROP`，要改改这两处。

> 18 HU 这个行程有**两处独立佐证**：`m_flDuckOffset == -18 × crouch`
> （727,649 个存活采样上精确成立，31 个例外全是死亡/重生的单 tick 毛刺），
> 以及碰撞盒 72 − 54 = 18。

#### 老文件会明确报错，不会假装有数据

```python
recipes.crouch_amount(v6_round)
# KeyError: this file has no `crouch` column — it was written in a format
#           older than cs2.demo.v7; re-extract the demo to get the crouch amount
```

`recipes.tick_sample()` 对 v7 文件会多返回 `crouch` / `crouching` / `eye_height`
三个键；对 v6 文件返回 v6 的键（不会返回一堆 0）。

#### 实测占比（`test.dem`，存活采样点）

| 量 | 占比 |
|---|---|
| `crouch > 0` | 4.66% |
| `crouch > 0.5` | 3.97% |
| `FL_DUCKING` | 3.52% |

下蹲过渡共 62,148 次。

#### 体积代价（实测，`test.dem` 每 tick 采样）

| | .csz |
|---|---|
| v6（19 列） | 7.371 MB |
| v7（+`crouch` +`crouching`） | 7.411 MB |
| **增量** | **+0.040 MB（+0.54%）** |

740,030 个玩家行摊下来 **0.054 字节/行**（约 0.43 bit/行）。这么便宜是因为两列
都极度可压缩：`crouch` 只有 4.66% 的存活采样非零，`crouching` 绝大多数是 `False`，
zstd 直接吃掉。

传到浏览器是另一回事：那里 `crouch` 量化成 **Uint8**（0..255），和
`shots` / `footsteps` / `jump` / `land` 一样。255 级铺在 18 HU 的行程上，每级
0.071 HU，眼睛位置误差最多 **0.035 HU = 0.9 mm**；不量化的话最大回合要多传约
340 KB base64。`.csz` 里始终是完整 float32，量化只发生在传输层。


---

## 5. `voxel` —— 底层（用 `recipes` 就不用碰）

`recipes` 已经把常用的都包好了。这里只列出来，方便你要做更底层的事。

### 常量

```python
voxel.VOXEL_GRID_DIM      # 32        主网格边长（格）
voxel.VOXEL_WORLD_SIZE    # 12.0      一格 = 12 HU
voxel.VOXEL_GRID_CENTER   # 15.5      网格中心下标（所以 15/16 这一对夹住引爆点）
voxel.KEYFRAME, voxel.DELTA, voxel.HEARTBEAT   # 3, 2, 0   记录类型
voxel.MODE_APPEND, voxel.MODE_FULL             # 'append', 'full'  日志写入模式
```

世界坐标换算就是 `world = detonation + (grid − VOXEL_GRID_CENTER) * VOXEL_WORLD_SIZE`，
也就是 `recipes.grid_to_world`。

### 时间轴

| 函数 | 作用 |
|---|---|
| `journal_at(inst, tick, cache=None)` | 某 tick 的**完整日志** bytes，不在存活期返回 `None`。`cache` 传一个 dict 可以在遍历很多 tick 时缓存折叠结果 |
| `state_at(inst, tick)` | `(m_nVoxelFrameDataSize, m_nVoxelUpdate)`，或 `None` |
| `change_ticks(inst)` | 所有"内容变了"的 tick 列表 |
| `iter_journals(inst)` | 逐个 `(tick, 日志)` 生成，增量折叠 |

### 日志 → 体素

| 函数 | 作用 |
|---|---|
| `boundary_voxels(journal, size=None)` | 日志明确点名的体素集合 `{(x,y,z)}`。稀疏采样，**不是**水密面，别拿它做遮挡（会漏掉 45.8% 的烟） |
| `solid_voxels(journal, size=None)` | 加上被围住的内部：**填满表面点云的凸包**（返回 `list`）。不是数据，但渲染和烟遮挡都用它 |
| `solid_mask(journal, size=None)` | 同上，直接给 `(32,32,32)` bool 栅格（热路径用它，省掉 5000 个 tuple 的往返） |
| `solid_rows(journal, size=None)` | 每个 (x,y) 的 Z 占据位掩码，自己想做体素运算时用 |
| `convex_hull_fill(cells, size=None)` | 任意点集 → 其凸包内的全部格子（`set`） |
| `convex_hull_mask(cells, size=None)` | 同上，直接给 `(32,32,32)` bool 栅格；和 `convex_hull_fill` 逐格相同 |
| `convex_hull_halfspaces(points)` | 点集 → 凸包半空间 `(nx, ny, nz, c)`，`n·p − c ≤ 0` 为内部。走 Qhull，没 scipy 时回落到纯 Python（`voxel.HULL_BACKEND` 看当前后端） |
| `world_voxels(journal, detonation, size=None, solid=True)` | 直接给世界坐标点列 |
| `grid_to_world(x, y, z, detonation)` | 单个网格坐标 → 世界坐标（返回 tuple） |

### 字节级

| 函数 | 作用 |
|---|---|
| `split_journal(raw, size=None)` | 切成 `[(seq, payload), ...]` 和消耗的字节数 |
| `frame_type(payload)` | 记录类型：`KEYFRAME` / `DELTA` / `HEARTBEAT` |
| `apply_keyframe(payload)` | 解析关键帧 → `(seeds, masks, 消耗字节)` |
| `apply_delta(payload)` | 解析增量 → `[(cell_morton, voxel_mask), ...]` |
| `journal_stats(journal, size=None)` | 诊断：各类记录各多少、解析到哪 |

### 编码

| 函数 | 作用 |
|---|---|
| `morton3(x, y, z, bits)` | 三维坐标 → Morton 编码整数 |
| `demorton3(index, bits)` | 反向 |

`bits=3` 是 512 个格子的主网格（每轴 **0..7**），`bits=2` 是 64 个子体素
（每轴 **0..3**）。**坐标超出这个范围就编不回去** —— `morton3` 只会取低 `bits` 位，
所以 `morton3(5, 9, 17, 3)` 是个静默丢位的错误用法（`9` 和 `17` 已经超出 0..7）。

---

## 6. `sidechannel` —— 你不需要碰

它负责调用和解析那个 Go 辅助程序（取火焰几何和烟雾体素）。
`extract()` 会自动调用，编译和缓存也自动。只有你要单独跑辅助程序时才用：

```python
sidechannel.ensure_binary()                      # 确保编译好了
sc = sidechannel.run(demo_path, ticks=[...])     # 跑一次，返回解析后的 dict
print(sidechannel.summary(sc))
```

出错时抛 `sidechannel.SideChannelError`（二进制缺失、编译失败、
辅助程序输出坏 magic 都走这个异常）。`extract()` 里它会被转成该回合的
`skipped_rounds` 记录，而不是把整个提取炸掉。

其他不打算给你用的：`binary_path` / `build` / `ensure_binary` / `run` / `parse`
（内部流程）、`MAGIC` / `VERSION` / `MODE_APPEND` / `MODE_FULL`（blob 协议常量）。

---

## 7. 速查表

| 我想…… | 用这个 |
|---|---|
| 打开文件 | `codec.load(path)` |
| 保存 | `codec.save(data, path)` |
| 验证存读一致 | `codec.equal(a, b)` |
| 取第 7 个回合 | `recipes.round_by_id(data, 7)` |
| 有哪些回合可用 | `recipes.round_ids(data)` |
| 哪些回合被跳过了、为什么 | `recipes.skipped_rounds(data)` |
| 某 tick 是第几个采样点 | `recipes.sample_index(rnd, tick)` |
| 遍历一个回合的采样点 | `recipes.iter_samples(rnd)` |
| 玩家状态 → `(T,10,16)` 张量 | `recipes.player_tensor(rnd)` |
| 玩家状态 + 地面/空中信号 → `(T,10,19)` | `recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V6)` |
| 谁踩在地上 | `recipes.on_ground_mask(rnd)` |
| 谁起跳 / 落地了 | `recipes.jump_mask(rnd)` / `recipes.land_mask(rnd)` |
| 逐字段 `(10,T)` 轨迹 | `recipes.player_tracks(rnd)` |
| 谁活着 | `recipes.alive_mask(rnd)` |
| yaw 的 sin/cos | `recipes.yaw_sincos(rnd)` |
| 换成米 | `recipes.to_meters(X)` |
| 身上有什么装备 | `recipes.inventory_multihot(rnd)` |
| 谁看得见谁 | `recipes.spotted_multihot(rnd)` |
| 缺值统计 | `recipes.nan_report(data)` |
| **有没有发出脚步声**（别用 `footsteps`） | 自己拿 `v` + `on_ground`，见 §4.3 —— 工具不再提供阈值函数 |
| 看 `footsteps` 有多不可靠 | `recipes.footstep_warning(data)` |
| 某一 tick 的全部东西 | `recipes.tick_sample(data, rid, tick)` |
| 这一 tick 哪里在烧 | `recipes.active_fire(rnd, tick)` |
| 这一 tick 哪里被烟盖住 | `recipes.active_smoke(rnd, tick)` |
| 烟的世界坐标点云 | `recipes.smoke_points_at(inst, tick)` |
| 烟的 32³ 占据网格 | `recipes.smoke_grid_at(inst, tick)` |
| 火场体素化 | `recipes.fire_volume_at(rnd, tick, resolution=32)` |
| 切训练窗口 | `recipes.iter_windows(data, length=64, stride=32)` |
| 看文件概览 | `recipes.summary(data)` |
| 命令行看文件概览 | `python -m demo_extract --inspect file.csz` |
