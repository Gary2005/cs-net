# spatial_prediction · 训练数据可视化

`data/demos/prepared/spatial_prediction/**/*.npz` 的逐样本检查工具。一条样本 = **一帧**，
标签是「这一帧之后会发生什么」，所以这个工具的重点不是回放，而是**一眼看出标签对不对**。

```bash
# 1. 先把样本造出来（density 默认 1 s，一局几千帧）
python -m scripts.training_data.prepare --task spatial_prediction

# 2. 开可视化（默认 127.0.0.1:5118）
python -m scripts.training_data_visualizer.spatial_prediction
python -m scripts.training_data_visualizer.spatial_prediction --port 5119 --open

# 指到别的目录 / 只读某个 split 的产物
python -m scripts.training_data_visualizer.spatial_prediction \
    --data-dir data/demos/prepared/spatial_prediction

# 不打开浏览器也能验一遍（逆变换 / 载荷 / 标签体检 / HTTP）
python -m scripts.training_data_visualizer.spatial_prediction.selftest
```

只依赖 **numpy** 和标准库 —— 不需要 `demoparser2` / `numba` / torch / Flask。
（样本是 npz，雷达底图是 `app/radar.py` 已经生成好的 PNG。）

## 页面上有什么

| 区块 | 看什么 |
|---|---|
| **顶栏** | 文件下拉（一个 npz = 一个回合的 K 帧）、◀ / ▶ 按**帧**连续翻、`重新扫描` |
| **概览** | split / 文件名 / 第几行 / 回合号 / 回合时间 / 帧下标 / 文件大小 / 存活人数 |
| **标签** | 5 张卡片：`ct_win`、`next_death`、`next_kill`、`alive_end`、`can_kill`；下面两个列表逐玩家列 bool |
| **这一帧** | 俯视雷达（位置 / 朝向 / 血量 / 活死）+ 玩家表 + 炸弹 / 回合时间 + 道具表 |
| **遮挡矩阵** | `obstruction[i][j]`：i 的眼睛看 j 的身体，0 通 / 1 烟 / 2 墙，有方向 |
| **原始特征** | `ctx_player (10,20)` 的 [0,1] 原值；勾「显示物理值」切到逆变换后的 HU / 度 / 点 |

雷达上的约定：

* 圆的颜色 = 队伍（CT 蓝 / T 橙），短线 = yaw 朝向（0° = +x、90° = +y），
  外圈弧 = 血量（绿 > 50 / 黄 > 25 / 红）。
* **死人不上图**：死人的位置列可能是 0 或残留值，画出来会落到地图角上。死人才在
  玩家表里出现（整行变暗）。
* `☠` = 标签里的「下一个死亡」，`✛` = 「下一个击杀」，两者都有虚线圈。
* `C4` 方块是**这一帧**的炸弹位置（会被携带、掉落、安放），旁边写着是否已安放。

### 可分享的 URL

地址栏的 hash 是 `#<split>/<文件名>/<行号>`，例如：

```
http://127.0.0.1:5118/#train/de_mirage_f0_r7_n93.npz/12
```

刷新 / 贴给别人都会落到同一帧。

### 键盘

| 键 | 作用 |
|---|---|
| `←` / `→` | 上一帧 / 下一帧（跨文件连续翻） |
| `[` / `]` | 跳到上一个 / 下一个**文件**的第一帧 |
| `Home` / `End` | 第一个 / 最后一个样本 |
| `P` | 切换原始值 / 物理值 |

## 一个 npz = 一个回合的 K 帧

`tasks/spatial_prediction.py` 的采样是 **demo 级**的：一个 demo 打 T 秒，就在所有回合的
所有采样帧上均匀随机、不放回地抽 `T / density` 帧（`density` 默认 1 s）。写盘时按回合
切，所以：

```
de_mirage_f0_r7_n93.npz      ← 文件名末尾的 _n93 = 这个文件里有 93 行样本
```

行是**按时间升序**排的（这一点很重要，下面几条体检靠它）。所以 `?k=` 缺一不可；
`/api/samples` 返回的每个文件都带 `n`，前端把"文件列表"摊平成"样本列表"来翻页。

## 这个工具会替你做的体检（标签不变式）

单帧样本看不出"未来算错了没有"，但**同一个文件的 K 行**是同一回合的时间序列，这给了一
组很硬的不变式。`data.py::_sequence_warnings()` 会逐条检查，违反的写进页面顶部的黄色
警告条（**不抛错** —— 让人看见，而不是让整页打不开）：

| 不变式 | 为什么必须成立 |
|---|---|
| `ct_win` 在文件内恒定 | 一个回合只有一个胜负 |
| `alive_end` 在文件内恒定 | "回合结束时还活着"和问的是哪一帧无关 |
| `next_death` / `next_kill` 一旦变成 `10` 就不能再变回来 | 之后没有死亡 / 击杀了，时间不会倒流 |
| `can_kill` 随时间单调收缩 | 越往后，"还能拿到的击杀"只会更少 |
| 有 `can_kill=True` 就必须有 `next_kill != 10` | 有人之后会击杀，却找不到下一个击杀者 |

外加几条**行级**检查：

* `next_death` / `next_kill` 指向的人**当前帧必须活着**（这是标签定义的一部分）；
* `can_kill[p]` 的人不能是死人；
* `ctx_player_team` 里有 `-1`（队伍未知）不该出现在正常数据里；
* `seconds` 和 `global_feat` 第 5 列（回合时间）必须同步（差 > 0.02 s 就报）；
* 死人的武器格必须是 `-1`;
* 文件名里的地图和 `map_id` 必须一致。

⚠ **最后一帧活着的人可以不在 `alive_end` 里**，这不是 bug：`end_reason ∈ {t_killed,
ct_killed}` 的回合，最后一枪落在 `ticks[-1] + 1`（比最后一帧晚 1 tick），造数那侧把它
当成"最后一帧之后"的击杀算进来了（实测 78 个回合里 49 个有这种情况）。

⚠ **出现"标签里的人当帧是死的"这类警告，说明造数 / 切片有问题，不是前端。** 这类错
在训练里只会让 loss 好看，不会崩。

## 坐标 / 逆变换

文件里的 `ctx_player` 每一列都在 **[0,1]**（`v01 = clip(v*scale+bias,0,1)`，
见 `tasks/aim.py::_SCALE`）。本工具逐列反解，公式和 `tasks/aim.py` / `data.py` 的
`SCALE` / `BIAS` **必须同步**（`selftest.py` 的往返测试钉的就是这个）：

```
x y z          (v01*2-1) * MAP_RANGE + MAP_CENTER      → 世界坐标 HU
yaw / pitch    atan2(sin, cos)                          → 度（文件里是 4 个 sin/cos 列）
vx vy vz       (v01*2-1) * 300 / 1200                   → HU/s
hp armor       v01 * 100                                → 点
since_shot     v01 * 100                                → 秒（≥99.9 = 本回合还没开过枪）
crouch         原样 0..1                                → 眼高 = 64.0938 − 18·crouch
```

`ctx_player_mask` 是**真 bool**（True = 这一帧活着），所以 `to_torch()` 把它变成
`torch.bool`，可以直接当 `key_padding_mask` / `attn_mask` 用。**队伍在
`ctx_player_team`**：`0` CT / `1` T / `-1` 未知 —— 死人也保留队伍（"哪队"和"死没死"
是两个正交的量）。`ctx_player` 第 15 列虽然也是 team，但它是 [0,1] 浮点，
`-1` 会被 clip 成 `0`，所以**别拿它判队伍**。页面上 `?` = 队伍未知。

样本里没有 `player_name`（那会把字符串混进训练张量）：玩家一律用下标 `P0..P9`，
和 `rnd["players"]` / `events.kills` 里的 `a` / `v` 是同一套编号。想知道是谁就回
原始 csz 查。

## 遮挡矩阵怎么读

`obstruction[i][j]` = 从 **i 的眼睛**（`z + 64.0938 − 18·crouch`）看向 **j 的包围盒 8 个
顶点**（半宽 16 HU、高 72 站 / 54 蹲），取 8 条射线里**最轻**的那个：

```
0 = 通（至少露出一个角）   1 = 被烟挡   2 = 被地图墙挡
```

* **有方向**：`[i][j]` 和 `[j][i]` 可以不同（两个人的眼睛不在同一个高度）。
* **死人的行 / 列恒为 0**，页面上画成暗格而不是"通"。
* **`0` ≠ "看得见"**：它只说明中间没有墙 / 烟，对方可能正背对着（`yaw` 不在这个矩阵
  里）。要做"谁看得见谁"得自己叠 FOV。

## 实现结构

```
data.py      npz → JSON（逆变换 + 标签体检；只依赖 numpy，不碰 HTTP）
server.py    标准库 ThreadingHTTPServer：扫描 / 发载荷 / 发雷达 PNG / 武器图标
static/      前端（原生 JS + canvas，没有构建步骤、没有外部依赖）
selftest.py  逆变换往返 / 合成载荷 / 标签体检 / HTTP 端到端
```

接口：

```
GET /                                 页面
GET /static/{index.html,app.js,style.css,favicon.svg}
GET /api/samples                      文件清单（每个文件 n 行）+ 每个 split 的帧数
GET /api/sample/<split>/<name>?k=N    第 N 行 → 单帧载荷（含 5 个标签 + warnings）
GET /api/radar/<map>                  地图俯视底图 PNG（没缓存就 404，前端退化成网格）
GET /api/weapon-icons                 武器名 → 图标 URL
GET /weapon-icons/<file>.png          CS2 HUD 武器图标（复用 3D 回放器的素材）
GET /api/health · /api/rescan         探活 / 重扫磁盘
```

`k` 越界 / 不是整数 → **400**（不是 500）；样本名带 `../` → 400；
指向别的任务的目录（比如 `prepared/aim`）→ 扫描阶段就明确报错"文件名里没有 `_n<K>`"。

## 数据体检会看到什么

1. **黄色警告条**：上面那组不变式 + 行级检查。第一次跑新数据先把它看完。
2. **`ctx_player_team` 里有 -1**：`rnd["teams"]` 缺人（正常 10 个 'CT'/'T'）。
3. **雷达缺失**：`app/data/thumbs/_maps/` 里没有这张图的 PNG，俯视图退化成坐标网格
   （坐标是对的，只是没有地图底图）。
4. **`since_shot ≥ 99.9`**：饱和值 = 本回合还没开过枪，不是"99.9 秒前开过枪"。

## 自检

```bash
python -m scripts.training_data_visualizer.spatial_prediction.selftest
```

四段：逆变换往返、合成样本载荷（含 6 条"故意造坏"的标签）、真实样本（有数据时）、
HTTP 端到端（有可写临时目录时，否则跳过并打印原因）。

## 和 aim 那个可视化器的关系

同一套骨架（同样的深色 web app、同样的逆变换套路、同一份武器图标素材）：`aim` 一条
样本是 257 帧的窗口，重点是**回放 + 第一人称 + 模型曲线**；这里一条样本是一帧，重点是
**标签对不对 + 这一帧的场面**。所以这版没有时间轴 / 播放 / 模型面板，换成标签卡片和
标签不变式体检。两个工具的 `data.py` 各自镜像一份常量 —— 它们故意不共享代码，因为
各自只依赖 numpy，不应该为了省几行去 import `scripts.training_data`（那会把
`demo_extract` 拖进来）。
