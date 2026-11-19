# cs-net Demo 库

本地 CS2 Demo 的入库、维护与回放工作台。它是 `visualizer/` 前面的一层：把散落在
磁盘上的 `.dem` 收进一个库，生成缩略图和比赛元数据，按玩家汇总，然后一键送进已有的 3D 回放。

    设置监听目录 ──> 轮询发现 .dem/.zip ──> 去重入库 ──> demo_extract 提取 .csz
                                                              │
                                      缩略图 + 记分板 + 关注玩家 + Steam 头像
                                                              │
                                            勾选一场或多场 ──> /replay?batch=<id>
                                                              │
                                                   按玩家汇总 ──> /players

和分析的联动就是回放本身 —— 这个应用不做 LLM 分析，`analysis` 指的就是
`visualizer/` 那套 3D 回放。

三个页面：`/library`（Demo 库）、`/players`（玩家索引，`/players/<steamid>` 是档案）、
`/replay`（3D 回放）。

---

## 快速开始

```powershell
conda activate cs-net-v5          # 依赖 demoparser2 / Flask / numpy / Pillow
cd D:\project\cs-net-v5
python -m app.server --port 5000
```

打开 <http://127.0.0.1:5000/> → 自动跳到 `/library`。

| 地址 | 内容 |
| --- | --- |
| `/` | 重定向到 `/library` |
| `/library` | Demo 库（本文件描述的部分） |
| `/replay` | 3D 回放（`visualizer/` 原有页面，未改动其数据通路） |
| `/api/library/*` | 库的接口 |
| `/api/*` | 回放的接口，原样保留 |

常用参数：

```
--host 0.0.0.0        允许局域网访问
--port 5000           端口
--maps-dir <dir>      指定 optimized_obj_files 目录
--no-watch            不启动目录监听（仍可手动「扫描」）
--debug               Flask 调试模式
```

### 第一次使用

库里默认是空的，因为 **监听目录必须由你指定** —— 5E / 完美 / 官匹 / FACEIT 的
下载路径各不相同，而且没有可靠的自动探测办法。所以：

1. 点顶栏 **设置**。
2. 在「本机探测到的候选目录」里挑一个，或者直接粘贴路径 → **添加**。
3. **保存设置**。
4. 把 `.dem` / `.zip` 丢进那个目录（或者点顶栏 **扫描** 立即入库）。

设置里还有 **关注玩家** —— 填进去的名字会在列表里高亮。
改完名单，已经入库的记录会**当场重新匹配**，不用逐场刷新。

---

## 功能

### Demo 库维护

- 网格 / 列表两种视图，缩略图、来源徽章、地图、比分、选手名单、时长、回合数、文件大小
- 搜索（文件名 / 展示名 / 备注 / 地图 / 队名）、按状态 / 来源 / 地图筛选、8 种排序
- 行内编辑：**展示名**（点击标题）和**备注**（点击备注框）。展示名不改磁盘文件名
- 手动修正来源、队名，一键**交换主客队**（自动判定偶尔会反）
- 删除：单条和批量走**同一个对话框**，库记录一定删，原始 `.dem` 默认保留，勾上才移走
- **刷新信息**：从已有的 `.csz` 重算比分、名单和高亮，不重新解析 demo
- **重新解析**：忽略已有的 `.csz`，按当前采样设置重新解析（改了采样设置后用这个）
- 选手名单里带头像，点名字直接进玩家档案

### 中英双语

四个页面（Demo 库 / 玩家 / 玩家档案 / 3D 回放）共用一个语言开关，在顶栏右侧
（`中文 | EN`）。默认中文，选择记在 `localStorage['csnet.lang']`，也可以直接
用 `?lang=en` 分享英文页面。

文案在 `visualizer/static/js/i18n/`（运行时 + 四张文案表），页面模板只写
`data-i18n` 属性和中文原文，JS 里用 `t()`；切换语言会刷新一次页面。做法和
验证方式见 `visualizer/README.md` §4.16。

### 缩略图

每张缩略图是**按地图实时生成的雷达底图**，不是截屏：

```
maps/optimized_obj_files/de_mirage.obj
        │  面积加权重心采样（每张地图只算一次，缓存到 app/data/thumbs/_maps/）
        ▼
   俯视高度场 ──> 空洞填充 ──> 山体阴影着色 ──> 1024×1024 雷达底图
        │
        ▼
   叠加选手轨迹（两队各一色）+ 比分 + 地图名 ──> 640×400 PNG
```

坐标系要和 `visualizer/static/js/scene.js` 严格一致：那边是
`three = (gy, gz, gx) * SCALE`、`SCALE = 0.0254`，地图 OBJ 未做变换直接放进场景。
所以 OBJ 顶点 `(ox, oy, oz)` 对应 demo 坐标 `(oz/S, ox/S, oy/S)`。
**这个变换错了不会报错，只会让选手穿墙** —— 所以 `app/tests/test_radar.py` 里
拿真实 demo 的选手坐标做投影，要求落在几何上的比例 > 80%，并且必须高于
四个错误变换（交换轴、翻 x、翻 y、翻两者）。当前实测 98.1%，最好的错误变换 84.1%。

### 模型预测（AI）

两份权重合计 665 MB，**不在仓库里**（`models/` 被 gitignore）：干净检出之后
AI 面板一定报"权重不存在"。第一次用之前先下 —— 库页面顶部的「下载模型」横幅点
一下（只在真缺的时候出现），或者 `python scripts/download_models.py`。脚本用
`huggingface_hub` 下；连不上 huggingface.co 时会把 `HF_ENDPOINT` 自动切到
hf-mirror.com（和自己 `$env:HF_ENDPOINT = "https://hf-mirror.com"` 等价）。
`--mirror` 强制走镜像、`--endpoint` 指定站点。

回放页底部多了两个开关：**AI 预测面板** 和 **AI 对枪**。它们把 `models/` 下
两份**写死**的训练产物接到眼前这场回放上：

| 模型 | 权重 | 每帧预测什么 | 画在哪 |
| --- | --- | --- | --- |
| `spatial_prediction` | `models/spatial_prediction/20261113-112930.pt` + `.yaml` | CT 赢面、逐人「活到回合结束」/「之后还能拿到击杀」、下一杀 / 下一死（11 类） | 视口顶部一块**只显示赢面**的面板 + 左下角每个选手一行的**五项指标表**；3D 里不画东西 |
| `aim` | `models/aim/20261119-013008.pt` + `.yaml` | `P(A 击杀 B)` | 死前 3 秒的弧形虚线，按胜率切成两段、各按那一侧上色（两端就是这场对枪的两个人） |

五个头都接到了界面上，一列都不少：

```
存活        P(活到回合结束)          cyan   左下表格第 1 列
可击杀      P(之后还能拿到击杀)      violet 左下表格第 2 列
下次击杀    P(下一个杀人的是他)      green  左下表格第 3 列（11 类 softmax 的第 p 项）
下次被击杀  P(下一个死的是他)        red    左下表格第 4 列（同上）
对枪        P(他赢下这场对枪)        purple 左下表格第 5 列 + 弧线两段颜色的分界（位置即胜率）
```

死人的列在服务端就清成 0（`next_*` 再按行归一化）——模型的 forward 不做这件事
（训练时是在 loss 里把死人的列填 `-inf` 掩掉的），不掩的话表里会出现"某个已经
死掉的人有 3% 概率成为下一个死的人"。

左下角这张表不只是看的：悬停列头会解释这一项在预测什么；点名字 = 选中并跟随
（和点左边选手卡片一样）；点数字会在表格上方画出**这一项整回合的曲线**，点曲线
上任意一点跳到那一刻并锁定镜头。曲线用的就是**同一份**整回合预测（`spatial.*`
里那一列），不会再发请求，所以它和表格里的数字永远对得上。对枪那一格没有曲线 ——
aim 只在死前 3 秒的窗口里有值。

顶部面板的两个**回合胜率**也能点：同一张曲线卡画出 `win_ct` / `win_t` 整回合
（T = 100% − CT），点曲线上任意一点跳时间；它不属于某个人，所以**不选中、不锁
镜头**。

**对枪只在"死前 3 秒"里**（`app/ai/service.py::DUEL_LEAD`）：窗口 = `[死前 3 秒,
受害者最后一个活帧]`，其余时间没有任何 aim 预测，也就没有弧线。两端用的是这一
回合**真实的击杀双方**（`.csz` 的 kills 事件，A = 受害者 / B = 击杀者）—— aim 模型
就是"这两个人谁会赢"，喂一对"模型猜的对枪"进去既不是训练分布，也会在没有对枪的
时候乱跳；"谁即将被打死"这件事由 `next_death` / `next_kill` 在指标表里回答。
窗口长度取 3 秒而不是训练用的 4 秒：训练时 label 在整条 4 秒窗口的**每一帧**上都
算一个样本，所以死前 3 秒的每一帧都在训练分布内，而多留 1 秒只会让"还没打起来"
的阶段也挂着一条弧线。

**回合结束那一枪也算一场对枪**（`app/ai/service.py::_round_kills`）：`end_reason ∈
{t_killed, ct_killed}` 的回合，最后一枪落在 `ticks[-1] + 1` —— 引擎在死亡那一 tick
就把 `total_rounds_played` 加一，所以那一帧的采样已经归到下一个回合了，
`round_events()` 的 `t <= ticks[-1]` 会把它整条滤掉。不补的话，"打死最后一个敌人、
回合立刻结束"的回合就永远没有最后一场对枪的弧线（实测 4 份 demo / 78 个回合里有
49 个，63%）。补法是把它挂到**最后一帧**上：受害者那一帧还活着、下一个 tick 才死，
和训练样本里"受害者最后一个活帧"是同一种情形。结果 JSON 的版本跟着升到 `v3`
（`CACHE_VERSION`），旧缓存自动作废。造数据那侧是同一件事：
`tasks/_util.py::late_kills()` 把这一枪补进 `aim`（多一条样本）和
`spatial_prediction`（`alive_end` / `next_death` 指向它）—— 但**已经造好的数据集里
没有这批样本**，要让模型真正在训练里见过它，得重跑 `prepare` 再重训。

**AI 数据统计也在同一次前向里算完**（`app/ai/stats.py`，payload 里多一个 `stats` 段，
`CACHE_VERSION` 因此升到 `v5`）：每个选手的「平均击杀影响力 / 平均回合影响力 /
平均站位死亡风险 / 平均站位击杀率 / 对枪评分」。回合影响力是逐采样点的 CT 胜率
涨落（末尾按真实胜负补一格），记在**涨落之前**那一帧的存活名单上、按人数均分；
某队全灭后沿用最后一个还有人活着的帧，所以每回合十个人加起来是 0。其余几项都建立
在已经算好的 spatial 逐帧输出上，只多一次 aim 前向（每场对枪问"击杀前 2 秒"的两
个方向，一个回合 ≈ 0.2 s）。
所以统计页上的胜率和回放页那条弧线是同一批预测，而且这些数和回合载荷一起缓存 ——
换页签、开关弹窗都不用再跑模型。整场比赛的汇总另存一份
`<demo>/<models>_s16_match.json`（`GET /api/ai/stats`），玩家页的 AI 页签读的就是它。

**路径写死**：没有模型选择器，也不看"哪个目录最新"。版本管理的开关就是
`app/ai/models.py::PINNED_WEIGHTS` 那两行 —— 想换 run / 换文件就改路径，刷新页面
即生效（缓存键里带权重的 `size + mtime`，旧预测自动作废）。

网页吃的是**扁平布局**：`models/<task>/<run>.pt` + 同名 `<run>.yaml`，由
`python scripts/export_weights.py` 从训练产物里抽出来（只留 `model` + `epoch` /
`step` / `best` / `cfg`，没有 AdamW 动量，两份合计 665 MB 而不是 2 GB ——
发给别人下载时不用连带传优化器）。训练目录的老布局 `<run>/last.pt` +
`<run>/config.yaml` 也认，`pinned_run` 会先找扁平配置再退回 `config.yaml`。
换路径时权重和配置要一起换。

输入不是另造一份：`app/ai/features.py` 直接调 `scripts/training_data/tasks/aim.py`
的 `_Round`（造训练数据用的那一份），从**已经加载在内存里的** demo dict 现算
`ctx_player` / 遮挡矩阵 / 道具 / 全局列，不重新 load `.csz`、也不落 npz。

**进度条**：冷分析一次十几秒，同步请求在这段时间里一个字节都不发，用户只能盯着
不动的画面猜。所以 `/api/ai/round/<idx>?progress=1` 是一条 **NDJSON 流**（一行一个
事件：`progress` / `result` / `error`），真正的计算跑在后台线程里，生成器每 80 ms
读一次加锁的进度快照。页面上的进度条显示**阶段 + 百分比 + 已用时**，阶段来自
`build_round_tensors` 的分块（玩家特征 1/10、两两遮挡 2048/6828 帧…）和两次前向的
分块。不带这个参数时行为和以前完全一样（纯 JSON）。

**代价**（RTX 3090 实测，一个 6800 采样 / 106 s 的回合）：首次 **15~25 秒**
（其中 4 s 是特征、3.4 s 是遮挡、1.4 s 读权重，另外 **7 s numba 编译 + 2~4 s 地图
BVH** 每个进程只付一次）；同一个回合再点是**读缓存 3~70 ms**。所以页面刚打开
（`/api/ai/status`）时会在后台线程把这 10 秒的固定开销先做掉，用户点的时候不用
再等。三层缓存：

```
进程内 · 输入张量   挂在会话 data["_ai_tensors"] 上，一个回合约 10 MB，留最近 3 个
进程内 · 模型       两个 checkpoint 常驻显存（约 0.7 GB）
磁盘 · 预测结果     app/data/ai/<demo>/<models>_s<stride>_r<回合>.json（约 130 KB）
```

磁盘缓存的键 = `.csz` 的 `size+mtime` + 两份权重的 `size+mtime` + `stride` + 回合号，
所以"换了 demo""重训了模型"都会自然 miss，不会拿旧预测骗人。payload 结构变了就把
`service.CACHE_VERSION` 加一，旧缓存自动作废。

两个前提不满足时**直接报错而不是瞎猜**：地图必须是训练用的那 8 张（坐标归一化
对未知地图是静默失真的），`.csz` 必须是逐 tick 抽样（模型的时间位置是
`seconds × 64`，stride/interval 抽出来的 4 秒窗口会跨几十秒）。

更细的取舍（为什么用真实击杀双方而不是预测配对、为什么两段弧线的分界是 `P(A 赢)`、
进度阶段的权重是怎么量的、缓存为什么分三层）见 `app/ai/service.py` 的模块 docstring。

### 目录自动监听

`watchdog` 没装，也没打算装 —— 用的是轮询 + **稳定性门槛**：同一个文件连续两次
轮询 `(size, mtime_ns)` 不变才算下载完成。这样 300 MB 的 demo 不会在写入过程中
被解析，也不会因为反复哈希而卡住界面。

- 每个监听目录可以单独设扫描深度、平台标签、启用/禁用
- 添加目录时会先探测一次，告诉你里面有多少 `.dem` / `.zip`
- 文件消失 → 标记为「文件丢失」，不删记录
- `.zip` 会解开取出里面的 demo；已经处理过的 zip 记在 `zip_extract_state` 里

### 与分析联动

在库里勾选一场或多场 → **▶ 载入分析** → 跳到 `/replay?batch=<id>`。

批次是服务端的一个短生命周期对象（6 小时）。回放页右上角多一个下拉切换场次，
选中的那一场走**原有的** `POST /api/load` 加载 —— 回放的数据通路一行没改，
批次只是多了一层「播哪一场」。还没提取完的场次在下拉里是禁用的，后台每 2.5 s
刷新一次，提取好会自动开播。

**回放只有这一个入口。** 回放页以前自带一个上传界面（拖拽 `.dem`、浏览本机
文件、选采样方式），已经删掉了 —— 见下面「踩过的坑 #14」。所以回放页顶栏的
导航和库页面是同一套，从回放页点「Demo 库」就回去了。

采样方式现在也只有一个说法：库里的**提取设置**。`.csz` 在提取时就把采样定死
了，回放只是读回来，不再在回放页上问一遍。

### 玩家档案

`/players` 是玩家索引，`/players/<steamid>` 是单个玩家的档案。两者都从
`scoreboard_json` 现算，没有第二张表 —— 见下面「为什么玩家数据不是一张表」。

索引列出库里出现过的每个人：场次、胜率、K/D、KPR、ADR、爆头率、地图数、最近出现，
可以按这些排序，也可以只看关注玩家。档案页多一张地图分布表，和这位玩家自己的
比赛列表；列表就是库列表换了个筛选条件（`?steamid=`），所以库那边的搜索在这里一样能用。

点卡片名单里的名字、或者列表视图里的名字，都能进档案。

#### 数据统计：传统 / AI 两个页签

档案页（大标题下面那块）是**数据统计**，两个页签 —— 和回放页「统计」弹窗是**同一套
指标、同一套解释文案**（同一个指标在两页必须是同一句话，抄一份只会让两边的定义慢慢
漂开）：

* **传统数据** —— `GET /api/library/players/<steamid>/stats`。跨场合计（首杀率 / 首死率 /
  先手开枪 / 侧背身击杀 / 侧背身被击杀 / 穿烟击杀）加一张每场一行的明细表（K / D / A /
  K/D / ADR 也在里面）。这六项都是**百分比**（首杀 / 首死按回合数，其余按各自的分母），
  样本数放在 `data-tip` 里悬停才显示 —— 和 AI 那一栏的读法一致，页面上不铺 `n=…`。
  服务端要把这位选手打过的每一场 `.csz` 各解一遍（第一次每场
  5~10 s），结果按 `.csz` 的 `size+mtime` 缓存在 `app/data/stats/<key>.json`；之后整页
  几十毫秒。页面加载后自动拉（它就是这位选手的主页数据）。
* **AI 数据** —— `GET /api/library/players/<steamid>/ai`。跨场平均（平均击杀影响力 /
  平均回合影响力 / 平均站位死亡风险 / 平均站位击杀率 / 对枪评分）加一张每场一行的表。
  **第一次要真的跑模型**（每场每回合一次），之后读 `app/data/ai/` 下那场比赛的汇总。
  所以这一栏是**点了页签才请求**的。

两个端点都支持 `?progress=1`（NDJSON），页面拿它画进度条 —— 第一次那几十秒 / 几分钟
不能只让用户看一个转圈。跨场平均的口径是「**先把每场的累计量相加、再除一次**」，
不是各场平均值的平均（否则 5 回合的一场会和 24 回合的一场一样重）。

两张表的版式只有一个规矩：**表头跟着数值走**。「比赛 / 地图 / 回合」三列左对齐，
其余数值列（表头一起）右对齐 —— `.mtable th` 本身是左对齐的，只把数值改成右对齐的
话，窗口一宽、列被撑开，表头就飘到自己的数值左边（1877px 下「侧背身被击杀」偏了
116px）。再给第一列一个 `width: 1%`，否则最长的那一列（比赛）会吃掉整张表的富余
宽度，文件名后面挂出两百多像素的空白。这条由
`visualizer/tests/test_player_stats.mjs` 量 `th` / `td` 文字的右边界守着（差 1px
以上就报错）。

指标本身的口径（先手开枪怎么判"先"、侧背身按哪一帧判、穿烟用什么矩阵、五个 AI 数
取的是哪一帧的胜率）写在 `visualizer/README.md` §3.5，实现是
`visualizer/stats.py` / `app/ai/stats.py`，两个页面上的 `?` 悬停显示同一份解释
（`visualizer/static/js/tips.js`）。

#### 两个名字，两个地方

demo 里只有**游戏内名字**（当场服务器上显示的那个）；**Steam 昵称**是账号资料名，
demo 里根本没有，要去 Steam 取。所以：

- **玩家页以 Steam 昵称为主**（`/players` 的卡片标题、`/players/<steamid>` 的大标题），
  下面用一张表列出这个 SteamID 用过的**所有游戏内名字**，每个名字带场次和最近出现时间。
  取不到 Steam 昵称时标题退回游戏内名字，页面不会空着。
- **Demo 库只显示游戏内名字**。一张卡片讲的是一场比赛，这场比赛里别人看到的就是那个名字；
  Steam 昵称属于账号，放在玩家页。

两个名字实测通常对不上：

| demo 里的名字 | Steam 昵称 |
| --- | --- |
| ZywOo | Herbus doubledoor |
| ropz | mmmmm |
| apEX | Petite Pute |
| Wicadia | amınaKE |
| mezii | leo mezii |

#### 搜索

搜索框同时匹配**游戏内名字**（含改名前用过的）、**Steam 昵称**和 **SteamID**，
输入时弹出一个下拉，列出按匹配度排序的前 8 个候选：整个字符串 > 前缀 > 子串 > 子序列，
所以 `wic` 命中 `Wicadia`，`wcda` 这种记不全的拼法也能命中。每个候选会标出**命中的是哪个字段**，
因为标题显示的是 Steam 昵称、命中的可能是游戏内名字，不标一下看不出为什么它会出现在这里。

下拉只是加速入口；下面那份列表仍然按子串筛选（`/api/library/players?q=`），
所以输入框里打了什么，列表就跟着筛什么。键盘上下选择、回车进档案、Esc 关掉。

Demo 库的搜索框也认 Steam 昵称：玩家页上读到的名字，在库那边搜得到同一个人的比赛。

**头像是去 Steam 取的，不是 demo 里带的。** demo 只有 SteamID64，脸在 Steam 的 CDN 上。
用的是公开的 mini-profile 接口，**不需要 API Key，也不需要登录**：

```
https://steamcommunity.com/miniprofile/<account_id>/json
```

`account_id` = SteamID64 − `76561197960265728`。返回 `persona_name` 和 `avatar_url`，
图片下载到 `app/data/avatars/<steamid>.<ext>` 本地存着，页面从自己的服务器取图。

三条刻意的取舍：

- **在后台任务里取，不在请求里取。** 十个人每人约 0.4 s，放在页面渲染里就是四秒白屏。
  提取成功后自动排一个同步任务；也可以点玩家页顶栏的「同步头像」手动来一次。
- **同步任务是合并的。** 一次扫描注册 60 场 demo，会触发 60 次「提取完成」，
  但只会产生大约两个头像任务：第一个建任务，其余把「跑完再来一轮」标记上，
  而那个任务开始时重新查一遍库里的全部 SteamID，所以中途提取完的场次也被覆盖到。
- **取不到是正常结果，不是错误。** `steamcommunity.com` 在国内直连不了，
  所以失败会记下原因（`player_avatars.error_msg`），界面退回名字首字，绝不阻塞入库。
  失败 6 小时后才重试，成功的 7 天后重取（设置里可调）。

#### 存储管理

设置面板底部列出 `app/data/` 下每一类缓存占了多少、有几个文件，可以整类清掉：
`.csz`、雷达缩略图、Steam 头像、回收站。清缓存是**两件事**：文件删掉，库里指向它的
路径也要清掉（`db.forget_csz` / `forget_thumbs` / `forget_avatar_files`），否则库会继续
声称某场比赛已经提取过，而它的回放数据已经没了。有后台任务在跑时整个请求返回 `409` ——
提取是在解析结束时才写 `.csz` 的，这时候删目录要么让任务失败，要么留下一个没人指向的文件。

清 `.csz` 会把 `ready` 的记录退回 `discovered`，所以下次扫描会重新提取；清头像只删图，
`player_avatars` 里的 Steam 昵称留着（那是已经花过一次网络请求换来的文字，玩家页的标题还要用）。

**原始 `.dem` 只显示不清理**：它在监听目录里，不在 `app/data/` 下，删一场录像是一次针对
单个 demo 的决定，在 Demo 库里勾选后批量做，不是清缓存。

### 为什么玩家数据不是一张表

玩家统计全部由 `scoreboard_json` 用 `json_each` 现算（见 `db.py` 的 `_PLAYER_TOTALS`）。
不建第二张表是因为那张表是个**可能和来源不一致的缓存**：每次重新解析 demo 都要同步重建，
而重建漏了一次，玩家页显示的就是一场不存在的比赛。现算的代价是一次全表扫描，
库到了几万场再说。

`json_valid()` 那层 `CASE` 不是装饰：`json_each` 遇到坏 JSON 会直接报错，
一条手改坏的记录就能让整个玩家页打不开。

---

## 代码结构

```
app/
  config.py     路径、平台路径候选、来源推断、线程安全的 JSON 配置
  db.py         SQLite（WAL）。demo_files + zip_extract_state + player_avatars，
                玩家聚合也在这一层（json_each 现算，不建表）
  probe.py      廉价的 demo 读取：parse_header() 拿地图和服务器名，
                parse_ticks(4 个 tick) 拿真实队名
  metadata.py   从提取结果算记分板 / 名单 / 关注玩家命中
  radar.py      OBJ → 雷达底图 → 缩略图
  avatars.py    Steam 头像：公开 mini-profile 接口、CDN 白名单、本地缓存、合并同步任务
  storage.py    app/data 下各类缓存的体积统计与整类清理
  ingest.py     发现、去重、提取流水线
  watcher.py    轮询式目录监听
  jobs.py       后台任务队列
  api.py        /api/library/*
  ai/           模型预测：features.py（复用训练侧特征 + 分块报进度）、
                models.py（写死的权重路径 + 加载）、service.py（5 个头 + 对枪窗口 +
                三层缓存 + 进度 + 预热）、api.py（/api/ai/*，含 NDJSON 流）
  server.py     create_app() 和入口
  templates/    library.html / players.html / player.html
  static/       三个页面的 CSS / JS（原生，无构建步骤）
  tests/        独立测试脚本，不需要 pytest
tools/demo_extract/   提取器（本应用不改它，只调用）
visualizer/           3D 回放。本应用加了 /replay 路由和批次下拉；顶栏改成
                      和库页面共用一份（static/css/topbar.css），上传页删了
```

### 状态机

```
discovered ──> extracting ──> ready
     │              │
     └──────────────┴──────> error
                            
missing   （文件从磁盘消失，记录保留）
```

### 去重

两级，都不需要用户操心：

1. **路径唯一** —— `path` 上有 UNIQUE 约束，重扫同一个文件不会产生第二行
2. **内容 MD5** —— 大小没变就复用已存的 md5（避免反复哈希 300 MB）；
   内容撞上已有记录时，新行直接复用那份 `.csz`，不重复提取

`.csz` 只在没有其它记录引用时才真正删除。

### 删除到底删了什么

删除的主体是**库记录**，磁盘文件另算：

| 目标 | 结果 |
| --- | --- |
| 库记录 | 一定删除 |
| `.csz` + 缩略图 | 一定删除，除非另一条记录还指向同一个 `.csz` |
| 原始 `.dem` | 默认留在原处；勾上「同时移走原始 .dem」才移到 `app/data/recycle/` |

勾选框**每次打开都是不勾的**，不记住上次的选择。库记录扫描一次就回来了，
删掉的比赛录像不一定还能再下到，所以危险的那一项不能有粘性默认值。
勾上以后确认按钮的文字会跟着变（「移除记录并移走 .dem」），
省得勾选框在页脚、按钮在眼前、点下去才发现不对。

选中的记录源文件全都已经丢失时勾选框会被禁用：那时「移走」没有对象。

`.csz` 跟着记录走是因为**扫描只认数据库**：没有记录指向的 `.csz` 再也不会被读到，
下次扫描会把 `.dem` 当新文件重新提取一遍，旧文件只是占着磁盘的孤儿。

因此删掉一条 `ready` 记录 = 下次要重新提取。**只是想让某个 demo 从列表里消失的话，
别删** —— 留着记录不影响任何事。

源文件挪不动（被别的进程占用）时整个请求返回 `409`，**记录也不会删**，
不会出现「文件没动但记录没了」的中间态。

---

## 数据目录

全部在 `app/data/` 下（已 gitignore），删掉它等于重置整个库：

```
app/data/
  config.json          设置
  library.db           库（SQLite，WAL）
  csz/                 <文件名>__<md5前8位>.csz
  thumbs/              每场一张 demo_<id>.png
  thumbs/_maps/        按地图缓存的雷达底图（带 .extent 边车文件）
  avatars/             <steamid>.<jpg|png|gif>，从 Steam 取的玩家头像
  ai/                  AI 预测缓存：<demo>/<权重>_s<stride>_r<回合>.json
  recycle/             批量删除时移过来的源文件
```

放在 `app/` 下面而不是仓库根目录，是因为**根目录的 `data/` 是放训练数据的**，
两者没有关系。早期版本用的是根目录 `data/`，所以启动时会做一次搬迁：
`db.migrate_legacy_data_dir()` 把整个目录挪到 `app/data/`，并把库里所有绝对
路径（`csz_path` / `thumb_path` / `watch_root` / 头像文件 / recycle 里的 `.dem`）
改写成新前缀 —— 否则目录搬了、路径还指着旧地方，库看上去就是空的。三道保险
让它绝不会误伤：两条路径都必须正好是默认值（测试把 `DATA_DIR` 指到临时目录，
不比对默认值的话就会把真实的库搬进临时目录）、旧目录必须真的有 `demo_files`
表（训练数据的 `data/` 没有）、新目录必须还不存在。搬迁只在启动时跑，跑完
根目录 `data/` 就没了，第二次自然是空操作。

`avatars/` 里的图是**下载下来的第三方图片**，不是 demo 的一部分。清掉它只会让
头像暂时变回名字首字，下次同步会重新下载。`player_avatars` 里的行会保留，
只把 `file_path` 清掉、状态退回 `pending`（见下面「存储管理」）。

体积可以在设置面板底部的**存储管理**里看，也可以在那边整类清掉。`recycle/` 里的
`.dem` 是最占地方的一块（这批 demo 的原始文件合计 1.3 GB，`.csz` 只有 44 MB）。

---

## HTTP 接口

前缀统一 `/api/library`，避免和回放的 `/api/*` 撞车。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/list` | 列表。`q` `status` `source` `map` `player` `steamid` `watched` `sort` `limit` `offset`。附带 `avatars`（已缓存头像的 URL） |
| GET | `/stats` | 总数、体积、按状态 / 来源 / 地图分组 |
| GET | `/demo/<id>` | 单条完整信息（含名单、记分板、关注命中） |
| PATCH | `/demo/<id>` | 改展示名 / 备注 / 来源 / 队名 / `swap_teams` |
| DELETE | `/demo/<id>` | 删除。`?delete_file=1` 连源文件移到 `app/data/recycle/`。回 `recycled` / `csz_removed` / `thumb_removed` |
| GET | `/thumb/<id>` | 缩略图 PNG |
| GET | `/players` | 玩家索引（聚合）。`q` 同时匹配游戏内名字（含历史）、Steam 昵称、SteamID。`watched` `sort` `limit` `offset` |
| GET | `/players/search` | `q` `limit`，按匹配度排序的候选（下拉用）。每项带 `score` / `match` / `match_text` |
| GET | `/players/<steamid>` | 单个玩家：总计 + 地图分布 + 用过的名字（`name_history`）+ 头像状态 + Demo 数 |
| GET | `/players/<steamid>/stats` | 这位玩家的**传统统计**：每场明细 + 跨场合计（首杀率 / 首死率 / 先手开枪 / 侧背身 / 穿烟，页面上一律百分比、悬停给样本数）。`?progress=1` 走 NDJSON 流 |
| GET | `/players/<steamid>/ai` | 这位玩家的 **AI 统计**：每场明细 + 跨场平均（影响力 / 站位 / 对枪评分）。**会跑模型**，`?progress=1` 走 NDJSON 流 |
| PATCH | `/players/<steamid>` | `{watched: bool}` 加/取消关注。回**实际生效后**的状态和 `watched_resynced` |
| GET | `/avatar/<steamid>` | 头像图片。没有缓存过就是 404 |
| POST | `/avatars/sync` | `{steamids?, force?}` 排队同步头像。已有任务在跑时回 `queued: false` |
| POST | `/extract` | `{ids:[...], force?}` 排队提取。`force` 忽略已有的 `.csz` 重新解析。全部 id 都不存在时返回 400 |
| POST | `/demo/<id>/refresh` | 重新提取并重算元数据 |
| POST | `/open` | `{ids:[...]}` → 建批次，返回 `replay_url`。全部 id 都不存在时返回 400 |
| GET | `/batch/<id>` | 批次内容（回放页用） |
| GET/PUT | `/watch` | 监听目录与开关。`active` = 真的在轮询，`running` = 线程活着 |
| POST | `/watch/inspect` | `{path}` 校验目录、数一下有几个 demo |
| GET | `/watch/suggest` | 本机候选目录 |
| POST | `/watch/scan` | `{force, extract}` 立即扫描；`extract` 缺省跟随 `auto_extract` |
| GET/PUT | `/config` | 关注玩家、采样、自动提取、头像开关与 TTL 等。改 `watched_players` 时回 `watched_resynced: {checked, changed}` |
| GET | `/storage` | `app/data` 下各类缓存的文件数与体积，外加原始 `.dem` 的合计 |
| POST | `/storage/clean` | `{targets: ["csz" / "thumbs" / "avatars" / "recycle"]}`。回每类的 `files`/`bytes`/`errors` 和一份新统计。有任务在跑时 409 |
| GET | `/jobs`、`/jobs/<id>` | 任务队列 |
| POST | `/jobs/clear` | 清除已完成 |

另有 `GET /healthz`。

### AI 预测（不在 `/api/library` 前缀下）

回放页的开关直接调这两条；它们读的是**当前回放会话**，所以必须先 `/api/load` 一场。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/ai/status` | 模型在不在、哪两份权重、什么设备。**不加载权重**（毫秒级），页面启动时问一次 |
| GET | `/api/ai/round/<idx>` | 一个回合的完整预测。`?refresh=1` 跳过缓存重算，`?stride=16` 改时间分辨率 |
| GET | `/api/ai/round/<idx>?progress=1` | 同上，但回 **NDJSON 流**：一行一个 `progress` / `result` / `error` 事件（进度条用） |
| GET | `/api/ai/stats` | 整场比赛的 **AI 统计**（五个平均值，逐人 + 每回合明细）。冷启动 = 逐回合跑模型（几分钟），之后读整场汇总的磁盘缓存；`?refresh=1` 重算 |
| GET | `/api/ai/stats?progress=1` | 同上，NDJSON 流 |
| GET | `/api/ai/models` | 写死的那几份权重在不在（只查本地文件，**不联网**）。库页面的「下载模型」横幅读它；`download` 字段顺手带上最近一次下载任务 |
| POST | `/api/ai/models/download` | 把缺的权重下下来，跑在 jobs 队列里（`scripts/download_models.py`），返回 `job_id`。`?force=1` 已有也重下；已经有一个在跑就返回它，不会排出第二个 665 MB |

`/api/ai/models*` 这两条不需要先 `/api/load` 一场 —— 它们和回放会话无关，是
"这台机器的 `models/` 齐不齐"。

`/api/ai/round/<idx>` 的返回（约 130 KB，`Cache-Control: no-store`）：

```
samples    每个预测点的 tick
stride     预测点间隔（默认 16 个采样点，逐 tick 的 demo 就是 4 Hz）
spatial    win_ct / alive[10] / can_kill[10] / next_death[11] / next_kill[11]（都是概率）
duels      每一场对枪：a（受害者）/ b（击杀者）/ i0..i1（帧）/ t0..t1（秒）
           i1 = 受害者最后一个活帧；回合结束那一枪的 i1 就是最后一帧
duel       a / b / p（P(A 赢)）/ t0 / t1（窗口起止秒数，null = 这一段没有对枪）
models     用了哪两份权重、哪个设备
```

流式的每一行长这样（`\n` 分隔，`mimetype: application/x-ndjson`）：

```
{"type":"progress","pct":0.42,"stage":"两两遮挡 2048/6828 帧…","elapsed":6.1}
{"type":"result","payload":{…},"elapsed":23.9}
{"type":"error","error":"…","kind":"models","elapsed":1.2}
```

错误：没有会话 `400`、数据不满足前提 `400`（`kind: "features"`）、
模型 / torch 不可用 `503`（`kind: "models"`）。**流式路径上 HTTP 头早就发出去了**，
所以错误也是一个 `error` 事件（`kind` 同上一行）。

---

## 采样

设置里可选三种，默认**每个 tick**：

| 模式 | 含义 | 取舍 |
| --- | --- | --- |
| `tick` | 每 tick 一个采样（≈64 Hz） | 最完整，`.csz` 最大 |
| `interval` | 每 N 秒一个采样 | 体积小 |
| `stride` | 每 N tick 一个采样 | 体积小 |

⚠ **下蹲过渡只有约 200 ms（13 tick）**。降采样会把这条斜坡吃掉，回放里的蹲起
就不连贯了。想看完整过程就别降采样。

提取本身没有进度回调，所以任务面板只报**阶段 + 已用秒数**，不编造百分比。

---

## 测试

不需要 pytest，直接跑：

```powershell
pwsh -File app/tests/run_all.ps1        # 八个都跑
```

或者单独跑：

| 脚本 | 覆盖 |
| --- | --- |
| `app/tests/test_library.py` | 来源推断、分队（含换边与替补）、记分板、SQLite、路径校验、**坏 demo 的容错**、崩溃恢复、配置默认值合并、任务队列停止时的行为 |
| `app/tests/test_radar.py` | 坐标变换、采样、真实 demo 对齐 |
| `app/tests/test_players.py` | 玩家聚合（同名前不同账号、改名、坏 JSON、胜负方向）、SteamID 换算、头像地址白名单、图片嗅探、缓存 TTL、失败记录、批量同步 |
| `app/tests/test_api.py` | 全部 HTTP 接口、错误分支、批次交接、玩家 / 头像接口 |
| `app/tests/test_ui_wiring.py` | 前端接线：三个页面的元素 id、`data-*` 钩子、JS 里调用的 API 路径是否真的存在、回放页集成 |
| `app/tests/test_ai.py` | AI：对枪窗口（只认敌人 / 不认自杀队友 / 死前 3 秒 / 终点取最后一个活帧 / 回合结束那一枪挂在最后一帧 / 重叠归谁）、4 秒窗口的补齐与 `seconds` 对零、进度事件（只增不减 / finish 后变 result）、缓存键与版本作废、写死的扁平 `<run>.pt` 路径与 `<run>.yaml` 配套（老布局 `last.pt` + `config.yaml` 仍认）、三条 `/api/ai/*` 路由的错误码与 NDJSON 流 |
| `app/tests/test_ai_stats.py` | AI 数据统计：击杀前 2 秒的探针（时间轴是 tick 不是采样下标）、五个指标的**手算**用例（击杀 / 死亡影响力、按胜负补最后一个采样点、站位风险只算活着、对枪评分 `1-p` / `-p`）、跨回合与跨场的聚合口径（先加再除）、一次比赛里坏掉的回合只记 `errors` 不拖垮整场、比赛级缓存（第二次不再跑模型）、玩家页两个端点的合计算法 |
| `app/tests/test_download_models.py` | 权重下载：要下哪些文件由 `PINNED_WEIGHTS` 推出来（`.pt` 与同名 `.yaml` 必须成套、落盘路径就是网页读的那几条）、`inspect()` 不联网且把几百字节的 LFS 指针文本算成"坏的"、站点优先级与**官方挂掉自动换镜像**（探测时就用镜像 + 下到一半失败再换，两家都挂要报出各自原因）、`local_dir` / `force_download` / 进度递进、没装 hub 时给一句能照做的话、`/api/ai/models*` 两条路由（齐了不排队、连点两下只排一个）。`hf_hub_download` 换成桩，**不下载 665 MB** |

`test_ui_wiring.py` 值得单独说一句：这里没有浏览器，界面点不了。它静态地交叉检查
每个 `.js` 引用的元素 id 是否在自己的 `.html` 里、每个 `/api/library/...`
路径是否在 Flask 的 url_map 里、页面里 `href` 的内部链接是否有对应路由。
URL 打错一个字在运行时只是个 404，在这里会直接报出来。

`test_players.py` 的**网络全部被替换掉**（`avatars._http_get` 是唯一一个出口），
所以它跑起来不依赖 steamcommunity.com 是否可达，也不会因为对面慢而挂住。
真实网络只在手工验证时用。

`test_radar.py` 的「真实 demo 对齐」需要一份 `.csz`，找不到就 SKIP（退出码 0）。
可以显式指定：

```powershell
$env:CS2_CSZ = "visualizer\outputs\aurora-vs-vitality-m2-inferno.csz"
python app/tests/test_radar.py
```

> **单独跑 `test_library.py` 时退出码可能是 1，但结果是全过。** 这个测试会故意
> 把坏文件喂给 demoparser，Rust 的 panic 信息打到 stderr，PowerShell 把它当成
> `NativeCommandError`。`run_all.ps1` 里已经把 stderr 丢弃、只看真实退出码。

删除对话框另有一个 **node 手工冒烟脚本**，用 DOM 桩把 `library.js` 真跑起来，
按用户的动作走一遍（打开、勾选、确认、取消），再检查它发出的请求：

```powershell
node .build\smoke\delete_dialog.mjs
```

静态检查只能证明代码里出现了 `withFile` 这个词，证明不了「不勾就发 `delete_file=0`」。
这个脚本证明得了，所以它盯的是行为而不是字符串。它不在 `run_all.ps1` 里，因为
那套检查只依赖 python。

回放那边的原有检查照旧：

```powershell
pwsh -File visualizer/tests/run_all.ps1
```

---

## 踩过的坑

记下来是因为这几个都不会报错，只会安静地做错事。

### 1. demoparser 的 panic 不是 `Exception`

demoparser 是 Rust 写的，**残缺的 demo 会让它 panic**，PyO3 把 panic 抛成
`PanicException` —— 这个类故意继承自 `BaseException` 而不是 `Exception`。
所以 `except Exception` 抓不住它：

- 一个下到一半的 `.dem` 会让整次目录扫描中断
- 在 watcher 线程里会让**目录监听永久静默死掉**
- 在任务队列里会让 worker 线程死掉，之后所有任务卡在「排队中」，
  而那个失败的任务永远显示「进行中」

而且这个类型**没法按名字 catch** —— 它没有作为属性暴露在 `demoparser2` 或
`pyo3_runtime` 上，`except` 里写不出来。所以 `probe.py` / `ingest.py` /
`jobs.py` / `watcher.py` 都是 `except _FATAL: raise` + `except BaseException`。
`test_library.py` 会真的触发一次 panic 并断言它不是 `Exception` 子类 ——
这条推理是被测过的，不是写死的注释。

### 2. 采样密度是按面积算的，别再加每三角形上限

雷达底图一开始每三角形最多采 4096 个点。真实地图有几万个三角形，看不出问题；
但一块由少数大三角形组成的地板就被饿死了 —— 一个 2×2 的测试地板只覆盖 1.5%。
更糟的是这在真实地图上也有影响：选手位置落在采样不足的大地板三角形上时，
会被误判成「不在地图上」。去掉上限、只留全局预算后，真实 demo 的对齐率
从 **91.6% 涨到 98.1%**。

### 3. `visualizer` 的坐标变换写错不会报错

`scene.js` 是 `three = (gy, gz, gx) * SCALE`，地图 OBJ 原样放进场景。
缩略图必须严格反过来。写错了图照样渲染，只是选手在穿墙 —— 所以
`test_radar.py` 拿真实坐标投影，要求命中率高于四个错误变换。

### 4. pandas 会把带点的列名改掉

`parse_ticks(["CCSTeam.m_szClanTeamname", ...])` 返回的 DataFrame，
用 `itertuples()` 会把这一列改名成 `_1`，于是队名全部查不到、静默退回
「Team A / Team B」。必须用 `to_dict("records")`。

### 5. `Config(path=CONFIG_PATH)` 会把路径冻进签名

默认参数在**类定义时**求值，所以事后改 `config.CONFIG_PATH` 没有任何效果，
测试会悄悄读写真实的 `app/data/config.json`。改成 `path=None` + 调用时解析。

### 6. 崩溃会把记录卡在「提取中」

提取只在本进程里发生，所以启动时不可能有任务在跑 —— 任何还写着 `extracting`
的记录都是被中断的（崩溃、被 kill、断电）。不管它就会永远显示「提取中」，
而且**永远不会有东西来收尾**，因为 watcher 只排队新发现的文件。

所以启动时会把这些记录改回 `discovered` 并打印一行；扫描时还会把「没有 `.csz`
的未完成记录」一起重新排队（`recovered` 计数）。有 `.csz` 的、`error` 的、
`missing` 的都不会被自动重试。

### 7. 「未完成」的判定会重复排队

接着上一条：watcher 每次轮询会把「新发现的」和「没有 `.csz` 的未完成记录」
合起来排队。但一条记录从**排队**到 worker 真正开始跑之间一直是 `discovered`，
所以天真的写法有两个坑：

- 每隔几秒把同一个 demo 重新排一次队（一个长提取排在队列里时尤其明显）
- 本次刚发现的文件同时出现在两个列表里，被**排两次队**，同一个 demo 提取两遍

前者用「本进程已排过队的 id 集合」挡掉；后者用 `dict.fromkeys` 去重。
两个都有测试盯着（`watcher re-queue guard`）。

### 8. 同一个「删除」有两套行为

第一版里卡片上的删除按钮直接删记录，批量删除却弹一个 `confirm()` 让你二选一
—— 而 `confirm()` 只有「确定 / 取消」，于是**「取消」被当成了「只删记录」**。
结果就是：单个删除碰不到 `.dem`，批量删除却能删，同一个动作两种能力，
而且没有任何一处说明这一点。

现在单条和批量走同一个 `#delete-modal`。`library.js` 里已经**不允许**
再出现 `confirm()` / `alert()`，`test_ui_wiring.py` 会拦（检查代码而非注释 ——
第一版这个检查被自己的注释里那句 `confirm()` 绊倒了，所以有了 `strip_js_comments`）。

不过修完这一版还是不对：页脚改成两个平级按钮（「保留」/「移到回收站」）之后，
**看起来像两个都得挑一个的并列选项**，而不是「一个默认 + 一个例外」。实际用起来
每次都要先读懂哪个是默认，而且「保留」被放在左边当主按钮，视觉上也不像主按钮。

现在的形态是一个确认按钮加一个勾选框：库记录一定删，`.dem` 默认保留，
勾上「同时移走原始 .dem」才动文件。勾选框每次打开重置为不勾，
确认按钮的文字随勾选变化。测试盯着三件事：标记里不能出现 `checked`、
`box.checked = false` 必须在每次打开时执行、取消判断必须是 `withFile === null`
而不能是 `if (!withFile)` —— 后者会把「只删记录」这个正常结果当成取消。

顺带修掉一个连带问题：删除对话框的体积和状态统计原来用 `S.items`（当前页）
去查选中的 id，而**选择是跨页保留的**，所以跨页选择时这些数字会静默偏小。
现在找不到的单独列出来并标注「不在当前页」。

### 9. demo 里的名字不是 Steam 昵称

第一版玩家页打算直接用 Steam 昵称当玩家名，理由是「那才是他现在的名字」。
实际拉下来一看，这批 demo 里没有一个人对得上：

| demo 里的名字 | Steam 昵称 |
| --- | --- |
| ZywOo | Herbus doubledoor |
| ropz | mmmmm |
| apEX | Petite Pute |

职业选手的比赛名和账号名是两回事。所以现在两个都留着：**demo 里的名字是主名字**
（那是这场比赛里别人看到的名字，也是「关注玩家」和搜索匹配的东西），
Steam 昵称只作副标题。反过来做的话，用户按自己认识的名字搜会搜不到。

### 10. 头像地址是第三方给的，不能拿来就用

`avatar_url` 来自 Steam 的响应，属于外部输入。直接 `urlopen` 就等于让对面
决定服务器去连谁，所以先过一层白名单：只允许 `https` 且 host 是
`*.steamstatic.com` 或 `steamcdn-a.akamaihd.net`。

顺带两个细节：

- **扩展名从字节里嗅，不从 URL 里读。** Steam 同一张图会给 `_full.jpg` 和
  `_medium.jpg` 两个地址，动图则是 `.gif`，只信 URL 迟早存出一个内容是 GIF
  的 `.jpg`。
- **下载有上限**（4 MB）。真头像 10–30 KB，超过就是坏响应。

### 11. 同步任务不合并的话，扫一次能排 60 个任务

提取成功后排头像任务，而一次扫描会注册 60 场 demo、触发 60 次「提取完成」。
不合并就是 60 个任务排队，每个都去查同一批 SteamID。

合并的做法是一个进程内只允许一个头像任务：后来者不排队，只把「跑完再来一轮」
标记上；那个任务**开始时重新查一遍库里的全部 SteamID**，所以中途提取完的场次
也覆盖得到，最后再补跑一轮。一个容易写错的点：补跑的那一轮不能用
「有没有任务在跑」来判定 —— 提交它的那一刻，**当前这个任务自己还处于 running**，
检查会把自己要排的任务挡掉，于是补跑永远不会发生。

### 12. 停掉的任务队列会把活留给下一次启动

`test_api.py` 每个分组都重建一次 app，而 `jobs.manager` 是模块级单例。
`stop()` 原来只置停止标记、等线程退出，**队列里没开始的任务原封不动留着**，
下一次 `start()` 就把它们接着跑。而那时 `config.LIBRARY_DB` 已经指向下一个分组的
临时库，于是上个分组为了假 demo 排的提取任务落到了下个分组刚插进去的记录上，
把它改成 `error`。表现是列表/筛选那组莫名其妙挂三条，单独跑 `test_api.py` 却全过
（连跑三次都是 200/0），因为这是竞态。

修法是 `stop()` 顺手把队列清空、把这些任务标成 `cancelled`。它们本来就没开始，
留着等于让「停止」这个说法不成立。`test_library.py` 里加了一组回归测试：
让一个任务卡在运行中，后面排三个，`stop()` 之后断言那三个一个都没跑、状态是
`cancelled`，并且重新 `start()` 也不会把它们捡回来。

### 13. 取消关注之后，之前解析过的比赛还亮着

`watched_json` 是**派生数据**：记分板 × 关注名单。卡片名单的高亮、`★ N 关注`
角标、`只看关注` 的 SQL 过滤，读的都是每条记录里存的那一份副本（这样列表接口
不用每行现算，过滤也能走 SQL）。问题是改名单时只写了配置，没人去动那些副本，
于是取消关注之后，已经入库的比赛继续亮着 —— 用户刚说了不感兴趣，唯一没听见的
就是那个高亮。

当时的说法是「点一次『刷新信息』就会重新高亮」。那句话技术上没错，但它把
一个数据一致性问题变成了用户的日常操作，而且只在**设置弹窗**保存时提示过，
从玩家档案页点「取消关注」根本没有提示。

修法是加一个 `ingest.resync_watched()`：从**已经存在库里的记分板**重新算一遍
`watched_json`，改名单的接口（`PUT /config` 和 `PATCH /players/<steamid>`）
调完就地调用。**不需要 `.csz`**，所以是纯 JSON 运算、毫秒级，可以直接放在请求里；
要重新加载 `.csz` 的话每场 7–17 秒，就只能扔后台了。

两个细节：

- 匹配规则抽成了 `metadata.match_watched()`，提取时和重算时走**同一个函数**。
  两份实现迟早会漂移，而漂移的表现是「同一个玩家在提取前后判定不一致」。
- 写入走 `db.set_watched()` 而不是 `update_demo()`，只动那一列。走 `update_demo`
  会顺手给**全库每一行**盖上新的 `updated_at`，一次改名单变成一次全库写。

回归测试盯的是用户视角：取消关注之后 `/api/library/list` 返回的 `watched` 必须是
空的、`?watched=1` 必须回到 0 条，而且**不能**依赖任何逐场刷新。另外钉住了
「重复保存同一份名单时 `changed` 为 0」，以及「某一行记分板是坏 JSON 时跳过它、
不拖垮整个库」。

**缩略图上不再画关注玩家的描线**，这是故意的：那张图在提取时就烤成了像素，改名单
不会让它动，要同步只能把每场的 `.csz` 重新读一遍（7–17 秒一场）。卡片已经高亮了，
而卡片能零成本地跟着名单走。留一条对不上的粉线，比少一个装饰更糟。

### 14. 回放页看起来像另一个产品

回放页是 `visualizer` 自己的 `index.html`，被 `app.server` 原样挂在 `/replay`。
它自带一套顶栏：品牌块的样式和库页面不一样（15px 渐变字 vs 17px 纯色），
高度和留白也不同，而且**没有导航** —— 所以从 Demo 库点进回放，顶部那排菜单
就消失了，只能靠角落里一个 `← Demo 库` 小链接回去。

同一个顶栏在 `library.css` 和 `style.css` 里各写了一遍，这是必然结果：改一边
另一边不会跟着变。

修法是抽出 `visualizer/static/css/topbar.css`，四个页面（库 / 玩家 / 玩家档案 /
回放）共用同一份**样式和结构**。放这个目录是因为只有 visualizer 的 static
两台服务器都会挂：`python -m app.server` 会把 visualizer 整个挂上（`/static/...`），
而 `python visualizer/server.py` 单独跑时也只有它。库页面本来就一直在用
`/static/logo.svg`，同一个道理。

顺带统一了一条规则：**当前页那一项是 `<span>` 不是 `<a>`**。在回放页上这不只是
好看 —— 点自己会重新加载并把 `?batch=` 丢掉。四个页面现在都这样，测试直接断言
「四个页面的导航标签一样、各自链另外两个、自己那项标记为当前」。

### 15. 回放页的上传界面删了

回放页以前自带一个上传页：拖拽 `.dem`/`.csz`、浏览 `examples/` 和 `~/Downloads`、
选采样方式。它和 Demo 库是**两个入口**，而两边的说法还不一致：上传页问「用哪种
采样」，库那边是提取设置说了算，而 `.csz` 在提取时采样就已经定死了。

删干净的地方（删一半最糟：页面上没有了、`main.js` 还在找它的元素，
`bindEvents` 第一行就抛，整个 3D 场景建不起来）：

- `index.html`：上传层、采样下拉、本机文件列表、进度条、`↺ 新文件` 按钮
- `main.js`：`samplingParams()`、拖拽/选文件/示例列表的绑定、`showLoadingBar`
  （进度改用已有的 `showLoading` 遮罩）、`startLoad({file})` → `startLoad(path)`
- `server.py`：`/api/examples`、`/api/load` 的 multipart 分支、2 GB 的
  `MAX_CONTENT_LENGTH`（没有上传了，留着这个上限只会让人以为还有上传）
- `style.css`：`.landing` / `.drop-zone` / `.example-*` / `.loading-bar` 等
  一整套落地页样式

`/replay` 现在没有 `?batch=`、服务端也没有活着的会话时，显示一屏「去 Demo 库」的
引导 —— 正常路径看不到它，但它比一个 404 或者一个空白页好。

### 16. 顶栏放在 `#app` 里面，等于没有顶栏

第 14 条只修了一半。顶栏确实四页共用了，但它**在 `#app` 里面**。

这在「正在回放」时看不出来：`#app` 是显示的，顶栏自然也在。可是 `/replay` 不带
`?batch=`、服务端又没有活着的会话时会走空状态，而空状态把整个 `#app` 藏起来 ——
**顶栏连同导航一起消失**。于是从 Demo 库点「可视化回放」，落地的是一个顶部什么都
没有的页面，读起来仍然是「跳到了另一个网站」，而不是切了个页签。

根因是把外壳和内容混在了一个容器里。顶栏是**外壳**，不是回放内容的一部分，所以它
现在和 `#empty` / `#app` 平级，直接挂在 `<body>` 下。高度由 `body` 的纵向 flex 分：
顶栏拿它要的那一行，`#empty` 和 `#app` 各自 `flex: 1; min-height: 0` 拿剩下的
（原来两处都是 `height: 100vh`，多一个兄弟节点就会溢出）。

顺带：顶栏右侧那排（地图 / 回合 / 采样 / `⬇ .csz`）以前无条件显示，没有数据时写着
「加载中…」、下载按钮点了只会报错。现在它们带 `hidden` 起始，`onDataReady` 里才放
出来 —— 没数据时的回放页顶栏和库页面长得一模一样。

测试用**文档顺序**钉住这个：`#topbar` 在 `#app` 打开之前出现，就不可能在里面。

### 17. 数据目录搬到 `app/data/`

库原来放在仓库根目录的 `data/`，但那个目录是放训练数据的。搬的时候有个陷阱：
**库里存的全是绝对路径**（`csz_path` / `thumb_path` / 头像文件 / recycle 里的
`.dem`），只 `mv` 目录的话，文件都在新地方、每一行却还指着旧地方 —— 库看上去是
空的。

所以 `db.migrate_legacy_data_dir()` 搬完还要把库里所有路径列改前缀。三道保险让它
绝不会误伤：

- **两条路径都必须正好是默认值。** 每个测试都会把 `config.DATA_DIR` 指到临时目录，
  不比对默认值的话，测试跑一次就会把开发机上真实的库搬进临时目录然后删掉。
- **旧目录必须真的有 `demo_files` 表。** 训练数据的 `data/` 里是 parquet，没有这
  张表，所以不会被当成库搬走。
- **新目录必须还不存在**，所以不可能覆盖一个活着的库。

调用点在 `create_app()` 里、`init_db()` **之前** —— `init_db()` 会建出数据目录，
一个刚建出来的空目录会让搬迁以为「已经搬过了」。

### 18. 界面文案开始解释自己

设置里曾经有一句「目录要手动添加，没有可靠的自动探测方式。来源按 demo 的服务器名
和文件名判断，不看目录，所以一个目录里混放几个平台也能分对。」玩家页的空状态写着
「玩家是从记分板里汇总出来的」，档案页挂着一个空 section 说「后面接入的分析模块放
在这里」。这些都不是写给用户的 —— 是在替实现方式辩护、复述实现、预告以后要做的事。
用户要的是扫一场 demo，不是读一段设计说明。

`test_ui_copy` 现在拿 `NARRATION_TELLS` 扫**去掉注释的**源码，命中的就是这类句子。
注释不算，因为推理本来就该写在注释里 —— 这个检查针对的是会出现在屏幕上的 `title=`
和模板字符串。同一条测试早就在管 `——` / `⚠` / `✓` 这类装饰性符号了。

顺带一提，回放页那 13 条开关说明当时夹着 `**加粗**`（HTML 属性里不解析 Markdown，
用户看到的是原样星号）、`⚠`、`——`，还有「取消勾选退回改造前那个圆点」这种像 commit
message 的自述。现在每条一句话，只留技术事实。

### 19. 名字列表不能靠 `json_group_array(DISTINCT ...)` 排序

玩家页要列出一个人用过的**所有**游戏内名字，按最近使用倒序。第一反应是
`json_group_array(DISTINCT name)`，但 `DISTINCT` 在聚合里会让 SQLite 走一棵 b-tree，
出来的顺序是按值排的 —— `Wicadia` 会排在 `amınaKE` 前面（大写字母的码位更小），
和「最近用的」没有关系。

能排序的写法是相关子查询套一层 `GROUP BY`：

```sql
(SELECT json_group_array(n) FROM (
    SELECT b.name AS n FROM sb b
    WHERE b.steamid = a.steamid AND IFNULL(b.name, '') <> ''
    GROUP BY b.name
    ORDER BY MAX(b.added_at) DESC, MAX(b.demo_id) DESC
 )) AS names_json
```

`ORDER BY` 里只能出现聚合，因为外面套着 `GROUP BY`；而 `ORDER BY` 里出现 `DISTINCT`
没选中的列会直接报错，所以 `SELECT DISTINCT n ... ORDER BY b.added_at` 这条路本来也走不通。
SQLite 允许 FROM 子句里的子查询引用外层的 `a.steamid`（标准 SQL 不允许），这一条是
靠 SQLite 才成立的 —— 换库要重写。

### 20. 「清缓存」是两件事，少一件就留下幽灵记录

存储管理里清 `.csz`，如果只删文件不动数据库，库会继续显示这些 demo「已入库」，
点进去回放页拿到的是一个不存在的路径。所以每个清理动作都是**文件 + 数据库**一对：

| 清什么 | 文件 | 数据库 |
| --- | --- | --- |
| `.csz` | `app/data/csz/*` | `csz_path = NULL`，`ready` 退回 `discovered` |
| 缩略图 | `app/data/thumbs/**`（含 `_maps/`） | `thumb_path` / `thumb_key` 都清 |
| 头像 | `app/data/avatars/*` | `file_path = NULL`、`status = 'pending'`，**`persona_name` 留着** |
| 回收站 | `app/data/recycle/*` | 没有对应记录，不用管 |

`thumb_key` 必须跟着 `thumb_path` 一起清：它记的是「这张图是从哪份 `.csz` 生成的」，
留着会让一张不存在的缩略图看起来像缓存命中。

头像那一格最容易做错：清图不该顺手把 `persona_name` 删了。那是花过一次网络请求换来的
文字，玩家页的标题正用它；清了图顶多退回名字首字，清了名字整页 Steam 昵称全空。

清理请求在**有后台任务在跑时直接 409**。提取是在解析结束那一刻才写 `.csz` 的，
中途把目录删掉，要么任务失败，要么留下一个没人指向的文件。

`ready` 退回 `discovered` 是有意的：`unfinished_ids()` 下次扫描会把它们捞回来重新提取，
所以整类清 `.csz` 是可恢复的操作，不是单向损失。

---

## 已知限制

- **没有 Steam / 5E / 完美 / FACEIT 安装时探测不到候选目录**，只能手动填路径。
  这是设计如此，不是 bug：这几个客户端的安装位置没有可靠的探测方式。
- **队名 A/B 的归属靠服务器里的战队名**，不是靠文件名 —— 文件名只说明是哪两队打的，
  不说明哪个是 A。拿不到战队名时回退到文件名，再回退到「Team A / Team B」。
  判反了就用「交换主客队」。
- **监听目录不校验是否存在**。外置盘或网络盘没挂载时目录本来就不存在，
  这是正常配置；watcher 会跳过它，等它回来再继续。界面在添加前会先探测并告诉你
  找到了几个 demo，所以打错字在那边就会被拦住。
- **提取是单 worker 的 FIFO 队列**。手动提取会排在自动提取后面，任务面板里能看到。
- **不做 LLM 分析**。所谓「分析」就是 3D 回放。
- 缩略图生成失败不会让入库失败，只是没有图。
- **头像依赖 steamcommunity.com**。国内直连不通时全部退回名字首字，玩家页其余部分
  照常工作。这不是可以绕过的：头像只存在于 Steam 那边，demo 里没有。
- **动图头像不追**。静态头像是从 mini-profile 的 JSON 里拿的，一次请求一个人；
  动图只出现在个人资料页的 HTML 里，为它多拉一个约 1 MB 的页面（每人）不值。
  拿到的是动图的静态版本，不会动。
- **玩家统计只覆盖库里的 demo**。它不是战绩平台，没有跨库的历史；
  Demo 一删，那些场次立刻从统计里消失。
- **ADR / KPR 按回合数算，不是按场次**。回合数来自 `.csz` 的 `total_rounds`，
  所以它和记分板里那个「总伤害」是同一个口径，不是 HLTV 的 ADR。
  没有 Rating —— 那需要一个来源不明的公式，不如不给。
- **玩家页的比赛列表按文件时间倒序**，因为 demo 里没有真正的比赛日期
  （见上面「日期」那段）。
- **搜索候选是全表打分的**。`/players/search` 把整个玩家视图拉出来在 Python 里算分，
  因为子序列匹配（`wcda` → `Wicadia`）写不成 SQL。玩家数按 SteamID 算，几万条之前
  都不慢；真到了那个量级要加一层 SQL 预筛，但预筛会漏掉模糊命中，得想清楚再动。
- **SteamID 搜索有最短长度**。`7656119` 是所有 SteamID64 的前缀，太短的纯数字查询
  会把整个库排成并列第一，所以少于 5 位的纯数字不参与 SteamID 匹配。
- **存储管理按类清理，不按文件清理**。要删单个 `.csz` 或某张头像，走 Demo 库的删除
  或者手动删文件（数据库里的路径会在下次扫描时按「文件丢失」处理）。
