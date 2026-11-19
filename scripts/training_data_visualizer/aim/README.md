# aim · 训练数据可视化

一个**只读本地 npz** 的 web app：扫 `data/demos/prepared/aim/{train,valid,test}/*.npz`，
选一个样本，**逐帧**看全场 10 个人的轨迹、俯视雷达、两两**遮挡矩阵**、道具和特征表。
再给它一个训练产物（目录里有 `config.yaml` + `best.pt`），它还会把**模型的预测**画出来：
每一帧的预测曲线（`T` 个点：第 t 列 = 看完第 0..t 帧时的判断），见下面「模型预测」。

```bash
python -m scripts.training_data_visualizer.aim
# 浏览器打开 http://127.0.0.1:5117/

python -m scripts.training_data_visualizer.aim --port 5118 --open
python -m scripts.training_data_visualizer.aim --data-dir data/demos/prepared/aim
python -m scripts.training_data_visualizer.aim --model-dir models/aim/20261108-131429
```

依赖只有 **numpy**（读 npz）。**不 import** `scripts.training_data` / `demo_extract` /
`demoparser2` / `numba` —— 在一台只装了训练依赖的机器上也能开。模型预测是**可选**的：
`torch` / `pyyaml` 只在第一次要预测时才 import（见 `predict.py`），没装或者用
`--no-model` 时页面其余部分照常，预测那块显示一句原因。

## 这一版和上一版的区别

样本布局换了（`scripts/training_data/tasks/aim.py`）：

| 旧版 | 这一版 |
|---|---|
| `atk` / `tgt` 两条 `(T,20)` 轨迹 | `ctx_player (T,10,20)` + `ctx_player_mask (T,10)`，**全场逐帧** |
| `ctx_player (10,20)` **只有首帧** | 人人都有整条轨迹；`mask` 的 0/1/2/3 = 死 / A / B / 其他人 |
| `atk_view` / `tgt_view` 两张 `(192,256,5)` 渲染图（**首帧**） | **删掉**，换成 `obstruction (T,10,10)` 逐帧两两遮挡 |
| `atk_weapon` / `tgt_weapon` 两条 | `ctx_player_weapon (T,10)`，-1 = 无武器 / 已死 |
| `ctx_projectile (32,4)` **只有首帧** | `ctx_projectile (T,32,4)` 逐帧 |
| `global_feat (5,)` **只有首帧** | `global_feat (T,5)` 逐帧（炸弹会被搬着走、计时器一直在走） |
| 窗口跨度 0.1–4 s **随机** | **固定 4 s** → 64 Hz 下 `T = 257`，全样本同长 |

所以页面上**没有「视角图」那一块了**，`GET /api/view/...` 这个接口也一起下线；那条路
本来就是"把首帧渲染图转成 PNG"。删它的理由在 `tasks/aim.py` 的 docstring 里：图取的是
窗口**第一帧**，而击杀在窗口**末尾**，中间 0.1–4 秒的走位、绕掩体一点不在图里；实测把
视角图换成别的样本的图，末帧 acc 只掉 0.6 个点，而"谁挡着谁"这个**关系**是真的有用
（449 条实测：**死前最后一帧 86.0%** 的样本里击杀者看得见受害者，**窗口第一帧只有
3.6%** —— 窗口现在是固定的 4 秒，起点离击杀更远，这个对比比旧版随机窗口的 17.8% 更极端）。
（壳版是 91.5% / 4.9%；烟雾改成凸包体积之后，原本"隔着烟"被漏记成"通"的那部分补上了，
所以末帧的"看得见"降了 5.5 个点 —— 这 5.5 个点正是隔着烟击杀的样本。）

> 样本粒度是**每次击杀一条**，外加**回合结束那一枪**（`tasks/_util.py::late_kills()`：
> 它落在 `ticks[-1] + 1`，实测 63% 的回合是"打死最后一个敌人、回合随即结束"）。它的
> 窗口终点同样是"受害者最后一个活帧"，所以 `T` 仍然恒为 257。

---

## 页面上有什么

| 区块 | 内容 |
|---|---|
| 顶栏 | 样本下拉（按 split 分组）、上一个 / 下一个、重新扫描 |
| 概览 | 文件名、地图 + `map_id`、帧数、时长、采样率、维度、双方队伍、爆头 / 伤害、**A / B 的玩家下标**、本窗口 A 开了几次枪 |
| 逐帧视角 | A / B 各自的第一人称（**合成**画面，不是渲染图）：目标包围盒、全场存活玩家、准星与整段 yaw/pitch 轨迹、开枪帧。HUD 逐帧给武器图标 + 位置 / 速度 / hp / 护甲 / 拆弹器 / 本帧是否开枪 / **存活与敌方人数** / **最近的敌人 + 遮挡结果** / **A→B 遮挡** / 角误差。下面还有**一格 20 维的原始 `[0,1]` 数组**（每格底部细条长度 = 原值） |
| 时间轴 | 进度条（红/蓝刻度 = A/B 开枪帧）、播放/暂停、±1 帧、速度、`t = seconds[i]`、回合时间 |
| 模型预测 | `P(A 击杀 B)` 随**帧号**变化的曲线（每帧一个点，共 `T` 个）+ 虚线的反方向；旁边是第 1 帧 / 当前帧 / 末帧 / 首次过半的读数 |
| 逐帧全场 | 俯视雷达（**当前帧**的 10 个人、轨迹尾巴、道具点、炸弹）+ `global_feat`（**当前帧**的炸弹 / 已安放多久 / 回合时间）+ `ctx_projectile`（**当前帧**道具表）+ `ctx_player`（**当前帧** 10 人表：角色 / 队伍 / 武器图标 / hp / 护甲 / 蹲 / xyz / **四格遮挡**） |
| 遮挡矩阵 | 当前帧的 10×10 网格：行 i 的眼睛看列 j 的身体，`0` 通（绿）/ `1` 烟（蓝）/ `2` 墙（灰）；**`通` 但观察者没朝那边看的格子加斜线**（遮挡和视野是两件事）；死人行列变暗、A/B 行列高亮、交叉两格加黄框。旁边是 **A→B 的整段色带**（点一下跳帧）和当前帧的"通/烟/墙"计数 |
| 当前帧特征 | `atk` / `tgt` 的 **20 列原值（文件里的 [0,1]）+ 逆变换后的物理值**，外加 `eye_z` / `box_h` 两个派生量；下面列出 npz 里所有 key / shape / dtype |

样本里**已经没有"只存一帧"的字段了**（`global_feat` 这一版也改成逐帧），所以**所有区块
都跟着进度条走**。

### 可分享的 URL

```
?sample=train/de_anubis_f0_r0_k001.npz   # 直接打开某个样本（id 就是 split/文件名）
&frame=128                               # 打开后停在第 128 帧
&trail=0&labels=0&boxes=0&grid=0         # 关掉某层（1/0）
&pred=0                                  # 不跑模型（CPU 机器上推理慢时用）
```

---

## 遮挡矩阵怎么读

`obstruction[t][i][j]`：

```
0  通     8 个包围盒顶点里至少有一个能直接从 i 的眼睛看到
1  烟     8 个顶点全被烟雾挡住（没有墙）
2  墙     8 个顶点全被地图 mesh 挡住
```

* **有方向**：`[i][j]` 是 i 看 j，两个人的眼睛不在同一个位置，所以 `[j][i]` 可以不同
  （实测只有 2% 的时刻不同）。
* **不看视野（FOV）**：只回答"挡不挡"，不回答"在不在视野里"。朝哪看是 `yaw/pitch` 的事。
  ⚠ 所以 **`通` 不等于"看得见"**。典型例子：`de_anubis_f0_r0_k001` 第 233 帧（回合
  63.41 s，A=#2 / B=#8，相距 659 HU）两个方向都是 `通` —— B 站在柱子后面只露出半个
  身子（A→B 的 8 个顶点里 4 个被墙挡住、4 个通），但 **B 的 yaw 偏离 B→A 138°**，
  B 其实背对着 A。页面上这格会画**斜线**：底色仍是遮挡结果，斜线表示"观察者没朝那边看"。
  换成 `visibility.js::pairVisibility()` 的位掩码说法，`BLOCK_FOV` 是**另一位**，
  和 `BLOCK_MESH` / `BLOCK_SMOKE` 并列，不进这个矩阵。
* **死人的行列恒为 0**，那不是"通"，是"没有意义"：页面按 `mask == 0` 把它们变暗。
* 判据和 `visualizer/static/js/visibility.js` 的 `pairVisibility()` 同一套（8 个顶点、
  `min` 取最轻的那个遮挡、包围盒半宽 16 HU、高 72 站 / 54 蹲、眼高
  `64.0938 − 18·crouch`），实现见 `scripts/training_data/obstruction.py`。
* 面板右上角那行读数把两件事分开写：`A(2) → B(8)：通 · A 视野内 1° ｜ B(8) → A(2)：通 ·
  B 背对 138°`；HUD 里的 `对手可见` 是合并判定（`否（通 · 背对 138°）` = 没挡住但没看）。

一条**硬语义**可以用来判断自己有没有看错：*死前最后一帧，A 应该看得见 B*（人是在
看得见的地方被打死的）。全量 449 条样本里 **86.0%** 成立（窗口第一帧只有 3.6%，正好是
反面对照），`selftest.py` 把这条钉死成断言（前 40 条 ≥ 70%）。

---

## 模型预测（可选）

给 `--model-dir` 一个**训练产物目录**（里面有 `config.yaml` 和 `best.pt`，就是
`training/aim/train.py` 的 run 目录），页面就会多一块预测面板。

不传 `--model-dir` 时自动取 `models/aim/`（以及老习惯的 `models/`）下**最新**的那个 run
（按目录 mtime），所以训练完直接开页面就是刚训的那个；也可以只给 run 名字
（`--model-dir 20261108-131429`），会去这两个目录里找。`--no-model` 关掉整块。

当前仓库里的 `models/aim/20261108-131429`（`best.pt`，epoch 17 / step 60333，
`valid/acc_last 0.811`）就是这一版布局训出来的，面板直接可用。

> 只有**旧样本布局**（atk/tgt + 视角图）的 `best.pt` 才会报"和新数据 / 新模型对不上"，
> 那种权重救不回来，只能重训 —— 接口没变，重训之后这块自动就能用。

模型是 decoder-only + 因果注意力，**一条样本一次前向**就吐出 `T` 个预测点，所以
"每帧一个预测"不需要循环：

| 列 | 看到了什么 | 页面上的叫法 |
|---|---|---|
| t (0..T-1) | 第 0..t 帧（同一帧的 A、B 都在这个预测点前面），时间戳 = `seconds[t]` | 拖到第 **i** 帧时看的是第 **i** 列 |

x 轴就是**帧号**（不是秒）：它和进度条共用同一个帧序。红竖线是开枪帧。

**两个方向一起算**：主曲线 `P(A 击杀 B)`（A = 磁盘上的击杀者，这里**不做**
`dataset.py` 那种 A/B 随机对调），虚线 `P(B 击杀 A)`。这对人恰好有一个杀掉了另一个，
所以两条线应当关于 0.5 镜像；面板上的 `自洽 |p+p′−1|` 就是这件事。

几个实现上的取舍：

* **输入不自己拼**：走 `training/aim/dataset.py` 的 `load_arrays` + `collate_aim`，
  和 `DataLoader` 是同一段代码。否则特征表 / 补齐逻辑一改，训练和可视化就会悄悄看
  不同的东西。
* **推理串行 + 缓存**：`ThreadingHTTPServer` 是多线程的，但模型只有一份，`Predictor`
  里一把锁；每条样本的结果按 `split/name` 做 LRU（24 条）。
* **`--no-model` 是真的不 import torch**：只有 numpy 的机器上启动不会因为缺 torch
  变慢或报错。

---

## 坐标 / 逆变换

样本里的 `ctx_player` 每一列都是 `[0,1]`（`tasks/aim.py::_SCALE`）。前端不做语义换算，
服务端把整列逆变换回物理量再发（`data.py`）：

```
x y z      (v01*2-1) * range + MAP_CENTER      → 世界 HU（脚底）
yaw/pitch  atan2(sin, cos)                      → 度
vx vy vz   (v01*2-1) * 300 / 1200               → HU/s
hp armor   v01 * 100
since_shot v01 * 100                            → 秒（>=99.9 是饱和）
helmet defuser team shot has_kill blinded       → 原样（0/1 或 CT=0/T=1）
```

`global_feat` 是 **`(T,5)` 逐帧**：`[x y z 安放秒数/50 回合秒数/200]`。回合时间的 0 点 =
本回合 `round_freeze_end`（和游戏里右上角倒计时一致），**不是**窗口起点；第 i 帧的回合时间
直接就是 `global_feat[i][4] * 200`，它和 `seconds` 严格同步（`global_feat[i][4]*200 -
global_feat[0][4]*200 == seconds[i]`，`selftest` 拿这条当断言）。**未安放时 xyz 是炸弹
携带者的位置**，对敌方是隐藏信息 —— 页面会标一句提醒。

旧样本（`global_feat` 是 `(5,)` 的首帧快照）也能打开：后端把它广播成 `T` 份，标题写
"首帧"，并在警告条里提示重造。

第一人称是**画出来的**，不是渲染图：把点投到等角距网格上（水平 90°、垂直 73.74°、4:3）

```
yaw_p   = atan2(dy, dx)
pitch_p = -asin(dz / |d|)          # pitch 正 = 向下看
sx = W/2 - (yaw_p   - yaw0)   / 90°    * W
sy = H/2 + (pitch_p - pitch0) / 73.74° * H
```

`sx` 的**负号**不能省：Source 的 yaw 从上方看逆时针增大，玩家的右手是
`f × up = (sin y, −cos y)`，所以 yaw 变大 = 往**左**看 = 屏幕**左边**（`view_render.py::
directions()` 里同一个符号，那边有详细说明）。写反就是左右镜像的第一人称。

### 武器图标

HUD 和玩家表里的枪都有 CS2 官方 HUD 图标，素材是 3D 回放器那份
（`visualizer/static/assets/weapon_icons/`，已 git 跟踪），本工具**不复制**，由
`server.py` 的 `/api/weapon-icons` 查表、`/weapon-icons/<file>` 转发；素材缺失时降级成
只显示名字。`map.json` 里漏登记的 **CZ75 Auto** 在 `server.py::_ICON_ALIASES` 里补了
一条（`weapon_cz75a_png.png`）。

俯视图的底图来自 `app/data/thumbs/_maps/{map}.*.png` + `.extent`（`app/radar.py` 生成的
缓存）。**没有底图的图会退化成世界坐标网格**。

---

## 数据体检会看到什么

* **旧版样本**（含 `atk` / `atk_view`）会被**明确拒绝**并提示重造 —— 列的含义完全
  对不上，硬解只会画出"看着合理但全是错的"图。
* `mask == 0` 的玩家：玩家表里变暗、武器和 hp 显示 `—`（文件里那几格是残留值，
  不是"这个死人还有 100 血"），遮挡矩阵里整行整列变暗。
* 维度不对（少于 20 列）、`map_id` 和文件名不一致、`global_feat` 不是 `(T,5)`（单帧
  快照 / 只有 4 列没有回合时间）、`T ≠ 257`（旧样本的随机跨度，或窗口被截到回合开头）、
  第一帧里找不到唯一的 A / B —— 都会进顶部警告条。
* `since_shot = 1.0` 显示成 `≥100 s（饱和）`，不是"100 秒"。
* `blinded` 和 `defuser` 都是玩家**私有**状态：对手看不到你被闪、也看不到你有没有
  钳子。做单视角模型时别把**别人**那两维当可观测输入（和 `global_feat` 里未安放时的
  炸弹位置是同一类问题）。

---

## 实现结构

```
aim/
  __main__.py    python -m ... 的入口
  server.py      标准库 ThreadingHTTPServer：/api/samples, /api/sample, /api/radar,
                 /api/weapon-icons + /weapon-icons/<file>（转发 3D 回放器的图标素材）、
                 /api/model + /api/predict（转给 predict.py）
  data.py        npz → JSON 载荷（逆变换、逐帧全场解码、遮挡矩阵、道具、雷达 extent）
  predict.py     可选：best.pt → T 个预测点（torch 懒加载，缺了就抛 ModelUnavailable）
  selftest.py    逆变换往返 + 遮挡语义 + 真实样本 + HTTP 端到端
  static/        index.html / app.js / style.css（原生 JS + canvas，无构建）
```

* **为什么不用 Flask**：目标机器可能只有 numpy；本地单人工具用标准库够了。所有响应
  都是 `Cache-Control: no-store`（样本随时会重新 `prepare`）。
* **为什么常量抄了一份**：`scripts/training_data/__init__.py` 会连带 import
  `demo_extract` → `demoparser2`。`data.py` 里的 `MAP_CENTER` / `WEAPON_NAMES` /
  `FEATURES` 是 `constants.py` 和 `tasks/aim.py` 的镜像；改那边记得同步这边。
* **载荷大小**：一条 `T=257` 的样本约 0.76 MB JSON（`players.phys` 是 channel-major 的
  `(T,10)` 矩阵），gzip 之后约 100 KB。缓存：样本载荷 6 条 LRU，雷达 PNG 按地图常驻。

## 自检

```bash
python -m scripts.training_data_visualizer.aim.selftest
```

四件事：

1. 用 `_SCALE` / `_BIAS` 正向编码一组已知物理量，再 `decode_player()` 回来，逐个字段
   对比（含 18 / 19 列旧样本、yaw 跨 ±180；`decode_players()` 的逐帧版必须逐格相同）；
2. **遮挡语义**：40 条真实样本里"死前最后一帧 A 看得见 B"的比例必须 ≥ 70%
   （实测 85%；烟雾改成凸包体积之前是 92%），取值只有 0/1/2、死人行列全 0、
   方向不一致 < 25%（实测 1.7%）；
3. 真实样本：`T == 257`（固定 4 s × 64 Hz + 1）/ 10 人 / 20 列 / `global_feat` 是
   `(T,5)` 且第 5 列和 `seconds` 严格同步（逐帧回合时间）/ A、B 下标和 `mask` 对得上 /
   单人视图和 `players` 那一行逐格相同 / 0-1 范围 / 死人的武器是 -1 / JSON 可序列化；
   旧版样本必须被拒绝；
4. 在随机端口起一个真 server，用 `urllib` 拉 `/api/samples`、`/api/sample`、
   `/api/radar`、`/api/weapon-icons`（再取一张真图标 PNG）、页面和静态资源，
   确认目录穿越、非法图标名、**已下线的 `/api/view`** 都被拒。

## 样本要重跑

这一版换了布局（删 `atk`/`tgt`/视角图，加逐帧全场 + 遮挡矩阵），旧样本必须重造：

```bash
python -m scripts.training_data.prepare --task aim --overwrite
```

`--overwrite` 会先清空 `train/valid/test`。造数成本主要在遮挡：一帧 `n_alive²×8` 条射线，
实测 **0.5 ms/帧**（10 人全活，mirage），一条 `T=257` 的样本纯遮挡约 0.13 s；四份 csz 共
449 条样本 **159 秒**（≈0.35 s/条，含解码），落盘 **66 KB/样本 → 449 条 28.7 MB**
（旧版：随机窗口 132 秒、320 KB/样本）。

补了"回合结束那一枪"之后，同一批 csz 是 **497 条**（anubis +13 / inferno +13 /
mirage +8 / cache +14，每条多一次 `T=257` 的遮挡）。旧样本本身没变，但要让数据集
包含它们，同样得 `--overwrite` 重跑一次（实测 4 份 ≈ 3.5 分钟）。
