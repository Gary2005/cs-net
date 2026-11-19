# CS2 游戏文件资产提取 — 可行性调研

**目的**：本机现在有真实的 CS2 安装（Windows），想知道**能从游戏文件里挖出什么真实资产**，
让 `visualizer/` 的 3D 回放更真实 —— 重点是你点名的四类：**烟雾、火焰、人物动画、枪械模型 / 枪械图标**。

**数据提取（`.dem` → tick 数据）不在本次范围内**，那部分 `tools/demo_extract` 已经做完了。

**调研日期**：2026-09-16 起 ｜ **游戏文件**：`F:\Program Files\steam\steamapps\common\Counter-Strike Global Offensive`
**工具**：Source2Viewer-CLI **20.0.6980**（Windows x64，本次新下载，见 §2）

**证据分级**（沿用本仓库其它 research 笔记的约定）：

| 标记 | 含义 |
|---|---|
| **[M]** | **本次实测** —— 在本机真跑出来的数字/结果，可复现 |
| **[V]** | **Valve 原始数据** —— 游戏 VPK 里的实际文件清单、decompile 出来的资源文本 |
| **[T]** | **工具文档/源码** —— VRF 自己的说明 |
| **[I]** | **我的推断** —— 明确标注，未验证 |

---

## 0. 结论先行

**能挖，而且比预期好得多。四类资产全部拿到了真东西。** 但有两个必须先知道的坑，
它们决定了工程方案，所以放在最前面。

### 两个关键发现

**① 人物模型默认导出 = 1062 MB，必须裁剪。**
CS2 agent 的 `--gltf_export_animations` 会把模型动画集里的**全部 2091 个 clip** 都塞进 GLB。
实测 `ctm_sas`：**2091 clips / 1062 MB**。回放只需要几十个。
用本次写的裁剪器压到 **92 clips / 22.5 MB（2.1%）**，
且**网格与蒙皮结果与裁剪前逐字符一致**（§5.4 校验 + §5.4.1 渲染验证）。

**② 纹理是全分辨率 PNG，且 VRF 没有缩放开关。**
agent 纹理 2048²、武器纹理高达 **4096²**，一张就 15–18 MB。
实测：一个 agent 的纹理 **49 MB**，6 把武器的纹理 **143 MB**。
必须后处理降分辨率 —— 1024² 后 agent 纹理降到 **16.3 MB**，武器降到 **33.2 MB**（§5.5）。

### 四类资产一句话结论

| 你要的 | 能拿到什么 | 质量 | 本机实测体积 |
|---|---|---|---|
| **烟雾** | 真实粒子定义 `.vpcf` + **真实 sprite 序列帧**（11 个序列、864 帧带 alpha，VRF 自动拆帧） | **高** | 定义 86 KB + 帧 23.7 MB |
| **火焰** | `C_Inferno` 的三层 vsnap 点云（core/filler/edge）+ 157 个 inferno 粒子 + 真实火焰贴图（495 帧） | **高** | vsnap 44 KB + 贴图 34 MB |
| **人物动画** | 完整 94 关节骨架（`root_motion`/`pelvis` 齐全）+ 92 个位移 clip（idle/walk/run/jump/crouch × 8 方向） | **中高**（见 §5.3） | 22.5 MB / agent |
| **枪械模型** | 136 个真实 `.vmdl_c`（世界模型 + 弹匣） | **高** | 0.6–5 MB / 把 |
| **枪械图标** | **68 张官方 HUD 图标**，统一 512×384 RGBA | **高** | **5.6 MB 全部** |

### 落地状态

| 项 | 状态 |
|---|---|
| 1 枪械图标 | ✅ **已实现**（§9.1），并已用于**击杀信息**（§9.1.1） |
| 2 烟雾真实 sprite | ✅ **已实现**（§9.2） |
| 3 火焰三层 sprite | ✅ **已实现**（§9.3），颜色已按实测校准（§9.3.1）、卡顿已修（§9.3.2） |
| 4 武器 3D 模型 | ✅ **已实现**（§9.4），**65 把**武器，1994 MB → **280.9 MB** |
| 4b 飞行中的投掷物模型 | ✅ **已实现**（§9.4.5），**6 个**投掷物模型（烟雾 / 高爆 / 闪光 / 诱饵 / 莫洛托夫 / 燃烧弹） |
| 5 玩家模型 + 位移动画 | ⛔ **本次不做**（按用户要求） |

实现记录、验证证据链、以及过程中抓到的真 bug 都在 §9。
**武器模型在真实浏览器里渲染出画面这一步没有直接截图证据**，原因见 §9.5 / §10.1。
（**投掷物模型**这一步有 —— 无头 Chrome 里真的渲染出 6,464 个非背景像素，见 §9.4.5。）

### 第二轮：用户试用后反馈的三个问题

用户实际跑起来后报了三件事，逐条查证的结果如下（细节见对应小节）：

| 反馈 | 真实原因 | 是不是我原来说的那样 |
|---|---|---|
| 手持 Five-SeveN / 刀 看不到模型 | **只提取了 6 把武器**（demo 用了 30 个显示名）；另外旧代码的"模糊匹配"连 `Knife` 都命中不了 | 不是。我原以为提取的 6 把够用 |
| 火焰是**白色**的，出现瞬间卡顿 | 火焰贴图是**灰度亮度遮罩**，我当彩色贴图贴了 → `白 × 灰度`；卡顿来自逐帧 `needsUpdate` + 首次同步编译 shader | 不是。两处都是我的实现问题 |
| 击杀信息想显示枪械图标 | 图标本来就有，只是没接到击杀信息上 | 是，纯功能缺失 |

其中最有价值的一条是**火焰贴图是灰度图**：实测三张序列的平均通道极差只有
**1.5–1.6（满值 255）**，即**纯灰度**；只有 `fire_onsurface` 是彩色的（极差 138.8）。
颜色来自 vpcf 里的粒子颜色渐变，贴图只管亮度。见 §4.2.1。

---

## 1. 环境（本次新建）

按你的要求用 conda 重建了 Python 环境（原来是 mac，环境不存在了）。

```powershell
conda create -n cs-net-v5 python=3.12 -y
conda activate cs-net-v5
pip install demoparser2 numpy zstandard rarfile flask pandas
```

| 项 | 值 |
|---|---|
| conda 环境 | **`cs-net-v5`** @ `D:\conda\envs\cs-net-v5`（标准位置） |
| Python | 3.12.14 |
| demoparser2 | 0.42.0（和 mac 时期同版本，见 `docs/PLAN.md`） |
| 新增 | flask 3.1.3、pandas 3.0.6、numpy 2.5.3、zstandard 0.25.0、rarfile 4.5 |

`conda env list` 与 `conda activate cs-net-v5` 均已验证正常，**按平时的方式用就行**。

> **一处归因更正。** 本次会话中途 `conda create -n` 曾报
> `NoWritableEnvsDirError: No writeable envs directories configured`，
> 我当时记成了「conda 无管理员权限导致的配置问题」，**这是错的**。
> 真实原因是我这个执行环境的**文件沙箱当时处于 `workspace-write`，
> 禁止写入工作区之外的任何路径**（`C:\Users\...\.conda`、`D:\conda\envs` 全被拒），
> conda 因此找不到可写的 envs 目录。放开沙箱后
> `conda create -n cs-net-v5` **一次成功**，落在标准位置，
> **不需要任何特殊处理**。
>
> 教训：**沙箱拒绝写文件，看起来很像程序自己的权限/配置问题**，
> 排查时别急着归因到工具链。这条与 `NOTES.md` §2(d) 记的那几个
> 「静默失败」坑是同一类错误。

`scripts/check_env.py` 里还检查 Go side-channel（`tools/demo_extract/sidechannel_go`）——
**本机没有 Go 工具链**（`go` 不在 PATH），所以那个二进制需要另外处理，本次未涉及。

---

## 2. 工具链：Source2Viewer-CLI（VRF）

mac 时期那份 `Source2Viewer-CLI 19.2.6339` 不在了，本次下了最新的 Windows 版：

| 项 | 值 |
|---|---|
| 版本 | **20.0.6980+a06886f7**（GitHub release `20.0`，2026-08-17） |
| 位置 | `tools/Source2Viewer-CLI/Source2Viewer-CLI.exe`（108 MB，随带 libSkiaSharp / spirv-cross） |
| 来源 | `https://github.com/ValveResourceFormat/ValveResourceFormat/releases/download/20.0/cli-windows-x64.zip` |
| License | VRF 本体 MIT；**不含任何 Valve 资产** |

**好消息：`--game` 这个 flag 现在有了。** 之前 mac 上的 19.x 明确拒绝 `--game`
（`notes/research-fidelity-ai.md` §2.1 记过这个坑），20.0 的 `--help` 里已列出
`--game <path to gameinfo.gi>`。**实践上仍然不需要它** —— 只要输入指向真实
`pak01_dir.vpk`，CLI 会自己从 `gameinfo.gi` 加载 search path（实测日志确认）：

```
Found "Counter-Strike 2" from "...\game\csgo\gameinfo.gi"
Preloading vpk "...\game\csgo\pak01_dir.vpk"
Added folder "...\game\csgo" to game search paths
Preloading vpk "...\game\csgo_imported\pak01_dir.vpk"
Preloading vpk "...\game\csgo_core\pak01_dir.vpk"
Preloading vpk "...\game\core\pak01_dir.vpk"
```

### 2.1 一个已知的非致命报错（每次导出都会刷屏）

```
Failed to get texture inputs for "g_tMetalness": Only VCS file versions 59 through 71
are supported for variable 'vcsFileVersion': 72 (0x48)
```

**[M]** 每次 agent/武器导出都会出现，抛 `UnexpectedMagicException` 的堆栈。
含义：**当前 CS2 的编译着色器版本 72 超出了 VRF 20.0 支持的 59–71 范围**，
于是 VRF 无法通过读 shader 来推断「哪个通道是 metallic」。

**影响范围（实测确认）**：只影响**金属度/粗糙度通道的语义推断**，
**不影响** mesh、skeleton、animation、颜色贴图、法线贴图、AO 贴图的导出 ——
这些全部正常产出（§5 有产物清单）。所以是「材质参数略糙」而不是「导不出来」。

**[I] 应对**：VRF 在 glTF material 的 `extras` 里保留了原始 shader 名和参数
（`notes/research-fidelity-ai.md` §2.3 已记录这个机制），可以在 Three.js 侧按 shader 名补。
或者等 VRF 更新支持 vcs 72。

### 2.2 参考命令骨架

```powershell
$BIN  = "D:\project\cs-net-v5\tools\Source2Viewer-CLI\Source2Viewer-CLI.exe"
$VPK  = "F:\Program Files\steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\pak01_dir.vpk"
```

`pak01_dir.vpk` 本体只有 **7,696,110 字节**，**全量列表 137,383 条**，
列出耗时 **2.4 秒** —— 索引很便宜，可以随便枚举。

---

## 3. VPK 全量清点

`--vpk_list` 全量 dump 后按类别统计（原始数据在 `notes/scratch/asset_inventory/`，
`SUMMARY.md` + 24 个 `list_*.txt`，脚本 `inventory.py` 可重跑）：

### 3.1 按扩展名（文件数 Top）

| 扩展名 | 数量 | 说明 |
|---|---:|---|
| `.vtex_c` | 71,707 | 纹理 |
| `.vmat_c` | 22,003 | 材质 |
| `.vsnd_c` | 20,384 | 音频 |
| `.vmdl_c` | 13,105 | **模型** |
| `.vnmclip_c` | 2,735 | **Animgraph2 动画 clip** |
| `.vpcf_c` | 2,622 | **粒子系统** |
| `.vcompmat_c` | 1,530 | 合成材质 |
| `.vsvg_c` | 1,181 | 矢量图（UI） |
| `.vsnap_c` | 98 | **粒子快照**（火焰用这个） |
| `.vnmskel_c` | 70 | **骨架** |
| `.vnmgraph_c` | 232 | **动画状态机图** |

### 3.2 按类别（关注体积）

| 类别 | 文件数 | 总体积 |
|---|---:|---:|
| `agents/models/` 玩家模型 `.vmdl_c` | **91** | **59.3 MB** |
| `weapons/models/` 武器模型 `.vmdl_c` | **136** | **42.6 MB** |
| `particles/` 全部 | 2,722 | **10.3 MB** |
| └ `.vpcf_c` | 2,621 | 8.2 MB |
| └ `.vsnap_c` | 98 | 1.0 MB |
| └ `particles/inferno_fx/`（火焰） | 157 | 0.6 MB |
| └ `particles/explosions_fx/`（含烟雾） | 324 | 1.1 MB |
| `materials/particle/` `.vtex_c` | 357 | **206.2 MB** |
| `panorama/images/`（UI 图标） | 35,008 | 6,487 MB |
| └ `econ/weapons/base_weapons/` | **68** | **5.6 MB** |

**值得注意的体积分布**：真正想要的东西**非常小** —— 全部粒子定义 10 MB、
91 个玩家模型 59 MB、136 个武器模型 43 MB、68 张图标 5.6 MB。
`panorama/images/` 那 6.5 GB 绝大部分是**皮肤贴图**（`econ/default_generated/` 6,804 个）
和**贴纸**（`econ/stickers/` 22,872 个），回放用不上。

### 3.3 玩家模型清单（91 个，12 个阵营目录）

| 阵营 | 文件数 | 阵营 | 文件数 |
|---|---:|---|---:|
| `tm_phoenix` | 18 | `tm_leet` | 12 |
| `tm_professional` | 16 | `tm_balkan` | 11 |
| `ctm_fbi` | 15 | `ctm_swat` | 10 |
| `tm_jungle_raider` | 14 | `ctm_gendarmerie` | 9 |
| `ctm_st6` | 13 | `ctm_diver` | 5 |
| `shared`（手套/手臂） | 12 | `ctm_sas` | 3 |

外加 `agents/models/shared/arms/glove_*` 共 **12 副手套/手臂**（`glove_karambit` 系没有，
是 `glove_bloodhound` / `glove_brokenfang` / `glove_hardknuckle` / `glove_specialist` … 这批）。

### 3.4 武器模型清单（136 个）

41 个武器子目录，每把通常 **2 个 `.vmdl_c`**：`weapon_XXX.vmdl_c`（本体）
+ `weapon_XXX_mag.vmdl_c`（弹匣，换弹动画用）。

覆盖完整：`ak47` `aug` `awp` `bizon` `cz75a` `deagle` `elite` `famas` `fiveseven`
`g3sg1` `galilar` `glock18` `m249` `m4a1_silencer` `m4a4` `mac10` `mag7` `mp5sd`
`mp7` `mp9` `negev` `nova` `p250` `p90` `revolver` `sawedoff` `scar20` `sg556`
`ssg08` `taser` `tec9` `ump45` `usp_silencer` `xm1014`，
5 种手雷（`hegrenade` `flashbang` `smokegrenade` `molotov` `incendiary` `decoy`）、
`c4`、`defuser`、以及 **24 把刀**（`knife_karambit` `knife_bayonet` `knife_butterfly`
`knife_m9` `knife_talon` … 全都有）。

---

## 4. 烟雾与火焰

### 4.1 烟雾：拿到了真实 sprite 序列帧

之前 `notes/research-fidelity.md` §2.5 只能从 `pak01_dir.txt` 里**猜**贴图路径，
现在**直接 decompile 出了真实引用**：

```powershell
& $BIN -i $VPK -f "particles/explosions_fx/explosion_smokegrenade_voxel.vpcf_c" -o .\out\ -d
```

**实测输出**：`explosion_smokegrenade_voxel.vpcf`，**86.3 KB 文本**。里面的贴图引用：

```c
m_hTexture = resource:"materials/particle/smoke/smokeburst/smokeloop_i_0_sc_hardedge.vtex"
m_nTextureChannels = "SPRITECARD_TEXTURE_CHANNEL_MIX_A"
m_hTexture = resource:"materials/particle/smoke/smokeburst/smokeloop_i_0_flwmix.vtex"
m_nTextureType = "SPRITECARD_TEXTURE_ANIMMOTIONVEC"
```

并且看到 `C_INIT_InitFromCPSnapshot` —— **证实了「烟雾从点云快照初始化」这个机制**
（和 `research-fidelity-ai.md` §1.1 的推断一致）。

#### 关键发现：VRF 会自动把 sprite sheet 拆成逐帧 PNG

```powershell
& $BIN -i $VPK -f "materials/particle/smoke/smokeburst/" -o .\out\ -d --threads 4
```

**实测**：拆出**逐帧带 alpha 的 PNG** + 一个 `.mks` 序列描述文件：

| 产物 | 内容 |
|---|---|
| `smokeburst_c.mks` | 2.2 KB，文本：`sequence 0` / `frame smokeburst_c_seq0_N.png 1` |
| `smokeburst_c_seq0_0..N.png` | **RGBA 逐帧**，31–51 KB/帧 |
| `smokeburst_c.vtex` | 1.5 KB，纹理源描述 |

**帧尺寸随帧号递增**（178×136 → 更大），这是**烟雾扩散**的形态 —— 说明这是真实的
烟雾生长序列，不是等大方图。**[I]** 原始 sheet 应该是把这些帧打包在一起的图集，
VRF 按 sequence 拆开了。

⚠️ **一处修正**：我最初按 `explosion_smokegrenade_voxel.vpcf` 引用的那两张贴图
（`smokeloop_i_0_sc_hardedge` / `smokeloop_i_0_flwmix`）以为只有 32 帧。
**把整个 `smokeburst/` 目录拉下来后发现序列多得多** —— 该目录共
**864 张 PNG / 23.7 MB**，含 **11 个序列**，其中：

| 序列 | 帧数 |
|---|---:|
| `smokeburst_c` / `_c_slate` / `_c_slate_flw` / `_e` / `_e_flw` | 各 64 |
| `smokeloop_i_0` / `_flw` / `_flwmix` / `_sc` | 各 64 |
| **`smokeloop_i_0_sc_hardedge`** | **128** |
| `smokeloop_g_1` | 32 |

**`smokeloop_i_0_sc_hardedge` 正是 voxel 烟雾引用的那张，128 帧。**

**对可视化的意义**：这比 `research-fidelity.md` §2.5 里
「最便宜可信：一个 raymarch 椭球」的方案高一个档次 —— 可以直接用
**真实烟雾 sprite + flipbook 动画**做体积感的粒子层。

### 4.2 火焰：三层点云快照 + 真实贴图

`C_Inferno` 的五个 snapshot handle 在 VPK 里**一一对应存在**（之前只是从 schema 推断）：

| 产物 | 大小 |
|---|---|
| `particles/inferno_fx/fire_core.vsnap` | **0.8 KB** |
| `particles/inferno_fx/fire_filler.vsnap` | **28.7 KB** |
| `particles/inferno_fx/fire_edge.vsnap` | **14.5 KB** |

这证实了 `research-fidelity.md` §3.4 的「火焰 = core + filler + edge 三层」结论。
`particles/inferno_fx/` 下共 **157 个文件**，含 `molotov_groundfire*`、
`incendiary_groundfire*`、`extinguish_fire*` 全家族。

**火焰贴图**（真实存在，非猜测）：

| 目录 | 文件数 |
|---|---:|
| `materials/particle/smoke/` | 24 |
| `materials/particle/flames/` | 6 |
| `materials/particle/fire_onsurface/` | 3 |
| `materials/particle/fire_small_sim/` | 11 |
| `materials/particle/fire_gas/` | 12 |

### 4.2.1 ⚠️ 火焰贴图是**灰度亮度遮罩**，不是彩色贴图

这条是第二轮才发现的，**它直接决定了火焰渲染必须上色**，所以单独列出来。

取每张序列的中间帧、只统计 alpha>10 的像素，量平均 RGB 和**每像素的通道极差**
（`max-min`，满值 255）：

| 序列 | 平均 RGB | 平均通道极差 | 性质 |
|---|---|---|---|
| `flames` | (65, 65, 65) | **1.5** | 纯灰度，很暗 |
| `flame_omni` | (105, 105, 105) | **1.6** | 纯灰度 |
| `flame_wisp` | (234, 234, 234) | **1.5** | 纯灰度，接近白 |
| `fire_onsurface` | (155, 67, 16) | **138.8** | ← **只有这张是彩色的** |

极差 1.5/255 ≈ 0.6%，也就是三张序列**完全没有颜色信息**。真实颜色来自
vpcf 里的粒子颜色渐变，贴图只负责"哪里亮、有多亮"。

**[V]** 后果：如果按颜色图直接贴上去，渲染结果就是 `白 × 灰度`。
`tools/cs2_assets/fire_tint_preview.py` 按 three 的实际 shader 顺序合成出来量过，
无 tint 时可见像素的 HSV 饱和度**100% 落在 0.20 以下**、三通道**完全相等**
（0.601, 0.601, 0.601）—— 就是用户看到的"火是白的"。校准后的数值见 §9.3.1。

**[I]** 顺带一提：`fire_onsurface` 是彩色的，说明 Valve 对这两类粒子的做法不同 ——
`fire_onsurface` 自己带颜色，`flames`/`flame_omni`/`flame_wisp` 靠外部渐变。
所以**不能**用 `fire_onsurface` 的颜色去推另外三张。

### 4.3 烟雾的特殊之处：烟雾手雷自己的 vsnap 也拿到了

```
particles/explosions_fx/smokegrenade/smokegrendae.vsnap_c          (Valve 的拼写错误，原样)
particles/explosions_fx/smokegrenade/weapon_smokegrenade_adaptive.vsnap_c
particles/explosions_fx/smokegrenade/weapon_smokegrenade_pointcloud.vsnap_c
particles/explosions_fx/smokegrenade/weapon_smokegrenade_pointcloud_col.vsnap_c
```

**[I]** 结合 `research-fidelity.md` §2.5 的 `m_VoxelFrameData` 结论：
CS2 的烟雾是**体素场 + 点云快照**两条路都在用。既然 `demo_extract` 已经能解出
真实 32³ 体素（`visualizer/README.md` §1 说浏览器已经在解码 Morton journal），
那么**体素负责几何/遮挡，sprite 序列帧负责外观**是最合理的组合。

### 4.4 粒子整包拉下来的实测体积（`--particles`）

把烟雾 + 火焰的粒子定义和贴图一次性拉全（`extract_cs2_assets.py --particles`）：

| 类型 | 文件数 | 体积 |
|---|---:|---:|
| **粒子定义** `.vpcf` | 6 | 0.1 MB |
| **粒子快照** `.vsnap` | 3 | 0.04 MB |
| **sprite 序列帧** `.png` | **1,359** | **57.7 MB** |
| `.mks` 序列描述 | 21 | 0.1 MB |
| `.vtex` 纹理源 | 22 | ~0 |

其中 sprite 帧的分布：

| 目录 | 帧数 | 体积 |
|---|---:|---:|
| `materials/particle/smoke/smokeburst/` | 864 | 23.7 MB |
| `materials/particle/fire_onsurface/` | 330 | 31.2 MB |
| `materials/particle/flames/` | 165 | 2.8 MB |

**57.7 MB 的序列帧不会全用上。** 实际只需要 voxel 烟雾引用的那 1–2 个序列
（`smokeloop_i_0_sc_hardedge` 128 帧 + `smokeloop_i_0_flwmix` 64 帧，
合计约 7 MB），火焰同理。**[I] 建议按需只取 `.vpcf` 里实际 `resource:`
引用的那几个序列**，其余不拉 —— 做法是先 `-d` decompile 单个 `.vpcf`
看它引用哪些 `.vtex`，再只拉那几个文件。

拉全的实测命令（约 30 秒）：

```powershell
& $PY tools\cs2_assets\extract_cs2_assets.py --particles --out assets
```


---

## 5. 人物模型与动画（最重要的一节）

### 5.1 导出成功，但体积是个大问题

```powershell
& $BIN -i $VPK -f "agents/models/ctm_sas/ctm_sas.vmdl_c" `
    -o .\ctm_sas.glb -d --gltf_export_format glb `
    --gltf_export_animations --gltf_export_materials --gltf_textures_adapt
```

**实测结果**（耗时 45.5 秒）：

| 项 | 值 |
|---|---:|
| **GLB 体积** | **1,062.22 MB** |
| **动画数** | **2,091** |
| 网格数 / 蒙皮数 | 70 / 70 |
| 节点数 | 735 |
| 材质 / 贴图 | 9 / 24 |
| accessors | 1,254,947 |
| **顶点数** | **32,382** |
| **三角面数** | **34,717** |
| 附带 PNG | 37 张 / 49 MB |

**1 GB 的 GLB 显然不能进浏览器**（而且 glTF 规范 GLB 上限 2 GB）。

### 5.2 动画：哪些是真有动作的

我一开始误判了 —— 抽样看到 `idle_knife` 只有 1 个关键帧、时长 0.00s，
以为动画导出坏了。**仔细核对后：那是源数据本身如此，不是 bug。**

统计 2091 个 clip 的关键帧数：

| 关键帧数 | clip 数 |
|---|---:|
| 1（**静态姿势，本来就没动作**） | 292 |
| 2–14 | 少数 |
| 13 | 111 |
| **>1（有动作）** | **1,799 / 2,091** |

具体实例（**[M]** 逐个读 accessor 确认）：

| clip | 关键帧 | 时长 |
|---|---:|---:|
| `world/knife/_default_knife/idle_knife` | **1** | 0.000s（**静态 idle 姿势**） |
| `world/knife/_default_knife/run_n_knife` | 23 | **0.733s** |
| `world/pistol/_default_pistol/walk_n_pistol` | 31 | **1.000s** |
| `viewmodel/pistol/pistol_cz75a/shoot_empty_cz` | 13 | 0.400s |
| `ui_anims/main_menu/ct/ct_main_menu_revolver_idle` | **793** | 最长 |

**1,052 个 clip 在 `pelvis.translation` 上有多于 1 个关键帧** —— 即确实有位移。

**[I] 关于时长**：`run` 0.733 s / `walk` 1.000 s 都偏短，但这符合
「一个循环周期」的语义（23 帧 @ ~31 fps ≈ 0.73 s）。CS2 的 Animgraph2 用
**循环播放 + 状态机混合**，不是烘焙整段长动画。这意味着**回放侧必须自己实现
「按速度选 clip + 按速度调播放速率 + 混合」** —— 和
`research-fidelity-ai.md` §2.5 的结论一致（VRF 从不执行 animgraph）。

### 5.3 骨架：完整的 94 关节，和 canonical skeleton 对得上

最大蒙皮 **94 joints**，开头是：

```
root_motion, pelvis, spine_0..3, neck_0, head_0,
clavicle_L/R, arm_upper_L/R, arm_lower_L/R, hand_L/R,
jiggle_primary, leg_upper_L/R, leg_lower_L/R, ankle_L/R,
wpnPivot, wpn, ...
```

**`root_motion` 是根、`pelvis` 是它的子节点、`wpnPivot`/`wpn` 都在** ——
和 `NOTES.md` §1 里 AG2 canonical skeleton（74 bones rooted at `root_motion`）完全一致。
**所有 2091 个 clip 的 channel 都指向 `root_motion` 节点**。

蒙皮完整性实测：

| 检查 | 结果 |
|---|---|
| 带蒙皮的 primitive | 74 |
| 蒙皮顶点 | 32,382 |
| 最大 joint index | 89 |
| **权重和 ≠ 1.0 的顶点** | **0** |
| **joint index 越界的顶点** | **0** |
| inverse-bind matrices | 975，**奇异 0 个** |

### 5.4 解决方案：裁剪器（本次新写）

`tools/cs2_assets/prune_glb.py` —— 按正则选 clip，**真正重建 buffer**
（不是留孤儿数据），mesh/skin/material 原样透传，全部引用重新映射。

**为什么必须自己写**：VRF 的 `--gltf_animation_list` **只认精确名字，不支持通配符**。
实测传 `"animation/anims/world/rifle/_default_rifle/*"` 得到：

```
--- glTF animation filter matched no animations for: animation/anims/world/rifle/_default_rifle/*
```

**所以路径只能是「全量导出一次 → 再裁剪」。**

**实测裁剪效果**（`ctm_sas`）：

| 配置 | clip 数 | GLB 体积 | 比例 |
|---|---:|---:|---:|
| 原始 | 2,091 | 1,062.22 MB | 100% |
| 全部 world/rifle + pistol + _default（去掉 ui_anims） | 531 | 157.27 MB | 14.8% |
| **locomotion only**（idle/walk/run/jump/crouch × 8 向） | **92** | **22.45 MB** | **2.11%** |

**裁剪前后蒙皮数据逐位一致**（这是关键的正确性证据）：

```
ctm_sas.glb        : 74 prims, 32,382 verts, max joint 89, 0 bad weights, 975 IBM (0 singular)
ctm_sas_loco.glb   : 74 prims, 32,382 verts, max joint 89, 0 bad weights, 975 IBM (0 singular)
```

另外写了 `validate_glb.py`（所有引用是否解析得开）和 `check_skinning.py`（蒙皮是否可用）。
**踩过的坑**：裁剪器第一版判断 accessor 引用时用了「key 里含 `accessor`」，
结果 glTF 的 accessor 引用是**以 `POSITION`/`NORMAL`/`indices` 这些 key 的「值」**出现的，
一个都没匹配到，产出的 GLB 有 308 个悬空引用。已改成按规范枚举 key 名。
（这个坑值得记：**任何 GLB 重写之后都要跑一遍引用校验**。）

### 5.4.1 已确认「真的能画出来、真的会动」（纯 CPU 光栅化验证）

结构校验只能证明引用不悬空，**不能证明模型画得出来**。本机 CDN 不通、
下不到 `GLTFLoader`（见 §10.1），所以另外写了一个
**纯 Python 的软件光栅化器** `tools/cs2_assets/render_glb_ascii.py`：
自己解 glTF、按时间采样动画通道（四元数走最短弧插值）、
算节点世界矩阵、做线性混合蒙皮（LBS）、再投影成 ASCII 深度图。

**关键证据 —— 同一个 clip 在两个时间点，轮廓明显不同**（`run_n_pistol`）：

| 时间 | bbox（X × Y × Z） |
|---|---|
| `t = 0.000s` | 5.4 × **4.3** × 5.1 |
| `t = 0.360s` | 5.3 × **4.5** × 5.2 |

bbox 变了、四肢的深度着色位置也变了 —— **说明动画求值和蒙皮都是通的**。
渲染出来是一个能辨认的人形（头/躯干/腿/手臂），不是碎片或 T-pose 塌陷。

**更强的证据：裁剪前 vs 裁剪后渲染结果逐字符相同。**

| 输入 | 顶点数 | `t=0.360s` 的 bbox | 输出 |
|---|---:|---|---|
| `ctm_sas.glb`（原始 1062 MB，2091 clips） | 32,382 | 5.3 × 4.5 × 5.2 | 与右侧一致 |
| `ctm_sas_loco.glb`（裁剪后 22.5 MB，92 clips） | 32,382 | 5.3 × 4.5 × 5.2 | 与左侧一致 |

**即：裁剪器在删掉 96% 动画数据的同时，mesh 和蒙皮结果完全没变。**

另外 `ctm_fbi.glb`（流水线产物，24,748 顶点）在 `t=0.00` 与 `t=0.36` 同样呈现
可见的姿态差异，跨模型也成立。

```powershell
# 用法
& $PY tools\cs2_assets\render_glb_ascii.py assets\agents\ctm_sas.glb `
      --clip run_n_pistol --time 0.36
```

**仍然没有验证的**：Three.js 里真实渲染（需要浏览器环境）。但「几何 + 蒙皮 + 动画
在 CPU 上算出来是对的」这一步已经坐实，剩下的只是加载器/渲染管线问题。


### 5.5 纹理：第二个体积瓶颈

**VRF 没有任何缩放/压缩开关**（`--help` 里只有 `--texture_decode_flags`，
取值 `none|auto|ForceLDR`，那是色彩空间不是分辨率）。所以只能后处理。

实测尺寸与体积：

| 资产 | 原分辨率 | 单张体积 |
|---|---|---|
| agent 法线贴图 `glove_hardknuckle_normal` | 2048² | **8.48 MB** |
| agent 法线 `ctm_sas_body_normal` | 2048² | 5.50 MB |
| **武器** `ak47_default_color_psd` | **4096²** | **15.67 MB** |
| **武器** `ak47_default_normal_png` | **4096²** | **15.66 MB** |
| **武器** `ak47_default_ao_..._orm` | **4096²** | **17.85 MB** |

**降分辨率实测**（`tools/cs2_assets/resize_textures.ps1`，System.Drawing 双三次重采样）：

| 资产 | 原始 | @1024 | @512 |
|---|---:|---:|---:|
| **agent 纹理（37 张）** | 49.0 MB | **16.3 MB (33.3%)** | **5.3 MB (10.7%)** |
| **武器纹理（47 张，6 把枪）** | 143.3 MB | **33.2 MB (23.2%)** | — |

### 5.5.1 再上一档：按贴图类型选编码器（Pillow 已装，实测）

降分辨率只解决了分辨率，PNG 本身仍是无损编码。装了 Pillow 12.3.0（WebP 可用）
之后实测了「缩放 + 重编码」的组合，**又省了 2–4 倍**：

| 资产 | 原始 | @1024 PNG | **@1024 按类型选编码** |
|---|---:|---:|---:|
| **agent 纹理（37 张）** | 49.0 MB | 14.0 MB | **7.1 MB (14.5%)** |
| **武器纹理（47 张）** | 143.3 MB | 23.9 MB | **12.1 MB (8.5%)** |

**但编码器不能一刀切。** 我在真实法线贴图上量了**角度误差**
（`tools/cs2_assets/normal_map_error.py`：把 RGB 解回单位法线向量，
逐 texel 算与原图的夹角）：

`glove_hardknuckle_normal`，512²，262,144 texel：

| 编码 | 体积 | 平均误差 | p99 | 最大 | >2° | **>5°** |
|---|---:|---:|---:|---:|---:|---:|
| png（基准） | 494,971 | 0.000° | 0.00° | 0.00° | 0.0% | 0.0% |
| webp-lossless | 414,682 | 0.000° | 0.00° | 0.00° | 0.0% | 0.0% |
| webp-q99 | 138,992 | 3.603° | 19.25° | 71.55° | 48.6% | **22.5%** |
| webp-q90 | 75,034 | 3.975° | 19.54° | 80.59° | 58.4% | **24.6%** |
| **jpeg-q95** | 205,344 | **1.664°** | 4.90° | 9.38° | 31.0% | **0.9%** |

**两个反直觉的结论**：

1. **有损 WebP 对法线贴图不可用** —— 即使 **q99 也有 22.5% 的 texel 偏差 >5°**，
   会在着色上表现为可见噪点。**[M]** 我还验证了这不是色度子采样造成的：
   `subsampling="4:4:4"` 与默认档**字节数与误差完全相同**（WebP 对 RGB 本就无子采样），
   所以是 WebP 的率失真行为本身不适合法线数据。
2. **JPEG q95 反而比 WebP 好得多**（>5° 只占 0.9%，比 WebP q99 低 25 倍）。
   虽然 JPEG 有子采样，但 `subsampling=0` 关掉后它对这种高频 RGB 数据更保真。

**最终策略**（已固化进 `tools/cs2_assets/encode_textures.py`）：

| 贴图类型 | 编码 | agent 实测 | 武器实测 |
|---|---|---:|---:|
| 颜色 / albedo（8 + 13 张） | **WebP q90** | 11.5 → **0.3 MB (2.6%)** | 49.3 → **0.8 MB (1.7%)** |
| 法线（10 + 13 张） | **WebP 无损** | 30.3 → **5.5 MB (18.0%)** | 44.6 → **4.8 MB (10.8%)** |
| ORM / AO / mask（18 + 12 张） | **WebP 无损** | 7.1 → **1.4 MB (19.2%)** | 49.4 → **6.5 MB (13.1%)** |
| **合计** | | 49.0 → **7.1 MB** | 143.3 → **12.1 MB** |

法线和 ORM 用**无损**是刻意的：这两类是**数据**不是图像，有损压缩的失真会直接
变成着色/材质错误，而不是「画质略降」。

**[M] 已校验**：84 个输出 `.webp` 全部可被 Pillow 解码，落盘体积与报告一致。

```powershell
& $PY tools\cs2_assets\encode_textures.py notes\scratch\asset_probe\animclip `
      --out assets\agents_web --max-dim 1024
```

**[I] 建议**：agent 用 **1024² + 按类型编码**（7.1 MB/agent），
武器用 1024²（12.1 MB / 6 把）。相比最初的全分辨率 PNG 省了 **7–12 倍**。

> `resize_textures.ps1`（System.Drawing）仍保留：它**零依赖**，
> 适合只想缩放、不想装 Pillow 的场合。要 WebP 就用 `encode_textures.py`。

### 5.6 网格构成：第一人称手臂占了一半

`ctm_sas` 的 70 个网格里，真正有几何的只有 5 个（其余 65 个是
`animation/skeletons/weapons/*.vnmskel.empty_mesh_reference`，**3–13 个三角面的占位体**）：

| 三角面 | 顶点 | 网格名 |
|---:|---:|---|
| 13,504 | 19,077 | `...vmdl_c.thirdperson_body` |
| **12,508** | 7,360 | **`...vmdl_c.firstperson_default_gloves_arms`** |
| 3,606 | 1,954 | `...vmdl_c.firstperson_sleeves` |
| 2,966 | 2,052 | `...vmdl_c.thirdperson_default_gloves` |
| 1,982 | 1,434 | `...vmdl_c.defusekit` |
| **合计** | | **34,717 三角面 / 32,382 顶点** |

**[I] 优化点**：`visualizer/` 是**第三人称回放**，
`firstperson_default_gloves_arms`（12,508 面）+ `firstperson_sleeves`（3,606 面）
**可以整块丢掉** —— 直接省掉 **46% 的三角面**。

材质分布（说明贴图该留哪些）：

| 材质 | 三角面 |
|---|---:|
| `glove_hardknuckle_black` | 14,258 |
| `ctm_sas_body` | 7,166 |
| `sleeve_ctm_sas` | 3,606 |
| `ctm_sas_bodylegs` | 3,414 |
| `ctm_sas_head_gasmask` | 2,784 |
| `w_eq_defuser` | 1,982 |
| `bare_arm_133` | 1,216 |
| `skeleton_material` | 151 |
| `ctm_sas_lenses` | 140 |

### 5.7 另一个路子：`.vnmclip_c` 能单独导出

`animation/anims/` 下有 **2,735 个 `.vnmclip_c`**：

| 目录 | clip 数 |
|---|---:|
| `animation/anims/world/` | 1,374 |
| `animation/anims/viewmodel/` | 674 |
| `animation/anims/ui_anims/` | 563 |
| `animation/anims/chicken/` | 112 |
| `animation/anims/egg/` | 12 |

单独导出实测可行：

```powershell
& $BIN -i $VPK -f "animation/anims/world/rifle/_default_rifle/" -o .\out\ -d --gltf_export_format glb
```

产出 **129 KB / clip**（`crouch_n_rifle.glb`，456 accessors，77 节点，1 动画，校验通过）。

**[I]** 这条路适合「按需取单个 clip」，但产出的是**独立骨架动画（无 mesh）**，
要自己往 agent 的 94 关节骨架上 retarget，比直接裁剪全量 GLB 麻烦。
**推荐还是走 §5.4 的裁剪路线。**

---

## 6. 枪械模型与枪械图标

### 6.1 枪械模型：快、小、干净

```powershell
& $BIN -i $VPK -f "weapons/models/ak47/weapon_rif_ak47.vmdl_c,..." `
    -o .\out\ -d --gltf_export_format glb --gltf_export_materials `
    --gltf_textures_adapt --threads 4
```

**6 把武器实测 5.3 秒**：

| 模型 | GLB |
|---|---:|
| `weapon_rif_ak47` | 3.10 MB |
| `weapon_snip_awp` | 5.02 MB |
| `weapon_c4` | 1.33 MB |
| `weapon_knife_karambit` | 1.00 MB |
| `weapon_smokegrenade` | 0.72 MB |
| `weapon_molotov` | 0.59 MB |

全部**校验通过**（引用全部解析得开）。每个还附带 `<name>_physics.glb`（碰撞体，很小，可丢）。

**注意**：武器 GLB 只有 2 个节点（无骨架），**没有动画** ——
武器的动画在 `animation/anims/viewmodel/` 和 `world/` 里，属于玩家骨架的动画。

#### 6.1.1 第二轮：扩到 **65 把**（这是"有些枪不显示"的真正原因）

第一轮只提取了 6 把，而这份 demo 实际用了 **30 个武器显示名** ——
所以 `Five-SeveN` 之类的武器**根本没有模型可显示**。这不是代码 bug，是覆盖面不够。

现在 `WEAPON_MODELS` 覆盖 **65 把**：全部步枪/狙击/冲锋枪/手枪/霰弹枪/机枪、
**22 把刀**、全部投掷物 + C4 + 医疗针。

| 项 | 数值 |
|---|---|
| 原始导出 | 540 文件 / **1994.4 MB**（410 张 4096² PNG 占 1842 MB） |
| 优化后 | **280.9 MB（14.1%）**，1024² + WebP |
| 最大单把 | `negev` 61.4 → 10.6 MB |
| 最小单把 | 默认刀 0.4 MB |

**模型文件名和图标文件名在好几处对不上，没有规律**，这是第二轮才踩到的坑：

| 显示名 | 图标文件 | 模型文件 |
|---|---|---|
| M4A4 | `weapon_m4a1_png.png` | `weapon_rif_m4a4.vmdl_c` |
| Glock-18 | `weapon_glock_png.png` | `weapon_pist_glock18.vmdl_c` |
| Incendiary Grenade | `weapon_incgrenade_png.png` | `weapon_incendiarygrenade.vmdl_c` |
| M9 Bayonet | `weapon_knife_m9_bayonet_png.png` | `weapon_knife_m9.vmdl_c` |
| Navaja Knife | `weapon_knife_gypsy_jackknife_png.png` | `weapon_knife_navaja.vmdl_c` |
| Talon Knife | `weapon_knife_widowmaker_png.png` | `weapon_knife_talon.vmdl_c` |
| Bowie Knife | `weapon_knife_survival_bowie_png.png` | `weapon_knife_bowie.vmdl_c` |

所以**不能**从图标文件名反推模型名。现在用一张显式别名表
（`visualizer/static/js/weapon-index.js` 的 `ICON_KEY_ALIAS`），target 一律是
**模型文件名的短 key**，因为模型文件名才是唯一真值。

**另外**：`weapon_knife`（默认刀）这个显示名在 demo 里就叫 `Knife`，不指定刀型。
它现在会优先取 `knife_default_ct`，取不到就取任意一把刀 —— 早先版本这种条目
**一把都匹配不上**，所以默认刀完全不显示（见 §9.7 的 bug 4）。

### 6.2 枪械图标：这是最容易拿、收益最高的一项

```powershell
& $BIN -i $VPK -f "panorama/images/econ/weapons/base_weapons/" -o .\icons\ -d --threads 8
```

**实测 0.8 秒，68 张 PNG，合计 5.6 MB。**

| 项 | 值 |
|---|---|
| 数量 | **68**（含 `ct_gloves` / `t_gloves` 两张手套图） |
| 尺寸 | **全部 512×384，RGBA，8-bit** |
| 单张 | 30–262 KB（AK-47 76 KB、AWP 66 KB、M4A1-S 106 KB） |

命名规则干净：`weapon_ak47_png.png`、`weapon_awp_png.png`、
`weapon_smokegrenade_png.png`、`weapon_knife_karambit_png.png` … 基本就是
`weapon_<items_game 名>_png`。

**[M] 内容已人工核对**：把 `weapon_ak47_png.png` 转成 ASCII 灰度图确认
**确实是 AK-47 侧面剪影**（机匣 + 斜插弹匣 + 枪托 + 准星），不是空白/损坏图。

**和 demo 的对接**：`demoparser2` 的 `active_weapon_name` 直接给显示名，
本次实测 `examples/demo/aurora-vs-vitality-m2-inferno.dem` 得到 **28 种武器**：

```
AK-47, Butterfly Knife, Karambit, M4A1-S, M4A4, Glock-18, USP-S, AWP,
Smoke Grenade, M9 Bayonet, Skeleton Knife, Flashbang, Five-SeveN,
MAC-10, Galil AR, ...
```

**所以只需要一张「显示名 → 图标文件名」的映射表**，就能把回放里每个玩家的
当前武器显示成**官方图标**。这是投入产出比最高的一项。

⚠️ **注意**：`base_weapons/` 只有**基础武器**图标。**皮肤图标**在
`panorama/images/econ/default_generated/`（**6,804 个**，命名如
`leather_handwraps_handwrap_camo_grey_heavy_png.vtex_c`）。
回放通常只需要基础武器图标。

---

## 7. 一键流水线

本次把上面所有实测结论固化成脚本，放在 **`tools/cs2_assets/`**：

| 文件 | 作用 |
|---|---|
| **`extract_cs2_assets.py`** | **主入口**：调 VRF 导出 agent / 武器 / 图标 / 粒子，再自动裁剪与校验；武器会写 `weapons_index.json`（精确 key→GLB，§9.4.4） |
| `prune_glb.py` | GLB 动画裁剪器（重建 buffer + 引用重映射） |
| `validate_glb.py` | 引用完整性校验（悬空 accessor / bufferView 越界 / 节点越界） |
| `check_skinning.py` | 蒙皮可用性校验（权重和 / joint 越界 / IBM 奇异） |
| `render_glb_ascii.py` | **纯 CPU 软件光栅化**：验证模型画得出来、动画真的会动（§5.4.1） |
| `render_glb_png.py` | **PNG 渲染器**（透视投影 + 深度缓冲 + 光照 + 按 joint 着色），用来肉眼判断模型对不对 |
| `glb_report.py` | GLB 结构报告（clip 列表、时长、体积分布） |
| `resize_textures.ps1` | 纹理降分辨率（System.Drawing，无外部依赖） |
| `encode_textures.py` | 缩放 + **按贴图类型选编码**（颜色用 WebP q90，法线/ORM 用无损），依据见 §5.5.1 |
| `normal_map_error.py` | **法线贴图压缩失真测量**（解回单位法线，逐 texel 算角度误差） |
| `build_weapon_icon_map.py` | **从游戏 schema 生成武器显示名 → HUD 图标映射表** |
| `install_web_assets.py` | **把提取产物归位到 `visualizer/static/assets/` 并生成 `manifest.json`**（解析 `.mks` 拿真实帧序，§9.0） |
| `optimize_weapon_glb.py` | **武器 GLB 瘦身**：贴图缩放 + 改写 GLB 里的 `images[].uri` + 重建容器；支持多目录、幂等（§9.4.1） |
| `weapon_orientation.py` | **量武器朝向**：两个独立指标（两端横截面 + 离质心最远顶点），支持多文件出对照表（§9.4.2） |
| `verify_weapon_textures.py` | **逐张校验贴图**：存在性 + Pillow 解码 + 尺寸 + 是否被材质引用（§9.5） |
| **`fire_tint_preview.py`** | **按 three 的实际 shader 顺序合成火焰三层**并出预览图 + 量化指标，用来校准 tint（§9.3.1） |

`notes/scratch/` 下的探针（**不是流水线的一部分，是一次性取证**）：

| 脚本 | 作用 |
|---|---|
| `probe_server.py` | 探服务器上资产的可达性与 MIME（§9.5） |
| `grenade_type_probe.py` | 读出 demo 里**真实的投掷物类名**与采样点数（§9.4.5） |
| `molotov_team_probe.py` | 火焰实例 ↔ 引爆时刻最近的投掷物配对，验"队伍→模型"（§9.4.5） |
| `molotov_exception_probe.py` | 把那 4 个不吻合的案例逐条挖开，区分"配对误差"和"真例外"（§9.4.5） |
| `where_is_server.py` | 枚举本机非回环 IPv4 并逐个探 `:5099`（回答"web 的 ip 是什么"） |

```powershell
$PY = "D:\conda\envs\cs-net-v5\python.exe"

# 看有什么
& $PY tools\cs2_assets\extract_cs2_assets.py --list agents
& $PY tools\cs2_assets\extract_cs2_assets.py --list weapons

# 一个 agent（自动裁剪到 92 个位移 clip）
& $PY tools\cs2_assets\extract_cs2_assets.py --agent ctm_sas --out assets

# 全部竞技 agent
& $PY tools\cs2_assets\extract_cs2_assets.py --agent-all --out assets

# 武器模型 + 全部官方 HUD 图标
& $PY tools\cs2_assets\extract_cs2_assets.py --weapons ak47,awp,deagle,m4a4 --weapon-icons --out assets

# 烟雾/火焰粒子定义 + sprite 序列帧
& $PY tools\cs2_assets\extract_cs2_assets.py --particles --out assets

# 纹理瘦身（二选一）
& tools\cs2_assets\resize_textures.ps1 -Src assets\agents -Dst assets\agents_web -MaxDim 1024
& $PY tools\cs2_assets\encode_textures.py assets\agents --out assets\agents_web --max-dim 1024
```

**端到端实测**（`--agent ctm_fbi --weapon-icons`）：

```
[agent] ctm_fbi
  animations : 2091 -> 92
  size       : 1,067.7 MB -> 23.3 MB (2.19%)
[weapon-icons] HUD icons
done -> notes\scratch\asset_probe\pipeline_out
TOTAL elapsed: 62.1s
```

产物：`agents/ctm_fbi.glb` **23.3 MB**（92 clips，校验通过，蒙皮通过）
+ `weapon_icons/` **68 张 / 5.6 MB**。

### 7.1 ⚠️ 两个会让接入失败的坑（本次实测发现）

这两个都是**「资产没问题但用不了」**的类型，只有在真的用加载器跑一遍才会暴露。

#### 坑 A：贴图必须和 GLB 同目录 —— 原脚本把 agent 贴图搬错了地方

VRF 把 GLB **和它的贴图 PNG 一起**写进输出目录，而 GLB 里用**裸文件名**引用它们
（`"uri": "ctm_sas_body_color_psd_d02da1ae.png"`）。所以 **GLB 必须留在那个目录里**。

原脚本的实现是：

```python
raw   = args.out / "_raw" / f"{key}.glb"     # GLB + 贴图都在这
final = args.out / "agents" / f"{key}.glb"   # 只把 GLB 挪走
raw.unlink(missing_ok=True)                   # 贴图留在 _raw，永远不搬
```

**实测证据**：`ctm_fbi.glb` 引用 24 张贴图，**同目录命中 0/24，全部在 `_raw/`**。
浏览器加载 `agents/ctm_fbi.glb` 会得到**一个没有贴图的灰模**。

**已修**：改成 `agents/<key>/<key>.glb`，GLB 原地裁剪替换（`pruned.replace(raw)`），
贴图留在旁边。修复后实测 **24/24 引用解析成功**。

```powershell
& $PY tools\cs2_assets\extract_cs2_assets.py --agent ctm_sas --out assets
#   -> agents\ctm_sas\ctm_sas.glb  (25 texture PNG(s) beside it)
```

#### 坑 B：武器 GLB 有蒙皮属性但没有 `skins`（three.js 会静默降级）

武器 GLB 的 primitive 带 `JOINTS_0` + `WEIGHTS_0`，但**没有 `skins` 数组**，
节点也**没有 `skin` 引用**。这是 glTF 规范里**非法的组合**。

实测这些数据是**刚性绑定**（每个顶点只受 1 个 joint 影响，权重和 = 1.0，
joint 索引 0–5），也就是「一把静态枪被导成了蒙皮网格」。

**用真的 three.js r160 GLTFLoader 在 Node 里跑**（`visualizer/tests/test_gltfloader_node.mjs`）：

| 模型 | 结果 |
|---|---|
| `weapon_rif_ak47.glb` | `parse OK`，`SkinnedMesh=0  plainMesh=4`，39,649 三角面 |
| `weapon_knife_karambit.glb` | `parse OK`，`SkinnedMesh=0  plainMesh=1`，13,938 三角面 |
| `ctm_fbi.glb`（agent） | `parse OK`，`SkinnedMesh=9 (withBones=9)`，37,670 三角面，92 动画 |

**结论：three.js 静默丢弃蒙皮属性，把武器当刚性 Mesh 渲染在 bind pose —— 不报错，
可以直接用。** 但要知道：**武器不会有任何骨骼动画**，而且不同 three.js 版本
是否同样宽容**未验证**（只测了 r160）。

> **好消息**：agent 的蒙皮是**正确**的（`skins=70`、`withBones=9`），
> 和武器的问题**不共享根因**。

### 7.2 浏览器加载验证：缺口缩小到最后一环

原来「完全没验证过」，现在拆成三段，前两段已闭环：

| 环节 | 状态 | 证据 |
|---|---|---|
| GLTFLoader 能被加载 | ✅ **已验证** | 起 Flask 实测 `/static/three/examples/jsm/loaders/GLTFLoader.js` → 200，`text/javascript`；r160 与 vendored three **版本一致** |
| GLB 能被 GLTFLoader 解析 | ✅ **已验证** | Node 里用真 GLTFLoader 解析，agent 得到 9 个带骨骼 SkinnedMesh + 92 动画；武器得到刚性 Mesh |
| 贴图能被取到 | ✅ **已验证** | 24/24 URI 同目录解析；Node 起 HTTP 服务器实测被请求 |
| **浏览器里贴图能解码并渲染出画面** | ❌ **未验证** | Node 无 `createImageBitmap`，解码步骤只能在浏览器完成 |

**还缺的就是最后一行** —— 需要真的浏览器（playwright 或人工打开页面）。

**顺带发现**：`visualizer/static/three/examples/jsm/` 里**原本只有 OBJLoader**，
没有 GLTFLoader。已补 `GLTFLoader.js` + `BufferGeometryUtils.js`（r160，
从 npmmirror 取，**不需要 CDN**）。`index.html` 的 importmap 里
`"three/addons/"` 早已声明，所以**零代码改动**即可 import。

#### 坑 C：`.gitignore` 的 `examples/` 会静默吃掉 vendored three addons

`.gitignore` 第 7 行原本是**未锚定**的 `examples/`（本意是忽略仓库根的
`examples/` demo 文件）。但未锚定的模式**匹配任意层级**，所以它同时匹配
`visualizer/static/three/examples/`。

**为什么一直没暴露**：gitignore **不作用于已跟踪文件**。现有的
`three.module.js` / `OBJLoader.js` / `OrbitControls.js` 早已被跟踪，所以看起来一切正常。
**但任何新加进 `three/examples/` 的文件都不会被提交** —— 也就是说
**全新 clone 之后，importmap 里的 `three/addons/` 会 404**，
而本地因为文件存在，**完全不会察觉**。

**已修**：改成锚定的 `/examples/`。验证：GLTFLoader 现在会被提交，
而 `examples/demo/*.dem` 仍被正确忽略。

> 这是本次第三个「**只在干净环境才会暴露**」的坑（另两个见 §7.1）。
> 本地能跑 ≠ 别人 clone 下来能跑。

---

## 8. 拿不到 / 需要妥协的东西

诚实列出边界：

| 项 | 情况 |
|---|---|
| **动画状态机（AG2 graph）** | **拿不到可执行逻辑**。`.vnmgraph_c` 有 232 个、能 decompile 成文档，但 **VRF 从不执行它**（`research-fidelity-ai.md` §2.5 已确认）。位移 clip 选择、混合空间、转向、IK 全部要**回放侧自己实现**。这是人物真实度的**最大缺口**。 |
| **静态光照 / lightmap** | 不导出。`research-fidelity-ai.md` §3 给的评估是 30–60% 真实度。 |
| **3D skybox** | 明确不导出。 |
| **材质精确度** | 着色器图被压平成 glTF PBR；多层混合材质**只导一层**。**外加本次的 vcs 72 不兼容**（§2.1），金属度通道推断会缺失。 |
| **布料 / 变形约束** | 布料求解不存在，布料骨骼被刚性固定。 |
| **命中盒（hitbox）** | glTF 没有这个概念，不导出。 |
| **仅导最高 LOD** | 低 LOD 被有意丢弃（VRF 设计如此）。 |
| **武器动画** | 武器 GLB 本身无动画；换弹/开火动画挂在玩家骨架的 `viewmodel/` clip 上。 |
| **第一人称手臂** | `*_pm_arm` 是独立一套，`NOTES.md` §7.4 记过它不带 animgraph 引用。**但本次发现 agent 模型自带 `firstperson_default_gloves_arms` 网格**（12,508 面），所以第一人称视角其实有素材。 |
| **WebP / JPEG 编码** | System.Drawing 无 WebP 编码器；**已装 Pillow 12.3.0**，WebP 路径实测可用（§5.5.1）。 |

---

## 9. 落地实现（前四项**已落地**，第 5 项未做）

§9 原本是一份「建议清单」。**第 1–4 项已经实现并验证**，这里改成实现记录；
第 5 项（玩家模型 + 位移动画）按用户要求**本次不做**。

### 9.0 资产落位

浏览器只能取到 `visualizer/static/` 下的东西，所以资产要**归位**。
`tools/cs2_assets/install_web_assets.py` 负责这一步，并生成 `manifest.json`：

```powershell
python tools\cs2_assets\install_web_assets.py --weapon-icons --smoke --fire `
    --weapons ak47,awp,karambit,smokegrenade,molotov,c4
```

**为什么必须有 manifest**：VRF 拆 sprite sheet 时把**帧顺序和 LOOP 标记**写在
`.mks` 边车文件里，**不在文件名里**；而且一个贴图可能有多条序列
（`fire_onsurface` 有 3 条共 330 帧）。所以帧列表必须从 manifest 读，
**按文件名排序猜是错的**。`install_web_assets.py` 解析 `.mks` 得到真实帧序。

`visualizer/static/assets/` 是 Valve 版权资产，**已入库**（903 个文件、
约 300 MB；理由和代价见 §9.0.1）。原始的提取产物 `assets/`、
`notes/scratch/asset_probe/`、`tools/Source2Viewer-CLI/` 仍然全部 gitignore。

#### 9.0.1 为什么资产最后还是入库了

原设计是"不入库，新克隆自己重跑"。**这条路走不通**：

- 归位脚本只认 `<probe>\pipeline_out\weapon_icons`、
  `<probe>\particles_out\particles\materials\particle\...`（`install_web_assets.py`
  的 136 / 163 / 179 行），武器索引又在 `<probe>` **根目录**找；而提取脚本默认
  `--out assets`，产出的是 `assets\weapon_icons\...`、`assets\particles\...`、
  `assets\weapons_index.json`。两边对不上，要凑齐得跑三次、三个不同的 `--out`。
- 图标映射表 `weapon_icon_map.json` 由 `build_weapon_icon_map.py` 生成，还要先从
  VPK 里取出 `items_game.txt` / `csgo_english.txt` —— 这一步没有任何脚本接上。
- 前提本身也不轻：本机 CS2 安装 + 108 MB 的 `Source2Viewer-CLI.exe`（同样是
  gitignore 的第三方二进制）。

也就是说"全新 clone 一条命令重建资产"当时是**假**的。所以决定把**归位后的**
资产入库：903 个文件（358 帧序列 + 68 张 HUD 图标 + 65 个武器目录），约 300 MB，
其中武器那部分按 §9.4.1 实测是优化后的 280.9 MB。

两笔代价是刻意接受的：

| 代价 | 具体 |
|---|---|
| 仓库体积 | clone 多下 ~300 MB；每次重新提取都是几百个二进制 diff |
| 许可 | Steam 只给了**个人非商业使用**（SSA §2.A）。**这个仓库必须保持私有**；公开、公开 fork、或者把资产挂进 release，就是分发 Valve 资产（讨论见 `research-assets.md` §6、`research-fidelity-ai.md` §2.6） |

连带改动一处：`.gitignore` 里那条**未锚定的 `*.glb`**（本意是"GLB 一律不进
仓库"）被**删掉了**。它当时会连 staged 的武器模型一起吃掉，而补一条
`!visualizer/static/assets/**/*.glb` 只是绕过去 —— 绕过去之后，谁把 GLB 放到
`assets/` 以外（比如测试夹具），照样会被**静默跳过**，和 §9.7 bug 3（
`examples/` 静默吃掉 vendored three addons）是同一个坑。

所以改成**按位置管**：GLB 只在 `/assets/` 和 `notes/scratch/asset_probe/` 这两个
**整目录被忽略**的地方被挡住（2 GB+ 的探针产物本来就是靠目录规则挡的，数过一遍
—— `*.glb` 在里面**一条都没多挡**），别处默认放行，新出现的 GLB 会在
`git status` 里露头，由人决定，而不是消失。

这个改动**当前对文件没有任何影响**，实测：仓库里一共 **209 个 `.glb`**，
144 个在 `notes/scratch/asset_probe/` 下（照旧忽略），65 个在
`visualizer/static/assets/weapons/`（照旧入库），209 = 144 + 65，没有第三个位置。

### 9.1 第 1 项：枪械 HUD 图标 ✅

* 68 张官方图标（512×384 RGBA）→ `assets/weapon_icons/`
* `build_weapon_icon_map.py` 从 `items_game.txt` 生成
  **显示名 → 图标文件名**映射（63 个显示名，demo 的 30 个武器名 **30/30 覆盖**）
* 渲染：玩家头顶 badge = 图标 + 名称（`visuals.js` 的 `_getWeaponBadge`）。
  图标是**非正方**的，所以按原始宽高比画；异步到达，先用文字占位再重画
* 开关：`toggle-weapon-icons`

#### 9.1.1 第二轮：击杀信息也显示图标

右上角击杀信息原来是 `击杀者 · 武器名 · ☠ 受害者` 的纯文字。
现在武器名位置直接换成官方图标（`title`/`alt` 保留名字做悬停提示）。

两个实现细节：

* `weaponIconTexture()` 返回的是 `THREE.Texture`，只能给 WebGL 用，
  塞进 `<img src>` 会得到 `[object Object]`。所以新开了
  `weaponIconUrl()` 返回真实 URL —— 这类"看起来是同一件事其实不通用"的接口
  很容易顺手用错
* 图标映射表是**异步**加载的。第一次调用时可能还没好，所以 `weaponIconUrl()`
  返回 null 并在加载完成后回调一次；条目已经画出来了就补上图标。
  没有图标的武器（未提取的皮肤刀）保留文字
* CSS 里**定高不定宽**（`height: 22px; width: auto`）—— 图标是 512×384 非正方，
  定宽定高会压扁

### 9.2 第 2 项：烟雾真实 sprite ✅

* `smokeloop_i_0_sc_hardedge`，**128 帧**，1.47 MB
* **关键发现**：vpcf 里第二张贴图 `smokeloop_i_0_flwmix` 是
  `SPRITECARD_TEXTURE_ANIMMOTIONVEC`（**流向图**，驱动粒子运动矢量），
  **不是第二张可见贴图**。所以外观就是这一张 128 帧序列
* 渲染：在体素云**包围盒中心**放一个面向相机的 billboard，按时间翻帧
  （`_placeSmokeSprite` / `_updateSmokeSpriteFrames`）。体素给真实形状和遮挡，
  sprite 给真实外观
* **翻帧速率是估的**：vpcf 里**没有帧率字段**。取 15 fps（128 帧 ≈ 8.5 秒一轮），
  可用 `setSmokeSpriteFPS()` 调。这是已知的近似，见 §10
* 时间基准取**该团烟引爆后的秒数**，所以不同时间扔的烟不会同步翻帧
* 开关：`toggle-smoke-sprites`

### 9.3 第 3 项：火焰三层 sprite ✅

按 `flames` / `flame_omni` / `flame_wisp` 三张序列叠三层：

| 层 | 序列 | 相对尺寸 | 不透明度 | 翻帧速率 | 作用 |
|---|---|---|---|---|---|
| core | `flames` (32 帧) | 0.55 | 0.95 | 24 fps | 亮芯 |
| filler | `flame_omni` (60 帧) | 0.85 | 0.62 | 18 fps | 主体 |
| edge | `flame_wisp` (48 帧) | 1.20 | 0.34 | 13 fps | 外沿飘散 |

**三层各用自己的 fps 和相位**（相位 0 / 0.37 / 0.71）—— 同步翻帧会一眼看出
「三层是同一张图」。外沿还沿法向多抬 2.6 cm，避免三层叠成一张贴纸。

每格仍躺在**真实表面法向**上（`m_firePositions` + `m_fireNormals`），只是每格画三片。

开关：`toggle-fire-sprites`（关掉时退回原来的单层程序贴图，不是让火消失）。

#### 9.3.1 第二轮修复：火是白的 —— 因为贴图是灰度亮度遮罩

见 §4.2.1 的实测：三张序列的通道极差只有 1.5/255，**完全没有颜色**。
原来的实现把它们当彩色贴图直接贴上去，加色混合下等于 `白 × 灰度`。

**修法**：每层给一个 `MeshBasicMaterial.color` tint。`color` 分量允许 >1 ——
加色混合下这就是"增益"，而渲染器开了 ACES tone mapping（`scene.js`，exposure 0.95），
超过 1 的部分被压回高光而不是硬切，正好是火芯该有的过曝滚降。

**tint 的数值不是拍脑袋定的**，是 `tools/cs2_assets/fire_tint_preview.py`
按 three 的**实际 shader 顺序**合成出来量的：

```
sRGB 解码 → 乘 tint × opacity（线性空间） → **逐层** ACES → sRGB 编码
→ 加色混合 contribution = rgb × (tex_alpha × opacity)
```

注意第 3 步是**逐层、在混合之前**做的（tone mapping 在片元着色器里，混合在帧缓冲里），
所以某层 tint 过烫会**自己**饱和成白，而不是和其它层一起被压 —— 这一点决定了
`flame_wisp` 的增益必须最小。

指标用可见像素的 HSV 饱和度分布（"有多白"就是这个）：

| 配色 | 平均色相 | 低饱和(<0.20)占比 | 亮部 RGB |
|---|---|---|---|
| **旧行为（无 tint）** | — | **100.0%** | (0.601, 0.601, 0.601) |
| A | 33.8° | 8.1% | (0.685, 0.575, 0.395) |
| **B（采用）** | **33.5°** | **3.3%** | (0.667, 0.526, 0.294) |
| C | 34.5° | 5.5% | (0.656, 0.529, 0.316) |
| D | 33.8° | 1.1% | (0.653, 0.503, 0.249) |

旧行为**三通道完全相等、100% 低饱和** —— 这就是"火是白的"的实测证据，
而不是"我觉得应该是这个颜色"。选 B：色相 33.5°（橙），只剩 3.3% 近白像素，
那点就是白热芯（真实火也有）。

| 层 | 序列 | 贴图线性均值 | tint |
|---|---|---|---|
| core | `flames` | 0.042 | (3.40, 2.00, 0.70) |
| filler | `flame_omni` | 0.167 | (2.60, 1.05, 0.20) |
| edge | `flame_wisp` | 0.312 | (1.60, 0.42, 0.07) |

增益按各层贴图亮度配：贴图暗的层给大增益，贴图亮的层给小增益，叠加后亮度才均衡。

预览图（合成 | core | filler | edge，以及 4 帧胶片条）：
`notes/scratch/fire_final.png` / `fire_final_frames.png`。

#### 9.3.2 第二轮修复：火出现瞬间的卡顿

两个原因，都修了：

**① 逐帧设 `material.needsUpdate = true`。**
原来每次翻帧都设，三层加起来 24+18+13 = **55 次/秒·每个燃烧实体**，
每次都 bump `material.version`，逼 `WebGLRenderer.setProgram` 重走一遍 program 判定。

**这个 `needsUpdate` 根本不需要**：换的是同一序列里的另一帧，
`colorSpace`/`flipY`/`mapping` 全都一样，shader 源码不变，
three 每帧自己会把新的 map uniform 传下去。
（只有 `map` 从 **null 变成贴图**时才真需要 —— 那时 `USE_MAP` 宏变了。
那几处仍然保留。）

**② 首次创建材质时同步编译 3 个 shader program。**
火焰材质是第一次看到火时才建的（`getFireCell`），编译停顿正好落在
"火刚烧起来"的那一帧。新增 `prewarmFireShaders()`：
在加载阶段用**完全相同的材质配置**、在离地图 10000 单位处画一帧，
调 `renderer.compile()` 提前编译，然后立刻拆掉。

必须**等序列加载完**再预热 —— `map` 从 null 变贴图会换一个 program，
用兜底贴图预热等于预热了个别的（所以这个函数是 async 的）。
`renderer.compile()` 走的是 `scene.traverse` 且看 `object.material`，
`Sprite` 也覆盖，所以烟雾 sprite 的 program 一并预热了。

### 9.4 第 4 项：武器 3D 模型 ✅

**65 把**武器挂到玩家右手：全部枪械 + 22 把刀 + 全部投掷物 + C4 + 医疗针。
（第一轮只有 6 把，这是"有些枪看不到模型"的真正原因，见 §6.1.1。）

#### 9.4.1 体积：1994 MB → 280.9 MB

VRF 按**源分辨率**导贴图（4096²，单张 15–18 MB），65 把合计 **1994 MB**，
直接发给浏览器不现实。`optimize_weapon_glb.py` 同时做两件事：

1. 贴图缩到 1024² 并重新编码（颜色图有损 WebP，法线/ORM 无损 —— 依据见 §5.5.1）
2. **改写 GLB 里的 `images[].uri`**（glTF 按文件名引用贴图，改名不写回就是 404）

第 2 步要重建容器：JSON chunk 长度变了，12 字节头和 chunk 表都要重算，
每个 chunk 补回 4 字节对齐。

| | 数值 |
|---|---|
| 原始 | 540 文件 / **1994.4 MB**（410 张 PNG 占 1842 MB） |
| 优化后 | **280.9 MB（14.1%）** |
| 最大单把 | `sg556` 67.9 → 9.0 MB（13.3%） |
| 最小单把 | 默认刀 0.4 MB |

**在 probe 里优化，再 staging** —— 这样只复制优化后的小文件，
不用把 2 GB 搬进 `visualizer/static/`。

脚本改成支持多目录 + **幂等**：源 PNG 在转换时就被删掉，
所以第二次跑会打印 `already optimized (no PNG left) -- skipped`，
而不是在 `after_tex/before_tex` 上除零。

#### 9.4.2 朝向是**量出来的**（这里我一开始搞错了，第二轮又修正了一次）

`weapon_orientation.py` 用**三个指标**，因为没有一个能通吃：

1. **两端横截面**：沿长轴切片，比两端横截面积。枪口/刀尖远细于枪托/刀柄。
   对枪和直刀可靠；**被细折叠枪托骗过**（mp9/bizon，见下）。
2. **离质心最远的顶点**：任何东西的尖端都是离质心最远的点。与曲率无关，
   所以能定弯刀。**但对枪不可靠** —— AK 的枪托比枪口更远离质心
   （重的机匣把质心往前拉了），它会报 −z。
3. **tall 轴**（横截面尺寸更大的那个垂直轴）：定"哪边朝上"用这个。
   枪比枪宽高，所以垂直轴必然更"高"。**不能用"顶点在中心以上的比例"**，
   那个量的是原点位置，见下。

**枪械（35 把全量实测）**：**34 把长轴是 z**（唯一例外见下），
其中 **32 把横截面细端是 +z**、2 把报 −z（是折叠枪托骗的，见下）。
覆盖步枪/狙击/冲锋/手枪/霰弹/机枪全部六类：

| 类别 | 代表 | 长轴 | 横截面细端 | tall 轴 | 比值 |
|---|---|---|---|---|---|
| 步枪 | `weapon_rif_ak47` | z 0.96 m | +z | **y** | 3.7× |
| 狙击 | `weapon_snip_awp` | z 1.39 m | +z | **y** | 2.4× |
| 冲锋 | `weapon_smg_mp5sd` | z | +z | **y** | 4.4× |
| 手枪 | `weapon_pist_fiveseven` | z 0.21 m | +z | **y** | 4.1× |
| 霰弹 | `weapon_shot_mag7` | z 0.67 m | +z | **y** | 3.9× |
| 机枪 | `weapon_mach_m249` | z 1.21 m | +z | **y** | 2.4× |

**"+Y 是上"的依据是 tall 轴，不是"顶点在中心以上"的比例。** 这一点我一开始写错了：
`above centre` 量的是**原点在包围盒里的位置**，和"哪边朝上"是两件事，而且差异极大 ——
AWP 的 y 是 48%（原点几乎居中），M249 的 y 是 **13%**（原点靠近顶部）。
照那个读，M249 会被判成"倒着的"，但它是好的。

真正的依据是：**枪比枪宽高**，所以垂直轴必然是横截面尺寸更大的那个轴。
34/35 把枪的 tall 轴是 y，最小比值也有 **1.30×**（`negev`）。配合"枪口在 +z"
和 Source 2 的 +Z 前 / +Y 上约定（玩家模型独立印证：面罩在 z=+0.03），
**+Y 就是上**。

唯一的例外是 `weapon_pist_elite`（Dual Berettas）：**两把枪并排**，
包围盒长轴是 **x（0.563 m）**，tall 轴变成 z。按几何判会得到完全错误的类别 ——
所以类别判定用的是 Valve 的**中缀**而不是几何（见下）。

**结论：所有枪械的局部坐标系（+Z 前、+Y 上）和玩家 group 完全一致，旋转必须是 0。**
我最初按「长轴 +Z 所以要绕 X 转 −90° 放平」设了 `rot: [-π/2,0,0]` ——
那个旋转把 +Z 映射到 **−Y**，枪会**朝下插进地里**。
（玩家朝向是本地 +Z：面罩在 z=+0.03、视线锥在 z=+0.34。）

**刀（22 把）**：长轴一律 y，20 把横截面细端和尖端**都**是 +y →
绕 X 转 −90°（+Y → +Z）让刀尖朝前，正确。

**⚠️ 但 `mp9` / `bizon` 差点被我改错。** 这两把枪的**两个指标都说枪口在 −z**，
和其余 40 多把枪相反。查下去发现是**折叠枪托**：

```
weapon_smg_mp9  slices along z:
   bin 0    1      2      3      4      5      6      7      8      9
   0.122  0.020  0.014  0.000  0.068  0.179  0.255  0.093  0.172  0.165
          ^^^^^^^^^^^^^^^^^^^^ 细金属杆 = 折叠托，不是枪口
```

bin 1–3 的横截面只有 0.000–0.020 m —— 那是折叠起来的**细金属托杆**，
而 +z 端（bin 9）是 0.165 的机匣/枪管。所以枪口仍在 +z，和其它枪一致。
**如果只看"两个指标都同意"就照做，这两把枪会被改成反的。**

**尺寸本身就是米**，不需要缩放（AK 0.96 m、AWP 1.39 m、C4 0.27 m，都是真实尺寸）。

**类别判定用 Valve 自己的中缀**（`rif`/`snip`/`smg`/`pist`/`mach`/`shot`），
不是从图标 key 反推、也不是从几何长轴推 —— 后者会被 Dual Berettas 骗（见上）。
姿态表 `CLASS_ORIENT` 里霰弹枪和机枪是**独立条目**（数值与步枪相同），
不并进 `rifle`：合并会让日志和测试打出 `MAG-7 -> [rifle]` 这种误导性输出，
而这正是排查朝向问题时最需要看的信息。`check_weapon_coverage.mjs` 会断言
**每个出现的类别都有姿态条目** —— 因为 `CLASS_ORIENT[cls] || CLASS_ORIENT.rifle`
这种静默兜底正是之前"刀被按步枪姿态挂"的成因。

#### 9.4.3 限制

* **没有骨骼动画**：武器 GLB 带 `JOINTS_0`/`WEIGHTS_0` 但**没有 `skins` 数组**
  （glTF 规范里非法）。实测是**刚性绑定**，three.js **静默丢弃**蒙皮属性、
  当刚性 Mesh 渲染在 bind pose。不报错、能用，但**不会有任何骨骼动画**
* 挂点用**原点**而不是包围盒中心：AK 的包围盒是 z∈[−0.296,+0.663]，
  原点就在握把附近，归零反而把枪整体前推 0.18 m。刀/投掷物的原点不在握持处，
  才按包围盒中心归零
* **挂点偏移是手调的**（沿用玩家模型里手部 x=±0.30、y=1.13 的常量），
  不是真实挂接骨骼。65 把枪尺寸差异大（0.21 m 手枪 ~ 1.39 m AWP），
  同一套偏移不可能对每把都最优。见 §10
* 开关：`toggle-weapon-models`

#### 9.4.4 显示名 → 模型：显式别名表，不再模糊匹配

原来的实现拿"图标 key"去**模糊匹配**模型目录名（`key.endsWith(dir)`），
坏得很安静。用现在完整的 65 把模型跑旧逻辑，仍然有 **8 个显示名解析不出来**：

```
FAIL  Knife, M4A4, M9 Bayonet, Incendiary Grenade, Glock-18,
      Navaja Knife, Bowie Knife, Talon Knife
```

原因是 Valve 的图标文件名和模型文件名在好几处**没有规律地**对不上（表见 §6.1.1）。

现在解析逻辑抽成纯函数模块 `visualizer/static/js/weapon-index.js`
（**不 import three**，所以 node 里能直接测）：

1. 模型文件名 → 短 key（去掉 `weapon_` 前缀和类别中缀）
2. 图标 key 直接命中，否则查 `ICON_KEY_ALIAS` 显式别名表
3. 通用 `Knife` 兜底：优先 `knife_default_ct`，否则任意一把刀
4. 同时登记**内部 key**（图标文件名里那个），因为 demoparser2 有时漏内部名
   —— 这份 demo 里就有 `knife_skeleton`（图标表里叫 "Skeleton Knife"）

配套测试 `visualizer/tests/check_weapon_coverage.mjs`：拿真实 manifest + 真实图标表
+ 真实 demo 武器清单（fixture）跑**浏览器实际用的那份解析代码**，
断言每个 demo 武器都能解析出模型、且 GLB 文件真实存在、且刀/投掷物类别没判错。

**这个测试有牙**：把 manifest 换回最初 6 把，它会报 **23/30 失败**。

#### 9.4.5 飞行中的投掷物：从"一个球"换成真实模型 ✅

用户反馈："把飞行中的道具换成真实的游戏模型，而不是一个圆点"。

改之前那是个半径 **0.2 m 的球**（`SphereGeometry(0.2)`），按类别染色
（烟=灰、闪=白、高爆=红、火=橙、诱饵=黄）。现在换成 6 个真实 GLB。

**① demo 里的投掷物类型是引擎类名，不是显示名 —— 必须实测，不能猜。**

`notes/scratch/grenade_type_probe.py`（de_inferno，24 回合，327266 个采样点）：

| `events.grenades[].ty` | 采样点 |
|---|---|
| `CSmokeGrenadeProjectile` | 249909 |
| `CHEGrenadeProjectile` | 46101 |
| `CMolotovProjectile` | 15435 |
| `CFlashbangProjectile` | 14456 |
| `CDecoyProjectile` | 1365 |

**就这 5 种。** 所以映射表用**完整类名精确匹配**（`NADE_PROJECTILE_MODEL`），
认不出来返回 `null` → 退回球，而不是挂个错模型。原来那套
`grenadeCategory()` 的显示名子串匹配是给**颜色**用的，
`'CHEGrenadeProjectile'.includes('HE')` 恰好成立纯属运气。

**② 莫洛托夫和燃烧弹在游戏里共用同一个 projectile 类，只能靠队伍区分 —— 而且有实测例外。**

CS2 里莫洛托夫只有 T 能买、燃烧弹只有 CT 能买。为了确认这条规则，写了两个探针
把 139 个火焰实例和"引爆时刻空间上最近的投掷物"配对
（`molotov_team_probe.py` / `molotov_exception_probe.py`）：

```
     64  队伍=CT   火焰类型=incendiary   ✅
     71  队伍=T    火焰类型=molotov      ✅
      2  队伍=CT   火焰类型=molotov      ❌
      2  队伍=T    火焰类型=incendiary   ❌
```

4 个例外逐条挖开看：2 个是**配对误差**（两颗落点很近，最近邻挑错了），
2 个是**真的** —— 回合 idx=3 有个 CT 丢出莫洛托夫（火焰 `type_name=molotov`，
配对距离仅 2.4 HU，不可能错），回合 idx=10 有个 T 丢出燃烧弹。
原因是 CS2 允许**捡起地上的投掷物再丢**。

**结论：这个映射大约 1.4% 会选错模型，是已知的、有据可查的误差。**
要完全准确得拿投掷物落点去和 `fire[].inferno_type`（游戏自己的真值）配对，
代价是把火焰数据耦合进 trajectories。目前没做。

**③ 顺带量出来的两件事：Valve 真的会共用网格。**

| 模型 | 并集包围盒（m） | primitive | 说明 |
|---|---|---|---|
| `smokegrenade` | 0.0527 × 0.1181 × 0.0594 | 1 | |
| `incendiarygrenade` | **0.0527 × 0.1181 × 0.0594** | 1 | 和烟雾弹**几何完全相同**，只有贴图不同 |
| `flashbang` | 0.0657 × 0.1518 × 0.0835 | 1 | |
| `decoy` | **0.0657 × 0.1518 × 0.0835** | 1 | 和闪光弹几何相同，连 BIN chunk 的 sha1 都一样（`3c3a6b3d200d`） |
| `hegrenade` | 0.0925 × 0.1270 × 0.0987 | 1 | |
| `molotov` | 0.1193 × 0.3506 × 0.2319 | **3** | 瓶身 / 火焰片 / **瓶里的液体**（液体比瓶底还低 0.10 m） |

莫洛托夫那 3 个 primitive 是个坑：只量第 0 个（瓶身）会得到 0.091×0.250×0.210，
而 three 的 `Box3().setFromObject()`（app 里真正用的那个）量到的是
0.119×0.351×0.232。`check_nade_models.mjs` 因此改成量**所有 primitive 的并集**，
现在和浏览器里量出来的数字逐位一致。

**④ 尺寸：放大 2.5 倍，但比原来那个球更接近真实。**

真实投掷物只有 5–25 cm，整图视距下只有两三个像素。但**原来那个球直径 0.4 m，
是真实烟雾弹（0.118 m）的 3.4 倍**。所以放大 2.5 倍（烟雾弹 0.30 m 高）反而
**比改造前更接近真实尺寸**。常数是 `visuals.js` 的 `GRENADE_MODEL_SCALE`，
想要严格 1:1 改成 1 即可。

**⑤ 朝向是从轨迹推的，因为 demo 里根本没有投掷物的角度数据。**

`extract.py` 的 `_grenades()` 只存 `x/y/z/t/ty/eid/th`，没有 `m_angRotation`。
所以把模型长轴（实测 6 个模型**全部**是 +Y，有断言盯着）对齐到**这一段的速度方向**。
抛物线飞行时模型会自然地从仰角转到俯角。**这是推导量，不是原始数据**；
真实投掷物是翻滚的，这里不假装知道翻滚相位。落地后速度≈0 时保持上一帧朝向。

**⑥ 兜底和开关。**

模型是异步加载的，所以每条轨迹的载体是 `THREE.Group`，先挂一个兜底球占位，
到货后挂模型层、隐藏球。**关掉「投掷物模型」开关 = 退回改造前那个球**（一键 A/B），
不是"什么都不画"。加载失败（manifest 缺 key / GLB 404 / 轨迹已清）一律保留球。

释放有一条硬规则：`clone()` **不复制 geometry/material**，所以
`clearGrenadeTrajectories()` 只能 dispose 兜底球的材质，**绝不能**碰模型共享的
几何和材质 —— 否则换回合后所有投掷物一起坏掉。

**⑦ 证据（不是"看着对"）。**

* `visualizer/tests/check_nade_models.mjs`（纯 node，无需服务器）：
  5 个实测类名全部有映射、队伍区分燃烧弹、6 个 GLB 都存在、
  **6 个模型长轴全部是 y**、尺寸都在 0.02–0.5 m
* 无头 Chrome 自检（`run_browser_selftest.ps1`，真的走 GLTFLoader + WebGL）：
  ```
  [PASS] 长轴对齐速度方向：最大偏差 1.11e-16
  [PASS] 6/6 个投掷物模型可直接用于飞行中的那一颗
  [PASS] smokegrenade: 1/1 贴图已解码, 长轴=y, 放大 2.5× 后中心偏差 3.1e-17 m
  [PASS] 渲染 smokegrenade（2.5×，长轴对齐 (1,0.6,0)）：6,464 个非背景像素 (10.1%)
  [PASS] uncaught errors / rejections: 0
  ```
  —— **这是本项目第一次有"模型在真实浏览器里渲染出像素"的机器可读证据**（§9.5 里
  武器模型那一步仍然只有"GLTFLoader 能解析 + 贴图能解码"）。

**⑧ 开关**：`toggle-nade-models`。数据检查面板的「投掷物轨迹（回合）」一行会列出
按模型分组的条数 + 还在加载的条数。

### 9.5 验证：证据链

浏览器渲染这一步原本是最大的未知（§10 旧第 1 条）。现在拆成可验证的几段：

| 环节 | 手段 | 结果 |
|---|---|---|
| three + GLTFLoader 可用 | 真 Flask 服务器 + 真 GLTFLoader | ✅ r160，与 vendored 版本一致 |
| GLB 能被解析 | Node 里跑真 GLTFLoader | ✅ AK 4 网格 39,649 面；agent 9 个带骨骼 SkinnedMesh + 92 动画 |
| 贴图引用能解析 | 逐个 uri 走 HTTP | ✅ 345/345 |
| **浏览器能建 WebGL 并解码贴图** | **headless Chrome 跑 `static/selftest.html`** | ✅ WebGL 2.0；官方图标 512×384；**四条 sprite 序列全部解码** |
| 345 张贴图都是真图 | Pillow 逐个解码 + 尺寸校验 | ✅ **345 可用 / 0 坏**（65 把武器全量） |
| 显示名 → 模型解析正确 | `check_weapon_coverage.mjs` 跑浏览器同款解析代码 | ✅ demo **30/30**；换回 6 把模型会报 23/30 失败 |
| 武器朝向 | `weapon_orientation.py` 两个独立指标，65 把全量 | ✅ 枪械一律 +z；刀 20/22 两指标一致 |
| 服务器真能取到新资产 | `notes/scratch/probe_server.py` | ✅ 65 个 GLB 全 200 `model/gltf-binary`，贴图 `image/webp` |
| 火焰颜色 | `fire_tint_preview.py` 按 three 的 shader 顺序合成 + 量化 | ✅ 低饱和像素 100% → 3.3%，色相 33.5° |
| **投掷物模型在浏览器里出画面** | headless Chrome 渲染 + `readPixels` 数非背景像素 | ✅ **6,464 像素（10.1%）**，见 §9.4.5 |
| 武器模型在浏览器里出画面 | headless Chrome 的渲染段 | ✅ **1,400 像素（2.2%）** |

**浏览器自测现在是完整跑通的**（`visualizer/static/selftest.html` +
`visualizer/tests/run_browser_selftest.ps1`），最近一次全绿：

```
[PASS] three loaded: r160
[PASS] WebGL context: WebGL 2.0 (OpenGL ES 3.0 Chromium)
[PASS] manifest: 5 sprite sequences, 65 weapon models, icon map weapon_icons/map.json
[PASS] weapon icon decoded: weapon_ak47_png.png -> 512x384
[PASS] sprite smokeloop_i_0_sc_hardedge: 128 frames, decoded probes 0,64,127 (120x113)
[PASS] sprite flames: 32 frames, decoded probes 0,16,31 (76x121)
[PASS] sprite flame_omni: 60 frames, decoded probes 0,30,59 (110x127)
[PASS] sprite flame_wisp: 48 frames, decoded probes 0,24,47 (196x168)
[PASS] weapon_rif_ak47: 4 meshes, 39,649 tris, 4/4 textured, bbox 0.08x0.28x0.96 m
[PASS] 6/6 个投掷物模型可直接用于飞行中的那一颗
[PASS] 渲染 smokegrenade（2.5×，长轴对齐 (1,0.6,0)）：6,464 个非背景像素 (10.1%)
[PASS] rendered weapon_rif_ak47: 1,400 non-background px (2.2%)
[PASS] uncaught errors / rejections: 0
ALL CHECKS PASSED
```

> ⚠️ **上面这张表以前写的是"武器模型出画面 ⚠️ 未取到"，那是错的。**
> 当时归因于"虚拟时钟掐断异步加载 + 沙箱禁止 Chrome 启动"，前者只是第一层原因，
> 后者是环境策略问题（本次已放开）。真正让结果永远是 `running…` 的是
> **回传时机**：`selftest.html` 在 `load` 事件里回传，而 `load` 在带
> top-level await 的 module settle **之前**就触发了 —— 回传的是初始文本。
> 修法见 §9.7 bug 14。教训：**"工具链不行"这个结论也要有证据**，
> 当时那个 15 字节的结果文件里就写着 `running…`，我没读它。

> **顺带发现（不是 bug）**：有若干贴图是 **1×1 占位图**
> （`default_normal_tga`、`default_ao_tga`、`sticker_gaps_vmat_g_tcolor`）。
> 源文件里就是 **84 字节的 1×1 PNG** —— Valve 故意放的，让材质回退到纯色。
> 我最初的校验脚本把它误报成失败，已修正判定。

### 9.6 新增/改动的文件

| 文件 | 作用 |
|---|---|
| `visualizer/static/js/cs2-assets.js` | **新**：manifest / 图标 / sprite 序列加载与缓存；`weaponIconUrl()` |
| `visualizer/static/js/weapon-index.js` | **新**：显示名→模型的纯函数解析（别名表 / 类别判定），node 可测 |
| `visualizer/static/js/weapon-models.js` | **新**：武器 GLB 加载、挂到玩家、卡片展示 |
| `visualizer/static/selftest.html` | **新**：浏览器资产自测页 |
| `visualizer/tests/run_browser_selftest.ps1` | **新**：真实时间驱动 headless Chrome |
| `visualizer/tests/check_imports.mjs` | **新**：跨模块导出校验（抓到 2 个真 bug，见 §9.7） |
| `visualizer/tests/check_syntax.mjs` | **新**：ESM 语法检查 |
| `visualizer/tests/check_weapon_coverage.mjs` | **新**：demo 武器 → 模型 覆盖率 + 类别姿态断言（§9.4.4） |
| `visualizer/tests/check_nade_models.mjs` | **新**：投掷物类名 → 模型 + 长轴断言（§9.4.5） |
| `visualizer/tests/check_dom_ids.mjs` | **新**：`main.js` 里 `$('id')` 都必须在 index.html 里存在（§9.7 bug 13） |
| `visualizer/tests/run_all.ps1` | **新**：一次跑完所有检查，把各自的 flag 编在里面（§9.7 bug 10） |
| `visualizer/tests/fixtures/demo_weapons.json` | **新**：demo 用过的武器显示名（元数据，**入库**） |
| `visualizer/tests/test_weapon_textures.mjs` | **新**：Node 里验武器贴图 |
| `tools/cs2_assets/install_web_assets.py` | **新**：资产归位 + 生成 manifest（改用精确索引，§9.4.4） |
| `tools/cs2_assets/optimize_weapon_glb.py` | **新**：贴图压缩 + 改写 GLB 的 uri；多目录 + 幂等 |
| `tools/cs2_assets/weapon_orientation.py` | **新**：量武器朝向（两个指标 + 对照表） |
| `tools/cs2_assets/verify_weapon_textures.py` | **新**：逐张校验贴图 |
| `tools/cs2_assets/fire_tint_preview.py` | **新**：按 three 的 shader 数学合成火焰三层 + 出预览图 |
| `notes/scratch/probe_server.py` | **新**：探服务器上的资产可达性与 MIME |
| `notes/scratch/grenade_type_probe.py` | **新**：读出 demo 里真实的投掷物类名与采样点数（§9.4.5） |
| `notes/scratch/molotov_team_probe.py` | **新**：把火焰实例和引爆时刻最近的投掷物配对，验"队伍→模型"（§9.4.5） |
| `notes/scratch/molotov_exception_probe.py` | **新**：把那 4 个不吻合的案例逐条挖开（§9.4.5） |
| `visualizer/static/css/style.css` | 改：击杀信息武器图标样式（定高不定宽） |
| `visualizer/static/js/weapon-index.js` | 改：加 `NADE_PROJECTILE_MODEL` / `nadeModelKey()` / `byShortKey` 索引 |
| `visualizer/static/js/weapon-models.js` | 改：抽出 manifest 加载；加 `loadNadeModel()` / `preloadNadeModels()` |
| `visualizer/static/js/visuals.js` | 改：图标 badge、烟雾 sprite 层、火焰三层 + tint + shader 预热、**投掷物真实模型**、开关 |
| `visualizer/static/js/main.js` | 改：接线 5 个新开关 + 预热序列 + shader 预热 + 击杀信息图标 |
| `visualizer/server.py` | 改：补 `.webp`/`.glb` 的 MIME；加自测回传端点；**`TEMPLATES_AUTO_RELOAD`**（§9.7 bug 13） |
| `visualizer/static/selftest.html` | 改：加投掷物模型段；**修回传时机**（§9.7 bug 14） |
| `visualizer/tests/run_browser_selftest.ps1` | 改：超时时打印"卡在哪一行"的部分结果 |

### 9.7 实现过程中抓到的真 bug

都属同一类：**不报错、只是静默失效**，看代码看不出来。

**第一轮：**

1. **agent 贴图被搬到别处**（§7.1 坑 A）—— GLB 移走但贴图留在 `_raw/`，
   实测同目录命中 **0/24**，浏览器会得到无贴图灰模。已修，修复后 **24/24**。
2. **`import { X } from './visuals.js'` 但 visuals.js 没导出 X** ——
   `preloadSpriteSequences` 和 `assetDebugStats` 两处。语法检查**抓不到**，
   浏览器里是硬报错、整页白屏。已写 `check_imports.mjs` 静态校验，
   它立刻抓出了第二个。
3. **`.gitignore` 的 `examples/` 未锚定**（§7.2 坑 C）—— 连带忽略
   `visualizer/static/three/examples/`，新加的 GLTFLoader **不会入库**，
   全新 clone 后 importmap 的 `three/addons/` 会 **404**。已改成锚定的 `/examples/`。

**第二轮（用户试用后反馈暴露的）：**

4. **`weapon-models.js` 的"模糊匹配"连 `Knife` 都命中不了。**
   拿图标 key 去 `endsWith` 模型目录名。用现在完整的 65 把模型跑旧逻辑，
   仍有 **8 个显示名解析不出来**：`Knife`、`M4A4`、`M9 Bayonet`、
   `Incendiary Grenade`、`Glock-18`、`Navaja Knife`、`Bowie Knife`、`Talon Knife`。
   用户看到的"刀不显示"就是这个。已换成显式别名表（§9.4.4）。
5. **`install_web_assets.py` 用 `w in p.stem` 子串匹配找 GLB，还取 `rglob` 第一个。**
   键 `m4a1` 是 `weapon_rif_m4a1_silencer` 的子串 —— **匹配到哪把取决于
   `rglob` 的顺序**。同时它把索引路径又拼了一层 `weapons/`，
   导致 65 把全部报 "not in the extract index"。已改为读提取器写的精确索引。
6. **`fetch(manifest.json, {cache: 'force-cache'})`。**
   这个模式会连**已过期**的缓存也直接用、不做任何重新校验 ——
   重新提取资产后刷新页面**仍然拿到旧的 manifest**，
   表现是"改了代码/重跑了提取，但界面毫无变化"，极难排查。
   已改成 `no-cache`（仍走缓存，但每次带 ETag 校验，Flask 对没变的文件回 304）。
7. **逐帧 `material.needsUpdate = true`**（火焰 + 烟雾翻帧）——
   55 次/秒·每个燃烧实体，每次都重走 program 判定。已去掉（§9.3.2）。
8. **`install_web_assets.py` 的 `mpath.relative_to(ROOT)` 在 `--dest` 是相对路径时崩。**
   只是打印一行路径，却让整个 staging 挂掉。已加 `rel_to()` 兜底。
9. **`check_syntax.mjs` 漏 flag 时报 11 个假失败。** 它要
   `--experimental-vm-modules`，忘了就每个文件都报
   `vm.SourceTextModule is not a constructor` + `0 ok, 11 failed` ——
   读起来像**全仓库语法崩了**，实际只是少个命令行参数。我自己就踩了一次。
   现在脚本自己检测并打印正确的命令（exit 2），另外加了
   `visualizer/tests/run_all.ps1` 把 flag 编进去，从根上不会再漏。
10. **`CLASS_ORIENT[cls] || CLASS_ORIENT.rifle` 这个静默兜底。**
    类别没登记时**不报错**、直接按步枪姿态挂 —— 这正是"刀竖着插在手里"的成因。
    现在姿态表移到纯模块 `weapon-index.js`（node 可测），
    测试断言**每个出现的类别都有姿态条目**。

**差点犯的第 11 个**：`mp9` / `bizon` 的两个朝向指标都说枪口在 −z，
和其余 40 多把枪相反。如果按"两个指标都同意就照做"，
这两把枪会被改成**反的** —— 实际是折叠枪托的细金属杆骗了两个指标（§9.4.2）。

**差点写错的第 12 个**（文档层面）：我一开始用"顶点在 Y 中心以上的比例"
论证"+Y 朝上"，写的是 AK 92%、M249 57%。但那个量的是**原点位置**：
M249 实际是 **13%**（原点靠上），照那个读会被判成倒着的。
改用 **tall 轴**（枪比枪宽高）之后，34/35 把枪一致、最小比值 1.30×，结论才站得住。

**第三轮（加投掷物模型时）：**

13. **Flask 的 Jinja 在非 debug 模式下把模板缓存在进程内存里 —— 改完 index.html
    不重启服务器就一直在发旧的。** 这次新加了 `toggle-nade-models`，main.js 里绑了它，
    而服务器还在发旧模板。`bindEvents()` **不在 try 里**，所以
    `$('toggle-nade-models').addEventListener` 会抛 TypeError，`init()` 当场中止：
    3D 场景不建、玩家不出现、时间轴不动。浏览器控制台只有一行
    `Cannot read properties of null`。
    `server.py` 里本来给 `/` 加了 `Cache-Control: no-cache` 并写着"改了 index.html
    之后刷新拿到旧页面……就会出现 `$('toggle-xxx') is null`" ——
    但它只解决了**浏览器**那一半，**服务器**那一半没解决。
    已加 `app.config["TEMPLATES_AUTO_RELOAD"] = True`，并新增
    `check_dom_ids.mjs` 断言"main.js 引用的 58 个 id 在 index.html 里都存在、
    15 个开关全部有绑定"。
    **这个 bug 是"验证服务真正发出的 HTML"这一步抓到的**（`GET /` 里 grep id），
    不是读代码读出来的。
14. **`selftest.html` 在 `load` 事件里回传结果 —— 而 `load` 在带 top-level await
    的 module settle 之前就触发了。** 于是回传的永远是初始的 `running…`
    （结果文件正好 15 字节）。症状看起来像"页面没加载 / POST 被挡"，
    实际上测试跑得好好的，只是**结果被回传早了**。
    这个假象骗过了上一轮：当时把它归因成"虚拟时钟掐断 + 沙箱禁止 Chrome"，
    写进了 §9.5 和 §10.1，并因此给武器模型那一步盖了"⚠️ 未取到"的章。
    已改成 module 体自己调 `window.__selftestPost(true)`，`load` 只发
    `done:false` 心跳（给"module 压根没加载起来"留现场），并用 `__selftestDone`
    防止覆盖；`run_browser_selftest.ps1` 超时时会打印最后停在哪一行。
    **教训：说"工具链不行"之前先读一读那个结果文件里到底写了什么。**

---

## 10. 明确的不确定性

1. ~~**没有在浏览器里真加载过 GLB。**~~ **已解决。**
   浏览器自测（`selftest.html` + `run_browser_selftest.ps1`）现在完整跑通：
   WebGL 2.0 可用、官方图标解码、四条 sprite 序列全部解码、
   **AK 4 网格 39,649 面 4/4 贴图解码**、**6/6 投掷物模型可用**、
   并且**真的渲染出来数了像素**（投掷物 6,464 px / 10.1%，AK 1,400 px / 2.2%），
   `uncaught errors / rejections: 0`。
   上一版把这一步判成"未取到"并归因于工具链，是**误判**，见 §9.7 bug 14。
   **仍然缺的**：没有截图给人看（本机模型读不了图），也没有在主应用页面
   （`/`，含地图 + 玩家 + 全部图层）里跑过端到端截图。
   投掷物模型那一格是本次新打的勾（§9.4.5）。
2. **`vcsFileVersion 72` 不兼容**只做了影响面推断（§2.1）。实际金属度/粗糙度
   缺失对观感的影响没有量化。
3. **`.vsnap` 的文件格式语义**没有深入解析 —— 只知道它们和 `C_Inferno` 的
   handle 一一对应、体积很小（0.8–28.7 KB）。**怎么把 vsnap 里的点读出来**
   没有验证（可能需要 VRF 的 GUI 或另写解析）。

4. **烟雾原始 sheet 的图集布局**没确认。VRF 只给出**拆好的逐帧 PNG + `.mks`
   序列清单**（21 个序列，帧数 12–128），原始 atlas 的行列排布未知。
   如果要做 atlas 以减少 HTTP 请求，需要自己按 `.mks` 拼回去 ——
   但**帧尺寸不等大**（随扩散递增），拼接不是简单网格，可能需要 padding。
5. **agent 阵营清单是「代表模型」而非全量**：每个阵营我挑了一个 variant
   （如 `ctm_swat_variantk`）。demo 里到底用了哪个 variant **没有确认** ——
   `CCSPlayerPawn.CBodyComponentBaseAnimGraph.m_hModel` 给的是**资源哈希**
   （实测两个值：`16117112829760262417` / `14379379687539722497`），不是路径。
   要精确对应需要再做一次哈希→路径的映射。
6. **只对 2 个 agent 做了完整裁剪 + 校验**（`ctm_sas`、`ctm_fbi`）。
   其它 9 个阵营没有逐个验证（但流水线是参数化的，`--agent-all` 可直接跑）。
7. ~~Pillow 未装成功~~ **已解决**：Pillow 12.3.0 装好，WebP 编码实测可用，
   编码策略与失真测量见 §5.5.1。
8. **法线失真只测了 1 张贴图**（`glove_hardknuckle_normal`，512²）。
   结论（WebP 有损不可用、JPEG q95 可用）方向明确，但不同材质的敏感度
   没有逐个量化。**另外 ORM 贴图的有损失真没有单独测** ——
   我把 ORM 也归到「无损」是保守选择，没有实测依据说明有损一定不行。
9. ~~**WebP 解码侧未验证**~~ **已解决（浏览器侧）**：headless Chrome 自测证实
   浏览器能解码这些 WebP/PNG（官方图标 512×384、四条 sprite 序列、
   AK 的 4/4 贴图、6/6 投掷物贴图全部解码成功，见 §9.5）。
   法线贴图的色彩空间：glTF 的 `normalTexture` 语义上必须是**线性**的，
   `cs2-assets.js` 里只对颜色图设 `SRGBColorSpace`、法线/ORM 保持默认（线性），
   代码是对的。但**法线朝向上的观感仍没有肉眼确认**，
   所以这条只降到「代码正确、能解码、未肉眼确认」。
10. **烟雾 sprite 的翻帧速率是估的。** `explosion_smokegrenade_voxel.vpcf` 里
   **没有帧率字段**（搜过 `m_flSequenceDuration` / `m_flFPS` / `m_flPlaybackRate`
   都没有）。现在取 **15 fps**（128 帧 ≈ 8.5 秒一轮，和烟雾 18 秒存活期大致匹配），
   可用 `setSmokeSpriteFPS()` 实时调。**这个值没有实测依据**，
   要定准得对着游戏录像数帧。
11. **火焰三层的相对尺寸和不透明度也是估的**（core 0.55 / filler 0.85 / edge 1.20）。
   三张序列是真实的，但**叠放比例没有从游戏里量出来** ——
   vpcf 里那些渲染器参数（`m_flDiffuseAmount` 等）没有逐层解析。
12. **武器 GLB 的挂点偏移是手调的**（`CLASS_ORIENT` 里的 `off`），
   依据是「手在 x=±0.30、y=1.13」这个玩家模型常量，不是游戏里的真实
   attachment 骨骼位置。要精确得从 `.vmdl_c` 里读 attachment 表。
   **武器从 6 把扩到 65 把之后这条更突出了**：尺寸跨度从 0.21 m（手枪）
   到 1.39 m（AWP），同一套偏移不可能对每把都最优。
13. **三把弯刀/异形刀的朝向没有定论**：`knife_karambit`、`knife_talon`、
   `knife_push`。两个朝向指标互相矛盾 —— 横截面说"细端在 −y"，
   离质心最远顶点说"尖端在 +z"。原因是它们**不是直的**：
   karambit 的 z 范围 0.195 和 y 范围 0.249 差不多大（其它刀 z 只有 0.058–0.108），
   刀刃是弯的，横截面启发式会指到指环那一端。
   现在统一按 `rot = −90°`（和其余 20 把直刀一致），
   **但 Karambit 在 demo 里确实会被用到**，需要肉眼确认。
14. **`mp9` / `bizon` 的朝向是"推断"而非"实测"**。两个指标都说枪口在 −z，
   和其余 40 多把枪相反；查出来是**折叠枪托的细金属杆**
   （bin 1–3 横截面只有 0.000–0.020 m）骗了两个指标，
   所以判定枪口仍在 +z（§9.4.2）。**这个判定基于对枪托形状的解释，
   不是直接测量**，要 100% 确定得看渲染图。
15. **投掷物（`hegrenade` / `smokegrenade` / `molotov` / `incendiarygrenade`）
   的朝向无法从几何判定** —— 它们近似旋转对称，两个指标互相矛盾
   （横截面 +y / 尖端 −y）。目前按 `grenade` 类转 −90°，
   对球状物视觉影响很小，但严格说没定论。
16. **火焰 tint 的绝对值仍然依赖 ACES + exposure 0.95 这个组合**。
   §9.3.1 的数值是在 three 的 shader 数学下量出来的，
   但**如果以后改了 `toneMappingExposure` 或换成别的 tone mapping，
   这些 tint 需要重新校准**（`fire_tint_preview.py` 里 exposure 是常量，
   要和 `scene.js` 保持一致）。另外"橙色相 33.5°"是量出来的，
   "这个橙色好不好看"仍然是主观判断 —— 预览图在
   `notes/scratch/fire_final.png`，可以直接看。
17. **飞行中的投掷物朝向是"从轨迹推的"，不是原始数据。**
   `extract.py` 的 `_grenades()` 只存 `x/y/z/t/ty/eid/th`，**没有角度**。
   现在把模型长轴对齐到该段速度方向（§9.4.5）—— 抛物线飞行时看着自然，
   但真实投掷物是**翻滚**的，翻滚相位无从得知。要拿到真值得给
   `parse_grenades(extra=[...])` 加角度属性**并重新提取全部 .csz**。
18. **莫洛托夫 vs 燃烧弹有 ~1.4% 的已知误判。** 两者共用
   `CMolotovProjectile` 类名，只能按投掷者队伍区分；实测 139 个火焰里
   135 个吻合、4 个不吻合（2 个配对误差 + 2 个真例外，原因是**可以捡起
   地上的投掷物再丢**）。要完全准确得拿投掷物落点去和
   `fire[].inferno_type`（游戏自己的真值）配对，代价是耦合火焰数据（§9.4.5）。
19. **投掷物模型放大 2.5 倍是刻意的**（`GRENADE_MODEL_SCALE`）。
   真实烟雾弹只有 0.118 m，整图视距下约 2 像素；放大后 0.30 m，
   仍然**比改造前那个直径 0.4 m 的球更接近真实尺寸**。
   但这个倍数是**主观选的**（对着"能看清"调的），不是量出来的 ——
   想要严格 1:1 改成 1。

---

## 附：本次产物位置

| 路径 | 内容 |
|---|---|
| `notes/scratch/asset_inventory/` | VPK 全量清单 + 分类统计（`SUMMARY.md`、24 个 `list_*.txt`） |
| `notes/scratch/asset_probe/` | 各类导出实测产物（agent GLB、裁剪 GLB、武器、图标、烟雾帧、粒子定义） |
| `notes/scratch/asset_probe/pipeline_out/` | 流水线端到端产物（`ctm_fbi.glb` + 68 张图标） |
| **`tools/cs2_assets/`** | **16 个可复用脚本（见 §7、§9.6）** |
| `tools/Source2Viewer-CLI/` | VRF CLI 20.0.6980（Windows x64） |

⚠️ **许可**：这些都是 Valve 的版权资产。按
[Steam Subscriber Agreement](https://store.steampowered.com/subscriber_agreement/)
**仅限个人非商业使用**；**不要公开托管、不要分发**。
`notes/research-assets.md` §6 和 `research-fidelity-ai.md` §2.6 有完整讨论。

**这个仓库对上面那条规则做了一个刻意的例外**：归位后的
`visualizer/static/assets/` **入库**（理由和代价见 §9.0.1），因为全新 clone
重建不了它。原始的提取产物仍然全部忽略：

```gitignore
# 锚定到仓库根 —— 不锚定的 "examples/" 会连带忽略
# visualizer/static/three/examples/（见 §9.7 bug 3）
/examples/
# 同理：不锚定的 "assets/" 会吞掉 visualizer/static/assets/（那个是入库的）
/assets/
tools/Source2Viewer-CLI/   # 108 MB 的第三方可执行文件
notes/scratch/asset_probe/ # 实测产物（现在 2 GB+）
# 这里原本还有一条未锚定的 "*.glb"，已删 —— 见 §9.0.1 末尾
```

`git check-ignore` 是在**改成"资产入库"之前**逐条验证的。这次的两处改动（资产入库
+ 删掉 `*.glb`）**都没有重跑验证** —— 本会话跑不了命令，上面那些结论是读代码和
glob 数出来的，不是跑出来的。改完请自己确认一遍：

```powershell
# 应当没有任何输出（= 没被忽略）
git check-ignore -v visualizer/static/assets/weapons/ak47/weapon_rif_ak47.glb
# 应当正好 65 行：未被忽略的 GLB 就是 staged 的那 65 个武器模型
git ls-files --others --exclude-standard | Select-String '\.glb$' | Measure-Object -Line
# 应当约 903 行
git status --short visualizer/static/assets | Measure-Object -Line
```

**一个例外**：`visualizer/tests/fixtures/demo_weapons.json` **要入库** ——
它只是"这份 demo 用过哪些武器显示名"的元数据（36 行、744 字节），
不含任何游戏资产。`check_weapon_coverage.mjs` 靠它断言覆盖率，
放在 gitignore 的目录里会让全新 clone 上的这个测试**静默失效**。
