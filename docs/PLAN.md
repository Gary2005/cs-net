# CS-NET v5 方案：真实游戏内容 → AI 训练

> 版本 v4（深化版）· 覆盖**两种训练范式** · 只出方案，不含实现
> 本机环境：macOS arm64 · conda `cs-net-v5` · demoparser2 0.42.0 · Rust 1.86 · dotnet 10.0.203
> 证据分级：**[V]** Valve 一手产物 · **[M]** 本机实测 · **[C]** 社区测量 · **[I]** 推断

---

## 0. 决策摘要（TL;DR）

| 结论 | 说明 |
|---|---|
| **"真实"要先分两种范式** | 状态类模型看不到模型 → 真实人物模型**无用**；像素类模型 → 真实人物模型**必需**。二者方案完全不同 |
| **本机拿不到真引擎画面** | CS2 **没有 macOS 版**（三重证据）。所有本机 in-engine 路线出局 |
| **但本机可以拿到真资产** 🎯 | CS2 的 VPK 是分块的：索引 `pak01_dir.vpk` 仅 **7.6 MB**，48.6 GB 分散在 **495 个 24–120 MB 的块**里。用 DepotDownloader `-manifest-only` + `-filelist` **只下需要的块**（+ 单张地图 VPK ~262 MB）→ **装得进本机 29 GB 空闲** |
| **真实火焰格子：已解决**  | 换用 **demoinfocs-golang（MIT，零补丁）** → 火焰 **16/16 格真实 3D 坐标 + 每格表面法向 + 蔓延树**。⚠️ **火焰不是平面**（实测垂直跨度可达 124 HU、37.5% 明显倾斜），3D 栅格要用这 4 条数组建，**不能只取 `ConvexHull2D` 的二维轮廓**。**不需要改解析器、不需要装游戏** |
| **真实烟雾体积：已解决** 🎯 | `m_VoxelFrameData` 是 **Morton journal**，不是栅格。格式已由 `CS2-insight-agent` 公开，并用本机数据独立验证：802/802 字节、44 种子 + 44 掩码、**1119 体素**、3D 半径 **155.9 HU**。栅格 32³ / 体素 12 HU / 边界洪泛填充 |
| **真实逐 tick 人类输入：其实存在** 🎯 | 我此前判「demo 无输入」是**层次错误**：`svc_UserCmds` 是**网络消息层**（protobuf），不在实体属性表里。用 Rust 可解出 **13 个按键位掩码** → BC 可用**真实动作标签**（§7.5） |
| **没人做过这件事** | **不存在**任何开源项目把「CS2 + 真模型 + 体积烟雾」渲染出来。你们填的是真实空白 |
| **"保真"的关键不是资产，是着色器** | Skinshotter 证明：浏览器端保真来自把 Source 2 着色器移植到 WGSL + 真实光照数据，**而不是**加载一个 GLB |

---

## 1. 问题定义：两种范式，两套方案

| | **范式 A：状态类**（cs-net 现在就是） | **范式 B：像素类**（视觉 / 世界模型 / RL） |
|---|---|---|
| 模型输入 | 结构化 state | 渲染画面 |
| "真实游戏内容"的作用 | 真实烟/火的**几何与机制** → 更强的输入特征 | 渲染保真度 = **sim-to-real gap** |
| 真实人物模型 | **完全无用** | **必需** |
| 地图真实贴图 | 无用（只要几何做射线） | 必需（或至少要有真实的视觉先验） |
| 需要层级 | **L0 + L1** | **L2 或 L3** |
| 本机可行性 | ✅ 完全可做 | ⚠️ L2 勉强可做，L3 不可做 |

**这条分界线是整个方案的支点。** 下面 §5 给范式 A 的方案，§6 给范式 B 的方案。

---

## 2. 现状体检：cs-net 现在错在哪

### 2.1 渲染层
| 元素 | 现在 | 位置 |
|---|---|---|
| 玩家 | 圆柱 + 球头 + 圆锥 | `visualizer/static/js/visuals.js:16-18` |
| 烟雾 | **一个**静态半透明球，半径硬编码 `3.2` | `visuals.js:232, 873-882` |
| 火焰 | **一个**贴地发光圆盘 `CircleGeometry(2.6)` | `visuals.js:236, 946` |
| 地图 | Open3D 导出**无材质灰模**（`grep -c usemtl` = 0） | `maps/optimized_obj_files/*.obj` |

### 2.2 数据层（根因）
`demo_parser/extract.py` → `_build_projectile_intervals()` 把烟/火压成 `{ts, te, x, y, z}` 一个点：
```python
"infernos": [{"ts": 11.22, "te": 18.23, "x": 1133.69, "y": -47.42, "z": 273.0}]
```

### 2.3 模型层（**对 AI 训练最致命**）
`scripts/training_data/feature_builder.py:395-424` 把活跃烟雾/火焰各编码成**单个 token**：
```python
N_PLAYERS = 10
N_MAX_PROJECTILES = 16        # 飞行物 + 烟 + 火 共享 16 个槽位
N_TOKENS = 10 + 1 + 16 = 27   # config.py:239-241
```
模型因此**完全不知道**：谁在烟里、哪条枪线被烟挡住、火封死了哪些路径、站进火里会掉多少血。真实一回合的投掷物**必然撑爆这 16 个槽位**（代码里为此写了 `overflow_count` 告警 + 优先级淘汰）。

---

## 3. 四层保真度

| 层级 | 内容 | 获取方式 | 本机 |
|---|---|---|---|
| **L0 真数据** | 逐 tick 状态、53 类事件、投掷物全轨迹 | demoparser2 | ✅ |
| **L1 真几何/机制** | 烟雾**逐体素占据场**、火焰**16 格 footprint**、命中盒、弹道物理 | demoinfocs-golang（**已实测拿到**）+ 实测机制参数 | ✅ |
| **L2 真资产渲染** | 真模型/贴图/真实材质地图 | DepotDownloader 选择性下载 + Source2Viewer-CLI | ✅ **（关键突破）** |
| **L3 真引擎** | CS2 本体 | Windows/Linux GPU 云 | ❌ 本机不可行 |

---

## 4. 真实游戏内容技术底账（全部数值有出处）

### 4.1 单位 / 坐标 / 相机
| 项 | 值 |
|---|---|
| Hammer unit（**地图**比例） | 1 HU = 0.75 in = **1.905 cm**（16 HU/ft）**[V]** |
| Hammer unit（**角色建模**比例） | 1 HU = 1 in = 2.54 cm（12 HU/ft）**[V]** |
| 朝向 | **+X 为 forward**，yaw 0 = +X，`forward = (cos yaw, sin yaw, 0)`；**+Z up，+Y 是左** **[V]** |
| 玩家原点 | 在**脚底** |
| FOV | **垂直 73.7398°**（= 4:3 下 90° 水平；16:9 下 106.26° 水平）。**没有** spectator/demo FOV cvar **[V]** |
| 色调映射 | Hable：.15/.50/.10/.20/.02/.30，white 4.0；gamma 2.2 **[V]** |
| AO / 雾 | AO 半径 30 HU、强度 1.2；雾 0 → 4000 HU，指数 2.0 **[V]** |
| 阴影 | 3 级 CSM，75° 阴影视锥，7500 HU 范围；每张地图 sun 亮度 2.4–6.35（带精确 RGB）**[V]** |
| tickrate | 官方 128；本 demo 实测 **64** —— **必须从 demo 里读，不能假定** **[M]** |

> ⚠️ **给渲染器的建议**：**直接以 HU 1:1 渲染**，只在 UI 层做单位换算。cs-net 现在用 `SCALE = 0.0254`（把 HU 当 inch）统一缩放 —— 数值上自洽，但会让上面这些游戏常数无法直接使用，还会把 FOV/雾/光照的调参变成"半猜"。

### 4.2 烟雾（真实参数，推翻我上一版的"球体"说法）
| 项 | 值 |
|---|---|
| **形状** | **被地面裁切的透镜/环体（lens/torus），不是球** |
| 环半径 / 子体半径 / 原点高度 / 边缘羽化 | `61` / `88` / **`68 HU`** / `21 HU`（cvar 硬编码，无动态调整） |
| 水平半径 | **≈149 HU**（直径 ≈298 HU ≈ **5.68 m**） |
| 云顶高度 | **≈156 HU（2.97 m）** |
| 引爆 | 引信**落地后才开始**；bloom **不到 1 秒**到满尺寸 |
| 持续 | **18 s** 遮挡，≈20 s 完全消失，**15 s** 雷达遮挡 |
| 子弹洞 | 1–2 s 自愈（霰弹更大）；HE 会开洞 2–3 s 后重新合拢 |
| Bot LOS 穿透 | 200 HU |
| 渲染方式 | **体积光线步进**：shader `smoke_volume{,_cs,_depth,_lpv,_mask}` + MBOIT |
| 关键粒子/材质 | `explosion_smokegrenade_voxel.vpcf_c`、`weapon_smokegrenade_pointcloud.vsnap_c`、`materials/particle/particle_smokegrenade.vmat_c` |
| **烟是分阵营染色的** | T = `[180,129,50]`（暖棕），CT = `[75,127,155]`（冷蓝），随烟龄褪成灰 |
| 剪影可见阈值 | `sv_smoke_volume_blind_start = 0.2` |
| **实测存活时长** | 中位 **1412 ticks = 22.06 s**（127/137 配对；10 个被回合结束截断）**[M]** |
| ✅ **交叉验证** | 本机从 demo 测出的 `m_vSmokeColor` = **`[180,129,50]`**，与 Valve cvar 的 T 方默认值**逐位吻合** —— 数据管线与参数表互相印证 **[M]+[V]** |
| **voxel 数据** | ⚠️ 早先说"读出来全是 0、必须重新模拟"——**这对 demoparser2 成立，但已被推翻**：换 demoinfocs-golang 后 **`m_VoxelFrameData` 拿到 3072 个元素、454 非零**（见 §7.2）。同时 `m_nVoxelFrameDataSize` 单调增长（722 → 2521，跨 1412 ticks）可作 bloom 曲线 **[M]** |

### 4.3 火焰（真实参数）
| 项 | Molotov | Incendiary |
|---|---|---|
| 扩散半径 | **150 HU**（2.86 m） | **110 HU**（2.10 m） |
| **火焰存活** | **7 s** | **5.5 s** |
| 伤害 | 40 DPS，每 **0.2 s** 一跳（⇒ **8 伤害/跳**），首跳 1→8 递增，**100% 破甲**，**可叠加** | 同 |
| 最大格子数 | **16**（`inferno_max_flames`）；数组长度 **64** | 同 |
| **格子间距** | **42 HU** | 同 |
| 初始喷发 | 每 **0.02 s** 生一格 | 自 2025-07-16 传播更快 |
| 子格扩散 | 深度 4，间隔 ×0.1/层，上限 0.5 s，窗口 3 s，生成角 45°，离面 15 HU | 同 |
| 弹道 | 落点坡度 ≤30° 碎裂，否则弹跳；**2 s 空爆**；离面 **<128 HU → 火焰落地** | 同 |
| 渲染 | 每格 `particles/inferno_fx/fire_{core,filler,edge}.vsnap_c` + `shaders/vfx/inferno`；焦痕 `materials/decals/molotovscorch.vmat_c` |
| **火焰精灵几何** | `InfernoFire_HalfWidth = 30 HU`、`InfernoFire_FullHeight = 80 HU`；每朵火焰随机 `m_maxSize` **70–90 HU**，绘制四边形 = 半宽 `size/3`、高 `size` ⇒ **≈80 HU（2.0 m）高、≈40 HU 宽的 billboard，落在 ≈60 HU 的地面块上** |
| **伤害起算点** | 从**点燃时刻**起算（不是从玩家踏入）：40 × 0.2 = 8 / 每 0.2 s，按 `m_damageRampTimer` 在**点燃后 2.0 s** 内爬坡；**`DMG_BURN` 完全无视护甲** |
| **🔥 Fire Sale 是分阵营 cvar** | `inferno_spread_speed_mult = 1`（T）vs **`inferno_spread_speed_mult_ct = 10`**（CT）⇒ **燃烧弹扩散快 10 倍**，但只到 **110 HU**（vs 150 HU）、烧 **5.5 s**（vs 7 s） |
| **实测存活** | **449–450 ticks @64 = 7.031 s** | **352 ticks @64 = 5.500 s** **[M]** | 同 |
| **实测存活** | **449–450 ticks @64 = 7.031 s** | **352 ticks @64 = 5.500 s** **[M]** |

> 实测与 Valve cvar 默认值**完全吻合**（7.03 s vs 7 s；5.50 s vs 5.5 s）—— 这同时验证了数据管线和参数表的正确性。
>
> **游戏性上真正重要的不是火焰粒子，而是"哪块地面不能走"** → 应该建模 `C_Inferno.m_firePositions` 的 footprint，粒子只是装饰。

### 4.4 玩家（真实参数）
| 项 | 值 |
|---|---|
| 站立碰撞盒 | **32 × 32 × 72 HU**，原点在脚底 |
| 站立眼高 | **64.0938 HU**；蹲下 **32 × 32 × 54**，眼高 **46.0938 HU**；下蹲耗时 200 ms |
| 下蹲的精确关系 | ✅ 已实测：`m_flDuckOffset == -18 × m_flDuckAmount`（727,649 个存活采样上精确成立），碰撞盒 `m_vecMaxs.z` **只取 72 / 54**，与 `m_fFlags` bit 2 在 100.0000% 的采样上一致。所以 18 HU 的行程有两处独立佐证 |
| ⚠️ 眼高绝对值 | 站立足眼高 **demo 拿不到**（CS2 没有网络化任何视点偏移字段，`m_vecViewOffset` 不存在，demoinfocs 的 `PositionEyes()` 在 Source 2 上 panic）。上表的 64.0938 是**文档值**；demo 里唯一候选 `m_vecZ` 众数是 64.000 但空中读数是垃圾。两者差 2.4 mm。详见 `tools/demo_extract/API.md` §4.12 |
| 命中组 | HEAD 1 / CHEST 2 / STOMACH 3 / LARM 4 / RARM 5 / LLEG 6 / RLEG 7 / NECK 8 / GEAR 10 |
| 伤害倍率 | 头 ×4，腹 ×1.25，胸/臂 ×1.0，腿 ×0.75 |
| 命中盒形状 | **胶囊体**（`CHitBox.m_flShapeRadius` + `m_nShapeType`） |
| 模型路径 | `agents/models/<family>/<file>.vmdl` —— **不是 `models/players/`**（那个路径在 CS2 里不存在） |
| 默认 CT / T | `agents/models/ctm_sas/ctm_sas.vmdl` / `agents/models/tm_phoenix/tm_phoenix.vmdl` |
| 模型家族 | ctm_sas / ctm_fbi / ctm_st6 / ctm_swat / ctm_gendarmerie / ctm_diver / tm_phoenix / tm_professional / tm_balkan / tm_jungle_raider / tm_leet（+ 12 个手套） |
| 骨骼 | 共享 `worldmodel.vnmskel`，**74 根骨**，根 `root_motion` → `pelvis` |
| 动画格式 | **Animgraph 2**（`.vnmclip`/`.vnmgraph`/`.vnmskel`），**没有 Source 1 格式** |
| **动画名** | **带武器后缀**：`idle_knife`、`run_n_knife`、`crouch_n_knife`、`jump_stand_knife`、`death_chest_a`、`flinch_head`、`equipment/c4/planting`、`defuse/defuse_loop_*`。**`idle`/`run`/`death` 不存在** |

### 4.5 数据层实测（`0823_inferno.dem`，328 MB，de_inferno，24 回合）
| 指标 | 实测值 |
|---|---|
| 全量 `parse_ticks`（31 属性 × 167,846 ticks × 10 人 = 1,668,280 行） | **7.32 s**，峰值内存 2.6 GB |
| `parse_grenades()` | 0.95 s，**228 万行**逐 tick 轨迹 |
| 事件总数 | **53 类** |

**两个必须知道的坑（实测）**：
1. **tick 偏移**：投掷物实体的原始 tick 字段（`m_nSmokeEffectTickBegin` 等）与 `tick` 列/事件用的**不是同一个 tick 基准**，差一个**每 demo 恒定的偏移**（inferno demo = 794，nuke = 830，mirage = 691）。**结论：一律用事件 tick。**
2. **entityid 会复用**：**不能**用投掷物实体生命周期去配对烟（实测中位 delta 达 7587 ticks = 118 s，`position_match=false`）。必须用 `smokegrenade_detonate` ↔ `_expired` 事件配对（两者携带**相同** `x/y/z`，可按 `(entityid, x, y, z)` 配对）。

**字段级校正（与常见预期不符，均已实测）**：
| 想要 | 实际 |
|---|---|
| `is_planting` | ❌ 实体属性层**没有**这个 netprop（`list_updated_fields()` 里搜不到 `m_bIsPlanting`）。用 `bomb_beginplant` 事件，或规则实体的 `m_hBombPlanter`。⚠️ demoinfocs 反而把它做成了现成字段 `Player.IsPlanting` |
| ~~`in_crouch` / `crouch_state` / `stamina` / `move_state` / `buttons` / `WALK` / `aim_punch_angle`~~ | ✅ **❗ 这一行原先写"全部不可用"，是错的** —— 我用的是短名字，而 demoparser2 的词汇表是**全限定名**，短名字被静默丢弃。实测**全都能取到**：`CCSPlayerPawn.CCSPlayer_MovementServices.m_bInCrouch` / `m_nCrouchState` / `m_flStamina`、`CCSPlayerPawn.m_MoveType`、`CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtonDownMaskPrev`、`CCSPlayerPawn.m_bIsWalking`、`CCSPlayerPawn.m_aimPunchAngle` |
| `kills` / `deaths` / `assists` | ❌ 用 `kills_total` 等 |
| `round_start` / `round_end` / `item_purchase` | ❌ **不存在**（3 张地图确认）。用 `round_prestart` / `round_freeze_end` / `round_officially_ended` + `round_win_reason` 属性；购买用 `item_pickup`/`item_equip` |
| `bomb_planted` 的坐标 | ❌ **没有 x/y/z**。`site` 是炸弹点**实体索引**（inferno 上 312=A，403=B）→ 需用安放者在该 tick 的 X/Y/Z 推导 |
| 皮肤 | ✅ `active_weapon_skin` **直接给皮肤名**（如 `'Aphrodite'`）+ `weapon_paint_seed` + `weapon_float` |
| 弹道 | ✅ `fire_bullets`（4186 条）= **理想弹道**：origin_x/y/z + angles_x/y/z + seed + spread |
| ⚠️ 未知属性名 | **静默丢弃、不报错**。**必须断言返回列**（`parse_grenades(extra=...)` 同样） |

**推荐用法**：`parse_grenades(grenades=False)` → 31 万行，只含 5 个 `*Projectile` 类，有真实逐 tick 轨迹；`grenades=True` 会把**手持武器实体**也带进来（那些 `x/y/z` 全是 NULL）。投掷物**按 `(entity_id 相同 & steamid 相同 & tick 差 == 1)` 切分成 run**，得到 137 烟 / 126 闪 / 114 手雷 / 87 燃烧 / 20 诱饵 —— 与事件计数完全一致。

**回合/时间模型**：总 167,846 ticks（无空洞）= 43.7 分钟；`time = tick/64`；**header 里没有 tickrate 字段**，必须自己从 `game_time` 斜率推。**热身期包含在内**（`is_warmup_period=True`，`begin_new_match` 在 tick 3533）→ **回放应从这里开始**。冻结期 960 ticks = 15.0 s；第 1 与第 13 回合（换边）为 1376 ticks = 21.5 s。

> **性能的真正瓶颈是内存不是时间**（2.6–3.1 GB 峰值）→ **解析一次、缓存到 Parquet、再按帧切片**。

**可用白名单**（`parse_grenades(extra=[...])` 实测能取到）：
```
m_vSmokeDetonationPos  m_vSmokeColor  m_nSmokeEffectTickBegin  m_bDidSmokeEffect
m_nVoxelFrameDataSize  m_nVoxelUpdate                    ← 烟雾 bloom 代理信号
m_vInitialVelocity     m_vInitialPosition                ← 可重建真实弹道物理
m_nBounces             m_flThrowStrength                 ← 真实弹跳次数 / 投掷力度
m_vecExplodeEffectOrigin  m_flDetonateTime  m_nDeployTick  m_bIsLive  m_hThrower
```

---

## 5. 【方案 A】状态类：真几何 → 模型输入表示升级

> 这一条**本机立刻可做、不需要游戏文件、不额外占磁盘**，而且对现有 cs-net 模型是实打实的收益。

### 5.1 现在错在哪
一个烟雾 = 一个 token（点 + 剩余时长）；一次燃烧 = 一个 token（点 + 剩余时长）。16 个槽位还要和飞行物共享。

### 5.2 目标特征（全部来自真实机制参数）

**烟雾**（真值：水平半径 ≈149 HU、地面裁切透镜、18 s、bloom <1 s）
- **10×10 两两可见性遮挡矩阵**：玩家 i→j 连线若被烟体挡住 → 枪线被挡。**这是真实游戏机制，现在模型完全不知道。**
  - ⚠️ **当前应按"高保真重建"规划**：用 149 HU 透镜 + 真实时长/bloom 做遮挡判定。真实体素密度**能读到**（§7.2 ②），但栅格→世界映射未解，解开后可升级为逐体素精确
- 每玩家 `in_smoke`（用真实烟体几何判定）+ 烟龄 / 剩余时间
- 支持多烟叠加（真实回合常同时 2–4 颗）
- bloom 曲线用 `m_nVoxelFrameDataSize` 单调增长实测值拟合

**火焰**（真值：16 格 × 42 HU 间距、半径 150/110 HU、寿命 7/5.5 s、40 DPS / 8 每跳 / 可叠加 / 100% 破甲）
- 顶视**占据 / 威胁栅格**（fire density map）
- 每玩家 `in_fire` + 按真实伤害节奏的**预计掉血**（含叠加）
- 被封死的路径 / 区域

**投掷物**
- 真实逐 tick 轨迹 + `m_nBounces` 弹跳 + `m_vInitialVelocity` 初速 → 落点 / 闪光时长预测

**玩家**（大部分可全量取到，属低成本增益）
- `pitch`、`ducking`/`duck_amount`、持枪、背包、`is_scoped` / `is_walking` / `is_airborne` / `is_defusing`、皮肤（`active_weapon_skin` 直接给名字）
- ❗ **更正**：这一行原先写"`is_planting`/`in_crouch`/`stamina`/`move_state`/按键位/`aim_punch_angle` 拿不到"，**是错的**（短名字查不到 ≠ 数据不存在）。实测除 `is_planting` 外**全都能取到**，包括**逐 tick 的 13 键掩码**（见 §7.6）
- ⚠️ 真正没有的只有 `is_planting` 这一个 netprop —— 用 `bomb_beginplant` 事件补
- **注意：这一层完全不需要人物模型**

### 5.3 编码改造（关键设计）
```
现状   N_TOKENS = 10 玩家 + 1 炸弹 + 16 投掷物 = 27
       16 槽位装不下飞行物 + 烟 + 火 → 真实回合必然溢出

建议   保留 10 个玩家 token
       投掷物/烟/火  →  多通道顶视空间栅格（如 128×128）
                        ch0 烟密度 / ch1 火密度 / ch2 飞行物 / ch3 剩余时间 / ch4 是否可通行
       可见性        →  10×10 遮挡矩阵（attention bias 或独立编码器）
```
这不是美化，是**输入表示的升级**，会改变模型的归纳偏置。

### 5.4 收益与验证
- 收益方向：可见性推理、投掷物效用、区域控制、「进烟/进火」风险
- 验证：仓库里已有 `scripts/evaluate_pretrain.py`、`scripts/spatial_only_predictor.py`、`scripts/evaluate_demos.py` 全链路，可直接做 A/B 对比量化

---

## 6. 【方案 B】像素类：真资产 / 真引擎

### 6.1 路线 B1 —— 真资产 Web 渲染（**本机可行，关键突破**）

**🎯 突破点：CS2 的资产可以"按需下载"（✅ 本机已实测）**

`DepotDownloader`（SteamRE）本机实测输出：
```
$ ./DepotDownloader -app 730 -depot 2347770 -os linux -manifest-only -dir ./cs2out
No username given. Using anonymous account with dedicated server subscription.
Logging anonymously into Steam3... Done!
Got depot key for 2347770 result: OK
Manifest 2053759441494650084 (09/09/2026 22:30:37)      ← 清单仅 357 KB
```
**匿名登录可用，无需拥有游戏。** 清单实测内容：

| 文件 | 实测 | 说明 |
|---|---|---|
| `game/csgo/pak01_dir.vpk` | **7,584,917 B ≈ 7.6 MB** | **索引**，不是内容 |
| `game/csgo/pak01_NNN.vpk` | **494 个**，单个 **~100–119 MB** | 真正的内容分块 |
| `game/csgo/maps/de_inferno.vpk` | 存在 | 每张地图独立打包 |
| `game/core/pak01_dir.vpk` | 82,315 B | 资产分散在 core / csgo / csgo_core / csgo_imported 多套 VPK |

⚠️ **诚实边界**：已验证**匿名访问 + 索引确实只有 7.6 MB**（否定了"必须下 60 GB"）；但**尚未实跑 `-filelist` 的真实内容下载**（只拉某几个 chunk）——这是下一步的确认点。

```
# 1) 只取清单
DepotDownloader -app 730 -depot 2347770 -manifest-only -os linux -dir ./cs2_manifest
# 2) 只下索引
printf 'game/csgo/pak01_dir.vpk\n' > files.txt
DepotDownloader -app 730 -depot 2347770 -os linux -dir ./cs2 -filelist files.txt
# 3) 用索引查出目标文件在哪个块
Source2Viewer-CLI -i ./cs2/game/csgo/pak01_dir.vpk --vpk_list -f "agents/models/" -e vmdl_c
# 4) 只下那几个块 + 单张地图 VPK（~262 MB）
DepotDownloader -app 730 -depot 2347770 -os linux -dir ./cs2 -filelist chunks.txt
```

**导出命令**（已对本机 v19.2.6339 校验；注意 `--game` 参数这个 build **不认**，要把 CLI 直接指向 VPK）：
```
# 人物模型 → GLB（含骨骼/动画/材质）
Source2Viewer-CLI -i pak01_dir.vpk --vpk_filepath "agents/models/ctm_sas/ctm_sas.vmdl_c" \
  -o ctm_sas.glb -d --gltf_export_format glb --gltf_export_animations --gltf_export_materials
# 地图 → glTF
Source2Viewer-CLI -i maps/de_dust2.vpk --vpk_filepath maps/de_dust2.vmap_c -o dust2.gltf -d --gltf_export_format gltf
# 贴图 → PNG     Source2Viewer-CLI -i pak01_dir.vpk --vpk_filepath "materials/particle/x.vtex_c" -o x.png -d
```

**必须提前知道的导出损失**（VRF 官方文档）
- 只导出最高 LOD；分层混合材质只导出一层；布料 sheen 无 glTF 等价物
- **不导出 hitbox**；**不导出烘焙光照（lightmap / light probe）**
- 地图只导出 `light_environment`，点光/聚光**不导出**；不打包 3D skybox；clutter 忽略
- 已知 issue：CS2 角色模型反编译时报 **"Invalid skinning bone index"**（VRF #845）
- 导出 >2 GB 必须用 `.gltf`（分块）而非 `.glb`

**地图的两条取法**
| | 取法 | 优点 | 缺点 |
|---|---|---|---|
| **快/灰** | 解码 `world_physics.vmdl_c`（**碰撞网格**） | 几何**精确**，免费得到射线/LOS 碰撞 | 无贴图 |
| **像** | `vmap` → glTF 真实渲染网格 + 材质 | 有真实网格与材质 | 要自己重新打光（无 lightmap） |

> ✅ **已落地（2026-11）**：走了第二条路，但**没有重建几何**。`worldnodes/` 导出的是带
> `TEXCOORD_0` + `.vmat` 的原始网格（de_inferno 8,419,800 三角形，327 个材质），把它们
> 按空间回投影到现有的 3.29M 三角形减面 OBJ 上，烘成"每三角形一个颜色"
> （`tools/cs2_assets/bake_map_colors.py`，9.4 MB/张，实测中位匹配误差 4.8 mm）。
> 原图直接进浏览器要 800 MB GLB，所以减面版保留、只搬颜色。
> 细节和取舍见 `visualizer/README.md` §4.14。
>
> ⏸ **后来停用了（2026-11）**：减面碰撞几何上，同色面之间仍会抢深度、显形的差异其实在
> 逐面法线，套材质色反而带来闪烁；`map-loader.js` 现在默认 `BAKED_COLORS = false`，
> 地图回到水泥灰，功能保留在 `?mapcolors=1` 后面。要不要捡回来，先看 §4.14 开头那段。

**玩家**：VRF `vmdl` → glTF（骨骼 + 动画可导出）。**注意动画名必须用武器后缀版**（`idle_knife` 等）。

**烟雾**：⚠️ demoinfocs 能读出真实体素密度负载（~800 字节、454 非零），但**栅格→世界坐标映射未解**（§7.2 ②）。**渲染按 §4.2 参数重建**（149 HU 透镜 + 18 s + <1 s bloom + 阵营染色）并**建模 HE 灭烟**；栅格解开后再切到真实体素。

**火焰**：用 `C_Inferno.m_firePositions` 的真实格子渲染（不是粒子）。

**保真度的真相**（这条最重要）：
> **"保真"来自着色器，不是来自资产。** Skinshotter 是证据也是警告：它的保真度来自把 Valve 的涂装/光照着色器**逆向并重写成 WGSL** + 导出**真实地图光照数据**，而不是"加载了一个 GLB"。glTF 导出的材质只是 PBR 近似，VRF 会把原始 shader 名与参数放进 `vmat` extras 块 —— 那才是恢复 Source 2 语义的钩子。

### 6.2 路线 B2 —— 真引擎（本机不可行）

- **CS2 没有 macOS 版**：Steam `appdetails` 的 `platforms.mac=false`；macOS depot `2347772` 仅 **9,276 字节**；CS Demo Manager 兼容矩阵 macOS 下 CS2 video ❌ / HLAE ❌ **[M]**
- 体积：depot 2347770 = **62.74 GB**（共享内容）+ Windows 7.71 GB + 1.14 GB ≈ **71.6 GB 安装 / 59.9 GB 下载**；Valve 官方要求 **85 GB 空闲**
- **HLAE**：Windows-only DLL 注入，强制 `-insecure`；CS2 支持自 2.146.1（2023-07-14），最新 2.192.2（2026-09-12），能力齐全（`mirv_streams` 直推 FFmpeg、`mirv_campath`、`mirv_cmd` tick 级脚本）。**录制是实时的**（40 分钟比赛 ≥40 分钟 GPU）
- CS2 自带 `startmovie` 是 `FCVAR_DEVELOPMENTONLY`，需 HLAE 解隐藏 + `-insecure`；`ui_render_task_generate_clips` 无公开 API，不可自动化
- **一条值得注意的现成方案**：`reka-ai/cs2-dem-renderer` —— **启动真 CS2 + 版本匹配的 C++ 服务器插件**，重放记录的输入，**直接捕获 CS2 自己的帧**输出 HEVC MP4。这是"真引擎像素真值"最干净的工程范式（Linux），`akiver/cs-demo-manager`（MIT）是该技术的参考实现
- **唯一本机可行的替代**：CS:GO 旧版在 2026-03-04 被重新以隐藏 app **4465480** 上架（免费、仅离线/bot，元数据声明 macOS 支持，约 **34.3 GB 安装 / 14.6 GB 下载**，macOS depot 733 = 277 MB，Rosetta 2 可跑，且 CS:GO 的 `startmovie` 是普通命令）。**但 CS:GO 放不了 CS2 demo，CS2 也放不了 CS:GO demo** —— 只对 CS:GO 时代的 `.dem` 有效。⚠️ 存在矛盾：公开 store API 对 4465480 报 `mac:false`，而其运行时配置声明 macOS，值得用 `demo_info` 先确认你的 demo 是哪个引擎时代
- 🆕 **重要更正：云端的"真引擎渲染"服务确实存在，而且有 API（我上一版说"没有"是错的）**
  | 服务 | 能力 | 价格 / 限制 |
  |---|---|---|
  | **DEMO-SLAP** | **OpenAPI 3.0**：`/analyze/url` → `/render-custom`（可指定 tick 区间 + `povSteamId`）→ 签名 CDN `.mp4`；`x-api-key` 鉴权；还有 MCP connector。引擎内证据：`hideUi`/`showXRay`/`showTracers`/`hideKillfeed` 等选项，analyze 的 `source` 枚举含 `rungame-url` | **≈€0.02–0.06 / 10 秒片段**，每天 17 免费 credits。⚠️ **两人瑞士副业项目**，原型级，**不要把产品建在它上面** |
  | **CLUTCHKINGS.gg** | 免费云端 CS2/CS:GO demo→片段，TrueView/X-Ray/HUD/radar，1080p60/120 可下载 | 无 API；Steam 资料公开 + ≥25 竞技胜场 → 每周 100 条普通优先级 |
  | **Scope.gg Clips** | 服务端录制，第一人称 + 对手 POV，1080p60 可下载 | 无 API |
  | **Leetify** | 有公开 API（`api-public-docs.cs-prod.leetify.com`） | 仅统计 + 集锦片段，**无整场 MP4** |
  - 仍然成立的两点：产出是 **POV 风格片段，不是自由机位电影镜头**；**没有任何服务提供"2D 雷达回放导出 MP4"**
  - 对 AI 训练的意义：这是**便宜的"真引擎像素"来源**，可用于**验证自建渲染器的 sim-to-real 偏差**（§6.4 也提到要拿真引擎画面做校验）
- 结论：
  1. **要少量真引擎像素** → 先用 DEMO-SLAP 免费额度 / CLUTCHKINGS（零本地安装、零磁盘）
  2. **要自由机位电影镜头或大规模生成** → 租云 Windows/Linux GPU（~150 GB 磁盘，建议快照那 60 GB 安装避免重复下载；AWS `g4dn.xlarge` $0.71/hr、spot ~$0.27，Azure `NV6ads_A10_v5` $0.730/hr、spot $0.135）。⚠️ **Windows 在 RunPod/Vast.ai/Paperspace 上不可用**，Shadow PC 的 ToS 禁止服务端/商业用途，且**没有任何公开记录显示有人在云主机上无人值守地跑通 CS2 + HLAE** —— 这是主要工程风险
  3. **不要在本机尝试 CS2 渲染**（无 macOS 版、HLAE 仅 Windows、~72 GB vs 29 GB 空闲）

### 6.3 现有工具版图（重要：确认你们没在造重复的轮子）
| 项目 | 3D | 资产 | 烟/火 | 玩家 | 许可 |
|---|---|---|---|---|---|
| **programmer1o1/demoscope** | ✅ 真 3D（Rust+WASM 自包含 HTML） | **真 CS2 地图碰撞网格**（从地图 VPK 解码，de_nuke 115,738 顶点） | 不渲染 | **彩色方块** | **无 LICENSE** ⚠️ 仅供阅读参考 |
| **cs2.cam** | ✅ 付费 3D Replay | 未证实 | 2D 里文档化了真实扩散/灭烟 | HUD | 闭源商业 |
| **Skinshotter** | ✅ WebGPU 高保真 | **真资产 + 真 Source 2 光照（WGSL 重写）** | 无 | **真 agent 模型 + 手套** | 闭源，**作者可谈授权** |
| **awpy** | ❌ 2D | 官方雷达图 + `.nav` | 2D 圆 | 2D 标记 | **MIT** |
| **reka-ai/cs2-dem-renderer** | ✅ 真引擎捕获 | 真游戏资产 | 引擎真值 | 真模型 | 无 LICENSE |
| **akiver/cs-demo-manager** | ❌ 2D | 雷达图 | 2D | 2D | **MIT** |
| **Igor-Losev/deadem** | ❌ | 纯 JS/WASM 解析器（21.5k ticks/s） | — | — | **MIT** |
| **DrEAmSs59/CS2-insight-agent** | ❌ 2D（本地高速 2D 回放 + 热力图 + OBS 自动录制 + LiteCut 精剪） | 真雷达图、真实饰品目录；**0 注入 / 0 Hook / 0 逆向游戏进程** | **✅ 解出真实烟雾体素 journal**（§7.2 ②） | 2D 标记 | **PolyForm Noncommercial** ⚠️ 非商业；换肤模块**闭源专有** |
| **Rupas1k/source2-demo** | ❌ | Rust 解析器 | — | — | **Apache-2.0** |

**常被误传的两条更正**：**Leetify 没有公开证据表明有 3D 回放**（服务端解析 + 视频集锦）；**Scope.gg 是 2D**（其"3D effects"指 2D 回放里的立体风格特效）。

> **结论（更新）：仍然没有任何开源项目把「CS2 + 真提取模型 + 体积烟雾」在 3D 里渲染出来** —— 而且 `CS2-insight-agent` 的出现反而**强化了这个判断**：它是一个成熟（870★）且功能完整的项目，但它的回放是 **2D** 的。
>
> 不过它在**数据侧**替我们解决了两大难题：**真实烟雾体素 journal 的解码**（§7.2 ②）和**真实逐 tick 输入**（§7.5）。这两项都是纯数据/格式知识，**可以直接用于本方案**（注意其非商业许可）。
>
> 也就是说：**数据侧的空白已经被社区填上了，3D 渲染侧的空白仍然空着** —— 这正是本方案的位置。

### 6.4 路线 B3 —— 离线模仿学习：**AI 训练真正该走的那条路**

针对"把真实游戏内容用于 AI 训练"这个诉求，最重要的一条结论是：

> **CS2 是「人类轨迹数据」的优质来源，但作为实时 RL 环境很糟糕。**
> 专用服务器**只模拟、不渲染**（无 GPU、Linux x86-64）；客户端**只渲染、不能被稳定地逐步驱动**。**没有任何东西能同时渲染 + 被程序稳定地步进** —— 所有已发表项目都只选一边。

**本机可行的 AI 训练正路（按可行性排序）**
| 目标 | 本机可行 | 说明 |
|---|---|---|
| **① 纯状态模仿学习（BC）** | ✅✅ | 从 demo 提 per-tick `(state, action_label)`，本机可做，零额外硬件 |
| **② 128×128 像素 BC** | ✅（小规模） | 需做 32fps↔64tick 映射 + 冻结视觉编码器（如 DINOv2）特征 |
| **③ 雷达/2D 世界模型** | ✅✅ | 用解析出的状态 + 碰撞网格自己渲染 2D 观测 |
| ④ 真实时 RL / 像素 RL | ❌ | 需多 GPU 集群，且**在任何机器上都不是个好计划** |
| ⑤ 动作条件视频世界模型 | ❌ | 多 GPU + 多 TB + 数月 |

**🔴 必须纠正的一条调研结论（我亲自复测后推翻）**
有调研声称 demoparser2 暴露 `usercmd_mouse_dx/dy`、`usercmd_buttonstate_*`，即**真实逐 tick 人类输入**（那样就不需要反解动作了）。**我在 `test.dem` 上实测：不存在。**
```
parse_ticks(["usercmd_mouse_dx","usercmd_mouse_dy"])        -> 返回列 []   （静默丢弃）
parse_ticks(["usercmd_buttonstate_1","usercmd_buttonstate_2"]) -> 返回列 []
parse_ticks(["usercmd_mouse_dx","X","Y"])                   -> 返回 ['X','Y']  ← 机制正常，是字段不存在
list_updated_fields(): 966 个字段中，'usercmd'/'buttonstate'/'mouse' 命中 0
```
**但这个结论只对「实体属性表」这一层成立 —— 我当时把层次搞错了。**

**❗ 重要更正（见 §7.5）：逐 tick 人类输入存在，只是在另一层。** `svc_UserCmds` 是一条**网络消息**（packet 层，protobuf + codegen-delta 编码），`demoparser2` 的**实体属性 API 根本不会暴露它** —— 所以我在 966 个字段里找不到 `usercmd_*` 是正常的，但这**不等于数据不存在**。

**结论（已更正）**：规划 BC 时**可以期望拿到真实动作标签**（§7.5 的 13 个按键位掩码）；但需要自己写 packet 层提取器，且**部分 demo 可能没有这段流**，必须做可用性检测，并保留「状态差分反解」作为降级路径。

**数据资源（缓解磁盘约束）**
- ✅ 已验证：**`blanchon/cs2_dataset_demo`**（HF，**CC-BY-4.0**，9,843 个文件，含 `data/metadata-*.parquet`）—— 元数据 parquet 可用 **DuckDB 直接查 `hf://`**，**不下载 demo 即可筛选**，对 29 GB 磁盘极友好
- OpenCS2（~11 TB / ~5,000 小时，含 10 个 POV + 键鼠）、Reka CS2-10k（10,000 小时，CC BY-NC）、EgoCS-400K（CC BY 4.0）
- ⚠️ `blanchon/cs2_vae`（声称的 CS2 世界模型，含一个值得警惕的负结果：**动作条件化未通过 pose-shuffle 检验，Δ=0.000**，说明模型在吃帧边界而非学姿态）—— **我用 HF API 查该 model id 返回空，未能验证其存在**，引用前请自行确认

**已知陷阱（来自调研，引用前建议复验）**：不存在可用的 CS2 RL 环境（`gym-csgo` 已废弃；TeaPearce 那套明确不再维护且 Windows-only；`htglvl/CS2_Imitation_Bot` 被指为 CS:GO 仓库的逐字拷贝，**不可引用**）。**无现成的"插件 → Python" RL 桥**，属于真实生态空白。

---

## 7. ✅ 原「唯一技术缺口」已完全解决：火焰格子与烟雾体积都拿到真值

> **本节是 v5 方案最重要的更新，含一处对我上一轮过度断言的修正。**
> 早先的结论是"真实火焰格子拿不到、烟雾体积只能重新模拟"。**换用 demoinfocs-golang 后，火焰已拿到真值；烟雾的数据也能读到，但栅格解码尚未完成**（§7.2 ② 有详细校准）。

### 7.1 为什么 demoparser2 拿不到（根因已定位）

`Grenade.m_firePositions` / `m_fireCount` / `m_bFireIsBurning` / `m_VoxelFrameData` **确实在 demo 的 sendtable 里**（`list_updated_fields()` 共 **966** 字段），但 demoparser2 三种途径全部取不出（逐条实测）：

| 尝试 | 结果 |
|---|---|
| `parse_ticks(["Grenade.m_firePositions"])` | 只返回玩家列，属性静默丢弃 |
| `parse_event("inferno_startburn", other=["Grenade.m_fireCount"])` | 只返回事件原生列 |
| `parse_grenades(extra=[...])`（短名/全名都试） | 火焰类**列根本不出现**；`m_VoxelFrameData` 这一列**会出现，但退化成 float64 标量**，不是那个 802 字节数组 |
| **对照** `parse_event("round_end", other=["CCSTeam.m_iScore"])` | ✅ 正常返回 → 机制本身可用 |
| **对照** `parse_grenades(extra=["m_vInitialVelocity"])` | ✅ 正常返回 3 元组 → **`extra=` 这条路本身是好的**，缺的只是火焰那一组（详见 §7.6 ①） |

根因在 `demoparser/src/parser/src/second_pass/game_events.rs:307`：`find_non_player_props()` 只处理 `PropType::Team` / `Rules` / `GameTime`，而 grenade 属性归一化成 `Grenade.*` 后命中 `PropType::Weapon`，落到 `_ => vec![]` **被静默丢弃**。

### 7.2 换一个解析器就解决了 —— demoinfocs-golang（MIT），**零补丁**

**本机实测结论（Go 1.26.1 + `demoinfocs-golang v4.5.1`，解析 `test.dem` / de_ancient）：**

**① 真实火焰格子 —— 完整拿到**
```
>>> MAX real fire cells in one inferno: 16   (Valve cap = 16)
--- inferno uid=7403816786288401511  fullest tick=10801  burning cells=16 ---
   cell[ 0] (1172.1, -220.8, 115.8)     cell[ 8] (1225.9, -231.0, 111.4)
   cell[ 1] (1190.2, -254.9, 111.1)     cell[ 9] (1174.3, -330.1,  92.4)
   cell[ 2] (1221.3, -296.5, 104.4)     cell[10] (1249.5, -203.5, 113.7)
   ...                                  cell[15] (1139.3, -191.0, 118.2)
```
**56 处火焰全部拿到 16/16 格**，位置呈真实几何散射（**不是**固定 25/30 HU 点阵）。库还提供 `Fires().Active()` 与 `ConvexHull2D()` —— **后者是投影到水平面的"这块地面不能走" footprint**，⚠️ 但它**丢掉了 Z 轴**，不能当火焰体积（见 §7.3 第 1 条与 `docs/TOOLS.md` §5.5）。

**② 真实烟雾体积 —— ✅ 已完全解开（并已用本机数据独立验证）**

本机探针读到的原始态：
```
=== SMOKE PROJECTILE class=CSmokeGrenadeProjectile (BLOOMED) ===
  m_VoxelFrameData elements=3072  nonZero=454
  m_bDidSmokeEffect=true  m_nSmokeEffectTickBegin=17542
  m_nVoxelFrameDataSize=802  m_nVoxelUpdate=1
  m_vSmokeColor=[180 129 50]  m_vSmokeDetonationPos=[662.83, 483.76, 125.13]
```

**我曾两次误判它**：先说"3072 = 16×16×12 栅格"，后说"栅格映射定不出来"。**两个都错了。**
真正的原因：**`m_VoxelFrameData` 根本不是稠密栅格，而是一条 Morton 编码的 journal（掩码增量流）。** 我按栅格去 reshape，所以三种启发式必然全部失败。

**格式规范**（来源：`DrEAmSs59/CS2-insight-agent` 的 `smoke_voxel_decode.py`，PolyForm Noncommercial）：

```
journal 记录：  u16le seq | u16le payload_len | payload
  payload[0] = phase, payload[1] = frame_type
  type 3 KEYFRAME : seed_count(u8)，然后 seed_count × 8 字节 [x, y, z, density, pad×4]
                    然后 mask_count(u16)，然后 mask_count × 10 字节 [cell_morton(u16), voxel_mask(u64)]
  type 2 DELTA    : mask_count(u16)，然后 mask_count × 10 字节（替换式掩码）
  type 0 HEARTBEAT: 无几何变化（payload == 00 00 00）

栅格      ：32 × 32 × 32；两级 Morton 均按 X, Y, Z 交错
            cell_index 用 3 bit/轴 → 8×8×8 = 512 个 cell
            voxel_mask 的 64 bit 用 2 bit/轴 → 每 cell 4×4×4 子体素
            → 8³ × 4³ = 32³ = 32768 ✓
世界坐标  ：world = detonation + (grid − 15.5) × 12.0      （每体素 12 HU，轴向符号 (1,1,1)）
内部填充  ：掩码给的是**云边界**；被边界包住的内部由**从六个外表面做洪泛填充**确定
```

**本机独立验证**（用他们公开的规范解我自己采集的数据，见 `notes/scratch/decode_smoke_voxels.py`）：

| 检验项 | 结果 | 判定 |
|---|---|---|
| journal 切割 | **消耗 802/802 字节**，正好 1 条记录 | ✅ 结构自洽 |
| 首记录头 | `seq=0, len=798` → 4 + 798 = **802 = `m_nVoxelFrameDataSize`** | ✅ 完全吻合 |
| 帧类型 | `phase=0, type=3` → **KEYFRAME** | ✅ 合理（首帧必为关键帧） |
| 种子/掩码 | `seed_count=44`，`mask_count=44`，消耗 797/798 | ✅ |
| 种子坐标 | x 11–19, y 11–19, z 16–20（栅格 0–31） | ✅ 居中 |
| 解出体素 | **1119 个** | — |
| 栅格包围盒 | x 8–23, y 8–23, z 12–23 → **居中团块**（非占满全格） | ✅ |
| 世界质心 | (654.5, 489.6, 141.4)，引爆点 (662.8, 483.8, 125.1) → 偏移 ~15 HU 且**在引爆点上方** | ✅ 符合烟雾上浮 |
| 水平半径 | 最大 **127.3 HU** | ✅ |
| **3D 半径** | 最大 **155.9 HU** | ✅ 与烟雾 ~149 HU / CS:GO 常量 166 吻合 |
| 垂直跨度 | **132 HU** | ✅ 符合地面裁切透镜 |

**结论**：烟雾体积**是真值、可解、可验证**。上一轮我列的三个"待攻关"方向（引擎约定 / 多帧追踪 / 联合拟合）**全部不再需要**。

> ❗ **更正（第四个自我修正）**：我原先写"很多 demo 只网络化一帧关键帧 → 烟雾是静态的，膨胀只能 BFS 合成"。**这是错的**，而且错因很典型：
>
> `m_nVoxelFrameDataSize` 是**一条只追加的日志的当前长度**。我上次是在烟雾**刚出现的第一帧**去采样的 —— 那时日志里**只有关键帧**（802 字节），增量帧还没写进去。**我采样采早了，就以为数据只有一帧。**
>
> 把每颗烟的**最大**日志解一遍（`test.dem`，**81 颗烟，全部解通、全部 `size/size` 字节正好用完**）：
>
> | 指标 | 实测 |
> |---|---|
> | 关键帧 | **每颗烟恰好 1 个**（无一例外） |
> | **增量帧 DELTA** | **最少 11、中位 20、最多 29** ← **真实演化就在这里** |
> | 心跳帧 HEARTBEAT | 100–190 |
> | 记录总数 | 最少 73、中位 **202**、最多 **202**（202 是上限，75/81 颗达到） |
> | 日志总长 | 最少 1434、中位 2547、最多 2741 字节（数组容量 3072，**从未触顶**） |
> | 全场合计 | 81 关键帧 / **1628 增量帧** / 14156 心跳帧 |
>
> **结论翻转：烟雾的膨胀过程是真实数据**（1 关键帧 + ~20 增量帧），**不需要 BFS 合成**。上次那句"合成是视觉近似"作废。

### 7.3 这对方案的三个后果

1. **火焰是真值，但它是三维的 —— 不能用 `ConvexHull2D` 糊弄过去**：每处火焰有 16 格，每格带**三维坐标** `m_firePositions.NNNN` + **表面朝向** `m_BurnNormal.NNNN` + **蔓延父节点** `m_fireParentPositions.NNNN`。实测（56 处火焰）：**垂直跨度最大 124.3 HU**、**平面拟合残差最大 53.5 HU**、**37.5% 的火焰法向量明显倾斜**（最斜 nz=0.117 ≈ 竖直墙面）、**11% 的火焰爬升超过 50 HU**。
   ⇒ **3D 占据栅格要用这 4 条数组直接建**；`ConvexHull2D()` 是**投影到水平面的二维轮廓**，做地面封锁判断够用，但**会丢掉整个垂直结构**。详见 `docs/TOOLS.md` §5.5。
2. **烟雾也是精确的**：Morton journal 解开后，**可见性遮挡可以按真实逐体素占据来算**（32³ / 12 HU / 边界洪泛填充），不再需要"球体近似"。**膨胀过程同样是真值**（1 关键帧 + 11～29 增量帧，见 §7.2 更正）—— 原先"膨胀只能合成"的说法已作废，**这一项现在没有任何合成成分**。
3. **不需要改 Rust 解析器、不需要装游戏、不需要 VPK**：Go 版解析器 MIT 许可、`go get` 即有、5.2 秒解析完整场 demo。

**实现代价**：新增一条 Go 侧解析通道（`go build` 一个命令行导出器 → JSON/Parquet），或直接用 Go 写数据管线。本机 Go 工具链现已可用（已把 `GOPATH`/`GOCACHE` 定位到工作区内以适配沙箱）。

> 复现脚本：`notes/scratch/goprobe/main.go`（`go build -o probe . && ./probe <demo>`）。

### 7.4 剩余缺口（重新分级）
| 缺口 | 等级 | 说明 |
|---|---|---|
| ~~烟雾体素栅格~~ | ✅ **已解决** | Morton journal 格式已由 `CS2-insight-agent` 公开，并用本机数据独立验证通过。单颗深解：802/802 字节、44 种子 + 44 掩码、**1119 体素**、3D 半径 **155.9 HU**（见 §7.2 ②）。**全量复核：`test.dem` 81 颗烟全部解通、全部 `size/size` 字节正好用完**。原「三种启发式」路线全部作废 |
| `m_firePositions` 的容量语义 | 🟢 细节 | 是 `VectorWS[64]`，**64 只是容量**，有效格数上限 `MIN(64, m_nMaxFlames)` = **16**（实测正好 16/16） |
| CS:GO 老 demo 兼容 | 🟢 已解决 | 走 `m_fireXDelta/YDelta/ZDelta`（整数截断偏移自原点）分支，demoinfocs 已内置兼容 |
| ~~烟雾逐 tick 演化~~ | ✅ **已解决** | 原判"部分 demo 只有一帧、演化需 BFS 合成"**推翻**：实测 81/81 颗烟都是 **1 关键帧 + 11～29 增量帧**（中位 20），演化是**真值**。上次是采样采早了（见 §7.2 的更正） |
| 日志长度上限 | 🟢 细节 | 记录数上限 **202**（75/81 颗达到）；日志最长 **2741** 字节，数组容量 3072，**从未触顶** → 目前不需要处理溢出/环形覆盖 |
### 7.5 ★ 第二个重大更正：逐 tick 人类输入**其实存在**（`svc_UserCmds`）

> 这条直接推翻了 §4.5 的一条「已核实」结论，也是**对 AI 训练影响最大的一条**。

**我错在哪**：我只在 demoparser2 的**实体属性表**里找 `usercmd_*`，找不到就下了「demo 没有逐 tick 人类输入」的结论。但 `svc_UserCmds` 是**网络消息（packet 层）**，用 protobuf + codegen-delta 编码，**实体属性 API 根本不会暴露它**。是层次错了，不是数据不存在。

**证据**（`DrEAmSs59/CS2-insight-agent` 的 `input_command.py`，PolyForm Noncommercial）：
- 模块 docstring 原文：*"Exact CS2 input tracks extracted from DEM `svc_UserCmds` messages."*
- 由**一个 Rust 二进制**（`demo-input-hud-track`）负责 protobuf/codegen-delta 解码，输出 slot 键控的紧凑轨道 + userinfo 身份时间线
- **13 个按键位掩码**（数组顺序即 bit 序）：
  `W, A, S, D, jump, crouch, walk, reload, fire, scope, use, inspect, scoreboard`
- 轨道用 delta 编码压缩（`"tickDelta.mask"` 的 base36 token 串），跨 tick 用二分 + 掩码并集求值
- **有完整性契约**：`format_version >= 10`，且 `decode_errors == 0`、`commands_without_player_slot == 0` 才接受报告（否则抛错）—— 说明这是一条**有真值保证**的通道，不是启发式猜测
- slot → SteamID 绑定靠 `player_identity_updates`（userinfo 时间线），**可处理中途换人**

**对方案的影响**：
| 维度 | 原结论 | 更正后 |
|---|---|---|
| BC 的动作来源 | 由状态差分**反解**（有噪声、非真值） | **真实按键标签** |
| 动作空间 | 需自己定义再反解 | 可直接采用游戏的 bitmask 语义 |
| 代价 | 低 | 需自写 packet 层解包器（对方用 Rust 实现） |
| 残留风险 | 反解误差进标签 | **部分 demo 可能没有这段流** → 必须做可用性检测，并保留反解作降级 |

**边界**：该实现只提取**按键**，不含鼠标位移/朝向增量（至少公开代码里只列了 13 个键）。**视角仍应从实体层的 `view_angle`/`angles` 取**——那一层是可得的。
> **给上游的副产品**：demoparser2 的这个缺口（`find_non_player_props` 缺 `PropType::Weapon` 分支）仍值得修 —— awpy 等生态都在用它。但**它不再是本方案的阻断项**，因为 demoinfocs 已经绕过了。

### 7.6 ★ 第三个更正：缺口比原先写的小得多 —— **取值路径**才是关键

§7.1/§7.5 之后我又做了一轮全量核对，结果推翻了两个我自己的说法。

**① 全量扫描：966 个字段，885 个能取，81 个静默丢弃（走 `parse_ticks`）**

但这 81 个**不等于** "demoparser2 取不到" —— **换一条取值路径就又能取到一批**：

| 路径 | 结果 |
|---|---|
| `parse_ticks(["Grenade.xxx"])` | 81 个丢弃 |
| `parse_grenades(extra=["xxx"])` | 上述 76 个叶子名里 **38 个能取到** |

换路径后**能取到**的包括（均为实测非空）：

```
m_vInitialVelocity  m_vInitialPosition  m_flThrowStrength     ← 可重建真实弹道物理
m_nBounces          m_flDetonateTime     m_bIsIncGrenade      ← 弹跳次数 / 引爆时间
m_vSmokeDetonationPos  m_vSmokeColor  m_nSmokeEffectTickBegin  m_bDidSmokeEffect
m_nVoxelFrameDataSize  m_nVoxelUpdate                        ← 烟雾 bloom 代理信号
m_vecExplodeEffectOrigin  m_hThrower  m_bIsLive  m_flDamage  m_DmgRadius  ...
```

> ❗ 我第一版 PLAN 里"投掷物属性全部取不到"的说法**过宽**了。**只有 `parse_ticks` 这条路径取不到。**

**② 真正的缺口只剩 2 类**（两条路径都试过，确认取不到）：

| 缺口 | 具体字段 |
|---|---|
| **火焰格子** | `m_firePositions`、`m_fireCount`、`m_bFireIsBurning`、`m_fireParentPositions`、`m_nInfernoType`、`m_nFireLifetime`、`m_nFireEffectTickBegin`、`m_BurnNormal` |
| **烟雾体素数组的"内容"** | `m_VoxelFrameData` 本身。⚠️ 它的**大小** `m_nVoxelFrameDataSize`（实测 **802**）和 `m_nVoxelUpdate` 是**能取到**的；取不到的是那 802 字节的内容 —— 在 demoparser2 里退化成 **float64 标量**（不是数组）。demoinfocs 给的则是**完整 6145 元素数组** |

> 这两类**恰好就是本方案真正需要的东西**（§5.2 的火焰占据栅格、烟雾遮挡矩阵），所以 §7.1–7.2 的结论**不受影响** —— 只是理由更精确了。

**③ 对照：管线现在在用的字段，demoinfocs 一个不缺**

我按 `demo_parser/extract.py` 里真正点名的 29 个 `PLAYER_PROPS` + 13 个事件逐项核对（明细见 `docs/TOOLS.md` §1.6）：

- **玩家状态**：位置 / 视角 / 速度 / 血量 / 护甲 / 头盔 / 拆弹器 / 存活 / 武器 / 背包 / 闪光时长 / `m_flFlashMaxAlpha`（实测取到 `255`）/ 地点名 / `m_bSpottedByMask`（实测非零）/ 队伍 / SteamID / tickrate（实测 **64.0000**）/ `m_pGameRules.m_bBombPlanted` —— **全部命中**
- **13 个事件**：实测条数 `round_freeze_end` 18、`round_end` 18、`player_death` 133、`player_hurt` 607、`weapon_fire` 4029、`player_footstep` 360、`bomb_*` 12/3/5、`smoke_*` 82/76、`inferno_*` 58/58 —— **全部命中**
- 唯一没有现成取法的是 `inventory_as_ids`，但它在 `extract.py` 里**只是被请求、从未被消费** → 不构成缺口

**④ 逐 tick 按键掩码：两个解析器完全一致（这条推翻了"demoinfocs 独有"的猜测）**

- demoparser2：`CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtonDownMaskPrev`
- demoinfocs：`Player.ButtonsPressedState`
- **实测 60 组 `(tick, steamid)` 对照，能对齐的 54 组零不一致**
- 把整份 demo 的掩码按位或起来，**13 个键在第 1 层全部出现**：W 390,069 行 / D 135,370 / A 133,592 / walk 120,488 / fire 85,328 / scoreboard 59,708 / S 34,999 / crouch 28,399 / scope(`Attack2`) 18,903 / inspect 8,602 / jump 5,049 / use 4,990 / reload 1,032
- ️ `Zoom`(0x400000000) 与 `TurnLeft/Right` 整份 demo **从未置起** → 判"开镜"要用 `Attack2`(0x800)，不能用 `Zoom`
- ⇒ **§7.5 的 `svc_UserCmds` 提取器，价值只剩"鼠标"**（逐 tick 位移 + 每条指令携带的精确视角）。按键不必碰 packet 层

**⑤ demoinfocs 反而更差的 4 处**（原来没写）

| 项目 | 情况 |
|---|---|
| `PositionEyes()` | ⚠️ 在 Source 2 demo 上**直接 panic**（`"not supported for Source 2 demos"`）→ 眼位要自己用坐标 + 视角算。管线本来就不用眼位，无影响 |
| 手雷轨迹 | ️ **只有关键点，不是逐 tick**：实测 294 颗共 **1618 个点、单颗最多 12 个**，而其中一颗帧跨度 **1953 帧（30 s）**，且出现相邻点重合与 **250 帧空洞**。**不能当逐 tick 轨迹**；demoparser2 的 `parse_grenades()` 是逐 tick 的 |
| 皮肤 item def index | ⚠️ 在投掷物实体上读不到（在武器实体上才有）；demoparser2 反而能从 `Grenade.*` 读到 |
| 取值方式 | ⚠️ **单次顺序流式解析**，不支持 `parse_ticks(ticks=[...])` 那种按需取 tick |

**结论不变**：**主链路继续用 demoparser2**，只用 demoinfocs 补「火焰格子 + 烟雾体素内容」这一个窄口子；若能给 demoparser2 补上 `PropType::Weapon` 分支，则连这个 Go 工具都不需要。

> 复现脚本：`notes/scratch/dp2_retrievability.py`（全量 966 字段扫描 → `notes/scratch/dp2_field_retrievability.json`）、`notes/scratch/dp2_namespaced.py`、`notes/scratch/coverprobe/main.go`、`notes/scratch/nadeprobe/main.go`

---

## 8. 成本、风险、硬约束

| 项 | 状态 |
|---|---|
| 磁盘 | **29 GB 可用**。范式 A/G 线无影响（Go 探针 + 模块缓存仅 **160 MB**）；L2 选择性下载约需 **2–6 GB**，可行；L3 需 ~150 GB → 换机器 |
| CS2 无 macOS 版 | 已三重核实。L3 只能云 / Windows / Linux |
| ~~火焰格子~~ | ✅ **已解决**：demoinfocs-golang 实测拿到 16/16 格真实坐标（§7.2） |
| ~~烟雾体积~~ | ✅ **已解决**：Morton journal 格式已解开并用本机数据验证（§7.2 ②） |
| 参考项目许可 | **CS2-Insight-Agent 是 PolyForm Noncommercial 1.0.0**（非 MIT）：可读可学，**不可用于商业**；其换肤模块还是闭源专有。我们只能借鉴**格式知识**，不能拷贝代码进商业产品 |
| Go 工具链 | 已装 Go 1.26.1；沙箱下需把 `GOPATH`/`GOCACHE` 指向工作区（`.build/`），否则 denied |
| demoparser2 上游缺口 | 仍存在（`PropType::Weapon` 分支缺失），但**已不再是阻断项**；修它属于回馈社区 |
| 光照 | VRF **不导出 lightmap**；真 CS2 观感需要自己重新打光 + 移植着色器 |
| 授权 | 本地个人提取是社区惯例；**不要分发、不要公开托管 Valve 资产**。本项目刻意把归位后的 `visualizer/static/assets/` 入了库（为了让全新 clone 直接能跑，见 `notes/research-gameassets.md` §9.0.1），代价是**这个仓库必须保持私有**。代码许可 ≠ 资产许可 |
| 人物模型 | VRF #845 "Invalid skinning bone index" 是已知坑 |

---

## 9. 分期执行建议（两条腿）

```
【解析通道】★ 新增：Go 侧 demoinfocs-golang 导出器（真实火焰格子 + 真实烟雾体素）
  G0  `go build` 一个导出器：遍历 Infernos()/SmokeGrenadeProjectile → JSON/Parquet
     （本机 Go 1.26.1 已可用；GOPATH/GOCACHE 需指向工作区以适配沙箱）
  G1  ✅ 已解决：按 §7.2 ② 的 Morton journal 规范实现解码器（关键帧/增量/心跳 + 两级 Morton + 边界洪泛填充）→ 直接得到真实逐体素占据
  G3  ★ 新增：`svc_UserCmds` packet 层提取器（protobuf + codegen-delta）→ 13 键位真实输入（§7.5）。**这是下一个最高价值的独立产出物**
  G2  与 Python 侧 demoparser2 管线按 tick 对齐（注意 §4.5 的 per-demo tick 偏移）

【范式 A 线】本机可做，建议优先
  A0  L0 真数据管线：demoparser2 → cs2.demo.v7（真实烟/火事件、真实弹道、玩家完整状态）
      ✅ 已落地：`tools/demo_extract`（火焰 16 格 + 烟雾体素、无损 codec、按回合切分）
      ✅ v6 新增地面接触：`on_ground`（`m_fFlags` bit 0，与 `m_hGroundEntity` 100% 一致）
         + `jump`（`player_jump` 事件）+ `land`（从 on_ground 跳变推导）
         → 移动噪声不用再写死速度阈值，模型自己从「速度 + 踩地」学
      ✅ v7 新增下蹲：`crouch`（`m_flDuckAmount`，连续 0..1，站→蹲中位 13 tick / 203 ms）
         + `crouching`（`m_fFlags` bit 2 = `FL_DUCKING`，碰撞盒 72→54 的阶跃）
         + 派生 `eye_height` / `eye_position`。体积代价 +0.54%（7.371 → 7.411 MB）
         可视化：底部 `碰撞盒 / 眼睛` 开关，画出真实碰撞盒 + 眼睛 + 脚到眼标尺
      ⚠️ 踩到上游 bug 并已绕过：demoparser2 的 `velocity*` 在 tick 列表**不连续**时
         返回别的 tick 的速度（stride=2 就有 84% 是错的）。对策是速度单独跑一次
         不带 `ticks` 的解析；`--selftest` 会断言 `v == hypot(vx, vy)` 且 `|vz| < 1200`
      处理两个坑：tick 偏移标定、用事件而非 entity 生命周期配对烟
  A1  火焰几何：接 demoinfocs-golang 通道，取真实 16 格 **3D 坐标 + BurnNormal + ParentPositions + IsBurning**（**真值，无近似**；`ConvexHull2D` 仅作地面封锁的 2D 视图）
  A2  ★ 输入表示改造：烟雾 10×10 遮挡矩阵 + 火焰/烟雾空间栅格 + 真实投掷物轨迹
  A3  用现有 evaluate 链路做 A/B，量化收益
  A4  （独立投入）方案① 打 Rust 补丁拿真值火焰格子 → 回馈上游

【范式 B 线】按需推进
  B0  DepotDownloader 选择性下载：索引 + 需要的块 + 一张地图 VPK
  B1  Source2Viewer-CLI 导出：地图（碰撞网格 + vmap glTF 两条）、人物模型 GLB、粒子贴图
  B2  渲染器升级：HU 1:1、FOV 73.7398°、Hable 色调映射、真实雾/AO/CSM
      烟雾按 149 HU 透镜 + 18 s + <1 s bloom + 阵营染色 + HE 灭烟重建；**G1 解开体素栅格后再切换到真实体素密度**
      火焰用真实 footprint（80 HU 高 billboard / 40 HU 宽 / 42 HU 间距 / 分阵营扩散速度）
  B3  ★ 保真的真正门槛：把 Source 2 着色器移植到 WGSL + 用真实地图光照数据
  B4  仅在需要像素真值时：云 Windows/Linux GPU 跑真引擎（参考 reka-ai 的插件+捕获范式）

【范式 C 线】离线模仿学习（若目标是"训练 AI"而非"看回放"，这条才是正路）
  C0  DuckDB 查 `hf://` 上的 cs2_dataset_demo 元数据 → 按需只下少量 demo（省磁盘）
  C1  纯状态 BC：per-tick (state, action)。**动作可用真实标签**（`svc_UserCmds` 的 13 键位掩码，§7.5）；无该消息的 demo 才退回状态差分反解
  C2  128×128 像素 BC：32fps↔64tick 映射 + 冻结视觉编码器特征
  C3  雷达/2D 世界模型：本机可行
  ✗ 不要做：实时 RL、像素 RL、动作条件视频世界模型（本机不可行，且在任何机器上都不是好计划）
```

**建议的起点**：`A0 → A2`（立刻见效、无外部依赖），并行评估 `A4` 与 `B0`。若真实目标是"训练 AI"而非"看回放"，则 `C0 → C1` 应优先于整条 B 线。

---

## 10. 待决策

1. **真实目标是"训练 AI"还是"看回放"？**
   - 若是**训练 AI** → `C 线（离线模仿学习）` 应优先于整条 B 线。范式 A（真几何特征）是它的前置增强。
   - 若是**看回放/对外展示** → `A 线 + B 线`。
2. **两条线的优先级**：先做范式 A（本机立刻、收益直接），还是先做范式 B（真资产视觉）？
3. ~~火焰格子怎么求真值？~~ ✅ **已解决，无需决策** —— demoinfocs-golang 直接给出真值（§7.2）。原"改 Rust 解析器"降级为可选的社区回馈项。
4. **真资产下载**：是否同意用 DepotDownloader 选择性下载（本机约 2–6 GB，需要匿名 Steam 访问）？
5. **要不要用云端"真引擎渲染"服务做校验/少量真值？** 建议：**先用 DEMO-SLAP 的免费额度 / CLUTCHKINGS 试一发**（零安装、零磁盘），用真引擎画面校验自建渲染器的偏差；但**不要把产品建在 DEMO-SLAP 上**（两人副业、原型级）。大规模仍须租云 GPU。
6. ~~烟雾体素栅格是否值得攻关？~~ ✅ **已解决**（§7.2 ②）。**新问题**：是否投入做 `svc_UserCmds` 提取器（§7.5 / G3）？它让 BC 用上**真实动作标签**，但需要自己写 protobuf/codegen-delta 解包。
7. **实时 RL / 像素 RL 是否完全不考虑？** 本方案建议：**不要做**，在任何机器上都不是好计划。
8. **`NOTES.md` 是谁的？** 工作区里出现了一个我没有创建的 `NOTES.md`（内容是关于把社区 CS2 模型迁移到 Animgraph 2，"26 个模型已在游戏中验证"）。本机没有 CS2、不可能做游戏内验证，所以我判断它**不是本次调研的产物**——如果你有另一个会话/工具在同一工作区写入，请告知，我以免覆盖。

---

## 附录：调研产出

| 文件 | 内容 |
|---|---|
| `notes/research-fidelity.md` | CS2 烟/火/人物/单位/相机/光照精确保真参数（859 行，逐条标注证据等级与出处） |
| `notes/research-assets.md` | 资产获取：DepotDownloader 选择性下载、CLI 命令、公开模型源、授权 |
| `notes/research-ingame-footage.md` | 真引擎录屏路线：HLAE / startmovie / 云 GPU / macOS 可行性（133 条引用） |
| `notes/research-demoparser2.md` | demoparser2 全量字段实测（835 行）：92 个可用渲染属性、事件清单、两个坑 |
| `notes/research-ai-environment.md` | CS2 作为 AI 训练环境（614 行 / ~60 引用）：服务器 vs 客户端、模仿学习、数据集、世界模型 |
| `notes/research-fidelity-ai.md` | 真资产渲染的保真度缺口（986 行）：VRF.Renderer vs glTF、粒子/光照损失、逐项匹配度 |
| `notes/scratch/goprobe/` | ★ **关键实验**：demoinfocs-golang 探针，实测拿到真实火焰格子 + 烟雾体素场 |
| `notes/scratch/decode_smoke_voxels.py` | ★ **烟雾体素 journal 解码器 + 自校验**：独立复现并验证了 §7.2 ② 的格式规范（802/802 字节、1119 体素、3D 半径 155.9 HU） |
| `notes/scratch/` | 其他本机实测脚本与原始结果（tickrate、性能、火焰格子验证、voxel 增长、tick 偏移） |
| `docs/PLAN.md` | 本文档 |
| **外部参考项目** | `DrEAmSs59/CS2-insight-agent`（870★，PolyForm Noncommercial）：其 `smoke_voxel_decode.py` 公开了烟雾 journal 格式，`input_command.py` 公开了 `svc_UserCmds` 输入提取方案。**本项目是 §7.2 ② 与 §7.5 两条关键突破的来源** |

> ⚠️ **重要环境变更**：本次调研期间，`~/Downloads/0823_inferno.dem` 与整个 `~/Downloads/test_demos/`（约 10 GB）**从磁盘上消失**。所有测量数据在删除前已保存到 `notes/scratch/*.json`，故本方案的数值结论不受影响，但**后续任何依赖这些 demo 的复现都会失败**。现存可用 demo：`~/Downloads/test.dem`（de_ancient）、`9207151584779083532_0.dem`（de_overpass）、`9210813275233280652_0.dem`，以及 `cs-net-dev/examples/demo/` 下两个。
