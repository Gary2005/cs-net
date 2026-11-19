/**
 * visuals.js — 3D 可视化：玩家 / 投掷物 / C4 / **真实烟雾体素** / **真实火焰格子**
 * / **子弹曳光**。
 *
 * 和 cs-net 旧版的区别（重点）
 * ---------------------------
 * * 旧版烟雾是**一个半径 3.2 的静态半透明球**，火焰是**一个贴地圆盘**：
 *   和真实数据无关。
 * * 这里烟雾来自 `demo_extract` 的 **Morton journal**，在浏览器端解码成
 *   32³ 体素云（表面壳 → 填满表面点云的凸包成实心云），逐体素用
 *   InstancedMesh 画出来；
 *   火焰来自 **16 格真实 3D 坐标 + 表面法向 + 燃烧位**，每格画一片
 *   **平铺在表面上的贴片**（不是立起来的火柱）。
 * * 曳光弹来自 `events.sound` 的**真实开火事件**（精确 tick + 开枪者），
 *   方向取那一帧的 yaw/pitch，遮挡和瞄准射线共用 `castRay()`。
 *
 * 坐标：game (x, y, z) z-up → three (y*S, z*S, x*S)，S = 0.0254（1 HU = 1 英寸）。
 */

import * as THREE from 'three';
import {
  scene, camera, renderer, gameToThree, SCALE, mapGroup, raycaster, registerMinimapPass,
} from './scene.js';
import {
  weaponName, grenadeCategory, grenadeColor,
  playerTeam, teamColorClass,
  eyeHeightFor, hullHeightFor, HULL_HALF_WIDTH,
  HULL_STAND_HEIGHT, CROUCH_EYE_DROP, GAIT_BOB,
} from './replay-core.js';
import { SmokeVoxelCache, VOXEL_WORLD_SIZE, VOXEL_GRID_DIM } from './smoke-voxels.js';
import { weaponIconTexture, loadSpriteSequence } from './cs2-assets.js';
import {
  attachWeapon, clearWeaponModels, setWeaponModelsVisible,
  setPlayerWeaponModelVisible, weaponModelDebugStats,
  loadNadeModel,
} from './weapon-models.js';
// 投掷物类名 -> 模型 key 的映射在纯模块里（node 可测 —— 见 check_nade_models.mjs）
import { nadeModelKey } from './weapon-index.js';
import { raycastBVH } from './map-raycast.js';
import { getMapBVH } from './map-loader.js';

/** HU 方向向量 → three 方向向量（只换轴，不缩放）。 */
function gameDirToThree(gx, gy, gz) {
  return new THREE.Vector3(gy, gz, gx).normalize();
}

// ── 玩家几何 / 材质 ─────────────────────────────────────────────────────────
//
// 小人按真实比例搭，不是三个基本体的堆叠：两条腿 + 上宽下窄的躯干 + 双臂 +
// 头 + 头盔 + 面罩 + 视线锥。总高正好 72 HU（= 站立碰撞盒高度），脚底在 y=0，
// 所以"头顶 ≈ 碰撞盒顶"，两者可以直接对照着看。
//
// 蹲下时躯干那一组按**连续**的 crouch 压缩 Y，而碰撞盒按**阶跃**的 crouching
// 跳 72→54 —— 这是刻意的，见 API.md §4.12。压缩比取
// (72 - 18·crouch)/72，所以蹲到底时小人正好和蹲下的碰撞盒一样高（54 HU），
// 而且和眼睛的下落量共用同一个 18 HU。
const P_LEG_TOP = 0.78;      // 胯高（腿顶）
const P_TORSO_TOP = 1.43;    // 肩线
const P_HEAD_Y = 1.64;       // 站立时头部中心（≈ 站立眼高 64.09 HU = 1.628）
const P_HEAD_R = 0.19;
// 腿胶囊总高 = 0.58 + 2×0.10 = 0.78 = P_LEG_TOP，所以"半长"就是胯高的一半。
const P_LEG_HALF = P_LEG_TOP / 2;
const P_ARM_Y = 1.13;        // 手臂中心离地高度（站立）

let legGeo, torsoGeo, armGeo, headGeo, helmetGeo, visorGeo, dirGeo, hpBarGeo;
/**
 * 步态数据：每个玩家一份 `{phase, amp}`（`buildGait()` 的产物，见 replay-core.js）。
 * 换回合时由 main.js 重新灌进来。空数组 = 不摆腿（旧数据 / 还没算）。
 * @type {Array<{phase: Float32Array, amp: Float32Array}>}
 */
let gaitData = null;
// 碰撞盒 / 眼睛：都用单位几何，每帧靠 scale 变形，不重建几何体。
let hullGeo, eyeGeo, rulerGeo;
let showCrouchDebug = false;

function ensureGeometries() {
  if (torsoGeo) return;
  // 腿：胶囊（总高 = 0.58 + 2×0.10 = 0.78，正好到胯）
  legGeo = new THREE.CapsuleGeometry(0.10, 0.58, 4, 8);
  // 躯干：上宽下窄的锥台，肩比腰宽 —— 比原来的直筒圆柱像人
  torsoGeo = new THREE.CylinderGeometry(0.23, 0.165, P_TORSO_TOP - P_LEG_TOP, 14);
  armGeo = new THREE.CapsuleGeometry(0.065, 0.42, 4, 8);
  headGeo = new THREE.SphereGeometry(P_HEAD_R, 16, 12);
  // 头盔：只取球的上半部分，扣在头上
  helmetGeo = new THREE.SphereGeometry(P_HEAD_R + 0.02, 16, 8, 0, Math.PI * 2, 0, Math.PI * 0.55);
  // 面罩：贴在脸前的扁方块，一眼能看出朝向
  visorGeo = new THREE.BoxGeometry(0.22, 0.085, 0.06);
  // 视线指示：细圆锥。比原来那个 r=0.15/h=0.5 的粗圆锥克制得多
  dirGeo = new THREE.ConeGeometry(0.05, 0.30, 8, 4);
  hpBarGeo = new THREE.PlaneGeometry(1.0, 0.12);
  // 1×1×1 的盒子棱线，缩放到真实碰撞盒尺寸（32×32×72 或 54 HU）
  hullGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
  eyeGeo = new THREE.SphereGeometry(0.075, 10, 10);
  // 从脚底到眼睛的竖线（单位长度），用 scale.y 拉伸到眼高
  rulerGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0),
  ]);
}

const matCT = new THREE.MeshStandardMaterial({
  color: 0x3d8bd4, roughness: 0.35, metalness: 0.25,
  emissive: 0x0d2f55, emissiveIntensity: 0.4,
});
const matT = new THREE.MeshStandardMaterial({
  color: 0xea9d2c, roughness: 0.35, metalness: 0.25,
  emissive: 0x4a2e05, emissiveIntensity: 0.4,
});
const matDead = new THREE.MeshStandardMaterial({
  color: 0x777777, roughness: 0.8, metalness: 0.1, transparent: true, opacity: 0.45,
});
const matHead = new THREE.MeshStandardMaterial({
  color: 0xffcc99, roughness: 0.6, metalness: 0.0,
});
// 腿 / 面罩 / 视线锥：全局共享，所有玩家一个材质就够
const matGear = new THREE.MeshStandardMaterial({
  color: 0x2f3640, roughness: 0.78, metalness: 0.12,
});
const matVisor = new THREE.MeshStandardMaterial({
  color: 0x121a24, roughness: 0.22, metalness: 0.65,
  emissive: 0x0d2b3d, emissiveIntensity: 0.55,
});
const matDir = new THREE.MeshStandardMaterial({
  color: 0xffffff, roughness: 0.3, metalness: 0.1,
  emissive: 0x8899aa, emissiveIntensity: 0.6,
  transparent: true, opacity: 0.75,
});

const matHPHigh = new THREE.MeshBasicMaterial({ color: 0x3fb950, side: THREE.DoubleSide, depthTest: false });
const matHPMid  = new THREE.MeshBasicMaterial({ color: 0xd2991d, side: THREE.DoubleSide, depthTest: false });
const matHPLow  = new THREE.MeshBasicMaterial({ color: 0xf85149, side: THREE.DoubleSide, depthTest: false });
const matHPBg   = new THREE.MeshBasicMaterial({ color: 0x333333, side: THREE.DoubleSide, depthTest: false });

// 碰撞盒 / 眼睛（下蹲调试层）。碰撞盒是**阶跃**的：站 72 HU、蹲 54 HU，
// 中间没有 55..71 的取值 —— 实测 m_vecMaxs.z 只有这两个数。
const matHullCT = new THREE.LineBasicMaterial({ color: 0x6fc3ff, transparent: true, opacity: 0.85, depthTest: false });
const matHullT  = new THREE.LineBasicMaterial({ color: 0xffc46f, transparent: true, opacity: 0.85, depthTest: false });
const matEye    = new THREE.MeshBasicMaterial({ color: 0x62ff8a, depthTest: false });
const matRuler  = new THREE.LineBasicMaterial({ color: 0x62ff8a, transparent: true, opacity: 0.55, depthTest: false });

let playerGroups = [];

// ── 「可见敌人」眼睛图标 ────────────────────────────────────────────────────
//
// 在左侧玩家卡里选中一个人（`expandedPlayerIdx`）之后，把**他能看到的敌人**
// 头顶标一个眼睛。数据是 `.csz` 里的 `spotted` 列（demo_extract 逐 tick 提取的
// `m_bSpottedByMask`：谁被谁看见），方向在 `replay-core.visibleEnemiesOf()`
// 里有说明和几何验证 —— 别搞反。
//
// 只有 10 个图标、而且长得一模一样，所以共用**一张贴图 + 一个材质**，
// 每帧只改 position.y 和 visible。
const SPOT_ICON_SIZE = 128;
/** 图标中心相对头顶再抬高多少（three 单位 = 米）。名字标签在 +0.27，往上让开。 */
const SPOT_ICON_LIFT = 0.72;
/** 图标在世界上多大（米）。比名字标签小一圈。 */
const SPOT_ICON_SCALE = 0.62;

let spotIconsEnabled = true;
/** @type {Set<number>|null} 选中玩家能看到的敌人下标；null = 没选中任何人。 */
let spotTargets = null;
let spotIconMaterial = null;

/** 画一个眼睛图标到 canvas（眼眶 + 虹膜 + 瞳孔 + 深色底）。 */
function drawEyeIcon(ctx, size) {
  const cx = size / 2;
  const cy = size / 2;
  const w = size * 0.86;
  const h = size * 0.54;
  const accent = '#00e5ff';

  // 深色圆底：地图有亮面也有暗面，不垫一层的话图标会糊在墙里
  ctx.beginPath();
  ctx.arc(cx, cy, size * 0.45, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(5, 11, 18, 0.62)';
  ctx.fill();

  // 眼眶：上下两条二次曲线
  ctx.beginPath();
  ctx.moveTo(cx - w / 2, cy);
  ctx.quadraticCurveTo(cx, cy - h, cx + w / 2, cy);
  ctx.quadraticCurveTo(cx, cy + h, cx - w / 2, cy);
  ctx.closePath();
  ctx.fillStyle = 'rgba(2, 9, 15, 0.8)';
  ctx.fill();
  ctx.lineWidth = size * 0.062;
  ctx.strokeStyle = accent;
  ctx.shadowColor = accent;
  ctx.shadowBlur = size * 0.14;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // 虹膜 + 瞳孔
  ctx.beginPath();
  ctx.arc(cx, cy, size * 0.15, 0, Math.PI * 2);
  ctx.fillStyle = accent;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx, cy, size * 0.066, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(4, 8, 14, 0.95)';
  ctx.fill();
}

/** 眼睛图标的共享材质（懒建，全部玩家共用一个实例）。 */
function getSpotIconMaterial() {
  if (spotIconMaterial) return spotIconMaterial;
  const canvas = document.createElement('canvas');
  canvas.width = SPOT_ICON_SIZE;
  canvas.height = SPOT_ICON_SIZE;
  drawEyeIcon(canvas.getContext('2d'), SPOT_ICON_SIZE);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  spotIconMaterial = new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false, depthWrite: false,
  });
  return spotIconMaterial;
}

function createSpotIcon() {
  const sprite = new THREE.Sprite(getSpotIconMaterial());
  sprite.name = 'spot-icon';
  sprite.scale.set(SPOT_ICON_SCALE, SPOT_ICON_SCALE, 1);
  sprite.renderOrder = 990;   // 和名字标签一样画在最上面
  sprite.visible = false;
  return sprite;
}

/**
 * 灌入"选中玩家此刻能看到的敌人"。传 `null` 表示没有选中任何人（全部不画）。
 * 由 main.js 每帧调用；可见性 + 高度在 `updatePlayers()` 里落到具体 sprite 上。
 * @param {number[]|null} enemies
 */
export function setSpotTargets(enemies) {
  spotTargets = enemies ? new Set(enemies) : null;
}

export function setSpotIconsVisible(visible) {
  spotIconsEnabled = !!visible;
  if (!visible) {
    for (const pg of playerGroups) if (pg.spotIcon) pg.spotIcon.visible = false;
  }
}

/** 调试 / 测试用：拿到某个玩家的眼睛图标 sprite。 */
export function spotIconAt(i) {
  return playerGroups[i] ? playerGroups[i].spotIcon : null;
}

/** 调试 / 测试用：每个玩家眼睛图标画出来的状态（世界坐标，米）。 */
export function spotIconDebug() {
  const out = [];
  for (let i = 0; i < playerGroups.length; i++) {
    const s = playerGroups[i].spotIcon;
    if (!s) { out.push(null); continue; }
    const w = s.getWorldPosition(new THREE.Vector3());
    out.push({
      visible: s.visible && playerGroups[i].group.visible,
      flagVisible: s.visible,
      target: !!(spotTargets && spotTargets.has(i)),
      y: s.position.y,
      world: [w.x, w.y, w.z],
      scale: s.scale.x,
    });
  }
  return out;
}

// 透视地图开关。回放中 updatePlayers 每帧都会跑，它写的闪光自发光会把
// setXRayMode 设的发光覆盖掉 —— 所以开着透视时必须跳过那段。
let xrayEnabled = false;

// ── 轨迹 ────────────────────────────────────────────────────────────────────
//
// 轨迹 = **最近 MAX_TRAIL_POINTS 个采样点**连成的折线，除此之外没有别的状态。
// 这也意味着它是采样点的纯函数：`trail(k)` 只由 `[k-149, k]` 决定，和"是怎么走到
// k 的"无关 —— 回到同一个采样点，画出来必须是同一条线。
//
// ⚠️ 原来不是这样：那是一个「每帧往环形缓冲里 push 一个点」的累积器。而拖时间轴
// 走的是 `seekTo()` —— 它只改 `currentSampleIdx`，一个点都不碰。于是跳转之后的
// 第一帧把「跳转前的位置」和「跳转后的位置」连成一条横穿整张图的长线：de_inferno
// 实测正常每采样点位移 0.34 m，跳转后出现 **25 m / 38 m** 的段，整条轨迹长度
// 从 26 m 涨到 279 m。来回拖几下就是一张蜘蛛网，看起来是轨迹自己在瞬移。
//
// 现在按「采样点是否连续」分两路：连续 +1 就追加一个点（便宜），其余全都
// （跳转、倒退、掉帧一次跳过好几个采样点、开关关过一阵再打开、换回合）按历史
// 重算整条轨迹。两路都严格是"窗口里每个采样点一个点"，所以结果逐点相同 ——
// `visualizer/tests/test_trail_seek.mjs` 量这个。
//
// 这里刻意**不做"相邻两点太近就合并"**：窗口本来就是按采样点切的，多出来的重
// 合点既不影响观感（零长度的段画不出东西），又会让"一路播过来"和"跳过来"在同
// 一段慢速移动上数出不同的点数（贪心去重的相位取决于从哪儿开始），于是重算和
// 播放对不上。去掉之后 `trail(k)` 是纯函数，两个数永远相等。
const MAX_TRAIL_POINTS = 150;

let playerTrails = [];
for (let i = 0; i < 10; i++) {
  const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setDrawRange(0, 0);
  // `lastSample` = 最后一个点对应的采样点下标；-1 = 缓冲是空的 / 不可信
  playerTrails.push({ positions, geo, count: 0, lastSample: -1 });
}

/**
 * 历史采样点来源：`(i) => [{x, y, z}, …]` —— 第 i 个采样点上所有人在**游戏坐标**
 * 里的位置，由 `main.js` 在 init 时注入。
 *
 * ⚠️ 这里要的是**采样点上的原始位置，不是插值后的**。`getPlayerState()` 会按
 * `currentTime` 在相邻采样点之间插值，同一个采样点在不同的播放相位下给出的坐标
 * 差几厘米；用它来重算，跳回来以后轨迹末端就会和原来差一截。取样值点之后，
 * `trail(k)` 才是采样点的纯函数。
 *
 * 重算轨迹只能问"第 i 个采样点上所有人在哪"，而这只有 replay-core 知道；visuals.js
 * 不 import 它（两个模块之间已经够绕了），所以用一个注入的回调。
 */
let trailHistoryAt = null;

export function setTrailHistorySource(fn) {
  trailHistoryAt = fn;
  // 换了数据源（换回合、换文件），旧缓冲一律作废：下一次 sync 会按新历史重算
  clearAllTrails();
}

// ── 瞄准射线 / 子弹曳光：公共的射线检测 ─────────────────────────────────────
//
// 两条线用的是**同一次检测**：从**真实眼睛**沿瞄准方向（yaw/pitch）射出，在最近的
// 遮挡处截断。遮挡有两种：
//   1. 其他**活着的玩家** —— 用真实碰撞盒（世界轴对齐的 32×32×72/54 盒子，
//      就是 m_vecMins/m_vecMaxs，跟下蹲调试层画的是同一个东西）
//   2. 地图几何 —— 走 map-raycast.js 的 BVH（823k 三角形，朴素 raycast 要 45 ms）
//
// 命中类型决定颜色：打人=红、打墙=琥珀、什么都没打到=暗青（画到 maxDist 为止）。

/** 没打到东西时画多长（three 单位 = 米）。约 2362 HU，比 CS2 里任何一条视线都长。 */
const AIM_MAX_DIST = 60;

/** 玩家碰撞盒半宽：32 HU / 2 = 16 HU。 */
const AIM_HULL_HALF_HU = 16;

/**
 * 某个玩家这一帧的「眼睛 + 瞄准方向」。瞄准射线和子弹曳光共用，保证两条线
 * 起点/方向完全一致（测试也靠这个交叉验证）。
 * @returns {{start: THREE.Vector3, forward: THREE.Vector3}}
 */
function eyeRay(state) {
  const pos = gameToThree(state.x, state.y, state.z);
  // 射线从**真实眼睛**发出。以前这里硬编码 64 HU：站立时误差可忽略
  // （64 vs 64.0938），但蹲下时会差整整 18 HU —— 射线会从站立眼高射出。
  const eyeH = state.eye_height != null ? state.eye_height : eyeHeightFor(state.crouch);
  const start = new THREE.Vector3(pos.x, pos.y + eyeH * SCALE, pos.z);

  const yawRad = THREE.MathUtils.degToRad(state.yaw || 0);
  const pitchRad = THREE.MathUtils.degToRad(state.pitch || 0);
  const forward = new THREE.Vector3(0, 0, 1);
  forward.applyAxisAngle(new THREE.Vector3(1, 0, 0), pitchRad);
  forward.applyAxisAngle(new THREE.Vector3(0, 1, 0), yawRad);
  forward.normalize();
  return { start, forward };
}

/**
 * 从 start 沿 dir 找**最近的遮挡**：先扫其他活着的玩家碰撞盒，再问地图 BVH。
 * 玩家更近就是打人，否则是打墙。BVH 没建好（或没地图）就退化成「只截断到玩家」。
 *
 * @param {Array} playerStates 全部玩家状态（selfIdx 那个会被跳过）
 * @param {number} selfIdx 开枪/观察的人自己，不算自己的盒子
 * @returns {{distance:number, hitPlayer:number, hit:boolean}}
 */
function castRay(playerStates, selfIdx, start, dir, maxDist) {
  let best = maxDist;
  let hitPlayer = -1;

  // ① 先扫其他**活着**的玩家。碰撞盒是世界轴对齐的 32×32×72/54 盒子
  //    （m_vecMins/m_vecMaxs，不随 yaw 转），所以直接做射线-AABB。
  for (let j = 0; j < playerStates.length; j++) {
    if (j === selfIdx) continue;
    const other = playerStates[j];
    if (!other || !other.alive) continue;
    const t = rayHull(other, start, dir, best);
    if (t >= 0 && t < best) { best = t; hitPlayer = j; }
  }

  // ② 再问地图 BVH。BVH 还没建好（或没地图）就退化成"只到玩家"。
  const bvh = getMapBVH();
  if (bvh) {
    const t = raycastBVH(bvh, start.x, start.y, start.z, dir.x, dir.y, dir.z, best);
    if (t >= 0 && t < best) { best = t; hitPlayer = -1; }
  }

  return { distance: best, hitPlayer, hit: hitPlayer >= 0 || best < maxDist };
}

const matAimPlayer = new THREE.LineBasicMaterial({
  color: 0xff4d4d, transparent: true, opacity: 0.9, depthTest: false,
});
const matAimWall = new THREE.LineBasicMaterial({
  color: 0xffb020, transparent: true, opacity: 0.85, depthTest: false,
});
const matAimFree = new THREE.LineBasicMaterial({
  color: 0x66ccff, transparent: true, opacity: 0.25, depthTest: false,
});
const matAimMarkPlayer = new THREE.MeshBasicMaterial({ color: 0xff4d4d, depthTest: false });
const matAimMarkWall = new THREE.MeshBasicMaterial({ color: 0xffb020, depthTest: false });
const aimMarkGeo = new THREE.SphereGeometry(0.055, 8, 6);

let aimRays = [];
let aimRaysVisible = false;

/** 灌入本回合的步态数据（`buildGait()` 的返回值）。换回合时调用。 */
export function setGait(data) {
  gaitData = data || null;
}

export function createAimRays() {
  clearAimRays();
  for (let i = 0; i < 10; i++) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 1),
    ]);
    const line = new THREE.Line(geo, matAimFree);
    line.visible = false;
    line.renderOrder = 650;
    scene.add(line);

    // 命中点的小球，打到了才显示
    const mark = new THREE.Mesh(aimMarkGeo, matAimMarkWall);
    mark.visible = false;
    mark.renderOrder = 651;
    scene.add(mark);

    aimRays.push({ line, mark });
  }
}

function clearAimRays() {
  for (const r of aimRays) {
    scene.remove(r.line);
    r.line.geometry.dispose();
    scene.remove(r.mark);
  }
  aimRays = [];
}

export function updateAimRays(playerStates) {
  // 地图 BVH 由 main.js 在打开开关时 ensureMapBVH() 建好；没建好就只截断到玩家。
  if (!aimRaysVisible) {
    for (const r of aimRays) { r.line.visible = false; r.mark.visible = false; }
    return;
  }
  for (let i = 0; i < aimRays.length; i++) {
    const state = playerStates[i];
    if (!state || !state.alive) {
      aimRays[i].line.visible = false;
      aimRays[i].mark.visible = false;
      continue;
    }

    const { start, forward } = eyeRay(state);
    const { distance: best, hitPlayer, hit } = castRay(playerStates, i, start, forward, AIM_MAX_DIST);
    const end = start.clone().addScaledVector(forward, best);

    const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
    aimRays[i].line.geometry.dispose();
    aimRays[i].line.geometry = geo;
    aimRays[i].line.material = !hit ? matAimFree
      : hitPlayer >= 0 ? matAimPlayer : matAimWall;
    aimRays[i].line.visible = true;

    aimRays[i].mark.visible = hit;
    if (hit) {
      aimRays[i].mark.position.copy(end);
      aimRays[i].mark.material = hitPlayer >= 0 ? matAimMarkPlayer : matAimMarkWall;
    }
    aimRays[i].hitPlayer = hitPlayer;
    aimRays[i].distance = hit ? best : -1;
  }
}

/**
 * 射线 vs 某个玩家的碰撞盒。世界轴对齐：半宽 16 HU，高度取 crouching（72 或 54 HU，
 * 阶跃 —— 和碰撞盒调试层用的是同一个 `hullHeightFor`）。命中返回距离，否则 -1。
 */
function rayHull(other, start, dir, maxDist) {
  const pos = gameToThree(other.x, other.y, other.z);
  const hw = AIM_HULL_HALF_HU * SCALE;
  const h = hullHeightFor(other.crouching) * SCALE;

  // 三个轴的 slab 测试，取进入距离
  let tmin = 0, tmax = maxDist;
  const lo = [pos.x - hw, pos.y, pos.z - hw];
  const hi = [pos.x + hw, pos.y + h, pos.z + hw];
  const o = [start.x, start.y, start.z];
  const d = [dir.x, dir.y, dir.z];

  for (let a = 0; a < 3; a++) {
    if (Math.abs(d[a]) < 1e-12) {
      if (o[a] < lo[a] || o[a] > hi[a]) return -1;   // 平行且在板外
      continue;
    }
    const inv = 1 / d[a];
    let t0 = (lo[a] - o[a]) * inv;
    let t1 = (hi[a] - o[a]) * inv;
    if (t0 > t1) { const s = t0; t0 = t1; t1 = s; }
    if (t0 > tmin) tmin = t0;
    if (t1 < tmax) tmax = t1;
    if (tmin > tmax) return -1;
  }
  return tmin;
}

/** 调试 / 测试用：拿到第 i 条瞄准射线的内部对象。 */
export function aimRayAt(i) {
  return aimRays[i] || null;
}

export function setAimRaysVisible(v) {
  aimRaysVisible = v;
  if (!v) for (const r of aimRays) { r.line.visible = false; r.mark.visible = false; }
}

// ── 几何可观测性：眼睛 → 目标碰撞盒 8 个角落的调试射线 ───────────────────────
//
// `visibility.js` 判定「B 能不能看到 A」时，真正测的是 眼睛 → A 的碰撞盒 8 个
// 角落的 8 条射线。这一层把**选中的那个观察者**的 9×8 = 72 条全画出来，颜色就是
// 判定结果：
//   绿 = 通（这一条就是"看得见"的依据）
//   红 = 被地图三角形挡
//   橙 = 被烟雾体素挡
//   灰 = 不在视锥里（根本没发射线）
//   紫 = 被别的玩家的碰撞盒挡（只在 blockPlayers 打开时会出现）
//
// 一根 LineSegments 装下 10 个观察者 × 9 个目标 × 8 个角落 = 720 段的上界，
// 用 drawRange 控制实际画几段 —— 不重建 geometry、不重新分配。

/** 角落射线的颜色（线性 sRGB 三元组），按 `visibility.js` 的 BLOCK_* 位分。 */
export const VIS_CORNER_COLORS = {
  clear: [0.30, 1.00, 0.40],
  fov: [0.45, 0.45, 0.52],
  map: [1.00, 0.28, 0.22],
  smoke: [1.00, 0.62, 0.10],
  player: [0.85, 0.35, 0.95],
};

const VIS_CORNER_MAX = 10 * 9 * 8;
const visCornerPos = new Float32Array(VIS_CORNER_MAX * 6);
const visCornerCol = new Float32Array(VIS_CORNER_MAX * 6);
let visCornerLines = null;
let visCornerWanted = false;
let visCornerCount = 0;

export function createVisCornerRays() {
  if (visCornerLines) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position',
    new THREE.BufferAttribute(visCornerPos, 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute('color',
    new THREE.BufferAttribute(visCornerCol, 3).setUsage(THREE.DynamicDrawUsage));
  geo.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.9,
    depthTest: false, depthWrite: false,
  });
  visCornerLines = new THREE.LineSegments(geo, mat);
  visCornerLines.name = 'vis-corner-rays';
  visCornerLines.frustumCulled = false;    // 顶点每帧全变，包围球没意义
  visCornerLines.renderOrder = 660;
  visCornerLines.visible = false;
  scene.add(visCornerLines);
}

export function clearVisCornerRays() {
  if (!visCornerLines) return;
  scene.remove(visCornerLines);
  visCornerLines.geometry.dispose();
  visCornerLines.material.dispose();
  visCornerLines = null;
  visCornerCount = 0;
}

/**
 * 灌入这一帧要画的线段（**game 空间**）。
 * @param {Array<{o:number[], t:number[], color:number[]}>} segs
 */
export function updateVisCornerRays(segs) {
  if (!visCornerLines) return;
  const n = Math.min(segs ? segs.length : 0, VIS_CORNER_MAX);
  for (let i = 0; i < n; i++) {
    const s = segs[i];
    const a = gameToThree(s.o[0], s.o[1], s.o[2]);
    const b = gameToThree(s.t[0], s.t[1], s.t[2]);
    const p = i * 6;
    visCornerPos[p] = a.x; visCornerPos[p + 1] = a.y; visCornerPos[p + 2] = a.z;
    visCornerPos[p + 3] = b.x; visCornerPos[p + 4] = b.y; visCornerPos[p + 5] = b.z;
    const c = s.color;
    visCornerCol[p] = c[0]; visCornerCol[p + 1] = c[1]; visCornerCol[p + 2] = c[2];
    visCornerCol[p + 3] = c[0]; visCornerCol[p + 4] = c[1]; visCornerCol[p + 5] = c[2];
  }
  visCornerCount = n;
  const attrs = visCornerLines.geometry.attributes;
  attrs.position.needsUpdate = true;
  attrs.color.needsUpdate = true;
  visCornerLines.geometry.setDrawRange(0, n * 2);
  visCornerLines.visible = visCornerWanted && n > 0;
}

export function setVisCornerRaysVisible(v) {
  visCornerWanted = !!v;
  if (visCornerLines) visCornerLines.visible = visCornerWanted && visCornerCount > 0;
}

/** 调试 / 测试用。 */
export function visCornerRayDebug() {
  return {
    created: !!visCornerLines,
    visible: !!(visCornerLines && visCornerLines.visible),
    count: visCornerCount,
    max: VIS_CORNER_MAX,
    drawnVertices: visCornerLines ? visCornerLines.geometry.drawRange.count : 0,
  };
}

// ── 几何可观测性：视锥可视化 ────────────────────────────────────────────────
//
// 从眼睛沿瞄准方向画一个矩形截面的四棱锥：4 条从眼睛出发的棱 + 远端 4 条边。
// 远端截面在沿 forward **D 米**处，半宽 `D·tan(水平半角)`、半高 `D·tan(垂直半角)`
// —— 和 `visibility.js` 判定角落时用的是**同一组半角**（90° / 16:9），所以画出来
// 的就是判定用的那个锥，不是另画一个"示意"。
//
// 只画**选中的那一个玩家**的锥：10 个锥叠在一起没法看。颜色用暖黄，和瞄准射线
// （青/琥珀/红）、角落射线（绿/红/橙/灰/紫）都不撞。

/** 视锥画多长（米）。太长会盖满整张图，太短看不出张角。 */
export const FRUSTUM_DIST = 25;
/** 和 visibility.js 的 DEFAULT_FOV_DEG / DEFAULT_ASPECT 保持一致。 */
const FRUSTUM_FOV_DEG = 90;
const FRUSTUM_ASPECT = 16 / 9;

const FRUSTUM_SEGS = 8;                       // 4 条棱 + 远端 4 条边
const frustumPos = new Float32Array(FRUSTUM_SEGS * 6);
let frustumLines = null;
let frustumWanted = false;
let frustumDrawn = false;

export function createViewFrustum() {
  if (frustumLines) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position',
    new THREE.BufferAttribute(frustumPos, 3).setUsage(THREE.DynamicDrawUsage));
  const mat = new THREE.LineBasicMaterial({
    color: 0xffe066, transparent: true, opacity: 0.55,
    depthTest: false, depthWrite: false,
  });
  frustumLines = new THREE.LineSegments(geo, mat);
  frustumLines.name = 'view-frustum';
  frustumLines.frustumCulled = false;         // 顶点每帧全变，包围球没意义
  frustumLines.renderOrder = 640;
  frustumLines.visible = false;
  scene.add(frustumLines);
}

export function clearViewFrustum() {
  if (!frustumLines) return;
  scene.remove(frustumLines);
  frustumLines.geometry.dispose();
  frustumLines.material.dispose();
  frustumLines = null;
  frustumDrawn = false;
}

const _fEye = new THREE.Vector3();
const _fFwd = new THREE.Vector3();
const _fRight = new THREE.Vector3();
const _fUp = new THREE.Vector3();
const _fWorldUp = new THREE.Vector3(0, 1, 0);
const _fCorner = [new THREE.Vector3(), new THREE.Vector3(),
  new THREE.Vector3(), new THREE.Vector3()];

/**
 * 按某个玩家的朝向重画视锥。
 * @param {object} state 玩家状态；null / 死了 / 开关关着 → 隐藏
 * @param {{distMeters?:number, fovDeg?:number, aspect?:number}} [opts]
 */
export function updateViewFrustum(state, opts = {}) {
  if (!frustumLines) return;
  if (!frustumWanted || !state || !state.alive) {
    frustumLines.visible = false;
    frustumDrawn = false;
    return;
  }
  const distM = opts.distMeters != null ? opts.distMeters : FRUSTUM_DIST;
  const tanH = Math.tan((opts.fovDeg != null ? opts.fovDeg : FRUSTUM_FOV_DEG)
    * Math.PI / 360);
  const tanV = tanH / (opts.aspect != null ? opts.aspect : FRUSTUM_ASPECT);

  // 起点/朝向和瞄准射线共用 eyeRay()，保证"画出来的锥"和"射出去的线"同源
  const { start, forward } = eyeRay(state);
  _fEye.copy(start);
  _fFwd.copy(forward);
  // right = worldUp × forward。正负号对**对称**的锥没影响，所以不纠结方向。
  _fRight.crossVectors(_fWorldUp, _fFwd);
  if (_fRight.lengthSq() < 1e-12) _fRight.set(1, 0, 0);   // pitch = ±90° 退化
  else _fRight.normalize();
  _fUp.crossVectors(_fFwd, _fRight);

  const hw = distM * tanH;
  const hv = distM * tanV;
  const SGN = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  for (let i = 0; i < 4; i++) {
    _fCorner[i].copy(_fEye)
      .addScaledVector(_fFwd, distM)
      .addScaledVector(_fRight, SGN[i][0] * hw)
      .addScaledVector(_fUp, SGN[i][1] * hv);
  }

  let w = 0;
  const put = (a, b) => {
    frustumPos[w++] = a.x; frustumPos[w++] = a.y; frustumPos[w++] = a.z;
    frustumPos[w++] = b.x; frustumPos[w++] = b.y; frustumPos[w++] = b.z;
  };
  for (let i = 0; i < 4; i++) put(_fEye, _fCorner[i]);                 // 4 条棱
  for (let i = 0; i < 4; i++) put(_fCorner[i], _fCorner[(i + 1) % 4]); // 远端 4 条边

  frustumLines.geometry.attributes.position.needsUpdate = true;
  frustumLines.visible = true;
  frustumDrawn = true;
}

export function setViewFrustumVisible(v) {
  frustumWanted = !!v;
  if (!frustumWanted && frustumLines) frustumLines.visible = false;
}

/** 调试 / 测试用：远端 4 个角的世界坐标（米）+ 张角，用来独立验算。 */
export function viewFrustumDebug() {
  const corners = [];
  if (frustumDrawn) {
    for (let i = 0; i < 4; i++) {
      corners.push([_fCorner[i].x, _fCorner[i].y, _fCorner[i].z]);
    }
  }
  return {
    created: !!frustumLines,
    visible: !!(frustumLines && frustumLines.visible),
    drawn: frustumDrawn,
    segments: frustumDrawn ? FRUSTUM_SEGS : 0,
    distMeters: FRUSTUM_DIST,
    fovDeg: FRUSTUM_FOV_DEG,
    aspect: FRUSTUM_ASPECT,
    eye: frustumDrawn ? [_fEye.x, _fEye.y, _fEye.z] : null,
    corners,
  };
}

// ── 子弹曳光 ────────────────────────────────────────────────────────────────
//
// 每开一枪（`events.sound` 里 `ty === "fire"` 的那一发）从开枪者的**眼睛**沿那一帧的
// yaw/pitch 飞出一小段**曳光弹**，到最近的遮挡（人 / 墙）就停，停住后闪一下再淡出。
// 什么都没打到就飞满 60 m（和瞄准射线的暗青线同一个上限）再淡出。
//
// ⚠️ 弹道是**理想直线**：demo 里没有子弹实体，`fire_bullets` 那种
// 「origin + angles + seed + spread」也没有提取，所以方向只能是 yaw/pitch，
// 遮挡只能是「眼睛 → 第一个碰撞盒/三角形」。这**不是**真实弹道 ——
// 真实子弹有 spread、有穿透、有距离衰减，也不一定从眼睛出发。
// 和瞄准射线共用 `eyeRay()` / `castRay()`，所以同一帧同一人，曳光的终点和
// 瞄准射线的终点**必须**重合（测试就是拿这条当不变量的）。

/** 曳光段长度（米）。要比每帧的行程长，否则高速时会断成一节一节。 */
const TRACER_LEN = 5.0;
/** 飞行速度（米/秒，**真实时间**，不随播放倍速变 —— 4× 时跟倍速会被一帧跳过）。 */
const TRACER_SPEED = 200;
/** 打到东西后原地停留多久（秒），然后开始淡出。 */
const TRACER_HOLD = 0.06;
/** 淡出时长（秒）。 */
const TRACER_FADE = 0.16;
/** 池大小。全自动一梭子 + 4× 播放也够用。 */
const TRACER_MAX = 48;
/** 曳光段半径（米）。 */
const TRACER_RADIUS = 0.03;
/**
 * 视觉起点相对眼睛沿射线前移多少（米）。demo 里的开火方向就是**眼睛**的朝向，
 * 但曳光从眼球中心冒出来不好看 —— 画的时候沿**同一条射线**往前挪一点，
 * 所以终点不变，和瞄准射线仍然完全重合。
 */
const TRACER_MUZZLE_OFFSET = 0.35;

const tracerGeo = new THREE.CylinderGeometry(1, 1, 1, 6, 1, true);
const tracerMat = new THREE.MeshBasicMaterial({
  color: 0xffe08a, transparent: true, opacity: 0.95,
  blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false,
});

/** @type {Array<{mesh:THREE.Mesh,start:THREE.Vector3,dir:THREE.Vector3,dist:number,travel:number,renderOffset:number,head:number,tail:number,segLen:number,phase:string,rest:number,hit:boolean,hitPlayer:number,impactSpawned:boolean}>} */
const tracers = [];
let tracersVisible = true;

const _tq = new THREE.Quaternion();
const _tUp = new THREE.Vector3(0, 1, 0);
const _tMid = new THREE.Vector3();
const _tHit = new THREE.Vector3();

/** 从池里拿一个曳光（优先复用隐藏的；池满就抢飞得最远的那个）。 */
function acquireTracer() {
  let reuse = null, bestProgress = -1;
  for (const t of tracers) {
    if (!t.mesh.visible) return t;
    const p = t.dist > 0 ? t.travel / t.dist : 1;
    if (p > bestProgress) { bestProgress = p; reuse = t; }
  }
  if (tracers.length < TRACER_MAX) {
    const mesh = new THREE.Mesh(tracerGeo, tracerMat.clone());
    mesh.renderOrder = 660;
    mesh.visible = false;
    mesh.frustumCulled = false;
    scene.add(mesh);
    const t = {
      mesh,
      start: new THREE.Vector3(), dir: new THREE.Vector3(),
      dist: 0, travel: 0, renderOffset: 0,
      head: 0, tail: 0, segLen: 0,
      phase: 'fly', rest: 0,
      hit: false, hitPlayer: -1, impactSpawned: false,
    };
    tracers.push(t);
    return t;
  }
  return reuse;
}

/**
 * 把曳光摆到「头在 head 处、尾巴在 head − 段长」的位置（都从眼睛起算），
 * 并把实际用到的 head / tail / segLen 记在对象上（调试和测试读的就是这三个）。
 *
 * 段长固定 `TRACER_LEN`，但低帧率时一帧的行程可能超过它，那样画出来会是一节一节
 * 的虚线 —— 所以段长会临时撑到盖住这一帧的行程（上限 2×，免得变成一根几十米的
 * 激光）。尾巴不越过 `renderOffset`（枪口），所以头还没飞出枪口时是看不见的。
 */
function applyTracerTransform(t, dt) {
  const head = Math.min(t.travel, t.dist);
  const stretch = Math.min(TRACER_LEN * 2, Math.max(TRACER_LEN, TRACER_SPEED * (dt || 0) * 1.25));
  const tail = Math.max(t.renderOffset, head - stretch);
  const len = Math.max(0, head - tail);
  t.head = head;
  t.tail = tail;
  t.segLen = len;
  // 头还没飞出枪口（len = 0）时也要把位置写好：这样 head / tail / segLen /
  // position / scale 五者永远自洽，调试和测试读到的就是画出来的。
  _tMid.copy(t.start).addScaledVector(t.dir, tail + len / 2);
  t.mesh.position.copy(_tMid);
  _tq.setFromUnitVectors(_tUp, t.dir);
  t.mesh.quaternion.copy(_tq);
  t.mesh.scale.set(len > 1e-4 ? TRACER_RADIUS : 0, len, len > 1e-4 ? TRACER_RADIUS : 0);
}

/**
 * 生成一发曳光弹：从 `playerStates[playerIdx]` 的眼睛沿瞄准方向射出，
 * 在最近的遮挡（人 / 墙）处停；没打到就飞满 `maxDist`。
 *
 * @returns {object|null} 生成的曳光（调试/测试可读），没生成返回 null
 */
export function spawnTracer(playerStates, playerIdx, opts = {}) {
  const state = playerStates && playerStates[playerIdx];
  if (!state || !state.alive) return null;

  const { start, forward } = eyeRay(state);
  const maxDist = opts.maxDist != null ? opts.maxDist : AIM_MAX_DIST;
  const { distance, hitPlayer } = castRay(playerStates, playerIdx, start, forward, maxDist);

  const t = acquireTracer();
  t.start.copy(start);
  t.dir.copy(forward);
  t.dist = Math.max(0, distance);
  // 贴脸打墙时（dist 很小）别把起点挪到墙后面去
  t.renderOffset = Math.min(TRACER_MUZZLE_OFFSET, t.dist * 0.5);
  t.travel = 0;
  t.head = 0;
  t.tail = 0;
  t.segLen = 0;
  t.phase = 'fly';
  t.rest = 0;
  t.hit = hitPlayer >= 0 || distance < maxDist;
  t.hitPlayer = hitPlayer;
  t.impactSpawned = false;
  t.mesh.material.opacity = 0.95;
  t.mesh.visible = tracersVisible;
  applyTracerTransform(t, 0);
  return t;
}

// ── 撞击点闪光 ──────────────────────────────────────────────────────────────

const impactGeo = new THREE.SphereGeometry(1, 8, 6);
const IMPACT_LIFE = 0.18;
const IMPACT_MAX = 32;
const impactPool = [];

/** 命中点的小闪光：打人偏红、打墙琥珀（和瞄准射线的撞击点同色系）。 */
function spawnImpact(pos, onPlayer) {
  let s = impactPool.find(x => !x.visible);
  if (!s) {
    if (impactPool.length >= IMPACT_MAX) {
      s = impactPool[0];
    } else {
      s = new THREE.Mesh(impactGeo, new THREE.MeshBasicMaterial({
        color: 0xffb020, transparent: true, opacity: 1,
        depthTest: false, depthWrite: false,
      }));
      s.renderOrder = 661;
      s.userData = { life: 0 };
      scene.add(s);
      impactPool.push(s);
    }
  }
  s.material.color.set(onPlayer ? 0xff6a5a : 0xffb020);
  s.position.copy(pos);
  s.scale.setScalar(0.08);
  s.material.opacity = 1;
  s.userData.life = IMPACT_LIFE;
  s.visible = true;
}

function updateImpacts(dt) {
  for (const s of impactPool) {
    if (!s.visible) continue;
    s.userData.life -= dt;
    if (s.userData.life <= 0) { s.visible = false; continue; }
    const k = s.userData.life / IMPACT_LIFE;   // 1 → 0
    s.scale.setScalar(0.08 + (1 - k) * 0.28);
    s.material.opacity = k;
  }
}

/**
 * 推进所有曳光（顺便推进撞击点闪光）。每帧调一次。
 *
 * 三个阶段：`fly`（头往前飞，尾巴跟着）→ `hold`（头停在命中点）→ `fade`（淡出）。
 */
export function updateTracers(dt) {
  const step = dt > 0 ? dt : 0.016;
  for (const t of tracers) {
    if (!t.mesh.visible) continue;

    if (t.phase === 'fly') {
      t.travel += TRACER_SPEED * step;
      if (t.travel >= t.dist) {
        // 到了命中点就**停**在这里（这就是"碰到人或者障碍物就停"）
        t.travel = t.dist;
        t.phase = 'hold';
        t.rest = 0;
        // 停住时把段长收成标称长度：低帧率下飞行途中可能被撑长过
        applyTracerTransform(t, 0);
        if (t.hit && !t.impactSpawned) {
          t.impactSpawned = true;
          spawnImpact(_tHit.copy(t.start).addScaledVector(t.dir, t.dist), t.hitPlayer >= 0);
        }
      } else {
        applyTracerTransform(t, step);
      }
      continue;
    }

    // hold → fade 用一条 rest 时间轴，而不是逐帧减 life：dt 再大也能算出
    // 正确的透明度，不会因为一次掉帧就把整个淡出跳过去。
    t.rest += step;
    const k = t.rest <= TRACER_HOLD ? 1 : 1 - (t.rest - TRACER_HOLD) / TRACER_FADE;
    if (k <= 0) { t.mesh.visible = false; continue; }
    t.phase = t.rest <= TRACER_HOLD ? 'hold' : 'fade';
    t.mesh.material.opacity = 0.95 * k;
  }
  updateImpacts(step);
}

export function setTracersVisible(v) {
  tracersVisible = !!v;
  if (!v) clearTracers();
}

/** 把在飞的曳光 / 撞击点全部收掉（换回合、关开关）。 */
export function clearTracers() {
  for (const t of tracers) t.mesh.visible = false;
  for (const s of impactPool) s.visible = false;
}

/** 单条曳光的调试信息（真实几何，不是数据）。`head` / `tail` 都从眼睛起算。 */
export function tracerInfo(t) {
  return {
    phase: t.phase,
    start: [t.start.x, t.start.y, t.start.z],
    dir: [t.dir.x, t.dir.y, t.dir.z],
    dist: t.dist,
    travel: t.travel,
    renderOffset: t.renderOffset,
    tail: t.tail,
    head: t.head,
    // segLen 是**画出来的**段长（低帧率下会被撑长），nominalSegLen 是标称长度
    segLen: t.segLen,
    nominalSegLen: TRACER_LEN,
    meshSegLen: t.mesh.scale.y,
    radius: t.mesh.scale.x,
    position: [t.mesh.position.x, t.mesh.position.y, t.mesh.position.z],
    hit: t.hit,
    hitPlayer: t.hitPlayer,
    opacity: t.mesh.material.opacity,
  };
}

/** 调试 / 测试用：当前在飞的曳光（真实几何，不是数据）。 */
export function tracerDebugStats() {
  const out = [];
  for (const t of tracers) {
    if (!t.mesh.visible) continue;
    out.push(tracerInfo(t));
  }
  return out;
}

// ── X-ray ───────────────────────────────────────────────────────────────────
//
// 这里有两个长得很像、但管的东西完全不同的开关：
//
//   `透视地图`（`setXRayMode`）        把**地图**换成 15% 不透明的灰色基本材质，
//                                      所有小人一起变成"不测深度 + 自发光"
//   `选手 X 光`（`setPlayerXRayMode`） 地图一动不动，只让**被挡住的那几个选手**
//                                      用半透明幽灵材质画出来；直接看得见的照旧
//                                      是不透明的正常外观
//
// 判定在 player-xray.js（纯几何，从**摄像机**量谁被墙挡住了），这里只管外观。
//
// ⚠️ 两个开关都会写小人的材质和 `renderOrder`，所以规则只有一条：**贴材质的动作
// 只有一处**（`applyFigureMaterials()`，由 `updatePlayers()` 每帧调，而且只在透视
// 地图**关着**时调）。透视地图开着时材质和绘制顺序整个交给 `setXRayMode()` 那趟
// 遍历；开/关的前后各贴一次，免得遍历把"当前材质"记成幽灵材质 —— 那样关掉透视
// 地图之后还原回来的还是幽灵。

const xrayMapMaterial = new THREE.MeshBasicMaterial({
  color: 0x5a6a7a, transparent: true, opacity: 0.15, depthWrite: true,
});

/** 幽灵（只能透视看到）的不透明度 / 自发光强度 / 绘制顺序。 */
const XRAY_GHOST_OPACITY = 0.42;
const XRAY_GHOST_EMISSIVE = 0.5;
const XRAY_GHOST_RENDER_ORDER = 980;

let playerXRayEnabled = true;     // `选手 X 光`，默认开
/** @type {Uint8Array|null} 每个玩家：1 = 被挡住，只能透视看到（player-xray.js 算的） */
let playerXRayFlags = null;

/**
 * 幽灵材质 = 正常材质的克隆 + 四个改动。
 *
 * **必须克隆**：腿 / 头 / 面罩 / 视线锥用的是 10 个人共享的材质，就地改会把
 * 所有选手一起变成幽灵。克隆出来的这份只属于一个人的一个部件，改它谁也牵连不到。
 *
 * 关掉深度测试才会画到墙上面去 —— 这就是"透视"的字面意思。不写深度缓冲，
 * 于是幽灵之间不互相遮挡，也不会在别人的血条 / 名字上留个洞。
 *
 * ⚠️ 这里设的颜色只是**出生时的样子**，不保证对：`createPlayers()` 拿不到这
 * 一局的分队（`teams[]` 是每回合的数据），只能按 `i < 5` 猜一个底色。真正的
 * 队色由 `applyFigureMaterials()` **每次贴幽灵时从源材质同步过来** —— 见那里的
 * 注释（不这么做就是"CT 的幽灵发黄光、T 的幽灵发蓝光"那个 bug）。
 */
function makeGhostMaterial(src) {
  const m = src.clone();
  m.transparent = true;
  m.opacity = XRAY_GHOST_OPACITY;
  m.depthTest = false;
  m.depthWrite = false;
  // 自发光拉到和本体色一样：不这样的话幽灵全是阴影里的黑块（光照打在墙上，
  // 不透明的小人却在地图外面）—— 一眼看不出是幽灵还是漏画了。
  if (m.emissive) {
    m.emissive.copy(m.color);
    m.emissiveIntensity = XRAY_GHOST_EMISSIVE;
  }
  return m;
}

/** 第 i 个选手现在算不算"只能透视看到"。透视地图开着时一律不算（另有外观）。 */
function ghostedAt(i) {
  return !xrayEnabled && playerXRayEnabled
    && !!playerXRayFlags && playerXRayFlags[i] === 1;
}

/**
 * 把第 i 个小人身上的部件贴成"正常"或"幽灵"。
 *
 * @param {number} i
 * @param {boolean} ghost 由调用方算好（`ghostedAt(i)`），这样"开透视地图之前
 *        先贴一遍正常材质"也能走同一条路，不用临时去翻模块状态。
 */
function applyFigureMaterials(i, ghost) {
  const pg = playerGroups[i];
  if (!pg || !pg.figure) return;
  for (const f of pg.figure) {
    f.mesh.material = ghost ? f.ghost : f.normal;
    // 幽灵要画在所有不透明物之后；999 留给透视地图那趟遍历。
    f.mesh.renderOrder = ghost ? XRAY_GHOST_RENDER_ORDER : 0;
    if (ghost) {
      // 每次贴上幽灵都把它的四个属性**重置一遍**。这不是多余的：材质是"活在"
      // 网格上的，别的代码只要写了 `pg.body.material.opacity`（不透明的老代码、
      // 闪光）就会写进上一帧留在这里的幽灵材质里，而且**不会自己变回来**。
      // 浏览器测试就是这么抓到"幽灵的躯干悄悄变成实心"的（test_player_xray_render D）。
      f.ghost.opacity = XRAY_GHOST_OPACITY;
      f.ghost.transparent = true;
      f.ghost.depthTest = false;
      f.ghost.depthWrite = false;
      // ── 颜色从**源材质现取**，绝不用克隆那一刻的快照 ──────────────────────
      // 躯干 / 双臂 / 头盔的源材质（`teamMat`）每帧都被灌进这一回合的真实队色，
      // 所以幽灵跟着它走就一定对；而克隆时的底色是 `createPlayers()` 按 `i < 5`
      // 猜的，和 `teams[]` 没有任何关系（实测一个回合的分队是
      // `['T','CT','T','CT',…]` 这种交错，不是"前五个一队"）。
      //
      // 这个 bug 长这样：幽灵的**自发光**是 0.5 强度的本色，占到画面里的一大半，
      // 于是按 `i < 5` 猜错了队的那些幽灵会整体发错色的光 —— 蓝色 CT 的幽灵发
      // 黄光、黄色 T 的幽灵发蓝光，看起来就像两队颜色被对调了。非幽灵状态看不
      // 出来（队色每帧写在"当前材质"上），所以只有 `选手 X 光` 会露馅。
      f.ghost.color.copy(f.normal.color);
      if (f.ghost.emissive) {
        f.ghost.emissive.copy(f.normal.color);
        f.ghost.emissiveIntensity = XRAY_GHOST_EMISSIVE;
      }
    }
  }
}

function applyAllFigureMaterials() {
  for (let i = 0; i < playerGroups.length; i++) applyFigureMaterials(i, ghostedAt(i));
}

/**
 * `选手 X 光` 开关。默认**开**：不这样的话，站在墙后的选手在画面上完全不存在，
 * 而这类回放里"这一枪是隔着什么打的"恰恰是最要看的东西。
 */
export function setPlayerXRayMode(enabled) {
  playerXRayEnabled = !!enabled;
  if (!playerXRayEnabled) playerXRayFlags = null;
  // 透视地图开着时不插手（材质归它管）
  if (!xrayEnabled) applyAllFigureMaterials();
}

/**
 * 灌入 player-xray.js 的判定结果（`Uint8Array`，1 = 只能透视看到）。
 * 传 null = 取消所有幽灵（开关关掉 / 还没算）。
 */
export function setPlayerXRayFlags(flags) {
  playerXRayFlags = (flags && flags.length) ? flags : null;
}

/** 调试 / 测试用：每个选手的幽灵状态（真实材质，不是数据）。 */
export function playerXRayDebug() {
  const out = [];
  for (let i = 0; i < playerGroups.length; i++) {
    const fig = playerGroups[i].figure || [];
    let ghostParts = 0;
    for (const f of fig) if (f.mesh.material === f.ghost) ghostParts++;
    out.push({
      enabled: playerXRayEnabled,
      flag: playerXRayFlags ? (playerXRayFlags[i] || 0) : 0,
      ghosted: ghostedAt(i),
      parts: fig.length,
      ghostParts,
      // 读**画出来的**材质，不是标志位
      opacity: fig.length ? fig[0].mesh.material.opacity : null,
      depthTest: fig.length ? fig[0].mesh.material.depthTest : null,
      renderOrder: fig.length ? fig[0].mesh.renderOrder : null,
      // 每个部件的不透明度：腿 / 头 / 面罩 / 视线锥是 10 个人**共享**材质，
      // 这里逐部件列出来，才能证明"幽灵没有把共享材质也改掉"。
      opacities: fig.map((f) => f.mesh.material.opacity),
      // 画出来的**颜色 / 自发光**（十六进制）。幽灵这两样必须跟着这一回合的
      // `teams[]` 走 —— 曾经是按 `i < 5` 猜的底色，于是蓝色 CT 的幽灵发黄光、
      // 黄色 T 的幽灵发蓝光（`test_player_xray_render.mjs` §F）。
      colors: fig.map((f) => f.mesh.material.color.getHex()),
      emissives: fig.map((f) => (f.mesh.material.emissive
        ? f.mesh.material.emissive.getHex() : null)),
    });
  }
  return out;
}

export function setXRayMode(enabled) {
  if (enabled) {
    // 先把幽灵收回去：下面那趟遍历会把"当前材质"记成 `_origMat`，记成幽灵材质
    // 的话，关掉透视地图之后还原回来的就是幽灵 —— 而且再也回不到正常外观。
    for (let i = 0; i < playerGroups.length; i++) applyFigureMaterials(i, false);
  }
  xrayEnabled = enabled;
  if (mapGroup) {
    mapGroup.traverse((child) => {
      if (child.isMesh && child.material && !child.material.isLineBasicMaterial) {
        if (enabled) {
          child.userData._origMat = child.material;
          child.material = xrayMapMaterial;
          child.renderOrder = 900;
        } else if (child.userData._origMat) {
          child.material = child.userData._origMat;
          child.renderOrder = 0;
        }
      }
    });
  }
  for (const pg of playerGroups) {
    if (!pg.group) continue;
    // 小人身上的所有部件都要处理：躯干/手臂/头盔用每人一份的 teamMat，
    // 腿/面罩/头/视线锥用全局共享材质。xray 是全局开关，所以对共享材质
    // 反复赋同样的值是幂等的，不用去重。
    for (const root of [pg.bodyGroup, pg.pitchGroup]) {
      if (!root) continue;
      root.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        const m = o.material;
        // 自发光要能还原：面罩和视线锥本来就有底光，关掉 xray 不能把它抹掉。
        // 共享材质会被反复访问，但记下来的底值总是一样的。
        if (m.userData._baseEmissive === undefined) {
          m.userData._baseEmissive = m.emissive ? m.emissive.clone() : null;
          m.userData._baseEmissiveIntensity = m.emissiveIntensity || 0;
        }
        o.renderOrder = enabled ? 999 : 0;
        m.depthTest = !enabled;
        m.depthWrite = !enabled;
        if (m.emissive) {
          if (enabled) {
            m.emissive.copy(m.color);
            m.emissiveIntensity = 0.8;
          } else {
            m.emissive.copy(m.userData._baseEmissive);
            m.emissiveIntensity = m.userData._baseEmissiveIntensity;
          }
        }
      });
    }
  }
  // 关掉透视地图：上面那趟遍历把 `_origMat`（正常材质）还原回来了，但"被挡住
  // 的人"这时候应该是幽灵 —— 再按当前判定贴一遍，不用等下一帧。
  // 开着的时候不贴：材质整个归上面那趟遍历管。
  if (!enabled) applyAllFigureMaterials();
}

export function setPlayerModelVisible(idx, visible) {
  if (idx >= 0 && idx < playerGroups.length) {
    playerGroups[idx].group.visible = visible;
    const t = playerTrails[idx];
    if (t && t.line) t.line.visible = visible;
  }
}

// ── 投掷物 ──────────────────────────────────────────────────────────────────
//
// 飞行中的投掷物画的是**游戏里的真实 GLB 模型**，不是以前那个球。
//
// ## 尺寸：模型放大 2.5 倍，但这比原来那个球**更接近真实**
//
// 模型本身是米，不需要缩放就能用。但真实的投掷物只有 5–25 cm
// （实测包围盒：烟雾弹 0.053×0.118×0.059 m、莫洛托夫 0.091×0.250×0.210 m），
// 在整张地图的视距下只有两三个像素。而**原来那个球半径 0.2 m（直径 0.4 m）**，
// 是真实烟雾弹的 3.4 倍。所以放大 2.5 倍后（烟雾弹 0.30 m 高）反而比旧的球
// 更接近真实尺寸 —— 想要严格 1:1 就把 GRENADE_MODEL_SCALE 改成 1。
const GRENADE_MODEL_SCALE = 2.5;
/** 模型加载完 / 关掉"投掷物模型"时用的兜底球（= 改造前的老样子）。 */
const nadeDotGeo = new THREE.SphereGeometry(0.2, 10, 10);
/** 6 个投掷物模型的长轴实测都是 **+Y**（见 check_nade_models.mjs 的断言）。 */
const NADE_LONG_AXIS = new THREE.Vector3(0, 1, 0);
const _nadeDir = new THREE.Vector3();

let grenadeTrajectories = [];
let nadeModelsVisible = true;

// ── C4 ──────────────────────────────────────────────────────────────────────

let bombGroup = null;
let bombPulseLight = null;

// ── 玩家创建 ────────────────────────────────────────────────────────────────

export function createPlayers() {
  ensureGeometries();
  clearPlayers();

  for (let i = 0; i < 10; i++) {
    const group = new THREE.Group();
    group.name = `player-${i}`;
    group.visible = false;

    // 每人一份的队服材质：躯干 / 手臂 / 头盔共用同一个实例，所以
    // 换队色和闪光只需要改一处（`pg.teamMat` 就是它）。
    //
    // ⚠️ `i < 5 ? matCT : matT` 只是**底色占位**：`.csz` 里一个回合的分队是
    // `teams[i]`，实测可以是 `['T','CT','T','CT',…]`，和下标没有任何关系。
    // 真正对得上的队色每帧由 `updatePlayers()` 写进 `pg.teamMat`；幽灵也是从
    // 它同步的（见 `applyFigureMaterials()`）。
    const teamMat = (i < 5 ? matCT : matT).clone();

    // ── 躯干组：腿 + 躯干 + 双臂 ─────────────────────────────────────────
    // ⚠️ 这一组**自己不缩放**。蹲下的 Y 压缩改成逐个网格设 scale.y ——
    // 因为腿现在要绕胯旋转，而"父节点非等比缩放 + 子节点旋转"会产生**剪切**：
    // 腿会被压歪、脚会被压扁成椭圆。缩放放在旋转的**同一节点或更内层**就没问题
    // （three 的局部矩阵是 T·R·S，S 先作用），所以每根腿是
    // legPivot(rotation) → leg(position + scale)，缩放比旋转更内层。
    const bodyGroup = new THREE.Group();
    bodyGroup.name = `body-${i}`;
    group.add(bodyGroup);

    const torso = new THREE.Mesh(torsoGeo, teamMat);
    torso.position.y = (P_LEG_TOP + P_TORSO_TOP) / 2;
    torso.castShadow = true;
    bodyGroup.add(torso);

    const armL = new THREE.Mesh(armGeo, teamMat);
    armL.position.set(-0.30, P_ARM_Y, 0);
    armL.castShadow = true;
    bodyGroup.add(armL);
    const armR = armL.clone();          // 共用 geometry / material
    armR.position.x = 0.30;
    bodyGroup.add(armR);

    // 腿：每根挂在一个**绕胯旋转**的 pivot 上。pivot 在胯高，腿从 pivot 往下挂。
    const legPivotL = new THREE.Group();
    legPivotL.name = `legpivot-l-${i}`;
    legPivotL.position.set(-0.13, P_LEG_TOP, 0);
    const legL = new THREE.Mesh(legGeo, matGear);
    legL.position.y = -P_LEG_HALF;      // 腿顶对齐 pivot
    legL.castShadow = true;
    legPivotL.add(legL);
    bodyGroup.add(legPivotL);

    const legPivotR = new THREE.Group();
    legPivotR.name = `legpivot-r-${i}`;
    legPivotR.position.set(0.13, P_LEG_TOP, 0);
    const legR = new THREE.Mesh(legGeo, matGear);
    legR.position.y = -P_LEG_HALF;
    legR.castShadow = true;
    legPivotR.add(legR);
    bodyGroup.add(legPivotR);

    // ── 头部组：绕**头心**俯仰，不再绕脚底甩 ────────────────────────────
    // 以前 pitchGroup 挂在原点、头在 y=1.6，俯仰 90° 时头会荡到身前 1.6 处。
    const pitchGroup = new THREE.Group();
    pitchGroup.name = `pitch-${i}`;
    pitchGroup.position.y = P_HEAD_Y;
    group.add(pitchGroup);

    const head = new THREE.Mesh(headGeo, matHead);
    head.castShadow = true;
    pitchGroup.add(head);

    const helmet = new THREE.Mesh(helmetGeo, teamMat);
    helmet.castShadow = true;
    pitchGroup.add(helmet);

    const visor = new THREE.Mesh(visorGeo, matVisor);
    visor.position.set(0, -0.005, P_HEAD_R - 0.03);
    pitchGroup.add(visor);

    const dir = new THREE.Mesh(dirGeo, matDir);
    dir.position.set(0, 0, 0.34);
    dir.rotation.x = Math.PI / 2;       // 圆锥默认朝 +Y，转成朝 +Z（前方）
    pitchGroup.add(dir);

    const hpBg = new THREE.Mesh(hpBarGeo, matHPBg);
    hpBg.position.y = P_HEAD_Y + P_HEAD_R + 0.07;
    hpBg.renderOrder = 1;
    group.add(hpBg);

    const hpFill = new THREE.Mesh(hpBarGeo, matHPHigh);
    hpFill.position.y = P_HEAD_Y + P_HEAD_R + 0.07;
    hpFill.position.z = 0.01;
    hpFill.renderOrder = 2;
    group.add(hpFill);

    const label = createLabelSprite('', i < 5 ? 0x5b9bd5 : 0xe8a240);
    label.position.y = P_HEAD_Y + P_HEAD_R + 0.27;
    label.scale.set(2.0, 0.5, 1);
    group.add(label);

    // 头顶 ID 牌（「选手 ID」开关）：和名字分开一个 sprite，见 ID_TAG_LIFT 的注释。
    // 底色先按 i < 5 猜，`updatePlayers()` 每帧按真实 `teams[i]` 纠正。
    const idTag = createIdTag(idTagTextFor(i), i < 5 ? 0x5b9bd5 : 0xe8a240);
    idTag.position.y = P_HEAD_Y + P_HEAD_R + ID_TAG_LIFT;
    idTag.visible = showPlayerIds;
    group.add(idTag);

    // 「选中玩家的可见敌人」眼睛：挂在 group 上（不挂 pitchGroup），
    // 这样它不会跟着俯仰角甩。高度每帧按头顶重算。
    const spotIcon = createSpotIcon();
    spotIcon.position.y = P_HEAD_Y + P_HEAD_R + SPOT_ICON_LIFT;
    group.add(spotIcon);

    // 武器 badge：**真实 CS2 官方 HUD 图标** + 名称。图标是异步解码的，
    // 所以先用文字占位，图标到了由 _loadWeaponBadgeIcon 重画（见其注释）。
    const weaponSprite = new THREE.Sprite(new THREE.SpriteMaterial({
      transparent: true, depthTest: false, depthWrite: false,
    }));
    weaponSprite.position.y = 0.15;
    weaponSprite.scale.set(1.5, 0.75, 1);   // badge 是 2:1，别压成正方形
    group.add(weaponSprite);

    // ── 下蹲调试层：碰撞盒棱线 + 眼睛 + 脚到眼的标尺 ────────────────────
    // 全部挂在 group（不挂 pitchGroup）上：眼睛位置不随俯仰角移动。
    const hull = new THREE.LineSegments(hullGeo, matHullCT);
    hull.renderOrder = 700;
    hull.visible = false;
    group.add(hull);

    const ruler = new THREE.Line(rulerGeo, matRuler);
    ruler.renderOrder = 700;
    ruler.visible = false;
    group.add(ruler);

    const eye = new THREE.Mesh(eyeGeo, matEye);
    eye.renderOrder = 701;
    eye.visible = false;
    group.add(eye);

    scene.add(group);

    const trailLine = new THREE.Line(
      playerTrails[i].geo,
      new THREE.LineBasicMaterial({ color: 0x5b9bd5, transparent: true, opacity: 0.5, linewidth: 1 })
    );
    trailLine.visible = true;
    trailLine.name = `trail-${i}`;
    // 包围球是几何体算一次就缓存的，而轨迹每帧都在变 —— 让它参与视锥剔除就会
    // 出现"镜头一转、整条轨迹消失"（球还停在第一个点上）。一条 150 点的线，
    // 剔除省不了什么。
    trailLine.frustumCulled = false;
    scene.add(trailLine);

    const deadX = new THREE.Group();
    deadX.name = `dead-x-${i}`;
    deadX.visible = false;
    const xBarGeo = new THREE.BoxGeometry(1.2, 0.12, 0.12);
    const xMat = new THREE.MeshBasicMaterial({ color: 0x444444, transparent: true, opacity: 0.9, depthTest: false });
    const bar1 = new THREE.Mesh(xBarGeo, xMat);
    bar1.rotation.y = Math.PI / 4;
    const bar2 = new THREE.Mesh(xBarGeo, xMat);
    bar2.rotation.y = -Math.PI / 4;
    deadX.add(bar1, bar2);
    deadX.renderOrder = 998;
    scene.add(deadX);

    // ── 选手 X 光的幽灵副本 ──────────────────────────────────────────────
    // 每个部件一份**自己的**幽灵材质。共享材质（腿 / 头 / 面罩 / 视线锥）就地改
    // 会把 10 个人一起变成幽灵，所以这里克隆。`figure` 就是 `updatePlayers()`
    // 每帧要贴材质的那份清单 —— 顺序就是遍历顺序（引擎侧无关，只是稳定）。
    const figure = [];
    for (const root of [bodyGroup, pitchGroup]) {
      root.traverse((o) => {
        if (!o.isMesh || !o.material || o.material.isLineBasicMaterial) return;
        figure.push({ mesh: o, normal: o.material, ghost: makeGhostMaterial(o.material) });
      });
    }

    playerGroups.push({
      group, bodyGroup, body: torso, pitchGroup, head, helmet, visor, dir,
      hpBg, hpFill, label, idTag, weaponSprite, spotIcon,
      hull, ruler, eye,
      torso, armL, armR, legL, legR, legPivotL, legPivotR,
      deadX, xMat, figure, teamMat, cachedGroundY: null,
    });

    playerTrails[i].line = trailLine;
  }

  createBomb();
}

function clearPlayers() {
  for (const pg of playerGroups) {
    scene.remove(pg.group);
    if (pg.deadX) scene.remove(pg.deadX);
  }
  for (const pt of playerTrails) {
    if (pt.line) { scene.remove(pt.line); pt.line = null; }
    pt.count = 0;
    pt.lastSample = -1;
    pt.geo.setDrawRange(0, 0);
  }
  playerGroups = [];
  if (bombGroup) { scene.remove(bombGroup); bombGroup = null; }
}

function createBomb() {
  bombGroup = new THREE.Group();
  bombGroup.name = 'bomb';
  bombGroup.visible = false;

  const box = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.3, 0.7),
    new THREE.MeshStandardMaterial({ color: 0x334422, roughness: 0.5, metalness: 0.2 })
  );
  box.castShadow = true;
  bombGroup.add(box);

  const kp = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 0.05, 0.4),
    new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.3, metalness: 0.5 })
  );
  kp.position.y = 0.18;
  bombGroup.add(kp);

  bombPulseLight = new THREE.Mesh(
    new THREE.SphereGeometry(0.08, 8, 8),
    new THREE.MeshBasicMaterial({ color: 0xff2222 })
  );
  bombPulseLight.position.y = 0.22;
  bombGroup.add(bombPulseLight);

  const ptLight = new THREE.PointLight(0xff0000, 2, 3);
  ptLight.position.y = 0.2;
  bombGroup.add(ptLight);

  scene.add(bombGroup);
}

function createLabelSprite(text, colorHex, fontSize = 12) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = `bold ${fontSize}px "SF Mono", "Consolas", monospace`;
  ctx.fillStyle = `#${colorHex.toString(16).padStart(6, '0')}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false, depthWrite: false,
  }));
  return sprite;
}

function updateLabelSprite(sprite, text, colorHex, fontSize = 12) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = `bold ${fontSize}px "SF Mono", "Consolas", monospace`;
  ctx.fillStyle = `#${colorHex.toString(16).padStart(6, '0')}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  sprite.material.map.dispose();
  sprite.material.map = tex;
  sprite.material.needsUpdate = true;
}

// ── 头顶的选手 ID（选手名字）──────────────────────────────────────────────
//
// 「选手 ID」开关画的就是它：写的是 `matchData.players[i].name`（取不到名字
// 才退回 01–10 的槽位号）。为什么用 `THREE.Sprite`：sprite 的朝向由渲染器
// 按摄像机现算，玩家转身、镜头绕到背后都不用逐帧改旋转 —— 这就是"永远朝向
// 摄像机"。
//
// 名字长短差很多（两三个字符到十几个都有），所以牌子的**宽度按文字实测**算、
// 高度固定，太长就逐级缩字号：固定 128px 会把长 ID 挤出框外。
//
// 和名字标签是两个独立对象：两个开关互不影响，ID 牌在名字上方一层（名字
// +0.27，ID +0.50）。渲染顺序 980 < 眼睛图标的 990 —— 同时开着的时候
// 眼睛图标仍然完整可见，ID 牌被压住一点也是可接受的。
const ID_TAG_LIFT = 0.50;
const ID_TAG_H = 64;              // 画布高度（像素）
const ID_TAG_WORLD_H = 0.22;      // 牌子在世界里的高度
const ID_TAG_MAX_W = 512;         // 再长就缩字号，别让牌子横穿半个地图
const ID_TAG_PAD_X = 26;          // 左右留白（含圆角）
/** 「选手 ID」开关的状态。默认开（和 index.html 里的 checked 一致）。 */
let showPlayerIds = true;
/** 槽位 → 选手名字，main.js 换局时灌进来。 */
let playerIdNames = [];

const _idTagMeasure = document.createElement('canvas').getContext('2d');

function idTagTextFor(i) {
  const name = playerIdNames[i];
  return (name && String(name).trim()) || String(i + 1).padStart(2, '0');
}

/** 按实测文字宽度定牌子宽度；太长就缩字号，保证不超过 ID_TAG_MAX_W。 */
function idTagMetrics(text) {
  let size = 38;
  let width = 0;
  for (;;) {
    _idTagMeasure.font = `bold ${size}px "SF Mono", "Consolas", monospace`;
    width = Math.ceil(_idTagMeasure.measureText(text).width) + ID_TAG_PAD_X * 2;
    if (width <= ID_TAG_MAX_W || size <= 20) break;
    size -= 2;
  }
  return { size, width: Math.max(128, Math.min(ID_TAG_MAX_W, width)) };
}

function paintIdTagCanvas(canvas, text, colorHex, fontSize = 38) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const color = `#${colorHex.toString(16).padStart(6, '0')}`;
  const r = 15;
  ctx.clearRect(0, 0, W, H);
  ctx.beginPath();
  ctx.moveTo(r + 2, 3);
  ctx.lineTo(W - r - 2, 3);
  ctx.quadraticCurveTo(W - 3, 3, W - 3, r);
  ctx.lineTo(W - 3, H - r);
  ctx.quadraticCurveTo(W - 3, H - 3, W - r, H - 3);
  ctx.lineTo(r + 2, H - 3);
  ctx.quadraticCurveTo(3, H - 3, 3, H - r);
  ctx.lineTo(3, r);
  ctx.quadraticCurveTo(3, 3, r, 3);
  ctx.closePath();
  // 深色底板：地图有亮面也有暗面，只画字的话在亮墙上会糊掉
  ctx.fillStyle = 'rgba(6, 11, 22, 0.78)';
  ctx.fill();
  ctx.lineWidth = 4;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.font = `bold ${fontSize}px "SF Mono", "Consolas", monospace`;
  ctx.fillStyle = '#f4f8ff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, W / 2, H / 2 + 2);
}

function createIdTag(text, colorHex) {
  const m = idTagMetrics(text);
  const canvas = document.createElement('canvas');
  canvas.width = m.width;
  canvas.height = ID_TAG_H;
  paintIdTagCanvas(canvas, text, colorHex, m.size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false, depthWrite: false,
  }));
  // 高度固定，宽度按画布比例走：字号在世界里大小一致，长名字只是牌子更宽
  sprite.scale.set(ID_TAG_WORLD_H * (m.width / ID_TAG_H), ID_TAG_WORLD_H, 1);
  sprite.renderOrder = 980;
  sprite.userData = { idText: text, colorHex, canvas };
  return sprite;
}

/** 文字 / 队色变了才重画贴图；宽度变了顺手改画布和 sprite 比例。 */
function refreshIdTag(sprite, text, colorHex) {
  const u = sprite.userData;
  if (u.idText === text && u.colorHex === colorHex) return;
  const m = idTagMetrics(text);
  if (u.canvas.width !== m.width) {
    u.canvas.width = m.width;
    u.canvas.height = ID_TAG_H;
    sprite.scale.x = ID_TAG_WORLD_H * (m.width / ID_TAG_H);
  }
  paintIdTagCanvas(u.canvas, text, colorHex, m.size);
  sprite.material.map.needsUpdate = true;
  u.idText = text;
  u.colorHex = colorHex;
}

/** 槽位 → 名字。换局时由 main.js 调用；名字到了一并刷新已经建好的牌子。 */
export function setPlayerIdNames(names) {
  playerIdNames = Array.isArray(names) ? names.slice() : [];
  for (let i = 0; i < playerGroups.length; i++) {
    const pg = playerGroups[i];
    if (pg.idTag) {
      refreshIdTag(pg.idTag, idTagTextFor(i), pg.idTag.userData.colorHex);
    }
  }
}

/** 「选手 ID」开关。关掉只是不画，不销毁贴图（再打开不用重建）。 */
export function setPlayerIdsVisible(visible) {
  showPlayerIds = !!visible;
  for (const pg of playerGroups) if (pg.idTag) pg.idTag.visible = showPlayerIds;
}

/** 测试 / 调试：每个玩家头顶 ID 牌真实画成了什么（文字、队色、显隐、位置）。 */
export function playerIdDebug() {
  return playerGroups.map((pg, i) => ({
    player: i,
    id: String(i + 1).padStart(2, '0'),
    name: playerIdNames[i] || '',
    text: pg.idTag ? pg.idTag.userData.idText : '',
    // visible = 这一帧**真的**画出来了（整组隐藏时它是 false）
    visible: !!(pg.idTag && pg.idTag.visible && pg.group.visible),
    flagVisible: !!(pg.idTag && pg.idTag.visible),
    sprite: !!(pg.idTag && pg.idTag.isSprite),
    scale: pg.idTag ? [pg.idTag.scale.x, pg.idTag.scale.y] : [],
    pos: pg.idTag ? [pg.idTag.position.x, pg.idTag.position.y, pg.idTag.position.z] : [],
  }));
}

// ── 武器 HUD 图标（真实 CS2 官方图标）─────────────────────────────────────
//
// 官方图标是 **512×384 的非正方**图，所以不能当正方形贴图贴 —— 按原始宽高比
// 画，否则枪会被横向拉扁。
//
// 图标是**异步**来的（首次要 fetch + 解码），而 `updatePlayers()` 是同步的。
// 所以这里走"先用文字占位、图标到了再重画一次"的路子，并且**按武器名缓存**：
// 10 个玩家常常拿同一把枪，只需解码一次。
const BADGE_W = 256, BADGE_H = 128;
const BADGE_ICON_H = 74;                 // 图标在 badge 里的高度
/** 武器显示名 -> {canvas, tex, state} */
const _weaponBadges = new Map();

function _paintBadge(entry, displayName, img, colorHex) {
  const ctx = entry.canvas.getContext('2d');
  ctx.clearRect(0, 0, BADGE_W, BADGE_H);
  entry.colorHex = colorHex;

  let textY = BADGE_H / 2;
  if (img && img.width && _weaponIconsVisible) {
    // 图标按自身宽高比缩放到 BADGE_ICON_H 高，水平居中
    const w = Math.min(BADGE_W - 8, BADGE_ICON_H * (img.width / img.height));
    const x = (BADGE_W - w) / 2;
    ctx.drawImage(img, x, 6, w, BADGE_ICON_H);
    textY = BADGE_ICON_H + 20;
  }

  const text = displayName.length > 14 ? `${displayName.slice(0, 13)}…` : displayName;
  ctx.font = 'bold 20px "SF Mono", "Consolas", monospace';
  ctx.fillStyle = `#${colorHex.toString(16).padStart(6, '0')}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, BADGE_W / 2, textY);

  entry.tex.needsUpdate = true;
}

function _getWeaponBadge(displayName, colorHex) {
  let entry = _weaponBadges.get(displayName);
  if (entry) return entry;

  const canvas = document.createElement('canvas');
  canvas.width = BADGE_W;
  canvas.height = BADGE_H;
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  entry = { canvas, tex, image: null, requested: false };
  _weaponBadges.set(displayName, entry);
  _paintBadge(entry, displayName, null, colorHex);
  return entry;
}

/** 图标到了之后重画所有用到这把枪的 badge。 */
function _loadWeaponBadgeIcon(displayName, colorHex) {
  const entry = _weaponBadges.get(displayName);
  if (!entry || entry.requested) return;
  entry.requested = true;
  weaponIconTexture(displayName).then((tex) => {
    if (!tex || !tex.image) return;      // 没有官方图标 → 保持文字
    entry.image = tex.image;
    _paintBadge(entry, displayName, entry.image, colorHex);
  }).catch(() => { /* 降级为文字，不报错 */ });
}

/** 换回合 / 换数据时清掉 badge 缓存。 */
export function clearWeaponBadges() {
  for (const [, e] of _weaponBadges) { e.tex.dispose(); }
  _weaponBadges.clear();
}

/** 关掉时退回纯文字标签（老行为），不再画官方图标。 */
let _weaponIconsVisible = true;
export function setWeaponIconsVisible(v) {
  _weaponIconsVisible = !!v;
  for (const [, e] of _weaponBadges) e.iconsEnabled = _weaponIconsVisible;
  // 立刻重画一遍，不用等下一帧
  for (const [name, e] of _weaponBadges) {
    _paintBadge(e, name, _weaponIconsVisible ? e.image : null, e.colorHex || 0xcccccc);
  }
}

/**
 * 给每个玩家挂上真实武器模型。按 (玩家, 武器名) 记忆，**只在变化时**重新挂，
 * 因为 `attachWeapon` 是异步的，每帧调用会白白压一堆 promise。
 */
export function updateWeaponModels(playerStates, show) {
  if (!show) {
    if (_weaponModelsShown) { setWeaponModelsVisible(false); _weaponModelsShown = false; }
    return;
  }
  if (!_weaponModelsShown) { setWeaponModelsVisible(true); _weaponModelsShown = true; }

  for (let i = 0; i < playerGroups.length; i++) {
    const pg = playerGroups[i];
    const st = playerStates[i];
    if (!pg || !pg.group) continue;
    if (!st || !st.alive) {
      if (_weaponModelKey.get(i)) {
        setPlayerWeaponModelVisible(i, false);
        _weaponModelKey.delete(i);
      }
      continue;
    }
    const wName = weaponName(st.weapon);
    if (_weaponModelKey.get(i) === wName) continue;
    _weaponModelKey.set(i, wName);
    // 不 await：模型是异步加载的，到了自己挂上去
    attachWeapon(i, pg.group, wName);
  }
}

/** playerIdx -> 已挂的武器显示名（用来判断"变了没有"） */
const _weaponModelKey = new Map();
let _weaponModelsShown = true;

/** 玩家身上的武器模型统计（给"数据检查"面板用）。 */
export function weaponModelStats() {
  return weaponModelDebugStats();
}

// ── 玩家更新 ────────────────────────────────────────────────────────────────

export function updatePlayers(playerStates, sampleIdx, teams, showNames, showTrails) {
  // 轨迹和"人画在哪"无关，先按采样点对齐（拖时间轴时这里就是重建的唯一入口）。
  // 放在最前面：下面有 `continue` 掉的支路（死人 / 缺状态），漏掉谁都不影响轨迹。
  if (showTrails) syncTrails(sampleIdx);

  for (let i = 0; i < playerGroups.length; i++) {
    const pg = playerGroups[i];
    const state = playerStates[i];
    const team = teams[i] || '?';

    if (!state) {
      pg.group.visible = false;
      if (pg.spotIcon) pg.spotIcon.visible = false;   // 同上：这一支也是 continue 掉的
      continue;
    }

    const pos = gameToThree(state.x, state.y, state.z);
    pg.group.visible = true;

    if (!state.alive) {
      pg.group.visible = false;
      if (pg.cachedGroundY === null) {
        let groundY = pos.y;
        try {
          const rayOrigin = new THREE.Vector3(pos.x, pos.y + 10, pos.z);
          const bvh = getMapBVH();
          if (bvh) {
            // 有 BVH 就用它：同一条射线朴素 raycast 要 45 ms，BVH 不到 1 ms
            const t = raycastBVH(bvh, rayOrigin.x, rayOrigin.y, rayOrigin.z, 0, -1, 0, 30);
            if (t >= 0) groundY = rayOrigin.y - t;
          } else {
            raycaster.set(rayOrigin, new THREE.Vector3(0, -1, 0));
            raycaster.far = 30;
            const hits = raycaster.intersectObjects(mapGroup.children, true);
            if (hits.length > 0) groundY = hits[0].point.y;
          }
        } catch (_) { /* 地图没加载就用玩家高度 */ }
        pg.cachedGroundY = groundY;
      }
      const deadColor = team === 'CT' ? 0x4da3ff : 0xff9f43;
      pg.xMat.color.set(deadColor);
      pg.deadX.position.set(pos.x, pg.cachedGroundY + 0.15, pos.z);
      pg.deadX.visible = true;
      // ⚠️ 必须**显式**清掉眼睛的 visible 标志。这一支是 `continue` 掉的，下面那行
      // `pg.spotIcon.visible = ...` 永远走不到，而 `group.visible = false` 只挡住
      // 渲染、不改 sprite 自己的标志 —— 于是死人会留着一个陈旧的 `visible = true`。
      // 平时看不出来（整组隐藏），但一旦这一组重新可见（下一回合、或开透视模式），
      // 就会冒出一个不该有的眼睛。`spotIconDebug().flagVisible` 就是抓这个的。
      if (pg.spotIcon) pg.spotIcon.visible = false;
      continue;
    }

    pg.group.position.set(pos.x, pos.y, pos.z);
    pg.group.visible = true;
    pg.pitchGroup.visible = true;
    pg.hpBg.visible = true;
    pg.hpFill.visible = true;
    if (pg.deadX) pg.deadX.visible = false;

    const teamColor = team === 'CT' ? 0x5b9bd5 : 0xe8a240;
    // ── 选手 X 光：**贴材质只在这一处**（`applyFigureMaterials`）──────────
    // 而且必须排在下面那几行写材质之前：
    //   · `pg.head.material = matHead` 会把头的幽灵材质覆盖回正常材质
    //   · `pg.body.material.opacity = 1` 写的是"**当前**材质"，也就是上一帧留在
    //     躯干上的那个 —— 于是幽灵的躯干会永久变成实心，而且不会有任何报错。
    //     （浏览器测试 test_player_xray_render.mjs §D 就是抓这个的）
    const ghosted = ghostedAt(i);
    pg.head.material = matHead;
    // 队色写在**队服材质本体**上，不是"当前材质"上。两者在幽灵身上不是同一个
    // 对象：贴了幽灵之后 `pg.body.material` 是幽灵那份克隆，写它就等于只改幽灵、
    // 本体永远停在创建时的猜色 —— 而幽灵的颜色正是从本体同步的（顺序：先写本体、
    // 再贴幽灵）。躯干 / 双臂 / 头盔共用本体，所以一处就够。
    pg.teamMat.color.set(teamColor);
    // 透视地图开着时，材质和 `renderOrder` **整个**归 `setXRayMode()` 那趟遍历管，
    // 这里不插手 —— 插手会把它的 `renderOrder = 999` 冲成 0，小人就不再画在那层
    // 半透明地图之上了（浏览器测试抓到的第二个退步）。
    if (!xrayEnabled) applyFigureMaterials(i, ghosted);
    // 幽灵材质自己的透明度自己管，冲回 1 就一点也不透明了
    if (!ghosted) {
      pg.body.material.opacity = 1;
      pg.body.material.transparent = false;
    }

    // ── 蹲下：小人跟着**连续**的 crouch 压下去 ──────────────────────────
    // 注意和下面的碰撞盒刻意不同：身体是"看起来像"，跟着连续的 crouch
    // 平滑下去（真实动画也是渐变的）；碰撞盒是物理真相，72/54 阶跃。
    // 同时打开调试层就能看到"身体在滑、盒子在跳"。
    // 压缩比用 (72 - 18·crouch)/72，蹲到底正好 54 HU = 蹲下的碰撞盒高度，
    // 而且和眼睛的下落量共用同一个 18 HU。
    const crouch = Math.min(1, Math.max(0, state.crouch || 0));
    const bodyRatio = (HULL_STAND_HEIGHT - CROUCH_EYE_DROP * crouch) / HULL_STAND_HEIGHT;
    // 逐网格压缩（不是整组 scale.y）：腿要绕胯旋转，整组缩放会把旋转剪切掉。
    // 位置也要一起乘 —— 不然腿的顶端会离开胯、脚会插到地底下。
    pg.torso.scale.y = bodyRatio;
    pg.torso.position.y = ((P_LEG_TOP + P_TORSO_TOP) / 2) * bodyRatio;
    pg.armL.scale.y = bodyRatio;
    pg.armL.position.y = P_ARM_Y * bodyRatio;
    pg.armR.scale.y = bodyRatio;
    pg.armR.position.y = P_ARM_Y * bodyRatio;
    pg.legL.scale.y = bodyRatio;
    pg.legL.position.y = -P_LEG_HALF * bodyRatio;
    pg.legR.scale.y = bodyRatio;
    pg.legR.position.y = -P_LEG_HALF * bodyRatio;
    pg.pitchGroup.position.y = P_HEAD_Y * bodyRatio;

    // ── 步态：摆腿 ──────────────────────────────────────────────────────
    // phase / amp 是 `buildGait()` 预计算好的**采样点纯函数**，所以拖时间轴、
    // 倒放、来回拖，同一点永远是同一个姿势（见 replay-core.js 的说明）。
    const gait = gaitData && gaitData[i];
    let legSwing = 0;
    if (gait) {
      const ph = gait.phase[sampleIdx] || 0;
      const amp = gait.amp[sampleIdx] || 0;
      legSwing = amp * Math.sin(ph);
      // 两条腿反相；绕 X 轴摆 = 沿模型正前方（+Z）前后摆，因为小人本身不随
      // yaw 旋转（group 只有 position，朝向由 pitchGroup 的锥体表示）。
      pg.legPivotL.rotation.x = legSwing;
      pg.legPivotR.rotation.x = -legSwing;
    } else {
      pg.legPivotL.rotation.x = 0;
      pg.legPivotR.rotation.x = 0;
    }
    // 胯部随摆腿下沉：剪式摆腿没有膝盖，髋高不变的话脚会离地 L(1-cosθ)，
    // 看着像在飘。按 1 - BOB(1-cosθ) 下沉后脚离地只剩一半，而且重心起伏
    // 和真人同相（步幅最大时最低、双腿并拢时最高）。
    const hipY = P_LEG_TOP * bodyRatio
      * (1 - GAIT_BOB * (1 - Math.cos(legSwing)));
    pg.legPivotL.position.y = hipY;
    pg.legPivotR.position.y = hipY;

    // 血条和名字跟着头顶走，否则蹲下时会浮在半空
    const headTop = P_HEAD_Y * bodyRatio + P_HEAD_R;
    pg.hpBg.position.y = headTop + 0.07;
    pg.hpFill.position.y = headTop + 0.07;
    if (pg.label) pg.label.position.y = headTop + 0.27;
    // ID 牌跟着头顶（蹲下一起沉），名字 / 队色变了才重画贴图
    if (pg.idTag) {
      pg.idTag.position.y = headTop + ID_TAG_LIFT;
      refreshIdTag(pg.idTag, idTagTextFor(i), teamColor);
    }

    // 「可见敌人」眼睛：位置跟着头顶走（蹲下时也要跟着沉下去），显隐由
    // `setSpotTargets()` 每帧灌进来的集合决定。死人在上面已经 continue 了
    // （整组不可见），所以这里不用再判 alive。
    if (pg.spotIcon) {
      pg.spotIcon.position.y = headTop + SPOT_ICON_LIFT;
      pg.spotIcon.visible = spotIconsEnabled && !!spotTargets && spotTargets.has(i);
    }

    const trail = playerTrails[i];
    if (trail && trail.line && trail.line.material) trail.line.material.color.set(teamColor);

    // 闪光只在非透视模式下写自发光：透视模式的发光由 setXRayMode 管，
    // 这里每帧覆盖会把透视的发光抹掉（老 bug）。幽灵的自发光同理 ——
    // 它得是"整体一个色"的，不能被闪光改成纯白。
    if (!xrayEnabled && !ghosted) {
      if (state.flash > 0 && state.alive) {
        pg.body.material.emissive = new THREE.Color(0xffffff);
        pg.body.material.emissiveIntensity = 1.0;
      } else {
        pg.body.material.emissive = new THREE.Color(0x000000);
        pg.body.material.emissiveIntensity = 0;
      }
    }

    const yawRad = THREE.MathUtils.degToRad(state.yaw);
    const pitchRad = THREE.MathUtils.degToRad(state.pitch);
    pg.group.rotation.set(0, yawRad, 0);
    pg.pitchGroup.rotation.set(pitchRad, 0, 0);

    const hpPct = Math.max(0, state.hp / 100);
    pg.hpFill.scale.x = hpPct;
    pg.hpFill.position.x = -(1 - hpPct) / 2;
    if (hpPct > 0.6) pg.hpFill.material = matHPHigh;
    else if (hpPct > 0.3) pg.hpFill.material = matHPMid;
    else pg.hpFill.material = matHPLow;
    pg.hpBg.visible = state.alive;
    pg.hpFill.visible = state.alive;

    if (pg.label) pg.label.visible = showNames;
    if (pg.idTag) pg.idTag.visible = showPlayerIds;

    // ── 碰撞盒 / 眼睛 ────────────────────────────────────────────────────
    // 碰撞盒高度取 crouching（FL_DUCKING，阶跃），眼高取 crouch（连续）。
    // 两者刻意分开：真实碰撞盒是 72/54 跳变，眼睛是滑下去的 —— 同时打开
    // 就能直接看到"盒子跳、眼睛滑"这件事。
    if (pg.hull) {
      const hullH = hullHeightFor(state.crouching);
      const eyeH = state.eye_height != null ? state.eye_height : eyeHeightFor(state.crouch);
      pg.hull.visible = showCrouchDebug;
      pg.ruler.visible = showCrouchDebug;
      pg.eye.visible = showCrouchDebug;
      if (showCrouchDebug) {
        // 盒子在 group 局部坐标里以脚底为底：底面 y=0，所以中心在 H/2
        pg.hull.scale.set(
          HULL_HALF_WIDTH * 2 * SCALE, hullH * SCALE, HULL_HALF_WIDTH * 2 * SCALE,
        );
        pg.hull.position.y = hullH * 0.5 * SCALE;
        pg.hull.material = team === 'CT' ? matHullCT : matHullT;
        pg.ruler.scale.y = eyeH * SCALE;
        pg.eye.position.y = eyeH * SCALE;
      }
    }

    if (pg.weaponSprite) {
      const wName = weaponName(state.weapon);
      if (!state.alive) {
        // 死亡时收起 badge，不显示上一把枪
        pg.weaponSprite.visible = false;
      } else {
        pg.weaponSprite.visible = true;
        // 队色和名字/血条一致，方便一眼分敌我
        const col = team === 'CT' ? 0x5b9bd5 : 0xe8a240;
        const badge = _getWeaponBadge(wName, col);
        if (pg.weaponSprite.material.map !== badge.tex) {
          pg.weaponSprite.material.map = badge.tex;
          pg.weaponSprite.material.needsUpdate = true;
        }
        _loadWeaponBadgeIcon(wName, col);
      }
    }
  }
}

export function updatePlayerNames(playerStates, teams) {
  for (let i = 0; i < playerGroups.length; i++) {
    const pg = playerGroups[i];
    if (!pg.label) continue;
    const team = teams[i] || '?';
    updateLabelSprite(pg.label, `#${i}`, team === 'CT' ? 0x5b9bd5 : 0xe8a240, 12);
  }
}

export function setPlayerNames(playerNames, currentTeams) {
  for (let i = 0; i < playerGroups.length; i++) {
    const pg = playerGroups[i];
    if (!pg.label) continue;
    const name = playerNames[i] || `Player ${i}`;
    const team = currentTeams ? currentTeams[i] : '?';
    updateLabelSprite(pg.label, name, team === 'CT' ? 0x5b9bd5 : 0xe8a240, 12);
  }
}

/** 往折线末尾写一个点；写满就把最老的挤出去（一个采样点一个点，不合并）。 */
function pushTrailPoint(trail, x, y, z) {
  const arr = trail.positions;
  if (trail.count < MAX_TRAIL_POINTS) {
    const o = trail.count * 3;
    arr[o] = x;
    arr[o + 1] = y;
    arr[o + 2] = z;
    trail.count += 1;
  } else {
    arr.copyWithin(0, 3);
    const o = (MAX_TRAIL_POINTS - 1) * 3;
    arr[o] = x;
    arr[o + 1] = y;
    arr[o + 2] = z;
  }
}

/** 把缓冲交给 geometry：复用同一块 Float32Array，不重新分配属性。 */
function uploadTrail(trail) {
  trail.geo.setDrawRange(0, trail.count);
  trail.geo.attributes.position.needsUpdate = true;
}

/**
 * 按历史重算**全部**轨迹（一次跳转只重建一趟，10 个人共用同一批 states）。
 *
 * `trail(k)` = 采样点 `[k-149, k]` 上这些人的位置，一个采样点一个点 ——
 * 和"一路播过来"得到的逐点相同，这就是修好瞬移的关键。
 */
function rebuildTrails(sampleIdx) {
  for (const t of playerTrails) t.count = 0;
  if (trailHistoryAt && sampleIdx >= 0) {
    const from = Math.max(0, sampleIdx - (MAX_TRAIL_POINTS - 1));
    for (let j = from; j <= sampleIdx; j++) {
      const states = trailHistoryAt(j);
      if (!states) continue;
      for (let i = 0; i < playerTrails.length; i++) {
        const st = states[i];
        if (!st) continue;
        const p = gameToThree(st.x, st.y, st.z);
        pushTrailPoint(playerTrails[i], p.x, p.y, p.z);
      }
    }
  }
  for (const t of playerTrails) {
    t.lastSample = sampleIdx;
    uploadTrail(t);
  }
}

/**
 * 每帧一次：把 10 条轨迹对齐到 `sampleIdx`。
 *
 * 采样点连续 +1（正常播放）→ 追加一个点，便宜；其余情况（拖时间轴、倒退、
 * 掉帧一次跳过好几个采样点、`轨迹` 开关关过一阵再打开、换回合）→ 按历史整条
 * 重算。两条路读的都是**同一个** `trailHistoryAt`，所以结果必然一致 ——
 * `visualizer/tests/test_trail_seek.mjs` 逐点量这个。
 */
function syncTrails(sampleIdx) {
  if (!playerTrails.length) return;
  const prev = playerTrails[0].lastSample;
  if (prev === sampleIdx) return;          // 同一帧里被调了两次（seekSample 之后又来一帧）
  if (sampleIdx !== prev + 1 || !trailHistoryAt) return rebuildTrails(sampleIdx);

  const states = trailHistoryAt(sampleIdx);
  if (!states) return rebuildTrails(sampleIdx);
  for (let i = 0; i < playerTrails.length; i++) {
    const trail = playerTrails[i];
    const st = states[i];
    if (st) {
      const p = gameToThree(st.x, st.y, st.z);
      pushTrailPoint(trail, p.x, p.y, p.z);
    }
    trail.lastSample = sampleIdx;
    uploadTrail(trail);
  }
}

export function clearAllTrails() {
  for (const trail of playerTrails) {
    trail.count = 0;
    trail.lastSample = -1;
    trail.geo.setDrawRange(0, 0);
  }
}

/** 给测试 / 调试用：每条轨迹画出来的点（世界坐标）、点数、最后一个点对应的采样点。 */
export function trailDebugStats() {
  const out = [];
  for (const t of playerTrails) {
    const arr = t.positions;
    const pts = [];
    for (let k = 0; k < t.count; k++) pts.push([arr[k * 3], arr[k * 3 + 1], arr[k * 3 + 2]]);
    out.push({
      count: t.count,
      lastSample: t.lastSample,
      drawRange: t.geo.drawRange.count,
      visible: t.line ? t.line.visible : false,
      pts,
    });
  }
  return out;
}

export function setTrailsVisible(visible) {
  for (const trail of playerTrails) if (trail.line) trail.line.visible = visible;
}

export function setNamesVisible(visible) {
  for (const pg of playerGroups) if (pg.label) pg.label.visible = visible;
}

/**
 * 碰撞盒 + 眼睛 + 脚到眼的标尺（下蹲调试层）。
 *
 * 碰撞盒画的是**真实物理盒**（站 72 / 蹲 54，阶跃），眼睛画的是**连续**的眼高
 * （64.0938 → 46.0938）。这是两个不同的量，同时打开就能看出区别：蹲到底的那
 * 一帧盒子从 72 跳到 54，而眼睛是滑下去的。
 */
/** 给测试脚本读渲染结果用：拿到某个玩家的 group 和它上面挂的碰撞盒 / 眼睛。 */
export function playerGroupAt(i) {
  return playerGroups[i];
}

export function setCrouchDebugVisible(visible) {
  showCrouchDebug = visible;
  if (!visible) {
    for (const pg of playerGroups) {
      if (!pg.hull) continue;
      pg.hull.visible = false;
      pg.ruler.visible = false;
      pg.eye.visible = false;
    }
  }
}

// ── C4 ──────────────────────────────────────────────────────────────────────

export function updateBomb(bombPos, planted, time) {
  if (!bombGroup) return;
  if (!bombPos) { bombGroup.visible = false; return; }
  bombGroup.visible = true;
  const pos = gameToThree(bombPos[0], bombPos[1], bombPos[2]);
  bombGroup.position.set(pos.x, pos.y + 0.3, pos.z);
  if (bombPulseLight && planted) {
    const pulse = 0.5 + 0.5 * Math.sin(time * 8);
    bombPulseLight.material.opacity = 0.5 + pulse * 0.5;
    bombPulseLight.scale.setScalar(1 + pulse * 0.5);
  }
}

// ── 投掷物 ──────────────────────────────────────────────────────────────────

export function buildGrenadeTrajectories(trajectories) {
  clearGrenadeTrajectories();
  for (const traj of trajectories) {
    const color = grenadeColor(traj.type);
    const pts3 = traj.points.map(p => {
      const t = gameToThree(p.x, p.y, p.z);
      return new THREE.Vector3(t.x, t.y, t.z);
    });

    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts3),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5, linewidth: 1, depthTest: true })
    );
    line.renderOrder = 500;
    scene.add(line);

    // 位置载体。**是 Group 不是 Mesh**：模型是异步加载的，加载完要把子节点从
    // "兜底球"换成真实模型，而 updateGrenadeTrajectories 只写 position/visible。
    // renderOrder 写在子节点上（Group 的不会下传）。
    const dot = new THREE.Group();
    dot.visible = false;
    dot.userData = {
      isGrenadeDot: true, type: traj.type, thrower: traj.thrower,
      eid: traj.eid, points: traj.points,
    };
    dot.name = `nade-${traj.type}-p${traj.thrower}-e${traj.eid}`;

    // 兜底球：模型到货之前先占位（本地 750 KB，一般看不到），
    // 以及用户关掉「投掷物模型」时**退回到改造前的老样子**。
    const fallback = new THREE.Mesh(nadeDotGeo, new THREE.MeshStandardMaterial({
      color, roughness: 0.2, metalness: 0.1, emissive: color, emissiveIntensity: 0.7,
    }));
    fallback.name = 'nade-fallback';
    fallback.renderOrder = 501;
    dot.add(fallback);
    scene.add(dot);

    const rec = {
      line, dot, fallback, points: traj.points, type: traj.type,
      thrower: traj.thrower, eid: traj.eid, settleTime: traj.settleTime,
      // 莫洛托夫 / 燃烧弹共用 `CMolotovProjectile`，靠投掷者队伍区分
      // （约 1.4% 的已知误差，见 weapon-index.js 的 nadeModelKey）
      modelKey: nadeModelKey(traj.type, playerTeam(traj.thrower)),
      spin: null, disposed: false,
    };
    grenadeTrajectories.push(rec);
    // 不 await：模型到货之前先显示兜底球。`.catch` 是保险 ——
    // 加载链现在不会 reject，但将来改坏了不该变成一个刷屏的 unhandled rejection。
    _installNadeModel(rec).catch((err) => {
      console.warn(`[v5] 投掷物模型 ${rec.modelKey} 安装失败：`, err?.message || err);
    });
  }
}

/**
 * 异步把兜底球换成真实模型。
 *
 * 失败的路径都要能退回球：manifest 里没有这个 key、GLB 404、或者轨迹已经被
 * 清掉（换回合了）。**不能**因为模型没到货就让投掷物整个消失。
 */
async function _installNadeModel(rec) {
  if (!rec.modelKey) return;
  const model = await loadNadeModel(rec.modelKey);
  if (!model || rec.disposed) return;

  const inst = model.root.clone(true);
  inst.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true;
    o.frustumCulled = false;
    o.renderOrder = 501;   // 和改造前那个球一致（Group 的 renderOrder 不会下传）
  });
  inst.scale.setScalar(GRENADE_MODEL_SCALE);
  // 按包围盒中心归零：模型原点不在几何中心。莫洛托夫的并集包围盒是
  // min=[-0.045,-0.128,-0.045] max=[0.074,0.223,0.187]（含瓶身 / 火焰片 /
  // 瓶里的液体三个 primitive），原点明显偏在一侧；不归零的话投掷物会整体
  // 偏离轨迹线。
  inst.position.set(
    -model.centre.x * GRENADE_MODEL_SCALE,
    -model.centre.y * GRENADE_MODEL_SCALE,
    -model.centre.z * GRENADE_MODEL_SCALE,
  );

  // 旋转单独一层：绕**包围盒中心**转，而不是绕模型原点
  const spin = new THREE.Group();
  spin.name = 'nade-spin';
  spin.add(inst);

  rec.spin = spin;
  rec.dot.add(spin);
  _applyNadeVisibility(rec);
}

/**
 * 显隐：模型开着且到货了 → 只画模型；否则画兜底球。
 *
 * 这样「投掷物模型」开关就真的是一键 A/B（关掉 = 改造前那个球），
 * 而不是"关掉之后投掷物整个看不见"。
 */
function _applyNadeVisibility(rec) {
  if (rec.spin) rec.spin.visible = nadeModelsVisible;
  rec.fallback.visible = !(nadeModelsVisible && rec.spin);
}

export function updateGrenadeTrajectories(timeSec) {
  for (const traj of grenadeTrajectories) {
    const pts = traj.points;
    if (pts.length < 2) { traj.dot.visible = false; continue; }
    const firstSec = pts[0].t;
    const settleTime = traj.settleTime || pts[pts.length - 1].t;
    if (timeSec < firstSec || timeSec > settleTime + 0.5) {
      traj.dot.visible = false;
      traj.line.visible = false;
      continue;
    }
    let i0 = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      if (timeSec >= pts[i].t && timeSec <= pts[i + 1].t) { i0 = i; break; }
      i0 = Math.min(i, pts.length - 2);
    }
    const tA = pts[i0].t;
    const tB = pts[Math.min(i0 + 1, pts.length - 1)].t;
    const frac = Math.max(0, Math.min(1, (timeSec - tA) / Math.max(0.001, tB - tA)));
    const a = gameToThree(pts[i0].x, pts[i0].y, pts[i0].z);
    const j = Math.min(i0 + 1, pts.length - 1);
    const b = gameToThree(pts[j].x, pts[j].y, pts[j].z);
    traj.dot.position.set(
      a.x + (b.x - a.x) * frac,
      a.y + (b.y - a.y) * frac,
      a.z + (b.z - a.z) * frac
    );
    _aimNadeModel(traj, a, b);
    traj.dot.visible = true;
    traj.line.visible = true;
  }
}

/**
 * 朝向：**demo 里没有投掷物的角度数据**（`_grenades()` 只存 x/y/z），所以朝向
 * 只能从轨迹本身推 —— 把模型长轴（实测 6 个模型都是 +Y）对齐到这一段的
 * **速度方向**。抛物线飞行时模型会自然地从头朝上转到头朝下。
 *
 * 这是**推导量**，不是原始数据：真实投掷物是翻滚的，这里不假装知道翻滚相位。
 * 落地后 `a`/`b` 重合（速度 ≈ 0），此时保持上一帧的朝向，不要"啪"地立正。
 */
function _aimNadeModel(traj, a, b) {
  if (!traj.spin) return;
  const vx = b.x - a.x, vy = b.y - a.y, vz = b.z - a.z;
  const len = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (len < 1e-6) return;
  _nadeDir.set(vx / len, vy / len, vz / len);
  traj.spin.quaternion.setFromUnitVectors(NADE_LONG_AXIS, _nadeDir);
}

/** 投掷物模型开关。关掉 = 退回改造前的圆点（不是"什么都不画"）。 */
export function setNadeModelsVisible(v) {
  nadeModelsVisible = !!v;
  for (const rec of grenadeTrajectories) _applyNadeVisibility(rec);
}

/** 数据检查面板用：多少条轨迹用上了真模型、多少条还在等 / 没有映射。 */
export function nadeModelDebugStats() {
  let withModel = 0, pending = 0, noKey = 0;
  const byKey = {};
  for (const rec of grenadeTrajectories) {
    if (rec.spin) {
      withModel++;
      byKey[rec.modelKey] = (byKey[rec.modelKey] || 0) + 1;
    } else if (rec.modelKey) {
      pending++;
    } else {
      noKey++;
    }
  }
  return {
    visible: nadeModelsVisible,
    total: grenadeTrajectories.length, with_model: withModel,
    pending, no_key: noKey, by_key: byKey,
  };
}

function clearGrenadeTrajectories() {
  for (const traj of grenadeTrajectories) {
    // 还在飞的加载回调不能再往已经移除的 group 上挂节点
    traj.disposed = true;
    scene.remove(traj.line);
    scene.remove(traj.dot);
    traj.line.geometry.dispose();
    traj.line.material.dispose();
    // 兜底球的材质是**每条轨迹自己**的，要释放。
    // 真实模型的 geometry / material 是**共享**的（`clone()` 只复制节点树，
    // 不复制几何和材质），绝对不能在这里 dispose —— 否则下一回合所有投掷物
    // 会一起变成黑块（或者直接报错）。
    traj.fallback.material.dispose();
  }
  grenadeTrajectories = [];
}

// ═══════════════════════════════════════════════════════════════════════════
// 真实烟雾体素
// ═══════════════════════════════════════════════════════════════════════════

/** 每个体素 12 HU = 0.3048 m；画得略小一点，避免相邻体素面完全重合。 */
const VOXEL_M = VOXEL_WORLD_SIZE * SCALE;
const VOXEL_BOX = VOXEL_M * 0.92;
const SMOKE_CAPACITY = 12288;  // 凸包重建后实测最多 6408（169 个真实烟雾），留足余量

const smokeGeo = new THREE.BoxGeometry(VOXEL_BOX, VOXEL_BOX, VOXEL_BOX);

/**
 * 烟雾材质：低不透明度、不写深度 → 大量体素叠加出体积感。
 *
 * 单个体素的不透明度要按**视线穿过的体素数**来配。凸包重建后穿过云心的射线
 * 实测中位数 ~25 个体素，0.14 时中心不透明度 1−0.86²⁵ ≈ 0.98，边缘自然变淡；
 * 只画边界壳时只经过 3~4 个，所以旧值 0.20 配的是空心壳。
 *
 * 这里的体素几何纯粹是**画面**：`solidFromShell` 只在浏览器里跑，不参与 .csz 读写。
 */
export const SMOKE_OPACITY = 0.14;

/**
 * 烟雾基色。**量出来的，不是挑出来的**。
 *
 * 云心穿过的体素 ~25 个 → 中心几乎不透明，画出来的颜色基本就是基色被灯光打亮
 * 的结果（实测约为基色亮度的 0.73~0.79）。所以基色直接决定烟有多"深"。
 *
 * 旧值 `0xdfe5ec`（亮度 228）在 24 团真实烟雾 × 多视线方向上量出来是**背景的
 * 1.074 倍，最坏一团 1.182 倍** —— 烟比它背后的墙还亮，所以看着像一层发白的雾，
 * 而不是一团实心的烟。逐像素差值图更直接：旧值下烟和地图的亮度差大多不足 8/255，
 * 轮廓基本看不出来。
 *
 * 现值 `0xa5a9af` 是旧值按 0.74 **等比压暗**（不是把蓝调加深 —— 手挑的
 * `#b6bfca` 那种绝对蓝偏移越压越大，会把烟越配越蓝）：
 *
 *   基色亮度   228 → 169     渲染亮度  167 → 133
 *   烟/地图   1.074 → 0.862  最坏一团  1.182 → 0.898（即任何一团都不会比背后的墙亮）
 *   烟/天空   0.869 → 0.697
 *   饱和度    0.045 → 0.058（色相没动）
 *
 * （"烟/地图"= 被烟盖住的那些像素，有烟 vs 没烟时的亮度比。
 * 回归：`node visualizer/tests/test_smoke_color.mjs http://127.0.0.1:5099 <有烟的.csz>`。）
 */
export const SMOKE_BASE_COLOR = 0xa5a9af;

function makeSmokeMaterial() {
  return new THREE.MeshStandardMaterial({
    color: SMOKE_BASE_COLOR,
    roughness: 1.0,
    metalness: 0.0,
    transparent: true,
    opacity: SMOKE_OPACITY,
    depthWrite: false,
    side: THREE.FrontSide,
  });
}

/** eid -> {mesh, cache, material} */
const smokeVolumes = new Map();
let smokeSolid = true;   // true = 实心体素云；false = 只画边界壳

const _m4 = new THREE.Matrix4();
const _v3 = new THREE.Vector3();

function getSmokeVolume(inst) {
  let vol = smokeVolumes.get(inst.entity_id);
  if (!vol) {
    const mesh = new THREE.InstancedMesh(smokeGeo, makeSmokeMaterial(), SMOKE_CAPACITY);
    mesh.name = `smoke-${inst.entity_id}`;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    mesh.count = 0;
    mesh.renderOrder = 950;
    scene.add(mesh);
    vol = { mesh, cache: new SmokeVoxelCache(inst, { solid: smokeSolid }) };
    smokeVolumes.set(inst.entity_id, vol);
  }
  return vol;
}

/** 清掉所有烟雾网格与解码缓存（换回合 / 切实心-边界时调用）。 */
export function resetSmokeVolumes() {
  for (const [, vol] of smokeVolumes) {
    scene.remove(vol.mesh);
    vol.mesh.material.dispose();
  }
  smokeVolumes.clear();
}

/**
 * 当前 tick 的烟雾遮挡体，喂给 `visibility.js`。
 *
 * 复用**渲染用的那一份体素缓存**（`getSmokeVolume()` 里的 `SmokeVoxelCache`），
 * 所以不会为了遮挡再解码一遍 journal。返回的是 game 空间的原始栅格 ——
 * `visibility.js` 自己按 `det + (i-15.5)*cell` 的约定走 DDA。
 *
 * 注意这里**不看 `toggle-smokes`**：关掉烟雾的显示不该让人"穿烟看见人"。
 * 但要看 `toggle-solid` —— 因为缓存本身是按 solid 建的，改开关会重建。
 *
 * @param {Array} instances 当前回合的 smoke 实例
 * @param {number} tick
 * @returns {Array<{det:number[], grid:Uint8Array, dim:number, cell:number}>}
 */
export function smokeOccludersAt(instances, tick) {
  const out = [];
  if (!instances) return out;
  for (const inst of instances) {
    if (tick < inst.first_tick || tick > inst.last_tick) continue;
    if (!inst.detonation) continue;      // 没引爆点就放不进世界坐标（同渲染路径）
    const geo = getSmokeVolume(inst).cache.at(tick);
    if (!geo || geo.count === 0) continue;
    out.push({
      det: inst.detonation,
      grid: geo.grid,
      dim: VOXEL_GRID_DIM,
      cell: VOXEL_WORLD_SIZE,
    });
  }
  return out;
}

/**
 * 更新当前 tick 的烟雾体素。
 * @param {Array} instances 当前回合的 smoke 实例（含 changes[].data=base64）
 * @param {number} tick     当前采样 tick
 * @param {{solid?: boolean}} opts
 */
export function updateSmokeVolumes(instances, tick, opts = {}) {
  const solid = opts.solid !== false;
  if (solid !== smokeSolid) {
    smokeSolid = solid;
    resetSmokeVolumes();
  }
  // sprite 层跟着体素层一起初始化（幂等）
  if (_smokeSpritesWanted) _ensureSmokeSpriteSeq();

  const active = new Set();
  for (const inst of instances) {
    if (tick < inst.first_tick || tick > inst.last_tick) continue;
    // 没有引爆点就没法把体素栅格放进世界坐标（world = detonation + (grid-15.5)*12）。
    // demo_extract 的 detonation 是按"实体 id + 最近 tick"匹配 smokegrenade_detonate
    // 事件得来的，匹配不到就是 null（实测 170 个烟雾里有 1 个，在最后一个回合）。
    // 这种实例**不渲染** —— 画到 [0,0,0] 会在地图原点凭空多出一团雾，比不画更糟。
    if (!inst.detonation) continue;
    active.add(inst.entity_id);
    const vol = getSmokeVolume(inst);
    const geo = vol.cache.at(tick);
    if (!geo || geo.count === 0) {
      vol.mesh.count = 0;
      vol.lastGrid = null;
      const s0 = smokeSprites.get(inst.entity_id);
      if (s0) s0.mesh.visible = false;
      continue;
    }

    // tick 没跨过变更边界时几何完全一样，不重写整批实例矩阵
    if (geo.grid === vol.lastGrid) continue;
    vol.lastGrid = geo.grid;

    const n = Math.min(geo.count, SMOKE_CAPACITY);
    if (geo.count > SMOKE_CAPACITY) {
      console.warn(`[visuals] 烟雾 ${inst.entity_id} 体素 ${geo.count} > 容量，已截断`);
    }
    const pts = geo.points;
    for (let i = 0; i < n; i++) {
      const t = gameToThree(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
      _v3.set(t.x, t.y, t.z);
      _m4.makeTranslation(_v3.x, _v3.y, _v3.z);
      vol.mesh.setMatrixAt(i, _m4);
    }
    vol.mesh.count = n;
    vol.mesh.instanceMatrix.needsUpdate = true;

    // 引爆时刻 = first_tick 对应的秒数，用于 sprite 翻帧的**独立**时间基准
    if (!_smokeSpriteBirth.has(inst.entity_id)) {
      const fps = inst.tickrate || 64;
      _smokeSpriteBirth.set(inst.entity_id, (inst.first_tick || 0) / fps);
    }
    _placeSmokeSprite(inst, geo);
  }

  // 已经消失的烟雾：藏起来（缓存留着，拖回去还能秒显示）
  for (const [eid, vol] of smokeVolumes) {
    if (!active.has(eid) && vol.mesh.count) {
      vol.mesh.count = 0;
      vol.lastGrid = null;
      const s = smokeSprites.get(eid);
      if (s) s.mesh.visible = false;
    }
  }
}

/** 当前每个烟雾实例的解码统计（给"数据检查"面板用）。 */
export function smokeDebugStats() {
  const out = [];
  for (const [eid, vol] of smokeVolumes) {
    out.push({ entity_id: eid, visible: vol.mesh.count, ...vol.cache.lastStats });
  }
  return out;
}

/**
 * 烟雾动画：极轻微的透明度呼吸。旧版靠随机缩放抖动，这里保持体素几何不动
 * （体素位置是**真实数据**，抖动反而破坏它），只做非常淡的亮度起伏。
 */
export function updateSmokeAnimations(dt, timeSec) {
  const t = performance.now() / 1000;
  for (const [, vol] of smokeVolumes) {
    if (!vol.mesh.count) continue;
    vol.mesh.material.opacity = SMOKE_OPACITY * (1.0 + 0.08 * Math.sin(t * 0.9));
  }
  _updateSmokeSpriteFrames(timeSec);
}

// ═══════════════════════════════════════════════════════════════════════════
// 真实烟雾 sprite 层（叠在体素云上）
// ═══════════════════════════════════════════════════════════════════════════
//
// 体素云给出**真实的形状和遮挡**（来自 `m_VoxelFrameData` 的 Morton journal），
// 但它是一堆半透明方块，近看是"一坨方块"。CS2 实际的外观来自
// `materials/particle/smoke/smokeburst/smokeloop_i_0_sc_hardedge` 这张
// **128 帧的 sprite 序列**（`explosion_smokegrenade_voxel.vpcf` 里的
// `m_vecTexturesInput` 第一个条目）。
//
// 所以这一层是：**在体素云的包围盒中心放一个面向相机的 billboard，
// 按时间翻帧**。体素负责体积和遮挡，sprite 负责"看起来像烟"。
//
// 注意 vpcf 里第二张贴图 `smokeloop_i_0_flwmix` 是
// `SPRITECARD_TEXTURE_ANIMMOTIONVEC`（**流向图**，驱动粒子运动矢量），
// 不是第二张可见贴图 —— 所以这里只用一张序列。
//
// 播放速率：vpcf 里**没有帧率字段**，所以这是估的。128 帧 @ 15fps ≈ 8.5 秒一轮，
// 和烟雾 18 秒的存活期大致匹配（能循环两次）。可用 `setSmokeSpriteFPS()` 调。

const SMOKE_SPRITE_SEQ = 'smokeloop_i_0_sc_hardedge';

/** 烟雾 sprite 的翻帧速率（帧/秒）。vpcf 没给，见上方注释。 */
let smokeSpriteFPS = 15;

/** 一圈 128 帧的总时长（秒） */
export function smokeSpriteDuration() {
  const seq = _smokeSpriteSeq;
  const n = seq && seq.textures ? seq.textures.length : 128;
  return n / smokeSpriteFPS;
}

export function setSmokeSpriteFPS(fps) {
  smokeSpriteFPS = Math.max(1, Math.min(60, fps));
}

export function smokeSpriteFPSValue() { return smokeSpriteFPS; }

/** entity_id -> {mesh, material} */
const smokeSprites = new Map();
let _smokeSpriteSeq = null;
let _smokeSpritesWanted = true;

/** `_placeSmokeSprite` 的排序缓冲：每帧调用，不要每次分配。 */
let _spriteDist = new Float32Array(0);

function _ensureSmokeSpriteSeq() {
  if (_smokeSpriteSeq) return;
  loadSpriteSequence(SMOKE_SPRITE_SEQ).then((seq) => {
    if (!seq) {
      console.warn('[visuals] 烟雾 sprite 序列不可用，只有体素云');
      return;
    }
    _smokeSpriteSeq = seq;
    // 序列刚到：把已有 sprite 的贴图补上
    for (const [, s] of smokeSprites) {
      if (s.mesh.material.map == null) {
        s.mesh.material.map = seq.textures[0];
        s.mesh.material.needsUpdate = true;
      }
    }
  }).catch(() => {});
}

function _getSmokeSprite(inst) {
  let s = smokeSprites.get(inst.entity_id);
  if (!s) {
    const mat = new THREE.SpriteMaterial({
      transparent: true,
      depthWrite: false,      // 和体素云一致：不写深度，靠叠加出体积
      opacity: 0.55,
      map: _smokeSpriteSeq ? _smokeSpriteSeq.textures[0] : null,
    });
    const mesh = new THREE.Sprite(mat);
    mesh.name = `smoke-sprite-${inst.entity_id}`;
    mesh.renderOrder = 951;   // 体素是 950，sprite 叠在上面
    mesh.visible = false;
    scene.add(mesh);
    s = { mesh, material: mat };
    smokeSprites.set(inst.entity_id, s);
  }
  return s;
}

/** 清掉烟雾 sprite 层。 */
export function resetSmokeSprites() {
  for (const [, s] of smokeSprites) {
    scene.remove(s.mesh);
    s.mesh.material.dispose();
  }
  smokeSprites.clear();
}

/** 开关烟雾 sprite 层（体素云不受影响）。 */
export function setSmokeSpritesVisible(v) {
  _smokeSpritesWanted = !!v;
  for (const [, s] of smokeSprites) s.mesh.visible = false;
}

export function smokeSpritesVisible() { return _smokeSpritesWanted; }

// ── 小地图里的烟：压暗 ─────────────────────────────────────────────────────
//
// 俯视图里的烟是「0.14 不透明度的体素叠起来 + 一张发白的 sprite」，铺在雷达底图上
// 亮得像块白斑，A/B 点和房间名全被盖住。3D 视角里那个亮度是**对的**（和游戏里逐
// 像素比过，见 `SMOKE_BASE_COLOR` 的注释），所以不能直接改材质 —— 改成在**小地图
// 那一次绘制**期间把颜色乘暗、画完立刻还原（`registerMinimapPass`）。
//
// 只乘颜色、不动不透明度：体积感是体素叠出来的，压不透明度会变成「烟变薄了」，
// 压颜色才是「烟变深了」。
//
// 系数是逐档截图挑的（整图视角下 1 / 0.5 / 0.3 / 0.15 各截一张比）：0.52 还是偏白，
// 0.15 已经和地面糊成一块脏斑 —— 0.3 一档烟还读得出是「一团烟」，又不会把雷达
// 底图上的 A/B 点和房间名压掉。这是**线性**空间的乘数，换成肉眼亮度约 ×0.58。
const MINIMAP_SMOKE_DIM = 0.3;

let _minimapSmokeDim = 1;

/** 原始色。压暗是在它上面乘系数，不是逐次相乘 —— 否则来回切会越乘越黑 */
const _minimapSmokeBase = new THREE.Color(SMOKE_BASE_COLOR);

/**
 * 把烟雾材质临时压暗 / 还原。
 * @param {number} dim 乘在颜色上的系数，1 = 原样
 */
function applyMinimapSmokeDim(dim) {
  if (dim === _minimapSmokeDim) return;
  _minimapSmokeDim = dim;
  for (const [, vol] of smokeVolumes) {
    vol.mesh.material.color.copy(_minimapSmokeBase).multiplyScalar(dim);
  }
  for (const [, s] of smokeSprites) {
    s.mesh.material.color.setScalar(dim);   // sprite 是白贴图，color 就是乘性 tint
  }
}

// 每次小地图绘制前后自动压暗 / 还原。同一个 rAF 里的 3D 绘制完全看不到这个改动。
registerMinimapPass((on) => applyMinimapSmokeDim(on ? MINIMAP_SMOKE_DIM : 1));

/**
 * 烟雾 sprite 的半径上限（HU，与 `SMOKE_BASE_COLOR` 同一套刻度）。
 *
 * 游戏里这团烟本身就是半径 ~144 HU 的球；在 98 团真实烟雾上实测，云的质心到体素
 * 距离的 p95 中位数是 **156 HU**（p10 141 / p90 168）。上限只用来兜住"某一帧的
 * 极少数离群体素"——那些体素是沿着墙被切剩下的碎片，让 sprite 跟着它们长大
 * 只会把贴图顶进墙里。取 160 约等于游戏半径，实测只有约 35% 的烟会被它碰到。
 */
const SMOKE_SPRITE_MAX_RADIUS_HU = 160;

/** 半径取到哪个分位：比 max 稳健，又能盖住云的绝大部分。 */
const SMOKE_SPRITE_RADIUS_PCT = 0.95;

/**
 * 把 sprite 放到体素云的**质心**，尺寸按质心到体素的**稳健半径**（p95）定。
 *
 * 这里曾经用"包围盒中心 + 包围盒最大水平跨度 ×1.15"，那是**错的**，而且
 * 实测偏得很厉害（98 团真实烟雾）：
 *
 * * 包围盒中心中位数离**引爆点** 52 HU（p90 = 72 HU），因为烟被地面和墙切过、
 *   本来就不对称，而包围盒对最远的那几个离群体素极其敏感。sprite 于是整体
 *   平移出烟团，贴到旁边的墙上 —— 看起来就是"体积烟雾穿模"。
 * * 边长中位数 **317 HU**（≈8 m），而烟本身只有 ~288 HU 跨径。
 *
 * 质心是**点云平均**，对同一批离群体素不敏感；配合 p95 半径，sprite 会稳定地
 * 落在烟团内部并盖住它。用**实测半径**而不是固定半径，是因为体素云在生长
 * （早期小、后期大），固定尺寸会让 sprite 从一开始就撑满整个最终体积。
 */
function _placeSmokeSprite(inst, geo) {
  const s = _getSmokeSprite(inst);
  const pts = geo.points;
  const n = Math.min(geo.count, SMOKE_CAPACITY);
  if (!n) { s.mesh.visible = false; return; }

  // 质心（game 空间，HU）
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < n; i++) {
    mx += pts[i * 3]; my += pts[i * 3 + 1]; mz += pts[i * 3 + 2];
  }
  mx /= n; my /= n; mz /= n;

  // 到质心的距离，取 p95。**排序返回新数组**（不要原地 sort 那个每帧复用的缓冲）
  if (_spriteDist.length < n) _spriteDist = new Float32Array(n * 2);
  const buf = _spriteDist;
  for (let i = 0; i < n; i++) {
    const dx = pts[i * 3] - mx, dy = pts[i * 3 + 1] - my, dz = pts[i * 3 + 2] - mz;
    buf[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  const sorted = Array.prototype.slice.call(buf, 0, n).sort((a, b) => a - b);
  let radius = sorted[Math.min(n - 1, Math.floor(n * SMOKE_SPRITE_RADIUS_PCT))];
  if (!(radius > 0)) radius = sorted[n - 1];
  if (!(radius > 0)) { s.mesh.visible = false; return; }
  if (radius > SMOKE_SPRITE_MAX_RADIUS_HU) radius = SMOKE_SPRITE_MAX_RADIUS_HU;

  const c = gameToThree(mx, my, mz);
  s.mesh.position.set(c.x, c.y, c.z);
  // 一个立方体刚好外接半径 r 的球 → 边长 2r，比原来的 1.15 倍还稳
  s.mesh.scale.set(radius * 2 * SCALE, radius * 2 * SCALE, 1);
  s.mesh.visible = _smokeSpritesWanted && !!s.mesh.material.map;
}

/**
 * 按时间翻帧。时间基准取**该烟雾引爆后的秒数**，所以两团不同时间扔的烟
 * 不会同步翻帧 —— 同步会立刻露馅。
 */
function _updateSmokeSpriteFrames(timeSec) {
  const seq = _smokeSpriteSeq;
  if (!seq) return;
  const n = seq.textures.length;
  for (const [eid, s] of smokeSprites) {
    if (!s.mesh.visible) continue;
    const inst = _smokeSpriteBirth.get(eid);
    const age = inst != null ? Math.max(0, timeSec - inst) : 0;
    const idx = Math.floor(age * smokeSpriteFPS) % n;
    const tex = seq.textures[idx];
    if (tex && s.mesh.material.map !== tex) {
      s.mesh.material.map = tex;
      // 同序列换帧不需要 needsUpdate（原因见 _updateFireFrames）
    }
  }
}

/** entity_id -> 引爆时刻（秒），由 updateSmokeVolumes 落盘 */
const _smokeSpriteBirth = new Map();

// ═══════════════════════════════════════════════════════════════════════════
// 真实火焰格子
// ═══════════════════════════════════════════════════════════════════════════

const FIRE_CELLS = 16;             // m_firePositions 固定 16 格

// 每格画一片**平铺在表面上的火贴片**（不是立起来的火柱）。
//
// 实测数据（`test.dem` 58 处 + de_inferno 144 处，共 202 处火焰）：16 格坐标在整个
// 燃烧期间**完全不动**（每个下标只有 1 个不同坐标），15,418 个燃烧格法向里 96% 的
// `normal.y ≥ 0.9`（中位数 ≈ 1.000），只有 3~4% 明显倾斜。也就是说这 16 格是
// **表面上的燃烧点**，火是"一滩"铺开的，不是 16 根柱子。
//
// PlaneGeometry 的默认法向是 +Z；绕 X 转 -90° 后法向变成 +Y，之后用
// "表面法向"对齐四元数（+Y → normal）就正好让贴片躺在那个面上。
const FIRE_PATCH_M = 1.35;         // 单格贴片直径（米）；相邻格间距中位数 2.19 m，靠叠加连成一片
const fireGeo = new THREE.PlaneGeometry(FIRE_PATCH_M, FIRE_PATCH_M);
fireGeo.rotateX(-Math.PI / 2);

/** eid -> {mesh, light, cells:[{pos, quat, phase}]} */
const fireCells = new Map();

/** 程序生成的径向渐变贴图：中心亮黄白 → 边缘橙红 → 透明，像一滩流动的火。 */
function makeFireTexture() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0.00, 'rgba(255, 244, 205, 1.00)');
  g.addColorStop(0.28, 'rgba(255, 190,  70, 0.88)');
  g.addColorStop(0.58, 'rgba(240, 110,  22, 0.48)');
  g.addColorStop(0.82, 'rgba(170,  50,   6, 0.16)');
  g.addColorStop(1.00, 'rgba(120,  30,   0, 0.00)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

let _fireTex = null;

// ── 火焰三层 sprite（core / filler / edge）──────────────────────────────────
//
// CS2 的火焰不是单层贴图，而是三张序列按不同尺寸叠出来的：
//
//   层        序列             相对尺寸   不透明度   翻帧速率   作用
//   core      flames           0.55       0.95       24 fps     亮芯
//   filler    flame_omni       0.85       0.62       18 fps     主体
//   edge      flame_wisp       1.20       0.34       13 fps     外沿飘散
//
// 三层的**翻帧速率和相位都不同**，否则会看出"三层一模一样在同步闪"，一眼假。
// 每层各自有起始帧偏移（phase 是 0~1 的圈数比例）。
//
// ── 为什么每层还要一个 tint（这一步一开始漏了，火是白的）──────────────────
//
// 这三张序列是**灰度亮度遮罩，不是彩色贴图**。实测（取中间帧、只统计 alpha>10
// 的像素，`max-min` 是每像素的通道极差）：
//
//   序列            平均 RGB         平均通道极差
//   flames          ( 65,  65,  65)        1.5      <- 纯灰度，很暗
//   flame_omni      (105, 105, 105)        1.6      <- 纯灰度
//   flame_wisp      (234, 234, 234)        1.5      <- 纯灰度，接近白
//   fire_onsurface  (155,  67,  16)      138.8      <- 只有这张是彩色的
//
// 真实颜色来自 vpcf 里的粒子颜色渐变，贴图只负责"哪里亮、有多亮"。直接把它
// 按颜色图贴上去，等于算 `白 × 灰度`，结果就是一片白 —— 尤其 `flame_wisp`
// 是 234 的近白而且尺寸最大（1.20），整片火会糊成白的。
//
// 所以每层给一个暖色 tint。`MeshBasicMaterial` 的 color 允许 >1：加色混合下
// 这就是"增益"，而渲染器开了 ACES tone mapping（`scene.js`，exposure 0.95），
// 超过 1 的部分会被压回高光而不是硬切 —— 正好是火芯该有的过曝滚降。
//
// tint 的数值**不是猜的**，是用离线资产管线按 three
// 的实际 shader 顺序（sRGB 解码 → 乘 tint×opacity → **逐层** ACES → sRGB 编码 →
// 加色混合 `rgb × (a × opacity)`）合成出来量的。指标是可见像素的 HSV 饱和度分布：
//
//   配色                        平均色相   低饱和(<0.20)占比   亮部 RGB
//   旧行为（无 tint）              --        100.0%          (0.601,0.601,0.601)  <- 纯灰
//   A                           33.8deg       8.1%           (0.685,0.575,0.395)
//   B（采用）                    33.5deg       3.3%           (0.667,0.526,0.294)
//   C                           34.5deg       5.5%           (0.656,0.529,0.316)
//   D                           33.8deg       1.1%           (0.653,0.503,0.249)
//
// 旧行为三通道**完全相等**、100% 低饱和 —— 这就是"火是白的"的实测证据。
// 选 B：色相 33.5°（橙）、只剩 3.3% 近白像素（那点就是白热芯，真实火也有）。
//
// 量级按各层贴图的**平均亮度**配：贴图暗的层（core 线性均值 0.042）给大增益，
// 贴图亮的层（edge 0.312）给小增益，三层叠加后的亮度才均衡。
//
// 每格仍然躺在真实表面法向上（`m_firePositions` + `m_fireNormals`），
// 只是每格画三片而不是一片。
const FIRE_LAYERS = [
  { key: 'core',   seq: 'flames',     scale: 0.55, opacity: 0.95, fps: 24, phase: 0.00,
    tint: [3.40, 2.00, 0.70] },   // 黄白亮芯：贴图最暗，增益最大
  { key: 'filler', seq: 'flame_omni', scale: 0.85, opacity: 0.62, fps: 18, phase: 0.37,
    tint: [2.60, 1.05, 0.20] },   // 橙
  { key: 'edge',   seq: 'flame_wisp', scale: 1.20, opacity: 0.34, fps: 13, phase: 0.71,
    tint: [1.60, 0.42, 0.07] },   // 暗橙红：贴图近白，增益最小，否则一片白
];

/** 层的暖色 tint（>1 的分量是加色增益，靠 ACES 压回高光）。 */
function _layerTint(L) {
  const t = L.tint || [1, 1, 1];
  return new THREE.Color(t[0], t[1], t[2]);
}

/** 序列名 -> SpriteSequence | null（null = 还在加载或加载失败） */
const _fireSeqs = new Map();

function _ensureFireSeqs() {
  for (const L of FIRE_LAYERS) {
    if (_fireSeqs.has(L.seq)) continue;
    _fireSeqs.set(L.seq, null);              // 先占位，避免并发重复发起
    loadSpriteSequence(L.seq).then((seq) => {
      if (!seq) return;
      _fireSeqs.set(L.seq, seq);
      // 序列迟到：把已有材质的兜底贴图换成真序列的第 0 帧
      for (const [, f] of fireCells) {
        const layer = f.layers && f.layers[L.key];
        if (layer && layer.mesh.material.map === _fireTex) {
          layer.mesh.material.map = seq.textures[0];
          layer.mesh.material.needsUpdate = true;
        }
      }
    }).catch(() => {});
  }
}

function _makeFireLayerMaterial(L, seq) {
  return new THREE.MeshBasicMaterial({
    // 序列没到之前用程序生成的软斑兜底，避免整片火是纯色方块
    map: seq ? seq.textures[0] : _fireFallback(),
    // 贴图是灰度亮度遮罩，颜色靠这个 tint —— 不设的话火是白的（见 FIRE_LAYERS 上方）
    color: _layerTint(L),
    transparent: true,
    opacity: L.opacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

function getFireCell(inst) {
  let f = fireCells.get(inst.entity_id);
  if (!f) {
    _ensureFireSeqs();
    const group = new THREE.Group();
    group.name = `fire-${inst.entity_id}`;
    scene.add(group);

    const layers = {};
    for (const L of FIRE_LAYERS) {
      const seq = _fireSeqs.get(L.seq) || null;
      const mesh = new THREE.InstancedMesh(fireGeo, _makeFireLayerMaterial(L, seq), FIRE_CELLS);
      mesh.name = `fire-${L.key}-${inst.entity_id}`;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.count = 0;
      mesh.renderOrder = 960;
      group.add(mesh);
      layers[L.key] = { cfg: L, mesh };
    }

    const light = new THREE.PointLight(0xff7020, 3.0, 12, 2);
    light.visible = false;
    scene.add(light);

    f = {
      group, layers, light, cells: [],
      inferno_type: inst.inferno_type, type_name: inst.type_name,
    };
    fireCells.set(inst.entity_id, f);
  }
  return f;
}

export function resetFireCells() {
  for (const [, f] of fireCells) {
    for (const k of Object.keys(f.layers || {})) {
      const m = f.layers[k].mesh;
      f.group.remove(m);
      m.material.dispose();
    }
    scene.remove(f.group);
    scene.remove(f.light);
  }
  fireCells.clear();
}

/** 找到 tick 之前最近的一次快照。 */
function fireSnapshotAt(inst, tick) {
  const changes = inst.changes || [];
  if (!changes.length || tick < changes[0].tick) return null;
  let lo = 0, hi = changes.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (changes[mid].tick <= tick) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return changes[ans];
}

/**
 * 更新当前 tick 的火焰格子。
 * 每格有真实 3D 坐标 + 表面法向；贴片**躺在**该表面上（法向决定朝哪一面）。
 * 16 格位置固定不动，火势蔓延只体现在"哪些格在烧"，所以贴片逐格出现即可。
 */
export function updateFireCells(instances, tick) {
  const active = new Set();
  for (const inst of instances) {
    if (tick < inst.first_tick || tick > inst.last_tick) continue;
    const snap = fireSnapshotAt(inst, tick);
    if (!snap) continue;

    const burning = snap.burning || [];
    const pos = snap.pos || [];
    const nrm = snap.normal || [];
    let n = Math.min(burning.length, snap.count || burning.length, FIRE_CELLS);
    let alive = 0;
    for (let i = 0; i < n; i++) if (burning[i]) alive++;
    if (!alive) { const f = getFireCell(inst); _setFireCount(f, 0); f.light.visible = false; continue; }

    active.add(inst.entity_id);
    const f = getFireCell(inst);
    f.cells = [];
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
      if (!burning[i]) continue;
      const g = gameToThree(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      const normal = gameDirToThree(nrm[i * 3] || 0, nrm[i * 3 + 1] || 0, nrm[i * 3 + 2] || 1);
      const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);
      // 贴片正好贴在表面上会和地面共面 → z-fighting，沿法向抬 1.5 cm。
      const p = new THREE.Vector3(g.x, g.y, g.z).addScaledVector(normal, 0.015);
      f.cells.push({ pos: p, normal, quat, phase: (i * 1.7) % (Math.PI * 2) });
      cx += g.x; cy += g.y; cz += g.z;
    }
    _setFireCount(f, f.cells.length);
    f.light.position.set(cx / f.cells.length, cy / f.cells.length + 0.4, cz / f.cells.length);
    f.light.visible = true;
    applyFireMatrices(f, performance.now() / 1000);
  }

  for (const [eid, f] of fireCells) {
    if (!active.has(eid)) { _setFireCount(f, 0); f.light.visible = false; }
  }
}

/** 三层一起改 count，避免漏掉某层导致"只剩芯"或"只剩外沿"。 */
function _setFireCount(f, n) {
  for (const k of Object.keys(f.layers)) f.layers[k].mesh.count = n;
}

/**
 * 火焰贴图层开关。关掉时**保留**贴片几何，只是把三层都换成程序生成的兜底贴图 ——
 * 这样"关掉"等于"退回旧的单层程序贴图外观"，而不是火直接消失。
 */
let _fireSpritesVisible = true;
export function setFireSpritesVisible(v) {
  _fireSpritesVisible = !!v;
  for (const [, f] of fireCells) {
    for (const k of Object.keys(f.layers)) {
      const layer = f.layers[k];
      const seq = _fireSeqs.get(layer.cfg.seq);
      const want = _fireSpritesVisible && seq ? seq.textures[0] : _fireFallback();
      if (layer.mesh.material.map !== want) {
        layer.mesh.material.map = want;
        layer.mesh.material.needsUpdate = true;
      }
      // 关掉时回到单一不透明度，不再按层区分
      layer.mesh.material.opacity = _fireSpritesVisible ? layer.cfg.opacity : 0.90;
    }
  }
}

function _fireFallback() {
  if (!_fireTex) _fireTex = makeFireTexture();
  return _fireTex;
}

/**
 * 预热火焰 / 烟雾 sprite 的 shader，**在加载阶段调用**。
 *
 * 火焰的材质是第一次看到火时才创建的（`getFireCell`），three 会在那一帧
 * **同步编译** 3 个新 program。几十毫秒的停顿正好落在"火刚烧起来"的瞬间，
 * 看起来就是"火焰出现的瞬间卡一下"。
 *
 * 这里用**完全相同的材质配置**在远离地图的地方画一帧，把 program 提前编译好。
 * `renderer.compile()` 本身就是同步的，但它发生在加载遮罩还盖着的时候，
 * 用户看不到。
 *
 * **必须等序列到齐**：`map` 从 null 变成贴图会改 `USE_MAP` 宏、换一个 program，
 * 用兜底贴图预热等于预热了个别的，白等。所以这个函数是 async 的。
 *
 * 用完即弃：几何 `fireGeo` 是共享的不能 dispose，材质是本函数自己建的，要。
 */
export async function prewarmFireShaders() {
  if (!renderer) return false;
  _ensureFireSeqs();
  _ensureSmokeSpriteSeq();

  // loadSpriteSequence 自带 promise 缓存，这里不会重复下载
  await Promise.all([
    ...FIRE_LAYERS.map((L) => loadSpriteSequence(L.seq).catch(() => null)),
    loadSpriteSequence(SMOKE_SPRITE_SEQ).catch(() => null),
  ]);

  const warm = new THREE.Group();
  warm.name = 'shader-prewarm';
  warm.position.set(0, -10000, 0);   // 地图外
  const meshes = [];

  const m0 = new THREE.Matrix4();
  for (const L of FIRE_LAYERS) {
    const mesh = new THREE.InstancedMesh(
      fireGeo, _makeFireLayerMaterial(L, _fireSeqs.get(L.seq) || null), 1);
    mesh.count = 1;
    mesh.frustumCulled = false;
    mesh.setMatrixAt(0, m0);
    warm.add(mesh);
    meshes.push(mesh);
  }

  // 烟雾 sprite 也是"第一团烟出现时"才编译
  const smokeMat = new THREE.SpriteMaterial({
    transparent: true, depthWrite: false, opacity: 0.55,
    map: _smokeSpriteSeq ? _smokeSpriteSeq.textures[0] : _fireFallback(),
  });
  const smokeWarm = new THREE.Sprite(smokeMat);
  smokeWarm.frustumCulled = false;
  warm.add(smokeWarm);

  scene.add(warm);
  try {
    // scene.traverse 会走完整个场景（含 Sprite），所以这一下把地图/玩家的
    // 材质也一起编译了 —— 正是我们想要的
    renderer.compile(scene, camera);
  } catch (e) {
    console.warn('[visuals] shader 预热失败（不影响功能）：', e.message);
    return false;
  } finally {
    scene.remove(warm);
    for (const m of meshes) m.material.dispose();
    smokeMat.dispose();
  }
  return true;
}

const _fq = new THREE.Quaternion();
const _fs = new THREE.Vector3();
const _fm = new THREE.Matrix4();
const _fp = new THREE.Vector3();

/**
 * 贴片只在**水平面内**缩放（x/z 同步，y 恒为 1），所以无论怎么闪都还是平的。
 * 旧版缩放 y 把锥体拉高，就是"地刺"的来源。
 */
function applyFireMatrices(f, t) {
  const nc = f.cells.length;
  for (const k of Object.keys(f.layers)) {
    const layer = f.layers[k];
    const L = layer.cfg;
    const lift = (L.scale - FIRE_LAYERS[0].scale) * 0.04;
    for (let i = 0; i < nc; i++) {
      const c = f.cells[i];
      const flick = 1.0
        + 0.13 * Math.sin(t * 5.5 + c.phase)
        + 0.07 * Math.sin(t * 12.7 + c.phase * 2.1);
      const s = Math.max(0.62, flick) * L.scale;
      _fs.set(s, 1.0, s);
      _fp.copy(c.pos);
      if (lift > 0) _fp.addScaledVector(c.normal, lift);
      _fm.compose(_fp, c.quat, _fs);
      layer.mesh.setMatrixAt(i, _fm);
    }
    layer.mesh.instanceMatrix.needsUpdate = true;
  }
}

/** 三层各按自己的 fps 和相位翻帧（相位错开，避免同步闪）。 */
function _updateFireFrames(timeSec) {
  if (!_fireSpritesVisible) return;      // 关掉时保持兜底贴图，不要被翻帧覆盖
  for (const [, f] of fireCells) {
    if (!f.cells.length) continue;
    for (const k of Object.keys(f.layers)) {
      const layer = f.layers[k];
      const seq = _fireSeqs.get(layer.cfg.seq);
      if (!seq || !seq.textures.length) continue;
      const n = seq.textures.length;
      const idx = Math.floor(timeSec * layer.cfg.fps + layer.cfg.phase * n) % n;
      const tex = seq.textures[idx];
      if (tex && layer.mesh.material.map !== tex) {
        layer.mesh.material.map = tex;
        // **不要设 material.needsUpdate = true。**
        //
        // 换的是同一序列里的另一帧，`colorSpace`/`flipY`/`mapping` 全都一样，
        // 所以 shader 源码不变，three 每帧自己会把新的 map uniform 传下去。
        // 设 needsUpdate 会 bump material.version，逼 `WebGLRenderer.setProgram`
        // 每帧重走一遍 program 判定 —— 三层加起来 24+18+13 = 55 次/秒·每个燃烧
        // 实体，正是"火一出现就卡一下"的来源之一。
        //
        // （只有 map 从 null 变成贴图时才真的需要 needsUpdate：那时 USE_MAP
        //   这个宏变了，shader 得重编译。那几处仍然保留。）
      }
    }
  }
}

/** 火焰动画：整片亮度呼吸 + 贴片轻微胀缩 + 三层各自翻帧 + 点光源呼吸。 */
export function updateInfernoAnimations(dt, timeSec) {
  const t = performance.now() / 1000;
  _updateFireFrames(t);
  for (const [, f] of fireCells) {
    if (!f.cells.length) continue;
    applyFireMatrices(f, t);
    // 各层在自己的基准不透明度上呼吸；基准值来自 FIRE_LAYERS
    const breathe = 0.80 + 0.14 * Math.sin(t * 3.1);
    for (const k of Object.keys(f.layers)) {
      const layer = f.layers[k];
      layer.mesh.material.opacity =
        (_fireSpritesVisible ? layer.cfg.opacity : 0.90) * breathe;
    }
    f.light.intensity = 2.4 + 1.0 * Math.sin(t * 7.3);
  }
}

/** 当前火焰格子数（给"数据检查"面板用）。 */
export function fireDebugStats() {
  const out = [];
  for (const [eid, f] of fireCells) {
    const core = f.layers && f.layers.core;
    out.push({
      entity_id: eid,
      type_name: f.type_name,
      cells: core ? core.mesh.count : 0,
      layers: f.layers ? Object.keys(f.layers).length : 0,
    });
  }
  return out;
}

// ── 枪口火焰 / 脚步 / 击杀 ──────────────────────────────────────────────────

const muzzleGeo = new THREE.SphereGeometry(0.5, 8, 8);
const muzzleMat = new THREE.MeshBasicMaterial({ color: 0xffcc44, transparent: true, opacity: 0.95 });
const muzzleFlashPool = [];
const MAX_MUZZLE = 20;

const fsGeo = new THREE.RingGeometry(0.2, 0.6, 16);
const fsMatTemplate = new THREE.MeshBasicMaterial({
  color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.8,
  depthTest: true, depthWrite: false,
});
const footstepPool = [];

export function spawnMuzzleFlash(playerIdx, pos) {
  let flash = muzzleFlashPool.find(m => !m.visible);
  if (flash) {
    flash.visible = true;
  } else {
    flash = new THREE.Mesh(muzzleGeo, muzzleMat.clone());
    flash.userData = { life: 0 };
    scene.add(flash);
    muzzleFlashPool.push(flash);
  }
  flash.position.set(pos.x, pos.y + 1.5, pos.z);
  flash.userData.life = 0.2;
  flash.material.opacity = 0.95;
  flash.scale.set(1, 1, 1);
}

export function updateMuzzleFlashes(dt) {
  for (const flash of muzzleFlashPool) {
    if (!flash.visible) continue;
    flash.userData.life -= dt;
    if (flash.userData.life <= 0) {
      flash.visible = false;
    } else {
      const t = flash.userData.life / 0.2;
      flash.material.opacity = t * 0.95;
      flash.scale.setScalar(1 + (1 - t) * 1.5);
    }
  }
}

export function spawnFootstep(playerIdx, pos) {
  let ring = footstepPool.find(r => !r.visible);
  if (!ring) {
    ring = new THREE.Mesh(fsGeo, fsMatTemplate.clone());
    ring.rotation.x = -Math.PI / 2;
    ring.userData = { life: 0 };
    ring.renderOrder = 997;
    ring.material.depthTest = true;
    ring.material.depthWrite = false;
    scene.add(ring);
    footstepPool.push(ring);
  }
  ring.position.set(pos.x, pos.y + 0.05, pos.z);
  ring.userData.life = 0.5;
  ring.visible = true;
  ring.scale.set(1, 1, 1);
}

export function updateFootsteps(dt) {
  for (const ring of footstepPool) {
    if (!ring.visible) continue;
    ring.userData.life -= dt;
    if (ring.userData.life <= 0) {
      ring.visible = false;
    } else {
      const t = ring.userData.life / 0.5;
      ring.scale.setScalar(1 + (1 - t) * 3);
      ring.material.opacity = t * 0.7;
    }
  }
}

export function showKillEffect(pos) {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.3, 0.5, 16),
    new THREE.MeshBasicMaterial({
      color: 0xff0000, side: THREE.DoubleSide,
      transparent: true, opacity: 0.9, depthTest: false, depthWrite: false,
    })
  );
  ring.position.copy(pos);
  ring.position.y += 1.2;
  ring.name = 'kill-effect';
  scene.add(ring);
  setTimeout(() => {
    scene.remove(ring);
    ring.geometry.dispose();
    ring.material.dispose();
  }, 1000);
}

// ── 清理 ────────────────────────────────────────────────────────────────────

export function clearAllEntities() {
  clearGrenadeTrajectories();
  for (const r of aimRays) r.line.visible = false;
  // 换回合要收掉还在飞的曳光：它是上一回合的枪
  clearTracers();

  resetSmokeVolumes();
  resetSmokeSprites();
  resetFireCells();
  // 武器模型 / 图标 badge：换回合必须重建。武器的显示名会变，而且模型是挂在
  // 玩家 group 上的子节点，`createPlayers()` 会重建 group —— 不清理会留下
  // 指向已销毁 group 的悬挂引用。
  clearWeaponModels();
  _weaponModelKey.clear();
  clearWeaponBadges();

  scene.children
    .filter(c => c.name === 'kill-effect')
    .forEach(c => {
      scene.remove(c);
      c.geometry?.dispose();
      c.material?.dispose();
    });

  for (const f of muzzleFlashPool) f.visible = false;
  for (const r of footstepPool) r.visible = false;

  clearAllTrails();
  if (bombGroup) bombGroup.visible = false;
}
