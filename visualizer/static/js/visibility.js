/**
 * visibility.js — 「B 能不能观测到 A」的纯几何判定（不 import three）。
 *
 * 和 `.csz` 里的 `spotted` 列**不是一回事**：
 *
 *   spotted / m_bSpottedByMask   引擎给的，含视锥 + 声音 + 引擎自己的时序，
 *                                而且实测被限速到 **33 tick（0.516 s）的网格**
 *   visibility.js                纯几何：眼睛 → 目标包围盒 8 个角落的射线，
 *                                逐帧算，没有延迟，没有声音，没有引擎黑盒
 *
 * 判定一个角落「通」需要三件事同时成立：
 *   ① 射线方向在观察者的视锥内（水平 FOV × 垂直 FOV，矩形视锥不是圆锥）
 *   ② 眼睛到角落之间没有地图三角形
 *   ③ 眼睛到角落之间没有烟雾体素
 * 8 个角落里**任意一个通**就算「A 能被 B 观测到」（默认；`exactCount` 见下）。
 *
 * ── 为什么是纯模块 ─────────────────────────────────────────────────────────
 *
 * 和 map-raycast.js / smoke-voxels.js 一样不 import three，所以能在 node 里
 * 直接跑对拍测试（visualizer/tests/test_visibility.mjs）。地图遮挡通过
 * **注入的 `castMapRay` 回调**进来，于是本模块对坐标系/单位完全无感 ——
 * 它只要求「玩家坐标」和「castMapRay 的入参」在同一个空间里。
 * 实际接线时用的是 Hammer 单位（game 空间），由 makeMapCaster() 负责转到
 * BVH 所在的 three 空间。
 *
 * ── 性能 ───────────────────────────────────────────────────────────────────
 *
 * 10 人 → 90 个有序对 × 8 个角落 = 720 条射线。BVH 实测 0.0020 ms/条，
 * 加上 FOV 早退（视锥外的角落根本不发射线）和「第一个通的角落就收工」，
 * 实测约 **0.6 ms/帧**（见 test_visibility.mjs 的性能用例）。
 */

import { raycastBVH } from './map-raycast.js';
import { eyeHeightFor, hullHeightFor } from './replay-core.js';

const DEG2RAD = Math.PI / 180;

// ── 常量 ────────────────────────────────────────────────────────────────────

/** 玩家碰撞盒半宽：32 HU / 2。世界轴对齐，不随 yaw 转（= m_vecMins/m_vecMaxs）。 */
export const HULL_HALF_HU = 16;

/** 站立 / 蹲下时的碰撞盒高度（HU）。和 hullHeightFor() 一致，列在这里是为了自解释。 */
export const HULL_STAND_HU = 72;
export const HULL_CROUCH_HU = 54;

/**
 * 8 个角落往盒内收多少 HU。
 *
 * 不收的话脚部那 4 个角落**正好落在地板平面上**：射线到它们的命中距离和
 * maxDist 在浮点上分不开，而 raycastBVH 用的是严格 `hit < best` ——
 * 误差符号随机，同一个姿势会时通时不通（帧间闪烁）。收 1 HU（2.54 cm）之后
 * 射线终点落在几何体内部，判定稳定。
 *
 * 收的是**判定盒**，不是碰撞盒本身：§4.6 的碰撞盒绘制和这里的判定互不影响。
 */
export const HULL_INSET_HU = 1.0;

export const CORNER_COUNT = 8;

/**
 * 水平 FOV（度）。**这个 90 是挑出来的常量，不是从游戏里提取的。**
 *
 * 实测（de_inferno，aurora-vs-vitality m2，2,172,400 行）说明它「对一半」：
 *
 * * 引擎自己的默认值**就是 90** —— 玩家状态里的 `m_iFOV` 只有 `0`（=用默认，
 *   95.23% 的活人 tick）、`90`（2.36%，默认值显式写出来）、`40`（2.04%）、
 *   `10`（0.37%）、`15`（~0）这 5 个值。所以**不开镜时 90 是对的**。
 * * 但 `40 / 10 / 15` 是**开镜的倍镜档位**，而且是坐实的不是猜的：`40` 有 97.8%
 *   出现在持 AWP 时（其余 2.2% 是 SSG 08），`10` 100% 是 AWP，`15` 100% 是
 *   SSG 08。**持 AWP 时有 64.7% 的 tick 在 40 或 10。**
 *   也就是说：一个架枪的狙击手视野是 40° 或 10°，这里给他按 90° 算 ——
 *   **恰好错在可见性工具最该看对的场景上。**
 *
 * 真字段是 `CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOV`（`0` = 用默认 90）。
 * ⚠️ **不是** `CCSPlayerController.m_iDesiredFOV` —— 那个在 demo 里 153 万行
 * **全是 0**，是个死字段。这个字段死人身上不会留陈旧值（死人 100% 是 0），
 * 不像 `spotted` 那样要自己滤。提取成本几乎为零：全场 10 个人一共只有
 * **394 次变化**（每 tick 存的话是 2,172,400 条），和 `spotted` 的 740 次同量级。
 *
 * **当前决定：不提取、就用这个常量**（2026-02 确认）。要接真值的话，`fovDeg` /
 * `vFovDeg` 这两个选项已经在了，`extract.py` 加一列就能覆盖。
 *
 * ⚠️ **还有一个没验证的点**：`m_iFOV=90` 到底是「16:9 下的水平角」还是 Source
 * 传统的「4:3 基准水平角」（宽屏按 Hor+ 横向加宽）。如果是后者，16:9 下真实水平角
 * 是 **≈106.26°**、垂直角 ≈73.74°，那这里 90°/58.7° **两个方向都窄了**。
 * 没能验证 —— Valve wiki 连不上，搜索也拿不到。**标为未验证，不猜。**
 */
export const DEFAULT_FOV_DEG = 90;

/** 屏幕宽高比，用来从水平 FOV 推垂直 FOV。 */
export const DEFAULT_ASPECT = 16 / 9;

// 遮挡原因位（可以按位或，所以一个对可能同时「有角落被烟挡」和「有角落不在视野」）
export const BLOCK_FOV = 1;
export const BLOCK_MAP = 2;
export const BLOCK_SMOKE = 4;
export const BLOCK_PLAYER = 8;

// ── 视角基 ──────────────────────────────────────────────────────────────────

/**
 * 观察者的眼睛世界坐标（game 空间），写进 out。
 *
 * `.csz` 的 `x/y/z` 是**脚底**，眼高是相对脚底算的，所以眼睛是
 * `z + eye_height`。状态里带了 `eye_height`（`getPlayerState()` 算好的连续值）
 * 就用它，否则按 `crouch` 现算 —— 和 visuals.js::eyeRay() 是同一条规则。
 */
export function eyePos(state, out) {
  out[0] = state.x;
  out[1] = state.y;
  out[2] = eyeZ(state);
  return out;
}

/** 上面那条规则本身。单独抽出来是因为热路径里不想为了取个 z 去建数组。 */
export function eyeZ(state) {
  return state.z + (state.eye_height != null
    ? state.eye_height : eyeHeightFor(state.crouch || 0));
}

/**
 * 观察者的正交基，写进 out（9 个 float：forward, right, up）。
 *
 * 约定和 visuals.js 的 `eyeRay()` **完全一致**（瞄准射线/曳光/这里的可见性
 * 必须是同一个朝向，否则「同一帧同一个人」三套东西会对不上）：
 * three 空间里 forward 是 (0,0,1) 先绕 X 转 pitch、再绕 Y 转 yaw；置换回
 * game 空间（three = (gy, gz, gx)）就是
 *
 *   f = ( cos y·cos p,  sin y·cos p,  −sin p )
 *   r = ( sin y,       −cos y,         0     )
 *   u = ( cos y·sin p,  sin y·sin p,   cos p )
 *
 * 验算：yaw=0,pitch=0 → f=(1,0,0)（Source 里 yaw 0 看 +X）、r=(0,−1,0)
 * （forward×up = X×Z = −Y）、u=(0,0,1)。yaw=90 → f=(0,1,0)。pitch>0 向下看
 * （Source 约定），f.z = −sin p < 0。都对。
 */
export function viewBasis(yaw, pitch, out) {
  const y = yaw * DEG2RAD, p = pitch * DEG2RAD;
  const cy = Math.cos(y), sy = Math.sin(y);
  const cp = Math.cos(p), sp = Math.sin(p);
  out[0] = cy * cp; out[1] = sy * cp; out[2] = -sp;
  out[3] = sy;      out[4] = -cy;     out[5] = 0;
  out[6] = cy * sp; out[7] = sy * sp; out[8] = cp;
  return out;
}

/** 方向向量（已归一化）在不在视锥里。矩形视锥：水平/垂直各自一个半角。 */
export function dirInFov(basis, dx, dy, dz, tanHalfH, tanHalfV) {
  const fd = dx * basis[0] + dy * basis[1] + dz * basis[2];
  if (fd <= 0) return false;                       // 在背后
  const rd = dx * basis[3] + dy * basis[4] + dz * basis[5];
  const ud = dx * basis[6] + dy * basis[7] + dz * basis[8];
  // |r| <= f·tanH 等价于 atan(|r|/f) <= H，但省一次 atan2（热路径）
  const limH = fd * tanHalfH;
  if (rd > limH || rd < -limH) return false;
  const limV = fd * tanHalfV;
  if (ud > limV || ud < -limV) return false;
  return true;
}

// ── 包围盒 8 个角落 ─────────────────────────────────────────────────────────

/**
 * 目标玩家的 8 个角落，game 空间，写进 out（24 个 float）。
 *
 * @param {number} x y z  **脚底**的世界坐标（和 .csz 的 x/y/z 一样）
 * @param {number} height 碰撞盒高度（HU）：站立 72 / 蹲下 54
 */
export function playerCorners(x, y, z, height, out, inset = HULL_INSET_HU) {
  const hw = HULL_HALF_HU - inset;
  const z0 = z + inset;
  const z1 = z + height - inset;
  let w = 0;
  for (let k = 0; k < 2; k++) {
    const zz = k ? z1 : z0;
    for (let j = 0; j < 2; j++) {
      const yy = j ? y + hw : y - hw;
      for (let i = 0; i < 2; i++) {
        out[w++] = i ? x + hw : x - hw;
        out[w++] = yy;
        out[w++] = zz;
      }
    }
  }
  return out;
}

// ── 烟雾：射线 vs 体素栅格的 3D DDA ─────────────────────────────────────────

/**
 * 一个烟雾实例的遮挡体。
 * @typedef {{det: number[], grid: Uint8Array, dim: number, cell: number}} SmokeVolume
 *   det  引爆点（game 空间）
 *   grid 长度 dim³ 的占据栅格，展平下标 = (i·dim + j)·dim + k
 *   cell 体素边长（HU），VOXEL_WORLD_SIZE = 12
 */

/**
 * 射线 vs 一个烟雾栅格，Amanatides–Woo 3D DDA。
 *
 * 栅格约定和 smoke-voxels.js 的 `gridToWorldPoints()` 一致：
 * 格子 (i,j,k) 的**中心** = det + (i − 15.5)·cell，也就是格子 i 覆盖
 * det + (i−16)·cell .. det + (i−15)·cell。整个烟是一个 32·cell = 384 HU
 * （9.75 m）的立方体。
 *
 * 返回 true = 射线在 maxDist 之内穿过了至少一个**被占据**的格子。
 *
 * @param {number} maxDist 只关心这个距离之内；通常是眼睛到角落的距离
 */
export function smokeRayBlocked(volumes, ox, oy, oz, dx, dy, dz, maxDist) {
  if (!volumes || volumes.length === 0) return false;
  for (let v = 0; v < volumes.length; v++) {
    if (ddaBlocked(volumes[v], ox, oy, oz, dx, dy, dz, maxDist)) return true;
  }
  return false;
}

function ddaBlocked(vol, ox, oy, oz, dx, dy, dz, maxDist) {
  const D = vol.dim, cell = vol.cell;
  const half = D * cell * 0.5;
  const bx = vol.det[0] - half, by = vol.det[1] - half, bz = vol.det[2] - half;

  // ① 先做 AABB 拒绝：绝大多数射线根本不在这个烟的 384 HU 立方里。
  //    这一步把每个烟的成本从「DDA 几十步」压到「3 个 slab」。
  let t0 = 0, t1 = maxDist;
  {
    let a, b;
    if (dx > -1e-12 && dx < 1e-12) {
      if (ox < bx || ox > bx + D * cell) return false;
    } else {
      const inv = 1 / dx;
      a = (bx - ox) * inv; b = (bx + D * cell - ox) * inv;
      if (a > b) { const t = a; a = b; b = t; }
      if (a > t0) t0 = a;
      if (b < t1) t1 = b;
      if (t0 > t1) return false;
    }
    if (dy > -1e-12 && dy < 1e-12) {
      if (oy < by || oy > by + D * cell) return false;
    } else {
      const inv = 1 / dy;
      a = (by - oy) * inv; b = (by + D * cell - oy) * inv;
      if (a > b) { const t = a; a = b; b = t; }
      if (a > t0) t0 = a;
      if (b < t1) t1 = b;
      if (t0 > t1) return false;
    }
    if (dz > -1e-12 && dz < 1e-12) {
      if (oz < bz || oz > bz + D * cell) return false;
    } else {
      const inv = 1 / dz;
      a = (bz - oz) * inv; b = (bz + D * cell - oz) * inv;
      if (a > b) { const t = a; a = b; b = t; }
      if (a > t0) t0 = a;
      if (b < t1) t1 = b;
      if (t0 > t1) return false;
    }
  }
  if (t1 < 0) return false;

  // ② 稍微推进一点，保证起点严格在盒内（贴着面进的时候 floor 会落到 D）
  t0 += 1e-6;
  if (t0 > t1) return false;

  const px = ox + dx * t0, py = oy + dy * t0, pz = oz + dz * t0;
  const gx = (px - bx) / cell, gy = (py - by) / cell, gz = (pz - bz) / cell;
  let i = Math.floor(gx), j = Math.floor(gy), k = Math.floor(gz);
  if (i < 0) i = 0; else if (i >= D) i = D - 1;
  if (j < 0) j = 0; else if (j >= D) j = D - 1;
  if (k < 0) k = 0; else if (k >= D) k = D - 1;

  const stepX = dx > 0 ? 1 : (dx < 0 ? -1 : 0);
  const stepY = dy > 0 ? 1 : (dy < 0 ? -1 : 0);
  const stepZ = dz > 0 ? 1 : (dz < 0 ? -1 : 0);

  const adx = dx < 0 ? -dx : dx, ady = dy < 0 ? -dy : dy, adz = dz < 0 ? -dz : dz;
  const tDeltaX = stepX ? cell / adx : Infinity;
  const tDeltaY = stepY ? cell / ady : Infinity;
  const tDeltaZ = stepZ ? cell / adz : Infinity;
  // 走到下一个格边界的射线参数。注意 i/j/k 可能被上面 clamp 过，
  // 这时 (边界 - g) 会算成 0 或负 → tMax = t0 → 立刻走出去，不会卡死。
  let tMaxX = stepX ? t0 + ((stepX > 0 ? i + 1 - gx : gx - i) * cell) / adx : Infinity;
  let tMaxY = stepY ? t0 + ((stepY > 0 ? j + 1 - gy : gy - j) * cell) / ady : Infinity;
  let tMaxZ = stepZ ? t0 + ((stepZ > 0 ? k + 1 - gz : gz - k) * cell) / adz : Infinity;

  const grid = vol.grid;
  // 一个 32³ 的格子最多被穿过 3·D 次，这个上界足够且能防退化数据把循环拖住
  for (let guard = 0; guard < 3 * D + 8; guard++) {
    if (grid[(i * D + j) * D + k]) return true;

    if (tMaxX <= tMaxY) {
      if (tMaxX <= tMaxZ) {
        if (tMaxX > t1) return false;
        i += stepX; if (i < 0 || i >= D) return false; tMaxX += tDeltaX;
      } else {
        if (tMaxZ > t1) return false;
        k += stepZ; if (k < 0 || k >= D) return false; tMaxZ += tDeltaZ;
      }
    } else if (tMaxY <= tMaxZ) {
      if (tMaxY > t1) return false;
      j += stepY; if (j < 0 || j >= D) return false; tMaxY += tDeltaY;
    } else {
      if (tMaxZ > t1) return false;
      k += stepZ; if (k < 0 || k >= D) return false; tMaxZ += tDeltaZ;
    }
  }
  return false;
}

// ── 射线 vs 玩家包围盒（可选遮挡物）─────────────────────────────────────────

/**
 * 射线 vs 一个世界轴对齐的玩家盒，命中返回进入距离，否则 -1。
 * 和 visuals.js 的 `rayHull()` 同一套 slab 测试（那边是 three 空间、这里 game 空间）。
 */
function rayHull(state, ox, oy, oz, dx, dy, dz, maxDist, inset) {
  const hw = HULL_HALF_HU - inset;
  const h = hullHeightFor(state.crouching);
  const lo0 = state.x - hw, hi0 = state.x + hw;
  const lo1 = state.y - hw, hi1 = state.y + hw;
  const lo2 = state.z + inset, hi2 = state.z + h - inset;
  let tmin = 0, tmax = maxDist;
  let a, b;
  for (let axis = 0; axis < 3; axis++) {
    const o = axis === 0 ? ox : axis === 1 ? oy : oz;
    const d = axis === 0 ? dx : axis === 1 ? dy : dz;
    const lo = axis === 0 ? lo0 : axis === 1 ? lo1 : lo2;
    const hi = axis === 0 ? hi0 : axis === 1 ? hi1 : hi2;
    if (d > -1e-12 && d < 1e-12) {
      if (o < lo || o > hi) return -1;
      continue;
    }
    const inv = 1 / d;
    a = (lo - o) * inv; b = (hi - o) * inv;
    if (a > b) { const t = a; a = b; b = t; }
    if (a > tmin) tmin = a;
    if (b < tmax) tmax = b;
    if (tmin > tmax) return -1;
  }
  return tmin;
}

// ── 把 BVH 包成 game 空间的 castMapRay ──────────────────────────────────────

/**
 * 把 three 空间的 `raycastBVH` 包成 game 空间的射线函数，喂给
 * `createVisibilityEvaluator({ castMapRay })`。
 *
 * game → three 是 `(gx,gy,gz) → (gy,gz,gx)·SCALE`（见 scene.js::gameToThree）。
 * 方向向量**只做同样的轴置换、不乘 SCALE** —— 这个置换是一个旋转
 * （循环移位），长度不变，所以归一化的方向置换后仍然归一化；
 * 距离则要乘/除 SCALE。
 *
 * @param {object} bvh   map-raycast.js 建出来的 BVH
 * @param {number} scale scene.js 的 SCALE（0.0254）
 * @returns {(ox,oy,oz,dx,dy,dz,maxDist) => number} 命中距离（game 单位），没命中 -1
 */
export function makeMapCaster(bvh, scale) {
  if (!bvh) return null;
  return (ox, oy, oz, dx, dy, dz, maxDist) => {
    const t = raycastBVH(
      bvh,
      oy * scale, oz * scale, ox * scale,   // 起点：置换 + 缩放
      dy, dz, dx,                            // 方向：只置换
      maxDist * scale,
    );
    return t < 0 ? -1 : t / scale;
  };
}

// ── 单个 (观察者, 目标) 对：带每个角落的明细 ────────────────────────────────

/**
 * 算一对，并把 8 个角落的明细都带回来。给调试层和测试用（会分配对象）。
 * 热路径用 `createVisibilityEvaluator()`，那个不分配。
 *
 * @returns {{
 *   visible: boolean, clear: number, reason: number,
 *   corners: Array<{x:number,y:number,z:number,clear:boolean,reason:number,distance:number}>
 * }}
 */
export function pairVisibility(viewer, target, opts = {}) {
  const castMapRay = opts.castMapRay || null;
  const occluders = opts.occluders || [];
  const inset = opts.inset ?? HULL_INSET_HU;
  const aspect = opts.aspect ?? DEFAULT_ASPECT;
  const fovDeg = opts.fovDeg ?? DEFAULT_FOV_DEG;
  const vFovDeg = opts.vFovDeg ?? null;
  const others = opts.others || null;      // blockPlayers 时传全部玩家状态
  const blockPlayers = !!opts.blockPlayers;
  const viewerIdx = opts.viewerIdx ?? -1;
  const targetIdx = opts.targetIdx ?? -1;

  const tanH = Math.tan(fovDeg * DEG2RAD * 0.5);
  const tanV = vFovDeg != null ? Math.tan(vFovDeg * DEG2RAD * 0.5) : tanH / aspect;

  const ex = viewer.x, ey = viewer.y;
  const ez = eyeZ(viewer);

  const basis = new Float64Array(9);
  viewBasis(viewer.yaw || 0, viewer.pitch || 0, basis);

  const corners = new Float64Array(CORNER_COUNT * 3);
  playerCorners(target.x, target.y, target.z, hullHeightFor(target.crouching), corners, inset);

  const detail = [];
  let clearCount = 0;
  let why = 0;

  for (let c = 0; c < CORNER_COUNT; c++) {
    const cx = corners[c * 3], cy = corners[c * 3 + 1], cz = corners[c * 3 + 2];
    let dx = cx - ex, dy = cy - ey, dz = cz - ez;
    const dist = Math.hypot(dx, dy, dz);
    if (dist < 1e-6) {
      clearCount++;
      detail.push({ x: cx, y: cy, z: cz, clear: true, reason: 0, distance: 0 });
      continue;
    }
    dx /= dist; dy /= dist; dz /= dist;

    let r = 0;
    if (!dirInFov(basis, dx, dy, dz, tanH, tanV)) r |= BLOCK_FOV;
    else if (castMapRay && castMapRay(ex, ey, ez, dx, dy, dz, dist) >= 0) r |= BLOCK_MAP;
    else if (blockPlayers && others && blockedByPlayer(
      others, viewerIdx, targetIdx, ex, ey, ez, dx, dy, dz, dist, inset)) r |= BLOCK_PLAYER;
    else if (occluders.length && smokeRayBlocked(occluders, ex, ey, ez, dx, dy, dz, dist)) r |= BLOCK_SMOKE;

    if (r === 0) clearCount++; else why |= r;
    detail.push({ x: cx, y: cy, z: cz, clear: r === 0, reason: r, distance: dist });
  }

  return {
    visible: clearCount > 0,
    clear: clearCount,
    reason: clearCount > 0 ? 0 : why,
    corners: detail,
  };
}

function blockedByPlayer(others, viewerIdx, targetIdx, ox, oy, oz, dx, dy, dz, maxDist, inset) {
  for (let i = 0; i < others.length; i++) {
    if (i === viewerIdx || i === targetIdx) continue;
    const o = others[i];
    if (!o || !o.alive) continue;
    if (rayHull(o, ox, oy, oz, dx, dy, dz, maxDist, inset) >= 0) return true;
  }
  return false;
}

// ── 热路径：预分配求值器 ────────────────────────────────────────────────────

/**
 * 建一个可复用的求值器（内部全是预分配 typed array，evaluate() 不分配）。
 *
 * ```js
 * const ev = createVisibilityEvaluator({
 *   castMapRay: makeMapCaster(getMapBVH(), SCALE),
 * });
 * ev.setOccluders(smokeOccludersAt(currentRound.smoke, tick));
 * const m = ev.evaluate(playerStates);
 * m.visible[b * m.n + a]   // b 看得见 a 吗
 * ```
 *
 * @param {object} [opts]
 * @param {(ox,oy,oz,dx,dy,dz,maxDist)=>number} [opts.castMapRay] 地图遮挡；不传=只看视锥和烟
 * @param {number} [opts.fovDeg=90]      水平 FOV
 * @param {number} [opts.vFovDeg]        垂直 FOV；不传就用 fovDeg/aspect 推
 * @param {number} [opts.aspect=16/9]
 * @param {number} [opts.inset=1]        角落内收（HU）
 * @param {boolean} [opts.exactCount]    true = 8 个角落全测，clear 是精确个数；
 *                                       false（默认）= 第一个通的角落就收工，
 *                                       此时 clear 只能是 0 或 1。判定结果两者相同。
 * @param {boolean} [opts.blockPlayers]  其他玩家的碰撞盒算不算遮挡（默认 false）
 */
export function createVisibilityEvaluator(opts = {}) {
  const castMapRay = typeof opts.castMapRay === 'function' ? opts.castMapRay : null;
  const fovDeg = opts.fovDeg ?? DEFAULT_FOV_DEG;
  const vFovDeg = opts.vFovDeg ?? null;
  const aspect = opts.aspect ?? DEFAULT_ASPECT;
  const inset = opts.inset ?? HULL_INSET_HU;
  const exactCount = !!opts.exactCount;
  const blockPlayers = !!opts.blockPlayers;

  const tanH = Math.tan(fovDeg * DEG2RAD * 0.5);
  const tanV = vFovDeg != null ? Math.tan(vFovDeg * DEG2RAD * 0.5) : tanH / aspect;

  const basis = new Float64Array(9);
  const corners = new Float64Array(CORNER_COUNT * 3);

  let occluders = [];
  let cap = 0;
  let visible = new Uint8Array(0);
  let clear = new Uint8Array(0);
  let reason = new Uint8Array(0);

  function ensure(n) {
    if (cap >= n) return;
    cap = n;
    visible = new Uint8Array(n * n);
    clear = new Uint8Array(n * n);
    reason = new Uint8Array(n * n);
  }

  return {
    /** 当前 tick 的烟雾遮挡体（`{det, grid, dim, cell}` 数组）。每帧灌一次。 */
    setOccluders(list) { occluders = list || []; },
    get occluderCount() { return occluders.length; },
    /** 地图 BVH 建好了吗 —— 没建好时地图不算遮挡，结果要打折看。 */
    get mapReady() { return !!castMapRay; },
    fovDeg, vFovDeg, aspect, exactCount, blockPlayers,

    /**
     * 算整张矩阵。
     * @param {Array} states 全部玩家状态（`getAllPlayerStates()` 的返回值）
     * @returns {{n:number, visible:Uint8Array, clear:Uint8Array,
     *            reason:Uint8Array, stats:object}}
     *   下标统一是 `b * n + a`，语义是「**b 看得见 a**」——
     *   注意和 `.csz` 的 `spotted[a] ∋ b` 方向相反，别搞混。
     */
    evaluate(states) {
      const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
      const n = states ? states.length : 0;
      ensure(n);
      visible.fill(0);
      clear.fill(0);
      reason.fill(0);

      const stats = {
        n, pairs: 0, rays: 0, fovRejects: 0,
        mapHits: 0, smokeHits: 0, playerHits: 0, ms: 0,
      };
      const hasSmoke = occluders.length > 0;

      for (let b = 0; b < n; b++) {
        const v = states[b];
        if (!v || !v.alive) continue;
        const ex = v.x, ey = v.y;
        const ez = eyeZ(v);
        viewBasis(v.yaw || 0, v.pitch || 0, basis);

        for (let a = 0; a < n; a++) {
          if (a === b) continue;
          const t = states[a];
          if (!t || !t.alive) continue;
          const idx = b * n + a;
          stats.pairs++;

          playerCorners(t.x, t.y, t.z, hullHeightFor(t.crouching), corners, inset);

          let nClear = 0, why = 0;
          for (let c = 0; c < CORNER_COUNT; c++) {
            const cx = corners[c * 3], cy = corners[c * 3 + 1], cz = corners[c * 3 + 2];
            let dx = cx - ex, dy = cy - ey, dz = cz - ez;
            const dist = Math.hypot(dx, dy, dz);

            // 退化情形（角落正好落在眼睛上，只有两个人完全重叠才可能）当成"通"。
            // 注意它必须走和普通"通"完全相同的收尾路径 —— 早退分支要是漏了它，
            // exactCount=false 的 clear 就可能变成 2，破坏「只会是 0 或 1」这条不变量。
            let blocked = 0;
            if (dist >= 1e-6) {
              dx /= dist; dy /= dist; dz /= dist;
              if (!dirInFov(basis, dx, dy, dz, tanH, tanV)) {
                blocked = BLOCK_FOV; stats.fovRejects++;
              } else {
                stats.rays++;
                if (castMapRay && castMapRay(ex, ey, ez, dx, dy, dz, dist) >= 0) {
                  blocked = BLOCK_MAP; stats.mapHits++;
                } else if (blockPlayers && blockedByPlayer(
                  states, b, a, ex, ey, ez, dx, dy, dz, dist, inset)) {
                  blocked = BLOCK_PLAYER; stats.playerHits++;
                } else if (hasSmoke
                  && smokeRayBlocked(occluders, ex, ey, ez, dx, dy, dz, dist)) {
                  blocked = BLOCK_SMOKE; stats.smokeHits++;
                }
              }
            }

            if (blocked) { why |= blocked; continue; }
            nClear++;
            if (!exactCount) break;    // 已经确定可见，剩下 7 个角落不用测
          }

          clear[idx] = nClear;
          visible[idx] = nClear > 0 ? 1 : 0;
          reason[idx] = nClear > 0 ? 0 : why;
        }
      }

      stats.ms = (typeof performance !== 'undefined' ? performance : Date).now() - t0;
      return { n, visible, clear, reason, stats };
    },
  };
}

// ── 和 .csz 的 spotted 列对拍用的小工具 ─────────────────────────────────────

/**
 * 把矩阵摊成「谁看得见谁」的对数，方便和 `spotted` 的几何验证对照。
 * 只统计**双方都活着**的对（`.csz` 里死人会留脏值，见 README §4.8）。
 *
 * @returns {{total:number, mutual:number, oneWay:number}}
 */
export function pairStats(result, states) {
  const { n, visible } = result;
  let total = 0, mutual = 0, oneWay = 0;
  for (let b = 0; b < n; b++) {
    for (let a = 0; a < n; a++) {
      if (a === b) continue;
      if (!states[b] || !states[b].alive || !states[a] || !states[a].alive) continue;
      if (!visible[b * n + a]) continue;
      total++;
      if (visible[a * n + b]) mutual++; else oneWay++;
    }
  }
  return { total, mutual, oneWay };
}
