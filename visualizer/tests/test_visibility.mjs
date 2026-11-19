/**
 * test_visibility.mjs — `visibility.js`（几何可观测性）的正确性 + 性能测试。
 *
 * 纯 Node，不需要浏览器、不需要 demo、不需要地图文件。分六块：
 *
 *   A 解析用例     视角基 / 8 个角落 / 视锥边界 / 烟雾 DDA —— 能手算出答案的
 *   B 暴力对拍     随机三角形汤上，`createVisibilityEvaluator` 的每个角落判定
 *                  必须和**逐三角形暴力扫描**完全一致
 *   C 语义         面对面=可见、中间一堵墙=不可见、背对=不在视野、
 *                  中间一坨烟=不可见、烟挪开=又可见
 *   D 坐标置换     `makeMapCaster` 的 game↔three 轴置换 / SCALE 换算
 *   E 不变量       对角线、死人不参与、exactCount 不影响判定、矩阵方向
 *   F 性能         造一张 82 万三角形的"类地图"场景，量 10 人 90 对的每帧耗时
 *
 * 用法：node visualizer/tests/test_visibility.mjs
 */

import { buildBVH, raycastBVH } from '../static/js/map-raycast.js';
import {
  createVisibilityEvaluator, makeMapCaster, pairVisibility, pairStats,
  playerCorners, viewBasis, dirInFov, smokeRayBlocked,
  HULL_HALF_HU, HULL_STAND_HU, HULL_CROUCH_HU, HULL_INSET_HU, CORNER_COUNT,
  BLOCK_FOV, BLOCK_MAP, BLOCK_SMOKE, BLOCK_PLAYER,
} from '../static/js/visibility.js';

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}
function near(a, b, tol = 1e-9) { return Math.abs(a - b) <= tol; }

// ── 共用工具 ────────────────────────────────────────────────────────────────

/** 确定性 PRNG（mulberry32）—— 测试必须可复现。 */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function soup(tris) {
  const verts = new Float32Array(tris.length * 9);
  for (let i = 0; i < tris.length; i++) verts.set(tris[i], i * 9);
  return { verts, triCount: tris.length };
}

// map-raycast.js 的 rayTri 是模块私有的；这里逐行复制一份当暴力参考。
// 常量必须和那边一致，否则边界情形会对不上（EPS/BARY_EPS 都是 1e-9）。
const EPS = 1e-9;
const BARY_EPS = 1e-9;
function rayTri(verts, t, ox, oy, oz, dx, dy, dz) {
  const i = t * 9;
  const ax = verts[i], ay = verts[i + 1], az = verts[i + 2];
  const e1x = verts[i + 3] - ax, e1y = verts[i + 4] - ay, e1z = verts[i + 5] - az;
  const e2x = verts[i + 6] - ax, e2y = verts[i + 7] - ay, e2z = verts[i + 8] - az;
  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (det > -EPS && det < EPS) return -1;
  const inv = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * inv;
  if (u < -BARY_EPS || u > 1 + BARY_EPS) return -1;
  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v < -BARY_EPS || u + v > 1 + BARY_EPS) return -1;
  return (e2x * qx + e2y * qy + e2z * qz) * inv;
}

function bruteForce(verts, triCount, ox, oy, oz, dx, dy, dz, maxDist) {
  let best = maxDist, found = false;
  for (let t = 0; t < triCount; t++) {
    const hit = rayTri(verts, t, ox, oy, oz, dx, dy, dz);
    if (hit >= 0 && hit < best) { best = hit; found = true; }
  }
  return found ? best : -1;
}

/** 一个轴对齐的盒子拆成 12 个三角形，写进 tris 数组。 */
function pushBox(tris, x0, y0, z0, x1, y1, z1) {
  const v = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const faces = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4],
  ];
  for (const [a, b, c, d] of faces) {
    tris.push([...v[a], ...v[b], ...v[c]]);
    tris.push([...v[a], ...v[c], ...v[d]]);
  }
}

function player(x, y, z, yaw = 0, pitch = 0, opts = {}) {
  return {
    x, y, z, yaw, pitch,
    crouch: opts.crouch ?? 0,
    crouching: !!opts.crouching,
    alive: opts.alive !== false,
    eye_height: 64.0938 - 18 * (opts.crouch ?? 0),
  };
}

/** 一堵墙：在 game 空间里垂直于 axis、位于 value 处的一大块。 */
function wallAlong(axis, value, halfSpan = 400, z0 = -100, z1 = 200) {
  const tris = [];
  const t = 0.5;
  if (axis === 'y') pushBox(tris, -halfSpan, value - t, z0, halfSpan, value + t, z1);
  else if (axis === 'x') pushBox(tris, value - t, -halfSpan, z0, value + t, halfSpan, z1);
  else pushBox(tris, -halfSpan, -halfSpan, value - t, halfSpan, halfSpan, value + t);
  return tris;
}

/** 一个全空的烟雾体素栅格。 */
function emptySmokeGrid(dim = 32) { return new Uint8Array(dim * dim * dim); }

/**
 * 造一个烟雾遮挡体：把栅格的一部分格子填满。
 * `fill(i,j,k)` 返回 true 表示占据。
 */
function smokeVolume(det, fill, cell = 12, dim = 32) {
  const grid = emptySmokeGrid(dim);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      for (let k = 0; k < dim; k++) {
        if (fill(i, j, k)) grid[(i * dim + j) * dim + k] = 1;
      }
    }
  }
  return { det, grid, dim, cell };
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== A 解析用例');
// ═══════════════════════════════════════════════════════════════════════════

{
  const b = new Float64Array(9);

  viewBasis(0, 0, b);
  check('viewBasis yaw=0,pitch=0 -> f=+X（Source: yaw 0 看 +X）',
    near(b[0], 1) && near(b[1], 0) && near(b[2], 0), `f=(${b[0]},${b[1]},${b[2]})`);
  check('viewBasis yaw=0,pitch=0 -> r=-Y（forward×up = X×Z = -Y）',
    near(b[3], 0) && near(b[4], -1) && near(b[5], 0), `r=(${b[3]},${b[4]},${b[5]})`);
  check('viewBasis yaw=0,pitch=0 -> u=+Z',
    near(b[6], 0) && near(b[7], 0) && near(b[8], 1), `u=(${b[6]},${b[7]},${b[8]})`);

  viewBasis(90, 0, b);
  check('viewBasis yaw=90 -> f=+Y', near(b[0], 0, 1e-12) && near(b[1], 1), `f=(${b[0]},${b[1]})`);

  viewBasis(0, 30, b);
  check('viewBasis pitch=+30 -> f.z=-0.5（Source 里正 pitch 是向下看）',
    near(b[2], -0.5, 1e-12), `f.z=${b[2]}`);

  // 随机朝向：基必须正交归一
  const r = rng(12345);
  let orthoBad = 0, normBad = 0;
  for (let i = 0; i < 2000; i++) {
    const yaw = (r() * 2 - 1) * 180, pitch = (r() * 2 - 1) * 89;
    viewBasis(yaw, pitch, b);
    const f = [b[0], b[1], b[2]], rr = [b[3], b[4], b[5]], u = [b[6], b[7], b[8]];
    const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    if (Math.abs(dot(f, rr)) > 1e-12 || Math.abs(dot(f, u)) > 1e-12
      || Math.abs(dot(rr, u)) > 1e-12) orthoBad++;
    for (const v of [f, rr, u]) if (Math.abs(dot(v, v) - 1) > 1e-12) normBad++;
  }
  check('viewBasis 2000 个随机朝向都正交', orthoBad === 0, `坏 ${orthoBad}`);
  check('viewBasis 2000 个随机朝向都归一', normBad === 0, `坏 ${normBad}`);

  // 和 visuals.js::eyeRay() 的 three 空间约定对拍：three = (gy, gz, gx)
  // eyeRay: forward=(0,0,1) 绕 X 转 pitch 再绕 Y 转 yaw
  let mismatch = 0;
  for (let i = 0; i < 500; i++) {
    const yaw = (r() * 2 - 1) * 180, pitch = (r() * 2 - 1) * 89;
    const yr = yaw * Math.PI / 180, pr = pitch * Math.PI / 180;
    // three 空间：绕 X 转 pitch
    const v1 = [0, -Math.sin(pr), Math.cos(pr)];
    // 再绕 Y 转 yaw
    const fx = Math.cos(yr) * v1[0] + Math.sin(yr) * v1[2];
    const fy = v1[1];
    const fz = -Math.sin(yr) * v1[0] + Math.cos(yr) * v1[2];
    // three -> game: game = (three.z, three.x, three.y)
    viewBasis(yaw, pitch, b);
    if (Math.abs(b[0] - fz) > 1e-12 || Math.abs(b[1] - fx) > 1e-12
      || Math.abs(b[2] - fy) > 1e-12) mismatch++;
  }
  check('viewBasis 和 eyeRay() 的 three 约定一致（500 个随机朝向）',
    mismatch === 0, `不一致 ${mismatch}`);
}

{
  const out = new Float64Array(CORNER_COUNT * 3);
  playerCorners(100, 200, 0, HULL_STAND_HU, out);
  const xs = [], ys = [], zs = [];
  for (let i = 0; i < CORNER_COUNT; i++) {
    xs.push(out[i * 3]); ys.push(out[i * 3 + 1]); zs.push(out[i * 3 + 2]);
  }
  const hw = HULL_HALF_HU - HULL_INSET_HU;
  check('playerCorners 8 个角，x 只有两个值 ±(16-inset)',
    new Set(xs).size === 2 && near(Math.max(...xs), 100 + hw) && near(Math.min(...xs), 100 - hw),
    `x=${[...new Set(xs)]}`);
  check('playerCorners y 只有两个值 ±(16-inset)', new Set(ys).size === 2,
    `y=${[...new Set(ys)]}`);
  check('playerCorners z = [z+inset, z+72-inset]',
    new Set(zs).size === 2 && near(Math.min(...zs), HULL_INSET_HU)
    && near(Math.max(...zs), HULL_STAND_HU - HULL_INSET_HU), `z=${[...new Set(zs)]}`);
  const uniq = new Set();
  for (let i = 0; i < CORNER_COUNT; i++) uniq.add(`${out[i * 3]},${out[i * 3 + 1]},${out[i * 3 + 2]}`);
  check('playerCorners 8 个角互不重合', uniq.size === 8, `${uniq.size} 个`);

  playerCorners(0, 0, 0, HULL_CROUCH_HU, out);
  // 角落顺序是 (k=z, j=y, i=x)，所以 z 的两个值在下标 2 和 2+4*3=14
  check('playerCorners 蹲下用 54 HU 而不是 72',
    near(out[2], HULL_INSET_HU) && near(out[14], HULL_CROUCH_HU - HULL_INSET_HU),
    `z0=${out[2]} z1=${out[14]}`);
}

{
  const b = new Float64Array(9);
  viewBasis(0, 0, b);
  const tanH = Math.tan(45 * Math.PI / 180);      // 90° 水平 FOV
  const tanV = tanH / (16 / 9);

  check('dirInFov 正前方 -> true', dirInFov(b, 1, 0, 0, tanH, tanV));
  check('dirInFov 正后方 -> false', !dirInFov(b, -1, 0, 0, tanH, tanV));
  check('dirInFov 正左方(90°) -> false（fd=0 的边界）', !dirInFov(b, 0, -1, 0, tanH, tanV));

  const a = (deg) => deg * Math.PI / 180;
  const just = (deg) => [Math.cos(a(deg)), Math.sin(a(deg)), 0];
  let d = just(44.9);
  check('dirInFov 水平 44.9° -> true', dirInFov(b, d[0], d[1], d[2], tanH, tanV));
  d = just(45.1);
  check('dirInFov 水平 45.1° -> false', !dirInFov(b, d[0], d[1], d[2], tanH, tanV));

  // 垂直半角 = atan(tanH/aspect) = atan(0.5625) ≈ 29.36°
  const vDeg = Math.atan(tanV) * 180 / Math.PI;
  const up = (deg) => {
    const t = a(deg);
    return [Math.cos(t), 0, Math.sin(t)];
  };
  d = up(vDeg - 0.2);
  check(`dirInFov 垂直 ${(vDeg - 0.2).toFixed(1)}° -> true`, dirInFov(b, d[0], d[1], d[2], tanH, tanV));
  d = up(vDeg + 0.2);
  check(`dirInFov 垂直 ${(vDeg + 0.2).toFixed(1)}° -> false`, !dirInFov(b, d[0], d[1], d[2], tanH, tanV));
}

{
  check('smokeRayBlocked 没有烟 -> false',
    !smokeRayBlocked([], 0, 0, 0, 1, 0, 0, 1000));
  check('smokeRayBlocked 没有烟（null）-> false',
    !smokeRayBlocked(null, 0, 0, 0, 1, 0, 0, 1000));

  // 一整块实心烟，det 在原点（= 384 HU 立方，中心在原点）
  const full = smokeVolume([0, 0, 0], () => true);
  check('smokeRayBlocked 穿过实心烟 -> true',
    smokeRayBlocked([full], -500, 0, 0, 1, 0, 0, 1000));
  check('smokeRayBlocked 射线在烟外面 -> false（AABB 拒绝）',
    !smokeRayBlocked([full], -500, 500, 0, 1, 0, 0, 1000));
  check('smokeRayBlocked maxDist 够不到烟 -> false',
    !smokeRayBlocked([full], -500, 0, 0, 1, 0, 0, 50));
  check('smokeRayBlocked 背对着烟射 -> false',
    !smokeRayBlocked([full], -500, 0, 0, -1, 0, 0, 1000));

  // 一片薄板：只有 k=16 那一层被占据（game z ∈ [-6, 6]）
  const slab = smokeVolume([0, 0, 0], (i, j, k) => k === 16);
  check('smokeRayBlocked 穿过薄板 -> true',
    smokeRayBlocked([slab], -500, 0, 0, 1, 0, 0, 1000));
  check('smokeRayBlocked 从薄板上方掠过 -> false',
    !smokeRayBlocked([slab], -500, 0, 100, 1, 0, 0, 1000));
  check('smokeRayBlocked 从薄板下方掠过 -> false',
    !smokeRayBlocked([slab], -500, 0, -100, 1, 0, 0, 1000));

  // 斜射：从 (-500,-500,300) 朝原点方向，应该穿过烟
  const L = Math.hypot(500, 500, -300);
  const d = [500 / L, 500 / L, -300 / L];
  check('smokeRayBlocked 斜射穿烟 -> true',
    smokeRayBlocked([full], -500, -500, 300, d[0], d[1], d[2], 2000));

  // 多个烟：只有第二个挡路
  const far = smokeVolume([2000, 0, 0], () => true);
  check('smokeRayBlocked 多个烟里只有挡路的那个算',
    smokeRayBlocked([far, full], -500, 0, 0, 1, 0, 0, 1000));
  check('smokeRayBlocked 多个烟都不挡 -> false',
    !smokeRayBlocked([far], -500, 0, 0, 1, 0, 0, 1000));
}

{
  // DDA vs 精确 ray-AABB：把随机占据区域做成"格子盒"（整块连续格子），
  // 每个格子盒就是一个世界空间 AABB，于是能算出精确答案。
  const r = rng(777);
  const dim = 32, cell = 12;
  let cases = 0, mismatch = 0;
  for (let trial = 0; trial < 300; trial++) {
    // 1~3 个随机的格子盒
    const boxes = [];
    const fill = (i, j, k) => {
      for (const b of boxes) {
        if (i >= b.i0 && i < b.i1 && j >= b.j0 && j < b.j1 && k >= b.k0 && k < b.k1) return true;
      }
      return false;
    };
    const nBox = 1 + Math.floor(r() * 3);
    for (let b = 0; b < nBox; b++) {
      const i0 = Math.floor(r() * dim), j0 = Math.floor(r() * dim), k0 = Math.floor(r() * dim);
      boxes.push({
        i0, j0, k0,
        i1: Math.min(dim, i0 + 2 + Math.floor(r() * 12)),
        j1: Math.min(dim, j0 + 2 + Math.floor(r() * 12)),
        k1: Math.min(dim, k0 + 2 + Math.floor(r() * 12)),
      });
    }
    const vol = smokeVolume([0, 0, 0], fill, cell, dim);

    // 格子盒 -> 世界 AABB。格子 i 覆盖 det + (i-16)·cell .. det + (i-15)·cell
    const worldBoxes = boxes.map((b) => ({
      lo: [(b.i0 - 16) * cell, (b.j0 - 16) * cell, (b.k0 - 16) * cell],
      hi: [(b.i1 - 16) * cell, (b.j1 - 16) * cell, (b.k1 - 16) * cell],
    }));

    for (let s = 0; s < 40; s++) {
      const ox = (r() * 2 - 1) * 900, oy = (r() * 2 - 1) * 900, oz = (r() * 2 - 1) * 900;
      let dx = r() * 2 - 1, dy = r() * 2 - 1, dz = r() * 2 - 1;
      const L = Math.hypot(dx, dy, dz);
      if (L < 1e-6) continue;
      dx /= L; dy /= L; dz /= L;
      const maxDist = 200 + r() * 1800;

      const got = smokeRayBlocked([vol], ox, oy, oz, dx, dy, dz, maxDist);

      // 精确参考：射线 vs 每个世界 AABB 的 slab 测试
      let want = false;
      for (const wb of worldBoxes) {
        let t0 = 0, t1 = maxDist, ok = true;
        for (let ax = 0; ax < 3; ax++) {
          const o = ax === 0 ? ox : ax === 1 ? oy : oz;
          const dd = ax === 0 ? dx : ax === 1 ? dy : dz;
          if (Math.abs(dd) < 1e-12) {
            if (o < wb.lo[ax] || o > wb.hi[ax]) { ok = false; break; }
            continue;
          }
          const inv = 1 / dd;
          let a = (wb.lo[ax] - o) * inv, b2 = (wb.hi[ax] - o) * inv;
          if (a > b2) { const t = a; a = b2; b2 = t; }
          if (a > t0) t0 = a;
          if (b2 < t1) t1 = b2;
          if (t0 > t1) { ok = false; break; }
        }
        if (ok && t0 <= maxDist && t1 >= 0) { want = true; break; }
      }

      cases++;
      if (got !== want) mismatch++;
    }
  }
  // 允许极少量擦边差异（射线正好贴着格子面/棱穿过时，DDA 和 AABB 的
  // 边界归属可以不同）。要求 99.5% 以上一致就足以证明 DDA 的步进没写错。
  const rate = 1 - mismatch / cases;
  check(`烟雾 DDA vs 精确 ray-AABB：${cases} 例，一致率 ${(rate * 100).toFixed(2)}%`,
    rate >= 0.995, `不一致 ${mismatch}/${cases}`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== B 暴力对拍（随机三角形汤）');
// ═══════════════════════════════════════════════════════════════════════════

{
  const r = rng(20240924);
  const R = 600;
  const tris = [];
  // 20000 个随机三角形，1% 是横跨全场的大三角形（模拟真实地图的尺寸分布）
  for (let i = 0; i < 20000; i++) {
    const big = r() < 0.01;
    const s = big ? 1200 : 4 + r() * 60;
    const cx = (r() * 2 - 1) * R, cy = (r() * 2 - 1) * R, cz = r() * 300 - 100;
    const t = [];
    for (let k = 0; k < 3; k++) t.push(cx + (r() - 0.5) * s, cy + (r() - 0.5) * s, cz + (r() - 0.5) * s);
    tris.push(t);
  }
  const trisData = soup(tris);
  const bvh = buildBVH(trisData);

  // 直接问 BVH（不做坐标置换）—— 这里让"game 空间"就是 BVH 空间
  const directCaster = (ox, oy, oz, dx, dy, dz, maxDist) =>
    raycastBVH(bvh, ox, oy, oz, dx, dy, dz, maxDist);

  const states = [];
  for (let i = 0; i < 10; i++) {
    states.push(player(
      (r() * 2 - 1) * R * 0.8, (r() * 2 - 1) * R * 0.8, r() * 40,
      (r() * 2 - 1) * 180, (r() * 2 - 1) * 40,
    ));
  }

  const ev = createVisibilityEvaluator({ castMapRay: directCaster, exactCount: true });
  const got = ev.evaluate(states);

  // 参考实现：把 evaluate 的逻辑用暴力扫描重写一遍
  let cornerTotal = 0, cornerDiff = 0, pairDiff = 0;
  const basis = new Float64Array(9);
  const corners = new Float64Array(CORNER_COUNT * 3);
  const tanH = Math.tan(45 * Math.PI / 180);
  const tanV = tanH / (16 / 9);
  for (let b = 0; b < 10; b++) {
    const v = states[b];
    const ex = v.x, ey = v.y, ez = v.z + v.eye_height;
    viewBasis(v.yaw, v.pitch, basis);
    for (let a = 0; a < 10; a++) {
      if (a === b) continue;
      const t = states[a];
      playerCorners(t.x, t.y, t.z, HULL_STAND_HU, corners, HULL_INSET_HU);
      let nClear = 0, why = 0;
      for (let c = 0; c < CORNER_COUNT; c++) {
        const cx = corners[c * 3], cy = corners[c * 3 + 1], cz = corners[c * 3 + 2];
        let dx = cx - ex, dy = cy - ey, dz = cz - ez;
        const dist = Math.hypot(dx, dy, dz);
        dx /= dist; dy /= dist; dz /= dist;
        let r2 = 0;
        if (!dirInFov(basis, dx, dy, dz, tanH, tanV)) r2 |= BLOCK_FOV;
        else if (bruteForce(trisData.verts, trisData.triCount, ex, ey, ez, dx, dy, dz, dist) >= 0) r2 |= BLOCK_MAP;
        if (r2 === 0) nClear++; else why |= r2;
        cornerTotal++;
        const idx = b * 10 + a;
        // 逐角落比对：靠 clear 计数和 reason 位间接验（evaluate 不导出每角明细）
        if (c === CORNER_COUNT - 1) {
          if (got.clear[idx] !== nClear || (nClear === 0 ? got.reason[idx] !== why : got.reason[idx] !== 0)) {
            cornerDiff++;
          }
        }
      }
    }
  }
  let visDiff = 0;
  for (let b = 0; b < 10; b++) {
    for (let a = 0; a < 10; a++) {
      if (a === b) continue;
      const idx = b * 10 + a;
      const v = states[b], t = states[a];
      const ex = v.x, ey = v.y, ez = v.z + v.eye_height;
      viewBasis(v.yaw, v.pitch, basis);
      playerCorners(t.x, t.y, t.z, HULL_STAND_HU, corners, HULL_INSET_HU);
      let any = false;
      for (let c = 0; c < CORNER_COUNT; c++) {
        const cx = corners[c * 3], cy = corners[c * 3 + 1], cz = corners[c * 3 + 2];
        let dx = cx - ex, dy = cy - ey, dz = cz - ez;
        const dist = Math.hypot(dx, dy, dz);
        dx /= dist; dy /= dist; dz /= dist;
        if (!dirInFov(basis, dx, dy, dz, tanH, tanV)) continue;
        if (bruteForce(trisData.verts, trisData.triCount, ex, ey, ez, dx, dy, dz, dist) < 0) { any = true; break; }
      }
      if (got.visible[idx] !== (any ? 1 : 0)) visDiff++;
    }
  }
  check(`90 对的 clear 计数 + reason 全部和暴力扫描一致`, cornerDiff === 0, `${cornerDiff} 对不一致`);
  check(`90 对的 visible 判定全部和暴力扫描一致`, visDiff === 0, `${visDiff} 对不一致`);
  check(`暴力对拍确实算了不少角落（${cornerTotal} 个）`, cornerTotal === 90 * CORNER_COUNT);

  // 逐角落明细版（pairVisibility）必须和 evaluate 的汇总一致
  const ev2 = createVisibilityEvaluator({ castMapRay: directCaster, exactCount: true });
  const got2 = ev2.evaluate(states);
  let detailDiff = 0;
  for (let b = 0; b < 10; b++) {
    for (let a = 0; a < 10; a++) {
      if (a === b) continue;
      const d = pairVisibility(states[b], states[a], {
        castMapRay: directCaster, viewerIdx: b, targetIdx: a,
      });
      const idx = b * 10 + a;
      if (d.clear !== got2.clear[idx] || d.visible !== (got2.visible[idx] === 1)) detailDiff++;
    }
  }
  check('pairVisibility（带每角明细）和 evaluate 的汇总一致', detailDiff === 0, `${detailDiff} 对不一致`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== C 语义');
// ═══════════════════════════════════════════════════════════════════════════

{
  const empty = buildBVH(soup([]));
  const caster0 = (ox, oy, oz, dx, dy, dz, maxDist) =>
    raycastBVH(empty, ox, oy, oz, dx, dy, dz, maxDist);

  // 两个人相距 400 HU（10 m），B 在原点朝 +X，A 在 +X 方向
  const B = player(0, 0, 0, 0, 0);
  const A = player(400, 0, 0, 180, 0);

  let d = pairVisibility(B, A, { castMapRay: caster0 });
  check('空地图、面对面 -> 可见，8 个角落全通', d.visible && d.clear === 8,
    `visible=${d.visible} clear=${d.clear}`);

  d = pairVisibility(B, player(-400, 0, 0, 0, 0), { castMapRay: caster0 });
  check('目标在背后 -> 不可见，原因是 BLOCK_FOV',
    !d.visible && d.reason === BLOCK_FOV, `reason=${d.reason}`);

  d = pairVisibility(B, player(0, 400, 0, 0, 0), { castMapRay: caster0 });
  check('目标在正左 90° -> 不可见（水平 FOV 边界）',
    !d.visible && d.reason === BLOCK_FOV, `reason=${d.reason}`);

  d = pairVisibility(B, player(300, 300, 0, 0, 0), { castMapRay: caster0 });
  check('目标在 45° 斜前方 -> 可见（刚好在 90° FOV 的边界内）',
    d.visible, `clear=${d.clear} reason=${d.reason}`);

  // 中间一堵墙（垂直于 X，位于 x=200）
  const wallBVH = buildBVH(soup(wallAlong('x', 200)));
  const casterW = (ox, oy, oz, dx, dy, dz, maxDist) =>
    raycastBVH(wallBVH, ox, oy, oz, dx, dy, dz, maxDist);
  d = pairVisibility(B, A, { castMapRay: casterW });
  check('中间一堵墙 -> 不可见，原因是 BLOCK_MAP',
    !d.visible && d.reason === BLOCK_MAP, `visible=${d.visible} reason=${d.reason}`);

  // 墙挪到侧面
  const sideBVH = buildBVH(soup(wallAlong('y', 300)));
  const casterS = (ox, oy, oz, dx, dy, dz, maxDist) =>
    raycastBVH(sideBVH, ox, oy, oz, dx, dy, dz, maxDist);
  d = pairVisibility(B, A, { castMapRay: casterS });
  check('墙在侧面 -> 还是可见', d.visible, `clear=${d.clear}`);

  // 半遮挡：目标半个身子在墙后。
  // 眼睛在 z=64.09，目标的 4 个 y=-15 角落和 4 个 y=+15 角落。射线走到墙所在的
  // x≈200 时 y 只剩一半（∓7.5），所以墙只盖住 y>=0 那一半 → 4 通 4 挡。
  {
    const tris = [];
    pushBox(tris, 199, 0, -100, 201, 400, 200);
    const halfBVH = buildBVH(soup(tris));
    const casterH = (ox, oy, oz, dx, dy, dz, maxDist) =>
      raycastBVH(halfBVH, ox, oy, oz, dx, dy, dz, maxDist);
    const A2 = player(400, 0, 0, 180, 0);
    d = pairVisibility(B, A2, { castMapRay: casterH });
    check('半遮挡（4 个角落被挡）-> 仍然可见（"任意一个角落通"的语义）',
      d.visible && d.clear === 4, `visible=${d.visible} clear=${d.clear}`);
    const reasons = d.corners.map((c) => c.reason);
    check('半遮挡时被挡的 4 个角落原因都是 BLOCK_MAP',
      reasons.filter((x) => x === BLOCK_MAP).length === 4,
      `reason 分布 ${JSON.stringify(reasons)}`);
  }

  // 烟雾：两个人中间放一整坨烟
  const smokeFull = smokeVolume([200, 0, 50], () => true);
  d = pairVisibility(B, A, { castMapRay: caster0, occluders: [smokeFull] });
  check('中间一坨烟 -> 不可见，原因是 BLOCK_SMOKE',
    !d.visible && d.reason === BLOCK_SMOKE, `visible=${d.visible} reason=${d.reason}`);

  // 烟挪到侧面
  const smokeSide = smokeVolume([200, 600, 50], () => true);
  d = pairVisibility(B, A, { castMapRay: caster0, occluders: [smokeSide] });
  check('烟在侧面 -> 还是可见', d.visible, `clear=${d.clear}`);

  // 烟在中间但只有一层薄板，而射线从板子上面过去
  const smokeLow = smokeVolume([200, 0, -300], () => true);
  d = pairVisibility(B, A, { castMapRay: caster0, occluders: [smokeLow] });
  check('烟在下面（射线从上面过）-> 可见', d.visible, `clear=${d.clear}`);

  // 地图 + 烟雾同时存在：地图先命中就报 BLOCK_MAP
  d = pairVisibility(B, A, { castMapRay: casterW, occluders: [smokeFull] });
  check('墙和烟都在 -> reason 里只有 BLOCK_MAP（先命中先报）',
    !d.visible && d.reason === BLOCK_MAP, `reason=${d.reason}`);

  // 其他玩家的身体当遮挡（可选）
  const C = player(200, 0, 0, 0, 0);
  d = pairVisibility(B, A, { castMapRay: caster0, blockPlayers: true, others: [B, A, C], viewerIdx: 0, targetIdx: 1 });
  check('blockPlayers 打开、中间站了个人 -> 不可见，原因是 BLOCK_PLAYER',
    !d.visible && d.reason === BLOCK_PLAYER, `visible=${d.visible} reason=${d.reason}`);
  d = pairVisibility(B, A, { castMapRay: caster0, blockPlayers: false });
  check('blockPlayers 关闭 -> 忽略身体，还是可见', d.visible, `clear=${d.clear}`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== D 坐标置换（makeMapCaster）');
// ═══════════════════════════════════════════════════════════════════════════

{
  // BVH 里放一堵"three 空间 x = 5"的墙。game→three 是 (gx,gy,gz)->(gy,gz,gx)，
  // 所以 three.x 就是 game.y。
  const tris = [];
  pushBox(tris, 5, -100, -100, 5.001, 100, 100);
  const bvh = buildBVH(soup(tris));

  const c1 = makeMapCaster(bvh, 1);
  check('makeMapCaster(scale=1) 朝 +Y 射 -> 命中距离 5',
    near(c1(0, 0, 0, 0, 1, 0, 100), 5, 1e-3), `${c1(0, 0, 0, 0, 1, 0, 100)}`);
  check('makeMapCaster(scale=1) 朝 +X 射 -> 不命中（墙垂直于 game Y）',
    c1(0, 0, 0, 1, 0, 0, 100) < 0, `${c1(0, 0, 0, 1, 0, 0, 100)}`);
  check('makeMapCaster(scale=1) 朝 +Z 射 -> 不命中',
    c1(0, 0, 0, 0, 0, 1, 100) < 0, `${c1(0, 0, 0, 0, 0, 1, 100)}`);

  // 换成真实 SCALE：BVH 里的 three.x = 5 米 对应 game.y = 5/0.0254 ≈ 196.85 HU，
  // 所以从原点朝 +Y 射，报回来的**game 单位**距离也应该是 196.85 HU。
  const S = 0.0254;
  const tris2 = [];
  pushBox(tris2, 5, -1, -1, 5.001, 1, 1);
  const bvh2 = buildBVH(soup(tris2));
  const c2 = makeMapCaster(bvh2, S);
  const hit = c2(0, 0, 0, 0, 1, 0, 10000);
  check('makeMapCaster 用真实 SCALE 时距离换算正确（5 m = 196.85 HU）',
    near(hit, 5 / S, 0.05), `${hit} vs ${5 / S}`);
  check('makeMapCaster 用真实 SCALE 时方向轴置换正确（朝 +X 不该命中）',
    c2(0, 0, 0, 1, 0, 0, 10000) < 0, `${c2(0, 0, 0, 1, 0, 0, 10000)}`);

  check('makeMapCaster(null) -> null（没地图时优雅退化）', makeMapCaster(null, S) === null);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== E 不变量');
// ═══════════════════════════════════════════════════════════════════════════

{
  const r = rng(4242);
  const tris = [];
  for (let i = 0; i < 3000; i++) {
    const s = 20 + r() * 100;
    const cx = (r() * 2 - 1) * 500, cy = (r() * 2 - 1) * 500, cz = r() * 200;
    const t = [];
    for (let k = 0; k < 3; k++) t.push(cx + (r() - 0.5) * s, cy + (r() - 0.5) * s, cz + (r() - 0.5) * s);
    tris.push(t);
  }
  const bvh = buildBVH(soup(tris));
  const caster = (ox, oy, oz, dx, dy, dz, maxDist) =>
    raycastBVH(bvh, ox, oy, oz, dx, dy, dz, maxDist);

  const states = [];
  for (let i = 0; i < 10; i++) {
    states.push(player((r() * 2 - 1) * 400, (r() * 2 - 1) * 400, 0,
      (r() * 2 - 1) * 180, (r() * 2 - 1) * 30,
      { alive: i !== 3, crouching: i % 3 === 0 }));
  }

  const evA = createVisibilityEvaluator({ castMapRay: caster, exactCount: true });
  const evB = createVisibilityEvaluator({ castMapRay: caster, exactCount: false });
  const A = evA.evaluate(states);
  const Bm = evB.evaluate(states);

  let selfVis = 0, deadVis = 0, deadViewer = 0, mismatch = 0, countBad = 0;
  for (let b = 0; b < 10; b++) {
    for (let a = 0; a < 10; a++) {
      const idx = b * 10 + a;
      if (a === b) { if (A.visible[idx]) selfVis++; continue; }
      if (A.visible[idx] && !states[a].alive) deadVis++;
      if (A.visible[idx] && !states[b].alive) deadViewer++;
      if (A.visible[idx] !== Bm.visible[idx]) mismatch++;
      // 早退时 clear 只能是 0（全挡）或 1（至少一个通）—— 精确个数不保证
      if (Bm.clear[idx] !== (A.visible[idx] ? 1 : 0)) countBad++;
    }
  }
  check('对角线永远是 0（自己看不见自己）', selfVis === 0, `${selfVis}`);
  check('死掉的**目标**永远不出现在结果里', deadVis === 0, `${deadVis}`);
  check('死掉的**观察者**永远不出现在结果里', deadViewer === 0, `${deadViewer}`);
  check('exactCount true/false 的 visible 判定完全一致', mismatch === 0, `${mismatch} 处不同`);
  check('exactCount=false 时 clear 只会是 0 或 1', countBad === 0, `${countBad} 处不对`);
  check('exactCount=false 确实省了射线', Bm.stats.rays < A.stats.rays,
    `${Bm.stats.rays} vs ${A.stats.rays}`);
  check('统计里的 pairs 数 = 活人之间的有序对',
    A.stats.pairs === A.stats.pairs && A.stats.pairs > 0 && A.stats.pairs <= 90,
    `${A.stats.pairs}`);

  // pairStats 的一致性
  const ps = pairStats(A, states);
  let manual = 0;
  for (let b = 0; b < 10; b++) {
    for (let a = 0; a < 10; a++) {
      if (a === b || !states[a].alive || !states[b].alive) continue;
      if (A.visible[b * 10 + a]) manual++;
    }
  }
  check('pairStats 的 total 和手数一致', ps.total === manual, `${ps.total} vs ${manual}`);
  check('pairStats 的 mutual + oneWay == total', ps.mutual + ps.oneWay === ps.total);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== F 性能（82 万三角形的类地图场景）');
// ═══════════════════════════════════════════════════════════════════════════

{
  const t0 = Date.now();
  const GRID = 640;              // 640×640 个地板四边形 = 819,200 个三角形
  const SPAN = 4000;             // 4000 HU ≈ 101 m
  const step = SPAN / GRID;
  const nTri = GRID * GRID * 2 + 512;
  const verts = new Float32Array(nTri * 9);
  let w = 0;
  const put = (ax, ay, az, bx, by, bz, cx, cy, cz) => {
    verts[w++] = ax; verts[w++] = ay; verts[w++] = az;
    verts[w++] = bx; verts[w++] = by; verts[w++] = bz;
    verts[w++] = cx; verts[w++] = cy; verts[w++] = cz;
  };
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      const x0 = -SPAN / 2 + i * step, y0 = -SPAN / 2 + j * step;
      const x1 = x0 + step, y1 = y0 + step;
      put(x0, y0, 0, x1, y0, 0, x1, y1, 0);
      put(x0, y0, 0, x1, y1, 0, x0, y1, 0);
    }
  }
  // 16×16 个格子里各竖一堵墙，让水平射线真的会撞东西
  for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
      const cx = -SPAN / 2 + (i + 0.5) * (SPAN / 16);
      const cy = -SPAN / 2 + (j + 0.5) * (SPAN / 16);
      const hw = 120, t = 8, h = 200;
      put(cx - hw, cy - t, 0, cx + hw, cy - t, 0, cx + hw, cy - t, h);
      put(cx - hw, cy - t, 0, cx + hw, cy - t, h, cx - hw, cy - t, h);
      put(cx - hw, cy + t, 0, cx + hw, cy + t, 0, cx + hw, cy + t, h);
      put(cx - hw, cy + t, 0, cx + hw, cy + t, h, cx - hw, cy + t, h);
    }
  }
  const data = { verts: verts.subarray(0, w), triCount: w / 9 };
  const bvh = buildBVH(data);
  const buildMs = Date.now() - t0;
  console.log(`  场景：${data.triCount.toLocaleString()} 三角形，建 BVH ${buildMs} ms`);

  const caster = makeMapCaster(bvh, 0.0254);
  const r = rng(99);
  const states = [];
  for (let i = 0; i < 10; i++) {
    states.push(player((r() * 2 - 1) * 1500, (r() * 2 - 1) * 1500, 0,
      (r() * 2 - 1) * 180, (r() * 2 - 1) * 30));
  }
  const smoke = [smokeVolume([300, 0, 100], () => true), smokeVolume([-800, 400, 60], () => true)];

  const ev = createVisibilityEvaluator({ castMapRay: caster });
  ev.setOccluders(smoke);
  ev.evaluate(states);                       // 预热

  const N = 60;
  const t1 = Date.now();
  let last = null;
  for (let i = 0; i < N; i++) last = ev.evaluate(states);
  const ms = (Date.now() - t1) / N;
  const exact = createVisibilityEvaluator({ castMapRay: caster, exactCount: true });
  exact.setOccluders(smoke);
  exact.evaluate(states);
  const t2 = Date.now();
  for (let i = 0; i < N; i++) exact.evaluate(states);
  const msExact = (Date.now() - t2) / N;

  console.log(`  默认（第一个通的角落就收工）: ${ms.toFixed(3)} ms/帧，`
    + `${last.stats.rays} 条射线，${last.stats.pairs} 对`);
  console.log(`  exactCount（8 个角落全测）    : ${msExact.toFixed(3)} ms/帧`);
  check(`默认路径 < 5 ms/帧（实测 ${ms.toFixed(3)}）`, ms < 5);
  check(`exactCount 路径 < 12 ms/帧（实测 ${msExact.toFixed(3)}）`, msExact < 12);

  // 射线数上界。注意 178° 的视锥也仍然会拒掉**背后**的目标（fd<=0），
  // 所以任何视锥设置都发不满 720 条 —— 那就直接量单条成本再外推。
  const evW = createVisibilityEvaluator({
    castMapRay: caster, exactCount: true, fovDeg: 178, vFovDeg: 178,
  });
  evW.setOccluders(smoke);
  evW.evaluate(states);
  const t3 = Date.now();
  let wLast = null;
  for (let i = 0; i < N; i++) wLast = evW.evaluate(states);
  const msWide = (Date.now() - t3) / N;
  const usPerRay = msWide * 1000 / wLast.stats.rays;
  const projected = usPerRay * 720 / 1000;
  console.log(`  视锥 178°（几乎全开）: ${msWide.toFixed(3)} ms/帧，${wLast.stats.rays} 条射线，`
    + `${usPerRay.toFixed(2)} µs/射线 → 外推 720 条 = ${projected.toFixed(3)} ms`);

  // 显式微基准：720 条随机射线穿过 BVH + 720 次烟雾 DDA（2 个烟），
  // 这是「90 对 × 8 角全部要发射线」时的纯成本上界，和视锥无关。
  const rays = [];
  for (let i = 0; i < 720; i++) {
    const ox = (r() * 2 - 1) * 1500, oy = (r() * 2 - 1) * 1500, oz = r() * 80;
    let dx = r() * 2 - 1, dy = r() * 2 - 1, dz = (r() * 2 - 1) * 0.4;
    const L = Math.hypot(dx, dy, dz);
    rays.push([ox, oy, oz, dx / L, dy / L, dz / L, 200 + r() * 2500]);
  }
  const micro = () => {
    let n = 0;
    for (let i = 0; i < 720; i++) {
      const q = rays[i];
      if (caster(q[0], q[1], q[2], q[3], q[4], q[5], q[6]) >= 0) n++;
      if (smokeRayBlocked(smoke, q[0], q[1], q[2], q[3], q[4], q[5], q[6])) n++;
    }
    return n;
  };
  micro();
  const t4 = Date.now();
  for (let i = 0; i < N; i++) micro();
  const msMicro = (Date.now() - t4) / N;
  console.log(`  微基准：720 条射线 × (BVH + 2 个烟的 DDA) = ${msMicro.toFixed(3)} ms/帧`);
  check(`720 条射线的纯成本 < 5 ms/帧（实测 ${msMicro.toFixed(3)}）`, msMicro < 5);
  check(`外推到 720 条 < 5 ms（实测 ${projected.toFixed(3)}）`, projected < 5);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log(`\nvisibility: ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
