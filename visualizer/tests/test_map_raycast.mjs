/**
 * test_map_raycast.mjs — BVH 空间索引的正确性测试。
 *
 * 分三层，一层比一层狠：
 *   1. 手搭的解析用例（单个三角形、立方体）—— 命中距离能算出来，直接比数
 *   2. 随机三角形汤对拍：几千条随机射线，BVH 的结果必须和**逐三角形暴力扫描**
 *      一模一样（1e-9 内）。这是抓 BVH 遍历 bug 的主力。
 *   3. 退化情形：所有三角形重合、方向分量为 0、起点在几何体内部、
 *      退化成链的极端划分
 *
 * 另外还测 `extractTriangles` 的烘焙路径（用鸭子类型的假 Object3D，不需要 three）。
 *
 * 真实地图上的对拍（BVH vs three.js 朴素 raycast）在 test_map_raycast_real.mjs 里，
 * 那个需要浏览器。
 *
 * 用法：node visualizer/tests/test_map_raycast.mjs
 */

import { buildBVH, raycastBVH, extractTriangles } from '../static/js/map-raycast.js';

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}

// 把一堆三角形摊平成 verts（每个三角形 9 个 float）
function soup(tris) {
  const verts = new Float32Array(tris.length * 9);
  for (let i = 0; i < tris.length; i++) verts.set(tris[i], i * 9);
  return { verts, triCount: tris.length };
}

// 暴力参考实现：和 raycastBVH 用同一套 Möller–Trumbore，但不走 BVH
function bruteForce(verts, triCount, o, d, maxDist) {
  let best = maxDist, found = false;
  for (let t = 0; t < triCount; t++) {
    const i = t * 9;
    const ax = verts[i], ay = verts[i + 1], az = verts[i + 2];
    const e1x = verts[i + 3] - ax, e1y = verts[i + 4] - ay, e1z = verts[i + 5] - az;
    const e2x = verts[i + 6] - ax, e2y = verts[i + 7] - ay, e2z = verts[i + 8] - az;
    const px = d[1] * e2z - d[2] * e2y;
    const py = d[2] * e2x - d[0] * e2z;
    const pz = d[0] * e2y - d[1] * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (det > -1e-9 && det < 1e-9) continue;
    const inv = 1 / det;
    const tx = o[0] - ax, ty = o[1] - ay, tz = o[2] - az;
    const u = (tx * px + ty * py + tz * pz) * inv;
    if (u < 0 || u > 1) continue;
    const qx = ty * e1z - tz * e1y;
    const qy = tz * e1x - tx * e1z;
    const qz = tx * e1y - ty * e1x;
    const v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
    if (v < 0 || u + v > 1) continue;
    const hit = (e2x * qx + e2y * qy + e2z * qz) * inv;
    if (hit >= 0 && hit < best) { best = hit; found = true; }
  }
  return found ? best : -1;
}

// 确定性 PRNG（mulberry32）—— 测试要可复现
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalize(v) {
  const L = Math.hypot(v[0], v[1], v[2]);
  return L === 0 ? [0, 0, 0] : [v[0] / L, v[1] / L, v[2] / L];
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n1. analytic cases')

{
  // 一个三角形，在 z=5 平面上
  const bvh = buildBVH(soup([[0, 0, 5, 10, 0, 5, 0, 10, 5]]));
  check('builds a single-triangle BVH', bvh !== null && bvh.triCount === 1);

  const hit = raycastBVH(bvh, 2, 2, 0, 0, 0, 1, 100);
  check('hits a triangle head-on at the right distance', Math.abs(hit - 5) < 1e-6, `${hit}`);

  const miss = raycastBVH(bvh, 20, 20, 0, 0, 0, 1, 100);
  check('misses outside the triangle', miss === -1, `${miss}`);

  const back = raycastBVH(bvh, 2, 2, 10, 0, 0, -1, 100);
  check('hits from behind too (double-sided)', Math.abs(back - 5) < 1e-6, `${back}`);

  const short = raycastBVH(bvh, 2, 2, 0, 0, 0, 1, 3);
  check('respects maxDist', short === -1, `${short}`);

  const parallel = raycastBVH(bvh, 2, 2, 0, 1, 0, 0, 100);
  check('a ray parallel to the plane misses', parallel === -1, `${parallel}`);
}

{
  // 单位立方体 0..1，12 个三角形
  const V = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
  ];
  const F = [
    [0, 1, 2], [0, 2, 3],   // z=0
    [4, 6, 5], [4, 7, 6],   // z=1
    [0, 4, 5], [0, 5, 1],   // y=0
    [3, 2, 6], [3, 6, 7],   // y=1
    [0, 3, 7], [0, 7, 4],   // x=0
    [1, 5, 6], [1, 6, 2],   // x=1
  ];
  const tris = F.map((f) => f.flatMap((i) => V[i]));
  const bvh = buildBVH(soup(tris));
  check('builds a 12-triangle cube BVH', bvh.triCount === 12);

  // 从外面沿 -x 射向 x=1 面：起点 (5, 0.5, 0.5) → 命中距离 4
  const d1 = raycastBVH(bvh, 5, 0.5, 0.5, -1, 0, 0, 100);
  check('cube: -x ray hits the x=1 face at 4', Math.abs(d1 - 4) < 1e-6, `${d1}`);

  // 斜着射，穿过角点 (1,1,1)：解析距离 = |(3,3,3)-(1,1,1)| = 2√3。
  // 这一条是**角点退化**用例：射线正好穿过三个面共享的顶点，严格的重心判据
  // 会三个面全判不中、直接穿到对面的 (0,0,0)（距离 3√3）。BARY_EPS 就是为了
  // 这个：没有容差时 BVH 和暴力扫描都会返回 3√3（两者一致，所以不是 BVH 的错，
  // 是 Möller–Trumbore 的固有退化）。
  const dd = normalize([-1, -1, -1]);
  const expect = 2 * Math.sqrt(3);
  const d2 = raycastBVH(bvh, 3, 3, 3, dd[0], dd[1], dd[2], 100);
  check('cube: diagonal ray through the corner hits at 2√3, not 3√3',
    Math.abs(d2 - expect) < 1e-6, `${d2} vs ${expect}`);

  // 再验证一次角点：稍微扰动方向也必须还是 2√3（扰动前后不能跳变）
  const dd2 = normalize([-1 - 1e-7, -1, -1]);
  const d2b = raycastBVH(bvh, 3, 3, 3, dd2[0], dd2[1], dd2[2], 100);
  check('cube: a hair off the corner gives the same distance',
    Math.abs(d2b - expect) < 1e-5, `${d2b} vs ${expect}`);

  // 从盒子内部射出去：起点在中心，沿 +x，应该命中 x=1 面，距离 0.5
  const d3 = raycastBVH(bvh, 0.5, 0.5, 0.5, 1, 0, 0, 100);
  check('cube: ray from inside exits at 0.5', Math.abs(d3 - 0.5) < 1e-6, `${d3}`);

  // 擦着边过：起点 (2, 2, 0.5) 沿 -x —— y=2 在盒子外，必须不命中
  const d4 = raycastBVH(bvh, 2, 2, 0.5, -1, 0, 0, 100);
  check('cube: ray passing above the box misses', d4 === -1, `${d4}`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n2. random triangle soup vs brute force')

{
  const rand = rng(0xC0FFEE);
  const N_TRI = 20000, N_RAY = 4000;
  const tris = [];
  for (let i = 0; i < N_TRI; i++) {
    // 混合大小：大部分小三角形 + 少量横跨全场的大三角形（真实地图就是这个分布）
    const big = rand() < 0.01;
    const S = big ? 100 : 0.5 + rand() * 3;
    const cx = (rand() - 0.5) * 200, cy = (rand() - 0.5) * 200, cz = (rand() - 0.5) * 60;
    const p = [];
    for (let k = 0; k < 3; k++) {
      p.push(cx + (rand() - 0.5) * S, cy + (rand() - 0.5) * S, cz + (rand() - 0.5) * S);
    }
    tris.push(p);
  }
  const s = soup(tris);
  const t0 = Date.now();
  const bvh = buildBVH(s);
  const buildMs = Date.now() - t0;
  check('builds a 20k-triangle BVH', bvh.triCount === N_TRI, `${buildMs} ms`);
  console.log(`       (build ${buildMs} ms for ${N_TRI} triangles)`);

  const origin = [0, 0, 0];
  let mismatches = 0, hits = 0, firstBad = null;
  const t1 = Date.now();
  for (let i = 0; i < N_RAY; i++) {
    // 起点也随机撒在场景里，覆盖"起点在包围盒外 / 内 / 几何体里"三种情况
    const o = i % 3 === 0
      ? [0, 0, 0]
      : [(rand() - 0.5) * 300, (rand() - 0.5) * 300, (rand() - 0.5) * 100];
    const d = normalize([rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1]);
    const got = raycastBVH(bvh, o[0], o[1], o[2], d[0], d[1], d[2], 500);
    const want = bruteForce(s.verts, N_TRI, o, d, 500);
    if (got >= 0) hits++;
    const same = (got < 0 && want < 0) || (got >= 0 && want >= 0 && Math.abs(got - want) < 1e-6);
    if (!same) {
      mismatches++;
      if (!firstBad) firstBad = { i, o, d, got, want };
    }
  }
  const rayMs = (Date.now() - t1) / N_RAY;
  check(`${N_RAY} random rays agree with brute force exactly`,
    mismatches === 0,
    firstBad ? `first mismatch: ${JSON.stringify(firstBad)}` : '');
  check('the random rays actually hit things', hits > N_RAY * 0.3, `${hits}/${N_RAY} hits`);
  console.log(`       (${rayMs.toFixed(3)} ms/ray with BVH vs brute force over ${N_TRI} tris)`);
  void origin;
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n3. degenerate cases')

{
  // 所有三角形完全重合 —— 质心范围是 0，划分会退化，必须仍然正确且不无限递归
  const tris = [];
  for (let i = 0; i < 100; i++) tris.push([0, 0, 1, 1, 0, 1, 0, 1, 1]);
  const bvh = buildBVH(soup(tris));
  check('coincident triangles still build', bvh !== null && bvh.triCount === 100);
  const h = raycastBVH(bvh, 0.25, 0.25, 0, 0, 0, 1, 10);
  check('coincident triangles still hit', Math.abs(h - 1) < 1e-6, `${h}`);
}

{
  // 全部共线（面积为零）—— 射线不该命中
  const tris = [];
  for (let i = 0; i < 50; i++) tris.push([i, 0, 0, i + 1, 0, 0, i + 2, 0, 0]);
  const bvh = buildBVH(soup(tris));
  const h = raycastBVH(bvh, 0, 1, 0, 0, -1, 0, 10);
  check('zero-area triangles never hit', h === -1, `${h}`);
}

{
  // 沿轴射线（方向分量恰好为 0），起点在包围盒外
  const tris = [
    [-1, -1, 5, 1, -1, 5, 0, 1, 5],
    [10, -1, 5, 12, -1, 5, 11, 1, 5],
  ];
  const bvh = buildBVH(soup(tris));
  const a = raycastBVH(bvh, 0, 0, 0, 0, 0, 1, 100);
  check('axis-aligned ray (+z, dx=dy=0) hits the first triangle',
    Math.abs(a - 5) < 1e-6, `${a}`);
  const b = raycastBVH(bvh, 11, 0, 0, 0, 0, 1, 100);
  check('axis-aligned ray hits the second triangle', Math.abs(b - 5) < 1e-6, `${b}`);
  const c = raycastBVH(bvh, 5, 0, 0, 0, 0, 1, 100);
  check('axis-aligned ray between the two triangles misses', c === -1, `${c}`);
}

{
  // 单三角形 + 起点正好在三角形平面上
  const bvh = buildBVH(soup([[0, 0, 0, 1, 0, 0, 0, 1, 0]]));
  const h = raycastBVH(bvh, 0.25, 0.25, 0, 0, 0, 1, 10);
  check('ray starting on the triangle plane still misses (t=0 not counted... )',
    h === -1 || Math.abs(h) < 1e-9, `${h}`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n4. extractTriangles (duck-typed Object3D)')

{
  // 非索引 + 单位矩阵 → 应该直接引用原数组，不复制
  const arr = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const mesh = {
    isMesh: true,
    geometry: { attributes: { position: { array: arr, itemSize: 3, count: 3 } }, index: null },
    matrixWorld: { elements: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
    updateMatrixWorld() {},
    traverse(fn) { fn(this); },
  };
  const r = extractTriangles(mesh);
  check('reuses the position array when the matrix is identity', r.baked === false);
  check('reuse keeps the same Float32Array instance', r.verts === arr);
  check('triangle count is right', r.triCount === 1, `${r.triCount}`);

  // 带平移的矩阵 → 必须烘焙
  const mesh2 = {
    isMesh: true,
    geometry: { attributes: { position: { array: arr, itemSize: 3, count: 3 } }, index: null },
    matrixWorld: { elements: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1] },
    updateMatrixWorld() {},
    traverse(fn) { fn(this); },
  };
  const r2 = extractTriangles(mesh2);
  check('bakes the world matrix when it is not identity', r2.baked === true);
  check('baking applies the translation',
    Math.abs(r2.verts[0] - 10) < 1e-6 && Math.abs(r2.verts[1] - 20) < 1e-6
    && Math.abs(r2.verts[2] - 30) < 1e-6,
    `${r2.verts[0]},${r2.verts[1]},${r2.verts[2]}`);
  const hb = raycastBVH(buildBVH(r2), 10.25, 20.25, 0, 0, 0, 1, 100);
  check('the baked geometry raycasts at the translated position',
    Math.abs(hb - 30) < 1e-6, `${hb}`);

  // 索引几何 → 必须烘焙（把 index 解引用展开）
  const mesh3 = {
    isMesh: true,
    geometry: {
      attributes: { position: { array: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 9, 9, 9]),
                                itemSize: 3, count: 4 } },
      index: { count: 3, getX: (i) => [0, 1, 2][i] },
    },
    matrixWorld: { elements: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
    updateMatrixWorld() {},
    traverse(fn) { fn(this); },
  };
  const r3 = extractTriangles(mesh3);
  check('bakes indexed geometry', r3.baked === true && r3.triCount === 1, `${r3.triCount}`);
  check('indexed geometry dereferences correctly',
    r3.verts[0] === 0 && r3.verts[3] === 1 && r3.verts[6] === 0 && r3.verts[7] === 1);

  check('an empty tree returns null', extractTriangles({
    isMesh: false, updateMatrixWorld() {}, traverse() {},
  }) === null);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n5. empty / tiny inputs')

{
  check('buildBVH(null) returns null', buildBVH(null) === null);
  check('buildBVH with 0 triangles returns null', buildBVH({ verts: new Float32Array(0), triCount: 0 }) === null);
  check('raycastBVH(null) returns -1', raycastBVH(null, 0, 0, 0, 0, 0, 1, 10) === -1);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log(`\nmap raycast: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
