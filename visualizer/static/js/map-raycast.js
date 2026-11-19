/**
 * map-raycast.js — 地图三角形的 BVH 空间索引（纯 typed array，不依赖 three）。
 *
 * 为什么需要它：de_ancient 的 OBJ 是**单个 mesh、823,214 个三角形**，three.js 的
 * 朴素 `Raycaster.intersectObjects(map.children, true)` 实测 **45 ms / 条射线**。
 * 10 个玩家每人一条瞄准射线 = 450 ms / 帧，完全没法用。而且三角形大小差 500 倍
 * （中位数 0.44 m，最大的一个 239 m —— 横跨整张地图的地板），所以均匀网格也不合适：
 * 1 m 网格要 1.38 亿次插入。BVH 对这种分布是天然的解法。
 *
 * 建完之后单条射线 <1 ms。同一个索引也顺手给 `updatePlayers` 里"找地面高度"的
 * 那条垂直射线用了。
 *
 * 这个文件**不 import three**，只吃 typed array，所以可以在 Node 里直接跑测试
 * （见 visualizer/tests/test_map_raycast.mjs，用 three.js 的朴素 raycast 当参考实现
 * 对拍）。
 */

const LEAF_SIZE = 8;      // 叶子里的三角形数；8 是遍历深度和测试量的常见折中
const MAX_DEPTH = 48;     // 防止退化数据把树拉成链
const EPS = 1e-9;
// 重心坐标的容差。严格 0..1 会让"射线正好穿过两个三角形共享的顶点/边"漏掉：
// 浮点误差让重心坐标落在 1+1e-16 这种位置，两个三角形就都被判成没命中，
// 射线会穿过去打到后面的东西（实测立方体对角射线会从 2√3 变成 3√3）。
// 1e-9 的重心容差 ≈ 亚纳米级的几何误差，对"射线打到哪面墙"没有影响。
const BARY_EPS = 1e-9;

// ── 从 three 的 Object3D 提取三角形 ─────────────────────────────────────────

function isIdentity(e) {
  return e[0] === 1 && e[1] === 0 && e[2] === 0 && e[3] === 0
      && e[4] === 0 && e[5] === 1 && e[6] === 0 && e[7] === 0
      && e[8] === 0 && e[9] === 0 && e[10] === 1 && e[11] === 0
      && e[12] === 0 && e[13] === 0 && e[14] === 0 && e[15] === 1;
}

/**
 * 把一棵 Object3D 子树里所有 mesh 的三角形展平成**世界坐标**的顶点表。
 *
 * 每个三角形占 9 个 float（ax,ay,az, bx,by,bz, cx,cy,cz），没有下标间接层 ——
 * 内层循环直接 `verts[t*9 + k]`，这是射线遍历的热路径。
 *
 * 常见情况（单 mesh + 单位矩阵 + 非索引几何，OBJLoader 就是这个形状）直接**引用**
 * 原来的 position 数组，不多花 30 MB。
 */
export function extractTriangles(root) {
  root.updateMatrixWorld(true);
  const meshes = [];
  root.traverse((o) => {
    if (o.isMesh && o.geometry && o.geometry.attributes && o.geometry.attributes.position) {
      meshes.push(o);
    }
  });
  if (meshes.length === 0) return null;

  const reusable = meshes.length === 1
    && isIdentity(meshes[0].matrixWorld.elements)
    && !meshes[0].geometry.index
    && meshes[0].geometry.attributes.position.array instanceof Float32Array
    && meshes[0].geometry.attributes.position.itemSize === 3;

  if (reusable) {
    const arr = meshes[0].geometry.attributes.position.array;
    return { verts: arr, triCount: Math.floor(arr.length / 9), baked: false };
  }

  // 慢路径：逐个 mesh 烘焙世界矩阵
  let triCount = 0;
  for (const m of meshes) {
    const pos = m.geometry.attributes.position;
    const idx = m.geometry.index;
    triCount += Math.floor((idx ? idx.count : pos.count) / 3);
  }
  const verts = new Float32Array(triCount * 9);
  let w = 0;
  for (const m of meshes) {
    const pos = m.geometry.attributes.position;
    const arr = pos.array;
    const idx = m.geometry.index;
    const e = m.matrixWorld.elements;
    const ident = isIdentity(e);
    const n = idx ? idx.count : pos.count;
    for (let i = 0; i < n; i++) {
      const vi = idx ? idx.getX(i) : i;
      const x = arr[vi * 3], y = arr[vi * 3 + 1], z = arr[vi * 3 + 2];
      if (ident) {
        verts[w] = x; verts[w + 1] = y; verts[w + 2] = z;
      } else {
        verts[w]     = e[0] * x + e[4] * y + e[8]  * z + e[12];
        verts[w + 1] = e[1] * x + e[5] * y + e[9]  * z + e[13];
        verts[w + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
      }
      w += 3;
    }
  }
  return { verts, triCount, baked: true };
}

// ── 建树 ────────────────────────────────────────────────────────────────────

/**
 * 建 BVH。返回的对象里全是 typed array，可以直接结构化克隆 / 传进 worker。
 *
 * 划分策略：按**质心范围最长轴的中点**切（不用排序，一趟双指针）。
 * 如果所有质心都落在同一侧（退化数据），退化成按下标中位数硬切 —— 保证每层都有进展，
 * 不会无限递归。
 *
 * @param {{verts: Float32Array, triCount: number}} tris  extractTriangles 的输出
 * @returns {object|null}
 */
export function buildBVH(tris) {
  if (!tris || tris.triCount <= 0) return null;
  const { verts, triCount } = tris;

  const triOrder = new Uint32Array(triCount);
  const triMin = new Float32Array(triCount * 3);
  const triMax = new Float32Array(triCount * 3);

  let gminx = Infinity, gminy = Infinity, gminz = Infinity;
  let gmaxx = -Infinity, gmaxy = -Infinity, gmaxz = -Infinity;

  for (let t = 0; t < triCount; t++) {
    triOrder[t] = t;
    const i = t * 9;
    const ax = verts[i], ay = verts[i + 1], az = verts[i + 2];
    const bx = verts[i + 3], by = verts[i + 4], bz = verts[i + 5];
    const cx = verts[i + 6], cy = verts[i + 7], cz = verts[i + 8];
    const mnx = Math.min(ax, bx, cx), mny = Math.min(ay, by, cy), mnz = Math.min(az, bz, cz);
    const mxx = Math.max(ax, bx, cx), mxy = Math.max(ay, by, cy), mxz = Math.max(az, bz, cz);
    triMin[t * 3] = mnx; triMin[t * 3 + 1] = mny; triMin[t * 3 + 2] = mnz;
    triMax[t * 3] = mxx; triMax[t * 3 + 1] = mxy; triMax[t * 3 + 2] = mxz;
    if (mnx < gminx) gminx = mnx;
    if (mny < gminy) gminy = mny;
    if (mnz < gminz) gminz = mnz;
    if (mxx > gmaxx) gmaxx = mxx;
    if (mxy > gmaxy) gmaxy = mxy;
    if (mxz > gmaxz) gmaxz = mxz;
  }

  // 节点表：先用普通数组 push，最后转 typed array
  const nMin = [], nMax = [], nLeft = [], nRight = [], nStart = [], nCount = [];

  function newNode() {
    nMin.push(0, 0, 0);
    nMax.push(0, 0, 0);
    nLeft.push(-1);
    nRight.push(-1);
    nStart.push(0);
    nCount.push(0);
    return nCount.length - 1;
  }

  function build(lo, hi, depth) {
    const ni = newNode();

    // 节点包围盒 + 质心包围盒
    let mnx = Infinity, mny = Infinity, mnz = Infinity;
    let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
    let cnx = Infinity, cny = Infinity, cnz = Infinity;
    let cxx = -Infinity, cxy = -Infinity, cxz = -Infinity;
    for (let k = lo; k < hi; k++) {
      const t = triOrder[k], a = t * 3;
      const ax = triMin[a], ay = triMin[a + 1], az = triMin[a + 2];
      const bx = triMax[a], by = triMax[a + 1], bz = triMax[a + 2];
      if (ax < mnx) mnx = ax;
      if (ay < mny) mny = ay;
      if (az < mnz) mnz = az;
      if (bx > mxx) mxx = bx;
      if (by > mxy) mxy = by;
      if (bz > mxz) mxz = bz;
      const gx = (ax + bx) * 0.5, gy = (ay + by) * 0.5, gz = (az + bz) * 0.5;
      if (gx < cnx) cnx = gx;
      if (gy < cny) cny = gy;
      if (gz < cnz) cnz = gz;
      if (gx > cxx) cxx = gx;
      if (gy > cxy) cxy = gy;
      if (gz > cxz) cxz = gz;
    }
    nMin[ni * 3] = mnx; nMin[ni * 3 + 1] = mny; nMin[ni * 3 + 2] = mnz;
    nMax[ni * 3] = mxx; nMax[ni * 3 + 1] = mxy; nMax[ni * 3 + 2] = mxz;

    const count = hi - lo;
    if (count <= LEAF_SIZE || depth >= MAX_DEPTH) {
      nStart[ni] = lo;
      nCount[ni] = count;
      return ni;
    }

    // 质心范围最长的轴
    const ex = cxx - cnx, ey = cxy - cny, ez = cxz - cnz;
    let axis = 0, ext = ex;
    if (ey > ext) { axis = 1; ext = ey; }
    if (ez > ext) { axis = 2; ext = ez; }

    if (ext <= 0) {                    // 所有质心重合，切不动了
      nStart[ni] = lo;
      nCount[ni] = count;
      return ni;
    }
    const split = (axis === 0 ? cnx + cxx : axis === 1 ? cny + cxy : cnz + cxz) * 0.5;

    // 双指针就地划分：质心 < split 的留在左边
    let i = lo, j = hi - 1;
    while (i <= j) {
      const t = triOrder[i];
      const c = (triMin[t * 3 + axis] + triMax[t * 3 + axis]) * 0.5;
      if (c < split) {
        i++;
      } else {
        const tmp = triOrder[i];
        triOrder[i] = triOrder[j];
        triOrder[j] = tmp;
        j--;
      }
    }
    let mid = i;
    if (mid === lo || mid === hi) mid = (lo + hi) >> 1;   // 退化：按下标中位数硬切

    nStart[ni] = 0;
    nCount[ni] = 0;
    nLeft[ni] = build(lo, mid, depth + 1);
    nRight[ni] = build(mid, hi, depth + 1);
    return ni;
  }

  build(0, triCount, 0);

  return {
    verts,
    triCount,
    triOrder,
    nodeMin: new Float32Array(nMin),
    nodeMax: new Float32Array(nMax),
    nodeLeft: new Int32Array(nLeft),
    nodeRight: new Int32Array(nRight),
    nodeStart: new Uint32Array(nStart),
    nodeCount: new Uint32Array(nCount),
    bounds: { min: [gminx, gminy, gminz], max: [gmaxx, gmaxy, gmaxz] },
    _stack: new Int32Array(256),
  };
}

// ── 射线遍历 ────────────────────────────────────────────────────────────────

/**
 * Möller–Trumbore，双面。命中返回射线参数 t（方向已归一化时就是距离），否则 -1。
 */
function rayTri(verts, t, ox, oy, oz, dx, dy, dz) {
  const i = t * 9;
  const ax = verts[i], ay = verts[i + 1], az = verts[i + 2];
  const e1x = verts[i + 3] - ax, e1y = verts[i + 4] - ay, e1z = verts[i + 5] - az;
  const e2x = verts[i + 6] - ax, e2y = verts[i + 7] - ay, e2z = verts[i + 8] - az;

  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (det > -EPS && det < EPS) return -1;          // 射线和三角形共面

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

/**
 * 射线 vs AABB 的 slab 测试。返回进入距离（射线起点在盒内时为 0），未命中返回 -1。
 * 方向分量可以恰好是 0（此时 inv 是 ±Infinity），标准写法下 NaN 只会出现在
 * "射线正好贴在slab 面上"这种零测度情形，会被比较运算自然忽略。
 */
function slab(nodeMin, nodeMax, ni, ox, oy, oz, invx, invy, invz) {
  const b = ni * 3;
  let t0 = (nodeMin[b] - ox) * invx, t1 = (nodeMax[b] - ox) * invx;
  let tmin = t0 < t1 ? t0 : t1;
  let tmax = t0 < t1 ? t1 : t0;

  t0 = (nodeMin[b + 1] - oy) * invy;
  t1 = (nodeMax[b + 1] - oy) * invy;
  if (t0 > t1) { const s = t0; t0 = t1; t1 = s; }
  if (t0 > tmin) tmin = t0;
  if (t1 < tmax) tmax = t1;
  if (tmin > tmax) return -1;

  t0 = (nodeMin[b + 2] - oz) * invz;
  t1 = (nodeMax[b + 2] - oz) * invz;
  if (t0 > t1) { const s = t0; t0 = t1; t1 = s; }
  if (t0 > tmin) tmin = t0;
  if (t1 < tmax) tmax = t1;
  if (tmin > tmax) return -1;

  return tmin < 0 ? 0 : tmin;
}

/**
 * 最近的交点距离。没命中返回 -1。
 *
 * @param {object} bvh
 * @param {number} ox,oy,oz 起点
 * @param {number} dx,dy,dz 方向（**必须已归一化**，否则返回的是参数不是距离）
 * @param {number} maxDist  最远距离
 */
export function raycastBVH(bvh, ox, oy, oz, dx, dy, dz, maxDist) {
  if (!bvh) return -1;
  const { verts, triOrder, nodeMin, nodeMax, nodeLeft, nodeRight, nodeStart, nodeCount } = bvh;
  const invx = 1 / dx, invy = 1 / dy, invz = 1 / dz;

  const stack = bvh._stack;
  let sp = 0;
  stack[sp++] = 0;

  let best = maxDist;
  let found = false;

  while (sp > 0) {
    const ni = stack[--sp];
    const tmin = slab(nodeMin, nodeMax, ni, ox, oy, oz, invx, invy, invz);
    if (tmin < 0 || tmin > best) continue;         // 整棵子树都比当前最近解远

    const c = nodeCount[ni];
    if (c > 0) {
      const s = nodeStart[ni];
      for (let k = s; k < s + c; k++) {
        const t = triOrder[k];
        const hit = rayTri(verts, t, ox, oy, oz, dx, dy, dz);
        if (hit >= 0 && hit < best) { best = hit; found = true; }
      }
    } else {
      // 远的先压栈，近的后压栈 → 先弹近的，能更早收紧 best 剪掉远的子树
      const l = nodeLeft[ni], r = nodeRight[ni];
      const tl = slab(nodeMin, nodeMax, l, ox, oy, oz, invx, invy, invz);
      const tr = slab(nodeMin, nodeMax, r, ox, oy, oz, invx, invy, invz);
      const lOk = tl >= 0 && tl <= best;
      const rOk = tr >= 0 && tr <= best;
      if (lOk && rOk) {
        if (tl <= tr) { stack[sp++] = r; stack[sp++] = l; }
        else { stack[sp++] = l; stack[sp++] = r; }
      } else if (lOk) {
        stack[sp++] = l;
      } else if (rOk) {
        stack[sp++] = r;
      }
    }
  }
  return found ? best : -1;
}
