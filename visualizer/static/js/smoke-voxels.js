/**
 * smoke-voxels.js — CS2 烟雾体素 journal 的 JS 解码器。
 *
 * 这是一个**逐行对照** `tools/demo_extract/voxel.py` 的移植。服务端把每个
 * 烟雾实例的**原始 journal 变更记录**（append / full）发给前端，这里负责：
 *
 *   1. 把变更记录折叠成"某个 tick 的完整 journal 字节"
 *   2. 解析 journal 记录（KEYFRAME / DELTA / HEARTBEAT）
 *   3. Morton 解码出**边界体素**（云表面）
 *   4. 表面采样很稀，填满表面点云的**凸包**，得到**实心体素**（云内部）
 *   5. 栅格坐标 → 世界坐标（Hammer 单位）
 *
 * 不依赖 three.js，所以可以直接用 node 跑单元测试对照 Python 的结果。
 *
 * journal 布局
 * ------------
 *   record  : u16le seq | u16le payload_len | payload
 *   payload[0] = phase, payload[1] = frame_type
 *     type 3 KEYFRAME : u8 seed_count,
 *                       seed_count x 8B [x, y, z, density, pad x 4],
 *                       u16 mask_count,
 *                       mask_count x 10B [cell_morton u16, voxel_mask u64]
 *     type 2 DELTA    : u16 mask_count, mask_count x 10B replacement masks
 *     type 0 HEARTBEAT: 无几何变化
 *
 * 栅格 32^3 = 32768 体素。世界坐标 world = detonation + (grid - 15.5) * 12.0
 */

export const VOXEL_GRID_DIM = 32;
export const VOXEL_WORLD_SIZE = 12.0; // 每个体素 12 Hammer unit
export const VOXEL_GRID_CENTER = (VOXEL_GRID_DIM - 1) / 2.0; // 15.5

export const KEYFRAME = 3;
export const DELTA = 2;
export const HEARTBEAT = 0;
export const FRAME_TYPE_NAMES = { 3: 'keyframe', 2: 'delta', 0: 'heartbeat' };

// ── journal 解析 ────────────────────────────────────────────────────────────

/** 把 journal 字节拆成 [{seq, payload}]；size 限制有效长度。 */
export function splitJournal(raw, size) {
  const n = size == null ? raw.length : Math.min(size, raw.length);
  const frames = [];
  let off = 0;
  while (off + 4 <= n) {
    const seq = raw[off] | (raw[off + 1] << 8);
    const length = raw[off + 2] | (raw[off + 3] << 8);
    const end = off + 4 + length;
    if (end > n) break;
    frames.push({ seq, payload: raw.subarray(off + 4, end) });
    off = end;
  }
  return frames;
}

export function frameType(payload) {
  return payload.length > 1 ? payload[1] : -1;
}

/** KEYFRAME payload → {seeds:[{x,y,z}], masks:[{cell, mask}]}。 */
export function applyKeyframe(payload) {
  const seeds = [];
  const masks = [];
  if (payload.length < 3) return { seeds, masks };
  const count = payload[2];
  let off = 3;
  for (let i = 0; i < count; i++) {
    if (off + 8 > payload.length) break;
    seeds.push({ x: payload[off], y: payload[off + 1], z: payload[off + 2] });
    off += 8;
  }
  if (off + 2 > payload.length) return { seeds, masks };
  const maskCount = payload[off] | (payload[off + 1] << 8);
  off += 2;
  for (let i = 0; i < maskCount; i++) {
    if (off + 10 > payload.length) break;
    const cell = payload[off] | (payload[off + 1] << 8);
    const mask = readU64(payload, off + 2);
    masks.push({ cell, mask });
    off += 10;
  }
  return { seeds, masks };
}

/** DELTA payload → [{cell, mask}]。 */
export function applyDelta(payload) {
  const masks = [];
  if (payload.length < 4) return masks;
  const maskCount = payload[2] | (payload[3] << 8);
  let off = 4;
  for (let i = 0; i < maskCount; i++) {
    if (off + 10 > payload.length) break;
    const cell = payload[off] | (payload[off + 1] << 8);
    const mask = readU64(payload, off + 2);
    masks.push({ cell, mask });
    off += 10;
  }
  return masks;
}

/** 小端 u64。体素掩码只有低 64 位有意义，用 BigInt 保精度。 */
function readU64(buf, off) {
  let v = 0n;
  for (let i = 7; i >= 0; i--) v = (v << 8n) | BigInt(buf[off + i]);
  return v;
}

// ── Morton ──────────────────────────────────────────────────────────────────

export function demorton3(index, bits) {
  let x = 0, y = 0, z = 0;
  for (let b = 0; b < bits; b++) {
    x |= ((index >> (3 * b)) & 1) << b;
    y |= ((index >> (3 * b + 1)) & 1) << b;
    z |= ((index >> (3 * b + 2)) & 1) << b;
  }
  return [x, y, z];
}

export function morton3(x, y, z, bits) {
  let out = 0;
  for (let b = 0; b < bits; b++) {
    out |= ((x >> b) & 1) << (3 * b);
    out |= ((y >> b) & 1) << (3 * b + 1);
    out |= ((z >> b) & 1) << (3 * b + 2);
  }
  return out;
}

// ── 几何 ────────────────────────────────────────────────────────────────────

/**
 * journal 中掩码直接点名的体素（云**边界**）。
 * @returns {Uint8Array} 长度 32768 的占据栅格
 */
export function boundaryGrid(journal, size) {
  const grid = new Uint8Array(VOXEL_GRID_DIM ** 3);
  const frames = splitJournal(journal, size);
  for (const { payload } of frames) {
    const ft = frameType(payload);
    if (ft === KEYFRAME) {
      const { seeds, masks } = applyKeyframe(payload);
      for (const s of seeds) mark(grid, s.x, s.y, s.z);
      for (const m of masks) applyMask(grid, m.cell, m.mask);
    } else if (ft === DELTA) {
      for (const m of applyDelta(payload)) applyMask(grid, m.cell, m.mask);
    }
  }
  return grid;
}

export function countGrid(grid) {
  let n = 0;
  for (let i = 0; i < grid.length; i++) if (grid[i]) n++;
  return n;
}

function mark(grid, x, y, z) {
  if (x >= 0 && x < 32 && y >= 0 && y < 32 && z >= 0 && z < 32) {
    grid[(x * 32 + y) * 32 + z] = 1;
  }
}

function applyMask(grid, cell, mask) {
  if (mask === 0n) return;
  const [cx, cy, cz] = demorton3(cell, 3);
  let m = mask;
  while (m) {
    // 取最低置位
    const bit = trailingZeros64(m);
    const [sx, sy, sz] = demorton3(bit, 2);
    mark(grid, cx * 4 + sx, cy * 4 + sy, cz * 4 + sz);
    m &= m - 1n;
  }
}

function trailingZeros64(v) {
  let n = 0;
  while ((v & 1n) === 0n) {
    v >>= 1n;
    n++;
  }
  return n;
}

/**
 * 边界 + 凸包内部（实心云）。
 *
 * 注意：journal 里的掩码是**云表面**，而且采样得很稀（真实 demo 上平均每列
 * 只有 ~4 个 Z、没有哪个格子的 64 位全满）。所以六个外表面的洪泛填充在真实
 * 数据上**一个内部体素都补不出来**（`test.dem` 82 个烟里 59 个增益为 0）——
 * 只画边界壳就是"两边高、中间空"的那种破形状。
 * 因此实心云改成 `solidFromShell`：把表面点云的**凸包**填满。
 *
 * @returns {Uint8Array} 长度 32768 的占据栅格
 */
export function solidGrid(journal, size) {
  return solidFromShell(boundaryGrid(journal, size));
}

/**
 * 判定「这个格子在不在凸包内」用的容差。
 *
 * 凸包的面正好穿过格点，所以很多格子恰好在边界上。实测 82 个真实烟雾：
 * 容差取 1e-12 ~ 1e-3 结果**完全相同**，所以这个值只需要吸收 JS 和 Python
 * 两套凸包实现之间的浮点噪声。超过 ~1e-3 填充就会开始变大。
 */
export const HULL_EPS = 1e-6;

function hullSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function hullCross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function hullDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function hullDist2(a, b) {
  const d = hullSub(a, b);
  return hullDot(d, d);
}

/** 点 p 到直线 ab 的距离平方（乘上 |ab|^2，省一次开方）。 */
function hullLineDist2(p, a, b) {
  const c = hullCross(hullSub(b, a), hullSub(p, a));
  return hullDot(c, c);
}

/**
 * 点云的凸包，返回半空间列表 `{nx, ny, nz, c}`，约定 `n·p - c <= 0` 为内部。
 *
 * 这是 `tools/demo_extract/voxel.py` 里 `convex_hull_halfspaces()` 的**逐行移植**
 * （增量式 beneath-beyond）。之所以不用 three.js 自带的 `ConvexHull`：那个文件
 * 自己 `import ... from 'three'`，会把本模块变成依赖 three —— 而本模块要在 node 里
 * 直接被 import 来做 Python 对照测试。移植版已在 Python 侧对 scipy 验证过。
 *
 * @param {Array<[number,number,number]>} points
 * @returns {Array<{nx:number,ny:number,nz:number,c:number}>|null}
 */
export function convexHullHalfspaces(points) {
  const seen = new Set();
  const pts = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const key = `${p[0]},${p[1]},${p[2]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pts.push([p[0], p[1], p[2]]);
  }
  if (pts.length < 4) return null;

  // ── 初始四面体 ──────────────────────────────────────────────────────────
  let i0 = 0;
  let i1 = 0;
  let best = -1;
  for (let i = 0; i < pts.length; i++) {
    const d = hullDist2(pts[i], pts[i0]);
    if (d > best) { best = d; i1 = i; }
  }
  if (hullDist2(pts[i1], pts[i0]) <= HULL_EPS) return null;

  let i2 = 0;
  best = -1;
  for (let i = 0; i < pts.length; i++) {
    const d = hullLineDist2(pts[i], pts[i0], pts[i1]);
    if (d > best) { best = d; i2 = i; }
  }
  if (hullLineDist2(pts[i2], pts[i0], pts[i1]) <= HULL_EPS) return null;

  const n012 = hullCross(hullSub(pts[i1], pts[i0]), hullSub(pts[i2], pts[i0]));
  let i3 = 0;
  best = -1;
  for (let i = 0; i < pts.length; i++) {
    const d = Math.abs(hullDot(n012, hullSub(pts[i], pts[i0])));
    if (d > best) { best = d; i3 = i; }
  }
  if (Math.abs(hullDot(n012, hullSub(pts[i3], pts[i0]))) <= HULL_EPS) return null;

  // 四面体的重心永远严格在凸包内部，用它来判断「哪边是外面」，从而给每个面定向。
  const centre = [0, 1, 2].map(
    (k) => (pts[i0][k] + pts[i1][k] + pts[i2][k] + pts[i3][k]) / 4.0
  );

  const oriented = (a, b, c) => {
    const n = hullCross(hullSub(pts[b], pts[a]), hullSub(pts[c], pts[a]));
    return hullDot(n, hullSub(pts[a], centre)) < 0 ? [a, c, b] : [a, b, c];
  };

  let faces = [
    oriented(i0, i1, i2),
    oriented(i0, i1, i3),
    oriented(i0, i2, i3),
    oriented(i1, i2, i3),
  ];
  const onHull = new Set([i0, i1, i2, i3]);

  for (let ip = 0; ip < pts.length; ip++) {
    if (onHull.has(ip)) continue;
    const p = pts[ip];
    const visible = [];
    for (let fi = 0; fi < faces.length; fi++) {
      const [a, b, c] = faces[fi];
      const n = hullCross(hullSub(pts[b], pts[a]), hullSub(pts[c], pts[a]));
      if (hullDot(n, hullSub(p, pts[a])) > HULL_EPS) visible.push(fi);
    }
    if (!visible.length) continue;
    onHull.add(ip);

    // 地平线 = 只被一个可见面用到的边（内部边被两个可见面共用，会抵消掉）
    const edges = new Map();
    for (const fi of visible) {
      const [a, b, c] = faces[fi];
      const tri = [[a, b], [b, c], [c, a]];
      for (const [u, v] of tri) {
        const rev = `${v},${u}`;
        if (edges.has(rev)) edges.set(rev, edges.get(rev) + 1);
        else edges.set(`${u},${v}`, (edges.get(`${u},${v}`) || 0) + 1);
      }
    }
    const horizon = [];
    for (const [key, count] of edges) {
      if (count !== 1) continue;
      const [u, v] = key.split(',').map(Number);
      horizon.push([u, v]);
    }

    const keep = [];
    for (let fi = 0; fi < faces.length; fi++) if (!visible.includes(fi)) keep.push(faces[fi]);
    faces = keep;
    for (const [a, b] of horizon) faces.push(oriented(a, b, ip));
  }

  const out = [];
  for (const [a, b, c] of faces) {
    const n = hullCross(hullSub(pts[b], pts[a]), hullSub(pts[c], pts[a]));
    const length = Math.sqrt(hullDot(n, n));
    if (length <= HULL_EPS) continue;
    const unit = [n[0] / length, n[1] / length, n[2] / length];
    out.push({ nx: unit[0], ny: unit[1], nz: unit[2], c: hullDot(unit, pts[a]) });
  }
  return out.length >= 4 ? out : null;
}

/**
 * 凸包内的所有格子。
 *
 * 凸体的竖直截面**一定是一个区间**，所以"填满凸包"就是逐列填满 —— 只是每列的
 * 上下界来自凸包的下/上表面，而不是"采样到的最高/最低体素"。这正是它不需要任何
 * 调参的原因：凸包自己把稀疏采样之间的空隙插值掉了。
 *
 * 近水平的半空间（`|nz| <= eps`）不含 Z 信息，它们决定的是凸包在 (x,y) 平面上的
 * **投影范围**，必须单独判定 —— 漏掉这一步填充会顺着整个栅格漏到凸包外面去。
 *
 * @param {Uint8Array} shell 边界栅格
 * @returns {Uint8Array} 实心栅格
 */
export function convexHullFillGrid(shell) {
  const D = VOXEL_GRID_DIM;
  const points = [];
  for (let x = 0; x < D; x++) {
    for (let y = 0; y < D; y++) {
      const base = (x * D + y) * D;
      for (let z = 0; z < D; z++) {
        if (shell[base + z]) points.push([x, y, z]);
      }
    }
  }
  const hs = convexHullHalfspaces(points);
  if (!hs) return shell.slice();

  const up = [];
  const down = [];
  const flat = [];
  for (const h of hs) {
    if (h.nz > HULL_EPS) up.push(h);
    else if (h.nz < -HULL_EPS) down.push(h);
    else flat.push(h);
  }
  if (!up.length || !down.length) return shell.slice();

  const out = new Uint8Array(D ** 3);
  for (let x = 0; x < D; x++) {
    for (let y = 0; y < D; y++) {
      // (x,y) 是否落在凸包的投影里 —— 只由近竖直的半空间决定
      let shadow = true;
      for (let i = 0; i < flat.length; i++) {
        const h = flat[i];
        if (h.nx * x + h.ny * y - h.c > HULL_EPS) { shadow = false; break; }
      }
      if (!shadow) continue;

      let hi = Infinity;
      for (let i = 0; i < up.length; i++) {
        const h = up[i];
        const z = (h.c - h.nx * x - h.ny * y) / h.nz;
        if (z < hi) hi = z;
      }
      let lo = -Infinity;
      for (let i = 0; i < down.length; i++) {
        const h = down[i];
        const z = (h.c - h.nx * x - h.ny * y) / h.nz;
        if (z > lo) lo = z;
      }
      if (hi < lo - HULL_EPS) continue;

      const z0 = Math.max(0, Math.ceil(lo - HULL_EPS));
      const z1 = Math.min(D - 1, Math.floor(hi + HULL_EPS));
      const base = (x * D + y) * D;
      for (let z = z0; z <= z1; z++) out[base + z] = 1;
    }
  }
  return out;
}

/**
 * 表面壳 → 实心云：填满表面点云的凸包。
 *
 * ⚠️ 这是**渲染启发式，不是格式事实**：journal 存的是稀疏表面，内部根本不在数据里，
 * 上游参考实现也没打算还原内部（它只取每列 min/max Z 画 2D 雷达轮廓）。
 * 凸包还会把所有凹陷填平。要真实占据请用 `boundaryGrid`。
 *
 * 这个函数只影响**画面**，不参与任何 `.csz` 的读写。
 */
export function solidFromShell(shell) {
  return convexHullFillGrid(shell);
}

/**
 * 栅格体素 → 世界坐标（Hammer 单位）。
 * @returns {Float32Array} 扁平 [x,y,z, x,y,z, ...]
 */
export function gridToWorldPoints(grid, detonation) {
  const [ox, oy, oz] = detonation;
  const out = [];
  const D = VOXEL_GRID_DIM;
  for (let x = 0; x < D; x++) {
    for (let y = 0; y < D; y++) {
      const base = (x * D + y) * D;
      for (let z = 0; z < D; z++) {
        if (!grid[base + z]) continue;
        out.push(
          ox + (x - VOXEL_GRID_CENTER) * VOXEL_WORLD_SIZE,
          oy + (y - VOXEL_GRID_CENTER) * VOXEL_WORLD_SIZE,
          oz + (z - VOXEL_GRID_CENTER) * VOXEL_WORLD_SIZE
        );
      }
    }
  }
  return Float32Array.from(out);
}

// ── base64 → 字节 ───────────────────────────────────────────────────────────

/** 服务端把 journal 变更的 data 用 base64 发过来。 */
export function b64ToBytes(b64) {
  if (!b64) return new Uint8Array(0);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ── 实例级重建缓存 ──────────────────────────────────────────────────────────

/**
 * 一个烟雾实例的 journal 折叠与体素重建缓存。
 *
 * 播放时 tick 单调前进，所以 journal 折叠可以增量做（append 模式只接尾巴）；
 * 反过来拖时间线时退化成从头重折，但每次重建结果按变更序号缓存，
 * 来回拖动不会重复算。
 */
export class SmokeVoxelCache {
  /** @param {object} inst 服务端发的 smoke 实例（changes[].data 为 base64） */
  constructor(inst, { solid = true } = {}) {
    this.inst = inst;
    this.solid = solid;
    this.detonation = inst.detonation || [0, 0, 0];
    this.changes = (inst.changes || []).map((c) => ({
      ...c,
      bytes: b64ToBytes(c.data),
    }));
    this.ticks = this.changes.map((c) => c.tick);
    this.reset();
  }

  reset() {
    this._foldIndex = -1;         // 已折叠到的变更序号
    this._foldBuf = new Uint8Array(0);
    this._foldSize = 0;
    this._voxIndex = -1;
    this._voxGrid = null;
    this._voxPoints = null;
    this._results = new Map();    // 变更序号 -> {grid, points}
    this.lastStats = { records: 0, keyframe: 0, delta: 0, heartbeat: 0, boundary: 0, solid: 0 };
  }

  /** 变更序号：最后一个 tick <= tick 的记录。返回 -1 表示还没开始。 */
  indexForTick(tick) {
    const t = this.ticks;
    if (!t.length) return -1;
    if (tick < t[0]) return -1;
    let lo = 0, hi = t.length - 1, ans = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (t[mid] <= tick) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return ans;
  }

  /** 折叠 journal 到变更序号 i（返回 {buf, size}）。 */
  _fold(i) {
    if (i < 0) return { buf: new Uint8Array(0), size: 0 };
    if (i === this._foldIndex) return { buf: this._foldBuf, size: this._foldSize };

    let buf, size, start;
    if (i > this._foldIndex && this._foldIndex >= 0) {
      // 增量：从上次的位置继续接
      buf = this._foldBuf;
      start = this._foldIndex + 1;
    } else {
      buf = new Uint8Array(0);
      start = 0;
    }
    for (let k = start; k <= i; k++) {
      const c = this.changes[k];
      if (c.mode === 'append') {
        const merged = new Uint8Array(buf.length + c.bytes.length);
        merged.set(buf, 0);
        merged.set(c.bytes, buf.length);
        buf = merged;
      } else {
        buf = c.bytes;
      }
      size = c.size || buf.length;
    }
    this._foldIndex = i;
    this._foldBuf = buf;
    this._foldSize = size;
    return { buf, size };
  }

  /**
   * 某个 tick 的几何。
   * @returns {{points: Float32Array, grid: Uint8Array, count: number}|null}
   *          null = 该 tick 烟雾还没出现 / 已消失
   */
  at(tick) {
    const inst = this.inst;
    if (tick < inst.first_tick || tick > inst.last_tick) return null;
    const i = this.indexForTick(tick);
    if (i < 0) return null;
    if (i === this._voxIndex) {
      return { points: this._voxPoints, grid: this._voxGrid, count: this._voxPoints.length / 3 };
    }
    const cached = this._results.get(i);
    if (cached) {
      this._voxIndex = i;
      this._voxGrid = cached.grid;
      this._voxPoints = cached.points;
      return { points: cached.points, grid: cached.grid, count: cached.points.length / 3 };
    }

    const { buf, size } = this._fold(i);
    const boundary = boundaryGrid(buf, size);
    const grid = this.solid ? solidFromShell(boundary) : boundary;
    const points = gridToWorldPoints(grid, this.detonation);
    this._results.set(i, { grid, points });
    this._voxIndex = i;
    this._voxGrid = grid;
    this._voxPoints = points;
    this._updateStats(buf, size, i, boundary, grid);
    return { points, grid, count: points.length / 3 };
  }

  _updateStats(buf, size, i, boundary, grid) {
    const frames = splitJournal(buf, size);
    const counts = { records: frames.length, keyframe: 0, delta: 0, heartbeat: 0, other: 0 };
    for (const { payload } of frames) {
      const name = FRAME_TYPE_NAMES[frameType(payload)];
      if (name) counts[name]++;
      else counts.other++;
    }
    this.lastStats = {
      ...counts,
      scope: 'journal',
      boundary: countGrid(boundary),
      solid: countGrid(grid),
      mode: this.solid ? 'solid' : 'boundary',
      change_index: i,
      changes_total: this.changes.length,
      declared_size: size,
    };
    return this.lastStats;
  }
}
