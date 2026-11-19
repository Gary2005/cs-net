/**
 * vis_common.mjs — `vis_offline.mjs`（批量出全表）和 `vis_oracle.mjs`（常驻单局面查询）
 * 共用的零件：加载 app 的 JS 模块、读地图 OBJ 建 BVH、烟雾遮挡体。
 *
 * 这里**没有一行几何代码** —— 判定全部来自 `visualizer/static/js/`，和浏览器里跑的是
 * 同一份 `visibility.js`。这个文件只负责"把 app 那套东西在 node 里搭起来"。见 README §3.9。
 *
 * 唯一一件慢事是 `loadMap()`：mirage 122 万三角形实测 **解析 0.7 s + 建树 0.3 s**。
 * 所以**常驻进程要把它做一次就留着** —— 这正是 `vis_oracle.mjs` 存在的理由。
 */

import { readFileSync } from 'node:fs';

export const JS_DIR = new URL('../../../visualizer/static/js/', import.meta.url);
export const load = (name) => import(new URL(name, JS_DIR).href);

export const core = await load('replay-core.js');
export const vis = await load('visibility.js');
export const { buildBVH } = await load('map-raycast.js');
export const { SmokeVoxelCache, VOXEL_GRID_DIM, VOXEL_WORLD_SIZE } = await load('smoke-voxels.js');

export const {
  decodeRoundPayload, setRound, seekTo, getAllPlayerStates,
  STANDING_EYE_HEIGHT, CROUCH_EYE_DROP, eyeHeightFor, hullHeightFor,
} = core;

/**
 * `scene.js` 的 SCALE。**不 import** —— scene.js 建 WebGLRenderer，node 里没有 document。
 * 改成从源码里抠出来，这样它变了我们会当场报错，而不是静默用错的比例尺。
 */
export function readScale() {
  const src = readFileSync(new URL('scene.js', JS_DIR), 'utf8');
  const m = /export\s+const\s+SCALE\s*=\s*([0-9][0-9.eE+-]*)/.exec(src);
  if (!m) throw new Error('scene.js 里找不到 `export const SCALE = <数字>`');
  return Number(m[1]);
}

/**
 * OBJ → 三角形汤 `{verts: Float32Array(9/tri), triCount}`（`buildBVH` 要的格式）。
 *
 * 解析规则**照抄** `app/radar.py::_read_obj`：只有 `v ` 和 `f ` 两种行，`f` 的每项可能是
 * `a` / `a/b` / `a/b/c`，多边形按扇形三角化，解析不了的行整行丢。顶点**不做任何变换**
 * —— 这些 obj 已经在 three 空间了（radar.py 是除以 SCALE 再置换回来的），
 * `makeMapCaster()` 负责转换。
 *
 * 单遍收集 + 展开时按最终顶点数过滤，避免"先数行数再填"那种数错就静默截断的写法
 * （`Int32Array` 越界写是**静默丢弃**的）。
 */
export function readObjSoup(path) {
  const lines = readFileSync(path, 'utf8').split('\n');
  const v = [];   // 顶点，x,y,z 平铺
  const f = [];   // 三角形顶点下标，3 个一组
  for (const ln of lines) {
    const c0 = ln.charCodeAt(0);
    if (c0 === 118 /* v */ && ln.charCodeAt(1) === 32) {
      const p = ln.slice(2).split(' ');
      v.push(+p[0], +p[1], +p[2]);
    } else if (c0 === 102 /* f */ && ln.charCodeAt(1) === 32) {
      const idx = [];
      for (const tok of ln.slice(2).split(' ')) {
        const s = tok.indexOf('/');
        const i = parseInt(s < 0 ? tok : tok.slice(0, s), 10) - 1;
        if (!(i >= 0)) { idx.length = 0; break; }
        idx.push(i);
      }
      for (let k = 1; k + 1 < idx.length; k++) f.push(idx[0], idx[k], idx[k + 1]);
    }
  }
  const nv = v.length / 3;
  if (!nv) throw new Error(`OBJ 里没有顶点：${path}`);

  const ok = (t) => f[t] < nv && f[t + 1] < nv && f[t + 2] < nv;
  let triCount = 0;
  for (let t = 0; t < f.length; t += 3) if (ok(t)) triCount++;

  const soup = new Float32Array(triCount * 9);
  let at = 0;
  for (let t = 0; t < f.length; t += 3) {
    if (!ok(t)) continue;
    for (let k = 0; k < 3; k++) {
      const s = f[t + k] * 3;
      soup[at++] = v[s]; soup[at++] = v[s + 1]; soup[at++] = v[s + 2];
    }
  }
  return { verts: soup, triCount, vertexCount: nv };
}

/**
 * 读 OBJ → 建 BVH → 包成 game 空间的 `castMapRay`。**这个进程里最贵的一次调用。**
 *
 * @returns {{objPath:string, tris:number, scale:number, castMapRay:Function,
 *            parse_ms:number, build_ms:number}}
 */
export function loadMap(objPath) {
  const t0 = Date.now();
  const soup = readObjSoup(objPath);
  const tParse = Date.now() - t0;
  const bvh = buildBVH(soup);
  if (!bvh) throw new Error(`buildBVH 返回 null：${objPath}`);
  const scale = readScale();
  return {
    objPath,
    tris: soup.triCount,
    scale,
    castMapRay: vis.makeMapCaster(bvh, scale),
    parse_ms: tParse,
    build_ms: Date.now() - t0 - tParse,
  };
}

/**
 * `visuals.js::smokeOccludersAt()` 的等价物（那边 import three，node 里跑不起来）。
 *
 * 逻辑逐条对齐：按 `first_tick/last_tick` 过滤 → 没有 `detonation` 就丢（放不进世界坐标，
 * 画到原点比不画更糟）→ `cache.at(tick)`，空/0 个体素就丢。
 *
 * ⚠ `cache` 要**跨 tick 复用**（app 里按 entity_id 缓存）：journal 折叠是增量的，
 * 每帧新建会退化成每帧从头重折。
 */
export function makeSmokeCaches(instances, solid) {
  const byId = new Map();
  for (const inst of instances || []) {
    if (!inst.detonation) continue;
    byId.set(inst.entity_id, new SmokeVoxelCache(inst, { solid }));
  }
  return byId;
}

export function smokeOccluders(caches, instances, tick) {
  const out = [];
  for (const inst of instances || []) {
    if (tick < inst.first_tick || tick > inst.last_tick) continue;
    const cache = caches.get(inst.entity_id);
    if (!cache) continue;
    const geo = cache.at(tick);
    if (!geo || geo.count === 0) continue;
    out.push({ det: inst.detonation, grid: geo.grid, dim: VOXEL_GRID_DIM, cell: VOXEL_WORLD_SIZE });
  }
  return out;
}
