/**
 * map-loader.js — Load optimized OBJ map files with material setup.
 */

import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { mapGroup } from './scene.js';
import { extractTriangles, buildBVH } from './map-raycast.js';
import { loadMinimapRadar } from './minimap-radar.js';

/** @type {Map<string, THREE.Group>} cache of loaded maps by name */
const mapCache = new Map();

/**
 * 是否使用离线烘的材质色（`bake_map_colors.py` 写的 `<map>.colors.bin`）。
 *
 * **默认关闭**，地图就是 `setupMapMaterials()` 那套水泥灰顶点色 —— 也就是这个
 * 功能加进来之前的样子。
 *
 * 关掉的原因：那套配色是照着"带贴图和材质的完整模型"设计的，套在 Open3D 导出的
 * 简化碰撞几何上并不划算 —— 8 色板下同色面之间照样会抢深度，法线也得一起烘才不
 * 闪，而收益只是"大致色块"。代价是一张图 20 MB 旁车 + 每张图一次烘焙。
 *
 * 想再打开：URL 上加 `?mapcolors=1`（`?mapcolors=0` 显式关掉）。
 */
const BAKED_COLORS = (() => {
  try {
    const q = new URLSearchParams(location.search).get('mapcolors');
    if (q !== null) return q !== '0';
    return localStorage.getItem('mapColors') === '1';
  } catch {
    return false;   // 无 localStorage 权限的嵌入场景按默认（关）处理
  }
})();

/** @type {THREE.Group|null} currently loaded map */
let currentMap = null;

/**
 * 当前地图的 BVH 空间索引（`map-raycast.js`）。
 *
 * **懒建**：de_ancient 有 823,214 个三角形，建一次要一两秒，所以不在加载地图时
 * 建，而是等真的要射线检测了（打开瞄准射线、或者要给尸体找地面高度）再建。
 * 建成之后一直留着，换地图才清掉。
 *
 * @type {object|null}
 */
let mapBVH = null;
/** @type {Promise<object|null>|null} 建树中的 promise，防止并发重复建 */
let bvhPromise = null;

/** @type {{triCount:number, ms:number}|null} 上次建树的统计，给测试和调试面板看 */
let bvhStats = null;

/**
 * 上一张地图的配色来源，给测试和调试面板看。
 * `source` 是 `'game'`（用 `bake_map_colors.py` 烘好的游戏材质色）或
 * `'synthetic'`（没有旁车文件，退回 `setupMapMaterials()` 的水泥灰）。
 * @type {{source:'game'|'synthetic', map:string, triangles:number,
 *         expected:number, version?:number, error?:string}|null}
 */
let colorStats = null;

/** 已烘好的每三角形颜色（+ 可选法线）。见 `tools/cs2_assets/bake_map_colors.py`。 */
const COLORS_MAGIC = 0x524c434d;   // "MCLR" little endian
/** v1 = 只有颜色；v2 = 颜色 + 每面 3 字节 int8 法线。 */
const COLORS_VERSION = 2;
const COLORS_HEADER = 12;          // magic u32 + version u16 + flags u16 + count u32

/**
 * 确保当前地图的 BVH 已建好。重复调用共享同一个 promise。
 * @returns {Promise<object|null>}
 */
export function ensureMapBVH() {
  if (mapBVH) return Promise.resolve(mapBVH);
  if (bvhPromise) return bvhPromise;
  if (!currentMap) return Promise.resolve(null);

  const target = currentMap;
  bvhPromise = new Promise((resolve) => {
    // 先让出一帧，好让调用方把"正在建索引"的界面画出来
    setTimeout(() => {
      const t0 = performance.now();
      const tris = extractTriangles(target);
      const bvh = buildBVH(tris);
      const ms = performance.now() - t0;
      // 建的过程中可能已经换了地图，那就丢掉
      if (currentMap === target) {
        mapBVH = bvh;
        bvhStats = { triCount: tris ? tris.triCount : 0, ms, baked: tris ? tris.baked : false };
        console.log(`[map] BVH 建好：${bvhStats.triCount} 个三角形，${Math.round(ms)} ms，`
          + `${bvhStats.baked ? '烘焙了世界坐标' : '直接复用 OBJ 的顶点数组'}`);
      }
      bvhPromise = null;
      resolve(currentMap === target ? bvh : null);
    }, 0);
  });
  return bvhPromise;
}

/** 已经建好的 BVH；没建好返回 null（调用方自己决定要不要退化成朴素 raycast）。 */
export function getMapBVH() {
  return mapBVH;
}

/** 建树统计（三角形数 / 耗时），没建过返回 null。 */
export function getMapBVHStats() {
  return bvhStats;
}

/**
 * Load a map OBJ file and add it to the scene.
 * @param {string} mapName - e.g. "de_ancient", "de_dust2"
 * @returns {Promise<THREE.Group>}
 */
export async function loadMap(mapName) {
  // Check cache
  if (mapCache.has(mapName)) {
    const cached = mapCache.get(mapName);
    if (currentMap !== cached) {
      if (currentMap) mapGroup.remove(currentMap);
      mapGroup.add(cached);
      currentMap = cached;
      resetBVH();
    }
    return cached;
  }

  // Check if this is a known ancient-themed map that should use de_ancient
  // (for demos parsed from older versions that report different names)
  const normalizedName = normalizeMapName(mapName);

  // 小地图底图（CS2 官方雷达图）和 OBJ 并行载入；没有这批资产时静默失败，
  // 小地图退回 scene.js 里的 OBJ 俯视画法。
  loadMinimapRadar(normalizedName).catch(() => {});

  const url = `/api/map/${normalizedName}`;
  const loader = new OBJLoader();

  return new Promise((resolve, reject) => {
    loader.load(
      url,
      async (obj) => {
        setupMapMaterials(obj);
        // 烘焙材质色默认关（见 BAKED_COLORS）：关着就停在水泥灰顶点色
        if (BAKED_COLORS) await applyBakedColors(obj, normalizedName);
        mapGroup.add(obj);
        mapCache.set(mapName, obj);
        if (currentMap) mapGroup.remove(currentMap);
        currentMap = obj;
        resetBVH();
        resolve(obj);
      },
      (xhr) => {
        // Progress - could be used for a loading bar
      },
      (err) => {
        console.error(`Failed to load map "${mapName}":`, err);
        // Create a fallback ground plane
        const fallback = createFallbackMap();
        mapGroup.add(fallback);
        mapCache.set(mapName, fallback);
        if (currentMap) mapGroup.remove(currentMap);
        currentMap = fallback;
        resetBVH();
        resolve(fallback);
      }
    );
  });
}

/**
 * Normalize map names that might differ between demo parser output and file names.
 */
function normalizeMapName(name) {
  // Map common variations
  const aliases = {
    'de_ancient': 'de_ancient',
    'de_anubis': 'de_anubis',
    'de_cache': 'de_cache',
    'de_dust2': 'de_dust2',
    'de_inferno': 'de_inferno',
    'de_mirage': 'de_mirage',
    'de_nuke': 'de_nuke',
    'de_overpass': 'de_overpass',
  };
  return aliases[name] || name;
}

// ── 小地图高程配色 ─────────────────────────────────────────────────────────
/**
 * 小地图是**正射俯视**，主视图那套「水泥灰 + 光照」在俯视下会被顶光冲成一片白，
 * 完全看不出高低。所以额外烘一份只跟高度有关的顶点色（`mmColor`）：
 * 越高越灰、越低越白，像分层设色的等高线图。渲染小地图时临时把地图换成
 * 不受光照的 `minimapMaterial` + `mmColor`，主视图不受影响。
 */
const MM_LOW_RGB  = [0.98, 0.98, 0.97];   // 最低处：接近纯白
const MM_HIGH_RGB = [0.36, 0.39, 0.44];   // 最高处：中深冷灰（跟 UI 蓝灰一个色系）
const MM_LUT_SIZE = 256;

/**
 * 灰阶 LUT：sRGB → 线性工作空间。
 * three r152+ 默认开色彩管理，顶点色按**线性**解释，所以不能直接把
 * 想要的 sRGB 灰值写进 buffer，否则整体会偏亮。
 */
const MM_LUT = (() => {
  const lut = new Float32Array(MM_LUT_SIZE * 3);
  const c = new THREE.Color();
  for (let i = 0; i < MM_LUT_SIZE; i++) {
    const k = i / (MM_LUT_SIZE - 1);
    c.setRGB(
      MM_LOW_RGB[0] + (MM_HIGH_RGB[0] - MM_LOW_RGB[0]) * k,
      MM_LOW_RGB[1] + (MM_HIGH_RGB[1] - MM_LOW_RGB[1]) * k,
      MM_LOW_RGB[2] + (MM_HIGH_RGB[2] - MM_LOW_RGB[2]) * k,
      THREE.SRGBColorSpace,
    );
    lut[i * 3] = c.r;
    lut[i * 3 + 1] = c.g;
    lut[i * 3 + 2] = c.b;
  }
  return lut;
})();

/** 小地图专用材质：不参与光照，直接把顶点色画出来 */
const minimapMaterial = new THREE.MeshBasicMaterial({
  vertexColors: true,
  fog: false,
  toneMapped: false,
});

/**
 * 高程归一化范围。用 1% / 97% 分位而不是 min/max —— 地图里几个尖顶
 * （de_ancient 最高 51 m，99% 的顶点在 30 m 以下）会把可玩区域压成一小段，
 * 分位数能让灰阶落在真正有内容的区间里。
 *
 * @param {{ys: Float32Array}[]} meshes
 * @returns {{yLow: number, yHigh: number}}
 */
function computeHeightRange(meshes) {
  const BINS = 512;
  let minY = Infinity, maxY = -Infinity;
  for (const m of meshes) {
    const ys = m.ys;
    for (let i = 0; i < ys.length; i++) {
      const y = ys[i];
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!(maxY > minY)) return { yLow: minY, yHigh: minY + 1 };

  const counts = new Uint32Array(BINS);
  const scale = BINS / (maxY - minY);
  let total = 0;
  for (const m of meshes) {
    const ys = m.ys;
    for (let i = 0; i < ys.length; i++) {
      let b = ((ys[i] - minY) * scale) | 0;
      if (b < 0) b = 0; else if (b >= BINS) b = BINS - 1;
      counts[b]++;
      total++;
    }
  }
  const pick = (frac) => {
    const want = total * frac;
    let acc = 0;
    for (let b = 0; b < BINS; b++) {
      acc += counts[b];
      if (acc >= want) return minY + (b + 0.5) / scale;
    }
    return maxY;
  };
  const yLow = pick(0.01);
  const yHigh = pick(0.97);
  return yHigh > yLow ? { yLow, yHigh } : { yLow, yHigh: yLow + 1 };
}

/**
 * Set up materials on the loaded OBJ mesh — 水泥灰、哑光、有颗粒质感。
 *
 * OBJ 是单个大 mesh 且无 UV，用「顶点色 + 法线」方案：
 *  - 按法线方向区分：朝上的面（地面/平台）→ 浅水泥灰；垂直面（墙面）→ 中水泥灰
 *  - 叠加空间噪声颗粒，产生混凝土质感
 *  - roughness 高值哑光 + metalness=0，避免塑料反光
 *
 * 同时烘一份小地图用的高程灰阶顶点色（`mmColor`），见上面的注释。
 */
function setupMapMaterials(group) {
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.92,
    metalness: 0.0,
    flatShading: false,
  });

  function hash3(x, y, z) {
    let h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
    return h - Math.floor(h);
  }

  // ── 第一遍：把顶点数据抓出来（高程范围要整张图一起算，不能按 mesh 各算各的）
  const meshes = [];
  group.traverse((child) => {
    if (!child.isMesh || !child.geometry) return;
    const pos = child.geometry.attributes.position;
    const n = pos.count;
    const xs = new Float32Array(n), ys = new Float32Array(n), zs = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = pos.getX(i); ys[i] = pos.getY(i); zs[i] = pos.getZ(i);
    }
    meshes.push({ child, n, xs, ys, zs });
  });
  if (meshes.length === 0) return;

  const { yLow, yHigh } = computeHeightRange(meshes);
  const invRange = 1 / (yHigh - yLow);

  // ── 第二遍：写两套顶点色（主视图水泥灰 / 小地图高程灰阶）
  for (const m of meshes) {
    const { child, n, xs, ys, zs } = m;
    const nor = child.geometry.attributes.normal;

    const colors = new Float32Array(n * 3);
    const mmColors = new Float32Array(n * 3);

    // 主视图那套「高度微亮」仍旧按单个 mesh 自己的范围算，保持原样
    let minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const y = ys[i];
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const yRange = Math.max(0.001, maxY - minY);

    for (let i = 0; i < n; i++) {
      const x = xs[i], y = ys[i], z = zs[i];
      // 法线 y 分量：>0.5 为地面/平台，否则为墙面
      const ny = nor ? nor.getY(i) : 0;
      const isGround = ny > 0.55;

      // 水泥灰（冷灰、微带蓝相）：
      // 地面/平台：浅水泥灰；墙面：中水泥灰（层次区分）
      let r, g, b;
      if (isGround) {
        r = 0.66; g = 0.68; b = 0.70;   // 浅水泥灰
      } else {
        r = 0.56; g = 0.58; b = 0.61;   // 中水泥灰
      }

      // 高度：高处略亮、低处略暗（自然光照感）
      const hF = (y - minY) / yRange;
      const heightShade = 0.88 + hF * 0.12;

      // 空间噪声颗粒（低频、柔和，避免表面发脏）+ 大尺度色块
      const noise = hash3(Math.floor(x * 1.2), Math.floor(y * 1.2), Math.floor(z * 1.2)) * 2 - 1;
      const block = hash3(Math.floor(x * 0.3), Math.floor(y * 0.3), Math.floor(z * 0.3)) * 2 - 1;
      const jitter = heightShade * (1.0 + noise * 0.03 + block * 0.03);

      colors[i * 3] = Math.min(0.9, Math.max(0.25, r * jitter));
      colors[i * 3 + 1] = Math.min(0.9, Math.max(0.25, g * jitter));
      colors[i * 3 + 2] = Math.min(0.9, Math.max(0.25, b * jitter));

      // 小地图高程灰阶：越低越白、越高越灰
      let t = (y - yLow) * invRange;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const li = ((t * (MM_LUT_SIZE - 1)) + 0.5) | 0;
      mmColors[i * 3] = MM_LUT[li * 3];
      mmColors[i * 3 + 1] = MM_LUT[li * 3 + 1];
      mmColors[i * 3 + 2] = MM_LUT[li * 3 + 2];
    }

    child.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    child.geometry.setAttribute('mmColor', new THREE.BufferAttribute(mmColors, 3));
    child.material = mat;
    child.castShadow = true;
    child.receiveShadow = true;
    // 计算平滑法线（OBJ 文件无 vn，默认 flat shading 会让表面出现棱线/显得糊）
    child.geometry.computeVertexNormals();
  }
}

// ── 游戏材质颜色（每三角形一个）────────────────────────────────────────────
/**
 * OBJ 是 Open3D 导出的**纯几何**（`vt` / `usemtl` / `vn` 全是 0 条），材质归属
 * 在导出那一步就丢了，所以 `setupMapMaterials()` 只能按法线糊一层水泥灰。
 *
 * 真颜色由 `tools/cs2_assets/bake_map_colors.py` 离线烘出来：从 CS2 地图 VPK
 * 的 `worldnodes/` 导出带 UV + 材质的原始几何，给 OBJ 的每个三角形找最近的原始
 * 三角形，继承它的材质、在材质的 base color 贴图上按 UV 采一个色。
 *
 * 产物是同目录的 `<map>.colors.bin`：12 字节头 + 每面 3 字节 sRGB（v1），
 * v2 再追加每面 3 字节 int8 法线。三角形顺序就是 OBJ 里 `f` 行的顺序，OBJLoader
 * 展开成非索引几何后正好是每 3 个顶点一个面。
 *
 * 没有这个文件、或者面数对不上（OBJ 换过版本），就静默退回水泥灰 —— 地图照常能看。
 */
const SRGB_TO_LINEAR = (() => {
  // 顶点色按**线性**解释（renderer 开了 ColorManagement），而旁车文件里存的是
  // sRGB。330 万个面各 new 一个 THREE.Color 太慢，按通道查表。
  const lut = new Float32Array(256);
  const c = new THREE.Color();
  for (let i = 0; i < 256; i++) {
    c.setRGB(i / 255, i / 255, i / 255, THREE.SRGBColorSpace);
    lut[i] = c.r;
  }
  return lut;
})();

/**
 * 把烘好的颜色写进地图的 `color` 顶点色属性（覆盖掉水泥灰）。
 * @param {THREE.Object3D} group OBJLoader 的产物
 * @param {string} mapName 规范化后的地图名
 */
async function applyBakedColors(group, mapName) {
  if (!BAKED_COLORS) return;      // 默认关，见文件头的 BAKED_COLORS
  const meshes = [];
  let expected = 0;
  group.traverse((child) => {
    if (!child.isMesh || !child.geometry) return;
    const attr = child.geometry.attributes.color;
    if (!attr) return;
    const tris = child.geometry.attributes.position.count / 3;
    meshes.push({ child, tris });
    expected += tris;
  });
  colorStats = { source: 'synthetic', map: mapName, triangles: 0, expected, reason: 'pending' };

  let buf;
  try {
    const res = await fetch(`/api/map/${encodeURIComponent(mapName)}/colors`);
    if (!res.ok) {
      // 404 = 这张图还没烘，属于正常情况，不吵
      colorStats.reason = `http-${res.status}`;
      if (res.status !== 404) console.warn(`[map] colours for ${mapName}: HTTP ${res.status}`);
      return;
    }
    buf = await res.arrayBuffer();
  } catch (err) {
    colorStats.reason = 'fetch-failed';
    console.warn(`[map] colours for ${mapName} failed:`, err);
    return;
  }

  if (buf.byteLength < COLORS_HEADER) {
    colorStats.reason = `truncated-${buf.byteLength}`;
    console.warn(`[map] colours for ${mapName}: truncated (${buf.byteLength} bytes)`);
    return;
  }
  const dv = new DataView(buf);
  const magic = dv.getUint32(0, true);
  const version = dv.getUint16(4, true);
  const palette = dv.getUint16(6, true);     // flags：烘的时候压成了几种颜色（0 = 没压）
  const count = dv.getUint32(8, true);
  if (magic !== COLORS_MAGIC || version < 1 || version > COLORS_VERSION) {
    colorStats.reason = `bad-header-${magic}-${version}`;
    console.warn(`[map] colours for ${mapName}: bad header (magic=${magic}, v=${version})`);
    return;
  }
  // v1 = 只有颜色（3 字节/面）；v2 = 颜色 + 平滑法线（再多 3 字节/面）
  const withNormals = version >= 2 && buf.byteLength === COLORS_HEADER + count * 6;
  const wantBytes = COLORS_HEADER + count * (withNormals ? 6 : 3);
  if (count !== expected || buf.byteLength !== wantBytes) {
    // OBJ 换过版本（重新减面/改过几何），旁车文件过期了：宁可不涂也不要涂错位
    colorStats.reason = `stale-${count}-vs-${expected}`;
    console.warn(`[map] colours for ${mapName} stale: ${count} tris in file, ${expected} in mesh`);
    return;
  }

  const src = new Uint8Array(buf, COLORS_HEADER, count * 3);
  const srcN = withNormals
    ? new Int8Array(buf, COLORS_HEADER + count * 3, count * 3)
    : null;
  const t0 = performance.now();
  let tri = 0;
  for (const { child, tris } of meshes) {
    const colors = child.geometry.attributes.color.array;
    // 平滑法线：OBJ 没有 vn，`computeVertexNormals()` 只能给逐面法线 —— 一块平地上
    // 几千个三角形明暗各不相同，远看就是"一片碎三角形"。旁车里的法线是离线按顶点
    // 平均过（带 60° 折角保护）的，写进去之后平地就是一个颜色。
    const normals = srcN ? child.geometry.attributes.normal : null;
    for (let t = 0; t < tris; t++, tri++) {
      const s = tri * 3;
      const r = SRGB_TO_LINEAR[src[s]];
      const g = SRGB_TO_LINEAR[src[s + 1]];
      const b = SRGB_TO_LINEAR[src[s + 2]];
      const v = t * 9;                       // 每个面 3 个顶点，每顶点 3 个分量
      colors[v] = r; colors[v + 1] = g; colors[v + 2] = b;
      colors[v + 3] = r; colors[v + 4] = g; colors[v + 5] = b;
      colors[v + 6] = r; colors[v + 7] = g; colors[v + 8] = b;
      if (normals) {
        const nx = srcN[s] / 127;
        const ny = srcN[s + 1] / 127;
        const nz = srcN[s + 2] / 127;
        const arr = normals.array;
        arr[v] = nx; arr[v + 1] = ny; arr[v + 2] = nz;
        arr[v + 3] = nx; arr[v + 4] = ny; arr[v + 5] = nz;
        arr[v + 6] = nx; arr[v + 7] = ny; arr[v + 8] = nz;
      }
    }
    child.geometry.attributes.color.needsUpdate = true;
    if (normals) normals.needsUpdate = true;
  }
  colorStats = {
    source: 'game',
    map: mapName,
    triangles: count,
    expected,
    version,
    normals: withNormals,
    palette,
    ms: Math.round(performance.now() - t0),
  };
  console.log(`[map] 游戏材质色：${count} 个三角形`
    + `${withNormals ? ' + 平滑法线' : ''}`
    + `${palette ? `，${palette} 色调色板` : ''}，${colorStats.ms} ms`);
}

/** 上一张地图的配色来源（测试和调试面板用）。 */
export function getMapColorStats() {
  if (!colorStats) return { source: 'synthetic', reason: BAKED_COLORS ? 'idle' : 'disabled', enabled: BAKED_COLORS };
  return colorStats;
}

/**
 * 把配色换成另一个旁车文件，用的还是**同一张地图的几何**。
 *
 * 换调色板数量的时候用得上：整页重载要重新解析 130 MB 的 OBJ，而这里只是重涂一遍
 * 顶点色（几十毫秒）。`colorMapName` 是 URL 里的那一段，所以把不同调色板的文件
 * 存成 `de_inferno_p8.colors.bin` 之类的名字就能直接比。
 *
 * @param {string} colorMapName 例如 `de_inferno` 或 `de_inferno_p8`
 * @returns {Promise<object|null>} `getMapColorStats()` 的结果
 */
export async function reloadMapColors(colorMapName) {
  if (!BAKED_COLORS) {
    console.warn('[map] 烘焙材质色已关闭（URL 加 ?mapcolors=1 打开），忽略这次重涂');
    return getMapColorStats();
  }
  if (!currentMap) return null;
  await applyBakedColors(currentMap, colorMapName);
  return colorStats;
}

// ── 小地图材质切换 ─────────────────────────────────────────────────────────
/** @type {{mesh: THREE.Mesh, material: THREE.Material, color: THREE.BufferAttribute|undefined}[]} */
const mmSwapped = [];

/**
 * 把地图临时切成小地图配色（不受光照的高程灰阶）。
 * 渲染完必须调 `restoreMapStyle()`，否则主视图也会变成灰阶。
 */
export function applyMinimapMapStyle() {
  restoreMapStyle(); // 幂等：万一上一次没收干净
  mapGroup.traverse((child) => {
    if (!child.isMesh || !child.geometry) return;
    const mmColor = child.geometry.attributes.mmColor;
    if (!mmColor) return;
    mmSwapped.push({
      mesh: child,
      material: child.material,
      color: child.geometry.attributes.color,
    });
    child.geometry.setAttribute('color', mmColor);
    child.material = minimapMaterial;
  });
}

/** 恢复主视图材质（切回来的只是属性指针，不重新上传顶点数据） */
export function restoreMapStyle() {
  for (const s of mmSwapped) {
    s.mesh.geometry.setAttribute('color', s.color);
    s.mesh.material = s.material;
  }
  mmSwapped.length = 0;
}

/**
 * Create a simple fallback "map" if OBJ loading fails.
 */
function createFallbackMap() {
  const group = new THREE.Group();
  const planeGeo = new THREE.PlaneGeometry(80, 80);
  const planeMat = new THREE.MeshStandardMaterial({
    color: 0x9aa4ad,
    roughness: 0.92,
    metalness: 0.0,
  });
  const plane = new THREE.Mesh(planeGeo, planeMat);
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = -0.1;
  plane.receiveShadow = true;
  group.add(plane);

  // Grid
  const grid = new THREE.GridHelper(80, 80, 0x7f8f9e, 0xa8b4c0);
  grid.position.y = 0;
  group.add(grid);

  return group;
}

/**
 * 丢掉当前地图的 BVH。换地图时必须调用 —— BVH 里存的是那张图的三角形。
 */
function resetBVH() {
  mapBVH = null;
  bvhStats = null;
  bvhPromise = null;
}

/**
 * Remove the current map from the scene.
 */
export function clearMap() {
  if (currentMap) {
    mapGroup.remove(currentMap);
    currentMap = null;
    resetBVH();
  }
}

/**
 * Get map bounding box for camera fitting.
 * @returns {THREE.Box3|null}
 */
export function getMapBounds() {
  if (!currentMap) return null;
  const box = new THREE.Box3();
  // Traverse to find all meshes (skip wireframe group)
  currentMap.traverse((child) => {
    if (child.isMesh && child.geometry) {
      child.geometry.computeBoundingBox();
      const childBox = child.geometry.boundingBox.clone();
      childBox.applyMatrix4(child.matrixWorld);
      box.expandByPoint(childBox.min);
      box.expandByPoint(childBox.max);
    }
  });
  return box.isEmpty() ? null : box;
}
