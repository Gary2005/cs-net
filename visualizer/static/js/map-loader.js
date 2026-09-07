/**
 * map-loader.js — Load optimized OBJ map files with material setup.
 */

import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { mapGroup } from './scene.js';

/** @type {Map<string, THREE.Group>} cache of loaded maps by name */
const mapCache = new Map();

/** @type {THREE.Group|null} currently loaded map */
let currentMap = null;

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
    }
    return cached;
  }

  // Check if this is a known ancient-themed map that should use de_ancient
  // (for demos parsed from older versions that report different names)
  const normalizedName = normalizeMapName(mapName);

  const url = `/api/map/${normalizedName}`;
  const loader = new OBJLoader();

  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (obj) => {
        setupMapMaterials(obj);
        mapGroup.add(obj);
        mapCache.set(mapName, obj);
        if (currentMap) mapGroup.remove(currentMap);
        currentMap = obj;
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

/**
 * Set up materials on the loaded OBJ mesh — 水泥灰、哑光、有颗粒质感。
 *
 * OBJ 是单个大 mesh 且无 UV，用「顶点色 + 法线」方案：
 *  - 按法线方向区分：朝上的面（地面/平台）→ 浅水泥灰；垂直面（墙面）→ 中水泥灰
 *  - 叠加空间噪声颗粒，产生混凝土质感
 *  - roughness 高值哑光 + metalness=0，避免塑料反光
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

  group.traverse((child) => {
    if (child.isMesh && child.geometry) {
      const pos = child.geometry.attributes.position;
      const nor = child.geometry.attributes.normal;
      const n = pos.count;

      const colors = new Float32Array(n * 3);
      child.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const xs = new Float32Array(n), ys = new Float32Array(n), zs = new Float32Array(n);
      let minY = Infinity, maxY = -Infinity;
      for (let i = 0; i < n; i++) {
        const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        xs[i] = x; ys[i] = y; zs[i] = z;
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
      }

      child.material = mat;
      child.castShadow = true;
      child.receiveShadow = true;
      // 计算平滑法线（OBJ 文件无 vn，默认 flat shading 会让表面出现棱线/显得糊）
      if (child.geometry) {
        child.geometry.computeVertexNormals();
      }
    }
  });
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
 * Remove the current map from the scene.
 */
export function clearMap() {
  if (currentMap) {
    mapGroup.remove(currentMap);
    currentMap = null;
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
