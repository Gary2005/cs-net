/**
 * minimap-radar.js — CS2 官方雷达图当小地图底图。
 *
 * 图和小地图的「世界坐标 → 像素」变换都来自游戏本身，不是拟合出来的：
 *
 *   visualizer/static/assets/radar/<map>.png      panorama/images/overheadmaps/<map>_radar_psd
 *   visualizer/static/assets/radar/map.json       resource/overviews/<map>.txt 里的 pos_x/pos_y/scale
 *
 * 变换（和游戏内雷达同一个式子）：
 *
 *   col = (x - pos_x) / scale
 *   row = (pos_y - y) / scale
 *
 * 也就是图左上角对应世界坐标 (pos_x, pos_y)，每个像素宽 scale 个游戏单位，+y 朝上。
 * 所以玩家标记和图上的房间、楼梯天然对得上，不需要再去调偏移。
 *
 * 这份数据由 `tools/cs2_assets/install_web_assets.py --radars` 生成，和
 * `app/radar.py` 里 Demo 卡缩略图用的是同一批图、同一份 map.json。
 *
 * 关于 `rotate`：de_dust2 的 overview 里写着 `"rotate" "1"`，但导出的 vtex 已经是
 * 世界朝向的，再转 90° 反而错位 —— 用 mirage 的 100 万条真实玩家轨迹验证过朝向
 * （0° 命中 99.6%），dust2 用 100 万几何顶点验证同样是 0° 最优。map.json 不存这个
 * 字段，就是这个原因。
 */

import * as THREE from 'three';

const BASE = '/static/assets/radar';

/** @type {Promise<object>|null} map.json（只取一次） */
let tablePromise = null;

/** @type {Map<string, Promise<object|null>>} 每张图的纹理，按地图名缓存 */
const textureCache = new Map();

/** @type {object|null} 当前地图的雷达数据 */
let current = null;

/** 最近一次请求的地图名：换图时旧请求可能晚到，不能让它覆盖新图 */
let requested = null;

const SCALE = 0.0254;   // 游戏单位 -> three.js，和 scene.js 保持一致

/**
 * 读 `radar/map.json`。文件不存在时返回空表 —— 那说明这批资产没抽（干净克隆的
 * 老版本），小地图退回原来的 OBJ 俯视画法。
 * @returns {Promise<object>}
 */
export function loadRadarTable() {
  if (!tablePromise) {
    tablePromise = fetch(`${BASE}/map.json`)
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({}));
  }
  return tablePromise;
}

/**
 * 载入一张地图的雷达图。
 *
 * 返回的对象同时给出像素尺寸和世界矩形，`scene.js` 据此把图贴成一块平面：
 *
 *   u0 = pos_x                 u1 = pos_x + width  * scale   (世界 x 的范围)
 *   v1 = pos_y                 v0 = pos_y - height * scale   (世界 y 的范围)
 *
 * @param {string} mapName
 * @returns {Promise<object|null>}
 */
export async function loadMinimapRadar(mapName) {
  requested = mapName;
  const table = await loadRadarTable();
  const info = table[mapName];
  if (!info) {
    if (requested === mapName) current = null;
    return null;
  }

  let pending = textureCache.get(mapName);
  if (!pending) {
    const loader = new THREE.TextureLoader();
    pending = (async () => {
      const texture = await loader.loadAsync(`${BASE}/${info.image}`);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 4;
      texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.magFilter = THREE.LinearFilter;
      let lower = null;
      if (info.lower) {
        lower = await loader.loadAsync(`${BASE}/${info.lower}`);
        lower.colorSpace = THREE.SRGBColorSpace;
        lower.anisotropy = 4;
        lower.minFilter = THREE.LinearMipmapLinearFilter;
        lower.magFilter = THREE.LinearFilter;
      }
      return { texture, lower };
    })().catch(() => null);
    textureCache.set(mapName, pending);
  }

  const tex = await pending;
  if (requested !== mapName) return null;      // 期间换了图，这次结果作废
  if (!tex) {
    current = null;
    return null;
  }

  const u0 = info.pos_x;
  const v1 = info.pos_y;
  current = {
    map: mapName,
    texture: tex.texture,
    lowerTexture: tex.lower,
    lowerBelowZ: typeof info.lower_below_z === 'number' ? info.lower_below_z : null,
    scale: info.scale,
    u0,
    u1: u0 + info.width * info.scale,
    v0: v1 - info.height * info.scale,
    v1,
  };
  return current;
}

/** 当前已载入的雷达数据；没载入返回 null（调用方走 OBJ 兜底）。 */
export function currentMinimapRadar() {
  return current;
}

/**
 * 世界矩形 -> three.js 里那块平面的四个角。
 *
 * 三个轴是置换过的（`three = (gy, gz, gx) * SCALE`）：图像列 u 对应 +Z，
 * 行 v 向下对应 -X，所以图上的「上」是 +X、「右」是 +Z。
 *
 * @param {object} radar
 * @returns {{xTop:number, xBottom:number, zLeft:number, zRight:number}}
 */
export function radarRect(radar) {
  return {
    xTop: radar.v1 * SCALE,
    xBottom: radar.v0 * SCALE,
    zLeft: radar.u0 * SCALE,
    zRight: radar.u1 * SCALE,
  };
}
