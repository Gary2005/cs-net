/**
 * cs2-assets.js — 从游戏文件提取的真实 CS2 资产的浏览器端加载器。
 *
 * 这里只管"取和缓存"，不管怎么画：manifest / 图标 / sprite 序列 / 武器 GLB。
 * 渲染分别落在 `visuals.js`（图标、烟、火）和 `weapon-models.js`（3D 模型）。
 *
 * ## 为什么要有 manifest
 *
 * VRF 拆 sprite sheet 时把**帧顺序和 LOOP 标记**写在 `.mks` 边车文件里，
 * **不在文件名里**。而且一个贴图可能有多条序列（`fire_onsurface` 有 3 条，
 * 共 330 帧）。所以帧列表必须从 manifest 读，**绝不能按文件名排序猜**。
 *
 * manifest 由 `tools/cs2_assets/install_web_assets.py` 生成。像素本身是
 * Valve 版权（个人非商业使用），和 manifest 一起**入库**了 —— 代价和它带来的
 * 约束（这个仓库必须保持私有）写在 `.gitignore` 末尾。
 *
 * ## 关于贴图色彩空间
 *
 * **法线贴图必须按线性空间采样。** 三.js 的 `TextureLoader` 默认
 * `NoColorSpace`（线性），颜色贴图要**显式**设 `SRGBColorSpace`。这里按
 * 用途区分：颜色图用 sRGB，法线/ORM 保持线性 —— 反了的话法线会整体偏色。
 */

import * as THREE from 'three';

const ASSET_ROOT = '/static/assets';
const MANIFEST_URL = `${ASSET_ROOT}/manifest.json`;

let _manifest = null;
let _manifestPromise = null;

/** 取（并缓存）资产清单。失败时返回 null，调用方应降级而不是崩。 */
export async function loadManifest() {
  if (_manifest) return _manifest;
  if (!_manifestPromise) {
    // **不要用 `force-cache`。** 那个模式会连"已过期"的缓存也直接用、不做任何
    // 重新校验，于是重新提取资产后刷新页面仍然拿到旧的 manifest.json ——
    // 表现就是"改了代码/重跑了提取，但界面毫无变化"，极难排查。
    // `no-cache` 仍然会缓存，但每次带 ETag 去校验；Flask 对没变的文件回 304，
    // 开销可以忽略。
    _manifestPromise = fetch(MANIFEST_URL, { cache: 'no-cache' })
      .then((r) => {
        if (!r.ok) throw new Error(`manifest HTTP ${r.status}`);
        return r.json();
      })
      .then((m) => { _manifest = m; return m; })
      .catch((e) => {
        console.warn('[cs2-assets] 资产清单不可用，真实资产功能将降级：', e.message);
        return null;
      });
  }
  return _manifestPromise;
}

/** 同步读已加载的清单（没加载好就返回 null）。 */
export function manifest() { return _manifest; }

// ── 武器 HUD 图标 ──────────────────────────────────────────────────────────

/** display name -> 图标文件名（如 "AK-47" -> "weapon_ak47_png.png"） */
let _iconMap = null;
/** 内部 key（`ak47`、`knife_skeleton`）-> 图标文件名。demoparser2 有时给内部名。 */
let _iconByKey = null;
/** 文件名 -> THREE.Texture */
const _iconTex = new Map();

/** `weapon_ak47_png.png` → `ak47`（和 weapon-index.js 的 iconFileToKey 同一规则）。 */
function iconFileKey(file) {
  return String(file || '').replace(/^weapon_/, '').replace(/_png\.png$/, '');
}

async function ensureIconMap() {
  if (_iconMap) return _iconMap;
  const m = await loadManifest();
  if (!m || !m.weapon_icons || !m.weapon_icons.map) {
    _iconMap = {}; _iconByKey = {}; return _iconMap;
  }
  try {
    const r = await fetch(`${ASSET_ROOT}/${m.weapon_icons.map}`, { cache: 'no-cache' });
    _iconMap = r.ok ? await r.json() : {};
  } catch {
    _iconMap = {};
  }
  // 反向索引：允许用内部名查图标（见 weapon-index.js 里同样的处理）
  _iconByKey = {};
  for (const [display, files] of Object.entries(_iconMap)) {
    const f = Array.isArray(files) ? files[0] : files;
    if (!f) continue;
    const k = iconFileKey(f);
    if (!(k in _iconByKey)) _iconByKey[k] = f;
    if (!(display in _iconByKey)) _iconByKey[display] = f;
  }
  return _iconMap;
}

/** 显示名（或内部名）→ 图标文件名，找不到返回 null。 */
function iconFileFor(name) {
  const entry = _iconMap[name];
  const file = Array.isArray(entry) ? entry[0] : entry;
  if (file) return file;
  return _iconByKey[name] || null;
}

/**
 * 按**显示名**取武器图标贴图（demo 里的 `active_weapon_name` 就是显示名）。
 * 没有对应图标时返回 null —— 调用方退回原来的文字标签。
 *
 * 图标是 512×384 的**非正方**图，别当正方形贴图用。
 */
export async function weaponIconTexture(displayName) {
  if (!displayName) return null;
  await ensureIconMap();
  // 映射表里一个显示名可能对应多个图标（如 "Desert Eagle"），取第一个即可
  const file = iconFileFor(displayName);
  if (!file) return null;
  if (_iconTex.has(file)) return _iconTex.get(file);

  const tex = await new Promise((resolve) => {
    new THREE.TextureLoader().load(
      `${ASSET_ROOT}/weapon_icons/${file}`,
      (t) => resolve(t),
      undefined,
      () => resolve(null),
    );
  });
  if (tex) {
    tex.colorSpace = THREE.SRGBColorSpace;   // 图标是颜色图
    tex.anisotropy = 4;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
  }
  _iconTex.set(file, tex);
  return tex;
}

/**
 * 按**显示名**取图标文件的 URL（给 DOM 用，如击杀信息的 `<img>`）。
 *
 * `weaponIconTexture` 返回的是 `THREE.Texture`，只能给 WebGL 用，塞进 `<img src>`
 * 会得到 `[object Object]`。所以单独开一个。
 *
 * 映射是**同步**的，但 icon map 要先异步加载；没加载好时返回 null，
 * 调用方退回纯文字。加载完成后 `onReady` 会被调一次，让已经画出来的条目补图。
 */
export function weaponIconUrl(displayName, onReady) {
  if (!displayName) return null;
  if (_iconMap) {
    const file = iconFileFor(displayName);
    return file ? `${ASSET_ROOT}/weapon_icons/${file}` : null;
  }
  // 首次调用：拉起映射，好了再回调一次
  ensureIconMap().then(() => { if (onReady) onReady(weaponIconUrl(displayName)); });
  return null;
}

// ── sprite 序列帧（烟雾 / 火焰）────────────────────────────────────────────

/**
 * 一个 sprite 序列的贴图数组 + 播放信息。
 * @typedef {{
 *   name: string, loop: boolean,
 *   textures: THREE.Texture[],
 *   width: number, height: number,
 * }} SpriteSequence
 */

/** name -> Promise<SpriteSequence|null>，避免同一序列并发加载两次 */
const _seqPromises = new Map();

/**
 * 加载一个 sprite 序列（manifest 里的 key，如 `smokeloop_i_0_sc_hardedge`）。
 *
 * 多条序列（`sequences[]`）会被**首尾相接**成一条时间线 —— VRF 按
 * `sequence N` 拆开只是文件组织方式，视觉上它们是连续动画。
 */
export function loadSpriteSequence(name) {
  if (_seqPromises.has(name)) return _seqPromises.get(name);

  const p = (async () => {
    const m = await loadManifest();
    const info = m && m.sprites && m.sprites[name];
    if (!info) {
      console.warn(`[cs2-assets] 清单里没有 sprite 序列 ${name}`);
      return null;
    }

    // 把多条序列拉平成一个帧列表，同时记住每条自己的 loop 标记
    const frames = [];
    let anyLoop = false;
    for (const s of info.sequences || []) {
      if (s.loop) anyLoop = true;
      for (const f of s.frames || []) frames.push(f.file);
    }
    if (!frames.length) return null;

    const loader = new THREE.TextureLoader();
    const dir = `${ASSET_ROOT}/sprites/${name}`;
    const textures = await Promise.all(frames.map((file) => new Promise((resolve) => {
      loader.load(
        `${dir}/${file}`,
        (t) => {
          // sprite 是带 alpha 的颜色图 → sRGB
          t.colorSpace = THREE.SRGBColorSpace;
          t.minFilter = THREE.LinearFilter;      // 帧间不插值，避免糊
          t.magFilter = THREE.LinearFilter;
          t.generateMipmaps = false;
          t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
          resolve(t);
        },
        undefined,
        () => resolve(null),
      );
    })));

    const kept = textures.filter(Boolean);
    if (!kept.length) return null;
    const img = kept[0].image || {};
    return {
      name,
      loop: anyLoop,
      textures: kept,
      width: img.width || 0,
      height: img.height || 0,
    };
  })();

  _seqPromises.set(name, p);
  return p;
}

/** 预热若干序列（不阻塞调用方）。 */
export function preloadSpriteSequences(names) {
  for (const n of names) loadSpriteSequence(n).catch(() => {});
}

/** 已加载的序列数（给"数据检查"面板用）。 */
export function assetDebugStats() {
  return {
    manifest: !!_manifest,
    icon_names: _iconMap ? Object.keys(_iconMap).length : 0,
    icons_loaded: _iconTex.size,
    sequences: _seqPromises.size,
  };
}

/** 清空所有缓存（换地图 / 测试用）。 */
export function clearAssetCaches() {
  for (const t of _iconTex.values()) t?.dispose();
  _iconTex.clear();
  for (const p of _seqPromises.values()) {
    p.then((s) => { if (s) for (const t of s.textures) t.dispose(); }).catch(() => {});
  }
  _seqPromises.clear();
  _iconMap = null;
  _iconByKey = null;
}
