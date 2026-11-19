/**
 * weapon-models.js — 真实武器 3D 模型（从游戏文件提取的 GLB）。
 *
 * ## 模型是怎么来的
 *
 * `Source2Viewer-CLI` 把 `.vmdl_c` 导成 GLB，贴图**以裸文件名**引用、和 GLB
 * 同目录，所以一个武器是一个目录（`assets/weapons/ak47/` 里同时有 .glb 和 .webp）。
 * `离线资产管线` 把贴图从 4096² 缩到 1024² 并把 GLB
 * 里的 uri 改写成 `.webp`（167 MB → 24 MB）。
 *
 * ## 两个必须知道的限制
 *
 * 1. **没有骨骼动画。** 武器 GLB 的 primitive 带 `JOINTS_0`/`WEIGHTS_0` 但**没有
 *    `skins` 数组**（glTF 规范里非法）。实测这些是**刚性绑定**（每顶点 1 个 joint、
 *    权重和 1.0），three.js 会**静默丢弃**蒙皮属性、当刚性 Mesh 渲染在 bind pose。
 *    不报错、能直接用，但**不会有任何骨骼动画**。
 * 2. **朝向不统一。** 长枪的长轴在 **+Z**（AK 0.96 m、AWP 1.39 m），但刀和投掷物的
 *    长轴在 **+Y**（karambit 0.25 m、molotov 0.35 m）。所以按类别给不同的挂载旋转，
 *    见 `CLASS_ORIENT`。尺寸本身就是**米**，不需要缩放。
 *
 * ## 挂点
 *
 * 挂在右手位置（玩家模型里手臂在 x=±0.30、y=1.13，手在臂端 ≈ y 1.34）。
 * 模型先按包围盒中心归零，再整体平移，这样"枪身中点落在挂点上"，
 * 不会因为各自原点不同而有的插进胸口、有的飘在身前。
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { scene } from './scene.js';
// 朝向表在 weapon-index.js 里（纯模块，node 可测 —— 见那边的注释）
import { buildWeaponIndex, CLASS_ORIENT, NADE_MODEL_KEYS } from './weapon-index.js';

const ASSET_ROOT = '/static/assets';

/**
 * 武器类别 → 挂载姿态。
 *
 * ## 朝向是**量出来的**，不是猜的
 *
 * `离线资产管线` 沿模型长轴切片、比较两端的横截面
 * （枪口端远细于枪托端），实测结果：
 *
 * | 模型 | 长轴 | 细端（枪口/刀尖） | 结论 |
 * |---|---|---|---|
 * | `weapon_rif_ak47` | z (0.96 m) | **+z** | 已经朝前 |
 * | `weapon_snip_awp` | z (1.39 m) | +z | 已经朝前 |
 * | `weapon_knife_karambit` | y (0.25 m) | **−y** | 竖直，刀尖朝下 |
 * | `weapon_smokegrenade` | y (0.12 m) | +y | 竖直，顶部朝上 |
 *
 * AK 还有 92% 顶点在 Y 中心以上（弹匣/握把在下方），说明 **+Y 就是"上"**。
 *
 * **关键结论：长枪的局部坐标系（+Z 前、+Y 上）和玩家 group 的坐标系完全一致，
 * 所以旋转必须是 0。** 早先按"长轴 +Z 所以要绕 X 转 −90° 放平"来设是错的 ——
 * 那个旋转会把 +Z 映射到 −Y，枪会朝下插进地里。
 *
 * 刀/投掷物是**竖直**建模的（长轴 +Y），要让它们朝前就得转 −90°（+Y → +Z）。
 */

/**
 * 显示名 → 模型条目。解析逻辑全在 `weapon-index.js`（纯函数，node 里能测）。
 *
 * 早先这里自己搞了一套"图标 key 模糊匹配模型目录名"，有两个真实故障：
 * `Knife`（demo 里没记录刀型时的显示名）一个目录都匹配不上 → 默认刀完全不显示；
 * `M4A4` 的图标文件叫 `weapon_m4a1_png.png`，而 `weapon_m4a1_silencer` 的目录名
 * 也以 `m4a1` 开头 → 有概率把 M4A1-S 的模型挂到 M4A4 上。现在改成显式别名表。
 */
let _nameToModel = null;
/** 模型短 key -> 条目。投掷物用（demo 只给引擎类名，没有图标 key 可走）。 */
let _byShortKey = null;
/** manifest.json 只拉一次，两个索引用同一份。 */
let _manifest = null;

async function ensureManifest() {
  if (_manifest) return _manifest;
  try {
    _manifest = await (await fetch(`${ASSET_ROOT}/manifest.json`, { cache: 'no-cache' })).json();
  } catch (e) {
    console.warn('[weapon-models] 读不到 manifest.json：', e.message);
    _manifest = {};
  }
  return _manifest;
}

async function ensureModelIndex() {
  if (_nameToModel) return _nameToModel;
  _nameToModel = {};
  try {
    const m = await ensureManifest();

    let iconMap = {};
    try {
      const r = await fetch(`${ASSET_ROOT}/${m.weapon_icons?.map}`, { cache: 'no-cache' });
      if (r.ok) iconMap = await r.json();
    } catch { /* 没图标映射就没有武器模型 */ }

    const { index, unresolved, byShortKey } = buildWeaponIndex(m.weapons || [], iconMap);
    _nameToModel = index;
    _byShortKey = byShortKey;
    if (unresolved.length) {
      // 不静默：有图标但没模型的显示名要说出来，否则"某些枪不显示"无从排查
      console.info(`[weapon-models] ${unresolved.length} 个显示名没有提取出模型：`,
                   unresolved.join(', '));
    }
  } catch (e) {
    console.warn('[weapon-models] 无法建立武器模型索引：', e.message);
  }
  return _nameToModel;
}

async function ensureModelKeyIndex() {
  if (!_byShortKey) await ensureModelIndex();
  return _byShortKey || {};
}

/** 显示名是否有可用的 3D 模型（给"数据检查"面板用）。 */
export async function hasWeaponModel(displayName) {
  return !!(await ensureModelIndex())[displayName];
}

/** 索引统计：{display_names, unresolved} 里的 unresolved 需要单独取，这里给个快照。 */
export async function weaponIndexStats() {
  const idx = await ensureModelIndex();
  const byClass = {};
  for (const e of Object.values(idx)) byClass[e.cls] = (byClass[e.cls] || 0) + 1;
  return { display_names: Object.keys(idx).length, by_class: byClass };
}

// ── 加载 ───────────────────────────────────────────────────────────────────

const loader = new GLTFLoader();
/** weapon key -> Promise<{scene, size, centre}|null> */
const _modelCache = new Map();

function _loadModel(entry) {
  const key = entry.name;
  if (_modelCache.has(key)) return _modelCache.get(key);

  const dir = entry.dir.replace(/\\/g, '/');
  const url = `${ASSET_ROOT}/${dir}/${entry.glb}`;
  const p = new Promise((resolve) => {
    loader.load(url, (gltf) => {
      const root = gltf.scene;
      // 模型自带的贴图色彩空间由 GLTFLoader 按 glTF 语义设好，这里只统一材质
      root.traverse((o) => {
        if (!o.isMesh) return;
        o.castShadow = true;
        o.frustumCulled = false;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        for (const mat of mats) {
          if (!mat) continue;
          // 提取时是刚性绑定 → three 会丢掉蒙皮属性；显式关掉蒙皮避免警告
          if (mat.map) mat.map.anisotropy = 4;
          mat.side = THREE.FrontSide;
        }
      });
      const box = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      const centre = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(centre);
      resolve({ root, size, centre });
    }, undefined, (err) => {
      console.warn(`[weapon-models] ${url} 加载失败：`, err?.message || err);
      resolve(null);
    });
  });
  _modelCache.set(key, p);
  return p;
}

// ── 投掷物模型（飞行中的那一颗） ───────────────────────────────────────────
//
// 和玩家手上的武器走**同一个** GLB 加载器 / 缓存，区别只在索引：投掷物用
// `byShortKey`（模型短 key），因为 demo 的 `events.grenades[].ty` 是引擎类名
// （`CSmokeGrenadeProjectile` 这种），压根没有图标 key 可走。
//
// 6 个模型 GLB 合计 ~4.5 MB、贴图 ~10 MB，一个回合里最多十几颗同时在飞，
// 所以共用一份加载结果，每条轨迹 `clone(true)` 一次 —— clone 不复制几何和材质
// （只是引用），所以 385 条轨迹也不会把显存撑爆。

/**
 * 按模型短 key 加载一个投掷物模型。
 * @param {string|null} shortKey `nadeModelKey()` 的返回值，如 `'smokegrenade'`
 * @returns {Promise<{root: object, size: object, centre: object}|null>}
 */
export async function loadNadeModel(shortKey) {
  if (!shortKey) return null;
  const entry = (await ensureModelKeyIndex())[shortKey];
  if (!entry) {
    console.warn(`[weapon-models] 投掷物模型 ${shortKey} 不在 manifest 里`);
    return null;
  }
  return _loadModel(entry);
}

/**
 * 预热全部 6 个投掷物模型。**不 await 也安全**（失败只是让飞行中的投掷物
 * 退回那个球）。在 `init()` 里和 sprite 序列一起发起，这样第一个回合开始播放
 * 时模型已经在缓存里，不会先闪一下球。
 */
export function preloadNadeModels() {
  return Promise.all(NADE_MODEL_KEYS.map((k) => loadNadeModel(k)));
}

// ── 玩家手上的武器 ─────────────────────────────────────────────────────────

/** playerIdx -> {holder, key} */
const _attached = new Map();
let _modelsVisible = true;

export function setWeaponModelsVisible(v) {
  _modelsVisible = !!v;
  for (const [, a] of _attached) a.holder.visible = _modelsVisible;
}

export function weaponModelsVisible() { return _modelsVisible; }

/**
 * 给玩家挂上武器模型（幂等：同一把枪不重建）。
 *
 * 玩家 group 的原点在脚底、+Z 是朝向，所以挂点直接写在这个局部空间里。
 */
export async function attachWeapon(playerIdx, playerGroup, displayName) {
  if (!displayName) return;
  const index = await ensureModelIndex();
  const entry = index[displayName];
  if (!entry) {
    // 这把枪没有提取出来的模型 → 静默跳过（图标仍然有）
    const cur0 = _attached.get(playerIdx);
    if (cur0) cur0.holder.visible = false;
    return;
  }

  const cur = _attached.get(playerIdx);
  if (cur && cur.key === entry.name) {
    cur.holder.visible = _modelsVisible;
    return;
  }
  if (cur) { playerGroup.remove(cur.holder); _attached.delete(playerIdx); }

  const model = await _loadModel(entry);
  if (!model) return;

  const key = entry.name;
  // 类别由 weapon-index.js 从**模型文件名**判定（Valve 的 rif/snip/smg/pist/
  // mach/shot 中缀），不是从图标 key 猜。
  const cls = entry.cls || 'rifle';
  const O = CLASS_ORIENT[cls] || CLASS_ORIENT.rifle;

  const holder = new THREE.Group();
  holder.name = `weapon-model-${playerIdx}`;

  // 克隆：同一个模型可能同时挂在多个玩家身上，不能共用节点
  const inst = model.root.clone(true);
  if (O.anchor === 'centre') {
    // 按包围盒中心归零 → "模型中点落在挂点上"，与模型自己的原点无关
    inst.position.set(-model.centre.x, -model.centre.y, -model.centre.z);
  }
  const spin = new THREE.Group();
  spin.rotation.set(O.rot[0], O.rot[1], O.rot[2]);
  spin.add(inst);
  holder.add(spin);

  // 挂点：右手位置 + 类别偏移。玩家 group 的原点在脚底、+Z 是朝向，
  // 所以挂点直接写在这个局部空间里。
  holder.position.set(0.30 + O.off[0], 1.13 + O.off[1], O.off[2]);
  holder.visible = _modelsVisible;
  playerGroup.add(holder);
  _attached.set(playerIdx, { holder, key, cls });
}

/** 摘掉所有玩家的武器模型（换回合 / 清场）。 */
export function clearWeaponModels() {
  for (const [, a] of _attached) a.holder.parent?.remove(a.holder);
  _attached.clear();
}

/** 单独隐藏某个玩家的武器（死亡 / 第一人称藏自己时用）。 */
export function setPlayerWeaponModelVisible(playerIdx, visible) {
  const a = _attached.get(playerIdx);
  if (a) a.holder.visible = !!visible && _modelsVisible;
}

export function weaponModelDebugStats() {
  const out = [];
  for (const [idx, a] of _attached) out.push({ player: idx, model: a.key, cls: a.cls });
  return out;
}

// ── 卡片里的武器展示小窗 ───────────────────────────────────────────────────
//
// 用一个**独立的小 renderer**（主 renderer 只画主场景）。只在卡片展开、
// 且武器变了的时候重画，所以开销可以忽略。

let _sc = null;   // {renderer, scene, camera, group, currentKey}

function _ensureShowcase(canvas) {
  if (_sc) return _sc;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const sc = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 2, 0.01, 100);
  sc.add(new THREE.AmbientLight(0xffffff, 1.6));
  const k = new THREE.DirectionalLight(0xffffff, 2.2);
  k.position.set(2, 3, 2);
  sc.add(k);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.9);
  fill.position.set(-2, -1, -2);
  sc.add(fill);
  const group = new THREE.Group();
  sc.add(group);
  _sc = { renderer, scene: sc, camera, group, currentKey: null };
  return _sc;
}

/** 按模型尺寸自动取景：相机距离 = 最长边 × 2.2，保证不出画。 */
function _frameShowcase(sc, size) {
  const longest = Math.max(size.x, size.y, size.z) || 1;
  const d = longest * 2.2;
  sc.camera.position.set(d * 0.55, d * 0.42, d * 0.75);
  sc.camera.lookAt(0, 0, 0);
  sc.camera.near = longest * 0.05;
  sc.camera.far = longest * 12;
  sc.camera.updateProjectionMatrix();
}

/**
 * 在卡片 canvas 里画某个武器的 3D 预览。武器没变就只转视角（不重载）。
 * @param {HTMLCanvasElement} canvas
 * @param {string} displayName
 * @param {number} spin 自转角（弧度）
 */
export async function renderWeaponShowcase(canvas, displayName, spin = 0) {
  const index = await ensureModelIndex();
  const entry = index[displayName];
  const sc = _ensureShowcase(canvas);

  // 尺寸跟着 CSS 尺寸走（卡片宽度会变）
  const w = canvas.clientWidth || 220, h = canvas.clientHeight || 110;
  if (canvas.width !== w || canvas.height !== h) {
    sc.renderer.setSize(w, h, false);
    sc.camera.aspect = w / h;
    sc.camera.updateProjectionMatrix();
  }

  if (!entry) {
    sc.renderer.setClearColor(0x000000, 0);
    sc.renderer.clear();
    return false;
  }

  if (sc.currentKey !== entry.name) {
    sc.currentKey = entry.name;
    while (sc.group.children.length) sc.group.remove(sc.group.children[0]);
    const model = await _loadModel(entry);
    if (!model) return false;
    const inst = model.root.clone(true);
    inst.position.set(-model.centre.x, -model.centre.y, -model.centre.z);
    sc.group.add(inst);
    _frameShowcase(sc, model.size);
  }

  sc.group.rotation.y = spin;
  sc.renderer.setClearColor(0x000000, 0);
  sc.renderer.clear();
  sc.renderer.render(sc.scene, sc.camera);
  return true;
}

export function showcaseReady() { return !!_sc; }
