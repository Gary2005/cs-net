/**
 * check_nade_models.mjs — 飞行中的投掷物能不能挂上正确的真实模型。
 *
 * ## 为什么需要这个测试
 *
 * 投掷物这条链路上有**三个**都能静默失败的地方：
 *
 * 1. **类名 → 模型 key 的映射错了** —— demo 的 `events.grenades[].ty` 是引擎类名
 *    （`CSmokeGrenadeProjectile`），不是显示名。映射漏一个就是"某种投掷物永远是
 *    个球"，不报错。
 * 2. **manifest 里没有这个模型** —— `loadNadeModel()` 只能 `console.warn` 然后
 *    退回球。
 * 3. **朝向假设错了** —— 代码把模型长轴当成 `+Y` 来对齐速度方向
 *    （`visuals.js` 的 `NADE_LONG_AXIS`）。要是某个模型长轴是 Z，它就会一直
 *    横着飞，而且不会有任何报错。
 *
 * 第 3 条是**量出来的**：直接读 GLB 的 POSITION accessor min/max。
 * 这些 GLB 的节点矩阵是单位阵（已断言），所以 accessor 的包围盒就是世界包围盒，
 * 和 three 的 `Box3().setFromObject()` 得到的是同一个东西。
 *
 * 用法：
 *     node visualizer/tests/check_nade_models.mjs
 *
 * 没有 manifest（全新 clone 还没跑提取）时打印 SKIP 并以 0 退出。
 */

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  buildWeaponIndex, nadeModelKey, NADE_PROJECTILE_MODEL, NADE_MODEL_KEYS,
} from '../static/js/weapon-index.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const ASSETS = join(ROOT, 'visualizer', 'static', 'assets');

const manifestPath = join(ASSETS, 'manifest.json');
if (!existsSync(manifestPath)) {
  console.log('SKIP: 没有 manifest.json —— 先跑 tools/cs2_assets/extract_cs2_assets.py');
  process.exit(0);
}

const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const { byShortKey } = buildWeaponIndex(manifest.weapons || [], {});

/**
 * demo 里真实出现过的投掷物类名与采样点数。
 * 来源：`notes/scratch/grenade_type_probe.py`（de_inferno 24 回合 / 327266 个采样点）。
 * 这份清单是**观察结果**，不是猜的 —— 所以映射表少一个、多一个都会被这里抓到。
 */
const OBSERVED_TYPES = {
  CSmokeGrenadeProjectile: 249909,
  CHEGrenadeProjectile: 46101,
  CMolotovProjectile: 15435,
  CFlashbangProjectile: 14456,
  CDecoyProjectile: 1365,
};

let failures = 0;
const check = (ok, label) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}`);
  if (!ok) failures++;
};

console.log(`manifest : ${(manifest.weapons || []).length} 个武器模型`);
console.log(`投掷物   : ${Object.keys(NADE_PROJECTILE_MODEL).length} 个类名 -> ${NADE_MODEL_KEYS.length} 个模型`);
console.log('');

// ── 1. demo 真实出现过的类名必须全部有映射 ─────────────────────────────────
const unmapped = Object.keys(OBSERVED_TYPES).filter((t) => !NADE_PROJECTILE_MODEL[t]);
for (const t of unmapped) console.log(`  !! ${t} 没有映射 -> 这种投掷物永远是个球`);
check(unmapped.length === 0,
  `demo 里实测的 ${Object.keys(OBSERVED_TYPES).length} 个类名全部有映射`);

// 反向：映射表里的每个类名都得是"看起来像 projectile 的类名"（防手滑写错）
const weird = Object.keys(NADE_PROJECTILE_MODEL).filter((t) => !/Projectile$/.test(t));
check(weird.length === 0, '映射表的键都是 *Projectile 类名');

// ── 2. 队伍 → 莫洛托夫 / 燃烧弹 ─────────────────────────────────────────────
check(nadeModelKey('CMolotovProjectile', 'CT') === 'incendiarygrenade',
  "CT 的 CMolotovProjectile -> incendiarygrenade（燃烧弹）");
check(nadeModelKey('CMolotovProjectile', 'T') === 'molotov',
  "T 的 CMolotovProjectile -> molotov（莫洛托夫）");
check(nadeModelKey('CMolotovProjectile', null) === 'molotov',
  '队伍未知时退回 molotov');
check(nadeModelKey('CSmokeGrenadeProjectile', 'CT') === 'smokegrenade',
  '烟雾弹不受队伍影响');
check(nadeModelKey('CNotARealProjectile', 'T') === null,
  '认不出来的类名返回 null（宁可退回球，也不挂错模型）');
check(nadeModelKey(undefined, 'T') === null, 'undefined 返回 null');

// ── 3. 每个 key 都必须在 manifest 里，且 GLB 真的存在 ───────────────────────
console.log('');
let missingKey = 0, missingFile = 0;
for (const key of NADE_MODEL_KEYS) {
  const e = byShortKey[key];
  if (!e) {
    console.log(`  !! ${key} 不在 manifest 里`);
    missingKey++;
    continue;
  }
  const glb = join(ASSETS, e.dir.replace(/\\/g, '/'), e.glb);
  if (!existsSync(glb)) {
    console.log(`  !! ${key} -> ${glb} 不存在`);
    missingFile++;
  }
}
check(missingKey === 0, `${NADE_MODEL_KEYS.length} 个模型 key 都在 manifest 里`);
check(missingFile === 0, `${NADE_MODEL_KEYS.length} 个 GLB 文件都存在`);

// ── 4. 长轴必须是 +Y（代码里 NADE_LONG_AXIS = (0,1,0)） ────────────────────
console.log('');

/** 读 GLB 的 JSON chunk。 */
function glbJson(path) {
  const b = readFileSync(path);
  if (b.toString('ascii', 0, 4) !== 'glTF') throw new Error('不是 GLB');
  let off = 12;
  while (off + 8 <= b.length) {
    const len = b.readUInt32LE(off);
    const type = b.readUInt32LE(off + 4);
    off += 8;
    if (type === 0x4e4f534a) return JSON.parse(b.toString('utf8', off, off + len).trim());
    off += len;
  }
  throw new Error('没有 JSON chunk');
}

const rows = [];
for (const key of NADE_MODEL_KEYS) {
  const e = byShortKey[key];
  if (!e) continue;
  const glb = join(ASSETS, e.dir.replace(/\\/g, '/'), e.glb);
  if (!existsSync(glb)) continue;
  const js = glbJson(glb);

  // 节点矩阵必须是单位阵，否则 accessor 的 min/max 不等于世界包围盒
  for (const n of js.nodes || []) {
    if (!n.matrix) continue;
    const m = n.matrix;
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
      .every((v, i) => Math.abs(m[i] - v) < 1e-4);
    if (!identity) {
      console.log(`  !! ${key} 的节点带非单位矩阵 -> 包围盒不能用 accessor 直接读`);
      failures++;
    }
  }

  const prims = js.meshes.flatMap((m) => m.primitives || []);
  // **所有 primitive 的并集**，不是只取第 0 个：莫洛托夫有 3 个
  // （瓶身 / 火焰片 / 瓶里的液体），只量瓶身会得到 0.091×0.250×0.210，
  // 而 three 的 `Box3().setFromObject()`（也就是 app 里真正用的那个）量到的是
  // 0.119×0.351×0.232 —— 液体那一块比瓶底还低 0.10 m。
  // 只量第 0 个的话，这个测试报的数字和 app 的行为对不上。
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  let tris = 0;
  for (const pr of prims) {
    const acc = js.accessors[pr.attributes.POSITION];
    for (let i = 0; i < 3; i++) {
      lo[i] = Math.min(lo[i], acc.min[i]);
      hi[i] = Math.max(hi[i], acc.max[i]);
    }
    if (pr.indices != null) tris += js.accessors[pr.indices].count / 3;
  }
  const size = hi.map((v, i) => v - lo[i]);
  const longest = size.indexOf(Math.max(...size));
  const axis = 'xyz'[longest];
  rows.push({ key, size, axis, tris, prims: prims.length });
}

console.log('  模型包围盒（米）与长轴 —— 所有 primitive 的并集，和 three 的 Box3 一致：');
for (const r of rows) {
  console.log(`    ${r.key.padEnd(20)} ${r.size.map((v) => v.toFixed(4)).join(' × ')}  `
    + `长轴=${r.axis}  primitive=${r.prims}  三角形=${Math.round(r.tris).toLocaleString()}`);
}
const notY = rows.filter((r) => r.axis !== 'y');
for (const r of notY) console.log(`  !! ${r.key} 的长轴是 ${r.axis}，不是 y`);
check(notY.length === 0 && rows.length === NADE_MODEL_KEYS.length,
  `${rows.length} 个投掷物模型长轴全部是 y（NADE_LONG_AXIS 的前提）`);

// 尺寸合理性：投掷物是"手里能拿的东西"，超过 0.5 m 或小于 0.02 m 就不对劲
const badSize = rows.filter((r) => Math.max(...r.size) > 0.5 || Math.max(...r.size) < 0.02);
for (const r of badSize) console.log(`  !! ${r.key} 最长边 ${Math.max(...r.size)} m 不像投掷物`);
check(badSize.length === 0, '尺寸都在 0.02–0.5 m（是手里拿的东西，不是枪）');

console.log('');
console.log(failures === 0 ? '全部通过' : `${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
