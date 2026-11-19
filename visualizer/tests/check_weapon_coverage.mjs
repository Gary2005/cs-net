/**
 * check_weapon_coverage.mjs — 每个 demo 用到的武器显示名，都必须能解析出模型。
 *
 * ## 为什么需要这个测试
 *
 * 原来的故障是**静默**的：`Five-SeveN` 压根没提取（只有 6 把武器被 staged），
 * 而默认刀 `Knife` 因为模糊匹配一个目录都命中不了。两种情况都只是"模型不显示"，
 * 不报错、不警告，看代码也看不出来。
 *
 * 这个测试直接跑 `weapon-index.js` 里那份**浏览器实际用的**解析逻辑（纯函数，
 * 不 import three），拿真实 manifest + 真实图标表 + 真实 demo 武器清单去对。
 *
 * 用法：
 *     node visualizer/tests/check_weapon_coverage.mjs
 *
 * 没有 manifest（全新 clone 还没跑提取）时打印 SKIP 并以 0 退出。
 */

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { buildWeaponIndex, CLASS_ORIENT } from '../static/js/weapon-index.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const ASSETS = join(ROOT, 'visualizer', 'static', 'assets');

const manifestPath = join(ASSETS, 'manifest.json');
if (!existsSync(manifestPath)) {
  console.log('SKIP: 没有 manifest.json —— visualizer/static/assets 不完整');
  process.exit(0);
}

const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const iconMapPath = join(ASSETS, manifest.weapon_icons?.map || 'weapon_icons/map.json');
if (!existsSync(iconMapPath)) {
  console.log(`SKIP: 没有图标映射 ${iconMapPath}`);
  process.exit(0);
}
const iconMap = JSON.parse(readFileSync(iconMapPath, 'utf8'));

const { index, unresolved } = buildWeaponIndex(manifest.weapons || [], iconMap);

// demo 实际用到的显示名（fixture）。它记录的是"这份 demo 用过哪些武器"，
// 属于元数据、可以入库；重新生成：
//   PYTHONPATH=tools python -c "from demo_extract import codec; ..."
// 见 visualizer/tests/fixtures/demo_weapons.json 的 _source 字段。
const fixturePath = join(HERE, 'fixtures', 'demo_weapons.json');
let demoWeapons = null;
if (existsSync(fixturePath)) {
  demoWeapons = JSON.parse(readFileSync(fixturePath, 'utf8')).weapons || [];
}

let failures = 0;
const check = (ok, label) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}`);
  if (!ok) failures++;
};

console.log(`manifest : ${(manifest.weapons || []).length} 个武器模型`);
console.log(`icon map : ${Object.keys(iconMap).length} 个显示名`);
console.log(`index    : ${Object.keys(index).length} 个 key 可解析`);
console.log('');

// ── 1. 每个解析出来的条目都要指向真实存在的 GLB ────────────────────────────
let filesMissing = 0;
const seen = new Set();
for (const e of Object.values(index)) {
  if (seen.has(e.name)) continue;
  seen.add(e.name);
  const glb = join(ASSETS, e.dir.replace(/\\/g, '/'), e.glb);
  if (!existsSync(glb)) {
    console.log(`  !! GLB 不存在: ${e.name} -> ${glb}`);
    filesMissing++;
  }
}
check(filesMissing === 0, `${seen.size} 个模型文件全部存在`);

// ── 2. 类别判定：不能有刀/投掷物落进枪的类别 ────────────────────────────────
const byClass = {};
for (const e of Object.values(index)) {
  if (seen.has(e.name) && !byClass[e.name]) byClass[e.name] = e.cls;
}
const misclassified = Object.entries(byClass).filter(([name, cls]) => {
  const isKnife = /knife|bayonet/.test(name);
  const isThrowable = /grenade|molotov|decoy|c4/.test(name);
  if (isKnife) return cls !== 'knife';
  if (isThrowable) return cls !== 'grenade' && cls !== 'c4';
  return false;
});
for (const [name, cls] of misclassified) console.log(`  !! ${name} 被判成 ${cls}`);
check(misclassified.length === 0, '刀 / 投掷物类别判定正确');

// ── 2b. 每个类别都必须有挂载姿态 ───────────────────────────────────────────
//
// 这条是补一个**已经真实发生过**的故障：`attachWeapon` 里写的是
// `CLASS_ORIENT[cls] || CLASS_ORIENT.rifle`，所以一个没登记的类别会被
// **静默**当成步枪挂 —— 表现是刀竖着插在手里，而不是报错。
// 当时 `weaponClass` 返回 'knife' 而表里没有（或返回了别的），就是这样。
const classCount = {};
for (const [name, cls] of Object.entries(byClass)) {
  classCount[cls] = (classCount[cls] || 0) + 1;
}
console.log('');
console.log('  类别分布：' +
  Object.entries(classCount).sort().map(([c, n]) => `${c}×${n}`).join('  '));
const noPose = Object.keys(classCount).filter((c) => !CLASS_ORIENT[c]);
for (const c of noPose) console.log(`  !! 类别 ${c} 没有 CLASS_ORIENT 条目 -> 会静默按 rifle 挂`);
check(noPose.length === 0,
  `${Object.keys(classCount).length} 个类别全部有挂载姿态`);

// 反向：表里有但用不到的类别不算错，只提示（可能是为将来准备的）
const unused = Object.keys(CLASS_ORIENT).filter((c) => !classCount[c]);
if (unused.length) console.log(`  （CLASS_ORIENT 里未用到的类别：${unused.join(', ')}）`);

// ── 3. demo 用到的武器必须全部解析出模型 ───────────────────────────────────
if (demoWeapons) {
  console.log('');
  console.log(`demo 用到 ${demoWeapons.length} 个显示名：`);
  const missing = [];
  for (const w of demoWeapons) {
    const e = index[w];
    if (e) {
      console.log(`  PASS  ${w.padEnd(24)} -> ${e.name.padEnd(30)} [${e.cls}]`);
    } else {
      console.log(`  FAIL  ${w.padEnd(24)} -> 没有模型`);
      missing.push(w);
    }
  }
  check(missing.length === 0, `demo 的 ${demoWeapons.length} 个武器全部有模型`);
}

console.log('');
if (unresolved.length) {
  console.log(`（有图标但没提取模型的显示名，${unresolved.length} 个：`);
  console.log(`  ${unresolved.join(', ')}）`);
}
console.log(failures === 0 ? '\n全部通过' : `\n${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
