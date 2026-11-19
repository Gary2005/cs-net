/**
 * check_dom_ids.mjs — `main.js` 里 `$('xxx')` 拿的每个 id，都必须在 index.html 里存在。
 *
 * ## 为什么需要这个测试
 *
 * `init()` 里 `bindEvents()` **不在 try 里**，所以任何一个 `$('xxx')` 返回 null
 * 都会让 `addEventListener` 抛 TypeError，`init()` 当场中止：3D 场景不建、玩家不
 * 出现、时间轴不动。而浏览器控制台里只有一行 `Cannot read properties of null`，
 * 看不出是哪个 id。
 *
 * 这类故障**真的发生过**：给 index.html 加了一个新开关、main.js 里绑了它，
 * 但 Flask 的 Jinja 在非 debug 模式下会**把模板缓存在进程内存里** —— 服务器不
 * 重启就一直在发旧的 index.html。于是浏览器拿到"新 JS + 旧 HTML"，页面直接挂掉。
 * （`server.py` 里已经补了 `TEMPLATES_AUTO_RELOAD = True`；这个测试守的是
 * 另一半：源码层面 id 本身就不能写错。）
 *
 * 用法：
 *     node visualizer/tests/check_dom_ids.mjs
 */

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const HTML = join(ROOT, 'visualizer', 'templates', 'index.html');
const JS = join(ROOT, 'visualizer', 'static', 'js', 'main.js');

for (const f of [HTML, JS]) {
  if (!existsSync(f)) {
    console.log(`SKIP: 找不到 ${f}`);
    process.exit(0);
  }
}

const html = readFileSync(HTML, 'utf8');
const js = readFileSync(JS, 'utf8');

/**
 * 这些 id 是**JS 现拼出来的**（统计弹窗切换页签 / 换回合时整块重建），
 * index.html 里当然没有。列在这里而不是把检查放宽成"匹配不到就算了"：
 * 上面那条 check 仍然是硬约束，只是把"合法地不存在于静态 HTML"的这几个
 * 明确登记一下。新增动态节点时记得一起加。
 */
const RUNTIME_IDS = new Set([
  'stats-round-detail',     // 传统统计 · 单回合明细的表体
  'stats-ai-round-detail',  // AI 统计 · 单回合明细
  'stats-progress-fill',    // AI 统计 · 进度条
  'stats-progress-stage',   // AI 统计 · 进度文案
  'stats-progress-pct',     // AI 统计 · 百分比 / 用时
]);

const htmlIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
// `$('xxx')` 和 `document.getElementById('xxx')` 两种写法都要覆盖
const used = new Set();
for (const m of js.matchAll(/\$\('([^']+)'\)/g)) used.add(m[1]);
for (const m of js.matchAll(/getElementById\('([^']+)'\)/g)) used.add(m[1]);

let failures = 0;
const check = (ok, label) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}`);
  if (!ok) failures++;
};

console.log(`index.html : ${htmlIds.size} 个 id`);
console.log(`main.js    : ${used.size} 个 id 被引用`);
console.log('');

const missing = [...used]
  .filter((id) => !htmlIds.has(id) && !RUNTIME_IDS.has(id))
  .sort();
for (const id of missing) {
  console.log(`  !! main.js 引用了 $('${id}')，但 index.html 里没有这个 id`);
}
check(missing.length === 0,
  `main.js 引用的 ${used.size} 个 id 在 index.html 里都存在（${RUNTIME_IDS.size} 个是运行时生成的）`);

// 开关（toggle-*）单独再点一次名：这是最容易"加了 HTML 忘了绑 / 绑了忘了加 HTML"的一类
const toggleIds = [...htmlIds].filter((id) => id.startsWith('toggle-')).sort();
const togglesUsed = [...used].filter((id) => id.startsWith('toggle-')).sort();
const unbound = toggleIds.filter((id) => !used.has(id));
const dangling = togglesUsed.filter((id) => !htmlIds.has(id));
console.log('');
console.log(`  开关：HTML ${toggleIds.length} 个，main.js 绑了 ${togglesUsed.length} 个`);
if (unbound.length) console.log(`  （HTML 里有但没绑事件的开关：${unbound.join(', ')}）`);
for (const id of dangling) console.log(`  !! 绑了 #${id} 但 HTML 里没有`);
check(dangling.length === 0, '没有"绑了事件但 HTML 里不存在"的开关');

// 反向只提示不判错：有些 id 是 JS 动态生成后塞进 DOM 的
const unused = [...htmlIds].filter((id) => !used.has(id)).sort();
if (unused.length) {
  console.log(`  （HTML 里有、main.js 没直接引用的 id：${unused.join(', ')}）`);
}

console.log('');
console.log(failures === 0 ? '全部通过' : `${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
