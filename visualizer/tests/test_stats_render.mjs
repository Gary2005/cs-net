#!/usr/bin/env node
/**
 * test_stats_render.mjs — 「统计」弹窗的**渲染**测试（headless Chrome + CDP）。
 *
 * 数据的正确性归 `test_stats.py` 和 `app/tests/test_library.py` 管（聚合、
 * 伤害口径、和库里的记分板对拍）。这里只管**点开之后画出来的东西**：
 *
 *   * 点右侧栏最上面的「统计」→ 弹窗真的开了，两个页签都在
 *   * 传统页签七张表都画好了（1 张每场合计 + 1 张细节 + 4 张每回合矩阵
 *     + 1 张单回合明细），表头的悬停解释也在
 *   * 切到 AI 页签「有反应」：有进度条（冷启动）或者直接出表（缓存命中）
 *   * 合计表的表头就是设计里那 11 列，顺序也对
 *   * 每张矩阵的回合列数 == 这场比赛的实际回合数，每人一行
 *   * **正在回放的那一回合**被标出来（换回合之后还得跟着动，所以这里用点回合号
 *     的方式走一遍真实交互）
 *   * 单回合明细的选回合方式和矩阵一样：一排能点的回合号；点一下 = 回放跳到
 *     那一回合、弹窗**不关**、矩阵高亮列和下面这张表一起换
 *   * 三条关闭路径：✕ / 点遮罩 / Esc —— 这是个**整页遮罩**，关不掉就等于把回放页
 *     锁死了（顶栏、时间轴、画布都在它下面）
 *   * 全程没有 JS 异常
 *
 * 最后截一张图存到 `.build/stats_modal.png`，肉眼看一眼版式（数字列对齐、
 * 表头不换行这些只有看图才知道）。
 *
 * 用法：
 *   python -m app.server --port 5099 --no-watch &
 *   node visualizer/tests/test_stats_render.mjs http://127.0.0.1:5099 /path/to/match.csz
 *
 * 需要 Chrome + 一个在跑的服务端 + 一份 .csz；缺任何一个都以 2 退出（= 没跑成，
 * 不是"通过"）。
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// 和其他 CDP 测试共用同一份候选表（Windows 上以前只能靠外面设 CHROME_PATH）
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const BASE = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3];
const SHOT = process.argv[4] || join(ROOT, '.build', 'stats_modal.png');

if (!CSZ) {
  console.error('用法: node visualizer/tests/test_stats_render.mjs <server-url> <match.csz>');
  process.exit(2);
}
const CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let passed = 0, failed = 0;
const check = (name, ok, detail = '') => {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
};

// ── 先让服务端把这场比赛的会话建起来（弹窗读的是 /api/stats）──────────────
const postJson = async (path, body) => {
  const r = await fetch(BASE + path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: r.status, json: await r.json().catch(() => null) };
};

const { status: loadStatus, json: load } = await postJson('/api/load', { path: CSZ });
if (loadStatus !== 200 || !load || !load.job_id) {
  console.error(`FAIL 无法加载 ${CSZ}: ${loadStatus} ${JSON.stringify(load)}`);
  process.exit(1);
}
let summary = null;
for (let i = 0; i < 900; i++) {
  const st = await (await fetch(`${BASE}/api/load/status?job_id=${load.job_id}`)).json();
  if (st.state === 'done') { summary = st.summary; break; }
  if (st.state === 'error') { console.error('解析失败: ' + st.error); process.exit(1); }
  await sleep(1000);
}
if (!summary) { console.error('FAIL 解析超时'); process.exit(1); }
check(`服务端加载了这场比赛（${summary.rounds.length} 回合）`, summary.rounds.length > 0);

// 接口层的形状先量一遍：渲染那边拿到的就是这份数据
const statsRes = await fetch(BASE + '/api/stats');
const stats = await statsRes.json();
const nPlayers = (stats.players || []).length;
check('/api/stats 有每场合计', statsRes.status === 200 && nPlayers > 0, `${statsRes.status}`);
check('每回合四个数组都和人数等长',
  (stats.per_round || []).length === stats.rounds
  && stats.per_round.every((r) => ['kills', 'deaths', 'damage', 'damage_taken']
    .every((k) => r[k].length === nPlayers)));

// ── 起 Chrome + 接 CDP ──────────────────────────────────────────────────────
// 随机端口：写死端口时连着跑两次，上一轮的 Chrome 还没释放就会连不上
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const profile = mkdtempSync(join(tmpdir(), 'stats-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
  // 软件 WebGL：headless 里没有真 GPU，不给这个 three.js 拿不到 context
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1400,900', 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  await sleep(250);
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    wsUrl = list.find((t) => t.type === 'page')?.webSocketDebuggerUrl;
  } catch { /* not up yet */ }
}
if (!wsUrl) {
  console.log('SKIP: could not reach Chrome DevTools');
  chrome.kill();
  process.exit(2);
}

const ws = new WebSocket(wsUrl);
await new Promise((r) => { ws.onopen = r; });
let msgId = 0;
const pending = new Map();
const consoleErrors = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.method === 'Runtime.exceptionThrown') {
    consoleErrors.push(m.params.exceptionDetails.exception?.description
      || m.params.exceptionDetails.text);
  }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    consoleErrors.push(m.params.args.map((a) => a.value || a.description).join(' '));
  }
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) => new Promise((res) => {
  const id = ++msgId; pending.set(id, res);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true });
  if (r.result?.exceptionDetails) {
    throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 600));
  }
  return r.result?.result?.value;
};

await send('Runtime.enable');
await send('Page.enable');
await send('Page.navigate', { url: `${BASE}/replay` });

let ready = false;
for (let i = 0; i < 120 && !ready; i++) {
  await sleep(500);
  ready = await evaluate(`(() => {
    const app = document.getElementById('app');
    return !!app && !app.classList.contains('hidden')
      && !!document.getElementById('player-card-0');
  })()`).catch(() => false);
}
check('回放页起来了（#app 可见 + 玩家卡片已在）', ready);

check('弹窗一开始是关着的',
  await evaluate(`document.getElementById('stats-modal').classList.contains('hidden')`) === true);

// 真点一下按钮（不是直接调 openStats()：验的是那根线接没接上）
await evaluate(`document.getElementById('btn-stats').click()`);
let opened = false;
for (let i = 0; i < 40 && !opened; i++) {
  await sleep(250);
  opened = await evaluate(`!document.getElementById('stats-modal').classList.contains('hidden')
    && document.querySelectorAll('#stats-body .stats-table').length >= 5`).catch(() => false);
}
check('点「统计」→ 弹窗打开且 5 张表都画好了', opened);

const shape = await evaluate(`(() => {
  const body = document.getElementById('stats-body');
  const tables = [...body.querySelectorAll('.stats-table')];
  const summary = tables[0];
  const detail = tables[1];
  const matrices = [...body.querySelectorAll('.stats-matrix-wrap')];
  const m0 = matrices[0];
  const rowsOf = (t) => t.querySelectorAll('tbody tr:not(.team-row)').length;
  return {
    n_tables: tables.length,
    rows: tables.map(rowsOf),
    head: [...summary.querySelectorAll('thead th')].map(th => th.textContent.trim()),
    detail_head: [...detail.querySelectorAll('thead th')]
      .map(th => th.textContent.replace('?', '').trim()),
    detail_cells: detail.querySelector('tbody tr:not(.team-row) td:nth-child(4)')?.textContent.trim(),
    tips: body.querySelectorAll('th.has-tip').length,
    tabs: [...document.querySelectorAll('.stats-tab')].map(b => b.textContent.trim()),
    subtitle: document.getElementById('stats-subtitle').textContent,
    matrix_titles: matrices.map(m => m.querySelector('.stats-matrix-head').textContent.trim()),
    round_cols: m0 ? m0.querySelectorAll('th.round-col').length : 0,
    active_cols: body.querySelectorAll('th.round-col.active').length,
    active_round: document.querySelector('th.round-col.active')?.dataset.roundCol,
    round_detail_rows: document.querySelectorAll('#stats-round-detail tbody tr:not(.team-row)').length,
    chips: document.querySelectorAll('#stats-body .round-chips .round-chip').length,
    chip_active: document.querySelector('#stats-body .round-chips .round-chip.active')
      ?.textContent.trim(),
    // 这排回合号得在「单回合明细」那个区块里，而不是飘在别处
    chips_in_detail: (() => {
      const block = document.querySelector('#stats-body .round-chips')?.closest('.stats-block');
      return !!block && block.querySelector('.section-title').textContent.includes('单回合明细');
    })(),
    scrolls: body.scrollHeight > body.clientHeight,
  };
})()`);

check('两个页签：传统数据 / AI 数据',
  JSON.stringify(shape.tabs) === JSON.stringify(['传统数据', 'AI 数据']),
  JSON.stringify(shape.tabs));
check('合计表的 11 列和设计一致',
  JSON.stringify(shape.head) === JSON.stringify(
    ['选手', 'K', 'D', 'A', '+/-', 'K/D', 'ADR', 'KPR', '爆头率', '伤害', '承伤']),
  JSON.stringify(shape.head));
check('细节表是「首杀 / 首死 / 先手 / 侧背身 / 穿烟」六列',
  JSON.stringify(shape.detail_head) === JSON.stringify(
    ['选手', '首杀', '首死', '先手开枪', '侧背身击杀', '侧背身被击杀', '穿烟击杀']),
  JSON.stringify(shape.detail_head));
check('细节表的格子里是「分子/分母」', /^\d+\/\d+/.test(shape.detail_cells || ''), shape.detail_cells);
check('表头带解释标记（悬停显示口径）', shape.tips >= 6, String(shape.tips));
check('四张矩阵齐全（击杀 / 死亡 / 造成伤害 / 承受伤害）',
  JSON.stringify(shape.matrix_titles) === JSON.stringify(
    ['击杀每回合 · 合计在最右', '死亡每回合 · 合计在最右',
      '造成伤害每回合 · 合计在最右', '承受伤害每回合 · 合计在最右']),
  JSON.stringify(shape.matrix_titles));
// 合计 + 细节 + 4 张矩阵 + 单回合明细 = 7 张
check(`一共 7 张表（合计 + 细节 + 4 矩阵 + 单回合明细），实际 ${shape.n_tables}`,
  shape.n_tables === 7);
check('每张表每人一行', shape.rows.every((r) => r === nPlayers), JSON.stringify(shape.rows));
check(`矩阵的回合列数 == 这场比赛的回合数（${stats.rounds}）`,
  shape.round_cols === stats.rounds, `${shape.round_cols}`);
check('副标题写出地图 / 回合数 / 人数',
  shape.subtitle === `${stats.map} · ${stats.rounds} 回合 · ${nPlayers} 人`, shape.subtitle);
check('正在回放的那一回合被标出来了',
  shape.active_cols === 4 && shape.active_round === '0',
  `active=${shape.active_cols} round=${shape.active_round}`);
check('单回合明细：回合号是一排能点的数字，一个回合一个，默认标着正在回放的那一回合',
  shape.chips === stats.rounds && shape.chip_active === '1' && shape.chips_in_detail === true
  && shape.round_detail_rows === nPlayers,
  `chips=${shape.chips} active=${shape.chip_active} in=${shape.chips_in_detail} rows=${shape.round_detail_rows}`);
check('弹窗里是滚动的，不是把整页撑开', shape.scrolls === true);

// 悬停表头 → 解释浮层（"把口径写在 title 上"就是这一条）
const tip = await evaluate(`(() => {
  const th = document.querySelector('#stats-body th.has-tip');
  const r = th.getBoundingClientRect();
  th.dispatchEvent(new MouseEvent('mouseover', {
    bubbles: true, clientX: r.left + 3, clientY: r.top + 3,
  }));
  const box = document.querySelector('.stat-tip');
  return box ? { show: box.classList.contains('show'), len: box.textContent.length } : null;
})()`);
check('悬停表头弹出解释浮层', !!tip && tip.show === true && tip.len > 20,
  JSON.stringify(tip));

// 换回合之后高亮要跟着动 —— 这是唯一"开着弹窗还能操作回放"的路径
await evaluate(`(() => {
  const sel = document.getElementById('round-select');
  sel.value = '4';
  sel.dispatchEvent(new Event('change'));
})()`);
await sleep(2500);
const follow = await evaluate(`(() => ({
  col: document.querySelector('th.round-col.active')?.dataset.roundCol,
  chip: document.querySelector('.round-chip.active')?.textContent.trim(),
  caption: document.querySelector('#stats-round-detail .stats-note')?.textContent || '',
}))()`);
check('换回合之后高亮跟着走（第 5 回合那一列）', follow.col === '4', JSON.stringify(follow));
// 明细没有自己的"选中回合"，它跟的就是回放：列头高亮、明细那排数字、表标题三者同步
check('明细的回合号 / 表标题也跟着换到第 5 回合',
  follow.chip === '5' && follow.caption.includes('第 5 回合'), JSON.stringify(follow));

// 切到 AI 页签：第一次要跑模型（几分钟），所以这里只要求"点了有反应" ——
// 有进度条（冷启动）或者直接出表（缓存命中）都算过。到这儿就别再等它跑完了。
await evaluate(`document.querySelector('.stats-tab[data-tab="ai"]').click()`);
let aiOn = false;
for (let i = 0; i < 40 && !aiOn; i++) {
  await sleep(250);
  aiOn = await evaluate(`(() => {
    if (!document.querySelector('.stats-tab[data-tab="ai"]').classList.contains('active')) return false;
    return !!document.getElementById('stats-progress-fill')
      || document.querySelectorAll('#stats-body .stats-ai').length > 0;
  })()`).catch(() => false);
}
check('点 AI 页签 → 有进度条或直接出表', aiOn);
await evaluate(`document.querySelector('.stats-tab[data-tab="trad"]').click()`);
await sleep(200);
check('切回传统页签还正常',
  await evaluate(`document.querySelectorAll('#stats-body .stats-table').length >= 6`) === true);

// 点单回合明细上面那排回合号：跳回放 + 弹窗不关 + 所有表跟着换。
// （矩阵列头是同一个动作，见下一条；这里点的是明细里的，因为那才是新控件。）
await evaluate(`document.querySelectorAll('#stats-body .round-chips .round-chip')[8].click()`);
await sleep(2500);
const after = await evaluate(`(() => ({
  hidden: document.getElementById('stats-modal').classList.contains('hidden'),
  round: document.getElementById('round-select').value,
  chip: document.querySelector('#stats-body .round-chips .round-chip.active')?.textContent.trim(),
  col: document.querySelector('th.round-col.active')?.dataset.roundCol,
  caption: document.querySelector('#stats-round-detail .stats-note')?.textContent || '',
}))()`);
check('点明细上的回合号 = 回放跳到第 9 回合，弹窗不关',
  after.hidden === false && after.round === '8', JSON.stringify(after));
check('点完所有表一起变（矩阵高亮列 / 回合号 / 表标题）',
  after.col === '8' && after.chip === '9' && after.caption.includes('第 9 回合'),
  JSON.stringify(after));

// 矩阵列头走同一条路：也是"跳回合 + 弹窗不关 + 一起变"
await evaluate(`document.querySelectorAll('th.round-col')[2].click()`);
await sleep(2500);
const colClick = await evaluate(`(() => ({
  hidden: document.getElementById('stats-modal').classList.contains('hidden'),
  round: document.getElementById('round-select').value,
  chip: document.querySelector('#stats-body .round-chips .round-chip.active')?.textContent.trim(),
}))()`);
check('点矩阵列头 = 同一个动作（跳第 3 回合，弹窗不关，明细跟着换）',
  colClick.hidden === false && colClick.round === '2' && colClick.chip === '3',
  JSON.stringify(colClick));

// 截图留档（现在弹窗还开着、停在刚跳过去的那一回合）
await sleep(300);
const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  writeFileSync(SHOT, Buffer.from(shot.result.data, 'base64'));
  console.log(`  (截图: ${SHOT})`);
}

// 三条关闭路径
await evaluate(`document.getElementById('btn-stats-close').click()`);
await sleep(200);
check('✕ 能关掉',
  await evaluate(`document.getElementById('stats-modal').classList.contains('hidden')`) === true);

await evaluate(`document.getElementById('btn-stats').click()`);
await sleep(200);
await evaluate(`document.getElementById('stats-modal').click()`);
await sleep(200);
check('点遮罩能关掉',
  await evaluate(`document.getElementById('stats-modal').classList.contains('hidden')`) === true);

await evaluate(`document.getElementById('btn-stats').click()`);
await sleep(200);
for (const type of ['keyDown', 'keyUp']) {
  await send('Input.dispatchKeyEvent',
    { type, key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
}
await sleep(300);
check('Esc 能关掉',
  await evaluate(`document.getElementById('stats-modal').classList.contains('hidden')`) === true);

check('全程没有 JS 异常', consoleErrors.length === 0, consoleErrors.join(' | '));

ws.close();
chrome.kill();
console.log(failed ? `\n${passed} passed, ${failed} FAILED`
                   : `\nstats modal render: ${passed} passed, 0 failed`);
process.exit(failed ? 1 : 0);
