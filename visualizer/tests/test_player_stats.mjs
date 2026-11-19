#!/usr/bin/env node
/**
 * test_player_stats.mjs — 玩家档案页的「数据统计」（传统 / AI 两个页签）。
 *
 * 数据层的对错归 `app/tests/test_ai_stats.py`（AI 那五个数的手算用例）和
 * `visualizer/tests/test_stats.py`（传统指标 + 恒等式）管。这里只管**接线和渲染**：
 *
 *   * 页面打开后传统统计自己就拉回来了（合计卡片 + 每场一行），
 *     比例类都显示成百分比、悬停给样本数，卡片上的数就是端点返回的合计换算来的
 *   * 两个页签在、切换真的换内容
 *   * 表头 / 卡片上的 `?` 悬停能弹出口径解释
 *   * 点 AI 页签「有反应」：有进度条（冷启动要跑模型）或者直接出表（缓存命中）
 *   * 全程没有 JS 异常
 *
 * ⚠ 跑一次会**真的触发服务端的计算**：传统统计要把这位选手打过的每场 `.csz`
 * 各解一遍（第一次每场 5~10 s，之后走 `app/data/stats/`），AI 那一栏更贵
 * （第一次是每场每回合跑一次模型）。缓存建好之后就很快了。
 *
 * 用法：
 *   python -m app.server --port 5099 --no-watch &
 *   node visualizer/tests/test_player_stats.mjs http://127.0.0.1:5099 <steamid>
 *
 * 需要 Chrome + 一个在跑的服务端；缺任何一个都以 2 退出（= 没跑成，不是"通过"）。
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

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
const SID = process.argv[3];
const SHOT = process.argv[4] || join(ROOT, '.build', 'player_stats.png');

if (!SID) {
  console.error('用法: node visualizer/tests/test_player_stats.mjs <server-url> <steamid>');
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

// ── 先看接口：页面拿到的就是这份数据 ─────────────────────────────────────
const statsRes = await fetch(`${BASE}/api/library/players/${SID}/stats`);
const stats = await statsRes.json().catch(() => null);
check('/api/library/players/<sid>/stats 返回 200',
  statsRes.status === 200 && !!stats && stats.ok === true, `${statsRes.status}`);
if (!stats || !stats.ok) {
  console.error(`FAIL 拿不到这位玩家的统计（${statsRes.status}）—— 换个 steamid 或先把库跑起来`);
  process.exit(1);
}
const totals = stats.totals || {};
check('合计里有新指标', ['first_kills', 'first_deaths', 'openers', 'duels',
  'side_kills', 'side_deaths', 'smoke_kills'].every((k) => k in totals),
  JSON.stringify(Object.keys(totals)));
check('每场一行', (stats.matches || []).length > 0, String((stats.matches || []).length));

// ── 起 Chrome + 接 CDP ──────────────────────────────────────────────────────
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const profile = mkdtempSync(join(tmpdir(), 'playerstats-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1600,1100', 'about:blank',
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
await send('Page.navigate', { url: `${BASE}/players/${SID}` });

let ready = false;
for (let i = 0; i < 120 && !ready; i++) {
  await sleep(500);
  ready = await evaluate(`!!document.querySelector('#profile-stats .tile')`).catch(() => false);
}
check('玩家页起来了', ready);

// 传统统计是自动拉的（第一次要解码每场 .csz，给足时间）
let loaded = false;
for (let i = 0; i < 240 && !loaded; i++) {
  await sleep(500);
  loaded = await evaluate(
    `document.querySelectorAll('#player-stats .pstats-table tbody tr').length > 0`
  ).catch(() => false);
}
check('传统统计画好了（每场一行）', loaded);

const shape = await evaluate(`(() => {
  const host = document.getElementById('player-stats');
  const tiles = [...host.querySelectorAll('.tile')];
  const firstRow = host.querySelector('.pstats-table tbody tr');
  return {
    tabs: [...document.querySelectorAll('.pstats-tab')].map(b => b.textContent.trim()),
    tiles: tiles.map(t => t.querySelector('.v').textContent.trim()),
    labels: tiles.map(t => t.querySelector('.l').textContent.trim()),
    tile_tips: tiles.filter(t => t.hasAttribute('data-tip')).length,
    first_tip: (tiles[0] && tiles[0].getAttribute('data-tip')) || '',
    row_cells: firstRow
      ? [...firstRow.querySelectorAll('td')].slice(8).map(td => ({
        text: td.textContent.trim(),
        tip: td.querySelector('[data-tip]')?.getAttribute('data-tip') || '',
      }))
      : [],
    heads: [...host.querySelectorAll('.pstats-table thead th')].map(t => t.textContent.replace('?','').trim()),
    rows: [...host.querySelectorAll('.pstats-table tbody tr')].length,
    note: document.getElementById('pstats-note').textContent,
  };
})()`);

const pctOf = (a, b) => (b ? `${(Math.round(((a || 0) * 1000) / b) / 10).toFixed(1)}%` : '—');
const expectTiles = [
  pctOf(totals.first_kills, totals.rounds),
  pctOf(totals.first_deaths, totals.rounds),
  pctOf(totals.openers, totals.duels),
  pctOf(totals.side_kills, totals.enemy_kills),
  pctOf(totals.side_deaths, totals.enemy_deaths),
  stats.smoke_ok === false ? '—' : pctOf(totals.smoke_kills, totals.enemy_kills),
];

check('两个页签：传统数据 / AI 数据',
  JSON.stringify(shape.tabs) === JSON.stringify(['传统数据', 'AI 数据']),
  JSON.stringify(shape.tabs));
check('六张合计卡片（首杀率 / 首死率 / 先手 / 侧背身杀 / 侧背身死 / 穿烟）',
  JSON.stringify(shape.labels)
  === JSON.stringify(['首杀率', '首死率', '先手开枪', '侧背身击杀', '侧背身被击杀', '穿烟击杀']),
  JSON.stringify(shape.labels));
check('每张卡片都能悬停看口径', shape.tile_tips === (stats.smoke_ok === false ? 5 : 6),
  String(shape.tile_tips));
check('每场一行，和接口一致', shape.rows === (stats.matches || []).length,
  `${shape.rows} vs ${(stats.matches || []).length}`);
check('卡片显示百分比（不是次数），且就是接口合计换算来的',
  JSON.stringify(shape.tiles) === JSON.stringify(expectTiles),
  `${JSON.stringify(shape.tiles)} vs ${JSON.stringify(expectTiles)}`);
check('卡片悬停带样本数', /样本数\s*\d+/.test(shape.first_tip), shape.first_tip);
check('明细表里的比例列也是百分比 + 悬停样本数',
  shape.row_cells.length === 6
  && shape.row_cells.every((c) => /^(\d+\.\d%|—)$/.test(c.text))
  && (stats.smoke_ok === false
    ? shape.row_cells.slice(0, 5).every((c) => /样本数/.test(c.tip))
    : shape.row_cells.every((c) => /样本数/.test(c.tip))),
  JSON.stringify(shape.row_cells));
check('明细表头是「比赛 / 地图 / 回合 + 新指标」',
  shape.heads.slice(-6).join(',') === '首杀率,首死率,先手开枪,侧背身击杀,侧背身被击杀,穿烟击杀',
  JSON.stringify(shape.heads));
check('副标题写了场次数', /\d+/.test(shape.note), shape.note);

// 表头和数值必须共用同一条右边线。这张表是 `.mtable` + 自己一套对齐：`.mtable
// th` 左对齐，数值从第 4 列起右对齐。只改 td 的话（曾经就是这样），窗口一宽
// 列被撑开，每个表头都往左飘，最宽的「侧背身被击杀」在 1877px 下偏了 116px。
const align = await evaluate(`(() => {
  const table = document.querySelector('#player-stats .pstats-table');
  const range = document.createRange();
  const rightEdge = (el) => {
    range.selectNodeContents(el);
    return Math.round(range.getBoundingClientRect().right);
  };
  const textW = (el) => {
    range.selectNodeContents(el);
    return Math.round(range.getBoundingClientRect().width);
  };
  const ths = [...table.querySelectorAll('thead th')];
  const tds = [...table.querySelectorAll('tbody tr')[0].querySelectorAll('td')];
  const cols = ths.map((th, i) => ({
    head: th.textContent.replace('?', '').trim(),
    delta: tds[i] ? rightEdge(tds[i]) - rightEdge(th) : null,
    thAlign: getComputedStyle(th).textAlign,
    tdAlign: tds[i] ? getComputedStyle(tds[i]).textAlign : null,
  }));
  const names = [...table.querySelectorAll('tbody td.name')].map(textW);
  return {
    cols,
    nameColW: Math.round(ths[0].getBoundingClientRect().width),
    longestNameW: Math.max(...names),
    tableW: Math.round(table.getBoundingClientRect().width),
  };
})()`);
const numeric = align.cols.slice(3);
check('数值列的表头和数值同一条右边线',
  numeric.length === 11
  && numeric.every((c) => c.tdAlign === 'right' && c.thAlign === 'right' && Math.abs(c.delta) <= 1),
  JSON.stringify(numeric.filter((c) => Math.abs(c.delta) > 1 || c.thAlign !== 'right')));
check('「比赛 / 地图 / 回合」三列仍是左对齐的表头',
  align.cols.slice(0, 3).every((c) => c.thAlign === 'left' && c.tdAlign === 'left'),
  JSON.stringify(align.cols.slice(0, 3)));
check('「比赛」列不再吃掉整张表的富余宽度',
  align.nameColW - align.longestNameW <= 40,
  `列宽 ${align.nameColW}, 最长文件名 ${align.longestNameW}, 整表 ${align.tableW}`);

// 悬停一张卡片 → 解释浮层
const tip = await evaluate(`(() => {
  const el = document.querySelector('#player-stats .tile.has-tip');
  el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
  const box = document.querySelector('.stat-tip');
  return box ? { show: box.classList.contains('show'), len: box.textContent.length } : null;
})()`);
check('悬停卡片弹出解释', !!tip && tip.show === true && tip.len > 20, JSON.stringify(tip));

await evaluate(`document.getElementById('player-analysis').scrollIntoView({ block: 'start' })`);
await sleep(300);
const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  writeFileSync(SHOT, Buffer.from(shot.result.data, 'base64'));
  console.log(`  (截图: ${SHOT})`);
}

// ── 切到 AI 页签：冷启动要跑模型，这里只要求"有反应" ─────────────────────
await evaluate(`document.querySelector('.pstats-tab[data-tab="ai"]').click()`);
let aiOn = false;
let aiDone = false;
for (let i = 0; i < 40 && !aiOn; i++) {
  await sleep(250);
  const st = await evaluate(`(() => {
    if (!document.querySelector('.pstats-tab[data-tab="ai"]').classList.contains('active')) return null;
    const host = document.getElementById('player-stats');
    return {
      progress: !!host.querySelector('.pstats-progress-fill'),
      table: host.querySelectorAll('.pstats-table tbody tr').length > 0,
      error: host.querySelector('.pstats-error')?.textContent || '',
    };
  })()`).catch(() => null);
  if (st) { aiOn = st.progress || st.table || !!st.error; aiDone = st.table; }
}
check('点 AI 页签 → 有进度条或直接出表', aiOn, '既没有进度条也没有表');

if (aiDone) {
  // 缓存命中：顺手把 AI 那一页的形状也验一遍
  const ai = await evaluate(`(() => {
    const host = document.getElementById('player-stats');
    return {
      tiles: [...host.querySelectorAll('.tile')].length,
      rows: [...host.querySelectorAll('.pstats-table tbody tr')].length,
      heads: [...host.querySelectorAll('.pstats-table thead th')].map(t => t.textContent.replace('?','').trim()),
    };
  })()`);
  check('AI 页签五张卡片', ai.tiles === 5, JSON.stringify(ai.tiles));
  check('AI 每场一行', ai.rows === (stats.matches || []).length,
    `${ai.rows} vs ${(stats.matches || []).length}`);
  check('AI 表头是那五个指标',
    ai.heads.slice(-5).join(',') === '平均击杀影响力,平均回合影响力,平均站位死亡风险,平均站位击杀率,对枪评分',
    JSON.stringify(ai.heads));
  const shot2 = await send('Page.captureScreenshot', { format: 'png' });
  if (shot2.result?.data) {
    writeFileSync(SHOT.replace('.png', '_ai.png'), Buffer.from(shot2.result.data, 'base64'));
    console.log(`  (截图: ${SHOT.replace('.png', '_ai.png')})`);
  }
} else {
  console.log('  (AI 统计还没算完 —— 这次只验证"点了有反应"，也就是进度条)');
}

check('全程没有 JS 异常', consoleErrors.length === 0, consoleErrors.join(' | '));

ws.close();
chrome.kill();
console.log(failed ? `\n${passed} passed, ${failed} FAILED`
                   : `\nplayer stats render: ${passed} passed, 0 failed`);
process.exit(failed ? 1 : 0);
