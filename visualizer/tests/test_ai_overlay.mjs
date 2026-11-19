#!/usr/bin/env node
/**
 * test_ai_overlay.mjs — AI 预测覆盖层的**渲染**测试（headless Chrome + CDP）。
 *
 * 数据层的正确性归 `app/tests/test_ai.py` 和 `test_ai_service.py`；这里只量
 * "勾上开关之后，画出来的东西对不对"：
 *
 *   * 两个开关都在、而且可用（服务端 `/api/ai/status` 认为模型在）
 *   * 勾上「AI 对枪」→ 当前回合的预测被拉下来（服务端已经算过，走缓存）
 *   * 挑一个预测点，弧线真的画出来了：两段、都有顶点、颜色不一样、两段首尾
 *     相接（中间**没有游标点**）、**3D 里一个文字 sprite 都没有**（数字只在左下表格）
 *   * 3D 层里没有光环 / 没有别的杂项：除了这条弧线什么都不画
 *   * 顶部面板的 CT / T 两个数加起来是 100
 *   * 左下表格 10 行 × 5 格、对枪那两行带 ⚔ 且胜率互补、悬停写着两位对手的名字
 *   * 表头悬停会解释这一项（各项各不相同、卡完整落在视口里）
 *   * 点名字 = 点左边卡片（选中 + 跟随），表格里那一行也亮起来
 *   * 点数字打开这一项整回合的曲线；点曲线任意一点跳到那一刻并锁定镜头；
 *     ✕ / Esc 关掉；点对枪那一格不会打开曲线
 *   * 点顶部的胜率也打开曲线（win_ct / win_t，T = 100% − CT）；点曲线只跳时间，
 *     不抢镜头；悬停胜率也有释义
 *   * 「选手 ID」开关默认勾选、每人头顶一个写名字的 Sprite，关掉后整批消失
 *   * 关掉开关 → 3D 层立刻隐藏
 *   * 全程没有 JS 异常
 *
 * 用法（先起服务端，再给一个 .csz）：
 *   python -m app.server --port 5099 --no-watch &
 *   node visualizer/tests/test_ai_overlay.mjs http://127.0.0.1:5099 app/data/csz/xxx.csz
 *
 * 需要 Chrome；没有就 SKIP（退出码 2，不算失败）。
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
const CSZ = process.argv[3];
const SHOT = process.argv[4] || join(ROOT, '.build', 'ai_overlay.png');
if (!CSZ) {
  console.error('用法: node visualizer/tests/test_ai_overlay.mjs <server-url> <match.csz>');
  process.exit(2);
}
const CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let passed = 0;
let failed = 0;
const check = (name, ok, detail = '') => {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
};

// ── 让服务端把这场比赛的会话建起来，并把第 0 回合的预测先算好 ─────────────
const post = async (path, body) => {
  const r = await fetch(BASE + path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: r.status, json: await r.json().catch(() => null) };
};
const { status: loadStatus, json: load } = await post('/api/load', { path: CSZ });
if (loadStatus !== 200 || !load?.job_id) {
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
check(`服务端加载了这场比赛（${summary?.rounds?.length ?? 0} 回合）`, !!summary);

const status = await (await fetch(BASE + '/api/ai/status')).json();
check('AI 服务端可用（模型 + torch 都在）', !!status.ok, JSON.stringify(status).slice(0, 300));

const t0 = Date.now();
const aiRes = await fetch(BASE + '/api/ai/round/0');
const ai = await aiRes.json();
check(`/api/ai/round/0 返回预测（${((Date.now() - t0) / 1000).toFixed(1)}s）`,
  aiRes.status === 200 && ai.ok === true, JSON.stringify(ai).slice(0, 300));
if (!ai.ok) process.exit(1);
const stride = ai.stride;
const armed = (ai.samples || []).map((_, k) => k).filter((k) => ai.duel.a[k] != null);
check(`有对枪预测点（${armed.length}/${ai.samples.length}）`, armed.length > 0);

// ── 五个头都在（少一个就是"某个头没接上"，界面只会永远空着）──────────────
for (const head of ['win_ct', 'alive', 'can_kill', 'next_death', 'next_kill']) {
  check(`payload 里有 spatial.${head}`, ai.spatial && ai.spatial[head] !== undefined);
}
check('alive / can_kill 是逐人 10 列',
  (ai.spatial.alive[0] || []).length === 10 && (ai.spatial.can_kill[0] || []).length === 10);
check('next_death / next_kill 是 11 类',
  (ai.spatial.next_death[0] || []).length === 11 && (ai.spatial.next_kill[0] || []).length === 11);

// ── 对枪只在"死前 3 秒"里 ─────────────────────────────────────────────────
const duels = ai.duels || [];
check(`payload 里列出每场对枪（${duels.length} 场）`, duels.length > 0);
check('每场对枪的窗口正好 3 秒（或从回合开头起）',
  duels.every((d) => d.t1 - d.t0 <= 3.05 && d.t1 >= d.t0), JSON.stringify(duels.slice(0, 2)));
{
  const outside = armed.filter((k) => {
    const s = ai.seconds[k];
    const t0 = ai.duel.t0[k];
    const t1 = ai.duel.t1[k];
    return !(t0 <= s + 1e-6 && s <= t1 + 1e-6);
  });
  check(`每个对枪预测点都落在自己的窗口里（越界 ${outside.length} 个）`,
    outside.length === 0, JSON.stringify(outside.slice(0, 3)));
  // 一个人死前 3 秒最多 13 个 4 Hz 的采样点（192 帧 = 12 个间隔，两端都算），
  // 所以对枪点不该铺满整个回合
  check(`对枪点只占回合的一小段（${armed.length}/${ai.samples.length}）`,
    armed.length <= 13 * duels.length + 4 && armed.length < ai.samples.length * 0.5,
    `${armed.length} / ${ai.samples.length}`);
  // 窗口之外一个预测都没有：这正是"没有虚线圆弧"的判据
  const empty = (ai.samples.length - armed.length);
  check(`窗口之外没有对枪预测（${empty} 个点）`, empty > ai.samples.length * 0.5);
}

// 找一个"两个人都活着"的采样点：弧线在有人死了的时候是故意藏起来的。
// 这里不猜，交给浏览器那边试几个候选点。
const candidates = [];
for (const k of armed) {
  if (candidates.length >= 24) break;
  if (k * stride > 40 && k * stride < ai.frames - 40) candidates.push(k);
}

// ── Chrome + CDP ─────────────────────────────────────────────────────────
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const profile = mkdtempSync(join(tmpdir(), 'csai-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
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
const errors = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.method === 'Runtime.exceptionThrown') {
    errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    errors.push(m.params.args.map((a) => a.value || a.description).join(' '));
  }
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) => new Promise((res) => {
  const id = ++msgId;
  pending.set(id, res);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
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
      && !!document.getElementById('player-card-0') && !!window.__v5;
  })()`).catch(() => false);
}
check('回放页起来了（app 可见 + 玩家卡片在位）', ready);
if (!ready) { chrome.kill(); process.exit(1); }

// 等状态查询回来（开关会被解除 disabled）
let enabled = false;
for (let i = 0; i < 40 && !enabled; i++) {
  await sleep(250);
  enabled = await evaluate(`(async () => {
    const el = document.getElementById('toggle-ai-duel');
    const st = window.__v5.ai.status;
    return !!st && st.ok === true && !el.disabled;
  })()`).catch(() => false);
}
check('两个 AI 开关可用（模型在）', enabled);

// 勾上两个开关（真的点一下，走的是 change 事件）
await evaluate(`(() => {
  for (const id of ['toggle-ai-spatial', 'toggle-ai-duel']) {
    const el = document.getElementById(id);
    el.checked = true;
    el.dispatchEvent(new Event('change'));
  }
})()`);

let loaded = false;
for (let i = 0; i < 320 && !loaded; i++) {
  await sleep(250);
  loaded = await evaluate(`!!window.__v5.ai.round && window.__v5.ai.loadedFor === window.__v5.ai.activeRound`);
}
check('勾上开关后当前回合的预测就绪（服务端缓存）', loaded);

// 逐步试候选采样点，直到弧线真的画出来（有人死了的时候它是藏着的）。
// 同时记下"两人离得多近"——截图要挑一对靠近的，不然相机得拉到几十米外。
const spread = [];
for (const k of armed) {
  const i = k * stride;
  if (i > 200 && i < ai.frames - 100) spread.push(k);
}
const picked = [];
for (let j = 0; j < spread.length && picked.length < 24; j += Math.max(1, Math.floor(spread.length / 24))) {
  picked.push(spread[j]);
}

let hit = null;
for (const k of picked) {
  await evaluate(`window.__v5.seekSample(${k * stride})`);
  await sleep(120);
  const info = await evaluate(`(() => {
    const ai = window.__v5.ai;
    const pl = window.__v5.players;
    const d = ai.frame && ai.frame.duel
      ? Math.hypot(pl[ai.frame.duel.a].x - pl[ai.frame.duel.b].x,
                   pl[ai.frame.duel.a].y - pl[ai.frame.duel.b].y)
      : 1e9;
    return { frame: ai.frame, overlay: ai.overlay, sample: window.__v5.sampleIdx, dist: d };
  })()`);
  if (!info?.overlay?.duelVisible) continue;
  if (!hit || info.dist < hit.dist) hit = { k, ...info };
  if (hit.dist < 900) break;          // 900 HU ≈ 23 m：够近，能看清两个人
}
check('弧线在一个有对枪的采样点上画出来了', !!hit);
if (hit) console.log(`       （采样点 ${hit.sample}，两人相距 ${Math.round(hit.dist)} HU）`);

if (hit) {
  const { overlay, frame } = hit;
  const arcs = overlay.arcs;
  check('弧线是两段（A 侧 / B 侧）', arcs.length === 2, JSON.stringify(arcs.map((a) => a.segments)));
  check('两段都有顶点', arcs.every((a) => a.segments > 0),
    JSON.stringify(arcs.map((a) => a.segments)));
  check('两侧颜色不同（一边偏绿、一边偏红）', arcs[0].color !== arcs[1].color,
    `${arcs[0].color} vs ${arcs[1].color}`);
  const pct = Math.round((frame.duel?.p ?? 0) * 100);
  // 头顶的百分比标签撤掉之后，3D 层里应当**一个文字对象都没有**：数字只在左下
  // 表格里。这条是防回归的（谁把文字加回去，这里立刻红）。
  check('3D 层不画文字（数字只在左下角表格里）',
    overlay.labels === undefined && arcs.every((a) => a.label === undefined),
    JSON.stringify({ labels: overlay.labels, arcs: arcs.map((a) => a.label) }));
  // 游标那个白色小球撤掉之后，两段弧必须首尾相接：A 段最后一个顶点 = B 段第一个
  // 顶点（虚线是隔一段取一个点，所以只能比两端，不能比中点）。
  const joint = Math.hypot(
    arcs[0].ends[1][0] - arcs[1].ends[0][0],
    arcs[0].ends[1][1] - arcs[1].ends[0][1],
    arcs[0].ends[1][2] - arcs[1].ends[0][2]);
  check(`两段弧首尾相接（接缝 ${joint.toFixed(2)}），中间没有游标点`, joint < 0.5,
    JSON.stringify({ a: arcs[0].ends, b: arcs[1].ends }));

  // 存活光环整层撤了：3D 里除了这条对枪弧线什么都没画（数字全在左下表格）
  check('3D 层里没有光环 / 没有别的杂项（只剩对枪弧线）',
    overlay.rings === undefined
      && JSON.stringify(Object.keys(overlay).sort())
        === JSON.stringify(['arcs', 'built', 'duelVisible']),
    JSON.stringify(Object.keys(overlay)));

  const panel = await evaluate(`(() => ({
    hidden: document.getElementById('ai-panel').classList.contains('hidden'),
    progressOnly: document.getElementById('ai-panel').classList.contains('progress-only'),
    ct: document.getElementById('ai-win-ct').textContent,
    t: document.getElementById('ai-win-t').textContent,
    progressHidden: document.getElementById('ai-progress').classList.contains('hidden'),
    model: document.getElementById('ai-model').textContent,
    modelTitle: document.getElementById('ai-model').title,
    state: document.getElementById('ai-state').textContent,
    tableHidden: document.getElementById('ai-table').classList.contains('hidden'),
    rows: Array.from(document.querySelectorAll('#ai-table-body .ai-row')).map((r) => ({
      player: Number(r.dataset.player),
      name: (r.querySelector('.ai-row-nick') || {}).textContent,
      dead: r.classList.contains('dead'),
      duel: r.classList.contains('duel'),
      role: (r.querySelector('.ai-row-role') || {}).textContent,
      duelTitle: (r.querySelector('.ai-cell[data-metric="duel"]') || {}).title,
      cells: Array.from(r.querySelectorAll('.ai-cell')).map((c) => ({
        metric: c.dataset.metric,
        text: (c.querySelector('b') || {}).textContent,
        width: (c.querySelector('i') || {}).style.width,
      })),
    })),
  }))()`);
  check('顶部 AI 面板显示出来了', panel.hidden === false);
  check('开了「AI 预测面板」时不是只剩进度条', panel.progressOnly === false);
  check('CT% + T% == 100', Number(panel.ct.replace('%', '')) + Number(panel.t.replace('%', '')) === 100,
    `${panel.ct} / ${panel.t}`);
  check('分析完之后进度条收起', panel.progressHidden === true);
  check('面板写了用的是哪份权重', panel.model.includes('aim') && panel.model.includes('spatial'),
    panel.model);
  // 权重路径写死（PINNED_WEIGHTS），悬停要能看出到底是哪个文件 —— 版本管理靠它
  check('悬停显示具体的权重文件名', /last\.pt|best\.pt/.test(panel.modelTitle), panel.modelTitle);

  // ── 左下角的选手指标表：10 行 × 5 格 ───────────────────────────────────
  check('指标表显示出来了', panel.tableHidden === false);
  check(`指标表有 10 行（${panel.rows.length}）`, panel.rows.length === 10);
  check('每行五个格子（存活 / 可击杀 / 下次击杀 / 下次被击杀 / 对枪）',
    panel.rows.every((r) => r.cells.length === 5
      && JSON.stringify(r.cells.map((c) => c.metric))
        === JSON.stringify(['alive', 'can_kill', 'next_kill', 'next_death', 'duel'])),
    JSON.stringify(panel.rows[0]?.cells));
  const aliveRows = panel.rows.filter((r) => !r.dead);
  check(`活着的行都填了数字（${aliveRows.length} 行活着）`,
    aliveRows.length > 0 && aliveRows.every((r) => r.cells.slice(0, 4)
      .every((c) => /^(\d+%|<1%)$/.test(c.text))),
    JSON.stringify(aliveRows[0]?.cells));
  check('细条宽度就是那个百分比（差不超过 1 个百分点，取整方式不同）',
    aliveRows.every((r) => r.cells.slice(0, 4).every((c) => {
      const num = c.text === '<1%' ? 0 : Number(c.text.replace('%', ''));
      return Math.abs(Number(c.width.replace('%', '')) - num) <= 1;
    })),
    JSON.stringify(aliveRows[0]?.cells));
  const deadRows = panel.rows.filter((r) => r.dead);
  check(`死了的行整行变淡、概率是 —（${deadRows.length} 行）`,
    deadRows.every((r) => r.cells.every((c) => c.text === '—' && c.width === '0%')),
    JSON.stringify(deadRows[0]?.cells));
  // 弧线两端就是表里被标成「对枪」的那两行，两边胜率互补（A 34% ↔ B 66%）
  const duelRows = panel.rows.filter((r) => r.duel);
  check(`对枪的两行被标出来了（${duelRows.map((r) => r.name).join(' / ')}）`,
    duelRows.length === 2 && duelRows.some((r) => r.player === frame.duel.a)
      && duelRows.some((r) => r.player === frame.duel.b),
    JSON.stringify(duelRows.map((r) => r.player)));
  check('对枪两行的胜率加起来 100%',
    duelRows.length === 2
      && Math.abs(duelRows.reduce((s, r) => s + Number(r.cells[4].text.replace('%', '')), 0) - 100) <= 1,
    duelRows.map((r) => r.cells[4].text).join(' + '));
  check('对枪那两行带 ⚔', duelRows.every((r) => r.role === '⚔'),
    JSON.stringify(duelRows.map((r) => r.role)));
  check('对枪那一格是 A 的胜率',
    duelRows.length === 2
      && Number(duelRows.find((r) => r.player === frame.duel.a).cells[4].text.replace('%', ''))
        === pct,
    `${duelRows.find((r) => r.player === frame.duel.a)?.cells[4].text} vs ${pct}%`);
  // 弧线上不再写名字，所以"这两个人是谁"由这一格说清楚（悬停可见）
  check('对枪那一格的悬停提示写着两位选手的名字',
    duelRows.length === 2 && duelRows.every((r) => r.duelTitle
      && r.duelTitle.includes(frame.duel.names[0]) && r.duelTitle.includes(frame.duel.names[1])),
    JSON.stringify(duelRows.map((r) => r.duelTitle)));

  // ══ 表格上的三种交互：悬停表头 / 点名字 / 点数字 ══════════════════════
  //
  // 这三样以前都不存在（`.ai-table` 整块 pointer-events:none，连原生 title 都
  // 弹不出来），所以每一条都从**真实 DOM 事件**走一遍：PointerEvent 派发到列头 /
  // 名字 / 数字上，读的是页面里真正显示出来的东西。

  // ── 1. 悬停表头 ───────────────────────────────────────────────────────
  // 释义卡有 120 ms 的淡入（CSS transition），而 headless 的 rAF 只有 1~3 fps：
  // 派发事件后必须真的等到两帧，否则读到的还是"过渡还没开始"的 opacity:0。
  const tip = await evaluate(`(async () => {
    const col = document.querySelector('.ai-col[data-metric="next_death"]');
    col.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
    await new Promise((r) => requestAnimationFrame(r));
    await new Promise((r) => requestAnimationFrame(r));
    const box = document.getElementById('ai-tip');
    const r = box.getBoundingClientRect();
    return {
      shown: box.classList.contains('show'),
      visibility: getComputedStyle(box).visibility,
      opacity: Number(getComputedStyle(box).opacity),
      metric: box.dataset.metric,
      title: document.getElementById('ai-tip-title').textContent,
      body: document.getElementById('ai-tip-body').textContent,
      hint: document.getElementById('ai-tip-hint').textContent,
      inView: r.left >= 0 && r.top >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1,
      w: Math.round(r.width),
    };
  })()`);
  check('悬停表头弹出了释义卡（可见、淡入完成）',
    tip.shown && tip.visibility === 'visible' && tip.opacity > 0.5,
    JSON.stringify({ shown: tip.shown, visibility: tip.visibility, opacity: tip.opacity }));
  check('释义卡解释的正是被悬停的那一列（下次被击杀）',
    tip.metric === 'next_death' && tip.title === '下次被击杀'
      && tip.body.includes('下一个死的会是他'),
    JSON.stringify({ metric: tip.metric, title: tip.title }));
  check('释义卡写了这一列怎么用（点数字看曲线）', tip.hint.includes('曲线'), tip.hint);
  check(`释义卡完整落在视口里（宽 ${tip.w}px）`, tip.inView, JSON.stringify(tip));
  const tipShot = await send('Page.captureScreenshot', { format: 'png' });
  if (tipShot.result?.data) {
    writeFileSync(join(ROOT, '.build', 'ai_tip.png'),
      Buffer.from(tipShot.result.data, 'base64'));
    console.log(`  (截图: ${join(ROOT, '.build', 'ai_tip.png')})`);
  }

  // 每一列 / 两个胜率各有各的说法：不能共用一句话（那样等于没解释）
  const help = await evaluate(`(() => {
    const out = {};
    const read = (key, el) => {
      el.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
      out[key] = {
        title: document.getElementById('ai-tip-title').textContent,
        body: document.getElementById('ai-tip-body').textContent,
      };
      el.dispatchEvent(new PointerEvent('pointerout', { bubbles: true }));
    };
    for (const m of ['alive', 'can_kill', 'next_kill', 'next_death', 'duel']) {
      read(m, document.querySelector('.ai-col[data-metric="' + m + '"]'));
    }
    // 顶部两个胜率：悬停弹的是同一张 #ai-tip
    read('win_ct', document.getElementById('ai-win-ct'));
    read('win_t', document.getElementById('ai-win-t'));
    return out;
  })()`);
  const tipTitles = Object.values(help).map((h) => h.title);
  const tipBodies = Object.values(help).map((h) => (h.body || '').trim());
  check(`七项指标各有一句释义（${tipTitles.join(' / ')}）`,
    // 判的是"每项有**自己的**说法"（标题和正文都不能撞车）＋ 不是空串。
    // 别再按字数卡：这几句是"下一个死的会是他。"这种短句，按长度卡会把
    // 文案改短变成测试失败 —— 真正要防的是七项共用一句话。
    new Set(tipTitles).size === 7
      && new Set(tipBodies).size === 7
      && tipBodies.every((b) => b.length >= 4),
    tipBodies.map((b) => b.length).join('/'));
  check('悬停结束后释义卡自己收起',
    await evaluate(`!document.getElementById('ai-tip').classList.contains('show')`));

  // ── 2. 点名字 = 点左边卡片（选中 + 镜头跟随）─────────────────────────
  const jump = await evaluate(`(() => {
    const row = document.querySelector('#ai-table-body .ai-row');
    const p = Number(row.dataset.player);
    row.querySelector('.ai-row-name').click();
    // 高亮是每帧刷表格时贴上去的：手动走一帧，否则下面读到的是上一帧的 DOM
    window.__v5.stepFrame(0.001, false);
    const lit = document.querySelector('#ai-table-body .ai-row.selected');
    return {
      p,
      selected: window.__v5.selectedPlayer,
      follow: window.__v5.cameraFollow.player,
      selectedRow: lit ? Number(lit.dataset.player) : -1,
      detailOpen: !document.getElementById('player-detail-' + p).classList.contains('hidden'),
    };
  })()`);
  check(`点名字选中并跟随这个选手（#${jump.p}）`,
    jump.selected === jump.p && jump.follow === jump.p && jump.detailOpen,
    JSON.stringify(jump));
  check('点名字之后表格里那一行也亮起来', jump.selectedRow === jump.p, String(jump.selectedRow));

  // ── 3. 点数字 = 打开这一项整回合的曲线 ───────────────────────────────
  await evaluate(`(() => {
    document.querySelector('#ai-table-body .ai-row.selected .ai-cell[data-metric="alive"]').click();
  })()`);
  await sleep(160);
  const chart = await evaluate(`(() => {
    const st = window.__v5.ai.chart;
    const box = document.getElementById('ai-chart');
    const c = box.getBoundingClientRect();
    const t = document.getElementById('ai-table').getBoundingClientRect();
    const m = document.querySelector('.minimap-container').getBoundingClientRect();
    // 曲线卡绝不能压着小地图：第一版固定"表格正上方"，在 1440×900 上正好盖住它
    const overlap = Math.max(0, Math.min(c.right, m.right) - Math.max(c.left, m.left))
      * Math.max(0, Math.min(c.bottom, m.bottom) - Math.max(c.top, m.top));
    return {
      ...st,
      hidden: box.classList.contains('hidden'),
      label: document.getElementById('ai-chart-title').textContent,
      place: box.dataset.place,
      sameWidth: Math.abs(c.width - t.width) < 2,
      minimapOverlap: Math.round(overlap),
      inView: c.top >= 0 && c.left >= 0 && c.right <= innerWidth + 1,
    };
  })()`);
  check('点数字打开了曲线卡', chart.open === true && chart.hidden === false, JSON.stringify(chart));
  check('曲线卡画的是被点的那个人 / 那一项（存活）',
    chart.player === jump.p && chart.metric === 'alive' && chart.label.includes('存活'),
    JSON.stringify({ player: chart.player, metric: chart.metric, label: chart.label }));
  check(`曲线覆盖整回合（${chart.points}/${chart.samples} 个点）`,
    chart.samples === ai.samples.length && chart.points > 50);
  check(`canvas 真的画上了东西（非透明像素 ${chart.ink}）`, chart.ink > 2000, String(chart.ink));
  check('曲线卡不压小地图、宽度和表格一致、完整在视口里',
    chart.minimapOverlap === 0 && chart.sameWidth && chart.inView,
    JSON.stringify({ place: chart.place, overlap: chart.minimapOverlap,
                     same: chart.sameWidth, inView: chart.inView }));

  // 点曲线任意一点：跳时间 + 锁定这个选手 + 先暂停播放
  const seeked = await evaluate(`(() => {
    const c = document.getElementById('ai-chart-canvas');
    const r = c.getBoundingClientRect();
    const before = window.__v5.ai.chart;
    const f = 0.4;
    const x = r.left + 30 + (r.width - 40) * f;   // 左边留白 30、右边 10，见 ai-chart.js
    c.dispatchEvent(new PointerEvent('pointerdown', {
      bubbles: true, button: 0, clientX: x, clientY: r.top + r.height / 2,
    }));
    return {
      want: before.duration * f,
      got: window.__v5.currentTime,
      playing: window.__v5.isPlaying,
      follow: window.__v5.cameraFollow.player,
      selected: window.__v5.selectedPlayer,
      chart: window.__v5.ai.chart,
    };
  })()`);
  check(`点曲线跳到了那一点（想要 ${seeked.want.toFixed(1)} s，实际 ${seeked.got.toFixed(1)} s）`,
    Math.abs(seeked.got - seeked.want) < Math.max(0.5, seeked.chart.duration * 0.03));
  check('跳过去之后镜头锁定这个选手',
    seeked.follow === chart.player && seeked.selected === chart.player,
    JSON.stringify({ follow: seeked.follow, selected: seeked.selected }));
  check('点曲线会先暂停播放（不然刚跳到的画面立刻被冲掉）', seeked.playing === false);
  check('曲线上的播放头跟到了跳转点',
    Math.abs(seeked.chart.cursor - seeked.got) < 0.05,
    `${seeked.chart.cursor} vs ${seeked.got}`);

  // 悬停曲线：读数变成「悬停 12.4 s · 63%」
  const hover = await evaluate(`(() => {
    const c = document.getElementById('ai-chart-canvas');
    const r = c.getBoundingClientRect();
    const x = r.left + 30 + (r.width - 40) * 0.7;
    c.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: x }));
    return {
      hover: window.__v5.ai.chart.hover,
      now: document.getElementById('ai-chart-now').textContent,
    };
  })()`);
  check('悬停曲线显示那一刻的读数', /^悬停/.test(hover.now) && hover.hover > 0,
    JSON.stringify(hover));

  const chartShot = await send('Page.captureScreenshot', { format: 'png' });
  if (chartShot.result?.data) {
    writeFileSync(join(ROOT, '.build', 'ai_chart.png'),
      Buffer.from(chartShot.result.data, 'base64'));
    console.log(`  (截图: ${join(ROOT, '.build', 'ai_chart.png')})`);
  }

  // 对枪那一格没有曲线：点它不能把当前这张换成"整回合的对枪概率"
  const duelClick = await evaluate(`(() => {
    const before = window.__v5.ai.chart;
    document.querySelector('#ai-table-body .ai-row .ai-cell[data-metric="duel"]').click();
    return { before, after: window.__v5.ai.chart };
  })()`);
  check('点对枪那一格不会打开曲线（它只在死前 3 秒里有值）',
    duelClick.after.open === true && duelClick.after.metric === 'alive'
      && duelClick.after.player === duelClick.before.player);

  // 关掉曲线：✕ 和 Esc 两条路都要能关
  await evaluate(`document.getElementById('ai-chart-close').click()`);
  await sleep(120);
  const closed = await evaluate(`({
    open: window.__v5.ai.chart.open,
    hidden: document.getElementById('ai-chart').classList.contains('hidden'),
  })`);
  check('✕ 关掉曲线卡', closed.open === false && closed.hidden === true, JSON.stringify(closed));

  await evaluate(`document.querySelector('#ai-table-body .ai-row .ai-cell[data-metric="can_kill"]').click()`);
  await sleep(120);
  const reopened = await evaluate(`window.__v5.ai.chart.open`);
  await evaluate(`document.dispatchEvent(new KeyboardEvent('keydown', { code: 'Escape', bubbles: true }))`);
  await sleep(100);
  const escClosed = await evaluate(
    `!window.__v5.ai.chart.open && document.getElementById('ai-chart').classList.contains('hidden')`);
  check('Esc 也能关掉曲线卡', reopened === true && escClosed === true,
    JSON.stringify({ reopened, escClosed }));

  // ── 顶部「回合胜率」：同一张卡，有曲线、能跳时间、不抢镜头 ──────────────
  // 先选中并跟随一个人：胜率曲线不属于任何选手，点它只跳时间，不许把镜头
  // 从这个选手身上甩走。
  await evaluate(`(() => {
    const row = document.querySelector('#ai-table-body .ai-row');
    row.querySelector('.ai-row-name').click();
    window.__v5.stepFrame(0.001, false);
  })()`);
  await evaluate(`document.getElementById('ai-win-ct').click()`);
  await sleep(150);
  const winChart = await evaluate(`(() => {
    const st = window.__v5.ai.chart;
    return {
      ...st,
      hint: document.getElementById('ai-chart-hint').textContent,
      aria: document.getElementById('ai-chart').getAttribute('aria-label'),
      selected: window.__v5.selectedPlayer,
      follow: window.__v5.cameraFollow.player,
    };
  })()`);
  check('点顶部 CT 胜率打开了胜率曲线',
    winChart.open === true && winChart.metric === 'win_ct' && winChart.player === -1,
    JSON.stringify({ open: winChart.open, metric: winChart.metric, player: winChart.player }));
  check(`胜率曲线覆盖整回合（${winChart.points}/${winChart.samples} 个点）`,
    winChart.samples === ai.samples.length && winChart.points > 50,
    JSON.stringify({ samples: winChart.samples, want: ai.samples.length }));
  check('胜率曲线的标题 / 无障碍文案说的是「回合胜率 · CT」',
    winChart.title.includes('回合胜率') && winChart.title.includes('CT')
      && winChart.aria.includes('胜率'),
    JSON.stringify({ title: winChart.title, aria: winChart.aria }));
  check('胜率曲线没有「阵亡」标记（它是整支队伍的）', winChart.death === null,
    String(winChart.death));
  check('胜率曲线的底部提示不写"锁定选手"（它不属于某个人）',
    winChart.hint.includes('跳到那一刻') && winChart.hint.includes('不锁定'),
    winChart.hint);
  check(`canvas 也画上了胜率曲线（非透明像素 ${winChart.ink}）`, winChart.ink > 2000,
    String(winChart.ink));

  const winShot = await send('Page.captureScreenshot', { format: 'png' });
  if (winShot.result?.data) {
    writeFileSync(join(ROOT, '.build', 'ai_win_chart.png'),
      Buffer.from(winShot.result.data, 'base64'));
    console.log(`  (截图: ${join(ROOT, '.build', 'ai_win_chart.png')})`);
  }

  const winSeek = await evaluate(`(() => {
    const before = window.__v5.ai.chart;
    const c = document.getElementById('ai-chart-canvas');
    const r = c.getBoundingClientRect();
    const f = 0.7;
    const x = r.left + 30 + (r.width - 40) * f;   // 左边留白 30、右边 10
    c.dispatchEvent(new PointerEvent('pointerdown', {
      bubbles: true, button: 0, clientX: x, clientY: r.top + r.height / 2,
    }));
    return {
      want: before.duration * f,
      got: window.__v5.currentTime,
      playing: window.__v5.isPlaying,
      before: { selected: ${winChart.selected}, follow: ${winChart.follow} },
      selected: window.__v5.selectedPlayer,
      follow: window.__v5.cameraFollow.player,
      chart: window.__v5.ai.chart,
    };
  })()`);
  check(`点胜率曲线跳到了那一点（想要 ${winSeek.want.toFixed(1)} s，实际 ${winSeek.got.toFixed(1)} s）`,
    Math.abs(winSeek.got - winSeek.want) < Math.max(0.5, winSeek.chart.duration * 0.03));
  check('点胜率曲线也会暂停播放', winSeek.playing === false);
  check('点胜率曲线不抢镜头（选中 / 跟随都还是原来那个人）',
    winSeek.selected === winSeek.before.selected && winSeek.follow === winSeek.before.follow,
    JSON.stringify({ before: winSeek.before, selected: winSeek.selected, follow: winSeek.follow }));

  // CT / T 是同一条 win_ct 的两面：同一个时刻两个读数必须互补。
  const bothSides = await evaluate(`(() => {
    const read = () => ({
      pct: document.getElementById('ai-chart-now').textContent,
      metric: window.__v5.ai.chart.metric,
      title: window.__v5.ai.chart.title,
    });
    const ct = read();
    document.getElementById('ai-win-t').click();
    const t = read();
    return { ct, t, cursor: window.__v5.ai.chart.cursor };
  })()`);
  check('点 T 的数字换成 T 的那条曲线',
    bothSides.t.metric === 'win_t' && /回合胜率.*T/.test(bothSides.t.title),
    JSON.stringify(bothSides.t));
  {
    const num = (s) => {
      const m = String(s).match(/(\d+(?:\.\d+)?)%/);
      return m ? Number(m[1]) : NaN;
    };
    const ctVal = num(bothSides.ct.pct);
    const tVal = num(bothSides.t.pct);
    check(`同一时刻 CT + T = 100%（${bothSides.ct.pct} / ${bothSides.t.pct}）`,
      Number.isFinite(ctVal) && Number.isFinite(tVal)
        && Math.abs(ctVal + tVal - 100) <= 2,
      JSON.stringify(bothSides));
  }

  // 悬停顶部胜率：和表头共用同一张释义卡
  const winTip = await evaluate(`(() => {
    const el = document.getElementById('ai-win-t');
    el.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
    const tip = {
      shown: document.getElementById('ai-tip').classList.contains('show'),
      title: document.getElementById('ai-tip-title').textContent,
      hint: document.getElementById('ai-tip-hint').textContent,
    };
    el.dispatchEvent(new PointerEvent('pointerout', { bubbles: true }));
    return tip;
  })()`);
  check('悬停胜率也有释义（和表头是同一张卡）',
    winTip.shown && winTip.title.includes('T') && winTip.hint.includes('曲线'),
    JSON.stringify(winTip));

  // ── 选手 ID：头顶的选手名字牌，默认勾选 ────────────────────────────────
  const idsOn = await evaluate(`(() => ({
    checked: document.getElementById('toggle-player-ids').checked,
    ids: window.__v5.playerIds,
    // 这一局每个槽位真正的名字（取不到名字才退回 01–10 的槽位号）
    names: ((window.__v5.summary || {}).players || []).map((p) => (p && p.name) || ''),
    // 当前这一帧哪些人还活着：ID 牌只跟着活人画（死人的整组是隐藏的）
    alive: window.__v5.players.map((p) => !!(p && p.alive)),
  }))()`);
  check('「选手 ID」开关默认勾选', idsOn.checked === true);
  check(`10 个玩家各有一块 ID 牌、写的是名字（${idsOn.ids.map((d) => d.text).join(' ')}）`,
    idsOn.ids.length === 10
      && idsOn.names.filter(Boolean).length >= 8
      && idsOn.ids.every((d, i) => d.text === (idsOn.names[i] || String(i + 1).padStart(2, '0'))),
    JSON.stringify(idsOn.ids.slice(0, 3)));
  check('ID 牌是 Sprite（方向永远朝向摄像机）',
    idsOn.ids.every((d) => d.sprite), JSON.stringify(idsOn.ids.slice(0, 2)));
  check('ID 牌默认都挂出来、位置在头顶上方，且只跟着活人显示',
    idsOn.ids.every((d) => d.flagVisible && d.pos[1] > 1.6)
      && idsOn.ids.every((d, i) => d.visible === idsOn.alive[i]),
    JSON.stringify(idsOn.ids.map((d, i) => [d.text, d.visible, idsOn.alive[i]])
      .slice(0, 4)));
  check('长名字把牌子撑宽、高度不变（不会把字压扁）',
    idsOn.ids.every((d) => Math.abs(d.scale[1] - 0.22) < 1e-6)
      && idsOn.ids.some((d) => d.scale[0] > 0.5),
    JSON.stringify(idsOn.ids.map((d) => d.scale)));

  const idsOff = await evaluate(`(() => {
    const box = document.getElementById('toggle-player-ids');
    box.checked = false;
    box.dispatchEvent(new Event('change'));
    return window.__v5.playerIds;
  })()`);
  check('取消勾选后 ID 牌立刻全部收起',
    idsOff.every((d) => !d.flagVisible && !d.visible), JSON.stringify(idsOff.slice(0, 2)));

  await evaluate(`(() => {
    const box = document.getElementById('toggle-player-ids');
    box.checked = true;
    box.dispatchEvent(new Event('change'));
  })()`);
  check('重新勾上又全部出现',
    await evaluate(`window.__v5.playerIds.every((d) => d.flagVisible)`));

  // 朝摄像机的翻转不用测旋转：three 的 Sprite 在顶点着色器里用
  // billboard 矩阵，任何父节点旋转都不会改变屏幕上它的朝向。

  // 后面的截图要拍 3D 里的弧线和 ID 牌：先把曲线卡关掉，别让它压住画面
  await evaluate(`document.getElementById('ai-chart-close').click()`);


  // 清理：取消选中 / 停止跟随，后面的截图画的是干净的 3D 画面
  await evaluate(`(() => {
    if (window.__v5.selectedPlayer >= 0) window.__v5.selectPlayer(-1);
    window.__v5.cameraFollow.stop();
  })()`);

  // ── 进度条：强制重算一次，看它真的走起来 ────────────────────────────────
  // （第一次加载走的是服务端磁盘缓存，毫秒级，进度条一闪而过 —— 那也算正常，
  //   所以这条用一个真的重算来测：`ensureAi(true)` = 服务端跳过缓存。）
  const prog = await evaluate(`(async () => {
    const val = () => ({
      hidden: document.getElementById('ai-progress').classList.contains('hidden'),
      running: document.getElementById('ai-progress').classList.contains('is-running'),
      stage: document.getElementById('ai-progress-stage').textContent,
      time: document.getElementById('ai-progress-time').textContent,
      width: document.getElementById('ai-progress-fill').style.width,
      pct: window.__v5.ai.progress.target,
      analyzing: window.__v5.ai.analyzing,
    });
    const seen = [];
    window.__v5.ensureAi(true);
    // 预算 6 分钟：这是**真重算**，冷回合要重建特征 + 两两遮挡。实测这台机器
    // 在 headless Chrome 同时跑着 3D 的时候一次能到 ~180 s（磁盘缓存里记的
    // 61.8 s 是没那么忙的时候），60 s 的预算会假失败。
    for (let i = 0; i < 3600; i++) {
      await new Promise((r) => setTimeout(r, 100));
      // 进度条是**每帧**由 updateAiProgress() 更新 DOM 的，而 headless 里 rAF
      // 只有 1~3 fps：张量已经在本进程里缓存时（同一回合刚算过一次），这次重算
      // 只有 ~0.5 s，可能一帧都没出 —— 那就是"采到 0 帧"的假失败。手动推一帧
      // （dt 极小，不动回放位置、不画 GPU）再采样。
      window.__v5.stepFrame(1e-6, false);
      const v = val();
      if (v.analyzing && v.running) seen.push(v);
      if (!v.analyzing) break;
    }
    return { seen, end: val(), round: window.__v5.ai.round };
  })()`);

  check(`重算时进度条出现了（采到 ${prog.seen.length} 帧）`, prog.seen.length > 0);
  if (prog.seen.length) {
    const stages = Array.from(new Set(prog.seen.map((s) => s.stage)));
    check(`进度条报了多个阶段（${stages.slice(0, 3).join(' → ')}…）`, stages.length >= 2,
      JSON.stringify(stages.slice(0, 6)));
    const pcts = prog.seen.map((s) => s.pct);
    check('进度单调不倒退', pcts.every((p, i) => i === 0 || p >= pcts[i - 1] - 1e-6),
      JSON.stringify(pcts.slice(0, 8)));
    check(`进度真的往前走了（${Math.round(pcts[0] * 100)}% → ${Math.round(pcts[pcts.length - 1] * 100)}%）`,
      pcts[pcts.length - 1] > pcts[0] + 0.1, JSON.stringify(pcts.slice(-3)));
    const widths = prog.seen.map((s) => Number(s.width.replace('%', '')));
    check(`进度条的宽度跟着走（最大 ${Math.max(...widths).toFixed(1)}%）`,
      Math.max(...widths) > 20, String(Math.max(...widths)));
    check('右下角报"已用时"', /s$/.test(prog.seen[prog.seen.length - 1].time),
      prog.seen[prog.seen.length - 1].time);
    check('百分比写在进度条旁边', /%/.test(prog.seen[prog.seen.length - 1].time),
      prog.seen[prog.seen.length - 1].time);
  }
  check('算完进度条收起', prog.end.hidden === true && prog.end.analyzing === false,
    JSON.stringify(prog.end));
  check('这次是真算的（服务端没走缓存）', prog.round && prog.round.cached === false,
    JSON.stringify(prog.round));


  // 把机位摆到这两个人旁边再截图：默认机位是"整张地图装进画面"，人物只有几个像素，
  // 弧线在画面外/太小都看不出效果。three 坐标 = (y, z, x) × 0.0254（见 scene.js）。
  // 先跳回**那一帧**：前面点过胜率曲线（seek 到 70%），那时早过了死前 3 秒的窗口，
  // 弧线已经不在，直接摆机位只会拍到一面墙。
  const focus = await evaluate(`(() => {
    window.__v5.seekSample(${hit.k * stride});
    const pl = window.__v5.players;
    const ga = pl[${frame.duel.a}], gb = pl[${frame.duel.b}];
    if (!ga || !gb) return null;
    const S = 0.0254;
    const ta = [ga.y * S, ga.z * S, ga.x * S];
    const tb = [gb.y * S, gb.z * S, gb.x * S];
    const mid = [(ta[0] + tb[0]) / 2, (ta[1] + tb[1]) / 2 + 1.6, (ta[2] + tb[2]) / 2];
    // 机位放在 A→B 连线的**垂直方向**上（而不是随便挑一个角）：弧线是竖直抬起
    // 的二次贝塞尔，从侧面看最舒展；顺着连线看会被压成一条直线。
    const span = Math.hypot(tb[0] - ta[0], tb[2] - ta[2]);
    const nx = span > 0.01 ? -(tb[2] - ta[2]) / span : 1;
    const nz = span > 0.01 ? (tb[0] - ta[0]) / span : 0;
    const d = Math.max(4, span * 0.75 + 4);
    const cam = window.__v5.camera, ctl = window.__v5.controls;
    window.__v5.setCameraMode('orbit');
    window.__v5.setFocusedPlayer(-1);
    cam.position.set(mid[0] + nx * d, mid[1] + d * 0.35, mid[2] + nz * d);
    ctl.target.set(mid[0], mid[1], mid[2]);
    ctl.update();
    // 截图前停一下回放：否则这 500 ms 里采样点会往后走，弧线跟着人一起出画
    if (window.__v5.isPlaying) document.getElementById('btn-play').click();
    return { mid, d };
  })()`);
  check('能摆一个看得到这两个人的机位', !!focus);
  await evaluate(`window.__v5.stepFrame(0.016, true)`);
  await sleep(500);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  if (shot.result?.data) {
    writeFileSync(SHOT, Buffer.from(shot.result.data, 'base64'));
    console.log(`  (截图: ${SHOT})`);
  }
}

// 换回合：上一份预测要作废、新回合要自己拉下来（服务端按回合缓存）。
// 等待预算给到 5 分钟：冷回合在这台机器上能算到 ~180 s（见上面重算那段的注释），
// 80 s 的预算会假失败 —— 这里只是等，不改变任何断言。
await evaluate(`(() => {
  const sel = document.getElementById('round-select');
  sel.value = '1';
  sel.dispatchEvent(new Event('change'));
})()`);
let switched = false;
for (let i = 0; i < 1200 && !switched; i++) {
  await sleep(250);
  switched = await evaluate(`(() => {
    const ai = window.__v5.ai;
    return ai.activeRound === 1 && ai.loadedFor === 1 && !!ai.round && ai.round.round === 1;
  })()`).catch(() => false);
}
check('换回合后自动拉新回合的预测（并丢掉旧的）', switched);

// 关掉开关 → 立刻隐藏
await evaluate(`(() => {
  for (const id of ['toggle-ai-spatial', 'toggle-ai-duel']) {
    const el = document.getElementById(id);
    el.checked = false;
    el.dispatchEvent(new Event('change'));
  }
})()`);
await sleep(300);
const off = await evaluate(`(() => {
  const o = window.__v5.ai.overlay;
  return { duel: o.duelVisible,
           hidden: document.getElementById('ai-panel').classList.contains('hidden'),
           tableHidden: document.getElementById('ai-table').classList.contains('hidden'),
           chartHidden: document.getElementById('ai-chart').classList.contains('hidden'),
           tipShown: document.getElementById('ai-tip').classList.contains('show') };
})()`);
check('关掉开关后 3D 层隐藏、面板 / 表格 / 曲线都收起',
  off.duel === false && off.hidden === true && off.tableHidden === true
    && off.chartHidden === true && off.tipShown === false,
  JSON.stringify(off));
check('全程没有 JS 异常', errors.length === 0, errors.slice(0, 3).join(' | '));

ws.close();
chrome.kill();
console.log(`\n${failed === 0 ? '全部通过' : `${failed} 项失败`}（${passed} passed, ${failed} failed）`);
process.exit(failed === 0 ? 0 : 1);
