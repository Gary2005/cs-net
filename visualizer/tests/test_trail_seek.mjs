#!/usr/bin/env node
/**
 * test_trail_seek.mjs — 「拖时间轴之后，轨迹不许把跳转前后的位置连起来」（headless Chrome + CDP）。
 *
 * 报告过的现象：拖进度条，3D 视图和**小地图**里的玩家轨迹都"瞬移"。
 * 根因是轨迹以前是个**每帧追加**的环形缓冲，而 `seekTo()` 只改 `currentSampleIdx`，
 * 不碰缓冲 —— 于是跳转后第一帧把"跳转前的位置"和"跳转后的位置"连成一条横穿地图
 * 的长线（de_inferno 实测：正常每采样点 0.34 m，跳转后出现 25 m / 38 m 的段，
 * 整条轨迹长度从 26 m 涨到 279 m）。小地图渲染的是同一个场景，所以两边一起坏。
 *
 * 修法：轨迹改成**采样点的纯函数** —— 连续 +1 就追加（便宜），不连续就按历史
 * 整条重算；两路都严格是"最近 150 个采样点，一个采样点一个点"。这个测试量的就是
 * 那条不变量，全部用**画出来的几何**（`__v5.trails` 读的是 BufferGeometry 里的点），
 * 参考值在 node 侧从原始采样点独立算，不让被测代码给自己打分：
 *
 *   A. 播到位 → 跳走 → 再跳回来：三条轨迹必须**逐点相同**（这就是"瞬移"的反面）；
 *   B. 跳到任意采样点 k：轨迹必须逐点等于 [k-149, k] 的历史重算结果；
 *   C. 任何一段都不能长过"正常每采样点位移"——跳转产生的长段直接抓出来；
 *   D. 关掉「轨迹」开关、播一段、再打开：仍然等于历史重算结果（另一条触发路径）；
 *   E. 之后小地图还能画（`redrawMinimap()` 非空），没有控制台报错；
 *   F. 用 CDP 真的按住鼠标拖时间线（25% → 85%），再验一遍播放头和轨迹。
 *
 * 用法（需要 Chrome/Edge 和一个已经在跑的 server）：
 *
 *     node visualizer/tests/test_trail_seek.mjs http://127.0.0.1:5099 \
 *          app/data/csz/aurora-vs-vitality-m2-inferno__ca728317.csz
 *
 * 没找到 Chrome 就打印 SKIP 并以 0 退出。
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
].filter(Boolean);

const BASE = (process.argv[2] || 'http://127.0.0.1:5099').replace(/\/$/, '');
const CSZ = process.argv[3] || 'app/data/csz/aurora-vs-vitality-m2-inferno__ca728317.csz';
const PORT = 9333 + (process.pid % 200);

/** scene.js 的 SCALE（游戏单位 → three 单位），测试自己算一遍，不复用被测代码 */
const SCALE = 0.0254;
/** visuals.js 的轨迹窗口（最近这么多个采样点），同样独立写一遍 */
const MAX_TRAIL_POINTS = 150;

const chrome = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chrome) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
function check(name, cond, detail = '') {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? `   ${detail}` : ''}`);
  if (!cond) failed++;
}

const profile = mkdtempSync(join(tmpdir(), 'trail-seek-'));
const proc = spawn(chrome, [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check',
  '--disable-gpu-sandbox', '--no-sandbox',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1280,800',
  'about:blank',
// 三个 stdio 都 'ignore'：不读 Chrome 的管道，采集只走 CDP 的 WebSocket
], { stdio: ['ignore', 'ignore', 'ignore'] });

let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  await sleep(250);
  try {
    const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
    const list = await res.json();
    wsUrl = list.find((t) => t.type === 'page')?.webSocketDebuggerUrl || null;
  } catch { /* not up yet */ }
}
if (!wsUrl) {
  console.log('SKIP: could not reach Chrome DevTools');
  proc.kill();
  process.exit(0);
}

const ws = new WebSocket(wsUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

let msgId = 0;
const pending = new Map();
const consoleErrors = [];
const pageErrors = [];

ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    consoleErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
  }
  if (m.method === 'Runtime.exceptionThrown') {
    const d = m.params.exceptionDetails;
    pageErrors.push(d.exception?.description || d.text);
  }
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
    const e = m.params.entry;
    // 浏览器自己请求的 /favicon.ico 不算应用的错
    if (!/favicon\.ico/.test(e.url || '')) consoleErrors.push(`${e.text} ${e.url || ''}`);
  }
};

function send(method, params = {}) {
  const id = ++msgId;
  return new Promise((res) => { pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
}

async function evaluate(expr, awaitPromise = false) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise });
  if (r.result?.exceptionDetails) {
    throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  }
  return r.result?.result?.value;
}

// ── 参考实现（node 侧，独立于被测代码）──────────────────────────────────────

/**
 * `[k-149, k]` 这些采样点里某个玩家的轨迹：世界坐标 = (y, z, x)·SCALE，
 * 一个采样点一个点（窗口按采样点切，所以点数 = min(k+1, 150)）。
 */
function reference(pl, k) {
  const from = Math.max(0, k - (MAX_TRAIL_POINTS - 1));
  const pts = [];
  for (let j = from; j <= k; j++) pts.push([pl.y[j] * SCALE, pl.z[j] * SCALE, pl.x[j] * SCALE]);
  return pts;
}

/** 同一窗口里相邻采样点之间的最大位移（"正常能有多大一段"的上界） */
function maxStep(pl, k) {
  const from = Math.max(1, k - (MAX_TRAIL_POINTS - 1));
  let m = 0;
  for (let j = from; j <= k; j++) {
    const d = Math.hypot(pl.y[j] - pl.y[j - 1], pl.z[j] - pl.z[j - 1], pl.x[j] - pl.x[j - 1]) * SCALE;
    if (d > m) m = d;
  }
  return m;
}

const maxSeg = (pts) => {
  let m = 0;
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]);
    if (d > m) m = d;
  }
  return m;
};

/** 逐点比较（缓冲是 Float32Array，留 1e-4 给 float32 舍入） */
function samePoints(a, b, tol = 1e-4) {
  if (a.length !== b.length) return `点数 ${a.length} ≠ ${b.length}`;
  for (let i = 0; i < a.length; i++) {
    for (let c = 0; c < 3; c++) {
      if (Math.abs(a[i][c] - b[i][c]) > tol) {
        return `第 ${i} 点差 ${Math.abs(a[i][c] - b[i][c]).toExponential(2)}（${a[i]} vs ${b[i]}）`;
      }
    }
  }
  return null;
}

// ── 页面侧的小工具 ──────────────────────────────────────────────────────────

/** 轨迹：读画出来的折线（`__v5.trails` 直接来自 BufferGeometry） */
const readTrail = (i) => `(window.__v5.trails[${i}] || null)`;

/** 一个 JS 任务里按固定 dt 推到 `target` 采样点，返回后一定是暂停状态 */
const stepTo = (target) => `(() => {
  const v = window.__v5;
  if (!v.isPlaying) document.getElementById('btn-play').click();
  let guard = 0;
  while (v.sampleIdx < ${target} && guard++ < 40000) v.stepFrame(1 / 64, false);
  if (v.isPlaying) document.getElementById('btn-play').click();
  return { sample: v.sampleIdx, guard };
})()`;

/** 跳到某个采样点（等价于在时间轴上点一下），并立刻同步一次视觉层 */
const seekSample = (k) => `(() => { window.__v5.seekSample(${k}); return window.__v5.sampleIdx; })()`;

try {
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: `${BASE}/replay` });

  let booted = false;
  for (let i = 0; i < 60 && !booted; i++) {
    await sleep(500);
    booted = await evaluate('!!window.__v5').catch(() => false);
  }
  console.log(`chrome: ${chrome}`);
  console.log(`page  : ${BASE}/replay`);
  console.log(`csz   : ${CSZ}\n`);
  check('the page booted the app', booted);
  if (!booted) throw new Error('page did not boot');

  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
  let roundLoaded = false;
  for (let i = 0; i < 80 && !roundLoaded; i++) {
    roundLoaded = await evaluate('!!(window.__v5.round && window.__v5.round.players.length)');
    if (!roundLoaded) await sleep(400);
  }
  check('a round is loaded', roundLoaded);
  if (!roundLoaded) throw new Error('round did not load');

  // ── 挑一个一直在跑的人 ───────────────────────────────────────────────────
  const pick = await evaluate(`(() => {
    const r = window.__v5.round, n = r.ticks.length, K = 260;
    let best = null;
    for (let p = 0; p < r.players.length; p++) {
      const pl = r.players[p];
      for (let s = 100; s + K < n; s += 17) {
        let d = 0, good = true;
        for (let k = 0; k < K; k++) {
          if (!pl.alive[s + k]) { good = false; break; }
          d += Math.hypot(pl.x[s+k+1]-pl.x[s+k], pl.y[s+k+1]-pl.y[s+k], pl.z[s+k+1]-pl.z[s+k]);
        }
        if (good && (!best || d > best.d)) best = { p, s, d: Math.round(d) };
      }
    }
    return best;
  })()`);
  check('found a player who keeps running', !!pick, pick ? `P${pick.p} @${pick.s}` : '');
  if (!pick) throw new Error('no moving player');
  const P = pick.p;
  const T = pick.s + 240;

  // 这个人的采样点（参考实现自己算，不用测试钩子）
  const pl = await evaluate(`(() => {
    const p = window.__v5.round.players[${P}];
    return { x: Array.from(p.x), y: Array.from(p.y), z: Array.from(p.z) };
  })()`);

  console.log('\nA. 播到位 → 跳走 → 跳回来');
  const stepped = await evaluate(stepTo(T));
  const TA = stepped.sample;             // 一次推进可能多走一个采样点，以真实落地为准
  const atT = await evaluate(readTrail(P));
  const refTA = reference(pl, TA);
  // 播出来的这条必须和"按历史重算"逐点相同（追加路径本身要对）
  const dPlay = samePoints(atT.pts, refTA);
  check('播放到位后的轨迹 = 历史重算', !dPlay, dPlay || `${atT.count} 点全等`);
  // 末端必须正好是这个人现在的脚底（轨迹是"尾迹"，不能拖在别处）
  const tail = atT.pts[atT.pts.length - 1];
  const feet = [pl.y[TA] * SCALE, pl.z[TA] * SCALE, pl.x[TA] * SCALE];
  const tailGap = Math.hypot(tail[0] - feet[0], tail[1] - feet[1], tail[2] - feet[2]);
  check('轨迹末端落在脚底', tailGap < 1e-4, `gap=${tailGap.toExponential(2)} m`);

  await evaluate(seekSample(TA - 40));
  const back = await evaluate(readTrail(P));
  await evaluate(seekSample(TA));
  const again = await evaluate(readTrail(P));
  // 跳回来是**重算**，必须和参考值逐点相同
  const dRef = samePoints(again.pts, refTA);
  check('跳回来之后 = 历史重算（逐点）', !dRef, dRef || `${again.count} 点全等`);
  const diff = samePoints(again.pts, atT.pts);
  check('跳走再跳回来，轨迹逐点不变', !diff, diff || `${again.count} 点全等`);
  check('（顺便）往回跳的那条也不含长段', maxSeg(back.pts) <= maxStep(pl, TA - 40) + 0.02,
    `maxSeg=${maxSeg(back.pts).toFixed(3)} 上界=${(maxStep(pl, TA - 40) + 0.02).toFixed(3)}`);

  console.log('\nB. 跳到任意采样点 = 按历史重算');
  for (const k of [TA - 120, TA + 90, pick.s + 5, TA - 1, TA + 200]) {
    const landed = await evaluate(seekSample(k));
    const tr = await evaluate(readTrail(P));
    const ref = reference(pl, landed);
    const d = samePoints(tr.pts, ref);
    check(`k=${landed}（${ref.length} 点）`, !d && landed === k, d || '逐点相同');
  }

  console.log('\nC. 没有"瞬移段"');
  for (const k of [TA - 100, TA + 60, TA - 200]) {
    await evaluate(seekSample(Math.max(0, k)));
    const tr = await evaluate(readTrail(P));
    const seg = maxSeg(tr.pts);
    const bound = maxStep(pl, Math.max(0, k)) + 0.02;   // 段长 = 相邻采样点位移，留点余量给浮点
    check(`k=${k} 最长一段 ${seg.toFixed(3)} m ≤ ${bound.toFixed(3)} m`, seg <= bound);
  }

  console.log('\nD. 关掉「轨迹」开关再打开');
  await evaluate(`document.getElementById('toggle-trails').click()`);
  const off = await evaluate(`document.getElementById('toggle-trails').checked`);
  check('开关已关闭', off === false);
  await evaluate(stepTo(TA + 60));
  await evaluate(`document.getElementById('toggle-trails').click()`);
  // 打开开关本身不画，逼一帧视觉层同步（真实使用里主循环下一帧就做了）
  await evaluate('window.__v5.stepFrame(0, false)');
  const resumed = await evaluate(readTrail(P));
  const backOn = await evaluate('window.__v5.sampleIdx');
  const dResume = samePoints(resumed.pts, reference(pl, backOn));
  check(`重新打开后 = 历史重算（k=${backOn}）`, !dResume, dResume || `${resumed.count} 点全等`);

  console.log('\nE. 小地图与报错');
  const mini = await evaluate('!!window.__v5.redrawMinimap()');
  check('小地图还能画（redrawMinimap 非空）', mini);

  console.log('\nF. 真的用指针拖时间线（端起鼠标按住的场景）');
  // 先回一个靠前的采样点，然后从 25% 拖到 85%
  await evaluate(seekSample(Math.max(0, TA - 200)));
  const track = await evaluate(`(() => {
    const r = document.getElementById('timeline-canvas').getBoundingClientRect();
    return { left: r.left, top: r.top + r.height / 2, width: r.width };
  })()`);
  const at = (f) => Math.round(track.left + track.width * f);
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: at(0.25), y: track.top, button: 'left', clickCount: 1 });
  for (const f of [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]) {
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: at(f), y: track.top, button: 'left', buttons: 1 });
    await sleep(40);
  }
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: at(0.85), y: track.top, button: 'left', clickCount: 1 });
  await sleep(200);
  const dragged = await evaluate('({ sample: window.__v5.sampleIdx, time: window.__v5.currentTime, dur: window.__v5.round.round_seconds[window.__v5.round.round_seconds.length - 1] })');
  check('拖动把播放头带到了 85% 附近', Math.abs(dragged.time / dragged.dur - 0.85) < 0.02,
    `time=${dragged.time.toFixed(2)}s / ${dragged.dur.toFixed(2)}s = ${(dragged.time / dragged.dur * 100).toFixed(1)}%`);
  check('拖动途中暂停了播放', (await evaluate('window.__v5.isPlaying')) === false);
  const dragTrail = await evaluate(readTrail(P));
  const dDrag = samePoints(dragTrail.pts, reference(pl, dragged.sample));
  check('拖完的轨迹 = 历史重算（没有横穿地图的长段）', !dDrag, dDrag || `${dragTrail.count} 点全等`);
  check('拖完最长一段仍在正常范围内', maxSeg(dragTrail.pts) <= maxStep(pl, dragged.sample) + 0.02,
    `maxSeg=${maxSeg(dragTrail.pts).toFixed(3)} 上界=${(maxStep(pl, dragged.sample) + 0.02).toFixed(3)}`);

  check('no page exceptions', pageErrors.length === 0, pageErrors.join(' | '));
  check('no console errors', consoleErrors.length === 0, consoleErrors.join(' | '));

  console.log(`\n${failed === 0 ? 'PASS' : `FAIL (${failed})`}: trail seek`);
} finally {
  try { ws.close(); } catch { /* ignore */ }
  proc.kill();
  await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* ignore */ }
}

process.exit(failed === 0 ? 0 : 1);
