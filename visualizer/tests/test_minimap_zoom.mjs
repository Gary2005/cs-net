/**
 * test_minimap_zoom.mjs — 小地图的三件事：
 *
 *   1. **炸弹点比玩家箭头小**。它标的只是 C4 的位置，之前 15px 的红饼会把落点
 *      附近的人和掩体整个盖住；
 *   2. **点一下放大成 2D 回放**。放大的是同一块画布、同一个正交相机，只是容器
 *      变大、后备分辨率跟着涨；Esc / 再点一次 / ✕ 都能收起，窗口尺寸变了还得是
 *      正方形、不许超出视口。放大态的取景**锁死成整张地图**（不跟人缩放平移），
 *      标记也换成小一号的固定尺寸 —— 所以这里还要验：推进回放后相机一动不动，
 *      而且取景宽度就是雷达图的宽度；
 *   3. **小地图里的烟被压暗，3D 场景那份不动**。烟和 3D 场景共用同一套材质，
 *      压暗只在「小地图那一次绘制」里生效（`registerMinimapPass`），这里在小地图
 *      渲染的那一刻把材质颜色抓出来比：画的时候是暗的，画完必须还原。
 *   4. **箭头旁边的名字能点**。放大态每个存活箭头旁边写着名字（按队伍色），
 *      点箭头或点名字 = 点他左边那张卡（选中 + 镜头跟随 + 卡片展开），点空白处才是收起。
 *
 * 尺寸都是**量像素**验的：把 `#minimap-overlay` 读出来，找红色斑点的包围盒，
 * 再对上 `__v5.minimap.glyph` 报的显示尺寸 —— 只断言常量的话，量不到"画出来的
 * 东西到底多大"。
 *
 * 用法：
 *     node visualizer/tests/test_minimap_zoom.mjs <server-url> <match.csz> [shot.png]
 *
 * 需要先起服务（`python -m app.server --port 5099 --no-watch`）；脚本自己
 * POST /api/load 把这场比赛装进会话。
 */

import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const BASE = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3];
const SHOT = process.argv[4] || join(ROOT, '.build', 'minimap_zoom.png');

if (!CSZ) {
  console.error('用法: node visualizer/tests/test_minimap_zoom.mjs <server-url> <match.csz> [shot.png]');
  process.exit(2);
}
const CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let passed = 0, failed = 0, skipped = 0;
const check = (name, ok, detail = '') => {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
};
const skip = (name, why) => { skipped++; console.log(`  SKIP ${name}   ${why}`); };

// ── 让服务端把这场比赛的会话建起来 ──────────────────────────────────────────
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

// ── 起 Chrome + 接 CDP ──────────────────────────────────────────────────────
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const profile = mkdtempSync(join(tmpdir(), 'mmzoom-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1600,900', 'about:blank',
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
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) {
    throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 600));
  }
  return r.result?.result?.value;
};

const SHOT_SMALL = SHOT.replace(/\.png$/i, '_small.png');
const screenshot = async (file, clip) => {
  const r = await send('Page.captureScreenshot', clip
    ? { format: 'png', clip: { ...clip, scale: 2 } }
    : { format: 'png' });
  if (!r.result?.data) return false;
  writeFileSync(file, Buffer.from(r.result.data, 'base64'));
  return true;
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
      && !!window.__v5 && !!window.__v5.round
      && !!document.getElementById('player-card-0');
  })()`).catch(() => false);
}
check('回放页起来了（#app 可见 + 玩家卡片 + 回合数据）', ready);
if (!ready) {
  console.log('SKIP: 页面没起来，后面的检查没法做');
  chrome.kill();
  process.exit(1);
}
await sleep(600);   // 让雷达底图和第一帧小地图画出来

const mm = () => evaluate(`window.__v5.minimap`);

/** 读 2D 叠加层：红色斑点（炸弹）的包围盒 + 箭头像素数，长度都换算成 CSS 像素 */
const probe = () => evaluate(`(() => {
  const cv = document.getElementById('minimap-overlay');
  const W = cv.width, H = cv.height;
  const d = cv.getContext('2d').getImageData(0, 0, W, H).data;
  let n = 0, minX = 1e9, maxX = -1, minY = 1e9, maxY = -1, arrows = 0;
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const i = (row + x) * 4;
      if (d[i + 3] < 160) continue;
      const r = d[i], g = d[i + 1], b = d[i + 2];
      if (r > 90 && g * 2 < r && b * 2 < r) {            // 炸弹 #ff2b2b（抗锯齿的边缘也算）
        n++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      } else if ((b > 170 && r < 130) || (r > 190 && g > 120 && b < 110)) {
        arrows++;                                        // CT #5b9bd5 / T #e8a240
      }
    }
  }
  const rect = cv.getBoundingClientRect();
  const k = rect.width / W;                              // CSS px / 后备 px
  return {
    backing: [W, H],
    css: [Math.round(rect.width), Math.round(rect.height)],
    bombPx: n,
    arrows,
    bomb: n ? {
      w: Math.round((maxX - minX + 1) * k * 100) / 100,
      h: Math.round((maxY - minY + 1) * k * 100) / 100,
    } : null,
  };
})()`);

// ── 1. 初始：角落那块 200px 的小地图 ───────────────────────────────────────
const m0 = await mm();
check('小地图初始是 200×200（CSS 像素）',
  Math.abs(m0.size[0] - 200) <= 1 && Math.abs(m0.size[1] - 200) <= 1, JSON.stringify(m0.size));
check('后备画布 256，叠加层同尺寸',
  m0.canvas === 256 && m0.overlay === 256, `${m0.canvas}/${m0.overlay}`);
check('标记尺寸：箭头 14px、炸弹 7px（炸弹比箭头小）',
  m0.glyph && m0.glyph.mode === 'small' && m0.glyph.arrow === 14 && m0.glyph.bomb === 7,
  JSON.stringify(m0.glyph));

/** 正交相机的取景框：`[left, right, bottom, top]`（水平 = z、垂直 = x） */
const camBox = () => evaluate(`(() => {
  const c = window.__v5.minimapCamera;
  return [c.left, c.right, c.bottom, c.top];
})()`);
/** 整张雷达图铺满时的取景宽度，和 scene.js 里 2D 回放的算法同一条式子 */
const fullSpan = () => evaluate(`(() => {
  const r = window.__v5.radar;
  if (!r) return null;
  return Math.max(r.u1 - r.u0, r.v1 - r.v0) * 0.0254 * 1.02;
})()`);

const smallCam = await camBox();
const smallSpan = smallCam[1] - smallCam[0];

// 找一个「C4 在地上」的采样点：只有那时才有炸弹标记可量
const bombSeek = await evaluate(`(() => {
  const r = window.__v5.round;
  const bp = r.bomb_position || [];
  for (let i = 0; i < bp.length / 3; i++) {
    if (!Number.isNaN(bp[i * 3]) && !Number.isNaN(bp[i * 3 + 1])) {
      window.__v5.seekSample(i);
      return { sample: i, planted: !!r.bomb_planted[i] };
    }
  }
  return null;
})()`);

let small = null;
if (bombSeek) {
  await evaluate(`window.__v5.redrawMinimap()`);
  await sleep(120);
  small = await probe();
  check('小地图上有玩家箭头',
    small.arrows > 40, `${small.arrows} px`);
  check('炸弹点的红色实心直径 ≈ 5~7px（旧的 15px 大红饼只剩这么点）',
    // 量的是**实心部分**：1px 的深色描边描在边上 + 抗锯齿，所以比标称直径小一圈。
    !!small.bomb && small.bomb.w >= 4.5 && small.bomb.w <= 7
    && Math.abs(small.bomb.w - small.bomb.h) <= 2,
    JSON.stringify(small.bomb));
  check('炸弹点在像素上也确实比箭头小',
    !!small.bomb && small.bomb.w < m0.glyph.arrow, `${small.bomb?.w} vs ${m0.glyph.arrow}`);
  await screenshot(SHOT_SMALL, await evaluate(`(() => {
    const r = document.getElementById('minimap-container').getBoundingClientRect();
    return { x: r.left - 6, y: r.top - 6, width: r.width + 12, height: r.height + 12 };
  })()`));
} else {
  skip('炸弹点尺寸（像素）', '本回合的 C4 一直在人身上，没有落地点可量');
}

// ── 2. 点一下 → 放大成 2D 回放 ─────────────────────────────────────────────
const clickAt = async (x, y) => {
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
};
const mmCenter = () => evaluate(`(() => {
  const r = document.getElementById('minimap-container').getBoundingClientRect();
  return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) };
})()`);

const c0 = await mmCenter();
await clickAt(c0.x, c0.y);
await sleep(250);
await evaluate(`window.__v5.redrawMinimap()`);

const m1 = await mm();
const vp0 = await evaluate(`(() => {
  const v = document.getElementById('viewport').getBoundingClientRect();
  return { w: Math.round(v.width), h: Math.round(v.height) };
})()`);
const wantSide = Math.max(240, Math.min(vp0.w, vp0.h) - 32);
check('点一下小地图 → 进入放大态',
  m1.expanded === true && m1.size[0] > 400, `${m1.expanded} ${JSON.stringify(m1.size)}`);
check('放大态是正方形，边长 = 视口短边 - 32',
  m1.size[0] === m1.size[1] && Math.abs(m1.size[0] - wantSide) <= 1,
  `${JSON.stringify(m1.size)} 期望 ${wantSide}`);
check('放大后后备分辨率跟着涨（不再是 256）',
  m1.canvas > 400 && m1.overlay === m1.canvas, `${m1.canvas}/${m1.overlay}`);
check('放大态标记固定为箭头 10px、炸弹 5px（不跟着面板长大）',
  m1.glyph && m1.glyph.mode === '2d' && m1.glyph.arrow === 10 && m1.glyph.bomb === 5,
  JSON.stringify(m1.glyph));

const bigCam = await camBox();
const wantFull = await fullSpan();
check('2D 回放的取景 = 整张雷达图（不是存活玩家的包围盒）',
  wantFull != null && Math.abs((bigCam[1] - bigCam[0]) - wantFull) <= wantFull * 0.01,
  `${(bigCam[1] - bigCam[0]).toFixed(3)} vs 期望 ${wantFull?.toFixed(3)}`);
check('2D 回放和角落小地图的取景不是同一套',
  Math.abs((bigCam[1] - bigCam[0]) - smallSpan) > 1e-6,
  `${(bigCam[1] - bigCam[0]).toFixed(3)} vs ${smallSpan.toFixed(3)}`);

if (bombSeek) {
  const big = await probe();
  check('放大后箭头还在画（叠加层跟着换分辨率）',
    big.arrows > 80 && big.backing[0] === m1.canvas, `${big.arrows} px @ ${big.backing[0]}`);
  check('放大后炸弹点依然比箭头小',
    !!big.bomb && big.bomb.w < m1.glyph.arrow,
    `${big.bomb?.w} vs ${m1.glyph.arrow}`);
  check('放大后炸弹点仍是小点（标称 5px，实心部分再小一圈描边）',
    // 炸弹有 ±8% 的呼吸缩放，量到的直径在 4.6~5.4 之间摆，不能卡死
    !!big.bomb && big.bomb.w >= 2.2 && big.bomb.w <= 5.8
    && Math.abs(big.bomb.w - m1.glyph.bomb) <= 2.2
    && Math.abs(big.bomb.w - big.bomb.h) <= 1.5,
    `${big.bomb?.w}×${big.bomb?.h} vs ${m1.glyph.bomb}`);
} else {
  skip('放大后的炸弹像素', '没有落地的 C4');
}

// 推进两秒（人会动、也会死人）再重画：2D 回放的取景应该一动不动
await evaluate(`(() => { window.__v5.stepReplay(2.0); window.__v5.redrawMinimap(); return true; })()`);
const bigCam2 = await camBox();
check('2D 回放：推进两秒 + 重画后取景一动不动（不缩放不平移）',
  bigCam2.every((v, i) => Math.abs(v - bigCam[i]) < 1e-9),
  `${JSON.stringify(bigCam)} → ${JSON.stringify(bigCam2)}`);

// ── 2b. 箭头旁边的名字 + 点箭头跳转 ───────────────────────────────────────
const hits0 = (await mm()).hits;
check('放大态每个存活箭头都带名字',
  hits0.length > 0 && hits0.every((h) => h.name && h.name.length > 0),
  `${hits0.length} 个: ${hits0.slice(0, 3).map((h) => h.name).join(' / ')}`);

// 名字得真的画出来，不能只是记了个热区：按名字那块矩形读叠加层的像素，数队伍色
const ink = await evaluate(`(() => {
  const v5 = window.__v5;
  const h = v5.minimap.hits[0];
  if (!h) return null;
  const cv = document.getElementById('minimap-overlay');
  const ui = cv.width / cv.getBoundingClientRect().width;
  const x0 = Math.max(0, Math.floor(h.nx0 * ui));
  const x1 = Math.min(cv.width, Math.ceil(h.nx1 * ui));
  const y0 = Math.max(0, Math.floor(h.ny0 * ui));
  const y1 = Math.min(cv.height, Math.ceil(h.ny1 * ui));
  const d = cv.getContext('2d').getImageData(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)).data;
  const t = v5.round.teams[h.idx] === 'CT' ? [207, 228, 247] : [255, 215, 168];
  let n = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] < 120) continue;
    const dr = d[i] - t[0], dg = d[i + 1] - t[1], db = d[i + 2] - t[2];
    if (dr * dr + dg * dg + db * db < 2025) n++;      // 45²：贴边的抗锯齿像素不算
  }
  return { name: h.name, ink: n };
})()`);
check('名字真的画出来了（按队伍色数到了笔画）',
  !!ink && ink.ink >= 12, ink ? `${ink.name}: ${ink.ink} px` : '没有热区');

const overlayOrigin = () => evaluate(`(() => {
  const r = document.getElementById('minimap-overlay').getBoundingClientRect();
  return { x: r.left, y: r.top };
})()`);

// 点第一个箭头 → 等价于点他左边那张卡：选中 + 跟随 + 卡片展开，然后 2D 收起
// （先给展开态留张图：名字、颜色、避让效果都在这一张里）
const SHOT_NAMES = SHOT.replace(/\.png$/i, '_names.png');
await screenshot(SHOT_NAMES, await evaluate(`(() => {
  const r = document.getElementById('minimap-container').getBoundingClientRect();
  return { x: r.left, y: r.top, width: r.width, height: r.height };
})()`));

const o0 = await overlayOrigin();
const t0 = hits0[0];
await clickAt(Math.round(o0.x + t0.x), Math.round(o0.y + t0.y));
await sleep(350);
const picked = await evaluate(`({
  sel: window.__v5.selectedPlayer,
  follow: window.__v5.cameraFollow.player,
  expanded: window.__v5.minimap.expanded,
  card: (() => {
    const c = document.getElementById('player-card-${t0.idx}');
    return !!c && c.classList.contains('expanded');
  })(),
})`);
check('点箭头 = 选中这名选手（卡片展开 + 开始跟随）',
  picked.sel === t0.idx && picked.follow === t0.idx && picked.card,
  `sel=${picked.sel} follow=${picked.follow} card=${picked.card}`);
check('选中后 2D 视图收起（不然镜头在 0.16 透明度的背景后面飞）',
  picked.expanded === false, String(picked.expanded));

// 留一张「点完之后」的图：3D 已经飞到他身上、左边那张卡也展开了
const SHOT_PICK = SHOT.replace(/\.png$/i, '_picked.png');
await screenshot(SHOT_PICK);

// 再展开，点**名字**那一条也得选中（不用非得点到 10px 的箭头本体）
await evaluate(`document.getElementById('minimap-container').click()`);
await sleep(250);
await evaluate(`window.__v5.redrawMinimap()`);
const hits1 = (await mm()).hits;
const t1 = hits1.find((h) => h.idx !== t0.idx) || hits1[0];
if (t1) {
  const o1 = await overlayOrigin();
  await clickAt(Math.round(o1.x + (t1.nx0 + t1.nx1) / 2),
    Math.round(o1.y + (t1.ny0 + t1.ny1) / 2));
  await sleep(350);
  const pick2 = await evaluate(`({ sel: window.__v5.selectedPlayer, expanded: window.__v5.minimap.expanded })`);
  check('点名字也能选中（名字和箭头是同一个热区）',
    pick2.sel === t1.idx && pick2.expanded === false,
    `期望 ${t1.idx}（${t1.name}），实际 ${pick2.sel}`);
} else {
  skip('点名字选中', '只有一名可见玩家，没有第二个名字可点');
}

// 点空白处：收起，但不会误选中谁
await evaluate(`document.getElementById('minimap-container').click()`);
await sleep(250);
await evaluate(`window.__v5.redrawMinimap()`);
const beforeSel = await evaluate(`window.__v5.selectedPlayer`);
const emptySpot = await evaluate(`(() => {
  const m = window.__v5.minimap;
  const r = document.getElementById('minimap-overlay').getBoundingClientRect();
  for (const [fx, fy] of [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92], [0.5, 0.04], [0.5, 0.96]]) {
    const x = r.width * fx, y = r.height * fy;
    const free = m.hits.every((h) => Math.hypot(h.x - x, h.y - y) > h.r + 14
      && !(x >= h.nx0 - 8 && x <= h.nx1 + 8 && y >= h.ny0 - 8 && y <= h.ny1 + 8));
    if (free) return { x, y };
  }
  return null;
})()`);
if (emptySpot) {
  const o2 = await overlayOrigin();
  await clickAt(Math.round(o2.x + emptySpot.x), Math.round(o2.y + emptySpot.y));
  await sleep(300);
  const after2 = await evaluate(`({ sel: window.__v5.selectedPlayer, expanded: window.__v5.minimap.expanded })`);
  check('点空白处 = 收起，且不改选中',
    after2.expanded === false && after2.sel === beforeSel,
    `expanded=${after2.expanded} sel=${after2.sel}（之前 ${beforeSel}）`);
} else {
  skip('点空白处收起', '没找到不压在热区上的空白');
}

// 复位：取消选中，重新展开，后面的截图 / Esc 路径照旧
await evaluate(`window.__v5.selectPlayer(-1)`);
if (!(await evaluate(`window.__v5.minimap.expanded`))) {
  await evaluate(`document.getElementById('minimap-container').click()`);
  await sleep(250);
}
await evaluate(`window.__v5.redrawMinimap()`);

await screenshot(SHOT);

// ── 3. Esc 收起 ────────────────────────────────────────────────────────────
const key = async (code, keyName, vk) => {
  await send('Input.dispatchKeyEvent', {
    type: 'rawKeyDown', code, key: keyName, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk,
  });
  await send('Input.dispatchKeyEvent', {
    type: 'keyUp', code, key: keyName, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk,
  });
};
await key('Escape', 'Escape', 27);
await sleep(200);
await evaluate(`window.__v5.redrawMinimap()`);
const m2 = await mm();
check('Esc 收起 → 回到 200×200',
  m2.expanded === false && Math.abs(m2.size[0] - 200) <= 1, JSON.stringify(m2.size));
check('收起后备分辨率也降回 256',
  m2.canvas === 256 && m2.overlay === 256, `${m2.canvas}/${m2.overlay}`);

// ── 4. 键盘 Enter 也能开，✕ 也能关 ─────────────────────────────────────────
await evaluate(`document.getElementById('minimap-container').focus()`);
await key('Enter', 'Enter', 13);
await sleep(200);
const m3 = await mm();
check('focus 后按 Enter → 同样能放大', m3.expanded === true, String(m3.expanded));

await evaluate(`document.getElementById('minimap-close').click()`);
await sleep(200);
const m4 = await mm();
check('点 ✕ → 收起（没有冒泡回容器又翻回去）', m4.expanded === false, String(m4.expanded));

// ── 5. 窄窗口下仍然是正方形、不超出视口 ────────────────────────────────────
await send('Emulation.setDeviceMetricsOverride', {
  width: 1280, height: 800, deviceScaleFactor: 1, mobile: false,
});
await sleep(400);
await evaluate(`document.getElementById('minimap-container').click()`);
await sleep(300);
const m5 = await mm();
const vp1 = await evaluate(`(() => {
  const v = document.getElementById('viewport').getBoundingClientRect();
  return { w: Math.round(v.width), h: Math.round(v.height) };
})()`);
const want1 = Math.max(240, Math.min(vp1.w, vp1.h) - 32);
check('窗口改成 1280×800 后：正方形且重新算过边长',
  m5.expanded && m5.size[0] === m5.size[1] && Math.abs(m5.size[0] - want1) <= 1,
  `${JSON.stringify(m5.size)} 期望 ${want1}`);
check('放大态没有超出视口',
  m5.size[0] <= vp1.w && m5.size[0] <= vp1.h, `${m5.size[0]} vs ${vp1.w}×${vp1.h}`);
await send('Emulation.clearDeviceMetricsOverride');

// ── 6. 小地图里的烟被压暗（3D 场景那份不动） ────────────────────────────────
//
// 烟和 3D 场景共用同一套材质，压暗只在「小地图那一次绘制」里生效。这里不等截图
// 比亮度（受地图配色影响），而是在**小地图渲染的那一刻**把材质颜色抓出来：
// 画的时候是暗的、画完必须还原成原值。
const smokeRound = await evaluate(`(() => {
  const rounds = (window.__v5.summary && window.__v5.summary.rounds) || [];
  let best = null;
  for (const r of rounds) {
    if (!(r.smoke_count > 0)) continue;
    if (!best || r.smoke_count > best.smoke_count) best = r;
  }
  return best ? { idx: best.idx, n: best.smoke_count } : null;
})()`);

if (!smokeRound) {
  skip('小地图里的烟被压暗', '这场比赛没有烟');
} else {
  await evaluate(`(() => {
    const sel = document.getElementById('round-select');
    sel.value = '${smokeRound.idx}';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  })()`);
  let roundReady = false;
  for (let i = 0; i < 120 && !roundReady; i++) {
    await sleep(300);
    roundReady = await evaluate(`(() => {
      const r = window.__v5.round;
      return !!r && Array.isArray(r.smoke) && r.smoke.some((s) => s.detonation);
    })()`).catch(() => false);
  }
  check(`切到有烟的回合（第 ${smokeRound.idx + 1} 回合，${smokeRound.n} 团）`, roundReady);

  const smokeSeek = roundReady ? await evaluate(`(() => {
    const r = window.__v5.round;
    const sm = r.smoke.filter((s) => s.detonation && s.last_tick > s.first_tick);
    if (!sm.length) return null;
    // 存活最久的那团：取它正中间，云最大、最好截图
    let s = sm[0];
    for (const t of sm) if (t.last_tick - t.first_tick > s.last_tick - s.first_tick) s = t;
    const mid = Math.round((s.first_tick + s.last_tick) / 2);
    let idx = 0;
    for (let i = 0; i < r.ticks.length; i++) if (r.ticks[i] <= mid) idx = i;
    window.__v5.seekSample(idx);
    return { sample: idx, tick: r.ticks[idx], first: s.first_tick, last: s.last_tick };
  })()`) : null;

  if (!smokeSeek) {
    skip('小地图里的烟被压暗', '回合里没有可定位的烟');
  } else {
    const probeMat = await evaluate(`(() => {
      const v5 = window.__v5;
      // 体素云是 InstancedMesh，sprite 是 Sprite（名字都以 smoke- 开头，按类型分）
      let vox = null, spr = null;
      for (const o of v5.scene.children) {
        if (!o.name) continue;
        if (o.isInstancedMesh && o.name.startsWith('smoke-') && o.count > 0) { vox = o; }
        else if (o.isSprite && o.name.startsWith('smoke-sprite-')) { spr = o; }
      }
      if (!vox) return null;
      const hex = (m) => m.color.getHex();
      const mVox = vox.material, mSpr = spr ? spr.material : null;
      const before = { vox: hex(mVox), spr: mSpr ? hex(mSpr) : null };

      const seen = [];
      const mr = v5.minimapRenderer;
      const orig = mr.render;
      mr.render = function (s, c) {
        seen.push({ vox: hex(mVox), spr: mSpr ? hex(mSpr) : null });
        return orig.call(this, s, c);
      };
      v5.redrawMinimap();
      mr.render = orig;
      const after = { vox: hex(mVox), spr: mSpr ? hex(mSpr) : null };
      return { before, during: seen[seen.length - 1] || null, renders: seen.length, after };
    })()`);

    check('在场景里找到了正在画的烟（体素云）', !!probeMat && !!probeMat.during);
    if (probeMat && probeMat.during) {
      const ch = (h) => [(h >> 16) & 255, (h >> 8) & 255, h & 255];
      const b = ch(probeMat.before.vox);
      const d = ch(probeMat.during.vox);
      const a = ch(probeMat.after.vox);
      const hx = (h) => '#' + h.toString(16).padStart(6, '0');
      check('小地图绘制时烟变暗（画完立刻还原，3D 场景那份没动）',
        d[0] < b[0] && d[1] < b[1] && d[2] < b[2]
        && a[0] === b[0] && a[1] === b[1] && a[2] === b[2],
        `画前 ${hx(probeMat.before.vox)} → 画时 ${hx(probeMat.during.vox)}`
        + ` → 画后 ${hx(probeMat.after.vox)}`);
      const ratio = d[0] / Math.max(b[0], 1);
      check('压暗幅度合理（明显变深，但不是塞成黑块）',
        ratio >= 0.4 && ratio <= 0.9, `R 通道 ${ratio.toFixed(3)}`);
      if (probeMat.before.spr != null && probeMat.during.spr != null) {
        check('烟雾 sprite 层跟着一起压暗',
          probeMat.during.spr < probeMat.before.spr
          && probeMat.after.spr === probeMat.before.spr,
          `${hx(probeMat.before.spr)} → ${hx(probeMat.during.spr)}`
          + ` → ${hx(probeMat.after.spr)}`);
      }
    }

    // 留一张 2D 回放里带烟的图（展开态截，角落那块太小看不清）
    const isExpanded = await evaluate(
      `document.getElementById('minimap-container').classList.contains('mm-expanded')`);
    if (!isExpanded) {
      await evaluate(`document.getElementById('minimap-container').click()`);
      await sleep(300);
    }
    await evaluate(`window.__v5.redrawMinimap()`);
    const SHOT_SMOKE = SHOT.replace(/\.png$/i, '_smoke.png');
    await screenshot(SHOT_SMOKE, await evaluate(`(() => {
      const r = document.getElementById('minimap-container').getBoundingClientRect();
      return { x: r.left, y: r.top, width: r.width, height: r.height };
    })()`));
    if (existsSync(SHOT_SMOKE)) console.log(`截图: ${SHOT_SMOKE}`);
  }
}

check('没有 JS 异常',
  consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));

console.log('');
if (existsSync(SHOT)) console.log(`截图: ${SHOT}`);
if (existsSync(SHOT_SMALL)) console.log(`截图: ${SHOT_SMALL}`);
if (existsSync(SHOT_NAMES)) console.log(`截图: ${SHOT_NAMES}`);
if (existsSync(SHOT_PICK)) console.log(`截图: ${SHOT_PICK}`);
console.log(`${passed} passed, ${failed} failed${skipped ? `, ${skipped} skipped` : ''}`);
ws.close();
chrome.kill();
process.exit(failed === 0 ? 0 : 1);
