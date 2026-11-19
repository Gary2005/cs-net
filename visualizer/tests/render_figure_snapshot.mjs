/**
 * render_figure_snapshot.mjs — 把玩家小人单独渲染成 PNG，用来**肉眼看**它长什么样。
 *
 * 为什么不用整页截图：`Page.captureScreenshot` 拍的是整个页面，HTML 控件面板会盖在
 * canvas 上，而且相机受 OrbitControls / 页面布局 / 场景色调映射影响，截出来的东西
 * 没法比。所以这里在页面里**新开一个离屏 WebGLRenderer**，用正交相机只渲染
 * `player-N` 这一个 group，纯黑背景、自己打光 —— 画面里除了小人什么都没有，
 * 站姿和蹲姿还能像素级对齐着比。
 *
 * 用法：
 *   python visualizer/server.py --port 5099 &          # 需要有一个在跑的 visualizer
 *   node visualizer/tests/render_figure_snapshot.mjs http://127.0.0.1:5099 match.csz [玩家号]
 *
 * 产出（默认写到 /tmp）：
 *   /tmp/figure_stand.png    该玩家一个站立采样点
 *   /tmp/figure_crouch.png   该玩家一个蹲到底的采样点
 *   /tmp/figure_side.png     站姿的侧面（能看清腿的间隙和视线的方向）
 *   /tmp/figure_walk_strip.png  **一个完整步态周期**的侧视胶片条（走路摆腿）
 *
 * 两个 PNG 是同一台相机拍的，所以可以直接叠起来看：脚底必须完全重合（0 HU），
 * 头应该低 ~16 HU。
 *
 * 相机标定（改了正交参数就要同步改这里）：世界 y ∈ [−0.10, 1.95] 米，
 * 1 HU = 0.0254 米 → 画面底 = −3.9 HU，画面顶 = 76.8 HU。
 */

import { spawn } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const URL = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3];
const PLAYER = Number(process.argv[4] || 7);
const OUT = process.env.SNAPSHOT_DIR || '/tmp';

if (!CSZ) {
  console.error('用法: node render_figure_snapshot.mjs <server-url> <match.csz> [玩家号]');
  process.exit(2);
}

const CHROME = process.env.CHROME_PATH
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── 起 headless Chrome + 接 CDP ─────────────────────────────────────────────
const profile = mkdtempSync(join(tmpdir(), 'figure-shot-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-sandbox',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=800,800', 'about:blank',
]);

let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  await sleep(250);
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    wsUrl = list.find((t) => t.type === 'page')?.webSocketDebuggerUrl;
  } catch { /* 还没起来 */ }
}
if (!wsUrl) {
  chrome.kill();
  console.error('连不上 Chrome 的 CDP，检查 CHROME_PATH');
  process.exit(1);
}

const ws = new WebSocket(wsUrl);
await new Promise((r) => { ws.onopen = r; });
let msgId = 0;
const pending = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) => new Promise((res) => {
  const id = ++msgId;
  pending.set(id, res);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
  if (r.result?.exceptionDetails) {
    throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 800));
  }
  return r.result?.result?.value;
};

await send('Runtime.enable');
await send('Page.enable');
await send('Page.navigate', { url: URL });
await sleep(3000);

// ── 离屏正交渲染器：只画一个玩家 group ──────────────────────────────────────
await evaluate(`window.__snap = async (p, yaw) => {
  const THREE = await import('/static/three/build/three.module.js');
  const src = window.__v5.scene.getObjectByName('player-' + p);
  if (!src) throw new Error('找不到 player-' + p);
  const g = src.clone(true);          // 共用 geometry / material，只拷变换
  g.position.set(0, 0, 0);
  g.rotation.set(0, yaw, 0);
  g.visible = true;
  const scene = new THREE.Scene();
  scene.add(g);
  scene.add(new THREE.AmbientLight(0xffffff, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-2, 4, 4);
  const fill = new THREE.DirectionalLight(0x88aaff, 1.0); fill.position.set(3, 2, -3);
  scene.add(key, fill);
  const W = 240, H = 447;
  const r = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  r.setPixelRatio(1); r.setSize(W, H, false); r.setClearColor(0x000000, 1);
  // 世界 y ∈ [-0.10, 1.95] 米；相机在 y=0.925 所以 top/bottom 是相对偏移
  const cam = new THREE.OrthographicCamera(-0.55, 0.55, 1.025, -1.025, 0.1, 50);
  cam.position.set(0, 0.925, 3);
  cam.lookAt(0, 0.925, 0);
  cam.updateProjectionMatrix();
  r.render(scene, cam);
  const url = r.domElement.toDataURL('image/png');
  r.dispose();
  return url;
}; true`);

await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
await sleep(2000);

// ── 找该玩家的一个站立采样点和一个蹲到底的采样点 ───────────────────────────
const samples = await evaluate(`(() => {
  const r = window.__v5.round, p = ${PLAYER};
  if (!r || !r.players[p]) return null;
  let stand = -1, crouch = -1;
  for (let s = 0; s < r.ticks.length; s++) {
    if (!r.players[p].alive[s]) continue;
    if (stand < 0 && r.players[p].crouch[s] === 0) stand = s;
    if (crouch < 0 && r.players[p].crouch[s] > 0.99) crouch = s;
  }
  return { stand, crouch, hasCrouch: r.players[p].has_crouch !== false };
})()`);
if (!samples) {
  console.error(`.csz 里没有玩家 ${PLAYER}`);
  process.exit(1);
}
console.log(`玩家 ${PLAYER}：站立采样 ${samples.stand}，蹲下采样 ${samples.crouch}`);

const shots = [
  ['figure_stand.png', samples.stand, 0, 'front', false],
  ['figure_crouch.png', samples.crouch, 0, 'front', false],
  ['figure_side.png', samples.stand, Math.PI / 2, 'side', false],
  // 最后一张把碰撞盒 / 眼睛也打开：直接看"身体在滑、盒子在跳"
  ['figure_crouch_box.png', samples.crouch, Math.PI * 0.22, 'front+box', true],
];
const setCrouchLayer = async (on) => {
  await evaluate(`(() => {
    const t = document.getElementById('toggle-crouch');
    if (t.checked !== ${on}) { t.checked = ${on}; t.dispatchEvent(new Event('change')); }
    return true;
  })()`);
};

for (const [file, sample, yaw, label, box] of shots) {
  if (sample < 0) { console.log(`跳过 ${file}（没有这个姿势的采样点）`); continue; }
  await setCrouchLayer(box);
  await evaluate(`window.__v5.seekSample(${sample})`);
  await sleep(250);
  const dataUrl = await evaluate(`window.__snap(${PLAYER}, ${yaw})`, true);
  writeFileSync(join(OUT, file), Buffer.from(dataUrl.split(',')[1], 'base64'));
  const dbg = await evaluate(`(() => { const d = window.__v5.crouchDebug[${PLAYER}];
    return { crouch: window.__v5.players[${PLAYER}].crouch, figureH: d.figureH,
             headY: d.headY, hullH: d.hullH, eyeY: d.eyeY }; })()`);
  console.log(`  ${join(OUT, file)}  [${label}]  crouch=${dbg.crouch.toFixed(3)}`
    + `  小人=${dbg.figureH.toFixed(1)}HU  头心=${dbg.headY.toFixed(1)}HU`
    + (box ? `  碰撞盒=${dbg.hullH.toFixed(0)}HU  眼=${dbg.eyeY.toFixed(1)}HU` : ''));
}
await setCrouchLayer(false);

// ── 走路的胶片条 ────────────────────────────────────────────────────────────
// 一个完整步态周期的侧视图拼成一张图。摆腿是那种"数据全对但看着别扭"的东西，
// 只能靠眼睛 —— 正面图看不出来（前后摆腿在正视图里被投影掉了），必须侧视。
const STRIP_FRAMES = 7;
const strip = await evaluate(`(async () => {
  const THREE = await import('/static/three/build/three.module.js');
  const r = window.__v5.round, p = ${PLAYER};
  // 找该玩家一段连续的全速跑
  let best = null;
  for (let s = 1; s < r.ticks.length; s++) {
    if (!(r.players[p].alive[s] && r.players[p].on_ground[s] && r.players[p].v[s] > 240)) continue;
    let e = s;
    while (e < r.ticks.length && r.players[p].alive[e] && r.players[p].on_ground[e]
           && r.players[p].v[e] > 240) e++;
    if (e - s >= 30 && (!best || e - s > best.e - best.s)) best = { s, e };
    s = e;
  }
  if (!best) return null;
  // 一个周期 = GAIT_STRIDE_HU / 速度 秒 = ... 采样
  const speed = r.players[p].v[best.s];
  const cycleSamples = 112 / speed * 64;
  const N = ${STRIP_FRAMES};
  const W = 150, H = 260;
  const out = document.createElement('canvas');
  out.width = W * N; out.height = H;
  const ctx = out.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, out.width, out.height);
  ctx.font = '11px monospace'; ctx.textBaseline = 'top';
  const info = [];
  for (let k = 0; k < N; k++) {
    const s = best.s + Math.round(k * cycleSamples / N);
    window.__v5.seekSample(s);
    const src = window.__v5.scene.getObjectByName('player-' + p);
    const g = src.clone(true);
    g.position.set(0, 0, 0); g.rotation.set(0, Math.PI / 2, 0); g.visible = true;
    const sc = new THREE.Scene(); sc.add(g);
    sc.add(new THREE.AmbientLight(0xffffff, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-2, 4, 4);
    const fill = new THREE.DirectionalLight(0x88aaff, 1.0); fill.position.set(3, 2, -3);
    sc.add(key, fill);
    const rr = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    rr.setPixelRatio(1); rr.setSize(W, H, false); rr.setClearColor(0x000000, 1);
    const half = (1.95 + 0.10) / 2, mid = (1.95 - 0.10) / 2;
    const cam = new THREE.OrthographicCamera(-half * W / H, half * W / H, half, -half, 0.1, 50);
    cam.position.set(0, mid, 3); cam.lookAt(0, mid, 0); cam.updateProjectionMatrix();
    rr.render(sc, cam);
    ctx.drawImage(rr.domElement, k * W, 0);
    rr.dispose();
    const d = window.__v5.crouchDebug[p];
    info.push({ s, swing: d.legSwingL, footL: d.footL, footR: d.footR,
                spread: d.footSpread, hip: d.hipY });
    ctx.fillStyle = '#7fd4ff';
    ctx.fillText('swing ' + d.legSwingL.toFixed(2), k * W + 6, 6);
    ctx.fillText('split ' + d.footSpread.toFixed(0) + ' HU', k * W + 6, 20);
    ctx.fillStyle = '#444';
    ctx.fillRect(k * W, 0, 1, H);
  }
  return { url: out.toDataURL('image/png'), info, speed, from: best.s, cycleSamples };
})()`, true);

if (strip) {
  writeFileSync(join(OUT, 'figure_walk_strip.png'),
    Buffer.from(strip.url.split(',')[1], 'base64'));
  console.log(`  ${join(OUT, 'figure_walk_strip.png')}  [走路 · 侧视 · 一个步态周期 `
    + `${strip.cycleSamples.toFixed(1)} 采样 @ ${strip.speed.toFixed(0)} HU/s]`);
  console.log(`       ${strip.info.map((f) => f.swing.toFixed(2)).join('  ')}   ← 摆角（弧度）`);
} else {
  console.log('  跳过 figure_walk_strip.png（这个玩家没有连续全速跑的一段）');
}

ws.close();
chrome.kill();
console.log('\n提示：两张正面图是同一台相机拍的，叠起来看脚底必须重合、头低 ~16 HU。');
