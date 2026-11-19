#!/usr/bin/env node
/**
 * test_first_person.mjs — 在真实浏览器里验收「第一人称摄像机」
 * （headless Chrome + CDP）。
 *
 * 覆盖用户操作路径和相机行为：
 *
 *   1. 点左侧卡片 → 详情里出现「🎥 摄像机」下拉，选「第一人称摄像机」
 *      → 进入该选手 POV，HUD 显示且数值与数据一致；
 *   2. 相机正好在选手眼睛上；自己的模型被隐藏；
 *   3. 准星在视口正中（屏幕中心 = 选手的瞄准方向）；
 *   4. 进入后鼠标拖拽 / 滚轮 / WASD 都不能移动相机；
 *   5. 退出（Esc）→ 相机恢复到进入前的 orbit 机位，拖拽恢复可旋转；
 *   6. 控制台没有报错。
 *
 * 用法（需要 Chrome/Edge 和一个已经在跑的 visualizer / app.server）：
 *
 *     node visualizer/tests/test_first_person.mjs http://127.0.0.1:5099 \
 *          app/data/csz/aurora-vs-vitality-m2-inferno__ca728317.csz
 *
 * 服务端已有会话时会直接复用；没有就先用第三个参数加载一场。
 * 没找到 Chrome 就打印 SKIP 并以 0 退出。
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
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
  '/usr/bin/chromium',
].filter(Boolean);

const BASE = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3] || 'app/data/csz/aurora-vs-vitality-m2-inferno__ca728317.csz';
const PORT = 9333 + (process.pid % 200);
const chrome = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chrome) { console.log('SKIP: no Chrome'); process.exit(0); }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
function check(name, cond, detail = '') {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${!cond && detail ? `   ${detail}` : ''}`);
  if (!cond) failed++;
}

const profile = mkdtempSync(join(tmpdir(), 'fp-test-'));
const proc = spawn(chrome, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--disable-gpu-sandbox', '--no-sandbox',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1280,800', 'about:blank',
], { stdio: ['ignore', 'ignore', 'ignore'] });

let wsUrl = null;
for (let i = 0; i < 80 && !wsUrl; i++) {
  await sleep(250);
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    wsUrl = list.find((t) => t.type === 'page')?.webSocketDebuggerUrl || null;
  } catch {}
}
if (!wsUrl) { console.log('SKIP: no CDP'); proc.kill(); process.exit(0); }

const ws = new WebSocket(wsUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let msgId = 0;
const pending = new Map();
const errors = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map(a => a.value ?? a.description).join(' '));
};
function send(method, params = {}) {
  const id = ++msgId;
  return new Promise((res) => { pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
}
async function evaluate(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
}
async function mouse(type, x, y, extra = {}) {
  await send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1, ...extra });
}
async function key(type, code, keyStr) {
  await send('Input.dispatchKeyEvent', { type, code, key: keyStr, windowsVirtualKeyCode: keyStr.toUpperCase().charCodeAt(0) });
}

await send('Page.enable');
await send('Runtime.enable');
await send('Page.navigate', { url: BASE + '/' });

let ready = false;
for (let i = 0; i < 40 && !ready; i++) {
  await sleep(250);
  try { ready = await evaluate('!!(window.__v5 && window.__v5.round && window.__v5.players.length)'); } catch {}
}
// 服务端还没有会话时，用第三个参数加载一场（和 test_camera_follow 一样）
if (!ready && existsSync(CSZ)) {
  console.log(`loading ${CSZ} …`);
  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
}
for (let i = 0; i < 160 && !ready; i++) {
  await sleep(250);
  try { ready = await evaluate('!!(window.__v5 && window.__v5.round && window.__v5.players.length)'); } catch {}
}
if (!ready) { console.log('SKIP: replay data not ready'); proc.kill(); process.exit(0); }

// 等地图加载完（onDataReady 里的 loadMap().then(fitCameraToMap) 会在加载完那
// 一刻整图取景）。不等的话，测试进行到一半它可能把"进入前机位"改掉，退出恢复
// 的断言就会失败 —— 那是地图加载的竞态，不是第一人称的问题。
for (let i = 0; i < 160; i++) {
  let mapReady = false;
  try {
    mapReady = await evaluate('(__v5.scene.getObjectByName("map-root")?.children.length || 0) > 0');
  } catch {}
  if (mapReady) break;
  await sleep(250);
}
await sleep(600);

// 暂停回放，避免相机因为"玩家在走"而变化，掩盖锁定测试
const playing = await evaluate('window.__v5.isPlaying');
if (playing) { await evaluate('document.getElementById("btn-play").click()'); await sleep(300); }
await evaluate('window.__v5.stepFrame(1/60, false)');

console.log('\n=== A 进入第一人称 ===');
const beforeCam = await evaluate('[__v5.camera.position.x, __v5.camera.position.y, __v5.camera.position.z]');
const entered = await evaluate('__v5.firstPerson.enter(0)');
await evaluate('window.__v5.stepFrame(1/60, false)');
const fp = await evaluate(`(() => {
  const st = __v5.players[0];
  const SCALE = 0.0254;
  const eye = [st.y * SCALE, (st.z + st.eye_height) * SCALE, st.x * SCALE];
  const c = __v5.camera.position;
  return {
    active: __v5.firstPerson.active, player: __v5.firstPerson.player,
    locked: __v5.firstPerson.locked, hudVisible: __v5.firstPerson.hudVisible,
    hud: __v5.firstPerson.hud,
    state: { hp: st.hp, armor: st.armor, helmet: st.helmet, weapon: st.weapon, inventory: st.inventory },
    eye, cam: [c.x, c.y, c.z],
    viewport: (() => { const r = document.getElementById('viewport').getBoundingClientRect();
      const h = document.getElementById('fp-hud').getBoundingClientRect();
      const ch = document.querySelector('.fp-crosshair').getBoundingClientRect();
      return { vp: [r.left, r.top, r.width, r.height], hud: [h.left, h.top, h.width, h.height],
        cross: [ch.left + ch.width / 2, ch.top + ch.height / 2] }; })(),
  };
})()`);

check('enter() 返回 true', entered === true);
check('模式处于 active，目标 = 0', fp.active === true && fp.player === 0, JSON.stringify({ a: fp.active, p: fp.player }));
check('镜头被锁定（first 模式 + OrbitControls 禁用）', fp.locked === true);
check('HUD 可见', fp.hudVisible === true);
check('相机位于该选手眼睛', Math.hypot(fp.cam[0] - fp.eye[0], fp.cam[1] - fp.eye[1], fp.cam[2] - fp.eye[2]) < 1e-6,
  JSON.stringify({ cam: fp.cam, eye: fp.eye }));
const aim = await evaluate(`(() => {
  const st = __v5.players[0];
  const yaw = (st.yaw || 0) * Math.PI / 180, pitch = (st.pitch || 0) * Math.PI / 180;
  // three 空间：基础朝向 (0,0,1) 先绕 X 转 pitch，再绕 Y 转 yaw
  const expected = [Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
  const d = __v5.camera.getWorldDirection(new (__v5.camera.position.constructor)());
  return { expected, got: [d.x, d.y, d.z], pitch: st.pitch, yaw: st.yaw };
})()`);
const aimErr = Math.hypot(...aim.got.map((v, i) => v - aim.expected[i]));
check('相机朝向 == 该选手的 yaw/pitch（屏幕中心就是瞄准方向）', aimErr < 1e-6,
  `err=${aimErr.toExponential(2)} ${JSON.stringify(aim)}`);
check('HUD 血量与数据一致', fp.hud.hp === String(Math.round(fp.state.hp)), `${fp.hud.hp} vs ${fp.state.hp}`);
check('HUD 护甲与数据一致', fp.hud.armor === String(Math.round(fp.state.armor)), `${fp.hud.armor} vs ${fp.state.armor}`);
check('HUD 头盔与数据一致', fp.hud.helmet === (fp.state.helmet ? '有' : '无'), `${fp.hud.helmet}`);
check('HUD 有选手名和队伍', fp.hud.name.length > 0 && (fp.hud.team === 'CT' || fp.hud.team === 'T'), JSON.stringify({ n: fp.hud.name, t: fp.hud.team }));
check('HUD 当前武器不是空', fp.hud.weapon !== '—' && fp.hud.weapon !== '', fp.hud.weapon);
const ownVisible = await evaluate('__v5.scene.getObjectByName("player-0").visible');
check('自己的模型被隐藏', ownVisible === false, `visible=${ownVisible}`);
check('准星在视口正中', Math.abs(fp.viewport.cross[0] - (fp.viewport.vp[0] + fp.viewport.vp[2] / 2)) < 1.5
  && Math.abs(fp.viewport.cross[1] - (fp.viewport.vp[1] + fp.viewport.vp[3] / 2)) < 1.5,
  JSON.stringify({ cross: fp.viewport.cross, vp: fp.viewport.vp }));

console.log('\n=== B 进入后鼠标 / WASD 不能移动镜头 ===');
const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) writeFileSync('.build/first-person.png', Buffer.from(shot.result.data, 'base64'));
const camBeforeDrag = await evaluate('[__v5.camera.position.x, __v5.camera.position.y, __v5.camera.position.z]');
const rect = fp.viewport.vp;
const cx = rect[0] + rect[2] / 2, cy = rect[1] + rect[3] / 2;
await mouse('mousePressed', cx, cy, { buttons: 1 });
for (let i = 1; i <= 6; i++) await mouse('mouseMoved', cx + i * 18, cy + i * 10, { buttons: 1 });
await mouse('mouseReleased', cx + 108, cy + 60, { buttons: 0 });
await mouse('mouseWheel', cx, cy, { deltaX: 0, deltaY: -400 });
await key('keyDown', 'KeyW', 'w');
await key('keyDown', 'KeyD', 'd');
for (let i = 0; i < 20; i++) await evaluate('window.__v5.stepFrame(1/60, false)');
await key('keyUp', 'KeyW', 'w');
await key('keyUp', 'KeyD', 'd');
const camAfterDrag = await evaluate('[__v5.camera.position.x, __v5.camera.position.y, __v5.camera.position.z]');
const dragDelta = Math.hypot(...camAfterDrag.map((v, i) => v - camBeforeDrag[i]));
check('拖拽 / 滚轮 / WASD 后相机没有移动', dragDelta < 1e-9, `delta=${dragDelta}`);
check('仍然处于第一人称', await evaluate('__v5.firstPerson.active') === true);

console.log('\n=== C 角色移动时第一人称跟着眼睛走 ===');
await evaluate('document.getElementById("btn-play").click()');
for (let i = 0; i < 30; i++) await evaluate('window.__v5.stepFrame(1/60, false)');
const followCheck = await evaluate(`(() => {
  const st = __v5.players[0]; const SCALE = 0.0254; const c = __v5.camera.position;
  const eye = [st.y * SCALE, (st.z + st.eye_height) * SCALE, st.x * SCALE];
  return { d: Math.hypot(c.x - eye[0], c.y - eye[1], c.z - eye[2]), yaw: window.__v5.players[0].yaw };
})()`);
check('相机始终贴着眼睛（随选手移动）', followCheck.d < 1e-6, `delta=${followCheck.d}`);
if (await evaluate('window.__v5.isPlaying')) { await evaluate('document.getElementById("btn-play").click()'); }

console.log('\n=== D 退出恢复 orbit 机位 ===');
await evaluate('window.__v5.stepFrame(1/60, false)');
const exitPos = await evaluate('[__v5.camera.position.x, __v5.camera.position.y, __v5.camera.position.z]');
const exited = await evaluate('__v5.firstPerson.exit()');
await evaluate('window.__v5.stepFrame(1/60, false)');
const after = await evaluate(`(() => ({
  active: __v5.firstPerson.active, hudHidden: document.getElementById('fp-hud').classList.contains('hidden'),
  fpClass: document.getElementById('viewport').classList.contains('fp-active'),
  cam: [__v5.camera.position.x, __v5.camera.position.y, __v5.camera.position.z],
}))()`);
const restoreDelta = Math.hypot(...after.cam.map((v, i) => v - beforeCam[i]));
check('exit() 返回 true', exited === true);
check('退出后 active = false / HUD 隐藏 / 视口 class 清掉', after.active === false && after.hudHidden === true && after.fpClass === false);
check('退出后相机恢复到进入前的位置', restoreDelta < 1e-6, `delta=${restoreDelta} exitAt=${exitPos}`);

console.log('\n=== E 退出后自由相机恢复 ===');
const orbitBefore = await evaluate('[__v5.camera.position.x, __v5.camera.position.y, __v5.camera.position.z]');
await mouse('mousePressed', cx, cy, { buttons: 1 });
for (let i = 1; i <= 8; i++) await mouse('mouseMoved', cx + i * 20, cy + i * 6, { buttons: 1 });
await mouse('mouseReleased', cx + 160, cy + 48, { buttons: 0 });
for (let i = 0; i < 20; i++) await evaluate('window.__v5.stepFrame(1/60, false)');
const orbitAfter = await evaluate('[__v5.camera.position.x, __v5.camera.position.y, __v5.camera.position.z]');
const orbitDelta = Math.hypot(...orbitAfter.map((v, i) => v - orbitBefore[i]));
check('退出后拖拽可以旋转相机', orbitDelta > 1e-4, `delta=${orbitDelta}`);

console.log('\n=== G 用户路径：点卡片 → 下拉选「第一人称摄像机」→ Esc 退出 ===');
await evaluate('document.getElementById("player-card-1").click()');
await evaluate('window.__v5.stepFrame(1/60, false)');
const selExists = await evaluate('!!document.getElementById("player-cam-1")');
check('展开卡片后出现摄像机下拉', selExists === true);
await evaluate(`(() => {
  const s = document.getElementById('player-cam-1');
  s.value = 'first';
  s.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`);
await evaluate('window.__v5.stepFrame(1/60, false)');
const viaUi = await evaluate(`(() => ({
  active: __v5.firstPerson.active, player: __v5.firstPerson.player,
  select: document.getElementById('player-cam-1').value,
  hudName: document.getElementById('fp-name').textContent,
  expected: __v5.summary.players[1].name,
}))()`);
check('下拉选择后进入 1 号选手的第一人称', viaUi.active === true && viaUi.player === 1, JSON.stringify(viaUi));
check('下拉框停在「第一人称摄像机」', viaUi.select === 'first');
check('HUD 显示的是 1 号选手', viaUi.hudName === viaUi.expected, `${viaUi.hudName} vs ${viaUi.expected}`);
await send('Input.dispatchKeyEvent', { type: 'keyDown', code: 'Escape', key: 'Escape', windowsVirtualKeyCode: 27 });
await send('Input.dispatchKeyEvent', { type: 'keyUp', code: 'Escape', key: 'Escape', windowsVirtualKeyCode: 27 });
await sleep(80);
await evaluate('window.__v5.stepFrame(1/60, false)');
const afterEsc = await evaluate(`(() => ({
  active: __v5.firstPerson.active,
  hudHidden: document.getElementById('fp-hud').classList.contains('hidden'),
  select: document.getElementById('player-cam-1').value,
}))()`);
check('Esc 退出第一人称', afterEsc.active === false && afterEsc.hudHidden === true, JSON.stringify(afterEsc));
check('退出后下拉框回到「跟随镜头」', afterEsc.select === 'follow', afterEsc.select);

console.log('\n=== F 控制台 ===');
check('没有 JS 报错', errors.length === 0, errors.slice(0, 3).join(' | '));

try { proc.kill(); } catch {}
await sleep(800);
try { rmSync(profile, { recursive: true, force: true }); } catch {}
console.log(`\n${failed ? failed + ' FAILED' : 'ALL OK'}`);
process.exit(failed ? 1 : 0);
