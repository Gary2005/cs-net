#!/usr/bin/env node
/**
 * test_camera_follow.mjs — 在真实浏览器里验证「选中一个玩家之后镜头跟着他走」
 * （headless Chrome + CDP）。
 *
 * 量的是**场景图里的相机**，不是数据：
 *
 *   1. 点玩家卡片 → 飞过去 + 开始跟随（卡片展开、跟随目标是他）；
 *   2. 飞行途中也要跟：落点必须正对着**此刻**的人，不是点击那一刻的快照；
 *   3. 继续播放 → 镜头的位移必须**逐分量等于**玩家的位移（相对视角不变），
 *      视角中心仍然钉在他身上；
 *   4. 再点一次卡片 / 重置视角 / 关掉开关 → 跟随立刻停（镜头不再跟着动）；
 *   5. 开关关掉时仍然会飞过去（只是不跟）；
 *   6. 换回合继续跟同一个人（卡片重建之后也要接回来）；
 *   7. 控制台没有报错。
 *
 * 为什么用 `__v5.stepFrame(dt, false)` 一帧一帧推：headless 的 rAF 只有 1~3 fps，
 * 而主循环里 `if (dt > 0.25) dt = 0.25` 会把掉帧削平 —— 按墙钟量出来的是帧率，
 * 不是跟随。在一个 JS 任务里连着调它，rAF 插不进来，位移就是确定的。
 * 第二个参数 false = 这一帧不碰 GPU（相机的位移计算和绘不绘制无关），
 * 不然一次 65 MB 地图的绘制要几百毫秒。
 *
 * 用法（需要 Chrome/Edge 和一个已经在跑的 visualizer / app.server）：
 *
 *     node visualizer/tests/test_camera_follow.mjs http://127.0.0.1:5099 \
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
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

const BASE = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3] || 'app/data/csz/aurora-vs-vitality-m2-inferno__ca728317.csz';
const PORT = 9333 + (process.pid % 200);

/** scene.js 的 SCALE（游戏单位 → three 单位），测试自己算一遍，不复用被测代码 */
const SCALE = 0.0254;

const chrome = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chrome) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
function check(name, cond, detail = '') {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${!cond && detail ? `   ${detail}` : ''}`);
  if (!cond) failed++;
}

const profile = mkdtempSync(join(tmpdir(), 'camera-follow-'));
const proc = spawn(chrome, [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check',
  '--disable-gpu-sandbox', '--no-sandbox',
  // 软件 WebGL：headless 里没有真 GPU，不给这个 three.js 会拿不到 context
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1280,800',
  'about:blank',
// 三个 stdio 都 'ignore'：这个测试从不读 Chrome 的 stderr，而 DSH 的沙箱里
// 子进程的 stdio 管道（named pipe）建不起来 —— spawn 会直接 EPERM。
// 采集是靠 CDP 的 WebSocket，不需要任何管道。
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
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
    return;
  }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    consoleErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
  }
  if (m.method === 'Runtime.exceptionThrown') {
    const d = m.params.exceptionDetails;
    pageErrors.push(d.exception?.description || d.text);
  }
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
    consoleErrors.push(m.params.entry.text);
  }
};

function send(method, params = {}) {
  const id = ++msgId;
  return new Promise((res) => {
    pending.set(id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expr, awaitPromise = false) {
  const r = await send('Runtime.evaluate', {
    expression: expr, returnByValue: true, awaitPromise,
  });
  if (r.result?.exceptionDetails) {
    throw new Error(r.result.exceptionDetails.exception?.description
      || r.result.exceptionDetails.text);
  }
  return r.result?.result?.value;
}

// ── 小工具（在 node 侧算，别让被测代码自己给自己打分）───────────────────────

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (v) => Math.hypot(v[0], v[1], v[2]);
const fmt = (v) => `[${v.map((x) => x.toFixed(3)).join(', ')}]`;

/**
 * 一次性读完「相机 / 视角中心 / 某个玩家的世界坐标」——必须在**同一个 JS 任务**
 * 里读，否则中间的 rAF 帧会让两边的时基错开。玩家的世界坐标由测试自己按
 * SCALE 从 `__v5.players[i]` 算，不用被测代码的换算函数。
 */
const snapOf = (i) => `(() => {
  const S = ${SCALE};
  const v = window.__v5;
  const st = v.players[${i}];
  const c = v.camera.position;
  const t = v.cameraFollow.target();
  return {
    time: v.currentTime,
    playing: v.isPlaying,
    camera: [c.x, c.y, c.z],
    target: t,
    offset: [c.x - t[0], c.y - t[1], c.z - t[2]],
    playerWorld: st ? [st.y * S, st.z * S, st.x * S] : null,
    follow: v.cameraFollow.player,
    active: v.cameraFollow.active,
    animating: v.cameraFollow.animating,
    selected: v.selectedPlayer,
  };
})()`;

/** 在一个 JS 任务里连推 N 帧固定 dt（不绘制） */
const stepFrames = (n, dt) => `(() => {
  for (let k = 0; k < ${n}; k++) window.__v5.stepFrame(${dt}, false);
  return window.__v5.currentTime;
})()`;

try {
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: BASE });
  await sleep(2500);

  console.log(`chrome: ${chrome}`);
  console.log(`page  : ${BASE}`);
  console.log(`csz   : ${CSZ}\n`);

  check('the page booted the app', await evaluate('!!window.__v5'));
  check('the follow debug hook exists',
    await evaluate('!!(window.__v5.cameraFollow && window.__v5.stepFrame && window.__v5.cameraFollow.target)'));

  console.log('\nload');
  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
  // `loadPath` 只等到 summary（onDataReady），**回合载荷**是之后异步拉的，所以
  // 这里轮询等 currentRound，而不是睡一个"应该够了"的固定时间（机器一忙就不够）。
  let roundLoaded = false;
  for (let i = 0; i < 60 && !roundLoaded; i++) {
    roundLoaded = await evaluate('!!(window.__v5.round && window.__v5.round.players.length)');
    if (!roundLoaded) await sleep(400);
  }
  check('a round is loaded', roundLoaded);

  // ── 挑一个「真的在跑」的人和一个够长的窗口 ────────────────────────────
  // 窗口里他必须一直活着，位移还要大（最后按总位移取最大）。
  const pick = await evaluate(`(() => {
    const r = window.__v5.round;
    const K = 400, n = r.ticks.length;
    let best = null;
    for (let p = 0; p < r.players.length; p++) {
      const pl = r.players[p];
      for (let s = 0; s + K < n; s += 17) {
        let d = 0, ok = true;
        for (let k = 0; k < K && ok; k++) {
          if (!pl.alive[s + k]) { ok = false; break; }
          d += Math.hypot(pl.x[s + k + 1] - pl.x[s + k],
                          pl.y[s + k + 1] - pl.y[s + k],
                          pl.z[s + k + 1] - pl.z[s + k]);
        }
        if (ok && (!best || d > best.d)) best = { p, s, d, K };
      }
    }
    return best;
  })()`);
  if (!pick) {
    check('found a player who runs for a while', false, 'no such window');
    throw new Error('no usable window');
  }
  const P = pick.p;
  console.log(`\nrunning player: P${P + 1} @ sample ${pick.s} `
    + `(${pick.d.toFixed(0)} HU over ${pick.K} samples)`);

  await evaluate(`window.__v5.seekSample(${pick.s})`);
  const playing = await evaluate(`(() => {
    if (!window.__v5.isPlaying) document.getElementById('btn-play').click();
    return window.__v5.isPlaying;
  })()`);
  check('playback is running', playing === true);

  // ── 1. 点卡片 = 飞过去 + 开始跟随 ─────────────────────────────────────
  console.log('\nselecting a player');
  const clicked = await evaluate(`(() => {
    document.getElementById('player-card-${P}').click();
    return {
      follow: window.__v5.cameraFollow.player,
      active: window.__v5.cameraFollow.active,
      animating: window.__v5.cameraFollow.animating,
      selected: window.__v5.selectedPlayer,
      enabled: window.__v5.cameraFollow.enabled,
    };
  })()`);
  check('the follow toggle is on by default', clicked.enabled === true);
  check('clicking his card starts the follow', clicked.follow === P, `follow=${clicked.follow}`);
  check('following is active', clicked.active === true);
  check('his card is expanded', clicked.selected === P);
  check('the fly-to animation is running', clicked.animating === true);
  const toast = await evaluate(`(() => {
    const t = document.getElementById('toast');
    return { hidden: t.classList.contains('hidden'), text: t.textContent };
  })()`);
  check('a toast tells you the camera is following',
    !toast.hidden && toast.text.includes('跟随'), JSON.stringify(toast));

  // ── 2. 飞行途中也在跟：落点正对此刻的人 ───────────────────────────────
  await evaluate(stepFrames(30, '1/30'));   // 30 帧 × 1/30 s = 1 s > 0.7s 的飞行
  const landed = await evaluate(snapOf(P));
  check('the flight finished', landed.animating === false);
  check('following survived the flight', landed.follow === P && landed.active === true,
    `follow=${landed.follow}`);
  check('it lands looking at the player (not at the click-time snapshot)',
    dist(landed.target, landed.playerWorld) < 0.01,
    `|target-player| = ${dist(landed.target, landed.playerWorld).toFixed(4)} m`);

  // ── 3. 继续播放：镜头跟着人走（核心）──────────────────────────────────
  console.log('\ntracking while playing');
  const a = await evaluate(snapOf(P));
  await evaluate(stepFrames(15, '1/30'));
  const b = await evaluate(snapOf(P));
  check('playback advanced', b.time > a.time, `${a.time} -> ${b.time}`);
  const dPlayer = sub(b.playerWorld, a.playerWorld);
  const dCam = sub(b.camera, a.camera);
  const dTarget = sub(b.target, a.target);
  check('the player really moved', len(dPlayer) > 0.2, `${len(dPlayer).toFixed(3)} m`);
  check('the camera moved by exactly the player displacement',
    len(sub(dCam, dPlayer)) < 0.005,
    `camera ${fmt(dCam)} vs player ${fmt(dPlayer)}`);
  check('the view centre moved with him', len(sub(dTarget, dPlayer)) < 0.005,
    `target ${fmt(dTarget)} vs player ${fmt(dPlayer)}`);
  check('the viewing offset is unchanged (no drifting angle/zoom)',
    len(sub(b.offset, a.offset)) < 0.005,
    `${fmt(a.offset)} -> ${fmt(b.offset)}`);
  check('he is still centred', dist(b.target, b.playerWorld) < 0.01,
    `${dist(b.target, b.playerWorld).toFixed(4)} m`);

  // ── 4. 再点一次卡片 = 取消选中 + 停止跟随 ─────────────────────────────
  console.log('\ndeselecting');
  const closed = await evaluate(`(() => {
    document.getElementById('player-card-${P}').click();
    return {
      follow: window.__v5.cameraFollow.player,
      active: window.__v5.cameraFollow.active,
      selected: window.__v5.selectedPlayer,
    };
  })()`);
  check('the follow stops', closed.follow === -1 && closed.active === false);
  check('and the card is collapsed', closed.selected === -1);
  const c = await evaluate(snapOf(P));
  await evaluate(stepFrames(15, '1/30'));
  const d = await evaluate(snapOf(P));
  check('the player kept moving', dist(d.playerWorld, c.playerWorld) > 0.2,
    `${dist(d.playerWorld, c.playerWorld).toFixed(3)} m`);
  check('the camera stayed put', len(sub(d.camera, c.camera)) < 0.005,
    fmt(sub(d.camera, c.camera)));
  check('the view centre stayed put', len(sub(d.target, c.target)) < 0.005,
    fmt(sub(d.target, c.target)));

  // ── 5. 关掉开关：还是飞过去，但不跟 ──────────────────────────────────
  console.log('\ntoggle off');
  const other = (P + 1) % 10;
  const off = await evaluate(`(() => {
    const t = document.getElementById('toggle-cam-follow');
    t.checked = false; t.dispatchEvent(new Event('change'));
    return window.__v5.cameraFollow.enabled;
  })()`);
  check('the toggle reports off', off === false);
  // 暂停之后飞：终点就是那一刻的人，位置不会在飞行途中变掉，好对拍
  await evaluate(`(() => {
    if (window.__v5.isPlaying) document.getElementById('btn-play').click();
    document.getElementById('player-card-${other}').click();
    return true;
  })()`);
  const offClick = await evaluate(`({
    follow: window.__v5.cameraFollow.player,
    selected: window.__v5.selectedPlayer,
    animating: window.__v5.cameraFollow.animating,
  })`);
  check('with the toggle off, selecting does not follow', offClick.follow === -1,
    `follow=${offClick.follow}`);
  check('but the card is still expanded', offClick.selected === other);
  await evaluate(stepFrames(30, '1/30'));
  const offLanded = await evaluate(snapOf(other));
  check('it still flies to him', dist(offLanded.target, offLanded.playerWorld) < 0.01,
    `${dist(offLanded.target, offLanded.playerWorld).toFixed(4)} m`);
  check('and the flight ended', offLanded.animating === false);

  // ── 6. 重置视角 = 停止跟随 ───────────────────────────────────────────
  console.log('\nreset view');
  const back = await evaluate(`(() => {
    const t = document.getElementById('toggle-cam-follow');
    t.checked = true; t.dispatchEvent(new Event('change'));
    document.getElementById('player-card-${P}').click();   // 重新选中（他的卡片刚才是收起的）
    return { enabled: window.__v5.cameraFollow.enabled, follow: window.__v5.cameraFollow.player };
  })()`);
  check('turning it back on + selecting follows again',
    back.enabled === true && back.follow === P, JSON.stringify(back));
  const afterReset = await evaluate(`(() => {
    document.getElementById('btn-reset-cam').click();
    return { follow: window.__v5.cameraFollow.player, active: window.__v5.cameraFollow.active };
  })()`);
  check('reset view stops the follow',
    afterReset.follow === -1 && afterReset.active === false);

  // ── 7. 换回合继续跟同一个人 ──────────────────────────────────────────
  console.log('\nnext round');
  await evaluate(`(() => {
    if (!window.__v5.isPlaying) document.getElementById('btn-play').click();
    window.__v5.cameraFollow.focus(${P});
    return window.__v5.cameraFollow.player;
  })()`);
  const firstTick = await evaluate('window.__v5.round.ticks[0]');
  await evaluate('document.getElementById("btn-next-round").click()');
  let swapped = false;
  for (let i = 0; i < 40 && !swapped; i++) {
    await sleep(250);
    swapped = (await evaluate('window.__v5.round ? window.__v5.round.ticks[0] : -1')) !== firstTick;
  }
  check('the next round loaded', swapped);
  const roundFollow = await evaluate(`({
    follow: window.__v5.cameraFollow.player,
    selected: window.__v5.selectedPlayer,
    enabled: window.__v5.cameraFollow.enabled,
  })`);
  check('the follow carries over to the new round',
    roundFollow.follow === P && roundFollow.enabled === true,
    JSON.stringify(roundFollow));
  check('and his card is expanded again', roundFollow.selected === P);
  await evaluate(stepFrames(30, '1/30'));   // 换回合会重新飞一次（0.7s）
  const newRound = await evaluate(snapOf(P));
  check('the re-frame flight finished', newRound.animating === false);
  check('the camera still looks at him after the round swap',
    newRound.playerWorld != null && dist(newRound.target, newRound.playerWorld) < 0.01,
    newRound.playerWorld ? `${dist(newRound.target, newRound.playerWorld).toFixed(4)} m` : 'no state');

  // ── 8. 切到第一人称 = 停止跟随（切回来才不会瞬移）────────────────────
  console.log('\ncamera mode');
  const modeOut = await evaluate(`(() => {
    window.__v5.cameraFollow.focus(${P});
    const before = window.__v5.cameraFollow.player;
    window.__v5.setCameraMode('first');
    const after = window.__v5.cameraFollow.player;
    window.__v5.setCameraMode('orbit');
    return { before, after };
  })()`);
  check('switching away from orbit stops the follow',
    modeOut.before === P && modeOut.after === -1, JSON.stringify(modeOut));

  // ── 9. WASD 手动推机位 = 自动取消选中 / 停止跟随 ─────────────────────
  console.log('\nWASD pans the camera');
  const wasd = await evaluate(`(() => {
    window.__v5.cameraFollow.focus(${P});
    for (let k = 0; k < 30; k++) window.__v5.stepFrame(1/30, false);   // 走完飞行
    const c0 = window.__v5.camera.position.clone();
    const t0 = window.__v5.cameraFollow.target();
    const dir = c0.clone().set(0, 0, 0);
    window.__v5.camera.getWorldDirection(dir);
    dir.y = 0; dir.normalize();
    const selBefore = window.__v5.selectedPlayer;

    // 焦点在下拉框里按 W：Chrome 的下拉框本身就是"打字跳选项"，镜头不该动
    document.getElementById('speed-select').dispatchEvent(
      new KeyboardEvent('keydown', { code: 'KeyW', bubbles: true }));
    for (let k = 0; k < 5; k++) window.__v5.stepFrame(1/30, false);
    const inFieldMoved = window.__v5.camera.position.distanceTo(c0);
    const selInField = window.__v5.selectedPlayer;

    // 正常按 W（10 帧 × 1/30 s × 15 m/s ≈ 5 m）
    window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW', bubbles: true }));
    for (let k = 0; k < 10; k++) window.__v5.stepFrame(1/30, false);
    window.dispatchEvent(new KeyboardEvent('keyup', { code: 'KeyW', bubbles: true }));
    const c1 = window.__v5.camera.position.clone();
    const t1 = window.__v5.cameraFollow.target();
    const move = c1.clone().sub(c0);
    const flat = move.clone(); flat.y = 0;
    const off0 = [c0.x - t0[0], c0.y - t0[1], c0.z - t0[2]];
    const off1 = [c1.x - t1[0], c1.y - t1[1], c1.z - t1[2]];
    const rigid = Math.abs(off1[0]-off0[0]) + Math.abs(off1[1]-off0[1]) + Math.abs(off1[2]-off0[2]);

    // 松开之后必须立刻停
    const c2 = window.__v5.camera.position.clone();
    for (let k = 0; k < 10; k++) window.__v5.stepFrame(1/30, false);
    const drift = window.__v5.camera.position.distanceTo(c2);

    // A/D 左右、Q/E 升降：方向用**摄像机自己的本地 X 轴**（three 的 +X = 屏幕右）
    // 来对拍，不复用 scene.js 里那套 cross 公式
    const press = (code) => {
      const a = window.__v5.camera.position.clone();
      window.dispatchEvent(new KeyboardEvent('keydown', { code, bubbles: true }));
      for (let k = 0; k < 10; k++) window.__v5.stepFrame(1/30, false);
      window.dispatchEvent(new KeyboardEvent('keyup', { code, bubbles: true }));
      return window.__v5.camera.position.clone().sub(a);
    };
    window.__v5.camera.updateMatrixWorld();
    const rightAxis = c1.clone().setFromMatrixColumn(window.__v5.camera.matrixWorld, 0);
    rightAxis.y = 0; rightAxis.normalize();
    const dMove = press('KeyD');
    const aMove = press('KeyA');
    const eMove = press('KeyE');
    const qMove = press('KeyQ');

    return {
      selBefore, selInField,
      inFieldMoved,
      moved: move.length(),
      flatRatio: move.length() > 1e-9 ? flat.length() / move.length() : 0,
      alongView: flat.length() > 1e-9 ? flat.clone().normalize().dot(dir) : 0,
      rigid, drift,
      dRight: dMove.clone().setY(0).normalize().dot(rightAxis.clone().normalize()),
      aRight: aMove.clone().setY(0).normalize().dot(rightAxis.clone().normalize()),
      dFlat: Math.abs(dMove.y),
      eUp: eMove.y,
      eFlat: Math.hypot(eMove.x, eMove.z),
      qUp: qMove.y,
      selAfter: window.__v5.selectedPlayer,
      followAfter: window.__v5.cameraFollow.player,
      cardExpanded: !document.getElementById('player-detail-${P}').classList.contains('hidden'),
    };
  })()`);
  check('a player is selected and followed before panning',
    wasd.selBefore === P, `selected=${wasd.selBefore}`);
  check('WASD with a form field focused does not move the camera',
    wasd.inFieldMoved < 1e-6, `${wasd.inFieldMoved.toFixed(6)} m`);
  check('...and does not drop the selection', wasd.selInField === P, `selected=${wasd.selInField}`);
  check('W pans the camera', wasd.moved > 1, `${wasd.moved.toFixed(3)} m`);
  check('the pan is horizontal', wasd.flatRatio > 0.999, `${wasd.flatRatio.toFixed(4)}`);
  check('W goes forward along the view direction', wasd.alongView > 0.999,
    `${wasd.alongView.toFixed(4)}`);
  check('D strafes to screen-right', wasd.dRight > 0.999 && wasd.dFlat < 1e-6,
    `dot=${wasd.dRight.toFixed(4)} dy=${wasd.dFlat.toExponential(2)}`);
  check('A strafes to screen-left', wasd.aRight < -0.999,
    `dot=${wasd.aRight.toFixed(4)}`);
  check('E lifts the camera straight up', wasd.eUp > 1 && wasd.eFlat < 1e-6,
    `dy=${wasd.eUp.toFixed(3)} flat=${wasd.eFlat.toExponential(2)}`);
  check('Q lowers the camera', wasd.qUp < -1, `dy=${wasd.qUp.toFixed(3)}`);
  check('the view is unchanged (camera and target moved together)',
    wasd.rigid < 1e-6, `${wasd.rigid.toExponential(2)}`);
  check('panning drops the selection', wasd.selAfter === -1, `selected=${wasd.selAfter}`);
  check('panning stops the follow', wasd.followAfter === -1, `follow=${wasd.followAfter}`);
  check('and the card is collapsed', wasd.cardExpanded === false);
  check('releasing the key stops the camera', wasd.drift < 1e-6, `${wasd.drift.toFixed(6)} m`);

  console.log('\nconsole');
  const realErrors = consoleErrors.filter((e) => !/favicon|404 \(Not Found\)/i.test(e));
  check('no console errors', realErrors.length === 0, realErrors.slice(0, 3).join(' | '));
  check('no uncaught exceptions', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '));

  console.log();
  console.log(failed ? `${failed} FAILED` : 'camera follow: all checks passed');
} catch (err) {
  console.error('\nharness error:', err.message);
  failed++;
} finally {
  ws.close();
  proc.kill();
  await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* ignore */ }
}
process.exit(failed ? 1 : 0);
