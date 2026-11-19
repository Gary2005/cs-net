#!/usr/bin/env node
/**
 * 在**真实浏览器**里验证下蹲可视化（headless Chrome + CDP）。
 *
 * 前面的 test_crouch_payload.py 只验到「数据解对了」。这个脚本验的是**画出来的东西**：
 * 碰撞盒的 scale 是不是 72/54、底面是不是贴在脚底、眼睛小球的高度是不是 64.09/46.09、
 * 开关有没有真的把三个对象显隐掉、以及控制台有没有报错。
 *
 * 用法（需要 Chrome 和一个已经在跑的 visualizer）：
 *
 *     node visualizer/tests/test_crouch_render.mjs http://127.0.0.1:5099 /tmp/v7_alltick.csz
 *
 * 没找到 Chrome 就打印 SKIP 并以 0 退出。
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME_CANDIDATES = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  process.env.CHROME_PATH,
].filter(Boolean);

const BASE = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3];
const PORT = 9333 + (process.pid % 200);

const chrome = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chrome) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(0);
}
if (!CSZ) {
  console.log('SKIP: pass a .csz path as the second argument');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
function check(name, cond, detail = '') {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${!cond && detail ? `   ${detail}` : ''}`);
  if (!cond) failed++;
}

const profile = mkdtempSync(join(tmpdir(), 'crouch-render-'));
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
], { stdio: ['ignore', 'ignore', 'pipe'] });

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

try {
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: BASE });
  await sleep(2500);

  console.log(`chrome: ${chrome}`);
  console.log(`page  : ${BASE}\n`);

  check('the page booted the app', await evaluate('!!window.__v5'));

  console.log('\nload');
  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
  await sleep(1200);
  const fmt = await evaluate('window.__v5.summary && window.__v5.summary.format');
  check('summary reports cs2.demo.v7', fmt === 'cs2.demo.v7', String(fmt));
  check('summary reports has_crouch',
    await evaluate('!!(window.__v5.summary.meta||{}).has_crouch'));
  check('a round is loaded',
    await evaluate('!!(window.__v5.round && window.__v5.round.players.length)'));

  // 找一条真的在下蹲的采样点（取 crouch 最大的一处）
  const peak = await evaluate(`(() => {
    const r = window.__v5.round; let best = {c: -1, s: 0, p: 0};
    for (let s = 0; s < r.ticks.length; s++)
      for (let p = 0; p < r.players.length; p++) {
        const c = r.players[p].crouch[s];
        if (r.players[p].alive[s] && c > best.c) best = {c, s, p};
      }
    return best;
  })()`);
  console.log(`\nthe most-crouched sample: player ${peak.p} @ sample ${peak.s}, crouch=${peak.c.toFixed(3)}`);

  // 打开开关，跳到那一刻
  await evaluate(`(() => {
    const t = document.getElementById('toggle-crouch');
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
  await evaluate(`window.__v5.seekSample(${peak.s})`);
  await sleep(200);

  const dbg = await evaluate('window.__v5.crouchDebug');
  const players = await evaluate('window.__v5.players');
  const me = dbg[peak.p];
  const st = players[peak.p];

  console.log('\nrendered geometry (HU)');
  check('the crouch layer is visible', me && me.visible, JSON.stringify(me));
  check('the state carries crouch', st && st.crouch > 0.9, String(st && st.crouch));
  check('the state carries eye_height', st && Math.abs(st.eye_height - (64.0938 - 18 * st.crouch)) < 1e-3,
    String(st && st.eye_height));

  if (me && st) {
    const wantHull = st.crouching ? 54 : 72;
    check(`hull height is the stepped ${wantHull} HU`, Math.abs(me.hullH - wantHull) < 0.01,
      `${me.hullH}`);
    check('the hull base sits on the feet (base y == 0 HU)',
      Math.abs(me.hullBaseY) < 0.01, `${me.hullBaseY}`);
    check('the eye marker is at the state eye height',
      Math.abs(me.eyeY - st.eye_height) < 0.01, `${me.eyeY} vs ${st.eye_height}`);
    check('the ruler spans feet -> eye',
      Math.abs(me.rulerY - st.eye_height) < 0.01, `${me.rulerY}`);
    check('the crouched eye is ~46 HU, not 64',
      me.eyeY < 50, `${me.eyeY}`);
  }

  // ── 小人本身也要蹲下去（这次新增的重点）────────────────────────────────
  console.log('\nthe figure itself crouches')
  if (me && st) {
    const wantRatio = (72 - 18 * st.crouch) / 72;
    check('the body squashes with the continuous crouch',
      Math.abs(me.bodyRatio - wantRatio) < 1e-3, `${me.bodyRatio} vs ${wantRatio}`);
    check('the crouched figure is ~54 HU tall (not 72)',
      Math.abs(me.figureH - 54) < 0.2, `${me.figureH}`);
    check('the head drops with the body',
      me.headY < 55 && me.headY > 45, `${me.headY} HU`);
    check('the head is not squashed (scale.y stays 1)',
      Math.abs(me.headScaleY - 1) < 1e-6, String(me.headScaleY));
    check('the eye marker lands inside the head sphere',
      Math.abs(me.eyeY - me.headY) < 0.19 / 0.0254,
      `eye ${me.eyeY} vs head ${me.headY}`);
    check('the hp bar follows the head down',
      me.hpY > me.headY && me.hpY - me.headY < 15, `hp ${me.hpY} vs head ${me.headY}`);
    check('the crouched figure is still taller than the crouched eye',
      me.figureH > st.eye_height, `figure ${me.figureH} vs eye ${st.eye_height}`);
  }

  // 新小人的整体尺寸：**站着**应该正好占满 72 HU 高，宽度明显小于身高。
  // 必须在站立的采样点上量 —— 蹲下时头不缩（头本来就不该缩），
  // 所以蹲姿的轮廓会略高于名义的 54 HU，那是对的。
  //
  // 还必须是**站着不动**的采样点：走路时腿在摆，两只脚都会离地
  // L·(1−BOB)(1−cos θ)（全速时约 2.7 HU），"脚底贴地"这条就不成立了。
  // 摆腿本身由 test_walk_cycle.mjs 守。
  console.log('\nthe figure has sane proportions')
  const standForBbox = await evaluate(`(() => {
    const r = window.__v5.round, p = ${peak.p};
    for (let s = 0; s < r.ticks.length; s++)
      if (r.players[p].alive[s] && r.players[p].crouch[s] === 0
          && r.players[p].on_ground[s] && r.players[p].v[s] < 1) return s;
    return -1;
  })()`);
  check('found a standing-still sample to measure the figure', standForBbox >= 0, `${standForBbox}`);
  await evaluate(`window.__v5.seekSample(${standForBbox})`);
  const bbox = await evaluate(`(() => {
    const V = window.__v5.camera.position.constructor;
    const pts = [];
    let meshes = 0;
    for (const name of ['body-${peak.p}', 'pitch-${peak.p}']) {
      const root = window.__v5.scene.getObjectByName(name);
      root.traverse((o) => {
        if (!o.isMesh) return;
        meshes++;
        const g = o.geometry;
        if (!g.boundingBox) g.computeBoundingBox();
        const bb = g.boundingBox;
        for (const x of [bb.min.x, bb.max.x])
          for (const y of [bb.min.y, bb.max.y])
            for (const z of [bb.min.z, bb.max.z]) {
              const v = new V(x, y, z);
              o.localToWorld(v);
              pts.push(v);
            }
      });
    }
    const S = 0.0254;
    const ys = pts.map(v => v.y), xs = pts.map(v => v.x), zs = pts.map(v => v.z);
    const feetY = window.__v5.players[${peak.p}].z * S;
    return { meshes,
             h: (Math.max(...ys) - Math.min(...ys)) / S,
             base: (Math.min(...ys) - feetY) / S,
             w: (Math.max(...xs) - Math.min(...xs)) / S,
             d: (Math.max(...zs) - Math.min(...zs)) / S };
  })()`);
  check('the figure is built from several parts, not three primitives',
    bbox.meshes >= 8, JSON.stringify(bbox));
  check('the standing figure stands on the feet (base y == 0)',
    Math.abs(bbox.base) < 0.5, `${bbox.base} HU`);
  check('the standing figure is 72 HU tall, matching the standing hull',
    Math.abs(bbox.h - 72) < 3, `${bbox.h.toFixed(1)} HU`);
  check('the figure is narrower than it is tall (a person, not a box)',
    bbox.w < bbox.h * 0.75, JSON.stringify(bbox));
  // 量完回到蹲姿，后面的对照才有意义
  await evaluate(`window.__v5.seekSample(${peak.s})`);

  // 一个站着的玩家做对照：盒子必须是 72，眼睛必须是 64.09
  const standIdx = players.findIndex((p, i) => p && p.alive && p.crouch === 0 && dbg[i]);
  if (standIdx >= 0) {
    console.log('\nstanding control');
    const sd = dbg[standIdx], ss = players[standIdx];
    check('standing hull is 72 HU', Math.abs(sd.hullH - 72) < 0.01, `${sd.hullH}`);
    check('standing eye is 64.09 HU', Math.abs(sd.eyeY - 64.0938) < 0.01, `${sd.eyeY}`);
    check('standing eye != crouched eye', Math.abs(sd.eyeY - me.eyeY) > 10,
      `${sd.eyeY} vs ${me.eyeY}`);
    check('the standing figure is a full 72 HU tall',
      Math.abs(sd.figureH - 72) < 0.01, `${sd.figureH}`);
    check('the standing head sits at the standing eye height',
      Math.abs(sd.headY - 64.09) < 6, `${sd.headY} HU vs eye 64.09`);
    check('crouching actually shortens the figure',
      sd.figureH - me.figureH > 17, `${sd.figureH} vs ${me.figureH}`);
  }

  // 世界坐标：眼睛小球挂在一个带 yaw 旋转、平移到脚底的 group 上，
  // 所以局部 y 不等于世界 y。这里把整条链走完，验证
  // world_eye_y == (game_z + eye_height) * SCALE，x/z 也落在脚底附近。
  console.log('\nworld transform (game -> three)')
  const world = await evaluate(`(() => {
    const S = 0.0254, p = ${peak.p}, st = window.__v5.players[p];
    const pg = window.__v5.scene.getObjectByName('player-' + p);
    if (!pg || !pg.children.find(c => c.type === 'Mesh' && c.geometry.type === 'SphereGeometry'
        && c.material.depthTest === false)) return null;
    const eye = pg.children.find(c => c.type === 'Mesh'
      && c.geometry.type === 'SphereGeometry' && c.material.depthTest === false);
    const hull = pg.children.find(c => c.type === 'LineSegments');
    const v = eye.getWorldPosition(new (window.__v5.camera.position.constructor)());
    const hb = hull.getWorldPosition(new (window.__v5.camera.position.constructor)());
    return { eye: [v.x, v.y, v.z], hullCentre: [hb.x, hb.y, hb.z],
             want: [st.y * S, (st.z + st.eye_height) * S, st.x * S],
             feet: [st.y * S, st.z * S, st.x * S],
             hullScale: [hull.scale.x / S, hull.scale.y / S, hull.scale.z / S],
             eyeWorldVisible: eye.visible && pg.visible };
  })()`);
  if (!world) {
    check('found the eye mesh in the scene graph', false, 'not found');
  } else {
    const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    check('the eye mesh sits at the world eye position',
      d(world.eye, world.want) < 1e-6,
      `${world.eye.map(n => n.toFixed(4))} vs ${world.want.map(n => n.toFixed(4))}`);
    check('the eye is well above the feet in world space',
      world.eye[1] - world.feet[1] > 40 * 0.0254,
      `${(world.eye[1] - world.feet[1]) / 0.0254} HU`);
    check('the hull is 32 x 32 HU wide',
      Math.abs(world.hullScale[0] - 32) < 0.01 && Math.abs(world.hullScale[2] - 32) < 0.01,
      JSON.stringify(world.hullScale));
    check('the hull centre is half its height above the feet',
      Math.abs((world.hullCentre[1] - world.feet[1]) / 0.0254 - world.hullScale[1] / 2) < 0.01,
      `${(world.hullCentre[1] - world.feet[1]) / 0.0254} vs ${world.hullScale[1] / 2}`);
    check('the eye mesh is actually visible in the graph', world.eyeWorldVisible);
  }

  // 关掉开关必须真的隐藏
  console.log('\ntoggle')
  await evaluate(`(() => {
    const t = document.getElementById('toggle-crouch');
    t.checked = false; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(200);
  const off = await evaluate('window.__v5.crouchDebug');
  check('turning the toggle off hides the layer',
    off.every((d) => !d || !d.visible), JSON.stringify(off.filter(Boolean)[0]));
  await evaluate(`(() => {
    const t = document.getElementById('toggle-crouch');
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(200);

  // 玩家详情卡
  console.log('\nplayer detail card')
  const cardTxt = await evaluate(`(() => {
    const el = document.querySelector('.pd-crouch');
    return el ? el.textContent : null;
  })()`);
  check('the detail card has a crouch row', cardTxt !== null, String(cardTxt));

  // 这次重做了整个小人（多部件 + 非等比压缩），顺手回归一下被牵连的两个开关：
  // 透视地图（setXRayMode 现在要遍历整个身体组）和第一/第三人称相机
  // （以前硬编码 64 HU 眼高）。
  console.log('\nneighbouring features still work')

  // 下面四条测的是**透视地图**自己的行为（地图半透明 + 所有小人发光、不测深度）。
  // `选手 X 光`（默认开）会把被墙挡住的那些小人换成半透明幽灵材质，而幽灵**也是**
  // depthTest:false —— 两条断言就分不清"深度是谁改的"了。所以先把它关掉，测完
  // 再恢复：这一步同时也在验"这个开关能干净地退回原状"。
  await evaluate(`(() => {
    const t = document.getElementById('toggle-player-xray');
    if (t.checked) { t.checked = false; t.dispatchEvent(new Event('change')); }
  })()`);
  await sleep(300);

  await evaluate(`(() => {
    const t = document.getElementById('toggle-xray');
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(300);
  const xray = await evaluate(`(() => {
    const pg = window.__v5.scene.getObjectByName('player-${peak.p}');
    const out = { meshes: 0, noDepth: 0, glowing: 0 };
    for (const name of ['body-${peak.p}', 'pitch-${peak.p}']) {
      const root = window.__v5.scene.getObjectByName(name);
      if (!root) continue;
      root.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        out.meshes++;
        if (o.material.depthTest === false) out.noDepth++;
        if (o.material.emissive && o.material.emissiveIntensity > 0) out.glowing++;
      });
    }
    return out;
  })()`);
  check('xray reaches every part of the figure',
    xray.meshes >= 8 && xray.noDepth === xray.meshes, JSON.stringify(xray));
  // 这条同时是一个 bug 回归：回放中 updatePlayers 每帧都跑，它写的闪光自发光
  // 以前会把透视的发光覆盖掉，所以这里等几帧再读，发光必须还在。
  check('xray makes the whole figure glow (and it survives updatePlayers)',
    xray.glowing === xray.meshes, JSON.stringify(xray));

  await evaluate(`(() => {
    const t = document.getElementById('toggle-xray');
    t.checked = false; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(300);
  const xrayOff = await evaluate(`(() => {
    const out = { meshes: 0, depth: 0 };
    for (const name of ['body-${peak.p}', 'pitch-${peak.p}']) {
      const root = window.__v5.scene.getObjectByName(name);
      if (!root) continue;
      root.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        out.meshes++;
        if (o.material.depthTest === true) out.depth++;
      });
    }
    return out;
  })()`);
  check('turning xray off restores depth testing',
    xrayOff.depth === xrayOff.meshes, JSON.stringify(xrayOff));
  // 关掉之后面罩 / 视线锥的底光必须还在（不能被"还原"成黑色）
  const baseGlow = await evaluate(`(() => {
    const root = window.__v5.scene.getObjectByName('pitch-${peak.p}');
    let visor = null, dir = null;
    root.traverse((o) => {
      if (!o.isMesh) return;
      if (o.geometry.type === 'BoxGeometry') visor = o;
      if (o.geometry.type === 'ConeGeometry') dir = o;
    });
    return { visor: visor && visor.material.emissiveIntensity,
             dir: dir && dir.material.emissiveIntensity };
  })()`);
  check('the visor keeps its base glow after xray', baseGlow.visor > 0, JSON.stringify(baseGlow));
  check('the aim cone keeps its base glow after xray', baseGlow.dir > 0, JSON.stringify(baseGlow));

  // 把 `选手 X 光` 恢复成默认的开（上面为了隔离透视地图把它关掉了）
  await evaluate(`(() => {
    const t = document.getElementById('toggle-player-xray');
    if (!t.checked) { t.checked = true; t.dispatchEvent(new Event('change')); }
  })()`);
  await sleep(300);

  // 第一人称相机：蹲着的时候必须比站着低 18 HU。
  // 相机位置是 renderFrame 每帧写的，所以要先切模式、再等一帧。
  const findSamples = await evaluate(`(() => {
    const r = window.__v5.round, p = ${peak.p};
    let crouchS = -1, standS = -1;
    for (let s = 0; s < r.ticks.length; s++) {
      if (!r.players[p].alive[s]) continue;
      if (crouchS < 0 && r.players[p].crouch[s] > 0.99) crouchS = s;
      if (standS < 0 && r.players[p].crouch[s] === 0) standS = s;
    }
    return { crouchS, standS };
  })()`);
  if (findSamples && findSamples.crouchS >= 0 && findSamples.standS >= 0) {
    await evaluate(`window.__v5.setCameraMode('first');
                    window.__v5.setFocusedPlayer(${peak.p}); true`);
    const camAt = async (s) => {
      await evaluate(`window.__v5.seekSample(${s})`);
      // 相机是 renderFrame 在 rAF 循环里写的，所以必须**等真的渲染过一帧**再读。
      // 固定 sleep 会偶发拿到上一帧的相机位置（实测 250ms 不够）。
      await evaluate(
        'new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))', true);
      return evaluate(`(() => {
        const S = 0.0254, st = window.__v5.players[${peak.p}];
        return { cam: window.__v5.camera.position.y,
                 feet: st.z * S,
                 eye: st.eye_height != null ? st.eye_height : 64.0938 };
      })()`);
    };
    const c = await camAt(findSamples.crouchS);
    const s0 = await camAt(findSamples.standS);
    const camEyeC = (c.cam - c.feet) / 0.0254;
    const camEyeS = (s0.cam - s0.feet) / 0.0254;
    check('the first-person camera sits at the crouched eye height',
      Math.abs(camEyeC - c.eye) < 0.5, `${camEyeC.toFixed(2)} vs ${c.eye.toFixed(2)}`);
    check('the first-person camera sits at the standing eye height',
      Math.abs(camEyeS - s0.eye) < 0.5, `${camEyeS.toFixed(2)} vs ${s0.eye.toFixed(2)}`);
    check('the first-person camera drops 18 HU when crouching',
      Math.abs((camEyeS - camEyeC) - 18) < 0.5,
      `${(camEyeS - camEyeC).toFixed(2)} HU`);
    await evaluate(`window.__v5.setCameraMode('orbit');
                    window.__v5.setFocusedPlayer(-1); true`);
    await sleep(200);
  } else {
    check('found a standing and a crouched sample for the camera check', false,
      JSON.stringify(findSamples));
  }

  console.log('\nsweep every 37th sample of the round')
  const sweep = await evaluate(`(() => {
    const r = window.__v5.round, bad = {hull: 0, eye: 0, base: 0, n: 0, body: 0, head: 0};
    for (let s = 0; s < r.ticks.length; s += 37) {
      window.__v5.seekSample(s);
      const d = window.__v5.crouchDebug, st = window.__v5.players;
      for (let p = 0; p < d.length; p++) {
        if (!d[p] || !st[p] || !st[p].alive) continue;
        bad.n++;
        if (![72, 54].some(v => Math.abs(d[p].hullH - v) < 0.01)) bad.hull++;
        if (d[p].eyeY < 46.0 || d[p].eyeY > 64.1) bad.eye++;
        if (Math.abs(d[p].hullBaseY) > 0.01) bad.base++;
        // 身体必须是连续的：figureH == 72 - 18*crouch
        if (Math.abs(d[p].figureH - (72 - 18 * st[p].crouch)) > 0.05) bad.body++;
        if (Math.abs(d[p].headScaleY - 1) > 1e-6) bad.head++;
      }
    }
    return bad;
  })()`);
  check('sampled states were checked', sweep.n > 50, JSON.stringify(sweep));
  check('every hull is exactly 72 or 54', sweep.hull === 0, JSON.stringify(sweep));
  check('every eye is inside [46.09, 64.09]', sweep.eye === 0, JSON.stringify(sweep));
  check('every hull base stays on the feet', sweep.base === 0, JSON.stringify(sweep));
  check('every figure height is exactly 72 - 18*crouch', sweep.body === 0, JSON.stringify(sweep));
  check('no head is ever squashed', sweep.head === 0, JSON.stringify(sweep));

  console.log('\nconsole')
  const realErrors = consoleErrors.filter((e) => !/favicon|404 \(Not Found\)/i.test(e));
  check('no console errors', realErrors.length === 0, realErrors.slice(0, 3).join(' | '));
  check('no uncaught exceptions', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '));

  console.log();
  if (failed) {
    console.log(`${failed} FAILED`);
  } else {
    console.log('crouch render: all checks passed');
  }
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
