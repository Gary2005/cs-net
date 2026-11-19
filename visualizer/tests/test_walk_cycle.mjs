/**
 * test_walk_cycle.mjs — 走路摆腿的测试。
 *
 * 分两块：
 *
 * **A. 步态数据模型**（纯 node，不需要浏览器，秒级）
 *   `buildGait()` 直接 import 进来跑合成回合。重点守三条不变量：
 *     - 相位只跟**水平距离**走（同距离 → 同相位增量；竖直位移不算）
 *     - 死人 / 空中 / 传送不推进相位
 *     - 摆幅跟**速度**走，站着不动就是 0
 *   还有一条最重要的：**相位是采样点的纯函数** —— 来回 seek、倒放，同一个
 *   采样点的姿势必须一模一样（这就是当初不用"按时间积分"的原因）。
 *
 * **B. 画出来的腿**（headless Chrome + CDP，读真实场景图）
 *   直接量两只脚的**世界坐标**，和解析公式对拍：
 *     - 脚离地 = L·(1−BOB)·(1−cos θ)      （θ = 摆角）
 *     - 胯高   = P_LEG_TOP·k·(1 − BOB(1−cos θ))
 *     - 前后差 = −2L·sin θ
 *     - 左右间距恒定、脚永远不低于地面、身高永远是 72 − 18·crouch（头不受步态影响）
 *   再验证频率：一段路程上的摆角过零次数 ≈ 2 × 距离 / GAIT_STRIDE_HU。
 *
 * 用法：
 *   python visualizer/server.py --port 5099 &
 *   node visualizer/tests/test_walk_cycle.mjs http://127.0.0.1:5099 /path/to/match.csz
 */

import {
  buildGait, GAIT_STRIDE_HU, GAIT_MAX_SWING, GAIT_RUN_SPEED_HU,
  GAIT_CROUCH_STRIDE, GAIT_CROUCH_SWING, GAIT_TELEPORT_HU, GAIT_BOB,
} from '../static/js/replay-core.js';

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}

const HU = 39.37007874015748;   // 1 m = 39.37 HU
const P_LEG_TOP = 0.78;         // 和 visuals.js 保持一致
const TICKRATE = 64;

/**
 * 合成一个回合。`players[i]` 的每个数组都由 `spec` 逐采样给出。
 * spec: { x[], y[], v[], on_ground[], alive[], crouch[] }
 */
function synthRound(players, n) {
  return {
    ticks: Array.from({ length: n }, (_, i) => i + 1000),
    round_seconds: Array.from({ length: n }, (_, i) => i / TICKRATE),
    players: players.map((p) => ({
      // 注意：游戏坐标里 (x, y) 是**水平**的，z 才是竖直的。步态只看 (x, y)。
      x: Float32Array.from(p.x),
      y: Float32Array.from(p.y),
      z: Float32Array.from(p.z ?? new Array(n).fill(0)),
      v: Float32Array.from(p.v),
      on_ground: Uint8Array.from(p.on_ground ?? new Array(n).fill(1)),
      alive: Uint8Array.from(p.alive ?? new Array(n).fill(1)),
      crouch: Float32Array.from(p.crouch ?? new Array(n).fill(0)),
    })),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\nA. gait data model (pure node)')

{
  // 匀速直线跑：每采样走 d HU
  const n = 400;
  const d = 250 / TICKRATE;                        // 全速跑，每采样 3.906 HU
  const rnd = synthRound([{
    x: Array.from({ length: n }, (_, i) => i * d),
    y: new Array(n).fill(0),
    v: new Array(n).fill(250),
  }], n);
  const [g] = buildGait(rnd);

  check('buildGait returns one entry per player', g && g.phase.length === n && g.amp.length === n);

  // 相位增量 = 2π·d/stride，恒定
  let maxErr = 0;
  for (let i = 2; i < n; i++) {
    let dp = g.phase[i] - g.phase[i - 1];
    if (dp < 0) dp += 2 * Math.PI;                 // 折回
    const want = (2 * Math.PI * d / GAIT_STRIDE_HU) % (2 * Math.PI);
    maxErr = Math.max(maxErr, Math.min(Math.abs(dp - want), Math.abs(dp - want + 2 * Math.PI)));
  }
  check('phase advances by exactly 2π·distance/stride each sample', maxErr < 1e-5,
    `max err ${maxErr}`);

  // 频率：整段路程应该正好走 总距离/stride 个周期
  const totalDist = d * (n - 1);
  const cycles = totalDist / GAIT_STRIDE_HU;
  console.log(`       (${totalDist.toFixed(0)} HU over ${n} samples → ${cycles.toFixed(2)} cycles, `
    + `${(cycles / ((n - 1) / TICKRATE)).toFixed(2)} cycles/s = `
    + `${(2 * cycles / ((n - 1) / TICKRATE)).toFixed(2)} steps/s)`);
  check('full-speed cadence is a plausible sprint (3.5–5.5 steps/s)',
    (2 * cycles / ((n - 1) / TICKRATE)) > 3.5 && (2 * cycles / ((n - 1) / TICKRATE)) < 5.5);

  check('phase stays inside [0, 2π)', g.phase.every((p) => p >= 0 && p < 2 * Math.PI));

  // 摆幅：全速应该顶到上限
  const ampMax = Math.max(...g.amp);
  check('full speed reaches the maximum swing amplitude',
    Math.abs(ampMax - GAIT_MAX_SWING) < 0.02, `${ampMax}`);

  // 站起来不动：摆幅必须是 0
  const still = synthRound([{
    x: new Array(n).fill(5), y: new Array(n).fill(5), v: new Array(n).fill(0),
  }], n);
  const [gs] = buildGait(still);
  check('standing still never advances the phase',
    gs.phase.every((p) => p === 0), `max ${Math.max(...gs.phase)}`);
  check('standing still gives zero amplitude',
    gs.amp.every((a) => a === 0), `max ${Math.max(...gs.amp)}`);
}

{
  // 摆幅必须**单调**随速度增长
  const n = 300;
  const speeds = [0, 25, 50, 100, 150, 200, 250];
  const players = speeds.map((v) => ({
    x: Array.from({ length: n }, (_, i) => i * v / TICKRATE),
    y: new Array(n).fill(0),
    v: new Array(n).fill(v),
  }));
  const gait = buildGait(synthRound(players, n));
  const amps = gait.map((g) => g.amp[n - 1]);
  console.log('       ' + speeds.map((s, i) => `${s}→${amps[i].toFixed(3)}`).join('  '));
  let mono = true;
  for (let i = 1; i < amps.length; i++) if (amps[i] < amps[i - 1] - 1e-6) mono = false;
  check('amplitude grows monotonically with speed', mono, JSON.stringify(amps));
  check('amplitude is exactly proportional to speed below the cap',
    speeds.slice(1, 5).every((v, i) => Math.abs(amps[i + 1] - GAIT_MAX_SWING * v / GAIT_RUN_SPEED_HU) < 0.02),
    JSON.stringify(amps));
  check('amplitude saturates above the run speed',
    Math.abs(amps[6] - GAIT_MAX_SWING) < 0.02, `${amps[6]}`);
}

{
  // 竖直位移（下落 / 跳）不该让腿走起来。
  // 游戏坐标里 (x, y) 是水平的、z 才是竖直的，所以下落要写进 z ——
  // 写进 y 的话那就是真的在水平移动，腿当然该走。
  const n = 200;
  const rnd = synthRound([{
    x: new Array(n).fill(0),
    y: new Array(n).fill(0),
    z: Array.from({ length: n }, (_, i) => -i * 8),      // 纯下落
    v: new Array(n).fill(250),                            // 速度列非零也没用
    on_ground: new Array(n).fill(0),
  }], n);
  const [g] = buildGait(rnd);
  check('falling straight down does not advance the phase',
    g.phase.every((p) => p === 0), `max ${Math.max(...g.phase)}`);
  check('airborne amplitude decays to ~0',
    g.amp[n - 1] < 1e-3, `${g.amp[n - 1]}`);
}

{
  // 传送 / 重生：单个采样跳一大步，不能推进相位
  const n = 100;
  const x = new Array(n).fill(0);
  for (let i = 50; i < n; i++) x[i] = 5000;              // 第 50 个采样瞬移
  const rnd = synthRound([{ x, y: new Array(n).fill(0), v: new Array(n).fill(0) }], n);
  const [g] = buildGait(rnd);
  check('a teleport bigger than GAIT_TELEPORT_HU does not advance the phase',
    g.phase.every((p) => p === 0), `max ${Math.max(...g.phase)}`);
  check('the teleport threshold is well above one sample of full-speed running',
    GAIT_TELEPORT_HU > 250 / TICKRATE * 10);
}

{
  // 死了不推进相位
  const n = 200;
  const d = 250 / TICKRATE;
  const alive = new Array(n).fill(1);
  for (let i = 100; i < n; i++) alive[i] = 0;
  const rnd = synthRound([{
    x: Array.from({ length: n }, (_, i) => i * d),
    y: new Array(n).fill(0), v: new Array(n).fill(250), alive,
  }], n);
  const [g] = buildGait(rnd);
  const frozen = g.phase.slice(100).every((p) => Math.abs(p - g.phase[99]) < 1e-6);
  check('a dead player stops advancing the phase', frozen);
  check('a dead player has zero amplitude', g.amp[n - 1] < 1e-6, `${g.amp[n - 1]}`);
}

{
  // 蹲走：步幅更短（同样距离摆得更多）+ 摆幅更小
  const n = 400;
  const d = 250 / TICKRATE;
  const x = Array.from({ length: n }, (_, i) => i * d);
  const y = new Array(n).fill(0);
  const v = new Array(n).fill(250);
  const stand = buildGait(synthRound([{ x, y, v }], n))[0];
  const duck = buildGait(synthRound([{ x, y, v, crouch: new Array(n).fill(1) }], n))[0];

  const span = (g) => {
    let s = 0;
    for (let i = 1; i < n; i++) { let dp = g.phase[i] - g.phase[i - 1]; if (dp < 0) dp += 2 * Math.PI; s += dp; }
    return s;
  };
  const ratio = span(duck) / span(stand);
  check('crouch-walking takes shorter strides (more phase per metre)',
    Math.abs(ratio - 1 / GAIT_CROUCH_STRIDE) < 0.02, `${ratio.toFixed(3)} vs ${1 / GAIT_CROUCH_STRIDE}`);
  check('crouch-walking swings less',
    Math.abs(duck.amp[n - 1] - GAIT_MAX_SWING * GAIT_CROUCH_SWING) < 0.02,
    `${duck.amp[n - 1]} vs ${GAIT_MAX_SWING * GAIT_CROUCH_SWING}`);
}

{
  // 纯函数：同一个回合算两次必须逐位相同
  const n = 500;
  const rnd = synthRound([{
    x: Array.from({ length: n }, (_, i) => Math.sin(i / 7) * 100 + i * 2),
    y: Array.from({ length: n }, (_, i) => Math.cos(i / 11) * 100),
    v: Array.from({ length: n }, (_, i) => (i % 40 < 20 ? 250 : 0)),
    alive: Array.from({ length: n }, (_, i) => (i % 97 === 0 ? 0 : 1)),
    on_ground: Array.from({ length: n }, (_, i) => (i % 31 < 5 ? 0 : 1)),
    crouch: Array.from({ length: n }, (_, i) => (i % 53 < 15 ? 1 : 0)),
  }], n);
  const a = buildGait(rnd)[0];
  const b = buildGait(rnd)[0];
  check('buildGait is deterministic (bit-identical across calls)',
    a.phase.every((v, i) => v === b.phase[i]) && a.amp.every((v, i) => v === b.amp[i]));
  // 关键：相位与"怎么走到这个采样点的"无关 —— 这正是拖时间轴不跳的原因
  const partial = buildGait({ ...rnd, ticks: rnd.ticks.slice(0, 200), round_seconds: rnd.round_seconds.slice(0, 200) })[0];
  check('a prefix of the round produces the same prefix of the phase',
    partial.phase.every((v, i) => v === a.phase[i]),
    'phase must depend only on the data, never on playback history');
}

// ═══════════════════════════════════════════════════════════════════════════
// B. 渲染出来的腿
const URL = process.argv[2];
const CSZ = process.argv[3];
if (!URL || !CSZ) {
  console.log(`\n(跳过渲染部分：用法 node test_walk_cycle.mjs <server-url> <match.csz>)`);
  console.log(`\nwalk cycle (data model only): ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
}

const { spawn } = await import('node:child_process');
const { mkdtempSync } = await import('node:fs');
const { tmpdir } = await import('node:os');
const { join } = await import('node:path');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const CHROME = process.env.CHROME_PATH
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));

const profile = mkdtempSync(join(tmpdir(), 'walk-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-sandbox',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1000,700', 'about:blank',
]);
let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  await sleep(250);
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    wsUrl = list.find((t) => t.type === 'page')?.webSocketDebuggerUrl;
  } catch { /* not up yet */ }
}
if (!wsUrl) { chrome.kill(); console.error('连不上 CDP'); process.exit(1); }
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
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
  if (r.result?.exceptionDetails) {
    throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 900));
  }
  return r.result?.result?.value;
};

await send('Runtime.enable');
await send('Page.enable');
await send('Network.enable');
await send('Network.setCacheDisabled', { cacheDisabled: true });
await send('Page.navigate', { url: URL });
for (let i = 0; i < 60; i++) {
  await sleep(500);
  if (await evaluate('!!(window.__v5 && window.__v5.loadPath)')) break;
}

console.log('\nload')
await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
await sleep(1800);
check('a round is loaded', await evaluate('!!window.__v5.round'));

// 找一个"连续静止"的采样点（不是刚停下来那种 —— 摆幅要时间衰减）
const stillAt = await evaluate(`(() => {
  const r = window.__v5.round;
  for (let p = 0; p < r.players.length; p++) {
    let run = 0;
    for (let s = 0; s < r.ticks.length; s++) {
      const still = r.players[p].alive[s] && r.players[p].on_ground[s] && r.players[p].v[s] < 1;
      run = still ? run + 1 : 0;
      if (run >= 30) return { p, s };
    }
  }
  return null;
})()`);
check('found a player that stands still for 30+ samples', !!stillAt, JSON.stringify(stillAt));

console.log('\nthe legs at rest')
if (stillAt) {
  await evaluate(`window.__v5.seekSample(${stillAt.s})`);
  const d = (await evaluate('window.__v5.crouchDebug'))[stillAt.p];
  check('a player still for 30 samples has (almost) no leg swing',
    Math.abs(d.legSwingL) < 0.02 && Math.abs(d.legSwingR) < 0.02,
    `L=${d.legSwingL} R=${d.legSwingR}`);
  check('both feet are on the ground when standing still',
    Math.abs(d.footL) < 0.1 && Math.abs(d.footR) < 0.1,
    `L=${d.footL} R=${d.footR} HU`);
  check('the hip is at the full standing height when standing still',
    Math.abs(d.hipY - P_LEG_TOP * HU) < 0.1, `${d.hipY} vs ${P_LEG_TOP * HU}`);
  check('the legs are together (no fore/aft split)', Math.abs(d.footSpread) < 0.5,
    `${d.footSpread} HU`);
}

// 找一个全速跑的采样点
const runAt = await evaluate(`(() => {
  const r = window.__v5.round;
  for (let s = 1; s < r.ticks.length; s++) {
    for (let p = 0; p < r.players.length; p++) {
      if (r.players[p].alive[s] && r.players[p].on_ground[s] && r.players[p].v[s] > 245) {
        return { p, s };
      }
    }
  }
  return null;
})()`);
check('found a player running at full speed', !!runAt, JSON.stringify(runAt));

console.log('\nthe legs while running')
if (runAt) {
  // 在一个完整周期上采样，验证解析公式
  const trace = await evaluate(`(() => {
    const out = [];
    const p = ${runAt.p}, s0 = ${runAt.s};
    for (let k = 0; k < 32; k++) {
      window.__v5.seekSample(s0 + k);
      const st = window.__v5.players[p];
      const d = window.__v5.crouchDebug[p];
      out.push({ v: st.v, crouch: st.crouch, ...d });
    }
    return out;
  })()`);

  const maxSwing = Math.max(...trace.map((t) => Math.abs(t.legSwingL)));
  check('running reaches (close to) the maximum swing amplitude',
    maxSwing > GAIT_MAX_SWING * 0.9, `${maxSwing} vs ${GAIT_MAX_SWING}`);
  console.log(`       (max |swing| = ${maxSwing.toFixed(3)} rad = `
    + `${(maxSwing * 180 / Math.PI).toFixed(1)}°)`);

  // 两条腿必须严格反相
  const worstOpposite = Math.max(...trace.map((t) => Math.abs(t.legSwingL + t.legSwingR)));
  check('the two legs swing exactly out of phase', worstOpposite < 1e-6, `${worstOpposite}`);

  // 解析对拍：脚离地 / 胯高 / 前后差
  let footErr = 0, hipErr = 0, spreadErr = 0, lateralErr = 0, below = 0, figErr = 0;
  for (const t of trace) {
    const k = (72 - 18 * t.crouch) / 72;
    const L = P_LEG_TOP * k;                       // 腿长，m
    const sw = t.legSwingL;
    const wantFoot = L * (1 - GAIT_BOB) * (1 - Math.cos(sw)) * HU;
    const wantHip = P_LEG_TOP * k * (1 - GAIT_BOB * (1 - Math.cos(sw))) * HU;
    const wantSpread = -2 * L * Math.sin(sw) * HU;
    footErr = Math.max(footErr, Math.abs(t.footL - wantFoot), Math.abs(t.footR - wantFoot));
    hipErr = Math.max(hipErr, Math.abs(t.hipY - wantHip));
    spreadErr = Math.max(spreadErr, Math.abs(t.footSpread - wantSpread));
    lateralErr = Math.max(lateralErr, Math.abs(t.footLateral + 0.26 * HU));
    if (Math.min(t.footL, t.footR) < -0.05) below++;
    if (Math.abs(t.figureH - (72 - 18 * t.crouch)) > 0.05) figErr++;
  }
  check('foot height above the ground matches L·(1−BOB)(1−cos θ)', footErr < 0.05,
    `max err ${footErr.toFixed(4)} HU`);
  check('hip height matches P_LEG_TOP·k·(1 − BOB(1−cos θ))', hipErr < 0.05,
    `max err ${hipErr.toFixed(4)} HU`);
  check('fore/aft foot split matches −2L·sin θ', spreadErr < 0.05,
    `max err ${spreadErr.toFixed(4)} HU`);
  check('the left/right leg spacing never changes', lateralErr < 0.05,
    `max err ${lateralErr.toFixed(4)} HU`);
  check('a foot never sinks below the ground', below === 0, `${below} samples below`);
  check('the gait never changes the figure height (the head is not affected)',
    figErr === 0, `${figErr} samples off`);

  const spreadMax = Math.max(...trace.map((t) => Math.abs(t.footSpread)));
  console.log(`       (stride: feet ${spreadMax.toFixed(1)} HU apart fore/aft, `
    + `lift ${Math.max(...trace.map((t) => Math.max(t.footL, t.footR))).toFixed(2)} HU, `
    + `hip ${Math.min(...trace.map((t) => t.hipY)).toFixed(1)}–`
    + `${Math.max(...trace.map((t) => t.hipY)).toFixed(1)} HU)`);
}

console.log('\nsweep: amplitude vs speed, and determinism')
{
  const sweep = await evaluate(`(() => {
    const r = window.__v5.round;
    const bySpeed = {}; let n = 0, below = 0, figBad = 0, asym = 0, maxSpread = 0;
    for (let s = 1; s < r.ticks.length; s += 3) {
      window.__v5.seekSample(s);
      const st = window.__v5.players, d = window.__v5.crouchDebug;
      for (let p = 0; p < 10; p++) {
        if (!st[p] || !st[p].alive || !st[p].on_ground) continue;
        const g = d[p]; if (!g) continue;
        n++;
        const b = Math.round(st[p].v / 50) * 50;
        (bySpeed[b] = bySpeed[b] || []).push(Math.abs(g.legSwingL));
        if (Math.min(g.footL, g.footR) < -0.05) below++;
        if (Math.abs(g.figureH - (72 - 18 * st[p].crouch)) > 0.05) figBad++;
        if (Math.abs(g.legSwingL + g.legSwingR) > 1e-6) asym++;
        maxSpread = Math.max(maxSpread, Math.abs(g.footSpread));
      }
    }
    const mean = {};
    for (const k of Object.keys(bySpeed).sort((a, b) => a - b)) {
      mean[k] = bySpeed[k].reduce((a, b) => a + b, 0) / bySpeed[k].length;
    }
    return { n, below, figBad, asym, maxSpread, mean };
  })()`);
  check('the sweep sampled a lot of grounded samples', sweep.n > 2000, `${sweep.n}`);
  check('no foot ever goes below the ground across the whole round',
    sweep.below === 0, `${sweep.below}`);
  check('the figure height is always 72 − 18·crouch across the whole round',
    sweep.figBad === 0, `${sweep.figBad}`);
  check('the legs are always exactly out of phase', sweep.asym === 0, `${sweep.asym}`);
  console.log('       mean |swing| by speed (HU/s): '
    + Object.entries(sweep.mean).map(([k, v]) => `${k}→${v.toFixed(3)}`).join('  '));
  const keys = Object.keys(sweep.mean).map(Number).sort((a, b) => a - b);
  let mono = true;
  for (let i = 1; i < keys.length; i++) {
    if (sweep.mean[keys[i]] < sweep.mean[keys[i - 1]] - 0.005) mono = false;
  }
  check('rendered swing amplitude grows monotonically with speed', mono,
    JSON.stringify(sweep.mean));

  // 纯函数：seek 到同一采样点两次，姿势必须逐位相同
  const det = await evaluate(`(() => {
    const r = window.__v5.round;
    const picks = [10, 500, 1234, 77];
    const read = () => window.__v5.crouchDebug.map((d) => d && [
      d.legSwingL, d.legSwingR, d.hipY, d.footL, d.footR, d.footSpread,
    ]);
    const first = {};
    for (const s of picks) {
      if (s >= r.ticks.length) continue;
      window.__v5.seekSample(s);
      first[s] = JSON.stringify(read());
    }
    // 到处乱跳之后再回来
    let bad = 0, checked = 0;
    for (const s of [999, 3, 2000, 42, 1500]) {
      if (s < r.ticks.length) window.__v5.seekSample(s);
    }
    for (const s of picks) {
      if (s >= r.ticks.length) continue;
      window.__v5.seekSample(s);
      checked++;
      if (JSON.stringify(read()) !== first[s]) bad++;
    }
    return { bad, checked };
  })()`);
  check('the pose at a sample is identical after seeking around (pure function)',
    det.bad === 0 && det.checked >= 3, JSON.stringify(det));
}

console.log('\ncadence')
{
  // 一段匀速跑，数摆角过零次数 → 和 2×距离/stride 对比
  const cad = await evaluate(`(() => {
    const r = window.__v5.round;
    // 找一个连续全速跑的窗口
    let best = null;
    for (let p = 0; p < r.players.length; p++) {
      let s = 1;
      while (s < r.ticks.length) {
        if (!(r.players[p].alive[s] && r.players[p].on_ground[s] && r.players[p].v[s] > 240)) { s++; continue; }
        let e = s;
        while (e < r.ticks.length && r.players[p].alive[e] && r.players[p].on_ground[e]
               && r.players[p].v[e] > 240) e++;
        if (!best || e - s > best.e - best.s) best = { p, s, e };
        s = e + 1;
      }
    }
    if (!best || best.e - best.s < 40) return null;
    const sw = [];
    for (let s = best.s; s < best.e; s++) {
      window.__v5.seekSample(s);
      sw.push(window.__v5.crouchDebug[best.p].legSwingL);
    }
    // 水平距离
    let dist = 0;
    for (let s = best.s + 1; s < best.e; s++) {
      const a = r.players[best.p];
      dist += Math.hypot(a.x[s] - a.x[s - 1], a.y[s] - a.y[s - 1]);
    }
    let crossings = 0;
    for (let i = 1; i < sw.length; i++) if ((sw[i - 1] < 0) !== (sw[i] < 0)) crossings++;
    return { p: best.p, n: sw.length, dist, crossings };
  })()`);
  check('found a long full-speed run to measure cadence', !!cad, JSON.stringify(cad));
  if (cad) {
    // 每个周期有 2 次过零
    const cycles = cad.crossings / 2;
    const want = cad.dist / GAIT_STRIDE_HU;
    const err = Math.abs(cycles - want) / want;
    console.log(`       (${cad.dist.toFixed(0)} HU in ${cad.n} samples → ${cycles} cycles `
      + `measured vs ${want.toFixed(2)} expected, ${(err * 100).toFixed(1)}% off)`);
    check('the measured cadence matches distance / GAIT_STRIDE_HU within 10%',
      err < 0.10, `${cycles} vs ${want.toFixed(2)}`);
  }
}

console.log('\nconsole');
check('no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 2).join(' | '));

console.log(`\nwalk cycle: ${passed} passed, ${failed} failed`);
ws.close();
chrome.kill();
process.exit(failed === 0 ? 0 : 1);
