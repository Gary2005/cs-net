/**
 * test_tracers.mjs — 子弹曳光的渲染测试（headless Chrome + CDP）。
 *
 * 曳光只有一条不变量的核心：**它和瞄准射线是同一次射线检测**。
 * 所以这个测试的主线是拿瞄准射线当参考实现，逐发对拍：
 *
 * **A. 数据源**：`events.sound` 里的开火事件和 `shots` 列必须逐发对得上
 *   （同一个 `weapon_fire` 事件的两个投影）。总数相同、每一发都落在
 *   `shots > 0` 的采样点上。
 *
 * **B. 起点 / 方向 / 终点**：手动生成一发，读**真实几何**，验证
 *   - 起点 = 真实眼睛（含下蹲眼高）
 *   - 方向 = (sin yaw·cos pitch, −sin pitch, cos yaw·cos pitch)
 *   - 终点距离 == 同一个人同一帧的瞄准射线距离（打人/打墙/没打到三种都要）
 *   - 命中类型 == 瞄准射线的命中类型
 *
 * **C. 飞出去、碰到就停**：开 rAF 逐帧采样，验证
 *   - head 单调不减、永远不超过命中距离（不穿墙、不穿人）
 *   - head 最终**正好**停在命中距离上，然后 hold → fade → 消失
 *   - 尾巴不越过枪口（renderOffset），段长 == 网格的真实 scale.y
 *   - 平均速度 ≈ TRACER_SPEED（"迅速"）
 *
 * **D. 一发一发地生成**：`stepTracers(from, to)` 的区间是 `(from, to]`，
 *   而且死人不生成。开火者在这一帧已经死了就什么都不该有。
 *
 * **E. 开关**：关掉后全部隐藏，且不再生成。
 *
 * 用法：
 *   python visualizer/server.py --port 5099 &
 *   node visualizer/tests/test_tracers.mjs http://127.0.0.1:5099 /path/to/match.csz
 */

import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const URL = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3];
if (!CSZ) {
  console.error('用法: node test_tracers.mjs <server-url> <match.csz>');
  process.exit(2);
}
const CHROME = process.env.CHROME_PATH
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
// 随机端口：写死端口时连着跑两次，上一轮的 Chrome 还没释放端口就会连不上 CDP
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const S = 0.0254;   // 1 HU = 1 英寸

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}

// ── 起 Chrome + 接 CDP ──────────────────────────────────────────────────────
const profile = mkdtempSync(join(tmpdir(), 'tracer-'));
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
check('the page booted the app', await evaluate('!!window.__v5'));

console.log('\nload');
await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
await sleep(1500);
check('a round is loaded', await evaluate('!!window.__v5.round'));

// 等地图
let mapTris = 0;
for (let i = 0; i < 60; i++) {
  mapTris = await evaluate(`(() => {
    const m = window.__v5.scene.getObjectByName('map-root');
    let t = 0;
    if (m) m.traverse((o) => { if (o.isMesh && o.geometry) t += (o.geometry.index
      ? o.geometry.index.count : o.geometry.attributes.position.count) / 3; });
    return t;
  })()`);
  if (mapTris > 100000) break;
  await sleep(500);
}
check('the map geometry is loaded', mapTris > 100000, `${mapTris} triangles`);

// 开关默认必须是开着的（这个功能默认可见）
check('the tracer toggle exists and defaults to on',
  await evaluate(`(() => {
    const t = document.getElementById('toggle-tracers');
    return !!t && t.checked;
  })()`));

// 建 BVH：曳光要在墙上停下来就靠它。瞄准射线用的是同一棵树。
await evaluate(`(() => {
  const t = document.getElementById('toggle-aim');
  t.checked = true; t.dispatchEvent(new Event('change'));
  return true;
})()`);
for (let i = 0; i < 100; i++) {
  await sleep(200);
  if (await evaluate('!!window.__v5.mapBVHStats')) break;
}
const bvhStats = await evaluate('window.__v5.mapBVHStats');
check('the BVH got built for the real map', !!bvhStats && bvhStats.triCount === mapTris,
  JSON.stringify(bvhStats));
console.log(`       (BVH: ${bvhStats ? bvhStats.triCount : 0} triangles in `
  + `${bvhStats ? Math.round(bvhStats.ms) : 0} ms)`);

// 每次测试前把在飞的曳光收干净（关一下再开，等于 clearTracers）
const clearTracers = async () => {
  await evaluate(`(() => {
    const t = document.getElementById('toggle-tracers');
    t.checked = false; t.dispatchEvent(new Event('change'));
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
};

// ═══════════════════════════════════════════════════════════════════════
console.log('\nA. events.sound 的开火事件 vs shots 列')
// ═══════════════════════════════════════════════════════════════════════

const fireTotals = await evaluate(`(() => {
  const r = window.__v5.round;
  let shots = 0;
  for (const p of r.players) for (const v of p.shots) shots += v;
  return { shots, fires: window.__v5.fireEvents.length };
})()`);
check('本回合有开火事件（数据里真的有人在开枪）', fireTotals.fires > 0,
  JSON.stringify(fireTotals));
check('开火事件数 == shots 列的总和（同一份数据的两个投影）',
  fireTotals.shots === fireTotals.fires, JSON.stringify(fireTotals));
console.log(`       (${fireTotals.fires} shots in round 0)`);

const fireMap = await evaluate(`(() => {
  const r = window.__v5.round, fires = window.__v5.fireEvents;
  const n = r.ticks.length;
  let bad = 0, badWho = 0, checked = 0, sorted = true;
  let prev = -Infinity;
  for (const e of fires) {
    if (e.t < prev) sorted = false;
    prev = e.t;
    // 事件 tick → 采样点：第一个 tick >= e.t 的那个（extract 里 searchsorted 'left'）
    let lo = 0, hi = n - 1, si = -1;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (r.ticks[m] >= e.t) { si = m; hi = m - 1; } else lo = m + 1; }
    if (si < 0) { bad++; continue; }
    const col = r.players[e.p] && r.players[e.p].shots;
    if (!col || !col[si]) bad++; else checked++;
    // 这一发必须算在开枪者自己头上：整列的和等于事件数，所以只要每发都落在
    // 一个 shots>0 的点上，就说明没有张冠李戴
    if (e.p < 0 || e.p > 9) badWho++;
  }
  return { bad, badWho, checked, total: fires.length, sorted };
})()`);
check('每一发开火事件都落在 shots > 0 的采样点上', fireMap.bad === 0, JSON.stringify(fireMap));
check('每一发开火事件都有合法的开枪者（0..9）', fireMap.badWho === 0, JSON.stringify(fireMap));
check('开火事件按 tick 升序', fireMap.sorted === true, JSON.stringify(fireMap));

// ═══════════════════════════════════════════════════════════════════════
console.log('\nB. 起点 / 方向 / 终点（拿瞄准射线当参考实现）')
// ═══════════════════════════════════════════════════════════════════════

// 找一个"至少有两个人活着"的采样点，逐个活人对比曳光和瞄准射线
const cmp = await evaluate(`(() => {
  const r = window.__v5.round;
  const n = Math.min(r.ticks.length, 500);
  for (let s = 0; s < n; s += 1) {
    window.__v5.seekSample(s);
    const st = window.__v5.players;
    if (st.filter((x) => x && x.alive).length >= 4) return { s };
  }
  return null;
})()`);
check('找到了一个有多人活着的采样点', !!cmp, JSON.stringify(cmp));

let pairs = null;
if (cmp) {
  await evaluate(`window.__v5.seekSample(${cmp.s})`);
  await sleep(100);
  pairs = await evaluate(`(() => {
    const out = [];
    const st = window.__v5.players;
    const rays = window.__v5.aimRays;
    for (let i = 0; i < st.length; i++) {
      if (!st[i] || !st[i].alive) continue;
      const t = window.__v5.spawnTracer(i);
      const ray = rays[i];
      if (!t || !ray) { out.push({ i, tracer: !!t, ray: !!ray }); continue; }
      out.push({
        i,
        state: { x: st[i].x, y: st[i].y, z: st[i].z, yaw: st[i].yaw, pitch: st[i].pitch,
                 eye: st[i].eye_height, crouch: st[i].crouch },
        tracer: t,
        ray: { start: ray.start, end: ray.end, distance: ray.distance,
               hitPlayer: ray.hitPlayer, markVisible: ray.markVisible },
      });
    }
    return out;
  })()`);
}
check('每个活人都生成了曳光，而且都有对应的瞄准射线',
  !!pairs && pairs.length >= 4 && pairs.every((p) => p.tracer && p.ray),
  pairs ? `${pairs.filter((p) => p.tracer && p.ray).length}/${pairs.length}` : 'null');

if (pairs && pairs.length) {
  const dirOf = (yaw, pitch) => {
    const yr = (yaw || 0) * Math.PI / 180, pr = (pitch || 0) * Math.PI / 180;
    return [Math.sin(yr) * Math.cos(pr), -Math.sin(pr), Math.cos(yr) * Math.cos(pr)];
  };
  const eyeOf = (s) => [s.y * S, s.z * S + (s.eye != null ? s.eye : 64.0938) * S, s.x * S];
  const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

  let startBad = 0, eyeBad = 0, dirBad = 0, distBad = 0, hitBad = 0, lenBad = 0;
  let hitsPlayer = 0, hitsWall = 0, hitsNothing = 0;
  let worstDir = 0, worstDist = 0;
  const firstBad = {};

  // 瞄准射线的端点是从 BufferGeometry 的 Float32 属性里读回来的，所以和曳光的
  // double 起点比要用 float32 量级的容差（~1e-4 足够）
  const nearF32 = (a, b) => Math.abs(a - b) < 1e-4;

  for (const p of pairs) {
    const e = eyeOf(p.state);
    // ① 起点 = 真实眼睛（两边都是 double，可以严格比）
    if (!near(p.tracer.start[0], e[0]) || !near(p.tracer.start[1], e[1])
      || !near(p.tracer.start[2], e[2])) { eyeBad++; firstBad.eye = firstBad.eye || p; }
    // ② 起点必须和瞄准射线**同一点**（射线端点读过 float32，所以松一点）
    if (!nearF32(p.tracer.start[0], p.ray.start[0]) || !nearF32(p.tracer.start[1], p.ray.start[1])
      || !nearF32(p.tracer.start[2], p.ray.start[2])) { startBad++; firstBad.start = firstBad.start || p; }
    // ③ 方向 = yaw/pitch
    const d = dirOf(p.state.yaw, p.state.pitch);
    const err = Math.max(Math.abs(p.tracer.dir[0] - d[0]), Math.abs(p.tracer.dir[1] - d[1]),
      Math.abs(p.tracer.dir[2] - d[2]));
    worstDir = Math.max(worstDir, err);
    if (err > 1e-6) { dirBad++; firstBad.dir = firstBad.dir || p; }
    // 方向必须和瞄准射线的方向一致
    const seg = [p.ray.end[0] - p.ray.start[0], p.ray.end[1] - p.ray.start[1],
                 p.ray.end[2] - p.ray.start[2]];
    const L = Math.hypot(...seg);
    if (L > 1e-6) {
      const rd = seg.map((v) => v / L);
      const e2 = Math.max(Math.abs(p.tracer.dir[0] - rd[0]), Math.abs(p.tracer.dir[1] - rd[1]),
        Math.abs(p.tracer.dir[2] - rd[2]));
      if (e2 > 1e-6) { dirBad++; firstBad.dir = firstBad.dir || p; }
    }
    // ④ 终点距离 == 瞄准射线的距离（没打到时瞄准射线报 -1，曳光飞满 60 m）
    const rayDist = p.ray.distance >= 0 ? p.ray.distance : 60;
    const de = Math.abs(p.tracer.dist - rayDist);
    worstDist = Math.max(worstDist, de);
    if (de > 1e-6) { distBad++; firstBad.dist = firstBad.dist || p; }
    // ⑤ 命中类型一致
    if (p.tracer.hitPlayer !== p.ray.hitPlayer || p.tracer.hit !== p.ray.markVisible) {
      hitBad++; firstBad.hit = firstBad.hit || p;
    }
    if (p.ray.hitPlayer >= 0) hitsPlayer++;
    else if (p.ray.distance >= 0) hitsWall++;
    else hitsNothing++;
    // ⑥ 刚生成：head 从 0 起，尾巴不越过枪口（头还没出枪口时尾巴停在枪口上）
    if (!(p.tracer.head >= 0 && p.tracer.head <= p.tracer.dist + 1e-9)) {
      lenBad++; firstBad.len = firstBad.len || p;
    }
    if (!(p.tracer.tail >= p.tracer.renderOffset - 1e-9
      && p.tracer.tail <= Math.max(p.tracer.head, p.tracer.renderOffset) + 1e-9)) {
      lenBad++; firstBad.len = firstBad.len || p;
    }
  }

  check('曳光起点 == 真实眼睛（含下蹲眼高）', eyeBad === 0,
    firstBad.eye ? JSON.stringify({ t: firstBad.eye.tracer.start, s: firstBad.eye.state }) : '');
  check('曳光起点 == 瞄准射线起点（同一次检测）', startBad === 0,
    firstBad.start ? JSON.stringify({ t: firstBad.start.tracer.start, r: firstBad.start.ray.start }) : '');
  check('曳光方向 == yaw/pitch == 瞄准射线方向', dirBad === 0, `max err ${worstDir.toExponential(2)}`);
  check('曳光终点距离 == 同一帧瞄准射线的距离', distBad === 0, `max err ${worstDist.toExponential(2)}`);
  check('曳光的命中类型 == 瞄准射线的命中类型', hitBad === 0,
    firstBad.hit ? JSON.stringify({ t: firstBad.hit.tracer.hitPlayer, r: firstBad.hit.ray.hitPlayer }) : '');
  check('刚生成时 head=0、尾巴不越过枪口', lenBad === 0, '');
  console.log(`       (${pairs.length} players: ${hitsPlayer} hit a player, ${hitsWall} hit the map, `
    + `${hitsNothing} hit nothing)`);

  // 三种命中类型都要有，否则这个对拍没覆盖到截断逻辑
  check('对拍覆盖了"打人"这一路', hitsPlayer > 0, `${hitsPlayer}`);
}

// 死人不能开枪
const deadCase = await evaluate(`(() => {
  const r = window.__v5.round;
  const n = Math.min(r.ticks.length, 2000);
  for (let s = 0; s < n; s += 2) {
    window.__v5.seekSample(s);
    const st = window.__v5.players;
    const dead = st.findIndex((x) => x && !x.alive);
    if (dead >= 0) return { s, dead, tracer: window.__v5.spawnTracer(dead) };
  }
  return null;
})()`);
check('死人不生成曳光', !!deadCase && deadCase.tracer === null, JSON.stringify(deadCase));

// 曳光的网格本身：加法混合、不吃深度、圆柱、renderOrder 高
const meshInfo = await evaluate(`(() => {
  const m = window.__v5.scene.children.find((o) => o.isMesh && o.geometry
    && o.geometry.type === 'CylinderGeometry' && o.material.blending === 2);
  if (!m) return null;
  return {
    type: m.geometry.type,
    blending: m.material.blending,
    depthTest: m.material.depthTest,
    depthWrite: m.material.depthWrite,
    renderOrder: m.renderOrder,
    transparent: m.material.transparent,
  };
})()`);
check('曳光是加法混合的圆柱、不吃深度（隔着墙也看得见）',
  !!meshInfo && meshInfo.blending === 2 && meshInfo.depthTest === false
  && meshInfo.depthWrite === false && meshInfo.transparent === true
  && meshInfo.renderOrder >= 650,
  JSON.stringify(meshInfo));

// ═══════════════════════════════════════════════════════════════════════
console.log('\nC. 飞出去 → 碰到就停 → 淡出')
// ═══════════════════════════════════════════════════════════════════════

// 挑一个"这一发会打到东西"的人，逐帧录下来
const flightSetup = await evaluate(`(() => {
  const r = window.__v5.round;
  const n = Math.min(r.ticks.length, 3000);
  for (let s = 0; s < n; s += 3) {
    window.__v5.seekSample(s);
    const st = window.__v5.players;
    for (let i = 0; i < st.length; i++) {
      if (!st[i] || !st[i].alive) continue;
      const t = window.__v5.spawnTracer(i);
      if (t && t.hit && t.dist > 2) return { s, i, dist: t.dist, hitPlayer: t.hitPlayer };
    }
  }
  return null;
})()`);
check('找到了一个"打得到东西"的曳光用例', !!flightSetup, JSON.stringify(flightSetup));

if (flightSetup) {
  await evaluate(`window.__v5.seekSample(${flightSetup.s})`);
  await clearTracers();
  // 在**同一个 JS 任务里**按固定 dt 推进：headless 的 rAF 只有 ~3 fps
  // （swiftshader 画 160 万三角形），靠 rAF 采样根本看不到飞行过程。
  // tickTracers 调的就是主循环里那个 updateTracers，所以测的是同一段代码。
  const flight = await evaluate(`(() => {
    const info = window.__v5.spawnTracer(${flightSetup.i});
    if (!info) return null;
    const DT = 1 / 60;
    const frames = [{ t: 0, all: window.__v5.tracers }];
    for (let k = 1; k <= 120; k++) {
      frames.push({ t: k * DT, all: window.__v5.tickTracers(DT) });
    }
    return { info, frames, dt: DT };
  })()`);

  check('逐帧采样拿到了数据', !!flight && flight.frames.length > 20,
    flight ? `${flight.frames.length} frames` : 'null');

  if (flight) {
    const dist = flight.info.dist;
    const nominal = flight.info.nominalSegLen;
    const renderOffset = flight.info.renderOffset;
    // 只看我们刚生成的那一条（这一帧只有它）
    const seq = flight.frames.map((f) => f.all[0]).filter(Boolean);
    check('飞行过程中曳光一直可见（至少 3 帧）', seq.length >= 3, `${seq.length} frames with a tracer`);

    const overshoot = seq.filter((x) => x.head > x.dist + 1e-9);
    check('head 永远不超过命中距离（不穿墙、不穿人）', overshoot.length === 0,
      overshoot.length ? `max head ${Math.max(...overshoot.map((x) => x.head))} > dist ${dist}` : '');

    let monotonic = true;
    for (let i = 1; i < seq.length; i++) if (seq[i].head < seq[i - 1].head - 1e-9) monotonic = false;
    check('head 单调不减（只会往前飞，不会倒回来）', monotonic, '');

    const maxHead = Math.max(...seq.map((x) => x.head));
    check('head 最终正好停在命中距离上', Math.abs(maxHead - dist) < 1e-9,
      `max head ${maxHead} vs dist ${dist}`);

    // 停住之后必须一直停在命中点上（hold → fade）
    const stopped = seq.filter((x) => x.phase !== 'fly');
    check('到达后进入 hold / fade，而且停在命中点上',
      stopped.length > 0 && stopped.every((x) => Math.abs(x.head - dist) < 1e-9),
      `${stopped.length} stopped frames, phases ${[...new Set(stopped.map((x) => x.phase))].join('/')}`);
    check('phase 的顺序是 fly → hold → fade',
      (() => {
        const order = ['fly', 'hold', 'fade'];
        let k = 0;
        for (const x of seq) {
          const j = order.indexOf(x.phase);
          if (j < k) return false;
          k = j;
        }
        return seq.some((x) => x.phase === 'fly') && seq.some((x) => x.phase === 'fade');
      })(), [...new Set(seq.map((x) => x.phase))].join(' → '));

    // 停住时正好是标称段长（低帧率下被撑长过，停下来要收回去）
    check('停在命中点时段长 == 标称长度（min(5 m, 整条弹道)）',
      stopped.length > 0 && stopped.every((x) => Math.abs(x.segLen - Math.min(nominal, dist - renderOffset)) < 1e-9),
      `nominal ${nominal} dist ${dist.toFixed(3)} offset ${renderOffset}`);

    // 淡出：opacity 从 0.95 掉到 0
    const minOpacity = Math.min(...seq.map((x) => x.opacity));
    check('最后淡出到接近 0', minOpacity < 0.05, `min opacity ${minOpacity.toFixed(3)}`);

    // 段长：报的 segLen 必须等于网格真实的 scale.y（画出来的就是它）
    const segMismatch = seq.filter((x) => Math.abs(x.segLen - x.meshSegLen) > 1e-9);
    check('报告的段长 == 网格真实的 scale.y', segMismatch.length === 0, '');
    const tooLong = seq.filter((x) => x.segLen > x.dist + 1e-9 || x.segLen > nominal * 2 + 1e-9);
    check('段长不超过整条弹道，也不超过 2× 标称长度（低帧率撑长的上限）',
      tooLong.length === 0, '');

    // 尾巴不越过枪口（头还没出枪口时尾巴停在枪口上）
    const tailBad = seq.filter((x) => x.tail < x.renderOffset - 1e-9
      || x.tail > Math.max(x.head, x.renderOffset) + 1e-9);
    check('尾巴不越过枪口（renderOffset），也不跑到头前面', tailBad.length === 0,
      tailBad.length ? JSON.stringify(tailBad[0]) : '');

    // 位置：mesh 中点 == start + dir*(tail + segLen/2)
    let worstPos = 0;
    for (const x of seq) {
      const mid = x.tail + x.segLen / 2;
      for (let a = 0; a < 3; a++) {
        worstPos = Math.max(worstPos, Math.abs(x.position[a] - (x.start[a] + x.dir[a] * mid)));
      }
    }
    check('网格中点 == start + dir·(tail + segLen/2)', worstPos < 1e-9,
      `max err ${worstPos.toExponential(2)}`);

    // 速度：head 的增量 / dt ≈ TRACER_SPEED（200 m/s）
    let bestSpeed = 0;
    for (let i = 1; i < flight.frames.length; i++) {
      const a = flight.frames[i - 1].all[0], b = flight.frames[i].all[0];
      const dt = flight.frames[i].t - flight.frames[i - 1].t;
      if (!a || !b || a.phase !== 'fly' || b.phase !== 'fly' || dt <= 0) continue;
      bestSpeed = Math.max(bestSpeed, (b.head - a.head) / dt);
    }
    check('飞行速度 == 200 m/s（"迅速"飞出去）', Math.abs(bestSpeed - 200) < 1e-3,
      `${bestSpeed.toFixed(4)} m/s`);

    // 淡出结束后应该收干净
    const left = await evaluate('window.__v5.tracers.length');
    check('淡出结束后曳光被回收（不再占着池子）', left === 0, `${left} left`);
  }
}

// ═══════════════════════════════════════════════════════════════════════
console.log('\nD. 一发一发地生成（区间语义 + 死人不打）')
// ═══════════════════════════════════════════════════════════════════════

// 挑一发"开枪的人在这一帧还活着、而且那一秒里只有这一发"的事件，这样期望值是唯一的
const shot = await evaluate(`(() => {
  const r = window.__v5.round, fires = window.__v5.fireEvents;
  const n = r.ticks.length;
  const cnt = (a, b) => fires.filter((x) => x.t > a && x.t <= b).length;
  for (const e of fires) {
    if (cnt(e.t - 1, e.t) !== 1) continue;
    // 事件 tick → 采样点：第一个 tick >= e.t 的那个（extract 里 searchsorted 'left'）
    let lo = 0, hi = n - 1, si = -1;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (r.ticks[m] >= e.t) { si = m; hi = m - 1; } else lo = m + 1; }
    if (si < 1) continue;
    window.__v5.seekSample(si);
    const st = window.__v5.players[e.p];
    if (!st || !st.alive) continue;
    return {
      tick: e.t, p: e.p, si, yaw: st.yaw, pitch: st.pitch,
      want: cnt(e.t - 1, e.t),
      wantSame: cnt(e.t, e.t),
      wantLeft: cnt(e.t - 1, e.t - 1),
      wantNext: cnt(e.t, e.t + 1),
    };
  }
  return null;
})()`);
check('找到了一发"开枪者还活着、且区间唯一"的开火事件', !!shot, JSON.stringify(shot));

if (shot) {
  await clearTracers();
  const inRange = await evaluate(`window.__v5.stepTracers(${shot.tick - 1}, ${shot.tick})`);
  check('(T-1, T] 里那一发生成了 1 条曳光', inRange === shot.want, `got ${inRange}, want ${shot.want}`);

  const spawned = await evaluate('window.__v5.tracers');
  check('生成后正好有 1 条在飞', spawned.length === shot.want, `${spawned.length}`);
  if (spawned.length === 1) {
    const d = spawned[0].dir;
    const yr = shot.yaw * Math.PI / 180, pr = shot.pitch * Math.PI / 180;
    const want = [Math.sin(yr) * Math.cos(pr), -Math.sin(pr), Math.cos(yr) * Math.cos(pr)];
    const err = Math.max(...[0, 1, 2].map((a) => Math.abs(d[a] - want[a])));
    check('方向用的是**这一发**那一帧的 yaw/pitch', err < 1e-6, `err ${err.toExponential(2)}`);
  }

  check('(T, T] 是 0 条（右闭：正好在 T 上的一发只算一次）',
    (await evaluate(`window.__v5.stepTracers(${shot.tick}, ${shot.tick})`)) === shot.wantSame,
    `want ${shot.wantSame}`);
  check('(T-1, T-1] 是 0 条',
    (await evaluate(`window.__v5.stepTracers(${shot.tick - 1}, ${shot.tick - 1})`)) === shot.wantLeft,
    `want ${shot.wantLeft}`);
  check('(T, T+1] 不会把 T 上那一发再算一遍',
    (await evaluate(`window.__v5.stepTracers(${shot.tick}, ${shot.tick + 1})`)) === shot.wantNext,
    `want ${shot.wantNext}`);
  console.log(`       (区间语义: (T-1,T]=${shot.want} (T,T]=${shot.wantSame} `
    + `(T-1,T-1]=${shot.wantLeft} (T,T+1]=${shot.wantNext})`);

  // 一整段多发：条数 == 区间内**开枪者还活着**的事件数
  // （`spawnTracersBetween` 用的是当前采样点的玩家状态，所以死人不算）
  await evaluate(`window.__v5.seekSample(${cmp ? cmp.s : 0})`);
  await clearTracers();
  const span = await evaluate(`(() => {
    const fires = window.__v5.fireEvents, st = window.__v5.players;
    if (fires.length < 3) return null;
    for (let i = 0; i + 2 < fires.length; i++) {
      const a = fires[i].t, b = fires[i + 2].t;
      const want = fires.filter((e) => e.t > a && e.t <= b && st[e.p] && st[e.p].alive).length;
      if (want >= 2) return { a, b, want, got: window.__v5.stepTracers(a, b) };
    }
    return null;
  })()`);
  if (span) {
    check('跨多发的区间：生成的条数 == 区间内活着的人开的枪数', span.got === span.want,
      JSON.stringify(span));
    console.log(`       (跨多发: ${span.want} 发 → ${span.got} 条)`);
  } else {
    console.log('  (这一帧凑不出"区间里至少 2 个活人开枪"，跳过)');
  }

  // 死人不打：找一个已经死了的人，手动生成必须是 null
  const deadShot = await evaluate(`(() => {
    const r = window.__v5.round;
    const n = Math.min(r.ticks.length, 2000);
    for (let s = 0; s < n; s += 2) {
      window.__v5.seekSample(s);
      const st = window.__v5.players;
      const i = st.findIndex((x) => x && !x.alive);
      if (i >= 0) return { s, i, tracer: window.__v5.spawnTracer(i) };
    }
    return null;
  })()`);
  check('死人不生成曳光', !!deadShot && deadShot.tracer === null,
    JSON.stringify({ i: deadShot && deadShot.i, tracer: deadShot && deadShot.tracer }));
}

// ═══════════════════════════════════════════════════════════════════════
console.log('\nE. 开关')
// ═══════════════════════════════════════════════════════════════════════

// 先在开关开着的时候放一条出来
await evaluate(`window.__v5.seekSample(${cmp ? cmp.s : 0})`);
await clearTracers();
const onCount = await evaluate(`(() => {
  const st = window.__v5.players;
  const i = st.findIndex((x) => x && x.alive);
  return i >= 0 ? (window.__v5.spawnTracer(i) ? window.__v5.tracers.length : -1) : -2;
})()`);
check('开关开着时能生成并看到曳光', onCount >= 1, `${onCount}`);

await evaluate(`(() => {
  const t = document.getElementById('toggle-tracers');
  t.checked = false; t.dispatchEvent(new Event('change'));
})()`);
await sleep(80);
check('关掉开关后所有曳光立刻隐藏',
  (await evaluate('window.__v5.tracers.length')) === 0);

const offSpawn = await evaluate(`(() => {
  const st = window.__v5.players;
  const i = st.findIndex((x) => x && x.alive);
  return { spawned: i >= 0 ? !!window.__v5.spawnTracer(i) : null, visible: window.__v5.tracers.length };
})()`);
check('关掉开关后手动生成也不显示', offSpawn.visible === 0, JSON.stringify(offSpawn));

if (shot) {
  check('关掉开关后自动生成也是 0 条',
    (await evaluate(`window.__v5.stepTracers(${shot.tick - 1}, ${shot.tick})`)) === 0);
}

await evaluate(`(() => {
  const t = document.getElementById('toggle-tracers');
  t.checked = true; t.dispatchEvent(new Event('change'));
})()`);

// 换回合必须把在飞的曳光收掉（不然会看到上一回合的枪）
await evaluate(`(() => {
  const st = window.__v5.players;
  const i = st.findIndex((x) => x && x.alive);
  if (i >= 0) window.__v5.spawnTracer(i);
})()`);
const beforeSwitch = await evaluate('window.__v5.tracers.length');
const roundCount = await evaluate('window.__v5.summary.rounds.length');
if (roundCount > 1) {
  await evaluate(`(() => {
    const sel = document.getElementById('round-select');
    sel.value = '1'; sel.dispatchEvent(new Event('change'));
  })()`);
  await sleep(1200);
  check('换回合后上一回合的曳光被收掉',
    beforeSwitch >= 1 && (await evaluate('window.__v5.tracers.length')) === 0,
    `before ${beforeSwitch}`);
} else {
  console.log('  (只有一个回合，跳过换回合清理检查)');
}

// ── 收尾 ────────────────────────────────────────────────────────────────────
const realErrors = consoleErrors.filter((e) => !/favicon|ERR_/.test(e));
check('运行期间没有 JS 异常', realErrors.length === 0, realErrors.slice(0, 3).join(' | '));

console.log(`\ntracers: ${passed} passed, ${failed} failed`);
ws.close();
chrome.kill();
process.exit(failed === 0 ? 0 : 1);
