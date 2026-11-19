/**
 * test_aim_ray.mjs — 瞄准射线的渲染测试（headless Chrome + CDP）。
 *
 * 分两块：
 *
 * **A. BVH 对拍**（最重要的一条）
 *   在**真实地图**（de_ancient，823,214 个三角形）上，用 three.js 自己的朴素
 *   `Raycaster` 当参考实现，逐条比对 `map-raycast.js` 的 BVH 结果。
 *   朴素 raycast 45 ms/条，所以只跑十几条 —— 但足以覆盖"BVH 遍历剪枝写错了"
 *   这类 bug（遍历 bug 几乎必然在随机方向上暴露）。
 *   注意要把地图材质的 side 临时改成 DoubleSide：three 默认 FrontSide 不命中背面，
 *   而瞄准射线两边都要挡。
 *
 * **B. 画出来的射线**
 *   直接读场景图里 `THREE.Line` 的两个端点，验证：
 *     - 起点 = 真实眼睛（含下蹲眼高）
 *     - 方向 = 瞄准方向
 *     - 长度 = min(地图命中, 玩家命中, maxDist)，即**在最近的遮挡处截断**
 *     - 打人时终点必须落在那个人的**碰撞盒表面**上（独立于距离是怎么算的）
 *     - 打墙时终点必须落在 BVH 给出的命中点上
 *     - 颜色：人=红 / 墙=琥珀 / 没打到=暗青
 *     - 关掉开关后全部隐藏
 *
 * 用法：
 *   python visualizer/server.py --port 5099 &
 *   node visualizer/tests/test_aim_ray.mjs http://127.0.0.1:5099 /path/to/match.csz
 */

import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const URL = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3];
if (!CSZ) {
  console.error('用法: node test_aim_ray.mjs <server-url> <match.csz>');
  process.exit(2);
}
const CHROME = process.env.CHROME_PATH
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
// 用随机端口：写死端口时连着跑两次，上一轮的 Chrome 还没释放端口就会
// "连不上 CDP"。CDP_PORT 仍然可以覆盖。
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}

// ── 起 Chrome + 接 CDP ──────────────────────────────────────────────────────
const profile = mkdtempSync(join(tmpdir(), 'aim-'));
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
    if (m) m.traverse((o) => { if (o.isMesh && o.geometry) t += o.geometry.attributes.position.count / 3; });
    return Math.round(t);
  })()`);
  if (mapTris > 0) break;
  await sleep(500);
}
check('the map geometry is loaded', mapTris > 100000, `${mapTris} triangles`);

// ── A. BVH vs three.js 朴素 raycast（真实地图上对拍）────────────────────────
console.log('\nA. BVH vs three.js naive raycast on the real map')

// 打开瞄准射线 → main.js 会 ensureMapBVH()
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
console.log(`       (BVH: ${bvhStats.triCount} triangles in ${Math.round(bvhStats.ms)} ms, `
  + `${bvhStats.baked ? 'baked' : 'reused the OBJ vertex array'})`);

const cross = await evaluate(`(async () => {
  const THREE = await import('/static/three/build/three.module.js');
  const map = window.__v5.scene.getObjectByName('map-root');
  const S = 0.0254;

  // three 的 Mesh.raycast 只看 material.side；地图默认 FrontSide 不命中背面，
  // 而瞄准射线两面都要挡 —— 临时改成 DoubleSide 才是公平对比。
  const sides = [];
  map.traverse((o) => { if (o.isMesh && o.material) { sides.push([o.material, o.material.side]);
    o.material.side = THREE.DoubleSide; } });

  const rc = new THREE.Raycaster();
  const out = [];
  // 用真实玩家的眼睛 + 瞄准方向当射线（比纯随机更有代表性）
  const states = window.__v5.players;
  const seeds = [];
  for (let i = 0; i < states.length; i++) {
    const st = states[i];
    if (!st || !st.alive) continue;
    seeds.push([st.y * S, st.z * S + (st.eye_height != null ? st.eye_height : 64.0938) * S, st.x * S,
                st.yaw || 0, st.pitch || 0]);
  }
  // 再补几条朝下/朝上的，保证一定打到地板/天花板
  const base = seeds[0] || [0, 0, 0, 0, 0];
  for (const p of [-70, -40, 0, 30, 70]) {
    seeds.push([base[0], base[1], base[2], base[3], p]);
  }
  seeds.push([base[0], base[1], base[2], (base[3] + 137) % 360, -10]);

  const raw = window.__v5.rawRaycast;
  for (const [ox, oy, oz, yaw, pitch] of seeds) {
    const yr = yaw * Math.PI / 180, pr = pitch * Math.PI / 180;
    const d = new THREE.Vector3(0, 0, 1);
    d.applyAxisAngle(new THREE.Vector3(1, 0, 0), pr);
    d.applyAxisAngle(new THREE.Vector3(0, 1, 0), yr);
    d.normalize();
    const t = raw(ox, oy, oz, d.x, d.y, d.z, 60);
    rc.set(new THREE.Vector3(ox, oy, oz), d);
    rc.far = 60;
    const hits = rc.intersectObjects(map.children, true);
    out.push({ t, naive: hits.length ? hits[0].distance : -1 });
  }
  for (const [mat, s] of sides) mat.side = s;
  return out;
})()`, true);

let agree = 0, disagree = 0, bothHit = 0, firstBad = null;
for (const c of cross) {
  const a = c.t, b = c.naive;
  const same = (a < 0 && b < 0) || (a >= 0 && b >= 0 && Math.abs(a - b) < 1e-4);
  if (same) agree++; else { disagree++; if (!firstBad) firstBad = c; }
  if (a >= 0 && b >= 0) bothHit++;
}
check(`all ${cross.length} rays agree with three.js's own raycaster`,
  disagree === 0, firstBad ? `first disagreement: ${JSON.stringify(firstBad)}` : '');
check('the cross-check actually hit geometry', bothHit >= cross.length * 0.7,
  `${bothHit}/${cross.length}`);
console.log(`       (${agree} agree, ${disagree} disagree; ${bothHit} hit something)`);

// ── B. 画出来的射线 ─────────────────────────────────────────────────────────
console.log('\nB. the rendered aim rays')

// 找一个"有人打到人"的采样点，专门验证玩家截断
const found = await evaluate(`(() => {
  const r = window.__v5.round;
  for (let s = 0; s < r.ticks.length; s += 1) {
    window.__v5.seekSample(s);
    const rays = window.__v5.aimRays;
    const hit = rays.findIndex((x) => x && x.visible && x.hitPlayer >= 0);
    if (hit >= 0) return { s, hit };
  }
  return null;
})()`);
check('found a sample where someone is aiming at someone else', !!found,
  JSON.stringify(found));

if (found) {
  await evaluate(`window.__v5.seekSample(${found.s})`);
  await sleep(200);
  const rays = await evaluate('window.__v5.aimRays');
  const states = await evaluate('window.__v5.players');
  const r = rays[found.hit];
  const shooter = states[found.hit];
  const target = states[r.hitPlayer];
  console.log(`       player ${found.hit} hits player ${r.hitPlayer} `
    + `at ${r.distance.toFixed(2)} m (sample ${found.s})`);

  // 起点 = 真实眼睛
  const S = 0.0254;
  const eyeY = shooter.z * S + (shooter.eye_height != null ? shooter.eye_height : 64.0938) * S;
  check('the ray starts at the real eye (crouch-aware eye height)',
    Math.abs(r.start[0] - shooter.y * S) < 1e-6
    && Math.abs(r.start[1] - eyeY) < 1e-6
    && Math.abs(r.start[2] - shooter.x * S) < 1e-6,
    `start=${JSON.stringify(r.start.map((v) => v.toFixed(4)))} eyeY=${eyeY.toFixed(4)}`);

  // 方向 = 瞄准方向
  const yr = (shooter.yaw || 0) * Math.PI / 180;
  const pr = (shooter.pitch || 0) * Math.PI / 180;
  const dir = [
    Math.sin(yr) * Math.cos(pr),
    -Math.sin(pr),
    Math.cos(yr) * Math.cos(pr),
  ];
  const seg = [r.end[0] - r.start[0], r.end[1] - r.start[1], r.end[2] - r.start[2]];
  const segLen = Math.hypot(...seg);
  const dot = (seg[0] * dir[0] + seg[1] * dir[1] + seg[2] * dir[2]) / segLen;
  check('the ray points along the aim direction', Math.abs(dot - 1) < 1e-5, `dot=${dot}`);

  // 容差 1 mm：BufferGeometry.setFromPoints 把顶点存成 **Float32Array**，
  // 坐标上百米时 float32 的精度就是 ~1e-5 m，1e-6 会假失败。
  check('the ray length equals the reported hit distance',
    Math.abs(segLen - r.distance) < 1e-3, `${segLen} vs ${r.distance}`);
  check('the ray is red when it hits a player', r.color === 0xff4d4d,
    `#${r.color.toString(16)}`);
  check('an impact marker is drawn', r.markVisible
    && Math.hypot(r.markPos[0] - r.end[0], r.markPos[1] - r.end[1],
                  r.markPos[2] - r.end[2]) < 1e-6);

  // 终点必须落在**目标碰撞盒的表面**上（独立验证，不依赖距离是怎么算的）
  const hw = 16 * S;
  const th = (target.crouching ? 54 : 72) * S;
  const lo = [target.y * S - hw, target.z * S, target.x * S - hw];
  const hi = [target.y * S + hw, target.z * S + th, target.x * S + hw];
  const inside = r.end.every((v, k) => v >= lo[k] - 1e-6 && v <= hi[k] + 1e-6);
  // 到最近一个面的距离（在盒内时是"离表面多远"）
  const distToSurface = Math.min(...[0, 1, 2].map((k) => Math.min(
    Math.abs(r.end[k] - lo[k]), Math.abs(r.end[k] - hi[k]))));
  check('the hit point is inside the target collision hull', inside,
    `end=${JSON.stringify(r.end.map((v) => v.toFixed(3)))} hull=${JSON.stringify([lo, hi])}`);
  check('the hit point lies on the hull surface', distToSurface < 2e-3,
    `${distToSurface} m from the nearest face`);

  // 射线必须在**进入**碰撞盒的那一面停下：命中点往后退一点必须在盒外，
  // 往前穿过整个盒子之后也必须在盒外。
  // （不能拿"往前 0.5 m 就在盒外"来测 —— 盒子进深 32 HU = 0.81 m，
  //   0.5 m 之后当然还在盒子里；那是测试写错了，不是功能错了。）
  const back = [
    r.start[0] + dir[0] * (r.distance - 0.02),
    r.start[1] + dir[1] * (r.distance - 0.02),
    r.start[2] + dir[2] * (r.distance - 0.02),
  ];
  check('just before the hit point is outside the hull (stopped at the near face)',
    !back.every((v, k) => v >= lo[k] && v <= hi[k]),
    JSON.stringify(back.map((v) => v.toFixed(3))));

  // 盒子的体对角线 + 余量，保证一定穿出去了
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const beyond = [
    r.start[0] + dir[0] * (r.distance + diag + 0.3),
    r.start[1] + dir[1] * (r.distance + diag + 0.3),
    r.start[2] + dir[2] * (r.distance + diag + 0.3),
  ];
  const beyondInside = beyond.every((v, k) => v >= lo[k] && v <= hi[k]);
  check('the ray does not continue through to the far side of the player', !beyondInside,
    JSON.stringify(beyond.map((v) => v.toFixed(3))));
}

// 打墙的射线：终点必须正好在 BVH 的命中点上
{
  await evaluate(`window.__v5.seekSample(0)`);
  await sleep(200);
  const rays = await evaluate('window.__v5.aimRays');
  const states = await evaluate('window.__v5.players');
  const wallIdx = rays.findIndex((x) => x && x.visible && x.hitPlayer < 0 && x.distance >= 0);
  check('found a ray that hits a wall', wallIdx >= 0, `${wallIdx}`);
  if (wallIdx >= 0) {
    const r = rays[wallIdx];
    const st = states[wallIdx];
    const S = 0.0254;
    const seg = [r.end[0] - r.start[0], r.end[1] - r.start[1], r.end[2] - r.start[2]];
    const L = Math.hypot(...seg);
    const d = seg.map((v) => v / L);
    const t = await evaluate(`window.__v5.rawRaycast(${r.start[0]}, ${r.start[1]}, ${r.start[2]}, `
      + `${d[0]}, ${d[1]}, ${d[2]}, 60)`);
    check('the wall hit point matches an independent BVH query',
      t >= 0 && Math.abs(t - r.distance) < 1e-5, `${t} vs ${r.distance}`);
    check('the ray is amber when it hits a wall', r.color === 0xffb020,
      `#${r.color.toString(16)}`);
    check('the wall ray starts at the shooter eye',
      Math.abs(r.start[1] - (st.z * S + (st.eye_height != null ? st.eye_height : 64.0938) * S)) < 1e-6);
  }
}

// 什么都没打到的射线：必须画满 maxDist、暗青色、没有撞击点
// （de_ancient 上大约 1% 的射线是这种 —— 朝天或者从高处往外看）
{
  const freeFound = await evaluate(`(() => {
    const r = window.__v5.round;
    for (let s = 0; s < r.ticks.length; s += 7) {
      window.__v5.seekSample(s);
      const rays = window.__v5.aimRays, st = window.__v5.players;
      for (let i = 0; i < rays.length; i++) {
        const x = rays[i];
        if (x && x.visible && st[i] && st[i].alive && x.hitPlayer < 0 && x.distance < 0) {
          return { s, i };
        }
      }
    }
    return null;
  })()`);
  check('found a ray that hits nothing at all', !!freeFound, JSON.stringify(freeFound));
  if (freeFound) {
    await evaluate(`window.__v5.seekSample(${freeFound.s})`);
    await sleep(200);
    const x = (await evaluate('window.__v5.aimRays'))[freeFound.i];
    const L = Math.hypot(x.end[0] - x.start[0], x.end[1] - x.start[1], x.end[2] - x.start[2]);
    check('a free ray is drawn to the full max distance (60 m)', Math.abs(L - 60) < 1e-3,
      `${L}`);
    check('a free ray is dim cyan', x.color === 0x66ccff, `#${x.color.toString(16)}`);
    check('a free ray has no impact marker', x.markVisible === false);
    // 独立确认：这条路线上 60 m 内确实没有地图几何
    const seg = [x.end[0] - x.start[0], x.end[1] - x.start[1], x.end[2] - x.start[2]];
    const d = seg.map((v) => v / L);
    const t = await evaluate(`window.__v5.rawRaycast(${x.start[0]}, ${x.start[1]}, ${x.start[2]}, `
      + `${d[0]}, ${d[1]}, ${d[2]}, 60)`);
    check('the BVH independently agrees there is no map hit', t < 0, `${t}`);
    console.log(`       (free ray: player ${freeFound.i} at sample ${freeFound.s})`);
  }
}

// 扫一遍：每条射线的长度都不能超过 maxDist，命中类型和颜色必须一致
{
  const sweep = await evaluate(`(() => {
    const r = window.__v5.round;
    const bad = { n: 0, tooLong: 0, color: 0, mark: 0, past: 0 };
    const S = 0.0254;
    for (let s = 0; s < r.ticks.length; s += 53) {
      window.__v5.seekSample(s);
      const rays = window.__v5.aimRays, st = window.__v5.players;
      for (let i = 0; i < rays.length; i++) {
        const x = rays[i];
        if (!x || !x.visible || !st[i] || !st[i].alive) continue;
        bad.n++;
        const seg = [x.end[0]-x.start[0], x.end[1]-x.start[1], x.end[2]-x.start[2]];
        const L = Math.hypot(...seg);
        // 容差 1 mm 是给 float32 顶点存储留的（见上面那条注释）
        if (L > 60 + 1e-3) bad.tooLong++;
        const want = x.hitPlayer >= 0 ? 0xff4d4d : (x.distance >= 0 ? 0xffb020 : 0x66ccff);
        if (x.color !== want) bad.color++;
        if (x.markVisible !== (x.hitPlayer >= 0 || x.distance >= 0)) bad.mark++;
        if (x.hitPlayer < 0) {
          const d = seg.map(v => v / L);
          const t = window.__v5.rawRaycast(
            x.start[0], x.start[1], x.start[2], d[0], d[1], d[2], 60);
          // 打墙：终点必须正好在 BVH 命中点上
          // 没打到（free）：BVH 也必须同样打不到，不能只是"画满了 60 m"
          if (x.distance < 0) { if (t >= 0) bad.past++; }
          else if (!(t >= 0 && Math.abs(t - L) < 1e-3)) bad.past++;
        }
      }
    }
    return bad;
  })()`);
  check('the sweep sampled enough rays', sweep.n > 200, JSON.stringify(sweep));
  check('no ray is longer than the max distance', sweep.tooLong === 0, JSON.stringify(sweep));
  check('every ray colour matches its hit type', sweep.color === 0, JSON.stringify(sweep));
  check('the impact marker matches the hit type', sweep.mark === 0, JSON.stringify(sweep));
  check('every wall ray ends exactly on the BVH hit', sweep.past === 0, JSON.stringify(sweep));
}

// 关掉开关
{
  await evaluate(`(() => {
    const t = document.getElementById('toggle-aim');
    t.checked = false; t.dispatchEvent(new Event('change'));
    return true;
  })()`);
  await sleep(400);
  const hidden = await evaluate(`window.__v5.aimRays.every((x) => !x || (!x.visible && !x.markVisible))`);
  check('turning the toggle off hides every ray and marker', hidden === true);
}

console.log('\nconsole');
check('no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 2).join(' | '));

// ── C. 换地图必须丢掉旧 BVH（可选：给了第二张图的 .csz 才跑）────────────────
// 用旧地图的 BVH 去检测新地图的射线，会**静默地**给出错误结果（打到不存在的墙、
// 或者穿墙），所以这条很值得守。默认不跑是因为它要求手上有两张不同地图的 .csz。
const CSZ2 = process.argv[4];
if (CSZ2) {
  console.log('\nC. BVH invalidation across a map switch');
  const trisOf = () => evaluate(`(() => {
    const m = window.__v5.scene.getObjectByName('map-root');
    let t = 0;
    if (m) m.traverse((o) => { if (o.isMesh && o.geometry) t += o.geometry.attributes.position.count / 3; });
    return Math.round(t);
  })()`);

  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ2)})`, true);
  await sleep(2000);
  let tris2 = 0;
  for (let i = 0; i < 120; i++) {
    tris2 = await trisOf();
    if (tris2 > 0 && tris2 !== mapTris) break;
    await sleep(500);
  }
  const map2 = await evaluate('window.__v5.summary.map');
  check('a different map actually loaded', tris2 !== mapTris && tris2 > 0,
    `${tris2} triangles (was ${mapTris})`);
  check('the old BVH was thrown away on the map switch',
    (await evaluate('window.__v5.mapBVHStats')) === null);
  console.log(`       (${map2}: ${tris2} triangles)`);

  // 再 toggle 一次触发重建
  await evaluate(`(() => {
    const t = document.getElementById('toggle-aim');
    t.checked = false; t.dispatchEvent(new Event('change'));
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
  for (let i = 0; i < 200; i++) {
    await sleep(300);
    if (await evaluate('!!window.__v5.mapBVHStats')) break;
  }
  const s2 = await evaluate('window.__v5.mapBVHStats');
  check('a fresh BVH is built for the new map', !!s2 && s2.triCount === tris2,
    JSON.stringify(s2));

  // 新地图上的每条射线都必须和**新地图自己的** BVH 对得上
  const verify = await evaluate(`(() => {
    const r = window.__v5.round;
    let ok = 0, n = 0;
    for (let s = 0; s < r.ticks.length; s += 97) {
      window.__v5.seekSample(s);
      const rays = window.__v5.aimRays, st = window.__v5.players;
      for (let i = 0; i < rays.length; i++) {
        const x = rays[i];
        if (!x || !x.visible || !st[i] || !st[i].alive) continue;
        n++;
        if (x.hitPlayer >= 0) { ok++; continue; }
        const seg = [x.end[0]-x.start[0], x.end[1]-x.start[1], x.end[2]-x.start[2]];
        const L = Math.hypot(...seg);
        const d = seg.map(v => v / L);
        const t = window.__v5.rawRaycast(x.start[0], x.start[1], x.start[2], d[0], d[1], d[2], 60);
        if (x.distance < 0) { if (t < 0) ok++; }
        else if (t >= 0 && Math.abs(t - L) < 1e-3) ok++;
      }
    }
    return { ok, n };
  })()`);
  check('every ray on the new map agrees with the new BVH',
    verify.n > 50 && verify.ok === verify.n, JSON.stringify(verify));
} else {
  console.log('\n(skipping the map-switch check: pass a second .csz from a different map as argv[4])');
}

console.log(`\naim ray: ${passed} passed, ${failed} failed`);
ws.close();
chrome.kill();
process.exit(failed === 0 ? 0 : 1);
