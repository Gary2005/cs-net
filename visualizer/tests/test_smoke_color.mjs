/**
 * test_smoke_color.mjs — 烟雾"深不深"的回归测试。
 *
 * 为什么需要它：烟雾颜色是个纯视觉常数，没有测试的话每次调都只能靠眼睛，而
 * "眼睛觉得浅"背后其实是个可以量的硬事实 —— 旧值 `0xdfe5ec` 渲染出来是
 * **背后那块墙的 1.074 倍亮（最坏一团 1.182 倍）**，也就是烟比它背后的墙还亮，
 * 看着像一层发白的雾。这种问题一旦回归（比如有人觉得"烟应该更白一点"改回去），
 * 必须有测试拦住。
 *
 * 量法（不猜"哪些像素算烟"）：
 *   同一条视线渲染四次 ——
 *     M = 只画地图        S = 地图也藏掉（背景 = 天空）
 *     Pm = 地图 + 烟      Ps = 天空 + 烟
 *   烟雾在像素上的**有效不透明度** a 和它**自己的颜色** C 可以解出来：
 *     Pm − Ps = (1−a)(M − S)   →  (1−a) = ⟨Pm−Ps, M−S⟩ / |M−S|²
 *     C = (Pm − (1−a)·M) / a
 *   这样 a 和 C 都跟背后是什么无关。C 只取 a > 0.9 的像素（云心）求均值，
 *   就是"烟到底什么颜色"的定义式答案。
 *
 * 顺带还量了两件事：
 *   · **有效不透明度分布** —— 云心必须接近 1（不透明），边缘必须有中间值
 *     （0.05~0.5 的像素得占一定比例），否则就不是一团有体积感的烟，而是一块
 *     硬边的贴片。这条盯的是 SMOKE_OPACITY。
 *   · **色相** —— 烟应该是中性偏一点点冷，不该被压成蓝的（等比压暗 vs 手挑
 *     蓝调的区别就在这里）。
 *
 * 需要一个有烟的 .csz（默认用仓库里那份 de_ancient）：
 *   python visualizer/server.py --port 5099 &
 *   node visualizer/tests/test_smoke_color.mjs http://127.0.0.1:5099 /path/to/match.csz
 *   CS2_CSZ=/path/to/match.csz node visualizer/tests/test_smoke_color.mjs
 *
 * 没有 .csz / 没装 Chrome 时打印 SKIP 并以 0 退出。
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const URL = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3] || process.env.CS2_CSZ
  || 'visualizer/outputs/parivision-vs-tyloo-m2-ancient.csz';
const CHROME = process.env.CHROME_PATH
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const MAX_CLOUDS = Number(process.env.MAX_CLOUDS || 6);

// 期望值（改动视觉效果时同步改这里，并且必须重新跑 probe 量一遍）
const EXPECT_COLOR = 0xa5a9af;
const EXPECT_OPACITY = 0.14;
// 烟/地图中位：要比背后的墙暗，但不能暗成黑块。
// 实测两份 .csz 分别是 0.838 / 0.825（采样到哪几团烟会影响这个数），所以下界留到 0.78。
const MAP_RATIO_BAND = [0.78, 0.96];
const SKY_RATIO_BAND = [0.56, 0.78];   // 烟/天空中位（实测 0.639，留够余量）
const OLD_COLOR = 0xdfe5ec;            // 灵敏度对照：旧值必须明显更亮

let pass = 0, fail = 0;
function check(name, ok, detail = '') {
  if (ok) { pass++; console.log(`  ✓ ${name}${detail ? `  ${detail}` : ''}`); }
  else { fail++; console.log(`  ✗ ${name}${detail ? `  ${detail}` : ''}`); }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const median = (a) => { const b = a.slice().sort((p, q) => p - q); return b[b.length >> 1]; };
const mean = (a) => a.reduce((p, q) => p + q, 0) / a.length;
const lum = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
// 线性 → sRGB（coreLinear 是线性值，报告时转回大家熟悉的显示值）
const lin2srgb = (v) => {
  const x = Math.min(1, Math.max(0, v / 255));
  return (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055) * 255;
};
const coreDisplay = (s) => s.coreLinear.map(lin2srgb);
const f = (v, n = 3) => (v === null || v === undefined ? 'n/a' : v.toFixed(n));

if (!existsSync(CSZ)) {
  console.log(`SKIP: 找不到 .csz（${CSZ}）`);
  console.log('      传一个**有烟雾**的文件：node visualizer/tests/test_smoke_color.mjs <url> <match.csz>');
  process.exit(0);
}
if (!existsSync(CHROME)) {
  console.log(`SKIP: 找不到 Chrome（${CHROME}）`);
  process.exit(0);
}

const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const profile = mkdtempSync(join(tmpdir(), 'smoke-color-'));
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
  console.log('SKIP: 连不上 headless Chrome 的 CDP');
  process.exit(0);
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
    throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 900));
  }
  return r.result?.result?.value;
};

const finish = async (code) => {
  try { ws.close(); } catch { /* 已经关了 */ }
  chrome.kill();
  await sleep(200);
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(code);
};

await send('Runtime.enable');
await send('Page.enable');
await send('Page.navigate', { url: URL });
await sleep(3000);

// ── 页面内：解出烟雾的有效不透明度和它自己的颜色 ────────────────────────────
await evaluate(`window.__smokeColorSample = async (smokeIdx, dirRank, colorOverride) => {
  const mean = (a) => a.reduce((p, q) => p + q, 0) / a.length;
  const median = (a) => { const b = a.slice().sort((p, q) => p - q); return b[b.length >> 1]; };
  const THREE = await import('/static/three/build/three.module.js');
  const v5 = window.__v5;
  const r = v5.round;
  const inst = r.smoke[smokeIdx];
  if (!inst || !inst.detonation) return null;
  const eid = inst.entity_id;

  const midTick = Math.round((inst.first_tick + inst.last_tick) / 2);
  let best = -1, bestD = Infinity;
  for (let s = 0; s < r.ticks.length; s++) {
    const d = Math.abs(r.ticks[s] - midTick);
    if (d < bestD) { bestD = d; best = s; }
  }
  v5.seekSample(best);
  const mesh = v5.scene.getObjectByName('smoke-' + eid);
  if (!mesh || !mesh.count) return null;

  // 体素质心 / 半径
  const m4 = new THREE.Matrix4(); const p = new THREE.Vector3();
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < mesh.count; i++) {
    mesh.getMatrixAt(i, m4); p.setFromMatrixPosition(m4);
    cx += p.x; cy += p.y; cz += p.z;
  }
  cx /= mesh.count; cy /= mesh.count; cz /= mesh.count;
  let radius = 0;
  for (let i = 0; i < mesh.count; i++) {
    mesh.getMatrixAt(i, m4); p.setFromMatrixPosition(m4);
    radius = Math.max(radius, Math.hypot(p.x - cx, p.y - cy, p.z - cz));
  }

  // 用地图 BVH 挑一条不被墙挡住的视线（相机落进墙里会量到一团被遮住的烟）
  const REACH = 60;
  const dirs = [];
  const GA = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < 64; i++) {
    const y = 1 - 2 * (i + 0.5) / 64;
    const rad = Math.sqrt(Math.max(0, 1 - y * y));
    const th = GA * i;
    dirs.push([Math.cos(th) * rad, y, Math.sin(th) * rad]);
  }
  const scored = dirs.map((d) => {
    const t = v5.rawRaycast(cx, cy, cz, d[0], d[1], d[2], REACH);
    return { d, free: t < 0 ? REACH : t };
  }).sort((a, b) => b.free - a.free);
  const pick = scored[Math.min(dirRank, scored.length - 1)];
  if (pick.free < radius + 0.5) return null;

  const W = 260, H = 200;
  const rr = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  rr.setPixelRatio(1); rr.setSize(W, H, false);
  const dist = Math.min(pick.free - 0.5, radius + 4.0);
  const cam = new THREE.PerspectiveCamera(55, W / H, 0.1, 600);
  cam.position.set(cx + pick.d[0] * dist, cy + pick.d[1] * dist, cz + pick.d[2] * dist);
  cam.lookAt(cx, cy, cz); cam.updateProjectionMatrix();
  const grab = () => {
    rr.render(v5.scene, cam);
    const gl = rr.getContext();
    const buf = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    return buf;
  };
  // 两套渲染设置，量的是两件不同的事：
  //  · 显示空间（ACES + sRGB）= 用户实际看到的 → 量"深不深"（烟像素 vs 同像素没烟时）
  //  · 线性空间（不做色调映射、不编码）= 帧缓冲里就是线性值 → 混色公式
  //      Pm = a·C + (1−a)·M  严格成立，可以精确解出有效不透明度 a 和烟自己的颜色 C。
  //    一开始我在显示空间解这个方程，结果 a 和 C 都偏 —— 因为帧缓冲里的值已经是
  //    编码+色调映射过的，混合不在线性空间做，方程本身就不成立。
  const setDisplay = () => {
    rr.toneMapping = THREE.ACESFilmicToneMapping; rr.toneMappingExposure = 0.95;
    rr.outputColorSpace = THREE.SRGBColorSpace;
  };
  const setLinear = () => {
    rr.toneMapping = THREE.NoToneMapping; rr.outputColorSpace = THREE.LinearSRGBColorSpace;
  };

  const mapRoot = v5.scene.getObjectByName('map-root');
  const prevOpacity = mesh.material.opacity;
  const prevColor = mesh.material.color.getHex();
  mesh.material.opacity = ${EXPECT_OPACITY};
  // 传了 colorOverride 就用它渲染（灵敏度对照用），量完再还原
  if (colorOverride) mesh.material.color.setHex(colorOverride);

  setDisplay();
  mesh.visible = false; mapRoot.visible = true;  const M  = grab();   // 只地图
  mesh.visible = true;                           const Pm = grab();   // 地图 + 烟
  // 天空也要一张**显示空间**的：烟在天上飞的时候背景就是它。
  // （一开始这里直接拿了线性空间那张 Sl，算出来的"烟/天空"整整偏高一截。）
  mesh.visible = false; mapRoot.visible = false; const Sd = grab();   // 只天空
  mapRoot.visible = true;
  setLinear();
  mesh.visible = false; mapRoot.visible = true;  const Ml  = grab();
  mapRoot.visible = false;                       const Sl  = grab();   // 只天空
  mapRoot.visible = true;  mesh.visible = true;  const Pl  = grab();   // 地图 + 烟
  mapRoot.visible = false;                       const Psl = grab();   // 天空 + 烟
  mapRoot.visible = true;

  const x0 = Math.floor(W * 0.15), x1 = Math.ceil(W * 0.85);
  const y0 = Math.floor(H * 0.15), y1 = Math.ceil(H * 0.85);
  const alphas = [], coreC = [], mapDisp = [], skyDisp = [], smokeDisp = [];
  let determined = 0, opaque = 0, softEdge = 0, coverage = 0, covered = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * W + x) * 4;
      // 显示空间：烟覆盖了哪些像素（有没有烟，差得出来）
      const diff = Math.max(Math.abs(Pm[i] - M[i]), Math.abs(Pm[i + 1] - M[i + 1]),
                            Math.abs(Pm[i + 2] - M[i + 2]));
      if (diff > 6) {
        covered++;
        smokeDisp.push([Pm[i], Pm[i + 1], Pm[i + 2]]);
        mapDisp.push([M[i], M[i + 1], M[i + 2]]);
        skyDisp.push([Sd[i], Sd[i + 1], Sd[i + 2]]);
      }
      // 线性空间：解 a 和 C
      const m = [Ml[i], Ml[i + 1], Ml[i + 2]];
      const s = [Sl[i], Sl[i + 1], Sl[i + 2]];
      const d = [m[0] - s[0], m[1] - s[1], m[2] - s[2]];
      const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (dd < 400) continue;            // 地图和天空太接近 → 解不出 a
      determined++;
      const pm = [Pl[i], Pl[i + 1], Pl[i + 2]];
      const ps = [Psl[i], Psl[i + 1], Psl[i + 2]];
      const e = [pm[0] - ps[0], pm[1] - ps[1], pm[2] - ps[2]];
      let oneMinusA = (e[0] * d[0] + e[1] * d[1] + e[2] * d[2]) / dd;
      oneMinusA = Math.min(1, Math.max(0, oneMinusA));
      const a = 1 - oneMinusA;
      if (a > 0.02) coverage++;
      if (a > 0.9) {
        opaque++;
        coreC.push([(pm[0] - oneMinusA * m[0]) / a,
                    (pm[1] - oneMinusA * m[1]) / a,
                    (pm[2] - oneMinusA * m[2]) / a]);
      } else if (a > 0.05 && a < 0.5) softEdge++;
      if (a > 0.05) alphas.push(a);
    }
  }
  mesh.material.opacity = prevOpacity;
  mesh.material.color.setHex(prevColor);
  const avg = (arr) => arr.length ? arr[0].map((_, k) => mean(arr.map((c) => c[k]))) : null;
  const out = {
    eid, sample: best, voxels: mesh.count, radius, camDist: dist,
    color: '#' + (colorOverride ?? prevColor).toString(16).padStart(6, '0'),
    opacity: prevOpacity,
    determined, coverage, covered,
    coverPct: 100 * covered / ((x1 - x0) * (y1 - y0)),
    alphaCoverPct: 100 * coverage / Math.max(1, determined),
    opaquePct: 100 * opaque / Math.max(1, determined),
    softEdgePct: 100 * softEdge / Math.max(1, determined),
    maxAlpha: alphas.length ? Math.max(...alphas) : null,
    alphaMedian: alphas.length ? median(alphas) : null,
    // 显示空间：用户看到的烟 / 同一像素没烟时的背景 / 天空
    smokeRGB: avg(smokeDisp), mapRGB: avg(mapDisp), skyRGB: avg(skyDisp),
    // 线性空间：烟自己的颜色（去掉了背景的贡献）
    coreLinear: avg(coreC),
  };
  rr.dispose();
  return out;
}; true`);

// ── 加载 ────────────────────────────────────────────────────────────────────
// 等页面把 `__v5` 建出来再 loadPath：`init()` 是异步跑完才挂这个对象的，
// navigate 完立刻调会拿到 undefined（"Cannot read properties of undefined"）。
for (let i = 0; i < 120; i++) {
  if (await evaluate(`!!window.__v5 && !!window.__v5.loadPath`).catch(() => false)) break;
  await sleep(500);
}
console.log(`加载 ${CSZ}`);
await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
await sleep(1500);

let mapTris = 0;
for (let i = 0; i < 80; i++) {
  mapTris = await evaluate(`(() => {
    let n = 0; const root = window.__v5.scene.getObjectByName('map-root');
    if (root) root.traverse((o) => { if (o.isMesh && o.geometry?.attributes?.position) n += o.geometry.attributes.position.count / 3; });
    return n;
  })()`);
  if (mapTris > 100000) break;
  await sleep(250);
}

console.log('\n0. 前置条件');
check('地图几何加载完成', mapTris > 100000, `${mapTris} 三角形`);
if (mapTris <= 100000) await finish(1);

// 挑视线要用 BVH
await evaluate(`(() => { const t = document.getElementById('toggle-aim');
  t.checked = true; t.dispatchEvent(new Event('change')); return true; })()`);
for (let i = 0; i < 120; i++) {
  await sleep(200);
  if (await evaluate('!!window.__v5.mapBVHStats')) break;
}
await evaluate(`(() => { const t = document.getElementById('toggle-aim');
  t.checked = false; t.dispatchEvent(new Event('change')); return true; })()`);
const bvh = await evaluate('window.__v5.mapBVHStats');
check('地图 BVH 建好了（挑视线用）', !!bvh, bvh ? `${bvh.triCount} 三角形 / ${Math.round(bvh.ms)} ms` : '');

// ── 1. 材质常数 ─────────────────────────────────────────────────────────────
console.log('\n1. 材质常数（从真正画出来的网格上读，不是读源码）');
const roundIds = await evaluate('window.__v5.summary.rounds.map((r) => r.id)');
const nRounds = roundIds.length;
/**
 * 切回合并**等它真的到位**。
 * 只等下拉框的 value 是不够的 —— 加载是异步的，value 立刻就是新值，而
 * `window.__v5.round` 还是上一个回合。踩过这个坑：两次跑测试采样到了不同的
 * 烟雾团，数字对不上，还以为是渲染不稳定。所以这里比的是回合 **id**。
 */
const selectRound = async (idx) => {
  await evaluate(`(() => { const s = document.getElementById('round-select');
    s.value = String(${idx}); s.dispatchEvent(new Event('change')); return true; })()`);
  for (let i = 0; i < 120; i++) {
    await sleep(150);
    const ok = await evaluate(`(() => { const r = window.__v5.round;
      return !!r && r.ticks && r.ticks.length > 0 && r.id === ${roundIds[idx]}; })()`);
    if (ok) return true;
  }
  return false;
};

let firstSmoke = null;
for (let ri = 0; ri < nRounds && !firstSmoke; ri++) {
  if (!await selectRound(ri)) continue;
  const n = await evaluate(`window.__v5.round.smoke.filter((s) => s.detonation).length`);
  if (n > 0) firstSmoke = { ri, n };
}
if (!firstSmoke) {
  console.log('SKIP: 这个 .csz 里没有带引爆点的烟雾');
  await finish(0);
}
const mat = await evaluate(`(async () => {
  const v5 = window.__v5;
  const inst = v5.round.smoke.find((s) => s.detonation);
  v5.seekSample(Math.round((inst.first_tick + inst.last_tick) / 2 - v5.round.ticks[0]));
  const mesh = v5.scene.getObjectByName('smoke-' + inst.entity_id);
  return mesh ? { color: '#' + mesh.material.color.getHexString(),
                  hex: mesh.material.color.getHex(),
                  opacity: mesh.material.opacity,
                  transparent: mesh.material.transparent,
                  depthWrite: mesh.material.depthWrite } : null;
})()`, true);
check('烟雾网格存在', !!mat);
if (!mat) await finish(1);
check('基色 = 期望值', mat.hex === EXPECT_COLOR,
  `${mat.color}（期望 #${EXPECT_COLOR.toString(16)}）`);
check('单个体素不透明度 = 期望值（呼吸会 ±8%，所以只查量级）',
  Math.abs(mat.opacity - EXPECT_OPACITY) < EXPECT_OPACITY * 0.12,
  `${f(mat.opacity)}（标称 ${EXPECT_OPACITY}）`);
check('仍然是不写深度的半透明材质', mat.transparent === true && mat.depthWrite === false,
  `transparent=${mat.transparent} depthWrite=${mat.depthWrite}`);

// ── 2. 跨烟雾采样 ───────────────────────────────────────────────────────────
console.log(`\n2. 跨烟雾 / 视线方向采样（最多 ${MAX_CLOUDS} 团）`);
const samples = [];
for (let ri = 0; ri < nRounds && samples.length < MAX_CLOUDS; ri++) {
  if (!await selectRound(ri)) continue;
  const n = await evaluate(`window.__v5.round.smoke.filter((s) => s.detonation).length`);
  for (let si = 0; si < n && samples.length < MAX_CLOUDS; si++) {
    for (const rank of [0, 8, 20]) {
      const res = await evaluate(`window.__smokeColorSample(${si}, ${rank})`, true);
      if (res && res.coverPct > 8 && res.alphaCoverPct > 20 && res.opaquePct > 0.5) {
        samples.push({ ri, si, rank, ...res });
        break;
      }
    }
  }
}
check('采到了足够的烟雾样本', samples.length >= 3, `${samples.length} 团`);
if (samples.length < 3) await finish(1);

console.log('\n   回合 烟 dir 体素  占画面%  a>0.9%  薄边%  a中位  a最大  '
  + '烟rgb(显示)        烟亮度  地图亮度  烟/地图  烟/天空  烟本色rgb(线性解出)');
for (const s of samples) {
  const core = coreDisplay(s);
  console.log(`   ${String(s.ri).padStart(4)} ${String(s.si).padStart(2)} ${String(s.rank).padStart(3)}`
    + ` ${String(s.voxels).padStart(5)} ${f(s.coverPct, 1).padStart(7)} ${f(s.opaquePct, 1).padStart(7)}`
    + ` ${f(s.softEdgePct, 1).padStart(6)} ${f(s.alphaMedian, 3).padStart(6)} ${f(s.maxAlpha, 3).padStart(6)}`
    + `  rgb(${s.smokeRGB.map((v) => v.toFixed(0).padStart(3)).join(',')})`
    + ` ${f(lum(s.smokeRGB), 1).padStart(7)} ${f(lum(s.mapRGB), 1).padStart(9)}`
    + ` ${f(lum(s.smokeRGB) / lum(s.mapRGB)).padStart(8)}`
    + ` ${f(lum(s.smokeRGB) / lum(s.skyRGB)).padStart(8)}`
    + `  rgb(${core.map((v) => v.toFixed(0).padStart(3)).join(',')})`);
}

// ── 3. 深度 ─────────────────────────────────────────────────────────────────
console.log('\n3. 深度：烟要比背后的墙暗，但不能暗成黑块');
// 量的就是"用户看到的"：有烟的像素 vs 同一像素没烟时（显示空间）
const mapRatios = samples.map((s) => lum(s.smokeRGB) / lum(s.mapRGB));
const skyRatios = samples.map((s) => lum(s.smokeRGB) / lum(s.skyRGB));
const mapMed = median(mapRatios), skyMed = median(skyRatios);
// 天空是个稳定又亮的参照物，所以"烟比天空暗"可以要求每一团都成立。
// 地图不行 —— 深色墙角本来就比烟暗，烟比它亮是物理上正常的，所以只看中位。
check('每一团烟都比天空暗（天空是稳定参照；旧值下这里也偏亮）',
  Math.max(...skyRatios) < 0.9,
  `最坏 ${f(Math.max(...skyRatios))}，中位 ${f(skyMed)}`);
check('烟/天空 落在目标区间', skyMed >= SKY_RATIO_BAND[0] && skyMed <= SKY_RATIO_BAND[1],
  `${f(skyMed)} ∈ [${SKY_RATIO_BAND[0]}, ${SKY_RATIO_BAND[1]}]`);
check('烟/地图 中位落在目标区间（旧值 1.074 = 比墙还亮，看着就是一层发白的雾）',
  mapMed >= MAP_RATIO_BAND[0] && mapMed <= MAP_RATIO_BAND[1],
  `${f(mapMed)} ∈ [${MAP_RATIO_BAND[0]}, ${MAP_RATIO_BAND[1]}]`);
check('多数烟团比背后的墙暗', mean(mapRatios.map((r) => (r < 1 ? 1 : 0))) >= 0.6,
  `${Math.round(100 * mean(mapRatios.map((r) => (r < 1 ? 1 : 0))))}% 的烟团 ratio < 1`);
check('云心不是黑块（亮度仍在中灰以上）',
  median(samples.map((s) => lum(s.smokeRGB))) > 90,
  `烟亮度中位 ${f(median(samples.map((s) => lum(s.smokeRGB))), 1)}`);

// ── 4. 体积感（有效不透明度分布）────────────────────────────────────────────
console.log('\n4. 体积感：云心不透明 + 边缘有过渡（这条盯的是 SMOKE_OPACITY）');
check('云心接近不透明', median(samples.map((s) => s.maxAlpha)) > 0.9,
  `a_max 中位 ${f(median(samples.map((s) => s.maxAlpha)))}`);
check('存在薄雾过渡带（不是硬边贴片）',
  median(samples.map((s) => s.softEdgePct)) > 3,
  `0.05<a<0.5 的像素占比中位 ${f(median(samples.map((s) => s.softEdgePct)), 1)}%`);
check('整团有相当比例是半透明的（真的能看到体积，不是实心块）',
  median(samples.map((s) => s.alphaMedian)) < 0.95,
  `a 中位 ${f(median(samples.map((s) => s.alphaMedian)))}`);

// ── 5. 色相 ─────────────────────────────────────────────────────────────────
console.log('\n5. 色相：中性偏一点冷，不能被压成蓝的');
// 色相看的是**烟本色**（线性空间解出来的 C，已去掉背景贡献），不是"烟+背景"的
// 混合色 —— 否则蓝天透过薄雾会把烟显成蓝的，量到的就不是材质的色相了。
const cores = samples.map(coreDisplay);
const sats = cores.map((c) => {
  const mx = Math.max(...c), mn = Math.min(...c);
  return mx > 0 ? (mx - mn) / mx : 0;
});
const blueShift = cores.map((c) => c[2] - c[0]);
const coreLums = cores.map(lum);
check('云心是中性灰（低饱和）', mean(sats) < 0.15, `饱和度均值 ${f(mean(sats))}`);
check('冷色方向（B ≥ R）', Math.min(...blueShift) >= 0, `B−R 最小 ${f(Math.min(...blueShift), 1)}`);
// 等比压暗时 (B−R)/亮度 应该基本不变；手挑蓝调会把它推上去
check('蓝偏移没有随压暗而变大（等比压暗 vs 手挑蓝调）',
  mean(blueShift) / mean(coreLums) < 0.09,
  `(B−R)/亮度 = ${f(mean(blueShift) / mean(coreLums))}`);

// ── 6. 灵敏度：这个测试真的能测出颜色变化 ──────────────────────────────────
console.log('\n6. 灵敏度对照：把颜色换回旧值，必须能量出明显更亮');
const ref = samples[samples.length - 1];
const oldRes = await evaluate(
  `window.__smokeColorSample(${ref.si}, ${ref.rank}, ${OLD_COLOR})`, true);
check('对照量到的是同一团烟', !!oldRes && oldRes.eid === ref.eid,
  oldRes ? `eid ${oldRes.eid} vs ${ref.eid}` : '量不到');
check('旧色渲染出来确实明显更亮',
  !!oldRes && lum(oldRes.smokeRGB) > lum(ref.smokeRGB) * 1.1,
  oldRes ? `旧 ${f(lum(oldRes.smokeRGB), 1)} vs 新 ${f(lum(ref.smokeRGB), 1)}`
         : '量不到');
check('旧色确实会"比墙亮"（说明这条回归是有意义的）',
  !!oldRes && lum(oldRes.smokeRGB) / lum(oldRes.mapRGB) > 1.0,
  oldRes ? `旧值 烟/地图 = ${f(lum(oldRes.smokeRGB) / lum(oldRes.mapRGB))}` : '量不到');

await finish(fail === 0 ? 0 : 1);
