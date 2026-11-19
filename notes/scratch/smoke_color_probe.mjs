/**
 * smoke_color_probe.mjs — 量烟雾**实际渲染出来**有多深，用来定材质 color。
 *
 * 为什么要量：烟雾是 0.14 不透明度的半透明体素，云心穿过的体素中位数 ~25 个，
 * 云心几乎不透明 —— 那里画出来的颜色基本就是材质 color 被灯光打亮的结果。
 * 改 color 到底会变多少、够不够"深"，靠脑补光照是猜不准的，直接读像素。
 *
 * 做法：同一帧里渲染两次（烟隐藏 / 烟显示），逐像素求差 → 差值大的像素就是烟
 * 覆盖的像素，不用去猜"哪些像素算烟"。两次渲染在同一个 JS 任务里跑完，所以
 * requestAnimationFrame 里的透明度呼吸不会插进来。
 *
 * 两个踩过的坑：
 *  1. 相机位置**必须用地图 BVH 挑视线**。一开始按固定方向退开 4 米，相机落在墙里，
 *     整团烟被挡住，只量到 4 个像素（覆盖率 0.02%），还以为烟变淡了。
 *  2. "芯"不能用"差值大"来判定 —— 差值大只说明背景和烟的颜色差得多，不说明烟
 *     厚。所以这里不再切芯/边，只看**全部烟像素的均值**（就是眼睛看到的），
 *     并且跨多团烟、多个方向取分布，避免拿单张图定颜色。
 *
 * 用法：
 *   python visualizer/server.py --port 5099 &
 *   node notes/scratch/smoke_color_probe.mjs http://127.0.0.1:5099 /tmp/v7_alltick.csz
 */

import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const URL = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3] || '/tmp/v7_alltick.csz';
const MAX_CLOUDS = Number(process.argv[4] || 8);

const CHROME = process.env.CHROME_PATH
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const profile = mkdtempSync(join(tmpdir(), 'smoke-probe-'));
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
if (!wsUrl) { chrome.kill(); console.error('连不上 CDP'); process.exit(1); }

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

await send('Runtime.enable');
await send('Page.enable');
await send('Page.navigate', { url: URL });
await sleep(3000);

// ── 页面内测量函数 ──────────────────────────────────────────────────────────
await evaluate(`window.__smokeSample = async (smokeIdx, dirRank, colors, nominalOpacity) => {
  const THREE = await import('/static/three/build/three.module.js');
  const v5 = window.__v5;
  const r = v5.round;
  const inst = r.smoke[smokeIdx];
  if (!inst || !inst.detonation) return null;
  const eid = inst.entity_id;

  // tick 落在烟雾存活期中段的采样点
  const midTick = Math.round((inst.first_tick + inst.last_tick) / 2);
  let best = -1, bestD = Infinity;
  for (let s = 0; s < r.ticks.length; s++) {
    const d = Math.abs(r.ticks[s] - midTick);
    if (d < bestD) { bestD = d; best = s; }
  }
  v5.seekSample(best);
  const mesh = v5.scene.getObjectByName('smoke-' + eid);
  if (!mesh || !mesh.count) return null;

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

  // 64 个均匀方向 → 按"最空"排序，取第 dirRank 个（0 = 最空）
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
  rr.toneMapping = THREE.ACESFilmicToneMapping;
  rr.toneMappingExposure = 0.95;

  const dist = Math.min(pick.free - 0.5, radius + 4.0);
  const cam = new THREE.PerspectiveCamera(55, W / H, 0.1, 600);
  cam.position.set(cx + pick.d[0] * dist, cy + pick.d[1] * dist, cz + pick.d[2] * dist);
  cam.lookAt(cx, cy, cz);
  cam.updateProjectionMatrix();

  const grab = () => {
    rr.render(v5.scene, cam);
    const gl = rr.getContext();
    const buf = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    return buf;
  };

  const prevOpacity = mesh.material.opacity;
  const prevColor = mesh.material.color.getHex();
  mesh.material.opacity = nominalOpacity;
  mesh.visible = false;
  const off = grab();
  mesh.visible = true;

  const x0 = Math.floor(W * 0.15), x1 = Math.ceil(W * 0.85);
  const y0 = Math.floor(H * 0.15), y1 = Math.ceil(H * 0.85);
  const lum = (R, G, B) => 0.2126 * R + 0.7152 * G + 0.0722 * B;

  const out = [];
  // 背景**只统计被烟盖住的那部分像素**（不是整个画面）—— 比的是"烟 vs 它背后
  // 那块墙"，不是"烟 vs 整个画面的平均亮度"。test_smoke_color.mjs 用同一个定义。
  for (const hex of colors) {
    mesh.material.color.setHex(hex);
    const on = grab();
    let n = 0, s = [0, 0, 0], b = [0, 0, 0], maxD = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = (y * W + x) * 4;
        const d = Math.max(Math.abs(on[i] - off[i]), Math.abs(on[i + 1] - off[i + 1]),
                           Math.abs(on[i + 2] - off[i + 2]));
        if (d < 6) continue;
        n++;
        s[0] += on[i]; s[1] += on[i + 1]; s[2] += on[i + 2];
        b[0] += off[i]; b[1] += off[i + 1]; b[2] += off[i + 2];
        maxD = Math.max(maxD, d);
      }
    }
    out.push({
      hex: '#' + hex.toString(16).padStart(6, '0'),
      px: n,
      coverPct: 100 * n / ((x1 - x0) * (y1 - y0)),
      smokeRGB: n ? s.map((v) => v / n) : null,
      smokeLum: n ? lum(...s.map((v) => v / n)) : null,
      bgRGB: n ? b.map((v) => v / n) : null,
      bgLum: n ? lum(...b.map((v) => v / n)) : null,
      maxDiff: maxD,
    });
  }

  // ── 同一相机再看一遍"背景是天空"的情况：把地图藏掉 ──────────────────────
  // 烟在天上飞的时候背景是天空，比地图亮得多。只按"烟 vs 地图"定颜色，可能会
  // 把烟配成对着天空像一块黑炭。
  const mapRoot = v5.scene.getObjectByName('map-root');
  const mapWasVisible = mapRoot ? mapRoot.visible : null;
  if (mapRoot) mapRoot.visible = false;
  mesh.visible = false;
  const skyOff = grab();
  mesh.visible = true;
  let bn = 0, bs = [0, 0, 0];
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * W + x) * 4;
      bs[0] += skyOff[i]; bs[1] += skyOff[i + 1]; bs[2] += skyOff[i + 2]; bn++;
    }
  }
  const skyRGB = bs.map((v) => v / bn);
  const sky = [];
  for (const hex of colors) {
    mesh.material.color.setHex(hex);
    const on = grab();
    let n = 0, s = [0, 0, 0];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = (y * W + x) * 4;
        const d = Math.max(Math.abs(on[i] - skyOff[i]), Math.abs(on[i + 1] - skyOff[i + 1]),
                           Math.abs(on[i + 2] - skyOff[i + 2]));
        if (d < 6) continue;
        n++; s[0] += on[i]; s[1] += on[i + 1]; s[2] += on[i + 2];
      }
    }
    sky.push({ px: n, smokeRGB: n ? s.map((v) => v / n) : null,
               smokeLum: n ? lum(...s.map((v) => v / n)) : null });
  }
  if (mapRoot) mapRoot.visible = mapWasVisible;

  mesh.material.opacity = prevOpacity;
  mesh.material.color.setHex(prevColor);
  rr.dispose();
  return { eid, sample: best, tick: r.ticks[best], voxels: mesh.count, radius,
           camDist: dist, free: pick.free,
           bgRGB: out[0].bgRGB, bgLum: out[0].bgLum,
           skyRGB, skyLum: lum(...skyRGB), perColor: out, sky };

  // （ASCII dump 走单独的函数，见 window.__smokeAscii）
}; true`);

/**
 * 把烟所在的那块画面降采样成 ASCII 亮度图。
 * 我看不了 PNG，所以"颜色对不对"最后只能靠这个看：烟应该是一团**有明暗渐变**
 * 的实体（中间实、边缘薄），而不是一片糊在背景上的平灰。
 */
await evaluate(`window.__smokeAscii = async (smokeIdx, hex, cols, rows) => {
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
  const m4 = new THREE.Matrix4(); const p = new THREE.Vector3();
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < mesh.count; i++) { mesh.getMatrixAt(i, m4); p.setFromMatrixPosition(m4); cx += p.x; cy += p.y; cz += p.z; }
  cx /= mesh.count; cy /= mesh.count; cz /= mesh.count;
  let radius = 0;
  for (let i = 0; i < mesh.count; i++) { mesh.getMatrixAt(i, m4); p.setFromMatrixPosition(m4); radius = Math.max(radius, Math.hypot(p.x - cx, p.y - cy, p.z - cz)); }
  const dirs = []; const GA = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < 64; i++) {
    const y = 1 - 2 * (i + 0.5) / 64; const rad = Math.sqrt(Math.max(0, 1 - y * y)); const th = GA * i;
    dirs.push([Math.cos(th) * rad, y, Math.sin(th) * rad]);
  }
  const scored = dirs.map((d) => { const t = v5.rawRaycast(cx, cy, cz, d[0], d[1], d[2], 60); return { d, free: t < 0 ? 60 : t }; }).sort((a, b) => b.free - a.free);
  const pick = scored[0];
  const W = 96, H = 40;
  const rr = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  rr.setPixelRatio(1); rr.setSize(W, H, false);
  rr.toneMapping = THREE.ACESFilmicToneMapping; rr.toneMappingExposure = 0.95;
  const dist = Math.min(pick.free - 0.5, radius + 4.0);
  const cam = new THREE.PerspectiveCamera(55, W / H, 0.1, 600);
  cam.position.set(cx + pick.d[0] * dist, cy + pick.d[1] * dist, cz + pick.d[2] * dist);
  cam.lookAt(cx, cy, cz); cam.updateProjectionMatrix();
  const grab = () => { rr.render(v5.scene, cam); const gl = rr.getContext();
    const buf = new Uint8Array(W * H * 4); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf); return buf; };
  const prevColor = mesh.material.color.getHex();
  if (hex) mesh.material.color.setHex(hex);
  const on = grab();
  mesh.visible = false; const off = grab(); mesh.visible = true;
  mesh.material.color.setHex(prevColor); rr.dispose();
  // readPixels 是自下而上的，翻转过来。
  // 打的是**有符号差值** on-off：这样一眼能看出烟比背景亮还是暗，以及云的轮廓。
  // （直接打绝对亮度没用：10 级灰阶里 12 个亮度差还不到一级，什么都看不出来。）
  const rowsOut = [];
  for (let y = H - 1; y >= 0; y--) {
    let line = '';
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const d = (0.2126 * (on[i] - off[i]) + 0.7152 * (on[i + 1] - off[i + 1])
                 + 0.0722 * (on[i + 2] - off[i + 2]));
      line += d < -40 ? '@' : d < -20 ? '%' : d < -8 ? '-' : d < 8 ? '.'
            : d < 20 ? '+' : d < 40 ? '*' : '#';
    }
    rowsOut.push(line);
  }
  return { rows: rowsOut, W, H, eid, radius, dist, free: pick.free };
}; true`);

await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
await sleep(1500);

for (let i = 0; i < 80; i++) {
  const tris = await evaluate(`(() => {
    let n = 0; const root = window.__v5.scene.getObjectByName('map-root');
    if (root) root.traverse((o) => { if (o.isMesh && o.geometry?.attributes?.position) n += o.geometry.attributes.position.count / 3; });
    return n;
  })()`);
  if (tris > 100000) break;
  await sleep(250);
}

await evaluate(`(() => {
  const t = document.getElementById('toggle-aim');
  t.checked = true; t.dispatchEvent(new Event('change')); return true;
})()`);
for (let i = 0; i < 120; i++) {
  await sleep(200);
  if (await evaluate('!!window.__v5.mapBVHStats')) break;
}
await evaluate(`(() => {
  const t = document.getElementById('toggle-aim');
  t.checked = false; t.dispatchEvent(new Event('change')); return true;
})()`);
const bvh = await evaluate('window.__v5.mapBVHStats');
console.log(`地图 BVH: ${bvh.triCount} 三角形 / ${Math.round(bvh.ms)} ms\n`);

// ── 跨回合、跨烟雾、跨视线方向采样 ─────────────────────────────────────────
// 候选色：把当前色 #dfe5ec 按比例压暗（**保持色相**，不是把蓝调加深）。
// 之前那批 #c4ccd6 / #b6bfca … 是手挑的，绝对蓝偏移越压越大（+13 → +24），
// 压暗的同时把烟越配越蓝。按比例缩放就不会动色相。
const BASE = [0xdf, 0xe5, 0xec];
const CANDIDATES = [1.00, 0.86, 0.80, 0.77, 0.74, 0.71, 0.68].map((k) => {
  const r = Math.round(BASE[0] * k), g = Math.round(BASE[1] * k), b = Math.round(BASE[2] * k);
  return (r << 16) | (g << 8) | b;
});
console.log('候选色:', CANDIDATES.map((c) => '#' + c.toString(16).padStart(6, '0')).join(' '), '\n');
const nRounds = await evaluate('window.__v5.summary.rounds.length');
const samples = [];

const selectRound = async (idx) => {
  await evaluate(`(() => { const s = document.getElementById('round-select');
    s.value = String(${idx}); s.dispatchEvent(new Event('change')); return true; })()`);
  for (let i = 0; i < 100; i++) {
    await sleep(150);
    const ok = await evaluate(`(() => { const r = window.__v5.round;
      return !!r && r.round_seconds && r.ticks.length > 0; })()`);
    const cur = await evaluate(`document.getElementById('round-select').value`);
    if (ok && Number(cur) === idx) return true;
  }
  return false;
};

const SKIP_SWEEP = !!process.env.SKIP_SWEEP;
for (let ri = 0; !SKIP_SWEEP && ri < nRounds && samples.length < MAX_CLOUDS * 3; ri++) {
  if (!await selectRound(ri)) continue;
  const nSmoke = await evaluate(`window.__v5.round.smoke.filter((s) => s.detonation).length`);
  for (let si = 0; si < nSmoke; si++) {
    for (const rank of [0, 6, 16]) {
      const res = await evaluate(
        `window.__smokeSample(${si}, ${rank}, ${JSON.stringify(CANDIDATES)}, 0.14)`, true);
      if (res && res.perColor[0].coverPct > 8) { samples.push({ ri, si, rank, ...res }); break; }
    }
    if (samples.filter((s) => s.ri === ri).length >= 3) break;
  }
  if (samples.length && samples.length % 3 === 0) {
    process.stdout.write(`  已采样 ${samples.length} 团…\n`);
  }
  if (samples.length >= MAX_CLOUDS * 3) break;
}

console.log(`\n共采样 ${samples.length} 团烟（回合 × 烟雾 × 视线方向）\n`);
console.log('  回合 烟  dir  体素  半径m  相机m  覆盖率  背景亮度');
for (const s of samples) {
  console.log(`  ${String(s.ri).padStart(4)} ${String(s.si).padStart(2)} ${String(s.rank).padStart(4)}`
    + `  ${String(s.voxels).padStart(5)}  ${s.radius.toFixed(1).padStart(5)}`
    + `  ${s.camDist.toFixed(1).padStart(5)}  ${s.perColor[0].coverPct.toFixed(0).padStart(5)}%`
    + `  ${s.bgLum.toFixed(1).padStart(7)}`);
}

// ── 汇总 ────────────────────────────────────────────────────────────────────
const lum = (R, G, B) => 0.2126 * R + 0.7152 * G + 0.0722 * B;
const median = (a) => { const b = a.slice().sort((p, q) => p - q); return b[b.length >> 1]; };
const mean = (a) => a.reduce((p, q) => p + q, 0) / a.length;

console.log('\n候选色 → 跨所有采样的渲染结果（均值 / 中位数）');
if (!samples.length) {
  console.log('  （SKIP_SWEEP，跳过）');
}
console.log('  color     基色亮度  烟rgb(中位采样)      烟亮度  烟/地图  烟/天空  最坏(地图)  饱和度');
const table = [];
for (let ci = 0; ci < CANDIDATES.length && samples.length; ci++) {
  const hex = CANDIDATES[ci];
  const base = lum((hex >> 16) & 255, (hex >> 8) & 255, hex & 255);
  const ls = samples.map((s) => s.perColor[ci].smokeLum);
  const ratios = samples.map((s) => s.perColor[ci].smokeLum / s.bgLum);
  const skyRatios = samples.map((s) => s.sky[ci].smokeLum / s.skyLum);
  // 饱和度 = (max-min)/max，看有没有偏色（烟应该是中性灰）
  const sats = samples.map((s) => {
    const c = s.perColor[ci].smokeRGB;
    const mx = Math.max(...c), mn = Math.min(...c);
    return mx > 0 ? (mx - mn) / mx : 0;
  });
  const mid = samples[Math.floor(samples.length / 2)].perColor[ci].smokeRGB;
  const row = {
    hex: '#' + hex.toString(16).padStart(6, '0'), base, midRGB: mid,
    lumMed: median(ls), ratioMed: median(ratios), ratioMean: mean(ratios),
    ratioMax: Math.max(...ratios), skyRatio: median(skyRatios), sat: mean(sats),
  };
  table.push(row);
  console.log(`  ${row.hex}  ${base.toFixed(1).padStart(8)}`
    + `  rgb(${mid.map((v) => v.toFixed(0).padStart(3)).join(',')})`
    + `  ${row.lumMed.toFixed(1).padStart(7)}`
    + `  ${row.ratioMed.toFixed(3).padStart(7)}  ${row.skyRatio.toFixed(3).padStart(7)}`
    + `  ${row.ratioMax.toFixed(3).padStart(10)}  ${row.sat.toFixed(3).padStart(7)}`);
}
if (samples.length) {
  console.log(`\n参考：背景（地图）亮度中位 ${median(samples.map((s) => s.bgLum)).toFixed(1)}`
    + `，天空亮度中位 ${median(samples.map((s) => s.skyLum)).toFixed(1)}`);
}

// ── ASCII 亮度图：确认烟还是一团有渐变的实体，不是平灰 ──────────────────────
// 左列显式用"基线色"渲染（默认旧值 #dfe5ec），右列用候选色。
// 不能让左列跟着源码里的当前值走 —— 改完 visuals.js 再跑，两边就变成同一个颜色了。
const BASE_COLOR = Number(process.env.BASE_COLOR || 0xdfe5ec);
const CHOSEN = Number(process.env.CHOSEN || 0xa5a9af);
const hx = (v) => '#' + v.toString(16).padStart(6, '0');
console.log(`\n有符号差值图（左 = 基线 ${hx(BASE_COLOR)}，右 = 候选 ${hx(CHOSEN)}）`);
console.log(`烟比背景**暗** → - % @ ；亮 → + * # ；.| 表示没有烟\n`);
const ASCII_ROUNDS = Number(process.env.ASCII_ROUNDS || 3);
for (let ri = 0; ri < ASCII_ROUNDS; ri++) {
  if (!await selectRound(ri)) continue;
  const a = await evaluate(`window.__smokeAscii(0, ${BASE_COLOR}, 96, 40)`, true);
  if (!a) continue;
  const b = await evaluate(`window.__smokeAscii(0, ${CHOSEN}, 96, 40)`, true);
  console.log(`── 回合 ${ri} · eid ${a.eid} · 半径 ${a.radius.toFixed(1)} m ──`);
  for (let i = 0; i < a.rows.length; i++) {
    console.log(`  ${a.rows[i]}  |  ${b.rows[i]}`);
  }
  console.log('');
}

ws.close();
chrome.kill();
