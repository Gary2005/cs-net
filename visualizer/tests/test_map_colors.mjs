#!/usr/bin/env node
/**
 * test_map_colors.mjs — 「地图用的是游戏材质的颜色，而且真的贴上去了」（headless Chrome + CDP）。
 *
 * 背景：`maps/optimized_obj_files/*.obj` 是 Open3D 导出的**纯几何**灰模
 * （`vt` / `vn` / `usemtl` 全是 0 条），以前靠 `map-loader.js` 按法线糊一层水泥灰。
 * 现在 `离线资产管线` 从 CS2 地图 VPK 里回投影出"每个三角形
 * 该是什么颜色"，存成 `<map>.colors.bin`，由 `map-loader.js` 在载入地图后涂到
 * `color` 顶点色属性上。
 *
 * 这个测试两头都查：
 *
 *   A. 服务端：`/api/map/<map>/colors` 返回的二进制头合法、长度自洽、面数 > 0，
 *      没烘过的图返回 404（前端要能退回水泥灰）；
 *   B. 前端：`__v5.mapColors.source === 'game'`，面数和网格/文件都对得上；
 *   C. 属性：每面 3 个角必须**同色**（旁车是"每三角形一个颜色"），且颜色足够多样
 *      —— 水泥灰那套只有一种色系，游戏材质有几万种；
 *   D. 小地图：换过顶点色之后 `redrawMinimap()` 还能出图（它会把 color 换成 mmColor）；
 *   E. 全程没有控制台报错。
 *
 * 用法（需要 Chrome/Edge 和一个已经在跑的 server，地图必须是烘过的）：
 *
 *     node visualizer/tests/test_map_colors.mjs http://127.0.0.1:5099 \
 *          app/data/csz/aurora-vs-vitality-m2-inferno__ca728317.csz
 *
 * 没有 Chrome、或者这张图没烘过，就打印 SKIP 并以 0 退出。
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
  '/usr/bin/google-chrome',
].filter(Boolean);

const BASE = (process.argv[2] || 'http://127.0.0.1:5099').replace(/\/$/, '');
const CSZ = process.argv[3] || 'app/data/csz/aurora-vs-vitality-m2-inferno__ca728317.csz';
const PORT = 9550 + (process.pid % 200);

const MAGIC = 0x524c434d;      // "MCLR"
const VERSION = 2;             // v1 = 只有颜色；v2 = 颜色 + 每面 3 字节 int8 平滑法线
const HEADER = 12;

const chrome = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chrome) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
function check(name, cond, detail = '') {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? `   ${detail}` : ''}`);
  if (!cond) failed++;
}

// ── A. 服务端：先把旁车文件本身验一遍 ────────────────────────────────────────
/** demo 名 -> 地图名（和 CSZ 文件名对齐，够用了；B 段还会用页面里的真值再核一次） */
const guessMap = (CSZ.match(/-(ancient|anubis|dust2|inferno|mirage|nuke|overpass|cache)\.dem/)
  || CSZ.match(/-(ancient|anubis|dust2|inferno|mirage|nuke|overpass|cache)__/))?.[1];
const MAP = guessMap ? `de_${guessMap}` : 'de_inferno';

console.log(`A. 服务端 /api/map/${MAP}/colors`);
let serverCount = 0;
try {
  const res = await fetch(`${BASE}/api/map/${MAP}/colors`);
  check('HTTP 200', res.ok, `status ${res.status}`);
  if (!res.ok) {
    console.log(`SKIP: ${MAP} has no baked colours (run 离线资产管线 --map ${MAP})`);
    process.exit(0);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  const magic = buf.readUInt32LE(0);
  const version = buf.readUInt16LE(4);
  serverCount = buf.readUInt32LE(8);
  check('magic "MCLR"', magic === MAGIC);
  check('version 1..2', version >= 1 && version <= VERSION, `v${version}`);
  // v1 只有颜色；v2 后面还有每个面的 3 字节 int8 法线
  const withNormals = version >= 2 && buf.length === HEADER + serverCount * 6;
  check('长度自洽', buf.length === HEADER + serverCount * (withNormals ? 6 : 3),
    `${buf.length} bytes / ${serverCount} tris${withNormals ? ' + normals' : ''}`);
  check('count > 1e6', serverCount > 1e6, `${serverCount.toLocaleString()} triangles`);
  const bogus = await fetch(`${BASE}/api/map/__no_such_map__/colors`);
  check('未烘过的图返回 404（前端据此退回水泥灰）', bogus.status === 404, `status ${bogus.status}`);
} catch (err) {
  console.log(`SKIP: cannot reach ${BASE} (${err.message})`);
  process.exit(0);
}

const profile = mkdtempSync(join(tmpdir(), 'map-colors-'));
const proc = spawn(chrome, [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check',
  '--disable-gpu-sandbox', '--no-sandbox',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1280,800',
  'about:blank',
], { stdio: ['ignore', 'ignore', 'ignore'] });

let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  await sleep(250);
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
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
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    consoleErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
  }
  if (m.method === 'Runtime.exceptionThrown') {
    const d = m.params.exceptionDetails;
    consoleErrors.push(d.exception?.description || d.text);
  }
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
    const e = m.params.entry;
    if (!/favicon\.ico/.test(e.url || '')) consoleErrors.push(`${e.text} ${e.url || ''}`);
  }
};
function send(method, params = {}) {
  const id = ++msgId;
  return new Promise((res) => { pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
}
async function evaluate(expr, awaitPromise = false) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise });
  if (r.result?.exceptionDetails) {
    throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  }
  return r.result?.result?.value;
}

try {
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Log.enable');
  await send('Page.navigate', { url: `${BASE}/replay?mapcolors=1` });
  for (let i = 0; i < 120; i++) { await sleep(400); if (await evaluate('!!window.__v5').catch(() => false)) break; }
  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
  for (let i = 0; i < 240; i++) {
    if (await evaluate('!!(window.__v5.round && window.__v5.round.players.length)')) break;
    await sleep(400);
  }

  // 130 MB 的 OBJ + 9.4 MB 的旁车，headless 下要等一会儿；reason 落定就是结束。
  // 注意 `idle`（还没开始）和 `pending`（正在 fetch）都不算落定，否则会在贴色之前
  // 就判失败。
  console.log('\nB. 前端贴颜色');
  let stats = null;
  const settled = (s) => s && s.source === 'game'
    || (s?.reason && s.reason !== 'pending' && s.reason !== 'idle');
  for (let i = 0; i < 400; i++) {
    stats = await evaluate('window.__v5.mapColors').catch(() => null);
    if (settled(stats)) break;
    await sleep(500);
  }
  check('拿到了 mapColors', !!stats);
  check('map：和 demo 一致', stats?.map === MAP, `${stats?.map} vs ${MAP}`);
  check('source = game（不是水泥灰）', stats?.source === 'game', stats?.source || stats?.reason);
  check('面数 = 网格面数', stats?.triangles === stats?.expected,
    `${stats?.triangles} / ${stats?.expected}`);
  check('面数 = 服务端文件里写的', stats?.triangles === serverCount,
    `${stats?.triangles} vs ${serverCount}`);

  // ── C. 顶点色属性本身 ────────────────────────────────────────────────────
  console.log('\nC. color 属性');
  const attr = await evaluate(`(() => {
    let mesh = null;
    window.__v5.scene.traverse((o) => { if (o.isMesh && o.geometry?.attributes?.mmColor) mesh = o; });
    if (!mesh) return { error: 'no map mesh' };
    const pos = mesh.geometry.attributes.position;
    const col = mesh.geometry.attributes.color;
    const nor = mesh.geometry.attributes.normal;
    const faces = pos.count / 3;
    const stride = Math.max(1, Math.floor(faces / 200000));   // 采样 ~20 万个面
    const seen = new Set();
    let sampled = 0, flat = 0, tinted = 0, grey = 0;
    let nSampled = 0, nUnit = 0, nAligned = 0, nChanged = 0;
    for (let f = 0; f < faces; f += stride) {
      const a = f * 9;
      const r = col.array[a], g = col.array[a + 1], b = col.array[a + 2];
      const same = Math.abs(col.array[a + 3] - r) < 1e-6
        && Math.abs(col.array[a + 4] - g) < 1e-6
        && Math.abs(col.array[a + 5] - b) < 1e-6
        && Math.abs(col.array[a + 6] - r) < 1e-6
        && Math.abs(col.array[a + 7] - g) < 1e-6
        && Math.abs(col.array[a + 8] - b) < 1e-6;
      sampled++;
      if (same) flat++;
      const spread = Math.max(r, g, b) - Math.min(r, g, b);
      if (spread > 0.01) tinted++;
      if (spread < 0.002) grey++;
      seen.add((Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255));

      if (nor && nSampled < 50000) {
        // 三种几何点算出来的面法线，和属性里的法线比：应该是"接近但不一样"（平滑过）
        const i0 = f * 3, i1 = i0 + 1, i2 = i0 + 2;
        const ax = pos.array[i1 * 3] - pos.array[i0 * 3];
        const ay = pos.array[i1 * 3 + 1] - pos.array[i0 * 3 + 1];
        const az = pos.array[i1 * 3 + 2] - pos.array[i0 * 3 + 2];
        const bx = pos.array[i2 * 3] - pos.array[i0 * 3];
        const by = pos.array[i2 * 3 + 1] - pos.array[i0 * 3 + 1];
        const bz = pos.array[i2 * 3 + 2] - pos.array[i0 * 3 + 2];
        let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        const nl = Math.hypot(nx, ny, nz);
        if (nl > 1e-12) {
          nx /= nl; ny /= nl; nz /= nl;
          const ux = nor.array[a], uy = nor.array[a + 1], uz = nor.array[a + 2];
          const ul = Math.hypot(ux, uy, uz);
          nSampled++;
          if (Math.abs(ul - 1) < 0.02) nUnit++;
          const dot = Math.abs(ux * nx + uy * ny + uz * nz) / (ul || 1);
          // 烘焙里的折角保护是 cos 0.5（60°），所以理论上限就是 60°；这里放宽到
          // 0.45 只抓"明显翻转/算错"，不抓"某块薄片上平滑得比较多"
          if (dot > 0.45) nAligned++;
          if (dot < 0.9999) nChanged++;
        }
      }
    }
    return { faces, sampled, flat, tinted, grey, distinct: seen.size,
             vertexColors: !!mesh.material.vertexColors,
             nSampled, nUnit, nAligned, nChanged,
             hasNormals: !!nor };
  })()`);
  check('找到地图网格', !attr?.error, attr?.error || '');
  check('每面 3 个角同色（旁车是每三角形一个颜色）', attr && attr.flat === attr.sampled,
    `${attr?.flat}/${attr?.sampled}`);

  // 压过色板就要求"颜色数 ≤ 色板大小"；没压则要求足够多样（水泥灰只有一个色系）
  const pal = stats?.palette || 0;
  if (pal > 1) {
    check(`颜色数受调色板限制（${pal} 色）`, attr && attr.distinct <= pal && attr.distinct >= 3,
      `${attr?.distinct} 种`);
  } else {
    check('颜色足够多样（水泥灰只有一个色系）', attr && attr.distinct >= 500, `${attr?.distinct} 种`);
  }
  check('大部分面有明确色相（不是灰的）', attr && attr.tinted / attr.sampled > 0.3,
    `${attr ? (100 * attr.tinted / attr.sampled).toFixed(1) : '?'}% 有彩`);
  check('不是一片中性灰', attr && attr.grey / attr.sampled < 0.3,
    `${attr ? (100 * attr.grey / attr.sampled).toFixed(1) : '?'}% 纯灰`);

  // 平滑法线：单位长度、和几何法线同向（不翻转）、且确实被平滑过（不是逐面原值）
  if (stats?.normals) {
    check('法线是单位向量', attr && attr.nUnit / attr.nSampled > 0.99,
      `${attr ? (100 * attr.nUnit / attr.nSampled).toFixed(1) : '?'}%`);
    check('法线和几何面同向（没翻转）', attr && attr.nAligned / attr.nSampled > 0.95,
      `${attr ? (100 * attr.nAligned / attr.nSampled).toFixed(1) : '?'}%`);    check('法线确实被平滑过', attr && attr.nChanged / attr.nSampled > 0.2,
      `${attr ? (100 * attr.nChanged / attr.nSampled).toFixed(1) : '?'}% 与面法线不同`);
  }

  // ── D. 小地图 ────────────────────────────────────────────────────────────
  console.log('\nD. 小地图');
  const mini = await evaluate('(() => { const d = window.__v5.redrawMinimap(); return d ? d.length : 0; })()');
  check('redrawMinimap() 还能出图', typeof mini === 'number' && mini > 5000, `${mini} chars`);

  // ── E. 控制台 ────────────────────────────────────────────────────────────
  console.log('\nE. 控制台');
  check('没有报错', consoleErrors.length === 0, consoleErrors.slice(0, 2).join(' | '));
} finally {
  ws.close();
  proc.kill();
  await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* ignore */ }
  console.log('');
  if (failed) {
    console.log(`FAIL: map colours (${failed} check(s))`);
    process.exitCode = 1;
  } else {
    console.log('PASS: map colours');
  }
}
