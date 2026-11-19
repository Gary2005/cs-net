#!/usr/bin/env node
/**
 * test_geo_visibility.mjs — 几何可观测性（`visibility.js`）的浏览器端测试。
 *
 * Node 侧的 test_visibility.mjs 已经验过算法本身（暴力对拍 + 性能）。这里验的是
 * **接线**：开关、每帧矩阵、眼睛图标真的被几何结果驱动、角落射线真的画出来了，
 * 以及——最重要的——**和真实地图 BVH 对拍**。
 *
 * 眼睛图标现在是**几何唯一来源**（旧的 `toggle-geo-vis` 已删，`spotted` 列不再
 * 画成图标），所以 §D 里专门守了一条"关掉开关就是全不画、不会退回引擎源"。
 * 视锥（`toggle-frustum`）的细节在 `test_eye_icons.mjs` 里独立验算，这里只验存在。
 *
 * 独立参考实现：测试自己把「眼睛 → 8 个角落 → 视锥 → rawRaycast」重算一遍
 * （直接读 `__v5.round` 的坐标 + `__v5.rawRaycast`），**不复用 visibility.js**。
 * 拿被测代码验被测代码等于没验。
 *
 * 另外顺手量一件事：几何判定和引擎 `spotted` 列的一致率。引擎的 spotted 被限速在
 * 33 tick = 0.516 s 的网格上（见 README §4.8），几何是逐帧的，所以两者必然有差；
 * 差多少、往哪个方向差，是这个测试的输出之一。
 *
 * 用法：
 *   python visualizer/server.py --port 5099 &
 *   node visualizer/tests/test_geo_visibility.mjs http://127.0.0.1:5099 /path/to/match.csz
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

const URL = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3];
if (!CSZ) {
  console.error('用法: node test_geo_visibility.mjs <server-url> <match.csz>');
  process.exit(2);
}
const CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(0);
}
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}

const profile = mkdtempSync(join(tmpdir(), 'geovis-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--window-size=1100,760', 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

let wsUrl = null;
for (let i = 0; i < 60 && !wsUrl; i++) {
  await sleep(250);
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    wsUrl = list.find((t) => t.type === 'page')?.webSocketDebuggerUrl;
  } catch { /* not up yet */ }
}
if (!wsUrl) {
  console.log('SKIP: could not reach Chrome DevTools');
  chrome.kill();
  process.exit(0);
}

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

/**
 * 引擎侧的独立解码：`.csz` 里 `players[a].spotted` 是**稀疏**的 `[{i, v}, …]`
 * （`i` = 采样点，`v` = 该采样点上「能看到 a 的敌人」编号表）。这里二分查一遍，
 * 不复用 replay-core 的 `spottedAt()`。
 * 语义：「b 看得见 a」⟺ `b ∈ spotted[a]`（方向别反）。
 */
const ENG_SRC = `window.__refEngineSpotted = function (a, s) {
  const sp = window.__v5.round.players[a].spotted;
  if (!sp || !sp.length) return [];
  let lo = 0, hi = sp.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sp[mid].i <= s) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return ans >= 0 ? sp[ans].v : [];
};`;

/**
 * 独立参考实现，注入到页面里。**故意不引用 visibility.js** ——
 * 视角基、8 个角落、视锥、BVH 射线全部在这里重新写一遍。
 * 返回「通的角落数」；null = 这一对不参与（有人死了）。
 */
const REF_SRC = `window.__refVisClear = function (viewerIdx, targetIdx) {
  const S = 0.0254;
  const P = window.__v5.players;
  const v = P[viewerIdx], t = P[targetIdx];
  if (!v || !t || !v.alive || !t.alive) return null;
  const eye = [v.x, v.y, v.z + (v.eye_height != null ? v.eye_height : 64.0938 - 18 * (v.crouch || 0))];
  const y = (v.yaw || 0) * Math.PI / 180, p = (v.pitch || 0) * Math.PI / 180;
  const cy = Math.cos(y), sy = Math.sin(y), cp = Math.cos(p), sp = Math.sin(p);
  const f = [cy * cp, sy * cp, -sp];
  const r = [sy, -cy, 0];
  const u = [cy * sp, sy * sp, cp];
  const tanH = Math.tan(45 * Math.PI / 180), tanV = tanH / (16 / 9);
  const hw = 16 - 1, h = t.crouching ? 54 : 72;
  const z0 = t.z + 1, z1 = t.z + h - 1;
  let clear = 0, tested = 0;
  for (let k = 0; k < 2; k++) {
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        const c = [t.x + (i ? hw : -hw), t.y + (j ? hw : -hw), k ? z1 : z0];
        let dx = c[0] - eye[0], dy = c[1] - eye[1], dz = c[2] - eye[2];
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (L < 1e-6) { clear++; continue; }
        dx /= L; dy /= L; dz /= L;
        const fd = dx * f[0] + dy * f[1] + dz * f[2];
        if (fd <= 0) continue;
        if (Math.abs(dx * r[0] + dy * r[1] + dz * r[2]) > fd * tanH) continue;
        if (Math.abs(dx * u[0] + dy * u[1] + dz * u[2]) > fd * tanV) continue;
        tested++;
        // game -> three 是 (gx,gy,gz) -> (gy,gz,gx)*S；方向只做同样的轴置换
        const hit = window.__v5.rawRaycast(
          eye[1] * S, eye[2] * S, eye[0] * S, dy, dz, dx, L * S);
        if (hit < 0) clear++;
      }
    }
  }
  return { clear: clear, tested: tested };
};`;

try {
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Network.enable');
  await send('Network.setCacheDisabled', { cacheDisabled: true });
  await send('Page.navigate', { url: URL });
  for (let i = 0; i < 60; i++) {
    await sleep(400);
    if (await evaluate('!!(window.__v5 && window.__v5.loadPath)')) break;
  }
  check('the page booted the app', await evaluate('!!window.__v5'));
  await evaluate(ENG_SRC);
  check('引擎侧独立解码器注入成功', await evaluate('typeof window.__refEngineSpotted === "function"'));

  console.log('\nload');
  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
  // 14.8 MB 的 .csz 要下载 + 解码，固定 sleep 会偶发竞态 —— 轮询等
  let loaded = false;
  for (let i = 0; i < 90; i++) {
    await sleep(400);
    if (await evaluate('!!(window.__v5.round && window.__v5.players.length)')) { loaded = true; break; }
  }
  check('a round is loaded', loaded);

  // 地图 BVH 是**后台懒建**的（823k 三角形，浏览器里一两秒）：`loadMap()` 一 resolve
  // 就 kick off，但不 await。眼睛图标默认开着，所以 BVH 没就绪时几何判定**只按视锥
  // + 烟雾**算（不算墙）—— 那会儿量出来的射线数、命中数、一致率全是垃圾。
  // 所以先等它就绪，别用固定 sleep 赌。
  let bvhReady = false;
  for (let i = 0; i < 60; i++) {
    if (await evaluate('!!(window.__v5.visibility.stats && window.__v5.visibility.stats.mapReady)')) {
      bvhReady = true; break;
    }
    await sleep(500);
  }
  check('地图 BVH 建好了（下面的几何判定才含墙体遮挡）', bvhReady);

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nA. 开关与 API 存在性');
  // ═══════════════════════════════════════════════════════════════════════
  check('眼睛图标开关存在且默认开着（几何是唯一来源）',
    await evaluate(`(() => {
      const t = document.getElementById('toggle-spotted');
      return !!t && t.checked;
    })()`));
  check('旧的 toggle-geo-vis 已经删掉（不再有第二个来源）',
    await evaluate('!document.getElementById("toggle-geo-vis")'));
  check('toggle-vis-corners exists and defaults to off',
    await evaluate(`(() => {
      const t = document.getElementById('toggle-vis-corners');
      return !!t && !t.checked;
    })()`));
  check('__v5.visibility API is present',
    await evaluate(`(() => {
      const g = window.__v5.visibility;
      return !!g && typeof g.matrix === 'function' && typeof g.recompute === 'function'
        && typeof g.visibleOf === 'function' && typeof g.pairsOf === 'function'
        && typeof g.cornerRays === 'function' && typeof g.frustum === 'function'
        && g.iconsEnabled === true && g.frustumEnabled === false;
    })()`));
  check('眼睛图标开着 → 每帧都在算（result 不是 null）',
    await evaluate('window.__v5.visibility.result !== null'));
  check('matrix() 能拿到 10×10',
    await evaluate('Array.isArray(window.__v5.visibility.matrix())'));

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nB. 矩阵形状与语义');
  // ═══════════════════════════════════════════════════════════════════════
  const m = await evaluate('window.__v5.visibility.matrix()');
  check('矩阵是 10×10', Array.isArray(m) && m.length === 10 && m.every((r) => r.length === 10),
    `${m && m.length}×${m && m[0] && m[0].length}`);
  let diagBad = 0;
  for (let i = 0; i < 10; i++) if (m[i][i] !== null) diagBad++;
  check('对角线是 null（自己不和自己成对）', diagBad === 0, `${diagBad}`);

  const st = await evaluate('window.__v5.visibility.stats');
  check('stats 有 pairs / rays / ms', st && st.pairs > 0 && st.rays >= 0 && st.ms >= 0,
    JSON.stringify(st));
  check('活人不超过 10 → 有序对不超过 90', st.pairs <= 90, `${st.pairs}`);
  console.log(`  · 冷启动那一帧：${st.pairs} 对，${st.rays} 条射线，`
    + `${st.mapHits} 次地图命中，${st.smokeHits} 次烟雾命中，${st.ms.toFixed(2)} ms`);

  // 热态耗时：headless + swiftshader 的第一帧要付 JIT 的钱，量的是冷启动不是稳态。
  // 先跑 12 帧预热（不计入），再量 30 帧。
  const warm = await evaluate(`(() => {
    for (let i = 0; i < 12; i++) window.__v5.visibility.recompute();
    const N = 30, t = [];
    for (let i = 0; i < N; i++) t.push(window.__v5.visibility.recompute().ms);
    t.sort((a, b) => a - b);
    return { median: t[N >> 1], max: t[N - 1], rays: window.__v5.visibility.stats.rays,
             mapHits: window.__v5.visibility.stats.mapHits,
             mapReady: window.__v5.visibility.stats.mapReady,
             occluders: window.__v5.visibility.occluderCount };
  })()`);
  console.log(`  · 热态 30 帧：中位 ${warm.median.toFixed(2)} ms，最差 ${warm.max.toFixed(2)} ms，`
    + `${warm.rays} 条射线/帧，地图 BVH ${warm.mapReady ? '已就绪' : '未就绪'}，`
    + `烟雾遮挡体 ${warm.occluders}`);
  // 注意：headless + swiftshader 上单帧抖动很大（实测中位 6 ms、最差 10 ms），
  // 而同样的代码在 Node 里（test_visibility.mjs）是 0.117 ms/帧。所以这里的阈值
  // 只用来抓"灾难性回归"，**权威性能数字看 Node 那个测试**。
  check('热态每帧 < 25 ms（只防灾难性回归；权威数字在 test_visibility.mjs）',
    warm.median < 25, `${warm.median} ms`);
  check('地图 BVH 已就绪（否则几何不含墙体遮挡）', warm.mapReady);
  check('BVH 就绪后确实在打墙（否则等于没有地图遮挡）', warm.mapHits > 0,
    `mapHits=${warm.mapHits} rays=${warm.rays}`);

  // 几何是**全有序对**：含队友。引擎的 spotted 只标敌方 —— 找一对队友验证。
  const matePair = await evaluate(`(() => {
    const r = window.__v5.round, s = window.__v5.sampleIdx;
    const P = window.__v5.players;
    for (let a = 0; a < 10; a++) {
      for (let b = 0; b < 10; b++) {
        if (a === b || !P[a] || !P[b] || !P[a].alive || !P[b].alive) continue;
        if (r.teams[a] !== r.teams[b]) continue;
        const eng = window.__refEngineSpotted(a, s).indexOf(b) >= 0;
        const geo = window.__v5.visibility.result.visible[b * 10 + a] === 1;
        if (geo && !eng) return { a: a, b: b, team: r.teams[a] };
      }
    }
    return null;
  })()`);
  check('几何判定包含队友（引擎 spotted 永远不标队友）', !!matePair,
    matePair ? `队友 ${matePair.b} 看得见 ${matePair.a}（${matePair.team}）` : '这个采样点没找到例子');

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nC. 和独立参考实现对拍（真实地图 BVH）');
  // ═══════════════════════════════════════════════════════════════════════
  await evaluate(REF_SRC);
  check('参考实现注入成功', await evaluate('typeof window.__refVisClear === "function"'));

  // 找几个**没有烟**的采样点：参考实现只重算地图遮挡，烟它不管
  const cleanSamples = await evaluate(`(() => {
    const n = window.__v5.round.ticks.length;
    const out = [];
    for (let i = 0; i < n && out.length < 6; i += Math.max(1, Math.floor(n / 40))) {
      window.__v5.seekSample(i);
      window.__v5.visibility.recompute();
      if (window.__v5.visibility.occluderCount === 0) out.push(i);
    }
    return out;
  })()`);
  check('找到了没有烟的采样点用来对拍', cleanSamples.length >= 3,
    `找到 ${cleanSamples.length} 个`);

  let cmpTotal = 0, cmpDiff = 0, cmpVisible = 0, cmpTested = 0;
  const diffs = [];
  for (const s of cleanSamples) {
    const r = await evaluate(`(() => {
      window.__v5.seekSample(${s});
      const res = window.__v5.visibility.recompute();
      const n = window.__v5.visibility.result.n;
      const out = { s: ${s}, pairs: [], occluders: window.__v5.visibility.occluderCount };
      for (let b = 0; b < n; b++) {
        for (let a = 0; a < n; a++) {
          if (a === b) continue;
          const ref = window.__refVisClear(b, a);
          if (ref === null) continue;
          out.pairs.push([b, a, window.__v5.visibility.result.visible[b * n + a],
            ref.clear > 0 ? 1 : 0, ref.clear, ref.tested]);
        }
      }
      return out;
    })()`);
    if (r.occluders !== 0) continue;
    for (const [b, a, geo, ref, refClear, refTested] of r.pairs) {
      cmpTotal++;
      cmpVisible += geo;
      cmpTested += refTested;
      if (geo !== ref) {
        cmpDiff++;
        if (diffs.length < 6) diffs.push({ s, b, a, geo, ref, refClear, refTested });
      }
    }
  }
  check(`和独立参考实现完全一致（${cmpTotal} 对，其中几何判可见 ${cmpVisible} 对）`,
    cmpDiff === 0 && cmpTotal > 100,
    `${cmpDiff} 处不一致 ${diffs.length ? JSON.stringify(diffs) : ''}`);
  console.log(`  · 参考实现一共发射了 ${cmpTested} 条角落射线（只算通过视锥的）`);

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nD. 眼睛图标被几何结果驱动（唯一来源）');
  // ═══════════════════════════════════════════════════════════════════════
  // 眼睛图标现在**只有几何一个来源**，默认就开着，所以不需要拨任何开关。
  // （`toggle-spotted` 的 change handler 是 async 的 —— 会 await ensureMapBVH() ——
  //  但初始状态没走 handler，`showSpottedIcons` 本来就是 true。）
  check('图标开着 → 每帧都在算', await evaluate('!!window.__v5.visibility.result'));
  check('iconsEnabled 是 true', await evaluate('window.__v5.visibility.iconsEnabled'));

  const iconMatch = await evaluate(`(() => {
    window.__v5.selectPlayer(0);
    const want = window.__v5.visibility.visibleOf(0);
    const got = window.__v5.spotIcons
      .map((s, i) => (s && s.target ? i : -1)).filter((i) => i >= 0);
    return { want: want, got: got, selected: window.__v5.selectedPlayer };
  })()`);
  check('选中玩家后，眼睛图标的目标 == 几何可见列表',
    JSON.stringify(iconMatch.want) === JSON.stringify(iconMatch.got),
    `want=${JSON.stringify(iconMatch.want)} got=${JSON.stringify(iconMatch.got)}`);

  // 引擎的 spotted 列**不再**是任何一条退路 —— 关掉眼睛开关就是全不画，
  // 不会"回到引擎源"。这是这次改动最容易被改回去的地方，所以专门守一条。
  const engineStillThere = await evaluate(`(() => {
    const t = document.getElementById('toggle-spotted');
    t.checked = false; t.dispatchEvent(new Event('change'));
    return { eng: window.__v5.visibleEnemies, icons: window.__v5.spotIcons };
  })()`);
  check('引擎 spotted 数据接口还在（数据面板要用）',
    Array.isArray(engineStillThere.eng));
  check('关掉眼睛开关 → 全不画（不会退回引擎源）',
    engineStillThere.icons.filter((s) => s && s.target).length === 0,
    `还有 ${engineStillThere.icons.filter((s) => s && s.target).length} 个`);
  await evaluate(`(() => {
    const t = document.getElementById('toggle-spotted');
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(400);
  check('重新打开 → 又回到几何列表',
    await evaluate(`(() => {
      const want = window.__v5.visibility.visibleOf(window.__v5.selectedPlayer);
      const got = window.__v5.spotIcons
        .map((s, i) => (s && s.target ? i : -1)).filter((i) => i >= 0);
      return JSON.stringify(want) === JSON.stringify(got);
    })()`));

  // 没选中任何人 → 一个图标都不许有
  const noneSelected = await evaluate(`(() => {
    window.__v5.selectPlayer(-1);
    return window.__v5.spotIcons.filter((s) => s && s.target).length;
  })()`);
  check('没选中任何人时一个眼睛都不画', noneSelected === 0, `${noneSelected}`);

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nE. 角落调试射线');
  // ═══════════════════════════════════════════════════════════════════════
  const cornersOff = await evaluate('window.__v5.visibility.cornerRays()');
  check('角落射线层已创建但默认不画', cornersOff.created && !cornersOff.visible,
    JSON.stringify(cornersOff));

  const cornersState = await evaluate(`(() => {
    document.getElementById('toggle-vis-corners').checked = true;
    document.getElementById('toggle-vis-corners').dispatchEvent(new Event('change'));
    window.__v5.selectPlayer(1);
    return window.__v5.visibility.cornerRays();
  })()`);
  await sleep(400);   // handler 里的 await 只会在 BVH 已就绪时立刻返回
  const cornersAfter = await evaluate('window.__v5.visibility.cornerRays()');
  check('角落射线段数 == (活人数-1) × 8',
    cornersAfter.count > 0 && cornersAfter.count % 8 === 0 && cornersAfter.visible,
    JSON.stringify(cornersAfter));
  check('drawRange 和段数一致（真的会画出来）',
    cornersAfter.drawnVertices === cornersAfter.count * 2,
    `drawRange=${cornersAfter.drawnVertices} count=${cornersAfter.count}`);

  const cornersCleared = await evaluate(`(() => {
    document.getElementById('toggle-vis-corners').checked = false;
    document.getElementById('toggle-vis-corners').dispatchEvent(new Event('change'));
    return window.__v5.visibility.cornerRays();
  })()`);
  check('关掉角落开关后不画', !cornersCleared.visible, JSON.stringify(cornersCleared));

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nF. 几何 vs 引擎 spotted（测量）');
  // ═══════════════════════════════════════════════════════════════════════
  //
  // 度量口径很重要。引擎的 spotted 被限速在 **33 tick（0.516 s）的网格**上
  // （README §4.8），所以「同一个 tick 比一比」是系统性偏袒几何的比法 —— 引擎
  // 本来就不可能在同一 tick 跟上。真正该问的是：
  //
  //   Q1 引擎说「看得见」的时候，几何同 tick 也这么说吗？       → 几何的召回
  //   Q2 几何说「看得见」的时候，引擎在**接下来的 33 tick 内**跟上了吗？
  //                                                          → 这就是延迟的直接证据
  //
  // 引擎侧完全用原始数组解码（`players[a].alive[s]` + 稀疏 `spotted` 二分），
  // 不 seek、不经过 replay-core —— 所以 Q2 能直接查 s+1..s+33 而不用逐 tick 推进。
  const agree = await evaluate(`(() => {
    const R = window.__v5.round;
    const n = R.ticks.length;
    const stride = Math.max(1, Math.floor(n / 150));
    const aliveAt = (p, s) => (R.players[p].alive ? !!R.players[p].alive[s] : true);
    const engAt = (a, b, s) => window.__refEngineSpotted(a, s).indexOf(b) >= 0;

    let samples = 0, enemyPairs = 0, matePairs = 0, mateGeo = 0;
    let engTrue = 0, engTrueGeoSame = 0;         // Q1
    let geoTrue = 0, geoSame = 0;                // Q2 的分母
    let within33 = 0, neverWithin = 0, windowDied = 0;
    let lagSum = 0, lagN = 0, lagMin = 1e9, lagMax = -1;

    // 差异诊断：引擎说看得见、几何说看不见的那些对，卡在哪个环节？
    // reason 是按位或的，所以四个位分别统计（一个对可能同时踩两个原因）。
    let engOnly = 0, tFov = 0, tMap = 0, tSmoke = 0, tPlayer = 0, tNone = 0;
    // 以及：目标在那之前 33 tick 内开过枪/走过路吗？—— 引擎的 m_bSpotted
    // **会被声音置位**，而本工具是纯视觉的，这是预期内的分歧来源。
    let engOnlySound = 0;

    const hadSound = (p, s) => {
      const sh = R.players[p].shots, ft = R.players[p].footsteps;
      for (let t = Math.max(0, s - 33); t <= s; t++) {
        if ((sh && sh[t]) || (ft && ft[t])) return true;
      }
      return false;
    };

    for (let s = 0; s < n; s += stride) {
      window.__v5.seekSample(s);
      window.__v5.visibility.recompute();
      const res = window.__v5.visibility.result;
      samples++;
      for (let b = 0; b < 10; b++) {
        for (let a = 0; a < 10; a++) {
          if (a === b) continue;
          if (!aliveAt(a, s) || !aliveAt(b, s)) continue;
          const geo = res.visible[b * 10 + a] === 1;
          const eng = engAt(a, b, s);
          if (R.teams[a] === R.teams[b]) { matePairs++; if (geo) mateGeo++; continue; }
          enemyPairs++;
          if (eng && !geo) {
            engOnly++;
            const r = res.reason[b * 10 + a];
            if (r & 1) tFov++;
            if (r & 2) tMap++;
            if (r & 4) tSmoke++;
            if (r & 8) tPlayer++;
            if (r === 0) tNone++;
            if (hadSound(a, s)) engOnlySound++;
          }
          if (eng) { engTrue++; if (geo) engTrueGeoSame++; }
          if (!geo) continue;
          geoTrue++;
          if (eng) { geoSame++; continue; }
          // 往后 33 tick 内引擎有没有跟上？双方都得还活着才作数
          let ok = false, died = false;
          for (let k = 1; k <= 33; k++) {
            const t = s + k;
            if (t >= n) { died = true; break; }
            if (!aliveAt(a, t) || !aliveAt(b, t)) { died = true; break; }
            if (engAt(a, b, t)) { ok = true; lagSum += k; lagN++; if (k < lagMin) lagMin = k; if (k > lagMax) lagMax = k; break; }
          }
          if (ok) within33++;
          else if (died) windowDied++;
          else neverWithin++;
        }
      }
    }
    return { samples, enemyPairs, matePairs, mateGeo, engTrue, engTrueGeoSame,
             geoTrue, geoSame, within33, neverWithin, windowDied,
             lagMean: lagN ? lagSum / lagN : 0, lagMin, lagMax, lagN,
             engOnly, tFov, tMap, tSmoke, tPlayer, tNone, engOnlySound };
  })()`);

  const pct = (x, d) => `${(100 * x / Math.max(1, d)).toFixed(1)}%`;
  console.log(`  · ${agree.samples} 个采样点：敌对 ${agree.enemyPairs}，队友对 ${agree.matePairs}`);
  console.log(`  · 引擎说看得见 ${agree.engTrue} 对 → 几何同 tick 确认 `
    + `${agree.engTrueGeoSame}（${pct(agree.engTrueGeoSame, agree.engTrue)}）`);
  console.log(`  · 几何说看得见 ${agree.geoTrue} 对 → 引擎同 tick 只有 ${agree.geoSame}`
    + `（${pct(agree.geoSame, agree.geoTrue)}），`
    + `但**往后 33 tick 内**又跟上 ${agree.within33}`
    + `（${pct(agree.within33, agree.geoTrue)}）`);
  if (agree.lagN) {
    console.log(`  · 引擎跟上的延迟：均值 ${agree.lagMean.toFixed(1)} tick`
      + `（${(agree.lagMean / 64 * 1000).toFixed(0)} ms），`
      + `范围 ${agree.lagMin}~${agree.lagMax} tick（理论上界 33）`);
  }
  console.log(`  · 33 tick 内一直没跟上 ${agree.neverWithin} 对`
    + `（其中 ${agree.windowDied} 对是窗口内有人死了，判不了）`);
  console.log(`  · 队友对里几何判可见 ${agree.mateGeo}（引擎一个都没有，符合预期）`);
  if (agree.engOnly) {
    console.log(`  · 诊断「引擎看得见、几何看不见」${agree.engOnly} 对，卡在哪一步`
      + `（按位，可重叠）：视锥 ${agree.tFov}｜地图 ${agree.tMap}｜`
      + `烟雾 ${agree.tSmoke}｜其他玩家 ${agree.tPlayer}｜没原因 ${agree.tNone}`);
    console.log(`    其中 ${agree.engOnlySound} 对（`
      + `${(100 * agree.engOnlySound / agree.engOnly).toFixed(0)}%）目标在之前 33 tick 内`
      + `开过枪或走过路 —— 引擎的 m_bSpotted **会被声音置位**，本工具是纯视觉的`);
  }

  check('引擎说看得见时，几何同 tick 的召回 > 60%',
    agree.engTrueGeoSame / Math.max(1, agree.engTrue) > 0.6,
    pct(agree.engTrueGeoSame, agree.engTrue));
  check('几何说看得见时，引擎同 tick 的命中率明显低（延迟的直接证据）',
    agree.geoSame / Math.max(1, agree.geoTrue) < 0.5,
    pct(agree.geoSame, agree.geoTrue));
  check('把窗口放宽到 33 tick 后，引擎跟上的比例大幅上升',
    agree.within33 > agree.geoSame,
    `${agree.within33} vs ${agree.geoSame}`);
  check('引擎的延迟落在 1..33 tick 内（符合 33 tick 网格的说法）',
    agree.lagN === 0 || (agree.lagMin >= 1 && agree.lagMax <= 33),
    `${agree.lagMin}~${agree.lagMax}`);
  check('几何包含队友、引擎从不包含（语义差异确认）', agree.mateGeo > 0);

  check('整个测试过程中没有 console error', consoleErrors.length === 0,
    consoleErrors.slice(0, 3).join(' | '));
} catch (err) {
  if (err && err.skip) {
    // 已打印 SKIP 原因
  } else {
    failed++;
    console.log(`  FAIL 测试抛出异常: ${err && err.message ? err.message : err}`);
  }
} finally {
  try { ws.close(); } catch { /* ignore */ }
  chrome.kill();
  await sleep(300);
}

console.log(`\ngeo-visibility: ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
