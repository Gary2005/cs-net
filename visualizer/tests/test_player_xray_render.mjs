#!/usr/bin/env node
/**
 * test_player_xray_render.mjs — `选手 X 光` 的浏览器端测试（接线 + 真的换材质）。
 *
 * Node 侧的 test_player_xray.mjs 已经验过判定算法本身（暴力对拍 + 性能）。
 * 这里验的是**接线**：
 *
 *   A 开关与 API         `#toggle-player-xray` 默认**开**、`__v5.playerXray` 在
 *   B 独立参考           用 `__v5.rawRaycast` 自己把那 3 条采样射线重算一遍，
 *                        **不复用 player-xray.js**，逐人核对 flags[i]
 *   C 两种外观同时存在    被挡住的那几个：0.42 / 不测深度 / renderOrder 980；
 *                        直接看得见的那几个：不透明 / 测深度。
 *                        顺带证明"幽灵没有改到共享材质"（腿是 10 人共用的）
 *   D 和透视地图不打架    开透视地图 → 幽灵让位；关掉 → 幽灵按判定回来，
 *                        而且**还原的是正常材质**（`_origMat` 没记成幽灵）
 *   E 关掉开关           全员不透明，再打开能回来
 *   F 幽灵的**颜色**      逐部件读画出来的颜色 / 自发光，必须跟这一回合的
 *                        `teams[]` 一致。挑一个"分队和 `i < 5` 猜的底色相反"的
 *                        采样点，所以它抓得到"蓝 CT 的幽灵发黄光、黄 T 的幽灵
 *                        发蓝光"那个 bug（幽灵的颜色曾经是克隆那一刻的快照）
 *
 * 用法：
 *   python visualizer/server.py --port 5099 &
 *   node visualizer/tests/test_player_xray_render.mjs http://127.0.0.1:5099 /path/to/match.csz
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
  console.error('用法: node test_player_xray_render.mjs <server-url> <match.csz>');
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

const profile = mkdtempSync(join(tmpdir(), 'pxray-'));
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
 * 独立参考实现，注入页面。**故意不引用 player-xray.js**：
 * 采样分数、身高式子、1 HU 容差全部在这里重写一遍，遮挡只问 `__v5.rawRaycast`
 * （真实地图 BVH）。返回 `[i] = 1` 表示第 i 个人被挡住。
 *
 * 判定用的摄像机位置从 `__v5.playerXray.cameraGame` 读 —— 也就是被测代码
 * **自己报告的那个**。这样相机在动也不影响：两边量的是同一个点。
 */
const REF_SRC = `window.__refPlayerXRay = function () {
  const g = window.__v5.playerXray;
  const cam = g.cameraGame;
  const P = window.__v5.players;
  const S = 0.0254;
  const FR = [0.88, 0.5, 0.16];
  const EPS = 1;
  const out = [];
  for (let i = 0; i < P.length; i++) {
    const s = P[i];
    if (!s || !s.alive) { out.push(0); continue; }
    const crouch = Math.min(1, Math.max(0, s.crouch || 0));
    const h = 72 - 18 * crouch;
    let clear = false;
    for (let k = 0; k < FR.length; k++) {
      const pz = s.z + h * FR[k];
      let dx = s.x - cam.x, dy = s.y - cam.y, dz = pz - cam.z;
      const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (L <= EPS) { clear = true; break; }
      dx /= L; dy /= L; dz /= L;
      const maxD = L - EPS;
      // game -> three：(gx,gy,gz) -> (gy,gz,gx)*S；方向只做同样的轴置换
      const hit = window.__v5.rawRaycast(
        cam.y * S, cam.z * S, cam.x * S, dy, dz, dx, maxD * S);
      if (hit < 0) { clear = true; break; }
    }
    out.push(clear ? 0 : 1);
  }
  return out;
};`;

/** 读一次完整状态：判定、材质、（开关状态）。 */
const SNAP = `(() => {
  const g = window.__v5.playerXray;
  return {
    enabled: g.enabled,
    stats: g.stats,
    flags: g.flags,
    ghosts: g.ghosts,
    players: window.__v5.players.map((p, i) => (p ? {
      alive: !!p.alive, crouch: p.crouch || 0, crouching: !!p.crouching,
    } : null)),
    checked: document.getElementById('toggle-player-xray').checked,
    mapXray: document.getElementById('toggle-xray').checked,
  };
})()`;

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
  await evaluate(REF_SRC);
  check('独立参考实现注入成功',
    await evaluate('typeof window.__refPlayerXRay === "function"'));

  console.log('\nload');
  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
  let loaded = false;
  for (let i = 0; i < 90; i++) {
    await sleep(400);
    if (await evaluate('!!(window.__v5.round && window.__v5.players.length)')) { loaded = true; break; }
  }
  check('a round is loaded', loaded);

  // 地图 BVH 后台懒建（823k 三角形）。没建好时"墙不算遮挡"，谁都不会是幽灵 ——
  // 那样 C 段一条都测不出来，所以先等它就绪，别用固定 sleep 赌。
  let bvhReady = false;
  for (let i = 0; i < 60; i++) {
    if (await evaluate('!!(window.__v5.playerXray.stats && window.__v5.playerXray.stats.mapReady)')) {
      bvhReady = true; break;
    }
    await sleep(500);
  }
  check('地图 BVH 建好了（判定才含墙体遮挡）', bvhReady);

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nA. 开关与 API');
  // ═══════════════════════════════════════════════════════════════════════
  check('选手 X 光开关存在且**默认开着**',
    await evaluate(`(() => {
      const t = document.getElementById('toggle-player-xray');
      return !!t && t.checked;
    })()`));
  check('__v5.playerXray API is present',
    await evaluate(`(() => {
      const g = window.__v5.playerXray;
      return !!g && g.enabled === true && typeof g.recompute === 'function'
        && Array.isArray(g.flags) && Array.isArray(g.ghosts)
        && !!g.cameraGame && !!g.stats;
    })()`));
  check('每帧都在算（stats 有 players / rays / ms）',
    await evaluate(`(() => {
      const s = window.__v5.playerXray.stats;
      return !!s && s.players > 0 && s.rays >= 0 && typeof s.ms === 'number';
    })()`));

  const st0 = await evaluate('window.__v5.playerXray.stats');
  console.log(`  · 冷启动那一帧：${st0.players} 个活人，${st0.rays} 条射线，`
    + `${st0.blocked} 人被挡住（地图命中 ${st0.mapHits}，烟命中 ${st0.smokeHits}），`
    + `${st0.ms.toFixed(2)} ms`);

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nB. 独立参考：3 条采样射线自己重算一遍');
  // ═══════════════════════════════════════════════════════════════════════
  {
    // 多找几个采样点（回合不同位置），把"有些人被挡、有些人没被挡"的场景撞出来
    const scan = await evaluate(`(() => {
      const out = [];
      const n = window.__v5.round.ticks.length;
      for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 12))) {
        window.__v5.seekSample(i);
        const g = window.__v5.playerXray;
        const ref = window.__refPlayerXRay();
        let diff = 0, blocked = 0, clear = 0;
        for (let k = 0; k < ref.length; k++) {
          if (ref[k] !== (g.flags[k] || 0)) diff++;
          if (g.flags[k] === 1) blocked++; else clear++;
        }
        out.push({ sample: i, diff, blocked, clear, flags: g.flags });
      }
      return out;
    })()`);

    const totalDiff = scan.reduce((a, s) => a + s.diff, 0);
    const totalBlocked = scan.reduce((a, s) => a + s.blocked, 0);
    const totalClear = scan.reduce((a, s) => a + s.clear, 0);
    check(`12 个采样点 × 10 人，独立参考和 flags 逐条一致（${scan.length * 10} 个判定，`
      + `不一致 ${totalDiff}）`, totalDiff === 0, JSON.stringify(scan.filter((s) => s.diff)));
    check('样本里"直接看得见"和"只能透视"都出现过',
      totalBlocked > 0 && totalClear > 0,
      `blocked=${totalBlocked} clear=${totalClear}`);
    console.log(`  · ${totalBlocked} 人次被判成幽灵，${totalClear} 人次直接可见`);
  }

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nC. 两种外观：真的换了材质');
  // ═══════════════════════════════════════════════════════════════════════
  let picked = null;
  {
    // 找一个同时存在"幽灵"和"直接可见"的采样点停留
    picked = await evaluate(`(() => {
      const n = window.__v5.round.ticks.length;
      for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 24))) {
        window.__v5.seekSample(i);
        const g = window.__v5.playerXray;
        const alive = window.__v5.players.filter((p) => p && p.alive).length;
        let blocked = 0;
        for (const f of g.flags) if (f === 1) blocked++;
        if (blocked > 0 && blocked < alive) return { sample: i, blocked, alive };
      }
      return null;
    })()`);
    check('找到一个"有人被挡、有人没被挡"的采样点', !!picked, JSON.stringify(picked));

    if (picked) {
      const snap = await evaluate(SNAP);
      const figParts = snap.ghosts.map((g) => g.parts);
      check('每个人身上 9 个部件全部纳管（躯干/双臂/双腿/头/盔/面罩/视线锥）',
        figParts.every((n) => n === 9), JSON.stringify(figParts));

      const ghostIdx = [], clearIdx = [];
      for (let i = 0; i < snap.ghosts.length; i++) {
        if (!snap.players[i] || !snap.players[i].alive) continue;
        if (snap.ghosts[i].ghosted) ghostIdx.push(i); else clearIdx.push(i);
      }
      check('ghosted 和 flags 对得上（幽灵 = flag 1）',
        ghostIdx.every((i) => snap.flags[i] === 1)
        && clearIdx.every((i) => snap.flags[i] === 0),
        JSON.stringify({ ghostIdx, clearIdx, flags: snap.flags }));

      const g0 = snap.ghosts[ghostIdx[0]];
      check('幽灵：不透明度 0.42', Math.abs(g0.opacity - 0.42) < 1e-6, `${g0.opacity}`);
      check('幽灵：不测深度（这就是"透视"）', g0.depthTest === false);
      check('幽灵：renderOrder 980（排在不透明物之后、透视地图 999 之前）',
        g0.renderOrder === 980, `${g0.renderOrder}`);
      check('幽灵：9 个部件**全都**是幽灵（不是只换了躯干）',
        g0.ghostParts === 9, `${g0.ghostParts}/9`);

      const c0 = snap.ghosts[clearIdx[0]];
      check('直接看得见：不透明（opacity 1）', c0.opacity === 1, `${c0.opacity}`);
      check('直接看得见：照旧测深度', c0.depthTest === true);
      check('直接看得见：renderOrder 0', c0.renderOrder === 0, `${c0.renderOrder}`);
      check('直接看得见：一个幽灵部件都没有', c0.ghostParts === 0, `${c0.ghostParts}`);

      // 腿 / 头 / 面罩 / 视线锥是 10 个人**共享**材质：幽灵如果就地改，
      // 直接可见的那个人的腿也会变半透明。逐部件列出来就是为了盯这一条。
      //
      // ⚠️ 视线锥（最后一个部件）本来就不是不透明的 —— `matDir` 是 0.75 的锥体，
      // 所以这里只能断言"没有一个是幽灵的 0.42"，不能断言"全等于 1"。
      check('共享材质没被改：直接可见的那个人一个 0.42 的部件都没有',
        c0.opacities.every((o) => Math.abs(o - 0.42) > 1e-6), JSON.stringify(c0.opacities));
      check('共享材质没被改：躯干/双臂/双腿/头/盔/面罩仍然不透明',
        c0.opacities.slice(0, 8).every((o) => o === 1), JSON.stringify(c0.opacities));
      check('幽灵的每个部件都是半透明的（视线锥也一起变成 0.42）',
        g0.opacities.every((o) => Math.abs(o - 0.42) < 1e-6), JSON.stringify(g0.opacities));

      console.log(`  · 采样点 ${picked.sample}：${ghostIdx.length} 个幽灵 `
        + `[${ghostIdx}]，${clearIdx.length} 个直接可见 [${clearIdx}]`);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nD. 和「透视地图」不打架');
  // ═══════════════════════════════════════════════════════════════════════
  {
    // 回到 §C 挑出来的那个采样点：这一段的重点就是"有幽灵的时候去开透视地图"，
    // 一个幽灵都没有的话它测的是空气。
    if (picked) await evaluate(`window.__v5.seekSample(${picked.sample})`);
    const before = await evaluate(SNAP);
    const ghostsBefore = before.ghosts.filter((g) => g.ghosted).length;
    check('这一段是从"有幽灵"的状态开始的（否则下面几条是空转）', ghostsBefore > 0,
      `${ghostsBefore} 个幽灵`);

    // 开透视地图：地图变半透明 + 所有小人变"不测深度 + 自发光"，
    // 这是**另一套外观**，幽灵材质这时候没有意义，必须让位。
    await evaluate(`(() => {
      const t = document.getElementById('toggle-xray');
      t.checked = true; t.dispatchEvent(new Event('change'));
    })()`);
    await sleep(200);
    const on = await evaluate(SNAP);
    check('开着透视地图时，没人挂着幽灵材质',
      on.ghosts.every((g) => g.ghosted === false && g.ghostParts === 0),
      JSON.stringify(on.ghosts.map((g) => g.ghostParts)));
    check('开着透视地图时，全员都是"不测深度 + renderOrder 999"',
      on.ghosts.every((g) => g.depthTest === false && g.renderOrder === 999),
      JSON.stringify(on.ghosts.map((g) => [g.depthTest, g.renderOrder])));
    check('开着透视地图时判定照旧在算（和独立参考一致）',
      await evaluate(`(() => {
        const ref = window.__refPlayerXRay();
        const f = window.__v5.playerXray.flags;
        for (let i = 0; i < ref.length; i++) if (ref[i] !== (f[i] || 0)) return false;
        return true;
      })()`));

    // 关掉：这里是最容易出 bug 的一步 —— `_origMat` 如果在开的时候记成了
    // 幽灵材质，或者那趟遍历把幽灵材质的 depthTest 改成了 true，关掉之后
    // 幽灵就再也画不到墙上面去了（而且不会有任何报错）。
    await evaluate(`(() => {
      const t = document.getElementById('toggle-xray');
      t.checked = false; t.dispatchEvent(new Event('change'));
    })()`);
    await sleep(200);
    const off = await evaluate(SNAP);
    check('关掉透视地图后，幽灵按判定回来（人数一致）',
      off.ghosts.filter((g) => g.ghosted).length === ghostsBefore,
      `${off.ghosts.filter((g) => g.ghosted).length} vs ${ghostsBefore}`);
    check('关掉透视地图后，直接可见的人**恢复成正常材质**（不透明 + 测深度）',
      off.ghosts.every((g) => g.ghosted || (g.opacity === 1 && g.depthTest === true)),
      JSON.stringify(off.ghosts.map((g) => [g.ghosted, g.opacity, g.depthTest])));
    check('关掉透视地图后，幽灵仍然是半透明 + 不测深度（没被遍历改坏）',
      off.ghosts.every((g) => !g.ghosted
        || (Math.abs(g.opacity - 0.42) < 1e-6 && g.depthTest === false)),
      JSON.stringify(off.ghosts.map((g) => [g.ghosted, g.opacity, g.depthTest])));
  }

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nE. 关掉开关');
  // ═══════════════════════════════════════════════════════════════════════
  {
    if (picked) await evaluate(`window.__v5.seekSample(${picked.sample})`);
    const on = await evaluate(SNAP);
    const ghostsOn = on.ghosts.filter((g) => g.ghosted).length;

    await evaluate(`(() => {
      const t = document.getElementById('toggle-player-xray');
      t.checked = false; t.dispatchEvent(new Event('change'));
    })()`);
    await sleep(200);
    const off = await evaluate(SNAP);
    check('关掉后 enabled = false', off.enabled === false);
    check('关掉后一个人都不是幽灵', off.ghosts.every((g) => !g.ghosted && g.ghostParts === 0),
      JSON.stringify(off.ghosts.map((g) => g.ghostParts)));
    check('关掉后全员不透明 + 测深度',
      off.ghosts.every((g) => g.opacity === 1 && g.depthTest === true),
      JSON.stringify(off.ghosts.map((g) => [g.opacity, g.depthTest])));

    await evaluate(`(() => {
      const t = document.getElementById('toggle-player-xray');
      t.checked = true; t.dispatchEvent(new Event('change'));
    })()`);
    await sleep(200);
    const back = await evaluate(SNAP);
    check('再打开：enabled = true', back.enabled === true);
    check('再打开：幽灵回来了（人数和关之前一致）',
      back.ghosts.filter((g) => g.ghosted).length === ghostsOn,
      `${back.ghosts.filter((g) => g.ghosted).length} vs ${ghostsOn}`);
    check('再打开：幽灵材质是对的（0.42 / 不测深度 / 980）',
      back.ghosts.every((g) => !g.ghosted
        || (Math.abs(g.opacity - 0.42) < 1e-6 && g.depthTest === false && g.renderOrder === 980)),
      JSON.stringify(back.ghosts.map((g) => [g.ghosted, g.opacity, g.depthTest, g.renderOrder])));
  }

  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nF. 幽灵的颜色跟着 teams[] 走（不是按 i < 5 猜的底色）');
  // ═══════════════════════════════════════════════════════════════════════
  {
    // 只有"选手 X 光开 + 透视地图关"这一种组合下幽灵材质才在画面上，先把开关摆好。
    await evaluate(`(() => {
      const x = document.getElementById('toggle-xray');
      if (x.checked) { x.checked = false; x.dispatchEvent(new Event('change')); }
      const p = document.getElementById('toggle-player-xray');
      if (!p.checked) { p.checked = true; p.dispatchEvent(new Event('change')); }
    })()`);
    await sleep(200);

    // 挑采样点：要有幽灵，而且最好有"分队和 `i < 5` 猜的底色**相反**"的幽灵。
    // 相反才有测试价值 —— `.csz` 的分队是 `['T','CT','T','CT',…]` 这种交错，
    // 猜错时幽灵会整体发错色的光（0.5 强度的自发光占了画面的一大半）。
    const spot = await evaluate(`(() => {
      const teams = window.__v5.round.teams || [];
      const n = window.__v5.round.ticks.length;
      let fallback = null;
      for (let s = 0; s < n; s += Math.max(1, Math.floor(n / 16))) {
        window.__v5.seekSample(s);
        const flags = window.__v5.playerXray.flags;
        const ghosts = [], wrongGuess = [];
        for (let i = 0; i < flags.length; i++) {
          if (flags[i] !== 1) continue;
          ghosts.push(i);
          if (teams[i] !== (i < 5 ? 'CT' : 'T')) wrongGuess.push(i);
        }
        if (ghosts.length && wrongGuess.length) {
          return { sample: s, ghosts, wrongGuess, teams };
        }
        if (!fallback && ghosts.length) fallback = { sample: s, ghosts, wrongGuess, teams };
      }
      return fallback ? Object.assign(fallback, { noMismatch: true }) : null;
    })()`);
    check('找到一个"有幽灵"的采样点', !!spot, JSON.stringify(spot));

    if (spot) {
      check('这个采样点里有幽灵的分队和 `i < 5` 猜的底色相反（否则这一段是空转）',
        !spot.noMismatch && spot.wrongGuess.length > 0,
        `wrongGuess=${JSON.stringify(spot.wrongGuess)} teams=${JSON.stringify(spot.teams)}`);
      console.log(`  · 采样点 ${spot.sample}：幽灵 [${spot.ghosts}]，`
        + `其中"猜错底色"的 [${spot.wrongGuess}]`);

      const ghostSnap = await evaluate(SNAP);

      // 基准：把 `选手 X 光` 关掉，同一批人身上的**正常材质**就是"幽灵该长什么样"的
      // 标准答案（队色每帧写在这份材质上）。这样比对完全不依赖"哪几个部件是队服"，
      // 也不依赖部件顺序 —— 幽灵是它的克隆，颜色就必须和它逐部件一致。
      await evaluate(`(() => {
        const t = document.getElementById('toggle-player-xray');
        t.checked = false; t.dispatchEvent(new Event('change'));
      })()`);
      await sleep(200);
      await evaluate(`window.__v5.seekSample(${spot.sample})`);
      await sleep(120);
      const plainSnap = await evaluate(SNAP);

      // 再打开，幽灵要一模一样地回来（顺带证明这段没把状态留在关掉的那一边）
      await evaluate(`(() => {
        const t = document.getElementById('toggle-player-xray');
        t.checked = true; t.dispatchEvent(new Event('change'));
      })()`);
      await sleep(200);
      await evaluate(`window.__v5.seekSample(${spot.sample})`);
      await sleep(120);
      const backSnap = await evaluate(SNAP);

      const COL = { CT: 0x5b9bd5, T: 0xe8a240 };
      // 逐通道比较：`Color.getHex()` 会走一趟 sRGB ↔ 线性，留 2 的容差。
      const near = (a, b, tol = 2) => {
        if (a == null || b == null) return false;
        for (let k = 0; k < 3; k++) {
          const da = (a >> (k * 8)) & 0xff, db = (b >> (k * 8)) & 0xff;
          if (Math.abs(da - db) > tol) return false;
        }
        return true;
      };
      const hex = (v) => '#' + (v >>> 0).toString(16).padStart(6, '0');
      const at = (snap, i) => ((snap.ghosts[i] || {}).colors || []);
      const emAt = (snap, i) => ((snap.ghosts[i] || {}).emissives || []);

      const diffColor = [], diffEmissive = [], badColor = [], noTeamColor = [];
      for (const i of spot.ghosts) {
        const team = spot.teams[i];
        const mine = COL[team], other = team === 'CT' ? COL.T : COL.CT;
        if (!mine) continue;   // 分队缺失（别的文件可能带的不是 CT/T）：不判颜色
        const gc = at(ghostSnap, i), pc = at(plainSnap, i), ge = emAt(ghostSnap, i);
        // ① 幽灵的每个部件 = 同一个人正常材质的那个部件（本体色就是队色）
        for (let j = 0; j < pc.length; j++) {
          if (!near(gc[j], pc[j])) {
            diffColor.push([i, team, j, hex(gc[j]), hex(pc[j])]);
          }
        }
        // ② "自发光 = 本体色"这条规则本身：幽灵发的光就是它自己的颜色
        for (let j = 0; j < gc.length; j++) {
          if (!near(ge[j], gc[j])) diffEmissive.push([i, j, hex(ge[j]), hex(gc[j])]);
        }
        // ③ 直白版：本色里必须有本队队色，而且一个部件都不许是对方队色
        if (!gc.some((c) => near(c, mine))) noTeamColor.push([i, team, gc.map(hex)]);
        if (gc.some((c) => near(c, other))) badColor.push([i, team, gc.map(hex)]);
        // ④ 幽灵在开关来回一趟之后必须原样回来
        for (let j = 0; j < gc.length; j++) {
          if (!near(gc[j], at(backSnap, i)[j])) {
            diffColor.push([`${i}(开关来回后)`, team, j, hex(at(backSnap, i)[j]), hex(gc[j])]);
          }
        }
      }
      check(`幽灵每个部件的颜色 = 同一个人正常材质的颜色（${spot.ghosts.length} 个幽灵 × 9 部件）`,
        diffColor.length === 0, JSON.stringify(diffColor));
      check('幽灵的自发光 = 它自己的本色（不是克隆那一刻按 i < 5 猜的那份快照）',
        diffEmissive.length === 0, JSON.stringify(diffEmissive));
      check('每个幽灵身上都有本队队色',
        noTeamColor.length === 0, JSON.stringify(noTeamColor));
      check('没有一个幽灵部件是**对方**球队的颜色（蓝 CT 变黄 / 黄 T 变蓝那个 bug）',
        badColor.length === 0, JSON.stringify(badColor));
    }
  }

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

console.log(`\nplayer-xray-render: ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
