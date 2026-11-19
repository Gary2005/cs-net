#!/usr/bin/env node
/**
 * test_eye_icons.mjs — 「可见性眼睛图标」+ 0.1× 倍速的**渲染**测试
 * （headless Chrome + CDP）。
 *
 * 这个文件只管**画出来的东西对不对**，不管可见性**判得对不对**：
 * 判定本身（几何 vs 独立参考实现、暴力对拍、性能）在 `test_visibility.mjs`
 * 和 `test_geo_visibility.mjs` 里。这里的分工是——
 *
 * **A. 0.1× 倍速**：`<option>` 加了不等于真的能放。这里走**真实交互路径** ——
 *   改 select、派发 change、播放、量真实推进量。顺便守 0.25×…4× 这些老选项没被
 *   改坏（倍速是 `advanceReplay()` 里乘上去的，改错了一整排都会错）。
 *
 * **B. 眼睛图标**：核心不变量是**集合、位置、生命周期**。
 *   - 画出来的集合 == `__v5.visibility.visibleOf(viewer)`（逐帧几何判定）
 *   - **包含队友** —— 眼睛图标以前只标敌方（引擎 `spotted` 只标敌方），现在走
 *     全有序对的几何判定，队友对也在里面。这是**故意**的语义变化。
 *   - 死人（不管是看的还是被看的）一个都不能画
 *   - 没选中任何人 / 关掉开关 → 一个都不能剩
 *   - 图标必须在**头顶之上**（不是脚下、不是身体里），而且蹲下要跟着沉
 *   - 贴图里真的有东西（不是空方块），而且要左右对称
 *
 * 参考实现是 `visibleOf()`（几何矩阵的查询接口）。它不算"拿被测代码验被测代码"：
 * 那个矩阵本身在 `test_geo_visibility.mjs` 里被一个**测试自己重写的**独立实现
 * 在真实地图 BVH 上对拍过（540 对 0 处不一致）。这里验的是**接线** ——
 * main.js 有没有把矩阵正确落到 sprite 上。
 *
 * 用法：
 *   python visualizer/server.py --port 5099 &
 *   node visualizer/tests/test_eye_icons.mjs http://127.0.0.1:5099 /path/to/match.csz
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// 以前这里只列了 macOS / Linux 的路径，Windows 上一直靠外面设 CHROME_PATH 才跑起来，
// 没设就静默 SKIP（看起来"通过"其实一条都没验）。补上 Windows，和
// test_geo_visibility.mjs 用同一份候选表。
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

const URL = process.argv[2] || 'http://127.0.0.1:5099';
const CSZ = process.argv[3];
if (!CSZ) {
  console.error('用法: node test_eye_icons.mjs <server-url> <match.csz>');
  process.exit(2);
}
const CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(2);   // 2 = 没跑成，不是"通过"（以前 exit 0 会被当成绿）
}
// 随机端口：写死端口时连着跑两次，上一轮的 Chrome 还没释放就会连不上 CDP
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const S = 0.0254;   // 1 HU = 1 英寸

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}

// ── 起 Chrome + 接 CDP ──────────────────────────────────────────────────────
const profile = mkdtempSync(join(tmpdir(), 'spot-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
  // 软件 WebGL：headless 里没有真 GPU，不给这个 three.js 会拿不到 context
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

  console.log('\nload');
  await evaluate(`window.__v5.loadPath(${JSON.stringify(CSZ)})`, true);
  // 14.8 MB 的 .csz 解析 + 建场景，固定 sleep 是赌运气（实测 1200 ms 偶尔不够，
  // 第一轮过了第二轮挂）。轮询到真的就绪为止。
  let loaded = false;
  for (let i = 0; i < 90; i++) {
    await sleep(400);
    if (await evaluate('!!(window.__v5.round && window.__v5.players.length)')) { loaded = true; break; }
  }
  check('a round is loaded', loaded);
  const hasSpotted = await evaluate('window.__v5.hasSpotted');
  if (!hasSpotted) {
    console.log('SKIP: this .csz has no `spotted` column (extracted with --no-spotted)');
    throw { skip: true };
  }
  check('the payload advertises the spotted column',
    await evaluate('!!(window.__v5.summary || {}).has_spotted'));

  // ═══════════════════════════════════════════════════════════════════════
  // A. 0.1× 倍速
  // ═══════════════════════════════════════════════════════════════════════
  console.log('\nplayback speed');
  const speeds = await evaluate(`Array.from(
    document.querySelectorAll('#speed-select option')).map(o => o.value)`);
  check('the speed select offers 0.1x', speeds.includes('0.1'), JSON.stringify(speeds));
  check('the slower half of the old ladder is still there',
    ['0.25', '0.5', '1', '2', '4'].every((v) => speeds.includes(v)), JSON.stringify(speeds));

  /** 走真实交互：改 select → change 事件 → 播放 → 推 N 步 → 暂停。 */
  async function measureSpeed(value, steps = 10, dt = 0.1) {
    return evaluate(`(() => {
      const sel = document.getElementById('speed-select');
      sel.value = ${JSON.stringify(String(value))};
      sel.dispatchEvent(new Event('change'));
      const btn = document.getElementById('btn-play');
      if (window.__v5.isPlaying) btn.click();            // 先确保是暂停的
      window.__v5.seekSample(0);
      const t0 = window.__v5.currentTime;
      btn.click();                                       // 播放
      // 连着推，rAF 插不进来 → 确定性
      for (let i = 0; i < ${steps}; i++) window.__v5.stepReplay(${dt});
      btn.click();                                       // 暂停
      return { delta: window.__v5.currentTime - t0, speed: window.__v5.playSpeed };
    })()`);
  }

  // 期望值：steps × dt × 倍速，**精确**。不用墙钟量 —— headless + swiftshader
  // 只有 ~2 fps，而主循环里有 `if (dt > 0.25) dt = 0.25`（防掉帧跳太远），
  // 按墙钟量出来的高倍速会被这个 clamp 削掉，量到的是帧率不是倍速。
  const steps = 10, dt = 0.1;
  const measured = {};
  for (const value of [0.1, 0.25, 1, 4]) {
    const r = await measureSpeed(value, steps, dt);
    const want = steps * dt * value;
    measured[value] = r.delta;
    check(`${value}x advances exactly ${value}x of the fed time`,
      Math.abs(r.delta - want) < 1e-6 && r.speed === value,
      `advanced ${r.delta} (want ${want}), playSpeed=${r.speed}`);
  }
  // 0.1× 必须真的比 1× 慢一个量级，而不只是"设上了"。上面那组已经精确验过
  // 倍速乘在 `advanceReplay()` 上了，这里再确认主循环那条真实路径：
  // **暂停时一点都不许动**（走的是同一个 advanceReplay，isPlaying=false）。
  // 注意**不能**按墙钟量"1.5 秒里走了多少"：headless + swiftshader 只有 ~1 fps，
  // 而主循环里有 `if (dt > 0.25) dt = 0.25`，量到的会是帧率而不是倍速
  // （实测同一台机器上 de_ancient 那一次 1× 在 1.5 s 里只走了 0.016 s）。
  await evaluate(`(() => {
    const btn = document.getElementById('btn-play');
    if (window.__v5.isPlaying) btn.click();
    return true;
  })()`);
  const pausedA = await evaluate('window.__v5.currentTime');
  await sleep(1200);
  const pausedB = await evaluate('window.__v5.currentTime');
  check('a paused replay does not advance on the real rAF path',
    pausedB === pausedA, `${pausedA} -> ${pausedB}`);
  check('the select reports the chosen speed to the replay core',
    await evaluate(`(() => {
      const sel = document.getElementById('speed-select');
      sel.value = '0.1'; sel.dispatchEvent(new Event('change'));
      return window.__v5.playSpeed;
    })()`) === 0.1);

  // ═══════════════════════════════════════════════════════════════════════
  // B. 眼睛图标（几何驱动）
  // ═══════════════════════════════════════════════════════════════════════
  console.log('\neye markers (geometric)');

  check('the eye toggle exists and defaults to on',
    await evaluate(`(() => {
      const t = document.getElementById('toggle-spotted');
      return !!t && t.checked;
    })()`));
  check('nothing is selected to start with',
    await evaluate('window.__v5.selectedPlayer') === -1);
  check('the eye icons are driven by geometry, not by the engine spotted column',
    await evaluate(`(() => {
      const g = window.__v5.visibility;
      return !!g && g.iconsEnabled === true && !('enabled' in g && g.enabled === undefined);
    })()`));
  check('the old `toggle-geo-vis` switch is gone (只有几何一个来源)',
    await evaluate('!document.getElementById("toggle-geo-vis")'));

  // 找一个"有东西可看、而且能看到**队友**"的时刻。队友这条是语义变化的关键：
  // 引擎的 spotted 只标敌方，几何判定走全有序对，所以队友必须会出现。
  const ref = await evaluate(`(() => {
    const T = window.__v5.round.ticks.length;
    for (let s = 0; s < T; s += 7) {
      window.__v5.seekSample(s);
      for (let v = 0; v < 10; v++) {
        const list = window.__v5.visibility.visibleOf(v);
        if (!list.length) continue;
        const mates = list.filter((e) => window.__v5.round.teams[e] === window.__v5.round.teams[v]);
        if (mates.length) return { sample: s, viewer: v, all: list, mates: mates };
      }
    }
    return null;
  })()`);
  check('found a sample where a player can observe a live teammate', !!ref,
    JSON.stringify(ref));
  if (!ref) throw { skip: true };
  console.log(`       sample ${ref.sample}: player ${ref.viewer} sees ${JSON.stringify(ref.all)}`
    + ` (teammates ${JSON.stringify(ref.mates)})`);

  // 选中 + 跳到那一刻。selectPlayer 走的和点卡片是同一条路（togglePlayerDetail）。
  await evaluate(`window.__v5.selectPlayer(${ref.viewer});
                  window.__v5.seekSample(${ref.sample}); true`);
  await sleep(250);

  check('selecting a card marks it as the viewer',
    await evaluate('window.__v5.selectedPlayer') === ref.viewer);
  check('the detail card is expanded for the selected player',
    await evaluate(`!document.getElementById('player-detail-${ref.viewer}')
      .classList.contains('hidden')`));

  const icons = await evaluate('window.__v5.spotIcons');
  const wantVisible = new Set(ref.all);
  const shown = new Set(icons.map((d, i) => (d && d.visible ? i : -1)).filter((i) => i >= 0));
  check('exactly the geometrically visible players get an eye',
    shown.size === wantVisible.size && [...shown].every((i) => wantVisible.has(i)),
    `shown ${JSON.stringify([...shown])} vs want ${JSON.stringify([...wantVisible])}`);
  check('teammates DO get an eye now (引擎 spotted 从不标队友)',
    ref.mates.every((i) => shown.has(i)),
    `mates ${JSON.stringify(ref.mates)} shown ${JSON.stringify([...shown])}`);
  check('no dead player gets an eye',
    icons.every((d, i) => !d || !d.visible || (wantVisible.has(i) && d.world !== null)));

  // 和公开接口对拍（接口只是被验对象之一，不是参考实现）
  const apiList = await evaluate(`window.__v5.visibility.visibleOf(${ref.viewer})`);
  check('the public accessor agrees with the rendered set',
    JSON.stringify([...apiList].sort((a, b) => a - b))
      === JSON.stringify([...wantVisible].sort((a, b) => a - b)),
    `api ${JSON.stringify(apiList)} vs want ${JSON.stringify([...wantVisible])}`);
  // 引擎那一路的接口还在（数据面板要用），但**不再**驱动图标
  const engList = await evaluate('window.__v5.visibleEnemies');
  console.log(`       （引擎 spotted 在同一时刻说是 ${JSON.stringify(engList)}，`
    + `已经不画了）`);

  // ── 图标必须在头顶之上 ────────────────────────────────────────────────
  console.log('\ngeometry');
  const geom = await evaluate(`(() => {
    const S = 0.0254, v = ${ref.viewer};
    const out = [];
    for (const e of ${JSON.stringify(ref.all)}) {
      const st = window.__v5.players[e];
      const d = window.__v5.spotIcons[e];
      out.push({
        e,
        iconY: d.world[1],
        headY: (st.z + 64.0938) * S,        // 眼睛世界高度（站立）
        eyeTopY: (st.z + 72) * S,           // 站立碰撞盒顶
        feetY: st.z * S,
        crouch: st.crouch,
        scale: d.scale,
      });
    }
    return out;
  })()`);
  for (const g of geom) {
    check(`icon over enemy ${g.e} sits above the head`,
      g.iconY > g.headY, `icon ${g.iconY.toFixed(3)} vs eye ${g.headY.toFixed(3)}`);
    check(`icon over enemy ${g.e} is well above the feet`,
      g.iconY - g.feetY > 1.6, `${(g.iconY - g.feetY) / S} HU`);
  }

  // 蹲下时图标要跟着头沉下去。注意**不能比世界坐标的 y** —— 那是海拔，
  // 换个采样点人可能站在地图上更高的地方，比出来是假的。要比 sprite 的
  // **局部 y**（相对脚底），再对着 `(72 - 18·crouch)/72` 的压缩比核一遍。
  const crouchSample = await evaluate(`(() => {
    const r = window.__v5.round, e = ${ref.all[0]};
    for (let s = 0; s < r.ticks.length; s++)
      if (r.players[e].alive[s] && r.players[e].crouch[s] > 0.9) return s;
    return -1;
  })()`);
  if (crouchSample >= 0) {
    const standY = await evaluate(`window.__v5.spotIcons[${ref.all[0]}].y`);
    const standCrouch = await evaluate(`window.__v5.players[${ref.all[0]}].crouch`);
    await evaluate(`window.__v5.seekSample(${crouchSample})`);
    const crouchY = await evaluate(`window.__v5.spotIcons[${ref.all[0]}].y`);
    const st = await evaluate(`window.__v5.players[${ref.all[0]}]`);
    // 画出来的是 P_HEAD_Y·bodyRatio + P_HEAD_R + 0.72（米）
    const want = (1.64 * (72 - 18 * st.crouch) / 72 + 0.19 + 0.72);
    check('the eye drops with the head when the enemy crouches',
      crouchY < standY - 0.1, `local y ${standY.toFixed(3)} (crouch ${standCrouch.toFixed(2)})`
        + ` -> ${crouchY.toFixed(3)} (crouch ${st.crouch.toFixed(2)})`);
    check('the crouched eye height matches the (72-18c)/72 body squash',
      Math.abs(crouchY - want) < 1e-6, `${crouchY} vs ${want}`);
    await evaluate(`window.__v5.seekSample(${ref.sample})`);
  } else {
    console.log('       (no fully-crouched sample for that enemy; skipped)');
  }

  // ── 死人不画 ──────────────────────────────────────────────────────────
  console.log('\ndead players');
  // 几何判定本身就只返回活人（`evaluate()` 里两侧都要求 alive），所以这里验的是
  // **接线**有没有把死人的 sprite 清掉。找一个"选中的观察者已经死了、但旁边还有
  // 活人"的采样点 —— 没有活人的话"没画"说明不了任何问题。
  const deadCase = await evaluate(`(() => {
    const r = window.__v5.round, T = r.ticks.length;
    const alive = (p, s) => !!r.players[p].alive[s];
    for (let s = 0; s < T; s += 11) {
      for (let v = 0; v < 10; v++) {
        if (alive(v, s)) continue;
        for (let e = 0; e < 10; e++) {
          if (e !== v && alive(e, s)) return { sample: s, viewer: v, other: e };
        }
      }
    }
    return null;
  })()`);
  if (deadCase) {
    await evaluate(`window.__v5.selectPlayer(${deadCase.viewer});
                    window.__v5.seekSample(${deadCase.sample}); true`);
    await sleep(250);
    const icons2 = await evaluate('window.__v5.spotIcons');
    check('a dead viewer never produces an eye',
      icons2.every((d) => !d || !d.visible),
      JSON.stringify(icons2.map((d, i) => (d && d.visible ? i : null)).filter((x) => x !== null)));
    check('the accessor agrees: no visibility list for a dead viewer',
      (await evaluate(`window.__v5.visibility.visibleOf(${deadCase.viewer})`)).length === 0);
    check('the geometric matrix has no live column for a dead viewer',
      await evaluate(`(() => {
        const r = window.__v5.visibility.result;
        if (!r) return true;
        for (let a = 0; a < r.n; a++) if (r.visible[${deadCase.viewer} * r.n + a]) return false;
        return true;
      })()`));
  } else {
    console.log('       (this .csz has no "dead viewer with a live neighbour" sample; skipped)');
  }

  // ── 图标本身画出来了没有 ──────────────────────────────────────────────
  // 前面验的都是"哪个 sprite 可见"。sprite 可见不等于**贴图里有东西**：
  // canvas 画歪了 / 全透明 / 全不透明都是一个"可见"的空方块。这里直接读
  // 贴图的像素：要有内容、要用的强调色、轮廓要**宽大于高**（眼睛的形状）、
  // 还要左右对称。
  console.log('\nicon texture');
  const px = await evaluate(`(() => {
    const img = window.__v5.spotIconAt(0).material.map.image;
    const W = img.width, H = img.height;
    const d = img.getContext('2d').getImageData(0, 0, W, H).data;
    let opaque = 0, accent = 0, aMinX = W, aMaxX = -1, aMinY = H, aMaxY = -1;
    let diff = 0, tot = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4, a = d[o + 3];
      if (a > 8) opaque++;
      // 强调色 #00e5ff：红低、绿蓝高
      if (a > 128 && d[o] < 120 && d[o + 1] > 150 && d[o + 2] > 180) {
        accent++;
        if (x < aMinX) aMinX = x; if (x > aMaxX) aMaxX = x;
        if (y < aMinY) aMinY = y; if (y > aMaxY) aMaxY = y;
      }
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W / 2; x++) {
      const l = d[(y * W + x) * 4 + 3], r = d[(y * W + (W - 1 - x)) * 4 + 3];
      diff += Math.abs(l - r); tot += Math.max(l, r);
    }
    return { W, H, frac: opaque / (W * H), accent,
             accentW: aMaxX - aMinX + 1, accentH: aMaxY - aMinY + 1,
             sym: tot ? 1 - diff / tot : 0 };
  })()`);
  check('the icon texture has drawn content (not blank, not a solid square)',
    px.frac > 0.3 && px.frac < 0.95, JSON.stringify(px));
  check('the icon is drawn in the accent colour', px.accent > 800, JSON.stringify(px));
  check('the icon outline is wider than tall (an eye, not a dot)',
    px.accentW > px.accentH * 1.2, `${px.accentW} x ${px.accentH}`);
  check('the icon is left/right symmetric', px.sym > 0.99, String(px.sym));

  // ── 取消选中 / 开关 ───────────────────────────────────────────────────
  console.log('\ndeselect and toggle');
  await evaluate(`window.__v5.selectPlayer(${ref.viewer});
                  window.__v5.seekSample(${ref.sample}); true`);
  await sleep(200);
  check('re-selecting brings the eyes back',
    (await evaluate('window.__v5.spotIcons')).some((d) => d && d.visible));

  await evaluate('window.__v5.selectPlayer(-1)');
  await sleep(200);
  check('deselecting the card removes every eye',
    (await evaluate('window.__v5.spotIcons')).every((d) => !d || !d.visible));

  await evaluate(`window.__v5.selectPlayer(${ref.viewer});
                  window.__v5.seekSample(${ref.sample}); true`);
  await sleep(200);
  await evaluate(`(() => {
    const t = document.getElementById('toggle-spotted');
    t.checked = false; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(200);
  check('turning the toggle off hides every eye',
    (await evaluate('window.__v5.spotIcons')).every((d) => !d || !d.visible));
  await evaluate(`(() => {
    const t = document.getElementById('toggle-spotted');
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(200);
  check('turning it back on restores exactly the visible set',
    JSON.stringify((await evaluate('window.__v5.spotIcons'))
      .map((d, i) => (d && d.visible ? i : -1)).filter((i) => i >= 0))
      === JSON.stringify([...wantVisible].sort((a, b) => a - b)));

  // ── 视锥 ──────────────────────────────────────────────────────────────
  // 视锥只画**选中的那一个**玩家，用的是判定那组半角（90° / 16:9），远端 25 m。
  // 这里独立算一遍：拿 `__v5.players[v]` 的 x/y/z/yaw/pitch 自己推基向量和 4 个角，
  // 再和 `frustum()` 报出来的世界坐标对拍 —— 不复用 visuals.js 的推导。
  console.log('\nview frustum');
  check('the frustum toggle exists and defaults to off',
    await evaluate(`(() => {
      const t = document.getElementById('toggle-frustum');
      return !!t && !t.checked;
    })()`));

  await evaluate(`window.__v5.selectPlayer(${ref.viewer});
                  window.__v5.seekSample(${ref.sample}); true`);
  await sleep(200);
  const fOff = await evaluate('window.__v5.visibility.frustum()');
  check('nothing is drawn while the frustum toggle is off', !fOff.visible, JSON.stringify(fOff));

  await evaluate(`(() => {
    const t = document.getElementById('toggle-frustum');
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(200);
  const fOn = await evaluate('window.__v5.visibility.frustum()');
  check('turning it on draws 8 segments', fOn.visible && fOn.drawn && fOn.segments === 8,
    JSON.stringify(fOn));
  check('the frustum is reported with the same angles as the visibility test',
    Math.abs(fOn.fovDeg - 90) < 1e-9 && Math.abs(fOn.aspect - 16 / 9) < 1e-9
      && Math.abs(fOn.distMeters - 25) < 1e-9, JSON.stringify(fOn));

  // 独立重算 4 个远端角。**全程在 three 空间算**（米）—— 别在 game 空间里用
  // 25 这个"米"数当 HU 用，那会把锥算小 39 倍（S = 0.0254），而且因为正比关系
  // 看起来"形状对、位置不对"，很难查。
  const fRef = await evaluate(`(() => {
    const S = 0.0254, v = ${ref.viewer}, st = window.__v5.players[v];
    const D = 25, tanH = Math.tan(90 * Math.PI / 360), tanV = tanH / (16 / 9);
    const y = st.yaw * Math.PI / 180, p = st.pitch * Math.PI / 180;
    // three 空间的基（和 visuals.js::eyeRay() 同一约定，朝向只做轴置换）
    const f = [Math.sin(y) * Math.cos(p), -Math.sin(p), Math.cos(y) * Math.cos(p)];
    const r = [-Math.cos(y), 0, Math.sin(y)];
    const u = [Math.sin(y) * Math.sin(p), Math.cos(p), Math.cos(y) * Math.sin(p)];
    const eye = [st.y * S, (st.z + st.eye_height) * S, st.x * S];
    const out = [];
    for (const [sr, su] of [[-1,-1],[1,-1],[1,1],[-1,1]]) {
      out.push([
        eye[0] + D*f[0] + sr*D*tanH*r[0] + su*D*tanV*u[0],
        eye[1] + D*f[1] + sr*D*tanH*r[1] + su*D*tanV*u[1],
        eye[2] + D*f[2] + sr*D*tanH*r[2] + su*D*tanV*u[2],
      ]);
    }
    return { eye: eye, corners: out, crouch: st.crouch };
  })()`);
  check('the reference recomputation used a standing sample',
    fRef.crouch < 1e-6, String(fRef.crouch));
  const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  check('the frustum apex is the eye', dist3(fOn.eye, fRef.eye) < 1e-6,
    `${JSON.stringify(fOn.eye)} vs ${JSON.stringify(fRef.eye)}`);
  // 4 个角是同一组（顺序可能被 right 的正负号镜像，所以按集合比）
  const used = new Set();
  let worst = 0;
  for (const c of fOn.corners) {
    let best = Infinity, bi = -1;
    fRef.corners.forEach((rc, i) => { const d = dist3(c, rc); if (d < best) { best = d; bi = i; } });
    used.add(bi);
    worst = Math.max(worst, best);
  }
  check('all 4 far corners match an independent recomputation', worst < 1e-6 && used.size === 4,
    `worst ${worst} matched ${used.size}/4`);
  // 远端截面到眼睛的距离必须正好是 D（矩形对角线方向会略长，所以按 forward 投影量）
  check('the far plane sits 25 m along the view direction',
    await evaluate(`(() => {
      const f = window.__v5.visibility.frustum(), S = 0.0254;
      const st = window.__v5.players[${ref.viewer}];
      const y = st.yaw * Math.PI / 180, p = st.pitch * Math.PI / 180;
      const fw = [Math.sin(y) * Math.cos(p), -Math.sin(p), Math.cos(y) * Math.cos(p)];
      const ex = [st.y * S, (st.z + st.eye_height) * S, st.x * S];
      let worst = 0;
      for (const c of f.corners) {
        const d = (c[0]-ex[0])*fw[0] + (c[1]-ex[1])*fw[1] + (c[2]-ex[2])*fw[2];
        worst = Math.max(worst, Math.abs(d - 25));
      }
      return worst;
    })()`) < 1e-6);

  await evaluate(`(() => {
    const t = document.getElementById('toggle-frustum');
    t.checked = false; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(200);
  check('turning it off hides the frustum',
    !(await evaluate('window.__v5.visibility.frustum()')).visible);
  // 没选中人时也不许画
  await evaluate('window.__v5.selectPlayer(-1)');
  await evaluate(`(() => {
    const t = document.getElementById('toggle-frustum');
    t.checked = true; t.dispatchEvent(new Event('change'));
  })()`);
  await sleep(200);
  check('no frustum is drawn when nothing is selected',
    !(await evaluate('window.__v5.visibility.frustum()')).visible);
  await evaluate(`(() => {
    const t = document.getElementById('toggle-frustum');
    t.checked = false; t.dispatchEvent(new Event('change'));
  })()`);

  // ── 扫一遍整个回合：不变量必须处处成立 ────────────────────────────────
  console.log('\nsweep every 53rd sample of the round');
  const sweep = await evaluate(`(() => {
    const r = window.__v5.round, teams = r.teams, T = r.ticks.length, P = r.players.length;
    const alive = (p, s) => !!r.players[p].alive[s];
    const bad = {
      n: 0, extra: 0, missing: 0, deadTarget: 0, deadViewer: 0,
      below: 0, untargeted: 0, mateSeen: 0, mateIcons: 0, self: 0,
    };
    const S = 0.0254;
    for (let s = 0; s < T; s += 53) {
      for (let v = 0; v < P; v++) {
        if (!alive(v, s)) continue;
        window.__v5.selectPlayer(v);
        window.__v5.seekSample(s);
        const want = new Set(window.__v5.visibility.visibleOf(v));
        const icons = window.__v5.spotIcons, st = window.__v5.players;
        bad.n++;
        for (let e = 0; e < P; e++) {
          const d = icons[e];
          if (!d) continue;
          if (d.visible && !want.has(e)) bad.extra++;
          if (!d.visible && want.has(e)) bad.missing++;
          if (d.visible && !alive(e, s)) bad.deadTarget++;
          if (d.visible && e === v) bad.self++;
          if (d.visible && teams[e] === teams[v]) bad.mateIcons++;
          if (d.visible && d.world[1] <= (st[e].z + 64.0938) * S) bad.below++;
          if (!want.has(e) && d.flagVisible) bad.untargeted++;
        }
        // 队友确实会被画（语义变化的正面证据）—— 几何列表里有队友时，
        // 图标也必须画上，不能悄悄过滤掉
        for (const e of want) if (teams[e] === teams[v] && icons[e] && icons[e].visible) bad.mateSeen++;
      }
      // 死掉的观察者：选中他，一个图标都不许有
      for (let v = 0; v < P; v++) {
        if (alive(v, s)) continue;
        window.__v5.selectPlayer(v);
        window.__v5.seekSample(s);
        const icons = window.__v5.spotIcons;
        bad.n++;
        for (let e = 0; e < P; e++) {
          if (icons[e] && icons[e].visible) bad.deadViewer++;
        }
      }
    }
    window.__v5.selectPlayer(-1);
    return bad;
  })()`);
  check('the sweep actually ran', sweep.n > 50, JSON.stringify(sweep));
  check('no eye is ever drawn on a player the viewer cannot see',
    sweep.extra === 0, JSON.stringify(sweep));
  check('no visible player is ever missing its eye',
    sweep.missing === 0, JSON.stringify(sweep));
  check('no eye is ever drawn on a dead player', sweep.deadTarget === 0, JSON.stringify(sweep));
  check('a dead viewer never gets any eye', sweep.deadViewer === 0, JSON.stringify(sweep));
  check('no eye is ever drawn on the viewer itself', sweep.self === 0, JSON.stringify(sweep));
  check('every eye is above the head', sweep.below === 0, JSON.stringify(sweep));
  check('no untargeted sprite is left visible', sweep.untargeted === 0, JSON.stringify(sweep));
  check('teammates really do get eyes now（引擎 spotted 从不标队友）',
    sweep.mateSeen > 0 && sweep.mateSeen === sweep.mateIcons,
    `mateSeen=${sweep.mateSeen} mateIcons=${sweep.mateIcons}`);

  console.log('\nconsole');
  const realErrors = consoleErrors.filter((e) => !/favicon|404 \(Not Found\)/i.test(e));
  check('no console errors', realErrors.length === 0, realErrors.slice(0, 3).join(' | '));

  console.log();
  if (failed) console.log(`${failed} FAILED`);
  else console.log(`eye icons + frustum + 0.1x speed: all ${passed} checks passed`);
} catch (err) {
  if (err && err.skip) {
    console.log('\n(partial run — skipped the render checks)');
  } else {
    console.error('\nharness error:', err && err.message ? err.message : err);
    failed++;
  }
} finally {
  ws.close();
  chrome.kill();
  await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* ignore */ }
}
process.exit(failed ? 1 : 0);
