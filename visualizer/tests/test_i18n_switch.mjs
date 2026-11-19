/**
 * test_i18n_switch.mjs — 中英语言开关的**浏览器**验证（headless Chrome + CDP）。
 *
 * 文案表本身由 `check_i18n.mjs` 查（键成对、模板无漏标）。这里查的是只有真
 * 浏览器才能证伪的部分：
 *
 *   * i18n 模块加载后页面文字真的被换掉了（不是只有 data-i18n 属性）；
 *   * 点 EN 会记住选择并刷新，刷新后仍是英文，`<html lang>` 也跟着变；
 *   * 换到另一个页面（/players、/replay）还是英文 —— 语言是全局的，不是
 *     单页的；
 *   * `?lang=` 能覆盖并持久化，点回「中文」也真的能回到中文；
 *   * 整个过程没有未捕获异常（t 被同名局部变量遮住这类错误会在控制台炸）。
 *
 * 需要一个在跑的服务端（和其余浏览器测试一样）：
 *
 *     python -m app.server --port 5099 --no-watch
 *     node visualizer/tests/test_i18n_switch.mjs http://127.0.0.1:5099
 *
 * 没有 Chrome 就 SKIP（退出码 2，不算失败）。
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const BASE = (process.argv[2] || 'http://127.0.0.1:5099').replace(/\/$/, '');

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);
const CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) {
  console.log('SKIP: no Chrome found (set CHROME_PATH)');
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let passed = 0;
const failed = [];
function check(name, ok, detail = '') {
  if (ok) {
    passed++;
    console.log(`  \u2713 ${name}`);
  } else {
    failed.push(name);
    console.log(`  \u2717 ${name}${detail ? `\n      ${detail}` : ''}`);
  }
}

// ── Chrome + CDP ─────────────────────────────────────────────────────────
const PORT = Number(process.env.CDP_PORT || (20000 + Math.floor(Math.random() * 20000)));
const profile = mkdtempSync(join(tmpdir(), 'csi18n-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
  '--lang=zh-CN', '--window-size=1400,900', 'about:blank',
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
  process.exit(2);
}

const ws = new WebSocket(wsUrl);
await new Promise((r) => { ws.onopen = r; });
let msgId = 0;
const pending = new Map();
const errors = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.method === 'Runtime.exceptionThrown') {
    errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  }
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) => new Promise((res) => {
  const id = ++msgId;
  pending.set(id, res);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) {
    throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 400));
  }
  return r.result?.result?.value;
};

await send('Runtime.enable');
await send('Page.enable');

/** 导航并等到 i18n 运行时把页面刷完。 */
async function goto(path, expectLang) {
  await send('Page.navigate', { url: `${BASE}${path}` });
  for (let i = 0; i < 80; i++) {
    await sleep(250);
    const ok = await evaluate(`(() => !!window.__i18n
      && window.__i18n.getLang() === ${JSON.stringify(expectLang)}
      && !!document.querySelector('.lang-switch'))()`).catch(() => false);
    if (ok) return true;
  }
  return false;
}

console.log('\n中文默认');
let ok = await goto('/library?lang=zh', 'zh');
check('Demo 库页起来了（i18n 运行时 + 顶栏开关）', ok);
let info = await evaluate(`(() => ({
  lang: window.__i18n.getLang(),
  htmlLang: document.documentElement.lang,
  nav: document.querySelector('.nav .nav-link').textContent.trim(),
  settings: document.querySelector('#btn-settings [data-i18n]').textContent.trim(),
  title: document.title,
  focused: document.activeElement && document.activeElement.tagName,
}))()`);
check('默认是中文', info.lang === 'zh' && info.htmlLang === 'zh-CN', JSON.stringify(info));
check('导航写的是「Demo 库」', info.nav === 'Demo 库', info.nav);
check('按钮写的是「设置」', info.settings === '设置', info.settings);
check('文档标题是中文', info.title === 'cs-net — Demo 库', info.title);
const zhPlaceholder = await evaluate(`document.querySelector('#search').placeholder`);
check('搜索框提示是中文', zhPlaceholder.startsWith('搜索文件名'), zhPlaceholder);

console.log('\n点 EN：刷新后整页变英文');
await evaluate(`document.querySelector('.lang-switch button[data-lang="en"]').click()`);
ok = await goto('/library', 'en');
check('刷新后语言是 en', ok);
info = await evaluate(`(() => ({
  lang: window.__i18n.getLang(),
  htmlLang: document.documentElement.lang,
  nav: document.querySelector('.nav .nav-link').textContent.trim(),
  settings: document.querySelector('#btn-settings [data-i18n]').textContent.trim(),
  title: document.title,
  placeholder: document.querySelector('#search').placeholder,
  stored: localStorage.getItem('csnet.lang'),
  pressed: document.querySelector('.lang-switch button[data-lang="en"]').getAttribute('aria-pressed'),
}))()`);
check('语言是 en', info.lang === 'en', JSON.stringify(info));
check('<html lang> 变成 en', info.htmlLang === 'en', info.htmlLang);
check('导航变成「Demo library」', info.nav === 'Demo library', info.nav);
check('按钮变成「Settings」', info.settings === 'Settings', info.settings);
check('文档标题变成英文', info.title === 'cs-net — Demo library', info.title);
check('搜索框提示变成英文', /^Search/.test(info.placeholder), info.placeholder);
check('选择被记住（localStorage）', info.stored === 'en', String(info.stored));
check('EN 按钮标成按下', info.pressed === 'true', String(info.pressed));

console.log('\n换页面还是英文（语言是全局的）');
ok = await goto('/players', 'en');
check('玩家页起来了', ok);
info = await evaluate(`(() => ({
  nav: [...document.querySelectorAll('.nav .nav-link')].map((a) => a.textContent.trim()),
  title: document.title,
  htmlLang: document.documentElement.lang,
}))()`);
check('玩家页导航是英文', info.nav.join(' / ') === 'Demo library / Players / 3D replay', info.nav.join(' / '));
check('玩家页标题是英文', info.title === 'cs-net — Players', info.title);
check('玩家页 <html lang> 是 en', info.htmlLang === 'en', info.htmlLang);

ok = await goto('/replay', 'en');
check('回放页起来了', ok);
info = await evaluate(`(() => ({
  trails: document.querySelector('#toggle-trails + span').textContent.trim(),
  players: document.querySelector('#left-panel .panel-title [data-i18n]').textContent.trim(),
  camera: document.getElementById('camera-hint').textContent.trim(),
}))()`);
check('回放页开关是「Trails」', info.trails === 'Trails', info.trails);
check('回放页面板标题是「Player status」', info.players === 'Player status', info.players);
check('回放页操作提示是英文', /^🖱 Drag to rotate/.test(info.camera), info.camera.slice(0, 40));

console.log('\n?lang= 覆盖 + 点回中文');
await send('Page.navigate', { url: `${BASE}/replay?lang=zh` });
for (let i = 0; i < 80; i++) {
  await sleep(250);
  const l = await evaluate(`window.__i18n && window.__i18n.getLang()`).catch(() => null);
  if (l === 'zh') break;
}
info = await evaluate(`(() => ({
  lang: window.__i18n.getLang(),
  trails: document.querySelector('#toggle-trails + span').textContent.trim(),
  stored: localStorage.getItem('csnet.lang'),
}))()`);
check('?lang=zh 覆盖已记住的 en', info.lang === 'zh', JSON.stringify(info));
check('回放页开关回到「轨迹」', info.trails === '轨迹', info.trails);
check('覆盖后的选择也写回 localStorage', info.stored === 'zh', String(info.stored));

console.log('\n带 ?lang=en 的链接里点回「中文」（参数不能把选择顶回去）');
ok = await goto('/library?lang=en', 'en');
check('从 ?lang=en 进英文', ok);
await evaluate(`document.querySelector('.lang-switch button[data-lang="zh"]').click()`);
ok = await goto('/library', 'zh');
info = await evaluate(`(() => ({
  lang: window.__i18n.getLang(),
  search: new URLSearchParams(location.search).get('lang'),
  title: document.title,
}))()`);
check('点「中文」后真的回到中文', ok && info.lang === 'zh', JSON.stringify(info));
check('刷新时 URL 上的 lang 参数已被摘掉', info.search === null, String(info.search));
check('标题也回到中文', info.title === 'cs-net — Demo 库', info.title);

console.log('\n控制台');
check('没有未捕获异常', errors.length === 0, errors.slice(0, 3).join('\n      '));

chrome.kill();
try { rmSync(profile, { recursive: true, force: true }); } catch { /* windows lock */ }

console.log(`\n${passed} passed, ${failed.length} failed`);
if (failed.length) {
  for (const name of failed) console.log(`  FAILED: ${name}`);
  process.exit(1);
}
