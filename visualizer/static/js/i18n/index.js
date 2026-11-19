/**
 * i18n — 中英双语（zh / en），四个页面共用。
 *
 * 目录里只有两件事：**文案表**（common.js / replay.js / library.js /
 * players.js，各自带 zh + en 两份）和这个运行时。页面脚本只认 `t()` 和
 * `data-i18n*` 属性，不关心语言存在哪。
 *
 * 为什么不是每个页面各写一份：顶栏、按钮名、单位这些词四个页面都在用，
 * 抄四遍就一定会出现"回放页叫『载入』、库页叫『打开』"这种漂移；而且语言
 * 是全局选择，用户切换后翻到另一页必须还是同一种语言。
 *
 * ── 用法 ──────────────────────────────────────────────────────────────
 *
 * HTML（**保留中文原文**，它同时是没有 JS 时的兜底和默认语言）：
 *
 *     <span data-i18n="replay.toggle.trails">轨迹</span>
 *     <button data-i18n-title="replay.btn.prev" title="上一回合">⏮</button>
 *     <input data-i18n-placeholder="library.search.ph" placeholder="搜索…">
 *     <div data-i18n-html="replay.stats.note">伤害 = …<b>实际扣掉的血</b>…</div>
 *
 * 五种属性：`data-i18n`（textContent）、`data-i18n-html`（innerHTML，用于
 * 文案里带 <b>/<br> 的）、`data-i18n-title`、`data-i18n-placeholder`、
 * `data-i18n-aria-label`。
 *
 * JS：
 *
 *     import { t } from './i18n/index.js';
 *     showToast(t('replay.toast.loadFail', { msg: err.message }), true);
 *
 * 占位符写成 `{name}`，`t(key, { name: 'x' })` 替换。英文的复数不做特殊
 * 处理 —— 文案尽量写成 "Selected: {n}" / "{n} demos" 这种对任何数字都读得
 * 通的句子，而不是 "1 demo / 2 demos" 两套。
 *
 * ── 语言怎么定 ────────────────────────────────────────────────────────
 *
 * 优先级：`?lang=` > localStorage > **默认中文**。
 *
 * 不做浏览器语言自动判断是有意的：这个产品的内容（demo、地图、选手名）
 * 本来就是中文语境，中国用户占绝大多数；`en-US` 的浏览器自动进英文界面
 * 反而会让"我只想看回放"的人多一步。英文用户点右上角的 EN 即可，选择会
 * 记住；分享链接可以带 `?lang=en`。
 *
 * 切换语言直接 **刷新页面**：页面里大量文案是渲染时一次性拼出来的（选手
 * 卡片、统计矩阵、指标表、AI 释义），逐个重渲染等于把每个页面再写一遍，
 * 漏一处就是"一半中文一半英文"。语言是低频操作，刷新最不容易出错。
 */

import common from './common.js';
import replay from './replay.js';
import library from './library.js';
import players from './players.js';

const MODULES = [common, replay, library, players];

/** 合并各页面的文案表。键重复 = 两个页面用了同一个键，后加载的覆盖先加载的。 */
const DICTS = { zh: {}, en: {} };
for (const mod of MODULES) {
  for (const lang of ['zh', 'en']) {
    Object.assign(DICTS[lang], (mod && mod[lang]) || {});
  }
}

export const LANGS = ['zh', 'en'];

const STORAGE_KEY = 'csnet.lang';
const HTML_LANG = { zh: 'zh-CN', en: 'en' };

const _warned = new Set();

/** 宽松地认语言码：zh / zh-CN / zh-Hans / en / en-US 都收，其余返回 null。 */
export function normalizeLang(raw) {
  if (!raw) return null;
  const v = String(raw).trim().toLowerCase();
  if (!v) return null;
  if (v === 'zh' || v.startsWith('zh-')) return 'zh';
  if (v === 'en' || v.startsWith('en-')) return 'en';
  return null;
}

function queryLang() {
  try {
    return normalizeLang(new URLSearchParams(window.location.search).get('lang'));
  } catch {
    return null;
  }
}

function storedLang() {
  try {
    return normalizeLang(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return null;
  }
}

const _fromQuery = queryLang();
if (_fromQuery) {
  // 带 ?lang= 进来就等于选过了：记住它，翻到别的页面也还是这种语言。
  try {
    window.localStorage.setItem(STORAGE_KEY, _fromQuery);
  } catch {
    /* 隐私模式：存不了就只管这一次 */
  }
}

let lang = _fromQuery || storedLang() || 'zh';

/** 当前语言：'zh' 或 'en'。 */
export function getLang() {
  return lang;
}

/** 取一条文案。找不到英文时退回中文，中文也没有就返回键名并警告一次。 */
export function t(key, vars) {
  const table = DICTS[lang] || DICTS.zh;
  let s = table[key];
  if (s === undefined) {
    s = DICTS.zh[key];
    if (s === undefined) {
      if (!_warned.has(key)) {
        _warned.add(key);
        console.warn(`[i18n] missing key: ${key}`);
      }
      return key;
    }
  }
  return vars ? s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] == null ? m : String(vars[k]))) : s;
}

/** 只读视图，给自测 / 调试用：`window.__i18n.dicts.en['nav.library']`。 */
export const dicts = DICTS;

function applyOne(el, key, mode) {
  if (!key) return;
  const s = t(key);
  if (s === key) return;             // 缺键：保留 HTML 里的原文，不要把界面弄空
  if (mode === 'html') el.innerHTML = s;
  else if (mode === 'title') el.setAttribute('title', s);
  else if (mode === 'placeholder') el.setAttribute('placeholder', s);
  else if (mode === 'aria-label') el.setAttribute('aria-label', s);
  else el.textContent = s;
}

/**
 * 把 root 里所有带 `data-i18n*` 的元素翻成当前语言。
 *
 * 动态插进来的子树也可以单独调（`applyI18n(dialog)`）—— 但用 `t()` 拼出来
 * 的字符串不需要再走一遍。
 */
export function applyI18n(root = document) {
  if (!root || !root.querySelectorAll) return;
  for (const el of root.querySelectorAll('[data-i18n]')) {
    applyOne(el, el.getAttribute('data-i18n'), 'text');
  }
  for (const el of root.querySelectorAll('[data-i18n-html]')) {
    applyOne(el, el.getAttribute('data-i18n-html'), 'html');
  }
  for (const el of root.querySelectorAll('[data-i18n-title]')) {
    applyOne(el, el.getAttribute('data-i18n-title'), 'title');
  }
  for (const el of root.querySelectorAll('[data-i18n-placeholder]')) {
    applyOne(el, el.getAttribute('data-i18n-placeholder'), 'placeholder');
  }
  for (const el of root.querySelectorAll('[data-i18n-aria-label]')) {
    applyOne(el, el.getAttribute('data-i18n-aria-label'), 'aria-label');
  }
}

/**
 * 切换语言：记住 + 刷新。
 *
 * URL 上如果有 `?lang=` 要先摘掉，否则刷新后"查询参数优先"又把它顶回去，
 * 点了等于没点。
 */
export function setLang(code) {
  const next = normalizeLang(code);
  if (!next || next === lang) return;
  lang = next;
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
  } catch {
    /* 存不了也照样切这一页 */
  }
  try {
    const url = new URL(window.location.href);
    if (url.searchParams.has('lang')) {
      url.searchParams.delete('lang');
      window.history.replaceState(null, '', url);
    }
  } catch {
    /* 文件协议之类没有 URL API 的场景：不摘参数，刷新后可能还是旧语言 */
  }
  window.location.reload();
}

/**
 * 顶栏右侧的语言开关：`中文 | EN`。
 *
 * 由 JS 插进去而不是写进四份模板：四个模板的顶栏结构不一样，但都有
 * `.tb-actions`；一份代码保证四页长得一样、漏改一页也不会出现"这页没有
 * 语言开关"。
 */
export function mountLangSwitcher() {
  if (document.querySelector('.lang-switch')) return;
  const host = document.querySelector('.topbar .tb-actions') || document.querySelector('.topbar');
  if (!host) return;

  const box = document.createElement('div');
  box.className = 'lang-switch';
  box.setAttribute('role', 'group');
  box.setAttribute('aria-label', 'Language / 语言');

  for (const code of LANGS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.dataset.lang = code;
    btn.textContent = code === 'zh' ? '中文' : 'EN';
    btn.title = code === 'zh' ? '中文界面' : 'English interface';
    btn.setAttribute('aria-pressed', String(code === lang));
    btn.addEventListener('click', () => setLang(code));
    box.appendChild(btn);
  }

  host.appendChild(box);
}

// 模块加载即生效：脚本都是 defer 的，此时 DOM 已经解析完，不用等 DOMContentLoaded。
document.documentElement.lang = HTML_LANG[lang];
applyI18n(document);
mountLangSwitcher();

// 调试 / 自测入口：`window.__i18n.getLang()`。不带的话浏览器测试没法确认
// 当前语言，只能靠界面上的文字反推。
window.__i18n = { getLang, setLang, t, applyI18n, dicts };
