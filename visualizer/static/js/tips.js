/**
 * 悬停解释 —— 表格表头上那个 `?`（以及任意带 `data-tip` 的东西）。
 *
 * 为什么不用原生 `title`：原生 tooltip 有一秒多的延迟、样式跟不了主题、而且
 * 长文本会被浏览器截断成一行。统计表里的解释动辄两三句（"侧背身"要讲清楚
 * 判定时刻和夹角），必须能换行、能贴着浅色面板显示。
 *
 * 为什么要一个共享模块：回放页的统计弹窗和玩家页的统计表都要这一套，两个
 * 文件各写一遍就会出现"一边有箭头一边没有""一边贴边翻车"。这里只有约 60 行，
 * 两边 import 同一份。
 *
 * 用法：
 *
 *     import { mountTips, tipTh, tipAttr } from './tips.js';
 *     mountTips();                       // 每页调一次（document 上做事件委托）
 *     html = `<tr>${tipTh(t('x.label'), 'x.tip')}</tr>`;   // 表头 + 解释
 *     html = `<span ${tipAttr('x.tip')}>7/14</span>`;      // 任意元素
 *
 * 文本都走 i18n 的 `t()`，所以中英文各自一套（键在 `i18n/*.js` 里，两边同名）。
 */

import { t } from './i18n/index.js';

let box = null;
let shownFor = null;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** `data-tip` 属性（已经转义好），用来塞进任意标签。 */
export function tipAttr(key) {
  return `data-tip="${esc(t(key))}"`;
}

/** 一个带 `?` 的表头单元格。`label` 是显示的文字，`tipKey` 是解释文案的键。 */
export function tipTh(label, tipKey) {
  return `<th class="has-tip"${tipKey ? ` ${tipAttr(tipKey)}` : ''}>`
    + `${esc(label)}${tipKey ? '<span class="tip-q">?</span>' : ''}</th>`;
}

function ensureBox() {
  if (box) return box;
  box = document.createElement('div');
  box.className = 'stat-tip';
  box.setAttribute('role', 'tooltip');
  document.body.appendChild(box);
  return box;
}

function hide() {
  shownFor = null;
  if (box) box.classList.remove('show');
}

function show(el) {
  const text = el.getAttribute('data-tip');
  if (!text) return hide();
  if (shownFor === el) return;
  shownFor = el;
  const b = ensureBox();
  b.textContent = text;
  b.classList.add('show');
  place(el, b);
}

function place(el, b) {
  const r = el.getBoundingClientRect();
  const bw = b.offsetWidth;
  const bh = b.offsetHeight;
  // 默认贴在下方；下方放不下就翻到上方（表头在第一行时常常需要翻）。
  let y = r.bottom + 8;
  if (y + bh > window.innerHeight - 8) y = Math.max(8, r.top - bh - 8);
  let x = r.left + r.width / 2 - bw / 2;
  x = Math.max(8, Math.min(x, window.innerWidth - bw - 8));
  b.style.left = `${Math.round(x)}px`;
  b.style.top = `${Math.round(y)}px`;
}

/** 在 document 上装一次事件委托（重复调用没有副作用）。 */
export function mountTips() {
  if (mountTips._done) return;
  mountTips._done = true;

  document.addEventListener('mouseover', (e) => {
    const el = e.target instanceof Element ? e.target.closest('[data-tip]') : null;
    if (el) show(el); else hide();
  });
  document.addEventListener('mouseout', (e) => {
    const el = e.target instanceof Element ? e.target.closest('[data-tip]') : null;
    if (el && el === shownFor) hide();
  });
  // 滚动 / 改窗口 / 点别处：贴着的浮层立刻收起来（位置已经不成立了）
  document.addEventListener('scroll', hide, true);
  window.addEventListener('resize', hide);
  document.addEventListener('click', hide, true);
  document.addEventListener('keydown', hide);
}
