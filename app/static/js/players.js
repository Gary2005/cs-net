/**
 * cs-net — 玩家索引。
 *
 * 一页列出库里出现过的所有玩家，带聚合数据。聚合在 SQL 里做，见
 * app/db.py 的 _PLAYER_TOTALS；这里只负责渲染和翻页。
 *
 * 卡片标题是 Steam 昵称，下面列出游戏里用过的名字。两个名字来自不同的地方：
 * 游戏内名字在 demo 里，Steam 昵称是从 steamcommunity.com 取的（见
 * app/avatars.py），所以只有缓存过头像的玩家才有 Steam 昵称可显示。
 *
 * 头像不在这个页面请求，只有已经缓存过的才会显示，其余用名字首字。
 * 要拉头像点顶栏「同步头像」，那是一个后台任务。
 */

// 中英文案（四个页面共用 visualizer/static 下的那一份）。
import { t } from '/static/js/i18n/index.js';

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: options.body ? { 'Content-Type': 'application/json' } : {},
    ...options,
  });
  let data = null;
  try { data = await res.json(); } catch { /* 空响应 */ }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

const post = (p, body) => api(p, { method: 'POST', body: JSON.stringify(body || {}) });

function toast(msg, kind = '') {
  const el = $('toast');
  el.textContent = msg;
  el.className = `toast ${kind}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), kind === 'error' ? 8000 : 4000);
}

const S = {
  q: '', sort: 'matches', watchedOnly: false,
  limit: 60, offset: 0, total: 0,
  items: [], avatars: {}, jobPoll: null,
  suggest: [], suggestOpen: false, suggestIndex: -1,
};

// ── 渲染 ──────────────────────────────────────────────────────────────────

/** 卡片标题：Steam 昵称优先，没有就退回游戏内名字。 */
function displayName(p) {
  return p.persona || p.name || t('player.common.unknown');
}

/** 游戏里用过的名字（最近的在最前面），去掉和标题重复的那个。 */
function otherNames(p) {
  const title = displayName(p).toLowerCase();
  return (p.names || []).filter((n) => n && n.toLowerCase() !== title);
}

/** 有缓存头像就显示图片，否则用名字首字。两者尺寸一致，不会跳版。 */
function avatarHTML(p, cls, label) {
  const av = S.avatars[p.steamid];
  if (av && av.url) {
    return `<img class="avatar ${cls}" src="${esc(av.url)}" alt="" loading="lazy">`;
  }
  const ch = String(label || p.name || '?').trim().charAt(0) || '?';
  return `<span class="avatar-fallback ${cls}">${esc(ch)}</span>`;
}

function pct(v) {
  return v == null ? '—' : `${v}%`;
}

function nameListHTML(p, others) {
  if (!others.length) return '';
  const chips = others.map((n) =>
    `<span class="nm${n === p.name ? ' current' : ''}">${esc(n)}</span>`).join('');
  return `<div class="pcard-names" title="${t('player.index.names.title')}">
    ${p.persona ? `<span class="nm-label">${t('player.common.inGame')}</span>` : ''}${chips}
  </div>`;
}

function cardHTML(p) {
  const title = displayName(p);
  return `
  <a class="pcard ${p.watched ? 'watched' : ''}" href="/players/${esc(p.steamid)}">
    ${avatarHTML(p, 'av-44', title)}
    <div class="pcard-main">
      <div class="pcard-name">
        <span>${esc(title)}</span>
        ${p.watched ? '<span class="star">★</span>' : ''}
      </div>
      ${nameListHTML(p, otherNames(p))}
      <div class="pcard-sub">${esc(p.steamid)}</div>
      <div class="pcard-stats">
        <span><span class="n">${p.matches}</span> ${t('player.common.unit.matches')}</span>
        <span><span class="n">${p.kd}</span> K/D</span>
        <span><span class="n">${p.adr}</span> ADR</span>
        <span><span class="n">${pct(p.win_pct)}</span> ${t('player.common.winRate')}</span>
      </div>
    </div>
  </a>`;
}

function render() {
  const content = $('content');
  if (!S.items.length) {
    content.innerHTML = (S.q || S.watchedOnly)
      ? `<div class="empty">${t('player.index.empty.filtered')}<br>
           <span class="dim">${t('player.index.empty.filteredHint')}</span></div>`
      : `<div class="empty">${t('player.index.empty.noDemos')}<br>
           <span class="dim">${t('player.index.empty.noDemosHint')}</span></div>`;
  } else {
    content.innerHTML = S.items.map(cardHTML).join('');
  }
  $('count').textContent = S.total ? t('player.index.count', { n: S.total }) : '';
  renderPager();
}

function renderPager() {
  const pager = $('pager');
  if (S.total <= S.limit) { pager.innerHTML = ''; return; }
  const page = Math.floor(S.offset / S.limit) + 1;
  const pages = Math.ceil(S.total / S.limit);
  pager.innerHTML = `
    <button ${S.offset <= 0 ? 'disabled' : ''} data-page="first">« ${t('player.common.pager.first')}</button>
    <button ${S.offset <= 0 ? 'disabled' : ''} data-page="prev">‹ ${t('player.common.pager.prev')}</button>
    <span class="page-n">${page} / ${pages}</span>
    <button ${page >= pages ? 'disabled' : ''} data-page="next">${t('player.common.pager.next')} ›</button>
    <button ${page >= pages ? 'disabled' : ''} data-page="last">${t('player.common.pager.last')} »</button>
    <span class="dim">${t('player.index.pager.total', { n: S.total })}</span>`;
}

// ── 数据 ──────────────────────────────────────────────────────────────────

function queryString() {
  const p = new URLSearchParams();
  if (S.q) p.set('q', S.q);
  if (S.watchedOnly) p.set('watched', '1');
  p.set('sort', S.sort);
  p.set('limit', S.limit);
  p.set('offset', S.offset);
  return p.toString();
}

async function load() {
  try {
    const r = await api(`/api/library/players?${queryString()}`);
    S.items = r.items || [];
    S.total = r.total || 0;
    // 合并而不是替换：搜索建议里可能有当前这页之外的玩家，它们的头像
    // 也在同一个映射里。
    S.avatars = { ...S.avatars, ...(r.avatars || {}) };
    render();
  } catch (err) {
    toast(t('player.common.toast.loadFail', { msg: err.message }), 'error');
  }
}

// ── 搜索建议 ──────────────────────────────────────────────────────────────
//
// 列表本身的筛选是子串匹配（后端 /api/library/players?q=），这里给出的是
// 排序后的少数几个候选：/api/library/players/search 会把游戏内名字、Steam
// 昵称和 SteamID 一起打分，所以「wcda」也能找到 Wicadia。

const MATCH_LABEL = {
  persona: t('player.common.persona'),
  name: t('player.common.inGameName'),
  steamid: 'SteamID',
};

function suggestHTML(p) {
  const av = S.avatars[p.steamid] || {};
  const title = displayName(p);
  const others = otherNames(p);
  const sub = others.length
    ? `${others.join(t('player.common.nameSep'))} · ${p.steamid}`
    : p.steamid;
  // 命中字段和标题一样时标签没有信息量，去掉。
  const tag = p.match_text && p.match_text.toLowerCase() !== title.toLowerCase()
    ? (MATCH_LABEL[p.match] || '') : '';
  const mark = av.url
    ? `<img class="avatar av-26" src="${esc(av.url)}" alt="" loading="lazy">`
    : `<span class="avatar-fallback av-26">${esc(title.charAt(0) || '?')}</span>`;

  return `
  <a class="sg-item" href="/players/${esc(p.steamid)}">
    ${mark}
    <div class="sg-main">
      <div class="sg-name"><span>${esc(title)}</span>${
        tag ? `<span class="sg-tag">${tag}</span>` : ''}</div>
      <div class="sg-sub">${esc(sub)}</div>
    </div>
    <div class="sg-meta">${t('player.common.matchCount', { n: p.matches })}</div>
  </a>`;
}

function closeSuggest() {
  S.suggestOpen = false;
  S.suggestIndex = -1;
  $('search-suggest').classList.add('hidden');
}

async function loadSuggest(q) {
  if (!q) return closeSuggest();
  try {
    const r = await api(`/api/library/players/search?q=${encodeURIComponent(q)}&limit=8`);
    // 输入框可能已经变了，晚到的响应丢掉。
    if (S.q !== q) return;
    S.avatars = { ...S.avatars, ...(r.avatars || {}) };
    S.suggest = r.items || [];
    S.suggestIndex = -1;
    const box = $('search-suggest');
    box.innerHTML = S.suggest.length
      ? S.suggest.map(suggestHTML).join('')
      : `<div class="sg-empty">${t('player.index.empty.filtered')}</div>`;
    box.classList.remove('hidden');
    S.suggestOpen = true;
  } catch {
    closeSuggest();
  }
}

function moveSuggest(delta) {
  if (!S.suggestOpen || !S.suggest.length) return;
  const items = $('search-suggest').querySelectorAll('.sg-item');
  S.suggestIndex = (S.suggestIndex + delta + items.length) % items.length;
  items.forEach((el, i) => el.classList.toggle('active', i === S.suggestIndex));
  items[S.suggestIndex].scrollIntoView({ block: 'nearest' });
}

// ── 头像同步 ──────────────────────────────────────────────────────────────

function pollJob(jobId) {
  clearInterval(S.jobPoll);
  S.jobPoll = setInterval(async () => {
    try {
      const job = await api(`/api/library/jobs/${jobId}`);
      $('avatar-stat').textContent = job.state === 'running' ? job.step : '';
      if (!['done', 'error', 'cancelled'].includes(job.state)) return;
      clearInterval(S.jobPoll);
      S.jobPoll = null;
      $('avatar-stat').textContent = '';
      const res = job.result || {};
      if (job.state === 'error') {
        toast(t('player.index.toast.syncFail', { msg: job.error }), 'error');
      } else if (res.skipped) {
        toast(res.note || t('player.index.toast.syncOff'));
      } else {
        const failed = res.failed
          ? t('player.index.toast.syncFailedPart', { n: res.failed }) : '';
        toast(t('player.index.toast.syncDone', { n: res.ok || 0 }) + failed,
          res.failed ? 'error' : 'ok');
      }
      load();
    } catch {
      clearInterval(S.jobPoll);
      S.jobPoll = null;
    }
  }, 1200);
}

async function syncAvatars() {
  const btn = $('btn-sync-avatars');
  btn.disabled = true;
  try {
    const r = await post('/api/library/avatars/sync', {});
    if (!r.queued) {
      toast(r.note || t('player.common.toast.syncRunning'));
    } else {
      toast(t('player.index.toast.syncStarted'));
    }
    if (r.job_id) pollJob(r.job_id);
  } catch (err) {
    toast(t('player.common.toast.submitFail', { msg: err.message }), 'error');
  } finally {
    btn.disabled = false;
  }
}

// ── 事件 ──────────────────────────────────────────────────────────────────

function bindEvents() {
  let searchTimer = null;
  $('search').oninput = (e) => {
    clearTimeout(searchTimer);
    const value = e.target.value.trim();
    searchTimer = setTimeout(() => {
      S.q = value;
      S.offset = 0;
      load();
      loadSuggest(value);
    }, 220);
  };

  $('search').addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowDown') { ev.preventDefault(); moveSuggest(1); }
    else if (ev.key === 'ArrowUp') { ev.preventDefault(); moveSuggest(-1); }
    else if (ev.key === 'Escape') { closeSuggest(); }
    else if (ev.key === 'Enter') {
      const hit = S.suggestOpen ? S.suggest[S.suggestIndex] : null;
      if (hit) { ev.preventDefault(); window.location.href = `/players/${hit.steamid}`; }
    }
  });

  // 点到别处就收起；点到建议本身就是跳转，不用管。
  document.addEventListener('click', (ev) => {
    if (!ev.target.closest('.search-wrap')) closeSuggest();
  });

  $('sort').onchange = (e) => { S.sort = e.target.value; S.offset = 0; load(); };
  $('filter-watched').onchange = (e) => {
    S.watchedOnly = e.target.checked;
    S.offset = 0;
    load();
  };

  $('pager').addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-page]');
    if (!btn || btn.disabled) return;
    const pages = Math.ceil(S.total / S.limit);
    const page = Math.floor(S.offset / S.limit);
    const target = { first: 0, prev: page - 1, next: page + 1, last: pages - 1 }[btn.dataset.page];
    S.offset = Math.max(0, Math.min(target, pages - 1)) * S.limit;
    load();
  });

  $('btn-sync-avatars').onclick = syncAvatars;
}

bindEvents();
load();
