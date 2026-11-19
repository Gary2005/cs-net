/**
 * cs-net — Demo 库前端。
 *
 * 无构建步骤、无框架，和 visualizer 那边的做法一致：一个模块、原生 DOM、
 * 模板字符串渲染。所有状态都在 `S` 里，`render()` 是唯一的出口。
 *
 * 后端接口见 app/api.py，前缀统一是 /api/library。
 */

// 中英文案。四个页面共用 visualizer/static 下的那份（两边服务器都会挂
// /static），顶栏的语言开关也是它插进去的。
import { t } from '/static/js/i18n/index.js';

// ── 小工具 ────────────────────────────────────────────────────────────────

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
const put = (p, body) => api(p, { method: 'PUT', body: JSON.stringify(body || {}) });
const patch = (p, body) => api(p, { method: 'PATCH', body: JSON.stringify(body || {}) });

function fmtSize(bytes) {
  if (!bytes) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let n = bytes, i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

function fmtDuration(mins) {
  if (mins == null) return '—';
  const m = Math.round(mins);
  return m >= 60 ? `${Math.floor(m / 60)}h${m % 60}m` : t('lib.card.durationMin', { n: m });
}

function toast(msg, kind = '') {
  const el = $('toast');
  el.textContent = msg;
  el.className = `toast ${kind}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), kind === 'error' ? 8000 : 4000);
}

// ── 状态 ──────────────────────────────────────────────────────────────────

// 状态表：值是接口里的状态码（筛选直接用它），第二项是当前语言的标签。
const STATUSES = [
  ['discovered', t('lib.status.discovered')], ['extracting', t('lib.status.extracting')],
  ['ready', t('lib.status.ready')], ['error', t('lib.status.error')],
  ['missing', t('lib.status.missing')],
];

/**
 * 卡片 / 列表行上的状态标签：和「全部状态」筛选用同一份文案，认不出的状态退回
 * 后端给的 status_label。
 */
const statusText = (d) =>
  (STATUSES.find(([value]) => value === d.status) || [])[1] || d.status_label || d.status;

// 平台名和存储目录名是**数据**（`SOURCE_ORDER` / `storage.py` 里的键），不是
// 模板里的字。这里按 key 翻一层显示名：英文界面看到 "Local" 而不是「本地」，
// 但写进筛选参数、配置和 `src-*` class 的仍然是原始值，翻译只影响显示。
const SOURCE_LABELS = {
  '本地': 'lib.source.local',
  '5E': 'lib.source.5e',
  '完美': 'lib.source.perfect',
  '官匹': 'lib.source.official',
};
const sourceText = (s) => (SOURCE_LABELS[s] ? t(SOURCE_LABELS[s]) : (s || ''));

const STORAGE_LABELS = {
  csz: ['lib.storage.label.csz', 'lib.storage.note.csz'],
  thumbs: ['lib.storage.label.thumbs', 'lib.storage.note.thumbs'],
  avatars: ['lib.storage.label.avatars', 'lib.storage.note.avatars'],
  recycle: ['lib.storage.label.recycle', 'lib.storage.note.recycle'],
  db: ['lib.storage.label.db', 'lib.storage.note.db'],
};
const storageLabel = (e) => (STORAGE_LABELS[e.key] ? t(STORAGE_LABELS[e.key][0]) : e.label);
const storageNote = (e) => (STORAGE_LABELS[e.key] ? t(STORAGE_LABELS[e.key][1]) : e.note);

const S = {
  view: localStorage.getItem('csnet.view') || 'grid',
  q: '', sort: 'library', status: '', source: '', map: '',
  watchedOnly: false,
  limit: 60, offset: 0, total: 0,
  items: [],
  avatars: {},
  selected: new Set(),
  sources: [], sourceLabels: {},
  jobPoll: null, lastJobSig: '',
};

// ── 数据加载 ──────────────────────────────────────────────────────────────

function queryString() {
  const p = new URLSearchParams();
  if (S.q) p.set('q', S.q);
  if (S.status) p.set('status', S.status);
  if (S.source) p.set('source', S.source);
  if (S.map) p.set('map', S.map);
  if (S.watchedOnly) p.set('watched', '1');
  p.set('sort', S.sort);
  p.set('limit', S.limit);
  p.set('offset', S.offset);
  return p.toString();
}

async function load({ keepSelection = false } = {}) {
  try {
    const [list, stats] = await Promise.all([
      api(`/api/library/list?${queryString()}`),
      api('/api/library/stats'),
    ]);
    S.items = list.items || [];
    S.total = list.total || 0;
    // 头像由列表接口一起带回来（只含已经缓存过的），卡片就不用再逐个去问。
    S.avatars = list.avatars || {};
    if (!keepSelection) {
      // 选中项在翻页/筛选后可能已经不在结果里，清掉避免误操作。
      const visible = new Set(S.items.map((d) => d.id));
      for (const id of [...S.selected]) if (!visible.has(id)) S.selected.delete(id);
    }
    renderStats(stats);
    renderFilterOptions(stats);
    render();
  } catch (err) {
    toast(t('lib.toast.loadFail', { msg: err.message }), 'error');
  }
}

async function loadConfig() {
  const cfg = await api('/api/library/config');
  S.sources = cfg.sources || [];
  S.sourceLabels = cfg.source_labels || {};
  return cfg;
}

// ── 渲染：统计条 ──────────────────────────────────────────────────────────

function renderStats(stats) {
  const by = stats.by_status || {};
  const parts = [
    `<div class="stat"><span class="n">${stats.total || 0}</span><span class="l">${t('lib.stats.demos')}</span></div>`,
    `<div class="stat"><span class="n">${fmtSize(stats.bytes)}</span><span class="l">${t('lib.stats.rawSize')}</span></div>`,
    `<div class="stat ok"><span class="n">${by.ready || 0}</span><span class="l">${t('lib.status.ready')}</span></div>`,
  ];
  if (by.discovered) parts.push(`<div class="stat warn"><span class="n">${by.discovered}</span><span class="l">${t('lib.status.discovered')}</span></div>`);
  if (by.extracting) parts.push(`<div class="stat"><span class="n">${by.extracting}</span><span class="l">${t('lib.status.extracting')}</span></div>`);
  if (by.error) parts.push(`<div class="stat err"><span class="n">${by.error}</span><span class="l">${t('lib.status.error')}</span></div>`);
  if (by.missing) parts.push(`<div class="stat warn"><span class="n">${by.missing}</span><span class="l">${t('lib.status.missing')}</span></div>`);

  const src = Object.entries(stats.by_source || {})
    .sort((a, b) => b[1] - a[1]).slice(0, 5)
    .map(([k, v]) => `${esc(sourceText(k))} ${v}`).join(' · ');
  if (src) parts.push(`<div class="stat"><span class="l">${t('lib.stats.source')}</span><span class="n" style="font-size:13px">${src}</span></div>`);

  $('stats-strip').innerHTML = parts.join('');
}

function renderFilterOptions(stats) {
  const st = $('filter-status');
  if (st.options.length <= 1) {
    st.innerHTML = `<option value="">${t('lib.filter.allStatus')}</option>` +
      STATUSES.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
  }
  const sc = $('filter-source');
  const counts = stats.by_source || {};
  sc.innerHTML = `<option value="">${t('lib.filter.allSource')}</option>` +
    S.sources.map((s) => `<option value="${esc(s)}">${esc(sourceText(s))}${counts[s] ? ` (${counts[s]})` : ''}</option>`).join('');
  sc.value = S.source;

  const mp = $('filter-map');
  const maps = stats.maps || [];
  if (mp.dataset.sig !== maps.join(',')) {
    mp.dataset.sig = maps.join(',');
    mp.innerHTML = `<option value="">${t('lib.filter.allMap')}</option>` +
      maps.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
    mp.value = S.map;
  }
}

// ── 渲染：列表 ────────────────────────────────────────────────────────────

function rosterOf(d) {
  const players = d.players || [];
  const watched = new Set((d.watched || []).map((w) => w.name));
  const sb = new Map((d.scoreboard || []).map((r) => [r.name, r]));
  return players.map((p) => {
    const stat = sb.get(p.name);
    const av = S.avatars[p.steamid] || {};
    return {
      steamid: String(p.steamid || ''),
      name: p.name, team: p.team || 0,
      watched: watched.has(p.name),
      kd: stat ? `${stat.kills}/${stat.deaths}` : '',
      avatar: av.url || '',
    };
  });
}

/**
 * 头像优先，没有就退回原来那个队色小圆点。
 *
 * 这里只出现游戏内名字：一场比赛记的就是当场用的名字，Steam 昵称属于账号，
 * 放在玩家页。有 SteamID 的话整条做成链接，直接跳到玩家档案。
 */
function playerChipHTML(p, cls) {
  const mark = p.avatar
    ? `<img class="pav" src="${esc(p.avatar)}" alt="" loading="lazy">`
    : '<span class="dot"></span>';
  const inner = `${mark}<span>${esc(p.name)}</span>${p.kd ? `<span class="kd">${p.kd}</span>` : ''}`;
  if (!p.steamid) {
    return `<div class="${cls}" title="${esc(p.name)}">${inner}</div>`;
  }
  return `<a class="${cls}" href="/players/${esc(p.steamid)}" title="${esc(p.name)}">${inner}</a>`;
}

function scoreHTML(d) {
  const a = d.team_a_score, b = d.team_b_score;
  const an = d.team_a_name || 'Team A', bn = d.team_b_name || 'Team B';
  if (a == null || b == null) {
    return `<div class="score-line"><span class="tn">${esc(an)}</span>
      <span class="sc" style="font-size:12px;color:#8fa3c4">vs</span>
      <span class="tn">${esc(bn)}</span></div>`;
  }
  const aWin = a > b, bWin = b > a;
  return `<div class="score-line">
    <span class="tn ${aWin ? 'win' : ''}" title="${esc(an)}">${esc(an)}</span>
    <span class="sc">${a} : ${b}</span>
    <span class="tn ${bWin ? 'win' : ''}" title="${esc(bn)}">${esc(bn)}</span>
  </div>`;
}

function thumbHTML(d, cls) {
  return d.thumb_path
    ? `<img src="/api/library/thumb/${d.id}" alt="" loading="lazy"
            onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'no-thumb',textContent:'NO RADAR'}))">`
    : `<div class="no-thumb">NO RADAR</div>`;
}

function cardHTML(d) {
  const title = d.display_name || d.filename;
  const roster = rosterOf(d);
  const team0 = roster.filter((p) => p.team === 0);
  const team1 = roster.filter((p) => p.team !== 0);
  const sel = S.selected.has(d.id);

  const rosterHTML = roster.length
    ? `<div class="roster">${[...team0, ...team1].map((p) =>
        playerChipHTML(p, `p team${p.team} ${p.watched ? 'watched' : ''}`)).join('')}</div>`
    : `<div class="dim" style="font-size:12px">${t('lib.card.noRoster')}</div>`;

  return `
  <article class="card ${sel ? 'selected' : ''}" data-id="${d.id}">
    <div class="card-hero">
      ${thumbHTML(d)}
      <div class="hero-top">
        <input type="checkbox" class="card-check" data-check="${d.id}" ${sel ? 'checked' : ''}>
        <span class="src-badge src-${esc(d.source || '本地')}">${esc(sourceText(d.source || '本地'))}</span>
        ${d.map_name ? `<span class="map-chip">${esc(d.map_name)}</span>` : ''}
      </div>
      <div class="hero-bottom">${scoreHTML(d)}</div>
    </div>

    <div class="card-body">
      <div class="title-row">
        <div class="card-title editable" data-edit="display_name" data-id="${d.id}"
             title="${t('lib.card.renameTitle')}">${esc(title)}</div>
      </div>
      ${d.display_name ? `<div class="filename">${esc(d.filename)}</div>` : ''}

      ${rosterHTML}

      <div class="meta-row">
        <span>${fmtDuration(d.duration_mins)}</span>
        <span>${t('lib.card.rounds', { n: d.total_rounds ?? '—' })}</span>
        <span>${fmtSize(d.file_size)}</span>
        <span title="${t('lib.card.fileTimeTitle')}">${t('lib.card.fileTime', { time: esc(d.match_date || (d.added_at || '').slice(0, 10)) })}</span>
      </div>

      <div class="remark ${d.remark ? '' : 'empty'}" data-edit="remark" data-id="${d.id}"
           title="${t('lib.card.remarkTitle')}">${d.remark ? esc(d.remark) : t('lib.card.remarkEmpty')}</div>

      <div class="card-foot">
        <span class="status-dot st-${d.status}">${esc(statusText(d))}</span>
        ${(d.watched || []).length ? `<span class="watched-chip">★ ${t('lib.card.watched', { n: (d.watched || []).length })}</span>` : ''}
        <span class="spacer"></span>
        <button class="btn-mini" data-act="open" data-id="${d.id}">▶ ${t('lib.card.analyze')}</button>
        <button class="btn-mini" data-act="extract" data-id="${d.id}">${d.status === 'ready' ? t('lib.card.refresh') : t('lib.card.extract')}</button>
        <button class="btn-mini" data-act="delete" data-id="${d.id}" title="${t('lib.card.deleteTitle')}">${t('common.action.delete')}</button>
      </div>
    </div>
  </article>`;
}

function rowHTML(d) {
  const title = d.display_name || d.filename;
  const a = d.team_a_score, b = d.team_b_score;
  const roster = rosterOf(d);
  const playersHTML = roster.map((p) =>
    p.steamid
      ? `<a class="w-link ${p.watched ? 'w' : ''}" href="/players/${esc(p.steamid)}">${esc(p.name)}</a>`
      : (p.watched ? `<span class="w">${esc(p.name)}</span>` : esc(p.name))).join(t('lib.row.playerSep'));
  const sel = S.selected.has(d.id);

  return `
  <div class="row ${sel ? 'selected' : ''}" data-id="${d.id}">
    <input type="checkbox" class="card-check" data-check="${d.id}" ${sel ? 'checked' : ''}>
    <div class="row-thumb">${thumbHTML(d)}</div>
    <div class="row-main">
      <div class="row-name">${esc(title)}</div>
      <div class="row-sub">${esc(d.filename)}</div>
    </div>
    <div class="row-score">
      ${a == null ? `<span class="tn">${esc(d.team_a_name || 'Team A')}</span>
        <span class="sc" style="font-size:11px;color:#8fa3c4">vs</span>
        <span class="tn">${esc(d.team_b_name || 'Team B')}</span>`
      : `<span class="tn ${a > b ? 'win' : ''}">${esc(d.team_a_name || 'Team A')}</span>
        <span class="sc">${a}:${b}</span>
        <span class="tn ${b > a ? 'win' : ''}">${esc(d.team_b_name || 'Team B')}</span>`}
    </div>
    <div class="row-players" title="${esc(roster.map((p) => p.name).join(t('lib.row.playerSep')))}">${playersHTML || '—'}</div>
    <div class="row-meta">
      <span class="status-dot st-${d.status}">${esc(statusText(d))}</span>
      <span>${d.map_name ? esc(d.map_name) : '—'}</span>
      <span>${t('lib.card.rounds', { n: d.total_rounds ?? '—' })}</span>
      <span>${fmtSize(d.file_size)}</span>
    </div>
    <div class="row-actions">
      <button class="btn-mini" data-act="open" data-id="${d.id}">▶</button>
      <button class="btn-mini" data-act="extract" data-id="${d.id}">${d.status === 'ready' ? '↻' : '⬇'}</button>
      <button class="btn-mini" data-act="delete" data-id="${d.id}" title="${t('lib.card.deleteTitle')}">✕</button>
    </div>
  </div>`;
}

function render() {
  const content = $('content');
  content.className = `content ${S.view === 'grid' ? 'grid-view' : 'list-view'}`;
  $('view-grid').classList.toggle('active', S.view === 'grid');
  $('view-list').classList.toggle('active', S.view === 'list');

  if (!S.items.length) {
    const anyFilter = S.q || S.status || S.source || S.map || S.watchedOnly;
    // 有筛选时这一句接在「没有匹配的 Demo。清空筛选，或者」后面，没有筛选时它
    // 自己起头。中文两句一样，英文的首字母大小写不同，所以各留一份文案。
    content.innerHTML = `<div class="empty">
      ${anyFilter ? t('lib.empty.noMatch') : ''}
      ${t(anyFilter ? 'lib.empty.orOpenSettings' : 'lib.empty.openSettings')}<br>
      <span class="dim">${t('lib.empty.dropHint')}</span>
    </div>`;
  } else {
    content.innerHTML = S.items.map(S.view === 'grid' ? cardHTML : rowHTML).join('');
  }

  renderSelectionBar();
  renderPager();
}

function renderSelectionBar() {
  const n = S.selected.size;
  $('selection-bar').classList.toggle('hidden', n === 0);
  $('sel-count').textContent = n;
}

function renderPager() {
  const pager = $('pager');
  if (S.total <= S.limit) { pager.innerHTML = S.total ? `<span class="dim">${t('lib.pager.total', { n: S.total })}</span>` : ''; return; }
  const page = Math.floor(S.offset / S.limit) + 1;
  const pages = Math.ceil(S.total / S.limit);
  pager.innerHTML = `
    <button ${S.offset <= 0 ? 'disabled' : ''} data-page="first">« ${t('lib.pager.first')}</button>
    <button ${S.offset <= 0 ? 'disabled' : ''} data-page="prev">‹ ${t('lib.pager.prev')}</button>
    <span class="page-n">${page} / ${pages}</span>
    <button ${page >= pages ? 'disabled' : ''} data-page="next">${t('lib.pager.next')} ›</button>
    <button ${page >= pages ? 'disabled' : ''} data-page="last">${t('lib.pager.last')} »</button>
    <span class="dim">${t('lib.pager.total', { n: S.total })}</span>`;
}

// ── 交互：选择 / 行内编辑 / 操作 ──────────────────────────────────────────

/**
 * 双击卡片时**不该**当成「打开分析」的元素。
 *
 * 卡片里这些都有自己的点击语义：复选框选中、按钮执行动作、选手标签跳档案、
 * 重命名框进编辑。它们都吃点击，双击自然也该归它们，否则在复选框上双击会
 * 顺手把整个 demo 打开。
 */
const SKIP_DBLCLICK = 'button, a, input, select, textarea, [data-act], [data-check], [data-edit]';

/**
 * 这一次双击要不要放行。
 *
 * 三个事实叠在一起，决定了只能这么判：
 *   1. 双击时浏览器在**第二下按下之后**才选中光标下的词，所以到 mouseup 和
 *      dblclick 一定看得见非空选区；拿"有没有选区"当门槛会把双击全禁掉
 *      （列表视图双击行名一开始就是这么失效的）。
 *   2. 那一下按下时选区还是空的，所以"按下瞬间有没有选区"也判不出拖拽。
 *   3. 双击的指针不动，拖拽的指针一定动。
 *
 * 所以状态机是：第一下按下时按"指针有没有动过"给这次交互定性（动过且手里有
 * 选区 = 上一轮拖选留下的，这一轮整个挡掉；否则先放行），第二下按下时不再改
 * 结论，而这一条正是把「拖选一段字、紧接着双击」挡在外面的关键。
 */
let pressState = 'clean';
/** 按下之后指针有没有真的移动过（区分拖拽和双击自带的选词）。 */
let pointerMoved = false;
let pressX = 0;
let pressY = 0;

document.addEventListener('mousedown', (ev) => {
  if (ev.button !== 0) return;
  if (ev.detail > 1) return;               // 第二下：沿用第一下的结论
  const sel = window.getSelection();
  const hasRange = !!(sel && !sel.isCollapsed);
  pressState = (pointerMoved && hasRange) ? 'dragged' : 'clean';
  // 下一段拖拽从这里算起
  pointerMoved = false;
  pressX = ev.clientX;
  pressY = ev.clientY;
  // 落在空白处按下会清掉选区，这是浏览器本来的行为，保持它。
  if (hasRange) sel.removeAllRanges();
}, true);

document.addEventListener('mousemove', (ev) => {
  if (ev.buttons & 1 && (Math.abs(ev.clientX - pressX) > 3 || Math.abs(ev.clientY - pressY) > 3)) {
    pointerMoved = true;
  }
}, true);

function bindContent() {
  const content = $('content');

  content.addEventListener('click', async (ev) => {
    const check = ev.target.closest('[data-check]');
    if (check) {
      const id = Number(check.dataset.check);
      check.checked ? S.selected.add(id) : S.selected.delete(id);
      check.closest('.card, .row')?.classList.toggle('selected', check.checked);
      renderSelectionBar();
      return;
    }

    const act = ev.target.closest('[data-act]');
    if (act) {
      const id = Number(act.dataset.id);
      if (act.dataset.act === 'open') return openInReplay([id]);
      if (act.dataset.act === 'extract') return extractMany([id]);
      if (act.dataset.act === 'delete') return deleteOne(id);
      return;
    }

    const edit = ev.target.closest('[data-edit]');
    if (edit) return startEdit(edit);
  });

  // 双击卡片 / 行直接进分析模式，省掉「先选中、再点 ▶ 分析」两步。
  //
  // 卡片里有一堆自己吃点击的东西，所以这里认的是**被点的元素**而不是卡片：
  // 点在复选框、按钮、选手链接、重命名框上时这次双击不算数，交给它们自己处理。
  //
  // 只开双击的那一张，不是当前选中集：选中是「批量操作」用的，双击是「就看
  // 这个」，把一个 20 个 demo 的选区甩进回放页不符合双击的直觉。批量走顶部那条。
  content.addEventListener('dblclick', (ev) => {
    if (ev.target.closest(SKIP_DBLCLICK)) return;
    // 刚拖选过文字就不当成「打开」，那样双击会把用户正在选的词吃掉，转而去
    // 跳页。判定放在 mousedown 里做（见 pressState 那段注释），这里只看结论。
    if (pressState === 'dragged') return;
    const el = ev.target.closest('.card[data-id], .row[data-id]');
    if (!el) return;
    const id = Number(el.dataset.id);
    // 和「▶ 分析」按钮同一个入口：取批次、必要时排队提取、然后跳转。
    // openInReplay 内部自己 catch，失败会出 toast，不用在这里再接一层。
    if (Number.isFinite(id)) openInReplay([id]);
  });

  $('pager').addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-page]');
    if (!btn || btn.disabled) return;
    const pages = Math.ceil(S.total / S.limit);
    const page = Math.floor(S.offset / S.limit);
    const target = {
      first: 0, prev: page - 1, next: page + 1, last: pages - 1,
    }[btn.dataset.page];
    S.offset = Math.max(0, Math.min(target, pages - 1)) * S.limit;
    load();
  });
}

/** 行内编辑显示名 / 备注。回车或失焦提交，Esc 取消。 */
function startEdit(el) {
  const id = Number(el.dataset.id);
  const field = el.dataset.edit;
  const item = S.items.find((d) => d.id === id);
  if (!item) return;
  const current = field === 'display_name'
    ? (item.display_name || '')
    : (item.remark || '');

  const isRemark = field === 'remark';
  const input = document.createElement(isRemark ? 'textarea' : 'input');
  input.className = isRemark ? 'remark-input' : 'input';
  if (isRemark) input.rows = 3;
  input.value = current;
  if (!isRemark) input.placeholder = item.filename;
  el.replaceWith(input);
  input.focus();
  if (!isRemark) input.select();

  let done = false;
  const commit = async (save) => {
    if (done) return;
    done = true;
    if (save && input.value.trim() !== current) {
      try {
        await patch(`/api/library/demo/${id}`, { [field]: input.value });
        item[field] = input.value.trim() || null;
      } catch (err) {
        toast(t('lib.toast.saveFail', { msg: err.message }), 'error');
      }
    }
    render();
  };

  input.addEventListener('blur', () => commit(true));
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') { ev.preventDefault(); commit(false); }
    if (ev.key === 'Enter' && (!isRemark || ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      commit(true);
    }
  });
}

// ── 操作 ──────────────────────────────────────────────────────────────────

/**
 * 排队提取。`force` 为真时忽略已有的 .csz 重新解析 demo。
 *
 * 不传 force 的话，已经有 .csz 的记录只会从那个文件重算元数据，便宜但完全不看
 * 采样设置。所以改了采样方式之后必须用 force，否则会「成功」地什么也没变。
 */
async function extractMany(ids, force = false) {
  if (!ids.length) return;
  try {
    const r = await post('/api/library/extract', { ids, force });
    toast(t(force ? 'lib.toast.queuedReparse' : 'lib.toast.queued', { n: r.queued }), 'ok');
    openDrawer();
    startJobPolling();
    load();
  } catch (err) {
    toast(t('lib.toast.submitFail', { msg: err.message }), 'error');
  }
}

async function openInReplay(ids) {
  if (!ids.length) return;
  try {
    const r = await post('/api/library/open', { ids });
    if (r.queued) toast(t('lib.toast.notExtracted', { n: r.queued }), 'ok');
    window.location.href = r.replay_url;
  } catch (err) {
    toast(t('lib.toast.openFail', { msg: err.message }), 'error');
  }
}

// ── 删除 ─────────────────────────────────────────────────────────────────
//
// 单个删除和批量删除走同一个对话框：库记录一定删，原始 .dem 默认保留，
// 勾上「同时移走原始 .dem」才连文件一起处理。默认值必须是不动文件，
// 库记录扫描一次就回来了，删掉的 400 MB 录像不一定还能再下到。
//
// 之前页脚并排两个按钮（保留 / 移到回收站），把「删不删记录」和「动不动
// 文件」压成了一次二选一，看起来像两个都得点的并列选项，而不是一个默认
// 加一个例外。

/** 删除对话框当前等待的 resolve；关闭对话框时一定要调用它，否则 Promise 悬空。 */
let _deleteResolve = null;

function closeDeleteModal(value) {
  $('delete-modal').classList.add('hidden');
  $('btn-del-confirm').onclick = null;
  $('del-with-file').onchange = null;
  const resolve = _deleteResolve;
  _deleteResolve = null;
  if (resolve) resolve(value);
}

/** 弹出删除确认。resolve 勾选框状态（true 连 .dem 一起移走 / false 只删记录），取消或直接关掉是 null。 */
function askDelete(ids) {
  // 选择可以跨页（切换分页/筛选都不清空 S.selected），而 S.items 只有当前页，
  // 所以找得到的和找不到的必须分开算，否则体积和「已提取」数量会静默偏小；
  // 不在当前页的那几条按 id 单独列出，体积只按已知的估算。
  const known = ids.map((id) => S.items.find((d) => d.id === id)).filter(Boolean);
  const offPage = ids.filter((id) => !S.items.some((d) => d.id === id));
  const bytes = known.reduce((n, d) => n + (d.file_size || 0), 0);
  const gone = known.filter((d) => d.status === 'missing').length;
  const ready = known.filter((d) => d.status === 'ready').length;
  // 源文件全都已经不在磁盘上时勾选框没有意义，留着会让人以为「移走」还能做点什么。
  const fileGone = offPage.length === 0 && gone === ids.length;

  $('delete-title').textContent =
    ids.length === 1 ? t('lib.delete.titleOne') : t('lib.delete.titleMany', { n: ids.length });

  $('delete-body').innerHTML = `
    <div class="del-lead">${t('lib.delete.lead')}</div>
    <div class="del-list">
      <div class="del-row"><span class="k">${t('lib.delete.rawFile')}</span>
        <span class="v">${offPage.length ? '≥ ' : ''}${fmtSize(bytes)}</span></div>
    </div>
    ${ready ? `<div class="del-note warn">${t('lib.delete.noteReady', { n: ready })}</div>` : ''}
    ${gone ? `<div class="del-note">${t('lib.delete.noteGone', { n: gone })}</div>` : ''}
    ${offPage.length ? `<div class="del-note">${t('lib.delete.noteOffPage', { n: offPage.length })}</div>` : ''}
    <div class="del-names">${[
      ...known.map((d) => esc(d.display_name || d.filename)),
      ...offPage.map((id) => t('lib.delete.offPageItem', { id })),
    ].map((n) => `<div>${n}</div>`).join('')}</div>`;

  const box = $('del-with-file');
  const confirm = $('btn-del-confirm');
  // 每次打开都回到默认：不动文件。记住上次的勾选等于把危险选项变成默认值。
  box.checked = false;
  box.disabled = fileGone;
  $('del-with-file-label').classList.toggle('off', fileGone);
  const relabel = () => {
    confirm.textContent = box.checked
      ? t('lib.delete.confirmWithFile')
      : t('lib.delete.confirm');
  };
  relabel();

  $('delete-modal').classList.remove('hidden');
  confirm.focus();

  return new Promise((resolve) => {
    _deleteResolve = resolve;
    box.onchange = relabel;
    confirm.onclick = () => closeDeleteModal(box.checked);
  });
}

async function doDelete(ids, withFile) {
  let ok = 0, fail = 0, moved = 0, busy = 0;
  for (const id of ids) {
    try {
      const res = await api(`/api/library/demo/${id}?delete_file=${withFile ? 1 : 0}`,
                            { method: 'DELETE' });
      if (res && res.recycled) moved++;
      S.selected.delete(id);
      ok++;
    } catch (err) {
      fail++;
      if (/占用/.test(err.message)) busy++;
    }
  }
  const bits = [t('lib.delete.done', { n: ok })];
  if (moved) bits.push(t('lib.delete.moved', { n: moved }));
  if (fail) bits.push(t(busy ? 'lib.toast.failedBusy' : 'lib.toast.failed', { n: fail }));
  toast(bits.join(t('lib.toast.sep')), fail ? 'error' : 'ok');
  load();
}

async function deleteOne(id) {
  const withFile = await askDelete([id]);
  // 必须是 === null：false 是「只删记录」这个正常结果，不能被当成取消。
  if (withFile === null) return;
  await doDelete([id], withFile);
}

async function deleteSelected() {
  const ids = [...S.selected];
  if (!ids.length) return;
  const withFile = await askDelete(ids);
  if (withFile === null) return;
  await doDelete(ids, withFile);
}

// ── 存储管理 ─────────────────────────────────────────────────────────────
//
// 设置里的「存储管理」列出 app/data 下每个缓存目录占了多少，并允许整类清掉。
// 清 .csz / 缩略图 / 头像都要连库里的路径一起清，那半边在 app/storage.py，
// 这里只管显示和确认。
//
// 原始 .dem 不在 app/data 里，只是被库记录指向，所以那一行是只读的：删一场
// 录像是一次针对单个 demo 的决定，在库里做，不是清缓存。

let STORAGE = null;

function sizeText(bytes) {
  return bytes > 0 ? fmtSize(bytes) : '0 B';
}

function storageRowHTML(e) {
  const action = e.removable
    ? `<button class="btn-mini" data-clean="${esc(e.key)}">${t('lib.clean.action')}</button>`
    : '';
  return `
  <div class="storage-row">
    <div class="storage-main">
      <div class="storage-label">${esc(storageLabel(e))}</div>
      <div class="storage-note">${esc(storageNote(e))}</div>
    </div>
    <div class="storage-size">
      <span class="n">${sizeText(e.bytes)}</span>
      <span class="l">${e.files ? t('lib.storage.files', { n: e.files }) : t('lib.storage.empty')}</span>
    </div>
    ${action}
  </div>`;
}

function renderStorage(data) {
  STORAGE = data;
  $('storage-note').textContent = t('lib.storage.total', {
    size: sizeText(data.total_bytes), dir: data.data_dir,
  });
  const demos = data.demos || { count: 0, bytes: 0 };
  $('storage-list').innerHTML =
    (data.entries || []).map(storageRowHTML).join('') + `
  <div class="storage-row static">
    <div class="storage-main">
      <div class="storage-label">${t('lib.storage.rawDemos')}</div>
      <div class="storage-note">${t('lib.storage.rawDemosNote')}</div>
    </div>
    <div class="storage-size">
      <span class="n">${sizeText(demos.bytes)}</span>
      <span class="l">${t('lib.storage.count', { n: demos.count })}</span>
    </div>
  </div>`;
}

async function loadStorage() {
  try {
    renderStorage(await api('/api/library/storage'));
  } catch (err) {
    $('storage-note').textContent = t('lib.storage.statFail', { msg: err.message });
    $('storage-list').innerHTML = '';
  }
}

/** 清理确认；resolve true 表示确认，取消或直接关掉是 null。 */
let _cleanResolve = null;

function closeCleanModal(value) {
  $('clean-modal').classList.add('hidden');
  $('btn-clean-confirm').onclick = null;
  const resolve = _cleanResolve;
  _cleanResolve = null;
  if (resolve) resolve(value);
}

function askClean(key) {
  const entry = ((STORAGE && STORAGE.entries) || []).find((e) => e.key === key);
  if (!entry) return Promise.resolve(null);

  $('clean-title').textContent = t('lib.clean.titleWith', { name: storageLabel(entry) });
  $('clean-body').innerHTML = `
    <div class="del-lead">${t('lib.clean.lead', { n: entry.files, size: sizeText(entry.bytes) })}</div>
    <div class="del-list">
      <div class="del-row"><span class="k">${t('lib.clean.dir')}</span>
        <span class="v">${esc(entry.path)}</span></div>
      <div class="del-row"><span class="k">${t('lib.clean.after')}</span>
        <span class="v">${esc(storageNote(entry))}</span></div>
    </div>`;
  $('clean-modal').classList.remove('hidden');
  $('btn-clean-confirm').focus();

  return new Promise((resolve) => {
    _cleanResolve = resolve;
    $('btn-clean-confirm').onclick = () => closeCleanModal(true);
  });
}

async function cleanCache(key) {
  if (await askClean(key) !== true) return;
  try {
    const r = await post('/api/library/storage/clean', { targets: [key] });
    renderStorage(r.overview || {});
    const res = (r.cleaned || {})[key] || {};
    const failed = (res.errors || []).length;
    const tail = failed ? t('lib.toast.failedTail', { n: failed }) : '';
    toast(t('lib.clean.done', { n: res.files || 0, size: sizeText(res.bytes) }) + tail,
          failed ? 'error' : 'ok');
    // .csz 清掉后库里的状态变了，列表要跟着走。
    load();
  } catch (err) {
    toast(t('lib.toast.cleanFail', { msg: err.message }), 'error');
  }
}

async function scanNow() {
  const btn = $('btn-scan');
  btn.disabled = true;
  try {
    const r = await post('/api/library/watch/scan', { force: true });
    toast(t('lib.toast.scanStarted'));
    openDrawer();
    startJobPolling();
    pollJobUntilDone(r.job_id, (job) => {
      if (job.state === 'done') {
        const s = job.result || {};
        const found = t('lib.toast.scanFound', { scanned: s.scanned || 0, registered: s.registered || 0 });
        const extra = s.errors && s.errors.length ? t('lib.toast.failedTail', { n: s.errors.length }) : '';
        // 上次被中断、这次补上的
        const back = s.recovered ? t('lib.toast.scanRecovered', { n: s.recovered }) : '';
        // auto_extract 关掉时扫描只入库不提取，不说清楚用户会以为卡住了
        const tail = (s.new_ids && s.new_ids.length && !s.extracted)
          ? t('lib.toast.autoExtractOff')
          : '';
        toast(t('lib.toast.scanDone', { detail: `${found}${back}${extra}${tail}` }), 'ok');
        load();
      }
    });
  } catch (err) {
    toast(t('lib.toast.scanFail', { msg: err.message }), 'error');
  } finally {
    btn.disabled = false;
  }
}

// ── 任务抽屉 ──────────────────────────────────────────────────────────────

function openDrawer() { $('activity-drawer').classList.remove('hidden'); }

const JOB_STATE = {
  queued: [t('lib.job.queued'), 'var(--text-dim)'], running: [t('lib.job.running'), 'var(--cyan-bright)'],
  done: [t('lib.job.done'), 'var(--green)'], error: [t('lib.job.error'), 'var(--red)'], cancelled: [t('lib.job.cancelled'), 'var(--orange)'],
};

function renderJobs(jobs) {
  const list = $('job-list');
  if (!jobs.length) {
    list.innerHTML = `<div class="dim" style="font-size:12.5px">${t('lib.job.empty')}</div>`;
    return;
  }
  list.innerHTML = jobs.map((j) => {
    const [label, color] = JOB_STATE[j.state] || [j.state, 'var(--text-dim)'];
    return `<div class="job ${j.state === 'error' ? 'err' : ''}">
      <div class="job-top">
        <span class="job-label">${esc(j.label)}</span>
        <span class="job-state" style="color:${color}">${label}</span>
        <span class="dim" style="font-size:11px">${j.elapsed}s</span>
      </div>
      <div class="job-step">${esc(j.error || j.step)}</div>
      ${j.progress != null && j.state === 'running'
        ? `<div class="job-bar"><i style="width:${Math.round(j.progress * 100)}%"></i></div>` : ''}
    </div>`;
  }).join('');
}

async function refreshJobs() {
  try {
    const r = await api('/api/library/jobs');
    renderJobs(r.jobs || []);
    const pending = r.pending || 0;
    const badge = $('job-badge');
    badge.textContent = pending;
    badge.classList.toggle('hidden', pending === 0);

    // 有任务完成时刷新列表（签名变化 = 状态变了）。
    const sig = (r.jobs || []).map((j) => `${j.id}:${j.state}`).join('|');
    if (S.lastJobSig && sig !== S.lastJobSig) {
      const finished = (r.jobs || []).some((j) =>
        (j.state === 'done' || j.state === 'error'));
      if (finished) load({ keepSelection: true });
    }
    S.lastJobSig = sig;

    if (pending === 0) stopJobPolling();
    return r;
  } catch { return { jobs: [], pending: 0 }; }
}

function startJobPolling() {
  if (S.jobPoll) return;
  S.jobPoll = setInterval(refreshJobs, 1500);
  refreshJobs();
}

function stopJobPolling() {
  clearInterval(S.jobPoll);
  S.jobPoll = null;
}

function pollJobUntilDone(jobId, onEach) {
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/library/jobs/${jobId}`);
      onEach && onEach(job);
      if (['done', 'error', 'cancelled'].includes(job.state)) clearInterval(timer);
    } catch { clearInterval(timer); }
  }, 1200);
}

// ── 仓库里缺的东西：AI 权重 / 地图几何 ───────────────────────────────────────
//
// 干净检出之后有两样东西一定不在：
//
//   models/                     两份 .pt 权重，665 MB，不进 git（要下载）
//   maps/optimized_obj_files/   仓库只带 .obj.zip，解压后 943 MB（要解压）
//
// 两件事的形状一模一样，所以只有一份实现：查本地状态，缺了就显形，按钮排一
// 个后台任务，轮询到结束，再查一遍。进度不另造接口：任务就在 jobs 队列里，
// 顶栏「任务」抽屉看到的是同一个。
//
//   GET  /api/ai/models             POST /api/ai/models/download      下载权重
//   GET  /api/library/maps          POST /api/library/maps/extract    解压地图
//
// "缺没缺"一律由后端判断（路径表 / zip 清单都在 Python 那边），这里不抄一份，
// 抄了换版本就会漂移。

/**
 * 一条"仓库里缺文件"的横幅。
 *
 * cfg:
 *   ids        五个元素的 id（横幅 / 说明 / 进度条 / 进度条填充 / 按钮）
 *   jobKey     状态对象里放最近那个任务的字段名（`download` / `extract`）
 *   loadState  () => Promise<状态>，状态里有 ok / missing
 *   submitJob  () => Promise<{job_id}>，排一个后台任务
 *   describe   状态 => 一句"缺什么"
 *   texts      { running, queued, preparing, done, failed } 五个 i18n 键
 */
function setupBanner(cfg) {
  const st = { state: null, job: null };
  const el = (key) => $(cfg.ids[key]);

  function render() {
    const box = el('box');
    if (!box) return;
    const state = st.state || { ok: true, missing: [] };
    const job = state[cfg.jobKey];
    const running = Boolean(job && (job.state === 'running' || job.state === 'queued'));
    const failed = Boolean(job && job.state === 'error' && !state.ok);
    const missing = state.missing || [];

    // 齐了、也没在跑、上次也没失败 → 整条横幅不出现
    box.classList.toggle('hidden', !missing.length && !running && !failed);

    let detail;
    if (running) {
      const pct = job.progress != null ? ` ${Math.round(job.progress * 100)}%` : '';
      // 后端 step 里已经带了"第几个文件 / 处理到多少字节"，直接显示，不重复翻译
      detail = `${t(cfg.texts.running)}${pct} · ${job.step || t(cfg.texts.queued)}`;
    } else if (failed) {
      detail = t(cfg.texts.failed, { msg: job.error || job.step || '' });
    } else {
      detail = cfg.describe(state);
    }
    el('detail').textContent = detail;

    const bar = el('bar');
    const showBar = Boolean(running && job.progress != null);
    bar.classList.toggle('hidden', !showBar);
    if (showBar) el('fill').style.width = `${Math.round(job.progress * 100)}%`;

    el('btn').disabled = running;
  }

  async function load() {
    try {
      st.state = await cfg.loadState();
    } catch {
      return;   // 老版本后端 / 网络抖动：横幅不出现，页面照常用
    }
    render();
    const job = st.state && st.state[cfg.jobKey];
    if (job && (job.state === 'running' || job.state === 'queued')) watch(job.id);
  }

  function watch(jobId) {
    if (st.job === jobId) return;   // 已经有人在看这个任务了
    st.job = jobId;
    pollJobUntilDone(jobId, (job) => {
      if (st.state) st.state[cfg.jobKey] = job;
      render();
      if (!['done', 'error', 'cancelled'].includes(job.state)) return;
      st.job = null;
      if (job.state === 'done') toast(t(cfg.texts.done), 'ok');
      else if (job.state === 'error') {
        toast(t(cfg.texts.failed, { msg: job.error || job.step }), 'error');
      }
      load();   // 再查一遍本地状态：横幅该消失就消失，该留着报错就留着
    });
  }

  async function submit() {
    const btn = el('btn');
    btn.disabled = true;
    // 先给一个"已经在动"的反馈：排进队列到真正开始之间还有几秒
    if (st.state) {
      st.state[cfg.jobKey] = {
        state: 'queued', step: t(cfg.texts.preparing), progress: null,
      };
      render();
    }
    try {
      const r = await cfg.submitJob();
      if (r.job_id) {
        watch(r.job_id);
        startJobPolling();   // 顶栏的任务角标 / 抽屉同步动起来
      } else {
        await load();        // 其实已经齐了
      }
    } catch (err) {
      btn.disabled = false;
      toast(t(cfg.texts.failed, { msg: err.message }), 'error');
      load();
    }
  }

  function bind() {
    const btn = el('btn');
    if (btn) btn.onclick = submit;
  }

  return { load, render, watch, submit, bind };
}

//: 缺 AI 权重（`app/ai/models.py::PINNED_WEIGHTS` 里那两份 .pt）。
const MODEL_BANNER = setupBanner({
  ids: {
    box: 'model-banner', detail: 'model-detail', bar: 'model-bar',
    fill: 'model-bar-fill', btn: 'btn-download-models',
  },
  jobKey: 'download',
  loadState: () => api('/api/ai/models'),
  submitJob: () => post('/api/ai/models/download'),
  describe: (state) => t('lib.models.missing', { n: (state.missing || []).length }),
  texts: {
    running: 'lib.models.running',
    queued: 'lib.models.queued',
    preparing: 'lib.models.preparing',
    done: 'lib.models.done',
    failed: 'lib.models.failed',
  },
});

//: 缺地图几何（`maps/optimized_obj_files/*.obj.zip` 还没解压）。
const MAP_BANNER = setupBanner({
  ids: {
    box: 'map-banner', detail: 'map-detail', bar: 'map-bar',
    fill: 'map-bar-fill', btn: 'btn-extract-maps',
  },
  jobKey: 'extract',
  loadState: () => api('/api/library/maps'),
  submitJob: () => post('/api/library/maps/extract'),
  describe: (state) => t('lib.maps.missing', {
    n: (state.missing || []).length,
    // 只报"还没解压的那部分"，和上面"缺几个文件"是同一句话的口径
    size: sizeText(Math.max(0, (state.unpacked_bytes || 0) - (state.present_bytes || 0))),
  }),
  texts: {
    running: 'lib.maps.running',
    queued: 'lib.maps.queued',
    preparing: 'lib.maps.preparing',
    done: 'lib.maps.done',
    failed: 'lib.maps.failed',
  },
});

// ── 设置弹窗 ──────────────────────────────────────────────────────────────

let CFG = null;

async function openSettings() {
  try {
    CFG = await loadConfig();
  } catch (err) {
    return toast(t('lib.toast.settingsFail', { msg: err.message }), 'error');
  }
  renderRoots();
  $('watched-players').value = (CFG.watched_players || []).join('\n');
  $('watch-enabled').checked = CFG.watch_enabled !== false;
  $('watch-interval').value = CFG.watch_interval_sec ?? 5;
  const s = CFG.sampling || {};
  $('sampling-mode').value = s.mode || 'tick';
  $('sampling-interval').value = s.interval ?? 0.25;
  $('sampling-stride').value = s.stride ?? 8;
  $('auto-extract').checked = CFG.auto_extract !== false;
  $('make-thumbnails').checked = CFG.make_thumbnails !== false;
  $('fetch-avatars').checked = CFG.fetch_avatars !== false;
  $('avatar-ttl').value = CFG.avatar_ttl_days ?? 7;
  renderAvatarStat(CFG.avatars || {});
  syncSamplingRows();
  $('settings-msg').textContent = '';
  $('settings-modal').classList.remove('hidden');
  loadSuggestions();
  loadStorage();
}

function renderRoots() {
  const roots = CFG.watch_roots || [];
  $('watch-roots').innerHTML = roots.length ? roots.map((r, i) => `
    <div class="root-item ${r.enabled === false ? 'off' : ''}">
      <input type="checkbox" data-root-toggle="${i}" ${r.enabled === false ? '' : 'checked'}>
      <span class="rp">${esc(r.path)}</span>
      <span class="rd">${esc(sourceText(r.source || '本地'))} · ${t('lib.settings.depthSuffix', { n: r.depth ?? 2 })}</span>
      <button class="btn-mini" data-root-del="${i}">${t('common.action.remove')}</button>
    </div>`).join('')
    : `<div class="dim" style="font-size:12px">${t('lib.settings.noRoots')}</div>`;
}

async function loadSuggestions() {
  const box = $('suggest-list');
  try {
    const r = await api('/api/library/watch/suggest');
    const s = r.suggestions || [];
    const have = new Set((CFG.watch_roots || []).map((x) => x.path));
    const fresh = s.filter((x) => !have.has(x.path));
    box.innerHTML = fresh.length ? fresh.map((x) => `
      <div class="suggest-item">
        <span class="sc" style="color:var(--cyan-bright)">${esc(sourceText(x.source))}</span>
        <span class="sp">${esc(x.path)}</span>
        <span class="sc">${t('lib.settings.suggestCount', { n: x.demo_count })}${x.zip_count ? t('lib.settings.suggestZip', { n: x.zip_count }) : ''}</span>
        <button class="btn-mini" data-suggest-add="${esc(x.path)}" data-suggest-src="${esc(x.source)}">${t('common.action.add')}</button>
      </div>`).join('')
      : `<span class="dim" style="font-size:12px">${t('lib.settings.noSuggestions')}</span>`;
  } catch {
    box.innerHTML = `<span class="dim" style="font-size:12px">${t('lib.settings.suggestFail')}</span>`;
  }
}

function syncSamplingRows() {
  const mode = $('sampling-mode').value;
  $('row-interval').style.display = mode === 'interval' ? '' : 'none';
  $('row-stride').style.display = mode === 'stride' ? '' : 'none';
}

function renderAvatarStat(stats) {
  const known = stats.known || 0;
  if (!known) {
    $('avatar-cache-stat').textContent = t('lib.settings.noAvatarCache');
    return;
  }
  const failed = stats.error ? t('lib.settings.avatarFailTail', { n: stats.error }) : '';
  $('avatar-cache-stat').textContent =
    t('lib.settings.avatarCached', { ok: stats.ok || 0, total: known }) + failed;
}

async function addRoot(path, source, depth) {
  const msg = $('new-root-msg');
  msg.className = 'inline-msg';
  msg.textContent = t('lib.settings.checking');
  try {
    const info = await post('/api/library/watch/inspect', { path });
    if (!info.valid) {
      msg.className = 'inline-msg err';
      msg.textContent = info.error;
      return;
    }
    CFG.watch_roots = [...(CFG.watch_roots || []), {
      path: info.path, depth: depth ?? 2,
      source: source || '本地', enabled: true,
    }];
    msg.className = 'inline-msg ok';
    msg.textContent = t('lib.settings.rootAdded', { demos: info.demo_count, zips: info.zip_count });
    renderRoots();
    loadSuggestions();
  } catch (err) {
    msg.className = 'inline-msg err';
    msg.textContent = err.message;
  }
}

async function saveSettings() {
  const btn = $('btn-save-settings');
  btn.disabled = true;
  const msg = $('settings-msg');
  try {
    await put('/api/library/watch', {
      roots: CFG.watch_roots || [],
      enabled: $('watch-enabled').checked,
      interval: Number($('watch-interval').value) || 5,
    });
    const saved = await put('/api/library/config', {
      watched_players: $('watched-players').value
        .split(/[\n,]/).map((s) => s.trim()).filter(Boolean),
      sampling: {
        mode: $('sampling-mode').value,
        interval: Number($('sampling-interval').value) || 0.25,
        stride: Number($('sampling-stride').value) || 8,
      },
      auto_extract: $('auto-extract').checked,
      make_thumbnails: $('make-thumbnails').checked,
      fetch_avatars: $('fetch-avatars').checked,
      avatar_ttl_days: Number($('avatar-ttl').value) || 7,
    });
    msg.className = 'inline-msg ok';
    msg.textContent = t('lib.settings.saved');
    $('settings-modal').classList.add('hidden');
    // 服务端已经按新名单重算了每条记录的 watched，卡片高亮立刻就是对的。
    const resynced = (saved && saved.watched_resynced) || {};
    toast(resynced.changed
      ? t('lib.settings.savedWatched', { n: resynced.changed })
      : t('lib.settings.savedToast'), 'ok');
    load();
  } catch (err) {
    msg.className = 'inline-msg err';
    msg.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

// ── 事件绑定 ──────────────────────────────────────────────────────────────

function bindEvents() {
  $('view-grid').onclick = () => { S.view = 'grid'; localStorage.setItem('csnet.view', 'grid'); render(); };
  $('view-list').onclick = () => { S.view = 'list'; localStorage.setItem('csnet.view', 'list'); render(); };

  let searchTimer = null;
  $('search').oninput = (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { S.q = e.target.value.trim(); S.offset = 0; load(); }, 250);
  };

  $('sort').onchange = (e) => { S.sort = e.target.value; S.offset = 0; load(); };
  $('filter-status').onchange = (e) => { S.status = e.target.value; S.offset = 0; load(); };
  $('filter-source').onchange = (e) => { S.source = e.target.value; S.offset = 0; load(); };
  $('filter-map').onchange = (e) => { S.map = e.target.value; S.offset = 0; load(); };
  $('filter-watched').onchange = (e) => { S.watchedOnly = e.target.checked; S.offset = 0; load(); };

  $('btn-clear-filters').onclick = () => {
    S.q = ''; S.status = ''; S.source = ''; S.map = ''; S.watchedOnly = false; S.offset = 0;
    $('search').value = ''; $('filter-status').value = '';
    $('filter-source').value = ''; $('filter-map').value = '';
    $('filter-watched').checked = false;
    load();
  };

  $('btn-select-page').onclick = () => {
    const all = S.items.every((d) => S.selected.has(d.id));
    S.items.forEach((d) => all ? S.selected.delete(d.id) : S.selected.add(d.id));
    render();
  };
  $('btn-clear-selection').onclick = () => { S.selected.clear(); render(); };
  $('btn-extract-selected').onclick = () => extractMany([...S.selected]);
  $('btn-reextract-selected').onclick = () => extractMany([...S.selected], true);
  $('btn-load-analysis').onclick = () => openInReplay([...S.selected]);
  $('btn-delete-selected').onclick = deleteSelected;

  $('btn-scan').onclick = scanNow;
  $('btn-settings').onclick = openSettings;
  MODEL_BANNER.bind();
  MAP_BANNER.bind();
  $('btn-activity').onclick = () => {
    $('activity-drawer').classList.remove('hidden');
    startJobPolling();
  };

  document.querySelectorAll('[data-close]').forEach((btn) => {
    btn.onclick = () => {
      const id = btn.dataset.close;
      // 两个确认框的关闭必须走各自的 close，否则等待中的 Promise 永远不 resolve。
      if (id === 'delete-modal') return closeDeleteModal(null);
      if (id === 'clean-modal') return closeCleanModal(null);
      $(id).classList.add('hidden');
    };
  });
  document.querySelectorAll('.modal').forEach((m) => {
    m.addEventListener('click', (ev) => {
      if (ev.target !== m) return;
      if (m.id === 'delete-modal') return closeDeleteModal(null);
      if (m.id === 'clean-modal') return closeCleanModal(null);
      m.classList.add('hidden');
    });
  });

  $('btn-save-settings').onclick = saveSettings;
  $('btn-add-root').onclick = () => {
    const path = $('new-root-path').value.trim();
    if (!path) return toast(t('lib.toast.pastePath'), 'error');
    addRoot(path, null, Number($('new-root-depth').value));
    $('new-root-path').value = '';
  };
  $('new-root-path').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') $('btn-add-root').click();
  });

  $('watch-roots').addEventListener('click', (ev) => {
    const del = ev.target.closest('[data-root-del]');
    if (del) {
      CFG.watch_roots.splice(Number(del.dataset.rootDel), 1);
      renderRoots();
      $('settings-msg').textContent = t('lib.settings.rememberSave');
      return;
    }
    const toggle = ev.target.closest('[data-root-toggle]');
    if (toggle) {
      CFG.watch_roots[Number(toggle.dataset.rootToggle)].enabled = toggle.checked;
      $('settings-msg').textContent = t('lib.settings.rememberSave');
    }
  });

  $('suggest-list').addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-suggest-add]');
    if (btn) addRoot(btn.dataset.suggestAdd, btn.dataset.suggestSrc, 2);
  });

  $('storage-list').addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-clean]');
    if (btn) cleanCache(btn.dataset.clean);
  });

  $('sampling-mode').onchange = syncSamplingRows;
  $('btn-clear-jobs').onclick = async () => {
    await post('/api/library/jobs/clear');
    refreshJobs();
  };

  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      closeDeleteModal(null);   // 没打开时是空操作
      closeCleanModal(null);
      document.querySelectorAll('.modal').forEach((m) => m.classList.add('hidden'));
      $('activity-drawer').classList.add('hidden');
    }
  });
}

// ── 启动 ──────────────────────────────────────────────────────────────────

async function init() {
  bindContent();
  bindEvents();
  $('sort').value = S.sort;
  await load();
  // 仓库里缺的东西（AI 权重 / 地图几何）各问一次本地状态，缺了就出横幅
  await Promise.all([MODEL_BANNER.load(), MAP_BANNER.load()]);
  // 打开页面时后台可能还有任务在跑（比如上次的提取），把轮询接上
  const jobs = await refreshJobs();
  if (jobs.pending > 0) startJobPolling();
}

init();
