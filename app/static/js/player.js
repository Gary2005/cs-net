/**
 * cs-net — 玩家档案。
 *
 * SteamID 从 <body data-steamid> 读，页面本身是空壳，数据全部来自
 * /api/library/players/<steamid> 和 /api/library/list?steamid=...。
 *
 * 比赛列表就是库列表换了个筛选条件，所以库那边的排序、搜索在这里一样能用。
 */

// 中英文案（四个页面共用 visualizer/static 下的那一份）。
import { t } from '/static/js/i18n/index.js';
// 悬停解释（和回放页的统计表共用一份）
import { mountTips, tipTh } from '/static/js/tips.js';
// NDJSON 流（统计端点边走边报进度）
import { streamJson } from '/static/js/ndjson.js';

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
const patch = (p, body) => api(p, { method: 'PATCH', body: JSON.stringify(body || {}) });

function toast(msg, kind = '') {
  const el = $('toast');
  el.textContent = msg;
  el.className = `toast ${kind}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), kind === 'error' ? 8000 : 4000);
}

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
  return m >= 60
    ? `${Math.floor(m / 60)}h${m % 60}m`
    : t('player.common.minutes', { n: m });
}

const SID = document.body.dataset.steamid || '';
const S = {
  q: '', limit: 25, offset: 0, total: 0, items: [],
  detail: null, avatar: null, watched: false, jobPoll: null, nameHistory: [],
};

// ── 头部 ──────────────────────────────────────────────────────────────────

/** 标题：Steam 昵称优先，没有就退回游戏内名字。 */
function displayName(p) {
  return p.persona || p.name || t('player.common.unknown');
}

/** 游戏里用过的名字（最近的在最前面），去掉和标题重复的那个。 */
function otherNames(p) {
  const title = displayName(p).toLowerCase();
  return (p.names || []).filter((n) => n && n.toLowerCase() !== title);
}

function avatarHTML(cls, label) {
  const av = S.avatar;
  if (av && av.url) {
    return `<img class="avatar ${cls}" src="${esc(av.url)}" alt="">`;
  }
  const ch = String(label || (S.detail && S.detail.name) || '?').trim().charAt(0) || '?';
  return `<span class="avatar-fallback ${cls}">${esc(ch)}</span>`;
}

function renderHead() {
  const p = S.detail || {};
  // Steam 昵称是这个人的账号名，游戏内名字是他在 demo 里用的；标题用前者，
  // 后者列在下面。没有缓存过 Steam 昵称时退回游戏内名字。
  const title = displayName(p);
  const av = S.avatar || {};
  const others = otherNames(p);

  $('profile-head').innerHTML = `
    ${avatarHTML('av-96', title)}
    <div class="phead-main">
      <div class="phead-name">
        <span>${esc(title)}</span>
        ${S.watched ? '<span class="star">★</span>' : ''}
      </div>
      ${others.length ? `<div class="phead-persona">${t('player.common.inGameName')} ${esc(others.join(t('player.common.nameSep')))}</div>` : ''}
      <div class="phead-id">SteamID ${esc(SID)}</div>
      ${av.error ? `<div class="phead-persona">${t('player.profile.avatarError', { msg: esc(av.error) })}</div>` : ''}
    </div>`;

  $('btn-watch').textContent = S.watched
    ? t('player.common.watch.remove')
    : t('player.common.watch.add');
}

function renderStats() {
  const p = S.detail || {};
  const win = p.win_pct == null ? 0 : p.win_pct;
  const kd = p.kd == null ? 0 : p.kd;
  const tiles = [
    [t('player.common.matches'), p.matches, ''],
    [t('player.common.winRate'), `${win}%`, win >= 50 ? 'good' : 'bad'],
    ['K/D', kd, kd >= 1 ? 'good' : 'bad'],
    ['KPR', p.kpr, ''],
    ['ADR', p.adr, ''],
    [t('player.profile.stats.headshot'), `${p.hs_pct}%`, ''],
    [t('player.profile.stats.maps'), p.map_count, ''],
    [t('player.profile.stats.last'), (p.last_seen || '').slice(0, 10) || '—', ''],
  ];
  $('profile-stats').innerHTML = tiles.map(([label, value, cls]) =>
    `<div class="tile ${cls}"><div class="v">${esc(value)}</div><div class="l">${label}</div></div>`
  ).join('');
}

function renderNames() {
  const rows = S.nameHistory || [];
  if (!rows.length) {
    $('name-table').innerHTML =
      `<div class="dim" style="font-size:12.5px">${t('player.common.noData')}</div>`;
    return;
  }
  $('name-table').innerHTML = `
  <table class="mtable">
    <thead><tr><th>${t('player.profile.names.col.name')}</th><th>${t('player.common.matches')}</th><th>${t('player.common.lastSeen')}</th></tr></thead>
    <tbody>${rows.map((n) => `
      <tr>
        <td class="name">${esc(n.name)}</td>
        <td>${n.matches}</td>
        <td>${esc((n.last_seen || '').slice(0, 10) || '—')}</td>
      </tr>`).join('')}
    </tbody>
  </table>`;
}

function renderMaps() {
  const maps = S.detail && S.detail.maps ? S.detail.maps : [];
  if (!maps.length) {
    $('map-table').innerHTML =
      `<div class="dim" style="font-size:12.5px">${t('player.common.noData')}</div>`;
    return;
  }
  $('map-table').innerHTML = `
  <table class="mtable">
    <thead><tr>
      <th>${t('player.profile.maps.col.map')}</th><th>${t('player.common.matches')}</th><th>${t('player.common.winRate')}</th><th>K</th><th>D</th><th>A</th><th>K/D</th><th>ADR</th>
    </tr></thead>
    <tbody>${maps.map((m) => `
      <tr>
        <td class="name">${esc(m.map_name || t('player.common.unknownMap'))}</td>
        <td>${m.matches}</td>
        <td>${m.win_pct}%</td>
        <td>${m.kills}</td>
        <td>${m.deaths}</td>
        <td>${m.assists}</td>
        <td>${m.kd}</td>
        <td>${m.adr}</td>
      </tr>`).join('')}
    </tbody>
  </table>`;
}

// ── 比赛列表 ──────────────────────────────────────────────────────────────

function statFor(d) {
  const rows = d.scoreboard || [];
  return rows.find((r) => String(r.steamid) === SID) || null;
}

function matchHTML(d) {
  const st = statFor(d);
  const title = d.display_name || d.filename;
  const a = d.team_a_score, b = d.team_b_score;
  const score = a == null ? 'vs' : `${a} : ${b}`;
  const thumb = d.thumb_path
    ? `<img src="/api/library/thumb/${d.id}" alt="" loading="lazy">` : '';
  const line = st
    ? `${st.kills} / ${st.deaths} / ${st.assists}`
    : t('player.profile.match.noStats');
  const adr = st && d.total_rounds ? (st.adr_damage / d.total_rounds).toFixed(1) : null;

  return `
  <div class="mrow" data-id="${d.id}">
    <div class="mrow-thumb">${thumb}</div>
    <div class="mrow-main">
      <div class="mrow-name" title="${esc(title)}">${esc(title)}</div>
      <div class="mrow-sub">
        ${esc(d.team_a_name || 'Team A')} ${score} ${esc(d.team_b_name || 'Team B')}
        · ${esc(d.map_name || '—')} · ${fmtDuration(d.duration_mins)} · ${fmtSize(d.file_size)}
      </div>
    </div>
    <div class="mrow-stat">
      <span class="k">${line}</span>
      ${adr ? `<span class="adr">ADR ${adr}</span>` : ''}
    </div>
    <div class="mrow-actions">
      <button class="btn-mini" data-act="open" data-id="${d.id}">▶ ${t('player.profile.analysis')}</button>
    </div>
  </div>`;
}

function renderDemos() {
  $('demo-count').textContent = S.total
    ? t('player.common.matchCount', { n: S.total })
    : '';
  if (!S.items.length) {
    $('demo-list').innerHTML =
      `<div class="dim" style="font-size:12.5px">${t('player.profile.demos.empty')}</div>`;
  } else {
    $('demo-list').innerHTML = S.items.map(matchHTML).join('');
  }
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
    <button ${page >= pages ? 'disabled' : ''} data-page="last">${t('player.common.pager.last')} »</button>`;
}

function demoQuery() {
  const p = new URLSearchParams();
  p.set('steamid', SID);
  if (S.q) p.set('q', S.q);
  p.set('sort', 'date');
  p.set('limit', S.limit);
  p.set('offset', S.offset);
  return p.toString();
}

async function loadDemos() {
  try {
    const r = await api(`/api/library/list?${demoQuery()}`);
    S.items = r.items || [];
    S.total = r.total || 0;
    renderDemos();
  } catch (err) {
    toast(t('player.profile.toast.demosFail', { msg: err.message }), 'error');
  }
}

async function loadPlayer() {
  try {
    const r = await api(`/api/library/players/${SID}`);
    S.detail = { ...r.player, maps: r.maps };
    S.nameHistory = r.name_history || [];
    S.avatar = r.avatar;
    S.watched = !!r.player.watched;
    renderHead();
    renderStats();
    renderNames();
    renderMaps();
  } catch (err) {
    $('profile-head').innerHTML =
      `<div class="phead-main"><div class="phead-name">${t('player.profile.notFound')}</div>
       <div class="phead-persona">${esc(err.message)}</div></div>`;
    $('profile-stats').innerHTML = '';
    $('name-table').innerHTML = '';
    $('map-table').innerHTML = '';
    $('btn-watch').disabled = true;
    $('btn-refresh-avatar').disabled = true;
  }
}

// ── 操作 ──────────────────────────────────────────────────────────────────

async function openInReplay(ids) {
  if (!ids.length) return;
  try {
    const r = await post('/api/library/open', { ids });
    if (r.queued) toast(t('player.profile.toast.queued', { n: r.queued }), 'ok');
    window.location.href = r.replay_url;
  } catch (err) {
    toast(t('player.common.toast.openFail', { msg: err.message }), 'error');
  }
}

async function loadAll() {
  const btn = $('btn-load-all');
  btn.disabled = true;
  try {
    // 回放批次最多收 100 个，所以这里按同样上限取一遍 id，顺序和列表一致。
    const r = await api(`/api/library/list?steamid=${SID}&sort=date&limit=100&offset=0`);
    const ids = (r.items || []).map((d) => d.id);
    if (!ids.length) return toast(t('player.profile.toast.noDemos'), 'error');
    if (r.total > ids.length) {
      toast(t('player.profile.toast.loadAllTotal', { n: r.total })
        + t('player.profile.toast.loadAllFirst', { n: ids.length }));
    }
    await openInReplay(ids);
  } catch (err) {
    toast(t('player.profile.toast.readFail', { msg: err.message }), 'error');
  } finally {
    btn.disabled = false;
  }
}

async function toggleWatch() {
  try {
    const r = await patch(`/api/library/players/${SID}`, { watched: !S.watched });
    S.watched = !!r.watched;
    renderHead();
    // 服务端同时重算了每条记录的 watched，库里的卡片高亮立刻跟着变。
    const changed = ((r.watched_resynced || {}).changed) || 0;
    const head = S.watched
      ? t('player.common.watch.added')
      : t('player.common.watch.removed');
    toast(changed
      ? head + t('player.common.watch.highlights', { n: changed })
      : head, 'ok');
  } catch (err) {
    toast(t('player.common.toast.submitFail', { msg: err.message }), 'error');
  }
}

function pollAvatarJob(jobId) {
  clearInterval(S.jobPoll);
  S.jobPoll = setInterval(async () => {
    try {
      const job = await api(`/api/library/jobs/${jobId}`);
      $('avatar-stat').textContent = job.state === 'running' ? job.step : '';
      if (!['done', 'error', 'cancelled'].includes(job.state)) return;
      clearInterval(S.jobPoll);
      S.jobPoll = null;
      $('avatar-stat').textContent = '';
      if (job.state === 'error') {
        return toast(t('player.profile.toast.avatarFail', { msg: job.error }), 'error');
      }
      const res = job.result || {};
      if (res.ok) toast(t('player.profile.toast.avatarUpdated'), 'ok');
      else toast((res.errors && res.errors[0]) || t('player.profile.toast.avatarMissing'), 'error');
      loadPlayer();
    } catch {
      clearInterval(S.jobPoll);
      S.jobPoll = null;
    }
  }, 1000);
}

async function refreshAvatar() {
  const btn = $('btn-refresh-avatar');
  btn.disabled = true;
  try {
    const r = await post('/api/library/avatars/sync', { steamids: [SID], force: true });
    if (r.job_id) pollAvatarJob(r.job_id);
    else toast(r.note || t('player.common.toast.syncRunning'));
  } catch (err) {
    toast(t('player.common.toast.submitFail', { msg: err.message }), 'error');
  } finally {
    btn.disabled = false;
  }
}

// 数据统计：传统在页面加载后自动拉，AI 点了页签才拉（要跑模型）。
// 两条都走 NDJSON 流报进度；列名和解释文案复用 `replay.stats.*`。

const PS = {
  tab: 'trad',
  trad: null, tradErr: null, tradLoading: false, tradPct: 0, tradStage: '',
  ai: null, aiErr: null, aiLoading: false, aiPct: 0, aiStage: '',
  seq: 0,
};

/** 比例 → 百分比文本（1 位小数，和 AI 那一栏一致）；没有样本时给 —。 */
function pctText(num, den) {
  if (!den) return '—';
  return `${(Math.round((num * 1000) / den) / 10).toFixed(1)}%`;
}

/** 比例单元格：百分比 + 悬停样本数（和 AI 那一栏一样）。 */
function rateCell(num, den) {
  return `<span data-tip="${esc(t('replay.stats.ai.samples', { n: den || 0 }))}">`
    + `${pctText(num, den)}</span>`;
}

function psProgress(label, pct, stage) {
  return `<div class="pstats-loading">
    <div class="pstats-progress"><div class="pstats-progress-fill" style="width:${(pct * 100).toFixed(1)}%"></div></div>
    <div class="pstats-progress-text"><span>${esc(label)}</span><span class="dim">${Math.round(pct * 100)}%</span></div>
    <div class="pstats-progress-text dim">${esc(stage || '')}</div>
  </div>`;
}

function psError(msg, retryTab) {
  return `<div class="pstats-error">${esc(msg)}
    <button class="btn-glass" data-ps-retry="${retryTab}">${t('player.stats.retry')}</button></div>`;
}

/** 一块「大数字 + 小标题」的卡片；`tip` 是悬停解释的 i18n 键，`samples` 是样本数。 */
function psTile(value, label, tip, samples) {
  const text = tip
    ? [t(tip), samples == null ? '' : t('replay.stats.ai.samples', { n: samples })]
      .filter(Boolean).join(' · ')
    : '';
  return `<div class="tile${text ? ' has-tip' : ''}"${text ? ` data-tip="${esc(text)}"` : ''}>
    <div class="v">${esc(value)}</div>
    <div class="l">${esc(label)}</div>
  </div>`;
}

function tradHTML(d) {
  const tot = d.totals || {};
  const noSmoke = d.smoke_ok === false;
  const tiles = [
    psTile(pctText(tot.first_kills, tot.rounds), t('player.stats.col.firstKillRate'),
      'player.stats.tip.firstKillRate', tot.rounds),
    psTile(pctText(tot.first_deaths, tot.rounds), t('player.stats.col.firstDeathRate'),
      'player.stats.tip.firstDeathRate', tot.rounds),
    psTile(pctText(tot.openers, tot.duels), t('replay.stats.col.opener'),
      'replay.stats.tip.opener', tot.duels),
    psTile(pctText(tot.side_kills, tot.enemy_kills), t('replay.stats.col.sideKill'),
      'replay.stats.tip.sideKill', tot.enemy_kills),
    psTile(pctText(tot.side_deaths, tot.enemy_deaths), t('replay.stats.col.sideDeath'),
      'replay.stats.tip.sideDeath', tot.enemy_deaths),
    noSmoke ? psTile('—', t('replay.stats.col.smokeKill'), null)
      : psTile(pctText(tot.smoke_kills, tot.enemy_kills), t('replay.stats.col.smokeKill'),
        'replay.stats.tip.smokeKill', tot.enemy_kills),
  ];

  const head = [
    t('player.stats.col.match'), t('player.profile.maps.col.map'), t('player.stats.col.rounds'),
    'K', 'D', 'A', 'K/D', 'ADR',
    tipTh(t('player.stats.col.firstKillRate'), 'player.stats.tip.firstKillRate'),
    tipTh(t('player.stats.col.firstDeathRate'), 'player.stats.tip.firstDeathRate'),
    tipTh(t('replay.stats.col.opener'), 'replay.stats.tip.opener'),
    tipTh(t('replay.stats.col.sideKill'), 'replay.stats.tip.sideKill'),
    tipTh(t('replay.stats.col.sideDeath'), 'replay.stats.tip.sideDeath'),
    tipTh(t('replay.stats.col.smokeKill'), 'replay.stats.tip.smokeKill'),
  ].map((h, i) => (i >= 8 ? h : `<th>${esc(h)}</th>`)).join('');

  const rows = (d.matches || []).map((m) => {
    const s = m.stats;
    const rounds = m.rounds || (s && s.rounds) || 0;
    if (!s) {
      return `<tr><td class="name">${esc(m.label)}</td><td>${esc(m.map || '—')}</td>
        <td>${rounds}</td><td colspan="11" class="dim">${esc(m.error || t('player.stats.noData'))}</td></tr>`;
    }
    return `<tr>
      <td class="name">${esc(m.label)}</td>
      <td>${esc(m.map || '—')}</td>
      <td>${rounds}</td>
      <td>${s.kills}</td><td>${s.deaths}</td><td>${s.assists}</td>
      <td>${s.kd}</td><td>${s.adr}</td>
      <td>${rateCell(s.first_kills, rounds)}</td>
      <td>${rateCell(s.first_deaths, rounds)}</td>
      <td>${rateCell(s.openers, s.duels)}</td>
      <td>${rateCell(s.side_kills, s.enemy_kills)}</td>
      <td>${rateCell(s.side_deaths, s.enemy_deaths)}</td>
      <td>${noSmoke ? '—' : rateCell(s.smoke_kills, s.enemy_kills)}</td>
    </tr>`;
  }).join('');

  return `
    <div class="stat-grid pstats-tiles">${tiles.join('')}</div>
    <div class="pstats-scroll">
      <table class="mtable pstats-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="pstats-note">${t('player.stats.tradNote', {
      n: (d.matches || []).length, s: (d.elapsed || 0).toFixed(1),
    })}</div>`;
}

function aiHTML(d) {
  const tot = d.totals || {};
  // 影响力 / 对枪评分是 -1~1 的相对量，和概率一样按百分比显示。
  const pct = (x, sign, n) => {
    if (x == null || n === 0) return '—';
    // 先按 0.1% 取整再拼，免得 -0.0004 显示成 "-0.0%"
    const v = (Math.round(x * 1000) / 10).toFixed(1);
    return `${sign && x > 0 ? '+' : ''}${v}%`;
  };
  const cell = (x, sign, n) => (n == null ? pct(x, sign)
    : `<span data-tip="${esc(t('replay.stats.ai.samples', { n }))}">${pct(x, sign, n)}</span>`);
  const tiles = [
    psTile(pct(tot.kill_impact, true, tot.kill_n), t('replay.stats.ai.killImpact'),
      'replay.stats.tip.killImpact', tot.kill_n),
    psTile(pct(tot.round_impact, true, tot.rounds), t('replay.stats.ai.roundImpact'),
      'replay.stats.tip.roundImpact', tot.rounds),
    psTile(pct(tot.death_risk, false, tot.risk_n), t('replay.stats.ai.deathRisk'),
      'replay.stats.tip.deathRisk', tot.risk_n),
    psTile(pct(tot.kill_rate, false, tot.kr_n), t('replay.stats.ai.killRate'),
      'replay.stats.tip.killRate', tot.kr_n),
    psTile(pct(tot.duel_score, true, tot.duel_n), t('replay.stats.ai.duelScore'),
      'replay.stats.tip.duelScore', tot.duel_n),
  ];
  const head = [
    `<th>${t('player.stats.col.match')}</th>`,
    `<th>${t('player.profile.maps.col.map')}</th>`,
    `<th>${t('player.stats.col.rounds')}</th>`,
    tipTh(t('replay.stats.ai.killImpact'), 'replay.stats.tip.killImpact'),
    tipTh(t('replay.stats.ai.roundImpact'), 'replay.stats.tip.roundImpact'),
    tipTh(t('replay.stats.ai.deathRisk'), 'replay.stats.tip.deathRisk'),
    tipTh(t('replay.stats.ai.killRate'), 'replay.stats.tip.killRate'),
    tipTh(t('replay.stats.ai.duelScore'), 'replay.stats.tip.duelScore'),
  ].join('');

  const rows = (d.matches || []).map((m) => {
    const s = m.stats;
    if (!s) {
      return `<tr><td class="name">${esc(m.label)}</td><td>${esc(m.map || '—')}</td>
        <td>${m.rounds || 0}</td><td colspan="5" class="dim">${esc(m.error || t('player.stats.noData'))}</td></tr>`;
    }
    const dcls = (v) => (v > 0.0005 ? 'good' : v < -0.0005 ? 'bad' : '');
    return `<tr>
      <td class="name">${esc(m.label)}</td>
      <td>${esc(m.map || '—')}</td>
      <td>${m.rounds || 0}</td>
      <td class="${dcls(s.kill_impact)}">${cell(s.kill_impact, true, s.kill_n)}</td>
      <td class="${dcls(s.round_impact)}">${cell(s.round_impact, true, m.rounds || s.rounds)}</td>
      <td>${cell(s.death_risk, false, s.risk_n)}</td>
      <td>${cell(s.kill_rate, false, s.kr_n)}</td>
      <td class="${dcls(s.duel_score)}">${cell(s.duel_score, true, s.duel_n)}</td>
    </tr>`;
  }).join('');

  return `
    <div class="stat-grid pstats-tiles">${tiles.join('')}</div>
    <div class="pstats-scroll">
      <table class="mtable pstats-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="pstats-note">${t('player.stats.aiNote', {
      n: (d.matches || []).length, rounds: tot.rounds || 0,
    })}</div>`;
}

function renderStatsSection() {
  const host = $('player-stats');
  if (!host) return;
  const note = $('pstats-note');
  if (PS.tab === 'trad') {
    if (PS.tradLoading) {
      host.innerHTML = psProgress(t('player.stats.loading'), PS.tradPct, PS.tradStage);
      if (note) note.textContent = '';
      return;
    }
    if (PS.tradErr) {
      host.innerHTML = psError(t('player.stats.fail', { msg: PS.tradErr.message }), 'trad');
      return;
    }
    if (!PS.trad) { host.innerHTML = `<div class="dim">${t('player.stats.idle')}</div>`; return; }
    host.innerHTML = tradHTML(PS.trad);
    if (note) {
      note.textContent = t('player.stats.note', {
        n: (PS.trad.matches || []).length, s: (PS.trad.elapsed || 0).toFixed(1),
      });
    }
  } else {
    if (PS.aiLoading) {
      host.innerHTML = psProgress(t('player.stats.aiLoading'), PS.aiPct, PS.aiStage);
      if (note) note.textContent = t('player.stats.aiSlow');
      return;
    }
    if (PS.aiErr) {
      host.innerHTML = psError(t('player.stats.fail', { msg: PS.aiErr.message }), 'ai');
      return;
    }
    if (!PS.ai) { host.innerHTML = `<div class="dim">${t('player.stats.idle')}</div>`; return; }
    host.innerHTML = aiHTML(PS.ai);
    if (note) note.textContent = t('player.stats.note', {
      n: (PS.ai.matches || []).length, s: (PS.ai.elapsed || 0).toFixed(1),
    });
  }
  mountTips();
}

function switchStatsTab(tab) {
  if (tab !== 'trad' && tab !== 'ai') return;
  PS.tab = tab;
  for (const btn of document.querySelectorAll('.pstats-tab')) {
    const on = btn.dataset.tab === tab;
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-selected', String(on));
  }
  renderStatsSection();
  if (tab === 'ai' && !PS.ai && !PS.aiLoading && !PS.aiErr) loadAiStats();
}

/** 两条统计请求共用的跑法（NDJSON 流 + 进度）。 */
async function runStats(kind) {
  const seq = ++PS.seq;
  const loadingKey = kind === 'trad' ? 'tradLoading' : 'aiLoading';
  const pctKey = kind === 'trad' ? 'tradPct' : 'aiPct';
  const stageKey = kind === 'trad' ? 'tradStage' : 'aiStage';
  const labelKey = kind === 'trad' ? 'player.stats.loading' : 'player.stats.aiLoading';
  PS[loadingKey] = true;
  PS[kind === 'trad' ? 'tradErr' : 'aiErr'] = null;
  PS[pctKey] = 0;
  PS[stageKey] = '';
  renderStatsSection();
  try {
    const payload = await streamJson(
      `/api/library/players/${encodeURIComponent(SID)}/${kind === 'trad' ? 'stats' : 'ai'}?progress=1`,
      (ev) => {
        if (seq !== PS.seq) return;
        if (typeof ev.pct === 'number') PS[pctKey] = Math.max(PS[pctKey], Math.min(1, ev.pct));
        if (typeof ev.stage === 'string' && ev.stage) PS[stageKey] = ev.stage;
        // 进度条只改宽度和文字，不重建整块（每 80 ms 一个事件，重建会闪）
        const fill = document.querySelector('.pstats-progress-fill');
        const txt = document.querySelectorAll('.pstats-progress-text');
        if (fill) fill.style.width = `${(PS[pctKey] * 100).toFixed(1)}%`;
        if (txt[0]) txt[0].innerHTML = `<span>${esc(t(labelKey))}</span>
          <span class="dim">${Math.round(PS[pctKey] * 100)}%</span>`;
        if (txt[1]) txt[1].textContent = PS[stageKey];
      },
      { brokenText: t('player.stats.streamBroken') },
    );
    if (seq !== PS.seq) return;
    PS[kind] = payload;
  } catch (err) {
    if (seq !== PS.seq) return;
    PS[kind === 'trad' ? 'tradErr' : 'aiErr'] = err;
  } finally {
    if (seq === PS.seq) {
      PS[loadingKey] = false;
      renderStatsSection();
    }
  }
}

function loadTradStats() { return runStats('trad'); }
function loadAiStats() { return runStats('ai'); }

function bindStatsTabs() {
  document.querySelectorAll('.pstats-tab').forEach((btn) => {
    btn.addEventListener('click', () => switchStatsTab(btn.dataset.tab));
  });
  $('player-stats').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-ps-retry]');
    if (!btn) return;
    if (btn.dataset.psRetry === 'trad') { PS.tradErr = null; loadTradStats(); }
    else { PS.aiErr = null; loadAiStats(); }
  });
  mountTips();
}

// ── 事件 ──────────────────────────────────────────────────────────────────

function bindEvents() {
  let searchTimer = null;
  $('demo-search').oninput = (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      S.q = e.target.value.trim();
      S.offset = 0;
      loadDemos();
    }, 250);
  };

  $('demo-list').addEventListener('click', (ev) => {
    const act = ev.target.closest('[data-act]');
    if (act) openInReplay([Number(act.dataset.id)]);
  });

  $('pager').addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-page]');
    if (!btn || btn.disabled) return;
    const pages = Math.ceil(S.total / S.limit);
    const page = Math.floor(S.offset / S.limit);
    const target = { first: 0, prev: page - 1, next: page + 1, last: pages - 1 }[btn.dataset.page];
    S.offset = Math.max(0, Math.min(target, pages - 1)) * S.limit;
    loadDemos();
  });

  $('btn-load-all').onclick = loadAll;
  $('btn-watch').onclick = toggleWatch;
  $('btn-refresh-avatar').onclick = refreshAvatar;
}

bindEvents();
bindStatsTabs();
loadPlayer();
loadDemos();
// 传统统计自动拉（它就是这位选手的主页数据）；AI 那一栏点了页签才拉
loadTradStats();
