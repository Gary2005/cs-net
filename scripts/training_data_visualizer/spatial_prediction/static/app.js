/* spatial_prediction · 训练数据可视化 —— 前端（原生 JS + canvas，无构建、无外部依赖）。
 *
 * 和 aim 那版的区别：**一条样本 = 一帧**，所以这里没有时间轴 / 播放 / 第一人称，
 * 取而代之的是"这一帧的俯视图 + 5 个标签 + 两张查错表（遮挡 / 原始特征）"。
 *
 * 数据流：
 *   /api/samples            → {files:[{split,name,map,round,n}], …}，一个 npz = K 行样本
 *   /api/sample/<split>/<name>?k=N → 一行样本的完整载荷（单帧 + 5 个标签 + 体检警告）
 *
 * 页面上"样本下标"是**扁平的**：第 f 个文件的第 k 行 → starts[f] + k。所以 ◀ / ▶ 可以
 * 跨文件连续翻，滑块则在当前文件内跳。
 */

'use strict';

const $ = (id) => document.getElementById(id);
const RAD = Math.PI / 180;
const SPLITS = ['train', 'valid', 'test'];
const TEAM_COLOR = { CT: '#4da3ff', T: '#f5a524', '?': '#6b7c8d' };

const S = {
  dataDir: '',
  files: [],
  starts: [],
  total: 0,
  index: 0,
  data: null,
  radarImg: null,
  icons: {},
  showPhys: false,
  seq: 0,          // 丢弃过期的 fetch 结果（连点 ◀ ▶ 时）
};

// ── 小工具 ────────────────────────────────────────────────────────────

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
function fmtBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}
function fmtSec(v, digits = 2) {
  return Number.isFinite(v) ? `${v.toFixed(digits)} s` : '—';
}
function teamColor(team) { return TEAM_COLOR[team] || TEAM_COLOR['?']; }
function pName(p) { return `P${p}`; }

function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 2600);
}

function setLoading(on) { $('loading').hidden = !on; }

function apiPath(...parts) {
  return '/api/' + parts.map((p) => encodeURIComponent(p)).join('/');
}

// ── 扁平下标 ⇄ (文件, 行) ─────────────────────────────────────────────

function rebuildIndex() {
  S.starts = [];
  let acc = 0;
  for (const f of S.files) {
    S.starts.push(acc);
    acc += Math.max(0, f.n | 0);
  }
  S.total = acc;
}

function fileOfIndex(i) {
  if (!S.files.length) return 0;
  let lo = 0, hi = S.files.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (S.starts[mid] <= i) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return ans;
}

function locOfIndex(i) {
  const f = fileOfIndex(i);
  return { f, k: i - S.starts[f] };
}

function indexOfLoc(f, k) {
  if (!S.files.length) return 0;
  const file = S.files[clamp(f, 0, S.files.length - 1)];
  return S.starts[clamp(f, 0, S.files.length - 1)] + clamp(k, 0, Math.max(0, file.n - 1));
}

// ── 样本列表 ──────────────────────────────────────────────────────────

async function loadSamples() {
  const resp = await fetch('/api/samples', { headers: { 'Accept-Encoding': 'identity' } });
  const body = await resp.json();
  if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
  S.dataDir = body.data_dir || '';
  S.files = body.files || [];
  rebuildIndex();
  $('data-dir').textContent = S.dataDir || '（未知目录）';
  buildFileSelect();
  document.title = `spatial_prediction · ${S.total} 帧 · 训练数据可视化`;
  if (!S.files.length) {
    $('sample-name').textContent = '一个样本都没有';
    $('label-time').textContent = '';
    toast('目录里没有 *_n{K}.npz：先跑 prepare --task spatial_prediction');
  }
}

function buildFileSelect() {
  const sel = $('file-select');
  sel.innerHTML = '';
  S.files.forEach((f, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `${f.split}/${f.name}  ·  ${f.n} 帧`
      + (f.round == null ? '' : `  ·  r${f.round}`);
    sel.appendChild(opt);
  });
}

function readUrlState() {
  const hash = decodeURIComponent(location.hash.replace(/^#/, ''));
  const m = hash.match(/^(train|valid|test)\/(.+?)\/(\d+)$/);
  if (!m) return null;
  const f = S.files.findIndex((x) => x.split === m[1] && x.name === m[2]);
  if (f < 0) return null;
  return indexOfLoc(f, Number(m[3]));
}

function writeUrlState() {
  if (!S.files.length) return;
  const { f, k } = locOfIndex(S.index);
  const file = S.files[f];
  const hash = `#${file.split}/${file.name}/${k}`;
  if (location.hash !== hash) history.replaceState(null, '', hash);
}

// ── 取一条样本 ────────────────────────────────────────────────────────

async function loadCurrent() {
  if (!S.files.length) return;
  S.index = clamp(S.index, 0, Math.max(0, S.total - 1));
  const { f, k } = locOfIndex(S.index);
  const file = S.files[f];
  const seq = ++S.seq;
  setLoading(true);
  try {
    const url = `${apiPath('sample', file.split, file.name)}?k=${k}`;
    const resp = await fetch(url, { headers: { 'Accept-Encoding': 'identity' } });
    const body = await resp.json();
    if (seq !== S.seq) return;              // 已经翻到别处了
    if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
    S.data = body;
    $('file-select').value = String(f);
    writeUrlState();
    render();
  } catch (err) {
    if (seq !== S.seq) return;
    S.data = null;
    toast(`读取失败：${err.message}`);
    renderEmpty(String(err.message));
  } finally {
    if (seq === S.seq) setLoading(false);
  }
}

function goto(index) {
  const next = clamp(index, 0, Math.max(0, S.total - 1));
  if (next === S.index && S.data) { updateNav(); return; }
  S.index = next;
  updateNav();
  loadCurrent();
}

function step(delta) { goto(S.index + delta); }

// ── 导航条 ────────────────────────────────────────────────────────────

function updateNav() {
  const { f, k } = locOfIndex(S.index);
  const file = S.files[f];
  $('sample-readout').textContent = `${S.index + 1} / ${S.total}`;
  const slider = $('row-slider');
  if (file) {
    slider.max = String(Math.max(0, file.n - 1));
    slider.value = String(k);
    slider.disabled = file.n <= 1;
    $('row-readout').textContent = `第 ${k + 1} / ${file.n} 行`;
  } else {
    slider.max = '0';
    slider.value = '0';
    slider.disabled = true;
    $('row-readout').textContent = '—';
  }
}

// ── 渲染入口 ──────────────────────────────────────────────────────────

function renderEmpty(msg) {
  $('sample-name').textContent = msg ? `⚠ ${msg}` : '未选择样本';
  $('sample-path').textContent = '';
  $('meta-chips').innerHTML = '';
  $('label-cards').innerHTML = '';
  $('list-alive-end').innerHTML = '';
  $('list-can-kill').innerHTML = '';
  $('player-table').innerHTML = '';
  $('global-table').innerHTML = '';
  $('proj-table').innerHTML = '';
  $('obstr-table').innerHTML = '';
  $('feat-table').innerHTML = '';
  $('label-time').textContent = '';
  $('field-readout').textContent = '—';
  $('obstr-readout').textContent = '—';
  $('alive-end-n').textContent = '';
  $('can-kill-n').textContent = '';
  drawEmptyRadar();
  updateNav();
}

function render() {
  const d = S.data;
  if (!d) { renderEmpty(''); return; }
  updateNav();
  renderWarnings();
  renderMeta();
  renderLabels();
  renderGlobal();
  renderProjectiles();
  renderPlayers();
  renderObstruction();
  renderFeatures();
  renderFiles();
  loadRadarImage();
  drawRadar();
}

// ── 警告条 ────────────────────────────────────────────────────────────

function renderWarnings() {
  const bar = $('warning-bar');
  const list = (S.data && S.data.warnings) || [];
  if (!list.length) { bar.hidden = true; bar.innerHTML = ''; return; }
  bar.innerHTML = `<b>⚠ ${list.length} 条提醒</b><ul>`
    + list.map((w) => `<li>${esc(w)}</li>`).join('') + '</ul>';
  bar.hidden = false;
}

// ── 概览 ──────────────────────────────────────────────────────────────

function addChip(container, label, value, cls = '') {
  const el = document.createElement('span');
  el.className = `chip ${cls}`.trim();
  el.innerHTML = `${esc(label)} <b>${esc(value)}</b>`;
  container.appendChild(el);
}

function renderMeta() {
  const d = S.data;
  const { f, k } = locOfIndex(S.index);
  const file = S.files[f] || {};
  $('sample-name').textContent = `${d.map_name} · r${d.round_id} · ${fmtSec(d.seconds)}`;
  $('sample-path').textContent = d.path || `${d.split}/${d.name}`;
  const chips = $('meta-chips');
  chips.innerHTML = '';
  addChip(chips, 'split', d.split, d.split === 'train' ? '' : 'warn');
  addChip(chips, '文件', d.name);
  addChip(chips, '行', `${k + 1} / ${d.rows}`);
  addChip(chips, '回合', `r${d.round_id}`);
  addChip(chips, '回合时间', fmtSec(d.seconds));
  addChip(chips, '帧下标', String(d.frame_index));
  addChip(chips, '文件大小', fmtBytes(d.bytes || file.bytes));
  addChip(chips, '玩家', String(d.n_players));
  addChip(chips, '特征列', String(d.n_features));
  const alive = (d.players.alive || []).filter(Boolean).length;
  addChip(chips, '存活', `${alive} / ${d.n_players}`, alive ? 'ok' : 'bad');
  $('label-time').textContent = `t = ${fmtSec(d.seconds, 3)}（回合内）`;
  $('field-readout').textContent = `${d.n_players} 人 · ${d.projectiles.length} 个道具`
    + (d.global.planted ? ` · C4 已安放 ${d.global.planted_seconds.toFixed(1)} s` : ' · C4 携带中');
}

// ── 标签卡片 ──────────────────────────────────────────────────────────

function playerBadge(d, p) {
  const team = (d.players.teams || [])[p] || '?';
  const cls = team === 'CT' ? 'ct' : team === 'T' ? 't' : 'unknown';
  return `<span class="pbadge ${cls}">${pName(p)}</span>`;
}

function playerSummary(d, p) {
  const team = (d.players.teams || [])[p] || '?';
  const hp = Math.round((d.players.phys.hp || [])[p] ?? 0);
  const wpn = (d.players.weapon_names || [])[p] || '无武器';
  return `${team} · ${hp} HP · ${wpn}`;
}

function labelCard(meta) {
  const d = S.data;
  const L = d.labels || {};
  const card = document.createElement('div');
  card.className = 'card';
  const head = document.createElement('div');
  head.className = 'card-k';
  head.innerHTML = `${esc(meta.title)} <span class="mono">${esc(meta.key)}</span>`;
  const val = document.createElement('div');
  val.className = 'card-v';
  const sub = document.createElement('div');
  sub.className = 'card-s';
  sub.textContent = meta.note;

  if (meta.key === 'ct_win') {
    val.classList.add(L.ct_win ? 'v-ok' : 'v-no');
    val.textContent = L.ct_win ? '是' : '否';
  } else if (meta.key === 'next_death' || meta.key === 'next_kill') {
    const p = L[meta.key];
    const isDeath = meta.key === 'next_death';
    const nPl = d.n_players || 10;
    // 10（= n_players）是"没有下一次"，不是玩家下标
    if (Number.isInteger(p) && p >= 0 && p < nPl) {
      val.classList.add(isDeath ? 'v-death' : 'v-kill');
      val.innerHTML = `${playerBadge(d, p)}<span class="unit">${esc(playerSummary(d, p))}</span>`;
      card.classList.add(isDeath ? 'picked-death' : 'picked-kill');
    } else {
      val.classList.add('v-none');
      val.innerHTML = `<span>—</span><span class="unit">${isDeath ? '本回合不会再有人死' : '本回合不会再有人击杀'}</span>`;
    }
  } else {
    // 两个 10 维 bool 标签：显示命中人数
    const n = meta.key === 'alive_end' ? L.alive_end_n : L.can_kill_n;
    const arr = meta.key === 'alive_end' ? L.alive_end : L.can_kill;
    const total = Array.isArray(arr) ? arr.length : 10;
    val.innerHTML = `<span>${n ?? '?'}<span class="unit"> / ${total}</span></span>`;
    val.classList.add(n > 0 ? 'v-ok' : 'v-no');
  }

  card.appendChild(head);
  card.appendChild(val);
  card.appendChild(sub);
  return card;
}

function renderLabels() {
  const d = S.data;
  const host = $('label-cards');
  host.innerHTML = '';
  for (const meta of (d.labels_meta || [])) host.appendChild(labelCard(meta));

  const L = d.labels || {};
  renderPlayerList($('list-alive-end'), L.alive_end);
  renderPlayerList($('list-can-kill'), L.can_kill);
  $('alive-end-n').textContent = `${L.alive_end_n ?? 0} / ${(L.alive_end || []).length}`;
  $('can-kill-n').textContent = `${L.can_kill_n ?? 0} / ${(L.can_kill || []).length}`;
}

function renderPlayerList(host, flags) {
  const d = S.data;
  host.innerHTML = '';
  (flags || []).forEach((flag, p) => {
    const alive = !!(d.players.alive || [])[p];
    const team = (d.players.teams || [])[p] || '?';
    const cls = team === 'CT' ? 'ct' : team === 'T' ? 't' : '';
    const el = document.createElement('span');
    el.className = `pchip ${cls} ${flag ? 'on' : 'off'} ${alive ? '' : 'dead'}`.trim();
    el.title = `${pName(p)} · ${team} · ${alive ? '当前帧存活' : '当前帧已死亡'}`;
    el.innerHTML = `<span class="idx">${pName(p)}</span>`
      + `<span class="mark">${flag ? '✓' : alive ? '✗' : '×'}</span>`;
    host.appendChild(el);
  });
}

// ── 全局 / 道具表 ─────────────────────────────────────────────────────

function renderGlobal() {
  const d = S.data;
  const g = d.global || {};
  const rows = [
    ['回合时间', fmtSec(d.seconds, 3)],
    ['global 回合时间', g.round_seconds == null ? '未记录' : fmtSec(g.round_seconds, 2)],
    ['炸弹', g.planted ? `已安放 ${fmtSec(g.planted_seconds, 2)}` : '携带 / 掉落中'],
    ['炸弹位置', `${g.x ?? 0}, ${g.y ?? 0}, ${g.z ?? 0}`],
    ['帧下标', String(d.frame_index)],
    ['行', `${(d.row ?? 0) + 1} / ${d.rows ?? 1}`],
  ];
  const table = $('global-table');
  table.innerHTML = rows
    .map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`)
    .join('');
}

function renderProjectiles() {
  const d = S.data;
  const projs = d.projectiles || [];
  const table = $('proj-table');
  if (!projs.length) {
    table.innerHTML = '<thead><tr><th>这一帧没有生效 / 飞行中的道具</th></tr></thead>';
    return;
  }
  const head = '<thead><tr><th>槽</th><th>类型</th><th>已存在</th><th>位置 x / y / z</th></tr></thead>';
  const body = projs.map((p) => (
    `<tr><td class="num">${p.slot}</td><td>${esc(p.type)}</td>`
    + `<td class="num">${p.seconds.toFixed(2)} s</td>`
    + `<td class="num">${p.x} / ${p.y} / ${p.z}</td></tr>`
  )).join('');
  table.innerHTML = head + `<tbody>${body}</tbody>`;
}

// ── 玩家表 ────────────────────────────────────────────────────────────

function weaponIcon(name) {
  const url = S.icons[name];
  return url ? `<img src="${esc(url)}" alt="">` : '';
}

function renderPlayers() {
  const d = S.data;
  const L = d.labels || {};
  const table = $('player-table');
  const head = `<thead><tr>
      <th>#</th><th>队伍</th><th>血量</th><th>护甲</th><th>头盔</th><th>武器</th>
      <th>存活</th><th>终局存活</th><th>还能击杀</th><th>标记</th>
    </tr></thead>`;
  const n = d.n_players | 0;
  const rows = [];
  for (let p = 0; p < n; p++) {
    const alive = !!(d.players.alive || [])[p];
    const team = (d.players.teams || [])[p] || '?';
    const hp = Math.round((d.players.phys.hp || [])[p] ?? 0);
    const armor = Math.round((d.players.phys.armor || [])[p] ?? 0);
    const helmet = ((d.players.phys.helmet || [])[p] ?? 0) > 0.5;
    const wpnIdx = (d.players.weapon || [])[p];
    const wpn = (d.players.weapon_names || [])[p] || '';
    const endAlive = (L.alive_end || [])[p];
    const canKill = (L.can_kill || [])[p];
    const marks = [];
    const cls = [];
    if (!alive) cls.push('dead');
    if (L.next_death === p) { marks.push('<span class="mk-death" title="下一个死亡">☠</span>'); cls.push('mark-death'); }
    if (L.next_kill === p) { marks.push('<span class="mk-kill" title="下一个击杀">✛</span>'); cls.push('mark-kill'); }
    rows.push(`<tr class="${cls.join(' ')}">
      <td class="mono">${pName(p)}</td>
      <td style="color:${teamColor(team)}">${esc(team)}</td>
      <td class="mono">${hp}</td>
      <td class="mono">${armor}</td>
      <td class="mono">${helmet ? '有' : '—'}</td>
      <td><span class="wpn">${weaponIcon(wpn)}${esc(wpn || (wpnIdx < 0 ? '—' : `#${wpnIdx}`))}</span></td>
      <td class="mono ${alive ? 'yes' : 'no'}">${alive ? '✓' : '×'}</td>
      <td class="mono ${endAlive ? 'yes' : 'no'}">${endAlive ? '✓' : '✗'}</td>
      <td class="mono ${canKill ? 'yes' : 'no'}">${canKill ? '✓' : '✗'}</td>
      <td>${marks.join(' ') || '<span class="dim">—</span>'}</td>
    </tr>`);
  }
  table.innerHTML = head + `<tbody>${rows.join('')}</tbody>`;
}

// ── 遮挡矩阵 ──────────────────────────────────────────────────────────

function renderObstruction() {
  const d = S.data;
  const obs = d.obstruction || [];
  const n = d.n_players | 0;
  const L = d.labels || {};
  const table = $('obstr-table');
  const headCell = (p) => {
    const cls = L.next_death === p ? 'hd-death' : L.next_kill === p ? 'hd-kill' : 'hd';
    const mark = L.next_death === p ? '☠' : L.next_kill === p ? '✛' : pName(p);
    return `<th class="${cls}">${mark}</th>`;
  };
  let html = '<thead><tr><th></th>';
  for (let p = 0; p < n; p++) html += headCell(p);
  html += '</tr></thead><tbody>';
  for (let i = 0; i < n; i++) {
    html += `<tr>${headCell(i)}`;
    for (let j = 0; j < n; j++) {
      if (i === j) { html += '<td class="diag"></td>'; continue; }
      const aliveI = !!(d.players.alive || [])[i];
      const aliveJ = !!(d.players.alive || [])[j];
      if (!aliveI || !aliveJ) { html += '<td class="odead">·</td>'; continue; }
      const v = ((obs[i] || [])[j]) | 0;
      const cls = ['o0', 'o1', 'o2'][v] || 'odead';
      html += `<td class="${cls}" title="P${i} 看 P${j}：${esc((d.obstruction_names || {})[String(v)] || v)}">${v}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody>';
  table.innerHTML = html;

  const a = L.next_death;
  const b = L.next_kill;
  const nPl = d.n_players || 10;
  const parts = [];
  if (Number.isInteger(a) && a >= 0 && a < nPl) parts.push(`下一个死亡 P${a}`);
  if (Number.isInteger(b) && b >= 0 && b < nPl) parts.push(`下一个击杀 P${b}`);
  $('obstr-readout').textContent = parts.join('  ·  ') || '本回合没有后续死亡 / 击杀';
}

// ── 原始特征矩阵 ──────────────────────────────────────────────────────

const FEATURE_NOTE = {
  x: '脚底世界坐标 x（HU）', y: '脚底世界坐标 y（HU）', z: '脚底世界坐标 z（HU）',
  yaw_sin: 'sin(yaw)：yaw 在 ±180 会跳变，所以拆成 sin/cos 两列',
  yaw_cos: 'cos(yaw)',
  pitch_sin: 'sin(pitch)：正 = 向下看',
  pitch_cos: 'cos(pitch)',
  vx: '水平速度 vx（HU/s，±300）', vy: '水平速度 vy（HU/s，±300）',
  vz: '垂直速度 vz（HU/s，±1200）',
  hp: '血量 0..100', crouch: '下蹲量 0..1（连续，眼高 = 64.0938 − 18·crouch）',
  armor: '护甲 0..100', helmet: '有没有头盔 0/1',
  team: '队伍常量列：CT=0 / T=1（沿回合不变）',
  shot: '本帧是否开枪 0/1',
  since_shot: '距上次开枪的秒数（÷100；99.9 = 本回合还没开过枪）',
  has_kill: '截至本帧**已经**拿到过击杀 0/1（不是标签 can_kill）',
  blinded: '本帧是否被闪光致盲 0/1（玩家私有状态）',
  defuser: '本帧是否带着拆弹器 0/1（玩家私有状态）',
};

function physText(name, p) {
  const arr = (S.data.players.phys || {})[name];
  const v = arr ? arr[p] : 0;
  if (name === 'x' || name === 'y' || name === 'z') return v.toFixed(0);
  if (name === 'yaw' || name === 'pitch') return v.toFixed(1);
  if (name === 'since_shot') return v >= 99.9 ? '99.9' : v.toFixed(2);
  if (name === 'hp' || name === 'armor') return v.toFixed(0);
  if (name === 'team' || name === 'shot' || name === 'has_kill' || name === 'helmet'
      || name === 'blinded' || name === 'defuser') return v.toFixed(0);
  return v.toFixed(2);
}

function renderFeatures() {
  const d = S.data;
  const n = d.n_players | 0;
  const L = d.labels || {};
  const names = d.features || [];
  const raw = d.players.raw || null;      // (10, F) 的 [0,1] 原值
  const table = $('feat-table');
  let html = '<thead><tr><th>特征 \\ 玩家</th>';
  for (let p = 0; p < n; p++) {
    const marks = [];
    if (L.next_death === p) marks.push('☠');
    if (L.next_kill === p) marks.push('✛');
    const team = (d.players.teams || [])[p] || '';
    html += `<th style="color:${teamColor(team)}">${pName(p)}${marks.length ? ' ' + marks.join('') : ''}</th>`;
  }
  html += '</tr></thead><tbody>';

  names.forEach((name, col) => {
    html += `<tr data-feature="${esc(name)}"><th title="${esc(FEATURE_NOTE[name] || name)}">${esc(name)}</th>`;
    for (let p = 0; p < n; p++) {
      const dead = !(d.players.alive || [])[p];
      const note = esc(FEATURE_NOTE[name] || name);
      if (S.showPhys) {
        html += `<td class="cell${dead ? ' dim' : ''}" title="${note}">`
          + `${esc(physText(name, p))}</td>`;
      } else {
        const v = raw && raw[p] ? Number(raw[p][col]) : null;
        const text = v == null || Number.isNaN(v) ? '—' : (Math.round(v * 1000) / 1000).toFixed(3);
        const hot = (name === 'shot' || name === 'blinded') && v > 0;
        const hi = name === 'has_kill' && v > 0;
        html += `<td class="cell${hot ? ' hot' : ''}${hi ? ' hi' : ''}" title="${note}">`
          + `${text}</td>`;
      }
    }
    html += '</tr>';
  });
  html += '</tbody>';
  table.innerHTML = html;
}

function renderFiles() {
  const d = S.data;
  const files = d.files || [];
  $('files-count').textContent = String(files.length);
  const table = $('files-table');
  const head = '<thead><tr><th>key</th><th>shape</th><th>dtype</th></tr></thead>';
  const body = files.map((f) => (
    `<tr><td>${esc(f.key)}</td><td class="num">${esc((f.shape || []).join(' × ') || '()')}</td>`
    + `<td>${esc(f.dtype)}</td></tr>`
  )).join('');
  table.innerHTML = head + `<tbody>${body}</tbody>`;
}


// ═══════════════════════════════════════════════════════════════════════
// 雷达：单帧俯视图
// ═══════════════════════════════════════════════════════════════════════
//
// 画的是**当前帧** mask != 0 的玩家：颜色 = 队伍，短线 = yaw 朝向，圆环 = 血量，
// 圆点里是玩家下标。标签里的「下一个死亡 / 下一个击杀」额外加一个虚线圈 + 图标。
// 死人没有有效坐标（位置列可能是 0 / 残留值），所以不画 —— 只出现在玩家表里。

let legendBuilt = false;
function buildRadarLegend() {
  if (legendBuilt) return;
  legendBuilt = true;
  const el = $('radar-legend');
  if (!el) return;
  el.innerHTML = `
    <span class="lg"><i style="background:#4da3ff"></i>CT 存活</span>
    <span class="lg"><i style="background:#f5a524"></i>T 存活</span>
    <span class="lg"><i style="background:#0a0f14;border:1px solid #223040"></i>死亡（不画）</span>
    <span class="lg" style="color:#ff5a5a"><b>☠</b> 下一个死亡</span>
    <span class="lg" style="color:#ffd257"><b>✛</b> 下一个击杀</span>
    <span class="lg"><i style="background:#ffd257;border-radius:2px"></i>C4</span>
    <span class="lg"><i style="background:#7fd2ff;border-radius:2px"></i>道具</span>`;
}

function loadRadarImage() {
  const d = S.data;
  S.radarImg = null;
  if (!d || !d.radar) { drawRadar(); return; }
  const img = new Image();
  img.onload = () => {
    if (S.data !== d) return;
    S.radarImg = img;
    drawRadar();
  };
  img.onerror = () => {
    if (S.data !== d) return;
    S.radarImg = null;
    drawRadar();
  };
  img.src = `${d.radar.url}?v=${d.bytes || 0}`;
}

function radarTransform(d, W, H) {
  if (d.radar && S.radarImg && S.radarImg.complete && S.radarImg.naturalWidth) {
    const [u0, u1, v0, v1] = d.radar.extent;
    return {
      hasImage: true,
      toPx: (x, y) => [((x - u0) / (u1 - u0 || 1)) * W, ((v1 - y) / (v1 - v0 || 1)) * H],
    };
  }
  // 没有雷达图：用这一帧活人和炸弹的位置估一个方框，画坐标网格
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const add = (x, y) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  };
  const n = d.n_players || 10;
  for (let p = 0; p < n; p++) {
    if (!((d.players.alive || [])[p])) continue;
    add(d.players.phys.x[p], d.players.phys.y[p]);
  }
  if (d.global) add(d.global.x, d.global.y);
  if (!Number.isFinite(minX)) { minX = -2500; maxX = 2500; minY = -2500; maxY = 2500; }
  const pad = Math.max(maxX - minX, maxY - minY, 600) * 0.12;
  minX -= pad; maxX += pad; minY -= pad; maxY += pad;
  const span = Math.max(maxX - minX, maxY - minY, 1);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  minX = cx - span / 2; maxX = cx + span / 2;
  minY = cy - span / 2; maxY = cy + span / 2;
  return {
    hasImage: false,
    bounds: [minX, maxX, minY, maxY],
    toPx: (x, y) => [((x - minX) / (maxX - minX || 1)) * W, ((maxY - y) / (maxY - minY || 1)) * H],
  };
}

function drawEmptyRadar() {
  const canvas = $('radar');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#05080c';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(150, 180, 205, 0.55)';
  ctx.font = '13px Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('没有样本', W / 2, H / 2);
}

function drawRadar() {
  const d = S.data;
  if (!d) { drawEmptyRadar(); return; }
  const canvas = $('radar');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#05080c';
  ctx.fillRect(0, 0, W, H);

  const t = radarTransform(d, W, H);
  if (t.hasImage) {
    ctx.drawImage(S.radarImg, 0, 0, W, H);
    ctx.fillStyle = 'rgba(5, 9, 14, 0.30)';
    ctx.fillRect(0, 0, W, H);
  } else {
    drawRadarGrid(ctx, W, H, t);
  }

  // ① 道具
  for (const p of (d.projectiles || [])) {
    const [px, py] = t.toPx(p.x, p.y);
    ctx.save();
    ctx.strokeStyle = projColor(p.id);
    ctx.fillStyle = 'rgba(5, 9, 14, 0.78)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = projColor(p.id);
    ctx.font = '8px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(projGlyph(p.id), px, py + 0.5);
    ctx.restore();
  }

  // ② 炸弹（这一帧的位置：会被携带、掉落、安放）
  const g = d.global || {};
  const [bx, by] = t.toPx(g.x ?? 0, g.y ?? 0);
  ctx.save();
  ctx.translate(bx, by);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = '#ffd257';
  ctx.strokeStyle = '#fff3b0';
  ctx.lineWidth = 1;
  ctx.fillRect(-6, -6, 12, 12);
  ctx.strokeRect(-6, -6, 12, 12);
  ctx.restore();
  ctx.save();
  ctx.fillStyle = 'rgba(255, 226, 150, 0.92)';
  ctx.font = '11px Consolas, monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(g.planted ? `C4 · 已安放 ${(g.planted_seconds ?? 0).toFixed(1)}s` : 'C4 · 携带中',
    bx + 10, by - 8);
  ctx.restore();

  // ③ 玩家
  const n = d.n_players | 0;
  const L = d.labels || {};
  for (let p = 0; p < n; p++) {
    if (!((d.players.alive || [])[p])) continue;
    const x = d.players.phys.x[p];
    const y = d.players.phys.y[p];
    const yaw = (d.players.phys.yaw[p] || 0) * RAD;
    const hp = d.players.phys.hp[p] ?? 0;
    const team = (d.players.teams || [])[p] || '?';
    const color = teamColor(team);
    const isDeath = L.next_death === p;
    const isKill = L.next_kill === p;
    const marked = isDeath || isKill;
    const r = marked ? 10 : 7;
    const [px, py] = t.toPx(x, y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;

    // yaw 朝向：世界方向 (cos, sin)，屏幕 y 轴朝下 → dy 取反
    ctx.save();
    ctx.strokeStyle = hexAlpha(color, 0.95);
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + Math.cos(yaw) * (r + 11), py - Math.sin(yaw) * (r + 11));
    ctx.stroke();
    ctx.restore();

    // 血量环（从 12 点方向顺时针）
    ctx.save();
    ctx.strokeStyle = hp > 50 ? '#45d483' : hp > 25 ? '#f5a524' : '#ff5a5a';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(px, py, r + 4, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * clamp(hp / 100, 0, 1));
    ctx.stroke();
    ctx.restore();

    // 身体
    ctx.save();
    ctx.fillStyle = hexAlpha(color, 0.92);
    ctx.strokeStyle = marked ? (isDeath ? '#ff5a5a' : '#ffd257') : 'rgba(255,255,255,0.5)';
    ctx.lineWidth = marked ? 2.4 : 1;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#061019';
    ctx.font = `${marked ? 'bold 11px' : '9px'} Consolas, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(p), px, py + 0.5);
    ctx.restore();

    // 血量数字（右边）
    ctx.save();
    ctx.fillStyle = 'rgba(220, 235, 248, 0.85)';
    ctx.font = '10px Consolas, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(Math.round(hp)), px + r + 6, py + 1);
    ctx.restore();

    // 标签标记
    if (marked) {
      ctx.save();
      ctx.strokeStyle = isDeath ? 'rgba(255, 90, 90, 0.75)' : 'rgba(255, 210, 87, 0.75)';
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(px, py, r + 12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = isDeath ? '#ff5a5a' : '#ffd257';
      ctx.font = 'bold 15px "Segoe UI Symbol", Consolas, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(isDeath ? '☠' : '✛', px, py - r - 4);
      ctx.restore();
    }
  }

  // ④ 角标
  ctx.save();
  ctx.fillStyle = 'rgba(150, 180, 205, 0.75)';
  ctx.font = '10px Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'bottom';
  ctx.fillText(
    `t = ${(d.seconds ?? 0).toFixed(2)}s · r${d.round_id} · 第 ${(d.row ?? 0) + 1}/${d.rows} 行 · x/y 平面`,
    W - 10, H - 10
  );
  ctx.restore();

  buildRadarLegend();
}

function drawRadarGrid(ctx, W, H, t) {
  const [x0, x1, y0, y1] = t.bounds;
  const span = Math.max(x1 - x0, y1 - y0);
  const step = niceStep(span / 8);
  ctx.save();
  ctx.strokeStyle = 'rgba(48, 76, 100, 0.55)';
  ctx.lineWidth = 1;
  ctx.font = '10px Consolas, monospace';
  ctx.fillStyle = 'rgba(120, 150, 175, 0.7)';
  ctx.textAlign = 'left';
  for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) {
    const [px] = t.toPx(x, y0);
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
    ctx.fillText(String(Math.round(x)), px + 3, H - 24);
  }
  for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
    const [, py] = t.toPx(x0, y);
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(W, py);
    ctx.stroke();
    ctx.fillText(String(Math.round(y)), 6, py - 4);
  }
  ctx.fillStyle = 'rgba(150, 180, 205, 0.8)';
  ctx.font = '11px Consolas, monospace';
  ctx.fillText('无雷达底图 · 网格单位 = 世界 HU', 12, 20);
  ctx.restore();
}

function niceStep(raw) {
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1))));
  const n = raw / pow;
  const pick = n <= 1.5 ? 1 : n <= 3.5 ? 2 : n <= 7.5 ? 5 : 10;
  return pick * pow;
}

function hexAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function projColor(id) {
  return ['#9aa9b8', '#ff7a3d', '#fff05e', '#c693ff', '#8fe36a', '#ff9d3d', '#7fd2ff'][id]
    || '#cccccc';
}

function projGlyph(id) {
  return ['烟', '火', '闪', '诱', '雷', '烧', '烟'][id] || '?';
}


// ═══════════════════════════════════════════════════════════════════════
// 事件 / 启动
// ═══════════════════════════════════════════════════════════════════════

async function loadWeaponIcons() {
  try {
    const resp = await fetch('/api/weapon-icons', { headers: { 'Accept-Encoding': 'identity' } });
    const body = await resp.json();
    S.icons = (resp.ok && body.icons) || {};
  } catch (_err) {
    S.icons = {};
  }
}

async function rescan() {
  try {
    const resp = await fetch('/api/rescan', { headers: { 'Accept-Encoding': 'identity' } });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
    await loadSamples();
    S.index = clamp(S.index, 0, Math.max(0, S.total - 1));
    updateNav();
    loadCurrent();
    toast(`重新扫描：${S.files.length} 个文件 / ${S.total} 帧`);
  } catch (err) {
    toast(`重新扫描失败：${err.message}`);
  }
}

function jumpFile(delta) {
  if (!S.files.length) return;
  const { f } = locOfIndex(S.index);
  goto(indexOfLoc(clamp(f + delta, 0, S.files.length - 1), 0));
}

function bindEvents() {
  $('prev').addEventListener('click', () => step(-1));
  $('next').addEventListener('click', () => step(1));
  $('rescan').addEventListener('click', rescan);

  $('file-select').addEventListener('change', (ev) => {
    const f = Number(ev.target.value);
    if (Number.isFinite(f)) goto(indexOfLoc(f, 0));
  });

  $('row-slider').addEventListener('input', (ev) => {
    const { f } = locOfIndex(S.index);
    goto(indexOfLoc(f, Number(ev.target.value)));
  });

  $('opt-phys').addEventListener('change', (ev) => {
    S.showPhys = !!ev.target.checked;
    if (S.data) renderFeatures();
  });

  window.addEventListener('hashchange', () => {
    const idx = readUrlState();
    if (idx != null && idx !== S.index) goto(idx);
  });

  document.addEventListener('keydown', (ev) => {
    const tag = (document.activeElement && document.activeElement.tagName) || '';
    if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (ev.key === 'ArrowLeft') { ev.preventDefault(); step(-1); }
    else if (ev.key === 'ArrowRight') { ev.preventDefault(); step(1); }
    else if (ev.key === 'Home') { ev.preventDefault(); goto(0); }
    else if (ev.key === 'End') { ev.preventDefault(); goto(S.total - 1); }
    else if (ev.key === '[') { ev.preventDefault(); jumpFile(-1); }
    else if (ev.key === ']') { ev.preventDefault(); jumpFile(1); }
    else if (ev.key === 'p' || ev.key === 'P') {
      const box = $('opt-phys');
      box.checked = !box.checked;
      box.dispatchEvent(new Event('change'));
    }
  });
}

async function boot() {
  bindEvents();
  buildRadarLegend();
  await loadWeaponIcons();
  try {
    await loadSamples();
  } catch (err) {
    toast(`扫描失败：${err.message}`);
    renderEmpty(err.message);
    return;
  }
  if (!S.files.length) {
    renderEmpty('没有样本，先跑 prepare --task spatial_prediction');
    return;
  }
  const fromUrl = readUrlState();
  S.index = fromUrl == null ? 0 : clamp(fromUrl, 0, Math.max(0, S.total - 1));
  updateNav();
  loadCurrent();
}

boot();

