/* aim · 训练数据可视化 —— 前端（原生 JS + canvas，无构建、无外部依赖）。
 *
 * 数据流：/api/samples → 样本列表；/api/sample/<split>/<name> → 一份 JSON 载荷
 * （逐帧全场 players + 遮挡矩阵 obstruction + 逐帧道具 + A/B 单人视图）。
 * 页面**不再请求** /api/view/...：样本里已经没有 192×256×5 的视角渲染图了。
 *
 * 样本的新布局（见 tasks/aim.py）：
 *   ctx_player     (T,10,20)  窗口内每一帧、全场 10 个玩家
 *   ctx_player_mask(T,10)     0 死亡 / 1 A(击杀者) / 2 B(受害者) / 3 其他存活
 *   obstruction    (T,10,10)  逐帧两两遮挡：0 通 / 1 烟 / 2 墙（有方向）
 * 载荷把这些压成 channel-major 的 `players`（每个物理量一个 (T,10) 矩阵），
 * 前端取"第 t 帧第 p 个人"就是 `players.x[t][p]`。`d.atk` / `d.tgt` 只是
 * mask 为 1 / 2 的那两行的单人视图，长度 T（HUD / 特征表 / raw 色带用它）。
 *
 * 页面上的所有几何都在这个文件里画，两套坐标系：
 *   世界（游戏）坐标 HU  ——  x/y 是脚底，z 也含（脚底）；眼位 = z + eye_z
 *   屏幕               ——  等角距 yaw/pitch 网格（90° × 73.74°）
 *
 * 投影公式（和样本侧生成一致）：
 *   yaw_p   = atan2(dy, dx)
 *   pitch_p = -asin(dz / |d|)                （pitch 正 = 向下看）
 *   sx = W/2 - wrap180(yaw_p - yaw0) / FOV_H * W      ← 负号！
 *   sy = H/2 + (pitch_p - pitch0)     / FOV_V * H
 *
 * 负号不能省：Source 的 yaw 从上方看逆时针增大（yaw 0 看 +X、yaw +90 看 +Y），
 * 玩家的右手是 f × up = (sin y, −cos y)，所以 yaw 变大 = 往**左**看 = 屏幕**左**边。
 * 写成 `+` 就会和真实第一人称画面左右镜像。
 */

'use strict';

const $ = (id) => document.getElementById(id);
const RAD = Math.PI / 180;
const DEG = 180 / Math.PI;

// 画布/数据里出错时别只在 console 里冒烟：给页面留一条可抓的错误记录（--dump-dom
// / 手动检查都能看到），同时弹 toast。
window.addEventListener('error', (e) => {
  const text = `${e.message} @ ${String(e.filename).split('/').pop()}:${e.lineno}`;
  if (document.body) document.body.dataset.lastError = text;
  if (typeof toast === 'function') toast(`前端异常：${text}`);
});

const SPLITS = ['train', 'valid', 'test'];
const SPLIT_LABEL = { train: 'train', valid: 'valid', test: 'test' };
const TEAM_NAME = ['CT', 'T'];
const TEAM_COLOR = ['#4da3ff', '#f5a524'];

const S = {
  info: null,
  samples: [],
  current: null,
  data: null,
  frame: 0,
  playing: false,
  raf: 0,
  acc: 0,
  lastTs: 0,
  speed: 0.5,
  loadToken: 0,
  radarImg: null,
  rawCells: { atk: [], tgt: [] },
  // 武器图标：显示名 → PNG URL（来自 /api/weapon-icons，素材是 3D 回放器那份）。
  weaponIcons: null,
  weaponIconNorm: null,
  opts: { trail: true, labels: true, boxes: true, grid: true },
  featCells: {},
  // 当前的逐帧表 / 遮挡矩阵：建一次 DOM，之后每帧只改内容（10 行 / 100 格，改起来很快）。
  ctxRows: [],
  obstrCells: [],           // 长度 100，index = i * 10 + j
  obstrVals: null,          // 上一帧的 100 个值，避免每帧无脑写 textContent
  obstrTimelineSig: '',     // 时间轴色带只在换样本时重画底图，之后只挪游标
  // 模型预测：`/api/model` 的元信息 + 当前样本的 T 个预测点（每帧一个，`/api/predict`）。
  // `predOn=false`（URL 里 `pred=0`）就不去跑模型 —— CPU 机器上推理很慢。
  predModel: null,
  pred: null,
  predToken: 0,
  predOn: true,
};

// ── 小工具 ────────────────────────────────────────────────────────────

function wrap180(a) {
  let x = (a + 180) % 360;
  if (x < 0) x += 360;
  return x - 180;
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function fmtBytes(n) {
  if (!n && n !== 0) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function fmtSigned(v, digits = 2) {
  const s = v.toFixed(digits);
  return v >= 0 ? `+${s}` : s;
}

function teamName(v) {
  const t = v >= 0.5 ? 1 : 0;
  return TEAM_NAME[t] || '?';
}

function teamOf(v) {
  return v >= 0.5 ? 1 : 0;
}

// ── 逐帧全场 / 遮挡的小访问器 ─────────────────────────────────────────
//
// 载荷是 channel-major：`players.phys.x[t][p]`（第 t 帧第 p 个人）。
// 这些函数统一处理"样本缺这一列 / 越界"的情况，省得每个调用点各写一次兜底。

function pPhys(d, key, t, p) {
  const ch = d.players && d.players.phys ? d.players.phys[key] : null;
  const row = ch ? ch[t] : null;
  return row && row[p] != null ? row[p] : 0;
}

function pMask(d, t, p) {
  const row = d.players && d.players.mask ? d.players.mask[t] : null;
  return row && row[p] != null ? row[p] : 0;
}

function pWeaponIdx(d, t, p) {
  const row = d.players && d.players.weapon ? d.players.weapon[t] : null;
  return row && row[p] != null ? row[p] : -1;
}

function pWeaponName(d, t, p) {
  const idx = pWeaponIdx(d, t, p);
  if (idx < 0) return null;
  return (d.weapons.names || [])[idx] || `weapon#${idx}`;
}

//: 当前帧所有存活玩家的下标（mask != 0）。死人在雷达 / 第一人称里都不画。
function alivePlayers(d, t) {
  const out = [];
  const mask = d.players && d.players.mask ? d.players.mask[t] : null;
  const n = mask ? mask.length : 0;
  for (let p = 0; p < n; p++) if (mask[p]) out.push(p);
  return out;
}

function obstrAt(d, t, i, j) {
  const frame = d.obstruction ? d.obstruction[t] : null;
  const row = frame ? frame[i] : null;
  return row && row[j] != null ? row[j] : 0;
}

function obstrName(d, v) {
  const names = d.obstruction_names || {};
  return names[String(v)] || String(v);
}

function obstrClass(v) {
  return v === 1 ? 'o1' : v === 2 ? 'o2' : 'o0';
}

// ── 视野（FOV）≠ 遮挡 ──────────────────────────────────────────────────
//
// 这两个量经常被看成一件事，但它们回答的是不同的问题：
//
//   obstruction[i][j]  **中间挡不挡**（射线打不打得穿墙 / 烟），和朝哪看无关；
//   viewState(i, j)    **i 有没有转过脸去看 j**，只由 i 的 yaw/pitch 决定。
//
// "i 此刻能不能看见 j" = 两者同时成立。所以 `obstruction == 通` 只说明"中间没有
// 东西"，B 背对 A 时它照样是 通（实测 de_anubis_f0_r0_k001 第 233 帧就是这种：
// A→B 通，B 的 yaw 与 B→A 差 179°，A 在 B 背后）。页面上的斜线格、HUD 的
// `对手可见`、遮挡面板的读数都靠这个函数把两件事分开写。
//
// 半角取 `geometry.fov_h / 2`（Source 水平 90°、垂直 73.74°，和第一人称画布同一套）。
function viewState(d, i, viewer, p) {
  const g = d.geometry || {};
  const halfH = (g.fov_h || 90) / 2;
  const halfV = (g.fov_v || 73.74) / 2;
  const ex = pPhys(d, 'x', i, viewer);
  const ey = pPhys(d, 'y', i, viewer);
  const ez = pPhys(d, 'z', i, viewer) + pPhys(d, 'eye_z', i, viewer);
  const tx = pPhys(d, 'x', i, p);
  const ty = pPhys(d, 'y', i, p);
  const tz = pPhys(d, 'z', i, p) + pPhys(d, 'box_h', i, p) * 0.6;
  const dx = tx - ex;
  const dy = ty - ey;
  const dz = tz - ez;
  const dist = Math.hypot(dx, dy, dz) || 1;
  const yawTo = Math.atan2(dy, dx) * DEG;
  const pitchTo = -Math.asin(clamp(dz / dist, -1, 1)) * DEG;
  const dYaw = wrap180(yawTo - pPhys(d, 'yaw', i, viewer));
  const dPitch = pitchTo - pPhys(d, 'pitch', i, viewer);
  return {
    dYaw, dPitch, dist,
    inFov: Math.abs(dYaw) <= halfH && Math.abs(dPitch) <= halfV,
    behind: Math.abs(dYaw) > 120,        // 大致"背对"：>120° 基本是侧后到正后方
  };
}

//: `viewState` 的一句话说法（HUD / 面板读数共用，别两处各写一套措辞）。
function viewText(vs) {
  if (!vs) return '—';
  if (vs.inFov) return `视野内 ${Math.abs(vs.dYaw).toFixed(0)}°`;
  return `${vs.behind ? '背对' : '视野外'} ${Math.abs(vs.dYaw).toFixed(0)}°`;
}

function apiPath(...parts) {
  return '/' + parts.map((p) => encodeURIComponent(String(p))).join('/');
}

async function apiJson(path) {
  const res = await fetch(path, { cache: 'no-store' });
  const ct = res.headers.get('content-type') || '';
  const body = ct.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = body && body.error ? body.error : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

let toastTimer = 0;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4200);
}

function setLoading(on) {
  $('loading').hidden = !on;
}

// ── 样本列表 ──────────────────────────────────────────────────────────

function readUrlState() {
  const q = new URLSearchParams(location.search);
  const num = (key) => {
    if (!q.has(key)) return null;
    const v = Number(q.get(key));
    return Number.isFinite(v) ? v : null;
  };
  const flag = (key) => {
    if (!q.has(key)) return null;
    const v = q.get(key);
    return !(v === '0' || v === 'false');
  };
  return {
    sample: q.get('sample'),
    frame: num('frame'),
    trail: flag('trail'),
    labels: flag('labels'),
    boxes: flag('boxes'),
    grid: flag('grid'),
    pred: flag('pred'),
  };
}

function applyUrlState(u) {
  const set = (input, key, stateKey) => {
    if (u[key] == null) return;
    $(input).checked = u[key];
    if (stateKey) S.opts[stateKey] = u[key];
  };
  set('opt-trail', 'trail', 'trail');
  set('opt-labels', 'labels', 'labels');
  set('opt-boxes', 'boxes', 'boxes');
  set('opt-grid', 'grid', 'grid');
  if (u.pred != null) S.predOn = u.pred;
}

async function boot() {
  bindEvents();
  loadWeaponIcons();               // 和扫样本并行；图标晚到会自己补画
  loadModelInfo();                 // 同上：模型信息晚到不影响页面其它部分
  const urlState = readUrlState();
  applyUrlState(urlState);
  try {
    await refreshSamples();
  } catch (err) {
    toast(`扫描样本失败：${err.message}`);
    return;
  }
  const saved = localStorage.getItem('aim-vis.sample');
  const pick =
    S.samples.find((s) => s.id === urlState.sample) ||
    S.samples.find((s) => s.id === saved) ||
    S.samples[0];
  if (pick) {
    await selectSample(pick.id);
    if (urlState.frame != null && S.data) setFrame(urlState.frame);
  } else {
    $('sample-name').textContent = '没有扫到样本';
    $('sample-path').textContent = '先跑 python -m scripts.training_data.prepare --task aim --overwrite';
  }
}

async function refreshSamples() {
  const info = await apiJson('/api/samples');
  S.info = info;
  S.samples = info.samples || [];
  $('data-dir').textContent = info.data_dir || '';
  buildSampleSelect();
}

function buildSampleSelect() {
  const sel = $('sample-select');
  sel.innerHTML = '';
  if (!S.samples.length) {
    const opt = document.createElement('option');
    opt.textContent = '（没有样本）';
    sel.appendChild(opt);
    return;
  }
  for (const split of SPLITS) {
    const items = S.samples.filter((s) => s.split === split);
    if (!items.length) continue;
    const group = document.createElement('optgroup');
    group.label = `${SPLIT_LABEL[split]} · ${items.length}`;
    for (const s of items) {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = s.name.replace(/\.npz$/i, '');
      group.appendChild(opt);
    }
    sel.appendChild(group);
  }
}

async function selectSample(id) {
  const sample = S.samples.find((s) => s.id === id);
  if (!sample) return;
  const token = ++S.loadToken;
  setLoading(true);
  stopPlayback();
  try {
    const data = await apiJson(
      apiPath('api', 'sample', sample.split, sample.name)
    );
    if (token !== S.loadToken) return;
    S.current = sample;
    S.data = data;
    S.frame = 0;
    $('sample-select').value = id;
    localStorage.setItem('aim-vis.sample', id);
    try {
      const url = new URL(location.href);
      url.searchParams.set('sample', id);
      url.searchParams.delete('frame');
      history.replaceState(null, '', url);
    } catch { /* file:// 之类不支持就不写 */ }

    renderMeta();
    renderWarnings();
    buildCtxTable();            // 逐帧玩家表：只建骨架，内容由 render() 每帧刷
    buildObstruction();         // 10×10 遮挡网格 + A→B 时间轴色带
    renderFilesTable();
    buildFeatureRows();
    buildRawStrips();
    buildShotTicks();
    loadRadarImage();

    $('frame-slider').max = String(data.frames - 1);
    $('frame-slider').value = '0';
    $('time-end').textContent = `${data.duration.toFixed(3)} s`;
    $('obstr-time-end').textContent = `${data.duration.toFixed(3)} s`;
    render();
    loadPrediction();              // 不 await：模型推理比页面渲染慢，先让页面出来
  } catch (err) {
    if (token === S.loadToken) toast(`读取样本失败：${err.message}`);
  } finally {
    if (token === S.loadToken) setLoading(false);
  }
}

function stepSample(delta) {
  if (!S.samples.length || !S.current) return;
  const idx = S.samples.findIndex((s) => s.id === S.current.id);
  const next = S.samples[(idx + delta + S.samples.length) % S.samples.length];
  if (next) selectSample(next.id);
}

// ── 事件 ──────────────────────────────────────────────────────────────

function bindEvents() {
  $('sample-select').addEventListener('change', (e) => selectSample(e.target.value));
  $('prev').addEventListener('click', () => stepSample(-1));
  $('next').addEventListener('click', () => stepSample(1));
  $('rescan').addEventListener('click', async () => {
    try {
      setLoading(true);
      await apiJson('/api/rescan');
      await refreshSamples();
      const saved = localStorage.getItem('aim-vis.sample');
      const pick = S.samples.find((s) => s.id === saved) || S.samples[0];
      if (pick) await selectSample(pick.id);
      toast(`重新扫描完成：${S.samples.length} 个样本`);
    } catch (err) {
      toast(`重新扫描失败：${err.message}`);
    } finally {
      setLoading(false);
    }
  });

  const slider = $('frame-slider');
  slider.addEventListener('input', () => {
    stopPlayback();
    setFrame(Number(slider.value), { fromSlider: true });
  });

  $('play').addEventListener('click', togglePlayback);
  $('step-back').addEventListener('click', () => { stopPlayback(); setFrame(S.frame - 1); });
  $('step-fwd').addEventListener('click', () => { stopPlayback(); setFrame(S.frame + 1); });
  $('to-first').addEventListener('click', () => { stopPlayback(); setFrame(0); });
  $('to-last').addEventListener('click', () => { stopPlayback(); setFrame(S.data ? S.data.frames - 1 : 0); });
  $('speed').addEventListener('change', (e) => { S.speed = Number(e.target.value) || 0.5; });

  $('opt-trail').addEventListener('change', (e) => { S.opts.trail = e.target.checked; render(); });
  $('opt-labels').addEventListener('change', (e) => { S.opts.labels = e.target.checked; render(); });
  $('opt-boxes').addEventListener('change', (e) => { S.opts.boxes = e.target.checked; render(); });
  $('opt-grid').addEventListener('change', (e) => { S.opts.grid = e.target.checked; render(); });

  // 遮挡时间轴：点哪一帧跳哪一帧（色带画在 canvas 上，坐标要按 CSS 缩放换算）。
  const obstrCv = $('obstr-timeline');
  obstrCv.addEventListener('click', (e) => {
    if (!S.data) return;
    const rect = obstrCv.getBoundingClientRect();
    const W = obstrCv.width;
    const x = (e.clientX - rect.left) * (W / Math.max(rect.width, 1));
    const T = S.data.frames;
    const k = clamp(Math.round((x / W) * (T - 1)), 0, T - 1);
    stopPlayback();
    setFrame(k);
  });
  // 悬停时给出"第几帧 / 什么遮挡"的提示，不用点也能看。
  obstrCv.addEventListener('mousemove', (e) => {
    if (!S.data) return;
    const rect = obstrCv.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (obstrCv.width / Math.max(rect.width, 1));
    const T = S.data.frames;
    const k = clamp(Math.round((x / obstrCv.width) * (T - 1)), 0, T - 1);
    const a = S.data.atk.index;
    const b = S.data.tgt.index;
    if (a == null || b == null) return;
    obstrCv.title = `第 ${k} 帧 · ${S.data.seconds[k].toFixed(3)} s · A→B = ${obstrName(S.data, obstrAt(S.data, k, a, b))}`;
  });

  window.addEventListener('keydown', (e) => {
    if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (!S.data) return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); stopPlayback(); setFrame(S.frame - (e.shiftKey ? 10 : 1)); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); stopPlayback(); setFrame(S.frame + (e.shiftKey ? 10 : 1)); }
    else if (e.key === ' ') { e.preventDefault(); togglePlayback(); }
    else if (e.key === 'Home') { e.preventDefault(); stopPlayback(); setFrame(0); }
    else if (e.key === 'End') { e.preventDefault(); stopPlayback(); setFrame(S.data.frames - 1); }
  });
}

// ── 帧 / 播放 ─────────────────────────────────────────────────────────

function setFrame(i, opts = {}) {
  if (!S.data) return;
  const T = S.data.frames;
  const next = clamp(Math.round(i), 0, T - 1);
  if (!opts.fromSlider) $('frame-slider').value = String(next);
  S.frame = next;
  render();
}

function togglePlayback() {
  if (!S.data) return;
  if (S.playing) { stopPlayback(); return; }
  if (S.frame >= S.data.frames - 1) S.frame = 0;
  S.playing = true;
  S.acc = 0;
  S.lastTs = 0;
  $('play').textContent = '❚❚';
  $('play').classList.add('active');
  S.raf = requestAnimationFrame(playTick);
}

function stopPlayback() {
  S.playing = false;
  if (S.raf) cancelAnimationFrame(S.raf);
  S.raf = 0;
  $('play').textContent = '▶';
  $('play').classList.remove('active');
}

function playTick(ts) {
  if (!S.playing || !S.data) return;
  if (!S.lastTs) S.lastTs = ts;
  const dt = Math.min((ts - S.lastTs) / 1000, 0.1); // 切标签页回来别一下跳完
  S.lastTs = ts;
  const fps = S.data.fps || (S.data.frames - 1) / Math.max(S.data.duration, 1e-6);
  S.acc += dt * fps * S.speed;
  const step = Math.floor(S.acc);
  if (step > 0) {
    S.acc -= step;
    const next = S.frame + step;
    if (next >= S.data.frames - 1) {
      setFrame(S.data.frames - 1);
      stopPlayback();
      return;
    }
    setFrame(next);
  }
  S.raf = requestAnimationFrame(playTick);
}

// ── 渲染入口 ──────────────────────────────────────────────────────────

function render() {
  if (!S.data) return;
  updateReadouts();
  drawPov('atk');
  drawPov('tgt');
  updateHud('atk');
  updateHud('tgt');
  updateRawStrips();
  drawRadar();
  renderBombTable();
  updateCtxTable();
  renderProjectileTable();
  updateObstruction();
  updateFeatureValues();
  renderPred();
}

function updateReadouts() {
  const d = S.data;
  const i = S.frame;
  $('frame-readout').textContent = `帧 ${i} / ${d.frames - 1}`;
  // 回合时间逐帧存（global_feat[T][4]），直接读当前帧；旧样本只有 4 列就是 null。
  const roundNow = d.global.round_seconds == null
    ? null
    : d.global.round_seconds[i];
  $('time-readout').textContent = roundNow == null
    ? `t = ${d.seconds[i].toFixed(3)} s`
    : `t = ${d.seconds[i].toFixed(3)} s · 回合 ${roundNow.toFixed(2)} s`;
  $('frame-slider').value = String(i);

  const aw = weaponAt('atk', i);
  const tw = weaponAt('tgt', i);
  $('frame-keyinfo').textContent =
    `F${i} · t=${d.seconds[i].toFixed(3)}s · A ${aw || '无武器'} · B ${tw || '无武器'}`;

  const shots = d.atk.phys.shot.filter((v) => v > 0.5).length;
  const shotTimes = d.atk.phys.shot
    .map((v, k) => (v > 0.5 ? k : -1))
    .filter((k) => k >= 0)
    .map((k) => d.seconds[k].toFixed(2));
  $('timeline-hint').textContent = shotTimes.length
    ? `本窗口 A 开枪 ${shots} 次：${shotTimes.slice(0, 10).join(' / ')} s`
    : '本窗口 A 没有开枪';

  const alive = alivePlayers(d, i).length;
  const spans = [];
  if (d.frames > 1) {
    spans.push(`轨迹覆盖 0.000–${d.duration.toFixed(3)} s`);
  }
  spans.push(`第 ${i} 帧存活 ${alive} / ${d.n_players}`);
  spans.push('画面正中 = 当前帧准星');
  $('trail-note').textContent = spans.join(' · ');
}

function weaponAt(kind, i) {
  if (!S.data) return null;
  const w = S.data.weapons;
  const series = kind === 'atk' ? w.atk_series : w.tgt_series;
  const legend = kind === 'atk' ? w.atk : w.tgt;
  const slot = series[i];
  if (slot == null || slot < 0) return null;
  return legend[slot] ? legend[slot].name : null;
}

// ── 武器图标 ──────────────────────────────────────────────────────────
//
// 图标来自 3D 回放器那份 CS2 HUD 素材（`visualizer/static/assets/weapon_icons`），
// 经 `/api/weapon-icons` 查表转发过来。**不是**样本里的东西：npz 只存
// `atk_weapon` / `tgt_weapon` 的全局下标。素材缺失时整条链降级成"只有名字"。
//
// 查表要容忍写法差异（`CZ75 Auto` vs 素材里的 `CZ75-Auto`），所以除精确匹配外
// 再做一次"去掉非字母数字再小写"的归一化匹配。

function iconKey(name) {
  return String(name).toLowerCase().replace(/[^a-z0-9]/g, '');
}

function loadWeaponIcons() {
  apiJson('/api/weapon-icons')
    .then((res) => {
      S.weaponIcons = res.icons || {};
      S.weaponIconNorm = {};
      for (const [name, url] of Object.entries(S.weaponIcons)) {
        S.weaponIconNorm[iconKey(name)] = url;
      }
      if (S.data) render();          // 图标晚到就把已经画好的 HUD 补上
    })
    .catch(() => { S.weaponIcons = {}; S.weaponIconNorm = {}; });
}

function weaponIconUrl(name) {
  if (!name || !S.weaponIcons) return null;
  return S.weaponIcons[name] || (S.weaponIconNorm && S.weaponIconNorm[iconKey(name)]) || null;
}

// ── 概览 ──────────────────────────────────────────────────────────────

function addChip(container, label, value, cls = '') {
  const el = document.createElement('span');
  el.className = `chip ${cls}`.trim();
  el.innerHTML = `${label} <b></b>`;
  el.querySelector('b').textContent = String(value);
  container.appendChild(el);
}

function renderMeta() {
  const d = S.data;
  $('sample-name').textContent = d.name;
  $('sample-path').textContent = d.path;
  const box = $('meta-chips');
  box.innerHTML = '';
  addChip(box, '地图', `${d.map_name} #${d.map_id}`);
  addChip(box, '帧数', d.frames);
  addChip(box, '时长', `${d.duration.toFixed(3)} s`);
  if (d.fps) addChip(box, '采样率', `${d.fps.toFixed(1)} Hz`);
  addChip(box, '维度', `${d.n_features} 维`);
  addChip(box, '队伍', `A ${teamName(d.atk.phys.team[0])} vs B ${teamName(d.tgt.phys.team[0])}`);
  addChip(box, d.hs ? '爆头' : '非爆头', `伤害 ${d.dmg}`, d.hs ? 'hs' : '');
  addChip(box, '全场', `${d.n_players} 人 · 逐帧`);
  if (d.atk.index != null && d.tgt.index != null) {
    addChip(box, '玩家下标', `A=${d.atk.index} · B=${d.tgt.index}`);
  }
  addChip(box, '文件', fmtBytes(d.bytes));
  const shotCount = d.atk.phys.shot.filter((v) => v > 0.5).length;
  addChip(box, 'A 开枪', `${shotCount} 次`);
}

function buildShotTicks() {
  const el = $('shot-ticks');
  el.innerHTML = '';
  const d = S.data;
  if (!d) return;
  const T = d.frames;
  const put = (k, cls) => {
    const tick = document.createElement('span');
    tick.className = `shot-tick ${cls}`;
    const frac = T > 1 ? k / (T - 1) : 0;
    tick.style.left = `calc(9px + (100% - 18px) * ${frac})`;
    el.appendChild(tick);
  };
  for (let k = 0; k < T; k++) if (d.atk.phys.shot[k] > 0.5) put(k, 'atk');
  for (let k = 0; k < T; k++) if (d.tgt.phys.shot[k] > 0.5) put(k, 'tgt');
}

function renderWarnings() {
  const bar = $('warning-bar');
  const list = (S.data && S.data.warnings) || [];
  if (!list.length) { bar.hidden = true; bar.innerHTML = ''; return; }
  bar.hidden = false;
  bar.innerHTML = '';
  const ul = document.createElement('ul');
  for (const w of list) {
    const li = document.createElement('li');
    li.textContent = w;
    ul.appendChild(li);
  }
  bar.appendChild(ul);
}

// ── 逐帧表（雷达旁边：炸弹 / 道具 / 全场玩家，全部跟当前帧） ──────────
//
// 这一节的三块表语义一致，都是**当前帧**：
//   global_feat    逐帧 `(T,5)`：炸弹位置 / 已安放多久 / 回合时间；旧样本只有首帧
//                  快照时才退回"整段一个值"，表里会写明；
//   ctx_projectile 是**当前帧**的生效 / 飞行中道具（每帧一个列表）；
//   ctx_player     是**当前帧**的全场 10 名玩家（载荷里 players，channel-major）。

function renderBombTable() {
  const d = S.data;
  const g = d.global;
  const i = S.frame;
  const t = $('bomb-table');
  t.innerHTML = '';
  if (!g.per_frame) {
    // 旧样本：global_feat 是 (5,) 的首帧快照，前端把它广播成了 T 份
    $('bomb-frame').textContent = '首帧';
  } else {
    $('bomb-frame').textContent = String(i);
  }
  const rows = [
    [
      '回合时间',
      g.round_seconds == null
        ? '未记录（旧样本：4 列 global_feat）'
        : `${g.round_seconds[i].toFixed(2)} s${g.round_clipped[i] ? '（已 clip）' : ''}`,
    ],
    ['x', `${g.x[i].toFixed(2)} HU`],
    ['y', `${g.y[i].toFixed(2)} HU`],
    ['z', `${g.z[i].toFixed(2)} HU`],
    ['已安放', g.planted[i] ? `是 · ${g.planted_seconds[i].toFixed(2)} s` : '否（xyz 是携带者位置）'],
  ];
  for (const [k, v] of rows) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = k;
    const td = document.createElement('td');
    td.textContent = v;
    tr.append(th, td);
    t.appendChild(tr);
  }
  const note = g.per_frame
    ? '未安放时 xyz 是携带者位置 —— 对敌方是隐藏信息。'
    : '旧样本只有首帧快照：这一列整段不变，别当成逐帧数据。';
  $('bomb-note').textContent = note;
}

function renderProjectileTable() {
  const d = S.data;
  if (!d) return;
  const i = S.frame;
  const rows = (d.projectiles && d.projectiles[i]) || [];
  $('proj-frame').textContent = String(i);
  const t = $('proj-table');
  t.innerHTML = '';
  const head = ['槽', '类型', 'x', 'y', 'z', '已存在'];
  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  for (const h of head) {
    const th = document.createElement('th');
    th.textContent = h;
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  t.appendChild(thead);
  const tbody = document.createElement('tbody');
  if (!rows.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = head.length;
    td.className = 'muted';
    td.textContent = `第 ${i} 帧没有生效 / 飞行中的道具（32 个槽全是 -1）`;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
  for (const p of rows) {
    const tr = document.createElement('tr');
    const cells = [
      p.slot,
      p.type,
      p.x.toFixed(0),
      p.y.toFixed(0),
      p.z.toFixed(0),
      `${p.seconds.toFixed(2)} s${p.clipped ? ' (clip)' : ''}`,
    ];
    for (const c of cells) {
      const td = document.createElement('td');
      td.textContent = c;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  t.appendChild(tbody);
}

//: 全场玩家表：列数固定，骨架只建一次（10 行 × 14 列），之后每帧只改字。
const CTX_HEAD = [
  '#', '角色', '队伍', '武器', 'hp', 'armor', '蹲', 'x', 'y', 'z',
  'A 看它', '它看 A', 'B 看它', '它看 B',
];

function buildCtxTable() {
  const d = S.data;
  const t = $('ctx-table');
  t.innerHTML = '';
  S.ctxRows = [];

  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  CTX_HEAD.forEach((h) => {
    const th = document.createElement('th');
    th.textContent = h;
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  t.appendChild(thead);

  const tbody = document.createElement('tbody');
  const n = d.n_players || 10;
  for (let p = 0; p < n; p++) {
    const tr = document.createElement('tr');
    const row = { tr, cells: {} };
    const add = (key, tag = 'td') => {
      const el = document.createElement(tag);
      tr.appendChild(el);
      row.cells[key] = el;
      return el;
    };
    add('idx');
    add('role');
    add('team');
    const wpn = add('weapon');
    const img = document.createElement('img');
    img.alt = '';
    img.hidden = true;
    img.addEventListener('error', () => { img.hidden = true; });
    const wpnName = document.createElement('span');
    wpn.append(img, wpnName);
    row.wpnImg = img;
    row.wpnName = wpnName;
    add('hp');
    add('armor');
    add('crouch');
    add('x');
    add('y');
    add('z');
    for (const key of ['aSees', 'seesA', 'bSees', 'seesB']) {
      const el = add(key);
      el.className = 'obs-cell';
    }
    tbody.appendChild(tr);
    S.ctxRows.push(row);
  }
  t.appendChild(tbody);
}

// 只在小字变化时才写 DOM：播放时每帧都会调这个函数，textContent 写 100+ 次
// 会触发不必要的样式重算。
function setText(el, text) {
  if (el && el.textContent !== text) el.textContent = text;
}

function updateCtxTable() {
  const d = S.data;
  if (!d || !S.ctxRows.length) return;
  const i = S.frame;
  $('ctx-frame').textContent = String(i);
  const a = d.atk.index;
  const b = d.tgt.index;
  const obsCell = (el, from, to) => {
    if (from == null || to == null) {
      el.className = 'obs-cell muted';
      el.textContent = '—';
      el.title = '';
      return;
    }
    const dead = pMask(d, i, from) === 0 || pMask(d, i, to) === 0 || from === to;
    if (dead) {
      el.className = 'obs-cell muted';
      el.textContent = '—';
      el.title = from === to ? '自己看自己：不采样' : '有一个人已死亡：遮挡恒为 0，不显示';
      return;
    }
    const v = obstrAt(d, i, from, to);
    el.className = `obs-cell ${obstrClass(v)}`;
    el.textContent = obstrName(d, v);
    el.title = `${from} 看 ${to}：${obstrName(d, v)}（${v}）`;
  };

  for (let p = 0; p < S.ctxRows.length; p++) {
    const row = S.ctxRows[p];
    const m = pMask(d, i, p);
    const dead = m === 0;
    row.tr.classList.toggle('dead', dead);
    row.tr.classList.toggle('is-a', p === a);
    row.tr.classList.toggle('is-b', p === b);

    setText(row.cells.idx, String(p));
    const role = (d.players.mask_names && d.players.mask_names[String(m)]) || String(m);
    setText(row.cells.role, role);
    row.cells.role.className = p === a ? 'role-a' : p === b ? 'role-b' : dead ? 'muted' : '';
    setText(row.cells.team, dead ? '—' : teamName(pPhys(d, 'team', i, p)));
    setText(row.cells.hp, dead ? '—' : pPhys(d, 'hp', i, p).toFixed(0));
    setText(row.cells.armor, dead ? '—' : pPhys(d, 'armor', i, p).toFixed(0));
    setText(row.cells.crouch, dead ? '—' : pPhys(d, 'crouch', i, p).toFixed(2));
    setText(row.cells.x, dead ? '—' : pPhys(d, 'x', i, p).toFixed(0));
    setText(row.cells.y, dead ? '—' : pPhys(d, 'y', i, p).toFixed(0));
    setText(row.cells.z, dead ? '—' : pPhys(d, 'z', i, p).toFixed(0));

    const wname = dead ? null : pWeaponName(d, i, p);
    setText(row.wpnName, dead ? '—' : (wname || '无武器'));
    const url = wname ? weaponIconUrl(wname) : null;
    if (url) {
      if (row.wpnImg.getAttribute('src') !== url) row.wpnImg.src = url;
      row.wpnImg.hidden = false;
    } else {
      row.wpnImg.hidden = true;
    }

    obsCell(row.cells.aSees, a, p);
    obsCell(row.cells.seesA, p, a);
    obsCell(row.cells.bSees, b, p);
    obsCell(row.cells.seesB, p, b);
  }
}

function renderFilesTable() {
  const d = S.data;
  const t = $('files-table');
  t.innerHTML = '';
  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  for (const h of ['key', 'shape', 'dtype']) {
    const th = document.createElement('th');
    th.textContent = h;
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  t.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const f of d.files) {
    const tr = document.createElement('tr');
    for (const v of [f.key, `(${f.shape.join(', ')})`, f.dtype]) {
      const td = document.createElement('td');
      td.textContent = v;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  t.appendChild(tbody);
  $('files-count').textContent = String(d.files.length);
}

// ── 特征表 ────────────────────────────────────────────────────────────

const PHYS_LABEL = {
  x: (v) => `${v.toFixed(1)} HU`,
  y: (v) => `${v.toFixed(1)} HU`,
  z: (v) => `${v.toFixed(1)} HU`,
  yaw: (v) => `${v.toFixed(2)}°`,
  pitch: (v) => `${v.toFixed(2)}°`,
  yaw_sin: (v) => v.toFixed(4),
  yaw_cos: (v) => v.toFixed(4),
  pitch_sin: (v) => v.toFixed(4),
  pitch_cos: (v) => v.toFixed(4),
  vx: (v) => `${v.toFixed(0)} HU/s`,
  vy: (v) => `${v.toFixed(0)} HU/s`,
  vz: (v) => `${v.toFixed(0)} HU/s`,
  hp: (v) => v.toFixed(0),
  crouch: (v) => v.toFixed(3),
  armor: (v) => v.toFixed(0),
  helmet: (v) => (v > 0.5 ? '有 (1)' : '无 (0)'),
  team: (v) => `${teamName(v)} (${v > 0.5 ? 1 : 0})`,
  shot: (v) => (v > 0.5 ? '开枪 (1)' : '— (0)'),
  since_shot: (v) => (v >= 99.9 ? '≥100 s（饱和）' : `${v.toFixed(2)} s`),
  has_kill: (v) => (v > 0.5 ? '是 (1)' : '否 (0)'),
  blinded: (v) => (v > 0.5 ? '致盲 (1)' : '— (0)'),
  defuser: (v) => (v > 0.5 ? '有 (1)' : '无 (0)'),
  eye_z: (v) => `${v.toFixed(2)} HU`,
  box_h: (v) => `${v.toFixed(1)} HU`,
};

const FEATURE_NOTE = {
  x: '地图归一化逆变换 → 世界坐标',
  y: '地图归一化逆变换 → 世界坐标',
  z: '脚底高度（不是眼睛）',
  yaw: 'atan2(sin, cos)，±180° 不跳变',
  pitch: '正 = 向下看（Source 约定）',
  yaw_sin: 'sin(yaw)，原值 = 0.5 + 0.5·sin',
  yaw_cos: 'cos(yaw)，原值 = 0.5 + 0.5·cos',
  pitch_sin: 'sin(pitch)，原值 = 0.5 + 0.5·sin',
  pitch_cos: 'cos(pitch)，原值 = 0.5 + 0.5·cos',
  vx: 'half = 300 HU/s',
  vy: 'half = 300 HU/s',
  vz: 'half = 1200 HU/s',
  hp: '÷100',
  crouch: '相机连续斜坡 0..1',
  armor: '÷100',
  helmet: '0/1',
  team: 'CT=0 / T=1',
  shot: '本帧是否开枪',
  since_shot: '距上次开枪秒数，≥100 s 为饱和',
  has_kill: '截至本帧是否已击杀',
  blinded: '本帧是否被闪致盲（玩家私有）',
  defuser: '本帧是否带拆弹器（玩家私有）',
};

function rawLabel(v) {
  const s = v.toFixed(4);
  return s === '-0.0000' ? '0.0000' : s;
}

function buildFeatureRows() {
  const d = S.data;
  const body = $('feat-body');
  body.innerHTML = '';
  S.featCells = {};

  const section = (text) => {
    const tr = document.createElement('tr');
    tr.className = 'section';
    const td = document.createElement('td');
    td.colSpan = 5;
    td.textContent = text;
    tr.appendChild(td);
    body.appendChild(tr);
  };

  // 列数不写死：样本是 18/19（旧）还是 20（当前）由载荷里的 features 决定。
  section(`atk / tgt 的 ${d.features.length} 列（AIM_FEATURES）`);

  for (const name of d.features) {
    const tr = document.createElement('tr');
    tr.dataset.feature = name;
    const tdName = document.createElement('td');
    const nameEl = document.createElement('span');
    nameEl.className = 'fname';
    nameEl.textContent = name;
    tdName.appendChild(nameEl);
    if (FEATURE_NOTE[name]) {
      const noteEl = document.createElement('span');
      noteEl.className = 'fnote';
      noteEl.textContent = FEATURE_NOTE[name];
      tdName.appendChild(noteEl);
    }
    tr.appendChild(tdName);
    const cells = {};
    for (const key of ['atkRaw', 'atkPhys', 'tgtRaw', 'tgtPhys']) {
      const td = document.createElement('td');
      tr.appendChild(td);
      cells[key] = td;
    }
    S.featCells[name] = cells;
    body.appendChild(tr);
  }

  section('派生量（不由文件直接存储，前端按 crouch 算出来）');
  for (const name of ['eye_z', 'box_h']) {
    const tr = document.createElement('tr');
    tr.className = 'derived';
    const tdName = document.createElement('td');
    tdName.textContent = name;
    tr.appendChild(tdName);
    const cells = {};
    for (const key of ['atkRaw', 'atkPhys', 'tgtRaw', 'tgtPhys']) {
      const td = document.createElement('td');
      tr.appendChild(td);
      cells[key] = td;
    }
    cells.atkRaw.textContent = '—';
    cells.tgtRaw.textContent = '—';
    S.featCells[name] = cells;
    body.appendChild(tr);
  }
}

function buildRawStrips() {
  const d = S.data;
  for (const kind of ['atk', 'tgt']) {
    const el = $(`raw-strip-${kind}`);
    el.innerHTML = '';
    el.style.setProperty('--cols', String(d.features.length));
    const cells = [];
    d.features.forEach((name, idx) => {
      const cell = document.createElement('div');
      cell.className = 'raw-cell';
      cell.title = `${idx} · ${name}${FEATURE_NOTE[name] ? ' — ' + FEATURE_NOTE[name] : ''}`;
      const idxEl = document.createElement('span');
      idxEl.className = 'idx';
      idxEl.textContent = String(idx);
      const valEl = document.createElement('span');
      valEl.className = 'val';
      valEl.textContent = '—';
      cell.append(idxEl, valEl);
      el.appendChild(cell);
      cells.push({ cell, valEl, name });
    });
    S.rawCells[kind] = cells;
    $(`raw-count-${kind}`).textContent = String(d.features.length);
  }
}

function updateRawStrips() {
  const d = S.data;
  if (!d) return;
  const i = S.frame;
  for (const kind of ['atk', 'tgt']) {
    const raw = d[kind].raw[i] || [];
    const cells = S.rawCells[kind] || [];
    for (let k = 0; k < cells.length; k++) {
      const { cell, valEl, name } = cells[k];
      const v = Number(raw[k] ?? 0);
      valEl.textContent = rawLabel(v);
      cell.style.setProperty('--v', String(v));
      const on = v > 0.5;
      cell.classList.toggle('raw-shot', name === 'shot' && on);
      cell.classList.toggle('raw-blind', name === 'blinded' && on);
      cell.classList.toggle('raw-kill', name === 'has_kill' && on);
      cell.classList.toggle('raw-kit', name === 'defuser' && on);
    }
  }
}

function updateFeatureValues() {
  const d = S.data;
  const i = S.frame;
  for (const [name, cells] of Object.entries(S.featCells)) {
    const isDerived = name === 'eye_z' || name === 'box_h';
    if (!isDerived) {
      const idx = d.features.indexOf(name);
      if (idx < 0) continue;
      const ar = d.atk.raw[i] ? d.atk.raw[i][idx] : 0;
      const tr = d.tgt.raw[i] ? d.tgt.raw[i][idx] : 0;
      cells.atkRaw.textContent = rawLabel(ar);
      cells.tgtRaw.textContent = rawLabel(tr);
    }
    const fmt = PHYS_LABEL[name] || ((v) => String(v));
    cells.atkPhys.textContent = fmt(d.atk.phys[name][i]);
    cells.tgtPhys.textContent = fmt(d.tgt.phys[name][i]);
  }

  const shotRow = $('feat-body').querySelector('tr[data-feature="shot"]');
  const blindRow = $('feat-body').querySelector('tr[data-feature="blinded"]');
  const killRow = $('feat-body').querySelector('tr[data-feature="has_kill"]');
  const kitRow = $('feat-body').querySelector('tr[data-feature="defuser"]');
  if (shotRow) shotRow.classList.toggle('row-hit',
    d.atk.phys.shot[i] > 0.5 || d.tgt.phys.shot[i] > 0.5);
  if (blindRow) blindRow.classList.toggle('row-hit',
    d.atk.phys.blinded[i] > 0.5 || d.tgt.phys.blinded[i] > 0.5);
  if (killRow) killRow.classList.toggle('row-on',
    d.atk.phys.has_kill[i] > 0.5 || d.tgt.phys.has_kill[i] > 0.5);
  if (kitRow) kitRow.classList.toggle('row-on',
    d.atk.phys.defuser[i] > 0.5 || d.tgt.phys.defuser[i] > 0.5);
}

// ── HUD ───────────────────────────────────────────────────────────────

function eyeOf(p, i) {
  return [p.x[i], p.y[i], p.z[i] + p.eye_z[i]];
}

function targetMetrics(obs, opp, i) {
  const eye = eyeOf(obs, i);
  const tx = opp.x[i];
  const ty = opp.y[i];
  const tz = opp.z[i] + opp.box_h[i] * 0.6;
  const dx = tx - eye[0];
  const dy = ty - eye[1];
  const dz = tz - eye[2];
  const dist = Math.hypot(dx, dy, dz);
  const yawTo = Math.atan2(dy, dx) * DEG;
  const pitchTo = -Math.asin(clamp(dz / (dist || 1), -1, 1)) * DEG;
  const dYaw = wrap180(yawTo - obs.yaw[i]);
  const dPitch = pitchTo - obs.pitch[i];
  const err = Math.hypot(dYaw * Math.cos(pitchTo * RAD), dPitch);
  return { dist, dYaw, dPitch, err, point: [tx, ty, tz] };
}

function hudItem(label, value, cls = '') {
  return `<span class="hud-item ${cls}">${label} <b>${value}</b></span>`;
}

function updateHud(kind) {
  const d = S.data;
  const i = S.frame;
  const me = kind === 'atk' ? d.atk.phys : d.tgt.phys;
  const other = kind === 'atk' ? d.tgt.phys : d.atk.phys;
  const m = targetMetrics(me, other, i);
  const speed = Math.hypot(me.vx[i], me.vy[i], me.vz[i]);
  const wname = weaponAt(kind, i);
  const wins = weaponIconUrl(wname);

  // 全场信息也跟着当前帧走：存活人数、最近的敌人（在 players 里找，不只 A/B）、A→B 遮挡。
  const viewer = kind === 'atk' ? d.atk.index : d.tgt.index;
  const alive = alivePlayers(d, i);
  const myTeam = teamOf(me.team[i]);
  let enemies = 0;
  let near = null;
  for (const p of alive) {
    if (p === viewer) continue;
    const team = teamOf(pPhys(d, 'team', i, p));
    if (team === myTeam) continue;
    enemies += 1;
    const dist = Math.hypot(
      pPhys(d, 'x', i, p) - me.x[i],
      pPhys(d, 'y', i, p) - me.y[i]
    );
    const obs = viewer == null ? 0 : obstrAt(d, i, viewer, p);
    if (!near || dist < near.dist) near = { p, dist, obs };
  }
  const abv = d.atk.index == null || d.tgt.index == null
    ? null
    : obstrAt(d, i, d.atk.index, d.tgt.index);
  // 「看得见」= 中间没挡（obstruction）**且**转过脸了（FOV）。两个条件分开算，
  // 页面上一律写清楚是哪一条不满足 —— 只写"通"会让人以为看见了。
  const myIdx = viewer;
  const otherIdx = kind === 'atk' ? d.tgt.index : d.atk.index;
  const myObs = (myIdx == null || otherIdx == null) ? null
    : obstrAt(d, i, myIdx, otherIdx);
  const myView = (myIdx == null || otherIdx == null) ? null
    : viewState(d, i, myIdx, otherIdx);
  const canSee = (myObs == null || (otherIdx != null && pMask(d, i, otherIdx) === 0)) ? '—'
    : myObs !== 0 ? `否（${obstrName(d, myObs)}）`
      : myView && myView.inFov ? '是'
        : `否（通 · ${viewText(myView)}）`;
  const nearView = (viewer == null || !near) ? null : viewState(d, i, viewer, near.p);

  const items = [
    // 装备第一项就是手里的枪：图标是 CS2 官方 HUD 图，逐帧跟着 `*_weapon` 变。
    `<span class="hud-item hud-wpn${wname ? '' : ' dim'}" title="当前武器（atk_series / tgt_series）">` +
      (wins ? `<img src="${wins}" alt="" onerror="this.remove()">` : '') +
      `<b>${wname || '无武器'}</b></span>`,
    hudItem('yaw', `${me.yaw[i].toFixed(1)}°`),
    hudItem('pitch', `${me.pitch[i].toFixed(1)}°`),
    hudItem('pos', `${me.x[i].toFixed(0)}, ${me.y[i].toFixed(0)}, ${me.z[i].toFixed(0)}`),
    hudItem('速度', `${speed.toFixed(0)} HU/s`),
    hudItem('hp', me.hp[i].toFixed(0)),
    hudItem('armor', me.armor[i].toFixed(0)),
    hudItem('头盔', me.helmet[i] > 0.5 ? '有' : '无'),
    hudItem('拆弹器', me.defuser[i] > 0.5 ? '有' : '无', me.defuser[i] > 0.5 ? 'good' : ''),
    hudItem('crouch', me.crouch[i].toFixed(2)),
    hudItem('shot', me.shot[i] > 0.5 ? '开枪' : '—', me.shot[i] > 0.5 ? 'hot' : ''),
    hudItem('距上次开枪', me.since_shot[i] >= 99.9 ? '≥100s' : `${me.since_shot[i].toFixed(2)}s`),
    hudItem('已击杀', me.has_kill[i] > 0.5 ? '是' : '否', me.has_kill[i] > 0.5 ? 'good' : ''),
    hudItem('致盲', me.blinded[i] > 0.5 ? '是' : '否', me.blinded[i] > 0.5 ? 'hot' : ''),
    hudItem('存活', `${alive.length} / ${d.n_players}`),
    hudItem('敌方', `${enemies} 人`, enemies ? '' : 'good'),
    near
      ? hudItem('最近敌人', `#${near.p} · ${near.dist.toFixed(0)} HU · ${obstrName(d, near.obs)}`
        + ` · ${viewText(nearView)}`,
        near.obs === 0 && nearView && nearView.inFov ? 'good' : '')
      : hudItem('最近敌人', '没有敌人'),
    hudItem('A→B 遮挡', abv == null ? '—' : obstrName(d, abv)),
    hudItem('对手可见', canSee, canSee === '是' ? 'good' : myObs === 0 ? 'hot' : ''),
    hudItem('对手距离', `${m.dist.toFixed(0)} HU / ${(m.dist * d.geometry.hu_to_m).toFixed(1)} m`),
    hudItem('角误差', `${m.err.toFixed(2)}° (Δyaw ${fmtSigned(m.dYaw)}° Δpitch ${fmtSigned(m.dPitch)}°)`,
      m.err < 2 ? 'good' : m.err > 8 ? 'hot' : ''),
  ];
  const el = $(`hud-${kind}`);
  el.innerHTML = items.join('');
}

// ── 画布：第一人称 ────────────────────────────────────────────────────

const BOX_EDGES = [
  [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
  [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
];

// 视线角偏移 → 画布 x。**负号**见文件头：yaw 变大是往左看，画布左边是 x 小。
function yawToX(dyaw, W, fovH) {
  return W / 2 - (dyaw / fovH) * W;
}

function projectPoint(px, py, pz, eye, yaw0, pitch0, W, H, fovH, fovV) {
  const dx = px - eye[0];
  const dy = py - eye[1];
  const dz = pz - eye[2];
  const dist = Math.hypot(dx, dy, dz);
  if (dist < 1e-6) return { x: W / 2, y: H / 2, dyaw: 0, dp: 0, dist: 0 };
  const yaw = Math.atan2(dy, dx) * DEG;
  const pitch = -Math.asin(clamp(dz / dist, -1, 1)) * DEG;
  const dyaw = wrap180(yaw - yaw0);
  const dp = pitch - pitch0;
  return {
    x: yawToX(dyaw, W, fovH),
    y: H / 2 + (dp / fovV) * H,
    dyaw, dp, dist,
  };
}

function drawPov(kind) {
  const d = S.data;
  if (!d) return;
  const i = S.frame;
  const me = kind === 'atk' ? d.atk.phys : d.tgt.phys;
  const other = kind === 'atk' ? d.tgt.phys : d.atk.phys;
  const canvas = $(`canvas-${kind}`);
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const g = d.geometry;
  const fovH = g.fov_h;
  const fovV = g.fov_v;
  const yaw0 = me.yaw[i];
  const pitch0 = me.pitch[i];
  const eye = eyeOf(me, i);
  const feetZ = me.z[i];

  ctx.clearRect(0, 0, W, H);

  drawSky(ctx, W, H, pitch0, fovV);
  if (S.opts.grid) drawGroundGrid(ctx, W, H, eye, feetZ, yaw0, pitch0, fovH, fovV);
  drawHorizon(ctx, W, H, pitch0, fovV);
  drawVignette(ctx, W, H);
  if (S.opts.trail) drawAimTrail(ctx, W, H, me, i, yaw0, pitch0, fovH, fovV);

  const m = targetMetrics(me, other, i);
  if (S.opts.boxes) {
    // 当前帧**全场**活着的人：A/B 之外的其他玩家用队伍色描边 + 遮挡标记
    // （样本现在是逐帧全场布局，不再有"只有第 0 帧才有 ctx_player"的限制）。
    drawOtherPlayers(ctx, kind, i, eye, yaw0, pitch0, W, H, fovH, fovV);

    const box = {
      x: other.x[i], y: other.y[i], z: other.z[i], h: other.box_h[i],
    };
    // enemy 盒（橙、虚线、半透明填充）：在目标/对手的语义下它就是这个人
    drawBox(ctx, box, {
      stroke: 'rgba(255,155,52,0.85)', width: 1.4, dash: [5, 4],
      fill: 'rgba(255,155,52,0.10)',
    }, eye, yaw0, pitch0, W, H, fovH, fovV);
    // target 盒（红、实线、发光）：把"这一次的对手"单独标出来
    drawBox(ctx, box, {
      stroke: 'rgba(255,77,77,1.0)', width: 2.2, dash: [],
      fill: 'rgba(255,77,77,0.10)', glow: 8,
    }, eye, yaw0, pitch0, W, H, fovH, fovV);

    drawBoxLabel(ctx, box, m, eye, yaw0, pitch0, W, H, fovH, fovV);
    drawTargetOffscreen(ctx, W, H, m);
  }

  drawCrosshair(ctx, W, H, me.shot[i] > 0.5, me.blinded[i] > 0.5);
  drawCornerTag(ctx, kind, me, i);
}

function drawVignette(ctx, W, H) {
  const g = ctx.createRadialGradient(
    W / 2, H / 2, Math.min(W, H) * 0.30,
    W / 2, H / 2, Math.max(W, H) * 0.72
  );
  g.addColorStop(0, 'rgba(0,0,0,0)');
  g.addColorStop(1, 'rgba(0,0,0,0.52)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

function drawSky(ctx, W, H, pitch0, fovV) {
  const horizonY = H / 2 + (pitch0 / fovV) * H;
  const hy = clamp(horizonY, 0, H);
  if (hy > 0) {
    const sky = ctx.createLinearGradient(0, 0, 0, Math.max(hy, 1));
    sky.addColorStop(0, '#070d14');
    sky.addColorStop(1, '#16232f');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, hy);
  }
  if (hy < H) {
    const ground = ctx.createLinearGradient(0, hy, 0, H);
    ground.addColorStop(0, '#131b23');
    ground.addColorStop(1, '#05080b');
    ctx.fillStyle = ground;
    ctx.fillRect(0, hy, W, H - hy);
  }
}

function drawHorizon(ctx, W, H, pitch0, fovV) {
  const y = H / 2 + (pitch0 / fovV) * H;
  if (y < 0 || y > H) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(86, 132, 170, 0.45)';
  ctx.lineWidth = 1;
  ctx.setLineDash([7, 5]);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(W, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(140, 178, 210, 0.7)';
  ctx.font = '10px Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.fillText('地平线 0°', W - 6, y - 4);
  ctx.restore();
}

function drawGroundGrid(ctx, W, H, eye, feetZ, yaw0, pitch0, fovH, fovV) {
  const CELL = 128;
  const N = 7;
  const cx = Math.round(eye[0] / CELL);
  const cy = Math.round(eye[1] / CELL);
  const z = feetZ + 0.5;

  ctx.save();
  ctx.strokeStyle = 'rgba(62, 104, 138, 0.72)';
  ctx.lineWidth = 1.05;

  const line = (x0, y0, x1, y1) => {
    const steps = Math.max(2, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 64));
    ctx.beginPath();
    let pen = false;
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const p = projectPoint(
        x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z,
        eye, yaw0, pitch0, W, H, fovH, fovV
      );
      if (!p || Math.abs(p.dyaw) > 100 || Math.abs(p.dp) > 80) {
        pen = false;
        continue;
      }
      if (!pen) { ctx.moveTo(p.x, p.y); pen = true; }
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
  };

  for (let k = -N; k <= N; k++) {
    line((cx + k) * CELL, (cy - N) * CELL, (cx + k) * CELL, (cy + N + 1) * CELL);
    line((cx - N) * CELL, (cy + k) * CELL, (cx + N + 1) * CELL, (cy + k) * CELL);
  }
  ctx.restore();
}

function drawAimTrail(ctx, W, H, me, i, yaw0, pitch0, fovH, fovV) {
  const T = S.data.frames;
  // yaw 存的是 [-180,180] 的 wrapped 角；直接逐点 wrap180 会在跨 ±180 时把
  // "转了 2°" 画成"横扫一整屏"。先沿时间轴展开成连续角，再减去当前帧。
  const absYaw = new Array(T);
  let prev = me.yaw[0];
  absYaw[0] = prev;
  for (let k = 1; k < T; k++) {
    prev += wrap180(me.yaw[k] - prev);
    absYaw[k] = prev;
  }
  const base = absYaw[i];

  const pts = new Array(T);
  for (let k = 0; k < T; k++) {
    const dy = absYaw[k] - base;
    const dp = me.pitch[k] - pitch0;
    pts[k] = (Math.abs(dy) <= 120 && Math.abs(dp) <= 85)
      ? { x: yawToX(dy, W, fovH), y: H / 2 + (dp / fovV) * H }
      : null;
  }

  const stroke = (a, b, color, width) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    let pen = false;
    for (let k = a; k <= b; k++) {
      const p = pts[k];
      if (!p) { pen = false; continue; }
      if (!pen) { ctx.moveTo(p.x, p.y); pen = true; }
      else {
        const q = pts[k - 1];
        if (q && Math.hypot(p.x - q.x, p.y - q.y) > W * 0.55) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
    }
    ctx.stroke();
    ctx.restore();
  };

  // 深色描边打底：底图是墙（浅灰）时纯蓝线几乎看不见
  stroke(i, T - 1, 'rgba(2, 8, 14, 0.45)', 3.2);
  stroke(0, i, 'rgba(2, 8, 14, 0.55)', 3.6);
  stroke(i, T - 1, 'rgba(96, 186, 255, 0.42)', 1.4);   // 未来
  stroke(0, i, 'rgba(120, 205, 255, 1.00)', 2.0);      // 过去

  // 当前帧的准星点（画面正中，和十字重合）
  ctx.save();
  ctx.strokeStyle = 'rgba(112, 190, 255, 0.95)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, 11, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // 开枪帧：点 + 爆发起点的时刻
  const shots = [];
  for (let k = 0; k < T; k++) if (me.shot[k] > 0.5) shots.push(k);
  if (!shots.length) return;

  ctx.save();
  ctx.font = '10px Consolas, monospace';
  ctx.textAlign = 'left';
  for (let s = 0; s < shots.length; s++) {
    const k = shots[s];
    const p = pts[k];
    if (!p) continue;
    ctx.fillStyle = 'rgba(255, 90, 90, 0.95)';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = 'rgba(255, 90, 90, 0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - 4);
    ctx.lineTo(p.x, p.y - 10);
    ctx.stroke();

    if (!S.opts.labels) continue;
    const prev = s > 0 ? shots[s - 1] : null;
    const gap = prev == null ? 1 : S.data.seconds[k] - S.data.seconds[prev];
    if (gap < 0.3) continue; // 连发只标爆发起点，不然满屏都是字
    const label = timeAt(k);
    // 标在点的**右下**：正上方是 target / enemy 的距离标签，会把字盖掉
    const below = p.y + 16 < H - 4;
    const tx = clamp(p.x + 8, 4, W - 62);
    const ty = below ? p.y + 16 : p.y - 14;
    const w = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(6, 10, 14, 0.82)';
    ctx.fillRect(tx - 3, ty - 9, w + 6, 13);
    ctx.fillStyle = 'rgba(255, 170, 170, 0.98)';
    ctx.fillText(label, tx, ty);
  }
  ctx.restore();
}

function timeAt(k) {
  const d = S.data;
  if (!d) return '';
  return `${d.seconds[k].toFixed(2)}s`;
}

function drawBox(ctx, box, style, eye, yaw0, pitch0, W, H, fovH, fovV) {
  const half = S.data.geometry.hull_half;
  const corners = [];
  for (let b = 0; b < 8; b++) {
    const sx = (b & 1) ? 1 : -1;
    const sy = (b & 2) ? 1 : -1;
    const sz = (b & 4) ? 1 : 0;
    corners.push(projectPoint(
      box.x + sx * half, box.y + sy * half, box.z + sz * box.h,
      eye, yaw0, pitch0, W, H, fovH, fovV
    ));
  }

  // 半透明填充：8 个角都在视锥里才算（否则凸包会横穿画面）
  const allVisible = corners.every((p) => p && Math.abs(p.dyaw) < 80 && Math.abs(p.dp) < 70);
  if (allVisible && style.fill) {
    const hull = convexHull(corners.map((p) => [p.x, p.y]));
    if (hull.length >= 3) {
      ctx.save();
      ctx.fillStyle = style.fill;
      ctx.beginPath();
      ctx.moveTo(hull[0][0], hull[0][1]);
      for (let k = 1; k < hull.length; k++) ctx.lineTo(hull[k][0], hull[k][1]);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  ctx.save();
  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = style.width;
  ctx.setLineDash(style.dash || []);
  if (style.glow) {
    ctx.shadowColor = 'rgba(255, 77, 77, 0.8)';
    ctx.shadowBlur = style.glow;
  }
  ctx.beginPath();
  for (const [a, b] of BOX_EDGES) {
    const pa = corners[a];
    const pb = corners[b];
    if (!pa || !pb) continue;
    if (Math.abs(pa.dyaw) > 110 || Math.abs(pb.dyaw) > 110) continue;
    if (Math.abs(pa.dp) > 88 || Math.abs(pb.dp) > 88) continue;
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
  }
  ctx.stroke();
  ctx.restore();
}

// 当前帧里除观察者与对手（A/B 中的另一个）以外的所有存活玩家。
// 队伍色区分（敌方橙 / 队友青）；遮挡（obstruction[t][viewer][p]）不改颜色，只降低
// 透明度 + 在标签后加"烟 / 墙"：包围盒是纯几何的，墙后的人也应该能看见"那里有人"。
function drawOtherPlayers(ctx, kind, i, eye, yaw0, pitch0, W, H, fovH, fovV) {
  const d = S.data;
  const viewer = kind === 'atk' ? d.atk.index : d.tgt.index;
  const opp = kind === 'atk' ? d.tgt.index : d.atk.index;
  if (viewer == null) return;
  const myTeam = teamOf(pPhys(d, 'team', i, viewer));

  for (const p of alivePlayers(d, i)) {
    if (p === viewer || p === opp) continue;
    const friendly = teamOf(pPhys(d, 'team', i, p)) === myTeam;
    const ob = obstrAt(d, i, viewer, p);
    const blocked = ob !== 0;
    const box = {
      x: pPhys(d, 'x', i, p),
      y: pPhys(d, 'y', i, p),
      z: pPhys(d, 'z', i, p),
      h: pPhys(d, 'box_h', i, p),
    };
    const rgb = friendly ? '94, 200, 255' : '255, 155, 52';
    drawBox(ctx, box, {
      stroke: `rgba(${rgb}, ${blocked ? 0.26 : 0.55})`,
      width: friendly ? 1 : 1.3,
      dash: blocked ? [2, 4] : [4, 3],
    }, eye, yaw0, pitch0, W, H, fovH, fovV);

    const top = projectPoint(box.x, box.y, box.z + box.h + 6,
      eye, yaw0, pitch0, W, H, fovH, fovV);
    if (!top) continue;
    if (top.x < 4 || top.x > W - 4 || top.y < 12 || top.y > H - 4) continue;
    if (Math.abs(top.dyaw) > 100 || Math.abs(top.dp) > 80) continue;
    const tag = `#${p}${friendly ? ' 友' : ''}${blocked ? ` ${obstrName(d, ob)}` : ''}`;
    ctx.save();
    ctx.font = '9px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = blocked ? 'rgba(190, 205, 220, 0.6)'
      : friendly ? 'rgba(150, 225, 255, 0.85)' : 'rgba(255, 190, 110, 0.85)';
    ctx.fillText(tag, top.x, top.y);
    ctx.restore();
  }
}

function drawBoxLabel(ctx, box, m, eye, yaw0, pitch0, W, H, fovH, fovV) {
  const p = projectPoint(
    box.x, box.y, box.z + box.h + 10,
    eye, yaw0, pitch0, W, H, fovH, fovV
  );
  if (!p) return;
  if (p.x < -60 || p.x > W + 60 || p.y < -20 || p.y > H + 20) return;
  const text = `target / enemy · ${m.dist.toFixed(0)} HU (${(m.dist * S.data.geometry.hu_to_m).toFixed(1)} m)`;
  label(ctx, p.x, p.y, text, 'rgba(255, 190, 190, 0.95)', 'rgba(255, 70, 70, 0.25)');
}

function label(ctx, x, y, text, color, bg) {
  ctx.save();
  ctx.font = '10.5px Consolas, monospace';
  ctx.textAlign = 'center';
  const w = ctx.measureText(text).width;
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const cx = clamp(x, w / 2 + 6, W - w / 2 - 6);
  const cy = clamp(y, 12, H - 8);
  const bx = cx - w / 2 - 4;
  ctx.fillStyle = bg || 'rgba(6, 10, 14, 0.7)';
  ctx.fillRect(bx, cy - 10, w + 8, 14);
  ctx.fillStyle = color;
  ctx.fillText(text, cx, cy);
  ctx.restore();
}

function drawTargetOffscreen(ctx, W, H, m) {
  const g = S.data.geometry;
  const fovH = g.fov_h;
  const fovV = g.fov_v;
  const inView = Math.abs(m.dYaw) < fovH / 2 && Math.abs(m.dPitch) < fovV / 2;
  if (inView) return;
  // 箭头贴在画布边缘、指向目标所在的**方向**：x 和旋转角都必须跟着 yawToX 一起取反。
  // 这里用的是半幅（W/2）——和原来一样，45° 偏角落在 1/4 处，不会一偏就贴边。
  const x = clamp(yawToX(clamp(m.dYaw, -fovH, fovH) / 2, W, fovH), 26, W - 26);
  const y = clamp(H / 2 + (m.dPitch / fovV) * (H / 2), 26, H - 26);
  const ang = Math.atan2(m.dPitch, -m.dYaw);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(ang);
  ctx.fillStyle = 'rgba(255, 77, 77, 0.9)';
  ctx.beginPath();
  ctx.moveTo(10, 0);
  ctx.lineTo(-7, -7);
  ctx.lineTo(-3, 0);
  ctx.lineTo(-7, 7);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.fillStyle = 'rgba(255, 170, 170, 0.85)';
  ctx.font = '10px Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('目标在视野外', clamp(x, 60, W - 60), clamp(y + 20, 14, H - 8));
  ctx.restore();
}

function drawCrosshair(ctx, W, H, shot, blinded) {
  const cx = W / 2;
  const cy = H / 2;
  const gap = 5;
  const len = 10;
  const color = shot ? 'rgba(255, 70, 70, 1)' : blinded ? 'rgba(255, 220, 120, 0.95)' : 'rgba(235, 244, 255, 0.92)';
  ctx.save();
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.lineWidth = 3.4;
  crossLines(ctx, cx, cy, gap, len);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.3;
  crossLines(ctx, cx, cy, gap, len);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, 1.6, 0, Math.PI * 2);
  ctx.fill();
  if (shot) {
    ctx.strokeStyle = 'rgba(255, 70, 70, 0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 17, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function crossLines(ctx, cx, cy, gap, len) {
  ctx.beginPath();
  ctx.moveTo(cx - gap - len, cy); ctx.lineTo(cx - gap, cy);
  ctx.moveTo(cx + gap, cy); ctx.lineTo(cx + gap + len, cy);
  ctx.moveTo(cx, cy - gap - len); ctx.lineTo(cx, cy - gap);
  ctx.moveTo(cx, cy + gap); ctx.lineTo(cx, cy + gap + len);
  ctx.stroke();
}

function drawCornerTag(ctx, kind, me, i) {
  const text = kind === 'atk' ? 'A POV' : 'B POV';
  const color = kind === 'atk' ? '#ffb454' : '#5ec8ff';
  ctx.save();
  ctx.font = 'bold 11px Consolas, monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = color;
  ctx.fillText(text, 12, 20);
  ctx.font = '10px Consolas, monospace';
  ctx.fillStyle = 'rgba(190, 210, 228, 0.75)';
  ctx.fillText(
    `${S.data.map_name}  ·  F${i}  ·  yaw ${me.yaw[i].toFixed(1)}° pitch ${me.pitch[i].toFixed(1)}°`,
    12, 34
  );
  ctx.restore();
}

function convexHull(points) {
  const pts = points.slice().sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  if (pts.length <= 2) return pts;
  const cross = (o, a, b) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let k = pts.length - 1; k >= 0; k--) {
    const p = pts[k];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

// ── 画布：逐帧俯视雷达 ───────────────────────────────────────────────
//
// 画的是**当前帧**所有 mask != 0 的玩家：
//   颜色 = 队伍（CT 蓝 / T 橙）；A / B 用粗描边 + 字母标签，其他人细描边 + 下标。
// 轨迹尾巴用 players 的 x/y 全序列按帧裁剪（A/B 另外用 d.atk / d.tgt 的 phys，
// 和 HUD 是同一份数据）；死人只留一条很淡的尾巴、不再画点。

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
  // 没有雷达图：用全场逐帧的玩家位置估一个包围盒，画坐标网格
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const add = (x, y) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  };
  const n = d.n_players || 10;
  for (let i = 0; i < d.frames; i += 4) {
    for (let p = 0; p < n; p++) {
      if (pMask(d, i, p)) add(pPhys(d, 'x', i, p), pPhys(d, 'y', i, p));
    }
  }
  // 炸弹也是逐帧的（会被搬着走）：整段扫一遍，别只取首帧那一个点
  for (let i = 0; i < (d.global.x || []).length; i += 4) add(d.global.x[i], d.global.y[i]);
  if (!Number.isFinite(minX)) { minX = -2500; maxX = 2500; minY = -2500; maxY = 2500; }
  const pad = Math.max(maxX - minX, maxY - minY, 600) * 0.12;
  minX -= pad; maxX += pad; minY -= pad; maxY += pad;
  const span = Math.max(maxX - minX, maxY - minY);
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

function drawRadar() {
  const d = S.data;
  if (!d) return;
  const i = S.frame;
  const canvas = $('radar');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
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

  // ① 轨迹尾巴：A / B 画整段（过去亮、未来暗），其他人只画过去的一小段。
  const T = d.frames;
  const step = Math.max(1, Math.floor(T / 180));
  drawRadarTrail(ctx, t, (k) => d.atk.phys.x[k], (k) => d.atk.phys.y[k],
    i, '#ffb454', 0.72, 0.20, true);
  drawRadarTrail(ctx, t, (k) => d.tgt.phys.x[k], (k) => d.tgt.phys.y[k],
    i, '#5ec8ff', 0.72, 0.20, true);
  const n = d.n_players || 10;
  for (let p = 0; p < n; p++) {
    if (p === d.atk.index || p === d.tgt.index) continue;
    drawPlayerTrail(ctx, t, d, p, i, TEAM_COLOR[teamOf(pPhys(d, 'team', i, p))]);
  }

  // ② 当前帧所有活着的人：队伍色圆点 + 下标；A / B 粗描边 + 字母。
  for (let p = 0; p < n; p++) {
    const m = pMask(d, i, p);
    if (!m) continue;
    const [px, py] = t.toPx(pPhys(d, 'x', i, p), pPhys(d, 'y', i, p));
    const team = teamOf(pPhys(d, 'team', i, p));
    const isA = p === d.atk.index || m === 1;
    const isB = p === d.tgt.index || m === 2;
    const r = isA || isB ? 9 : 6;
    ctx.save();
    ctx.fillStyle = TEAM_COLOR[team];
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = isA ? '#ffeccc' : isB ? '#dcf5ff' : 'rgba(255, 255, 255, 0.45)';
    ctx.lineWidth = isA || isB ? 2.4 : 1;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#061019';
    ctx.font = isA || isB ? 'bold 10px Consolas, monospace' : '9px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(isA ? 'A' : isB ? 'B' : String(p), px, py + 0.5);
    ctx.restore();
  }

  // ③ A / B 的文字标签（队名 + 角色），避开雷达边缘
  radarNameTag(ctx, t, d, d.atk.index, i, `A · ${teamName(d.atk.phys.team[i])}`, '#ffdca8');
  radarNameTag(ctx, t, d, d.tgt.index, i, `B · ${teamName(d.tgt.phys.team[i])}`, '#c9eeff');

  // ④ 炸弹（逐帧 global_feat：位置会跟着携带者走，安放后固定在包点）
  const g = d.global;
  const [bx, by] = t.toPx(g.x[i], g.y[i]);
  ctx.save();
  ctx.translate(bx, by);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = '#ffd257';
  ctx.strokeStyle = '#fff3b0';
  ctx.lineWidth = 1;
  ctx.fillRect(-5, -5, 10, 10);
  ctx.strokeRect(-5, -5, 10, 10);
  ctx.restore();
  ctx.save();
  ctx.fillStyle = 'rgba(255, 226, 150, 0.92)';
  ctx.font = '10px Consolas, monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(g.planted[i]
    ? `C4 · 已安放 ${g.planted_seconds[i].toFixed(1)}s`
    : 'C4 · 携带中', bx + 9, by - 6);
  ctx.restore();

  // ⑤ 当前帧的道具（projectiles[t]）
  for (const p of (d.projectiles && d.projectiles[i]) || []) {
    const [px, py] = t.toPx(p.x, p.y);
    ctx.save();
    ctx.strokeStyle = projColor(p.id);
    ctx.fillStyle = 'rgba(5, 9, 14, 0.75)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = projColor(p.id);
    ctx.font = '7.5px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(projGlyph(p.id), px, py + 0.5);
    ctx.restore();
  }

  // ⑥ 角标
  ctx.save();
  ctx.fillStyle = 'rgba(150, 180, 205, 0.75)';
  ctx.font = '10px Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'bottom';
  ctx.fillText(`俯视图 · 第 ${i} / ${T - 1} 帧 · x/y 平面`, W - 10, H - 10);
  ctx.restore();

  if (new URLSearchParams(location.search).get('debug') === '1') {
    const rect = canvas.getBoundingClientRect();
    const a = d.atk.index;
    const b = d.tgt.index;
    document.body.dataset.radarDebug = JSON.stringify({
      frame: i,
      rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
      extent: d.radar ? d.radar.extent : null,
      atk: a == null ? null : t.toPx(d.atk.phys.x[i], d.atk.phys.y[i]),
      tgt: b == null ? null : t.toPx(d.tgt.phys.x[i], d.tgt.phys.y[i]),
      alive: alivePlayers(d, i),
      img: S.radarImg ? [S.radarImg.naturalWidth, S.radarImg.naturalHeight] : null,
    });
  }

  buildRadarLegend();
}

// A / B 的整段轨迹：from..to 用 getX/getY 取值（A/B 用 phys 的数组，别人用 players 通道）。
function drawRadarTrail(ctx, t, getX, getY, i, color, pastAlpha, futureAlpha, withFuture) {
  const T = S.data.frames;
  const step = Math.max(1, Math.floor(T / 180));
  ctx.save();
  ctx.lineWidth = 1.6;
  ctx.lineJoin = 'round';
  if (withFuture) {
    ctx.strokeStyle = hexAlpha(color, futureAlpha);
    traceSegments(ctx, t, i, T - 1, step, getX, getY);
  }
  ctx.strokeStyle = hexAlpha(color, pastAlpha);
  traceSegments(ctx, t, 0, i, step, getX, getY);
  ctx.restore();
}

// 其他玩家的淡尾巴：只画活着的帧段，死了就断开（死人的坐标是 0，会画到地图角上）。
function drawPlayerTrail(ctx, t, d, p, i, color) {
  const step = Math.max(1, Math.floor(d.frames / 180));
  ctx.save();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = hexAlpha(color, 0.30);
  ctx.beginPath();
  let pen = false;
  for (let k = 0; k <= i; k += step) {
    if (pMask(d, k, p) === 0) { pen = false; continue; }
    const [x, y] = t.toPx(pPhys(d, 'x', k, p), pPhys(d, 'y', k, p));
    if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
  }
  if (pen && pMask(d, i, p)) {
    const [x, y] = t.toPx(pPhys(d, 'x', i, p), pPhys(d, 'y', i, p));
    ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function traceSegments(ctx, t, from, to, step, getX, getY) {
  if (to < from) return;
  ctx.beginPath();
  let pen = false;
  for (let k = from; k <= to; k += step) {
    const [x, y] = t.toPx(getX(k), getY(k));
    if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
  }
  const [x, y] = t.toPx(getX(to), getY(to));
  ctx.lineTo(x, y);
  ctx.stroke();
}

function radarNameTag(ctx, t, d, p, i, text, color) {
  if (p == null || pMask(d, i, p) === 0) return;
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const [px, py] = t.toPx(pPhys(d, 'x', i, p), pPhys(d, 'y', i, p));
  ctx.save();
  ctx.fillStyle = 'rgba(5, 9, 14, 0.72)';
  ctx.font = '10px Consolas, monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  const w = ctx.measureText(text).width;
  const tx = clamp(px + 13, 4, W - w - 8);
  const ty = clamp(py - 14, 10, H - 10);
  ctx.fillRect(tx - 3, ty - 7, w + 6, 14);
  ctx.fillStyle = color;
  ctx.fillText(text, tx, ty);
  ctx.restore();
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
    ctx.fillText(String(Math.round(x)), px + 3, H - 22);
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
  ctx.textAlign = 'left';
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
  return ['#9aa9b8', '#ff7a3d', '#fff05e', '#c693ff', '#8fe36a', '#ff9d3d', '#7fd2ff'][id] || '#cccccc';
}

function projGlyph(id) {
  return ['烟', '火', '闪', '诱', '雷', '烧', '烟'][id] || '?';
}

let legendBuilt = false;
function buildRadarLegend() {
  if (legendBuilt) return;
  legendBuilt = true;
  const el = $('radar-legend');
  el.innerHTML = '';
  const add = (html) => {
    const span = document.createElement('span');
    span.className = 'key';
    span.innerHTML = html;
    el.appendChild(span);
  };
  add(`<span class="dot" style="background:${TEAM_COLOR[0]}"></span>CT（当前帧）`);
  add(`<span class="dot" style="background:${TEAM_COLOR[1]}"></span>T`);
  add('<span class="ring" style="border-color:#ffeccc"></span>A · 击杀者（粗描边 + 轨迹）');
  add('<span class="ring" style="border-color:#dcf5ff"></span>B · 受害者（粗描边 + 轨迹）');
  add('<span class="dot dim"></span>其他存活玩家（细描边 + 下标）');
  add('<span class="diamond"></span>C4（逐帧 global_feat）');
  add('<span class="dot" style="background:#9aa9b8"></span>道具（当前帧 projectiles）');
}

// ── 遮挡矩阵 + A→B 时间轴 ─────────────────────────────────────────────
//
// obstruction[t][i][j]：从 i 的眼睛看 j 的身体，0 通 / 1 烟 / 2 墙（有方向）。
// 面板左边是当前帧的 10×10 网格（DOM 只建一次，每帧改 class / 文案）；
// 右边色带是 obstruction[t][A][B] 的整段序列，点一下就能跳到那一帧。
//
// 死人的行 / 列在数据里恒为 0（"通"），但那不是真的看得见 —— 面板里按 mask
// 画成暗格，并且不参与统计，免得整张表被"死人通视"刷成一片绿。

const OBS_COLOR = { 0: '#3fb979', 1: '#3e8fd8', 2: '#7d8894' };

function buildObstruction() {
  const d = S.data;
  const table = $('obstr-table');
  table.innerHTML = '';
  S.obstrCells = [];
  S.obstrVals = null;

  const n = d.n_players || 10;
  const a = d.atk.index;
  const b = d.tgt.index;
  const hTitle = (p) => (p === a ? `玩家 ${p}：A（击杀者）`
    : p === b ? `玩家 ${p}：B（受害者）` : `玩家 ${p}`);

  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  const corner = document.createElement('th');
  corner.className = 'corner';
  corner.textContent = 'i \\ j';
  corner.title = '行 = 观察者 i，列 = 被看的 j';
  htr.appendChild(corner);
  for (let j = 0; j < n; j++) {
    const th = document.createElement('th');
    th.textContent = String(j);
    th.className = j === a ? 'h-a' : j === b ? 'h-b' : '';
    th.title = hTitle(j);
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (let i = 0; i < n; i++) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = String(i);
    th.className = i === a ? 'h-a' : i === b ? 'h-b' : '';
    th.title = hTitle(i);
    tr.appendChild(th);
    for (let j = 0; j < n; j++) {
      const td = document.createElement('td');
      td.className = 'obstr-cell';
      tr.appendChild(td);
      S.obstrCells.push({ td, i, j });
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  drawObstrTimeline();
}

function updateObstruction() {
  const d = S.data;
  if (!d || !S.obstrCells.length) return;
  const i = S.frame;
  const n = d.n_players || 10;
  const a = d.atk.index;
  const b = d.tgt.index;

  for (const cell of S.obstrCells) {
    const td = cell.td;
    const pi = cell.i;
    const pj = cell.j;
    const selfDead = pMask(d, i, pi) === 0;
    const otherDead = pMask(d, i, pj) === 0;
    let cls = 'obstr-cell';
    let text;
    if (pi === pj) {
      cls += ' diag';
      text = '·';
    } else if (selfDead || otherDead) {
      cls += ' dead-cell';
      text = '·';
    } else {
      const v = obstrAt(d, i, pi, pj);
      const vs = viewState(d, i, pi, pj);
      cls += ` ${obstrClass(v)}`;
      // 没被挡住、但观察者压根没往那边看：加一层斜线，别让"通"被读成"看见了"。
      if (v === 0 && !vs.inFov) cls += ' out-fov';
      text = obstrName(d, v);
    }
    if (pi === a || pj === a) cls += ' ax';
    if (pi === b || pj === b) cls += ' bx';
    if (a != null && b != null
        && ((pi === a && pj === b) || (pi === b && pj === a))) cls += ' cross';
    if (td.className !== cls) td.className = cls;
    if (td.textContent !== text) td.textContent = text;
    if (pi === pj) {
      td.title = `${pi} 看自己：不采样`;
    } else if (selfDead || otherDead) {
      td.title = `${pi} 看 ${pj}：有一方已死亡（数据恒为 0，不算"通"）`;
    } else {
      const v = obstrAt(d, i, pi, pj);
      const vs = viewState(d, i, pi, pj);
      td.title = `第 ${i} 帧 · ${pi} 看 ${pj}：${obstrName(d, v)}（${v}）`
        + (v === 0 ? ` · ${pi} ${viewText(vs)}` : '');
    }
  }

  const readout = $('obstr-readout');
  const timeReadout = $('obstr-time-readout');
  if (a == null || b == null) {
    readout.textContent = '样本里找不到 A(=1) / B(=2) 的唯一行，遮挡读数不可用';
    timeReadout.textContent = '—';
  } else {
    const ab = obstrAt(d, i, a, b);
    const ba = obstrAt(d, i, b, a);
    // 方向是"A 看 B"，所以视野状态取的是**观察者**的：前半句跟 A 的 yaw，后半句跟 B 的。
    const av = viewState(d, i, a, b);
    const bv = viewState(d, i, b, a);
    readout.textContent =
      `A(${a}) → B(${b})：${obstrName(d, ab)} · A ${viewText(av)}`
      + ` ｜ B(${b}) → A(${a})：${obstrName(d, ba)} · B ${viewText(bv)}`;
    timeReadout.textContent = `F${i} · ${d.seconds[i].toFixed(3)} s · ${obstrName(d, ab)}(${ab})`;
  }

  // 当前帧的可见对统计：跳过死人和对角线，别让"死人恒 0"混进"通"里。
  let c0 = 0;
  let c1 = 0;
  let c2 = 0;
  for (let pi = 0; pi < n; pi++) {
    if (pMask(d, i, pi) === 0) continue;
    for (let pj = 0; pj < n; pj++) {
      if (pi === pj || pMask(d, i, pj) === 0) continue;
      const v = obstrAt(d, i, pi, pj);
      if (v === 1) c1 += 1;
      else if (v === 2) c2 += 1;
      else c0 += 1;
    }
  }
  const alive = alivePlayers(d, i).length;
  $('obstr-stats').textContent =
    `存活 ${alive} / ${n} · 有人看的有向对 ${c0 + c1 + c2} 组：` +
    `通 ${c0} / 烟 ${c1} / 墙 ${c2}`;

  drawObstrTimeline();
}

// A→B 的整段色带 + 当前帧游标。色带按帧铺满画布宽度（T 列），游标用帧号定位。
function drawObstrTimeline() {
  const d = S.data;
  const cv = $('obstr-timeline');
  if (!cv || !d) return;
  const ctx = cv.getContext('2d');
  const W = cv.width;
  const H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#05080c';
  ctx.fillRect(0, 0, W, H);

  const T = d.frames;
  const a = d.atk.index;
  const b = d.tgt.index;
  const top = 12;
  const bandH = 22;

  // 每帧一格：第 k 帧占 [k/T, (k+1)/T)
  for (let k = 0; k < T; k++) {
    const v = (a == null || b == null) ? 0 : obstrAt(d, k, a, b);
    const x0 = (k / T) * W;
    const x1 = ((k + 1) / T) * W;
    ctx.fillStyle = OBS_COLOR[v] || OBS_COLOR[0];
    ctx.fillRect(x0, top, Math.max(x1 - x0, 1), bandH);
  }

  // 整秒刻度：用 seconds 找出"秒数变化"的帧，画一条暗竖线（只在色带上画，不压字）
  ctx.strokeStyle = 'rgba(3, 8, 12, 0.55)';
  ctx.lineWidth = 1;
  let lastSec = Math.floor(d.seconds[0] || 0);
  for (let k = 1; k < T; k++) {
    const sec = Math.floor(d.seconds[k]);
    if (sec !== lastSec) {
      lastSec = sec;
      const x = (k / T) * W;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, top + bandH);
      ctx.stroke();
    }
  }

  // 边框
  ctx.strokeStyle = 'rgba(120, 150, 175, 0.35)';
  ctx.strokeRect(0.5, top + 0.5, W - 1, bandH - 1);

  // 当前帧游标（帧号映射到 0..W-1，和上面色带的右边界对齐）
  const i = S.frame;
  const cx = T <= 1 ? 0 : (i / (T - 1)) * (W - 1);
  ctx.strokeStyle = '#f4fbff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, 2);
  ctx.lineTo(cx, H - 2);
  ctx.stroke();
  ctx.fillStyle = '#f4fbff';
  ctx.beginPath();
  ctx.moveTo(cx - 4, 2);
  ctx.lineTo(cx + 4, 2);
  ctx.lineTo(cx, 9);
  ctx.closePath();
  ctx.fill();
}

// ── 模型预测 ──────────────────────────────────────────────────────────
//
// 服务端 `/api/predict/<split>/<name>` 跑一次模型，返回 **T** 个预测点
// （`predict.py`）。序列是 `[A_t] [B_t] [PRED_t]`，因果注意力下第 t 列看完了
// 第 0..t 帧（同帧的 A、B 就在它前面），所以：
//
//   列 i  刚看完第 i 帧 → 拖到第 i 帧时"当前"的预测就是它（x 轴就是帧号）
//
// 没有"一帧都没看到"那一列：那时模型不知道 A、B 是谁（身份只在逐帧 token 里），
// 旧版那一列实测 acc 0.58 ≈ 瞎猜，已经删掉了。
//
// 两个方向一次算完：`probs` = P(ATK 击杀 TGT)，`probs_rev` = P(TGT 击杀 ATK)。
// 这对人**恰好**有一个击杀了另一个，两条曲线应当关于 0.5 镜像 —— 偏得越多越不自洽。

const PRED_MAIN = '#ffb454';   // = --atk
const PRED_REV = '#5ec8ff';    // = --tgt

let predInfoPromise = null;

function loadModelInfo() {
  if (!predInfoPromise) {
    predInfoPromise = apiJson('/api/model')
      .then((info) => { S.predModel = info; setPredModelLabel(); return info; })
      .catch((err) => {
        S.predModel = { ok: false, reason: `拿不到模型信息：${err.message}` };
        setPredModelLabel();
        return S.predModel;
      });
  }
  return predInfoPromise;
}

function setPredModelLabel() {
  const info = S.predModel;
  const el = $('pred-model');
  if (!info) { el.textContent = '模型信息加载中 …'; return; }
  if (!info.ok) { el.textContent = '不可用'; el.title = info.reason || ''; return; }
  const bits = [info.name, info.weights, info.device];
  if (info.epoch != null) bits.push(`epoch ${info.epoch}`);
  bits.push(`${(info.params / 1e6).toFixed(2)} M`);
  el.textContent = bits.join(' · ');
  const best = info.best || {};
  const metric = Object.entries(best)
    .filter(([, v]) => typeof v === 'number')
    .map(([k, v]) => `${k}=${v.toFixed(4)}`)
    .join(' ');
  el.title = [info.dir, metric].filter(Boolean).join('\n');
}

function setPredNote(text) {
  $('pred-note').textContent = text || '';
}

// 请求新样本的预测。`S.predToken` 是防抢跑用的：快速连点"下一个"时，先到的旧
// 响应不能盖掉后发的那个（和 `selectSample` 的 `loadToken` 一个道理）。
async function loadPrediction() {
  const d = S.data;
  if (!d) return;
  const token = ++S.predToken;
  S.pred = null;
  $('pred-panel').classList.remove('pred-off');
  if (!S.predOn) {
    setPredNote('预测已关闭（URL 里 pred=0）');
    renderPred();
    return;
  }
  const info = await loadModelInfo();
  if (token !== S.predToken) return;
  if (!info.ok) return showPredUnavailable(info.reason);
  setPredNote('推理中 …');
  renderPred();
  try {
    const out = await apiJson(apiPath('api', 'predict', d.split, d.name));
    if (token !== S.predToken) return;
    S.pred = out;
    setPredNote('');
    renderPred();
  } catch (err) {
    if (token !== S.predToken) return;
    showPredUnavailable(err.message);
  }
}

function showPredUnavailable(reason) {
  S.pred = null;
  $('pred-panel').classList.add('pred-off');
  setPredNote(`没有预测：${reason}`);
  renderPred();
}

function renderPred() {
  drawPredCurve();
  updatePredStats();
}

function fmtProb(v) {
  return v == null ? '—' : `${(v * 100).toFixed(1)}%`;
}

function updatePredStats() {
  const p = S.pred;
  if (!p) {
    for (const id of ['pred-first', 'pred-cur', 'pred-last', 'pred-cross']) {
      $(id).textContent = '—';
      $(id).className = '';
    }
    for (const id of ['pred-cur-s', 'pred-last-s', 'pred-cross-s']) {
      $(id).textContent = '—';
    }
    $('pred-rows').innerHTML = '';
    return;
  }
  const T = p.frames;
  const i = S.frame;
  const col = Math.min(i, T - 1);            // 第 i 列 = 刚看完第 i 帧
  const cur = p.probs[col];
  const mark = (id, v) => {
    const el = $(id);
    el.textContent = fmtProb(v);
    el.className = v >= 0.5 ? 'hi' : 'lo';
  };
  mark('pred-first', p.first);
  mark('pred-cur', cur);
  mark('pred-last', p.last);
  $('pred-cur-s').textContent = `已看第 0..${i} 帧`;
  $('pred-last-s').textContent = `第 ${T - 1} 列 · 全部 ${T} 帧`;

  const k = p.probs.findIndex((v) => v >= 0.5);
  const cross = $('pred-cross');
  const crossS = $('pred-cross-s');
  cross.className = '';
  if (k < 0) {
    cross.textContent = '没有';
    crossS.textContent = '整段都 < 50%';
  } else {
    const t = p.time[k];
    cross.textContent = `F${k}`;
    crossS.textContent = `第 ${k} 列${t == null ? '' : ` · ${t.toFixed(2)} s`}`;
  }

  const gap = Math.abs(p.last + p.last_rev - 1);
  const rows = [
    ['反向 P(TGT→ATK)', `${fmtProb(p.first_rev)} → ${fmtProb(p.last_rev)}`],
    ['自洽 |p+p′−1|', gap.toFixed(3), gap > 0.5 ? 'warn' : gap > 0.25 ? 'mid' : 'ok'],
    ['真值', 'ATK 击杀了 TGT（磁盘上样本都是正样本）'],
    ['推理', `${p.ms.toFixed(1)} ms · ${p.device}`],
    ['权重', p.weights || '—'],
  ];
  const box = $('pred-rows');
  box.innerHTML = '';
  for (const [k2, v, cls] of rows) {
    const row = document.createElement('div');
    row.className = 'pred-row';
    const a = document.createElement('span');
    a.textContent = k2;
    const b = document.createElement('b');
    b.textContent = v;
    if (cls) b.className = cls;
    row.append(a, b);
    box.appendChild(row);
  }
}

function drawPredCurve() {
  const cv = $('pred-canvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width;
  const H = cv.height;
  ctx.clearRect(0, 0, W, H);

  const padL = 54, padR = 18, padT = 18, padB = 34;
  const pw = W - padL - padR;
  const ph = H - padT - padB;
  const p = S.pred;
  const d = S.data;

  // x 轴刻度先算出来：画网格和曲线都要用。x 就是帧号，列 0..T-1 铺满整宽。
  const T = p ? p.frames : 1;
  const last = Math.max(0, T - 1);
  const xAt = (col) => padL + pw * (T <= 1 ? 0.5 : col / last);
  const yAt = (v) => padT + ph * (1 - clamp(v, 0, 1));

  // ── 网格 + y 轴（0/25/50/75/100%）──
  ctx.font = '11px Consolas, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    const y = yAt(v);
    ctx.strokeStyle = v === 0.5 ? 'rgba(220, 231, 242, 0.28)' : '#131d27';
    ctx.lineWidth = 1;
    ctx.setLineDash(v === 0.5 ? [6, 5] : []);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = v === 0.5 ? '#9fb4c8' : '#5d6e80';
    ctx.fillText(`${Math.round(v * 100)}%`, padL - 8, y);
  }

  if (!p) {
    ctx.textAlign = 'center';
    ctx.fillStyle = '#5d6e80';
    ctx.font = '13px "Segoe UI", sans-serif';
    const why = S.predModel && !S.predModel.ok
      ? '模型不可用（见上方说明）'
      : S.predOn ? '推理中 …' : '预测已关闭';
    ctx.fillText(why, padL + pw / 2, padT + ph / 2);
    return;
  }

  // ── x 轴：帧号 0..T-1 ──
  const ticks = [...new Set([0, Math.round(last * 0.25), Math.round(last * 0.5),
    Math.round(last * 0.75), last])].sort((a, b) => a - b);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const col of ticks) {
    const x = xAt(col);
    ctx.strokeStyle = '#131d27';
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + ph);
    ctx.stroke();
    ctx.fillStyle = '#5d6e80';
    ctx.fillText(String(col), x, padT + ph + 7);
  }
  ctx.fillStyle = '#5d6e80';
  ctx.textAlign = 'right';
  ctx.fillText('帧 →', W - padR, H - 14);

  // ── 开枪帧：竖细线（和进度条上的红刻度同一批帧）──
  if (d && d.atk && d.atk.phys && d.atk.phys.shot) {
    for (let k = 0; k < d.atk.phys.shot.length && k < T; k++) {
      if (!(d.atk.phys.shot[k] > 0.5)) continue;
      ctx.strokeStyle = 'rgba(255, 90, 90, 0.28)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xAt(k), padT);
      ctx.lineTo(xAt(k), padT + ph);
      ctx.stroke();
    }
  }

  const line = (vals, color, width, dash) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash || []);
    ctx.lineJoin = 'round';
    ctx.beginPath();
    vals.forEach((v, k) => {
      const x = xAt(k);
      const y = yAt(v);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  };

  // 主曲线：先铺一层渐变底（画在两条线**下面**，不然反向虚线会被盖灰）
  const grad = ctx.createLinearGradient(0, padT, 0, padT + ph);
  grad.addColorStop(0, hexAlpha(PRED_MAIN, 0.24));
  grad.addColorStop(1, hexAlpha(PRED_MAIN, 0.02));
  ctx.beginPath();
  p.probs.forEach((v, k) => {
    const x = xAt(k);
    const y = yAt(v);
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.lineTo(xAt(last), padT + ph);
  ctx.lineTo(xAt(0), padT + ph);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // 反向曲线（虚线、淡）：和主曲线关于 0.5 镜像才对
  line(p.probs_rev, hexAlpha(PRED_REV, 0.8), 1.5, [5, 4]);
  line(p.probs, PRED_MAIN, 2, []);

  // ── 当前帧：竖线 + 两条曲线上的点 ──
  const i = S.frame;
  const col = Math.min(i, last);
  const x = xAt(col);
  ctx.strokeStyle = 'rgba(220, 231, 242, 0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, padT);
  ctx.lineTo(x, padT + ph);
  ctx.stroke();

  const dot = (v, color, r) => {
    ctx.beginPath();
    ctx.arc(x, yAt(v), r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#04070a';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };
  dot(p.probs_rev[col], PRED_REV, 3);
  dot(p.probs[col], PRED_MAIN, 4);
  // 标在点的上方；贴左右边缘时避开图例 / 轴标签（label() 会居中画）。
  const labelX = x < padL + 70 ? x + 52 : x > padL + pw - 70 ? x - 52 : x;
  label(ctx, labelX, yAt(p.probs[col]) - 16,
    `F${i} · ${fmtProb(p.probs[col])}`, '#ffd9a8', 'rgba(48, 32, 8, 0.88)');

  // ── 图例（左上角，垫一层底避免压住曲线看不清）──
  ctx.fillStyle = 'rgba(4, 7, 10, 0.8)';
  ctx.fillRect(padL + 2, padT + 1, 196, 38);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  const legend = (y, color, dash, text) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(padL + 6, y);
    ctx.lineTo(padL + 34, y);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = '#9fb4c8';
    ctx.font = '11.5px Consolas, monospace';
    ctx.fillText(text, padL + 40, y);
  };
  legend(padT + 10, PRED_MAIN, [], 'P(ATK 击杀 TGT)');
  legend(padT + 27, hexAlpha(PRED_REV, 0.9), [5, 4], 'P(TGT 击杀 ATK)');
}

// ── 启动 ──────────────────────────────────────────────────────────────

boot();
