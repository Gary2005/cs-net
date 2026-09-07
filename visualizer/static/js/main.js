/**
 * main.js — CS2 Vision Studio 入口
 *
 * 面向客户的 3D 回放 + AI 路径预测工具。
 * 复用引擎：scene.js / map-loader.js / replay-core.js / visuals.js
 * 预测渲染：prediction.js
 */

import * as THREE from 'three';
import {
  initScene, renderFrame, resetCamera, scene, camera, renderer, controls,
  renderMinimap, setCameraMode, setFocusedPlayer, cameraMode, raycaster, mapGroup,
  gameToThree, flyCameraTo,
} from './scene.js';
import { loadMap, getMapBounds } from './map-loader.js';
import {
  matchData, loadMatchData, loadRound,
  currentRound, currentSampleIdx, currentTime, isPlaying, playSpeed,
  getAllPlayerStates, getPlayerState, getBombPosition, seekTo,
  getRoundDuration, setPlaying, setPlaySpeed, playerName,
  playerTeamInRound, totalRounds, getSampleCount, isBombPlanted,
  advanceReplay, weaponName, placeName,
  buildGrenadeTrajectories as getGrenadeTrajectories,
} from './replay-core.js';
import {
  createPlayers, createAimRays, updatePlayers, clearAllEntities,
  setPlayerNames, updateBomb, updateSmokes, updateInfernos,
  updateSmokeAnimations, updateInfernoAnimations,
  updateGrenadeTrajectories, buildGrenadeTrajectories, spawnMuzzleFlash, updateMuzzleFlashes,
  spawnFootstep, updateFootsteps, showKillEffect, setTrailsVisible,
  setNamesVisible, setPlayerModelVisible, updateAimRays,
} from './visuals.js';
import {
  renderPrediction, clearPrediction, setShowPrediction, updatePredictionAnimation,
  renderPlayerSamples, clearPlayerSamples,
  renderSpatialCurve, clearMetricChart, hideChartTooltip,
  renderPlayerMetricCurve, redrawPlayerCurve, METRIC_LABELS,
} from './prediction.js?v=20260903j';
import { startLoadingAnimation, stopLoadingAnimation } from './loading.js';

// ═══════════════════════════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════════════════════════

const $ = (id) => document.getElementById(id);

function showToast(msg, isError = false) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden', 'error');
  if (isError) t.classList.add('error');
  setTimeout(() => t.classList.add('hidden'), 4000);
}

function formatTime(sec) {
  if (sec == null || !isFinite(sec)) return '00:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// ═══════════════════════════════════════════════════════════════════════
// 粒子背景
// ═══════════════════════════════════════════════════════════════════════

const particles = [];
function initParticles() {
  const canvas = $('bg-particles');
  const ctx = canvas.getContext('2d');
  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  let W, H;

  function resize() {
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * DPR; canvas.height = H * DPR;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  const N = Math.min(90, Math.floor(window.innerWidth / 16));
  for (let i = 0; i < N; i++) {
    particles.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.6 + 0.4,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      hue: Math.random() < 0.6 ? 200 : 265,   // 青 / 紫
      alpha: Math.random() * 0.35 + 0.10,
      tw: Math.random() * Math.PI * 2,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const p of particles) {
      p.x += p.vx; p.y += p.vy; p.tw += 0.02;
      if (p.x < -10) p.x = W + 10; if (p.x > W + 10) p.x = -10;
      if (p.y < -10) p.y = H + 10; if (p.y > H + 10) p.y = -10;
      const a = p.alpha * (0.7 + 0.3 * Math.sin(p.tw));
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 85%, 60%, ${a})`;
      ctx.shadowColor = `hsla(${p.hue}, 85%, 55%, 0.5)`;
      ctx.shadowBlur = 5;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  draw();
}

// ═══════════════════════════════════════════════════════════════════════
// 上传 / 加载
// ═══════════════════════════════════════════════════════════════════════

let loadedSource = '';

async function loadFile(file) {
  const fd = new FormData();
  fd.append('file', file);

  showLoading(true, `正在处理 ${file.name}…`);
  try {
    const res = await fetch('/api/load', { method: 'POST', body: fd });
    const json = await res.json();
    if (!res.ok || json.error) {
      throw new Error(json.error || `HTTP ${res.status}`);
    }
    onDataReady(json);
  } catch (err) {
    showLoading(false);
    showToast('加载失败: ' + err.message, true);
  }
}

async function loadExample(path) {
  showLoading(true, '正在加载示例…');
  try {
    const res = await fetch('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    onDataReady(json);
  } catch (err) {
    showLoading(false);
    showToast('加载示例失败: ' + err.message, true);
  }
}

function onDataReady(json) {
  const data = json.data;
  loadedSource = json.source || '';

  loadMatchData(data);

  // 切换到主界面
  $('landing').classList.add('hidden');
  $('app').classList.remove('hidden');

  // 地图
  $('map-name').textContent = data.map || 'unknown';
  $('source-name').textContent = loadedSource;
  loadMap(data.map).then(() => fitCameraToMap()).catch(() => {
    showToast('地图加载失败（可能是新地图），仍可查看玩家轨迹', true);
  });

  // 玩家名
  if (data.rounds.length > 0 && data.rounds[0].teams) {
    setPlayerNames(data.players.map(p => p.name), data.rounds[0].teams);
  }

  // 回合选择
  const sel = $('round-select');
  sel.disabled = false;
  sel.innerHTML = '';
  data.rounds.forEach((_, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `回合 ${i + 1}`;
    sel.appendChild(opt);
  });

  // 玩家面板 + 回合统计
  updatePlayerPanel();
  updateRoundStats();

  // 预测面板：重置
  clearPrediction();
  clearMetricChart();
  $('prediction-banner').classList.add('hidden');
  $('predict-status').textContent = '';
  $('pred-results').innerHTML = '';
  // 模型已加载且数据就绪 → 启用预测按钮
  $('btn-predict').disabled = !modelLoaded;

  // 扫描面板：重置（新文件 → 服务端缓存键变化，旧结果不再适用）
  lastScanPlayer = -1;
  $('scan-status').textContent = '';
  renderScanList(null);
  document.querySelectorAll('.pd-scan-btn, .pd-scan-all-btn, #btn-scan-all-players').forEach(b => b.disabled = !modelLoaded);

  // tick 滑块范围
  const T = getSampleCount();
  const tickInput = $('pred-tick');
  tickInput.max = Math.max(0, T - 1);
  tickInput.value = Math.min(Math.floor(T / 2), tickInput.max);
  $('pred-tick-val').textContent = tickInput.value;

  // spatial-only：新文件 → 旧预测作废；模型可用时自动预测当前回合
  resetSpatialRound();
  closePlayerCurve();
  updateSpatialChips();
  refreshSpatialStatus();

  // 构建投掷物轨迹（初始回合）
  try {
    buildGrenadeTrajectories(getGrenadeTrajectories());
  } catch (err) {
    console.warn('[Vision] 投掷物轨迹构建失败:', err);
  }

  showLoading(false);
  showToast(`✓ 已加载 ${data.rounds.length} 个回合`);
}

// ═══════════════════════════════════════════════════════════════════════
// 玩家面板 & 回合信息
// ═══════════════════════════════════════════════════════════════════════

/** 当前展开详情的玩家索引（-1 = 无） */
let expandedPlayerIdx = -1;

function updatePlayerPanel() {
  const list = $('player-list');
  if (!matchData) { list.innerHTML = '<div class="placeholder-text">—</div>'; return; }
  const names = matchData.players || [];
  const teams = currentRound ? currentRound.teams : (matchData.rounds[0] || {}).teams || [];
  list.innerHTML = '';
  expandedPlayerIdx = -1;

  // 按队伍分组：CT / T
  const groups = { CT: [], T: [] };
  for (let i = 0; i < 10; i++) {
    const team = teams[i] === 'T' ? 'T' : 'CT';
    groups[team].push(i);
  }

  for (const team of ['CT', 'T']) {
    const grp = document.createElement('div');
    grp.className = `team-group ${team === 'CT' ? 'ct' : 't'}`;
    grp.innerHTML = `
      <div class="team-header">
        <span class="team-name">${team === 'CT' ? 'CT · 反恐精英' : 'T · 恐怖分子'}</span>
        <span class="team-count">${groups[team].length} 人</span>
      </div>
    `;
    for (const i of groups[team]) {
      const card = document.createElement('div');
      card.className = 'player-card';
      card.id = `player-card-${i}`;
      card.dataset.player = i;
      card.innerHTML = `
        <div class="pc-main">
          <span class="pc-idx">${i + 1}</span>
          <span class="pc-name">${(names[i] && names[i].name) || `P${i + 1}`}</span>
          <span class="pc-hp">—</span>
          <div class="pc-hpbar"><div class="pc-hpfill"></div></div>
          <span class="pc-chevron">▾</span>
        </div>
        <div class="pc-spatial" data-player="${i}">
          <span class="pc-chip" data-metric="alive_end" title="回合末存活概率 · 点击查看整回合曲线">—</span>
          <span class="pc-chip" data-metric="future_kill" title="未来击杀概率 · 点击查看整回合曲线">—</span>
        </div>
      `;
      card.addEventListener('click', (e) => {
        // 点击概率 chip 有独立处理（曲线弹窗），不触发卡片展开/聚焦
        if (e.target.closest('.pc-chip')) return;
        // 展开卡片时同步摄像机拉近对准该玩家
        if (togglePlayerDetail(i)) focusCameraOnPlayer(i);
      });
      grp.appendChild(card);

      // 详情卡（默认隐藏，点击展开）
      const detail = document.createElement('div');
      detail.className = 'player-detail hidden';
      detail.id = `player-detail-${i}`;
      detail.innerHTML = `
        <div class="pd-row">
          <span class="pd-label">血量</span><b class="pd-hp">—</b>
        </div>
        <div class="pd-row">
          <span class="pd-label">护甲</span><b class="pd-armor">—</b>
        </div>
        <div class="pd-row">
          <span class="pd-label">当前武器</span><b class="pd-weapon">—</b>
        </div>
        <div class="pd-row">
          <span class="pd-label">装备</span><span class="pd-inv">—</span>
        </div>
        <div class="pd-row">
          <span class="pd-label">状态</span><span class="pd-status">—</span>
        </div>
        <div class="pd-row">
          <span class="pd-label">位置</span><span class="pd-place">—</span>
        </div>
        <div class="pd-row pd-sample-row">
          <span class="pd-label">路径采样</span>
          <span class="pd-sample-ctrl">
            <input type="number" class="pd-sample-num" data-player="${i}" min="2" max="32" value="8"
                   title="采样条数（2-32）">
            <button class="btn-mini pd-sample-btn" data-player="${i}">🎲 采样</button>
          </span>
        </div>
        <div class="pd-row pd-scan-row">
          <span class="pd-label">走位扫描</span>
          <span class="pd-sample-ctrl">
            <button class="btn-mini pd-scan-btn" data-player="${i}"
                    data-tip="扫描本回合该玩家的低概率移动（未来路径 log p 分数最低的时刻，保留 3 条）">🔍 扫描</button>
            <button class="btn-mini pd-scan-all-btn" data-player="${i}"
                    data-tip="扫描该玩家全部回合的低概率移动（每回合保留 3 条，与单回合扫描同口径），完成后按回合汇总显示">📊 全部回合</button>
          </span>
        </div>
      `;
      grp.appendChild(detail);
    }
    list.appendChild(grp);
  }
  // 采样/扫描按钮初始状态（模型未加载时禁用）
  document.querySelectorAll('.pd-sample-btn').forEach(b => b.disabled = !modelLoaded || !matchData);
  document.querySelectorAll('.pd-scan-btn, .pd-scan-all-btn, #btn-scan-all-players').forEach(b => b.disabled = !modelLoaded || !matchData);
}

/** 切换玩家详情卡展开/收起（同一时刻只展开一张）；forceOpen=true 时强制展开不收起。
 *  返回 true = 展开（或强制保持展开），false = 收起/无操作。 */
function togglePlayerDetail(i, forceOpen = false) {
  const detail = $(`player-detail-${i}`);
  if (!detail) return false;
  const willOpen = forceOpen || detail.classList.contains('hidden');
  // 收起所有
  for (let p = 0; p < 10; p++) {
    const d = $(`player-detail-${p}`);
    if (d) d.classList.add('hidden');
    const c = $(`player-card-${p}`);
    if (c) c.classList.remove('expanded');
  }
  // 展开目标
  if (willOpen) {
    detail.classList.remove('hidden');
    const card = $(`player-card-${i}`);
    if (card) card.classList.add('expanded');
    expandedPlayerIdx = i;
    // 关注该玩家：扫描结果与缓存状态跟随 TA
    lastScanPlayer = i;
    refreshScanStatus();
  } else {
    expandedPlayerIdx = -1;
  }
  return willOpen;
}

/**
 * 用 raycaster 拾取 3D 视口中的玩家模型，命中后：
 *  1) 展开左侧对应详情卡（点击画面中的玩家 → 卡片自动打开）
 *  2) 摄像机拉近对准该玩家
 * 玩家模型 group 命名规则：player-{i} / dead-x-{i}（visuals.js）。
 */
function pickPlayerAt(clientX, clientY) {
  if (!currentRound || !matchData) return;
  const rect = $('three-canvas').getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ndc = new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1
  );
  raycaster.setFromCamera(ndc, camera);
  // 重置 far：共享 raycaster 的 far 会被 updateAimRays 每帧设为射线长度，
  // 距离较远的玩家/叉号会超出 far 而无法命中。
  raycaster.far = Infinity;
  raycaster.near = 0.1;

  // 收集所有可见的玩家模型 group + 死亡叉号（死亡后模型隐藏只剩叉号）
  const targets = [];
  scene.traverse((o) => {
    if (!o.visible) return;
    const m = o.name && o.name.match(/^(player|dead-x)-(\d+)$/);
    if (m) targets.push({ obj: o, idx: parseInt(m[2], 10) });
  });
  if (!targets.length) return;

  const hits = raycaster.intersectObjects(targets.map(t => t.obj), true);
  if (!hits.length) return;
  const hitObj = hits[0].object;
  // 从命中的 mesh 向上找所属的 player-{i} / dead-x-{i}
  let node = hitObj;
  while (node) {
    const m = node.name && node.name.match(/^(player|dead-x)-(\d+)$/);
    if (m) {
      const idx = parseInt(m[2], 10);
      togglePlayerDetail(idx, true);          // 1) 卡片自动打开
      focusCameraOnPlayer(idx);               // 2) 摄像机拉近对准
      return;
    }
    node = node.parent;
  }
}

/** 拉近摄像机对准某玩家（orbit 平滑飞向；第一/三人称先切回 orbit） */
function focusCameraOnPlayer(playerIdx) {
  if (cameraMode !== 'orbit') setCameraMode('orbit');
  const st = getPlayerState(playerIdx, currentSampleIdx);
  if (!st) return;
  const wp = gameToThree(st.x, st.y, st.z);
  flyCameraTo(new THREE.Vector3(wp.x, wp.y, wp.z), 8, 5);   // 拉近：水平 8 / 高 5
}

function updateRoundStats() {
  if (!currentRound) return;
  $('stat-round').textContent = `#${currentRound.id ?? currentRoundIdx() + 1}`;
  $('stat-winner').textContent = currentRound.winner || '—';
  $('stat-reason').textContent = currentRound.end_reason || '—';
  $('round-stats').classList.remove('hidden');
}

/** 武器索引 → 短名（去掉 "weapon_" 前缀） */
function shortWeaponName(idx) {
  const n = weaponName(idx);
  return n.startsWith('weapon_') ? n.slice(7) : n;
}

// 每帧更新玩家面板：血量 + 展开的详情卡
function updatePlayerPanelFrame() {
  if (!currentRound) return;
  const idx = currentSampleIdx;
  const states = getAllPlayerStates(idx);
  for (let i = 0; i < 10; i++) {
    const card = $(`player-card-${i}`);
    if (!card) continue;
    const st = states[i];
    const team = currentRound.teams[i];
    card.className = `player-card ${team === 'CT' ? 'ct' : 't'}`;
    if (st) {
      if (!st.alive) {
        card.classList.add('dead');
        card.querySelector('.pc-hp').textContent = '✕';
        card.querySelector('.pc-hpfill').style.width = '0%';
      } else {
        card.classList.remove('dead');
        const hp = Math.round(st.hp);
        const hpEl = card.querySelector('.pc-hp');
        hpEl.textContent = hp;
        hpEl.classList.toggle('low', hp <= 30);
        card.querySelector('.pc-hpfill').style.width = hp + '%';
      }
    }

    // 展开的详情卡实时更新
    const detail = $(`player-detail-${i}`);
    if (detail && !detail.classList.contains('hidden') && st) {
      const hpEl = detail.querySelector('.pd-hp');
      const armorEl = detail.querySelector('.pd-armor');
      const weaponEl = detail.querySelector('.pd-weapon');
      const invEl = detail.querySelector('.pd-inv');
      const statusEl = detail.querySelector('.pd-status');
      const placeEl = detail.querySelector('.pd-place');

      if (hpEl) hpEl.textContent = st.alive ? `${Math.round(st.hp)} / 100` : '—';
      if (armorEl) armorEl.textContent = st.alive ? `${Math.round(st.armor)} / 100` : '—';
      if (weaponEl) weaponEl.textContent = st.alive ? shortWeaponName(st.weapon) : '—';
      if (invEl) {
        invEl.textContent = st.alive && Array.isArray(st.inventory) && st.inventory.length
          ? st.inventory.map(shortWeaponName).join(' · ')
          : '—';
      }
      if (statusEl) {
        const bits = [];
        if (st.helmet) bits.push('头盔');
        if (st.defuser) bits.push('拆弹器');
        if (st.flash > 0) bits.push(`闪光 ${Math.round(st.flash * 100)}%`);
        statusEl.textContent = st.alive ? (bits.length ? bits.join(' · ') : '正常') : '💀 已阵亡';
      }
      if (placeEl) placeEl.textContent = st.place != null ? placeName(st.place) : '—';
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 击杀信息 & 事件
// ═══════════════════════════════════════════════════════════════════════

function addKillFeed(kill) {
  const feed = $('kill-feed');
  const killerName = playerName(kill.a);
  const victimName = playerName(kill.v);
  // kill.w 是武器索引（数字），用 weaponName 转成武器名
  const weapon = weaponName(kill.w);
  const item = document.createElement('div');
  item.className = 'kill-item';
  item.innerHTML = `
    <span class="killer">${killerName}</span>
    <span class="kweap">${weapon}</span>
    <span class="victim">☠ ${victimName}</span>
  `;
  feed.appendChild(item);
  while (feed.children.length > 4) feed.removeChild(feed.firstChild);
  setTimeout(() => { if (item.parentNode) item.remove(); }, 5000);
}

function handleReplayEvents(events) {
  if (!events) return;
  for (const k of (events.kills || [])) {
    addKillFeed(k);
    const idx = currentSampleIdx;
    const victimState = getPlayerState(k.v, idx);
    if (victimState) showKillEffect(gameToThreeV(victimState.x, victimState.y, victimState.z));
  }
  // 烟雾/火焰由 updateAllVisuals 每帧按活跃集合维护（带缓存 key，不会重复重建）。
  // 这里不再直接 updateSmokes/updateInfernos——否则每次新事件都会清掉全部重画 → 闪烁。
}

function gameToThreeV(x, y, z) {
  return new THREE.Vector3(y * 0.0254, z * 0.0254, x * 0.0254);
}

// ═══════════════════════════════════════════════════════════════════════
// 时间线
// ═══════════════════════════════════════════════════════════════════════

function drawTimeline() {
  const canvas = $('timeline-canvas');
  const wrap = canvas.parentElement;
  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  const w = wrap.clientWidth;
  const h = 40;
  canvas.width = w * DPR;
  canvas.height = h * DPR;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(DPR, DPR);

  ctx.clearRect(0, 0, w, h);

  // 背景轨道
  ctx.fillStyle = 'rgba(255,255,255,0.07)';
  ctx.beginPath();
  ctx.roundRect(0, h / 2 - 5, w, 10, 5);
  ctx.fill();

  if (!currentRound) return;
  const dur = getRoundDuration();
  if (dur <= 0) return;
  const x = (t) => (t / dur) * w;

  // 炸弹阶段
  const plantT = currentRound.bomb_planted_time;
  if (plantT) {
    ctx.fillStyle = 'rgba(255, 92, 122, 0.12)';
    ctx.fillRect(x(plantT), h / 2 - 5, w - x(plantT), 10);
    ctx.strokeStyle = 'rgba(255, 92, 122, 0.6)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x(plantT), 0); ctx.lineTo(x(plantT), h); ctx.stroke();
    ctx.setLineDash([]);
  }

  // 击杀标记
  for (const k of (currentRound.events.kills || [])) {
    const secs = currentRound.round_seconds || [];
    const ticks = currentRound.ticks || [];
    let kt = 0;
    for (let i = 0; i < ticks.length; i++) {
      if (ticks[i] >= k.t) { kt = secs[i] || 0; break; }
    }
    ctx.fillStyle = '#ff5c7a';
    ctx.beginPath();
    ctx.moveTo(x(kt), h / 2 - 7);
    ctx.lineTo(x(kt) - 4, h / 2 + 5);
    ctx.lineTo(x(kt) + 4, h / 2 + 5);
    ctx.closePath();
    ctx.fill();
  }

  // 炸弹事件
  for (const b of (currentRound.events.bomb || [])) {
    ctx.fillStyle = '#ff3ea5';
    ctx.beginPath();
    ctx.arc(x(b.s), h / 2, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // 播放头
  const px = x(currentTime);
  ctx.fillStyle = '#00e5ff';
  ctx.shadowColor = 'rgba(0, 229, 255, 0.8)';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(px, 2); ctx.lineTo(px - 6, 12); ctx.lineTo(px + 6, 12);
  ctx.closePath(); ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillRect(px - 1, 8, 2, h - 8);

  // 进度条
  $('timeline-progress').style.width = (currentTime / dur * 100) + '%';
}

// ═══════════════════════════════════════════════════════════════════════
// 预测
// ═══════════════════════════════════════════════════════════════════════

let modelLoaded = false;
let lastPredResult = null;
let predTickManual = false;   // 用户是否手动拖动过预测 tick（手动后停止自动跟随）
let lastSmokeKey = '';        // 当前活跃烟雾集合指纹（避免每帧重建）
let lastInfernoKey = '';      // 当前活跃火焰集合指纹

// ── spatial-only 自动预测状态 ──────────────────────────────────────────
let spatialAvailable = false;            // 服务端是否已加载 spatial 模型
let spatialRoundCache = new Map();       // roundIdx -> /api/predict/spatial/round payload
let currentSpatialData = null;           // 当前回合的整回合预测（null = 未就绪）
let lastSpatialTick = -1;                // 上次刷新 chips/曲线标记的 tick
let spatialLoadingRound = -1;            // 正在请求的回合（并发去重）
let curveModal = { open: false, playerIdx: -1, metric: null };

function setModelStatus(loaded, text) {
  modelLoaded = loaded;
  for (const id of ['model-dot', 'model-dot2']) {
    const dot = $(id);
    dot.className = 'model-dot' + (loaded ? ' loaded' : '');
  }
  $('model-status-text').textContent = text;
  $('model-status-text2').textContent = text;
  $('btn-predict').disabled = !loaded || !matchData;
  // 采样/扫描按钮随模型加载状态启用/禁用
  document.querySelectorAll('.pd-sample-btn').forEach(b => b.disabled = !loaded || !matchData);
  document.querySelectorAll('.pd-scan-btn, .pd-scan-all-btn, #btn-scan-all-players').forEach(b => b.disabled = !loaded || !matchData);
}

async function uploadModel(file) {
  // 优先用主面板（回放页）的 device 选择；登录页的作为兜底
  const deviceSel = $('model-device2') || $('model-device');
  const device = deviceSel ? deviceSel.value : 'mps';
  const fd = new FormData();
  fd.append('file', file);
  fd.append('device', device);
  setModelStatus(false, '加载模型中…');
  $('model-dot').classList.add('loading');
  showLoading(true, '加载预训练模型…（首次较慢）');
  try {
    const res = await fetch('/api/model/upload', { method: 'POST', body: fd });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    setModelStatus(true, `✓ ${json.checkpoint} (step ${json.step ?? '?'}, ${json.device})`);
    showToast(`模型已加载：${json.checkpoint}（${json.params_m}M 参数，${json.device}）`);
    $('pred-model-row').classList.remove('hidden');
    $('model-device-row')?.classList.add('hidden');
    $('model-device-row2')?.classList.add('hidden');
  } catch (err) {
    setModelStatus(false, '模型加载失败');
    showToast('模型加载失败: ' + err.message, true);
  } finally {
    showLoading(false);
    $('model-dot').classList.remove('loading');
  }
}

/**
 * 将预测起始 tick 同步到当前播放位置（currentSampleIdx）。
 * 只在用户未手动调整过 tick 时调用，保证"看到哪预测到哪"。
 */
function syncPredTickToPlayhead() {
  if (!currentRound) return;
  const tickInput = $('pred-tick');
  const T = getSampleCount();
  if (T <= 0) return;
  // currentSampleIdx 是当前画面所在 tick；预测需保证其后有足够未来 tick
  const maxTick = Math.max(0, T - 1);
  tickInput.max = maxTick;   // 同步滑块范围（切回合后 T 可能变化）
  const t = Math.min(currentSampleIdx, maxTick);
  if (tickInput.value !== String(t)) {
    tickInput.value = t;
    $('pred-tick-val').textContent = t;
  }
}

async function runPrediction(focusPlayer = -1, gtInfo = null) {
  if (!modelLoaded) { showToast('请先上传预训练模型', true); return; }
  if (!matchData) { showToast('请先加载回放数据', true); return; }

  // 若未手动调整过 tick，预测前同步到当前播放位置（所见即所测）
  if (!predTickManual && currentRound) {
    syncPredTickToPlayhead();
  }
  const roundIdx = parseInt($('round-select').value) || 0;
  const tick = parseInt($('pred-tick').value) || 0;
  const temperature = parseFloat($('pred-temp').value) || 0;

  $('predict-status').textContent = '预测中…';
  $('btn-predict').disabled = true;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ round_idx: roundIdx, tick, temperature, return_logp: true }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);

    lastPredResult = json;
    // focusPlayer ≥ 0 时 3D 只绘制该玩家的预测/GT 轨迹（其余玩家路径隐藏），
    // 并在路径终点旁标注分数：紫色=预测分，绿色=实际分（扫描条目带过来）
    renderPrediction(json, currentRound.teams, matchData.players.map(p => p.name), focusPlayer,
      (gtInfo && gtInfo.per_tick != null && isFinite(gtInfo.per_tick)) ? gtInfo.per_tick : null);
    $('prediction-banner').classList.remove('hidden');
    const focusNm = focusPlayer >= 0 && matchData.players[focusPlayer]
      ? matchData.players[focusPlayer].name : null;
    $('pred-banner-meta').textContent =
      `回合 ${roundIdx + 1} · tick ${json.query_tick} · 预测 ${json.output_T} ticks · temp ${temperature}` +
      (focusNm ? ` · 聚焦 ${focusNm}` : '');
    $('predict-status').textContent = `✓ 完成（${json.output_T} ticks）`;

    // 预测结果列表（focusPlayer ≥ 0 时只显示该玩家；gtInfo 时附实际分/差值）
    renderPredResults(json, focusPlayer, gtInfo);
  } catch (err) {
    $('predict-status').classList.add('error');
    $('predict-status').textContent = '✕ ' + err.message;
  } finally {
    $('btn-predict').disabled = false;
  }
}

/**
 * 渲染预测结果列表。
 * @param {object} result      — /api/predict 返回
 * @param {number} focusPlayer — ≥0 时只显示该玩家（扫描条目跳转后聚焦用）；
 *                               -1 显示全部 10 人
 * @param {object|null} gtInfo — 扫描条目携带的真实路径分数 {per_tick}，
 *                               存在时显示"实际分 + 差值"对比
 */
function renderPredResults(result, focusPlayer = -1, gtInfo = null) {
  const box = $('pred-results');
  box.innerHTML = '';
  const trajs = focusPlayer >= 0
    ? result.trajectories.filter(t => t.player_idx === focusPlayer)
    : result.trajectories;
  if (focusPlayer >= 0 && !trajs.length) {
    box.innerHTML = `<div class="scan-empty">该玩家不在预测结果中</div>`;
    return;
  }
  for (const t of trajs) {
    const row = document.createElement('div');
    row.className = 'pred-player-row' + (focusPlayer >= 0 ? ' focused' : '');
    if (!t.is_alive) {
      row.innerHTML = `<span class="pp-idx">${t.player_idx + 1}</span><span class="pp-name">P${t.player_idx + 1}</span><span class="pp-dist" style="color:var(--text-dim)">已死亡</span>`;
    } else {
      const dist = (a, b) => {
        if (!a.length || !b.length) return null;
        const dx = a[a.length-1][0] - b[b.length-1][0];
        const dy = a[a.length-1][1] - b[b.length-1][1];
        const dz = a[a.length-1][2] - b[b.length-1][2];
        return Math.sqrt(dx*dx + dy*dy + dz*dz);
      };
      const d = dist(t.pred_points, t.gt_points);
      const dStr = d === null ? '—' : d.toFixed(0) + 'u';
      const cls = d === null ? '' : (d < 200 ? 'pp-good' : d < 600 ? '' : 'pp-bad');
      // 模型对预测路径的自评分（/api/predict 附带 pred_logp）
      let lpHtml = '';
      if (t.pred_logp && isFinite(t.pred_logp.per_tick)) {
        const lp = t.pred_logp;
        const lpCls = lp.per_tick < -2.5 ? 'pp-bad' : lp.per_tick < -1.5 ? 'pp-warn' : 'pp-good';
        lpHtml = `<span class="pp-lp ${lpCls}" title="预测路径自评分（tick 等权 log p，越接近 0 模型越认可）· 总 ${lp.total.toFixed(1)} · ${lp.ticks} ticks · ${lp.tokcount} tokens">预测分 ${lp.per_tick.toFixed(2)}</span>`;
      }
      // 实际走法分数（扫描条目带过来）+ 差值对比
      let gtHtml = '', diffHtml = '';
      if (gtInfo && gtInfo.per_tick != null && isFinite(gtInfo.per_tick)
          && t.pred_logp && isFinite(t.pred_logp.per_tick)) {
        const gtCls = gtInfo.per_tick < -2.5 ? 'pp-bad' : gtInfo.per_tick < -1.5 ? 'pp-warn' : 'pp-good';
        gtHtml = `<span class="pp-lp pp-gt ${gtCls}" title="该玩家实际走法的分数（扫描时 teacher-forcing 算出，越低 = 走得越不寻常）">实际分 ${gtInfo.per_tick.toFixed(2)}</span>`;
        const diff = t.pred_logp.per_tick - gtInfo.per_tick;
        const diffCls = diff >= 0.1 ? 'pp-good' : diff <= -0.1 ? 'pp-bad' : '';
        const sign = diff > 0 ? '+' : '';
        diffHtml = `<span class="pp-lp pp-diff ${diffCls}" title="预测分 − 实际分：正数 = 预测路径比实际走法更被模型认可（偏差越大说明这一步越不像职业打法）">差 ${sign}${diff.toFixed(2)}</span>`;
      }
      const nm = (matchData && matchData.players && matchData.players[t.player_idx]
        && matchData.players[t.player_idx].name) || `P${t.player_idx + 1}`;
      if (focusPlayer >= 0) {
        row.innerHTML = `
          <div class="pp-main">
            <span class="pp-idx">${t.player_idx + 1}</span>
            <span class="pp-name">${nm} · ${t.pred_steps}/${t.gt_steps} ticks</span>
            <span class="pp-dist ${cls}">${dStr}</span>
          </div>
          <div class="pp-scores">
            ${lpHtml}
            ${gtHtml}
            ${diffHtml}
          </div>
        `;
      } else {
        row.innerHTML = `
          <span class="pp-idx">${t.player_idx + 1}</span>
          <span class="pp-name">P${t.player_idx + 1} · ${t.pred_steps}/${t.gt_steps} ticks</span>
          ${lpHtml}
          ${gtHtml}
          ${diffHtml}
          <span class="pp-dist ${cls}">${dStr}</span>
        `;
      }
    }
    box.appendChild(row);
  }
}

/**
 * 对单个玩家并行采样多条路径（/api/predict/player-sampled）。
 * 采样条数取玩家详情卡里的输入框；温度沿用预测面板的温度滑块。
 */
async function runPlayerSampling(playerIdx) {
  if (!modelLoaded) { showToast('请先上传预训练模型', true); return; }
  if (!matchData) { showToast('请先加载回放数据', true); return; }

  // 若未手动调整过 tick，采样前同步到当前播放位置
  if (!predTickManual && currentRound) {
    syncPredTickToPlayhead();
  }
  const roundIdx = parseInt($('round-select').value) || 0;
  const tick = parseInt($('pred-tick').value) || 0;
  let temperature = parseFloat($('pred-temp').value) || 0;
  // 采样需要温度 > 0（0=argmax 时所有路径相同，没有意义）；滑块为 0 时自动用 1.0
  if (temperature <= 0) temperature = 1.0;

  const numInput = document.querySelector(`.pd-sample-num[data-player="${playerIdx}"]`)
    || document.querySelector(`#player-detail-${playerIdx} .pd-sample-num`);
  const num_samples = Math.max(2, Math.min(32, parseInt(numInput?.value) || 8));
  const btn = document.querySelector(`.pd-sample-btn[data-player="${playerIdx}"]`);

  $('predict-status').textContent = `P${playerIdx + 1} 采样 ${num_samples} 条中…`;
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/predict/player-sampled', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        round_idx: roundIdx, tick,
        player_idx: playerIdx,
        num_samples,
        temperature,
      }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);

    if (!json.is_alive) {
      $('predict-status').textContent = `✕ P${playerIdx + 1} 在该 tick 已阵亡，无法采样`;
      return;
    }

    renderPlayerSamples(json, currentRound.teams, matchData.players.map(p => p.name));
    $('prediction-banner').classList.remove('hidden');
    $('pred-banner-meta').textContent =
      `回合 ${roundIdx + 1} · tick ${json.query_tick} · P${playerIdx + 1} 采样 ×${json.num_samples} · temp ${temperature}`;
    const nSteps = json.samples.length
      ? Math.max(...json.samples.map(s => s.pred_steps)) : 0;
    $('predict-status').textContent =
      `✓ P${playerIdx + 1} 采样完成（${json.num_samples} 条 · 最长 ${nSteps} ticks）`;
    showToast(`✓ 已为 P${playerIdx + 1} 采样 ${json.num_samples} 条路径`);
  } catch (err) {
    $('predict-status').classList.add('error');
    $('predict-status').textContent = '✕ ' + err.message;
    showToast('采样失败: ' + err.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

/** 刷新 spatial-only 模型状态（页面加载 / 上传后调用） */
function refreshSpatialStatus() {
  fetch('/api/model/status').then(r => r.json()).then((st) => {
    const sp = st.spatial || {};
    spatialAvailable = !!(sp.available);
    const el = $('spatial-status');
    if (el) {
      if (spatialAvailable) {
        el.textContent = `模型：${(sp.tasks || []).join(' / ')} · ${sp.device || 'cpu'}`;
        el.classList.remove('error');
        el.classList.add('ok');
      } else {
        el.textContent = '单局面模型：未配置';
        el.classList.remove('ok');
      }
    }
    // 数据已就绪且模型可用 → 自动预测当前回合
    if (spatialAvailable && matchData) loadSpatialRound(currentRoundIdx());
  }).catch(() => {});
}

/** 上传一个或多个 spatial-only 任务 checkpoint（任务由 ckpt 内部 task 字段识别） */
async function uploadSpatialModels(files) {
  const fd = new FormData();
  for (const f of files) fd.append('file', f);
  // 不传 device：spatial 推理用服务端默认（cpu，MPS 有内存损坏问题）
  $('spatial-status').textContent = `单局面模型：上传中（${files.length} 个）…`;
  try {
    const res = await fetch('/api/spatial/upload', { method: 'POST', body: fd });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    const bad = (json.results || []).filter(r => r.error);
    const tasks = (json.tasks || []).join(' / ');
    spatialAvailable = true;
    spatialRoundCache.clear();          // 模型变了 → 旧预测作废
    currentSpatialData = null;
    lastSpatialTick = -1;
    const el = $('spatial-status');
    el.textContent = `模型：${tasks}（${json.device || 'cpu'}）`;
    el.classList.remove('error');
    el.classList.add('ok');
    showToast(`spatial-only 模型已加载：${tasks}`, !!bad.length);
    if (bad.length) console.warn('spatial 上传失败：', bad);
    if (matchData) loadSpatialRound(currentRoundIdx());
    else updateSpatialChips();
  } catch (err) {
    const el = $('spatial-status');
    el.textContent = '单局面模型：加载失败';
    el.classList.add('error');
    showToast('spatial-only 模型加载失败: ' + err.message, true);
  }
}

/**
 * 切回合自动预测该回合全部 tick（服务端缓存，来回切换不重算）。
 * 命中前端缓存直接应用；否则请求 /api/predict/spatial/round 并缓存。
 */
async function loadSpatialRound(roundIdx) {
  if (!spatialAvailable || !matchData) return;
  const cached = spatialRoundCache.get(roundIdx);
  if (cached) { applySpatialRoundData(cached); return; }
  if (spatialLoadingRound === roundIdx) return;   // 同一回合的请求进行中
  spatialLoadingRound = roundIdx;
  const el = $('spatial-status');
  if (el && currentRoundIdx() === roundIdx) {
    el.textContent = `第 ${roundIdx + 1} 回合预测中…`;
    el.classList.remove('error');
  }
  try {
    const res = await fetch('/api/predict/spatial/round', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ round_idx: roundIdx }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    spatialRoundCache.set(roundIdx, json);
    if (currentRoundIdx() === roundIdx) applySpatialRoundData(json);
  } catch (err) {
    if (currentRoundIdx() === roundIdx) {
      const el2 = $('spatial-status');
      if (el2) { el2.textContent = '✕ ' + err.message; el2.classList.add('error'); }
      clearMetricChart();
    }
  } finally {
    if (spatialLoadingRound === roundIdx) spatialLoadingRound = -1;
  }
}

/** 应用某回合的整回合预测数据：刷新状态栏 / 玩家卡片数值 / 聚合胜率曲线 */
function applySpatialRoundData(data) {
  currentSpatialData = data;
  lastSpatialTick = -1;          // 强制下一帧刷新 chips + 曲线标记
  updateSpatialChips();
  renderSpatialCurve(data, currentSampleIdx);
  const el = $('spatial-status');
  if (el) {
    const n = (data.ticks && data.ticks.length) || 0;
    el.textContent = `第 ${data.round_idx + 1} 回合 · ${n} ticks · ${data.cached ? '缓存' : '已预测'}`;
    el.classList.remove('error');
    el.classList.add('ok');
  }
  if (curveModal.open) redrawPlayerCurve(currentSampleIdx);
}

/** 把当前 tick 的每玩家概率刷到左侧玩家卡片的 chips 上 */
function updateSpatialChips() {
  const data = currentSpatialData;
  const tick = currentSampleIdx;
  for (let p = 0; p < 10; p++) {
    const row = document.querySelector(`.pc-spatial[data-player="${p}"]`);
    if (!row) continue;
    const chips = row.querySelectorAll('.pc-chip');
    if (!data || !data.ticks || tick < 0 || tick >= data.ticks.length) {
      chips.forEach(c => { c.textContent = '—'; c.classList.remove('has', 'dead'); });
      continue;
    }
    const t = data.ticks[tick];
    const alive = !!(t.alive_mask && t.alive_mask[p]);
    for (const c of chips) {
      const m = c.dataset.metric;
      let v = null;
      if (t[m]) v = t[m][p];
      if (v == null || !isFinite(v)) { c.textContent = '—'; c.classList.remove('has', 'dead'); continue; }
      c.textContent = `${(v * 100).toFixed(0)}%`;
      c.classList.toggle('dead', !alive);
      c.classList.add('has');
    }
  }
}

// ── 玩家指标曲线弹窗 ────────────────────────────────────────────────

/** 点击玩家卡片上的概率数值：打开该玩家该指标的整回合曲线弹窗 */
function openPlayerCurve(playerIdx, metric) {
  if (!currentSpatialData) { showToast('该回合 spatial-only 预测尚未完成', true); return; }
  const t = currentSpatialData.ticks[currentSampleIdx] || {};
  const loaded = metric === 'winrate' ? !!t.winrate_team : !!t[metric];
  if (!loaded) { showToast(`「${METRIC_LABELS[metric] || metric}」任务未加载`, true); return; }
  curveModal = { open: true, playerIdx, metric };
  const name = (matchData.players[playerIdx] && matchData.players[playerIdx].name) || `P${playerIdx + 1}`;
  const team = (currentRound && currentRound.teams && currentRound.teams[playerIdx]) || '?';
  const tag = $('cm-team');
  tag.textContent = team;
  tag.className = 'team-tag ' + (team === 'CT' ? 'ct' : (team === 'T' ? 't' : ''));
  $('cm-title').textContent = `${name} · ${METRIC_LABELS[metric] || metric}曲线`;
  $('curve-modal').classList.remove('hidden');
  renderPlayerMetricCurve(currentSpatialData, playerIdx, metric, currentSampleIdx);
}

function closePlayerCurve() {
  curveModal = { open: false, playerIdx: -1, metric: null };
  $('curve-modal').classList.add('hidden');
  hideChartTooltip();
}

/** 刷新当前回合的 spatial 数据（清缓存后 / 换文件后调用） */
function resetSpatialRound() {
  spatialRoundCache.clear();
  currentSpatialData = null;
  lastSpatialTick = -1;
  clearMetricChart();
  updateSpatialChips();
}

// ═══════════════════════════════════════════════════════════════════════
// 回合低概率移动扫描
// ═══════════════════════════════════════════════════════════════════════

/** 扫描列表条数：只显示分数最低的 N 个移动（服务端还会按起点间隔 ≥8 tick 去重）。 */
const SCAN_TOP_N = 3;
/** 最近一次扫描的玩家索引（-1 = 无）；扫描结果/缓存状态跟随 TA。 */
let lastScanPlayer = -1;

/** 扫描指定玩家的低概率移动（服务端按 (文件, checkpoint, 回合, 玩家) 缓存）。
 *  silent=true 时不做全屏 loading（用于从缓存静默恢复结果）。 */
async function scanPlayer(playerIdx, silent = false) {
  if (!modelLoaded) { showToast('请先上传预训练模型', true); return; }
  if (!matchData) { showToast('请先加载回放数据', true); return; }

  const roundIdx = currentRoundIdx();
  lastScanPlayer = playerIdx;
  const nm = (matchData.players[playerIdx] && matchData.players[playerIdx].name) || `P${playerIdx + 1}`;
  const btn = document.querySelector(`.pd-scan-btn[data-player="${playerIdx}"]`);
  if (btn) btn.disabled = true;
  $('scan-status').textContent = '扫描中…';
  if (!silent) showLoading(true, `正在扫描第 ${roundIdx + 1} 回合 ${nm} 的低概率移动…`);

  try {
    const res = await fetch('/api/scan/round', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ round_idx: roundIdx, top_n: SCAN_TOP_N, player_idx: playerIdx }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    renderScanList(json);
    $('scan-status').textContent = json.cached
      ? `已缓存（${json.items.length} 条）`
      : `完成（${json.items.length} 条）`;
    if (!silent) {
      showToast(json.cached
        ? `✓ ${nm}：命中缓存，未重复计算`
        : `✓ ${nm}：扫描完成（${json.items.length} 个低概率移动）`);
    }
  } catch (err) {
    $('scan-status').textContent = '扫描失败';
    showToast(`扫描 ${nm} 失败: ` + err.message, true);
  } finally {
    if (btn) btn.disabled = false;
    if (!silent) showLoading(false);
  }
}

/** 最近一次全回合汇总结果（跨回合跳转后用于恢复汇总列表显示）。 */
let scanAllJson = null;
let scanAllPollTimer = null;

/** 批量扫描某玩家所有回合的低概率移动：后台逐回合算，轮询状态增量显示。 */
async function scanAllRounds(playerIdx) {
  if (!modelLoaded) { showToast('请先上传预训练模型', true); return; }
  if (!matchData) { showToast('请先加载回放数据', true); return; }
  const nm = (matchData.players[playerIdx] && matchData.players[playerIdx].name) || `P${playerIdx + 1}`;
  lastScanPlayer = playerIdx;
  setScanButtonsDisabled(true);
  $('scan-status').textContent = '全回合扫描中…';
  showLoading(true, `正在扫描 ${nm} 的全部回合（每回合保留 ${SCAN_TOP_N} 个）…`);
  try {
    const res = await fetch('/api/scan/all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_idx: playerIdx, top_n: SCAN_TOP_N }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    if (json.running && !json.started) {
      showLoading(false);
      setScanButtonsDisabled(false);
      showToast('该玩家已有全回合扫描在进行，正在读取进度…');
    }
    pollScanAll(playerIdx);
  } catch (err) {
    showLoading(false);
    setScanButtonsDisabled(false);
    $('scan-status').textContent = '扫描失败';
    showToast(`全回合扫描失败: ` + err.message, true);
  }
}

/** 轮询全回合扫描进度；边算边渲染已完成的回合，完成后停止。 */
function pollScanAll(playerIdx) {
  if (scanAllPollTimer) clearInterval(scanAllPollTimer);
  scanAllPollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/scan/all/status?player_idx=${playerIdx}`);
      const st = await res.json();
      if (!res.ok || st.error) throw new Error(st.error || `HTTP ${res.status}`);
      if (st.items && st.items.length) {
        scanAllJson = st;
        renderScanList(st);
      }
      if (st.done) {
        clearInterval(scanAllPollTimer);
        scanAllPollTimer = null;
        setScanButtonsDisabled(false);
        showLoading(false);
        const nm = (matchData.players[playerIdx] && matchData.players[playerIdx].name) || `P${playerIdx + 1}`;
        $('scan-status').textContent = `全部 ${st.total_rounds} 回合 · ${st.n_items} 条`;
        showToast(`✓ ${nm}：全部 ${st.total_rounds} 回合扫描完成（${st.n_items} 个低概率移动）`);
      } else {
        $('scan-status').textContent = `全回合扫描中 ${st.current_round + 1}/${st.total_rounds}…`;
      }
    } catch (err) {
      clearInterval(scanAllPollTimer);
      scanAllPollTimer = null;
      setScanButtonsDisabled(false);
      showLoading(false);
      $('scan-status').textContent = '扫描失败';
      showToast('全回合扫描失败: ' + err.message, true);
    }
  }, 1000);
}

/** 批量扫描期间禁用扫描相关按钮，避免并发触发。 */
function setScanButtonsDisabled(v) {
  document.querySelectorAll('.pd-scan-btn, .pd-scan-all-btn').forEach(b => b.disabled = v);
}

/** 一键预热：扫描全部玩家 × 全部回合（只填服务端共享缓存，不做汇总显示）。
 *  预热完成后，点任一玩家的“扫描全部回合”全部命中缓存、秒回。 */
let warmAllPollTimer = null;
let warmAllRunning = false;

async function scanAllPlayers() {
  if (!modelLoaded) { showToast('请先上传预训练模型', true); return; }
  if (!matchData) { showToast('请先加载回放数据', true); return; }
  if (warmAllRunning) { showToast('全部玩家预热已在运行中', true); return; }
  warmAllRunning = true;
  const btn = $('btn-scan-all-players');
  if (btn) { btn.disabled = true; btn.textContent = '预热中…'; }
  setScanButtonsDisabled(true);
  $('scan-status').textContent = '预热全部玩家…';
  showLoading(true, `正在扫描全部玩家 × 全部回合（每回合每玩家保留 ${SCAN_TOP_N} 个），结果写入缓存…`);
  try {
    const res = await fetch('/api/scan/all/players', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ top_n: SCAN_TOP_N }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    pollWarmAll();
  } catch (err) {
    warmAllRunning = false;
    if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label; }
    setScanButtonsDisabled(false);
    showLoading(false);
    $('scan-status').textContent = '';
    showToast('全部玩家预热失败: ' + err.message, true);
  }
}

function pollWarmAll() {
  if (warmAllPollTimer) clearInterval(warmAllPollTimer);
  warmAllPollTimer = setInterval(async () => {
    try {
      const res = await fetch('/api/scan/all/players/status');
      const st = await res.json();
      if (!res.ok || st.error) throw new Error(st.error || `HTTP ${res.status}`);
      if (st.running) {
        const pct = st.total_jobs ? Math.min(100, Math.round(st.done_jobs / st.total_jobs * 100)) : 0;
        const msg = `预热 ${st.done_jobs}/${st.total_jobs}（${pct}% · R${st.current_round + 1}/${st.total_rounds}）…`;
        $('scan-status').textContent = msg;
        // 全屏遮罩把页面盖住了，进度必须写进遮罩文字用户才看得到
        const lt = $('loading-overlay-text');
        if (lt) lt.textContent = msg;
        setWarmProgressBar(pct);
      } else if (st.done) {
        clearInterval(warmAllPollTimer);
        warmAllPollTimer = null;
        warmAllRunning = false;
        clearWarmProgressBar();
        const btn = $('btn-scan-all-players');
        if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label; }
        setScanButtonsDisabled(false);
        showLoading(false);
        const players = st.total_rounds ? Math.round(st.total_jobs / st.total_rounds) : 0;
        $('scan-status').textContent = `预热完成（${st.done_jobs} 项已缓存）`;
        showToast(`✓ 全部 ${st.total_rounds} 回合 × ${players} 名玩家已预热，各玩家“扫描全部回合”将秒回`);
      } else {
        // started=false 且没有 running/done：任务被清空或服务端异常，停止轮询
        clearInterval(warmAllPollTimer);
        warmAllPollTimer = null;
        warmAllRunning = false;
        clearWarmProgressBar();
        const btn = $('btn-scan-all-players');
        if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label; }
        setScanButtonsDisabled(false);
        showLoading(false);
        $('scan-status').textContent = '';
        showToast('预热任务不存在或已被清空', true);
      }
    } catch (err) {
      clearInterval(warmAllPollTimer);
      warmAllPollTimer = null;
      warmAllRunning = false;
      clearWarmProgressBar();
      const btn = $('btn-scan-all-players');
      if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label; }
      setScanButtonsDisabled(false);
      showLoading(false);
      $('scan-status').textContent = '';
      showToast('预热进度读取失败: ' + err.message, true);
    }
  }, 1000);
}

/** 全屏遮罩上的预热进度条：首次调用在遮罩里创建，之后只更新宽度。 */
function setWarmProgressBar(pct) {
  const ov = $('loading-overlay');
  if (!ov) return;
  let bar = document.getElementById('warm-progress-bar');
  if (!bar) {
    const wrap = document.createElement('div');
    wrap.className = 'loading-progress';
    wrap.innerHTML = '<div class="loading-progress-bar" id="warm-progress-bar"></div>';
    ov.appendChild(wrap);
    bar = wrap.firstElementChild;
  }
  bar.style.width = pct + '%';
}

function clearWarmProgressBar() {
  const wrap = document.querySelector('.loading-progress');
  if (wrap) wrap.remove();
}

/** 渲染扫描结果列表（json 为 null 时清空；scan_all 汇总按分数从低到高平铺显示）。 */
function renderScanList(json) {
  const box = $('scan-list');
  if (!box) return;
  box.innerHTML = '';
  if (!json || !json.items || !json.items.length) {
    scanAllJson = null;
    box.innerHTML = '<div class="scan-empty">本回合没有可扫描的条件（回合太短或全员已阵亡）</div>';
    return;
  }
  // 列标题行（悬浮每列标题可看含义；data-tip 即时气泡，不依赖浏览器原生 title）
  const head = document.createElement('div');
  head.className = 'scan-head-row';
  head.innerHTML = `
    <span class="scan-col-tick" data-tip="回合内第 N 个 tick（0.25 秒一个），即该时刻的走位被判定为低概率">tick</span>
    <span class="scan-col-name" data-tip="该时刻的玩家（扫描的是左侧展开的那位）">玩家</span>
    <span class="scan-col-meta" data-tip="从该时刻起未来路径的总位移（u = 游戏单位，越大 = 跑得越远）· 有效 tick 数（起点后至少存活 2 秒的移动才入选，此处即未来存活长度）">位移 · 有效t</span>
    <span class="scan-col-score" data-tip="真实走法的分数（tick 等权 log p）：越负 = 模型越觉得这一步不该这么走。红 &lt; -2.5，黄 &lt; -1.5，绿 = 还算正常">分数</span>
  `;
  box.appendChild(head);
  const names = matchData ? matchData.players : [];
  if (json.scan_all) {
    // 汇总模式：全回合合并，按分数从低到高（最不可能在前），
    // 不再按回合分组；每行带回合小标签，点击仍会先切到对应回合
    const items = [...json.items].sort((a, b) => a.per_tick - b.per_tick);
    for (const it of items) box.appendChild(buildScanItem(it, names, true));
  } else {
    for (const it of json.items) box.appendChild(buildScanItem(it, names));
  }
}

/** 构建单条扫描条目（含点击跳转）。showRound=true 时行内显示所属回合（汇总模式用）。 */
function buildScanItem(it, names, showRound) {
  const nm = (names[it.player] && names[it.player].name) || `P${it.player + 1}`;
  const team = it.team === 'CT' || it.team === 'T' ? it.team : '?';
  const roundTag = showRound && it.round_idx != null
    ? ` <span class="scan-item-round">R${it.round_idx + 1}</span>` : '';
  const score = it.per_tick == null || !isFinite(it.per_tick)
    ? '—'
    : it.per_tick.toFixed(2);
  const cls = it.per_tick == null || !isFinite(it.per_tick)
    ? ''
    : (it.per_tick < -2.5 ? 'bad' : it.per_tick < -1.5 ? 'warn' : '');
  const row = document.createElement('div');
  row.className = 'scan-item';
  row.dataset.tick = it.tick;
  row.dataset.player = it.player;
  row.innerHTML = `
    <span class="scan-item-tick">t${it.tick}</span>
    <span class="scan-item-name">${nm}${roundTag}
      <span class="scan-item-team ${team === 'CT' ? 'ct' : 't'}">${team}</span>
    </span>
    <span class="scan-item-meta">${it.disp != null ? Math.round(it.disp) + 'u' : ''}${it.tokcount ? ' · ' + Math.max(1, Math.round(it.tokcount / 7)) + 't' : ''}</span>
    <span class="scan-item-score ${cls}">${score}</span>
  `;
  row.addEventListener('click', () => jumpToScanItem(it));
  return row;
}

/** 点击扫描条目：跳到该 tick → 摄像机移到该玩家 → 设置预测起点 → 展开玩家 → 运行预测。 */
function jumpToScanItem(item) {
  if (!matchData) return;

  // 锁定画面：停止播放，避免预测耗时（数秒）期间播放头继续前进，
  // 导致预测轨迹（起点 = 该 tick）与 3D 玩家位置（播放头位置）错位
  // （曾出现：预测 t116，画面已播到 t181 → 轨迹起点和玩家位置差 65 tick）
  setPlaying(false);
  updatePlayButton();

  // 汇总列表可能来自其它回合：先切到目标回合
  // （onRoundChanged 会清空列表/预测并把 scanAllJson 置空，故先保存再恢复）
  const targetRound = item.round_idx != null ? item.round_idx : currentRoundIdx();
  const all = scanAllJson;
  if (targetRound !== currentRoundIdx()) {
    // 必须先同步 #round-select 的值：currentRoundIdx() 读的是下拉框，
    // 只调 loadRound 会让预测/扫描接口拿到旧回合（预测起点与 3D 位置错位）
    const sel = $('round-select');
    if (sel) sel.value = String(targetRound);
    loadRound(targetRound);
    onRoundChanged();
    if (all) renderScanList(all);
  }
  if (!currentRound) return;

  // 跳转到该 tick（round_seconds 秒；timeline 点击后同样要手动重绘）
  const secs = currentRound.round_seconds;
  if (secs && secs[item.tick] != null) {
    seekTo(secs[item.tick]);
    drawTimeline();
  }

  // 摄像机移动聚焦到该玩家（orbit 模式平滑飞过去；
  // 第一/三人称模式先切回 orbit，避免跟拍的是别的玩家）
  if (cameraMode !== 'orbit') setCameraMode('orbit');
  const st = getPlayerState(item.player, item.tick);
  if (st && st.alive) {
    const wp = gameToThree(st.x, st.y, st.z);
    flyCameraTo(new THREE.Vector3(wp.x, wp.y, wp.z));
  }

  // 预测起点设为该 tick，并停止自动跟随
  predTickManual = true;
  const tickInput = $('pred-tick');
  // 先刷新滑块范围再设值：切回合后若没触发 onRoundChanged（同回合条目），
  // max 可能是旧回合的 T-1，value 会被 range 钳制 → 预测到错误的 tick
  tickInput.max = Math.max(0, getSampleCount() - 1);
  tickInput.value = item.tick;
  $('pred-tick-val').textContent = item.tick;

  // 高亮选中行 + 展开该玩家详情
  document.querySelectorAll('.scan-item').forEach(el => el.classList.remove('active'));
  const sel = document.querySelector(
    `.scan-item[data-tick="${item.tick}"][data-player="${item.player}"]`);
  if (sel) sel.classList.add('active');
  togglePlayerDetail(item.player);

  // 运行路径预测（结果含预测分 / 实际分 / 差值对比；
  // 结果列表与 3D 轨迹都只聚焦该玩家）
  runPrediction(item.player, { per_tick: item.per_tick });
  showToast(`已跳转到 t${item.tick}（${item.name}），正在预测…`);
}

/** 刷新当前扫描玩家的缓存状态（切回合/切玩家时调用，不触发计算）。
 *  已缓存且列表为空 → 静默恢复结果（命中缓存秒回）。 */
function refreshScanStatus() {
  const el = $('scan-status');
  if (!el) return;
  if (lastScanPlayer < 0) { el.textContent = ''; return; }
  fetch(`/api/scan/status?round_idx=${currentRoundIdx()}&player_idx=${lastScanPlayer}&top_n=${SCAN_TOP_N}`)
    .then(r => r.json())
    .then((st) => {
      if (!el) return;
      if (st.cached) {
        el.textContent = `已缓存（${st.n_items} 条）`;
        const list = $('scan-list');
        if (list && !list.children.length) scanPlayer(lastScanPlayer, true);
      } else {
        el.textContent = '未扫描';
      }
    })
    .catch(() => {});
}

/** 释放所有服务端缓存（demo 解析 / 回合扫描 / 下游权重）。 */
async function clearAllCaches() {
  try {
    const res = await fetch('/api/cache/clear', { method: 'POST' });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    const c = json.cleared || {};
    renderScanList(null);
    $('scan-status').textContent = '';
    // spatial：服务端整回合缓存已清 → 前端缓存一并作废并重算当前回合
    resetSpatialRound();
    if (spatialAvailable && matchData) loadSpatialRound(currentRoundIdx());
    showToast(`✓ 已释放缓存：demo ${c.demo} · 扫描 ${c.scan} · spatial ${c.spatial ?? 0}`);
  } catch (err) {
    showToast('释放缓存失败: ' + err.message, true);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 回放控制
// ═══════════════════════════════════════════════════════════════════════

function togglePlay() {
  if (!currentRound) return;
  setPlaying(!isPlaying);
  updatePlayButton();
}

function updatePlayButton() {
  $('play-icon').style.display = isPlaying ? 'none' : '';
  $('pause-icon').style.display = isPlaying ? '' : 'none';
}

function updateTimeDisplay() {
  if (!currentRound) { $('time-display').textContent = '00:00 / 00:00'; return; }
  const dur = getRoundDuration();
  let html = `${formatTime(currentTime)} / ${formatTime(dur)}`;
  // 炸弹倒计时
  const plantT = currentRound.bomb_planted_time;
  if (plantT != null && currentTime >= plantT) {
    const remain = Math.max(0, 40 - (currentTime - plantT));
    html += ` <span style="color:#ff5c7a;font-weight:600">💣 ${formatTime(remain)}</span>`;
  }
  $('time-display').innerHTML = html;
}

function updateAllVisuals() {
  if (!currentRound) return;
  const idx = currentSampleIdx;
  const playerStates = getAllPlayerStates(idx);
  const teams = currentRound.teams;
  const showNames = $('toggle-names').checked;
  const showTrails = $('toggle-trails').checked;
  const showSmokesToggle = $('toggle-smokes').checked;

  updatePlayers(playerStates, idx, teams, showNames, showTrails);

  const bombPos = getBombPosition(idx);
  updateBomb(bombPos, isBombPlanted(idx), currentTime);

  updateGrenadeTrajectories(currentTime);

  const ct = currentTime;
  const activeSmokes = showSmokesToggle
    ? (currentRound.events.smokes || []).filter(s => s.ts <= ct && s.te >= ct)
    : [];
  const activeInfernos = showSmokesToggle
    ? (currentRound.events.infernos || []).filter(inf => inf.ts <= ct && (inf.te === null || inf.te >= ct))
    : [];
  // 只在集合变化时重建（避免每帧删除重建导致对象随机重置 → 抽动）
  const smokeKey = activeSmokes.map(s => `${s.x},${s.y},${s.z},${s.ts}`).join('|');
  const infernoKey = activeInfernos.map(i => `${i.x},${i.y},${i.z},${i.ts}`).join('|');
  if (smokeKey !== lastSmokeKey) {
    updateSmokes(activeSmokes);
    lastSmokeKey = smokeKey;
  }
  if (infernoKey !== lastInfernoKey) {
    updateInfernos(activeInfernos);
    lastInfernoKey = infernoKey;
  }

  if (cameraMode === 'first' && focusedPlayerIdx >= 0) {
    setPlayerModelVisible(focusedPlayerIdx, false);
  }

  updateAimRays(playerStates, raycaster, mapGroup);
  updatePlayerPanelFrame();

  // spatial-only：切 tick 时刷新玩家卡片数值 + 聚合曲线标记 + 弹窗竖线
  if (currentSampleIdx !== lastSpatialTick) {
    lastSpatialTick = currentSampleIdx;
    updateSpatialChips();
    if (currentSpatialData) renderSpatialCurve(currentSpatialData, currentSampleIdx);
    if (curveModal.open) redrawPlayerCurve(currentSampleIdx);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 主循环
// ═══════════════════════════════════════════════════════════════════════

let lastTime = performance.now();
let frameCount = 0;

function animate(timestamp) {
  requestAnimationFrame(animate);
  const now = performance.now();
  let dt = (now - lastTime) / 1000;
  lastTime = now;
  // 封顶 dt（防止切后台/卡顿后大跳变），但允许低帧率环境（如 30fps/60fps）
  // 正常推进：dt 保持真实间隔，仅在异常大（>0.25s）时截断。
  if (dt > 0.25) dt = 0.25;

  try {
    // advanceReplay 内部已检查 currentRound / isPlaying，这里不重复判断
    const result = advanceReplay(dt);
    if (result.advanced) {
      handleReplayEvents(result.events);
      drawTimeline();
    }
    if (result.roundEnded) {
      setPlaying(false);
      updatePlayButton();
    }

    const fps = renderFrame(dt, (idx) => getPlayerState(idx, currentSampleIdx));
    updateAllVisuals();
    updateTimeDisplay();
    updatePredictionAnimation(dt);
    // 烟雾/火焰动画（膨胀、上浮、跳动、闪烁）
    updateSmokeAnimations(dt, currentTime);
    updateInfernoAnimations(dt, currentTime);

    // 预测起始 tick 自动跟随当前播放画面（用户手动调整后停止跟随）
    if (!predTickManual && currentRound) {
      syncPredTickToPlayhead();
    }

    // Minimap（每 3 帧）
    if (frameCount % 3 === 0 && matchData && currentRound) {
      try {
        renderMinimap(getAllPlayerStates(currentSampleIdx), currentRound.teams, getBombPosition(currentSampleIdx));
      } catch (_) {}
    }
    frameCount++;
  } catch (err) {
    if (frameCount < 5 || frameCount % 600 === 0) {
      console.warn('[Vision] render error:', err);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 事件绑定
// ═══════════════════════════════════════════════════════════════════════

function bindEvents() {
  // 上传
  const dropZone = $('drop-zone');
  const fileInput = $('file-input');
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) loadFile(fileInput.files[0]);
    fileInput.value = '';
  });

  // 示例
  $('btn-load-example').addEventListener('click', async () => {
    const list = $('example-list');
    if (!list.classList.contains('hidden')) { list.classList.add('hidden'); return; }
    try {
      const res = await fetch('/api/examples');
      const json = await res.json();
      list.innerHTML = '';
      for (const ex of json.examples.slice(0, 8)) {
        const btn = document.createElement('button');
        btn.className = 'example-item';
        btn.textContent = `${ex.type === 'demo' ? '🎮' : '📄'} ${ex.name}`;
        btn.addEventListener('click', () => loadExample(ex.path));
        list.appendChild(btn);
      }
      list.classList.remove('hidden');
    } catch (err) {
      showToast('获取示例失败: ' + err.message, true);
    }
  });

  // 模型上传
  const modelInput = $('model-input');
  $('btn-model-upload').addEventListener('click', () => modelInput.click());
  modelInput.addEventListener('change', () => {
    if (modelInput.files.length) uploadModel(modelInput.files[0]);
    modelInput.value = '';
  });
  const modelInput2 = $('model-input2');
  $('btn-model-upload2').addEventListener('click', () => modelInput2.click());
  modelInput2.addEventListener('change', () => {
    if (modelInput2.files.length) uploadModel(modelInput2.files[0]);
    modelInput2.value = '';
  });

  // spatial-only 模型上传（支持多选）
  const spatialInput = $('spatial-input');
  $('btn-spatial-upload').addEventListener('click', () => spatialInput.click());
  spatialInput.addEventListener('change', () => {
    if (spatialInput.files.length) uploadSpatialModels([...spatialInput.files]);
    spatialInput.value = '';
  });

  // 预测
  $('btn-predict').addEventListener('click', runPrediction);
  // 一键预热全部玩家 × 全部回合缓存（扫描入口在左侧玩家卡片 🔍 扫描）
  $('btn-scan-all-players').addEventListener('click', scanAllPlayers);
  // 缓存释放（扫描入口在左侧玩家卡片 🔍 扫描）
  $('btn-clear-cache').addEventListener('click', clearAllCaches);
  $('btn-clear-pred').addEventListener('click', () => {
    clearPrediction();
    clearPlayerSamples();
    $('prediction-banner').classList.add('hidden');
    $('pred-results').innerHTML = '';
    lastPredResult = null;
    // spatial 聚合曲线属于自动模式：有数据就重绘，否则隐藏
    if (currentSpatialData) renderSpatialCurve(currentSpatialData, currentSampleIdx);
    else clearMetricChart();
  });
  // spatial-only 自动预测（无手动按钮；切回合自动触发，见 onRoundChanged / loadSpatialRound）
  // 玩家卡内按钮（事件委托：玩家卡片每次重建，不能直接绑）
  $('player-list').addEventListener('click', (e) => {
    // 概率 chip → 打开该玩家该指标的整回合曲线
    const chip = e.target.closest('.pc-chip');
    if (chip) {
      const row = chip.closest('.pc-spatial');
      const p = row ? parseInt(row.dataset.player) : NaN;
      if (!isNaN(p)) { e.stopPropagation(); openPlayerCurve(p, chip.dataset.metric); }
      return;
    }
    const btn = e.target.closest('.pd-sample-btn');
    if (btn) {
      const p = parseInt(btn.dataset.player);
      if (!isNaN(p)) { e.stopPropagation(); runPlayerSampling(p); }
      return;
    }
    const sbtn = e.target.closest('.pd-scan-btn');
    if (sbtn) {
      const p = parseInt(sbtn.dataset.player);
      if (!isNaN(p)) { e.stopPropagation(); scanPlayer(p); }
      return;
    }
    const abtn = e.target.closest('.pd-scan-all-btn');
    if (abtn) {
      const p = parseInt(abtn.dataset.player);
      if (!isNaN(p)) { e.stopPropagation(); scanAllRounds(p); }
    }
  });
  // 玩家指标曲线弹窗
  $('cm-close').addEventListener('click', closePlayerCurve);
  $('curve-modal').addEventListener('click', (e) => {
    if (e.target === $('curve-modal')) closePlayerCurve();   // 点击遮罩关闭
  });
  $('pred-tick').addEventListener('input', (e) => {
    predTickManual = true;   // 手动拖动 → 停止自动跟随
    $('pred-tick-val').textContent = e.target.value;
  });
  // 「使用当前画面」：把预测 tick 设回当前播放位置，恢复自动跟随
  $('btn-use-current').addEventListener('click', () => {
    predTickManual = false;
    syncPredTickToPlayhead();
    $('pred-tick-val').textContent = $('pred-tick').value;
    showToast('预测起点已同步到当前画面');
  });
  $('pred-temp').addEventListener('input', (e) => {
    $('pred-temp-val').textContent = (parseFloat(e.target.value)).toFixed(1);
  });
  $('toggle-pred').addEventListener('change', (e) => {
    setShowPrediction(e.target.checked);
  });

  // 回放控制
  $('btn-play').addEventListener('click', togglePlay);  $('speed-select').addEventListener('change', (e) => setPlaySpeed(parseFloat(e.target.value)));
  $('btn-prev-round').addEventListener('click', () => {
    if (currentRoundIdx() > 0) {
      loadRound(currentRoundIdx() - 1);
      onRoundChanged();
    }
  });
  $('btn-next-round').addEventListener('click', () => {
    if (currentRoundIdx() < totalRounds - 1) {
      loadRound(currentRoundIdx() + 1);
      onRoundChanged();
    }
  });
  $('round-select').addEventListener('change', (e) => {
    loadRound(parseInt(e.target.value));
    onRoundChanged();
  });

  $('btn-reset-cam').addEventListener('click', () => resetCamera('free'));
  $('btn-back').addEventListener('click', () => {
    $('app').classList.add('hidden');
    $('landing').classList.remove('hidden');
    clearPrediction();
  });
  $('btn-download').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/download');
      if (!res.ok) throw new Error((await res.json()).error || 'HTTP ' + res.status);
      const blob = await res.blob();
      const a = document.createElement('a');
      const url = URL.createObjectURL(blob);
      a.href = url;
      a.download = (loadedSource.replace(/\.[^.]+$/, '') || 'replay') + '.json.gz';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      showToast('下载失败: ' + err.message, true);
    }
  });

  // 键盘
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    if (e.code === 'ArrowLeft' && currentRound) seekTo(Math.max(0, currentTime - 1));
    if (e.code === 'ArrowRight' && currentRound) seekTo(Math.min(getRoundDuration(), currentTime + 1));
    if (e.code === 'KeyR') resetCamera('free');
    if (e.code === 'Escape') closePlayerCurve();
  });

  // 点击 3D 视口中的玩家 → 展开左侧对应详情卡
  // 用 pointerdown/up 区分拖拽（旋转/平移相机）与真正的点击
  const vpCanvas = $('three-canvas');
  let _vpDown = null;
  vpCanvas.addEventListener('pointerdown', (e) => {
    _vpDown = { x: e.clientX, y: e.clientY };
  });
  vpCanvas.addEventListener('pointerup', (e) => {
    if (!_vpDown) return;
    const dx = e.clientX - _vpDown.x;
    const dy = e.clientY - _vpDown.y;
    _vpDown = null;
    if (dx * dx + dy * dy > 25) return;   // 拖拽超过 5px → 视为相机操作，不拾取
    if (cameraMode === 'fly') return;     // fly 模式（pointer lock 瞄准）不拾取
    pickPlayerAt(e.clientX, e.clientY);
  });

  // 时间线点击 seek
  const tl = $('timeline-canvas');
  tl.addEventListener('click', (e) => {
    if (!currentRound) return;
    const rect = tl.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    seekTo(frac * getRoundDuration());
    drawTimeline();
  });

  // 切换开关
  $('toggle-trails').addEventListener('change', (e) => setTrailsVisible(e.target.checked));
  $('toggle-names').addEventListener('change', (e) => setNamesVisible(e.target.checked));
}

function currentRoundIdx() {
  // replay-core 的 currentRoundIdx 是模块内变量，未导出 getter，用 round-select 同步
  const sel = $('round-select');
  return parseInt(sel.value) || 0;
}

function onRoundChanged() {
  setPlaying(false);
  updatePlayButton();
  // 预测 tick 滑块范围随回合更新（各回合时长不同；旧代码只在加载 demo 时设置一次，
  // 切回合后 max 停留在首个回合的 T-1 —— 曾导致回合 3 里预测 tick 被钳制在 116）
  const tickInput = $('pred-tick');
  tickInput.max = Math.max(0, getSampleCount() - 1);
  updateRoundStats();
  drawTimeline();
  // 重建玩家面板：换边后队伍会变（第 13 回合起 CT/T 对调），需按当前回合 teams 重新分组
  updatePlayerPanel();
  // 重置烟雾/火焰缓存（新回合需要重建）
  lastSmokeKey = '';
  lastInfernoKey = '';
  updateAllVisuals();
  // 重建投掷物轨迹（新回合）
  try {
    buildGrenadeTrajectories(getGrenadeTrajectories());
  } catch (err) {
    console.warn('[Vision] 投掷物轨迹构建失败:', err);
  }
  clearPrediction();
  $('prediction-banner').classList.add('hidden');
  $('pred-results').innerHTML = '';
  // spatial-only：新回合数据未到时隐藏曲线/数值，模型可用则自动预测整回合
  currentSpatialData = null;
  lastSpatialTick = -1;
  closePlayerCurve();
  clearMetricChart();
  updateSpatialChips();
  if (spatialAvailable) loadSpatialRound(currentRoundIdx());
  // 扫描面板：清空上一回合列表；若当前关注玩家在新回合已缓存则静默恢复
  renderScanList(null);
  refreshScanStatus();
}

function fitCameraToMap() {
  const bounds = getMapBounds();
  if (!bounds) return;
  const center = new THREE.Vector3();
  bounds.getCenter(center);
  const size = new THREE.Vector3();
  bounds.getSize(size);
  const maxDim = Math.max(size.x, size.z, 1);   // 水平尺寸（地图是扁平的）

  // 俯视视角（~50°）：拉近 + 抬高，让扁平地图占满画面
  const dist = maxDim * 0.42;
  camera.position.set(
    center.x + dist,
    center.y + size.y * 2.6 + maxDim * 0.18,
    center.z + dist
  );
  controls.target.copy(center);
  controls.update();
}

function showLoading(show, text) {
  const ov = $('loading-overlay');
  if (show) {
    $('loading-overlay-text').textContent = text || '加载中…';
    ov.classList.remove('hidden');
    startLoadingAnimation();
  } else {
    ov.classList.add('hidden');
    stopLoadingAnimation();
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 初始化
// ═══════════════════════════════════════════════════════════════════════

function init() {
  initParticles();
  bindEvents();

  try {
    initScene($('three-canvas'));
    createPlayers();
    createAimRays();
  } catch (err) {
    console.error('[Vision] 3D init failed:', err);
    showToast('3D 场景初始化失败: ' + err.message, true);
  }

  // 检查服务端是否已有预加载 checkpoint
  fetch('/api/model/status').then(r => r.json()).then((st) => {
    if (st.available) {
      setModelStatus(true, `✓ ${st.checkpoint} (${st.device})`);
    }
  }).catch(() => {});
  refreshSpatialStatus();

  requestAnimationFrame(animate);
  console.log('[Vision] CS2 Vision Studio ready');
}

init();
