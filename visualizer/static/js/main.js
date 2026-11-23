/**
 * main.js — cs-net-v5 入口：demo_extract 3D 回放工作台。
 *
 * 只做三件事：
 *   1. 把 .dem / .csz 交给服务端的 demo_extract 解析（后台任务 + 进度）
 *   2. 逐 tick 平滑回放玩家（切回合时按需拉那一回合的紧凑载荷）
 *   3. 画**真实烟雾体素**和**真实火焰格子**
 *
 * 另外把 `app/ai` 的两份模型预测画到场景里（只有对枪弧线，数字在左下表格），见
 * ai-overlay.js 和本文件「AI 预测」那一段。旧的路径预测 / 低概率扫描那套
 * 依赖 demo_parser 的东西已经删干净了。
 */

import * as THREE from 'three';
import {
  initScene, renderFrame, resetCamera, scene, camera, renderer, controls,
  renderMinimap, setCameraMode, setFocusedPlayer, cameraMode, raycaster, mapGroup,
  minimapRenderer, minimapCamera, minimapGlyphMetrics, setMinimapMode2D,
  minimapPlayerAt, minimapHitDebug,
  gameToThree, flyCameraTo, SCALE,
  setFollowPlayer, getFollowPlayer, isCameraAnimating, setManualMoveHandler,
} from './scene.js';
import {
  loadMap, getMapBounds, ensureMapBVH, getMapBVHStats, getMapBVH, getMapColorStats,
  reloadMapColors,
} from './map-loader.js';
import { raycastBVH } from './map-raycast.js';
import { currentMinimapRadar } from './minimap-radar.js';
import {
  matchData, loadMatchData, loadRound, currentRound, currentSampleIdx, currentTime,
  buildGait,
  isPlaying, playSpeed, getAllPlayerStates, getPlayerState, getBombPosition, seekTo,
  getRoundDuration, setPlaying, setPlaySpeed, playerName, totalRounds, getSampleCount,
  isBombPlanted, advanceReplay, weaponName, placeName,
  visibleEnemiesOf, hasSpottedData,
  buildGrenadeTrajectories as getGrenadeTrajectories,
} from './replay-core.js';
import {
  createPlayers, createAimRays, updatePlayers, clearAllEntities,
  setPlayerNames, updateBomb, updateSmokeVolumes, updateFireCells,
  updateSmokeAnimations, updateInfernoAnimations,
  updateGrenadeTrajectories, buildGrenadeTrajectories, spawnMuzzleFlash, updateMuzzleFlashes,
  showKillEffect, setTrailsVisible, setNamesVisible, setPlayerModelVisible, updateAimRays,
  setAimRaysVisible, setGait,
  updateTracers, setTracersVisible, spawnTracer, tracerDebugStats, tracerInfo,
  smokeDebugStats, fireDebugStats, setXRayMode, setCrouchDebugVisible, playerGroupAt,
  aimRayAt, setSpotTargets, setSpotIconsVisible, spotIconAt, spotIconDebug,
  setSmokeSpritesVisible, setWeaponIconsVisible, setPlayerXRayMode, setPlayerXRayFlags,
  playerXRayDebug,
  setFireSpritesVisible, updateWeaponModels, weaponModelStats, prewarmFireShaders,
  setNadeModelsVisible, nadeModelDebugStats,
  smokeOccludersAt, createVisCornerRays, updateVisCornerRays,
  setVisCornerRaysVisible, visCornerRayDebug, VIS_CORNER_COLORS,
  createViewFrustum, updateViewFrustum, setViewFrustumVisible, viewFrustumDebug,
  setTrailHistorySource, trailDebugStats, setPlayerIdsVisible, playerIdDebug,
  setPlayerIdNames,
} from './visuals.js';
import {
  createVisibilityEvaluator, makeMapCaster, pairVisibility, eyePos,
  BLOCK_FOV, BLOCK_MAP, BLOCK_SMOKE, BLOCK_PLAYER,
} from './visibility.js';
import { createPlayerXRayTester } from './player-xray.js';
import { preloadSpriteSequences, assetDebugStats, weaponIconUrl } from './cs2-assets.js';
import {
  attachWeapon, clearWeaponModels, setWeaponModelsVisible,
  renderWeaponShowcase, weaponModelDebugStats, preloadNadeModels,
} from './weapon-models.js';
import { startLoadingAnimation, stopLoadingAnimation } from './loading.js';
import { updateAiOverlay, clearAiOverlay, aiOverlayDebug } from './ai-overlay.js';
import { createAiChart } from './ai-chart.js';
// 中英文案。`t()` 取当前语言的字符串，`data-i18n*` 由这个模块在加载时统一刷。
import { t } from './i18n/index.js';
// 统计表头上的「?」解释 —— 和玩家页共用同一份（tips.js），别各写一遍。
import { mountTips, tipTh } from './tips.js';
// NDJSON 流（AI 回合预测 / 整场统计）—— 和玩家页共用同一份解析
import { streamJson } from './ndjson.js';
// 分享图（canvas 画一张 PNG）。和玩家页共用同一份实现，见该文件的文件头。
import { downloadCard } from './share-card.js';

const $ = (id) => document.getElementById(id);

function showToast(msg, isError = false) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.remove('hidden', 'error');
  if (isError) el.classList.add('error');
  setTimeout(() => el.classList.add('hidden'), 4200);
}

function formatTime(sec) {
  if (sec == null || !isFinite(sec)) return '00:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function fmtInt(n) {
  return (n == null) ? '—' : Number(n).toLocaleString('en-US');
}

/** 转义要插进 innerHTML 的文本（选手名是 demo 给的，可能有 < & 之类）。 */
function escHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
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
      hue: Math.random() < 0.6 ? 200 : 265,
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
// 加载（后台解析任务 + 进度轮询）
// ═══════════════════════════════════════════════════════════════════════

let loadedSource = '';
let activeRoundIdx = 0;
let roundLoadSeq = 0;
/**
 * 本回合的开火事件（`events.sound` 里 `ty === "fire"`，按 tick 升序）。
 * 有精确 tick，所以能一发一发地生成曳光。老 `.csz` 没有这个字段 → null，
 * 调用方回退到 `shots` 列（每个采样窗口内的开枪数）。换回合时重算。
 * @type {Array<{t:number, p:number, w:number, sil:boolean}>|null}
 */
let roundFireEvents = null;

/** 采样方式已经在提取时定死在 .csz 里了，所以这里只报给用户看，不再选。 */

async function startLoad(path) {
  const fd = new FormData();
  fd.append('path', path);

  // 路径可能来自 Windows，所以 / 和 \ 都要当分隔符
  const label = String(path).split(/[\\/]/).pop();
  showLoading(true, t('replay.load.reading', { name: label }));
  try {
    const res = await fetch('/api/load', { method: 'POST', body: fd });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
    const summary = await pollLoadJob(json.job_id, label);
    onDataReady(summary);
  } catch (err) {
    showLoading(false);
    showToast(t('replay.toast.loadFail', { msg: err.message }), true);
  }
}

function pollLoadJob(jobId, label) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const res = await fetch(`/api/load/status?job_id=${jobId}`);
        const st = await res.json();
        if (!res.ok || (st.error && st.state !== 'error')) {
          throw new Error(st.error || `HTTP ${res.status}`);
        }
        if (st.state === 'error') throw new Error(st.error || t('replay.error.parseFail'));
        showLoading(true, t('replay.load.progress', {
          name: label,
          step: st.step || st.state,
          s: st.elapsed ?? 0,
        }));
        if (st.state === 'done') {
          showLoading(false);
          // 摘要只在第一次 done 轮询里返回；断线重试时回落到 /api/session
          if (st.summary) { resolve(st.summary); return; }
          const sres = await fetch('/api/session');
          const sess = await sres.json();
          if (!sess.loaded) throw new Error(t('replay.error.noSession'));
          resolve(sess);
          return;
        }
        setTimeout(tick, 400);
      } catch (err) {
        reject(err);
      }
    };
    tick();
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 库批次（/replay?batch=<id>）
//
// Demo 库里勾选一场或多场 → /api/library/open 建一个批次 → 跳到本页。
// 这里只负责把批次渲染成一个下拉，选中哪一场就用**原有的** POST /api/load
// 去加载它 —— 回放的数据通路完全没变，批次只是多了一层"播哪一场"的选择。
// 还没提取完的场次在下拉里显示为禁用，后台每 2.5 s 刷一次，提取好就自动可用。
//
// 回放页**只有这一个入口**：以前还有一个独立的上传页（拖拽 .dem / 浏览本机
// 文件 / 选采样），现在删了。少一个入口就少一套「这份数据是从哪来的」的说法。
// ═══════════════════════════════════════════════════════════════════════

let batchId = null;
let batchTimer = null;
//: 最近一次拿到的批次条目（队名 / 比分 / 雷达缩略图）。分享图要用它补上
//: `/api/stats` 里没有的比赛信息 —— 那里只有"队伍 A / 队伍 B"。
let batchItems = [];

function batchOptionText(it) {
  const bits = [it.name];
  if (it.team_a_score != null && it.team_b_score != null) {
    bits.push(`${it.team_a_score}:${it.team_b_score}`);
  }
  if (it.map) bits.push(it.map);
  if (!it.ready) bits.push(t('replay.batch.extracting'));
  return bits.join(' · ');
}

function renderBatchSelect(items) {
  const sel = $('batch-select');
  if (!sel) return;
  if (!items.length) { sel.classList.add('hidden'); return; }
  sel.classList.remove('hidden');
  const active = sel.dataset.activePath || '';
  sel.innerHTML = items.map((it) => {
    const label = batchOptionText(it);
    if (!it.ready) {
      return `<option value="" disabled>${label}</option>`;
    }
    const isActive = it.csz_path === active ? ' selected' : '';
    return `<option value="${it.csz_path}"${isActive}>${label}</option>`;
  }).join('');
}

async function refreshBatch() {
  if (!batchId) return;
  try {
    const res = await fetch(`/api/library/batch/${batchId}`);
    if (!res.ok) return;
    const data = await res.json();
    const items = data.items || [];
    batchItems = items;
    const hadReady = $('batch-select') && $('batch-select').dataset.activePath;
    renderBatchSelect(items);

    // 之前一场都没就绪，现在有就绪的了 → 自动开播，用户不用手动点
    if (!hadReady) {
      const first = items.find((it) => it.ready);
      if (first) {
        showLoading(false);
        showToast(t('replay.toast.extractDone', { name: first.name }));
        startLoad(first.csz_path);
      }
    }
    if (!items.some((it) => !it.ready)) {
      clearInterval(batchTimer);
      batchTimer = null;
    }
  } catch { /* 批次过期就安静放弃，回放本身不受影响 */ }
}

async function initBatch() {
  const id = new URLSearchParams(location.search).get('batch');
  if (!id) return false;
  batchId = id;
  try {
    const res = await fetch(`/api/library/batch/${batchId}`);
    if (!res.ok) throw new Error(t('replay.error.batchExpired'));
    const data = await res.json();
    const items = data.items || [];
    batchItems = items;
    renderBatchSelect(items);

    const sel = $('batch-select');
    if (sel) {
      sel.onchange = () => {
        const path = sel.value;
        if (!path) return;
        sel.dataset.activePath = path;
        startLoad(path);
      };
    }

    const first = items.find((it) => it.ready);
    if (first) {
      if (sel) sel.dataset.activePath = first.csz_path;
      showToast(items.length > 1
        ? t('replay.toast.batchMultiple', { n: items.length })
        : t('replay.toast.playing', { name: first.name }));
      startLoad(first.csz_path);
    } else {
      showLoading(true, t('replay.load.waiting', { n: items.length }));
    }
    if (items.some((it) => !it.ready)) {
      batchTimer = setInterval(refreshBatch, 2500);
    }
    return true;
  } catch (err) {
    showToast(t('replay.toast.batchFail', { msg: err.message }), true);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 数据就绪
// ═══════════════════════════════════════════════════════════════════════

function onDataReady(summary) {
  loadMatchData(summary);
  loadedSource = summary.source || '';
  activeRoundIdx = 0;

  // 换了一场比赛：上一场的统计和那个弹窗都不算数了。缓存清掉，下次开弹窗
  // 会重新拉 /api/stats（服务端那边换了会话自然也是新的一份）。
  statsPayload = null;
  // AI 统计是另一条请求：序号 +1 让还没回来的那条结果直接作废（服务端会自己
  // 算完并写进磁盘缓存，不白跑）
  aiStatsPayload = null;
  aiStatsError = null;
  aiStatsLoading = false;
  aiStatsPct = 0;
  aiStatsSeq++;
  closeStats();
  // 换了一场比赛：上一场的 AI 预测（含 3D 覆盖层）全部作废
  resetAiRound();

  // 没有批次也没有服务端会话时这一屏留着，它自己就是「去 Demo 库」的引导。
  // 注意顶栏**不在** #app 里：它是外壳，藏 #app 不会把导航一起藏掉。
  $('empty').classList.add('hidden');
  $('app').classList.remove('hidden');

  // 顶栏右侧那一排（地图 / 回合 / 采样 / 下载）只有真有数据时才成立
  $('tb-info').classList.remove('hidden');
  $('btn-download').classList.remove('hidden');

  $('map-name').textContent = summary.map || 'unknown';
  $('source-name').textContent = loadedSource;
  $('sampling-badge').textContent = summary.sampling || '';

  loadMap(summary.map).then(() => {
    fitCameraToMap();
    // 地图 BVH 是所有"打到墙上就停"的功能共用的：曳光、瞄准射线、以及**默认开着**
    // 的几何可见性眼睛。所以这里无条件在后台建（不挡界面），建好之前曳光只截断到
    // 玩家、几何可见性只按视锥 + 烟雾判（`ensureVisEvaluator()` 会在 BVH 到位后
    // 按对象身份自动重建求值器）。
    ensureMapBVH().catch(() => {});
  }).catch(() => {
    showToast(t('replay.toast.mapFail'), true);
  });

  // 回合下拉
  const sel = $('round-select');
  sel.disabled = false;
  sel.innerHTML = '';
  (summary.rounds || []).forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = t('replay.round.option', { n: i + 1, samples: r.sample_count });
    sel.appendChild(opt);
  });

  renderExtractionInspector(summary);

  showLoading(false);
  showToast(t('replay.toast.loaded', {
    rounds: summary.rounds.length,
    samples: fmtInt(summary.totals.sampled_ticks),
    smoke: summary.totals.smoke,
    fire: summary.totals.fire,
  }));

  // 第一回合
  setRound(0);
}

async function setRound(idx) {
  const seq = ++roundLoadSeq;
  const sel = $('round-select');
  if (sel) sel.value = String(idx);
  showLoading(true, t('replay.load.round', { n: idx + 1 }));
  try {
    const round = await loadRound(idx);
    if (seq !== roundLoadSeq) return;   // 快速连点时只认最后一次
    if (!round) return;
    activeRoundIdx = idx;
    onRoundChanged();
  } catch (err) {
    if (seq === roundLoadSeq) showToast(t('replay.toast.roundFail', { msg: err.message }), true);
  } finally {
    if (seq === roundLoadSeq) showLoading(false);
  }
}

function onRoundChanged() {
  // 换回合时卡片会重建、玩家状态整批换掉。第一人称立刻退出，免得镜头
  // 停在上一个回合的某个眼高上。这里不刷新：下面本来就会 updateAllVisuals。
  if (isFirstPerson()) exitFirstPerson({ restore: true, toast: false, refresh: false });
  setPlaying(false);
  // 步态（摆腿）是采样点的纯函数，换回合重算一次就行 —— 见 replay-core.js
  setGait(currentRound ? buildGait(currentRound) : null);
  // 开火事件同理：换回合重取一次（精确 tick 列表，见 collectFireEvents）
  roundFireEvents = collectFireEvents();
  updatePlayButton();
  updateRoundStats();
  renderRoundInspector();
  updatePlayerPanel();
  // `updatePlayerPanel()` 会把选中清掉（卡片是重建的）。上一回合正在跟谁，就把
  // 谁的卡片接回来，并且**重新飞到他新回合的出生点** —— 只靠位移累加的话，镜头
  // 会硬生生瞬移过去（人换回合是瞬移），重新飞一次是 0.7s 的平滑过渡，落点也
  // 更准（飞行目的地就是那一刻的真实坐标，见 focusCameraOnPlayer 的基准点）。
  if (followEnabled && getFollowPlayer() >= 0) {
    const follow = getFollowPlayer();
    togglePlayerDetail(follow, true);
    focusCameraOnPlayer(follow, { toast: false });
  }
  drawTimeline();
  // 统计弹窗里"正在回放的那一列"跟着走（只在这个弹窗开着的时候动 DOM）
  markActiveStatsColumn();
  // 换回合必须丢掉上一回合的烟雾体素 / 火焰格子缓存：
  // entity_id 会跨回合复用，不清理会把上一回合的几何画到新回合上。
  clearAllEntities();
  // AI 预测只对算过的那个回合有效（服务端也是按回合缓存），所以换回合就作废，
  // 开关开着的话顺手把新回合拉下来。
  resetAiRound();
  if (aiWanted()) ensureAiRound();
  updateAllVisuals();
  lastInspectorTick = -1;
  try {
    buildGrenadeTrajectories(getGrenadeTrajectories());
  } catch (err) {
    console.warn('[v5] 投掷物轨迹构建失败:', err);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 数据检查面板
// ═══════════════════════════════════════════════════════════════════════

function inspectRows(el, rows) {
  el.innerHTML = rows.map(([k, v, cls]) =>
    `<div class="inspect-row"><span>${k}</span><b class="${cls || ''}">${v}</b></div>`
  ).join('');
}

function renderExtractionInspector(s) {
  const m = s.meta || {};
  const rows = [
    [t('replay.inspector.format'), s.format],
    [t('replay.inspector.map'), s.map],
    ['tickrate', m.sidechannel_tickrate || m.tickrate || '—'],
    [t('replay.inspector.sampling'), s.sampling],
    [t('replay.inspector.samplesTotal'), `${fmtInt(m.sampled_ticks)} / ${fmtInt(m.total_ticks)}`],
    [t('replay.inspector.rounds'), s.rounds.length],
    [t('replay.inspector.skipped'), s.skipped_rounds.length
      ? t('replay.inspector.skippedN', {
        n: s.skipped_rounds.length,
        ids: s.skipped_rounds.map(x => x.id).join(','),
      }) : t('replay.inspector.none')],
    [t('replay.inspector.fireInstances'), s.totals.fire],
    [t('replay.inspector.smokeInstances'), s.totals.smoke],
    [t('replay.inspector.fireSmokeData'), m.has_fire_smoke
      ? t('replay.inspector.hasSidechannel') : t('replay.inspector.none')],
    [t('replay.inspector.groundContact'), m.has_ground_contact
      ? t('replay.inspector.hasV6') : t('replay.inspector.none')],
    [t('replay.inspector.crouch'), m.has_crouch
      ? t('replay.inspector.hasV7') : t('replay.inspector.noneV6')],
    [t('replay.inspector.spotted'), s.has_spotted === false
      ? t('replay.inspector.spottedNone') : t('replay.inspector.spottedYes')],
    [t('replay.inspector.mapBvh'), (() => {
      const st = getMapBVHStats();
      return st ? t('replay.inspector.mapBvhStats', {
        tris: fmtInt(st.triCount), ms: Math.round(st.ms),
      }) : t('replay.inspector.mapBvhMissing');
    })()],
  ];
  inspectRows($('inspector-extract'), rows);
}

function renderRoundInspector() {
  if (!currentRound) return;
  const r = currentRound;
  const dur = getRoundDuration();
  const smokeAlive = r.smoke.filter(s => currentTick() >= s.first_tick && currentTick() <= s.last_tick).length;
  const fireAlive = r.fire.filter(f => currentTick() >= f.first_tick && currentTick() <= f.last_tick).length;
  inspectRows($('inspector-round'), [
    [t('replay.inspector.roundId'), r.id],
    [t('replay.inspector.samples'), fmtInt(r.sample_count)],
    [t('replay.inspector.duration'), `${dur.toFixed(1)}s`],
    [t('replay.inspector.tickRange'), `${r.ticks[0]} – ${r.ticks[r.sample_count - 1]}`],
    [t('replay.inspector.currentSample'), `${currentSampleIdx} / ${currentTick()}`],
    [t('replay.inspector.smokeRound'), r.smoke.length],
    [t('replay.inspector.fireRound'), r.fire.length],
    [t('replay.inspector.nadeRound'), (() => {
      const n = nadeModelDebugStats();
      if (!n.total) return '0';
      const byKey = Object.entries(n.by_key).sort()
        .map(([k, c]) => `${k}×${c}`).join(' ');
      const tail = n.pending
        ? ` · ${t('replay.inspector.nadePending', { n: n.pending })}` : '';
      const none = n.no_key
        ? ` · ${t('replay.inspector.nadeNoKey', { n: n.no_key })}` : '';
      return `${n.total}${byKey ? `（${byKey}）` : ''}${tail}${none}`;
    })()],
    [t('replay.inspector.aliveSmokeFire'), `${smokeAlive} / ${fireAlive}`],
    [t('replay.inspector.missingSamples'), r.missing_samples.length],
  ]);
}

// ═══════════════════════════════════════════════════════════════════════
// 比赛统计弹窗（右侧栏最上面的「统计」按钮）
//
// 数据来自 /api/stats：服务端从**整场** demo_extract 结果算一次
// （见 visualizer/stats.py），浏览器这边只负责画。**不逐回合拉载荷自己加** ——
// 那等于为了几个数字把几十 MB 的逐帧数据再传一遍。
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
// 比赛统计弹窗（右侧栏最上面的「统计」按钮）
//
// 两个页签，共用一块内容区：
//
//   传统数据   /api/stats     服务端扫**整场** demo 算一次（visualizer/stats.py）
//   AI 数据    /api/ai/stats  服务端逐回合跑模型（冷启动几分钟，之后走磁盘缓存）
//
// 为什么 AI 那一页要单独请求：一整场 24 个回合 = 24 次特征构建 + 前向，实测
// 5~8 分钟。绝大多数人点「统计」只是想看 K/D/ADR，不该为此等五分钟。所以
// **切到 AI 页签才发这条请求**（需求原话），而且它走 NDJSON 流报进度。
//
// 两个页签都是"整场 + 每个回合"两个口径：
//   * 每回合的 K/D/伤害 → 老的四张矩阵（列 = 回合，一眼看完整场）
//   * 每回合的其余指标   → 一个回合选择器 + 一张表（七列再乘 24 列没法看）
// ═══════════════════════════════════════════════════════════════════════

/** 传统统计。换比赛时清掉（onDataReady），下次开弹窗重新拉。 */
let statsPayload = null;
/** AI 统计。同上，但**切到 AI 页签才拉**。 */
let aiStatsPayload = null;
let aiStatsError = null;
let aiStatsLoading = false;
let aiStatsPct = 0;
let aiStatsStage = '';
let aiStatsElapsed = 0;
/** 请求序号：换比赛 / 重开一次时，旧那条请求的结果直接丢掉。 */
let aiStatsSeq = 0;

/** 当前页签：'trad' | 'ai'。关掉弹窗再打开时保留（用户刚才看的是哪一页）。 */
let statsTab = 'trad';
// 单回合明细没有自己的"选中回合"了：它永远显示 `activeRoundIdx`，也就是正在
// 回放的那一回合。点明细上方的回合号 = 跳回放（和矩阵表头同一个动作），所以
// 一个回合只有一个真相，不存在"选择器说 5、回放停在 3"这种两套状态。

/** 每回合那四张矩阵：字段名 → 表头。顺序就是显示顺序。 */
const STATS_METRICS = [
  ['kills', t('replay.stats.metric.kills')],
  ['deaths', t('replay.stats.metric.deaths')],
  ['damage', t('replay.stats.metric.damage')],
  ['damage_taken', t('replay.stats.metric.damageTaken')],
];

// ── 传统统计的「细节」列 ───────────────────────────────────────────────
//
// 整场和单回合共用这一份定义：`cell(view)` 里的 `view` 是"这个人、这个口径"
// 的数，整场直接就是 /api/stats 的 player 行，单回合从 per_round 的定长数组
// 现拼一个同形状的对象（见 `tradView`）。所以两个口径只有一套渲染代码。

/** 「7/14 · 50%」：分子 / 分母 + 百分比。分母为 0 时显示 0/0。 */
function fracCell(a, b) {
  const pct = b ? Math.round((a / b) * 100) : 0;
  return `<td class="${a ? '' : 'zero'}">${a}/${b}`
    + `<span class="pct">${pct}%</span></td>`;
}

const TRAD_COLS = [
  {
    label: 'replay.stats.col.firstKill', tip: 'replay.stats.tip.firstKill',
    cell: (v) => `<td class="${v.first_kills ? 'sum' : 'zero'}">${v.first_kills || 0}</td>`,
  },
  {
    label: 'replay.stats.col.firstDeath', tip: 'replay.stats.tip.firstDeath',
    cell: (v) => `<td class="${v.first_deaths ? 'bad' : 'zero'}">${v.first_deaths || 0}</td>`,
  },
  {
    label: 'replay.stats.col.opener', tip: 'replay.stats.tip.opener',
    cell: (v) => fracCell(v.openers, v.enemy_kills + v.enemy_deaths),
  },
  {
    label: 'replay.stats.col.sideKill', tip: 'replay.stats.tip.sideKill',
    cell: (v) => fracCell(v.side_kills, v.enemy_kills),
  },
  {
    label: 'replay.stats.col.sideDeath', tip: 'replay.stats.tip.sideDeath',
    cell: (v) => fracCell(v.side_deaths, v.enemy_deaths),
  },
  {
    label: 'replay.stats.col.smokeKill', tip: 'replay.stats.tip.smokeKill',
    // 地图几何 / numba 不可用时服务端给 smoke_ok=false：那一列整块没有数据，
    // 显示 "—" 比显示一排 0 诚实（0 的意思是"真的没打穿过烟"）。
    cell: (v) => (statsPayload && statsPayload.smoke_ok === false
      ? `<td class="zero" title="${escHtml(t('replay.stats.smokeUnavailable'))}">—</td>`
      : fracCell(v.smoke_kills, v.enemy_kills)),
  },
];

// ── AI 统计的列 ────────────────────────────────────────────────────────

/** 样本数放在悬停里，不占格子。 */
function samplesTip(n) {
  return n == null ? '' : ` data-tip="${escHtml(t('replay.stats.ai.samples', { n }))}"`;
}

/** 相对量（-1~1）→ 带符号的百分比，正绿负红。 */
function impactCell(x, n) {
  if (x == null || n === 0) return '<td class="zero">—</td>';
  const cls = x > 0.0005 ? 'good' : x < -0.0005 ? 'bad' : '';
  const sign = x > 0 ? '+' : '';
  // 先按 0.1% 取整再拼，免得 -0.0004 显示成 "-0.0%"
  const v = (Math.round(x * 1000) / 10).toFixed(1);
  return `<td class="${cls}"${samplesTip(n)}>${sign}${v}%</td>`;
}

/** 概率（0~1）→ 百分数。 */
function pctCell(x, n) {
  if (x == null || n === 0) return '<td class="zero">—</td>';
  return `<td${samplesTip(n)}>${(x * 100).toFixed(1)}%</td>`;
}

const AI_COLS = [
  {
    label: 'replay.stats.ai.killImpact', tip: 'replay.stats.tip.killImpact',
    cell: (v) => impactCell(v.kill_impact, v.kill_n),
  },
  {
    label: 'replay.stats.ai.roundImpact', tip: 'replay.stats.tip.roundImpact',
    cell: (v) => impactCell(v.round_impact, v.rounds),
  },
  {
    label: 'replay.stats.ai.deathRisk', tip: 'replay.stats.tip.deathRisk',
    cell: (v) => pctCell(v.death_risk, v.risk_n),
  },
  {
    label: 'replay.stats.ai.killRate', tip: 'replay.stats.tip.killRate',
    cell: (v) => pctCell(v.kill_rate, v.kr_n),
  },
  {
    label: 'replay.stats.ai.duelScore', tip: 'replay.stats.tip.duelScore',
    cell: (v) => impactCell(v.duel_score, v.duel_n),
  },
];

// ── 开关 / 拉数据 ──────────────────────────────────────────────────────

function openStats() {
  $('stats-modal').classList.remove('hidden');
  if (statsPayload) {
    renderStats();
    return;
  }
  $('stats-body').innerHTML = `<div class="placeholder-text">${t('replay.stats.loading')}</div>`;
  loadStats();
}

function closeStats() {
  $('stats-modal').classList.add('hidden');
}

/** 切页签。AI 那一页在这里**才**触发预测（需求就是"点了才算"）。 */
function switchStatsTab(tab) {
  if (tab !== 'trad' && tab !== 'ai') return;
  statsTab = tab;
  for (const btn of document.querySelectorAll('.stats-tab')) {
    const on = btn.dataset.tab === tab;
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-selected', String(on));
  }
  renderStats();
  if (tab === 'ai' && !aiStatsPayload && !aiStatsLoading && !aiStatsError) {
    loadAiStats();
  }
}

async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    statsPayload = data;
    renderStats();
  } catch (err) {
    $('stats-body').innerHTML =
      `<div class="placeholder-text">${t('replay.stats.loadFail', { msg: escHtml(err.message) })}</div>`;
  }
}

/**
 * 拉一整场的 AI 统计。第一次会跑模型（几分钟），服务端边走边报进度。
 *
 * 请求发出去之后**不随弹窗关闭而取消**：服务端已经在算了，取消只会让下次
 * 从头再来（进度条回到 0）。只有换比赛（onDataReady）才丢弃旧结果。
 */
async function loadAiStats(force = false) {
  if (aiStatsLoading) return;
  const seq = ++aiStatsSeq;
  aiStatsLoading = true;
  aiStatsError = null;
  aiStatsPct = 0;
  aiStatsStage = t('replay.stats.ai.preparing');
  aiStatsElapsed = 0;
  renderStats();
  try {
    const qs = new URLSearchParams({ progress: '1' });
    if (force) qs.set('refresh', '1');
    const payload = await streamJson(`/api/ai/stats?${qs}`, (ev) => {
      if (seq !== aiStatsSeq) return;
      if (typeof ev.pct === 'number') {
        aiStatsPct = Math.max(aiStatsPct, Math.min(1, ev.pct));
      }
      if (typeof ev.stage === 'string' && ev.stage) aiStatsStage = ev.stage;
      if (typeof ev.elapsed === 'number') aiStatsElapsed = ev.elapsed;
      paintStatsProgress();
    });
    if (seq !== aiStatsSeq) return;
    aiStatsPayload = payload;
    aiStatsPct = 1;
    aiStatsElapsed = payload.elapsed || aiStatsElapsed;
  } catch (err) {
    if (seq !== aiStatsSeq) return;
    aiStatsError = err;
  } finally {
    if (seq === aiStatsSeq) {
      aiStatsLoading = false;
      renderStats();
    }
  }
}

/** 进度条：只动那几个节点，不重建整块（每 80 ms 一个事件，重建会闪）。 */
function paintStatsProgress() {
  const fill = $('stats-progress-fill');
  const stage = $('stats-progress-stage');
  const pct = $('stats-progress-pct');
  if (fill) fill.style.width = `${(aiStatsPct * 100).toFixed(1)}%`;
  if (stage) stage.textContent = aiStatsStage;
  if (pct) {
    pct.textContent = `${Math.round(aiStatsPct * 100)}%`
      + (aiStatsElapsed ? ` · ${aiStatsElapsed.toFixed(0)} s` : '');
  }
}

// ── 渲染 ──────────────────────────────────────────────────────────────

/** 队伍顺序；队内按击杀降序（并列看 ADR，再看下标）。所有表共用同一顺序。 */
function statsTeamOrder(s) {
  const players = s.players || [];
  return (s.teams || []).map((team) => ({
    ...team,
    rows: (team.players || []).map((i) => players[i]).filter(Boolean)
      .sort((a, b) => b.kills - a.kills || b.adr - a.adr || a.idx - b.idx),
  }));
}

/** 队伍的着色看它**开场**站哪边（半场会换边，所以只是给个颜色，不是身份）。 */
function statsTeamClass(team) {
  return team.side0 === 'CT' ? 'ct' : team.side0 === 'T' ? 't' : '';
}

function statsTeamHead(team, colspan) {
  const cls = statsTeamClass(team);
  const side = team.side0 === 'CT' ? t('replay.stats.side.ctStart')
    : team.side0 === 'T' ? t('replay.stats.side.tStart') : t('replay.stats.side.unknown');
  // 「队伍 A / B」是服务端 stats.py 给的 label（数据），显示时翻一层：英文界面
  // 不该在这一行冒中文。认不出的 label 原样显示。
  const label = team.label === '队伍 A' ? t('replay.stats.teamA')
    : team.label === '队伍 B' ? t('replay.stats.teamB') : team.label;
  return `<tr class="team-row ${cls}"><td colspan="${colspan}">${escHtml(label)}`
    + `<span class="dim"> · ${side} · ${team.score} ${t('replay.stats.points')}</span></td></tr>`;
}

function statsNameCell(p, cls) {
  const name = p.name || `P${p.idx + 1}`;
  return `<td class="pname ${cls}" title="${escHtml(name)}">${escHtml(name)}</td>`;
}

function summaryTable(teams) {
  return teams.map((team) => {
    const cls = statsTeamClass(team);
    const rows = team.rows.map((p) => {
      const diff = p.kills - p.deaths;
      const dcls = diff > 0 ? 'good' : diff < 0 ? 'bad' : '';
      return `<tr>
        ${statsNameCell(p, cls)}
        <td>${p.kills}</td><td>${p.deaths}</td><td>${p.assists}</td>
        <td class="${dcls}">${diff > 0 ? '+' : ''}${diff}</td>
        <td>${p.kd}</td><td>${p.adr}</td><td>${p.kpr}</td>
        <td>${p.hs_pct}%</td><td>${p.damage}</td><td>${p.damage_taken}</td>
      </tr>`;
    }).join('');
    return statsTeamHead(team, 11) + rows;
  }).join('');
}

/** 一张矩阵：行 = 选手，列 = 回合，最右是合计。 */
function matrixTable(s, teams, key) {
  const label = (STATS_METRICS.find(([k]) => k === key) || [key, key])[1];
  const rounds = s.per_round || [];
  const head = rounds.map((r) => {
    const w = r.winner ? ` · ${r.winner} ${t('replay.stats.won')}` : '';
    return `<th class="round-col" data-round-col="${r.idx}"`
      + ` title="${t('replay.stats.roundCol.title', { n: r.idx + 1, w })}">${r.idx + 1}</th>`;
  }).join('');

  const body = teams.map((team) => {
    const cls = statsTeamClass(team);
    const rows = team.rows.map((p) => {
      let total = 0;
      const cells = rounds.map((r) => {
        const v = (r[key] || [])[p.idx] || 0;
        total += v;
        return `<td class="${v ? '' : 'zero'}">${v}</td>`;
      }).join('');
      const name = p.name || `P${p.idx + 1}`;
      return `<tr><td class="rname ${cls}" title="${escHtml(name)}">`
        + `${escHtml(name)}</td>${cells}<td class="sum">${total}</td></tr>`;
    }).join('');
    return statsTeamHead(team, rounds.length + 2) + rows;
  }).join('');

  return `
    <div class="stats-matrix-wrap">
      <div class="stats-matrix-head">${label}<span class="dim">${t('replay.stats.matrix.sub')}</span></div>
      <div class="stats-matrix-scroll">
        <table class="stats-table stats-matrix">
          <thead><tr><th>${t('replay.stats.player')}</th>${head}<th>${t('replay.stats.total')}</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    </div>`;
}

/** 单回合视图：把 per_round 的定长数组拼成和 player 行同形状的对象。 */
function tradView(p, perRound, roundIdx) {
  if (roundIdx < 0) return p;
  const r = (perRound || [])[roundIdx] || {};
  const at = (k) => ((r[k] || [])[p.idx] || 0);
  return {
    idx: p.idx,
    first_kills: at('first_kills'),
    first_deaths: at('first_deaths'),
    openers: at('openers'),
    enemy_kills: at('enemy_kills'),
    enemy_deaths: at('enemy_deaths'),
    side_kills: at('side_kills'),
    side_deaths: at('side_deaths'),
    smoke_kills: at('smoke_kills'),
  };
}

/** 细节表（首杀 / 先手 / 侧背身 / 穿烟）。`roundIdx < 0` = 整场。 */
function detailTable(s, teams, roundIdx) {
  const head = TRAD_COLS.map((c) => tipTh(t(c.label), c.tip)).join('');
  const body = teams.map((team) => {
    const cls = statsTeamClass(team);
    const rows = team.rows.map((p) => {
      const view = tradView(p, s.per_round, roundIdx);
      return `<tr>${statsNameCell(p, cls)}`
        + `${TRAD_COLS.map((c) => c.cell(view)).join('')}</tr>`;
    }).join('');
    return statsTeamHead(team, TRAD_COLS.length + 1) + rows;
  }).join('');
  return `<table class="stats-table stats-detail">
      <thead><tr><th>${t('replay.stats.player')}</th>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table>`;
}

/** 一排回合号：和矩阵表头的回合号是同一个东西（点一下 = 跳到那一回合）。
 *
 * 以前这里是个 `<select>`（还有一项「跟当前回放」），和上面四张矩阵的列头是两套
 * 交互；现在统一成"一堆能点的数字"：点哪个，回放就跳到哪，矩阵的高亮列和下面这
 * 张明细表一起换 —— 全弹窗只有 `activeRoundIdx` 一个真相。
 */
function roundChips(rounds) {
  const chips = rounds.map((r) => {
    const w = r.winner ? ` · ${r.winner} ${t('replay.stats.won')}` : '';
    return `<button type="button" class="round-chip" data-round-col="${r.idx}"`
      + ` title="${t('replay.stats.roundCol.title', { n: r.idx + 1, w })}">${r.idx + 1}</button>`;
  }).join('');
  return `<div class="round-chips">${chips}</div>`;
}

function roundCaption(r) {
  if (!r) return '';
  const w = r.winner ? ` · ${r.winner} ${t('replay.stats.won')}` : '';
  return `<div class="stats-note">${t('replay.stats.pickRound', { n: r.idx + 1 })}`
    + `<span class="dim">${w} · ${r.duration}s</span></div>`;
}

function tradBody(s, teams) {
  // 平台把赛前热身也录进 demo 时（5E 就是），热身击杀曾经算进第一回合，
  // K/D 因此偏高。服务端现在把它们剔掉了，这里明说剔掉了多少 —— 否则用户
  // 只会看到"这里的 K/D 和对战平台对不上"，而不知道该信哪个。
  const warm = s.warmup_kills
    ? `<div class="stats-note">${t('replay.stats.warmupNote', { n: s.warmup_kills })}</div>`
    : '';
  return `
    <div class="stats-block">
      <div class="section-title">${t('replay.stats.matchTotal')}</div>
      <table class="stats-table stats-summary">
        <thead><tr>
          <th>${t('replay.stats.player')}</th><th>K</th><th>D</th><th>A</th><th>+/-</th><th>K/D</th>
          <th>ADR</th><th>KPR</th><th>${t('replay.stats.hs')}</th><th>${t('replay.stats.damage')}</th><th>${t('replay.stats.damageTaken')}</th>
        </tr></thead>
        <tbody>${summaryTable(teams)}</tbody>
      </table>
      ${warm}
      <div class="stats-note">${t('replay.stats.note')}</div>
    </div>
    <div class="stats-block">
      <div class="section-title">${t('replay.stats.extra')}<span class="hint">${t('replay.stats.extraHint')}</span></div>
      ${detailTable(s, teams, -1)}
      <div class="stats-note">${t('replay.stats.extraNote')}</div>
    </div>
    <div class="stats-block">
      <div class="section-title">${t('replay.stats.rounds')}<span class="hint">${t('replay.stats.roundsHint')}</span></div>
      <div class="stats-grid">
        ${STATS_METRICS.map(([key]) => matrixTable(s, teams, key)).join('')}
      </div>
    </div>
    <div class="stats-block">
      <div class="stats-detail-head">
        <div class="section-title">${t('replay.stats.roundDetail')}<span class="hint">${t('replay.stats.detailHint')}</span></div>
        ${roundChips(s.per_round || [])}
      </div>
      <div id="stats-round-detail"></div>
    </div>`;
}

function aiBody(s, teams) {
  if (aiStatsLoading) {
    return `<div class="stats-block">
        <div class="section-title">${t('replay.stats.ai.analyzing')}</div>
        <div class="stats-progress"><div id="stats-progress-fill" class="stats-progress-fill"></div></div>
        <div class="stats-progress-text">
          <span class="stage" id="stats-progress-stage">${escHtml(aiStatsStage)}</span>
          <span id="stats-progress-pct"></span>
        </div>
        <div class="stats-note">${t('replay.stats.ai.progressNote')}</div>
      </div>`;
  }
  if (aiStatsError) {
    return `<div class="stats-block">
        <div class="section-title">${t('replay.stats.ai.title')}</div>
        <div class="stats-empty">${t('replay.stats.ai.fail', { msg: escHtml(aiStatsError.message) })}</div>
        <div class="stats-note"><button id="stats-ai-retry" class="btn-glass">${t('replay.stats.ai.retry')}</button></div>
      </div>`;
  }
  if (!aiStatsPayload) {
    return `<div class="stats-block">
        <div class="section-title">${t('replay.stats.ai.title')}</div>
        <div class="stats-empty">${t('replay.stats.ai.idle')}</div>
      </div>`;
  }
  const p = aiStatsPayload;
  const rows = (p.players || []).length ? p.players : [];
  const head = AI_COLS.map((c) => tipTh(t(c.label), c.tip)).join('');
  const body = teams.map((team) => {
    const cls = statsTeamClass(team);
    const tr = team.rows.map((pl) => {
      const row = rows[pl.idx] || {};
      return `<tr>${statsNameCell(pl, cls)}${AI_COLS.map((c) => c.cell(row)).join('')}</tr>`;
    }).join('');
    return statsTeamHead(team, AI_COLS.length + 1) + tr;
  }).join('');

  const meta = [
    t('replay.stats.ai.metaRounds', { n: p.rounds_total || 0 }),
    p.cached ? t('replay.stats.ai.fromCache')
      : t('replay.stats.ai.fresh', { s: (p.elapsed || 0).toFixed(0) }),
  ];
  if ((p.errors || []).length) meta.push(t('replay.stats.ai.metaErrors', { n: p.errors.length }));

  return `
    <div class="stats-block">
      <div class="section-title">${t('replay.stats.ai.title')}<span class="hint">${t('replay.stats.ai.sub')}</span></div>
      <table class="stats-table stats-ai">
        <thead><tr><th>${t('replay.stats.player')}</th>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
      <div class="stats-note">${meta.join(' · ')}</div>
    </div>
    <div class="stats-block">
      <div class="stats-detail-head">
        <div class="section-title">${t('replay.stats.roundDetail')}<span class="hint">${t('replay.stats.detailHint')}</span></div>
        ${roundChips(s.per_round || [])}
      </div>
      <div id="stats-ai-round-detail"></div>
    </div>`;
}

function renderStats() {
  const s = statsPayload;
  if (!s) return;
  const teams = statsTeamOrder(s);
  const nPlayers = (s.players || []).length;
  $('stats-subtitle').textContent = t('replay.stats.subtitle', {
    map: s.map || '—',
    rounds: s.rounds || 0,
    players: nPlayers,
  });

  if (!nPlayers) {
    $('stats-body').innerHTML = `<div class="placeholder-text">${t('replay.stats.empty')}</div>`;
    return;
  }

  $('stats-body').innerHTML = statsTab === 'trad' ? tradBody(s, teams) : aiBody(s, teams);
  if (statsTab === 'ai') paintAiRoundDetail();
  markActiveStatsColumn();
}

/** 单回合明细（传统）：永远显示正在回放的那一回合。 */
function paintRoundDetail() {
  const s = statsPayload;
  const host = $('stats-round-detail');
  if (!s || !host) return;
  const rounds = s.per_round || [];
  const idx = activeRoundIdx;
  host.innerHTML = roundCaption(rounds[idx]) + detailTable(s, statsTeamOrder(s), idx);
}

/** 单回合明细（AI）。`per_round` 在 AI 载荷里是"每个选手的每回合明细"。 */
function paintAiRoundDetail() {
  const s = statsPayload;
  const p = aiStatsPayload;
  const host = $('stats-ai-round-detail');
  if (!s || !p || !host) return;
  const rounds = s.per_round || [];
  const idx = activeRoundIdx;
  const head = AI_COLS.map((c) => tipTh(t(c.label), c.tip)).join('');
  const body = statsTeamOrder(s).map((team) => {
    const cls = statsTeamClass(team);
    const rows = team.rows.map((pl) => {
      const row = (p.players || [])[pl.idx] || {};
      const cell = ((row.per_round || [])[idx]) || {};
      return `<tr>${statsNameCell(pl, cls)}${AI_COLS.map((c) => c.cell(cell)).join('')}</tr>`;
    }).join('');
    return statsTeamHead(team, AI_COLS.length + 1) + rows;
  }).join('');
  host.innerHTML = roundCaption(rounds[idx])
    + `<table class="stats-table stats-ai">
        <thead><tr><th>${t('replay.stats.player')}</th>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>`;
}

/** 把"当前回合"在弹窗里标出来。弹窗开着时换回合也会跟着动。 */
function markActiveStatsColumn() {
  const modal = $('stats-modal');
  if (!modal || modal.classList.contains('hidden')) return;
  paintActiveRound(activeRoundIdx);
  // 单回合明细永远跟着当前回放走
  if (statsTab === 'trad') paintRoundDetail();
  else paintAiRoundDetail();
}

/** 只挪高亮、不动表格内容：换回合是异步的（round 要重新取），点下去先把高亮
 *  挪过去，不然那半秒看着像"点了没反应"。 */
function paintActiveRound(idx) {
  $('stats-modal')?.querySelectorAll('[data-round-col]').forEach((el) => {
    el.classList.toggle('active', Number(el.dataset.roundCol) === idx);
  });
}

// ── 分享图（单场）──────────────────────────────────────────────────────────
//
// 数据全部来自已经在页面里的两份载荷：`statsPayload`（传统）和
// `aiStatsPayload`（AI，点了页签才有）。队名 / 真实比分 / 雷达缩略图 / 平台
// 来源只有 Demo 库那一行才有，所以从批次条目里补 —— 不在批次里（直接开会话，
// 或者拖一个 .csz 进来）时退回"队伍 A / 队伍 B"、并且**不写来源**：那种情况下
// 页面里根本没有平台信息（`loadedSource` 是文件名，不是平台）。
//
// 画法在 /static/js/share-card.js，和玩家页共用；那边不管 url 布局，只管
// 拿 spec 画图。

function activeBatchItem() {
  const sel = $('batch-select');
  const path = sel && sel.dataset.activePath;
  if (!path) return null;
  return (batchItems || []).find((it) => it.csz_path === path) || null;
}

function shareMatchSpec() {
  const s = statsPayload;
  if (!s) return null;
  const item = activeBatchItem();
  const aiByIdx = new Map();
  for (const row of (aiStatsPayload && aiStatsPayload.players) || []) {
    aiByIdx.set(Number(row.idx), row);
  }
  const byIdx = new Map((s.players || []).map((p) => [Number(p.idx), p]));

  const teams = (s.teams || []).map((team) => ({
    idx: team.idx,
    // 服务端的 label 是中文的"队伍 A / B"，只用它的 idx 决定取哪个真实队名
    label: item ? (team.idx === 0 ? item.team_a_name : item.team_b_name) : '',
    score: team.score,
    side0: team.side0,
    players: (team.players || []).map((i) => {
      const p = byIdx.get(Number(i)) || {};
      return {
        name: p.name,
        kills: p.kills,
        deaths: p.deaths,
        assists: p.assists,
        headshots: p.headshots,
        kd: p.kd,
        adr: p.adr,
        ai: aiByIdx.get(Number(i)) || null,
      };
    }),
  }));

  const rounds = s.rounds || 0;
  return {
    kind: 'match',
    teams,
    ai: Boolean(aiStatsPayload),
    map: s.map || (item && item.map) || '',
    rounds,
    source: item ? item.source : '',   // 平台代码（5E / 完美 / 官匹），图里翻成显示名
    date: item ? item.date : '',
    thumb: item ? item.thumb : null,
    filename: (item && item.name) || s.map || 'match',
  };
}

async function shareMatchImage() {
  const spec = shareMatchSpec();
  if (!spec) {
    showToast(t('replay.stats.empty'), true);
    return;
  }
  const btn = $('btn-stats-share');
  if (btn) btn.disabled = true;
  try {
    const out = await downloadCard(spec, t);
    showToast(t('common.share.done', { name: out.filename }));
  } catch (err) {
    showToast(t('common.share.fail', { msg: err.message }), true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function currentTick() {
  if (!currentRound) return -1;
  return currentRound.ticks[Math.min(currentSampleIdx, currentRound.ticks.length - 1)];
}

// ═══════════════════════════════════════════════════════════════════════
// 玩家面板
// ═══════════════════════════════════════════════════════════════════════

let expandedPlayerIdx = -1;

/**
 * 「跟随选中玩家」开关（底部 `toggle-cam-follow`，默认开）。
 * 关掉 = 立刻停掉当前跟随，并且之后选中玩家只飞过去、不跟。
 */
let followEnabled = true;

// ═══════════════════════════════════════════════════════════════════════
// 第一人称摄像机（左侧卡片 → 摄像机 → 第一人称）
//
// 这里只做「进入 / 退出 + HUD」。镜头本身由 scene.js 的 `cameraMode = 'first'`
// 每帧按 focusedPlayerIdx 的 yaw / pitch / eye_height 摆好；本文件负责：
//   * 进入时锁死 OrbitControls，**存下进入前的 orbit 机位**，退出时原样还原；
//   * 把选中卡片 / 下拉框 / HUD 三方状态同步；
//   * 每帧把该选手的血量、护甲、头盔、装备写进 HUD。
//
// `firstPersonIdx >= 0` 是唯一可信的「第一人称模式开着」标志。不要用
// `cameraMode === 'first'` 判断：测试 / 调试钩子也会直接调 setCameraMode('first')，
// 那种情况下没有 HUD，也不该改卡片状态。
// ═══════════════════════════════════════════════════════════════════════

/** 正在以第一人称观察的玩家下标；-1 = 没开。 */
let firstPersonIdx = -1;

/** 进入第一人称之前的 orbit 机位（退出时恢复，保证「退出后能正常移动」）。 */
let fpSavedView = null;

function isFirstPerson() { return firstPersonIdx >= 0; }

/** 把所有卡片上的摄像机下拉框同步成当前状态。 */
function syncCameraSelects() {
  for (let i = 0; i < 10; i++) {
    const sel = $(`player-cam-${i}`);
    if (sel) sel.value = i === firstPersonIdx ? 'first' : 'follow';
  }
  for (let i = 0; i < 10; i++) {
    const card = $(`player-card-${i}`);
    if (card) card.classList.toggle('fp-source', i === firstPersonIdx);
  }
}

function updatePlayerPanel() {
  const list = $('player-list');
  if (!matchData || !currentRound) { list.innerHTML = '<div class="placeholder-text">—</div>'; return; }
  const names = matchData.players || [];
  // 头顶「选手 ID」写名字、不是槽位号；名字跟着对局走，换局刷一次就够
  setPlayerIdNames(names.map((p) => (p && p.name) || ''));
  const teams = currentRound.teams || [];
  list.innerHTML = '';
  expandedPlayerIdx = -1;

  const groups = { CT: [], T: [] };
  for (let i = 0; i < 10; i++) groups[teams[i] === 'T' ? 'T' : 'CT'].push(i);

  for (const team of ['CT', 'T']) {
    const grp = document.createElement('div');
    grp.className = `team-group ${team === 'CT' ? 'ct' : 't'}`;
    grp.innerHTML = `
      <div class="team-header">
        <span class="team-name">${team === 'CT' ? t('replay.team.ct') : t('replay.team.t')}</span>
        <span class="team-count">${t('replay.team.count', { n: groups[team].length })}</span>
      </div>`;
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
        </div>`;
      card.addEventListener('click', () => {
        // 第一人称是「锁定」模式：点卡片不再换目标 / 收卡片，避免镜头被 UI
        // 带着跳走。要换人就先按 Esc / HUD 的退出按钮。
        if (isFirstPerson()) {
          showToast(t('replay.toast.fpLocked'), true);
          return;
        }
        if (togglePlayerDetail(i)) focusCameraOnPlayer(i);
      });
      grp.appendChild(card);

      const detail = document.createElement('div');
      detail.className = 'player-detail hidden';
      detail.id = `player-detail-${i}`;
      detail.innerHTML = `
        <div class="pd-row pd-cam-row">
          <span class="pd-label">${t('replay.player.camera')}</span>
          <select class="pd-cam" id="player-cam-${i}"
                  title="${t('replay.player.cam.title')}">
            <option value="follow">${t('replay.player.cam.follow')}</option>
            <option value="first">${t('replay.player.cam.first')}</option>
          </select>
        </div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.hp')}</span><b class="pd-hp">—</b></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.armor')}</span><b class="pd-armor">—</b></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.weapon')}</span><b class="pd-weapon">—</b></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.equipment')}</span><span class="pd-inv">—</span></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.status')}</span><span class="pd-status">—</span></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.place')}</span><span class="pd-place">—</span></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.speed')}</span><b class="pd-v">—</b></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.groundRow')}</span><span class="pd-ground">—</span></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.crouchRow')}</span><span class="pd-crouch">—</span></div>
        <div class="pd-row"><span class="pd-label">${t('replay.player.eventsRow')}</span><span class="pd-events">—</span></div>`;
      detail.querySelector('.pd-cam').addEventListener('change', (e) => {
        // 详情是卡片的兄弟节点，这里其实不会冒泡到卡片；停掉只是为了以后
        // 把下拉挪进卡片内部时行为不变。
        e.stopPropagation();
        if (e.target.value === 'first') enterFirstPerson(i);
        else exitFirstPerson({ toast: i === firstPersonIdx });
        e.target.blur();   // 交还键盘焦点，退出后 WASD 立刻可用
      });
      grp.appendChild(detail);
    }
    list.appendChild(grp);
  }
  syncCameraSelects();
}

function togglePlayerDetail(i, forceOpen = false) {
  const detail = $(`player-detail-${i}`);
  if (!detail) return false;
  const willOpen = forceOpen || detail.classList.contains('hidden');
  for (let p = 0; p < 10; p++) {
    $(`player-detail-${p}`)?.classList.add('hidden');
    $(`player-card-${p}`)?.classList.remove('expanded');
  }
  if (willOpen) {
    detail.classList.remove('hidden');
    $(`player-card-${i}`)?.classList.add('expanded');
    expandedPlayerIdx = i;
  } else {
    // 取消选中 = 也把相机跟随一起停掉（跟随的语义是「跟着选中的那个人」）
    expandedPlayerIdx = -1;
    setFollowPlayer(-1);
  }
  return willOpen;
}

/**
 * 取消选中：关掉展开的卡片，并停掉相机跟随。WASD 手动接管镜头时走这里 ——
 * 「跟着某个人」和「我在推机位」同时成立的话，两边会抢同一个 controls.target。
 */
function deselectPlayer() {
  if (expandedPlayerIdx >= 0) togglePlayerDetail(expandedPlayerIdx);
  else setFollowPlayer(-1);   // 没有选中但还在跟（正常不会发生）：直接停
}

function pickPlayerAt(clientX, clientY) {
  if (!currentRound || !matchData) return;
  const rect = $('three-canvas').getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ndc = new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1
  );
  raycaster.setFromCamera(ndc, camera);
  raycaster.far = Infinity;
  raycaster.near = 0.1;

  const targets = [];
  scene.traverse((o) => {
    if (!o.visible) return;
    if (o.name && /^(player|dead-x)-\d+$/.test(o.name)) targets.push(o);
  });
  if (!targets.length) return;
  const hits = raycaster.intersectObjects(targets, true);
  if (!hits.length) return;
  let node = hits[0].object;
  while (node) {
    const m = node.name && node.name.match(/^(player|dead-x)-(\d+)$/);
    if (m) {
      const idx = parseInt(m[2], 10);
      togglePlayerDetail(idx, true);
      focusCameraOnPlayer(idx);
      return;
    }
    node = node.parent;
  }
}

/**
 * 进入某个选手的第一人称视角。
 *
 * 本函数不碰相机飞行 / 跟随：它只保存当前 orbit 机位，然后把镜头交给
 * scene.js 的 first 模式（每帧按这个人的 yaw/pitch/眼高重算）。
 * 因此进入后鼠标、滚轮、WASD、右键平移全部无效 —— controls.enabled = false。
 */
function enterFirstPerson(i) {
  if (!currentRound) {
    showToast(t('replay.toast.noData'), true);
    return false;
  }
  if (i < 0 || i >= 10) return false;
  const st = getPlayerState(i, currentSampleIdx);
  if (!st || !st.alive) {
    showToast(t('replay.toast.fpDead'), true);
    syncCameraSelects();
    return false;
  }
  if (firstPersonIdx === i && cameraMode === 'first') return true;

  // 选中这张卡片（只展开详情，不飞镜头 —— 第一人称下一帧就会接管相机）。
  // 卡片详情里的下拉框和这里必须指向同一个人。
  togglePlayerDetail(i, true);

  if (!isFirstPerson()) {
    // 存 orbit 机位。退出时原样还原，用户回到"进入之前那样"的相机位置。
    fpSavedView = {
      pos: camera.position.clone(),
      target: controls.target.clone(),
    };
  }

  firstPersonIdx = i;
  setCameraMode('first');
  setFocusedPlayer(i);
  $('viewport').classList.add('fp-active');
  $('fp-hud').classList.remove('hidden');
  $('fp-hud').setAttribute('aria-hidden', 'false');
  syncCameraSelects();
  updateAllVisuals();            // 立刻隐藏自己的模型并填 HUD

  const name = playerName(i) || `P${i + 1}`;
  showToast(t('replay.toast.fpEnter', { name }));
  return true;
}

/**
 * 退出第一人称，恢复进入之前的 orbit 机位（默认）。
 *
 * @param {{restore?: boolean, toast?: boolean, refresh?: boolean, revealModel?: boolean}} [opts]
 *   restore=false  —— 不还原机位（例如马上要飞到烟/火上，或换回合）。
 *   refresh=false  —— 不立即 updateAllVisuals（调用方本来就要刷新时省一趟）。
 *   revealModel=false —— 不把观察对象的模型重新显示（阵亡退出时用：updatePlayers()
 *     已经把尸体藏起来了，这里再设回 visible 会闪一帧）。
 */
function exitFirstPerson(opts = {}) {
  const { restore = true, toast = false, refresh = true, revealModel = true } = opts;
  if (!isFirstPerson()) return false;
  const idx = firstPersonIdx;
  firstPersonIdx = -1;           // 先清状态：下面 updateAllVisuals 不会再次触发退出

  setCameraMode('orbit');        // 同时 controls.enabled = true / 取消飞行动画
  setFocusedPlayer(-1);
  if (restore && fpSavedView) {
    // 关键：恢复的是**进入前**的 camera.position + controls.target，不是
    // "当前玩家旁边"。这样退出后的拖拽 / 滚轮 / WASD 与进入前完全一致。
    camera.position.copy(fpSavedView.pos);
    controls.target.copy(fpSavedView.target);
    controls.update();
  }
  fpSavedView = null;

  // 隐藏自己时连轨迹一起藏了（setPlayerModelVisible），退出要还回来；
  // 轨迹总开关关着的时候保持关着，不能因为退出就冒出一条线。
  if (revealModel) {
    setPlayerModelVisible(idx, true);
    if ($('toggle-trails')) setTrailsVisible($('toggle-trails').checked);
  }

  $('viewport').classList.remove('fp-active');
  $('fp-hud').classList.add('hidden');
  $('fp-hud').setAttribute('aria-hidden', 'true');
  syncCameraSelects();
  if (refresh) updateAllVisuals();

  if (toast) {
    const name = playerName(idx) || `P${idx + 1}`;
    showToast(t('replay.toast.fpExit', { name }));
  }
  return true;
}

/**
 * 每帧把观察对象的状态写进 HUD。
 * 只在 `firstPersonIdx >= 0` 时调用；`st` 可能是 null（数据正在切回合）。
 */
function updateFirstPersonHud(st) {
  if (!isFirstPerson()) return;
  const team = (currentRound && currentRound.teams && currentRound.teams[firstPersonIdx]) || '?';
  const teamEl = $('fp-team');
  teamEl.textContent = (team === 'T' || team === 'CT') ? team : '—';
  teamEl.classList.toggle('t', team === 'T');
  $('fp-name').textContent = playerName(firstPersonIdx) || `P${firstPersonIdx + 1}`;

  const hpNum = $('fp-hp');
  const hpFill = $('fp-hpfill');
  if (!st) {
    hpNum.textContent = '—';
    hpNum.classList.remove('low');
    hpFill.style.width = '0%';
    hpFill.classList.remove('low');
    $('fp-armor').textContent = '—';
    $('fp-helmet').textContent = '—';
    $('fp-weapon').textContent = '—';
    $('fp-inventory').textContent = '—';
    return;
  }

  const hp = Math.max(0, Math.min(100, Math.round(st.hp || 0)));
  hpNum.textContent = st.alive ? String(hp) : '✕';
  hpNum.classList.toggle('low', hp <= 30);
  hpFill.style.width = `${st.alive ? hp : 0}%`;
  hpFill.classList.toggle('low', hp <= 30);

  $('fp-armor').textContent = st.alive ? String(Math.round(st.armor || 0)) : '—';
  $('fp-helmet').textContent = st.alive
    ? (st.helmet ? t('replay.fp.has') : t('replay.fp.none')) : '—';

  const cur = shortWeaponName(st.weapon);
  $('fp-weapon').textContent = st.alive ? cur : '—';

  // inventory 里通常也包含手里那把，所以扣掉当前武器、再去重；
  // 死亡时数据是最后一帧的残留值，直接按空处理。
  const seen = new Set();
  const items = [];
  for (const id of (st.inventory || [])) {
    const n = shortWeaponName(id);
    if (!n || n === '—' || n === cur || seen.has(n)) continue;
    seen.add(n);
    items.push(n);
  }
  $('fp-inventory').textContent = st.alive
    ? (items.length ? items.join(' · ') : t('replay.fp.empty')) : '—';
}

/**
 * 选中一个玩家：把镜头飞到他旁边，然后**跟着他走**（见 scene.js 的相机跟随）。
 * 跟随只做平移，所以到位之后仍然可以拖鼠标绕着他转 / 滚轮缩放。
 *
 * 基准点用**这次飞行的目的地**（点击那一刻他的位置）灌进去：跟随是拿位移累加的，
 * 基准点要是从下一帧现取，这一帧的位移就丢了，镜头会永远差一个帧的量。
 */
function focusCameraOnPlayer(playerIdx, { toast = true } = {}) {
  // 第一人称是锁定模式；任何"飞到某人身旁"的动作都必须先退出它。
  if (isFirstPerson()) exitFirstPerson({ restore: false, refresh: false });
  if (cameraMode !== 'orbit') setCameraMode('orbit');
  const st = getPlayerState(playerIdx, currentSampleIdx);
  if (!st) return;
  const p = gameToThree(st.x, st.y, st.z);
  const wp = new THREE.Vector3(p.x, p.y, p.z);
  flyCameraTo(wp, 8, 5);   // flyCameraTo 自己会 clone，不会改 wp
  setFollowPlayer(followEnabled ? playerIdx : -1, wp);
  if (followEnabled && toast) {
    const name = playerName(playerIdx) || `P${playerIdx + 1}`;
    showToast(t('replay.toast.follow', { name }));
  }
}

function updateRoundStats() {
  if (!currentRound) return;
  $('stat-round').textContent = `#${currentRound.id ?? activeRoundIdx}`;
  $('stat-winner').textContent = currentRound.winner || '—';
  $('stat-reason').textContent = currentRound.end_reason || '—';
  $('stat-samples').textContent = fmtInt(currentRound.sample_count);
  $('round-stats').classList.remove('hidden');
}

function shortWeaponName(idx) {
  const n = weaponName(idx);
  return n.startsWith('weapon_') ? n.slice(7) : n;
}

function updatePlayerPanelFrame() {
  if (!currentRound) return;
  const states = getAllPlayerStates(currentSampleIdx);
  for (let i = 0; i < 10; i++) {
    const card = $(`player-card-${i}`);
    if (!card) continue;
    const st = states[i];
    const team = currentRound.teams[i];
    // className 是整串覆盖的（队伍 / 存活状态每帧都可能变），所以 `expanded`
    // 得自己带过来 —— 不带的话点开卡片的下一次刷新就被抹掉，选中态根本显示不出来。
    card.className = `player-card ${team === 'CT' ? 'ct' : 't'}`
      + (card.classList.contains('expanded') ? ' expanded' : '');
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

    const detail = $(`player-detail-${i}`);
    if (detail && !detail.classList.contains('hidden') && st) {
      const q = (sel) => detail.querySelector(sel);
      if (q('.pd-hp')) q('.pd-hp').textContent = st.alive ? `${Math.round(st.hp)} / 100` : '—';
      if (q('.pd-armor')) q('.pd-armor').textContent = st.alive ? `${Math.round(st.armor)} / 100` : '—';
      if (q('.pd-weapon')) q('.pd-weapon').textContent = st.alive ? shortWeaponName(st.weapon) : '—';
      if (q('.pd-inv')) {
        q('.pd-inv').textContent = st.alive && st.inventory.length
          ? st.inventory.map(shortWeaponName).join(' · ') : '—';
      }
      if (q('.pd-status')) {
        const bits = [];
        if (st.helmet) bits.push(t('replay.fp.helmet'));
        if (st.defuser) bits.push(t('replay.player.defuser'));
        if (st.flash > 0) bits.push(t('replay.player.flashed', { pct: Math.round(st.flash * 100) }));
        q('.pd-status').textContent = st.alive
          ? (bits.length ? bits.join(' · ') : t('replay.player.status.normal'))
          : t('replay.player.status.dead');
      }
      if (q('.pd-place')) q('.pd-place').textContent = st.place != null ? placeName(st.place) : '—';
      if (q('.pd-v')) q('.pd-v').textContent = st.alive ? `${st.v.toFixed(0)} u/s` : '—';
      if (q('.pd-ground')) {
        q('.pd-ground').textContent = st.alive
          ? t('replay.player.ground', {
            ground: st.on_ground ? t('replay.player.onGround') : t('replay.player.airborne'),
            jump: st.jump,
            land: st.land,
          }) : '—';
      }
      if (q('.pd-events')) q('.pd-events').textContent = st.alive ? `${st.shots} / ${st.footsteps}` : '—';
      if (q('.pd-crouch')) {
        // 碰撞盒（阶跃，看 crouching）和眼高（连续，看 crouch）是两个量，
        // 所以分开写清楚，别让读的人以为盒子是渐变的。
        if (!st.alive) {
          q('.pd-crouch').textContent = '—';
        } else {
          const hull = st.crouching ? 54 : 72;
          q('.pd-crouch').textContent = t('replay.player.crouchEye', {
            crouch: st.crouch.toFixed(2),
            hull,
            eye: st.eye_height.toFixed(1),
          });
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 击杀 / 事件
// ═══════════════════════════════════════════════════════════════════════

function addKillFeed(kill) {
  const feed = $('kill-feed');
  const item = document.createElement('div');
  item.className = 'kill-item';
  const wname = weaponName(kill.w);
  item.innerHTML = `
    <span class="killer">${playerName(kill.a)}</span>
    <span class="kweap">${wname}</span>
    <span class="victim">☠ ${playerName(kill.v)}</span>`;
  feed.appendChild(item);
  while (feed.children.length > 4) feed.removeChild(feed.firstChild);
  setTimeout(() => { if (item.parentNode) item.remove(); }, 5000);

  // 有官方武器图标就把文字换成图标。图标是 **512×384 非正方**，CSS 里定高不定宽，
  // 否则会被压扁。没有图标的武器（比如刀的各种皮肤）保留文字。
  const slot = item.querySelector('.kweap');
  const applyIcon = (url) => {
    if (!url || !slot || !slot.parentNode) return;
    const img = document.createElement('img');
    img.className = 'kweap-icon';
    img.src = url;
    img.alt = wname;
    img.title = wname;
    slot.replaceWith(img);
  };
  // 映射表可能还没加载好：这时返回 null，加载完会回调一次
  const url = weaponIconUrl(wname, applyIcon);
  if (url) applyIcon(url);
}

function handleReplayEvents(events) {
  if (!events) return;
  for (const k of (events.kills || [])) {
    addKillFeed(k);
    const victimState = getPlayerState(k.v, currentSampleIdx);
    if (victimState) {
      const v = gameToThree(victimState.x, victimState.y, victimState.z);
      showKillEffect(new THREE.Vector3(v.x, v.y, v.z));
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 子弹曳光
// ═══════════════════════════════════════════════════════════════════════

/**
 * 本回合的开火事件（精确 tick + 开枪者）。没有 `events.sound`（老 `.csz`）返回 null。
 */
function collectFireEvents() {
  const sound = currentRound && currentRound.events && currentRound.events.sound;
  if (!Array.isArray(sound)) return null;
  const fires = sound.filter((e) => e && e.ty === 'fire' && e.p >= 0);
  if (!fires.length) return null;
  fires.sort((a, b) => a.t - b.t);
  return fires;
}

/**
 * 为 `(fromTick, toTick]` 这段里的每一发开火生成一条曳光。
 *
 * 只在**正常播放**（一帧推进若干个采样点）时调用；拖时间轴走的是 `seekTo()`，
 * 不经过这里，所以来回拖不会一次性喷出一大把曳光。
 *
 * @returns {number} 生成了几条
 */
function spawnTracersBetween(fromTick, toTick) {
  if (!currentRound || !(toTick > fromTick)) return 0;
  if (!$('toggle-tracers') || !$('toggle-tracers').checked) return 0;

  const states = getAllPlayerStates(currentSampleIdx);
  let n = 0;

  if (roundFireEvents) {
    // 一发一条：事件里有精确 tick，也有是谁开的
    for (const e of roundFireEvents) {
      if (e.t > fromTick && e.t <= toTick) {
        if (spawnTracer(states, e.p)) n++;
      }
    }
    return n;
  }

  // 回退：`shots` 是本采样窗口内的开枪数（每 tick 采样时是 0/1，粗采样时会 >1）
  for (let i = 0; i < states.length; i++) {
    const st = states[i];
    if (!st || !st.alive) continue;
    const k = Math.min(st.shots | 0, 3);
    for (let s = 0; s < k; s++) if (spawnTracer(states, i)) n++;
  }
  return n;
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
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = 'rgba(255,255,255,0.07)';
  ctx.beginPath();
  ctx.roundRect(0, h / 2 - 5, w, 10, 5);
  ctx.fill();

  if (!currentRound) return;
  const dur = getRoundDuration();
  if (dur <= 0) return;
  const x = (t) => (t / dur) * w;

  const plantT = currentRound.bomb_planted_time;
  if (plantT) {
    ctx.fillStyle = 'rgba(255, 92, 122, 0.12)';
    ctx.fillRect(x(plantT), h / 2 - 5, w - x(plantT), 10);
    ctx.strokeStyle = 'rgba(255, 92, 122, 0.6)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x(plantT), 0); ctx.lineTo(x(plantT), h); ctx.stroke();
    ctx.setLineDash([]);
  }

  // 击杀
  for (const k of (currentRound.events.kills || [])) {
    // post = 回合结束后、下一回合开始前的有效交火：只进 KDA，不在本回合时间轴上
    if (k.post) continue;
    const kt = secForTick(k.t);
    ctx.fillStyle = '#ff5c7a';
    ctx.beginPath();
    ctx.moveTo(x(kt), h / 2 - 7);
    ctx.lineTo(x(kt) - 4, h / 2 + 5);
    ctx.lineTo(x(kt) + 4, h / 2 + 5);
    ctx.closePath(); ctx.fill();
  }
  // 炸弹
  for (const b of (currentRound.events.bomb || [])) {
    ctx.fillStyle = '#ff3ea5';
    ctx.beginPath(); ctx.arc(x(b.s), h / 2, 3, 0, Math.PI * 2); ctx.fill();
  }
  // 烟雾 / 火焰持续段（真实提取的区间）
  const band = (a, b, color) => {
    ctx.fillStyle = color;
    ctx.fillRect(x(Math.max(0, a)), h / 2 + 6, Math.max(1, x(b) - x(a)), 3);
  };
  for (const s of (currentRound.events.smokes || [])) band(s.ts, s.te ?? dur, 'rgba(200,210,225,0.55)');
  for (const f of (currentRound.events.infernos || [])) band(f.ts, f.te ?? dur, 'rgba(255,140,60,0.7)');

  const px = x(currentTime);
  ctx.fillStyle = '#00e5ff';
  ctx.shadowColor = 'rgba(0, 229, 255, 0.8)';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(px, 2); ctx.lineTo(px - 6, 12); ctx.lineTo(px + 6, 12);
  ctx.closePath(); ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillRect(px - 1, 8, 2, h - 8);

  $('timeline-progress').style.width = (currentTime / dur * 100) + '%';
}

/** tick → 回合内秒（用当前回合的 tick 表，逐 tick 时就是精确值）。 */
function secForTick(tick) {
  const ticks = currentRound.ticks;
  const secs = currentRound.round_seconds;
  let lo = 0, hi = ticks.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (ticks[mid] <= tick) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return secs[ans] || 0;
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
  const plantT = currentRound.bomb_planted_time;
  if (plantT != null && currentTime >= plantT) {
    const remain = Math.max(0, 40 - (currentTime - plantT));
    html += ` <span style="color:#ff5c7a;font-weight:600">💣 ${formatTime(remain)}</span>`;
  }
  $('time-display').innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════════
// 几何可观测性（visibility.js 的接线）
//
// 眼睛图标**只有这一个来源**：逐帧算的纯几何判定。眼睛 → 目标碰撞盒 8 个角落，
// 每个角落要求 ① 在视锥里 ② 没被地图三角形挡 ③ 没被烟雾体素挡，
// 8 个里**任意一个通**就算看得见。
//
// 以前这里还挂过 `.csz` 的 `spotted` 列（引擎的 m_bSpottedByMask），已经**撤掉**：
// 那一列被限速在 33 tick = 0.516 s 的网格上（见 README §4.8），和逐帧的几何放在
// 同一个图标上只会让人分不清看到的是哪一个。原始 `spotted` 数据仍然照常提取、
// 传输、显示在右侧「数据检查」面板里（`__v5.visibleEnemiesOf` 也还在），
// 只是**不再画成眼睛**。
//
// 开销：BVH 实测 0.0020 ms/条，加上视锥早退和"第一个通的角落就收工"，
// 10 人 90 对实测 0.1~0.6 ms/帧（见 tests/test_visibility.mjs 的性能用例）。
// ═══════════════════════════════════════════════════════════════════════

let showSpottedIcons = true;    // 眼睛图标（`toggle-spotted`，几何驱动）
let showFrustum = false;        // 选中玩家的视锥（`toggle-frustum`）
let geoVisCorners = false;      // 选中观察者的 8 角落调试射线（`toggle-vis-corners`）
let visEvaluator = null;
let visEvaluatorBVH = null;
let visCastMapRay = null;
let visResult = null;           // 最近一次 evaluate() 的结果
let visOccluders = [];
let visWarnedNoMap = false;
let visStats = { pairs: 0, rays: 0, fovRejects: 0, mapHits: 0, smokeHits: 0, playerHits: 0, ms: 0 };
const _eyeTmp = new Float64Array(3);

/**
 * 懒建求值器。BVH 是懒建的（第一次用才建，要一两秒），换地图还会换对象，
 * 所以这里按**对象身份**判断要不要重建 —— 只判 null 会在换图后继续拿旧 BVH。
 */
function ensureVisEvaluator() {
  const bvh = getMapBVH();
  if (visEvaluator && visEvaluatorBVH === bvh) return visEvaluator;
  visCastMapRay = makeMapCaster(bvh, SCALE);
  visEvaluator = createVisibilityEvaluator({ castMapRay: visCastMapRay });
  visEvaluatorBVH = bvh;
  if (!bvh && !visWarnedNoMap) {
    visWarnedNoMap = true;
    console.warn('[visibility] 地图 BVH 还没建好，几何可见性暂时只按视锥 + 烟雾判定'
      + '（不算墙体遮挡），建好后会自动重算');
  }
  if (bvh) visWarnedNoMap = false;
  return visEvaluator;
}

/** 算一次整张矩阵，结果存进 `visResult` / `visStats`。 */
function computeGeoVisibility(playerStates, tick, occluders) {
  const ev = ensureVisEvaluator();
  visOccluders = occluders || smokeOccludersAt(currentRound ? currentRound.smoke : null, tick);
  ev.setOccluders(visOccluders);
  visResult = ev.evaluate(playerStates);
  visStats = { ...visResult.stats, occluders: visOccluders.length, mapReady: ev.mapReady };
  return visResult;
}

// ═══════════════════════════════════════════════════════════════════════
// 选手 X 光（player-xray.js 的接线）
//
// 和上面的几何可观测性同源但**方向不同**：那边算的是「选手 B 看不看得见选手 A」
// （有视锥、逐角落），这边算的是「**这台摄像机**看不看得见某个选手」（无视锥，
// 屏幕上看得见就是看得见）。结果只用来决定小人画成不透明的还是半透明幽灵，
// 于是"真的看到"和"透过墙看到"一眼能分开。
//
// 判定是纯几何、每帧算：10 个活人 × 3 个采样点（实测 0.2 ms 以内；BVH 没建好时
// 墙不算遮挡，谁都不会被画成幽灵 —— 地图还没加载也就没有墙）。
// ═══════════════════════════════════════════════════════════════════════

let showPlayerXray = true;                  // `选手 X 光`，默认开
/** 每个玩家的判定结果：1 = 被挡住，只能透视看到。判定器复用的就是这一个数组。 */
const playerXrayFlags = new Uint8Array(10);
let playerXrayTester = null;
let playerXrayTesterBVH = null;
let playerXrayStats = null;

/**
 * 摄像机位置换算到 **game 空间**（Hammer 单位）—— player-xray.js 和
 * playerStates 用同一套坐标。这是 scene.js 那条 `(gx,gy,gz) → (gy,gz,gx)·SCALE`
 * 的逆：`gx = three.z / SCALE`、`gy = three.x / SCALE`、`gz = three.y / SCALE`。
 * 复用一个对象：这个函数每帧都跑。
 */
const _camGame = { x: 0, y: 0, z: 0 };
function cameraGamePos() {
  _camGame.x = camera.position.z / SCALE;
  _camGame.y = camera.position.x / SCALE;
  _camGame.z = camera.position.y / SCALE;
  return _camGame;
}

/**
 * 懒建判定器。和 `ensureVisEvaluator()` 同一个理由、按**对象身份**判要不要重建：
 * 换地图会换 BVH 对象，只判 null 的话换图后会继续拿旧 BVH 判。
 */
function ensurePlayerXrayTester() {
  const bvh = getMapBVH();
  if (playerXrayTester && playerXrayTesterBVH === bvh) return playerXrayTester;
  playerXrayTester = createPlayerXRayTester({
    castMapRay: makeMapCaster(bvh, SCALE),
    flags: playerXrayFlags,
  });
  playerXrayTesterBVH = bvh;
  return playerXrayTester;
}

/** 某个观察者能观测到的**全部**其他人（含队友；`.csz` 的 spotted 只标敌方）。 */
function geoVisibleOf(viewerIdx, result) {
  const out = [];
  if (!result || viewerIdx < 0 || viewerIdx >= result.n) return out;
  for (let a = 0; a < result.n; a++) {
    if (a !== viewerIdx && result.visible[viewerIdx * result.n + a]) out.push(a);
  }
  return out;
}

/**
 * 几何判定要地图 BVH 才准（没有它就只能按视锥 + 烟雾判，墙不算遮挡）。
 * BVH 是懒建的（823k 三角形一两秒），建的时候把开关标成 busy 免得以为卡死。
 */
async function ensureVisBVH(label) {
  if (getMapBVH()) return true;
  if (label) label.classList.add('busy');
  try {
    const bvh = await ensureMapBVH();
    if (!bvh) {
      console.warn('[visibility] 地图还没加载好，几何可见性暂时只按视锥 + 烟雾判定');
      showToast(t('replay.toast.visNoMap'), true);
    }
    return !!bvh;
  } finally {
    if (label) label.classList.remove('busy');
  }
}

/** 角落射线的颜色，按 visibility.js 的 BLOCK_* 位分。 */
function visCornerColor(reason) {
  if (reason === 0) return VIS_CORNER_COLORS.clear;
  if (reason & BLOCK_FOV) return VIS_CORNER_COLORS.fov;
  if (reason & BLOCK_MAP) return VIS_CORNER_COLORS.map;
  if (reason & BLOCK_SMOKE) return VIS_CORNER_COLORS.smoke;
  if (reason & BLOCK_PLAYER) return VIS_CORNER_COLORS.player;
  return VIS_CORNER_COLORS.fov;
}

/**
 * 选中观察者的 9×8 = 72 条角落射线（game 空间）。走 `pairVisibility` 的
 * 逐角落明细版 —— 会分配对象，所以只在调试开关打开时才调用。
 */
function buildVisCornerSegments(viewerIdx, playerStates, tick) {
  const segs = [];
  const v = playerStates[viewerIdx];
  if (!v || !v.alive) return segs;
  eyePos(v, _eyeTmp);
  const occluders = smokeOccludersAt(currentRound ? currentRound.smoke : null, tick);
  for (let a = 0; a < playerStates.length; a++) {
    if (a === viewerIdx) continue;
    const target = playerStates[a];
    if (!target || !target.alive) continue;
    const d = pairVisibility(v, target, {
      castMapRay: visCastMapRay, occluders, viewerIdx, targetIdx: a,
    });
    for (const c of d.corners) {
      segs.push({
        o: [_eyeTmp[0], _eyeTmp[1], _eyeTmp[2]],
        t: [c.x, c.y, c.z],
        color: visCornerColor(c.reason),
      });
    }
  }
  return segs;
}

/**
 * 第 `i` 个采样点上所有人的**原始**位置（游戏坐标，不插值）。
 *
 * 轨迹重算用这个，不用 `getAllPlayerStates()`：后者按 `currentTime` 在相邻采样点
 * 之间插值，同一个采样点在不同播放相位读到的坐标能差几厘米 —— 拿它重算，拖走再
 * 拖回来轨迹末端就对不上（看起来像"轨迹被抹了一下"）。取原始采样点之后，
 * `trail(k)` 才是采样点的纯函数，同一点永远是同一张轨迹。
 *
 * 10 个对象都是新建的，但只在**跳转**那一趟里被调 150 次（重算一整条轨迹），
 * 正常播放走的是"追加一个点"，一帧一次。
 */
function rawPositionsAt(i) {
  if (!currentRound) return null;
  const players = currentRound.players;
  const n = currentRound.ticks.length;
  const j = Math.min(Math.max(0, i | 0), Math.max(0, n - 1));
  const out = [];
  for (let k = 0; k < players.length; k++) {
    const p = players[k];
    out.push(p ? { x: p.x[j], y: p.y[j], z: p.z[j] } : null);
  }
  return out;
}

function updateAllVisuals() {
  if (!currentRound) return;
  const idx = currentSampleIdx;
  const tick = currentTick();
  const playerStates = getAllPlayerStates(idx);
  const teams = currentRound.teams;
  const showNames = $('toggle-names').checked;
  const showTrails = $('toggle-trails').checked;
  const showSmokes = $('toggle-smokes').checked;
  const showFire = $('toggle-fire').checked;
  const solid = $('toggle-solid').checked;
  const showWeaponModels = $('toggle-weapon-models').checked;

  // 几何可观测性：眼睛图标、角落射线、视锥都吃这一份结果。眼睛图标默认开着，
  // 所以这里是**每帧**算；三个开关全关掉时一次都不算，开销为 0。
  const needMatrix = showSpottedIcons || geoVisCorners;
  // 烟雾遮挡体是「眼睛图标」和「选手 X 光」共用的（两边都要它来判"隔着烟看不见"），
  // 而 smokeOccludersAt() 每帧要解一次体素缓存 —— 所以只解一次。
  const occluders = (needMatrix || showPlayerXray)
    ? smokeOccludersAt(currentRound ? currentRound.smoke : null, tick)
    : [];
  if (needMatrix) computeGeoVisibility(playerStates, tick, occluders);

  // 选手 X 光：从**这台摄像机**量每个选手是直接看得见还是被挡住。默认开着，
  // 所以也是每帧算。⚠️ 必须排在 `updatePlayers()` 之前：幽灵材质是那一趟按
  // 这个结果贴的。
  if (showPlayerXray) {
    const tester = ensurePlayerXrayTester();
    tester.setOccluders(occluders);
    playerXrayStats = tester.classify(cameraGamePos(), playerStates).stats;
    setPlayerXRayFlags(playerXrayFlags);
  } else {
    setPlayerXRayFlags(null);
  }

  if (geoVisCorners && expandedPlayerIdx >= 0) {
    updateVisCornerRays(buildVisCornerSegments(expandedPlayerIdx, playerStates, tick));
  } else {
    updateVisCornerRays(null);
  }
  // 视锥只画选中的那一个（10 个叠一起没法看）。不需要矩阵，只要有朝向。
  updateViewFrustum(expandedPlayerIdx >= 0 ? playerStates[expandedPlayerIdx] : null);

  // 「可见敌人」眼睛：只在左侧卡片选中了人（expandedPlayerIdx >= 0）时才有目标。
  //
  // 数据源**只有几何**一个（`visibility.js` 逐帧算的矩阵，含队友 —— 全有序对，
  // 只排除自己）。引擎的 `spotted` 列不再画成图标，见本文件上面那段注释。
  //
  // ⚠️ 必须在 `updatePlayers()` **之前**灌进去：图标的位置和显隐是
  // updatePlayers 落盘的，顺序反了的话一次 updateAllVisuals() 里画出来的还是
  // 上一帧的目标（点卡片要等下一帧才出眼睛，拖时间轴会差一帧）。
  if (!showSpottedIcons || expandedPlayerIdx < 0) setSpotTargets(null);
  else setSpotTargets(geoVisibleOf(expandedPlayerIdx, visResult));

  updatePlayers(playerStates, idx, teams, showNames, showTrails);
  // 真实武器 3D 模型挂在玩家手上（静态 bind pose，无骨骼动画）
  updateWeaponModels(playerStates, showWeaponModels);

  const bombPos = getBombPosition(idx);
  updateBomb(bombPos, isBombPlanted(idx), currentTime);
  updateGrenadeTrajectories(currentTime);

  // 烟 / 火：都在各自的 changes 上按 tick 取最近快照，没有就返回 null
  updateSmokeVolumes(showSmokes ? currentRound.smoke : [], tick, { solid });
  updateFireCells(showFire ? currentRound.fire : [], tick);

  // 第一人称：藏掉自己的模型（否则镜头就卡在模型头里），并把 HUD 刷成
  // 这个人这一帧的真实状态。死亡时自动退出 —— 死人的眼睛看东西没有意义，
  // 而且卡在原地会让人以为相机坏了。
  if (isFirstPerson()) {
    const fpState = playerStates[firstPersonIdx];
    if (fpState && !fpState.alive) {
      const name = playerName(firstPersonIdx) || `P${firstPersonIdx + 1}`;
      exitFirstPerson({ restore: true, toast: false, refresh: false, revealModel: false });
      showToast(t('replay.toast.playerDead', { name }));
    } else {
      // updatePlayers() 刚把 group.visible 置回 true，所以必须在这之后藏。
      setPlayerModelVisible(firstPersonIdx, false);
      updateFirstPersonHud(fpState || null);
    }
  }

  updateAimRays(playerStates);
  updatePlayerPanelFrame();
  // AI 预测：插值 + 3D 覆盖层 + 顶部面板。没有数据时它只是把开关对应的东西藏掉。
  updateAiFrame(playerStates);
}

// ═══════════════════════════════════════════════════════════════════════
// AI 预测（spatial 五项指标 + aim 对枪）
//
// 数据流：服务端 `/api/ai/round/<idx>?progress=1` 是一条 **NDJSON 流** —— 一行
// 一个事件：`{type:'progress', pct, stage, elapsed}` / `{type:'result', payload}`
// / `{type:'error', error, kind}`。冷分析一次二十秒，所以进度条是必须的：用户
// 得知道"现在是构建特征还是跑模型、还要等多久"。
//
// 拿到整回合的预测之后，这边只做四件事：
//
//   1. 按 `currentSampleIdx` 在相邻两个预测点之间**线性插值**，60 fps 的画面
//      就不会一跳一跳；"是谁"这种量（对枪的两个人）没法插值，取最近的点。
//   2. 顶部面板 = 赢面（+ 进度条）；左下表格 = 每个选手的五项概率。
//   3. 3D 覆盖层（ai-overlay.js）= 只有对枪弧线（只在死前 3 秒里出现）。
//   4. 表格上的交互：悬停表头看释义（`#ai-tip`）、点名字跳人（复用卡片那套
//      选中 + 跟随）、点数字看整回合曲线（ai-chart.js，点曲线上任意一点 =
//      seekTo + 锁定该选手）。
//
// 回合缓存：`aiRound` 只对 `aiRoundFor` 那个回合有效。切回合先清空，再按开关
// 状态重新拉一次 —— 服务端那边同一回合第二次是读 JSON 缓存（毫秒级）。曲线和
// 释义卡都挂在同一份数据上，所以换回合时一起收掉（见 `resetAiRound`）。
// ═══════════════════════════════════════════════════════════════════════

/** `/api/ai/status` 的结果（模型路径 / 设备 / 不可用原因）。 */
let aiStatus = null;
/** 当前回合的预测载荷；`null` = 还没拉到。 */
let aiRound = null;
/** `aiRound` 属于哪个回合（-1 = 无效）。 */
let aiRoundFor = -1;
/** 请求序号：切回合/连点时只认最后一次。 */
let aiLoadSeq = 0;
/** 在飞的流式请求（换回合就 abort，别让服务端白算一个不要的回合）。 */
let aiAbort = null;
/** 正在请求哪个回合（-1 = 没有）。同回合的重复调用直接吞掉，见 `ensureAiRound`。 */
let aiLoadingFor = -1;
/** 上一次写进面板的赢面，没变就不动 DOM。 */
let aiWinKey = '';
/** 上一次写进面板的"用的哪份权重"（run 名 + 文件名）。 */
let aiModelKey = '';
/** 上一次写进去的进度文本。 */
let aiProgressKey = '';
/** 进度：目标值来自服务端，显示值向它靠（80 ms 一推，直接跳会一顿一顿的）。 */
let aiProgressTarget = 0;
let aiProgressShown = 0;
/** 上次算平滑的时间戳（按 dt 收敛，见 `updateAiProgress`）。 */
let aiProgressAt = 0;
let aiStage = '';
let aiElapsed = 0;
/** 正在算（进度条 / 面板跟着它显示）。 */
let aiAnalyzing = false;
/** 表格的行属于哪个回合（换了回合要重建行，因为队名 / 名字可能变）。 */
let aiTableFor = -1;
/** 每行上一次写进去的内容，没变就不动 DOM。 */
const aiRowKeys = [];
/** 指针正停在哪个表头上（'' = 没有）。用来避免重复填 DOM。 */
let aiTipMetric = '';
/** 曲线卡：第一次点数字时才建（它有 canvas，页面一打开就建没必要）。 */
let aiChart = null;

/**
 * 表头释义：悬停 `.ai-col` 时由 `#ai-tip` 显示。
 *
 * 为什么不用 `title` 属性：`.ai-table` 整块是 `pointer-events:none`（不能挡住
 * 3D 视口的拖拽），原生 title 在这种元素上根本不会出现；而且原生的要等半秒、
 * 字体和配色也不受控。这套文案只在这里维护一份：列头文字、曲线卡标题、悬停
 * 说明用的是同一组名字（列头本身写在 index.html 里，改的时候两边一起改）。
 */
const AI_METRIC_HELP = {
  win_ct: {
    title: t('replay.ai.help.win_ct.title'),
    src: t('replay.ai.help.win_ct.src'),
    color: '#8cc2f7',
    body: t('replay.ai.help.win_ct.body'),
    hint: t('replay.ai.help.win_ct.hint'),
  },
  win_t: {
    title: t('replay.ai.help.win_t.title'),
    src: t('replay.ai.help.win_t.src'),
    color: '#f3b969',
    body: t('replay.ai.help.win_t.body'),
    hint: t('replay.ai.help.win_t.hint'),
  },
  name: {
    title: t('replay.ai.metric.name'),
    src: t('replay.ai.help.name.src'),
    color: '#8fa3c4',
    body: t('replay.ai.help.name.body'),
    hint: t('replay.ai.help.name.hint'),
  },
  alive: {
    title: t('replay.ai.metric.alive'),
    src: t('replay.ai.help.alive.src'),
    color: '#7dd3fc',
    body: t('replay.ai.help.alive.body'),
    hint: t('replay.ai.help.metric.hint'),
  },
  can_kill: {
    title: t('replay.ai.metric.can_kill'),
    src: t('replay.ai.help.can_kill.src'),
    color: '#b69bff',
    body: t('replay.ai.help.can_kill.body'),
    hint: t('replay.ai.help.metric.hint'),
  },
  next_kill: {
    title: t('replay.ai.metric.next_kill'),
    src: t('replay.ai.help.next_kill.src'),
    color: '#34d399',
    body: t('replay.ai.help.next_kill.body'),
    hint: t('replay.ai.help.metric.hint'),
  },
  next_death: {
    title: t('replay.ai.metric.next_death'),
    src: t('replay.ai.help.next_death.src'),
    color: '#f87171',
    body: t('replay.ai.help.next_death.body'),
    hint: t('replay.ai.help.metric.hint'),
  },
  duel: {
    title: t('replay.ai.metric.duel'),
    src: t('replay.ai.help.duel.src'),
    color: '#d8c8ff',
    body: t('replay.ai.help.duel.body'),
    hint: t('replay.ai.help.duel.hint'),
  },
};

function aiWanted() {
  return $('toggle-ai-spatial').checked || $('toggle-ai-duel').checked;
}

function setAiState(text, isError = false) {
  const el = $('ai-state');
  el.textContent = text || '';
  el.classList.toggle('error', !!isError);
}

function showAiPanel(on, progressOnly = false) {
  const panel = $('ai-panel');
  panel.classList.toggle('hidden', !on);
  // 只开了「AI 对枪」时，面板只用来放进度条：赢面那行没有意义。
  panel.classList.toggle('progress-only', !!progressOnly);
}

/** 启动时问一次模型在不在（不会加载权重，毫秒级）。不可用就把两个开关禁掉。 */
async function loadAiStatus() {
  try {
    const res = await fetch('/api/ai/status');
    aiStatus = await res.json();
  } catch (err) {
    aiStatus = { ok: false, reason: t('replay.ai.statusFail', { msg: err.message }) };
  }
  const ok = !!(aiStatus && aiStatus.ok);
  for (const id of ['toggle-ai-spatial', 'toggle-ai-duel']) {
    const input = $(id);
    input.disabled = !ok;
    const label = input.closest('label');
    if (!label) continue;
    if (label.dataset.originalTitle === undefined) {
      label.dataset.originalTitle = label.title;
    }
    label.title = ok
      ? label.dataset.originalTitle
      : t('replay.ai.unavailableTitle', {
        reason: aiStatus?.reason || t('replay.ai.unknownReason'),
      });
  }
  if (!ok) {
    showAiPanel(false);
    console.warn('[ai] 不可用:', aiStatus && aiStatus.reason);
  }
  return ok;
}

/** 这个 demo 做不了 AI（地图不在 8 张里 / 不是逐 tick 抽样 / 模型没了）：
 *  把两个开关关掉并禁用，免得每次换回合都去撞同一条错误。 */
function disableAi(reason) {
  for (const id of ['toggle-ai-spatial', 'toggle-ai-duel']) {
    const input = $(id);
    input.checked = false;
    input.disabled = true;
    const label = input.closest('label');
    if (label) label.title = t('replay.ai.unavailableTitle', { reason });
  }
  showAiPanel(false);
  resetAiRound();
}

/**
 * 拉一个回合的预测，边走边报进度。
 *
 * 服务端 `?progress=1` 回的是 **NDJSON**（一行一个事件）。用 `fetch` + 流式读取
 * 而不是 `EventSource`：这里是 GET、只要一条一次性的流，而 `EventSource` 自带
 * 重连语义 —— 一次二十秒的 GPU 计算重连没有意义。
 *
 * 拿不到流（服务端是旧版本 / 浏览器不给 `res.body`）就退回一次性 JSON：同一个
 * 路由不带 `progress` 参数就是纯 JSON，行为完全一样。
 *
 * @returns {Promise<object>} 服务端的预测 payload
 */
async function fetchAiRound(idx, onEvent, refresh = false) {
  const qs = new URLSearchParams({ progress: '1' });
  if (refresh) qs.set('refresh', '1');
  return streamJson(`/api/ai/round/${idx}?${qs}`, onEvent, {
    signal: aiAbort?.signal,
    brokenText: t('replay.error.streamBroken'),
  });
}

/**
 * 拉当前回合的预测。已经在手上就不动（`force` = 强制重算，服务端会跳过缓存）。
 */
async function ensureAiRound(force = false) {
  if (!aiWanted()) return;
  if (!force && aiRound && aiRoundFor === activeRoundIdx) return;
  const idx = activeRoundIdx;
  // 两个开关各触发一次 change，第 2 次不能再发一条请求：那会把第 1 条 abort 掉，
  // 而服务端的计算停不下来（worker 继续跑、继续占着那把锁），于是新请求排在
  // 自己废弃的那个后面，进度条从 0 重来一遍、白白多等一个回合。
  if (!force && aiLoadingFor === idx) return;
  const seq = ++aiLoadSeq;
  aiAbort?.abort();
  const ac = new AbortController();
  aiAbort = ac;
  aiLoadingFor = idx;

  aiAnalyzing = true;
  aiProgressTarget = 0;
  aiProgressShown = 0;
  aiProgressAt = performance.now();
  aiElapsed = 0;
  aiStage = force ? t('replay.ai.stage.reanalyzing') : t('replay.ai.stage.preparing');
  aiProgressKey = '';
  aiWinKey = '';
  showAiPanel(true, !$('toggle-ai-spatial').checked);
  setAiState(t('replay.ai.state.analyzing'));
  showToast(force ? t('replay.ai.toast.reanalyzing') : t('replay.ai.toast.analyzing'));

  try {
    const json = await fetchAiRound(idx, (ev) => {
      if (seq !== aiLoadSeq) return;          // 用户已经切到别的回合
      if (typeof ev.pct === 'number') {
        aiProgressTarget = Math.max(aiProgressTarget, Math.min(1, ev.pct));
      }
      if (typeof ev.stage === 'string' && ev.stage) aiStage = ev.stage;
      if (typeof ev.elapsed === 'number') aiElapsed = ev.elapsed;
    }, force);
    if (seq !== aiLoadSeq) return;
    aiAnalyzing = false;
    aiLoadingFor = -1;
    aiProgressTarget = 1;
    aiProgressShown = 1;
    aiRound = json;
    aiRoundFor = idx;
    aiWinKey = '';
    aiTableFor = -1;
    const ms = Math.round(json.ms || 0);
    setAiState(json.cached ? t('replay.ai.state.cached')
      : t('replay.ai.state.fresh', { s: (ms / 1000).toFixed(1) }));
    showToast(t('replay.ai.toast.ready', {
      n: idx + 1,
      detail: json.cached ? t('replay.ai.cacheHit') : (ms / 1000).toFixed(1) + ' s',
    }));
    updateAllVisuals();
  } catch (err) {
    if (seq !== aiLoadSeq || ac.signal.aborted) return;
    aiAnalyzing = false;
    aiLoadingFor = -1;
    aiRound = null;
    aiRoundFor = -1;
    setAiState(t('replay.ai.state.unavailable'), true);
    showToast(t('replay.ai.toast.fail', { msg: err.message }), true);
    if (err.kind === 'features' || err.kind === 'models') disableAi(err.message);
    updateAllVisuals();
  }
}

/** 清掉当前回合的预测（换回合 / 换比赛时）。 */
function resetAiRound() {
  aiLoadSeq++;                 // 在飞的请求全部作废
  aiAbort?.abort();
  aiAbort = null;
  aiAnalyzing = false;
  aiLoadingFor = -1;
  aiRound = null;
  aiRoundFor = -1;
  aiWinKey = '';
  aiProgressKey = '';
  aiTableFor = -1;
  aiRowKeys.length = 0;
  const body = $('ai-table-body');
  if (body) body.innerHTML = '';
  // 曲线是这一回合的数据画的，换回合必须收掉 —— 留着就是上一回合的曲线套在
  // 新回合的时间轴上，比不显示还误导。
  aiChart?.close();
  hideAiTip();
  clearAiOverlay();
}

/** 当前采样点上的插值结果；没有数据返回 null。 */
function aiFrameAt(sampleIdx) {
  if (!aiRound || aiRoundFor !== activeRoundIdx) return null;
  const n = (aiRound.samples || []).length;
  if (!n) return null;
  const stride = aiRound.stride || 16;
  const k = Math.min(n - 1, Math.max(0, sampleIdx / stride));
  const i0 = Math.floor(k);
  const i1 = Math.min(n - 1, i0 + 1);
  const f = k - i0;
  const between = (a, b) => (Number.isFinite(a) ? a : 0) * (1 - f)
    + (Number.isFinite(b) ? b : 0) * f;
  const nearest = f < 0.5 ? i0 : i1;

  const sp = aiRound.spatial;
  const per = (rows, p) => between(rows[i0]?.[p], rows[i1]?.[p]);
  const players = [];
  for (let p = 0; p < 10; p++) {
    players.push({
      alive: per(sp.alive, p),
      canKill: per(sp.can_kill, p),
      nextDeath: per(sp.next_death, p),
      nextKill: per(sp.next_kill, p),
    });
  }

  // 对枪：两个相邻预测点必须是**同一对**人才能插值胜率，否则取最近的那个点
  const da = aiRound.duel.a;
  const db = aiRound.duel.b;
  const samePair = da[i0] != null && da[i0] === da[i1] && db[i0] === db[i1];
  let duel = null;
  if (samePair) {
    duel = {
      a: da[i0],
      b: db[i0],
      p: between(aiRound.duel.p[i0] ?? 0.5, aiRound.duel.p[i1] ?? 0.5),
      t0: aiRound.duel.t0[i0],
      t1: aiRound.duel.t1[i0],
    };
  } else if (da[nearest] != null) {
    duel = {
      a: da[nearest],
      b: db[nearest],
      p: aiRound.duel.p[nearest] ?? 0.5,
      t0: aiRound.duel.t0[nearest],
      t1: aiRound.duel.t1[nearest],
    };
  }
  if (duel) {
    // 名字只给左下角表格的悬停提示用（3D 里不写字了，见 ai-overlay.js）
    duel.names = [
      playerName(duel.a) || `P${duel.a + 1}`,
      playerName(duel.b) || `P${duel.b + 1}`,
    ];
    // 窗口开场 0.35 秒淡入：弧线不该"啪"地一下出现在两个人之间
    duel.fade = Math.min(1, Math.max(0, (currentTime - (duel.t0 ?? 0)) / 0.35));
  }

  return {
    players,
    alive: players.map((x) => x.alive),
    winCt: between(aiRound.spatial.win_ct[i0], aiRound.spatial.win_ct[i1]),
    duel,
    cached: !!aiRound.cached,
  };
}

/** 面板里那行「用的哪份权重」。run 名看得见，悬停是 `last.pt` 这种具体文件名。 */
function syncAiModelLabel() {
  const runs = aiStatus && aiStatus.runs;
  const label = runs
    ? `aim ${runs.aim.run} · spatial ${runs.spatial.run}`
    : (aiRound && aiRound.models ? aiRound.models.label : '—');
  const title = runs
    ? `${runs.aim.run}/${runs.aim.weights} · ${runs.spatial.run}/${runs.spatial.weights}`
    : '';
  const key = `${label}\u0000${title}`;
  if (key === aiModelKey) return;
  aiModelKey = key;
  $('ai-model').textContent = label;
  $('ai-model').title = title;
}

/** 百分比：整数；极小但不是 0 的值显示 `<1%`，免得"模型认为有可能"被画成 0。 */
function aiPct(v) {
  if (!Number.isFinite(v)) return '—';
  const pct = v * 100;
  if (pct > 0 && pct < 1) return '<1%';
  return `${Math.round(pct)}%`;
}

// ── 表头释义（悬停说明）─────────────────────────────────────────────────

function hideAiTip() {
  if (!aiTipMetric) return;
  aiTipMetric = '';
  $('ai-tip').classList.remove('show');
}

/**
 * 把 `#ai-tip` 摆到列头上方（够不着就翻到下方）。
 *
 * 先 `show` 再量尺寸：`visibility:hidden` 也参与布局，所以量得到真实宽高；
 * 量完才写 left/top，否则会先按旧位置画一帧、再跳过去。
 */
function showAiTip(col) {
  const help = AI_METRIC_HELP[col.dataset.metric];
  if (!help) return;
  const tip = $('ai-tip');
  if (!tip) return;
  aiTipMetric = col.dataset.metric;
  tip.dataset.metric = aiTipMetric;
  $('ai-tip-dot').style.background = help.color;
  $('ai-tip-dot').style.color = help.color;
  $('ai-tip-title').textContent = help.title;
  $('ai-tip-src').textContent = help.src;
  $('ai-tip-body').textContent = help.body;
  $('ai-tip-hint').textContent = help.hint;
  tip.classList.add('show');

  const rect = col.getBoundingClientRect();
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  let left = rect.left + rect.width / 2 - w / 2;
  left = Math.max(8, Math.min(window.innerWidth - w - 8, left));
  let top = rect.top - h - 9;
  if (top < 8) top = rect.bottom + 9;
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(top)}px`;
}

// ── 指标曲线（点数字打开）────────────────────────────────────────────────

/**
 * 点指标表里的名字 / 点曲线：和点左边选手卡片一样（选中 + 镜头飞过去跟随）。
 *
 * 和卡片点击的唯一区别：卡片是"再点一次收起"，这里永远是"选中他" —— 点名字
 * 的意图是"跳过去"，不是"取消"。第一人称下和卡片一样拒绝换人（镜头锁着）。
 */
function focusPlayerFromAi(i, { toast = true } = {}) {
  if (i < 0 || i >= 10) return;
  if (isFirstPerson()) {
    showToast(t('replay.toast.fpLocked'), true);
    return;
  }
  togglePlayerDetail(i, true);
  focusCameraOnPlayer(i, { toast });
}

function ensureAiChart() {
  if (!aiChart) {
    aiChart = createAiChart({
      // 点曲线 = 暂停 + 跳到那一刻 + 锁定这个选手（用户是来"看这一刻"的，
      // 播放头继续跑会立刻把刚跳到的画面冲掉）
      onSeek: (seconds, info) => {
        if (isPlaying) { setPlaying(false); updatePlayButton(); }
        seekTo(seconds);
        drawTimeline();
        if (info && info.player >= 0) focusPlayerFromAi(info.player, { toast: false });
        updateAllVisuals();
        aiChart.update(currentTime);
      },
      onClose: hideAiTip,
    });
  }
  return aiChart;
}

/**
 * 从整回合的预测里抽「某个选手 + 某一项指标」的整条曲线（4 Hz）。
 *
 * `values` 里的 `null` 是"这一刻没有意义"：人还没进场 / 已经死了。不能填 0 ——
 * 服务端把死人的列清 0 是"别显示"，画成 0% 就成了"模型认为他不可能活"。
 * 阵亡时刻取"第一个不在活的采样点"，画一条红色虚线；真实死亡在两点之间。
 */
function aiSeriesFor(playerIdx, metric) {
  if (!aiRound || aiRoundFor !== activeRoundIdx) return null;
  const rows = aiRound.spatial && aiRound.spatial[metric];
  const seconds = aiRound.seconds || [];
  if (!rows || !rows.length || !seconds.length) return null;
  const stride = aiRound.stride || 16;
  const pl = currentRound && currentRound.players && currentRound.players[playerIdx];
  const frames = currentRound ? currentRound.ticks.length : 0;
  const values = new Array(rows.length);
  let death = null;
  for (let k = 0; k < rows.length; k++) {
    const frameIdx = frames > 0 ? Math.min(frames - 1, k * stride) : 0;
    const alive = pl && frames > 0 ? !!pl.alive[frameIdx] : true;
    values[k] = alive && rows[k] ? (rows[k][playerIdx] ?? null) : null;
    if (!alive && death == null && k > 0) death = seconds[k];
  }
  const help = AI_METRIC_HELP[metric];
  const name = playerName(playerIdx) || `P${playerIdx + 1}`;
  return {
    player: playerIdx,
    metric,
    name,
    label: help ? help.title : metric,
    // 曲线卡自己拼标题 / tooltip / 无障碍文案：选手曲线和胜率曲线的说法不一样
    heading: `${name} · ${help ? help.title : metric}`,
    aria: t('replay.chart.aria.player', { name, label: help ? help.title : metric }),
    title: t('replay.chart.title.metric', { label: help ? help.title : metric, metric }),
    hint: t('replay.chart.hint.player'),
    seconds: seconds.slice(),
    values,
    death,
    cursor: currentTime,
  };
}

/**
 * 顶部那个「回合胜率」的曲线（和选手曲线共用同一张卡）。
 *
 * 数据是同一个 `win_ct` 列：CT 直接用，T 用 `1 - p`。它不属于任何选手，所以
 * `player = -1` —— 点曲线只跳时间、不抢镜头（用户可能正跟着某个人看，跳一下
 * 胜率曲线不该把他甩掉）。胜率没有"阵亡"概念，`death` 恒为 null。
 */
function aiWinSeries(metric) {
  if (!aiRound || aiRoundFor !== activeRoundIdx) return null;
  const ct = aiRound.spatial && aiRound.spatial.win_ct;
  const seconds = aiRound.seconds || [];
  if (!ct || !ct.length || !seconds.length) return null;
  const side = metric === 'win_t' ? 'T' : 'CT';
  const values = ct.map((v) => {
    if (!Number.isFinite(v)) return null;
    return side === 'T' ? 1 - v : v;
  });
  const help = AI_METRIC_HELP[metric];
  return {
    player: -1,
    metric,
    name: t('replay.ai.winTitle'),
    label: side,
    heading: t('replay.ai.winTitle.side', { side }),
    aria: t('replay.chart.aria.win', { side }),
    title: t('replay.chart.title.win', { side }),
    hint: t('replay.chart.hint.win'),
    emptyText: t('replay.chart.empty.win'),
    seconds: seconds.slice(),
    values,
    death: null,
    cursor: currentTime,
  };
}

/** 点数字（`playerIdx >= 0`）或点胜率（`-1`）都走这里取曲线。 */
function aiSeriesForCell(playerIdx, metric) {
  return playerIdx < 0 ? aiWinSeries(metric) : aiSeriesFor(playerIdx, metric);
}

/** 点同一个数字 = 收起；点另一个 = 换一条曲线。 */
function toggleAiChart(playerIdx, metric) {
  const chart = ensureAiChart();
  const st = chart.state;
  if (st.open && st.player === playerIdx && st.metric === metric) {
    chart.close();
    return;
  }
  const series = aiSeriesForCell(playerIdx, metric);
  if (!series) {
    showToast(t('replay.toast.aiNotReady'), true);
    return;
  }
  hideAiTip();
  chart.open(series);
  chart.update(currentTime);
}

/** 进度条：目标值来自服务端，显示值按**时间**追（服务端 80 ms 一推）。 */
function updateAiProgress(analyzing) {
  const box = $('ai-progress');
  if (!box) return;
  box.classList.toggle('hidden', !analyzing);
  box.classList.toggle('is-running', analyzing);
  if (!analyzing) return;
  // 按 dt 收敛，不按帧数：60 fps 时是平滑的，浏览器被限到 2 fps 时也能在
  // 半秒内追上服务端的真实进度（按帧数收敛在低帧率下会永远落后半张进度条）。
  const now = performance.now();
  const dt = Math.min(0.5, Math.max(0, (now - aiProgressAt) / 1000));
  aiProgressAt = now;
  const gap = aiProgressTarget - aiProgressShown;
  aiProgressShown += gap * (1 - Math.exp(-dt / 0.18));
  if (Math.abs(gap) < 0.002) aiProgressShown = aiProgressTarget;
  const pct = Math.max(0, Math.min(100, Math.round(aiProgressShown * 100)));
  const key = `${pct}|${aiStage}|${aiElapsed.toFixed(1)}`;
  if (key === aiProgressKey) return;
  aiProgressKey = key;
  $('ai-progress-fill').style.width = `${(aiProgressShown * 100).toFixed(1)}%`;
  $('ai-progress-stage').textContent = aiStage || t('replay.ai.state.analyzingEllipsis');
  $('ai-progress-time').textContent = `${pct}% · ${aiElapsed.toFixed(1)} s`;
}

/**
 * 顶部面板：**只放赢面** + 进度条。
 *
 * 只开了「AI 对枪」时面板照样出现（进度条总得有地方显示），但加上
 * `progress-only` 把赢面那行藏掉 —— 那个开关本来就没要 win_ct。
 */
function updateAiPanel(frame) {
  const spatial = $('toggle-ai-spatial').checked;
  const showWin = spatial && !!frame;
  const show = showWin || aiAnalyzing;
  if (!show) {
    if (!$('ai-panel').classList.contains('hidden')) showAiPanel(false);
    return;
  }
  // 还没拿到这一回合的数据时（第一次分析）赢面那行没有数字可写：只留进度条，
  // 免得出现"CT — / 50% 的条 / T —"这种半成品。
  showAiPanel(true, !showWin);
  syncAiModelLabel();
  updateAiProgress(aiAnalyzing);
  if (!showWin) return;

  const ct = Math.round((frame.winCt ?? 0.5) * 100);
  const key = `w${ct}`;
  if (key === aiWinKey) return;
  aiWinKey = key;
  $('ai-win-ct').textContent = `${ct}%`;
  $('ai-win-t').textContent = `${100 - ct}%`;
  $('ai-bar-ct').style.width = `${ct}%`;
}

/**
 * 左下角的选手指标表：每行一个选手、五格概率。
 *
 * 行只在**换回合 / 换比赛**时重建（队名、选手名、左右分组都可能变），之后每帧
 * 只改数字和细条宽度 —— 而且只有取整后的百分比变了才动 DOM。
 */
function ensureAiTable() {
  if (aiTableFor === aiRoundFor) return;
  aiTableFor = aiRoundFor;
  aiRowKeys.length = 0;
  const body = $('ai-table-body');
  if (!body) return;
  body.innerHTML = '';
  const teams = (currentRound && currentRound.teams) || [];
  const order = [];
  for (const team of ['CT', 'T']) {
    for (let i = 0; i < 10; i++) {
      if ((teams[i] === 'T' ? 'T' : 'CT') === team) order.push(i);
    }
  }
  for (const i of order) {
    const team = teams[i] === 'T' ? 'T' : 'CT';
    const row = document.createElement('div');
    row.className = `ai-row ${team === 'CT' ? 'ct' : 't'}`;
    row.dataset.player = String(i);
    const cells = ['alive', 'can_kill', 'next_kill', 'next_death', 'duel']
      .map((m) => `<div class="ai-cell" data-metric="${m}"><i></i><b>—</b></div>`)
      .join('');
    row.innerHTML = `
      <span class="ai-row-name">
        <i class="ai-row-team"></i>
        <b class="ai-row-slot">${String(i + 1).padStart(2, '0')}</b>
        <span class="ai-row-role"></span>
        <span class="ai-row-nick">${escHtml(playerName(i) || `P${i + 1}`)}</span>
      </span>${cells}`;
    body.appendChild(row);
  }
}

/** 一格：细条宽度 + 数字。`v = null` 表示"这一格没有意义"（比赛已结束 / 人已死）。 */
function setAiCell(cell, v) {
  const b = cell.querySelector('b');
  const fill = cell.querySelector('i');
  const text = v == null ? '—' : aiPct(v);
  if (b.textContent !== text) b.textContent = text;
  const w = v == null ? '0%' : `${(Math.min(1, Math.max(0, v)) * 100).toFixed(0)}%`;
  if (fill.style.width !== w) fill.style.width = w;
  cell.classList.toggle('empty', v == null);
}

/** 每帧刷新表格（只在数字变了的时候动 DOM）。 */
function updateAiTable(frame, playerStates) {
  const on = $('toggle-ai-spatial').checked && !!frame;
  const table = $('ai-table');
  if (!on) {
    if (!$('ai-table').classList.contains('hidden')) table.classList.add('hidden');
    // 表格都不显示了，悬停说明和曲线就都是"孤儿"：一起收掉
    hideAiTip();
    aiChart?.close();
    return;
  }
  table.classList.remove('hidden');
  ensureAiTable();
  const body = $('ai-table-body');
  if (!body) return;
  const duel = frame.duel;
  // 弧线上不再写名字，所以"这两个人是谁"只在这里说：悬停对枪那一格能看到
  // 「对枪：A vs B」+ 自己的胜率。
  const pair = duel && duel.names ? `${duel.names[0]} vs ${duel.names[1]}` : '';
  for (const row of body.children) {
    const p = Number(row.dataset.player);
    const state = playerStates ? playerStates[p] : null;
    const isAlive = !!(state && state.alive);
    const vals = frame.players[p] || {};
    const isVictim = !!duel && duel.a === p;
    const isKiller = !!duel && duel.b === p;
    const duelWin = isVictim ? duel.p : (isKiller ? 1 - duel.p : null);
    // 选中状态跟着左边卡片走：点名字 / 点曲线 / 点卡片之后，同一行要亮起来。
    // 它不进下面的 key —— 选中的变化和概率无关，放在 key 里会让整行每帧重写。
    const selected = p === expandedPlayerIdx;
    if (row.classList.contains('selected') !== selected) {
      row.classList.toggle('selected', selected);
    }
    row.classList.toggle('dead', !isAlive);
    row.classList.toggle('duel', !!(isVictim || isKiller));
    row.querySelector('.ai-row-role').textContent = (isVictim || isKiller) ? '⚔' : '';
    const cells = row.querySelectorAll('.ai-cell');
    if (cells.length === 5) {
      const tip = (isVictim || isKiller) && pair
        ? t('replay.ai.tip.duel', { pair, pct: aiPct(duelWin) })
        : (pair ? t('replay.ai.tip.notInDuel')
          : t('replay.ai.tip.noDuel'));
      if (cells[4].title !== tip) cells[4].title = tip;
    }
    const values = isAlive
      ? [vals.alive, vals.canKill, vals.nextKill, vals.nextDeath, duelWin]
      : [null, null, null, null, null];
    const key = values.map((v) => (v == null ? 'x' : Math.round(v * 100))).join('|')
      + (isVictim ? 'v' : isKiller ? 'k' : '');
    if (aiRowKeys[p] === key) continue;
    aiRowKeys[p] = key;
    for (let c = 0; c < cells.length; c++) setAiCell(cells[c], values[c]);
  }
}

/** 每帧：插值 → 3D 覆盖层 + 顶部面板 + 左下表格 + 曲线上的播放头。 */
function updateAiFrame(playerStates) {
  const frame = aiFrameAt(currentSampleIdx);
  // 3D 里只有对枪弧线；存活率、下一杀 / 下一死这些都在左下表格里。
  updateAiOverlay({
    players: playerStates,
    duel: frame ? frame.duel : null,
    showDuel: $('toggle-ai-duel').checked && !!frame,
  });
  updateAiPanel(frame);
  updateAiTable(frame, playerStates);
  // 曲线开着的时候，播放头跟着走（时间没变时 update() 自己会跳过重画）
  if (aiChart && aiChart.state.open) aiChart.update(currentTime);
}

// ═══════════════════════════════════════════════════════════════════════
// 主循环
// ═══════════════════════════════════════════════════════════════════════

let lastTime = performance.now();
let frameCount = 0;
let lastInspectorTick = -1;

/**
 * 一帧：推进回放 → 相机 → 视觉层 → 面板。
 *
 * 从 `animate()` 里拆出来是为了让测试能用**固定 dt** 驱动它（`__v5.stepFrame`）：
 * headless 里 rAF 只有 1~3 fps，而 dt 又被 `if (dt > 0.25) dt = 0.25` 削平，
 * 按墙钟量相机跟随会量到帧率而不是跟随本身。在同一个 JS 任务里连调它，
 * rAF 插不进来，推进量就是确定的。
 *
 * @param {number} dt   真实 dt（秒）
 * @param {{draw?: boolean}} [opts] `draw: false` = 不碰 GPU（相机的位移计算在
 *   `renderFrame` 里，和绘不绘制无关），大批量步进时用它免得一次几百毫秒。
 */
function runFrame(dt, opts = {}) {
  const draw = opts.draw !== false;
  // 曳光按 (上一帧 tick, 这一帧 tick] 的开火事件生成，所以要在推进**之前**
  // 记住上一帧的 tick
  const prevTick = currentTick();
  const result = advanceReplay(dt);
  if (result.advanced) {
    handleReplayEvents(result.events);
    spawnTracersBetween(prevTick, currentTick());
    drawTimeline();
  }
  if (result.roundEnded) {
    setPlaying(false);
    updatePlayButton();
  }

  renderFrame(dt, (idx) => getPlayerState(idx, currentSampleIdx), { skipRender: !draw });
  updateAllVisuals();
  updateTimeDisplay();
  updateSmokeAnimations(dt, currentTime);
  updateInfernoAnimations(dt, currentTime);
  updateTracers(dt);

  const tick = currentTick();
  if (tick !== lastInspectorTick) {
    lastInspectorTick = tick;
    if (currentRound) renderRoundInspector();
  }

  if (draw && frameCount % 3 === 0 && matchData && currentRound) {
    try {
      renderMinimap(getAllPlayerStates(currentSampleIdx), currentRound.teams, getBombPosition(currentSampleIdx));
    } catch (_) {}
  }
  frameCount++;
}

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  let dt = (now - lastTime) / 1000;
  lastTime = now;
  if (dt > 0.25) dt = 0.25;

  try {
    runFrame(dt);
  } catch (err) {
    if (frameCount < 5 || frameCount % 600 === 0) console.warn('[v5] render error:', err);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 事件绑定
// ═══════════════════════════════════════════════════════════════════════

function bindEvents() {
  // 回放
  $('btn-play').addEventListener('click', togglePlay);
  $('speed-select').addEventListener('change', (e) => setPlaySpeed(parseFloat(e.target.value)));
  $('btn-prev-round').addEventListener('click', () => {
    if (activeRoundIdx > 0) setRound(activeRoundIdx - 1);
  });
  $('btn-next-round').addEventListener('click', () => {
    if (activeRoundIdx < totalRounds - 1) setRound(activeRoundIdx + 1);
  });
  $('round-select').addEventListener('change', (e) => setRound(parseInt(e.target.value)));

  // 重置视角 = 退出第一人称（如果有）并回到默认 orbit 机位
  $('btn-reset-cam').addEventListener('click', () => {
    if (isFirstPerson()) exitFirstPerson({ restore: false, toast: false, refresh: false });
    resetCamera('free');
  });
  // HUD 上的退出按钮
  $('fp-exit').addEventListener('click', () => exitFirstPerson({ toast: true }));

  // 相机跟随开关（默认开）。取消勾选 = 立刻停止当前跟随；重新勾选不会自己
  // 开始跟，要再点一次玩家卡片。
  $('toggle-cam-follow').addEventListener('change', (e) => {
    followEnabled = e.target.checked;
    if (!followEnabled) setFollowPlayer(-1);
  });

  $('btn-download').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/download');
      if (!res.ok) throw new Error((await res.json()).error || 'HTTP ' + res.status);
      const blob = await res.blob();
      const a = document.createElement('a');
      const url = URL.createObjectURL(blob);
      a.href = url;
      a.download = (loadedSource.replace(/\.[^.]+$/, '') || 'replay') + '.csz';
      a.click();
      URL.revokeObjectURL(url);
      showToast(t('replay.toast.downloaded'));
    } catch (err) {
      showToast(t('replay.toast.downloadFail', { msg: err.message }), true);
    }
  });

  // 开关只影响画面，重算一次即时的几何/材质。
  const refreshVisuals = () => updateAllVisuals();
  $('toggle-trails').addEventListener('change', (e) => setTrailsVisible(e.target.checked));
  $('toggle-names').addEventListener('change', (e) => setNamesVisible(e.target.checked));
  $('toggle-player-ids').addEventListener('change', (e) => setPlayerIdsVisible(e.target.checked));
  // 浏览器可能恢复上次的表单状态（刷新 / 后退）：让模块状态跟 DOM 对齐，
  // 否则勾选框显示"关"、头顶却还挂着编号。
  setPlayerIdsVisible($('toggle-player-ids').checked);
  $('toggle-smokes').addEventListener('change', refreshVisuals);
  $('toggle-fire').addEventListener('change', refreshVisuals);
  $('toggle-solid').addEventListener('change', refreshVisuals);
  // 真实资产层：sprite 只影响画面，图标/模型开关立刻生效
  $('toggle-smoke-sprites').addEventListener('change', (e) => {
    setSmokeSpritesVisible(e.target.checked);
    updateAllVisuals();
  });
  $('toggle-fire-sprites').addEventListener('change', (e) => {
    setFireSpritesVisible(e.target.checked);
    updateAllVisuals();
  });
  $('toggle-weapon-models').addEventListener('change', refreshVisuals);
  // 投掷物模型：关掉 = 退回改造前那个圆点（不是不画），所以不用重建轨迹
  $('toggle-nade-models').addEventListener('change', (e) => {
    setNadeModelsVisible(e.target.checked);
    updateAllVisuals();
  });
  $('toggle-weapon-icons').addEventListener('change', (e) => {
    setWeaponIconsVisible(e.target.checked);
  });
  $('toggle-crouch').addEventListener('change', (e) => {
    setCrouchDebugVisible(e.target.checked);
    updateAllVisuals();
  });
  $('toggle-aim').addEventListener('change', async (e) => {
    if (e.target.checked) {
      // 地图 BVH 是懒建的：823k 三角形要一两秒。建的时候先把按钮禁用，
      // 免得用户以为卡死了，建完再画。
      const label = e.target.closest('label');
      if (label) label.classList.add('busy');
      try {
        const bvh = await ensureMapBVH();
        if (!bvh) console.warn('[aim] 地图还没加载好，射线只会截断到玩家');
      } finally {
        if (label) label.classList.remove('busy');
      }
    }
    setAimRaysVisible(e.target.checked);
    updateAllVisuals();
  });
  $('toggle-tracers').addEventListener('change', async (e) => {
    setTracersVisible(e.target.checked);
    if (!e.target.checked) return;
    // 曳光要在墙上停下来就得有 BVH。和瞄准射线一样是懒建的：建的时候先把
    // 标签标成 busy，免得用户以为卡死了。建好之前曳光只截断到玩家。
    const label = e.target.closest('label');
    if (label) label.classList.add('busy');
    try {
      const bvh = await ensureMapBVH();
      if (!bvh) console.warn('[tracer] 地图还没加载好，曳光只会截断到玩家');
    } finally {
      if (label) label.classList.remove('busy');
    }
  });
  $('toggle-xray').addEventListener('change', (e) => setXRayMode(e.target.checked));

  // 选手 X 光：默认开，所以这里多半是"关掉"。判定要有地图 BVH 才知道谁被墙
  // 挡着（BVH 是懒建的，823k 三角形要一两秒），和瞄准射线一样先标 busy。
  $('toggle-player-xray').addEventListener('change', async (e) => {
    showPlayerXray = e.target.checked;
    setPlayerXRayMode(showPlayerXray);
    if (showPlayerXray) await ensureVisBVH(e.target.closest('label'));
    updateAllVisuals();
  });

  // 眼睛图标：**几何驱动**，所以要有地图 BVH 才准（不然只能按视锥 + 烟雾判）。
  // BVH 是懒建的（823k 三角形要一两秒），和瞄准射线一样先标 busy。
  $('toggle-spotted').addEventListener('change', async (e) => {
    showSpottedIcons = e.target.checked;
    setSpotIconsVisible(showSpottedIcons);
    if (showSpottedIcons) await ensureVisBVH(e.target.closest('label'));
    updateAllVisuals();
  });
  // 视锥只看朝向，不需要 BVH —— 所以是同步的，立刻生效
  $('toggle-frustum').addEventListener('change', (e) => {
    showFrustum = e.target.checked;
    setViewFrustumVisible(showFrustum);
    updateAllVisuals();
  });
  $('toggle-vis-corners').addEventListener('change', async (e) => {
    geoVisCorners = e.target.checked;
    setVisCornerRaysVisible(geoVisCorners);
    if (geoVisCorners) await ensureVisBVH(e.target.closest('label'));
    updateAllVisuals();
  });

  // AI 预测：打开任意一个就把**当前回合**的预测拉下来（已经算过就是读缓存）。
  // 关掉只是不画，`aiRound` 留着 —— 再打开不用重新等。
  for (const id of ['toggle-ai-spatial', 'toggle-ai-duel']) {
    $(id).addEventListener('change', () => {
      if (aiWanted()) ensureAiRound();
      updateAllVisuals();
    });
  }

  // ── 左下指标表上的三种交互 ────────────────────────────────────────────
  // `.ai-table` 整块 pointer-events:none，事件只从列头 / 名字 / 数字上冒上来。
  const aiCols = document.querySelector('#ai-table .ai-table-cols');
  aiCols?.addEventListener('pointerover', (e) => {
    const col = e.target.closest('.ai-col, .ai-col-name');
    if (col) showAiTip(col);
  });
  aiCols?.addEventListener('pointerout', (e) => {
    if (!e.target.closest('.ai-col, .ai-col-name')) return;
    const to = e.relatedTarget;
    // 从一列直接挪到下一列时也会先 out：别把刚弹出的下一张释义闪掉
    if (to && to.closest && to.closest('.ai-col, .ai-col-name')) return;
    hideAiTip();
  });
  $('ai-table-body').addEventListener('click', (e) => {
    const row = e.target.closest('.ai-row');
    if (!row) return;
    const idx = Number(row.dataset.player);
    if (!Number.isInteger(idx)) return;
    if (e.target.closest('.ai-row-name')) {
      hideAiTip();
      focusPlayerFromAi(idx);
      return;
    }
    const cell = e.target.closest('.ai-cell');
    if (!cell) return;
    if (cell.dataset.metric === 'duel') {
      // 对枪只有死前 3 秒有值：一张整回合的曲线在这里没有意义
      showToast(t('replay.toast.duelNoChart'), true);
      return;
    }
    toggleAiChart(idx, cell.dataset.metric);
  });
  window.addEventListener('resize', () => {
    hideAiTip();
    // 曲线是贴着表格上方摆的：窗口一变就得重新量一次
    if (aiChart && aiChart.state.open) aiChart.place();
  });

  // ── 顶部面板的胜率：同一张曲线卡，不锁镜头 ─────────────────────────────
  // 两个数字是 `.ai-panel` 里**唯一**接鼠标的元素（面板本身 pointer-events:none）。
  const aiWin = document.querySelector('#ai-panel .ai-win');
  aiWin?.addEventListener('pointerover', (e) => {
    const el = e.target.closest('.ai-pct[data-metric]');
    if (el) showAiTip(el);
  });
  aiWin?.addEventListener('pointerout', (e) => {
    if (!e.target.closest('.ai-pct[data-metric]')) return;
    const to = e.relatedTarget;
    // 从 CT 直接挪到 T 时也会先 out：别把刚弹出的下一张释义闪掉
    if (to && to.closest && to.closest('.ai-pct[data-metric]')) return;
    hideAiTip();
  });
  aiWin?.addEventListener('click', (e) => {
    const el = e.target.closest('.ai-pct[data-metric]');
    if (el) toggleAiChart(-1, el.dataset.metric);
  });
  // tabindex="0" 给了键盘用户一条同样的路（Enter / 空格 = 点一下）。
  // 必须 stopPropagation：document 的 Space 是"播放/暂停"，不拦住会同时触发。
  aiWin?.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const el = e.target.closest('.ai-pct[data-metric]');
    if (!el) return;
    e.preventDefault();
    e.stopPropagation();
    toggleAiChart(-1, el.dataset.metric);
  });

  // 比赛统计弹窗
  $('btn-stats').addEventListener('click', openStats);
  $('btn-stats-close').addEventListener('click', closeStats);
  $('btn-stats-share').addEventListener('click', shareMatchImage);
  // 点弹窗外面的暗色区域也算关闭（点卡片内部不关）
  $('stats-modal').addEventListener('click', (e) => {
    if (e.target === $('stats-modal')) closeStats();
  });
  // 两个页签（传统数据 / AI 数据）。AI 那一页在 switchStatsTab 里才发请求。
  document.querySelectorAll('.stats-tab').forEach((btn) => {
    btn.addEventListener('click', () => switchStatsTab(btn.dataset.tab));
  });
  // 点回合号 = 跳到那一回合。矩阵表头和单回合明细上面那排数字是同一个东西，
  // 点谁都是"回放跳过去 + 所有表跟着换"。弹窗**不关**：关掉就看不到"一起变"
  // 了，而且明细表就是拿来一回合回合翻的。要看画面按 ✕ / Esc。
  $('stats-body').addEventListener('click', (e) => {
    const col = e.target.closest('[data-round-col]');
    if (col && currentRound) {
      const idx = Number(col.dataset.roundCol);
      // 先挪高亮（round 加载是异步的），再把回放跳过去
      paintActiveRound(idx);
      if (idx !== activeRoundIdx) setRound(idx);
      return;
    }
    if (e.target.closest('#stats-ai-retry')) loadAiStats(true);
  });
  mountTips();

  // ── 小地图：点一下放大成 2D 回放 ──────────────────────────────────────
  // 放大的是**同一块画布、同一个正交相机**：scene.js::renderMinimap 每帧按显示
  // 尺寸同步后备分辨率，所以变的只是容器大小 —— 位置 / 朝向不用另算一套，
  // 3D 场景照常在后面跑，收起来就回到角落那块小地图。
  //
  // 放大态还多两件事（`setMinimapMode2D`）：取景锁死成整张地图、不跟着存活玩家
  // 缩放平移；标记换成小一号的固定尺寸 + 每个箭头旁边写上名字。这是一块「回放」
  // 而不是「跟着镜头走的雷达」，位置关系要稳定，箭头也不能长得把房间盖住。
  const mmBox = $('minimap-container');
  const mmIsExpanded = () => mmBox.classList.contains('mm-expanded');
  /** 放大态是正方形：视口短边留 32px（雷达底图是方的，拉伸只会露出图外） */
  const layoutMinimap = () => {
    if (!mmIsExpanded()) return;
    const vp = $('viewport');
    const side = Math.max(240, Math.min(vp.clientWidth, vp.clientHeight) - 32);
    mmBox.style.width = `${side}px`;
    mmBox.style.height = `${side}px`;
  };
  const setMinimapExpanded = (on) => {
    if (on === mmIsExpanded()) return;
    mmBox.classList.toggle('mm-expanded', on);
    $('viewport').classList.toggle('mm-2d', on);
    setMinimapMode2D(on);
    if (on) {
      // 第一人称的准星 / 血量 HUD 是压在 3D 视野上的，2D 地图一打开就没意义了 ——
      // 先退出去（机位不还原，收起来还是刚才那个位置）。
      if (isFirstPerson()) exitFirstPerson({ restore: false, toast: false, refresh: false });
      // 曲线卡也贴在视口里，放大后会被地图整块盖住 —— 先收掉，别留一张看不见的卡
      if (aiChart && aiChart.state.open) aiChart.close();
      layoutMinimap();
    } else {
      mmBox.style.width = '';
      mmBox.style.height = '';
      mmBox.style.cursor = '';
    }
  };

  /**
   * 在 2D 回放里点中一名选手：和点左边那张卡片**同一个动作**（展开他的详情卡 +
   * 镜头飞过去并开始跟随），然后把 2D 视图收起来 —— 镜头是在 3D 那边飞的，地图
   * 开着的话你只能看到一块 0.16 透明度的背景，等于没跳。第一人称是锁定模式，
   * 和点卡片一样只弹提示、不换目标。
   */
  const focusPlayerFromMinimap = (idx) => {
    if (isFirstPerson()) {
      showToast(t('replay.toast.fpLocked'), true);
      return;
    }
    togglePlayerDetail(idx, true);
    focusCameraOnPlayer(idx);
    setMinimapExpanded(false);
  };

  mmBox.addEventListener('click', (e) => {
    if (mmIsExpanded()) {
      // 放大态：点箭头 / 名字 = 选中这名选手，点空白处才是收起。坐标要按
      // `#minimap-overlay` 算 —— 容器有 1px 边框，直接拿容器的 rect 会差一点。
      const r = $('minimap-overlay').getBoundingClientRect();
      const idx = minimapPlayerAt(e.clientX - r.left, e.clientY - r.top);
      if (idx >= 0) { focusPlayerFromMinimap(idx); return; }
    }
    setMinimapExpanded(!mmIsExpanded());
  });
  // 指到箭头 / 名字上换小手：不然没人知道这些字能点
  mmBox.addEventListener('pointermove', (e) => {
    if (!mmIsExpanded()) { mmBox.style.cursor = ''; return; }
    const r = $('minimap-overlay').getBoundingClientRect();
    mmBox.style.cursor = minimapPlayerAt(e.clientX - r.left, e.clientY - r.top) >= 0 ? 'pointer' : '';
  });
  // 容器是 role=button：键盘用户和点表格里的数字一样，Enter / 空格也能开合
  mmBox.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    e.stopPropagation();          // 空格在文档上是「播放 / 暂停」，不拦住会一起触发
    setMinimapExpanded(!mmIsExpanded());
  });
  $('minimap-close').addEventListener('click', (e) => {
    e.stopPropagation();          // 别再冒泡到容器上把它翻回去
    setMinimapExpanded(false);
  });
  window.addEventListener('resize', layoutMinimap);

  // 键盘
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Escape') closeStats();
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
    // Esc 优先退出第一人称；它和"关闭统计弹窗"共用同一个键，两个都开时一起关。
    if (e.code === 'Escape' && isFirstPerson()) { exitFirstPerson({ toast: true }); return; }
    // 再其次是放大成 2D 回放的小地图：它盖住了大半个视口
    if (e.code === 'Escape' && mmIsExpanded()) { setMinimapExpanded(false); return; }
    // 最后是曲线卡：它盖在 3D 视口上，Esc 关掉比去够右上角的 ✕ 快
    if (e.code === 'Escape' && aiChart && aiChart.state.open) { aiChart.close(); return; }
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    if (e.code === 'ArrowLeft' && currentRound) seekTo(Math.max(0, currentTime - 1));
    if (e.code === 'ArrowRight' && currentRound) seekTo(Math.min(getRoundDuration(), currentTime + 1));
    if (e.code === 'KeyR') {
      if (isFirstPerson()) exitFirstPerson({ restore: false, toast: false, refresh: false });
      resetCamera('free');
    }
  });

  // 点击玩家
  const vpCanvas = $('three-canvas');
  let down = null;
  vpCanvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
  vpCanvas.addEventListener('pointerup', (e) => {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    down = null;
    if (dx * dx + dy * dy > 25) return;
    if (cameraMode === 'fly') return;
    // 第一人称下点 3D 视口不再换人：换目标会把镜头带离当前选手，
    // 和"进入后无法移动相机"冲突。退出后再点。
    if (isFirstPerson()) return;
    pickPlayerAt(e.clientX, e.clientY);
  });

  // 时间线 seek：点一下跳过去，按住拖动实时跟手。
  //
  // 拖动时每次 `pointermove` 都是一次 `seekTo()`，一帧里可能跳几十个采样点 ——
  // 轨迹以前就是在这里坏掉的（见 visuals.js 的「轨迹」注释），修完之后每一次
  // 跳转都按历史重算，拖多快都不会拖出一条横穿地图的线。
  const tl = $('timeline-canvas');
  let scrubbing = false;

  const seekFromPointer = (e) => {
    if (!currentRound) return;
    const rect = tl.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    seekTo(f * getRoundDuration());
    drawTimeline();
  };

  tl.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || !currentRound) return;
    scrubbing = true;
    // 按住拖动期间先暂停，否则播放头会和手指抢同一个 currentTime
    if (isPlaying) { setPlaying(false); updatePlayButton(); }
    // 捕获指针：拖到时间线外面（甚至窗口外）也继续跟手，松手才算结束
    try { tl.setPointerCapture(e.pointerId); } catch (_) { /* 老浏览器 */ }
    seekFromPointer(e);
  });
  tl.addEventListener('pointermove', (e) => { if (scrubbing) seekFromPointer(e); });
  const endScrub = (e) => {
    if (!scrubbing) return;
    scrubbing = false;
    try { tl.releasePointerCapture(e.pointerId); } catch (_) { /* 已经释放 */ }
  };
  tl.addEventListener('pointerup', endScrub);
  tl.addEventListener('pointercancel', endScrub);
}

function fitCameraToMap() {
  const bounds = getMapBounds();
  if (!bounds) return;
  // 地图是几百毫秒 ~ 几秒后才加载完的。用户已经选中了某个玩家（镜头正跟着他）
  // 时不要抢镜头 —— 那次自动"把整张图装进画面"会把跟随直接掀掉。
  if (getFollowPlayer() >= 0) return;
  // 第一人称的相机由 scene.js 每帧按选手状态接管，这里抢镜头会让视角瞬移。
  if (isFirstPerson()) return;
  const center = new THREE.Vector3();
  bounds.getCenter(center);
  const size = new THREE.Vector3();
  bounds.getSize(size);
  const maxDim = Math.max(size.x, size.z, 1);
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
    $('loading-overlay-text').textContent = text || t('common.action.loading');
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
    // WASD 手动推机位 = 用户自己接管镜头：取消选中（卡片收起）+ 停止跟随。
    // 注册在 scene.js 的**移动回调**上，而不是再听一次 keydown —— 只有镜头真的
    // 被推动了才取消选中（焦点在下拉框里时不会误伤）。
    setManualMoveHandler(deselectPlayer);
    createPlayers();
    createAimRays();
    createVisCornerRays();
    createViewFrustum();
    // 轨迹的重算要读"第 i 个采样点上所有人在哪"，只有 replay-core 知道；视觉层
    // 不认识它，所以用一个回调注入（见 visuals.js 的 setTrailHistorySource）。
    // 拖时间轴之后按历史整条重算，轨迹才不会把跳转前后的位置连成一条线。
    setTrailHistorySource(rawPositionsAt);
  } catch (err) {
    console.error('[v5] 3D init failed:', err);
    showToast(t('replay.toast.initFail', { msg: err.message }), true);
  }
  // 真实 sprite 序列（烟雾 128 帧 + 火焰三张）体积不小，趁用户还在选文件就先拉。
  // 不 await：拿不到就退回程序生成的兜底贴图，不阻塞启动。
  preloadSpriteSequences([
    'smokeloop_i_0_sc_hardedge', 'flames', 'flame_omni', 'flame_wisp',
  ]);
  // AI 模型在不在（只扫目录，不加载权重）：不可用就把两个开关禁掉并说明原因。
  loadAiStatus().catch(() => {});
  // 序列到齐后预热火焰/烟雾的 shader。不做的话，第一次出现火的那一帧要同步
  // 编译 3 个 program，停顿正好落在"火刚烧起来"的瞬间。详见 prewarmFireShaders。
  prewarmFireShaders().then((ok) => {
    if (ok) console.info('[v5] 火焰 / 烟雾 shader 已预热');
  }).catch(() => {});
  // 投掷物模型（6 个，GLB ~4.5 MB + 贴图 ~10 MB）。预热是为了**第一个回合开始
  // 播放时飞行中的投掷物就已经是真模型**，而不是先闪一下兜底球。
  // 失败无所谓：`_installNadeModel` 会保留那个球。
  preloadNadeModels().then((ms) => {
    const ok = ms.filter(Boolean).length;
    console.info(`[v5] 投掷物模型已预热 ${ok}/${ms.length}`);
  }).catch(() => {});
  // 服务端已经有一份解析好的数据（比如刚才那次 .dem 提取）→ 直接接管，不用重传
  // Demo 库批次优先：?batch=<id> 时由它决定加载哪一场。
  initBatch().then((handled) => {
    if (handled) return;
    fetch('/api/session').then(r => r.json()).then((st) => {
      if (st && st.loaded) {
        showToast(t('replay.toast.session', { source: st.source, sampling: st.sampling }));
        onDataReady(st);
      }
    }).catch(() => {});
  }).catch(() => {});
  requestAnimationFrame(animate);
  // 调试钩子：控制台里可以直接查场景 / 当前回合（写测试脚本也用得上）
  window.__v5 = {
    get scene() { return scene; },
    get camera() { return camera; },
    // OrbitControls：截图脚本要自己摆机位（拍 AI 弧线、拍某个玩家）时用它
    get controls() { return controls; },
    // 小地图渲染器：截图脚本用它把 256px 的画布临时放大，拍出能看清对齐的照片
    get minimapRenderer() { return minimapRenderer; },
    get minimapCamera() { return minimapCamera; },
    get radar() { return currentMinimapRadar(); },
    /**
     * 同步重画一次小地图并返回它的 dataURL。
     *
     * 截图脚本必须这么取：小地图是 WebGL 画布且没开 preserveDrawingBuffer，
     * 隔一个 JS 任务再去读就是空的/旧的；在同一个任务里画完立刻读才拿到真帧。
     */
    redrawMinimap: () => {
      if (!currentRound) return null;
      renderMinimap(
        getAllPlayerStates(currentSampleIdx),
        currentRound.teams,
        getBombPosition(currentSampleIdx),
      );
      // 底图和标记分在两张画布上（`#minimap-canvas` 是 WebGL，`#minimap-overlay`
      // 是 2D），叠一下再导出，否则截出来的图没有玩家箭头。
      const base = document.getElementById('minimap-canvas');
      const overlay = document.getElementById('minimap-overlay');
      if (!overlay) return base.toDataURL('image/png');
      const c = document.createElement('canvas');
      c.width = base.width;
      c.height = base.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(base, 0, 0);
      ctx.drawImage(overlay, 0, 0);
      return c.toDataURL('image/png');
    },
    /**
     * 小地图：放大态 / 容器尺寸 / 标记的实际显示尺寸（CSS 像素）。
     * 测试用它核对「点一下真的放大了」和「炸弹比箭头小」。
     */
    get minimap() {
      const box = $('minimap-container');
      const r = box.getBoundingClientRect();
      return {
        expanded: box.classList.contains('mm-expanded'),
        size: [Math.round(r.width), Math.round(r.height)],
        canvas: $('minimap-canvas').width,
        overlay: $('minimap-overlay').width,
        glyph: minimapGlyphMetrics(),
        // 2D 回放里每个箭头 + 名字的点击热区（相对画布的 CSS 像素）：测试照着它点
        hits: minimapHitDebug(),
      };
    },
    get round() { return currentRound; },
    get summary() { return matchData; },
    // 地图配色来源：'game' = 用 CS2 材质烘的颜色，'synthetic' = 退回水泥灰
    get mapColors() { return getMapColorStats(); },
    // 换一套配色（换调色板时用）：只重涂顶点色，不重载 130 MB 的 OBJ
    mapColorsFrom: (name) => reloadMapColors(name),
    get smoke() { return smokeDebugStats(); },
    get fire() { return fireDebugStats(); },
    /**
     * AI 预测：状态 / 当前回合的载荷概要 / 插值后的这一帧 / 3D 层真实画了什么。
     *
     * `frame.duel = {a, b, p, t0, t1, fade, names}`，其中 `a` 是**这场对枪
     * 会死的那个**（弧线靠近他的那一端）、`p = P(a 赢)` —— 弧线就是从 a 到 b
     * 按它切成两段落色的，数字在左下表格的「对枪」那一格。
     * `frame.players[p] = {alive, canKill, nextDeath, nextKill}` 是指标表的五列。
     */
    get ai() {
      const frame = aiFrameAt(currentSampleIdx);
      return {
        status: aiStatus,
        round: aiRound ? {
          round: aiRound.round,
          stride: aiRound.stride,
          samples: aiRound.samples.length,
          cached: aiRound.cached,
          ms: aiRound.ms,
          duels: (aiRound.duels || []).length,
          points: aiRound.duel.a.filter((v) => v != null).length,
        } : null,
        loadedFor: aiRoundFor,
        activeRound: activeRoundIdx,
        analyzing: aiAnalyzing,
        progress: {
          target: aiProgressTarget,
          shown: aiProgressShown,
          stage: aiStage,
          elapsed: aiElapsed,
        },
        frame,
        overlay: aiOverlayDebug(),
        // 表头释义 / 曲线卡：测试要能读到"弹没弹、画没画"，而不只是状态变量
        tip: aiTipMetric
          ? {
            metric: aiTipMetric,
            title: $('ai-tip-title').textContent,
            body: $('ai-tip-body').textContent,
            hint: $('ai-tip-hint').textContent,
            shown: $('ai-tip').classList.contains('show'),
          }
          : null,
        chart: aiChart ? aiChart.state : { open: false },
      };
    },
    /** 测试 / 调试：手动保证当前回合有预测（`true` = 跳过缓存重算）。 */
    ensureAi: (force = false) => ensureAiRound(!!force),
    // 真实资产层：清单/图标/序列加载状态 + 每个玩家挂的武器模型
    get assets() { return assetDebugStats(); },
    get weapons() { return weaponModelStats(); },
    // 飞行中的投掷物：多少条轨迹用上了真模型 / 还在等 / 没有映射
    get nades() { return nadeModelDebugStats(); },
    // 给脚本用的驱动入口：加载一个本机文件、跳到某个采样点、读当前玩家状态
    loadPath: (p) => startLoad(p),
    seekSample: (i) => {
      if (!currentRound) return false;
      seekTo(currentRound.round_seconds[
        Math.min(Math.max(0, i | 0), currentRound.ticks.length - 1)] || 0);
      updateAllVisuals();
      return true;
    },
    get players() { return currentRound ? getAllPlayerStates(currentSampleIdx) : []; },
    /** 头顶的选手编号牌（「选手 ID」开关）：文字 / 显隐 / 是否 Sprite / 位置。 */
    get playerIds() { return playerIdDebug(); },
    /**
     * 玩家轨迹（`轨迹` 开关画的那条折线）：每条的画出来的点 / 点数 / 对应的采样点。
     *
     * 拖时间轴之后这里必须是"按历史重算过的"轨迹 —— 同一条轨迹同一个采样点，
     * 换一种到达方式（一路播过来 / 直接跳过来）读出来的点要一模一样。
     */
    get trails() { return trailDebugStats(); },
    get sampleIdx() { return currentSampleIdx; },
    get currentTime() { return currentTime; },
    get playSpeed() { return playSpeed; },
    get isPlaying() { return isPlaying; },
    /**
     * 测试用：手动按固定 dt 推进回放（一次调用 = 一帧）。
     *
     * 和 `tickTracers` 同一个套路：主循环也在按真实 dt 推进，但测试是在**同一个
     * JS 任务里**连着调它，rAF 插不进来 —— 所以推进量是确定性的，不受 headless
     * 帧率影响。量倍速只能这么量：headless + swiftshader 只有 ~2 fps，而主循环
     * 里有 `if (dt > 0.25) dt = 0.25` 防掉帧跳太远，按墙钟量出来的高倍速会被
     * 这个 clamp 削掉。
     */
    stepReplay: (dt) => {
      if (!currentRound) return null;
      const before = currentTime;
      const r = advanceReplay(dt);
      updateAllVisuals();
      updateTimeDisplay();
      return {
        delta: currentTime - before,
        time: currentTime,
        sample: currentSampleIdx,
        advanced: r.advanced,
        roundEnded: r.roundEnded,
      };
    },
    /**
     * 测试用：按固定 dt 走**一帧完整主循环**（和 animate 里那一趟一模一样，
     * 见 runFrame）—— 只有它会把相机一起推。相机跟随要量「人和镜头的位移是不是
     * 同一个」，必须这么走：headless 下 rAF 只有 1~3 fps。
     *
     * 第二个参数传 false = 这一帧不画（相机逻辑照跑，只是省掉一次几百毫秒的
     * 绘制），大批量步进时用。
     */
    stepFrame: (dt, draw = true) => {
      runFrame(dt || 1 / 60, { draw: draw !== false });
      return {
        time: currentTime,
        sample: currentSampleIdx,
        cameraAnimating: isCameraAnimating(),
        camera: [camera.position.x, camera.position.y, camera.position.z],
        target: [controls.target.x, controls.target.y, controls.target.z],
      };
    },
    // 相机模式（第一人称的眼高也要跟着下蹲走，见 scene.js）
    setCameraMode, setFocusedPlayer,
    get mapBVHStats() { return getMapBVHStats(); },
    /**
     * 直接问一次地图 BVH（不经过渲染管线）。测试用它当"独立参考"，
     * 验证画出来的射线终点确实落在 BVH 的命中点上。
     */
    rawRaycast: (ox, oy, oz, dx, dy, dz, maxDist) => {
      const bvh = getMapBVH();
      return bvh ? raycastBVH(bvh, ox, oy, oz, dx, dy, dz, maxDist) : -1;
    },
    /** 画出来的瞄准射线（真实几何，不是数据）：起点 / 终点 / 命中类型 / 颜色 */
    get aimRays() {
      const out = [];
      for (let i = 0; i < 10; i++) {
        const r = aimRayAt(i);
        if (!r) { out.push(null); continue; }
        const a = r.line.geometry.attributes.position;
        out.push({
          visible: r.line.visible,
          start: a ? [a.getX(0), a.getY(0), a.getZ(0)] : null,
          end: a ? [a.getX(1), a.getY(1), a.getZ(1)] : null,
          hitPlayer: r.hitPlayer != null ? r.hitPlayer : -1,
          distance: r.distance != null ? r.distance : -1,
          color: r.line.material.color.getHex(),
          markVisible: r.mark.visible,
          markPos: [r.mark.position.x, r.mark.position.y, r.mark.position.z],
        });
      }
      return out;
    },
    // ── 子弹曳光 ────────────────────────────────────────────────────────
    /** 当前在飞的曳光（真实几何：起点 / 方向 / 已飞距离 / 命中类型 / 段长） */
    get tracers() { return tracerDebugStats(); },
    /** 本回合的开火事件（精确 tick + 开枪者）；老 .csz 没有就是空表 */
    get fireEvents() { return roundFireEvents || []; },
    /** 测试用：手动让某个玩家在当前采样点打一发曳光，返回它的调试信息 */
    spawnTracer: (playerIdx) => {
      if (!currentRound) return null;
      const tracer = spawnTracer(getAllPlayerStates(currentSampleIdx), playerIdx);
      return tracer ? tracerInfo(tracer) : null;
    },
    /** 测试用：走一遍主循环里的"这段 tick 里有谁开了枪"逻辑，返回生成条数 */
    stepTracers: (fromTick, toTick) => spawnTracersBetween(fromTick, toTick),
    /**
     * 测试用：手动按固定 dt 推进曳光（一次调用 = 一帧）。
     * 主循环也在按真实 dt 推进，但测试是在**同一个 JS 任务里**连着调它，
     * rAF 插不进来 —— 所以动画是确定性的，不受 headless 帧率影响。
     */
    tickTracers: (dt) => { updateTracers(dt || 1 / 60); return tracerDebugStats(); },
    // 下蹲调试层：直接读每个玩家画出来的碰撞盒高度 / 眼高，验证渲染而不只看数据
    get crouchDebug() {
      const out = [];
      for (let i = 0; i < 10; i++) {
        const pg = playerGroupAt(i);
        if (!pg || !pg.hull) { out.push(null); continue; }
        // 小人的世界坐标：身体组被非等比压缩过，所以量头顶要用世界包围盒，
        // 不能只看 scale。注意要减掉 group 的世界 y（= 地图高度），
        // 否则量到的是"海拔"而不是"离地多高"。
        const headW = pg.head.getWorldPosition(new THREE.Vector3());
        const bodyScale = pg.body.scale.y;
        out.push({
          hullH: pg.hull.scale.y / SCALE,
          hullBaseY: (pg.hull.position.y - pg.hull.scale.y / 2) / SCALE,
          eyeY: pg.eye.position.y / SCALE,
          rulerY: pg.ruler.scale.y / SCALE,
          visible: pg.hull.visible && pg.eye.visible && pg.ruler.visible,
          bodyRatio: bodyScale,
          figureH: bodyScale * 72,                       // 压缩后的名义身高（HU）
          headY: (headW.y - pg.group.position.y) / SCALE, // 头心离地高度（HU）
          headScaleY: pg.head.scale.y,                   // 必须恒为 1：头不能被压扁
          hpY: pg.hpBg.position.y / SCALE,               // 血条离地高度（HU）
          // 步态：直接读画出来的腿
          legSwingL: pg.legPivotL.rotation.x,             // 弧度，+ 是往前
          legSwingR: pg.legPivotR.rotation.x,
          hipY: pg.legPivotL.position.y / SCALE,          // 胯高（HU）
          // 两只脚各自离地多高、前后差多远（HU）—— 用来验证"真的在迈步"。
          // 注意不能用 getWorldPosition()：那给的是腿的**中心**。脚底是腿网格
          // 局部坐标里的 (0, -P_LEG_HALF, 0)，要走完整条链（含网格自己的 scale.y）。
          ...(() => {
            const V = pg.group.position.constructor;
            // 脚底 = 腿网格局部坐标的 (0, -P_LEG_HALF, 0)，要走完整条链
            // （含网格自己的 scale.y）。不能用 getWorldPosition()，那给的是腿的**中心**。
            const fl = pg.legL.localToWorld(new V(0, -0.39, 0));
            const fr = pg.legR.localToWorld(new V(0, -0.39, 0));
            const gy = pg.group.position.y;
            // 前后差必须在**模型局部坐标**里量：group 是按 yaw 转过的，
            // 世界 z 差里混了 yaw（还有 ±0.13 的左右腿间距），会看出一个假相移。
            const ll = pg.group.worldToLocal(fl.clone());
            const lr = pg.group.worldToLocal(fr.clone());
            return {
              footL: (fl.y - gy) / SCALE,
              footR: (fr.y - gy) / SCALE,
              footSpread: (ll.z - lr.z) / SCALE,          // 沿模型正前方的前后差
              footLateral: (ll.x - lr.x) / SCALE,         // 左右间距（应该恒定 -0.26 m）
            };
          })(),
        });
      }
      return out;
    },
    // ── 「可见敌人」眼睛 ─────────────────────────────────────────────────
    /** 左侧卡片里当前选中的玩家下标；-1 = 没选中（此时不画任何眼睛） */
    get selectedPlayer() { return expandedPlayerIdx; },
    /** 测试用：等价于点一下某张玩家卡（选中 + 展开详情），传 -1 取消选中 */
    selectPlayer: (i) => {
      if (i < 0) { togglePlayerDetail(expandedPlayerIdx); }
      else { togglePlayerDetail(i, true); }
      updateAllVisuals();
      return expandedPlayerIdx;
    },

    // ── 第一人称摄像机（左侧卡片 → 摄像机 → 第一人称摄像机）──────────────
    firstPerson: {
      /** 正在观察的玩家下标；-1 = 没开 */
      get player() { return firstPersonIdx; },
      get active() { return isFirstPerson(); },
      /** 镜头真的锁住了吗：first 模式 + OrbitControls 已禁用 */
      get locked() { return cameraMode === 'first' && controls.enabled === false; },
      /** HUD 是否可见（DOM 状态，不是变量） */
      get hudVisible() {
        const el = $('fp-hud');
        return !!el && !el.classList.contains('hidden');
      },
      /** 当前 HUD 上写着的数字（从 DOM 读，验证渲染而不只看状态） */
      get hud() {
        return {
          name: $('fp-name').textContent,
          team: $('fp-team').textContent,
          hp: $('fp-hp').textContent,
          hpFill: $('fp-hpfill').style.width,
          armor: $('fp-armor').textContent,
          helmet: $('fp-helmet').textContent,
          weapon: $('fp-weapon').textContent,
          inventory: $('fp-inventory').textContent,
        };
      },
      /** 进入某个选手的第一人称（等价于卡片下拉里选「第一人称摄像机」） */
      enter(i) { return enterFirstPerson(i); },
      /** 退出并恢复进入前的 orbit 机位 */
      exit(opts) { return exitFirstPerson(opts); },
    },

    // ── 相机跟随（选中玩家之后镜头跟着他走，见 scene.js）────────────────
    cameraFollow: {
      /** 开关（`toggle-cam-follow`，默认开） */
      get enabled() { return followEnabled; },
      /** 正在跟的玩家下标；-1 = 没在跟 */
      get player() { return getFollowPlayer(); },
      /** 真的在跟吗（orbit 模式下且有人可跟） */
      get active() { return cameraMode === 'orbit' && getFollowPlayer() >= 0; },
      /** 飞向玩家那 0.7s 的动画还在跑吗 */
      get animating() { return isCameraAnimating(); },
      /** 摄像机相对视角中心的偏移；跟随期间它必须恒定 */
      get offset() {
        return [
          camera.position.x - controls.target.x,
          camera.position.y - controls.target.y,
          camera.position.z - controls.target.z,
        ];
      },
      /** 视角中心（controls.target）的世界坐标 */
      target() {
        return [controls.target.x, controls.target.y, controls.target.z];
      },
      /** 测试用：等价于点一下某张玩家卡（选中 + 飞过去 + 开始跟随） */
      focus(i) {
        if (i >= 0) togglePlayerDetail(i, true);
        focusCameraOnPlayer(i);
        return getFollowPlayer();
      },
      stop() { setFollowPlayer(-1); return getFollowPlayer(); },
    },
    /** 当前采样点下，某个玩家能看到的敌人（已按队伍过滤，升序） */
    visibleEnemiesOf: (viewer) => visibleEnemiesOf(viewer, currentSampleIdx),
    /** 当前选中玩家的可见敌人；没选中就是空表 */
    get visibleEnemies() {
      return expandedPlayerIdx >= 0 ? visibleEnemiesOf(expandedPlayerIdx, currentSampleIdx) : [];
    },
    /** 这份数据到底有没有 spotted 列（`--no-spotted` 提取的没有） */
    get hasSpotted() { return hasSpottedData(); },
    /** 画出来的眼睛图标（真实 sprite：可见性 / 世界坐标 / 高度） */
    get spotIcons() { return spotIconDebug(); },
    spotIconAt,

    // ── 几何可观测性（visibility.js）─────────────────────────────────────
    visibility: {
      /** 眼睛图标开着吗（`toggle-spotted`）。图标**只有几何这一个来源** */
      get iconsEnabled() { return showSpottedIcons; },
      /** 视锥开关（`toggle-frustum`） */
      get frustumEnabled() { return showFrustum; },
      /** 角落调试射线开关（`toggle-vis-corners`） */
      get cornersEnabled() { return geoVisCorners; },
      /** 最近一次 evaluate() 的统计：pairs / rays / mapHits / smokeHits / ms … */
      get stats() { return visStats; },
      /** 原始结果：visible / clear / reason 都是 Uint8Array，下标 `b*n+a` = 「b 看得见 a」 */
      get result() { return visResult; },
      /** 当前 tick 的烟雾遮挡体个数 */
      get occluderCount() { return visOccluders.length; },

      /**
       * 整张 10×10 矩阵，`matrix[b][a]` = 「b 看得见 a」；对角线是 null。
       * 开关没打开时会**按需算一次**（结果也会灌进 `stats` / `result`）。
       */
      matrix() {
        if (!currentRound) return null;
        if (!visResult) computeGeoVisibility(getAllPlayerStates(currentSampleIdx), currentTick());
        const r = visResult;
        const rows = [];
        for (let b = 0; b < r.n; b++) {
          const row = new Array(r.n);
          for (let a = 0; a < r.n; a++) row[a] = a === b ? null : r.visible[b * r.n + a];
          rows.push(row);
        }
        return rows;
      },
      /** 强制重算一次（不管开关） */
      recompute() {
        if (!currentRound) return null;
        computeGeoVisibility(getAllPlayerStates(currentSampleIdx), currentTick());
        return visStats;
      },
      /** 某个观察者能观测到的全部其他人（含队友，升序）。默认用当前选中的那个 */
      visibleOf(viewerIdx) {
        const i = viewerIdx == null ? expandedPlayerIdx : viewerIdx;
        return geoVisibleOf(i, visResult);
      },
      /** 每个角落的明细（给脚本/测试用）：`{viewer, target, eye, corners:[…]}` */
      pairsOf(viewerIdx) {
        if (!currentRound) return [];
        const states = getAllPlayerStates(currentSampleIdx);
        const out = [];
        for (let a = 0; a < states.length; a++) {
          if (a === viewerIdx) continue;
          if (!states[a] || !states[a].alive) continue;
          if (!states[viewerIdx] || !states[viewerIdx].alive) continue;
          const d = pairVisibility(states[viewerIdx], states[a], {
            castMapRay: visCastMapRay, occluders: visOccluders, viewerIdx, targetIdx: a,
          });
          out.push({ viewer: viewerIdx, target: a, visible: d.visible, clear: d.clear, reason: d.reason, corners: d.corners });
        }
        return out;
      },
      /** 画出来的角落调试射线（段数 / drawRange） */
      cornerRays: visCornerRayDebug,
      /** 画出来的视锥（远端 4 个角的世界坐标 / 张角 / 起点） */
      frustum: viewFrustumDebug,
    },

    // ── 选手 X 光（player-xray.js）：从**摄像机**量谁被墙挡住 ────────────
    playerXray: {
      /** 开关（`toggle-player-xray`，默认开） */
      get enabled() { return showPlayerXray; },
      /** 最近一次判定的统计：players / rays / blocked / mapHits / smokeHits / ms */
      get stats() { return playerXrayStats; },
      /** `[i] = 1` 表示第 i 个选手被挡住，只能透视看到 */
      get flags() { return Array.from(playerXrayFlags); },
      /** 判定用的摄像机位置（**game 空间**，和玩家状态同一套坐标） */
      get cameraGame() { const c = cameraGamePos(); return { x: c.x, y: c.y, z: c.z }; },
      /** 画出来的幽灵状态：读**真实材质**（不透明度 / 测不测深度 / 绘制顺序） */
      get ghosts() { return playerXRayDebug(); },
      /** 不管开关，按当前摄像机强制重算一次 */
      recompute() {
        if (!currentRound) return null;
        const tester = ensurePlayerXrayTester();
        tester.setOccluders(smokeOccludersAt(currentRound.smoke, currentTick()));
        playerXrayStats = tester.classify(
          cameraGamePos(), getAllPlayerStates(currentSampleIdx)).stats;
        setPlayerXRayFlags(playerXrayFlags);
        return playerXrayStats;
      },
    },
  };
  console.log('[v5] cs-net-v5 demo_extract studio ready');
}

init();
