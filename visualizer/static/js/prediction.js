/**
 * prediction.js — AI 路径预测可视化
 *
 * 渲染 PredictionEngine 返回的预测轨迹（pred）与真实轨迹（GT）。
 * 坐标从游戏坐标（world-aligned）转换到 Three.js 空间。
 *
 * 特性：
 *  - 视角方向箭头：沿轨迹每个采样点放置小箭头（方向 = yaw/pitch）
 *  - 透过墙显示：轨迹线/箭头/标记 depthTest=false，始终可见不被建筑物遮挡
 */

import * as THREE from 'three';
import { scene, gameToThree } from './scene.js';

// ── 状态 ──────────────────────────────────────────────────────────────

let predGroup = null;       // 所有预测对象的容器
let predLines = [];         // 每个玩家的 pred 轨迹线
let gtLines = [];
let predMarkers = [];       // 起点/终点标记
let gtMarkers = [];
let predArrows = [];        // 视角方向箭头（pred）
let gtArrows = [];          // 视角方向箭头（GT）
let labelSprites = [];

// 单玩家多采样（renderPlayerSamples）
let sampleGroup = null;     // 采样路径容器
let sampleLines = [];       // K 条采样轨迹线
let sampleGtLine = null;    // 该玩家的 GT 轨迹线
let sampleMarkers = [];     // 起点/终点标记

// 颜色（浅色背景下的高对比配色）
const PRED_COLOR = 0x6a2df0;      // 深紫
const GT_COLOR = 0x00a86b;        // 深绿
const MARKER_COLOR = 0xe5484d;    // 红（起点/终点强调）
const SAMPLE_ALPHA = 0.55;        // 采样路径透明度（细线）

/** 是否显示预测 */
export let showPrediction = true;

export function setShowPrediction(v) {
  showPrediction = v;
  if (predGroup) predGroup.visible = v;
  if (sampleGroup) sampleGroup.visible = v;
}

/** 清空所有预测对象 */
export function clearPrediction() {
  if (predGroup) {
    scene.remove(predGroup);
    predGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
        else o.material.dispose();
      }
    });
    predGroup = null;
  }
  predLines = [];
  gtLines = [];
  predMarkers = [];
  gtMarkers = [];
  predArrows = [];
  gtArrows = [];
  labelSprites = [];
  clearPlayerSamples();
}

/** 清空单玩家采样路径 */
export function clearPlayerSamples() {
  if (sampleGroup) {
    scene.remove(sampleGroup);
    sampleGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
        else o.material.dispose();
      }
    });
    sampleGroup = null;
  }
  sampleLines = [];
  sampleGtLine = null;
  sampleMarkers = [];
}

/**
 * 渲染预测结果。
 * @param {object} result — /api/predict 返回（含 trajectories）
 * @param {object} teams  — 当前 round 的 teams 数组 [10]
 * @param {Array}  names  — 玩家名列表
 * @param {number} focusPlayer — ≥0 时只绘制该玩家的预测/GT 轨迹（扫描条目跳转后聚焦用）；
 *                               -1 绘制全部 10 人
 * @param {number|null} gtScore — 该玩家实际走法的分数（扫描条目 per_tick）。
 *                                ≥0 聚焦时在 GT 路径终点旁用绿色显示"实际 x.xx"
 */
export function renderPrediction(result, teams, names, focusPlayer = -1, gtScore = null) {
  clearPrediction();
  if (!result || !result.trajectories) return;

  const trajs = focusPlayer >= 0
    ? result.trajectories.filter(t => t.player_idx === focusPlayer)
    : result.trajectories;
  if (!trajs.length) return;

  predGroup = new THREE.Group();
  predGroup.name = 'prediction';
  // 全组忽略深度测试 → 轨迹/箭头/标记不被墙遮挡（但保持组内正常层级）
  predGroup.renderOrder = 999;
  scene.add(predGroup);

  for (const t of trajs) {
    const p = t.player_idx;
    const team = (teams && teams[p]) || '?';

    // ── 起点标记（当前 tick 位置）──────────────
    const start = gameToThree(t.start_pos[0], t.start_pos[1], t.start_pos[2]);
    const startMarker = makeMarker(start, team === 'CT' ? 0x4da3ff : 0xff9f43, 0.35);
    predGroup.add(startMarker);
    predMarkers.push(startMarker);

    // ── 预测轨迹 ───────────────────────────────
    if (t.pred_points && t.pred_points.length >= 2) {
      const pts = t.pred_points.map(([x, y, z]) => gameToThree(x, y, z));
      const line = makeTrajectoryLine(pts, PRED_COLOR, 0.09);
      predGroup.add(line);
      predLines.push(line);

      // 视角方向箭头（沿轨迹采样，最多 6 个避免太密）
      if (Array.isArray(t.pred_yaw) && t.pred_yaw.length >= 2) {
        const arrowPts = sampleArrowIndices(pts.length, 6);
        for (const ai of arrowPts) {
          const arrow = makeDirectionArrow(
            pts[ai],
            t.pred_yaw[ai] || 0,
            t.pred_pitch && t.pred_pitch[ai] || 0,
            PRED_COLOR,
            0.5
          );
          predGroup.add(arrow);
          predArrows.push(arrow);
        }
      }

      // 终点
      const endPt = pts[pts.length - 1];
      const endMarker = makeMarker(endPt, PRED_COLOR, 0.3);
      predGroup.add(endMarker);
      predMarkers.push(endMarker);
    }

    // ── GT 轨迹 ────────────────────────────────
    if (t.gt_points && t.gt_points.length >= 2) {
      const pts = t.gt_points.map(([x, y, z]) => gameToThree(x, y, z));
      const line = makeTrajectoryLine(pts, GT_COLOR, 0.06);
      predGroup.add(line);
      gtLines.push(line);

      // 视角方向箭头（GT）
      if (Array.isArray(t.gt_yaw) && t.gt_yaw.length >= 2) {
        const arrowPts = sampleArrowIndices(pts.length, 6);
        for (const ai of arrowPts) {
          const arrow = makeDirectionArrow(
            pts[ai],
            t.gt_yaw[ai] || 0,
            t.gt_pitch && t.gt_pitch[ai] || 0,
            GT_COLOR,
            0.45
          );
          predGroup.add(arrow);
          gtArrows.push(arrow);
        }
      }

      const endPt = pts[pts.length - 1];
      const endMarker = makeMarker(endPt, GT_COLOR, 0.24);
      predGroup.add(endMarker);
      gtMarkers.push(endMarker);
    }

    // ── 玩家标签 ───────────────────────────────
    if (names && names[p]) {
      const sprite = makeLabelSprite(names[p], team === 'CT' ? 0x1f5fa8 : 0xb4630a);
      sprite.position.copy(start);
      sprite.position.y += 1.2;
      predGroup.add(sprite);
      labelSprites.push(sprite);
    }

    // ── 分数标签（聚焦玩家时：紫色=预测分在预测路径终点旁，绿色=实际分在 GT 路径终点旁）──
    if (focusPlayer >= 0) {
      if (t.pred_logp && isFinite(t.pred_logp.per_tick)
          && t.pred_points && t.pred_points.length >= 2) {
        const e = t.pred_points[t.pred_points.length - 1];
        const e3 = gameToThree(e[0], e[1], e[2]);
        const lbl = makeLabelSprite(`预测 ${t.pred_logp.per_tick.toFixed(2)}`, PRED_COLOR, 1.15);
        lbl.position.set(e3.x, e3.y + 1.0, e3.z);
        predGroup.add(lbl);
        labelSprites.push(lbl);
      }
      if (gtScore != null && isFinite(gtScore)
          && t.gt_points && t.gt_points.length >= 2) {
        const e = t.gt_points[t.gt_points.length - 1];
        const e3 = gameToThree(e[0], e[1], e[2]);
        const lbl = makeLabelSprite(`实际 ${gtScore.toFixed(2)}`, GT_COLOR, 1.15);
        lbl.position.set(e3.x, e3.y + 0.45, e3.z);
        predGroup.add(lbl);
        labelSprites.push(lbl);
      }
    }
  }

  predGroup.visible = showPrediction;
}

/**
 * 渲染单个玩家的多条采样路径（/api/predict/player-sampled 返回）。
 *
 * K 条路径用队伍颜色的细线绘制（区别于单条预测的紫色 / GT 绿色），
 * 同时画出该玩家的 GT 轨迹便于对比。
 *
 * @param {object} result — /api/predict/player-sampled 返回
 * @param {object} teams  — 当前 round 的 teams 数组 [10]
 * @param {Array}  names  — 玩家名列表
 */
export function renderPlayerSamples(result, teams, names) {
  clearPlayerSamples();
  if (!result || !result.is_alive || !Array.isArray(result.samples) || !result.samples.length) return;

  const p = result.player_idx;
  const team = (teams && teams[p]) || '?';
  const color = team === 'CT' ? 0x4da3ff : 0xff9f43;

  sampleGroup = new THREE.Group();
  sampleGroup.name = 'player-samples';
  sampleGroup.renderOrder = 999;
  scene.add(sampleGroup);

  // ── 起点标记（队伍颜色）──────────────
  const start = gameToThree(result.start_pos[0], result.start_pos[1], result.start_pos[2]);
  const startMarker = makeMarker(start, color, 0.35);
  sampleGroup.add(startMarker);
  sampleMarkers.push(startMarker);

  // ── K 条采样轨迹（细线，无箭头/光点，避免杂乱）──
  for (const s of result.samples) {
    if (!s.pred_points || s.pred_points.length < 2) continue;
    const pts = s.pred_points.map(([x, y, z]) => gameToThree(x, y, z));
    const line = makeSampleLine(pts, color);
    sampleGroup.add(line);
    sampleLines.push(line);

    // 每条采样终点小标记
    const endPt = pts[pts.length - 1];
    const endMarker = makeMarker(endPt, color, 0.18);
    endMarker.material.opacity = SAMPLE_ALPHA;
    sampleGroup.add(endMarker);
    sampleMarkers.push(endMarker);
  }

  // ── GT 轨迹（目标玩家的真实路径，绿色）──
  if (result.gt && result.gt.gt_points && result.gt.gt_points.length >= 2) {
    const pts = result.gt.gt_points.map(([x, y, z]) => gameToThree(x, y, z));
    const line = makeSampleLine(pts, GT_COLOR);
    sampleGroup.add(line);
    sampleGtLine = line;

    const endPt = pts[pts.length - 1];
    const endMarker = makeMarker(endPt, GT_COLOR, 0.22);
    sampleGroup.add(endMarker);
    sampleMarkers.push(endMarker);
  }

  // ── 玩家标签 ───────────────────────
  if (names && names[p]) {
    const sprite = makeLabelSprite(
      `${names[p]} · 采样×${result.samples.length}`,
      team === 'CT' ? 0x1f5fa8 : 0xb4630a
    );
    sprite.position.copy(start);
    sprite.position.y += 1.2;
    sampleGroup.add(sprite);
  }

  sampleGroup.visible = showPrediction;
}

// ── 构建对象 ─────────────────────────────────────────────────────────

/**
 * 在轨迹上均匀采样箭头位置（避开首尾重复）
 */
function sampleArrowIndices(n, maxArrows) {
  if (n <= 2) return [Math.floor(n / 2)];
  const count = Math.min(maxArrows, n - 1);
  const idxs = [];
  // 从第 1 个点开始，均匀分布
  for (let i = 0; i < count; i++) {
    idxs.push(Math.min(n - 1, Math.round((i + 0.5) * (n - 1) / count)));
  }
  // 去重
  return [...new Set(idxs)].filter(i => i > 0);
}

/**
/**
 * 视角方向小箭头：圆锥 + 短杆，方向由 yaw/pitch（度）决定。
 * 与玩家小人方向锥（visuals.js）使用完全相同的两层旋转结构：
 *   外层 group.rotation.y = yawRad（绕 Y）
 *   内层 pitchGroup.rotation.x = pitchRad（绕 X）
 *   箭头默认指向 +Z（yaw=0, pitch=0 时）
 * 使用 depthTest=false 保证不被建筑物遮挡。
 */
function makeDirectionArrow(pos, yawDeg, pitchDeg, color, size) {
  const root = new THREE.Group();       // 外层：yaw
  const pitchGroup = new THREE.Group(); // 内层：pitch
  root.add(pitchGroup);

  // 杆（沿 +Z 方向，指向箭头）
  const shaftGeo = new THREE.CylinderGeometry(size * 0.06, size * 0.06, size * 0.55, 6);
  shaftGeo.rotateX(Math.PI / 2);   // 圆柱默认沿 Y → 转成沿 Z
  const shaftMat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.9, depthTest: false,
  });
  const shaft = new THREE.Mesh(shaftGeo, shaftMat);
  shaft.position.z = size * 0.275;
  shaft.renderOrder = 999;
  pitchGroup.add(shaft);

  // 箭头（圆锥，默认朝 +Y → 转成朝 +Z）
  const headGeo = new THREE.ConeGeometry(size * 0.18, size * 0.4, 8);
  headGeo.rotateX(Math.PI / 2);   // 锥尖朝 +Z
  const headMat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.95, depthTest: false,
  });
  const head = new THREE.Mesh(headGeo, headMat);
  head.position.z = size * 0.75;
  head.renderOrder = 999;
  pitchGroup.add(head);

  // 两层旋转（与玩家方向锥完全一致）：
  // CS2: yaw=0 → 游戏 +X → Three +Z → rotation.y = yawRad（直接映射）
  //      pitch<0 → 朝下；pitchGroup.rotation.x = pitchRad
  const yawRad = THREE.MathUtils.degToRad(yawDeg);
  const pitchRad = THREE.MathUtils.degToRad(pitchDeg);
  root.rotation.set(0, yawRad, 0);
  pitchGroup.rotation.set(pitchRad, 0, 0);

  root.position.copy(pos);
  root.position.y += 0.2;  // 略抬高，避免贴地
  root.name = 'dir-arrow';
  return root;
}

function makeTrajectoryLine(pts, color, width) {
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.95,
    linewidth: 1,   // WebGL 限制：大多 1
    depthTest: false,   // 透过墙可见
  });
  const line = new THREE.Line(geo, mat);
  line.name = 'traj-line';
  line.renderOrder = 999;

  // 发光光晕线（更粗的透明线）
  const glowGeo = new THREE.BufferGeometry().setFromPoints(pts);
  const glowMat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.25,
    depthTest: false,
  });
  const glow = new THREE.Line(glowGeo, glowMat);
  glow.position.y += 0.02;
  glow.renderOrder = 999;
  line.add(glow);

  // 移动光点（沿轨迹的发光小球，动画用）
  const dotMat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
  });
  const dot = new THREE.Mesh(new THREE.SphereGeometry(width * 0.9, 8, 8), dotMat);
  dot.userData.trajPoints = pts;
  dot.userData.phase = Math.random() * pts.length;
  dot.renderOrder = 999;
  line.add(dot);
  line.userData.dot = dot;

  return line;
}

function makeMarker(pos, color, size) {
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.85,
    depthTest: false,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 12, 12), mat);
  mesh.position.copy(pos);
  mesh.renderOrder = 999;
  mesh.name = 'marker';
  return mesh;
}

/**
 * 采样路径细线：纯线 + 半透明，无发光光晕/移动光点（K 条太多会杂乱）。
 */
function makeSampleLine(pts, color) {
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: SAMPLE_ALPHA,
    depthTest: false,   // 透过墙可见
  });
  const line = new THREE.Line(geo, mat);
  line.name = 'sample-line';
  line.renderOrder = 999;
  return line;
}

function makeLabelSprite(text, color, scale = 1) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 256, 64);
  ctx.font = 'bold 30px "SF Pro Display", "PingFang SC", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0,0,0,0.7)';
  ctx.shadowBlur = 8;
  ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
  ctx.fillText(text, 128, 32);

  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthTest: false,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(1.6 * scale, 0.4 * scale, 1);
  sprite.renderOrder = 999;
  return sprite;
}

/** 每帧更新预测光点动画（在渲染循环中调用） */
export function updatePredictionAnimation(dt) {
  if (!predGroup || !predGroup.visible) return;
  for (const line of predLines) {
    const dot = line.userData.dot;
    if (!dot || !dot.userData.trajPoints) continue;
    const pts = dot.userData.trajPoints;
    dot.userData.phase += dt * 2.5;
    const t = (dot.userData.phase % pts.length + pts.length) % pts.length;
    const i0 = Math.floor(t);
    const i1 = Math.min(i0 + 1, pts.length - 1);
    const f = t - i0;
    dot.position.lerpVectors(pts[i0], pts[i1], f);
    dot.position.y += 0.3;
  }
}
// ═══════════════════════════════════════════════════════════════════════
// spatial-only 单局面预测渲染（无路径；玩家卡片数值 + 整回合曲线 + 弹窗曲线）
// ═══════════════════════════════════════════════════════════════════════

/** 各任务的中文名与曲线颜色（chips / 弹窗共用） */
const METRIC_META = {
  winrate:     { label: '队伍胜率',   color: '#38bdf8' },
  alive_end:   { label: '回合末存活', color: '#34d399' },
  future_kill: { label: '未来击杀',   color: '#f472b6' },
};
export const METRIC_LABELS = Object.fromEntries(
  Object.entries(METRIC_META).map(([k, v]) => [k, v.label]));

// ── 曲线公共工具 ──────────────────────────────────────────────────────

let _metricChartState = null;   // {curveData, curTick} —— 聚合曲线 hover 重绘快照

let _chartTooltip = null;

/** 0-1 概率 → 画布 y 坐标（1.0 在上，0.0 在下） */
function makeY(padT, padB, H) {
  return (v) => padT + (1 - Math.max(0, Math.min(1, v))) * (H - padT - padB);
}

/** 横向网格（0.5 主中线 + 0.25/0.75 辅线）+ 纵轴标签（1.0 在上 / 0.5 中 / 0.0 在下） */
function drawGrid(ctx, W, H, padL, padR, padT, padB, y) {
  ctx.save();
  ctx.strokeStyle = 'rgba(140, 170, 220, 0.16)';
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(padL, y(0.5)); ctx.lineTo(W - padR, y(0.5)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = 'rgba(140, 170, 220, 0.08)';
  ctx.setLineDash([2, 4]);
  for (const g of [0.25, 0.75]) {
    ctx.beginPath(); ctx.moveTo(padL, y(g)); ctx.lineTo(W - padR, y(g)); ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = '#7d90b5';
  ctx.font = '10px sans-serif';
  ctx.fillText('1.0', 4, y(1.0) + 3);
  ctx.fillText('0.5', 4, y(0.5) + 3);
  ctx.fillText('0.0', 4, y(0.0) + 3);
  ctx.restore();
}

/** x 轴刻度（起点 / ¼ / ½ / ¾ / 终点） */
function drawXTicks(ctx, W, H, padL, padR, x, n) {
  ctx.fillStyle = '#7d90b5';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  const stops = [...new Set([0, 0.25, 0.5, 0.75, 1].map(f => Math.round(f * (n - 1))))];
  for (const t of stops) ctx.fillText(`t${t}`, x(t), H - 8);
  ctx.textAlign = 'left';
}

/** hex 颜色 → rgba 字符串（渐变填充用） */
function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map(c => c + c).join('') : h;
  const num = parseInt(full, 16);
  return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
}

// ── 悬浮提示（tooltip 挂 body，fixed 定位跟随光标）──────────────────────

function getChartTooltip() {
  if (!_chartTooltip) {
    _chartTooltip = document.createElement('div');
    _chartTooltip.className = 'chart-tooltip';
    document.body.appendChild(_chartTooltip);
  }
  return _chartTooltip;
}

export function hideChartTooltip() {
  if (_chartTooltip) _chartTooltip.classList.remove('show');
}

function showChartTooltip(e, html) {
  const el = getChartTooltip();
  el.innerHTML = html;
  el.classList.add('show');
  const pad = 14;
  const r = el.getBoundingClientRect();
  let left = e.clientX + pad;
  let top = e.clientY + pad;
  if (left + r.width > window.innerWidth - 8) left = e.clientX - r.width - pad;
  if (top + r.height > window.innerHeight - 8) top = e.clientY - r.height - pad;
  el.style.left = Math.max(8, left) + 'px';
  el.style.top = Math.max(8, top) + 'px';
}

/**
 * 给曲线 canvas 绑定一次 hover 监听（之后每次绘制更新 canvas._chartHover）：
 *   onMove(e, px, py) — 鼠标移动；回调内改 canvas._hoverTick 并重绘十字线 + tooltip
 *   onLeave()          — 鼠标移出；清除十字线
 */
function bindChartHover(canvas) {
  if (canvas._chartHoverBound) return;
  canvas._chartHoverBound = true;
  canvas.addEventListener('mousemove', (e) => {
    const h = canvas._chartHover;
    if (!h || !h.onMove) return;
    const rect = canvas.getBoundingClientRect();
    h.onMove(e, e.clientX - rect.left, e.clientY - rect.top);
  });
  canvas.addEventListener('mouseleave', () => {
    const h = canvas._chartHover;
    if (h && h.onLeave) h.onLeave();
    hideChartTooltip();
  });
}

/** 渲染整回合聚合 CT 胜率曲线到 #metric-chart（数据来自 /api/predict/spatial/round） */
export function renderSpatialCurve(curveData, curTick = -1) {
  const wrap = document.getElementById('metric-chart-wrap');
  const canvas = document.getElementById('metric-chart');
  if (!wrap || !canvas) return;
  if (!curveData || !Array.isArray(curveData.curve) || curveData.curve.length === 0) {
    wrap.classList.add('hidden');
    return;
  }
  wrap.classList.remove('hidden');
  _metricChartState = { curveData, curTick };

  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 320;
  const cssH = canvas.clientHeight || 150;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const padL = 34, padR = 10, padT = 12, padB = 22;
  const W = cssW, H = cssH;
  const n = curveData.curve.length;
  const x = (i) => padL + (n > 1 ? (i / (n - 1)) : 0) * (W - padL - padR);
  const y = makeY(padT, padB, H);

  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, W, H, padL, padR, padT, padB, y);

  // 有效点分段（null 处断开，面积与曲线都按段绘制）
  const segs = [];
  let cur = [];
  for (let i = 0; i < n; i++) {
    const v = curveData.curve[i].ct_winrate;
    if (v == null || !isFinite(v)) { if (cur.length) { segs.push(cur); cur = []; } continue; }
    cur.push({ x: x(i), y: y(v) });
  }
  if (cur.length) segs.push(cur);

  // 渐变面积填充
  for (const seg of segs) {
    const g = ctx.createLinearGradient(0, padT, 0, H - padB);
    g.addColorStop(0, 'rgba(0, 230, 118, 0.26)');
    g.addColorStop(1, 'rgba(0, 230, 118, 0.02)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(seg[0].x, H - padB);
    for (const p of seg) ctx.lineTo(p.x, p.y);
    ctx.lineTo(seg[seg.length - 1].x, H - padB);
    ctx.closePath();
    ctx.fill();
  }

  // 曲线：光晕 + 主线（圆角连接更平滑）
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const [color, width] of [['rgba(0, 230, 118, 0.22)', 3.5], ['#00e676', 1.6]]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    for (const seg of segs) {
      ctx.beginPath();
      seg.forEach((p, idx) => idx ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      ctx.stroke();
    }
  }
  ctx.restore();

  // 当前播放 tick 竖线 + 圆点（红色）
  if (curTick >= 0 && curTick < n) {
    ctx.strokeStyle = '#ff5252';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(x(curTick), padT); ctx.lineTo(x(curTick), H - padB); ctx.stroke();
    ctx.setLineDash([]);
    const cv = curveData.curve[curTick].ct_winrate;
    if (cv != null && isFinite(cv)) {
      ctx.fillStyle = '#ff5252';
      ctx.beginPath(); ctx.arc(x(curTick), y(cv), 3, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // 鼠标悬停十字线 + 数值点（白色虚线 + 绿色实心点）
  const hov = canvas._hoverTick;
  if (hov != null && hov >= 0 && hov < n) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.42)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(x(hov), padT); ctx.lineTo(x(hov), H - padB); ctx.stroke();
    ctx.setLineDash([]);
    const hv = curveData.curve[hov].ct_winrate;
    if (hv != null && isFinite(hv)) {
      ctx.fillStyle = '#00e676';
      ctx.beginPath(); ctx.arc(x(hov), y(hv), 4, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  drawXTicks(ctx, W, H, padL, padR, x, n);
  ctx.fillStyle = '#7d90b5';
  ctx.font = '10px sans-serif';
  ctx.fillText('聚合 CT 胜率', padL, 8);

  // 悬浮交互
  bindChartHover(canvas);
  canvas._chartHover = {
    onMove: (e, px) => {
      const span = (W - padL - padR) / Math.max(1, n - 1);
      const i = Math.round((px - padL) / span);
      canvas._hoverTick = (i >= 0 && i < n) ? i : null;
      renderSpatialCurve(_metricChartState.curveData, _metricChartState.curTick);
      if (canvas._hoverTick == null) { hideChartTooltip(); return; }
      const v = curveData.curve[i].ct_winrate;
      const secs = (i * 0.25).toFixed(1);
      const html = (v == null || !isFinite(v))
        ? `tick <b>${i}</b>（${secs}s）<br><span style="color:#7d90b5">无数据</span>`
        : `tick <b>${i}</b>（${secs}s）<br>CT 胜率 <b style="color:#00e676">${(v * 100).toFixed(1)}%</b>`;
      showChartTooltip(e, html);
    },
    onLeave: () => {
      canvas._hoverTick = null;
      renderSpatialCurve(_metricChartState.curveData, _metricChartState.curTick);
    },
  };

  const winner = curveData.winner;
  if (winner) {
    const tag = document.getElementById('curve-winner-tag');
    if (tag) {
      const color = winner === 'T' ? '#ffaa00' : (winner === 'CT' ? '#5aa5ff' : '#888');
      tag.innerHTML = `<span class="ml-dot" style="background:${color}"></span>Winner: ${winner}`;
    }
  }
}

export function clearMetricChart() {
  const wrap = document.getElementById('metric-chart-wrap');
  if (wrap) wrap.classList.add('hidden');
  _metricChartState = null;
  const canvas = document.getElementById('metric-chart');
  if (canvas) canvas._hoverTick = null;
}

// ── 玩家指标曲线弹窗 ────────────────────────────────────────────────

let _curveState = null;   // {data, playerIdx, metric, tick} —— 弹窗打开时的数据

/**
 * 打开/重绘玩家指标曲线弹窗。
 * @param {object} data       — /api/predict/spatial/round 的 payload
 * @param {number} playerIdx  — 玩家索引 [0,10)
 * @param {string} metric     — winrate | alive_end | future_kill
 * @param {number} currentTick— 当前播放 tick（竖线标记）
 */
export function renderPlayerMetricCurve(data, playerIdx, metric, currentTick) {
  _curveState = { data, playerIdx, metric, tick: currentTick };
  const canvas = document.getElementById('cm-canvas');
  if (canvas) canvas._hoverTick = null;   // 换玩家/指标时清掉残留悬停十字线
  redrawPlayerCurve(currentTick);
}

/** 按当前状态重绘（播放头移动时调用，弹窗不关闭） */
export function redrawPlayerCurve(currentTick) {
  const canvas = document.getElementById('cm-canvas');
  if (!canvas || !_curveState) return;
  const { data, playerIdx, metric } = _curveState;
  _curveState.tick = currentTick;
  const meta = METRIC_META[metric] || { label: metric, color: '#38bdf8' };
  const ticks = data.ticks;
  if (!Array.isArray(ticks) || ticks.length === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 620;
  const cssH = canvas.clientHeight || 210;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const padL = 40, padR = 14, padT = 16, padB = 26;
  const W = cssW, H = cssH;
  const n = ticks.length;
  const x = (i) => padL + (n > 1 ? (i / (n - 1)) : 0) * (W - padL - padR);
  const y = makeY(padT, padB, H);

  // 每 tick 该玩家数值（winrate 用队伍胜率）
  const val = (t) => (metric === 'winrate'
    ? (t.winrate_team && t.winrate_team[playerIdx])
    : (t[metric] && t[metric][playerIdx]));
  const alive = ticks.map(t => (t.alive_mask && t.alive_mask[playerIdx]) ? 1 : 0);
  let deathTick = -1;
  for (let i = 0; i < n; i++) if (!alive[i]) { deathTick = i; break; }

  ctx.clearRect(0, 0, W, H);

  // 背景：存活区间淡色 / 阵亡区间压暗
  const aliveEnd = deathTick >= 0 ? deathTick : n;
  const bg = ctx.createLinearGradient(0, padT, 0, H - padB);
  bg.addColorStop(0, 'rgba(56, 189, 248, 0.05)');
  bg.addColorStop(1, 'rgba(56, 189, 248, 0.02)');
  ctx.fillStyle = bg;
  ctx.fillRect(padL, padT, x(Math.max(0, aliveEnd - 1)) - padL, H - padT - padB);
  if (deathTick >= 0) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.28)';
    ctx.fillRect(x(deathTick), padT, W - padR - x(deathTick), H - padT - padB);
  }

  // 网格 + 纵轴标签（1.0 在上、0.0 在下）
  drawGrid(ctx, W, H, padL, padR, padT, padB, y);

  // 阵亡前的数值点分段（null 处断开）
  const endI = deathTick >= 0 ? deathTick : n;
  const segs = [];
  let cur = [];
  for (let i = 0; i < endI; i++) {
    const v = val(ticks[i]);
    if (v == null || !isFinite(v)) { if (cur.length) { segs.push(cur); cur = []; } continue; }
    cur.push({ x: x(i), y: y(v) });
  }
  if (cur.length) segs.push(cur);

  // 渐变面积填充
  for (const seg of segs) {
    const g = ctx.createLinearGradient(0, padT, 0, H - padB);
    g.addColorStop(0, hexToRgba(meta.color, 0.28));
    g.addColorStop(1, hexToRgba(meta.color, 0.02));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(seg[0].x, H - padB);
    for (const p of seg) ctx.lineTo(p.x, p.y);
    ctx.lineTo(seg[seg.length - 1].x, H - padB);
    ctx.closePath();
    ctx.fill();
  }

  // 曲线：光晕 + 主线
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  const strokeSegs = (color, width, blur) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.shadowColor = blur ? meta.color : 'transparent';
    ctx.shadowBlur = blur;
    for (const seg of segs) {
      ctx.beginPath();
      seg.forEach((p, idx) => idx ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      ctx.stroke();
    }
  };
  strokeSegs(hexToRgba(meta.color, 0.25), 4.5, 10);
  strokeSegs(meta.color, 2.2, 0);
  ctx.restore();

  // 阵亡线
  if (deathTick >= 0) {
    ctx.strokeStyle = 'rgba(248, 113, 113, 0.85)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x(deathTick), padT); ctx.lineTo(x(deathTick), H - padB); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f87171';
    ctx.font = '10px sans-serif';
    ctx.fillText('阵亡', x(deathTick) + 4, padT + 8);
  }

  // 当前 tick 竖线 + 数值点
  const ct = Math.max(0, Math.min(currentTick, n - 1));
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath(); ctx.moveTo(x(ct), padT); ctx.lineTo(x(ct), H - padB); ctx.stroke();
  ctx.setLineDash([]);
  const cv = val(ticks[ct]);
  if (cv != null && isFinite(cv) && alive[ct]) {
    ctx.fillStyle = meta.color;
    ctx.beginPath(); ctx.arc(x(ct), y(cv), 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // 鼠标悬停十字线 + 数值点
  const hov = canvas._hoverTick;
  if (hov != null && hov >= 0 && hov < n) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(x(hov), padT); ctx.lineTo(x(hov), H - padB); ctx.stroke();
    ctx.setLineDash([]);
    if (alive[hov]) {
      const hv = val(ticks[hov]);
      if (hv != null && isFinite(hv)) {
        ctx.fillStyle = meta.color;
        ctx.beginPath(); ctx.arc(x(hov), y(hv), 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  drawXTicks(ctx, W, H, padL, padR, x, n);
  ctx.fillStyle = '#8fa3c4';
  ctx.font = '10px sans-serif';
  ctx.fillText(meta.label, padL, 6);

  // 悬浮交互
  bindChartHover(canvas);
  canvas._chartHover = {
    onMove: (e, px) => {
      const span = (W - padL - padR) / Math.max(1, n - 1);
      const i = Math.round((px - padL) / span);
      canvas._hoverTick = (i >= 0 && i < n) ? i : null;
      redrawPlayerCurve(_curveState.tick);
      if (canvas._hoverTick == null) { hideChartTooltip(); return; }
      const secs = (i * 0.25).toFixed(1);
      let html = `tick <b>${i}</b>（${secs}s）`;
      if (alive[i]) {
        const hv = val(ticks[i]);
        html += (hv == null || !isFinite(hv))
          ? `<br><span style="color:#7d90b5">无数据</span>`
          : `<br>${meta.label} <b style="color:${meta.color}">${(hv * 100).toFixed(1)}%</b>`;
      } else {
        html += `<br><span style="color:#f87171">已阵亡</span>`;
      }
      showChartTooltip(e, html);
    },
    onLeave: () => {
      canvas._hoverTick = null;
      redrawPlayerCurve(_curveState.tick);
    },
  };

  // 底部信息
  const foot = document.getElementById('cm-foot');
  if (foot) {
    const secs = (t) => (t * 0.25).toFixed(1);
    let html = `tick <b>${ct}</b>（${secs(ct)}s）`;
    if (cv != null && isFinite(cv) && alive[ct]) html += ` · 当前 <b style="color:${meta.color}">${(cv * 100).toFixed(1)}%</b>`;
    if (deathTick >= 0) {
      html += ` · 存活区间 0–${deathTick} tick（${secs(deathTick)}s）`;
    } else {
      html += ` · 整回合存活`;
    }
    foot.innerHTML = html;
  }
}
