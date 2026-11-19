/**
 * ai-overlay.js — 模型的预测画在 3D 场景里（**只画，不算**）。
 *
 * 只画一样东西：**对枪弧线**（aim）。两个要对枪的人头顶之间一条弧，从中间
 * 按 `P(A 对枪获胜)` 切成两段，**每一段按那一侧那个人的胜率上色**：
 *
 *   - A 侧（靠近会死的那位）用 `p`，B 侧用 `1 - p`；
 *   - 0% 红 → 50% 黑 → 100% 绿，所以哪一侧更绿，哪一侧更可能赢。
 *
 * 弧线**只在死前 3 秒到有人死之间**出现（服务端只在那段时间里给预测），所以场上
 * 没有对枪就没有弧线；窗口开头的 0.35 秒淡入（`duel.fade`）。
 *
 * 弧线上**没有游标点**（原先那个白色小球撤掉了）：它和两段颜色表达的是同一件
 * 事，留两个指标反而要多解释一遍；胜负的读数看左下表格的「对枪」格。整条弧线
 * 也不写任何文字。
 *
 * **其余一概不画**：头顶的百分比、脚底下的存活光环都撤了 —— 存活 / 可击杀 /
 * 下一杀 / 下一死 / 对枪胜率全在左下角的选手指标表里（`main.js::updateAiTable()`）。
 * 少画一层就少一层挡视线的东西，也少一处会跟表格打架的数字。
 *
 * 这里不做任何时序/插值：`main.js` 每帧算好"当前采样点应该是多少"再调
 * `updateAiOverlay()`。所以这个模块可以单独跑测试（`aiOverlayDebug()`）。
 */

import * as THREE from 'three';
import { scene, gameToThree } from './scene.js';

// ── 几何常量（three 单位；1 HU = 0.0254）─────────────────────────────────
/** 弧线锚点离脚底多高：站立头顶 1.64 + 一点余量。 */
const ANCHOR_LIFT = 1.98;
/** 弧线的取样段数（越大越圆，48 段在 10 米外已经看不出折角）。 */
const ARC_SEGMENTS = 48;

const COLOR_RED = new THREE.Color(0xff3b30);
const COLOR_BLACK = new THREE.Color(0x161616);
const COLOR_GREEN = new THREE.Color(0x2ee66b);

// ── 模块状态 ─────────────────────────────────────────────────────────────
let built = false;
let root = null;              // 所有 AI 对象都挂在这个 Group 上
let duelGroup = null;
let arcA = null;              // A 侧（靠近 A 的那半条弧）
let arcB = null;

// ── 小工具 ───────────────────────────────────────────────────────────────

/** 概率 → 颜色：0 红、0.5 黑、1 绿。`p` 会被夹到 [0,1]。 */
export function probColor(p, out = new THREE.Color()) {
  const v = Math.min(1, Math.max(0, Number.isFinite(p) ? p : 0.5));
  if (v < 0.5) return out.copy(COLOR_RED).lerp(COLOR_BLACK, v / 0.5);
  return out.copy(COLOR_BLACK).lerp(COLOR_GREEN, (v - 0.5) / 0.5);
}

// ── 构建 ─────────────────────────────────────────────────────────────────

function ensureBuilt() {
  if (built) return true;
  if (!scene) return false;                 // 3D 还没初始化（首帧之前不会走到）

  root = new THREE.Group();
  root.name = 'ai-overlay';
  scene.add(root);

  duelGroup = new THREE.Group();
  duelGroup.name = 'ai-duel';
  duelGroup.visible = false;
  root.add(duelGroup);

  const arcMat = () => new THREE.LineBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.95,
    depthTest: false, depthWrite: false,
  });
  const mkArc = () => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(
      new Float32Array(ARC_SEGMENTS * 2 * 3), 3));
    geo.setDrawRange(0, 0);
    const line = new THREE.LineSegments(geo, arcMat());
    line.frustumCulled = false;
    line.renderOrder = 900;
    duelGroup.add(line);
    return line;
  };
  arcA = mkArc();
  arcB = mkArc();

  built = true;
  return true;
}

// ── 每帧更新 ─────────────────────────────────────────────────────────────

const _va = new THREE.Vector3();
const _vb = new THREE.Vector3();
const _vc = new THREE.Vector3();
const _pts = Array.from({ length: ARC_SEGMENTS + 1 }, () => new THREE.Vector3());
const _color = new THREE.Color();

/** 玩家脚底 + 头顶的世界坐标，写进 `target`（不共享临时向量，A/B 两个都要留）。 */
function anchorOf(state, target, lift = ANCHOR_LIFT) {
  const p = gameToThree(state.x, state.y, state.z);
  return target.set(p.x, p.y + lift, p.z);
}

/**
 * 把对枪弧线摆到两个人之间。
 * @param {null|{a:number,b:number,p:number,t0?:number,t1?:number,fade?:number}} duel
 *        `a` = 这场对枪里会死的一方（弧线的 A 端），`p` = P(A 赢)
 */
function updateDuel(duel, players, show) {
  const stA = duel && players ? players[duel.a] : null;
  const stB = duel && players ? players[duel.b] : null;
  if (!show || !duel || !stA || !stB || !stA.alive || !stB.alive
      || !Number.isFinite(duel.p)) {
    duelGroup.visible = false;
    return;
  }
  duelGroup.visible = true;

  const p = Math.min(1, Math.max(0, duel.p));
  const a = anchorOf(stA, _va);
  const b = anchorOf(stB, _vb);
  // 控制点：中点抬高，抬高量随两人距离增长。抬高小了看起来就是一条直线，
  // 而"弧"本身是判读的一部分（谁在左、谁在右一眼能分清）。上限 2.4 个单位
  // （≈ 95 HU ≈ 2.4 m）：真实击杀的对枪常有 20 m 开外，不封顶的话弧会顶到
  // 画面外面去，反而看不出两端是谁。
  const dist = a.distanceTo(b);
  const lift = Math.min(2.4, 0.45 + dist * 0.34);
  _vc.set((a.x + b.x) / 2, (a.y + b.y) / 2 + lift, (a.z + b.z) / 2);

  const last = ARC_SEGMENTS;
  for (let i = 0; i <= last; i++) {
    const t = i / last;
    const u = 1 - t;
    const pt = _pts[i];
    pt.set(
      u * u * a.x + 2 * u * t * _vc.x + t * t * b.x,
      u * u * a.y + 2 * u * t * _vc.y + t * t * b.y,
      u * u * a.z + 2 * u * t * _vc.z + t * t * b.z,
    );
  }

  const split = Math.max(1, Math.min(last - 1, Math.round(p * last)));
  writeDashes(arcA, _pts, 0, split);
  writeDashes(arcB, _pts, split, last);

  // 颜色：每一侧按"那一侧那个人的胜率"上色（A 侧 = p，B 侧 = 1-p）
  arcA.material.color.copy(probColor(p, _color));
  arcB.material.color.copy(probColor(1 - p, _color));
  // 窗口开始的 0.35 秒淡入（`duel.fade` 由 main.js 按窗口起点算）：弧线不该
  // 在两个人头顶"啪"地一下出现。
  const fade = Number.isFinite(duel.fade) ? Math.min(1, Math.max(0, duel.fade)) : 1;
  const alpha = 0.35 + 0.65 * fade;
  arcA.material.opacity = alpha;
  arcB.material.opacity = alpha;
}

/** 把 `pts[from..to]` 写成"隔一段画一段"的虚线顶点。 */
function writeDashes(line, pts, from, to) {
  const attr = line.geometry.attributes.position;
  const arr = attr.array;
  let n = 0;
  for (let i = from; i < to; i += 2) {
    const p0 = pts[i];
    const p1 = pts[Math.min(i + 1, to)];
    arr[n++] = p0.x; arr[n++] = p0.y; arr[n++] = p0.z;
    arr[n++] = p1.x; arr[n++] = p1.y; arr[n++] = p1.z;
  }
  attr.needsUpdate = true;
  line.geometry.setDrawRange(0, n / 3);
}

/**
 * 每帧调一次。
 *
 * @param {object} s
 * @param {Array} s.players   `getAllPlayerStates(idx)`（10 个）
 * @param {object|null} s.duel `{a, b, p, t0, t1, fade}`，
 *        `p = P(A 赢 = A 击杀 B)`；`fade` 是窗口开场那 0.35 秒的淡入（0..1）
 * @param {boolean} s.showDuel   `AI 对枪` 开关
 */
export function updateAiOverlay(s = {}) {
  if (!ensureBuilt()) return;
  updateDuel(s.duel || null, s.players || null, !!s.showDuel);
}

/** 换回合 / 关掉开关时清干净（不销毁 GPU 资源，回合之间会复用）。 */
export function clearAiOverlay() {
  if (!built) return;
  duelGroup.visible = false;
}

/** 测试用：把"现在画成什么样"读出来（真实几何 + 颜色，不是期望值）。 */
export function aiOverlayDebug() {
  if (!built) return { built: false };
  const arcs = [arcA, arcB].map((l) => {
    const attr = l.geometry.attributes.position;
    const n = l.geometry.drawRange.count;
    const pts = [];
    for (let i = 0; i < n; i++) {
      pts.push([attr.getX(i), attr.getY(i), attr.getZ(i)]);
    }
    return {
      visible: l.visible && l.parent.visible,
      segments: n / 2,
      color: l.material.color.getHex(),
      opacity: l.material.opacity,
      ends: pts.length ? [pts[0], pts[pts.length - 1]] : [],
      // 所有虚线顶点：测试用它验"两段确实接在同一条弧上、每段都有顶点"
      // （虚线是隔一段画的，所以不能只比较两段的端点）
      points: pts,
    };
  });
  return {
    built: true,
    duelVisible: duelGroup.visible,
    arcs,
  };
}
