/**
 * loading.js — 加载遮罩解析动画（Canvas 2D）
 *
 * 展示"解析/加载中"的动态反馈，风格克制、明亮主题、跨浏览器安全：
 *   - 旋转渐变环（青 → 紫）+ 中心脉冲点
 *   - 底部流动的"解析波形"（数据流式条带，模拟解析进度感）
 *
 * 纯 Canvas 2D，无 WebGL / 无实验 API，所有现代浏览器可用。
 * 用法：
 *   import { startLoadingAnimation, stopLoadingAnimation } from './loading.js';
 *   startLoadingAnimation();
 *   stopLoadingAnimation();
 */

const CYAN = [0, 151, 216];
const PURPLE = [124, 77, 255];

let _raf = 0;
let _canvas = null;

/**
 * 启动加载动画（幂等：已启动则不重复）。
 * 读取 #loading-canvas，若不存在则静默跳过（动画是可选的装饰层）。
 */
export function startLoadingAnimation() {
  if (_raf) return;
  const canvas = document.getElementById('loading-canvas');
  if (!canvas) return;
  _canvas = canvas;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  const W = 160, H = 150;               // 逻辑尺寸
  canvas.width = W * DPR;
  canvas.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

  const CX = W / 2;
  const CY = 62;                        // 环中心
  const R = 38;                         // 主环半径
  const t0 = performance.now();

  function frame(now) {
    const t = (now - t0) / 1000;
    ctx.clearRect(0, 0, W, H);

    // ── 主环：青→紫渐变，顺时针旋转 ──
    const grad = ctx.createLinearGradient(CX - R, CY - R, CX + R, CY + R);
    grad.addColorStop(0, `rgba(${CYAN[0]},${CYAN[1]},${CYAN[2]},0.95)`);
    grad.addColorStop(1, `rgba(${PURPLE[0]},${PURPLE[1]},${PURPLE[2]},0.95)`);
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.strokeStyle = grad;
    ctx.shadowColor = `rgba(${CYAN[0]},${CYAN[1]},${CYAN[2]},0.5)`;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    // 起点固定 -90°，扫过 ~300°，整环随时间旋转
    ctx.arc(CX, CY, R, -Math.PI / 2 + t * 1.2, -Math.PI / 2 + t * 1.2 + Math.PI * 1.7);
    ctx.stroke();

    // ── 副环：反向细弧（层次感）──
    ctx.lineWidth = 2;
    ctx.strokeStyle = `rgba(${PURPLE[0]},${PURPLE[1]},${PURPLE[2]},0.35)`;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(CX, CY, R - 9, Math.PI / 2 - t * 0.8, Math.PI / 2 - t * 0.8 + Math.PI * 0.9);
    ctx.stroke();

    // ── 中心脉冲点 ──
    const pulse = 0.5 + 0.5 * Math.sin(t * 3.2);
    ctx.fillStyle = `rgba(${CYAN[0]},${CYAN[1]},${CYAN[2]},${0.45 + 0.4 * pulse})`;
    ctx.beginPath();
    ctx.arc(CX, CY, 2.5 + pulse * 2, 0, Math.PI * 2);
    ctx.fill();

    // ── 底部解析波形（数据流条带，缓慢流动）──
    const bars = 26;
    const gap = 2.5;
    const barW = (W - 28 - gap * (bars - 1)) / bars;
    const baseY = 122;
    const maxH = 20;
    for (let i = 0; i < bars; i++) {
      // 两条不同频率正弦叠加 → 平滑的"解析进度"起伏（无随机闪烁）
      const env = 0.5 + 0.5 * Math.sin(t * 1.8 + i * 0.62);
      const env2 = 0.55 + 0.45 * Math.sin(t * 2.6 + i * 0.31 + 1.3);
      const h = 4 + maxH * env * env2;
      const x = 14 + i * (barW + gap);
      const alpha = 0.22 + 0.4 * env;
      ctx.fillStyle = `rgba(${CYAN[0]},${CYAN[1]},${CYAN[2]},${alpha})`;
      ctx.beginPath();
      // roundRect 是较新 API，低版本浏览器回退为普通矩形
      if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(x, baseY - h, barW, h, 2);
      } else {
        ctx.rect(x, baseY - h, barW, h);
      }
      ctx.fill();
    }
    // 波形基线
    ctx.fillStyle = `rgba(${CYAN[0]},${CYAN[1]},${CYAN[2]},0.15)`;
    ctx.fillRect(14, baseY + 2, W - 28, 1.5);

    _raf = requestAnimationFrame(frame);
  }

  _raf = requestAnimationFrame(frame);
}

/** 停止加载动画（幂等）。 */
export function stopLoadingAnimation() {
  if (_raf) {
    cancelAnimationFrame(_raf);
    _raf = 0;
  }
  if (_canvas) {
    const ctx = _canvas.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, _canvas.width, _canvas.height);
    _canvas = null;
  }
}
