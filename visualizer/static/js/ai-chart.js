/**
 * ai-chart.js — 点左下指标表里的数字，弹出来的「这一项整回合的概率曲线」。
 *
 * 为什么要曲线
 * ------------
 * 表格给的是**这一刻**的值；"他是不是正在变危险"只能看走势：存活率从 80% 滑到
 * 40% 是一条下坡，比单独看一个 40% 有信息量得多。数据不用再请求一次 —— 服务端
 * 本来就是整回合一次算完（`aiRound.spatial[metric]`，4 Hz 一个点），这里只是把
 * 那一列画出来。
 *
 * 画什么
 * ------
 *   * 折线 + 淡填充，颜色和表格列头一致（存活 cyan / 可击杀 violet / 下次击杀
 *     green / 下次被击杀 red）；
 *   * 死掉的采样点是 `null`：线在这里断开，并画一条红色虚线标出阵亡时刻 ——
 *     把死后的 0 画成"概率归零"是假数据（服务端对死人的列本来就清 0）；
 *   * 白色竖线 + 圆点 = 当前播放头，跟着 `currentTime` 走。只有时间真的变了才
 *     重画（4 Hz 的数据配 60 fps 的回放，每帧重画是白烧 CPU）；
 *   * 悬停时换成虚线游标，右上角的读数变成「悬停 12.4 s · 63%」。
 *
 * 顶部面板那两个「回合胜率」用的是同一张卡（`win_ct` / `win_t`，蓝 / 橙）：
 * 曲线一样画、时间一样跳，但那条线不属于某个人，所以跳转**不锁镜头**；
 * 底部提示文案由数据带进来（`hint`），这里不写死。
 *
 * 交互
 * ----
 * 点曲线上任意一点 = `onSeek(seconds)`：由 main.js 负责 `seekTo()` + 选中并
 * 跟随这个选手。这里只认 x 轴比例，时间轴就是 `round_seconds`，和顶部时间轴、
 * AI 预测点用的是同一根轴。
 *
 * 这个模块只 import i18n 运行时（不碰 three，也不碰 main.js 的状态），DOM id 写死
 * 在 index.html 里 —— 和 ai-overlay.js 一样，它只负责"画出来 + 报状态"。
 */

import { t } from './i18n/index.js';

/** 绘图区四周留白（CSS 像素）。左边 30px 放 0 / 50 / 100% 刻度。 */
const PAD = { l: 30, r: 10, t: 10, b: 16 };

/** 有曲线的指标：颜色和 `.ai-cell[data-metric=...]` / 顶部胜率一一对应。 */
const METRIC_STYLE = {
  alive: { color: '#7dd3fc', label: t('replay.ai.metric.alive') },
  can_kill: { color: '#b69bff', label: t('replay.ai.metric.can_kill') },
  next_kill: { color: '#34d399', label: t('replay.ai.metric.next_kill') },
  next_death: { color: '#f87171', label: t('replay.ai.metric.next_death') },
  // 顶部面板的两个胜率：同一根 win_ct 轴，颜色和那两个数字一致
  win_ct: { color: '#8cc2f7', label: t('replay.ai.metric.win_ct') },
  win_t: { color: '#f3b969', label: t('replay.ai.metric.win_t') },
};

const MONO = '"SF Mono", Menlo, Consolas, monospace';
const SANS = '"SF Pro Display", -apple-system, "PingFang SC", "Segoe UI", Roboto, sans-serif';

/** `#rrggbb` + alpha → canvas 能吃的 `rgba()`。 */
function ink(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function pct(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  const p = v * 100;
  if (p > 0 && p < 1) return '<1%';
  return `${Math.round(p)}%`;
}

function fmtTime(t) {
  if (!Number.isFinite(t)) return '—';
  return `${t.toFixed(1)} s`;
}

/**
 * 创建（或取回）页面上的那唯一一张曲线卡。
 *
 * @param {{onSeek?: (seconds: number, info: object) => void,
 *          onClose?: () => void}} [hooks]
 * @returns {{open: Function, close: Function, update: Function, place: Function,
 *            state: object}}
 */
export function createAiChart({ onSeek, onClose } = {}) {
  const box = document.getElementById('ai-chart');
  const canvas = document.getElementById('ai-chart-canvas');
  const dotEl = document.getElementById('ai-chart-dot');
  const titleEl = document.getElementById('ai-chart-title');
  const nowEl = document.getElementById('ai-chart-now');
  const hintEl = document.getElementById('ai-chart-hint');
  const deathEl = document.getElementById('ai-chart-death');
  const closeEl = document.getElementById('ai-chart-close');
  if (!box || !canvas) throw new Error(t('replay.chart.error.missing'));
  const ctx = canvas.getContext('2d');

  /** 当前曲线：`{player, metric, name, label, seconds[], values[], death}`。 */
  let data = null;
  /** 悬停落在第几个采样点（-1 = 没在悬停）。 */
  let hoverIdx = -1;
  /** 播放头时间（秒）。 */
  let cursor = 0;
  /** 需要重画的理由变了（打开 / 挪位置 / 悬停 / 尺寸变了）。 */
  let dirty = true;

  // ── 几何 ────────────────────────────────────────────────────────────────

  function duration() {
    const s = data && data.seconds;
    return s && s.length ? s[s.length - 1] : 0;
  }

  /**
   * 一次算出这一帧要用的全部几何 + 两个映射函数。
   *
   * 不能把 `canvas.clientWidth` 写进 `xOf()`：每画一个点读一次布局，
   * 400 个点就是 400 次强制回流。这里读一次，之后全是纯算术。
   */
  function geometry() {
    const w = canvas.clientWidth || 360;
    const h = canvas.clientHeight || 132;
    const plotW = Math.max(1, w - PAD.l - PAD.r);
    const plotH = Math.max(1, h - PAD.t - PAD.b);
    const dur = duration();
    return {
      w,
      h,
      plotW,
      plotH,
      dur,
      x: (t) => PAD.l + (dur > 0 ? Math.min(1, Math.max(0, t / dur)) : 0) * plotW,
      y: (v) => PAD.t + (1 - Math.min(1, Math.max(0, v))) * plotH,
    };
  }

  /** 时间 → 最近的采样点下标（二分）。 */
  function nearest(t) {
    const s = data.seconds;
    let lo = 0;
    let hi = s.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (s[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && Math.abs(s[lo - 1] - t) <= Math.abs(s[lo] - t)) return lo - 1;
    return lo;
  }

  /** 任意时刻的概率（相邻两点线性插值；碰到死亡边界就贴到活着的那一侧）。 */
  function valueAt(t) {
    const s = data.seconds;
    if (!s.length) return null;
    const i1 = nearest(t);
    const i0 = Math.max(0, i1 - 1);
    if (i0 === i1) return data.values[i1];
    const v0 = data.values[i0];
    const v1 = data.values[i1];
    if (v0 == null || v1 == null) return v0 == null ? v1 : v0;
    const span = s[i1] - s[i0];
    const f = span > 0 ? Math.min(1, Math.max(0, (t - s[i0]) / span)) : 0;
    return v0 + (v1 - v0) * f;
  }

  // ── 画 ──────────────────────────────────────────────────────────────────

  function ensureCanvas() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = canvas.clientWidth || 360;
    const h = canvas.clientHeight || 132;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function grid(g) {
    // 底板：让曲线落在"一块玻璃"上，而不是直接浮在面板底色里
    ctx.fillStyle = 'rgba(120, 170, 255, 0.045)';
    ctx.fillRect(PAD.l, PAD.t, g.plotW, g.plotH);

    ctx.font = `9px ${MONO}`;
    ctx.lineWidth = 1;
    ctx.textBaseline = 'middle';

    // 横线：0 / 50 / 100%（25 / 75% 只画线不写字，省得刻度比曲线还抢眼）
    for (const v of [1, 0.75, 0.5, 0.25, 0]) {
      const y = g.y(v);
      ctx.strokeStyle = v === 0.5 ? 'rgba(160, 190, 230, 0.22)' : 'rgba(120, 170, 255, 0.12)';
      ctx.beginPath();
      ctx.moveTo(PAD.l, y);
      ctx.lineTo(g.w - PAD.r, y);
      ctx.stroke();
      if (v === 1 || v === 0.5 || v === 0) {
        ctx.fillStyle = 'rgba(150, 172, 204, 0.62)';
        ctx.textAlign = 'right';
        ctx.fillText(`${Math.round(v * 100)}%`, PAD.l - 5, y);
      }
    }

    // 时间刻度：0 / 一半 / 结束。表盘用的就是 round_seconds，和顶部时间轴一致。
    ctx.fillStyle = 'rgba(150, 172, 204, 0.55)';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText('0s', PAD.l, g.h - PAD.b + 3);
    ctx.textAlign = 'right';
    ctx.fillText(`${Math.round(g.dur)}s`, g.w - PAD.r, g.h - PAD.b + 3);
    if (g.dur > 0) {
      ctx.textAlign = 'center';
      ctx.fillText(`${Math.round(g.dur / 2)}s`, g.x(g.dur / 2), g.h - PAD.b + 3);
    }
  }

  /** 把有限值的连续段拆开（null 是"死了 / 还没进场"，不是 0）。 */
  function runs() {
    const out = [];
    let cur = null;
    for (let k = 0; k < data.values.length; k++) {
      const v = data.values[k];
      if (v == null) { cur = null; continue; }
      if (!cur) { cur = []; out.push(cur); }
      cur.push([data.seconds[k], v]);
    }
    return out;
  }

  function drawSeries(color, g) {
    const bottom = g.h - PAD.b;
    const grad = ctx.createLinearGradient(0, PAD.t, 0, bottom);
    grad.addColorStop(0, ink(color, 0.3));
    grad.addColorStop(1, ink(color, 0));

    for (const run of runs()) {
      if (!run.length) continue;
      // 面积：淡色填充到 0 线，让"高 / 低"在余光里就能看出来
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(g.x(run[0][0]), bottom);
      for (const [t, v] of run) ctx.lineTo(g.x(t), g.y(v));
      ctx.lineTo(g.x(run[run.length - 1][0]), bottom);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      run.forEach(([t, v], i) => {
        const x = g.x(t);
        const y = g.y(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // 单点的片段（比如只在死前有一帧）画不出线：补一个点
      if (run.length === 1) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(g.x(run[0][0]), g.y(run[0][1]), 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawDeath(g) {
    if (data.death == null) return;
    const x = g.x(data.death);
    ctx.save();
    ctx.strokeStyle = 'rgba(248, 113, 113, 0.55)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, PAD.t);
    ctx.lineTo(x, g.h - PAD.b);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = 'rgba(248, 113, 113, 0.9)';
    ctx.font = `bold 10px ${SANS}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('✕', x, PAD.t - 1);
  }

  function drawMarker(color, g) {
    const hovering = hoverIdx >= 0;
    const k = hovering ? hoverIdx : nearest(cursor);
    const v = hovering ? data.values[k] : valueAt(cursor);
    const x = hovering ? g.x(data.seconds[k]) : g.x(cursor);
    if (!Number.isFinite(x)) return;

    ctx.save();
    ctx.strokeStyle = hovering ? 'rgba(255, 255, 255, 0.7)' : 'rgba(255, 255, 255, 0.38)';
    ctx.lineWidth = 1;
    if (hovering) ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, PAD.t);
    ctx.lineTo(x, g.h - PAD.b);
    ctx.stroke();
    ctx.restore();

    if (v == null || !Number.isFinite(v)) return;
    ctx.beginPath();
    ctx.arc(x, g.y(v), 3.2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = 'rgba(10, 17, 33, 0.9)';
    ctx.stroke();
  }

  function emptyText(g) {
    ctx.fillStyle = 'rgba(150, 172, 204, 0.6)';
    ctx.font = `10px ${SANS}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(data.emptyText || t('replay.chart.empty.player'), g.w / 2, g.h / 2);
  }

  function draw() {
    if (!data || box.classList.contains('hidden')) return;
    ensureCanvas();
    const g = geometry();
    const style = METRIC_STYLE[data.metric] || METRIC_STYLE.alive;
    ctx.clearRect(0, 0, g.w, g.h);
    grid(g);
    drawSeries(style.color, g);
    drawDeath(g);
    if (data.values.every((v) => v == null)) emptyText(g);
    drawMarker(style.color, g);
    dirty = false;
  }

  // ── 读数 ────────────────────────────────────────────────────────────────

  function refreshNow() {
    if (!data) { nowEl.textContent = '—'; return; }
    if (hoverIdx >= 0) {
      nowEl.textContent = t('replay.chart.now.hover', {
        time: fmtTime(data.seconds[hoverIdx]),
        pct: pct(data.values[hoverIdx]),
      });
      return;
    }
    nowEl.textContent = t('replay.chart.now.current', {
      time: fmtTime(cursor),
      pct: pct(valueAt(cursor)),
    });
  }

  function refreshDeath() {
    deathEl.textContent = data && data.death != null
      ? t('replay.chart.death', { time: fmtTime(data.death) })
      : '';
  }

  // ── 布局：优先贴在指标表正上方，放不下就挪到表格右边 ────────────────────

  function place() {
    const table = document.getElementById('ai-table');
    const vp = document.getElementById('viewport');
    if (!table || !vp) return;
    const tr = table.getBoundingClientRect();
    const vr = vp.getBoundingClientRect();
    if (!tr.width) return;
    const gap = 10;
    box.style.width = `${Math.round(Math.min(tr.width, vr.width - 28))}px`;
    const h = box.offsetHeight;
    const left = Math.round(Math.max(14, tr.left - vr.left));
    const topAbove = Math.round(Math.max(8, tr.top - vr.top - h - gap));
    // 小地图钉在左上角（200×200，窄屏 140）。表格正上方只剩几十像素时，
    // 硬塞会正好盖住小地图（第一版就是这么翻车的）—— 那时改成放在表格右边、
    // 底边对齐，读起来还是"这一行的详情"。
    const mini = document.querySelector('.minimap-container');
    const miniBottom = mini ? mini.getBoundingClientRect().bottom - vr.top : 0;
    const rightLeft = Math.round(tr.right - vr.left + gap);
    if (topAbove >= miniBottom + 8) {
      box.style.left = `${left}px`;
      box.style.top = `${topAbove}px`;
      box.dataset.place = 'above';
    } else if (rightLeft + 300 <= vr.width - 8) {
      box.style.left = `${rightLeft}px`;
      box.style.top = `${Math.round(Math.max(8, tr.bottom - vr.top - h))}px`;
      box.dataset.place = 'right';
    } else {
      // 窄屏两边都没位置：宁可盖住小地图，也要让曲线完整可见
      box.style.left = `${left}px`;
      box.style.top = `${topAbove}px`;
      box.dataset.place = 'above';
    }
    dirty = true;
  }

  // ── 打开 / 关闭 / 更新 ─────────────────────────────────────────────────

  function open(next) {
    data = next;
    hoverIdx = -1;
    cursor = Number.isFinite(next.cursor) ? next.cursor : 0;
    const style = METRIC_STYLE[next.metric] || METRIC_STYLE.alive;
    dotEl.style.color = style.color;
    dotEl.style.background = style.color;
    // 选手曲线和胜率曲线的说法不一样：标题 / tooltip / 无障碍文案 / 底部提示
    // 都由数据带进来（main.js 的两个 series 构造函数各写各的）。
    titleEl.textContent = next.heading || `${next.name} · ${next.label}`;
    titleEl.title = next.title
      || t('replay.chart.title.metric', { label: next.label, metric: next.metric });
    box.setAttribute('aria-label',
      next.aria || t('replay.chart.aria.player', { name: next.name, label: next.label }));
    if (hintEl) {
      hintEl.textContent = next.hint
        || t('replay.chart.hint.player');
    }
    box.classList.remove('hidden');
    // 曲线会压住视口底部的操作提示（"WASD 移动机位…"）：开着的时候让它淡出，
    // 和第一人称 HUD 一个处理方式。
    document.getElementById('viewport')?.classList.add('chart-open');
    place();
    refreshDeath();
    draw();
    refreshNow();
  }

  function close() {
    const wasOpen = !box.classList.contains('hidden') || !!data;
    data = null;
    hoverIdx = -1;
    box.classList.add('hidden');
    document.getElementById('viewport')?.classList.remove('chart-open');
    nowEl.textContent = '—';
    deathEl.textContent = '';
    if (wasOpen) onClose?.();
  }

  function update(timeSec) {
    if (!data || box.classList.contains('hidden')) return;
    const t = Number.isFinite(timeSec) ? timeSec : 0;
    // 数据是 4 Hz：播放头挪 0.05 s（不到 1/5 个采样点）再重画就够顺了。
    // 按 60 fps 重画一条 400 点的折线纯属白烧 CPU，肉眼看不出差别。
    if (!dirty && Math.abs(t - cursor) < 0.05) return;
    cursor = t;
    draw();
    refreshNow();
  }

  // ── 指针 ────────────────────────────────────────────────────────────────

  function indexAt(clientX) {
    if (!data) return -1;
    const rect = canvas.getBoundingClientRect();
    const g = geometry();
    const f = (clientX - rect.left - PAD.l) / Math.max(1, g.plotW);
    const t = Math.min(g.dur, Math.max(0, f * g.dur));
    return nearest(t);
  }

  canvas.addEventListener('pointermove', (e) => {
    if (!data) return;
    const k = indexAt(e.clientX);
    if (k === hoverIdx) return;
    hoverIdx = k;
    draw();
    refreshNow();
  });
  canvas.addEventListener('pointerleave', () => {
    if (hoverIdx < 0) return;
    hoverIdx = -1;
    draw();
    refreshNow();
  });
  canvas.addEventListener('pointerdown', (e) => {
    if (!data || e.button !== 0) return;
    const k = indexAt(e.clientX);
    if (k < 0) return;
    onSeek?.(data.seconds[k], { player: data.player, metric: data.metric, index: k });
  });
  closeEl?.addEventListener('click', () => close());

  return {
    open,
    close,
    update,
    place,
    /** 给测试 / 调试用：卡片现在是什么状态（DOM 与数据都读真实值）。 */
    get state() {
      if (!data) return { open: false };
      const isOpen = !box.classList.contains('hidden');
      return {
        open: isOpen,
        player: data.player,
        metric: data.metric,
        name: data.name,
        label: data.label,
        title: titleEl.textContent,
        now: nowEl.textContent,
        hint: hintEl ? hintEl.textContent : '',
        samples: data.values.length,
        points: data.values.filter((v) => v != null).length,
        duration: duration(),
        death: data.death ?? null,
        cursor,
        hover: hoverIdx >= 0 ? data.seconds[hoverIdx] : null,
        hoverIndex: hoverIdx,
        // canvas 真的画上去东西了吗（数一下非透明像素；只给测试用）
        ink: isOpen && canvas.width > 1 ? inkCount() : 0,
      };
    },
  };
}

/** 非透明像素个数。只被 `state.ink` 调用（测试断言"真画了"），不是热路径。 */
function inkCount() {
  const canvas = document.getElementById('ai-chart-canvas');
  if (!canvas) return 0;
  const g = canvas.getContext('2d');
  try {
    const d = g.getImageData(0, 0, canvas.width, canvas.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
    return n;
  } catch {
    return 0;
  }
}
