/**
 * cs-net 分享图 —— 把一屏数据画成一张 PNG。
 *
 * 为什么在浏览器里画
 * ------------------
 * 要画的东西**已经在页面里了**：`/api/stats` 的整场统计、`/api/ai/stats` 的
 * AI 统计、玩家页的两份跨场统计、Steam 头像、雷达缩略图。服务端另画一张等于把
 * 这些再查一遍，而且还得在 exe 里塞一套中文字体（Pillow 画中文没字体就是豆腐块，
 * 打包体积又要涨十几 MB）。canvas 这里一个依赖都不用加，中文直接走系统字体。
 *
 * 为什么 `t` 是**参数**而不是 import
 * ----------------------------------
 * 这个文件在 visualizer 的 static 下（`/static/js/share-card.js`）—— 那是四个
 * 页面都挂着的目录，回放页和玩家页都用同一个 URL 加载它。但页面脚本自己可以
 * 用相对路径 import，这个文件不行：玩家页的模块在 `/lib-static/js/` 下，相对
 * import 会指到那个目录里去。所以这里不 import 任何东西，文案函数由调用方传进来。
 * 键名仍然全在 i18n 表里（`common.share.*`），中文界面画中文，英文界面画英文。
 *
 * 版面按 1600 宽设计、高度按内容算（见 `matchHeight` / `playerHeight`），所以有
 * AI 数据时卡片就高一截，不会留一片空白。**画布本身按 `SCALE` 倍画**（1600 × 2.4
 * = 3840 宽），放大看是清楚的，不跟 devicePixelRatio —— 这是一张要发出去的图，
 * 不是自适应界面。
 *
 * 每张图都带 logo 和项目地址，这是这个功能的硬性要求 —— 图片会离开这个应用，
 * 得能认出来源。
 */

// ── 配色（和 visualizer/static/css/style.css 的 :root 一致）──────────────────
const C = {
  text: '#e8eefb',
  dim: '#8fa3c4',
  faint: '#5c6f92',
  cyan: '#38bdf8',
  cyanBright: '#7dd3fc',
  purple: '#8b5cf6',
  green: '#34d399',
  orange: '#fbbf24',
  red: '#f87171',
  t: '#ff9f43',
  ct: '#5aa5ff',
  panel: 'rgba(255, 255, 255, 0.035)',
  line: 'rgba(110, 160, 255, 0.16)',
  track: 'rgba(255, 255, 255, 0.06)',
};

const FAMILY = '"Segoe UI", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", sans-serif';
const MONO = '"Cascadia Mono", Consolas, "SF Mono", Menlo, monospace';

const CARD_W = 1600;
const PAD = 72;

/**
 * 导出倍率：画布宽 = `CARD_W * SCALE` = 3840（4K 宽度）。
 *
 * 所有坐标和字号仍然按设计稿（1600 宽）写，靠一次 `ctx.scale()` 放过去 ——
 * 文字、圆角、渐变都是按 3840 的分辨率**重画**的。反过来做（画 1600 再拉大）
 * 只会糊：canvas 是位图，放大不补信息。
 */
const SCALE = 2.4;

// 各块的高度（px，设计稿单位）。改了这里就要连着 `matchHeight` / `playerHeight`
// 一起改。
const M = { header: 156, title: 172, thumb: 72, teamHead: 48, row: 46, gap: 44, footer: 108 };
const P = { header: 156, identity: 176, tiles: 126, tilesGap: 56, recentHead: 64, row: 44, gap: 40, footer: 108 };

//: 雷达缩略图的正方形盒子（px）和它的上沿。缩略图按**原比例**装进这个盒子，
//: 上下居中 —— 不是裁成正方形：旧缩略图是 640×400，直接当正方形画会拉扁。
//: 盒子底 = 140 + 240 = 380，表头 = 156 + 172 + `M.thumb` = 400，中间留一口气。
const THUMB_BOX = 240;
const THUMB_TOP = 140;

//: 页头的 logo（高 92，宽按 viewBox 的 131.76 : 101.11 算）。原图是"cs·net"
//: 的字标，所以旁边不再写一遍品牌名。
const LOGO_H = 92;
const LOGO_Y = 22;
const LOGO_RATIO = 131.76 / 101.11;

// ── 基础绘制 ────────────────────────────────────────────────────────────────

function font(size, weight = 400, mono = false) {
  return `${weight} ${size}px ${mono ? MONO : FAMILY}`;
}

function roundRect(ctx, x, y, w, h, r) {
  const rad = Math.min(r, h / 2, w / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

/** 一行文字；`align` 是 canvas 的 left/center/right。超宽就省略号。 */
function text(ctx, str, x, y, opts = {}) {
  const {
    size = 22, weight = 400, color = C.text, align = 'left',
    mono = false, maxW = 0, baseline = 'alphabetic',
  } = opts;
  ctx.font = font(size, weight, mono);
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  let s = String(str == null ? '' : str);
  if (maxW > 0) s = clipText(ctx, s, maxW);
  ctx.fillText(s, x, y);
  return ctx.measureText(s).width;
}

/** 超宽 → 中间省略。比截尾巴更容易认出是谁（"资深大神 .Sp…" 不如 "资深…Spinx"）。 */
function clipText(ctx, s, maxW) {
  if (!maxW || ctx.measureText(s).width <= maxW) return s;
  const chars = [...s];
  let lo = 0;
  let hi = chars.length;
  let best = '…';
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const head = Math.ceil(mid / 2);
    const tail = mid - head;
    const cand = tail > 0
      ? chars.slice(0, head).join('') + '…' + chars.slice(chars.length - tail).join('')
      : chars.slice(0, mid).join('') + '…';
    if (ctx.measureText(cand).width <= maxW) { best = cand; lo = mid + 1; } else { hi = mid - 1; }
  }
  return best;
}

/** 百分比 / 小数格式化（和页面上的显示口径一致）。 */
function pct(num, den) {
  if (!den) return '—';
  return `${(Math.round((num * 1000) / den) / 10).toFixed(1)}%`;
}

function signedPct(x, n) {
  if (x == null || n === 0) return '—';
  const v = (Math.round(x * 1000) / 10).toFixed(1);
  return `${x > 0 ? '+' : ''}${v}%`;
}

function loadImage(url) {
  if (!url) return Promise.resolve(null);
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

/** 圆形裁剪画头像（没有图就不画，调用方已经画了首字底衬）。 */
function drawAvatar(ctx, img, x, y, size) {
  if (!img) return false;
  ctx.save();
  ctx.beginPath();
  ctx.arc(x + size / 2, y + size / 2, size / 2, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();
  ctx.drawImage(img, x, y, size, size);
  ctx.restore();
  ctx.beginPath();
  ctx.arc(x + size / 2, y + size / 2, size / 2, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(125, 211, 252, 0.45)';
  ctx.lineWidth = 3;
  ctx.stroke();
  return true;
}

// ── 卡片骨架 ────────────────────────────────────────────────────────────────

/**
 * 建一张画布，返回已经按 `SCALE` 缩放好的 2D 上下文。
 *
 * 之后所有绘制都用设计稿单位（1600 宽那一套），不用再想倍率的事。画布尺寸
 * 仍然是 `* SCALE`，所以是**按 3840 重画**，不是把 1600 的图放大。
 */
function newCard(w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(w * SCALE);
  canvas.height = Math.round(h * SCALE);
  const ctx = canvas.getContext('2d');
  ctx.scale(SCALE, SCALE);
  return { canvas, ctx };
}

function drawBackground(ctx, w, h) {
  const g = ctx.createLinearGradient(0, 0, w * 0.7, h);
  g.addColorStop(0, '#070b16');
  g.addColorStop(0.55, '#0b1224');
  g.addColorStop(1, '#0e1730');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  const glowA = ctx.createRadialGradient(w * 0.12, 0, 0, w * 0.12, 0, w * 0.6);
  glowA.addColorStop(0, 'rgba(56, 189, 248, 0.14)');
  glowA.addColorStop(1, 'rgba(56, 189, 248, 0)');
  ctx.fillStyle = glowA;
  ctx.fillRect(0, 0, w, h);

  const glowB = ctx.createRadialGradient(w * 0.92, h, 0, w * 0.92, h, w * 0.55);
  glowB.addColorStop(0, 'rgba(139, 92, 246, 0.16)');
  glowB.addColorStop(1, 'rgba(139, 92, 246, 0)');
  ctx.fillStyle = glowB;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(110, 160, 255, 0.22)';
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, w - 2, h - 2);
}

/**
 * 把 logo 的深色笔画提亮，得到一个能直接放在深色底上的版本。
 *
 * 原图是给浅底做的：`cs` 是橙的、圆点是绿的，但主体（那个准星 + `net`）是
 * 深蓝 #28397f —— 在卡片背景上等于看不见。垫一块白底能看清，但那是贴纸，
 * 不是 logo。
 *
 * 所以逐像素改：**只把暗的像素换成浅色**，橙 / 绿留着（原图的配色还在），
 * alpha 不动（描边的抗锯齿边缘因此还是渐变的）。取不出来（画布被污染）就
 * 原样返回，卡片退化成"logo 很淡"，但不会炸。
 */
function lightLogo(img) {
  if (!img) return null;
  try {
    // 放大再改像素，是为了让这张离屏位图够 `SCALE` 倍导出用：logo 在卡片上
    // 只有 92 设计像素高，导出时是 221 物理像素，窄了就会糊。
    const scale = 6;
    const w = Math.max(1, Math.round(img.naturalWidth * scale));
    const h = Math.max(1, Math.round(img.naturalHeight * scale));
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const octx = off.getContext('2d');
    octx.drawImage(img, 0, 0, w, h);
    const pixels = octx.getImageData(0, 0, w, h);
    const data = pixels.data;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] < 8) continue;
      const lum = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
      if (lum > 110) continue;                    // 橙色 / 绿色：原样留着
      data[i] = 232;
      data[i + 1] = 238;
      data[i + 2] = 251;
    }
    octx.putImageData(pixels, 0, 0);
    return off;
  } catch {
    return img;
  }
}

function drawHeader(ctx, w, t, kindLabel, logo) {
  const y = M.header;
  // 右侧标签和 logo 的墨迹视觉居中（logo 顶 22、高 92 → 中心 68）
  const labelY = LOGO_Y + LOGO_H / 2 + 9;
  let x = PAD;
  if (logo) {
    // 提亮过的 logo 直接画在卡片背景上（没有底板）
    const wid = LOGO_H * LOGO_RATIO;
    ctx.drawImage(logo, PAD, LOGO_Y, wid, LOGO_H);
    x = PAD + wid + 26;
  } else {
    // logo 取不到时的退路：只有文字标
    text(ctx, 'cs-', PAD, labelY, { size: 32, weight: 300, color: C.text });
    const w1 = ctx.measureText('cs-').width;
    text(ctx, 'net', PAD + w1, labelY, { size: 32, weight: 800, color: C.cyanBright });
    x = PAD + w1 + ctx.measureText('net').width + 26;
  }

  // 项目地址紧跟 logo —— 图会被转走，得让人一眼能找回来
  text(ctx, t('common.share.project'), x, labelY, {
    size: 21, weight: 500, color: C.cyan, mono: true,
    maxW: w - PAD - x - 220,
  });

  text(ctx, kindLabel, w - PAD, labelY, { size: 24, weight: 500, color: C.dim, align: 'right' });

  ctx.strokeStyle = C.line;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(PAD, y - 24);
  ctx.lineTo(w - PAD, y - 24);
  ctx.stroke();
}

function drawFooter(ctx, w, h, t) {
  const y = h - P.footer;
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(PAD, y);
  ctx.lineTo(w - PAD, y);
  ctx.stroke();

  // 项目地址在上面、logo 旁边；这里只留一行出处，和左边缘对齐
  text(ctx, t('common.share.generated'), PAD, y + 52, { size: 20, color: C.faint });
}

/** 板块小标题：左边一根青色竖条 + 标题。 */
function sectionTitle(ctx, label, x, y, color = C.cyan) {
  ctx.fillStyle = color;
  roundRect(ctx, x, y - 18, 5, 24, 2.5);
  ctx.fill();
  text(ctx, label, x + 16, y, { size: 22, weight: 700, color: C.text });
}

/** 一块「大数字 + 小标题」的方块，返回占用的宽度。 */
function drawTile(ctx, x, y, w, h, value, label, accent = C.text) {
  ctx.fillStyle = C.panel;
  roundRect(ctx, x, y, w, h, 14);
  ctx.fill();
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  text(ctx, value, x + w / 2, y + 62, { size: 38, weight: 700, color: accent, align: 'center' });
  text(ctx, label, x + w / 2, y + 96, { size: 19, color: C.dim, align: 'center', maxW: w - 16 });
}

// ── 单场卡片 ────────────────────────────────────────────────────────────────

const TRAD_COLS = [
  { key: 'kills', label: 'K', w: 66 },
  { key: 'deaths', label: 'D', w: 66 },
  { key: 'assists', label: 'A', w: 66 },
  { key: 'kd', label: 'K/D', w: 92 },
  { key: 'adr', label: 'ADR', w: 96 },
  { key: 'hs', label: 'HS', w: 86 },
];

const AI_COLS = [
  { key: 'kill_impact', w: 128, ai: true },
  { key: 'round_impact', w: 128, ai: true },
  { key: 'death_risk', w: 118, ai: true },
  { key: 'kill_rate', w: 112, ai: true },
  { key: 'duel_score', w: 124, ai: true },
];

/**
 * AI 列的列名。
 *
 * 写成「键 → 一次 t() 调用」而不是拼出来的键名（"common.share.ai." + key）：
 * 拼出来的键 `visualizer/tests/check_i18n.mjs` 看不见（它只认字面量），少一条
 * 文案就会一直到英文界面里冒出中文键名才发现。
 */
const AI_LABEL_FN = {
  kill_impact: (t) => t('common.share.ai.killImpact'),
  round_impact: (t) => t('common.share.ai.roundImpact'),
  death_risk: (t) => t('common.share.ai.deathRisk'),
  kill_rate: (t) => t('common.share.ai.killRate'),
  duel_score: (t) => t('common.share.ai.duelScore'),
};

function aiLabel(t, key) {
  const fn = AI_LABEL_FN[key];
  return fn ? fn(t) : key;
}

/**
 * 每个 AI 指标的分母字段名。
 *
 * 五个指标都是 0~1 的比值，分母各不同（击杀数 / 回合数 / 站位样本数…）。
 * 口径由 `app/ai/stats.py::derive()` 定，这里只是照着取；分母一起显示在
 * 页面的悬停提示里，图里省略。
 */
const AI_N_KEY = {
  kill_impact: 'kill_n',
  round_impact: 'rounds',
  death_risk: 'risk_n',
  kill_rate: 'kr_n',
  duel_score: 'duel_n',
};

function aiValue(t, row, key) {
  const n = row ? row[AI_N_KEY[key]] : 0;
  return signedPct(row ? row[key] : null, n);
}

/** 单场卡片的高度：两队的表头 + 每人的一行，加完再算上标题和页脚。 */
function matchHeight(spec, { thumb = false } = {}) {
  const rows = (spec.teams || []).reduce((a, team) => a + (team.players || []).length, 0);
  const teams = (spec.teams || []).length;
  // 22 = 每队写完之后留的那口气（和 drawMatchTable 里的 `cy += 22` 同一个数），
  // `M.thumb` = 雷达缩略图（正方形盒子）压下来的那一截
  return M.header + M.title + (thumb ? M.thumb : 0)
    + teams * (M.teamHead + 22) + rows * M.row + M.gap + M.footer;
}

function matchColumns(spec) {
  if (spec.ai) return [...TRAD_COLS, ...AI_COLS];
  return TRAD_COLS;
}

/**
 * 平台代码 → 文案键。
 *
 * `app/config.py::SOURCE_LABELS` 里的键才是数据（筛选参数、配置文件、库页的
 * `src-*` class 都用它），图上要写的是人能读的名字。文案键复用库页那份
 * `lib.source.*`：四个页面的文案表在 i18n 运行时里合并成一张，平台名必须和
 * 库页筛选器里一模一样，另抄一份就一定会漂。表里没有的代码（用户在库里手填
 * 过、或者以后新增的平台）原样写出来 —— 总比不写强。
 */
const SOURCE_LABEL_KEY = {
  '本地': 'lib.source.local',
  '5E': 'lib.source.5e',
  '完美': 'lib.source.perfect',
  '官匹': 'lib.source.official',
};

function sourceLabel(t, code) {
  const key = SOURCE_LABEL_KEY[code];
  return key ? t(key) : String(code == null ? '' : code);
}

//: 来源徽章的高度（px）和最大宽度。`maxW` 只是防用户手填了一个超长的来源。
const SOURCE_BADGE_H = 36;
const SOURCE_MAX_W = 460;

/**
 * 来源徽章：`来源  5E 对战平台`，一个圆角胶囊。返回它的右边界。
 *
 * `y` 是**基线**：胶囊以它为中心上下摊开，所以后面的地图 / 回合数 / 日期
 * 跟它同一行文字对齐，不用另算一套坐标。
 */
function drawSourceBadge(ctx, code, x, y, t, maxW) {
  const prefix = t('common.share.source');
  const padX = 15;
  const gap = 9;

  // 量宽度得先把字体设上（clipText 也用当前 ctx.font）
  ctx.font = font(18, 400);
  const wPrefix = ctx.measureText(prefix).width;
  ctx.font = font(21, 700);
  const rest = Math.max(56, maxW - wPrefix - gap - padX * 2);
  const name = clipText(ctx, sourceLabel(t, code), rest);
  const wName = ctx.measureText(name).width;

  const w = padX * 2 + wPrefix + gap + wName;
  const top = y - SOURCE_BADGE_H / 2 - 7;   // 字形中线 ≈ 基线 - 7

  ctx.fillStyle = 'rgba(56, 189, 248, 0.13)';
  roundRect(ctx, x, top, w, SOURCE_BADGE_H, SOURCE_BADGE_H / 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(125, 211, 252, 0.36)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  text(ctx, prefix, x + padX, y, { size: 18, color: C.dim });
  text(ctx, name, x + padX + wPrefix + gap, y, { size: 21, weight: 700, color: C.cyanBright });
  return x + w;
}

function drawMatchScore(ctx, spec, y, t, w, maxRight = w - PAD) {
  const [a, b] = spec.teams || [];
  const left = a ? a.label || t('replay.stats.teamA') : 'Team A';
  const right = b ? b.label || t('replay.stats.teamB') : 'Team B';
  const sa = a && a.score != null ? String(a.score) : '–';
  const sb = b && b.score != null ? String(b.score) : '–';

  text(ctx, left, PAD, y, { size: 40, weight: 700, color: C.text, maxW: 520 });
  const midX = PAD + 560;
  text(ctx, sa, midX, y, { size: 40, weight: 800, color: C.cyan, align: 'right' });
  text(ctx, ':', midX + 24, y, { size: 34, weight: 400, color: C.faint, align: 'center' });
  text(ctx, sb, midX + 48, y, { size: 40, weight: 800, color: C.purple });
  text(ctx, right, midX + 48 + ctx.measureText(sb).width + 32, y, {
    size: 40, weight: 700, color: C.text, maxW: 360,
  });

  const bits = [];
  if (spec.map) bits.push(spec.map);
  if (spec.rounds) bits.push(t('common.share.rounds', { n: spec.rounds }));
  if (spec.date) bits.push(spec.date);

  // 来源**打头**：先说清这是哪个平台的一场比赛，地图 / 回合数 / 日期跟着它。
  // 以前来源夹在这行字中间、写的还是内部代码（`5E`），图发出去没人知道那是什么。
  const metaY = y + 46;
  let x = PAD;
  if (spec.source) x = drawSourceBadge(ctx, spec.source, x, metaY, t, SOURCE_MAX_W) + 20;
  if (bits.length && maxRight - x > 40) {
    text(ctx, bits.join('  ·  '), x, metaY, { size: 22, color: C.dim, maxW: maxRight - x });
  }
}

function drawMatchTable(ctx, spec, y, t, w) {
  const width = w - PAD * 2;
  let cols = matchColumns(spec);
  let fixed = cols.reduce((a, c) => a + c.w, 0);
  let nameW = width - fixed;
  // 没有 AI 列时（最常见的情况：还没跑过模型）名字列会一个人吃掉九百多像素，
  // 六个数值列全挤在右边。把富余宽度均摊给数值列，整张表才铺得开。
  const NAME_MAX = 560;
  if (nameW > NAME_MAX) {
    const extra = (nameW - NAME_MAX) / cols.length;
    cols = cols.map((c) => ({ ...c, w: c.w + extra }));
    fixed = cols.reduce((a, c) => a + c.w, 0);
    nameW = width - fixed;
  }
  const xEnd = PAD + width;

  let cy = y;
  for (const team of spec.teams || []) {
    // 队头：一条底色 + 队名 + 比分
    ctx.fillStyle = team.idx === 1 ? 'rgba(139, 92, 246, 0.10)' : 'rgba(56, 189, 248, 0.10)';
    roundRect(ctx, PAD, cy, width, M.teamHead - 8, 10);
    ctx.fill();
    const side = team.side0 === 'CT' ? 'CT' : team.side0 === 'T' ? 'T' : '';
    const label = team.label || (team.idx === 1 ? 'Team B' : 'Team A');
    text(ctx, label, PAD + 18, cy + 29, {
      size: 22, weight: 700, color: team.idx === 1 ? C.purple : C.cyan, maxW: nameW - 120,
    });
    if (side) {
      text(ctx, side, PAD + 18 + ctx.measureText(label).width + 14, cy + 29, {
        size: 17, weight: 600, color: C.faint,
      });
    }
    text(ctx, String(team.score ?? '–'), xEnd - 18, cy + 30, {
      size: 24, weight: 800, color: C.text, align: 'right',
    });
    cy += M.teamHead;

    // 列头（只在第一队里画一次，第二队省掉；列宽一样，不影响对齐）
    if (team === spec.teams[0]) {
      let cx = PAD + nameW;
      text(ctx, t('common.share.col.player'), PAD + 18, cy + 26, { size: 18, color: C.faint });
      for (const col of cols) {
        const label = col.ai ? aiLabel(t, col.key) : col.label;
        text(ctx, label, cx + col.w - 16, cy + 26, {
          size: 17, color: col.ai ? C.cyan : C.faint, align: 'right',
          maxW: col.w - 8,
        });
        cx += col.w;
      }
      cy += 30;
    }

    const players = (team.players || []).slice().sort((p, q) => (q.kills || 0) - (p.kills || 0));
    for (const p of players) {
      text(ctx, p.name, PAD + 18, cy + 30, { size: 22, color: C.text, maxW: nameW - 30 });
      let cx = PAD + nameW;
      for (const col of cols) {
        let value;
        if (col.ai) {
          value = aiValue(t, p.ai, col.key);
        } else if (col.key === 'hs') {
          value = pct(p.headshots, p.kills);
        } else if (col.key === 'kd') {
          value = p.kd == null ? '—' : String(p.kd);
        } else {
          value = p[col.key] == null ? '—' : String(p[col.key]);
        }
        const accent = col.ai ? C.cyanBright : C.text;
        text(ctx, value, cx + col.w - 16, cy + 30, { size: 21, color: accent, align: 'right' });
        cx += col.w;
      }
      cy += M.row;
    }
    cy += 22;                 // 两个队之间留一口气，不然看着像一张长表
  }
  return cy;
}

async function renderMatch(spec, t) {
  // 先把图拿齐再算高度：雷达缩略图占掉标题那一行的一截，卡片就得高 40px。
  // 反过来（先定高度）在有缩略图时会让表格压到页脚上。
  const [logoImg, thumb] = await Promise.all([
    loadImage('/static/logo.svg'),
    loadImage(spec.thumb),
  ]);
  const logo = lightLogo(logoImg);
  const h = matchHeight(spec, { thumb: Boolean(thumb) });

  const { canvas, ctx } = newCard(CARD_W, h);
  const kindLabel = t('common.share.kind.match');

  drawBackground(ctx, CARD_W, h);
  drawHeader(ctx, CARD_W, t, kindLabel, logo);

  // 雷达缩略图在右上角时，标题那一行的可用宽度要收窄（元信息行也算在内 ——
  // 它现在带着来源徽章，比以前长，不兜住就会压到缩略图上）
  const maxRight = thumb ? CARD_W - PAD - THUMB_BOX - 24 : CARD_W - PAD;
  drawMatchScore(ctx, spec, M.header + 66, t, CARD_W, maxRight);

  // 雷达缩略图（有就画在右上角，没有就留空 —— 卡片不缺这一块）。
  // 按原比例缩进 `THUMB_BOX` 的正方形里：缩略图现在是正方形的整张地图，
  // 但旧文件是 640×400 —— 硬拉成正方形会把地图压扁。
  if (thumb) {
    const k = Math.min(THUMB_BOX / thumb.width, THUMB_BOX / thumb.height);
    const tw = thumb.width * k;
    const th = thumb.height * k;
    const tx = CARD_W - PAD - THUMB_BOX + (THUMB_BOX - tw) / 2;
    const ty = THUMB_TOP + (THUMB_BOX - th) / 2;
    ctx.save();
    roundRect(ctx, tx, ty, tw, th, 12);
    ctx.clip();
    ctx.drawImage(thumb, tx, ty, tw, th);
    ctx.restore();
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 1.5;
    roundRect(ctx, tx, ty, tw, th, 12);
    ctx.stroke();
  }

  const tableTop = M.header + M.title + (thumb ? M.thumb : 0);
  drawMatchTable(ctx, spec, tableTop, t, CARD_W);
  drawFooter(ctx, CARD_W, h, t);
  return canvas;
}

// ── 玩家卡片 ────────────────────────────────────────────────────────────────

function playerHeight(spec) {
  const rows = Math.min((spec.recent || []).length, 8);
  const tileRows = spec.aiTotals ? 2 : 1;
  return P.header + P.identity + tileRows * (P.tiles + P.tilesGap)
    + P.recentHead + rows * P.row + P.gap + P.footer;
}

function drawPlayerIdentity(ctx, spec, y, t, w, avatar) {
  const size = 132;
  const drew = drawAvatar(ctx, avatar, PAD, y - size / 2, size);
  if (!drew) {
    ctx.fillStyle = 'rgba(56, 189, 248, 0.16)';
    ctx.beginPath();
    ctx.arc(PAD + size / 2, y, size / 2, 0, Math.PI * 2);
    ctx.fill();
    const ch = String(spec.name || '?').trim().charAt(0) || '?';
    text(ctx, ch, PAD + size / 2, y + 20, { size: 56, weight: 700, color: C.cyanBright, align: 'center' });
  }

  const x = PAD + size + 34;
  text(ctx, spec.name || '?', x, y - 4, { size: 44, weight: 700, color: C.text, maxW: w - x - PAD });
  const bits = [];
  if ((spec.names || []).length) bits.push(spec.names.join(' / '));
  if (spec.steamid) bits.push(`SteamID ${spec.steamid}`);
  if (bits.length) {
    text(ctx, bits.join('  ·  '), x, y + 36, { size: 21, color: C.dim, maxW: w - x - PAD });
  }
  if (spec.lastSeen) {
    text(ctx, t('common.share.lastSeen', { date: spec.lastSeen }), x, y + 68, {
      size: 19, color: C.faint,
    });
  }
}

function drawTiles(ctx, y, w, tiles) {
  const n = Math.max(1, tiles.length);
  const gap = 18;
  const tileW = (w - PAD * 2 - gap * (n - 1)) / n;
  tiles.forEach((tile, i) => {
    drawTile(ctx, PAD + i * (tileW + gap), y, tileW, P.tiles,
      tile.value, tile.label, tile.accent || C.text);
  });
}

function tradTiles(spec, t) {
  const tot = spec.totals || {};
  const win = tot.win_pct == null ? 0 : tot.win_pct;
  const kd = tot.kd == null ? 0 : tot.kd;
  return [
    { label: t('common.share.tile.matches'), value: String(tot.matches ?? 0) },
    { label: t('player.common.winRate'), value: `${win}%`, accent: win >= 50 ? C.green : C.red },
    { label: 'K/D', value: String(kd), accent: kd >= 1 ? C.green : C.red },
    { label: 'KPR', value: String(tot.kpr ?? '—') },
    { label: 'ADR', value: String(tot.adr ?? '—') },
    { label: t('common.share.tile.hs'), value: `${tot.hs_pct ?? 0}%` },
  ];
}

function aiTiles(spec, t) {
  const ai = spec.aiTotals || {};
  return AI_COLS.map((col) => ({
    label: aiLabel(t, col.key),
    value: aiValue(t, ai, col.key),
    accent: C.cyanBright,
  }));
}

function drawRecent(ctx, spec, y, t, w) {
  const rows = (spec.recent || []).slice(0, 8);
  sectionTitle(ctx, t('common.share.recent', { n: rows.length }), PAD, y + 4);
  let cy = y + P.recentHead - 18;

  const cols = [
    { key: 'kills', label: 'K', w: 110 },
    { key: 'deaths', label: 'D', w: 110 },
    { key: 'assists', label: 'A', w: 110 },
    { key: 'kd', label: 'K/D', w: 140 },
    { key: 'adr', label: 'ADR', w: 140 },
  ];
  const roundsW = 130;
  const width = w - PAD * 2;
  const fixed = cols.reduce((a, c) => a + c.w, 0) + roundsW;
  const mapW = width - fixed;
  const roundsX = PAD + mapW + roundsW - 14;

  // 列头
  text(ctx, t('player.profile.maps.col.map'), PAD + 12, cy + 24, { size: 17, color: C.faint });
  text(ctx, t('common.share.col.rounds'), roundsX, cy + 24, { size: 17, color: C.faint, align: 'right' });
  let cx = PAD + mapW + roundsW;
  for (const col of cols) {
    text(ctx, col.label, cx + col.w - 14, cy + 24, { size: 17, color: C.faint, align: 'right' });
    cx += col.w;
  }
  cy += 34;

  for (const m of rows) {
    text(ctx, m.map || '—', PAD + 12, cy + 28, { size: 21, color: C.text, maxW: mapW - 28 });
    text(ctx, String(m.rounds ?? '—'), roundsX, cy + 28, {
      size: 21, color: C.dim, align: 'right',
    });
    let x = PAD + mapW + roundsW;
    for (const col of cols) {
      text(ctx, m[col.key] == null ? '—' : String(m[col.key]), x + col.w - 14, cy + 28, {
        size: 21, color: C.text, align: 'right',
      });
      x += col.w;
    }
    cy += P.row;
  }
  return cy;
}

async function renderPlayer(spec, t) {
  const h = playerHeight(spec);
  const { canvas, ctx } = newCard(CARD_W, h);
  const kindLabel = t('common.share.kind.player');

  const [logoImg, avatar] = await Promise.all([
    loadImage('/static/logo.svg'),
    loadImage(spec.avatarUrl),
  ]);
  const logo = lightLogo(logoImg);

  drawBackground(ctx, CARD_W, h);
  drawHeader(ctx, CARD_W, t, kindLabel, logo);

  let y = P.header + 40;
  drawPlayerIdentity(ctx, spec, y + 44, t, CARD_W, avatar);

  y = P.header + P.identity + 24;
  sectionTitle(ctx, t('common.share.section.trad'), PAD, y + 4);
  drawTiles(ctx, y + 18, CARD_W, tradTiles(spec, t));
  y += P.tiles + P.tilesGap;

  if (spec.aiTotals) {
    sectionTitle(ctx, t('common.share.section.ai'), PAD, y + 4, C.purple);
    drawTiles(ctx, y + 18, CARD_W, aiTiles(spec, t));
    y += P.tiles + P.tilesGap;
  }

  drawRecent(ctx, spec, y, t, CARD_W);
  drawFooter(ctx, CARD_W, h, t);
  return canvas;
}

// ── 出口 ────────────────────────────────────────────────────────────────────

/**
 * 画一张分享图。
 *
 * @param {object} spec `kind: 'match' | 'player'`，字段见文件头的说明
 * @param {function} t  i18n 的 `t()`（由调用方传，见文件头）
 * @returns {Promise<HTMLCanvasElement>}
 */
export async function renderCard(spec, t) {
  if (spec.kind === 'player') return renderPlayer(spec, t);
  return renderMatch(spec, t);
}

function safeName(s) {
  return String(s || '')
    .replace(/[\\/:*?"<>|\s]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 48) || 'cs-net';
}

function stamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`;
}

/** 画完直接下载成 PNG。返回 `{ filename, width, height }`，给调用方报 toast。 */
export async function downloadCard(spec, t) {
  const canvas = await renderCard(spec, t);
  const filename = `cs-net_${safeName(spec.filename || spec.name || spec.map)}_${stamp()}.png`;
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('canvas.toBlob returned null');
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  return { filename, width: canvas.width, height: canvas.height };
}
