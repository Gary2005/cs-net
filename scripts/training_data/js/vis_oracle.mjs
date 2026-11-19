/**
 * vis_oracle.mjs — 常驻进程：**地图只加载一次**，然后反复回答「A 能不能看见 B」。
 *
 * 为什么需要常驻：单次判定有两块成本 ——
 *   * 几何本身：10 人一帧（90 个有序对）实测 **~2 ms**，一个对 **~8 µs**
 *   * **加载地图**：mirage 122 万三角形实测 **解析 0.7 s + 建树 0.3 s**
 * 前者躲不掉，后者必须摊掉 —— 所以这个进程把 `castMapRay` 留着，
 * 从 stdin 收 JSON 行、往 stdout 吐 JSON 行。见 README §3.10。
 *
 * 用法：
 *     node vis_oracle.mjs --obj <map.obj> [--boundary] [--fov 90] [--aspect 1.7778]
 *                         [--inset 1] [--exact] [--block-players]
 *
 * 协议：每行一个请求，`id` 原样回传；出错回 `{"id":..,"error":"..."}` 并继续活着。
 *
 *     {"op":"ping"}
 *     {"op":"map","obj":"<path>"}                       换地图（1 s）
 *     {"op":"load_round","rid":0,"payload":{...}}       灌一个回合（浏览器那份载荷）
 *     {"op":"states","tick":123}                        这个 tick 的 10 个状态（调试用）
 *     {"op":"pair","a":..,"b":..,"tick":123,"opts":{}}  **单对**：A 能不能看见 B
 *     {"op":"pairs","pairs":[[a,b],...],"tick":123}     一次问多对（省 IPC，见下）
 *     {"op":"matrix","tick":123}                        整张 10×10（和出表那条路同一份代码）
 *     {"op":"quit"}
 *
 * ⚠ **一趟往返固定 ~50 µs**，而几何本身只要 ~8 µs/对 —— 所以"一对一问 90 次"光 IPC 就
 * 4.5 ms，同一帧 90 个对请走 `pairs`（或 `matrix`）。
 *
 * `a` / `b` 两种给法：
 *   * **状态对象** `{x,y,z,yaw,pitch,alive,crouch,crouching,eye_height}`（HU，game 空间）
 *   * **整数下标**：灌过回合之后，`tick` 那帧的第 i 个玩家
 *
 * 烟雾：回合模式下不传 `smoke` 就自动用这个回合的（按 tick 过滤 + 缓存）；
 * 裸模式下传 `smoke: [实例...]` + `smoke_key`（同 key 不重复建体素）。
 *
 * ⚠ 两个必须知道的语义坑（都在 `visibility.js` 里，不是这里的选择）：
 *   1. `pairVisibility()` **自己不查 `alive`** —— 死人照算。这里补上：任一方 `alive:false`
 *      直接返回 `{visible:false, dead:true}`（`pairVisibility` 只负责几何）。
 *   2. 观察者眼高看 `eye_height`（没有就按 **连续** 的 `crouch` 插值 64.0938−18·crouch），
 *      目标的盒子高度看 `crouching`（**布尔阶跃** 72 / 54）。字段名不对称，
 *      Python 侧用 `player_state()` 一次填齐，别手写。
 */

import { createInterface } from 'node:readline';

import {
  decodeRoundPayload, setRound, seekTo, getAllPlayerStates,
  vis, makeSmokeCaches, smokeOccluders, loadMap,
} from './vis_common.mjs';

// ── 参数 ────────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const arg = (name, def) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
};
const has = (name) => argv.includes(`--${name}`);

const objPath = arg('obj', null);
if (!objPath) {
  console.error('用法: node vis_oracle.mjs --obj <map.obj> [--boundary] [--fov 90] '
    + '[--aspect 1.7778] [--inset 1] [--exact] [--block-players]');
  process.exit(2);
}

const solid = !has('boundary');
const opts = {
  fovDeg: Number(arg('fov', vis.DEFAULT_FOV_DEG)),
  aspect: Number(arg('aspect', vis.DEFAULT_ASPECT)),
  inset: Number(arg('inset', vis.HULL_INSET_HU)),
  exactCount: has('exact'),
  blockPlayers: has('block-players'),
};

let map = loadMap(objPath);     // 整个进程最贵的一步
let ev = vis.createVisibilityEvaluator({ castMapRay: map.castMapRay, ...opts });
console.error(`[oracle] ${objPath}: ${map.tris} 三角形，`
  + `解析 ${map.parse_ms} ms + 建树 ${map.build_ms} ms`);

// ── 状态 ────────────────────────────────────────────────────────────────────

let cur = null;                 // {rid, round, tickIdx:Map<tick,下标>, caches}
const roundOcc = new Map();     // tick -> occluders（当前回合；烟的配置绝大多数 tick 没变）
const rawOcc = new Map();       // tick -> occluders（裸 smoke，换 smoke_key 时清空）
let rawKey = null;
let rawInsts = null;
let rawCaches = null;

function statesAt(tick) {
  if (!cur) throw new Error('还没 load_round，给不了玩家下标');
  const idx = cur.tickIdx.get(tick);
  if (idx === undefined) throw new Error(`这个回合里没有 tick ${tick}`);
  seekTo(cur.round.round_seconds[idx]);
  return getAllPlayerStates(idx);
}

/** 这个请求该用哪批烟雾遮挡体（按 tick 缓存，别每问一对就重折一遍 journal）。 */
function occFor(req) {
  if (req.tick == null) return [];
  if (req.smoke == null) {
    if (!cur) return [];
    let occ = roundOcc.get(req.tick);
    if (occ === undefined) {
      occ = smokeOccluders(cur.caches, cur.round.smoke || [], req.tick);
      roundOcc.set(req.tick, occ);
    }
    return occ;
  }
  const key = req.smoke_key ?? '';
  if (key !== rawKey) {          // 实例变了才重建体素缓存（这一步贵）
    rawKey = key;
    rawInsts = req.smoke;
    rawCaches = makeSmokeCaches(req.smoke, solid);
    rawOcc.clear();
  }
  let occ = rawOcc.get(req.tick);
  if (occ === undefined) {
    occ = smokeOccluders(rawCaches, rawInsts, req.tick);
    rawOcc.set(req.tick, occ);
  }
  return occ;
}

function resolve(req) {
  const pick = (v, st) => (typeof v === 'number' ? (st[v] ?? null) : v);
  const need = typeof req.a === 'number' || typeof req.b === 'number'
    || (opts.blockPlayers && req.others == null);
  const st = need ? statesAt(req.tick) : null;
  return { a: pick(req.a, st), b: pick(req.b, st), st };
}

// ── 各个 op ─────────────────────────────────────────────────────────────────

function pairOne(req, occ, a, b, st, ai, bi) {
  if (!a || !b) throw new Error('a/b 取不到（tick 不在这个回合里？下标越界？）');
  if (a.alive === false || b.alive === false) {
    return { visible: false, clear: 0, reason: 0, dead: true };   // pairVisibility 不查 alive
  }
  const o = { ...opts, ...(req.opts || {}), castMapRay: map.castMapRay, occluders: occ };
  if (o.blockPlayers) {
    o.others = Array.isArray(req.others) ? req.others : (st || []);
    o.viewerIdx = ai;
    o.targetIdx = bi;
  }
  return vis.pairVisibility(a, b, o);
}

function opPair(req) {
  const occ = occFor(req);
  const { a, b, st } = resolve(req);
  const r = pairOne(req, occ, a, b, st,
    typeof req.a === 'number' ? req.a : -1, typeof req.b === 'number' ? req.b : -1);
  const out = { visible: r.visible, clear: r.clear, reason: r.reason, occluders: occ.length };
  if (r.dead) out.dead = true;
  if (req.opts && req.opts.corners) out.corners = r.corners;
  return out;
}

/**
 * 一次问多对。**这个 op 存在的理由是 IPC**：一趟往返固定 ~50 µs，而几何本身只要 ~8 µs/对，
 * 所以"一对一问 90 次"光 IPC 就要 4.5 ms —— 同一帧 90 个对走这里只要一次往返。
 */
function opPairs(req) {
  const occ = occFor(req);
  const list = Array.isArray(req.pairs) ? req.pairs : [];
  const need = list.some((p) => typeof p[0] === 'number' || typeof p[1] === 'number')
    || (opts.blockPlayers && req.others == null);
  const st = need ? statesAt(req.tick) : null;
  const results = [];
  for (const p of list) {
    const [ra, rb] = p;
    const pick = (v) => (typeof v === 'number' ? (st[v] ?? null) : v);
    try {
      const r = pairOne(req, occ, pick(ra), pick(rb), st,
        typeof ra === 'number' ? ra : -1, typeof rb === 'number' ? rb : -1);
      results.push({ visible: r.visible, clear: r.clear, reason: r.reason, dead: !!r.dead });
    } catch (e) {
      results.push({ error: String((e && e.message) || e) });
    }
  }
  return { results, occluders: occ.length };
}

function opMatrix(req) {
  const t0 = performance.now();
  const occ = occFor(req);
  const occMs = performance.now() - t0;
  let st = Array.isArray(req.states) ? req.states : null;
  if (!st) {
    if (req.tick == null) throw new Error('matrix 要么给 states 数组，要么 load_round 之后给 tick');
    st = statesAt(req.tick);
  }
  const t1 = performance.now();
  ev.setOccluders(occ);
  const res = ev.evaluate(st);
  const evMs = performance.now() - t1;
  const masks = [];
  let visPairs = 0;
  for (let b = 0; b < res.n; b++) {
    let m = 0;
    for (let a = 0; a < res.n; a++) if (res.visible[b * res.n + a]) m |= 1 << a;
    masks.push(m);
    let x = m;
    while (x) { x &= x - 1; visPairs++; }
  }
  return {
    n: res.n, masks, visible: visPairs, occluders: occ.length, stats: res.stats,
    occ_ms: +occMs.toFixed(3), ev_ms: +evMs.toFixed(3),
  };
}

function opLoadRound(req) {
  const round = decodeRoundPayload(req.payload);
  setRound(round);
  const tickIdx = new Map();
  (round.ticks || []).forEach((t, i) => tickIdx.set(t, i));
  cur = {
    rid: req.rid ?? round.id ?? 0,
    round,
    tickIdx,
    caches: makeSmokeCaches(round.smoke, solid),
  };
  roundOcc.clear();
  return {
    rid: cur.rid,
    n_t: (round.ticks || []).length,
    smoke: (round.smoke || []).length,
  };
}

function opMap(req) {
  map = loadMap(req.obj);
  ev = vis.createVisibilityEvaluator({ castMapRay: map.castMapRay, ...opts });
  cur = null;
  roundOcc.clear();
  return { obj: req.obj, tris: map.tris, parse_ms: map.parse_ms, build_ms: map.build_ms };
}

function handle(req) {
  switch (req.op) {
    case 'ping':
      return { map: objPath, tris: map.tris, solid, opts, round: cur ? cur.rid : null };
    case 'map':
      return opMap(req);
    case 'load_round':
      return opLoadRound(req);
    case 'states':
      return { states: statesAt(req.tick) };
    case 'pair':
      return opPair(req);
    case 'pairs':
      return opPairs(req);
    case 'matrix':
      return opMatrix(req);
    case 'quit':
      process.exit(0);
      break;
    default:
      throw new Error(`不认识的 op ${JSON.stringify(req.op)}`
        + '（ping/map/load_round/states/pair/pairs/matrix/quit）');
  }
  return null;
}

// ── 主循环 ──────────────────────────────────────────────────────────────────

const write = (o) => process.stdout.write(JSON.stringify(o) + '\n');

createInterface({ input: process.stdin, terminal: false }).on('line', (line) => {
  const s = line.trim();
  if (!s) return;
  let req;
  try {
    req = JSON.parse(s);
  } catch (e) {
    write({ error: `请求不是合法 JSON：${e.message}` });
    return;
  }
  const t0 = performance.now();
  try {
    const out = handle(req);
    write({ id: req.id, ok: true, ...out, ms: +(performance.now() - t0).toFixed(3) });
  } catch (e) {
    write({ id: req.id, error: String((e && e.message) || e) });
  }
});
