/**
 * replay-core.js — 回放数据层（demo_extract v6 / v7 / .csz）。
 *
 * 数据来源
 * --------
 * 服务端把 demo 解析成 `demo_extract` 的结构后**按回合**发过来：
 *
 *   summary  : 地图 / 玩家 / 武器表 / 地名表 / 回排名 / 采样方式（一次）
 *   round    : 单回合的玩家列（base64 的 Float32/Int16/Uint8）+ 事件 + 烟/火
 *
 * 为什么按回合发：all-tick（≈64 Hz）下一整场的玩家列 JSON 有 160+ MB，
 * 单个回合最大 ~5 MB，前端切回合时才拉，内存和带宽都可控。
 *
 * 字段和 v6/v7 的对应关系见 tools/demo_extract/API.md。旧的 `ts/te` 是服务端
 * 从 v6 的 `s0/s1` 归一化来的，方便沿用原来的区间过滤逻辑。
 */

// ── 下蹲 / 眼高常数（HU） ───────────────────────────────────────────────────
//
// demo 里**精确可得**的只有"增量"：`m_flDuckOffset == -18 × crouch` 在全部
// 727,649 个存活采样上成立，而且碰撞盒 `m_vecMaxs.z` 只在 72 / 54 两个值之间
// 跳变，所以 18 HU 这个行程有两处独立佐证。
//
// 站立足眼高是整条链上**唯一** demo 说不准的数：CS2 没有网络化任何视点偏移
// 字段（没有 m_vecViewOffset），唯一的候选 `CCSPlayerPawn.m_vecZ` 众数正好是
// 64.000，但它空中读数是垃圾（中位 13.1）、地面也有 ~18% 越界，不能采信。
// 这里取 Valve 文档值 64.0938（PLAN.md §4.4），与 64.000 差 0.0938 HU = 2.4 mm，
// 画面上不可见。要改就改这里一处。
export const STANDING_EYE_HEIGHT = 64.0938;
export const CROUCH_EYE_DROP = 18.0;

// 碰撞盒：32 × 32 × 72（站）/ 32 × 32 × 54（蹲），原点在**脚底**。
// 实测：m_vecMins = (-16,-16,0)、m_vecMaxs = (16,16,72|54)，mins.z 恒为 0。
export const HULL_HALF_WIDTH = 16.0;
export const HULL_STAND_HEIGHT = 72.0;
export const HULL_CROUCH_HEIGHT = 54.0;

/** 蹲下量 0..1 → 眼高（相对脚底，HU）。 */
export function eyeHeightFor(crouch) {
  return STANDING_EYE_HEIGHT - CROUCH_EYE_DROP * (crouch || 0);
}

/** 蹲下状态 → 碰撞盒高度（HU）。这是**阶跃**的，不是渐变。 */
export function hullHeightFor(crouching) {
  return crouching ? HULL_CROUCH_HEIGHT : HULL_STAND_HEIGHT;
}

// ── 步态（走路时摆腿）──────────────────────────────────────────────────────
//
// 两条原则，都是为了"脚不打滑"：
//
//   1. **相位跟着走过的水平距离走，不跟时间走。** 摆腿角 = A·sin(2π·d/周期)，
//      d 是累积水平位移。这样步频自动 = 速度 / 步幅，慢走就是慢摆、大步就是低频，
//      不需要任何"速度 → 频率"的映射表；而且脚落地的那一刻线速度正好等于身体速度。
//      如果改成按时间积分，同一个采样点会随播放历史摆到不同角度，拖时间轴就会跳。
//   2. **摆幅跟着速度走**（`v` 列，HU/s，实测和位移完全一致：p50 都是 213.3、
//      p95 都是 250.0）。站着不动 → 摆幅 0 → 腿完全不动。
//
// 所以整个步态是**采样点的纯函数**：`buildGait(round)` 一次算好两个数组，
// 之后拖到任意采样点、来回拖、倒着放，姿态都完全一样，也可以写测试。

/** 一个完整步态周期（左一步 + 右一步）走过的水平距离，HU。 */
// 实测 CS2 跑速上限就是 250 HU/s（p90 = 250.0，直方图 225-250 占 40%）。
// 112 HU/周期 → 全速时 2.23 周期/s ≈ 4.5 步/s，是短跑的量级；
// 走速（~130 HU/s）时降到 1.16 周期/s，看起来就是走路。
export const GAIT_STRIDE_HU = 112;
/** 全速时的摆腿角度，弧度。0.60 rad ≈ 34°。 */
export const GAIT_MAX_SWING = 0.60;
/** 摆幅归一化用的参考速度（CS2 持刀跑速上限），HU/s。 */
export const GAIT_RUN_SPEED_HU = 250;
/** 蹲走时步幅缩到多少 —— 步子小，所以同样的距离摆动次数更多。 */
export const GAIT_CROUCH_STRIDE = 0.60;
/** 蹲走时摆幅缩到多少。 */
export const GAIT_CROUCH_SWING = 0.55;
/**
 * 单个采样内水平位移超过这个就当**传送 / 重生**，不推进相位。
 * 全速跑一采样才 3.9 HU，200 HU 只可能是复活或者回合重置。
 */
export const GAIT_TELEPORT_HU = 200;
/**
 * 起跳 / 落地 / 开火停顿是**阶跃**的，摆幅按这个时间常数平滑过去，秒。
 * 不平滑的话腿会在离地那一帧瞬间弹回中立位。
 */
export const GAIT_AIR_TAU = 0.08;
/**
 * 髋部随摆腿下沉的比例。纯剪式摆腿（腿是刚体、没有膝盖）如果髋高不变，
 * 脚会离地 `L(1-cosθ)`，看着像在飘；按 `1 - BOB(1-cosθ)` 下沉之后
 * 脚离地只剩 `L·BOB·(1-cosθ)`，正好减半，而且髋部在步幅最大时最低、
 * 双腿并拢时最高 —— 和真人走路的重心起伏同相。
 */
export const GAIT_BOB = 0.5;

/**
 * 把一个回合的步态预计算出来。
 *
 * @param {object} round `loadRound()` 的返回值
 * @returns {Array<{phase: Float32Array, amp: Float32Array}>} 每个玩家一份
 */
export function buildGait(round) {
  const n = round.ticks.length;
  const secs = round.round_seconds || [];
  const out = [];

  for (const p of round.players) {
    const phase = new Float32Array(n);
    const amp = new Float32Array(n);
    let ph = 0;
    let a = 0;

    for (let i = 0; i < n; i++) {
      if (i > 0) {
        const dt = Math.max(0, (secs[i] || 0) - (secs[i - 1] || 0));
        const alive = !!p.alive[i];
        const crouch = p.crouch ? (p.crouch[i] || 0) : 0;

        // ① 相位：只吃**水平**位移。竖直方向（下落、跳）不该让腿走起来。
        const dx = p.x[i] - p.x[i - 1];
        const dy = p.y[i] - p.y[i - 1];
        const d = Math.hypot(dx, dy);
        if (alive && d > 0 && d < GAIT_TELEPORT_HU) {
          const stride = GAIT_STRIDE_HU * (1 - (1 - GAIT_CROUCH_STRIDE) * crouch);
          ph += 2 * Math.PI * d / stride;
        }

        // ② 摆幅：吃速度列。死了或者在空中的话收回中立位。
        const v = p.v ? (p.v[i] || 0) : 0;
        let target = GAIT_MAX_SWING * Math.min(1, v / GAIT_RUN_SPEED_HU);
        target *= 1 - (1 - GAIT_CROUCH_SWING) * crouch;
        if (!alive || !p.on_ground || !p.on_ground[i]) target = 0;

        // 阶跃的 target 平滑一下
        a = dt > 0 ? a + (target - a) * (1 - Math.exp(-dt / GAIT_AIR_TAU)) : target;
      }
      // 折回 [0, 2π)：Float32 存得下，精度 ~7.5e-7 rad
      phase[i] = ph % (2 * Math.PI);
      amp[i] = a;
    }
    out.push({ phase, amp });
  }
  return out;
}

// ── 全局状态 ────────────────────────────────────────────────────────────────

/** @type {object|null} 摘要（不是完整数据） */
export let matchData = null;

/** @type {number} 当前回合下标 */
export let currentRoundIdx = 0;

/** @type {number} 当前回合内的采样下标 */
export let currentSampleIdx = 0;

/** @type {number} 当前回合相对秒数（连续，用于插值） */
export let currentTime = 0;

/** @type {boolean} 播放状态 */
export let isPlaying = false;

/** @type {number} 播放倍速 */
export let playSpeed = 1.0;

/** @type {object|null} 当前回合（已解码） */
export let currentRound = null;

/** @type {number} 回合总数 */
export let totalRounds = 0;

export function setPlaying(v) { isPlaying = v; }
export function setPlaySpeed(v) { playSpeed = v; }

// 反查表
let idxToWeapon = {};
let idxToPlace = {};

// 已解码回合缓存（切回已看过的回合不用重新请求）
const roundPayloadCache = new Map();

// 事件去重
const firedKills = new Set();
const firedDamage = new Set();

// ── base64 → typed array ────────────────────────────────────────────────────

function b64ToBytes(b64) {
  if (!b64) return new Uint8Array(0);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** base64 → Float32Array（字节长度必为 4 的倍数，ArrayBuffer 从 0 开始，天然对齐）。 */
function f32(b64) {
  const bytes = b64ToBytes(b64);
  return new Float32Array(bytes.buffer, 0, bytes.length >> 2);
}

function i32(b64) {
  const bytes = b64ToBytes(b64);
  return new Int32Array(bytes.buffer, 0, bytes.length >> 2);
}

function i16(b64) {
  const bytes = b64ToBytes(b64);
  return new Int16Array(bytes.buffer, 0, bytes.length >> 1);
}

function u8(b64) {
  return b64ToBytes(b64);
}

// ── 摘要 ────────────────────────────────────────────────────────────────────

/**
 * 载入一份回放摘要（服务端 /api/session 或加载完成后的 summary）。
 * 不包含任何回合的逐帧数据——那要 `loadRound(i)`。
 */
export function loadMatchData(summary) {
  matchData = summary;
  totalRounds = (summary.rounds || []).length;
  currentRoundIdx = 0;
  currentSampleIdx = 0;
  currentTime = 0;
  isPlaying = false;
  currentRound = null;
  roundPayloadCache.clear();

  idxToWeapon = {};
  for (const [name, idx] of Object.entries(summary.weapons || {})) idxToWeapon[idx] = name;
  idxToPlace = {};
  for (const [name, idx] of Object.entries(summary.places || {})) idxToPlace[idx] = name;
}

/** 回合元信息（下标 idx 是 rounds[] 的位置，不是 round id）。 */
export function roundMeta(idx) {
  return matchData ? (matchData.rounds || [])[idx] : null;
}

/** 已解码的回合载荷（没有就去请求）。 */
export async function fetchRoundPayload(idx) {
  if (roundPayloadCache.has(idx)) return roundPayloadCache.get(idx);
  const res = await fetch(`/api/round/${idx}`);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { msg = (await res.json()).error || msg; } catch (_) {}
    throw new Error(msg);
  }
  const payload = await res.json();
  roundPayloadCache.set(idx, payload);
  return payload;
}

/**
 * 把服务端的紧凑回合载荷解码成前端用的结构。
 * 玩家列全部是 typed array（零拷贝 view），避免 7 万 × 10 个 JS number。
 */
export function decodeRoundPayload(p) {
  const ticks = i32(p.ticks);
  const n = p.sample_count;
  const tickrate = p.tickrate || 64;
  const t0 = ticks.length ? ticks[0] : 0;
  const secs = new Float32Array(n);
  for (let i = 0; i < n; i++) secs[i] = (ticks[i] - t0) / tickrate;

  const players = (p.players || []).map((pl) => {
    const flags = u8(pl.flags);
    // flags 展开成 5 条 Uint8 列（每位一列）。n 最大 ~7900，10 个玩家也就
    // ~320 KB，比 Proxy 视图快几个数量级，热路径 getPlayerState 直接用。
    const helmet = new Uint8Array(n);
    const defuser = new Uint8Array(n);
    const alive = new Uint8Array(n);
    const onGround = new Uint8Array(n);
    const crouching = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const f = flags[i];
      helmet[i] = f & 1;
      defuser[i] = (f >> 1) & 1;
      alive[i] = (f >> 2) & 1;
      onGround[i] = (f >> 3) & 1;
      crouching[i] = (f >> 4) & 1;
    }
    // 蹲下量在传输时量化成 0..255（.csz 里仍是完整 float32），
    // 这里一次展开成 Float32Array，热路径不用再除。
    const crouchRaw = pl.crouch ? u8(pl.crouch) : null;
    const crouch = new Float32Array(n);
    if (crouchRaw) {
      for (let i = 0; i < n; i++) crouch[i] = crouchRaw[i] / 255;
    }
    return {
      x: f32(pl.x),
      y: f32(pl.y),
      z: f32(pl.z),
      yaw: f32(pl.yaw),
      pitch: f32(pl.pitch),
      v: f32(pl.v),
      flash: f32(pl.flash),
      flash_alpha: f32(pl.flash_alpha),
      hp: i16(pl.hp),
      armor: i16(pl.armor),
      weapon: i16(pl.weapon),
      place: i16(pl.place),
      shots: u8(pl.shots),
      footsteps: u8(pl.footsteps),
      jump: u8(pl.jump),
      land: u8(pl.land),
      helmet, defuser, alive, on_ground: onGround, crouching,
      crouch,
      has_crouch: !!crouchRaw,
      flags,
      inventory: pl.inventory || [],   // 稀疏：{i, v}
      // spotted：每个采样点上"能看到这个玩家的敌人"编号，同样稀疏成 {i, v}。
      // 方向很重要，别搞反了 —— 见 `visibleEnemiesOf()`。
      spotted: pl.spotted || [],
      _invCache: { idx: -1, items: [] },
      _spotCache: { idx: -1, items: [] },
    };
  });

  const bp = p.bomb_position ? f32(p.bomb_position) : new Float32Array(0);

  return {
    idx: p.idx,
    id: p.id,
    teams: p.teams || [],
    winner: p.winner,
    end_reason: p.end_reason,
    start_tick: p.start_tick,
    tickrate,
    ticks,
    round_seconds: secs,
    sample_count: n,
    bomb_planted_time: p.bomb_planted_time,
    bomb_planted: u8(p.bomb_planted),
    bomb_dropped: u8(p.bomb_dropped),
    bomb_position: bp,
    missing_samples: p.missing_samples || [],
    players,
    events: p.events || {},
    fire: p.fire || [],
    smoke: p.smoke || [],
  };
}

/**
 * 拉取并切到某个回合。
 * @returns {Promise<object>} 解码后的回合
 */
export async function loadRound(idx) {
  if (!matchData || idx < 0 || idx >= totalRounds) return null;
  const payload = await fetchRoundPayload(idx);
  const round = decodeRoundPayload(payload);
  currentRoundIdx = idx;
  currentRound = round;
  currentSampleIdx = 0;
  currentTime = round.round_seconds[0] || 0;
  firedKills.clear();
  firedDamage.clear();
  return round;
}

/** 已经解码好的回合直接装上（缓存命中，同步）。 */
export function setRound(round) {
  if (!round) return null;
  currentRoundIdx = round.idx;
  currentRound = round;
  currentSampleIdx = 0;
  currentTime = round.round_seconds[0] || 0;
  firedKills.clear();
  firedDamage.clear();
  return round;
}

export function getCachedRound(idx) {
  const payload = roundPayloadCache.get(idx);
  return payload ? decodeRoundPayload(payload) : null;
}

// ── 采样 / 时间 ─────────────────────────────────────────────────────────────

export function getSampleCount() {
  return currentRound ? currentRound.ticks.length : 0;
}

export function getRoundDuration() {
  if (!currentRound) return 0;
  const s = currentRound.round_seconds;
  return s.length ? s[s.length - 1] : 0;
}

/**
 * 按真实时间推进（round_seconds 连续，不吸附到采样点）。
 */
export function advanceReplay(deltaSec) {
  if (!currentRound || !isPlaying) return { events: null, advanced: false, roundEnded: false };

  const maxTime = getRoundDuration();
  const prevIdx = currentSampleIdx;

  currentTime += deltaSec * playSpeed;
  if (currentTime >= maxTime) currentTime = maxTime;

  const secs = currentRound.round_seconds;
  let newIdx = prevIdx;
  for (let i = prevIdx; i < secs.length; i++) {
    if (secs[i] > currentTime) { newIdx = Math.max(prevIdx, i - 1); break; }
    newIdx = i;
  }
  currentSampleIdx = newIdx;

  const events = collectEvents(prevIdx, newIdx);
  const roundEnded = currentTime >= maxTime;
  return { events, advanced: newIdx !== prevIdx || roundEnded, roundEnded };
}

function collectEvents(fromIdx, toIdx) {
  if (!currentRound) return null;
  const fromTick = currentRound.ticks[fromIdx] || 0;
  const toTick = currentRound.ticks[toIdx] || Infinity;
  const ev = currentRound.events;

  const result = { kills: [], damage: [], bomb: [], newSmokes: [], newInfernos: [] };

  for (const k of (ev.kills || [])) {
    if (k.t > fromTick && k.t <= toTick && !firedKills.has(k.t + '_' + k.v)) {
      firedKills.add(k.t + '_' + k.v);
      result.kills.push(k);
    }
  }
  for (const d of (ev.damage || [])) {
    if (d.t > fromTick && d.t <= toTick && !firedDamage.has(d.t + '_' + d.v + '_' + d.hp)) {
      firedDamage.add(d.t + '_' + d.v + '_' + d.hp);
      result.damage.push(d);
    }
  }
  for (const b of (ev.bomb || [])) {
    if (b.t > fromTick && b.t <= toTick) result.bomb.push(b);
  }
  const ct = currentTime;
  for (const s of (ev.smokes || [])) {
    if (s.ts <= ct && (s.te == null || s.te >= ct)) result.newSmokes.push(s);
  }
  for (const inf of (ev.infernos || [])) {
    if (inf.ts <= ct && (inf.te == null || inf.te >= ct)) result.newInfernos.push(inf);
  }
  return result;
}

/** 跳到当前回合内的某个时间点。 */
export function seekTo(timeSec) {
  if (!currentRound) return;
  const secs = currentRound.round_seconds;
  let idx = 0;
  for (let i = 0; i < secs.length; i++) {
    if (secs[i] >= timeSec) { idx = i; break; }
    idx = i;
  }
  currentSampleIdx = idx;
  currentTime = secs[idx] || timeSec;

  firedKills.clear();
  firedDamage.clear();
  const ct = currentRound.ticks[idx] || 0;
  for (const k of (currentRound.events.kills || [])) if (k.t <= ct) firedKills.add(k.t + '_' + k.v);
  for (const d of (currentRound.events.damage || [])) {
    if (d.t <= ct) firedDamage.add(d.t + '_' + d.v + '_' + d.hp);
  }
}

// ── 玩家状态 ────────────────────────────────────────────────────────────────

/** 稀疏 inventory → 某个采样点的物品列表。 */
function inventoryAt(pl, i) {
  const inv = pl.inventory;
  if (!inv || !inv.length) return [];
  if (pl._invCache.idx === i) return pl._invCache.items;
  let lo = 0, hi = inv.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (inv[mid].i <= i) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  const items = ans >= 0 ? inv[ans].v : [];
  pl._invCache = { idx: i, items };
  return items;
}

/** 共享的空表：没有 spotted 列时所有人都会拿到它，避免每次分配。 */
const EMPTY_SPOTTED = [];

/** 稀疏 spotted → 某个采样点上"能看到 pl 的敌人"编号列表。 */
function spottedAt(pl, i) {
  const sp = pl.spotted;
  if (!sp || !sp.length) return EMPTY_SPOTTED;
  if (pl._spotCache.idx === i) return pl._spotCache.items;
  let lo = 0, hi = sp.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sp[mid].i <= i) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  const items = ans >= 0 ? sp[ans].v : EMPTY_SPOTTED;
  pl._spotCache = { idx: i, items };
  return items;
}

/**
 * **某个玩家此刻能看到的敌人**（用于画头顶的眼睛图标）。
 *
 * ⚠️ 方向别搞反：`spotted[E]` 记的是「**能看到 E 的敌人**」，不是「E 能看到的
 * 敌人」。所以" S 看得见 E "⟺ `S ∈ spotted[E]`，要**反着查**。
 *
 * 这一条在 de_inferno / de_ancient 上用几何独立验证过：把 `spotted[i]` 里
 * 单向出现（`i ∉ spotted[j]`）的 (i, j) 对挑出来，量 j 的朝向和 j→i 连线的
 * 夹角 —— 471 对里 94.9% 在 20° 以内，而 i 的朝向只有 52.2% 在 20° 以内。
 * 也就是说确实是 **j 在看 i**。
 *
 * ⚠️ 还必须自己按 `alive` 过滤。`approximate_spotted_by` 在 pawn 消失（死亡）
 * 之后**留着最后一个值不清零**：实测 de_inferno 一整场 130,519 个 spotted 对里，
 * 123,310 个的"看到者"其实已经死了，另有 1,174 个的"被看到者"已经死了。
 * 不滤的话会看到"死人的眼睛飘在活人头上"。
 *
 * `spotted` 里本来就只会有敌人（提取时按队伍过滤过），这里再按队伍过一遍，
 * 是为了让"只在敌方头顶画"这条约束在渲染层也成立、不依赖数据来源。
 *
 * @returns {number[]} 敌人玩家下标（升序，可能为空）
 */
export function visibleEnemiesOf(viewerIdx, sampleIdx) {
  const out = [];
  if (!currentRound || viewerIdx == null || viewerIdx < 0) return out;
  const players = currentRound.players;
  if (viewerIdx >= players.length) return out;
  const teams = currentRound.teams || [];
  const viewerTeam = teams[viewerIdx];
  const n = currentRound.ticks.length;
  const i = Math.min(Math.max(0, sampleIdx | 0), Math.max(0, n - 1));

  // 死人看不了东西（数据里的 stale 值不能信）
  if (!aliveAt(players[viewerIdx], i)) return out;

  for (let e = 0; e < players.length; e++) {
    if (e === viewerIdx) continue;
    if (viewerTeam && teams[e] === viewerTeam) continue;   // 只看敌方
    if (!aliveAt(players[e], i)) continue;                 // 死人的 spotted 是残留值
    const seenBy = spottedAt(players[e], i);
    if (seenBy.length && seenBy.indexOf(viewerIdx) >= 0) out.push(e);
  }
  return out;
}

/** `alive` 列缺失时当作活着（老数据没有这一列时不要把所有人都滤掉）。 */
function aliveAt(pl, i) {
  return pl.alive ? !!pl.alive[i] : true;
}

/** 调试 / 测试用：某个采样点上"能看到 pl 的敌人"（不按查看者过滤）。 */
export function spottedBy(playerIdx, sampleIdx) {
  if (!currentRound) return [];
  const pl = currentRound.players[playerIdx];
  if (!pl) return [];
  const n = currentRound.ticks.length;
  const i = Math.min(Math.max(0, sampleIdx | 0), Math.max(0, n - 1));
  return spottedAt(pl, i).slice();
}

/**
 * 这份数据到底有没有 `spotted` 列（`--no-spotted` 提取的文件没有）。
 *
 * 只信**摘要**里的 `has_spotted` —— 服务端是从 `.csz` 第一回合的玩家列直接看
 * 有没有这个字段的。**不能**从解码后的回合判断：传输层总是带 `spotted` 这个键
 * （`--no-spotted` 提取的会是一个空表），所以"有没有这一列"和"这一列是不是
 * 全是空"在载荷里长得一模一样；也不能用"有没有非空值"来判断，一整个回合里
 * 一次可见性都没发生过是完全正常的。
 */
export function hasSpottedData() {
  return !!(matchData && matchData.has_spotted);
}

/**
 * 某个采样点的玩家状态；位置/朝向按 currentTime 在采样点之间插值。
 * 字段名与旧版一致，visuals.js / main.js 直接可用。
 */
export function getPlayerState(playerIdx, sampleIdx) {
  if (!currentRound) return null;
  const p = currentRound.players[playerIdx];
  if (!p) return null;

  const n = currentRound.ticks.length;
  const i = Math.min(Math.max(0, sampleIdx), n - 1);
  const secs = currentRound.round_seconds;

  let t = 0;
  if (i < n - 1 && p.alive[i]) {
    const t0 = secs[i];
    const t1 = secs[i + 1];
    const interval = t1 - t0;
    if (interval > 0.001) t = Math.max(0, Math.min(1, (currentTime - t0) / interval));
  }

  const l = (arr, i, i2) => {
    if (t <= 0 || i2 >= n) return arr[i] || 0;
    return lerp(arr[i] || 0, arr[i2] || 0, t);
  };
  // yaw 是**角度**，不能直接用 lerp。数据里它是 [-180, 180]（demo 的
  // `m_angEyeAngles` 就是这个范围），连续转身跨过 seam 时相邻两个采样点长得像
  // "差了 350°"：普通 lerp 会绕远路，15 ms 内把镜头甩到背后再甩回来 —— 就是
  // 选手第一人称里那种"抽搐"。实测 `de_anubis` 第 1 回合 ZywOo 在 80.56–81.02 s
  // 连续跨 seam 四次，其中 178.1224 → -175.5416 这一段的中点被插成 1.29°，
  // 相对真实轨迹偏了 176.8°。走最短弧之后这一段一共只转 6.34°。
  // pitch 不需要：它被夹在 ±89°，不存在 seam。
  const lYaw = (arr, i, i2) => {
    if (t <= 0 || i2 >= n) return arr[i] || 0;
    return lerpAngle(arr[i] || 0, arr[i2] || 0, t);
  };
  const i2 = Math.min(i + 1, n - 1);
  const periodic = t > 0 && i < n - 1 && p.alive[i];

  // 蹲下量是连续的，所以跟着位置一起插值；crouching 是阶跃的，只能取整点。
  const crouchNow = periodic ? l(p.crouch, i, i2) : (p.crouch[i] || 0);

  return {
    x: l(p.x, i, i2),
    y: l(p.y, i, i2),
    z: l(p.z, i, i2),
    yaw: lYaw(p.yaw, i, i2),
    pitch: periodic ? l(p.pitch, i, i2) : (p.pitch[i] || 0),
    v: periodic ? l(p.v, i, i2) : (p.v[i] || 0),
    hp: p.hp[i] || 0,
    armor: p.armor[i] || 0,
    helmet: !!p.helmet[i],
    defuser: !!p.defuser[i],
    alive: !!p.alive[i],
    on_ground: !!p.on_ground[i],
    crouch: crouchNow,
    crouching: !!p.crouching[i],
    // 眼高相对**脚底**，而 (x, y, z) 就是脚底 —— 所以眼睛世界坐标是
    // (x, y, z + eye_height)，不是 (x, y, z)。
    eye_height: eyeHeightFor(crouchNow),
    weapon: p.weapon[i],
    inventory: inventoryAt(p, i),
    flash: p.flash[i] || 0,
    flash_alpha: p.flash_alpha[i] || 0,
    place: p.place[i],
    shots: p.shots[i] || 0,
    footsteps: p.footsteps[i] || 0,
    jump: p.jump[i] || 0,
    land: p.land[i] || 0,
  };
}

export function getAllPlayerStates(sampleIdx) {
  if (!currentRound) return [];
  const states = [];
  for (let i = 0; i < 10; i++) states.push(getPlayerState(i, sampleIdx));
  return states;
}

// ── 炸弹 ────────────────────────────────────────────────────────────────────

/** C4 位置（插值）；没拿在手上时返回 null。 */
export function getBombPosition(sampleIdx) {
  if (!currentRound) return null;
  const bp = currentRound.bomb_position;
  const n = bp.length / 3;
  if (!n) return null;
  const i = Math.min(sampleIdx, n - 1);
  const secs = currentRound.round_seconds;

  let t = 0;
  const hasA = !Number.isNaN(bp[i * 3]);
  const hasB = i + 1 < n && !Number.isNaN(bp[(i + 1) * 3]);
  if (i < n - 1 && hasA && hasB) {
    const interval = secs[i + 1] - secs[i];
    if (interval > 0.001) t = Math.max(0, Math.min(1, (currentTime - secs[i]) / interval));
  }
  if (!hasA) return hasB ? [bp[(i + 1) * 3], bp[(i + 1) * 3 + 1], bp[(i + 1) * 3 + 2]] : null;

  const ax = bp[i * 3], ay = bp[i * 3 + 1], az = bp[i * 3 + 2];
  if (t <= 0 || !hasB) return [ax, ay, az];
  const bx = bp[(i + 1) * 3], by = bp[(i + 1) * 3 + 1], bz = bp[(i + 1) * 3 + 2];
  return [lerp(ax, bx, t), lerp(ay, by, t), lerp(az, bz, t)];
}

export function isBombPlanted(sampleIdx) {
  if (!currentRound) return false;
  const i = Math.min(sampleIdx, currentRound.bomb_planted.length - 1);
  return !!currentRound.bomb_planted[i];
}

// ── 投掷物轨迹 ──────────────────────────────────────────────────────────────

export function tickToTime(absTick) {
  if (!currentRound) return absTick / 64.0;
  const ticks = currentRound.ticks;
  const secs = currentRound.round_seconds;
  if (absTick <= ticks[0]) return secs[0];
  if (absTick >= ticks[ticks.length - 1]) return secs[secs.length - 1];
  for (let i = 0; i < ticks.length - 1; i++) {
    if (absTick >= ticks[i] && absTick <= ticks[i + 1]) {
      const frac = (absTick - ticks[i]) / (ticks[i + 1] - ticks[i]);
      return secs[i] + frac * (secs[i + 1] - secs[i]);
    }
  }
  return secs[0] + (absTick - ticks[0]) / 64.0;
}

export function buildGrenadeTrajectories() {
  if (!currentRound) return [];
  const raw = currentRound.events.grenades || [];
  if (raw.length === 0) return [];

  // 一个投掷物 = 一个 eid。`grenade_entity_id` 在整个飞行期间不变，实测两场比赛
  // （de_inferno 581 个 + de_ancient 294 个投掷物）：0 个非法 eid、0 个跨类型、
  // 0 个中途断开、0 个同回合内被复用。所以 eid 就是权威的分组键。
  //
  // 旧实现按 `thrower|type` 分组，再用"和上一点距离 < 300 HU 就接上"的启发式切分。
  // 同一个玩家连丢两颗同类投掷物时，第二颗的起点常常落在第一颗终点 300 HU 以内，
  // 于是两颗被并成一条轨迹 —— 表现就是"第一颗落地后不消失，继续飘到第二颗的落点"。
  // 实测 mercury 在 de_inferno 第 7 回合的两颗闪光：第一颗终点 (396,1850,232)、
  // 第二颗起点 (468,1854,213)，只差 74 HU，被并成 1 条 208 点的轨迹
  // （tick 56693..58497，跨越了中间 598 tick 的空档）。
  const byEid = new Map();
  const fallback = new Map();
  for (const g of raw) {
    const eid = g.eid;
    if (typeof eid === 'number' && Number.isFinite(eid) && eid >= 0) {
      let bucket = byEid.get(eid);
      if (!bucket) { bucket = []; byEid.set(eid, bucket); }
      bucket.push(g);
    } else {
      // eid 缺失时才退回旧的就近归并（真实 demo 上还没遇到过）
      const key = `${g.th}|${g.ty}`;
      let bucket = fallback.get(key);
      if (!bucket) { bucket = []; fallback.set(key, bucket); }
      bucket.push(g);
    }
  }

  const trajectories = [];
  for (const [eid, events] of byEid) {
    events.sort((a, b) => a.t - b.t);
    if (events.length >= 2) trajectories.push(_makeTraj(events, eid));
  }
  for (const [, events] of fallback) {
    events.sort((a, b) => a.t - b.t);
    for (const batch of _splitByProximity(events)) {
      if (batch.length >= 2) trajectories.push(_makeTraj(batch, null));
    }
  }
  trajectories.sort((a, b) => a.points[0].t - b.points[0].t);
  return trajectories;
}

/** eid 缺失时的兜底：按"和上一点距离 < 300 HU"把采样切成若干段。 */
function _splitByProximity(events) {
  const tracks = [];
  for (const e of events) {
    let best = null;
    let bestDist = 300;
    for (const tr of tracks) {
      const d = _dist(e, { x: tr.lastX, y: tr.lastY, z: tr.lastZ });
      if (d < bestDist) { bestDist = d; best = tr; }
    }
    if (best) {
      best.lastX = e.x; best.lastY = e.y; best.lastZ = e.z;
      best.points.push(e);
    } else {
      tracks.push({ lastX: e.x, lastY: e.y, lastZ: e.z, points: [e] });
    }
  }
  return tracks.map((tr) => tr.points);
}

function _dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function _makeTraj(batch, eid) {
  const points = batch.map((b) => ({ t: tickToTime(b.t), x: b.x, y: b.y, z: b.z }));
  let settleTime = points[points.length - 1].t;
  for (let i = points.length - 2; i >= 1; i--) {
    if (_dist(points[i], points[i + 1]) > 5) break;
    settleTime = points[i].t;
  }
  return { thrower: batch[0].th, type: batch[0].ty, eid, points, settleTime };
}

// ── 查表 ────────────────────────────────────────────────────────────────────

function lerp(a, b, t) { return a + (b - a) * Math.min(Math.max(t, 0), 1); }

/**
 * 角度插值：从 a 转到 b 走**最短弧**（度）。
 *
 * 两个采样点之间的 yaw 只是"这一小段时间里转了多少度"，跨 ±180 时两端数值
 * 相差 350°+ 并不代表真转了一圈。取差值折回 [-180, 180) 再乘 t，就是转过的
 * 那一小段 —— 结果可能落在 [-360, 360] 之外一点，但所有消费方（three 的
 * `rotation`、`viewBasis()`、雷达的 `ctx.rotate()`）都只看三角函数 / 模 360，
 * 不需要归一化。
 */
function lerpAngle(a, b, t) {
  let d = (b - a) % 360;
  if (d > 180) d -= 360;
  else if (d < -180) d += 360;
  return a + d * Math.min(Math.max(t, 0), 1);
}

export function weaponName(idx) {
  if (idx === undefined || idx === null || idx < 0) return '—';
  return idxToWeapon[idx] || `weapon_${idx}`;
}

export function placeName(idx) {
  if (idx === undefined || idx === null || idx < 0) return '—';
  return idxToPlace[idx] || `place_${idx}`;
}

export function playerName(idx) {
  if (!matchData || idx < 0 || idx >= matchData.players.length) return '?';
  return matchData.players[idx].name;
}

export function playerTeam(idx) {
  if (!currentRound || idx < 0 || idx >= currentRound.teams.length) return '?';
  return currentRound.teams[idx];
}

export function playerTeamInRound(playerIdx, round) {
  if (!round || playerIdx < 0 || playerIdx >= (round.teams?.length || 0)) return '?';
  return round.teams[playerIdx];
}

export function teamColorClass(team) { return team === 'CT' ? 'ct' : 't'; }

export function grenadeCategory(typeStr) {
  if (!typeStr) return 'unknown';
  if (typeStr.includes('Smoke')) return 'smoke';
  if (typeStr.includes('Flash')) return 'flash';
  if (typeStr.includes('HE')) return 'he';
  if (typeStr.includes('Incendiary') || typeStr.includes('Molotov')) return 'fire';
  if (typeStr.includes('Decoy')) return 'decoy';
  return 'unknown';
}

export function grenadeColor(typeStr) {
  switch (grenadeCategory(typeStr)) {
    case 'smoke': return 0x888888;
    case 'flash': return 0xf0f0f0;
    case 'he':    return 0xff4444;
    case 'fire':  return 0xff6633;
    case 'decoy': return 0xffcc00;
    default:      return 0x00ff00;
  }
}

export function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 10);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
}

export function getMapName() {
  return matchData?.map || 'unknown';
}
