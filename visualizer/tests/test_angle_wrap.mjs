#!/usr/bin/env node
/**
 * test_angle_wrap.mjs — 采样点之间插值 yaw 时必须走**最短弧**。
 *
 * 为什么需要这个测试：`getPlayerState()` 把状态按 `currentTime` 在两个采样点之间
 * 插值，而 yaw 是**角度**，数据里是 [-180, 180]（demo 的 `m_angEyeAngles`）。
 * 跨 seam 时相邻两点的值看起来差 350°+，普通 lerp 会绕远路 —— 表现为选手 POV
 * 在约 15 ms 内甩到背后再甩回来，也就是"抽搐"。
 *
 * 真实样本（`app/data/csz/aurora-vs-vitality-m1-anubis__ecd40737.csz` 第 1 回合，
 * ZywOo）：78.2656 s yaw = 178.1224，78.2812 s yaw = -175.5416。这只是 **+6.34°**
 * 的连续右转；两个采样点之间普通 lerp 会经过 1.29°，即整整 176° 的反向偏移。
 * 同一回合 80.56–81.02 s 里连续跨 seam 四次，看起来就是半秒的抽动。
 *
 * 纯 node，不需要浏览器 / 服务器 / demo：
 *
 *     node visualizer/tests/test_angle_wrap.mjs
 */
import {
  decodeRoundPayload, setRound, seekTo, setPlaying, advanceReplay, getPlayerState,
} from '../static/js/replay-core.js';

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}

const TICKRATE = 64;

const f32b64 = (arr) => Buffer.from(Float32Array.from(arr).buffer, 0, arr.length * 4).toString('base64');
const i16b64 = (arr) => Buffer.from(Int16Array.from(arr).buffer, 0, arr.length * 2).toString('base64');
const u8b64 = (arr) => Buffer.from(Uint8Array.from(arr)).toString('base64');

/** 一个只填了插值需要的那几列的合成载荷。n 由 yaw 的长度决定。 */
function payload(yaw, pitch = null, alive = null) {
  const n = yaw.length;
  const fill = (v) => new Array(n).fill(v);
  return {
    idx: 0, id: 0, teams: ['CT', 'T'], winner: 'CT', end_reason: 'target_bombed',
    start_tick: 1000, tickrate: TICKRATE,
    ticks: Buffer.from(Int32Array.from({ length: n }, (_, i) => 1000 + i).buffer, 0, n * 4)
      .toString('base64'),
    sample_count: n,
    bomb_planted_time: null,
    bomb_planted: u8b64(fill(0)),
    bomb_dropped: u8b64(fill(0)),
    bomb_position: '',
    missing_samples: [],
    players: [{
      x: f32b64(fill(0)),
      y: f32b64(fill(0)),
      z: f32b64(fill(0)),
      yaw: f32b64(yaw),
      pitch: f32b64(pitch || fill(0)),
      v: f32b64(fill(0)),
      flash: f32b64(fill(0)),
      flash_alpha: f32b64(fill(0)),
      hp: i16b64(fill(100)),
      armor: i16b64(fill(0)),
      weapon: i16b64(fill(7)),
      place: i16b64(fill(-1)),
      shots: u8b64(fill(0)),
      footsteps: u8b64(fill(0)),
      jump: u8b64(fill(0)),
      land: u8b64(fill(0)),
      flags: u8b64((alive || fill(1)).map((a) => (a ? 4 : 0) | 8)),   // alive<<2 | on_ground<<3
      crouch: u8b64(fill(0)),
      inventory: [],
      spotted: [],
    }],
    events: {}, fire: [], smoke: [],
  };
}

/** 装上合成回合，返回 { round, ... } 。 */
function mount(yaw, pitch, alive) {
  return setRound(decodeRoundPayload(payload(yaw, pitch, alive)));
}

/**
 * 从采样 i 到 i+1 的中点取状态：先 seek 到 i（t=0），再推进半个采样间隔。
 * 这是回放里真实发生的事 —— `currentTime` 是连续的，渲染帧落在采样点之间。
 */
function stateMidway(round, i) {
  seekTo(round.round_seconds[i]);
  setPlaying(true);
  advanceReplay(0.5 / TICKRATE);
  setPlaying(false);
  return getPlayerState(0, i);
}

const norm = (a) => ((a % 360) + 540) % 360 - 180;   // 折到 [-180, 180)
const angDist = (a, b) => Math.abs(norm(a - b));

// ═══════════════════════════════════════════════════════════════════════════
console.log('\nA. 跨 ±180 的 seam');

{
  // 真实样本里的那一段：178.1224 → -175.5416，实际只转了 +6.34°
  const yaw = [178.1224, -175.5416, -175.0, -174.5];
  const st = stateMidway(mount(yaw, [3.7, 3.7, 3.7, 3.7]), 0);

  // 走最短弧：中点应该是 178.1224 + 6.336/2 = 181.2904 → -178.7096
  const want = 178.1224 + 6.336 / 2;
  check('跨 seam 的中点落在 seam 另一侧（不是绕远路到 0° 附近）',
    angDist(st.yaw, want) < 0.05, `yaw=${st.yaw.toFixed(4)} want≈${want.toFixed(4)}`);
  check('中点相对起点的角位移 <= 两个采样点角位移的一半（+1° 容差）',
    angDist(st.yaw, 178.1224) <= 6.336 / 2 + 1,
    `${angDist(st.yaw, 178.1224).toFixed(3)}° of ${(6.336 / 2).toFixed(3)}°`);
  check('pitch 照常线性插值（角度不跨 seam，不该改行为）',
    Math.abs(st.pitch - 3.7) < 1e-5, `${st.pitch}`);
}

{
  // 反方向跨 seam：-178 → +179，实际只转了 -3°
  const st = stateMidway(mount([-178, 179, 179, 179]), 0);
  check('反向跨 seam 同样走最短弧', angDist(st.yaw, -179.5) < 0.05,
    `yaw=${st.yaw.toFixed(4)} want≈-179.5`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\nB. 连续跨 seam（真实样本 80.56–81.02 s，半秒内 4 次）');

{
  // ZywOo 这一段的原始采样值（每两个相邻点取中点检查）。最后一个值只为凑够
  // 采样点，不参与断言。
  const seq = [
    178.1220, -175.5416, -169.6454, -165.5976,  // 78.250 → 78.297
    176.3179, 176.8898, 178.6940, -179.8541,    // 80.797 → 80.844
    -179.4139, 179.8819, 178.7379, 178.1660,    // 80.859 → 80.906
    179.9701, -179.2378, -178.2700, -177.1700,  // 81.000 → 81.047
  ];
  const round = mount(seq);
  let worst = 0, worstAt = null;
  for (let i = 0; i < seq.length - 1; i++) {
    const st = stateMidway(round, i);
    const span = angDist(seq[i], seq[i + 1]);
    // 中点离两个端点的角距离，都不该超过实际角位移的一半（含 1° 容差）
    const d = Math.min(angDist(st.yaw, seq[i]), angDist(st.yaw, seq[i + 1]));
    if (d > worst) { worst = d; worstAt = { i, yaw: st.yaw, a: seq[i], b: seq[i + 1] }; }
    check(`segment ${i}: ${seq[i].toFixed(2)} → ${seq[i + 1].toFixed(2)} 中点不越界`,
      d <= span / 2 + 1, `mid=${st.yaw.toFixed(2)} span=${span.toFixed(2)}`);
  }
  console.log(`       (worst midpoint deviation ${worst.toFixed(3)}° at ${JSON.stringify(worstAt)})`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\nC. 不跨 seam 时必须和线性插值一致（不能顺手改掉正常行为）');

{
  const cases = [[10, 20, 15], [-90, -80, -85], [170, 175, 172.5], [0, 0, 0], [-179, 179, -180]];
  for (const [a, b, want] of cases) {
    const st = stateMidway(mount([a, b, b, b]), 0);
    check(`${a} → ${b} 的中点 = ${want}`, angDist(st.yaw, want) < 0.02,
      `yaw=${st.yaw.toFixed(4)}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\nD. 死人 / 采样点边界不插值');

{
  const round = mount([-179.5, 179.5, 179.5, 179.5], null, [0, 1, 1, 1]);
  seekTo(round.round_seconds[0]);
  setPlaying(true); advanceReplay(0.5 / TICKRATE); setPlaying(false);
  const st = getPlayerState(0, 0);
  check('死人取整点的原始 yaw（不插值）', st.yaw === -179.5, `${st.yaw}`);
  check('死人 alive = false', st.alive === false);
}

console.log(`\n${failed ? failed + ' FAILED' : `ALL OK (${passed} checks)`}`);
process.exit(failed ? 1 : 0);
