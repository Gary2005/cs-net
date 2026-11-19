/**
 * test_player_xray.mjs — `player-xray.js`（「摄像机 → 选手」的遮挡判定）测试。
 *
 * 纯 Node：不需要浏览器、不需要 demo、不需要地图文件。分五块：
 *
 *   A 解析用例   能手算出答案的几何：空地 / 一堵墙 / 矮墙只挡腿 / 站着 vs 蹲着 /
 *                墙在选手**后面**不算遮挡
 *   B 暴力对拍   随机三角形汤上，判定必须和**逐三角形暴力扫描**完全一致 ——
 *                先对纯逻辑（暴力 caster），再对整条真实链路（BVH + 轴置换 + SCALE）
 *   C 语义       烟挡 = 被挡住、关掉烟 = 直接看得见、烟在选手后面不算
 *   D 不变量     死人 / 空位 / 摄像机贴在身上 / 结果数组复用 / 统计自洽
 *   E 性能       13 万三角形的"类地图"上量 10 个选手的每帧耗时
 *
 * 用法：node visualizer/tests/test_player_xray.mjs
 */

import { buildBVH } from '../static/js/map-raycast.js';
import { makeMapCaster, smokeRayBlocked } from '../static/js/visibility.js';
import {
  createPlayerXRayTester, classifyPlayerXRay, drawnHeightFor,
  XRAY_SAMPLE_COUNT, XRAY_SAMPLE_FRACTIONS, XRAY_BLOCK_EPS_HU,
} from '../static/js/player-xray.js';

let passed = 0, failed = 0;
function check(name, ok, detail = '') {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '   ' + detail : ''}`); }
}
function eq(name, got, want) {
  check(name, got === want, `got ${got}, want ${want}`);
}

// ── 共用工具（和 test_visibility.mjs 同一套写法）─────────────────────────────

/** 确定性 PRNG（mulberry32）—— 测试必须可复现。 */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// map-raycast.js 的 rayTri 是模块私有的；这里逐行复制一份当暴力参考。
// 常量必须和那边一致（EPS / BARY_EPS 都是 1e-9），否则边界情形会对不上。
const EPS = 1e-9;
const BARY_EPS = 1e-9;
function rayTri(verts, t, ox, oy, oz, dx, dy, dz) {
  const i = t * 9;
  const ax = verts[i], ay = verts[i + 1], az = verts[i + 2];
  const e1x = verts[i + 3] - ax, e1y = verts[i + 4] - ay, e1z = verts[i + 5] - az;
  const e2x = verts[i + 6] - ax, e2y = verts[i + 7] - ay, e2z = verts[i + 8] - az;
  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (det > -EPS && det < EPS) return -1;
  const inv = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * inv;
  if (u < -BARY_EPS || u > 1 + BARY_EPS) return -1;
  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v < -BARY_EPS || u + v > 1 + BARY_EPS) return -1;
  return (e2x * qx + e2y * qy + e2z * qz) * inv;
}

function bruteForce(verts, triCount, ox, oy, oz, dx, dy, dz, maxDist) {
  let best = maxDist, found = false;
  for (let t = 0; t < triCount; t++) {
    const hit = rayTri(verts, t, ox, oy, oz, dx, dy, dz);
    if (hit >= 0 && hit < best) { best = hit; found = true; }
  }
  return found ? best : -1;
}

/** 暴力扫描版的 castMapRay（game 空间）—— 参考实现。 */
function bruteCaster(verts, triCount) {
  return (ox, oy, oz, dx, dy, dz, maxDist) =>
    bruteForce(verts, triCount, ox, oy, oz, dx, dy, dz, maxDist);
}

function soup(tris) {
  const verts = new Float32Array(tris.length * 9);
  for (let i = 0; i < tris.length; i++) verts.set(tris[i], i * 9);
  return { verts, triCount: tris.length };
}

/** 一个轴对齐的盒子拆成 12 个三角形，写进 tris 数组。 */
function pushBox(tris, x0, y0, z0, x1, y1, z1) {
  const v = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const faces = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4],
  ];
  for (const [a, b, c, d] of faces) {
    tris.push([...v[a], ...v[b], ...v[c]]);
    tris.push([...v[a], ...v[c], ...v[d]]);
  }
}

/**
 * 一堵垂直于 X 轴、位于 x=value 的墙。
 * `z1` 就是墙高：矮墙（z1=50）只挡得住身体下半截，全高墙（z1=200）全挡。
 */
function wallX(value, z1 = 200, halfSpan = 400) {
  const tris = [];
  pushBox(tris, value - 0.5, -halfSpan, -100, value + 0.5, halfSpan, z1);
  return tris;
}

/** 一个全空的烟雾体素栅格。 */
function emptySmokeGrid(dim = 32) { return new Uint8Array(dim * dim * dim); }

/** 造一个烟雾遮挡体：`fill(i,j,k)` 返回 true 表示该格被占据。 */
function smokeVolume(det, fill, cell = 12, dim = 32) {
  const grid = emptySmokeGrid(dim);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      for (let k = 0; k < dim; k++) {
        if (fill(i, j, k)) grid[(i * dim + j) * dim + k] = 1;
      }
    }
  }
  return { det, grid, dim, cell };
}

function player(x, y, z, opts = {}) {
  return {
    x, y, z, yaw: 0, pitch: 0,
    crouch: opts.crouch ?? 0,
    crouching: !!opts.crouching,
    alive: opts.alive !== false,
  };
}

/** 摄像机的 game 空间位置（站着的人眼高 ≈ 64 HU）。 */
const CAM = { x: 0, y: 0, z: 64 };

/** 只有 1 个选手时，取他的判定结果。 */
function flagOf(cam, states, opts = {}) {
  return classifyPlayerXRay(cam, states, opts)[0];
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== A 解析用例');
// ═══════════════════════════════════════════════════════════════════════════

{
  // 没有地图（caster 为 null）：谁都不会被判成透视 —— 地图还没加载就没有墙。
  eq('没有地图 BVH 时，站在空地 = 直接看得见',
    flagOf(CAM, [player(800, 0, 0)]), 0);

  const empty = soup([]);
  const solid = soup(wallX(400));       // 全高墙
  const low = soup(wallX(400, 50));     // 只到 z=50 的矮墙

  const far = player(800, 0, 0);

  eq('中间一堵全高墙 = 只能透视看到',
    flagOf(CAM, [far], { castMapRay: bruteCaster(solid.verts, solid.triCount) }), 1);

  const behindWall = soup(wallX(1200));
  eq('墙挪到选手后面（x=1200）= 直接看得见（射线终点在选手身上，不打后面的东西）',
    flagOf(CAM, [far], { castMapRay: bruteCaster(behindWall.verts, behindWall.triCount) }), 0);
  eq('矮墙只挡到 z=50 = 直接看得见（头顶那个采样点在墙上面）',
    flagOf(CAM, [far], { castMapRay: bruteCaster(low.verts, low.triCount) }), 0);
  eq('摄像机站在矮墙和选手之间（x=200）= 直接看得见',
    flagOf({ x: 200, y: 0, z: 64 }, [far],
      { castMapRay: bruteCaster(low.verts, low.triCount) }), 0);

  // 蹲下：采样点跟着**画出来的**身高走（72 → 54），所以同一堵矮墙挡得住蹲着的人。
  const lowTall = soup(wallX(400, 58));   // 比蹲姿头顶(54)高、比站姿头顶(72)矮
  const stand = player(800, 0, 0, { crouch: 0 });
  const crouched = player(800, 0, 0, { crouch: 1, crouching: true });
  eq('z=58 的墙：站着的人头顶露在外面 = 直接看得见',
    flagOf(CAM, [stand], { castMapRay: bruteCaster(lowTall.verts, lowTall.triCount) }), 0);
  eq('z=58 的墙：同一个人蹲下（头顶 54）= 只能透视看到',
    flagOf(CAM, [crouched], { castMapRay: bruteCaster(lowTall.verts, lowTall.triCount) }), 1);

  // drawnHeightFor 必须和 visuals.js 里那具身体同一个式子（bodyRatio·72）
  eq('drawnHeightFor(crouch=0) = 72', drawnHeightFor({ crouch: 0 }), 72);
  eq('drawnHeightFor(crouch=1) = 54', drawnHeightFor({ crouch: 1 }), 54);
  eq('drawnHeightFor(crouch=0.5) = 63（连续压缩，不是 54/72 阶跃）',
    drawnHeightFor({ crouch: 0.5 }), 63);
  eq('drawnHeightFor 对越界的 crouch 也收敛在 [54, 72]',
    drawnHeightFor({ crouch: 3 }) + drawnHeightFor({ crouch: -3 }), 126);

  // 摄像机正好落在采样点上（第一人称看自己 / 贴脸看）：距离退化，方向向量会
  // 变成 0/0。这一条不只是"结果是 0"，还要求**一条射线都不发** —— 那才是
  // 距离早退真的生效了的证据。
  const headZ = drawnHeightFor({ crouch: 0 }) * XRAY_SAMPLE_FRACTIONS[0];
  const near = createPlayerXRayTester({ castMapRay: bruteCaster(solid.verts, solid.triCount) })
    .classify({ x: 800, y: 0, z: headZ }, [player(800, 0, 0)]);
  eq('摄像机正好在头顶采样点上 = 直接看得见', near.flags[0], 0);
  eq('  而且一条射线都没发（距离退化时直接收工，不会算出 0/0）', near.stats.rays, 0);

  // 容差就是那个常量，别在实现里写死另一个数
  eq('XRAY_BLOCK_EPS_HU = 1', XRAY_BLOCK_EPS_HU, 1);
  eq('采样点个数 = 3', XRAY_SAMPLE_COUNT, 3);
  check('采样点分数从头到脚单调递减',
    XRAY_SAMPLE_FRACTIONS.every((f, i) => i === 0 || f < XRAY_SAMPLE_FRACTIONS[i - 1]),
    JSON.stringify(XRAY_SAMPLE_FRACTIONS));
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== B 暴力对拍');
// ═══════════════════════════════════════════════════════════════════════════

/**
 * 按 player-xray.js 的同一条规则算期望值，但用**暴力扫描**当地图遮挡。
 * 和实现唯一的区别就是"墙打哪儿"，所以对拍出来的是实现自己的逻辑。
 */
function expectedFlags(cam, states, verts, triCount, occluders, useSmoke) {
  const out = new Uint8Array(states.length);
  for (let i = 0; i < states.length; i++) {
    const s = states[i];
    if (!s || !s.alive) { out[i] = 0; continue; }
    const h = drawnHeightFor(s);
    let clear = false;
    for (let k = 0; k < XRAY_SAMPLE_COUNT; k++) {
      const pz = s.z + h * XRAY_SAMPLE_FRACTIONS[k];
      let dx = s.x - cam.x, dy = s.y - cam.y, dz = pz - cam.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist <= XRAY_BLOCK_EPS_HU) { clear = true; break; }
      dx /= dist; dy /= dist; dz /= dist;
      const maxD = dist - XRAY_BLOCK_EPS_HU;
      let blocked = false;
      if (maxD > 0) {
        if (bruteForce(verts, triCount, cam.x, cam.y, cam.z, dx, dy, dz, maxD) >= 0) {
          blocked = true;
        } else if (useSmoke
          && smokeRayBlocked(occluders, cam.x, cam.y, cam.z, dx, dy, dz, maxD)) {
          blocked = true;
        }
      }
      if (!blocked) { clear = true; break; }
    }
    out[i] = clear ? 0 : 1;
  }
  return out;
}

/**
 * 造一个**一定会同时出现两种结果**的场景，B1 / B2 共用。
 *
 * 关键是别把摄像机埋进三角形汤里 —— 那样每条射线都会立刻撞到东西，对拍就
 * 退化成"全是 1"，等于什么都没验。这里把随机三角形堆在 x ∈ [150, 550] 的板子
 * 里当遮挡，选手一半放在板子后面（会被挡）、一半放在摄像机这一侧（看得见），
 * 板子中间再插一堵实墙保证一定有确定的遮挡。
 */
function mixedScene(seed) {
  const r = rng(seed);
  const tris = [];
  for (let i = 0; i < 400; i++) {
    const t = [];
    for (let k = 0; k < 3; k++) t.push(150 + r() * 400, (r() * 2 - 1) * 600, r() * 300 - 50);
    tris.push(t);
  }
  tris.push(...wallX(300, 220, 800));

  const states = [];
  for (let i = 0; i < 10; i++) {
    const behind = i % 2 === 0;
    states.push(player(
      behind ? 650 + r() * 300 : -(100 + r() * 600),
      (r() * 2 - 1) * 600,
      r() * 40,
      { crouch: i % 5 === 0 ? 1 : 0, crouching: i % 5 === 0 },
    ));
  }
  return { tris, states };
}

/** 摄像机：站着的人眼高，正对着那块板子。 */
const MIXED_CAM = { x: 0, y: 0, z: 64 };

{
  // ── B1 纯逻辑：暴力 caster vs 参考实现（同样的暴力扫描）──────────────────
  let mismatch = 0, seenClear = 0, seenBlocked = 0, cases = 0;
  for (const seed of [1, 7, 42, 1234, 99999]) {
    const { tris, states } = mixedScene(seed);
    // 一个死人 + 一个空位：两边都必须无视
    states[8] = player(0, 0, 0, { alive: false });
    states[9] = null;
    const { verts, triCount } = soup(tris);

    const tester = createPlayerXRayTester({
      castMapRay: bruteCaster(verts, triCount), flags: new Uint8Array(10),
    });
    const got = tester.classify(MIXED_CAM, states).flags;
    const want = expectedFlags(MIXED_CAM, states, verts, triCount, [], false);

    for (let i = 0; i < states.length; i++) {
      cases++;
      if (got[i] !== want[i]) mismatch++;
      if (got[i] === 1) seenBlocked++; else seenClear++;
    }
  }
  check(`B1 暴力 caster 下 5 个种子 × 10 人全部一致（${cases} 个判定，不一致 ${mismatch}）`,
    mismatch === 0, `mismatch=${mismatch}`);
  check('B1 对拍样本里"直接看得见"和"只能透视"都出现过',
    seenClear > 0 && seenBlocked > 0, `clear=${seenClear} blocked=${seenBlocked}`);

  // ── B2 整条真实链路：BVH + makeMapCaster（轴置换 + SCALE）vs 暴力 ────────
  // game → three 是 `(gx,gy,gz) → (gy,gz,gx)·SCALE`（scene.js::gameToThree）。
  const SCALE = 0.0254;
  let mismatch2 = 0, cases2 = 0, seenClear2 = 0, seenBlocked2 = 0;
  for (const seed of [3, 55, 808]) {
    const { tris, states } = mixedScene(seed);
    const game = soup(tris);

    // 同一份三角形烘焙到 three 空间，BVH 建在那上面
    const three = new Float32Array(game.verts.length);
    for (let i = 0; i < game.verts.length; i += 3) {
      const gx = game.verts[i], gy = game.verts[i + 1], gz = game.verts[i + 2];
      three[i] = gy * SCALE; three[i + 1] = gz * SCALE; three[i + 2] = gx * SCALE;
    }
    const bvh = buildBVH({ verts: three, triCount: game.triCount });
    const caster = makeMapCaster(bvh, SCALE);
    check(`B2 种子 ${seed}：makeMapCaster 拿到了 BVH`, typeof caster === 'function');

    const got = createPlayerXRayTester({ castMapRay: caster }).classify(MIXED_CAM, states).flags;
    const want = expectedFlags(MIXED_CAM, states, game.verts, game.triCount, [], false);
    for (let i = 0; i < states.length; i++) {
      cases2++;
      if (got[i] !== want[i]) mismatch2++;
      if (got[i] === 1) seenBlocked2++; else seenClear2++;
    }
  }
  check(`B2 BVH 管线（含轴置换 / SCALE）下 3 个种子 × 10 人一致`
    + `（${cases2} 个判定，不一致 ${mismatch2}）`, mismatch2 === 0, `mismatch=${mismatch2}`);
  check('B2 对拍样本里两种结果都出现过',
    seenClear2 > 0 && seenBlocked2 > 0, `clear=${seenClear2} blocked=${seenBlocked2}`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== C 语义（烟雾）');
// ═══════════════════════════════════════════════════════════════════════════

{
  const far = player(800, 0, 0);
  // 摄像机 (0,0,64) 与选手之间（x≈400）放一整团实的烟
  const between = [smokeVolume([400, 0, 64], () => true)];
  const behind = [smokeVolume([1300, 0, 64], () => true)];

  const t1 = createPlayerXRayTester({});
  t1.setOccluders(between);
  const r1 = t1.classify(CAM, [far]);
  eq('中间一团烟 = 只能透视看到', r1.flags[0], 1);
  eq('  这一条是被烟挡的（mapHits 为 0）', r1.stats.mapHits, 0);
  check('  统计里记到了烟命中', r1.stats.smokeHits > 0, `smokeHits=${r1.stats.smokeHits}`);
  eq('  烟遮挡体个数', t1.occluderCount, 1);

  const t2 = createPlayerXRayTester({ smoke: false });
  t2.setOccluders(between);
  eq('关掉烟（smoke:false）= 直接看得见', t2.classify(CAM, [far]).flags[0], 0);

  const t3 = createPlayerXRayTester({});
  t3.setOccluders(behind);
  const r3 = t3.classify(CAM, [far]);
  eq('烟在选手后面 = 直接看得见（maxDist 收到目标上）', r3.flags[0], 0);
  eq('  一次烟命中都不该记到', r3.stats.smokeHits, 0);

  const t4 = createPlayerXRayTester({});
  t4.setOccluders([]);
  eq('没烟的时候直接看得见', t4.classify(CAM, [far]).flags[0], 0);
  eq('  没人被挡住', t4.classify(CAM, [far]).stats.blocked, 0);

  // 烟只在"第一堵墙没挡住"之后才测：被墙挡住的采样点不用再去问烟
  const wall = soup(wallX(400));
  const t5 = createPlayerXRayTester({ castMapRay: bruteCaster(wall.verts, wall.triCount) });
  t5.setOccluders(between);
  const r5 = t5.classify(CAM, [far]);
  eq('墙 + 烟同时存在 = 只能透视看到', r5.flags[0], 1);
  check('  墙先命中，烟就没必要再测（mapHits > 0 且 smokeHits = 0）',
    r5.stats.mapHits > 0 && r5.stats.smokeHits === 0,
    `mapHits=${r5.stats.mapHits} smokeHits=${r5.stats.smokeHits}`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== D 不变量');
// ═══════════════════════════════════════════════════════════════════════════

{
  const wall = soup(wallX(400));
  const caster = bruteCaster(wall.verts, wall.triCount);

  const states = [
    player(800, 0, 0),                       // 0 被墙挡
    player(200, 0, 0),                       // 1 墙这边
    player(800, 200, 0, { alive: false }),   // 2 死人
    null,                                    // 3 空位
    player(800, -200, 0, { crouch: 1, crouching: true }),  // 4 蹲着被挡
  ];
  const tester = createPlayerXRayTester({ castMapRay: caster, flags: new Uint8Array(10) });
  const res = tester.classify(CAM, states);

  eq('活的、墙后面的人 = 1', res.flags[0], 1);
  eq('墙这边的人 = 0', res.flags[1], 0);
  eq('死人 = 0（本来就不画，省射线）', res.flags[2], 0);
  eq('空位 = 0', res.flags[3], 0);
  eq('蹲在墙后的人 = 1', res.flags[4], 1);

  eq('统计：只数活人', res.stats.players, 3);
  eq('统计：被挡住的人数', res.stats.blocked, 2);
  check('统计：n 是数组长度（含死人和空位）', res.stats.n === 5, `n=${res.stats.n}`);
  check('统计：rays 不超过 活人 × 采样点', res.stats.rays <= 3 * XRAY_SAMPLE_COUNT,
    `rays=${res.stats.rays}`);
  check('统计：ms 是个非负数', typeof res.stats.ms === 'number' && res.stats.ms >= 0);
  eq('统计：mapReady 反映 caster 在不在', res.stats.mapReady, true);
  check('tester.stats 和最近一次的结果是同一个对象', tester.stats === res.stats);

  // 复用：同一个 tester 反复 classify，返回的必须是**同一个**数组（热路径不分配）
  const again = tester.classify(CAM, states);
  check('classify() 复用同一个结果数组（不每帧分配）', again.flags === res.flags);

  // 人数变多：内部数组自动长大，且不会把上一次的结果留在后面
  const many = [];
  for (let i = 0; i < 14; i++) many.push(player(800, 0, 0));
  const grown = tester.classify(CAM, many);
  check('人数变多时结果数组自动长大', grown.flags.length >= 14, `${grown.flags.length}`);
  eq('长大之后每个人仍然判对（都该是 1）', grown.flags[13], 1);

  // 空输入
  const empty = tester.classify(CAM, []);
  eq('没有玩家时不报错、也没人被挡住', empty.stats.players, 0);
  eq('没有玩家时 rays = 0', empty.stats.rays, 0);

  // 不传 states 也不能炸
  eq('states 为 undefined 时 n = 0', tester.classify(CAM, undefined).stats.n, 0);

  // caster 缺失 = 墙不算遮挡，但烟还算（和 visibility.js 的退化行为一致）
  const noMap = createPlayerXRayTester({});
  noMap.setOccluders([smokeVolume([400, 0, 64], () => true)]);
  const r = noMap.classify(CAM, [player(800, 0, 0)]);
  eq('没有地图时 mapReady = false', r.stats.mapReady, false);
  eq('没有地图时墙不算遮挡，但烟仍然算', r.flags[0], 1);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n=== E 性能');
// ═══════════════════════════════════════════════════════════════════════════

{
  // 造一张"类地图"：6000×6000 HU 的地板 + 16×16 堵竖墙（和 test_visibility.mjs
  // 同一个思路，只是规模小一点 —— 那边已经有一份 82 万三角形的用例了）。
  const SPAN = 6000, GRID = 256;
  const verts = new Float32Array((GRID * GRID * 2 + 16 * 16 * 4) * 9);
  let w = 0;
  const put = (ax, ay, az, bx, by, bz, cx, cy, cz) => {
    verts[w++] = ax; verts[w++] = ay; verts[w++] = az;
    verts[w++] = bx; verts[w++] = by; verts[w++] = bz;
    verts[w++] = cx; verts[w++] = cy; verts[w++] = cz;
  };
  const step = SPAN / GRID;
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      const x0 = -SPAN / 2 + i * step, y0 = -SPAN / 2 + j * step;
      const x1 = x0 + step, y1 = y0 + step;
      put(x0, y0, 0, x1, y0, 0, x1, y1, 0);
      put(x0, y0, 0, x1, y1, 0, x0, y1, 0);
    }
  }
  for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
      const cx = -SPAN / 2 + (i + 0.5) * (SPAN / 16);
      const cy = -SPAN / 2 + (j + 0.5) * (SPAN / 16);
      const hw = 120, t = 8, h = 200;
      put(cx - hw, cy - t, 0, cx + hw, cy - t, 0, cx + hw, cy - t, h);
      put(cx - hw, cy - t, 0, cx + hw, cy - t, h, cx - hw, cy - t, h);
      put(cx - hw, cy + t, 0, cx + hw, cy + t, 0, cx + hw, cy + t, h);
      put(cx - hw, cy + t, 0, cx + hw, cy + t, h, cx - hw, cy + t, h);
    }
  }
  const data = { verts: verts.subarray(0, w), triCount: w / 9 };
  const t0 = Date.now();
  const bvh = buildBVH(data);
  const buildMs = Date.now() - t0;
  console.log(`  场景：${data.triCount.toLocaleString()} 三角形，建 BVH ${buildMs} ms`);

  const SCALE = 0.0254;
  const caster = makeMapCaster(bvh, SCALE);
  const r = rng(4242);
  const states = [];
  for (let i = 0; i < 10; i++) {
    states.push(player((r() * 2 - 1) * 1500, (r() * 2 - 1) * 1500, 0,
      { crouch: i % 3 === 0 ? 1 : 0, crouching: i % 3 === 0 }));
  }
  // 摄像机站得高一点、离得远一点：典型俯瞰视角，射线又长又多
  const cam = { x: 0, y: 0, z: 1200 };
  const smoke = [smokeVolume([800, 400, 60], () => true), smokeVolume([-900, -600, 60], () => true)];

  const tester = createPlayerXRayTester({ castMapRay: caster });
  tester.setOccluders(smoke);
  tester.classify(cam, states);              // 预热

  const N = 200;
  const t1 = Date.now();
  let last = null;
  for (let i = 0; i < N; i++) last = tester.classify(cam, states);
  const ms = (Date.now() - t1) / N;

  console.log(`  实测：${ms.toFixed(4)} ms/帧，${last.stats.rays} 条射线，`
    + `${last.stats.blocked}/${last.stats.players} 个人被挡住`);
  check(`10 个选手每帧 < 1 ms（实测 ${ms.toFixed(4)}）`, ms < 1);
  // 上界：30 条**归一化**长射线的纯成本，不含任何判定逻辑
  const rays = [];
  for (let i = 0; i < 30; i++) {
    const s = states[i % 10];
    const tx = s.x, ty = s.y, tz = s.z + 40;
    let dx = tx - cam.x, dy = ty - cam.y, dz = tz - cam.z;
    const L = Math.hypot(dx, dy, dz);
    rays.push([cam.x, cam.y, cam.z, dx / L, dy / L, dz / L, L]);
  }
  const micro = () => {
    let n = 0;
    for (let i = 0; i < 30; i++) {
      const q = rays[i];
      if (caster(q[0], q[1], q[2], q[3], q[4], q[5], q[6]) >= 0) n++;
      if (smokeRayBlocked(smoke, q[0], q[1], q[2], q[3], q[4], q[5], q[6])) n++;
    }
    return n;
  };
  micro();
  const t2 = Date.now();
  for (let i = 0; i < N; i++) micro();
  const msMicro = (Date.now() - t2) / N;
  console.log(`  上界（30 条长射线，不早退）：${msMicro.toFixed(4)} ms/帧`);
  check('判定本身没有比"只发 30 条射线"慢出一个量级',
    ms < msMicro * 4 + 1, `classify=${ms.toFixed(4)} rays=${msMicro.toFixed(4)}`);
}

// ═══════════════════════════════════════════════════════════════════════════
console.log(`\nplayer-xray: ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
