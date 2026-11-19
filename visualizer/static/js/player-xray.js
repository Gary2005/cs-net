/**
 * player-xray.js — 「**这台摄像机**此刻能不能直接看到某个选手」的纯几何判定
 * （不 import three）。
 *
 * 和另外两套「看得见吗」不是一回事，别搞混：
 *
 *   `.csz` 的 `spotted`  引擎给的，含视锥 + 声音 + 引擎自己的时序，还被限速
 *   visibility.js        **玩家 → 玩家**：眼睛 + 视锥 + 碰撞盒 8 个角落
 *   player-xray.js       **摄像机 → 玩家**（本文件）：没有视锥（屏幕上看得见就是
 *                        看得见），只看摄像机到身上那三个采样点之间有没有地图
 *                        三角形 / 烟雾
 *
 * 用途只有一个：给「选手 X 光」分两层。直接看得见的照旧画成不透明的小人，
 * 被墙挡住的画成半透明幽灵 —— 于是"真的看到"和"透过墙看到"一眼能分开。
 *
 * ── 采样点为什么是"身上三个点"，不用碰撞盒的 8 个角落 ──────────────────────
 *
 * 碰撞盒是 32×32（半宽 16 HU）的方盒，比画出来的那具身体（肩宽约 10 HU）**宽
 * 得多**，它的 8 个角落落在身体外面的空气里。于是一个只从掩体后露出半个肩膀的
 * 选手，按角落判会判成"看得见"而其实是"勉强看到"；贴着墙站的时候角落又容易穿进
 * 墙里，判成"看不见"而其实露着头。所以这里沿身体中轴取头 / 胸 / 腿三个点，
 * **任何一个通**就算直接看得见 —— 和屏幕上那具身体对得上。
 *
 * 死人一律算"直接看得见"（0）：`updatePlayers` 本来就不画他们，画不画幽灵
 * 无所谓，省下射线是实打实的。
 *
 * ── 性能 ───────────────────────────────────────────────────────────────────
 *
 * 10 个活人 × 3 个采样点 = 30 条射线上限，BVH 实测 0.0020 ms/条
 * （见 tests/test_map_raycast.mjs），加上"第一个通的采样点就收工"，实测
 * < 0.1 ms/帧（见 tests/test_player_xray.mjs 的性能用例）。烟雾走
 * `smokeRayBlocked` 的 AABB 早退，不在烟里的时候每个烟只要 3 次 slab。
 */

import { smokeRayBlocked } from './visibility.js';
import { HULL_STAND_HEIGHT, CROUCH_EYE_DROP } from './replay-core.js';

/** 沿身体中轴取几个采样点。 */
export const XRAY_SAMPLE_COUNT = 3;

/**
 * 采样点在身体高度上的位置（0 = 脚底，1 = 头顶），从头到脚。
 *
 * 0.88 ≈ 站立眼高（64/72）；0.5 是胸口；0.16 ≈ 大腿（避免正好压在地板平面上）。
 * 上中下各一个：只露出头的、只露出腿的，都能判对。
 */
export const XRAY_SAMPLE_FRACTIONS = [0.88, 0.5, 0.16];

/**
 * 命中距离必须比"到采样点的距离"近这么多才算挡住（HU）。
 *
 * 射线终点在身体内部，理论上不该碰到任何地图三角形；但腿上那个采样点离地只有
 * ~11 HU，贴地板 / 站在斜坡 / 站在箱子上时，命中距离和射线长度在浮点上分不开。
 * 1 HU = 2.54 cm，和 visibility.js 的 `HULL_INSET_HU` 同一个理由、同一个量级。
 */
export const XRAY_BLOCK_EPS_HU = 1.0;

/**
 * 画出来的小人有多高（HU）。
 *
 * ⚠️ 和 `hullHeightFor(crouching)` **故意不同**：那个是物理真相（72 / 54 阶跃），
 * 这个是屏幕上那具身体（跟着**连续** crouch 压下去，蹲到底正好 54）。采样点要贴
 * 着**看得见的东西**，所以用这个 —— 和 visuals.js 里 `bodyRatio` 是同一个式子。
 */
export function drawnHeightFor(state) {
  const c = Math.min(1, Math.max(0, state.crouch || 0));
  return HULL_STAND_HEIGHT - CROUCH_EYE_DROP * c;
}

/** `performance.now()`，Node 里退化成 `Date.now()`（测试直接在 node 里跑）。 */
function now() {
  return (typeof performance !== 'undefined' ? performance : Date).now();
}

/**
 * 建一个可复用的判定器（`classify()` 只在第一次分配结果数组）。
 *
 * ```js
 * const tester = createPlayerXRayTester({ castMapRay: makeMapCaster(getMapBVH(), SCALE) });
 * tester.setOccluders(smokeOccludersAt(currentRound.smoke, tick));
 * const { flags, stats } = tester.classify(camGame, getAllPlayerStates(idx));
 * flags[i]   // 1 = 这个选手被挡住，只能透视看到
 * ```
 *
 * @param {object} [opts]
 * @param {(ox,oy,oz,dx,dy,dz,maxDist)=>number} [opts.castMapRay]
 *        地图遮挡（game 空间）。不传 = 墙不算遮挡，所有人都算直接看得见
 *        —— BVH 是懒建的，建好之前就是这个状态，正好也是对的：地图还没加载
 *        就没有墙可挡。
 * @param {boolean} [opts.smoke=true]  烟雾算不算遮挡（和 visibility.js 一致：
 *        烟里看不见人）
 * @param {number} [opts.blockEpsHu]   命中容差，默认 `XRAY_BLOCK_EPS_HU`
 * @param {Uint8Array} [opts.flags]    复用的结果数组（不传就自己建）
 */
export function createPlayerXRayTester(opts = {}) {
  const castMapRay = typeof opts.castMapRay === 'function' ? opts.castMapRay : null;
  const useSmoke = opts.smoke !== false;
  const eps = opts.blockEpsHu ?? XRAY_BLOCK_EPS_HU;

  let occluders = [];
  let flags = opts.flags instanceof Uint8Array ? opts.flags : new Uint8Array(0);
  let lastStats = null;

  return {
    /** 当前 tick 的烟雾遮挡体（`smokeOccludersAt()` 的返回值）。每帧灌一次。 */
    setOccluders(list) { occluders = list || []; },
    get occluderCount() { return occluders.length; },
    /** 地图 BVH 到位了吗 —— 没到位时墙不算遮挡，谁都不会被判成透视。 */
    get mapReady() { return !!castMapRay; },
    /** 最近一次 classify() 的统计（面板 / 测试用）。 */
    get stats() { return lastStats; },

    /**
     * 逐个选手判一次。
     *
     * @param {{x:number,y:number,z:number}} cam 摄像机位置，**game 空间**
     *        （Hammer 单位，和 playerStates 同一套坐标）
     * @param {Array} states 全部玩家状态（`getAllPlayerStates()` 的返回值）
     * @returns {{flags:Uint8Array, stats:object}} `flags[i]`：0 = 直接看得见，
     *          1 = 被挡住（只能透视看到）
     */
    classify(cam, states) {
      const t0 = now();
      const n = states ? states.length : 0;
      if (flags.length < n) flags = new Uint8Array(n);

      const stats = {
        n, players: 0, rays: 0, blocked: 0, mapHits: 0, smokeHits: 0,
        mapReady: !!castMapRay, ms: 0,
      };
      // 烟雾那一层的早退：没烟的时候连函数都不进
      const hasSmoke = useSmoke && occluders.length > 0;

      for (let i = 0; i < n; i++) {
        const s = states[i];
        // 死人 / 空位 = 直接看得见（本来就不画）
        if (!s || !s.alive) { flags[i] = 0; continue; }
        stats.players++;

        const h = drawnHeightFor(s);
        let clear = false;
        for (let k = 0; k < XRAY_SAMPLE_COUNT; k++) {
          const pz = s.z + h * XRAY_SAMPLE_FRACTIONS[k];
          let dx = s.x - cam.x, dy = s.y - cam.y, dz = pz - cam.z;
          const dist = Math.hypot(dx, dy, dz);
          // 摄像机贴在采样点上（第一人称看自己、或者贴脸看）：算直接看得见。
          // 不做这个早退的话，方向向量会退化成 0/0。
          if (dist <= eps) { clear = true; break; }
          dx /= dist; dy /= dist; dz /= dist;

          stats.rays++;
          // 把 maxDist 收掉 eps：返回值非负 = 在"身体之前"打到了东西（见文件头）。
          // raycastBVH 用的是严格 `hit < best`，所以正好落在 dist-eps 上的命中不算。
          const maxD = dist - eps;
          let blocked = false;
          if (maxD > 0) {
            if (castMapRay && castMapRay(cam.x, cam.y, cam.z, dx, dy, dz, maxD) >= 0) {
              blocked = true; stats.mapHits++;
            } else if (hasSmoke
              && smokeRayBlocked(occluders, cam.x, cam.y, cam.z, dx, dy, dz, maxD)) {
              blocked = true; stats.smokeHits++;
            }
          }
          // 任何一个采样点通就收工：剩下的点不影响结论，省射线
          if (!blocked) { clear = true; break; }
        }

        flags[i] = clear ? 0 : 1;
        if (!clear) stats.blocked++;
      }

      stats.ms = now() - t0;
      lastStats = stats;
      return { flags, stats };
    },
  };
}

/**
 * 一次性的便利版本（测试 / 脚本用，会分配）。渲染热路径请用
 * `createPlayerXRayTester()`。
 *
 * @returns {Uint8Array} `[i] = 1` 表示第 i 个人只能透视看到
 */
export function classifyPlayerXRay(cam, states, opts = {}) {
  const tester = createPlayerXRayTester(opts);
  tester.setOccluders(opts.occluders || []);
  return tester.classify(cam, states).flags;
}
