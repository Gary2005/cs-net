/**
 * scene.js — Three.js scene, camera, renderer, lighting, multi-mode camera.
 *
 * Coordinate system (game → Three.js):
 *   Game (x, y, z) with z=up
 *   → Three.js (Y-up): (y * 0.0254, z * 0.0254, x * 0.0254)
 *
 *   SCALE = 0.0254  (inches per game unit)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { eyeHeightFor, playerName } from './replay-core.js';
import { applyMinimapMapStyle, restoreMapStyle } from './map-loader.js';
import { currentMinimapRadar, radarRect } from './minimap-radar.js';

export const SCALE = 0.0254;

/** Convert game coordinates to Three.js world space */
export function gameToThree(gx, gy, gz) {
  return {
    x:  gy * SCALE,
    y:  gz * SCALE,
    z:  gx * SCALE,
  };
}

/** Convert game vector [x,y,z] to Three.js Vector3 */
export function gameVecToThree([gx, gy, gz]) {
  return new THREE.Vector3(gy * SCALE, gz * SCALE, gx * SCALE);
}

// ── Scene globals ──────────────────────────────────────────────────────────

export let scene;
export let camera;
export let renderer;
export let controls;       // OrbitControls instance (only active in 'orbit' mode)
export let mapGroup;
export let raycaster;
export let mouse;

// Minimap
export let minimapRenderer = null;
export let minimapCamera = null;

/** 小地图上那一层的 2D context（玩家箭头 / 炸弹），见 `drawMinimapMarkers` */
let minimapOverlay = null;

// ── Camera mode system ─────────────────────────────────────────────────────

/** @type {'orbit'|'fly'|'first'|'third'} */
export let cameraMode = 'orbit';

/** @type {number} player index for first/third person view (-1 = none) */
export let focusedPlayerIdx = -1;

/** @type {THREE.Vector3} third-person camera offset behind player */
const thirdPersonOffset = new THREE.Vector3(0, 2.5, 6);

// Fly-mode state
const keys = {};
let flyYaw = 0;
let flyPitch = 0;
let pointerLocked = false;

// WASD 手动移动机位（orbit 模式）。单位是世界坐标的米 / 秒 —— 和 fly 模式同一
// 套速度，Shift 加速。
const PAN_SPEED = 15;
const PAN_SPEED_FAST = 40;

/**
 * 「用户手动接管镜头」的通知。main.js 注册它来**取消选中玩家并停止相机跟随** ——
 * 手动推机位和"跟着某个人"是两件事，同时成立会互相拽。
 *
 * 为什么用回调而不是在 main.js 里再听一次 keydown：谁来听键盘不重要，重要的是
 * **镜头真的被键盘推动了**。焦点在 <select> 里时（主循环/倍速下拉）我们不该动
 * 镜头，回调也就不会被调用，选中状态自然保留。
 */
let manualMoveHandler = null;
export function setManualMoveHandler(fn) { manualMoveHandler = fn || null; }

/** 有没有按着移动键（fly 模式用；orbit 的平移自己算方向）。 */
function anyMoveKeyDown() {
  return !!(keys['KeyW'] || keys['KeyA'] || keys['KeyS'] || keys['KeyD']
    || keys['KeyQ'] || keys['KeyE']);
}

const _panDir = new THREE.Vector3();   // 前（水平）
const _panRight = new THREE.Vector3(); // 右
const _panMove = new THREE.Vector3();  // 本帧位移

/**
 * orbit 模式下的 WASD 手动移动：W/S 沿**水平**视角方向前后、A/D 左右、Q/E 升降，
 * Shift 加速。
 *
 * 位移**同时**加到 `camera.position` 和 `controls.target` 上 —— 也就是说只搬机位，
 * 视角（朝向 + 俯仰 + 距离）一点不动，和右键平移、和相机跟随用的是同一个动作。
 * 只搬相机不搬 target 的话，镜头会变成绕着 target 甩，那和"移动机位"不是一回事。
 *
 * @returns {THREE.Vector3|null} 本帧位移；没有按键时是 null
 */
function keyboardPan(dt) {
  const fwd = (keys['KeyW'] ? 1 : 0) - (keys['KeyS'] ? 1 : 0);
  const str = (keys['KeyD'] ? 1 : 0) - (keys['KeyA'] ? 1 : 0);
  const vert = (keys['KeyE'] ? 1 : 0) - (keys['KeyQ'] ? 1 : 0);
  if (!fwd && !str && !vert) return null;

  const speed = (keys['ShiftLeft'] || keys['ShiftRight']) ? PAN_SPEED_FAST : PAN_SPEED;
  const step = speed * dt;

  camera.getWorldDirection(_panDir);
  _panDir.y = 0;                                  // 水平：否则一低头就会钻到地下
  if (_panDir.lengthSq() < 1e-6) _panDir.set(0, 0, 1);
  _panDir.normalize();
  _panRight.crossVectors(_panDir, camera.up).normalize();

  _panMove.set(0, 0, 0);
  _panMove.addScaledVector(_panDir, fwd * step);
  _panMove.addScaledVector(_panRight, str * step);
  _panMove.y += vert * step;
  return _panMove;
}

// Smooth camera fly-to animation (orbit mode)
let camAnim = null; // { t, dur, fromPos, toPos, fromTarget, toTarget }

export function setCameraMode(mode) {
  cameraMode = mode;
  // 跟随只有 orbit 有（first / third 自己跟着 focusedPlayerIdx）。切走就停：
  // 留着的话基准点会一直不更新，切回 orbit 的第一帧会把这一整段位移一次性
  // 搬到镜头上 —— 那是一次横跨地图的瞬移。
  if (mode !== 'orbit') {
    setFollowPlayer(-1);
    // 没跑完的飞行动画也要一起取消：它只在 orbit 分支里推进，如果留着，
    // 从第一人称切回 orbit 的第一帧会突然接着飞向一个几秒前的旧目标。
    camAnim = null;
  }
  if (mode === 'orbit') {
    controls.enabled = true;
    document.exitPointerLock?.();
    pointerLocked = false;
  } else {
    controls.enabled = false;
    if (mode === 'fly') {
      // Init fly rotation from current camera
      const dir = new THREE.Vector3();
      camera.getWorldDirection(dir);
      flyYaw = Math.atan2(dir.x, dir.z);
      flyPitch = Math.asin(Math.max(-1, Math.min(1, dir.y)));
    }
  }
}

export function setFocusedPlayer(idx) {
  focusedPlayerIdx = idx;
  if (idx >= 0 && (cameraMode === 'first' || cameraMode === 'third')) {
    controls.enabled = false;
  }
}

// ── 相机跟随（在左侧卡片 / 3D 里选中一个玩家之后，镜头跟着他走）─────────────
//
// 做法是「把玩家这一帧的位移原样搬到摄像机上」：`camera.position` 和
// `controls.target` 同时加上同一个 delta。三个直接结果：
//
//   * 相对视角完全不变 —— 用户拖 OrbitControls 转出来的角度、滚轮缩放出来的
//     距离都原样保持，所以「跟着人走」和「绕着他转 / 拉近拉远」可以同时用，
//     不会互相打架（每次拖拽只是换了这个固定的偏移量）；
//   * 不引入任何平滑 / 弹簧：位置本来就在采样点之间线性插值（见 replay-core.js
//     的 getPlayerState），所以跟着走是逐帧连续的，既不滞后也不抖；
//   * 复活 / 换回合这种瞬移会把镜头一起搬过去（偏移量不变），人不会跑出画面。
//
// 只在 orbit 模式下生效：first / third 有自己的跟法（见 renderFrame）。
let followIdx = -1;
const followLastPos = new THREE.Vector3();
let followHasLast = false;

/**
 * 让摄像机跟着某个玩家（传 -1 / null 取消跟随）。
 *
 * `worldPos` 是「基准确认点」：给了就用它当上一帧的位置，没给就在下一帧现取。
 * 必须能显式给，是因为**选中玩家那一刻镜头正飞向他点击时的位置**，而下一帧
 * 玩家又往前走了一格 —— 基准点要是从"下一帧"取，这一格的位移就丢了，镜头会
 * 永远差这一个帧的量（实测 dt=1/30 s、250 HU/s 时是 0.21 m，一直留着）。
 */
export function setFollowPlayer(idx, worldPos = null) {
  followIdx = (idx == null || idx < 0) ? -1 : idx;
  if (followIdx >= 0 && worldPos) {
    followLastPos.copy(worldPos);
    followHasLast = true;
  } else {
    followHasLast = false;
  }
}

/** 正在跟随的玩家下标；-1 = 没在跟。 */
export function getFollowPlayer() { return followIdx; }

/** 飞向玩家那 0.7s 的动画还在跑吗（测试和调试用）。 */
export function isCameraAnimating() { return camAnim != null; }

const _followDelta = new THREE.Vector3();

/**
 * 玩家这一帧走了多少（世界坐标位移）。
 *
 * 第一次调用 —— 或者刚换了跟随目标、数据刚换回合（getPlayerState 返 null）时 ——
 * 只记基准点、返回 null：否则会把「从上一个人 / 上一回合跳过来」的那一整段
 * 位移也算进去，镜头会莫名其妙地飞。
 */
function followDelta(getPlayerStateFn) {
  if (followIdx < 0 || !getPlayerStateFn) return null;
  const st = getPlayerStateFn(followIdx);
  if (!st || !isFinite(st.x) || !isFinite(st.y) || !isFinite(st.z)) {
    followHasLast = false;
    return null;
  }
  const p = gameToThree(st.x, st.y, st.z);
  if (!followHasLast) {
    followLastPos.set(p.x, p.y, p.z);
    followHasLast = true;
    return null;
  }
  _followDelta.set(p.x - followLastPos.x, p.y - followLastPos.y, p.z - followLastPos.z);
  followLastPos.set(p.x, p.y, p.z);
  return _followDelta.lengthSq() > 0 ? _followDelta : null;
}

const clock = new THREE.Clock();
let fpsFrames = 0;
let fpsTime = 0;
let currentFPS = 0;

// ── Initialization ──────────────────────────────────────────────────────────

export function initScene(canvas) {
  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;   // 更清晰的阴影（PCFSoft 边缘发糊）
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;

  // Scene — 清新明亮风格
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xbcd6ee);
  scene.fog = new THREE.Fog(0xbcd6ee, 120, 480);

  // Camera（FOV 稍大，容纳扁平地图）
  camera = new THREE.PerspectiveCamera(68, 2, 0.1, 600);
  camera.position.set(30, 25, 30);
  camera.lookAt(0, 0, 0);

  // Orbit Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.minDistance = 1;
  controls.maxDistance = 180;
  controls.maxPolarAngle = Math.PI * 0.52;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.PAN,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.update();

  // ── 明亮但克制的光照（避免把材质冲白）──────────────────────────
  // 环境光：柔和白光
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));

  // 半球光：天空淡蓝 → 地面浅灰
  scene.add(new THREE.HemisphereLight(0xcfdff2, 0xf2f5f9, 0.65));

  // 主光：暖白阳光
  const sun = new THREE.DirectionalLight(0xfff2dd, 0.85);
  sun.position.set(50, 80, 30);
  sun.castShadow = true;
  sun.shadow.mapSize.width = 4096;
  sun.shadow.mapSize.height = 4096;
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 400;
  sun.shadow.camera.left = -80; sun.shadow.camera.right = 80;
  sun.shadow.camera.top = 80; sun.shadow.camera.bottom = -80;
  sun.shadow.bias = -0.0001;
  scene.add(sun);

  // 补充光：冷色方向光，让模型轮廓清晰
  const fill = new THREE.DirectionalLight(0xe8f1fb, 0.3);
  fill.position.set(-30, 20, -20);
  scene.add(fill);

  // 底部补光：让地图下侧不暗
  const rim = new THREE.DirectionalLight(0xffffff, 0.2);
  rim.position.set(0, -20, 0);
  scene.add(rim);

  // ── 浅色网格地面（半透明，仅视觉辅助）──────────────────────────
  // 小地图铺雷达贴图时要把它藏掉：雷达图四角是透明的，网格会从底下透出来，
  // 在雷达底图上盖一层没意义的格线。主视图照常显示。
  const grid = new THREE.GridHelper(180, 72, 0x9db8d8, 0xbfd0e4);
  grid.material.transparent = true;
  grid.material.opacity = 0.3;
  grid.position.y = -2.2;
  grid.name = 'scene-grid';
  scene.add(grid);


  // Map root
  mapGroup = new THREE.Group();
  mapGroup.name = 'map-root';
  scene.add(mapGroup);

  // Raycaster
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  // ── Input bindings ────────────────────────────────────────────────────
  // 焦点在表单控件里时不记移动键：Chrome 的下拉框本身就是"打字跳选项"，
  // 按 W 会一边换回合一边把镜头推走。keyup 照记不误 —— 清键永远是安全的。
  const isFormField = (el) =>
    !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA'
      || el.tagName === 'SELECT' || el.isContentEditable);
  window.addEventListener('keydown', (e) => {
    if (isFormField(e.target)) return;
    keys[e.code] = true;
  });
  window.addEventListener('keyup',   (e) => { keys[e.code] = false; });

  // Pointer lock for fly mode
  renderer.domElement.addEventListener('click', () => {
    if (cameraMode === 'fly' && !pointerLocked) {
      renderer.domElement.requestPointerLock?.();
    }
  });
  document.addEventListener('pointerlockchange', () => {
    pointerLocked = document.pointerLockElement === renderer.domElement;
  });
  document.addEventListener('mousemove', (e) => {
    if (cameraMode !== 'fly' || !pointerLocked) return;
    flyYaw   -= e.movementX * 0.002;
    flyPitch -= e.movementY * 0.002;
    flyPitch  = Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, flyPitch));
  });

  // ── Minimap ──────────────────────────────────────────────────────────
  const mmCanvas = document.getElementById('minimap-canvas');
  if (mmCanvas) {
    const mmSize = 256; // fixed size for crisp rendering
    mmCanvas.width = mmSize;
    mmCanvas.height = mmSize;
    minimapRenderer = new THREE.WebGLRenderer({ canvas: mmCanvas, antialias: true, alpha: true });
    minimapRenderer.setSize(mmSize, mmSize, false);
    minimapCamera = new THREE.OrthographicCamera(-30, 30, 30, -30, 0.1, 200);
    // 相机基：右 = +Z（世界 x），上 = +X（世界 y）。
    //
    // 这是**游戏内雷达的朝向**：overview 里的 u 轴（图像列）长在世界 x 上，
    // v 轴（图像行，向下）长在世界 y 上。three 里 x=game y、z=game x，所以
    // 屏幕上「右」是 +z、「上」是 +x —— 取景框的 left/right 用 z，top/bottom 用 x。
    //
    // 之前这里写的是 up=(0,0,-1)（右=+x、上=-z），那是把雷达整体转了 90°，
    // 用自己的灰阶底图时看不出来，换成带 A/B 点和房间名的官方雷达图后就会歪。
    //
    // 顺序要紧：`up` 必须在 `lookAt` **之前**设好。`lookAt` 当场用 `this.up`
    // 算四元数，之后再改 `up` 不会重算 —— 原来那行 up.set 写在 lookAt 后面，
    // 其实一直没生效（俯视时 up 与视线平行，three 自己挑了个方向兜底）。
    minimapCamera.position.set(0, 100, 0);
    minimapCamera.up.set(1, 0, 0);
    minimapCamera.lookAt(0, 0, 0);

    // 玩家/炸弹标记画在这块 2D 画布上（`index.html` 里一直就有，之前没用）。
    // 不用 3D 网格的理由见 `drawMinimapMarkers`。
    const mmOverlay = document.getElementById('minimap-overlay');
    if (mmOverlay) {
      mmOverlay.width = mmSize;
      mmOverlay.height = mmSize;
      minimapOverlay = mmOverlay.getContext('2d');
    }
  }

  // Resize — 用 ResizeObserver 监听 viewport 尺寸变化
  // （初始时 app 隐藏 → clientWidth=0 会跳过；切到主界面 viewport 可见时自动修正分辨率）
  const vpEl = document.getElementById('viewport');
  if (vpEl) {
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(() => { onResize(); });
      ro.observe(vpEl);
    } else {
      window.addEventListener('resize', onResize);
    }
  }
  onResize();

  return { scene, camera, renderer, controls, clock };
}

function onResize() {
  const vp = document.getElementById('viewport');
  if (!vp) return;
  const w = vp.clientWidth, h = vp.clientHeight;
  if (w === 0 || h === 0) return;
  // updateStyle=false：只改渲染缓冲，不改 CSS（CSS 由 flex 布局控制）
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}

// ── Per-frame update ────────────────────────────────────────────────────────

/**
 * Call each frame. Handles camera mode logic, updates controls, renders.
 * @param {number} dt - real delta-time in seconds
 * @param {Function|null} getPlayerStateFn - (playerIdx) => state or null
 * @param {{skipRender?: boolean}} [opts] - 只跑相机逻辑、不画（测试用固定 dt
 *   驱动相机时，一次绘制要几百毫秒，量出来的东西和渲染无关）
 * @returns {number} current FPS
 */
export function renderFrame(dt, getPlayerStateFn, opts = {}) {
  // ── Camera mode update ─────────────────────────────────────────────────
  switch (cameraMode) {
    case 'orbit': {
      // 相机跟随：把玩家这一帧的位移搬到摄像机上。飞行途中也要跟 —— 见下面。
      const shift = followDelta(getPlayerStateFn);
      // WASD 手动推机位。用户一按，正在跑的飞行动画就让位（手动输入优先）。
      const pan = keyboardPan(dt);
      if (pan && camAnim) camAnim = null;
      // 平滑飞向目标（flyCameraTo 触发）：位置与视角中心同步插值
      if (camAnim) {
        // 飞行途中跟随时，from / to 两端一起平移 = 整条轨迹平移，缓动本身不变
        // （lerp(a+d, b+d, k) === lerp(a, b, k) + d）。这样到站时镜头正好停在
        // 玩家**此刻**的位置，而不是点击那一刻的快照，落点不会差出一截。
        if (shift) {
          camAnim.fromPos.add(shift);
          camAnim.toPos.add(shift);
          camAnim.fromTarget.add(shift);
          camAnim.toTarget.add(shift);
        }
        camAnim.t += dt / camAnim.dur;
        const k = camAnim.t >= 1 ? 1 : camAnim.t * camAnim.t * (3 - 2 * camAnim.t); // easeInOutCubic
        camera.position.lerpVectors(camAnim.fromPos, camAnim.toPos, k);
        controls.target.lerpVectors(camAnim.fromTarget, camAnim.toTarget, k);
        if (camAnim.t >= 1) camAnim = null;
      } else {
        // 平移跟随：相机和视角中心同时搬，相对视角（用户拖出来的那个角度）不变
        if (shift) {
          camera.position.add(shift);
          controls.target.add(shift);
        }
        // WASD：同样只搬机位，视角不动
        if (pan) {
          camera.position.add(pan);
          controls.target.add(pan);
        }
      }
      // 真的被键盘推了 → 通知 main.js 取消选中 / 停止跟随（见 setManualMoveHandler）
      if (pan && manualMoveHandler) manualMoveHandler();
      controls.update();
      break;
    }

    case 'fly': {
      const speed = keys['ShiftLeft'] ? 40 : 15;
      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);
      const right = new THREE.Vector3();
      right.crossVectors(forward, camera.up).normalize();

      if (keys['KeyW']) camera.position.addScaledVector(forward, speed * dt);
      if (keys['KeyS']) camera.position.addScaledVector(forward, -speed * dt);
      if (keys['KeyA']) camera.position.addScaledVector(right, -speed * dt);
      if (keys['KeyD']) camera.position.addScaledVector(right, speed * dt);
      if (keys['KeyQ']) camera.position.y -= speed * dt;
      if (keys['KeyE']) camera.position.y += speed * dt;

      // Apply fly rotation
      const qx = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), flyPitch);
      const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), flyYaw);
      camera.quaternion.copy(qy).multiply(qx);
      // fly 模式也是拿 WASD 推机位 —— 同一个"手动接管"语义，一样要退出跟随
      if (anyMoveKeyDown() && manualMoveHandler) manualMoveHandler();
      break;
    }

    case 'first':
    case 'third': {
      if (focusedPlayerIdx >= 0 && getPlayerStateFn) {
        const state = getPlayerStateFn(focusedPlayerIdx);
        if (state) {
          const pos = gameToThree(state.x, state.y, state.z);
          // 眼高跟着下蹲走：以前硬编码 64 HU，站立时无所谓，
          // 蹲下时第一人称视角会高出整整 18 HU（半个头）。
          const eyeH = state.eye_height != null ? state.eye_height : eyeHeightFor(state.crouch);
          const eyeY = pos.y + eyeH * SCALE;
          const eyePos = new THREE.Vector3(pos.x, eyeY, pos.z);

          // CS2: yaw=0→+X(game)→+Z(Three), pitch<0→down
          // Ry(yaw)·(0,0,1) = (sin(yaw),0,cos(yaw)) in Three = (cos,yaw, sin,yaw) in game
          const yawRad = (state.yaw || 0) * Math.PI / 180;
          const pitchRad = (state.pitch || 0) * Math.PI / 180;
          const lookDir = new THREE.Vector3(0, 0, 1);
          lookDir.applyAxisAngle(new THREE.Vector3(1, 0, 0), pitchRad);
          lookDir.applyAxisAngle(new THREE.Vector3(0, 1, 0), yawRad);

          if (cameraMode === 'first') {
            camera.position.copy(eyePos);
            camera.lookAt(eyePos.clone().add(lookDir));
          } else {
            // Third person: behind and above player
            const behind = lookDir.clone().multiplyScalar(-thirdPersonOffset.z);
            behind.y += thirdPersonOffset.y;
            const camPos = eyePos.clone().add(behind);
            camera.position.lerp(camPos, 0.15);
            camera.lookAt(eyePos);
          }
        }
      }
      break;
    }
  }

  if (!opts.skipRender) renderer.render(scene, camera);

  // FPS counter
  fpsFrames++;
  fpsTime += clock.getDelta();
  if (fpsTime >= 1.0) {
    currentFPS = Math.round(fpsFrames / fpsTime);
    fpsFrames = 0;
    fpsTime = 0;
  }
  return currentFPS;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

export function resetCamera(preset = 'free') {
  camAnim = null;   // 取消进行中的飞向动画
  setFollowPlayer(-1);  // 「重置视角」= 交出镜头控制权，跟随一起停
  switch (preset) {
    case 'top':
      camera.position.set(0, 60, 0);
      controls.target.set(0, 0, 0);
      break;
    case 'free':
    default:
      camera.position.set(25, 20, 25);
      controls.target.set(0, 0, 0);
      break;
  }
  if (cameraMode === 'orbit') controls.update();
}

/**
 * 平滑飞向某个世界坐标（仅 orbit 模式生效）：
 * 视角中心移到目标点，摄像机按当前视角方向拉近到其身边。
 * 动画在 renderFrame 的 orbit 分支里逐帧推进。
 * @param {THREE.Vector3} worldPos — 目标世界坐标（如玩家位置）
 * @param {number} distance — 摄像机到目标的水平距离（默认 11）
 * @param {number} height   — 摄像机相对目标的高度（默认 7）
 */
export function flyCameraTo(worldPos, distance = 11, height = 7) {
  if (cameraMode !== 'orbit' || !controls) return;
  const target = worldPos.clone();
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  dir.y = 0;
  if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
  dir.normalize();
  const camPos = target.clone().add(dir.clone().multiplyScalar(-distance));
  camPos.y = Math.max(target.y + 1, target.y + height);
  camAnim = {
    t: 0,
    dur: 0.7,
    fromPos: camera.position.clone(),
    toPos: camPos,
    fromTarget: controls.target.clone(),
    toTarget: target,
  };
}

// Smooth auto-zoom state for minimap
let mmTargetCenter = new THREE.Vector2(0, 0);
let mmTargetSpan = 60;
let mmCurCenter = new THREE.Vector2(0, 0);
let mmCurSpan = 60;

/**
 * 小地图上的玩家箭头 / 炸弹标记，画在 `#minimap-overlay` 这块 2D 画布上。
 *
 * 这几个标记原来是 three 的网格，跟着场景一起渲染；问题一堆：
 *
 *  * 形状只能是多边形，没有圆角、没有真正的描边，十几个像素的箭头放大看就是一坨锯齿；
 *  * 雷达底图是 `transparent: true`，渲染器会把它排到**所有不透明物体之后**，标记要
 *    想不被盖住就得跟着进透明队列，靠 renderOrder 抢顺序（踩过：箭头只在底图透明的
 *    地方露一点，看起来「若隐若现」，但轨迹线看得见）；
 *  * 大小跟着取景缩放走 —— 全图视角下箭头只有 8px，拉近了又变成 30px 的大三角。
 *
 * 挪到 2D 画布上三件事一起解决：路径 + 圆角连接 + 阴影随便调，层级就是 DOM 层级
 * （`#minimap-overlay` 在 `#minimap-canvas` 后面），大小直接按像素写死。
 * 投影用的还是同一个正交相机（屏幕右 = +z、上 = +x），所以位置和底图仍然严丝合缝。
 */

/**
 * 角落小地图的箭头总长 / 炸弹直径（CSS 像素，按 200px 的面板算）。真正落笔画的
 * 时候还要乘 `ui`（1 CSS px = 多少后备像素）和 `grow`（面板放大后的补偿），见
 * `drawMinimapMarkers`。
 *
 * 炸弹刻意画得**比箭头还小**：它标的是一个点，之前 15px 的红饼会把落点附近的
 * 玩家和掩体整个盖住，反而看不出炸弹贴在哪。
 */
const MM_ARROW_CSS = 14;
const MM_BOMB_CSS = 7;

/**
 * 2D 回放（点开放大）那一档的箭头 / 炸弹：**固定尺寸，不跟着面板放大**。
 *
 * 它的取景是整张地图（见 `setMinimapMode2D`），面板又有 500px 上下 —— 再按
 * √面积 长就是 20px 往上的大三角，16 个人的箭头连成一片，反而看不出谁是谁。
 * 所以这一档比角落小地图还小一号；炸弹跟着缩，仍旧比箭头小得多。
 */
const MM_ARROW_2D_CSS = 10;
const MM_BOMB_2D_CSS = 5;

/** 描边色 / 阴影：底图深浅都有，一层深色描边 + 一层淡阴影保证两种底子上都压得住 */
const MM_OUTLINE = 'rgba(8, 12, 20, 0.9)';
const MM_SHADOW = 'rgba(0, 0, 0, 0.55)';

/**
 * 2D 回放里写在箭头旁边的名字。
 *
 * 字号按 CSS 像素写死（落笔前乘 `ui`）：面板多大，字看起来都一样大。颜色取队伍的
 * 箭头色的浅一档 + 深色描边 —— 雷达底图从浅沙色到深灰都有，纯白字在浅底上会糊掉。
 */
const MM_NAME_CSS = 11;
const MM_NAME_FONT = '"Segoe UI", "PingFang SC", system-ui, sans-serif';
const MM_NAME_OUTLINE = 'rgba(6, 10, 16, 0.92)';
const MM_NAME_CT = '#cfe4f7';
const MM_NAME_T = '#ffd7a8';

/** 小地图当前是不是「2D 回放」那一档（固定取景 + 小一号标记），见 `setMinimapMode2D` */
let minimapMode2D = false;
/** 下一帧直接落到目标取景、不做缓动（切档时用，省得地图自己滑一段） */
let mmSnapNext = true;

/**
 * 2D 回放里每个箭头 + 名字的点击热区（CSS 像素，相对画布左上角）。
 * 每次重画都重建 —— 玩家一直在动，热区必须跟着走。
 */
const mmHits = [];

/**
 * 把取景切到 / 切回「2D 回放」档。
 *
 * 和角落小地图只差两件事：取景**锁死成整张地图**（不跟着存活玩家缩放、不跟着他们
 * 平移）、标记用小一号的固定尺寸。画布 / 相机 / 雷达底图还是同一份 —— 点开以后
 * 位置关系不变，只是看得更清楚，而且不再随每一帧抖。
 *
 * @param {boolean} on
 */
export function setMinimapMode2D(on) {
  on = !!on;
  if (on === minimapMode2D) return;
  minimapMode2D = on;
  mmSnapNext = true;
}

/** 面板变大时标记的放大系数：按 √面积 长，封顶 2 倍（2D 回放那一档不用它） */
function glyphGrow(css) {
  return Math.max(1, Math.min(2, Math.sqrt(css / 200)));
}

/**
 * 当前该用多大的箭头 / 炸弹（CSS 像素）。
 * @param {number} css 面板的显示边长
 * @returns {{arrow: number, bomb: number, grow: number}}
 */
function glyphSizes(css) {
  if (minimapMode2D) return { arrow: MM_ARROW_2D_CSS, bomb: MM_BOMB_2D_CSS, grow: 1 };
  const grow = glyphGrow(css);
  return { arrow: MM_ARROW_CSS * grow, bomb: MM_BOMB_CSS * grow, grow };
}

/**
 * @param {Array<object>} playerStates
 * @param {Array<string>} teams
 * @param {number[]|null} bombPos
 * @param {number} w - 画布后备宽（像素）
 * @param {number} h - 画布后备高（像素）
 * @param {number} css - 画布的显示边长（CSS 像素）
 */
function drawMinimapMarkers(playerStates, teams, bombPos, w, h, css) {
  const ctx = minimapOverlay;
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);

  const cam = minimapCamera;
  const spanZ = cam ? cam.right - cam.left : 0;
  const spanX = cam ? cam.top - cam.bottom : 0;
  if (!(spanZ > 0) || !(spanX > 0)) return;

  const toScreen = (p) => [
    (p.z - cam.left) / spanZ * w,
    (cam.top - p.x) / spanX * h,
  ];
  const visible = ([x, y]) => x > -32 && y > -32 && x < w + 32 && y < h + 32;

  // 尺寸都写成 CSS 像素再换算成后备像素：面板放大时后备分辨率跟着翻，标记不该
  // 跟着翻（那是 `grow` 单独管的事，而且只长一半），否则同一块地图上箭头会比小
  // 地图里大一个数量级。
  const ui = w / Math.max(css, 1);
  const sizes = glyphSizes(css);
  const arrow = sizes.arrow * ui;
  const bomb = sizes.bomb * ui;

  // 箭头本体：尖端 (L, 0)、两翼 (-B, ±W)、尾部凹口 (-N, 0)。按箭头总长等比缩放，
  // 凹口只有 1px 上下：读起来还是箭头，但不会在十几像素的尺寸上把尾巴劈成一撇「鱼尾」。
  const L = arrow * 0.62;
  const B = arrow * 0.38;
  const W = arrow * 0.32;
  const N = arrow * 0.31;

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineWidth = 1 * ui;
  ctx.strokeStyle = MM_OUTLINE;
  ctx.shadowColor = MM_SHADOW;
  ctx.shadowBlur = 2 * ui;
  ctx.shadowOffsetY = 0.8 * ui;

  mmHits.length = 0;
  const spots = [];                     // 名字那一趟要复用投影结果

  for (let i = 0; i < playerStates.length; i++) {
    const s = playerStates[i];
    if (!s || !s.alive) continue;
    const [sx, sy] = toScreen(gameToThree(s.x, s.y, s.z));
    if (!visible([sx, sy])) continue;
    spots.push({ idx: i, sx, sy });

    ctx.save();
    ctx.translate(sx, sy);
    // 世界朝向 (game x, game y) = (cos yaw, sin yaw)；屏幕右 = +game x、屏幕上 =
    // +game y，所以屏幕方向是 (cos yaw, -sin yaw)，也就是 canvas 旋转 -yaw。
    ctx.rotate(-(s.yaw || 0) * Math.PI / 180);
    ctx.beginPath();
    ctx.moveTo(L, 0);
    ctx.lineTo(-B, W);
    ctx.lineTo(-N, 0);
    ctx.lineTo(-B, -W);
    ctx.closePath();
    ctx.fillStyle = teams[i] === 'CT' ? '#5b9bd5' : '#e8a240';
    ctx.fill();
    ctx.shadowColor = 'transparent';   // 填充已经带过阴影，描边不要再叠一层
    ctx.stroke();
    ctx.restore();
  }

  // ── 名字：只有 2D 回放这一档画 ────────────────────────────────────────
  // 角落那块 200px 的地图上 10 个名字会糊成一片，而且那里本来就只是"人在哪"的
  // 参考；点开的 2D 回放才是"看清楚谁是谁"的地方。
  //
  // 位置要**避让**：一坨人挤在同一个房间时，名字全写右边就会叠成一串
  // （"XANTARESJimpthat"）。所以每个名字按「右 → 左 → 右下/右上 …→ 正上/正下」
  // 依次试，取第一个既不出画布、又不压已放好的名字、也不压任何箭头的位置。
  // 名字单独一趟、画在所有箭头之后：写名字的时候箭头已经全部落笔。
  if (minimapMode2D) {
    ctx.save();
    ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 3 * ui;
    ctx.strokeStyle = MM_NAME_OUTLINE;
    ctx.shadowColor = 'transparent';
    ctx.font = `600 ${MM_NAME_CSS * ui}px ${MM_NAME_FONT}`;
    const halfText = MM_NAME_CSS * ui * 0.62;
    const lineH = MM_NAME_CSS * ui * 1.35;
    const gap = arrow * 0.55 + 2 * ui;
    const pad = 2 * ui;

    const rectOf = (c, tw) => {
      const x0 = c.align === 'left' ? c.x : c.align === 'right' ? c.x - tw : c.x - tw / 2;
      return { x0, x1: x0 + tw, y0: c.y - halfText, y1: c.y + halfText };
    };
    const clash = (a, b) =>
      a.x0 < b.x1 + pad && a.x1 > b.x0 - pad && a.y0 < b.y1 + pad && a.y1 > b.y0 - pad;
    const hitsCircle = (r, cx, cy, cr) => {
      const dx = Math.max(r.x0 - cx, 0, cx - r.x1);
      const dy = Math.max(r.y0 - cy, 0, cy - r.y1);
      return dx * dx + dy * dy < cr * cr;
    };

    const taken = [];
    for (const { idx, sx, sy } of spots) {
      const text = playerName(idx);
      if (!text) continue;
      const tw = ctx.measureText(text).width;
      const cands = [
        { x: sx + gap, y: sy, align: 'left' },
        { x: sx - gap, y: sy, align: 'right' },
        { x: sx + gap, y: sy - lineH, align: 'left' },
        { x: sx + gap, y: sy + lineH, align: 'left' },
        { x: sx - gap, y: sy - lineH, align: 'right' },
        { x: sx - gap, y: sy + lineH, align: 'right' },
        { x: sx, y: sy - arrow * 0.85 - halfText, align: 'center' },
        { x: sx, y: sy + arrow * 0.85 + halfText, align: 'center' },
      ];
      let pick = null;
      for (const c of cands) {
        const r = rectOf(c, tw);
        if (r.x0 < 1 || r.x1 > w - 1 || r.y0 < 1 || r.y1 > h - 1) continue;
        if (taken.some((t) => clash(r, t))) continue;
        if (spots.some((s) => hitsCircle(r, s.sx, s.sy, arrow * 0.55))) continue;
        pick = { c, r };
        break;
      }
      // 实在挤不下（10 个人贴在一起）：退回右边那一档，宁可压一点也别把名字丢了
      if (!pick) pick = { c: cands[0], r: rectOf(cands[0], tw) };
      taken.push(pick.r);

      ctx.textAlign = pick.c.align;
      ctx.strokeText(text, pick.c.x, pick.c.y);
      ctx.fillStyle = teams[idx] === 'CT' ? MM_NAME_CT : MM_NAME_T;
      ctx.fillText(text, pick.c.x, pick.c.y);

      // 热区记在 CSS 像素上：点击是 DOM 事件给的坐标，别在这里再来回换算
      mmHits.push({
        idx,
        x: sx / ui,
        y: sy / ui,
        r: Math.max(sizes.arrow * 0.75, 8),
        nx0: pick.r.x0 / ui - 1, nx1: pick.r.x1 / ui + 1,
        ny0: pick.r.y0 / ui, ny1: pick.r.y1 / ui,
      });
    }
    ctx.restore();
  }

  if (bombPos) {
    const [bx, by] = toScreen(gameToThree(bombPos[0], bombPos[1], bombPos[2]));
    if (visible([bx, by])) {
      const r = bomb / 2 * (1 + 0.08 * Math.sin(performance.now() / 300));
      ctx.beginPath();
      ctx.arc(bx, by, r, 0, Math.PI * 2);
      ctx.fillStyle = '#ff2b2b';
      ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.stroke();
    }
  }

  ctx.restore();
}

/**
 * 2D 回放里某个位置点中的玩家下标（-1 = 没点中）。
 *
 * 两趟：先按箭头找（圆，取最近的一个），再看名字那一条矩形 —— 名字是贴着箭头写的，
 * 点名字和点箭头应该是一回事。
 *
 * @param {number} cssX 相对 `#minimap-overlay` 左上角的 CSS 像素
 * @param {number} cssY
 * @returns {number} 玩家下标，-1 = 空白处
 */
export function minimapPlayerAt(cssX, cssY) {
  if (!minimapMode2D) return -1;
  let best = -1, bestD = Infinity;
  for (const h of mmHits) {
    const dx = cssX - h.x, dy = cssY - h.y;
    const d = dx * dx + dy * dy;
    if (d <= h.r * h.r && d < bestD) { bestD = d; best = h.idx; }
  }
  if (best >= 0) return best;
  for (const h of mmHits) {
    if (cssX >= h.nx0 && cssX <= h.nx1 && cssY >= h.ny0 && cssY <= h.ny1) return h.idx;
  }
  return -1;
}

/** 2D 回放里的点击热区（测试 / 调试用），坐标都是相对画布的 CSS 像素。 */
export function minimapHitDebug() {
  return mmHits.map((h) => ({ ...h, name: playerName(h.idx) }));
}

/**
 * 标记在小地图上的实际显示尺寸（CSS 像素）。
 *
 * 测试拿它核对「炸弹比箭头小」，截图脚本也能用它把量到的像素换回来 —— 这几个
 * 数不该只活在绘制函数里。
 *
 * @returns {{mode: string, css: number, canvas: number, grow: number,
 *            arrow: number, bomb: number}|null}
 */
export function minimapGlyphMetrics() {
  if (!minimapRenderer) return null;
  const el = minimapRenderer.domElement;
  const css = Math.min(el.clientWidth, el.clientHeight);
  if (!css) return null;
  const m = glyphSizes(css);
  return {
    mode: minimapMode2D ? '2d' : 'small',
    css,
    canvas: el.width,
    grow: m.grow,
    arrow: m.arrow,
    bomb: m.bomb,
  };
}

// ── 雷达贴图平面 ────────────────────────────────────────────────────────────
/** 当前地图的雷达平面；换图时重建 */
let radarMesh = null;
let radarMeshMap = null;

/** 雷达平面离地高度：抬一点，避免和地图地面 / 烟雾贴片 z-fighting */
const RADAR_PLANE_Y = 0.02;

/**
 * 把雷达图铺成一块平面，盖住它对应的世界矩形（`radarRect`）。
 *
 * 关键在朝向：图像列 u 沿世界 +x，图像行 v 向下沿世界 -y；three 里
 * x=game y、z=game x，所以「图像右」是 +z、「图像上」是 +x。UV 照这个挂上去，
 * 玩家标记才和图上的房间对得上 —— 注意 v 是反的，原因见下面那段注释。
 *
 * @param {object} radar - `minimap-radar.js` 载入的数据
 * @returns {THREE.Mesh}
 */
function ensureRadarMesh(radar) {
  if (radarMesh && radarMeshMap === radar.map) return radarMesh;
  if (radarMesh) {
    scene.remove(radarMesh);
    radarMesh.geometry.dispose();
    radarMesh.material.dispose();
    radarMesh = null;
    radarMeshMap = null;
  }

  const rect = radarRect(radar);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute([
    rect.xTop,    RADAR_PLANE_Y, rect.zLeft,     // 0 = 图像左上
    rect.xTop,    RADAR_PLANE_Y, rect.zRight,    // 1 = 右上
    rect.xBottom, RADAR_PLANE_Y, rect.zRight,    // 2 = 右下
    rect.xBottom, RADAR_PLANE_Y, rect.zLeft,     // 3 = 左下
  ], 3));
  // UV 的 v 要反过来：`TextureLoader` 载入的贴图默认 `flipY = true`（上传 GPU 时
  // 上下翻一次），所以 **v=1 才是图像的顶行**。角 0/1 在 +x 侧（屏幕上方），正是
  // 图像顶行，因此给 v=1；角 2/3 给 v=0。写反了图就是上下颠倒的 —— 房间、楼梯
  // 全对不上，但 A/B 点和玩家标记还在正确的位置，很容易误判成「转了个角度」。
  geo.setAttribute('uv', new THREE.Float32BufferAttribute([
    0, 1, 1, 1, 1, 0, 0, 0,
  ], 2));
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  geo.computeBoundingSphere();

  const mat = new THREE.MeshBasicMaterial({
    map: radar.texture,
    transparent: true,       // 雷达图四角是透明的
    depthWrite: false,       // 半透明底图不该往深度缓冲里写
    toneMapped: false,       // 主视图开着 tone mapping，雷达图要保持原色
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'minimap-radar';
  mesh.renderOrder = -1;
  mesh.frustumCulled = false;
  scene.add(mesh);

  radarMesh = mesh;
  radarMeshMap = radar.map;
  return mesh;
}

/**
 * nuke / train / vertigo 有上下两层雷达图，和游戏一样按高度切换：
 * 优先看第一人称视角跟随的那名玩家，否则看存活玩家里的多数。
 */
function pickRadarTexture(radar, playerStates, focusIdx) {
  if (!radar.lowerTexture || radar.lowerBelowZ === null) return radar.texture;
  const below = (z) => z < radar.lowerBelowZ;
  if (focusIdx >= 0) {
    const s = playerStates[focusIdx];
    if (s && s.alive && Number.isFinite(s.z)) {
      return below(s.z) ? radar.lowerTexture : radar.texture;
    }
  }
  let lower = 0, total = 0;
  for (const s of playerStates) {
    if (!s || !s.alive || !Number.isFinite(s.z)) continue;
    total++;
    if (below(s.z)) lower++;
  }
  return total > 0 && lower * 2 > total ? radar.lowerTexture : radar.texture;
}

/**
 * 后备分辨率跟着**显示尺寸**走。
 *
 * 角落那块小地图是 200px，256 的后备画布够锐；点开成 2D 回放后同一块画布会有
 * 600px 以上，还用 256 的话底图直接糊成一片。反过来收起时要把分辨率放下来 ——
 * 一张 1024² 的 WebGL 画布每 3 帧重绘一次并不便宜。
 *
 * 每帧量一次 `clientWidth` 就够了：放大 / 缩小窗口、点开 2D 回放都会在这一步
 * 自动跟上，调用方不需要再通知谁。
 *
 * @returns {number} 画布的显示边长（CSS 像素）；面板还没显示出来时是 0
 */
function syncMinimapCanvasSize() {
  const el = minimapRenderer.domElement;
  const css = Math.min(el.clientWidth, el.clientHeight);
  if (!css) return 0;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const target = Math.max(256, Math.min(1024, Math.round(css * dpr)));
  if (el.width !== target) {
    minimapRenderer.setSize(target, target, false);   // false = 别动 CSS
    if (minimapOverlay) {
      minimapOverlay.canvas.width = target;
      minimapOverlay.canvas.height = target;
    }
  }
  return css;
}

/**
 * 整张地图在世界里的范围（three 坐标）：`{xMin, xMax, zMin, zMax}`。
 *
 * 2D 回放要把整张图铺进面板，取景就不能再按存活玩家算。优先用官方雷达图那块
 * 矩形 —— 它就是游戏里雷达能覆盖的范围；没有雷达图（老地图 / 没抽资产）时退回
 * 地图几何的包围盒，**算一次就缓存**（130 MB 的 OBJ 上遍历一遍并不便宜）。
 *
 * @returns {{xMin:number, xMax:number, zMin:number, zMax:number}|null}
 */
let mmExtentCache = null;
let mmExtentKey = '';
function mapExtent() {
  const radar = currentMinimapRadar();
  if (radar) {
    const r = radarRect(radar);
    return { xMin: r.xBottom, xMax: r.xTop, zMin: r.zLeft, zMax: r.zRight };
  }
  if (!mapGroup) return null;
  // 换图后 children 会换一批，用它当缓存键
  const key = mapGroup.children.length
    ? `${mapGroup.children.length}:${mapGroup.children[0].uuid}` : '';
  if (mmExtentCache && key === mmExtentKey) return mmExtentCache;

  mapGroup.updateMatrixWorld(true);
  const box = new THREE.Box3();
  mapGroup.traverse((child) => {
    if (!child.isMesh || !child.geometry) return;
    child.geometry.computeBoundingBox();
    if (!child.geometry.boundingBox) return;
    box.union(child.geometry.boundingBox.clone().applyMatrix4(child.matrixWorld));
  });
  if (box.isEmpty()) return null;
  mmExtentCache = { xMin: box.min.x, xMax: box.max.x, zMin: box.min.z, zMax: box.max.z };
  mmExtentKey = key;
  return mmExtentCache;
}

/** 小地图那一次绘制的前后钩子，见 `registerMinimapPass` */
let minimapPassHook = null;

/**
 * 注册「小地图绘制前后」的钩子：`fn(true)` 在画之前、`fn(false)` 在画完（finally）。
 *
 * 小地图和 3D 场景共用同一份 scene / 材质。想让某个东西**只在小地图里**换个样子，
 * 只能在这一帧的小地图绘制期间临时改材质、画完立刻改回去 —— 现在只有 `visuals.js`
 * 用它把烟压暗，这里不必知道烟是什么。
 *
 * @param {((on: boolean) => void)|null} fn
 */
export function registerMinimapPass(fn) {
  minimapPassHook = fn || null;
}

/**
 * 渲染小地图：底图（雷达图 / OBJ 俯视）+ 2D 叠加层上的玩家箭头和炸弹标记。
 */
export function renderMinimap(playerStates, teams, bombPos) {
  if (!minimapRenderer || !minimapCamera) return;

  const css = syncMinimapCanvasSize();
  if (!css) return;                       // #app 还藏着，没有可量的尺寸

  const w = minimapRenderer.domElement.width;
  const h = minimapRenderer.domElement.height;

  // ── 取景 ──────────────────────────────────────────────────────────────
  if (minimapMode2D) {
    // 2D 回放：整张地图、固定不动。这一档是「看回放」，位置关系要的是稳定参照 ——
    // 跟着存活玩家缩放 / 平移的话，每换一个采样点地图就自己动一下。
    const ext = mapExtent();
    if (ext) {
      // 方块面板：取长边，短边自然留白，整张图一定装得下
      mmTargetSpan = Math.max(ext.zMax - ext.zMin, ext.xMax - ext.xMin) * 1.02;
      mmTargetCenter.set((ext.zMin + ext.zMax) / 2, (ext.xMin + ext.xMax) / 2);
    }
  } else {
    const alivePositions = [];
    for (let i = 0; i < playerStates.length; i++) {
      const s = playerStates[i];
      if (!s || !s.alive) continue;
      const pos = gameToThree(s.x, s.y, s.z);
      alivePositions.push({ x: pos.x, z: pos.z });
    }
    if (bombPos) {
      const bp = gameToThree(bombPos[0], bombPos[1], bombPos[2]);
      alivePositions.push({ x: bp.x, z: bp.z });
    }

    if (alivePositions.length > 0) {
      let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
      for (const p of alivePositions) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.z < minZ) minZ = p.z;
        if (p.z > maxZ) maxZ = p.z;
      }
      const span = Math.max(maxX - minX, maxZ - minZ, 15) * 1.4;
      // 取景中心记成 (z, x)：屏幕右 = +z、上 = +x（见相机基的注释）
      if (cameraMode === 'first' && focusedPlayerIdx >= 0) {
        const fp = playerStates[focusedPlayerIdx];
        if (fp && fp.alive) {
          const fpos = gameToThree(fp.x, fp.y, fp.z);
          mmTargetCenter.set(fpos.z, fpos.x);
        } else {
          mmTargetCenter.set((minZ + maxZ) / 2, (minX + maxX) / 2);
        }
      } else {
        mmTargetCenter.set((minZ + maxZ) / 2, (minX + maxX) / 2);
      }
      mmTargetSpan = span;
    } else {
      mmTargetSpan = 60; mmTargetCenter.set(0, 0);
    }
  }

  if (mmSnapNext) {
    // 刚切档：直接落位，别让地图自己滑一段（切过去的第一帧就是最终取景）
    mmCurSpan = mmTargetSpan;
    mmCurCenter.copy(mmTargetCenter);
    mmSnapNext = false;
  } else {
    mmCurCenter.x += (mmTargetCenter.x - mmCurCenter.x) * 0.08;   // z
    mmCurCenter.y += (mmTargetCenter.y - mmCurCenter.y) * 0.08;   // x
    mmCurSpan += (mmTargetSpan - mmCurSpan) * 0.08;
  }

  const halfSpan = mmCurSpan / 2;
  const hsx = halfSpan * (w / h);
  minimapCamera.left   = mmCurCenter.x - hsx;          // 水平轴 = z
  minimapCamera.right  = mmCurCenter.x + hsx;
  minimapCamera.bottom = mmCurCenter.y - halfSpan;     // 垂直轴 = x
  minimapCamera.top    = mmCurCenter.y + halfSpan;
  minimapCamera.updateProjectionMatrix();

  // 标记的屏幕上大小是固定的（画在 2D 叠加层上，按像素写死），所以这里不需要
  // 再按取景缩放去补偿。

  // ── Render everything in one pass ─────────────────────────────────────
  // 有官方雷达图时：把 OBJ 藏掉、铺一块贴了雷达图的平面，地图不用再切灰阶材质；
  // 没有时退回原来的「OBJ 俯视 + 高程灰阶」，所以老地图/没抽资产的环境照样能用。
  const origBg = scene.background;
  const origFog = scene.fog;
  scene.background = new THREE.Color(0x1a2a3a);
  scene.fog = null;

  const radar = currentMinimapRadar();
  const useRadar = !!(radar && radar.texture);
  const sceneGrid = useRadar ? scene.getObjectByName('scene-grid') : null;
  if (useRadar) {
    const mesh = ensureRadarMesh(radar);
    const tex = pickRadarTexture(radar, playerStates, cameraMode === 'first' ? focusedPlayerIdx : -1);
    if (mesh.material.map !== tex) {
      mesh.material.map = tex;
      mesh.material.needsUpdate = true;
    }
    mesh.visible = true;
    mapGroup.visible = false;
    if (sceneGrid) sceneGrid.visible = false;
  }
  try {
    // 只在这一次绘制期间生效的视觉调整（现在只有「烟压暗」），画完必须还原
    if (minimapPassHook) minimapPassHook(true);
    if (!useRadar) applyMinimapMapStyle();
    minimapRenderer.render(scene, minimapCamera);
  } finally {
    if (minimapPassHook) minimapPassHook(false);
    if (useRadar) {
      mapGroup.visible = true;
      if (radarMesh) radarMesh.visible = false;
      if (sceneGrid) sceneGrid.visible = true;
    } else {
      restoreMapStyle();
    }
  }

  scene.background = origBg;
  scene.fog = origFog;

  // ── 标记画到 2D 叠加层 ───────────────────────────────────────────────
  // 底图渲染完之后再画，读代码时顺序就是「先底图、再标记」；两块画布本来就是分开的
  // （`#minimap-overlay` 在 DOM 里排在 `#minimap-canvas` 后面），所以这里只是叙事顺序。
  drawMinimapMarkers(playerStates, teams, bombPos, w, h, css);
}

export function focusOn(pos) {
  controls.target.copy(pos);
  if (cameraMode === 'orbit') controls.update();
}

/** Get current camera mode label */
export function getCameraModeLabel() {
  switch (cameraMode) {
    case 'orbit': return 'Orbit (Free)';
    case 'fly':   return 'WASD Fly';
    case 'first': return '1st Person';
    case 'third': return '3rd Person';
  }
}
