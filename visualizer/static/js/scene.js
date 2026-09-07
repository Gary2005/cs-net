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

// Smooth camera fly-to animation (orbit mode)
let camAnim = null; // { t, dur, fromPos, toPos, fromTarget, toTarget }

export function setCameraMode(mode) {
  cameraMode = mode;
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
  const grid = new THREE.GridHelper(180, 72, 0x9db8d8, 0xbfd0e4);
  grid.material.transparent = true;
  grid.material.opacity = 0.3;
  grid.position.y = -2.2;
  scene.add(grid);


  // Map root
  mapGroup = new THREE.Group();
  mapGroup.name = 'map-root';
  scene.add(mapGroup);

  // Raycaster
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  // ── Input bindings ────────────────────────────────────────────────────
  window.addEventListener('keydown', (e) => { keys[e.code] = true; });
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
    minimapCamera.position.set(0, 100, 0);
    minimapCamera.lookAt(0, 0, 0);
    minimapCamera.up.set(0, 0, -1); // +X=right, -Z=up (north), no X-flip
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
 * @returns {number} current FPS
 */
export function renderFrame(dt, getPlayerStateFn) {
  // ── Camera mode update ─────────────────────────────────────────────────
  switch (cameraMode) {
    case 'orbit': {
      // 平滑飞向目标（flyCameraTo 触发）：位置与视角中心同步插值
      if (camAnim) {
        camAnim.t += dt / camAnim.dur;
        const k = camAnim.t >= 1 ? 1 : camAnim.t * camAnim.t * (3 - 2 * camAnim.t); // easeInOutCubic
        camera.position.lerpVectors(camAnim.fromPos, camAnim.toPos, k);
        controls.target.lerpVectors(camAnim.fromTarget, camAnim.toTarget, k);
        if (camAnim.t >= 1) camAnim = null;
      }
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
      break;
    }

    case 'first':
    case 'third': {
      if (focusedPlayerIdx >= 0 && getPlayerStateFn) {
        const state = getPlayerStateFn(focusedPlayerIdx);
        if (state) {
          const pos = gameToThree(state.x, state.y, state.z);
          const eyeY = pos.y + 64 * SCALE;
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

  renderer.render(scene, camera);

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
 * Render the minimap with auto-zoom and enhanced player indicators.
 */
// Shared 3D marker geometries for minimap
let mmDotGeo, mmConeGeo, mmBombGeo;
const mmMarkerGroup = new THREE.Group();
mmMarkerGroup.name = 'minimap-markers';
mmMarkerGroup.renderOrder = 999;

function ensureMmGeos() {
  if (!mmDotGeo) {
    mmDotGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.5, 16);
    mmConeGeo = new THREE.ConeGeometry(1.0, 2.5, 8, 1);
    mmBombGeo = new THREE.SphereGeometry(2.0, 16, 12);
  }
}

/**
 * Render minimap with 3D markers for perfect alignment.
 */
export function renderMinimap(playerStates, teams, bombPos) {
  if (!minimapRenderer || !minimapCamera) return;
  ensureMmGeos();

  const w = minimapRenderer.domElement.width;
  const h = minimapRenderer.domElement.height;

  // ── Compute auto-zoom ────────────────────────────────────────────────
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
    if (cameraMode === 'first' && focusedPlayerIdx >= 0) {
      const fp = playerStates[focusedPlayerIdx];
      if (fp && fp.alive) {
        const fpos = gameToThree(fp.x, fp.y, fp.z);
        mmTargetCenter.set(fpos.x, fpos.z);
      } else {
        mmTargetCenter.set((minX + maxX) / 2, (minZ + maxZ) / 2);
      }
    } else {
      mmTargetCenter.set((minX + maxX) / 2, (minZ + maxZ) / 2);
    }
    mmTargetSpan = span;
  } else {
    mmTargetSpan = 60; mmTargetCenter.set(0, 0);
  }

  mmCurCenter.x += (mmTargetCenter.x - mmCurCenter.x) * 0.08;
  mmCurCenter.y += (mmTargetCenter.y - mmCurCenter.y) * 0.08;
  mmCurSpan += (mmTargetSpan - mmCurSpan) * 0.08;

  const halfSpan = mmCurSpan / 2;
  const hsx = halfSpan * (w / h);
  // Camera basis: X = world +X, Y = world -Z
  // → left/right map world X directly → bottom/top use NEGATED world Z
  minimapCamera.left   = mmCurCenter.x - hsx;
  minimapCamera.right  = mmCurCenter.x + hsx;
  minimapCamera.bottom = -(mmCurCenter.y + halfSpan);
  minimapCamera.top    = -(mmCurCenter.y - halfSpan);
  minimapCamera.updateProjectionMatrix();

  // ── Add 3D markers to scene temporarily ─────────────────────────────
  for (let i = 0; i < playerStates.length; i++) {
    const s = playerStates[i];
    if (!s || !s.alive) continue;
    const pos = gameToThree(s.x, s.y, s.z);
    const team = teams[i];
    const color = team === 'CT' ? 0x5b9bd5 : 0xe8a240;

    // Disc floating above ground
    const dot = new THREE.Mesh(mmDotGeo, new THREE.MeshBasicMaterial({ color, depthTest: false }));
    dot.position.set(pos.x, pos.y + 15, pos.z);
    dot.renderOrder = 999;
    mmMarkerGroup.add(dot);

    // Direction cone (floating high above all buildings)
    const cone = new THREE.Mesh(mmConeGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
    const yawRad = (s.yaw || 0) * Math.PI / 180;
    cone.position.set(
      pos.x + Math.sin(yawRad) * 2.5,
      pos.y + 16,
      pos.z + Math.cos(yawRad) * 2.5
    );
    cone.renderOrder = 999;
    mmMarkerGroup.add(cone);
  }

  // Bomb marker
  if (bombPos) {
    const bp = gameToThree(bombPos[0], bombPos[1], bombPos[2]);
    const bomb = new THREE.Mesh(mmBombGeo, new THREE.MeshBasicMaterial({ color: 0xff2222 }));
    bomb.position.set(bp.x, bp.y + 15, bp.z);
    bomb.renderOrder = 999;
    bomb.scale.setScalar(1 + 0.3 * Math.sin(performance.now() / 300));
    mmMarkerGroup.add(bomb);
  }

  scene.add(mmMarkerGroup);

  // ── Render everything in one pass ─────────────────────────────────────
  const origBg = scene.background;
  const origFog = scene.fog;
  scene.background = new THREE.Color(0x1a2a3a);
  scene.fog = null;
  minimapRenderer.render(scene, minimapCamera);

  // ── Cleanup markers ──────────────────────────────────────────────────
  scene.remove(mmMarkerGroup);
  while (mmMarkerGroup.children.length > 0) {
    const c = mmMarkerGroup.children[0];
    mmMarkerGroup.remove(c);
    if (c.material) c.material.dispose();
  }

  scene.background = origBg;
  scene.fog = origFog;
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
