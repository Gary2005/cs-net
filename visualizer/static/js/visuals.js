/**
 * visuals.js — 3D visualization of players, grenades, bomb, smokes, infernos.
 *
 * All positions use Three.js world coordinates (converted from game coords).
 */

import * as THREE from 'three';
import { scene, gameToThree, SCALE, mapGroup, raycaster } from './scene.js';
import {
  weaponName, grenadeCategory, grenadeColor,
  playerTeam, teamColorClass,
} from './replay-core.js';

// ── Shared geometries (reused for all players) ────────────────────────────
let bodyGeo, headGeo, dirGeo, hpBarGeo;

function ensureGeometries() {
  if (!bodyGeo) {
    // Source engine standing hull: 32×32×72 units. 1 unit = 0.0254m.
    // Body: radius=16 units=0.406m, height=55 units=1.4m (visual body, not full hull)
    bodyGeo = new THREE.CylinderGeometry(0.4, 0.4, 1.4, 12);
    // Head: ~8 unit radius = 0.2m
    headGeo = new THREE.SphereGeometry(0.2, 10, 10);
    dirGeo = new THREE.ConeGeometry(0.15, 0.5, 6, 4);
    hpBarGeo = new THREE.PlaneGeometry(1.0, 0.12);
  }
}

// ── Materials ──────────────────────────────────────────────────────────────
const matCT = new THREE.MeshStandardMaterial({
  color: 0x3d8bd4, roughness: 0.35, metalness: 0.25,
  emissive: 0x0d2f55, emissiveIntensity: 0.4,
});
const matT = new THREE.MeshStandardMaterial({
  color: 0xea9d2c, roughness: 0.35, metalness: 0.25,
  emissive: 0x4a2e05, emissiveIntensity: 0.4,
});
const matDead = new THREE.MeshStandardMaterial({
  color: 0x777777, roughness: 0.8, metalness: 0.1, transparent: true, opacity: 0.45,
});
const matHead = new THREE.MeshStandardMaterial({
  color: 0xffcc99, roughness: 0.6, metalness: 0.0,
});
const matDir = new THREE.MeshStandardMaterial({
  color: 0xffffff, roughness: 0.3, metalness: 0.1, emissive: 0x444444, emissiveIntensity: 0.5,
});

// HP bar materials
const matHPHigh = new THREE.MeshBasicMaterial({ color: 0x3fb950, side: THREE.DoubleSide, depthTest: false });
const matHPMid  = new THREE.MeshBasicMaterial({ color: 0xd2991d, side: THREE.DoubleSide, depthTest: false });
const matHPLow  = new THREE.MeshBasicMaterial({ color: 0xf85149, side: THREE.DoubleSide, depthTest: false });
const matHPBg   = new THREE.MeshBasicMaterial({ color: 0x333333, side: THREE.DoubleSide, depthTest: false });

// ── Player group storage ───────────────────────────────────────────────────
/** @type {Array<{group: THREE.Group, body: THREE.Mesh, head: THREE.Mesh,
 *    dir: THREE.Mesh, hpBg: THREE.Mesh, hpFill: THREE.Mesh,
 *    label: THREE.Sprite, weaponSprite: THREE.Sprite}>} */
let playerGroups = [];

// ── Trail storage ──────────────────────────────────────────────────────────
const MAX_TRAIL_POINTS = 150;
let playerTrails = [];         // Array of {line: THREE.Line, positions: Float32Array, writeIdx: number}
const trailGeoTemplate = new THREE.BufferGeometry();
// Pre-allocate
for (let i = 0; i < 10; i++) {
  const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setDrawRange(0, 0);
  playerTrails.push({ positions, geo, writeIdx: 0, count: 0 });
}

// ── Player aim rays ────────────────────────────────────────────────────────
let aimRays = []; // {line, maxLen}
const aimRayMat = new THREE.LineBasicMaterial({ color: 0xff4444, transparent: true, opacity: 0.35, depthTest: true });
let aimRaysVisible = false;

/** Create one aim-ray line per player */
export function createAimRays() {
  clearAimRays();
  for (let i = 0; i < 10; i++) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, 1),
    ]);
    const line = new THREE.Line(geo, aimRayMat.clone());
    line.visible = false;
    line.renderOrder = 600;
    line.name = `aim-ray-${i}`;
    scene.add(line);
    aimRays.push({ line, maxLen: 60 }); // 60 Three units ≈ 2362 game units
  }
}

function clearAimRays() {
  for (const r of aimRays) {
    scene.remove(r.line);
    r.line.geometry.dispose();
  }
  aimRays = [];
}

/**
 * Update aim rays for all alive players.
 * @param {Array} playerStates - from getAllPlayerStates()
 * @param {THREE.Raycaster} raycaster
 * @param {THREE.Group} mapGroup - to raycast against
 */
export function updateAimRays(playerStates, raycaster, mapGroup) {
  if (!aimRaysVisible) {
    for (const r of aimRays) r.line.visible = false;
    return;
  }

  for (let i = 0; i < aimRays.length; i++) {
    const state = playerStates[i];
    if (!state || !state.alive) {
      aimRays[i].line.visible = false;
      continue;
    }

    const pos = gameToThree(state.x, state.y, state.z);
    const eyeY = pos.y + 64 * SCALE; // eye height
    const start = new THREE.Vector3(pos.x, eyeY, pos.z);

    // CS2: yaw=0 → game+X → Three+Z.  Ry(yaw)·(0,0,1) = (sin_yaw, 0, cos_yaw)
    const yawRad = (state.yaw || 0) * Math.PI / 180;
    const pitchRad = (state.pitch || 0) * Math.PI / 180;
    const forward = new THREE.Vector3(0, 0, 1);
    forward.applyAxisAngle(new THREE.Vector3(1, 0, 0), pitchRad);
    forward.applyAxisAngle(new THREE.Vector3(0, 1, 0), yawRad);
    forward.normalize();

    // Raycast against map geometry
    raycaster.set(start, forward);
    raycaster.far = aimRays[i].maxLen;
    const hits = raycaster.intersectObjects(mapGroup.children, true);
    const end = hits.length > 0
      ? hits[0].point
      : start.clone().addScaledVector(forward, aimRays[i].maxLen);

    // Update line geometry
    const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
    aimRays[i].line.geometry.dispose();
    aimRays[i].line.geometry = geo;
    aimRays[i].line.visible = true;
  }
}

export function setAimRaysVisible(v) {
  aimRaysVisible = v;
  if (!v) {
    for (const r of aimRays) r.line.visible = false;
  }
}

// ── X-ray mode ─────────────────────────────────────────────────────────────
let xrayEnabled = false;
const xrayMapMaterial = new THREE.MeshBasicMaterial({
  color: 0x5a6a7a, transparent: true, opacity: 0.15, depthWrite: true,
});
const xrayPlayerGlow = new THREE.MeshBasicMaterial({
  color: 0xffffff, transparent: true, opacity: 0.9, depthTest: false, depthWrite: false,
});

export function setXRayMode(enabled) {
  xrayEnabled = enabled;
  // Map: swap to transparent
  if (mapGroup) {
    mapGroup.traverse((child) => {
      if (child.isMesh && child.material && !child.material.isLineBasicMaterial) {
        if (enabled) {
          child.userData._origMat = child.material;
          child.material = xrayMapMaterial;
          child.renderOrder = 900;
        } else if (child.userData._origMat) {
          child.material = child.userData._origMat;
          child.renderOrder = 0;
        }
      }
    });
  }
  // Players: glow through walls
  for (const pg of playerGroups) {
    if (!pg.group || !pg.body) continue;
    if (enabled) {
      pg.body.renderOrder = 999;
      pg.body.material.depthTest = false;
      pg.body.material.depthWrite = false;
      pg.body.material.emissive = new THREE.Color(pg.body.material.color);
      pg.body.material.emissiveIntensity = 0.8;
      if (pg.pitchGroup) pg.pitchGroup.children.forEach(c => {
        if (c.isMesh) { c.renderOrder = 999; c.material.depthTest = false; c.material.depthWrite = false; }
      });
    } else {
      pg.body.renderOrder = 0;
      pg.body.material.depthTest = true;
      pg.body.material.depthWrite = true;
      pg.body.material.emissive = new THREE.Color(0x000000);
      pg.body.material.emissiveIntensity = 0;
      if (pg.pitchGroup) pg.pitchGroup.children.forEach(c => {
        if (c.isMesh) { c.renderOrder = 0; c.material.depthTest = true; c.material.depthWrite = true; }
      });
    }
  }
}

/** Hide/show a specific player model (for first-person view) */
export function setPlayerModelVisible(idx, visible) {
  if (idx >= 0 && idx < playerGroups.length) {
    playerGroups[idx].group.visible = visible;
    // Also hide trail
    const t = playerTrails[idx];
    if (t && t.line) t.line.visible = visible;
  }
}

// ── Grenade trajectory storage ─────────────────────────────────────────────
// Each trajectory: { line: THREE.Line, dot: THREE.Mesh, points: [{t,x,y,z}], type: str }
let grenadeTrajectories = [];
const nadeDotGeo = new THREE.SphereGeometry(0.2, 10, 10);

// ── Bomb model ─────────────────────────────────────────────────────────────
let bombGroup = null;
let bombPulseLight = null;

// ── Smoke / Inferno storage ────────────────────────────────────────────────
let activeSmokeMeshes = [];
let activeInfernoMeshes = [];

// 简约风格：单个柔和半透明球体（体积感靠多层叠加的固定 alpha 实现，无随机抖动）
const smokeGeo = new THREE.SphereGeometry(1.0, 20, 14);
const SMOKE_RADIUS = 3.2;

// 火焰：单个柔和发光圆盘 + 中心亮斑（干净、低饱和、无粒子乱跳）
const infernoGeo = new THREE.CircleGeometry(2.6, 28);

// ── Flash effect mesh ──────────────────────────────────────────────────────
const flashGeo = new THREE.SphereGeometry(0.5, 8, 8);
const flashMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
const flashMeshes = [];

// Muzzle flash pool
const muzzleGeo = new THREE.SphereGeometry(0.5, 8, 8);
const muzzleMat = new THREE.MeshBasicMaterial({ color: 0xffcc44, transparent: true, opacity: 0.95 });
const muzzleFlashPool = [];
const MAX_MUZZLE = 20;

// Footstep indicator pool
const fsGeo = new THREE.RingGeometry(0.2, 0.6, 16);
const fsMatTemplate = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.8, depthTest: true, depthWrite: false });
const footstepPool = [];
const MAX_FOOTSTEPS = 30;

// ── Initialization ─────────────────────────────────────────────────────────

/**
 * Create all 10 player groups and add to scene.
 *
 * Hierarchy:
 *   group (yaw + world position)
 *   ├── body
 *   ├── pitchGroup  (rotation.x = -pitch, handles looking up/down)
 *   │   ├── head
 *   │   └── dir (cone, pre-rotated to point forward)
 *   ├── hpBg / hpFill
 *   └── labels
 */
export function createPlayers() {
  ensureGeometries();
  clearPlayers();

  for (let i = 0; i < 10; i++) {
    const group = new THREE.Group();
    group.name = `player-${i}`;
    group.visible = false;

    // Body (cylinder height=1.4, centered at half height)
    const body = new THREE.Mesh(bodyGeo, matCT.clone());
    body.position.y = 0.7;
    body.castShadow = true;
    group.add(body);

    // Pitch sub-group — rotates vertically for pitch
    const pitchGroup = new THREE.Group();
    pitchGroup.name = `pitch-${i}`;
    group.add(pitchGroup);

    // Head (child of pitch group, top of body + head radius)
    const head = new THREE.Mesh(headGeo, matHead);
    head.position.y = 1.6;    // body top (1.4) + head radius (0.2)
    head.castShadow = true;
    pitchGroup.add(head);

    // Direction cone (child of pitch group, chest height)
    const dir = new THREE.Mesh(dirGeo, matDir);
    dir.position.y = 1.0;     // chest height
    dir.position.z = 0.45;    // in front of body
    dir.rotation.x = Math.PI / 2; // tip-forward
    pitchGroup.add(dir);

    // HP Bar background
    const hpBg = new THREE.Mesh(hpBarGeo, matHPBg);
    hpBg.position.y = 1.9;
    hpBg.renderOrder = 1;
    group.add(hpBg);

    // HP Bar fill
    const hpFill = new THREE.Mesh(hpBarGeo, matHPHigh);
    hpFill.position.y = 1.9;
    hpFill.position.z = 0.01;
    hpFill.renderOrder = 2;
    group.add(hpFill);

    // Name label
    const label = createLabelSprite('', i < 5 ? 0x5b9bd5 : 0xe8a240);
    label.position.y = 2.1;
    label.scale.set(2.0, 0.5, 1);
    group.add(label);

    // Weapon icon label
    const weaponLabel = createLabelSprite('', 0xcccccc, 10);
    weaponLabel.position.y = 0.15;
    weaponLabel.scale.set(1.5, 0.35, 1);
    group.add(weaponLabel);

    scene.add(group);

    // Trail — 每个玩家独立材质（颜色在 updatePlayers 中按实际队伍设置）
    const trailLine = new THREE.Line(
      playerTrails[i].geo,
      new THREE.LineBasicMaterial({
        color: 0x5b9bd5,
        transparent: true,
        opacity: 0.5,
        linewidth: 1,
      })
    );
    trailLine.visible = true;
    trailLine.name = `trail-${i}`;
    scene.add(trailLine);

    // 死亡标记：队伍色叉号（两条交叉的扁条，贴地平放），默认隐藏
    const deadX = new THREE.Group();
    deadX.name = `dead-x-${i}`;
    deadX.visible = false;
    const xBarGeo = new THREE.BoxGeometry(1.2, 0.12, 0.12);
    const xMat = new THREE.MeshBasicMaterial({
      color: 0x444444, transparent: true, opacity: 0.9, depthTest: false,
    });
    const bar1 = new THREE.Mesh(xBarGeo, xMat);
    bar1.rotation.y = Math.PI / 4;      // 45°
    const bar2 = new THREE.Mesh(xBarGeo, xMat);   // 共享同一材质（颜色一起变）
    bar2.rotation.y = -Math.PI / 4;     // -45°
    deadX.add(bar1, bar2);
    deadX.renderOrder = 998;
    scene.add(deadX);

    playerGroups.push({
      group, body, pitchGroup, head, dir,
      hpBg, hpFill, label, weaponSprite: weaponLabel,
      deadX, xMat,
      cachedGroundY: null,   // 死亡地面高度缓存
    });

    playerTrails[i].line = trailLine;
  }

  // Create bomb model
  createBomb();
}

function clearPlayers() {
  for (const pg of playerGroups) {
    scene.remove(pg.group);
    if (pg.deadX) scene.remove(pg.deadX);   // 清理死亡叉号
  }
  // Only remove trail line meshes from scene, keep the data arrays
  for (const pt of playerTrails) {
    if (pt.line) {
      scene.remove(pt.line);
      pt.line = null;
    }
    // Reset trail data
    pt.writeIdx = 0;
    pt.count = 0;
    pt.geo.setDrawRange(0, 0);
  }
  playerGroups = [];
  if (bombGroup) {
    scene.remove(bombGroup);
    bombGroup = null;
  }
}

/**
 * Create the bomb (C4) model.
 */
function createBomb() {
  bombGroup = new THREE.Group();
  bombGroup.name = 'bomb';
  bombGroup.visible = false;

  // Main box
  const boxGeo = new THREE.BoxGeometry(0.5, 0.3, 0.7);
  const boxMat = new THREE.MeshStandardMaterial({
    color: 0x334422, roughness: 0.5, metalness: 0.2,
  });
  const box = new THREE.Mesh(boxGeo, boxMat);
  box.castShadow = true;
  bombGroup.add(box);

  // Keypad
  const kpGeo = new THREE.BoxGeometry(0.3, 0.05, 0.4);
  const kpMat = new THREE.MeshStandardMaterial({
    color: 0x222222, roughness: 0.3, metalness: 0.5,
  });
  const kp = new THREE.Mesh(kpGeo, kpMat);
  kp.position.y = 0.18;
  bombGroup.add(kp);

  // Blinking light
  const lightGeo = new THREE.SphereGeometry(0.08, 8, 8);
  const lightMat = new THREE.MeshBasicMaterial({ color: 0xff2222 });
  bombPulseLight = new THREE.Mesh(lightGeo, lightMat);
  bombPulseLight.position.y = 0.22;
  bombGroup.add(bombPulseLight);

  // Point light for glow
  const ptLight = new THREE.PointLight(0xff0000, 2, 3);
  ptLight.position.y = 0.2;
  bombGroup.add(ptLight);

  scene.add(bombGroup);
}

/**
 * Create a canvas-based text sprite for labels.
 */
function createLabelSprite(text, colorHex, fontSize = 12) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = `bold ${fontSize}px "SF Mono", "Consolas", monospace`;
  ctx.fillStyle = `#${colorHex.toString(16).padStart(6, '0')}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);

  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false, depthWrite: false,
  });
  const sprite = new THREE.Sprite(mat);
  return sprite;
}

function updateLabelSprite(sprite, text, colorHex, fontSize = 12) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = `bold ${fontSize}px "SF Mono", "Consolas", monospace`;
  ctx.fillStyle = `#${colorHex.toString(16).padStart(6, '0')}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);

  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  sprite.material.map.dispose();
  sprite.material.map = tex;
  sprite.material.needsUpdate = true;
}

// ── Update functions ───────────────────────────────────────────────────────

/**
 * Update all player positions, rotations, and states for the current sample.
 * @param {Array} playerStates - from getAllPlayerStates(currentSampleIdx)
 * @param {number} sampleIdx
 * @param {Array<string>} teams - team array for current round
 * @param {boolean} showNames
 * @param {boolean} showTrails
 */
export function updatePlayers(playerStates, sampleIdx, teams, showNames, showTrails) {
  for (let i = 0; i < playerGroups.length; i++) {
    const pg = playerGroups[i];
    const state = playerStates[i];
    const team = teams[i] || '?';

    if (!state) {
      pg.group.visible = false;
      continue;
    }

    const pos = gameToThree(state.x, state.y, state.z);
    pg.group.visible = true;

    // ── Dead player: 隐藏模型，在最后位置显示队伍色叉号 ─────────────
    if (!state.alive) {
      // 隐藏玩家模型（身体/头/血条/标签等）
      pg.group.visible = false;

      // 用 raycast 检测正下方的真实地面高度（斜坡/台阶也能贴地）。
      // 只在刚死亡时算一次并缓存，避免每帧对大地图 raycast。
      if (pg.cachedGroundY === null) {
        let groundY = pos.y;
        try {
          const rayOrigin = new THREE.Vector3(pos.x, pos.y + 10, pos.z);
          raycaster.set(rayOrigin, new THREE.Vector3(0, -1, 0));
          raycaster.far = 30;
          const hits = raycaster.intersectObjects(mapGroup.children, true);
          if (hits.length > 0) {
            groundY = hits[0].point.y;
          }
        } catch (_) { /* 地图未加载时用玩家高度 */ }
        pg.cachedGroundY = groundY;
      }

      // 叉号：队伍色（CT 蓝 / T 橙），贴地平放
      const deadColor = team === 'CT' ? 0x4da3ff : 0xff9f43;
      pg.xMat.color.set(deadColor);
      pg.deadX.position.set(pos.x, pg.cachedGroundY + 0.15, pos.z);
      pg.deadX.visible = true;

      // Continue trail for dead players too
      if (showTrails) updateTrail(i, pos);
      continue; // skip alive-specific rendering
    }

    // ── Alive player ────────────────────────────────────────────────────
    pg.group.position.set(pos.x, pos.y, pos.z);
    pg.group.visible = true;
    pg.pitchGroup.visible = true;
    pg.hpBg.visible = true;
    pg.hpFill.visible = true;
    // 复活时隐藏死亡叉号
    if (pg.deadX) pg.deadX.visible = false;

    // Team color (set color only, keep cloned material instance)
    const teamColor = team === 'CT' ? 0x5b9bd5 : 0xe8a240;
    pg.body.material.color.set(teamColor);
    pg.body.material.opacity = 1;
    pg.body.material.transparent = false;
    pg.head.material = matHead;

    // Trail 颜色与队伍同步（每个玩家独立材质，互不影响）
    const trail = playerTrails[i];
    if (trail && trail.line && trail.line.material) {
      trail.line.material.color.set(teamColor);
    }

    // Flash glow: white body when flash-blinded
    if (state.flash > 0 && state.alive) {
      pg.body.material.emissive = new THREE.Color(0xffffff);
      pg.body.material.emissiveIntensity = 1.0;
    } else {
      pg.body.material.emissive = new THREE.Color(0x000000);
      pg.body.material.emissiveIntensity = 0;
    }

    // CS2 convention: yaw=0 → +X in game,   pitch<0 → look down
    // Game +X → Three +Z, so yaw=0 → Three +Z → rotation.y = 0
    // yaw=90 → game +Y → Three +X → rotation.y = π/2
    //   Ry(θ)·(0,0,1) = (sinθ,0,cosθ), so θ = yawRad directly.
    const yawRad = THREE.MathUtils.degToRad(state.yaw);
    const pitchRad = THREE.MathUtils.degToRad(state.pitch);
    pg.group.rotation.set(0, yawRad, 0);
    pg.pitchGroup.rotation.set(pitchRad, 0, 0);

    // HP Bar
    const hpPct = Math.max(0, state.hp / 100);
    pg.hpFill.scale.x = hpPct;
    pg.hpFill.position.x = -(1 - hpPct) / 2;
    if (hpPct > 0.6) pg.hpFill.material = matHPHigh;
    else if (hpPct > 0.3) pg.hpFill.material = matHPMid;
    else pg.hpFill.material = matHPLow;
    pg.hpBg.visible = state.alive;
    pg.hpFill.visible = state.alive;

    // Name label
    if (pg.label) {
      pg.label.visible = showNames;
    }

    // Weapon label
    if (pg.weaponSprite) {
      const wName = weaponName(state.weapon);
      const shortName = wName.length > 10 ? wName.substring(0, 9) + '…' : wName;
      updateLabelSprite(pg.weaponSprite, state.alive ? shortName : '💀', 0xcccccc, 10);
    }

    // Update trail
    if (showTrails && state.alive) {
      updateTrail(i, pos);
    }
  }
}

/**
 * Update player name labels.
 */
export function updatePlayerNames(playerStates, teams) {
  for (let i = 0; i < playerGroups.length; i++) {
    const pg = playerGroups[i];
    if (!pg.label) continue;
    const state = playerStates[i];
    const team = teams[i] || '?';
    const name = state ? `#${i}` : '';
    const color = team === 'CT' ? 0x5b9bd5 : 0xe8a240;
    updateLabelSprite(pg.label, name, color, 12);
  }
}

/**
 * Set player name labels from match data.
 */
export function setPlayerNames(playerNames, currentTeams) {
  for (let i = 0; i < playerGroups.length; i++) {
    const pg = playerGroups[i];
    if (!pg.label) continue;
    const name = playerNames[i] || `Player ${i}`;
    const team = currentTeams ? currentTeams[i] : '?';
    const color = team === 'CT' ? 0x5b9bd5 : 0xe8a240;
    updateLabelSprite(pg.label, name, color, 12);
  }
}

/**
 * Add a point to a player's trail.
 */
function updateTrail(playerIdx, pos) {
  const trail = playerTrails[playerIdx];
  if (!trail) return;

  const arr = trail.positions;
  const idx = trail.writeIdx * 3;

  // Check distance from last point
  if (trail.count > 0) {
    const lastIdx = ((trail.writeIdx - 1 + MAX_TRAIL_POINTS) % MAX_TRAIL_POINTS) * 3;
    const dx = pos.x - arr[lastIdx];
    const dy = pos.y - arr[lastIdx];
    const dz = pos.z - arr[lastIdx];
    if (dx * dx + dy * dy + dz * dz < 0.01) return; // too close, skip
  }

  arr[idx] = pos.x;
  arr[idx + 1] = pos.y;
  arr[idx + 2] = pos.z;
  trail.writeIdx = (trail.writeIdx + 1) % MAX_TRAIL_POINTS;
  trail.count = Math.min(trail.count + 1, MAX_TRAIL_POINTS);

  // Rebuild line from circular buffer
  const allPts = [];
  for (let i = 0; i < trail.count; i++) {
    const ri = (trail.writeIdx - trail.count + i + MAX_TRAIL_POINTS) % MAX_TRAIL_POINTS;
    allPts.push(arr[ri * 3], arr[ri * 3 + 1], arr[ri * 3 + 2]);
  }
  trail.geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(allPts), 3));
  trail.geo.setDrawRange(0, trail.count);
  if (trail.line) trail.line.geometry = trail.geo;
}

/**
 * Clear all player trails.
 */
export function clearAllTrails() {
  for (const trail of playerTrails) {
    trail.writeIdx = 0;
    trail.count = 0;
    if (trail.line) trail.line.geometry.setDrawRange(0, 0);
  }
}

/**
 * Show/hide trails.
 */
export function setTrailsVisible(visible) {
  for (const trail of playerTrails) {
    if (trail.line) trail.line.visible = visible;
  }
}

/**
 * Show/hide name labels.
 */
export function setNamesVisible(visible) {
  for (const pg of playerGroups) {
    if (pg.label) pg.label.visible = visible;
  }
}

// ── Bomb ───────────────────────────────────────────────────────────────────

/**
 * Update bomb position and state.
 * @param {Array|null} bombPos - game [x,y,z] or null
 * @param {boolean} planted
 * @param {number} time - current time for pulse animation
 */
export function updateBomb(bombPos, planted, time) {
  if (!bombGroup) return;
  if (!bombPos) {
    bombGroup.visible = false;
    return;
  }

  bombGroup.visible = true;
  const pos = gameToThree(bombPos[0], bombPos[1], bombPos[2]);
  bombGroup.position.set(pos.x, pos.y + 0.3, pos.z);

  // Pulse the light
  if (bombPulseLight && planted) {
    const pulse = 0.5 + 0.5 * Math.sin(time * 8);
    bombPulseLight.material.opacity = 0.5 + pulse * 0.5;
    bombPulseLight.scale.setScalar(1 + pulse * 0.5);
  }
}

// ── Grenades ───────────────────────────────────────────────────────────────

/**
 * Build trajectory meshes from pre-grouped trajectory data.
 * Call once per round.
 * @param {Array} trajectories - from buildGrenadeTrajectories()
 */
export function buildGrenadeTrajectories(trajectories) {
  clearGrenadeTrajectories();

  for (const traj of trajectories) {
    const color = grenadeColor(traj.type);
    const pts3 = traj.points.map(p => {
      const t = gameToThree(p.x, p.y, p.z);
      return new THREE.Vector3(t.x, t.y, t.z);
    });

    // Trajectory line
    const lineGeo = new THREE.BufferGeometry().setFromPoints(pts3);
    const lineMat = new THREE.LineBasicMaterial({
      color, transparent: true, opacity: 0.5, linewidth: 1, depthTest: true,
    });
    const line = new THREE.Line(lineGeo, lineMat);
    line.renderOrder = 500;
    scene.add(line);

    // Animated dot
    const dotMat = new THREE.MeshStandardMaterial({
      color, roughness: 0.2, metalness: 0.1,
      emissive: color, emissiveIntensity: 0.7,
    });
    const dot = new THREE.Mesh(nadeDotGeo, dotMat);
    dot.renderOrder = 501;
    dot.visible = false;
    // Store metadata for click-to-inspect
    dot.userData = {
      isGrenadeDot: true,
      type: traj.type,
      thrower: traj.thrower,
      points: traj.points,
    };
    dot.name = `nade-${traj.type}-p${traj.thrower}`;
    scene.add(dot);

    grenadeTrajectories.push({
      line, dot,
      points: traj.points,
      type: traj.type,
      thrower: traj.thrower,
      settleTime: traj.settleTime,
    });
  }
}

/**
 * Update grenade dot positions based on current playback time.
 * @param {number} timeSec - current round-relative time in seconds
 */
export function updateGrenadeTrajectories(timeSec) {
  for (const traj of grenadeTrajectories) {
    const pts = traj.points;
    if (pts.length < 2) {
      traj.dot.visible = false;
      continue;
    }

    const firstSec = pts[0].t;
    const settleTime = traj.settleTime || pts[pts.length - 1].t;

    // Hide if not yet thrown or already settled (+short grace period)
    if (timeSec < firstSec || timeSec > settleTime + 0.5) {
      traj.dot.visible = false;
      traj.line.visible = false;
      continue;
    }

    // Find the two surrounding time points (both in round-relative seconds)
    let i0 = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      if (timeSec >= pts[i].t && timeSec <= pts[i + 1].t) {
        i0 = i;
        break;
      }
      i0 = Math.min(i, pts.length - 2);
    }

    const tA = pts[i0].t;
    const tB = pts[Math.min(i0 + 1, pts.length - 1)].t;
    let frac = (timeSec - tA) / Math.max(0.001, tB - tA);
    frac = Math.max(0, Math.min(1, frac));

    const a = gameToThree(pts[i0].x, pts[i0].y, pts[i0].z);
    const b = gameToThree(pts[Math.min(i0 + 1, pts.length - 1)].x,
                          pts[Math.min(i0 + 1, pts.length - 1)].y,
                          pts[Math.min(i0 + 1, pts.length - 1)].z);

    traj.dot.position.set(
      a.x + (b.x - a.x) * frac,
      a.y + (b.y - a.y) * frac,
      a.z + (b.z - a.z) * frac
    );
    traj.dot.visible = true;
    traj.line.visible = true;
  }
}

function clearGrenadeTrajectories() {
  for (const traj of grenadeTrajectories) {
    scene.remove(traj.line);
    scene.remove(traj.dot);
    traj.line.geometry.dispose();
    traj.line.material.dispose();
    traj.dot.material.dispose();
  }
  grenadeTrajectories = [];
}

/**
 * @deprecated — replaced by buildGrenadeTrajectories / updateGrenadeTrajectories
 */
export function addGrenade(_grenade) { /* no-op */ }
export function updateGrenades(_grenades) { /* no-op */ }

// ── Smokes ─────────────────────────────────────────────────────────────────

/**
 * Update active smoke cloud meshes.
 * @param {Array} smokes - active smoke intervals at current time
 */
export function updateSmokes(smokes) {
  // Remove old
  for (const m of activeSmokeMeshes) {
    scene.remove(m);
    if (m.material) m.material.dispose();
    if (m.geometry && m.geometry !== smokeGeo) m.geometry.dispose();
  }
  activeSmokeMeshes = [];

  if (!smokes || smokes.length === 0) return;

  for (const s of smokes) {
    const pos = gameToThree(s.x, s.y, s.z);
    const group = new THREE.Group();
    group.name = 'smoke-group';
    group.position.set(pos.x, pos.y, pos.z);

    // 简约：单个柔和半透明球体，带轻微纵向拉伸（烟柱感）
    const mat = new THREE.MeshBasicMaterial({
      color: 0xdfe4ea,          // 柔和浅灰
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
    });
    const puff = new THREE.Mesh(smokeGeo, mat);
    puff.scale.set(SMOKE_RADIUS, SMOKE_RADIUS * 1.25, SMOKE_RADIUS);
    puff.position.y = SMOKE_RADIUS * 1.1;
    puff.renderOrder = 999;
    puff.userData = {
      baseOpacity: mat.opacity,
      phase: Math.random() * Math.PI * 2,
    };
    group.add(puff);

    scene.add(group);
    activeSmokeMeshes.push(group);
  }
}

/**
 * 每帧更新烟雾动画：极慢的透明度呼吸（几乎察觉不到，保持静止的简约感）。
 */
export function updateSmokeAnimations(dt, timeSec) {
  // 用实时时钟驱动，非常慢的呼吸（±0.03，肉眼几乎不可见 → 简约不跳）
  const t = performance.now() / 1000;
  for (const group of activeSmokeMeshes) {
    group.traverse((child) => {
      if (child.isMesh && child.userData.baseOpacity !== undefined) {
        const u = child.userData;
        const ph = u.phase + t * 0.3;
        child.material.opacity = u.baseOpacity * (1.0 + 0.08 * Math.sin(ph));
      }
    });
  }
}

// ── Infernos ───────────────────────────────────────────────────────────────

/**
 * Update active inferno meshes — 简约：柔和发光圆盘 + 中心亮斑。
 * @param {Array} infernos - active inferno intervals at current time
 */
export function updateInfernos(infernos) {
  for (const m of activeInfernoMeshes) {
    scene.remove(m);
    if (m.material) {
      if (Array.isArray(m.material)) m.material.forEach(mm => mm.dispose());
      else m.material.dispose();
    }
  }
  activeInfernoMeshes = [];

  if (!infernos || infernos.length === 0) return;

  for (const inf of infernos) {
    const pos = gameToThree(inf.x, inf.y, inf.z);
    const group = new THREE.Group();
    group.name = 'inferno-group';
    group.position.set(pos.x, pos.y, pos.z);

    // 火焰主体：柔和橙色半透明圆盘（贴地，代表燃烧区域）
    const mat = new THREE.MeshBasicMaterial({
      color: 0xff7a2d,
      transparent: true,
      opacity: 0.4,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
    });
    const disk = new THREE.Mesh(infernoGeo, mat);
    disk.rotation.x = -Math.PI / 2;
    disk.position.y = 0.1;
    disk.renderOrder = 998;
    disk.userData = {
      baseOpacity: mat.opacity,
      phase: Math.random() * Math.PI * 2,
    };
    group.add(disk);

    // 中心亮斑：温暖的橙黄（火焰核心）
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0xffc26b,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    const core = new THREE.Mesh(new THREE.SphereGeometry(0.7, 12, 10), coreMat);
    core.position.y = 0.25;
    core.renderOrder = 999;
    core.userData = {
      baseOpacity: coreMat.opacity,
      phase: Math.random() * Math.PI * 2,
      isCore: true,
    };
    group.add(core);

    scene.add(group);
    activeInfernoMeshes.push(group);
  }
}

/**
 * 每帧更新火焰动画：极慢的柔和呼吸（几乎察觉不到，简约不闪）。
 */
export function updateInfernoAnimations(dt, timeSec) {
  const t = performance.now() / 1000;
  for (const group of activeInfernoMeshes) {
    group.traverse((child) => {
      if (child.isMesh && child.userData.baseOpacity !== undefined) {
        const u = child.userData;
        const ph = u.phase + t * 0.25;   // 非常慢
        const breathe = 1.0 + 0.07 * Math.sin(ph);
        if (child.material) {
          child.material.opacity = u.baseOpacity * breathe;
        }
        if (u.isCore) {
          child.scale.setScalar(1 + 0.05 * Math.sin(ph * 0.8));
        }
      }
    });
  }
}

// ── Muzzle flash ───────────────────────────────────────────────────────────

/**
 * Spawn a muzzle flash at a player's position.
 * @param {number} playerIdx
 * @param {object} pos - Three.js position
 */
export function spawnMuzzleFlash(playerIdx, pos) {
  const mesh = muzzleFlashPool.find(m => !m.visible);
  let flash;
  if (mesh) {
    flash = mesh;
    flash.visible = true;
  } else {
    flash = new THREE.Mesh(muzzleGeo, muzzleMat.clone());
    flash.userData = { life: 0 };
    scene.add(flash);
    muzzleFlashPool.push(flash);
  }

  flash.position.set(pos.x, pos.y + 1.5, pos.z);
  flash.userData.life = 0.2; // 200ms visible
  flash.material.opacity = 0.95;
  flash.scale.set(1, 1, 1);
}

/**
 * Update muzzle flashes (fade out).
 */
export function updateMuzzleFlashes(dt) {
  for (const flash of muzzleFlashPool) {
    if (!flash.visible) continue;
    flash.userData.life -= dt;
    if (flash.userData.life <= 0) {
      flash.visible = false;
    } else {
      const t = flash.userData.life / 0.2;
      flash.material.opacity = t * 0.95;
      flash.scale.setScalar(1 + (1 - t) * 1.5);
    }
  }
}

/**
 * Spawn a footstep indicator ring at a player position.
 */
export function spawnFootstep(playerIdx, pos) {
  let ring = footstepPool.find(r => !r.visible);
  if (!ring) {
    ring = new THREE.Mesh(fsGeo, fsMatTemplate.clone());
    ring.rotation.x = -Math.PI / 2; // lay flat on ground
    ring.userData = { life: 0 };
    ring.renderOrder = 997;
    ring.material.depthTest = true;
    ring.material.depthWrite = false;
    scene.add(ring);
    footstepPool.push(ring);
  }
  ring.position.set(pos.x, pos.y + 0.05, pos.z);
  ring.userData.life = 0.5; // 500ms visible
  ring.visible = true;
  ring.scale.set(1, 1, 1);
}

/**
 * Update footstep indicators (fade + expand out).
 */
export function updateFootsteps(dt) {
  for (const ring of footstepPool) {
    if (!ring.visible) continue;
    ring.userData.life -= dt;
    if (ring.userData.life <= 0) {
      ring.visible = false;
    } else {
      const t = ring.userData.life / 0.5;
      ring.scale.setScalar(1 + (1 - t) * 3);
      ring.material.opacity = t * 0.7;
    }
  }
}

// ── Kill effect ────────────────────────────────────────────────────────────

/**
 * Show a kill effect at a position.
 * @param {object} pos - Three.js position
 */
export function showKillEffect(pos) {
  const geo = new THREE.RingGeometry(0.3, 0.5, 16);
  const mat = new THREE.MeshBasicMaterial({
    color: 0xff0000, side: THREE.DoubleSide,
    transparent: true, opacity: 0.9, depthTest: false, depthWrite: false,
  });
  const ring = new THREE.Mesh(geo, mat);
  ring.position.copy(pos);
  ring.position.y += 1.2;
  ring.userData = { life: 1.0 };
  ring.name = 'kill-effect';
  scene.add(ring);

  // Auto-remove after animation
  setTimeout(() => {
    scene.remove(ring);
    ring.geometry.dispose();
    ring.material.dispose();
  }, 1000);
}

// ── Cleanup ────────────────────────────────────────────────────────────────

/**
 * Remove all visual entities (for round change).
 */
export function clearAllEntities() {
  // Grenade trajectories
  clearGrenadeTrajectories();
  // Hide aim rays
  for (const r of aimRays) r.line.visible = false;

  // Smokes
  for (const m of activeSmokeMeshes) {
    scene.remove(m);
    if (m.material) m.material.dispose();
  }
  activeSmokeMeshes = [];

  // Infernos
  for (const m of activeInfernoMeshes) {
    scene.remove(m);
    if (m.material) m.material.dispose();
  }
  activeInfernoMeshes = [];

  // Kill effects
  scene.children
    .filter(c => c.name === 'kill-effect')
    .forEach(c => {
      scene.remove(c);
      c.geometry?.dispose();
      c.material?.dispose();
    });

  // Muzzle flashes
  for (const f of muzzleFlashPool) f.visible = false;
  // Footstep rings
  for (const r of footstepPool) r.visible = false;

  // Clear trails
  clearAllTrails();

  // Hide bomb
  if (bombGroup) bombGroup.visible = false;
}
