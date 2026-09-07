/**
 * replay-core.js — Data loading, parsing, tick interpolation, replay engine.
 *
 * The V2 JSON format (see docs/demo-json-format.md) is loaded and stored as-is.
 * This module provides accessors and interpolation helpers.
 */

// ── Global state (shared across modules) ──────────────────────────────────

/** @type {object|null} Raw V2 JSON data */
export let matchData = null;

/** @type {number} Current round index (0-based) */
export let currentRoundIdx = 0;

/** @type {number} Current sample index within the round */
export let currentSampleIdx = 0;

/** @type {number} Current round-relative time in seconds */
export let currentTime = 0;

/** @type {boolean} Playback state */
export let isPlaying = false;

/** @type {number} Playback speed multiplier */
export let playSpeed = 1.0;

/** @type {object|null} Current round data (cached) */
export let currentRound = null;

// ── State setters (ES module imports are read‑only, so modules that need
//    to mutate state must go through these) ──────────────────────────────

export function setPlaying(v) { isPlaying = v; }
export function togglePlaying() { isPlaying = !isPlaying; }
export function setPlaySpeed(v) { playSpeed = v; }

/** @type {number} Total number of rounds */
export let totalRounds = 0;

// Reverse lookups
let idxToWeapon = {};   // int → name
let idxToPlace = {};    // int → name

// Event tracking per round
let firedKills = new Set();
let firedDamage = new Set();

/**
 * Load match data from parsed V2 JSON.
 * @param {object} data
 */
export function loadMatchData(data) {
  matchData = data;
  totalRounds = data.rounds.length;
  currentRoundIdx = 0;
  currentSampleIdx = 0;
  currentTime = 0;
  isPlaying = false;

  // Build reverse lookups
  idxToWeapon = {};
  for (const [name, idx] of Object.entries(data.weapons || {})) {
    idxToWeapon[idx] = name;
  }
  idxToPlace = {};
  for (const [name, idx] of Object.entries(data.places || {})) {
    idxToPlace[idx] = name;
  }

  // Load first round
  loadRound(0);
}

/**
 * Load a specific round.
 * @param {number} roundIdx
 */
export function loadRound(roundIdx) {
  if (!matchData || roundIdx < 0 || roundIdx >= totalRounds) return;
  currentRoundIdx = roundIdx;
  currentRound = matchData.rounds[roundIdx];
  currentSampleIdx = 0;
  currentTime = currentRound.round_seconds[0] || 0;
  firedKills.clear();
  firedDamage.clear();
}

/**
 * Get the current sample count for the active round.
 */
export function getSampleCount() {
  if (!currentRound) return 0;
  return currentRound.ticks.length;
}

/**
 * Get the round duration in seconds.
 */
export function getRoundDuration() {
  if (!currentRound) return 0;
  const secs = currentRound.round_seconds;
  return secs.length > 0 ? secs[secs.length - 1] : 0;
}

/**
 * Advance the replay by deltaTime seconds (game time).
 * currentTime progresses continuously (not snapped to samples) so
 * interpolation looks smooth at any playback speed.
 * @param {number} deltaSec - real-time seconds elapsed
 * @returns {object} { events: {...}, advanced: bool, roundEnded: bool }
 */
export function advanceReplay(deltaSec) {
  if (!currentRound || !isPlaying) return { events: null, advanced: false, roundEnded: false };

  const maxTime = getRoundDuration();
  const prevTime = currentTime;
  const prevIdx = currentSampleIdx;

  // Advance game time continuously
  currentTime += deltaSec * playSpeed;
  if (currentTime >= maxTime) {
    currentTime = maxTime;
  }

  // Find which sample index the new time falls at
  const secs = currentRound.round_seconds;
  let newIdx = prevIdx;
  for (let i = prevIdx; i < secs.length; i++) {
    if (secs[i] > currentTime) {
      newIdx = Math.max(prevIdx, i - 1);
      break;
    }
    newIdx = i;
  }
  currentSampleIdx = newIdx;

  // Gather events that occurred in the time window we crossed
  const events = collectEvents(prevIdx, newIdx);

  // Gather sound events with exact tick→time conversion
  const newSounds = collectSoundEvents(prevIdx, newIdx);

  const roundEnded = currentTime >= maxTime;

  return { events, newSounds, advanced: newIdx !== prevIdx || roundEnded, roundEnded };
}

/**
 * Collect sound events between two sample indices, converting ticks to
 * round-relative seconds for precise frame-level triggering.
 */
function collectSoundEvents(fromIdx, toIdx) {
  if (!currentRound) return [];
  const fromTick = currentRound.ticks[fromIdx] || 0;
  const toTick = currentRound.ticks[toIdx] || Infinity;
  const sounds = currentRound.events.sound || [];
  return sounds
    .filter(s => s.t > fromTick && s.t <= toTick)
    .map(s => ({ ...s, sec: tickToTime(s.t) }));
}

/**
 * Collect events that should fire between two sample indices.
 */
function collectEvents(fromIdx, toIdx) {
  if (!currentRound) return null;

  const fromTick = currentRound.ticks[fromIdx] || 0;
  const toTick = currentRound.ticks[toIdx] || Infinity;
  const ev = currentRound.events;

  const result = {
    kills: [],
    damage: [],
    bomb: [],
    newSmokes: [],
    newInfernos: [],
    endedSmokes: [],
    endedInfernos: [],
  };

  // Kills
  for (const k of (ev.kills || [])) {
    if (k.t > fromTick && k.t <= toTick && !firedKills.has(k.t + '_' + k.v)) {
      firedKills.add(k.t + '_' + k.v);
      result.kills.push(k);
    }
  }

  // Damage
  for (const d of (ev.damage || [])) {
    if (d.t > fromTick && d.t <= toTick && !firedDamage.has(d.t + '_' + d.v + '_' + d.hp)) {
      firedDamage.add(d.t + '_' + d.v + '_' + d.hp);
      result.damage.push(d);
    }
  }

  // Bomb events
  for (const b of (ev.bomb || [])) {
    if (b.t > fromTick && b.t <= toTick) {
      result.bomb.push(b);
    }
  }

  // Active smokes/infernos at current time
  const ct = currentTime;
  for (const s of (ev.smokes || [])) {
    if (s.ts <= ct && s.te >= ct) {
      result.newSmokes.push(s);
    }
    if (s.te && s.te > ct - 0.1 && s.te <= ct + 0.2) {
      result.endedSmokes.push(s);
    }
  }
  for (const inf of (ev.infernos || [])) {
    if (inf.ts <= ct && (inf.te === null || inf.te >= ct)) {
      result.newInfernos.push(inf);
    }
    if (inf.te && inf.te > ct - 0.1 && inf.te <= ct + 0.2) {
      result.endedInfernos.push(inf);
    }
  }

  return result;
}

/**
 * Seek to a specific time within the current round.
 */
export function seekTo(timeSec) {
  if (!currentRound) return;
  const secs = currentRound.round_seconds;
  let idx = 0;
  for (let i = 0; i < secs.length; i++) {
    if (secs[i] >= timeSec) {
      idx = i;
      break;
    }
    idx = i;
  }
  currentSampleIdx = idx;
  currentTime = secs[idx] || timeSec;

  // Reset event tracking up to this point
  firedKills.clear();
  firedDamage.clear();
  const ct = currentRound.ticks[idx] || 0;
  for (const k of (currentRound.events.kills || [])) {
    if (k.t <= ct) firedKills.add(k.t + '_' + k.v);
  }
  for (const d of (currentRound.events.damage || [])) {
    if (d.t <= ct) firedDamage.add(d.t + '_' + d.v + '_' + d.hp);
  }
}

/**
 * Get player state at the current time, with smooth interpolation
 * between the two nearest sample points.
 * @param {number} playerIdx - player index 0-9
 * @param {number} sampleIdx - floor sample index (for non-interpolated fields)
 * @returns {object} player state
 */
export function getPlayerState(playerIdx, sampleIdx) {
  if (!currentRound) return null;
  const p = currentRound.players[playerIdx];
  if (!p) return null;

  const n = currentRound.ticks.length;
  const i = Math.min(Math.max(0, sampleIdx), n - 1);
  const secs = currentRound.round_seconds;

  // Compute interpolation fraction: how far between sample i and i+1
  // based on the continuous currentTime
  let t = 0; // 0 = at sample i,  1 = at sample i+1
  if (i < n - 1 && p.alive[i]) {
    const t0 = secs[i];
    const t1 = secs[i + 1];
    const interval = t1 - t0;
    if (interval > 0.001) {
      t = (currentTime - t0) / interval;
      t = Math.max(0, Math.min(1, t));
    }
  }

  // Helper: lerp array values, with NaN guard
  const l = (arr, i, i2) => {
    if (t <= 0 || i2 >= n) return arr[i] || 0;
    const a = arr[i] || 0;
    const b = arr[i2] || 0;
    return lerp(a, b, t);
  };

  const i2 = Math.min(i + 1, n - 1);
  const periodic = t > 0 && i < n - 1 && p.alive[i];

  return {
    x:          l(p.x, i, i2),
    y:          l(p.y, i, i2),
    z:          l(p.z, i, i2),
    yaw:        periodic ? l(p.yaw, i, i2) : (p.yaw[i] || 0),
    pitch:      periodic ? l(p.pitch, i, i2) : (p.pitch[i] || 0),
    v:          periodic ? l(p.v, i, i2) : (p.v[i] || 0),
    hp:         p.hp[i] || 0,
    armor:      p.armor[i] || 0,
    helmet:     p.helmet[i] || false,
    defuser:    p.defuser[i] || false,
    alive:      p.alive[i] || false,
    weapon:     p.weapon[i],
    inventory:  p.inventory[i] || [],
    flash:      p.flash[i] || 0,
    place:      p.place[i],
    spotted:    p.spotted[i] || [],
    shots:      p.shots[i] || 0,
    footsteps:  p.footsteps[i] || 0,
  };
}

/**
 * Get all player states at the current sample index.
 */
export function getAllPlayerStates(sampleIdx) {
  if (!currentRound) return [];
  const states = [];
  for (let i = 0; i < 10; i++) {
    states.push(getPlayerState(i, sampleIdx));
  }
  return states;
}

/**
 * Get active grenade entities near the current time.
 */
export function getActiveGrenades(fromIdx, toIdx) {
  if (!currentRound) return [];
  const fromTick = currentRound.ticks[fromIdx] || 0;
  const toTick = currentRound.ticks[toIdx] || fromTick;
  return (currentRound.events.grenades || []).filter(
    g => g.t > fromTick && g.t <= toTick
  );
}

/**
 * Convert an absolute game tick to round-relative seconds.
 * Uses the round's tick→time mapping for precision.
 */
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
  // Fallback: linear tick→time
  return secs[0] + (absTick - ticks[0]) / 64.0;
}

/**
 * Group raw grenade events into continuous trajectories.
 * Events are matched by (thrower, type) and split when there's a time gap > 2s.
 * Each point's tick is converted to round-relative seconds for playback.
 * @returns {Array<{thrower: number, type: string, points: Array<{t: number, x: number, y: number, z: number}>}>}
 */
export function buildGrenadeTrajectories() {
  if (!currentRound) return [];
  const raw = currentRound.events.grenades || [];
  if (raw.length === 0) return [];

  // Group by (thrower, type)
  const groups = new Map();
  for (const g of raw) {
    const key = `${g.th}|${g.ty}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(g);
  }

  const trajectories = [];
  for (const [, events] of groups) {
    events.sort((a, b) => a.t - b.t);

    // Multi-target tracking: each "track" follows one grenade entity.
    // New positions are matched to the closest active track within 300 units.
    const tracks = []; // each: {lastTick, lastX, lastY, lastZ, points[]}
    for (const e of events) {
      // Find the closest track within 300 game units
      let bestTrack = null;
      let bestDist = 300;
      for (const tr of tracks) {
        const d = _dist(e, {x: tr.lastX, y: tr.lastY, z: tr.lastZ});
        if (d < bestDist) {
          bestDist = d;
          bestTrack = tr;
        }
      }
      if (bestTrack) {
        bestTrack.lastTick = e.t;
        bestTrack.lastX = e.x; bestTrack.lastY = e.y; bestTrack.lastZ = e.z;
        bestTrack.points.push(e);
      } else {
        // New grenade entity
        tracks.push({
          lastTick: e.t,
          lastX: e.x, lastY: e.y, lastZ: e.z,
          points: [e],
        });
      }
    }

    // Convert tracks with >= 2 points into trajectories
    for (const tr of tracks) {
      if (tr.points.length >= 2) {
        trajectories.push(_makeTraj(tr.points));
      }
    }
  }

  return trajectories;
}

function _dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function _makeTraj(batch) {
  const points = batch.map(b => ({
    t: tickToTime(b.t),
    x: b.x, y: b.y, z: b.z,
  }));

  // Detect when the grenade settled (velocity ≈ 0 for >= 3 consecutive samples)
  let settleTime = points[points.length - 1].t;
  for (let i = points.length - 2; i >= 1; i--) {
    const d = _dist(points[i], points[i + 1]);
    if (d > 5) break;
    settleTime = points[i].t;
  }

  return { thrower: batch[0].th, type: batch[0].ty, points, settleTime };
}

/**
 * Get bomb position with smooth interpolation between samples.
 * @param {number} sampleIdx - floor sample index
 * @returns {Array|null} [x, y, z] in game units, or null
 */
export function getBombPosition(sampleIdx) {
  if (!currentRound) return null;
  const bp = currentRound.bomb_position;
  const n = bp.length;
  const i = Math.min(sampleIdx, n - 1);
  const secs = currentRound.round_seconds;

  // Interpolation fraction
  let t = 0;
  if (i < n - 1 && bp[i] && bp[i + 1]) {
    const t0 = secs[i];
    const t1 = secs[i + 1];
    const interval = t1 - t0;
    if (interval > 0.001) {
      t = (currentTime - t0) / interval;
      t = Math.max(0, Math.min(1, t));
    }
  }

  const a = bp[i];
  if (!a) return null;
  if (t <= 0) return a;

  const b = bp[Math.min(i + 1, n - 1)];
  if (!b) return a;

  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

/**
 * Check if bomb is planted at a sample index.
 */
export function isBombPlanted(sampleIdx) {
  if (!currentRound) return false;
  const i = Math.min(sampleIdx, currentRound.bomb_planted.length - 1);
  return currentRound.bomb_planted[i] || false;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function lerp(a, b, t) {
  return a + (b - a) * Math.min(Math.max(t, 0), 1);
}

/**
 * Get weapon name from index.
 */
export function weaponName(idx) {
  if (idx === undefined || idx === null || idx < 0) return '—';
  return idxToWeapon[idx] || `weapon_${idx}`;
}

/**
 * Get place name from index.
 */
export function placeName(idx) {
  if (idx === undefined || idx === null || idx < 0) return '—';
  return idxToPlace[idx] || `place_${idx}`;
}

/**
 * Get player name.
 */
export function playerName(idx) {
  if (!matchData || idx < 0 || idx >= matchData.players.length) return '?';
  return matchData.players[idx].name;
}

/**
 * Get player team for current round.
 */
export function playerTeam(idx) {
  if (!currentRound || idx < 0 || idx >= currentRound.teams.length) return '?';
  return currentRound.teams[idx];
}

/**
 * Get the team for a player in a specific round.
 */
export function playerTeamInRound(playerIdx, round) {
  if (!round || playerIdx < 0 || playerIdx >= (round.teams?.length || 0)) return '?';
  return round.teams[playerIdx];
}

/**
 * Get the team color CSS class.
 */
export function teamColorClass(team) {
  return team === 'CT' ? 'ct' : 't';
}

/**
 * Check if a grenade type is a specific kind.
 */
export function grenadeCategory(typeStr) {
  if (!typeStr) return 'unknown';
  if (typeStr.includes('Smoke')) return 'smoke';
  if (typeStr.includes('Flash')) return 'flash';
  if (typeStr.includes('HE')) return 'he';
  if (typeStr.includes('Molotov') || typeStr.includes('Incendiary')) return 'fire';
  if (typeStr.includes('Decoy')) return 'decoy';
  return 'unknown';
}

/**
 * Get grenade display color.
 */
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

/**
 * Format time as MM:SS.ms
 */
export function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 10);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
}

/**
 * Get map name from match data.
 */
export function getMapName() {
  return matchData?.map || 'unknown';
}
