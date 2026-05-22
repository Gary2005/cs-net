// 2D top-down radar + per-tick metric panels.
//
// Map-space → pixel-space transform is taken from
// third_party/csgo-2d-demo-viewer-dev/parser/pkg/parser/map.go (Translate + /Scale),
// identical per-map constants. Overview PNGs are 1024×1024 squares; we render at
// 100% of the container and position players in percent coordinates.

const MAP_TRANSFORMS = {
  de_ancient:  { pzero: [-2953, 2164], scale: 5 },
  de_ancient_night: { pzero: [-2953, 2164], scale: 5 },
  de_anubis:   { pzero: [-2796, 3328], scale: 5.22 },
  de_dust2:    { pzero: [-2476, 3239], scale: 4.4 },
  de_inferno:  { pzero: [-2087, 3870], scale: 4.9 },
  de_mirage:   { pzero: [-3230, 1713], scale: 5 },
  de_nuke:     { pzero: [-3453, 2887], scale: 7 },
  de_overpass: { pzero: [-4831, 1781], scale: 5.2 },
  de_train:    { pzero: [-2308, 2078], scale: 4.082077 },
  de_vertigo:  { pzero: [-3168, 1762], scale: 4 },
};

const OVERVIEW_PX = 1024;

// ---- internal state ----

let _selectedPlayer = null;   // { name, team, index }
let _currentTick = null;
let _currentRound = null;

// ---- color helpers ----

function _lerp(a, b, t) {
  return a + (b - a) * t;
}

function _clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function _lerpColor(hexA, hexB, t) {
  const a = parseInt(hexA.slice(1), 16);
  const b = parseInt(hexB.slice(1), 16);
  const r = Math.round(_lerp((a >> 16) & 0xff, (b >> 16) & 0xff, t));
  const g = Math.round(_lerp((a >> 8) & 0xff, (b >> 8) & 0xff, t));
  const bl = Math.round(_lerp(a & 0xff, b & 0xff, t));
  return `rgb(${r},${g},${bl})`;
}

// 0% = #ef4444 (red, unfavored), 50% = #000000 (neutral), 100% = #10b981 (green, favored)
function _duelColor(pct) {
  const v = _clamp(pct, 0, 1);
  if (v <= 0.5) {
    return _lerpColor("#ef4444", "#000000", v / 0.5);
  }
  return _lerpColor("#000000", "#10b981", (v - 0.5) / 0.5);
}

// Next death: high prob = red (dangerous), low prob = green (safe)
function _deathColor(pct) {
  return _duelColor(1.0 - _clamp(pct, 0, 1));
}

// Next kill: high prob = green (good), low prob = red (bad)
function _killColor(pct) {
  return _duelColor(_clamp(pct, 0, 1));
}

// ---- coordinate / map helpers ----

function worldToPercent(mapName, x, y) {
  const xform = MAP_TRANSFORMS[mapName] || MAP_TRANSFORMS.de_dust2;
  const [pzx, pzy] = xform.pzero;
  const tx = (x - pzx) / xform.scale;
  const ty = (pzy - y) / xform.scale;
  return {
    xPercent: (tx / OVERVIEW_PX) * 100,
    yPercent: (ty / OVERVIEW_PX) * 100,
  };
}

function overviewUrl(mapName) {
  const supported = MAP_TRANSFORMS[mapName] ? mapName : null;
  if (!supported) return "/static/overviews/empty.png";
  return `/static/overviews/${supported}.png`;
}

function teamClass(teamNum, team1Players, team2Players, playerName) {
  return teamNum === "CT" ? "ct" : "t";
}

function isFlashed(player) {
  const dur = Number(player.flash_duration || 0);
  const maxA = Number(player.flash_max_alpha || 0);
  return dur > 0.05 && maxA > 0.3;
}

function _grenadeClass(grenadeType) {
  const t = String(grenadeType || "").toLowerCase();
  if (t.includes("smoke")) return "smoke";
  if (t.includes("inferno") || t.includes("molotov")) return "fire";
  if (t.includes("flash")) return "flash";
  if (t.includes("he")) return "he";
  if (t.includes("decoy")) return "decoy";
  return "";
}

// ---- CSS.escape polyfill (rare but safe) ----

function _cssEscape(str) {
  if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(str);
  return String(str).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

// ---- tooltip ----

function _removeTooltip(container) {
  const el = container.querySelector("#radar-tooltip");
  if (el) el.remove();
}

function _showTooltip(container, playerEl, player, tick) {
  _removeTooltip(container);
  if (!playerEl || !tick) return;

  const rect = playerEl.getBoundingClientRect();
  const wrapRect = container.getBoundingClientRect();

  const tooltip = document.createElement("div");
  tooltip.className = "radar-tooltip";
  tooltip.id = "radar-tooltip";

  const deathProb = Number((tick.next_death || [])[player.index] || 0);
  const killProb  = Number((tick.next_kill || [])[player.index] || 0);

  const deathColor = _deathColor(deathProb);
  const killColor  = _killColor(killProb);

  let inner = `<div class="tooltip-name">${player.name}</div>`;
  inner += `<div class="tooltip-row"><span class="tooltip-label">下一阵亡概率</span><span style="color:${deathColor}; font-weight:600;">${(deathProb * 100).toFixed(1)}%</span></div>`;
  inner += `<div class="tooltip-row"><span class="tooltip-label">下一击杀概率</span><span style="color:${killColor}; font-weight:600;">${(killProb * 100).toFixed(1)}%</span></div>`;
  inner += `<div class="tooltip-hint">已选中 · 连线=单挑胜率</div>`;

  tooltip.innerHTML = inner;

  // Position to upper-right of the player marker, clamped to container.
  let left = rect.left - wrapRect.left + 22;
  let top  = rect.top  - wrapRect.top - 10;

  const maxLeft = wrapRect.width - 210;
  const maxTop  = wrapRect.height - 80;
  left = Math.max(4, Math.min(left, maxLeft));
  top  = Math.max(4, Math.min(top, maxTop));

  tooltip.style.left = left + "px";
  tooltip.style.top  = top  + "px";
  container.appendChild(tooltip);
}

// ---- duel lines (SVG overlay) ----

function _renderDuelLines(container) {
  // Ensure SVG overlay exists
  let svg = container.querySelector("svg.radar-svg-overlay");
  if (!svg) {
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "radar-svg-overlay");
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("preserveAspectRatio", "none");
    container.appendChild(svg);
  }
  svg.innerHTML = "";

  const tick = _currentTick;
  const round = _currentRound;
  const sel = _selectedPlayer;
  if (!tick || !round || !sel) return;

  const players = tick.players_info || [];
  const duel = tick.duel;
  if (!duel) return;

  const selData = players[sel.index];
  if (!selData || !selData.is_alive) return;

  const selTeamSide = selData.team_num;  // "CT" or "T"

  // Gather alive opposite-team players
  const targets = [];
  players.forEach((p, i) => {
    if (!p.is_alive) return;
    if (p.name === sel.name) return;
    if (p.team_num === selTeamSide) return;  // same team, no duel data
    targets.push({ ...p, index: i });
  });

  if (targets.length === 0) return;

  const mapName = round.map_name || "de_dust2";
  const selX = Number(selData.X);
  const selY = Number(selData.Y);
  const selPct = worldToPercent(mapName, selX, selY);

  targets.forEach((tgt) => {
    const tgtX = Number(tgt.X);
    const tgtY = Number(tgt.Y);
    const tgtPct = worldToPercent(mapName, tgtX, tgtY);

    // Duel probability from selected player's perspective
    let prob;
    if (selTeamSide === "CT") {
      prob = Number(duel[sel.index]?.[tgt.index] ?? 0.5);
    } else {
      // T vs CT: prob = 1 - duel[CT][T_sel] where CT is tgt
      prob = 1.0 - Number(duel[tgt.index]?.[sel.index] ?? 0.5);
    }

    const color = _duelColor(prob);
    const midX = (selPct.xPercent + tgtPct.xPercent) / 2;
    const midY = (selPct.yPercent + tgtPct.yPercent) / 2;

    // Line
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(selPct.xPercent));
    line.setAttribute("y1", String(selPct.yPercent));
    line.setAttribute("x2", String(tgtPct.xPercent));
    line.setAttribute("y2", String(tgtPct.yPercent));
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "0.8");
    line.setAttribute("stroke-opacity", "0.7");
    svg.appendChild(line);

    // Text label at midpoint (background for readability)
    const bg = document.createElementNS("http://www.w3.org/2000/svg", "text");
    bg.setAttribute("x", String(midX));
    bg.setAttribute("y", String(midY));
    bg.setAttribute("fill", "rgba(0,0,0,0.6)");
    bg.setAttribute("font-size", "3.5");
    bg.setAttribute("font-family", "IBM Plex Mono, monospace");
    bg.setAttribute("text-anchor", "middle");
    bg.setAttribute("dominant-baseline", "central");
    bg.setAttribute("font-weight", "700");
    bg.setAttribute("stroke", "rgba(0,0,0,0.6)");
    bg.setAttribute("stroke-width", "0.6");
    bg.setAttribute("paint-order", "stroke");
    bg.textContent = `${(prob * 100).toFixed(0)}%`;
    svg.appendChild(bg);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", String(midX));
    text.setAttribute("y", String(midY));
    text.setAttribute("fill", color);
    text.setAttribute("font-size", "3.5");
    text.setAttribute("font-family", "IBM Plex Mono, monospace");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dominant-baseline", "central");
    text.setAttribute("font-weight", "600");
    text.textContent = `${(prob * 100).toFixed(0)}%`;
    svg.appendChild(text);
  });
}

// ---- selection management ----

function _clearSelection(container) {
  _selectedPlayer = null;
  _removeTooltip(container);
  const svg = container.querySelector("svg.radar-svg-overlay");
  if (svg) svg.innerHTML = "";
  // Remove .selected class from player dots
  container.querySelectorAll(".radar-player.selected").forEach((el) => {
    el.classList.remove("selected");
  });
}

function _updateRadarDisplay(container) {
  if (!container) return;
  _removeTooltip(container);

  if (_selectedPlayer && _currentTick) {
    const playerEl = container.querySelector(
      `.radar-player[data-player-name="${_cssEscape(_selectedPlayer.name)}"]`
    );
    if (playerEl) {
      playerEl.classList.add("selected");
      const players = _currentTick.players_info || [];
      const playerData = players.find((p) => p.name === _selectedPlayer.name);
      if (playerData) {
        _showTooltip(container, playerEl, {
          name: playerData.name,
          index: players.indexOf(playerData),
        }, _currentTick);
      }
    }
    _renderDuelLines(container);
  }
}

// ---- interaction setup ----

function _setupRadarInteraction(container) {
  if (container.dataset.radarSetup === "1") return;
  container.dataset.radarSetup = "1";

  container.addEventListener("click", (e) => {
    const playerEl = e.target.closest(".radar-player");
    if (!playerEl) {
      _clearSelection(container);
      return;
    }
    const name = playerEl.dataset.playerName;
    const team = playerEl.dataset.playerTeam;
    const idx = parseInt(playerEl.dataset.playerIndex, 10);
    if (isNaN(idx)) return;

    // If clicking the already-selected player, deselect
    if (_selectedPlayer && _selectedPlayer.name === name) {
      _clearSelection(container);
      return;
    }

    _selectedPlayer = { name, team, index: idx };
    _updateRadarDisplay(container);
  });
}

// ---- main render ----

function renderRadar(container, mapName, tick, round) {
  if (!container) return;
  const tm = MAP_TRANSFORMS[mapName];
  container.style.backgroundImage = `url('${overviewUrl(mapName)}')`;
  container.classList.toggle("unsupported-map", !tm);

  _currentTick = tick;
  _currentRound = round;
  _setupRadarInteraction(container);

  if (!tick || !tm) {
    container.innerHTML = "";
    return;
  }

  const team1 = round?.team1_players || [];
  const team2 = round?.team2_players || [];
  const players = tick.players_info || [];

  const nodes = players.map((p, idx) => {
    const teamCls = teamClass(p.team_num, team1, team2, p.name);
    if (p.X == null || p.Y == null) return "";
    const { xPercent, yPercent } = worldToPercent(mapName, Number(p.X), Number(p.Y));
    const dead = !p.is_alive;
    const flashed = !dead && isFlashed(p);
    const hp = Math.max(0, Math.min(100, Number(p.health || 0)));
    const rot = Number(p.yaw || 0);
    const classes = [
      "radar-player",
      teamCls,
      dead ? "dead" : "",
      flashed ? "flashed" : "",
    ].filter(Boolean).join(" ");

    const arrow = dead
      ? ""
      : `<div class="radar-arrow" style="transform: rotate(${-rot + 90}deg);"></div>`;

    const nameText = `${idx + 1}. ${p.name || "?"}`;
    const safeName = p.name ? p.name.replace(/"/g, "&quot;") : "?";

    return `
      <div class="${classes}"
           style="left:${xPercent}%; top:${yPercent}%; --hp:${hp}%;"
           data-player-name="${safeName}"
           data-player-team="${teamCls}"
           data-player-index="${idx}">
        ${arrow}
        <div class="radar-nametag">${nameText}</div>
      </div>`;
  }).join("");

  let bomb = "";
  const bp = tick.bomb_position;
  if (bp && Array.isArray(bp) && bp.length >= 2) {
    const { xPercent, yPercent } = worldToPercent(mapName, Number(bp[0]), Number(bp[1]));
    const planted = !!tick.is_bomb_planted;
    bomb = `<div class="radar-bomb ${planted ? "planted" : "dropped"}" style="left:${xPercent}%; top:${yPercent}%;"></div>`;
  }

  // Utility projectiles (smoke, inferno, flying grenades)
  let utilityHtml = "";
  const projectiles = tick.projectiles || [];
  const entityGrenades = tick.entity_grenades || [];

  // Approximate smoke radius in game units → percent of overview
  const SMOKE_RADIUS_UNITS = 150;
  const xform = MAP_TRANSFORMS[mapName] || MAP_TRANSFORMS.de_dust2;
  const smokeRadiusPct = (SMOKE_RADIUS_UNITS / xform.scale) / OVERVIEW_PX * 100;

  projectiles.forEach((p) => {
    const pos = p.position;
    if (!pos || (Array.isArray(pos) && pos.length < 2)) return;
    const x = Array.isArray(pos) ? Number(pos[0]) : Number(pos[0]);
    const y = Array.isArray(pos) ? Number(pos[1]) : Number(pos[1]);
    if (isNaN(x) || isNaN(y)) return;
    const { xPercent, yPercent } = worldToPercent(mapName, x, y);
    const type = _grenadeClass(p.type);
    if (!type) return;
    const dur = Number(p.duration || 0);

    let extraStyle = "";
    if (type === "smoke") {
      // Expand 0→full over 3s, hold 3→15s, fade 15→18s
      const expandT = Math.min(dur / 3.0, 1.0);
      const sizePct = smokeRadiusPct * 2 * (0.2 + 0.8 * expandT);
      const opacity = dur > 15 ? Math.max(0.15, 1.0 - (dur - 15) / 3.0) : 1;
      extraStyle = `width:${sizePct}%; height:${sizePct}%; margin-left:${-sizePct/2}%; margin-top:${-sizePct/2}%; opacity:${opacity.toFixed(2)};`;
    } else if (type === "fire") {
      // Fire flickers and fades near end of 7s life
      const opacity = dur > 5 ? Math.max(0.2, 1.0 - (dur - 5) / 2.0) : 1;
      const fireSize = smokeRadiusPct * 1.3;
      extraStyle = `width:${fireSize}%; height:${fireSize}%; margin-left:${-fireSize/2}%; margin-top:${-fireSize/2}%; opacity:${opacity.toFixed(2)};`;
    }
    utilityHtml += `<div class="radar-util ${type}" style="left:${xPercent}%; top:${yPercent}%; ${extraStyle}" title="${p.type} · ${dur.toFixed(1)}s"></div>`;
  });

  entityGrenades.forEach((g) => {
    const pos = g.position;
    if (!pos || (Array.isArray(pos) && pos.length < 2)) return;
    const x = Array.isArray(pos) ? Number(pos[0]) : Number(pos[0]);
    const y = Array.isArray(pos) ? Number(pos[1]) : Number(pos[1]);
    if (isNaN(x) || isNaN(y)) return;
    const { xPercent, yPercent } = worldToPercent(mapName, x, y);
    const type = _grenadeClass(g.type);
    if (!type) return;
    utilityHtml += `<div class="radar-nade ${type}" style="left:${xPercent}%; top:${yPercent}%;" title="${g.type} · ${g.name || ''}"></div>`;
  });

  container.innerHTML = nodes + utilityHtml + bomb;

  // Re-apply selection state after re-render
  _updateRadarDisplay(container);
}

// ----- metric panels (alive / kill / death / duel) -----

function pctBar(value, team) {
  const v = Math.max(0, Math.min(1, Number(value || 0)));
  const cls = team === "ct" ? "ct" : team === "t" ? "t" : "neutral";
  return `
    <div class="metric-bar ${cls}">
      <div class="metric-bar-fill" style="width:${(v * 100).toFixed(1)}%"></div>
      <div class="metric-bar-val">${(v * 100).toFixed(1)}%</div>
    </div>`;
}

// Render a per-player column (alive / next-kill / next-death).
function renderPlayerProbColumn(container, players, probs, round, labelHeader) {
  if (!container) return;
  const rows = players.map((p, i) => {
    const team = teamClass(p.team_num, round.team1_players, round.team2_players, p.name);
    const prob = Number((probs || [])[i] || 0);
    const alive = !!p.is_alive;
    const dim = alive ? "" : "dim";
    return `
      <tr class="${dim}">
        <td class="metric-cell-name"><span class="team-dot ${team}"></span>${p.name || "?"}</td>
        <td>${alive ? pctBar(prob, team) : `<span class="metric-dead">DEAD</span>`}</td>
      </tr>`;
  }).join("");
  container.innerHTML = `
    <table class="metric-table">
      <thead><tr><th>Player</th><th>${labelHeader}</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderAlivePanel(container, tick, round) {
  if (!container || !tick) { if (container) container.innerHTML = ""; return; }
  renderPlayerProbColumn(container, tick.players_info || [], tick.alive_pred, round, "Alive in 5s");
}

function renderNextKillPanel(container, tick, round) {
  if (!container || !tick) { if (container) container.innerHTML = ""; return; }
  const players = tick.players_info || [];
  const probs = tick.next_kill || [];
  renderPlayerProbColumn(container, players, probs, round, "P(next kill is by...)");
  const noKill = Number((probs || [])[10] || 0);
  if (noKill > 0) {
    container.insertAdjacentHTML("beforeend",
      `<div class="metric-nokill">P(no kill): <strong>${(noKill * 100).toFixed(1)}%</strong></div>`);
  }
}

function renderNextDeathPanel(container, tick, round) {
  if (!container || !tick) { if (container) container.innerHTML = ""; return; }
  const players = tick.players_info || [];
  const probs = tick.next_death || [];
  renderPlayerProbColumn(container, players, probs, round, "P(next death is...)");
  const noDeath = Number((probs || [])[10] || 0);
  if (noDeath > 0) {
    container.insertAdjacentHTML("beforeend",
      `<div class="metric-nokill">P(no death): <strong>${(noDeath * 100).toFixed(1)}%</strong></div>`);
  }
}

function duelCellColor(p) {
  const v = Math.max(0, Math.min(1, Number(p || 0)));
  if (v >= 0.5) {
    const t = (v - 0.5) * 2;
    const r = Math.round(255 - 161 * t);
    const g = Math.round(255 - 55 * t);
    const b = Math.round(255 - 0 * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
  const t = (0.5 - v) * 2;
  const r = Math.round(255 - 0 * t);
  const g = Math.round(255 - 117 * t);
  const b = Math.round(255 - 199 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function renderDuelPanel(container, tick, round) {
  if (!container || !tick) { if (container) container.innerHTML = ""; return; }
  const players = tick.players_info || [];
  const duel = tick.duel;
  if (!duel) {
    container.innerHTML = `<p class="metric-empty">Duel matrix unavailable</p>`;
    return;
  }
  const ctIdxs = players.map((p, i) => p.team_num === "CT" ? i : -1).filter(i => i >= 0);
  const tIdxs  = players.map((p, i) => p.team_num === "T"  ? i : -1).filter(i => i >= 0);

  const header = `<th class="duel-corner">CT \\ T</th>` +
    tIdxs.map(j => {
      const p = players[j];
      const alive = p.is_alive;
      return `<th class="${alive ? "" : "duel-dead"}">${p.name || "?"}</th>`;
    }).join("");

  const rows = ctIdxs.map(i => {
    const pi = players[i];
    const aliveI = pi.is_alive;
    const cells = tIdxs.map(j => {
      const pj = players[j];
      const aliveJ = pj.is_alive;
      if (!aliveI || !aliveJ) return `<td class="duel-dead">-</td>`;
      const v = Number(duel[i]?.[j] ?? 0.5);
      const color = duelCellColor(v);
      return `<td class="duel-cell" style="background:${color}">${(v * 100).toFixed(0)}%</td>`;
    }).join("");
    return `<tr><th class="${aliveI ? "" : "duel-dead"}">${pi.name || "?"}</th>${cells}</tr>`;
  }).join("");

  container.innerHTML = `
    <div class="duel-legend">CT win prob vs each T opponent (rows=CT, cols=T)</div>
    <table class="duel-table">
      <thead><tr>${header}</tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// Helper: pick tick by round_seconds.
function pickTick(round, roundSeconds) {
  const ticks = round?.ticks || [];
  if (ticks.length === 0) return null;
  let best = ticks[0];
  let bestGap = Math.abs(Number(best.round_seconds || 0) - roundSeconds);
  for (let i = 1; i < ticks.length; i++) {
    const gap = Math.abs(Number(ticks[i].round_seconds || 0) - roundSeconds);
    if (gap < bestGap) { bestGap = gap; best = ticks[i]; }
  }
  return best;
}

window.CSNetRadar = {
  MAP_TRANSFORMS,
  renderRadar,
  renderAlivePanel,
  renderNextKillPanel,
  renderNextDeathPanel,
  renderDuelPanel,
  pickTick,
  clearSelection: function(container) {
    _clearSelection(container || document.getElementById("radar-wrapper"));
  },
};
