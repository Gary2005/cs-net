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
  if ((team1Players || []).includes(playerName)) return "team1";
  if ((team2Players || []).includes(playerName)) return "team2";
  return teamNum === "CT" ? "team1" : "team2";
}

function isFlashed(player) {
  const dur = Number(player.flash_duration || 0);
  const maxA = Number(player.flash_max_alpha || 0);
  return dur > 0.05 && maxA > 0.3;
}

// Render the radar for a single tick.
// container: the .radar-wrapper element.
// mapName: e.g., "de_dust2".
// tick: one element from round.ticks (has players_info, bomb_position, is_bomb_planted).
// round: current dashboard round (for team1/team2 membership).
function renderRadar(container, mapName, tick, round) {
  if (!container) return;
  const tm = MAP_TRANSFORMS[mapName];
  container.style.backgroundImage = `url('${overviewUrl(mapName)}')`;
  container.classList.toggle("unsupported-map", !tm);

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

    return `
      <div class="${classes}" style="left:${xPercent}%; top:${yPercent}%; --hp:${hp}%;">
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

  container.innerHTML = nodes + bomb;
}

// ----- metric panels (alive / kill / death / duel) -----

function pctBar(value, team) {
  const v = Math.max(0, Math.min(1, Number(value || 0)));
  const cls = team === "team1" ? "team1" : team === "team2" ? "team2" : "neutral";
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
  // 0.5 neutral white; >0.5 blue (CT wins); <0.5 orange (T wins)
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
};
