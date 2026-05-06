const state = {
  config: {},
  map: "",
  samples: [],
  selected: -1,
  draft: [],
  image: new Image(),
  layer: "upper",
  showPlace: "",
  annotationLang: "cn",
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const tip = document.getElementById("tip");
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[ch]);
}

function setStatus(text) {
  const status = $("status");
  if (status) status.textContent = text || "";
}

function mapConfig() {
  state.config.maps ||= {};
  state.config.maps[state.map] ||= { nearest_threshold: 300, polygons_cn: [], polygons_en: [] };
  const cfg = state.config.maps[state.map];
  cfg.polygons_cn ||= [];
  cfg.polygons_en ||= [];
  if (Array.isArray(cfg.polygons) && cfg.polygons_cn.length === 0) {
    cfg.polygons_cn = cfg.polygons;
    delete cfg.polygons;
  }
  return cfg;
}

function transform() {
  return mapConfig().transform || null;
}

function hasLowerOverview() {
  return Boolean(mapConfig().lower_overview);
}

function currentOverview() {
  const cfg = mapConfig();
  const path = state.layer === "lower" && cfg.lower_overview ? cfg.lower_overview : cfg.overview;
  return path ? path.split(/[\\/]/).pop() : "empty.png";
}

function setLayer(layer) {
  state.layer = layer === "lower" && hasLowerOverview() ? "lower" : "upper";
  const layerBadge = $("layer-badge");
  if (layerBadge) layerBadge.textContent = `layer: ${state.layer}`;
  const toggle = $("layer-toggle");
  if (toggle) {
    toggle.hidden = !hasLowerOverview();
    toggle.textContent = state.layer === "lower" ? "Lower layer" : "Upper layer";
    toggle.classList.toggle("lower", state.layer === "lower");
  }
}

function loadOverview() {
  const overview = currentOverview();
  state.image = new Image();
  state.image.onload = render;
  state.image.onerror = () => {
    setStatus(`overview not found: ${overview}`);
    render();
  };
  state.image.src = `/overviews/${overview}`;
}

function toggleLayer() {
  if (!hasLowerOverview()) return;
  setLayer(state.layer === "lower" ? "upper" : "lower");
  loadOverview();
  render();
}

function syncLayerControls() {
  setLayer(state.layer);
}

function worldToCanvas(x, y) {
  const tr = transform();
  if (!tr || !Array.isArray(tr.pzero) || tr.pzero.length < 2 || !Number.isFinite(Number(tr.scale))) {
    return null;
  }
  const [pzx, pzy] = tr.pzero.map(Number);
  const wx = Number(x);
  const wy = Number(y);
  if (!Number.isFinite(wx) || !Number.isFinite(wy)) return null;
  return { x: (wx - pzx) / Number(tr.scale), y: (pzy - wy) / Number(tr.scale) };
}

function canvasToWorld(x, y) {
  const tr = transform();
  if (!tr || !Array.isArray(tr.pzero) || tr.pzero.length < 2 || !Number.isFinite(Number(tr.scale))) {
    return null;
  }
  const [pzx, pzy] = tr.pzero.map(Number);
  return [
    Number((pzx + x * Number(tr.scale)).toFixed(2)),
    Number((pzy - y * Number(tr.scale)).toFixed(2)),
  ];
}

function pointVisible(pt) {
  return pt && pt.x >= -12 && pt.x <= 1036 && pt.y >= -12 && pt.y <= 1036;
}

function polygons() {
  return mapConfig()[state.annotationLang === "en" ? "polygons_en" : "polygons_cn"];
}

function drawPoly(points, color, fill) {
  if (!points || points.length < 2) return;
  ctx.beginPath();
  let moved = false;
  for (const p of points) {
    const pt = worldToCanvas(p[0], p[1]);
    if (!pointVisible(pt)) continue;
    if (!moved) {
      ctx.moveTo(pt.x, pt.y);
      moved = true;
    } else {
      ctx.lineTo(pt.x, pt.y);
    }
  }
  if (!moved) return;
  if (points.length >= 3) ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  if (points.length >= 3) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
}

function render() {
  ctx.clearRect(0, 0, 1024, 1024);
  if (state.image.complete && state.image.naturalWidth > 1) {
    ctx.drawImage(state.image, 0, 0, 1024, 1024);
  } else {
    ctx.fillStyle = "#0b1015";
    ctx.fillRect(0, 0, 1024, 1024);
  }

  const place = state.showPlace;
  let drawn = 0;
  for (const s of state.samples) {
    if (place && s.last_place_name !== place) continue;
    const pt = worldToCanvas(s.x, s.y);
    if (!pointVisible(pt)) continue;
    drawn += 1;
    ctx.fillStyle = place ? "rgba(255, 218, 105, .82)" : "rgba(105, 220, 255, .34)";
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, place ? 3.5 : 2.25, 0, Math.PI * 2);
    ctx.fill();
  }
  $("drawn-badge").textContent = `drawn: ${drawn}`;

  polygons().forEach((p, i) => {
    drawPoly(
      p.points,
      i === state.selected ? "#5daaff" : "#ffd166",
      i === state.selected ? "rgba(93,170,255,.18)" : "rgba(255,209,102,.09)",
    );
  });
  drawPoly(state.draft, "#73f7a8", "rgba(115,247,168,.16)");
  renderLists();
}

function renderLists() {
  const langLabel = state.annotationLang === "en" ? "English" : "中文";
  $("poly-name-label").childNodes[0].nodeValue = state.annotationLang === "en" ? "English callout " : "中文点位名 ";
  $("polygons").innerHTML = polygons()
    .map((p, i) => `<div class="item ${i === state.selected ? "active" : ""}" data-i="${i}"><span>${escapeHtml(p.name || p.name_cn || p.name_en || "unnamed")}</span><span class="mono">${langLabel} · ${(p.points || []).length}</span></div>`)
    .join("");
  document.querySelectorAll("#polygons .item").forEach((n) => {
    n.onclick = () => selectPolygon(Number(n.dataset.i));
  });
}

function renderPlaces(counts) {
  $("places").innerHTML = (counts || [])
    .map(([name, count]) => `<div class="item" data-place="${escapeHtml(name)}"><span>${escapeHtml(name)}</span><span class="mono">${count}</span></div>`)
    .join("");
  document.querySelectorAll("#places .item").forEach((n) => {
    n.onclick = () => {
      state.showPlace = state.showPlace === n.dataset.place ? "" : n.dataset.place;
      render();
    };
  });
}

function refreshMapOptions() {
  const maps = Object.keys(state.config.maps || {}).sort();
  $("map").innerHTML = maps.map((m) => `<option value="${m}">${m}</option>`).join("");
}

function setMap(map) {
  state.map = map;
  $("map").value = map;
  $("map-badge").textContent = `map: ${map}`;
  setLayer("upper");
  state.selected = -1;
  state.draft = [];
  clearForm();

  const cfg = mapConfig();
  loadOverview();
  syncLayerControls();

  if (!cfg.transform) {
    setStatus("missing overview transform; samples loaded but cannot be projected");
  } else if (!state.samples.length) {
    setStatus(hasLowerOverview() ? "use the layer button to toggle upper/lower overview" : "");
  }
  render();
}

function clearForm() {
  $("poly-name").value = "";
  $("z-min").value = "";
  $("z-max").value = "";
  $("points").value = "[]";
}

function selectPolygon(i) {
  const p = polygons()[i];
  if (!p) return;
  state.selected = i;
  state.draft = [];
  $("poly-name").value = p.name || p.name_cn || p.name_en || "";
  $("z-min").value = p.z_min ?? "";
  $("z-max").value = p.z_max ?? "";
  $("points").value = JSON.stringify(p.points || [], null, 2);
  render();
}

canvas.addEventListener("click", (event) => {
  if (!transform()) return;
  const rect = canvas.getBoundingClientRect();
  const cx = ((event.clientX - rect.left) / rect.width) * 1024;
  const cy = ((event.clientY - rect.top) / rect.height) * 1024;
  const pt = canvasToWorld(cx, cy);
  if (!pt) return;
  if (event.shiftKey && state.draft.length >= 1) {
    const first = state.draft[0];
    state.draft = [first, [pt[0], first[1]], pt, [first[0], pt[1]]];
  } else {
    state.draft.push(pt);
  }
  $("points").value = JSON.stringify(state.draft, null, 2);
  render();
});

canvas.addEventListener("mousemove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const cx = ((event.clientX - rect.left) / rect.width) * 1024;
  const cy = ((event.clientY - rect.top) / rect.height) * 1024;
  let best = null;
  let bestD = 9999;
  for (const s of state.samples) {
    const pt = worldToCanvas(s.x, s.y);
    if (!pointVisible(pt)) continue;
    const d = Math.hypot(pt.x - cx, pt.y - cy);
    if (d < bestD) {
      bestD = d;
      best = s;
    }
  }
  if (best && bestD < 9) {
    tip.style.display = "block";
    tip.style.left = `${event.clientX + 12}px`;
    tip.style.top = `${event.clientY + 12}px`;
    tip.textContent = `${best.last_place_name || "(no last_place_name)"}\n${best.name}\nX ${best.x.toFixed(1)}  Y ${best.y.toFixed(1)}  Z ${best.z.toFixed(1)}`;
  } else {
    tip.style.display = "none";
  }
});

if ($("layer-toggle")) {
  $("layer-toggle").onclick = toggleLayer;
}

$("map").onchange = (event) => setMap(event.target.value);
$("annotation-lang").onchange = (event) => {
  state.annotationLang = event.target.value === "en" ? "en" : "cn";
  state.selected = -1;
  state.draft = [];
  clearForm();
  render();
};
$("new").onclick = () => {
  state.selected = -1;
  state.draft = [];
  clearForm();
  render();
};
$("clear").onclick = () => {
  state.draft = [];
  $("points").value = "[]";
  render();
};
$("delete").onclick = () => {
  if (state.selected >= 0) {
    polygons().splice(state.selected, 1);
    state.selected = -1;
    clearForm();
    render();
  }
};
$("apply").onclick = () => {
  let points = state.draft;
  try {
    const parsed = JSON.parse($("points").value || "[]");
    if (Array.isArray(parsed)) points = parsed;
  } catch {}
  const fallbackName = state.annotationLang === "en" ? "unnamed callout" : "未命名点位";
  const poly = { name: $("poly-name").value.trim() || fallbackName, points };
  if ($("z-min").value !== "") poly.z_min = Number($("z-min").value);
  if ($("z-max").value !== "") poly.z_max = Number($("z-max").value);
  if (state.selected >= 0) polygons()[state.selected] = poly;
  else {
    polygons().push(poly);
    state.selected = polygons().length - 1;
  }
  state.draft = [];
  selectPolygon(state.selected);
};
$("save").onclick = async () => {
  const resp = await fetch(`/api/config/${encodeURIComponent(state.map)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(mapConfig()),
  });
  const data = await resp.json();
  alert(data.ok ? `Saved ${data.path}` : data.error);
};
$("load-demo").onclick = async () => {
  const file = $("demo").files[0];
  if (!file) return alert("Choose a .dem file first");
  setStatus("parsing demo...");
  const form = new FormData();
  form.append("demo", file);
  const resp = await fetch("/api/demo", { method: "POST", body: form });
  const data = await resp.json();
  if (!resp.ok || data.error) {
    setStatus(data.error || "failed to parse demo");
    return;
  }
  state.samples = data.samples || [];
  $("sample-badge").textContent = `samples: ${data.sample_count}`;
  if (!state.config.maps[data.map_name]) {
    state.config.maps[data.map_name] = { nearest_threshold: 300, polygons_cn: [], polygons_en: [] };
    refreshMapOptions();
  }
  setMap(data.map_name);
  renderPlaces(data.last_place_counts);
  setStatus(`demo loaded; skipped invalid rows: ${data.skipped_samples || 0}`);
};

async function init() {
  state.config = await fetch("/api/config").then((r) => r.json());
  refreshMapOptions();
  const maps = Object.keys(state.config.maps || {}).sort();
  setMap(maps.includes("de_mirage") ? "de_mirage" : maps[0] || "unknown_map");
}

init();
