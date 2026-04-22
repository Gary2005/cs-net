const state = {
  analysisId: null,
  dashboard: null,
  selectedRoundIndex: 0,
  chart: null,
  runId: null,
  advancedExpanded: false,
};

const refs = {
  analyzeForm: document.getElementById("analyze-form"),
  analyzeBtn: document.getElementById("analyze-btn"),
  appLanguage: document.getElementById("app-language"),
  modelPath: document.getElementById("model-path"),
  device: document.getElementById("device"),
  batchSize: document.getElementById("batch-size"),
  statusLog: document.getElementById("status-log"),
  roundView: document.getElementById("round-view"),
  roundSummaryView: document.getElementById("round-summary-view"),
  overallSummaryView: document.getElementById("overall-summary-view"),
  llmView: document.getElementById("llm-view"),
  roundTabs: document.getElementById("round-tabs"),
  chart: document.getElementById("chart"),
  hoverPlayerStats: document.getElementById("hover-player-stats"),
  roundSummaryTable: document.getElementById("round-summary-table"),
  overallSummaryTable: document.getElementById("overall-summary-table"),
  matchBadge: document.getElementById("match-badge"),
  llmBtn: document.getElementById("llm-btn"),
  llmOut: document.getElementById("llm-output"),
  llmApiKey: document.getElementById("llm-api-key"),
  llmModel: document.getElementById("llm-model"),
  llmBaseUrl: document.getElementById("llm-base-url"),
  llmTemperature: document.getElementById("llm-temperature"),
  viewerLaunchView: document.getElementById("viewer-launch-view"),
  viewerOpen: document.getElementById("viewer-open"),
  advancedView: document.getElementById("advanced-metrics-view"),
  advancedBody: document.getElementById("advanced-metrics-body"),
  advancedToggle: document.getElementById("advanced-toggle"),
};

const USER_PREFS_KEY = "csnet.user.preferences.v1";

const I18N = {
  zh: {
    ui_language: "界面语言 (language)",
    subtitle: "上传 DEM 后自动计算每回合胜率、击杀影响和玩家贡献，并生成可交互复盘与 LLM 解读。",
    section_run: "1. 运行分析",
    dem_file: "DEM 文件",
    model_path: "模型根目录",
    model_path_placeholder: "模型根目录（含 alive / nxt_kill / nxt_death / win_rate / duel 子目录）",
    device: "设备",
    batch_size: "Batch Size",
    analyze_hint: "进度说明: 解析 demo 一般需要 30~60 秒，运行模型通常不到 1 分钟。分析中会显示当前阶段和正在处理的回合。",
    analyze_btn: "开始分析",
    status_waiting: "等待上传 DEM...",
    section_round_trend: "2. 回合走势",
    section_viewer: "3. 2D 回放器",
    viewer_open: "在新标签页打开 2D 回放",
    viewer_hint: "在新标签页打开同一 demo 的 2D 回放（基于 third_party/csgo-2d-demo-viewer-dev，包含烟雾/闪光/手雷弹道）。",
    section_tick_state: "4. 回合实时态势",
    section_round_summary: "5. 当前回合最终贡献",
    section_overall_summary: "6. 全场平均贡献",
    section_advanced: "7. 高级指标",
    advanced_toggle: "展开/收起",
    advanced_kill_rank: "击杀难度 / 影响排行",
    advanced_player_table: "玩家高级指标",
    advanced_kill_rank_hint: "按 |swing| 降序；难度 difficulty>1 表示在被预测劣势下完成击杀。",
    col_attacker: "击杀者",
    col_victim: "被杀",
    col_swing: "Swing",
    col_difficulty: "难度",
    col_round: "回合",
    col_second: "时刻(s)",
    col_avg_kill_opp: "平均击杀机会",
    col_avg_death_opp: "平均阵亡威胁",
    col_avg_survive: "平均存活率",
    col_hard_win: "困难对枪胜率",
    col_easy_win: "简单对枪胜率",
    col_highlight: "高光回合占比",
    term_swing_help: "Swing：该次击杀对局势（胜率曲线）的影响幅度。绝对值越大，代表越关键。",
    term_difficulty_help: "难度：对枪难度系数。>1 通常表示在模型看来更难的对枪，<1 表示相对更容易。",
    term_avg_kill_opp_help: "平均击杀机会：该玩家在各个 tick 成为“下一击杀者”的平均概率。",
    term_avg_death_opp_help: "平均阵亡威胁：该玩家在各个 tick 成为“下一阵亡者”的平均概率。",
    term_avg_survive_help: "平均存活率：该玩家在各个 tick 下，未来短时间内保持存活的平均概率。",
    term_hard_win_help: "困难对枪胜率：在高难度对枪（difficulty>1）中的获胜比例。",
    term_easy_win_help: "简单对枪胜率：在低难度对枪（0<difficulty<1）中的获胜比例。",
    term_highlight_help: "高光回合占比：该玩家在回合总结中 total_contribution 达到阈值的回合占比。当前阈值为 total_contribution >= 0.20（20%）。",
    term_avg_kill_help: "平均 Kill：该玩家跨回合的平均击杀贡献（kill_contribution）。",
    term_avg_tactical_help: "平均 Tactical：该玩家跨回合的平均战术贡献（tactical_contribution）。",
    term_avg_total_help: "平均 Total：该玩家跨回合的平均总贡献（kill+tactical）。",
    section_llm: "8. 语言模型总结",
    api_key: "API Key",
    model_name: "模型名",
    model_name_placeholder: "gpt-4.1 / deepseek-chat / qwen-max",
    base_url: "Base URL (OpenAI 兼容)",
    temperature: "Temperature",
    llm_btn: "生成 AI 复盘",
    llm_empty: "暂无总结",
    no_data: "暂无数据",
    team1: "team1",
    team2: "team2",
    tie: "平局",
    unknown: "未知",
    col_player: "玩家",
    col_team: "阵营",
    col_kill: "Kill",
    col_tactical: "Tactical",
    col_total: "Total",
    col_avg_kill: "平均 Kill",
    col_avg_tactical: "平均 Tactical",
    col_avg_total: "平均 Total",
    col_rounds: "回合数",
    match_badge: "比分 {team1Label} {team1} : {team2} {team2Label} · 胜方 {winner}",
    round_tab: "R{round} · {winner}",
    chart_round: "回合",
    chart_team1_wr: "team1 胜率",
    chart_kill: "击杀",
    chart_time: "时间",
    chart_weapon: "武器",
    chart_impact: "影响",
    status_submitting: "任务提交中...\n提示: 解析 demo 一般 30~60s，模型推理一般不到 1 分钟。",
    status_failed: "分析失败: {error}",
    status_partial_failed: "分析完成，但部分回合失败:\n{errors}",
    status_phase: "阶段: {phase}",
    status_progress: "进度: {progress}%",
    status_round_progress: "回合进度: {ratio}",
    status_current_round: "当前回合: R{round}",
    status_logs: "日志:",
    status_no_logs: "(暂无)",
    llm_need_analysis: "请先完成 DEM 分析",
    llm_need_fields: "请填写 API Key 和模型名",
    llm_generating: "LLM 正在生成复盘，请稍候...",
    llm_failed: "生成失败: {error}",
    llm_empty_result: "模型返回为空",
    llm_call_failed: "LLM 调用失败",
    browser_no_stream: "当前浏览器不支持流式读取",
  },
  en: {
    ui_language: "Language (界面语言)",
    subtitle: "Upload a DEM to compute round win rates, kill impact, and player contribution with interactive replay and LLM insights.",
    section_run: "1. Run Analysis",
    dem_file: "DEM File",
    model_path: "Model Root",
    model_path_placeholder: "Model root dir containing alive / nxt_kill / nxt_death / win_rate / duel",
    device: "Device",
    batch_size: "Batch Size",
    analyze_hint: "Progress note: demo parsing usually takes 30-60s, and model inference is usually under 1 minute.",
    analyze_btn: "Start Analysis",
    status_waiting: "Waiting for DEM upload...",
    section_round_trend: "2. Round Trend",
    section_viewer: "3. 2D Replay Viewer",
    viewer_open: "Open 2D Replay in New Tab",
    viewer_hint: "Open the same demo in the bundled 2D replay viewer (from third_party/csgo-2d-demo-viewer-dev, includes smoke/flash/grenade trajectories).",
    section_tick_state: "4. Round Live State",
    section_round_summary: "5. Final Contribution (Current Round)",
    section_overall_summary: "6. Overall Average Contribution",
    section_advanced: "7. Advanced Metrics",
    advanced_toggle: "Expand/Collapse",
    advanced_kill_rank: "Kill Difficulty / Impact Ranking",
    advanced_player_table: "Per-Player Advanced Metrics",
    advanced_kill_rank_hint: "Sorted by |swing|. difficulty>1 means the kill happened against a model-predicted disadvantage.",
    col_attacker: "Attacker",
    col_victim: "Victim",
    col_swing: "Swing",
    col_difficulty: "Difficulty",
    col_round: "Round",
    col_second: "Time(s)",
    col_avg_kill_opp: "Avg Kill Opportunity",
    col_avg_death_opp: "Avg Death Threat",
    col_avg_survive: "Avg Survival",
    col_hard_win: "Hard-Duel Win",
    col_easy_win: "Easy-Duel Win",
    col_highlight: "Highlight Rate",
    term_swing_help: "Swing: impact magnitude of a kill on the win-rate trajectory. Larger absolute value means more decisive.",
    term_difficulty_help: "Difficulty: duel difficulty coefficient. >1 is typically harder in model expectation, <1 is easier.",
    term_avg_kill_opp_help: "Avg Kill Opportunity: mean probability that this player gets the next kill across ticks.",
    term_avg_death_opp_help: "Avg Death Threat: mean probability that this player dies next across ticks.",
    term_avg_survive_help: "Avg Survival: mean short-horizon survival probability across ticks.",
    term_hard_win_help: "Hard-Duel Win: win rate in hard duels (difficulty>1).",
    term_easy_win_help: "Easy-Duel Win: win rate in easy duels (0<difficulty<1).",
    term_highlight_help: "Highlight Rate: fraction of rounds where total_contribution reaches the threshold. Current threshold: total_contribution >= 0.20 (20%).",
    term_avg_kill_help: "Avg Kill: cross-round average kill contribution (kill_contribution).",
    term_avg_tactical_help: "Avg Tactical: cross-round average tactical contribution (tactical_contribution).",
    term_avg_total_help: "Avg Total: cross-round average total contribution (kill + tactical).",
    section_llm: "8. LLM Summary",
    api_key: "API Key",
    model_name: "Model Name",
    model_name_placeholder: "gpt-4.1 / deepseek-chat / qwen-max",
    base_url: "Base URL (OpenAI-compatible)",
    temperature: "Temperature",
    llm_btn: "Generate AI Review",
    llm_empty: "No summary yet",
    no_data: "No data",
    team1: "team1",
    team2: "team2",
    tie: "Tie",
    unknown: "Unknown",
    col_player: "Player",
    col_team: "Team",
    col_kill: "Kill",
    col_tactical: "Tactical",
    col_total: "Total",
    col_avg_kill: "Avg Kill",
    col_avg_tactical: "Avg Tactical",
    col_avg_total: "Avg Total",
    col_rounds: "Rounds",
    match_badge: "Score {team1Label} {team1} : {team2} {team2Label} · Winner {winner}",
    round_tab: "R{round} · {winner}",
    chart_round: "Round",
    chart_team1_wr: "team1 Win Rate",
    chart_kill: "Kill",
    chart_time: "Time",
    chart_weapon: "Weapon",
    chart_impact: "Impact",
    status_submitting: "Submitting task...\nTip: demo parsing usually takes 30-60s, and model inference is usually under 1 minute.",
    status_failed: "Analysis failed: {error}",
    status_partial_failed: "Analysis finished, but some rounds failed:\n{errors}",
    status_phase: "Phase: {phase}",
    status_progress: "Progress: {progress}%",
    status_round_progress: "Round progress: {ratio}",
    status_current_round: "Current round: R{round}",
    status_logs: "Logs:",
    status_no_logs: "(none)",
    llm_need_analysis: "Please finish DEM analysis first",
    llm_need_fields: "Please provide API Key and model name",
    llm_generating: "LLM is generating the review, please wait...",
    llm_failed: "Generation failed: {error}",
    llm_empty_result: "Model returned empty content",
    llm_call_failed: "LLM call failed",
    browser_no_stream: "This browser does not support streaming reads",
  },
};

function currentLang() {
  const raw = (refs.appLanguage?.value || "zh").toLowerCase();
  return raw === "en" ? "en" : "zh";
}

function t(key, vars = {}) {
  const lang = currentLang();
  const dict = I18N[lang] || I18N.zh;
  const fallback = I18N.zh[key] || key;
  const template = dict[key] || fallback;
  return template.replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? `{${k}}`));
}

function localizeWinner(value) {
  if (value === "team1") return t("team1");
  if (value === "team2") return t("team2");
  if (value === "Tie") return t("tie");
  return t("unknown");
}

function applyStaticI18n() {
  document.documentElement.lang = currentLang() === "en" ? "en" : "zh-CN";

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (!key) return;
    el.textContent = t(key);
  });

  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (!key) return;
    el.setAttribute("placeholder", t(key));
  });
}

function applyLanguage() {
  applyStaticI18n();
  if (state.dashboard) {
    renderDashboard();
  }
}

function readUserPrefs() {
  try {
    const raw = window.localStorage.getItem(USER_PREFS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeUserPrefs(prefs) {
  try {
    window.localStorage.setItem(USER_PREFS_KEY, JSON.stringify(prefs));
  } catch {
    // Ignore storage errors (private mode / quota exceeded).
  }
}

function getModelPathOptions() {
  const list = document.getElementById("model-path-list");
  if (!list) return [];
  return Array.from(list.querySelectorAll("option"))
    .map((opt) => (opt.value || "").trim())
    .filter(Boolean);
}

function getCurrentUserPrefs() {
  return {
    app_language: refs.appLanguage?.value || "zh",
    model_path: refs.modelPath?.value || "",
    device: refs.device?.value || "",
    batch_size: refs.batchSize?.value || "",
    llm_api_key: refs.llmApiKey?.value || "",
    llm_model: refs.llmModel?.value || "",
    llm_base_url: refs.llmBaseUrl?.value || "",
    llm_temperature: refs.llmTemperature?.value || "",
  };
}

function saveUserPrefs() {
  writeUserPrefs(getCurrentUserPrefs());
}

function restoreUserPrefs() {
  const prefs = readUserPrefs();

  const savedLang =
    typeof prefs.app_language === "string"
      ? prefs.app_language
      : typeof prefs.llm_language === "string"
        ? prefs.llm_language
        : "zh";

  if (refs.appLanguage) {
    refs.appLanguage.value = savedLang === "en" ? "en" : "zh";
  }

  if (typeof prefs.model_path === "string" && refs.modelPath) {
    const savedModelPath = prefs.model_path.trim();
    const modelOptions = getModelPathOptions();

    // Keep backend default unless the saved path is explicitly in current options.
    if (savedModelPath && modelOptions.includes(savedModelPath)) {
      refs.modelPath.value = savedModelPath;
    }
  }
  if (typeof prefs.device === "string" && refs.device) {
    refs.device.value = prefs.device;
  }
  if (typeof prefs.batch_size === "string" && refs.batchSize) {
    refs.batchSize.value = prefs.batch_size;
  }
  if (typeof prefs.llm_api_key === "string" && refs.llmApiKey) {
    refs.llmApiKey.value = prefs.llm_api_key;
  }
  if (typeof prefs.llm_model === "string" && refs.llmModel) {
    refs.llmModel.value = prefs.llm_model;
  }
  if (typeof prefs.llm_base_url === "string" && refs.llmBaseUrl) {
    refs.llmBaseUrl.value = prefs.llm_base_url;
  }
  if (typeof prefs.llm_temperature === "string" && refs.llmTemperature) {
    refs.llmTemperature.value = prefs.llm_temperature;
  }
}

function bindUserPrefsPersistence() {
  const fields = [
    refs.appLanguage,
    refs.modelPath,
    refs.device,
    refs.batchSize,
    refs.llmApiKey,
    refs.llmModel,
    refs.llmBaseUrl,
    refs.llmTemperature,
  ];

  fields.forEach((field) => {
    if (!field) return;
    field.addEventListener("input", saveUserPrefs);
    field.addEventListener("change", saveUserPrefs);
  });
}

function logStatus(text) {
  refs.statusLog.textContent = text;
}

function renderLlmOutput(text) {
  const content = String(text || "");
  if (!content) {
    refs.llmOut.textContent = "";
    return;
  }

  if (window.marked && typeof window.marked.parse === "function") {
    const rawHtml = window.marked.parse(content, { gfm: true, breaks: true });
    if (window.DOMPurify && typeof window.DOMPurify.sanitize === "function") {
      refs.llmOut.innerHTML = window.DOMPurify.sanitize(rawHtml);
    } else {
      refs.llmOut.innerHTML = rawHtml;
    }
    return;
  }

  refs.llmOut.textContent = content;
}

function n4(value) {
  const num = Number(value || 0);
  return Number.isFinite(num) ? num.toFixed(4) : "0.0000";
}

function toPercent(value) {
  const num = Number(value || 0);
  return `${(num * 100).toFixed(1)}%`;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function contributionTextColor(value) {
  const v = Number(value || 0);
  const t = clamp(Math.abs(v) / 0.2, 0, 1);

  // Negative is red, positive is green, and zero is white.
  if (v >= 0) {
    const r = Math.round(lerp(255, 34, t));
    const g = Math.round(lerp(255, 139, t));
    const b = Math.round(lerp(255, 34, t));
    return `rgb(${r}, ${g}, ${b})`;
  }

  const r = Math.round(lerp(255, 220, t));
  const g = Math.round(lerp(255, 40, t));
  const b = Math.round(lerp(255, 40, t));
  return `rgb(${r}, ${g}, ${b})`;
}

function contributionCell(value) {
  const color = contributionTextColor(value);
  const percent = `${(Number(value || 0) * 100).toFixed(2)}%`;
  return `<span class="mono contribution-cell" style="color:${color}">${percent}</span>`;
}

function signedColor(signed, scale = 0.2) {
  const v = Number(signed || 0);
  const t = clamp(Math.abs(v) / scale, 0, 1);
  if (v >= 0) {
    const r = Math.round(lerp(255, 34, t));
    const g = Math.round(lerp(255, 139, t));
    const b = Math.round(lerp(255, 34, t));
    return `rgb(${r}, ${g}, ${b})`;
  }
  const r = Math.round(lerp(255, 220, t));
  const g = Math.round(lerp(255, 40, t));
  const b = Math.round(lerp(255, 40, t));
  return `rgb(${r}, ${g}, ${b})`;
}

function difficultyCell(value) {
  const num = Number(value || 0);
  if (!Number.isFinite(num)) {
    return `<span class="mono contribution-cell table-meta-text">-</span>`;
  }
  const color = signedColor(1 - num, 1.5);
  return `<span class="mono contribution-cell" style="color:${color}">${num.toFixed(3)}</span>`;
}

function rateCell(value, { center = 0.5, scale = 0.5, invert = false } = {}) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return `<span class="mono contribution-cell table-meta-text">-</span>`;
  }
  const signed = invert ? center - num : num - center;
  const color = signedColor(signed, scale);
  return `<span class="mono contribution-cell" style="color:${color}">${(num * 100).toFixed(2)}%</span>`;
}

function magnitudeCell(value, { scale = 0.5, polarity = "good" } = {}) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return `<span class="mono contribution-cell table-meta-text">-</span>`;
  }
  const signed = polarity === "good" ? num : -num;
  const color = signedColor(signed, scale);
  return `<span class="mono contribution-cell" style="color:${color}">${(num * 100).toFixed(2)}%</span>`;
}

function escapeAttr(text) {
  return String(text || "").replace(/[&<>\"']/g, (ch) => {
    if (ch === "&") return "&amp;";
    if (ch === "<") return "&lt;";
    if (ch === ">") return "&gt;";
    if (ch === '"') return "&quot;";
    return "&#39;";
  });
}

function withTermHelp(label, helpText) {
  const safeHelp = escapeAttr(helpText);
  return `<span class="th-label">${label}<span class="term-help" data-tooltip="${safeHelp}" title="${safeHelp}" aria-label="${safeHelp}">?</span></span>`;
}

function playerCell(player, row) {
  const badge = row.badge
    ? `<span class="player-badge ${row.badge.toLowerCase()}">${row.badge}</span>`
    : "";
  return `<span class="player-cell">${badge}<span>${player}</span></span>`;
}

function teamForPlayer(round, player) {
  if ((round.team1_players || []).includes(player)) return "team1";
  if ((round.team2_players || []).includes(player)) return "team2";
  return "?";
}

function renderTable(container, rows, columns) {
  if (!rows || rows.length === 0) {
    container.innerHTML = `<p>${t("no_data")}</p>`;
    return;
  }

  const header = columns.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows
    .map((row) => {
      const tds = columns
        .map((c) => {
          const value = c.render ? c.render(row[c.key], row) : row[c.key];
          return `<td>${value}</td>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");

  container.innerHTML = `<div class="table-wrap"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function buildTableHtml(rows, columns) {
  if (!rows || rows.length === 0) {
    return `<p>${t("no_data")}</p>`;
  }

  const header = columns.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows
    .map((row) => {
      const tds = columns
        .map((c) => {
          const value = c.render ? c.render(row[c.key], row) : row[c.key];
          return `<td>${value}</td>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");

  return `<div class="table-wrap"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderTeamSplitTables(container, rows, columns, layout = "row") {
  const team1Rows = (rows || []).filter((r) => r.team === "team1");
  const team2Rows = (rows || []).filter((r) => r.team === "team2");

  const team1Html = buildTableHtml(team1Rows, columns);
  const team2Html = buildTableHtml(team2Rows, columns);

  container.innerHTML = [
    `<div class="team-split-grid ${layout === "stack" ? "stack" : ""}">`,
    '<section class="team-split-card">',
    `<h4>${t("team1")}</h4>`,
    team1Html,
    '</section>',
    '<section class="team-split-card">',
    `<h4>${t("team2")}</h4>`,
    team2Html,
    '</section>',
    '</div>',
  ].join("");
}

function findNearestIndex(winRateSeries, t) {
  let bestIdx = 0;
  let bestGap = Number.MAX_SAFE_INTEGER;
  winRateSeries.forEach((item, idx) => {
    const gap = Math.abs(Number(item.round_seconds || 0) - t);
    if (gap < bestGap) {
      bestGap = gap;
      bestIdx = idx;
    }
  });
  return bestIdx;
}

function buildKillPoints(round) {
  const line = round.win_rate || [];
  return (round.kills || []).map((k) => {
    const idx = findNearestIndex(line, Number(k.round_seconds || 0));
    const wr = line[idx] ? Number(line[idx].team1_win_rate || 0) : 0;
    const killerTeam = teamForPlayer(round, k.killer);

    return {
      value: [Number(k.round_seconds || 0), wr],
      killer: k.killer,
      victim: k.victim,
      weapon: k.weapon,
      kill_impact: Number(k.kill_impact || 0),
      killerTeam,
      color: killerTeam === "team1" ? "#5ec8ff" : "#ff8a38",
    };
  });
}

function renderHoverContrib(round, index) {
  const snapshots = round.player_data || [];
  const snapshot = snapshots[index] || snapshots[snapshots.length - 1] || {};

  const rows = Object.entries(snapshot)
    .map(([player, value]) => {
      const kill = Number(value.kill_contribution || 0);
      const tactical = Number(value.tactical_contribution || 0);
      return {
        player,
        team: teamForPlayer(round, player),
        kill_contribution: kill,
        tactical_contribution: tactical,
        total_contribution: kill + tactical,
      };
    })
    .sort((a, b) => b.total_contribution - a.total_contribution);

  renderTeamSplitTables(refs.hoverPlayerStats, rows, [
    { key: "player", label: t("col_player"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "team", label: t("col_team"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "kill_contribution", label: t("col_kill"), render: (v) => contributionCell(v) },
    { key: "tactical_contribution", label: t("col_tactical"), render: (v) => contributionCell(v) },
    { key: "total_contribution", label: t("col_total"), render: (v) => contributionCell(v) },
  ]);
}

function renderRoundSummary(round) {
  const rows = (round.round_summary?.per_player || []).map((r) => ({
    ...r,
    team: teamForPlayer(round, r.player),
  }));

  renderTeamSplitTables(refs.roundSummaryTable, rows, [
    { key: "player", label: t("col_player"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "team", label: t("col_team"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "kill_contribution", label: t("col_kill"), render: (v) => contributionCell(v) },
    { key: "tactical_contribution", label: t("col_tactical"), render: (v) => contributionCell(v) },
    { key: "total_contribution", label: t("col_total"), render: (v) => contributionCell(v) },
  ], "stack");
}

function renderOverallSummary() {
  const overall = (state.dashboard?.overall || []).map((row) => ({ ...row }));
  const match = state.dashboard?.match || {};
  const mvpPlayer = match?.mvp?.player;
  const svpPlayer = match?.svp?.player;

  overall.forEach((row) => {
    if (mvpPlayer && row.player === mvpPlayer) {
      row.badge = "MVP";
    } else if (svpPlayer && row.player === svpPlayer) {
      row.badge = "SVP";
    }
  });

  renderTeamSplitTables(refs.overallSummaryTable, overall, [
    { key: "player", label: t("col_player"), render: (v, row) => `<span class="table-meta-text">${playerCell(v, row)}</span>` },
    { key: "team", label: t("col_team"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "avg_kill_contribution", label: t("col_avg_kill"), render: (v) => contributionCell(v) },
    { key: "avg_tactical_contribution", label: t("col_avg_tactical"), render: (v) => contributionCell(v) },
    { key: "avg_total_contribution", label: t("col_avg_total"), render: (v) => contributionCell(v) },
    { key: "rounds", label: t("col_rounds"), render: (v) => `<span class="mono table-meta-text">${v}</span>` },
  ], "stack");

  refs.matchBadge.textContent = t("match_badge", {
    team1Label: t("team1"),
    team2Label: t("team2"),
    team1: match.team1_round_wins ?? 0,
    team2: match.team2_round_wins ?? 0,
    winner: localizeWinner(match.winner),
  });
}

function renderRoundTabs() {
  const rounds = state.dashboard?.rounds || [];
  refs.roundTabs.innerHTML = rounds
    .map((rd, idx) => {
      const active = idx === state.selectedRoundIndex ? "active" : "";
      return `<button class="tab-btn ${active}" data-round-index="${idx}">${t("round_tab", { round: rd.round_id, winner: localizeWinner(rd.winner) })}</button>`;
    })
    .join("");

  refs.roundTabs.querySelectorAll(".tab-btn").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedRoundIndex = Number(el.dataset.roundIndex || 0);
      renderRoundTabs();
      renderCurrentRound();
    });
  });
}

function renderCurrentRound() {
  const rounds = state.dashboard?.rounds || [];
  const round = rounds[state.selectedRoundIndex];
  if (!round) return;

  const killPoints = buildKillPoints(round);
  const lineDataTeam1 = (round.win_rate || []).map((x) => [Number(x.round_seconds || 0), Number(x.team1_win_rate || 0)]);

  if (!state.chart) {
    state.chart = echarts.init(refs.chart);
    window.addEventListener("resize", () => state.chart && state.chart.resize());
  }

  const option = {
    backgroundColor: "transparent",
    animationDuration: 650,
    grid: { top: 52, right: 20, bottom: 52, left: 50 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "cross" },
      formatter: (params) => {
        const linePoint = params.find((p) => p.seriesName === "Team1 Win Rate");
        if (!linePoint) return "";
        const idx = linePoint.dataIndex;
        const sec = linePoint.value[0];
        const wr = linePoint.value[1];

        renderHoverContrib(round, idx);

        return [
          `<strong>${t("chart_round")} ${round.round_id} · ${sec.toFixed(2)}s</strong>`,
          `${t("chart_team1_wr")}: ${toPercent(wr)}`,
        ].join("<br/>");
      },
    },
    xAxis: {
      type: "value",
      name: "Round Seconds",
      axisLabel: { color: "#5a7b97" },
      axisLine: { lineStyle: { color: "#4f7796" } },
      splitLine: { lineStyle: { color: "rgba(126,165,194,0.15)" } },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      name: "Team1 Win Rate",
      axisLabel: { color: "#5a7b97", formatter: (v) => `${(v * 100).toFixed(0)}%` },
      axisLine: { lineStyle: { color: "#4f7796" } },
      splitLine: { lineStyle: { color: "rgba(126,165,194,0.15)" } },
    },
    legend: {
      top: 10,
      textStyle: { color: "#4f7190" },
    },
    series: [
      {
        name: "Team1 Win Rate",
        type: "line",
        smooth: true,
        symbol: "circle",
        symbolSize: 7,
        lineStyle: { width: 3, color: "#5ec8ff" },
        itemStyle: { color: "#5ec8ff" },
        areaStyle: { color: "rgba(94, 200, 255, 0.16)" },
        z: 4,
        data: lineDataTeam1,
      },
      {
        name: "Kill Markers",
        type: "scatter",
        symbolSize: 16,
        z: 20,
        zlevel: 2,
        data: killPoints,
        itemStyle: {
          color: (params) => params.data.color,
          borderColor: "#ffffff",
          borderWidth: 2,
          shadowBlur: 8,
          shadowColor: "rgba(0, 0, 0, 0.15)",
        },
        tooltip: {
          trigger: "item",
          formatter: (param) => {
            const d = param.data;
            return [
              `<strong>${d.killer} (${d.killerTeam})</strong>`,
              `${t("chart_kill")}: ${d.victim}`,
              `${t("chart_time")}: ${d.value[0].toFixed(2)}s`,
              `${t("chart_weapon")}: ${d.weapon}`,
              `${t("chart_impact")}: ${n4(d.kill_impact)}`,
            ].join("<br/>");
          },
        },
      },
    ],
  };

  state.chart.setOption(option, true);
  renderRoundSummary(round);
  renderHoverContrib(round, 0);
}

function renderViewerLink() {
  if (!refs.viewerOpen) return;
  if (!state.runId) {
    refs.viewerOpen.classList.add("disabled");
    refs.viewerOpen.removeAttribute("href");
    return;
  }
  refs.viewerOpen.classList.remove("disabled");
  const demoUrl = `${window.location.origin}/api/demo_file/${encodeURIComponent(state.runId)}.dem`;
  const params = [
    `demourl=${encodeURIComponent(demoUrl)}`,
    "directfetch=1",
  ];
  if (state.analysisId) {
    const timelineUrl = `${window.location.origin}/api/winrate_timeline/${encodeURIComponent(state.analysisId)}`;
    params.push(`winrateurl=${encodeURIComponent(timelineUrl)}`);
  }
  refs.viewerOpen.setAttribute("href", `/viewer/player?${params.join("&")}`);
}

function fmtPercent2(v) {
  const num = Number(v);
  if (!Number.isFinite(num)) return "-";
  return `${(num * 100).toFixed(2)}%`;
}

function fmtFloat3(v) {
  const num = Number(v);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(3);
}

function renderAdvancedMetrics() {
  if (!refs.advancedBody) return;
  const adv = state.dashboard?.advanced || {};
  const killRanking = adv.kill_ranking || [];
  const playerStats = adv.player_stats || [];
  const overall = state.dashboard?.overall || [];
  const overallByPlayer = new Map(overall.map((row) => [row.player, row]));

  if (!killRanking.length && !playerStats.length) {
    refs.advancedBody.innerHTML = `<p>${t("no_data")}</p>`;
    return;
  }

  const killCols = [
    { key: "round", label: t("col_round"), render: (v) => `<span class="mono table-meta-text">${v}</span>` },
    { key: "round_seconds", label: t("col_second"), render: (v) => `<span class="mono table-meta-text">${fmtFloat3(v)}</span>` },
    { key: "attacker", label: t("col_attacker"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "victim", label: t("col_victim"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "swing", label: withTermHelp(t("col_swing"), t("term_swing_help")), render: (v) => contributionCell(v) },
    { key: "difficulty", label: withTermHelp(t("col_difficulty"), t("term_difficulty_help")), render: (v) => difficultyCell(v) },
  ];

  const playerRows = playerStats.map((row) => {
    const overallRow = overallByPlayer.get(row.player) || {};
    return {
      ...row,
      avg_kill_contribution: overallRow.avg_kill_contribution,
      avg_tactical_contribution: overallRow.avg_tactical_contribution,
      avg_total_contribution: overallRow.avg_total_contribution,
    };
  });

  const playerCols = [
    { key: "player", label: t("col_player"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "team", label: t("col_team"), render: (v) => `<span class="table-meta-text">${v}</span>` },
    { key: "avg_survive_chance", label: withTermHelp(t("col_avg_survive"), t("term_avg_survive_help")), render: (v) => magnitudeCell(v, { scale: 0.5, polarity: "good" }) },
    { key: "hard_win_rate", label: withTermHelp(t("col_hard_win"), t("term_hard_win_help")), render: (v) => rateCell(v, { center: 0.5, scale: 0.5 }) },
    { key: "easy_win_rate", label: withTermHelp(t("col_easy_win"), t("term_easy_win_help")), render: (v) => rateCell(v, { center: 0.5, scale: 0.5 }) },
    { key: "highlight_rate", label: withTermHelp(t("col_highlight"), t("term_highlight_help")), render: (v) => rateCell(v, { center: 0.1, scale: 0.2 }) },
    { key: "avg_kill_contribution", label: withTermHelp(t("col_avg_kill"), t("term_avg_kill_help")), render: (v) => contributionCell(v) },
    { key: "avg_tactical_contribution", label: withTermHelp(t("col_avg_tactical"), t("term_avg_tactical_help")), render: (v) => contributionCell(v) },
    { key: "avg_total_contribution", label: withTermHelp(t("col_avg_total"), t("term_avg_total_help")), render: (v) => contributionCell(v) },
  ];

  const collapsed = !state.advancedExpanded;
  const ranking = collapsed ? killRanking.slice(0, 8) : killRanking;
  const ranking_more = killRanking.length > ranking.length
    ? `<p class="hint">+${killRanking.length - ranking.length} more — ${t("advanced_toggle")}</p>`
    : "";

  refs.advancedBody.innerHTML = [
    `<h3>${t("advanced_kill_rank")}</h3>`,
    `<p class="hint">${t("advanced_kill_rank_hint")}</p>`,
    buildTableHtml(ranking, killCols),
    ranking_more,
    `<h3>${t("advanced_player_table")}</h3>`,
    buildTableHtml(playerRows, playerCols),
  ].join("");
}

function renderDashboard() {
  refs.roundView.classList.remove("hidden");
  refs.roundSummaryView.classList.remove("hidden");
  refs.overallSummaryView.classList.remove("hidden");
  refs.llmView.classList.remove("hidden");
  if (refs.viewerLaunchView) refs.viewerLaunchView.classList.remove("hidden");
  if (refs.advancedView) refs.advancedView.classList.remove("hidden");

  renderRoundTabs();
  renderCurrentRound();
  renderOverallSummary();
  renderViewerLink();
  renderAdvancedMetrics();

  const errors = state.dashboard?.errors || {};
  if (Object.keys(errors).length > 0) {
    const errText = Object.entries(errors).map(([k, v]) => `${k}: ${v}`).join("\n");
    logStatus(t("status_partial_failed", { errors: errText }));
  }
}

async function runAnalyze(formData) {
  const resp = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });

  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || "分析失败");
  }
  return data;
}

async function fetchAnalyzeStatus(jobId) {
  const resp = await fetch(`/api/analyze_status/${jobId}`);
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || "获取进度失败");
  }
  return data;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderProgress(status) {
  const phase = status.phase || "运行中";
  const message = status.message || "";
  const progress = Number(status.progress || 0);
  const processed = Number(status.processed_rounds || 0);
  const total = Number(status.total_rounds || 0);
  const roundText = status.current_round ? `当前回合: R${status.current_round}` : "";
  const ratio = total > 0 ? `${processed}/${total}` : "-";

  const lines = [
    t("status_phase", { phase }),
    t("status_progress", { progress }),
    t("status_round_progress", { ratio }),
    roundText ? t("status_current_round", { round: status.current_round }) : "",
    message,
    "",
    t("status_logs"),
    status.logs || t("status_no_logs"),
  ].filter(Boolean);

  logStatus(lines.join("\n"));
}

async function waitAnalyzeDone(jobId) {
  while (true) {
    const status = await fetchAnalyzeStatus(jobId);
    renderProgress(status);

    if (status.status === "succeeded") {
      return status;
    }
    if (status.status === "failed") {
      throw new Error(status.error || status.message || "分析失败");
    }

    await sleep(1000);
  }
}

if (refs.advancedToggle) {
  refs.advancedToggle.addEventListener("click", () => {
    state.advancedExpanded = !state.advancedExpanded;
    renderAdvancedMetrics();
  });
}

restoreUserPrefs();
bindUserPrefsPersistence();
applyLanguage();

if (refs.appLanguage) {
  refs.appLanguage.addEventListener("change", () => {
    saveUserPrefs();
    applyLanguage();
  });
}

refs.analyzeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  refs.analyzeBtn.disabled = true;
  logStatus(t("status_submitting"));

  try {
    const formData = new FormData(refs.analyzeForm);
    const startData = await runAnalyze(formData);
    const jobId = startData.job_id;
    const done = await waitAnalyzeDone(jobId);

    state.analysisId = done.analysis_id;
    state.dashboard = done.dashboard;
    state.selectedRoundIndex = 0;
    state.runId = done.run_id || null;

    renderDashboard();
    renderProgress(done);
  } catch (err) {
    logStatus(t("status_failed", { error: err.message }));
  } finally {
    refs.analyzeBtn.disabled = false;
  }
});

refs.llmBtn.addEventListener("click", async () => {
  if (!state.analysisId) {
    renderLlmOutput(t("llm_need_analysis"));
    return;
  }

  const apiKey = refs.llmApiKey.value.trim();
  const modelName = refs.llmModel.value.trim();
  const baseUrl = refs.llmBaseUrl.value.trim();
  const temperature = Number(refs.llmTemperature.value || 0.95);
  const language = currentLang();

  if (!apiKey || !modelName) {
    renderLlmOutput(t("llm_need_fields"));
    return;
  }

  saveUserPrefs();
  refs.llmBtn.disabled = true;
  renderLlmOutput(t("llm_generating"));

  try {
    const resp = await fetch("/api/llm_summary_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_id: state.analysisId,
        api_key: apiKey,
        model_name: modelName,
        base_url: baseUrl,
        temperature,
        language,
      }),
    });

    if (!resp.ok) {
      let errMsg = t("llm_call_failed");
      try {
        const errData = await resp.json();
        errMsg = errData.error || errMsg;
      } catch {
        const plainText = await resp.text();
        if (plainText) errMsg = plainText;
      }
      throw new Error(errMsg);
    }

    if (!resp.body) {
      throw new Error(t("browser_no_stream"));
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullText = "";

    renderLlmOutput("");

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      fullText += decoder.decode(value, { stream: true });
      renderLlmOutput(fullText);
    }

    fullText += decoder.decode();
    renderLlmOutput(fullText || t("llm_empty_result"));
  } catch (err) {
    renderLlmOutput(t("llm_failed", { error: err.message }));
  } finally {
    refs.llmBtn.disabled = false;
  }
});
