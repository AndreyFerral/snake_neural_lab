function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
function escapeText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
function average(values) {
  if (!values.length) { return 0; }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function quantile(values, ratio) {
  if (!values.length) { return 0; }
  const sorted = [...values].sort((a, b) => a - b);
  const index = clamp((sorted.length - 1) * ratio, 0, sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) { return sorted[lower]; }
  const fraction = index - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}
function colorForActivation(value) {
  const normalized = clamp((value + 1) / 2, 0, 1);
  const hue = 32 + normalized * 128;
  const lightness = 28 + normalized * 42;
  return `hsl(${hue}, 78%, ${lightness}%)`;
}
function drawCanvasLines(ctx, lines, x, y, lineHeight) {
  lines.forEach((line, index) => {
    ctx.fillText(line, x, y + index * lineHeight);
  });
}
function formatReason(reason) {
  const runtimeMap = (typeof REASON_LABELS === "object" && REASON_LABELS) ? REASON_LABELS : null;
  return runtimeMap?.[reason] ?? reason ?? "Ожидание";
}
function formatAppleVerdictTag(verdict) {
  const map = {
    safe: "безопасно",
    blocked: "блок",
    no_tail_after_apple: "нет хвоста",
    not_enough_space: "мало места",
    none: "-",
  };
  return map[verdict] ?? formatReason(verdict);
}
function formatSignalLabel(label) {
  const runtimeMap = (typeof SIGNAL_LABELS === "object" && SIGNAL_LABELS) ? SIGNAL_LABELS : null;
  return runtimeMap?.[label] ?? label;
}
async function apiJson(path, options = {}) {
  const timeoutMs = Number(options.timeoutMs ?? 6000);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const { timeoutMs: _timeoutMs, headers: optionHeaders, ...fetchOptions } = options;
  const headers = { ...(optionHeaders ?? {}) };
  if (fetchOptions.body !== undefined && !Object.keys(headers).some((key) => key.toLowerCase() === "content-type")) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, {
    cache: "no-store",
    headers,
    ...fetchOptions,
    signal: controller.signal,
  }).finally(() => window.clearTimeout(timeout));
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return response.json();
}

const DEFAULT_INPUT_GRID = 25;
const DEFAULT_OUTPUT_LABELS = ["Вперед", "Влево", "Вправо"];
const OUTPUT_DISPLAY_ORDER = [0, 1, 2];
const API_POLL_MS = 120;
const TRAINING_API_POLL_MS = 500;
const FRAME_POLL_MS = 16;
const FRAME_IDLE_POLL_MS = 240;
const FRAME_HIDDEN_POLL_MS = 1000;
const DETAIL_PAYLOAD_MS = 180;
const LIVE_DETAIL_PAYLOAD_MS = 1500;
const NETWORK_PAYLOAD_MS = 350;
const LIVE_NETWORK_PAYLOAD_MS = 1800;
const NETWORK_RENDER_MS = 350;
const CHART_RENDER_MS = 900;
const UI_PANEL_RENDER_MS = 220;
const SNAPSHOT_RENDER_MS = 900;
const FORCE_PANEL_REFRESH_COMMANDS = new Set([
  "set_agent_kind",
  "set_mode",
  "set_smart_model",
  "set_input_mode",
  "set_board_size",
  "cycle_board_size",
  "reset_agent",
]);
const DEFERRED_STATE_FIELDS = [
  "reward_rules",
  "step_review",
  "step_review_source",
  "step_review_history",
  "observation",
  "global_signals",
  "last_death_snapshot",
  "decision_stats",
  "history",
];
const STALE_FRAME_FIELDS = [
  "game",
  "stats",
  "q_values",
  "action_scores",
  "last_action",
  "last_action_random",
  "last_greedy_action",
  "last_reason",
  "last_outcome_reason",
  "last_penalty_reasons",
];
const FRAME_PRESERVED_FIELDS = [
  "smart_model",
  "available_devices",
  "available_agent_kinds",
  "available_input_modes",
  "available_input_grids",
  "local_map_channels",
  "output_labels",
  "network",
  ...DEFERRED_STATE_FIELDS,
];
const NETWORK_CONNECTION_KEYS = ["global_hidden1", "hidden1_hidden2", "hidden2_output", "hidden2_value"];
const DEFAULT_TERMINAL_REASONS = [];
const TERMINAL_REASONS = new Set(DEFAULT_TERMINAL_REASONS);
let REASON_LABELS = Object.freeze({});
let SIGNAL_LABELS = Object.freeze({});
function applyRuntimeReasonConfig(state) {
  if (!state || typeof state !== "object") return;
  const terminalReasons = Array.isArray(state.terminal_reasons) ? state.terminal_reasons : null;
  if (terminalReasons) {
    TERMINAL_REASONS.clear();
    terminalReasons.forEach((reason) => TERMINAL_REASONS.add(String(reason)));
  }
  const reasonLabels = state.reason_labels;
  if (reasonLabels && typeof reasonLabels === "object") {
    REASON_LABELS = Object.freeze({ ...reasonLabels });
  }
  const signalLabels = state.signal_labels;
  if (signalLabels && typeof signalLabels === "object") {
    SIGNAL_LABELS = Object.freeze({ ...signalLabels });
  }
}
const KEY_DIRECTIONS = Object.freeze({
  arrowup: 0,
  w: 0,
  ц: 0,
  arrowright: 1,
  d: 1,
  в: 1,
  arrowdown: 2,
  s: 2,
  ы: 2,
  arrowleft: 3,
  a: 3,
  ф: 3,
});
