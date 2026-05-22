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
const TERMINAL_REASONS = new Set([
  "wall_collision",
  "body_collision",
  "starvation",
  "starvation_food_unreachable",
  "starvation_no_relayout_path",
  "starvation_food_available",
  "starvation_safe_loop",
  "starvation_endgame_trap",
  "victory",
]);
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
