from __future__ import annotations

STARTUP_BOARD_SIZE = 10
BOARD_SIZES = (10, 15, 20, 25, 30, 35)
AGENT_KINDS = ("evolution", "dqn")
MODEL_ACTION_LABELS = ("Вперед", "Влево", "Вправо")

GLOBAL_SIGNAL_LABELS = (
    "food_ahead",
    "food_right",
    "food_left",
    "food_behind",
    "apple_step_forward",
    "apple_step_right",
    "apple_step_left",
    "danger_forward",
    "danger_right",
    "danger_left",
    "tail_ahead",
    "tail_right",
    "tail_left",
    "tail_blocked_forward",
    "tail_blocked_right",
    "tail_blocked_left",
    "trap_ahead",
    "trap_right",
    "trap_left",
)
GLOBAL_SIGNAL_INDEX = {label: index for index, label in enumerate(GLOBAL_SIGNAL_LABELS)}
DIRECTIONS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
# Ограничение длины "Истории обучения" (для UI и snake_model_stats.json)
HISTORY_LIMIT = 100
DECISION_STATS_WINDOW = 100
AVERAGE_SCORE_WINDOW_RATIO = 0.25
SAVE_THROTTLE_SECONDS = 8.0
APPLE_REWARD = 4.0
DEATH_PENALTY = 8.0
ELITE_EPISODE_MAX_TRANSITIONS = 320
EPISODE_TRANSITION_LIMIT = 2_000
DQN_DEATH_REPLAY_EXTRA_COPIES = 3
DQN_PRE_DEATH_CONTEXT_REPLAYS = 3
DQN_STARVATION_PRIORITY_SCALE = 0.9
DQN_STARVATION_EXTRA_REPLAY_COPIES = 0
ENDGAME_FILL_RATIO = 0.70
EVOLUTION_STARVATION_FITNESS_FLOOR = -25_000.0
EVOLUTION_STARVATION_DEATH_MULTIPLIER = 1.6
EVOLUTION_STARVATION_EFFICIENCY_MULTIPLIER = 1.5
EVOLUTION_STARVATION_HUNGER_MULTIPLIER = 1.3

CPU_BATCH_SIZE = 128
HIDDEN_LAYER_1 = 32
HIDDEN_LAYER_2 = 16


STEP_REASON_ORDER = (
    "step",
    "apple",
    "looping",
    "wall_collision",
    "body_collision",
    "victory",
)
DEATH_REASON_ORDER = (
    "victory",
    "wall_collision",
    "body_collision",
    "starvation",
)
PERF_LOG_MAX_BYTES = 512_000
SLOW_API_MS = 35.0
SLOW_STEP_MS = 35.0
SLOW_NETWORK_MS = 35.0
NETWORK_INPUT_ADAPTER_MODE = "strict"  # strict | trim | pad
NETWORK_INPUT_WARN_ON_ADAPT = True
NETWORK_INPUT_PAD_VALUE = 0.0

# --- Каталог штрафов для UI (строится автоматически из констант) ---
# (catalog_id, label, short_label, константа)
_PENALTY_MAP = [
    ("death",   "Смерть",  "Смерть",          DEATH_PENALTY),
]

PENALTY_CATALOG = {
    cid: {"label": label, "short_label": short_label, "value": -value}
    for cid, label, short_label, value in _PENALTY_MAP
}

# Исключения (не ложатся в шаблон)
PENALTY_CATALOG["looping"] = {"label": "Цикл", "short_label": "Цикл", "value": "до -0.35"}

REWARD_CATALOG = {
    "apple": {"label": "Яблоко", "short_label": "Яблоко", "value": APPLE_REWARD},
    "victory": {"label": "Победа", "short_label": "Победа", "value": 10.0},
}

REASON_LABELS = {
    "new_game": "Новая игра",
    "step": "Шаг",
    "turn": "Поворот",
    "apple": "Яблоко",
    "victory": "Победа",
    "looping": "Ходит по кругу",
    "blocked": "ход заблокирован",
    "safe": "безопасно",
    "none": "нет",
    "wall_collision": "Смерть: стена",
    "body_collision": "Смерть: тело",
    "starvation": "Смерть: голод",
    "death": "Смерть",
}

SIGNAL_LABELS = {
    "food_ahead": "🍎↑",
    "food_right": "🍎→",
    "food_left": "🍎←",
    "food_behind": "🍎↓",
    "danger_forward": "💀 ↑",
    "danger_right": "💀 →",
    "danger_left": "💀 ←",
    "apple_step_forward": "🍎+↑",
    "apple_step_right": "🍎+→",
    "apple_step_left": "🍎+←",
    "tail_ahead": "🐍↑",
    "tail_right": "🐍→",
    "tail_left": "🐍←",
    "tail_blocked_forward": "🔗↑",
    "tail_blocked_right": "🔗→",
    "tail_blocked_left": "🔗←",
    "trap_ahead": "🕸️↑",
    "trap_right": "🕸️→",
    "trap_left": "🕸️←",
}

TERMINAL_REASONS = tuple(dict.fromkeys((
    "wall_collision",
    "body_collision",
    "victory",
    "starvation",
)))

def hunger_budget_for(score: int, board_size: int) -> float:
    board_cells = max(1, board_size * board_size)
    snake_length = min(float(score + 3), float(board_cells))
    return board_size * 4.0 + snake_length * 1.0
