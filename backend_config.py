from __future__ import annotations

STARTUP_BOARD_SIZE = 10
BOARD_SIZES = (10, 15, 20, 25, 30, 35)
AGENT_KINDS = ("evolution", "dqn")
MODEL_ACTION_LABELS = ("Вперед", "Влево", "Вправо")

GLOBAL_SIGNAL_LABELS = (
    "food_ahead",
    "food_right",
    "food_left",
    "apple_step_forward",
    "apple_step_right",
    "apple_step_left",
    "danger_forward",
    "danger_right",
    "danger_left",
    "forward_food_reachable",
    "forward_tail_reachable",
    "forward_enough_space",
    "forward_food_tail_safe",
    "left_food_reachable",
    "left_tail_reachable",
    "left_enough_space",
    "left_food_tail_safe",
    "right_food_reachable",
    "right_tail_reachable",
    "right_enough_space",
    "right_food_tail_safe",
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
TRAP_AFTER_APPLE_PENALTY = 3.0
TAIL_PATH_LOST_PENALTY = 7.0
STEP_PENALTY = 0.02
TURN_PENALTY = 0.1
DEATH_PENALTY = 5.0
DQN_TAIL_PATH_LOST_REPLAY_EXTRA_COPIES = 4
ELITE_EPISODE_MAX_TRANSITIONS = 320
EPISODE_TRANSITION_LIMIT = 2_000
DQN_DEATH_REPLAY_EXTRA_COPIES = 3
DQN_PRE_DEATH_CONTEXT_REPLAYS = 3
DQN_STARVATION_PRIORITY_SCALE = 0.9
DQN_STARVATION_EXTRA_REPLAY_COPIES = 0
ENDGAME_FILL_RATIO = 0.70
EVOLUTION_STARVATION_FITNESS_FLOOR = -25_000.0
EVOLUTION_STARVATION_DEATH_MULTIPLIER = 1.25
EVOLUTION_STARVATION_EFFICIENCY_MULTIPLIER = 1.2
EVOLUTION_STARVATION_HUNGER_MULTIPLIER = 1.3

CPU_BATCH_SIZE = 128
HIDDEN_LAYER_1 = 96
HIDDEN_LAYER_2 = 48

STEP_REASON_ORDER = (
    "step",
    "apple",
    "looping",
    "wall_collision",
    "body_collision",
    "starvation",
    "starvation_food_unreachable",
    "starvation_food_available",
    "starvation_safe_loop",
    "victory",
)
DEATH_REASON_ORDER = (
    "victory",
    "wall_collision",
    "body_collision",
    "starvation",
    "starvation_food_unreachable",
    "starvation_no_relayout_path",
    "starvation_food_available",
    "starvation_safe_loop",
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
    ("step",             "Шаг",            "Шаг",            STEP_PENALTY),
    ("turn",             "Поворот",        "Поворот",        TURN_PENALTY),
    ("wall_collision",   "Смерть: стена",  "Стена",          DEATH_PENALTY),
    ("body_collision",   "Смерть: тело",   "Тело",           DEATH_PENALTY),
    ("starvation",       "Голодание",      "Голод",          DEATH_PENALTY),
    ("tail_path_lost",   "Нет пути к хвосту", "Потеря хвоста", TAIL_PATH_LOST_PENALTY),
    ("trap_after_apple", "Тупик после 🍎", "Тупик после 🍎", TRAP_AFTER_APPLE_PENALTY),
]

PENALTY_CATALOG = {
    cid: {"label": label, "short_label": short_label, "value": -value}
    for cid, label, short_label, value in _PENALTY_MAP
}

# Исключения (не ложатся в шаблон)
PENALTY_CATALOG["looping"] = {"label": "Цикл", "short_label": "Цикл", "value": "до -0.35"}
PENALTY_CATALOG["avoidable_collision"] = {
    "label": "Избежимое столкновение",
    "short_label": "Избежимое столкновение",
    "value": "метка",
    "show_in_ui": False,
}

REWARD_CATALOG = {
    "apple": {"label": "Яблоко", "short_label": "Яблоко", "value": APPLE_REWARD},
    "victory": {"label": "Победа", "short_label": "Победа", "value": 10.0},
}

def hunger_budget_for(score: int, board_size: int) -> float:
    board_cells = max(1, board_size * board_size)
    snake_length = min(float(score + 3), float(board_cells))
    return board_size * 4.0 + snake_length * 1.0
