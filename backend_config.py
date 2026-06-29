from __future__ import annotations

STARTUP_BOARD_SIZE = 10
BOARD_SIZES = (10, 15, 20, 25, 30, 35)
AGENT_KINDS = ("evolution", "dqn")
MODEL_ACTION_LABELS = ("Вперед", "Влево", "Вправо")

# Все относительные группы используют единый порядок действий:
# forward, left, right. Behind-сигналы идут после этих направлений.
GLOBAL_SIGNAL_LABELS = (
    "food_ahead",
    "food_left",
    "food_right",
    "food_behind",
    "apple_step_forward",
    "apple_step_left",
    "apple_step_right",
    "danger_forward",
    "danger_left",
    "danger_right",
    "tail_ahead",
    "tail_left",
    "tail_right",
    "tail_behind",
    "tail_blocked_forward",
    "tail_blocked_left",
    "tail_blocked_right",
    "space_tight_forward",
    "space_tight_left",
    "space_tight_right",
    "tail_close_forward",
    "tail_close_left",
    "tail_close_right",
    "partition_forward",
    "partition_left",
    "partition_right",
)
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
TAIL_LOSS_PENALTY = 12.0
TAIL_CLOSE_PENALTY = 1.2
PARTITION_PENALTY_LARGE = 1.5
PARTITION_PENALTY_SMALL = 0.05
EPISODE_TRANSITION_LIMIT = 2_000

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

PENALTY_CATALOG = {
    "death": {"label": "Смерть", "short_label": "Смерть", "value": -DEATH_PENALTY},
    "looping": {"label": "Цикл", "short_label": "Цикл", "value": "до -0.35"},
    "tail_loss": {"label": "Потеря хвоста", "short_label": "Хвост", "value": -TAIL_LOSS_PENALTY},
    "tail_close": {"label": "Близко к хвосту", "short_label": "Близко", "value": f"до -{2 * TAIL_CLOSE_PENALTY:.2g}"},
    "partition": {"label": "Разрыв поля", "short_label": "Разрыв", "value": f"до -{PARTITION_PENALTY_LARGE:.2g}"},
}

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
    "partition": "Разрыв поля",
    "blocked": "ход заблокирован",
    "safe": "безопасно",
    "none": "нет",
    "wall_collision": "Смерть: стена",
    "body_collision": "Смерть: тело",
    "starvation": "Смерть: голод",
    "death": "Смерть",
    "tail_loss": "Потеря хвоста",
    "tail_close": "Близко к хвосту",
}

SIGNAL_LABELS = {
    "food_ahead": "🍎↑",
    "food_left": "🍎←",
    "food_right": "🍎→",
    "food_behind": "🍎↓",
    "apple_step_forward": "🍎+↑",
    "apple_step_left": "🍎+←",
    "apple_step_right": "🍎+→",
    "danger_forward": "💀 ↑",
    "danger_left": "💀 ←",
    "danger_right": "💀 →",
    "tail_ahead": "🐍↑",
    "tail_left": "🐍←",
    "tail_right": "🐍→",
    "tail_behind": "🐍↓",
    "tail_blocked_forward": "⛔↑",
    "tail_blocked_left": "⛔←",
    "tail_blocked_right": "⛔→",
    "space_tight_forward": "⚠️↑",
    "space_tight_left": "⚠️←",
    "space_tight_right": "⚠️→",
    "tail_close_forward": "🐍🤏↑",
    "tail_close_left": "🐍🤏←",
    "tail_close_right": "🐍🤏→",
}

TERMINAL_REASONS = (
    "wall_collision",
    "body_collision",
    "victory",
    "starvation",
)

def hunger_budget_for(score: int, board_size: int) -> float:
    board_cells = max(1, board_size * board_size)
    snake_length = min(float(score + 3), float(board_cells))
    return board_size * 4.0 + snake_length * 2.0