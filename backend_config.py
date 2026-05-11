from __future__ import annotations

DEFAULT_INPUT_GRID = 25
STARTUP_INPUT_GRID = 0
STARTUP_INPUT_MODE = "signals"
STARTUP_BOARD_SIZE = 10
INPUT_GRID_OPTIONS = (0, 5, 10, 15, 20, 25)
INPUT_MODES = ("signals", "map", "hybrid")
LOCAL_MAP_CHANNEL_LABELS = ("body", "head", "food", "tail", "wall")
LOCAL_MAP_CHANNEL_COUNT = len(LOCAL_MAP_CHANNEL_LABELS)
BOARD_SIZES = (10, 15, 20, 25, 30, 35)
MODEL_SCHEMA_VERSION = 52
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
    "forward_route_follow",
    "left_route_follow",
    "right_route_follow",
    "forward_route_apple_line",
    "left_route_apple_line",
    "right_route_apple_line",
    "forward_route_return",
    "left_route_return",
    "right_route_return",
)
GLOBAL_SIGNAL_INDEX = {label: index for index, label in enumerate(GLOBAL_SIGNAL_LABELS)}
DIRECTIONS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
HISTORY_LIMIT = 100
DECISION_STATS_WINDOW = 100
AVERAGE_SCORE_WINDOW_RATIO = 0.25
SAVE_THROTTLE_SECONDS = 8.0
APPLE_REWARD = 4.0
TRAP_AFTER_APPLE_PENALTY = 3.0
TAIL_PATH_LOST_PENALTY = 7.0
DQN_TAIL_PATH_LOST_REPLAY_EXTRA_COPIES = 4
ELITE_EPISODE_MAX_TRANSITIONS = 320
EPISODE_TRANSITION_LIMIT = 2_000
DQN_DEATH_REPLAY_EXTRA_COPIES = 3
DQN_PRE_DEATH_CONTEXT_REPLAYS = 3
ENDGAME_FILL_RATIO = 0.70
ROUTE_FOLLOW_BONUS = 0.22
ROUTE_DEVIATION_PENALTY = 0.55
ROUTE_APPLE_LINE_BONUS = 0.25
ROUTE_RETURN_AFTER_APPLE_BONUS = 0.40
CPU_BATCH_SIZE = 128
HIDDEN_LAYER_1 = 96
HIDDEN_LAYER_2 = 48
LOCAL_BRANCH_HIDDEN = 48
GLOBAL_BRANCH_HIDDEN = 24
LOCAL_CONV_CHANNELS = 32
STEP_REASON_ORDER = (
    "step",
    "apple",
    "looping",
    "route_dead_end",
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

def local_input_size(input_grid: int) -> int:
    return input_grid * input_grid * LOCAL_MAP_CHANNEL_COUNT

def global_signal_count_for_mode(input_mode: str) -> int:
    return 0 if input_mode == "map" else len(GLOBAL_SIGNAL_LABELS)

def hunger_budget_for(score: int, board_size: int) -> float:
    board_cells = max(1, board_size * board_size)
    snake_length = min(float(score + 3), float(board_cells))
    return board_size * 4.0 + snake_length * 1.0