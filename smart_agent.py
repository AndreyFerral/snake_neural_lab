from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from backend_config import DIRECTIONS, MODEL_ACTION_LABELS

if TYPE_CHECKING:
    from snake_env import SnakeEnv

SMART_MODEL_GLOBS = ("*.h5", "*.keras")
SMART_STATE_SIZE = 40
SMART_LONG_SNAKE_FILL_THRESHOLD = 0.35

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def normalize_signed(value: float, scale: float) -> float:
    return clamp(float(value) / max(1.0, float(scale)), -1.0, 1.0)

def relative_to_absolute(current_direction: int, action: int) -> int:
    if action == 0:
        return current_direction
    if action == 1:
        return (current_direction - 1) % 4
    return (current_direction + 1) % 4

def smart_model_candidates(runtime_dir: Path) -> list[Path]:
    if not runtime_dir.is_dir():
        return []
    candidates: list[Path] = []
    for pattern in SMART_MODEL_GLOBS:
        candidates.extend(runtime_dir.glob(pattern))
    def modified_at(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0
    return sorted(candidates, key=modified_at, reverse=True)

def latest_smart_model_path(runtime_dir: Path) -> Path | None:
    candidates = smart_model_candidates(runtime_dir)
    return candidates[0] if candidates else None

class SmartH5Agent:
    def __init__(self, runtime_dir: Path, static_root: Path, model_path: Path | None = None) -> None:
        self.runtime_dir = runtime_dir
        self.static_root = static_root
        self.model_path = model_path or latest_smart_model_path(runtime_dir)
        self.model: Any | None = None
        self.error: str | None = None
        self.loaded = False
        self.label = "Zmey-smart .h5"
        self.lookahead_counter = 0
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path is None:
            self.error = "Файл .h5 не найден в .runtime"
            return
        try:
            import tensorflow as tf  # type: ignore
            model = tf.keras.models.load_model(self.model_path, compile=False)
            input_size = int(model.input_shape[-1])
            output_size = int(model.output_shape[-1])
            if input_size != SMART_STATE_SIZE or output_size != len(MODEL_ACTION_LABELS):
                self.error = f"Ожидалась модель {SMART_STATE_SIZE}->{len(MODEL_ACTION_LABELS)}, получено {input_size}->{output_size}"
                return
            self.model = model
            self.loaded = True
            self.error = None
            self.label = f"Zmey-smart: {self.model_path.name}"
        except Exception as exc:
            self.model = None
            self.loaded = False
            self.error = f"{type(exc).__name__}: {exc}"

    def set_model_path(self, model_path: Path) -> None:
        try:
            same_model = self.model_path is not None and self.model_path.resolve() == model_path.resolve()
        except OSError:
            same_model = self.model_path == model_path
            
        if same_model and self.loaded:
            return

        self.model_path = model_path
        self.model = None
        self.error = None
        self.loaded = False
        self.label = "Zmey-smart .h5"
        self.lookahead_counter = 0
        self._load_model()

    def close(self) -> None:
        pass

    def _next_head_from(self, head: tuple[int, int], direction: int) -> tuple[int, int]:
        move_x, move_y = DIRECTIONS[direction]
        return head[0] + move_x, head[1] + move_y

    def _will_collide_fast(self, board_size: int, point: tuple[int, int], body: set[tuple[int, int]]) -> bool:
        return (
            point[0] < 0 or point[0] >= board_size or
            point[1] < 0 or point[1] >= board_size or
            point in body
        )

    def _neighbors_for_independent(self, cell: tuple[int, int], board_size: int) -> list[tuple[int, int]]:
        x, y = cell
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                neighbors.append((nx, ny))
        return neighbors

    def _has_path_independent(self, start: tuple[int, int], target: tuple[int, int], blocked: set[tuple[int, int]], board_size: int) -> bool:
        if start in blocked or target in blocked:
            return False
        if start == target:
            return True
        visited = {start}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            if curr == target:
                return True
            for nx, ny in self._neighbors_for_independent(curr, board_size):
                next_cell = (nx, ny)
                if next_cell not in visited and next_cell not in blocked:
                    visited.add(next_cell)
                    queue.append(next_cell)
        return False

    def _reachable_space_size_independent(self, start: tuple[int, int], blocked: set[tuple[int, int]], board_size: int) -> int:
        if start in blocked:
            return 0
        visited = {start}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for nx, ny in self._neighbors_for_independent(curr, board_size):
                next_cell = (nx, ny)
                if next_cell not in visited and next_cell not in blocked:
                    visited.add(next_cell)
                    queue.append(next_cell)
        return len(visited)

    def _free_distance_until_obstacle_fast(self, board_size: int, head: tuple[int, int], direction: int, body: set[tuple[int, int]]) -> int:
        move_x, move_y = DIRECTIONS[direction]
        x, y = head
        distance = 0
        while True:
            x += move_x
            y += move_y
            if x < 0 or x >= board_size or y < 0 or y >= board_size or (x, y) in body:
                return distance
            distance += 1

    def _free_distance_until_obstacle(self, game: "SnakeEnv", head: tuple[int, int], direction: int, body: set[tuple[int, int]]) -> int:
        return self._free_distance_until_obstacle_fast(game.board_size, head, direction, body)

    def _will_collide(self, game: "SnakeEnv", point: tuple[int, int], body: set[tuple[int, int]]) -> bool:
        return self._will_collide_fast(game.board_size, point, body)

    def _quick_local_area_ratio_fast(self, board_size: int, head: tuple[int, int], blocked_cells: set[tuple[int, int]]) -> float:
        distances = (
            self._free_distance_until_obstacle_fast(board_size, head, 1, blocked_cells),
            self._free_distance_until_obstacle_fast(board_size, head, 3, blocked_cells),
            self._free_distance_until_obstacle_fast(board_size, head, 2, blocked_cells),
            self._free_distance_until_obstacle_fast(board_size, head, 0, blocked_cells),
        )
        return clamp(sum(distances) / max(1.0, 4.0 * board_size), 0.0, 1.0)

    def _quick_local_area_ratio(self, game: "SnakeEnv", head: tuple[int, int], blocked_cells: set[tuple[int, int]]) -> float:
        return self._quick_local_area_ratio_fast(game.board_size, head, blocked_cells)

    def _should_use_full_floodfill(self, game: "SnakeEnv", periodic_tick: int, body: set[tuple[int, int]] | None = None) -> bool:
        snake = game.snake
        head = snake[0]
        board_size = game.board_size
        if body is None:
            body = set(snake[1:])
        fill_ratio = len(snake) / max(1.0, float(board_size * board_size))
        if fill_ratio >= SMART_LONG_SNAKE_FILL_THRESHOLD * 0.70:
            return True

        direction = game.direction
        left_dir = (direction - 1) % 4
        right_dir = (direction + 1) % 4
        nearby_body_or_wall = (
            self._will_collide_fast(board_size, self._next_head_from(head, direction), body) or
            self._will_collide_fast(board_size, self._next_head_from(head, left_dir), body) or
            self._will_collide_fast(board_size, self._next_head_from(head, right_dir), body)
        )
        if nearby_body_or_wall:
            return True

        local_space = min(
            self._free_distance_until_obstacle_fast(board_size, head, direction, body),
            self._free_distance_until_obstacle_fast(board_size, head, left_dir, body),
            self._free_distance_until_obstacle_fast(board_size, head, right_dir, body),
        )
        if local_space <= 2:
            return True
        return periodic_tick % 4 == 0

    def _simulate_relative_action_features(
        self,
        game: "SnakeEnv",
        action: int,
        use_full: bool,
        *,
        snake: list[tuple[int, int]] | None = None,
        body_without_tail: set[tuple[int, int]] | None = None,
        full_body: set[tuple[int, int]] | None = None,
    ) -> list[float]:
        if snake is None:
            snake = game.snake
        head = snake[0]
        board_size = game.board_size
        new_direction = relative_to_absolute(game.direction, action)
        new_head = self._next_head_from(head, new_direction)
        ate_food = new_head == game.food
        board_cells = max(1, board_size * board_size)
        if body_without_tail is None:
            body_without_tail = set(snake[:-1])
        if full_body is None:
            full_body = set(snake)
        collision_body = full_body if ate_food else body_without_tail
        if self._will_collide_fast(board_size, new_head, collision_body):
            return [1.0, 0.0, 0.0, 0.0]

        blocked_cells = set(full_body if ate_food else body_without_tail)
        if use_full:
            area_count = self._reachable_space_size_independent(new_head, blocked_cells, board_size)
            area_ratio = area_count / float(board_cells)
            food_reachable = 1.0 if self._has_path_independent(new_head, game.food, blocked_cells, board_size) else 0.0

            if len(snake) <= 1:
                tail_target = new_head
            elif ate_food:
                tail_target = snake[-1]
            else:
                tail_target = snake[-2]
            tail_blocked = set(blocked_cells)
            if not ate_food:
                tail_blocked.discard(tail_target)
            tail_reachable = 1.0 if self._has_path_independent(new_head, tail_target, tail_blocked, board_size) else 0.0
        else:
            area_ratio = self._quick_local_area_ratio_fast(board_size, new_head, blocked_cells)
            food_reachable = 1.0
            tail_reachable = 1.0 if area_ratio > 0.08 else 0.0
        return [0.0, food_reachable, tail_reachable, area_ratio]

    def build_state(self, game: "SnakeEnv") -> np.ndarray:
        snake = game.snake
        head = snake[0]
        body = set(snake[1:])
        body_without_tail = set(snake[:-1])
        full_body = set(snake)
        food_x, food_y = game.food
        direction = game.direction
        left_dir = (direction - 1) % 4
        right_dir = (direction + 1) % 4
        back_dir = (direction + 2) % 4
        board_size = game.board_size
        max_axis = max(1, board_size)

        dx_food = food_x - head[0]
        dy_food = food_y - head[1]
        direction_x, direction_y = DIRECTIONS[direction]
        right_x, right_y = DIRECTIONS[right_dir]
        forward_component = dx_food * direction_x + dy_food * direction_y
        lateral_component = dx_food * right_x + dy_food * right_y
        fill_ratio = len(snake) / max(1.0, float(board_size * board_size))
        long_snake_flag = float(fill_ratio >= SMART_LONG_SNAKE_FILL_THRESHOLD)

        self.lookahead_counter += 1
        use_full_lookahead = self._should_use_full_floodfill(game, self.lookahead_counter, body)

        state = np.empty(SMART_STATE_SIZE, dtype=np.float32)
        state[0] = float(head[0] < food_x)
        state[1] = float(head[0] > food_x)
        state[2] = float(head[1] < food_y)
        state[3] = float(head[1] > food_y)
        state[4] = float(direction == 0)
        state[5] = float(direction == 2)
        state[6] = float(direction == 3)
        state[7] = float(direction == 1)
        state[8] = float(self._will_collide_fast(board_size, (head[0] + 1, head[1]), body))
        state[9] = float(self._will_collide_fast(board_size, (head[0] - 1, head[1]), body))
        state[10] = float(self._will_collide_fast(board_size, (head[0], head[1] + 1), body))
        state[11] = float(self._will_collide_fast(board_size, (head[0], head[1] - 1), body))
        state[12] = float(self._will_collide_fast(board_size, self._next_head_from(head, direction), body))
        state[13] = float(self._will_collide_fast(board_size, self._next_head_from(head, left_dir), body))
        state[14] = float(self._will_collide_fast(board_size, self._next_head_from(head, right_dir), body))
        state[15] = normalize_signed(self._free_distance_until_obstacle_fast(board_size, head, direction, body), max_axis)
        state[16] = normalize_signed(self._free_distance_until_obstacle_fast(board_size, head, left_dir, body), max_axis)
        state[17] = normalize_signed(self._free_distance_until_obstacle_fast(board_size, head, right_dir, body), max_axis)
        state[18] = normalize_signed(self._free_distance_until_obstacle_fast(board_size, head, back_dir, body), max_axis)
        state[19] = normalize_signed(dx_food, board_size)
        state[20] = normalize_signed(dy_food, board_size)
        state[21] = normalize_signed(forward_component, max_axis)
        state[22] = normalize_signed(lateral_component, max_axis)
        state[23] = normalize_signed(manhattan(head, game.food), board_size + board_size)
        state[24] = normalize_signed(len(snake), board_size * board_size)
        state[25] = normalize_signed(min(head[0], board_size - 1 - head[0], head[1], board_size - 1 - head[1]), max_axis)

        offset = 26
        for action in range(len(MODEL_ACTION_LABELS)):
            features = self._simulate_relative_action_features(
                game, action, use_full_lookahead,
                snake=snake, body_without_tail=body_without_tail, full_body=full_body,
            )
            state[offset : offset + 4] = features
            offset += 4
        state[38] = fill_ratio
        state[39] = long_snake_flag
        return state

    def select_action(self, game: "SnakeEnv") -> tuple[int, np.ndarray, bool, int]:
        if not self.loaded or self.model is None:
            q_values = np.zeros(len(MODEL_ACTION_LABELS), dtype=np.float32)
            return 0, q_values, False, 0
        state = self.build_state(game)
        try:
            output = self.model(state.reshape(1, -1), training=False)
            if hasattr(output, "numpy"):
                output = output.numpy()
            q_values = np.asarray(output[0], dtype=np.float32)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.loaded = False
            q_values = np.zeros(len(MODEL_ACTION_LABELS), dtype=np.float32)
        action = int(np.argmax(q_values))
        return action, q_values, False, action