from __future__ import annotations

from collections import deque
import random
from typing import Any, Deque, Literal, TypedDict
import numpy as np

from backend_config import (
    APPLE_REWARD,
    STARTUP_BOARD_SIZE,
    DIRECTIONS,
    ENDGAME_FILL_RATIO,
    DEATH_PENALTY,
    hunger_budget_for,
)

# Типы для причин завершения игры
GameEndReason = Literal[
    "new_game",
    "wall_collision",
    "body_collision",
    "apple",
    "victory",
    "restart_pending",
    "shutdown_pending",
]

PenaltyReason = Literal[
    "wall_collision",
    "body_collision",
    "looping",
]


class AppleOpportunitySafe(TypedDict):
    adjacent: Literal[True]
    action: int
    action_label: str
    safe: bool
    verdict: Literal["blocked", "safe"]


class AppleOpportunityNone(TypedDict):
    adjacent: Literal[False]
    action: None
    safe: Literal[False]
    verdict: Literal["none"]


AppleOpportunity = AppleOpportunitySafe | AppleOpportunityNone


class StepInfo(TypedDict):
    score: int
    steps: int
    reason: GameEndReason


class StepFullInfo(StepInfo):
    penalty_reasons: list[PenaltyReason]
    direction: int
    apple_opportunity: AppleOpportunity

# Глобальные кэши для оптимизации
_BOARD_NEIGHBOR_CACHE: dict[int, dict[tuple[int, int], tuple[tuple[int, int], ...]]] = {}


def board_neighbors(board_size: int) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    cached = _BOARD_NEIGHBOR_CACHE.get(board_size)
    if cached is not None:
        return cached
    neighbors: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for y in range(board_size):
        for x in range(board_size):
            cells: list[tuple[int, int]] = []
            for move_x, move_y in DIRECTIONS:
                next_x, next_y = x + move_x, y + move_y
                if 0 <= next_x < board_size and 0 <= next_y < board_size:
                    cells.append((next_x, next_y))
            neighbors[(x, y)] = tuple(cells)
    _BOARD_NEIGHBOR_CACHE[board_size] = neighbors
    return neighbors


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def relative_to_absolute(current_direction: int, action: int) -> int:
    if action == 0:
        return current_direction
    if action == 1:
        return (current_direction - 1) % 4
    return (current_direction + 1) % 4





class SnakeEnv:
    def __init__(self, board_size: int = STARTUP_BOARD_SIZE) -> None:
        self.board_size = board_size
        self._path_cache: dict[tuple[str, ...], Any] = {}
        self._path_cache_max_size = 10000
        self._obs_cache_step: int = -1
        self._obs_cache_vec: np.ndarray | None = None
        self.reset()

    def _clear_path_cache(self) -> None:
        self._path_cache.clear()

    def _clear_observation_cache(self) -> None:
        self._obs_cache_step = -1
        self._obs_cache_vec = None

    def _check_cache_size(self) -> None:
        if len(self._path_cache) > self._path_cache_max_size:
            self._path_cache.clear()


    def _neighbors_for(self, cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        return board_neighbors(self.board_size).get(cell, ())

    def reset(self) -> np.ndarray:
        self._clear_path_cache()
        self._clear_observation_cache()
        # Безопасно сохраняем значение ДО сброса (0 при первом запуске, реальное число при смерти)
        self.last_steps_without_food = getattr(self, 'steps_without_food', 0)
        center = self.board_size // 2
        self.direction = 1
        self.score = 0
        self.step_count = 0
        self.steps_without_food = 0
        self.max_steps_without_food = 0
        self.done = False
        self.last_reward = 0.0
        self.last_reason = "new_game"
        self.last_penalty_reasons: list[str] = []
        self.eaten_food_trail: Deque[tuple[int, int]] = deque(maxlen=24)
        self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        self.recent_heads = deque([self.snake[0]], maxlen=18)
        self.recent_turns = deque(maxlen=8)
        self.food = self._spawn_food()
        return self.get_observation_grid()

    def _spawn_food(self) -> tuple[int, int]:
        occupied = set(self.snake)
        free_cells = [
            (x, y)
            for y in range(self.board_size)
            for x in range(self.board_size)
            if (x, y) not in occupied
        ]
        return random.choice(free_cells) if free_cells else (0, 0)

    def _normalize_direction(self, direction: int | None) -> int:
        if direction is None:
            return self.direction
        opposite = (self.direction + 2) % 4
        if len(self.snake) > 1 and direction == opposite:
            return self.direction
        return direction

    def _would_hit_self(self, next_head: tuple[int, int], will_grow: bool) -> bool:
        body = self.snake[:-1] if not will_grow else self.snake
        return next_head in body

    def _next_head_for_direction(self, direction: int) -> tuple[int, int]:
        head_x, head_y = self.snake[0]
        move_x, move_y = DIRECTIONS[direction]
        return head_x + move_x, head_y + move_y

    def _danger_for_relative_action(self, relative_action: int) -> float:
        absolute_direction = relative_to_absolute(self.direction, relative_action)
        next_head = self._next_head_for_direction(absolute_direction)
        will_eat = next_head == self.food
        hits_wall = not (
            0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size
        )
        hits_body = self._would_hit_self(next_head, will_eat)
        return 1.0 if hits_wall or hits_body else 0.0

    def relative_move_risks(self) -> dict[str, dict[str, float]]:
        return {
            "forward": {
                "danger": self._danger_for_relative_action(0),
                "tail_blocked": self._is_tail_blocked_after_relative_action(0),
                "space_tight": self._is_space_tight_after_relative_action(0),
            },
            "left": {
                "danger": self._danger_for_relative_action(1),
                "tail_blocked": self._is_tail_blocked_after_relative_action(1),
                "space_tight": self._is_space_tight_after_relative_action(1),
            },
            "right": {
                "danger": self._danger_for_relative_action(2),
                "tail_blocked": self._is_tail_blocked_after_relative_action(2),
                "space_tight": self._is_space_tight_after_relative_action(2),
            },
        }

    def _is_endgame(self) -> bool:
        return len(self.snake) >= self.board_size * self.board_size * ENDGAME_FILL_RATIO

    def step(
        self, direction: int
    ) -> tuple[np.ndarray, float, bool, StepFullInfo]:
        if self.done:
            return (
                self.get_observation_grid(),
                self.last_reward,
                True,
                {"score": self.score, "steps": self.step_count, "reason": self.last_reason},  # type: ignore[typeddict-item]
            )

        chosen_direction = self._normalize_direction(direction)
        previous_direction = self.direction
        self.direction = chosen_direction

        head, move = self.snake[0], DIRECTIONS[chosen_direction]
        next_head = (head[0] + move[0], head[1] + move[1])
        will_eat = next_head == self.food

        # В начале step(), до self.step_count += 1:
        self._clear_observation_cache()

        reward = 0.0
        reason = "step"
        penalty_reasons = []

        self.step_count += 1
        self.steps_without_food += 1
        self.max_steps_without_food = max(self.max_steps_without_food, self.steps_without_food)

        hits_wall = not (
            0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size
        )
        hunger_limit = hunger_budget_for(self.score, self.board_size)
        if self.steps_without_food > hunger_limit:
            self.done = True
            reward = -DEATH_PENALTY
            reason = "starvation"
            penalty_reasons.append("looping")
        elif hits_wall or self._would_hit_self(next_head, will_eat):
            self.done = True
            reward = -DEATH_PENALTY
            reason = "wall_collision" if hits_wall else "body_collision"
            penalty_reasons.append(reason)
        else:
            self.snake.insert(0, next_head)
            if will_eat:
                self.score += 1
                self.steps_without_food = 0
                reward = APPLE_REWARD
                reason = "apple"
                self.eaten_food_trail.append(self.food)
                self.recent_turns.clear()
                if len(self.snake) >= self.board_size * self.board_size:
                    self.done = True
                    reward = reward + 10.0
                    reason = "victory"
                else:
                    self.food = self._spawn_food()
            else:
                tail = self.snake.pop()
                if tail in self.eaten_food_trail:
                    self.eaten_food_trail.remove(tail)

                # Штрафы за петли
                loop_penalty = 0.0
                revisit_count = sum(1 for p in self.recent_heads if p == next_head)
                if revisit_count >= 2:
                    loop_penalty += 0.03 * (revisit_count - 1)

                turn_delta = (chosen_direction - previous_direction) % 4
                self.recent_turns.append(
                    1 if turn_delta == 1 else (-1 if turn_delta == 3 else 0)
                )

                if loop_penalty > 0:
                    reward -= min(0.35, loop_penalty)
                    penalty_reasons.append("looping")

                # Счётчик голода остаётся только для UI/метрик: без смерти от голода.
            self.recent_heads.append(next_head)

        self.last_reward = reward
        self.last_reason = reason
        self.last_penalty_reasons = penalty_reasons
        self._check_cache_size()

        observation = self.get_observation_grid()
        return (
            observation,
            reward,
            self.done,
            {
                "score": self.score,
                "steps": self.step_count,
                "reason": reason,  # type: ignore[typeddict-item]
                "penalty_reasons": list(penalty_reasons),
                "direction": chosen_direction,
                "apple_opportunity": None,
            },
        )

    def _has_path(self, start: tuple[int, int], target: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
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
            for next_cell in self._neighbors_for(curr):
                if next_cell not in visited and next_cell not in blocked:
                    visited.add(next_cell)
                    queue.append(next_cell)
        return False

    def _reachable_space_size(self, start: tuple[int, int], blocked: set[tuple[int, int]]) -> int:
        if start in blocked:
            return 0
        visited = {start}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for next_cell in self._neighbors_for(curr):
                if next_cell not in visited and next_cell not in blocked:
                    visited.add(next_cell)
                    queue.append(next_cell)
        return len(visited)

    def _is_tail_blocked_after_relative_action(self, relative_action: int) -> float:
        absolute_direction = relative_to_absolute(self.direction, relative_action)
        next_head = self._next_head_for_direction(absolute_direction)
        will_eat = next_head == self.food
        hits_wall = not (0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size)
        if hits_wall or self._would_hit_self(next_head, will_eat):
            return 1.0

        # Симулируем тело после хода: при обычном ходе хвост освобождается,
        # при яблоке остаётся на месте и продолжает блокировать клетку.
        if will_eat:
            simulated_snake = [next_head] + self.snake
        else:
            simulated_snake = [next_head] + self.snake[:-1]

        tail_target = simulated_snake[-1]
        blocked = set(simulated_snake[1:])
        if not will_eat:
            blocked.discard(tail_target)
        has_path = self._has_path(next_head, tail_target, blocked)
        return 0.0 if has_path else 1.0

    def _is_space_tight_after_relative_action(self, relative_action: int) -> float:
        """Возвращает 1.0 если доступное пространство после хода меньше длины змеи."""
        absolute_direction = relative_to_absolute(self.direction, relative_action)
        next_head = self._next_head_for_direction(absolute_direction)
        will_eat = next_head == self.food
        hits_wall = not (0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size)
        if hits_wall or self._would_hit_self(next_head, will_eat):
            return 1.0
        if will_eat:
            simulated_snake = [next_head] + self.snake
        else:
            simulated_snake = [next_head] + self.snake[:-1]
        blocked = set(simulated_snake[1:])
        space_size = self._reachable_space_size(next_head, blocked)
        return 1.0 if space_size < len(simulated_snake) else 0.0



    def get_extra_signals(self, target_size: int = STARTUP_BOARD_SIZE) -> np.ndarray:
        # Кэш на текущий шаг: если фронтенд или логгер вызовут метод повторно,
        # мы вернём готовый вектор без повторных BFS.
        if self.step_count == self._obs_cache_step and self._obs_cache_vec is not None:
            return self._obs_cache_vec

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dx, dy = food_x - head_x, food_y - head_y

        if self.direction == 0:
            fwd, rgt = -dy, dx
        elif self.direction == 1:
            fwd, rgt = dx, dy
        elif self.direction == 2:
            fwd, rgt = dy, -dx
        else:
            fwd, rgt = -dx, -dy

        tail_x, tail_y = self.snake[-1]
        tx, ty = tail_x - head_x, tail_y - head_y

        if self.direction == 0:
            t_fwd, t_rgt = -ty, tx
        elif self.direction == 1:
            t_fwd, t_rgt = tx, ty
        elif self.direction == 2:
            t_fwd, t_rgt = ty, -tx
        else:
            t_fwd, t_rgt = -tx, -ty

        # Порядок относительных сигналов синхронизирован с MODEL_ACTION_LABELS:
        # forward, left, right. Behind-сигналы идут после этих направлений.
        signals = [
            1.0 if fwd > 0 else 0.0,
            1.0 if rgt < 0 else 0.0,
            1.0 if rgt > 0 else 0.0,
            1.0 if fwd < 0 else 0.0,  # food_behind

            1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 0)) == self.food else 0.0,
            1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 1)) == self.food else 0.0,
            1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 2)) == self.food else 0.0,

            self._danger_for_relative_action(0),
            self._danger_for_relative_action(1),
            self._danger_for_relative_action(2),

            1.0 if t_fwd > 0 else 0.0,  # tail_ahead
            1.0 if t_rgt < 0 else 0.0,  # tail_left
            1.0 if t_rgt > 0 else 0.0,  # tail_right
            1.0 if t_fwd < 0 else 0.0,  # tail_behind

            self._is_tail_blocked_after_relative_action(0),  # tail_blocked_forward
            self._is_tail_blocked_after_relative_action(1),  # tail_blocked_left
            self._is_tail_blocked_after_relative_action(2),  # tail_blocked_right

            self._is_space_tight_after_relative_action(0),  # space_tight_forward
            self._is_space_tight_after_relative_action(1),  # space_tight_left
            self._is_space_tight_after_relative_action(2),  # space_tight_right
        ]
        self._obs_cache_step = self.step_count
        self._obs_cache_vec = np.array(signals, dtype=np.float32)
        return self._obs_cache_vec

    def get_observation_grid(self, target_size: int = STARTUP_BOARD_SIZE) -> np.ndarray:
        # Возвращаем ТОЛЬКО вектор сигналов. Карта и гибридные режимы удалены.
        return self.get_extra_signals(target_size)

    def export_state(self) -> dict[str, object]:
        return {
            "board_size": self.board_size,
            "direction": self.direction,
            "score": self.score,
            "step_count": self.step_count,
            "steps_without_food": self.steps_without_food,
            "last_steps_without_food": getattr(self, 'last_steps_without_food', 0),  # <-- ДОБАВИТЬ ЭТУ СТРОКУ
            "max_steps_without_food": self.max_steps_without_food,
            "done": self.done,
            "last_reward": self.last_reward,
            "last_reason": self.last_reason,
            "last_penalty_reasons": list(self.last_penalty_reasons),
            "eaten_food_trail": [[x, y] for x, y in self.eaten_food_trail],
            "snake": [[x, y] for x, y in self.snake],
            "food": [self.food[0], self.food[1]],
            "recent_heads": [[x, y] for x, y in self.recent_heads],
            "recent_turns": list(self.recent_turns),
        }

    def load_state(self, payload: dict[str, object], target_size: int = STARTUP_BOARD_SIZE) -> np.ndarray:
        try:
            self.board_size = int(payload.get("board_size", STARTUP_BOARD_SIZE))
            self.snake = [(int(p[0]), int(p[1])) for p in payload.get("snake", [])]
            self.food = (int(payload["food"][0]), int(payload["food"][1]))
            self.direction = int(payload.get("direction", 1))
            self.score = int(payload.get("score", 0))
            self.step_count = int(payload.get("step_count", 0))
            self.steps_without_food = int(payload.get("steps_without_food", 0))
            self.done = bool(payload.get("done", False))
            self.last_reason = str(payload.get("last_reason", "new_game"))

            eft = payload.get("eaten_food_trail", [])
            self.eaten_food_trail = deque([(int(p[0]), int(p[1])) for p in eft], maxlen=24)
            rh = payload.get("recent_heads", [])
            self.recent_heads = deque([(int(p[0]), int(p[1])) for p in rh], maxlen=18)
            rt = payload.get("recent_turns", [])
            self.recent_turns = deque([int(t) for t in rt], maxlen=8)

            self._clear_path_cache()
            self._clear_observation_cache()
            return self.get_observation_grid(target_size)
        except Exception:
            return self.reset()
