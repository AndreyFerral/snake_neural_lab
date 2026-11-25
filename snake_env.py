from __future__ import annotations

from collections import deque
import random
from typing import Any, Deque, Literal, TypedDict
import numpy as np

from backend_config import (
    APPLE_REWARD,
    STARTUP_BOARD_SIZE,
    DIRECTIONS,
    DEATH_PENALTY,
    PPO_MAP_SIZE,
    PPO_MAP_CHANNELS,
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


class StepInfo(TypedDict):
    score: int
    steps: int
    reason: GameEndReason


class StepFullInfo(StepInfo):
    penalty_reasons: list[PenaltyReason]
    direction: int

# Глобальные кэши для оптимизации
_BOARD_NEIGHBOR_CACHE: dict[int, dict[tuple[int, int], tuple[tuple[int, int], ...]]] = {}
_FLAT_NEIGHBOR_CACHE: dict[int, tuple[tuple[int, ...], ...]] = {}
_EGO_OFFSET_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _ego_offsets(win: int, direction: int) -> tuple[np.ndarray, np.ndarray]:
    key = (win, direction)
    cached = _EGO_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached
    radius = win // 2
    fx, fy = DIRECTIONS[direction]
    rx, ry = DIRECTIONS[(direction + 1) % 4]
    rows = np.arange(win, dtype=np.int32)[:, None]
    cols = np.arange(win, dtype=np.int32)[None, :]
    fwd = radius - rows
    rgt = cols - radius
    dx = (fwd * fx + rgt * rx).astype(np.int32)
    dy = (fwd * fy + rgt * ry).astype(np.int32)
    _EGO_OFFSET_CACHE[key] = (dx, dy)
    return dx, dy


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


def flat_board_neighbors(board_size: int) -> tuple[tuple[int, ...], ...]:
    cached = _FLAT_NEIGHBOR_CACHE.get(board_size)
    if cached is not None:
        return cached
    flat: list[tuple[int, ...]] = []
    for y in range(board_size):
        for x in range(board_size):
            cells: list[int] = []
            for move_x, move_y in DIRECTIONS:
                next_x, next_y = x + move_x, y + move_y
                if 0 <= next_x < board_size and 0 <= next_y < board_size:
                    cells.append(next_y * board_size + next_x)
            flat.append(tuple(cells))
    result = tuple(flat)
    _FLAT_NEIGHBOR_CACHE[board_size] = result
    return result


def relative_to_absolute(current_direction: int, action: int) -> int:
    if action == 0:
        return current_direction
    if action == 1:
        return (current_direction - 1) % 4
    return (current_direction + 1) % 4





class SnakeEnv:
    def __init__(self, board_size: int = STARTUP_BOARD_SIZE) -> None:
        self.board_size = board_size
        self._obs_cache_step: int = -1
        self._obs_cache_vec: np.ndarray | None = None
        self._area_ratio_cache_step: int = -1
        self._area_ratio_cache_vec: np.ndarray | None = None
        self._ppo_obs_cache_step: int = -1
        self._ppo_obs_cache_vec: np.ndarray | None = None
        self._ego_map_cache_step: int = -1
        self._ego_map_cache_size: int = 0
        self._ego_map_cache_arr: np.ndarray | None = None
        self._shortest_path_cache_step: int = -1
        self._shortest_path_cache_val: list[tuple[int, int]] | None = None
        self._ensure_bfs_buffers(board_size)
        self.reset()

    def _ensure_bfs_buffers(self, board_size: int) -> None:
        total_cells = max(1, board_size * board_size)
        if not hasattr(self, "_bfs_visited") or len(self._bfs_visited) != total_cells:
            self._bfs_visited = [0] * total_cells
            self._bfs_blocked = [0] * total_cells
            self._bfs_parent = [-1] * total_cells
            self._bfs_stamp = 0
            self._block_stamp = 0
            self._flat_neighbors = flat_board_neighbors(board_size)
            self._ego_ttl_buf = np.zeros(total_cells, dtype=np.float32)
            self._ego_reach_buf = np.zeros(total_cells, dtype=np.float32)

    def _clear_observation_cache(self) -> None:
        self._obs_cache_step = -1
        self._obs_cache_vec = None
        self._area_ratio_cache_step = -1
        self._area_ratio_cache_vec = None
        self._ppo_obs_cache_step = -1
        self._ppo_obs_cache_vec = None
        self._ego_map_cache_step = -1
        self._ego_map_cache_size = 0
        self._ego_map_cache_arr = None
        self._shortest_path_cache_step = -1
        self._shortest_path_cache_val = None

    def _neighbors_for(self, cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        return board_neighbors(self.board_size).get(cell, ())

    def reset(self, *, return_obs: bool = True) -> np.ndarray:
        self._clear_observation_cache()
        self._ensure_bfs_buffers(self.board_size)
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
        self.last_penalty_reasons = []
        self.eaten_food_trail = deque(maxlen=24)
        self.tail_track = deque(maxlen=4)
        self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        self.recent_heads = deque([self.snake[0]], maxlen=18)
        self.recent_turns = deque(maxlen=8)
        self.food = self._spawn_food()
        if not return_obs:
            return np.empty(0, dtype=np.float32)
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
        signals = self.get_extra_signals()
        return {
            "forward": {
                "danger": float(signals[8]),
                "tail_blocked": float(signals[15]),
                "space_tight": self._is_space_tight_after_relative_action(0),
            },
            "left": {
                "danger": float(signals[9]),
                "tail_blocked": float(signals[16]),
                "space_tight": self._is_space_tight_after_relative_action(1),
            },
            "right": {
                "danger": float(signals[10]),
                "tail_blocked": float(signals[17]),
                "space_tight": self._is_space_tight_after_relative_action(2),
            },
        }



    def step(
        self, direction: int, *, return_obs: bool = True
    ) -> tuple[np.ndarray, float, bool, StepFullInfo]:
        """
        Выполняет один шаг игры. Возвращает базовую награду среды (яблоко, шаг, смерть).
        
        АРХИТЕКТУРНЫЙ СТАНДАРТ ФОРМИРОВАНИЯ НАГРАД (REWARD SHAPING):
        Физика игровой среды (SnakeEnv) должна оставаться «чистой» и независимой от специфики 
        обучения ИИ. Все дополнительные штрафы и бонусы для агентов (DQN, PPO)
        формируются по следующему единому принципу:
        
        1. Определение константы:
           Любой новый штраф или награда (например, TAIL_LOSS_PENALTY = 0.50) ДОЛЖНЫ быть объявлены 
           как централизованная константа в `backend_config.py` рядом с остальными наградами.
           
        2. Регистрация в UI:
           Добавьте описание и значение в PENALTY_CATALOG или REWARD_CATALOG в `backend_config.py`, 
           используя созданную константу, чтобы интерфейс (Лог лидера, правила) отображал её автоматически.
           
        3. Импорт и Применение в Агентах:
           Импортируйте константу из `backend_config.py` в модули агентов/раннеров (например,
           в `web_backend.py` для DQN или в `ppo_agent.py` для PPO).
           
        4. Расчет рисков:
           Оцените состояние ДО вызова step() с помощью аналитических методов среды:
             - self.relative_move_risks() — возвращает опасности каждого относительного действия.
             - self._is_tail_blocked_after_relative_action(action) — проверяет доступность хвоста.
             - self._is_space_tight_after_relative_action(action) — проверяет тесноту пространства.
           И примените импортированный штраф (например, `reward -= NEW_PENALTY`) к результирующей 
           награде, чтобы она зафиксировалась в логах лидера и использовалась для обучения.
        """
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
                self.recent_heads.clear()
                
                # Dynamic recent_heads history size based on snake length (multiplier 2.5 to fit loops twice)
                new_maxlen = max(18, int(len(self.snake) * 2.5))
                if self.recent_heads.maxlen != new_maxlen:
                    self.recent_heads = deque(maxlen=new_maxlen)

                if len(self.snake) >= self.board_size * self.board_size:
                    self.done = True
                    reward = reward + 10.0
                    reason = "victory"
                else:
                    self.food = self._spawn_food()
            else:
                tail = self.snake.pop()
                self.tail_track.append(tail)
                if tail in self.eaten_food_trail:
                    self.eaten_food_trail.remove(tail)

                # Штрафы за петли
                loop_penalty = 0.0
                revisit_count = sum(1 for p in self.recent_heads if p == next_head)
                if revisit_count >= 1:
                    loop_penalty += 0.03 * revisit_count

                turn_delta = (chosen_direction - previous_direction) % 4
                self.recent_turns.append(
                    1 if turn_delta == 1 else (-1 if turn_delta == 3 else 0)
                )

                if loop_penalty > 0:
                    reward -= min(0.35, loop_penalty)
                    penalty_reasons.append("looping")

                # Счётчик голода учитывается выше для завершения эпизода по starvation.
            self.recent_heads.append(next_head)

        self.last_reward = reward
        self.last_reason = reason
        self.last_penalty_reasons = penalty_reasons

        observation = self.get_observation_grid() if return_obs else np.empty(0, dtype=np.float32)
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
            },
        )

    def _next_bfs_stamp(self) -> int:
        self._bfs_stamp += 1
        if self._bfs_stamp > 1_000_000_000:
            self._bfs_visited = [0] * len(self._bfs_visited)
            self._bfs_stamp = 1
        return self._bfs_stamp

    def _next_block_stamp(self) -> int:
        self._block_stamp += 1
        if self._block_stamp > 1_000_000_000:
            self._bfs_blocked = [0] * len(self._bfs_blocked)
            self._block_stamp = 1
        return self._block_stamp

    def _fast_has_path(self, start_idx: int, target_idx: int, b_stamp: int) -> bool:
        blocked = self._bfs_blocked
        if blocked[start_idx] == b_stamp or blocked[target_idx] == b_stamp:
            return False
        if start_idx == target_idx:
            return True
        stamp = self._next_bfs_stamp()
        visited = self._bfs_visited
        neighbors = self._flat_neighbors
        visited[start_idx] = stamp
        queue = [start_idx]
        qi = 0
        while qi < len(queue):
            curr = queue[qi]
            qi += 1
            for next_idx in neighbors[curr]:
                if next_idx == target_idx:
                    return True
                if visited[next_idx] != stamp and blocked[next_idx] != b_stamp:
                    visited[next_idx] = stamp
                    queue.append(next_idx)
        return False

    def _fast_reachable_space(self, start_idx: int, b_stamp: int, limit: int | None = None) -> int:
        blocked = self._bfs_blocked
        if blocked[start_idx] == b_stamp:
            return 0
        if limit is not None and limit <= 1:
            return 1
        stamp = self._next_bfs_stamp()
        visited = self._bfs_visited
        neighbors = self._flat_neighbors
        visited[start_idx] = stamp
        queue = [start_idx]
        qi = 0
        while qi < len(queue):
            curr = queue[qi]
            qi += 1
            for next_idx in neighbors[curr]:
                if visited[next_idx] != stamp and blocked[next_idx] != b_stamp:
                    visited[next_idx] = stamp
                    queue.append(next_idx)
                    if limit is not None and len(queue) >= limit:
                        return len(queue)
        return len(queue)

    def _fast_empty_components(self, b_stamp: int) -> int:
        blocked = self._bfs_blocked
        visited = self._bfs_visited
        neighbors = self._flat_neighbors
        stamp = self._next_bfs_stamp()
        total = len(blocked)
        comps = 0
        for start in range(total):
            if blocked[start] != b_stamp and visited[start] != stamp:
                comps += 1
                visited[start] = stamp
                queue = [start]
                qi = 0
                while qi < len(queue):
                    curr = queue[qi]
                    qi += 1
                    for nxt in neighbors[curr]:
                        if blocked[nxt] != b_stamp and visited[nxt] != stamp:
                            visited[nxt] = stamp
                            queue.append(nxt)
        return comps

    def _has_path(self, start: tuple[int, int], target: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
        if start in blocked or target in blocked:
            return False
        if start == target:
            return True
        self._ensure_bfs_buffers(self.board_size)
        n = self.board_size
        b_stamp = self._next_block_stamp()
        blk = self._bfs_blocked
        for bx, by in blocked:
            if 0 <= bx < n and 0 <= by < n:
                blk[by * n + bx] = b_stamp
        return self._fast_has_path(start[1] * n + start[0], target[1] * n + target[0], b_stamp)

    def _reachable_space_size(self, start: tuple[int, int], blocked: set[tuple[int, int]], limit: int | None = None) -> int:
        if start in blocked:
            return 0
        self._ensure_bfs_buffers(self.board_size)
        n = self.board_size
        b_stamp = self._next_block_stamp()
        blk = self._bfs_blocked
        for bx, by in blocked:
            if 0 <= bx < n and 0 <= by < n:
                blk[by * n + bx] = b_stamp
        return self._fast_reachable_space(start[1] * n + start[0], b_stamp, limit=limit)

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
        blocked.discard(tail_target)
        has_path = self._has_path(next_head, tail_target, blocked)
        return 0.0 if has_path else 1.0

    def _is_space_tight_after_relative_action(self, relative_action: int) -> float:
        """Возвращает 1.0 если доступное пространство после хода меньше динамического порога (с ограничением снизу)."""
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
        
        free_cells = (self.board_size * self.board_size) - len(simulated_snake)
        # Динамический порог с вычитанием 6 клеток и ограничением снизу в 6 клеток
        thresh = max(6, len(simulated_snake) - 6)
        threshold = min(thresh, free_cells)
        
        space_size = self._reachable_space_size(next_head, blocked, limit=threshold)
        return 1.0 if space_size < threshold else 0.0

    def _is_currently_partitioned(self) -> bool:
        """Возвращает True, если еда недоступна из головы в текущем состоянии."""
        head = self.snake[0]
        blocked = set(self.snake[1:])  # тело без головы
        return not self._has_path(head, self.food, blocked)

    def _is_food_reachable(self) -> float:
        """Возвращает 1.0, если путь от головы до еды существует, иначе 0.0."""
        head = self.snake[0]
        food = self.food
        if head == food:
            return 1.0
        self._ensure_bfs_buffers(self.board_size)
        n = self.board_size
        b_stamp = self._next_block_stamp()
        blk = self._bfs_blocked
        for bx, by in self.snake[1:]:
            blk[by * n + bx] = b_stamp
        return 1.0 if self._fast_has_path(head[1] * n + head[0], food[1] * n + food[0], b_stamp) else 0.0

    def _is_space_partitioned_after_relative_action(self, relative_action: int) -> float:
        """Возвращает 1.0 если после хода еда становится недоступна из головы."""
        absolute_direction = relative_to_absolute(self.direction, relative_action)
        next_head = self._next_head_for_direction(absolute_direction)
        will_eat = next_head == self.food
        if will_eat:
            return 0.0
        hits_wall = not (0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size)
        if hits_wall or self._would_hit_self(next_head, will_eat):
            return 1.0
        simulated_snake = [next_head] + self.snake[:-1]
        blocked = set(simulated_snake[1:])  # тело без головы
        if not self._has_path(next_head, self.food, blocked):
            return 1.0
        return 0.0

    def _shortest_path_to_tail(self) -> list[tuple[int, int]] | None:
        if self.step_count == self._shortest_path_cache_step:
            return self._shortest_path_cache_val

        head = self.snake[0]
        tail = self.snake[-1]
        if head == tail:
            self._shortest_path_cache_step = self.step_count
            self._shortest_path_cache_val = []
            return []

        self._ensure_bfs_buffers(self.board_size)
        n = self.board_size
        b_stamp = self._next_block_stamp()
        blk = self._bfs_blocked
        for bx, by in self.snake[1:-1]:
            blk[by * n + bx] = b_stamp

        head_idx = head[1] * n + head[0]
        tail_idx = tail[1] * n + tail[0]
        stamp = self._next_bfs_stamp()
        visited = self._bfs_visited
        parent = self._bfs_parent
        neighbors = self._flat_neighbors
        visited[head_idx] = stamp
        parent[head_idx] = -1
        queue = [head_idx]
        qi = 0
        found = False
        while qi < len(queue):
            curr = queue[qi]
            qi += 1
            for next_idx in neighbors[curr]:
                if visited[next_idx] != stamp and blk[next_idx] != b_stamp:
                    visited[next_idx] = stamp
                    parent[next_idx] = curr
                    if next_idx == tail_idx:
                        found = True
                        break
                    queue.append(next_idx)
            if found:
                break

        path_found: list[tuple[int, int]] | None = None
        if found:
            rev_path: list[tuple[int, int]] = []
            cur = tail_idx
            while cur != head_idx and cur != -1:
                rev_path.append((cur % n, cur // n))
                cur = parent[cur]
            rev_path.reverse()
            path_found = rev_path

        self._shortest_path_cache_step = self.step_count
        self._shortest_path_cache_val = path_found
        return path_found

    def _is_tail_close_after_relative_action(self, relative_action: int) -> float:
        absolute_direction = relative_to_absolute(self.direction, relative_action)
        next_head = self._next_head_for_direction(absolute_direction)
        
        # Наказываем, если следующая клетка головы — это текущий хвост ИЛИ лежит на недавних следах хвоста
        if next_head != self.snake[-1] and next_head not in self.tail_track:
            return 0.0
            
        path = self._shortest_path_to_tail()
        if not path or len(path) >= 3:
            return 0.0
            
        head = self.snake[0]
        p_1 = path[0]
        dx = p_1[0] - head[0]
        dy = p_1[1] - head[1]
        path_direction = DIRECTIONS.index((dx, dy))
        
        if absolute_direction == path_direction:
            # Не штрафуем, если нет безопасных альтернатив (вынужденный ход для выживания)
            alternatives = [a for a in (0, 1, 2) if a != relative_action]
            has_safe_alternative = False
            for alt_act in alternatives:
                if self._danger_for_relative_action(alt_act) < 0.5:
                    has_safe_alternative = True
                    break
            if not has_safe_alternative:
                return 0.0
            return 1.0
        return 0.0



    def get_extra_signals(self, target_size: int = STARTUP_BOARD_SIZE) -> np.ndarray:
        # Кэш на текущий шаг: если фронтенд или логгер вызовут метод повторно,
        # мы вернём готовый вектор без повторных BFS.
        if self.step_count == self._obs_cache_step and self._obs_cache_vec is not None:
            return self._obs_cache_vec

        n = self.board_size
        board_cells = n * n
        snake = self.snake
        snake_len = len(snake)
        head_x, head_y = snake[0]
        food_x, food_y = self.food
        dx, dy = food_x - head_x, food_y - head_y

        direction = self.direction
        if direction == 0:
            fwd, rgt = -dy, dx
        elif direction == 1:
            fwd, rgt = dx, dy
        elif direction == 2:
            fwd, rgt = dy, -dx
        else:
            fwd, rgt = -dx, -dy

        tail_x, tail_y = snake[-1]
        tx, ty = tail_x - head_x, tail_y - head_y

        if direction == 0:
            t_fwd, t_rgt = -ty, tx
        elif direction == 1:
            t_fwd, t_rgt = tx, ty
        elif direction == 2:
            t_fwd, t_rgt = ty, -tx
        else:
            t_fwd, t_rgt = -tx, -ty

        self._ensure_bfs_buffers(n)
        b_stamp = self._next_block_stamp()
        blk = self._bfs_blocked
        visited = self._bfs_visited
        neighbors = self._flat_neighbors

        occupied_no_tail = 0
        for bx, by in snake[:-1]:
            fidx = by * n + bx
            if blk[fidx] != b_stamp:
                blk[fidx] = b_stamp
                occupied_no_tail += 1

        head_idx = head_y * n + head_x
        food_idx = food_y * n + food_x
        tail_idx = tail_y * n + tail_x
        empty_no_tail = board_cells - occupied_no_tail

        # 1. Объединённый BFS от головы: одновременно вычисляет food_reachable и current_comps
        saved_tail_cur = blk[tail_idx]
        tail_was_free = saved_tail_cur != b_stamp
        blk[tail_idx] = b_stamp
        total_empty_cur = empty_no_tail - (1 if tail_was_free else 0)

        stamp = self._next_bfs_stamp()
        current_comps = 0
        visited_empty_cur = 0
        if snake_len == 1:
            food_reachable = 1.0
            for start in range(board_cells):
                if blk[start] != b_stamp and visited[start] != stamp:
                    current_comps += 1
                    visited[start] = stamp
                    q = [start]
                    qi = 0
                    while qi < len(q):
                        curr = q[qi]
                        qi += 1
                        for nxt in neighbors[curr]:
                            if blk[nxt] != b_stamp and visited[nxt] != stamp:
                                visited[nxt] = stamp
                                q.append(nxt)
                    visited_empty_cur += len(q)
                    if visited_empty_cur >= total_empty_cur:
                        break
        else:
            # Начинаем BFS с пустых соседей головы (не переходя через саму голову)
            for nb in neighbors[head_idx]:
                if blk[nb] != b_stamp and visited[nb] != stamp:
                    current_comps += 1
                    visited[nb] = stamp
                    q = [nb]
                    qi = 0
                    while qi < len(q):
                        curr = q[qi]
                        qi += 1
                        for nxt in neighbors[curr]:
                            if blk[nxt] != b_stamp and visited[nxt] != stamp:
                                visited[nxt] = stamp
                                q.append(nxt)
                    visited_empty_cur += len(q)

            if head_idx == food_idx:
                food_reachable = 1.0
            elif head_x == tail_x and head_y == tail_y:
                food_reachable = 0.0
            else:
                food_reachable = 1.0 if visited[food_idx] == stamp else 0.0

            if visited_empty_cur < total_empty_cur:
                for start in range(board_cells):
                    if blk[start] != b_stamp and visited[start] != stamp:
                        current_comps += 1
                        visited[start] = stamp
                        q = [start]
                        qi = 0
                        while qi < len(q):
                            curr = q[qi]
                            qi += 1
                            for nxt in neighbors[curr]:
                                if blk[nxt] != b_stamp and visited[nxt] != stamp:
                                    visited[nxt] = stamp
                                    q.append(nxt)
                        visited_empty_cur += len(q)
                        if visited_empty_cur >= total_empty_cur:
                            break

        blk[tail_idx] = saved_tail_cur
        board_split = 1.0 if current_comps > 1 else 0.0

        apple_steps = [0.0, 0.0, 0.0]
        dangers = [0.0, 0.0, 0.0]
        tail_blocked = [0.0, 0.0, 0.0]
        partitions = [0.0, 0.0, 0.0]
        pocket_splits = [0.0, 0.0, 0.0]
        pocket_merges = [0.0, 0.0, 0.0]
        area_ratios = np.zeros(3, dtype=np.float32)

        tail_track = self.tail_track
        endgame = 1.0 if (snake_len / board_cells) >= 0.75 else 0.0
        hunger_limit = max(1.0, float(hunger_budget_for(self.score, n)))
        hunger_ratio = float(self.steps_without_food) / hunger_limit
        hunger_high = 1.0 if hunger_ratio >= 0.65 else 0.0
        hunger_critical = 1.0 if hunger_ratio >= 0.85 else 0.0

        # Предвычисляем next_head для всех 3 относительных действий
        next_coords: list[tuple[int, int, int, bool, bool]] = []
        for act in (0, 1, 2):
            abs_dir = direction if act == 0 else ((direction - 1) % 4 if act == 1 else (direction + 1) % 4)
            mx, my = DIRECTIONS[abs_dir]
            nx, ny = head_x + mx, head_y + my
            will_eat = (nx == food_x and ny == food_y)
            if will_eat:
                apple_steps[act] = 1.0
            hits_wall = not (0 <= nx < n and 0 <= ny < n)
            if hits_wall:
                is_danger = True
                next_idx = -1
            else:
                next_idx = ny * n + nx
                hits_body = (blk[next_idx] == b_stamp) or (will_eat and next_idx == tail_idx)
                is_danger = hits_body
            dangers[act] = 1.0 if is_danger else 0.0
            next_coords.append((nx, ny, next_idx, will_eat, is_danger))

        # 2. Один объединённый BFS на каждое безопасное действие:
        #    одновременно даёт area_ratio, tail_blocked, partition, pocket_split и pocket_merge.
        for act in (0, 1, 2):
            nx, ny, next_idx, will_eat, is_danger = next_coords[act]
            if is_danger:
                tail_blocked[act] = 1.0
                partitions[act] = 0.0 if will_eat else 1.0
                pocket_splits[act] = 1.0
                continue

            if will_eat:
                saved_tail = blk[tail_idx]
                blk[tail_idx] = b_stamp
                tail_target_idx = tail_idx
                new_len = snake_len + 1
                empty_after = empty_no_tail - (1 if saved_tail != b_stamp else 0) - 1
            else:
                if snake_len >= 2:
                    tx2, ty2 = snake[-2]
                    tail_target_idx = ty2 * n + tx2
                else:
                    tail_target_idx = next_idx
                new_len = snake_len
                empty_after = empty_no_tail - 1

            saved_nh = blk[next_idx]
            blk[next_idx] = b_stamp

            stamp = self._next_bfs_stamp()
            next_comps = 0
            reach_nbs = 0
            for nb in neighbors[next_idx]:
                if blk[nb] != b_stamp and visited[nb] != stamp:
                    next_comps += 1
                    visited[nb] = stamp
                    q = [nb]
                    qi = 0
                    while qi < len(q):
                        curr = q[qi]
                        qi += 1
                        for nxt in neighbors[curr]:
                            if blk[nxt] != b_stamp and visited[nxt] != stamp:
                                visited[nxt] = stamp
                                q.append(nxt)
                    reach_nbs += len(q)

            # area_ratio: reach = 1 (сам next_idx) + все пустые клетки, достижимые из next_idx
            free_cells_ratio = max(1, board_cells - new_len)
            area_ratios[act] = min(1.0, float(1 + reach_nbs) / float(free_cells_ratio))

            # partition: достижима ли еда из next_idx
            if will_eat:
                partitions[act] = 0.0
            else:
                partitions[act] = 0.0 if visited[food_idx] == stamp else 1.0

            # tail_blocked: достижим ли tail_target_idx из next_idx
            if next_idx == tail_target_idx:
                has_tail_path = True
            elif blk[tail_target_idx] != b_stamp:
                has_tail_path = visited[tail_target_idx] == stamp
            else:
                has_tail_path = False
                for t_nb in neighbors[tail_target_idx]:
                    if t_nb == next_idx or visited[t_nb] == stamp:
                        has_tail_path = True
                        break
            tail_blocked[act] = 0.0 if has_tail_path else 1.0

            # next_comps: если все пустые клетки уже обойдены (90%+ случаев), доп. сканирование не нужно
            if reach_nbs < empty_after:
                visited_total = reach_nbs
                for start in range(board_cells):
                    if blk[start] != b_stamp and visited[start] != stamp:
                        next_comps += 1
                        visited[start] = stamp
                        q = [start]
                        qi = 0
                        while qi < len(q):
                            curr = q[qi]
                            qi += 1
                            for nxt in neighbors[curr]:
                                if blk[nxt] != b_stamp and visited[nxt] != stamp:
                                    visited[nxt] = stamp
                                    q.append(nxt)
                        visited_total += len(q)
                        if visited_total >= empty_after:
                            break

            pocket_splits[act] = 1.0 if next_comps > current_comps else 0.0
            pocket_merges[act] = 1.0 if next_comps < current_comps else 0.0

            blk[next_idx] = saved_nh
            if will_eat:
                blk[tail_idx] = saved_tail

        # tail_close проверяется лениво (только если хотя бы одно из 3 направлений попадает на хвост/след)
        tail_close = [0.0, 0.0, 0.0]
        if any(((nx, ny) == (tail_x, tail_y) or (nx, ny) in tail_track) for nx, ny, _, _, _ in next_coords):
            path = self._shortest_path_to_tail()
            if path and len(path) < 3:
                p_1 = path[0]
                path_dir = DIRECTIONS.index((p_1[0] - head_x, p_1[1] - head_y))
                for act in (0, 1, 2):
                    nx, ny, _, _, _ = next_coords[act]
                    if (nx, ny) != (tail_x, tail_y) and (nx, ny) not in tail_track:
                        continue
                    abs_dir = direction if act == 0 else ((direction - 1) % 4 if act == 1 else (direction + 1) % 4)
                    if abs_dir == path_dir:
                        has_safe_alt = any(dangers[alt] < 0.5 for alt in (0, 1, 2) if alt != act)
                        if has_safe_alt:
                            tail_close[act] = 1.0

        # Порядок относительных сигналов синхронизирован с MODEL_ACTION_LABELS:
        # forward, left, right. Behind-сигналы идут после этих направлений.
        signals = [
            1.0 if fwd > 0 else 0.0,
            1.0 if rgt < 0 else 0.0,
            1.0 if rgt > 0 else 0.0,
            1.0 if fwd < 0 else 0.0,  # food_behind

            apple_steps[0],
            apple_steps[1],
            apple_steps[2],

            food_reachable,

            dangers[0],
            dangers[1],
            dangers[2],

            1.0 if t_fwd > 0 else 0.0,  # tail_ahead
            1.0 if t_rgt < 0 else 0.0,  # tail_left
            1.0 if t_rgt > 0 else 0.0,  # tail_right
            1.0 if t_fwd < 0 else 0.0,  # tail_behind

            tail_blocked[0],  # tail_blocked_forward
            tail_blocked[1],  # tail_blocked_left
            tail_blocked[2],  # tail_blocked_right

            tail_close[0],   # tail_close_forward
            tail_close[1],   # tail_close_left
            tail_close[2],   # tail_close_right
            partitions[0],   # partition_forward
            partitions[1],   # partition_left
            partitions[2],   # partition_right
            endgame,         # endgame
            pocket_splits[0],  # pocket_split_forward
            pocket_splits[1],  # pocket_split_left
            pocket_splits[2],  # pocket_split_right
            hunger_high,       # hunger_high
            hunger_critical,   # hunger_critical
            board_split,       # board_split
            pocket_merges[0],  # pocket_merge_forward
            pocket_merges[1],  # pocket_merge_left
            pocket_merges[2],  # pocket_merge_right
        ]
        self._obs_cache_step = self.step_count
        self._obs_cache_vec = np.array(signals, dtype=np.float32)
        self._area_ratio_cache_step = self.step_count
        self._area_ratio_cache_vec = area_ratios
        return self._obs_cache_vec

    def get_egocentric_map(self, window_size: int = PPO_MAP_SIZE) -> np.ndarray:
        """
        Возвращает эгоцентричную карту (4, W, W) вокруг головы, повёрнутую по курсу:
          0: wall (стена / граница поля)
          1: body_ttl (1.0 у головы -> ~0 у хвоста; сколько ходов клетка ещё занята)
          2: food (1.0 в клетке яблока, если оно в окне)
          3: reachable (1.0 для достижимых из головы свободных клеток / кончика хвоста)
        Верх окна (row=0) — впереди змеи, лево — слева, право — справа.
        """
        win = max(3, int(window_size))
        if (
            self.step_count == self._ego_map_cache_step
            and self._ego_map_cache_size == win
            and self._ego_map_cache_arr is not None
        ):
            return self._ego_map_cache_arr

        n = self.board_size
        snake = self.snake
        snake_len = max(1, len(snake))
        head_x, head_y = snake[0]
        food_x, food_y = self.food
        radius = win // 2

        self._ensure_bfs_buffers(n)
        b_stamp = self._next_block_stamp()
        blk = self._bfs_blocked
        ttl_buf = self._ego_ttl_buf
        ttl_buf.fill(0.0)
        inv_len = 1.0 / float(snake_len)
        last_idx = snake_len - 1
        for idx, (bx, by) in enumerate(snake):
            flat_idx = by * n + bx
            ttl_buf[flat_idx] = float(snake_len - idx) * inv_len
            if idx < last_idx:
                blk[flat_idx] = b_stamp

        head_idx = head_y * n + head_x
        reach_stamp = self._next_bfs_stamp()
        visited = self._bfs_visited
        neighbors = self._flat_neighbors
        reach_buf = self._ego_reach_buf
        reach_buf.fill(0.0)
        visited[head_idx] = reach_stamp
        if blk[head_idx] != b_stamp:
            reach_buf[head_idx] = 1.0
        queue = [head_idx]
        qi = 0
        while qi < len(queue):
            curr = queue[qi]
            qi += 1
            for nxt in neighbors[curr]:
                if visited[nxt] != reach_stamp and blk[nxt] != b_stamp:
                    visited[nxt] = reach_stamp
                    reach_buf[nxt] = 1.0
                    queue.append(nxt)

        dx_grid, dy_grid = _ego_offsets(win, self.direction)
        wx = head_x + dx_grid
        wy = head_y + dy_grid
        in_bounds = (wx >= 0) & (wx < n) & (wy >= 0) & (wy < n)

        grid = np.zeros((PPO_MAP_CHANNELS, win, win), dtype=np.float32)
        grid[0] = (~in_bounds).astype(np.float32)
        valid_flat = wy[in_bounds] * n + wx[in_bounds]
        grid[1, in_bounds] = ttl_buf[valid_flat]
        grid[3, in_bounds] = reach_buf[valid_flat]

        fx, fy = DIRECTIONS[self.direction]
        rx, ry = DIRECTIONS[(self.direction + 1) % 4]
        dfx, dfy = food_x - head_x, food_y - head_y
        f_row = radius - (dfx * fx + dfy * fy)
        f_col = radius + (dfx * rx + dfy * ry)
        if 0 <= f_row < win and 0 <= f_col < win:
            grid[2, f_row, f_col] = 1.0

        self._ego_map_cache_step = self.step_count
        self._ego_map_cache_size = win
        self._ego_map_cache_arr = grid
        return grid

    def get_ppo_area_ratios(self) -> np.ndarray:
        """Возвращает 3 признака свободной достижимой площади после каждого относительного хода."""
        if self.step_count == self._area_ratio_cache_step and self._area_ratio_cache_vec is not None:
            return self._area_ratio_cache_vec
        self.get_extra_signals()
        assert self._area_ratio_cache_vec is not None
        return self._area_ratio_cache_vec

    def get_ppo_signals(self, target_size: int = STARTUP_BOARD_SIZE) -> np.ndarray:
        """Расширенный вектор сигналов для PPO: 34 базовых + 3 признака свободной площади."""
        if self.step_count == self._ppo_obs_cache_step and self._ppo_obs_cache_vec is not None:
            return self._ppo_obs_cache_vec
        base = self.get_extra_signals(target_size)
        ratios = self.get_ppo_area_ratios()
        vec = np.concatenate((base, ratios), axis=0).astype(np.float32)
        self._ppo_obs_cache_step = self.step_count
        self._ppo_obs_cache_vec = vec
        return vec

    def get_observation_grid(self, target_size: int = STARTUP_BOARD_SIZE) -> np.ndarray:
        # Возвращаем вектор сигналов (общая база признаков для DQN).
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
            "tail_track": [[x, y] for x, y in self.tail_track],
            "snake": [[x, y] for x, y in self.snake],
            "food": [self.food[0], self.food[1]],
            "recent_heads": [[x, y] for x, y in self.recent_heads],
            "recent_turns": list(self.recent_turns),
        }

    def load_state(self, payload: dict[str, object], target_size: int = STARTUP_BOARD_SIZE) -> np.ndarray:
        try:
            self.board_size = int(payload.get("board_size", STARTUP_BOARD_SIZE))
            self._ensure_bfs_buffers(self.board_size)
            self.snake = [(int(p[0]), int(p[1])) for p in payload.get("snake", [])]
            self.food = (int(payload["food"][0]), int(payload["food"][1]))
            self.direction = int(payload.get("direction", 1))
            self.score = int(payload.get("score", 0))
            self.step_count = int(payload.get("step_count", 0))
            self.steps_without_food = int(payload.get("steps_without_food", 0))
            self.last_steps_without_food = int(payload.get("last_steps_without_food", 0))
            self.max_steps_without_food = int(payload.get("max_steps_without_food", 0))
            self.done = bool(payload.get("done", False))
            self.last_reward = float(payload.get("last_reward", 0.0))
            self.last_reason = str(payload.get("last_reason", "new_game"))
            raw_penalties = payload.get("last_penalty_reasons", [])
            self.last_penalty_reasons = [str(x) for x in raw_penalties] if isinstance(raw_penalties, list) else []

            eft = payload.get("eaten_food_trail", [])
            self.eaten_food_trail = deque([(int(p[0]), int(p[1])) for p in eft], maxlen=24)
            tt = payload.get("tail_track", [])
            self.tail_track = deque([(int(p[0]), int(p[1])) for p in tt], maxlen=4)
            rh = payload.get("recent_heads", [])
            self.recent_heads = deque([(int(p[0]), int(p[1])) for p in rh], maxlen=max(18, int(len(self.snake) * 2.5)))
            rt = payload.get("recent_turns", [])
            self.recent_turns = deque([int(t) for t in rt], maxlen=8)

            self._clear_observation_cache()
            return self.get_observation_grid(target_size)
        except Exception:
            return self.reset()
