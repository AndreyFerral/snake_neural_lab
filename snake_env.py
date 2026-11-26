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
from snake_bfs_numba import (
    _EMPTY_TAIL_TRACK,
    compute_signals_and_ego_numba,
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
        self._all_empty_reachable_step: int = -1
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
            self._nb_blk = np.zeros(total_cells, dtype=np.int32)
            self._nb_visited = np.zeros(total_cells, dtype=np.int32)
            self._nb_queue = np.zeros(total_cells, dtype=np.int32)
            self._nb_parent = np.zeros(total_cells, dtype=np.int32)
            self._nb_sig = np.zeros(34, dtype=np.float32)
            self._nb_area = np.zeros(3, dtype=np.float32)
            self._nb_ego = np.zeros((PPO_MAP_CHANNELS, PPO_MAP_SIZE, PPO_MAP_SIZE), dtype=np.float32)
            self._nb_ego_win = PPO_MAP_SIZE

    def _compute_numba_fused(self, *, want_ego: bool, win: int = PPO_MAP_SIZE) -> None:
        n = self.board_size
        self._ensure_bfs_buffers(n)
        if want_ego and self._nb_ego_win != win:
            self._nb_ego = np.zeros((PPO_MAP_CHANNELS, win, win), dtype=np.float32)
            self._nb_ego_win = win
        snake_xy = np.asarray(self.snake, dtype=np.int32)
        tt_xy = np.asarray(self.tail_track, dtype=np.int32) if self.tail_track else _EMPTY_TAIL_TRACK
        h_lim = max(1.0, float(hunger_budget_for(self.score, n)))
        all_reach = compute_signals_and_ego_numba(
            n,
            snake_xy,
            int(self.direction),
            int(self.food[0]),
            int(self.food[1]),
            int(self.steps_without_food),
            h_lim,
            tt_xy,
            int(win),
            bool(want_ego),
            self._nb_blk,
            self._nb_visited,
            self._nb_queue,
            self._nb_parent,
            self._ego_ttl_buf,
            self._ego_reach_buf,
            self._nb_sig,
            self._nb_area,
            self._nb_ego,
        )
        self._obs_cache_step = self.step_count
        self._obs_cache_vec = self._nb_sig.copy()
        self._area_ratio_cache_step = self.step_count
        self._area_ratio_cache_vec = self._nb_area.copy()
        self._all_empty_reachable_step = self.step_count if all_reach else -1
        if want_ego:
            self._ego_map_cache_step = self.step_count
            self._ego_map_cache_size = win
            self._ego_map_cache_arr = self._nb_ego.copy()

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
        self._all_empty_reachable_step = -1
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
        """Возвращает 34 базовых сигнала (единый Numba-движок BFS с кэшем на шаг)."""
        if self.step_count == self._obs_cache_step and self._obs_cache_vec is not None:
            return self._obs_cache_vec
        self._compute_numba_fused(want_ego=False)
        assert self._obs_cache_vec is not None
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
        self._compute_numba_fused(want_ego=True, win=win)
        assert self._ego_map_cache_arr is not None
        return self._ego_map_cache_arr

    def get_ppo_area_ratios(self) -> np.ndarray:
        """Возвращает 3 признака свободной достижимой площади после каждого относительного хода."""
        if self.step_count == self._area_ratio_cache_step and self._area_ratio_cache_vec is not None:
            return self._area_ratio_cache_vec
        self.get_extra_signals()
        assert self._area_ratio_cache_vec is not None
        return self._area_ratio_cache_vec

    def get_ppo_signals(self, target_size: int = STARTUP_BOARD_SIZE) -> np.ndarray:
        """Расширенный вектор сигналов для PPO: 34 базовых + 3 area_ratio + 3 масштабно-инвариантных."""
        if self.step_count == self._ppo_obs_cache_step and self._ppo_obs_cache_vec is not None:
            return self._ppo_obs_cache_vec
        if self.step_count != self._obs_cache_step:
            # Один объединённый Numba-проход сразу строит 34 сигнала, 3 area_ratio и 11x11 эго-карту
            self._compute_numba_fused(want_ego=True, win=PPO_MAP_SIZE)
        base = self.get_extra_signals(target_size)
        ratios = self.get_ppo_area_ratios()
        n = self.board_size
        max_dist = max(1.0, float(2 * (n - 1)))
        hx, hy = self.snake[0]
        fx, fy = self.food
        tx, ty = self.snake[-1]
        scale_feats = np.array([
            float(len(self.snake)) / float(max(1, n * n)),
            float(abs(fx - hx) + abs(fy - hy)) / max_dist,
            float(abs(tx - hx) + abs(ty - hy)) / max_dist,
        ], dtype=np.float32)
        vec = np.concatenate((base, ratios, scale_feats), axis=0).astype(np.float32)
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
