from __future__ import annotations

from collections import deque
import random
from typing import Any, Deque
import numpy as np

from backend_config import (
    APPLE_REWARD,
    STARTUP_BOARD_SIZE,
    DIRECTIONS,
    ENDGAME_FILL_RATIO,
    MODEL_ACTION_LABELS,
    TAIL_PATH_LOST_PENALTY,
)

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


def absolute_to_relative(previous_direction: int, chosen_direction: int) -> int:
    delta = (chosen_direction - previous_direction) % 4
    if delta == 0:
        return 0
    if delta == 3:
        return 1
    if delta == 1:
        return 2
    return 0


class SnakeEnv:
    def __init__(self, board_size: int = STARTUP_BOARD_SIZE) -> None:
        self.board_size = board_size
        self._path_cache: dict[tuple[Any, ...], Any] = {}
        self._path_cache_max_size = 10000
        self._obs_cache_step: int = -1
        self._obs_cache_vec: np.ndarray | None = None
        self.reset()

    def _clear_path_cache(self) -> None:
        self._path_cache.clear()

    def _check_cache_size(self) -> None:
        if len(self._path_cache) > self._path_cache_max_size:
            self._path_cache.clear()

    def _blocked_key(self, blocked: set[tuple[int, int]]) -> int:
        mask = 0
        for x, y in blocked:
            mask |= 1 << (y * self.board_size + x)
        return mask

    def _neighbors_for(self, cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        return board_neighbors(self.board_size).get(cell, ())

    def reset(self) -> np.ndarray:
        self._clear_path_cache()
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
        self.tail_path_lost_penalty_active = False
        self.trap_after_apple_count = 0
        self.avoidable_collision_count = 0
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

    def adjacent_apple_opportunity(self) -> dict[str, Any]:
        for relative_action in range(len(MODEL_ACTION_LABELS)):
            absolute_direction = relative_to_absolute(self.direction, relative_action)
            next_head = self._next_head_for_direction(absolute_direction)
            if next_head != self.food:
                continue

            signals = self._lookahead_signals_for_absolute_direction(absolute_direction)
            blocked, food_reachable, tail_reachable, enough_space, food_tail_safe = signals

            safe = (
                blocked < 0.5
                and food_tail_safe > 0.5
                and (tail_reachable > 0.5 or enough_space > 0.5)
            )
            if blocked >= 0.5:
                verdict = "blocked"
            elif food_tail_safe < 0.5:
                verdict = "no_tail_after_apple"
            elif tail_reachable < 0.5 and enough_space < 0.5:
                verdict = "not_enough_space"
            else:
                verdict = "safe"

            return {
                "adjacent": True,
                "action": relative_action,
                "action_label": MODEL_ACTION_LABELS[relative_action],
                "safe": safe,
                "verdict": verdict,
                "tail_reachable": tail_reachable,
                "enough_space": enough_space,
                "food_tail_safe": food_tail_safe,
            }
        return {"adjacent": False, "action": None, "safe": False, "verdict": "none"}

    def relative_move_risks(self) -> dict[str, float]:
        return {
            "forward": self._danger_for_relative_action(0),
            "left": self._danger_for_relative_action(1),
            "right": self._danger_for_relative_action(2),
        }

    def relative_move_safety(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for key, relative_action in (("forward", 0), ("left", 1), ("right", 2)):
            blocked, food, tail, space, safe = self._lookahead_signals_for_relative_action(
                relative_action
            )
            result[key] = {
                "risk": blocked,
                "food": food,
                "tail": tail,
                "space": space,
                "safe": safe,
            }
        return result

    def _is_dead_end_without_alternative(self) -> bool:
        for relative_action in range(len(MODEL_ACTION_LABELS)):
            blocked, _, tail, space, _ = self._lookahead_signals_for_relative_action(
                relative_action
            )
            if blocked < 0.5 and (tail > 0.5 or space > 0.5):
                return False
        return True

    def _current_tail_reachable(self) -> bool:
        if len(self.snake) <= 1:
            return True
        head, tail = self.snake[0], self.snake[-1]
        blocked_without_tail = set(self.snake[1:-1])
        return self._has_path(head, tail, blocked_without_tail)

    def _is_endgame(self) -> bool:
        return len(self.snake) >= self.board_size * self.board_size * ENDGAME_FILL_RATIO

    def _minimum_useful_no_tail_area(self) -> float:
        return float(max(4, min(len(self.snake), self.board_size)))

    def _tail_path_lost_now(self) -> bool:
        tail_reachable = self._current_tail_reachable()
        if tail_reachable:
            self.tail_path_lost_penalty_active = False
            return False
        if self.tail_path_lost_penalty_active:
            return False
        self.tail_path_lost_penalty_active = True
        return True

    def _is_avoidable_collision(self, chosen_relative_action: int) -> bool:
        chosen_blocked = self._lookahead_signals_for_relative_action(
            chosen_relative_action
        )[0]
        if chosen_blocked < 0.5:
            return False
        for relative_action in range(len(MODEL_ACTION_LABELS)):
            if relative_action == chosen_relative_action:
                continue
            if self._lookahead_signals_for_relative_action(relative_action)[0] < 0.5:
                return True
        return False

    def _has_compact_head_loop(self, next_head: tuple[int, int]) -> bool:
        head_window = list(self.recent_heads)[-20:] + [next_head]
        if len(head_window) < 16:
            return False
        unique_heads = len(set(head_window))
        unique_limit = max(4, min(8, len(self.snake) // 4 + 2))
        return unique_heads <= unique_limit

    def _reachable_cell_count(
        self, start: tuple[int, int], blocked: set[tuple[int, int]]
    ) -> int:
        cache_key = ("reachable_count", start, self._blocked_key(blocked))
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            return int(cached)

        size = self.board_size
        start_idx = start[1] * size + start[0]
        # Быстрая маска блокировок (list[bool] быстрее set[tuple] в цикле)
        blocked_mask = [False] * (size * size)
        for x, y in blocked:
            blocked_mask[y * size + x] = True

        if blocked_mask[start_idx]:
            return 0

        visited = [False] * (size * size)
        visited[start_idx] = True
        queue = [start_idx]
        head = 0
        count = 1

        while head < len(queue):
            curr = queue[head]
            head += 1
            cx, cy = curr % size, curr // size
            # Инлайн соседей без вызова _neighbors_for
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    nidx = ny * size + nx
                    if not visited[nidx] and not blocked_mask[nidx]:
                        visited[nidx] = True
                        queue.append(nidx)
                        count += 1

        self._path_cache[cache_key] = count
        return count

    def _has_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> bool:
        cache_key = ("has_path", start, goal, self._blocked_key(blocked))
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        if start == goal:
            return True

        size = self.board_size
        start_idx = start[1] * size + start[0]
        goal_idx = goal[1] * size + goal[0]

        blocked_mask = [False] * (size * size)
        for x, y in blocked:
            blocked_mask[y * size + x] = True

        if blocked_mask[start_idx] or blocked_mask[goal_idx]:
            self._path_cache[cache_key] = False
            return False

        visited = [False] * (size * size)
        visited[start_idx] = True
        queue = [start_idx]
        head = 0

        while head < len(queue):
            curr = queue[head]
            head += 1
            if curr == goal_idx:
                self._path_cache[cache_key] = True
                return True
            cx, cy = curr % size, curr // size
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    nidx = ny * size + nx
                    if not visited[nidx] and not blocked_mask[nidx]:
                        visited[nidx] = True
                        queue.append(nidx)

        self._path_cache[cache_key] = False
        return False

    def _shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        cache_key = ("shortest_path", start, goal, self._blocked_key(blocked))
        if cache_key in self._path_cache:
            cached_path = self._path_cache[cache_key]
            return None if cached_path is None else list(cached_path)
        if start == goal:
            return [start]

        visited, parents, queue = {start}, {}, deque([start])
        while queue:
            cell = queue.popleft()
            for next_cell in self._neighbors_for(cell):
                if next_cell not in visited and next_cell not in blocked:
                    parents[next_cell] = cell
                    if next_cell == goal:
                        path = [goal]
                        while path[-1] != start:
                            path.append(parents[path[-1]])
                        path.reverse()
                        self._path_cache[cache_key] = tuple(path)
                        return path
                    visited.add(next_cell)
                    queue.append(next_cell)
        self._path_cache[cache_key] = None
        return None

    def _reachable_cells(
        self, start: tuple[int, int], blocked: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        cache_key = ("reachable_cells", start, self._blocked_key(blocked))
        if cache_key in self._path_cache:
            return set(self._path_cache[cache_key])
        if start in blocked:
            return set()

        visited, queue = {start}, deque([start])
        while queue:
            cell = queue.popleft()
            for next_cell in self._neighbors_for(cell):
                if next_cell not in visited and next_cell not in blocked:
                    visited.add(next_cell)
                    queue.append(next_cell)
        self._path_cache[cache_key] = tuple(visited)
        return visited

    def _free_components(
        self, blocked: set[tuple[int, int]]
    ) -> list[set[tuple[int, int]]]:
        remaining = {
            (x, y)
            for y in range(self.board_size)
            for x in range(self.board_size)
            if (x, y) not in blocked
        }
        components: list[set[tuple[int, int]]] = []
        while remaining:
            start = next(iter(remaining))
            component = self._reachable_cells(start, blocked)
            components.append(component)
            remaining.difference_update(component)
        return components

    def _food_tail_safe_after_path_for(
        self,
        path: list[tuple[int, int]] | None,
        snake: list[tuple[int, int]],
        food: tuple[int, int],
    ) -> float:
        if not path:
            return 0.0
        simulated_snake = list(snake)
        for next_cell in path[1:]:
            ate_food = next_cell == food
            simulated_snake.insert(0, next_cell)
            if not ate_food:
                simulated_snake.pop()
        if len(simulated_snake) <= 1:
            return 1.0
        head, tail = simulated_snake[0], simulated_snake[-1]
        return 1.0 if self._has_path(head, tail, set(simulated_snake[1:-1])) else 0.0

    def _strategic_space_metrics_for(
        self, snake: list[tuple[int, int]], food: tuple[int, int]
    ) -> dict[str, float]:
        cache_key = ("strategic", tuple(snake), food)
        if cache_key in self._path_cache:
            return dict(self._path_cache[cache_key])

        total_cells = float(self.board_size * self.board_size)
        blocked = set(snake[1:])
        total_walkable = max(1.0, total_cells - len(blocked))
        head = snake[0]
        head_area = self._reachable_cells(head, blocked)
        food_reachable = 1.0 if food in head_area else 0.0
        food_area_count = len(head_area) if food_reachable else self._reachable_cell_count(food, blocked)
        components = self._free_components(blocked)
        pocket_count = sum(1 for c in components if head not in c)
        largest_pocket = max((len(c) for c in components if head not in c), default=0)
        path_to_food = self._shortest_path(head, food, blocked)

        result = {
            "food_reachable": food_reachable,
            "food_area_ratio": clamp(food_area_count / total_walkable, 0.0, 1.0),
            "food_tail_safe": self._food_tail_safe_after_path_for(path_to_food, snake, food),
            "pocket_count_norm": min(1.0, pocket_count / (self.board_size / 2.0)),
            "largest_pocket_ratio": clamp(largest_pocket / total_walkable, 0.0, 1.0),
            "head_area_ratio": clamp(len(head_area) / total_walkable, 0.0, 1.0),
        }
        self._path_cache[cache_key] = result
        return result

    def _strategic_space_metrics(self) -> dict[str, float]:
        return self._strategic_space_metrics_for(list(self.snake), self.food)

    def _post_move_space_probe(self, absolute_direction: int) -> dict[str, Any]:
        cache_key = (
            "post_move_space",
            tuple(self.snake),
            self.food,
            self.direction,
            absolute_direction,
        )
        if cache_key in self._path_cache:
            return dict(self._path_cache[cache_key])

        next_head = self._next_head_for_direction(absolute_direction)
        ate_food = next_head == self.food
        hits_wall = not (
            0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size
        )

        occupied = set(self.snake if ate_food else self.snake[:-1])
        if hits_wall or next_head in occupied:
            res = {
                "blocked": 1.0,
                "area": 0.0,
                "tail": 0.0,
                "next_head": next_head,
                "simulated_snake": [],
            }
        else:
            simulated_snake = [next_head] + (
                list(self.snake) if ate_food else list(self.snake[:-1])
            )
            blocked_cells = set(simulated_snake[1:])
            area_count = float(self._reachable_cell_count(next_head, blocked_cells))
            tail_target = simulated_snake[-1]
            tail_blocked = set(blocked_cells)
            if not ate_food:
                tail_blocked.discard(tail_target)
            tail_reachable = 1.0 if self._has_path(next_head, tail_target, tail_blocked) else 0.0
            res = {
                "blocked": 0.0,
                "area": area_count,
                "tail": tail_reachable,
                "next_head": next_head,
                "simulated_snake": simulated_snake,
            }

        self._path_cache[cache_key] = res
        return res

    def _best_no_tail_area(self) -> float:
        best_area = 0.0
        for rel_act in range(len(MODEL_ACTION_LABELS)):
            probe = self._post_move_space_probe(relative_to_absolute(self.direction, rel_act))
            if probe["blocked"] < 0.5 and probe["tail"] < 0.5:
                best_area = max(best_area, probe["area"])
        return best_area

    def _has_available_tail_path(self) -> bool:
        for rel_act in range(len(MODEL_ACTION_LABELS)):
            probe = self._post_move_space_probe(relative_to_absolute(self.direction, rel_act))
            if probe["blocked"] < 0.5 and probe["tail"] >= 0.5:
                return True
        return False

    def _lookahead_signals_for_absolute_direction(
        self, absolute_direction: int
    ) -> list[float]:
        cache_key = (
            "lookahead",
            tuple(self.snake),
            self.food,
            self.direction,
            absolute_direction,
        )
        if cache_key in self._path_cache:
            return list(self._path_cache[cache_key])

        probe = self._post_move_space_probe(absolute_direction)
        if probe["blocked"] >= 0.5:
            res = [1.0, 0.0, 0.0, 0.0, 0.0]
            self._path_cache[cache_key] = tuple(res)
            return res

        sim_snake = probe["simulated_snake"]
        blocked_cells = set(sim_snake[1:])
        food_reach = 1.0 if self._has_path(probe["next_head"], self.food, blocked_cells) else 0.0
        tail_reach = probe["tail"]

        if tail_reach >= 0.5:
            enough_space = 1.0
        elif self._has_available_tail_path():
            enough_space = 0.0
        else:
            best_area = self._best_no_tail_area()
            enough_space = (
                1.0
                if best_area >= self._minimum_useful_no_tail_area() and probe["area"] >= best_area
                else 0.0
            )

        path_to_food = (
            [probe["next_head"]]
            if probe["next_head"] == self.food
            else self._shortest_path(probe["next_head"], self.food, blocked_cells)
        )
        res = [
            0.0,
            food_reach,
            tail_reach,
            enough_space,
            self._food_tail_safe_after_path_for(path_to_food, sim_snake, self.food),
        ]
        self._path_cache[cache_key] = tuple(res)
        return res

    def _lookahead_signals_for_relative_action(self, relative_action: int) -> list[float]:
        return self._lookahead_signals_for_absolute_direction(
            relative_to_absolute(self.direction, relative_action)
        )

    def step(
        self, direction: int
    ) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return (
                np.empty((0,), dtype=np.float32),
                self.last_reward,
                True,
                {"score": self.score, "steps": self.step_count, "reason": self.last_reason},
            )

        chosen_direction = self._normalize_direction(direction)
        previous_direction = self.direction
        apple_opportunity = self.adjacent_apple_opportunity()
        chosen_rel_act = absolute_to_relative(previous_direction, chosen_direction)
        avoidable_collision = self._is_avoidable_collision(chosen_rel_act)
        self.direction = chosen_direction

        head, move = self.snake[0], DIRECTIONS[chosen_direction]
        next_head = (head[0] + move[0], head[1] + move[1])
        will_eat = next_head == self.food

        # В начале step(), до self.step_count += 1:
        self._obs_cache_step = -1
        self._obs_cache_vec = None

        reward, reason, penalty_reasons = -0.01, "step", []
        self.step_count += 1
        self.steps_without_food += 1
        self.max_steps_without_food = max(self.max_steps_without_food, self.steps_without_food)

        hits_wall = not (
            0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size
        )
        if hits_wall or self._would_hit_self(next_head, will_eat):
            self.done = True
            reward = -5.0
            reason = "wall_collision" if hits_wall else "body_collision"
            penalty_reasons.append(reason)
            if avoidable_collision:
                self.avoidable_collision_count += 1
                penalty_reasons.append("avoidable_collision")
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

                # Логика маршрута (route/line/return) полностью удалена

                if self._tail_path_lost_now():
                    reward -= TAIL_PATH_LOST_PENALTY
                    penalty_reasons.append("tail_path_lost")

                # Голодание и тупики
                starve_limit = max(60, int(self.board_size * self.board_size * 0.8))
                if self.steps_without_food > starve_limit or self._has_compact_head_loop(next_head):
                    self.done = True
                    reward = -5.0
                    reason = "starvation"
                    penalty_reasons.append(reason)

            self.recent_heads.append(next_head)

        self.last_reward = reward
        self.last_reason = reason
        self.last_penalty_reasons = penalty_reasons
        self._check_cache_size()

        return (
            np.empty((0,), dtype=np.float32),
            reward,
            self.done,
            {
                "score": self.score,
                "steps": self.step_count,
                "reason": reason,
                "penalty_reasons": list(penalty_reasons),
                "direction": chosen_direction,
                "apple_opportunity": apple_opportunity,
            },
        )

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

        signals = [
            1.0 if fwd > 0 else 0.0,
            1.0 if rgt > 0 else 0.0,
            1.0 if rgt < 0 else 0.0,
            1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 0)) == self.food else 0.0,
            1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 2)) == self.food else 0.0,
            1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 1)) == self.food else 0.0,
            self._danger_for_relative_action(0),
            self._danger_for_relative_action(2),
            self._danger_for_relative_action(1),
            *self._lookahead_signals_for_relative_action(0)[1:],
            *self._lookahead_signals_for_relative_action(1)[1:],
            *self._lookahead_signals_for_relative_action(2)[1:],
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
            "max_steps_without_food": self.max_steps_without_food,
            "done": self.done,
            "last_reward": self.last_reward,
            "last_reason": self.last_reason,
            "last_penalty_reasons": list(self.last_penalty_reasons),
            "tail_path_lost_penalty_active": self.tail_path_lost_penalty_active,
            "trap_after_apple_count": self.trap_after_apple_count,
            "avoidable_collision_count": self.avoidable_collision_count,
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
            return self.get_observation_grid(target_size)
        except Exception:
            return self.reset()