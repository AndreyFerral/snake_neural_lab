from __future__ import annotations

from collections import deque
import random
from typing import Any, Deque

import numpy as np

from backend_config import (
    APPLE_REWARD,
    BOARD_SIZES,
    DEFAULT_INPUT_GRID,
    DIRECTIONS,
    ENDGAME_FILL_RATIO,
    LOCAL_MAP_CHANNEL_COUNT,
    MODEL_ACTION_LABELS,
    ROUTE_APPLE_LINE_BONUS,
    ROUTE_DEVIATION_PENALTY,
    ROUTE_FOLLOW_BONUS,
    ROUTE_RETURN_AFTER_APPLE_BONUS,
    STARTUP_INPUT_MODE,
    TAIL_PATH_LOST_PENALTY,
    TRAP_AFTER_APPLE_PENALTY,
)

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
                next_x = x + move_x
                next_y = y + move_y
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
    def __init__(self, board_size: int = DEFAULT_INPUT_GRID) -> None:
        self.board_size = board_size
        self._path_cache: dict[tuple[Any, ...], Any] = {}
        self.reset()

    def _clear_path_cache(self) -> None:
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
        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center),
        ]
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

    def _relative_offset_to_world(self, rel_x: int, rel_y: int) -> tuple[int, int]:
        head_x, head_y = self.snake[0]
        if self.direction == 0:
            world_dx, world_dy = rel_x, rel_y
        elif self.direction == 1:
            world_dx, world_dy = -rel_y, rel_x
        elif self.direction == 2:
            world_dx, world_dy = -rel_x, -rel_y
        else:
            world_dx, world_dy = rel_y, -rel_x
        return head_x + world_dx, head_y + world_dy

    def _danger_for_relative_action(self, relative_action: int) -> float:
        absolute_direction = relative_to_absolute(self.direction, relative_action)
        next_head = self._next_head_for_direction(absolute_direction)
        will_eat = next_head == self.food
        hits_wall = (
            next_head[0] < 0
            or next_head[0] >= self.board_size
            or next_head[1] < 0
            or next_head[1] >= self.board_size
        )
        hits_body = self._would_hit_self(next_head, will_eat)
        return 1.0 if hits_wall or hits_body else 0.0

    def adjacent_apple_opportunity(self) -> dict[str, Any]:
        for relative_action in range(len(MODEL_ACTION_LABELS)):
            absolute_direction = relative_to_absolute(self.direction, relative_action)
            next_head = self._next_head_for_direction(absolute_direction)
            if next_head != self.food:
                continue
            blocked, food_reachable, tail_reachable, enough_space, food_tail_safe = self._lookahead_signals_for_absolute_direction(absolute_direction)
            safe = blocked < 0.5 and food_tail_safe > 0.5 and (tail_reachable > 0.5 or enough_space > 0.5)
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
            blocked, food, tail, space, safe = self._lookahead_signals_for_relative_action(relative_action)
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
            blocked, _, tail, space, _ = self._lookahead_signals_for_relative_action(relative_action)
            if blocked < 0.5 and (tail > 0.5 or space > 0.5):
                return False
        return True

    def _current_tail_reachable(self) -> bool:
        if len(self.snake) <= 1:
            return True
        head = self.snake[0]
        tail = self.snake[-1]
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
        chosen_blocked = self._lookahead_signals_for_relative_action(chosen_relative_action)[0]
        if chosen_blocked < 0.5:
            return False
        for relative_action in range(len(MODEL_ACTION_LABELS)):
            if relative_action == chosen_relative_action:
                continue
            blocked = self._lookahead_signals_for_relative_action(relative_action)[0]
            if blocked < 0.5:
                return True
        return False

    def _has_compact_head_loop(self, next_head: tuple[int, int]) -> bool:
        head_window = list(self.recent_heads)[-20:] + [next_head]
        if len(head_window) < 16:
            return False
        unique_heads = len(set(head_window))
        unique_limit = max(4, min(8, len(self.snake) // 4 + 2))
        return unique_heads <= unique_limit

    def _reachable_cell_count(self, start: tuple[int, int], blocked: set[tuple[int, int]]) -> int:
        cache_key = ("reachable_count", start, self._blocked_key(blocked))
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        if start in blocked:
            self._path_cache[cache_key] = 0
            return 0

        visited: set[tuple[int, int]] = {start}
        queue: Deque[tuple[int, int]] = deque([start])

        while queue:
            cell = queue.popleft()
            for next_cell in self._neighbors_for(cell):
                if next_cell not in visited and next_cell not in blocked:
                    visited.add(next_cell)
                    queue.append(next_cell)

        result = len(visited)
        self._path_cache[cache_key] = result
        return result

    def _has_path(self, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
        cache_key = ("has_path", start, goal, self._blocked_key(blocked))
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        if start == goal:
            self._path_cache[cache_key] = True
            return True

        visited: set[tuple[int, int]] = {start}
        queue: Deque[tuple[int, int]] = deque([start])

        while queue:
            cell = queue.popleft()
            for next_cell in self._neighbors_for(cell):
                if next_cell == goal:
                    self._path_cache[cache_key] = True
                    return True
                if next_cell not in visited and next_cell not in blocked:
                    visited.add(next_cell)
                    queue.append(next_cell)

        self._path_cache[cache_key] = False
        return False

    def _shortest_path(self, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        cache_key = ("shortest_path", start, goal, self._blocked_key(blocked))
        if cache_key in self._path_cache:
            cached_path = self._path_cache[cache_key]
            return None if cached_path is None else list(cached_path)
        if start == goal:
            self._path_cache[cache_key] = (start,)
            return [start]

        visited: set[tuple[int, int]] = {start}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        queue: Deque[tuple[int, int]] = deque([start])

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

    def _reachable_cells(self, start: tuple[int, int], blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
        cache_key = ("reachable_cells", start, self._blocked_key(blocked))
        if cache_key in self._path_cache:
            return set(self._path_cache[cache_key])
        if start in blocked:
            self._path_cache[cache_key] = tuple()
            return set()

        visited: set[tuple[int, int]] = {start}
        queue: Deque[tuple[int, int]] = deque([start])
        while queue:
            cell = queue.popleft()
            for next_cell in self._neighbors_for(cell):
                if next_cell not in visited and next_cell not in blocked:
                    visited.add(next_cell)
                    queue.append(next_cell)
        self._path_cache[cache_key] = tuple(visited)
        return visited

    def _free_components(self, blocked: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
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
        simulated_head = simulated_snake[0]
        simulated_tail = simulated_snake[-1]
        blocked_without_tail = set(simulated_snake[1:-1])
        return 1.0 if self._has_path(simulated_head, simulated_tail, blocked_without_tail) else 0.0

    def _strategic_space_metrics_for(self, snake: list[tuple[int, int]], food: tuple[int, int]) -> dict[str, float]:
        cache_key = ("strategic", tuple(snake), food)
        cached = self._path_cache.get(cache_key)
        if isinstance(cached, dict):
            return dict(cached)
        total_cells = max(1.0, float(self.board_size * self.board_size))
        blocked = set(snake[1:])
        total_walkable = max(1.0, total_cells - len(blocked))
        head = snake[0]
        head_area = self._reachable_cells(head, blocked)
        food_reachable = 1.0 if food in head_area else 0.0
        food_area_count = len(head_area) if food_reachable else self._reachable_cell_count(food, blocked)
        components = self._free_components(blocked)
        pocket_count = sum(1 for component in components if head not in component)
        largest_pocket = max((len(component) for component in components if head not in component), default=0)
        path_to_food = self._shortest_path(head, food, blocked)
        food_tail_safe = self._food_tail_safe_after_path_for(path_to_food, snake, food)
        food_area_ratio = food_area_count / total_walkable
        pocket_count_norm = min(1.0, pocket_count / max(1.0, self.board_size / 2.0))
        largest_pocket_ratio = largest_pocket / total_walkable
        result = {
            "food_reachable": food_reachable,
            "food_area_ratio": clamp(food_area_ratio, 0.0, 1.0),
            "food_tail_safe": food_tail_safe,
            "pocket_count_norm": pocket_count_norm,
            "largest_pocket_ratio": clamp(largest_pocket_ratio, 0.0, 1.0),
            "head_area_ratio": clamp(len(head_area) / total_walkable, 0.0, 1.0),
        }
        self._path_cache[cache_key] = result
        return dict(result)

    def _strategic_space_metrics(self) -> dict[str, float]:
        return self._strategic_space_metrics_for(list(self.snake), self.food)

    def _post_move_space_probe(self, absolute_direction: int) -> dict[str, float | list[tuple[int, int]] | tuple[int, int]]:
        cache_key = ("post_move_space", tuple(self.snake), self.food, self.direction, absolute_direction)
        cached = self._path_cache.get(cache_key)
        if isinstance(cached, dict):
            return dict(cached)

        next_head = self._next_head_for_direction(absolute_direction)
        ate_food = next_head == self.food
        hits_wall = (
            next_head[0] < 0
            or next_head[0] >= self.board_size
            or next_head[1] < 0
            or next_head[1] >= self.board_size
        )
        occupied_for_collision = set(self.snake if ate_food else self.snake[:-1])
        blocked = hits_wall or next_head in occupied_for_collision
        if blocked:
            result: dict[str, float | list[tuple[int, int]] | tuple[int, int]] = {
                "blocked": 1.0,
                "area": 0.0,
                "tail": 0.0,
                "next_head": next_head,
                "simulated_snake": [],
            }
            self._path_cache[cache_key] = result
            return dict(result)

        simulated_snake = [next_head] + (list(self.snake) if ate_food else list(self.snake[:-1]))
        blocked_cells = set(simulated_snake[1:])
        area_count = float(self._reachable_cell_count(next_head, blocked_cells))
        tail_target = simulated_snake[-1]
        tail_blocked = set(blocked_cells)
        if not ate_food:
            tail_blocked.discard(tail_target)
        tail_reachable = 1.0 if self._has_path(next_head, tail_target, tail_blocked) else 0.0
        result = {
            "blocked": 0.0,
            "area": area_count,
            "tail": tail_reachable,
            "next_head": next_head,
            "simulated_snake": simulated_snake,
        }
        self._path_cache[cache_key] = result
        return dict(result)

    def _best_no_tail_area(self) -> float:
        best_area = 0.0
        for relative_action in range(len(MODEL_ACTION_LABELS)):
            absolute_direction = relative_to_absolute(self.direction, relative_action)
            probe = self._post_move_space_probe(absolute_direction)
            if float(probe.get("blocked", 1.0)) < 0.5 and float(probe.get("tail", 0.0)) < 0.5:
                best_area = max(best_area, float(probe.get("area", 0.0)))
        return best_area

    def _has_available_tail_path(self) -> bool:
        for relative_action in range(len(MODEL_ACTION_LABELS)):
            absolute_direction = relative_to_absolute(self.direction, relative_action)
            probe = self._post_move_space_probe(absolute_direction)
            if float(probe.get("blocked", 1.0)) < 0.5 and float(probe.get("tail", 0.0)) >= 0.5:
                return True
        return False

    def _route_path(self) -> list[tuple[int, int]]:
        size = self.board_size
        if size <= 1:
            return [(0, 0)]
        if size % 2 != 0:
            path = [(x, 0) for x in range(size)]
            path.extend((size - 1, y) for y in range(1, size))
            path.extend((x, size - 1) for x in range(size - 2, -1, -1))
            path.extend((0, y) for y in range(size - 2, 0, -1))
            return path
        path = [(x, 0) for x in range(size)]
        for x in range(size - 1, 0, -1):
            y_values = range(1, size) if (size - 1 - x) % 2 == 0 else range(size - 1, 0, -1)
            path.extend((x, y) for y in y_values)
        path.extend((0, y) for y in range(size - 1, 0, -1))
        return path

    def _route_next_cell(self, cell: tuple[int, int]) -> tuple[int, int] | None:
        path = self._route_path()
        try:
            index = path.index(cell)
        except ValueError:
            return None
        return path[(index + 1) % len(path)]

    def _route_direction_from(self, cell: tuple[int, int]) -> int | None:
        next_cell = self._route_next_cell(cell)
        if next_cell is None:
            return None
        dx = next_cell[0] - cell[0]
        dy = next_cell[1] - cell[1]
        for index, (move_x, move_y) in enumerate(DIRECTIONS):
            if (dx, dy) == (move_x, move_y):
                return index
        return None

    def _is_route_move_available(self, direction: int) -> bool:
        probe = self._post_move_space_probe(direction)
        return float(probe.get("blocked", 1.0)) < 0.5

    def _is_apple_on_direction_line(self, direction: int) -> bool:
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        move_x, move_y = DIRECTIONS[direction]
        if move_x != 0 and (food_y != head_y or (food_x - head_x) * move_x <= 0):
            return False
        if move_y != 0 and (food_x != head_x or (food_y - head_y) * move_y <= 0):
            return False
        blocked = set(self.snake[1:-1])
        x, y = head_x + move_x, head_y + move_y
        while 0 <= x < self.board_size and 0 <= y < self.board_size:
            if (x, y) in blocked:
                return False
            if (x, y) == self.food:
                return True
            x += move_x
            y += move_y
        return False

    def _can_return_to_route_after_move(self, simulated_snake: list[tuple[int, int]]) -> bool:
        if not simulated_snake:
            return False
        route_direction = self._route_direction_from(simulated_snake[0])
        if route_direction is None:
            return False
        move_x, move_y = DIRECTIONS[route_direction]
        next_cell = (simulated_snake[0][0] + move_x, simulated_snake[0][1] + move_y)
        if not (0 <= next_cell[0] < self.board_size and 0 <= next_cell[1] < self.board_size):
            return False
        return next_cell not in set(simulated_snake[1:-1])

    def _lookahead_signals_for_absolute_direction(self, absolute_direction: int) -> list[float]:
        cache_key = ("lookahead", tuple(self.snake), self.food, self.direction, absolute_direction)
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        probe = self._post_move_space_probe(absolute_direction)
        next_head = probe["next_head"]
        ate_food = next_head == self.food
        blocked = float(probe.get("blocked", 1.0)) >= 0.5
        if blocked:
            result = [1.0, 0.0, 0.0, 0.0, 0.0]
            self._path_cache[cache_key] = tuple(result)
            return result

        simulated_snake = list(probe.get("simulated_snake", []))
        blocked_cells = set(simulated_snake[1:])
        area_count = float(probe.get("area", 0.0))
        food_reachable = 1.0 if self._has_path(next_head, self.food, blocked_cells) else 0.0
        tail_reachable = float(probe.get("tail", 0.0))

        if tail_reachable >= 0.5:
            enough_space = 1.0
        elif self._has_available_tail_path():
            enough_space = 0.0
        else:
            best_no_tail_area = self._best_no_tail_area()
            minimum_area = self._minimum_useful_no_tail_area()
            enough_space = 1.0 if best_no_tail_area >= minimum_area and area_count >= best_no_tail_area else 0.0
        path_to_food = [next_head] if ate_food else self._shortest_path(next_head, self.food, blocked_cells)
        food_tail_safe = self._food_tail_safe_after_path_for(path_to_food, simulated_snake, self.food)
        result = [0.0, food_reachable, tail_reachable, enough_space, food_tail_safe]
        self._path_cache[cache_key] = tuple(result)
        return result

    def _lookahead_signals_for_relative_action(self, relative_action: int) -> list[float]:
        return self._lookahead_signals_for_absolute_direction(relative_to_absolute(self.direction, relative_action))

    def step(self, direction: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            self._clear_path_cache()
            return np.empty((0,), dtype=np.float32), self.last_reward, True, {
                "score": self.score,
                "steps": self.step_count,
                "reason": self.last_reason,
                "direction": self.direction,
            }

        chosen_direction = self._normalize_direction(direction)
        previous_direction = self.direction
        apple_opportunity = self.adjacent_apple_opportunity()
        chosen_relative_action = absolute_to_relative(previous_direction, chosen_direction)
        avoidable_collision = self._is_avoidable_collision(chosen_relative_action)
        self.direction = chosen_direction

        head_x, head_y = self.snake[0]
        move_x, move_y = DIRECTIONS[chosen_direction]
        next_head = (head_x + move_x, head_y + move_y)
        will_eat = next_head == self.food
        route_direction = self._route_direction_from(self.snake[0])
        route_available = route_direction is not None and self._is_route_move_available(route_direction)
        chosen_follows_route = route_direction == chosen_direction
        chosen_apple_line = self._is_apple_on_direction_line(chosen_direction)
        endgame = self._is_endgame()

        reward = 0.0
        reason = "step"
        penalty_reasons: list[str] = []

        self.step_count += 1
        self.steps_without_food += 1
        self.max_steps_without_food = max(self.max_steps_without_food, self.steps_without_food)

        hits_wall = (
            next_head[0] < 0
            or next_head[0] >= self.board_size
            or next_head[1] < 0
            or next_head[1] >= self.board_size
        )
        if hits_wall or self._would_hit_self(next_head, will_eat):
            self.done = True
            reason = "wall_collision" if hits_wall else "body_collision"
            reward = -5.0
            penalty_reasons.append(reason)
            if avoidable_collision:
                self.avoidable_collision_count += 1
                penalty_reasons.append("avoidable_collision")
        else:
            self.snake.insert(0, next_head)
            reward = -0.01
            loop_penalty = 0.0
            fill_ratio_after_move = len(self.snake) / max(1.0, float(self.board_size * self.board_size))
            turn_delta = (chosen_direction - previous_direction) % 4
            if will_eat:
                self.score += 1
                self.steps_without_food = 0
                reward = APPLE_REWARD
                reason = "apple"
                self.eaten_food_trail.append(self.food)
                self.recent_turns.clear()
                if len(self.snake) >= self.board_size * self.board_size:
                    self.done = True
                    reward += 10.0
                    reason = "victory"
                else:
                    self.food = self._spawn_food()
            else:
                removed_tail = self.snake.pop()
                try:
                    self.eaten_food_trail.remove(removed_tail)
                except ValueError:
                    pass
                snake_length_factor = max(0.6, min(2.1, len(self.snake) / 6.0))
                revisit_count = sum(1 for position in self.recent_heads if position == next_head)
                if revisit_count >= 2:
                    loop_penalty += 0.03 * (revisit_count - 1) * snake_length_factor

                if turn_delta == 1:
                    self.recent_turns.append(1)
                elif turn_delta == 3:
                    self.recent_turns.append(-1)
                else:
                    self.recent_turns.append(0)


                turning_steps = [turn for turn in self.recent_turns if turn != 0]
                if self.steps_without_food >= max(10, self.board_size // 2):
                    if len(turning_steps) >= 6 and len(turning_steps) == len(self.recent_turns):
                        if all(turn == turning_steps[0] for turn in turning_steps):
                            loop_penalty += 0.06 * snake_length_factor

                recent_window = list(self.recent_heads)[-10:]
                unique_recent = len(set(recent_window))
                if self.steps_without_food >= max(14, self.board_size):
                    if len(recent_window) >= 10 and unique_recent <= 3:
                        loop_penalty += 0.10 * snake_length_factor
                    elif len(recent_window) >= 10 and unique_recent <= 5:
                        loop_penalty += 0.04 * snake_length_factor

                max_loop_penalty = min(0.35, 0.16 + max(0, len(self.snake) - 3) * 0.015)
                if loop_penalty > max_loop_penalty:
                    loop_penalty = max_loop_penalty

                loop_reason_threshold = max(0.03, 0.055 - max(0, len(self.snake) - 3) * 0.002)
                if loop_penalty > loop_reason_threshold:
                    reason = "looping"

                if loop_penalty > 0:
                    reward -= loop_penalty
                    penalty_reasons.append("looping")

                if not self.done and reason in {"step", "looping", "apple"}:
                    if chosen_apple_line and (not endgame or will_eat):
                        reward += ROUTE_APPLE_LINE_BONUS
                        penalty_reasons.append("route_apple_line")
                    elif chosen_follows_route:
                        reward += ROUTE_FOLLOW_BONUS
                        penalty_reasons.append("route_follow")
                    elif route_available:
                        reward -= ROUTE_DEVIATION_PENALTY
                        penalty_reasons.append("route_deviation")
                    if will_eat and self._can_return_to_route_after_move(list(self.snake)):
                        reward += ROUTE_RETURN_AFTER_APPLE_BONUS
                        penalty_reasons.append("route_return")

                if will_eat and reason == "apple" and self._is_dead_end_without_alternative():
                    reward = -TRAP_AFTER_APPLE_PENALTY
                    self.trap_after_apple_count += 1
                    penalty_reasons.append("trap_after_apple")
                elif not self.done and reason in {"step", "looping", "apple"} and self._tail_path_lost_now():
                    reward = -TAIL_PATH_LOST_PENALTY
                    reason = "route_dead_end"
                    penalty_reasons.append("route_dead_end")

                body_cells = self.board_size * self.board_size
                short_snake = len(self.snake) <= 5 and self.score <= 2
                base_starvation = body_cells * (0.52 if short_snake else 0.82)
                length_allowance = max(0, len(self.snake) - 3) * self.board_size * 1.75
                starvation_limit = max(28, int(base_starvation + length_allowance))
                compact_head_loop = self._has_compact_head_loop(next_head)
                starvation_metrics: dict[str, float] | None = None
                if not self.done and reason == "looping" and short_snake and self.steps_without_food >= max(45, self.board_size * 4):
                    recent_loop_window = list(self.recent_heads)[-14:]
                    if len(recent_loop_window) >= 12 and len(set(recent_loop_window)) <= 4:
                        self.done = True
                        reward = -5.0
                        reason = "starvation_safe_loop"
                        penalty_reasons.append(reason)
                if not self.done and self.steps_without_food >= max(60, self.board_size * 5):
                    if compact_head_loop:
                        self.done = True
                        reward = -5.0
                        reason = "starvation_safe_loop"
                        penalty_reasons.append(reason)
                if self.steps_without_food > starvation_limit:
                    self.done = True
                    reward = -5.0
                    if starvation_metrics is None:
                        starvation_metrics = self._strategic_space_metrics()
                    if starvation_metrics["food_reachable"] < 0.5:
                        reason = "starvation_food_unreachable"
                    elif starvation_metrics["food_tail_safe"] < 0.5:
                        reason = "starvation_no_relayout_path"
                    elif compact_head_loop:
                        reason = "starvation_safe_loop"
                    else:
                        reason = "starvation_food_available"
                    penalty_reasons.append(reason)

                self.recent_heads.append(next_head)

        self.last_reward = reward
        self.last_reason = reason
        self.last_penalty_reasons = penalty_reasons
        # Кэш путей больше не очищается на каждом шаге, что значительно ускоряет обучение.
        # Он будет очищен только при сбросе игры или изменении размера поля.

        return np.empty((0,), dtype=np.float32), reward, self.done, {
            "score": self.score,
            "steps": self.step_count,
            "reason": reason,
            "penalty_reasons": list(penalty_reasons),
            "direction": chosen_direction,
            "apple_opportunity": apple_opportunity,
        }

    def get_local_observation_grid(self, target_size: int = DEFAULT_INPUT_GRID) -> np.ndarray:
        if target_size <= 0:
            return np.empty((0,), dtype=np.float32)

        observation = np.zeros((LOCAL_MAP_CHANNEL_COUNT, target_size, target_size), dtype=np.float32)
        center = target_size // 2
        body_cells = set(self.snake[1:])
        head = self.snake[0]
        tail = self.snake[-1] if self.snake else head

        y_coords, x_coords = np.mgrid[0:target_size, 0:target_size]
        rel_x = x_coords - center
        rel_y = y_coords - center

        if self.direction == 0:
            world_dx, world_dy = rel_x, rel_y
        elif self.direction == 1:
            world_dx, world_dy = -rel_y, rel_x
        elif self.direction == 2:
            world_dx, world_dy = -rel_x, -rel_y
        else:
            world_dx, world_dy = rel_y, -rel_x

        world_x = head[0] + world_dx
        world_y = head[1] + world_dy

        in_bounds = (world_x >= 0) & (world_x < self.board_size) & \
                   (world_y >= 0) & (world_y < self.board_size)

        observation[4][~in_bounds] = 1.0

        valid_indices = np.where(in_bounds)
        for i in range(len(valid_indices[0])):
            iy, ix = valid_indices[0][i], valid_indices[1][i]
            wx, wy = int(world_x[iy, ix]), int(world_y[iy, ix])
            
            if (wx, wy) in body_cells:
                observation[0, iy, ix] = 1.0
            if (wx, wy) == head:
                observation[1, iy, ix] = 1.0
            if (wx, wy) == self.food:
                observation[2, iy, ix] = 1.0
            if (wx, wy) == tail:
                observation[3, iy, ix] = 1.0

        return observation.reshape(-1)

    def get_extra_signals(self, target_size: int = DEFAULT_INPUT_GRID) -> np.ndarray:
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        dx = food_x - head_x
        dy = food_y - head_y
        if self.direction == 0:
            forward_component = -dy
            right_component = dx
        elif self.direction == 1:
            forward_component = dx
            right_component = dy
        elif self.direction == 2:
            forward_component = dy
            right_component = -dx
        else:
            forward_component = -dx
            right_component = -dy
        food_ahead = 1.0 if forward_component > 0 else 0.0
        food_right = 1.0 if right_component > 0 else 0.0
        food_left = 1.0 if right_component < 0 else 0.0
        apple_step_forward = 1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 0)) == self.food else 0.0
        apple_step_right = 1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 2)) == self.food else 0.0
        apple_step_left = 1.0 if self._next_head_for_direction(relative_to_absolute(self.direction, 1)) == self.food else 0.0
        danger_forward = self._danger_for_relative_action(0)
        danger_right = self._danger_for_relative_action(2)
        danger_left = self._danger_for_relative_action(1)
        forward_lookahead = self._lookahead_signals_for_relative_action(0)
        left_lookahead = self._lookahead_signals_for_relative_action(1)
        right_lookahead = self._lookahead_signals_for_relative_action(2)
        forward_safety = forward_lookahead[1:]
        left_safety = left_lookahead[1:]
        right_safety = right_lookahead[1:]
        route_direction = self._route_direction_from(self.snake[0])

        def route_signals(relative_action: int) -> tuple[float, float, float]:
            absolute_direction = relative_to_absolute(self.direction, relative_action)
            probe = self._post_move_space_probe(absolute_direction)
            blocked = float(probe.get("blocked", 1.0)) >= 0.5
            route_follow = 1.0 if not blocked and route_direction == absolute_direction else 0.0
            apple_line = 1.0 if not blocked and self._is_apple_on_direction_line(absolute_direction) else 0.0
            simulated_snake = probe.get("simulated_snake", [])
            route_return = 1.0 if (
                apple_line > 0.5
                and isinstance(simulated_snake, list)
                and self._can_return_to_route_after_move(simulated_snake)
            ) else 0.0
            return route_follow, apple_line, route_return

        forward_route_follow, forward_route_apple_line, forward_route_return = route_signals(0)
        left_route_follow, left_route_apple_line, left_route_return = route_signals(1)
        right_route_follow, right_route_apple_line, right_route_return = route_signals(2)

        signals = np.asarray(
            [
                food_ahead,
                food_right,
                food_left,
                apple_step_forward,
                apple_step_right,
                apple_step_left,
                danger_forward,
                danger_right,
                danger_left,
                *forward_safety,
                *left_safety,
                *right_safety,
                forward_route_follow,
                left_route_follow,
                right_route_follow,
                forward_route_apple_line,
                left_route_apple_line,
                right_route_apple_line,
                forward_route_return,
                left_route_return,
                right_route_return,
            ],
            dtype=np.float32,
        )
        return signals

    def get_observation_grid(self, target_size: int = DEFAULT_INPUT_GRID, input_mode: str = STARTUP_INPUT_MODE) -> np.ndarray:
        local_observation = self.get_local_observation_grid(target_size)
        if input_mode == "map":
            return local_observation.astype(np.float32, copy=False)
        extra_signals = self.get_extra_signals(target_size)
        return np.concatenate((local_observation, extra_signals), dtype=np.float32)

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

    def load_state(self, payload: dict[str, object], target_size: int = DEFAULT_INPUT_GRID) -> np.ndarray:
        try:
            board_size = int(payload.get("board_size", DEFAULT_INPUT_GRID))
            snake_values = payload.get("snake", [])
            food_value = payload.get("food", [0, 0])
            snake = [(int(pair[0]), int(pair[1])) for pair in snake_values]  # type: ignore[index]
            food = (int(food_value[0]), int(food_value[1]))  # type: ignore[index]
            if board_size not in BOARD_SIZES or not snake:
                raise ValueError("Invalid board payload")
            self.board_size = board_size
            self.direction = int(payload.get("direction", 1))
            self.score = int(payload.get("score", 0))
            self.step_count = int(payload.get("step_count", 0))
            self.steps_without_food = int(payload.get("steps_without_food", 0))
            self.max_steps_without_food = int(payload.get("max_steps_without_food", self.steps_without_food))
            self.done = bool(payload.get("done", False))
            self.last_reward = float(payload.get("last_reward", 0.0))
            self.last_reason = str(payload.get("last_reason", "new_game"))
            penalty_reasons = payload.get("last_penalty_reasons", [])
            if isinstance(penalty_reasons, list):
                self.last_penalty_reasons = [str(item) for item in penalty_reasons]
            else:
                self.last_penalty_reasons = []
            self.tail_path_lost_penalty_active = bool(payload.get("tail_path_lost_penalty_active", False))
            self.trap_after_apple_count = int(payload.get("trap_after_apple_count", 0))
            self.avoidable_collision_count = int(payload.get("avoidable_collision_count", 0))
            eaten_food_trail = payload.get("eaten_food_trail", [])
            if isinstance(eaten_food_trail, list):
                self.eaten_food_trail = deque(
                    [(int(pair[0]), int(pair[1])) for pair in eaten_food_trail if isinstance(pair, (list, tuple)) and len(pair) >= 2],
                    maxlen=24,
                )
            else:
                self.eaten_food_trail = deque(maxlen=24)
            self.snake = snake
            self.food = food
            self._clear_path_cache()
            recent_heads = payload.get("recent_heads", [])
            if recent_heads:
                self.recent_heads = deque(
                    [(int(pair[0]), int(pair[1])) for pair in recent_heads],  # type: ignore[index]
                    maxlen=18,
                )
            else:
                self.recent_heads = deque([self.snake[0]], maxlen=18)
            recent_turns = payload.get("recent_turns", [])
            if recent_turns:
                self.recent_turns = deque([int(turn) for turn in recent_turns], maxlen=8)
            else:
                self.recent_turns = deque(maxlen=8)
            return self.get_observation_grid(target_size)
        except Exception:
            self.reset()
            return self.get_observation_grid(target_size)
