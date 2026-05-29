from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest

if importlib.util.find_spec("numpy") is None:
    class FakeArray(list):
        @property
        def size(self) -> int:
            return len(self)

        def tolist(self) -> list[float]:
            return list(self)

        def copy(self) -> "FakeArray":
            return FakeArray(self)

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = "float32"
    fake_numpy.ndarray = FakeArray
    fake_numpy.array = lambda values, dtype=None: FakeArray(values)
    fake_numpy.empty = lambda shape, dtype=None: FakeArray(
        [0.0] * (shape[0] if isinstance(shape, tuple) else int(shape))
    )
    sys.modules["numpy"] = fake_numpy

backend_config = importlib.import_module("backend_config")
snake_env = importlib.import_module("snake_env")

GLOBAL_SIGNAL_LABELS = backend_config.GLOBAL_SIGNAL_LABELS
SnakeEnv = snake_env.SnakeEnv


def signal_map(env: SnakeEnv) -> dict[str, float]:
    values = env.get_extra_signals().tolist()
    return dict(zip(GLOBAL_SIGNAL_LABELS, values, strict=True))


class SnakeEnvSignalTests(unittest.TestCase):
    def test_signal_label_order_is_canonical(self) -> None:
        self.assertEqual(
            list(GLOBAL_SIGNAL_LABELS),
            [
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
            ],
        )

    def test_observation_size_matches_signal_labels(self) -> None:
        env = SnakeEnv(board_size=5)
        observation = env.get_observation_grid()
        self.assertEqual(observation.size, len(GLOBAL_SIGNAL_LABELS))
        self.assertEqual(len(GLOBAL_SIGNAL_LABELS), 20)

    def test_relative_signal_order_is_forward_left_right(self) -> None:
        env = SnakeEnv(board_size=5)
        env.snake = [(2, 2), (1, 2), (0, 2)]
        env.direction = 1  # moving right: left is up, right is down
        env.food = (2, 1)
        env._clear_observation_cache()

        signals = signal_map(env)
        self.assertEqual(signals["food_left"], 1.0)
        self.assertEqual(signals["food_right"], 0.0)
        self.assertEqual(signals["apple_step_left"], 1.0)
        self.assertEqual(signals["apple_step_right"], 0.0)

    def test_danger_order_is_forward_left_right(self) -> None:
        env = SnakeEnv(board_size=5)
        env.snake = [(2, 2), (2, 1), (1, 1), (1, 2)]
        env.direction = 1  # moving right: left cell (2, 1) is body
        env.food = (4, 4)
        env._clear_observation_cache()

        signals = signal_map(env)
        self.assertEqual(signals["danger_left"], 1.0)
        self.assertEqual(signals["danger_right"], 0.0)

    def test_tail_behind_signal(self) -> None:
        env = SnakeEnv(board_size=5)
        env.snake = [(2, 2), (1, 2), (0, 2)]
        env.direction = 1
        env.food = (4, 4)
        env._clear_observation_cache()

        signals = signal_map(env)
        self.assertEqual(signals["tail_behind"], 1.0)
        self.assertEqual(signals["tail_ahead"], 0.0)

    def test_load_state_clears_observation_cache(self) -> None:
        env = SnakeEnv(board_size=5)
        env.snake = [(2, 2), (1, 2), (0, 2)]
        env.direction = 1
        env.food = (4, 2)
        env.step_count = 7
        env._clear_observation_cache()
        before = signal_map(env)
        self.assertEqual(before["food_ahead"], 1.0)

        env.load_state(
            {
                "board_size": 5,
                "snake": [[2, 2], [1, 2], [0, 2]],
                "food": [0, 2],
                "direction": 1,
                "score": 0,
                "step_count": 7,
                "steps_without_food": 0,
                "done": False,
            }
        )
        after = signal_map(env)
        self.assertEqual(after["food_ahead"], 0.0)
        self.assertEqual(after["food_behind"], 1.0)

    def test_tail_blocked_treats_apple_move_as_growth(self) -> None:
        env = SnakeEnv(board_size=5)
        env.snake = [(2, 2), (1, 2), (1, 3), (2, 3)]
        env.direction = 1
        env.food = (3, 2)
        env._clear_observation_cache()

        self.assertEqual(env._is_tail_blocked_after_relative_action(0), 1.0)

    def test_space_tight_uses_grown_snake_length(self) -> None:
        env = SnakeEnv(board_size=5)
        env.snake = [(2, 2), (1, 2), (0, 2)]
        env.direction = 1
        env.food = (3, 2)
        env._reachable_space_size = lambda start, blocked: len(env.snake)  # type: ignore[method-assign]

        self.assertEqual(env._is_space_tight_after_relative_action(0), 1.0)

    def test_step_returns_current_observation(self) -> None:
        env = SnakeEnv(board_size=5)
        observation, _reward, _done, _info = env.step(env.direction)
        self.assertEqual(observation.size, len(GLOBAL_SIGNAL_LABELS))


if __name__ == "__main__":
    unittest.main()
