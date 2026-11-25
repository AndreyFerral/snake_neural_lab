from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend_config import (
    APPLE_REWARD,
    DEATH_PENALTY,
    GLOBAL_SIGNAL_LABELS,
    PPO_MAP_CHANNELS,
    PPO_MAP_SIZE,
    PPO_SIGNAL_LABELS,
)
from snake_env import SnakeEnv, relative_to_absolute


class TestSnakeEnv(unittest.TestCase):
    def test_reset_initial_state_and_observation_shape(self) -> None:
        env = SnakeEnv(board_size=10)
        obs = env.reset()
        self.assertEqual(obs.shape, (len(GLOBAL_SIGNAL_LABELS),))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(len(env.snake), 3)
        self.assertNotIn(env.food, env.snake)
        self.assertFalse(env.done)

    def test_opposite_direction_ignored(self) -> None:
        env = SnakeEnv(board_size=10)
        env.reset()
        env.direction = 1  # right
        _, _, done, info = env.step(3)  # opposite: left
        self.assertFalse(done)
        self.assertEqual(info["direction"], 1)

    def test_wall_collision_gives_death_penalty(self) -> None:
        env = SnakeEnv(board_size=10)
        env.reset()
        env.snake = [(9, 5), (8, 5), (7, 5)]
        env.direction = 1
        env.food = (0, 0)
        env._clear_observation_cache()
        _, reward, done, info = env.step(1)
        self.assertTrue(done)
        self.assertEqual(info["reason"], "wall_collision")
        self.assertEqual(reward, -DEATH_PENALTY)

    def test_eating_apple_grows_snake_and_gives_reward(self) -> None:
        env = SnakeEnv(board_size=10)
        env.reset()
        env.snake = [(4, 5), (3, 5), (2, 5)]
        env.direction = 1
        env.food = (5, 5)
        env._clear_observation_cache()
        _, reward, done, info = env.step(1)
        self.assertFalse(done)
        self.assertEqual(info["reason"], "apple")
        self.assertEqual(reward, APPLE_REWARD)
        self.assertEqual(env.score, 1)
        self.assertEqual(len(env.snake), 4)

    def test_relative_move_risks_matches_observation_signals(self) -> None:
        env = SnakeEnv(board_size=10)
        env.reset()
        obs = env.get_observation_grid()
        risks = env.relative_move_risks()
        for idx, key in enumerate(("forward", "left", "right")):
            self.assertEqual(risks[key]["danger"], float(obs[8 + idx]))
            self.assertEqual(risks[key]["tail_blocked"], float(obs[15 + idx]))
            self.assertEqual(risks[key]["space_tight"], env._is_space_tight_after_relative_action(idx))

    def test_export_load_state_round_trip(self) -> None:
        env = SnakeEnv(board_size=15)
        env.reset()
        for act in (0, 2, 0, 1):
            env.step(relative_to_absolute(env.direction, act))
        exported = env.export_state()

        restored = SnakeEnv(board_size=10)
        obs_restored = restored.load_state(exported)
        self.assertEqual(restored.board_size, 15)
        self.assertEqual(restored.snake, env.snake)
        self.assertEqual(restored.food, env.food)
        self.assertEqual(restored.max_steps_without_food, env.max_steps_without_food)
        self.assertTrue(np.array_equal(obs_restored, env.get_observation_grid()))

    def test_endgame_and_pocket_split_signals(self) -> None:
        env = SnakeEnv(board_size=10)
        env.reset()
        # Snake cutting across column x=5 with head at (5, 1) moving up (dir=0):
        # Forward (5,0) touches wall edge and splits left/right empty regions -> pocket_split_forward = 1.0
        env.snake = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (4, 9)]
        env.direction = 0
        env.food = (1, 1)
        env._clear_observation_cache()
        obs = env.get_observation_grid()
        idx_endgame = GLOBAL_SIGNAL_LABELS.index("endgame")
        idx_split_fwd = GLOBAL_SIGNAL_LABELS.index("pocket_split_forward")
        idx_split_left = GLOBAL_SIGNAL_LABELS.index("pocket_split_left")
        self.assertEqual(float(obs[idx_endgame]), 0.0)
        self.assertEqual(float(obs[idx_split_fwd]), 1.0)
        self.assertEqual(float(obs[idx_split_left]), 0.0)

        # Length >= 75 on 10x10 triggers endgame = 1.0
        snake_75 = []
        for y in range(10):
            xs = range(10) if y % 2 == 0 else range(9, -1, -1)
            for x in xs:
                snake_75.append((x, y))
                if len(snake_75) >= 75:
                    break
            if len(snake_75) >= 75:
                break
        snake_75.reverse()
        env.snake = snake_75
        env.direction = 1
        env.food = (9, 9)
        env._clear_observation_cache()
        obs_end = env.get_observation_grid()
        self.assertEqual(float(obs_end[idx_endgame]), 1.0)

        # Binary hunger threshold signals
        idx_h_high = GLOBAL_SIGNAL_LABELS.index("hunger_high")
        idx_h_crit = GLOBAL_SIGNAL_LABELS.index("hunger_critical")
        env.steps_without_food = 0
        env._clear_observation_cache()
        obs_fed = env.get_observation_grid()
        self.assertEqual(float(obs_fed[idx_h_high]), 0.0)
        self.assertEqual(float(obs_fed[idx_h_crit]), 0.0)
        env.steps_without_food = 1000
        env._clear_observation_cache()
        obs_starving = env.get_observation_grid()
        self.assertEqual(float(obs_starving[idx_h_high]), 1.0)
        self.assertEqual(float(obs_starving[idx_h_crit]), 1.0)

        # Already split board sets board_split = 1.0
        idx_board_split = GLOBAL_SIGNAL_LABELS.index("board_split")
        env.snake = [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9)]
        env.direction = 1
        env.food = (1, 1)
        env._clear_observation_cache()
        obs_split = env.get_observation_grid()
        self.assertEqual(float(obs_split[idx_board_split]), 1.0)

    def test_egocentric_map_shape_rotation_and_channels(self) -> None:
        env = SnakeEnv(board_size=10)
        env.reset()
        # Head at (9, 5), moving RIGHT (dir=1): ahead (x=10) is out of bounds -> row=3, col=4 wall=1.0
        # Food 2 cells ahead? Wait: put food to the LEFT of head -> (9, 4) -> row=4, col=3 food=1.0
        env.snake = [(9, 5), (8, 5), (7, 5)]
        env.direction = 1
        env.food = (9, 4)
        env._clear_observation_cache()
        ego_map = env.get_egocentric_map()
        self.assertEqual(ego_map.shape, (PPO_MAP_CHANNELS, PPO_MAP_SIZE, PPO_MAP_SIZE))
        self.assertEqual(ego_map.dtype, np.float32)

        center = PPO_MAP_SIZE // 2
        # Wall ahead (row = center - 1, col = center)
        self.assertEqual(float(ego_map[0, center - 1, center]), 1.0)
        # Head TTL at center = 1.0, body behind at row=center, col=center-1 is blocked? Wait:
        # direction=1 (right) -> behind is LEFT column (col = center - 1? No! fwd=-1 is row = center + 1!)
        self.assertAlmostEqual(float(ego_map[1, center, center]), 1.0)
        self.assertGreater(float(ego_map[1, center + 1, center]), 0.5)
        self.assertGreater(float(ego_map[1, center + 2, center]), 0.1)
        # Food at (9, 4): when moving right, y-1 is LEFT -> col = center - 1, row = center
        self.assertEqual(float(ego_map[2, center, center - 1]), 1.0)
        # Reachable open cell at (9, 4) -> reach=1.0; wall cell has reach=0.0
        self.assertEqual(float(ego_map[3, center, center - 1]), 1.0)
        self.assertEqual(float(ego_map[3, center - 1, center]), 0.0)

    def test_ppo_area_ratios_and_extended_signals_do_not_change_dqn_signals(self) -> None:
        env = SnakeEnv(board_size=10)
        env.reset()
        # Head at right wall (9, 5) facing right -> forward is wall (0.0), left/right open (> 0.5)
        env.snake = [(9, 5), (8, 5), (7, 5)]
        env.direction = 1
        env.food = (2, 2)
        env._clear_observation_cache()

        dqn_obs = env.get_observation_grid()
        self.assertEqual(dqn_obs.shape, (len(GLOBAL_SIGNAL_LABELS),))
        self.assertEqual(len(GLOBAL_SIGNAL_LABELS), 34)

        ratios = env.get_ppo_area_ratios()
        self.assertEqual(ratios.shape, (3,))
        self.assertEqual(float(ratios[0]), 0.0)
        self.assertGreater(float(ratios[1]), 0.8)
        self.assertGreater(float(ratios[2]), 0.8)

        ppo_obs = env.get_ppo_signals()
        self.assertEqual(ppo_obs.shape, (len(PPO_SIGNAL_LABELS),))
        self.assertTrue(np.array_equal(ppo_obs[: len(GLOBAL_SIGNAL_LABELS)], dqn_obs))
        self.assertTrue(np.array_equal(ppo_obs[len(GLOBAL_SIGNAL_LABELS) :], ratios))
        self.assertEqual(PPO_MAP_SIZE, 11)


if __name__ == "__main__":
    unittest.main()
