from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend_config import (
    APPLE_STEP_START_IDX,
    DANGER_START_IDX,
    PPO_MAP_CHANNELS,
    PPO_MAP_SIZE,
    TAIL_BLOCKED_START_IDX,
    TAIL_LOSS_PENALTY,
)
from dqn_agent import (
    DQNAgent,
    apply_step_risk_shaping,
    compute_dqn_priority_scale,
    evaluate_pre_step_risks,
    filter_dqn_random_action,
)
from ppo_agent import PPOAgent, ParallelPPORunner
from snake_env import SnakeEnv


class TestAgents(unittest.TestCase):
    def test_dqn_sumtree_permanent_buffer_and_optimize(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        agent = DQNAgent(device=torch.device("cpu"), batch_size=16)
        s = np.zeros(agent.global_signal_count, dtype=np.float32)

        for i in range(40):
            scale = 4.0 if i % 10 == 0 else (3.0 if i % 7 == 0 else 1.0)
            agent.remember(s, i % 3, float(i % 4), s, i % 15 == 0, priority_scale=scale)

        self.assertEqual(len(agent.replay_buffer), 40)
        self.assertGreater(agent.replay_buffer._perm_size, 0)
        self.assertLess(agent.replay_buffer.max_priority, 100.0)
        loss = agent.optimize()
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))

    def test_replay_buffer_no_exponential_priority_explosion(self) -> None:
        agent = DQNAgent(device=torch.device("cpu"))
        s = np.zeros(agent.global_signal_count, dtype=np.float32)
        for _ in range(100):
            agent.remember(s, 0, -35.0, s, True, priority_scale=5.0)
        self.assertLess(agent.replay_buffer.max_priority, 10.0)

    def test_dqn_checkpoint_round_trip_float16_and_float32(self) -> None:
        agent = DQNAgent(device=torch.device("cpu"))
        with torch.no_grad():
            agent.online_net.hidden1_layer.weight.fill_(0.25)
        ckpt = agent.export_checkpoint()
        self.assertEqual(ckpt["online_net"]["hidden1_layer.weight"].dtype, torch.float32)

        # Simulate legacy float16 checkpoint
        legacy_ckpt = {
            "online_net": {k: v.to(torch.float16) for k, v in ckpt["online_net"].items()},
            "target_net": {k: v.to(torch.float16) for k, v in ckpt["target_net"].items()},
            "optimizer": ckpt["optimizer"],
        }
        restored = DQNAgent(device=torch.device("cpu"))
        restored.import_checkpoint(legacy_ckpt)
        self.assertAlmostEqual(float(restored.online_net.hidden1_layer.weight[0, 0].detach()), 0.25, places=3)

    def test_ppo_map_agent_action_mask_optimize_and_checkpoint(self) -> None:
        torch.manual_seed(7)
        np.random.seed(7)
        agent = PPOAgent(
            device=torch.device("cpu"),
            num_envs=2,
            rollout_steps=8,
            minibatch_size=8,
            epochs=2,
        )
        sig = np.zeros(agent.global_signal_count, dtype=np.float32)
        sig[DANGER_START_IDX + 0] = 1.0
        sig[DANGER_START_IDX + 1] = 1.0  # only right (2) is safe
        ego = np.zeros((PPO_MAP_CHANNELS, PPO_MAP_SIZE, PPO_MAP_SIZE), dtype=np.float32)

        for _ in range(10):
            act, probs, _, greedy = agent.select_action(sig, ego_map=ego, training=True)
            self.assertEqual(act, 2)
            self.assertEqual(greedy, 2)
            self.assertAlmostEqual(float(probs[0]), 0.0, places=4)
            self.assertAlmostEqual(float(probs[1]), 0.0, places=4)
            self.assertAlmostEqual(float(probs[2]), 1.0, places=4)

        runner = ParallelPPORunner(2, 10, env_factory=SnakeEnv)
        for _ in range(9):
            runner.step_alive(agent, record_replays=False)
        self.assertGreaterEqual(agent.total_updates, 1)
        self.assertTrue(np.isfinite(agent.last_loss))

        with torch.no_grad():
            agent.online_net.hidden1_layer.weight.fill_(0.15)
        ckpt = agent.export_checkpoint()
        restored = PPOAgent(device=torch.device("cpu"), num_envs=2, rollout_steps=8)
        restored.import_checkpoint(ckpt)
        self.assertEqual(restored.total_updates, agent.total_updates)
        self.assertAlmostEqual(float(restored.online_net.hidden1_layer.weight[0, 0].detach()), 0.15, places=4)

    def test_shared_dqn_step_helpers(self) -> None:
        env = SnakeEnv(board_size=10)
        obs = env.reset()

        # Random filter overrides dangerous random move with greedy move
        bad_state = np.zeros_like(obs)
        bad_state[DANGER_START_IDX + 1] = 1.0
        bad_state[APPLE_STEP_START_IDX + 0] = 1.0
        act, rnd = filter_dqn_random_action(bad_state, predicted_action=1, greedy_action=0, was_random=True)
        self.assertEqual(act, 0)
        self.assertFalse(rnd)

        # Risk shaping and priority scale
        risks = evaluate_pre_step_risks(env, obs, 0)
        shaped, reasons = apply_step_risk_shaping(0.0, "step", len(env.snake), (True, False, -1, False))
        self.assertEqual(shaped, -TAIL_LOSS_PENALTY)
        self.assertEqual(reasons, ["tail_loss"])
        split_shaped, split_reasons = apply_step_risk_shaping(
            0.0,
            "step",
            80,
            (False, False, -1, False),
            is_endgame=True,
            is_pocket_split=True,
            endgame_split_penalty=0.45,
        )
        self.assertAlmostEqual(split_shaped, -0.45)
        self.assertEqual(split_reasons, ["pocket_split"])
        merge_shaped, _ = apply_step_risk_shaping(
            0.0,
            "step",
            80,
            (False, False, -1, False),
            is_endgame=True,
            is_pocket_merge=True,
            endgame_merge_bonus=0.25,
        )
        self.assertAlmostEqual(merge_shaped, 0.25)
        self.assertEqual(compute_dqn_priority_scale(False, "apple", False, (0, 0), 10), 4.0)
        self.assertEqual(compute_dqn_priority_scale(True, "wall_collision", False, (0, 1), 10), 5.0)
        self.assertEqual(
            compute_dqn_priority_scale(
                False,
                "step",
                False,
                (2, 2),
                10,
                is_pocket_split=True,
                is_endgame=True,
                pocket_split_priority=2.5,
                endgame_priority=2.0,
            ),
            2.5,
        )
        self.assertEqual(len(risks), 4)

    def test_dqn_nstep_and_critical_hunger_shaping(self) -> None:
        agent = DQNAgent(device=torch.device("cpu"), gamma=0.9, batch_size=4, n_step=3)
        obs = np.zeros(agent.global_signal_count, dtype=np.float32)
        agent.remember(obs, 0, 1.0, obs, False)
        agent.remember(obs, 1, 2.0, obs, False)
        self.assertEqual(len(agent.replay_buffer), 0)
        agent.remember(obs, 2, 3.0, obs, True, priority_scale=5.0)
        # Episode end flushes all 3 queued transitions (3-step, 2-step, 1-step)
        self.assertEqual(len(agent.replay_buffer), 3)
        self.assertAlmostEqual(float(agent.replay_buffer._rewards[0]), 1.0 + 0.9 * 2.0 + 0.81 * 3.0, places=5)
        self.assertAlmostEqual(float(agent.replay_buffer._discounts[0]), 0.9 ** 3, places=5)
        agent.remember(obs, 0, 0.5, obs, True)
        self.assertEqual(len(agent.replay_buffer), 4)
        loss = agent.optimize()
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))

        hungry_reward, _ = apply_step_risk_shaping(
            0.0,
            "step",
            40,
            (False, False, -1, False),
            is_hunger_critical=True,
            critical_hunger_penalty=0.08,
        )
        self.assertAlmostEqual(hungry_reward, -0.08)

    def test_dqn_soft_target_and_cosine_lr_decay(self) -> None:
        agent = DQNAgent(
            device=torch.device("cpu"),
            batch_size=8,
            learning_rate=0.0008,
            lr_min=0.0002,
            lr_decay_updates=10,
            target_tau=0.05,
        )
        obs = np.zeros(agent.global_signal_count, dtype=np.float32)
        for i in range(10):
            agent.remember(obs, i % 3, 1.0, obs, False)
        before_target = [p.detach().clone() for p in agent.target_net.parameters()]
        agent.optimize()
        self.assertLess(agent.optimizer.param_groups[0]["lr"], 0.0008)
        self.assertTrue(
            any(not torch.equal(b, a.detach()) for b, a in zip(before_target, agent.target_net.parameters()))
        )

    def test_dqn_custom_hidden_sizes_and_frame_stack(self) -> None:
        agent = DQNAgent(device=torch.device("cpu"), hidden1=32, hidden2=16, frame_stack=2)
        self.assertEqual(agent.input_size, agent.global_signal_count * 2)
        self.assertEqual(agent.online_net.hidden1_layer.out_features, 32)
        self.assertEqual(agent.online_net.hidden2_layer.out_features, 16)
        obs = np.zeros(agent.input_size, dtype=np.float32)
        act, q_vals, was_rnd, greedy = agent.select_action(obs, training=False)
        self.assertIn(act, (0, 1, 2))
        self.assertEqual(q_vals.shape, (3,))
        self.assertFalse(was_rnd)
        self.assertEqual(act, greedy)
        ckpt = agent.export_checkpoint()
        self.assertEqual(ckpt["arch_meta"]["hidden1"], 32)
        self.assertEqual(ckpt["arch_meta"]["frame_stack"], 2)

    def test_dqn_delayed_lr_decay_start(self) -> None:
        agent = DQNAgent(
            device=torch.device("cpu"),
            batch_size=4,
            learning_rate=0.001,
            lr_min=0.0001,
            lr_decay_updates=4,
            lr_decay_start_update=2,
        )
        obs = np.zeros(agent.global_signal_count, dtype=np.float32)
        for i in range(6):
            agent.remember(obs, i % 3, 1.0, obs, False)
        agent.optimize()
        agent.optimize()
        self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 0.001, places=7)
        agent.optimize()
        self.assertLess(agent.optimizer.param_groups[0]["lr"], 0.001)


if __name__ == "__main__":
    unittest.main()
