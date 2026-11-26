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
    PPO_SIGNAL_LABELS,
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
from ppo_agent import PPOAgent, ParallelPPORunner, build_multiboard_layout, format_multiboard_summary
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
            agent.online_net.critic_hidden1_layer.weight.fill_(-0.25)
        self.assertEqual(agent.global_signal_count, len(PPO_SIGNAL_LABELS))
        snap = agent.snapshot(sig, ego_map=ego)
        self.assertIn("critic_hidden2", snap)
        self.assertEqual(snap["hidden1"].shape[0], agent.online_net.hidden1_size)
        self.assertEqual(snap["hidden2"].shape[0], agent.online_net.hidden2_size)

        ckpt = agent.export_checkpoint()
        restored = PPOAgent(device=torch.device("cpu"), num_envs=2, rollout_steps=8)
        restored.import_checkpoint(ckpt)
        self.assertEqual(restored.total_updates, agent.total_updates)
        self.assertAlmostEqual(float(restored.online_net.hidden1_layer.weight[0, 0].detach()), 0.15, places=4)
        self.assertAlmostEqual(float(restored.online_net.critic_hidden1_layer.weight[0, 0].detach()), -0.25, places=4)

    def test_ppo_kl_early_stop_tail_soft_bias_and_best_snapshot_protection(self) -> None:
        torch.manual_seed(11)
        np.random.seed(11)
        agent = PPOAgent(
            device=torch.device("cpu"),
            num_envs=2,
            rollout_steps=8,
            minibatch_size=8,
            epochs=6,
            lr=0.05,
            target_kl=0.001,
            tail_blocked_logit_bias=0.5,
        )
        # Soft preference for non-tail_blocked moves when safe alternative exists
        with torch.no_grad():
            agent.online_net.output.weight.zero_()
            agent.online_net.output.bias.zero_()
        sig = np.zeros(agent.global_signal_count, dtype=np.float32)
        sig[TAIL_BLOCKED_START_IDX + 0] = 1.0  # forward tail_blocked; left & right OK
        ego = np.zeros((PPO_MAP_CHANNELS, PPO_MAP_SIZE, PPO_MAP_SIZE), dtype=np.float32)
        _, probs, _, _ = agent.select_action(sig, ego_map=ego, training=False)
        self.assertLess(float(probs[0]), float(probs[1]))
        self.assertLess(float(probs[0]), float(probs[2]))

        # Large lr + tight target_kl should trigger KL early stopping cleanly
        runner = ParallelPPORunner(2, 10, env_factory=SnakeEnv)
        for _ in range(9):
            runner.step_alive(agent, record_replays=False)
        self.assertGreaterEqual(agent.total_updates, 1)
        self.assertGreaterEqual(agent.kl_early_stops, 1)
        self.assertTrue(np.isfinite(agent.last_approx_kl))

        # Best policy snapshot & soft rollback on sustained degradation
        with torch.no_grad():
            agent.online_net.hidden2_layer.weight.fill_(0.42)
        for _ in range(agent.best_min_episodes):
            agent.record_full_episode(40, "body_collision")
        self.assertIsNotNone(agent.best_policy_state)
        self.assertAlmostEqual(agent.best_avg_score, 40.0, places=3)

        # Corrupt policy weights and feed low episode scores to trigger soft restore
        with torch.no_grad():
            agent.online_net.hidden2_layer.weight.fill_(-0.99)
        for _ in range(35):
            agent.record_full_episode(5, "wall_collision")
        self.assertGreaterEqual(agent.restores_from_best, 1)
        self.assertAlmostEqual(float(agent.online_net.hidden2_layer.weight[0, 0].detach()), 0.42, places=4)

        # Best policy snapshot survives checkpoint round-trip
        ckpt = agent.export_checkpoint()
        self.assertIn("best_policy_net", ckpt)
        restored = PPOAgent(device=torch.device("cpu"))
        restored.import_checkpoint(ckpt)
        self.assertIsNotNone(restored.best_policy_state)
        self.assertAlmostEqual(restored.best_avg_score, 40.0, places=3)

    def test_parallel_ppo_runner_failure_replay_and_honest_stats(self) -> None:
        torch.manual_seed(3)
        np.random.seed(3)
        agent = PPOAgent(device=torch.device("cpu"), num_envs=4, rollout_steps=64)
        runner = ParallelPPORunner(
            4,
            10,
            env_factory=SnakeEnv,
            replay_env_count=1,
            failure_min_score=25,
            failure_snapshot_interval=2,
            failure_rewind_target=4,
        )
        # Seed env 0 with score >= 25 and several snapshots, then force imminent wall collision
        env0 = runner.envs[0]
        env0.reset()
        env0.score = 30
        env0.step_count = 10
        env0.snake = [(5, 5), (4, 5), (3, 5)]
        env0.direction = 1
        env0.food = (0, 0)
        env0._clear_observation_cache()
        runner._state_snapshots[0].append((10, env0.export_state()))
        # Move env0 into a doomed position (all 3 moves blocked: wall, wall, non-tail body) at step 22
        env0.step_count = 22
        env0.snake = [(9, 0), (8, 0), (8, 1), (9, 1), (9, 2)]
        env0.direction = 1
        env0.food = (0, 9)
        env0._clear_observation_cache()

        completed = runner.step_alive(agent, record_replays=False)
        honest_indices = [c["index"] for c in completed]
        self.assertIn(0, honest_indices)
        self.assertEqual(len(runner.failure_replay_buffer), 1)

        # Force replay slot (index 3) to finish its initial game and load from failure buffer
        env3 = runner.envs[3]
        env3.snake = [(9, 0), (8, 0), (8, 1), (9, 1), (9, 2)]
        env3.direction = 1
        env3.food = (0, 9)
        env3._clear_observation_cache()
        runner.step_alive(agent, record_replays=False)
        self.assertTrue(runner.is_replay_env[3])

        # Now kill the replay episode on slot 3: it must NOT appear in `completed`
        env3.snake = [(9, 0), (8, 0), (8, 1), (9, 1), (9, 2)]
        env3.direction = 1
        env3.food = (0, 9)
        env3._clear_observation_cache()
        completed_after = runner.step_alive(agent, record_replays=False)
        self.assertNotIn(3, [c["index"] for c in completed_after])
        self.assertGreaterEqual(runner.failure_replays_completed, 1)

        # Board size change / reset_all clears failure replay buffer
        runner.reset_all(board_size=15)
        self.assertEqual(len(runner.failure_replay_buffer), 0)
        self.assertEqual(len(runner.endgame_failure_buffer), 0)
        self.assertFalse(any(runner.is_replay_env))

    def test_ppo_tail_bias_annealing_board_switch_and_endgame_failure_replay(self) -> None:
        agent = PPOAgent(device=torch.device("cpu"), tail_blocked_logit_bias=0.5)
        agent.best_fill_ratio = 0.50
        self.assertAlmostEqual(agent.effective_tail_bias(), 0.5, places=5)
        agent.best_fill_ratio = 0.80
        self.assertAlmostEqual(agent.effective_tail_bias(), 0.25, places=5)
        agent.best_fill_ratio = 0.90
        self.assertAlmostEqual(agent.effective_tail_bias(), 0.0, places=5)
        agent.best_fill_ratio = 0.98
        self.assertAlmostEqual(agent.effective_tail_bias(), 0.0, places=5)

        # Board size switch adapts LR/entropy on unseen larger map and preserves per-board fill
        agent.total_updates = 600
        for _ in range(agent.best_min_episodes):
            agent.record_full_episode(90, "body_collision", board_size=10)
        self.assertGreater(agent.best_fill_by_board.get(10, 0.0), 0.90)
        lr_polished = agent._current_lr()
        ent_polished = agent._current_ent_coef()

        agent.on_board_size_changed(15)
        self.assertAlmostEqual(agent._board_adapt_boost, 1.0, places=4)
        self.assertEqual(agent._critic_warmup_updates, 6)
        self.assertAlmostEqual(agent.effective_tail_bias(), 0.0, places=5)
        self.assertGreater(agent._current_lr(), lr_polished)
        self.assertLessEqual(agent._current_lr(), 1.31e-4)
        self.assertGreater(agent._current_ent_coef(), ent_polished)
        self.assertLessEqual(agent._current_ent_coef(), 0.0031)

        for _ in range(agent.best_min_episodes):
            agent.record_full_episode(150, "starvation", board_size=15)
        self.assertGreater(agent.best_fill_by_board.get(15, 0.0), 0.60)
        ckpt = agent.export_checkpoint()
        restored = PPOAgent(device=torch.device("cpu"))
        restored.import_checkpoint(ckpt)
        self.assertAlmostEqual(restored.best_fill_by_board.get(15, 0.0), agent.best_fill_by_board[15], places=4)

        # SWA->EMA averaging activates once fill >= 0.90, feeds inference_net, round-trips, and resets on board switch
        ema_agent = PPOAgent(device=torch.device("cpu"))
        ema_agent.best_fill_ratio = 0.95
        with torch.no_grad():
            ema_agent.online_net.output.weight.fill_(1.0)
        for _ in range(4):
            ema_agent._update_ema_policy()
        with torch.no_grad():
            ema_agent.online_net.output.weight.fill_(3.0)
        for _ in range(4):
            ema_agent._update_ema_policy()
        ema_agent.sync_inference_net()
        self.assertTrue(ema_agent.ema_active)
        self.assertEqual(ema_agent.ema_updates, 8)
        # Running SWA mean of four 1.0s and four 3.0s is exactly 2.0
        self.assertAlmostEqual(float(ema_agent.inference_net.output.weight[0, 0].detach()), 2.0, places=4)
        ema_ckpt = ema_agent.export_checkpoint()
        ema_restored = PPOAgent(device=torch.device("cpu"))
        ema_restored.import_checkpoint(ema_ckpt)
        self.assertTrue(ema_restored.ema_active)
        self.assertEqual(ema_restored.ema_updates, 8)
        self.assertAlmostEqual(float(ema_restored.inference_net.output.weight[0, 0].detach()), 2.0, places=4)
        ema_restored.on_board_size_changed(20)
        self.assertFalse(ema_restored.ema_active)
        self.assertEqual(ema_restored.ema_updates, 0)
        # Adaptive polish rollout batch: 66 early -> 132 at fill >= 0.90 (and drops back to 66 during board adapt)
        batch_agent = PPOAgent(device=torch.device("cpu"))
        self.assertEqual(batch_agent.num_envs, 16)
        self.assertEqual(batch_agent.online_net.hidden1_size, 128)
        self.assertEqual(batch_agent.online_net.hidden2_size, 64)
        self.assertEqual(batch_agent.effective_rollout_steps(), 66)
        batch_agent.best_fill_ratio = 0.92
        self.assertEqual(batch_agent.effective_rollout_steps(), 132)
        batch_agent.best_fill_ratio = 0.98
        self.assertLessEqual(batch_agent._current_lr(), 1.2e-5)
        self.assertAlmostEqual(batch_agent._current_clip_coef(), 0.10, places=4)
        self.assertAlmostEqual(float(batch_agent._current_target_kl() or 0.0), 0.008, places=4)
        batch_agent.total_updates = 400
        batch_agent.on_board_size_changed(15)
        self.assertEqual(batch_agent.effective_rollout_steps(), 66)

        # Anti-drift: Quality-Gated EMA freezes on drop > 3.5%, and tight endgame rollback fires on -6.5% drop
        drift_agent = PPOAgent(device=torch.device("cpu"))
        for _ in range(drift_agent.best_min_episodes):
            drift_agent.record_full_episode(95, "body_collision", board_size=10)
        self.assertGreaterEqual(drift_agent.best_fill_ratio, 0.97)
        with torch.no_grad():
            drift_agent.online_net.output.weight.fill_(1.0)
        for _ in range(6):
            drift_agent._update_ema_policy()
        ema_before = float(drift_agent.ema_policy_state["output.weight"][0, 0].detach())
        # Simulate degradation (score 85 -> fill 0.88 vs best 0.98 -> drop 0.10 > 0.035 & > 0.065)
        with torch.no_grad():
            drift_agent.online_net.output.weight.fill_(-5.0)
        for _ in range(drift_agent.best_min_episodes):
            drift_agent.record_full_episode(85, "body_collision", board_size=10)
        # EMA must stay frozen while degraded
        drift_agent._update_ema_policy()
        self.assertAlmostEqual(float(drift_agent.ema_policy_state["output.weight"][0, 0].detach()), ema_before, places=4)
        # Sustained endgame drift (3 windows at fill >= 0.98) triggers rollback without waiting for score 67
        for _ in range(48):
            drift_agent.record_full_episode(85, "body_collision", board_size=10)
        self.assertGreaterEqual(drift_agent.restores_from_best, 1)
        self.assertGreater(drift_agent._post_restore_cooldown, 0)

        # Ceiling refresh: once best_fill_ratio == 1.0, best_policy_updates still refreshes after >= 50 updates
        ceil_agent = PPOAgent(device=torch.device("cpu"))
        ceil_agent.total_updates = 100
        for _ in range(ceil_agent.best_min_episodes):
            ceil_agent.record_full_episode(97, "victory", board_size=10)
        self.assertEqual(ceil_agent.best_fill_ratio, 1.0)
        self.assertEqual(ceil_agent.best_policy_updates, 100)
        ceil_agent.total_updates = 160
        refreshed = ceil_agent.record_full_episode(97, "victory", board_size=10)
        self.assertTrue(refreshed)
        self.assertEqual(ceil_agent.best_policy_updates, 160)

        # Two-tier endgame failure replay + stale pruning after 6 failures
        runner = ParallelPPORunner(4, 10, env_factory=SnakeEnv, replay_env_count=1, failure_min_score=25)
        env0 = runner.envs[0]
        env0.reset()
        env0.score = 82
        env0.step_count = 10
        env0.snake = [(5, 5), (4, 5), (3, 5)]
        env0.direction = 1
        env0.food = (0, 0)
        env0._clear_observation_cache()
        runner._state_snapshots[0].append((10, env0.export_state()))
        env0.step_count = 25
        env0.snake = [(9, 0), (8, 0), (8, 1), (9, 1), (9, 2)]
        env0.direction = 1
        env0.food = (0, 9)
        env0._clear_observation_cache()
        runner.step_alive(agent, record_replays=False)
        self.assertEqual(len(runner.endgame_failure_buffer), 1)
        entry = runner.endgame_failure_buffer[0]
        entry["fails"] = 5
        runner.is_replay_env[3] = True
        runner._active_replay_entry[3] = entry
        env3 = runner.envs[3]
        env3.snake = [(9, 0), (8, 0), (8, 1), (9, 1), (9, 2)]
        env3.direction = 1
        env3.food = (0, 9)
        env3._clear_observation_cache()
        runner.step_alive(agent, record_replays=False)
        self.assertEqual(len(runner.endgame_failure_buffer), 0)

    def test_ppo_multiboard_layout_and_runner(self) -> None:
        torch.manual_seed(17)
        np.random.seed(17)
        lay10 = build_multiboard_layout(10, 16)
        self.assertEqual(len(lay10), 16)
        self.assertEqual(lay10[0], 10)
        self.assertEqual(lay10[-1], 10)
        self.assertEqual(format_multiboard_summary(lay10, primary_board=10), "8×10 + 4×15 + 4×20")

        lay15 = build_multiboard_layout(15, 16)
        self.assertEqual(lay15[0], 15)
        self.assertEqual(lay15[-1], 15)
        self.assertEqual(format_multiboard_summary(lay15, primary_board=15), "8×15 + 5×10 + 3×20")

        lay20 = build_multiboard_layout(20, 16)
        self.assertEqual(lay20[0], 20)
        self.assertEqual(lay20[-1], 20)
        self.assertEqual(format_multiboard_summary(lay20, primary_board=20), "7×20 + 6×10 + 3×15")

        lay25 = build_multiboard_layout(25, 16)
        self.assertEqual(lay25[0], 25)
        self.assertEqual(lay25[-1], 25)
        self.assertEqual(format_multiboard_summary(lay25, primary_board=25), "6×25 + 6×10 + 2×15 + 2×20")

        # ParallelPPORunner runs mixed board sizes & optimizes cleanly with per-board failure buffers
        mb_slots = build_multiboard_layout(10, 8)
        agent = PPOAgent(device=torch.device("cpu"), num_envs=8, rollout_steps=6, minibatch_size=16, epochs=2)
        runner = ParallelPPORunner(8, 10, board_sizes=mb_slots, env_factory=SnakeEnv)
        self.assertTrue(runner.is_multiboard)
        self.assertEqual([int(env.board_size) for env in runner.envs], list(mb_slots))
        self.assertEqual(set(runner.failure_buffers_by_board.keys()), set(mb_slots))
        for _ in range(7):
            runner.step_alive(agent, record_replays=False)
        self.assertGreaterEqual(agent.total_updates, 1)
        self.assertTrue(np.isfinite(agent.last_loss))

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
