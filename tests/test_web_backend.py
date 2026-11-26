from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import web_api
import web_backend
from backend_config import GLOBAL_SIGNAL_LABELS, PPO_MAP_CHANNELS, PPO_MAP_SIZE, PPO_SIGNAL_LABELS
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent


def _reject_nonfinite_json(constant: str) -> None:
    raise ValueError(f"Non-standard JSON constant found: {constant}")


class TestWebBackendAndApi(unittest.TestCase):
    def test_preload_switching_ai_json_and_smart_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            orig_paths = (
                web_backend.RUNTIME_DIR,
                web_backend.AUX_STATS_PATH,
                web_backend.ACTIVE_CONTEXT_PATH,
                web_backend.PID_PATH,
                web_backend.PERF_LOG_PATH,
            )
            web_backend.RUNTIME_DIR = tmp
            web_backend.AUX_STATS_PATH = tmp / "snake_model_stats.json"
            web_backend.ACTIVE_CONTEXT_PATH = tmp / "active_context.json"
            web_backend.PID_PATH = tmp / ".web_backend.pid"
            web_backend.PERF_LOG_PATH = tmp / "backend_perf.log"

            try:
                dqn = DQNAgent(device=torch.device("cpu"))
                with torch.no_grad():
                    dqn.online_net.hidden1_layer.weight.fill_(0.1234)
                    dqn.target_net.hidden1_layer.weight.fill_(0.1234)
                dqn_ckpt = dqn.export_checkpoint()
                dqn_ckpt["arch_signature"] = web_backend.ARCH_SIGNATURE
                torch.save(dqn_ckpt, tmp / "snake_dqn.pt")

                ppo = PPOAgent(device=torch.device("cpu"))
                with torch.no_grad():
                    ppo.online_net.hidden1_layer.weight.fill_(-0.2345)
                ppo_ckpt = ppo.export_checkpoint()
                ppo_ckpt["arch_signature"] = web_backend.PPO_ARCH_SIGNATURE
                torch.save(ppo_ckpt, tmp / "snake_ppo.pt")

                (tmp / "active_context.json").write_text(
                    json.dumps({"agent_kind": "ppo", "mode": "manual", "board_size": 10}),
                    encoding="utf-8",
                )

                backend = web_backend.SnakeWebBackend(force_cpu=True)
                try:
                    ppo_w = float(backend.agent.online_net.hidden1_layer.weight[0, 0].detach())
                    self.assertAlmostEqual(ppo_w, -0.2345, places=4)
                    ppo_snap = backend.state_snapshot(include_network=True, include_details=True)
                    ego_map = ppo_snap["network"]["ego_map"]
                    self.assertIsNotNone(ego_map)
                    self.assertEqual(len(ego_map), PPO_MAP_CHANNELS)
                    self.assertEqual(len(ego_map[0]), PPO_MAP_SIZE)
                    self.assertEqual(ppo_snap["ppo_map_size"], PPO_MAP_SIZE)
                    self.assertEqual(ppo_snap["model_input_size"], len(PPO_SIGNAL_LABELS))
                    self.assertEqual(len(ppo_snap["global_signals"]["labels"]), len(PPO_SIGNAL_LABELS))

                    # Failure-replay episodes in PPO runner must NOT leak into UI stats/history
                    assert backend.training_runner is not None
                    saved_state = backend.training_runner.envs[0].export_state()
                    backend.training_runner.failure_replay_buffer.append(saved_state)
                    replay_slot = backend.training_runner.num_envs - 1
                    backend.training_runner.is_replay_env[replay_slot] = True
                    r_env = backend.training_runner.envs[replay_slot]
                    r_env.score = 77
                    r_env.snake = [(9, 0), (8, 0), (8, 1), (9, 1), (9, 2)]
                    r_env.direction = 1
                    r_env.food = (0, 9)
                    r_env._clear_observation_cache()
                    with backend.lock:
                        backend._perform_parallel_training_step()
                    self.assertGreaterEqual(backend.training_runner.failure_replays_completed, 1)
                    self.assertEqual(backend.episode_counter, 0)
                    self.assertEqual(backend.best_score, 0)
                    self.assertEqual(len(backend.history), 0)

                    # PPO showcase death populates last_death_snapshot with recent_steps & move_risks
                    # without inflating honest episode stats, and background completions do not overwrite it
                    with backend.lock:
                        backend.game.score = 12
                        backend.game.snake = [(9, 0), (8, 0), (8, 1), (9, 1), (9, 2)]
                        backend.game.direction = 1
                        backend.game.food = (0, 9)
                        backend.game._clear_observation_cache()
                        backend._perform_ppo_showcase_step()
                        self.assertEqual(backend.episode_counter, 0)
                        self.assertIsNotNone(backend.last_death_snapshot)
                        self.assertEqual(backend.last_death_snapshot["score"], 12)
                        self.assertGreaterEqual(len(backend.last_death_snapshot["recent_steps"]), 1)
                        self.assertIn("move_risks", backend.last_death_snapshot["recent_steps"][-1])
                        backend._record_training_completion(
                            {"score": 99, "step_count": 200, "reason": "wall_collision"},
                            update_death_snapshot=False,
                        )
                        self.assertEqual(backend.last_death_snapshot["score"], 12)
                        self.assertEqual(backend.episode_counter, 1)

                    # Changing board size clears PPO failure replay buffer and updates multi-board layout
                    self.assertTrue(backend.ppo_multiboard)
                    self.assertEqual(backend._ppo_multiboard_summary(), "8×10 + 4×15 + 4×20")
                    # Secondary-board completion must NOT leak into active board best_score or history
                    with backend.lock:
                        ep_before_mb = backend.episode_counter
                        best_before_mb = backend.best_score
                        backend._record_training_completion(
                            {"score": 350, "step_count": 1200, "reason": "wall_collision", "board_size": 20},
                            update_death_snapshot=False,
                        )
                        self.assertEqual(backend.episode_counter, ep_before_mb)
                        self.assertEqual(backend.best_score, best_before_mb)
                        self.assertEqual(backend.board_best_scores.get(20), 350)

                    res_board = backend.handle_command({"type": "set_board_size", "value": 15})
                    self.assertTrue(res_board["ok"])
                    self.assertEqual(len(backend.training_runner.failure_replay_buffer), 0)
                    self.assertEqual(len(backend.training_runner.endgame_failure_buffer), 0)
                    self.assertEqual(backend._ppo_multiboard_summary(), "8×15 + 5×10 + 3×20")

                    # Toggling Multi-Board off/on rebuilds runner and persists in active_context.json
                    res_mb_off = backend.handle_command({"type": "set_ppo_multiboard", "value": False})
                    self.assertTrue(res_mb_off["ok"])
                    self.assertFalse(backend.ppo_multiboard)
                    self.assertFalse(backend.training_runner.is_multiboard)
                    self.assertEqual(backend._ppo_multiboard_summary(), "")
                    ctx_saved = json.loads((tmp / "active_context.json").read_text(encoding="utf-8"))
                    self.assertFalse(ctx_saved["ppo_multiboard"])
                    res_mb_on = backend.handle_command({"type": "set_ppo_multiboard", "value": True})
                    self.assertTrue(res_mb_on["ok"])
                    self.assertTrue(backend.ppo_multiboard)
                    self.assertTrue(backend.training_runner.is_multiboard)
                    backend.handle_command({"type": "set_board_size", "value": 10})

                    # Greedy eval runs cleanly without polluting episode stats or history
                    ep_before = backend.episode_counter
                    res_eval = backend.handle_command({"type": "eval_greedy", "value": {"episodes": 3, "sync": True}})
                    self.assertTrue(res_eval["ok"])
                    eval_data = backend.last_eval_result
                    self.assertIsNotNone(eval_data)
                    self.assertEqual(eval_data["agent_kind"], "ppo")
                    self.assertEqual(eval_data["episodes"], 3)
                    self.assertEqual(eval_data["board_size"], 10)
                    self.assertTrue(eval_data["pure_policy"])
                    self.assertEqual(backend.episode_counter, ep_before)

                    # Agent-specific reward rules: PPO has +30 victory and hunger_critical
                    ppo_rules = backend.state_snapshot(include_network=False, include_details=True)["reward_rules"]
                    ppo_bonus_map = {b["id"]: b["value"] for b in ppo_rules["bonuses"]}
                    ppo_pen_ids = {p["id"] for p in ppo_rules["penalties"]}
                    self.assertEqual(ppo_bonus_map["victory"], "+30")
                    self.assertIn("pocket_merge", ppo_bonus_map)
                    self.assertIn("hunger_critical", ppo_pen_ids)

                    # Switching agent cancels/clears PPO eval result so it never leaks to DQN
                    res = backend.handle_command({"type": "set_agent_kind", "value": "dqn"})
                    self.assertTrue(res["ok"])
                    self.assertIsNone(backend.last_eval_result)
                    self.assertFalse(backend._eval_running)
                    dqn_w = float(backend.agent.online_net.hidden1_layer.weight[0, 0].detach())
                    self.assertAlmostEqual(dqn_w, 0.1234, places=4)
                    dqn_snap = backend.state_snapshot(include_network=True, include_details=True)
                    self.assertEqual(dqn_snap["model_input_size"], len(GLOBAL_SIGNAL_LABELS))
                    self.assertIsNone(dqn_snap["stats"]["last_eval_result"])
                    dqn_bonus_map = {b["id"]: b["value"] for b in dqn_snap["reward_rules"]["bonuses"]}
                    dqn_pen_ids = {p["id"] for p in dqn_snap["reward_rules"]["penalties"]}
                    self.assertEqual(dqn_bonus_map["victory"], "+10")
                    self.assertIn("pocket_merge", dqn_bonus_map)
                    self.assertNotIn("hunger_critical", dqn_pen_ids)

                    # Legacy manual mode in active_context.json coerces cleanly to ai
                    self.assertEqual(backend.mode, "ai")
                    res_unknown = backend.handle_command({"type": "manual_action", "value": 1})
                    self.assertFalse(res_unknown["ok"])
                    backend.handle_command({"type": "set_mode", "value": "ai"})
                    with backend.lock:
                        backend._perform_step(training_mode=False)
                        self.assertIn(backend.last_action, (0, 1, 2))

                    backend._save_state()
                    raw_json = (tmp / "snake_model_stats.json").read_text(encoding="utf-8")
                    json.loads(raw_json, parse_constant=_reject_nonfinite_json)

                    backend.mode = "smart"
                    res_smart = backend.handle_command({"type": "reset_agent"})
                    self.assertTrue(res_smart["ok"])
                    self.assertIn("state", res_smart)
                finally:
                    backend.shutdown(fast=True)
            finally:
                (
                    web_backend.RUNTIME_DIR,
                    web_backend.AUX_STATS_PATH,
                    web_backend.ACTIVE_CONTEXT_PATH,
                    web_backend.PID_PATH,
                    web_backend.PERF_LOG_PATH,
                ) = orig_paths

    def test_http_endpoints_and_utf8_debug_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            orig_paths = (
                web_backend.RUNTIME_DIR,
                web_backend.AUX_STATS_PATH,
                web_backend.ACTIVE_CONTEXT_PATH,
                web_backend.PID_PATH,
                web_backend.PERF_LOG_PATH,
                web_api.STATIC_ROOT,
            )
            web_backend.RUNTIME_DIR = tmp
            web_backend.AUX_STATS_PATH = tmp / "snake_model_stats.json"
            web_backend.ACTIVE_CONTEXT_PATH = tmp / "active_context.json"
            web_backend.PID_PATH = tmp / ".web_backend.pid"
            web_backend.PERF_LOG_PATH = tmp / "backend_perf.log"
            (tmp / ".runtime").mkdir(parents=True, exist_ok=True)
            log_text = '{"msg":"Привет, мир!"}\n'
            (tmp / ".runtime" / "debug_log.jsonl").write_text(log_text, encoding="utf-8")
            web_api.STATIC_ROOT = tmp

            backend = web_backend.SnakeWebBackend(force_cpu=True)
            server = web_api.SnakeHTTPServer(("127.0.0.1", 0), web_api.SnakeRequestHandler, backend)
            port = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(json.loads(resp.read().decode("utf-8")), {"ok": True})

                with urlopen(f"http://127.0.0.1:{port}/api/state?network=1&details=1", timeout=2) as resp:
                    state = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(state["model_input_size"], len(GLOBAL_SIGNAL_LABELS))
                    self.assertIsNotNone(state["network"])

                with urlopen(f"http://127.0.0.1:{port}/api/frame_stream", timeout=2) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
                    first_line = resp.readline().decode("utf-8")
                    self.assertTrue(first_line.startswith("data: "))
                    streamed = json.loads(first_line[6:])
                    self.assertIn("game", streamed)
                    self.assertIn("visual_speed", streamed)

                with urlopen(f"http://127.0.0.1:{port}/api/debug_log", timeout=2) as resp:
                    body = resp.read()
                    self.assertEqual(int(resp.headers["Content-Length"]), len(log_text.encode("utf-8")))
                    self.assertEqual(body.decode("utf-8"), log_text)
            finally:
                server.shutdown()
                server.server_close()
                backend.shutdown(fast=True)
                (
                    web_backend.RUNTIME_DIR,
                    web_backend.AUX_STATS_PATH,
                    web_backend.ACTIVE_CONTEXT_PATH,
                    web_backend.PID_PATH,
                    web_backend.PERF_LOG_PATH,
                    web_api.STATIC_ROOT,
                ) = orig_paths


if __name__ == "__main__":
    unittest.main()
