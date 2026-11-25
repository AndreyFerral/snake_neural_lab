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
    def test_preload_switching_manual_json_and_smart_reset(self) -> None:
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
                    json.dumps({"agent_kind": "ppo", "mode": "manual", "board_size": 10, "record_replays": False}),
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

                    # Changing board size clears PPO failure replay buffer
                    res_board = backend.handle_command({"type": "set_board_size", "value": 15})
                    self.assertTrue(res_board["ok"])
                    self.assertEqual(len(backend.training_runner.failure_replay_buffer), 0)
                    backend.handle_command({"type": "set_board_size", "value": 10})

                    res = backend.handle_command({"type": "set_agent_kind", "value": "dqn"})
                    self.assertTrue(res["ok"])
                    dqn_w = float(backend.agent.online_net.hidden1_layer.weight[0, 0].detach())
                    self.assertAlmostEqual(dqn_w, 0.1234, places=4)
                    dqn_snap = backend.state_snapshot(include_network=True, include_details=True)
                    self.assertEqual(dqn_snap["model_input_size"], len(GLOBAL_SIGNAL_LABELS))

                    backend.handle_command({"type": "set_mode", "value": "manual"})
                    with backend.lock:
                        cur_dir = backend.game.direction
                        backend.pending_manual_direction = (cur_dir - 1) % 4
                        backend._perform_step(training_mode=False)
                        self.assertEqual(backend.last_action, 1)

                    backend._save_state()
                    raw_json = (tmp / "snake_model_stats.json").read_text(encoding="utf-8")
                    json.loads(raw_json, parse_constant=_reject_nonfinite_json)

                    replays_dir = tmp / "replays"
                    replays_dir.mkdir(exist_ok=True)
                    dummy_replay = replays_dir / "replay_score_10_ep_1_live_1.json.gz"
                    dummy_replay.write_bytes(b"test")
                    backend.mode = "smart"
                    res_smart = backend.handle_command({"type": "reset_agent"})
                    self.assertTrue(res_smart["ok"])
                    self.assertIn("state", res_smart)
                    self.assertTrue(dummy_replay.exists())
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
