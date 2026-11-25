from __future__ import annotations

import os
# Настройки окружения и потоков CPU (баланс скорости PPO-батчей и отзывчивости UI/SSE)
_CPU_TORCH_THREADS = max(1, min(8, (os.cpu_count() or 4) // 2 + 2))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = str(_CPU_TORCH_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(_CPU_TORCH_THREADS)
os.environ['MKL_NUM_THREADS'] = str(_CPU_TORCH_THREADS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(_CPU_TORCH_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(_CPU_TORCH_THREADS)

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Deque, Iterator
import numpy as np
import torch
torch.set_num_threads(_CPU_TORCH_THREADS)
torch.set_num_interop_threads(1)

try:
    import orjson
    def _encode_frame_json(payload: dict[str, Any]) -> bytes:
        return orjson.dumps(payload)
except ImportError:
    def _encode_frame_json(payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _win_timer_begin() -> bool:
    if os.name != "nt":
        return False
    try:
        return ctypes.windll.winmm.timeBeginPeriod(1) == 0
    except Exception:
        return False


def _win_timer_end(enabled: bool) -> None:
    if not enabled or os.name != "nt":
        return
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass

# Импорты только тех констант, которые реально используются
from backend_config import (
    AGENT_KINDS,
    BOARD_SIZES,
    DEATH_REASON_ORDER,
    DECISION_STATS_WINDOW,
    DIRECTIONS,
    ENDGAME_IDX,
    ENDGAME_MERGE_BONUS,
    ENDGAME_SPLIT_PENALTY,
    GLOBAL_SIGNAL_LABELS,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    HISTORY_LIMIT,
    MODEL_ACTION_LABELS,
    NETWORK_INPUT_ADAPTER_MODE,
    POCKET_MERGE_START_IDX,
    POCKET_SPLIT_START_IDX,
    PPO_CONV1_CHANNELS,
    PPO_CONV2_CHANNELS,
    PPO_HIDDEN_LAYER_1,
    PPO_HIDDEN_LAYER_2,
    PPO_MAP_CHANNELS,
    PPO_MAP_FEAT_SIZE,
    PPO_MAP_SIZE,
    PPO_SIGNAL_LABELS,
    SAVE_THROTTLE_SECONDS,
    SLOW_API_MS,
    SLOW_NETWORK_MS,
    SLOW_STEP_MS,
    STARTUP_BOARD_SIZE,
    REASON_LABELS,
    SIGNAL_LABELS,
    TERMINAL_REASONS,
    hunger_budget_for,
)

from dqn_agent import (
    DQNAgent,
    apply_step_risk_shaping,
    compute_dqn_priority_scale,
    evaluate_pre_step_risks,
    filter_dqn_random_action,
)
from ppo_agent import PPOAgent, ParallelPPORunner
from runtime_storage import (
    ReplayManager,
    checkpoint_path_for_agent as _checkpoint_path_for_agent,
    clear_aux_stats_store as _clear_aux_stats_store,
    compact_model_stats,
    load_aux_stats_store as _load_aux_stats_store,
    log_perf as _log_perf,
    model_stats_key,
    prune_model_stats_store,
    reset_session_perf_log as _reset_session_perf_log,
    save_aux_stats_store as _save_aux_stats_store,
    write_json_atomic,
)
from snake_env import SnakeEnv, relative_to_absolute
from smart_agent import SmartH5Agent, smart_model_candidates
from telemetry import (
    CHOICE_TO_SYMBOL,
    DEATH_TO_SYMBOL,
    STEP_TO_SYMBOL,
    SYMBOL_TO_CHOICE,
    SYMBOL_TO_DEATH,
    SYMBOL_TO_STEP,
    AgentSlot,
    EpisodeStats,
    average_score_window,
    build_connection_payload,
    build_reward_rules_payload,
    compact_death_snapshot,
    export_decision_stats,
    make_decision_stat_bucket,
    make_decision_stats,
    record_decision_stat,
)


def safe_hunger_budget_for(score: int, board_size: int) -> float:
    return float(hunger_budget_for(score, board_size))

# =========================================================================
# ЖЁСТКАЯ ФИКСАЦИЯ: ТОЛЬКО СИГНАЛЫ + АВТОПОДПИСЬ АРХИТЕКТУРЫ
# =========================================================================
GLOBAL_SIGNAL_COUNT = len(GLOBAL_SIGNAL_LABELS)
PPO_SIGNAL_COUNT = len(PPO_SIGNAL_LABELS)
# Подпись меняется автоматически при изменении количества сигналов или размеров слоёв
ARCH_SIGNATURE = f"sig_{GLOBAL_SIGNAL_COUNT}_{HIDDEN_LAYER_1}_{HIDDEN_LAYER_2}"
PPO_ARCH_SIGNATURE = (
    f"ppo_split_p1_c{PPO_CONV1_CHANNELS}_{PPO_CONV2_CHANNELS}_f{PPO_MAP_FEAT_SIZE}_"
    f"map{PPO_MAP_CHANNELS}x{PPO_MAP_SIZE}_sig{PPO_SIGNAL_COUNT}_{PPO_HIDDEN_LAYER_1}_{PPO_HIDDEN_LAYER_2}"
)


def arch_signature_for_agent(agent_kind: str) -> str:
    return PPO_ARCH_SIGNATURE if agent_kind == "ppo" else ARCH_SIGNATURE
# =========================================================================

STATIC_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = STATIC_ROOT / ".runtime"
AUX_STATS_PATH = RUNTIME_DIR / "snake_model_stats.json"
ACTIVE_CONTEXT_PATH = RUNTIME_DIR / "active_context.json"
PID_PATH = RUNTIME_DIR / ".web_backend.pid"
PERF_LOG_PATH = RUNTIME_DIR / "backend_perf.log"


def reset_session_perf_log() -> None:
    _reset_session_perf_log(RUNTIME_DIR, PERF_LOG_PATH)


def log_perf(event: str, elapsed_ms: float, **fields: Any) -> None:
    _log_perf(RUNTIME_DIR, PERF_LOG_PATH, event, elapsed_ms, **fields)


def list_available_devices() -> list[dict[str, str]]:
    return [{"id": "cpu", "label": "CPU"}]


def checkpoint_path_for_agent(agent_kind: str) -> Path:
    return _checkpoint_path_for_agent(RUNTIME_DIR, agent_kind)


def load_aux_stats_store() -> dict[str, Any]:
    return _load_aux_stats_store(AUX_STATS_PATH)


def save_aux_stats_store(store: dict[str, Any]) -> None:
    _save_aux_stats_store(store, AUX_STATS_PATH)


def clear_aux_stats_store() -> None:
    _clear_aux_stats_store(AUX_STATS_PATH)


def resolve_device(mode: str) -> tuple[torch.device, str, str]:
    return torch.device("cpu"), "CPU", "CPU"


class SnakeWebBackend:
    def __init__(self, force_cpu: bool = False) -> None:
        preferred_mode = "cpu"
        self.device_mode = preferred_mode
        self.device, self.device_kind, self.device_details = resolve_device(preferred_mode)
        self.device_label = self.device_kind
        self.network_source = "Python backend"
        self.available_devices = list_available_devices()
        
        self.lock = threading.RLock()
        self.agent_lock = threading.RLock()
        self.agent_slot_lock = threading.RLock()
        self.ppo_worker_lock = threading.Lock()
        self._ppo_paused = False
        self._ppo_wake_event = threading.Event()
        self._frame_cond = threading.Condition()
        self._frame_seq = 0
        self._frame_json_bytes: bytes = b"{}"
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snake-save")
        self._save_in_flight = False
        self._win_timer_enabled = _win_timer_begin()
        self.agent_slots: dict[str, AgentSlot] = {}
        self.stop_event = threading.Event()
        
        self._pending_save = False
        self._next_save_at = 0.0
        self._frame_cache: dict[str, Any] | None = None
        self._last_frame_payload: dict[str, Any] | None = None
        self._snapshot_cache: dict[tuple[bool, bool], dict[str, Any]] = {}
        self._network_payload_cache: dict[str, Any] | None = None
        self.frame_id = 0
        self._smart_models_cache: list[dict[str, Any]] = []
        self._smart_models_cache_at = 0.0
        self._smart_models_cache_key: tuple[tuple[str, int, int], ...] = ()
        self.restart_pending = False

        self.game = SnakeEnv(board_size=STARTUP_BOARD_SIZE)
        self.agent_kind = "dqn"
        self.agent: DQNAgent | PPOAgent = DQNAgent(device=self.device)
        self.training_runner: ParallelPPORunner | None = None
        self.smart_agent = SmartH5Agent(RUNTIME_DIR, STATIC_ROOT)
        self.running = True
        self.training = False
        self.mode = "ai"
        self.speed = 8
        self.live_reward = 0.0
        self.episode_reward = 0.0
        self.fitness_ema = 0.0
        self.best_score = 0
        self.board_best_scores: dict[int, int] = {self.game.board_size: 0}
        self.episode_counter = 0
        self.last_action = 0
        self.last_action_random = False
        self.last_greedy_action = 0
        self.last_reason = "new_game"
        self.last_outcome_reason = "new_game"
        self.last_death_snapshot: dict[str, Any] | None = None
        self.last_state = self._current_observation()
        self.last_q_values = np.zeros(len(MODEL_ACTION_LABELS), dtype=np.float32)
        self.history: list[EpisodeStats] = []
        self.persistence_enabled = True
        self.decision_stats = make_decision_stats()
        self.recent_step_snapshots: Deque[dict[str, Any]] = deque(maxlen=18)
        self.training_display_index = 0
        self.current_replay_steps: list[dict[str, Any]] = []
        self.replays_dir = RUNTIME_DIR / "replays"
        self.replay_manager = ReplayManager(self.replays_dir, perf_logger=log_perf)
        self.record_replays = True
        self.last_eval_result: dict[str, Any] | None = None
        self._eval_running = False
        self._eval_generation = 0
        self._eval_cancel_event = threading.Event()

        self._sync_network_source()
        self._load_active_context()
        self._load_state(force_cpu=force_cpu)
        self._store_active_agent_slot(loaded_checkpoint=True)
        
        startup_board_size = int(self.game.board_size)
        self._reset_training_runner(startup_board_size)
        self.game = SnakeEnv(board_size=startup_board_size)
        self._select_board_best_score(startup_board_size)
        self._reset_environment_state()
        self.running = False
        self._update_frame_cache()

        self._preload_native_agent_slots(blocking=True)
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.ppo_thread = threading.Thread(target=self._ppo_training_loop, daemon=True)
        self.thread.start()
        self.ppo_thread.start()

    @contextmanager
    def _pause_ppo_worker(self) -> Iterator[None]:
        self._ppo_paused = True
        self.ppo_worker_lock.acquire()
        try:
            yield
        finally:
            self._ppo_paused = False
            self.ppo_worker_lock.release()
            if self.running and self.training and self.agent_kind == "ppo":
                self._ppo_wake_event.set()

    def shutdown(self, *, fast: bool = False) -> None:
        self.stop_event.set()
        self._cancel_greedy_eval(clear_result=False)
        self._ppo_paused = True
        self._ppo_wake_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        self.ppo_thread.join(timeout=0.5 if fast else 2.0)
        self.thread.join(timeout=0.4 if fast else 2.0)
        self._save_executor.shutdown(wait=not fast)
        payloads = None
        acquired = self.lock.acquire(timeout=0.5 if fast else -1)
        if acquired:
            try:
                self._pending_save = False
                payloads = self._capture_save_payloads()
            finally:
                self.lock.release()
        if payloads is not None:
            self._write_save_payloads(*payloads)
        self.replay_manager.shutdown(wait=not fast)
        _win_timer_end(self._win_timer_enabled)
        self._win_timer_enabled = False

    def _current_observation(self) -> np.ndarray:
        if self.agent_kind == "ppo":
            return self.game.get_ppo_signals()
        # Для DQN всегда возвращаем базовый вектор из 34 сигналов
        return self.game.get_observation_grid()

    def _checkpoint_path(self) -> Path:
        return checkpoint_path_for_agent(self.agent_kind)

    def _expected_state_size(self) -> int:
        return PPO_SIGNAL_COUNT if self.agent_kind == "ppo" else GLOBAL_SIGNAL_COUNT

    def _current_signal_labels(self) -> list[str]:
        return list(PPO_SIGNAL_LABELS) if self.agent_kind == "ppo" else list(GLOBAL_SIGNAL_LABELS)

    def _sync_board_best_score(self) -> None:
        self.board_best_scores[int(self.game.board_size)] = max(
            int(self.board_best_scores.get(int(self.game.board_size), 0)),
            int(self.best_score),
        )

    def _select_board_best_score(self, board_size: int | None = None) -> None:
        target_size = int(board_size if board_size is not None else self.game.board_size)
        self.best_score = int(self.board_best_scores.get(target_size, 0))

    def _ensure_runtime_consistency(self) -> None:
        expected_state_size = self._expected_state_size()
        network_input_size = int(getattr(self.agent.online_net, "input_size", -1))
        
        # Проверяем только размер входа сети
        if network_input_size != expected_state_size:
            self.agent = self._create_native_agent(self.agent_kind)
            self._store_active_agent_slot()
            self._reset_training_runner()
            self.fitness_ema = 0.0
            
        if int(getattr(self.last_state, "size", 0)) != expected_state_size:
            self.last_state = self._current_observation()

    def _agent_cache_key(self, agent_kind: str) -> str:
        return agent_kind

    def _create_native_agent(self, agent_kind: str) -> DQNAgent | PPOAgent:
        if agent_kind == "ppo":
            return PPOAgent(device=self.device)
        return DQNAgent(device=self.device)

    def _store_active_agent_slot(self, *, loaded_checkpoint: bool = False) -> None:
        key = self._agent_cache_key(self.agent_kind)
        with self.agent_slot_lock:
            self.agent_slots[key] = AgentSlot(
                agent=self.agent,
                ready=True,
                loading=False,
                error="",
                loaded_checkpoint=loaded_checkpoint,
            )

    def _load_agent_checkpoint_into(self, agent: DQNAgent | PPOAgent, agent_kind: str | None = None) -> bool:
        target_kind = agent_kind or self.agent_kind
        checkpoint_path = checkpoint_path_for_agent(target_kind)
        if not checkpoint_path.exists():
            return False
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            return False
        if payload.get("arch_signature") != arch_signature_for_agent(target_kind):
            return False
        agent.import_checkpoint(payload)
        return True

    def _preload_native_agent_slots(self, *, blocking: bool = False) -> None:
        for agent_kind in AGENT_KINDS:
            key = self._agent_cache_key(agent_kind)
            with self.agent_slot_lock:
                slot = self.agent_slots.get(key)
                if slot and (slot.ready or slot.loading):
                    continue
                self.agent_slots[key] = AgentSlot(loading=True)

            def preload(kind: str = agent_kind) -> None:
                loaded = False
                error = ""
                agent_obj: DQNAgent | PPOAgent | None = None
                try:
                    agent_obj = self._create_native_agent(kind)
                    loaded = self._load_agent_checkpoint_into(agent_obj, kind)
                except Exception as exc:
                    error = str(exc)
                with self.agent_slot_lock:
                    self.agent_slots[kind] = AgentSlot(
                        agent=agent_obj,
                        ready=agent_obj is not None,
                        loading=False,
                        error=error,
                        loaded_checkpoint=loaded,
                    )

            if blocking:
                preload()
            else:
                threading.Thread(target=preload, daemon=True).start()

    def _agent_slots_payload(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        with self.agent_slot_lock:
            for agent_kind in AGENT_KINDS:
                slot = self.agent_slots.get(self._agent_cache_key(agent_kind))
                payload[agent_kind] = {
                    "ready": bool(slot and slot.ready),
                    "loading": bool(slot and slot.loading),
                    "error": "" if slot is None else slot.error,
                    "loaded_checkpoint": bool(slot and slot.loaded_checkpoint),
                }
        return payload

    def _sync_network_source(self) -> None:
        if self.mode == "smart":
            self.network_source = self.smart_agent.label
        elif self.agent_kind == "ppo":
            self.network_source = f"PPO ({PPO_MAP_SIZE}x{PPO_MAP_SIZE} Map)"
        else:
            self.network_source = "DQN policy"

    def _active_aux_stats_key(self) -> str:
        if self.mode == "smart":
            model_name = self.smart_agent.model_path.stem if self.smart_agent.model_path else "none"
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_") or "none"
            return f"snake_smart_{safe_name}_{self.game.board_size}"
        return self._checkpoint_path().stem

    def _load_active_aux_stats(self) -> bool:
        stats_store = load_aux_stats_store()
        stats_models = stats_store.get("models", {}) if isinstance(stats_store, dict) else {}
        if not isinstance(stats_models, dict):
            return False
        stats_payload = stats_models.get(self._active_aux_stats_key())
        if isinstance(stats_payload, dict):
            self._load_aux_stats_payload(stats_payload)
            return True
        return False

    def _reset_training_runner(self, board_size: int | None = None) -> None:
        target_size = int(board_size if board_size is not None else self.game.board_size)
        if self.agent_kind == "ppo":
            num_envs = int(getattr(self.agent, "num_envs", 8))
            self.training_runner = ParallelPPORunner(
                num_envs,
                target_size,
                env_factory=SnakeEnv,
                action_labels=MODEL_ACTION_LABELS,
            )
        else:
            self.training_runner = None
        self.training_display_index = 0

    def _smart_models_payload(self) -> list[dict[str, Any]]:
        active_path = self.smart_agent.model_path.resolve() if self.smart_agent.model_path else None
        now = time.monotonic()
        
        # Гарантируем существование папки перед сканированием
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            candidates = smart_model_candidates(RUNTIME_DIR)
            cache_key = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in candidates)
        except OSError:
            candidates = []
            cache_key = ()
            
        # Возвращаем кэш только если директория не менялась и кэш валиден
        if self._smart_models_cache and now - self._smart_models_cache_at < 3.0 and cache_key == self._smart_models_cache_key:
            active_name = active_path.name if active_path else ""
            return [
                {**m, "active": bool(active_name and m.get("name") == active_name)}
                for m in self._smart_models_cache
            ]

        models: list[dict[str, Any]] = []
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
                
            epoch_match = re.search(r"epoch[_-]?(\d+)", path.stem, re.IGNORECASE)
            epoch = int(epoch_match.group(1)) if epoch_match else None
            
            models.append({
                "name": path.name,
                "path": str(path),
                "active": bool(active_path and resolved == active_path),
                "size": path.stat().st_size if path.exists() else 0,
                "epoch": epoch,
                "label": f"epoch {epoch}" if epoch is not None else path.stem,
            })
            
        self._smart_models_cache = models
        self._smart_models_cache_at = now
        self._smart_models_cache_key = cache_key
        return models

    def _set_smart_model(self, model_name: str) -> tuple[bool, str]:
        candidate = (RUNTIME_DIR / Path(model_name).name).resolve()
        runtime_root = RUNTIME_DIR.resolve()
        if candidate.parent != runtime_root or candidate.suffix.lower() != ".h5" or not candidate.exists():
            return False, "Smart model file not found"

        previous_model_path = self.smart_agent.model_path
        self.smart_agent.set_model_path(candidate)
        if not self.smart_agent.loaded:
            error = self.smart_agent.error or "Zmey-smart model is unavailable"
            if previous_model_path is not None and previous_model_path != candidate:
                self.smart_agent.set_model_path(previous_model_path)
            self._sync_network_source()
            return False, error

        self._cancel_greedy_eval(clear_result=True)

        self._sync_network_source()
        self._smart_models_cache_at = 0.0
        self.mode = "smart"
        self.training = False
        self.running = False
        self.game = SnakeEnv(board_size=self.game.board_size)
        if not self._load_active_aux_stats():
            self._clear_runtime_review_state()
        self._reset_environment_state()
        self.last_state = self._current_observation()
        _, self.last_q_values, _, self.last_greedy_action = self._select_current_action(self.last_state, False)
        return True, "ok"

    def _select_current_action(self, state: np.ndarray, training_mode: bool) -> tuple[int, np.ndarray, bool, int]:
        if not training_mode:
            self._ensure_runtime_consistency()
        if self.mode == "smart" and not training_mode:
            return self.smart_agent.select_action(self.game)
        ego_map = self.game.get_egocentric_map() if self.agent_kind == "ppo" else None
        with self.agent_lock:
            if self.agent_kind == "ppo":
                if self.mode == "ai" and not training_mode:
                    return self.agent.select_best_action(state, ego_map=ego_map)
                return self.agent.select_action(
                    state,
                    ego_map=ego_map,
                    training=training_mode,
                    snake_length=len(self.game.snake),
                )
            if self.mode == "ai" and not training_mode:
                return self.agent.select_best_action(state)
            return self.agent.select_action(
                state,
                training=training_mode,
                snake_length=len(self.game.snake),
            )

    def _visual_speed(self) -> int:
        return max(1, int(self.speed))

    def _training_burst(self) -> int:
        speed = max(1, int(self.speed))
        return max(1, min(16, speed))

    def _ppo_training_loop(self) -> None:
        while not self.stop_event.is_set():
            if not (self.running and self.training and self.agent_kind == "ppo" and not self._ppo_paused):
                self._ppo_wake_event.wait(timeout=0.05)
                self._ppo_wake_event.clear()
                continue

            step_started = time.perf_counter()
            completed: list[dict[str, Any]] = []
            agent_ref: DQNAgent | PPOAgent | None = None
            runner_ref: ParallelPPORunner | None = None
            prev_updates = 0
            record_replays = False

            with self.ppo_worker_lock:
                if (
                    self.stop_event.is_set()
                    or self._ppo_paused
                    or not (self.running and self.training and self.agent_kind == "ppo")
                ):
                    continue
                if self.training_runner is None:
                    with self.lock:
                        if self.training_runner is None and self.agent_kind == "ppo":
                            self._reset_training_runner()
                runner_ref = self.training_runner
                agent_ref = self.agent
                record_replays = bool(self.record_replays)
                if runner_ref is None or not isinstance(agent_ref, PPOAgent):
                    continue
                prev_updates = int(agent_ref.total_updates)
                completed = runner_ref.step_alive(agent_ref, record_replays=record_replays)

            did_update = isinstance(agent_ref, PPOAgent) and int(agent_ref.total_updates) > prev_updates
            if completed or did_update:
                with self.lock:
                    if self.agent is agent_ref and self.training_runner is runner_ref:
                        new_best_policy = False
                        for item in completed:
                            self._record_training_completion(item, update_death_snapshot=False)
                            if isinstance(agent_ref, PPOAgent):
                                if agent_ref.record_full_episode(
                                    int(item.get("score", 0)),
                                    str(item.get("reason", "")),
                                    board_size=int(item.get("board_size", self.game.board_size)),
                                ):
                                    new_best_policy = True
                            replay_steps = item.get("replay_steps")
                            if record_replays and isinstance(replay_steps, list) and replay_steps:
                                self._save_replay(
                                    replay_steps,
                                    int(item.get("score", 0)),
                                    str(item.get("reason", "unknown")),
                                    prefix=f"ppo_u{int(agent_ref.total_updates)}",
                                )
                        has_victory = any(item.get("reason") == "victory" for item in completed)
                        if new_best_policy or did_update or has_victory:
                            self._schedule_save(urgent=has_victory)

            speed = int(self.speed)
            if speed < 32:
                target_dt = 1.0 / max(4.0, float(speed) * 2.5)
                remaining = target_dt - (time.perf_counter() - step_started)
                if remaining > 0.001:
                    time.sleep(min(0.05, remaining))
            else:
                time.sleep(0)

    def _perform_ppo_showcase_step(self) -> None:
        """Плавный неблокирующий шаг показательной змейки на поле во время PPO-обучения."""
        if not isinstance(self.agent, PPOAgent):
            return
        state = self.game.get_ppo_signals()
        ego_map = self.game.get_egocentric_map()
        act, probs, _, greedy = self.agent.select_showcase_action(state, ego_map=ego_map)

        pre_step_snapshot = self._build_recent_step_snapshot(
            absolute_direction=relative_to_absolute(self.game.direction, act),
            action=act,
            action_random=False,
            greedy_action=greedy,
        )
        pre_step_risks = evaluate_pre_step_risks(self.game, state, act)
        is_endgame = float(state[ENDGAME_IDX]) > 0.5
        is_pocket_split = float(state[POCKET_SPLIT_START_IDX + act]) > 0.5
        is_pocket_merge = float(state[POCKET_MERGE_START_IDX + act]) > 0.5

        abs_dir = relative_to_absolute(self.game.direction, act)
        _, reward, done, info = self.game.step(abs_dir, return_obs=False)
        reason = str(info["reason"])
        shaped_reward, extra_penalties = apply_step_risk_shaping(
            reward,
            reason,
            len(self.game.snake),
            pre_step_risks,
            is_endgame=is_endgame,
            is_pocket_split=is_pocket_split,
            is_pocket_merge=is_pocket_merge,
            endgame_split_penalty=ENDGAME_SPLIT_PENALTY,
            endgame_merge_bonus=ENDGAME_MERGE_BONUS,
        )
        all_penalties = [str(item) for item in info.get("penalty_reasons", [])] + extra_penalties
        step_num = int(info.get("steps", self.game.step_count))
        if step_num == 1:
            self.recent_step_snapshots.clear()
        pre_step_snapshot["outcome_reason"] = reason
        pre_step_snapshot["penalty_reasons"] = list(all_penalties)
        pre_step_snapshot["steps_without_food"] = int(self.game.steps_without_food)
        pre_step_snapshot["max_steps_without_food"] = int(self.game.max_steps_without_food)
        self.recent_step_snapshots.append(pre_step_snapshot)

        self.last_q_values = probs
        self.last_action = act
        self.last_action_random = False
        self.last_greedy_action = greedy
        self.last_reason = reason
        self.live_reward = float(shaped_reward)
        self.last_penalty_reasons = all_penalties
        self._record_decision_stat("steps", reason)
        self._record_decision_stat("choices", MODEL_ACTION_LABELS[act])

        if done:
            recent_steps = self._recent_unique_death_steps(limit=8)
            self.last_death_snapshot = {
                "reason": reason,
                "size": self.game.board_size,
                "score": self.game.score,
                "step_count": self.game.step_count,
                "steps_without_food": self.game.steps_without_food,
                "direction": self.game.direction,
                "snake": [{"x": x, "y": y} for x, y in self.game.snake],
                "food": {"x": self.game.food[0], "y": self.game.food[1]},
                "eaten_food_trail": [{"x": x, "y": y} for x, y in self.game.eaten_food_trail],
                "recent_steps": recent_steps,
            }
            self.last_outcome_reason = reason
            self.game.reset(return_obs=False)
            self.recent_step_snapshots.clear()
        self.last_state = self._current_observation()

    def _simulation_loop(self) -> None:
        last_tick = time.perf_counter()
        accumulator = 0.0
        last_idle_publish = last_tick

        while not self.stop_event.is_set():
            now = time.perf_counter()
            delta = min(now - last_tick, 0.25)
            last_tick = now
            did_work = False
            dqn_training_steps = 0
            sleep_hint = 0.008

            with self.lock:
                if self.running:
                    if self.training and self.agent_kind == "ppo":
                        self._ppo_wake_event.set()
                        step_interval = 1.0 / max(1, self._visual_speed())
                        accumulator = min(accumulator + delta, step_interval * 2.0)
                        if accumulator >= step_interval:
                            accumulator -= step_interval
                            self._perform_ppo_showcase_step()
                            self._update_frame_cache()
                            did_work = True
                        sleep_hint = max(0.001, min(0.004, (step_interval - accumulator) * 0.5))
                    elif self.training:
                        accumulator = min(accumulator + delta, 0.12)
                        if accumulator >= (1.0 / 60.0):
                            accumulator -= 1.0 / 60.0
                            dqn_training_steps = self._training_burst()
                            did_work = True
                        sleep_hint = 0.001
                    else:
                        step_interval = 1.0 / max(1, self.speed)
                        accumulator = min(accumulator + delta, step_interval * 2.0)
                        if accumulator >= step_interval:
                            accumulator -= step_interval
                            self._perform_step(training_mode=False)
                            self._update_frame_cache()
                            did_work = True
                        sleep_hint = max(0.001, min(0.004, (step_interval - accumulator) * 0.5))
                else:
                    accumulator = 0.0
                    sleep_hint = 0.01

            if dqn_training_steps > 0:
                for _ in range(dqn_training_steps):
                    with self.lock:
                        if not self.running or not self.training or self.agent_kind != "dqn":
                            break
                        self._perform_step(training_mode=True)
                with self.lock:
                    self._update_frame_cache()
            elif not did_work and (now - last_idle_publish) >= 0.5:
                with self.lock:
                    if not self.running:
                        last_idle_publish = now
                    elif self.training and self.agent_kind != "ppo":
                        self._update_frame_cache()
                        last_idle_publish = now
                last_idle_publish = now

            self._flush_pending_save()
            time.sleep(0.001 if did_work and dqn_training_steps > 0 else sleep_hint)

    def _record_training_completion(self, payload: dict[str, Any], *, update_death_snapshot: bool = True) -> None:
        score = int(payload.get("score", 0))
        step_count = int(payload.get("step_count", 0))
        reason = str(payload.get("reason", "unknown"))

        self.episode_counter += 1
        self.best_score = max(self.best_score, score)
        self._sync_board_best_score()
        if reason in DEATH_REASON_ORDER:
            self._record_decision_stat("deaths", reason)

        loss_val = float(payload.get("loss", getattr(self.agent, "last_loss", 0.0)))
        self.history.append(EpisodeStats(
            episode=self.episode_counter,
            score=score,
            reward=float(payload.get("reward", 0.0)),
            mutation=0.0,
            fitness=-loss_val,
            steps=step_count,
            best_score=self.best_score,
            reason=reason,
            loss=loss_val,
        ))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)

        if update_death_snapshot:
            self.last_death_snapshot = self._snapshot_for_completed_training(payload)
            self.last_outcome_reason = reason
            self.last_penalty_reasons = [str(item) for item in payload.get("penalty_reasons", [])]
        self.fitness_ema = -loss_val if self.fitness_ema == 0 else self.fitness_ema * 0.9 + (-loss_val) * 0.1

    def _snapshot_for_completed_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "reason": str(payload.get("reason", "unknown")),
            "size": int(payload.get("board_size", self.game.board_size)),
            "score": int(payload.get("score", 0)),
            "step_count": int(payload.get("step_count", 0)),
            "steps_without_food": int(payload.get("steps_without_food", 0)),
            "direction": int(payload.get("direction", 1)),
            "snake": [{"x": x, "y": y} for x, y in payload.get("snake", [])],
            "food": {"x": payload["food"][0], "y": payload["food"][1]} if payload.get("food") else {"x": 0, "y": 0},
            "eaten_food_trail": [{"x": x, "y": y} for x, y in payload.get("eaten_food_trail", [])],
            "recent_steps": [],
            "individual": int(payload.get("index", -1)) + 1,
        }

    def _perform_parallel_training_step(self) -> None:
        started_at = time.perf_counter()
        self._ensure_runtime_consistency()
        if self.training_runner is None:
            self._reset_training_runner()
        assert self.training_runner is not None

        prev_updates = int(getattr(self.agent, "total_updates", 0))
        with self.agent_lock:
            completed = self.training_runner.step_alive(self.agent, record_replays=self.record_replays)
        new_best_policy = False
        for item in completed:
            self._record_training_completion(item)
            if isinstance(self.agent, PPOAgent):
                with self.agent_lock:
                    if self.agent.record_full_episode(
                        int(item.get("score", 0)),
                        str(item.get("reason", "")),
                        board_size=int(item.get("board_size", self.game.board_size)),
                    ):
                        new_best_policy = True
            replay_steps = item.get("replay_steps")
            if self.record_replays and isinstance(replay_steps, list) and replay_steps:
                self._save_replay(
                    replay_steps,
                    int(item.get("score", 0)),
                    str(item.get("reason", "unknown")),
                    prefix=f"ppo_u{int(getattr(self.agent, 'total_updates', 0))}",
                )

        if (
            new_best_policy
            or int(getattr(self.agent, "total_updates", 0)) > prev_updates
            or any(item.get("reason") == "victory" for item in completed)
        ):
            self._schedule_save(urgent=True)

        self.training_display_index = self.training_runner.update_leader()
        self.game = self.training_runner.leader_env()
        self.last_state = self._current_observation()
        if (
            self.training_runner.last_probs is not None
            and self.training_runner.last_greedy_actions is not None
            and 0 <= self.training_display_index < len(self.training_runner.last_probs)
        ):
            self.last_q_values = self.training_runner.last_probs[self.training_display_index]
            self.last_greedy_action = int(self.training_runner.last_greedy_actions[self.training_display_index])
        else:
            ego_map = self.game.get_egocentric_map()
            with self.agent_lock:
                _, self.last_q_values, _, self.last_greedy_action = self.agent.select_best_action(
                    self.last_state,
                    ego_map=ego_map,
                )
        self.last_action = self.last_greedy_action
        self.last_action_random = False
        self.last_reason = self.game.last_reason
        self.live_reward = self.game.last_reward
        self._record_decision_stat("steps", self.last_reason)
        self._record_decision_stat("choices", MODEL_ACTION_LABELS[self.last_action])

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if elapsed_ms >= SLOW_STEP_MS:
            log_perf(
                "slow_ppo_training_step",
                elapsed_ms,
                completed=len(completed),
                updates=int(getattr(self.agent, "total_updates", 0)),
                leader=self.training_display_index + 1,
                board=self.game.board_size,
                snake=len(self.game.snake),
                score=self.game.score,
            )

        time.sleep(0)

    def _perform_step(self, training_mode: bool) -> None:
        started_at = time.perf_counter()
        done = False
        reason = ""
        self._ensure_runtime_consistency()
        state = self._current_observation()
        predicted_action, q_values, was_random, greedy_action = self._select_current_action(state, training_mode)
        if training_mode and self.agent_kind == "dqn" and was_random:
            predicted_action, was_random = filter_dqn_random_action(
                state,
                predicted_action,
                greedy_action,
                was_random,
            )
        absolute_direction = relative_to_absolute(self.game.direction, predicted_action)
        visible_action = predicted_action

        pre_step_snapshot = self._build_recent_step_snapshot(
            absolute_direction=absolute_direction,
            action=visible_action,
            action_random=was_random,
            greedy_action=greedy_action,
        )

        pre_step_risks = evaluate_pre_step_risks(self.game, state, visible_action)
        is_transition_to_tail_loss = pre_step_risks[0]
        is_endgame = float(state[ENDGAME_IDX]) > 0.5
        is_pocket_split = float(state[POCKET_SPLIT_START_IDX + visible_action]) > 0.5
        is_pocket_merge = float(state[POCKET_MERGE_START_IDX + visible_action]) > 0.5

        _, reward, done, info = self.game.step(absolute_direction)
        reason = str(info["reason"])
        next_state = self._current_observation()

        reward, extra_penalties = apply_step_risk_shaping(
            reward,
            reason,
            len(self.game.snake),
            pre_step_risks,
            is_endgame=is_endgame,
            is_pocket_split=is_pocket_split,
            is_pocket_merge=is_pocket_merge,
            endgame_split_penalty=ENDGAME_SPLIT_PENALTY if self.agent_kind == "dqn" else 0.0,
            endgame_merge_bonus=ENDGAME_MERGE_BONUS if self.agent_kind == "dqn" else 0.0,
        )

        self.last_state = next_state
        self.last_q_values = q_values
        self.live_reward = reward
        self.episode_reward += reward
        self.last_action = visible_action
        self.last_action_random = was_random
        self.last_greedy_action = greedy_action
        self.last_reason = reason
        self.last_penalty_reasons = [str(item) for item in info.get("penalty_reasons", [])] + extra_penalties
        step_number = int(info.get("steps", self.game.step_count))
        self._record_decision_stat("choices", MODEL_ACTION_LABELS[visible_action])
        self._record_decision_stat("steps", self.last_reason)
        pre_step_snapshot["outcome_reason"] = self.last_reason
        pre_step_snapshot["penalty_reasons"] = list(self.last_penalty_reasons)
        pre_step_snapshot["steps_without_food"] = int(self.game.steps_without_food)
        pre_step_snapshot["max_steps_without_food"] = int(self.game.max_steps_without_food)
        self.recent_step_snapshots.append(pre_step_snapshot)

        # Record replay step only when recording is enabled
        if self.record_replays:
            q_vals = [round(float(v), 2) for v in q_values] if q_values is not None and len(q_values) > 0 else []
            self.current_replay_steps.append({
                "step": int(self.game.step_count),
                "snake": [[int(coord[0]), int(coord[1])] for coord in self.game.snake],
                "food": [int(self.game.food[0]), int(self.game.food[1])],
                "direction": int(self.game.direction),
                "action": int(visible_action),
                "q_values": q_vals,
                "was_random": bool(was_random),
                "reward": float(reward),
                "reason": str(reason),
                "penalties": list(self.last_penalty_reasons),
                "steps_without_food": int(self.game.steps_without_food),
            })
            if len(self.current_replay_steps) > 500:
                self.current_replay_steps.pop(0)

        if training_mode and self.agent_kind == "dqn":
            priority_scale = compute_dqn_priority_scale(
                done,
                reason,
                is_transition_to_tail_loss,
                self.game.snake[0],
                self.game.board_size,
            )

            self.agent.remember(
                state,
                visible_action,
                reward,
                next_state,
                done,
                priority_scale=priority_scale,
            )
            # Optimize once every 8 steps (or when done) to reduce CPU bottleneck and accelerate training
            if step_number % 8 == 0 or done:
                loss = self.agent.optimize()
                if loss is not None:
                    loss_fitness = -float(loss)
                    self.fitness_ema = loss_fitness if self.fitness_ema == 0 else self.fitness_ema * 0.92 + loss_fitness * 0.08

        if done:
            self._finish_episode(str(info["reason"]), training_mode=training_mode)

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if elapsed_ms >= SLOW_STEP_MS:
            log_perf(
                "slow_perform_step",
                elapsed_ms,
                training=training_mode,
                done=done,
                reason=reason,
                snake=len(self.game.snake),
                mode=self.mode,
            )

    def _finish_episode(self, reason: str, *, training_mode: bool) -> None:
        self.episode_counter += 1
        is_new_best = self.game.score > self.best_score
        self.best_score = max(self.best_score, self.game.score)
        self._sync_board_best_score()
        if reason in DEATH_REASON_ORDER:
            self._record_decision_stat("deaths", reason)
        recent_steps = self._recent_unique_death_steps(limit=8)
        self.last_death_snapshot = {
            "reason": reason,
            "size": self.game.board_size,
            "score": self.game.score,
            "step_count": self.game.step_count,
            "steps_without_food": self.game.steps_without_food,
            "direction": self.game.direction,
            "snake": [{"x": x, "y": y} for x, y in self.game.snake],
            "food": {"x": self.game.food[0], "y": self.game.food[1]},
            "eaten_food_trail": [{"x": x, "y": y} for x, y in self.game.eaten_food_trail],
            "recent_steps": recent_steps,
        }
        fitness = self.agent.finish_episode(
            score=self.game.score,
            reward=self.episode_reward,
            steps=self.game.step_count,
            reason=reason,
            board_size=self.game.board_size,
            max_steps_without_food=self.game.max_steps_without_food,
        ) if training_mode else float(getattr(self.agent, "last_fitness", 0.0))

        # Save replay for individual game
        self._save_replay(self.current_replay_steps, self.game.score, reason, prefix="live")
        self.current_replay_steps = []
        
        if training_mode:
            self.fitness_ema = fitness if self.fitness_ema == 0 else self.fitness_ema * 0.85 + fitness * 0.15

        self.history.append(EpisodeStats(
            episode=self.episode_counter,
            score=self.game.score,
            reward=self.episode_reward,
            mutation=float(getattr(self.agent, "epsilon", getattr(self.agent, "mutation_rate", 0.0))),
            fitness=fitness,
            steps=self.game.step_count,
            best_score=self.best_score,
            reason=reason,
            loss=float(getattr(self.agent, "last_loss", 0.0)) if self.mode != "smart" else 0.0,
        ))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)

        self.last_outcome_reason = reason
        self.game.reset()
        self.episode_reward = 0.0
        self.last_state = self._current_observation()
        _, self.last_q_values, _, self.last_greedy_action = self._select_current_action(self.last_state, False)
        self._schedule_save(urgent=is_new_best or reason == "victory")

    def _recent_unique_death_steps(self, limit: int = 5) -> list[dict[str, Any]]:
        unique_steps: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()

        for snapshot in reversed(self.recent_step_snapshots):
            signature = json.dumps(
                {
                    "snake": snapshot.get("snake"),
                    "food": snapshot.get("food"),
                    "planned_target": snapshot.get("planned_target"),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_steps.append(snapshot)
            if len(unique_steps) >= limit:
                break

        unique_steps.reverse()
        return unique_steps

    def _reset_environment_state(self) -> None:
        self._ensure_runtime_consistency()
        self.live_reward = 0.0
        self.episode_reward = 0.0
        self.last_action = 0
        self.last_action_random = False
        self.last_greedy_action = 0
        self.last_reason = "new_game"
        self.last_outcome_reason = "new_game"
        self.last_penalty_reasons = []
        self.recent_step_snapshots.clear()
        self.current_replay_steps.clear()
        self.last_state = self._current_observation()
        _, self.last_q_values, _, self.last_greedy_action = self._select_current_action(self.last_state, False)

    def _save_replay(self, steps: list[dict[str, Any]], score: int, reason: str, prefix: str) -> None:
        if not self.record_replays or not steps:
            return
        self.replay_manager.submit(
            steps,
            score,
            reason,
            prefix,
            self.game.board_size,
            self.episode_counter,
            self.agent_kind,
        )

    def _clear_replays_directory(self) -> None:
        self.replay_manager.clear()

    def _build_recent_step_snapshot(
        self,
        *,
        absolute_direction: int,
        action: int,
        action_random: bool,
        greedy_action: int,
    ) -> dict[str, Any]:
        head_x, head_y = self.game.snake[0]
        move_x, move_y = DIRECTIONS[absolute_direction]
        next_head = {"x": head_x + move_x, "y": head_y + move_y}
        return {
            "size": self.game.board_size,
            "score": self.game.score,
            "step_count": self.game.step_count,
            "direction": self.game.direction,
            "snake": [{"x": x, "y": y} for x, y in self.game.snake],
            "food": {"x": self.game.food[0], "y": self.game.food[1]},
            "eaten_food_trail": [{"x": x, "y": y} for x, y in self.game.eaten_food_trail],
            "planned_direction": absolute_direction,
            "planned_target": next_head,
            "action": int(action),
            "action_random": bool(action_random),
            "greedy_action": int(greedy_action),
            "move_risks": self.game.relative_move_risks(),
        }

    def _reset_decision_stats(self) -> None:
        self.decision_stats = make_decision_stats()

    def _reset_death_stats(self) -> None:
        self.decision_stats["deaths"] = make_decision_stat_bucket(DEATH_REASON_ORDER)

    def _clear_runtime_review_state(self) -> None:
        self._reset_death_stats()
        self.last_death_snapshot = None
        self.recent_step_snapshots.clear()
        self.last_outcome_reason = "new_game"
        self.last_penalty_reasons = []

    def _record_decision_stat(self, section: str, key: str) -> None:
        record_decision_stat(self.decision_stats, section, key)

    def _reward_rules_payload(self) -> dict[str, Any]:
        return build_reward_rules_payload(agent_kind=self.agent_kind, mode=self.mode)

    def _active_context_payload(self) -> dict[str, object]:
        return {
            "agent_kind": self.agent_kind,
            "mode": self.mode,
            "board_size": int(self.game.board_size),
            "smart_model": self.smart_agent.model_path.name if self.smart_agent.model_path else "",
            "speed": self.speed,
            "record_replays": self.record_replays,
        }

    def _save_active_context(self) -> None:
        if not self.persistence_enabled:
            return
        write_json_atomic(ACTIVE_CONTEXT_PATH, self._active_context_payload())

    def _load_active_context(self) -> bool:
        payload = None
        if ACTIVE_CONTEXT_PATH.exists():
            try:
                loaded = json.loads(ACTIVE_CONTEXT_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = None
        if payload is None:
            store = load_aux_stats_store()
            payload = store.get("active_context") if isinstance(store, dict) else None
        if not isinstance(payload, dict):
            return False

        agent_kind = str(payload.get("agent_kind", self.agent_kind))
        if agent_kind in AGENT_KINDS:
            self.agent_kind = agent_kind

        try:
            board_size = int(payload.get("board_size", self.game.board_size))
        except (TypeError, ValueError):
            board_size = self.game.board_size
        if board_size in BOARD_SIZES:
            self.game = SnakeEnv(board_size=board_size)

        smart_model = str(payload.get("smart_model", "")).strip()
        if smart_model:
            candidate = (RUNTIME_DIR / Path(smart_model).name).resolve()
            try:
                if candidate.parent == RUNTIME_DIR.resolve() and candidate.suffix.lower() == ".h5" and candidate.exists():
                    self.smart_agent.set_model_path(candidate)
            except OSError:
                pass

        mode = str(payload.get("mode", self.mode))
        if mode == "smart" and self.smart_agent.loaded:
            self.mode = "smart"
        else:
            self.mode = "ai"

        try:
            self.speed = int(payload.get("speed", self.speed))
        except (TypeError, ValueError):
            pass

        self.record_replays = bool(payload.get("record_replays", True))

        self.agent = self._create_native_agent(self.agent_kind)
        self._reset_training_runner(self.game.board_size)
        self._sync_network_source()
        return True

    def _agent_state_payload(self) -> dict[str, Any]:
        raw_best_fitness = float(getattr(self.agent, "best_fitness", float("-inf")))
        safe_best_fitness = raw_best_fitness if np.isfinite(raw_best_fitness) else None
        if self.agent_kind == "ppo":
            return {
                "total_updates": int(getattr(self.agent, "total_updates", 0)),
                "last_loss": float(getattr(self.agent, "last_loss", 0.0)),
                "last_policy_loss": float(getattr(self.agent, "last_policy_loss", 0.0)),
                "last_value_loss": float(getattr(self.agent, "last_value_loss", 0.0)),
                "last_entropy": float(getattr(self.agent, "last_entropy", 0.0)),
                "last_value": float(getattr(self.agent, "last_value", 0.0)),
                "last_approx_kl": float(getattr(self.agent, "last_approx_kl", 0.0)),
                "best_avg_score": float(getattr(self.agent, "best_avg_score", 0.0)),
                "best_fill_ratio": float(getattr(self.agent, "best_fill_ratio", 0.0)),
                "effective_tail_bias": float(self.agent.effective_tail_bias()) if isinstance(self.agent, PPOAgent) else 0.0,
            }
        return {
            "epsilon": float(getattr(self.agent, "epsilon", 1.0)),
            "total_updates": int(getattr(self.agent, "total_updates", 0)),
            "last_loss": float(getattr(self.agent, "last_loss", 0.0)),
            "best_score": int(getattr(self.agent, "best_score", 0)),
            "best_fitness": safe_best_fitness,
            "last_fitness": float(getattr(self.agent, "last_fitness", 0.0)),
        }

    def _runtime_payload(self) -> dict[str, Any]:
        return {
            "game": self.game.export_state(),
            "running": bool(self.running),
            "mode": str(self.mode),
            "speed": int(self.speed),
            "live_reward": float(self.live_reward),
            "episode_reward": float(self.episode_reward),
            "last_action": int(self.last_action),
            "last_action_random": bool(self.last_action_random),
            "last_greedy_action": int(self.last_greedy_action),
            "last_reason": str(self.last_reason),
            "last_outcome_reason": str(self.last_outcome_reason),
            "last_penalty_reasons": list(self.last_penalty_reasons),
            "record_replays": bool(self.record_replays),
        }

    def _load_agent_state_payload(self, payload: dict[str, Any] | None) -> None:
        payload = payload if isinstance(payload, dict) else {}
        if self.agent_kind == "ppo":
            self.agent.total_updates = int(payload.get("total_updates", getattr(self.agent, "total_updates", 0)))
            self.agent.last_loss = float(payload.get("last_loss", getattr(self.agent, "last_loss", 0.0)))
            self.agent.last_policy_loss = float(payload.get("last_policy_loss", getattr(self.agent, "last_policy_loss", 0.0)))
            self.agent.last_value_loss = float(payload.get("last_value_loss", getattr(self.agent, "last_value_loss", 0.0)))
            self.agent.last_entropy = float(payload.get("last_entropy", getattr(self.agent, "last_entropy", 0.0)))
            self.agent.last_value = float(payload.get("last_value", getattr(self.agent, "last_value", 0.0)))
            self.agent.last_approx_kl = float(payload.get("last_approx_kl", getattr(self.agent, "last_approx_kl", 0.0)))
            self.agent.best_avg_score = max(
                float(getattr(self.agent, "best_avg_score", 0.0)),
                float(payload.get("best_avg_score", 0.0) or 0.0),
            )
            self.agent.best_fill_ratio = max(
                float(getattr(self.agent, "best_fill_ratio", 0.0)),
                float(payload.get("best_fill_ratio", 0.0) or 0.0),
            )
            return
        raw_best_fitness = payload.get("best_fitness")
        loaded_best_fitness = float("-inf") if raw_best_fitness is None else float(raw_best_fitness)
        self.agent.epsilon = float(payload.get("epsilon", getattr(self.agent, "epsilon", 1.0)))
        self.agent.total_updates = int(payload.get("total_updates", getattr(self.agent, "total_updates", 0)))
        self.agent.last_loss = float(payload.get("last_loss", getattr(self.agent, "last_loss", 0.0)))
        self.agent.best_score = int(payload.get("best_score", getattr(self.agent, "best_score", 0)))
        self.agent.best_fitness = loaded_best_fitness
        self.agent.last_fitness = float(payload.get("last_fitness", getattr(self.agent, "last_fitness", 0.0)))

    def _load_runtime_payload(self, payload: dict[str, Any] | None) -> None:
        payload = payload if isinstance(payload, dict) else {}
        game_payload = payload.get("game")
        if isinstance(game_payload, dict):
            self.last_state = self.game.load_state(game_payload)
            self.last_state = self._current_observation()
        self.running = bool(payload.get("running", self.running))
        self.training = False
        loaded_mode = str(payload.get("mode", self.mode))
        self.mode = "smart" if (loaded_mode == "smart" and self.smart_agent.loaded) else "ai"
        self.speed = int(payload.get("speed", self.speed))
        self.live_reward = float(payload.get("live_reward", 0.0))
        self.episode_reward = float(payload.get("episode_reward", 0.0))
        self.last_action = int(payload.get("last_action", 0))
        self.last_action_random = bool(payload.get("last_action_random", False))
        self.last_greedy_action = int(payload.get("last_greedy_action", self.last_action))
        self.last_reason = str(payload.get("last_reason", "new_game"))
        self.last_outcome_reason = str(payload.get("last_outcome_reason", "new_game"))
        penalty_reasons = payload.get("last_penalty_reasons", [])
        self.last_penalty_reasons = [str(item) for item in penalty_reasons] if isinstance(penalty_reasons, list) else []

    # =========================================================================
    # ОПТИМИЗИРОВАННАЯ СЕРИАЛИЗАЦИЯ СТАТИСТИКИ (КОМПАКТНЫЙ JSON)
    # =========================================================================
    def _pack_aux_stats(self) -> dict[str, Any]:
        packed_choices = [CHOICE_TO_SYMBOL.get(x, x) for x in self.decision_stats["choices"]["recent"]]
        packed_steps = [STEP_TO_SYMBOL.get(x, x) for x in self.decision_stats["steps"]["recent"]]
        packed_deaths = [DEATH_TO_SYMBOL.get(x, x) for x in self.decision_stats["deaths"]["recent"]]
        
        return {
            "ep": self.episode_counter,
            "bs": self.best_score,
            "bbs": {str(k): v for k, v in self.board_best_scores.items()},
            "fit": round(self.fitness_ema, 3),
            "hist": [
                [
                    i.episode,
                    i.score,
                    i.steps,
                    DEATH_TO_SYMBOL.get(i.reason, i.reason),
                    round(i.fitness, 3),
                    round(i.reward, 3),
                    round(i.mutation, 4),
                    round(i.loss, 4),
                    i.best_score,
                ]
                for i in self.history
            ],
            "ds": {
                "c": {"a": self.decision_stats["choices"]["all"], "r": packed_choices},
                "s": {"a": self.decision_stats["steps"]["all"], "r": packed_steps},
                "d": {"a": self.decision_stats["deaths"]["all"], "r": packed_deaths},
            },
            "ag": self._agent_state_payload(),
            "rt": self._runtime_payload(),
        }

    def _unpack_aux_stats(self, payload: dict[str, Any], *, include_runtime: bool = False) -> None:
        if not isinstance(payload, dict):
            return
        if include_runtime:
            self._load_runtime_payload(payload.get("rt"))
        self._load_agent_state_payload(payload.get("ag"))
        
        self.episode_counter = int(payload.get("ep", 0))
        self.best_score = int(payload.get("bs", 0))
        self.fitness_ema = float(payload.get("fit", 0.0))
        
        self.board_best_scores = {}
        bbs = payload.get("bbs", {})
        if isinstance(bbs, dict):
            for key, value in bbs.items():
                try:
                    board_size = int(key)
                    if board_size in BOARD_SIZES:
                        self.board_best_scores[board_size] = int(value)
                except (TypeError, ValueError):
                    continue
        
        if not self.board_best_scores:
            self.board_best_scores[int(self.game.board_size)] = self.best_score
        else:
            current_board_size = int(self.game.board_size)
            self.board_best_scores[current_board_size] = max(
                int(self.board_best_scores.get(current_board_size, 0)), self.best_score
            )
        self._select_board_best_score()
        
        self.history.clear()
        for item in payload.get("hist", []):
            try:
                if isinstance(item, list) and len(item) >= 9:
                    episode = int(item[0])
                    score = int(item[1])
                    steps = int(item[2])
                    reason = SYMBOL_TO_DEATH.get(str(item[3]), str(item[3]))
                    fitness = float(item[4])
                    reward = float(item[5])
                    mutation = float(item[6])
                    loss = float(item[7])
                    best_score = int(item[8])
                else:
                    continue
                
                self.history.append(EpisodeStats(
                    episode=episode,
                    score=score,
                    reward=reward,
                    mutation=mutation,
                    fitness=fitness,
                    steps=steps,
                    best_score=best_score,
                    reason=reason,
                    loss=loss,
                ))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        if len(self.history) > HISTORY_LIMIT:
            self.history = self.history[-HISTORY_LIMIT:]

        ds = payload.get("ds", {})
        self._reset_decision_stats()
        if isinstance(ds, dict):
            for section, code, symbol_map in [
                ("choices", "c", SYMBOL_TO_CHOICE),
                ("steps", "s", SYMBOL_TO_STEP),
                ("deaths", "d", SYMBOL_TO_DEATH)
            ]:
                section_payload = ds.get(code, {})
                if not isinstance(section_payload, dict):
                    continue
                bucket = self.decision_stats[section]
                all_payload = section_payload.get("a", {})
                if isinstance(all_payload, dict):
                    for key, value in all_payload.items():
                        bucket["all"][str(key)] = int(value)
                recent_payload = section_payload.get("r", [])
                if isinstance(recent_payload, list):
                    bucket["recent"] = deque(
                        (symbol_map.get(str(i), str(i)) for i in recent_payload[-DECISION_STATS_WINDOW:]),
                        maxlen=DECISION_STATS_WINDOW
                    )

    def _aux_stats_payload(self) -> dict[str, Any]:
        return self._pack_aux_stats()

    def _load_aux_stats_payload(self, payload: dict[str, Any] | None, *, include_runtime: bool = False) -> None:
        self._unpack_aux_stats(payload, include_runtime=include_runtime)
    # =========================================================================

    def _reset_aux_stats(self) -> None:
        self.best_score = 0
        self.board_best_scores = {int(self.game.board_size): 0}
        self.episode_counter = 0
        self.fitness_ema = 0.0
        self.history.clear()
        self._reset_decision_stats()
        self.last_death_snapshot = None

    def _save_state(self) -> None:
        self._save_active_context()
        payloads = self._capture_save_payloads()
        if payloads is None:
            return
        self._write_save_payloads(*payloads)
        self._next_save_at = 0.0

    def _clear_aux_stats_json(self) -> None:
        clear_aux_stats_store()

    def _capture_save_payloads(self) -> tuple[Path, dict[str, object], str, dict[str, object]] | None:
        if not self.persistence_enabled:
            return None
        checkpoint_path = self._checkpoint_path()
        checkpoint_payload = self.agent.export_checkpoint()
        # Вшиваем подпись архитектуры в чекпоинт
        checkpoint_payload["arch_signature"] = arch_signature_for_agent(self.agent_kind)
        aux_payload = self._aux_stats_payload()
        return checkpoint_path, checkpoint_payload, self._active_aux_stats_key(), aux_payload

    def _write_save_payloads(
        self,
        checkpoint_path: Path,
        checkpoint_payload: dict[str, object],
        checkpoint_key: str,
        aux_payload: dict[str, object],
    ) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self._save_active_context()
        torch.save(checkpoint_payload, checkpoint_path)
        store = load_aux_stats_store()
        models = store.setdefault("models", {})
        if isinstance(models, dict):
            if self.mode != "smart":
                compact_model_stats(models, checkpoint_key, self.agent_kind)
            models[checkpoint_key] = aux_payload
            keep_keys = {
                checkpoint_key,
                model_stats_key(self.agent_kind),
                checkpoint_path.stem,
                self._active_aux_stats_key(),
            }
            # Сохраняем слоты статистики для ОБОИХ нативных режимов (dqn + ppo),
            # чтобы при переключении режима не удалялись данные другого агента.
            for native_kind in AGENT_KINDS:
                keep_keys.add(checkpoint_path_for_agent(native_kind).stem)
                keep_keys.add(model_stats_key(native_kind))
            prune_model_stats_store(models, keep_keys=keep_keys, smart_limit=3)
        save_aux_stats_store(store)

    def _delete_active_model_storage(self) -> None:
        checkpoint_path = self._checkpoint_path()
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError:
            pass
        store = load_aux_stats_store()
        models = store.get("models", {})
        if isinstance(models, dict):
            for key in {
                self._active_aux_stats_key(),
                checkpoint_path.stem,
                model_stats_key(self.agent_kind),
            }:
                models.pop(key, None)
        save_aux_stats_store(store)

    def _schedule_save(self, *, urgent: bool = False) -> None:
        if self.persistence_enabled:
            self._pending_save = True
            if urgent:
                self._next_save_at = 0.0
            elif self._next_save_at <= 0:
                self._next_save_at = time.perf_counter() + SAVE_THROTTLE_SECONDS

    def _flush_pending_save(self) -> bool:
        with self.lock:
            if not self._pending_save or self._save_in_flight:
                return False
            now = time.perf_counter()
            if self._next_save_at > 0 and now < self._next_save_at:
                return False
            self._pending_save = False
            self._next_save_at = 0.0
            payloads = self._capture_save_payloads()
            if payloads is None:
                return False
            self._save_in_flight = True

        def _bg_save() -> None:
            try:
                self._write_save_payloads(*payloads)
            finally:
                self._save_in_flight = False

        try:
            self._save_executor.submit(_bg_save)
        except RuntimeError:
            self._save_in_flight = False
            self._write_save_payloads(*payloads)
        return True

    def _load_state(self, *, force_cpu: bool) -> bool:
        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            return False

        try:
            payload = torch.load(checkpoint_path, map_location="cpu")
            
            # Проверяем совместимость архитектуры по сигнатуре
            stored_signature = payload.get("arch_signature")
            if stored_signature != arch_signature_for_agent(self.agent_kind):
                # Архитектура изменилась, игнорируем старые веса
                return False

            self._swap_device("cpu", payload)
            stats_store = load_aux_stats_store()
            stats_models = stats_store.get("models", {}) if isinstance(stats_store, dict) else {}
            stats_payload = None
            if isinstance(stats_models, dict):
                stats_payload = stats_models.get(self._active_aux_stats_key())
                if not isinstance(stats_payload, dict):
                    stats_payload = stats_models.get(self._checkpoint_path().stem)
                if not isinstance(stats_payload, dict):
                    stats_payload = stats_models.get(model_stats_key(self.agent_kind))
            
            if isinstance(stats_payload, dict):
                self._load_aux_stats_payload(stats_payload, include_runtime=True)
            else:
                self._reset_aux_stats()

            self.last_state = self._current_observation()
            _, self.last_q_values, _, self.last_greedy_action = self._select_current_action(self.last_state, False)
            self._sync_network_source()
            return True
        except Exception:
            self.device_mode = "cpu"
            self.device, self.device_kind, self.device_details = resolve_device(self.device_mode)
            self.device_label = self.device_kind
            self.agent = self._create_native_agent(self.agent_kind)
            self._store_active_agent_slot()
            self.board_best_scores = {int(self.game.board_size): 0}
            self.best_score = 0
            self._reset_environment_state()
            self._sync_network_source()
            return False

    def _swap_device(self, mode: str, checkpoint_payload: dict[str, object] | None = None) -> None:
        desired_mode = "cpu"
        self.device_mode = desired_mode
        self.device, self.device_kind, self.device_details = resolve_device(desired_mode)
        self.device_label = self.device_kind
        if checkpoint_payload is None:
            self.agent.move_to_device(self.device)
        else:
            self.agent = self._create_native_agent(self.agent_kind)
            self.agent.import_checkpoint(checkpoint_payload)
            self._reset_training_runner()
        self._store_active_agent_slot(loaded_checkpoint=checkpoint_payload is not None)
        self._sync_network_source()

    def _set_device_mode(self, mode: str) -> tuple[bool, str]:
        if mode != "cpu":
            return False, "Unknown device mode"
        if mode == self.device_mode:
            return True, "ok"
        self._swap_device(mode)
        _, self.last_q_values, _, self.last_greedy_action = self._select_current_action(self.last_state, False)
        return True, "ok"

    def _set_agent_kind(self, agent_kind: str) -> tuple[bool, str]:
        if agent_kind not in AGENT_KINDS:
            return False, "Unknown agent kind"
        if agent_kind == self.agent_kind:
            was_smart_mode = self.mode == "smart"
            if was_smart_mode:
                self._cancel_greedy_eval(clear_result=True)
            self.mode = "ai"
            self.training = False
            self.running = False
            self._sync_network_source()
            if was_smart_mode:
                current_board_size = self.game.board_size
                self.game = SnakeEnv(board_size=current_board_size)
                if not self._load_active_aux_stats():
                    self._clear_runtime_review_state()
                self.fitness_ema = 0.0
                self._reset_environment_state()
            return True, "ok"

        self._cancel_greedy_eval(clear_result=True)
        current_board_size = self.game.board_size
        self._sync_board_best_score()
        self._store_active_agent_slot()
        if self.mode != "smart":
            self._save_state()
        self.agent_kind = agent_kind
        key = self._agent_cache_key(self.agent_kind)
        loaded = False
        with self.agent_slot_lock:
            slot = self.agent_slots.get(key)
            if slot and slot.ready and slot.agent is not None:
                self.agent = slot.agent
                loaded = slot.loaded_checkpoint
            else:
                self.agent = self._create_native_agent(self.agent_kind)
                try:
                    loaded = self._load_agent_checkpoint_into(self.agent, self.agent_kind)
                    error = ""
                except Exception as exc:
                    loaded = False
                    error = str(exc)
                self.agent_slots[key] = AgentSlot(
                    agent=self.agent,
                    ready=True,
                    loading=False,
                    error=error,
                    loaded_checkpoint=loaded,
                )

        self._reset_training_runner(current_board_size)
        self.game = SnakeEnv(board_size=current_board_size)
        self._select_board_best_score(current_board_size)
        stats_store = load_aux_stats_store()
        stats_models = stats_store.get("models", {}) if isinstance(stats_store, dict) else {}
        stats_payload = None
        if isinstance(stats_models, dict):
            checkpoint_key = checkpoint_path_for_agent(self.agent_kind).stem
            stats_payload = stats_models.get(checkpoint_key)
            if not isinstance(stats_payload, dict):
                stats_payload = stats_models.get(model_stats_key(self.agent_kind))
        if isinstance(stats_payload, dict):
            self._load_aux_stats_payload(stats_payload)
        else:
            self._reset_aux_stats()

        self.mode = "ai"
        self.training = False
        self.running = False
        self.fitness_ema = 0.0
        self._cancel_greedy_eval(clear_result=True)
        self._reset_environment_state()
        self._schedule_save(urgent=True)
        return True, "ok"

    def _set_board_size(self, board_size: int) -> tuple[bool, str]:
        if board_size not in BOARD_SIZES:
            return False, "Invalid board size"
        if board_size == self.game.board_size:
            return True, "ok"
        self.running = False
        self._sync_board_best_score()
        self.game = SnakeEnv(board_size=board_size)
        self._reset_training_runner(board_size)
        if isinstance(self.agent, PPOAgent):
            with self.agent_lock:
                self.agent.clear_rollout()
                self.agent.on_board_size_changed(board_size)
        self._cancel_greedy_eval(clear_result=True)
        self._select_board_best_score(board_size)
        self._reset_environment_state()
        self._schedule_save(urgent=True)
        return True, "ok"

    def _cycle_board_size(self) -> tuple[bool, str]:
        self.running = False
        self._sync_board_best_score()
        current = BOARD_SIZES.index(self.game.board_size)
        next_board_size = BOARD_SIZES[(current + 1) % len(BOARD_SIZES)]
        return self._set_board_size(next_board_size)

    def _network_payload(self) -> dict[str, Any]:
        
        started_at = time.perf_counter()
        cache_used = False
        global_signal_labels = self._current_signal_labels()
        expected_signal_count = self._expected_state_size()
        
        self._ensure_runtime_consistency()
        use_ppo_inference = self.agent_kind == "ppo" and isinstance(self.agent, PPOAgent)
        acquired_agent_lock = False
        if not use_ppo_inference:
            acquired_agent_lock = self.agent_lock.acquire(timeout=0.02 if self.running else -1)
            if not acquired_agent_lock:
                output_values = np.asarray(self.last_q_values, dtype=np.float32)
                if self._network_payload_cache is not None:
                    cache_used = True
                    cached = dict(self._network_payload_cache)
                    cached["output"] = output_values.tolist()
                    cached["global_signals"]["values"] = self.last_state.tolist()
                    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                    if elapsed_ms >= SLOW_NETWORK_MS:
                        log_perf("slow_network_payload", elapsed_ms, cache=cache_used, locked=False)
                    return cached

                net_h1 = int(getattr(self.agent.online_net, "hidden1_size", HIDDEN_LAYER_1))
                net_h2 = int(getattr(self.agent.online_net, "hidden2_size", HIDDEN_LAYER_2))
                payload = {
                    "global_signals": {
                        "labels": global_signal_labels,
                        "values": self.last_state.tolist(),
                    },
                    "hidden1": [0.0] * net_h1,
                    "hidden2": [0.0] * net_h2,
                    "output": output_values.tolist(),
                    "value": [],
                    "connections": {
                        "global_hidden1": [],
                        "hidden1_hidden2": [],
                        "hidden2_output": [],
                        "hidden2_value": [],
                    },
                }
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                if elapsed_ms >= SLOW_NETWORK_MS:
                    log_perf("slow_network_payload", elapsed_ms, cache=cache_used, locked=False)
                return payload

        try:
            ego_map = self.game.get_egocentric_map() if use_ppo_inference else None
            if use_ppo_inference:
                snapshot = self.agent.snapshot(self.last_state, ego_map=ego_map, use_inference_net=True)
                net = self.agent.inference_net
            else:
                snapshot = self.agent.snapshot(self.last_state)
                net = self.agent.online_net
            net_h1 = int(getattr(net, "hidden1_size", HIDDEN_LAYER_1))
            net_h2 = int(getattr(net, "hidden2_size", HIDDEN_LAYER_2))
            input_values = np.asarray(snapshot.get("input", np.zeros(expected_signal_count)), dtype=np.float32)
            hidden1_values = np.asarray(snapshot.get("hidden1", np.zeros(net_h1)), dtype=np.float32)
            hidden2_values = np.asarray(snapshot.get("hidden2", np.zeros(net_h2)), dtype=np.float32)
            output_values = np.asarray(snapshot.get("output", np.zeros(len(MODEL_ACTION_LABELS))), dtype=np.float32)
            value_values = np.asarray(snapshot.get("value", np.array([], dtype=np.float32)), dtype=np.float32)

            input_hidden1_connections = []
            hidden1_hidden2_connections = []
            hidden2_output_connections = []
            hidden2_value_connections = []

            try:
                weight_ctx = self.agent._inference_lock if use_ppo_inference else None
                if weight_ctx is not None:
                    weight_ctx.acquire()
                try:
                    if hasattr(net, 'hidden1_layer'):
                        if hasattr(net, 'input_layer'):
                            w_in = net.input_layer.weight.detach()
                            w_h1 = net.hidden1_layer.weight.detach()
                            w_in_h1 = (w_h1 @ w_in).cpu().numpy()
                        else:
                            w_in_h1 = net.hidden1_layer.weight.detach()[:, :len(input_values)].cpu().numpy()

                        input_hidden1_connections = build_connection_payload(
                            input_values, hidden1_values, w_in_h1,
                            len(input_values), 14, 48
                        )

                    # hidden1 -> hidden2: используется вес именно hidden2_layer (H2, H1)
                    if hasattr(net, 'hidden1_layer') and hasattr(net, 'hidden2_layer'):
                        w_h1_h2 = net.hidden2_layer.weight.detach().cpu().numpy()
                        hidden1_hidden2_connections = build_connection_payload(
                            hidden1_values, hidden2_values, w_h1_h2,
                            14, 10, 90
                        )

                    w_out = None
                    if hasattr(net, 'advantage_head'):
                        w_out = net.advantage_head.weight.detach().cpu().numpy()
                    elif hasattr(net, 'output'):
                        w_out = net.output.weight.detach().cpu().numpy()

                    if w_out is not None:
                        hidden2_output_connections = build_connection_payload(
                            hidden2_values, output_values, w_out,
                            10, len(output_values), 30
                        )

                    if hasattr(net, 'value_head'):
                        w_val = net.value_head.weight.detach().cpu().numpy()
                        val_source = np.asarray(snapshot.get("critic_hidden2", hidden2_values), dtype=np.float32)
                        hidden2_value_connections = build_connection_payload(
                            val_source, value_values, w_val,
                            10, len(value_values), 16
                        )
                finally:
                    if weight_ctx is not None:
                        weight_ctx.release()
            except Exception as e:
                log_perf("network_payload_weight_error", 0.0, error=str(e))

            payload = {
                "global_signals": {
                    "labels": global_signal_labels,
                    "values": input_values.tolist(),
                },
                "ego_map": ego_map.tolist() if ego_map is not None else None,
                "hidden1": hidden1_values.tolist(),
                "hidden2": hidden2_values.tolist(),
                "output": output_values.tolist(),
                "value": value_values.tolist(),
                "connections": {
                    "global_hidden1": input_hidden1_connections,
                    "hidden1_hidden2": hidden1_hidden2_connections,
                    "hidden2_output": hidden2_output_connections,
                    "hidden2_value": hidden2_value_connections,
                },
            }
            self._network_payload_cache = payload
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if elapsed_ms >= SLOW_NETWORK_MS:
                log_perf("slow_network_payload", elapsed_ms, cache=cache_used, locked=True, dense=False)
            return payload
        except Exception as e:
            log_perf("network_payload_error", 0.0, error=str(e))
            return {
                "global_signals": {"labels": global_signal_labels, "values": []},
                "hidden1": [], "hidden2": [], "output": [], "value": [],
                "connections": {"global_hidden1": [], "hidden1_hidden2": [], "hidden2_output": [], "hidden2_value": []},
            }
        finally:
            if acquired_agent_lock:
                self.agent_lock.release()

    def _cached_state_snapshot(self, include_network: bool, include_details: bool) -> dict[str, Any] | None:
        cached = self._snapshot_cache.get((include_network, include_details))
        if cached is None and not include_network:
            cached = self._snapshot_cache.get((False, True))
        if cached is None:
            cached = self._snapshot_cache.get((True, True)) or self._snapshot_cache.get((True, False))
        if cached is None:
            return None
        payload = dict(cached)
        if not include_network:
            payload["network"] = None
        if not include_details:
            for key in (
                "reward_rules",
                "observation", "global_signals", "last_death_snapshot", "decision_stats", "history",
            ):
                payload.pop(key, None)
        payload["backend_busy"] = True
        return payload

    def _stats_payload(self) -> dict[str, Any]:
        best_fitness = 0.0
        if self.mode != "smart":
            raw_best_fitness = float(getattr(self.agent, "best_fitness", 0.0))
            best_fitness = raw_best_fitness if np.isfinite(raw_best_fitness) else 0.0
        score_window_size = average_score_window(self.episode_counter, len(self.history))
        recent_scores = [float(item.score) for item in self.history[-score_window_size:]] if score_window_size else []
        average_score_100 = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
        is_dqn = self.agent_kind == "dqn" and self.mode != "smart"
        is_ppo = self.agent_kind == "ppo" and self.mode != "smart"
        replay_buffer = getattr(self.agent, "replay_buffer", None)
        if is_dqn:
            buffer_size = len(replay_buffer) if replay_buffer is not None else 0
            buffer_capacity = 0
        elif is_ppo:
            buffer_size = int(getattr(self.agent, "rollout_buffer_len", 0))
            buffer_capacity = int(getattr(self.agent, "rollout_capacity", 0))
        else:
            buffer_size = 0
            buffer_capacity = 0

        ppo_envs = int(getattr(self.training_runner, "num_envs", 0)) if (is_ppo and self.training_runner is not None) else 0
        return {
            "episode": self.episode_counter,
            "best_score": self.best_score,
            "average_score_100": average_score_100,
            "average_score_window": len(recent_scores),
            "reward": self.live_reward,
            "fitness": self.fitness_ema,
            "mutation": 0.0,
            "generation": 0,
            "individual": int(self.training_display_index + 1) if (is_ppo and self.training) else 1,
            "population": ppo_envs,
            "victory_archive": 0,
            "best_fitness": best_fitness,
            "avg_fitness": 0.0,
            "alive": 1 if (self.training and is_dqn) else (ppo_envs if (self.training and is_ppo) else 0),
            "completed": 0,
            "leader": 1 if (self.training and is_dqn) else ((self.training_display_index + 1) if (self.training and is_ppo) else 0),
            "epsilon": float(getattr(self.agent, "epsilon", 0.0)) if is_dqn else 0.0,
            "buffer": buffer_size,
            "rollout_capacity": buffer_capacity,
            "value_estimate": float(getattr(self.agent, "last_value", 0.0)) if is_ppo else 0.0,
            "entropy": float(getattr(self.agent, "last_entropy", 0.0)) if is_ppo else 0.0,
            "best_fill_ratio": float(getattr(self.agent, "best_fill_ratio", 0.0)) if is_ppo else 0.0,
            "effective_tail_bias": float(self.agent.effective_tail_bias()) if (is_ppo and isinstance(self.agent, PPOAgent)) else 0.0,
            "eval_running": bool(self._eval_running),
            "last_eval_result": (
                dict(self.last_eval_result)
                if (
                    isinstance(self.last_eval_result, dict)
                    and self.last_eval_result.get("agent_kind") == self._active_eval_kind()
                    and int(self.last_eval_result.get("board_size", 0)) == int(self.game.board_size)
                )
                else None
            ),
            "loss": float(getattr(self.agent, "last_loss", 0.0)) if (is_dqn or is_ppo) else 0.0,
            "updates": int(getattr(self.agent, "total_updates", 0)) if (is_dqn or is_ppo) else 0,
            "steps": self.game.step_count,
            "hunger_steps": max(0, int(self.game.steps_without_food)),
            "hunger_budget": safe_hunger_budget_for(self.game.score, self.game.board_size),
            "hunger_ratio": min(
                1.0,
                float(max(0, int(self.game.steps_without_food)))
                / max(1.0, safe_hunger_budget_for(self.game.score, self.game.board_size)),
            ),
            "max_hunger_steps": max(0, int(self.game.max_steps_without_food)),
        }

    def _frame_payload_unlocked(self) -> dict[str, Any]:
        # Кэшируем tolist() один раз для q_values
        q_values_list = self.last_q_values.tolist()
        # Кэшируем snake как tuple для быстрого копирования
        snake_tuple = tuple(self.game.snake)
        food_tuple = self.game.food
        trail_tuple = tuple(self.game.eaten_food_trail)
        
        return {
            "device_kind": self.device_kind,
            "device_label": self.device_label,
            "device_mode": self.device_mode,
            "agent_kind": self.agent_kind,
            "network_source": self.network_source,
            "frame_id": self.frame_id,
            "running": self.running,
            "training": self.training,
            "mode": self.mode,
            "speed": self.speed,
            "visual_speed": self._visual_speed(),
            "last_action": self.last_action,
            "last_action_random": self.last_action_random,
            "last_greedy_action": self.last_greedy_action,
            "last_reason": self.last_reason,
            "last_outcome_reason": self.last_outcome_reason,
            "last_penalty_reasons": list(self.last_penalty_reasons),
            "game": {
                "size": self.game.board_size,
                "score": self.game.score,
                "step_count": self.game.step_count,
                "direction": self.game.direction,
                "snake": [{"x": x, "y": y} for x, y in snake_tuple],
                "food": {"x": food_tuple[0], "y": food_tuple[1]},
                "eaten_food_trail": [{"x": x, "y": y} for x, y in trail_tuple],
            },
            "stats": self._stats_payload(),
            "action_scores": q_values_list,
            "q_values": q_values_list,
            "reason_labels": dict(REASON_LABELS),
            "signal_labels": dict(SIGNAL_LABELS),
            "terminal_reasons": list(TERMINAL_REASONS),
        }

    def _update_frame_cache(self) -> None:
        self.frame_id += 1
        self._snapshot_cache.clear()
        payload = self._frame_payload_unlocked()
        encoded = _encode_frame_json(payload)
        self._frame_cache = payload
        self._last_frame_payload = dict(payload)
        self._frame_json_bytes = encoded
        with self._frame_cond:
            self._frame_seq += 1
            self._frame_cond.notify_all()

    def frame_json_bytes(self) -> bytes:
        data = self._frame_json_bytes
        if data and data != b"{}":
            return data
        return _encode_frame_json(self.frame_snapshot())

    def wait_next_frame_bytes(self, last_seq: int, timeout: float = 0.25) -> tuple[int, bytes | None]:
        with self._frame_cond:
            if self._frame_seq <= last_seq and not self.stop_event.is_set():
                self._frame_cond.wait(timeout=timeout)
            if self._frame_seq > last_seq:
                return self._frame_seq, self._frame_json_bytes
            return self._frame_seq, None

    def frame_snapshot(self) -> dict[str, Any]:
        if self._frame_cache is not None:
            return dict(self._frame_cache)
        acquired = self.lock.acquire(timeout=0.004)
        if not acquired:
            if self._last_frame_payload is not None:
                return dict(self._last_frame_payload)
            return {
                "frame_id": self.frame_id,
                "backend_busy": True,
                "running": self.running,
                "training": self.training,
                "mode": self.mode,
                "speed": self.speed,
                "visual_speed": self._visual_speed(),
                "game": {
                    "size": self.game.board_size,
                    "score": self.game.score,
                    "step_count": self.game.step_count,
                    "direction": self.game.direction,
                    "snake": [{"x": x, "y": y} for x, y in self.game.snake],
                    "food": {"x": self.game.food[0], "y": self.game.food[1]},
                    "eaten_food_trail": [{"x": x, "y": y} for x, y in self.game.eaten_food_trail],
                },
                "stats": self._stats_payload(),
                "action_scores": self.last_q_values.tolist(),
                "q_values": self.last_q_values.tolist(),
                "reason_labels": dict(REASON_LABELS),
                "signal_labels": dict(SIGNAL_LABELS),
                "terminal_reasons": list(TERMINAL_REASONS),
            }
        try:
            payload = self._frame_payload_unlocked()
            self._frame_cache = payload
            self._last_frame_payload = dict(payload)
            return payload
        finally:
            if acquired:
                self.lock.release()

    def _busy_state_snapshot(self) -> dict[str, Any]:
        # Мгновенный возврат последнего кэша без пересборки payload и tolist()
        base = self._last_frame_payload or self._frame_cache
        if base is not None:
            payload = dict(base)
            payload["backend_busy"] = True
            return payload
        return {
            "frame_id": self.frame_id, "backend_busy": True, "running": self.running,
            "training": self.training, "mode": self.mode, "speed": self.speed,
            "game": {"size": self.game.board_size, "score": self.game.score, "step_count": self.game.step_count},
            "stats": self._stats_payload(),
        }

    def state_snapshot(self, *, include_network: bool = True, include_details: bool = True) -> dict[str, Any]:
        started_at = time.perf_counter()
        cache_used = False
        acquired = self.lock.acquire(timeout=0.025 if self.running else -1)
        if not acquired:
            cached = self._cached_state_snapshot(include_network, include_details)
            if cached is not None:
                cache_used = True
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                if elapsed_ms >= SLOW_API_MS:
                    log_perf("slow_state_snapshot", elapsed_ms, cache=cache_used, network=include_network, details=include_details, running=self.running, training=self.training)
                return cached
            if self.running or self.training:
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                if elapsed_ms >= SLOW_API_MS:
                    log_perf("busy_state_snapshot", elapsed_ms, network=include_network, details=include_details, running=self.running, training=self.training)
                return self._busy_state_snapshot()
            self.lock.acquire()
            acquired = True
        try:
            # Under lock, check if we already have this snapshot cached for the current frame
            cached = self._snapshot_cache.get((include_network, include_details))
            if cached is not None:
                cache_used = True
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                if elapsed_ms >= SLOW_API_MS:
                    log_perf("slow_state_snapshot", elapsed_ms, cache=cache_used, network=include_network, details=include_details, running=self.running, training=self.training)
                return dict(cached)
                
            self._ensure_runtime_consistency()
            self._sync_network_source()
            payload = {
                "device_kind": self.device_kind,
                "device_label": self.device_label,
                "device_details": self.device_details,
                "device_mode": self.device_mode,
                "available_devices": self.available_devices,
                "model_input_size": int(self.last_state.size),
                "ppo_map_size": PPO_MAP_SIZE,
                "network_input_adapter_mode": NETWORK_INPUT_ADAPTER_MODE,
                "agent_kind": self.agent_kind,
                "available_agent_kinds": list(AGENT_KINDS),
                "agent_slots": self._agent_slots_payload(),
                "restart_pending": self.restart_pending,
                "network_source": self.network_source,
                "frame_id": self.frame_id,
                "smart_model": {
                    "loaded": self.smart_agent.loaded,
                    "path": str(self.smart_agent.model_path) if self.smart_agent.model_path else "",
                    "name": self.smart_agent.model_path.name if self.smart_agent.model_path else "",
                    "error": self.smart_agent.error or None,
                    "models": self._smart_models_payload(),
                },
                "output_labels": list(MODEL_ACTION_LABELS),
                "running": self.running,
                "training": self.training,
                "mode": self.mode,
                "speed": self.speed,
                "visual_speed": self._visual_speed(),
                "last_action": self.last_action,
                "last_action_random": self.last_action_random,
                "last_greedy_action": self.last_greedy_action,
                "last_reason": self.last_reason,
                "last_outcome_reason": self.last_outcome_reason,
                "last_penalty_reasons": list(self.last_penalty_reasons),
                "persistence_enabled": self.persistence_enabled,
                "record_replays": self.record_replays,
                "game": {
                    "size": self.game.board_size,
                    "score": self.game.score,
                    "step_count": self.game.step_count,
                    "direction": self.game.direction,
                    "snake": [{"x": x, "y": y} for x, y in self.game.snake],
                    "food": {"x": self.game.food[0], "y": self.game.food[1]},
                    "eaten_food_trail": [{"x": x, "y": y} for x, y in self.game.eaten_food_trail],
                },
                "stats": self._stats_payload(),
                "action_scores": self.last_q_values.tolist(),
                "q_values": self.last_q_values.tolist(),
                "network": self._network_payload() if include_network else None,
                "reason_labels": dict(REASON_LABELS),
                "signal_labels": dict(SIGNAL_LABELS),
                "terminal_reasons": list(TERMINAL_REASONS),
            }
            if include_details:
                payload.update({
                    "reward_rules": self._reward_rules_payload(),
                    "observation": self.last_state.tolist(),
                    "global_signals": {
                        "labels": self._current_signal_labels(),
                        "values": self.last_state.tolist(),
                    },
                    "last_death_snapshot": self.last_death_snapshot,
                    "decision_stats": export_decision_stats(self.decision_stats),
                    "history": [item.to_dict() for item in self.history],
                })
            self._snapshot_cache[(include_network, include_details)] = payload
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if elapsed_ms >= SLOW_API_MS:
                log_perf("slow_state_snapshot", elapsed_ms, cache=cache_used, network=include_network, details=include_details, running=self.running, training=self.training)
            return payload
        finally:
            if acquired:
                self.lock.release()

    def _active_eval_kind(self) -> str:
        return "smart" if self.mode == "smart" else str(self.agent_kind)

    def _cancel_greedy_eval(self, *, clear_result: bool = True) -> None:
        self._eval_generation += 1
        self._eval_cancel_event.set()
        self._eval_running = False
        if clear_result:
            self.last_eval_result = None

    def _run_greedy_eval_sync(
        self,
        episodes: int | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any] | None:
        """Выполняет честную жадную оценку (pure policy, bias=0.0 для PPO) без влияния на обучение."""
        with self.lock:
            if cancel_event is not None and cancel_event.is_set():
                return None
            board_size = int(self.game.board_size)
            agent_kind = str(self.agent_kind)
            mode = str(self.mode)
            target_episodes = int(episodes) if episodes is not None else (20 if board_size <= 20 else 10)
            target_episodes = max(1, min(50, target_episodes))
            eval_agent: Any = None
            if mode == "smart":
                if not self.smart_agent.loaded:
                    return None
                eval_kind = "smart"
                eval_agent = self.smart_agent
            elif agent_kind == "ppo" and isinstance(self.agent, PPOAgent):
                eval_kind = "ppo"
                eval_agent = PPOAgent(
                    global_signal_count=self.agent.global_signal_count,
                    device=torch.device("cpu"),
                    map_channels=self.agent.map_channels,
                    map_size=self.agent.map_size,
                    conv1_channels=self.agent.conv1_channels,
                    conv2_channels=self.agent.conv2_channels,
                    map_feat_size=self.agent.map_feat_size,
                    hidden1=self.agent.hidden1,
                    hidden2=self.agent.hidden2,
                    tail_blocked_logit_bias=0.0,
                )
                with self.agent._inference_lock:
                    snap_weights = {k: v.detach().cpu().clone() for k, v in self.agent.inference_net.state_dict().items()}
                eval_agent.online_net.load_state_dict(snap_weights)
                eval_agent.online_net.eval()
                eval_agent.sync_inference_net()
            else:
                eval_kind = "dqn"
                eval_agent = DQNAgent(device=torch.device("cpu"))
                with self.agent_lock:
                    snap_weights = {k: v.detach().cpu().clone() for k, v in self.agent.online_net.state_dict().items()}
                eval_agent.online_net.load_state_dict(snap_weights)
                eval_agent.online_net.eval()

        started_at = time.perf_counter()
        victory_score = board_size * board_size - 3
        max_steps_cap = board_size * board_size * 25
        scores: list[int] = []
        wins = 0
        completed_eps = 0

        for _ in range(target_episodes):
            if self.stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                return None
            env = SnakeEnv(board_size=board_size)
            env.reset(return_obs=False)
            while not env.done and env.step_count < max_steps_cap:
                if self.stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                    return None
                if eval_kind == "ppo":
                    sig = env.get_ppo_signals()
                    ego = env.get_egocentric_map(eval_agent.map_size)
                    act, _, _, _ = eval_agent.select_best_action(sig, ego_map=ego)
                elif eval_kind == "smart":
                    act, _, _, _ = eval_agent.select_action(env)
                else:
                    sig = env.get_observation_grid()
                    act, _, _, _ = eval_agent.select_best_action(sig)
                env.step(relative_to_absolute(env.direction, act), return_obs=False)
            if self.stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                return None
            completed_eps += 1
            scores.append(int(env.score))
            if env.last_reason == "victory":
                wins += 1

        elapsed_s = max(0.01, time.perf_counter() - started_at)
        avg_score = float(sum(scores)) / float(len(scores)) if scores else 0.0
        avg_fill_pct = 100.0 * (avg_score + 3.0) / float(board_size * board_size) if scores else 0.0
        win_rate_pct = 100.0 * float(wins) / float(completed_eps) if completed_eps else 0.0
        return {
            "agent_kind": eval_kind,
            "board_size": board_size,
            "episodes": completed_eps,
            "victories": wins,
            "victory_score": victory_score,
            "win_rate_pct": round(win_rate_pct, 1),
            "avg_score": round(avg_score, 1),
            "min_score": int(min(scores)) if scores else 0,
            "max_score": int(max(scores)) if scores else 0,
            "avg_fill_pct": round(avg_fill_pct, 1),
            "elapsed_s": round(elapsed_s, 2),
            "pure_policy": eval_kind == "ppo",
        }

    def _start_greedy_eval(self, episodes: int | None = None, *, sync: bool = False) -> bool:
        if self._eval_running:
            return False
        if self.mode == "smart" and not self.smart_agent.loaded:
            return False
        self._eval_generation += 1
        gen = self._eval_generation
        cancel_event = threading.Event()
        self._eval_cancel_event = cancel_event
        self._eval_running = True
        if sync:
            try:
                res = self._run_greedy_eval_sync(episodes=episodes, cancel_event=cancel_event)
                with self.lock:
                    if gen == self._eval_generation and not cancel_event.is_set():
                        if res is not None:
                            self.last_eval_result = res
                        self._eval_running = False
                        self._update_frame_cache()
            except Exception:
                with self.lock:
                    if gen == self._eval_generation:
                        self._eval_running = False
                        self._update_frame_cache()
            return True

        def _worker() -> None:
            try:
                res = self._run_greedy_eval_sync(episodes=episodes, cancel_event=cancel_event)
                with self.lock:
                    if gen == self._eval_generation and not cancel_event.is_set() and res is not None:
                        self.last_eval_result = res
            finally:
                with self.lock:
                    if gen == self._eval_generation:
                        self._eval_running = False
                        self._update_frame_cache()

        threading.Thread(target=_worker, daemon=True, name="snake-greedy-eval").start()
        return True

    def _command_state_response(self, command: str) -> dict[str, Any] | None:
        light_state_commands = {
            "toggle_running", "toggle_training", "set_agent_kind", "set_mode",
            "set_smart_model", "set_board_size", "cycle_board_size",
            "set_speed", "set_device", "reset_agent", "eval_greedy",
            "restart_server", "shutdown_server",
        }
        if command in light_state_commands:
            return self.frame_snapshot()
        return self.state_snapshot(include_network=False, include_details=False)

    def handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        command = str(payload.get("type", ""))
        value = payload.get("value")

        quick_training_toggle = command == "toggle_training"
        quick_running_stop = command == "toggle_running" and self.running
        acquired = self.lock.acquire(timeout=0.5 if (quick_training_toggle or quick_running_stop) else -1)
        if not acquired:
            return {"ok": False, "error": "Backend busy", "busy": True}

        try:
            if command == "toggle_running":
                self.running = not self.running
                if self.running and self.training and self.agent_kind == "ppo":
                    self._ppo_wake_event.set()
            elif command == "set_mode":
                if self.training and self.agent_kind == "ppo":
                    self.training = False
                    with self._pause_ppo_worker():
                        pass
                # Перезагружаем модель если путь есть, но модель не загружена
                if value == "smart" and not self.smart_agent.loaded and self.smart_agent.model_path:
                    self.smart_agent.set_model_path(self.smart_agent.model_path)
                
                if value == "smart" and not self.smart_agent.loaded:
                    # Разрешаем выбор режима smart даже без модели для демонстрации
                    # Но устанавливаем флаг что модель недоступна
                    pass  # Продолжаем вместо возврата ошибки
                
                if value in {"ai", "smart"}:
                    previous_mode = self.mode
                    switching_agent_mode = {previous_mode, str(value)} <= {"ai", "smart"} and previous_mode != str(value)
                    if switching_agent_mode:
                        self._cancel_greedy_eval(clear_result=True)
                    self.mode = str(value)
                    self.training = False
                    if value == "smart":
                        self.running = bool(self.smart_agent.loaded)
                        self.game = SnakeEnv(board_size=self.game.board_size)
                        if not self._load_active_aux_stats():
                            # У smart-режима может не быть собственного сохранённого слота статистики.
                            # В этом случае не обнуляем глобальную историю/счётчики, а очищаем
                            # только runtime-обзор, чтобы не затирать последние значения в store.
                            self._clear_runtime_review_state()
                        self.fitness_ema = 0.0
                    elif value == "ai":
                        self.game = SnakeEnv(board_size=self.game.board_size)
                        self.running = True
                        if switching_agent_mode and not self._load_active_aux_stats():
                            self._clear_runtime_review_state()
                    self.last_state = self._current_observation()
                    _, self.last_q_values, _, self.last_greedy_action = self._select_current_action(self.last_state, False)
            elif command == "set_agent_kind":
                if not isinstance(value, str):
                    return {"ok": False, "error": "Invalid agent kind"}
                with self._pause_ppo_worker():
                    ok, message = self._set_agent_kind(value)
                if not ok:
                    return {"ok": False, "error": message}
            elif command == "set_smart_model":
                if not isinstance(value, str):
                    return {"ok": False, "error": "Invalid smart model"}
                with self._pause_ppo_worker():
                    ok, message = self._set_smart_model(value)
                if not ok:
                    return {"ok": False, "error": message}
            elif command == "toggle_training":
                if self.mode == "smart":
                    return {"ok": False, "error": "Zmey-smart training is unavailable"}
                next_training = bool(value) if isinstance(value, bool) else not self.training
                if not next_training and self.training and self.agent_kind == "ppo":
                    self.training = False
                    with self._pause_ppo_worker():
                        pass
                else:
                    self.training = next_training
                self.mode = "ai"
                if self.training:
                    self.running = True
                    if self.agent_kind == "ppo":
                        self._ppo_wake_event.set()
            elif command == "set_board_size":
                if isinstance(value, int) and value in BOARD_SIZES:
                    with self._pause_ppo_worker():
                        ok, message = self._set_board_size(value)
                    if not ok:
                        return {"ok": False, "error": message}
            elif command == "cycle_board_size":
                with self._pause_ppo_worker():
                    self._cycle_board_size()
            elif command == "set_speed":
                if isinstance(value, int):
                    self.speed = max(1, min(64, value))
                    self._save_active_context()
            elif command == "set_device":
                if not isinstance(value, str):
                    return {"ok": False, "error": "Invalid device value"}
                ok, message = self._set_device_mode(value)
                if not ok:
                    return {"ok": False, "error": message}
            elif command == "set_record_replays":
                if not isinstance(value, bool):
                    return {"ok": False, "error": "Value must be a boolean"}
                self.record_replays = value
            elif command == "eval_greedy":
                episodes_val: int | None = None
                sync_val = False
                if isinstance(value, int):
                    episodes_val = value
                elif isinstance(value, dict):
                    if isinstance(value.get("episodes"), int):
                        episodes_val = int(value["episodes"])
                    sync_val = bool(value.get("sync", False))
                self._start_greedy_eval(episodes=episodes_val, sync=sync_val)
            elif command == "reset_agent":
                with self._pause_ppo_worker():
                    self._cancel_greedy_eval(clear_result=True)
                    if self.mode == "smart":
                        store = load_aux_stats_store()
                        models = store.get("models", {})
                        if isinstance(models, dict):
                            models.pop(self._active_aux_stats_key(), None)
                            if self.smart_agent.model_path:
                                models.pop(self.smart_agent.model_path.stem, None)
                        save_aux_stats_store(store)
                        self._clear_runtime_review_state()
                        self.game = SnakeEnv(board_size=self.game.board_size)
                        self.fitness_ema = 0.0
                        self.best_score = 0
                        self.board_best_scores = {int(self.game.board_size): 0}
                        self.episode_counter = 0
                        self.history.clear()
                        self.training = False
                        self.running = False
                        self._reset_environment_state()
                    else:
                        self._clear_replays_directory()
                        board_size = self.game.board_size
                        self._delete_active_model_storage()
                        self.game = SnakeEnv(board_size=board_size)
                        self.agent = self._create_native_agent(self.agent_kind)
                        self._store_active_agent_slot()
                        self._reset_training_runner(board_size)
                        self._reset_decision_stats()
                        self._clear_runtime_review_state()
                        self.fitness_ema = 0.0
                        self.best_score = 0
                        self.board_best_scores = {board_size: 0}
                        self.episode_counter = 0
                        self.history.clear()
                        self.training = False
                        self.mode = "ai"
                        self.running = False
                        self._reset_environment_state()
                        self._save_state()
            elif command in {"restart_server", "shutdown_server"}:
                self.running = False
                self.training = False
                with self._pause_ppo_worker():
                    self.restart_pending = command == "restart_server"
                    self.last_reason = "restart_pending" if command == "restart_server" else "shutdown_pending"
                    self.last_penalty_reasons = []
                    self._save_active_context()
                    self._save_state()
            else:
                return {"ok": False, "error": f"Unknown command: {command}"}

            if command in {
                "set_agent_kind", "set_mode", "set_smart_model", "set_board_size",
                "cycle_board_size", "set_speed", "reset_agent",
                "restart_server", "shutdown_server", "set_record_replays",
            }:
                self._save_active_context()
            self._update_frame_cache()
            result = {
                "ok": True,
                "state": self._command_state_response(command),
            }
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if elapsed_ms >= SLOW_API_MS:
                log_perf("slow_command", elapsed_ms, command=command, running=self.running, training=self.training)
            return result
        finally:
            self.lock.release()

def main() -> None:
    from web_api import main as api_main
    api_main(
        backend_class=SnakeWebBackend,
        static_root=STATIC_ROOT,
        runtime_dir=RUNTIME_DIR,
        pid_path=PID_PATH,
        reset_perf_log_func=reset_session_perf_log,
        log_perf_func=log_perf,
    )

if __name__ == "__main__":
    main()
