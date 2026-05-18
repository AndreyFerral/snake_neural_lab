from __future__ import annotations

import os
# Настройки окружения
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Deque, Literal, TypedDict
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Импорты только тех констант, которые реально используются
from backend_config import (
    AGENT_KINDS,
    AVERAGE_SCORE_WINDOW_RATIO,
    BOARD_SIZES,
    DEATH_REASON_ORDER,
    DECISION_STATS_WINDOW,
    DIRECTIONS,
    DQN_DEATH_REPLAY_EXTRA_COPIES,
    DQN_PRE_DEATH_CONTEXT_REPLAYS,
    ELITE_EPISODE_MAX_TRANSITIONS,
    EPISODE_TRANSITION_LIMIT,
    GLOBAL_SIGNAL_LABELS,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    HISTORY_LIMIT,
    MODEL_ACTION_LABELS,
    PERF_LOG_MAX_BYTES,
    SAVE_THROTTLE_SECONDS,
    SLOW_API_MS,
    SLOW_NETWORK_MS,
    SLOW_STEP_MS,
    STARTUP_BOARD_SIZE,
    STEP_REASON_ORDER,
    PENALTY_CATALOG,
    REWARD_CATALOG,
    hunger_budget_for,
)

from dqn_agent import DQNAgent
from evolution_agent import EvolutionAgent, ParallelEvolutionRunner
from snake_env import SnakeEnv, relative_to_absolute
from smart_agent import SmartH5Agent, smart_model_candidates

# =========================================================================
# ЖЁСТКАЯ ФИКСАЦИЯ: ТОЛЬКО СИГНАЛЫ + АВТОПОДПИСЬ АРХИТЕКТУРЫ
# =========================================================================
GLOBAL_SIGNAL_COUNT = len(GLOBAL_SIGNAL_LABELS)
# Подпись меняется автоматически при изменении количества сигналов или размеров слоёв
ARCH_SIGNATURE = f"sig_{GLOBAL_SIGNAL_COUNT}_{HIDDEN_LAYER_1}_{HIDDEN_LAYER_2}"
# =========================================================================

STATIC_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = STATIC_ROOT / ".runtime"
AUX_STATS_PATH = RUNTIME_DIR / "snake_model_stats.json"
ACTIVE_CONTEXT_PATH = RUNTIME_DIR / "active_context.json"
PID_PATH = STATIC_ROOT / ".web_backend.pid"
RESTART_FLAG_PATH = RUNTIME_DIR / "restart_requested.flag"
PERF_LOG_PATH = RUNTIME_DIR / "backend_perf.log"
RESTART_IN_PARENT_ENV = "SNAKE_RESTART_IN_PARENT"
RESTART_EXIT_CODE = 42


def reset_session_perf_log() -> None:
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        PERF_LOG_PATH.write_text("", encoding="utf-8")
        rotated = PERF_LOG_PATH.with_suffix(".log.1")
        if rotated.exists():
            rotated.unlink()
    except OSError:
        pass


def log_perf(event: str, elapsed_ms: float, **fields: Any) -> None:
    if not RUNTIME_DIR.exists():
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if PERF_LOG_PATH.exists() and PERF_LOG_PATH.stat().st_size > PERF_LOG_MAX_BYTES:
            PERF_LOG_PATH.replace(PERF_LOG_PATH.with_suffix(".log.1"))
    except OSError:
        pass
    
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {event} {elapsed_ms:.1f}ms"
    if suffix:
        line += f" {suffix}"
    try:
        with open(PERF_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def top_indices(values: np.ndarray, count: int) -> list[int]:
    if values.size == 0:
        return []
    return np.argsort(np.abs(values))[-count:].tolist()


def list_available_devices() -> list[dict[str, str]]:
    return [{"id": "cpu", "label": "CPU"}]


def make_decision_stat_bucket(initial_keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        "all": {key: 0 for key in initial_keys},
        "recent": deque(maxlen=DECISION_STATS_WINDOW),
    }


def make_decision_stats() -> dict[str, dict[str, Any]]:
    return {
        "choices": make_decision_stat_bucket(MODEL_ACTION_LABELS),
        "steps": make_decision_stat_bucket(STEP_REASON_ORDER),
        "deaths": make_decision_stat_bucket(DEATH_REASON_ORDER),
    }


def export_decision_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for section, bucket in stats.items():
        payload[section] = {
            "all": dict(bucket.get("all", {})),
            "recent": list(bucket.get("recent", [])),
        }
    return payload


def average_score_window(total: int, available: int) -> int:
    total = max(0, int(total))
    available = max(0, int(available))
    if total <= 0 or available <= 0:
        return 0
    return min(available, max(1, int(round(total * AVERAGE_SCORE_WINDOW_RATIO))))


def compact_death_snapshot(snapshot: Any) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    
    allowed_keys = {
        "reason", "size", "score", "step_count", "direction", "snake",
        "food", "eaten_food_trail", "recent_steps", "planned_direction",
        "planned_target", "action", "action_random", "greedy_action",
        "move_risks", "outcome_reason", "penalty_reasons",
    }
    compact = {key: value for key, value in snapshot.items() if key in allowed_keys}
    recent_steps = compact.get("recent_steps")
    if isinstance(recent_steps, list):
        compact["recent_steps"] = [
            compact_step
            for step in recent_steps
            if (compact_step := compact_death_snapshot(step)) is not None
        ]
    return compact


def checkpoint_path_for_agent(agent_kind: str) -> Path:
    # Имя файла без версии, версия проверяется внутри файла по ARCH_SIGNATURE
    return RUNTIME_DIR / f"snake_{agent_kind}.pt"


def model_stats_key(agent_kind: str) -> str:
    return f"snake_{agent_kind}"


def compact_model_stats(models: dict[str, Any], checkpoint_key: str, agent_kind: str) -> None:
    prefix = f"snake_{agent_kind}"
    for key in list(models.keys()):
        if key == checkpoint_key:
            continue
        if key.startswith(prefix):
            models.pop(key, None)


def load_aux_stats_store() -> dict[str, Any]:
    default_store: dict[str, Any] = {"schema_version": 1, "models": {}}
    if not AUX_STATS_PATH.exists():
        return default_store
    try:
        payload = json.loads(AUX_STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default_store
    if not isinstance(payload, dict):
        return default_store
    models = payload.get("models", {})
    return {
        "schema_version": int(payload.get("schema_version", 1)),
        "models": models if isinstance(models, dict) else {},
    }


def save_aux_stats_store(store: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = AUX_STATS_PATH.with_suffix(".tmp")
    payload = json.dumps(store, ensure_ascii=False, separators=(",", ":"))
    for attempt in range(5):
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(AUX_STATS_PATH)
            break
        except OSError:
            if attempt < 4:
                time.sleep(0.05)


def resolve_device(mode: str) -> tuple[torch.device, str, str]:
    return torch.device("cpu"), "CPU", "CPU"


def build_connection_payload(
    from_values: np.ndarray,
    to_values: np.ndarray,
    weights: np.ndarray,
    from_count: int,
    to_count: int,
    limit: int,
) -> list[dict[str, float | int]]:
    if weights.ndim != 2 or weights.shape[0] < to_values.size or weights.shape[1] < from_values.size:
        return []

    important_from = sorted(top_indices(from_values, from_count))
    important_to = sorted(top_indices(to_values, to_count))

    candidates: list[tuple[float, int, int, float]] = []
    for to_index in important_to:
        for from_index in important_from:
            weight = float(weights[to_index, from_index])
            influence = abs(weight * float(from_values[from_index]) * float(to_values[to_index]))
            candidates.append((influence, from_index, to_index, weight))

    if not candidates:
        return []

    # Фронтенд будет использовать influence как opacity/brightness, поэтому 1.0 → 65%
    max_influence = max(c[0] for c in candidates)
    if max_influence > 0.0:
        candidates = [((inf / max_influence) ** 1.4 * 0.5, fi, ti, w) for inf, fi, ti, w in candidates]

    payload: list[dict[str, float | int]] = []
    ranked = sorted(candidates, reverse=True)
    for influence, from_index, to_index, weight in ranked[:limit]:
        if influence <= 0.05:  # Порог отсечения слабых связей
            continue
        payload.append({
            "from_index": from_index,
            "to_index": to_index,
            "weight": weight,
            "influence": influence,
        })

    # Фоллбэк: если слой слабый, показываем топ по весу с минимальной видимой яркостью
    if not payload and ranked:
        fallback_count = max(6, min(limit, limit // 3))
        fallback = sorted(ranked, key=lambda item: abs(item[3]), reverse=True)[:fallback_count]
        for influence, from_index, to_index, weight in fallback:
            payload.append({
                "from_index": from_index,
                "to_index": to_index,
                "weight": weight,
                "influence": max(0.05, influence),
            })

    return payload


@dataclass(slots=True)
class EpisodeStats:
    episode: int
    score: int
    reward: float
    mutation: float
    fitness: float
    steps: int
    best_score: int
    reason: str
    loss: float = 0.0


@dataclass
class AgentSlot:
    agent: Any | None = None
    ready: bool = False
    loading: bool = False
    error: str = ""
    loaded_checkpoint: bool = False
    updated_at: float = 0.0


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
        self.agent_slots: dict[str, AgentSlot] = {}
        self.stop_event = threading.Event()
        
        self._pending_save = False
        self._next_save_at = 0.0
        self._last_save_at = 0.0
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
        self.agent_kind = "evolution"
        # Создаем агента только с глобальными сигналами (аргументы input_grid/mode удалены)
        self.agent = EvolutionAgent(device=self.device)
        
        self.training_runner = ParallelEvolutionRunner(
            self.agent.population_size,
            STARTUP_BOARD_SIZE,
            env_factory=SnakeEnv,
            action_labels=MODEL_ACTION_LABELS,
        )
        self.smart_agent = SmartH5Agent(RUNTIME_DIR, STATIC_ROOT)
        self.running = True
        self.training = False
        self.mode = "manual"
        self.speed = 8
        self.pending_manual_direction = self.game.direction
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
        self.step_review_entries: Deque[dict[str, Any]] = deque(maxlen=120)
        self.last_completed_step_review_entries: list[dict[str, Any]] = []
        self.step_review_history: Deque[dict[str, Any]] = deque(maxlen=1)
        self.current_episode_transitions: Deque[tuple[np.ndarray, int, float, np.ndarray, bool, float]] = deque(maxlen=EPISODE_TRANSITION_LIMIT)
        self.training_display_index = 0
        self._training_step_counter = 0

        self._sync_network_source()
        self._load_active_context()
        self._load_state(force_cpu=force_cpu)
        self._store_active_agent_slot(loaded_checkpoint=True)
        
        startup_board_size = int(self.game.board_size)
        self.training_runner = ParallelEvolutionRunner(
            self.agent.population_size,
            startup_board_size,
            env_factory=SnakeEnv,
            action_labels=MODEL_ACTION_LABELS,
        )
        self.game = SnakeEnv(board_size=startup_board_size)
        self._select_board_best_score(startup_board_size)
        self._reset_environment_state()
        self.running = False
        self._update_frame_cache()

        self._preload_native_agent_slots(blocking=True)
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()

    def shutdown(self, *, fast: bool = False) -> None:
        self.stop_event.set()
        self.thread.join(timeout=0.4 if fast else 2.0)
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

    def _current_observation(self) -> np.ndarray:
        # Всегда возвращает только вектор сигналов
        return self.game.get_observation_grid()

    def _checkpoint_path(self) -> Path:
        return checkpoint_path_for_agent(self.agent_kind)

    def _expected_state_size(self) -> int:
        return GLOBAL_SIGNAL_COUNT

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
            self.current_episode_transitions.clear()
            
        if int(getattr(self.last_state, "size", 0)) != expected_state_size:
            self.last_state = self._current_observation()

    def _agent_cache_key(self, agent_kind: str) -> str:
        return agent_kind

    def _create_native_agent(self, agent_kind: str) -> EvolutionAgent | DQNAgent:
        # Агенты сами вычисляют global_signal_count из констант
        if agent_kind == "dqn":
            return DQNAgent(device=self.device)
        return EvolutionAgent(device=self.device)

    def _store_active_agent_slot(self, *, loaded_checkpoint: bool = False) -> None:
        key = self._agent_cache_key(self.agent_kind)
        with self.agent_slot_lock:
            self.agent_slots[key] = AgentSlot(
                agent=self.agent,
                ready=True,
                loading=False,
                error="",
                loaded_checkpoint=loaded_checkpoint,
                updated_at=time.monotonic(),
            )

    def _load_agent_checkpoint_into(self, agent: EvolutionAgent | DQNAgent) -> bool:
        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            return False
        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict):
            agent.import_checkpoint(payload)
            return True
        return False

    def _preload_native_agent_slots(self, *, blocking: bool = False) -> None:
        for agent_kind in AGENT_KINDS:
            key = self._agent_cache_key(agent_kind)
            with self.agent_slot_lock:
                slot = self.agent_slots.get(key)
                if slot and (slot.ready or slot.loading):
                    continue
                self.agent_slots[key] = AgentSlot(loading=True, updated_at=time.monotonic())

            def preload(kind: str = agent_kind) -> None:
                loaded = False
                error = ""
                agent_obj: EvolutionAgent | DQNAgent | None = None
                try:
                    agent_obj = self._create_native_agent(kind)
                    loaded = self._load_agent_checkpoint_into(agent_obj)
                except Exception as exc:
                    error = str(exc)
                with self.agent_slot_lock:
                    self.agent_slots[kind] = AgentSlot(
                        agent=agent_obj,
                        ready=agent_obj is not None,
                        loading=False,
                        error=error,
                        loaded_checkpoint=loaded,
                        updated_at=time.monotonic(),
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
        elif self.agent_kind == "dqn":
            self.network_source = "DQN policy"
        else:
            self.network_source = "Evolution policy"

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
        self.training_runner = ParallelEvolutionRunner(
            self.agent.population_size,
            target_size,
            env_factory=SnakeEnv,
            action_labels=MODEL_ACTION_LABELS,
        )
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

        self._sync_network_source()
        self._smart_models_cache_at = 0.0
        self.mode = "smart"
        self.training = False
        self.running = False
        self.game = SnakeEnv(board_size=self.game.board_size)
        if not self._load_active_aux_stats():
            self._reset_aux_stats()
        self._reset_environment_state()
        self.last_state = self._current_observation()
        _, self.last_q_values, _, self.last_greedy_action = self._select_current_action(self.last_state, False)
        return True, "ok"

    def _select_current_action(self, state: np.ndarray, training_mode: bool) -> tuple[int, np.ndarray, bool, int]:
        if not training_mode:
            self._ensure_runtime_consistency()
        if self.mode == "smart" and not training_mode:
            return self.smart_agent.select_action(self.game)
        endgame = self.game._is_endgame()
        with self.agent_lock:
            if self.mode == "ai" and not training_mode:
                return self.agent.select_best_action(state, endgame=endgame)
            return self.agent.select_action(
                state,
                training=training_mode,
                snake_length=len(self.game.snake),
                endgame=endgame,
            )

    def _training_burst(self) -> int:
        speed = max(1, int(self.speed))
        return max(1, min(4, speed))

    def _simulation_loop(self) -> None:
        last_tick = time.perf_counter()
        accumulator = 0.0

        while not self.stop_event.is_set():
            now = time.perf_counter()
            delta = min(now - last_tick, 0.25)
            last_tick = now
            did_work = False
            training_steps = 0

            with self.lock:
                if self.running:
                    if self.training:
                        accumulator = min(accumulator + delta, 0.12)
                        if accumulator >= (1.0 / 60.0):
                            accumulator -= 1.0 / 60.0
                            training_steps = self._training_burst()
                            did_work = True
                    else:
                        step_interval = 1.0 / max(1, self.speed)
                        accumulator = min(accumulator + delta, step_interval * 2.0)
                        if accumulator >= step_interval:
                            accumulator -= step_interval
                            self._perform_step(training_mode=False)
                            did_work = True
                else:
                    accumulator = 0.0

            for _ in range(training_steps):
                with self.lock:
                    if not self.running or not self.training:
                        break
                    if self.agent_kind == "dqn":
                        self._perform_step(training_mode=True)
                    else:
                        self._perform_parallel_training_step()

            self._update_frame_cache()
            if self._flush_pending_save():
                did_work = True

            # Уменьшаем задержку для более плавной отрисовки
            time.sleep(0.001 if did_work else 0.005)

    def _record_training_completion(self, payload: dict[str, Any]) -> None:
        env = payload["env"]
        reason = str(payload.get("reason", env.last_reason))
        index = int(payload.get("index", -1))
        if 0 <= index < len(self.training_runner.step_logs):
            self.last_completed_step_review_entries = list(self.training_runner.step_logs[index])
            if self.last_completed_step_review_entries:
                self.step_review_history.appendleft({
                    "id": f"training-{self.episode_counter + 1}-{index + 1}-{int(time.time() * 1000)}",
                    "episode": self.episode_counter + 1,
                    "score": env.score,
                    "steps": env.step_count,
                    "reason": reason,
                    "entries": list(self.last_completed_step_review_entries),
                })
        
        self.episode_counter += 1
        self.best_score = max(self.best_score, env.score)
        self._sync_board_best_score()
        if reason in DEATH_REASON_ORDER:
            self._record_decision_stat("deaths", reason)
        
        self.history.append(EpisodeStats(
            episode=self.episode_counter,
            score=env.score,
            reward=float(payload.get("reward", 0.0)),
            mutation=float(getattr(self.agent, "mutation_rate", 0.0)),
            fitness=float(payload.get("fitness", 0.0)),
            steps=env.step_count,
            best_score=self.best_score,
            reason=reason,
            loss=float(payload.get("loss", 0.0)),
        ))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)
            
        self.last_death_snapshot = {
            "reason": reason,
            "size": env.board_size,
            "score": env.score,
            "step_count": env.step_count,
            "steps_without_food": env.steps_without_food,
            "direction": env.direction,
            "snake": [{"x": x, "y": y} for x, y in env.snake],
            "food": {"x": env.food[0], "y": env.food[1]},
            "eaten_food_trail": [{"x": x, "y": y} for x, y in env.eaten_food_trail],
            "recent_steps": [],
        }
        self.last_outcome_reason = reason
        self.last_penalty_reasons = [str(item) for item in payload.get("penalty_reasons", [])]
        fitness = float(payload.get("fitness", 0.0))
        self.fitness_ema = fitness if self.fitness_ema == 0 else self.fitness_ema * 0.85 + fitness * 0.15

    def _snapshot_for_completed_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        env = payload["env"]
        reason = str(payload.get("reason", env.last_reason))
        return {
            "reason": reason,
            "size": env.board_size,
            "score": env.score,
            "step_count": env.step_count,
            "steps_without_food": env.steps_without_food,
            "direction": env.direction,
            "snake": [{"x": x, "y": y} for x, y in env.snake],
            "food": {"x": env.food[0], "y": env.food[1]},
            "eaten_food_trail": [{"x": x, "y": y} for x, y in env.eaten_food_trail],
            "recent_steps": [],
            "individual": int(payload.get("index", -1)) + 1,
            "fitness": float(payload.get("fitness", 0.0)),
        }

    def _perform_parallel_training_step(self) -> None:
        started_at = time.perf_counter()
        self._ensure_runtime_consistency()
        alive_before = self.training_runner.alive_count()
        # Убраны input_grid и input_mode
        completed = self.training_runner.step_alive(self.agent)
        runner_profile = dict(self.training_runner.last_profile_ms)
        record_started_at = time.perf_counter()
        for item in completed:
            self._record_training_completion(item)
        record_ms = (time.perf_counter() - record_started_at) * 1000.0
        best_completed = self.training_runner.best_completed
        if isinstance(best_completed, dict):
            self.last_death_snapshot = self._snapshot_for_completed_training(best_completed)

        evolve_ms = 0.0
        if self.training_runner.all_done():
            evolve_started_at = time.perf_counter()
            self.agent.evolve_generation()
            self.training_runner.reset_generation(board_size=self.game.board_size)
            self._schedule_save(urgent=True)
            evolve_ms = (time.perf_counter() - evolve_started_at) * 1000.0

        leader_started_at = time.perf_counter()
        self.training_display_index = self.training_runner.update_leader()
        self.agent.current_index = self.training_display_index
        self.game = self.training_runner.leader_env()
        display_leader_ms = (time.perf_counter() - leader_started_at) * 1000.0
        if 0 <= self.training_display_index < len(self.training_runner.step_logs):
            self.step_review_entries = deque(
                self.training_runner.step_logs[self.training_display_index],
                maxlen=120,
            )
        display_obs_started_at = time.perf_counter()
        self.last_state = self._current_observation()
        display_observation_ms = (time.perf_counter() - display_obs_started_at) * 1000.0
        display_inference_started_at = time.perf_counter()
        _, self.last_q_values, _, self.last_greedy_action = self.agent.select_individual_action(
            self.training_display_index,
            self.last_state,
            endgame=self.game._is_endgame(),
        )
        display_inference_ms = (time.perf_counter() - display_inference_started_at) * 1000.0
        self.last_action = self.last_greedy_action
        self.last_action_random = False
        self.last_reason = self.game.last_reason
        self.live_reward = self.game.last_reward
        self._record_decision_stat("steps", self.last_reason)
        self._record_decision_stat("choices", MODEL_ACTION_LABELS[self.last_action])

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if elapsed_ms >= SLOW_STEP_MS:
            log_perf(
                "slow_parallel_training_step",
                elapsed_ms,
                alive_before=alive_before,
                completed=len(completed),
                generation=self.agent.generation,
                leader=self.training_display_index + 1,
                board=self.game.board_size,
                snake=len(self.game.snake),
                score=self.game.score,
                obs_ms=runner_profile.get("observation", 0.0),
                infer_ms=runner_profile.get("inference", 0.0),
                env_ms=runner_profile.get("env_step", 0.0),
                evolve_ms=round(evolve_ms, 1),
                record_ms=round(record_ms, 1),
                leader_ms=round(display_leader_ms, 1),
                display_obs_ms=round(display_observation_ms, 1),
                display_infer_ms=round(display_inference_ms, 1),
            )
        
        time.sleep(0)

    def _perform_step(self, training_mode: bool) -> None:
        started_at = time.perf_counter()
        done = False
        reason = ""
        self._ensure_runtime_consistency()
        state = self._current_observation()
        predicted_action, q_values, was_random, greedy_action = self._select_current_action(state, training_mode)
        if (
            training_mode
            and self.agent_kind == "dqn"
            and was_random
            and self.game._danger_for_relative_action(predicted_action) >= 0.5
        ):
            predicted_action = greedy_action
            was_random = False
        if self.mode == "manual" and not self.training:
            absolute_direction = self.pending_manual_direction
            visible_action = predicted_action
        else:
            absolute_direction = relative_to_absolute(self.game.direction, predicted_action)
            visible_action = predicted_action

        pre_step_snapshot = self._build_recent_step_snapshot(
            absolute_direction=absolute_direction,
            action=visible_action,
            action_random=was_random,
            greedy_action=greedy_action,
        )

        _, reward, done, info = self.game.step(absolute_direction)
        reason = str(info["reason"])
        next_state = self._current_observation()

        self.last_state = next_state
        self.last_q_values = q_values
        self.live_reward = reward
        self.episode_reward += reward
        self.last_action = visible_action
        self.last_action_random = was_random
        self.last_greedy_action = greedy_action
        self.last_reason = reason
        self.last_penalty_reasons = [str(item) for item in info.get("penalty_reasons", [])]
        train_reward = reward
        step_number = int(info.get("steps", self.game.step_count))
        if step_number == 1 and self.step_review_entries:
            self.step_review_entries.clear()
        self.step_review_entries.append({
            "step": step_number,
            "reward": float(reward),
            "train_reward": float(train_reward),
            "action": int(visible_action),
            "action_label": MODEL_ACTION_LABELS[visible_action] if 0 <= visible_action < len(MODEL_ACTION_LABELS) else "-",
            "reason": self.last_reason,
            "penalties": list(self.last_penalty_reasons),
            "random": bool(was_random),
            "apple_opportunity": info.get("apple_opportunity"),
        })
        self._record_decision_stat("choices", MODEL_ACTION_LABELS[visible_action])
        self._record_decision_stat("steps", self.last_reason)
        pre_step_snapshot["outcome_reason"] = self.last_reason
        pre_step_snapshot["penalty_reasons"] = list(self.last_penalty_reasons)
        self.recent_step_snapshots.append(pre_step_snapshot)

        if training_mode and self.agent_kind == "dqn":
            priority_scale = self._training_priority_scale(
                done=done,
                reason=self.last_reason,
                penalty_reasons=self.last_penalty_reasons,
            )
            self.agent.remember(
                state,
                visible_action,
                reward,
                next_state,
                done,
                priority_scale=priority_scale,
            )
            for _ in range(self._dqn_extra_replay_count(done=done, reason=self.last_reason, penalty_reasons=self.last_penalty_reasons)):
                self.agent.remember(
                    state,
                    visible_action,
                    reward,
                    next_state,
                    done,
                    priority_scale=priority_scale,
                )
            self.current_episode_transitions.append(
                (state.copy(), visible_action, reward, next_state.copy(), done, priority_scale)
            )
            if done and self.last_reason in DEATH_REASON_ORDER and self.last_reason != "victory":
                self._remember_dqn_death_context()
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
        self.last_completed_step_review_entries = list(self.step_review_entries)
        self.episode_counter += 1
        previous_best_score = self.best_score
        is_new_best = self.game.score > self.best_score
        self.best_score = max(self.best_score, self.game.score)
        self._sync_board_best_score()
        if self.last_completed_step_review_entries:
            self.step_review_history.appendleft({
                "id": f"episode-{self.episode_counter}-{int(time.time() * 1000)}",
                "episode": self.episode_counter,
                "score": self.game.score,
                "steps": self.game.step_count,
                "reason": reason,
                "entries": list(self.last_completed_step_review_entries),
            })
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
            trap_after_apple_count=self.game.trap_after_apple_count,
            avoidable_collision_count=self.game.avoidable_collision_count,
            max_steps_without_food=self.game.max_steps_without_food,
        ) if training_mode else float(getattr(self.agent, "last_fitness", 0.0))
        
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
            loss=float(getattr(self.agent, "last_loss", 0.0)) if self.agent_kind == "dqn" else 0.0,
        ))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)

        if training_mode and self.agent_kind == "dqn":
            elite_priority = self._episode_elite_priority(previous_best_score, reason)
            if elite_priority > 0:
                self.agent.remember_elite_episode(
                    self._compact_elite_transitions(self.current_episode_transitions),
                    min_priority=elite_priority,
                )
        self.current_episode_transitions.clear()

        self.last_outcome_reason = reason
        self.game.reset()
        self.pending_manual_direction = self.game.direction
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
        self.pending_manual_direction = self.game.direction
        self.live_reward = 0.0
        self.episode_reward = 0.0
        self.last_action = 0
        self.last_action_random = False
        self.last_greedy_action = 0
        self.last_reason = "new_game"
        self.last_outcome_reason = "new_game"
        self.last_penalty_reasons = []
        self.recent_step_snapshots.clear()
        self.step_review_entries.clear()
        self.current_episode_transitions.clear()
        self.last_state = self._current_observation()
        _, self.last_q_values, _, self.last_greedy_action = self._select_current_action(self.last_state, False)

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
            "move_safety": self.game.relative_move_safety(),
        }

    def _reset_decision_stats(self) -> None:
        self.decision_stats = make_decision_stats()

    def _reset_death_stats(self) -> None:
        self.decision_stats["deaths"] = make_decision_stat_bucket(DEATH_REASON_ORDER)

    def _clear_runtime_review_state(self) -> None:
        self._reset_death_stats()
        self.last_death_snapshot = None
        self.recent_step_snapshots.clear()
        self.step_review_entries.clear()
        self.last_completed_step_review_entries.clear()
        self.step_review_history.clear()
        self.last_outcome_reason = "new_game"
        self.last_penalty_reasons = []

    def _record_decision_stat(self, section: str, key: str) -> None:
        bucket = self.decision_stats.setdefault(section, make_decision_stat_bucket(tuple()))
        all_counts = bucket.setdefault("all", {})
        recent = bucket.setdefault("recent", deque(maxlen=DECISION_STATS_WINDOW))
        if isinstance(recent, list):
            recent = deque(recent[-DECISION_STATS_WINDOW:], maxlen=DECISION_STATS_WINDOW)
            bucket["recent"] = recent
        all_counts[key] = int(all_counts.get(key, 0)) + 1
        recent.append(key)

    def _training_priority_scale(self, *, done: bool, reason: str, penalty_reasons: list[str]) -> float:
        if reason == "apple":
            return 2.0
        if reason == "victory":
            return 6.0
        if done and reason in {"wall_collision", "body_collision"}:
            return 4.0 if reason == "wall_collision" else 5.0
        if done and reason.startswith("starvation"):
            return 3.0
        if "trap_after_apple" in penalty_reasons:
            return 2.5
        return 1.0

    def _dqn_extra_replay_count(self, *, done: bool, reason: str, penalty_reasons: list[str]) -> int:
        if self.agent_kind != "dqn":
            return 0
        if done and reason in DEATH_REASON_ORDER and reason != "victory":
            return DQN_DEATH_REPLAY_EXTRA_COPIES
        return 0

    def _remember_dqn_death_context(self) -> None:
        if self.agent_kind != "dqn" or not self.current_episode_transitions:
            return
        transitions_list = list(self.current_episode_transitions)
        context = transitions_list[-(DQN_PRE_DEATH_CONTEXT_REPLAYS + 1):-1]
        for state, action, reward, next_state, done, priority_scale in context:
            self.agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
                priority_scale=max(priority_scale, 4.0),
            )

    def _compact_elite_transitions(
        self,
        transitions: Deque[tuple[np.ndarray, int, float, np.ndarray, bool, float]],
    ) -> list[tuple[np.ndarray, int, float, np.ndarray, bool, float]]:
        transitions_list = list(transitions)
        if len(transitions_list) <= ELITE_EPISODE_MAX_TRANSITIONS:
            return transitions_list
        important = [item for item in transitions_list if item[4] or item[2] > 0.5 or item[5] > 1.0]
        stride = max(1, len(transitions_list) // ELITE_EPISODE_MAX_TRANSITIONS)
        sampled = list(transitions_list[::stride])
        merged: list[tuple[np.ndarray, int, float, np.ndarray, bool, float]] = []
        seen: set[int] = set()
        for item in important + sampled:
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
            if len(merged) >= ELITE_EPISODE_MAX_TRANSITIONS:
                break
        return merged

    def _episode_elite_priority(self, previous_best_score: int, reason: str) -> float:
        score = self.game.score
        max_score = max(1, self.game.board_size * self.game.board_size - 3)
        if reason == "victory" or score >= max_score:
            return 6.0
        if score >= max(2, previous_best_score):
            return 3.0
        if previous_best_score >= 3 and score >= max(2, int(previous_best_score * 0.8)):
            return 2.4
        if score >= 1 and self.game.step_count >= max(80, self.game.board_size * 4):
            return 1.6
        return 0.0

    def _reward_rules_payload(self) -> dict[str, Any]:
        # Динамический каталог из backend_config.py
        bonuses: list[dict[str, Any]] = []
        for reward_id, item in REWARD_CATALOG.items():
            value = item.get("value")
            value_text = f"+{value:.1f}" if isinstance(value, (int, float)) else str(value)
            bonuses.append({
                "id": reward_id,
                "label": item.get("label", reward_id),
                "short_label": item.get("short_label", reward_id),
                "value": value_text,
                "active": True,
            })
        penalties: list[dict[str, Any]] = []
        for penalty_id, item in PENALTY_CATALOG.items():
            if item.get("show_in_ui", True) is False:
                continue
            value = item.get("value")
            value_text = f"{value:.1f}" if isinstance(value, (int, float)) else str(value)
            penalties.append({
                "id": penalty_id,
                "label": item.get("label", penalty_id),
                "short_label": item.get("short_label", penalty_id),
                "value": value_text,
                "active": True,
            })
        return {
            "bonuses": bonuses,
            "penalties": penalties,
            "note": " ",
        }

    def _agent_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "agent_kind": self.agent_kind,
            "device_mode": self.device_mode,
            "game": self.game.export_state(),
            "running": self.running,
            "training": False,
            "mode": self.mode,
            "speed": self.speed,
            "pending_manual_direction": self.pending_manual_direction,
            "live_reward": self.live_reward,
            "episode_reward": self.episode_reward,
            "fitness_ema": self.fitness_ema,
            "generation": getattr(self.agent, "generation", 0),
            "current_individual": getattr(self.agent, "current_index", 0) + 1,
            "population_size": getattr(self.agent, "population_size", 1),
            "victory_archive_size": int(getattr(self.agent, "victory_archive_size", 0)),
            "best_fitness": self.agent.best_fitness,
            "best_score": self.best_score,
            "board_best_scores": {str(key): int(value) for key, value in self.board_best_scores.items()},
            "episode_counter": self.episode_counter,
            "last_action": self.last_action,
            "last_action_random": self.last_action_random,
            "last_greedy_action": self.last_greedy_action,
            "last_reason": self.last_reason,
            "last_outcome_reason": self.last_outcome_reason,
            "last_penalty_reasons": list(self.last_penalty_reasons),
        }

    def _active_context_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "agent_kind": self.agent_kind,
            "mode": self.mode,
            "board_size": int(self.game.board_size),
            "smart_model": self.smart_agent.model_path.name if self.smart_agent.model_path else "",
            "speed": self.speed,
        }

    def _save_active_context(self) -> None:
        if not self.persistence_enabled:
            return
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = ACTIVE_CONTEXT_PATH.with_suffix(".tmp")
        payload = json.dumps(self._active_context_payload(), ensure_ascii=False, separators=(",", ":"))
        for attempt in range(5):
            try:
                tmp_path.write_text(payload, encoding="utf-8")
                tmp_path.replace(ACTIVE_CONTEXT_PATH)
                break
            except OSError:
                if attempt < 4:
                    time.sleep(0.05)

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
        elif mode in {"manual", "ai"}:
            self.mode = mode

        try:
            self.speed = int(payload.get("speed", self.speed))
        except (TypeError, ValueError):
            pass

        self.agent = self._create_native_agent(self.agent_kind)
        self._reset_training_runner(self.game.board_size)
        self._sync_network_source()
        return True

    def _agent_state_payload(self) -> dict[str, object]:
        if self.agent_kind == "dqn":
            return {
                "epsilon": float(getattr(self.agent, "epsilon", 0.0)),
                "total_updates": int(getattr(self.agent, "total_updates", 0)),
                "last_loss": float(getattr(self.agent, "last_loss", 0.0)),
                "best_score": int(getattr(self.agent, "best_score", 0)),
                "best_fitness": float(getattr(self.agent, "best_fitness", 0.0)),
                "last_fitness": float(getattr(self.agent, "last_fitness", 0.0)),
            }
        return {
            "generation": int(getattr(self.agent, "generation", 1)),
            "current_index": int(getattr(self.agent, "current_index", 0)),
            "best_fitness": float(getattr(self.agent, "best_fitness", 0.0)),
            "best_score": int(getattr(self.agent, "best_score", 0)),
            "last_fitness": float(getattr(self.agent, "last_fitness", 0.0)),
            "last_generation_average": float(getattr(self.agent, "last_generation_average", 0.0)),
            "completed_evaluations": int(getattr(self.agent, "completed_evaluations", 0)),
            "settings": self.agent.settings_payload(),
        }

    def _load_agent_state_payload(self, payload: dict[str, Any] | None) -> None:
        payload = payload if isinstance(payload, dict) else {}
        if self.agent_kind == "dqn":
            self.agent.epsilon = float(payload.get("epsilon", getattr(self.agent, "epsilon", 1.0)))
            self.agent.total_updates = int(payload.get("total_updates", getattr(self.agent, "total_updates", 0)))
            self.agent.last_loss = float(payload.get("last_loss", getattr(self.agent, "last_loss", 0.0)))
            self.agent.best_score = int(payload.get("best_score", getattr(self.agent, "best_score", 0)))
            self.agent.best_fitness = float(payload.get("best_fitness", getattr(self.agent, "best_fitness", float("-inf"))))
            self.agent.last_fitness = float(payload.get("last_fitness", getattr(self.agent, "last_fitness", 0.0)))
            return

        self.agent.generation = max(1, int(payload.get("generation", getattr(self.agent, "generation", 1))))
        self.agent.current_index = max(0, min(int(payload.get("current_index", 0)), self.agent.population_size - 1))
        self.agent.best_fitness = float(payload.get("best_fitness", getattr(self.agent, "best_fitness", float("-inf"))))
        self.agent.best_score = int(payload.get("best_score", getattr(self.agent, "best_score", 0)))
        self.agent.last_fitness = float(payload.get("last_fitness", getattr(self.agent, "last_fitness", 0.0)))
        self.agent.last_generation_average = float(payload.get("last_generation_average", getattr(self.agent, "last_generation_average", 0.0)))
        self.agent.completed_evaluations = int(payload.get("completed_evaluations", getattr(self.agent, "completed_evaluations", 0)))
        self.agent._load_current_genome()

    def _load_runtime_payload(self, payload: dict[str, Any] | None) -> None:
        payload = payload if isinstance(payload, dict) else {}
        game_payload = payload.get("game")
        if isinstance(game_payload, dict):
            self.last_state = self.game.load_state(game_payload)
            self.last_state = self._current_observation()
        self.running = bool(payload.get("running", self.running))
        self.training = False
        self.mode = str(payload.get("mode", self.mode))
        if self.mode == "smart" and not self.smart_agent.loaded:
            self.mode = "manual"
        self.speed = int(payload.get("speed", self.speed))
        self.pending_manual_direction = int(payload.get("pending_manual_direction", self.game.direction))
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
        return {
            "v": 1,
            "ep": self.episode_counter,
            "bs": self.best_score,
            "bbs": {str(k): v for k, v in self.board_best_scores.items()},
            "fit": self.fitness_ema,
            "hist": [{"e": i.episode, "s": i.score, "st": i.steps, "r": i.reason, "f": i.fitness} for i in self.history],
            "ds": {
                "c": {"a": self.decision_stats["choices"]["all"], "r": list(self.decision_stats["choices"]["recent"])},
                "s": {"a": self.decision_stats["steps"]["all"], "r": list(self.decision_stats["steps"]["recent"])},
                "d": {"a": self.decision_stats["deaths"]["all"], "r": list(self.decision_stats["deaths"]["recent"])},
            }
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
            if isinstance(item, dict):
                try:
                    self.history.append(EpisodeStats(
                        episode=item["e"], score=item["s"], reward=0.0, mutation=0.0,
                        fitness=item["f"], steps=item["st"], best_score=self.best_score,
                        reason=item["r"], loss=0.0
                    ))
                except (KeyError, TypeError):
                    continue

        ds = payload.get("ds", {})
        self._reset_decision_stats()
        if isinstance(ds, dict):
            for section, code in {"choices": "c", "steps": "s", "deaths": "d"}.items():
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
                        (str(i) for i in recent_payload[-DECISION_STATS_WINDOW:]),
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
        self._last_save_at = time.perf_counter()
        self._next_save_at = 0.0

    def _capture_save_payloads(self) -> tuple[Path, dict[str, object], str, dict[str, object]] | None:
        if not self.persistence_enabled:
            return None
        checkpoint_path = self._checkpoint_path()
        checkpoint_payload = self.agent.export_checkpoint(self._agent_metadata())
        # Вшиваем подпись архитектуры в чекпоинт
        checkpoint_payload["arch_signature"] = ARCH_SIGNATURE
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
        store["active_context"] = self._active_context_payload()
        models = store.setdefault("models", {})
        if isinstance(models, dict):
            if self.mode != "smart":
                compact_model_stats(models, checkpoint_key, self.agent_kind)
            models[checkpoint_key] = aux_payload
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
            if not self._pending_save:
                return False
            now = time.perf_counter()
            if self._next_save_at > 0 and now < self._next_save_at:
                return False
            self._pending_save = False
            self._next_save_at = 0.0
            payloads = self._capture_save_payloads()
        if payloads is None:
            return False
        self._write_save_payloads(*payloads)
        self._last_save_at = time.perf_counter()
        return True

    def _load_state(self, *, force_cpu: bool, preferred_device_mode: str | None = None) -> bool:
        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            return False

        try:
            payload = torch.load(checkpoint_path, map_location="cpu")
            
            # Проверяем совместимость архитектуры по сигнатуре
            stored_signature = payload.get("arch_signature")
            if stored_signature != ARCH_SIGNATURE:
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
            self.mode = "ai"
            self.training = False
            self.running = False
            self._sync_network_source()
            if was_smart_mode:
                current_board_size = self.game.board_size
                self.game = SnakeEnv(board_size=current_board_size)
                if not self._load_active_aux_stats():
                    self._reset_aux_stats()
                self.fitness_ema = 0.0
                self.current_episode_transitions.clear()
                self._reset_environment_state()
            return True, "ok"

        current_board_size = self.game.board_size
        self._sync_board_best_score()
        self._store_active_agent_slot()
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
                    loaded = self._load_agent_checkpoint_into(self.agent)
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
                    updated_at=time.monotonic(),
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
        self._reset_environment_state()
        self.current_episode_transitions.clear()
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
        global_signal_labels = list(GLOBAL_SIGNAL_LABELS)
        
        self._ensure_runtime_consistency()
        locked = self.agent_lock.acquire(timeout=0.02 if self.running else -1)
        if not locked:
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
            
            payload = {
                "global_signals": {
                    "labels": global_signal_labels,
                    "values": self.last_state.tolist(),
                },
                "hidden1": [0.0] * HIDDEN_LAYER_1,
                "hidden2": [0.0] * HIDDEN_LAYER_2,
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
            snapshot = self.agent.snapshot(self.last_state)
            input_values = np.asarray(snapshot.get("input", np.zeros(GLOBAL_SIGNAL_COUNT)), dtype=np.float32)
            hidden1_values = np.asarray(snapshot.get("hidden1", np.zeros(HIDDEN_LAYER_1)), dtype=np.float32)
            hidden2_values = np.asarray(snapshot.get("hidden2", np.zeros(HIDDEN_LAYER_2)), dtype=np.float32)
            output_values = np.asarray(snapshot.get("output", np.zeros(len(MODEL_ACTION_LABELS))), dtype=np.float32)
            value_values = np.asarray(snapshot.get("value", np.array([], dtype=np.float32)), dtype=np.float32)

            net = self.agent.online_net
            input_hidden1_connections = []
            hidden1_hidden2_connections = []
            hidden2_output_connections = []
            hidden2_value_connections = []

            try:
                # === ИСПРАВЛЕННЫЙ БЛОК: корректное сопоставление весов и активаций ===
                
                # input -> hidden1: промежуточный слой 64 не выводится в UI, поэтому используем
                # эффективную матрицу W_h1 @ W_in размером (HIDDEN_LAYER_1, input_size)
                if hasattr(net, 'input_layer') and hasattr(net, 'hidden1_layer'):
                    w_in = net.input_layer.weight.detach()
                    w_h1 = net.hidden1_layer.weight.detach()
                    w_in_h1 = (w_h1 @ w_in).cpu().numpy()
                    input_hidden1_connections = build_connection_payload(
                        input_values, hidden1_values, w_in_h1,
                        len(input_values), len(hidden1_values), 48
                    )
                
                # hidden1 -> hidden2: используется вес именно hidden2_layer (H2, H1)
                if hasattr(net, 'hidden1_layer') and hasattr(net, 'hidden2_layer'):
                    w_h1_h2 = net.hidden2_layer.weight.detach().cpu().numpy()
                    hidden1_hidden2_connections = build_connection_payload(
                        hidden1_values, hidden2_values, w_h1_h2,
                        len(hidden1_values), len(hidden2_values), 90
                    )
                 
                w_out = None
                if hasattr(net, 'advantage_head'):
                    w_out = net.advantage_head.weight.detach().cpu().numpy()
                elif hasattr(net, 'output'):
                    w_out = net.output.weight.detach().cpu().numpy()
                
                if w_out is not None:
                    hidden2_output_connections = build_connection_payload(
                        hidden2_values, output_values, w_out, 
                        len(hidden2_values), len(output_values), 30
                    )
                
                if hasattr(net, 'value_head'):
                    w_val = net.value_head.weight.detach().cpu().numpy()
                    hidden2_value_connections = build_connection_payload(
                        hidden2_values, value_values, w_val,
                        len(hidden2_values), len(value_values), 16
                    )
            except Exception as e:
                log_perf("network_payload_weight_error", 0.0, error=str(e))

            payload = {
                "global_signals": {
                    "labels": global_signal_labels,
                    "values": input_values.tolist(),
                },
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
                "reward_rules", "step_review", "step_review_source", "step_review_history",
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
        replay_buffer = getattr(self.agent, "replay_buffer", None)
        buffer_size = len(replay_buffer) if replay_buffer is not None else 0
        hunger_budget = hunger_budget_for(self.game.score, self.game.board_size)
        hunger_steps = max(0, int(self.game.steps_without_food))
        hunger_ratio = min(1.0, float(hunger_steps) / max(1.0, hunger_budget))
        return {
            "episode": self.episode_counter,
            "best_score": self.best_score,
            "average_score_100": average_score_100,
            "average_score_window": len(recent_scores),
            "reward": self.live_reward,
            "fitness": self.fitness_ema,
            "mutation": 0.0 if self.mode == "smart" else getattr(self.agent, "mutation_rate", 0.0),
            "hunger_steps": hunger_steps,
            "hunger_budget": hunger_budget,
            "hunger_ratio": hunger_ratio,
            "max_hunger_steps": max(0, int(self.game.max_steps_without_food)),
            "generation": 0 if self.mode == "smart" else getattr(self.agent, "generation", 0),
            "individual": 0 if self.mode == "smart" else getattr(self.agent, "current_index", 0) + 1,
            "population": 0 if self.mode == "smart" else getattr(self.agent, "population_size", 1),
            "victory_archive": 0 if self.mode == "smart" else int(getattr(self.agent, "victory_archive_size", 0)),
            "best_fitness": best_fitness,
            "avg_fitness": 0.0 if self.mode == "smart" else getattr(self.agent, "last_generation_average", 0.0),
            "alive": 1 if self.training and is_dqn else (0 if not self.training else self.training_runner.alive_count()),
            "completed": 0 if is_dqn or not self.training else self.training_runner.completed_count,
            "leader": 1 if self.training and is_dqn else (0 if not self.training else self.training_display_index + 1),
            "epsilon": 0.0 if not is_dqn else float(getattr(self.agent, "epsilon", 0.0)),
            "buffer": buffer_size if is_dqn else 0,
            "loss": 0.0 if not is_dqn else float(getattr(self.agent, "last_loss", 0.0)),
            "updates": 0 if not is_dqn else int(getattr(self.agent, "total_updates", 0)),
            "steps": self.game.step_count,
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
        }

    def _update_frame_cache(self) -> None:
        self.frame_id += 1
        self._snapshot_cache.clear()
        self._frame_cache = self._frame_payload_unlocked()
        self._last_frame_payload = dict(self._frame_cache)

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
            self._ensure_runtime_consistency()
            self._sync_network_source()
            payload = {
                "device_kind": self.device_kind,
                "device_label": self.device_label,
                "device_details": self.device_details,
                "device_mode": self.device_mode,
                "available_devices": self.available_devices,
                "model_input_size": int(self.last_state.size),
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
                "last_action": self.last_action,
                "last_action_random": self.last_action_random,
                "last_greedy_action": self.last_greedy_action,
                "last_reason": self.last_reason,
                "last_outcome_reason": self.last_outcome_reason,
                "last_penalty_reasons": list(self.last_penalty_reasons),
                "persistence_enabled": self.persistence_enabled,
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
            }
            if include_details:
                terminal_reasons = {"wall_collision", "body_collision", "victory"}
                terminal_reasons.update(reason for reason in STEP_REASON_ORDER if reason.startswith("starvation"))
                step_review_source = "current"
                step_review_payload: list[dict[str, Any]] = list(self.step_review_entries)
                if (
                    self.training
                    and self.running
                    and not step_review_payload
                    and self.last_outcome_reason in terminal_reasons
                    and self.last_completed_step_review_entries
                ):
                    step_review_source = "last_death"
                    step_review_payload = list(self.last_completed_step_review_entries)
                step_review_payload = step_review_payload[-100:]
                payload.update({
                    "reward_rules": self._reward_rules_payload(),
                    "step_review": step_review_payload,
                    "step_review_source": step_review_source,
                    "step_review_history": list(self.step_review_history),
                    "observation": self.last_state.tolist(),
                    "global_signals": {
                        "labels": list(GLOBAL_SIGNAL_LABELS),
                        "values": self.last_state.tolist(),
                    },
                    "last_death_snapshot": self.last_death_snapshot,
                    "decision_stats": export_decision_stats(self.decision_stats),
                    "history": [asdict(item) for item in self.history],
                })
            self._snapshot_cache[(include_network, include_details)] = payload
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if elapsed_ms >= SLOW_API_MS:
                log_perf("slow_state_snapshot", elapsed_ms, cache=cache_used, network=include_network, details=include_details, running=self.running, training=self.training)
            return payload
        finally:
            if acquired:
                self.lock.release()

    def _command_state_response(self, command: str) -> dict[str, Any] | None:
        if command == "manual_action":
            return None
        light_state_commands = {
            "toggle_running", "toggle_training", "set_agent_kind", "set_mode",
            "set_smart_model", "set_board_size", "cycle_board_size",
            "set_speed", "set_device", "reset_agent",
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
        acquired = self.lock.acquire(timeout=0.15 if (quick_training_toggle or quick_running_stop) else -1)
        if not acquired:
            if quick_training_toggle:
                if self.mode == "smart":
                    return {"ok": False, "error": "Zmey-smart training is unavailable"}
                desired_training = bool(value) if isinstance(value, bool) else False
                self.training = desired_training
                if desired_training:
                    self.mode = "ai"
                    self.running = True
                return {"ok": True, "state": None, "busy": True}
            if quick_running_stop:
                self.running = False
                self._save_active_context()
                return {"ok": True, "state": None, "busy": True}
            return {"ok": False, "error": "Backend busy"}

        try:
            if command == "toggle_running":
                self.running = not self.running
            elif command == "set_mode":
                # Перезагружаем модель если путь есть, но модель не загружена
                if value == "smart" and not self.smart_agent.loaded and self.smart_agent.model_path:
                    self.smart_agent.set_model_path(self.smart_agent.model_path)
                
                if value == "smart" and not self.smart_agent.loaded:
                    # Разрешаем выбор режима smart даже без модели для демонстрации
                    # Но устанавливаем флаг что модель недоступна
                    pass  # Продолжаем вместо возврата ошибки
                
                if value in {"manual", "ai", "smart"}:
                    previous_mode = self.mode
                    switching_agent_mode = {previous_mode, str(value)} <= {"ai", "smart"} and previous_mode != str(value)
                    self.mode = str(value)
                    self.training = False
                    if value == "smart":
                        self.running = False
                        self.game = SnakeEnv(board_size=self.game.board_size)
                        if not self._load_active_aux_stats():
                            self._reset_aux_stats()
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
                ok, message = self._set_agent_kind(value)
                if not ok:
                    return {"ok": False, "error": message}
            elif command == "set_smart_model":
                if not isinstance(value, str):
                    return {"ok": False, "error": "Invalid smart model"}
                ok, message = self._set_smart_model(value)
                if not ok:
                    return {"ok": False, "error": message}
            elif command == "toggle_training":
                if self.mode == "smart":
                    return {"ok": False, "error": "Zmey-smart training is unavailable"}
                self.training = bool(value) if isinstance(value, bool) else not self.training
                self.mode = "ai"
                if self.training:
                    self.running = True
            elif command == "set_board_size":
                if isinstance(value, int) and value in BOARD_SIZES:
                    ok, message = self._set_board_size(value)
                    if not ok:
                        return {"ok": False, "error": message}
            elif command == "cycle_board_size":
                self._cycle_board_size()
            elif command == "set_speed":
                if isinstance(value, int):
                    self.speed = max(1, min(16, value))
                    self._save_active_context()
            elif command == "manual_action":
                if isinstance(value, int) and 0 <= value < 4:
                    self.pending_manual_direction = value
            elif command == "set_device":
                if not isinstance(value, str):
                    return {"ok": False, "error": "Invalid device value"}
                ok, message = self._set_device_mode(value)
                if not ok:
                    return {"ok": False, "error": message}
            elif command == "reset_agent":
                if self.mode == "smart":
                    return {"ok": False, "error": "Reset is unavailable for Zmey-smart"}
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
                self.mode = "manual"
                self.running = False
                self._reset_environment_state()
                self.current_episode_transitions.clear()
                self._save_state()
            elif command in {"restart_server", "shutdown_server"}:
                self.running = False
                self.training = False
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
                "restart_server", "shutdown_server",
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
