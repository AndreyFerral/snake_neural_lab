from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np

from backend_config import (
    APPLE_REWARD,
    AVERAGE_SCORE_WINDOW_RATIO,
    DEATH_PENALTY,
    DEATH_REASON_ORDER,
    DECISION_STATS_WINDOW,
    ENDGAME_MERGE_BONUS,
    MODEL_ACTION_LABELS,
    PENALTY_CATALOG,
    PPO_CRITICAL_HUNGER_PENALTY,
    PPO_FOOD_STEP_AWAY_PENALTY,
    PPO_FOOD_STEP_CLOSER_BONUS,
    PPO_VICTORY_EXTRA_BONUS,
    REWARD_CATALOG,
    STEP_REASON_ORDER,
)

CHOICE_TO_SYMBOL = {
    "Вперед": "^",
    "Влево": "<",
    "Вправо": ">",
}
SYMBOL_TO_CHOICE = {v: k for k, v in CHOICE_TO_SYMBOL.items()}

STEP_TO_SYMBOL = {
    "step": "s",
    "apple": "a",
    "looping": "l",
    "wall_collision": "w",
    "body_collision": "b",
    "victory": "v",
}
SYMBOL_TO_STEP = {v: k for k, v in STEP_TO_SYMBOL.items()}

DEATH_TO_SYMBOL = {
    "victory": "v",
    "wall_collision": "w",
    "body_collision": "b",
    "starvation": "s",
}
SYMBOL_TO_DEATH = {v: k for k, v in DEATH_TO_SYMBOL.items()}


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "score": self.score,
            "reward": self.reward,
            "mutation": self.mutation,
            "fitness": self.fitness,
            "steps": self.steps,
            "best_score": self.best_score,
            "reason": self.reason,
            "loss": self.loss,
        }


@dataclass
class AgentSlot:
    agent: Any | None = None
    ready: bool = False
    loading: bool = False
    error: str = ""
    loaded_checkpoint: bool = False


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


def record_decision_stat(decision_stats: dict[str, dict[str, Any]], section: str, key: str) -> None:
    bucket = decision_stats.setdefault(section, make_decision_stat_bucket(tuple()))
    all_counts = bucket.setdefault("all", {})
    recent = bucket.setdefault("recent", deque(maxlen=DECISION_STATS_WINDOW))
    if isinstance(recent, list):
        recent = deque(recent[-DECISION_STATS_WINDOW:], maxlen=DECISION_STATS_WINDOW)
        bucket["recent"] = recent
    all_counts[key] = int(all_counts.get(key, 0)) + 1
    recent.append(key)


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


def top_indices(values: np.ndarray, count: int) -> list[int]:
    if values.size == 0:
        return []
    return np.argsort(np.abs(values))[-count:].tolist()


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

    max_influence = max(c[0] for c in candidates)
    if max_influence > 0.0:
        candidates = [((inf / max_influence) ** 1.4 * 0.5, fi, ti, w) for inf, fi, ti, w in candidates]

    payload: list[dict[str, float | int]] = []
    ranked = sorted(candidates, reverse=True)
    for influence, from_index, to_index, weight in ranked[:limit]:
        if influence <= 0.05:
            continue
        payload.append({
            "from_index": from_index,
            "to_index": to_index,
            "weight": weight,
            "influence": influence,
        })

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


def _format_rule_value(value: Any, prefix: str = "") -> str:
    if isinstance(value, (int, float)):
        s = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{prefix}{s}"
    return str(value)


def build_reward_rules_payload(agent_kind: str = "dqn", mode: str = "ai") -> dict[str, Any]:
    is_smart = str(mode) == "smart"
    is_ppo = (not is_smart) and str(agent_kind) == "ppo"

    base_victory = float(REWARD_CATALOG.get("victory", {}).get("value", 10.0))
    bonuses: list[dict[str, Any]] = [
        {
            "id": "apple",
            "label": "Яблоко",
            "short_label": "Яблоко",
            "value": _format_rule_value(APPLE_REWARD, prefix="+"),
            "active": True,
        },
        {
            "id": "victory",
            "label": "Победа",
            "short_label": "Победа",
            "value": _format_rule_value(base_victory + (PPO_VICTORY_EXTRA_BONUS if is_ppo else 0.0), prefix="+"),
            "formula": "+10 среда +20 PPO бонус (+4 посл. яблоко)" if is_ppo else "+10 среда (+4 посл. яблоко)",
            "active": True,
        },
    ]
    if not is_smart:
        bonuses.append({
            "id": "pocket_merge",
            "label": "Слияние",
            "short_label": "Слияние",
            "value": _format_rule_value(ENDGAME_MERGE_BONUS, prefix="+"),
            "formula": "Объединение изолированных карманов в эндшпиле",
            "active": True,
        })
    if is_ppo:
        bonuses.append({
            "id": "food_closer",
            "label": "К яблоку",
            "short_label": "К яблоку",
            "value": _format_rule_value(PPO_FOOD_STEP_CLOSER_BONUS, prefix="+"),
            "formula": "Шаг к достижимому яблоку до эндшпиля",
            "active": True,
        })

    penalties: list[dict[str, Any]] = []
    if is_smart:
        penalties.append({
            "id": "death",
            "label": "Смерть",
            "short_label": "Смерть",
            "value": _format_rule_value(-DEATH_PENALTY),
            "active": True,
        })
    else:
        for penalty_id, item in PENALTY_CATALOG.items():
            if item.get("show_in_ui", True) is False:
                continue
            penalties.append({
                "id": penalty_id,
                "label": item.get("label", penalty_id),
                "short_label": item.get("short_label", penalty_id),
                "value": _format_rule_value(item.get("value")),
                "active": True,
            })
        if is_ppo:
            penalties.append({
                "id": "hunger_critical",
                "label": "Крит. голод",
                "short_label": "Голод",
                "value": _format_rule_value(-PPO_CRITICAL_HUNGER_PENALTY),
                "formula": "Штраф за каждый шаг при голоде >= 85% лимита",
                "active": True,
            })
            penalties.append({
                "id": "food_away",
                "label": "От яблока",
                "short_label": "От яблока",
                "value": _format_rule_value(-PPO_FOOD_STEP_AWAY_PENALTY),
                "formula": "Шаг от достижимого яблока до эндшпиля",
                "active": True,
            })

    return {
        "bonuses": bonuses,
        "penalties": penalties,
        "note": "Только инференс .h5 — модель не обучается." if is_smart else "",
    }


# 40 входных признаков внешней Keras-модели Zmey-smart (соответствуют SmartH5Agent.build_state)
SMART_SIGNAL_LABELS: tuple[str, ...] = (
    "sm_food_east",
    "sm_food_west",
    "sm_food_south",
    "sm_food_north",
    "sm_dir_up",
    "sm_dir_down",
    "sm_dir_left",
    "sm_dir_right",
    "sm_wall_east",
    "sm_wall_west",
    "sm_wall_south",
    "sm_wall_north",
    "sm_danger_fwd",
    "sm_danger_left",
    "sm_danger_right",
    "sm_free_fwd",
    "sm_free_left",
    "sm_free_right",
    "sm_free_back",
    "sm_food_dx",
    "sm_food_dy",
    "sm_food_fwd_proj",
    "sm_food_lat_proj",
    "sm_food_manh",
    "sm_snake_len",
    "sm_wall_margin",
    "sm_sim_fwd_dead",
    "sm_sim_fwd_food",
    "sm_sim_fwd_tail",
    "sm_sim_fwd_area",
    "sm_sim_left_dead",
    "sm_sim_left_food",
    "sm_sim_left_tail",
    "sm_sim_left_area",
    "sm_sim_right_dead",
    "sm_sim_right_food",
    "sm_sim_right_tail",
    "sm_sim_right_area",
    "sm_fill_ratio",
    "sm_long_snake",
)

SMART_SIGNAL_SHORT_LABELS: dict[str, str] = {
    "sm_food_east": "🍎→",
    "sm_food_west": "🍎←",
    "sm_food_south": "🍎↓",
    "sm_food_north": "🍎↑",
    "sm_dir_up": "🧭↑",
    "sm_dir_down": "🧭↓",
    "sm_dir_left": "🧭←",
    "sm_dir_right": "🧭→",
    "sm_wall_east": "🧱→",
    "sm_wall_west": "🧱←",
    "sm_wall_south": "🧱↓",
    "sm_wall_north": "🧱↑",
    "sm_danger_fwd": "⚠↑",
    "sm_danger_left": "⚠←",
    "sm_danger_right": "⚠→",
    "sm_free_fwd": "📏↑",
    "sm_free_left": "📏←",
    "sm_free_right": "📏→",
    "sm_free_back": "📏↓",
    "sm_food_dx": "🍎dX",
    "sm_food_dy": "🍎dY",
    "sm_food_fwd_proj": "🍎F",
    "sm_food_lat_proj": "🍎L",
    "sm_food_manh": "🍎📏",
    "sm_snake_len": "🐍#",
    "sm_wall_margin": "🧱📏",
    "sm_sim_fwd_dead": "💀↑",
    "sm_sim_fwd_food": "🎯↑",
    "sm_sim_fwd_tail": "🐍↑",
    "sm_sim_fwd_area": "📐↑",
    "sm_sim_left_dead": "💀←",
    "sm_sim_left_food": "🎯←",
    "sm_sim_left_tail": "🐍←",
    "sm_sim_left_area": "📐←",
    "sm_sim_right_dead": "💀→",
    "sm_sim_right_food": "🎯→",
    "sm_sim_right_tail": "🐍→",
    "sm_sim_right_area": "📐→",
    "sm_fill_ratio": "📊%",
    "sm_long_snake": "🐍+",
}

_SMART_WEIGHTS_CACHE: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = {}


def _extract_keras_dense_layers(model: Any) -> list[tuple[np.ndarray, np.ndarray, str]]:
    cache_key = id(model)
    cached = _SMART_WEIGHTS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    extracted: list[tuple[np.ndarray, np.ndarray, str]] = []
    for layer in getattr(model, "layers", []):
        weights = layer.get_weights() if hasattr(layer, "get_weights") else []
        if len(weights) >= 2 and getattr(weights[0], "ndim", 0) == 2:
            w = np.asarray(weights[0], dtype=np.float32)
            b = np.asarray(weights[1], dtype=np.float32)
            act_fn = getattr(layer, "activation", None)
            act_name = getattr(act_fn, "__name__", "linear") if act_fn is not None else "linear"
            extracted.append((w, b, str(act_name)))
    if len(_SMART_WEIGHTS_CACHE) > 8:
        _SMART_WEIGHTS_CACHE.clear()
    _SMART_WEIGHTS_CACHE[cache_key] = extracted
    return extracted


def _apply_numpy_activation(x: np.ndarray, act_name: str) -> np.ndarray:
    if act_name == "relu":
        return np.maximum(x, 0.0)
    if act_name == "tanh":
        return np.tanh(x)
    if act_name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    if act_name == "softmax":
        shifted = x - np.max(x)
        ex = np.exp(shifted)
        return ex / np.maximum(1e-8, np.sum(ex))
    return x


def build_smart_network_payload(smart_agent: Any, game: Any) -> dict[str, Any]:
    """Строит реальный граф Keras-модели Zmey-smart чистым NumPy без изменения smart_agent.py."""
    labels = list(SMART_SIGNAL_LABELS)
    empty_payload = {
        "global_signals": {"labels": labels, "values": [0.0] * len(labels)},
        "ego_map": None,
        "hidden1": [],
        "hidden2": [],
        "output": [0.0] * len(MODEL_ACTION_LABELS),
        "value": [],
        "connections": {
            "global_hidden1": [],
            "hidden1_hidden2": [],
            "hidden2_output": [],
            "hidden1_output": [],
            "hidden2_value": [],
        },
    }
    if not getattr(smart_agent, "loaded", False) or getattr(smart_agent, "model", None) is None:
        return empty_payload

    # Не сдвигаем lookahead_counter агента при визуализации графа
    saved_counter = int(getattr(smart_agent, "lookahead_counter", 0))
    try:
        x_in = np.asarray(smart_agent.build_state(game), dtype=np.float32).ravel()
    finally:
        smart_agent.lookahead_counter = saved_counter

    dense_layers = _extract_keras_dense_layers(smart_agent.model)
    if not dense_layers:
        empty_payload["global_signals"]["values"] = x_in.tolist()
        return empty_payload

    activations: list[np.ndarray] = [x_in]
    cur = x_in
    for w, b, act_name in dense_layers:
        cur = _apply_numpy_activation(cur @ w + b, act_name).astype(np.float32, copy=False)
        activations.append(cur)

    output_vals = activations[-1]
    hidden_count = max(0, len(dense_layers) - 1)
    h1_vals = activations[1] if hidden_count >= 1 else np.empty(0, dtype=np.float32)
    h2_vals = activations[2] if hidden_count >= 2 else np.empty(0, dtype=np.float32)

    conns_in_h1: list[dict[str, Any]] = []
    conns_h1_h2: list[dict[str, Any]] = []
    conns_h2_out: list[dict[str, Any]] = []
    conns_h1_out: list[dict[str, Any]] = []

    if hidden_count >= 1:
        w0 = dense_layers[0][0].T  # (H1, IN)
        conns_in_h1 = build_connection_payload(x_in, h1_vals, w0, len(x_in), min(16, len(h1_vals)), 56)
    if hidden_count == 1:
        w_out = dense_layers[1][0].T  # (OUT, H1)
        conns_h1_out = build_connection_payload(h1_vals, output_vals, w_out, min(16, len(h1_vals)), len(output_vals), 36)
    elif hidden_count >= 2:
        w1 = dense_layers[1][0].T  # (H2, H1)
        conns_h1_h2 = build_connection_payload(h1_vals, h2_vals, w1, min(14, len(h1_vals)), min(10, len(h2_vals)), 90)
        w_out = dense_layers[-1][0].T  # (OUT, H2)
        conns_h2_out = build_connection_payload(h2_vals, output_vals, w_out, min(10, len(h2_vals)), len(output_vals), 30)

    return {
        "global_signals": {"labels": labels[: len(x_in)], "values": x_in.tolist()},
        "ego_map": None,
        "hidden1": h1_vals.tolist(),
        "hidden2": h2_vals.tolist(),
        "output": output_vals.tolist(),
        "value": [],
        "connections": {
            "global_hidden1": conns_in_h1,
            "hidden1_hidden2": conns_h1_h2,
            "hidden2_output": conns_h2_out,
            "hidden1_output": conns_h1_out,
            "hidden2_value": [],
        },
    }
