from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np

from backend_config import (
    AVERAGE_SCORE_WINDOW_RATIO,
    DEATH_REASON_ORDER,
    DECISION_STATS_WINDOW,
    MODEL_ACTION_LABELS,
    PENALTY_CATALOG,
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


def build_reward_rules_payload() -> dict[str, Any]:
    bonuses: list[dict[str, Any]] = []
    for reward_id, item in REWARD_CATALOG.items():
        bonuses.append({
            "id": reward_id,
            "label": item.get("label", reward_id),
            "short_label": item.get("short_label", reward_id),
            "value": _format_rule_value(item.get("value"), prefix="+"),
            "active": True,
        })
    penalties: list[dict[str, Any]] = []
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
    return {
        "bonuses": bonuses,
        "penalties": penalties,
        "note": " ",
    }
