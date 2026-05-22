from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend_config import GLOBAL_SIGNAL_LABELS

if TYPE_CHECKING:
    from snake_env import SnakeEnv

_LOG_PATH = Path(__file__).parent / ".runtime" / "debug_log.jsonl"
_episode_buffer: list[dict[str, Any]] = []


def start_episode() -> None:
    _episode_buffer.clear()


def record_step(
    env: SnakeEnv,
    observation: np.ndarray,
    reward: float,
    info: dict[str, Any],
    *,
    action: int | None = None,
    q_values: np.ndarray | None = None,
) -> None:
    step = info.get("steps", env.step_count)
    score = info.get("score", env.score)
    reason = str(info.get("reason", ""))
    penalties: list[str] = [str(p) for p in info.get("penalty_reasons", [])]
    head = env.snake[0]
    food = env.food

    obs = np.asarray(observation, dtype=np.float32).flatten()
    signals_list = [float(obs[i]) if i < len(obs) else float("nan") for i in range(len(GLOBAL_SIGNAL_LABELS))]

    record: dict[str, Any] = {
        "s": step,
        "sc": score,
        "l": len(env.snake),
        "d": env.direction,
        "b": env.board_size,
        "hx": int(head[0]),
        "hy": int(head[1]),
        "fx": int(food[0]),
        "fy": int(food[1]),
        "r": round(float(reward), 4),
        "ds": round(float(getattr(env, 'last_dist_shaping', 0.0)), 4),
        "rs": reason,
        "p": penalties,
        "sig": signals_list,
    }
    if action is not None:
        record["a"] = int(action)
    if q_values is not None:
        q = np.asarray(q_values, dtype=np.float32).flatten()
        record["q"] = [round(float(q[i]), 4) for i in range(min(3, len(q)))]

    _episode_buffer.append(record)


def finish_episode() -> None:
    if not _episode_buffer:
        return
    lines = [json.dumps(r, ensure_ascii=False) for r in _episode_buffer]
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _episode_buffer.clear()
