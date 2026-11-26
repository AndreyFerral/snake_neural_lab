from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from backend_config import PERF_LOG_MAX_BYTES


def write_json_atomic(target_path: Path, payload: Any) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    for attempt in range(5):
        try:
            tmp_path.write_text(encoded, encoding="utf-8")
            tmp_path.replace(target_path)
            return
        except OSError:
            if attempt < 4:
                time.sleep(0.05)


def checkpoint_path_for_agent(runtime_dir: Path, agent_kind: str) -> Path:
    return runtime_dir / f"snake_{agent_kind}.pt"


def model_stats_key(agent_kind: str) -> str:
    return f"snake_{agent_kind}"


def compact_model_stats(models: dict[str, Any], checkpoint_key: str, agent_kind: str) -> None:
    prefix = f"snake_{agent_kind}"
    for key in list(models.keys()):
        if key == checkpoint_key:
            continue
        if key.startswith(prefix):
            models.pop(key, None)


def prune_model_stats_store(models: dict[str, Any], *, keep_keys: set[str], smart_limit: int = 3) -> None:
    smart_keys = [key for key in models.keys() if key.startswith("snake_smart_")]
    if len(smart_keys) > smart_limit:
        for key in sorted(smart_keys)[:-smart_limit]:
            if key not in keep_keys:
                models.pop(key, None)
    for key in list(models.keys()):
        if key in keep_keys:
            continue
        if key.startswith("snake_dqn") or key.startswith("snake_ppo") or key.startswith("snake_evolution"):
            models.pop(key, None)


def load_aux_stats_store(aux_stats_path: Path) -> dict[str, Any]:
    default_store: dict[str, Any] = {"models": {}}
    if not aux_stats_path.exists():
        return default_store
    try:
        payload = json.loads(aux_stats_path.read_text(encoding="utf-8"))
    except Exception:
        return default_store
    if not isinstance(payload, dict):
        return default_store
    models = payload.get("models", {})
    return {"models": models if isinstance(models, dict) else {}}


def save_aux_stats_store(store: dict[str, Any], aux_stats_path: Path) -> None:
    write_json_atomic(aux_stats_path, store)


def clear_aux_stats_store(aux_stats_path: Path) -> None:
    save_aux_stats_store({"models": {}}, aux_stats_path)


def reset_session_perf_log(runtime_dir: Path, perf_log_path: Path) -> None:
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        perf_log_path.write_text("", encoding="utf-8")
        rotated = perf_log_path.with_suffix(".log.1")
        if rotated.exists():
            rotated.unlink()
    except OSError:
        pass


def log_perf(runtime_dir: Path, perf_log_path: Path, event: str, elapsed_ms: float, **fields: Any) -> None:
    if not runtime_dir.exists():
        runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        if perf_log_path.exists() and perf_log_path.stat().st_size > PERF_LOG_MAX_BYTES:
            perf_log_path.replace(perf_log_path.with_suffix(".log.1"))
    except OSError:
        pass

    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {event} {elapsed_ms:.1f}ms"
    if suffix:
        line += f" {suffix}"
    try:
        with open(perf_log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
