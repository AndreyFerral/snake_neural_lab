from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
import gzip
import json
from pathlib import Path
import time
from typing import Any

from backend_config import PERF_LOG_MAX_BYTES, SLOW_API_MS


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


class ReplayManager:
    """Ограниченный фоновый архиватор реплеев (.json.gz) с ротацией."""

    def __init__(self, replays_dir: Path, *, keep_count: int = 10, perf_logger: Any | None = None) -> None:
        self.replays_dir = replays_dir
        self.keep_count = keep_count
        self.perf_logger = perf_logger
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snake-replay")
        self._closed = False

    def submit(
        self,
        steps: list[dict[str, Any]],
        score: int,
        reason: str,
        prefix: str,
        board_size: int,
        episode_counter: int,
        agent_kind: str,
    ) -> None:
        if self._closed or not steps:
            return
        steps_copy = list(steps)
        try:
            self._executor.submit(
                self._save_worker,
                steps_copy,
                score,
                reason,
                prefix,
                board_size,
                episode_counter,
                agent_kind,
            )
        except RuntimeError:
            pass

    def _ensure_readme(self) -> None:
        readme_path = self.replays_dir / "README.md"
        if readme_path.exists():
            return
        try:
            readme_content = (
                "# Папка реплеев / Replays Folder\n\n"
                "Здесь сохраняются JSON-записи сыгранных игр для последующего анализа.\n\n"
                "## Как запустить анализатор:\n\n"
                "Все инструменты лежат в папке `analyzer/` в корне проекта.\n\n"
                "1. **Запуск комплексного анализа всех реплеев:**\n"
                "   ```bash\n"
                "   python analyzer/run_analysis.py\n"
                "   ```\n\n"
                "2. **Визуализация шагов конкретного реплея:**\n"
                "   ```bash\n"
                "   python analyzer/visualize_steps.py .runtime/replays/<replay_file>.json.gz\n"
                "   ```\n"
            )
            readme_path.write_text(readme_content, encoding="utf-8")
        except OSError:
            pass

    def _save_worker(
        self,
        steps: list[dict[str, Any]],
        score: int,
        reason: str,
        prefix: str,
        board_size: int,
        episode_counter: int,
        agent_kind: str,
    ) -> None:
        started_at = time.perf_counter()
        replay_data = {
            "agent_kind": agent_kind,
            "board_size": board_size,
            "episode": int(episode_counter),
            "total_steps": int(steps[-1]["step"]) if steps else 0,
            "final_score": score,
            "final_reason": reason,
            "steps": steps,
        }

        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_readme()

        try:
            json_str = json.dumps(replay_data, ensure_ascii=False)
            compressed_bytes = gzip.compress(json_str.encode("utf-8"), compresslevel=6)
        except Exception as exc:
            print(f"Error serializing/compressing replay data: {exc}")
            compressed_bytes = None

        if compressed_bytes is not None:
            timestamp = int(time.time())
            filename = f"replay_score_{score}_ep_{episode_counter}_{prefix}_{timestamp}.json.gz"
            archive_path = self.replays_dir / filename
            try:
                archive_path.write_bytes(compressed_bytes)
            except Exception as exc:
                print(f"Error saving archived replay: {exc}")

        self.prune()

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if elapsed_ms >= SLOW_API_MS and self.perf_logger is not None:
            self.perf_logger("slow_save_replay", elapsed_ms, steps=len(steps), score=score, reason=reason)

    def prune(self) -> None:
        if not self.replays_dir.is_dir():
            return
        files: list[tuple[float, Path]] = []
        for path in self.replays_dir.glob("replay_score_*.json.gz"):
            try:
                files.append((path.stat().st_mtime, path))
            except OSError:
                continue
        files.sort(key=lambda item: item[0], reverse=True)
        for _, path in files[self.keep_count :]:
            try:
                if path.is_file():
                    path.unlink()
            except OSError as exc:
                if getattr(exc, "errno", None) != errno.ENOENT:
                    print(f"Error deleting old replay {path.name}: {exc}")

    def clear(self) -> None:
        if not self.replays_dir.is_dir():
            return
        for path in self.replays_dir.glob("*.json*"):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass

    def shutdown(self, *, wait: bool = False) -> None:
        self._closed = True
        self._executor.shutdown(wait=wait)
