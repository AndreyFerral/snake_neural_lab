from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from backend_config import SLOW_API_MS


STATIC_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = STATIC_ROOT / ".runtime"
PID_PATH = STATIC_ROOT / ".web_backend.pid"


def log_perf(event: str, elapsed_ms: float, **fields: Any) -> None:
    return


def reset_session_perf_log() -> None:
    return


class SnakeRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory or str(STATIC_ROOT), **kwargs)

    def handle(self) -> None:
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            return

    @property
    def backend(self) -> Any:
        return self.server.backend  # type: ignore[attr-defined]

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        started_at = time.perf_counter()
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        encode_ms = (time.perf_counter() - started_at) * 1000.0
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        write_started_at = time.perf_counter()
        self.wfile.write(encoded)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if elapsed_ms >= SLOW_API_MS:
            log_perf(
                "slow_send_json",
                elapsed_ms,
                status=int(status),
                bytes=len(encoded),
                encode_ms=round(encode_ms, 1),
                write_ms=round((time.perf_counter() - write_started_at) * 1000.0, 1),
            )

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        started_at = time.perf_counter()
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                include_network = query.get("network", ["0"])[0] != "0"
                include_details = query.get("details", ["0"])[0] != "0"
                self._send_json(self.backend.state_snapshot(include_network=include_network, include_details=include_details))
                return
            if parsed.path == "/api/frame":
                self._send_json(self.backend.frame_snapshot())
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True})
                return
            if parsed.path == "/":
                self.path = "/index.html"
            super().do_GET()
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if elapsed_ms >= SLOW_API_MS:
                log_perf("slow_get", elapsed_ms, path=parsed.path)

    def do_POST(self) -> None:
        started_at = time.perf_counter()
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/api/command":
                self._send_json({"ok": False, "error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return

            result = self.backend.handle_command(payload)
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
            self._send_json(result, status=status)
            if result.get("ok") and payload.get("type") in {"restart_server", "shutdown_server"}:
                if payload.get("type") == "restart_server":
                    self.server.restart_requested = True  # type: ignore[attr-defined]
                elif payload.get("close_console"):
                    self.server.close_console_requested = True  # type: ignore[attr-defined]
                threading.Thread(target=self.server.shutdown, daemon=True).start()
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if elapsed_ms >= SLOW_API_MS:
                log_perf("slow_post", elapsed_ms, path=parsed.path)

    def log_message(self, format: str, *args: Any) -> None:
        return


class SnakeHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler: type[SnakeRequestHandler], backend: Any) -> None:
        super().__init__(server_address, handler)
        self.backend = backend
        self.restart_requested = False
        self.close_console_requested = False

    def handle_error(self, request: Any, client_address: Any) -> None:
        error_type = sys.exc_info()[0]
        if error_type in {ConnectionResetError, ConnectionAbortedError, BrokenPipeError}:
            return
        super().handle_error(request, client_address)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake browser frontend with Python backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def backend_process_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(sys.argv[0]).resolve()),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
    ]
    if args.force_cpu:
        command.append("--force-cpu")
    return command


def close_console_window() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if not hwnd:
            return False
        return bool(ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0))
    except Exception:
        return False


def restart_backend_in_place(args: argparse.Namespace) -> None:
    command = backend_process_command(args)
    print("Restarting backend in the same console...")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, command)


def main(
    *,
    backend_class: type[Any] | None = None,
    static_root: Path | None = None,
    runtime_dir: Path | None = None,
    pid_path: Path | None = None,
    reset_perf_log_func: Any | None = None,
    log_perf_func: Any | None = None,
) -> None:
    global STATIC_ROOT, RUNTIME_DIR, PID_PATH, log_perf, reset_session_perf_log
    if backend_class is None:
        from web_backend import (
            PID_PATH as backend_pid_path,
            RUNTIME_DIR as backend_runtime_dir,
            STATIC_ROOT as backend_static_root,
            SnakeWebBackend,
            log_perf as backend_log_perf,
            reset_session_perf_log as backend_reset_session_perf_log,
        )

        backend_class = SnakeWebBackend
        static_root = backend_static_root
        runtime_dir = backend_runtime_dir
        pid_path = backend_pid_path
        reset_perf_log_func = backend_reset_session_perf_log
        log_perf_func = backend_log_perf

    if static_root is not None:
        STATIC_ROOT = static_root
    if runtime_dir is not None:
        RUNTIME_DIR = runtime_dir
    if pid_path is not None:
        PID_PATH = pid_path
    if reset_perf_log_func is not None:
        reset_session_perf_log = reset_perf_log_func
    if log_perf_func is not None:
        log_perf = log_perf_func

    args = parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    reset_session_perf_log()
    PID_PATH.write_text(f"process:{os.getpid()}", encoding="ascii")
    backend = backend_class(force_cpu=True)
    handler = partial(SnakeRequestHandler, directory=str(STATIC_ROOT))
    server = SnakeHTTPServer((args.host, args.port), handler, backend)

    print(f"Snake backend running at http://{args.host}:{args.port}")
    print(f"Device: {backend.device_kind}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        restart_requested = server.restart_requested
        close_console_requested = server.close_console_requested
        server.server_close()
        backend.shutdown(fast=restart_requested)
        if PID_PATH.exists():
            try:
                PID_PATH.unlink()
            except OSError:
                pass
    if restart_requested:
        time.sleep(0.25)
        restart_backend_in_place(args)
    if close_console_requested:
        time.sleep(0.15)
        close_console_window()


if __name__ == "__main__":
    main()



