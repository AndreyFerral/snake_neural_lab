from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend_config import (
    APPLE_STEP_START_IDX,
    DIRECTIONS,
    ENDGAME_IDX,
    ENDGAME_MERGE_BONUS,
    ENDGAME_SPLIT_PENALTY,
    FOOD_REACHABLE_IDX,
    GLOBAL_SIGNAL_LABELS,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    HUNGER_CRITICAL_IDX,
    POCKET_MERGE_START_IDX,
    POCKET_SPLIT_START_IDX,
    TAIL_BLOCKED_START_IDX,
)
from dqn_agent import (
    DQNAgent,
    apply_step_risk_shaping,
    compute_dqn_priority_scale,
    evaluate_pre_step_risks,
    filter_dqn_random_action,
)
from snake_env import SnakeEnv, relative_to_absolute


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _FrameStacker:
    """Формирует конкатенированный стек последних N векторов наблюдений."""

    def __init__(self, frame_stack: int = 1, keep_indices: list[int] | None = None) -> None:
        self.frame_stack = max(1, int(frame_stack))
        self.keep_indices = keep_indices
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    def _project(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.keep_indices] if self.keep_indices is not None else obs

    def reset(self, obs: np.ndarray) -> np.ndarray:
        proj = self._project(obs)
        if self.frame_stack == 1:
            return proj
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(proj)
        return np.concatenate(list(self._frames), axis=0)

    def push(self, obs: np.ndarray) -> np.ndarray:
        proj = self._project(obs)
        if self.frame_stack == 1:
            return proj
        self._frames.append(proj)
        return np.concatenate(list(self._frames), axis=0)


def _resolve_keep_indices(
    drop_signals: str | list[str] | None,
    *,
    strict: bool = True,
) -> tuple[list[str], list[int] | None]:
    if not drop_signals:
        return [], None
    raw = drop_signals.split(",") if isinstance(drop_signals, str) else list(drop_signals)
    drops = [item.strip() for item in raw if item.strip()]
    if not drops:
        return [], None
    unknown = [name for name in drops if name not in GLOBAL_SIGNAL_LABELS]
    if unknown and strict:
        raise ValueError(f"Unknown signals in --drop-signals: {unknown}")
    valid_drops = [name for name in drops if name in GLOBAL_SIGNAL_LABELS]
    if not valid_drops:
        return [], None
    drop_set = set(valid_drops)
    keep = [idx for idx, name in enumerate(GLOBAL_SIGNAL_LABELS) if name not in drop_set]
    return valid_drops, keep


def _empty_components_count(board_size: int, snake: list[tuple[int, int]]) -> int:
    blocked = set(snake)
    visited: set[tuple[int, int]] = set()
    components = 0
    for y in range(board_size):
        for x in range(board_size):
            if (x, y) not in blocked and (x, y) not in visited:
                components += 1
                queue = deque([(x, y)])
                visited.add((x, y))
                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in DIRECTIONS:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < board_size and 0 <= ny < board_size:
                            cell = (nx, ny)
                            if cell not in blocked and cell not in visited:
                                visited.add(cell)
                                queue.append(cell)
    return components


def _diagnose_failed_trace(trace: list[dict[str, Any]], board_size: int) -> tuple[bool, bool, float]:
    """
    Возвращает (avoidable_final_split, missed_adjacent_apple, endgame_food_reachable_ratio)
    для одного незавершённого победой эпизода.
    """
    last_apple_idx = 0
    for i in range(1, len(trace)):
        if trace[i]["score"] > trace[i - 1]["score"]:
            last_apple_idx = i

    endgame = trace[last_apple_idx:]
    food_ok_steps = sum(1 for st in endgame if float(st["obs"][FOOD_REACHABLE_IDX]) > 0.5)
    food_ratio = food_ok_steps / max(1, len(endgame))

    # 1. Проверка пропуска смежного безопасного яблока в эндшпиле
    missed_adjacent = False
    for st in endgame:
        obs = st["obs"]
        chosen = st["action"]
        if float(obs[APPLE_STEP_START_IDX + chosen]) > 0.5:
            continue
        for alt in (0, 1, 2):
            if alt != chosen and float(obs[APPLE_STEP_START_IDX + alt]) > 0.5 and float(obs[TAIL_BLOCKED_START_IDX + alt]) < 0.5:
                missed_adjacent = True
                break
        if missed_adjacent:
            break

    # 2. Проверка последнего разрыва поля на >= 2 компоненты до последнего яблока
    avoidable_split = False
    last_single_idx = None
    for i in range(last_apple_idx, -1, -1):
        if _empty_components_count(board_size, trace[i]["snake"]) == 1:
            last_single_idx = i
            break

    if last_single_idx is not None and last_single_idx < len(trace) - 1:
        st = trace[last_single_idx]
        env = SnakeEnv(board_size)
        env.snake = list(st["snake"])
        env.food = st["food"]
        env.direction = st["direction"]
        env._clear_observation_cache()
        for act in (0, 1, 2):
            if act == st["action"]:
                continue
            ad = relative_to_absolute(env.direction, act)
            dx, dy = DIRECTIONS[ad]
            nh = (env.snake[0][0] + dx, env.snake[0][1] + dy)
            if not (0 <= nh[0] < board_size and 0 <= nh[1] < board_size):
                continue
            will_eat = nh == env.food
            if nh in (env.snake if will_eat else env.snake[:-1]):
                continue
            sim_snake = [nh] + (env.snake if will_eat else env.snake[:-1])
            if _empty_components_count(board_size, sim_snake) == 1 and env._is_tail_blocked_after_relative_action(act) < 0.5:
                avoidable_split = True
                break

    return avoidable_split, missed_adjacent, food_ratio


def evaluate_agent(
    agent: DQNAgent,
    *,
    episodes: int = 100,
    board_size: int = 10,
    seed: int = 2026,
    diagnose_traps: bool = True,
) -> dict[str, Any]:
    _set_seed(seed)
    victory_score = board_size * board_size - 3

    scores: list[int] = []
    steps_list: list[int] = []
    reasons: Counter[str] = Counter()
    buckets: Counter[str] = Counter()

    avoidable_splits = 0
    missed_adjacent_apples = 0
    endgame_food_ratios: list[float] = []

    started_at = time.perf_counter()
    total_steps = 0
    stacker = _FrameStacker(
        getattr(agent, "frame_stack", 1),
        getattr(agent, "_keep_indices", None),
    )

    for _ in range(episodes):
        env = SnakeEnv(board_size=board_size)
        obs = env.reset()
        net_obs = stacker.reset(obs)
        trace: list[dict[str, Any]] = []
        while not env.done:
            action, _, _, _ = agent.select_best_action(net_obs)
            if diagnose_traps:
                trace.append({
                    "score": env.score,
                    "snake": list(env.snake),
                    "food": env.food,
                    "direction": env.direction,
                    "obs": obs.copy(),
                    "action": action,
                })
            obs, _, _, info = env.step(relative_to_absolute(env.direction, action))
            net_obs = stacker.push(obs)
            total_steps += 1

        score = env.score
        reason = str(info["reason"])
        scores.append(score)
        steps_list.append(env.step_count)
        reasons[reason] += 1

        if score >= victory_score:
            buckets["victory"] += 1
        elif score >= 90:
            buckets["90-96"] += 1
        elif score >= 80:
            buckets["80-89"] += 1
        elif score >= 60:
            buckets["60-79"] += 1
        else:
            buckets["<60"] += 1

        if diagnose_traps and reason != "victory" and trace:
            av_split, missed_adj, food_ratio = _diagnose_failed_trace(trace, board_size)
            if av_split:
                avoidable_splits += 1
            if missed_adj:
                missed_adjacent_apples += 1
            endgame_food_ratios.append(food_ratio)

    elapsed_s = max(1e-6, time.perf_counter() - started_at)
    failed_count = episodes - reasons.get("victory", 0)
    h1 = getattr(agent, "hidden1", HIDDEN_LAYER_1)
    h2 = getattr(agent, "hidden2", HIDDEN_LAYER_2)
    fstack = getattr(agent, "frame_stack", 1)
    eff_signals = getattr(agent, "global_signal_count", len(GLOBAL_SIGNAL_LABELS))
    arch_sig = (
        f"sig_{eff_signals}x{fstack}_{h1}_{h2}"
        if fstack > 1
        else f"sig_{eff_signals}_{h1}_{h2}"
    )
    return {
        "signal_count": eff_signals,
        "dropped_signals": getattr(agent, "_drop_signals", []),
        "frame_stack": fstack,
        "hidden_layers": [h1, h2],
        "arch_signature": arch_sig,
        "board_size": board_size,
        "episodes": episodes,
        "seed": seed,
        "victory_score": victory_score,
        "victories": reasons.get("victory", 0),
        "victory_rate_pct": round(reasons.get("victory", 0) * 100.0 / max(1, episodes), 2),
        "avg_score": round(sum(scores) / max(1, len(scores)), 2),
        "median_score": float(np.median(scores)) if scores else 0.0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "avg_steps": round(sum(steps_list) / max(1, len(steps_list)), 1),
        "total_steps": total_steps,
        "elapsed_s": round(elapsed_s, 3),
        "steps_per_sec": round(total_steps / elapsed_s, 1),
        "reasons": dict(reasons),
        "score_buckets": dict(buckets),
        "trap_diagnostics": {
            "failed_episodes": failed_count,
            "avoidable_final_splits": avoidable_splits,
            "missed_solvable_apple": missed_adjacent_apples,
            "avg_endgame_food_reachable_pct": round(
                (sum(endgame_food_ratios) / len(endgame_food_ratios) * 100.0) if endgame_food_ratios else 100.0,
                2,
            ),
        } if diagnose_traps else None,
    }


def load_dqn_checkpoint(ckpt_path: Path) -> DQNAgent:
    payload = torch.load(ckpt_path, map_location="cpu")
    arch_meta = payload.get("arch_meta", {}) if isinstance(payload, dict) else {}
    hidden1 = int(arch_meta.get("hidden1", HIDDEN_LAYER_1))
    hidden2 = int(arch_meta.get("hidden2", HIDDEN_LAYER_2))
    frame_stack = int(arch_meta.get("frame_stack", 1))
    drop_signals, keep_indices = _resolve_keep_indices(arch_meta.get("drop_signals"), strict=False)
    signal_count = int(arch_meta.get("signal_count", len(keep_indices) if keep_indices is not None else len(GLOBAL_SIGNAL_LABELS)))
    if not arch_meta and isinstance(payload, dict):
        online = payload.get("online_net")
        if isinstance(online, dict) and "hidden1_layer.weight" in online and "hidden2_layer.weight" in online:
            w1 = online["hidden1_layer.weight"]
            w2 = online["hidden2_layer.weight"]
            if w1.dim() == 2 and w2.dim() == 2:
                hidden1 = int(w1.shape[0])
                hidden2 = int(w2.shape[0])
                in_dim = int(w1.shape[1])
                sig_n = len(GLOBAL_SIGNAL_LABELS)
                if sig_n > 0 and in_dim % sig_n == 0 and in_dim >= sig_n:
                    frame_stack = in_dim // sig_n
                    signal_count = sig_n
    agent = DQNAgent(
        device=torch.device("cpu"),
        hidden1=hidden1,
        hidden2=hidden2,
        frame_stack=frame_stack,
        signal_count=signal_count,
    )
    agent._drop_signals = drop_signals  # type: ignore[attr-defined]
    agent._keep_indices = keep_indices  # type: ignore[attr-defined]
    if isinstance(payload, dict):
        agent.import_checkpoint(payload)
    return agent


def save_dqn_checkpoint(agent: DQNAgent, ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.export_checkpoint()
    drop_signals = getattr(agent, "_drop_signals", [])
    if isinstance(payload.get("arch_meta"), dict):
        payload["arch_meta"]["drop_signals"] = drop_signals  # type: ignore[index]
    sig_prefix = (
        f"sig_{agent.global_signal_count}x{agent.frame_stack}"
        if agent.frame_stack > 1
        else f"sig_{agent.global_signal_count}"
    )
    payload["arch_signature"] = f"{sig_prefix}_{agent.hidden1}_{agent.hidden2}"
    torch.save(payload, ckpt_path)


def cmd_eval(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    agent = load_dqn_checkpoint(ckpt_path)
    report = evaluate_agent(
        agent,
        episodes=args.episodes,
        board_size=args.board_size,
        seed=args.seed,
        diagnose_traps=not args.fast,
    )
    report["checkpoint"] = str(ckpt_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    drop_signals, keep_indices = _resolve_keep_indices(args.drop_signals)
    signal_count = len(keep_indices) if keep_indices is not None else len(GLOBAL_SIGNAL_LABELS)
    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.is_file():
            print(f"[ERROR] Init checkpoint not found: {init_path}")
            sys.exit(1)
        agent = load_dqn_checkpoint(init_path)
        agent.gamma = float(args.gamma)
        agent.lr_start = float(args.lr)
        agent.lr_min = float(args.lr_min) if args.lr_min is not None else float(args.lr)
        agent.lr_decay_updates = max(0, int(args.lr_decay_updates))
        agent.lr_decay_start_update = max(0, int(args.lr_decay_start))
        agent.epsilon = float(args.eps_start)
        agent.epsilon_min = float(args.eps_min)
        agent.epsilon_decay = float(args.eps_decay)
        agent.target_tau = float(args.tau)
        agent.total_updates = 0
        for pg in agent.optimizer.param_groups:
            pg["lr"] = float(args.lr)
        drop_signals = getattr(agent, "_drop_signals", [])
        keep_indices = getattr(agent, "_keep_indices", None)
    else:
        agent = DQNAgent(
            device=torch.device("cpu"),
            gamma=args.gamma,
            learning_rate=args.lr,
            lr_min=args.lr_min,
            lr_decay_updates=args.lr_decay_updates,
            lr_decay_start_update=args.lr_decay_start,
            epsilon_start=args.eps_start,
            epsilon_min=args.eps_min,
            epsilon_decay=args.eps_decay,
            target_tau=args.tau,
            buffer_size=args.buffer_size,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            frame_stack=args.frame_stack,
            n_step=args.n_step,
            signal_count=signal_count,
        )
        agent._drop_signals = drop_signals  # type: ignore[attr-defined]
        agent._keep_indices = keep_indices  # type: ignore[attr-defined]
    env = SnakeEnv(board_size=args.board_size)
    stacker = _FrameStacker(agent.frame_stack, keep_indices)
    state = env.reset()
    net_state = stacker.reset(state)

    started_at = time.perf_counter()
    episodes_done = 0
    scores_window: deque[int] = deque(maxlen=100)
    checkpoints_log: list[dict[str, Any]] = []
    best_eval_victory = -1.0
    best_eval_avg = -1.0

    out_ckpt = Path(args.out_ckpt)
    best_ckpt = out_ckpt.with_name(f"{out_ckpt.stem}_best{out_ckpt.suffix}")

    for step in range(1, args.steps + 1):
        if args.late_eps_step > 0 and step == args.late_eps_step:
            agent.epsilon_min = float(args.late_eps_min)
            agent.epsilon_decay = float(args.late_eps_decay)
        predicted_action, _, was_random, greedy_action = agent.select_action(
            net_state,
            training=True,
            snake_length=len(env.snake),
        )
        if was_random:
            predicted_action, was_random = filter_dqn_random_action(
                state,
                predicted_action,
                greedy_action,
                was_random,
            )

        pre_risks = evaluate_pre_step_risks(env, state, predicted_action)
        is_tail_loss = pre_risks[0]
        is_pocket_split = float(state[POCKET_SPLIT_START_IDX + predicted_action]) > 0.5
        is_pocket_merge = float(state[POCKET_MERGE_START_IDX + predicted_action]) > 0.5
        is_endgame = float(state[ENDGAME_IDX]) > 0.5
        is_hunger_critical = float(state[HUNGER_CRITICAL_IDX]) > 0.5

        _, reward, done, info = env.step(relative_to_absolute(env.direction, predicted_action))
        reason = str(info["reason"])
        next_state = env.get_observation_grid()
        next_net_state = stacker.push(next_state)

        reward, _ = apply_step_risk_shaping(
            reward,
            reason,
            len(env.snake),
            pre_risks,
            is_endgame=is_endgame,
            is_pocket_split=is_pocket_split,
            is_pocket_merge=is_pocket_merge,
            endgame_split_penalty=args.endgame_split_penalty,
            endgame_merge_bonus=args.endgame_merge_bonus,
            is_hunger_critical=is_hunger_critical,
            critical_hunger_penalty=args.critical_hunger_penalty,
        )
        priority_scale = compute_dqn_priority_scale(
            done,
            reason,
            is_tail_loss,
            env.snake[0],
            env.board_size,
            is_pocket_split=is_pocket_split,
            is_endgame=is_endgame,
            pocket_split_priority=args.split_per,
            endgame_priority=args.endgame_per,
            score=env.score,
            late_endgame_priority=args.late_endgame_per,
        )
        force_perm = (reason == "victory") and (args.late_endgame_per > 1.0)

        if step <= args.warmup_steps:
            agent.replay_buffer.push(
                net_state,
                predicted_action,
                reward,
                next_net_state,
                done,
                priority_scale=priority_scale,
                discount=agent.gamma,
                force_permanent=force_perm,
            )
        else:
            agent.remember(
                net_state,
                predicted_action,
                reward,
                next_net_state,
                done,
                priority_scale=priority_scale,
                force_permanent=force_perm,
            )
            if env.step_count % 8 == 0 or done:
                agent.optimize()

        state = next_state
        net_state = next_net_state
        if done:
            episodes_done += 1
            scores_window.append(env.score)
            agent.finish_episode(
                score=env.score,
                reward=0.0,
                steps=env.step_count,
                reason=reason,
                board_size=env.board_size,
                max_steps_without_food=env.max_steps_without_food,
            )
            state = env.reset()
            net_state = stacker.reset(state)

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_res = evaluate_agent(
                agent,
                episodes=args.eval_episodes,
                board_size=args.board_size,
                seed=args.seed + 1000,
                diagnose_traps=False,
            )
            elapsed = time.perf_counter() - started_at
            row = {
                "step": step,
                "episodes": episodes_done,
                "epsilon": round(agent.epsilon, 4),
                "train_avg100": round(sum(scores_window) / max(1, len(scores_window)), 2),
                "eval_victory_pct": eval_res["victory_rate_pct"],
                "eval_avg_score": eval_res["avg_score"],
                "steps_per_sec": round(step / max(1e-6, elapsed), 1),
            }
            checkpoints_log.append(row)
            print(
                f"[step {step:>7}] ep={episodes_done:<5} eps={row['epsilon']:.3f} "
                f"train_avg100={row['train_avg100']:>5.2f} eval_avg={row['eval_avg_score']:>5.2f} "
                f"win={row['eval_victory_pct']:>5.1f}% ({row['steps_per_sec']:.0f} step/s)",
                flush=True,
            )
            if (eval_res["victory_rate_pct"], eval_res["avg_score"]) > (best_eval_victory, best_eval_avg):
                best_eval_victory = eval_res["victory_rate_pct"]
                best_eval_avg = eval_res["avg_score"]
                save_dqn_checkpoint(agent, best_ckpt)

    save_dqn_checkpoint(agent, out_ckpt)
    final_eval = evaluate_agent(
        agent,
        episodes=args.final_eval_episodes,
        board_size=args.board_size,
        seed=2026,
        diagnose_traps=True,
    )
    total_elapsed = time.perf_counter() - started_at
    summary = {
        "train_steps": args.steps,
        "train_episodes": episodes_done,
        "train_seed": args.seed,
        "train_config": {
            "gamma": args.gamma,
            "lr": args.lr,
            "lr_min": args.lr_min,
            "lr_decay_updates": args.lr_decay_updates,
            "eps_min": args.eps_min,
            "eps_decay": args.eps_decay,
            "tau": args.tau,
            "split_per": args.split_per,
            "endgame_per": args.endgame_per,
            "endgame_split_penalty": args.endgame_split_penalty,
            "endgame_merge_bonus": args.endgame_merge_bonus,
            "critical_hunger_penalty": args.critical_hunger_penalty,
            "buffer_size": args.buffer_size,
            "n_step": args.n_step,
            "late_endgame_per": args.late_endgame_per,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "frame_stack": args.frame_stack,
            "drop_signals": drop_signals,
        },
        "elapsed_s": round(total_elapsed, 2),
        "train_steps_per_sec": round(args.steps / max(1e-6, total_elapsed), 1),
        "output_checkpoint": str(out_ckpt),
        "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
        "curve": checkpoints_log,
        "final_eval": final_eval,
    }
    print("\n=== FINAL EVAL ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_compare(args: argparse.Namespace) -> None:
    base_raw = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    cand_raw = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    base = base_raw.get("final_eval", base_raw)
    cand = cand_raw.get("final_eval", cand_raw)

    def _get_trap(d: dict[str, Any], key: str) -> float:
        traps = d.get("trap_diagnostics") or {}
        return float(traps.get(key, 0.0))

    rows = [
        ("Сигналы (вход)", base.get("signal_count", 0), cand.get("signal_count", 0)),
        ("Побед (97/97), %", base.get("victory_rate_pct", 0.0), cand.get("victory_rate_pct", 0.0)),
        ("Средний счёт", base.get("avg_score", 0.0), cand.get("avg_score", 0.0)),
        ("Медиана счёта", base.get("median_score", 0.0), cand.get("median_score", 0.0)),
        ("Мин. счёт", base.get("min_score", 0), cand.get("min_score", 0)),
        ("Скорость (шаг/с)", base.get("steps_per_sec", 0.0), cand.get("steps_per_sec", 0.0)),
        ("Избегаемых разрывов", _get_trap(base, "avoidable_final_splits"), _get_trap(cand, "avoidable_final_splits")),
        ("Пропусков смежн. яблока", _get_trap(base, "missed_solvable_apple"), _get_trap(cand, "missed_solvable_apple")),
        ("Доступ еды в эндшпиле %", _get_trap(base, "avg_endgame_food_reachable_pct"), _get_trap(cand, "avg_endgame_food_reachable_pct")),
    ]
    print(f"{'Метрика':<28} | {'Бейзлайн':>10} | {'Кандидат':>10} | {'Дельта':>10}")
    print("-" * 68)
    for label, b_val, c_val in rows:
        delta = float(c_val) - float(b_val)
        print(f"{label:<28} | {b_val:>10} | {c_val:>10} | {delta:>+10.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Оценка, headless-обучение и A/B-сравнение моделей DQN.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Оценка чекпоинта на фиксированных эпизодах.")
    p_eval.add_argument("--ckpt", default=str(PROJECT_ROOT / ".runtime" / "snake_dqn.pt"))
    p_eval.add_argument("--episodes", type=int, default=100)
    p_eval.add_argument("--board-size", type=int, default=10)
    p_eval.add_argument("--seed", type=int, default=2026)
    p_eval.add_argument("--fast", action="store_true", help="Пропустить диагностику ловушек")
    p_eval.add_argument("--out", default=None, help="Путь для сохранения JSON-отчёта")
    p_eval.set_defaults(func=cmd_eval)

    p_train = sub.add_parser("train", help="Headless-обучение DQN с нуля в отдельный чекпоинт.")
    p_train.add_argument("--steps", type=int, default=300_000)
    p_train.add_argument("--board-size", type=int, default=10)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--eval-every", type=int, default=50_000)
    p_train.add_argument("--eval-episodes", type=int, default=25)
    p_train.add_argument("--final-eval-episodes", type=int, default=100)
    p_train.add_argument("--gamma", type=float, default=0.97)
    p_train.add_argument("--lr", type=float, default=0.0008)
    p_train.add_argument("--lr-min", type=float, default=None)
    p_train.add_argument("--lr-decay-updates", type=int, default=0)
    p_train.add_argument("--lr-decay-start", type=int, default=0)
    p_train.add_argument("--eps-start", type=float, default=1.0)
    p_train.add_argument("--eps-min", type=float, default=0.08)
    p_train.add_argument("--eps-decay", type=float, default=0.9997)
    p_train.add_argument("--late-eps-step", type=int, default=0)
    p_train.add_argument("--late-eps-min", type=float, default=0.02)
    p_train.add_argument("--late-eps-decay", type=float, default=0.9999)
    p_train.add_argument("--warmup-steps", type=int, default=0)
    p_train.add_argument("--init-ckpt", default=None)
    p_train.add_argument("--tau", type=float, default=0.0)
    p_train.add_argument("--split-per", type=float, default=1.0)
    p_train.add_argument("--endgame-per", type=float, default=1.0)
    p_train.add_argument("--late-endgame-per", type=float, default=1.0)
    p_train.add_argument("--buffer-size", type=int, default=100_000)
    p_train.add_argument("--n-step", type=int, default=1)
    p_train.add_argument("--endgame-split-penalty", type=float, default=ENDGAME_SPLIT_PENALTY)
    p_train.add_argument("--endgame-merge-bonus", type=float, default=ENDGAME_MERGE_BONUS)
    p_train.add_argument("--critical-hunger-penalty", type=float, default=0.0)
    p_train.add_argument("--hidden1", type=int, default=HIDDEN_LAYER_1)
    p_train.add_argument("--hidden2", type=int, default=HIDDEN_LAYER_2)
    p_train.add_argument("--frame-stack", type=int, default=1)
    p_train.add_argument("--drop-signals", default=None)
    p_train.add_argument("--out-ckpt", default=str(PROJECT_ROOT / ".runtime" / "experiments" / "dqn_exp.pt"))
    p_train.add_argument("--out", default=None, help="Путь для сохранения JSON-отчёта")
    p_train.set_defaults(func=cmd_train)

    p_cmp = sub.add_parser("compare", help="Сравнить два JSON-отчёта (бейзлайн vs кандидат).")
    p_cmp.add_argument("baseline", help="JSON-отчёт бейзлайна")
    p_cmp.add_argument("candidate", help="JSON-отчёт новой итерации")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
