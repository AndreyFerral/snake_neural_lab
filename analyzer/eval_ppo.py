from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend_config import (
    DANGER_START_IDX,
    DIRECTIONS,
    FOOD_REACHABLE_IDX,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    POCKET_SPLIT_START_IDX,
    PPO_MAP_CHANNELS,
    PPO_MAP_SIZE,
    PPO_SIGNAL_LABELS,
    TAIL_BLOCKED_START_IDX,
)
from ppo_agent import PPOAgent, ParallelPPORunner
from snake_env import SnakeEnv, relative_to_absolute


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _count_dead_end_cells(board_size: int, snake: list[tuple[int, int]]) -> int:
    """Количество пустых клеток поля, имеющих <= 1 свободного соседа (тупики/дырки)."""
    blocked = set(snake)
    dead_ends = 0
    for y in range(board_size):
        for x in range(board_size):
            if (x, y) in blocked:
                continue
            free_nbs = 0
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size and (nx, ny) not in blocked:
                    free_nbs += 1
            if free_nbs <= 1:
                dead_ends += 1
    return dead_ends


def _classify_failure(
    trace: list[dict[str, Any]],
    reason: str,
    board_size: int,
    map_size: int,
) -> str:
    """Классифицирует корневую причину проигрыша по последним шагам траектории."""
    if reason == "starvation":
        return "starvation_loop"
    if not trace:
        return reason

    window = trace[-20:]
    # Проверяем, мешал ли tail_blocked_logit_bias (raw_greedy != chosen_action)
    bias_overrides = sum(1 for st in window if st.get("raw_greedy") != st.get("action"))
    # Проверяем, отрезала ли змейка хвост ради яблока или при доступной альтернативе
    tail_cut_for_apple = False
    tail_cut_avoidable = False
    blind_spot_tail = False
    dead_end_buildup = False
    radius = map_size // 2

    dead_end_counts = []
    for st in window:
        obs = st["obs"]
        act = int(st["action"])
        tb_chosen = float(obs[TAIL_BLOCKED_START_IDX + act]) > 0.5
        has_safe_tb_ok = any(
            float(obs[DANGER_START_IDX + a]) < 0.5 and float(obs[TAIL_BLOCKED_START_IDX + a]) < 0.5
            for a in (0, 1, 2)
        )
        if tb_chosen and has_safe_tb_ok:
            if st.get("ate_apple"):
                tail_cut_for_apple = True
            else:
                tail_cut_avoidable = True
        head = st["snake"][0]
        tail = st["snake"][-1]
        if max(abs(head[0] - tail[0]), abs(head[1] - tail[1])) > radius:
            blind_spot_tail = True
        dead_end_counts.append(_count_dead_end_cells(board_size, st["snake"]))

    if dead_end_counts and max(dead_end_counts) >= 2 and dead_end_counts[-1] > dead_end_counts[0]:
        dead_end_buildup = True

    if bias_overrides > 0 and not tail_cut_avoidable:
        return "tail_bias_override"
    if tail_cut_for_apple:
        return "tail_cut_for_apple"
    if tail_cut_avoidable:
        return "tail_cut_avoidable"
    if dead_end_buildup:
        return "dead_end_parity_trap"
    if blind_spot_tail:
        return "blind_spot_outside_map"
    return f"other_{reason}"


def load_ppo_checkpoint(
    ckpt_path: Path,
    *,
    use_best_net: bool = False,
    tail_blocked_logit_bias: float | None = None,
) -> PPOAgent:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid PPO checkpoint: {ckpt_path}")

    state_key = "best_policy_net" if (use_best_net and isinstance(payload.get("best_policy_net"), dict)) else "policy_net"
    net_state = payload.get(state_key)
    if not isinstance(net_state, dict):
        raise ValueError(f"Missing {state_key} in {ckpt_path}")

    w1 = net_state.get("hidden1_layer.weight")
    w2 = net_state.get("hidden2_layer.weight")
    c0 = net_state.get("map_conv.0.weight")
    mp = net_state.get("map_proj.weight")

    map_channels = int(c0.shape[1]) if torch.is_tensor(c0) else PPO_MAP_CHANNELS
    map_feat_size = int(mp.shape[0]) if torch.is_tensor(mp) else 32
    hidden1 = int(w1.shape[0]) if torch.is_tensor(w1) else HIDDEN_LAYER_1
    hidden2 = int(w2.shape[0]) if torch.is_tensor(w2) else HIDDEN_LAYER_2
    sig_count = int(w1.shape[1]) - map_feat_size if torch.is_tensor(w1) else len(PPO_SIGNAL_LABELS)

    # Определяем map_size по conv_out_dim
    conv_out_dim = int(mp.shape[1]) if torch.is_tensor(mp) else 384
    map_size = PPO_MAP_SIZE
    for cand in (9, 11, 13, 15):
        probe = PPOAgent(
            global_signal_count=sig_count,
            device=torch.device("cpu"),
            map_channels=map_channels,
            map_size=cand,
            hidden1=hidden1,
            hidden2=hidden2,
        )
        if int(probe.online_net.map_proj.weight.shape[1]) == conv_out_dim:
            map_size = cand
            break

    kwargs: dict[str, Any] = {
        "global_signal_count": sig_count,
        "device": torch.device("cpu"),
        "map_channels": map_channels,
        "map_size": map_size,
        "hidden1": hidden1,
        "hidden2": hidden2,
    }
    if tail_blocked_logit_bias is not None:
        kwargs["tail_blocked_logit_bias"] = float(tail_blocked_logit_bias)

    agent = PPOAgent(**kwargs)
    agent.import_checkpoint(payload)
    if use_best_net and isinstance(payload.get("best_policy_net"), dict):
        agent.online_net.load_state_dict(payload["best_policy_net"])
        agent.sync_inference_net()
    return agent


def evaluate_ppo(
    agent: PPOAgent,
    *,
    episodes: int = 100,
    board_size: int = 10,
    seed: int = 2026,
    diagnose_failures: bool = True,
    max_steps_cap: int | None = None,
) -> dict[str, Any]:
    _set_seed(seed)
    victory_score = board_size * board_size - 3
    scores: list[int] = []
    fill_ratios: list[float] = []
    steps_list: list[int] = []
    reasons: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()
    failure_samples: list[dict[str, Any]] = []

    started_at = time.perf_counter()
    total_steps = 0
    env_ms_total = 0.0
    net_ms_total = 0.0

    for ep_idx in range(episodes):
        env = SnakeEnv(board_size=board_size)
        env.reset(return_obs=False)
        trace: list[dict[str, Any]] = []
        ep_cap = max_steps_cap or (board_size * board_size * 40)

        while not env.done and env.step_count < ep_cap:
            t_env0 = time.perf_counter()
            if agent.global_signal_count == len(PPO_SIGNAL_LABELS):
                sig = env.get_ppo_signals()
            elif agent.global_signal_count > len(env.get_extra_signals()):
                sig = env.get_ppo_signals()[: agent.global_signal_count]
            else:
                sig = env.get_extra_signals()
            ego = env.get_egocentric_map(agent.map_size)
            env_ms_total += (time.perf_counter() - t_env0) * 1000.0

            t_net0 = time.perf_counter()
            action, probs, _, greedy = agent.select_best_action(sig, ego_map=ego)
            net_ms_total += (time.perf_counter() - t_net0) * 1000.0

            raw_greedy = greedy
            abs_dir = relative_to_absolute(env.direction, action)
            old_score = env.score
            t_step0 = time.perf_counter()
            _, _, _, info = env.step(abs_dir, return_obs=False)
            env_ms_total += (time.perf_counter() - t_step0) * 1000.0
            total_steps += 1

            if diagnose_failures:
                trace.append({
                    "step": env.step_count,
                    "score": old_score,
                    "snake": list(env.snake),
                    "food": env.food,
                    "direction": env.direction,
                    "obs": sig.copy(),
                    "ego": ego.copy(),
                    "action": action,
                    "raw_greedy": raw_greedy,
                    "ate_apple": env.score > old_score,
                })
                if len(trace) > 25:
                    trace.pop(0)

        reason = str(env.last_reason if env.done else "step_cap")
        score = int(env.score)
        scores.append(score)
        fill_ratios.append(float(score + 3) / float(board_size * board_size))
        steps_list.append(int(env.step_count))
        reasons[reason] += 1

        if diagnose_failures and reason != "victory":
            if agent.effective_tail_bias() != 0.0:
                saved_bias = agent.tail_blocked_logit_bias
                agent.tail_blocked_logit_bias = 0.0
                for st in trace:
                    _, _, _, rg = agent.select_best_action(st["obs"], ego_map=st["ego"])
                    st["raw_greedy"] = rg
                agent.tail_blocked_logit_bias = saved_bias
            fclass = _classify_failure(trace, reason, board_size, agent.map_size)
            failure_classes[fclass] += 1
            if len(failure_samples) < 12:
                failure_samples.append({
                    "ep": ep_idx,
                    "score": score,
                    "fill_pct": round(100.0 * (score + 3) / (board_size * board_size), 1),
                    "steps": int(env.step_count),
                    "reason": reason,
                    "class": fclass,
                })

    elapsed_s = max(1e-6, time.perf_counter() - started_at)
    wins = reasons.get("victory", 0)
    return {
        "board_size": board_size,
        "episodes": episodes,
        "seed": seed,
        "map_size": agent.map_size,
        "signal_count": agent.global_signal_count,
        "tail_bias": round(agent.effective_tail_bias(), 4),
        "victory_score": victory_score,
        "victories": wins,
        "win_rate_pct": round(100.0 * wins / max(1, episodes), 2),
        "avg_score": round(float(np.mean(scores)), 2),
        "median_score": round(float(np.median(scores)), 1),
        "p10_score": round(float(np.percentile(scores, 10)), 1),
        "min_score": int(min(scores)),
        "max_score": int(max(scores)),
        "avg_fill_pct": round(100.0 * float(np.mean(fill_ratios)), 2),
        "avg_steps": round(float(np.mean(steps_list)), 1),
        "elapsed_s": round(elapsed_s, 2),
        "steps_per_sec": round(total_steps / elapsed_s, 1),
        "env_ms_per_step": round(env_ms_total / max(1, total_steps), 3),
        "net_ms_per_step": round(net_ms_total / max(1, total_steps), 3),
        "reasons": dict(reasons),
        "failure_classes": dict(failure_classes),
        "failure_samples": failure_samples,
    }


def cmd_eval(args: argparse.Namespace) -> None:
    ckpt = Path(args.ckpt)
    agent = load_ppo_checkpoint(
        ckpt,
        use_best_net=args.best,
        tail_blocked_logit_bias=args.tail_bias,
    )
    report = evaluate_ppo(
        agent,
        episodes=args.episodes,
        board_size=args.board_size,
        seed=args.seed,
        diagnose_failures=not args.fast,
    )
    report["checkpoint"] = str(ckpt)
    report["use_best_net"] = bool(args.best)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_multiboard(args: argparse.Namespace) -> None:
    ckpt = Path(args.ckpt)
    sizes = [int(x.strip()) for x in args.boards.split(",") if x.strip()]
    eps_map = {10: args.ep10, 15: args.ep15, 20: args.ep20, 25: args.ep25, 30: args.ep30, 35: args.ep35}
    results = []
    for bias in ([args.tail_bias] if args.tail_bias is not None else [0.5, 0.0]):
        agent = load_ppo_checkpoint(ckpt, use_best_net=args.best, tail_blocked_logit_bias=bias)
        for bs in sizes:
            ep = eps_map.get(bs, 20)
            rep = evaluate_ppo(
                agent,
                episodes=ep,
                board_size=bs,
                seed=args.seed,
                diagnose_failures=True,
            )
            results.append(rep)
            print(
                f"[board={bs:>2}x{bs:<2} bias={bias:.1f} ep={ep:>3}] "
                f"win={rep['win_rate_pct']:>6.2f}% | avg={rep['avg_score']:>6.2f}/{rep['victory_score']} "
                f"(fill={rep['avg_fill_pct']:>5.1f}%) | p10={rep['p10_score']:>5.1f} min={rep['min_score']:>3} | "
                f"env={rep['env_ms_per_step']:.3f}ms net={rep['net_ms_per_step']:.3f}ms | "
                f"fails={rep['failure_classes']}",
                flush=True,
            )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Оценка и диагностика PPO (Map) на разных размерах поля.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Жадная оценка PPO на одном размере поля.")
    p_eval.add_argument("--ckpt", default=str(PROJECT_ROOT / ".runtime" / "snake_ppo.pt"))
    p_eval.add_argument("--best", action="store_true", help="Загрузить best_policy_net вместо policy_net")
    p_eval.add_argument("--board-size", type=int, default=10)
    p_eval.add_argument("--episodes", type=int, default=200)
    p_eval.add_argument("--seed", type=int, default=2026)
    p_eval.add_argument("--tail-bias", type=float, default=None)
    p_eval.add_argument("--fast", action="store_true")
    p_eval.add_argument("--out", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_mb = sub.add_parser("multiboard", help="Оценка на нескольких размерах поля (10..35).")
    p_mb.add_argument("--ckpt", default=str(PROJECT_ROOT / ".runtime" / "snake_ppo.pt"))
    p_mb.add_argument("--best", action="store_true")
    p_mb.add_argument("--boards", default="10,15,20,35")
    p_mb.add_argument("--ep10", type=int, default=300)
    p_mb.add_argument("--ep15", type=int, default=30)
    p_mb.add_argument("--ep20", type=int, default=15)
    p_mb.add_argument("--ep25", type=int, default=10)
    p_mb.add_argument("--ep30", type=int, default=8)
    p_mb.add_argument("--ep35", type=int, default=5)
    p_mb.add_argument("--seed", type=int, default=2026)
    p_mb.add_argument("--tail-bias", type=float, default=None)
    p_mb.add_argument("--out", default=None)
    p_mb.set_defaults(func=cmd_multiboard)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
