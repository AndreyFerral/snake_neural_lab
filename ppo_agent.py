from __future__ import annotations

from collections import deque
import random
from typing import Any, Callable, Deque
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from backend_config import (
    DANGER_START_IDX,
    ENDGAME_IDX,
    ENDGAME_MERGE_BONUS,
    ENDGAME_SPLIT_PENALTY,
    FOOD_REACHABLE_IDX,
    GLOBAL_SIGNAL_LABELS,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    MODEL_ACTION_LABELS,
    POCKET_MERGE_START_IDX,
    POCKET_SPLIT_START_IDX,
    PPO_MAP_CHANNELS,
    PPO_MAP_SIZE,
    PPO_SIGNAL_LABELS,
    TAIL_BLOCKED_START_IDX,
)
from dqn_agent import (
    apply_step_risk_shaping,
    evaluate_pre_step_risks,
    load_matching_state,
    move_optimizer_state,
)
from networks import PPOMapNetwork
from snake_env import relative_to_absolute

PPO_NUM_ENVS = 8
PPO_ROLLOUT_STEPS = 128
PPO_GAMMA = 0.992
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_COEF = 0.2
PPO_VF_CLIP_COEF = 0.25
PPO_TARGET_KL = 0.015
PPO_ENT_COEF = 0.025
PPO_VF_COEF = 0.5
PPO_LR = 4e-4
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 256
PPO_MAX_GRAD_NORM = 0.5
PPO_REWARD_SCALE = 0.1
PPO_BEST_WINDOW = 30
PPO_BEST_MIN_EPISODES = 20
PPO_FAILURE_MIN_SCORE = 25
PPO_FAILURE_BUFFER_MAXLEN = 64
PPO_FAILURE_SNAPSHOT_INTERVAL = 4
PPO_FAILURE_REWIND_TARGET = 12
PPO_TAIL_BLOCKED_LOGIT_BIAS = 0.5


def valid_action_mask_from_signals(signals: torch.Tensor) -> torch.Tensor:
    """Возвращает булеву маску допустимых действий (запрещает смертельные, если есть хотя бы 1 безопасное)."""
    dangers = signals[:, DANGER_START_IDX : DANGER_START_IDX + len(MODEL_ACTION_LABELS)]
    valid = dangers < 0.5
    none_valid = ~valid.any(dim=-1, keepdim=True)
    return torch.where(none_valid, torch.ones_like(valid, dtype=torch.bool), valid)


class PPOAgent:
    def __init__(
        self,
        global_signal_count: int = len(PPO_SIGNAL_LABELS),
        *,
        device: torch.device,
        map_channels: int = PPO_MAP_CHANNELS,
        map_size: int = PPO_MAP_SIZE,
        hidden1: int = HIDDEN_LAYER_1,
        hidden2: int = HIDDEN_LAYER_2,
        num_envs: int = PPO_NUM_ENVS,
        rollout_steps: int = PPO_ROLLOUT_STEPS,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
        clip_coef: float = PPO_CLIP_COEF,
        vf_clip_coef: float = PPO_VF_CLIP_COEF,
        target_kl: float | None = PPO_TARGET_KL,
        ent_coef: float = PPO_ENT_COEF,
        vf_coef: float = PPO_VF_COEF,
        lr: float = PPO_LR,
        epochs: int = PPO_EPOCHS,
        minibatch_size: int = PPO_MINIBATCH_SIZE,
        tail_blocked_logit_bias: float = PPO_TAIL_BLOCKED_LOGIT_BIAS,
    ) -> None:
        self.device = device
        self.global_signal_count = int(global_signal_count)
        self.map_channels = int(map_channels)
        self.map_size = int(map_size)
        self.num_envs = int(num_envs)
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_coef = float(clip_coef)
        self.vf_clip_coef = float(vf_clip_coef)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.base_lr = float(lr)
        self.epochs = int(epochs)
        self.minibatch_size = int(minibatch_size)
        self.tail_blocked_logit_bias = float(tail_blocked_logit_bias)

        self.online_net = PPOMapNetwork(
            input_size=self.global_signal_count,
            map_channels=self.map_channels,
            map_size=self.map_size,
            hidden1=hidden1,
            hidden2=hidden2,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=self.base_lr, eps=1e-5)

        self.total_updates = 0
        self.last_loss = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.last_value = 0.0
        self.last_approx_kl = 0.0
        self.kl_early_stops = 0

        # Мягкая защита лучшего прогресса по скользящему среднему полных эпизодов
        self.best_window_size = PPO_BEST_WINDOW
        self.best_min_episodes = PPO_BEST_MIN_EPISODES
        self._recent_episode_scores: Deque[int] = deque(maxlen=self.best_window_size)
        self.best_avg_score: float = 0.0
        self.best_policy_state: dict[str, torch.Tensor] | None = None
        self.best_policy_updates: int = 0
        self._episodes_since_best_check: int = 0
        self._degradation_streak: int = 0
        self.restores_from_best: int = 0

        self._rollout_signals: list[np.ndarray] = []
        self._rollout_maps: list[np.ndarray] = []
        self._rollout_actions: list[np.ndarray] = []
        self._rollout_logprobs: list[np.ndarray] = []
        self._rollout_rewards: list[np.ndarray] = []
        self._rollout_dones: list[np.ndarray] = []
        self._rollout_values: list[np.ndarray] = []

    def _masked_policy_logits(self, logits: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        valid_mask = valid_action_mask_from_signals(signals)
        adjusted = logits
        if self.tail_blocked_logit_bias != 0.0 and signals.shape[1] >= TAIL_BLOCKED_START_IDX + len(MODEL_ACTION_LABELS):
            tail_blocked = signals[:, TAIL_BLOCKED_START_IDX : TAIL_BLOCKED_START_IDX + len(MODEL_ACTION_LABELS)] > 0.5
            safe_and_tail_ok = valid_mask & (~tail_blocked)
            has_tail_ok = safe_and_tail_ok.any(dim=-1, keepdim=True)
            penalize_mask = has_tail_ok & valid_mask & tail_blocked
            adjusted = torch.where(penalize_mask, adjusted - self.tail_blocked_logit_bias, adjusted)
        return adjusted.masked_fill(~valid_mask, -1e9)

    def _current_lr(self) -> float:
        progress = min(1.0, float(self.total_updates) / 1800.0)
        return max(4e-5, self.base_lr * (1.0 - 0.90 * progress))

    def reset_progress_tracker(self, *, keep_best_snapshot: bool = False) -> None:
        self._recent_episode_scores.clear()
        self._episodes_since_best_check = 0
        self._degradation_streak = 0
        if not keep_best_snapshot:
            self.best_avg_score = 0.0
            self.best_policy_state = None
            self.best_policy_updates = 0

    def record_full_episode(self, score: int, reason: str = "") -> bool:
        """Обновляет трекер скользящего среднего только по честным полным эпизодам.

        Сохраняет снимок весов при новом лучшем среднем и мягко откатывает веса при
        устойчивой сильной просадке в длинной сессии.
        """
        self._recent_episode_scores.append(int(score))
        if len(self._recent_episode_scores) < self.best_min_episodes:
            return False

        avg_score = float(sum(self._recent_episode_scores)) / float(len(self._recent_episode_scores))
        if avg_score > self.best_avg_score:
            self.best_avg_score = avg_score
            self.best_policy_state = {
                k: v.detach().to(device="cpu", dtype=torch.float32).clone()
                for k, v in self.online_net.state_dict().items()
            }
            self.best_policy_updates = int(self.total_updates)
            self._episodes_since_best_check = 0
            self._degradation_streak = 0
            return True

        self._episodes_since_best_check += 1
        if (
            self.best_policy_state is not None
            and self.best_avg_score >= 15.0
            and self._episodes_since_best_check >= 15
        ):
            self._episodes_since_best_check = 0
            if avg_score < self.best_avg_score * 0.72 and (self.best_avg_score - avg_score) >= 6.0:
                self._degradation_streak += 1
                if self._degradation_streak >= 2:
                    self.restore_best_policy()
            else:
                self._degradation_streak = 0
        return False

    def restore_best_policy(self) -> bool:
        if self.best_policy_state is None:
            return False
        load_matching_state(self.online_net, self.best_policy_state)
        self.online_net.to(self.device)
        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=self._current_lr(), eps=1e-5)
        self.clear_rollout()
        self._recent_episode_scores.clear()
        self._episodes_since_best_check = 0
        self._degradation_streak = 0
        self.restores_from_best += 1
        return True

    @property
    def rollout_buffer_len(self) -> int:
        return len(self._rollout_actions) * self.num_envs

    @property
    def rollout_capacity(self) -> int:
        return self.rollout_steps * self.num_envs

    def clear_rollout(self) -> None:
        self._rollout_signals.clear()
        self._rollout_maps.clear()
        self._rollout_actions.clear()
        self._rollout_logprobs.clear()
        self._rollout_rewards.clear()
        self._rollout_dones.clear()
        self._rollout_values.clear()

    def move_to_device(self, device: torch.device) -> None:
        if device == self.device:
            return
        self.device = device
        self.online_net = self.online_net.to(self.device)
        move_optimizer_state(self.optimizer, self.device)

    @torch.inference_mode()
    def select_action(
        self,
        state: np.ndarray,
        *,
        ego_map: np.ndarray | None = None,
        training: bool = True,
        snake_length: int = 3,
    ) -> tuple[int, np.ndarray, bool, int]:
        self.online_net.eval()
        sig_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        map_t = (
            torch.as_tensor(ego_map, dtype=torch.float32, device=self.device).unsqueeze(0)
            if ego_map is not None
            else None
        )
        logits, value = self.online_net(sig_t, map_t)
        masked_logits = self._masked_policy_logits(logits, sig_t)
        probs = torch.softmax(masked_logits, dim=-1)
        greedy_action = int(torch.argmax(probs[0]).item())
        self.last_value = float(value[0].item())

        if training:
            dist = Categorical(probs=probs)
            action = int(dist.sample().item())
            was_random = action != greedy_action
        else:
            action = greedy_action
            was_random = False
        return action, probs[0].detach().cpu().numpy().astype(np.float32), was_random, greedy_action

    @torch.inference_mode()
    def select_best_action(
        self,
        state: np.ndarray,
        *,
        ego_map: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray, bool, int]:
        return self.select_action(state, ego_map=ego_map, training=False)

    @torch.inference_mode()
    def select_batch_actions(
        self,
        states: np.ndarray,
        ego_maps: np.ndarray,
        *,
        training: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.online_net.eval()
        sig_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        map_t = torch.as_tensor(ego_maps, dtype=torch.float32, device=self.device)
        logits, values = self.online_net(sig_t, map_t)
        masked_logits = self._masked_policy_logits(logits, sig_t)
        probs = torch.softmax(masked_logits, dim=-1)
        greedy = torch.argmax(probs, dim=-1)

        if training:
            dist = Categorical(probs=probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        else:
            actions = greedy
            dist = Categorical(probs=probs)
            log_probs = dist.log_prob(actions)

        return (
            actions.detach().cpu().numpy().astype(np.int64),
            log_probs.detach().cpu().numpy().astype(np.float32),
            values.detach().cpu().numpy().astype(np.float32),
            probs.detach().cpu().numpy().astype(np.float32),
            greedy.detach().cpu().numpy().astype(np.int64),
        )

    @torch.inference_mode()
    def estimate_values(self, states: np.ndarray, ego_maps: np.ndarray) -> np.ndarray:
        self.online_net.eval()
        sig_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        map_t = torch.as_tensor(ego_maps, dtype=torch.float32, device=self.device)
        _, values = self.online_net(sig_t, map_t)
        return values.detach().cpu().numpy().astype(np.float32)

    def store_rollout_step(
        self,
        states: np.ndarray,
        ego_maps: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self._rollout_signals.append(np.asarray(states, dtype=np.float32).copy())
        self._rollout_maps.append(np.asarray(ego_maps, dtype=np.float32).copy())
        self._rollout_actions.append(np.asarray(actions, dtype=np.int64).copy())
        self._rollout_logprobs.append(np.asarray(log_probs, dtype=np.float32).copy())
        self._rollout_rewards.append(np.asarray(rewards, dtype=np.float32).copy() * PPO_REWARD_SCALE)
        self._rollout_dones.append(np.asarray(dones, dtype=np.float32).copy())
        self._rollout_values.append(np.asarray(values, dtype=np.float32).copy())

    def ready_to_optimize(self) -> bool:
        return len(self._rollout_actions) >= self.rollout_steps

    def optimize(self, next_values: np.ndarray) -> float | None:
        if not self._rollout_actions:
            return None

        T = len(self._rollout_actions)
        N = len(next_values)
        rewards = np.stack(self._rollout_rewards, axis=0)      # (T, N)
        dones = np.stack(self._rollout_dones, axis=0)          # (T, N)
        values = np.stack(self._rollout_values, axis=0)        # (T, N)
        advantages = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)

        for t in range(T - 1, -1, -1):
            next_non_terminal = 1.0 - dones[t]
            next_val = next_values if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        b_signals = torch.as_tensor(np.concatenate(self._rollout_signals, axis=0), dtype=torch.float32, device=self.device)
        b_maps = torch.as_tensor(np.concatenate(self._rollout_maps, axis=0), dtype=torch.float32, device=self.device)
        b_actions = torch.as_tensor(np.concatenate(self._rollout_actions, axis=0), dtype=torch.int64, device=self.device)
        b_logprobs = torch.as_tensor(np.concatenate(self._rollout_logprobs, axis=0), dtype=torch.float32, device=self.device)
        b_values = torch.as_tensor(values.reshape(-1), dtype=torch.float32, device=self.device)
        b_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=self.device)

        total_samples = int(b_actions.shape[0])
        minibatch = max(32, min(self.minibatch_size, total_samples))
        if b_advantages.numel() > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1e-8)

        progress = min(1.0, float(self.total_updates) / 1800.0)
        cur_ent_coef = max(0.0008, self.ent_coef * (1.0 - 0.968 * progress))
        cur_lr = self._current_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = cur_lr

        self.online_net.train()
        total_loss_acc = 0.0
        pg_loss_acc = 0.0
        vf_loss_acc = 0.0
        ent_acc = 0.0
        kl_acc = 0.0
        batch_updates = 0
        early_stopped = False

        for _ in range(self.epochs):
            perm = torch.randperm(total_samples, device=self.device)
            epoch_kl_sum = 0.0
            epoch_batches = 0
            for start in range(0, total_samples, minibatch):
                idx = perm[start : start + minibatch]
                sig_mb = b_signals[idx]
                map_mb = b_maps[idx]
                logits, new_values = self.online_net(sig_mb, map_mb)
                masked_logits = self._masked_policy_logits(logits, sig_mb)
                probs = torch.softmax(masked_logits, dim=-1)
                dist = Categorical(probs=probs)
                new_logprob = dist.log_prob(b_actions[idx])
                entropy = dist.entropy().mean()

                log_ratio = new_logprob - b_logprobs[idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())

                if self.target_kl is not None and batch_updates > 0 and approx_kl > 1.5 * self.target_kl:
                    early_stopped = True
                    break

                adv_mb = b_advantages[idx]
                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                old_val_mb = b_values[idx]
                ret_mb = b_returns[idx]
                if self.vf_clip_coef > 0.0:
                    v_clipped = old_val_mb + torch.clamp(
                        new_values - old_val_mb,
                        -self.vf_clip_coef,
                        self.vf_clip_coef,
                    )
                    v_loss_unclipped = F.smooth_l1_loss(new_values, ret_mb, beta=1.0, reduction="none")
                    v_loss_clipped = F.smooth_l1_loss(v_clipped, ret_mb, beta=1.0, reduction="none")
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = F.smooth_l1_loss(new_values, ret_mb, beta=1.0)

                loss = pg_loss - cur_ent_coef * entropy + self.vf_coef * v_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), PPO_MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss_acc += float(loss.item())
                pg_loss_acc += float(pg_loss.item())
                vf_loss_acc += float(v_loss.item())
                ent_acc += float(entropy.item())
                kl_acc += approx_kl
                epoch_kl_sum += approx_kl
                epoch_batches += 1
                batch_updates += 1

            if early_stopped:
                break
            if self.target_kl is not None and epoch_batches > 0 and (epoch_kl_sum / epoch_batches) > self.target_kl:
                early_stopped = True
                break

        self.clear_rollout()
        if batch_updates == 0:
            return None

        if early_stopped:
            self.kl_early_stops += 1

        self.total_updates += 1
        self.last_loss = total_loss_acc / batch_updates
        self.last_policy_loss = pg_loss_acc / batch_updates
        self.last_value_loss = vf_loss_acc / batch_updates
        self.last_entropy = ent_acc / batch_updates
        self.last_approx_kl = kl_acc / batch_updates
        return self.last_loss

    @torch.inference_mode()
    def snapshot(self, state: np.ndarray, ego_map: np.ndarray | None = None) -> dict[str, np.ndarray]:
        self.online_net.eval()
        sig_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        map_t = (
            torch.as_tensor(ego_map, dtype=torch.float32, device=self.device).unsqueeze(0)
            if ego_map is not None
            else None
        )
        activations = self.online_net.forward_with_activations(sig_t, map_t)
        masked_logits = self._masked_policy_logits(activations["output"].unsqueeze(0), sig_t)
        probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
        self.last_value = float(activations["value"].item()) if activations["value"].numel() else 0.0
        result = {
            "input": activations["input"].detach().cpu().numpy().astype(np.float32),
            "map_features": activations["map_features"].detach().cpu().numpy().astype(np.float32),
            "hidden1": activations["hidden1"].detach().cpu().numpy().astype(np.float32),
            "hidden2": activations["hidden2"].detach().cpu().numpy().astype(np.float32),
            "output": probs.detach().cpu().numpy().astype(np.float32),
            "value": activations["value"].detach().cpu().numpy().astype(np.float32),
        }
        if "critic_hidden2" in activations:
            result["critic_hidden2"] = activations["critic_hidden2"].detach().cpu().numpy().astype(np.float32)
        return result

    def export_checkpoint(self) -> dict[str, object]:
        policy_cpu = {k: v.detach().to(device="cpu", dtype=torch.float32) for k, v in self.online_net.state_dict().items()}
        payload: dict[str, object] = {
            "policy_net": policy_cpu,
            "optimizer": self.optimizer.state_dict(),
            "total_updates": int(self.total_updates),
            "last_loss": float(self.last_loss),
            "last_policy_loss": float(self.last_policy_loss),
            "last_value_loss": float(self.last_value_loss),
            "last_entropy": float(self.last_entropy),
            "last_approx_kl": float(self.last_approx_kl),
            "kl_early_stops": int(self.kl_early_stops),
            "best_avg_score": float(self.best_avg_score),
            "best_policy_updates": int(self.best_policy_updates),
            "restores_from_best": int(self.restores_from_best),
        }
        if self.best_policy_state is not None:
            payload["best_policy_net"] = {
                k: v.detach().to(device="cpu", dtype=torch.float32)
                for k, v in self.best_policy_state.items()
            }
        return payload

    def import_checkpoint(self, payload: dict[str, object]) -> None:
        policy_state = payload.get("policy_net")
        load_matching_state(self.online_net, policy_state if isinstance(policy_state, dict) else None)
        optimizer_state = payload.get("optimizer")
        if isinstance(optimizer_state, dict):
            try:
                self.optimizer.load_state_dict(optimizer_state)
                move_optimizer_state(self.optimizer, self.device)
            except Exception:
                self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=self.base_lr, eps=1e-5)
        self.total_updates = int(payload.get("total_updates", 0) or 0)
        self.last_loss = float(payload.get("last_loss", 0.0) or 0.0)
        self.last_policy_loss = float(payload.get("last_policy_loss", 0.0) or 0.0)
        self.last_value_loss = float(payload.get("last_value_loss", 0.0) or 0.0)
        self.last_entropy = float(payload.get("last_entropy", 0.0) or 0.0)
        self.last_approx_kl = float(payload.get("last_approx_kl", 0.0) or 0.0)
        self.kl_early_stops = int(payload.get("kl_early_stops", 0) or 0)
        self.best_avg_score = float(payload.get("best_avg_score", 0.0) or 0.0)
        self.best_policy_updates = int(payload.get("best_policy_updates", 0) or 0)
        self.restores_from_best = int(payload.get("restores_from_best", 0) or 0)
        best_state = payload.get("best_policy_net")
        if isinstance(best_state, dict) and best_state:
            self.best_policy_state = {
                k: v.detach().to(device="cpu", dtype=torch.float32).clone()
                for k, v in best_state.items()
                if torch.is_tensor(v)
            }
        else:
            self.best_policy_state = None
        self._recent_episode_scores.clear()
        self._episodes_since_best_check = 0
        self._degradation_streak = 0
        self.clear_rollout()


class ParallelPPORunner:
    """Параллельный сборщик rollout-батчей по N средам SnakeEnv с встроенным Failure-Replay."""

    def __init__(
        self,
        num_envs: int,
        board_size: int,
        *,
        env_factory: Callable[[int], Any],
        action_labels: tuple[str, ...] = MODEL_ACTION_LABELS,
        replay_env_count: int | None = None,
        failure_min_score: int = PPO_FAILURE_MIN_SCORE,
        failure_buffer_maxlen: int = PPO_FAILURE_BUFFER_MAXLEN,
        failure_snapshot_interval: int = PPO_FAILURE_SNAPSHOT_INTERVAL,
        failure_rewind_target: int = PPO_FAILURE_REWIND_TARGET,
    ) -> None:
        self.num_envs = max(1, int(num_envs))
        self.board_size = int(board_size)
        self.env_factory = env_factory
        self.action_labels = action_labels
        if replay_env_count is None:
            if self.num_envs >= 8:
                self.replay_env_count = 2
            elif self.num_envs >= 4:
                self.replay_env_count = 1
            else:
                self.replay_env_count = 0
        else:
            self.replay_env_count = max(0, min(self.num_envs - 1, int(replay_env_count)))
        self.failure_min_score = int(failure_min_score)
        self.failure_buffer_maxlen = max(1, int(failure_buffer_maxlen))
        self.failure_snapshot_interval = max(1, int(failure_snapshot_interval))
        self.failure_rewind_target = max(1, int(failure_rewind_target))

        self.envs: list[Any] = [env_factory(self.board_size) for _ in range(self.num_envs)]
        self.episode_rewards: list[float] = [0.0 for _ in range(self.num_envs)]
        self.step_logs: list[Deque[dict[str, Any]]] = [deque(maxlen=120) for _ in range(self.num_envs)]
        self.replay_histories: list[list[dict[str, Any]]] = [[] for _ in range(self.num_envs)]
        self._state_snapshots: list[Deque[tuple[int, dict[str, object]]]] = [
            deque(maxlen=8) for _ in range(self.num_envs)
        ]
        self.is_replay_env: list[bool] = [False for _ in range(self.num_envs)]
        self.failure_replay_buffer: Deque[dict[str, object]] = deque(maxlen=self.failure_buffer_maxlen)
        self.failure_replays_completed = 0
        self.display_index = 0
        self.last_profile_ms: dict[str, float] = {}
        self.last_completed_env_payload: dict[str, Any] | None = None

    def _is_replay_slot(self, idx: int) -> bool:
        return self.replay_env_count > 0 and idx >= (self.num_envs - self.replay_env_count)

    def clear_failure_buffer(self) -> None:
        self.failure_replay_buffer.clear()
        for snap_q in self._state_snapshots:
            snap_q.clear()

    def reset_all(self, board_size: int | None = None) -> None:
        if board_size is not None:
            self.board_size = int(board_size)
        self.envs = [self.env_factory(self.board_size) for _ in range(self.num_envs)]
        self.episode_rewards = [0.0 for _ in range(self.num_envs)]
        self.step_logs = [deque(maxlen=120) for _ in range(self.num_envs)]
        self.replay_histories = [[] for _ in range(self.num_envs)]
        self._state_snapshots = [deque(maxlen=8) for _ in range(self.num_envs)]
        self.is_replay_env = [False for _ in range(self.num_envs)]
        self.failure_replay_buffer.clear()
        self.failure_replays_completed = 0
        self.display_index = 0
        self.last_completed_env_payload = None

    def leader_env(self) -> Any:
        return self.envs[self.display_index]

    def update_leader(self) -> int:
        best_idx = 0
        best_key = (-1, -1, -1)
        for idx, env in enumerate(self.envs):
            honest_priority = 0 if self.is_replay_env[idx] else 1
            key = (honest_priority, int(env.score), int(env.step_count))
            if key > best_key:
                best_key = key
                best_idx = idx
        self.display_index = best_idx
        return best_idx

    def _pick_failure_rewind_snapshot(self, idx: int, death_step: int) -> dict[str, object] | None:
        snaps = self._state_snapshots[idx]
        if not snaps:
            return None
        target_step = max(0, death_step - self.failure_rewind_target)
        best_state: dict[str, object] | None = None
        best_dist = 10**9
        for snap_step, state in snaps:
            if snap_step >= death_step - 2:
                continue
            if bool(state.get("done", False)):
                continue
            if int(state.get("board_size", self.board_size)) != self.board_size:
                continue
            dist = abs(snap_step - target_step)
            if dist < best_dist:
                best_dist = dist
                best_state = state
        return best_state

    def _reset_env_slot(self, idx: int) -> None:
        env = self.envs[idx]
        self.episode_rewards[idx] = 0.0
        self.step_logs[idx].clear()
        self.replay_histories[idx].clear()
        self._state_snapshots[idx].clear()
        if self._is_replay_slot(idx) and len(self.failure_replay_buffer) > 0:
            saved_state = random.choice(tuple(self.failure_replay_buffer))
            env.load_state(saved_state)
            if env.done or int(env.board_size) != self.board_size:
                env.reset()
                self.is_replay_env[idx] = False
            else:
                self.is_replay_env[idx] = True
        else:
            env.reset()
            self.is_replay_env[idx] = False

    def _extract_signals(self, env: Any, agent: PPOAgent) -> np.ndarray:
        if agent.global_signal_count == len(PPO_SIGNAL_LABELS) and hasattr(env, "get_ppo_signals"):
            return env.get_ppo_signals()
        return env.get_extra_signals()

    def step_alive(self, agent: PPOAgent, *, record_replays: bool = False) -> list[dict[str, Any]]:
        states = np.stack([self._extract_signals(env, agent) for env in self.envs], axis=0).astype(np.float32)
        ego_maps = np.stack(
            [env.get_egocentric_map(agent.map_size) for env in self.envs],
            axis=0,
        ).astype(np.float32)
        actions, log_probs, values, probs_np, greedy_actions = agent.select_batch_actions(
            states,
            ego_maps,
            training=True,
        )

        step_rewards = np.zeros(self.num_envs, dtype=np.float32)
        step_dones = np.zeros(self.num_envs, dtype=np.float32)
        completed: list[dict[str, Any]] = []

        for idx, env in enumerate(self.envs):
            is_replay = self.is_replay_env[idx]
            if not is_replay and (env.step_count % self.failure_snapshot_interval == 0):
                self._state_snapshots[idx].append((int(env.step_count), env.export_state()))

            act = int(actions[idx])
            was_random = act != int(greedy_actions[idx])
            state = states[idx]
            pre_step_risks = evaluate_pre_step_risks(env, state, act)
            is_endgame = float(state[ENDGAME_IDX]) > 0.5
            is_pocket_split = float(state[POCKET_SPLIT_START_IDX + act]) > 0.5
            is_pocket_merge = float(state[POCKET_MERGE_START_IDX + act]) > 0.5
            food_reachable = float(state[FOOD_REACHABLE_IDX]) > 0.5
            old_food_dist = abs(env.snake[0][0] - env.food[0]) + abs(env.snake[0][1] - env.food[1])

            abs_dir = relative_to_absolute(env.direction, act)
            _, reward, done, info = env.step(abs_dir)
            reason = str(info["reason"])
            shaped_reward, extra_penalties = apply_step_risk_shaping(
                reward,
                reason,
                len(env.snake),
                pre_step_risks,
                is_endgame=is_endgame,
                is_pocket_split=is_pocket_split,
                is_pocket_merge=is_pocket_merge,
                endgame_split_penalty=ENDGAME_SPLIT_PENALTY,
                endgame_merge_bonus=ENDGAME_MERGE_BONUS,
            )
            if not done and reason == "step" and food_reachable and not is_endgame:
                new_food_dist = abs(env.snake[0][0] - env.food[0]) + abs(env.snake[0][1] - env.food[1])
                if new_food_dist < old_food_dist:
                    shaped_reward += 0.03
                elif new_food_dist > old_food_dist:
                    shaped_reward -= 0.04

            all_penalties = [str(item) for item in info.get("penalty_reasons", [])] + extra_penalties
            step_rewards[idx] = float(shaped_reward)
            step_dones[idx] = 1.0 if done else 0.0
            self.episode_rewards[idx] += float(shaped_reward)

            step_num = int(info.get("steps", env.step_count))
            if not is_replay:
                if step_num == 1:
                    self.step_logs[idx].clear()
                    self.replay_histories[idx].clear()
                self.step_logs[idx].append({
                    "step": step_num,
                    "reward": float(shaped_reward),
                    "train_reward": float(shaped_reward),
                    "action": act,
                    "action_label": self.action_labels[act] if 0 <= act < len(self.action_labels) else "-",
                    "reason": reason,
                    "penalties": list(all_penalties),
                    "random": bool(was_random),
                    "steps_without_food": int(env.steps_without_food),
                    "max_steps_without_food": int(env.max_steps_without_food),
                })
                if record_replays:
                    self.replay_histories[idx].append({
                        "step": int(env.step_count),
                        "snake": [[int(c[0]), int(c[1])] for c in env.snake],
                        "food": [int(env.food[0]), int(env.food[1])],
                        "direction": int(env.direction),
                        "action": act,
                        "q_values": [round(float(v), 3) for v in probs_np[idx]],
                        "was_random": bool(was_random),
                        "reward": float(shaped_reward),
                        "reason": reason,
                        "penalties": list(all_penalties),
                        "steps_without_food": int(env.steps_without_food),
                    })
                    if len(self.replay_histories[idx]) > 500:
                        self.replay_histories[idx].pop(0)

            if done:
                if is_replay:
                    self.failure_replays_completed += 1
                else:
                    if reason != "victory" and int(env.score) >= self.failure_min_score:
                        rewind_state = self._pick_failure_rewind_snapshot(idx, int(env.step_count))
                        if rewind_state is not None:
                            self.failure_replay_buffer.append(rewind_state)

                    finished_snapshot = {
                        "index": idx,
                        "score": int(env.score),
                        "step_count": int(env.step_count),
                        "steps_without_food": int(env.steps_without_food),
                        "direction": int(env.direction),
                        "board_size": int(env.board_size),
                        "snake": list(env.snake),
                        "food": tuple(env.food),
                        "eaten_food_trail": list(env.eaten_food_trail),
                        "reason": reason,
                        "reward": float(self.episode_rewards[idx]),
                        "penalty_reasons": list(all_penalties),
                        "step_log": list(self.step_logs[idx]),
                        "replay_steps": list(self.replay_histories[idx]) if record_replays else [],
                        "loss": float(agent.last_loss),
                    }
                    completed.append(finished_snapshot)
                    self.last_completed_env_payload = finished_snapshot
                self._reset_env_slot(idx)

        agent.store_rollout_step(
            states,
            ego_maps,
            actions,
            log_probs,
            step_rewards,
            step_dones,
            values,
        )
        if agent.ready_to_optimize():
            next_states = np.stack([self._extract_signals(env, agent) for env in self.envs], axis=0).astype(np.float32)
            next_maps = np.stack(
                [env.get_egocentric_map(agent.map_size) for env in self.envs],
                axis=0,
            ).astype(np.float32)
            next_values = agent.estimate_values(next_states, next_maps)
            agent.optimize(next_values)

        return completed
