from __future__ import annotations

from collections import deque
import random
import threading
import time
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
    HUNGER_CRITICAL_IDX,
    HUNGER_HIGH_IDX,
    MODEL_ACTION_LABELS,
    POCKET_MERGE_START_IDX,
    POCKET_SPLIT_START_IDX,
    PPO_CONV1_CHANNELS,
    PPO_CONV2_CHANNELS,
    PPO_CRITICAL_HUNGER_PENALTY,
    PPO_FOOD_STEP_AWAY_PENALTY,
    PPO_FOOD_STEP_CLOSER_BONUS,
    PPO_HIDDEN_LAYER_1,
    PPO_HIDDEN_LAYER_2,
    PPO_MAP_CHANNELS,
    PPO_MAP_FEAT_SIZE,
    PPO_MAP_SIZE,
    PPO_SIGNAL_LABELS,
    PPO_VICTORY_EXTRA_BONUS,
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

PPO_NUM_ENVS = 16
PPO_ROLLOUT_STEPS = 66
PPO_POLISH_ROLLOUT_STEPS = 132
PPO_GAMMA = 0.995
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_COEF = 0.2
PPO_VF_CLIP_COEF = 0.25
PPO_TARGET_KL = 0.015
PPO_ENT_COEF = 0.025
PPO_ENT_MIN = 0.0002
PPO_VF_COEF = 0.5
PPO_LR = 4e-4
PPO_LR_MIN = 5e-6
PPO_SCHEDULE_UPDATES = 3600.0
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 256
PPO_POLISH_MINIBATCH_SIZE = 512
PPO_MAX_GRAD_NORM = 0.5
PPO_REWARD_SCALE = 0.1
PPO_BEST_WINDOW = 30
PPO_BEST_MIN_EPISODES = 20
PPO_FAILURE_MIN_SCORE = 25
PPO_FAILURE_MIN_FILL = 0.25
PPO_FAILURE_ENDGAME_FILL = 0.80
PPO_FAILURE_BUFFER_MAXLEN = 64
PPO_FAILURE_ENDGAME_MAXLEN = 32
PPO_FAILURE_SNAPSHOT_INTERVAL = 5
PPO_FAILURE_REWIND_TARGET = 14
PPO_FAILURE_STARVATION_REWIND = 22
PPO_TAIL_BLOCKED_LOGIT_BIAS = 0.5
PPO_TAIL_BIAS_ANNEAL_START_FILL = 0.70
PPO_TAIL_BIAS_ANNEAL_END_FILL = 0.90
PPO_EMA_TAU = 0.988
PPO_EMA_POLISH_TAU = 0.978
PPO_EMA_START_FILL = 0.90
PPO_EMA_MIN_UPDATES = 5
PPO_HUNGRY_SAMPLE_MIN_PROB = 0.05
PPO_CRIT_HUNGER_TEMP = 2.0


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
        conv1_channels: int = PPO_CONV1_CHANNELS,
        conv2_channels: int = PPO_CONV2_CHANNELS,
        map_feat_size: int = PPO_MAP_FEAT_SIZE,
        hidden1: int = PPO_HIDDEN_LAYER_1,
        hidden2: int = PPO_HIDDEN_LAYER_2,
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
        self.conv1_channels = int(conv1_channels)
        self.conv2_channels = int(conv2_channels)
        self.map_feat_size = int(map_feat_size)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
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
            conv1_channels=self.conv1_channels,
            conv2_channels=self.conv2_channels,
            map_feat_size=self.map_feat_size,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
        ).to(self.device)
        # Лёгкая копия весов для неблокирующего показа живой игры во время optimize()
        self.inference_net = PPOMapNetwork(
            input_size=self.global_signal_count,
            map_channels=self.map_channels,
            map_size=self.map_size,
            conv1_channels=self.conv1_channels,
            conv2_channels=self.conv2_channels,
            map_feat_size=self.map_feat_size,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
        ).to(self.device)
        self._inference_lock = threading.Lock()
        self.ema_tau: float = float(PPO_EMA_TAU)
        self.ema_updates: int = 0
        self.ema_policy_state: dict[str, torch.Tensor] | None = None
        self.sync_inference_net()
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
        self._recent_fill_ratios: Deque[float] = deque(maxlen=self.best_window_size)
        self._last_recent_avg_fill: float = 0.0
        self.best_avg_score: float = 0.0
        self.best_fill_ratio: float = 0.0
        self.best_fill_by_board: dict[int, float] = {}
        self.best_policy_state: dict[str, torch.Tensor] | None = None
        self.best_policy_updates: int = 0
        self._episodes_since_best_check: int = 0
        self._degradation_streak: int = 0
        self.restores_from_best: int = 0
        self._board_adapt_boost: float = 0.0
        self._critic_warmup_updates: int = 0
        self._post_restore_cooldown: int = 0

        self._rollout_signals: list[np.ndarray] = []
        self._rollout_maps: list[np.ndarray] = []
        self._rollout_actions: list[np.ndarray] = []
        self._rollout_logprobs: list[np.ndarray] = []
        self._rollout_rewards: list[np.ndarray] = []
        self._rollout_dones: list[np.ndarray] = []
        self._rollout_values: list[np.ndarray] = []

    def _global_best_fill(self) -> float:
        """Максимальная достигнутая заполненность по всем размерам карт (монотонный индикатор зрелости политики)."""
        best = float(self.best_fill_ratio)
        if self.best_fill_by_board:
            best = max(best, max(self.best_fill_by_board.values()))
        return best

    def effective_tail_bias(self) -> float:
        """Плавно зануляет мягкий сдвиг логита tail_blocked (0.5 -> 0.0) при выходе в эндшпиль."""
        base = self.tail_blocked_logit_bias
        if base <= 0.0:
            return 0.0
        fill = self._global_best_fill()
        if fill <= PPO_TAIL_BIAS_ANNEAL_START_FILL:
            return base
        if fill >= PPO_TAIL_BIAS_ANNEAL_END_FILL:
            return 0.0
        span = max(1e-6, PPO_TAIL_BIAS_ANNEAL_END_FILL - PPO_TAIL_BIAS_ANNEAL_START_FILL)
        frac = (fill - PPO_TAIL_BIAS_ANNEAL_START_FILL) / span
        return base * (1.0 - frac)

    def _masked_policy_logits(self, logits: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        valid_mask = valid_action_mask_from_signals(signals)
        adjusted = logits
        eff_bias = self.effective_tail_bias()
        if eff_bias != 0.0 and signals.shape[1] >= TAIL_BLOCKED_START_IDX + len(MODEL_ACTION_LABELS):
            tail_blocked = signals[:, TAIL_BLOCKED_START_IDX : TAIL_BLOCKED_START_IDX + len(MODEL_ACTION_LABELS)] > 0.5
            safe_and_tail_ok = valid_mask & (~tail_blocked)
            has_tail_ok = safe_and_tail_ok.any(dim=-1, keepdim=True)
            penalize_mask = has_tail_ok & valid_mask & tail_blocked
            adjusted = torch.where(penalize_mask, adjusted - eff_bias, adjusted)
        return adjusted.masked_fill(~valid_mask, -1e9)

    def _current_lr(self) -> float:
        progress = min(1.0, float(self.total_updates) / PPO_SCHEDULE_UPDATES)
        cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        lr = PPO_LR_MIN + (self.base_lr - PPO_LR_MIN) * cosine
        g_fill = self._global_best_fill()
        if g_fill >= 0.88:
            # Плавное охлаждение LR в эндшпиле (8.8e-5 при 0.88 -> 1.2e-5 при 0.95 -> 6e-6 при 0.985)
            if g_fill >= 0.95:
                ultra_frac = min(1.0, (g_fill - 0.95) / 0.04)
                ultra_cap = 1.6e-5 * (1.0 - ultra_frac) + 6.0e-6 * ultra_frac
                lr = min(lr, max(PPO_LR_MIN, ultra_cap))
            else:
                lr = min(lr, max(PPO_LR_MIN, self.base_lr * 0.22))
        if self._post_restore_cooldown > 0:
            lr *= 0.6
        if self._board_adapt_boost > 0.0:
            # Мягкий transfer LR при переходе на новую карту: не разрушает зрелую политику
            lr = min(max(lr, 6e-5) + self._board_adapt_boost * 4e-5, 1.3e-4 if g_fill >= 0.75 else self.base_lr)
        return max(PPO_LR_MIN, lr)

    def _current_clip_coef(self) -> float:
        if self._board_adapt_boost > 0.0:
            return self.clip_coef
        g_fill = self._global_best_fill()
        if g_fill >= 0.97:
            return min(self.clip_coef, 0.10)
        if g_fill >= 0.94:
            return min(self.clip_coef, 0.12)
        return self.clip_coef

    def _current_target_kl(self) -> float | None:
        if self.target_kl is None or self._board_adapt_boost > 0.0:
            return self.target_kl
        g_fill = self._global_best_fill()
        if g_fill >= 0.97:
            return min(self.target_kl, 0.008)
        if g_fill >= 0.94:
            return min(self.target_kl, 0.010)
        return self.target_kl

    def _current_ent_coef(self) -> float:
        progress = min(1.0, float(self.total_updates) / PPO_SCHEDULE_UPDATES)
        cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        ent = PPO_ENT_MIN + (self.ent_coef - PPO_ENT_MIN) * cosine
        g_fill = self._global_best_fill()
        if g_fill >= 0.88:
            polish_frac = min(1.0, (g_fill - 0.88) / 0.10)
            polish_cap = max(PPO_ENT_MIN, 0.0012 * (1.0 - 0.80 * polish_frac))
            ent = min(ent, polish_cap)
        if self._board_adapt_boost > 0.0:
            # Небольшое оживление энтропии (до ~0.0025), без сброса к 0.025
            ent = min(0.003, max(ent, PPO_ENT_MIN + self._board_adapt_boost * 0.0022)) if g_fill >= 0.75 else max(ent, PPO_ENT_MIN + self._board_adapt_boost * 0.004)
        return max(PPO_ENT_MIN, ent)

    def reset_progress_tracker(self, *, keep_best_snapshot: bool = False) -> None:
        self._recent_episode_scores.clear()
        self._recent_fill_ratios.clear()
        self._last_recent_avg_fill = 0.0
        self._episodes_since_best_check = 0
        self._degradation_streak = 0
        if not keep_best_snapshot:
            self.best_avg_score = 0.0
            self.best_fill_ratio = 0.0
            self.best_fill_by_board.clear()
            self.best_policy_state = None
            self.best_policy_updates = 0
            self._board_adapt_boost = 0.0
            self._critic_warmup_updates = 0
            self._post_restore_cooldown = 0
            self.ema_policy_state = None
            self.ema_updates = 0
            self.sync_inference_net()

    def on_board_size_changed(self, new_board_size: int) -> None:
        """Сбрасывает локальное окно метрик и мягко адаптирует критик/расписание к новому размеру поля."""
        bs = int(new_board_size)
        prev_global_fill = self._global_best_fill()
        self._recent_episode_scores.clear()
        self._recent_fill_ratios.clear()
        self._last_recent_avg_fill = 0.0
        self._episodes_since_best_check = 0
        self._degradation_streak = 0
        self._post_restore_cooldown = 0
        self.best_fill_ratio = float(self.best_fill_by_board.get(bs, 0.0))
        self.ema_policy_state = None
        self.ema_updates = 0
        self.sync_inference_net()
        if self.best_fill_ratio < 0.70 and (self.total_updates >= 200 or prev_global_fill >= 0.70):
            self._board_adapt_boost = 1.0
            self._critic_warmup_updates = 6 if prev_global_fill >= 0.70 else 0
        else:
            self._board_adapt_boost = 0.0
            self._critic_warmup_updates = 0

    def record_full_episode(self, score: int, reason: str = "", board_size: int = 10) -> bool:
        """Обновляет трекер скользящего среднего только по честным полным эпизодам.

        Отслеживает универсальную заполненность поля (fill_ratio) для каждого размера карты,
        сохраняет снимок весов (приоритетно сглаженный EMA) при новом лучшем среднем и
        жёстко защищает эндшпиль от просадки (95 -> 85) в длинной сессии.
        """
        bs = max(3, int(board_size))
        fill_ratio = min(1.0, float(int(score) + 3) / float(bs * bs))
        self._recent_episode_scores.append(int(score))
        self._recent_fill_ratios.append(fill_ratio)
        if len(self._recent_episode_scores) < self.best_min_episodes:
            return False

        avg_score = float(sum(self._recent_episode_scores)) / float(len(self._recent_episode_scores))
        avg_fill = float(sum(self._recent_fill_ratios)) / float(len(self._recent_fill_ratios))
        self._last_recent_avg_fill = avg_fill
        prev_board_best = float(self.best_fill_by_board.get(bs, 0.0))

        improved = avg_fill > prev_board_best or (bs == 10 and avg_score > self.best_avg_score)
        at_ceiling_refresh = (
            prev_board_best >= 0.995
            and avg_fill >= prev_board_best - 0.002
            and (int(self.total_updates) - int(self.best_policy_updates)) >= 50
        )
        if improved or at_ceiling_refresh:
            self.best_avg_score = max(self.best_avg_score, avg_score) if bs == 10 else avg_score
            self.best_fill_ratio = max(prev_board_best, avg_fill)
            self.best_fill_by_board[bs] = self.best_fill_ratio
            source_sd = self.ema_policy_state if self.ema_active else self.online_net.state_dict()
            self.best_policy_state = {
                k: v.detach().to(device="cpu", dtype=torch.float32).clone()
                for k, v in source_sd.items()
            }
            self.best_policy_updates = int(self.total_updates)
            self._episodes_since_best_check = 0
            self._degradation_streak = 0
            return True

        self._episodes_since_best_check += 1
        if (
            self.best_policy_state is not None
            and self.best_fill_ratio >= 0.18
            and self._episodes_since_best_check >= 15
        ):
            self._episodes_since_best_check = 0
            # В эндшпиле (best_fill >= 0.92) ловим дрейф рано, но на потолке (>=0.98) даём 3 окна от шума стохастики
            if self.best_fill_ratio >= 0.98:
                is_degraded = (self.best_fill_ratio - avg_fill) >= 0.075
                need_streak = 3
            elif self.best_fill_ratio >= 0.92:
                is_degraded = (self.best_fill_ratio - avg_fill) >= 0.065
                need_streak = 2
            else:
                is_degraded = avg_fill < self.best_fill_ratio * 0.72 and (self.best_fill_ratio - avg_fill) >= 0.06
                need_streak = 2
            if is_degraded:
                self._degradation_streak += 1
                if self._degradation_streak >= need_streak:
                    self.restore_best_policy()
            else:
                self._degradation_streak = 0
        return False

    @property
    def ema_active(self) -> bool:
        return self.ema_policy_state is not None and self.ema_updates >= PPO_EMA_MIN_UPDATES

    def _update_ema_policy(self) -> None:
        """Quality-Gated SWA->EMA: сглаживает веса на зрелой фазе и замораживает EMA при просадке online_net."""
        if self._global_best_fill() < PPO_EMA_START_FILL:
            return
        # Если online_net заметно просела относительно пика — не отравляем EMA деградировавшими весами!
        if (
            self.ema_active
            and len(self._recent_fill_ratios) >= self.best_min_episodes
            and self.best_fill_ratio >= PPO_EMA_START_FILL
            and self._last_recent_avg_fill < (self.best_fill_ratio - 0.035)
        ):
            return
        with torch.no_grad():
            sd = self.online_net.state_dict()
            if self.ema_policy_state is None or self.ema_updates <= 0:
                self.ema_policy_state = {k: v.detach().to(device=self.device).clone() for k, v in sd.items()}
                self.ema_updates = 1
                return
            # Точный SWA-разгон (k/(k+1)) на первых ~50-80 обновлениях, затем плавный Polyak EMA
            target_tau = PPO_EMA_POLISH_TAU if self.effective_rollout_steps() >= PPO_POLISH_ROLLOUT_STEPS else self.ema_tau
            eff_tau = min(target_tau, float(self.ema_updates) / float(self.ema_updates + 1))
            alpha = 1.0 - eff_tau
            for k, v in sd.items():
                ema_v = self.ema_policy_state.get(k)
                if ema_v is None or ema_v.shape != v.shape:
                    self.ema_policy_state[k] = v.detach().to(device=self.device).clone()
                elif v.is_floating_point():
                    ema_v.mul_(eff_tau).add_(v.detach().to(device=self.device, dtype=ema_v.dtype), alpha=alpha)
                else:
                    self.ema_policy_state[k] = v.detach().to(device=self.device).clone()
            self.ema_updates += 1

    def sync_inference_net(self) -> None:
        source_state = (
            self.ema_policy_state
            if self.ema_active
            else self.online_net.state_dict()
        )
        state = {k: v.detach().to(device=self.device).clone() for k, v in source_state.items()}
        with self._inference_lock:
            self.inference_net.load_state_dict(state)
            self.inference_net.eval()

    def _pick_eval_action(
        self,
        probs_np: np.ndarray,
        sig_t: torch.Tensor,
        masked_logits: torch.Tensor | None = None,
    ) -> tuple[int, bool, int]:
        greedy_action = int(np.argmax(probs_np))
        # При критическом голоде (hunger_critical=1) мягко повышаем температуру T=2.0 на логитах самой
        # обученной политики π (p >= 0.05), чтобы выходить из детерминированных циклов без лишних столкновений.
        if sig_t.shape[1] > HUNGER_CRITICAL_IDX and float(sig_t[0, HUNGER_CRITICAL_IDX].item()) > 0.5:
            if masked_logits is not None and PPO_CRIT_HUNGER_TEMP > 1.0:
                scaled_np = (
                    torch.softmax(masked_logits / float(PPO_CRIT_HUNGER_TEMP), dim=-1)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                scaled_np = probs_np.astype(np.float64)
            filtered = np.where(scaled_np >= PPO_HUNGRY_SAMPLE_MIN_PROB, scaled_np, 0.0)
            total = float(filtered.sum())
            if total > 0.0:
                p_norm = filtered / total
                if int((p_norm > 0.0).sum()) >= 2:
                    chosen = int(np.random.choice(len(p_norm), p=p_norm))
                    return chosen, (chosen != greedy_action), greedy_action
        return greedy_action, False, greedy_action

    @torch.inference_mode()
    def select_showcase_action(
        self,
        state: np.ndarray,
        *,
        ego_map: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray, bool, int]:
        """Неблокирующий выбор хода для плавного показа поля и инференса EMA."""
        sig_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        map_t = (
            torch.as_tensor(ego_map, dtype=torch.float32, device=self.device).unsqueeze(0)
            if ego_map is not None
            else None
        )
        with self._inference_lock:
            logits, value = self.inference_net(sig_t, map_t)
        masked_logits = self._masked_policy_logits(logits, sig_t)
        probs = torch.softmax(masked_logits, dim=-1)
        probs_np = probs[0].detach().cpu().numpy().astype(np.float32)
        self.last_value = float(value[0].item())
        action, was_random, greedy_action = self._pick_eval_action(probs_np, sig_t, masked_logits)
        return action, probs_np, was_random, greedy_action

    def restore_best_policy(self) -> bool:
        if self.best_policy_state is None:
            return False
        load_matching_state(self.online_net, self.best_policy_state)
        self.online_net.to(self.device)
        # Сохраняем лучший снимок как базу EMA (не теряем сглаженный пик при откате)
        if self._global_best_fill() >= PPO_EMA_START_FILL:
            self.ema_policy_state = {
                k: v.detach().to(device=self.device).clone()
                for k, v in self.online_net.state_dict().items()
            }
            self.ema_updates = max(PPO_EMA_MIN_UPDATES, 10)
        else:
            self.ema_policy_state = None
            self.ema_updates = 0
        self._post_restore_cooldown = 40
        self.sync_inference_net()
        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=self._current_lr(), eps=1e-5)
        self.clear_rollout()
        self._recent_episode_scores.clear()
        self._recent_fill_ratios.clear()
        self._last_recent_avg_fill = self.best_fill_ratio
        self._episodes_since_best_check = 0
        self._degradation_streak = 0
        self.restores_from_best += 1
        return True

    def effective_rollout_steps(self) -> int:
        """В зрелой фазе полировки (fill >= 0.90) удваивает размер батча (88 -> 176) для снижения дисперсии PPO."""
        if (
            self.rollout_steps == PPO_ROLLOUT_STEPS
            and self._global_best_fill() >= PPO_EMA_START_FILL
            and self._board_adapt_boost <= 0.0
        ):
            return PPO_POLISH_ROLLOUT_STEPS
        return self.rollout_steps

    @property
    def rollout_buffer_len(self) -> int:
        return len(self._rollout_actions) * self.num_envs

    @property
    def rollout_capacity(self) -> int:
        return self.effective_rollout_steps() * self.num_envs

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
        if self.ema_policy_state is not None:
            self.ema_policy_state = {k: v.to(self.device) for k, v in self.ema_policy_state.items()}
        with self._inference_lock:
            self.inference_net = self.inference_net.to(self.device)
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
            dist = Categorical(probs=probs, validate_args=False)
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
        if self.ema_active:
            return self.select_showcase_action(state, ego_map=ego_map)
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
        probs_np = probs[0].detach().cpu().numpy().astype(np.float32)
        self.last_value = float(value[0].item())
        action, was_random, greedy_action = self._pick_eval_action(probs_np, sig_t, masked_logits)
        return action, probs_np, was_random, greedy_action

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
            dist = Categorical(probs=probs, validate_args=False)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        else:
            actions = greedy
            dist = Categorical(probs=probs, validate_args=False)
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
        return len(self._rollout_actions) >= self.effective_rollout_steps()

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
        base_mb = PPO_POLISH_MINIBATCH_SIZE if (self.minibatch_size == PPO_MINIBATCH_SIZE and total_samples >= 2000) else self.minibatch_size
        minibatch = max(32, min(base_mb, total_samples))
        if b_advantages.numel() > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1e-8)

        cur_ent_coef = self._current_ent_coef()
        cur_lr = self._current_lr()
        cur_clip = self._current_clip_coef()
        cur_target_kl = self._current_target_kl()
        for pg in self.optimizer.param_groups:
            pg["lr"] = cur_lr

        # Адаптивный масштаб клиппинга критика для переносимости на большие карты (15x15..35x35)
        ret_std = float(b_returns.std(unbiased=False).item()) if b_returns.numel() > 1 else 1.0
        eff_vf_clip = max(self.vf_clip_coef, ret_std * 0.5) if self.vf_clip_coef > 0.0 else 0.0

        critic_only = self._critic_warmup_updates > 0

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
                dist = Categorical(probs=probs, validate_args=False)
                new_logprob = dist.log_prob(b_actions[idx])
                entropy = dist.entropy().mean()

                log_ratio = new_logprob - b_logprobs[idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())

                if not critic_only and cur_target_kl is not None and batch_updates > 0 and approx_kl > 1.5 * cur_target_kl:
                    early_stopped = True
                    break

                adv_mb = b_advantages[idx]
                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * torch.clamp(ratio, 1.0 - cur_clip, 1.0 + cur_clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                old_val_mb = b_values[idx]
                ret_mb = b_returns[idx]
                if eff_vf_clip > 0.0 and not critic_only:
                    v_clipped = old_val_mb + torch.clamp(
                        new_values - old_val_mb,
                        -eff_vf_clip,
                        eff_vf_clip,
                    )
                    v_loss_unclipped = F.smooth_l1_loss(new_values, ret_mb, beta=1.0, reduction="none")
                    v_loss_clipped = F.smooth_l1_loss(v_clipped, ret_mb, beta=1.0, reduction="none")
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = F.smooth_l1_loss(new_values, ret_mb, beta=1.0)

                if critic_only:
                    loss = self.vf_coef * v_loss
                else:
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
            if not critic_only and cur_target_kl is not None and epoch_batches > 0 and (epoch_kl_sum / epoch_batches) > cur_target_kl:
                early_stopped = True
                break

        self.clear_rollout()
        if batch_updates == 0:
            return None

        if early_stopped:
            self.kl_early_stops += 1

        self.total_updates += 1
        if self._post_restore_cooldown > 0:
            self._post_restore_cooldown -= 1
        if self._critic_warmup_updates > 0:
            self._critic_warmup_updates -= 1
        else:
            self._update_ema_policy()
        if self._board_adapt_boost > 0.0:
            self._board_adapt_boost = max(0.0, self._board_adapt_boost - (1.0 / 120.0))
        self.last_loss = total_loss_acc / batch_updates
        self.last_policy_loss = pg_loss_acc / batch_updates
        self.last_value_loss = vf_loss_acc / batch_updates
        self.last_entropy = ent_acc / batch_updates
        self.last_approx_kl = kl_acc / batch_updates
        self.sync_inference_net()
        return self.last_loss

    @torch.inference_mode()
    def snapshot(
        self,
        state: np.ndarray,
        ego_map: np.ndarray | None = None,
        *,
        use_inference_net: bool = False,
    ) -> dict[str, np.ndarray]:
        if use_inference_net:
            with self._inference_lock, torch.no_grad():
                net = self.inference_net
                net.eval()
                sig_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                map_t = (
                    torch.as_tensor(ego_map, dtype=torch.float32, device=self.device).unsqueeze(0)
                    if ego_map is not None
                    else None
                )
                activations = net.forward_with_activations(sig_t, map_t)
                masked_logits = self._masked_policy_logits(activations["output"].unsqueeze(0), sig_t)
                probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
                val_scalar = float(activations["value"].item()) if activations["value"].numel() else 0.0
                self.last_value = val_scalar
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

        with torch.no_grad():
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
            "best_fill_ratio": float(self.best_fill_ratio),
            "best_fill_by_board": {str(k): float(v) for k, v in self.best_fill_by_board.items()},
            "best_policy_updates": int(self.best_policy_updates),
            "restores_from_best": int(self.restores_from_best),
            "ema_updates": int(self.ema_updates),
        }
        if self.ema_policy_state is not None and self.ema_updates > 0:
            payload["ema_policy_net"] = {
                k: v.detach().to(device="cpu", dtype=torch.float32)
                for k, v in self.ema_policy_state.items()
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
        self.best_fill_ratio = float(payload.get("best_fill_ratio", 0.0) or 0.0)
        raw_bfb = payload.get("best_fill_by_board", {})
        self.best_fill_by_board = {}
        if isinstance(raw_bfb, dict):
            for k, v in raw_bfb.items():
                try:
                    self.best_fill_by_board[int(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        if self.best_fill_ratio <= 0.0 and self.best_avg_score > 0.0:
            self.best_fill_ratio = min(1.0, (self.best_avg_score + 3.0) / 100.0)
            self.best_fill_by_board.setdefault(10, self.best_fill_ratio)
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
        ema_state = payload.get("ema_policy_net")
        ema_updates = int(payload.get("ema_updates", 0) or 0)
        if isinstance(ema_state, dict) and ema_state and ema_updates > 0:
            self.ema_policy_state = {
                k: v.detach().to(device=self.device, dtype=torch.float32).clone()
                for k, v in ema_state.items()
                if torch.is_tensor(v)
            }
            self.ema_updates = ema_updates
        else:
            self.ema_policy_state = None
            self.ema_updates = 0
        self._recent_episode_scores.clear()
        self._recent_fill_ratios.clear()
        self._episodes_since_best_check = 0
        self._degradation_streak = 0
        self._board_adapt_boost = 0.0
        self._critic_warmup_updates = 0
        self.clear_rollout()
        self.sync_inference_net()


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
        self.recent_random_flags: list[Deque[bool]] = [deque(maxlen=12) for _ in range(self.num_envs)]
        self.replay_histories: list[list[dict[str, Any]]] = [[] for _ in range(self.num_envs)]
        self._state_snapshots: list[Deque[tuple[int, dict[str, object]]]] = [
            deque(maxlen=8) for _ in range(self.num_envs)
        ]
        self.is_replay_env: list[bool] = [False for _ in range(self.num_envs)]
        self._active_replay_entry: list[dict[str, Any] | None] = [None for _ in range(self.num_envs)]
        self.failure_replay_buffer: Deque[dict[str, Any]] = deque(maxlen=self.failure_buffer_maxlen)
        self.endgame_failure_buffer: Deque[dict[str, Any]] = deque(maxlen=PPO_FAILURE_ENDGAME_MAXLEN)
        self.failure_replays_completed = 0
        self.display_index = 0
        self._last_leader_switch_at = time.monotonic()
        self.last_probs: np.ndarray | None = None
        self.last_greedy_actions: np.ndarray | None = None
        self.last_profile_ms: dict[str, float] = {}
        self.last_completed_env_payload: dict[str, Any] | None = None

    def _is_replay_slot(self, idx: int, *, active_count: int | None = None) -> bool:
        cnt = self.replay_env_count if active_count is None else max(0, min(self.replay_env_count, int(active_count)))
        return cnt > 0 and idx >= (self.num_envs - cnt)

    def clear_failure_buffer(self) -> None:
        self.failure_replay_buffer.clear()
        self.endgame_failure_buffer.clear()
        self._active_replay_entry = [None for _ in range(self.num_envs)]
        for snap_q in self._state_snapshots:
            snap_q.clear()

    def reset_all(self, board_size: int | None = None) -> None:
        if board_size is not None:
            self.board_size = int(board_size)
        self.envs = [self.env_factory(self.board_size) for _ in range(self.num_envs)]
        self.episode_rewards = [0.0 for _ in range(self.num_envs)]
        self.recent_random_flags = [deque(maxlen=12) for _ in range(self.num_envs)]
        self.replay_histories = [[] for _ in range(self.num_envs)]
        self._state_snapshots = [deque(maxlen=8) for _ in range(self.num_envs)]
        self.is_replay_env = [False for _ in range(self.num_envs)]
        self._active_replay_entry = [None for _ in range(self.num_envs)]
        self.failure_replay_buffer.clear()
        self.endgame_failure_buffer.clear()
        self.failure_replays_completed = 0
        self.display_index = 0
        self.last_completed_env_payload = None

    def leader_env(self) -> Any:
        return self.envs[self.display_index]

    def update_leader(self) -> int:
        now = time.monotonic()
        cur = self.display_index
        cur_valid = 0 <= cur < self.num_envs and not self.is_replay_env[cur] and self.envs[cur].step_count > 0

        best_idx = 0
        best_key = (-1, -1, -1)
        for idx, env in enumerate(self.envs):
            honest_priority = 0 if self.is_replay_env[idx] else 1
            key = (honest_priority, int(env.score), int(env.step_count))
            if key > best_key:
                best_key = key
                best_idx = idx

        if cur_valid and best_idx != cur:
            cur_score = int(self.envs[cur].score)
            best_score = int(self.envs[best_idx].score)
            if (now - self._last_leader_switch_at) < 2.0 or (best_score - cur_score) < 10:
                return cur

        if best_idx != cur:
            self._last_leader_switch_at = now
        self.display_index = best_idx
        return best_idx

    def _pick_failure_rewind_snapshot(
        self,
        idx: int,
        death_step: int,
        *,
        rewind_target: int | None = None,
    ) -> dict[str, object] | None:
        snaps = self._state_snapshots[idx]
        if not snaps:
            return None
        rewind = self.failure_rewind_target if rewind_target is None else max(2, int(rewind_target))
        target_step = max(0, death_step - rewind)
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

    def _sample_failure_entry(self, idx: int) -> tuple[dict[str, object] | None, dict[str, Any] | None]:
        # Последний слот приоритетно берёт из эндгейм-буфера (fill >= 0.80), если он не пуст
        use_endgame = (
            len(self.endgame_failure_buffer) > 0
            and (idx == self.num_envs - 1 or len(self.failure_replay_buffer) == 0)
        )
        buf = self.endgame_failure_buffer if use_endgame else self.failure_replay_buffer
        if not buf:
            return None, None
        items = list(buf)
        weights = []
        for item in items:
            if isinstance(item, dict) and "state" in item and isinstance(item["state"], dict):
                fill = float(item.get("fill", 0.3))
                fails = int(item.get("fails", 0))
                weights.append(max(0.05, (fill * fill) / (1.0 + 0.5 * fails)))
            else:
                weights.append(0.25)
        chosen = random.choices(items, weights=weights, k=1)[0]
        if isinstance(chosen, dict) and "state" in chosen and isinstance(chosen["state"], dict):
            return chosen["state"], chosen
        return chosen, None

    def _reset_env_slot(self, idx: int, *, active_replay_count: int | None = None) -> None:
        env = self.envs[idx]
        self.episode_rewards[idx] = 0.0
        self.recent_random_flags[idx].clear()
        self.replay_histories[idx].clear()
        self._state_snapshots[idx].clear()
        self._active_replay_entry[idx] = None
        has_any_failure = len(self.failure_replay_buffer) > 0 or len(self.endgame_failure_buffer) > 0
        if self._is_replay_slot(idx, active_count=active_replay_count) and has_any_failure:
            saved_state, entry_ref = self._sample_failure_entry(idx)
            if saved_state is not None:
                env.load_state(saved_state)
            else:
                env.reset(return_obs=False)
            if saved_state is None or env.done or int(env.board_size) != self.board_size:
                env.reset(return_obs=False)
                self.is_replay_env[idx] = False
            else:
                self.is_replay_env[idx] = True
                self._active_replay_entry[idx] = entry_ref
        else:
            env.reset(return_obs=False)
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
        self.last_probs = probs_np
        self.last_greedy_actions = greedy_actions

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
            is_hunger_critical = float(state[HUNGER_CRITICAL_IDX]) > 0.5
            is_pocket_split = float(state[POCKET_SPLIT_START_IDX + act]) > 0.5
            is_pocket_merge = float(state[POCKET_MERGE_START_IDX + act]) > 0.5
            food_reachable = float(state[FOOD_REACHABLE_IDX]) > 0.5
            old_food_dist = abs(env.snake[0][0] - env.food[0]) + abs(env.snake[0][1] - env.food[1])

            abs_dir = relative_to_absolute(env.direction, act)
            _, reward, done, info = env.step(abs_dir, return_obs=False)
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
                is_hunger_critical=is_hunger_critical,
                critical_hunger_penalty=PPO_CRITICAL_HUNGER_PENALTY,
            )
            if reason == "victory":
                shaped_reward += PPO_VICTORY_EXTRA_BONUS
            if not done and reason == "step" and food_reachable and not is_endgame:
                new_food_dist = abs(env.snake[0][0] - env.food[0]) + abs(env.snake[0][1] - env.food[1])
                if new_food_dist < old_food_dist:
                    shaped_reward += PPO_FOOD_STEP_CLOSER_BONUS
                elif new_food_dist > old_food_dist:
                    shaped_reward -= PPO_FOOD_STEP_AWAY_PENALTY

            all_penalties = [str(item) for item in info.get("penalty_reasons", [])] + extra_penalties
            step_rewards[idx] = float(shaped_reward)
            step_dones[idx] = 1.0 if done else 0.0
            self.episode_rewards[idx] += float(shaped_reward)

            step_num = int(info.get("steps", env.step_count))
            if not is_replay:
                if step_num == 1:
                    self.recent_random_flags[idx].clear()
                    self.replay_histories[idx].clear()
                self.recent_random_flags[idx].append(bool(was_random))
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
                    entry_ref = self._active_replay_entry[idx]
                    if entry_ref is not None:
                        if reason == "victory":
                            entry_ref["fails"] = 0
                        else:
                            entry_ref["fails"] = int(entry_ref.get("fails", 0)) + 1
                            if entry_ref["fails"] >= 6:
                                try:
                                    self.failure_replay_buffer.remove(entry_ref)
                                except ValueError:
                                    pass
                                try:
                                    self.endgame_failure_buffer.remove(entry_ref)
                                except ValueError:
                                    pass
                else:
                    board_cells = max(1, int(env.board_size) * int(env.board_size))
                    fill_ratio = float(int(env.score) + 3) / float(board_cells)
                    best_fill = float(getattr(agent, "best_fill_ratio", 0.0))
                    adaptive_min_fill = max(PPO_FAILURE_MIN_FILL, best_fill * 0.85)
                    passes_min = (
                        fill_ratio >= adaptive_min_fill
                        and (self.board_size != 10 or int(env.score) >= self.failure_min_score)
                    )
                    had_recent_random = any(self.recent_random_flags[idx])
                    allow_failure_record = (not had_recent_random) or len(self.failure_replay_buffer) < 12
                    if reason != "victory" and passes_min and allow_failure_record:
                        rewind_target = (
                            PPO_FAILURE_STARVATION_REWIND
                            if reason == "starvation"
                            else self.failure_rewind_target
                        )
                        rewind_state = self._pick_failure_rewind_snapshot(
                            idx,
                            int(env.step_count),
                            rewind_target=rewind_target,
                        )
                        if rewind_state is not None:
                            entry = {"state": rewind_state, "fill": fill_ratio, "fails": 0}
                            self.failure_replay_buffer.append(entry)
                            if fill_ratio >= PPO_FAILURE_ENDGAME_FILL:
                                self.endgame_failure_buffer.append(entry)

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
                        "replay_steps": list(self.replay_histories[idx]) if record_replays else [],
                        "loss": float(agent.last_loss),
                    }
                    completed.append(finished_snapshot)
                    self.last_completed_env_payload = finished_snapshot
                # На пике (best_fill >= 0.95) оставляем 1 слот Failure-Replay вместо 2, чтобы не забывать мидгейм
                active_replay_cnt = 1 if (self.replay_env_count >= 2 and float(getattr(agent, "best_fill_ratio", 0.0)) >= 0.95) else self.replay_env_count
                self._reset_env_slot(idx, active_replay_count=active_replay_cnt)

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
