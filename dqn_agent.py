from __future__ import annotations

from collections import deque
import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from backend_config import (
    APPLE_STEP_START_IDX,
    CPU_BATCH_SIZE,
    DANGER_START_IDX,
    FOOD_REACHABLE_IDX,
    GLOBAL_SIGNAL_LABELS,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    MODEL_ACTION_LABELS,
    NETWORK_INPUT_ADAPTER_MODE,
    PARTITION_PENALTY_LARGE,
    PARTITION_PENALTY_SMALL,
    PARTITION_START_IDX,
    TAIL_BLOCKED_START_IDX,
    TAIL_CLOSE_PENALTY,
    TAIL_CLOSE_START_IDX,
    TAIL_LOSS_PENALTY,
)
from networks import QNetwork

def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def load_matching_state(module: nn.Module, saved_state: dict[str, torch.Tensor] | None) -> int:
    if not saved_state:
        return 0
    current_state = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in saved_state.items():
        if key in current_state:
            if current_state[key].shape == value.shape:
                matched[key] = value.to(dtype=current_state[key].dtype)
            elif key == "hidden1_layer.weight" and current_state[key].dim() == 2 and value.dim() == 2:
                if current_state[key].shape[0] == value.shape[0]:
                    matched[key] = value.to(dtype=current_state[key].dtype)
    current_state.update(matched)
    module.load_state_dict(current_state)
    return len(matched)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 0.0015,
        priority_epsilon: float = 1e-3,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self._size = 0
        self.position = 0
        self.max_priority = 1.0
        self.alpha = float(alpha)
        self.beta = float(beta_start)
        self.beta_increment = float(beta_increment)
        self.priority_epsilon = float(priority_epsilon)
        self._priority_cap = float(np.finfo(np.float32).max)

        # Дерево отрезков (SumTree) для O(log N) сэмплирования и обновления приоритетов
        tree_cap = 1
        depth = 0
        while tree_cap < self.capacity:
            tree_cap <<= 1
            depth += 1
        self._tree_cap = tree_cap
        self._tree_depth = depth
        self._tree = np.zeros(2 * tree_cap, dtype=np.float64)

        # Предвыделенные массивы (инициализируются лениво по размерности state)
        self._state_dim = 0
        self._states: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=np.float32)
        self._dones = np.empty(self.capacity, dtype=np.float32)
        self._discounts = np.empty(self.capacity, dtype=np.float32)
        self._priority_scales = np.empty(self.capacity, dtype=np.float32)

        # Постоянная память критических ситуаций (смерть, потеря хвоста, угловое яблоко)
        self.permanent_capacity = 2000
        self._perm_size = 0
        self.permanent_position = 0
        self._perm_states: np.ndarray | None = None
        self._perm_next_states: np.ndarray | None = None
        self._perm_actions = np.empty(self.permanent_capacity, dtype=np.int64)
        self._perm_rewards = np.empty(self.permanent_capacity, dtype=np.float32)
        self._perm_dones = np.empty(self.permanent_capacity, dtype=np.float32)
        self._perm_discounts = np.empty(self.permanent_capacity, dtype=np.float32)
        self._perm_priority_scales = np.empty(self.permanent_capacity, dtype=np.float32)
        self.permanent_priorities = np.empty(self.permanent_capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def _ensure_state_arrays(self, state_dim: int) -> None:
        if self._state_dim == state_dim and self._states is not None:
            return
        self._state_dim = state_dim
        self._states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self._next_states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self._perm_states = np.empty((self.permanent_capacity, state_dim), dtype=np.float32)
        self._perm_next_states = np.empty((self.permanent_capacity, state_dim), dtype=np.float32)

    def _tree_set(self, data_idx: int, raw_priority: float) -> None:
        scaled = float(max(raw_priority, self.priority_epsilon)) ** self.alpha
        if not np.isfinite(scaled):
            scaled = float(self._priority_cap) ** self.alpha
        node = self._tree_cap + data_idx
        tree = self._tree
        tree[node] = scaled
        node >>= 1
        while node >= 1:
            tree[node] = tree[node << 1] + tree[(node << 1) | 1]
            node >>= 1

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        priority_scale: float = 1.0,
        discount: float = 0.97,
        force_permanent: bool = False,
    ) -> None:
        state_arr = np.asarray(state, dtype=np.float32).ravel()
        next_arr = np.asarray(next_state, dtype=np.float32).ravel()
        self._ensure_state_arrays(int(state_arr.shape[0]))

        scale = max(1.0, float(priority_scale))
        if not np.isfinite(scale):
            scale = 1.0
        base_priority = self.max_priority if np.isfinite(self.max_priority) and self.max_priority > 0.0 else 1.0
        priority = base_priority * scale
        if not np.isfinite(priority):
            priority = self._priority_cap
        safe_priority = min(max(float(priority), self.priority_epsilon), self._priority_cap)

        if self._size < self.capacity:
            index = self._size
            self._size += 1
        else:
            index = self.position

        self._states[index] = state_arr  # type: ignore[index]
        self._next_states[index] = next_arr  # type: ignore[index]
        self._actions[index] = int(action)
        self._rewards[index] = float(reward)
        self._dones[index] = 1.0 if done else 0.0
        self._discounts[index] = float(discount)
        self._priority_scales[index] = scale
        self._tree_set(index, safe_priority)

        # Не масштабируем max_priority на scale при push, чтобы избежать экспоненциального роста 5^N -> float32.max
        self.max_priority = min(base_priority, self._priority_cap)
        self.position = (index + 1) % self.capacity

        # Дублируем критические переходы (смерть, потеря хвоста, угловое яблоко, победа) в постоянную память
        if done or scale >= 3.0 or force_permanent:
            if self._perm_size < self.permanent_capacity:
                perm_idx = self._perm_size
                self._perm_size += 1
            else:
                perm_idx = self.permanent_position
            self._perm_states[perm_idx] = state_arr  # type: ignore[index]
            self._perm_next_states[perm_idx] = next_arr  # type: ignore[index]
            self._perm_actions[perm_idx] = int(action)
            self._perm_rewards[perm_idx] = float(reward)
            self._perm_dones[perm_idx] = 1.0 if done else 0.0
            self._perm_discounts[perm_idx] = float(discount)
            self._perm_priority_scales[perm_idx] = scale
            self.permanent_priorities[perm_idx] = safe_priority
            self.permanent_position = (perm_idx + 1) % self.permanent_capacity

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        size = self._size
        if size <= 0 or self._states is None or self._next_states is None:
            raise ValueError("cannot sample from an empty replay buffer")

        perm_size_available = self._perm_size
        target_perm_size = min(int(batch_size * 0.125), perm_size_available)
        regular_batch_size = max(1, batch_size - target_perm_size)

        # 1. Стратифицированное O(log N) сэмплирование из SumTree
        tree = self._tree
        total = float(tree[1])
        if total <= 0.0 or not np.isfinite(total):
            regular_indices = np.random.randint(0, size, size=regular_batch_size, dtype=np.int64)
            regular_probs = np.full(regular_batch_size, 1.0 / size, dtype=np.float64)
        else:
            segment = total / regular_batch_size
            offsets = np.random.random(regular_batch_size)
            targets = (np.arange(regular_batch_size, dtype=np.float64) + offsets) * segment
            np.clip(targets, 0.0, total * (1.0 - 1e-12), out=targets)

            idx = np.ones(regular_batch_size, dtype=np.int64)
            for _ in range(self._tree_depth):
                left = idx << 1
                left_vals = tree[left]
                go_right = targets > left_vals
                targets = np.where(go_right, targets - left_vals, targets)
                idx = np.where(go_right, left + 1, left)

            regular_indices = np.clip(idx - self._tree_cap, 0, size - 1)
            regular_probs = np.maximum(tree[self._tree_cap + regular_indices], 1e-12) / total

        regular_weights = np.power(size * regular_probs, -self.beta)

        states_batch = self._states[regular_indices]
        next_states_batch = self._next_states[regular_indices]
        actions_batch = self._actions[regular_indices]
        rewards_batch = self._rewards[regular_indices]
        dones_batch = self._dones[regular_indices]
        discounts_batch = self._discounts[regular_indices]
        scales_batch = self._priority_scales[regular_indices]
        final_indices = regular_indices

        if target_perm_size > 0 and self._perm_states is not None and self._perm_next_states is not None:
            perm_priorities = self.permanent_priorities[:perm_size_available].astype(np.float64, copy=False)
            perm_scaled = np.power(np.maximum(perm_priorities, self.priority_epsilon), self.alpha)
            perm_total = float(perm_scaled.sum())
            if perm_total <= 0.0 or not np.isfinite(perm_total):
                perm_probs = np.full(perm_size_available, 1.0 / perm_size_available, dtype=np.float64)
            else:
                perm_probs = perm_scaled / perm_total

            perm_indices = np.random.choice(perm_size_available, size=target_perm_size, replace=False, p=perm_probs)
            perm_weights = np.power(perm_size_available * perm_probs[perm_indices], -self.beta)

            states_batch = np.concatenate((states_batch, self._perm_states[perm_indices]), axis=0)
            next_states_batch = np.concatenate((next_states_batch, self._perm_next_states[perm_indices]), axis=0)
            actions_batch = np.concatenate((actions_batch, self._perm_actions[perm_indices]), axis=0)
            rewards_batch = np.concatenate((rewards_batch, self._perm_rewards[perm_indices]), axis=0)
            dones_batch = np.concatenate((dones_batch, self._perm_dones[perm_indices]), axis=0)
            discounts_batch = np.concatenate((discounts_batch, self._perm_discounts[perm_indices]), axis=0)
            scales_batch = np.concatenate((scales_batch, self._perm_priority_scales[perm_indices]), axis=0)
            final_indices = np.concatenate((regular_indices, -(perm_indices + 1)), axis=0)
            weights_raw = np.concatenate((regular_weights, perm_weights), axis=0)
        else:
            weights_raw = regular_weights

        weights_arr = weights_raw.astype(np.float32)
        max_weight = float(weights_arr.max())
        if max_weight > 0.0 and np.isfinite(max_weight):
            weights_arr /= max_weight

        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            states_batch,
            actions_batch,
            rewards_batch,
            next_states_batch,
            dones_batch,
            final_indices,
            weights_arr,
            scales_batch,
            discounts_batch,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray, priority_scales: np.ndarray | None = None) -> None:
        if priority_scales is None:
            priority_scales = np.ones_like(priorities, dtype=np.float32)
        size = self._size
        perm_size = self._perm_size

        for index, priority, scale in zip(indices.tolist(), priorities.tolist(), priority_scales.tolist()):
            index = int(index)
            safe_scale = max(1.0, float(scale))
            if not np.isfinite(safe_scale):
                safe_scale = 1.0
            safe_priority = max(float(priority), self.priority_epsilon)
            if not np.isfinite(safe_priority):
                safe_priority = self._priority_cap
            boosted = safe_priority * safe_scale
            if not np.isfinite(boosted):
                boosted = self._priority_cap
            clipped = min(max(boosted, self.priority_epsilon), self._priority_cap)

            if index >= 0:
                if 0 <= index < size:
                    self._tree_set(index, clipped)
            else:
                perm_idx = abs(index) - 1
                if 0 <= perm_idx < perm_size:
                    self.permanent_priorities[perm_idx] = clipped

            self.max_priority = max(min(self.max_priority, self._priority_cap), clipped)


class DQNAgent:
    def __init__(
        self,
        device: torch.device,
        *,
        gamma: float = 0.97,
        learning_rate: float = 0.0008,
        lr_min: float | None = None,
        lr_decay_updates: int = 0,
        lr_decay_start_update: int = 0,
        batch_size: int = CPU_BATCH_SIZE,
        buffer_size: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.08,
        epsilon_decay: float = 0.9997,
        target_sync_interval: int = 800,
        target_tau: float = 0.0,
        hidden1: int = HIDDEN_LAYER_1,
        hidden2: int = HIDDEN_LAYER_2,
        frame_stack: int = 1,
        n_step: int = 1,
        signal_count: int | None = None,
    ) -> None:
        self.device = device
        self.global_signal_count = int(signal_count) if signal_count is not None else len(GLOBAL_SIGNAL_LABELS)
        self.hidden1 = max(4, int(hidden1))
        self.hidden2 = max(4, int(hidden2))
        self.frame_stack = max(1, int(frame_stack))
        self.n_step = max(1, int(n_step))
        self.input_size = self.global_signal_count * self.frame_stack
        self._nstep_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, float, bool]] = deque()
        self.gamma = float(gamma)
        self.lr_start = float(learning_rate)
        self.lr_min = float(lr_min) if lr_min is not None else float(learning_rate)
        self.lr_decay_updates = max(0, int(lr_decay_updates))
        self.lr_decay_start_update = max(0, int(lr_decay_start_update))
        self.batch_size = int(batch_size)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.target_sync_interval = int(target_sync_interval)
        self.target_tau = float(target_tau)
        self.total_updates = 0
        self.last_loss = 0.0
        self.best_score = 0
        self.best_fitness = float("-inf")
        self.last_fitness = 0.0
        self.generation = 0
        self.current_index = 0
        self.population_size = 1
        self.mutation_rate = 0.0
        self.victory_archive: list[object] = []

        self.online_net = QNetwork(
            input_size=self.input_size,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            input_adapter_mode=NETWORK_INPUT_ADAPTER_MODE,
        ).to(device)
        self.target_net = QNetwork(
            input_size=self.input_size,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            input_adapter_mode=NETWORK_INPUT_ADAPTER_MODE,
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def _array_to_float_tensor(self, values: np.ndarray) -> torch.Tensor:
        if not isinstance(values, np.ndarray):
            values = np.asarray(values, dtype=np.float32)
        elif values.dtype != np.float32 or not values.flags.c_contiguous:
            values = np.ascontiguousarray(values, dtype=np.float32)
        tensor = torch.from_numpy(values)
        if tensor.device != self.device:
            tensor = tensor.to(self.device, non_blocking=True)
        return tensor

    def _array_to_tensor(self, values: np.ndarray, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = torch.from_numpy(values)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if tensor.device != self.device:
            tensor = tensor.to(self.device, non_blocking=True)
        return tensor

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        *,
        snake_length: int = 1,
    ) -> tuple[int, np.ndarray, bool, int]:
        state_tensor = self._array_to_float_tensor(state)
        with torch.inference_mode():
            q_values = self.online_net(state_tensor).squeeze(0)
        greedy_action = int(torch.argmax(q_values).item())
        was_random = bool(training and random.random() < self.epsilon)
        action = random.randrange(len(MODEL_ACTION_LABELS)) if was_random else greedy_action
        return action, q_values.detach().cpu().numpy(), was_random, greedy_action

    def select_best_action(self, state: np.ndarray) -> tuple[int, np.ndarray, bool, int]:
        return self.select_action(state, training=False)

    def _push_nstep_head(self) -> None:
        head_state, head_action, _, _, _, _, _ = self._nstep_buffer[0]
        total_reward = 0.0
        discount = 1.0
        last_next = self._nstep_buffer[0][3]
        last_done = False
        max_scale = 1.0
        any_force = False
        for _, _, r, ns, d, sc, fp in self._nstep_buffer:
            total_reward += discount * r
            discount *= self.gamma
            last_next = ns
            last_done = d
            if sc > max_scale:
                max_scale = sc
            if fp:
                any_force = True
            if d:
                break
        self.replay_buffer.push(
            head_state,
            head_action,
            total_reward,
            last_next,
            last_done,
            priority_scale=max_scale,
            discount=discount,
            force_permanent=any_force,
        )
        self._nstep_buffer.popleft()

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        priority_scale: float = 1.0,
        force_permanent: bool = False,
    ) -> None:
        if self.n_step <= 1:
            self.replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
                priority_scale=priority_scale,
                discount=self.gamma,
                force_permanent=force_permanent,
            )
        else:
            self._nstep_buffer.append((state, action, float(reward), next_state, bool(done), float(priority_scale), bool(force_permanent)))
            if len(self._nstep_buffer) >= self.n_step:
                self._push_nstep_head()
            if done:
                while self._nstep_buffer:
                    self._push_nstep_head()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights, priority_scales, discounts = self.replay_buffer.sample(self.batch_size)

        states_tensor = self._array_to_tensor(states)
        actions_tensor = self._array_to_tensor(actions, dtype=torch.int64)
        rewards_tensor = self._array_to_tensor(rewards)
        next_states_tensor = self._array_to_tensor(next_states)
        dones_tensor = self._array_to_tensor(dones)
        weights_tensor = self._array_to_tensor(weights)
        discounts_tensor = self._array_to_tensor(discounts)

        q_values = self.online_net(states_tensor)
        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_net(next_states_tensor).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_tensor).gather(1, next_actions).squeeze(1)
            targets = rewards_tensor + discounts_tensor * next_q * (1.0 - dones_tensor)

        td_errors = targets - current_q
        losses = F.smooth_l1_loss(current_q, targets, reduction="none")
        loss = (losses * weights_tensor).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=2.0)
        self.optimizer.step()

        td_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-3
        self.replay_buffer.update_priorities(indices, td_priorities, priority_scales)

        self.total_updates += 1
        if (
            self.lr_decay_updates > 0
            and self.lr_min < self.lr_start
            and self.total_updates > self.lr_decay_start_update
        ):
            frac = min(1.0, (self.total_updates - self.lr_decay_start_update) / self.lr_decay_updates)
            new_lr = self.lr_min + 0.5 * (self.lr_start - self.lr_min) * (1.0 + math.cos(math.pi * frac))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
        if self.target_tau > 0.0:
            tau = min(1.0, self.target_tau)
            one_minus_tau = 1.0 - tau
            with torch.no_grad():
                for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
                    target_param.data.mul_(one_minus_tau).add_(online_param.data, alpha=tau)
        elif self.total_updates % self.target_sync_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def finish_episode(
        self,
        *,
        score: int,
        reward: float,
        steps: int,
        reason: str,
        board_size: int,
        max_steps_without_food: int = 0,
    ) -> float:
        fitness = float(score) * 100.0 + float(steps) * 0.1
        self.last_fitness = fitness
        if int(score) > self.best_score:
            self.best_score = int(score)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
        return fitness

    def snapshot(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        tensor = self._array_to_float_tensor(observation)
        with torch.inference_mode():
            activations = self.online_net.forward_with_activations(tensor)
        return {name: value.detach().cpu().numpy() for name, value in activations.items()}

    def move_to_device(self, device: torch.device) -> None:
        self.device = device
        self.online_net = self.online_net.to(device)
        self.target_net = self.target_net.to(device)
        move_optimizer_state(self.optimizer, device)

    def export_checkpoint(self) -> dict[str, object]:
        online_state = {key: value.detach().to(device="cpu", dtype=torch.float32) for key, value in self.online_net.state_dict().items()}
        target_state = {key: value.detach().to(device="cpu", dtype=torch.float32) for key, value in self.target_net.state_dict().items()}
        return {
            "online_net": online_state,
            "target_net": target_state,
            "optimizer": self.optimizer.state_dict(),
            "arch_meta": {
                "signal_count": self.global_signal_count,
                "frame_stack": self.frame_stack,
                "hidden1": self.hidden1,
                "hidden2": self.hidden2,
            },
        }

    def import_checkpoint(self, payload: dict[str, object]) -> None:
        online_state = payload.get("online_net")
        matched = load_matching_state(self.online_net, online_state if isinstance(online_state, dict) else None)
        target_state = payload.get("target_net")
        target_matched = load_matching_state(self.target_net, target_state if isinstance(target_state, dict) else None)
        if matched > 0 and target_matched == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        optimizer_state = payload.get("optimizer")
        if isinstance(optimizer_state, dict):
            try:
                self.optimizer.load_state_dict(optimizer_state)
                move_optimizer_state(self.optimizer, self.device)
            except Exception:
                self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=0.0008)


def filter_dqn_random_action(
    state: np.ndarray,
    predicted_action: int,
    greedy_action: int,
    was_random: bool,
) -> tuple[int, bool]:
    """Не тратит случайный шаг на смертельный ход, потерю хвоста или пропуск смежного яблока."""
    if not was_random:
        return predicted_action, False
    leads_to_collision = float(state[DANGER_START_IDX + predicted_action]) >= 0.5
    leads_to_tail_loss = float(state[TAIL_BLOCKED_START_IDX + predicted_action]) > 0.5
    greedy_eats_food = float(state[APPLE_STEP_START_IDX + greedy_action]) > 0.5
    random_skips_food = greedy_eats_food and float(state[APPLE_STEP_START_IDX + predicted_action]) < 0.5
    if leads_to_collision or leads_to_tail_loss or random_skips_food:
        return greedy_action, False
    return predicted_action, True


def evaluate_pre_step_risks(
    env: object,
    state: np.ndarray,
    action: int,
) -> tuple[bool, bool, int, bool]:
    """Оценивает переходы в потерю хвоста, близость к хвосту и разрыв поля до вызова env.step()."""
    next_tail_blocked = float(state[TAIL_BLOCKED_START_IDX + action]) > 0.5
    is_tail_close = float(state[TAIL_CLOSE_START_IDX + action]) > 0.5

    is_transition_to_tail_loss = False
    tail_dist = -1
    if next_tail_blocked or is_tail_close:
        tail_path = env._shortest_path_to_tail()  # type: ignore[attr-defined]
        tail_dist = len(tail_path) if tail_path else -1
        current_tail_blocked = tail_path is None
        is_transition_to_tail_loss = (not current_tail_blocked) and next_tail_blocked

    current_partition = float(state[FOOD_REACHABLE_IDX]) < 0.5
    next_partition = float(state[PARTITION_START_IDX + action]) > 0.5
    is_transition_to_partition = (not current_partition) and next_partition
    return is_transition_to_tail_loss, is_tail_close, tail_dist, is_transition_to_partition


def apply_step_risk_shaping(
    reward: float,
    reason: str,
    snake_len: int,
    pre_step_risks: tuple[bool, bool, int, bool],
    *,
    is_endgame: bool = False,
    is_pocket_split: bool = False,
    is_pocket_merge: bool = False,
    endgame_split_penalty: float = 0.0,
    endgame_merge_bonus: float = 0.0,
    is_hunger_critical: bool = False,
    critical_hunger_penalty: float = 0.0,
) -> tuple[float, list[str]]:
    """Применяет штрафы формирования наград (reward shaping) поверх базовой награды среды."""
    is_transition_to_tail_loss, is_tail_close, tail_dist, is_transition_to_partition = pre_step_risks
    extra_penalties: list[str] = []

    if is_transition_to_tail_loss and reason not in ("apple", "victory"):
        reward -= TAIL_LOSS_PENALTY
        extra_penalties.append("tail_loss")

    if is_tail_close and tail_dist != -1:
        reward -= (3 - tail_dist) * TAIL_CLOSE_PENALTY
        extra_penalties.append("tail_close")

    if is_transition_to_partition:
        penalty_val = PARTITION_PENALTY_LARGE if snake_len >= 20 else PARTITION_PENALTY_SMALL
        reward -= penalty_val
        extra_penalties.append("partition")

    if is_endgame and is_pocket_split and endgame_split_penalty > 0.0 and reason not in ("apple", "victory"):
        reward -= float(endgame_split_penalty)
        extra_penalties.append("pocket_split")

    if is_endgame and is_pocket_merge and endgame_merge_bonus > 0.0:
        reward += float(endgame_merge_bonus)

    if is_hunger_critical and critical_hunger_penalty > 0.0 and reason not in ("apple", "victory"):
        reward -= float(critical_hunger_penalty)

    return reward, extra_penalties


def compute_dqn_priority_scale(
    done: bool,
    reason: str,
    is_transition_to_tail_loss: bool,
    head_pos: tuple[int, int],
    board_size: int,
    *,
    is_pocket_split: bool = False,
    is_endgame: bool = False,
    pocket_split_priority: float = 1.0,
    endgame_priority: float = 1.0,
    score: int = 0,
    late_endgame_priority: float = 1.0,
) -> float:
    """Вычисляет коэффициент PER Boost для критических переходов."""
    scale = 1.0
    if done:
        scale = 6.0 if (reason == "victory" and late_endgame_priority > 1.0) else 5.0
    elif is_transition_to_tail_loss and reason not in ("apple", "victory"):
        scale = 3.0
    elif reason in ("apple", "victory"):
        corners = (
            (0, 0),
            (board_size - 1, 0),
            (0, board_size - 1),
            (board_size - 1, board_size - 1),
        )
        if head_pos in corners:
            scale = 4.0
    if is_pocket_split and reason not in ("apple", "victory"):
        scale = max(scale, float(pocket_split_priority))
    if is_endgame:
        scale = max(scale, float(endgame_priority))
    if score >= 90 and late_endgame_priority > 1.0:
        scale = max(scale, float(late_endgame_priority))
    return scale
