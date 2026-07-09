from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from backend_config import (
    MODEL_ACTION_LABELS,
    CPU_BATCH_SIZE,
    GLOBAL_SIGNAL_LABELS,
    NETWORK_INPUT_ADAPTER_MODE,
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
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool, float]] = []
        self.priorities = np.empty(self.capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        self.alpha = float(alpha)
        self.beta = float(beta_start)
        self.beta_increment = float(beta_increment)
        self.priority_epsilon = float(priority_epsilon)
        self._priority_cap = float(np.finfo(np.float32).max)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        priority_scale: float = 1.0,
    ) -> None:
        item = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
            float(priority_scale),
        )
        scale = max(1.0, float(priority_scale))
        if not np.isfinite(scale):
            scale = 1.0
        base_priority = self.max_priority if np.isfinite(self.max_priority) and self.max_priority > 0.0 else 1.0
        priority = base_priority * scale
        if not np.isfinite(priority):
            priority = self._priority_cap
        if len(self.buffer) < self.capacity:
            index = len(self.buffer)
            self.buffer.append(item)
        else:
            index = self.position
            self.buffer[index] = item
        safe_priority = min(max(float(priority), self.priority_epsilon), self._priority_cap)
        self.priorities[index] = safe_priority
        self.max_priority = max(min(base_priority, self._priority_cap), safe_priority)
        self.position = (index + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        size = len(self.buffer)
        if size <= 0:
            raise ValueError("cannot sample from an empty replay buffer")
        safe_batch_size = min(max(1, int(batch_size)), size)
        priorities = self.priorities[:size].astype(np.float64, copy=False)
        scaled = np.power(np.maximum(priorities, self.priority_epsilon), self.alpha)
        total = float(scaled.sum())
        if total <= 0.0 or not np.isfinite(total):
            probabilities = np.full(size, 1.0 / size, dtype=np.float64)
        else:
            probabilities = scaled / total
        indices = np.random.choice(size, size=safe_batch_size, replace=False, p=probabilities)
        states, actions, rewards, next_states, dones, priority_scales = zip(*(self.buffer[int(index)] for index in indices))
        weights = np.power(size * probabilities[indices], -self.beta)
        max_weight = float(weights.max())
        if max_weight > 0.0 and np.isfinite(max_weight):
            weights /= max_weight
        self.beta = min(1.0, self.beta + self.beta_increment)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
            weights.astype(np.float32, copy=False),
            np.asarray(priority_scales, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray, priority_scales: np.ndarray | None = None) -> None:
        if priority_scales is None:
            priority_scales = np.ones_like(priorities, dtype=np.float32)
        size = len(self.buffer)
        for index, priority, scale in zip(indices.tolist(), priorities.tolist(), priority_scales.tolist()):
            index = int(index)
            if not 0 <= index < size:
                continue
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
            self.priorities[index] = clipped
            self.max_priority = max(min(self.max_priority, self._priority_cap), clipped)


class DQNAgent:
    def __init__(
        self,
        device: torch.device,
        *,
        gamma: float = 0.96,
        learning_rate: float = 0.0008,
        batch_size: int = CPU_BATCH_SIZE,
        buffer_size: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.08,
        epsilon_decay: float = 0.9997,
        target_sync_interval: int = 800,
    ) -> None:
        self.device = device
        self.global_signal_count = len(GLOBAL_SIGNAL_LABELS)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.target_sync_interval = int(target_sync_interval)
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
            input_size=self.global_signal_count,
            input_adapter_mode=NETWORK_INPUT_ADAPTER_MODE,
        ).to(device)
        self.target_net = QNetwork(
            input_size=self.global_signal_count,
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

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, *, priority_scale: float = 1.0) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done, priority_scale=priority_scale)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights, priority_scales = self.replay_buffer.sample(self.batch_size)

        states_tensor = self._array_to_tensor(states)
        actions_tensor = self._array_to_tensor(actions, dtype=torch.int64)
        rewards_tensor = self._array_to_tensor(rewards)
        next_states_tensor = self._array_to_tensor(next_states)
        dones_tensor = self._array_to_tensor(dones)
        weights_tensor = self._array_to_tensor(weights)

        q_values = self.online_net(states_tensor)
        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_net(next_states_tensor).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_tensor).gather(1, next_actions).squeeze(1)
            targets = rewards_tensor + self.gamma * next_q * (1.0 - dones_tensor)

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
        if self.total_updates % self.target_sync_interval == 0:
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
        online_state = {key: value.detach().to(device="cpu", dtype=torch.float16) for key, value in self.online_net.state_dict().items()}
        target_state = {key: value.detach().to(device="cpu", dtype=torch.float16) for key, value in self.target_net.state_dict().items()}
        return {
            "online_net": online_state,
            "target_net": target_state,
            "optimizer": self.optimizer.state_dict(),
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
