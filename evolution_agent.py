from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
import time
from typing import Any, Callable, Deque
import gc  # <-- ДОБАВЛЕНО: импортируем сборщик мусора

import numpy as np
import torch
from torch import nn

# --- ДОБАВЛЕНО: Оптимизация Garbage Collector ---
# По умолчанию Python собирает мусор каждые 700 объектов. 
# Мы увеличиваем лимит до 50 000. Сборка будет происходить реже
# (в фоне) и паузы по 130-140мс практически исчезнут.
gc.set_threshold(50000, 50, 50)

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - fallback for older PyTorch builds
    from torch.nn.utils.stateless import functional_call

from backend_config import (
    DEFAULT_INPUT_GRID,
    HIDDEN_LAYER_2,
    MODEL_ACTION_LABELS,
    local_input_size,
    hunger_budget_for,
)
from networks import HybridBoardEncoder

DEFAULT_GLOBAL_SIGNAL_COUNT = 30
EVOLUTION_HUNGER_CULL_REASON = "evolution_hunger_cut"
EVOLUTION_HUNGER_CULL_PROTECTED_RATIO = 0.45
EVOLUTION_HUNGER_CULL_MIN_PROTECTED = 2
EVOLUTION_POPULATION_SIZE = 48
EVOLUTION_ELITE_COUNT = 4
EVOLUTION_MUTATION_RATE = 0.045
EVOLUTION_MUTATION_STRENGTH = 0.12
EVOLUTION_CROSSOVER_RATE = 0.65
EVOLUTION_RANDOM_INJECTION = 4
EVOLUTION_VICTORY_ARCHIVE_SIZE = 8
EVOLUTION_VICTORY_SEED_COUNT = 4
EVOLUTION_VICTORY_MUTATION_RATE_SCALE = 0.55
EVOLUTION_VICTORY_MUTATION_STRENGTH_SCALE = 0.35

class PolicyNetwork(HybridBoardEncoder):
    def __init__(self, input_size: int = DEFAULT_INPUT_GRID * DEFAULT_INPUT_GRID, input_grid: int = DEFAULT_INPUT_GRID, global_signal_count: int = DEFAULT_GLOBAL_SIGNAL_COUNT) -> None:
        super().__init__(input_grid=input_grid, global_signal_count=global_signal_count)
        self.output = nn.Linear(HIDDEN_LAYER_2, len(MODEL_ACTION_LABELS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, _, hidden2, _ = self.merged_hidden(inputs)
        return self.output(hidden2)

    def forward_with_activations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs, hidden1, hidden2, local_heatmap = self.merged_hidden(inputs)
        output = self.output(hidden2)
        return {
            "input": inputs.squeeze(0),
            "hidden1": hidden1.squeeze(0),
            "hidden2": hidden2.squeeze(0),
            "output": output.squeeze(0),
            "value": torch.empty(0, dtype=output.dtype, device=output.device),
            "local_heatmap": local_heatmap.squeeze(0),
        }


@dataclass
class EvolutionIndividual:
    genome: dict[str, torch.Tensor]
    fitness: float | None = None
    score: int = 0
    reward: float = 0.0
    steps: int = 0
    reason: str = "new"


class EvolutionAgent:
    def __init__(
        self,
        device: torch.device,
        input_grid: int = DEFAULT_INPUT_GRID,
        *,
        global_signal_count: int = DEFAULT_GLOBAL_SIGNAL_COUNT,
        population_size: int = EVOLUTION_POPULATION_SIZE,
        elite_count: int = EVOLUTION_ELITE_COUNT,
        mutation_rate: float = EVOLUTION_MUTATION_RATE,
        mutation_strength: float = EVOLUTION_MUTATION_STRENGTH,
        crossover_rate: float = EVOLUTION_CROSSOVER_RATE,
        random_injection: int = EVOLUTION_RANDOM_INJECTION,
    ) -> None:
        self.device = device
        self.input_grid = input_grid
        self.global_signal_count = global_signal_count
        self.population_size = max(4, int(population_size))
        self.elite_count = max(1, min(int(elite_count), self.population_size - 1))
        self.mutation_rate = float(mutation_rate)
        self.mutation_strength = float(mutation_strength)
        self.crossover_rate = float(crossover_rate)
        self.random_injection = max(0, min(int(random_injection), self.population_size - self.elite_count))
        self.generation = 1
        self.current_index = 0
        self.best_fitness = float("-inf")
        self.best_score = 0
        self.best_genome: dict[str, torch.Tensor] | None = None
        self.victory_archive: list[EvolutionIndividual] = []
        self.last_fitness = 0.0
        self.last_generation_average = 0.0
        self.completed_evaluations = 0
        input_size = local_input_size(input_grid) + global_signal_count
        self.online_net = PolicyNetwork(
            input_size=input_size,
            input_grid=input_grid,
            global_signal_count=global_signal_count,
        ).to(device)
        self.population = [EvolutionIndividual(self._fresh_genome()) for _ in range(self.population_size)]
        self._load_current_genome()

    @property
    def victory_archive_size(self) -> int:
        return len(self.victory_archive)

    def settings_payload(self) -> dict[str, int | float]:
        return {
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "mutation_rate": self.mutation_rate,
            "mutation_strength": self.mutation_strength,
            "crossover_rate": self.crossover_rate,
            "random_injection": self.random_injection,
        }

    def _clone_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    def _fresh_genome(self) -> dict[str, torch.Tensor]:
        self.online_net.apply(self._reset_module)
        return self._clone_state(self.online_net.state_dict())

    @staticmethod
    def _reset_module(module: nn.Module) -> None:
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()

    def _load_current_genome(self) -> None:
        if not self.population:
            self.population = [EvolutionIndividual(self._fresh_genome()) for _ in range(self.population_size)]
        self.current_index %= len(self.population)
        self.online_net.load_state_dict(self.population[self.current_index].genome)
        self.online_net = self.online_net.to(self.device)
        self.online_net.eval()

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        *,
        snake_length: int = 1,
        endgame: bool = False,
    ) -> tuple[int, np.ndarray, bool, int]:
        return self._select_action_with_network(state, self.online_net, endgame=endgame)

    def select_best_action(self, state: np.ndarray, *, endgame: bool = False) -> tuple[int, np.ndarray, bool, int]:
        if self.best_genome is None:
            return self.select_action(state, training=False, endgame=endgame)
        return self._select_action_with_genome(state, self.best_genome, endgame=endgame)

    def _select_action_with_network(
        self,
        state: np.ndarray,
        network: PolicyNetwork,
        *,
        endgame: bool = False,
    ) -> tuple[int, np.ndarray, bool, int]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_scores = network(state_tensor).squeeze(0)
        action = int(torch.argmax(action_scores).item())
        return action, action_scores.detach().cpu().numpy(), False, action

    def _select_action_with_genome(
        self,
        state: np.ndarray,
        genome: dict[str, torch.Tensor],
        *,
        endgame: bool = False,
    ) -> tuple[int, np.ndarray, bool, int]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_scores = functional_call(self.online_net, genome, (state_tensor,)).squeeze(0)
        action = int(torch.argmax(action_scores).item())
        return action, action_scores.detach().cpu().numpy(), False, action

    def select_individual_action(self, index: int, state: np.ndarray, *, endgame: bool = False) -> tuple[int, np.ndarray, bool, int]:
        if not self.population:
            return self.select_action(state, training=True, endgame=endgame)
        safe_index = max(0, min(index, len(self.population) - 1))
        return self._select_action_with_genome(state, self.population[safe_index].genome, endgame=endgame)

    def finish_episode(
        self,
        *,
        score: int,
        reward: float,
        steps: int,
        reason: str,
        board_size: int,
        trap_after_apple_count: int = 0,
        avoidable_collision_count: int = 0,
        max_steps_without_food: int = 0,
    ) -> float:
        fitness = self.finish_individual(
            self.current_index,
            score=score,
            reward=reward,
            steps=steps,
            reason=reason,
            board_size=board_size,
            trap_after_apple_count=trap_after_apple_count,
            avoidable_collision_count=avoidable_collision_count,
            max_steps_without_food=max_steps_without_food,
        )
        self.current_index += 1
        if self.current_index >= self.population_size:
            self.evolve_generation()
        else:
            self._load_current_genome()
        return fitness

    def finish_individual(
        self,
        index: int,
        *,
        score: int,
        reward: float,
        steps: int,
        reason: str,
        board_size: int,
        trap_after_apple_count: int = 0,
        avoidable_collision_count: int = 0,
        max_steps_without_food: int = 0,
    ) -> float:
        fitness = self._fitness(
            score=score,
            reward=reward,
            steps=steps,
            reason=reason,
            board_size=board_size,
            trap_after_apple_count=trap_after_apple_count,
            avoidable_collision_count=avoidable_collision_count,
            max_steps_without_food=max_steps_without_food,
        )
        safe_index = max(0, min(index, len(self.population) - 1))
        individual = self.population[safe_index]
        individual.fitness = fitness
        individual.score = int(score)
        individual.reward = float(reward)
        individual.steps = int(steps)
        individual.reason = reason
        self.last_fitness = fitness
        self.completed_evaluations += 1
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_score = int(score)
            self.best_genome = self._clone_state(individual.genome)
        if reason == "victory":
            self._remember_victory(individual)
        return fitness

    def _clone_individual(self, individual: EvolutionIndividual) -> EvolutionIndividual:
        return EvolutionIndividual(
            self._clone_state(individual.genome),
            individual.fitness,
            individual.score,
            individual.reward,
            individual.steps,
            individual.reason,
        )

    def _remember_victory(self, individual: EvolutionIndividual) -> None:
        if individual.reason != "victory":
            return
        self.victory_archive.append(self._clone_individual(individual))
        self.victory_archive = sorted(
            self.victory_archive,
            key=lambda item: (
                float(item.fitness if item.fitness is not None else float("-inf")),
                int(item.score),
                -int(item.steps),
            ),
            reverse=True,
        )[:EVOLUTION_VICTORY_ARCHIVE_SIZE]

    def _fitness(
        self,
        *,
        score: int,
        reward: float,
        steps: int,
        reason: str,
        board_size: int,
        trap_after_apple_count: int = 0,
        avoidable_collision_count: int = 0,
        max_steps_without_food: int = 0,
    ) -> float:
        board_cells = max(1, board_size * board_size)
        apple_value = float(score) * 220.0 + float(score * score) * 12.0
        survival_value = min(float(steps), float(board_cells * 2)) * 0.025
        reward_value = float(reward) * 4.0
        death_penalties = {
            "wall_collision": 210.0,
            "body_collision": 260.0,
            "starvation": 135.0,
            "starvation_food_unreachable": 115.0,
            "starvation_no_relayout_path": 150.0,
            "starvation_food_available": 560.0,
            "starvation_safe_loop": 390.0,
            EVOLUTION_HUNGER_CULL_REASON: 520.0,
        }
        death_penalty = death_penalties.get(reason, 0.0)
        if reason in {"wall_collision", "body_collision"}:
            early_target = max(40, board_size * 5)
            death_penalty += max(0.0, float(early_target - steps)) * 1.4
            fill_ratio = min(1.0, float(score + 3) / float(board_cells))
            death_penalty += fill_ratio * fill_ratio * 5000.0
            collision_score_penalty = max(0.0, float(score - 6)) * (150.0 if reason == "body_collision" else 115.0)
            late_collision_penalty = max(0.0, float(score - 9)) ** 2 * (24.0 if reason == "body_collision" else 18.0)
            long_run_collision_penalty = min(
                1200.0,
                max(0.0, float(steps) - board_size * board_size * 0.7) * (0.95 if reason == "body_collision" else 0.7),
            )
            death_penalty += collision_score_penalty + late_collision_penalty + long_run_collision_penalty
            if score >= int(board_cells * 0.7):
                death_penalty += 6000.0 if reason == "body_collision" else 4500.0
            if reason == "body_collision":
                death_penalty += max(0.0, float(3 - score)) * 30.0
                death_penalty += max(0.0, float(board_size * 2.5 - steps)) * 0.9
        if reason == "starvation_safe_loop":
            no_progress_penalty = max(0.0, float(18 - score)) * 150.0
            idle_loop_penalty = min(620.0, max(0.0, float(steps - board_size * board_size * 0.55)) * 0.85)
            death_penalty += no_progress_penalty + idle_loop_penalty
        if reason == "starvation_food_available":
            no_progress_penalty = max(0.0, float(16 - score)) * 155.0
            idle_food_penalty = min(980.0, max(0.0, float(steps - board_size * board_size * 0.3)) * 1.25)
            death_penalty += no_progress_penalty + idle_food_penalty
        trap_after_apple_penalty = float(max(0, trap_after_apple_count)) * 650.0
        avoidable_collision_penalty = float(max(0, avoidable_collision_count)) * 900.0
        if reason.startswith("starvation") and score > 0:
            if reason in {"starvation_safe_loop", "starvation_food_available"}:
                death_penalty *= 1.0
            else:
                death_penalty *= max(0.68, 1.0 - min(score, 7) * 0.045)
        efficiency_budget = max(35.0, board_size * 3.0 + float(score) * board_size * 2.5)
        inefficient_steps = max(0.0, float(steps) - efficiency_budget)
        if reason in {"starvation_safe_loop", "starvation_food_available"}:
            efficiency_penalty = inefficient_steps * 0.9
        else:
            efficiency_penalty = inefficient_steps * (0.55 if reason.startswith("starvation") else 0.24)
        if score == 0:
            efficiency_penalty *= 1.45
        hunger_budget = hunger_budget_for(score, board_size)
        hunger_overage = max(0.0, float(max_steps_without_food) - hunger_budget)
        # Увеличен коэффициент с 1.6 до 4.0 для соответствия масштабу других штрафов (starvation/collision)
        hunger_penalty = hunger_overage * hunger_overage * 4.0
        if score <= max(3, int(board_cells * 0.12)):
            # Усилен множитель для ранних стадий с 2.0 до 3.0
            hunger_penalty *= 3.0
        elif score >= int(board_cells * 0.70):
            hunger_penalty *= 0.85
        victory_bonus = 15000.0 if reason == "victory" else 0.0
        return (
            apple_value
            + survival_value
            + reward_value
            + victory_bonus
            - death_penalty
            - efficiency_penalty
            - hunger_penalty
            - trap_after_apple_penalty
            - avoidable_collision_penalty
        )

    def evolve_generation(self) -> None:
        evaluated = [item for item in self.population if item.fitness is not None]
        if not evaluated:
            self.current_index = 0
            self._load_current_genome()
            return
        ranked = sorted(evaluated, key=lambda item: float(item.fitness), reverse=True)
        self.last_generation_average = float(sum(float(item.fitness) for item in evaluated) / len(evaluated))
        elites = [self._clone_individual(item) for item in ranked[: self.elite_count]]
        next_population: list[EvolutionIndividual] = elites
        random_slots = self.random_injection
        victory_slots = min(
            EVOLUTION_VICTORY_SEED_COUNT,
            len(self.victory_archive),
            max(0, self.population_size - len(next_population) - random_slots),
        )
        for slot in range(victory_slots):
            champion = self.victory_archive[slot % len(self.victory_archive)]
            if slot == 0:
                next_population.append(self._clone_individual(champion))
            else:
                next_population.append(
                    EvolutionIndividual(
                        self._mutate(
                            champion.genome,
                            mutation_rate_scale=EVOLUTION_VICTORY_MUTATION_RATE_SCALE,
                            mutation_strength_scale=EVOLUTION_VICTORY_MUTATION_STRENGTH_SCALE,
                        )
                    )
                )
        while len(next_population) + random_slots < self.population_size:
            parent_a = self._select_parent(ranked)
            parent_b = self._select_parent(ranked)
            child = self._crossover(parent_a.genome, parent_b.genome)
            child = self._mutate(child)
            next_population.append(EvolutionIndividual(child))
        while len(next_population) < self.population_size:
            next_population.append(EvolutionIndividual(self._fresh_genome()))
        self.population = next_population[: self.population_size]
        self.current_index = 0
        self.generation += 1
        self._load_current_genome()

    def _select_parent(self, ranked: list[EvolutionIndividual]) -> EvolutionIndividual:
        tournament_size = min(7, len(ranked))
        candidates = random.sample(ranked, tournament_size)
        return max(candidates, key=lambda item: float(item.fitness if item.fitness is not None else float("-inf")))

    def _crossover(self, first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if random.random() > self.crossover_rate:
            return self._clone_state(first)
        child: dict[str, torch.Tensor] = {}
        for key, first_value in first.items():
            second_value = second.get(key, first_value)
            if first_value.dtype.is_floating_point:
                mask = torch.rand_like(first_value.float()) < 0.5
                child[key] = torch.where(mask, first_value.float(), second_value.float()).to(dtype=first_value.dtype).cpu()
            else:
                child[key] = first_value.detach().cpu().clone()
        return child

    def _mutate(
        self,
        genome: dict[str, torch.Tensor],
        *,
        mutation_rate_scale: float = 1.0,
        mutation_strength_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        mutated: dict[str, torch.Tensor] = {}
        mutation_rate = max(0.0, min(1.0, self.mutation_rate * float(mutation_rate_scale)))
        mutation_strength = max(0.0, self.mutation_strength * float(mutation_strength_scale))
        for key, value in genome.items():
            tensor = value.detach().cpu().clone()
            if tensor.dtype.is_floating_point:
                mask = torch.rand_like(tensor.float()) < mutation_rate
                noise = torch.randn_like(tensor.float()) * mutation_strength
                tensor = tensor.float() + mask.float() * noise
                tensor = tensor.to(dtype=value.dtype)
            mutated[key] = tensor
        return mutated

    def _serialized_genome(self, genome: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().to(dtype=torch.float16) for key, value in genome.items()}

    def _individual_payload(self, individual: EvolutionIndividual) -> dict[str, object]:
        return {
            "genome": self._serialized_genome(individual.genome),
            "fitness": individual.fitness,
            "score": individual.score,
            "reward": individual.reward,
            "steps": individual.steps,
            "reason": individual.reason,
        }

    def _load_genome_payload(self, genome_payload: object, current_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        if not isinstance(genome_payload, dict):
            return None
        genome = self._clone_state(current_state)
        matched = 0
        for key, value in genome_payload.items():
            if key in current_state and torch.is_tensor(value) and current_state[key].shape == value.shape:
                genome[key] = value.detach().cpu().to(dtype=current_state[key].dtype)
                matched += 1
        return genome if matched else None

    def _load_individual_payload(self, item: object, current_state: dict[str, torch.Tensor]) -> EvolutionIndividual | None:
        if not isinstance(item, dict):
            return None
        genome = self._load_genome_payload(item.get("genome"), current_state)
        if genome is None:
            return None
        return EvolutionIndividual(
            genome=genome,
            fitness=None if item.get("fitness") is None else float(item.get("fitness", 0.0)),
            score=int(item.get("score", 0)),
            reward=float(item.get("reward", 0.0)),
            steps=int(item.get("steps", 0)),
            reason=str(item.get("reason", "loaded")),
        )

    def _ranked_population(self) -> list[EvolutionIndividual]:
        return sorted(
            self.population,
            key=lambda item: float(item.fitness if item.fitness is not None else float("-inf")),
            reverse=True,
        )

    def snapshot(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            activations = self.online_net.forward_with_activations(tensor)
        return {name: value.detach().cpu().numpy() for name, value in activations.items()}

    def move_to_device(self, device: torch.device) -> None:
        self.device = device
        self.online_net = self.online_net.to(device)
        self._load_current_genome()

    def export_checkpoint(self, metadata: dict[str, object]) -> dict[str, object]:
        ranked = self._ranked_population()
        best_individual = ranked[0] if ranked else None
        return {
            "checkpoint_mode": "elite",
            "elites": [] if best_individual is None else [self._individual_payload(best_individual)],
            "victory_archive": [],
            "best_genome": None if self.best_genome is None else self._serialized_genome(self.best_genome),
        }

    def import_checkpoint(self, payload: dict[str, object]) -> None:
        loaded_population: list[EvolutionIndividual] = []
        current_state = self.online_net.state_dict()
        raw_elites = payload.get("elites")
        raw_population = raw_elites if isinstance(raw_elites, list) else payload.get("population")
        if isinstance(raw_population, list):
            for item in raw_population:
                individual = self._load_individual_payload(item, current_state)
                if individual is not None:
                    loaded_population.append(individual)
        if loaded_population:
            ranked_loaded = sorted(
                loaded_population,
                key=lambda item: float(item.fitness if item.fitness is not None else float("-inf")),
                reverse=True,
            )
            self.population = [self._clone_individual(item) for item in ranked_loaded[: self.elite_count]]
            while len(self.population) < self.population_size:
                parent = random.choice(ranked_loaded)
                self.population.append(EvolutionIndividual(self._mutate(parent.genome)))
        loaded_victories: list[EvolutionIndividual] = []
        raw_victory_archive = payload.get("victory_archive")
        if isinstance(raw_victory_archive, list):
            for item in raw_victory_archive:
                individual = self._load_individual_payload(item, current_state)
                if individual is not None and individual.reason == "victory":
                    loaded_victories.append(individual)
        if loaded_victories:
            self.victory_archive = []
            for individual in loaded_victories:
                self._remember_victory(individual)
        best_payload = payload.get("best_genome")
        best_genome = self._load_genome_payload(best_payload, current_state)
        if best_genome is not None:
            self.best_genome = best_genome
        self._load_current_genome()


class ParallelEvolutionRunner:
    def __init__(
        self,
        population_size: int,
        board_size: int,
        *,
        env_factory: Callable[[int], Any],
        action_labels: tuple[str, ...] = MODEL_ACTION_LABELS,
        hunger_cull_enabled: bool = True,
        hunger_cull_protected_ratio: float = EVOLUTION_HUNGER_CULL_PROTECTED_RATIO,
        hunger_cull_min_protected: int = EVOLUTION_HUNGER_CULL_MIN_PROTECTED,
    ) -> None:
        self.population_size = population_size
        self.board_size = board_size
        self.env_factory = env_factory
        self.action_labels = action_labels
        self.hunger_cull_enabled = bool(hunger_cull_enabled)
        self.hunger_cull_protected_ratio = max(0.0, min(1.0, float(hunger_cull_protected_ratio)))
        self.hunger_cull_min_protected = max(1, int(hunger_cull_min_protected))
        self.envs: list[Any] = []
        self.done: list[bool] = []
        self.episode_rewards: list[float] = []
        self.last_fitnesses: list[float | None] = []
        self.step_logs: list[Deque[dict[str, Any]]] = []
        self.best_completed: dict[str, Any] | None = None
        self.leader_index = -1
        self.completed_count = 0
        self.last_profile_ms: dict[str, float | int] = {}
        self.reset_generation(board_size=board_size)

    @staticmethod
    def _manhattan(first: tuple[int, int], second: tuple[int, int]) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @staticmethod
    def _relative_to_absolute(current_direction: int, action: int) -> int:
        if action == 0:
            return current_direction
        if action == 1:
            return (current_direction - 1) % 4
        return (current_direction + 1) % 4

    def reset_generation(self, *, board_size: int | None = None) -> None:
        if board_size is not None:
            self.board_size = board_size
        self.envs = [self.env_factory(self.board_size) for _ in range(self.population_size)]
        self.done = [False] * self.population_size
        self.episode_rewards = [0.0] * self.population_size
        self.last_fitnesses = [None] * self.population_size
        self.step_logs = [deque(maxlen=120) for _ in range(self.population_size)]
        self.best_completed = None
        self.leader_index = -1
        self.completed_count = 0
        self.last_profile_ms = {}

    def alive_count(self) -> int:
        return sum(1 for item in self.done if not item)

    def all_done(self) -> bool:
        return self.alive_count() == 0

    def hunger_cull_protected_count(self) -> int:
        ratio_count = int(self.population_size * self.hunger_cull_protected_ratio + 0.999)
        return min(self.population_size, max(self.hunger_cull_min_protected, ratio_count))

    def _can_cull_for_hunger(self) -> bool:
        return self.hunger_cull_enabled and self.alive_count() > self.hunger_cull_protected_count()

    def _should_cull_for_hunger(self, env: Any) -> bool:
        if not self._can_cull_for_hunger():
            return False
        hunger_budget = hunger_budget_for(int(env.score), int(env.board_size))
        return float(env.steps_without_food) > hunger_budget

    def _apply_hunger_cull(self, env: Any, reward: float, info: dict[str, Any]) -> tuple[float, bool, dict[str, Any]]:
        # Исправлено: убрано прямое изменение атрибутов env (env.done, env.last_reward и т.д.), 
        # так как это нарушает инкапсуляцию симулятора и вызывает зависания при следующем шаге.
        reward = min(float(reward), -5.0)
        
        penalty_reasons = [str(item) for item in info.get("penalty_reasons", [])]
        if EVOLUTION_HUNGER_CULL_REASON not in penalty_reasons:
            penalty_reasons.append(EVOLUTION_HUNGER_CULL_REASON)
            
        info = dict(info)
        info["reason"] = EVOLUTION_HUNGER_CULL_REASON
        info["penalty_reasons"] = penalty_reasons
        info["hunger_budget"] = hunger_budget_for(int(env.score), int(env.board_size))
        info["hunger_cull_protected"] = self.hunger_cull_protected_count()
        
        return reward, True, info

    def _display_stale_loop(self, index: int) -> bool:
        env = self.envs[index]
        if self.done[index] or not env.snake:
            return False
        
        # Безопасный доступ к recent_heads с защитой от роста без ограничений
        recent_heads_raw = getattr(env, "recent_heads", None)
        if isinstance(recent_heads_raw, (list, tuple)):
            recent_heads = list(recent_heads_raw)[-12:]
        elif hasattr(recent_heads_raw, '__iter__'):
            # Для deque или других итерируемых объектов берём последние элементы безопасно
            from collections import deque
            recent_heads = list(deque(recent_heads_raw, maxlen=12))[-12:] if isinstance(recent_heads_raw, deque) else []
        else:
            recent_heads = []
            
        unique_recent = len(set(recent_heads))
        small_snake = len(env.snake) <= 5
        long_without_food = env.steps_without_food >= max(18, env.board_size * 2)
        tight_loop = len(recent_heads) >= 10 and unique_recent <= 4
        repeated_loop_reason = getattr(env, "last_reason", "") == "looping" and env.steps_without_food >= max(12, env.board_size)
        return small_snake and long_without_food and (tight_loop or repeated_loop_reason)

    def _leader_value(self, index: int) -> float:
        env = self.envs[index]
        distance = self._manhattan(env.snake[0], env.food) if env.snake else 0
        alive_bonus = 10_000.0 if not self.done[index] else 0.0
        
        # Безопасный доступ к recent_heads
        recent_heads_raw = getattr(env, "recent_heads", None)
        if isinstance(recent_heads_raw, (list, tuple)):
            recent_heads = list(recent_heads_raw)[-12:]
        elif hasattr(recent_heads_raw, '__iter__'):
            from collections import deque
            recent_heads = list(deque(recent_heads_raw, maxlen=12))[-12:] if isinstance(recent_heads_raw, deque) else []
        else:
            recent_heads = []
            
        unique_recent = len(set(recent_heads))
        stale_penalty = 35_000.0 if self._display_stale_loop(index) else 0.0
        return (
            alive_bonus
            + env.score * 100_000.0
            + len(env.snake) * 700.0
            + unique_recent * 120.0
            - env.steps_without_food * 95.0
            - distance * 45.0
            - stale_penalty
        )

    def update_leader(self) -> int:
        if not self.envs:
            self.leader_index = 0
            return self.leader_index
        current_valid = 0 <= self.leader_index < len(self.envs)
        if current_valid and not self.done[self.leader_index] and not self._display_stale_loop(self.leader_index):
            return self.leader_index
        alive_indices = [index for index, done in enumerate(self.done) if not done]
        if current_valid and not self.done[self.leader_index]:
            fresh_alive = [index for index in alive_indices if not self._display_stale_loop(index)]
            if not fresh_alive:
                return self.leader_index
            candidates = fresh_alive
        else:
            candidates = alive_indices or list(range(len(self.envs)))
        self.leader_index = max(candidates, key=self._leader_value)
        return self.leader_index

    def leader_env(self) -> Any:
        self.update_leader()
        return self.envs[self.leader_index]

    def step_alive(
        self,
        agent: EvolutionAgent,
        *,
        input_grid: int,
        input_mode: str,
    ) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        observation_ms = 0.0
        inference_ms = 0.0
        env_step_ms = 0.0
        bookkeeping_ms = 0.0
        alive_processed = 0
        for index, env in enumerate(self.envs):
            if self.done[index]:
                continue
            alive_processed += 1
            if self._should_cull_for_hunger(env):
                part_started_at = time.perf_counter()
                action = 0
                action_scores = np.zeros(len(self.action_labels), dtype=np.float32)
                was_random = False
                greedy_action = 0
                reward, done, info = self._apply_hunger_cull(
                    env,
                    -5.0,
                    {
                        "steps": env.step_count,
                        "reason": EVOLUTION_HUNGER_CULL_REASON,
                        "penalty_reasons": [EVOLUTION_HUNGER_CULL_REASON],
                    },
                )
                bookkeeping_ms += (time.perf_counter() - part_started_at) * 1000.0
                step_number = int(info.get("steps", env.step_count))
                self.step_logs[index].append(
                    {
                        "step": step_number,
                        "reward": float(reward),
                        "train_reward": float(reward),
                        "action": int(action),
                        "action_label": self.action_labels[action] if 0 <= action < len(self.action_labels) else "-",
                        "reason": EVOLUTION_HUNGER_CULL_REASON,
                        "penalties": [EVOLUTION_HUNGER_CULL_REASON],
                        "random": False,
                        "individual": index + 1,
                        "apple_opportunity": None,
                    }
                )
                fitness = agent.finish_individual(
                    index,
                    score=env.score,
                    reward=self.episode_rewards[index] + reward,
                    steps=env.step_count,
                    reason=EVOLUTION_HUNGER_CULL_REASON,
                    board_size=env.board_size,
                    trap_after_apple_count=env.trap_after_apple_count,
                    avoidable_collision_count=env.avoidable_collision_count,
                    max_steps_without_food=env.max_steps_without_food,
                )
                self.episode_rewards[index] += reward
                self.last_fitnesses[index] = fitness
                self.done[index] = True
                self.completed_count += 1
                item = {
                    "index": index,
                    "env": env,
                    "reason": EVOLUTION_HUNGER_CULL_REASON,
                    "reward": self.episode_rewards[index],
                    "fitness": fitness,
                    "action": action,
                    "action_scores": action_scores,
                    "random": was_random,
                    "greedy_action": greedy_action,
                    "penalty_reasons": [EVOLUTION_HUNGER_CULL_REASON],
                }
                completed.append(item)
                if self.best_completed is None or fitness > float(self.best_completed.get("fitness", float("-inf"))):
                    self.best_completed = dict(item)
                continue
            part_started_at = time.perf_counter()
            state = env.get_observation_grid(input_grid, input_mode)
            observation_ms += (time.perf_counter() - part_started_at) * 1000.0
            part_started_at = time.perf_counter()
            action, action_scores, was_random, greedy_action = agent.select_individual_action(index, state, endgame=env._is_endgame())
            inference_ms += (time.perf_counter() - part_started_at) * 1000.0
            absolute_direction = self._relative_to_absolute(env.direction, action)
            part_started_at = time.perf_counter()
            _, reward, done, info = env.step(absolute_direction)
            if not done and self._should_cull_for_hunger(env):
                reward, done, info = self._apply_hunger_cull(env, reward, info)
            env_step_ms += (time.perf_counter() - part_started_at) * 1000.0
            part_started_at = time.perf_counter()
            self.episode_rewards[index] += reward
            step_number = int(info.get("steps", env.step_count))
            if step_number <= 1:
                self.step_logs[index].clear()
            self.step_logs[index].append(
                {
                    "step": step_number,
                    "reward": float(reward),
                    "train_reward": float(reward),
                    "action": int(action),
                    "action_label": self.action_labels[action] if 0 <= action < len(self.action_labels) else "-",
                    "reason": str(info.get("reason", env.last_reason)),
                    "penalties": [str(item) for item in info.get("penalty_reasons", [])],
                    "random": bool(was_random),
                    "individual": index + 1,
                    "apple_opportunity": info.get("apple_opportunity"),
                }
            )
            if done:
                reason = str(info.get("reason", env.last_reason))
                fitness = agent.finish_individual(
                    index,
                    score=env.score,
                    reward=self.episode_rewards[index],
                    steps=env.step_count,
                    reason=reason,
                    board_size=env.board_size,
                    trap_after_apple_count=env.trap_after_apple_count,
                    avoidable_collision_count=env.avoidable_collision_count,
                    max_steps_without_food=env.max_steps_without_food,
                )
                self.last_fitnesses[index] = fitness
                self.done[index] = True
                self.completed_count += 1
                item = {
                    "index": index,
                    "env": env,
                    "reason": reason,
                    "reward": self.episode_rewards[index],
                    "fitness": fitness,
                    "action": action,
                    "action_scores": action_scores,
                    "random": was_random,
                    "greedy_action": greedy_action,
                    "penalty_reasons": list(info.get("penalty_reasons", [])),
                }
                completed.append(item)
                if self.best_completed is None or fitness > float(self.best_completed.get("fitness", float("-inf"))):
                    self.best_completed = dict(item)
            bookkeeping_ms += (time.perf_counter() - part_started_at) * 1000.0
        leader_started_at = time.perf_counter()
        self.update_leader()
        leader_ms = (time.perf_counter() - leader_started_at) * 1000.0
        bookkeeping_ms += leader_ms
        self.last_profile_ms = {
            "alive": alive_processed,
            "observation": round(observation_ms, 1),
            "inference": round(inference_ms, 1),
            "env_step": round(env_step_ms, 1),
            "bookkeeping": round(bookkeeping_ms, 1),
            "leader": round(leader_ms, 1),
        }
        return completed
