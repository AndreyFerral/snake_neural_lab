from __future__ import annotations
import torch
import torch.nn as nn
from backend_config import (
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    GLOBAL_SIGNAL_LABELS,
    MODEL_ACTION_LABELS,
)

class BaseSignalNetwork(nn.Module):
    def __init__(self, input_size: int = len(GLOBAL_SIGNAL_LABELS)) -> None:
        super().__init__()
        self.input_size = input_size

        self.input_layer = nn.Linear(self.input_size, 64)
        self.hidden1_layer = nn.Linear(64, HIDDEN_LAYER_1)
        self.hidden2_layer = nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2)

    def forward_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.shape[1] > self.input_size:
            inputs = inputs[:, -self.input_size:]

        x = self.input_layer(inputs)
        x = torch.tanh(x)
        x = self.hidden1_layer(x)
        x = torch.tanh(x)
        hidden2 = self.hidden2_layer(x)
        hidden2 = torch.tanh(hidden2)

        return inputs, hidden2

    def forward_with_activations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.shape[1] > self.input_size:
            inputs = inputs[:, -self.input_size:]

        x = self.input_layer(inputs)
        x = torch.tanh(x)
        hidden1 = self.hidden1_layer(x)
        hidden1 = torch.tanh(hidden1)
        hidden2 = self.hidden2_layer(hidden1)
        hidden2 = torch.tanh(hidden2)

        return {
            "input": inputs.squeeze(0),
            "hidden1": hidden1.squeeze(0),
            "hidden2": hidden2.squeeze(0),
        }

class QNetwork(BaseSignalNetwork):
    def __init__(self, input_size: int = len(GLOBAL_SIGNAL_LABELS)) -> None:
        super().__init__(input_size)
        self.value_head = nn.Linear(HIDDEN_LAYER_2, 1)
        self.advantage_head = nn.Linear(HIDDEN_LAYER_2, len(MODEL_ACTION_LABELS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden2 = self.forward_features(inputs)
        value = self.value_head(hidden2)
        advantage = self.advantage_head(hidden2)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def forward_with_activations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        activations = super().forward_with_activations(inputs)
        hidden2 = activations["hidden2"].unsqueeze(0)
        value = self.value_head(hidden2)
        advantage = self.advantage_head(hidden2)
        output = value + advantage - advantage.mean(dim=1, keepdim=True)
        activations["output"] = output.squeeze(0)
        activations["value"] = value.squeeze(0)
        return activations

class PolicyNetwork(BaseSignalNetwork):
    def __init__(self, input_size: int = len(GLOBAL_SIGNAL_LABELS)) -> None:
        super().__init__(input_size)
        self.output = nn.Linear(HIDDEN_LAYER_2, len(MODEL_ACTION_LABELS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden2 = self.forward_features(inputs)
        return self.output(hidden2)

    def forward_with_activations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        activations = super().forward_with_activations(inputs)
        hidden2 = activations["hidden2"].unsqueeze(0)
        output = self.output(hidden2)
        activations["output"] = output.squeeze(0)
        activations["value"] = torch.empty(0, dtype=output.dtype, device=output.device)
        return activations