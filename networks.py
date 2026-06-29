from __future__ import annotations
import torch
import torch.nn as nn
from backend_config import (
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    GLOBAL_SIGNAL_LABELS,
    MODEL_ACTION_LABELS,
    NETWORK_INPUT_ADAPTER_MODE,
    NETWORK_INPUT_WARN_ON_ADAPT,
    NETWORK_INPUT_PAD_VALUE,
)

class BaseSignalNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = len(GLOBAL_SIGNAL_LABELS),
        *,
        input_adapter_mode: str = NETWORK_INPUT_ADAPTER_MODE,
        warn_on_adapt: bool = NETWORK_INPUT_WARN_ON_ADAPT,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_adapter_mode = input_adapter_mode
        self.warn_on_adapt = warn_on_adapt
        self._adapter_warning_emitted = False

        self.hidden1_layer = nn.Linear(self.input_size, HIDDEN_LAYER_1)
        self.hidden2_layer = nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2)

    def load_state_dict(self, state_dict, strict=True):
        key = "hidden1_layer.weight"
        if key in state_dict:
            val = state_dict[key]
            curr_val = self.state_dict()[key]
            if val.shape != curr_val.shape and val.dim() == 2 and curr_val.dim() == 2:
                out_feat, in_feat = curr_val.shape
                s_out_feat, s_in_feat = val.shape
                if out_feat == s_out_feat:
                    if s_in_feat == 26 and in_feat == 27:
                        # 26 -> 27 adaptation (inserting food_reachable at index 7)
                        new_weight = curr_val.clone()
                        new_weight[:, :7] = val[:, :7]
                        new_weight[:, 7] = 0.0
                        new_weight[:, 8:] = val[:, 7:]
                    elif s_in_feat == 23 and in_feat == 27:
                        # 23 -> 27 adaptation (no partition, no food_reachable)
                        new_weight = curr_val.clone()
                        new_weight.fill_(0.0)
                        new_weight[:, :7] = val[:, :7]
                        new_weight[:, 8:24] = val[:, 7:23]
                    else:
                        # Default fallback
                        new_weight = curr_val.clone()
                        copy_cols = min(in_feat, s_in_feat)
                        new_weight[:, :copy_cols] = val[:, :copy_cols]
                        if in_feat > s_in_feat:
                            new_weight[:, s_in_feat:] = 0.0
                    
                    state_dict[key] = new_weight
                    if not self._adapter_warning_emitted:
                        print(f"[network] Adapted hidden1_layer.weight from loaded shape {tuple(val.shape)} to target shape {tuple(curr_val.shape)}")
                        self._adapter_warning_emitted = True
        return super().load_state_dict(state_dict, strict=strict)

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() != 2:
            raise ValueError(f"Expected 1D/2D tensor input, got shape={tuple(inputs.shape)}")
        if inputs.dtype != self.hidden1_layer.weight.dtype:
            inputs = inputs.to(dtype=self.hidden1_layer.weight.dtype)
        width = int(inputs.shape[1])
        if width == self.input_size:
            return inputs
        mode = self.input_adapter_mode
        if mode == "strict":
            raise ValueError(f"Input feature size mismatch: got {width}, expected {self.input_size}")
        if mode == "trim":
            if self.warn_on_adapt and not self._adapter_warning_emitted:
                print(f"[network] trim input from {width} to {self.input_size}")
                self._adapter_warning_emitted = True
            return inputs[:, -self.input_size:] if width > self.input_size else inputs
        if mode == "pad":
            if width > self.input_size:
                if self.warn_on_adapt and not self._adapter_warning_emitted:
                    print(f"[network] trim input from {width} to {self.input_size} in pad mode")
                    self._adapter_warning_emitted = True
                return inputs[:, -self.input_size:]
            pad_cols = self.input_size - width
            pad = torch.full(
                (inputs.shape[0], pad_cols),
                float(NETWORK_INPUT_PAD_VALUE),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if self.warn_on_adapt and not self._adapter_warning_emitted:
                print(f"[network] pad input from {width} to {self.input_size}")
                self._adapter_warning_emitted = True
            return torch.cat((inputs, pad), dim=1)
        raise ValueError(f"Unsupported input adapter mode: {mode}")

    def forward_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._prepare_inputs(inputs)

        x = self.hidden1_layer(inputs)
        x = torch.tanh(x)
        hidden2 = self.hidden2_layer(x)
        hidden2 = torch.tanh(hidden2)

        return inputs, hidden2

    def forward_with_activations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = self._prepare_inputs(inputs)

        hidden1 = self.hidden1_layer(inputs)
        hidden1 = torch.tanh(hidden1)
        hidden2 = self.hidden2_layer(hidden1)
        hidden2 = torch.tanh(hidden2)

        return {
            "input": inputs.squeeze(0),
            "hidden1": hidden1.squeeze(0),
            "hidden2": hidden2.squeeze(0),
        }

class QNetwork(BaseSignalNetwork):
    def __init__(self, input_size: int = len(GLOBAL_SIGNAL_LABELS), *, input_adapter_mode: str = NETWORK_INPUT_ADAPTER_MODE) -> None:
        super().__init__(input_size, input_adapter_mode=input_adapter_mode)
        self.value_head = nn.Linear(HIDDEN_LAYER_2, 1)
        self.value_head.bias.data.zero_()
        self.advantage_head = nn.Linear(HIDDEN_LAYER_2, len(MODEL_ACTION_LABELS))
        self.advantage_head.bias.data.zero_()

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
    def __init__(self, input_size: int = len(GLOBAL_SIGNAL_LABELS), *, input_adapter_mode: str = NETWORK_INPUT_ADAPTER_MODE) -> None:
        super().__init__(input_size, input_adapter_mode=input_adapter_mode)
        self.output = nn.Linear(HIDDEN_LAYER_2, len(MODEL_ACTION_LABELS))
        self.output.bias.data.zero_()

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
