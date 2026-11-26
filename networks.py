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
    PPO_CONV1_CHANNELS,
    PPO_CONV2_CHANNELS,
    PPO_HIDDEN_LAYER_1,
    PPO_HIDDEN_LAYER_2,
    PPO_MAP_CHANNELS,
    PPO_MAP_FEAT_SIZE,
    PPO_MAP_SIZE,
    PPO_SIGNAL_LABELS,
)

class BaseSignalNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = len(GLOBAL_SIGNAL_LABELS),
        *,
        hidden1: int = HIDDEN_LAYER_1,
        hidden2: int = HIDDEN_LAYER_2,
        input_adapter_mode: str = NETWORK_INPUT_ADAPTER_MODE,
        warn_on_adapt: bool = NETWORK_INPUT_WARN_ON_ADAPT,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden1_size = int(hidden1)
        self.hidden2_size = int(hidden2)
        self.input_adapter_mode = input_adapter_mode
        self.warn_on_adapt = warn_on_adapt
        self._adapter_warning_emitted = False

        self.hidden1_layer = nn.Linear(self.input_size, self.hidden1_size)
        self.hidden2_layer = nn.Linear(self.hidden1_size, self.hidden2_size)

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
    def __init__(
        self,
        input_size: int = len(GLOBAL_SIGNAL_LABELS),
        *,
        hidden1: int = HIDDEN_LAYER_1,
        hidden2: int = HIDDEN_LAYER_2,
        input_adapter_mode: str = NETWORK_INPUT_ADAPTER_MODE,
    ) -> None:
        super().__init__(
            input_size,
            hidden1=hidden1,
            hidden2=hidden2,
            input_adapter_mode=input_adapter_mode,
        )
        self.value_head = nn.Linear(self.hidden2_size, 1)
        self.value_head.bias.data.zero_()
        self.advantage_head = nn.Linear(self.hidden2_size, len(MODEL_ACTION_LABELS))
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

def _ortho_init(layer: nn.Module, gain: float = 1.41421356237) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class PPOMapNetwork(nn.Module):
    """Гибридная Actor-Critic сеть для PPO: эгоцентричная 2D-карта + вектор глобальных сигналов.

    После fusion-слоя ветки Actor (policy) и Critic (value) разделены на независимые
    скрытые слои, чтобы градиенты value-функции не разрушали политику.
    """

    def __init__(
        self,
        input_size: int = len(PPO_SIGNAL_LABELS),
        *,
        map_channels: int = PPO_MAP_CHANNELS,
        map_size: int = PPO_MAP_SIZE,
        conv1_channels: int = PPO_CONV1_CHANNELS,
        conv2_channels: int = PPO_CONV2_CHANNELS,
        map_feat_size: int = PPO_MAP_FEAT_SIZE,
        hidden1: int = PPO_HIDDEN_LAYER_1,
        hidden2: int = PPO_HIDDEN_LAYER_2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.map_channels = int(map_channels)
        self.map_size = int(map_size)
        self.conv1_channels = int(conv1_channels)
        self.conv2_channels = int(conv2_channels)
        self.map_feat_size = int(map_feat_size)
        self.hidden1_size = int(hidden1)
        self.hidden2_size = int(hidden2)

        self.map_conv = nn.Sequential(
            nn.Conv2d(self.map_channels, self.conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv1_channels, self.conv2_channels, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.map_channels, self.map_size, self.map_size)
            conv_out_dim = int(self.map_conv(dummy).shape[1])

        self.map_proj = nn.Linear(conv_out_dim, self.map_feat_size)
        fused_dim = self.input_size + self.map_feat_size

        # Ветка Actor (policy): первые input_size столбцов hidden1_layer напрямую
        # соответствуют GLOBAL_SIGNAL_LABELS для совместимости с графом сети в UI.
        self.hidden1_layer = nn.Linear(fused_dim, self.hidden1_size)
        self.hidden2_layer = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.output = nn.Linear(self.hidden2_size, len(MODEL_ACTION_LABELS))

        # Независимая ветка Critic (value) после fusion-слоя
        self.critic_hidden1_layer = nn.Linear(fused_dim, self.hidden1_size)
        self.critic_hidden2_layer = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.value_head = nn.Linear(self.hidden2_size, 1)

        for mod in self.map_conv:
            _ortho_init(mod, gain=1.41421356237)
        _ortho_init(self.map_proj, gain=0.25)
        _ortho_init(self.hidden1_layer, gain=1.41421356237)
        _ortho_init(self.hidden2_layer, gain=1.41421356237)
        _ortho_init(self.output, gain=0.01)
        _ortho_init(self.critic_hidden1_layer, gain=1.41421356237)
        _ortho_init(self.critic_hidden2_layer, gain=1.41421356237)
        _ortho_init(self.value_head, gain=1.0)
        self.map_conv.to(memory_format=torch.channels_last)

    def _prepare_pair(
        self,
        signals: torch.Tensor,
        ego_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if signals.dim() == 1:
            signals = signals.unsqueeze(0)
        if signals.dim() != 2 or int(signals.shape[1]) != self.input_size:
            raise ValueError(f"Expected signals shape (*, {self.input_size}), got {tuple(signals.shape)}")
        target_dtype = self.hidden1_layer.weight.dtype
        target_device = self.hidden1_layer.weight.device
        signals = signals.to(device=target_device, dtype=target_dtype)

        if ego_map is None:
            ego_map = torch.zeros(
                (signals.shape[0], self.map_channels, self.map_size, self.map_size),
                dtype=target_dtype,
                device=target_device,
            )
        else:
            if ego_map.dim() == 3:
                ego_map = ego_map.unsqueeze(0)
            ego_map = ego_map.to(device=target_device, dtype=target_dtype, memory_format=torch.channels_last)
        return signals, ego_map

    def forward_features(
        self,
        signals: torch.Tensor,
        ego_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sig_t, map_t = self._prepare_pair(signals, ego_map)
        map_feat = torch.tanh(self.map_proj(self.map_conv(map_t)))
        fused = torch.cat((sig_t, map_feat), dim=1)
        actor_h1 = torch.tanh(self.hidden1_layer(fused))
        actor_h2 = torch.tanh(self.hidden2_layer(actor_h1))
        critic_h1 = torch.tanh(self.critic_hidden1_layer(fused))
        critic_h2 = torch.tanh(self.critic_hidden2_layer(critic_h1))
        return sig_t, map_feat, actor_h1, actor_h2, critic_h1, critic_h2

    def forward(
        self,
        signals: torch.Tensor,
        ego_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _, actor_h2, _, critic_h2 = self.forward_features(signals, ego_map)
        logits = self.output(actor_h2)
        value = self.value_head(critic_h2).squeeze(-1)
        return logits, value

    def forward_with_activations(
        self,
        signals: torch.Tensor,
        ego_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sig_t, map_feat, actor_h1, actor_h2, critic_h1, critic_h2 = self.forward_features(signals, ego_map)
        logits = self.output(actor_h2)
        value = self.value_head(critic_h2)
        return {
            "input": sig_t.squeeze(0),
            "map_features": map_feat.squeeze(0),
            "hidden1": actor_h1.squeeze(0),
            "hidden2": actor_h2.squeeze(0),
            "critic_hidden1": critic_h1.squeeze(0),
            "critic_hidden2": critic_h2.squeeze(0),
            "output": logits.squeeze(0),
            "value": value.squeeze(0),
        }
