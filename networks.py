from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend_config import (
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    LOCAL_BRANCH_HIDDEN,
    GLOBAL_BRANCH_HIDDEN,
    LOCAL_CONV_CHANNELS,
    LOCAL_MAP_CHANNEL_COUNT,
    local_input_size,
)


class HybridBoardEncoder(nn.Module):
    def __init__(self, input_grid: int, global_signal_count: int) -> None:
        super().__init__()
        self.input_grid = int(input_grid)
        self.global_signal_count = int(global_signal_count)
        self.local_input_size = local_input_size(input_grid)
        self.input_size = self.local_input_size + self.global_signal_count
        self.dense_connection_visualization = False
        if input_grid > 0:
            self.local_conv1 = nn.Conv2d(LOCAL_MAP_CHANNEL_COUNT, 16, kernel_size=3, padding=1)
            self.local_conv2 = nn.Conv2d(16, LOCAL_CONV_CHANNELS, kernel_size=3, padding=1)
            self.local_conv3 = nn.Conv2d(LOCAL_CONV_CHANNELS, LOCAL_CONV_CHANNELS, kernel_size=3, padding=1)
            self.local_pool = nn.AdaptiveAvgPool2d((2, 2))
            self.local_proj = nn.Linear(LOCAL_CONV_CHANNELS * 4, LOCAL_BRANCH_HIDDEN)
        else:
            self.local_conv1 = None
            self.local_conv2 = None
            self.local_conv3 = None
            self.local_pool = None
            self.local_proj = None
        if self.global_signal_count > 0:
            self.global_mlp = nn.Sequential(
                nn.Linear(self.global_signal_count, 64),
                nn.Tanh(),
                nn.Linear(64, GLOBAL_BRANCH_HIDDEN),
                nn.Tanh(),
            )
        else:
            self.global_mlp = None
        self.layer1 = nn.Linear(LOCAL_BRANCH_HIDDEN + GLOBAL_BRANCH_HIDDEN, HIDDEN_LAYER_1)
        self.layer2 = nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2)

    def merged_hidden(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        local_features: torch.Tensor | None = None
        global_features: torch.Tensor | None = None
        local_heatmap: torch.Tensor

        if (
            self.input_grid > 0
            and self.local_conv1 is not None
            and self.local_conv2 is not None
            and self.local_conv3 is not None
            and self.local_pool is not None
            and self.local_proj is not None
        ):
            local_flat = inputs[:, : self.local_input_size]
            local_map = local_flat.reshape(-1, LOCAL_MAP_CHANNEL_COUNT, self.input_grid, self.input_grid)
            conv1 = torch.tanh(self.local_conv1(local_map))
            conv2 = torch.tanh(self.local_conv2(conv1))
            conv3 = torch.tanh(self.local_conv3(conv2))
            local_encoded = self.local_pool(conv3).reshape(inputs.size(0), -1)
            local_features = torch.tanh(self.local_proj(local_encoded))
            local_heatmap = conv3.abs().mean(dim=1)
        else:
            local_heatmap = inputs.new_zeros((inputs.size(0), max(1, self.input_grid), max(1, self.input_grid)))

        if self.global_mlp is not None:
            global_inputs = inputs[:, self.local_input_size :]
            global_features = self.global_mlp(global_inputs)

        # Fast paths avoid allocating/copying a zero branch when the model is in
        # signal-only mode (the default startup path) or map-only mode.  The
        # layer shape stays checkpoint-compatible with the original architecture.
        if local_features is None and global_features is not None:
            hidden1 = torch.tanh(F.linear(global_features, self.layer1.weight[:, LOCAL_BRANCH_HIDDEN:], self.layer1.bias))
        elif local_features is not None and global_features is None:
            hidden1 = torch.tanh(F.linear(local_features, self.layer1.weight[:, :LOCAL_BRANCH_HIDDEN], self.layer1.bias))
        elif local_features is not None and global_features is not None:
            merged = torch.cat((local_features, global_features), dim=1)
            hidden1 = torch.tanh(self.layer1(merged))
        else:
            merged = inputs.new_zeros((inputs.size(0), LOCAL_BRANCH_HIDDEN + GLOBAL_BRANCH_HIDDEN))
            hidden1 = torch.tanh(self.layer1(merged))
        hidden2 = torch.tanh(self.layer2(hidden1))
        return inputs, hidden1, hidden2, local_heatmap
