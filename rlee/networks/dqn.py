"""
Deep Q-Network.

Human-level control through deep reinforcement learning
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
"""
from typing import List

import torch
import torch.nn as nn


class FCDQN(nn.Module):
    """Deep Q-network for environments with vector observations."""

    def __init__(
        self, num_inputs: int, num_actions: int, layer_sizes: List[int] = None
    ) -> None:
        assert num_inputs > 0
        assert layer_sizes is None or len(layer_sizes) >= 1
        if layer_sizes is None:
            layer_sizes = [64, 64]

        super().__init__()

        self.layers = []  # type: ignore
        last_layer_size = num_inputs
        for layer_size in layer_sizes:
            self.layers.append(nn.Linear(last_layer_size, layer_size))
            self.layers.append(nn.ReLU())
            last_layer_size = layer_size

        self.layers.append(nn.Linear(last_layer_size, num_actions))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Propagate DQN forward."""
        x = self.layers(x)  # type: ignore
        return x


class DQN(nn.Module):
    """
    Deep Q-network for Atari environments.

    The DQN architecture specified in DQN2015 paper. Expects 84x84 frame
    environments.
    """

    def __init__(self, num_inputs: int, num_actions: int) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Propagate DQN forward."""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
