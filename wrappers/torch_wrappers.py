"""OpenAI Gym Environment Wrappers for PyTorch-style observation."""

from typing import Any, Tuple

import gym
import numpy as np
import torch
from gym import spaces


class TorchTensorWrapper(gym.Wrapper):
    """
    Wrapper to change return types to `torch.Tensor`.

    OpenAI Gym Environment Wrapper that changes return types of
    `env.reset()` and `env.step()` to `torch.Tensor`.
    """

    def __init__(self, env: Any) -> None:
        gym.Wrapper.__init__(self, env)

    def reset(self) -> torch.Tensor:
        """Return `torch.Tensor` formatted observation of the wrapped env."""
        ob = self.env.reset()
        ob = torch.FloatTensor([ob])
        return ob

    def step(self, action: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Return `torch.Tensor` formatted returns of wrapped env."""
        ob, reward, done, info = self.env.step(action)
        ob = torch.FloatTensor([ob])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])
        return (ob, reward, done, info)


class TorchPermuteWrapper(gym.ObservationWrapper):
    """
    Wrapper to permute observation to PyTorch style: NCHW.

    OpenAI Gym Environment Wrapper for raw visual input type
    environments that permutes observation to PyTorch style: NCHW.
    """

    def __init__(self, env: Any) -> None:
        gym.ObservationWrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(shp[2], shp[0], shp[1]), dtype=np.float32
        )

    def observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Permute observation to PyTorch style."""
        return observation.permute(0, 3, 1, 2)


def wrap_pytorch(env: Any) -> Any:
    """Wrap environment to be compliant to PyTorch agents."""
    env = TorchTensorWrapper(env)
    env = TorchPermuteWrapper(env)

    return env
