"""Wrappers to use raw visual observations on easy Gym environments."""

from typing import Any

import gym
import numpy as np


class AcrobotWrapper(gym.ObservationWrapper):
    """Wrapper to use raw visual observation for Acrobot-v1."""

    def __init__(self, env: Any):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, original_obs: Any) -> np.ndarray:
        """Return raw visual input as observation."""
        screen = self.env.render(mode='rgb_array')
        screen = np.ascontiguousarray(screen)

        return screen
