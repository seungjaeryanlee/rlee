from typing import Any

import gym
import numpy as np


class AcrobotWrapper(gym.ObservationWrapper):
    def __init__(self, env: Any):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, original_obs: Any) -> np.ndarray:
        screen = self.env.render(mode='rgb_array')
        screen = np.ascontiguousarray(screen)

        return screen
