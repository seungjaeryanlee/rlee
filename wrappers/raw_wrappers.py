import gym
import numpy as np
import torch


class AcrobotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, original_obs):
        screen = self.env.render(mode='rgb_array')
        screen = np.ascontiguousarray(screen)

        return screen
