"""OpenAI Gym Environment wrappers."""
from typing import Any

import gym

from .atari_wrappers import make_atari, wrap_deepmind, WarpFrame, FrameStack
from .raw_wrappers import AcrobotWrapper
from .torch_wrappers import wrap_pytorch


def make_env(env_id: str) -> Any:
    """
    Return an OpenAI Gym environment wrapped with appropriately.

    Throws error if unrecognized env_id is given.

    Parameters
    ----------
    env_id : str
        OpenAI Gym ID for environment.

    Returns
    -------
    env
        Wrapped OpenAI Gym environment.

    """
    if env_id in ['Acrobot']:
        # Create environment
        env_id = 'Acrobot-v1'
        env = gym.make(env_id)  # Don't use Frameskip, NoopReset

        env = AcrobotWrapper(env)

        # Wrap environment to fit DeepMind-style environment
        env = WarpFrame(env)
        env = FrameStack(env, 4)

        # Wrap environment for PyTorch agents
        env = wrap_pytorch(env)

    elif env_id in ['Pong']:
        # Create environment
        env_id = env_id + 'NoFrameskip-v4'
        env = make_atari(env_id)

        # Wrap environment to fit DeepMind-style environment
        env = wrap_deepmind(env, frame_stack=True)

        # Wrap environment for PyTorch agents
        env = wrap_pytorch(env)
    else:
        raise ValueError('{} is not a supported environment.'.format(env_id))

    return env
