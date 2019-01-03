"""
envs.py
"""
from .atari_wrappers import make_atari, wrap_deepmind
from .torch_wrappers import wrap_pytorch


def make_env(env_id):
    """
    Return an OpenAI Gym environment wrapped with appropriate wrappers. Throws
    error if env_id is not recognized.

    Parameters
    ----------
    env_id : str
        OpenAI Gym ID for environment.

    Returns
    -------
    env
        Wrapped OpenAI Gym environment.
    """
    if env_id in ['Pong']:
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
