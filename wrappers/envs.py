"""
envs.py
"""

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
        # TODO Create environment
        env_id = env_id + 'NoFrameskip-v4'
        env = make_atari(env_id)

        # TODO Wrap environment to fit DeepMind-style environment
        env = wrap_deepmind(env, frame_stack=True)

        # TODO Wrap environment for PyTorch agents
        env = TorchWrapper(env)
    else:
        raise ValueError('{} is not a supported environment.'.format(env_id))
