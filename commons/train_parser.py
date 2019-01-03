"""
argument_parser.py
"""
import argparse


def get_train_args(description='endtoendai/baselines', default_args=None):
    """
    Parse arguments for training agents and return hyperparameters as a
    Namespace.

    Parameters
    ----------
    description: str
        Description for the argument parser. Defaults to endtoendai/baselines.

    Returns
    -------
    args
        Namespace containing hyperparameter options specified by user or set
        by default.
    """

    parser = argparse.ArgumentParser(description)
    parser.add_argument('--env-id', action='store', dest='ENV_ID',
                        default='Pong', type=str,
                        help='Environment to train the agent in. Defaults to Pong.')

    # Hyperparameters for DQN
    parser.add_argument('--lr', action='store', dest='LR',
                        default=1e-4, type=float,
                        help='Learning rate for optimizing DQN.')
    parser.add_argument('--discount', action='store', dest='DISCOUNT',
                        default=0.99, type=float,
                        help='Discount factor for DQN.')
    parser.add_argument('--epsilon', action='store', dest='EPSILON',
                        default=0.1, type=float,
                        help='Epsilon for epsilon-greedy exploration DQN.')
    args = parser.parse_args()

    if args.ENV_ID not in ['Pong']:
        raise ValueError('{} is not a supported environment.'.format(args.ENV_ID))

    return args
