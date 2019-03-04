"""Parser for evaluating an agent."""
from typing import Optional

import configargparse


def get_eval_args(
    description: str = "endtoendai/rlee", default_args: Optional[dict] = None
) -> configargparse.Namespace:
    """
    Parse arguments for evaluating agents and return hyperparameters as
    a Namespace.

    Parameters
    ----------
    description: str
        Description for the argument parser. Defaults to endtoendai/rlee.

    Returns
    -------
    args
        Namespace containing hyperparameter options specified by user or
        set by default.

    """
    parser = configargparse.ArgumentParser(description)
    parser.add(
        "-c",
        "--config",
        help="config file path",
        default="configs/pong/dqn2015.train.conf",
        is_config_file=True,
    )

    # Environment
    parser.add_argument(
        "--env-id",
        action="store",
        dest="ENV_ID",
        default="Pong",
        type=str,
        help="Environment to evaluate the agent in. Defaults to Pong.",
    )
    parser.add_argument(
        "--env-render",
        action="store_true",
        dest="ENV_RENDER",
        help="Render the environment if true. Defaults to False.",
    )

    # Agent
    parser.add_argument(
        "--agent",
        action="store",
        dest="AGENT",
        default="dqn2015",
        type=str,
        help="Agent to evaluate. Defaults to dqn2015.",
    )

    # Hyperparameters for DQN
    parser.add_argument(
        "--dqn-type",
        action="store",
        dest="DQN_TYPE",
        default="CNN",
        type=str,
        help="Type of DQN (CNN or FC). Defaults to CNN.",
    )
    parser.add_argument(
        "--nb-episodes",
        action="store",
        dest="NB_EPISODES",
        default=1,
        type=int,
        help="Number of episodes for evaluating DQN. Defaults to 1.",
    )

    # Hyperparameters for Reproducibility
    parser.add_argument(
        "--seed",
        action="store",
        dest="SEED",
        default=None,
        type=int,
        help="Seed to reproduce the results.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        dest="DETERMINISTIC",
        help="Whether to make cuDNN deterministic. This slows down the performance",
    )

    args = parser.parse_args()

    if args.ENV_ID not in ["Acrobot", "CartPole", "MountainCar", "LunarLander", "Pong"]:
        raise ValueError("{} is not a supported environment.".format(args.ENV_ID))
    if args.SEED is None:
        print("[WARNING] Seed not set: this run is not reproducible!")
    else:
        print("[INFO] Seed set to {}".format(args.SEED))

    return args
