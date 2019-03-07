"""Parser for evaluating an agent."""
from typing import Optional

import configargparse


def get_eval_args(
    description: str = "endtoendai/rlee", default_args: Optional[dict] = None
) -> configargparse.Namespace:
    """
    Parse arguments for evaluating agents and return hyperparameter Namespace.

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

    # Hyperparameters for Replay Buffer
    parser.add_argument(
        "--replay-buffer-type",
        action="store",
        dest="REPLAY_BUFFER_TYPE",
        default="uniform",
        type=str,
        help="Type of experience replay buffer. Defaults to uniform.",
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

    parser.add_argument(
        "--load-dir",
        action="store",
        dest="LOAD_DIR",
        default="saved_models/",
        type=str,
        help="Directory of saved files.",
    )

    args = parser.parse_args()

    if args.ENV_ID not in ["Acrobot", "CartPole", "MountainCar", "LunarLander", "Pong"]:
        raise ValueError("{} is not a supported environment.".format(args.ENV_ID))

    if args.AGENT not in ["dqn2015", "doubledqn"]:
        raise ValueError("{} is not a supported agent.".format(args.AGENT))

    if args.REPLAY_BUFFER_TYPE not in ["uniform", "combined", "prioritized"]:
        raise ValueError(
            "{} is not a supported replay buffer type.".format(args.REPLAY_BUFFER_TYPE)
        )

    if args.SEED is None:
        print("[WARNING] Seed not set: this run is not reproducible!")
    else:
        print("[INFO] Seed set to {}".format(args.SEED))

    args.LOAD_PREFIX = "{}/{}_{}_{}_best_".format(
        args.LOAD_DIR, args.ENV_ID, args.AGENT, args.REPLAY_BUFFER_TYPE
    )

    return args
