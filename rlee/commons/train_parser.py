"""Parser for training an agent."""
from typing import Dict, Optional, Tuple

import configargparse


def get_train_args(
    description: str = "endtoendai/rlee", default_args: Optional[dict] = None
) -> Tuple[configargparse.Namespace, Dict[str, configargparse.Namespace]]:
    """
    Parse arguments for training agents and return hyperparameters as a Namespace.

    Parameters
    ----------
    description: str
        Description for the argument parser. Defaults to endtoendai/rlee.

    Returns
    -------
    ARGS
        Namespace containing hyperparameter options specified by user or set
        by default.
    GROUPED_ARGS
        Dictionary of Namespaces containing hyperparameter options of
        relevant usage.

    """
    parser = configargparse.ArgumentParser(description)
    parser.add(
        "--config",
        help="config file path",
        default="configs/pong.train.conf",
        is_config_file=True,
    )

    # Environment
    parser.add_argument(
        "--env-id",
        action="store",
        dest="ENV_ID",
        default="Pong",
        type=str,
        help="Environment to train the agent in. Defaults to Pong.",
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
        help="Environment to train the agent in. Defaults to dqn2015.",
    )

    # Hyperparameters for DQN
    parser.add_argument(
        "--nb-steps",
        action="store",
        dest="NB_STEPS",
        default=10000000,
        type=int,
        help="Number of steps for training DQN.",
    )
    parser.add_argument(
        "--discount",
        action="store",
        dest="DISCOUNT",
        default=0.99,
        type=float,
        help="Discount factor for DQN.",
    )
    parser.add_argument(
        "--target-update-freq",
        action="store",
        dest="TARGET_UPDATE_FREQ",
        default=10000,
        type=int,
        help="Target network update frequency for DQN.",
    )
    parser.add_argument(
        "--no-huber-loss",
        action="store_true",
        dest="NO_HUBER_LOSS",
        help="Disable Huber loss (Smooth L1 loss).",
    )

    # Hyperparameters for RMSprop
    # https://twitter.com/FlorinGogianu/status/1080139414695759872
    # TODO Check weight decay hyperparameter again: is this the missing
    # 0.95 hyperparameter?
    parser.add_argument(
        "--rmsprop-lr",
        action="store",
        dest="RMSPROP_LR",
        default=2.5e-4,
        type=float,
        help="RMSprop learning rate. Defaults to 2.5e-4",
    )
    parser.add_argument(
        "--rmsprop-alpha",
        action="store",
        dest="RMSPROP_ALPHA",
        default=0.95,
        type=float,
        help="RMSprop smoothing constant. Defaults to 0.95",
    )
    parser.add_argument(
        "--rmsprop-eps",
        action="store",
        dest="RMSPROP_EPS",
        default=0.01,
        type=float,
        help=(
            "RMSprop constant term in denominator to improve numerical"
            " stability. Defaults to 0.01"
        ),
    )
    parser.add_argument(
        "--rmsprop-weight-decay",
        action="store",
        dest="RMSPROP_WEIGHT_DECAY",
        default=0,
        type=float,
        help="RMSprop weight decay . Defaults to 0.",
    )
    parser.add_argument(
        "--rmsprop-momentum",
        action="store",
        dest="RMSPROP_MOMENTUM",
        default=0,
        type=float,
        help="RMSprop momentum . Defaults to 0.",
    )
    parser.add_argument(
        "--rmsprop-not-centered",
        action="store_true",
        dest="RMSPROP_NOT_CENTERED",
        help="RMSprop is not centered. Defaults to False.",
    )

    # Hyperparameters for Adam
    parser.add_argument(
        "--use-adam",
        dest="USE_ADAM",
        help="Use Adam optimizer instead of RMSprop.",
        action="store_true",
    )
    parser.add_argument(
        "--adam-lr",
        dest="ADAM_LR",
        help="Adam learning rate.",
        default=1e-4,
        type=float,
    )

    # Hyperparameters for Replay Buffer
    parser.add_argument(
        "--replay-buffer-size",
        action="store",
        dest="REPLAY_BUFFER_SIZE",
        default=1000000,
        type=int,
        help="Size of the experience replay buffer. Defaults to 1000000",
    )
    parser.add_argument(
        "--batch-size",
        action="store",
        dest="BATCH_SIZE",
        default=32,
        type=int,
        help="Batch size for sampling from experience replay buffer. Defaults to 32.",
    )
    parser.add_argument(
        "--min-replay-buffer-size",
        action="store",
        dest="MIN_REPLAY_BUFFER_SIZE",
        default=10000,
        type=int,
        help="Minimum replay buffer size before sampling. Defaults to 32.",
    )

    # Hyperparameters for epsilon decay
    parser.add_argument(
        "--epsilon-decay-start",
        action="store",
        dest="EPSILON_DECAY_START",
        default=1,
        type=float,
        help="Starting epsilon value. Defaults to 1.",
    )
    parser.add_argument(
        "--epsilon-decay-final",
        action="store",
        dest="EPSILON_DECAY_FINAL",
        default=0.1,
        type=float,
        help="Minimum epsilon value. Defaults to 0.1.",
    )
    parser.add_argument(
        "--epsilon-decay-duration",
        action="store",
        dest="EPSILON_DECAY_DURATION",
        default=1000000,
        type=int,
        help="Number of frames to decay epsilon for. Defaults to 1000000.",
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

    # Hyperparameters for Logging
    parser.add_argument(
        "--wandb-entity",
        action="store",
        dest="WANDB_ENTITY",
        default="endtoendai",
        type=str,
        help="Entity of wandb project. Defaults to endtoendai.",
    )
    parser.add_argument(
        "--wandb-project",
        action="store",
        dest="WANDB_PROJECT",
        default="rlee",
        type=str,
        help="Wandb project name. Defaults to rlee.",
    )
    parser.add_argument(
        "--wandb-dir",
        action="store",
        dest="WANDB_DIR",
        default="wandb_log/",
        type=str,
        help="Directory for wandb logs. Defaults to wandb_log/.",
    )
    parser.add_argument(
        "--wandb-interval",
        action="store",
        dest="WANDB_INTERVAL",
        default=100,
        type=int,
        help="How frequently logs should be sent to wandb. Defaults to 100.",
    )

    ARGS = parser.parse_args()

    if ARGS.ENV_ID not in ["Acrobot", "CartPole", "MountainCar", "LunarLander", "Pong"]:
        raise ValueError("{} is not a supported environment.".format(ARGS.ENV_ID))
    if ARGS.SEED is None:
        print("[WARNING] Seed not set: this run is not reproducible!")
    else:
        print("[INFO] Seed set to {}".format(ARGS.SEED))

    ARGS.USE_HUBER_LOSS = not ARGS.NO_HUBER_LOSS
    ARGS.RMSPROP_CENTERED = not ARGS.RMSPROP_NOT_CENTERED

    # Group arguments
    ENV_ARGS = configargparse.Namespace(ENV_ID=ARGS.ENV_ID, ENV_RENDER=ARGS.ENV_RENDER)
    AGENT_ARGS = configargparse.Namespace(AGENT=ARGS.AGENT)
    DQN_ARGS = configargparse.Namespace(
        NB_STEPS=ARGS.NB_STEPS,
        DISCOUNT=ARGS.DISCOUNT,
        TARGET_UPDATE_FREQ=ARGS.TARGET_UPDATE_FREQ,
        NO_HUBER_LOSS=ARGS.NO_HUBER_LOSS,
    )
    OPTIMIZER_ARGS = configargparse.Namespace(
        RMSPROP_LR=ARGS.RMSPROP_LR,
        RMSPROP_ALPHA=ARGS.RMSPROP_ALPHA,
        RMSPROP_EPS=ARGS.RMSPROP_EPS,
        RMSPROP_WEIGHT_DECAY=ARGS.RMSPROP_WEIGHT_DECAY,
        RMSPROP_MOMENTUM=ARGS.RMSPROP_MOMENTUM,
        RMSPROP_NOT_CENTERED=ARGS.RMSPROP_NOT_CENTERED,
        USE_ADAM=ARGS.USE_ADAM,
        ADAM_LR=ARGS.ADAM_LR,
    )
    REPLAY_BUFFER_ARGS = configargparse.Namespace(
        REPLAY_BUFFER_SIZE=ARGS.REPLAY_BUFFER_SIZE,
        BATCH_SIZE=ARGS.BATCH_SIZE,
        MIN_REPLAY_BUFFER_SIZE=ARGS.MIN_REPLAY_BUFFER_SIZE,
    )
    EPSILON_DECAY_ARGS = configargparse.Namespace(
        EPSILON_DECAY_START=ARGS.EPSILON_DECAY_START,
        EPSILON_DECAY_FINAL=ARGS.EPSILON_DECAY_FINAL,
        EPSILON_DECAY_DURATION=ARGS.EPSILON_DECAY_DURATION,
    )
    REPRODUCIBILITY_ARGS = configargparse.Namespace(
        SEED=ARGS.SEED, DETERMINISTIC=ARGS.DETERMINISTIC
    )
    LOGGING_ARGS = configargparse.Namespace(
        WANDB_ENTITY=ARGS.WANDB_ENTITY,
        WANDB_PROJECT=ARGS.WANDB_PROJECT,
        WANDB_DIR=ARGS.WANDB_DIR,
        WANDB_INTERVAL=ARGS.WANDB_INTERVAL,
    )
    GROUPED_ARGS = {
        "ENV_ARGS": ENV_ARGS,
        "AGENT_ARGS": AGENT_ARGS,
        "DQN_ARGS": DQN_ARGS,
        "OPTIMIZER_ARGS": OPTIMIZER_ARGS,
        "REPLAY_BUFFER_ARGS": REPLAY_BUFFER_ARGS,
        "EPSILON_DECAY_ARGS": EPSILON_DECAY_ARGS,
        "REPRODUCIBILITY_ARGS": REPRODUCIBILITY_ARGS,
        "LOGGING_ARGS": LOGGING_ARGS,
    }

    return ARGS, GROUPED_ARGS
