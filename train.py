#!/usr/bin/env python3
"""Train DQN agent."""
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from rlee.commons import get_linear_decay, get_train_args
from rlee.networks import DQN, FCDQN
from rlee.wrappers import make_env


def main() -> None:
    """Train DQN agent."""
    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse arguments
    ARGS = get_train_args()

    # Setup wandb
    if ARGS.WANDB:
        wandb.init(
            entity=ARGS.WANDB_ENTITY,
            project=ARGS.WANDB_PROJECT,
            dir=ARGS.WANDB_DIR,
            config=ARGS,
        )

    # Setup Environment
    env = make_env(ARGS.ENV_ID)

    # For reproducibility
    if ARGS.SEED is not None:
        env.seed(ARGS.SEED)
        random.seed(ARGS.SEED)
        np.random.seed(ARGS.SEED)
        torch.manual_seed(ARGS.SEED)
    if ARGS.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup Loss Criterion
    if ARGS.USE_HUBER_LOSS:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    if ARGS.DQN_TYPE == "CNN":
        dqn = DQN(
            num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n
        ).to(device)
    elif ARGS.DQN_TYPE == "FC":
        dqn = FCDQN(
            num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n
        ).to(device)

    # Watch DQN on model
    if ARGS.WANDB:
        wandb.watch(dqn)

    optimizer: Any = None  # noqa: E999
    if ARGS.USE_ADAM:
        optimizer = optim.Adam(dqn.parameters(), lr=ARGS.ADAM_LR)
    else:
        optimizer = optim.RMSprop(
            dqn.parameters(),
            lr=ARGS.RMSPROP_LR,
            alpha=ARGS.RMSPROP_ALPHA,
            eps=ARGS.RMSPROP_EPS,
            weight_decay=ARGS.RMSPROP_WEIGHT_DECAY,
            momentum=ARGS.RMSPROP_MOMENTUM,
            centered=ARGS.RMSPROP_CENTERED,
        )

    epsilon_func = get_linear_decay(
        ARGS.EPSILON_DECAY_START, ARGS.EPSILON_DECAY_FINAL, ARGS.EPSILON_DECAY_DURATION
    )

    # Initialize replay buffer
    if ARGS.REPLAY_BUFFER_TYPE == "uniform":
        from rlee.replays import UniformReplayBuffer

        replay_buffer = UniformReplayBuffer(ARGS.REPLAY_BUFFER_SIZE)  # type: ignore
        print("[INFO] Using UNIFORM replay buffer.")
    elif ARGS.REPLAY_BUFFER_TYPE == "combined":
        from rlee.replays import CombinedReplayBuffer

        replay_buffer = CombinedReplayBuffer(ARGS.REPLAY_BUFFER_SIZE)  # type: ignore
        print("[INFO] Using COMBINED replay buffer.")

    # Initialize agent
    if ARGS.AGENT == "dqn2015":
        from rlee.agents import DQN2015Agent

        Agent = DQN2015Agent  # type: ignore
    elif ARGS.AGENT == "doubledqn":
        from rlee.agents import DoubleDQNAgent

        Agent = DoubleDQNAgent  # type: ignore

    agent = Agent(
        env,
        dqn,
        optimizer,
        criterion,
        replay_buffer,
        epsilon_func,
        device,
        ARGS.ENV_RENDER,
        ARGS.DISCOUNT,
        ARGS.BATCH_SIZE,
        ARGS.MIN_REPLAY_BUFFER_SIZE,
        ARGS.TARGET_UPDATE_FREQ,
        ARGS.WANDB,
        ARGS.WANDB_INTERVAL,
        ARGS.SAVE_PREFIX,
    )

    # Train agent
    agent.train(ARGS.NB_STEPS)


if __name__ == "__main__":
    main()
