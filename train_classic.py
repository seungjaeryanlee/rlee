#!/usr/bin/env python3
"""Train DQN agent on Classic Control environment."""
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from rlee.agents import DQN2015Agent
from rlee.commons import get_linear_decay, get_train_args
from rlee.networks import FCDQN
from rlee.replays import UniformReplayBuffer
from rlee.wrappers import make_env


def main() -> None:
    """Train DQN agent on Classic Control environment."""
    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse arguments
    ARGS = get_train_args()

    # Setup wandb
    wandb.init(project="rlee", dir=".wandb_log", config=ARGS)

    # Setup Environment
    env = make_env(ARGS.ENV_ID)

    # Setup Loss Criterion
    if ARGS.USE_HUBER_LOSS:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    dqn = FCDQN(
        num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n
    ).to(device)

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
    replay_buffer = UniformReplayBuffer(ARGS.REPLAY_BUFFER_SIZE)
    agent = DQN2015Agent(
        env,
        dqn,
        optimizer,
        criterion,
        replay_buffer,
        epsilon_func,
        device,
        ARGS.DISCOUNT,
        ARGS.BATCH_SIZE,
        ARGS.MIN_REPLAY_BUFFER_SIZE,
        ARGS.TARGET_UPDATE_FREQ,
        ARGS.WANDB_INTERVAL,
    )

    # Train agent
    agent.train(ARGS.NB_STEPS)


if __name__ == "__main__":
    main()
