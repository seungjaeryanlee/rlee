#!/usr/bin/env python3
"""
train_atari.py
"""
import torch.optim as optim

from agents import NaiveDQNAgent
from commons import get_train_args
from networks import DQN
from wrappers import make_env


def main():
    # Parse arguments
    ARGS = get_train_args()

    # Setup Environment
    env = make_env(ARGS.ENV_ID)

    # Setup NaiveDQNAgent
    dqn = DQN(num_inputs=env.observation_space.shape[0],
              num_actions=env.action_space.n)
    optimizer = optim.Adam(dqn.parameters(), lr=ARGS.LR)
    agent = NaiveDQNAgent(env, dqn, optimizer,
                          ARGS.DISCOUNT,
                          ARGS.EPSILON)

    # Train agent
    agent.train(100)


if __name__ == '__main__':
    main()
