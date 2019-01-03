#!/usr/bin/env python3
"""
train_atari.py
"""
from commons import get_train_args


def main():
    # Parse arguments
    ARGS = get_train_args()

    # TODO Setup Environment
    env = make_env(ARGS.ENV_ID)

    # TODO Implement DQN network
    dqn = DQN()
    # TODO Implement NaiveDQNAgent
    agent = NaiveDQNAgent()

    # TODO Implement NaiveDQNAgent.train()
    agent.train()


if __name__ == '__main__':
    main()
