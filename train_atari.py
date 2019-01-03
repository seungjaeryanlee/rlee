#!/usr/bin/env python3
"""
train_atari.py
"""
from commons import get_train_args
from networks import DQN
from wrappers import make_env


def main():
    # Parse arguments
    ARGS = get_train_args()

    # TODO Setup Environment
    env = make_env(ARGS.ENV_ID)

    # Implement DQN network
    dqn = DQN(num_inputs=env.observation_space.shape[0],
              num_actions=env.action_space.n)

    # TODO Implement NaiveDQNAgent
    agent = NaiveDQNAgent()

    # TODO Implement NaiveDQNAgent.train()
    agent.train()


if __name__ == '__main__':
    main()
