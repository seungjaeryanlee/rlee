#!/usr/bin/env python3
"""
train_atari.py
"""
import gym


def main():
    # TODO Parse arguments
    ARGS = get_args()

    # TODO Setup Environment
    env = gym.make(ARGS.ENV_NAME)

    # TODO Implement DQN network
    dqn = DQN()
    # TODO Implement NaiveDQNAgent
    agent = NaiveDQNAgent()

    # TODO Implement NaiveDQNAgent.train()
    agent.train()


if __name__ == '__main__':
    main()
