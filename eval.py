#!/usr/bin/env python3
"""Evaluate saved DQN agent."""
import random

import numpy as np
import torch

from rlee.agents import DQN2015Agent
from rlee.commons import get_eval_args
from rlee.networks import DQN, FCDQN
from rlee.wrappers import make_env


def main() -> None:
    """Train DQN agent."""
    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse arguments
    ARGS = get_eval_args()

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

    if ARGS.DQN_TYPE == "CNN":
        dqn = DQN(
            num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n
        ).to(device)
    elif ARGS.DQN_TYPE == "FC":
        dqn = FCDQN(
            num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n
        ).to(device)

    agent = DQN2015Agent(  # type: ignore
        env, dqn, None, None, None, None, None, None, None, None, None, None, None
    )

    # Load agent
    agent.load_model(LOAD_PATH="saved_models/")

    # Evaluate agent
    # This is intentionally not modularized inside the Agent class
    # (as in, agent.eval) because we often want to add additional
    # code here to analyze the agent's behaviors.
    episode_rewards = []
    for episode_idx in range(1, ARGS.NB_EPISODES + 1):
        obs = env.reset()
        if ARGS.ENV_RENDER:
            env.render()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(obs, 0)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward.item()
            if ARGS.ENV_RENDER:
                env.render()

        episode_rewards.append(episode_reward)
        print(
            "Episode {:2d}/{:2d} Episode Reward: {}".format(
                episode_idx, ARGS.NB_EPISODES, episode_reward
            )
        )

    print("Average Episode Reward: {}".format(sum(episode_rewards) / ARGS.NB_EPISODES))
    env.close()


if __name__ == "__main__":
    main()
