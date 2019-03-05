"""
Agent equivalent to DQN 2015 paper.

Human-level control through deep reinforcement learning
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
"""
import copy
import pathlib
import random
import time
from typing import Any, Callable, Tuple

import torch
import wandb


class DQN2015Agent:
    """
    Agent equivalent to DQN 2015 paper.

    A Deep Q-Network (DQN) agent that can be trained with environments that
    have feature vectors as states and discrete values as actions. Uses
    experience replay and target network.

    Human-level control through deep reinforcement learning
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """

    def __init__(
        self,
        env: Any,
        dqn: Any,
        optimizer: Any,
        criterion: Any,
        replay_buffer: Any,
        epsilon_func: Callable[[int], float],
        device: bool,
        ENV_RENDER: bool,
        DISCOUNT: float,
        BATCH_SIZE: int,
        MIN_REPLAY_BUFFER_SIZE: int,
        TARGET_UPDATE_FREQ: int,
        WANDB_INTERVAL: int,
        SAVE_PREFIX: str,
    ):
        self.env = env
        self.current_dqn = dqn
        self.target_dqn = copy.deepcopy(dqn)
        self.optimizer = optimizer
        self.criterion = criterion
        self.replay_buffer = replay_buffer
        self.epsilon_func = epsilon_func
        self.device = device

        self.ENV_RENDER = ENV_RENDER
        self.DISCOUNT = DISCOUNT
        self.BATCH_SIZE = BATCH_SIZE
        self.MIN_REPLAY_BUFFER_SIZE = MIN_REPLAY_BUFFER_SIZE
        self.TARGET_UPDATE_FREQ = TARGET_UPDATE_FREQ
        self.WANDB_INTERVAL = WANDB_INTERVAL
        self.SAVE_PREFIX = SAVE_PREFIX

    def act(self, state: torch.Tensor, epsilon: float) -> int:
        """
        Return an action sampled from an epsilon-greedy policy.

        Parameters
        ----------
        state
            The state to compute the epsilon-greedy action of.
        epsilon : float
            Epsilon in epsilon-greedy policy: probability of choosing a random
            action.

        Returns
        -------
        action : int
            An integer representing a discrete action chosen by the agent.

        """
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.current_dqn(state.to(self.device)).cpu()
            action = q_values.max(1)[1].item()
        else:
            action = self.env.action_space.sample()

        return action

    def train(self, nb_frames: int) -> None:
        """
        Train the agent by interacting with the environment.

        Parameters
        ----------
        nb_frames: int
            Number of frames to train the agent.

        """
        episode_reward = 0
        max_episode_reward = 0
        episode_length = 0
        loss = torch.FloatTensor([0])
        state = self.env.reset()

        if self.ENV_RENDER:
            self.env.render()

        for frame_idx in range(1, nb_frames + 1):
            # Start timer
            t_start = time.time()

            # Interact and save to replay buffer
            epsilon = self.epsilon_func(frame_idx)
            action = self.act(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward.item()
            episode_length += 1

            if done:
                state = self.env.reset()
                init_state_value_estimate = (
                    self.current_dqn(state.to(self.device)).max(1)[0].cpu().item()
                )

                # Save model if the episode is improved
                if episode_reward > max_episode_reward:
                    max_episode_reward = episode_reward
                    self.save()

                print(
                    "Frame {:5d}/{:5d}\tReturn {:3.2f}\tLoss {:2.4f}".format(
                        frame_idx + 1, nb_frames, episode_reward, loss.item()
                    )
                )
                wandb.log(
                    {
                        "Episode Reward": episode_reward,
                        "Episode Length": episode_length,
                        "Value Estimate of Initial State": init_state_value_estimate,
                    },
                    step=frame_idx,
                )

                episode_reward = 0
                episode_length = 0

            # Train DQN if the replay buffer is populated enough
            if len(self.replay_buffer) > self.MIN_REPLAY_BUFFER_SIZE:
                self.optimizer.zero_grad()
                replay_batch = self.replay_buffer.sample(self.BATCH_SIZE)
                loss = self._compute_loss(replay_batch)
                loss.backward()
                self.optimizer.step()

                if frame_idx % self.WANDB_INTERVAL == 0:
                    wandb.log({"Loss": loss}, step=frame_idx)

            # Update Target DQN periodically
            if frame_idx % self.TARGET_UPDATE_FREQ == 0:
                self._update_target()

            # Render environment
            if self.ENV_RENDER:
                self.env.render()

            # End timer
            t_end = time.time()
            t_delta = t_end - t_start
            fps = 1 / (t_end - t_start)

            # Log to wandb
            if frame_idx % self.WANDB_INTERVAL == 0:
                wandb.log(
                    {
                        "Epsilon": epsilon,
                        "Reward": reward,
                        "Time per frame": t_delta,
                        "FPS": fps,
                    },
                    step=frame_idx,
                )

    def _compute_loss(self, batch: Tuple) -> torch.Tensor:
        """
        Compute batch MSE loss between 1-step target Q and prediction Q.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A tuple of batches: (state_batch, action_batch, reward_batch,
            next_state_batch, done_batch).
        DISCOUNT: float
            Discount factor for computing Q-value.

        Returns
        -------
        loss : torch.FloatTensor
            MSE loss of target Q and prediction Q that can be backpropagated.
            Has shape torch.Size([1]).

        """
        state_b, action_b, reward_b, next_state_b, done_b = self.replay_buffer.sample(
            self.BATCH_SIZE
        )
        state_batch = state_b.to(self.device)
        action_batch = action_b.to(self.device)
        reward_batch = reward_b.to(self.device)
        next_state_batch = next_state_b.to(self.device)
        done_batch = done_b.to(self.device)

        # Predicted Q: Q_current(s, a)
        # q_values : torch.Size([BATCH_SIZE, self.env.action_space.n])
        # action   : torch.Size([BATCH_SIZE])
        # q_value  : torch.Size([BATCH_SIZE])
        q_values = self.current_dqn(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze().cpu()

        # Target Q: r + gamma * max_{a'} Q_target(s', a')
        # next_q_values    : torch.Size([BATCH_SIZE, self.env.action_space.n])
        # next_q_value     : torch.Size([BATCH_SIZE])
        # expected_q_value : torch.Size([BATCH_SIZE])
        with torch.no_grad():
            # Q_target(s', a')
            next_q_values = self.target_dqn(next_state_batch)
            next_q_value = next_q_values.max(dim=1)[0].squeeze()
            expected_q_value = (
                reward_batch + self.DISCOUNT * next_q_value * (1 - done_batch)
            ).cpu()

        assert expected_q_value.shape == q_value.shape

        # Compute loss
        loss = self.criterion(q_value, expected_q_value.detach())

        return loss

    def _update_target(self) -> None:
        """Update weights of Target DQN with weights of current DQN."""
        self.target_dqn.load_state_dict(self.current_dqn.state_dict())

    def save(self) -> None:
        """Save DQN and optimizer."""
        # Make directory if it doesn't exist yet
        pathlib.Path("saved_models/").mkdir(parents=True, exist_ok=True)

        DQN_SAVE_PATH = "{}dqn.pt".format(self.SAVE_PREFIX)
        OPTIM_SAVE_PATH = "{}optim.pt".format(self.SAVE_PREFIX)
        torch.save(self.current_dqn.state_dict(), DQN_SAVE_PATH)
        torch.save(self.optimizer.state_dict(), OPTIM_SAVE_PATH)

    def load_for_training(self, LOAD_PREFIX: str) -> None:
        """Load DQN and optimizer."""
        DQN_SAVE_PATH = "{}dqn.pt".format(LOAD_PREFIX)
        OPTIM_SAVE_PATH = "{}optim.pt".format(LOAD_PREFIX)
        self.current_dqn.load_state_dict(torch.load(DQN_SAVE_PATH))
        self._update_target()
        self.optimizer.load_state_dict(torch.load(OPTIM_SAVE_PATH))

    def load_model(self, LOAD_PREFIX: str) -> None:
        """Load DQN."""
        DQN_SAVE_PATH = "{}dqn.pt".format(LOAD_PREFIX)
        self.current_dqn.load_state_dict(torch.load(DQN_SAVE_PATH))
        self._update_target()
