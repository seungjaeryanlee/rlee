"""
Agent equivalent to Double DQN paper.

Deep Reinforcement Learning with Double Q-learning
https://arxiv.org/abs/1509.06461
"""
from typing import Any, Callable, Tuple

import torch

from .DQN2015Agent import DQN2015Agent


class DoubleDQNAgent(DQN2015Agent):
    """
    Agent equivalent to Double DQN paper.

    Deep Reinforcement Learning with Double Q-learning
    https://arxiv.org/abs/1509.06461
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
        WANDB: bool,
        WANDB_INTERVAL: int,
        SAVE_PREFIX: str,
    ):
        super().__init__(
            env,
            dqn,
            optimizer,
            criterion,
            replay_buffer,
            epsilon_func,
            device,
            ENV_RENDER,
            DISCOUNT,
            BATCH_SIZE,
            MIN_REPLAY_BUFFER_SIZE,
            TARGET_UPDATE_FREQ,
            WANDB,
            WANDB_INTERVAL,
            SAVE_PREFIX,
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
            Loss of target Q and prediction Q that can be backpropagated.
            Has shape torch.Size([1]).
        losses : torch.FloatTensor
            Losses of target Q and prediction Q for each sample. Used in
            prioritized experience replay. Has shape
            torch.Size([BATCH_SIZE]).

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

        # (DQN2015)    Target Q: r + gamma * max_{a'} Q_target(s', a')
        # (Double DQN) Target Q: r + gamma * Q_target(s', argmax_a Q(s', a'))
        # next_q_values    : torch.Size([BATCH_SIZE, self.env.action_space.n])
        # best_action      : torch.Size([BATCH_SIZE, 1])
        # next_q_value     : torch.Size([BATCH_SIZE])
        # expected_q_value : torch.Size([BATCH_SIZE])
        with torch.no_grad():
            # Q(s', a')
            next_q_values = self.current_dqn(next_state_batch)
            # argmax_a Q(s', a')
            best_action = next_q_values.max(dim=1)[1].unsqueeze(1)
            # Q_target(s', argmax_a Q(s', a'))
            next_q_value = (
                self.target_dqn(next_state_batch).gather(1, best_action).squeeze(1)
            )

            expected_q_value = (
                reward_batch + self.DISCOUNT * next_q_value * (1 - done_batch)
            ).cpu()

        assert expected_q_value.shape == q_value.shape

        # Compute losses
        losses = self.criterion(q_value, expected_q_value.detach())
        loss = losses.mean().unsqueeze(0)

        # Return loss for backward, losses for updating priorities for
        # Prioritized Experience Replay
        return loss, losses
