"""Uniform experience replay used in DQN2013 and DQN2015."""
import random
from collections import deque
from typing import List

import torch


class DequeUniformReplayBuffer:
    """
    Uniform experience replay used in DQN2013 and DQN2015.

    Implemented using Python's collections.deque.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)  # noqa: E999

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Add a new interaction / experience to the replay buffer.

        Parameters
        ----------
        state : torch.Tensor of torch.float32
            Has shape (1, L) (for feature-type states) or (1, C, H, W) (for
            image-type states).
        action : int
        reward : torch.Tensor of torch.float32
            Has shape (1, 1)
        next_state : torch.Tensor of torch.float32
            Has shape (1, L) (for feature-type states) or (1, C, H, W) (for
            image-type states).
        done : torch.Tensor of torch.float32
            Has shape (1, 1)

        """
        self.buffer.append(
            (state, torch.LongTensor([action]), reward, next_state, done)
        )

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch from the replay buffer.

        This function does not check if the buffer is bigger than the
        `batch_size`.

        Parameters
        ----------
        batch_size : int
            Size of the output batch. Should be at most the current size of the
            buffer.

        Returns
        -------
        batch: tuple of torch.Tensor
            A tuple of batches: (state_batch, action_batch, reward_batch,
            next_state_batch, done_batch).

        """
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.cat(state),
            torch.cat(action),
            torch.cat(reward),
            torch.cat(next_state),
            torch.cat(done),
            None,  # Used only in PrioritizedReplayBuffer
        )

    def update_priorities(
        self, sampled_indices: List[int], priorities: List[float]
    ) -> None:
        """
        Do nothing.

        This is a stub function only implemented in PrioritizedReplayBuffer
        """
        pass


class ListUniformReplayBuffer:
    """
    Uniform experience replay used in DQN2013 and DQN2015.

    Implemented using Python List.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.transitions = []  # type: ignore

    def __len__(self) -> int:
        return len(self.transitions)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Add a new interaction / experience to the replay buffer.

        Parameters
        ----------
        state : torch.Tensor of torch.float32
            Has shape (1, L) (for feature-type states) or (1, C, H, W) (for
            image-type states).
        action : int
        reward : torch.Tensor of torch.float32
            Has shape (1, 1)
        next_state : torch.Tensor of torch.float32
            Has shape (1, L) (for feature-type states) or (1, C, H, W) (for
            image-type states).
        done : torch.Tensor of torch.float32
            Has shape (1, 1)

        """
        self.transitions.append(
            (state, torch.LongTensor([action]), reward, next_state, done)
        )

        # Remove oldest transitions
        while len(self.transitions) > self.capacity:
            del self.transitions[0]

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch from the replay buffer.

        This function does not check if the buffer is bigger than the
        `batch_size`.

        Parameters
        ----------
        batch_size : int
            Size of the output batch. Should be at most the current size of the
            buffer.

        Returns
        -------
        batch: tuple of torch.Tensor
            A tuple of batches: (state_batch, action_batch, reward_batch,
            next_state_batch, done_batch).

        """
        state, action, reward, next_state, done = zip(
            *random.sample(self.transitions, batch_size)
        )
        return (
            torch.cat(state),
            torch.cat(action),
            torch.cat(reward),
            torch.cat(next_state),
            torch.cat(done),
            None,  # Used only in PrioritizedReplayBuffer
        )

    def update_priorities(
        self, sampled_indices: List[int], priorities: List[float]
    ) -> None:
        """
        Do nothing.

        This is a stub function only implemented in PrioritizedReplayBuffer
        """
        pass
