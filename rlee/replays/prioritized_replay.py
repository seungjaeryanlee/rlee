"""
Prioritized experience replay in PER.

Prioritized Experience Replay
https://arxiv.org/abs/1511.05952
"""
import random
from typing import List

import torch

from .sum_tree import SumTree


class ListPrioritizedReplayBuffer:
    """
    Prioritized experience replay in PER.

    Prioritized Experience Replay
    https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity: int, DEFAULT_PRIORITY: float = 1) -> None:
        self.capacity = capacity
        self.transitions = []  # type: ignore
        self.sum_tree = SumTree(capacity)
        self.cursor = 0
        self.DEFAULT_PRIORITY = DEFAULT_PRIORITY

    def __len__(self) -> int:
        return len(self.transitions)

    def _increment_cursor(self) -> None:
        """Increment cursor position."""
        self.cursor = (self.cursor + 1) % self.capacity

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
        self.sum_tree.set(self.cursor, self.DEFAULT_PRIORITY)
        self._increment_cursor()

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
        sampled_indices = self.sum_tree.stratified_sample(batch_size)
        # Some of these indices might be invalid. We select uniformly
        # random to replace these invalid indices.
        samples = []
        for sampled_index in sampled_indices:
            if len(self.transitions) - 1 < sampled_index:  # Invalid index
                sample = random.sample(self.transitions, 1)[0]
                print(
                    "[INFO] Stratified sampling led to invalid index"
                    " {} when current length was {}".format(
                        sampled_index, len(self.transitions)
                    )
                )
                exit()
            else:
                sample = self.transitions[sampled_index]
            samples.append(sample)
        state, action, reward, next_state, done = zip(*samples)

        return (
            torch.cat(state),
            torch.cat(action),
            torch.cat(reward),
            torch.cat(next_state),
            torch.cat(done),
            sampled_indices,  # None for other replay types
        )

    def update_priorities(
        self, sampled_indices: List[int], priorities: List[float]
    ) -> None:
        """Update priorities."""
        for sampled_index, priority in zip(sampled_indices, priorities):
            self.sum_tree.set(sampled_index, priority)
