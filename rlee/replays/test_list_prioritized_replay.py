"""Test ListPrioritizedReplayBuffer."""

import torch

from .prioritized_replay import ListPrioritizedReplayBuffer


def test_list_prioritized_batch_type() -> None:
    """Test if `replay_buffer.sample()` returns five `torch.Tensor` objects."""
    BATCH_SIZE = 1
    replay_buffer = ListPrioritizedReplayBuffer(BATCH_SIZE)

    for _ in range(BATCH_SIZE):
        state = torch.FloatTensor([[0, 0, 0, 0]])
        action = 0
        reward = torch.FloatTensor([1])
        next_state = torch.FloatTensor([[0, 0, 0, 0]])
        done = torch.FloatTensor([0])
        replay_buffer.push(state, action, reward, next_state, done)

    state_b, action_b, reward_b, next_state_b, done_b, sampled_indices = replay_buffer.sample(  # noqa: B950
        1
    )

    print("S  : ", state_b)
    print("A  : ", action_b)
    print("R  : ", reward_b)
    print("S' : ", next_state_b)
    print("D  : ", done_b)
    assert type(state_b) == torch.Tensor
    assert type(action_b) == torch.Tensor
    assert type(reward_b) == torch.Tensor
    assert type(next_state_b) == torch.Tensor
    assert type(done_b) == torch.Tensor


def test_list_prioritized_batch_shape() -> None:
    """Test if `replay_buffer.sample()` returns have correct shapes."""
    BATCH_SIZE = 2
    STATE_LEN = 4
    replay_buffer = ListPrioritizedReplayBuffer(BATCH_SIZE)

    for _ in range(BATCH_SIZE):
        state = torch.FloatTensor([[0, 0, 0, 0]])
        action = 0
        reward = torch.FloatTensor([1])
        next_state = torch.FloatTensor([[0, 0, 0, 0]])
        done = torch.FloatTensor([0])
        replay_buffer.push(state, action, reward, next_state, done)

    state_b, action_b, reward_b, next_state_b, done_b, sampled_indices = replay_buffer.sample(  # noqa: B950
        BATCH_SIZE
    )

    print("S  : ", state_b)
    print("A  : ", action_b)
    print("R  : ", reward_b)
    print("S' : ", next_state_b)
    print("D  : ", done_b)
    assert state_b.shape == torch.Size([BATCH_SIZE, STATE_LEN])
    assert action_b.shape == torch.Size([BATCH_SIZE])
    assert reward_b.shape == torch.Size([BATCH_SIZE])
    assert next_state_b.shape == torch.Size([BATCH_SIZE, STATE_LEN])
    assert done_b.shape == torch.Size([BATCH_SIZE])


def test_list_prioritized_batch_default_priorities() -> None:
    """Test if transitions have correct default priorities."""
    BATCH_SIZE = 2
    STATE_LEN = 4
    replay_buffer = ListPrioritizedReplayBuffer(BATCH_SIZE)

    for _ in range(BATCH_SIZE):
        state = torch.FloatTensor([[0] * STATE_LEN])
        action = 0
        reward = torch.FloatTensor([1])
        next_state = torch.FloatTensor([[0] * STATE_LEN])
        done = torch.FloatTensor([0])
        replay_buffer.push(state, action, reward, next_state, done)

    for node_index in range(BATCH_SIZE):
        print(
            "Priority of index {}: {}".format(
                node_index, replay_buffer.sum_tree.get(node_index)
            )
        )
        assert replay_buffer.sum_tree.get(node_index) == 1


def test_list_prioritized_batch_update_priorities() -> None:
    """Test update_priorities()."""
    BATCH_SIZE = 2
    STATE_LEN = 4
    replay_buffer = ListPrioritizedReplayBuffer(BATCH_SIZE)

    for _ in range(BATCH_SIZE):
        state = torch.FloatTensor([[0] * STATE_LEN])
        action = 0
        reward = torch.FloatTensor([1])
        next_state = torch.FloatTensor([[0] * STATE_LEN])
        done = torch.FloatTensor([0])
        replay_buffer.push(state, action, reward, next_state, done)

    replay_buffer.update_priorities([0], [0])
    assert replay_buffer.sum_tree.get(0) == 0
