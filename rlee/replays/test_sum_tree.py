"""Test SumTree."""
from .sum_tree import SumTree


def test_sum_tree_zero_total_priority_on_init() -> None:
    """Test if SumTree has 0 total priority when initialized."""
    sum_tree = SumTree(4)

    print("Total Priority: ", sum_tree._total_priority())
    assert sum_tree._total_priority() == 0


def test_sum_tree_zero_set() -> None:
    """Test if SumTree.set() correctly sets the value."""
    sum_tree = SumTree(4)

    sum_tree.set(0, 1)
    print("Total Priority: ", sum_tree._total_priority())
    assert sum_tree._total_priority() == 1

    sum_tree.set(1, 1)
    print("Total Priority: ", sum_tree._total_priority())
    assert sum_tree._total_priority() == 2

    sum_tree.set(0, 0)
    print("Total Priority: ", sum_tree._total_priority())
    assert sum_tree._total_priority() == 1


def test_sum_tree_zero_sample_from_one_nonzero_value() -> None:
    """Test sample() when there is only one nonzero value."""
    sum_tree = SumTree(4)

    sum_tree.set(0, 1)
    sampled_idx = sum_tree.sample()

    print("Sampled node index: ", sampled_idx)
    assert sampled_idx == 0


def test_sum_tree_zero_sample_from_two_nonzero_values() -> None:
    """Test sample() when there are two nonzero values."""
    sum_tree = SumTree(4)

    sum_tree.set(2, 1)
    sum_tree.set(3, 1)
    sampled_idx = sum_tree.sample()

    print("Sampled node index: ", sampled_idx)
    assert sampled_idx in [2, 3]
