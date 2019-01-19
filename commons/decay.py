"""
decay.py
"""
from typing import Callable


def get_linear_decay(decay_start: float, decay_final: float, decay_duration: int) -> Callable[[int], float]:
    """
    Return a linear decay function. Assumes index start at 1.

    Parameters
    ----------
    decay_start : float
        Starting epsilon value.
    decay_final : float
        Final epsilon value.
    decay_duration : int
        Number of frames to decay epsilon for.

    Returns
    -------
    decay_func
        Function that returns epsilon given an index.
    """
    assert decay_duration > 1
    assert 0 <= decay_start <= 1
    assert 0 <= decay_final <= 1

    def decay_func(idx: int) -> float:
        """
        A linear decay function.
        """
        slope = (decay_final - decay_start) / (decay_duration - 1)
        return slope * (idx - 1) + decay_start if idx <= decay_duration else decay_final

    return decay_func
