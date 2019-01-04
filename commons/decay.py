"""
decay.py
"""


def get_linear_decay(decay_start, decay_final, decay_duration):
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

    def decay_func(idx):
        """
        A linear decay function.
        """
        slope = (decay_final - decay_start) / (decay_duration - 1)
        return slope * (idx - 1) + decay_start if idx <= decay_duration else decay_final

    return decay_func
