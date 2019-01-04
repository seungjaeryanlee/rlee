"""
test_decay.py
"""
import pytest

from .decay import get_linear_decay


def test_get_linear_decay_start():
    decay_start, decay_final, decay_duration = (1, 0.1, 1000000)
    decay_func = get_linear_decay(decay_start, decay_final, decay_duration)

    assert pytest.approx(decay_func(1), decay_start)


def test_get_linear_decay_final():
    decay_start, decay_final, decay_duration = (1, 0.1, 1000000)
    decay_func = get_linear_decay(decay_start, decay_final, decay_duration)

    assert pytest.approx(decay_func(decay_duration), decay_final)
