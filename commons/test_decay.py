"""
test_decay.py
"""
import pytest

from .decay import get_linear_decay


def test_get_linear_decay_start():
    """
    Test that linear decay function returns startng value with input 1.
    """
    decay_start, decay_final, decay_duration = (1, 0.1, 1000000)
    decay_func = get_linear_decay(decay_start, decay_final, decay_duration)

    assert pytest.approx(decay_func(1), decay_start)


def test_get_linear_decay_final():
    """
    Test that linear decay function returns final value at the end of duration.
    """
    decay_start, decay_final, decay_duration = (1, 0.1, 1000000)
    decay_func = get_linear_decay(decay_start, decay_final, decay_duration)

    assert pytest.approx(decay_func(decay_duration), decay_final)


def test_get_linear_decay_no_decay():
    """
    Test that linear decay function does not decay if starting and final values are same.
    """
    decay_start, decay_final, decay_duration = (0.5, 0.5, 1000000)
    decay_func = get_linear_decay(decay_start, decay_final, decay_duration)

    assert pytest.approx(decay_func(1), decay_func(decay_duration))


def test_get_linear_decay_after_duration():
    """
    Test that linear decay function does not return value below minimum value.
    """
    decay_start, decay_final, decay_duration = (1, 0.3, 1000)
    decay_func = get_linear_decay(decay_start, decay_final, decay_duration)

    assert pytest.approx(decay_func(decay_duration + 1000), decay_final)


def test_get_linear_decay_thorough():
    """
    Test all values of linear decay function.
    """
    decay_start, decay_final, decay_duration = (1, 0, 11)
    decay_func = get_linear_decay(decay_start, decay_final, decay_duration)

    assert pytest.approx(decay_func(1), 1)
    assert pytest.approx(decay_func(2), 0.9)
    assert pytest.approx(decay_func(3), 0.8)
    assert pytest.approx(decay_func(4), 0.7)
    assert pytest.approx(decay_func(5), 0.6)
    assert pytest.approx(decay_func(6), 0.5)
    assert pytest.approx(decay_func(7), 0.4)
    assert pytest.approx(decay_func(8), 0.3)
    assert pytest.approx(decay_func(9), 0.2)
    assert pytest.approx(decay_func(10), 0.1)
    assert pytest.approx(decay_func(11), 0)
    assert pytest.approx(decay_func(12), 0)
