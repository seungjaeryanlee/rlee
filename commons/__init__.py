"""Miscellaneous functions that are generally useful."""

from .train_parser import get_train_args
from .decay import get_linear_decay


__all__ = [
    'get_train_args',
    'get_linear_decay',
]
