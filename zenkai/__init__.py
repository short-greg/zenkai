# flake8: noqa

__version__ = "0.0.4"

from . import targetprob, tansaku, utils, ensemble, feedback, scikit
from .kaku import *
from .utils import (
    _params as params, _build as build, 
    _memory as memory
)


def example_function(param1, param2):
    """
    Example function that performs an operation.

    Args:
        param1 (str): The first parameter.
        param2 (int): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.
    """
    return True
