# flake8: noqa
"""Zenkai — a framework for building learning machines beyond backpropagation."""

__version__ = "0.0.9"

# Sub-packages
from . import lm, nnz, optimz, utils

# _core holds the framework's shared primitives, re-exported at the zenkai root.
from ._core import *
