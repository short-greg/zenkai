# flake8: noqa
"""Tensor and parameter utilities: type conversion, gradient context managers,
and the ``memory`` sub-package for storing batches of samples.
"""

from . import memory
from ._convert import checkattr, module_factory
from ._grad import grad_undo
from .memory import BatchMemory
