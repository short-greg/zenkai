# 1st party
import typing
import uuid
from dataclasses import dataclass

# 3rd party
import torch
import torch.nn as nn

# TODO:
# Think more about this. I do not necessarily want to store
# the outputs (y values).
# If I store the outputs, for the nested version, I should
# store the intermediate evaluations as well.
#
# When the nested version is used, it will loop over the
# forward_iter if the inputs are defined else it will not
#


class ID(object):
    def __init__(self, id: uuid.UUID = None):

        self.x = id if id is not None else uuid.uuid4()


class _UNDEFINED:
    def __str__(self):
        return "UNDEFINED"


UNDEFINED = _UNDEFINED()


# TODO
class Func(nn.Module):
    """Module that wraps a function call"""

    def __init__(self, f: typing.Callable, *args, **kwargs):
        super().__init__()
        self._f = f
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self._f(x, *self.args, **self.kwargs)


class Gen(nn.Module):
    """ """

    def __init__(self, generator: typing.Callable[[], torch.Tensor], *args, **kwargs):
        super().__init__()
        self._f = generator
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: bool):

        if x is True:
            return self._f(*self.args, **self.kwargs)
        return UNDEFINED


@dataclass
class Info:
    tags: typing.List[str] = None
    annotation: str = None
