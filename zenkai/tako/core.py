# 1st party
import typing
import uuid
from dataclasses import dataclass

# 3rd party
import torch
import torch.nn as nn



class ID(object):
    """ID for a Tako
    """
    def __init__(self, id: uuid.UUID = None):

        self.x = id if id is not None else uuid.uuid4()


class _UNDEFINED:
    """Class used to indicate something is undefined
    """
    def __str__(self):
        return "UNDEFINED"


UNDEFINED = _UNDEFINED() # instance of undefined object


class Gen(nn.Module):
    """Module that executes only if the input is true"""

    def __init__(self, generator: typing.Callable[[], torch.Tensor], *args, **kwargs):
        """initializer

        Args:
            generator (typing.Callable[[], torch.Tensor]): _description_
        """
        super().__init__()
        self._f = generator
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: bool) -> typing.Any:

        if x is True:
            return self._f(*self.args, **self.kwargs)
        return UNDEFINED


@dataclass
class Info:
    """Information on the Tako
    """
    tags: typing.List[str] = None
    annotation: str = None
