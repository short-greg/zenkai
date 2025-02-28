import torch
from torch import nn
import typing


class Lambda(nn.Module):

    def __init__(self, f: typing.Callable, *args, **kwargs):

        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def forward(self, *x: torch.Tensor):

        y = self.f(*x)
        if len(x) == 1:
            return y[0]
        return y


class Null(nn.Module):
    """
    Module that does not act on the inputs
    """

    def forward(self, *x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        """Send multiple single value forward

        Returns:
            typing.Tuple[torch.Tensor]: The inputs
        """
        if len(x) == 1:
            return x[0]
        return x
    
    def reverse(self, *x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        """Send multiple single value forward

        Returns:
            typing.Tuple[torch.Tensor]: The inputs
        """
        if len(x) == 1:
            return x[0]
        return x
