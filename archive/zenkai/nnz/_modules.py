import torch
from torch import nn
import typing


class Lambda(nn.Module):
    """Wrap a general function within a module
    """

    def __init__(self, f: typing.Callable, *args, **kwargs):
        """Wrap the function specified by f in a module

        Args:
            f (typing.Callable): The function to wrap
        """
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def forward(self, *x: torch.Tensor) -> typing.Any:
        """Get the output of the wrapped function

        Returns:
            The output of the function
        """

        y = self.f(*x, *self.args, **self.kwargs)
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
