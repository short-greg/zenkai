# 1st Party
import typing

# 3rd Party
import torch
import torch.nn as nn

# Local
from .._core import grad_set, p_loop, p_transfer


class grad_undo(object):
    """Context that runs an operation updating the parameters then sets them back for a subset of those parameters."""

    def __init__(
        self,
        values: typing.Iterable[
            typing.Union[typing.Callable[[], typing.Iterator], nn.Module, torch.Tensor, nn.parameter.Parameter]
        ],
    ):
        """Undoes updates to the gradients on the parameters passed in.

        Useful if you only want to update a subset of the gradients.

        Args:
            values: The values whose gradient updates should be undone. May be a callable
                returning an iterator, an ``nn.Module``, a tensor, or a parameter.
        """
        if isinstance(values, nn.Module) or isinstance(values, torch.Tensor):
            values = [values]

        self._values = values
        self._stored = []
        for value in self._values:
            if isinstance(value, torch.Tensor) or isinstance(value, torch.nn.parameter.Parameter):
                self._stored.append(value.grad.clone() if value.grad is not None else None)
            else:
                self._stored.append(list(p_loop(value, lambda v: v.grad.clone() if v.grad is not None else None)))

    def __enter__(self):

        pass

    def __exit__(self, exc_type, exc_val, exc_tb):

        if exc_type is not None:
            exc_val
        for stored, value in zip(self._stored, self._values):

            if isinstance(value, torch.Tensor) or isinstance(value, torch.nn.parameter.Parameter):
                if value.grad is not None and stored is not None:
                    with torch.no_grad():
                        value.grad.copy_(stored)
                elif value.grad is not None:
                    value.grad = None
                else:
                    value.grad = stored
            else:
                p_transfer(value, stored, lambda p1, p2: grad_set(p1, p2))
