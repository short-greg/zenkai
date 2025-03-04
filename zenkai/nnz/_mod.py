import torch
import torch.nn as nn
import typing

class Updater(nn.Module):
    """Use for updating a tensor (such as with a decay function)
    """

    def __init__(self, update_f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None, *args, **kwargs):
        """Module that handles updating a tensor with an update function

        Args:
            update_f (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): The update function. If it is None, the default behavior will be to do nothing but the behavior can be overridden by sublcasses. Defaults to None.
        """
        super().__init__()
        self.update_f = update_f
        self.args = args
        self.kwargs = kwargs
        self.cur_val = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update the tensor stored internally and return it

        Args:
            x (torch.Tensor): The tensor to update with

        Returns:
            torch.Tensor: The updated tensor
        """
        if self.cur_val is None:
            self.cur_val = x

        elif self.update_f is not None:
            self.cur_val = self.update_f(
                x, self.cur_val, *self.args, **self.kwargs
            )
            return self.cur_val
        return x
