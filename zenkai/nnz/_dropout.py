"""
Modules to implement exploration on the forward pass.
"""

# 3rd party
import torch
import torch.nn as nn


class FreezeDropout(nn.Module):
    """Freeze the dropout parameter so that the same parameter will be used"""

    def __init__(self, p: float, freeze: bool = False):
        """Create a FreezeDropout to keep the parameter frozen. This is useful if you want
        to go through the network multiple times and get the same output.

        Args:
            p (float): The dropout rate.
            freeze (bool, optional): Whether to freeze the dropout. Defaults to False.

        Raises:
            ValueError: If p is greater or equal to one or less than zero.
        """
        super().__init__()
        if p >= 1.0 or p < 0.0:
            raise ValueError(f"P must be in range [0.0, 1.0) not {p}")
        self.p = p
        self.freeze = freeze
        self._cur = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute dropout on the input.

        Args:
            x (torch.Tensor): The input to dropout.

        Returns:
            torch.Tensor: The input with dropout applied.
        """
        if self.p == 0.0:
            return x

        if not self.training:
            return x

        if self.freeze and self._cur is not None:
            f = self._cur
        else:
            f = (torch.rand_like(x) > self.p).type_as(x)

        self._cur = f
        return (f * x) * (1 / 1 - self.p)
