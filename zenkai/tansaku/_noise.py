"""
Modules to implement exploration
on the forward pass
"""
# 1st party
import typing

# 3rd party
import torch.nn as nn
import torch


def gaussian_sample(
    mean: torch.Tensor, std: torch.Tensor, k: int = None
) -> torch.Tensor:
    """generate a sample from a gaussian

    Args:
        mean (torch.Tensor): _description_
        std (torch.Tensor): _description_
        k (int): The number of samples to generate. If None will generate 1 sample and the dimension
         will not be expanded

    Returns:
        torch.Tensor: The sample or samples generated
    """
    if k is None:
        return torch.randn_like(mean) * std + mean

    if k <= 0:
        raise ValueError(f"Argument {k} must be greater than 0")
    return (
        torch.randn([k, *mean.shape], device=mean.device, dtype=mean.dtype)
        * std[None]
        + mean[None]
    )


def gaussian_noise(x: torch.Tensor, std: float=1.0, mean: float=0.0) -> torch.Tensor:
    """Add Gaussian noise to the input

    Args:
        x (torch.Tensor): The input
        std (float, optional): The standard deviation for the noise. Defaults to 1.0.
        mean (float, optional): The bias for the noise. Defaults to 0.0.

    Returns:
        torch.Tensor: The input with noise added
    """

    return x + torch.randn_like(x) * std + mean


def binary_noise(x: torch.Tensor, flip_p: bool = 0.5, signed_neg: bool = True) -> torch.Tensor:
    """Flip the input features with a certain probability

    Args:
        x (torch.Tensor): The input features
        flip_p (bool, optional): The probability to flip with. Defaults to 0.5.
        signed_neg (bool, optional): Whether negative is signed or 0. Defaults to True.

    Returns:
        torch.Tensor: The input with noise added
    """
    to_flip = torch.rand_like(x) > flip_p
    if signed_neg:
        return to_flip.float() * -x + (~to_flip).float() * x
    return (x - to_flip.float()).abs()


def binary_prob(
    x: torch.Tensor, loss: torch.Tensor, retrieve_counts: bool = False
) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Calculate binary probability based on a binary-valued vector input

    Args:
        x (torch.Tensor): The population input
        loss (torch.Tensor): The loss
        retrieve_counts (bool, optional): Whether to return the positive
          and negative counts in the result. Defaults to False.

    Returns:
        typing.Union[ torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor] ]: _description_
    """
    is_pos = (x == 1).unsqueeze(-1)
    is_neg = ~is_pos
    pos_count = is_pos.sum(dim=0)
    neg_count = is_neg.sum(dim=0)
    positive_loss = (loss[:, :, None] * is_pos.float()).sum(dim=0) / pos_count
    negative_loss = (loss[:, :, None] * is_neg.float()).sum(dim=0) / neg_count
    updated = (positive_loss < negative_loss).type_as(x).mean(dim=-1)

    if not retrieve_counts:
        return updated

    return updated, pos_count.squeeze(-1), neg_count.squeeze(-1)


class FreezeDropout(nn.Module):
    """Freeze the dropout parameter so that the same parameter will be used """

    def __init__(self, p: float, freeze: bool = False):
        """Create a FreezeDropout to keep the parameter frozen. This is useful if you want
        to go through the network multiple times and get the same output

        Args:
            p (float): The dropout rate
            freeze (bool, optional): Whether to freeze the dropout. Defaults to False.

        Raises:
            ValueError: If p is greater or equal to one or less than zero
        """
        super().__init__()
        if p >= 1.0 or p < 0.0:
            raise ValueError(f"P must be in range [0.0, 1.0) not {p}")
        self.p = p
        self.freeze = freeze
        self._cur = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute dropout on the input

        Args:
            x (torch.Tensor): The input to dropout

        Returns:
            torch.Tensor: The 
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
