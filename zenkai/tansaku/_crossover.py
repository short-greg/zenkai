# 1st party
import typing
import random
import math

# 3rd party
import torch
import torch.nn as nn


class CrossOver(nn.Module):
    """CrossOver chromosomes
    """

    def __init__(self, f: typing.Callable=None, **kwargs):
        """Create a CrossOver module with a specified function

        Args:
            f (typing.Callable, optional): The function to do crossover with. Defaults to None.
        """
        super().__init__()
        self.f = f
        self.kwargs = kwargs

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Crossover the parent chromosomes together

        Args:
            x1 (torch.Tensor): The second parent
            x2 (torch.Tensor): The first parent

        Returns:
            torch.Tensor: The children
        """
        if self.f is not None:
            return self.f(x1, x2, **self.kwargs)
        return x1


def full_crossover(x1: torch.Tensor, x2: torch.Tensor, p1: float=0.5) -> torch.Tensor:
    """Choose either x1 or x2 as the parent

    Args:
        x1 (torch.Tensor): The first parent
        x2 (torch.Tensor): The second parent
        p1 (float, optional): The prob of choosing x1. Defaults to 0.5.

    Returns:
        torch.Tensor: The children (x1 or x2)
    """
    if random.random() < p1:
        return x1
    return x2


def smooth_crossover(x1: torch.Tensor, x2: torch.Tensor, pref1: float=0.0, dim: int=None) -> torch.Tensor:
    """Smoothly interpolate between the two parents

    Args:
        x1 (torch.Tensor): The first parent
        x2 (torch.Tensor): The second parent
        pref1 (float, optional): The preference for x1. If greater than 0, x1 is preferred. Preference is computed using the exponent. Defaults to 0.0.
        dim (int, optional): The dimension to crossover on. If not specified all elements will be crossed over. Defaults to None.

    Returns:
        torch.Tensor: The children
    """
    exp_weight = math.exp(pref1)

    if dim is None:
        p = torch.rand_like(x1)
    else:
        shape = list(x1.shape)
        shape[dim] = 1
        p = torch.rand(shape, x1.device, x1.dtype)
    
    p = p ** exp_weight
    return (
        p * x1 + (1 - p) * x2
    )


def hard_crossover(x1: torch.Tensor, x2: torch.Tensor, x1_thresh: float=0.5, dim: int=None) -> torch.Tensor:
    """Choose either x1 or x2

    Args:
        x1 (torch.Tensor): The first parent
        x2 (torch.Tensor): The second parent
        x1_thresh (float, optional): The preference for x1. The higher the threshold the more likely this parent is chosen. If it is 1.0, it is always chosen. If it is 0.0 it is never chosen. Defaults to 0.5.
        dim (int, optional): The dimension to crossover. If none, all elements will be crossed over. Defaults to None.

    Returns:
        torch.Tensor: The children
    """

    if dim is None:
        p = torch.rand_like(x1)
    else:
        shape = list(x1.shape)
        shape[dim] = 1
        p = torch.rand(shape, x1.device, x1.dtype)
        
    choose1 = p < x1_thresh
    return (
        choose1 * x1 + (~choose1) * x2
    )


def cross_pairs(
    x1: typing.Iterable[torch.Tensor], x2: typing.Iterable[torch.Tensor], f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Cross over multiple pairs

    Args:
        x1 (torch.Tensor): The first parent
        x2 (torch.Tensor): The second parent
        f (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The crossover function

    Returns:
        torch.Tensor: The children of the parents
    """
    return tuple(
        f(x1_i, x2_i) for x1_i, x2_i in zip(x1, x2)
    )
