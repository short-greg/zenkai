# 3rd party
import torch


def gausian_noise(x: torch.Tensor, std: float=1.0, mean: float=0.0) -> torch.Tensor:
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
