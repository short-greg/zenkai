import typing

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


def noise(x: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor], dim: int=1) -> torch.Tensor:
    """Add noise to a regular tensor

    Args:
        x (torch.Tensor): The input to add noise to
        k (int): The size of the population
        f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
        dim (int, optional): The population dim. Defaults to 1.

    Returns:
        torch.Tensor: The noise added to the Tensor
    """

    shape = list(x.shape)
    shape.insert(dim, k)
    x = x.unsqueeze(dim)

    return f(x, shape, x.dtype, x.device)


def noise_cat(x: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor], dim: int=1) -> torch.Tensor:
    """Add noise to a regular tensor and then concatenate
    the original tensor

    Args:
        x (torch.Tensor): The input to add noise to
        k (int): The size of the population
        f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
        dim (int, optional): The population dim. Defaults to 1.

    Returns:
        torch.Tensor: The noise added to the Tensor
    """
    shape = list(x.shape)
    shape.insert(dim, k)
    x = x.unsqueeze(dim)

    out = f(x, shape, x.dtype, x.device)
    return torch.cat(
        [x, out], dim=dim
    )


def noise_pop(x: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor], dim: int=1):
    """Add noise to a population tensor

    Args:
        x (torch.Tensor): The input to add noise to
        k (int): The size of the population
        f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
        dim (int, optional): The population dim. Defaults to 1.

    Returns:
        torch.Tensor: The noise added to the Tensor
    """

    shape = list(x.shape)
    base_shape = list(x.shape)
    shape.insert(dim + 1, k)
    base_shape[dim] = base_shape[dim] * k

    x = x.unsqueeze(dim + 1)

    y = f(x, shape, x.dtype, x.device)
    return y.reshape(base_shape)


def noise_cat_pop(x: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor], dim: int=1):
    """Add noise to a population tensor and then
    concatenate to the original tensor

    Args:
        x (torch.Tensor): The input to add noise to
        k (int): The size of the population
        f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
        dim (int, optional): The population dim. Defaults to 1.

    Returns:
        torch.Tensor: The noise added to the Tensor
    """
    shape = list(x.shape)
    base_shape = list(x.shape)
    shape.insert(dim + 1, k)
    base_shape[dim] = base_shape[dim] * k

    x = x.unsqueeze(dim + 1)

    y = f(x, shape, x.dtype, x.device)
    out = y.reshape(base_shape)
    return torch.cat(
        [y, out], dim=dim
    )
