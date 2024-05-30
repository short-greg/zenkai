# 1st Party
import math
import typing

# 3rd party
import numpy as np
import torch
import torch.nn as nn

# TODO: Organize better


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_th(
    x: np.ndarray,
    dtype: torch.dtype,
    device: torch.device = None,
    requires_grad: bool = False,
    retains_grad: bool = False,
) -> torch.Tensor:
    """

    Args:
        x (np.ndarray): Array to convert
        dtype (torch.dtype): type to convert to
        device (torch.device): device to convert to
        requires_grad (bool, optional): Whether the tensor requires grad. Defaults to False.
        retains_grad (bool, optional): Whether the tensor retains a grad. Defaults to False.

    Returns:
        torch.Tensor: result
    """
    x: torch.Tensor = torch.tensor(
        x, dtype=dtype, requires_grad=requires_grad, device=device
    )
    if retains_grad:
        x.retain_grad()
    return x


def to_th_as(
    x: np.ndarray,
    as_: torch.Tensor,
    requires_grad: bool = False,
    retains_grad: bool = False,
) -> torch.Tensor:
    """

    Args:
        x (np.ndarray): Array to convert
        as_ (torch.Tensor): The array to base conversion off of
        requires_grad (bool, optional): Whether the tensor requires grad. Defaults to False.
        retains_grad (bool, optional): Whether the tensor retains a grad. Defaults to False.

    Returns:
        torch.Tensor: result
    """

    x: torch.Tensor = torch.tensor(
        x, dtype=as_.dtype, requires_grad=requires_grad, device=as_.device
    )
    if retains_grad:
        x.retain_grad()
    return x


def freshen(x: torch.Tensor, requires_grad: bool = True, inplace: bool = False):
    if not isinstance(x, torch.Tensor):
        return x
    if inplace:
        x.detach_()
    else:
        x = x.detach()
    if requires_grad:
        x = x.requires_grad_(requires_grad)
        x.retain_grad()
    return x


def checkattr(name: str):
    # Check that a class has the attribute specified
    def _wrap(f):

        def _(self, *args, **kwargs):
            if not hasattr(self, name):
                raise AttributeError(
                    f'Class of type {type(self)} requires attribute {name} to be set'
                )
            return f(self, *args, **kwargs)
        
        return _
    return _wrap


def to_zero_neg(x: torch.Tensor) -> torch.Tensor:
    """convert a 'signed' binary tensor to have zeros for negatives

    Args:
        x (torch.Tensor): Signed binary tensor. Tensor must be all -1s or 1s to get expected result

    Returns:
        torch.Tensor: The binary tensor with negatives as zero
    """

    return (x + 1) / 2


def to_signed_neg(x: torch.Tensor) -> torch.Tensor:
    """convert a 'zero' binary tensor to have negative ones for negatives

    Args:
        x (torch.Tensor): Binary tensor with zeros for negatives. 
            Tensor must be all zeros and ones to get expected result

    Returns:
        torch.Tensor: The signed binary tensor
    """
    return (x * 2) - 1


def binary_encoding(
    x: torch.LongTensor, n_size: int, bit_size: bool = False
) -> torch.Tensor:
    """Convert an integer tensor to a binary encoding

    Args:
        x (torch.LongTensor): The integer tensor
        n_size (int): The size of the encoding (e.g. number of bits if using bits or the max number)
        bit_size (bool, optional): Whether the size is described in terms of number of bits . Defaults to False.

    Returns:
        torch.Tensor: The binary encoding
    """

    if not bit_size:
        n_size = int(math.ceil(math.log2(n_size)))
    results = []
    for _ in range(n_size):
        results.append(x)
        x = x >> 1
    results = torch.stack(tuple(reversed(results))) & 1
    shape = list(range(results.dim()))
    shape = shape[1:] + shape[0:1]
    return results.permute(*shape)


def module_factory(module: typing.Union[str, nn.Module], *args, **kwargs) -> nn.Module:

    if isinstance(module, nn.Module):
        if len(args) != 0:
            raise ValueError("Cannot set args if module is already defined")
        if len(kwargs) != 0:
            raise ValueError("Cannot set kwargs if module is already defined")

        return module

    return getattr(nn, module)(*args, **kwargs)
