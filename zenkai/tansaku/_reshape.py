import torch
import typing

import torch
import torch.nn as nn


def unsqueeze_to(source: torch.Tensor, align_to: torch.Tensor) -> torch.Tensor:
    """Unsqueeze a tensor to align with another tensor that has more dimensions
    Will only work if source has fewer dimensions than align to and all of those dimensions
    are already aligned

    Args:
        source (torch.Tensor): the tensor to unsqueeze
        align_to (torch.Tensor): the tensor to align to

    Returns:
        torch.Tensor: the aligned tensor
    """
    for i in range(source.dim(), align_to.dim()):
        source = source.unsqueeze(i)
    return source


def unsqueeze_vector(source: torch.Tensor, align_to: torch.Tensor, dim: int=0) -> torch.Tensor:
    """Unsqueeze a 1d  

    Args:
        source (torch.Tensor): the tensor to unsqueeze
        align_to (torch.Tensor): the tensor to align to

    Returns:
        torch.Tensor: the aligned tensor
    """
    for i in range(align_to.dim()):
        if i != dim:
            source = source.unsqueeze(i)
    return source


def align(source: torch.Tensor, align_to: torch.Tensor) -> torch.Tensor:
    """Unsqueeze a tensor to align with another tensor that has more dimensions
    Will only work if source has fewer dimensions than align to and all of those dimensions
    are already aligned

    Args:
        source (torch.Tensor): the tensor to unsqueeze
        align_to (torch.Tensor): the tensor to align to

    Returns:
        torch.Tensor: the aligned tensor
    """
    shape = [1] * source.dim()
    for i in range(source.dim(), align_to.dim()):
        source = source.unsqueeze(i)
        shape.append(align_to.shape[i])
    source = source.repeat(*shape)
    return source



def expand_k(x: torch.Tensor, k: int, reshape: bool = True) -> torch.Tensor:
    """expand the trial dimension in the tensor (separates the trial dimension from the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        k (int): The number of trials
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    shape = torch.Size([k, -1, *x.shape[1:]])
    if reshape:
        return x.reshape(shape)
    return x.view(shape)


def collapse_k(x: torch.Tensor, reshape: bool = True) -> torch.Tensor:
    """collapse the trial dimension in the tensor (merges the trial dimension with the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The collapsed tensor
    """
    if reshape:
        return x.reshape(-1, *x.shape[2:])
    return x.view(-1, *x.shape[2:])


def expand_dim0(x: torch.Tensor, k: int, reshape: bool = False) -> torch.Tensor:
    """Expand an input to repeat k times

    Args:
        x (torch.Tensor): input tensor
        k (int): Number of times to repeat. Must be greater than 0
        reshape (bool, optional): Whether to reshape the output so the first 
            and second dimensions are combined. Defaults to False.

    Raises:
        ValueError: If k is less than or equal to 0

    Returns:
        torch.Tensor: the expanded tensor
    """
    if k <= 0:
        raise ValueError(f"Argument k must be greater than 0 not {k}")

    y = x[None]

    y = y.repeat(k, *([1] * len(y.shape[1:])))  # .transpose(0, 1)
    if reshape:
        return y.view(y.shape[0] * y.shape[1], *y.shape[2:])
    return y


def flatten_dim0(x: torch.Tensor):
    """Flatten the population and batch dimensions of a population"""
    if x.dim() < 2:
        return x
    return x.view(x.shape[0] * x.shape[1], *x.shape[2:])


def deflatten_dim0(x: torch.Tensor, k: int) -> torch.Tensor:
    """Deflatten the population and batch dimensions of a population"""
    if x.dim() == 0:
        raise ValueError("Input dimension == 0")

    return x.view(k, -1, *x.shape[1:])


def undo_cat1d(model: nn.Module, x: torch.Tensor) -> torch.Tensor:

    if isinstance(model, nn.Module):
        model = model.parameters()

    tensors = []
    start = 0
    for p in model:
        end = start + p.numel()
        tensors.append(
            x[start: end].reshape(p.shape)
        )
        start = end

    return tensors


def cat1d(tensors: typing.List[torch.Tensor]) -> torch.Tensor:

    return torch.cat(
        [tensor.flatten(0) for tensor in tensors], dim=0
    )

