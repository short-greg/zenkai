# 1st party
import typing

# 3rd party
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


def unsqueeze_vector(source: torch.Tensor, align_to: torch.Tensor, dim: int = 0) -> torch.Tensor:
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


def shape_as(source: torch.Tensor, n: int) -> torch.Size:
    """Get the shape of a non-population source and add in the population size

    Args:
        source (torch.Tensor): The tensor to base the shape off of
        n (int): The population size

    Returns:
        torch.Size: The size with the population
    """
    shape = list(source.shape)
    shape.insert(0, n)
    return torch.Size(shape)


def tensor_align(source: torch.Tensor, align_to: torch.Tensor) -> torch.Tensor:
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


def batch_separate(x: torch.Tensor, k: int, reshape: bool = True, pop_dim: int = 0) -> torch.Tensor:
    """expand the batch and trial dimension in the tensor (separates the trial dimension from the sample dimension)

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


def batch_collapse(x: torch.Tensor, reshape: bool = True) -> torch.Tensor:
    """collapse the batch and population dimension in the tensor (merges the trial dimension with the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The collapsed tensor
    """
    if reshape:
        return x.reshape(-1, *x.shape[2:])
    return x.view(-1, *x.shape[2:])


def feature_collapse(x: torch.Tensor, dim: int = 2, reshape: bool = True) -> torch.Tensor:
    """Collapse the feature dimension and population dimensions into one dimension

    Args:
        x (torch.Tensor): The tensor to expand
        reshape (bool, optional): Whether to use reshape or view. Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    permutation = list(range(x.dim()))
    permutation = [*permutation[1:dim], 0, *permutation[dim:]]

    shape = list(x.shape)
    shape[dim] = shape[0] * shape[dim]
    shape.pop(0)

    x = x.permute(permutation)
    if reshape:
        return x.reshape(shape)
    return x.view(shape)


def feature_separate(x: torch.Tensor, k: int, feature_dim: int = 2, reshape: bool = True) -> torch.Tensor:
    """Separate the feature dimension for when the population and feature dimensions have been collapsed

    Args:
        x (torch.Tensor): The tensor to expand
        k (int): The population size
        reshape (bool, optional): Whether to use reshape or view. Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    shape = list(x.shape)
    shape.insert(1, k)
    shape[feature_dim] = -1

    if reshape:
        x = x.reshape(shape)
    else:
        x = x.view(shape)
    permutation = list(range(x.dim()))
    permutation = [
        permutation[feature_dim - 1],
        *permutation[: feature_dim - 1],
        *permutation[feature_dim:],
    ]
    return x.permute(permutation)


def dim_separate(x: torch.Tensor, n: int, dim: int) -> torch.Tensor:

    shape = list(x.shape)
    shape[dim] = -1
    shape.insert(dim, n)
    return x.reshape(x)


def dim_combine(x: torch.Tensor, from_dim: int):

    shape = list(x.shape)
    shape[from_dim] = shape[from_dim] * shape[from_dim + 1]
    shape.pop(from_dim + 1)
    return x.view(shape)


def undo_cat1d(model: nn.Module, x: torch.Tensor) -> typing.List[torch.Tensor]:
    """Undo the concatenation

    Args:
        model (nn.Module): The model
        x (torch.Tensor): The concatenated tensors

    Returns:
        torch.Tensor: The tensors
    """
    if isinstance(model, nn.Module):
        model = model.parameters()

    tensors = []
    start = 0
    for p in model:
        end = start + p.numel()
        tensors.append(x[start:end].reshape(p.shape))
        start = end

    return tensors


def cat1d(tensors: typing.List[torch.Tensor]) -> torch.Tensor:
    """Concatenate tensors to a 1d tensor

    Args:
        tensors (typing.List[torch.Tensor]): The tensors to concatenate

    Returns:
        torch.Tensor: The concatenated tensors
    """
    return torch.cat([tensor.flatten(0) for tensor in tensors], dim=0)
