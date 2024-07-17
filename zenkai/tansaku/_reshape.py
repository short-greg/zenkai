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


def separate_batch(x: torch.Tensor, k: int, reshape: bool = True, pop_dim: int=0) -> torch.Tensor:
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


def collapse_batch(x: torch.Tensor, reshape: bool = True) -> torch.Tensor:
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


def collapse_feature(x: torch.Tensor, feature_dim: int=2, reshape: bool=True) -> torch.Tensor:
    """Collapse the feature dimension and population dimensions into one dimension

    Args:
        x (torch.Tensor): The tensor to expand
        reshape (bool, optional): Whether to use reshape or view. Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    permutation = list(range(x.dim()))
    permutation = [
        *permutation[1:feature_dim],
        0, *permutation[feature_dim:]
    ]

    shape = list(x.shape)
    shape[feature_dim] = shape[0] * shape[feature_dim]
    shape.pop(0)

    x = x.permute(permutation)
    if reshape:
        return x.reshape(shape)
    return x.view(shape)


def separate_feature(x: torch.Tensor, k: int, feature_dim: int=2, reshape: bool=True) -> torch.Tensor:
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
    else: x = x.view(shape)
    permutation = list(range(x.dim()))
    permutation = [
        permutation[feature_dim - 1], 
        *permutation[:feature_dim - 1],
        *permutation[feature_dim:]
    ]
    print(permutation)
    return x.permute(permutation)


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
        tensors.append(
            x[start: end].reshape(p.shape)
        )
        start = end

    return tensors


def cat1d(tensors: typing.List[torch.Tensor]) -> torch.Tensor:
    """Concatenate tensors to a 1d tensor

    Args:
        tensors (typing.List[torch.Tensor]): The tensors to concatenate

    Returns:
        torch.Tensor: The concatenated tensors
    """
    return torch.cat(
        [tensor.flatten(0) for tensor in tensors], dim=0
    )

# TODO: Depracate the following

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

