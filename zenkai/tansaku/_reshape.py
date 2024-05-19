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


def separate_batch(x: torch.Tensor, k: int, reshape: bool = True) -> torch.Tensor:
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


def collapse_feature(x: torch.Tensor, reshape: bool=True) -> torch.Tensor:
    """Collapse the feature dimension and population dimensions into one dimension

    Args:
        x (torch.Tensor): The tensor to expand
        reshape (bool, optional): Whether to use reshape or view. Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    shape = list(x.shape)
    shape[1] = shape[1] * shape[2]
    shape.pop(2)
    if reshape:
        return x.reshape(shape)
    return x.view(shape)


def separate_feature(x: torch.Tensor, k: int, reshape: bool=True) -> torch.Tensor:
    """Separate the feature dimension for when
    the population and feature dimensions have been collapsed

    Args:
        x (torch.Tensor): The tensor to expand
        k (int): The population size
        reshape (bool, optional): Whether to use reshape or view. Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    shape = list(x.shape)
    shape[1] = k
    shape.insert(2, -1)
    if reshape:
        return x.reshape(shape)
    return x.view(shape)


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


class AdaptBatch(nn.Module):
    """Use to adapt a population of samples for evaluating perturbations
    of samples. Useful for optimizing "x"
    """

    def __init__(self, module: nn.Module):
        """Instantiate the AdaptBatch model

        Args:
            module (nn.Module): 
        """
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        k = x[0].shape(1)
        
        x = tuple(collapse_batch(x_i) for x_i in x)
        x = self.module(*x)
        if isinstance(x, typing.Tuple):
            return tuple(
                separate_batch(x_i, k) for x_i in x
            )
        
        return separate_batch(x, k)


class AdaptFeature(nn.Module):
    """Use to adapt a population of samples for evaluating perturbations
    of models. Useful for optimizing the parameters
    """

    def __init__(self, module: nn.Module):
        """Adapt module

        Args:
            module (nn.Module): 
        """
        super().__init__()
        self.module = module

    def forward(self, *x: torch.Tensor) -> torch.Tensor:

        k = x[0].shape(1)
        
        x = tuple(collapse_feature(x_i) for x_i in x)
        x = self.module(*x)
        if isinstance(x, typing.Tuple):
            return tuple(
                separate_feature(x_i, k) for x_i in x
            )
        
        return separate_feature(x, k)
