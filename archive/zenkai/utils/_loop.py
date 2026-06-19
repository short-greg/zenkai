import typing

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def minibatch(
    *x: torch.Tensor, shuffle: bool=True, drop_last: bool=False
) -> typing.Iterator[torch.Tensor]:
    
    """
    Generates minibatches from the given tensors.
    Args:
        *x (torch.Tensor): One or more tensors to be split into minibatches.
        shuffle (bool, optional): If True, shuffles the data before splitting into minibatches. Defaults to True.
        drop_last (bool, optional): If True, drops the last incomplete minibatch if the dataset size is not divisible by the batch size. Defaults to False.
    Returns:
        typing.Iterator[torch.Tensor]: An iterator over minibatches of the input tensors.
    """
    dataset = TensorDataset(*x)
    loader = DataLoader(drop_last=drop_last, shuffle=shuffle, dataset=dataset)

    for mb in loader:
        yield mb


def filter_module(
    module: nn.Module, filter_type: typing.Type
):
    """
    Recursively filters and yields submodules of a given module that match a specified type.
    Args:
        module (nn.Module): The parent module to filter.
        filter_type (typing.Type): The type of submodules to filter for.
    Yields:
        nn.Module: Submodules of the given module that are instances of the specified type.
    """
    for child in module.children():
        if isinstance(child, filter_type):
            yield child
        yield from filter_module(child, filter_type)


def apply_module(
    module: nn.Module, apply_type: typing.Type, f: typing.Callable
):
    """
    Recursively applies a given function to all submodules of a specified type within a PyTorch module.
    Args:
        module (nn.Module): The root module to start the search from.
        apply_type (typing.Type): The type of submodules to which the function should be applied.
        f (typing.Callable): The function to apply to each submodule of the specified type.
    Returns:
        None
    """
    for child in module.children():
        if isinstance(child, apply_type):
            f(child)
        apply_module(child, apply_type, f)
