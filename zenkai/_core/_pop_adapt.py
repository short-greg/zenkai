# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from ._shape import (
    batch_collapse,
    batch_separate,
    feature_collapse,
    feature_separate,
)


def feature_adapt(
    module: nn.Module, *x: torch.Tensor, dim: int = 2
) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
    """Adapt a module to operate on a population by collapsing the feature dimension.

    Args:
        module (nn.Module): The module to adapt.
        dim (int): The feature dimension to collapse. Defaults to 2.

    Returns:
        torch.Tensor: The output of the module with the feature dimension restored.
    """
    k = x[0].size(0)

    x = tuple(feature_collapse(x_i, dim) for x_i in x)
    x = module(*x)

    if isinstance(x, typing.Tuple):
        return tuple(feature_separate(x_i, k, dim) for x_i in x)
    return feature_separate(x, k, dim)


def batch_adapt(module: nn.Module, *x: torch.Tensor) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor]]:
    """Adapt a module to operate on a population by collapsing the batch dimension.

    Args:
        module (nn.Module): The module to adapt.

    Returns:
        torch.Tensor: The output of the module with the batch dimension restored.
    """
    k = x[0].size(0)

    x = tuple(batch_collapse(x_i) for x_i in x)
    x = module(*x)
    if isinstance(x, typing.Tuple):
        return tuple(batch_separate(x_i, k) for x_i in x)

    return batch_separate(x, k)
