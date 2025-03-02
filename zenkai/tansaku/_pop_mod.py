# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn
# local
from ..utils import (
    collapse_batch, collapse_feature, separate_batch, separate_feature
)
from ._pop_params import PopModule

# TODO: REMOVE!


# TODO: add tests
class AdaptPopBatch(nn.Module):
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
        """Adapt the module with  apopulation separating by the batch dimension

        Returns:
            torch.Tensor: The output of the module
        """
        k = x[0].size(0)
        
        x = tuple(collapse_batch(x_i) for x_i in x)
        x = self.module(*x)
        if isinstance(x, typing.Tuple):
            return tuple(
                separate_batch(x_i, k) for x_i in x
            )
        
        return separate_batch(x, k)


class AdaptPopFeature(PopModule):
    """Use to adapt a population of samples for evaluating perturbations of models. 
    """

    def __init__(
        self, module: nn.Module, n_members: int=None, feature_dim: int=1
    ):
        """Adapt module to work with a population of inputs

        Args:
            module (nn.Module): The module to a adapt
        """
        super().__init__(n_members, feature_dim, True)
        self.module = module
        self.feature_dim = feature_dim

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Adapt the module with  apopulation separating by feature

        Returns:
            torch.Tensor: The output of the module
        """
        k = x[0].size(0)
        
        x = tuple(collapse_feature(x_i, self.feature_dim) for x_i in x)
        x = self.module(*x)

        if isinstance(x, typing.Tuple):
            return tuple(
                separate_feature(x_i, k, self.feature_dim) for x_i in x
            )
        return separate_feature(x, k, self.feature_dim)


class NullPopAdapt(PopModule):
    """Use for modules that already have a population component
    """

    def __init__(self, module: nn.Module, n_members: int=None, dim: int=0):
        """Adapt module to work with a population of inputs

        Args:
            module (nn.Module): The module to a adapt
        """
        super().__init__(n_members, out_dim=dim, p_dim=dim, mixed=False)
        self.module = module

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Adapt the module with  apopulation separating by feature

        Returns:
            torch.Tensor: The output of the module
        """
        return self.module(*x)


def adapt_feature(module: nn.Module, *x: torch.Tensor, dim: int=2) -> torch.Tensor | typing.Tuple[torch.Tensor]:        
    
    k = x[0].size(0)
        
    x = tuple(collapse_feature(x_i, dim) for x_i in x)
    x = module(*x)

    if isinstance(x, typing.Tuple):
        return tuple(
            separate_feature(x_i, k, dim) for x_i in x
        )
    return separate_feature(x, k, dim)


def adapt_batch(module: nn.Module, *x: torch.Tensor) -> torch.Tensor | typing.Tuple[torch.Tensor]:

    k = x[0].size(0)
    
    x = tuple(collapse_batch(x_i) for x_i in x)
    x = module(*x)
    if isinstance(x, typing.Tuple):
        return tuple(
            separate_batch(x_i, k) for x_i in x
        )
    
    return separate_batch(x, k)


