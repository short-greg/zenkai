# 1st party
from abc import abstractmethod, ABC
from typing_extensions import Self
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from . import _params
from ..utils import _params as base_params
from ._reshape import collapse_batch, collapse_feature, separate_batch, separate_feature


class PopModule(nn.Module, ABC):
    """Parent class for a module that outputs a population
    """
    def __init__(self, n_members: int=None):
        """Create a population module with the specified number of members

        Args:
            n_members (int): The number of members in the population
        """
        super().__init__()
        self._n_members = n_members

    @property
    def n_members(self) -> int:
        """
        Returns:
            int: The number of members in the module
        """
        return self._n_members

    @abstractmethod
    def forward(self, x: torch.Tensor, ind: int=None) -> torch.Tensor:
        """Output the population

        Args:
            x (torch.Tensor): The input


        Returns:
            torch.Tensor: The population output
        """
        pass 

    # def mean(self) -> Self:
    #     """Spawn a module and set the values to the mean

    #     Returns:
    #         Self: The mean of the population
    #     """
    #     if self._n_members is None:
    #         return self
        
    #     module = self.spawn(None)
    #     vec = _params.to_pvec(self, self._n_members)
    #     base_params.set_pvec(module, vec.mean(dim=0))
    #     return module

    # @abstractmethod
    # def spawn(self, n_members: int) -> Self:
    #     """Spawn a module with same parameters as this and a different number of modules

    #     Args:
    #         n_members (int): The number of members for the module

    #     Returns:
    #         Self: The spawned pop module
    #     """
    #     pass


    # def to_pop(self) -> Self:
    #     """Convert a module that is not a not a population

    #     Returns:
    #         Self: 
    #     """
    #     if self._n_members is not None:
    #         raise RuntimeError(
    #             'Module already is a population module')
        
    #     return self.spawn(1)

    # def member(self, i: int) -> Self:
    #     """Retrieve a member with the same params

    #     Args:
    #         i (int): The member to retrieve

    #     Raises:
    #         RuntimeError: If the module does not have members

    #     Returns:
    #         Self: A module created with the member
    #     """
    #     if self._n_members is None:
    #         raise RuntimeError(
    #             'Module has no members'
    #         )
    #     module = self.spawn(None)
    #     vec = _params.to_pvec(self, self._n_members)
    #     base_params.set_pvec(module, vec[i])
    #     return module



# TODO: add tests
class AdaptBatch(PopModule):
    """Use to adapt a population of samples for evaluating perturbations
    of samples. Useful for optimizing "x"
    """

    def __init__(self, module: nn.Module, n_members: int=None):
        """Instantiate the AdaptBatch model

        Args:
            module (nn.Module): 
        """
        super().__init__(n_members)
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


class AdaptFeature(PopModule):
    """Use to adapt a population of samples for evaluating perturbations of models. 
    """

    def __init__(self, module: nn.Module, n_members: int=None):
        """Adapt module to work with a population of inputs

        Args:
            module (nn.Module): The module to a adapt
        """
        super().__init__(n_members)
        self.module = module

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Adapt the module with  apopulation separating by feature

        Returns:
            torch.Tensor: The output of the module
        """
        k = x[0].size(0)
        
        x = tuple(collapse_feature(x_i) for x_i in x)
        x = self.module(*x)
        if not isinstance(x, typing.Tuple):
            return separate_feature(x, k)
        return tuple(
            separate_feature(x_i, k) for x_i in x
        )

