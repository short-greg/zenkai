# 1st party
from abc import abstractmethod, ABC
from typing_extensions import Self

# 3rd party
import torch
import torch.nn as nn

# local
from . import _params
from ..utils import _params as base_params


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
    def spawn(self, n_members: int) -> Self:
        """Spawn a module with same parameters as this and a different number of modules

        Args:
            n_members (int): The number of members for the module

        Returns:
            Self: The spawned pop module
        """
        pass

    def mean(self) -> Self:
        """Spawn a module and set the values to the mean

        Returns:
            Self: The mean of the population
        """
        if self._n_members is None:
            return self
        
        module = self.spawn(None)
        vec = _params.to_pvec(self, self._n_members)
        base_params.set_pvec(module, vec.mean(dim=0))
        return module

    def to_pop(self) -> Self:
        """Convert a module that is not a not a population

        Returns:
            Self: 
        """
        if self._n_members is not None:
            raise RuntimeError(
                'Module already is a population module')
        
        return self.spawn(1)

    def member(self, i: int) -> Self:
        """Retrieve a member with the same params

        Args:
            i (int): The member to retrieve

        Raises:
            RuntimeError: If the module does not have members

        Returns:
            Self: A module created with the member
        """
        if self._n_members is None:
            raise RuntimeError(
                'Module has no members'
            )
        module = self.spawn(None)
        vec = _params.to_pvec(self, self._n_members)
        base_params.set_pvec(module, vec[i])
        return module

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output the population

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The population output
        """
        pass 
