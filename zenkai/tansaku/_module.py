from abc import abstractmethod, ABC
from typing_extensions import Self

import torch
import torch.nn as nn


class PopModule(nn.Module, ABC):

    def __init__(self, n_members: int=None):
        """
        Args:
            n_members (int): The number of members in the population
        """
        super().__init__()
        self._n_members = n_members

    @property
    def n_members(self) -> int:
        return self._n_members
    
    @abstractmethod
    def mean(self) -> Self:
        pass

    @abstractmethod
    def to_pop(self) -> Self:
        pass
    
    @abstractmethod
    def member(self, i: int) -> Self:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass 
