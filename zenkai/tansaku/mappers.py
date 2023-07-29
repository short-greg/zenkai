from abc import ABC, abstractmethod

from .core import Individual, Population

import torch


class IndividualMapper(ABC):
    """Mixes two individuals together"""

    @abstractmethod
    def __call__(self, individual: Individual) -> Individual:
        pass

    @abstractmethod
    def spawn(self) -> "IndividualMapper":
        pass


class PopulationMapper(ABC):
    """Mixes two populations together"""

    @abstractmethod
    def __call__(self, population: Population) -> Population:
        pass

    @abstractmethod
    def spawn(self) -> "PopulationMapper":
        pass
