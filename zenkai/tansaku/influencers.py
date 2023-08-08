# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch

from ..utils import to_signed_neg, to_zero_neg
from .core import Individual, Population

# local
from .reducers import SlopeReducer


class IndividualInfluencer(ABC):
    """Modifies an Individual based on the Population"""

    @abstractmethod
    def __call__(self, individual: Individual, population: Population) -> Individual:
        pass

    @abstractmethod
    def spawn(self) -> "IndividualInfluencer":
        pass


class PopulationInfluencer(ABC):
    """"""

    @abstractmethod
    def __call__(self, population: Population, individual: Population) -> Population:
        pass

    @abstractmethod
    def spawn(self) -> "PopulationInfluencer":
        pass


class SlopeInfluencer(IndividualInfluencer):
    """Modifies an individual based on the slope of a population"""

    def __init__(
        self, momentum: float, lr: float = 0.1, x: str = "x", maximize: bool = False
    ):
        """initializer

        Args:
            momentum (float): The amount of momentum for the slope
            lr (float, optional): The "learning" rate (i.e. how much to go in the direction).
                Defaults to 0.1.
            x (str, optional): The name for the x value. Defaults to "x".
            maximize (bool, optional): Whether to maximize or minimize.
              If minimizing, the sign of the lr will be reversed. Defaults to False.
        """
        self._slope_selector = SlopeReducer(momentum)
        self._lr = lr if maximize else -lr
        self.x = x
        self._momentum = momentum
        self._multiplier = 1 if maximize else -1

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = abs(lr) * self._multiplier

    @property
    def maximize(self) -> bool:
        return self._multiplier == 1

    @maximize.setter
    def maximize(self, maximize):
        self._multiplier = 1 if maximize else -1

    def __call__(self, original: Individual, population: Population) -> Individual:
        x = original[self.x]
        slope = self._slope_selector(population)[self.x]
        return Individual(**{self.x: x + self.lr * slope})

    def spawn(self) -> "SlopeInfluencer":
        return SlopeInfluencer(self._momentum, self.lr, self.x, self._multiplier == 1)


class PopulationLimiter(PopulationInfluencer):
    """
    
    """

    def __init__(self, limit: torch.LongTensor=None):

        self.limit = limit

    def __call__(
        self,
        population: Population,
        individual: Individual,
        limit: torch.LongTensor = None,
    ) -> Population:
        """

        Args:
            population (Population): The population to limit
            individual (Individual): The individual
            limit (torch.LongTensor, optional): The index to use to limit. Defaults to None.

        Returns:
            Population: The limited population
        """
        result = {}

        if self.limit is None:
            return population

        for k, v in population:
            individual_v = individual[k][None].clone()
            individual_v = individual_v.repeat(v.size(0), 1, 1)
            individual_v[:, :, limit] = v[:, :, limit].detach()
            result[k] = individual_v
        return Population(**result)
    
    def spawn(self) -> 'PopulationLimiter':
        return PopulationLimiter(self.limit.clone())
