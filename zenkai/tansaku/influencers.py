# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch


# local
from .reducers import SlopeReducer
from ..utils import to_signed_neg, to_zero_neg
from .core import Individual, Population, expand_dim0
from .populators import RepeatPopulator
from ..kaku import State



class IndividualInfluencer(ABC):
    """Modifies an Individual based on the Population"""

    def __call__(self, individual: Individual, population: Population, state: State=None) -> Individual:
        return self.influence(
            individual, population, state or State()
        )

    @abstractmethod
    def influence(self, individual: Individual, population: Population, state: State) -> Individual:
        pass

    @abstractmethod
    def spawn(self) -> "IndividualInfluencer":
        pass


class PopulationInfluencer(ABC):
    """"""

    @abstractmethod
    def influence(self, population: Population, individual: Population, state: State) -> Population:
        pass

    def __call__(self, population: Population, individual: Population, state: State=None) -> Population:
        return self.influence(
            population, individual, state or State()
        )

    @abstractmethod
    def spawn(self) -> "PopulationInfluencer":
        pass


class JoinIndividual(PopulationInfluencer):
    """"""

    def influence(self, population: Population, individual: Individual, state: State) -> Population:
        
        new_population = {**population}

        for k, v in individual:
            new_population[k] = expand_dim0(v, population.n_individuals)
        
        return Population(**new_population)

    def spawn(self) -> "PopulationInfluencer":
        return JoinIndividual()
    
    def join_t(self, population: Population, t: torch.Tensor):

        return self(population, Individual(t=t))

    def join_tensor(self, population: Population, key: str, val: torch.Tensor):

        return self(population, Individual(**{key: val}))


class SlopeInfluencer(IndividualInfluencer):

    def __init__(
        self, momentum: float, lr: float = 0.1, x: str = "x", maximize: bool = False
    ):
        """Modifies an individual based on the slope of a population

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

    def influence(self, original: Individual, population: Population, state: State) -> Individual:
        x = original[self.x]
        slope = self._slope_selector(population)[self.x]
        return Individual(**{self.x: x + self.lr * slope})

    def spawn(self) -> "SlopeInfluencer":
        return SlopeInfluencer(self._momentum, self.lr, self.x, self._multiplier == 1)


class JoinInfluencer(PopulationInfluencer):
    """
    Add the individual to the front of the pouplation
    """

    def influence(
        self,
        population: Population,
        individual: Individual,
        state: State
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

        for k, v in population.items():
            result[k] = torch.cat(
                [individual[k][None], v], dim=0
            )
        return Population(**result)
    
    def spawn(self) -> 'PopulationLimiter':
        return JoinInfluencer()


def keep_feature(original: Individual, population: Population, limit: torch.LongTensor):
    
    result = {}

    for k, v in population.items():
        individual_v = original[k][None].clone()
        individual_v = individual_v.repeat(v.size(0), 1, 1)
        individual_v[:, :, limit] = v[:, :, limit].detach()
        result[k] = individual_v
    return Population(**result)


class PopulationLimiter(PopulationInfluencer):
    """
    Allows the user to specify certain indices that can be updated in the individual.
    Values at other indices will remain the same
    """

    def __init__(self, limit: torch.LongTensor=None):
        """initializer

        Args:
            limit (torch.LongTensor, optional): The indices to use in the update. If None there is no limit. Defaults to None.
        """

        self.limit = limit

    def influence(
        self,
        population: Population,
        individual: Individual,
        state: State
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

        for k, v in population.items():
            individual_v = individual[k][None].clone()
            individual_v = individual_v.repeat(v.size(0), 1, 1)
            individual_v[:, :, self.limit] = v[:, :, self.limit].detach()
            result[k] = individual_v
        return Population(**result)
    
    def spawn(self) -> 'PopulationLimiter':
        return PopulationLimiter(self.limit.clone())
