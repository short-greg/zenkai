# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch

# local
from .core import Population, Individual


class IndividualMixer(ABC):
    """Mixes two individuals together"""

    @abstractmethod
    def __call__(self, individual1: Individual, individual2: Individual) -> Individual:
        pass

    @abstractmethod
    def spawn(self) -> "IndividualMixer":
        pass


class PopulationMixer(ABC):
    """Mixes two populations together"""

    @abstractmethod
    def __call__(self, population1: Population, population2: Population) -> Population:
        pass

    @abstractmethod
    def spawn(self) -> "PopulationMixer":
        pass


class KeepMixer(IndividualMixer):
    """Modify the original based on the selection by keeping the values in
    the individual with a set probability"""

    def __init__(self, keep_p: float):
        """initializer

        Args:
            keep_p (float): Probability of keeping the value for the first individual

        Raises:
            ValueError: If the probability is invalid
        """
        if not (0 < keep_p < 1.0):
            raise ValueError(f"{keep_p} must be between 0 and 1.")
        self.keep_p = keep_p

    def __call__(self, individual1: Individual, individual2: Individual) -> Individual:
        """Randomly choose whether to select original or selection for each value

        Args:
            individual (Individual): The individual to modify
            individual2 (Population): The population to modify based on

        Returns:
            Individual: The modified individual
        """
        individual1, individual2 = individual2
        new_values = {}
        for k, v in individual1:
            if k in individual2:
                keep = (torch.rand_like(v) < self.keep_p).type_as(v)
                new_values[k] = keep * v + (1 - keep) * individual2[k]

        return Individual(**{new_values})

    def spawn(self) -> "KeepMixer":
        return KeepMixer(self.keep_p)


class StandardPopulationMixer(PopulationMixer):

    @abstractmethod
    def mix(self, key: str, val1: torch.Tensor, val2: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, population1: Population, population2: Population) -> Population:

        results = {}
        for k, v in population1:
            results[k] = self.mix(k, v, population2[k])

        return Population(**results)


class KBestElitism(PopulationMixer):
    """Add the k best from the previous generation to the new generation
    """

    def __init__(self, k: int):
        """initializer

        Args:
            k (int): The number to keep
        """
        if k <= 0:
            raise ValueError(f'Argument k must be greater than 0 not {self.k}')
        self.k = k

    def __call__(self, population1: Population, population2: Population) -> Population:
        """

        Args:
            population1 (Population): previous generation
            population2 (Population): new generation

        Returns:
            Population: the updated new generation
        """
        
        _, indices = population1.stack_assessments().reduce('samplemeans').topk(self.k)
        results = {}
        for k, v in population1:
            if k in population2:
                results[k] = torch.cat(
                    [v[indices], population2[k]]
                )

        return Population(**results)


class BinaryRandCrossOverBreeder(StandardPopulationMixer):
    """Mix two tensors together by choosing one gene for each
    """

    def mix(self, key: str, val1: torch.Tensor, val2: torch.Tensor) -> torch.Tensor:
        """Mix two tensors together by choosing one gene for each

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix
    
        Returns:
            torch.Tensor: The mixed result
        """
        to_choose = (torch.rand_like(val1) > 0.5)
        return val1 * to_choose.type_as(val1) + val2 * (~to_choose).type_as(val2)


class SmoothCrossOverBreeder(StandardPopulationMixer):
    """Do a smooth interpolation between the values to breed
    """

    def mix(self, key: str, val1: torch.Tensor, val2: torch.Tensor) -> torch.Tensor:
        """Mix two tensors together with smooth interpolation

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix

        Returns:
            torch.Tensor: The mixed result
        """
        degree = torch.rand_like(val1)
        return val1 * degree + val2 * (1 - degree)

