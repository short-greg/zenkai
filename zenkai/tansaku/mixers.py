# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch

# local
from .core import Population, Individual
from ..kaku import State


class IndividualMixer(ABC):
    """Mixes two individuals together"""

    @abstractmethod
    def mix(self, individual1: Individual, individual2: Individual, state: State) -> Individual:
        pass

    def __call__(self, individual1: Individual, individual2: Individual, state: State=None) -> Individual:
        return self.mix(individual1, individual2, state or State())

    @abstractmethod
    def spawn(self) -> "IndividualMixer":
        pass


class PopulationMixer(ABC):
    """Mixes two populations together"""

    @abstractmethod
    def mix(self, population1: Population, population2: Population, state: State) -> Population:
        pass

    def __call__(self, population1: Population, population2: Population, state: State=None) -> Population:
        return self.mix(
            population1, population2, state or State()
        )

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
        if not (0 <= keep_p <= 1.0):
            raise ValueError(f"{keep_p} must be between 0 and 1.")
        self.keep_p = keep_p

    def mix(self, individual1: Individual, individual2: Individual, state: State) -> Individual:
        """Randomly choose whether to select original or selection for each value

        Args:
            individual (Individual): The individual to modify
            individual2 (Population): The population to modify based on

        Returns:
            Individual: The modified individual
        """
        new_values = {}
        for k, v in individual1.items():
            if k in individual2:
                keep = (torch.rand_like(v) < self.keep_p).type_as(v)
                new_values[k] = keep * v + (1 - keep) * individual2[k]

        return Individual(**new_values)

    def spawn(self) -> "KeepMixer":
        return KeepMixer(self.keep_p)


class StandardPopulationMixer(PopulationMixer):

    @abstractmethod
    def mix_field(self, key: str, val1: torch.Tensor, val2: torch.Tensor, state: State) -> torch.Tensor:
        pass

    def mix(self, population1: Population, population2: Population, state: State) -> Population:

        results = {}
        for k, v in population1.items():
            results[k] = self.mix_field(k, v, population2[k], state)

        return Population(**results)

# from ..kaku import TopKSelector

# def kbest_elitism(old_population, new_population, k, divide_start, state):
   
#     selector = TopKSelector(k=k, dim=0)
#     index_map = selector(old_population.stack_assessments())
#     selection = old_population.select_by(index_map)
#     return selection.join(new_population)

    # for k, x1, x2 in old_population.connect(new_population):


class KBestElitism(PopulationMixer):
    """Add the k best from the previous generation to the new generation
    """

    def __init__(self, k: int, divide_start: int=1):
        """initializer

        Args:
            k (int): The number to keep
        """
        if k <= 0:
            raise ValueError(f'Argument k must be greater than 0 not {self.k}')
        self.k = k
        self.divide_start = divide_start

    def mix(self, population1: Population, population2: Population, state: State) -> Population:
        """

        Args:
            population1 (Population): previous generation
            population2 (Population): new generation

        Returns:
            Population: the updated new generation
        """
        # 
        # assessemnt = assessmetn.reduce_image(2, 'mean') DONE
        # selector = ProbSelector(dim=2)
        # selector = TopKSelector(k=2, dim=0)
        # index_map = selector(assessment)
        # select = population.select_index(index_map)
        # selected = index_map(features)


        
        assessment = population1.stack_assessments().reduce_image(self.divide_start)


        _, indices = assessment.value.topk(self.k, largest=assessment.maximize)
        results = {}
        for k, v in population1.items():
            if k in population2:
                results[k] = torch.cat(
                    [v[indices], population2[k]]
                )

        return Population(**results)
    
    def spawn(self) -> 'KBestElitism':
        return KBestElitism(self.k)


class BinaryRandCrossOverBreeder(StandardPopulationMixer):
    """Mix two tensors together by choosing one gene for each
    """

    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p

    def mix_field(self, key: str, val1: torch.Tensor, val2: torch.Tensor, state: State) -> torch.Tensor:
        """Mix two tensors together by choosing one gene for each

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix
    
        Returns:
            torch.Tensor: The mixed result
        """
        to_choose = (torch.rand_like(val1) > self.p)
        return val1 * to_choose.type_as(val1) + val2 * (~to_choose).type_as(val2)

    def spawn(self) -> 'BinaryRandCrossOverBreeder':
        return BinaryRandCrossOverBreeder(self.p)


class SmoothCrossOverBreeder(StandardPopulationMixer):
    """Do a smooth interpolation between the values to breed
    """

    def mix_field(self, key: str, val1: torch.Tensor, val2: torch.Tensor, state: State) -> torch.Tensor:
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
    
    def spawn(self) -> 'SmoothCrossOverBreeder':
        return SmoothCrossOverBreeder()
