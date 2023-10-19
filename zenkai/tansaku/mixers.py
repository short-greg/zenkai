# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch

# local
from .core import Population, Individual, TensorDict
from ..kaku import State


def keep_mixer(original: TensorDict, updated: TensorDict, keep_p: float) -> typing.Union[Population, Individual]:
    new_values = {}
    for k, original_v, updated_v in original.loop_over(updated, union=False):
        keep = (torch.rand_like(original_v) < keep_p).type_as(original_v)
        new_values[k] = keep * original_v + (1 - keep) * updated_v

    return original.__class__(**new_values)


class Elitism(ABC):

    @abstractmethod
    def __call__(self, population1: Population, population2: Population) -> Population:
        pass


class KBestElitism(Elitism):
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

    def __call__(self, population1: Population, population2: Population) -> Population:
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
        for k, v1, v2 in population1.loop_over(population2, only_my_k=True, union=False):
            results[k] = torch.cat(
                [v1[indices], v2]
            )

        return Population(**results)
    
    def spawn(self) -> 'KBestElitism':
        return KBestElitism(self.k)


class CrossOver(ABC):

    @abstractmethod
    def __call__(self, parents1: Population, parents2: Population) -> Population:
        pass


class BinaryRandCrossOver(CrossOver):
    """Mix two tensors together by choosing one gene for each
    """

    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p

    def __call__(self, parents1: Population, parents2: Population) -> Population:
        """Mix two tensors together by choosing one gene for each

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix
    
        Returns:
            torch.Tensor: The mixed result
        """
        result = {}
        for k, p1, p2 in parents1.loop_over(parents2, only_my_k=True, union=False):
            to_choose = (torch.rand_like(p1) > self.p)
            result[k] = p1 * to_choose.type_as(p1) + p2 * (~to_choose).type_as(p2)
        return Population(**result)

    def spawn(self) -> 'BinaryRandCrossOver':
        return BinaryRandCrossOver(self.p)


class SmoothCrossOver(CrossOver):
    """Do a smooth interpolation between the values to breed
    """


    def __call__(self, parents1: Population, parents2: Population) -> Population:
        """Mix two tensors together by choosing one gene for each

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix
    
        Returns:
            torch.Tensor: The mixed result
        """
        result = {}
        for k, p1, p2 in parents1.loop_over(parents2, only_my_k=True, union=False):
            degree = torch.rand_like(p1)
            result[k] = p1 * degree + p2 * (1 - degree)
        return Population(**result)
    
    def spawn(self) -> 'SmoothCrossOver':
        return SmoothCrossOver()



# class IndividualMixer(ABC):
#     """Mixes two individuals together"""

#     @abstractmethod
#     def mix(self, individual1: Individual, individual2: Individual, state: State) -> Individual:
#         pass

#     def __call__(self, individual1: Individual, individual2: Individual, state: State=None) -> Individual:
#         return self.mix(individual1, individual2, state or State())

#     @abstractmethod
#     def spawn(self) -> "IndividualMixer":
#         pass


# class PopulationMixer(ABC):
#     """Mixes two populations together"""

#     @abstractmethod
#     def mix(self, population1: Population, population2: Population, state: State) -> Population:
#         pass

#     def __call__(self, population1: Population, population2: Population, state: State=None) -> Population:
#         return self.mix(
#             population1, population2, state or State()
#         )

#     @abstractmethod
#     def spawn(self) -> "PopulationMixer":
#         pass


# class StandardPopulationMixer(PopulationMixer):

#     @abstractmethod
#     def mix_field(self, key: str, val1: torch.Tensor, val2: torch.Tensor, state: State) -> torch.Tensor:
#         pass

#     def mix(self, population1: Population, population2: Population, state: State) -> Population:

#         results = {}
#         for k, v in population1.items():
#             results[k] = self.mix_field(k, v, population2[k], state)

#         return Population(**results)

# from ..kaku import TopKSelector

# def kbest_elitism(old_population, new_population, k, divide_start, state):
   
#     selector = TopKSelector(k=k, dim=0)
#     index_map = selector(old_population.stack_assessments())
#     selection = old_population.select_by(index_map)
#     return selection.join(new_population)

    # for k, x1, x2 in old_population.connect(new_population):