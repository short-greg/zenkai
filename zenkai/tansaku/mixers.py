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