# TODO: Probably remove

# from abc import ABC, abstractmethod

# import torch
# from ..kaku import Population
# from ._select import BestSelector
# import math
# from ..utils import unsqueeze_to


# # TODO: Consider more carefully how to do this and make it more extensible


# class ParticleUpdater(ABC):

#     @abstractmethod
#     def update(self, population: Population):
#         pass

#     @abstractmethod
#     def __call__(self, population: Population) -> Population:
#         pass


# class GaussianParticleUpdater(ParticleUpdater):
#     """
#     """

#     sqrt2 = math.sqrt(2)
    
#     def __init__(self, weight: float=0.01):

#         self._mean = None
#         self._var = None
#         self._weight = weight
#         self._cur = None

#     @property
#     def mean(self) -> torch.Tensor:
#         return self._mean

#     @property
#     def var(self) -> torch.Tensor:
#         return self._var
    
#     def update(self, population: Population):

#         value = population.assessment.value

#         if self._mean is None:
#             self._mean = value
#         elif self._var is None:
#             self._var = (value - self._mean) ** 2
#             self._mean = (1 - self._weight) * self._mean + self._weight * value
#         else:
            
#             self._mean = (1 - self._weight) * self._mean + self._weight * value
#             self._var = (1 - self._weight ) * self._var + self._weight * (value - self._mean) ** 2

#     def cdf(self, x: Population) -> Population:

#         if self._var is None:
#             return 1.0

#         result = {}
#         for k, v in x.items():
#             mean = unsqueeze_to(self._mean, v)
#             var = unsqueeze_to(self._var, v)
#             result[k] = 0.5 * (
#                 1 + (v - mean) / (torch.sqrt(var) * self.sqrt2)
#             )
#         return Population(**result)

#     def __call__(self, population: Population) -> Population:
        
#         if self._cur is None:
#             self._cur = population
#             return self._cur
        
#         cur_weight = self.cdf(self._cur)
#         new_weight = self.cdf(population)
#         self._cur = cur_weight * self._cur + new_weight * population
#         return self._cur


# class GlobalBest(object):

#     def __init__(self, updater: ParticleUpdater, dim: int=0):
#         """

#         Args:
#             updater (ParticleUpdater): 
#             dim (int, optional): . Defaults to 0.
#         """
#         self._selector = BestSelector(dim)
#         self._updater = updater

#     def __call__(self, population: Population) -> Population:
#         """

#         Args:
#             population (Population): 

#         Returns:
#             Population: 
#         """
#         best = self._selector(population)
#         self._updater.update(best)
#         return self._updater(best)


# class LocalBest(object):

#     def __init__(self, updater: ParticleUpdater):
#         """

#         Args:
#             updater (ParticleUpdater): 
#         """
#         self._updater = updater

#     def __call__(self, population: Population) -> Population:
#         """

#         Args:
#             population (Population): 

#         Returns:
#             Population: 
#         """
#         self._updater.update(population)
#         return self._updater(population)
