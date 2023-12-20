from abc import ABC, abstractmethod

import torch
from ..kaku import Population
from ._select import BestSelector
import math
from ..utils import unsqueeze_to


class ParticleUpdater(ABC):

    @abstractmethod
    def update(self, population: Population):
        pass

    @abstractmethod
    def __call__(self, population: Population) -> Population:
        pass


class GaussianParticleUpdater(ParticleUpdater):

    sqrt2 = math.sqrt(2)
    
    def __init__(self, weight: float=0.01):

        self._mean = None
        self._var = None
        self._weight = weight
        self._cur = None

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def var(self) -> torch.Tensor:
        return self._var
    
    def update(self, population: Population):

        value = population.assessment.value

        if self._mean is None:
            self._mean = value
        elif self._var is None:
            self._var = (value - self._mean) ** 2
            self._mean = (1 - self._weight) * self._mean + self._weight * value
        else:
            
            self._mean = (1 - self._weight) * self._mean + self._weight * value
            self._var = (1 - self._weight ) * self._var + self._weight * (value - self._mean) ** 2

    def cdf(self, x: Population) -> Population:

        if self._var is None:
            return 1.0

        result = {}
        for k, v in x.items():
            mean = unsqueeze_to(self._mean, v)
            var = unsqueeze_to(self._var, v)
            result[k] = 0.5 * (
                1 + (v - mean) / (torch.sqrt(var) * self.sqrt2)
            )
        return Population(**result)

    def __call__(self, population: Population) -> Population:
        
        if self._cur is None:
            self._cur = population
            return self._cur
        
        cur_weight = self.cdf(self._cur)
        new_weight = self.cdf(population)
        self._cur = cur_weight * self._cur + new_weight * population
        return self._cur


class GlobalBest(object):

    def __init__(self, updater: ParticleUpdater, dim: int=0):
        """

        Args:
            updater (ParticleUpdater): 
            dim (int, optional): . Defaults to 0.
        """
        self._selector = BestSelector(dim)
        self._updater = updater

    def __call__(self, population: Population) -> Population:
        """

        Args:
            population (Population): 

        Returns:
            Population: 
        """
        best = self._selector(population)
        self._updater.update(best)
        return self._updater(best)


class LocalBest(object):

    def __init__(self, updater: ParticleUpdater):
        """

        Args:
            updater (ParticleUpdater): 
        """
        self._updater = updater

    def __call__(self, population: Population) -> Population:
        """

        Args:
            population (Population): 

        Returns:
            Population: 
        """
        self._updater.update(population)
        return self._updater(population)


# class GlobalParticleSmooth(GlobalParticle):
#     """Mix two tensors together by choosing one gene for each"""

#     def __init__(self, dim: int = 0):
#         super().__init__()
#         self.dim = dim

#     def _select(self):
#         pass

#     def _calc_weight(self):
#         pass

#     def _update_best(self):
#         pass

#     def __init_state__(self, state: State):

#         state.get_or_set((self, 'dist'), None)
#         state.get_or_set((self, 'global_value'), None)
#         state.get_or_set((self, 'global_best'), None)

#     def __call__(self, population: Population, state: State) -> Individual:
#         """Mix two tensors together by choosing one gene for each

#         Args:
#             key (str): The name of the field
#             val1 (torch.Tensor): The first value to mix
#             val2 (torch.Tensor): The second value to mix

#         Returns:
#             torch.Tensor: The mixed result
#         """

#         assessment = population.stack_assessments()
#         if self.dim > 1:
#             assessment = assessment.reduce_image(self.dim)
#         self.__init_state__(state)
#         my_state = state.mine(self)
#         # TODO: Break it down into steps

#         # self._select()
#         # if self._dist is None:
#         #     global_best = self._init_best()
#         # else:
#         #     weight = self._calc_weight()
#         #     global_best = self._update_best(weight, state)
#         #     self._update_dist()

#         selector = _select.BestSelector(self.dim)
#         value = assessment.value.mean(dim=0)
#         index_map = selector.select(assessment)
#         best_value = index_map.assessment
#         selected = index_map.select_index(population)

#         if my_state.dist is None:
#             my_state.dist = torch.distributions.Normal(value, torch.ones_like(value))
#             my_state.global_best = selected
#             my_state.global_value = value
#         else:
#             # calculate the new global best
#             cur_weight = my_state.dist.cdf(best_value.value) ** 4
#             global_weight = my_state.dist.cdf(my_state.global_value) ** 4
#             print('Shape: ', my_state.global_value)
#             cur_weight = cur_weight / (cur_weight + global_weight)
#             if assessment.maximize:
#                 global_weight = 1 - cur_weight
#             else:
#                 global_weight = cur_weight
#                 cur_weight = 1 - cur_weight

#             # TODO: This won't work because the dimensions are not correct
#             for k, v1, v2 in population.loop_over(my_state.global_best):
#                 cur_weight_k = utils.unsqueeze_to(cur_weight, v1)
#                 global_weight_k = utils.unsqueeze_to(global_weight, v1)

#                 print((cur_weight_k * v1).shape, (global_weight_k * v2).shape)
#                 my_state.global_best[k] = (
#                     cur_weight_k * v1 + global_weight_k * v2
#                 )
#             my_state.global_best = global_weight * my_state.global_best + cur_weight * population
#             my_state.global_value = global_weight * my_state.global_value + cur_weight * value

#             # update the distribution
#             my_state.dist = torch.distributions.Normal(
#                 0.95 * my_state.dist.mean + 0.05 * value,
#                 0.95 * my_state.dist.scale
#                 + 0.05 * (value.mean(dim=0) - my_state.dist.mean) ** 2,
#             )

#         return my_state.global_best.get_i(0)

#     def spawn(self) -> "GlobalParticleSmooth":
#         return GlobalParticleSmooth(self.dim)


# class LocalParticleSmooth(LocalParticle):
#     """Mix two tensors together by choosing one gene for each"""

#     def __init__(self, dim: int = 0):
#         super().__init__()
#         self.dim = dim
#         self._dist = None
#         self._local_best = None

#     def __call__(self, population: Population, state: State) -> Individual:
#         """Mix two tensors together by choosing one gene for each

#         Args:
#             key (str): The name of the field
#             val1 (torch.Tensor): The first value to mix
#             val2 (torch.Tensor): The second value to mix

#         Returns:
#             torch.Tensor: The mixed result
#         """

#         # TODO: Break it down into steps

#         # self._select()
#         # if self._dist is None:
#         #     global_best = self._init_best()
#         #     self._init_dist()
#         # else:
#         #     weight = self._calc_weight()
#         #     local_best = self._update_best(weight, state)
#         #     self._update_dist()

#         assessment = population.stack_assessments().reduce_image(self.dim)
#         value = assessment.value

#         if self._dist is None:
#             self._local_best = population
#             self._local_value = value
#             self._dist = torch.distributions.Normal(value, torch.ones_like(value))
#         else:
#             cur_p = self._dist.cdf(value) ** 4
#             global_p = self._dist.cdf(self._local_value) ** 4

#             cur_weight = cur_p / (cur_p + global_p)

#             if assessment.maximize:
#                 best_weight = cur_weight
#                 cur_weight = 1 - cur_weight
#             else:
#                 best_weight = 1 - cur_weight

#             # TODO: this will not work yes because the dimension is not right

#             for k, v1, v2 in population.loop_over(self._local_best):
#                 cur_weight_k = utils.unsqueeze_to(cur_weight, v1)
#                 best_weight_k = utils.unsqueeze_to(best_weight, v1)

#                 self._local_best[k] = cur_weight_k * v1 + best_weight_k * v2
#             self._local_value = cur_weight * value + best_weight * self._local_value
#             weight = best_weight * 0.95
#             self._dist = torch.distributions.Normal(
#                 weight * self._dist.mean + (1 - weight) * population,
#                 weight * self._dist.mean
#                 + (1 - weight) * (population - self._dist.mean) ** 2,
#             )
#         return self._local_best

#     def spawn(self) -> "LocalParticleSmooth":
#         return LocalParticleSmooth(self.dim)
