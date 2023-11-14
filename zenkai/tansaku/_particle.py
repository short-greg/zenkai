from abc import ABC, abstractmethod

import torch
from ..kaku import Population, Individual, State
from . import _select
from .. import utils


class GlobalParticle(ABC):
    @abstractmethod
    def __call__(self, population: Population) -> Individual:
        pass


class LocalParticle(ABC):
    @abstractmethod
    def __call__(self, population: Population) -> Population:
        pass


class GlobalParticleSmooth(GlobalParticle):
    """Mix two tensors together by choosing one gene for each"""

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
        self._dist = None
        self._global_value = None

    def _select(self):
        pass

    def _calc_weight(self):
        pass

    def _update_best(self):
        pass

    def __call__(self, population: Population, state: State) -> Individual:
        """Mix two tensors together by choosing one gene for each

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix

        Returns:
            torch.Tensor: The mixed result
        """
        assessment = population.stack_assessments().reduce_image(self.dim)

        # TODO: Break it down into steps

        # self._select()
        # if self._dist is None:
        #     global_best = self._init_best()
        # else:
        #     weight = self._calc_weight()
        #     global_best = self._update_best(weight, state)
        #     self._update_dist()

        selector = _select.BestSelector(1, self.dim, assessment.maximize)
        value = assessment.value.mean(dim=0)
        index_map = selector.select(assessment)
        best_value = selector.select_value(assessment)
        selected = index_map.select_index(population)

        if self._dist is None:
            self._dist = torch.distributions.Normal(value, torch.ones_like(value))
            self._global_best = selected
            self._global_value = value
        else:
            # calculate the new global best
            cur_weight = self._dist.cdf(best_value) ** 4
            global_weight = self._dist.cdf(self._global_value) ** 4
            cur_weight = cur_weight / (cur_weight + global_weight)
            if assessment.maximize:
                global_weight = 1 - cur_weight
            else:
                global_weight = cur_weight
                cur_weight = 1 - cur_weight

            # TODO: This won't work because the dimensions are not correct
            for k, v1, v2 in population.loop_over(self._global_best):
                cur_weight_k = utils.unsqueeze_to(cur_weight, v1)
                global_weight_k = utils.unsqueeze_to(global_weight, v1)

                self._global_best[k] = cur_weight_k * v1 + global_weight_k * v2
            self._global_best = global_weight * self._global_best + cur_weight * population
            self._global_value = global_weight * self._global_value + cur_weight * value

            # update the distribution
            self._dist = torch.distributions.Normal(
                0.95 * self._dist.mean + 0.05 * value,
                0.95 * self._dist.scale
                + 0.05 * (value.mean(dim=0) - self._dist.mean) ** 2,
            )

        return self._global_best

    def spawn(self) -> "GlobalParticleSmooth":
        return GlobalParticleSmooth(self.dim)


class LocalParticleSmooth(LocalParticle):
    """Mix two tensors together by choosing one gene for each"""

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
        self._dist = None
        self._local_best = None

    def __call__(self, population: Population, state: State) -> Individual:
        """Mix two tensors together by choosing one gene for each

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix

        Returns:
            torch.Tensor: The mixed result
        """

        # TODO: Break it down into steps

        # self._select()
        # if self._dist is None:
        #     global_best = self._init_best()
        #     self._init_dist()
        # else:
        #     weight = self._calc_weight()
        #     local_best = self._update_best(weight, state)
        #     self._update_dist()

        assessment = population.stack_assessments().reduce_image(self.dim)
        value = assessment.value

        if self._dist is None:
            self._local_best = population
            self._local_value = value
            self._dist = torch.distributions.Normal(value, torch.ones_like(value))
        else:
            cur_p = self._dist.cdf(value) ** 4
            global_p = self._dist.cdf(self._local_value) ** 4

            cur_weight = cur_p / (cur_p + global_p)

            if assessment.maximize:
                best_weight = cur_weight
                cur_weight = 1 - cur_weight
            else:
                best_weight = 1 - cur_weight

            # TODO: this will not work yes because the dimension is not right

            for k, v1, v2 in population.loop_over(self._local_best):
                cur_weight_k = utils.unsqueeze_to(cur_weight, v1)
                best_weight_k = utils.unsqueeze_to(best_weight, v1)

                self._local_best[k] = cur_weight_k * v1 + best_weight_k * v2
            self._local_value = cur_weight * value + best_weight * self._local_value
            weight = best_weight * 0.95
            self._dist = torch.distributions.Normal(
                weight * self._dist.mean + (1 - weight) * population,
                weight * self._dist.mean
                + (1 - weight) * (population - self._dist.mean) ** 2,
            )
        return self._local_best

    def spawn(self) -> "LocalParticleSmooth":
        return LocalParticleSmooth(self.dim)
