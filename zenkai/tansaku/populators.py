# 1st part
import math
import typing
from abc import ABC, abstractmethod

# 3rd party
import torch
from torch.nn.parameter import Parameter

# local
from ..kaku import Assessor
from .core import (
    Individual,
    Population,
    binary_prob,
    cat_params,
    deflatten,
    expand,
    flatten,
)


class Populator(ABC):
    """Base class for creating a population from an individual"""

    @abstractmethod
    def __call__(self, individual: Individual) -> Population:
        """Spawn a population from an individual

        Args:
            individual (Individual): The individual to populate based on

        Returns:
            Population: The resulting population
        """
        pass

    @abstractmethod
    def spawn(self) -> "Populator":
        """Spawn a new populator from the current populator

        Returns:
            Populator: The spawned populator
        """
        pass


class StandardPopulator(Populator):
    """Populator that uses a standard populator method for all values in the individual"""

    @abstractmethod
    def populate(
        self, key: str, val: typing.Union[torch.Tensor, Parameter]
    ) -> typing.Union[torch.Tensor, Parameter]:
        pass

    def __call__(self, individual: Individual) -> Population:
        """Call the populate method for each value and spawn the population

        Args:
            individual (Individual): The individual to populate based no

        Returns:
            Population: The resulting population
        """
        expanded = {}
        for key, val in individual:
            cur = self.populate(key, val)
            if cur is not None:
                expanded[key] = cur
        return Population(**expanded)


class PopulatorDecorator(Populator):
    """Decorate the results of a populator"""

    def __init__(self, base_populator: Populator):
        """initializer

        Args:
            base_populator (Populator): The populator to decorate
        """
        self.base_populator = base_populator

    @abstractmethod
    def decorate(
        self, key: str, base_val, val: typing.Union[torch.Tensor, Parameter]
    ) -> typing.Union[torch.Tensor, Parameter]:
        """Decorate each value in the population

        Args:
            key (str): the key for the value in the dictionary
            base_val (): the value before populating
            val (typing.Union[torch.Tensor, Parameter]): The value after populating

        Returns:
            typing.Union[torch.Tensor, Parameter]: The result of the decoration
        """
        pass

    def __call__(self, individual: Individual) -> Population:
        """Spawn a population from an individual

        Args:
            individual (Individual): The individual to populate based on

        Returns:
            Population: The resulting population
        """
        populated = self.base_populator(individual)
        expanded = {}
        for key, val in populated:
            if key in individual:
                expanded[key] = self.decorate(key, individual[key], val)

        return Population(**expanded)

    @abstractmethod
    def spawn(self) -> "PopulatorDecorator":
        pass


class RepeatPopulator(StandardPopulator):
    """Populator that outputs all the same values for the population dimension"""

    def __init__(self, k: int):
        """initializer

        Args:
            k (int): The size of the population
        """
        self.k = k

    def populate(
        self, key: str, val: typing.Union[torch.Tensor, Parameter]
    ) -> typing.Union[torch.Tensor, Parameter]:
        """Expands each of the values by repeating along the population dimension

        Args:
            key (str): The name of the value
            val (typing.Union[torch.Tensor, Parameter]): the value to repeat

        Returns:
            typing.Union[torch.Tensor, Parameter]: The expanded value
        """
        return expand(val, self.k)

    def spawn(self) -> "RepeatPopulator":
        return RepeatPopulator(self.k)


def populate_t(t: torch.Tensor, k: int, name: str = "t") -> Population:
    """Convenience function to expand the t dimension along the population dimension

    Args:
        t (torch.Tensor): the tensor to expand
        k (int): the size of the population
        name (str, optional): the name of the value. Defaults to "t".

    Returns:
        Population: The result of the expansion
    """

    t = Individual(**{name: t})
    populator = RepeatPopulator(k)
    return populator(t)


class SimpleGaussianPopulator(StandardPopulator):
    """initializer"""

    def __init__(self, k: int):
        """initializer

        Args:
            k (int): The size of the population
        """
        self.k = k

    def populate(
        self, key: str, val: typing.Union[torch.Tensor, Parameter]
    ) -> typing.Union[torch.Tensor, Parameter]:
        """

        Args:
            key (str): The key for the element
            val (typing.Union[torch.Tensor, Parameter]): The element to create the population for

        Returns:
            typing.Union[torch.Tensor, Parameter]: The element expanded along the dimension
        """

        perturbation = torch.randn(self.k - 1, *val.shape, device=val.device)
        val = torch.cat([val[None], val[None] + perturbation])

    def spawn(self) -> "SimpleGaussianPopulator":
        return SimpleGaussianPopulator(self.k)


class GaussianPopulator(StandardPopulator):
    def __init__(self, k: int, std: float = 1, equal_change_dim: int = None):
        self.k = k
        self.std = std
        self.equal_change_dim = equal_change_dim

    def populate(self, key: str, val: torch.Tensor):

        shape = [self.k - 1, *val.shape]
        if self.equal_change_dim:
            shape[self.equal_change_dim] = 1
        noise = torch.randn(*shape, device=val.device) * self.std

        return torch.cat([val[None], val[None] + noise])

    def spawn(self) -> "GaussianPopulator":
        return GaussianPopulator(self.k, self.std, self.equal_change_dim)


class BinaryPopulator(StandardPopulator):
    def __init__(
        self,
        k: int = 1,
        keep_p: float = 0.1,
        equal_change_dim: int = None,
        to_change: typing.Union[int, float] = None,
        reorder_params: bool = True,
        zero_neg: bool = False,
    ):
        if 0.0 >= keep_p or 1.0 < keep_p:
            raise RuntimeError("Argument p must be in range (0.0, 1.0] not {keep_p}")
        assert k > 1
        self.keep_p = keep_p
        self.k = k
        self._equal_change_dim = equal_change_dim
        self._is_percent_change = isinstance(to_change, float)
        if self._is_percent_change:
            assert 0 < to_change <= 1.0
        elif to_change is not None:
            assert to_change > 0
        self._to_change = to_change
        self._reorder_params = reorder_params
        self._zero_neg = zero_neg

    # TODO: Move this to a "PopulationModifier" or a Decorator
    def _generate_keep(self, param: torch.Tensor):

        shape = [self.k - 1, *param.shape]
        if self._equal_change_dim is not None:
            shape[self._equal_change_dim] = 1

        param = (param > 0.0).type_as(param)
        keep = (torch.rand(*shape, device=param.device) < self.keep_p).type(param.dtype)

        if self._to_change is None:
            return keep

        if self._is_percent_change:
            ignore_change = (
                torch.rand(1, 1, *param.shape[1:], device=param.device)
                > self._to_change
            ).type_as(param)
        else:
            _, indices = torch.rand(
                math.prod(param.shape[1:]), device=param.device
            ).topk(self._to_change, dim=-1)
            ignore_change = torch.ones(math.prod(param.shape[1:]), device=param.device)
            ignore_change[indices] = 0.0
            ignore_change = ignore_change.view(1, 1, *param.shape[1:])

        return torch.max(keep, ignore_change)

    def populate(self, key: str, val: torch.Tensor):

        keep = self._generate_keep(val)

        changed = -val[None] if not self._zero_neg else (1 - val[None])
        perturbed_params = keep * val[None] + (1 - keep) * changed
        concatenated = cat_params(val, perturbed_params, reorder=True)
        if not self._reorder_params:
            return concatenated
        reordered = torch.randperm(len(concatenated), device=concatenated.device)
        return concatenated[reordered]

    def spawn(self) -> "BinaryPopulator":
        return BinaryPopulator(
            self.k,
            self.keep_p,
            self._equal_change_dim,
            self._to_change,
            self._reorder_params,
            self._zero_neg,
        )


class ConservativePopulator(PopulatorDecorator):
    def __init__(
        self,
        base_populator: Populator,
        percent_change: float = 0.1,
        same_change: bool = True,
    ):
        super().__init__(base_populator)
        if not (0.0 <= percent_change <= 1.0):
            raise ValueError("Percent change must be between 0 and 1")
        self.percent_change = percent_change
        self.same_change = same_change

    def decorate(
        self,
        key: str,
        base_val: torch.Tensor,
        val: typing.Union[torch.Tensor, Parameter],
    ) -> typing.Union[torch.Tensor, Parameter]:

        base_val = base_val[None]
        size = list(val.size())
        if self.same_change:
            size[0] = 1

        to_change = (
            torch.rand(*size, device=val.device) < self.percent_change
        ).type_as(val)
        return to_change * val + (1 - to_change) * base_val

    def spawn(self) -> "ConservativePopulator":
        return ConservativePopulator(
            self.base_populator.spawn(), self.percent_change, self.same_change
        )


class PerceptronProbPopulator(Populator):
    def __init__(
        self, learner: Assessor, k: int, x: str = "x", unactivated: str = "unactivated"
    ):
        self.k = k
        self.learner = learner
        self.unactivated = unactivated
        self.x = x

    def populate(
        self, key: str, val: typing.Union[torch.Tensor, Parameter]
    ) -> typing.Union[torch.Tensor, Parameter]:

        if key != self.unactivated:
            return None

        x_ = torch.clamp((val + 1) / 2, 0.001, 0.999)
        return (x_[None] < torch.rand(self.k, *x_.shape, device=x_.device)).type_as(
            x_
        ) * 2 - 1


class BinaryProbPopulator(Populator):
    def __init__(
        self,
        learner: Assessor,
        k: int,
        zero_neg: bool = True,
        loss_name: str = "loss",
        x: str = "x",
        t: str = "t",
    ):
        self.learner = learner
        self.k = k
        self.zero_neg = zero_neg
        self.loss_name = loss_name
        self.x = x
        self.t = t

    def generate_sample(
        self,
        base_size: torch.Size,
        dtype: torch.dtype,
        device=torch.device,
        prob: torch.Tensor = None,
    ):

        prob = prob or 0.5
        sample = torch.rand(self.k, *base_size, dtype=dtype, device=device)

        sample = (sample > prob).type_as(sample)
        if not self.zero_neg:
            sample = sample * 2 - 1
        return sample

    def __call__(self, x: Individual) -> Population:

        sample = self.generate_sample(x.size(), x.dtype, x.device)
        t = x[self.t]
        t = expand(t[0], self.k)
        sample = flatten(sample)
        assessment = self.learner.assess(sample, t, "none")["loss"]
        value = assessment.value[:, None]
        sample = sample.unsqueeze(sample.dim())
        value = deflatten(value, self.k)
        sample = deflatten(sample, self.k)
        prob = binary_prob(sample, value)
        sample = self.generate_sample(x.size(), x.dtype, x.device, prob)
        return Population(x=sample)

    def spawn(self) -> "BinaryProbPopulator":
        return BinaryProbPopulator(
            self.learner, self.k, self.zero_neg, self.loss_name, self.x, self.t
        )


class PopulationLimiter(object):
    def __call__(
        self,
        individual: Individual,
        population: Population,
        limit: torch.LongTensor = None,
    ) -> Population:

        if limit is None:
            return population

        result = {}

        for k, v in population:
            individual_v = individual[k][None]
            individual_v = individual_v.repeat(v.size(0), 1, 1)
            individual_v[:, :, limit] = v[:, :, limit].detach()
            result[k] = individual_v
        return Population(**result)
