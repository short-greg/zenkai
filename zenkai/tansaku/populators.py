# 1st part
import typing
from abc import ABC, abstractmethod
import math

# 3rd party
import torch
from torch.nn.parameter import Parameter

# local
from .core import (
    Individual,
    Population,
    binary_prob,
    cat_params,
)
from ..kaku import State
from ..utils import Voter, expand_dim0


class Populator(ABC):
    """Base class for creating a population from an individual"""


    @abstractmethod
    def populate(self, individual: Individual, state: State) -> Population:
        """Spawn a population from an individual

        Args:
            individual (Individual): The individual to populate based on

        Returns:
            Population: The resulting population
        """
        pass

    def __call__(self, individual: Individual, state: State=None) -> Population:
        """Spawn a population from an individual

        Args:
            individual (Individual): The individual to populate based on

        Returns:
            Population: The resulting population
        """
        return self.populate(
            individual, state or State()
        )

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
    def populate_field(
        self, key: str, val: typing.Union[torch.Tensor, Parameter], state: State
    ) -> typing.Union[torch.Tensor, Parameter]:
        pass

    def populate(self, individual: Individual, state: State) -> Population:
        """Call the populate method for each value and spawn the population

        Args:
            individual (Individual): The individual to populate based no

        Returns:
            Population: The resulting population
        """
        expanded = {}
        for key, val in individual.items():
            cur = self.populate_field(key, val, state)
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
        self, key: str, base_val, val: typing.Union[torch.Tensor, Parameter], state: State
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

    def populate(self, individual: Individual, state: State) -> Population:
        """Spawn a population from an individual

        Args:
            individual (Individual): The individual to populate based on

        Returns:
            Population: The resulting population
        """
        populated = self.base_populator(individual)
        expanded = {}
        for key, val in populated.items():
            if key in individual:
                expanded[key] = self.decorate(key, individual[key], val, state)

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

    def populate_field(
        self, key: str, val: typing.Union[torch.Tensor, Parameter], state: State
    ) -> typing.Union[torch.Tensor, Parameter]:
        """Expands each of the values by repeating along the population dimension

        Args:
            key (str): The name of the value
            val (typing.Union[torch.Tensor, Parameter]): the value to repeat

        Returns:
            typing.Union[torch.Tensor, Parameter]: The expanded value
        """
        return expand_dim0(val, self.k, False)

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


class VoterPopulator(Populator):
    """Populator that uses multiple outputs from votes
    """
    
    def __init__(self, voter: Voter, x_name: str):
        """initializer

        Args:
            voter (Voter): the module to use for voting
            x_name (str): the name of the input into x
        """
        self.voter = voter
        self.x_name = x_name

    def populate(self, individual: Individual, state: State) -> Population:
        """Populator function

        Args:
            individual (Individual): the individual to populate based on

        Returns:
            Population: The resulting population
        """
        # x = expand(individual[self.x_name], self.k)
        x = individual[self.x_name]
        y = self.voter(x)
        result = {self.x_name: y}
        return Population(**result)
    
    def spawn(self) -> Populator:
        return VoterPopulator(
            self.voter, self.x_name
        )


class SimpleGaussianPopulator(StandardPopulator):
    """initializer"""

    def __init__(self, k: int):
        """initializer

        Args:
            k (int): The size of the population
        """
        self.k = k

    def populate_field(
        self, key: str, val: typing.Union[torch.Tensor, Parameter], state: State
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


# TODO: Why not just use the mutator? 
# 1) populate -> mutate
class GaussianPopulator(StandardPopulator):
    """Create a population using Gaussian noise on the individual
    """

    def __init__(self, k: int, std: float = 1, equal_change_dim: int = None):
        self.k = k
        self.std = std
        self.equal_change_dim = equal_change_dim

    def populate_field(self, key: str, val: torch.Tensor, state: State):

        shape = [self.k - 1, *val.shape]
        if self.equal_change_dim:
            shape[self.equal_change_dim] = 1
        noise = torch.randn(*shape, device=val.device) * self.std

        return torch.cat([val[None], val[None] + noise])

    def spawn(self) -> "GaussianPopulator":
        return GaussianPopulator(self.k, self.std, self.equal_change_dim)


class ConservativePopulator(PopulatorDecorator):
    """Decorator for a populator that replaces the initial results of the populator
    algorithm with the original values
    """

    def __init__(
        self,
        base_populator: Populator,
        percent_change: float = 0.1,
        same_change: bool = True,
    ):
        """initializer

        Args:
            base_populator (Populator): The populator decorated
            percent_change (float, optional): The percentage to change the population. Defaults to 0.1.
            same_change (bool, optional): Whether the same elements should be 'conserved' for the entire population. Defaults to True.

        Raises:
            ValueError: If the percent change is less than 0 or greater than 1s
        """
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
        state: State
    ) -> typing.Union[torch.Tensor, Parameter]:
        """Decorate the population

        Args:
            key (str): The name of the item
            base_val (torch.Tensor): The original value
            val (typing.Union[torch.Tensor, Parameter]): The updated value after 'populating'

        Returns:
            typing.Union[torch.Tensor, Parameter]: The decorated value
        """
        base_val = base_val[None]
        size = list(val.size())
        if self.same_change:
            size[0] = 1

        to_change = (
            torch.rand(*size, device=val.device) < self.percent_change
        ).type_as(val)
        return to_change * val + (1 - to_change) * base_val

    def spawn(self) -> "ConservativePopulator":
        """
        Returns:
            ConservativePopulator: A new conservative spawner with the same parameters
        """
        return ConservativePopulator(
            self.base_populator.spawn(), self.percent_change, self.same_change
        )


class BinaryPopulator(StandardPopulator):
    """
    """

    def __init__(
        self,
        k: int = 1,
        keep_p: float = 0.1,
        equal_change_dim: int = None,
        to_change: typing.Union[int, float] = None,
        reorder_params: bool = True,
        zero_neg: bool = False,
    ):
        """initializer

        Args:
            k (int, optional): The population size. Defaults to 1.
            keep_p (float, optional): Probability of keeping the current value. Defaults to 0.1.
            equal_change_dim (int, optional): Whether to change all values in an individual the same. Defaults to None.
            to_change (typing.Union[int, float], optional): the number of elements to change. Defaults to None.
            reorder_params (bool, optional): . Defaults to True.
            zero_neg (bool, optional): whether the negative is 0 or -1. Defaults to False.

        Raises:
            RuntimeError: If the probability of keeping p is not valid
        """
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

    def populate_field(self, key: str, val: torch.Tensor, state: State):

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
