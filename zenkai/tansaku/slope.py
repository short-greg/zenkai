# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch


# local
from ..utils import to_signed_neg, to_zero_neg
from .functional import Individual, Population, expand_dim0
from ..kaku import State, Assessment


class SlopeCalculator(object):
    """
    'Calculates' the population to the slope from current evaluation.
    Can be used to add to the value before 'spawning'
    """

    def __init__(self, momentum: float = None):
        if momentum is not None and momentum <= 0.0:
            raise ValueError(
                f"Momentum must be greater or equal to 0 or None, not {momentum}"
            )
        self._momentum = momentum
        self._slope = None
        self._slopes = {}

    def __call__(
        self, population: Population
    ) -> torch.Tensor:
        # TODO: Add in momentum for slope (?)

        slopes = {}
        assessment = population.stack_assessments()
        for k, pop_val in population.items():

            evaluation = assessment.value[:, :, None]
            ssx = (pop_val**2).sum(0) - (1 / len(pop_val)) * (pop_val.sum(0)) ** 2
            ssy = (pop_val * evaluation).sum(0) - (1 / len(pop_val)) * (
                (pop_val.sum(0) * evaluation.sum(0))
            )
            slope = ssy / ssx
            self._slopes[k] = (
                self._slopes[k] * self._momentum + slope
                if k in self._slopes and self._momentum is not None
                else slope
            )
            slopes[k] = self._slopes[k]
        return Population(**slopes)

    def spawn(self) -> "SlopeCalculator":
        return SlopeCalculator(self._momentum)


class SlopeUpdater(object):

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
        self._slope_selector = SlopeCalculator(momentum)
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

    def __call__(self, original: Individual, population: Population) -> Individual:
        x = original[self.x]
        slope = self._slope_selector(population)[self.x]
        return Individual(**{self.x: x + self.lr * slope})

    def spawn(self) -> "SlopeUpdater":
        return SlopeUpdater(self._momentum, self.lr, self.x, self._multiplier == 1)



# class JoinIndividual(PopulationInfluencer):
#     """"""

#     def influence(self, population: Population, individual: Individual, state: State) -> Population:
        
#         new_population = {**population}

#         for k, v in individual:
#             new_population[k] = expand_dim0(v, population.n_individuals)
        
#         return Population(**new_population)

#     def spawn(self) -> "PopulationInfluencer":
#         return JoinIndividual()
    
#     def join_t(self, population: Population, t: torch.Tensor):

#         return self(population, Individual(t=t))

#     def join_tensor(self, population: Population, key: str, val: torch.Tensor):

#         return self(population, Individual(**{key: val}))


# class JoinInfluencer(PopulationInfluencer):
#     """
#     Add the individual to the front of the pouplation
#     """

#     def influence(
#         self,
#         population: Population,
#         individual: Individual,
#         state: State
#     ) -> Population:
#         """

#         Args:
#             population (Population): The population to limit
#             individual (Individual): The individual
#             limit (torch.LongTensor, optional): The index to use to limit. Defaults to None.

#         Returns:
#             Population: The limited population
#         """
#         result = {}

#         for k, v in population.items():
#             result[k] = torch.cat(
#                 [individual[k][None], v], dim=0
#             )
#         return Population(**result)
    
#     def spawn(self) -> 'PopulationLimiter':
#         return JoinInfluencer()

# class PopulationLimiter(PopulationInfluencer):
#     """
#     Allows the user to specify certain indices that can be updated in the individual.
#     Values at other indices will remain the same
#     """

#     def __init__(self, limit: torch.LongTensor=None):
#         """initializer

#         Args:
#             limit (torch.LongTensor, optional): The indices to use in the update. If None there is no limit. Defaults to None.
#         """

#         self.limit = limit

#     def influence(
#         self,
#         population: Population,
#         individual: Individual,
#         state: State
#     ) -> Population:
#         """

#         Args:
#             population (Population): The population to limit
#             individual (Individual): The individual
#             limit (torch.LongTensor, optional): The index to use to limit. Defaults to None.

#         Returns:
#             Population: The limited population
#         """
#         result = {}

#         if self.limit is None:
#             return population

#         for k, v in population.items():
#             individual_v = individual[k][None].clone()
#             individual_v = individual_v.repeat(v.size(0), 1, 1)
#             individual_v[:, :, self.limit] = v[:, :, self.limit].detach()
#             result[k] = individual_v
#         return Population(**result)
    
#     def spawn(self) -> 'PopulationLimiter':
#         return PopulationLimiter(self.limit.clone())


# class IndividualInfluencer(ABC):
#     """Modifies an Individual based on the Population"""

#     def __call__(self, individual: Individual, population: Population, state: State=None) -> Individual:
#         return self.influence(
#             individual, population, state or State()
#         )

#     @abstractmethod
#     def influence(self, individual: Individual, population: Population, state: State) -> Individual:
#         pass

#     @abstractmethod
#     def spawn(self) -> "IndividualInfluencer":
#         pass


# class PopulationInfluencer(ABC):
#     """"""

#     @abstractmethod
#     def influence(self, population: Population, individual: Population, state: State) -> Population:
#         pass

#     def __call__(self, population: Population, individual: Population, state: State=None) -> Population:
#         return self.influence(
#             population, individual, state or State()
#         )

#     @abstractmethod
#     def spawn(self) -> "PopulationInfluencer":
#         pass


# class Populator(ABC):
#     """Base class for creating a population from an individual"""


#     @abstractmethod
#     def populate(self, individual: Individual, state: State) -> Population:
#         """Spawn a population from an individual

#         Args:
#             individual (Individual): The individual to populate based on

#         Returns:
#             Population: The resulting population
#         """
#         pass

#     def __call__(self, individual: Individual, state: State=None) -> Population:
#         """Spawn a population from an individual

#         Args:
#             individual (Individual): The individual to populate based on

#         Returns:
#             Population: The resulting population
#         """
#         return self.populate(
#             individual, state or State()
#         )

#     @abstractmethod
#     def spawn(self) -> "Populator":
#         """Spawn a new populator from the current populator

#         Returns:
#             Populator: The spawned populator
#         """
#         pass


# class StandardPopulator(Populator):
#     """Populator that uses a standard populator method for all values in the individual"""

#     @abstractmethod
#     def populate_field(
#         self, key: str, val: typing.Union[torch.Tensor, Parameter], state: State
#     ) -> typing.Union[torch.Tensor, Parameter]:
#         pass

#     def populate(self, individual: Individual, state: State) -> Population:
#         """Call the populate method for each value and spawn the population

#         Args:
#             individual (Individual): The individual to populate based no

#         Returns:
#             Population: The resulting population
#         """
#         expanded = {}
#         for key, val in individual.items():
#             cur = self.populate_field(key, val, state)
#             if cur is not None:
#                 expanded[key] = cur
#         return Population(**expanded)


# class PopulatorDecorator(Populator):
#     """Decorate the results of a populator"""

#     def __init__(self, base_populator: Populator):
#         """initializer

#         Args:
#             base_populator (Populator): The populator to decorate
#         """
#         self.base_populator = base_populator

#     @abstractmethod
#     def decorate(
#         self, key: str, base_val, val: typing.Union[torch.Tensor, Parameter], state: State
#     ) -> typing.Union[torch.Tensor, Parameter]:
#         """Decorate each value in the population

#         Args:
#             key (str): the key for the value in the dictionary
#             base_val (): the value before populating
#             val (typing.Union[torch.Tensor, Parameter]): The value after populating

#         Returns:
#             typing.Union[torch.Tensor, Parameter]: The result of the decoration
#         """
#         pass

#     def populate(self, individual: Individual, state: State) -> Population:
#         """Spawn a population from an individual

#         Args:
#             individual (Individual): The individual to populate based on

#         Returns:
#             Population: The resulting population
#         """
#         populated = self.base_populator(individual)
#         expanded = {}
#         for key, val in populated.items():
#             if key in individual:
#                 expanded[key] = self.decorate(key, individual[key], val, state)

#         return Population(**expanded)

#     @abstractmethod
#     def spawn(self) -> "PopulatorDecorator":
#         pass


# class RepeatPopulator(StandardPopulator):
#     """Populator that outputs all the same values for the population dimension"""

#     def __init__(self, k: int):
#         """initializer

#         Args:
#             k (int): The size of the population
#         """
#         self.k = k

#     def populate_field(
#         self, key: str, val: typing.Union[torch.Tensor, Parameter], state: State
#     ) -> typing.Union[torch.Tensor, Parameter]:
#         """Expands each of the values by repeating along the population dimension

#         Args:
#             key (str): The name of the value
#             val (typing.Union[torch.Tensor, Parameter]): the value to repeat

#         Returns:
#             typing.Union[torch.Tensor, Parameter]: The expanded value
#         """
#         return expand_dim0(val, self.k, False)

#     def spawn(self) -> "RepeatPopulator":
#         return RepeatPopulator(self.k)


# class SimpleGaussianPopulator(StandardPopulator):
#     """initializer"""

#     def __init__(self, k: int):
#         """initializer

#         Args:
#             k (int): The size of the population
#         """
#         self.k = k

#     def populate_field(
#         self, key: str, val: typing.Union[torch.Tensor, Parameter], state: State
#     ) -> typing.Union[torch.Tensor, Parameter]:
#         """

#         Args:
#             key (str): The key for the element
#             val (typing.Union[torch.Tensor, Parameter]): The element to create the population for

#         Returns:
#             typing.Union[torch.Tensor, Parameter]: The element expanded along the dimension
#         """

#         perturbation = torch.randn(self.k - 1, *val.shape, device=val.device)
#         val = torch.cat([val[None], val[None] + perturbation])

#     def spawn(self) -> "SimpleGaussianPopulator":
#         return SimpleGaussianPopulator(self.k)


# # TODO: Why not just use the mutator? 
# # 1) populate -> mutate
# class GaussianPopulator(StandardPopulator):
#     """Create a population using Gaussian noise on the individual
#     """

#     def __init__(self, k: int, std: float = 1, equal_change_dim: int = None):
#         self.k = k
#         self.std = std
#         self.equal_change_dim = equal_change_dim

#     def populate_field(self, key: str, val: torch.Tensor, state: State):

#         shape = [self.k - 1, *val.shape]
#         if self.equal_change_dim:
#             shape[self.equal_change_dim] = 1
#         noise = torch.randn(*shape, device=val.device) * self.std

#         return torch.cat([val[None], val[None] + noise])

#     def spawn(self) -> "GaussianPopulator":
#         return GaussianPopulator(self.k, self.std, self.equal_change_dim)


# class ConservativePopulator(PopulatorDecorator):
#     """Decorator for a populator that replaces the initial results of the populator
#     algorithm with the original values
#     """

#     def __init__(
#         self,
#         base_populator: Populator,
#         percent_change: float = 0.1,
#         same_change: bool = True,
#     ):
#         """initializer

#         Args:
#             base_populator (Populator): The populator decorated
#             percent_change (float, optional): The percentage to change the population. Defaults to 0.1.
#             same_change (bool, optional): Whether the same elements should be 'conserved' for the entire population. Defaults to True.

#         Raises:
#             ValueError: If the percent change is less than 0 or greater than 1s
#         """
#         super().__init__(base_populator)
#         if not (0.0 <= percent_change <= 1.0):
#             raise ValueError("Percent change must be between 0 and 1")
#         self.percent_change = percent_change
#         self.same_change = same_change

#     def decorate(
#         self,
#         key: str,
#         base_val: torch.Tensor,
#         val: typing.Union[torch.Tensor, Parameter], 
#         state: State
#     ) -> typing.Union[torch.Tensor, Parameter]:
#         """Decorate the population

#         Args:
#             key (str): The name of the item
#             base_val (torch.Tensor): The original value
#             val (typing.Union[torch.Tensor, Parameter]): The updated value after 'populating'

#         Returns:
#             typing.Union[torch.Tensor, Parameter]: The decorated value
#         """
#         base_val = base_val[None]
#         size = list(val.size())
#         if self.same_change:
#             size[0] = 1

#         to_change = (
#             torch.rand(*size, device=val.device) < self.percent_change
#         ).type_as(val)
#         return to_change * val + (1 - to_change) * base_val

#     def spawn(self) -> "ConservativePopulator":
#         """
#         Returns:
#             ConservativePopulator: A new conservative spawner with the same parameters
#         """
#         return ConservativePopulator(
#             self.base_populator.spawn(), self.percent_change, self.same_change
#         )


# class BinaryPopulator(StandardPopulator):
#     """
#     """

#     def __init__(
#         self,
#         k: int = 1,
#         keep_p: float = 0.1,
#         equal_change_dim: int = None,
#         to_change: typing.Union[int, float] = None,
#         reorder_params: bool = True,
#         zero_neg: bool = False,
#     ):
#         """initializer

#         Args:
#             k (int, optional): The population size. Defaults to 1.
#             keep_p (float, optional): Probability of keeping the current value. Defaults to 0.1.
#             equal_change_dim (int, optional): Whether to change all values in an individual the same. Defaults to None.
#             to_change (typing.Union[int, float], optional): the number of elements to change. Defaults to None.
#             reorder_params (bool, optional): . Defaults to True.
#             zero_neg (bool, optional): whether the negative is 0 or -1. Defaults to False.

#         Raises:
#             RuntimeError: If the probability of keeping p is not valid
#         """
#         if 0.0 >= keep_p or 1.0 < keep_p:
#             raise RuntimeError("Argument p must be in range (0.0, 1.0] not {keep_p}")
#         assert k > 1
#         self.keep_p = keep_p
#         self.k = k
#         self._equal_change_dim = equal_change_dim
#         self._is_percent_change = isinstance(to_change, float)
#         if self._is_percent_change:
#             assert 0 < to_change <= 1.0
#         elif to_change is not None:
#             assert to_change > 0
#         self._to_change = to_change
#         self._reorder_params = reorder_params
#         self._zero_neg = zero_neg

#     # TODO: Move this to a "PopulationModifier" or a Decorator
#     def _generate_keep(self, param: torch.Tensor):

#         shape = [self.k - 1, *param.shape]
#         if self._equal_change_dim is not None:
#             shape[self._equal_change_dim] = 1

#         param = (param > 0.0).type_as(param)
#         keep = (torch.rand(*shape, device=param.device) < self.keep_p).type(param.dtype)

#         if self._to_change is None:
#             return keep

#         if self._is_percent_change:
#             ignore_change = (
#                 torch.rand(1, 1, *param.shape[1:], device=param.device)
#                 > self._to_change
#             ).type_as(param)
#         else:
#             _, indices = torch.rand(
#                 math.prod(param.shape[1:]), device=param.device
#             ).topk(self._to_change, dim=-1)
#             ignore_change = torch.ones(math.prod(param.shape[1:]), device=param.device)
#             ignore_change[indices] = 0.0
#             ignore_change = ignore_change.view(1, 1, *param.shape[1:])

#         return torch.max(keep, ignore_change)

#     def populate_field(self, key: str, val: torch.Tensor, state: State):

#         keep = self._generate_keep(val)

#         changed = -val[None] if not self._zero_neg else (1 - val[None])
#         perturbed_params = keep * val[None] + (1 - keep) * changed
#         concatenated = cat_params(val, perturbed_params, reorder=True)
#         if not self._reorder_params:
#             return concatenated
#         reordered = torch.randperm(len(concatenated), device=concatenated.device)
#         return concatenated[reordered]

#     def spawn(self) -> "BinaryPopulator":
#         return BinaryPopulator(
#             self.k,
#             self.keep_p,
#             self._equal_change_dim,
#             self._to_change,
#             self._reorder_params,
#             self._zero_neg,
#         )
