# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch

from ..utils import to_signed_neg, to_zero_neg
from .core import Individual, Population
from .exploration import EqualsAssessmentDist

# local
from .reducers import BinaryProbReducer, SlopeReducer


class IndividualInfluencer(ABC):
    """Modifies an Individual based on the Population"""

    @abstractmethod
    def __call__(self, individual: Individual, population: Population) -> Individual:
        pass

    @abstractmethod
    def spawn(self) -> "IndividualInfluencer":
        pass


class PopulationInfluencer(ABC):
    """"""

    @abstractmethod
    def __call__(self, population: Population, individual: Population) -> Population:
        pass

    @abstractmethod
    def spawn(self) -> "PopulationInfluencer":
        pass


class SlopeInfluencer(IndividualInfluencer):
    """Modifies an individual based on the slope of a population"""

    def __init__(
        self, momentum: float, lr: float = 0.1, x: str = "x", maximize: bool = False
    ):
        """initializer

        Args:
            momentum (float): The amount of momentum for the slope
            lr (float, optional): The "learning" rate (i.e. how much to go in the direction).
                Defaults to 0.1.
            x (str, optional): The name for the x value. Defaults to "x".
            maximize (bool, optional): Whether to maximize or minimize.
              If minimizing, the sign of the lr will be reversed. Defaults to False.
        """
        self._slope_selector = SlopeReducer(momentum)
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

    def spawn(self) -> "SlopeInfluencer":
        return SlopeInfluencer(self._momentum, self.lr, self.x, self._multiplier == 1)


class PopulationLimiter(PopulationInfluencer):
    """
    """

    def __call__(
        self,
        population: Population,
        individual: Individual,
        limit: torch.LongTensor = None,
    ) -> Population:
        """

        Args:
            population (Population): 
            individual (Individual): 
            limit (torch.LongTensor, optional): . Defaults to None.

        Returns:
            Population: 
        """

        if limit is None:
            return population

        result = {}

        for k, v in population:
            individual_v = individual[k][None]
            individual_v = individual_v.repeat(v.size(0), 1, 1)
            individual_v[:, :, limit] = v[:, :, limit].detach()
            result[k] = individual_v
        return Population(**result)
    
    def spawn(self) -> 'PopulationLimiter':
        return PopulationLimiter()


# TODO: Probably remove
class BinaryAdjGaussianInfluencer(IndividualInfluencer):
    """Choose whether each value should be positive or negative based on the value of each"""

    def __init__(self, k: int, x: str = "x", zero_neg: bool = False):
        """initializer
        Args:
            k (int): The number of
            x (str, optional): The name of the x value. The individual/population must have it. Defaults to 'x'.
            zero_neg (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """
        # k is an upper bound. If the dimension size is smaller
        # the dimension size will be used
        super().__init__()
        if k <= 0:
            raise ValueError(f"Argument k must be greater than 0 not {k}")
        self.x = x
        self.k = k
        self.zero_neg = zero_neg
        self.pos_assessment_calc = EqualsAssessmentDist(1)
        self.neg_assessment_calc = EqualsAssessmentDist(-1 if not zero_neg else 0)

    def __call__(self, original: Individual, population: Population) -> Individual:
        """Estimate the value of positives and negatives and update

        Args:
            original (Individual): The individual to modify
            population (Population): The population to modify based on

        Returns:
            Individual: The modified individual
        """

        assessments = population.stack_assessments()
        x_pop = population[self.x]
        x_ind = original[self.x]
        k = min(self.k, x_ind.shape[1])

        # calculate the average value of positives and negatives
        pos = self.pos_assessment_calc.mean(assessments, x_pop)
        neg = self.neg_assessment_calc.mean(assessments, x_pop)

        difference = pos - neg
        difference[pos.isnan()] = -1
        difference[neg.isnan()] = 1
        base_result = torch.sign(difference)
        # make it into a "maximization" problem
        adj_difference = 1 / (difference.abs() + 1.0)
        # reduce to 0 if no change
        adj_difference[x_ind == base_result] = 0.0
        _, best_indices = adj_difference.topk(dim=1, k=k, largest=True)
        best_differences = difference.gather(index=best_indices, dim=1)
        result = torch.sign(best_differences)
        if self.zero_neg:
            result = to_zero_neg(result)
        x_ind = x_ind.scatter(1, best_indices, result)

        return Individual(**{self.x: x_ind})

    def spawn(self) -> "BinaryAdjGaussianInfluencer":
        return BinaryAdjGaussianInfluencer(self.k, self.x, self.zero_neg)


# TODO: Probably remove
class BinaryGaussianInfluencer(IndividualInfluencer):
    """Choose up to k updates for the individual. If there are no
    updates that produce a large change in value. None will be updated"""

    def __init__(self, k: int, x: str = "x", zero_neg: bool = False):
        super().__init__()
        if k <= 0:
            raise ValueError(f"Argument k must be greater than 0 not {k}")
        self.x = x
        self.k = k
        self.zero_neg = zero_neg
        self.pos_assessment_calc = EqualsAssessmentDist(1)
        self.neg_assessment_calc = EqualsAssessmentDist(-1 if not zero_neg else 0)

    def __call__(self, original: Individual, population: Population) -> Individual:

        assessments = population.stack_assessments()
        x_pop = population[self.x]
        x_ind = original[self.x]

        k = min(self.k, x_ind.shape[1])
        pos = self.pos_assessment_calc.mean(assessments, x_pop)
        neg = self.neg_assessment_calc.mean(assessments, x_pop)

        difference = pos - neg
        base_result = torch.sign(difference)
        _, best_indices = difference.abs().topk(dim=1, k=k, largest=False)
        best_results = base_result.gather(dim=1, index=best_indices)
        result = x_ind.scatter(1, best_indices, best_results)

        if self.zero_neg:
            result = to_zero_neg(result)
        return Individual(**{self.x: result})

    def spawn(self) -> "BinaryGaussianInfluencer":
        return BinaryGaussianInfluencer(self.k, self.x, self.zero_neg)


# TODO: Probably remove
class BinaryProbInfluencer(IndividualInfluencer):


    def __init__(self, zero_neg: bool, x: str = "x") -> None:
        """initializer

        Args:
            zero_neg (bool): Whether the negative value is represented by zero
            x (str, optional): The name of the value to modify. Defaults to "x".
        """
        super().__init__()
        self.zero_neg = zero_neg
        self._binary_selector = BinaryProbReducer()
        self.x = x

    def __call__(self, original: Individual, population: Population) -> Individual:
        """

        Args:
            original (Individual): The individual to modify
            population (Population): The population to modify based on

        Returns:
            Individual
        """

        updated = self._binary_selector(population)
        original = original[self.x]
        neg_count = updated[self._binary_selector.neg_count]
        pos_count = updated[self._binary_selector.pos_count]
        x = updated[self.x]
        x = (torch.rand(x.size(), device=x.device) < x).type_as(x)

        no_change = (pos_count == 0) | (neg_count == 0)

        if not self.zero_neg:
            original = (original + 1) / 2
        result = (~no_change).float() * x + (no_change).float() * original
        if not self.zero_neg:
            result = to_signed_neg(result)
        return Individual(x=result)

    def spawn(self) -> "BinaryProbInfluencer":
        return BinaryProbInfluencer(self.zero_neg, self.x)
