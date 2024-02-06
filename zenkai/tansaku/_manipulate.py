
# 3rd party
import torch

# local
from ._keep import Individual, Population, TensorDict
from ..kaku import State, Meta
from ..kaku import (
    Individual,
    Population,
)


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

    def __call__(self, population: Population) -> torch.Tensor:
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


class ApplyMomentum(object):
    """Reduces the population using momentum"""

    def __init__(self, momentum: float = None):
        """initializer

        Args:
            momentum (float, optional): Weight for previous time step Defaults to None.
            maximize (bool, optional): Whether to maximize the evaluation. Defaults to True.
        """
        super().__init__()
        self._momentum = momentum
        self._params_updated = None
        self._keep_s = True
        self.diff = None
        self.cur = None
        self.dx = None
        self._ = Meta()
        self._['diff'] = {}
        self._['cur'] = {}

    def apply_momentum(
        self, k: str, individual: torch.Tensor
    ) -> torch.Tensor:
        """Decorates the individual with the momentum

        Args:
            key (str): The name of the field
            individual (torch.Tensor): The individual to decorate
            assessment (Assessment): The assessment for the individual

        Returns:
            torch.Tensor: The decorated reducer
        """

        diff = self._['diff'].get(k)
        cur = self._['cur'].get(k)

        if diff is None and cur is None:
            self._.cur[k] = individual
            # my_state.cur = individual
        elif self._.diff.get(k) is None:
            self._.diff[k] = individual - self._.cur[k]
            self._.cur[k] = individual

        else:
            self._.diff[k] = (individual - self._.cur[k]) + self._momentum * self._.diff[k]
            self._.cur[k] = self._.diff[k] + individual

        return self._.cur[k]

    def __call__(self, individual: Individual) -> Individual:
        """Reduces the population and decorates it

        Args:
            population (Population): The population to reduce

        Returns:
            Individual: The reduction (individual)
        """

        result = {}
        for k, v in individual.items():
            result[k] = self.apply_momentum(k, v)
        return Individual(**result)

    def spawn(self) -> "ApplyMomentum":
        return ApplyMomentum(self._momentum)


class Apply(object):
    """Mixes two populations together"""

    def __init__(self, noise_f, *args, **kwargs):

        self.noise_f = lambda x: noise_f(x, *args, **kwargs)
        self._args = args
        self._kwargs = kwargs

    def __call__(self, tensor_dict: TensorDict) -> TensorDict:
        
        return tensor_dict.apply(
            self.noise_f
        )

    def spawn(self) -> "Apply":
        return Apply(
            self.noise_f, self._args, self._kwargs
        )
