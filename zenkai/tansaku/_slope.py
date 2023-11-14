# 3rd party
import torch

# local
from ._keep import Individual, Population


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
