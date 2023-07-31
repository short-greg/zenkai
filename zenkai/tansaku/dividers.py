# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import numpy as np

# local
from .core import Population


class Divider(ABC):

    @abstractmethod
    def __call__(self, population: Population) -> typing.Tuple[Population]:
        pass

    @abstractmethod
    def spawn(self) -> 'Divider':
        pass


class FitnessProportionateDivider(Divider):
    """Divide the population into two based on the fitness proportionality
    """

    def __init__(self, n_divisions: int):
        """initializer

        Args:
            n_divisions (int): number of pairs to generate
        """

        self.n_divisions = n_divisions

    def __call__(self, population: Population) -> typing.Tuple[Population]:
        """Divide the population into two based on the fitness proportionality

        Args:
            population (Population): The population to divide

        Returns:
            typing.Tuple[Population]: The two parents
        """
        assessment = population.stack_assessments()
        assessment = assessment.view(assessment.shape[0], -1)
        assessment.mean(1)
        loss = assessment.value
        loss = (0.1 + loss)
        prob = (loss / loss.sum()).numpy()
        parents1, parents2 = [], []
        for _ in range(self.n_divisions):
            parent1, parent2 = np.random.choice(
                len(assessment), 2, False, prob
            )
            parents1.append(parent1)
            parents2.append(parent2)
        return population[parents1], population[parents2]
