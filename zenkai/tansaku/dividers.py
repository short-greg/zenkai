# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import numpy as np
import torch

# local
from .core import Population


class Divider(ABC):

    @abstractmethod
    def divide(self, population: Population) -> typing.Tuple[Population]:
        pass

    def __call__(self, population: Population) -> typing.Tuple[Population]:
        return self.divide(
            population
        )

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
        super().__init__()
        self.n_divisions = n_divisions

    def divide(self, population: Population) -> typing.Tuple[Population]:
        """Divide the population into two based on the fitness proportionality

        Args:
            population (Population): The population to divide

        Returns:
            typing.Tuple[Population]: The two parents
        """
        assessment = population.stack_assessments()
        assessment = assessment.view(assessment.shape[0], -1)
        assessment = assessment.mean(dim=1)
        loss = assessment.value
        if not assessment.maximize:
            loss = 1 / (0.1 + loss)
        prob = (loss / loss.sum()).numpy()
        if (prob < 0.0).any():
            raise ValueError('All assessments must be greater than 0 to use this divider')
        parents1, parents2 = [], []
        for _ in range(self.n_divisions):
            parent1, parent2 = np.random.choice(
                len(assessment), 2, False, prob
            )
            parents1.append(parent1)
            parents2.append(parent2)
        return population.sub[parents1], population.sub[parents2]

    def spawn(self) -> Divider:
        return FitnessProportionateDivider(self.n_divisions)


class EqualDivider(Divider):

    def divide(self, population: Population) -> typing.Tuple[Population]:
        """Divide the population into two based on the fitness proportionality

        Args:
            population (Population): The population to divide

        Returns:
            typing.Tuple[Population]: The two parents
        """
        fitness = population.stack_assessments().reduce('samplemeans')
        if not fitness.maximize:
            p = torch.nn.functional.softmin(fitness.value, dim=0).detach()
        else:
            p = torch.nn.functional.softmax(fitness.value, dim=0).detach()
        
        selection1, selection2 = torch.multinomial(
            p, 2 * len(fitness), True
        ).view(2, -1)

        return population.sub[selection1], population.sub[selection2]

    def spawn(self) -> Divider:
        return EqualDivider()
