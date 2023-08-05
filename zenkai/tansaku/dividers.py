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
        if not assessment.maximize:
            loss = 1 / (0.1 + loss)
        prob = (loss / loss.sum()).numpy()
        
        parents1, parents2 = [], []
        for _ in range(self.n_divisions):
            parent1, parent2 = np.random.choice(
                len(assessment), 2, False, prob
            )
            parents1.append(parent1)
            parents2.append(parent2)
        return population[parents1], population[parents2]


class EqualSelector(Divider):

    def __call__(self, population: Population) -> typing.Tuple[Population]:
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
        
        selection = torch.multinomial(
            p, 2 * len(fitness), True
        )
        parents1, parents2 = [], []

        for k, v in population:
        # selection = selection.view(2, p.size(0))
            selected1, selected2 = v[selection].view(2, selection.size(0) // 2, *v.shape[1:])
            parents1.append(selected1)
            parents2.append(selected2)
        return Population(*parents1), Population(*parents2)
