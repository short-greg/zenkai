# 1st party
from abc import ABC, abstractmethod

# local
from ..kaku import Population
from . import _select as selection


class Elitism(ABC):
    @abstractmethod
    def __call__(self, population1: Population, population2: Population) -> Population:
        pass


class KBestElitism(Elitism):
    """Add the k best from the previous generation to the new generation"""

    def __init__(self, k: int, divide_start: int = 1):
        """initializer

        Args:
            k (int): The number to keep
        """
        if k <= 0:
            raise ValueError(f"Argument k must be greater than 0 not {self.k}")
        self.k = k
        self.divide_start = divide_start

    def __call__(self, population1: Population, population2: Population) -> Population:
        """

        Args:
            population1 (Population): previous generation
            population2 (Population): new generation

        Returns:
            Population: the updated new generation
        """
        selector = selection.TopKSelector(self.k)
        assessment = population1.stack_assessments().reduce_image(self.divide_start)
        index_map = selector.select(assessment)

        population1 = index_map.select_index(population1)

        return population1.pstack([population2])

    def spawn(self) -> "KBestElitism":
        return KBestElitism(self.k)
