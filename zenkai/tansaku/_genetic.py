# 1st party
from abc import abstractmethod
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch

# local
from ..kaku import Population
from . import _select as selection


class Divider(object):
    """Divide the population into two based on the fitness proportionality"""

    def __init__(self, selector: selection.Selector):
        """initializer

        Args:
        """
        super().__init__()
        # self.divide_start = divide_start
        self.selector = selector
        # if divide_start < 1:
        #     raise ValueError("Divide start must be greater than 1")

    def __call__(self, population: Population) -> typing.Tuple[Population]:
        """Divide the population into two based on the fitness proportionality

        Args:
            population (Population): The population to divide

        Returns:
            typing.Tuple[Population]: The two parents
        """

        # calc_probs()
        # assessment = population.stack_assessments()

        # shape = assessment.shape
        # reduced = assessment.reduce_image(self.divide_start)

        population = self.selector(population)
        # index_map = selector.select(reduced)
        # result = index_map.select_index(population)
        # return Population(**result[0]), Population(**result[1])
        return selection.split_tensor_dict(
            population, -1
        )

    def spawn(self) -> 'Divider':
        return Divider(self.selector)


class Elitism(object):
    """Add the k best from the previous generation to the new generation"""

    def __init__(self, selector: selection.Selector):
        """initializer

        Args:
            k (int): The number to keep
        """
        self.selector = selector

    def __call__(self, population1: Population, population2: Population) -> Population:
        """

        Args:
            population1 (Population): previous generation
            population2 (Population): new generation

        Returns:
            Population: the updated new generation
        """
        # selector = selection.TopKSelector(self.k)
        #assessment = population1.stack_assessments().reduce_image(self.divide_start)
        # index_map = selector.select(assessment)

        # population1 = index_map(population1)

        population1 = self.selector(population1)
        return population1.pstack([population2])

    def spawn(self) -> "Elitism":
        return Elitism(self.selector)



class CrossOver(ABC):
    @abstractmethod
    def __call__(self, parents1: Population, parents2: Population) -> Population:
        pass


class BinaryRandCrossOver(CrossOver):
    """Mix two tensors together by choosing one gene for each"""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __call__(self, parents1: Population, parents2: Population) -> Population:
        """Mix two tensors together by choosing one gene for each

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix

        Returns:
            torch.Tensor: The mixed result
        """
        result = {}
        for k, p1, p2 in parents1.loop_over(parents2, only_my_k=True, union=False):
            to_choose = torch.rand_like(p1) > self.p
            result[k] = p1 * to_choose.type_as(p1) + p2 * (~to_choose).type_as(p2)
        return Population(**result)

    def spawn(self) -> "BinaryRandCrossOver":
        return BinaryRandCrossOver(self.p)


class SmoothCrossOver(CrossOver):
    """Do a smooth interpolation between the values to breed"""

    def __call__(self, parents1: Population, parents2: Population) -> Population:
        """Mix two tensors together by choosing one gene for each

        Args:
            key (str): The name of the field
            val1 (torch.Tensor): The first value to mix
            val2 (torch.Tensor): The second value to mix

        Returns:
            torch.Tensor: The mixed result
        """
        result = {}
        for k, p1, p2 in parents1.loop_over(parents2, only_my_k=True, union=False):
            degree = torch.rand_like(p1)
            result[k] = p1 * degree + p2 * (1 - degree)
        return Population(**result)

    def spawn(self) -> "SmoothCrossOver":
        return SmoothCrossOver()
