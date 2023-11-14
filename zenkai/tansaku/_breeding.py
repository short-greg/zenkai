# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch

# local
from ..kaku import Population


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
