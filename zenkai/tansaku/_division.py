
# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch

# local
from ..kaku import Population
from ..kaku import State
from . import _select as selection


class Divider(ABC):

    @abstractmethod
    def divide(self, population: Population, state: State) -> typing.Tuple[Population]:
        pass

    def __call__(self, population: Population, state: State=None) -> typing.Tuple[Population]:
        return self.divide(
            population, state or State()
        )

    @abstractmethod
    def spawn(self) -> 'Divider':
        pass


class ProbDivider(Divider):
    """Divide the population into two based on the fitness proportionality
    """

    def __init__(self, selector: selection.Selector, divide_start: int=1):
        """initializer

        Args:
            n_divisions (int): number of pairs to generate
        """
        super().__init__()
        self.divide_start = divide_start
        self.selector = selector
        if divide_start < 1:
            raise ValueError(f'Divide start must be greater than 1')

    def divide(self, population: Population, state: State) -> typing.Tuple[Population]:
        """Divide the population into two based on the fitness proportionality

        Args:
            population (Population): The population to divide

        Returns:
            typing.Tuple[Population]: The two parents
        """

        # calc_probs() 
        assessment = population.stack_assessments()

        # shape = assessment.shape
        reduced = assessment.reduce_image(self.divide_start)
        
        index_map = self.selector.select(reduced)
        # index_map = selector.select(reduced)
        
        result = index_map.select_index(population)
        return Population(**result[0]), Population(**result[1])

    def spawn(self) -> Divider:
        return ProbDivider(self.n_divisions)


class EqualDivider(Divider):

    def divide(self, population: Population, state: State) -> typing.Tuple[Population]:
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


# TODO: REMOVE
# def select_parents(population: Population, prob: torch.Tensor, n_divisions: int):
#     parents1, parents2 = [], []
    
#     base_shape = prob.shape
    
#     # Figure out how to divide this up
#     # (population, ...)
#     # select()
#     if prob.dim() > 1:
#         r = torch.arange(0, len(prob.shape)).roll(-1).tolist()
#         prob = prob.transpose(*r)
#     # (..., population)
#     prob = prob[None]
#     # (1, ..., population)
#     prob = prob.repeat(n_divisions, *[1] * len(prob.shape))
#     # (n_divisions * ..., population)
#     prob = prob.reshape(-1, prob.shape[-1])
#     parents1, parents2 = torch.multinomial(
#         prob, 2, False
#     ).transpose(1, 0)
#     # (n_divisions * ...), (n_divisions * ...)
#     parents1 = parents1.reshape(n_divisions, *base_shape[1:])
#     parents2 = parents2.reshape(n_divisions, *base_shape[1:])
#     # (n_divisions, ...)
    
#     # not guaranteed to be the correct size
#     if parents1.dim() == 1:
#         return population.sub[parents1], population.sub[parents2]
#     return population.gather_sub(parents1), population.gather_sub(parents2)
