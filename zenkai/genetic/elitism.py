
import torch
from abc import ABC, abstractmethod
from ..tansaku.core import Population
from ..kaku import selection


class Elitism(ABC):

    @abstractmethod
    def __call__(self, population1: Population, population2: Population) -> Population:
        pass


class KBestElitism(Elitism):
    """Add the k best from the previous generation to the new generation
    """

    def __init__(self, k: int, divide_start: int=1):
        """initializer

        Args:
            k (int): The number to keep
        """
        if k <= 0:
            raise ValueError(f'Argument k must be greater than 0 not {self.k}')
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
        # 
        # assessemnt = assessmetn.reduce_image(2, 'mean') DONE
        # selector = ProbSelector(dim=2)
        # selector = TopKSelector(k=2, dim=0)
        # index_map = selector(assessment)
        # select = population.select_index(index_map)
        # selected = index_map(features)
        selector = selection.TopKSelector(self.k)
        assessment = population1.stack_assessments().reduce_image(self.divide_start)
        index_map = selector.select(assessment)

        population1 = population1.select_index(index_map)

        return population1.pstack(population2)
        # _, indices = assessment.value.topk(self.k, largest=assessment.maximize)
        # results = {}
        # for k, v1, v2 in population1.loop_over(population2, only_my_k=True, union=False):
        #     results[k] = torch.cat(
        #         [v1, v2]
        #     )

        # return Population(**results)
    
    def spawn(self) -> 'KBestElitism':
        return KBestElitism(self.k)

