import torch

from .assess import Assessment
from abc import abstractmethod, ABC


class IndexMap(object):
    """Use to select indices from a multidimensional tensor. Only works for dimension 0
    """

    def __init__(self, index: torch.LongTensor, dim: int=0):
        super().__init__()
        self.index = index
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        index = self.index.clone()
        if index.dim() > x.dim():
            raise ValueError(f'Gather By dim must be less than or equal to the value dimension')
        shape = [1] * index.dim()
        for i in range(index.dim(), x.dim()):
            index = index.unsqueeze(i)
            shape.append(x.shape[i])
        index = index.repeat(*shape)
        print(x.shape, index.shape)
        return x.gather(self.dim, index)


class Selector(ABC):
    """Use to select indices from a multidimensional tensor. Only works for dimension 0 so must be reshaped
    """

    @abstractmethod
    def select(self, assessment: Assessment) -> 'IndexMap':
        pass

    def __call__(self, assessment: Assessment) -> 'IndexMap':
        
        return self.select(assessment)


class TopKSelector(Selector):

    def __init__(self, k: int, dim: int=0, largest: bool=True):
        self.k = k
        self.largest = largest
        self.dim = dim

    def select(self, assessment: Assessment) -> IndexMap:
        
        _, topk = assessment.value.topk(self.k, dim=self.dim, largest=self.largest)
        return IndexMap(topk)


class BestSelector(Selector):

    def __init__(self, k: int, dim: int=0, largest: bool=True):
        self.k = k
        self.largest = largest
        self.dim = dim

    def select(self, assessment: Assessment) -> IndexMap:
        
        if self.largest:
            _, best = assessment.value.max(self.k, dim=self.dim, keepdim=True)
        else:
            _, best = assessment.value.min(self.k, dim=self.dim, keepdim=True)
        return IndexMap(best)
