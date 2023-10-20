import torch

from .assess import Assessment
from abc import abstractmethod, ABC


class IndexMap(object):
    """Use to select indices from a multidimensional tensor. Only works for dimension 0
    """

    def __init__(self, *index: torch.LongTensor, dim: int=0):
        super().__init__()
        self.index = index
        self.dim = dim
    
    def __getitem__(self, i: int) -> 'IndexMap':

        return IndexMap(self.index[i], dim=self.dim)
    
    def index_for(self, i: int, x: torch.Tensor) -> torch.Tensor:

        index = self.index[i].clone()
        if index.dim() > x.dim():
            raise ValueError(f'Gather By dim must be less than or equal to the value dimension')
        shape = [1] * index.dim()
        for i in range(index.dim(), x.dim()):
            index = index.unsqueeze(i)
            shape.append(x.shape[i])
        index = index.repeat(*shape)
        return x.gather(self.dim, index)
    
    def __len__(self) -> int:
        return len(self.index)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        result = tuple(self.index_for(i, x) for i in range(len(self)))
        if len(result) == 1:
            return result[0]
        return result


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
        return IndexMap(topk, dim=0)


class ParentSelector(Selector):

    def __init__(self, k: int, divide_from: int=1, dim: int=0, largest: bool=True):
        self.k = k
        self.largest = largest
        self.dim = dim
        self.divide_from = divide_from
    
    def select(self, assessment: Assessment) -> IndexMap:
        
        base_shape = assessment.shape
        loss = assessment.value
        if not assessment.maximize:
            loss = 1 / (0.01 + loss)
        prob = (loss / loss.sum(dim=0, keepdim=True))
        if (prob < 0.0).any():
            raise ValueError('All assessments must be greater than 0 to use this divider')
        
        # Figure out how to divide this up
        # (population, ...)
        # select()
        if prob.dim() > 1:
            r = torch.arange(0, len(prob.shape)).roll(-1).tolist()
            prob = prob.transpose(*r)

        # (..., population)
        prob = prob[None]

        # (1, ..., population)
        prob = prob.repeat(self.k, *[1] * len(prob.shape))
        # (n_divisions * ..., population)
        prob = prob.reshape(-1, prob.shape[-1])
        parents1, parents2 = torch.multinomial(
            prob, 2, False
        ).transpose(1, 0)

        parents1 = parents1.reshape(self.k, *base_shape[1:])
        parents2 = parents2.reshape(self.k, *base_shape[1:])
        # (n_divisions * ...), (n_divisions * ...)

        # assessment = assessment.reduce_image(self.divide_from)

        return IndexMap(parents1, parents2, dim=0)


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
        return IndexMap(best, dim=0)
