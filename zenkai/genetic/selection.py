import torch

from ..assess.selection import Selector, IndexMap
from ..kaku.assess import Assessment



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
