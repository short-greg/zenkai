

import torch


# 1st party
import typing
from abc import ABC, abstractmethod
import functools

# 3rd party
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from ..kaku import IndexMap, Selector

# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch

# local
from ..kaku import State, Population, Individual, TensorDict
from ..utils import gahter_idx_from_population


# local
from ..utils import get_model_parameters, update_model_parameters, expand_dim0, flatten_dim0, gather_idx_from_population

from ..kaku import IO, Assessment
from ..kaku import Reduction, Criterion, State, Criterion

from copy import deepcopy


import torch

from ..kaku.assess import Assessment
from abc import abstractmethod, ABC

# TODO: Move to utils


# Only use a class if I think that it will be 'replaceable'
# Elitism() <-
# Mixer() <- remove   tansaku.conserve(old_p, new_p, prob=...)
# Crossover()
# Perturber()
# Sampler() (Include reduers in here)
# SlopeCalculator() <- doesn't need to be a functor.. I should combine this with "SlopeMapper"... Think about this more
# concat <- add in concat
# Limiter??? - similar to "keep mixer" -> tansaku.limit_feature(population, limit=...)
# Divider() -> ParentSelector() <- rename
# Assessor
# concat()
# 


# TODO: Remove
def gen_like(f, k: int, orig_p: torch.Tensor, requires_grad: bool=False) -> typing.Dict:
    """generate a tensor like another

    Args:
        f (_type_): _description_
        k (int): _description_
        orig_p (torch.Tensor): _description_
        requires_grad (bool, optional): _description_. Defaults to False.

    Returns:
        typing.Dict: _description_
    """
    return f([k] + [*orig_p.shape[1:]], dtype=orig_p.dtype, device=orig_p.device, requires_grad=requires_grad)


def binary_prob(
    x: torch.Tensor, loss: torch.Tensor, retrieve_counts: bool = False
) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """

    Args:
        x (torch.Tensor): The population input
        loss (torch.Tensor): The loss
        retrieve_counts (bool, optional): Whether to return the positive 
          and negative counts in the result. Defaults to False.

    Returns:
        typing.Union[ torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor] ]: _description_
    """
    is_pos = (x == 1).unsqueeze(-1)
    is_neg = ~is_pos
    pos_count = is_pos.sum(dim=0)
    neg_count = is_neg.sum(dim=0)
    positive_loss = (loss[:, :, None] * is_pos.float()).sum(dim=0) / pos_count
    negative_loss = (loss[:, :, None] * is_neg.float()).sum(dim=0) / neg_count
    updated = (positive_loss < negative_loss).type_as(x).mean(dim=-1)

    if not retrieve_counts:
        return updated

    return updated, pos_count.squeeze(-1), neg_count.squeeze(-1)


# TODO: Remove
def select_best_individual(
    pop_val: torch.Tensor, assessment: Assessment
) -> torch.Tensor:
    """
    Args:
        pop_val (torch.Tensor): The tensor for the population
        assessment (Assessment): The evaluation of the individuals in a population

    Returns:
        Tensor: the best individual in the population
    """
    if (assessment.value.dim() != 1):
        raise ValueError('Expected one assessment for each individual')
    _, idx = assessment.best(0, True)
    return pop_val[idx[0]]


# TODO: Remove
def select_best_sample(pop_val: torch.Tensor, assessment: Assessment) -> torch.Tensor:
    """
    Args:
        pop_val (torch.Tensor): The population to select from
        assessment (Assessment): The evaluation of the features in the population

    Returns:
        torch.Tensor: The best features in the population
    """

    value = assessment.value
    if assessment.maximize:
        idx = value.argmax(0, True)
    else:
        idx = value.argmin(0, True)

    if (assessment.value.dim() != 2):
        raise ValueError('Expected assessment for each sample for each individual')
    pop_val = pop_val.view(value.shape[0], value.shape[1], -1)
    idx = idx[:, :, None].repeat(1, 1, pop_val.shape[2])
    return pop_val.gather(0, idx).squeeze(0)


def populate(x: torch.Tensor, k: int, name: str = "t") -> Population:
    """Convenience function to expand the t dimension along the population dimension

    Args:
        t (torch.Tensor): the tensor to expand
        k (int): the size of the population
        name (str, optional): the name of the value. Defaults to "t".

    Returns:
        Population: The result of the expansion
    """
    individual = Individual(**{name: x})
    populator = individual.populate(k)
    return populator(x)



def expand_k(x: torch.Tensor, k: int, reshape: bool = True) -> torch.Tensor:
    """expand the trial dimension in the tensor (separates the trial dimension from the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        k (int): The number of trials
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    shape = torch.Size([k, -1, *x.shape[1:]])
    if reshape:
        return x.reshape(shape)
    return x.view(shape)


def collapse_k(x: torch.Tensor, reshape: bool = True) -> torch.Tensor:
    """collapse the trial dimension in the tensor (merges the trial dimension with the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The collapsed tensor
    """
    if reshape:
        return x.reshape(-1, *x.shape[2:])
    return x.view(-1, *x.shape[2:])


class Indexer(object):
    """"""

    def __init__(self, idx: torch.LongTensor, k: int, maximize: bool = False):
        """initializer

        Args:
            idx (torch.LongTensor): index the tensor
            k (int): the number of samples in the population
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        """
        self.idx = idx
        self.k = k
        self.maximize = maximize

    def index(self, io: IO, detach: bool = False):
        ios = []
        for io_i in io:
            io_i = io_i.view(self.k, -1, *io_i.shape[1:])
            ios.append(gather_idx_from_population(io_i, self.idx)[0])
        return IO(*ios, detach=detach)


class RepeatSpawner(object):
    """Repeat the samples in the batch k times
    """

    def __init__(self, k: int):
        """initializer

        Args:
            k (int): the population size
        """
        self.k = k

    def __call__(self, x: torch.Tensor):

        return (
            x[None]
            .repeat(self.k, *([1] * len(x.shape)))
            .reshape(self.k * x.shape[0], *x.shape[1:])
        )

    def spawn_io(self, io: IO):
        """
        Args:
            io (IO): the io to spawn

        Returns:
            IO: The spawned IO
        """
        xs = []
        for x in io:
            if isinstance(x, torch.Tensor):
                x = self(x)
            xs.append(x)
        return IO(*xs)

    def select(self, assessment: Assessment) -> typing.Tuple[Assessment, Indexer]:
        """Select the best assessment from the tensor

        Args:
            assessment (Assessment): the assessment

        Returns:
            typing.Tuple[Assessment, Indexer]: The best assessment and the tensor
        """
        assert assessment.value.dim() == 1
        expanded = expand_k(assessment.value, self.k, False)
        if assessment.maximize:
            value, idx = expanded.max(dim=0, keepdim=True)
        else:
            value, idx = expanded.min(dim=0, keepdim=True)
        return Assessment(value, assessment.maximize), Indexer(
            idx, self.k, assessment.maximize
        )


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
