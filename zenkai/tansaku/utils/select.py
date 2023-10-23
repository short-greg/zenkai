

import torch


# 1st party
import typing
from abc import ABC, abstractmethod
import functools

# 3rd party
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch

# local
from ...kaku import State, Population, Individual, TensorDict


# local
from ...utils import get_model_parameters, update_model_parameters, expand_dim0, flatten_dim0, gather_idx_from_population

from ...kaku import IO, Assessment
from .generate import expand_k
from ...kaku import Reduction, Criterion, State, Criterion

from copy import deepcopy


import torch

from ...kaku.assess import Assessment
from abc import abstractmethod, ABC

# TODO: Move to utils


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

    def select(self, assessment: Assessment) -> typing.Tuple[Assessment, 'Indexer']:
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

    def select_index(self, tensor_dict: TensorDict) -> typing.Union['TensorDict', typing.Tuple['TensorDict']]:
        
        if len(self) == 1:
            result = {}
            for k, v in tensor_dict.items():
                result[k] = self(v)
            return tensor_dict.spawn(result)

        result = []
        for i in range(len(self.index)):
            cur_result = {}
            for k, v in tensor_dict.items():
                cur_result[k] = self.index_for(i, v)
            result.append(tensor_dict.spawn(cur_result))
        return tuple(result)



class Selector(ABC):
    """Use to select indices from a multidimensional tensor. Only works for dimension 0 so must be reshaped
    """

    @abstractmethod
    def select(self, assessment: Assessment) -> 'IndexMap':
        pass

    def __call__(self, assessment: Assessment) -> 'IndexMap':
        
        return self.select(assessment)


class TopKSelector(Selector):

    def __init__(self, k: int, dim: int=0):
        self.k = k
        self.dim = dim

    def select(self, assessment: Assessment) -> IndexMap:
        
        _, topk = assessment.value.topk(self.k, dim=self.dim, largest=assessment.maximize)
        return IndexMap(topk, dim=0)


class BestSelector(Selector):

    def __init__(self, k: int, dim: int=0, largest: bool=True):
        self.k = k
        self.dim = dim

    def select(self, assessment: Assessment) -> IndexMap:
        
        if assessment.maximize:
            _, best = assessment.value.max(self.k, dim=self.dim, keepdim=True)
        else:
            _, best = assessment.value.min(self.k, dim=self.dim, keepdim=True)
        return IndexMap(best, dim=0)


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

