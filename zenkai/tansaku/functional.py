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



# local
from ..utils import get_model_parameters, update_model_parameters, expand_dim0, flatten_dim0

from ..kaku import IO, Assessment
from ..kaku import Reduction, Criterion, State, Criterion
from copy import deepcopy
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


class Objective(ABC):

    def __init__(self, maximize: bool=True) -> None:
        super().__init__()
        self.maximize = maximize

    @abstractmethod
    def __call__(self, reduction: str, **kwargs: torch.Tensor) -> Assessment:
        pass


class Constraint(ABC):
    
    @abstractmethod
    def __call__(self, **kwargs: torch.Tensor):
        pass

    def __add__(self, other: 'Constraint') -> 'CompoundConstraint':

        return CompoundConstraint([self, other])


class CompoundConstraint(Constraint):

    def __init__(self, constraints: typing.List[Constraint]) -> None:
        super().__init__()
        self.constraints = []
        for constraint in constraints:
            if isinstance(constraint, CompoundConstraint):
                self.constraints.extend(constraint.flattened)
            else: self.constraints.append(constraint)

    @property
    def flattened(self):
        return self.constraints
        
    def __call__(self, **kwargs: torch.Tensor) -> typing.Dict[str, torch.BoolTensor]:
        
        result = {}
        for constraint in self.constraints:
            cur = constraint(**kwargs)
            for key, value in cur.items():
                if key in result:
                    result[key] = value | result[key]
                elif key in cur:
                    result[key] = value
        return result


def keep_mixer(original: TensorDict, updated: TensorDict, keep_p: float) -> typing.Union[Population, Individual]:
    new_values = {}
    for k, original_v, updated_v in original.loop_over(updated, union=False):
        keep = (torch.rand_like(original_v) < keep_p).type_as(original_v)
        new_values[k] = keep * original_v + (1 - keep) * updated_v

    return original.spawn(new_values)


def keep_feature(original: Individual, population: Population, limit: torch.LongTensor):
    
    result = {}

    for k, v in population.items():
        individual_v = original[k][None].clone()
        individual_v = individual_v.repeat(v.size(0), 1, 1)
        individual_v[:, :, limit] = v[:, :, limit].detach()
        result[k] = individual_v
    return Population(**result)



# # TODO: Remove
# def cat_params(
#     params: torch.Tensor, perturbed_params: torch.Tensor, reorder: bool = False
# ):
#     """Reorder the parameters for the perturber

#     Args:
#         value (torch.Tensor): _description_
#         perturbed (torch.Tensor): _description_

#     Returns:
#         _type_: _description_
#     """
#     if params.shape != perturbed_params.shape[1:]:
#         raise RuntimeError(
#             f"The parameters shape {params.shape} does not match "
#             f"the perturbed_params shape {perturbed_params.shape}"
#         )
#     ordered = torch.cat([params[None], perturbed_params])
#     if reorder:
#         reordered = torch.randperm(len(perturbed_params) + 1, device=params.device)
#         return ordered[reordered]
#     return ordered


# # TODO: Remove
# def expand_t(t: IO, k: int) -> IO:
#     """expand the population dimension for t

#     Args:
#         t (IO): the target IO
#         k (int): the size of the population

#     Returns:
#         IO: the expanded target IO
#     """

#     ts = []
#     for t_i in t:
#         ts.append(expand_dim0(t_i, k, True))

#     return IO(*ts)


# # TODO: Remove
# def reduce_assessment_dim1(
#     assessment: Assessment, k: int, flattened: bool = True, reduction: str = "mean"
# ) -> Assessment:
#     """
#     Args:
#         assessment (Assessment): The assessment for the population
#         k (int): The size of the population
#         flattened (bool, optional): Whether the population and batch dimensions are flattened. Defaults to True.
#         reduction (str, optional): The name of the reduction.. Defaults to "mean".

#     Returns:
#         Assessment: The reduced assessment
#     """

#     if not flattened:
#         value = assessment.value.view(k * assessment.value.size(1))
#     else:
#         value = assessment.value

#     return Assessment(Reduction[reduction].sample_reduce(value).view(k, -1))


# TODO:
# add functional
# cat, topk, math
# 

# TODO: Remove
# def reduce_assessment_dim0(
#     assessment: Assessment, k: int, reduction: str = "mean"
# ) -> Assessment:
#     """
#     Args:
#         assessment (Assessment): The assessment for the population
#         k (int): The size of the population
#         reduction (str, optional): The name of the reduction. Defaults to "mean".

#     Returns:
#         Assessment: The reduced assessment
#     """
#     return Assessment(
#         Reduction[reduction].sample_reduce(assessment.value.view(k, -1).value)
#     )
