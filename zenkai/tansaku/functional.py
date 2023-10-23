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
