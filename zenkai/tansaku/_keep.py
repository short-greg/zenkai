# 1st party
import typing

# 3rd party
import torch

# local
from ..kaku import Population, Individual, TensorDict


def keep_mixer(
    original: TensorDict, updated: TensorDict, keep_p: float
) -> typing.Union[Population, Individual]:
    """Keep a certain percentage of values randomly chosen based on keep_p

    Args:
        original (TensorDict): the original tensor dict
        updated (TensorDict): the updated tensor dict
        keep_p (float): The rate to keep values at

    Returns:
        typing.Union[Population, Individual]: The updated tensor dict
    """
    new_values = {}
    for k, original_v, updated_v in original.loop_over(updated, union=False):
        keep = (torch.rand_like(original_v) < keep_p).type_as(original_v)
        new_values[k] = keep * original_v + (1 - keep) * updated_v

    return original.spawn(new_values)


def keep_feature(
    original: Individual, population: Population, limit: torch.LongTensor
) -> Population:
    """Keep a feature dim based on indices

    Args:
        original (Individual): The individual used to spawn the population
        population (Population): The population
        limit (torch.LongTensor): The feature indices to keep

    Returns:
        Population: The updated population
    """
    result = {}

    for k, v in population.items():
        individual_v = original[k][None].clone()
        individual_v = individual_v.repeat(v.size(0), 1, 1)
        individual_v[:, :, limit] = v[:, :, limit].detach()
        result[k] = individual_v
    return Population(**result)
