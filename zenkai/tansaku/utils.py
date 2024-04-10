# 1st party
import typing

# 3rd party
import torch

# local
from ..kaku import Population, Individual, TensorDict


def gather_idx_from_population(pop: torch.Tensor, idx: torch.LongTensor):
    """Retrieve the indices from population. idx is a 2 dimensional tensor"""
    repeat_by = [1] * len(idx.shape)
    for i, sz in enumerate(pop.shape[2:]):
        idx = idx.unsqueeze(i + 2)
        repeat_by.append(sz)
    idx = idx.repeat(*repeat_by)
    return pop.gather(0, idx)


# TODO: Remove
def gen_like(
    f, k: int, orig_p: torch.Tensor, requires_grad: bool = False
) -> typing.Dict:
    """generate a tensor like another

    Args:
        f (_type_): _description_
        k (int): _description_
        orig_p (torch.Tensor): _description_
        requires_grad (bool, optional): _description_. Defaults to False.

    Returns:
        typing.Dict: _description_
    """
    return f(
        [k] + [*orig_p.shape[1:]],
        dtype=orig_p.dtype,
        device=orig_p.device,
        requires_grad=requires_grad,
    )


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


def keep_original(
    candidate: torch.Tensor,
    original: torch.Tensor,
    keep_prob: float,
    batch_equal: bool = False,
) -> torch.Tensor:
    """Will keep the original given the probability passed in with keep_prob

    Args:
        candidate (torch.Tensor): the candidate (i.e. updated) value
        original (torch.Tensor): the original value
        keep_prob (float): the probability with which to keep the features in the batch
        batch_equal (bool): whether to have updates be the same for all samples in the batch
    Returns:
        torch.Tensor: the updated tensor
    """
    shape = candidate.shape if not batch_equal else [1, *candidate.shape[1:]]

    to_keep = torch.rand(shape, device=candidate.device) > keep_prob
    return (~to_keep).float() * candidate + to_keep.float() * original


def keep_mixer(
    original: torch.Tensor, updated: torch.Tensor, keep_p: float
) -> torch.Tensor:
    """Keep a certain percentage of values randomly chosen based on keep_p

    Args:
        original (TensorDict): the original tensor dict
        updated (TensorDict): the updated tensor dict
        keep_p (float): The rate to keep values at

    Returns:
        typing.Union[Population, Individual]: The updated tensor dict
    """
    keep = (torch.rand_like(original) < keep_p)
    return keep * original + (~keep) * updated


def keep_feature(
    original: torch.Tensor, population: torch.Tensor, limit: torch.LongTensor, dim: int=-1
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
    
    original = original.unsqueeze(1)
    original = original.clone().transpose(dim, 0)
    population = population.clone().transpose(dim, 0)
    population[limit] = original
    return population.transpose(dim, 0)


# def populate(x: torch.Tensor, k: int, name: str = "t") -> Population:
#     """Convenience function to expand the t dimension along the population dimension

#     Args:
#         t (torch.Tensor): the tensor to expand
#         k (int): the size of the population
#         name (str, optional): the name of the value. Defaults to "t".

#     Returns:
#         Population: The result of the expansion
#     """
#     individual = Individual(**{name: x})
#     populator = individual.populate(k)
#     return populator(x)

