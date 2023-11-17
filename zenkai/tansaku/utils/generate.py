# 1st party
import typing

# 3rd party
import torch

# local
from ...kaku import Population, Individual


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
