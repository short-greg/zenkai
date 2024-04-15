# 3rd party
import torch

# local
from ._utils import align


def rand_update(
    cur: torch.Tensor,
    prev: torch.Tensor,
    keep_prob: float,
    batch_equal: bool = False,
) -> torch.Tensor:
    """Will keep the original given the probability passed in with keep_prob

    Args:
        candidate (torch.Tensor): the candidate (i.e. updated) value
        prev (torch.Tensor): the original value
        keep_prob (float): the probability with which to keep the features in the batch
        batch_equal (bool): whether to have updates be the same for all samples in the batch
    Returns:
        torch.Tensor: the updated tensor
    """
    shape = cur.shape if not batch_equal else [1, *cur.shape[1:]]

    to_keep = torch.rand(shape, device=cur.device) < keep_prob
    return (~to_keep).float() * cur + to_keep * prev


def mix_cur(
    cur: torch.Tensor, prev: torch.Tensor,
    exp_p: float=1.0
) -> torch.Tensor:
    """Keep a certain percentage of values randomly chosen based on keep_p

    Args:
        cur (TensorDict): the original tensor dict
        updated (TensorDict): the updated tensor dict
        keep_p (float): The rate to keep values at

    Returns:
        typing.Union[Population, Individual]: The updated tensor dict
    """
    keep = torch.rand_like(cur) ** exp_p
    return keep * cur + (1 - keep) * prev


def update_feature(
    cur: torch.Tensor, prev: torch.Tensor, limit: torch.LongTensor, dim: int=-1
) -> torch.Tensor:
    """Keep a feature dim based on indices

    Args:
        cur (Population): The population
        original (Individual): The individual used to spawn the population
        limit (torch.LongTensor): The feature indices to keep

    Returns:
        Population: The updated population
    """
    prev = prev.clone().transpose(dim, 0)
    cur = cur.clone().transpose(dim, 0)
    cur[limit] = prev[limit]
    return cur.transpose(dim, 0)


def update_mean(
    cur: torch.Tensor, mean: torch.Tensor=None, dim: int=-1, weight: float=0.9
) -> torch.Tensor:
    """Update the mean

    Args:
        cur (torch.Tensor): The cur value
        mean (torch.Tensor): The cur mean
        dim (int, optional): The dim to calculate the mean on. Defaults to -1.
        weight (float, optional): The weight on the mean. Defaults to 0.9.

    Returns:
        torch.Tensor: 
    """
    cur_mean = cur.mean(dim=dim, keepdim=True)
    if mean is None:
        return cur_mean

    return (
        (1 - weight) * cur_mean + weight * mean
    )


def update_var(
    cur: torch.Tensor, mean: torch.Tensor, var: torch.Tensor=None, dim: int=-1, weight: float=0.9
) -> torch.Tensor:
    """Update the mean

    Args:
        cur (torch.Tensor): The cur value to update
        mean (torch.Tensor): The Current mean
        var (torch.Tensor): The Current variance
        dim (int, optional): . Defaults to -1.
        weight (float, optional): . Defaults to 0.9.

    Returns:
        torch.Tensor: 
    """
    cur_var = ((cur - mean) ** 2).mean(dim=dim, keepdim=True) 
    if var is None:
        return cur_var

    return (
        (1 - weight) * cur_var + weight * var
    )


def update_momentum(
    cur_val: torch.Tensor, prev_val: torch.Tensor, momentum: torch.Tensor=None, a: float=0.9
) -> torch.Tensor:
    """Update the momentum

    Args:
        cur_val (torch.Tensor): The cur value of the value to calc momentum for
        prev_val (torch.Tensor): The prev value of the value to calc momentum for
        momentum (torch.Tensor, optional): The current momentum value. Defaults to None.
        a (float, optional): The momentum param. Defaults to 0.9.

    Returns:
        torch.Tensor: The updated momentum value
    """
    if momentum is None:
        return a * (cur_val - prev_val)

    return a * (cur_val - prev_val) + momentum


def decay(cur_val: torch.Tensor, prev_val: torch.Tensor=None, decay: float=0.9) -> torch.Tensor:
    """

    Args:
        cur_val (torch.Tensor): The current value
        prev_val (torch.Tensor, optional): The previous value output by decay. Defaults to None.
        decay (float, optional): The amount to decay the previous value by. Defaults to 0.9.

    Returns:
        torch.Tensor: _description_
    """

    if prev_val is None:
        return cur_val
    
    return cur_val + prev_val * decay


def calc_slope(val: torch.Tensor, assessment: torch.Tensor) -> torch.Tensor:
    """Calculate the slope based on the assessment

    Args:
        val (torch.Tensor): The value
        assessment (torch.Tensor): The assessment for the value. Must have fewer or the same number of dimensions as val but not be 0-dim

    Returns:
        torch.Tensor: The slope
    """

    evaluation = align(assessment, val)
    base_shape = val.shape
    val = val.view(
        val.size(0), val.size(1), -1
    )
    ssx = (val**2).sum(0) - (1 / len(val)) * (val.sum(0)) ** 2
    ssy = (val * evaluation).sum(0) - (1 / len(val)) * (
        (val.sum(0) * evaluation.sum(0))
    )
    slope = ssy / ssx
    return slope.reshape(base_shape)
