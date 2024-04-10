# 3rd party
import torch


def rand_update(
    cur: torch.Tensor,
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
    shape = cur.shape if not batch_equal else [1, *cur.shape[1:]]

    to_keep = torch.rand(shape, device=cur.device) > keep_prob
    return (~to_keep).float() * cur + to_keep.float() * original


def mix_original(
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


def update_feature(
    cur: torch.Tensor, original: torch.Tensor, limit: torch.LongTensor, dim: int=-1
) -> torch.Tensor:
    """Keep a feature dim based on indices

    Args:
        cur (Population): The population
        original (Individual): The individual used to spawn the population
        limit (torch.LongTensor): The feature indices to keep

    Returns:
        Population: The updated population
    """
    original = original.unsqueeze(1)
    original = original.clone().transpose(dim, 0)
    cur = cur.clone().transpose(dim, 0)
    cur[limit] = original
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
        cur (torch.Tensor): 
        std (torch.Tensor): 
        dim (int, optional): . Defaults to -1.
        weight (float, optional): . Defaults to 0.9.

    Returns:
        torch.Tensor: 
    """
    if var is None:
        return cur.var(dim=dim, keepdim=True)

    return (
        (1 - weight) * ((cur - mean) ** 2).mean(dim=dim, keepdim=True) + weight * var
    )


def update_momentum(
    cur_val: torch.Tensor, prev_val: torch.Tensor, momentum: torch.Tensor=None, a: float=0.9
) -> torch.Tensor:

    if momentum is None:
        return a * (cur_val - prev_val)

    return a * (cur_val - prev_val) + momentum


def decay(cur_val: torch.Tensor, prev_val: torch.Tensor=None, decay: float=0.9) -> torch.Tensor:

    if prev_val is None:
        return cur_val
    
    return cur_val + prev_val * decay


def calc_slope(val: torch.Tensor, assessment: torch.Tensor) -> torch.Tensor:

    evaluation = assessment.value[:, :, None]
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
