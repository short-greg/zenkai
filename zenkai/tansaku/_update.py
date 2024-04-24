# 1st party
import typing

# 3rd party
import torch
from torch import nn

# local
from ._reshape import align


def rand_update(
    cur_val: torch.Tensor,
    prev_val: torch.Tensor,
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
    shape = cur_val.shape if not batch_equal else [1, *cur_val.shape[1:]]

    to_keep = torch.rand(shape, device=cur_val.device) < keep_prob
    return (~to_keep).float() * cur_val + to_keep * prev_val


def mix_cur(
    cur_val: torch.Tensor, prev_val: torch.Tensor,
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
    keep = torch.rand_like(cur_val) ** exp_p
    return keep * cur_val + (1 - keep) * prev_val


def update_feature(
    cur_val: torch.Tensor, prev_val: torch.Tensor, limit: torch.LongTensor, dim: int=-1
) -> torch.Tensor:
    """Keep a feature dim based on indices

    Args:
        cur (Population): The population
        original (Individual): The individual used to spawn the population
        limit (torch.LongTensor): The feature indices to keep

    Returns:
        Population: The updated population
    """
    prev_val = prev_val.clone().transpose(dim, 0)
    cur_val = cur_val.clone().transpose(dim, 0)
    cur_val[limit] = prev_val[limit]
    return cur_val.transpose(dim, 0)


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


class Updater(nn.Module):

    def __init__(self, update_f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None, *args, **kwargs):
        """Module that handles updating a tensor with an update function

        Args:
            update_f (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): The update function. If it is None, the default behavior will be to do nothing but the behavior can be overridden by sublcasses. Defaults to None.
        """

        super().__init__()
        self.update_f = update_f
        self.args = args
        self.kwargs = kwargs
        self.cur_val = None

    def forward(self, x: torch.Tensor):

        if self.cur_val is None:
            self.cur_val = x

        elif self.update_f is not None:
            self.cur_val = self.update_f(
                x, self.cur_val, *self.args, **self.kwargs
            )
            return self.cur_val
        return x


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


def calc_scale(cur_val: torch.Tensor, ref: torch.Tensor, scale: float=None) -> torch.Tensor:
    """
    Args:
        cur_val (torch.Tensor): The cur val
        ref (torch.Tensor): The reference to set it to
        scale (float, optional): The amount to scale the rate by. Should be less than 1. Defaults to None.

    Returns:
        torch.Tensor: The resulting scale to multiply by
    """

    rate = (ref.abs() / cur_val.abs()) 
    if scale is not None:
        rate = rate * scale
    return rate
