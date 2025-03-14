# 3rd party
import torch

# local
from ..utils._shape import unsqueeze_to


def pop_mean(
    x: torch.Tensor, 
    norm_weight: torch.Tensor=None, 
    dim: int=0, 
    keepdim: bool=True
) -> torch.Tensor:
    """Calculate the mean of a population

    Args:
        x (torch.Tensor): Tensor to calculate the mean on
        weight (torch.Tensor, optional): The weights to use if computing the weighted mean (sum to 1). Defaults to None.
        dim (int, optional): The dim to calculate the mean on. Defaults to 0.

    Returns:
        torch.Tensor: The mean of the tensor
    """

    if norm_weight is None:
        return x.mean(dim=dim, keepdim=keepdim)
    
    norm_weight = unsqueeze_to(
        norm_weight, x
    )
    return (x * norm_weight).sum(dim=dim, keepdim=keepdim)


def pop_quantile(
    x: torch.Tensor, q: float, 
    norm_weight: torch.Tensor=None, 
    dim: int=0, keepdim: bool=True
) -> torch.Tensor:
    """Calculate the quantile of a population

    Note: No interpolation done if using weights

    Args:
        x (torch.Tensor): The x to get the median for
        q (float): The quantile to retrieve for
        norm_weight (torch.Tensor, optional): Weights to use if computing the weighted quantile. Defaults to None.
        dim (int, optional): Dim to calculate the median on. Defaults to 0.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to True.

    Returns:
        torch.Tensor: The median
    """

    if norm_weight is None:
        return x.quantile(q, dim=dim, keepdim=keepdim, interpolation='nearest')
    
    # TODO: Currently not working so fix
    sorted_values, sorted_idx = x.sort(dim=dim)
    sorted_weight = norm_weight.gather(dim, sorted_idx)

    sorted_weight = sorted_weight.cumsum(dim)

    # When using max with torch, the first value that
    # maximizes it will be returned
    _, median_idx = (
        sorted_weight >= q
    ).float().max(dim=dim, keepdim=True)
    result, result_idx = sorted_values.gather(
        dim, median_idx
    ), sorted_idx.gather(dim, median_idx)

    if keepdim is False:
        return result.squeeze(dim), result_idx.squeeze(dim)
    
    return result, result_idx


def pop_median(x: torch.Tensor, norm_weight: torch.Tensor=None, dim: int=0, keepdim: bool=True) -> torch.Tensor:
    """Calculate the median of a population

    Note: No interpolation done if using weights

    Args:
        x (torch.Tensor): The x to get the median for
        norm_weight (torch.Tensor, optional): Weights to use if computing the weighted median. Defaults to None.
        dim (int, optional): Dim to calculate the median on. Defaults to 0.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to True.

    Returns:
        torch.Tensor: The median
    """
    if norm_weight is None:
        return x.median(dim=dim, keepdim=keepdim)
    
    return pop_quantile(
        x, 0.5, norm_weight, dim, keepdim
    )


def pop_normalize(
    x: torch.Tensor, 
    mean: torch.Tensor=None, 
    std: torch.Tensor=None, 
    dim: int=0, 
    eps: float=1e-6
) -> torch.Tensor:
    """Normalize the population based on the mean and standard deviation

    Args:
        x (torch.Tensor): The tensor to normalize mean (torch.Tensor, optional): The mean to use, if not defined will use the mean of x. Defaults to None.
        std (torch.Tensor, optional): The standard deviation, if not. Defaults to None.
        dim (int, optional): The dimension to normalize on. Defaults to 0.

    Returns:
        torch.Tensor: The normalized tensor
    """
    mean = x.mean(dim, keepdim=True) if mean is None else mean
    std = x.std(dim, keepdim=True) if std is None else std

    return (x - mean) / (std + eps)
