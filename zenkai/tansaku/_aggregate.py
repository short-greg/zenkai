import torch

from ._utils import unsqueeze_to


def mean(x: torch.Tensor, norm_weight: torch.Tensor=None, dim: int=0, keepdim: bool=True) -> torch.Tensor:
    """Calculate the mean of x

    Args:
        x (torch.Tensor): Tensor to calculate the mean on
        norm_weight (torch.Tensor, optional): Normalized weights to use (sum to 1) if computing the weighted mean. Defaults to None.
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


def median(x: torch.Tensor, norm_weight: torch.Tensor=None, dim: int=0, keepdim: bool=True) -> torch.Tensor:
    """Calculate the median. 

    Note: No interpolation done if using weights

    Args:
        x (torch.Tensor): _description_
        norm_weight (torch.Tensor, optional): Weights to use if computing the weighted median. Defaults to None.
        dim (int, optional): Dim to calculate the median on. Defaults to 0.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to True.

    Returns:
        torch.Tensor: The median
    """

    if norm_weight is None:
        return x.median(dim=dim, keepdim=keepdim)
    
    sorted_weight, idx = norm_weight.sort(dim=dim)
    idx = torch.abs(sorted_weight - 0.5).min(dim=dim, keep_dim=True)[1]
    result = torch.gather(x, dim, idx)
    if keepdim:
        return result
    
    return result.squeeze(dim)


def quantile(x: torch.Tensor, quantile: float, norm_weight: torch.Tensor=None, dim: int=0, keepdim: bool=True) -> torch.Tensor:
    """Calculate the specified quantile. 

    Note: No interpolation done if using weights

    Args:
        x (torch.Tensor): _description_
        norm_weight (torch.Tensor, optional): Weights to use if computing the weighted quantile. Defaults to None.
        dim (int, optional): Dim to calculate the median on. Defaults to 0.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to True.

    Returns:
        torch.Tensor: The median
    """

    if norm_weight is None:
        return x.quantile(quantile, dim=dim, keepdim=keepdim)
    
    sorted_weight, idx = norm_weight.sort(dim=dim)
    idx = torch.abs(sorted_weight - quantile).min(dim=dim, keep_dim=True)[1]
    result = torch.gather(x, dim, idx)
    if keepdim:
        return result
    return result.squeeze(dim)

