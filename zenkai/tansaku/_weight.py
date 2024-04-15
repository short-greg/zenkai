import torch
import torch.nn.functional
import math

from ._reshape import unsqueeze_vector


def normalize_weight(weight: torch.Tensor, pop_dim: int=0) -> torch.Tensor:

    return weight / weight.sum(dim=pop_dim, keepdim=True)


def softmax_weight(weight: torch.Tensor, pop_dim: int=0, maximize: bool=False) -> torch.Tensor:
    """

    Args:
        weight (torch.Tensor): The weight
        pop_dim (int, optional): The population dim. Defaults to 0.
        maximize (bool, optional): Whether to maximize or minimize the weight. Defaults to False.

    Returns:
        torch.Tensor: Normalized weights
    """

    if maximize:
        return torch.softmax(
            weight, dim=pop_dim
        )
    return torch.nn.functional.softmin(
        weight, dim=pop_dim
    )


def rank_weight(weight: torch.Tensor, pop_dim: int=0, maximize: bool=False) -> torch.Tensor:
    """
    Args:
        weight (torch.Tensor): The original weight
        pop_dim (int, optional): The population dim. Defaults to 0.
        maximize (bool, optional): Whether to maximize or minimize. Defaults to False.

    Returns:
        torch.Tensor: Weights based on the ranking
    """
    _, ind = weight.sort(
        dim=pop_dim, descending=not maximize
    )

    # Create the ranks
    ranks = torch.arange(
        1, weight.shape[pop_dim] + 1, 1, device=weight.device
    )
    ranks = unsqueeze_vector(ranks, weight)

    shape = list(weight.shape)
    shape[pop_dim] = 1
    ranks = ranks.repeat(shape)
    return ranks.gather(pop_dim, ind).float()


def log_weight(norm_weight: torch.Tensor, maximize: bool=False, eps: float=1e-7) -> torch.Tensor:
    """Use the log scale to calculate weights

    Args:
        norm_weight (torch.Tensor): Values passed in must be normalized weights

    Returns:
        torch.Tensor: Unnormalized weights calculated on the log scale
    """
    if not maximize:
        norm_weight = 1 - norm_weight

    return -torch.log(norm_weight + eps)
    

def gauss_cdf_weight(weight: torch.Tensor, pop_dim: int=0) -> torch.Tensor:
    """Calculate the weight using the Gaussian CDF

    Args:
        weight (torch.Tensor): The weights
        pop_dim (int, optional): _description_. Defaults to 0.

    Returns:
        torch.Tensor: Weights calcualted using the Gaussian CDF
    """
    
    mean = weight.mean(dim=pop_dim, keepdim=True)
    scale = weight.std(dim=pop_dim, keepdim=True)

    return (0.5 * (1 + torch.erf(weight - mean) * scale.reciprocal() / math.sqrt(2)))
