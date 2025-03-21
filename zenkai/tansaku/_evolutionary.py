# 1st party
import typing

# 3rd party
import torch

# local
from ._aggregate import pop_normalize, pop_mean
from ..utils._shape import unsqueeze_to


def es_dx(dw: torch.Tensor, assessment: torch.Tensor, assessment_ref: typing.Union[torch.Tensor, str] = 'mean', pop_dim: int=0) -> torch.Tensor:
    """Use to calculate the change due to evolution strategy

    Args:
        dw (torch.Tensor): The displacement used for w
        assessment (torch.Tensor): The assessment for the population
        assessment_ref (typing.Union[torch.Tensor, str], optional): The reference for the assessment. This is used for normalization. Defaults to 'mean'.
        pop_dim (int, optional): . Defaults to 0.

    Returns:
        torch.Tensor: The estimation of dx using evolution strategies
    """
    assessment = pop_normalize(
        assessment, assessment_ref if assessment_ref != 'mean' else None, dim=pop_dim
    )
    assessment = unsqueeze_to(assessment, dw)
    return pop_mean(dw * assessment, keepdim=False)
