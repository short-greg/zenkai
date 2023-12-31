# 3rd party
import torch


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
