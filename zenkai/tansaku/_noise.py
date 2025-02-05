"""
Modules to implement exploration
on the forward pass
"""

# 1st party
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 3rd party
import torch
import torch.nn as nn

# # local
# from ..utils._params import get_params, update_model_params


def gaussian_sample(
    mean: torch.Tensor, std: torch.Tensor, k: int = None
) -> torch.Tensor:
    """generate a sample from a gaussian

    Args:
        mean (torch.Tensor): _description_
        std (torch.Tensor): _description_
        k (int): The number of samples to generate. If None will generate 1 sample and the dimension
         will not be expanded

    Returns:
        torch.Tensor: The sample or samples generated
    """
    if k is None:
        return torch.randn_like(mean) * std + mean

    if k <= 0:
        raise ValueError(f"Argument {k} must be greater than 0")
    return (
        torch.randn([k, *mean.shape], device=mean.device, dtype=mean.dtype)
        * std[None]
        + mean[None]
    )


def gaussian_noise(x: torch.Tensor, std: float=1.0, mean: float=0.0) -> torch.Tensor:
    """Add Gaussian noise to the input

    Args:
        x (torch.Tensor): The input
        std (float, optional): The standard deviation for the noise. Defaults to 1.0.
        mean (float, optional): The bias for the noise. Defaults to 0.0.

    Returns:
        torch.Tensor: The input with noise added
    """

    return x + torch.randn_like(x) * std + mean


def binary_noise(x: torch.Tensor, flip_p: bool = 0.5, signed_neg: bool = True) -> torch.Tensor:
    """Flip the input features with a certain probability

    Args:
        x (torch.Tensor): The input features
        flip_p (bool, optional): The probability to flip with. Defaults to 0.5.
        signed_neg (bool, optional): Whether negative is signed or 0. Defaults to True.

    Returns:
        torch.Tensor: The input with noise added
    """
    to_flip = torch.rand_like(x) > flip_p
    if signed_neg:
        return to_flip.float() * -x + (~to_flip).float() * x
    return (x - to_flip.float()).abs()


@dataclass
class TInfo:
    """Dataclass to store the information for a tensor
    """

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device

    @property
    def attr(self) -> typing.Dict:
        return {
            'dtype': self.dtype,
            'device': self.device
        }


def add_noise(x: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, TInfo], torch.Tensor], pop_dim: int=0, expand: bool=True) -> torch.Tensor:
    """Add noise to a regular tensor

    Args:
        x (torch.Tensor): The input to add noise to
        k (int): The size of the population
        f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
        pop_dim (int, optional): The population dim. Defaults to 0.
        expand (bool): Whether to expand the population dimension before executing the function

    Returns:
        torch.Tensor: The noise added to the Tensor
    """
    shape = list(x.shape)
    shape.insert(pop_dim, k)
    x = x.unsqueeze(pop_dim)
    if expand:
        expand_shape = [1] * len(shape)
        expand_shape[pop_dim] = k
        x = x.repeat(expand_shape)

    return f(x, TInfo(shape, x.dtype, x.device))


def cat_noise(x: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, TInfo], torch.Tensor], pop_dim: int=0, expand: bool=True) -> torch.Tensor:
    """Add noise to a regular tensor and then concatenate
    the original tensor

    Args:
        x (torch.Tensor): The input to add noise to
        k (int): The size of the population
        f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
        expand (bool): Whether to expand the population dimension before executing the function

    Returns:
        torch.Tensor: The noise added to the Tensor
    """
    shape = list(x.shape)
    shape.insert(pop_dim, k)
    x = x.unsqueeze(pop_dim)

    if expand:
        expand_shape = [1] * len(shape)
        expand_shape[pop_dim] = k
        x_in = x.repeat(expand_shape)
    else:
        x_in = x
    out = f(x_in, TInfo(shape, x.dtype, x.device))
    return torch.cat(
        [x, out], dim=pop_dim
    )


def add_pop_noise(pop: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, TInfo], torch.Tensor], pop_dim: int=0) -> torch.Tensor:
    """Add noise to a population tensor

    Args:
        x (torch.Tensor): The input to add noise to
        k (int): The number to generate for each member of the population
        f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
        pop_dim (int, optional): The population dim. Defaults to 1.

    Returns:
        torch.Tensor: The noise added to the Tensor
    """
    shape = list(pop.shape)
    base_shape = list(pop.shape)
    shape.insert(pop_dim + 1, k)
    base_shape[pop_dim] = base_shape[pop_dim] * k

    pop = pop.unsqueeze(pop_dim + 1)

    y = f(pop, TInfo(shape, pop.dtype, pop.device))
    return y.reshape(base_shape)


def cat_pop_noise(x: torch.Tensor, k: int, f: typing.Callable[[torch.Tensor, TInfo], torch.Tensor], pop_dim: int=0):
    """Add noise to a population tensor and then
    concatenate to the original tensor

    Args:
        x (torch.Tensor): The input to add noise to
        k (int): The size of the population
        f (typing.Callable[[torch.Tensor, torch.Size, torch.dtype, torch.device], torch.Tensor]): The noise function
        pop_dim (int, optional): The population dim. Defaults to 1.

    Returns:
        torch.Tensor: The noise added to the Tensor
    """
    shape = list(x.shape)
    base_shape = list(x.shape)
    shape.insert(pop_dim + 1, k)
    base_shape[pop_dim] = base_shape[pop_dim] * k

    x = x.unsqueeze(pop_dim + 1)

    y = f(x, TInfo(shape, x.dtype, x.device))
    out = y.reshape(base_shape)
    return torch.cat(
        [x.squeeze(pop_dim + 1), out], dim=pop_dim
    )


class AssessmentDist(ABC):
    """
    Class that is used to calculate a distribution based on the input and assessment
    """

    @abstractmethod
    def __call__(
        self, assessment: torch.Tensor, x: torch.Tensor, maximize: bool=False
    ) -> typing.Union[torch.Tensor, torch.Tensor]:
        """

        Args:
            assessment (torch.Tensor): the assessment. Must be of dimension [k, batch]
            x (torch.Tensor): the input to assess. must be of dimension
              [k, batch, feature]

        Returns:
            typing.Union[torch.Tensor, torch.Tensor]:
              The mean of the assessment, the standard deviation of the
              assessment
        """
        pass


class EqualsAssessmentDist(AssessmentDist):
    """Determine the distribution of the assessment to draw samples
    or get the mean. Use for binary or disrete sets"""

    def __init__(self, equals_value):
        """Create a distribution based on whether the assessment equals a value. Use primarily for categorical or binary-valued tensors

        Args:
            equals_value: The value to get the distribution for
        """

        self.equals_value = equals_value

    def __call__(self, assessment: torch.Tensor, x: torch.Tensor, maximize: bool=False) -> torch.Tensor:
        """Calculate the assessment distribution of the input

        Args:
            assessment (torch.Tensor): The assessment of the
            x (torch.Tensor): the input tensor
            maximize (bool): whether to maximize or minimize
        Raises:
            ValueError: The dimension of value is not 3
            ValueError: The dimension of x is not 3

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor] : mean, std
        """
        if assessment.dim() != 2:
            raise ValueError("Value must have dimension of 2 ")
        if x.dim() == 3:
            value = assessment[:, :, None]
        else:
            value = assessment
        if x.dim() not in (2, 3):
            raise ValueError("Argument x must have dimension of 2 or 3")
        equals = (x == self.equals_value).type_as(x)
        value_assessment = (equals).type_as(x) * value
        var = value_assessment.var(dim=0)
        weight = x.shape[0] / equals.sum(dim=0)
        return (
            weight * value_assessment.mean(dim=0),
            torch.sqrt(weight * var + 1e-8),
        )

    def sample(
        self, assessment: torch.Tensor, x: torch.Tensor, n_samples: int = None
    ) -> torch.Tensor:
        """Generate a sample from the distribution

        Args:
            assessment (torch.Tensor): The assessment
            x (torch.Tensor): The input
            n_samples (int, optional): the number of samples. Defaults to None.

        Returns:
            torch.Tensor: The sample value for the input
        """
        mean, std = self(assessment, x)
        return gaussian_sample(mean, std, n_samples)

    def mean(self, assessment: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calculate the mean from the distribution

        Args:
            assessment (Assessment): The assessment of the population
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The mean value for the input
        """
        mean, _ = self(assessment, x)
        return mean


class FreezeDropout(nn.Module):
    """Freeze the dropout parameter so that the same parameter will be used """

    def __init__(self, p: float, freeze: bool = False):
        """Create a FreezeDropout to keep the parameter frozen. This is useful if you want
        to go through the network multiple times and get the same output

        Args:
            p (float): The dropout rate
            freeze (bool, optional): Whether to freeze the dropout. Defaults to False.

        Raises:
            ValueError: If p is greater or equal to one or less than zero
        """
        super().__init__()
        if p >= 1.0 or p < 0.0:
            raise ValueError(f"P must be in range [0.0, 1.0) not {p}")
        self.p = p
        self.freeze = freeze
        self._cur = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute dropout on the input

        Args:
            x (torch.Tensor): The input to dropout

        Returns:
            torch.Tensor: The 
        """
        if self.p == 0.0:
            return x

        if not self.training:
            return x

        if self.freeze and self._cur is not None:
            f = self._cur
        else:
            f = (torch.rand_like(x) > self.p).type_as(x)

        self._cur = f
        return (f * x) * (1 / 1 - self.p)


def binary_prob(
    x: torch.Tensor, loss: torch.Tensor, retrieve_counts: bool = False
) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Calculate binary probability based on a binary-valued vector input

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

