"""
Modules to implement exploration
on the forward pass
"""

import typing
from abc import ABC, abstractmethod

# 1st party
from typing import Any

# 3rd party
import torch
import torch.nn as nn

# local
from ..kaku import IO, Assessment
from ..utils import get_model_parameters, update_model_parameters
from .core import gather_idx_from_population, gaussian_sample

# TODO: Consider how to handle these
# Probably get rid of the first


class NoiseReplace(torch.autograd.Function):
    """
    Replace x with a noisy value. The gradInput for x will be the gradOutput and
    for the noisy value it will be x

    Note: May cause problems if only evaluating on a subset of outputs.
    The gradient may be 0 but in that case so it will set the target to
    be "noise" which is likely undesirable. In that case, use NoiseReplace2
    """

    @staticmethod
    def forward(ctx, x, noisy):
        ctx.save_for_backward(x, noisy)
        return noisy.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        x, noisy = ctx.saved_tensors
        return (noisy + grad_output) - x, None


class NoiseReplace2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, noisy):
        ctx.save_for_backward(x, noisy)
        return noisy

    @staticmethod
    def backward(ctx, grad_output):

        x, noisy = ctx.saved_tensors
        grad_input = (noisy + grad_output) - x
        direction = torch.sign(grad_input)
        magnitude = torch.min(grad_output.abs(), grad_input.abs())
        return direction * magnitude, None


class NoiseReplace3(torch.autograd.Function):
    """
    Replace x with a noisy value. The gradInput for x will be the gradOutput and
    for the noisy value it will be x.

    Uses kind of a hack with 'chosen_idx' so that all entries that are not chosen
    will be zero
    """

    @staticmethod
    def forward(ctx, x, noisy, chosen_idx):
        ctx.save_for_backward(x, noisy)
        return noisy.clone(), chosen_idx.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, chosen_idx):

        x, noisy = ctx.saved_tensors
        # grad_input_ = grad_input.gather(1, chosen_idx)
        # noisy = noisy.gather(1, chosen_idx)
        # x = x.gather(1, chosen_idx)
        # grad_output = grad_output.gather(1, chosen_idx)
        grad_input_base = (noisy + grad_output) - x

        grad_input_base = grad_input_base.view(
            chosen_idx.shape[0], chosen_idx.shape[1], -1
        )
        chosen_idx_cur = chosen_idx[:, :, None].repeat(1, 1, grad_input_base.shape[2])

        grad_input_zeros = torch.zeros_like(grad_input_base)
        grad_input_zeros.scatter_(
            1, chosen_idx_cur, grad_input_base.gather(1, chosen_idx_cur)
        )

        return grad_input_zeros.view(grad_output.shape), None, chosen_idx


class ChooseIdx(torch.autograd.Function):
    """
    Use with NoiseReplace3

    This chooses an index so that on the backpropagation only
    """

    @staticmethod
    def forward(ctx, x, chosen_idx):
        ctx.save_for_backward(chosen_idx)
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        (chosen_idx,) = ctx.saved_tensors
        return grad_output, chosen_idx


class ExplorerNoiser(nn.Module):
    """Add noise to the input"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class GaussianNoiser(ExplorerNoiser):
    """Add Gaussian noise to the exploration"""

    def __init__(self, std: float = 1.0, mu: float = 0.0):
        super().__init__()
        self.std = std
        self.mu = mu

    def forward(self, x: torch.Tensor):
        return (
            torch.randn(x.size(), dtype=x.dtype, device=x.device) * self.std + self.mu
        )


class ExplorerSelector(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, noisy: torch.Tensor):
        pass


class RandSelector(ExplorerSelector):
    """Randomly choose whether to select the noisy value or the original x"""

    def __init__(self, select_noise_prob: float):
        super().__init__()
        self.select_noise_prob = select_noise_prob

    def forward(self, x: torch.Tensor, noisy: torch.Tensor):
        assert noisy.size() == x.size()
        selected_noise = (
            torch.rand(noisy.size(), device=x.device) <= self.select_noise_prob
        )
        return (
            selected_noise.type_as(noisy) * noisy + (~selected_noise).type_as(noisy) * x
        )


class Explorer(nn.Module):
    def __init__(self, noiser: ExplorerNoiser, selector: ExplorerSelector):
        super().__init__()
        self._noiser = noiser
        self._selector = selector

    def forward(self, x: torch.Tensor):

        with torch.no_grad():
            noisy = self._selector(x, self._noiser(x))
        return NoiseReplace.apply(x, noisy)


def remove_noise(
    x: torch.Tensor, x_noisy: torch.Tensor, k: int, remove_idx: int = 0
) -> torch.Tensor:
    """Remove noise at specified index. Assumes that the trials are in dimension 0

    Args:
        x (torch.Tensor): The original tensor
        x_noisy (torch.Tensor): The tensor with noise added to it
        k (int): The
        remove_idx (int, optional): _description_. Defaults to 0.

    Returns:
        torch.Tensor: noisy tensor with value at specified indexed replaced with non-noisy version
    """
    original_shape = x_noisy.shape
    new_shape = torch.Size([k, x.shape[0] // k, *x.shape[1:]])
    x = x.reshape(new_shape)
    x_noisy = x_noisy.reshape(new_shape)
    x_noisy[remove_idx] = x[remove_idx]
    # mask_x = torch.zeros_like(x)
    # mask_noisy = torch.ones_like(x_noisy)
    # mask_x[remove_idx] = 1
    # mask_noisy[remove_idx] = 0
    # x_noisy = mask_x * x + mask_noisy * x_noisy
    return x_noisy.reshape(original_shape)


def expand_k(x: torch.Tensor, k: int, reshape: bool = True) -> torch.Tensor:
    """expand the trial dimension in the tensor (separates the trial dimension from the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        k (int): The number of trials
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    shape = torch.Size([k, -1, *x.shape[1:]])
    if reshape:
        return x.reshape(shape)
    return x.view(shape)


def collapse_k(x: torch.Tensor, reshape: bool = True) -> torch.Tensor:
    """collapse the trial dimension in the tensor (merges the trial dimension with the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The collapsed tensor
    """
    if reshape:
        return x.reshape(-1, *x.shape[2:])
    return x.view(-1, *x.shape[2:])


class Indexer(object):
    """"""

    def __init__(self, idx: torch.LongTensor, k: int, maximize: bool = False):
        """initializer

        Args:
            idx (torch.LongTensor): index the tensor
            k (int): the number of samples in the population
            maximize (bool, optional): Whether to maximize or minimize. Defaults to False.
        """
        self.idx = idx
        self.k = k
        self.maximize = maximize
        # self.spawner = spawner

    def index(self, io: IO, detach: bool = False):
        ios = []
        for io_i in io:
            io_i = io_i.view(self.k, -1, *io_i.shape[1:])
            ios.append(gather_idx_from_population(io_i, self.idx)[0])
        return IO(*ios, detach=detach)


class RepeatSpawner(object):
    """"""

    def __init__(self, k: int):
        self.k = k

    def __call__(self, x: torch.Tensor):

        return (
            x[None]
            .repeat(self.k, *([1] * len(x.shape)))
            .reshape(self.k * x.shape[0], *x.shape[1:])
        )

    def spawn_io(self, io: IO):
        """

        Args:
            io (IO): _description_

        Returns:
            _type_: _description_
        """
        xs = []
        for x in io:
            if isinstance(x, torch.Tensor):
                x = self(x)
            xs.append(x)
        return IO(*xs)

    def select(self, assessment: Assessment) -> typing.Tuple[Assessment, Indexer]:
        """

        Args:
            assessment (Assessment):

        Returns:
            typing.Tuple[Assessment, Indexer]:
        """
        assert assessment.value.dim() == 1
        expanded = expand_k(assessment.value, self.k, False)
        if assessment.maximize:
            value, idx = expanded.max(dim=0, keepdim=True)
        else:
            value, idx = expanded.min(dim=0, keepdim=True)
        return Assessment(value, assessment.maximize), Indexer(
            idx, self.k, assessment.maximize
        )


class ModuleNoise(nn.Module):
    """Use to add noise to the model that is dependent on the direction that the model is moving in"""

    def __init__(self, module_clone: nn.Module, n_instances: int, weight: float = 0.1):
        super().__init__()
        if not (0.0 < weight < 1.0):
            raise ValueError("Weight must be in range (0, 1)")
        self._module_clone = module_clone
        self._weight = weight
        self._parameters = get_model_parameters(module_clone)
        self._direction_mean = torch.zeros_like(self._parameters)
        self._direction_std = torch.zeros_like(self._parameters)
        self._updated = False
        self._n_instances = n_instances

    def update(self, base_module):
        parameters = get_model_parameters(base_module)
        dp = parameters - self._parameters
        self._direction_var = (
            1 - self._weight
        ) * self._direction_var + self._weight * (dp - self._direction_mean) ** 2

        if self._updated:
            self._direction_mean = (
                1 - self._weight
            ) * self._direction_mean + self._weight * (dp)
        else:
            self._direction_mean = dp

        self._updated = True

    def forward(self, x: torch.Tensor):
        x = x.view(self._n_instances, -1, *x.shape)
        ps = (
            torch.randn(1, *self._direction_mean.shape, dtype=x.dtype, device=x.device)
            * torch.sqrt(self._direction_var[None])
            + self._direction_mean[None]
        )
        for x_i, p_i in (x, ps):
            update_model_parameters(self._module_clone, p_i)
            ys = self._module_clone(x_i)
        ys = torch.vstack(ys)
        return ys.view(ys.shape[0] * ys.shape[1], *ys.shape)


class AssessmentDist(ABC):
    @abstractmethod
    def __call__(
        self, assessment: Assessment, x: torch.Tensor
    ) -> typing.Union[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            assessment (Assessment): the assessment. Must be of dimension [k, batch]
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
    or get the mean"""

    def __init__(self, equals_value):

        self.equals_value = equals_value

    def __call__(self, assessment: Assessment, x: torch.Tensor) -> torch.Tensor:

        value = assessment.value[:, :, None]
        if value.dim() != 3:
            raise ValueError("Value must have dimension of 3 ")
        if x.dim() != 3:
            raise ValueError("Argument x must have dimension of 3")
        equals = (x == self.equals_value).type_as(x)
        value_assessment = (equals).type_as(x) * value
        var = value_assessment.var(dim=0)
        weight = x.shape[0] / equals.sum(dim=0)
        return (
            weight * value_assessment.mean(dim=0),
            torch.sqrt(weight * var + 1e-8),
        )

    def sample(
        self, assessment: Assessment, x: torch.Tensor, n_samples: int = None
    ) -> torch.Tensor:
        mean, std = self(assessment, x)
        return gaussian_sample(mean, std, n_samples)

    def mean(self, assessment: Assessment, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self(assessment, x)
        return mean
