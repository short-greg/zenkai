"""
Modules to implement exploration
on the forward pass
"""

# 1st party
import typing
from abc import ABC, abstractmethod

# 3rd party
import torch
import torch.nn as nn

# local
from ..kaku import Assessment
from ..utils import get_model_parameters, update_model_parameters


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
    if k is not None:
        if k <= 0:
            raise ValueError(f"Argument {k} must be greater than 0")
        return (
            torch.randn([k, *mean.shape], device=mean.device, dtype=mean.dtype)
            * std[None]
            + mean[None]
        )
    return torch.randn_like(mean) * std + mean


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


class ExplorerNoiser(nn.Module):
    """Add noise to the input"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class GaussianNoiser(ExplorerNoiser):
    """Add Gaussian noise to the exploration"""

    def __init__(self, std: float = 1.0, mu: float = 0.0):
        super().__init__()
        if std < 0:
            raise ValueError(f"Standard deviation must be greater than 0 not {std}")
        self.std = std
        self.mu = mu

    def forward(self, x: torch.Tensor):
        return (
            torch.randn(x.size(), dtype=x.dtype, device=x.device) * self.std + self.mu
        )


class ExplorerSelector(nn.Module):
    """Use to select the noise or the output"""

    @abstractmethod
    def forward(self, x: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        pass


class RandSelector(ExplorerSelector):
    """Randomly choose whether to select the noisy value or the original x"""

    def __init__(self, select_noise_prob: float):
        """initializer

        Args:
            select_noise_prob (float): The probability that
        """
        super().__init__()
        self.select_noise_prob = select_noise_prob

    def forward(self, x: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """Randomly select the noise or the input tensor

        Args:
            x (torch.Tensor): the input tensor to add noise to
            noisy (torch.Tensor): the noisy tensor

        Returns:
            torch.Tensor: the noisy tensor
        """

        selected_noise = (
            torch.rand(noisy.size(), device=x.device) <= self.select_noise_prob
        )
        return (
            selected_noise.type_as(noisy) * noisy + (~selected_noise).type_as(noisy) * x
        )


class Explorer(nn.Module):
    """
    Explorer is used to explore different inputs to feed into a Module
    """

    def __init__(self, noiser: ExplorerNoiser, selector: ExplorerSelector):
        """Instantiate the explorer with a noiser and selector

        Args:
            noiser (ExplorerNoiser): The noiser to use to add exploration to the input space
            selector (ExplorerSelector): The selector to use for selecting from the noise
        """
        super().__init__()
        self._noiser = noiser
        self._selector = selector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: The input tensor with noise added
        """

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


class ModuleNoise(nn.Module):
    """Use to add noise to the model that is dependent on the direction that the model is moving in"""

    def __init__(self, module_clone: nn.Module, n_instances: int, weight: float = 0.1):
        """initializer

        Args:
            module_clone (nn.Module): Clone of the model to add noise to
            n_instances (int): The number of model instances
            weight (float, optional): The weight on momentum. Defaults to 0.1.

        Raises:
            ValueError: If weight is an invalid value
        """
        super().__init__()
        if not (0.0 < weight < 1.0):
            raise ValueError("Weight must be in range (0, 1)")
        self._module_clone = module_clone
        self._weight = weight
        self._p = get_model_parameters(module_clone)
        self._direction_mean = torch.zeros_like(self._p)
        self._direction_var = torch.zeros_like(self._p)
        self._updated = False
        self._n_instances = n_instances

    def update(self, base_module):
        """Update the base model by weighting it with the current direction

        Args:
            base_module: The module to update
        """

        parameters = get_model_parameters(base_module)
        dp = parameters - self._p
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the noisy output from the input

        TODO: Make work with varying numbers of xs

        Args:
            x (torch.Tensor): The input dimensions[population, sample, *feature]

        Returns:
            torch.Tensor: The output
        """
        x = x.view(self._n_instances, -1, *x.shape[1:])
        ps = (
            torch.randn(
                self._n_instances,
                *self._direction_mean.shape,
                dtype=x.dtype,
                device=x.device,
            )
            * torch.sqrt(self._direction_var[None])
            + self._direction_mean[None]
        ) + get_model_parameters(self._module_clone)[None]
        ys = []
        for x_i, p_i in zip(x, ps):
            update_model_parameters(self._module_clone, p_i)
            ys.append(self._module_clone(x_i))

        return torch.cat(ys)


class AssessmentDist(ABC):
    """
    Class that is used to calculate a distribution based on the input and assessment
    """

    @abstractmethod
    def __call__(
        self, assessment: Assessment, x: torch.Tensor
    ) -> typing.Union[torch.Tensor, torch.Tensor]:
        """

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
    or get the mean. Use for binary or disrete sets"""

    def __init__(self, equals_value):
        """initializer

        Args:
            equals_value: The value to get the distribution for
        """

        self.equals_value = equals_value

    def __call__(self, assessment: Assessment, x: torch.Tensor) -> torch.Tensor:
        """Calculate the assessment distribution of the input

        Args:
            assessment (Assessment): The assessment of the
            x (torch.Tensor): the input tensor

        Raises:
            ValueError: The dimension of value is not 3
            ValueError: The dimension of x is not 3

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor] : mean, std
        """
        if assessment.value.dim() != 2:
            raise ValueError("Value must have dimension of 2 ")
        if x.dim() == 3:
            value = assessment.value[:, :, None]
        else:
            value = assessment.value
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
        self, assessment: Assessment, x: torch.Tensor, n_samples: int = None
    ) -> torch.Tensor:
        """Generate a sample from the distribution

        Args:
            assessment (Assessment): The assessment
            x (torch.Tensor): The input
            n_samples (int, optional): the number of samples. Defaults to None.

        Returns:
            torch.Tensor: The sample value for the input
        """
        mean, std = self(assessment, x)
        return gaussian_sample(mean, std, n_samples)

    def mean(self, assessment: Assessment, x: torch.Tensor) -> torch.Tensor:
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
    """Freeze the dropout"""

    def __init__(self, p: float, freeze: bool = False):
        """Create a FreezeDropout

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

        if self.p == 0.0:
            return x

        if not self.training:
            return x * (1 / 1 - self.p)

        if self.freeze and self._cur is not None:
            f = self._cur
        else:
            f = (torch.rand_like(x) > self.p).type_as(x)

        self._cur = f
        return f * x
