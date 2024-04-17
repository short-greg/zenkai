# 1st party
from abc import ABC, abstractmethod

# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional as nn_func


class Reversible(nn.Module):
    """Base class for reversible modules"""

    @abstractmethod
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Null(Reversible):
    """
    Module that does not act on the inputs
    """

    def __init__(self, multi: bool = False):
        """initializer

        Args:
            multi (bool, optional): Whether the module can be reversed. Defaults to False.
        """
        super().__init__()
        if multi:
            self.forward = self.multi_forward
            self.reverse = self.multi_reverse
        else:
            self.forward = self.single_forward
            self.reverse = self.single_reverse

        self.multi = multi

    def multi_forward(self, *x: torch.Tensor):
        return x

    def single_forward(self, x: torch.Tensor):
        return x

    def multi_reverse(self, *y) -> torch.Tensor:
        return y

    def single_reverse(self, y) -> torch.Tensor:
        return y


class TargetReverser(ABC):
    """reverse the target"""

    @abstractmethod
    def reverse(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass


class SequenceReversible(Reversible):
    """Reverse a sequence"""

    def __init__(self, *reversibles: Reversible):
        """initialzier

        Args:
            reversible (Reversible): The erversible layers in the sequence
        """
        super().__init__()
        self._reversibles = nn.ModuleList(reversibles)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): the output

        Returns:
            torch.Tensor: the output passed through the reversed sequence
        """
        for module in reversed(self._reversibles):
            y = module.reverse(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The input passed through the sequence
        """
        for module in self._reversibles:
            x = module(x)
        return x


class SigmoidInvertable(Reversible):
    """Invert the sigmoid operation"""

    def reverse(self, y: torch.Tensor):
        return torch.log(y / (1 - y))

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(x)


class SoftMaxReversible(Reversible):
    """Reversed softmax. Uses the inversion of the sigmoid to reverse
    This is not necessarily correct since the SoftMax is not invertable
    """

    def __init__(self, dim=-1):
        """initializer

        Args:
            dim (int, optional): _description_. Defaults to -1.
        """

        super().__init__()
        self._dim = dim

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse the softmax operation by doing a sigmoid operation

        Args:
            y (torch.Tensor): the output of the layer

        Returns:
            Tensor: the inverted output
        """
        return torch.log(y / (1 - y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use SoftMax on the the layer

        Args:
            x (torch.Tensor): _description_

        Returns:
            Tensor: _description_
        """
        return nn_func.softmax(x, dim=self._dim)


class BatchNorm1DReversible(Reversible):
    """Invert the batch norm operation. Presently it doe not support the learnable parameters"""

    def __init__(self, n_features: int, momentum: float = 0.1):
        """initializer

        Args:
            n_features (int): The number of features to normalize
            momentum (float, optional): The momentum of the batchnorm. Defaults to 0.1.
        """
        super().__init__()
        self._momentum = momentum
        self._batch_norm = nn.BatchNorm1d(n_features, momentum=momentum, affine=False)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """invert the BatchNorm operation

        Args:
            y (torch.Tensor): the output of the layer

        Returns:
            torch.Tensor: the inverted batch norm
        """
        return (
            y * torch.sqrt(self._batch_norm.running_var[None])
            + self._batch_norm.running_mean[None]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """batch normalize the input

        Args:
            x (torch.Tensor): The input to the layer

        Returns:
            Tensor: The output of the batch norm operation
        """
        return self._batch_norm.forward(x)


class LeakyReLUInvertable(Reversible):
    """LeakyReLU that can be inverted"""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        assert negative_slope > 0.0
        self._negative_slope = negative_slope

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse the LeakyReLU function

        Args:
            y (torch.Tensor): the output to the layer

        Returns:
            torch.Tensor: the inverted output
        """
        return nn_func.leaky_relu(y, 1 / self._negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Send the LeakyReLU forward

        Args:
            x (torch.Tensor): the input to the layer

        Returns:
            torch.Tensor: the output of the layer
        """
        return nn_func.leaky_relu(x, self._negative_slope)


class BoolToSigned(Reversible):
    """Converts binary valued inputs so that -1 is negative, and 1 is positive"""

    def reverse(self, y: torch.Tensor):
        """Convert Negative one to zero

        Args:
            y (torch.Tensor): The tensor with negative one for negatives

        Returns:
            torch.Tensor: The tensor with zeros for negatives
        """
        return (y + 1) / 2

    def forward(self, x: torch.Tensor):
        """Convert Zero to Negative one

        Args:
            x (torch.Tensor): The tensor with zeros for negatives

        Returns:
            torch.Tensor: the tensor with Negative one for
        """
        return (x * 2) - 1


class SignedToBool(Reversible):
    """Converts binary valued inputs so that 0 is negative, and 1 is positive"""

    def __init__(self):
        super().__init__()
        self._neg = BoolToSigned()

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Convert Zero to Negative one

        Args:
            y (torch.Tensor): The tensor with zeros for negatives

        Returns:
            torch.Tensor: the tensor with Negative one for
        """
        return (y * 2) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert Negative one to zero

        Args:
            x (torch.Tensor): The tensor with negative one for negatives

        Returns:
            torch.Tensor: The tensor with zeros for negatives
        """
        return (x + 1) / 2
