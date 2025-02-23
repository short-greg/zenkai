import typing
import torch.nn as nn
import torch


class Argmax(nn.Module):
    """
    A custom PyTorch module that computes the index of the maximum value along a specified dimension.
    Args:
        dim (int): The dimension along which to compute the argmax. Default is -1.
        keepdim (bool): Whether to retain the reduced dimension in the output tensor. Default is False.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the index of the maximum value along the specified dimension of the input tensor.
    Example:
        >>> import torch
        >>> from zenkai.utils._modules import Argmax
        >>> x = torch.tensor([[1, 3, 2], [4, 0, 5]])
        >>> argmax = Argmax(dim=1)
        >>> result = argmax(x)
        >>> print(result)
        tensor([1, 2])
    """

    def __init__(self, dim: int=-1, keepdim: bool=False):
        """
        Initializes the instance with the specified dimension and keepdim flag.
        Args:
            dim (int, optional): The dimension to reduce. Default is -1.
            keepdim (bool, optional): Whether to retain the reduced dimension in the output. Default is False.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass by computing the index of the maximum value along a specified dimension.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: A tensor containing the indices of the maximum values along the specified dimension.
        """
        return x.argmax(self.dim, self.keepdim)


class ExpandDim(nn.Module):
    """
    A custom PyTorch module that reshapes the input tensor by expanding a specified dimension.
    Args:
        dim (int): The dimension to expand.
        size1 (int): The size of the expanded dimension.
        size2 (int): The size of the other dimension.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Reshapes the input tensor by expanding the specified dimension.
    Example:
        >>> import torch
        >>> from zenkai.utils._modules import Expand
        >>> x = torch.tensor([1, 2, 3])
        >>> expand = Expand(dim=0, size1=3, size2=1)
        >>> result = expand(x)
        >>> print(result)
        tensor([[1],
                [2],
                [3]])
    """

    def __init__(self, dim: int, size1: int, size2: int):
        """
        Initializes the instance with the specified dimension and sizes.
        Args:
            dim (int): The dimension to expand.
            size1 (int): The size of the expanded dimension.
            size2 (int): The size of the other dimension.
        """
        super().__init__()
        self.dim = dim
        self.size1 = size1
        self.size2 = size2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass by reshaping the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: A reshaped tensor with the specified expanded dimension.
        """
        shape = list(x.shape)
        shape.insert(self.dim, self.size1)
        shape[self.dim + 1] = self.size2
        return x.reshape(*shape)


class Sign(nn.Module):
    """
    A custom PyTorch module that computes the sign of each element in the input tensor.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the sign of each element in the input tensor.
    Example:
        >>> import torch
        >>> from zenkai.utils._modules import Sign
        >>> x = torch.tensor([[-1.5, 0, 2.3], [4.1, -0.2, -3.3]])
        >>> sign = Sign()
        >>> result = sign(x)
        >>> print(result)
        tensor([[-1.,  0.,  1.],
                [ 1., -1., -1.]])
    """

    def __init__(self):
        """
        Initializes the instance.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass by computing the sign of each element in the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: A tensor containing the signs of the input tensor elements.
        """
        return x.sign()


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


class Null(nn.Module):
    """
    Module that does not act on the inputs
    """

    def forward(self, *x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        """Send multiple single value forward

        Returns:
            typing.Tuple[torch.Tensor]: The inputs
        """
        if len(x) == 1:
            return x[0]
        return x
    
    def reverse(self, *x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        """Send multiple single value forward

        Returns:
            typing.Tuple[torch.Tensor]: The inputs
        """
        if len(x) == 1:
            return x[0]
        return x

class Updater(nn.Module):
    """Use for updating a tensor (such as with a decay function)
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update the tensor stored internally and return it

        Args:
            x (torch.Tensor): The tensor to update with

        Returns:
            torch.Tensor: The updated tensor
        """
        if self.cur_val is None:
            self.cur_val = x

        elif self.update_f is not None:
            self.cur_val = self.update_f(
                x, self.cur_val, *self.args, **self.kwargs
            )
            return self.cur_val
        return x
