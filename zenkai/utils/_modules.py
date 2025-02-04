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
