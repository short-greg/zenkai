# 3rd party
import torch
import torch.nn as nn


class ExpandDim(nn.Module):
    """A module that reshapes the input tensor by expanding a specified dimension.

    Args:
        dim (int): The dimension to expand.
        size1 (int): The size of the expanded dimension.
        size2 (int): The size of the other dimension.

    Example:
        >>> import torch
        >>> from zenkai.nnz import ExpandDim
        >>> x = torch.tensor([1, 2, 3])
        >>> expand = ExpandDim(dim=0, size1=3, size2=1)
        >>> result = expand(x)
        >>> print(result)
        tensor([[1],
                [2],
                [3]])
    """

    def __init__(self, dim: int, size1: int, size2: int):
        """Initialize the module with the specified dimension and sizes.

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
        """Perform a forward pass by reshaping the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A reshaped tensor with the specified expanded dimension.
        """
        shape = list(x.shape)
        shape.insert(self.dim, self.size1)
        shape[self.dim + 1] = self.size2
        return x.reshape(*shape)
