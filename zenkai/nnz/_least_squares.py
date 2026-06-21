# 1st party
import typing
from abc import abstractmethod

# 3rd party
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

# local
from zenkai._core import to_np, to_th_as


class LeastSquaresSolver(nn.Module):
    """ABC for solvers using least squares"""

    @abstractmethod
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Abstract method for the least squares solver

        Args:
            a (torch.Tensor): Input
            b (torch.Tensor): Target

        Returns:
            torch.Tensor: the least squares solution
        """
        pass

    def solve(self, *args, **kwargs) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(*args, **kwargs)


class LeastSquaresStandardSolver(LeastSquaresSolver):
    """Solve least squares"""

    def __init__(self, bias: bool = False):
        """Create a least squares solver

        Args:
            bias (bool, optional): Whether there is a bias. Defaults to False.
        """
        super().__init__()
        if bias:
            self._prepare = self._prepare_with_bias
        else:
            self._prepare = self._prepare_without_bias

    def _prepare_without_bias(self, a: np.ndarray, b: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return a, b

    def _prepare_with_bias(self, a: np.ndarray, b: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        m, _ = np.shape(a)
        a = np.hstack([a, np.ones((m, 1))])
        return a, b

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Solve least squares between a and b

        Args:
            a (torch.Tensor): input
            b (torch.Tensor): target

        Returns:
            torch.Tensor: the least squares solution
        """
        new_, _, _, _ = scipy.linalg.lstsq(*self._prepare(to_np(a), to_np(b)))
        return to_th_as(new_, a).T

    solve = forward


class LeastSquaresRidgeSolver(LeastSquaresSolver):
    """Solve least squares using ridge regression"""

    def __init__(self, lam: float = 1e-1, bias: bool = False):
        """Create a solver that will use ridge regression

        Args:
            lam (float, optional): The penalty on the regression. Defaults to 1e-1.
            bias (bool, optional): Whether to use a bias or not. Defaults to False.
        """
        super().__init__()
        self._bias = bias
        self._lambda = lam
        if self._bias:
            self._prepare = self._prepare_with_bias
        else:
            self._prepare = self._prepare_without_bias

    def _prepare_without_bias(self, a: np.ndarray, b: np.ndarray):
        _, n = np.shape(a)
        lower_half = np.zeros((n, n))
        np.fill_diagonal(lower_half, np.sqrt(self._lambda))
        return (np.vstack((a, lower_half)), np.vstack([b, np.zeros((n, b.shape[1]))]))

    def _prepare_with_bias(self, a: np.ndarray, b: np.ndarray):
        m, n = np.shape(a)
        upper_half = np.hstack([a, np.ones((m, 1))])
        lower = np.zeros((n, n))
        np.fill_diagonal(lower, np.sqrt(self._lambda))
        lower_half = np.hstack([lower, np.zeros((n, 1))])
        return (
            np.vstack((upper_half, lower_half)),
            np.vstack([b, np.zeros((n, b.shape[1]))]),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        """Solve least squares between a and b

        Args:
            a (torch.Tensor): input
            b (torch.Tensor): target

        Returns:
            torch.Tensor: the least squares solution
        """
        A, B = self._prepare(to_np(a), to_np(b))
        new_, _, _, _ = scipy.linalg.lstsq(A.T @ A, A.T @ B)

        return to_th_as(new_, a).T

    solve = forward
