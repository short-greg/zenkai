# 1st party
import typing
from abc import ABC, abstractmethod

# 3rd party
import torch
import torch.nn as nn
import numpy as np
import scipy.linalg

# local
from ..kaku import (
    IO,
    AssessmentDict,
    LearningMachine,
    State,
    StepTheta,
    StepX,
    ThLoss,
    update_io,
)
from ..utils import to_np, to_th_as


class LeastSquaresSolver(ABC):
    @abstractmethod
    def solve(self, a: torch.Tensor, b: torch.Tensor):
        pass


class LeastSquaresStandardSolver(LeastSquaresSolver):
    """Solve least squares"""

    def __init__(self, bias: bool = False):
        """initializer

        Args:
            bias (bool, optional): Whether there is a bias. Defaults to False.
        """

        if bias:
            self._prepare = self._prepare_with_bias
        else:
            self._prepare = self._prepare_without_bias

    def _prepare_without_bias(
        self, a: np.ndarray, b: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        return a, b

    def _prepare_with_bias(
        self, a: np.ndarray, b: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        m, _ = np.shape(a)
        a = np.hstack([a, np.ones((m, 1))])
        return a, b

    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Solve least squares between a and b

        Args:
            a (torch.Tensor): input
            b (torch.Tensor): target

        Returns:
            torch.Tensor: the least squares solution
        """
        new_, _, _, _ = scipy.linalg.lstsq(*self._prepare(to_np(a), to_np(b)))
        return to_th_as(new_, a).T


class LeastSquaresRidgeSolver(LeastSquaresSolver):
    """Solve least squares using ridge regression"""

    def __init__(self, lam: float = 1e-1, bias: bool = False):
        """initializer

        Args:
            lam (float, optional): The penalty on the regression. Defaults to 1e-1.
            bias (bool, optional): Whether to use a bias or not. Defaults to False.
        """
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

    def solve(self, a: torch.Tensor, b: torch.Tensor):
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


class LeastSquaresStepTheta(StepTheta):
    def __init__(
        self,
        linear: nn.Linear,
        solver: LeastSquaresSolver = LeastSquaresStandardSolver,
        optimize_dw: bool = False,
    ):
        """initializer

        Args:
            linear (nn.Linear): The linear model to optmize theta for
            solver (LeastSquaresSolver, optional): _description_. Defaults to LeastSquaresStandardSolver.
            optimize_dw (bool, optional): Whether to optimize the delta or the raw value. 
              In general recommended to optimize delta to minimize the change bewteen updates. Defaults to False.
        """
        self.linear = linear
        self.solver = solver
        self._optimize = self._optimize_dw if optimize_dw else self._optimize_w

    def split(self, p: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """split the parameter into a weight and bias

        Args:
            p (torch.Tensor): The parameter to split

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor]: the parameter split. If no bias, "None" will be returned
        """
        if self.linear.bias is None:
            return p, None

        return p[:, :-1], p[:, -1]

    def _optimize_dw(self, x: torch.Tensor, t: torch.Tensor):
        t_delta = t - self.linear.forward(x)
        dweight, dbias = self.split(self.solver.solve(x, t_delta))
        self.linear.weight.data = self.linear.weight.data + dweight
        if dbias is not None:
            self.linear.bias.data = self.linear.bias.data + dbias

    def _optimize_w(self, x: torch.Tensor, t: torch.Tensor):
        weight, bias = self.split(self.solver.solve(x, t))
        self.linear.weight.data = weight
        if bias is not None:
            self.linear.bias.data = bias

    def step(self, x: IO, t: IO, state: State):
        self._optimize(x[0], t[0])


class LeastSquaresStepX(StepX):
    """Update x based on the least squares estimator"""

    def __init__(
        self,
        linear: nn.Linear,
        solver: LeastSquaresSolver = LeastSquaresStandardSolver,
        optimize_dx: bool = False
    ):
        """initializer

        Args:
            linear (nn.Linear): The linear model to use
            solver (LeastSquaresSolver, optional): The solver to use for updating. Defaults to LeastSquaresStandardSolver.
            optimize_dx (bool, optional): Whether to minimize the delta of x or x. In general recommended to use delta. Defaults to False.
        """
        self.linear = linear
        self.solver = solver
        self._optimize = self._optimize_dx if optimize_dx else self._optimize_x

    def _optimize_dx(self, x: torch.Tensor, t: torch.Tensor):
        y = self.linear(x)
        if self.linear.bias is not None:
            t = t - self.linear.bias[None]
            y = y - self.linear.bias[None]
        t_delta = t - y
        dx = self.solver.solve(self.linear.weight, t_delta.T)
        return x + dx

    def _optimize_x(self, x: torch.Tensor, t: torch.Tensor):
        if self.linear.bias is not None:
            t = t - self.linear.bias[None]
        return self.solver.solve(self.linear.weight, t.T)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update x

        Args:
            conn (Conn): The connection to update with
            state (State): The current learning state

        Returns:
            Conn: The connection with x updated
        """
        x_prime = self._optimize(x[0], t[0])
        update_io(IO(x_prime), x)
        return x


class LeastSquaresLearner(LearningMachine):
    """Learner that uses least squares to optimize theta and x. It wraps a standard linear model. 
    Uses a ridge regresion solver"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        optimize_dx: bool = True
    ):
        """initializer

        Args:
            in_features (int): The number of features into the linear model
            out_features (int): The number of features out of the model
            bias (bool, optional): Whether to use the bias. Defaults to True.
            optimize_dx (bool, optional): Whether to minimize the delta. Defaults to True.
        """
        super().__init__()
        self._linear = nn.Linear(in_features, out_features, bias)
        self._loss = ThLoss("mse", "mean")
        self._step_x = LeastSquaresStepX(
            self._linear, LeastSquaresRidgeSolver(1e-4, False), optimize_dx
        )
        self._step_theta = LeastSquaresStepTheta(
            self._linear, LeastSquaresRidgeSolver(1e-3, bias=bias), optimize_dx
        )

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> AssessmentDict:
        return self._loss.assess_dict(y[0], t[0], reduction_override=reduction_override)

    def step(self, x: IO, t: IO, state: State):
        self._step_theta.step(x, t, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        return self._step_x.step_x(x, t, state)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen(False)
        return IO(self._linear(x[0]), detach=release)
