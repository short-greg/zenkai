# 1st party
import typing
from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg

# 3rd party
import torch
import torch.nn as nn

from ..kaku.machine import (
    IO,
    AssessmentDict,
    Conn,
    LearningMachine,
    State,
    StepTheta,
    StepX,
    ThLoss,
    update_io,
)

# local
from ..utils import to_np, to_th_as


# TODO: Move to itadaki
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
        # print('Preparing w/')
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
        # print(new_.shape)

        return to_th_as(new_, a).T


class LeastSquaresStepTheta(StepTheta):
    def __init__(
        self,
        linear: nn.Linear,
        solver: LeastSquaresSolver = LeastSquaresStandardSolver,
        optimize_dw: bool = False,
        lr: typing.Optional[float] = None,
    ):
        """initializer

        Args:
            linear (nn.Linear): _description_
            solver (LeastSquaresSolver, optional): _description_. Defaults to LeastSquaresStandardSolver.
            optimize_dw (bool, optional): _description_. Defaults to False.
            lr (typing.Optional[float], optional): _description_. Defaults to None.
            is_fresh (bool, optional): _description_. Defaults to True.
        """
        self.linear = linear
        self.solver = solver
        self._optimize = self._optimize_dw if optimize_dw else self._optimize_w
        self._lr = lr

    def split(self, p: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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

    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        self._optimize(conn.step.x[0], conn.step.t[0])
        return conn.connect_in(from_)


class LeastSquaresStepX(StepX):
    def __init__(
        self,
        linear: nn.Linear,
        solver: LeastSquaresSolver = LeastSquaresStandardSolver,
        optimize_dx: bool = False,
        lr: typing.Optional[float] = None,
    ):
        """initializer

        Args:
            linear (nn.Linear): _description_
            solver (LeastSquaresSolver, optional): _description_. Defaults to LeastSquaresStandardSolver.
            optimize_dx (bool, optional): _description_. Defaults to False.
            lr (typing.Optional[float], optional): _description_. Defaults to None.
            is_fresh (bool, optional): _description_. Defaults to True.
        """
        self.linear = linear
        self.solver = solver
        self._optimize = self._optimize_dx if optimize_dx else self._optimize_x
        self._lr = lr

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

    def step_x(self, conn: Conn, state: State) -> Conn:
        """Update x

        Args:
            conn (Conn): The connection to update with
            state (State): The current learning state

        Returns:
            Conn: The connection
        """
        x = self._optimize(conn.step_x.x[0], conn.step_x.t[0])
        update_io(IO(x), conn.step_x.x)
        conn.tie_inp()
        return conn


class LeastSquaresLearner(LearningMachine):
    """Learner that uses least squares"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        optimize_dx: bool = True,
        x_lr: float = 1e-2,
        x_reduction: str = "mean",
    ):
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

    def step(self, conn: Conn, state: State, from_: IO = None) -> Conn:
        conn = self._step_theta.step(conn, state, from_)
        return conn

    def step_x(self, conn: Conn, state: State) -> Conn:
        conn = self._step_x.step_x(conn, state)
        return conn

    def forward(self, x: IO, state: State, detach: bool = True) -> IO:
        x.freshen(False)
        return IO(self._linear(x[0]), detach=detach)
