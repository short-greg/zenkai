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
    Criterion,
    NNLoss,
    OptimFactory
)
from ..kaku._state import State
from ..kaku._io2 import (
    IO as IO, iou
)
from ..kaku._lm2 import (
    LearningMachine as LearningMachine,
    StepTheta as StepTheta,
    StepX as StepX,

)
from ..utils import to_np, to_th_as
from ..kaku._grad import GradStepTheta


class LeastSquaresSolver(ABC):
    """ABC for solvers using least squares
    """

    @abstractmethod
    def solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Abstract method for the least squares solver

        Args:
            a (torch.Tensor): Input
            b (torch.Tensor): Target

        Returns:
            torch.Tensor: the least squares solution
        """
        pass


class LeastSquaresStandardSolver(LeastSquaresSolver):
    """Solve least squares"""

    def __init__(self, bias: bool = False):
        """Create a least squares solver

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
        """Create a solver that will use ridge regression

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
    """A StepTheta that uses least squares """

    def __init__(
        self,
        linear: nn.Linear,
        solver: LeastSquaresSolver = LeastSquaresStandardSolver,
        optimize_dw: bool = False,
    ):
        """Create a StepTheta that uses 

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
        with torch.no_grad():
            t_delta = t - self.linear.forward(x)
            dweight, dbias = self.split(self.solver.solve(x, t_delta))
            self.linear.weight.copy_(self.linear.weight + dweight)
            if dbias is not None:
                self.linear.bias.copy_(self.linear.bias + dbias)

    def _optimize_w(self, x: torch.Tensor, t: torch.Tensor):
        with torch.no_grad():
            weight, bias = self.split(self.solver.solve(x, t))
            self.linear.weight.copy_(weight)
            if bias is not None:
                self.linear.bias.copy_(bias)

    def step(self, x: IO, t: IO, state: State):
        self._optimize(x.f, t.f)


class LeastSquaresStepX(StepX):
    """Update x based on the least squares estimator"""

    def __init__(
        self,
        linear: nn.Linear,
        solver: LeastSquaresSolver = LeastSquaresStandardSolver,
        optimize_dx: bool = False,
    ):
        """Create a StepX that uses least squares

        Args:
            linear (nn.Linear): The linear model to use
            solver (LeastSquaresSolver, optional): The solver to use for updating. 
                Defaults to LeastSquaresStandardSolver.
            optimize_dx (bool, optional): Whether to minimize the delta of x or x. 
                In general recommended to use delta. Defaults to False.
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

    def step_x(self, x: IO, t: IO, state: State, **kwargs) -> IO:
        """Update x

        Args:
            conn (Conn): The connection to update with

        Returns:
            Conn: The connection with x updated
        """
        x_prime = self._optimize(x.f, t.f)
        return iou(x_prime)

        # return update_io(IO(x_prime), x)


class LeastSquaresLearner(LearningMachine):
    """Learner that uses least squares to optimize theta and x. It wraps a 
    standard linear model. Uses a ridge regresion solver"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        optimize_dx: bool = True,
        lam_theta: float = 1e-3,
        lam_x: float = 1e-4,
    ):
        """Create a learner to use gradient for the parameter update and least squares for the input update

        Args:
            in_features (int): The number of features into the linear model
            out_features (int): The number of features out of the model
            bias (bool, optional): Whether to use the bias. Defaults to True.
            optimize_dx (bool, optional): Whether to minimize the delta. Defaults to True.
        """
        super().__init__()
        self._linear = nn.Linear(in_features, out_features, bias)
        self._loss = Criterion("MSELoss", "mean")
        self._step_x = LeastSquaresStepX(
            self._linear, LeastSquaresRidgeSolver(lam_x, False), optimize_dx
        )
        self._step_theta = LeastSquaresStepTheta(
            self._linear, LeastSquaresRidgeSolver(lam_theta, bias=bias), optimize_dx
        )

    def step(self, x: IO, t: IO, state: State):
        """Update the parameters using least squares

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
        """
        self._step_theta.step(x, t, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update the input using least squares

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state

        Returns:
            IO: Updated X
        """
        return self._step_x.step_x(x, t, state)

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        """Use a linear function on x

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            typing.Union[typing.Tuple, typing.Any]: The linear 
        """
        return iou(
            self._linear(x.f)
        )


class GradLeastSquaresLearner(LearningMachine):
    """Learner that uses grad to optimize theta and least squares to optimize x. 
    It wraps a standard linear model. Uses a ridge regresion solver"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        optimize_dx: bool = True,
        optim_factory: OptimFactory = None,
        loss: Criterion = None,
        lam_x: float = 1e-4,
    ):
        """Create a learner to use gradient for the parameter update and least squares for the input update

        Args:
            in_features (int): The number of features into the linear model
            out_features (int): The number of features out of the model
            bias (bool, optional): Whether to use the bias. Defaults to True.
            optimize_dx (bool, optional): Whether to minimize the delta. Defaults to True.
            optim_factory (OptimFactory, optional): The optimizer to use. Defaults to None.
            loss (Objective, optional): The loss to minimize. Since this is grad 
                descent it must be a minimization function. Defaults to None.
            lam_x (float, optional): The regularization parameter. Defaults to 1e-4.
        """
        super().__init__()
        self._linear = nn.Linear(in_features, out_features, bias)
        self._loss = loss or NNLoss("MSELoss", "mean")
        self._step_x = LeastSquaresStepX(
            self._linear, LeastSquaresRidgeSolver(lam_x, False), optimize_dx
        )
        optim_factory = optim_factory or OptimFactory("Adam", lr=1e-3)
        self._step_theta = GradStepTheta(
            self, learn_criterion=NNLoss('MSELoss'), optimf=optim_factory
        )

    def accumulate(self, x: IO, t: IO, state: State):
        """Use least squares to update the weights

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state
        """
        self._step_theta.accumulate(x, t, state.sub('least'), y=state._y)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Use least squares to update x

        Args:
            x (IO): The input
            t (IO): The target
            state (State): The learning state

        Returns:
            IO: The updated x
        """
        return self._step_x.step_x(x, t, state.sub('grad'))

    def forward_nn(self, x: IO, state: State) -> IO:
        """Use the linear layer to output

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            IO: The output
        """
        return self._linear(x.f)

    def step(self, x: IO, t: typing.Union[IO, None], state: State):
        """Update the accumulated parameters using least squares

        Args:
            x (IO): The input
            t (typing.Union[IO, None]): The target
            state (State): The learning state
        """
        self._step_theta.step(x, t, state.sub('least'), y=state._y)
