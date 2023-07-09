# 3rd party
import pytest
import torch
import torch.nn as nn

# local
from zenkai.kaku import IO, State
from zenkai.kikai import least_squares


@pytest.fixture
def linear():
    return nn.Linear(3, 2)

@pytest.fixture
def linear2():
    return nn.Linear(3, 2)

@pytest.fixture
def linear_wo_bias():
    return nn.Linear(3, 2, bias=False)

@pytest.fixture
def linear2_wo_bias():
    return nn.Linear(3, 2, bias=False)


@pytest.fixture
def conn1():
    return (
        IO(torch.rand(3, 3)), IO(torch.rand(3, 2)), IO(torch.rand(3, 2))
    )

@pytest.fixture
def conn2():
    return (
        IO(torch.rand(3, 3)), IO(torch.rand(3, 2)), IO(torch.rand(3, 2))
    )


class TestLeastSquaresStepTheta:

    def test_step_x_with_optimize_dx(self, linear, conn1):
        x, t, y = conn1
        solver = least_squares.LeastSquaresStandardSolver(True)
        step_theta = least_squares.LeastSquaresStepTheta(
            linear, solver, True
        )
        before = torch.clone(step_theta.linear.weight)
        step_theta.step(x, t, State())
        assert (before != step_theta.linear.weight).any()

    def test_step_x_with_optimize_x(self, linear, conn1):
        x, t, y = conn1
        solver = least_squares.LeastSquaresStandardSolver(True)
        step_theta = least_squares.LeastSquaresStepTheta(
            linear, solver, False
        )
        before = torch.clone(step_theta.linear.weight)
        step_theta.step(x, t, State())
        assert (before != step_theta.linear.weight).any()

    def test_step_x_with_no_bias(self, linear_wo_bias, conn1):
        x, t, y = conn1
        solver = least_squares.LeastSquaresStandardSolver(False)
        step_theta = least_squares.LeastSquaresStepTheta(
            linear_wo_bias, solver, False
        )
        before = torch.clone(step_theta.linear.weight)
        step_theta.step(x, t, State())
        assert (before != step_theta.linear.weight).any()


class TestLeastSquaresStepX:

    def test_step_x_with_optimize_dx(self, linear2, conn2):
        x, t, y = conn2
        solver = least_squares.LeastSquaresStandardSolver(False)
        step_x = least_squares.LeastSquaresStepX(
            linear2, solver, True
        )
        before = torch.clone(x[0])
        x = step_x.step_x(x, t, State())
        assert (before != x[0]).any()

    def test_step_x_with_optimize_x(self, linear2, conn2):
        x, t, y = conn2
        solver = least_squares.LeastSquaresStandardSolver(False)
        step_x = least_squares.LeastSquaresStepX(
            linear2, solver, False
        )
        before = torch.clone(x[0])
        x = step_x.step_x(x, t, State())
        assert (before != x[0]).any()

    def test_step_x_with_optimize_x_wo_bias(self, linear2_wo_bias, conn2):
        x, t, y = conn2
        solver = least_squares.LeastSquaresStandardSolver(False)
        step_x = least_squares.LeastSquaresStepX(
            linear2_wo_bias, solver, False
        )
        before = torch.clone(x[0])
        x = step_x.step_x(x, t, State())
        assert (before != x[0]).any()
