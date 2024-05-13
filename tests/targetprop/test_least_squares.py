# 3rd party
import pytest
import torch
import torch.nn as nn

# local
from zenkai.kaku._io2 import IO2 as IO
from zenkai.targetprob import _least_squares
from zenkai.utils._params import get_model_params


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
    return (IO(torch.rand(3, 3)), IO(torch.rand(3, 2)), IO(torch.rand(3, 2)))


@pytest.fixture
def conn2():
    return (IO(torch.rand(3, 3)), IO(torch.rand(3, 2)), IO(torch.rand(3, 2)))


class TestLeastSquaresStepTheta:

    def test_step_x_with_optimize_dx(self, linear, conn1):
        x, t, y = conn1
        solver = _least_squares.LeastSquaresStandardSolver(True)
        step_theta = _least_squares.LeastSquaresStepTheta(linear, solver, True)
        before = torch.clone(step_theta.linear.weight)
        step_theta.step(x, t)
        assert (before != step_theta.linear.weight).any()

    def test_step_x_with_optimize_x(self, linear, conn1):
        x, t, y = conn1
        solver = _least_squares.LeastSquaresStandardSolver(True)
        step_theta = _least_squares.LeastSquaresStepTheta(linear, solver, False)
        before = torch.clone(step_theta.linear.weight)
        step_theta.step(x, t)
        assert (before != step_theta.linear.weight).any()

    def test_step_x_with_no_bias(self, linear_wo_bias, conn1):
        x, t, y = conn1
        solver = _least_squares.LeastSquaresStandardSolver(False)
        step_theta = _least_squares.LeastSquaresStepTheta(linear_wo_bias, solver, False)
        before = torch.clone(step_theta.linear.weight)
        step_theta.step(x, t)
        assert (before != step_theta.linear.weight).any()


class TestLeastSquaresStepX:
    def test_step_x_with_optimize_dx(self, linear2, conn2):
        x, t, y = conn2
        solver = _least_squares.LeastSquaresStandardSolver(False)
        step_x = _least_squares.LeastSquaresStepX(linear2, solver, True)
        before = torch.clone(x.f)
        x = step_x.step_x(x, t)
        assert (before != x.f).any()

    def test_step_x_with_optimize_x(self, linear2, conn2):
        x, t, y = conn2
        solver = _least_squares.LeastSquaresStandardSolver(False)
        step_x = _least_squares.LeastSquaresStepX(linear2, solver, False)
        before = torch.clone(x.f)
        x = step_x.step_x(x, t)
        assert (before != x.f).any()

    def test_step_x_with_optimize_x_wo_bias(self, linear2_wo_bias, conn2):
        x, t, y = conn2
        solver = _least_squares.LeastSquaresStandardSolver(False)
        step_x = _least_squares.LeastSquaresStepX(linear2_wo_bias, solver, False)
        before = torch.clone(x.f)
        x = step_x.step_x(x, t)
        assert (before != x.f).any()


class TestLeastSquaresGrad:
    
    def test_step_with_optimize(self, linear2, conn2):
        x, t, y = conn2
        learner = _least_squares.GradLeastSquaresLearner(3, 2, False, True)
        before = get_model_params(learner)
        learner(x)
        learner.accumulate(x, t)
        learner.step(x, t)
        assert (before != get_model_params(learner)).any()

    def test_step_x(self, linear2, conn2):
        x, t, y = conn2
        learner = _least_squares.GradLeastSquaresLearner(3, 2, False, True)
        x_prime = learner.step_x(x, t)
        assert (x.f != x_prime.f).any()

