# 3rd party
import torch

# local
from zenkai.nnz._least_squares import (
    LeastSquaresRidgeSolver,
    LeastSquaresStandardSolver,
)


class TestLeastSquaresStandardSolver:
    def test_solve_with_bias(self):
        solver = LeastSquaresStandardSolver(True)
        a = torch.rand(3, 3)
        b = torch.rand(3, 2)
        result = solver.solve(a, b)
        # weight (2, 3) + bias column => (2, 4)
        assert result.shape == torch.Size([2, 4])

    def test_solve_without_bias(self):
        solver = LeastSquaresStandardSolver(False)
        a = torch.rand(3, 3)
        b = torch.rand(3, 2)
        result = solver.solve(a, b)
        assert result.shape == torch.Size([2, 3])

    def test_forward_matches_solve(self):
        torch.manual_seed(0)
        solver = LeastSquaresStandardSolver(False)
        a = torch.rand(3, 3)
        b = torch.rand(3, 2)
        assert (solver.forward(a, b) == solver.solve(a, b)).all()


class TestLeastSquaresRidgeSolver:
    def test_solve_with_bias(self):
        solver = LeastSquaresRidgeSolver(1e-1, True)
        a = torch.rand(3, 3)
        b = torch.rand(3, 2)
        result = solver.solve(a, b)
        assert result.shape == torch.Size([2, 4])

    def test_solve_without_bias(self):
        solver = LeastSquaresRidgeSolver(1e-1, False)
        a = torch.rand(3, 3)
        b = torch.rand(3, 2)
        result = solver.solve(a, b)
        assert result.shape == torch.Size([2, 3])

    def test_forward_matches_solve(self):
        torch.manual_seed(0)
        solver = LeastSquaresRidgeSolver(1e-1, False)
        a = torch.rand(3, 3)
        b = torch.rand(3, 2)
        assert (solver.forward(a, b) == solver.solve(a, b)).all()
