import torch

from zenkai._core import IO
from zenkai.nnz._assess import NNLoss
from zenkai.nnz._constraints import (
    GT,
    GTE,
    LT,
    LTE,
    CriterionObjective,
    FuncObjective,
    NullConstraint,
    ValueConstraint,
)


class TestNullConstraint:
    def test_null_constraint_returns_empty(self):

        constraint = NullConstraint()
        result = constraint(x=torch.randn(4, 2))
        assert result == {}


class TestValueConstraint:
    def test_value_constraint_flags_violations(self):

        constraint = ValueConstraint(lambda v, c: v >= c, x=0.0)
        x = torch.tensor([[-1.0, 1.0]])
        result = constraint(x=x)
        assert (result["x"] == torch.tensor([[False, True]])).all()

    def test_value_constraint_ignores_unconstrained_keys(self):

        constraint = ValueConstraint(lambda v, c: v >= c, x=0.0)
        result = constraint(y=torch.randn(4, 2))
        assert "y" not in result

    def test_value_constraint_reduces_dim(self):

        constraint = ValueConstraint(lambda v, c: v >= c, reduce_dim=1, x=0.0)
        x = torch.tensor([[-1.0, 1.0], [-1.0, -1.0]])
        result = constraint(x=x)
        assert result["x"].shape == torch.Size([2])
        assert (result["x"] == torch.tensor([True, False])).all()


class TestLT:
    def test_lt_violates_when_value_ge_constraint(self):

        constraint = LT(x=1.0)
        x = torch.tensor([0.0, 1.0, 2.0])
        result = constraint(x=x)
        assert (result["x"] == torch.tensor([False, True, True])).all()


class TestLTE:
    def test_lte_violates_when_value_gt_constraint(self):

        constraint = LTE(x=1.0)
        x = torch.tensor([0.0, 1.0, 2.0])
        result = constraint(x=x)
        assert (result["x"] == torch.tensor([False, False, True])).all()


class TestGT:
    def test_gt_violates_when_value_le_constraint(self):

        constraint = GT(x=1.0)
        x = torch.tensor([0.0, 1.0, 2.0])
        result = constraint(x=x)
        assert (result["x"] == torch.tensor([True, True, False])).all()


class TestGTE:
    def test_gte_violates_when_value_lt_constraint(self):

        constraint = GTE(x=1.0)
        x = torch.tensor([0.0, 1.0, 2.0])
        result = constraint(x=x)
        assert (result["x"] == torch.tensor([True, False, False])).all()


class TestFuncObjective:
    def test_func_objective_returns_reduced_value(self):

        objective = FuncObjective(lambda x: x)
        x = torch.randn(4, 2)
        assessment = objective("mean", x=x)
        assert assessment.allclose(x.mean())

    def test_func_objective_applies_penalty_on_violation(self):

        objective = FuncObjective(lambda x: x, constraint=GT(x=0.0), penalty=1000.0)
        x = torch.zeros(4, 2)
        assessment = objective("none", x=x)
        assert (assessment == -1000.0).all()

    def test_func_objective_raises_on_negative_penalty(self):

        try:
            FuncObjective(lambda x: x, penalty=-1.0)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for negative penalty")


class TestCriterionObjective:
    def test_criterion_objective_delegates_to_criterion(self):

        criterion = NNLoss("MSELoss")
        objective = CriterionObjective(criterion)
        x = torch.randn(4, 2)
        t = torch.randn(4, 2)
        assessment = objective("mean", x=x, t=t)
        expected = criterion.assess(IO(x), IO(t), "mean")
        assert assessment.allclose(expected)
