import torch

from zenkai._core import Reduction
from zenkai.nnz._objective import (
    Constraint,
    Objective,
    impose,
)


class ObjectiveSample(Objective):
    def forward(self, reduction: str, **kwargs: torch.Tensor) -> torch.Tensor:
        return Reduction[reduction].forward(kwargs["x"])


class EqualityConstraint(Constraint):
    def __init__(self, value=0) -> None:
        super().__init__()
        self.value = value

    def forward(self, **kwargs: torch.Tensor) -> torch.BoolTensor:

        return {k: v == self.value for k, v in kwargs.items()}


class TestObjective:
    def test_objective_returns_assessment(self):

        objective = ObjectiveSample(True)
        x = torch.randn(4, 2)
        assessment = objective("mean", x=x)
        assert (assessment).allclose(x.mean())


class TestEqualityConstraint:
    def test_equality_constraint_returns_true(self):

        constraint = EqualityConstraint()
        x = torch.zeros(4, 2)
        result = constraint(x=x)
        assert result["x"].all()

    def test_equality_constraint_returns_false(self):

        constraint = EqualityConstraint(1)
        x = torch.zeros(4, 2)
        result = constraint(x=x)
        assert ~result["x"].all()


class TestCompoundConstraint:
    def test_equality_constraint_returns_true(self):

        constraint = EqualityConstraint(1) + EqualityConstraint(0)
        x = torch.cat([torch.zeros(4, 2), torch.ones(4, 2)])

        result = constraint(x=x)
        assert result["x"].all()


class TestImpose:
    def test_impose_does_not_change_when_constraint_is_false(self):

        constraint = EqualityConstraint(1)
        x = torch.zeros(4, 2)
        result = constraint(x=x)
        constrained = impose(x, result["x"], 1000)

        assert (constrained == x).all()

    def test_impose_sets_to_penalty_where_constraint_is_true(self):

        constraint = EqualityConstraint(0)
        x = torch.zeros(4, 2)
        result = constraint(x=x)
        constrained = impose(x, result["x"], 1000)

        assert (constrained == 1000).all()

    def test_impose_returns_value_when_constraint_is_none(self):

        x = torch.zeros(4, 2)
        constrained = impose(x)

        assert (constrained == x).all()
