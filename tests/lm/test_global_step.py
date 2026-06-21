import torch
import torch.nn as nn

from zenkai._core import IO, State, iou, to_pvec
from zenkai.lm._global_step import GlobalTargetLearner
from zenkai.lm._grad import GradLearner
from zenkai.lm._lm import LMode
from zenkai.nnz import NNLoss


class THGradLearnerT1(GradLearner):
    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            linear,
            criterion=NNLoss(nn.MSELoss),
        )
        self._optim = torch.optim.Adam(linear.parameters(), lr=1e-3)

    def step(self, x: IO, t: IO, state: State):

        self._optim.step()
        self._optim.zero_grad()


class DummyGlobalTargetLearner(GlobalTargetLearner):

    def __init__(self, lmode: LMode = LMode.Standard):
        super().__init__(lmode)

        self.learner1 = THGradLearnerT1(2, 4)
        self.learner2 = THGradLearnerT1(4, 2)

    def forward_iter(self, x, state, **kwargs):

        y1 = self.learner1.forward_io(x, state.sub("sub1"))
        y2 = self.learner2.forward_io(y1, state.sub("sub2"))

        yield self.learner1, y1, state.sub("sub1")
        yield self.learner2, y2, state.sub("sub2")

    def optim_x(self, x, t, state):

        return iou(torch.randn_like(x.f))


class TestGlobalTargetLearner:

    def test_forward_outputs_correct_value(self):

        learner = DummyGlobalTargetLearner()
        x = torch.randn(4, 2)
        t = learner.learner2(learner.learner1(x))
        y = learner(x)

        assert (y == t).all()

    def test_parameters_are_all_updated_on_backward(self):

        learner = DummyGlobalTargetLearner(LMode.WithStep)
        x = torch.randn(4, 2)
        t = torch.rand(4, 2)
        y = learner(x)
        before = to_pvec(learner)
        (y - t).pow(2).mean().backward()
        after = to_pvec(learner)

        assert (before != after).any()

    def test_step_x_returns_new_x(self):

        learner = DummyGlobalTargetLearner(LMode.WithStep)
        x = torch.randn(4, 3)
        t = torch.rand(4, 2)
        base = nn.Linear(3, 2)
        y = learner(base(x))
        (y - t).pow(2).mean().backward()

        assert base.weight.grad is not None
