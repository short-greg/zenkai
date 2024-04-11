# 3rd party
import typing
import pytest
import torch
import torch.optim
from torch import nn

from zenkai import utils

# local
from zenkai.kaku import IO, IDable
from zenkai.kaku import _machine as core
from zenkai.kaku import _assess


class Base:
    def __init__(self) -> None:
        pass


class X(Base, IDable):

    @property
    def id(self):
        return str(id(self))


class SimpleLearner(core.LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = _assess.ThLoss(nn.MSELoss, reduction="mean")
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self.loss.assess(y, t, reduction_override)

    def step_x(self, x: IO, t: IO) -> IO:
        if 'y' not in x._(self):
            assessment = self.assess(x, t.detach(), release=False)
            assessment.backward()

        return IO(x.f - x.f.grad)

    def step(self, x: IO, t: IO):
        if 'y' not in x._(self):
            y = self(x, release=False)
        else:
            y = x._(self).y
        self.optim.zero_grad()
        assessment = self.assess_y(y, t.detach())
        assessment.backward()
        self.optim.step()

    def forward(self, x: IO, release: bool = True) -> torch.Tensor:
        x.freshen(False)
        y = x._(self).y = IO(self.linear(x.f))
        return y.out(release)


class SimpleAccLearner(core.LearningMachine):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = _assess.ThLoss(nn.MSELoss, reduction="mean")
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self.loss.assess(y, t, reduction_override)

    def accumulate(self, x: IO, t: IO) -> IO:
        if 'y' not in x._(self):
            y = self(x, release=False)
        else:
            y = x._(self).y
        self.optim.zero_grad()
        assessment = self.assess_y(y, t.detach())
        assessment.backward()

    def step_x(self, x: IO, t: IO) -> IO:
        if 'y' not in x._(self):
            assessment = self.assess(x, t.detach(), release=False)
            assessment.backward()

        return IO(x.f - x.f.grad)

    def step(self, x: IO, t: IO):
        self.optim.step()

    def forward(self, x: IO, release: bool = True) -> torch.Tensor:
        x.freshen(False)
        y = x._(self).y = IO(self.linear(x.f))
        return y.out(release)


class DummyHook(core.LearnerPostHook):

    def __call__(
        self, x: IO, t: IO, y: IO, assessment: torch.Tensor
    ) -> typing.Tuple[IO, IO]:
        
        x._(self).hi = 'hi'


class TestLearningMachineWithSimpleLearner:
    def test_assess_y_uses_correct_reduction(self):

        learner = SimpleLearner(2, 3)
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, "sum")
        target = nn.MSELoss(reduction="sum")(*y, *t)
        assert result.item() == target.item()

    def test_grad_will_not_be_available_in_trans(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        y = learner(x, release=True)
        assert y.f.grad_fn is None

    def test_grad_will_be_available_in_trans_if_not_detaching(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        y = learner(x, release=False)
        assert y.f.grad_fn is not None

    def test_step_x_updates_x(self):

        learner = SimpleLearner(2, 3)
        base_x = torch.rand(2, 2)
        x = IO(torch.clone(base_x))
        t = IO(torch.rand(2, 3))
        learner(x)
        learner.step(x, t)
        x_prime = learner.step_x(x, t)

        assert (x_prime.f != base_x).any()

    def test_step_updates_parameters(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        before = utils.get_model_params(learner)
        learner(x)
        learner.step(x, t)
        after = utils.get_model_params(learner)
        assert (before != after).any()

    def test_learn_hook_called_after_learning_and_sets_state_to_hi(self):
        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        hook = DummyHook()
        learner.learner_hook(hook, True, False)
        learner.learn(x, t)
        assert x._(hook).hi == "hi"

    def test_learn_hook_not_called_after_testing_and_state_not_set_to_hi(self):
        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        hook = DummyHook()
        learner.learner_hook(hook, True, False)
        learner.test(x, t)
        # TODO: Check this
        assert 'hi' not in x._(hook)

    def test_learn_hook_called_after_testing_and_state_set_to_hi(self):
        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        hook = DummyHook()
        learner.learner_hook(hook, True, True)
        learner.test(x, t)
        assert x._(hook).hi == "hi"


class LayeredLearner(core.LearningMachine):

    def __init__(self, m1: SimpleLearner, m2: SimpleLearner):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self.m2.assess_y(y, t, reduction_override=reduction_override)

    def step_x(self, x: IO, t: IO) -> IO:
        t = self._.x_step_t
        return self.m1.step_x(x, t)

    def step(self, x: IO, t: IO):
        self.m2.step(self._.y1, t)
        t1 = self.m2.step_x(self._.y1, t)
        self.m1.step(x, t1)
        self._.x_step_t = t1

    def forward(self, x: IO, release: bool = True) -> torch.Tensor:
        y1 = self._.y1 = self.m1(x)
        y2 = self._.y2 = self.m2(y1)
        return y2.out(release)


class TestLearningMachineWithComplexLearner:

    def test_assess_y_uses_correct_reduction(self):

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, "sum")
        target = nn.MSELoss(reduction="sum")(*y, *t)
        assert result.item() == target.item()

    def test_assess_y_uses_default_reduction(self):

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t)
        target = nn.MSELoss(reduction="mean")(*y, *t)
        assert result.item() == target.item()

    def test_assess_uses_default_reduction(self):

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        x = IO(torch.rand(3, 2))
        t = IO(torch.rand(3, 3))
        result = learner.assess(x, t)
        y = learner(x)
        target = nn.MSELoss(reduction="mean")(y.f, t.f)
        assert result.item() == target.item()

    def test_excite_detaches_y(self):

        torch.manual_seed(1)
        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        x = IO(torch.rand(2, 2))
        y = learner(x, release=True)
        assert y.f.grad_fn is None

    def test_step_x_updates_x(self):
        torch.manual_seed(1)

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        x = IO(torch.rand(2, 2))
        x_ = x.clone(True)
        t = IO(torch.rand(2, 3))
        learner(x)
        learner.step(x, t)
        x_prime = learner.step_x(x, t)
        assert (x_prime.f != x_.f).any()

    def test_step_updates_parameters(self):
        torch.manual_seed(1)

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        before = utils.get_model_params(learner)
        learner.forward(x)
        learner.step(x, t)
        after = utils.get_model_params(learner)
        assert (before != after).any()


class DependentLearner(core.LearningMachine):
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = _assess.ThLoss(nn.MSELoss, reduction="mean")
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self.loss.assess(y, t, reduction_override)

    @core.step_dep("stepped")
    def step_x(self, x: IO, t: IO) -> IO:
        if 'y' not in x._(self):
            assessment = self.assess(x, t.detach(), release=False)
            assessment.backward()

        return IO(x.f - x.f.grad)

    @core.forward_dep("y")
    def step(self, x: IO, t: IO):
        y = x._(self).y
        self.optim.zero_grad()
        assessment = self.assess_y(y, t.detach())
        assessment.backward()
        self.optim.step()
        x._(self).stepped = True

    def forward(self, x: IO, release: bool = True) -> torch.Tensor:
        x.freshen(False)
        y = x._(self).y = IO(self.linear(x.f))
        return y.out(release)


class TestDependencies:

    def test_step_executes_forward(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        dependent = DependentLearner(2, 3)
        dependent(x)
        dependent.step(x, t)
        assert x._(dependent).y is not None

    def test_step_does_not_execute_forward_if_already_called(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        dependent = DependentLearner(2, 3)
        dependent(x)
        prev = x._(dependent).y
        dependent.step(x, t)
        assert x._(dependent).y is prev

    def test_step_x_if_not_stepped(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        dependent = DependentLearner(2, 3)
        with pytest.raises(RuntimeError):
            dependent.step_x(x, t)

    def test_step_x_executes_if_stepped(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        dependent = DependentLearner(2, 3)
        dependent(x)
        dependent.step(x, t)
        x_prime = dependent.step_x(x, t)
        assert x_prime is not None


class TestNullLearner:

    def test_assess_y_uses_correct_reduction(self):

        learner = core.NullLearner(loss=_assess.ThLoss("MSELoss"))
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, "sum")
        target = nn.MSELoss(reduction="sum")(*y, *t)
        assert result.item() == target.item()

    def test_null_learner_returns_y_for_step_x(self):

        learner = core.NullLearner(loss=_assess.ThLoss("MSELoss"))
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        x_prime = learner.step_x(y, t)
        assert (x_prime.f == y.f).all()

    def test_null_learner_outputs_x_for_forward(self):

        learner = core.NullLearner(loss=_assess.ThLoss("MSELoss"))
        x = IO(torch.rand(2, 3))
        IO(torch.rand(2, 3))
        y = learner.forward(x)
        assert y is x
