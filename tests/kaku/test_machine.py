# 3rd party
import typing
import pytest
import torch
import torch.optim
from torch import nn

from zenkai import utils

# local
from zenkai.kaku import IO, IDable, Assessment, State
from zenkai.kaku import _machine as core
from zenkai.kaku import _assess


class Base:

    def __init__(self) -> None:
        pass


class X(Base, core.IDable):

    @property
    def id(self):
        return str(id(self))


class Base:

    def __init__(self) -> None:
        pass


class X(Base, IDable):

    @property
    def id(self):
        return str(id(self))


@pytest.fixture
def x():
    return torch.randn(4, 2)

@pytest.fixture
def t():
    return torch.randn(4, 2)

@pytest.fixture
def y():
    return torch.randn(4, 2)

@pytest.fixture
def x2():
    return torch.randn(4, 2)

@pytest.fixture
def t2():
    return torch.randn(4, 2)

@pytest.fixture
def y2():
    return torch.randn(4, 2)


@pytest.fixture
def idx():
    return [1, 2]


@pytest.fixture
def idx2():
    return [1, 2, 0]


class SimpleLearner(core.LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = _assess.ThLoss(nn.MSELoss, reduction='mean')
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t:IO, reduction_override: str = None) -> core.Assessment:
        return self.loss.assess(y, t, reduction_override)
    
    def step_x(self, x: IO, t: IO, state: core.State) -> IO:
        if ((self, x), 'y') not in state:
            assessment = self.assess(x,  t.detach(), state=state, release=False)
            assessment.backward()
            
        return IO(x.f - x.f.grad)

    def step(self, x: IO, t: IO, state: core.State):
        if ((self, x), 'y') not in state:
            y = self(x, state, release=False)
        else: y = state[(self, x), 'y']
        self.optim.zero_grad()
        assessment = self.assess_y(y, t.detach())
        assessment.backward()
        self.optim.step()

    def forward(self, x: IO, state: core.State, release: bool=True) -> torch.Tensor:
        x.freshen(False)
        y = state[(self, x), 'y'] = IO(self.linear(x.f)) 
        return y.out(release)


class SimpleAccLearner(core.LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = _assess.ThLoss(nn.MSELoss, reduction='mean')
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t:IO, reduction_override: str = None) -> core.Assessment:
        return self.loss.assess(y, t, reduction_override)
    
    def accumulate(self, x: IO, t: IO, state: core.State) -> IO:
        if ((self, x), 'y') not in state:
            y = self(x, state, release=False)
        else: y = state[(self, x), 'y']
        self.optim.zero_grad()
        assessment = self.assess_y(y, t.detach())
        assessment.backward()

    def step_x(self, x: IO, t: IO, state: core.State) -> IO:
        if ((self, x), 'y') not in state:
            assessment = self.assess(x,  t.detach(), state=state, release=False)
            assessment.backward()
            
        return IO(x.f - x.f.grad)

    def step(self, x: IO, t: IO, state: core.State):
        self.optim.step()

    def forward(self, x: IO, state: core.State, release: bool=True) -> torch.Tensor:
        x.freshen(False)
        y = state[(self, x), 'y'] = IO(self.linear(x.f)) 
        return y.out(release)

# # # # # # # TODO: UPdate simplelearner
# # # # # # # TODO: Update tests
# # # # # # # TODO: write tests for LayeredLearner


class DummyHook(core.LearnerPostHook):

    def __call__(self, x: IO, t: IO, state: State, y: IO, assessment: Assessment) -> typing.Tuple[IO, IO]:
        state[self, 'hi'] = 'hi'


class TestLearningMachineWithSimpleLearner:

    def test_assess_y_uses_correct_reduction(self):

        learner = SimpleLearner(2, 3)
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, 'sum')
        target = nn.MSELoss(reduction='sum')(*y, *t)
        assert result.item() == target.item()

    def test_grad_will_not_be_available_in_trans(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        y = learner(x, core.State(), release=True)
        assert y.f.grad_fn is None

    def test_grad_will_be_available_in_trans_if_not_detaching(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        y = learner(x, core.State(), release=False)
        assert y.f.grad_fn is not None

    def test_step_x_updates_x(self):

        learner = SimpleLearner(2, 3)
        base_x = torch.rand(2, 2)
        x = IO(torch.clone(base_x))
        t = IO(torch.rand(2, 3))
        state = core.State()
        learner(x, state)
        learner.step(x, t, state)
        x_prime = learner.step_x(x, t, state)

        assert (x_prime.f != base_x).any()

    def test_step_updates_parameters(self):

        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = core.State()
        before = utils.get_model_parameters(learner)
        y = learner(x, state)
        learner.step(x, t, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()

    def test_learn_hook_called_after_learning_and_sets_state_to_hi(self):
        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        hook = DummyHook()
        learner.learner_posthook(hook, True, False)
        state = State()
        learner.learn(x, t, state)
        assert state[hook, 'hi'] == 'hi'

    def test_learn_hook_not_called_after_testing_and_state_not_set_to_hi(self):
        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        hook = DummyHook()
        learner.learner_posthook(hook, True, False)
        state = State()
        learner.test(x, t, state)
        assert (hook, 'hi') not in state

    def test_learn_hook_called_after_testing_and_state_set_to_hi(self):
        learner = SimpleLearner(2, 3)
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        hook = DummyHook()
        learner.learner_posthook(hook, True, True)
        state = State()
        learner.test(x, t, state)
        assert state[hook, 'hi'] == 'hi'


class LayeredLearner(core.LearningMachine):

    def __init__(self, m1: SimpleLearner, m2: SimpleLearner):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
    
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> core.Assessment:
        return self.m2.assess_y(
            y, t, reduction_override=reduction_override)
    
    def step_x(self, x: IO, t: IO, state: core.State) -> IO:
        t = state[self, 'x_step_t']
        return self.m1.step_x(x, t, state)
        
    def step(self, x: IO, t: IO, state: core.State):
        self.m2.step(state[self, "y1"], t, state)
        t1 = self.m2.step_x(state[self, "y1"], t, state)
        self.m1.step(x, t1, state)
        state[self, "x_step_t"] = t1

    def forward(self, x: IO, state: core.State, release: bool=True) -> torch.Tensor:
        y1 = state[self, 'y1'] = self.m1(x, state)
        y2 = state[self, 'y2'] = self.m2(y1, state)
        return y2.out(release)


class TestLearningMachineWithComplexLearner:

    def test_assess_y_uses_correct_reduction(self):

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        y = IO(torch.rand(2, 3))
        t = IO(torch.rand(2, 3))
        result = learner.assess_y(y, t, 'sum')
        target = nn.MSELoss(reduction='sum')(*y, *t)
        assert result.item() == target.item()

    def test_excite_detaches_y(self):

        torch.manual_seed(1)
        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        x = IO(torch.rand(2, 2))
        y = learner(x, core.State(), release=True)
        assert y.f.grad_fn is None

    def test_step_x_updates_x(self):
        torch.manual_seed(1)

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))
        x = IO(torch.rand(2, 2))
        x_ = x.clone(True)
        t = IO(torch.rand(2, 3))
        state = core.State()
        learner(x, state)
        learner.step(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x_prime.f != x_.f).any()

    def test_step_updates_parameters(self):
        torch.manual_seed(1)

        learner = LayeredLearner(SimpleLearner(2, 3), SimpleLearner(3, 3))        
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = core.State()
        before = utils.get_model_parameters(learner)
        y = learner.forward(x, state)
        learner.step(x, t, state)
        after = utils.get_model_parameters(learner)
        assert (before != after).any()


class DependentLearner(core.LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = _assess.ThLoss(nn.MSELoss, reduction='mean')
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-1)

    def assess_y(self, y: IO, t:IO, reduction_override: str = None) -> core.Assessment:
        return self.loss.assess(y, t, reduction_override)
    
    @core.step_dep('stepped', exec=False)
    def step_x(self, x: IO, t: IO, state: core.State) -> IO:
        if ((self, x), 'y') not in state:
            assessment = self.assess(x,  t.detach(), state=state, release=False)
            assessment.backward()
            
        return IO(x.f - x.f.grad)

    @core.forward_dep('y', exec=True)
    def step(self, x: IO, t: IO, state: core.State):
        y = state[(self, x), 'y']
        self.optim.zero_grad()
        assessment = self.assess_y(y, t.detach())
        assessment.backward()
        self.optim.step()
        state[(self, x), 'stepped'] = True

    def forward(self, x: IO, state: core.State, release: bool=True) -> torch.Tensor:
        x.freshen(False)
        y = state[(self, x), 'y'] = IO(self.linear(x.f)) 
        return y.out(release)


class TestDependencies:

    def test_step_executes_forward(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = core.State()
        dependent = DependentLearner(2, 3)
        dependent.step(x, t, state)
        assert state[(dependent, x), 'y'] is not None

    def test_step_does_not_execute_forward_if_already_called(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = core.State()
        dependent = DependentLearner(2, 3)
        dependent(x, state)
        prev = state[(dependent, x), 'y']
        dependent.step(x, t, state)
        assert state[(dependent, x), 'y'] is prev

    def test_step_x_if_not_stepped(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = core.State()
        dependent = DependentLearner(2, 3)
        with pytest.raises(RuntimeError):
            dependent.step_x(x, t, state)

    def test_step_x_executes_if_stepped(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        state = core.State()
        dependent = DependentLearner(2, 3)
        dependent.step(x, t, state)
        x_prime = dependent.step_x(x, t, state)
        assert x_prime is not None
