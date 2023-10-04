# 3rd party
import typing
import pytest
import torch
import torch.optim
from torch import nn

from zenkai import utils

# local
from zenkai.kaku import IO, IDable, Assessment, State
from zenkai.kaku import machine as core


class Base:

    def __init__(self) -> None:
        pass


class X(Base, core.IDable):

    @property
    def id(self):
        return str(id(self))


class TestState:

    def test_store_stores_data(self):
        
        x = X()
        state = core.State()
        state.store(x, 'value', 1)
        assert state.get(x, 'value') == 1

    def test_getitem_raises_error_if_invalid_obj(self):
        
        x = X()
        state = core.State()
        with pytest.raises(KeyError):
            state[x, 1] == 2

    def test_getitem_raises_error_if_invalid_key(self):
        
        x = X()
        state = core.State()
        state.store(x, 2, 2)
        with pytest.raises(KeyError):
            state[x, 1] == 2

    def test_add_sub_adds_substate(self):
        
        x = X()
        state = core.State()
        sub1 = state.add_sub(x, 'sub')
        sub2 = state.sub(x, 'sub', to_add=False)
        assert sub1 is sub2

    def test_data_is_in_state_after_setting(self):

        io = IO()
        state = core.State()
        state[(self, io), 'x'] = 2
        assert ((self, io), 'x') in state
        
    def test_data_is_in_state_after_storing_and_keeping(self):

        io = IO()
        state = core.State()
        state.store((self, io), 'x', 2, keep=True)
        assert ((self, io), 'x') in state

    def test_data_is_in_state_after_storing_and_spawning(self):

        io = IO()
        state = core.State()
        state.store((self, io), 'x', 2, keep=True)
        state2 = state.spawn()
        assert ((self, io), 'x') in state2

    def test_data_is_in_state_after_storing_with_idx_and_spawning(self):

        io = IO()
        state = core.State()
        state[(self, io), 'x'] = 2
        state.keep((self, io), 'x', True)
        state2 = state.spawn()
        assert ((self, io), 'x') in state2

    def test_data_is_not_in_state_after_storing_with_idx_and_spawning(self):

        io = IO()
        state = core.State()
        state[(self, io), 'x'] = 2
        state.keep((self, io), 'x', False)
        state2 = state.spawn()
        assert ((self, io), 'x') not in state2

    def test_assessment_log_logs_assessment(self):

        obj = 'x'
        state = core.State()
        assessment = Assessment(torch.tensor(1.0))
        state.log_assessment(obj, 'x', 'k', assessment)
        result = state.logs.as_assessment_dict()
        assert (result['x_k'].value == assessment.value).all()


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


class TestIO:
    
    def test_freshen_inplace_does_not_change_the_tensor(self):
        x = torch.rand(2, 2)
        io = IO(x)
        io.freshen(True)
        assert x is io.f

    def test_freshen_not_inplace_changes_the_tensor(self):
        x = torch.rand(2, 2)
        io = IO(x)
        io.freshen()
        assert x is not io.f

    def test_freshen_sets_required_grad_to_true(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        io.freshen(True)
        assert x.requires_grad is True
        assert x2.requires_grad is True

    def test_items_returns_a_dictionary_of_all_items(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        items = io.items()
        assert items[0] is x
        assert items[1] is x2

    def test_vals_returns_all_tensors(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        _x1, _x2 = io.totuple()
        assert x is _x1
        assert x2 is _x2

    def test_getitem_returns_the_correct_tensor(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        assert io.f is x
        
    def test_iter_iterates_over_all_elements(self):
        x = torch.rand(2, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        elements = []
        for element in io:
            elements.append(element)
        assert x is elements[0]
        assert x2 is elements[1]
    
    def test_clone_clones_all_tensors(self):
        x = torch.rand(4, 2)
        x2 = torch.rand(2, 2)
        io = IO(x, x2)
        _x, _x2 = io.clone()
        assert (x == _x).all()
        assert (x2 == _x2).all()
        
    def test_is_empty_is_true_when_no_elements(self):
        io = IO()
        assert io.is_empty()

    def test_is_empty_is_false_when_elements(self):
        x = torch.rand(4, 2)
        io = IO(x)
        assert not io.is_empty()

    def test_the_values_of_two_trans_are_the_same_if_tensor_is_the_same(self):
        val = torch.rand(3, 2)
        x = IO(val)
        y = IO(val)
        assert x.f is y.f

    def test_the_values_of_two_trans_are_not_the_same_if_has_been_detached(self):
        val = torch.rand(3, 2)
        x = IO(val).detach()
        y = IO(val)
        assert not x.f is y.f


class SimpleLearner(core.LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = core.ThLoss(nn.MSELoss, reduction='mean')
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


class TestMyState:

    def test_my_state_sets_value_in_state(self):

        x = X()
        state = core.State()
        my_state = state.mine(x)
        my_state.x = 2
        assert state.get(x, 'x') is my_state.x

    def test_my_state_gets_correct_value_in_state(self):

        x = X()
        state = core.State()
        state[x, 'x'] = 2
        my_state = state.mine(x)
        assert my_state.x is state[x, 'x']

    def test_my_state_gets_correct_sub_state(self):
        x = X()
        
        state = core.State()
        state.add_sub(x, "sub")
        mine = state.mine(x)
        assert mine.subs['sub'] is state.sub(x, 'sub')


class DependentLearner(core.LearningMachine):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.loss = core.ThLoss(nn.MSELoss, reduction='mean')
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
