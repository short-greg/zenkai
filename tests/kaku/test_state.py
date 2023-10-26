# 3rd party
import pytest
import torch
import torch.optim

# local
from zenkai.kaku import IO, Assessment
from zenkai.kaku import _machine as core


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
