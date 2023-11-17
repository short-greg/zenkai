# 3rd party
import pytest
import torch
import torch.optim
from torch import nn

# local
from zenkai.kaku import IO, Assessment
from zenkai.kaku import _state as state2
from zenkai.kaku._state import AssessmentLog


class Base:
    def __init__(self) -> None:
        pass


class X(state2.IDable, Base):
    pass


class Y(state2.IDable, nn.Linear):

    pass


class TestIDable:
    def test_x_sets_id_to_object_id(self):
        x = X()
        assert x.id is not None

    def test_x_sets_id_to_linear_module(self):
        x = Y(2, 3)
        assert x.id is not None

    def test_x_state_dict_retrieves_id(self):
        x = Y(2, 3)
        assert x.state_dict()["id"] == x.id

    def test_x_load_state_dict_sets_id_to_original(self):
        x = Y(2, 3)
        x2 = Y(2, 3)
        state_dict = x.state_dict()
        x2.load_state_dict(state_dict)
        assert x2.id == x.id


class TestAssessmentLog:
    def test_assessment_log_update_adds(self):
        log = AssessmentLog()
        assessment = Assessment(torch.rand(1)[0], True)
        log.update("x", "name", "validation", assessment)
        assert log.dict["x"][None]["name"]["validation"].value == assessment.value

    def test_assessment_log_as_assessment_dict_gets_assessment(self):
        log = AssessmentLog()
        assessment = Assessment(torch.rand(1)[0], True)
        log.update("x", "name", "validation", assessment)
        assert log.as_assessment_dict()["name_validation"].value == assessment.value

    def test_update_overwrites_initial_assessment(self):
        log = AssessmentLog()
        assessment = Assessment(torch.rand(1)[0], True)
        assessment2 = Assessment(torch.rand(1)[0], True)
        log.update("x", "name", "validation", assessment)
        log.update("x", "name", "validation", assessment2)
        assert log.as_assessment_dict()["name_validation"].value == assessment2.value

    def test_update_overwrites_initial_assessment_even_when_keys_are_different(self):
        log = AssessmentLog()
        assessment = Assessment(torch.rand(1)[0], True)
        assessment2 = Assessment(torch.rand(1)[0], True)
        log.update("x", "name", "validation", assessment)
        log.update("y", "name", "validation", assessment2)
        assert log.as_assessment_dict()["name_validation"].value == assessment2.value


class TestState:
    def test_store_stores_data(self):

        x = X()
        state = state2.State()
        state.set((x, "value"), 1)
        assert state.get((x, "value")) == 1

    def test_getitem_raises_error_if_invalid_obj(self):

        x = X()
        state = state2.State()
        with pytest.raises(KeyError):
            state[x, 1] == 2

    def test_getitem_raises_error_if_invalid_key(self):

        x = X()
        state = state2.State()
        state.set((x, 2), 2)
        with pytest.raises(KeyError):
            state[x, 1] == 2

    def test_add_sub_adds_substate(self):

        x = X()
        state = state2.State()
        sub1 = state.add_sub((x, "sub"))
        sub2 = state.sub((x, "sub"), to_add=False)
        assert sub1 is sub2

    def test_data_is_in_state_after_setting(self):

        io = IO()
        state = state2.State()
        state[self, io, "x"] = 2
        assert (self, io, "x") in state

    def test_data_is_in_state_after_storing_and_keeping(self):

        io = IO()
        state = state2.State()
        state.set((self, io, "x"), 2, to_keep=True)
        assert (self, io, "x") in state

    def test_data_is_in_state_after_storing_and_spawning(self):

        io = IO()
        state = state2.State()
        state.set((self, io, "x"), 2, to_keep=True)
        state_ = state.spawn()
        assert (self, io, "x") in state_

    def test_data_is_in_state_after_storing_with_idx_and_spawning(self):

        io = IO()
        state = state2.State()
        state[self, io, "x"] = 2
        state.keep((self, io, "x"), True)
        state_ = state.spawn()
        assert (self, io, "x") in state_

    def test_data_is_not_in_state_after_storing_with_idx_and_spawning(self):

        io = IO()
        state = state2.State()
        state[self, io, "x"] = 2
        state.keep((self, io, "x"), False)
        state_ = state.spawn()
        assert (self, io, "x") not in state_

    def test_assessment_log_logs_assessment(self):

        obj = "x"
        state = state2.State()
        assessment = Assessment(torch.tensor(1.0))
        state.log_assessment(obj, "x", "k", assessment)
        result = state.logs.as_assessment_dict()
        assert (result["x_k"].value == assessment.value).all()

    def test_get_or_set_gets_value_if_already_set(self):

        obj = "x"
        state = state2.State()
        state[obj, "y"] = 2
        assert state.get_or_set((obj, "y"), 3) == 2

    def test_get_or_set_gets_value_if_already_set_when_three(self):

        obj = "x"
        state = state2.State()
        t = IO()
        state[obj, t, "y"] = 2
        assert state.get_or_set((obj, t, "y"), 3) == 2

    def test_get_or_set_sets_value_when_not_set(self):

        obj = "x"
        state = state2.State()
        state.get_or_set((obj, "y"), 3)
        assert state.get((obj, "y")) == 3

    def test_value_in_state_after_get_and_set(self):

        obj = "x"
        state = state2.State()
        state.get_or_set((obj, "y"), 3)
        assert (obj, "y") in state

    def test_sub_iter_loops_over_all_subs(self):

        obj = "x"
        state = state2.State()
        sub1 = state.sub((obj, "sub"))
        sub2 = state.sub((obj, "sub2"))
        subs = set(sub for _, sub in state.sub_iter(obj))
        assert sub1 in subs and sub2 in subs

    def test_log_assessment_updates_the_logs(self):

        obj = "x"
        state = state2.State()
        state.log_assessment(
            obj, "name", "validation", Assessment(torch.tensor(2), True)
        )
        assert state.logs.as_assessment_dict()["name_validation"].value == 2

    def test_spawn_keeps_values_in_keep(self):

        obj = "x"
        state = state2.State()
        state.set((obj, "z"), 2, True)
        state_ = state.spawn()
        assert state_[obj, "z"] == 2

    def test_spawn_keeps_the_sub(self):

        obj = "x"
        state = state2.State()
        sub = state.sub((obj, "sub2"))
        sub.set((obj, "z"), 2, True)
        state_ = state.spawn()
        sub2 = state_.sub((obj, "sub2"), False)
        assert sub2[obj, "z"] == 2


class TestMyState:

    def test_my_state_sets_value_in_state(self):

        x = X()
        state = state2.State()
        my_state = state.mine(x)
        my_state.x = 2
        assert state.get((x, "x")) is my_state.x

    def test_my_state_gets_correct_value_in_state(self):

        x = X()
        state = state2.State()
        state[x, "x"] = 2
        my_state = state.mine(x)
        assert my_state.x is state[x, "x"]

    def test_my_state_adds_the_sub(self):

        x = X()
        state = state2.State()
        state[x, "x"] = 2
        my_state = state.mine(x)
        my_state.my_sub("x")
        assert state.sub((x, "x")) is not None

    def test_my_state_set_sets_the_value(self):

        x = X()
        state = state2.State()
        state[x, "x"] = 2
        my_state = state.mine(x)
        my_state.set("x", 2)
        assert state[x, "x"] == 2

    def test_my_state_gets_correct_sub_state(self):
        x = X()
        state = state2.State()
        state.add_sub((x, "sub"))
        mine = state.mine(x)
        assert mine.subs["sub"] is state.sub((x, "sub"))
