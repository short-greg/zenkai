# 3rd party
import pytest
import torch
import torch.optim
from torch import nn

# local
from zenkai.kaku import _state as state2


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
