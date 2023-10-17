import torch
import torch.nn as nn

from zenkai import IO, State, Objective
from zenkai.kikai import ReversibleMachine
from ..kaku.test_machine import SimpleLearner
from zenkai.utils import get_model_parameters
from zenkai.utils import reversibles


class TestReversibleMachine:

    def test_step_x_reverses(self):

        reversible = ReversibleMachine(
            reversibles.Neg1ToZero(), Objective('mse')
        )
        x = IO(torch.randn(4, 3).sign())
        t = IO((x.f + 1) / 2)
        assert (reversible.step_x(x, t, State()).f == x.f).all()

    def test_step_x_results_in_valid_values(self):

        reversible = ReversibleMachine(
            reversibles.Neg1ToZero(), Objective('mse')
        )
        x = IO(torch.randn(4, 3).sign())
        t = IO((x.f + 1) / 2)
        reversed = reversible.step_x(x, t, State()).f
        assert ((reversed == -1) | (reversed == 1)).all()

    def test_forward_converts_to_correct_value(self):

        reversible = ReversibleMachine(
            reversibles.Neg1ToZero(), Objective('mse')
        )
        x = IO(torch.randn(4, 3).sign())
        t = (x.f + 1) / 2
        y = reversible(x, State()).f
        assert (y == t).all()
