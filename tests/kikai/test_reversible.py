import torch
import torch.nn as nn

from zenkai import IO, State, Criterion
from zenkai.kikai import ReversibleMachine
from ..kaku.test_machine import SimpleLearner
from zenkai.utils import get_model_parameters
from zenkai.mod import reversible
import torch
import torch.nn as nn
from zenkai.kaku import IO, State, Assessment, LearningMachine, Criterion, ThLoss

from zenkai.mod.reversible import BackTarget
from zenkai.utils import Lambda



class TestReversibleMachine:

    def test_step_x_reverses(self):

        reversible = ReversibleMachine(
            reversible.Neg1ToZero(), Criterion('mse')
        )
        x = IO(torch.randn(4, 3).sign())
        t = IO((x.f + 1) / 2)
        assert (reversible.step_x(x, t, State()).f == x.f).all()

    def test_step_x_results_in_valid_values(self):

        reversible = ReversibleMachine(
            reversible.Neg1ToZero(), Criterion('mse')
        )
        x = IO(torch.randn(4, 3).sign())
        t = IO((x.f + 1) / 2)
        reversed = reversible.step_x(x, t, State()).f
        assert ((reversed == -1) | (reversed == 1)).all()

    def test_forward_converts_to_correct_value(self):

        reversible = ReversibleMachine(
            reversible.Neg1ToZero(), Criterion('mse')
        )
        x = IO(torch.randn(4, 3).sign())
        t = (x.f + 1) / 2
        y = reversible(x, State()).f
        assert (y == t).all()


class TestBackTarget:

    def test_back_target_reverses_view(self):

        x = IO(torch.rand(2, 4, 2))
        t = IO(torch.rand(2, 8))
        state = State()
        view = BackTarget(lambda x: x.view(2, 8))
        y = view(x, state)
        x_prime = view.step_x(x, t, state)
        assert (x_prime.f == t.f.view(2, 4, 2)).all()

    def test_back_target_reverses_index(self):

        x = IO(torch.rand(2, 4, 2))
        t = IO(torch.rand(2, 4))
        state = State()
        view = BackTarget(lambda x: x[:,:,0])
        y = view(x, state)
        x_prime = view.step_x(x, t, state)
        assert (x_prime.f[:, :, 0] == t.f).all()
