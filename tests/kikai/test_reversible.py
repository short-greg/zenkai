import torch

from zenkai import IO, Criterion
from zenkai.kikai._reversible import ReversibleMachine, reverse
from zenkai.kikai._backtarget import BackTarget
from zenkai.mod import _reversible


class TestReversibleMachine:
    def test_step_x_reverses(self):

        machine = ReversibleMachine(_reversible.SignedToBool(), Criterion("MSELoss"))
        x = IO(torch.randn(4, 3).sign())
        t = IO((x.f + 1) / 2)
        assert (machine.step_x(x, t).f == x.f).all()

    def test_step_x_results_in_valid_values(self):

        machine = ReversibleMachine(_reversible.SignedToBool(), Criterion("MSELoss"))
        x = IO(torch.randn(4, 3).sign())
        t = IO((x.f + 1) / 2)
        reversed = machine.step_x(x, t).f
        assert ((reversed == -1) | (reversed == 1)).all()

    def test_forward_converts_to_correct_value(self):

        machine = ReversibleMachine(_reversible.SignedToBool(), Criterion("MSELoss"))
        x = IO(torch.randn(4, 3).sign())
        t = (x.f + 1) / 2
        y = machine(x).f
        assert (y == t).all()


class TestReverse:
    def test_reverse_produces_backtarget(self):

        machine = reverse(lambda x: x.view(2, 4))
        IO(torch.randn(8))
        assert isinstance(machine, BackTarget)

    def test_backtarget_reverses_input(self):

        machine = reverse(lambda x: x.view(2, 4))
        x = IO(torch.randn(8))
        y = machine(x)
        x_prime = machine.step_x(x, y)
        assert (x_prime.f == x.f).all()

    def test_reverse_produces_reversible(self):

        machine = reverse(_reversible.SignedToBool())
        assert isinstance(machine, ReversibleMachine)
