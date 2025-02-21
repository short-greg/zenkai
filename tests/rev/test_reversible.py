import torch

from zenkai import Criterion, State, iou
from zenkai.rev._reversible import ReversibleMachine
# from zenkai.kaku._backtarget import BackTarget
from zenkai.rev import _reversible_mods


class TestReversibleMachine:

    def test_step_x_reverses(self):

        machine = ReversibleMachine(
            _reversible_mods.SignedToBool(), Criterion("MSELoss")
        )
        state = State()
        x = iou(torch.randn(4, 3).sign())
        t = iou((x.f + 1) / 2)
        assert (machine.step_x(x, t, state).f == x.f).all()

    def test_step_x_results_in_valid_values(self):

        machine = ReversibleMachine(
            _reversible_mods.SignedToBool(), 
            Criterion("MSELoss")
        )
        state = State()
        x = iou(torch.randn(4, 3).sign())
        t = iou((x.f + 1) / 2)
        reversed = machine.step_x(x, t, state).f
        assert ((reversed == -1) | (reversed == 1)).all()

    def test_forward_converts_to_correct_value(self):

        machine = ReversibleMachine(_reversible_mods.SignedToBool(), Criterion("MSELoss"))
        x = iou(torch.randn(4, 3).sign())
        state = State()
        t = (x.f + 1) / 2
        y = machine.forward_io(x, state).f
        assert (y == t).all()


# class TestReverse:
#     # def test_reverse_produces_backtarget(self):

#     #     machine = reverse(lambda x: x.view(2, 4))
#     #     IO(torch.randn(8))
#     #     assert isinstance(machine, BackTarget)

#     def test_backtarget_reverses_input(self):

#         machine = reverse(lambda x: x.view(2, 4))
#         x = IO(torch.randn(8))
#         y = machine(x)
#         x_prime = machine.step_x(x, y)
#         assert (x_prime.f == x.f).all()

#     def test_reverse_produces_reversible(self):

#         machine = reverse(_reversible_mods.SignedToBool())
#         assert isinstance(machine, ReversibleMachine)
