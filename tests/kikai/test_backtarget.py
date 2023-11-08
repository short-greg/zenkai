import torch
import torch.nn as nn

from zenkai import IO, State
from zenkai.kikai._reversible import BackTarget
import torch
import torch.nn as nn
from zenkai.kaku import IO, State, Criterion


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
