import torch

from zenkai import IO
from zenkai.kaku._backtarget import BackTarget


class TestBackTarget:
    def test_back_target_reverses_view(self):

        x = IO(torch.rand(2, 4, 2))
        t = IO(torch.rand(2, 8))
        view = BackTarget(lambda x: x.view(2, 8))
        view(x)
        x_prime = view.step_x(x, t)
        assert (x_prime.f == t.f.view(2, 4, 2)).all()

    def test_back_target_reverses_index(self):

        x = IO(torch.rand(2, 4, 2))
        t = IO(torch.rand(2, 4))
        view = BackTarget(lambda x: x[:, :, 0])
        view(x)
        x_prime = view.step_x(x, t)
        assert (x_prime.f[:, :, 0] == t.f).all()
